"""``RLSearchAgent``: decision-time ISMCTS over a MaskablePPO checkpoint.

The policy is the move prior (a composed action's prior is the product of its
atom probabilities along its ``atom_sequence`` path) and the critic is the leaf
evaluator (V+ = V + own public VP / 10, per experiment 009). No retraining:
the same checkpoint as ``RLAgent``, which remains both the greedy fallback and
the no-search arm of experiment 011. The search itself is torch-free and lives
in ``agents/ismcts.py``; torch and sb3 are imported lazily here.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from agents.ismcts import (
    ActionKey,
    OpponentModel,
    SearchConfig,
    action_key,
    keyed_actions,
    search,
)
from agents.rl_agent import RLAgent, _load_model, lean_predict_batch
from engine.actions import Action, RejectTrade
from engine.game import victory_points
from engine.state import GameState, Phase
from rl.action_space import N_ATOMS, ActionComposer, atom_sequence, is_trade_proposal
from rl.encoder import ObservationEncoder

VP_SCALE = 10.0  # ShapedReward's phi = public VP / 10


class PolicyEvaluator:
    """Priors and V+ from one fused, batched forward pass per state."""

    def __init__(self, checkpoint: str, num_players: int) -> None:
        self.policy = _load_model(checkpoint, "cpu").policy
        self.encoder = ObservationEncoder(num_players)
        self._board: Any = None

    def ensure_board(self, state: GameState) -> None:
        """Board statics are per game; counterfactual copies share them."""
        if state.board is not self._board:
            self.encoder.reset(state)
            self._board = state.board

    def evaluate(
        self, state: GameState, seat: int, legal: Any
    ) -> tuple[dict[ActionKey, float], float]:
        import torch

        from rl.bc import _forward

        n = len(state.players)
        seqs: list[tuple[ActionKey, tuple[int, ...]]] = []
        nexts: dict[tuple[int, ...], set[int]] = {}
        for a in legal:
            if is_trade_proposal(a):
                continue
            seq = atom_sequence(a, seat, n)
            seqs.append((action_key(a, seat, n), seq))
            for d in range(len(seq)):
                nexts.setdefault(seq[:d], set()).add(seq[d])
        prefixes = list(nexts)  # () is first: every sequence has length >= 1
        obs = np.stack(
            [self.encoder.encode(state, seat, buffer_prefix=p) for p in prefixes]
        )
        masks = np.zeros((len(prefixes), N_ATOMS), dtype=bool)
        for i, p in enumerate(prefixes):
            masks[i, list(nexts[p])] = True
        with torch.no_grad():
            logits, values = _forward(
                self.policy,
                torch.as_tensor(obs, dtype=torch.float32),
                torch.as_tensor(masks),
            )
            probs = torch.softmax(logits, dim=-1).numpy()
        row = {p: i for i, p in enumerate(prefixes)}
        priors: dict[ActionKey, float] = defaultdict(float)
        for key, seq in seqs:
            prob = 1.0
            for d, atom in enumerate(seq):
                prob *= float(probs[row[seq[:d]], atom])
            priors[key] += prob
        total = sum(priors.values()) or 1.0
        value = float(values[0]) + victory_points(state, seat) / VP_SCALE
        return {k: v / total for k, v in priors.items()}, value


class SelfModelOpponent:
    """Opponents play the checkpoint's own greedy atom-wise policy."""

    name = "self_model"

    def __init__(self, evaluator: PolicyEvaluator) -> None:
        self._ev = evaluator

    def reset(self) -> None:
        pass

    def choose_action(
        self, state: GameState, legal_actions: list[Action], player_idx: int
    ) -> Action:
        if state.phase is Phase.AWAIT_TRADE_RESPONSE:
            return next(a for a in legal_actions if isinstance(a, RejectTrade))
        n = len(state.players)
        composer = ActionComposer()
        while True:
            mask = np.array(composer.mask(legal_actions, player_idx, n), dtype=bool)
            obs = self._ev.encoder.encode(
                state, player_idx, buffer_prefix=composer.prefix
            )
            atom = int(
                lean_predict_batch(self._ev.policy, obs[None], mask[None], "cpu")[0]
            )
            emitted = composer.push(atom, legal_actions, player_idx, n)
            if emitted is not None:
                return emitted


@dataclass(slots=True)
class DecisionStats:
    """One record per top-level decision of the search seat."""

    phase: str
    ms: float  # search wall time only (the greedy comparison is excluded)
    simulations: int  # 0 when search was skipped
    overrode: bool  # search chose a different action than greedy RLAgent
    edge_stats: tuple[tuple[ActionKey, int, float, float], ...] = ()


@dataclass(slots=True)
class SearchLog:
    decisions: list[DecisionStats] = field(default_factory=list)


class RLSearchAgent:
    """ISMCTS at decision time; falls back to ``RLAgent`` when there is
    nothing to search (budget 0, a forced move, a trade response)."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        config: SearchConfig | None = None,
        name: str = "rl_search",
        rng: random.Random | None = None,
    ) -> None:
        self.name = name
        self.config = config or SearchConfig()
        self.checkpoint = str(checkpoint)
        self.rng = rng or random.Random()
        self.log = SearchLog()
        self._greedy = RLAgent(self.checkpoint, name=name, rng=self.rng)
        self._evaluator: PolicyEvaluator | None = None
        self._opponent: OpponentModel | None = None

    def reset(self) -> None:
        self._greedy.reset()

    def _build(self, num_players: int) -> tuple[PolicyEvaluator, OpponentModel]:
        if (
            self._evaluator is None
            or self._evaluator.encoder.num_players != num_players
        ):
            self._evaluator = PolicyEvaluator(self.checkpoint, num_players)
            if self.config.opponent == "heuristic":
                from agents.heuristic import HeuristicAgent

                self._opponent = HeuristicAgent(name="model")
            elif self.config.opponent == "self":
                self._opponent = SelfModelOpponent(self._evaluator)
            else:
                raise ValueError(f"unknown opponent model {self.config.opponent!r}")
        assert self._opponent is not None
        return self._evaluator, self._opponent

    def choose_action(
        self, state: GameState, legal_actions: list[Action], player_idx: int
    ) -> Action:
        if state.phase is Phase.AWAIT_TRADE_RESPONSE or self.config.simulations <= 0:
            return self._greedy.choose_action(state, legal_actions, player_idx)
        num_players = len(state.players)
        keyed = keyed_actions(legal_actions, player_idx, num_players)
        if len(keyed) == 1:
            self.log.decisions.append(
                DecisionStats(state.phase.name, 0.0, 0, overrode=False)
            )
            return next(iter(keyed.values()))
        evaluator, opponent = self._build(num_players)
        evaluator.ensure_board(state)
        t0 = time.perf_counter()
        result = search(
            state,
            player_idx,
            legal_actions,
            config=self.config,
            evaluator=evaluator,
            opponent=opponent,
            rng=self.rng,
        )
        ms = (time.perf_counter() - t0) * 1000.0
        greedy = self._greedy.choose_action(state, legal_actions, player_idx)
        self.log.decisions.append(
            DecisionStats(
                state.phase.name,
                ms,
                self.config.simulations,
                overrode=action_key(greedy, player_idx, num_players) != result.key,
                edge_stats=result.edge_stats,
            )
        )
        return result.action
