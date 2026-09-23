"""``CatanEnv``: a single-agent gymnasium env over the real engine.

Deliberately *not* a subclass of ``gamekit.rl.env.SingleAgentEnv``. That class
serves the flat masked-``Discrete`` case -- one index in, one mask out -- and
its own docstring names catan as the game it does not serve. What catan reuses
instead is ``gamekit.rl.protocols``/``driver``/``selfplay``, which are
head-agnostic, plus ``gamekit.rl.masking.coerce_to_legal``.

The contract below mirrors ``SingleAgentEnv``'s where there is no reason to
differ -- reset draw order, ``on_illegal="coerce"``, zero observation on a
terminal step, ``action_masks()`` named for sb3-contrib's ``ActionMasker``
convention -- so the two envs stay comparable. What it adds is the composition
buffer (see ``rl.action_space``), which lets one ``Discrete`` head express
discard multisets and ordered road-building pairs exactly.

**Sub-steps are not environment transitions.** Filling in a partial action
applies nothing to the engine, so those steps return reward 0.0 and do not
advance the step budget. Only an emitted engine action is a real transition.
That keeps ``ShapedReward``'s potential term telescoping over genuine state
changes rather than accumulating ``(gamma - 1) * phi`` per keystroke.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gamekit.agent import Agent
from gamekit.rl.driver import advance_until_learner
from gamekit.rl.masking import coerce_to_legal
from gamekit.rl.protocols import RewardFn
from gamekit.rl.selfplay import OpponentPool
from gymnasium import spaces
from numpy.typing import NDArray

from engine.actions import Action
from engine.game import MAX_PLAYERS, MIN_PLAYERS
from engine.state import GameState
from rl.action_space import N_ATOMS, ActionComposer
from rl.adapter import CatanTurnBasedGame
from rl.encoder import OBS_DIM, ObservationEncoder
from rl.reward import ShapedReward

# Matches experiments/rollout.py's STEP_BUDGET_DEFAULT: the budget belongs to
# the driver, never to is_terminal.
STEP_BUDGET_DEFAULT = 20_000

# A deal whose game ends before the learner ever acts is re-dealt. Bounded so a
# misconfigured opponent set fails loudly instead of spinning forever --
# SingleAgentEnv's equivalent loop is unbounded.
MAX_RESET_REDEALS = 100


class CatanEnv(gym.Env[NDArray[np.float32], np.int64]):
    """One ``step()`` is one learner decision: an atom, not always an action."""

    metadata: dict[str, list[str]] = {"render_modes": []}

    def __init__(
        self,
        *,
        num_players: int = 4,
        agents: Sequence[Agent[GameState, Action]] | None = None,
        opponent_pool: OpponentPool[GameState, Action] | None = None,
        reward: RewardFn[GameState] | None = None,
        randomize_seat: bool = True,
        on_illegal: Literal["coerce", "raise"] = "coerce",
        step_budget: int | None = STEP_BUDGET_DEFAULT,
        allow_trade_proposals: bool = False,
        seed: int | None = None,
    ) -> None:
        if not (MIN_PLAYERS <= num_players <= MAX_PLAYERS):
            raise ValueError("base-game Catan supports 3 or 4 players")
        if (agents is None) == (opponent_pool is None):
            raise ValueError("pass exactly one of `agents` or `opponent_pool`")
        if agents is not None and len(agents) != num_players - 1:
            raise ValueError(
                f"`agents` must have num_players - 1 = {num_players - 1} entries "
                f"(one per non-learner seat), got {len(agents)}"
            )
        if on_illegal not in ("coerce", "raise"):
            raise ValueError(
                f"on_illegal must be 'coerce' or 'raise', got {on_illegal!r}"
            )

        self.num_players = num_players
        self._game = CatanTurnBasedGame(num_players=num_players)
        self._fixed_agents = agents
        self._opponent_pool = opponent_pool
        self._reward: RewardFn[GameState] = (
            reward if reward is not None else ShapedReward()
        )
        self._randomize_seat = randomize_seat
        self._on_illegal = on_illegal
        self._step_budget = step_budget
        self._rng = random.Random(seed)
        self._encoder = ObservationEncoder(num_players)
        self._composer = ActionComposer(allow_trade_proposals=allow_trade_proposals)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ATOMS)

        self._state: GameState | None = None
        self._learner_seat = 0
        self._seat_agents: list[Agent[GameState, Action] | None] = []
        self._episode_steps = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset, following ``SingleAgentEnv``'s draw order exactly.

        reseed -> pick the learner seat -> resample or place opponents ->
        ``agent.reset()`` each -> deal -> ``on_episode_start`` -> advance
        opponents, re-dealing if the game ends before the learner ever acts.
        Every draw comes from one stream, so the order is part of the contract.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)

        self._learner_seat = (
            self._rng.randint(0, self.num_players - 1) if self._randomize_seat else 0
        )

        if self._opponent_pool is not None:
            seat_agents = self._opponent_pool.seat_agents(
                self.num_players, self._learner_seat
            )
        else:
            assert self._fixed_agents is not None  # enforced in __init__
            seat_agents = _place_fixed_agents(
                self._fixed_agents, self.num_players, self._learner_seat
            )
        for agent in seat_agents:
            if agent is not None:
                agent.reset()
        self._seat_agents = seat_agents
        self._composer.reset()

        for _ in range(MAX_RESET_REDEALS):
            state = self._game.reset(seed=self._rng.randint(0, 2**31 - 1))
            self._encoder.reset(state)
            self._reward.on_episode_start(state, self._learner_seat)
            advance_until_learner(self._game, state, seat_agents)
            if not self._game.is_terminal(state):
                break
        else:
            raise RuntimeError(
                f"every one of {MAX_RESET_REDEALS} deals ended before the learner "
                f"acted -- check the opponent configuration"
            )

        self._state = state
        self._episode_steps = 0
        return self._observe(state), {}

    def step(
        self, action: np.int64
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("call reset() before step()")
        if self._game.is_terminal(self._state):
            raise RuntimeError("step() called on an already-terminal episode")

        state = self._state
        legal = self._game.legal_actions(state)
        mask = self._composer.mask(legal, self._learner_seat, self.num_players)
        index = int(action)
        if not (0 <= index < N_ATOMS and mask[index]):
            if self._on_illegal == "raise":
                raise ValueError(f"illegal atom {index}")
            index = coerce_to_legal(index, mask, self._rng)

        emitted = self._composer.push(
            index, legal, self._learner_seat, self.num_players
        )
        if emitted is None:
            # Mid-composition: nothing reached the engine, so this is not a
            # state transition. No reward, no budget consumed.
            return self._observe(state), 0.0, False, False, {"state": state}

        self._game.apply_action(state, emitted)
        self._episode_steps += 1
        self._episode_steps += advance_until_learner(
            self._game, state, self._seat_agents
        )

        terminated = self._game.is_terminal(state)
        truncated = (
            not terminated
            and self._step_budget is not None
            and self._episode_steps >= self._step_budget
        )
        reward = self._reward.compute(state, self._learner_seat, terminated)

        if terminated or truncated:
            obs = np.zeros(OBS_DIM, dtype=np.float32)
        else:
            obs = self._observe(state)
        return obs, reward, terminated, truncated, {"state": state}

    def action_masks(self) -> NDArray[np.bool_]:
        """Boolean mask over atoms.

        Named for sb3-contrib's ``ActionMasker`` convention; this module never
        imports sb3-contrib, exactly as ``gamekit.rl.env`` does not.
        """
        if self._state is None or self._game.is_terminal(self._state):
            return np.zeros(N_ATOMS, dtype=np.bool_)
        legal = self._game.legal_actions(self._state)
        mask = self._composer.mask(legal, self._learner_seat, self.num_players)
        return np.array(mask, dtype=np.bool_)

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Introspection, for tests and for wrapping the policy as an Agent
    # ------------------------------------------------------------------

    @property
    def learner_seat(self) -> int:
        return self._learner_seat

    @property
    def state(self) -> GameState | None:
        return self._state

    def legal_actions(self) -> list[Action]:
        """The engine's legal actions at the current decision point.

        Exposed so callers that need the *engine* action (not the atom head)
        -- BC dataset generation replaying ``HeuristicAgent`` through this env
        is the motivating case -- never reach into ``self._game``.
        """
        if self._state is None:
            raise RuntimeError("call reset() before legal_actions()")
        return self._game.legal_actions(self._state)

    def _observe(self, state: GameState) -> NDArray[np.float32]:
        return self._encoder.encode(
            state, self._learner_seat, buffer_prefix=self._composer.prefix
        )


def _place_fixed_agents(
    agents: Sequence[Agent[GameState, Action]], num_seats: int, learner_seat: int
) -> list[Agent[GameState, Action] | None]:
    """Spread ``agents`` over every seat but ``learner_seat``, in order."""
    out: list[Agent[GameState, Action] | None] = []
    cursor = 0
    for seat in range(num_seats):
        if seat == learner_seat:
            out.append(None)
        else:
            out.append(agents[cursor])
            cursor += 1
    return out


def make_env(
    *,
    num_players: int = 4,
    agent_factory: Callable[[random.Random], Agent[GameState, Action]] | None = None,
    seed: int | None = None,
    **kwargs: Any,
) -> CatanEnv:
    """Convenience constructor used by tests and, later, by training.

    ``agent_factory`` is called once per non-learner seat with that seat's own
    RNG, mirroring ``server/bots.py:build_agents``.
    """
    if agent_factory is None:
        from agents.random_agent import RandomAgent

        def agent_factory(rng: random.Random) -> Agent[GameState, Action]:
            return RandomAgent(rng=rng)

    rng = random.Random(seed)
    agents = [
        agent_factory(random.Random(rng.randint(0, 2**31 - 1)))
        for _ in range(num_players - 1)
    ]
    return CatanEnv(num_players=num_players, agents=agents, seed=seed, **kwargs)
