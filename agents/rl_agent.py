"""A ``gamekit.Agent`` backed by a trained MaskablePPO checkpoint.

This is the seam that lets a trained policy be evaluated by the *existing*
Phase 3 machinery instead of a parallel one: ``RLAgent`` satisfies the same
``Agent`` protocol as ``HeuristicAgent``, so it drops into
``experiments/benchmark.py`` (mandatory seat rotation, Wilson CIs, stamped
result JSON), ``experiments/rollout.py``, and ``server/bots.py`` -- the last
of which makes a trained agent playable in the web GUI.

The policy emits *atoms*, not actions, so ``choose_action`` runs the
composition loop from ``rl.action_space`` internally and returns a single
fully-formed engine ``Action``. Callers never see the factored head.

sb3-contrib and torch are imported lazily inside ``_load_model`` so this
module is importable -- and testable -- with only the ``rl`` extra installed.
"""

from __future__ import annotations

import random
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from engine.actions import Action
from engine.state import GameState
from rl.action_space import ActionComposer
from rl.encoder import ObservationEncoder

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sb3_contrib import MaskablePPO


@cache
def _load_model(checkpoint: str, device: str) -> MaskablePPO:
    """Load once per (path, device) per process.

    ``OpponentPool.sample()`` calls its loader afresh for every non-learner
    seat of every episode, so without this cache a self-play worker would
    re-read the same zip thousands of times -- the caching responsibility
    gamekit explicitly leaves to the integrator.

    Also the one place to fix a real oversubscription bug found running the
    PR #18 gate benchmark: each of BLAS/torch's own intra-op thread pools is
    sized off the *host's* core count by default, not the pool's own share of
    it. Under ``experiments/rollout.py``'s bare ``ProcessPoolExecutor`` (no
    initializer) that meant every one of N worker processes independently
    spun up ~33 threads on a 20-core box -- measured at load average ~210
    with 16 workers. ``rl/train.py`` already pins this via
    ``torch.set_num_threads(1)`` for its own subprocesses; this is the same
    fix for the evaluation path, applied once here (this function runs
    exactly once per process, before the first forward pass) rather than at
    every call site that might load a checkpoint. ``OMP_NUM_THREADS`` is not
    used for this -- it has to be set before the process starts, which is not
    available to us here, and torch's own setter is unconditional besides.
    """
    import torch

    torch.set_num_threads(1)
    from sb3_contrib import MaskablePPO

    return MaskablePPO.load(checkpoint, device=device)


class RLAgent:
    """Greedy (or sampled) MaskablePPO policy as a drop-in ``CatanAgent``."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        name: str = "rl",
        deterministic: bool = True,
        device: str = "cpu",
        rng: random.Random | None = None,
    ) -> None:
        self.name = name
        self.checkpoint = str(checkpoint)
        self.deterministic = deterministic
        self.device = device
        self.rng = rng or random.Random()
        self._encoder: ObservationEncoder | None = None
        self._encoded_board: Any = None

    @property
    def model(self) -> MaskablePPO:
        return _load_model(self.checkpoint, self.device)

    def reset(self) -> None:
        self._encoded_board = None

    def choose_action(
        self, state: GameState, legal_actions: list[Action], player_idx: int
    ) -> Action:
        num_players = len(state.players)
        if self._encoder is None or self._encoder.num_players != num_players:
            self._encoder = ObservationEncoder(num_players)
            self._encoded_board = None
        # Board statics are per-game; recompute when the board object changes
        # rather than trusting every driver to call reset() first.
        if state.board is not self._encoded_board:
            self._encoder.reset(state)
            self._encoded_board = state.board

        composer = ActionComposer()
        model = self.model
        while True:
            mask = composer.mask(legal_actions, player_idx, num_players)
            if not any(mask):
                raise RuntimeError(
                    "no encodable action available -- every legal action was an "
                    "open-ended trade sentinel, which the atom head reserves"
                )
            obs = self._encoder.encode(state, player_idx, buffer_prefix=composer.prefix)
            atom, _ = model.predict(
                obs,
                action_masks=np.array(mask, dtype=bool),
                deterministic=self.deterministic,
            )
            emitted = composer.push(int(atom), legal_actions, player_idx, num_players)
            if emitted is not None:
                return emitted
