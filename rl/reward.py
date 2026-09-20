"""Reward functions satisfying ``gamekit.rl.protocols.RewardFn[GameState]``.

Both use the **public** ``victory_points``, never ``true_victory_points``.
``engine/game.py`` is explicit that the gap between those two *is* the hidden
information, so shaping on the true total would leak opponents' unrevealed VP
cards into the learner's return.

gamekit ships no ``RewardShaper`` base class on purpose, so ``compute`` and
``on_episode_start`` are both defined here even where the latter is a no-op --
structural conformance needs the method to exist.
"""

from __future__ import annotations

from engine.game import victory_points
from engine.state import WINNING_VICTORY_POINTS, GameState


class SparseReward:
    """+1 for winning, -1 for losing, 0 everywhere else."""

    def compute(self, state: GameState, seat: int, done: bool) -> float:
        if not done:
            return 0.0
        return 1.0 if state.winner == seat else -1.0

    def on_episode_start(self, state: GameState, seat: int) -> None:
        return None


class ShapedReward:
    """Sparse outcome plus potential-based shaping on public victory points.

    ``F(s, s') = gamma * phi(s') - phi(s)`` with ``phi = victory_points / 10``
    is policy-invariant (Ng, Harada & Russell 1999): it cannot change which
    policy is optimal, only how fast the signal arrives. That guarantee is why
    shaping is safe to default to here and why it can live entirely inside the
    reward callable, leaving decision 10 untouched.

    It earns its place because catan episodes are long. A 4-player game runs
    roughly 1200 engine steps, so a terminal-only signal is very far away from
    the opening placements -- a different regime from truco-py, whose episodes
    are single hands and which got away with sparse rewards at ``gamma=0.99``.

    ``phi`` is defined as 0 at a terminal state, as the policy-invariance
    result requires; the win/loss term carries the outcome instead.
    """

    def __init__(self, gamma: float = 0.999, weight: float = 1.0) -> None:
        self.gamma = gamma
        self.weight = weight
        self._previous_potential = 0.0

    def _potential(self, state: GameState, seat: int) -> float:
        return victory_points(state, seat) / WINNING_VICTORY_POINTS

    def compute(self, state: GameState, seat: int, done: bool) -> float:
        outcome = 0.0
        if done:
            outcome = 1.0 if state.winner == seat else -1.0
        next_potential = 0.0 if done else self._potential(state, seat)
        shaping = self.gamma * next_potential - self._previous_potential
        self._previous_potential = next_potential
        return outcome + self.weight * shaping

    def on_episode_start(self, state: GameState, seat: int) -> None:
        self._previous_potential = self._potential(state, seat)
