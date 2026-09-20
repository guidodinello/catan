"""Win-rate evaluation with confidence intervals.

Two entry points, deliberately different:

``evaluate_winrate`` runs episodes in-process against fixed opponents and is
what the training loop calls between chunks. It is cheap, and its job is to
notice a run going wrong *while it is still running*. truco-py lost two
multi-hour self-play runs to collapse precisely because nothing evaluated
during training, so this is not an optional refinement.

The authoritative number -- the one that goes in a PR -- comes from
``experiments/benchmark.py``'s ``rl_vs_*`` modes instead, which apply the
mandatory exact seat rotation README decision 12 requires and stamp the result
JSON. In-loop eval randomises the learner's seat per episode, which balances
seats only in expectation.

Every proportion reported here carries a Wilson interval from ``gamekit.mc``.
truco-py reported RL win rates as bare percentages at n=50-150, where the
interval is +/-7 to +/-10 points and a "plateau" is indistinguishable from
noise.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from gamekit.agent import Agent
from gamekit.mc import wilson_interval

from engine.actions import Action
from engine.state import GameState
from rl.env import CatanEnv


@dataclass(frozen=True, slots=True)
class WinRate:
    """A win count with its Wilson 95% interval."""

    wins: int
    n: int
    rate: float
    ci_low: float
    ci_high: float

    def beats(self, threshold: float) -> bool:
        """Whether the interval excludes ``threshold`` from below.

        The gate is the *interval*, not the point estimate -- a 27% point
        estimate at n=200 says nothing about a 25% null.
        """
        return self.ci_low > threshold

    def __str__(self) -> str:
        return (
            f"{self.rate:.1%} [{self.ci_low:.1%}, {self.ci_high:.1%}] "
            f"({self.wins}/{self.n})"
        )


def summarize(wins: int, n: int) -> WinRate:
    """``n == 0`` raises, via ``wilson_interval`` -- an empty evaluation is a
    programming error, not a 0% win rate."""
    interval = wilson_interval(wins, n)
    return WinRate(
        wins=wins,
        n=n,
        rate=wins / n,
        ci_low=interval.lower,
        ci_high=interval.upper,
    )


def evaluate_winrate(
    predict: Callable[[np.ndarray, np.ndarray], int],
    *,
    opponent_factory: Callable[[random.Random], Agent[GameState, Action]],
    num_players: int = 4,
    n_episodes: int = 200,
    seed: int = 0,
) -> WinRate:
    """Play ``n_episodes`` and report how often the learner's seat won.

    ``predict`` takes ``(observation, action_mask)`` and returns one atom
    index, so this function never imports sb3-contrib and stays testable with
    a scripted policy.
    """
    rng = random.Random(seed)
    agents: Sequence[Agent[GameState, Action]] = [
        opponent_factory(random.Random(rng.randint(0, 2**31 - 1)))
        for _ in range(num_players - 1)
    ]
    env = CatanEnv(num_players=num_players, agents=agents, seed=seed)

    wins = 0
    for episode in range(n_episodes):
        env.reset(seed=seed * 1_000_003 + episode)
        while True:
            mask = env.action_masks()
            obs = env._observe(env.state) if env.state is not None else None
            assert obs is not None
            atom = predict(obs, mask)
            _, _, terminated, truncated, _ = env.step(np.int64(atom))
            if terminated or truncated:
                break
        state = env.state
        if terminated and state is not None and state.winner == env.learner_seat:
            wins += 1
    return summarize(wins, n_episodes)


def masked_ppo_predictor(
    model: object, *, deterministic: bool = True
) -> Callable[[np.ndarray, np.ndarray], int]:
    """Adapt a MaskablePPO model to ``evaluate_winrate``'s ``predict``."""

    def predict(obs: np.ndarray, mask: np.ndarray) -> int:
        atom, _ = model.predict(  # type: ignore[attr-defined]
            obs, action_masks=mask, deterministic=deterministic
        )
        return int(atom)

    return predict


def random_predictor(seed: int = 0) -> Callable[[np.ndarray, np.ndarray], int]:
    """Uniform over the legal *atoms* -- the reference a trained policy must
    clear, and the sanity check that the harness itself is sound.

    Worth being precise about: this is **not** the same distribution as
    ``RandomAgent``, which is uniform over legal *actions*. Sampling atoms
    weights action *kinds* roughly evenly rather than weighting each concrete
    instance evenly, so it behaves much more like ``StratifiedRandomAgent``.
    Measured at n=300: 32.3% [27.3%, 37.8%] against three ``RandomAgent``s but
    24.0% [19.5%, 29.1%] against three ``StratifiedRandomAgent``s -- i.e. it
    already clears the 25% fair-seat share against flat-random opponents
    before any training at all. Compare a trained policy against *this*, not
    against 25%, to know whether it learned anything.
    """
    rng = random.Random(seed)

    def predict(obs: np.ndarray, mask: np.ndarray) -> int:
        return int(rng.choice(np.flatnonzero(mask).tolist()))

    return predict
