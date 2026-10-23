"""Rule under test: ``CatanEnv.reset`` draws from its RNG in a fixed order.

Reseed -> learner seat -> one deal seed per attempt (``game.reset``), all from
one stream, then ``agent.reset()`` on the opponents. The order is part of the
contract shared with ``gamekit.rl.env.SingleAgentEnv`` (gamekit#7, D2;
gamekit#55): changing it silently changes training data for a given seed.

``GOLDEN`` was recorded from ``CatanEnv`` *before* it moved onto
``gamekit.rl.driver.place_fixed_agents`` / ``deal_until_learner`` and must
pass unchanged on both sides of that refactor. Each entry is
``(learner_seat, deal_seeds, observation_sha256_prefix, next_rng_draw)``;
``next_rng_draw`` pins how many draws the reset consumed.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("gymnasium")

from gamekit.rl.selfplay import OpponentPool  # noqa: E402

from agents.heuristic import HeuristicAgent  # noqa: E402
from engine.actions import Action  # noqa: E402
from engine.state import GameState  # noqa: E402
from rl.env import CatanEnv, make_env  # noqa: E402

RESET_SEEDS = (0, 1, 2, 3, 4, 5)

Trace = tuple[int, tuple[int, ...], str, float]


class _RecordingGame:
    """Delegates to the real game; records every deal seed and can make the
    first ``dead_deals`` deals terminal so the re-deal loop runs."""

    def __init__(self, game: Any, dead_deals: int = 0) -> None:
        self._game = game
        self._dead_deals = dead_deals
        # Held so a dead state's id() can't be recycled by a later deal.
        self._dead_states: list[Any] = []
        self.deal_seeds: list[int] = []

    def reset(self, seed: int | None = None) -> Any:
        assert seed is not None
        self.deal_seeds.append(seed)
        state = self._game.reset(seed=seed)
        if len(self.deal_seeds) <= self._dead_deals:
            self._dead_states.append(state)
        return state

    def is_terminal(self, state: Any) -> bool:
        return any(state is d for d in self._dead_states) or bool(
            self._game.is_terminal(state)
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._game, name)


def _trace(env: CatanEnv, dead_deals: int = 0) -> list[Trace]:
    out: list[Trace] = []
    for seed in RESET_SEEDS:
        game = _RecordingGame(env._game, dead_deals)
        env._game = game  # type: ignore[assignment]
        obs, _ = env.reset(seed=seed)
        env._game = game._game
        digest = hashlib.sha256(obs.tobytes()).hexdigest()[:16]
        out.append(
            (env.learner_seat, tuple(game.deal_seeds), digest, env._rng.random())
        )
    return out


def _pool(tmp_path: Path) -> OpponentPool[GameState, Action]:
    return OpponentPool(
        tmp_path,
        "catan_selfplay_*.zip",
        load_opponent=lambda path: HeuristicAgent(),
        baseline_factory=HeuristicAgent,
        rng=random.Random(0),
    )


GOLDEN: dict[str, list[Trace]] = {
    "fixed": [
        (3, (1806341205,), "d461e6560721d360", 0.04048437818077755),
        (1, (271041745,), "84ff1c933573a71b", 0.2550690257394217),
        (0, (393353683,), "d64052fb608db940", 0.08487199515892163),
        (1, (560161641,), "06468ff307e30e49", 0.36995516654807925),
        (1, (1302657532,), "35c9d1ba2addb466", 0.1031660342307158),
        (2, (1539898300,), "48a114f88fa59d2f", 0.7951935655656966),
    ],
    "pool": [
        (3, (1806341205,), "e9c6296a2f2ae3df", 0.04048437818077755),
        (1, (271041745,), "31ec80fd748d4c25", 0.2550690257394217),
        (0, (393353683,), "d0e1b46f77300180", 0.08487199515892163),
        (1, (560161641,), "86858c7b0d5e178f", 0.36995516654807925),
        (1, (1302657532,), "f4f70ac3ca29be0e", 0.1031660342307158),
        (2, (1539898300,), "48a114f88fa59d2f", 0.7951935655656966),
    ],
    "redeal": [
        (
            3,
            (1806341205, 173879092, 1112038970),
            "dfd0afb3be47ea8b",
            0.9654648863619172,
        ),
        (
            1,
            (271041745, 1095513148, 506456969),
            "c9803858ff62a994",
            0.49543508709194095,
        ),
        (0, (393353683, 364522461, 1550730039), "fa066f673978deb6", 0.8354988781294496),
        (1, (560161641, 1588945316, 2036044446), "da6b198521c0ac35", 0.625720304108054),
        (
            1,
            (1302657532, 443094727, 1701057193),
            "41edb1adff8fab1c",
            0.4788783949238976,
        ),
        (
            2,
            (1539898300, 124576495, 1999834075),
            "0b6035ca97e38963",
            0.7759585674357169,
        ),
    ],
    "fixed_seat0": [
        (0, (1654615998,), "aadaab2f8f65775e", 0.7579544029403025),
        (0, (577090037,), "819c1f0c22ed2b6f", 0.5692038748222122),
        (0, (242886303,), "0bc6157d6905dc45", 0.09158478740507359),
        (0, (1022050301,), "fcfda29549f53256", 0.5926409106271656),
        (0, (1013818839,), "624263aa340cef17", 0.30329858588551006),
        (0, (1097127993,), "0dc406245b8a41ca", 0.7417869892607294),
    ],
}


def test_fixed_agents_reset_draw_order_is_unchanged() -> None:
    assert _trace(make_env(seed=7)) == GOLDEN["fixed"]


def test_opponent_pool_reset_draw_order_is_unchanged(tmp_path: Path) -> None:
    env = CatanEnv(num_players=4, opponent_pool=_pool(tmp_path), seed=7)
    assert _trace(env) == GOLDEN["pool"]


def test_redeal_draw_order_is_unchanged() -> None:
    assert _trace(make_env(seed=7), dead_deals=2) == GOLDEN["redeal"]


def test_randomize_seat_off_reset_draw_order_is_unchanged() -> None:
    assert _trace(make_env(seed=7, randomize_seat=False)) == GOLDEN["fixed_seat0"]


def test_reset_raises_when_every_deal_ends_before_the_learner_acts() -> None:
    env = make_env(seed=7)
    game = _RecordingGame(env._game, dead_deals=10**6)
    env._game = game  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="100 deals"):
        env.reset(seed=0)

    assert len(game.deal_seeds) == 100
