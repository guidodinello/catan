"""Rule under test: the in-loop eval's *reported* rate is a fresh sample at every
checkpoint, while the *guard* rate stays on one fixed, paired game set
(issue #25; before the fix both replayed the same 200 setups every time).

Torch-free, so it runs in the ``Tests (RL)`` CI job: ``evaluate_winrate`` is
monkeypatched, and on ``device="cpu"`` ``_cpu_eval_model`` is a no-op.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("gymnasium")

from rl import train as train_mod  # noqa: E402
from rl.evaluate import RegressionGuard, summarize  # noqa: E402
from rl.train import TrainConfig, evaluate_in_loop, in_loop_eval_seeds  # noqa: E402

EVAL_EVERY = 250_000


def test_reported_seed_advances_and_guard_seed_is_fixed() -> None:
    cfg = TrainConfig(seed=3)
    guard_a, reported_a = in_loop_eval_seeds(cfg, EVAL_EVERY)
    guard_b, reported_b = in_loop_eval_seeds(cfg, 2 * EVAL_EVERY)
    assert guard_a == guard_b == 3 + 977  # unchanged from 004/006
    assert reported_a != reported_b
    assert reported_a != guard_a


def test_resumed_leg_never_replays_the_first_legs_reported_seeds() -> None:
    cfg = TrainConfig(seed=3)
    first_leg = {
        in_loop_eval_seeds(cfg, step)[1]
        for step in range(EVAL_EVERY, 2_000_001, EVAL_EVERY)
    }
    # A resumed leg's `done` is offset by the loaded model's num_timesteps.
    resumed_from = 2_031_616
    second_leg = {
        in_loop_eval_seeds(cfg, resumed_from + step)[1]
        for step in range(EVAL_EVERY, 2_000_001, EVAL_EVERY)
    }
    assert first_leg.isdisjoint(second_leg)


def test_two_consecutive_in_loop_evals_get_different_reported_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeds: list[int] = []

    def fake_evaluate_winrate(*_args: Any, seed: int, **_kwargs: Any) -> Any:
        seeds.append(seed)
        return summarize(10, 200)

    monkeypatch.setattr(train_mod, "evaluate_winrate", fake_evaluate_winrate)
    monkeypatch.setattr(train_mod, "masked_ppo_predictor", lambda _model: None)
    cfg = TrainConfig(seed=0, device="cpu")

    model: Any = object()
    evaluate_in_loop(cfg, model, EVAL_EVERY)
    evaluate_in_loop(cfg, model, 2 * EVAL_EVERY)

    guard_1, reported_1, guard_2, reported_2 = seeds
    assert guard_1 == guard_2 == 977
    assert reported_1 != reported_2
    assert reported_1 != guard_1
    assert reported_2 != guard_2


def test_best_checkpoint_follows_the_reported_rate_not_the_guard_rate() -> None:
    """Mirrors the wiring in ``train()``: two trackers, each fed its own rate."""
    guard = RegressionGuard(margin=0.10, patience=2)
    best = RegressionGuard(margin=0.10, patience=2)
    evals = [  # (checkpoint, guard rate, reported rate)
        (Path("ckpt_1.zip"), 0.30, 0.15),  # lucky on the fixed set only
        (Path("ckpt_2.zip"), 0.20, 0.25),
    ]
    for checkpoint, guard_rate, reported_rate in evals:
        guard.observe(guard_rate, checkpoint)
        best.observe(reported_rate, checkpoint)
    assert guard.best_checkpoint == Path("ckpt_1.zip")
    assert best.best_checkpoint == Path("ckpt_2.zip")
