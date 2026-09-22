"""Rule under test: self-play wiring and the automatic-stop-on-regression
control -- the two things PR #19 adds over PR #18's fixed-opponent training.

Nothing here trains a model or touches torch. ``RegressionGuard`` is pure
Python; the self-play env construction is exercised through ``OpponentPool``
with an *empty* checkpoint directory, so it deterministically falls back to
``HeuristicAgent`` without ever calling ``load_opponent`` -- exactly the path
that never touches sb3-contrib. That's what lets this whole module run in the
torch-free ``Tests (RL)`` CI job.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

pytest.importorskip("gymnasium")

import numpy as np  # noqa: E402

from rl.evaluate import RegressionGuard  # noqa: E402
from rl.train import (  # noqa: E402
    TrainConfig,
    TrainResult,
    _checkpoint_dir,
    _resume_step_bookkeeping,
    _seed_selfplay_pool,
    build_env,
    build_opponent,
    eval_opponent_kind,
)

# ---------------------------------------------------------------------------
# RegressionGuard
# ---------------------------------------------------------------------------


def test_a_new_best_never_stops_and_updates_best_rate_and_checkpoint() -> None:
    guard = RegressionGuard(margin=0.10, patience=2)
    assert guard.observe(0.30, Path("a.zip")) is False
    assert guard.best_rate == 0.30
    assert guard.best_checkpoint == Path("a.zip")
    assert guard.observe(0.45, Path("b.zip")) is False
    assert guard.best_rate == 0.45
    assert guard.best_checkpoint == Path("b.zip")


def test_a_dip_within_margin_does_not_count_as_a_regression() -> None:
    guard = RegressionGuard(margin=0.10, patience=1)
    guard.observe(0.50, Path("best.zip"))
    # 0.45 is only 5 points below best -- inside the 10-point margin.
    assert guard.observe(0.45, Path("dip.zip")) is False
    assert guard.best_checkpoint == Path("best.zip"), "best must not move on a dip"


def test_a_single_regression_past_margin_does_not_stop_with_patience_two() -> None:
    guard = RegressionGuard(margin=0.10, patience=2)
    guard.observe(0.50, Path("best.zip"))
    assert guard.observe(0.30, Path("bad1.zip")) is False


def test_sustained_regression_past_margin_stops_at_patience() -> None:
    guard = RegressionGuard(margin=0.10, patience=2)
    guard.observe(0.50, Path("best.zip"))
    assert guard.observe(0.30, Path("bad1.zip")) is False
    assert guard.observe(0.25, Path("bad2.zip")) is True


def test_a_recovery_between_regressions_resets_the_streak() -> None:
    """The actual shape of truco's collapses -- sustained, not a single bad
    eval -- is what this guards; a recovery mid-slide should not carry over
    streak count toward a later, unrelated regression."""
    guard = RegressionGuard(margin=0.10, patience=2)
    guard.observe(0.50, Path("best.zip"))
    guard.observe(0.30, Path("bad1.zip"))  # streak=1
    guard.observe(0.48, Path("recovered.zip"))  # within margin -> streak reset
    assert guard.observe(0.30, Path("bad2.zip")) is False, "streak should have reset"


def test_the_stopping_checkpoint_is_reported_as_best_not_last() -> None:
    guard = RegressionGuard(margin=0.10, patience=1)
    guard.observe(0.60, Path("peak.zip"))
    guard.observe(0.20, Path("collapsed.zip"))
    assert guard.best_checkpoint == Path("peak.zip")
    assert guard.best_rate == 0.60


# ---------------------------------------------------------------------------
# Resume step bookkeeping -- the regression test for a real incident: a
# resumed run's checkpoints silently overwrote an earlier leg's checkpoints
# of the same name, because both legs counted steps from zero.
# ---------------------------------------------------------------------------


def test_a_fresh_run_starts_counting_at_zero_and_resets_sb3s_counter() -> None:
    start, target, reset = _resume_step_bookkeeping(None, 2_000_000)
    assert start == 0
    assert target == 2_000_000
    assert reset is True


def test_a_resumed_run_continues_the_cumulative_counter_and_never_resets() -> None:
    start, target, reset = _resume_step_bookkeeping(4_500_000, 2_000_000)
    assert start == 4_500_000
    assert target == 6_500_000
    assert reset is False


def test_steps_always_means_new_steps_this_invocation_not_a_cumulative_target() -> None:
    """The bug in one sentence: --steps 2000000 must run 2,000,000 *more*
    steps regardless of how much the resumed checkpoint already trained."""
    for resumed_from in (None, 0, 1, 4_500_000, 10_000_000):
        start, target, _ = _resume_step_bookkeeping(resumed_from, 2_000_000)
        assert target - start == 2_000_000


def test_resumed_checkpoint_step_numbers_never_collide_with_a_fresh_run() -> None:
    """The actual incident, reproduced directly: two invocations of the same
    label -- one fresh, one resumed from real progress -- must never both
    produce a checkpoint named e.g. ``label_250000.zip`` for genuinely
    different amounts of training."""
    fresh_start, fresh_target, _ = _resume_step_bookkeeping(None, 2_000_000)
    resumed_start, resumed_target, _ = _resume_step_bookkeeping(4_500_000, 2_000_000)
    chunk = 250_000
    fresh_checkpoints = {
        fresh_start + i * chunk
        for i in range(1, (fresh_target - fresh_start) // chunk + 1)
    }
    resumed_checkpoints = {
        resumed_start + i * chunk
        for i in range(1, (resumed_target - resumed_start) // chunk + 1)
    }
    assert fresh_checkpoints.isdisjoint(resumed_checkpoints)


# ---------------------------------------------------------------------------
# TrainConfig / CLI plumbing for the new fields
# ---------------------------------------------------------------------------


def test_new_train_config_fields_are_values_not_slot_descriptors() -> None:
    """Same regression class as the member_descriptor bug PR #18 hit --
    extended to cover every field added in this PR."""
    cfg = TrainConfig()
    assert isinstance(cfg.baseline_mix, float)
    assert isinstance(cfg.regression_margin, float)
    assert isinstance(cfg.regression_patience, int)
    assert cfg.selfplay_dir is None
    assert cfg.eval_opponents is None


def test_eval_opponent_kind_falls_back_to_opponents_when_unset() -> None:
    cfg = TrainConfig(opponents="random", eval_opponents=None)
    assert eval_opponent_kind(cfg) == "random"


def test_eval_opponent_kind_prefers_the_explicit_override() -> None:
    """The whole point of self-play: eval must measure against the actual
    gate (heuristic) regardless of what cfg.opponents says, since opponents
    stops describing training opponents once selfplay_dir is set."""
    cfg = TrainConfig(opponents="random", eval_opponents="heuristic")
    assert eval_opponent_kind(cfg) == "heuristic"


def test_checkpoint_dir_is_run_dir_in_fixed_opponent_mode() -> None:
    plain = TrainConfig(run_dir=Path("/tmp/plain"))
    assert _checkpoint_dir(plain) == Path("/tmp/plain")


def test_checkpoint_dir_is_run_scoped_under_selfplay_dir_by_label() -> None:
    """Run-scoped by gamekit#23/v0.3.0's run_id, not the bare selfplay_dir --
    the whole point is that two runs never share a glob."""
    selfplay = TrainConfig(
        run_dir=Path("/tmp/plain"), selfplay_dir=Path("/tmp/pool"), label="run_a"
    )
    assert _checkpoint_dir(selfplay) == Path("/tmp/pool/run_a")

    other = TrainConfig(
        run_dir=Path("/tmp/plain"), selfplay_dir=Path("/tmp/pool"), label="run_b"
    )
    assert _checkpoint_dir(other) == Path("/tmp/pool/run_b")
    assert _checkpoint_dir(selfplay) != _checkpoint_dir(other)


# ---------------------------------------------------------------------------
# Pool seeding
# ---------------------------------------------------------------------------


def test_seed_selfplay_pool_copies_resume_into_the_run_scoped_directory(
    tmp_path: Path,
) -> None:
    resume = tmp_path / "checkpoint.zip"
    resume.write_bytes(b"fake model bytes")
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()

    cfg = TrainConfig(label="myrun", resume=resume, selfplay_dir=pool_dir)
    _seed_selfplay_pool(cfg)

    # Run-scoped: under pool_dir/myrun/, not pool_dir/ directly.
    seeded = pool_dir / "myrun" / "myrun_seed.zip"
    assert seeded.exists()
    assert seeded.read_bytes() == b"fake model bytes"


def test_seed_selfplay_pool_is_idempotent(tmp_path: Path) -> None:
    resume = tmp_path / "checkpoint.zip"
    resume.write_bytes(b"v1")
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    cfg = TrainConfig(label="myrun", resume=resume, selfplay_dir=pool_dir)

    _seed_selfplay_pool(cfg)
    seeded = pool_dir / "myrun" / "myrun_seed.zip"
    seeded.write_bytes(b"already there, do not overwrite")
    _seed_selfplay_pool(cfg)

    assert seeded.read_bytes() == b"already there, do not overwrite"


def test_seed_selfplay_pool_is_a_noop_without_both_resume_and_selfplay_dir(
    tmp_path: Path,
) -> None:
    resume = tmp_path / "checkpoint.zip"
    resume.write_bytes(b"x")

    _seed_selfplay_pool(TrainConfig(resume=resume, selfplay_dir=None))  # no dir
    _seed_selfplay_pool(TrainConfig(resume=None, selfplay_dir=tmp_path))  # no resume
    assert list(tmp_path.iterdir()) == [resume]


# ---------------------------------------------------------------------------
# Self-play env construction (empty pool -> baseline fallback, no torch)
# ---------------------------------------------------------------------------


def test_build_env_in_selfplay_mode_builds_a_playable_env_from_an_empty_pool(
    tmp_path: Path,
) -> None:
    """An empty pool directory falls back to HeuristicAgent unconditionally
    (gamekit's own documented behavior), so this never calls load_opponent
    and never touches sb3-contrib -- the whole self-play wiring is
    exercisable without a trained checkpoint or torch installed."""
    cfg = TrainConfig(
        selfplay_dir=tmp_path, label="wiring", num_players=4, seed=1, envs=1
    )
    env = build_env(cfg, rank=0)

    obs, info = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape
    rng = random.Random(1)
    for _ in range(500):
        mask = env.action_masks()
        legal = np.flatnonzero(mask)
        assert legal.size > 0
        _, _, terminated, truncated, _ = env.step(np.int64(rng.choice(legal.tolist())))
        if terminated or truncated:
            env.reset(seed=rng.randint(0, 2**31 - 1))


def test_build_env_in_fixed_mode_is_unaffected_by_selfplay_fields() -> None:
    """selfplay_dir=None must still take the PR #18 fixed-opponent path --
    this is the regression test for accidentally always branching to
    self-play."""
    cfg = TrainConfig(opponents="heuristic", selfplay_dir=None, envs=1, seed=2)
    env = build_env(cfg, rank=0)
    obs, _ = env.reset(seed=2)
    assert obs.shape == env.observation_space.shape


def test_two_runs_sharing_a_parent_directory_never_see_each_others_checkpoints(
    tmp_path: Path,
) -> None:
    """The actual property gamekit#23/v0.3.0 exists for: this is what let a
    stale, collapsed truco_selfplay_final.zip stay sample-able forever before
    the fix. Exercised through catan's own _checkpoint_dir, not by
    constructing OpponentPool directly, so a regression here is caught even
    if only catan's plumbing changes."""
    from gamekit.rl.selfplay import OpponentPool

    parent = tmp_path / "selfplay"
    run_a = TrainConfig(label="run_a", selfplay_dir=parent)
    run_b = TrainConfig(label="run_b", selfplay_dir=parent)

    dir_a = _checkpoint_dir(run_a)
    dir_b = _checkpoint_dir(run_b)
    dir_a.mkdir(parents=True)
    dir_b.mkdir(parents=True)
    (dir_a / "run_a_500000.zip").write_bytes(b"a")
    (dir_b / "run_b_500000.zip").write_bytes(b"b")

    pool_a: OpponentPool[object, object] = OpponentPool(
        parent,
        "run_a_*.zip",
        load_opponent=lambda p: None,  # type: ignore[arg-type,return-value]
        baseline_factory=lambda: None,  # type: ignore[arg-type,return-value]
        run_id="run_a",
    )
    assert [p.name for p in pool_a.checkpoints()] == ["run_a_500000.zip"]


def test_build_opponent_and_train_result_are_unaffected_by_the_new_fields() -> None:
    """Cheap sanity check that adding fields to TrainConfig didn't change
    unrelated behavior."""
    agent = build_opponent("heuristic", random.Random(0))
    assert agent.name == "heuristic"
    result = TrainResult(
        final_checkpoint=Path("f.zip"),
        best_checkpoint=Path("b.zip"),
        best_win_rate=0.9,
        steps_completed=1000,
        stopped_early=True,
    )
    assert result.stopped_early
