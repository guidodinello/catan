"""Rule under test: a resumed leg (``--resume``) neither runs different
hyperparameters than the command line says, nor loses/overwrites the previous
leg's progress, and its ``RegressionGuard`` starts from the previous leg's
result rather than -1.0 (gamekit#009's prerequisites).

Uses tiny real ``MaskablePPO`` models (1 env, small ``n_steps``), so this
needs the training stack and is skipped without it.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("sb3_contrib")

from rl import train as train_mod  # noqa: E402
from rl.train import (  # noqa: E402
    TrainConfig,
    _load_resumed_model,
    _make_guard,
    _save_new,
    build_model,
    build_vec_env,
    train,
)

TINY = {
    "envs": 1,
    "opponents": "heuristic",
    "n_steps": 64,
    "batch_size": 64,
    "n_epochs": 1,
    "net_arch": [16],
    "eval_episodes": 1,
    "torch_threads": 1,
}


def _cfg(tmp_path: Path, **overrides: object) -> TrainConfig:
    return TrainConfig(**{**TINY, "run_dir": tmp_path / "run", **overrides})  # type: ignore[arg-type]


def _saved_checkpoint(tmp_path: Path, **overrides: object) -> Path:
    """A real checkpoint that has trained a little, so num_timesteps > 0."""
    cfg = _cfg(tmp_path, learning_rate=1e-4, ent_coef=0.01, **overrides)
    vec_env = build_vec_env(cfg)
    model = build_model(cfg, vec_env)
    model.learn(128, progress_bar=False)
    path = tmp_path / "start.zip"
    model.save(path)
    vec_env.close()
    return path


def test_explicit_hyperparameter_mismatch_on_resume_is_refused(tmp_path: Path) -> None:
    start = _saved_checkpoint(tmp_path)
    for name, wrong in (("learning_rate", 3e-4), ("ent_coef", 0.02)):
        cfg = _cfg(
            tmp_path,
            resume=start,
            explicit_hparams=frozenset({name}),
            **{name: wrong},
        )
        vec_env = build_vec_env(cfg)
        with pytest.raises(RuntimeError, match="differs from the resumed"):
            _load_resumed_model(cfg, vec_env)
        vec_env.close()


def test_matching_or_implicit_hyperparameters_resume_fine(tmp_path: Path) -> None:
    start = _saved_checkpoint(tmp_path)
    matching = _cfg(
        tmp_path,
        resume=start,
        learning_rate=1e-4,
        ent_coef=0.01,
        explicit_hparams=frozenset({"learning_rate", "ent_coef"}),
    )
    # Implicit: the CLI defaults (3e-4) differ from the pickled 1e-4, but
    # were not typed, so the pickled values win silently-but-logged.
    implicit = _cfg(tmp_path, resume=start)
    for cfg in (matching, implicit):
        vec_env = build_vec_env(cfg)
        model = _load_resumed_model(cfg, vec_env)
        assert model.lr_schedule(1.0) == pytest.approx(1e-4)
        assert model.ent_coef == pytest.approx(0.01)
        vec_env.close()


def test_resume_logs_the_effective_hyperparameters(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    start = _saved_checkpoint(tmp_path)
    cfg = _cfg(tmp_path, resume=start)
    vec_env = build_vec_env(cfg)
    with caplog.at_level("INFO", logger="rl.train"):
        _load_resumed_model(cfg, vec_env)
    vec_env.close()
    text = caplog.text
    assert "learning_rate=0.0001" in text
    assert "ent_coef=0.01" in text
    assert "num_timesteps=128" in text


def test_explicit_n_steps_overrides_the_pickled_rollout_length(tmp_path: Path) -> None:
    start = _saved_checkpoint(tmp_path)
    cfg = _cfg(
        tmp_path, resume=start, n_steps=32, explicit_hparams=frozenset({"n_steps"})
    )
    vec_env = build_vec_env(cfg)
    model = _load_resumed_model(cfg, vec_env)
    assert model.n_steps == 32
    assert model.rollout_buffer.buffer_size == 32
    vec_env.close()


def test_save_new_refuses_to_overwrite_and_leaves_the_file_intact(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    vec_env = build_vec_env(cfg)
    model = build_model(cfg, vec_env)
    target = tmp_path / "ckpt.zip"
    _save_new(model, target)
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        _save_new(model, target)
    assert target.read_bytes() == before
    vec_env.close()


def test_guard_is_seeded_from_the_previous_leg_on_resume(tmp_path: Path) -> None:
    start = tmp_path / "start.zip"
    cfg = _cfg(tmp_path, resume=start, init_rate=0.162)
    guard = _make_guard(cfg)
    assert guard.best_rate == pytest.approx(0.162)
    assert guard.best_checkpoint == start

    elsewhere = tmp_path / "best.zip"
    guard = _make_guard(dataclasses.replace(cfg, init_best=elsewhere))
    assert guard.best_checkpoint == elsewhere

    fresh = _make_guard(_cfg(tmp_path, init_rate=0.5))  # no warm start: ignored
    assert fresh.best_rate == pytest.approx(-1.0)


def test_resumed_leg_continues_the_counter_and_never_touches_earlier_files(
    tmp_path: Path,
) -> None:
    start = _saved_checkpoint(tmp_path)  # num_timesteps == 128
    cfg = _cfg(
        tmp_path,
        resume=start,
        selfplay_dir=tmp_path / "pool",
        label="leg2",
        steps=128,
        eval_every=64,
        regression_margin=1.0,  # never trips: this test is about bookkeeping
        baseline_mix=1.0,  # the fake survivor file must never be sampled
    )
    ckpt_dir = train_mod._checkpoint_dir(cfg)
    ckpt_dir.mkdir(parents=True)
    # A previous leg's file, at a name the resumed leg will reach (128 + 64).
    survivor = ckpt_dir / "leg2_192.zip"
    survivor.write_bytes(b"previous leg")

    with pytest.raises(FileExistsError):
        train(cfg)
    assert survivor.read_bytes() == b"previous leg"

    survivor.unlink()
    result = train(cfg)
    names = {p.name for p in ckpt_dir.glob("leg2_*.zip")}
    assert {"leg2_192.zip", "leg2_256.zip", "leg2_final.zip"} <= names
    assert not {"leg2_64.zip", "leg2_128.zip"} & names
    assert result.steps_completed == 256
