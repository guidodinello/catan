"""Rule under test: ``--ent-schedule`` is a piecewise-linear function of the
model's *cumulative* ``num_timesteps``, applied before every rollout by a
callback -- so it does not restart per ``learn()`` chunk, and overwrites the
pickled ``ent_coef`` on ``--resume`` (gamekit note 010 / 009's trap).

The pure-function and CLI tests need only the ``rl`` extra (they run in CI's
test-rl job); the model tests need the training stack and skip without it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("gymnasium")

from rl.train import (  # noqa: E402
    TrainConfig,
    build_parser,
    config_from_args,
    ent_coef_at,
    parse_ent_schedule,
)

RAMP = parse_ent_schedule("0:0.005,1000000:0.02")
DECAY = parse_ent_schedule("0:0.02,3000000:0.005")


def test_parse_reads_breakpoints() -> None:
    assert RAMP == ((0, 0.005), (1_000_000, 0.02))


@pytest.mark.parametrize(
    "text", ["", "5", "0:0.1,0:0.2", "10:0.1,5:0.2", "-1:0.1", "0:-0.1", "a:b"]
)
def test_parse_rejects_malformed_schedules(text: str) -> None:
    with pytest.raises(ValueError):
        parse_ent_schedule(text)


def test_interpolates_and_clamps() -> None:
    assert ent_coef_at(RAMP, 0) == 0.005
    assert ent_coef_at(RAMP, 500_000) == pytest.approx(0.0125)
    assert ent_coef_at(RAMP, 1_000_000) == 0.02
    assert ent_coef_at(RAMP, 3_047_424) == 0.02  # constant after the ramp
    assert ent_coef_at(DECAY, 1_500_000) == pytest.approx(0.0125)
    assert ent_coef_at(DECAY, 3_047_424) == 0.005  # clamped past the last point


def test_single_point_is_the_constant_shorthand() -> None:
    constant = parse_ent_schedule("0:0.01")
    assert {ent_coef_at(constant, s) for s in (0, 1, 10**7)} == {0.01}


def test_cli_schedule_and_ent_coef_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--ent-coef", "0.01", "--ent-schedule", "0:0.01"])


def test_cli_schedule_sets_step_zero_ent_coef() -> None:
    cfg = config_from_args(build_parser(), ["--ent-schedule", "0:0.005,1000000:0.02"])
    assert cfg.ent_schedule == RAMP
    assert cfg.ent_coef == 0.005
    assert "ent_coef" not in cfg.explicit_hparams


def test_cli_without_schedule_is_unchanged() -> None:
    cfg = config_from_args(build_parser(), ["--ent-coef", "0.02"])
    assert cfg.ent_schedule is None
    assert cfg.ent_coef == 0.02


# -- model-level: needs the training stack -----------------------------------

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


def _record_ent_coef_per_update(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Capture ``self.ent_coef`` at the moment each PPO update starts."""
    from sb3_contrib import MaskablePPO

    seen: list[float] = []
    real_train = MaskablePPO.train

    def spy(self: MaskablePPO) -> None:
        seen.append(self.ent_coef)
        real_train(self)

    monkeypatch.setattr(MaskablePPO, "train", spy)
    return seen


def test_callback_follows_cumulative_steps_across_learn_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("sb3_contrib")
    from rl.train import _ent_schedule_callback, build_model, build_vec_env

    schedule = parse_ent_schedule("0:0.0,256:0.256")  # 1e-3 per step
    cfg = _cfg(tmp_path, ent_coef=0.5)  # constructor value must be overwritten
    vec_env = build_vec_env(cfg)
    model = build_model(cfg, vec_env)
    seen = _record_ent_coef_per_update(monkeypatch)
    logged: list[float] = []
    from stable_baselines3.common.logger import Logger

    real_record = Logger.record

    def spy_record(self: Logger, key: str, value: float, **kw: object) -> None:
        if key == "train/ent_coef_effective":
            logged.append(value)
        real_record(self, key, value, **kw)

    monkeypatch.setattr(Logger, "record", spy_record)
    callback = _ent_schedule_callback(schedule)

    model.learn(128, callback=callback, reset_num_timesteps=True, progress_bar=False)
    model.learn(128, callback=callback, reset_num_timesteps=False, progress_bar=False)
    vec_env.close()

    # rollouts start at num_timesteps 0, 64, 128, 192 (64 steps each)
    assert seen == pytest.approx([0.0, 0.064, 0.128, 0.192])  # no per-chunk restart
    assert logged == pytest.approx(seen)  # what is logged is what the update used


def test_resume_overwrites_the_pickled_ent_coef_before_the_first_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("sb3_contrib")
    from sb3_contrib import MaskablePPO

    from rl.train import _ent_schedule_callback, build_model, build_vec_env

    cfg = _cfg(tmp_path, ent_coef=0.01)
    vec_env = build_vec_env(cfg)
    model = build_model(cfg, vec_env)
    model.learn(128, progress_bar=False)
    path = tmp_path / "start.zip"
    model.save(path)
    vec_env.close()

    vec_env = build_vec_env(cfg)
    resumed = MaskablePPO.load(path, env=vec_env)
    assert resumed.ent_coef == 0.01  # the trap: the pickled value survives a load
    start = resumed.num_timesteps
    schedule = parse_ent_schedule("0:0.02,1000000:0.03")
    seen = _record_ent_coef_per_update(monkeypatch)
    resumed.learn(
        64,
        callback=_ent_schedule_callback(schedule),
        reset_num_timesteps=False,
        progress_bar=False,
    )
    vec_env.close()

    assert seen == pytest.approx([ent_coef_at(schedule, start)])
    assert seen[0] != 0.01
