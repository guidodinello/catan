"""Experiments 011/012's driver (``experiments/search_eval.py``): seed hygiene,
journaled chunked resume, graceful/immediate stop, and the paired statistics.
Most tests stub the per-game worker; the last ones play real games with a tiny
checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("numpy")  # critic_calibration needs the rl extra

from experiments import critic_calibration as cc  # noqa: E402
from experiments import search_eval as se  # noqa: E402


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _range(name: str) -> tuple[int, int]:
    lo, n = se.SEED_RANGES[name]
    return lo, lo + n - 1


def test_smoke_seeds_are_disjoint_from_every_used_range() -> None:
    smoke = _range("smoke")
    used = [(s, s + m - 1) for s, m in cc.SEED_RANGES.values()] + list(cc.OFF_LIMITS)
    assert not any(_overlaps(smoke, u) for u in used)
    assert se.SEED_RANGES["smoke"][1] % 4 == 0


def test_fresh_012_seeds_are_disjoint_from_every_used_range() -> None:
    fresh = _range("fresh_012")
    million = 1_000_000
    used = (
        [(s, s + m - 1) for s, m in cc.SEED_RANGES.values()]  # 009's arms and smoke
        + list(cc.OFF_LIMITS)  # benchmark 1..10000 and 008's smoke 90001..100000
        + [_range("smoke"), _range("measured")]  # 011
        # whole million blocks: #38's reserved 3000001+ and 011's smoke 4000001+
        + [(3 * million + 1, 4 * million), (4 * million + 1, 5 * million)]
    )
    assert not any(_overlaps(fresh, u) for u in used)
    assert se.SEED_RANGES["fresh_012"][1] % 4 == 0


def test_measured_seeds_are_the_benchmark_boards() -> None:
    assert se.SEED_RANGES["measured"][0] == 1


def test_mcnemar_exact() -> None:
    assert se.mcnemar_exact(0, 0) == 1.0
    assert se.mcnemar_exact(5, 5) == 1.0
    # 10 vs 0 discordant pairs: 2 * (1/1024)
    assert se.mcnemar_exact(10, 0) == pytest.approx(2 / 1024)


def test_compare_pairs_on_common_boards() -> None:
    a = [{"e": e, "won": int(e % 2 == 0)} for e in range(1, 9)]
    b = [{"e": e, "won": int(e % 4 == 0)} for e in range(1, 9)]
    out = se.compare(a, b)
    assert (out["n"], out["wins_a"], out["wins_b"]) == (8, 4, 2)
    assert (out["only_a"], out["only_b"]) == (2, 0)


def test_gate_needs_both_conditions() -> None:
    cmp_ok = {"diff": 0.06, "p_value": 0.001, "mcnemar_p": 0.0004}
    assert se.gate({"wilson95": [0.26, 0.29]}, cmp_ok)["passed"]
    assert not se.gate({"wilson95": [0.24, 0.29]}, cmp_ok)["passed"]
    assert not se.gate({"wilson95": [0.26, 0.29]}, {**cmp_ok, "p_value": 0.2})["passed"]
    assert not se.gate({"wilson95": [0.26, 0.29]}, {**cmp_ok, "diff": -0.01})["passed"]


FAKE_BASE = 100  # a fake seed block, inside no real range: ranges are patched in


def _fake_play(job: se.Job) -> dict[str, Any]:
    return {
        "e": job.engine_seed,
        "worker_s": 0.1,
        "rss_mb": 1.0,
        "module_paths": {"agents": str(se.ROOT / "agents")},
        "digest": f"d{job.engine_seed}",
        "won": job.engine_seed % 3 == 0,
    }


def _args(tmp: Path, games: int, ckpt: Path) -> argparse.Namespace:
    return argparse.Namespace(
        arm="heur", sims=8, games=games, seed_base=None, seed_range=None, chunk=8,
        workers=1, checkpoint=str(ckpt), out_dir=str(tmp), label="t",
        timing_only=False, allow_commit_change=False,
    )  # fmt: skip


@pytest.fixture
def ckpt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "ck.zip"
    path.write_bytes(b"x")
    monkeypatch.setattr(se, "_git", lambda *a: "" if a[0] == "status" else "abc123")
    # tiny fake board range so a few chunks are a whole run
    monkeypatch.setitem(se.SEED_RANGES, "measured", (1, 24))
    return path


def _sequential(
    jobs: Any, workers: int, on_result: Any, stop_file: Path, fn: Any = None
) -> str:
    """Stand-in for ``_run_jobs``: same contract, no processes."""
    for job in jobs:
        if stop_file.exists():
            return "stop_file"
        on_result(_fake_play(job))
    return "completed"


def _stable(arm_dir: Path) -> list[dict[str, Any]]:
    """Games minus the fields that legitimately differ between runs."""
    return [
        {k: v for k, v in g.items() if k not in {"worker_s", "rss_mb", "invocation_id"}}
        for g in se.load_arm(arm_dir)
    ]


def test_stop_then_resume_reproduces_an_uninterrupted_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ckpt: Path
) -> None:
    monkeypatch.setattr(se, "_run_jobs", _sequential)
    se.run(_args(tmp_path / "full", 24, ckpt))
    full = _stable(tmp_path / "full" / "t")
    assert [g["e"] for g in full] == list(range(1, 25))

    part = tmp_path / "part"
    arm = part / "t"
    stop = arm / se.STOP_FILE
    replayed: list[int] = []

    def stopping(
        jobs: Any, workers: int, on_result: Any, stop_file: Path, fn: Any = None
    ) -> str:
        replayed.extend(j.engine_seed for j in jobs)
        n = 0

        def counted(game: dict[str, Any]) -> None:
            nonlocal n
            on_result(game)
            n += 1
            if n == 11:  # mid second chunk
                stop.touch()

        return _sequential(jobs, workers, counted, stop_file)

    monkeypatch.setattr(se, "_run_jobs", stopping)
    with pytest.raises(SystemExit) as exc:
        se.run(_args(part, 24, ckpt))
    assert exc.value.code == se.EXIT_STOPPED
    assert not stop.exists()  # consumed: resuming is the same command
    assert (arm / "chunk_1.json").exists()
    assert len((arm / "chunk_9.partial.jsonl").read_text().splitlines()) == 1 + 3

    monkeypatch.setattr(se, "_run_jobs", _sequential)
    replayed.clear()
    se.run(_args(part, 24, ckpt))
    assert _stable(arm) == full
    assert not list(arm.glob("*.partial.jsonl"))
    prov = se.provenance(arm)
    assert [i["reason"] for i in prov["invocations"]] == ["stop_file", "completed"]
    assert len({i for ids in prov["chunk_invocations"].values() for i in ids}) == 2


def test_resume_replays_only_the_missing_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ckpt: Path
) -> None:
    seen: list[int] = []

    def crashing(
        jobs: Any, workers: int, on_result: Any, stop_file: Path, fn: Any = None
    ) -> str:
        seen.extend(j.engine_seed for j in jobs)
        for job in list(jobs)[:3]:
            on_result(_fake_play(job))
        raise KeyboardInterrupt

    monkeypatch.setattr(se, "_run_jobs", crashing)
    with pytest.raises(KeyboardInterrupt):
        se.run(_args(tmp_path, 24, ckpt))
    seen.clear()

    def recording(
        jobs: Any, workers: int, on_result: Any, stop_file: Path, fn: Any = None
    ) -> str:
        seen.extend(j.engine_seed for j in jobs)
        return "completed"

    monkeypatch.setattr(se, "_run_jobs", recording)
    se.run(_args(tmp_path, 24, ckpt))
    assert seen == list(range(4, 25))  # seeds 1..3 were journaled


def test_truncated_journal_line_is_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ckpt: Path
) -> None:
    def partial_run(
        jobs: Any, workers: int, on_result: Any, stop_file: Path, fn: Any = None
    ) -> str:
        for job in list(jobs)[:3]:
            on_result(_fake_play(job))
        return "signal"

    monkeypatch.setattr(se, "_run_jobs", partial_run)
    with pytest.raises(SystemExit):
        se.run(_args(tmp_path, 8, ckpt))
    journal = tmp_path / "t" / "chunk_1.partial.jsonl"
    journal.write_text(journal.read_text() + '{"e": 4, "won')  # crash mid-write
    monkeypatch.setattr(se, "_run_jobs", _sequential)
    se.run(_args(tmp_path, 8, ckpt))
    assert [g["e"] for g in se.load_arm(tmp_path / "t")] == list(range(1, 9))


def test_resume_refuses_a_different_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ckpt: Path
) -> None:
    monkeypatch.setattr(se, "_run_jobs", _sequential)
    se.run(_args(tmp_path, 8, ckpt))
    changed = _args(tmp_path, 8, ckpt)
    changed.sims = 16
    with pytest.raises(SystemExit):
        se.run(changed)


def test_resume_refuses_a_different_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ckpt: Path
) -> None:
    monkeypatch.setattr(se, "_run_jobs", _sequential)
    se.run(_args(tmp_path, 8, ckpt))
    monkeypatch.setattr(se, "environment", lambda: {"python": "0", "torch": "other"})
    with pytest.raises(SystemExit):
        se.run(_args(tmp_path, 8, ckpt))


def test_a_stale_stop_file_refuses_to_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ckpt: Path
) -> None:
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / se.STOP_FILE).touch()
    with pytest.raises(SystemExit) as exc:
        se.run(_args(tmp_path, 8, ckpt))
    assert exc.value.code == se.EXIT_STOPPED


def test_measured_run_refuses_a_dirty_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ckpt: Path
) -> None:
    monkeypatch.setattr(
        se, "_git", lambda *a: "M file" if a[0] == "status" else "abc123"
    )
    with pytest.raises(SystemExit):
        se.run(_args(tmp_path, 8, ckpt))


def test_timing_runs_cannot_use_measured_seeds(tmp_path: Path, ckpt: Path) -> None:
    args = _args(tmp_path, 8, ckpt)
    args.timing_only, args.seed_range = True, "measured"
    with pytest.raises(SystemExit):
        se.run(args)


# --- the real process pool, with a fake game ---------------------------------


def slow_fake_game(job: se.Job) -> dict[str, Any]:
    (Path(job.checkpoint) / f"pid{os.getpid()}").touch()  # checkpoint = a tmp dir here
    time.sleep(0.3)
    return _fake_play(job)


def _jobs(n: int, scratch: Path) -> list[se.Job]:
    return [
        se.Job("m", str(scratch), "heur", 8, 4, e, e, True, False)
        for e in range(1, n + 1)
    ]


def test_pool_drains_in_flight_games_on_a_stop_file(tmp_path: Path) -> None:
    stop, got = tmp_path / "STOP", []

    def on_result(game: dict[str, Any]) -> None:
        got.append(game["e"])
        stop.touch()

    reason = se._run_jobs(_jobs(40, tmp_path), 2, on_result, stop, slow_fake_game)
    assert reason == "stop_file"
    assert 2 <= len(got) <= 4  # only the games already in flight finished


def test_pool_kills_workers_on_sigterm(tmp_path: Path) -> None:
    got: list[int] = []
    timer = threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.start()
    try:
        reason = se._run_jobs(
            _jobs(40, tmp_path), 2, lambda g: got.append(g["e"]), tmp_path / "STOP",
            slow_fake_game,
        )  # fmt: skip
    finally:
        timer.cancel()
    assert reason == "signal"
    assert len(got) < 40
    pids = [int(f.name[3:]) for f in tmp_path.glob("pid*")]
    assert pids
    deadline = time.time() + 10
    while time.time() < deadline and any(Path(f"/proc/{p}").exists() for p in pids):
        time.sleep(0.1)
    assert not any(Path(f"/proc/{p}").exists() for p in pids)  # no orphaned workers


# --- real games, tiny checkpoint ----------------------------------------------


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> str:
    pytest.importorskip("gymnasium")
    pytest.importorskip("sb3_contrib")
    from rl.train import TrainConfig, build_model, build_vec_env

    tmp = tmp_path_factory.mktemp("tiny-search-eval")
    cfg = TrainConfig(
        envs=1, opponents="heuristic", n_steps=64, batch_size=64, n_epochs=1,
        net_arch=[16], torch_threads=1, run_dir=tmp / "run",
    )  # fmt: skip
    vec_env = build_vec_env(cfg)
    path = tmp / "tiny.zip"
    build_model(cfg, vec_env).save(path)
    vec_env.close()
    return str(path)


def _real_job(ckpt: str, seed: int, arm: str = "self") -> se.Job:
    return se.Job(se.ARMS[arm], ckpt, arm, 4, 4, seed, seed, True, False)


def _identity(game: dict[str, Any]) -> tuple[Any, ...]:
    """What must replay exactly: the game record digest, the winner, and every
    search decision except its wall time."""
    decisions = [[d[0], *d[2:]] for d in game.get("decisions", [])]
    return game["e"], game["digest"], game["won"], decisions


def test_a_real_game_replays_identically_in_any_process_state(
    tiny_checkpoint: str,
) -> None:
    """Resume safety: a game is a pure function of its seeds, not of what ran
    before it in the worker (the cached model, encoder, evaluator) nor of hash
    randomization."""
    fresh = se.play_one(_real_job(tiny_checkpoint, 20_000_001))
    se.play_one(_real_job(tiny_checkpoint, 20_000_002))  # dirty the process state
    again = se.play_one(_real_job(tiny_checkpoint, 20_000_001))
    assert _identity(again) == _identity(fresh)
    assert fresh["decisions"], "the self-model arm must have searched something"

    script = (
        "import json; from experiments import search_eval as se; "
        "print(json.dumps(se.play_one(se.Job("
        f"{se.ARMS['self']!r}, {tiny_checkpoint!r}, "
        "'self', 4, 4, 20_000_001, 20_000_001, True, False))))"
    )
    env = {**os.environ, "PYTHONHASHSEED": "12345", "PYTHONPATH": str(se.ROOT)}
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True,
        env=env, cwd=se.ROOT,
    )  # fmt: skip
    other = json.loads(out.stdout.strip().splitlines()[-1])
    assert _identity(other) == _identity(fresh)
