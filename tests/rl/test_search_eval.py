"""Experiment 011's driver (``experiments/search_eval.py``): seed hygiene,
chunked resume, and the paired statistics. No games are played here -- the
per-game worker is stubbed."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from experiments import critic_calibration as cc
from experiments import search_eval as se


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def test_smoke_seeds_are_disjoint_from_every_used_range() -> None:
    lo, n = se.SEED_RANGES["smoke"]
    smoke = (lo, lo + n - 1)
    used = [(s, s + m - 1) for s, m in cc.SEED_RANGES.values()] + list(cc.OFF_LIMITS)
    assert not any(_overlaps(smoke, u) for u in used)
    assert n % 4 == 0


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


def _fake_play(job: se.Job) -> dict[str, Any]:
    return {
        "e": job.engine_seed,
        "worker_s": 0.1,
        "rss_mb": 1.0,
        "module_paths": {"agents": str(se.ROOT / "agents")},
        "won": job.engine_seed % 3 == 0,
    }


def _args(tmp: Path, games: int, ckpt: Path) -> argparse.Namespace:
    return argparse.Namespace(
        arm="heur", sims=8, games=games, seed_base=1, chunk=8, workers=1,
        checkpoint=str(ckpt), out_dir=str(tmp), label="t", timing_only=False,
        allow_commit_change=False,
    )  # fmt: skip


def test_resume_reproduces_an_uninterrupted_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ckpt = tmp_path / "ck.zip"
    ckpt.write_bytes(b"x")
    calls: list[int] = []

    def fake_map(jobs: Any, workers: int) -> list[dict[str, Any]]:
        calls.append(len(jobs))
        return [_fake_play(j) for j in jobs]

    monkeypatch.setattr(se, "_map_jobs", fake_map)
    monkeypatch.setattr(se, "_git", lambda *a: "" if a[0] == "status" else "abc123")

    se.run(_args(tmp_path / "full", 24, ckpt))
    full = se.load_arm(tmp_path / "full" / "t")

    part = tmp_path / "part"
    se.run(_args(part, 24, ckpt))
    (part / "t" / "chunk_9.json").unlink()  # a shutdown lost the middle chunk
    calls.clear()
    se.run(_args(part, 24, ckpt))
    assert calls == [8]  # only the lost chunk was replayed
    assert se.load_arm(part / "t") == full
    assert [g["e"] for g in full] == list(range(1, 25))


def test_resume_refuses_a_different_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ckpt = tmp_path / "ck.zip"
    ckpt.write_bytes(b"x")
    monkeypatch.setattr(se, "_map_jobs", lambda jobs, w: [_fake_play(j) for j in jobs])
    monkeypatch.setattr(se, "_git", lambda *a: "" if a[0] == "status" else "abc123")
    se.run(_args(tmp_path, 8, ckpt))
    changed = _args(tmp_path, 8, ckpt)
    changed.sims = 16
    with pytest.raises(SystemExit):
        se.run(changed)


def test_measured_run_refuses_a_dirty_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ckpt = tmp_path / "ck.zip"
    ckpt.write_bytes(b"x")
    monkeypatch.setattr(
        se, "_git", lambda *a: "M file" if a[0] == "status" else "abc123"
    )
    with pytest.raises(SystemExit):
        se.run(_args(tmp_path, 8, ckpt))
