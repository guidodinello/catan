"""Experiment 011 driver: decision-time search vs the same checkpoint without it.

Arms (all vs 3 ``HeuristicAgent``, seat-rotated, the benchmark's boards):

* ``none``  -- plain ``RLAgent`` (the no-search baseline, A0).
* ``heur``  -- ``RLSearchAgent`` whose search simulates opponents with
  ``HeuristicAgent`` (a known opponent model: an upper bound, A1).
* ``self``  -- ``RLSearchAgent`` whose search simulates opponents with the
  checkpoint's own greedy policy (the transferable number, A2).

Games run in chunks of ``--chunk`` seeds; each finished chunk is written
atomically to ``--out-dir/<label>/chunk_<base>.json`` so a laptop shutdown
loses at most one chunk and ``--resume`` continues. Every game is seeded by
its ``(engine_seed, driver_seed)`` pair, so the numbers do not depend on the
worker count or on where a run was resumed.

``--timing-only`` (the smoke run) records latency, throughput, memory and
search diagnostics and **never records or prints who won**.

Run from the worktree with the main checkout's interpreter:
    PYTHONPATH=$PWD OMP_NUM_THREADS=1 /path/to/main/.venv/bin/python \\
        -m experiments.search_eval run --arm heur --sims 64 --games 4000 ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gamekit.mc import wilson_interval
from gamekit.mc.testing import two_proportion_test
from gamekit.seats import rotate

from agents.ismcts import SearchConfig
from experiments.benchmark import MODE_LINEUPS, benchmark_agent_factory
from experiments.rollout import run_game
from rl.action_space import SIMPLE_END_TURN

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_OUT = ROOT / "rl_runs" / "search_eval"

# CPU cap: 10 while experiment #40 trained, 18 after it finished (scheduling
# only; results are seeded and ordered by seed).
MAX_WORKERS = 18

ARMS: dict[str, str] = {
    "none": "rl_vs_heuristic",
    "heur": "rl_search_vs_heuristic",
    "self": "rl_search_vs_heuristic",
}
OPPONENT = {"heur": "heuristic", "self": "self"}

# Consecutive seeds, length divisible by 4 so the rotation is exact. The smoke
# range is never measured; 1.. are the benchmark boards of 004/006/008/010.
SEED_RANGES: dict[str, tuple[int, int]] = {
    "smoke": (4_000_001, 400),
    "measured": (1, 4000),
}
MIN_EDGE_N = 4  # edges with fewer leaf samples are left out of the noise check


@dataclass(frozen=True, slots=True)
class Job:
    mode: str
    checkpoint: str
    arm: str
    sims: int
    num_players: int
    engine_seed: int
    driver_seed: int
    record_outcome: bool
    record_noise: bool


def module_paths() -> dict[str, str]:
    import agents
    import engine
    import experiments
    import rl

    return {
        m.__name__: str(Path(str(m.__file__)).resolve().parent)
        for m in (agents, engine, rl, experiments)
    }


def _search_config(job: Job) -> SearchConfig | None:
    if job.arm == "none":
        return None
    return SearchConfig(
        simulations=job.sims,
        opponent=OPPONENT[job.arm],
        record_edge_stats=job.record_noise,
    )


def _noise(edge_stats: Sequence[tuple[Any, int, float, float]]) -> list[float | None]:
    """[SD across determinizations at EndTurn leaves, SD of sibling edge means]."""
    big = [e for e in edge_stats if e[1] >= MIN_EDGE_N]
    end = [e[3] for e in big if tuple(e[0]) == (SIMPLE_END_TURN,)]
    sib = [e[2] for e in big]
    return [
        end[0] if end else None,
        statistics.stdev(sib) if len(sib) >= 2 else None,
    ]


def play_one(job: Job) -> dict[str, Any]:
    """Pool worker: one game through the benchmark's own agent factory."""
    holder: list[Any] = []
    cfg = _search_config(job)

    def factory(n: int, e: int, d: int) -> list[Any]:
        agents = benchmark_agent_factory(job.mode, job.checkpoint, n, e, d, cfg)
        holder.extend(agents)
        return agents

    t0 = time.perf_counter()
    record = run_game(job.num_players, job.engine_seed, job.driver_seed, factory)
    out: dict[str, Any] = {
        "e": job.engine_seed,
        "worker_s": time.perf_counter() - t0,
        "rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "module_paths": module_paths(),
    }
    if job.record_outcome:
        lineup = MODE_LINEUPS[job.mode](job.num_players)
        rl_seat = rotate(lineup, job.engine_seed).index(lineup[0])
        out["won"] = int(record.winning_seat == rl_seat)
    searcher = next((a for a in holder if hasattr(a, "log")), None)
    if searcher is not None:
        out["decisions"] = [
            [d.phase, round(d.ms, 3), d.simulations, int(d.overrode)]
            + ([_noise(d.edge_stats)] if job.record_noise else [])
            for d in searcher.log.decisions
        ]
    return out


def _sha12(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _check_paths(samples: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    paths = {tuple(sorted(s["module_paths"].items())) for s in samples}
    bad = [p for tup in paths for _, p in tup if not Path(p).is_relative_to(ROOT)]
    if bad:
        raise RuntimeError(
            f"workers imported modules outside {ROOT}: {sorted(set(bad))}"
        )
    return [dict(t) for t in paths]


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def _map_jobs(jobs: Sequence[Job], workers: int) -> list[dict[str, Any]]:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(play_one, jobs, chunksize=2))  # ordered by seed


def run(args: argparse.Namespace) -> None:
    if args.workers > MAX_WORKERS:
        raise SystemExit(f"--workers {args.workers} exceeds MAX_WORKERS={MAX_WORKERS}")
    if args.games % args.chunk or args.chunk % 4:
        raise SystemExit("--games must be a multiple of --chunk, itself of 4")
    measured = not args.timing_only
    commit, dirty = _git("rev-parse", "HEAD"), bool(_git("status", "--porcelain"))
    if measured and dirty:
        raise SystemExit("refusing a measured run from a dirty worktree")
    kind = "measured" if measured else "smoke"
    lo, n_range = SEED_RANGES[kind]
    if args.seed_base < lo or args.seed_base + args.games > lo + n_range:
        raise SystemExit(f"seeds outside the {kind} range {lo}..{lo + n_range - 1}")
    checkpoint = str(Path(args.checkpoint).resolve())
    sha = _sha12(checkpoint)
    config = {
        "arm": args.arm,
        "sims": args.sims if args.arm != "none" else 0,
        "mode": ARMS[args.arm],
        "checkpoint_sha256_12": sha,
        "timing_only": args.timing_only,
    }
    chash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    label = args.label or f"{args.arm}" + (
        f"_s{args.sims}" if args.arm != "none" else ""
    )
    out_dir = Path(args.out_dir) / label
    for base in range(args.seed_base, args.seed_base + args.games, args.chunk):
        path = out_dir / f"chunk_{base}.json"
        if path.exists():
            prev = json.loads(path.read_text())
            if prev["config_hash"] != chash or prev["git_commit"] != commit:
                if not args.allow_commit_change or prev["config_hash"] != chash:
                    raise SystemExit(f"{path} was written by a different config/commit")
            print(f"chunk {base}: present, skipped", flush=True)
            continue
        jobs = [
            Job(
                ARMS[args.arm],
                checkpoint,
                args.arm,
                args.sims,
                4,
                e,
                e,
                record_outcome=measured,
                record_noise=args.timing_only,
            )
            for e in range(base, base + args.chunk)
        ]
        t0 = time.perf_counter()
        samples = _map_jobs(jobs, args.workers)
        wall = time.perf_counter() - t0
        payload = {
            "config": config,
            "config_hash": chash,
            "git_commit": commit,
            "workers": args.workers,
            "seed_base": base,
            "n": args.chunk,
            "wall_s": wall,
            "module_paths": _check_paths(samples),
            "games": [
                {k: v for k, v in s.items() if k != "module_paths"} for s in samples
            ],
        }
        _write_atomic(path, payload)
        print(f"chunk {base}: {args.chunk} games in {wall:.0f}s -> {path}", flush=True)


# --- Analysis ----------------------------------------------------------------


def load_arm(arm_dir: Path) -> list[dict[str, Any]]:
    """All games of an arm, in seed order."""
    games: list[dict[str, Any]] = []
    for f in sorted(arm_dir.glob("chunk_*.json"), key=lambda p: int(p.stem[6:])):
        games.extend(json.loads(f.read_text())["games"])
    return sorted(games, key=lambda g: g["e"])


def _pct(xs: Sequence[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def latency_summary(games: Sequence[dict[str, Any]]) -> dict[str, Any]:
    decisions = [d for g in games for d in g.get("decisions", [])]
    searched = [d for d in decisions if d[2] > 0]
    if not searched:
        return {}
    ms = [d[1] for d in searched]
    by_phase: dict[str, list[list[Any]]] = {}
    for d in searched:
        by_phase.setdefault(d[0], []).append(d)
    out: dict[str, Any] = {
        "decisions": len(decisions),
        "searched": len(searched),
        "skip_rate": 1 - len(searched) / len(decisions),
        "ms_mean": statistics.fmean(ms),
        "ms_p95": _pct(ms, 0.95),
        "ms_max": max(ms),
        "sims_per_s": sum(d[2] for d in searched) / (sum(ms) / 1000),
        "override_rate": sum(d[3] for d in searched) / len(searched),
        "override_by_phase": {
            p: [len(v), sum(d[3] for d in v) / len(v)]
            for p, v in sorted(by_phase.items())
        },
        "worker_s_per_game": statistics.fmean(g["worker_s"] for g in games),
        "peak_rss_mb": max(g["rss_mb"] for g in games),
    }
    noise = [d[4] for d in searched if len(d) > 4]
    end = [n[0] for n in noise if n[0] is not None]
    sib = [n[1] for n in noise if n[1] is not None]
    if end and sib:
        out["leaf_noise"] = {
            "endturn_sd_mean": statistics.fmean(end),
            "sibling_mean_sd_mean": statistics.fmean(sib),
            "n_endturn": len(end),
            "n_sibling": len(sib),
        }
    return out


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the discordant-pair counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / (1 << n)
    return float(min(1.0, 2 * tail))


def compare(a: Sequence[dict[str, Any]], b: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Arm ``a`` (search) vs arm ``b`` (baseline) on their common boards."""
    common = sorted({g["e"] for g in a} & {g["e"] for g in b})
    wa = {g["e"]: g["won"] for g in a}
    wb = {g["e"]: g["won"] for g in b}
    ka, kb, n = sum(wa[e] for e in common), sum(wb[e] for e in common), len(common)
    t = two_proportion_test(ka, n, kb, n)
    return {
        "n": n,
        "wins_a": ka,
        "wins_b": kb,
        "rate_a": ka / n,
        "rate_b": kb / n,
        "diff": (ka - kb) / n,
        "z": t.z,
        "p_value": t.p_value,
        "mcnemar_p": mcnemar_exact(
            sum(1 for e in common if wa[e] and not wb[e]),
            sum(1 for e in common if wb[e] and not wa[e]),
        ),
        "only_a": sum(1 for e in common if wa[e] and not wb[e]),
        "only_b": sum(1 for e in common if wb[e] and not wa[e]),
    }


def arm_summary(games: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(games)}
    if games and "won" in games[0]:
        k = sum(g["won"] for g in games)
        ci = wilson_interval(k, len(games))
        out |= {"wins": k, "win_rate": k / len(games), "wilson95": [ci.lower, ci.upper]}
    out["latency"] = latency_summary(games)
    return out


def analyze(args: argparse.Namespace) -> None:
    root = Path(args.out_dir)
    arms = {name: load_arm(root / name) for name in args.arm_dirs}
    report: dict[str, Any] = {n: arm_summary(g) for n, g in arms.items()}
    if args.baseline:
        base = arms[args.baseline]
        report["vs_baseline"] = {
            n: compare(g, base) for n, g in arms.items() if n != args.baseline
        }
    print(json.dumps(report, indent=2))
    if args.write:
        RESULTS_DIR.mkdir(exist_ok=True)
        path = RESULTS_DIR / f"{args.write}.json"
        path.write_text(json.dumps(report, indent=2))
        print(f"wrote {path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", choices=sorted(ARMS), required=True)
    r.add_argument("--sims", type=int, default=64)
    r.add_argument("--games", type=int, required=True)
    r.add_argument("--seed-base", type=int, default=1)
    r.add_argument("--chunk", type=int, default=200)
    r.add_argument("--workers", type=int, default=10)
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--out-dir", default=str(DEFAULT_OUT))
    r.add_argument("--label", default=None)
    r.add_argument("--timing-only", action="store_true")
    r.add_argument("--allow-commit-change", action="store_true")
    a = sub.add_parser("analyze")
    a.add_argument("arm_dirs", nargs="+")
    a.add_argument("--baseline", default=None)
    a.add_argument("--out-dir", default=str(DEFAULT_OUT))
    a.add_argument("--write", default=None)
    args = parser.parse_args()
    run(args) if args.cmd == "run" else analyze(args)


if __name__ == "__main__":
    main()
