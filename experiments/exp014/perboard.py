"""Exp 014 Stage 2: per-board replay of a benchmarked checkpoint.

Used for the McNemar and per-rotation secondaries. The benchmark JSON keeps only
aggregates. Play is deterministic given (engine_seed, driver_seed), so replaying
the same 4000 fresh_014 boards reproduces the same games; this script saves the
per-board rl-win flag and asserts the total equals the benchmark JSON's rl wins.
Run from the catan-014-run worktree:
    PYTHONPATH=$PWD python perboard.py <label> <workers>
"""

from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path

from gamekit.seats import rotate

from experiments.benchmark import MODE_LINEUPS, benchmark_agent_factory
from experiments.rollout import run_many

R = Path("/home/guido/projects/catan/rl_runs/exp014")
MODE, N, BASE = "rl_vs_heuristic", 4000, 21_000_001


def main() -> None:
    label, workers = sys.argv[1], int(sys.argv[2])
    out = R / "bench" / f"perboard_{label}.json"
    if out.exists():
        print(f"skip {label} (done)")
        return
    ckpt = R / "selfplay" / label / f"{label}_3000000.zip"
    lineup = MODE_LINEUPS[MODE](4)
    pairs = [(BASE + i, BASE + i) for i in range(N)]
    records = run_many(
        4,
        pairs,
        agent_factory=partial(benchmark_agent_factory, MODE, str(ckpt)),
        workers=workers,
    )
    rl_won = [
        int(
            r.winning_seat is not None
            and rotate(lineup, r.engine_seed)[r.winning_seat] == "rl"
        )
        for r in records
    ]
    ref = json.load(
        open(R / "bench" / f"benchmark_rl_vs_heuristic_p4_{label}_3000000.json")
    )
    expect = ref["by_role"]["rl"]["wins"]
    assert sum(rl_won) == expect, f"{label}: replay {sum(rl_won)} != benchmark {expect}"
    out.write_text(
        json.dumps({"label": label, "engine_seed_base": BASE, "rl_won": rl_won})
    )
    print(f"done {label}: {sum(rl_won)} wins == benchmark {expect}")


if __name__ == "__main__":
    main()
