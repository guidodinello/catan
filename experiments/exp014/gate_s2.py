"""Exp 014 Gate S2, as pre-registered in docs/experiments/014-graph-encoder.md.

Reads the four n=4000 benchmark JSONs and perboard_*.json, writes analysis/gate_s2.json.
Run from the catan-014-run worktree with the frozen venv (PYTHONPATH=$PWD).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from gamekit.mc import wilson_interval
from gamekit.mc.testing import two_proportion_test

from experiments.search_eval import mcnemar_exact

R = Path("/home/guido/projects/catan/rl_runs/exp014")
N = 4000
RUNS = {
    "flat_s3": "mlp_s3",
    "flat_s4": "mlp_s4",
    "graph_s3": "graph_s3",
    "graph_s4": "graph_s4",
}
EPS_PTS = 2.2  # clause 3: the run-to-run spread observed in 010


def label(run: str) -> str:
    return f"catan014_ft_{RUNS[run]}"


def wins(run: str) -> int:
    f = R / "bench" / f"benchmark_rl_vs_heuristic_p4_{label(run)}_3000000.json"
    d = json.loads(f.read_text())
    assert d["n_games"] == N and d["engine_seed_base"] == 21_000_001
    return int(d["by_role"]["rl"]["wins"])


def perboard(run: str) -> list[int]:
    f = R / "bench" / f"perboard_{label(run)}.json"
    return [int(v) for v in json.loads(f.read_text())["rl_won"]]


def curve(run: str) -> list[tuple[int, float, float]]:
    rx = re.compile(
        r"steps=(\d+)/3000000 .*?reported=([\d.]+)%.*?guard\(fixed\)=([\d.]+)%"
    )
    text = (R / "logs" / f"ft_{RUNS[run]}.log").read_text()
    return [(int(s), float(a), float(b)) for s, a, b in rx.findall(text)]


def main() -> None:
    w = {r: wins(r) for r in RUNS}
    rows: dict[str, dict[str, Any]] = {}
    for r, k in w.items():
        lo, hi = wilson_interval(k, N)
        rows[r] = {"wins": k, "n": N, "rate": k / N, "wilson95": [lo, hi]}
    kg, kf = w["graph_s3"] + w["graph_s4"], w["flat_s3"] + w["flat_s4"]
    t = two_proportion_test(kg, 2 * N, kf, 2 * N)
    diff_pts = 100 * (kg / (2 * N) - kf / (2 * N))
    rate = {r: float(rows[r]["rate"]) for r in rows}
    min_graph = min(rate["graph_s3"], rate["graph_s4"])
    max_flat = max(rate["flat_s3"], rate["flat_s4"])
    c1 = t.p_value < 0.05 and diff_pts > 0
    c2 = min_graph > max_flat
    c3 = diff_pts > EPS_PTS
    out = {
        "rows": rows,
        "pooled": {
            "graph": [kg, 2 * N],
            "flat": [kf, 2 * N],
            "diff_pts": diff_pts,
            "test": t._asdict(),
        },
        "clauses": {"1_pooled_test": c1, "2_seed_consistency": c2, "3_gt_2.2pts": c3},
        "gate_s2_passed": c1 and c2 and c3,
    }
    pb = {r: perboard(r) for r in RUNS}
    assert all(sum(pb[r]) == w[r] for r in RUNS)
    mc = {}
    for g in ("graph_s3", "graph_s4"):
        for f in ("flat_s3", "flat_s4"):
            only_g = sum(1 for a, b in zip(pb[g], pb[f], strict=True) if a and not b)
            only_f = sum(1 for a, b in zip(pb[g], pb[f], strict=True) if b and not a)
            mc[f"{g}_vs_{f}"] = {
                "only_graph": only_g,
                "only_flat": only_f,
                "p_exact": mcnemar_exact(only_g, only_f),
            }
    out["mcnemar"] = mc
    # rl role sits at lineup index 0, rotated by engine_seed % 4
    out["per_rotation_wins"] = {
        r: [sum(v for i, v in enumerate(pb[r]) if i % 4 == p) for p in range(4)]
        for r in RUNS
    }
    flat_final = kf / (2 * N)
    out["steps_to_flat_final"] = {
        "flat_final_rate_n4000_pooled": flat_final,
        "first_chunk_reported_ge": {
            g: next((s for s, rep, _ in curve(g) if rep / 100 >= flat_final), None)
            for g in ("graph_s3", "graph_s4")
        },
    }
    (R / "analysis").mkdir(exist_ok=True)
    (R / "analysis" / "gate_s2.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
