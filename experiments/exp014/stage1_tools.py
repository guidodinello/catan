"""Experiment 014 Stage 1 helpers: sealed-val sweep dataset, lr choice, analysis.

make-sweep-ds : copy rl_runs/bc/v1 without the val games (game_id % 10 == 0), so the
                lr-sweep runs cannot score val at all (their val metrics are NaN).
choose-lr     : per arm, the lr whose seed-0 sweep run has the highest best-epoch
                tune spatial accuracy (read from the run log); ties go to 1e-3.
gate          : Gate S1 from the six final_* run logs. Parses the single val report
                each log prints; never reloads a checkpoint, so val is read once.
check-splits  : sanity checks on the BC dataset's game ids and the split sizes.

``gate`` and ``check-splits`` were added after the runs (they only read logs and
``game_id.npy``); the other two subcommands are the as-run code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

FILES = ("obs", "masks", "labels", "returns", "game_id")
LRS = ("1e-3", "3e-4")
TUNE_RE = re.compile(r"tune_spatial_acc=([0-9.]+)%")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def make_sweep_ds(src: Path, dst: Path) -> None:
    if (dst / "manifest.json").exists():
        print("sweep dataset already built:", dst)
        return
    dst.mkdir(parents=True, exist_ok=True)
    game_id = np.load(src / "game_id.npy", mmap_mode="r")
    keep = np.flatnonzero(np.asarray(game_id) % 10 != 0)
    manifest = {
        "source": str(src),
        "rows_kept": int(len(keep)),
        "rows_total": int(len(game_id)),
    }
    for name in FILES:
        arr = np.load(src / f"{name}.npy", mmap_mode="r")
        np.save(dst / f"{name}.npy", np.asarray(arr[keep]))
        manifest[name] = sha256(dst / f"{name}.npy")
    gid = np.load(dst / "game_id.npy")
    assert (gid % 10 != 0).all()  # no val game in the sweep dataset
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def best_tune(log: Path) -> float:
    text = log.read_text()
    return max(float(m) for m in TUNE_RE.findall(text))


def choose_lr(logdir: Path, out: Path) -> None:
    choice = {}
    for arm in ("mlp", "graph"):
        scores = {lr: best_tune(logdir / f"sweep_{arm}_lr{lr}.log") for lr in LRS}
        # Ties (printed to 0.1 pt) go to 1e-3, 004's lr; disclosed if it happens.
        best = max(LRS, key=lambda lr: (scores[lr], lr == "1e-3"))
        choice[arm] = {
            "lr": best,
            "tune_spatial_acc_pct": scores,
            "tie": len(set(scores.values())) == 1,
        }
    out.write_text(json.dumps(choice, indent=2))
    print(json.dumps(choice, indent=2))


BLOCK_RE = re.compile(r"(\w+)=([0-9.]+|nan)%")
VAL_SPATIAL_RE = re.compile(r"val spatial \(pooled\): ([0-9.]+)%")
VAL_RE = re.compile(
    r"val:\s+raw_acc=([0-9.]+)% masked_acc=([0-9.]+)% value_mse=([0-9.]+)"
)
STOP_RE = re.compile(r"early stop at epoch (\d+)")
EPOCH_RE = re.compile(r"epoch=(\d+)/\d+ .*tune_spatial_acc=([0-9.]+)%")
GATE_MIN_DIFF = 3.0


def parse_final(log: Path) -> dict:
    text = log.read_text()
    if "EXIT=0" not in text:
        raise SystemExit(f"{log} has no EXIT=0")
    blocks_line = next(ln for ln in text.splitlines() if "val by block:" in ln)
    val, spatial = VAL_RE.search(text), VAL_SPATIAL_RE.search(text)
    if val is None or spatial is None:
        raise SystemExit(f"{log} has no val report")
    tune = [(int(e), float(a)) for e, a in EPOCH_RE.findall(text)]
    stop = STOP_RE.search(text)
    return {
        "val_spatial_pct": float(spatial.group(1)),
        "val_masked_acc_pct": float(val.group(2)),
        "val_value_mse": float(val.group(3)),
        "val_by_block_pct": {
            k: None if v == "nan" else float(v)
            for k, v in BLOCK_RE.findall(blocks_line)
        },
        "best_tune_spatial_pct": max(a for _, a in tune),
        "epochs_run": tune[-1][0],
        "early_stop_epoch": int(stop.group(1)) if stop else None,
    }


def gate(logdir: Path, out: Path) -> None:
    arms = {
        arm: [parse_final(logdir / f"final_{arm}_s{s}.log") for s in range(3)]
        for arm in ("mlp", "graph")
    }
    acc = {a: [r["val_spatial_pct"] for r in rs] for a, rs in arms.items()}
    mean = {a: sum(v) / len(v) for a, v in acc.items()}
    diff = mean["graph"] - mean["mlp"]
    c1, c2 = diff >= GATE_MIN_DIFF, min(acc["graph"]) > max(acc["mlp"])
    result = {
        "metric": "val masked top-1 accuracy pooled over vertex_settlement, edge, "
        "hex (%), as printed (0.1-pt precision) by each run's single val report",
        "seeds": arms,
        "mean_val_spatial_pct": mean,
        "mean_diff_graph_minus_flat_pts": diff,
        "clause1_mean_diff_ge_3": c1,
        "clause2_min_graph_gt_max_flat": c2,
        "gate_s1_passed": c1 and c2,
    }
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def check_splits(dataset: Path) -> None:
    g = np.load(dataset / "game_id.npy")
    ids = np.unique(g)
    assert len(ids) == 3000 and (ids == np.arange(3000)).all(), "ids are not 0..2999"
    blocks = int((np.r_[True, g[1:] != g[:-1]]).sum())
    assert blocks == len(ids), "a game's rows are not one contiguous block"
    r = g % 10
    parts = {"train": (r != 0) & (r != 1), "tune": r == 1, "val": r == 0}
    games = {k: set(g[m].tolist()) for k, m in parts.items()}
    for a, b in (("train", "tune"), ("train", "val"), ("tune", "val")):
        assert not games[a] & games[b], f"{a}/{b} share a game_id"
    sizes = {k: int(m.sum()) for k, m in parts.items()}
    assert sizes == {"train": 404464, "tune": 52243, "val": 51893}, sizes
    print(f"ok: {len(g)} rows, {len(ids)} games, contiguous, disjoint, rows {sizes}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-sweep-ds")
    m.add_argument("--src", type=Path, required=True)
    m.add_argument("--dst", type=Path, required=True)
    c = sub.add_parser("choose-lr")
    c.add_argument("--logdir", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    gt = sub.add_parser("gate")
    gt.add_argument("--logdir", type=Path, required=True)
    gt.add_argument("--out", type=Path, required=True)
    cs = sub.add_parser("check-splits")
    cs.add_argument("--dataset", type=Path, required=True)
    a = p.parse_args()
    if a.cmd == "make-sweep-ds":
        make_sweep_ds(a.src, a.dst)
    elif a.cmd == "choose-lr":
        choose_lr(a.logdir, a.out)
    elif a.cmd == "gate":
        gate(a.logdir, a.out)
    else:
        check_splits(a.dataset)
