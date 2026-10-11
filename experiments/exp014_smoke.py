"""Experiment 014's cheap smoke / timing check.

See ``docs/experiments/014-graph-encoder.md``.

Reads **no win rate**. On a throwaway BC dataset (smoke-block seeds, never
``rl_runs/bc/v1``) it times, for the flat and the graph policy:

* BC seconds per epoch on the GPU (a 1-epoch and a 3-epoch run after an untimed
  warm-up; the difference is two epochs, so model build, warm-up and the final
  report cancel out);
* CPU single-sample inference latency (``lean_predict_batch`` and
  ``MaskablePPO.predict``), because self-play opponents and the in-loop eval run
  on CPU;
* a short PPO fine-tune in the experiment's shape (BC-initialised, self-play pool
  seeded with the clone, ``--envs 8``): wall time and steps per second;
* parameter counts.

Results go to ``<out>/smoke.json``; run it in a named tmux session with a log
file under the same directory (see the log's Run protocol).
"""

from __future__ import annotations

import argparse
import json
import logging
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from rl.bc import VALIDATION_MOD, generate, load_dataset, train_bc
from rl.train import TrainConfig, train

logger = logging.getLogger("exp014_smoke")

ARMS = ("mlp", "graph")
# Inside the reserved 4,000,001+ block, past 011's smoke (4,000,001-400) and 012's.
SMOKE_SEED_BASE = 4_000_501
FULL_TRAIN_ROWS = 456_707  # log 004's train split, for the per-epoch extrapolation
LATENCY_SAMPLES = 200
WARMUP_CALLS = 20


def _param_counts(policy: Any) -> dict[str, int]:
    ext = policy.mlp_extractor
    pi_trunk = ext.pi_trunk if hasattr(ext, "pi_trunk") else ext.policy_net
    pi = sum(p.numel() for p in pi_trunk.parameters())
    pi += sum(p.numel() for p in policy.action_net.parameters())
    total = sum(p.numel() for p in policy.parameters())
    return {"total": total, "pi_side": pi, "vf_side": total - pi}


def _timed_bc(dataset: Any, arm: str, save: Path, epochs: int, args: Any) -> float:
    start = time.perf_counter()
    train_bc(
        dataset,
        save=save,
        epochs=epochs,
        patience=epochs + 1,
        batch_size=1024,
        lr=1e-3,
        seed=0,
        device="cuda",
        policy_name=arm,
        gnn_dim=args.gnn_dim,
        gnn_layers=args.gnn_layers,
        select_split="tune",
    )
    return time.perf_counter() - start


def _cpu_latency(clone: Path, dataset: Any) -> dict[str, Any]:
    import torch
    from sb3_contrib import MaskablePPO

    from agents.rl_agent import lean_predict_batch

    torch.set_num_threads(1)
    model = MaskablePPO.load(clone, device="cpu")
    model.policy.eval()
    rng = np.random.default_rng(0)
    rows = rng.choice(dataset.n, size=LATENCY_SAMPLES + WARMUP_CALLS, replace=False)
    obs = np.asarray(dataset.obs[np.sort(rows)], dtype=np.float32)
    masks = np.asarray(dataset.masks[np.sort(rows)], dtype=bool)

    def time_it(call: Any) -> list[float]:
        ms = []
        for i in range(len(rows)):
            t0 = time.perf_counter()
            call(i)
            ms.append((time.perf_counter() - t0) * 1000)
        return ms[WARMUP_CALLS:]

    lean = time_it(
        lambda i: lean_predict_batch(
            model.policy, obs[i : i + 1], masks[i : i + 1], "cpu"
        )
    )
    predict = time_it(
        lambda i: model.predict(obs[i], action_masks=masks[i], deterministic=True)
    )

    def stats(ms: list[float]) -> dict[str, float]:
        ordered = sorted(ms)
        return {
            "mean_ms": statistics.fmean(ms),
            "median_ms": statistics.median(ms),
            "p95_ms": ordered[int(0.95 * len(ordered)) - 1],
        }

    return {"lean_predict_batch": stats(lean), "model_predict": stats(predict)}


def _ppo(arm: str, clone: Path, out: Path, args: Any) -> dict[str, Any]:
    cfg = TrainConfig(
        steps=args.ppo_steps,
        envs=args.envs,
        opponents="heuristic",
        eval_opponents="heuristic",
        seed=3,
        learning_rate=1e-4,
        ent_coef=0.01,
        n_steps=512,
        eval_every=args.ppo_steps,
        eval_episodes=args.eval_episodes,
        label=f"smoke_{arm}",
        run_dir=out / f"run_{arm}",
        bc_init=clone,
        init_rate=0.10,
        selfplay_dir=out / "selfplay",
        baseline_mix=0.5,
        regression_patience=1000,
        device="cuda",
    )
    start = time.perf_counter()
    result = train(cfg)
    wall = time.perf_counter() - start
    return {
        "wall_s": wall,
        "steps": result.steps_completed,
        "steps_per_s_incl_eval": result.steps_completed / wall,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ppo-steps", type=int, default=16_384)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--gnn-dim", type=int, default=64)
    parser.add_argument("--gnn-layers", type=int, default=3)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s - %(message)s",
        stream=sys.stdout,
    )

    result_path = args.out / "smoke.json"
    if result_path.exists():
        raise SystemExit(f"{result_path} exists; use a fresh --out directory")
    args.out.mkdir(parents=True, exist_ok=True)

    dataset_dir = args.out / "ds"
    generate(
        games=args.games,
        out_dir=dataset_dir,
        workers=args.workers,
        seed_base=SMOKE_SEED_BASE,
    )
    dataset = load_dataset(dataset_dir, mmap=False)
    train_rows = int((dataset.game_id % VALIDATION_MOD > 1).sum())
    logger.info("smoke dataset: %d rows, %d train rows", dataset.n, train_rows)

    report: dict[str, Any] = {
        "dataset": {"games": args.games, "rows": dataset.n, "train_rows": train_rows},
        "arms": {},
    }
    for arm in ARMS:
        clone = args.out / f"{arm}_clone.zip"
        _timed_bc(dataset, arm, args.out / f"{arm}_warmup.zip", 1, args)  # untimed
        t1 = _timed_bc(dataset, arm, args.out / f"{arm}_t1.zip", 1, args)
        t3 = _timed_bc(dataset, arm, clone, 3, args)
        per_epoch = (t3 - t1) / 2
        logger.info("%s: BC %.2fs/epoch on %d train rows", arm, per_epoch, train_rows)

        import torch
        from sb3_contrib import MaskablePPO

        policy = MaskablePPO.load(clone, device="cpu").policy
        entry = {
            "params": _param_counts(policy),
            "bc_s_per_epoch": per_epoch,
            "bc_s_per_epoch_full_dataset": per_epoch / train_rows * FULL_TRAIN_ROWS,
            "cpu_latency": _cpu_latency(clone, dataset),
            "torch": torch.__version__,
        }
        logger.info("%s: %s", arm, json.dumps(entry["cpu_latency"]))
        entry["ppo"] = _ppo(arm, clone, args.out, args)
        entry["max_rss_children_mb"] = (
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
        )
        report["arms"][arm] = entry
        logger.info("%s: %s", arm, json.dumps(entry["ppo"]))

    result_path.write_text(json.dumps(report, indent=2))
    logger.info("wrote %s", result_path)


if __name__ == "__main__":
    main()
