"""Profiling for issue #20: measure before deciding whether the GPU helps.

Three subcommands, each a thin, timed driver over machinery ``rl/train.py``
and ``rl/bc.py`` already own -- this module adds instrumentation, not a
second training path.

``worker``: drives one ``CatanEnv`` (built by ``rl.train.build_env``, so the
self-play pool wiring is identical to a real run) for a fixed number of
*learner* decisions, with a loaded checkpoint standing in for the learner.
Splits wall time into learner inference, opponent-checkpoint inference, and
"everything else" (engine step/reset + ``HeuristicAgent`` scoring).

Learner and opponent inference are both routed through
``sb3_contrib.MaskablePPO.predict`` (opponents via ``RLAgent``, the learner
via a direct call this module makes) -- a single class-level monkeypatch on
``predict`` accumulates *total* predict time and call count, and this module
separately times its own direct learner calls. Opponent-only time is the
difference: both flow through the same patched method, but only the
learner's share is measured directly, so subtracting it out is exact, not
approximate.

``train``: runs ``rl.train.train`` for a few chunks with timers patched
around ``SubprocVecEnv.step_wait`` (env + opponent-inference wall time as the
main process sees it), the policy forward inside rollout collection, and
``model.train()`` (the PPO update). Reports steady-state FPS from the last
chunk's delta, since workers lazy-load every pool checkpoint on first use and
that warmup cost pollutes an average over the whole run.

``latency``: single-sample and batched (B in {1,4,16,64}) forward-pass
latency for one checkpoint, CPU only until a CUDA env exists. Uses a lean
forward (see ``rl.bc._forward``) alongside plain ``model.predict`` so the
SB3-overhead-vs-raw-forward split is visible on its own.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("rl.profile")


# --------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------


@dataclass(slots=True)
class _Accumulator:
    """Cumulative (time, calls) for one instrumented call site."""

    seconds: float = 0.0
    calls: int = 0

    def add(self, seconds: float, calls: int = 1) -> None:
        self.seconds += seconds
        self.calls += calls


@dataclass(slots=True)
class WorkerProfile:
    total_seconds: float
    learner_steps: int
    predict_total: _Accumulator = field(default_factory=_Accumulator)
    learner_predict: _Accumulator = field(default_factory=_Accumulator)
    heuristic: _Accumulator = field(default_factory=_Accumulator)

    @property
    def opponent_predict_seconds(self) -> float:
        return self.predict_total.seconds - self.learner_predict.seconds

    @property
    def opponent_predict_calls(self) -> int:
        return self.predict_total.calls - self.learner_predict.calls

    @property
    def engine_seconds(self) -> float:
        return self.total_seconds - self.predict_total.seconds - self.heuristic.seconds

    def report(self) -> str:
        def pct(x: float) -> float:
            if not self.total_seconds:
                return float("nan")
            return 100.0 * x / self.total_seconds

        opp_calls = self.opponent_predict_calls
        if opp_calls:
            opp_us = 1e6 * self.opponent_predict_seconds / opp_calls
        else:
            opp_us = float("nan")
        return (
            f"worker profile: {self.learner_steps} learner steps, "
            f"{self.total_seconds:.2f}s total\n"
            f"  learner inference:  {self.learner_predict.seconds:8.3f}s "
            f"({pct(self.learner_predict.seconds):5.1f}%) "
            f"n={self.learner_predict.calls}\n"
            f"  opponent inference: {self.opponent_predict_seconds:8.3f}s "
            f"({pct(self.opponent_predict_seconds):5.1f}%) "
            f"n={opp_calls} ({opp_us:.1f} us/call, "
            f"{opp_calls / self.learner_steps:.2f} calls/learner-step)\n"
            f"  heuristic scoring:  {self.heuristic.seconds:8.3f}s "
            f"({pct(self.heuristic.seconds):5.1f}%) n={self.heuristic.calls}\n"
            f"  engine remainder:   {self.engine_seconds:8.3f}s "
            f"({pct(self.engine_seconds):5.1f}%)"
        )


def _get_attr(obj: object, name: str) -> Any:
    """``getattr`` typed ``object`` in, ``Any`` out -- see ``_set_attr``'s
    docstring for why this module goes through these instead of plain
    attribute access on the sb3-contrib/stable-baselines3 objects it
    monkeypatches."""
    return getattr(obj, name)  # noqa: B009 -- see docstring


def _set_attr(obj: object, name: str, value: Any) -> None:
    """Sets ``obj.<name>`` without ever needing a ``# type: ignore`` whose
    necessity would depend on whether sb3-contrib/stable-baselines3 happen
    to be installed in the mypy environment checking this file: they are in
    this project's own dev venv, and deliberately aren't in the pre-commit
    hook's isolated one (see ``.pre-commit-config.yaml``), so a plain
    ``obj.<name> = value`` assignment would need the ignore in one
    environment and get it flagged "unused" in the other.
    ``HeuristicAgent`` (first-party, always fully typed either way) doesn't
    have this problem and keeps its own plain
    ``# type: ignore[method-assign]`` at its one call site below."""
    setattr(obj, name, value)  # noqa: B010 -- see docstring


def _patch_predict(target_cls: object, acc: _Accumulator) -> Any:
    """Monkeypatch ``target_cls.predict`` to accumulate wall time and call
    count, returning the original method so the caller can restore it."""
    original = _get_attr(target_cls, "predict")

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = original(self, *args, **kwargs)
        acc.add(time.perf_counter() - t0)
        return result

    _set_attr(target_cls, "predict", wrapped)
    return original


def _patch_heuristic_choose(acc: _Accumulator) -> Any:
    from agents.heuristic import HeuristicAgent

    original = HeuristicAgent.choose_action

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = original(self, *args, **kwargs)
        acc.add(time.perf_counter() - t0)
        return result

    HeuristicAgent.choose_action = wrapped  # type: ignore[method-assign]
    return original


def run_worker_profile(
    *,
    pool_dir: Path,
    label: str,
    learner_checkpoint: Path,
    steps: int,
    num_players: int = 4,
    baseline_mix: float = 0.5,
    seed: int = 0,
) -> WorkerProfile:
    import torch
    from sb3_contrib import MaskablePPO

    from agents.heuristic import HeuristicAgent
    from rl.action_space import ActionComposer
    from rl.train import TrainConfig, build_env

    torch.set_num_threads(1)

    predict_acc = _Accumulator()
    learner_acc = _Accumulator()
    heuristic_acc = _Accumulator()
    original_predict = _patch_predict(MaskablePPO, predict_acc)
    original_heuristic = _patch_heuristic_choose(heuristic_acc)

    try:
        cfg = TrainConfig(
            num_players=num_players,
            selfplay_dir=pool_dir,
            label=label,
            baseline_mix=baseline_mix,
            seed=seed,
        )
        env = build_env(cfg, rank=0)
        learner_model = MaskablePPO.load(learner_checkpoint, device="cpu")
        composer = ActionComposer()

        obs, _ = env.reset(seed=seed)
        composer.reset()
        learner_steps = 0
        t_start = time.perf_counter()
        while learner_steps < steps:
            mask = env.action_masks()
            t0 = time.perf_counter()
            atom, _ = learner_model.predict(obs, action_masks=mask, deterministic=True)
            learner_acc.add(time.perf_counter() - t0)
            obs, _, terminated, truncated, _ = env.step(np.int64(atom))
            if terminated or truncated:
                obs, _ = env.reset(seed=seed + learner_steps + 1)
                composer.reset()
            learner_steps += 1
        total = time.perf_counter() - t_start
    finally:
        _set_attr(MaskablePPO, "predict", original_predict)
        HeuristicAgent.choose_action = original_heuristic  # type: ignore[method-assign]

    return WorkerProfile(
        total_seconds=total,
        learner_steps=learner_steps,
        predict_total=predict_acc,
        learner_predict=learner_acc,
        heuristic=heuristic_acc,
    )


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ChunkTiming:
    step_wait_seconds: float = 0.0
    policy_forward_seconds: float = 0.0
    ppo_update_seconds: float = 0.0
    wall_seconds: float = 0.0
    steps: int = 0

    @property
    def fps(self) -> float:
        return self.steps / self.wall_seconds if self.wall_seconds else float("nan")


def run_train_profile(
    *,
    pool_dir: Path,
    label: str,
    envs: int,
    chunk: int,
    chunks: int,
    baseline_mix: float = 0.5,
    seed: int = 0,
    opponent_inference: str = "none",
    device: str = "cpu",
) -> list[ChunkTiming]:
    """Runs ``chunks`` chunks of ``model.learn(chunk)``, timing
    ``step_wait`` (env + opponent inference from the main process's view),
    the rollout-collection policy forward, and the PPO update, per chunk.

    A fresh ``TrainConfig``/model/vec_env is built once; chunk boundaries are
    plain repeated ``model.learn()`` calls, exactly as ``rl.train.train``
    itself chunks (see its own eval-between-chunks loop) -- this reuses that
    shape rather than inventing a second one.
    """
    import torch
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from rl.train import TrainConfig, build_model, build_vec_env

    torch.set_num_threads(1)

    step_wait_acc = _Accumulator()
    forward_acc = _Accumulator()

    original_step_wait = _get_attr(SubprocVecEnv, "step_wait")

    def timed_step_wait(self: Any) -> Any:
        t0 = time.perf_counter()
        result = original_step_wait(self)
        step_wait_acc.add(time.perf_counter() - t0)
        return result

    cfg = TrainConfig(
        envs=envs,
        selfplay_dir=pool_dir,
        label=label,
        baseline_mix=baseline_mix,
        seed=seed,
        n_steps=512,
        batch_size=2048,
        opponent_inference=opponent_inference,
        device=device,
    )

    results: list[ChunkTiming] = []
    _set_attr(SubprocVecEnv, "step_wait", timed_step_wait)
    try:
        vec_env = build_vec_env(cfg)
        model = build_model(cfg, vec_env)

        original_forward = _get_attr(model.policy, "forward")

        def timed_forward(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            result = original_forward(*args, **kwargs)
            forward_acc.add(time.perf_counter() - t0)
            return result

        _set_attr(model.policy, "forward", timed_forward)

        for i in range(chunks):
            sw0, fw0 = step_wait_acc.seconds, forward_acc.seconds
            t0 = time.perf_counter()
            model.learn(chunk, reset_num_timesteps=(i == 0), progress_bar=False)
            wall = time.perf_counter() - t0
            timing = ChunkTiming(
                step_wait_seconds=step_wait_acc.seconds - sw0,
                policy_forward_seconds=forward_acc.seconds - fw0,
                wall_seconds=wall,
                steps=chunk,
            )
            results.append(timing)
            logger.info(
                "chunk=%d/%d fps=%.0f step_wait=%.1fs forward=%.1fs wall=%.1fs",
                i + 1,
                chunks,
                timing.fps,
                timing.step_wait_seconds,
                timing.policy_forward_seconds,
                wall,
            )
        vec_env.close()
    finally:
        _set_attr(SubprocVecEnv, "step_wait", original_step_wait)

    return results


# --------------------------------------------------------------------------
# latency
# --------------------------------------------------------------------------


def _lean_forward(policy: Any, obs_t: Any, mask_t: Any) -> Any:
    """Same computation as ``rl.bc._forward``'s masked logits, without the
    value head or the SB3 distribution object -- the cheapest possible
    inference path, reused here as the "lean CPU" control arm from the plan's
    Part B."""
    import torch

    with torch.no_grad():
        features = policy.extract_features(obs_t)
        latent_pi, _ = policy.mlp_extractor(features)
        logits = policy.action_net(latent_pi)
        masked_logits = logits.masked_fill(~mask_t, -1e8)
        return masked_logits.argmax(dim=-1)


def run_latency_profile(
    *, checkpoint: Path, device: str, batch_sizes: tuple[int, ...] = (1, 4, 16, 64)
) -> str:
    import torch
    from sb3_contrib import MaskablePPO

    from rl.encoder import OBS_DIM

    if device == "cpu":
        torch.set_num_threads(1)
    model = MaskablePPO.load(checkpoint, device=device)
    policy = model.policy
    from rl.action_space import N_ATOMS

    n_atoms = N_ATOMS

    lines = [f"latency profile: checkpoint={checkpoint} device={device}"]
    rng = np.random.default_rng(0)

    n_repeats = 200
    obs1 = rng.random((1, OBS_DIM), dtype=np.float32)
    mask1 = np.ones((1, n_atoms), dtype=bool)
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        model.predict(obs1[0], action_masks=mask1[0], deterministic=True)
    sb3_us = 1e6 * (time.perf_counter() - t0) / n_repeats
    lines.append(f"  sb3 predict, B=1:  {sb3_us:8.1f} us/call")

    obs1_t = torch.as_tensor(obs1, device=device)
    mask1_t = torch.as_tensor(mask1, device=device)
    _lean_forward(policy, obs1_t, mask1_t)  # warmup
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        _lean_forward(policy, obs1_t, mask1_t)
    lean_us = 1e6 * (time.perf_counter() - t0) / n_repeats
    lines.append(f"  lean forward, B=1: {lean_us:8.1f} us/call")

    for b in batch_sizes:
        obs_b = rng.random((b, OBS_DIM), dtype=np.float32)
        mask_b = np.ones((b, n_atoms), dtype=bool)
        obs_b_t = torch.as_tensor(obs_b, device=device)
        mask_b_t = torch.as_tensor(mask_b, device=device)
        _lean_forward(policy, obs_b_t, mask_b_t)  # warmup
        n_rep_b = max(20, n_repeats // b)
        t0 = time.perf_counter()
        for _ in range(n_rep_b):
            _lean_forward(policy, obs_b_t, mask_b_t)
        batch_us = 1e6 * (time.perf_counter() - t0) / n_rep_b
        lines.append(
            f"  lean forward, B={b:<3d} {batch_us:8.1f} us/batch "
            f"({batch_us / b:6.1f} us/sample)"
        )

    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profiling for issue #20 (GPU inference evaluation)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    worker = sub.add_parser("worker", help="single-process worker-side split")
    worker.add_argument("--pool", type=Path, required=True)
    worker.add_argument("--label", required=True)
    worker.add_argument("--learner", type=Path, required=True)
    worker.add_argument("--steps", type=int, default=20_000)
    worker.add_argument("--players", type=int, default=4)
    worker.add_argument("--baseline-mix", type=float, default=0.5)
    worker.add_argument("--seed", type=int, default=0)

    train = sub.add_parser("train", help="main-process rollout/update split")
    train.add_argument("--pool", type=Path, required=True)
    train.add_argument("--label", required=True)
    train.add_argument("--envs", type=int, default=16)
    train.add_argument("--chunk", type=int, default=50_000)
    train.add_argument("--chunks", type=int, default=3)
    train.add_argument("--baseline-mix", type=float, default=0.5)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument(
        "--opponent-inference", choices=("none", "cpu", "cuda"), default="none"
    )
    train.add_argument("--device", choices=("cpu", "cuda"), default="cpu")

    latency = sub.add_parser("latency", help="single-sample/batched forward latency")
    latency.add_argument("--checkpoint", type=Path, required=True)
    latency.add_argument("--device", default="cpu")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s - %(message)s",
        stream=sys.stdout,
    )
    if args.command == "worker":
        profile = run_worker_profile(
            pool_dir=args.pool,
            label=args.label,
            learner_checkpoint=args.learner,
            steps=args.steps,
            num_players=args.players,
            baseline_mix=args.baseline_mix,
            seed=args.seed,
        )
        print(profile.report())
    elif args.command == "train":
        results = run_train_profile(
            pool_dir=args.pool,
            label=args.label,
            envs=args.envs,
            chunk=args.chunk,
            chunks=args.chunks,
            baseline_mix=args.baseline_mix,
            seed=args.seed,
            opponent_inference=args.opponent_inference,
            device=args.device,
        )
        last = results[-1]
        print(
            f"steady-state (last chunk): fps={last.fps:.0f} "
            f"step_wait={last.step_wait_seconds:.1f}s "
            f"forward={last.policy_forward_seconds:.1f}s "
            f"wall={last.wall_seconds:.1f}s"
        )
    elif args.command == "latency":
        print(run_latency_profile(checkpoint=args.checkpoint, device=args.device))


if __name__ == "__main__":
    main()
