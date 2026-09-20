"""MaskablePPO training against fixed opponents.

Run:
    uv run python -m rl.train --steps 2000000 --envs 16 --opponents random

Design notes worth keeping in view:

**gamma defaults to 0.999, not 0.99.** A 4-player episode is ~450 learner
decisions and ~1580 engine steps (measured). truco-py's 0.99 would put the
effective horizon around 100 steps, leaving a terminal win signal nearly
invisible at the opening placements. Paired with ``ShapedReward``'s
potential-based term, which is policy-invariant and so cannot distort the
optimum.

**Evaluation runs inside the loop, not after it.** Training proceeds in
chunks and evaluates between them, logging a win rate with a Wilson interval
every time. truco-py had no in-training eval and lost two multi-hour runs to a
collapse nobody saw until afterwards; this is the cheap fix for that.

sb3-contrib and torch are imported inside the functions that need them, so
this module stays importable with only the ``rl`` extra. That is what lets CI
unit-test ``TrainConfig`` and the argument parser without a training stack --
worth doing, because the first bug here was an argparse default silently
resolving to a ``member_descriptor`` (``@dataclass(slots=True)`` replaces
class-level attributes with slot descriptors, so ``TrainConfig.gamma`` is
*not* the default value; ``DEFAULTS.gamma`` is).
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agents.heuristic import HeuristicAgent
from agents.random_agent import RandomAgent
from rl.env import CatanEnv
from rl.evaluate import WinRate, evaluate_winrate, masked_ppo_predictor
from rl.reward import ShapedReward

LOG_DIR = Path("rl_runs")
OPPONENT_KINDS = ("random", "heuristic")

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sb3_contrib import MaskablePPO

logger = logging.getLogger("rl.train")


@dataclass(slots=True)
class TrainConfig:
    """Everything one run needs. A dataclass rather than 13 parameters."""

    steps: int = 2_000_000
    envs: int = 16
    opponents: str = "random"
    num_players: int = 4
    seed: int = 0
    gamma: float = 0.999
    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 2048
    n_epochs: int = 4
    ent_coef: float = 0.01
    clip_range: float = 0.2
    gae_lambda: float = 0.95
    net_arch: list[int] = field(default_factory=lambda: [256, 256])
    eval_every: int = 250_000
    eval_episodes: int = 200
    torch_threads: int = 1
    label: str = "catan_ppo"
    run_dir: Path = LOG_DIR
    resume: Path | None = None


def build_opponent(kind: str, rng: random.Random) -> RandomAgent | HeuristicAgent:
    """Mirrors ``experiments/benchmark.py:_build_role`` and
    ``server/bots.py:build_agent`` -- one vocabulary of opponent names."""
    if kind == "random":
        return RandomAgent(rng, name="random")
    if kind == "heuristic":
        return HeuristicAgent(name="heuristic")
    raise ValueError(
        f"unknown opponent kind {kind!r}, expected one of {OPPONENT_KINDS}"
    )


def _action_mask_fn(env: Any) -> Any:
    """``ActionMasker``'s hook. Typed loosely on purpose: sb3-contrib hands it
    a ``gym.Env[Any, Any]``, so a ``CatanEnv`` annotation would be a lie that
    only mypy notices."""
    return env.action_masks()


def make_single_env(cfg: TrainConfig, rank: int) -> Any:
    """One masked env. Module-level so ``SubprocVecEnv`` can pickle it."""
    from sb3_contrib.common.wrappers import ActionMasker

    rng = random.Random(cfg.seed * 7919 + rank)
    agents = [
        build_opponent(cfg.opponents, random.Random(rng.randint(0, 2**31 - 1)))
        for _ in range(cfg.num_players - 1)
    ]
    env = CatanEnv(
        num_players=cfg.num_players,
        agents=agents,
        reward=ShapedReward(gamma=cfg.gamma),
        seed=cfg.seed * 104_729 + rank,
    )
    return ActionMasker(env, _action_mask_fn)


def build_vec_env(cfg: TrainConfig) -> Any:
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        SubprocVecEnv,
        VecMonitor,
    )

    # partial, not a lambda with a default argument: picklable by name, which
    # is what SubprocVecEnv's worker processes need.
    env_fns: list[Callable[[], Any]] = [
        partial(make_single_env, cfg, rank) for rank in range(cfg.envs)
    ]
    if cfg.envs == 1:
        return VecMonitor(DummyVecEnv(env_fns))
    return VecMonitor(SubprocVecEnv(env_fns, start_method="fork"))


def evaluate(cfg: TrainConfig, model: MaskablePPO, opponents: str) -> WinRate:
    return evaluate_winrate(
        masked_ppo_predictor(model),
        opponent_factory=lambda rng: build_opponent(opponents, rng),
        num_players=cfg.num_players,
        n_episodes=cfg.eval_episodes,
        seed=cfg.seed + 977,
    )


def train(cfg: TrainConfig) -> Path:
    import torch
    from sb3_contrib import MaskablePPO

    # One thread per process. With `envs` subprocesses already saturating the
    # box, torch's default intra-op pool oversubscribes the cores and the
    # policy forward pass -- tiny, on a [256, 256] MLP -- gets slower, not
    # faster. This is the single biggest throughput knob on CPU.
    torch.set_num_threads(cfg.torch_threads)

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    vec_env = build_vec_env(cfg)

    if cfg.resume is not None:
        logger.info("resuming from %s", cfg.resume)
        model = MaskablePPO.load(cfg.resume, env=vec_env, device="cpu")
    else:
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            learning_rate=cfg.learning_rate,
            ent_coef=cfg.ent_coef,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_range,
            verbose=0,
            tensorboard_log=str(cfg.run_dir / "tb"),
            seed=cfg.seed,
            device="cpu",
            policy_kwargs={"net_arch": cfg.net_arch},
        )

    chunk = min(cfg.eval_every, cfg.steps)
    done = 0
    # Training and evaluation are timed separately. Folding eval into the FPS
    # number understates throughput badly -- in-loop eval is single-process and
    # steps the policy one observation at a time, so it costs far more per step
    # than a 16-way vectorised rollout does.
    train_seconds = 0.0
    eval_seconds = 0.0
    final_path = cfg.run_dir / f"{cfg.label}_final.zip"

    while done < cfg.steps:
        this_chunk = min(chunk, cfg.steps - done)
        t0 = time.perf_counter()
        model.learn(this_chunk, reset_num_timesteps=(done == 0), progress_bar=False)
        train_seconds += time.perf_counter() - t0
        done += this_chunk
        checkpoint = cfg.run_dir / f"{cfg.label}_{done}.zip"
        model.save(checkpoint)

        t0 = time.perf_counter()
        win_rate = evaluate(cfg, model, cfg.opponents)
        eval_seconds += time.perf_counter() - t0

        logger.info(
            "steps=%d/%d train_fps=%.0f train=%.1fmin eval=%.1fmin vs_%s=%s -> %s",
            done,
            cfg.steps,
            done / train_seconds,
            train_seconds / 60,
            eval_seconds / 60,
            cfg.opponents,
            win_rate,
            checkpoint.name,
        )

    model.save(final_path)
    vec_env.close()
    logger.info("saved %s", final_path)
    return final_path


def build_parser() -> argparse.ArgumentParser:
    """Defaults come from a ``TrainConfig`` *instance*.

    Reading them off the class would silently yield ``member_descriptor``
    objects, because ``@dataclass(slots=True)`` replaces class attributes with
    slot descriptors -- a mistake argparse accepts happily and that only
    surfaces mid-rollout as ``unsupported operand type(s) for *``.
    """
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train MaskablePPO on catan.")
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--envs", type=int, default=defaults.envs)
    parser.add_argument("--opponents", choices=OPPONENT_KINDS, default="random")
    parser.add_argument("--players", type=int, default=defaults.num_players)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--ent-coef", type=float, default=defaults.ent_coef)
    parser.add_argument("--eval-every", type=int, default=defaults.eval_every)
    parser.add_argument("--eval-episodes", type=int, default=defaults.eval_episodes)
    parser.add_argument("--torch-threads", type=int, default=defaults.torch_threads)
    parser.add_argument("--label", default=defaults.label)
    parser.add_argument("--run-dir", type=Path, default=defaults.run_dir)
    parser.add_argument("--resume", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s - %(message)s",
        stream=sys.stdout,
    )
    cfg = TrainConfig(
        steps=args.steps,
        envs=args.envs,
        opponents=args.opponents,
        num_players=args.players,
        seed=args.seed,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        torch_threads=args.torch_threads,
        label=args.label,
        run_dir=args.run_dir,
        resume=args.resume,
    )
    train(cfg)


if __name__ == "__main__":
    main()
