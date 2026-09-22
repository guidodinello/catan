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

**Self-play (``--selfplay-dir``) and automatic-stop are joined at the hip.**
Training opponents come from a ``gamekit.rl.selfplay.OpponentPool`` instead of
a fixed lineup, sampling past checkpoints mixed with a fixed ``HeuristicAgent``
floor (``--baseline-mix``, default 0.2 -- matching gamekit's own default;
truco-py's own postmortems show neither 0.2 nor 0.5 alone prevented collapse,
so the floor's exact value matters less than having ``RegressionGuard`` (in
``rl/evaluate.py``) actually watching the eval curve and stopping before hours
are wasted past a collapse, which is what neither of truco's runs had). Eval
always measures against ``--eval-opponents`` (defaults to ``--opponents`` when
unset, so plain training is unaffected) -- in self-play mode this should be
set explicitly to ``heuristic``, since that's the actual Phase 5 gate,
independent of whatever the training-side pool happens to contain at a given
point.

**The pool is always run-scoped, not opt-in.** ``OpponentPool`` gained a
``run_id`` argument in gamekit 0.3.0 (gamekit#23) specifically because an
unscoped pool globs its whole directory forever, so a checkpoint left over
from an earlier or collapsed run stays sample-able indefinitely -- exactly
how truco-py kept drawing a collapsed ``truco_selfplay_final.zip`` long after
that run ended. This module always passes ``run_id=cfg.label`` rather than
exposing a separate flag someone could forget to set; there is no unscoped
self-play path here. Checkpoints are written to ``pool.checkpoint_dir``
(gamekit's own resolved path, read from a throwaway pool instance -- never
re-derived here, since gamekit owns that arithmetic, not catan) rather than
anywhere catan computes independently.

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
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gamekit.rl.selfplay import OpponentPool

from agents.heuristic import HeuristicAgent
from agents.random_agent import RandomAgent
from agents.rl_agent import RLAgent
from engine.actions import Action
from engine.state import GameState
from rl.env import CatanEnv
from rl.evaluate import RegressionGuard, WinRate, evaluate_winrate, masked_ppo_predictor
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

    # Self-play. `opponents` above stops governing training opponents once
    # `selfplay_dir` is set (they come from the pool instead) but keeps
    # governing eval, unless `eval_opponents` overrides it -- see `evaluate()`.
    selfplay_dir: Path | None = None
    baseline_mix: float = 0.2
    eval_opponents: str | None = None

    # A warm-started *policy* rather than a resumed *run*: loads BC-clone
    # weights but keeps step counting at 0 (unlike `resume`, which restores
    # the model's own cumulative `num_timesteps`). Mutually exclusive with
    # `resume` -- see `build_parser`.
    bc_init: Path | None = None
    # The BC clone's own win rate (e.g. from `rl.bc`'s n=200 sanity eval),
    # used only to seed `RegressionGuard`'s baseline -- see `train()`.
    bc_init_rate: float = -1.0

    # Automatic stop-on-regression -- see RegressionGuard in rl/evaluate.py.
    regression_margin: float = 0.10
    regression_patience: int = 2


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


def _load_checkpoint_agent(path: Path) -> RLAgent:
    """``gamekit.rl.selfplay.OpponentPool``'s ``load_opponent``.

    Constructs a fresh ``RLAgent`` wrapper per call (as the pool expects), but
    the actual ``MaskablePPO.load`` only happens once per unique path per
    process -- ``RLAgent``'s own ``_load_model`` is ``@cache``'d at module
    level, which is exactly the "lazy, cached wrapper" gamekit's own
    docstring says a self-play worker needs, since ``sample()`` calls this
    afresh for every non-learner seat of every episode.
    """
    return RLAgent(path, name="selfplay")


def build_env(cfg: TrainConfig, rank: int) -> CatanEnv:
    """The env itself, with no sb3-contrib dependency -- split out from
    ``make_single_env`` specifically so the self-play/fixed-opponent wiring
    (the part worth testing) is exercisable with only the ``rl`` extra
    installed. ``make_single_env`` is the one that needs the training stack,
    for the ``ActionMasker`` wrap alone."""
    rng = random.Random(cfg.seed * 7919 + rank)
    reward = ShapedReward(gamma=cfg.gamma)
    seed = cfg.seed * 104_729 + rank

    if cfg.selfplay_dir is not None:
        pool: OpponentPool[GameState, Action] = OpponentPool(
            cfg.selfplay_dir,
            f"{cfg.label}_*.zip",
            load_opponent=_load_checkpoint_agent,
            baseline_factory=HeuristicAgent,
            baseline_mix=cfg.baseline_mix,
            run_id=cfg.label,
            rng=random.Random(rng.randint(0, 2**31 - 1)),
        )
        env = CatanEnv(
            num_players=cfg.num_players,
            opponent_pool=pool,
            reward=reward,
            seed=seed,
        )
    else:
        agents = [
            build_opponent(cfg.opponents, random.Random(rng.randint(0, 2**31 - 1)))
            for _ in range(cfg.num_players - 1)
        ]
        env = CatanEnv(
            num_players=cfg.num_players,
            agents=agents,
            reward=reward,
            seed=seed,
        )
    return env


def make_single_env(cfg: TrainConfig, rank: int) -> Any:
    """One masked env. Module-level so ``SubprocVecEnv`` can pickle it."""
    from sb3_contrib.common.wrappers import ActionMasker

    return ActionMasker(build_env(cfg, rank), _action_mask_fn)


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


def eval_opponent_kind(cfg: TrainConfig) -> str:
    """What the in-loop eval measures against.

    Independent of what training opponents are, on purpose: in self-play
    mode, ``cfg.opponents`` no longer describes training opponents at all
    (see ``make_single_env``), but eval must still measure against a fixed,
    interpretable target -- ``HeuristicAgent``, the actual Phase 5 gate --
    regardless of what the pool happens to contain at a given point in
    training. Falls back to ``cfg.opponents`` when unset, so plain
    (non-self-play) training's eval behavior from PR #18 is unchanged.
    """
    return cfg.eval_opponents if cfg.eval_opponents is not None else cfg.opponents


def evaluate(cfg: TrainConfig, model: MaskablePPO) -> WinRate:
    opponents = eval_opponent_kind(cfg)
    return evaluate_winrate(
        masked_ppo_predictor(model),
        opponent_factory=lambda rng: build_opponent(opponents, rng),
        num_players=cfg.num_players,
        n_episodes=cfg.eval_episodes,
        seed=cfg.seed + 977,
    )


@dataclass(frozen=True, slots=True)
class TrainResult:
    """What a run produced, and which checkpoint the gate benchmark should
    actually use -- not necessarily ``final_checkpoint``, if training stopped
    early on a regression or simply kept training a little past its peak."""

    final_checkpoint: Path
    best_checkpoint: Path
    best_win_rate: float
    steps_completed: int
    stopped_early: bool


def _resume_step_bookkeeping(
    resumed_from: int | None, steps: int
) -> tuple[int, int, bool]:
    """``(start_step, target_step, reset_num_timesteps)`` for one invocation.

    ``resumed_from`` is the loaded model's own ``num_timesteps`` when
    resuming, or ``None`` for a fresh model. ``steps`` is always "how many
    *new* steps this invocation runs" (the CLI's ``--steps``), never a
    cumulative target -- so a resumed invocation's loop bound and every
    checkpoint filename must be offset by ``resumed_from``, not start counting
    from zero again.

    Pulled out as pure arithmetic, and tested as such, because getting this
    wrong is exactly what caused a real incident: a resumed self-play run's
    early checkpoints (``..._250000.zip``, ``..._500000.zip``, ...) silently
    overwrote an earlier leg's checkpoints of the same name, because both legs
    counted from zero. See ``docs/experiments/`` for the run this happened on.
    ``reset_num_timesteps`` is ``True`` only for a genuinely fresh model's
    first chunk -- a resumed model already restored its true cumulative count
    (confirmed empirically: ``MaskablePPO`` persists ``num_timesteps`` across
    save/load), and resetting it here would discard that.
    """
    if resumed_from is None:
        return 0, steps, True
    return resumed_from, resumed_from + steps, False


def _checkpoint_dir(cfg: TrainConfig) -> Path:
    """Where checkpoints are written and, in self-play mode, where the pool
    looks for them -- the same directory, so a run's own progress becomes its
    own next self-play opponent without a separate copy step.

    In self-play mode this is deliberately *not* re-derived from
    ``cfg.selfplay_dir``/``cfg.label`` here -- it is read off a throwaway
    ``OpponentPool``'s own ``checkpoint_dir`` property, gamekit 0.3.0's
    resolved-path source of truth (gamekit#23), so this can never drift from
    what the pool itself will actually glob. The throwaway callables are
    real, not stubs, but never invoked: only the property is read.
    """
    if cfg.selfplay_dir is None:
        return cfg.run_dir
    pool: OpponentPool[GameState, Action] = OpponentPool(
        cfg.selfplay_dir,
        f"{cfg.label}_*.zip",
        load_opponent=_load_checkpoint_agent,
        baseline_factory=HeuristicAgent,
        run_id=cfg.label,
    )
    return pool.checkpoint_dir


def _seed_selfplay_pool(cfg: TrainConfig) -> None:
    """Copy ``--resume`` or ``--bc-init`` into the pool's checkpoint directory
    under its own glob pattern, so self-play has a real opponent from step 0
    instead of only ever falling back to ``HeuristicAgent`` until the first
    eval checkpoint.

    Skipped if a seed is already there (idempotent across resumed runs) or if
    there's nothing to seed from. ``resume`` takes priority when both are set
    (they are mutually exclusive on the CLI, so in practice only one ever is).
    """
    source = cfg.resume or cfg.bc_init
    if cfg.selfplay_dir is None or source is None:
        return
    checkpoint_dir = _checkpoint_dir(cfg)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    seed_path = checkpoint_dir / f"{cfg.label}_seed.zip"
    if seed_path.exists():
        return
    shutil.copy2(source, seed_path)
    logger.info("seeded self-play pool: %s -> %s", source, seed_path)


def build_model(cfg: TrainConfig, vec_env: Any) -> MaskablePPO:
    """A fresh ``MaskablePPO`` over ``vec_env``, ``cfg``'s own settings.

    Split out from ``train()`` so ``rl/bc.py`` builds the *identical*
    architecture (same ``net_arch``, gamma, seed) for its cross-entropy
    pre-training step -- the BC-initialised checkpoint and a from-scratch
    training run can never drift apart on policy shape, because both go
    through this one constructor.
    """
    from sb3_contrib import MaskablePPO

    return MaskablePPO(
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


def train(cfg: TrainConfig) -> TrainResult:
    import torch
    from sb3_contrib import MaskablePPO

    # One thread per process. With `envs` subprocesses already saturating the
    # box, torch's default intra-op pool oversubscribes the cores and the
    # policy forward pass -- tiny, on a [256, 256] MLP -- gets slower, not
    # faster. This is the single biggest throughput knob on CPU.
    torch.set_num_threads(cfg.torch_threads)

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = _checkpoint_dir(cfg)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _seed_selfplay_pool(cfg)

    vec_env = build_vec_env(cfg)

    if cfg.resume is not None:
        logger.info("resuming from %s", cfg.resume)
        model = MaskablePPO.load(cfg.resume, env=vec_env, device="cpu")
        # Confirmed empirically: MaskablePPO persists num_timesteps across
        # save/load, so a resumed model already knows its own true cumulative
        # step count.
        resumed_from = model.num_timesteps
    elif cfg.bc_init is not None:
        logger.info("warm-starting policy from BC checkpoint %s", cfg.bc_init)
        # A warm *policy*, not a resumed *run*: reuse this invocation's own
        # lr/ent_coef (never the BC checkpoint's pickled schedule -- BC never
        # touches `model.policy.optimizer`, so those values are whatever the
        # PPO constructor that produced the checkpoint happened to use) and
        # never restore `num_timesteps`, so step counting starts at 0 exactly
        # as it would for a from-scratch run.
        model = MaskablePPO.load(
            cfg.bc_init,
            env=vec_env,
            device="cpu",
            custom_objects={
                "learning_rate": cfg.learning_rate,
                "lr_schedule": lambda progress_remaining: cfg.learning_rate,
                "ent_coef": cfg.ent_coef,
            },
        )
        # Confirmed empirically (rl/train.py's own test suite): a schedule
        # that isn't constant across a chunk would sawtooth on every one of
        # `learn()`'s per-chunk progress sweeps.
        if (
            model.lr_schedule(0.0) != cfg.learning_rate
            or model.lr_schedule(1.0) != cfg.learning_rate
        ):
            raise RuntimeError(
                "bc-init lr_schedule override did not take -- expected a "
                f"constant {cfg.learning_rate}"
            )
        if model.ent_coef != cfg.ent_coef:
            raise RuntimeError(
                f"bc-init ent_coef override did not take -- expected {cfg.ent_coef}"
            )
        resumed_from = None
    else:
        model = build_model(cfg, vec_env)
        resumed_from = None

    start_step, target_step, reset_num_timesteps = _resume_step_bookkeeping(
        resumed_from, cfg.steps
    )
    chunk = min(cfg.eval_every, cfg.steps)
    done = start_step
    # Training and evaluation are timed separately. Folding eval into the FPS
    # number understates throughput badly -- in-loop eval is single-process and
    # steps the policy one observation at a time, so it costs far more per step
    # than a 16-way vectorised rollout does.
    train_seconds = 0.0
    eval_seconds = 0.0
    final_path = checkpoint_dir / f"{cfg.label}_final.zip"
    # Seeded with the BC clone's own rate/path when warm-starting, so a
    # fine-tune that immediately regresses below the clone is caught as the
    # regression it is -- an unseeded guard (best_rate=-1.0) would instead
    # adopt the first, possibly-worse, post-chunk eval as its new baseline
    # and never stop until the run fell *another* `margin` below that.
    guard = RegressionGuard(
        margin=cfg.regression_margin,
        patience=cfg.regression_patience,
        best_rate=cfg.bc_init_rate if cfg.bc_init is not None else -1.0,
        best_checkpoint=cfg.bc_init,
    )
    eval_opponents = eval_opponent_kind(cfg)
    stopped_early = False

    while done < target_step:
        this_chunk = min(chunk, target_step - done)
        t0 = time.perf_counter()
        model.learn(
            this_chunk, reset_num_timesteps=reset_num_timesteps, progress_bar=False
        )
        reset_num_timesteps = False  # never reset again after the first chunk
        train_seconds += time.perf_counter() - t0
        done += this_chunk
        checkpoint = checkpoint_dir / f"{cfg.label}_{done}.zip"
        model.save(checkpoint)

        t0 = time.perf_counter()
        win_rate = evaluate(cfg, model)
        eval_seconds += time.perf_counter() - t0
        should_stop = guard.observe(win_rate.rate, checkpoint)

        logger.info(
            "steps=%d/%d train_fps=%.0f train=%.1fmin eval=%.1fmin vs_%s=%s "
            "best=%.1f%%@%s -> %s",
            done,
            target_step,
            (done - start_step) / train_seconds,
            train_seconds / 60,
            eval_seconds / 60,
            eval_opponents,
            win_rate,
            guard.best_rate * 100,
            guard.best_checkpoint.name if guard.best_checkpoint else "-",
            checkpoint.name,
        )

        if should_stop:
            logger.warning(
                "stopping early at steps=%d: vs_%s win rate regressed more than "
                "%.0f points below its best (%.1f%%) for %d consecutive evals -- "
                "possible self-play collapse. Best checkpoint was %s",
                done,
                eval_opponents,
                cfg.regression_margin * 100,
                guard.best_rate * 100,
                cfg.regression_patience,
                guard.best_checkpoint,
            )
            stopped_early = True
            break

    model.save(final_path)
    vec_env.close()
    logger.info("saved %s", final_path)
    return TrainResult(
        final_checkpoint=final_path,
        best_checkpoint=guard.best_checkpoint or final_path,
        best_win_rate=max(guard.best_rate, 0.0),
        steps_completed=done,
        stopped_early=stopped_early,
    )


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
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--ent-coef", type=float, default=defaults.ent_coef)
    parser.add_argument("--eval-every", type=int, default=defaults.eval_every)
    parser.add_argument("--eval-episodes", type=int, default=defaults.eval_episodes)
    parser.add_argument("--torch-threads", type=int, default=defaults.torch_threads)
    parser.add_argument("--label", default=defaults.label)
    parser.add_argument("--run-dir", type=Path, default=defaults.run_dir)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", type=Path, default=None)
    resume_group.add_argument(
        "--bc-init",
        type=Path,
        default=None,
        help=(
            "warm-start the policy from a BC-clone checkpoint (rl.bc); "
            "unlike --resume, step counting starts at 0 -- a warm policy, "
            "not a resumed run. Mutually exclusive with --resume"
        ),
    )
    parser.add_argument(
        "--bc-init-rate",
        type=float,
        default=-1.0,
        help=(
            "the BC clone's own win rate (e.g. rl.bc's n=200 sanity eval), "
            "used only to seed RegressionGuard's baseline when --bc-init "
            "is set"
        ),
    )
    parser.add_argument(
        "--selfplay-dir",
        type=Path,
        default=None,
        help=(
            "enable self-play: training opponents are sampled from a "
            "gamekit OpponentPool over this directory instead of "
            "--opponents; checkpoints are written here too"
        ),
    )
    parser.add_argument("--baseline-mix", type=float, default=defaults.baseline_mix)
    parser.add_argument(
        "--eval-opponents",
        choices=OPPONENT_KINDS,
        default=None,
        help="what the in-loop eval measures against; defaults to --opponents. "
        "Set explicitly to heuristic for self-play, since --opponents no "
        "longer controls training opponents once --selfplay-dir is set",
    )
    parser.add_argument(
        "--regression-margin", type=float, default=defaults.regression_margin
    )
    parser.add_argument(
        "--regression-patience", type=int, default=defaults.regression_patience
    )
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
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        torch_threads=args.torch_threads,
        label=args.label,
        run_dir=args.run_dir,
        resume=args.resume,
        bc_init=args.bc_init,
        bc_init_rate=args.bc_init_rate,
        selfplay_dir=args.selfplay_dir,
        baseline_mix=args.baseline_mix,
        eval_opponents=args.eval_opponents,
        regression_margin=args.regression_margin,
        regression_patience=args.regression_patience,
    )
    result = train(cfg)
    print(
        f"best checkpoint: {result.best_checkpoint} "
        f"(win_rate={result.best_win_rate:.1%}, "
        f"steps={result.steps_completed}, "
        f"stopped_early={result.stopped_early})"
    )


if __name__ == "__main__":
    main()
