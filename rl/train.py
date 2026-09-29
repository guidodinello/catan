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
import math
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
INFERENCE_KINDS = ("none", "cpu", "cuda")

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sb3_contrib import MaskablePPO

    from rl.inference_server import InferenceHandle

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
    # The PPO model's own device (build_model/resume/bc-init loads). Separate
    # from opponent_inference below -- see docs/experiments/005-gpu-inference.md
    # for why the two are independent knobs with independent verdicts.
    device: str = "cpu"

    # Self-play. `opponents` above stops governing training opponents once
    # `selfplay_dir` is set (they come from the pool instead) but keeps
    # governing eval, unless `eval_opponents` overrides it -- see `evaluate()`.
    selfplay_dir: Path | None = None
    baseline_mix: float = 0.2
    eval_opponents: str | None = None
    # "none" (default): each worker's RLAgent opponents load their own model
    # and call MaskablePPO.predict locally, exactly as before this option
    # existed. "cpu"/"cuda": route every opponent atom decision through a
    # single InferenceServer instead -- see rl/inference_server.py and
    # docs/experiments/005-gpu-inference.md for when this actually wins.
    opponent_inference: str = "none"

    # A warm-started *policy* rather than a resumed *run*: loads BC-clone
    # weights but keeps step counting at 0 (unlike `resume`, which restores
    # the model's own cumulative `num_timesteps`). Mutually exclusive with
    # `resume` -- see `build_parser`.
    bc_init: Path | None = None
    # The loaded checkpoint's own win rate (the BC clone's, or the previous
    # leg's authoritative n=4000 rate on `resume`), used only to seed
    # `RegressionGuard`'s baseline -- see `train()`. -1.0 means unseeded.
    init_rate: float = -1.0
    # Which checkpoint that rate belongs to; defaults to the one loaded
    # (`resume` or `bc_init`). Set it when the rate was measured on a
    # different checkpoint than the one being resumed from.
    init_best: Path | None = None
    # Names of the hyperparameters (learning_rate / ent_coef / n_steps) the
    # user passed explicitly on the CLI, as opposed to inherited defaults.
    # Only explicit ones are checked against, or applied over, a resumed
    # checkpoint's pickled values -- see `_load_resumed_model`.
    explicit_hparams: frozenset[str] = frozenset()

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


def _load_checkpoint_agent(
    path: Path, *, handle: InferenceHandle | None = None
) -> RLAgent:
    """``gamekit.rl.selfplay.OpponentPool``'s ``load_opponent``.

    Constructs a fresh ``RLAgent`` wrapper per call (as the pool expects), but
    the actual ``MaskablePPO.load`` only happens once per unique path per
    process -- ``RLAgent``'s own ``_load_model`` is ``@cache``'d at module
    level, which is exactly the "lazy, cached wrapper" gamekit's own
    docstring says a self-play worker needs, since ``sample()`` calls this
    afresh for every non-learner seat of every episode.

    ``handle`` is set only when ``cfg.opponent_inference`` enables the
    ``InferenceServer`` (see ``build_vec_env``): it is bound via
    ``functools.partial`` before this is handed to ``OpponentPool`` as its
    ``load_opponent``, so the pool's own one-arg call signature never has to
    know inference routing exists.
    """
    return RLAgent(path, name="selfplay", inference=handle)


def build_env(
    cfg: TrainConfig, rank: int, handle: InferenceHandle | None = None
) -> CatanEnv:
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
            load_opponent=partial(_load_checkpoint_agent, handle=handle),
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


def make_single_env(
    cfg: TrainConfig, rank: int, handle: InferenceHandle | None = None
) -> Any:
    """One masked env. Module-level so ``SubprocVecEnv`` can pickle it."""
    from sb3_contrib.common.wrappers import ActionMasker

    return ActionMasker(build_env(cfg, rank, handle), _action_mask_fn)


def build_vec_env(cfg: TrainConfig) -> Any:
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        SubprocVecEnv,
        VecMonitor,
    )

    if cfg.opponent_inference not in INFERENCE_KINDS:
        raise ValueError(
            f"opponent_inference must be one of {INFERENCE_KINDS}, "
            f"got {cfg.opponent_inference!r}"
        )

    handles: list[InferenceHandle | None]
    server = None
    if cfg.opponent_inference == "none":
        handles = [None] * cfg.envs
    else:
        # Pipes must exist before SubprocVecEnv forks -- see
        # rl/inference_server.py's module docstring on why. Constructed
        # here, never inside a worker.
        from rl.inference_server import InferenceServer

        server = InferenceServer(n_envs=cfg.envs, device=cfg.opponent_inference)
        handles = [server.make_handle(rank) for rank in range(cfg.envs)]

    # partial, not a lambda with a default argument: picklable by name, which
    # is what SubprocVecEnv's worker processes need.
    env_fns: list[Callable[[], Any]] = [
        partial(make_single_env, cfg, rank, handles[rank]) for rank in range(cfg.envs)
    ]
    vec_env: Any
    if cfg.envs == 1:
        vec_env = VecMonitor(DummyVecEnv(env_fns))
    else:
        vec_env = VecMonitor(SubprocVecEnv(env_fns, start_method="fork"))
    # Keeps the server (and its daemon thread) alive for vec_env's lifetime
    # without changing this function's return type -- every existing caller
    # (rl/bc.py, tests) passes opponent_inference="none" and gets `server is
    # None` here, so this is a no-op for them. `vec_env: Any` above (not the
    # class stable_baselines3 actually returns) so this assignment needs no
    # `# type: ignore` whose necessity would otherwise depend on whether
    # stable_baselines3 happens to be installed in the mypy environment
    # (installed here, deliberately absent from the pre-commit hook's --
    # see .pre-commit-config.yaml's comment on why).
    vec_env._inference_server = server
    return vec_env


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


def _cpu_eval_model(cfg: TrainConfig, model: MaskablePPO) -> Any:
    """A CPU copy of ``model``'s policy for in-loop eval.

    In-loop eval calls ``model.predict`` one observation at a time (see
    ``rl.evaluate.evaluate_winrate``); ``docs/experiments/005-gpu-inference.md``
    measured single-sample ``predict`` on CUDA as *slower* than CPU (kernel-
    launch latency dominates a batch of one), so evaluating directly on
    ``cfg.device`` when it is "cuda" would be a pure regression, not a
    convenience. A no-op when ``cfg.device`` is already "cpu".

    Returns the bare policy (``MaskableActorCriticPolicy``), not a second
    ``MaskablePPO`` -- its own ``predict(observation, state, episode_start,
    deterministic, action_masks)`` has the identical signature
    ``masked_ppo_predictor`` calls, so nothing downstream needs to know the
    difference.

    Built through ``build_model`` (the identical architecture ``model``
    itself was built with) with the *training run's* global RNG state saved
    and restored around the call -- not ``copy.deepcopy(model.policy)``,
    which was tried first and fails on a real (non-leaf) policy: torch's
    tensor ``__deepcopy__`` only supports graph-leaf tensors, and a policy
    that has actually been optimized is not one (confirmed empirically,
    ``RuntimeError: Only Tensors created explicitly by the user (graph
    leaves) support the deepcopy protocol``, raised from inside a real
    training run's first eval chunk, not synthetically).

    The RNG save/restore is why ``build_model`` is safe to call here despite
    its own docstring's warning elsewhere in this module: ``MaskablePPO.__init__``
    -> ``_setup_model()`` calls SB3's ``set_random_seed`` unconditionally,
    reseeding the *global* Python/numpy/torch RNGs as a side effect
    (confirmed in stable-baselines3's source). Left unguarded, that would
    silently reseed the training run's own randomness (env draws,
    opponent-pool sampling, PPO's stochastic action sampling) on every
    in-loop eval chunk, only when ``cfg.device == "cuda"`` -- exactly the bug
    ``tests/rl/test_cpu_eval_model.py`` pins by asserting the global RNG
    state is bit-identical before and after this call.
    """
    if cfg.device == "cpu":
        return model
    import dataclasses
    import random

    import numpy as np
    import torch

    random_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        cpu_cfg = dataclasses.replace(cfg, device="cpu")
        cpu_model = build_model(cpu_cfg, model.env)
        cpu_model.policy.load_state_dict(model.policy.state_dict())
        return cpu_model.policy
    finally:
        random.setstate(random_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)


_EVAL_SEED_OFFSET = 977


def in_loop_eval_seeds(cfg: TrainConfig, step: int) -> tuple[int, int]:
    """``(guard_seed, reported_seed)`` for the in-loop eval at cumulative ``step``.

    The guard seed is fixed for the whole run (and unchanged from 004/006, so
    guard numbers stay comparable): a fixed game set makes the checkpoint-to-
    checkpoint comparison paired, which is what a stop-on-regression rule wants.
    The reported seed advances with ``step`` so the rate that picks the best
    checkpoint is a fresh sample each time rather than the same 200 setups
    replayed (issue #25). Derived from ``step`` -- already offset by
    ``resumed_from`` -- not a loop counter, so a resumed leg never replays the
    previous leg's seeds. ``step >= 1``, so the two seeds never coincide.
    """
    guard = cfg.seed + _EVAL_SEED_OFFSET
    return guard, guard + step


def evaluate(cfg: TrainConfig, eval_model: Any, *, seed: int) -> WinRate:
    opponents = eval_opponent_kind(cfg)
    return evaluate_winrate(
        masked_ppo_predictor(eval_model),
        opponent_factory=lambda rng: build_opponent(opponents, rng),
        num_players=cfg.num_players,
        n_episodes=cfg.eval_episodes,
        seed=seed,
    )


@dataclass(frozen=True, slots=True)
class InLoopEval:
    """Both in-loop rates for one checkpoint. ``guard`` (fixed game set) only
    drives the stop decision; ``reported`` (fresh seed) alone picks the best
    checkpoint. Max-over-evals selection bias remains either way -- n=4000
    confirmation is still mandatory (docs/experiments/README.md)."""

    guard: WinRate
    reported: WinRate


def evaluate_in_loop(cfg: TrainConfig, model: MaskablePPO, step: int) -> InLoopEval:
    guard_seed, reported_seed = in_loop_eval_seeds(cfg, step)
    eval_model = _cpu_eval_model(cfg, model)  # built once, used for both evals
    return InLoopEval(
        guard=evaluate(cfg, eval_model, seed=guard_seed),
        reported=evaluate(cfg, eval_model, seed=reported_seed),
    )


@dataclass(frozen=True, slots=True)
class TrainResult:
    """What a run produced, and which checkpoint the gate benchmark should
    actually use -- not necessarily ``final_checkpoint``, if training stopped
    early on a regression or simply kept training a little past its peak.
    ``best_*`` come from the fresh-seed in-loop rate, not the guard's fixed
    game set; still a max over noisy evals, so confirm at n=4000."""

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


def _save_new(model: MaskablePPO, path: Path) -> None:
    """``model.save`` that refuses to replace an existing file.

    Checkpoints are the experiment's only durable output, and a previous
    leg's files sit in the same directory as a resumed leg's (see
    ``_resume_step_bookkeeping`` for the incident this guards against).
    """
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint {path}")
    model.save(path)


def _load_resumed_model(cfg: TrainConfig, vec_env: Any) -> MaskablePPO:
    """``MaskablePPO.load`` for ``--resume``, with no silent hyperparameters.

    A plain load keeps the pickled ``learning_rate`` / ``ent_coef`` and
    ignores the CLI's. That is right for a same-config continuation but a
    trap otherwise, so: an *explicitly passed* ``--learning-rate`` /
    ``--ent-coef`` that differs from the pickled value is a ``RuntimeError``
    (gamekit#009), and ``--n-steps`` -- a rollout-shape knob, not something
    to refuse -- is applied over the pickled value. The effective values are
    always logged.
    """
    from sb3_contrib import MaskablePPO

    assert cfg.resume is not None
    custom_objects: dict[str, Any] = {}
    if "n_steps" in cfg.explicit_hparams:
        custom_objects["n_steps"] = cfg.n_steps
    model = MaskablePPO.load(
        cfg.resume, env=vec_env, device=cfg.device, custom_objects=custom_objects
    )
    pickled = {
        "learning_rate": model.lr_schedule(1.0),
        "ent_coef": model.ent_coef,
    }
    for name, value in pickled.items():
        wanted = getattr(cfg, name)
        if name in cfg.explicit_hparams and not math.isclose(
            value, wanted, rel_tol=1e-9
        ):
            raise RuntimeError(
                f"--{name.replace('_', '-')}={wanted} differs from the resumed "
                f"checkpoint's pickled {name}={value}; a resume keeps the "
                "pickled value, so refusing to run different hyperparameters "
                "than the command line says"
            )
    if model.n_steps != model.rollout_buffer.buffer_size:
        raise RuntimeError("n_steps override did not reach the rollout buffer")
    logger.info(
        "resumed hyperparameters: learning_rate=%s (lr_schedule(0)=%s "
        "lr_schedule(1)=%s) ent_coef=%s n_steps=%d n_envs=%d num_timesteps=%d",
        model.learning_rate,
        model.lr_schedule(0.0),
        model.lr_schedule(1.0),
        model.ent_coef,
        model.n_steps,
        model.n_envs,
        model.num_timesteps,
    )
    return model


def _make_guard(cfg: TrainConfig) -> RegressionGuard:
    """Seeded from ``init_rate`` / ``init_best`` on either warm-start path
    (``resume`` or ``bc_init``); unseeded (-1.0) for a from-scratch run."""
    warm_start = cfg.resume or cfg.bc_init
    return RegressionGuard(
        margin=cfg.regression_margin,
        patience=cfg.regression_patience,
        best_rate=cfg.init_rate if warm_start is not None else -1.0,
        best_checkpoint=cfg.init_best or warm_start,
    )


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
        device=cfg.device,
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
        model = _load_resumed_model(cfg, vec_env)
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
            device=cfg.device,
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
    # Seeded with the loaded checkpoint's own rate/path (BC clone, or the
    # previous leg's n=4000 result on --resume), so a run that immediately
    # regresses below its start is caught as the
    # regression it is -- an unseeded guard (best_rate=-1.0) would instead
    # adopt the first, possibly-worse, post-chunk eval as its new baseline
    # and never stop until the run fell *another* `margin` below that.
    guard = _make_guard(cfg)
    # Picks the checkpoint handed to n=4000; the guard above only decides stops.
    # `observe()`'s stop verdict is deliberately ignored.
    best = _make_guard(cfg)
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
        _save_new(model, checkpoint)

        t0 = time.perf_counter()
        ev = evaluate_in_loop(cfg, model, done)
        eval_seconds += time.perf_counter() - t0
        should_stop = guard.observe(ev.guard.rate, checkpoint)
        best.observe(ev.reported.rate, checkpoint)

        logger.info(
            "steps=%d/%d train_fps=%.0f train=%.1fmin eval=%.1fmin "
            "vs_%s reported=%s guard(fixed)=%s "
            "best=%.1f%%@%s guard_best=%.1f%% -> %s",
            done,
            target_step,
            (done - start_step) / train_seconds,
            train_seconds / 60,
            eval_seconds / 60,
            eval_opponents,
            ev.reported,
            ev.guard,
            best.best_rate * 100,
            best.best_checkpoint.name if best.best_checkpoint else "-",
            guard.best_rate * 100,
            checkpoint.name,
        )

        if should_stop:
            logger.warning(
                "stopping early at steps=%d: vs_%s win rate regressed more than "
                "%.0f points below its best (%.1f%%) for %d consecutive evals -- "
                "possible self-play collapse (fixed guard set). Best checkpoint "
                "by the fresh-seed rate was %s",
                done,
                eval_opponents,
                cfg.regression_margin * 100,
                guard.best_rate * 100,
                cfg.regression_patience,
                best.best_checkpoint,
            )
            stopped_early = True
            break

    _save_new(model, final_path)
    vec_env.close()
    logger.info("saved %s", final_path)
    return TrainResult(
        final_checkpoint=final_path,
        best_checkpoint=best.best_checkpoint or final_path,
        best_win_rate=max(best.best_rate, 0.0),
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
    # None sentinels: whether these were passed explicitly matters on --resume
    # (see _load_resumed_model); main() falls back to TrainConfig's defaults.
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="rollout length per env; on --resume, overrides the pickled value "
        "(e.g. 256 with --envs 16 keeps a 4096-step rollout)",
    )
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
        "--init-rate",
        "--bc-init-rate",
        dest="init_rate",
        type=float,
        default=None,
        help="win rate of the checkpoint loaded by --bc-init or --resume "
        "(the BC clone's, or the previous leg's authoritative n=4000 rate), "
        "used only to seed RegressionGuard's baseline",
    )
    parser.add_argument(
        "--init-best",
        type=Path,
        default=None,
        help="the checkpoint --init-rate was measured on, if not the one "
        "loaded; defaults to the --resume / --bc-init checkpoint",
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
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=defaults.device,
        help="device for the PPO model itself (build/resume/bc-init loads). "
        "In-loop eval always runs on a CPU copy regardless -- see "
        "docs/experiments/005-gpu-inference.md",
    )
    parser.add_argument(
        "--opponent-inference",
        choices=INFERENCE_KINDS,
        default=defaults.opponent_inference,
        help="'none' (default): each worker's RLAgent opponents call "
        "MaskablePPO.predict locally, as before this option existed. "
        "'cpu'/'cuda': route opponent-checkpoint inference through a single "
        "batched InferenceServer -- see rl/inference_server.py",
    )
    return parser


def config_from_args(
    parser: argparse.ArgumentParser, argv: list[str] | None = None
) -> TrainConfig:
    """Parse ``argv`` into a ``TrainConfig``, validating cross-flag rules
    argparse can't express. Split from ``main`` so it is unit-testable."""
    args = parser.parse_args(argv)
    warm_start = args.resume or args.bc_init
    if warm_start is None and (
        args.init_rate is not None or args.init_best is not None
    ):
        parser.error("--init-rate / --init-best require --resume or --bc-init")

    defaults = TrainConfig()
    explicit = frozenset(
        name
        for name in ("learning_rate", "ent_coef", "n_steps")
        if getattr(args, name) is not None
    )

    def pick(name: str) -> Any:
        value = getattr(args, name)
        return getattr(defaults, name) if value is None else value

    return TrainConfig(
        steps=args.steps,
        envs=args.envs,
        opponents=args.opponents,
        num_players=args.players,
        seed=args.seed,
        gamma=args.gamma,
        learning_rate=pick("learning_rate"),
        ent_coef=pick("ent_coef"),
        n_steps=pick("n_steps"),
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        torch_threads=args.torch_threads,
        label=args.label,
        run_dir=args.run_dir,
        resume=args.resume,
        bc_init=args.bc_init,
        init_rate=defaults.init_rate if args.init_rate is None else args.init_rate,
        init_best=args.init_best,
        explicit_hparams=explicit,
        selfplay_dir=args.selfplay_dir,
        baseline_mix=args.baseline_mix,
        eval_opponents=args.eval_opponents,
        regression_margin=args.regression_margin,
        regression_patience=args.regression_patience,
        device=args.device,
        opponent_inference=args.opponent_inference,
    )


def main() -> None:
    cfg = config_from_args(build_parser())

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s - %(message)s",
        stream=sys.stdout,
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
