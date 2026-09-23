"""Behaviour-cloning warm start: dataset generation + cross-entropy pre-training.

Tests [gamekit's BC-warm-start note](https://github.com/guidodinello/gamekit/blob/main/docs/research/002-bc-warm-start.md)
against self-play PPO's ~12% ceiling
(``docs/experiments/003-selfplay-v2-baseline-mix-0.5.md``). See
``docs/experiments/004-bc-warm-start.md`` for the run this module produced.

Two subcommands::

    uv run python -m rl.bc generate --games 600 --workers 16 --out rl_runs/bc/v1
    uv run python -m rl.bc train --dataset rl_runs/bc/v1 \\
        --save rl_runs/bc/catan_bc_clone.zip

**Dataset generation drives the real ``CatanEnv``.** Four ``HeuristicAgent``s
play each other -- three seated as fixed opponents (``CatanEnv``'s own
contract), the fourth driving the "learner" seat's decisions from outside the
env, exactly the way a trained policy will at inference time. So the
recorded ``(observation, mask, label)`` triples come from the production
composition buffer, coercion path, and encoder, with zero distribution shift
between BC data and PPO rollouts by construction. ``HeuristicAgent``'s chosen
engine ``Action`` is converted to its atom sequence via
``rl.action_space.atom_sequence`` -- the same round-trip function
``ActionComposer.push`` consumes at inference time -- so a 3-card ``Discard``
becomes three resource-atom samples in canonical order, and
``PlayRoadBuilding``/``TradeBank``/``TradePort`` become an opener atom
followed by their parameter atoms. Every recorded label is asserted legal
under its own mask before being written: that assertion *is* the
encode/decode round-trip check, running on every sample rather than trusted
from the design.

**Value targets are Monte-Carlo returns, fit rather than left random.**
``env.step()`` returns ``ShapedReward``'s value at *every* call, including
the 0.0 of a mid-composition sub-step -- and SB3 counts every one of those
calls as one environment step for its own gamma discounting (a sub-step
doesn't advance ``CatanEnv``'s internal episode/step-budget bookkeeping, but
it does advance the vec-env step count PPO's rollout buffer and GAE are built
over). So the backward MC recursion here runs over the *flat atom sequence*
recorded per episode, one entry per sample -- matching the granularity PPO's
own value function is fit against, not one entry per completed engine action.
Episodes that truncate on the step budget are dropped entirely: no bootstrap
value is available for their last recorded returns.

Fitting the value head is free insurance, not a preference: ``net_arch =
{"pi": [256, 256], "vf": [256, 256]}`` (SB3's default expansion of a flat
list) shares no parameters between the policy and value nets (confirmed
empirically -- see ``tests/rl/test_bc.py``), so the value loss cannot
perturb the policy weights at all. Meanwhile a random critic is the fastest
known way to wreck a clone once PPO resumes: its first advantage estimates
would be pure noise applied at full policy-gradient strength, and the MC
returns needed to avoid that fall out of the same generation rollouts for
free.

Two caveats on the accuracy numbers this module reports, both worth stating
next to them rather than after: ``HeuristicAgent`` is deterministic, so
Bayes-optimal top-1 accuracy is 100% and every point of gap is pure
generalization error -- which is what makes top-1 the right headline metric
here. And its own tie-breaking (``max(candidates, key=...)`` over
``legal_actions()``'s enumeration order) is arbitrary among equally-scored
candidates, so some labels are arbitrary and the clone is scored wrong even
when it picks an equally good alternative.
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from agents.heuristic import HeuristicAgent
from rl.action_space import (
    EDGE_BASE,
    HEX_BASE,
    N_ATOMS,
    OPENER_BASE,
    RESOURCE_BASE,
    SIMPLE_BASE,
    STEAL_BASE,
    TRADE_BASE,
    VERTEX_CITY_BASE,
    VERTEX_SETTLEMENT_BASE,
    ActionEncodingError,
    atom_sequence,
)
from rl.env import CatanEnv
from rl.evaluate import evaluate_winrate, masked_ppo_predictor
from rl.reward import ShapedReward

logger = logging.getLogger("rl.bc")

GAMMA_DEFAULT = 0.999
NUM_PLAYERS_DEFAULT = 4
VALIDATION_MOD = 10  # game_id % VALIDATION_MOD == 0 -> validation split

# Atom-block boundaries, derived from rl/action_space.py's own layout
# constants (SSOT -- never a second copy of the block widths). Used only for
# the per-block accuracy breakdown.
ATOM_BLOCKS: tuple[tuple[str, int, int], ...] = (
    ("vertex_settlement", VERTEX_SETTLEMENT_BASE, VERTEX_CITY_BASE),
    ("vertex_city", VERTEX_CITY_BASE, EDGE_BASE),
    ("edge", EDGE_BASE, HEX_BASE),
    ("hex", HEX_BASE, STEAL_BASE),
    ("steal", STEAL_BASE, RESOURCE_BASE),
    ("resource", RESOURCE_BASE, SIMPLE_BASE),
    ("simple", SIMPLE_BASE, OPENER_BASE),
    ("opener", OPENER_BASE, TRADE_BASE),
    ("trade_reserved", TRADE_BASE, N_ATOMS),
)


# --------------------------------------------------------------------------
# Dataset generation
# --------------------------------------------------------------------------


@dataclass(slots=True)
class GameSamples:
    """One game's recorded (obs, mask, label, return) samples."""

    obs: np.ndarray  # (T, OBS_DIM) float32
    masks: np.ndarray  # (T, N_ATOMS) bool
    labels: np.ndarray  # (T,) int64
    returns: np.ndarray  # (T,) float32


def _play_one_game(
    game_id: int, seed: int, gamma: float, num_players: int
) -> GameSamples | None:
    """Replay one heuristic-vs-heuristic-vs-heuristic-vs-heuristic game
    through the real ``CatanEnv``, recording the learner seat's decisions.

    Returns ``None`` for an episode that truncates on the step budget (no
    bootstrap value available for its MC returns -- dropped, not padded).

    Module-level and picklable by name so ``ProcessPoolExecutor`` can run
    many games in parallel, mirroring ``experiments/rollout.py:run_many``.
    """
    opponents = [HeuristicAgent(name="heuristic") for _ in range(num_players - 1)]
    learner = HeuristicAgent(name="heuristic")
    env = CatanEnv(
        num_players=num_players,
        agents=opponents,
        reward=ShapedReward(gamma=gamma),
        seed=seed,
    )
    env.reset(seed=seed)
    learner.reset()

    obs_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    label_list: list[int] = []
    reward_list: list[float] = []

    done = False
    truncated = False
    while not done:
        legal = env.legal_actions()
        assert env.state is not None
        action = learner.choose_action(env.state, legal, env.learner_seat)
        seq = atom_sequence(action, env.learner_seat, env.num_players)
        for atom in seq:
            mask = env.action_masks()
            if not mask[atom]:
                raise ActionEncodingError(
                    f"heuristic-chosen atom {atom} illegal under its own "
                    f"mask (game_id={game_id}, seed={seed}, action={action!r})"
                )
            obs_list.append(env._observe(env.state))
            mask_list.append(mask)
            label_list.append(atom)
            _, reward, terminated, trunc, _ = env.step(np.int64(atom))
            reward_list.append(reward)
            if terminated or trunc:
                done = True
                truncated = trunc
                break

    if truncated:
        return None

    # Backward MC recursion over the flat atom sequence -- see module
    # docstring on why this is atom-level, not decision-level.
    returns = np.empty(len(reward_list), dtype=np.float32)
    running = 0.0
    for i in range(len(reward_list) - 1, -1, -1):
        running = reward_list[i] + gamma * running
        returns[i] = running

    return GameSamples(
        obs=np.asarray(obs_list, dtype=np.float32),
        masks=np.asarray(mask_list, dtype=bool),
        labels=np.asarray(label_list, dtype=np.int64),
        returns=returns,
    )


def generate(
    *,
    games: int,
    out_dir: Path,
    workers: int = 1,
    gamma: float = GAMMA_DEFAULT,
    num_players: int = NUM_PLAYERS_DEFAULT,
    seed_base: int = 1,
) -> None:
    """Generate ``games`` heuristic-vs-heuristic episodes and write the
    resulting dataset as five aligned ``.npy`` files under ``out_dir``.

    Stored as separate arrays (not one ``.npz``) so ``rl.bc train`` can
    ``np.load(..., mmap_mode="r")`` each one instead of holding the whole
    dataset in memory.
    """
    seeds = [(game_id, seed_base + game_id) for game_id in range(games)]

    if workers <= 1:
        results = [
            _play_one_game(game_id, seed, gamma, num_players) for game_id, seed in seeds
        ]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_play_one_game, game_id, seed, gamma, num_players): game_id
                for game_id, seed in seeds
            }
            by_game: dict[int, GameSamples | None] = {}
            for fut in as_completed(futures):
                by_game[futures[fut]] = fut.result()
        results = [by_game[game_id] for game_id, _ in seeds]

    dropped = sum(1 for r in results if r is None)
    kept: list[tuple[int, GameSamples]] = [
        (game_id, r)
        for (game_id, _), r in zip(seeds, results, strict=True)
        if r is not None
    ]
    if not kept:
        raise RuntimeError(
            f"every one of {games} games truncated on the step budget -- "
            f"nothing to write"
        )

    obs = np.concatenate([r.obs for _, r in kept])
    masks = np.concatenate([r.masks for _, r in kept])
    labels = np.concatenate([r.labels for _, r in kept])
    returns = np.concatenate([r.returns for _, r in kept])
    game_id_arr = np.concatenate(
        [np.full(len(r.labels), game_id, dtype=np.int32) for game_id, r in kept]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "obs.npy", obs.astype(np.float16))
    np.save(out_dir / "masks.npy", masks)
    np.save(out_dir / "labels.npy", labels.astype(np.int16))
    np.save(out_dir / "returns.npy", returns)
    np.save(out_dir / "game_id.npy", game_id_arr)

    logger.info(
        "generated %d samples from %d/%d games (%d dropped: truncated on "
        "step budget) -> %s",
        len(labels),
        len(kept),
        games,
        dropped,
        out_dir,
    )


# --------------------------------------------------------------------------
# Cross-entropy pre-training
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Dataset:
    obs: np.ndarray
    masks: np.ndarray
    labels: np.ndarray
    returns: np.ndarray
    game_id: np.ndarray

    @property
    def n(self) -> int:
        return len(self.labels)


def load_dataset(path: Path, *, mmap: bool = True) -> Dataset:
    mode: Literal["r"] | None = "r" if mmap else None
    return Dataset(
        obs=np.load(path / "obs.npy", mmap_mode=mode),
        masks=np.load(path / "masks.npy", mmap_mode=mode),
        labels=np.load(path / "labels.npy", mmap_mode=mode),
        returns=np.load(path / "returns.npy", mmap_mode=mode),
        game_id=np.load(path / "game_id.npy", mmap_mode=mode),
    )


def game_split(game_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(train_idx, val_idx)``, split **by game** so no board leaks across
    the split -- samples inside one game share a board and a trajectory, so
    a random per-sample split would let the model see (near-)duplicates of
    validation boards during training."""
    is_val = (game_id % VALIDATION_MOD) == 0
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


@dataclass(frozen=True, slots=True)
class BCMetrics:
    n_train: int
    n_val: int
    train_acc_raw: float
    train_acc_masked: float
    val_acc_raw: float
    val_acc_masked: float
    val_acc_by_block: dict[str, float]
    train_value_mse: float
    val_value_mse: float
    return_mean: float
    return_std: float

    def report(self) -> str:
        blocks = ", ".join(
            f"{name}={acc:.1%}" for name, acc in self.val_acc_by_block.items()
        )
        return (
            f"BC metrics (n_train={self.n_train}, n_val={self.n_val}):\n"
            f"  train: raw_acc={self.train_acc_raw:.1%} "
            f"masked_acc={self.train_acc_masked:.1%} "
            f"value_mse={self.train_value_mse:.4f}\n"
            f"  val:   raw_acc={self.val_acc_raw:.1%} "
            f"masked_acc={self.val_acc_masked:.1%} "
            f"value_mse={self.val_value_mse:.4f}\n"
            f"  val by block: {blocks}\n"
            f"  returns: mean={self.return_mean:.3f} std={self.return_std:.3f}"
        )


DEFAULT_EVAL_CHUNK = 8192  # rows per forward pass when scoring a full split


def _forward(policy: Any, obs_t: Any, mask_t: Any) -> tuple[Any, Any]:
    """``(masked_logits, values)`` -- the exact computation
    ``MaskableActorCriticPolicy.forward`` performs for
    ``share_features_extractor=True`` (SB3's default, verified against the
    installed sb3-contrib 2.9.0 source), so what BC optimises is literally
    what inference evaluates. Pinned by
    ``tests/rl/test_bc.py::test_forward_matches_inference``.
    """
    features = policy.extract_features(obs_t)
    latent_pi, latent_vf = policy.mlp_extractor(features)
    logits = policy.action_net(latent_pi)
    values = policy.value_net(latent_vf).flatten()
    masked_logits = logits.masked_fill(~mask_t, -1e8)
    return masked_logits, values


def _zero_by_block() -> dict[str, int]:
    return dict.fromkeys((name for name, *_ in ATOM_BLOCKS), 0)


@dataclass(slots=True)
class _SplitScore:
    """Accumulated (not averaged-of-averages) totals over one split, scored
    in chunks so a ~400k-row dataset is never materialised as one batch."""

    n: int = 0
    raw_correct: int = 0
    masked_correct: int = 0
    squared_error: float = 0.0
    block_n: dict[str, int] = field(default_factory=_zero_by_block)
    block_correct: dict[str, int] = field(default_factory=_zero_by_block)

    @property
    def raw_acc(self) -> float:
        return self.raw_correct / self.n if self.n else float("nan")

    @property
    def masked_acc(self) -> float:
        return self.masked_correct / self.n if self.n else float("nan")

    @property
    def value_mse(self) -> float:
        return self.squared_error / self.n if self.n else float("nan")

    @property
    def acc_by_block(self) -> dict[str, float]:
        return {
            name: (self.block_correct[name] / n if n else float("nan"))
            for name, n in self.block_n.items()
        }


def _score_split(
    policy: Any, dataset: Dataset, idx: np.ndarray, chunk: int
) -> _SplitScore:
    """Score ``idx`` in ``chunk``-sized pieces, reading only that piece off
    the memory-mapped dataset arrays at a time."""
    import torch
    import torch.nn.functional as functional

    score = _SplitScore()
    policy.eval()
    with torch.no_grad():
        for start in range(0, len(idx), chunk):
            piece = idx[start : start + chunk]
            obs_t = torch.as_tensor(np.asarray(dataset.obs[piece], dtype=np.float32))
            mask_t = torch.as_tensor(np.asarray(dataset.masks[piece], dtype=bool))
            labels_t = torch.as_tensor(
                np.asarray(dataset.labels[piece], dtype=np.int64)
            )
            returns_t = torch.as_tensor(
                np.asarray(dataset.returns[piece], dtype=np.float32)
            )

            raw_logits = policy.action_net(
                policy.mlp_extractor(policy.extract_features(obs_t))[0]
            )
            masked_logits, values = _forward(policy, obs_t, mask_t)

            raw_preds = raw_logits.argmax(dim=-1)
            masked_preds = masked_logits.argmax(dim=-1)
            score.n += len(piece)
            score.raw_correct += int((raw_preds == labels_t).sum().item())
            score.masked_correct += int((masked_preds == labels_t).sum().item())
            score.squared_error += float(
                functional.mse_loss(values, returns_t, reduction="sum").item()
            )
            for name, lo, hi in ATOM_BLOCKS:
                keep = (labels_t >= lo) & (labels_t < hi)
                n_keep = int(keep.sum().item())
                if n_keep == 0:
                    continue
                score.block_n[name] += n_keep
                score.block_correct[name] += int(
                    (masked_preds[keep] == labels_t[keep]).sum().item()
                )
    policy.train()
    return score


def train_bc(
    dataset: Dataset,
    *,
    save: Path,
    net_arch: list[int] | None = None,
    gamma: float = GAMMA_DEFAULT,
    num_players: int = NUM_PLAYERS_DEFAULT,
    epochs: int = 30,
    patience: int = 3,
    batch_size: int = 1024,
    lr: float = 1e-3,
    seed: int = 0,
    value_loss_weight: float = 0.5,
) -> BCMetrics:
    """Cross-entropy pre-train a fresh ``MaskablePPO`` policy on ``dataset``,
    save it to ``save``, and return held-out metrics.

    Builds the model through ``rl.train.build_model`` -- the same
    constructor a from-scratch training run uses -- so the architecture can
    never drift between a BC-initialised checkpoint and a cold-started one.
    """
    import torch
    import torch.nn.functional as functional

    from rl.train import TrainConfig, build_model, build_vec_env

    cfg = TrainConfig(
        envs=1,
        num_players=num_players,
        opponents="heuristic",
        seed=seed,
        gamma=gamma,
        net_arch=net_arch or [256, 256],
    )
    vec_env = build_vec_env(cfg)
    model = build_model(cfg, vec_env)
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    train_idx, val_idx = game_split(dataset.game_id)
    logger.info(
        "BC training: %d train samples, %d val samples (game-level split)",
        len(train_idx),
        len(val_idx),
    )

    def _batch(idx: np.ndarray) -> tuple[Any, Any, Any, Any]:
        obs_t = torch.as_tensor(np.asarray(dataset.obs[idx], dtype=np.float32))
        mask_t = torch.as_tensor(np.asarray(dataset.masks[idx], dtype=bool))
        labels_t = torch.as_tensor(np.asarray(dataset.labels[idx], dtype=np.int64))
        returns_t = torch.as_tensor(np.asarray(dataset.returns[idx], dtype=np.float32))
        return obs_t, mask_t, labels_t, returns_t

    rng = np.random.default_rng(seed)
    best_acc = -1.0
    best_state: dict[str, Any] | None = None
    stale = 0

    for epoch in range(epochs):
        order = rng.permutation(train_idx)
        epoch_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            obs_t, mask_t, labels_t, returns_t = _batch(batch_idx)
            masked_logits, values = _forward(policy, obs_t, mask_t)
            policy_loss = functional.cross_entropy(masked_logits, labels_t)
            value_loss = functional.mse_loss(values, returns_t)
            loss = policy_loss + value_loss_weight * value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(batch_idx)

        val_score = _score_split(policy, dataset, val_idx, DEFAULT_EVAL_CHUNK)
        logger.info(
            "epoch=%d/%d train_loss=%.4f val_masked_acc=%.1f%% val_value_mse=%.4f",
            epoch + 1,
            epochs,
            epoch_loss / len(order),
            val_score.masked_acc * 100,
            val_score.value_mse,
        )
        if val_score.masked_acc > best_acc:
            best_acc = val_score.masked_acc
            best_state = {k: v.clone() for k, v in policy.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                logger.info(
                    "early stop at epoch %d (no val improvement for %d epochs)",
                    epoch + 1,
                    patience,
                )
                break

    assert best_state is not None
    policy.load_state_dict(best_state)

    metrics = _compute_metrics(policy, dataset, train_idx, val_idx)
    save.parent.mkdir(parents=True, exist_ok=True)
    model.save(save)
    logger.info("saved BC checkpoint: %s", save)
    return metrics


def _compute_metrics(
    policy: Any,
    dataset: Dataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    chunk: int = DEFAULT_EVAL_CHUNK,
) -> BCMetrics:
    """Final held-out report, scored in ``chunk``-sized pieces via
    ``_score_split`` -- see its docstring on why."""
    train_score = _score_split(policy, dataset, train_idx, chunk)
    val_score = _score_split(policy, dataset, val_idx, chunk)

    returns_arr = np.asarray(dataset.returns)
    return BCMetrics(
        n_train=train_score.n,
        n_val=val_score.n,
        train_acc_raw=train_score.raw_acc,
        train_acc_masked=train_score.masked_acc,
        val_acc_raw=val_score.raw_acc,
        val_acc_masked=val_score.masked_acc,
        val_acc_by_block=val_score.acc_by_block,
        train_value_mse=train_score.value_mse,
        val_value_mse=val_score.value_mse,
        return_mean=float(returns_arr.mean()),
        return_std=float(returns_arr.std()),
    )


def sanity_eval(
    checkpoint: Path, *, n_episodes: int = 200, num_players: int = 4
) -> Any:
    """n=200 in-process win rate vs 3 ``HeuristicAgent``s -- a cheap read on
    whether the clone is anywhere near the 25% ceiling before spending time
    on the authoritative n=2000+ ``experiments.benchmark`` run."""
    import torch

    torch.set_num_threads(1)
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(checkpoint, device="cpu")
    return evaluate_winrate(
        masked_ppo_predictor(model),
        opponent_factory=lambda _rng: HeuristicAgent(name="heuristic"),
        num_players=num_players,
        n_episodes=n_episodes,
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BC dataset generation + cross-entropy pre-training for catan RL."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate a heuristic-vs-heuristic dataset")
    gen.add_argument("--games", type=int, default=600)
    gen.add_argument("--workers", type=int, default=1)
    gen.add_argument("--gamma", type=float, default=GAMMA_DEFAULT)
    gen.add_argument("--players", type=int, default=NUM_PLAYERS_DEFAULT)
    gen.add_argument("--seed-base", type=int, default=1)
    gen.add_argument("--out", type=Path, required=True)

    train_p = sub.add_parser(
        "train", help="cross-entropy pre-train a MaskablePPO policy"
    )
    train_p.add_argument("--dataset", type=Path, required=True)
    train_p.add_argument("--save", type=Path, required=True)
    train_p.add_argument("--gamma", type=float, default=GAMMA_DEFAULT)
    train_p.add_argument("--players", type=int, default=NUM_PLAYERS_DEFAULT)
    train_p.add_argument("--epochs", type=int, default=30)
    train_p.add_argument("--patience", type=int, default=3)
    train_p.add_argument("--batch-size", type=int, default=1024)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--seed", type=int, default=0)
    train_p.add_argument("--sanity-episodes", type=int, default=200)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s - %(message)s",
        stream=sys.stdout,
    )
    if args.command == "generate":
        generate(
            games=args.games,
            out_dir=args.out,
            workers=args.workers,
            gamma=args.gamma,
            num_players=args.players,
            seed_base=args.seed_base,
        )
    elif args.command == "train":
        dataset = load_dataset(args.dataset)
        metrics = train_bc(
            dataset,
            save=args.save,
            gamma=args.gamma,
            num_players=args.players,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )
        print(metrics.report())
        win_rate = sanity_eval(
            args.save, n_episodes=args.sanity_episodes, num_players=args.players
        )
        print(
            f"sanity eval (n={args.sanity_episodes}) vs 3 HeuristicAgents: {win_rate}"
        )


if __name__ == "__main__":
    main()
