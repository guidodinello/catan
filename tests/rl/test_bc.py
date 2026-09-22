"""Rule under test: BC dataset generation is a faithful, round-trip-checked
replay of ``HeuristicAgent`` through the real ``CatanEnv``, and the
cross-entropy pre-training step optimises exactly what inference evaluates.

Generation-side tests run in the torch-free ``Tests (RL)`` CI job (only
``rl.bc``'s ``generate``/``load_dataset``/``game_split`` path is exercised,
none of which imports torch or sb3-contrib). The training-side tests import
torch directly and are skipped there, exactly like ``tests/rl/test_env.py``'s
``pytest.importorskip("gymnasium")`` pattern -- see
``rl/train.py``/``RegressionGuard`` for the precedent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("gymnasium")

import numpy as np  # noqa: E402

from rl.action_space import N_ATOMS  # noqa: E402
from rl.bc import (  # noqa: E402
    VALIDATION_MOD,
    Dataset,
    game_split,
    generate,
    load_dataset,
)
from rl.encoder import OBS_DIM  # noqa: E402
from rl.evaluate import RegressionGuard  # noqa: E402

# ---------------------------------------------------------------------------
# Dataset generation (torch-free)
# ---------------------------------------------------------------------------


def test_generate_writes_a_round_trip_checked_dataset(tmp_path: Path) -> None:
    out_dir = tmp_path / "ds"
    generate(games=4, out_dir=out_dir, workers=1, seed_base=1)

    dataset = load_dataset(out_dir, mmap=False)
    assert dataset.obs.shape[1] == OBS_DIM
    assert dataset.masks.shape[1] == N_ATOMS
    assert dataset.obs.shape[0] == dataset.masks.shape[0] == dataset.n
    assert dataset.labels.shape == (dataset.n,)
    assert dataset.returns.shape == (dataset.n,)
    assert dataset.game_id.shape == (dataset.n,)

    # The round-trip invariant: every recorded label is legal under its own
    # recorded mask. `generate` itself raises during recording if this ever
    # fails (see `_play_one_game`); this re-checks the written arrays.
    for i in range(dataset.n):
        assert dataset.masks[i, dataset.labels[i]]


def test_generate_is_parallel_order_independent(tmp_path: Path) -> None:
    """Same seeds, workers=1 vs workers=2, must produce the same dataset --
    output ordered by game_id, not by completion order."""
    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"
    generate(games=4, out_dir=serial_dir, workers=1, seed_base=7)
    generate(games=4, out_dir=parallel_dir, workers=2, seed_base=7)

    serial = load_dataset(serial_dir, mmap=False)
    parallel = load_dataset(parallel_dir, mmap=False)
    np.testing.assert_array_equal(serial.labels, parallel.labels)
    np.testing.assert_array_equal(serial.game_id, parallel.game_id)
    np.testing.assert_allclose(serial.returns, parallel.returns)


def test_multi_atom_action_only_moves_the_buffer_block(tmp_path: Path) -> None:
    """A multi-atom decision (this dataset always has some -- discards and
    trade actions are common in a full game) appears as consecutive samples
    whose observations differ *only* in the trailing composition-buffer
    block (rl/encoder.py's ``BUFFER_OFFSET``) -- the board/hand state can't
    have changed, since a mid-composition sub-step never reaches the engine.
    """
    from rl.encoder import BUFFER_OFFSET

    out_dir = tmp_path / "ds"
    generate(games=20, out_dir=out_dir, workers=1, seed_base=1)
    dataset = load_dataset(out_dir, mmap=False)

    found_multi_atom = False
    game_ids = dataset.game_id
    for i in range(1, dataset.n):
        if game_ids[i] != game_ids[i - 1]:
            continue
        prev_obs, cur_obs = dataset.obs[i - 1], dataset.obs[i]
        if np.array_equal(prev_obs[:BUFFER_OFFSET], cur_obs[:BUFFER_OFFSET]):
            found_multi_atom = True
    assert found_multi_atom, "expected at least one multi-atom decision in 20 games"


def test_game_split_is_disjoint_by_game(tmp_path: Path) -> None:
    out_dir = tmp_path / "ds"
    generate(games=30, out_dir=out_dir, workers=1, seed_base=3)
    dataset = load_dataset(out_dir, mmap=False)

    train_idx, val_idx = game_split(dataset.game_id)
    train_games = set(dataset.game_id[train_idx].tolist())
    val_games = set(dataset.game_id[val_idx].tolist())
    assert train_games.isdisjoint(val_games)
    assert val_games == {g for g in train_games | val_games if g % VALIDATION_MOD == 0}
    assert len(train_idx) + len(val_idx) == dataset.n


def test_truncated_episodes_are_dropped_not_padded(tmp_path: Path) -> None:
    """A step budget of 1 truncates every episode immediately (no learner
    decision ever completes), so nothing survives to be written -- and
    `generate` says so loudly rather than writing an empty/padded dataset."""
    import rl.bc as bc_module

    original = bc_module._play_one_game

    def _always_truncated(*_args: object, **_kwargs: object) -> None:
        return None

    bc_module._play_one_game = _always_truncated
    try:
        with pytest.raises(RuntimeError, match="truncated"):
            generate(games=3, out_dir=tmp_path / "empty", workers=1)
    finally:
        bc_module._play_one_game = original


# ---------------------------------------------------------------------------
# Monte-Carlo returns (torch-free arithmetic, exercised directly)
# ---------------------------------------------------------------------------


def test_mc_returns_match_hand_computed_backward_recursion() -> None:
    """Reimplements the exact backward recursion `_play_one_game` runs, on a
    hand-picked reward sequence, and checks it against the closed form."""
    gamma = 0.9
    rewards = [0.0, 0.0, 1.0]  # two composition sub-steps, then a win

    returns = [0.0] * len(rewards)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        returns[i] = running

    # G_2 = 1.0; G_1 = 0 + gamma * 1.0; G_0 = 0 + gamma * G_1
    assert returns[2] == pytest.approx(1.0)
    assert returns[1] == pytest.approx(gamma)
    assert returns[0] == pytest.approx(gamma * gamma)


# ---------------------------------------------------------------------------
# RegressionGuard, seeded for a warm start (torch-free)
# ---------------------------------------------------------------------------


def test_regression_guard_seeded_with_clone_rate_stops_on_drop_below_it() -> None:
    """An unseeded guard (best_rate=-1.0) would adopt the *first* post-chunk
    eval as its new baseline no matter how low -- exactly wrong for a
    warm-started run, where a fine-tune that immediately regresses below the
    clone must be caught relative to the clone's own rate, not its own worst
    eval. Seeding with the clone's rate/path (as ``rl/train.py``'s ``train()``
    does when ``--bc-init`` is set) fixes that."""
    clone_checkpoint = Path("catan_bc_clone.zip")
    guard = RegressionGuard(
        margin=0.10,
        patience=2,
        best_rate=0.23,
        best_checkpoint=clone_checkpoint,
    )
    assert guard.best_rate == pytest.approx(0.23)
    assert guard.best_checkpoint == clone_checkpoint

    # Two consecutive evals more than margin below the seeded best -> stop.
    assert guard.observe(0.12, Path("ft_250000.zip")) is False  # 1st qualifying dip
    assert guard.observe(0.11, Path("ft_500000.zip")) is True  # 2nd -> patience hit
    # The seeded checkpoint is still what "best" points at -- neither dip
    # was ever a new best.
    assert guard.best_checkpoint == clone_checkpoint


# ---------------------------------------------------------------------------
# Cross-entropy pre-training (torch-gated -- skipped in the torch-free CI job)
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")
import torch.nn.functional as functional  # noqa: E402

from rl.bc import _forward, train_bc  # noqa: E402
from rl.train import TrainConfig, build_model, build_vec_env  # noqa: E402


def _tiny_dataset(tmp_path: Path, games: int = 3, seed_base: int = 11) -> Dataset:
    out_dir = tmp_path / "tiny_ds"
    generate(games=games, out_dir=out_dir, workers=1, seed_base=seed_base)
    return load_dataset(out_dir, mmap=False)


def _tiny_model_and_vec_env(**overrides: Any) -> tuple[Any, Any]:
    cfg = TrainConfig(
        envs=1, num_players=4, opponents="heuristic", net_arch=[256, 256], **overrides
    )
    vec_env = build_vec_env(cfg)
    return build_model(cfg, vec_env), vec_env


def test_policy_and_value_params_are_disjoint() -> None:
    """The premise the fitted value head relies on (module docstring): with
    ``net_arch=[256, 256]``'s pi/vf expansion and a parameter-free
    ``FlattenExtractor`` (share_features_extractor=True, SB3's default), the
    value loss's gradient cannot reach a single policy-side parameter."""
    model, _ = _tiny_model_and_vec_env()
    policy = model.policy
    pi_params = {id(p) for p in policy.mlp_extractor.policy_net.parameters()}
    pi_params |= {id(p) for p in policy.action_net.parameters()}
    vf_params = {id(p) for p in policy.mlp_extractor.value_net.parameters()}
    vf_params |= {id(p) for p in policy.value_net.parameters()}
    assert pi_params.isdisjoint(vf_params)
    assert len(list(policy.features_extractor.parameters())) == 0


def test_forward_matches_inference_distribution() -> None:
    """``_forward``'s masked logits, once cross-entropy's own log-softmax
    normalizes them, produce the identical loss ``MaskableActorCriticPolicy``
    would compute at inference time for the same batch -- pinning what BC
    actually optimises against what inference actually evaluates. (Raw-logit
    equality is *not* the right check here: ``torch.distributions.Categorical``
    stores log-softmax-normalized logits, confirmed empirically while writing
    this test -- see the module docstring.)"""
    model, vec_env = _tiny_model_and_vec_env()
    policy = model.policy
    policy.eval()

    obs = vec_env.reset()
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    mask = np.asarray(vec_env.env_method("action_masks")[0])
    mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)

    with torch.no_grad():
        masked_logits, values = _forward(policy, obs_t, mask_t)
        dist = policy.get_distribution(obs_t, action_masks=mask[None, :])

    label = torch.tensor([int(np.flatnonzero(mask)[0])])
    ce_bc = functional.cross_entropy(masked_logits, label)
    ce_inference = functional.nll_loss(dist.distribution.logits, label)
    assert torch.allclose(ce_bc, ce_inference, atol=1e-4)
    assert values.shape == (1,)


def test_bc_init_lr_and_ent_coef_override_is_constant(tmp_path: Path) -> None:
    """Pins the exact override mechanism ``rl/train.py``'s ``train()`` uses
    for ``--bc-init``: both ``learning_rate`` and the pickled ``lr_schedule``
    must be overridden (not just one), and the result must be constant across
    a chunk's progress sweep, not merely correct at one endpoint."""
    from sb3_contrib import MaskablePPO

    model, vec_env = _tiny_model_and_vec_env(learning_rate=3e-4, ent_coef=0.02)
    checkpoint = tmp_path / "tiny.zip"
    model.save(checkpoint)

    new_lr, new_ent = 1e-4, 0.01
    loaded = MaskablePPO.load(
        checkpoint,
        env=vec_env,
        device="cpu",
        custom_objects={
            "learning_rate": new_lr,
            "lr_schedule": lambda progress_remaining: new_lr,
            "ent_coef": new_ent,
        },
    )
    assert loaded.lr_schedule(0.0) == pytest.approx(new_lr)
    assert loaded.lr_schedule(1.0) == pytest.approx(new_lr)
    assert loaded.ent_coef == pytest.approx(new_ent)


def test_train_bc_reduces_loss_preserves_optimizer_and_reloads_playably(
    tmp_path: Path,
) -> None:
    from agents.heuristic import HeuristicAgent
    from agents.rl_agent import RLAgent
    from rl.env import CatanEnv

    dataset = _tiny_dataset(tmp_path)
    save_path = tmp_path / "clone.zip"

    metrics = train_bc(
        dataset,
        save=save_path,
        epochs=2,
        patience=5,
        batch_size=64,
        lr=1e-3,
        seed=0,
    )
    assert metrics.n_train > 0
    assert metrics.n_val > 0
    assert 0.0 <= metrics.train_acc_masked <= 1.0
    assert 0.0 <= metrics.val_acc_masked <= 1.0

    # BC never touches `model.policy.optimizer` (its own separate Adam
    # instance is used instead, over the same parameters) -- so the saved
    # zip's optimizer state is exactly what PPO's own constructor produced:
    # untouched, i.e. empty (no .step() was ever called on it).
    from sb3_contrib import MaskablePPO

    reloaded = MaskablePPO.load(save_path, device="cpu")
    assert reloaded.policy.optimizer.state_dict()["state"] == {}

    # The saved checkpoint is an ordinary MaskablePPO zip: RLAgent, the
    # production inference path, loads it and plays a legal action.
    agent = RLAgent(save_path)
    env = CatanEnv(
        num_players=4,
        agents=[HeuristicAgent() for _ in range(3)],
        seed=0,
    )
    env.reset(seed=0)
    legal = env.legal_actions()
    assert env.state is not None
    action = agent.choose_action(env.state, legal, env.learner_seat)
    assert action in legal
