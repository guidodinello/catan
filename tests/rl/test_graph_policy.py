"""Rule under test: the graph policy (``rl/graph_policy.py``, issue #42) reads
the same flat observation as the MLP, reads every dim exactly once, respects
the board's symmetries, and drops into every consumer of the
``extract_features -> mlp_extractor -> action_net`` chain; the flat MlpPolicy
path is untouched by default.

Needs the training stack (torch, sb3-contrib, gymnasium); skipped without it.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("numpy")
pytest.importorskip("gymnasium")
pytest.importorskip("sb3_contrib")
torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402
from gymnasium import spaces  # noqa: E402
from sb3_contrib import MaskablePPO  # noqa: E402
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy  # noqa: E402

from agents.heuristic import HeuristicAgent  # noqa: E402
from agents.ismcts import SearchConfig  # noqa: E402
from agents.random_agent import RandomAgent  # noqa: E402
from agents.rl_agent import RLAgent, _load_model  # noqa: E402
from agents.rl_search import RLSearchAgent  # noqa: E402
from engine.board import (  # noqa: E402
    GEOMETRY,
    NUM_EDGES,
    NUM_LAND_HEXES,
    NUM_VERTICES,
)
from engine.game import CatanGame  # noqa: E402
from engine.state import GameState, acting_player  # noqa: E402
from experiments import search_eval as se  # noqa: E402
from experiments.benchmark import benchmark_agent_factory  # noqa: E402
from experiments.rollout import run_game  # noqa: E402
from rl import graph_policy as gp  # noqa: E402
from rl.action_space import (  # noqa: E402
    EDGE_BASE,
    HEX_BASE,
    N_ATOMS,
    STEAL_BASE,
    VERTEX_CITY_BASE,
)
from rl.bc import _forward, generate, load_dataset, train_bc  # noqa: E402
from rl.encoder import (  # noqa: E402
    EDGE_OFFSET,
    HEX_OFFSET,
    OBS_DIM,
    RESERVED_OFFSET,
    TERRAINS,
    VERTEX_OFFSET,
    ObservationEncoder,
)
from rl.train import (  # noqa: E402
    TrainConfig,
    _adopt_checkpoint_policy,
    build_model,
    build_vec_env,
    train,
)

TINY_GNN: dict[str, Any] = {"gnn_dim": 16, "gnn_layers": 2}
TINY_TRAIN: dict[str, Any] = {
    "envs": 1,
    "opponents": "heuristic",
    "n_steps": 64,
    "batch_size": 64,
    "n_epochs": 1,
    "eval_episodes": 1,
    "torch_threads": 1,
}
MILLION = 1_000_000


def _graph_policy(seed: int = 0, noise: float = 0.0) -> gp.GraphMaskablePolicy:
    """A bare policy (no env). ``noise`` perturbs every parameter so outputs
    are not squashed by the near-zero initial head."""
    torch.manual_seed(seed)
    policy = gp.GraphMaskablePolicy(
        spaces.Box(0.0, 1.0, (OBS_DIM,), dtype=np.float32),
        spaces.Discrete(N_ATOMS),
        lambda _: 1e-3,
        **TINY_GNN,
    )
    if noise:
        with torch.no_grad():
            for p in policy.parameters():
                p.add_(noise * torch.randn_like(p))
    policy.eval()
    return policy


def _mid_game_state(seed: int = 4) -> GameState:
    game = CatanGame(num_players=4)
    state = game.reset(seed=seed)
    agent = RandomAgent(rng=random.Random(seed + 500))
    steps = 0
    while not game.is_terminal(state) and steps < 300:
        actor = acting_player(state)
        game.apply_action(
            state, agent.choose_action(state, game.legal_actions(state), actor)
        )
        steps += 1
    return state


def _encode(state: GameState, **kw: Any) -> np.ndarray:
    enc = ObservationEncoder(4)
    enc.reset(state)
    return enc.encode(state, acting_player(state), **kw)


def _tiny_vec_env_and_model(**overrides: Any) -> tuple[Any, Any]:
    cfg = TrainConfig(**{**TINY_TRAIN, "policy": "graph", **TINY_GNN, **overrides})
    vec_env = build_vec_env(cfg)
    return vec_env, build_model(cfg, vec_env)


# ---------------------------------------------------------------------------
# 1. Topology
# ---------------------------------------------------------------------------


def test_topology_tables_agree_with_the_board_geometry() -> None:
    t = gp.build_topology()
    assert t["hex_vertex"].shape == (NUM_LAND_HEXES, 6)
    assert t["edge_vertex"].shape == (NUM_EDGES, 2)
    assert (t["hex_vertex"] >= 0).all()  # every land hex has 6 corners
    assert (t["edge_vertex"] >= 0).all()  # every edge has 2 ends
    for e, (a, b) in enumerate(GEOMETRY.edge_vertices):
        assert set(t["edge_vertex"][e]) == {a, b}

    # Incidence is symmetric: v in hex h's corners <=> h in vertex v's hexes.
    for h in range(NUM_LAND_HEXES):
        for v in t["hex_vertex"][h]:
            assert h in t["vertex_hex"][v]
    for v in range(NUM_VERTICES):
        for h in t["vertex_hex"][v][t["vertex_hex"][v] >= 0]:
            assert v in t["hex_vertex"][h]
        for e in t["vertex_edge"][v][t["vertex_edge"][v] >= 0]:
            assert v in t["edge_vertex"][e]
    for e in range(NUM_EDGES):
        for v in t["edge_vertex"][e]:
            assert e in t["vertex_edge"][v]
    assert (t["vertex_hex"] >= 0).sum() == NUM_LAND_HEXES * 6


# ---------------------------------------------------------------------------
# 2. Slicing coverage
# ---------------------------------------------------------------------------


def test_every_non_reserved_obs_dim_is_read_exactly_once() -> None:
    count = gp.obs_slice_coverage()
    assert (count[:RESERVED_OFFSET] == 1).all()
    assert (count[RESERVED_OFFSET:] == 0).all()  # the reserved zeros are skipped


def test_node_tensors_carry_the_encoders_per_entity_features() -> None:
    state = _mid_game_state()
    enc = ObservationEncoder(4)
    enc.reset(state)
    seat = acting_player(state)
    chosen_edge = 7
    obs = enc.encode(state, seat, buffer_prefix=(EDGE_BASE + chosen_edge,))
    hx, vx, ex, cx = gp.split_obs(torch.as_tensor(obs[None]))

    for i, hexagon in enumerate(GEOMETRY.land_hexes):
        terrain = TERRAINS.index(state.board.terrain[hexagon])
        assert int(hx[0, i, : len(TERRAINS)].argmax()) == terrain
    assert np.array_equal(vx[0].numpy().ravel(), obs[VERTEX_OFFSET:EDGE_OFFSET])
    assert np.array_equal(hx[0].numpy().ravel(), obs[HEX_OFFSET:VERTEX_OFFSET])
    # The buffer's chosen-edge mask is an edge feature, only on the chosen edge.
    assert ex[0, chosen_edge, -1] == 1.0
    assert ex[0, :, -1].sum() == 1.0
    assert cx.shape == (1, gp.CTX_DIM)


# ---------------------------------------------------------------------------
# 3. Equivariance under a 60-degree board rotation
# ---------------------------------------------------------------------------


def _rotate(cube: tuple[int, int, int]) -> tuple[int, int, int]:
    q, r, s = cube
    return (-r, -s, -q)


def _rotation_permutations() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``perm[i]`` = the id entity ``i`` moves to under a 60-degree rotation."""
    hexes = {h: i for i, h in enumerate(GEOMETRY.land_hexes)}
    perm_h = np.array([hexes[_rotate(h)] for h in GEOMETRY.land_hexes])
    by_hexes = {frozenset(s): v for v, s in enumerate(GEOMETRY.vertex_hexes)}
    perm_v = np.array(
        [by_hexes[frozenset(map(_rotate, s))] for s in GEOMETRY.vertex_hexes]
    )
    by_edge = {frozenset(s): e for e, s in enumerate(GEOMETRY.edge_hexes)}
    perm_e = np.array(
        [by_edge[frozenset(map(_rotate, s))] for s in GEOMETRY.edge_hexes]
    )
    return perm_h, perm_v, perm_e


def _permute_obs(obs: np.ndarray, perms: tuple[np.ndarray, ...]) -> np.ndarray:
    perm_h, perm_v, perm_e = perms
    out = obs.copy()
    for offset, perm, width in (
        (HEX_OFFSET, perm_h, obs[HEX_OFFSET:VERTEX_OFFSET].size // NUM_LAND_HEXES),
        (VERTEX_OFFSET, perm_v, obs[VERTEX_OFFSET:EDGE_OFFSET].size // NUM_VERTICES),
        (EDGE_OFFSET, perm_e, gp.OWNER_SLOTS),
    ):
        for i, j in enumerate(perm):
            out[offset + j * width : offset + (j + 1) * width] = obs[
                offset + i * width : offset + (i + 1) * width
            ]
    for i, j in enumerate(perm_e):  # the buffer's per-edge chosen mask
        out[gp.EDGE_MASK_OFFSET + j] = obs[gp.EDGE_MASK_OFFSET + i]
    return out


def _logits_and_value(policy: Any, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    obs_t = torch.as_tensor(obs[None], dtype=torch.float32)
    with torch.no_grad():
        pi, vf = policy.mlp_extractor(policy.extract_features(obs_t))
        return (
            policy.action_net(pi)[0].numpy(),
            policy.value_net(vf)[0].numpy(),
        )


def test_a_board_rotation_permutes_spatial_logits_and_fixes_the_rest() -> None:
    perms = _rotation_permutations()
    for perm in perms:  # the rotation really moves things
        assert not np.array_equal(perm, np.arange(len(perm)))
    policy = _graph_policy(noise=0.3)
    state = _mid_game_state()
    obs = _encode(state, buffer_prefix=(EDGE_BASE + 5,))
    rotated = _permute_obs(obs, perms)
    assert not np.array_equal(obs, rotated)

    logits, value = _logits_and_value(policy, obs)
    r_logits, r_value = _logits_and_value(policy, rotated)
    perm_h, perm_v, perm_e = perms
    atol = 1e-4
    assert np.ptp(logits) > 10 * atol  # the check has something to compare
    np.testing.assert_allclose(r_logits[perm_v], logits[:NUM_VERTICES], atol=atol)
    np.testing.assert_allclose(
        r_logits[VERTEX_CITY_BASE + perm_v],
        logits[VERTEX_CITY_BASE : VERTEX_CITY_BASE + NUM_VERTICES],
        atol=atol,
    )
    np.testing.assert_allclose(
        r_logits[EDGE_BASE + perm_e],
        logits[EDGE_BASE : EDGE_BASE + NUM_EDGES],
        atol=atol,
    )
    np.testing.assert_allclose(
        r_logits[HEX_BASE + perm_h],
        logits[HEX_BASE : HEX_BASE + NUM_LAND_HEXES],
        atol=atol,
    )
    np.testing.assert_allclose(r_logits[STEAL_BASE:], logits[STEAL_BASE:], atol=atol)
    np.testing.assert_allclose(r_value, value, atol=atol)

    # Negative control: comparing without undoing the permutation must fail.
    assert not np.allclose(r_logits[:NUM_VERTICES], logits[:NUM_VERTICES], atol=atol)


# ---------------------------------------------------------------------------
# 4. Output shape and masking
# ---------------------------------------------------------------------------


def test_logits_span_every_atom_and_masked_argmax_is_always_legal() -> None:
    policy = _graph_policy(noise=0.3)
    rng = np.random.default_rng(0)
    obs = torch.as_tensor(rng.random((8, OBS_DIM)), dtype=torch.float32)
    mask = torch.as_tensor(rng.random((8, N_ATOMS)) < 0.1)
    mask[:, 0] = True  # never an empty mask
    logits, values = _forward(policy, obs, mask)
    assert logits.shape == (8, N_ATOMS)
    assert values.shape == (8,)
    picked = logits.argmax(dim=-1)
    assert mask[torch.arange(8), picked].all()


# ---------------------------------------------------------------------------
# 5. _forward == inference
# ---------------------------------------------------------------------------


def test_bc_forward_matches_the_policys_inference_distribution() -> None:
    vec_env, model = _tiny_vec_env_and_model()
    policy = model.policy.eval()
    obs_t = torch.as_tensor(vec_env.reset(), dtype=torch.float32)
    mask = np.asarray(vec_env.env_method("action_masks")[0])
    mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
    with torch.no_grad():
        masked_logits, values = _forward(policy, obs_t, mask_t)
        dist = policy.get_distribution(obs_t, action_masks=mask[None, :])
    label = torch.tensor([int(np.flatnonzero(mask)[0])])
    ce_bc = torch.nn.functional.cross_entropy(masked_logits, label)
    ce_inference = torch.nn.functional.nll_loss(dist.distribution.logits, label)
    assert torch.allclose(ce_bc, ce_inference, atol=1e-4)
    assert values.shape == (1,)
    vec_env.close()


# ---------------------------------------------------------------------------
# 6. pi / vf separation
# ---------------------------------------------------------------------------


def test_policy_and_value_parameters_are_disjoint() -> None:
    policy = _graph_policy()
    ext = policy.mlp_extractor
    pi = {id(p) for m in (ext.pi_trunk, policy.action_net) for p in m.parameters()}
    vf = {
        id(p)
        for m in (ext.vf_trunk, ext.vf_head, policy.value_net)
        for p in m.parameters()
    }
    assert pi and vf and pi.isdisjoint(vf)
    assert pi | vf == {id(p) for p in policy.parameters()}  # nothing unaccounted for
    assert len(list(policy.features_extractor.parameters())) == 0


# ---------------------------------------------------------------------------
# 7. Save/load, RLAgent, RLSearchAgent
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def graph_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> str:
    tmp = tmp_path_factory.mktemp("graph-ckpt")
    vec_env, model = _tiny_vec_env_and_model(run_dir=tmp / "run")
    path = tmp / "graph.zip"
    model.save(path)
    vec_env.close()
    return str(path)


def test_checkpoint_round_trips_and_plays_a_game_and_a_searched_decision(
    graph_checkpoint: str,
) -> None:
    loaded = MaskablePPO.load(graph_checkpoint, device="cpu")
    assert type(loaded.policy) is gp.GraphMaskablePolicy
    assert (loaded.policy.gnn_dim, loaded.policy.gnn_layers) == (16, 2)

    state = _mid_game_state()
    obs = _encode(state)
    fresh = MaskablePPO.load(graph_checkpoint, device="cpu").policy.eval()
    a, va = _logits_and_value(loaded.policy.eval(), obs)
    b, vb = _logits_and_value(fresh, obs)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(va, vb)

    def factory(n: int, e: int, d: int) -> list[Any]:
        return [
            RLAgent(graph_checkpoint, rng=random.Random(d))
            if i == e % n
            else HeuristicAgent()
            for i in range(n)
        ]

    assert run_game(4, 1, 1, factory).turn_count > 0

    def search_factory(n: int, e: int, d: int) -> list[Any]:
        return benchmark_agent_factory(
            "rl_search_vs_heuristic", graph_checkpoint, n, e, d,
            SearchConfig(simulations=2),
        )  # fmt: skip

    holder: list[Any] = []

    def capturing(n: int, e: int, d: int) -> list[Any]:
        agents = search_factory(n, e, d)
        holder.extend(agents)
        return agents

    run_game(4, 2, 2, capturing)
    searcher = next(a for a in holder if hasattr(a, "log"))
    assert isinstance(searcher, RLSearchAgent)
    assert any(d.simulations == 2 for d in searcher.log.decisions)
    _load_model.cache_clear()


# ---------------------------------------------------------------------------
# 8. Tiny end-to-end BC and PPO runs
# ---------------------------------------------------------------------------


def test_train_bc_with_the_graph_policy_and_a_sealed_val_split(
    tmp_path: Path,
) -> None:
    generate(games=12, out_dir=tmp_path / "ds", workers=1, seed_base=1)
    dataset = load_dataset(tmp_path / "ds", mmap=False)
    save = tmp_path / "clone.zip"
    metrics = train_bc(
        dataset,
        save=save,
        epochs=1,
        patience=3,
        batch_size=64,
        policy_name="graph",
        select_split="tune",
        gnn_dim=16,
        gnn_layers=2,
    )
    assert 0.0 <= metrics.val_acc_spatial <= 1.0
    assert metrics.tune_acc_spatial is not None
    assert 0.0 <= metrics.tune_acc_spatial <= 1.0
    loaded = MaskablePPO.load(save, device="cpu")
    assert type(loaded.policy) is gp.GraphMaskablePolicy


def test_a_tiny_ppo_run_from_a_graph_checkpoint_keeps_the_graph_policy(
    tmp_path: Path, graph_checkpoint: str
) -> None:
    cfg = TrainConfig(
        **{**TINY_TRAIN, "run_dir": tmp_path / "run"},
        steps=128,
        bc_init=Path(graph_checkpoint),
        label="graph_tiny",
        # policy deliberately left at the default: the checkpoint wins
    )
    result = train(cfg)
    assert result.steps_completed >= 128
    assert cfg.policy == "graph"
    assert (cfg.gnn_dim, cfg.gnn_layers) == (16, 2)


def test_a_contradicting_explicit_policy_flag_is_refused(graph_checkpoint: str) -> None:
    cfg = TrainConfig(**TINY_TRAIN, policy="mlp", policy_explicit=True)
    model = MaskablePPO.load(graph_checkpoint, device="cpu")
    with pytest.raises(RuntimeError, match="contradicts"):
        _adopt_checkpoint_policy(cfg, model)


# ---------------------------------------------------------------------------
# 9. Determinism
# ---------------------------------------------------------------------------


def test_the_same_seed_plays_the_same_game(graph_checkpoint: str) -> None:
    def factory(n: int, e: int, d: int) -> list[Any]:
        return [
            RLAgent(graph_checkpoint, rng=random.Random(d))
            if i == e % n
            else HeuristicAgent()
            for i in range(n)
        ]

    assert run_game(4, 3, 3, factory) == run_game(4, 3, 3, factory)


# ---------------------------------------------------------------------------
# 10. fresh_014 seed hygiene
# ---------------------------------------------------------------------------


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def test_fresh_014_seeds_are_disjoint_from_every_used_range() -> None:
    # league_013 needs gamekit.league (>= 0.4.0); skip where it is not installed.
    lg = pytest.importorskip("experiments.league_013")
    lo, n = se.SEED_RANGES["fresh_014"]
    fresh = (lo, lo + n - 1)
    assert n % 4 == 0  # exact seat rotation
    used = list(lg.USED_RANGES) + [(1, 3000)]  # BC boards (rl.bc generate, seed 1)
    used += [(0, 12_100_000)]  # in-loop evals: cfg.seed + 977 + step, step <~ 12.05M
    used += [(3 * MILLION + 1, 4 * MILLION), (4 * MILLION + 1, 5 * MILLION)]
    for name in ("smoke", "measured", "fresh_012"):
        used.append((se.SEED_RANGES[name][0], sum(se.SEED_RANGES[name]) - 1))
    assert not any(_overlaps(fresh, u) for u in used if u != fresh)
    assert fresh in lg.USED_RANGES  # registered, so later leagues avoid it


# ---------------------------------------------------------------------------
# The flat path is untouched by default
# ---------------------------------------------------------------------------


def test_the_default_policy_is_still_the_stock_mlp_policy() -> None:
    cfg = TrainConfig(**TINY_TRAIN)
    assert (cfg.policy, cfg.policy_explicit) == ("mlp", False)
    vec_env = build_vec_env(cfg)
    model = build_model(cfg, vec_env)
    assert type(model.policy) is MaskableActorCriticPolicy
    assert model.policy_kwargs == {"net_arch": [256, 256]}
    vec_env.close()


def test_an_unknown_policy_name_is_refused() -> None:
    cfg = TrainConfig(**TINY_TRAIN, policy="transformer")
    vec_env = build_vec_env(cfg)
    with pytest.raises(ValueError, match="unknown policy"):
        build_model(cfg, vec_env)
    vec_env.close()
