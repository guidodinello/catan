"""Board-aware (graph) policy network for ``MaskablePPO`` (issue #42).

The observation, action space and BC dataset are unchanged: this module only
replaces the *network*. It re-slices the flat ``OBS_DIM`` vector with the
offsets exported by ``rl.encoder`` into hex / vertex / edge node features plus
one global context vector, runs ``L`` rounds of message passing over the fixed
board incidence (``engine.board.GEOMETRY``), and reads the spatial atoms off
the nodes they name (a pointer head), so every consumer of
``extract_features -> mlp_extractor -> action_net / value_net`` keeps working.

Design points (see ``docs/plans/graph-encoder.md``):

* pi and vf get **independent trunks**, like the flat arm's disjoint pi/vf
  nets, so the value loss never shapes the policy's features.
* No node-ID embeddings: aggregation is a masked mean over neighbours, so the
  network is equivariant to any automorphism of the incidence structure
  (the 12 board symmetries) -- tested, not assumed. The price is that it
  cannot learn ``HeuristicAgent``'s enumeration-order tie-break.
* Pure torch; no GNN library.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import torch
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from torch import nn

from engine.board import GEOMETRY, NUM_EDGES, NUM_LAND_HEXES, NUM_VERTICES
from rl.action_space import (
    EDGE_BASE,
    HEX_BASE,
    N_ATOMS,
    STEAL_BASE,
    VERTEX_CITY_BASE,
    VERTEX_SETTLEMENT_BASE,
)
from rl.encoder import (
    BUFFER_BLOCK,
    BUFFER_OFFSET,
    EDGE_BLOCK,
    EDGE_OFFSET,
    HEX_BLOCK,
    HEX_FEATURES,
    HEX_OFFSET,
    OBS_DIM,
    OWN_OFFSET,
    OWNER_SLOTS,
    TRADE_BLOCK,
    TRADE_OFFSET,
    VERTEX_BLOCK,
    VERTEX_FEATURES,
    VERTEX_OFFSET,
)

GNN_DIM_DEFAULT = 64
GNN_LAYERS_DEFAULT = 3

# Observation slices, all derived from rl.encoder's own constants.
# BUFFER = composition prefix, then the 72-wide per-edge chosen-edge mask.
BUFFER_PREFIX = BUFFER_BLOCK - NUM_EDGES
EDGE_MASK_OFFSET = BUFFER_OFFSET + BUFFER_PREFIX
EDGE_NODE_FEATURES = OWNER_SLOTS + 1  # road owner one-hot + chosen-in-buffer flag
# Context = OWN + OPPONENT + GLOBAL (contiguous), the buffer prefix and TRADE.
CTX_SLICES: tuple[tuple[int, int], ...] = (
    (OWN_OFFSET, BUFFER_OFFSET + BUFFER_PREFIX),
    (TRADE_OFFSET, TRADE_OFFSET + TRADE_BLOCK),
)
CTX_DIM = sum(hi - lo for lo, hi in CTX_SLICES)

# The pointer head writes the atom blocks in this order; it must tile
# ``[0, N_ATOMS)`` exactly, or the logits would be misaligned with the env.
N_VERTEX_ATOMS = NUM_VERTICES
N_OTHER_ATOMS = N_ATOMS - STEAL_BASE
_BLOCK_STARTS = (
    (VERTEX_SETTLEMENT_BASE, 0),
    (VERTEX_CITY_BASE, N_VERTEX_ATOMS),
    (EDGE_BASE, 2 * N_VERTEX_ATOMS),
    (HEX_BASE, 2 * N_VERTEX_ATOMS + NUM_EDGES),
    (STEAL_BASE, 2 * N_VERTEX_ATOMS + NUM_EDGES + NUM_LAND_HEXES),
)
if any(actual != expected for actual, expected in _BLOCK_STARTS):
    raise RuntimeError(
        "rl.action_space's atom layout no longer matches PointerHead's "
        "(settlement, city, road, robber, other) order"
    )


def obs_slice_coverage() -> np.ndarray:
    """How many times the network reads each observation dim (test hook).

    Every non-reserved dim must be read exactly once; the reserved zero block
    is deliberately never read.
    """
    count = np.zeros(OBS_DIM, dtype=np.int64)
    count[HEX_OFFSET : HEX_OFFSET + HEX_BLOCK] += 1
    count[VERTEX_OFFSET : VERTEX_OFFSET + VERTEX_BLOCK] += 1
    count[EDGE_OFFSET : EDGE_OFFSET + EDGE_BLOCK] += 1
    count[EDGE_MASK_OFFSET : EDGE_MASK_OFFSET + NUM_EDGES] += 1
    for lo, hi in CTX_SLICES:
        count[lo:hi] += 1
    return count


def build_topology() -> dict[str, np.ndarray]:
    """Padded neighbour tables (``-1`` pads), indexed like the encoder's blocks.

    ``hex_vertex`` (19, 6), ``edge_vertex`` (72, 2), ``vertex_hex`` (54, 3)
    (land hexes only), ``vertex_edge`` (54, 3).
    """
    hex_index = {h: i for i, h in enumerate(GEOMETRY.land_hexes)}
    hex_vertex = np.full((NUM_LAND_HEXES, 6), -1, dtype=np.int64)
    for h, i in hex_index.items():
        vs = sorted(GEOMETRY.hex_vertices[h])
        hex_vertex[i, : len(vs)] = vs
    edge_vertex = np.asarray(GEOMETRY.edge_vertices, dtype=np.int64)
    vertex_hex = np.full((NUM_VERTICES, 3), -1, dtype=np.int64)
    vertex_edge = np.full((NUM_VERTICES, 3), -1, dtype=np.int64)
    for v in range(NUM_VERTICES):
        hs = sorted(hex_index[h] for h in GEOMETRY.vertex_hexes[v] if h in hex_index)
        vertex_hex[v, : len(hs)] = hs
        es = sorted(GEOMETRY.vertex_edges[v])
        vertex_edge[v, : len(es)] = es
    return {
        "hex_vertex": hex_vertex,
        "edge_vertex": edge_vertex,
        "vertex_hex": vertex_hex,
        "vertex_edge": vertex_edge,
    }


def _mlp(i: int, o: int, hidden: int | None = None) -> nn.Sequential:
    h = hidden or o
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


class _Gather(nn.Module):
    """Masked mean of ``src`` (B, N, d) over a padded neighbour table."""

    index: torch.Tensor
    valid: torch.Tensor
    inv_count: torch.Tensor

    def __init__(self, table: np.ndarray) -> None:
        super().__init__()
        t = torch.as_tensor(table)
        self.register_buffer("index", t.clamp(min=0), persistent=False)
        self.register_buffer("valid", (t >= 0).float().unsqueeze(-1), persistent=False)
        self.register_buffer(
            "inv_count",
            1.0 / (t >= 0).sum(-1, keepdim=True).clamp(min=1).float(),
            persistent=False,
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        gathered = src[:, self.index]  # (B, N, K, d)
        return (gathered * self.valid).sum(dim=2) * self.inv_count


class _Round(nn.Module):
    """One synchronous message-passing round over hex / vertex / edge nodes."""

    def __init__(self, d: int, topo: dict[str, np.ndarray]) -> None:
        super().__init__()
        self.v_from_hex = _Gather(topo["vertex_hex"])
        self.v_from_edge = _Gather(topo["vertex_edge"])
        self.e_from_vertex = _Gather(topo["edge_vertex"])
        self.h_from_vertex = _Gather(topo["hex_vertex"])
        self.upd_v = _mlp(4 * d, d)
        self.upd_e = _mlp(3 * d, d)
        self.upd_h = _mlp(3 * d, d)
        self.norm_v = nn.LayerNorm(d)
        self.norm_e = nn.LayerNorm(d)
        self.norm_h = nn.LayerNorm(d)

    def forward(
        self, h: torch.Tensor, v: torch.Tensor, e: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        def bc(n: int) -> torch.Tensor:
            return c.unsqueeze(1).expand(-1, n, -1)

        v2 = self.norm_v(
            v
            + self.upd_v(
                torch.cat(
                    [v, self.v_from_hex(h), self.v_from_edge(e), bc(NUM_VERTICES)], -1
                )
            )
        )
        e2 = self.norm_e(
            e + self.upd_e(torch.cat([e, self.e_from_vertex(v), bc(NUM_EDGES)], -1))
        )
        h2 = self.norm_h(
            h
            + self.upd_h(torch.cat([h, self.h_from_vertex(v), bc(NUM_LAND_HEXES)], -1))
        )
        return h2, v2, e2


def split_obs(
    obs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flat obs ``(B, OBS_DIM)`` -> hex ``(B, 19, 8)``, vertex ``(B, 54, 14)``,
    edge ``(B, 72, 6)`` and context ``(B, CTX_DIM)`` features."""
    b = obs.shape[0]
    hx = obs[:, HEX_OFFSET : HEX_OFFSET + HEX_BLOCK].reshape(
        b, NUM_LAND_HEXES, HEX_FEATURES
    )
    vx = obs[:, VERTEX_OFFSET : VERTEX_OFFSET + VERTEX_BLOCK].reshape(
        b, NUM_VERTICES, VERTEX_FEATURES
    )
    ex = obs[:, EDGE_OFFSET : EDGE_OFFSET + EDGE_BLOCK].reshape(
        b, NUM_EDGES, OWNER_SLOTS
    )
    chosen = obs[:, EDGE_MASK_OFFSET : EDGE_MASK_OFFSET + NUM_EDGES].unsqueeze(-1)
    ex = torch.cat([ex, chosen], dim=-1)
    cx = torch.cat([obs[:, lo:hi] for lo, hi in CTX_SLICES], dim=-1)
    return hx, vx, ex, cx


class _Trunk(nn.Module):
    """Encode the flat obs into per-node embeddings and a context embedding."""

    def __init__(self, d: int, layers: int, topo: dict[str, np.ndarray]) -> None:
        super().__init__()
        self.enc_h = _mlp(HEX_FEATURES, d)
        self.enc_v = _mlp(VERTEX_FEATURES, d)
        self.enc_e = _mlp(EDGE_NODE_FEATURES, d)
        self.enc_c = _mlp(CTX_DIM, d)
        self.rounds = nn.ModuleList(_Round(d, topo) for _ in range(layers))

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hx, vx, ex, cx = split_obs(obs)
        h, v, e, c = self.enc_h(hx), self.enc_v(vx), self.enc_e(ex), self.enc_c(cx)
        for rnd in self.rounds:
            h, v, e = rnd(h, v, e, c)
        return h, v, e, c


class GraphExtractor(nn.Module):
    """Stands in for SB3's ``MlpExtractor``: ``latent_pi`` is the flattened
    (hex, vertex, edge, context) embeddings of the pi trunk, ``latent_vf`` the
    value trunk's pooled summary."""

    def __init__(self, d: int, layers: int) -> None:
        super().__init__()
        topo = build_topology()
        self.d = d
        self.pi_trunk = _Trunk(d, layers, topo)
        self.vf_trunk = _Trunk(d, layers, topo)
        self.vf_head = _mlp(7 * d, d)
        self.latent_dim_pi = (NUM_LAND_HEXES + NUM_VERTICES + NUM_EDGES + 1) * d
        self.latent_dim_vf = d

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        h, v, e, c = self.pi_trunk(features)
        b = features.shape[0]
        return torch.cat([h.reshape(b, -1), v.reshape(b, -1), e.reshape(b, -1), c], -1)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        h, v, e, c = self.vf_trunk(features)
        latent: torch.Tensor = self.vf_head(_pool(h, v, e, c))
        return latent

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)


def _pool(
    h: torch.Tensor, v: torch.Tensor, e: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    parts = [c]
    for x in (h, v, e):
        parts += [x.mean(dim=1), x.amax(dim=1)]
    return torch.cat(parts, dim=-1)


class PointerHead(nn.Module):
    """Maps ``latent_pi`` to ``N_ATOMS`` logits, in atom order: settlement@v,
    city@v, road@e, robber@hex (each read off the node it names), then the
    non-spatial atoms from pooled node summaries."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.d = d
        self.settle = _mlp(2 * d, 1, d)
        self.city = _mlp(2 * d, 1, d)
        self.road = _mlp(2 * d, 1, d)
        self.robber = _mlp(2 * d, 1, d)
        self.other = _mlp(7 * d, N_OTHER_ATOMS, d)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        b, d = latent.shape[0], self.d
        nodes = latent[:, :-d].reshape(b, -1, d)
        c = latent[:, -d:]
        h = nodes[:, :NUM_LAND_HEXES]
        v = nodes[:, NUM_LAND_HEXES : NUM_LAND_HEXES + NUM_VERTICES]
        e = nodes[:, NUM_LAND_HEXES + NUM_VERTICES :]

        def point(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
            cc = c.unsqueeze(1).expand(-1, x.shape[1], -1)
            logit: torch.Tensor = net(torch.cat([x, cc], -1)).squeeze(-1)
            return logit

        return torch.cat(
            [
                point(self.settle, v),
                point(self.city, v),
                point(self.road, e),
                point(self.robber, h),
                self.other(_pool(h, v, e, c)),
            ],
            dim=-1,
        )


class GraphMaskablePolicy(MaskableActorCriticPolicy):
    """``MaskablePPO`` policy with the graph network above. Select with
    ``policy_kwargs={"gnn_dim": ..., "gnn_layers": ...}``."""

    def __init__(
        self,
        *args: Any,
        gnn_dim: int = GNN_DIM_DEFAULT,
        gnn_layers: int = GNN_LAYERS_DEFAULT,
        **kwargs: Any,
    ) -> None:
        self.gnn_dim = gnn_dim
        self.gnn_layers = gnn_layers
        super().__init__(*args, **kwargs)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data: dict[str, Any] = super()._get_constructor_parameters()
        data.update(gnn_dim=self.gnn_dim, gnn_layers=self.gnn_layers)
        return data

    def _build_mlp_extractor(self) -> None:
        # Duck-types SB3's MlpExtractor: forward / forward_actor / forward_critic
        # and latent_dim_pi / latent_dim_vf.
        extractor: Any = GraphExtractor(self.gnn_dim, self.gnn_layers)
        self.mlp_extractor = extractor

    def _build(self, lr_schedule: Any) -> None:
        self._build_mlp_extractor()
        self.action_net = PointerHead(self.gnn_dim)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        if self.ortho_init:
            for module, gain in (
                (self.mlp_extractor, np.sqrt(2)),
                (self.value_net, 1.0),
            ):
                module.apply(partial(self.init_weights, gain=gain))
            # Near-uniform initial policy, like the flat arm's 0.01 action_net.
            for net in (
                self.action_net.settle,
                self.action_net.city,
                self.action_net.road,
                self.action_net.robber,
                self.action_net.other,
            ):
                self.init_weights(net[0], gain=np.sqrt(2))
                self.init_weights(net[2], gain=0.01)
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
