"""Analytic (non-simulated) dice, production, and vertex-quality tables.

Pure combinatorics over ``engine.board`` -- no rollouts. This is the ground
truth the empirical income tables in ``exp_tables.py`` are validated
against.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from engine.board import (
    GEOMETRY,
    TERRAIN_RESOURCE,
    Board,
    PortType,
    Resource,
    Terrain,
)

DICE_FACES = range(1, 7)

# Number of (d1, d2) combinations, out of 36, that sum to each total 2..12.
# This is also the standard Catan "pip count" for a token (6/8 -> 5 pips,
# 2/12 -> 1 pip).
DICE_WAYS: dict[int, int] = {
    total: sum(1 for d1, d2 in product(DICE_FACES, DICE_FACES) if d1 + d2 == total)
    for total in range(2, 13)
}


def dice_probability(total: int) -> float:
    """P(2d6 == total), exact."""
    return DICE_WAYS.get(total, 0) / 36


def pips(token: int) -> int:
    """Standard Catan pip count for a number token (number of dice-roll
    combinations that produce it)."""
    return DICE_WAYS.get(token, 0)


def vertex_production(
    board: Board, vertex_id: int, amount: int = 1
) -> dict[Resource, float]:
    """Expected resource cards per roll for a settlement (``amount=1``) or
    city (``amount=2``) at ``vertex_id``, given ``board``'s current terrain,
    tokens, and robber position.

    This is a long-run average conditioned on the robber *staying put* --
    valid for comparison against empirical income only while the robber
    hasn't moved from ``board.robber_hex`` (see the analytic-vs-empirical
    cross-check in ``tests/test_experiments.py``, which filters production
    events by robber position for exactly this reason).
    """
    expected: dict[Resource, float] = {}
    for h in GEOMETRY.vertex_hexes[vertex_id]:
        if not GEOMETRY.is_land(h) or h == board.robber_hex:
            continue
        terrain = board.terrain[h]
        if terrain is Terrain.DESERT:
            continue
        resource = TERRAIN_RESOURCE[terrain]
        token = board.tokens[h]
        expected[resource] = expected.get(resource, 0.0) + amount * dice_probability(
            token
        )
    return expected


@dataclass(frozen=True, slots=True)
class VertexFeatures:
    """Static site-quality features for one vertex, independent of the
    current robber position (pip count is the standard Catan usage: a
    fixed property of the board, not of where the robber currently sits).
    """

    vertex_id: int
    pip_sum: int
    distinct_resources: int
    num_hexes: int
    touches_desert: bool
    port_type: PortType | None
    port_rate: int | None


def vertex_features(board: Board, vertex_id: int) -> VertexFeatures:
    land_hexes = [h for h in GEOMETRY.vertex_hexes[vertex_id] if GEOMETRY.is_land(h)]
    non_desert = [h for h in land_hexes if board.terrain[h] is not Terrain.DESERT]
    resources = {TERRAIN_RESOURCE[board.terrain[h]] for h in non_desert}
    pip_sum = sum(pips(board.tokens[h]) for h in non_desert)
    touches_desert = len(non_desert) < len(land_hexes)

    port_type: PortType | None = None
    port_rate: int | None = None
    for edge_id in GEOMETRY.vertex_edges[vertex_id]:
        candidate = board.port_types.get(edge_id)
        if candidate is not None:
            port_type = candidate
            port_rate = 3 if candidate is PortType.GENERIC else 2
            break

    return VertexFeatures(
        vertex_id=vertex_id,
        pip_sum=pip_sum,
        distinct_resources=len(resources),
        num_hexes=len(land_hexes),
        touches_desert=touches_desert,
        port_type=port_type,
        port_rate=port_rate,
    )
