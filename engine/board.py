"""Hex geometry, board generation, ports, terrain/token layout for base-game Catan.

Geometry (which hexes exist, how vertices/edges are derived and numbered, which
edges are port locations) is fixed, RNG-free, and shared across all games: it
lives in the module-level ``GEOMETRY`` singleton. Per-game randomized content
(terrain placement, number tokens, port types, robber start) lives on the
mutable ``Board`` produced by ``generate_board``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto

Cube = tuple[int, int, int]

DIRECTIONS: tuple[Cube, ...] = (
    (1, -1, 0),
    (1, 0, -1),
    (0, 1, -1),
    (-1, 1, 0),
    (-1, 0, 1),
    (0, -1, 1),
)

LAND_RADIUS = 2
WATER_RADIUS = 3
NUM_LAND_HEXES = 19
NUM_VERTICES = 54
NUM_EDGES = 72
NUM_PORTS = 9
MAX_RESHUFFLE_ATTEMPTS = 10_000


class BoardGenerationError(Exception):
    """Raised when board generation cannot satisfy a required constraint."""


def _cube_add(a: Cube, b: Cube) -> Cube:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _cube_radius(h: Cube) -> int:
    return max(abs(h[0]), abs(h[1]), abs(h[2]))


def _cube_scale(a: Cube, k: int) -> Cube:
    return (a[0] * k, a[1] * k, a[2] * k)


class Resource(Enum):
    LUMBER = auto()
    WOOL = auto()
    GRAIN = auto()
    BRICK = auto()
    ORE = auto()


class Terrain(Enum):
    FOREST = auto()
    PASTURE = auto()
    FIELD = auto()
    HILL = auto()
    MOUNTAIN = auto()
    DESERT = auto()


TERRAIN_RESOURCE: dict[Terrain, Resource] = {
    Terrain.FOREST: Resource.LUMBER,
    Terrain.PASTURE: Resource.WOOL,
    Terrain.FIELD: Resource.GRAIN,
    Terrain.HILL: Resource.BRICK,
    Terrain.MOUNTAIN: Resource.ORE,
}

TERRAIN_COUNTS: dict[Terrain, int] = {
    Terrain.FOREST: 4,
    Terrain.PASTURE: 4,
    Terrain.FIELD: 4,
    Terrain.HILL: 3,
    Terrain.MOUNTAIN: 3,
    Terrain.DESERT: 1,
}

TOKEN_MULTISET: tuple[int, ...] = (
    2,
    3,
    3,
    4,
    4,
    5,
    5,
    6,
    6,
    8,
    8,
    9,
    9,
    10,
    10,
    11,
    11,
    12,
)
RED_NUMBERS = frozenset({6, 8})


class PortType(Enum):
    GENERIC = auto()
    LUMBER = auto()
    WOOL = auto()
    GRAIN = auto()
    BRICK = auto()
    ORE = auto()


PORT_RATE: dict[PortType, int] = {
    PortType.GENERIC: 3,
    PortType.LUMBER: 2,
    PortType.WOOL: 2,
    PortType.GRAIN: 2,
    PortType.BRICK: 2,
    PortType.ORE: 2,
}

PORT_RESOURCE: dict[PortType, Resource] = {
    PortType.LUMBER: Resource.LUMBER,
    PortType.WOOL: Resource.WOOL,
    PortType.GRAIN: Resource.GRAIN,
    PortType.BRICK: Resource.BRICK,
    PortType.ORE: Resource.ORE,
}

PORT_TYPE_MULTISET: tuple[PortType, ...] = (
    PortType.GENERIC,
    PortType.GENERIC,
    PortType.GENERIC,
    PortType.GENERIC,
    PortType.LUMBER,
    PortType.WOOL,
    PortType.GRAIN,
    PortType.BRICK,
    PortType.ORE,
)


@dataclass(frozen=True, slots=True)
class Geometry:
    """Fixed, RNG-free board geometry: derived vertex/edge ids and adjacency."""

    land_hexes: tuple[Cube, ...]
    water_hexes: tuple[Cube, ...]
    vertex_hexes: tuple[frozenset[Cube], ...]
    vertex_neighbors: tuple[frozenset[int], ...]
    vertex_edges: tuple[frozenset[int], ...]
    edge_vertices: tuple[tuple[int, int], ...]
    edge_hexes: tuple[frozenset[Cube], ...]
    hex_vertices: dict[Cube, tuple[int, ...]]
    hex_edges: dict[Cube, tuple[int, ...]]
    port_locations: tuple[int, ...]

    def is_land(self, hexagon: Cube) -> bool:
        return _cube_radius(hexagon) <= LAND_RADIUS

    def is_coastal_edge(self, edge_id: int) -> bool:
        hexes = self.edge_hexes[edge_id]
        return sum(1 for h in hexes if self.is_land(h)) == 1

    def hex_neighbors(self, hexagon: Cube) -> tuple[Cube, ...]:
        return tuple(_cube_add(hexagon, d) for d in DIRECTIONS)


def _generate_land_hexes() -> tuple[Cube, ...]:
    hexes = []
    for q in range(-LAND_RADIUS, LAND_RADIUS + 1):
        for r in range(-LAND_RADIUS, LAND_RADIUS + 1):
            s = -q - r
            h = (q, r, s)
            if _cube_radius(h) <= LAND_RADIUS:
                hexes.append(h)
    return tuple(sorted(hexes))


def _generate_water_ring() -> tuple[Cube, ...]:
    hexes = []
    for q in range(-WATER_RADIUS, WATER_RADIUS + 1):
        for r in range(-WATER_RADIUS, WATER_RADIUS + 1):
            s = -q - r
            h = (q, r, s)
            if _cube_radius(h) == WATER_RADIUS:
                hexes.append(h)
    return tuple(sorted(hexes))


def _hex_ring(center: Cube, radius: int) -> list[Cube]:
    """Standard hex-ring traversal: radius*6 hexes in ring order."""
    if radius == 0:
        return [center]
    results = []
    current = _cube_add(center, _cube_scale(DIRECTIONS[4], radius))
    for direction in range(6):
        for _ in range(radius):
            results.append(current)
            current = _cube_add(current, DIRECTIONS[direction])
    return results


def _build_geometry() -> Geometry:
    land_hexes = _generate_land_hexes()
    water_hexes = _generate_water_ring()
    assert len(land_hexes) == NUM_LAND_HEXES

    def corner_key(h: Cube, i: int) -> frozenset[Cube]:
        return frozenset(
            {h, _cube_add(h, DIRECTIONS[i]), _cube_add(h, DIRECTIONS[(i + 1) % 6])}
        )

    vertex_key_set: set[frozenset[Cube]] = set()
    for h in land_hexes:
        for i in range(6):
            vertex_key_set.add(corner_key(h, i))

    def canonical(key: frozenset[Cube]) -> tuple[Cube, ...]:
        return tuple(sorted(key))

    sorted_keys = sorted(vertex_key_set, key=canonical)
    assert len(sorted_keys) == NUM_VERTICES
    vertex_id_of: dict[frozenset[Cube], int] = {
        key: i for i, key in enumerate(sorted_keys)
    }
    vertex_hexes: tuple[frozenset[Cube], ...] = tuple(sorted_keys)

    hex_vertices: dict[Cube, tuple[int, ...]] = {}
    edge_key_set: set[frozenset[int]] = set()
    for h in land_hexes:
        corner_ids = [vertex_id_of[corner_key(h, i)] for i in range(6)]
        hex_vertices[h] = tuple(corner_ids)
        for i in range(6):
            edge_key_set.add(frozenset({corner_ids[i], corner_ids[(i + 1) % 6]}))

    def edge_canonical(key: frozenset[int]) -> tuple[int, int]:
        a, b = sorted(key)
        return (a, b)

    sorted_edge_keys = sorted(edge_key_set, key=edge_canonical)
    assert len(sorted_edge_keys) == NUM_EDGES
    edge_id_of: dict[frozenset[int], int] = {
        key: i for i, key in enumerate(sorted_edge_keys)
    }
    edge_vertices: tuple[tuple[int, int], ...] = tuple(
        edge_canonical(key) for key in sorted_edge_keys
    )

    hex_edges: dict[Cube, tuple[int, ...]] = {}
    for h in land_hexes:
        hex_corner_ids = hex_vertices[h]
        ids = []
        for i in range(6):
            key = frozenset({hex_corner_ids[i], hex_corner_ids[(i + 1) % 6]})
            ids.append(edge_id_of[key])
        hex_edges[h] = tuple(ids)

    vertex_edges_acc: list[set[int]] = [set() for _ in range(NUM_VERTICES)]
    vertex_neighbors_acc: list[set[int]] = [set() for _ in range(NUM_VERTICES)]
    edge_hexes: list[frozenset[Cube]] = []
    for edge_id, (v1, v2) in enumerate(edge_vertices):
        vertex_edges_acc[v1].add(edge_id)
        vertex_edges_acc[v2].add(edge_id)
        vertex_neighbors_acc[v1].add(v2)
        vertex_neighbors_acc[v2].add(v1)
        shared = vertex_hexes[v1] & vertex_hexes[v2]
        assert len(shared) == 2
        edge_hexes.append(shared)

    vertex_edges = tuple(frozenset(s) for s in vertex_edges_acc)
    vertex_neighbors = tuple(frozenset(s) for s in vertex_neighbors_acc)

    # Port locations: walk the radius-2 boundary ring in order, collecting
    # each hex's coastal edges (in per-hex corner order), then pick 9 evenly
    # spaced entries.
    land_set = set(land_hexes)

    def is_land_hex(h: Cube) -> bool:
        return h in land_set

    coastal_in_ring_order: list[int] = []
    for h in _hex_ring((0, 0, 0), LAND_RADIUS):
        for edge_id in hex_edges[h]:
            hexes = edge_hexes[edge_id]
            if (
                sum(1 for x in hexes if is_land_hex(x)) == 1
                and edge_id not in coastal_in_ring_order
            ):
                coastal_in_ring_order.append(edge_id)

    step = len(coastal_in_ring_order) // NUM_PORTS
    port_locations = tuple(coastal_in_ring_order[i * step] for i in range(NUM_PORTS))
    assert len(port_locations) == NUM_PORTS

    return Geometry(
        land_hexes=land_hexes,
        water_hexes=water_hexes,
        vertex_hexes=vertex_hexes,
        vertex_neighbors=vertex_neighbors,
        vertex_edges=vertex_edges,
        edge_vertices=edge_vertices,
        edge_hexes=tuple(edge_hexes),
        hex_vertices=hex_vertices,
        hex_edges=hex_edges,
        port_locations=port_locations,
    )


GEOMETRY: Geometry = _build_geometry()


def _desert_hex(terrain: dict[Cube, Terrain]) -> Cube:
    for h, t in terrain.items():
        if t is Terrain.DESERT:
            return h
    raise BoardGenerationError("no desert hex in terrain assignment")


def _hexes_are_adjacent(a: Cube, b: Cube) -> bool:
    return any(_cube_add(a, d) == b for d in DIRECTIONS)


@dataclass(slots=True)
class Board:
    """Per-game randomized board content. Geometry lives in ``GEOMETRY``."""

    terrain: dict[Cube, Terrain] = field(default_factory=dict)
    tokens: dict[Cube, int] = field(default_factory=dict)
    port_types: dict[int, PortType] = field(default_factory=dict)
    robber_hex: Cube = (0, 0, 0)

    def copy(self) -> Board:
        return Board(
            terrain=dict(self.terrain),
            tokens=dict(self.tokens),
            port_types=dict(self.port_types),
            robber_hex=self.robber_hex,
        )


def _shuffle_terrain(rng: random.Random) -> dict[Cube, Terrain]:
    pool: list[Terrain] = []
    for terrain, count in TERRAIN_COUNTS.items():
        pool.extend([terrain] * count)
    assert len(pool) == NUM_LAND_HEXES
    rng.shuffle(pool)
    return dict(zip(GEOMETRY.land_hexes, pool, strict=True))


def _shuffle_tokens(
    rng: random.Random, terrain: dict[Cube, Terrain]
) -> dict[Cube, int]:
    non_desert_hexes = tuple(
        h for h in GEOMETRY.land_hexes if terrain[h] is not Terrain.DESERT
    )
    assert len(non_desert_hexes) == len(TOKEN_MULTISET)
    for _ in range(MAX_RESHUFFLE_ATTEMPTS):
        pool = list(TOKEN_MULTISET)
        rng.shuffle(pool)
        tokens = dict(zip(non_desert_hexes, pool, strict=True))
        if _tokens_valid(tokens):
            return tokens
    raise BoardGenerationError(
        f"could not find a valid 6/8-adjacency token layout in "
        f"{MAX_RESHUFFLE_ATTEMPTS} attempts"
    )


def _tokens_valid(tokens: dict[Cube, int]) -> bool:
    red_hexes = [h for h, n in tokens.items() if n in RED_NUMBERS]
    for i, a in enumerate(red_hexes):
        for b in red_hexes[i + 1 :]:
            if _hexes_are_adjacent(a, b):
                return False
    return True


def _shuffle_ports(rng: random.Random) -> dict[int, PortType]:
    pool = list(PORT_TYPE_MULTISET)
    rng.shuffle(pool)
    return dict(zip(GEOMETRY.port_locations, pool, strict=True))


def generate_board(rng: random.Random) -> Board:
    """Generate a fresh randomized board using ``rng`` (owned by GameState)."""
    terrain = _shuffle_terrain(rng)
    tokens = _shuffle_tokens(rng, terrain)
    port_types = _shuffle_ports(rng)
    robber_hex = _desert_hex(terrain)
    return Board(
        terrain=terrain, tokens=tokens, port_types=port_types, robber_hex=robber_hex
    )
