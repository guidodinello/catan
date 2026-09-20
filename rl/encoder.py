"""Seat-relative observation encoder.

Two properties this module is responsible for, both load-bearing:

**Seat relativity.** Every player-indexed feature is stored at the offset
``(player - seat) % num_players``, so the learner always reads itself at slot
0. One policy then transfers across all seats under ``randomize_seat``, the
same trick truco-py's seat one-hot plays.

**No hidden information.** The observation carries exactly what
``server/serialize.py:player_view`` would publish to this seat: opponents
contribute *counts* (hand size, dev-hand size) and never composition, and the
public ``victory_points`` is used rather than ``true_victory_points`` -- the
difference between those two literally is the hidden information, as
``engine/game.py`` says in so many words. ``tests/rl/test_encoder.py`` pins
this with an invariance test rather than trusting the review.

Board statics are precomputed once per episode by ``reset()``; only the
robber-dependent parts are recomputed per step.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from engine.board import (
    GEOMETRY,
    NUM_EDGES,
    NUM_LAND_HEXES,
    NUM_VERTICES,
    Board,
    Cube,
    PortType,
    Terrain,
)
from engine.game import MAX_PLAYERS, MIN_PLAYERS, victory_points
from engine.state import (
    BANK_RESOURCE_COUNT,
    CITIES_PER_PLAYER,
    DEV_DECK_SIZE,
    ROADS_PER_PLAYER,
    SETTLEMENTS_PER_PLAYER,
    WINNING_VICTORY_POINTS,
    DevCardType,
    GameState,
    Phase,
    acting_player,
)
from experiments.features import dice_probability, pips
from rl.action_space import (
    N_OPENERS,
    OPENER_BASE,
    RESOURCES,
)

TERRAINS: tuple[Terrain, ...] = tuple(Terrain)
PORT_TYPES: tuple[PortType, ...] = tuple(PortType)
DEV_CARD_TYPES: tuple[DevCardType, ...] = tuple(DevCardType)
PHASES: tuple[Phase, ...] = tuple(Phase)

MAX_PIPS = 5
MAX_KNIGHTS = 14
MAX_PROGRESS_CARDS = 6
MAX_VP_CARDS = 5
DICE_TOTALS = tuple(range(2, 13))

# --- Block widths ----------------------------------------------------------
HEX_FEATURES = len(TERRAINS) + 2  # terrain one-hot, pip, robber
HEX_BLOCK = NUM_LAND_HEXES * HEX_FEATURES  # 152

OWNER_SLOTS = 1 + MAX_PLAYERS  # unowned + 4 seat-relative owners
VERTEX_FEATURES = OWNER_SLOTS + 1 + (1 + len(PORT_TYPES)) + 1  # 14
VERTEX_BLOCK = NUM_VERTICES * VERTEX_FEATURES  # 756

EDGE_BLOCK = NUM_EDGES * OWNER_SLOTS  # 360

OWN_BLOCK = len(RESOURCES) + len(DEV_CARD_TYPES) + 9  # 19
OPPONENT_FEATURES = 8
OPPONENT_BLOCK = (MAX_PLAYERS - 1) * OPPONENT_FEATURES  # 24

GLOBAL_BLOCK = (
    len(PHASES)  # phase one-hot
    + len(RESOURCES)  # bank
    + 1  # dev deck remaining
    + OWNER_SLOTS  # longest road owner
    + OWNER_SLOTS  # largest army owner
    + 2  # raw dice faces
    + len(DICE_TOTALS)  # dice total one-hot
    + MAX_PLAYERS  # turn player, seat-relative
    + MAX_PLAYERS  # acting player, seat-relative
    + 2  # num_players one-hot
    + 1  # setup progress
)  # 49

BUFFER_BLOCK = (1 + N_OPENERS) + 1 + 1 + len(RESOURCES) + NUM_EDGES  # 85
TRADE_BLOCK = len(RESOURCES) + len(RESOURCES) + MAX_PLAYERS + 1  # 15
RESERVED_BLOCK = 24  # zeros, so new features never move OBS_DIM

HEX_OFFSET = 0
VERTEX_OFFSET = HEX_OFFSET + HEX_BLOCK
EDGE_OFFSET = VERTEX_OFFSET + VERTEX_BLOCK
OWN_OFFSET = EDGE_OFFSET + EDGE_BLOCK
OPPONENT_OFFSET = OWN_OFFSET + OWN_BLOCK
GLOBAL_OFFSET = OPPONENT_OFFSET + OPPONENT_BLOCK
BUFFER_OFFSET = GLOBAL_OFFSET + GLOBAL_BLOCK
TRADE_OFFSET = BUFFER_OFFSET + BUFFER_BLOCK
RESERVED_OFFSET = TRADE_OFFSET + TRADE_BLOCK

OBS_DIM = RESERVED_OFFSET + RESERVED_BLOCK  # 1484

MAX_TRADE_SIDE = 4  # engine's MAX_TRADE_OFFER_SIDE, for normalization only


def _clip01(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


class ObservationEncoder:
    """Encodes ``GameState`` into a fixed-width ``[0, 1]`` float32 vector."""

    def __init__(self, num_players: int) -> None:
        if not (MIN_PLAYERS <= num_players <= MAX_PLAYERS):
            raise ValueError("base-game Catan supports 3 or 4 players")
        self.num_players = num_players
        self._hex_static = np.zeros(HEX_BLOCK, dtype=np.float32)
        self._vertex_port = np.zeros(
            (NUM_VERTICES, 1 + len(PORT_TYPES)), dtype=np.float32
        )
        self._vertex_income_by_vertex: list[list[tuple[Cube, float]]] = []

    def reset(self, state: GameState) -> None:
        """Precompute everything that is fixed for this board."""
        board = state.board
        self._hex_static[:] = 0.0
        for i, hexagon in enumerate(GEOMETRY.land_hexes):
            base = i * HEX_FEATURES
            terrain = board.terrain[hexagon]
            self._hex_static[base + TERRAINS.index(terrain)] = 1.0
            token = board.tokens.get(hexagon)
            if token is not None:
                self._hex_static[base + len(TERRAINS)] = pips(token) / MAX_PIPS

        self._vertex_port[:] = 0.0
        for vertex in range(NUM_VERTICES):
            port_type = _vertex_port_type(board, vertex)
            slot = 0 if port_type is None else 1 + PORT_TYPES.index(port_type)
            self._vertex_port[vertex, slot] = 1.0

        self._vertex_income_by_vertex = [
            _vertex_income_terms(board, vertex) for vertex in range(NUM_VERTICES)
        ]

    def encode(
        self,
        state: GameState,
        seat: int,
        *,
        buffer_prefix: tuple[int, ...] = (),
    ) -> NDArray[np.float32]:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        board = state.board
        n = self.num_players

        def rel(player: int) -> int:
            return (player - seat) % n

        # --- hexes ---------------------------------------------------------
        obs[HEX_OFFSET : HEX_OFFSET + HEX_BLOCK] = self._hex_static
        robber_index = GEOMETRY.land_hexes.index(board.robber_hex)
        obs[HEX_OFFSET + robber_index * HEX_FEATURES + len(TERRAINS) + 1] = 1.0

        # --- vertices ------------------------------------------------------
        owner_of_vertex: dict[int, tuple[int, bool]] = {}
        for player in state.players:
            for vertex in player.settlement_vertices:
                owner_of_vertex[vertex] = (player.player_id, False)
            for vertex in player.city_vertices:
                owner_of_vertex[vertex] = (player.player_id, True)

        for vertex in range(NUM_VERTICES):
            base = VERTEX_OFFSET + vertex * VERTEX_FEATURES
            owned = owner_of_vertex.get(vertex)
            if owned is None:
                obs[base] = 1.0
            else:
                owner, is_city = owned
                obs[base + 1 + rel(owner)] = 1.0
                obs[base + OWNER_SLOTS] = 1.0 if is_city else 0.0
            port_base = base + OWNER_SLOTS + 1
            obs[port_base : port_base + 1 + len(PORT_TYPES)] = self._vertex_port[vertex]
            income = sum(
                value
                for hexagon, value in self._vertex_income_by_vertex[vertex]
                if hexagon != board.robber_hex
            )
            obs[base + VERTEX_FEATURES - 1] = _clip01(income)

        # --- edges ---------------------------------------------------------
        owner_of_edge: dict[int, int] = {}
        for player in state.players:
            for edge in player.road_edges:
                owner_of_edge[edge] = player.player_id
        for edge in range(NUM_EDGES):
            base = EDGE_OFFSET + edge * OWNER_SLOTS
            edge_owner = owner_of_edge.get(edge)
            if edge_owner is None:
                obs[base] = 1.0
            else:
                obs[base + 1 + rel(edge_owner)] = 1.0

        # --- own player (full information about oneself) ---------------------
        me = state.players[seat]
        cursor = OWN_OFFSET
        for resource in RESOURCES:
            obs[cursor] = _clip01(me.resources[resource] / BANK_RESOURCE_COUNT)
            cursor += 1
        for card_type in DEV_CARD_TYPES:
            count = sum(1 for c in me.dev_hand if c.card_type is card_type)
            obs[cursor] = _clip01(count / MAX_VP_CARDS)
            cursor += 1
        obs[cursor] = _clip01(me.played_knights / MAX_KNIGHTS)
        obs[cursor + 1] = _clip01(me.played_progress_count / MAX_PROGRESS_CARDS)
        obs[cursor + 2] = _clip01(me.revealed_vp_cards / MAX_VP_CARDS)
        obs[cursor + 3] = me.settlements_remaining / SETTLEMENTS_PER_PLAYER
        obs[cursor + 4] = me.cities_remaining / CITIES_PER_PLAYER
        obs[cursor + 5] = me.roads_remaining / ROADS_PER_PLAYER
        obs[cursor + 6] = 1.0 if me.has_played_dev_card_this_turn else 0.0
        obs[cursor + 7] = _clip01(victory_points(state, seat) / WINNING_VICTORY_POINTS)
        obs[cursor + 8] = _clip01(me.resource_card_count() / BANK_RESOURCE_COUNT)

        # --- opponents (counts only -- never composition) --------------------
        for player in state.players:
            if player.player_id == seat:
                continue
            slot = rel(player.player_id) - 1
            base = OPPONENT_OFFSET + slot * OPPONENT_FEATURES
            obs[base] = _clip01(player.resource_card_count() / BANK_RESOURCE_COUNT)
            obs[base + 1] = _clip01(len(player.dev_hand) / DEV_DECK_SIZE)
            obs[base + 2] = _clip01(player.played_knights / MAX_KNIGHTS)
            obs[base + 3] = _clip01(player.revealed_vp_cards / MAX_VP_CARDS)
            obs[base + 4] = _clip01(
                victory_points(state, player.player_id) / WINNING_VICTORY_POINTS
            )
            obs[base + 5] = player.settlements_remaining / SETTLEMENTS_PER_PLAYER
            obs[base + 6] = player.cities_remaining / CITIES_PER_PLAYER
            obs[base + 7] = player.roads_remaining / ROADS_PER_PLAYER

        # --- global ----------------------------------------------------------
        cursor = GLOBAL_OFFSET
        obs[cursor + PHASES.index(state.phase)] = 1.0
        cursor += len(PHASES)
        for resource in RESOURCES:
            obs[cursor] = _clip01(state.bank[resource] / BANK_RESOURCE_COUNT)
            cursor += 1
        obs[cursor] = _clip01(len(state.dev_deck) / DEV_DECK_SIZE)
        cursor += 1
        for title_owner in (state.longest_road_owner, state.largest_army_owner):
            slot = 0 if title_owner is None else 1 + rel(title_owner)
            obs[cursor + slot] = 1.0
            cursor += OWNER_SLOTS
        if state.dice_roll is not None:
            d1, d2 = state.dice_roll
            obs[cursor] = d1 / 6
            obs[cursor + 1] = d2 / 6
            obs[cursor + 2 + DICE_TOTALS.index(d1 + d2)] = 1.0
        cursor += 2 + len(DICE_TOTALS)
        obs[cursor + rel(state.current_player)] = 1.0
        cursor += MAX_PLAYERS
        obs[cursor + rel(acting_player(state))] = 1.0
        cursor += MAX_PLAYERS
        obs[cursor + (n - MIN_PLAYERS)] = 1.0
        cursor += 2
        if state.setup_sequence:
            obs[cursor] = _clip01(state.setup_position / len(state.setup_sequence))

        # --- composition buffer ------------------------------------------------
        _encode_buffer(obs, buffer_prefix)

        # --- trade offer (public: every seat sees the offer on the table) ------
        offer = state.trade_offer
        if offer is not None:
            cursor = TRADE_OFFSET
            for resource in RESOURCES:
                obs[cursor] = _clip01(offer.give.get(resource, 0) / MAX_TRADE_SIDE)
                cursor += 1
            for resource in RESOURCES:
                obs[cursor] = _clip01(offer.receive.get(resource, 0) / MAX_TRADE_SIDE)
                cursor += 1
            obs[cursor + rel(offer.proposer)] = 1.0
            cursor += MAX_PLAYERS
            obs[cursor] = 1.0 if offer.counter_of is not None else 0.0

        return obs


def _encode_buffer(obs: NDArray[np.float32], prefix: tuple[int, ...]) -> None:
    """Which multi-step action is half-built, and what has been chosen so far.

    The mask already encodes what may come next; this tells the *value*
    function what has been committed to, which the mask alone does not say.
    """
    from rl.action_space import EDGE_BASE, RESOURCE_BASE

    cursor = BUFFER_OFFSET
    if not prefix:
        obs[cursor] = 1.0
        return
    opener = prefix[0]
    if OPENER_BASE <= opener < OPENER_BASE + N_OPENERS:
        obs[cursor + 1 + (opener - OPENER_BASE)] = 1.0
    cursor += 1 + N_OPENERS
    is_discard = RESOURCE_BASE <= opener < RESOURCE_BASE + len(RESOURCES)
    obs[cursor] = 1.0 if is_discard else 0.0
    obs[cursor + 1] = _clip01(len(prefix) / MAX_TRADE_SIDE)
    cursor += 2
    for atom in prefix:
        if RESOURCE_BASE <= atom < RESOURCE_BASE + len(RESOURCES):
            index = atom - RESOURCE_BASE
            obs[cursor + index] = _clip01(obs[cursor + index] + 1 / MAX_TRADE_SIDE)
    cursor += len(RESOURCES)
    for atom in prefix:
        if EDGE_BASE <= atom < EDGE_BASE + NUM_EDGES:
            obs[cursor + (atom - EDGE_BASE)] = 1.0


def _vertex_port_type(board: Board, vertex_id: int) -> PortType | None:
    for edge_id in GEOMETRY.vertex_edges[vertex_id]:
        candidate = board.port_types.get(edge_id)
        if candidate is not None:
            return candidate
    return None


def _vertex_income_terms(board: Board, vertex_id: int) -> list[tuple[Cube, float]]:
    """Per-hex expected-card contributions for a settlement at ``vertex_id``.

    Kept per-hex rather than summed so the robber can be subtracted at encode
    time without recomputing the board. Mirrors
    ``experiments.features.vertex_production`` with ``amount=1``.
    """
    terms: list[tuple[Cube, float]] = []
    for hexagon in GEOMETRY.vertex_hexes[vertex_id]:
        if not GEOMETRY.is_land(hexagon):
            continue
        terrain = board.terrain[hexagon]
        if terrain is Terrain.DESERT:
            continue
        terms.append((hexagon, dice_probability(board.tokens[hexagon])))
    return terms
