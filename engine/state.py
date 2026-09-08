"""Mutable game state: phases, per-player state, and the top-level GameState.

Mirrors truco-py's convention: a mutable ``@dataclass(slots=True)`` state with
an explicit ``copy()`` that forks every mutable field (including the RNG
stream), so rollout branches never alias each other's data.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto

from .board import Board, Resource

SETTLEMENTS_PER_PLAYER = 5
CITIES_PER_PLAYER = 4
ROADS_PER_PLAYER = 15
BANK_RESOURCE_COUNT = 19
MAX_HAND_BEFORE_DISCARD = 7
LONGEST_ROAD_MIN_LENGTH = 5
LARGEST_ARMY_MIN_KNIGHTS = 3
WINNING_VICTORY_POINTS = 10


class Phase(Enum):
    SETUP_SETTLEMENT = auto()
    SETUP_ROAD = auto()
    ROLL = auto()
    DISCARD = auto()
    MOVE_ROBBER = auto()
    STEAL = auto()
    MAIN = auto()
    AWAIT_TRADE_RESPONSE = auto()
    GAME_OVER = auto()


class DevCardType(Enum):
    KNIGHT = auto()
    ROAD_BUILDING = auto()
    YEAR_OF_PLENTY = auto()
    MONOPOLY = auto()
    VICTORY_POINT = auto()


PROGRESS_CARD_TYPES = frozenset(
    {DevCardType.ROAD_BUILDING, DevCardType.YEAR_OF_PLENTY, DevCardType.MONOPOLY}
)

DEV_DECK_COUNTS: dict[DevCardType, int] = {
    DevCardType.KNIGHT: 14,
    DevCardType.ROAD_BUILDING: 2,
    DevCardType.YEAR_OF_PLENTY: 2,
    DevCardType.MONOPOLY: 2,
    DevCardType.VICTORY_POINT: 5,
}
DEV_DECK_SIZE = sum(DEV_DECK_COUNTS.values())


@dataclass(slots=True)
class DevCard:
    """One development-card instance held in a player's hand."""

    card_type: DevCardType
    bought_this_turn: bool = True

    def copy(self) -> DevCard:
        return DevCard(card_type=self.card_type, bought_this_turn=self.bought_this_turn)


@dataclass(slots=True)
class PlayerState:
    player_id: int
    resources: dict[Resource, int] = field(
        default_factory=lambda: {r: 0 for r in Resource}
    )
    dev_hand: list[DevCard] = field(default_factory=list)
    played_knights: int = 0
    played_progress_count: int = 0
    revealed_vp_cards: int = 0
    settlements_remaining: int = SETTLEMENTS_PER_PLAYER
    cities_remaining: int = CITIES_PER_PLAYER
    roads_remaining: int = ROADS_PER_PLAYER
    settlement_vertices: set[int] = field(default_factory=set)
    city_vertices: set[int] = field(default_factory=set)
    road_edges: set[int] = field(default_factory=set)
    has_played_dev_card_this_turn: bool = False

    def resource_card_count(self) -> int:
        return sum(self.resources.values())

    def copy(self) -> PlayerState:
        return PlayerState(
            player_id=self.player_id,
            resources=dict(self.resources),
            dev_hand=[c.copy() for c in self.dev_hand],
            played_knights=self.played_knights,
            played_progress_count=self.played_progress_count,
            revealed_vp_cards=self.revealed_vp_cards,
            settlements_remaining=self.settlements_remaining,
            cities_remaining=self.cities_remaining,
            roads_remaining=self.roads_remaining,
            settlement_vertices=set(self.settlement_vertices),
            city_vertices=set(self.city_vertices),
            road_edges=set(self.road_edges),
            has_played_dev_card_this_turn=self.has_played_dev_card_this_turn,
        )


@dataclass(slots=True)
class TradeOffer:
    proposer: int
    give: dict[Resource, int]
    receive: dict[Resource, int]
    # The original proposer, when this offer is a counter-offer -- None for a
    # fresh ProposeTrade. Also the depth bound: a counter-offer may not itself
    # be countered (see engine/game.py's _trade_response_legal).
    counter_of: int | None = None

    def copy(self) -> TradeOffer:
        return TradeOffer(
            proposer=self.proposer,
            give=dict(self.give),
            receive=dict(self.receive),
            counter_of=self.counter_of,
        )


@dataclass(slots=True)
class GameState:
    board: Board
    players: list[PlayerState]
    current_player: int
    phase: Phase
    dev_deck: list[DevCardType]
    bank: dict[Resource, int]
    rng: random.Random

    robber_return_phase: Phase = Phase.MAIN
    pending_discards: list[int] = field(default_factory=list)
    discard_amounts: dict[int, int] = field(default_factory=dict)

    trade_offer: TradeOffer | None = None
    trade_responders: list[int] = field(default_factory=list)

    longest_road_owner: int | None = None
    largest_army_owner: int | None = None

    setup_sequence: list[int] = field(default_factory=list)
    setup_position: int = 0
    last_settlement_vertex: int | None = None

    winner: int | None = None
    dice_roll: tuple[int, int] | None = None

    def copy(self) -> GameState:
        rng_clone = random.Random()
        rng_clone.setstate(self.rng.getstate())
        return GameState(
            board=self.board.copy(),
            players=[p.copy() for p in self.players],
            current_player=self.current_player,
            phase=self.phase,
            dev_deck=list(self.dev_deck),
            bank=dict(self.bank),
            rng=rng_clone,
            robber_return_phase=self.robber_return_phase,
            pending_discards=list(self.pending_discards),
            discard_amounts=dict(self.discard_amounts),
            trade_offer=self.trade_offer.copy()
            if self.trade_offer is not None
            else None,
            trade_responders=list(self.trade_responders),
            longest_road_owner=self.longest_road_owner,
            largest_army_owner=self.largest_army_owner,
            setup_sequence=list(self.setup_sequence),
            setup_position=self.setup_position,
            last_settlement_vertex=self.last_settlement_vertex,
            winner=self.winner,
            dice_roll=self.dice_roll,
        )


def acting_player(state: GameState) -> int:
    """Return the player index who must act now.

    Distinct from ``state.current_player`` (the turn player): during DISCARD
    it is the next player still owing a discard, and during
    AWAIT_TRADE_RESPONSE it is the next player still owed a response.
    """
    if state.phase == Phase.DISCARD:
        return state.pending_discards[0]
    if state.phase == Phase.AWAIT_TRADE_RESPONSE:
        return state.trade_responders[0]
    return state.current_player
