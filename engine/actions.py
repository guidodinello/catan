"""Tagged union of Catan actions as frozen dataclasses.

One class per action type; ``Action`` is the union alias used by
``engine.game``. Resource multisets (``give``/``receive``) are plain
``dict[Resource, int]`` with only positive counts present.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .board import Cube, Resource


@dataclass(frozen=True, slots=True)
class PlaceSettlement:
    vertex_id: int


@dataclass(frozen=True, slots=True)
class PlaceRoad:
    edge_id: int


@dataclass(frozen=True, slots=True)
class PlaceCity:
    vertex_id: int


@dataclass(frozen=True, slots=True)
class RollDice:
    pass


@dataclass(frozen=True, slots=True)
class Discard:
    resources: dict[Resource, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MoveRobber:
    hex_id: Cube


@dataclass(frozen=True, slots=True)
class StealFrom:
    player_idx: int


@dataclass(frozen=True, slots=True)
class BuyDevCard:
    pass


@dataclass(frozen=True, slots=True)
class PlayKnight:
    pass


@dataclass(frozen=True, slots=True)
class PlayRoadBuilding:
    edge_id_1: int
    edge_id_2: int


@dataclass(frozen=True, slots=True)
class PlayYearOfPlenty:
    resource_1: Resource
    resource_2: Resource


@dataclass(frozen=True, slots=True)
class PlayMonopoly:
    resource: Resource


@dataclass(frozen=True, slots=True)
class PlayVictoryPoint:
    pass


@dataclass(frozen=True, slots=True)
class TradeBank:
    give: dict[Resource, int]
    receive: dict[Resource, int]


@dataclass(frozen=True, slots=True)
class TradePort:
    give: dict[Resource, int]
    receive: dict[Resource, int]


@dataclass(frozen=True, slots=True)
class ProposeTrade:
    give: dict[Resource, int]
    receive: dict[Resource, int]


@dataclass(frozen=True, slots=True)
class AcceptTrade:
    pass


@dataclass(frozen=True, slots=True)
class RejectTrade:
    pass


@dataclass(frozen=True, slots=True)
class CounterTrade:
    give: dict[Resource, int]
    receive: dict[Resource, int]


@dataclass(frozen=True, slots=True)
class EndTurn:
    pass


Action = (
    PlaceSettlement
    | PlaceRoad
    | PlaceCity
    | RollDice
    | Discard
    | MoveRobber
    | StealFrom
    | BuyDevCard
    | PlayKnight
    | PlayRoadBuilding
    | PlayYearOfPlenty
    | PlayMonopoly
    | PlayVictoryPoint
    | TradeBank
    | TradePort
    | ProposeTrade
    | AcceptTrade
    | RejectTrade
    | CounterTrade
    | EndTurn
)
