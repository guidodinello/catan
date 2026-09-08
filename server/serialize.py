"""``GameState``/``Board``/``Action`` -> JSON-safe dict, with per-viewer redaction.

Pure functions only, no FastAPI imports (``docs/plans/gui-web-frontend.md``'s
Phase 1) so they're testable directly against ``engine/`` types. Board
geometry (``engine.board.GEOMETRY``) is RNG-free and identical across every
game, so it gets its own serializer (``serialize_geometry``) meant to be
called once per game, not once per turn.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any

from engine.actions import Action
from engine.board import GEOMETRY, Board, Cube, Resource
from engine.game import (
    CITY_COST,
    DEV_CARD_COST,
    ROAD_COST,
    SETTLEMENT_COST,
    victory_points,
)
from engine.state import GameState, PlayerState, TradeOffer, acting_player
from server.bots import TrailEntry


def _cube_to_list(hexagon: Cube) -> list[int]:
    return list(hexagon)


def _resources_to_dict(resources: dict[Resource, int]) -> dict[str, int]:
    return {r.name: resources[r] for r in Resource}


def _sparse_resources_to_dict(resources: dict[Resource, int]) -> dict[str, int]:
    """Like ``_resources_to_dict``, but for a sparse cost dict (e.g.
    ``ROAD_COST`` has no ``WOOL``/``ORE`` entries at all) -- iterating all of
    ``Resource`` the way ``_resources_to_dict`` does would ``KeyError`` on a
    missing resource, and a ``{"WOOL": 0}`` entry would carry no information
    anyway.
    """
    return {r.name: n for r, n in resources.items()}


def serialize_build_costs() -> dict[str, dict[str, int]]:
    """Serialize the fixed build costs. RNG-free and identical for every
    game, like ``serialize_geometry`` -- meant to be called once, not once
    per turn/game, and viewer-independent (build costs are public rules, not
    per-player state).
    """
    return {
        "ROAD": _sparse_resources_to_dict(ROAD_COST),
        "SETTLEMENT": _sparse_resources_to_dict(SETTLEMENT_COST),
        "CITY": _sparse_resources_to_dict(CITY_COST),
        "DEV_CARD": _sparse_resources_to_dict(DEV_CARD_COST),
    }


def serialize_geometry() -> dict[str, Any]:
    """Serialize the fixed, RNG-free board topology. Call once per game."""
    hexes = [
        {
            "hex": _cube_to_list(h),
            "vertex_ids": list(GEOMETRY.hex_vertices[h]),
            "edge_ids": list(GEOMETRY.hex_edges[h]),
        }
        for h in GEOMETRY.land_hexes
    ]
    vertices = [
        {
            "vertex_id": vertex_id,
            "hexes": [_cube_to_list(h) for h in GEOMETRY.vertex_hexes[vertex_id]],
            "neighbors": sorted(GEOMETRY.vertex_neighbors[vertex_id]),
            "edges": sorted(GEOMETRY.vertex_edges[vertex_id]),
        }
        for vertex_id in range(len(GEOMETRY.vertex_hexes))
    ]
    edges = [
        {
            "edge_id": edge_id,
            "vertices": list(v1_v2),
            "hexes": [_cube_to_list(h) for h in GEOMETRY.edge_hexes[edge_id]],
        }
        for edge_id, v1_v2 in enumerate(GEOMETRY.edge_vertices)
    ]
    return {
        "land_hexes": [_cube_to_list(h) for h in GEOMETRY.land_hexes],
        "water_hexes": [_cube_to_list(h) for h in GEOMETRY.water_hexes],
        "hexes": hexes,
        "vertices": vertices,
        "edges": edges,
        "port_locations": list(GEOMETRY.port_locations),
    }


def _serialize_board(board: Board) -> dict[str, Any]:
    terrain = [
        {"hex": _cube_to_list(h), "terrain": t.name} for h, t in board.terrain.items()
    ]
    tokens = [
        {"hex": _cube_to_list(h), "token": token} for h, token in board.tokens.items()
    ]
    port_types = [
        {"edge_id": edge_id, "port_type": port_type.name}
        for edge_id, port_type in board.port_types.items()
    ]
    return {
        "terrain": terrain,
        "tokens": tokens,
        "port_types": port_types,
        "robber_hex": _cube_to_list(board.robber_hex),
    }


def _serialize_player(
    player: PlayerState, state: GameState, *, reveal: bool
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "player_id": player.player_id,
        "settlement_vertices": sorted(player.settlement_vertices),
        "city_vertices": sorted(player.city_vertices),
        "road_edges": sorted(player.road_edges),
        "settlements_remaining": player.settlements_remaining,
        "cities_remaining": player.cities_remaining,
        "roads_remaining": player.roads_remaining,
        "played_knights": player.played_knights,
        "played_progress_count": player.played_progress_count,
        "revealed_vp_cards": player.revealed_vp_cards,
        "has_played_dev_card_this_turn": player.has_played_dev_card_this_turn,
        "victory_points": victory_points(state, player.player_id),
    }
    if reveal:
        entry["resources"] = _resources_to_dict(player.resources)
        entry["dev_hand"] = [
            {"card_type": c.card_type.name, "bought_this_turn": c.bought_this_turn}
            for c in player.dev_hand
        ]
    else:
        entry["resource_count"] = player.resource_card_count()
        entry["dev_card_count"] = len(player.dev_hand)
    return entry


def _serialize_trade_offer(offer: TradeOffer) -> dict[str, Any]:
    # A domestic trade offer is announced to every player by the real rules
    # (decision 5) -- unlike a hand, it is never hidden information, so it's
    # never redacted by viewer, whether it's the live offer on `state`
    # (player_view) or one already resolved (serialize_trail_entry).
    return {
        "proposer": offer.proposer,
        "give": {r.name: c for r, c in offer.give.items()},
        "receive": {r.name: c for r, c in offer.receive.items()},
    }


def player_view(state: GameState, viewer: int | None) -> dict[str, Any]:
    """JSON-safe projection of ``state`` for one viewing seat.

    ``viewer=None`` is a spectator/debug view that reveals every hand;
    a seat index sees its own ``resources``/``dev_hand`` in full and every
    other seat's hand only as ``resource_count``/``dev_card_count``.
    """
    players = [
        _serialize_player(p, state, reveal=viewer is None or p.player_id == viewer)
        for p in state.players
    ]
    trade_offer = (
        _serialize_trade_offer(state.trade_offer)
        if state.trade_offer is not None
        else None
    )
    return {
        "phase": state.phase.name,
        "current_player": state.current_player,
        "acting_player": acting_player(state),
        "dice_roll": list(state.dice_roll) if state.dice_roll is not None else None,
        "bank": _resources_to_dict(state.bank),
        "board": _serialize_board(state.board),
        "players": players,
        "longest_road_owner": state.longest_road_owner,
        "largest_army_owner": state.largest_army_owner,
        "pending_discards": list(state.pending_discards),
        "discard_amounts": [
            {"player_id": pid, "amount": amount}
            for pid, amount in state.discard_amounts.items()
        ],
        "trade_offer": trade_offer,
        "trade_responders": list(state.trade_responders),
        "winner": state.winner,
    }


def _serialize_field_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {(k.name if isinstance(k, Enum) else k): v for k, v in value.items()}
    return value


def _action_fields(
    action: Action, *, exclude: frozenset[str] = frozenset()
) -> dict[str, Any]:
    """Field name -> JSON-safe value for every dataclass field on ``action``,
    minus ``exclude`` -- shared by ``serialize_action`` (indexed, for the
    legal-actions list) and ``serialize_trail_entry`` (unindexed, for a
    public record of an already-applied action) so the two never drift on
    what a given action kind's fields actually are.
    """
    return {
        f.name: _serialize_field_value(getattr(action, f.name))
        for f in dataclasses.fields(action)
        if f.name not in exclude
    }


def serialize_action(action: Action, index: int) -> dict[str, Any]:
    """One legal action as an indexed, render-hinted dict.

    Field names/values come straight from the action dataclass — vertex/edge/
    hex ids need no lookup. ``ProposeTrade``'s empty-``give``/``receive``
    sentinel (``engine/game.py``'s open-ended trade affordance) is flagged
    with ``open_ended: true`` so the client knows to open a bundle form
    instead of just posting the index back.
    """
    fields = _action_fields(action)
    kind = type(action).__name__
    result: dict[str, Any] = {"index": index, "kind": kind, **fields}
    if kind == "ProposeTrade" and not fields["give"] and not fields["receive"]:
        result["open_ended"] = True
    return result


def serialize_legal_actions(actions: list[Action]) -> list[dict[str, Any]]:
    return [serialize_action(a, i) for i, a in enumerate(actions)]


# Fields to omit from an already-applied action's public trail entry, for
# action kinds whose fields would otherwise leak information the rest of
# the redaction model treats as private. Discard.resources is the one case
# today: which specific cards a player discarded isn't exposed anywhere
# else in the wire protocol (only the aggregate discard_amounts -- how many
# each player *owes*, not what they discarded), so it's redacted here too
# for consistency, unlike every other action kind's fields (all of which
# are already public knowledge in the real game -- a placed settlement, a
# rolled die, a declared Monopoly resource, an announced trade bundle...).
_TRAIL_REDACTED_FIELDS: dict[str, frozenset[str]] = {
    "Discard": frozenset({"resources"}),
}


def serialize_trail_entry(entry: TrailEntry) -> dict[str, Any]:
    """One already-applied action (``server/bots.py``'s ``TrailEntry``) as a
    public record: who acted, what kind of action, its (possibly redacted)
    fields, and -- only for ``RollDice`` -- the actual roll plus who gained
    which resources from it, since ``RollDice`` itself carries no fields to
    serialize. ``production``'s player-id keys become strings, since JSON
    object keys always are. ``trade_offer`` is set only for ``AcceptTrade``/
    ``RejectTrade`` -- neither carries fields of its own, so the deal (or
    rejected offer) they were responding to is otherwise invisible; it's the
    live ``state.trade_offer`` from just before this response cleared it.
    """
    kind = type(entry.action).__name__
    exclude = _TRAIL_REDACTED_FIELDS.get(kind, frozenset())
    fields = _action_fields(entry.action, exclude=exclude)
    result: dict[str, Any] = {"player_id": entry.player_id, "kind": kind, **fields}
    if entry.dice_roll is not None:
        result["dice_roll"] = list(entry.dice_roll)
    if entry.production:
        result["production"] = {
            str(player_id): {r.name: amount for r, amount in gains.items()}
            for player_id, gains in entry.production.items()
        }
    if entry.trade_offer is not None:
        result["trade_offer"] = _serialize_trade_offer(entry.trade_offer)
    return result


def serialize_trail(entries: list[TrailEntry]) -> list[dict[str, Any]]:
    return [serialize_trail_entry(e) for e in entries]
