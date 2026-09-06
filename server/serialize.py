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
from engine.game import victory_points
from engine.state import GameState, PlayerState, acting_player


def _cube_to_list(hexagon: Cube) -> list[int]:
    return list(hexagon)


def _resources_to_dict(resources: dict[Resource, int]) -> dict[str, int]:
    return {r.name: resources[r] for r in Resource}


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
    trade_offer = None
    if state.trade_offer is not None:
        # A domestic trade offer is announced to every player by the real
        # rules (decision 5) -- unlike a hand, it is never hidden information,
        # so it is not redacted by viewer.
        trade_offer = {
            "proposer": state.trade_offer.proposer,
            "give": {r.name: c for r, c in state.trade_offer.give.items()},
            "receive": {r.name: c for r, c in state.trade_offer.receive.items()},
        }
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


def serialize_action(action: Action, index: int) -> dict[str, Any]:
    """One legal action as an indexed, render-hinted dict.

    Field names/values come straight from the action dataclass — vertex/edge/
    hex ids need no lookup. ``ProposeTrade``'s empty-``give``/``receive``
    sentinel (``engine/game.py``'s open-ended trade affordance) is flagged
    with ``open_ended: true`` so the client knows to open a bundle form
    instead of just posting the index back.
    """
    fields = {
        f.name: _serialize_field_value(getattr(action, f.name))
        for f in dataclasses.fields(action)
    }
    kind = type(action).__name__
    result: dict[str, Any] = {"index": index, "kind": kind, **fields}
    if kind == "ProposeTrade" and not fields["give"] and not fields["receive"]:
        result["open_ended"] = True
    return result


def serialize_legal_actions(actions: list[Action]) -> list[dict[str, Any]]:
    return [serialize_action(a, i) for i, a in enumerate(actions)]
