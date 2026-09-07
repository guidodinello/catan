"""``server/serialize.py``: GameState/Board/Action -> JSON-safe dict.

Rule under test: every serializer output must survive a ``json.dumps``/
``json.loads`` round trip unchanged (no live objects, no non-string dict
keys leaking through), ``player_view``'s redaction must hide a non-viewing
seat's ``resources``/``dev_hand`` behind ``resource_count``/``dev_card_count``
while a spectator (``viewer=None``) or the seat itself sees the real hand, and
``serialize_legal_actions`` must preserve ``legal_actions``' order and flag
the open-ended ``ProposeTrade`` sentinel (``engine/game.py``'s single
empty-give/receive affordance) with ``open_ended: True``.
"""

import json

from engine.actions import Discard, PlaceSettlement, ProposeTrade, RollDice
from engine.board import (
    GEOMETRY,
    NUM_EDGES,
    NUM_LAND_HEXES,
    NUM_PORTS,
    NUM_VERTICES,
    Resource,
)
from engine.game import CatanGame, victory_points
from engine.state import Phase
from server.bots import TrailEntry
from server.serialize import (
    player_view,
    serialize_geometry,
    serialize_legal_actions,
    serialize_trail,
)


def _json_roundtrip(obj: object) -> object:
    return json.loads(json.dumps(obj))


def test_serialize_geometry_counts_match_the_geometry_singleton() -> None:
    geometry = serialize_geometry()
    assert len(geometry["land_hexes"]) == NUM_LAND_HEXES
    assert len(geometry["hexes"]) == NUM_LAND_HEXES
    assert len(geometry["vertices"]) == NUM_VERTICES
    assert len(geometry["edges"]) == NUM_EDGES
    assert len(geometry["port_locations"]) == NUM_PORTS


def test_serialize_geometry_vertex_and_edge_ids_match_geometry_indices() -> None:
    geometry = serialize_geometry()
    for vertex_id, entry in enumerate(geometry["vertices"]):
        assert entry["vertex_id"] == vertex_id
        assert entry["neighbors"] == sorted(GEOMETRY.vertex_neighbors[vertex_id])
        assert entry["edges"] == sorted(GEOMETRY.vertex_edges[vertex_id])
    for edge_id, entry in enumerate(geometry["edges"]):
        assert entry["edge_id"] == edge_id
        assert entry["vertices"] == list(GEOMETRY.edge_vertices[edge_id])


def test_serialize_geometry_is_json_safe_and_idempotent() -> None:
    geometry = serialize_geometry()
    assert _json_roundtrip(geometry) == geometry


def test_player_view_reveals_own_hand_and_redacts_others() -> None:
    game = CatanGame(num_players=4)
    state = game.reset(seed=1)
    view = player_view(state, viewer=0)

    own = view["players"][0]
    assert "resources" in own
    assert "dev_hand" in own
    assert "resource_count" not in own

    for other in view["players"][1:]:
        assert "resources" not in other
        assert "dev_hand" not in other
        other_player = state.players[other["player_id"]]
        assert other["resource_count"] == other_player.resource_card_count()
        assert other["dev_card_count"] == len(other_player.dev_hand)


def test_player_view_spectator_reveals_every_hand() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=2)
    view = player_view(state, viewer=None)
    for entry in view["players"]:
        assert "resources" in entry
        assert "dev_hand" in entry


def test_player_view_victory_points_match_the_engine() -> None:
    game = CatanGame(num_players=4)
    state = game.reset(seed=3)
    view = player_view(state, viewer=None)
    for entry in view["players"]:
        assert entry["victory_points"] == victory_points(state, entry["player_id"])


def test_player_view_acting_player_and_phase_are_exposed() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=4)
    view = player_view(state, viewer=None)
    assert view["phase"] == state.phase.name
    assert view["phase"] == Phase.SETUP_SETTLEMENT.name
    assert view["acting_player"] == state.setup_sequence[0]


def test_player_view_is_json_safe_and_idempotent() -> None:
    game = CatanGame(num_players=4)
    state = game.reset(seed=5)
    view = player_view(state, viewer=1)
    assert _json_roundtrip(view) == view


def test_serialize_legal_actions_preserves_order_and_kind() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=6)
    actions = game.legal_actions(state)
    serialized = serialize_legal_actions(actions)
    assert len(serialized) == len(actions)
    for i, (action, entry) in enumerate(zip(actions, serialized, strict=True)):
        assert entry["index"] == i
        assert entry["kind"] == type(action).__name__


def test_serialize_legal_actions_flags_the_open_ended_propose_trade_sentinel() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=6)
    # Drive to the MAIN phase, which is where the ProposeTrade sentinel
    # appears in legal_actions.
    while state.phase != Phase.MAIN and not game.is_terminal(state):
        actions = game.legal_actions(state)
        game.apply_action(state, actions[0])
    assert state.phase == Phase.MAIN

    serialized = serialize_legal_actions(game.legal_actions(state))
    propose_trade_entries = [e for e in serialized if e["kind"] == "ProposeTrade"]
    assert len(propose_trade_entries) == 1
    entry = propose_trade_entries[0]
    assert entry["open_ended"] is True
    assert entry["give"] == {}
    assert entry["receive"] == {}


def test_serialize_legal_actions_is_json_safe_and_idempotent() -> None:
    game = CatanGame(num_players=4)
    state = game.reset(seed=7)
    serialized = serialize_legal_actions(game.legal_actions(state))
    assert _json_roundtrip(serialized) == serialized


def test_serialize_trail_entry_includes_player_id_kind_and_fields() -> None:
    entry = TrailEntry(player_id=2, action=PlaceSettlement(vertex_id=17))
    result = serialize_trail(entries=[entry])[0]
    assert result == {"player_id": 2, "kind": "PlaceSettlement", "vertex_id": 17}
    assert (
        "index" not in result
    )  # never a legal-action selector, unlike serialize_action


def test_serialize_trail_entry_includes_dice_roll_only_for_roll_dice() -> None:
    roll_entry = TrailEntry(player_id=0, action=RollDice(), dice_roll=(4, 3))
    result = serialize_trail(entries=[roll_entry])[0]
    assert result == {"player_id": 0, "kind": "RollDice", "dice_roll": [4, 3]}

    non_roll_entry = TrailEntry(player_id=0, action=PlaceSettlement(vertex_id=1))
    assert "dice_roll" not in serialize_trail(entries=[non_roll_entry])[0]


def test_serialize_trail_redacts_discards_specific_resources() -> None:
    entry = TrailEntry(
        player_id=1, action=Discard(resources={Resource.LUMBER: 2, Resource.ORE: 1})
    )
    result = serialize_trail(entries=[entry])[0]
    assert result == {"player_id": 1, "kind": "Discard"}


def test_serialize_trail_exposes_a_real_propose_trade_bundle_not_the_sentinel() -> None:
    # Unlike serialize_action's open_ended sentinel flag, a trail entry only
    # ever describes an already-applied (thus fully-specified) action.
    entry = TrailEntry(
        player_id=0,
        action=ProposeTrade(give={Resource.LUMBER: 2}, receive={Resource.ORE: 1}),
    )
    result = serialize_trail(entries=[entry])[0]
    assert result == {
        "player_id": 0,
        "kind": "ProposeTrade",
        "give": {"LUMBER": 2},
        "receive": {"ORE": 1},
    }
    assert "open_ended" not in result


def test_serialize_trail_preserves_order_and_is_json_safe() -> None:
    entries = [
        TrailEntry(player_id=0, action=RollDice(), dice_roll=(2, 5)),
        TrailEntry(player_id=0, action=PlaceSettlement(vertex_id=3)),
        TrailEntry(player_id=1, action=RollDice(), dice_roll=(6, 6)),
    ]
    serialized = serialize_trail(entries)
    assert [e["player_id"] for e in serialized] == [0, 0, 1]
    assert _json_roundtrip(serialized) == serialized
