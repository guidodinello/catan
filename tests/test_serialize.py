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

from engine.actions import (
    AcceptTrade,
    Discard,
    PlaceSettlement,
    ProposeTrade,
    RejectTrade,
    RollDice,
)
from engine.board import (
    GEOMETRY,
    NUM_EDGES,
    NUM_LAND_HEXES,
    NUM_PORTS,
    NUM_VERTICES,
    Resource,
)
from engine.game import CatanGame, _check_win, victory_points
from engine.state import DevCard, DevCardType, Phase, TradeOffer
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


def test_opponent_hand_stays_redacted_until_the_win() -> None:
    """Load-bearing redaction check for the hidden-VP auto-win fix
    (engine/game.py's ``true_victory_points``/``_check_win``): a hidden VP
    card must not leak through ``player_view`` for a non-viewing seat one
    action before the win, but the win itself is allowed to reveal it --
    that's the real rule (revealing is showing proof once the game ends).
    """
    game = CatanGame(num_players=3)
    state = game.reset(seed=5)
    winner_idx = 1
    state.current_player = winner_idx
    winner = state.players[winner_idx]
    winner.settlement_vertices = {1, 2, 3}  # 3
    winner.city_vertices = {4}  # +2 = 5
    winner.revealed_vp_cards = 2  # +2 = 7 public
    winner.dev_hand.append(
        DevCard(card_type=DevCardType.KNIGHT, bought_this_turn=False)
    )
    winner.dev_hand.append(
        DevCard(card_type=DevCardType.VICTORY_POINT, bought_this_turn=False)
    )  # +1 hidden = 8 true; not yet a winning true total
    state.phase = Phase.MAIN

    # One action short of a win: the true total (8) hasn't crossed 10 yet, so
    # nothing has been revealed and the seat still looks fully redacted.
    view = player_view(state, viewer=0)
    entry = next(e for e in view["players"] if e["player_id"] == winner_idx)
    assert "dev_hand" not in entry
    assert "resources" not in entry
    assert entry["dev_card_count"] == 2
    assert entry["victory_points"] == 7  # public tally only

    # Cross the true-VP threshold: two more revealed VP cards, then the win
    # check that engine/game.py's _build_settlement/_build_road/etc. all run.
    winner.revealed_vp_cards += 2  # public 7 -> 9
    _check_win(state)
    assert state.phase == Phase.GAME_OVER
    assert state.winner == winner_idx

    view = player_view(state, viewer=0)
    entry = next(e for e in view["players"] if e["player_id"] == winner_idx)
    assert entry["victory_points"] == 10  # public tally now reflects the win
    assert entry["dev_card_count"] == 1  # KNIGHT only; VP card was revealed
    assert "dev_hand" not in entry  # still redacted for a non-owner seat
    assert "resources" not in entry


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


def test_serialize_trail_entry_includes_production_with_string_player_id_keys() -> None:
    entry = TrailEntry(
        player_id=0,
        action=RollDice(),
        dice_roll=(3, 4),
        production={0: {Resource.WOOL: 1}, 2: {Resource.ORE: 2, Resource.BRICK: 1}},
    )
    result = serialize_trail(entries=[entry])[0]
    assert result["production"] == {
        "0": {"WOOL": 1},
        "2": {"ORE": 2, "BRICK": 1},
    }


def test_serialize_trail_entry_omits_production_when_nothing_was_gained() -> None:
    # apply_and_record always sets `production` to a dict (possibly empty)
    # for a RollDice entry, e.g. a 7 or a roll matching no settled hex --
    # the wire format should omit the key entirely rather than send `{}`.
    entry = TrailEntry(player_id=0, action=RollDice(), dice_roll=(3, 4), production={})
    result = serialize_trail(entries=[entry])[0]
    assert "production" not in result


def test_serialize_trail_entry_includes_trade_offer_for_accept_trade() -> None:
    offer = TradeOffer(proposer=1, give={Resource.LUMBER: 1}, receive={Resource.ORE: 1})
    entry = TrailEntry(player_id=0, action=AcceptTrade(), trade_offer=offer)
    result = serialize_trail(entries=[entry])[0]
    assert result == {
        "player_id": 0,
        "kind": "AcceptTrade",
        "trade_offer": {
            "proposer": 1,
            "give": {"LUMBER": 1},
            "receive": {"ORE": 1},
        },
    }


def test_serialize_trail_entry_includes_trade_offer_for_reject_trade() -> None:
    offer = TradeOffer(proposer=1, give={Resource.LUMBER: 1}, receive={Resource.ORE: 1})
    entry = TrailEntry(player_id=2, action=RejectTrade(), trade_offer=offer)
    result = serialize_trail(entries=[entry])[0]
    assert result["trade_offer"]["proposer"] == 1


def test_serialize_trail_entry_omits_trade_offer_for_unrelated_kinds() -> None:
    entry = TrailEntry(player_id=0, action=PlaceSettlement(vertex_id=1))
    result = serialize_trail(entries=[entry])[0]
    assert "trade_offer" not in result


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
