"""Setup-phase rules: starting player, snake placement order, resource grant,
and the setup road constraint.

Rule under test (Almanac p. 12): highest 2d6 roll starts; round 1 goes
clockwise, round 2 counterclockwise, giving a snake order (1,2,3,4,4,3,2,1 for
4 players, 1,2,3,3,2,1 for 3); the second settlement immediately grants 1
resource per adjacent LAND, non-desert hex; the setup road must touch the
settlement just placed; the distance rule applies during setup exactly as
elsewhere.
"""

from engine.actions import Action, PlaceRoad, PlaceSettlement
from engine.board import GEOMETRY, Terrain
from engine.game import CatanGame
from engine.state import GameState, Phase


def _settlement_vertices(actions: list[Action]) -> list[int]:
    return [a.vertex_id for a in actions if isinstance(a, PlaceSettlement)]


def _road_edges(actions: list[Action]) -> list[int]:
    return [a.edge_id for a in actions if isinstance(a, PlaceRoad)]


def test_snake_order_for_four_players() -> None:
    game = CatanGame(num_players=4)
    state = game.reset(seed=3)
    start = state.setup_sequence[0]
    expected = [start, (start + 1) % 4, (start + 2) % 4, (start + 3) % 4]
    expected = expected + list(reversed(expected))
    assert state.setup_sequence == expected


def test_snake_order_for_three_players() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=3)
    start = state.setup_sequence[0]
    expected = [start, (start + 1) % 3, (start + 2) % 3]
    expected = expected + list(reversed(expected))
    assert state.setup_sequence == expected


def test_starting_player_places_second_settlement_last() -> None:
    game = CatanGame(num_players=4)
    state = game.reset(seed=5)
    start = state.setup_sequence[0]
    assert state.setup_sequence[-1] == start
    assert state.setup_sequence[3] == state.setup_sequence[4]


def _place_first_settlement_and_road(game: CatanGame, state: GameState) -> int:
    vertex = _settlement_vertices(game.legal_actions(state))[0]
    game.apply_action(state, PlaceSettlement(vertex))
    edge = _road_edges(game.legal_actions(state))[0]
    game.apply_action(state, PlaceRoad(edge))
    return vertex


def test_second_settlement_grants_resource_per_adjacent_land_nondesert_hex() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=11)

    # Drive through round 1 for all players.
    for _ in range(3):
        _place_first_settlement_and_road(game, state)
    assert state.phase == Phase.SETUP_SETTLEMENT

    # Round 2, first player's second settlement.
    vertex = _settlement_vertices(game.legal_actions(state))[0]
    player_idx = state.current_player
    before = sum(state.players[player_idx].resources.values())
    game.apply_action(state, PlaceSettlement(vertex))

    expected_grant = sum(
        1
        for h in GEOMETRY.vertex_hexes[vertex]
        if GEOMETRY.is_land(h) and state.board.terrain[h] is not Terrain.DESERT
    )
    after = sum(state.players[player_idx].resources.values())
    assert after - before == expected_grant


def test_setup_road_must_touch_settlement_just_placed() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=13)
    vertex = _settlement_vertices(game.legal_actions(state))[0]
    game.apply_action(state, PlaceSettlement(vertex))

    road_edges = _road_edges(game.legal_actions(state))
    assert road_edges, "setup road phase must offer at least one legal road"
    for edge_id in road_edges:
        assert vertex in GEOMETRY.edge_vertices[edge_id]


def test_distance_rule_applies_during_setup() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=17)
    vertex = _settlement_vertices(game.legal_actions(state))[0]
    game.apply_action(state, PlaceSettlement(vertex))
    edge = _road_edges(game.legal_actions(state))[0]
    game.apply_action(state, PlaceRoad(edge))

    # Next player's settlement options must exclude the occupied vertex and
    # all of its neighbors.
    forbidden = {vertex} | GEOMETRY.vertex_neighbors[vertex]
    offered = set(_settlement_vertices(game.legal_actions(state)))
    assert offered.isdisjoint(forbidden)
