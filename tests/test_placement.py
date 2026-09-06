"""Placement rules outside setup: distance rule, road/settlement connectivity,
opponent-blocking, one road per edge, and city-upgrade piece recycling.

Rule under test (Almanac pp. 7, 10, 11): the distance rule applies regardless
of owner; a new settlement must connect to the builder's own road; a road
must connect to the builder's own road/settlement/city and is not extendable
"through" an opponent's building; only one road per edge ever; a city can
only upgrade the builder's own settlement in place and returns the settlement
piece to the supply, so a new settlement can be built again if pieces remain.
"""

from engine.actions import PlaceCity, PlaceRoad, PlaceSettlement, RollDice
from engine.board import GEOMETRY, Resource
from engine.game import (
    CITY_COST,
    ROAD_COST,
    SETTLEMENT_COST,
    CatanGame,
    IllegalActionError,
)
from engine.state import GameState, Phase


def _advance_to_main(game: CatanGame, state: GameState) -> None:
    """Finish setup (2 players' worth is enough for these unit tests) and
    fast-forward to the acting player's MAIN phase without rolling a 7."""
    while state.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        legal = game.legal_actions(state)
        action = legal[0]
        game.apply_action(state, action)
    while state.phase == Phase.ROLL:
        # force a non-7 roll by retrying with a fresh rng draw is not
        # possible deterministically here; just roll and handle 7 if it
        # happens by resolving discard/robber/steal trivially.
        game.apply_action(state, RollDice())
        while state.phase in (Phase.DISCARD, Phase.MOVE_ROBBER, Phase.STEAL):
            legal = game.legal_actions(state)
            game.apply_action(state, legal[0])
        if state.phase == Phase.MAIN:
            return


def _give_resources(
    state: GameState, player_idx: int, cost: dict[Resource, int]
) -> None:
    player = state.players[player_idx]
    for r, c in cost.items():
        player.resources[r] += c
        state.bank[r] -= c


def test_distance_rule_blocks_neighboring_vertex_regardless_of_owner() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=2)
    _advance_to_main(game, state)
    p = state.current_player
    player = state.players[p]

    # Pick one of the player's own settlements (from setup) and confirm its
    # neighbors are excluded from legal settlement vertices for every player,
    # including the owner.
    assert player.settlement_vertices, "setup should have placed settlements"
    occupied = next(iter(player.settlement_vertices))
    for neighbor in GEOMETRY.vertex_neighbors[occupied]:
        legal = game.legal_actions(state)
        offered = {a.vertex_id for a in legal if isinstance(a, PlaceSettlement)}
        assert neighbor not in offered


def test_settlement_must_connect_to_own_road_outside_setup() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=4)
    _advance_to_main(game, state)
    p = state.current_player
    _give_resources(state, p, SETTLEMENT_COST)

    legal = game.legal_actions(state)
    settlement_actions = [a for a in legal if isinstance(a, PlaceSettlement)]
    player = state.players[p]
    road_vertices: set[int] = set()
    for e in player.road_edges:
        road_vertices.update(GEOMETRY.edge_vertices[e])
    for action in settlement_actions:
        assert action.vertex_id in road_vertices


def test_only_one_road_per_edge() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=6)
    _advance_to_main(game, state)
    p = state.current_player
    _give_resources(state, p, ROAD_COST)
    legal = [a for a in game.legal_actions(state) if isinstance(a, PlaceRoad)]
    edge = legal[0].edge_id
    game.apply_action(state, PlaceRoad(edge))

    _give_resources(state, p, ROAD_COST)
    legal_after = [a for a in game.legal_actions(state) if isinstance(a, PlaceRoad)]
    assert edge not in {a.edge_id for a in legal_after}
    with_error = False
    try:
        game.apply_action(state, PlaceRoad(edge))
    except IllegalActionError:
        with_error = True
    assert with_error


def test_road_cannot_extend_through_opponent_settlement() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=9)
    _advance_to_main(game, state)
    p = state.current_player
    other = (p + 1) % 3

    # Build a road for p up to some vertex v, then place an opponent
    # settlement on v (bypassing legality checks directly for test setup),
    # and confirm p can no longer extend a road "through" v to a further edge.
    _give_resources(state, p, ROAD_COST)
    legal = [a for a in game.legal_actions(state) if isinstance(a, PlaceRoad)]
    edge = legal[0].edge_id
    game.apply_action(state, PlaceRoad(edge))
    v1, v2 = GEOMETRY.edge_vertices[edge]
    player = state.players[p]
    # pick whichever endpoint is not already an owned building
    far_vertex = v1 if v1 not in player.settlement_vertices else v2

    # place an opponent building directly on far_vertex (distance rule
    # aside; this is a direct state edit purely to set up the blocking test)
    for neighbor in GEOMETRY.vertex_neighbors[far_vertex]:
        state.players[other].settlement_vertices.discard(neighbor)
    state.players[other].settlement_vertices.add(far_vertex)

    beyond_edges = [e for e in GEOMETRY.vertex_edges[far_vertex] if e != edge]
    _give_resources(state, p, ROAD_COST)
    legal_edges = {
        a.edge_id for a in game.legal_actions(state) if isinstance(a, PlaceRoad)
    }
    for e in beyond_edges:
        other_endpoint = [v for v in GEOMETRY.edge_vertices[e] if v != far_vertex][0]
        connects_via_other_endpoint = (
            other_endpoint
            in {vv for re in player.road_edges for vv in GEOMETRY.edge_vertices[re]}
            or other_endpoint in player.settlement_vertices
        )
        if not connects_via_other_endpoint:
            assert e not in legal_edges


def test_city_upgrade_returns_settlement_piece_and_allows_new_settlement() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=15)
    _advance_to_main(game, state)
    p = state.current_player
    player = state.players[p]

    # Exhaust settlement pieces down to 0 by direct manipulation, keeping one
    # settlement vertex to upgrade.
    settlement_vertex = next(iter(player.settlement_vertices))
    player.settlements_remaining = 0

    _give_resources(state, p, CITY_COST)
    legal = [a for a in game.legal_actions(state) if isinstance(a, PlaceCity)]
    assert any(a.vertex_id == settlement_vertex for a in legal)
    game.apply_action(state, PlaceCity(settlement_vertex))

    assert settlement_vertex in player.city_vertices
    assert settlement_vertex not in player.settlement_vertices
    assert player.settlements_remaining == 1

    # Extend the road network two hops out from the city, clearing any
    # opponent setup pieces in the way (direct state edits purely to
    # construct a scenario with a free vertex to test against) so there is
    # guaranteed to be a distance-rule-legal vertex to build the recycled
    # settlement piece on.
    def owned_by_anyone(e: int) -> bool:
        return any(e in pl.road_edges for pl in state.players)

    e1 = next(
        e for e in GEOMETRY.vertex_edges[settlement_vertex] if not owned_by_anyone(e)
    )
    intermediate = [v for v in GEOMETRY.edge_vertices[e1] if v != settlement_vertex][0]
    e2 = next(
        e
        for e in GEOMETRY.vertex_edges[intermediate]
        if e != e1 and not owned_by_anyone(e)
    )
    far = [v for v in GEOMETRY.edge_vertices[e2] if v != intermediate][0]

    for other in state.players:
        if other is player:
            continue
        other.settlement_vertices.discard(intermediate)
        other.settlement_vertices.discard(far)
        for n in GEOMETRY.vertex_neighbors[far]:
            other.settlement_vertices.discard(n)

    player.road_edges.add(e1)
    player.road_edges.add(e2)
    player.roads_remaining -= 2

    _give_resources(state, p, SETTLEMENT_COST)
    legal_after = [
        a for a in game.legal_actions(state) if isinstance(a, PlaceSettlement)
    ]
    assert legal_after, "settlement should be buildable again after city upgrade"
    assert far in {a.vertex_id for a in legal_after}


def test_city_must_upgrade_own_settlement_only() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=21)
    _advance_to_main(game, state)
    p = state.current_player
    other = (p + 1) % 3
    other_vertex = next(iter(state.players[other].settlement_vertices))

    _give_resources(state, p, CITY_COST)
    with_error = False
    try:
        game.apply_action(state, PlaceCity(other_vertex))
    except IllegalActionError:
        with_error = True
    assert with_error
