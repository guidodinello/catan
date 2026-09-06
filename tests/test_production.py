"""Resource production on a non-7 roll, including the robber blocking a hex
and the per-resource-type bank shortage rule.

Rule under test (p. 4; Almanac p. 10): each settlement adjacent to a rolled
hex pays 1 card of that hex's resource, each city pays 2; the robber's hex
never produces; if the bank cannot cover everyone's production of a given
resource, nobody gets it that turn UNLESS only one player earned it, in which
case they get whatever the bank has left (any shortfall lost) -- other
resource types are unaffected either way.
"""

from engine.board import GEOMETRY, TERRAIN_RESOURCE, Resource
from engine.game import CatanGame, _produce
from engine.state import GameState


def _blank_state(num_players: int = 3) -> GameState:
    game = CatanGame(num_players=num_players)
    state = game.reset(seed=1)
    # Wipe any setup-phase production/placements for a clean slate.
    for p in state.players:
        p.settlement_vertices.clear()
        p.city_vertices.clear()
        for r in Resource:
            p.resources[r] = 0
    return state


def test_settlement_pays_one_city_pays_two() -> None:
    state = _blank_state()
    hex_with_5 = next(h for h, n in state.board.tokens.items() if n == 5)
    v_settlement, v_city = GEOMETRY.hex_vertices[hex_with_5][:2]
    state.players[0].settlement_vertices.add(v_settlement)
    state.players[1].city_vertices.add(v_city)

    _produce(state, 5)

    resource = TERRAIN_RESOURCE[state.board.terrain[hex_with_5]]
    assert state.players[0].resources[resource] == 1
    assert state.players[1].resources[resource] == 2


def test_robber_hex_produces_nothing() -> None:
    state = _blank_state()
    hex_with_6 = next(h for h, n in state.board.tokens.items() if n == 6)
    v = GEOMETRY.hex_vertices[hex_with_6][0]
    state.players[0].settlement_vertices.add(v)
    state.board.robber_hex = hex_with_6

    _produce(state, 6)
    assert sum(state.players[0].resources.values()) == 0


def test_bank_shortage_single_player_gets_remainder() -> None:
    state = _blank_state()
    hex_with_9 = next(h for h, n in state.board.tokens.items() if n == 9)
    resource = TERRAIN_RESOURCE[state.board.terrain[hex_with_9]]
    v = GEOMETRY.hex_vertices[hex_with_9][0]
    state.players[0].city_vertices.add(v)  # earns 2, but bank only has 1 left
    state.bank[resource] = 1

    _produce(state, 9)
    assert state.players[0].resources[resource] == 1
    assert state.bank[resource] == 0


def test_bank_shortage_multiple_players_get_nothing_but_other_resources_ok() -> None:
    state = _blank_state()
    hex_with_9 = next(h for h, n in state.board.tokens.items() if n == 9)
    resource = TERRAIN_RESOURCE[state.board.terrain[hex_with_9]]
    vertices = GEOMETRY.hex_vertices[hex_with_9]
    state.players[0].settlement_vertices.add(vertices[0])
    state.players[1].settlement_vertices.add(vertices[2])
    state.bank[resource] = 1  # not enough for both players' 1 card each

    other_hex = next(
        h for h, n in state.board.tokens.items() if n == 9 and h != hex_with_9
    )
    other_resource = TERRAIN_RESOURCE[state.board.terrain[other_hex]]
    other_vertex = GEOMETRY.hex_vertices[other_hex][0]
    state.players[2].settlement_vertices.add(other_vertex)

    before_bank = dict(state.bank)
    _produce(state, 9)

    assert state.players[0].resources[resource] == 0
    assert state.players[1].resources[resource] == 0
    assert state.bank[resource] == before_bank[resource]
    if other_resource != resource:
        assert state.players[2].resources[other_resource] == 1
