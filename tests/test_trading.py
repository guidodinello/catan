"""Trading rules: 4:1 bank, 3:1 generic / 2:1 specific port (gated by
building ownership on the port vertex), domestic trade restricted to the
current turn's player, no like-for-like, no gifts, no dev-card trades.

Rule under test (p. 4; Almanac p. 9, p. 14): 4:1 is always available; 3:1
generic and 2:1 specific ports require a settlement/city on a vertex
bordering that port, and a 2:1 port grants no better rate on other
resources; domestic trade is only ever with the current turn's player; both
sides of any trade must be non-empty and must not share a resource type.
"""

from engine.actions import AcceptTrade, ProposeTrade, RejectTrade
from engine.board import GEOMETRY, PortType, Resource
from engine.game import (
    MAX_TRADE_OFFER_SIDE,
    CatanGame,
    IllegalActionError,
    _apply_trade_bank,
    _apply_trade_port,
    _propose_trade_actions,
)
from engine.state import Phase


def test_four_to_one_bank_trade_always_available() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    state.phase = Phase.MAIN
    p = state.current_player
    player = state.players[p]
    for r in Resource:
        player.resources[r] = 0
    player.resources[Resource.LUMBER] = 4

    _apply_trade_bank(state, p, {Resource.LUMBER: 4}, {Resource.ORE: 1})
    assert player.resources[Resource.LUMBER] == 0
    assert player.resources[Resource.ORE] == 1


def test_bank_trade_rejects_non_four_to_one_ratio() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    p = state.current_player
    state.players[p].resources[Resource.LUMBER] = 3
    with_error = False
    try:
        _apply_trade_bank(state, p, {Resource.LUMBER: 3}, {Resource.ORE: 1})
    except IllegalActionError:
        with_error = True
    assert with_error


def test_port_trade_requires_building_on_port_vertex() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    p = state.current_player
    state.players[p].resources[Resource.LUMBER] = 3
    with_error = False
    try:
        _apply_trade_port(state, p, {Resource.LUMBER: 3}, {Resource.ORE: 1})
    except IllegalActionError:
        with_error = True
    assert with_error


def test_specific_port_gives_two_to_one_but_not_for_other_resources() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    p = state.current_player
    player = state.players[p]
    lumber_port_edge = next(
        e for e, t in state.board.port_types.items() if t is PortType.LUMBER
    )
    v = GEOMETRY.edge_vertices[lumber_port_edge][0]
    player.settlement_vertices.add(v)

    player.resources[Resource.LUMBER] = 2
    _apply_trade_port(state, p, {Resource.LUMBER: 2}, {Resource.ORE: 1})
    assert player.resources[Resource.ORE] == 1

    # no better rate for a different resource without its own port
    player.resources[Resource.WOOL] = 2
    with_error = False
    try:
        _apply_trade_port(state, p, {Resource.WOOL: 2}, {Resource.ORE: 1})
    except IllegalActionError:
        with_error = True
    assert with_error


def test_generic_port_gives_three_to_one() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    p = state.current_player
    player = state.players[p]
    generic_edge = next(
        e for e, t in state.board.port_types.items() if t is PortType.GENERIC
    )
    v = GEOMETRY.edge_vertices[generic_edge][0]
    player.settlement_vertices.add(v)

    player.resources[Resource.GRAIN] = 3
    _apply_trade_port(state, p, {Resource.GRAIN: 3}, {Resource.ORE: 1})
    assert player.resources[Resource.ORE] == 1


def test_no_like_for_like_trade() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    p = state.current_player
    state.players[p].resources[Resource.LUMBER] = 8
    with_error = False
    try:
        _apply_trade_bank(state, p, {Resource.LUMBER: 4}, {Resource.LUMBER: 1})
    except IllegalActionError:
        with_error = True
    assert with_error


def test_no_gift_trade_requires_both_sides_non_empty() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    p = state.current_player
    with_error = False
    try:
        _apply_trade_bank(state, p, {}, {Resource.ORE: 1})
    except IllegalActionError:
        with_error = True
    assert with_error


def test_propose_trade_actions_is_a_single_open_ended_affordance() -> None:
    """legal_actions() offers one sentinel, not a pre-enumeration -- the
    give/receive bundle is constructed by the caller (see engine.game's
    _propose_trade_actions docstring)."""
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    p = state.current_player
    player = state.players[p]
    for r in Resource:
        player.resources[r] = 0

    assert _propose_trade_actions(state, p) == []

    player.resources[Resource.LUMBER] = 1
    actions = _propose_trade_actions(state, p)
    assert actions == [ProposeTrade(give={}, receive={})]


def test_propose_trade_supports_full_multi_resource_bundles() -> None:
    """Real Catan trades are routinely multi-resource on both sides (e.g.
    2 lumber + 1 brick for 1 ore) -- this must not be restricted to a single
    resource type per side."""
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    state.phase = Phase.MAIN
    p = state.current_player
    player = state.players[p]
    player.resources[Resource.LUMBER] = 2
    player.resources[Resource.BRICK] = 1

    game.apply_action(
        state,
        ProposeTrade(
            give={Resource.LUMBER: 2, Resource.BRICK: 1},
            receive={Resource.ORE: 1, Resource.GRAIN: 1},
        ),
    )
    assert state.phase == Phase.AWAIT_TRADE_RESPONSE
    offer = state.trade_offer
    assert offer is not None
    assert offer.give == {Resource.LUMBER: 2, Resource.BRICK: 1}
    assert offer.receive == {Resource.ORE: 1, Resource.GRAIN: 1}


def test_propose_trade_rejects_offer_exceeding_per_side_card_cap() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    state.phase = Phase.MAIN
    p = state.current_player
    player = state.players[p]
    player.resources[Resource.LUMBER] = MAX_TRADE_OFFER_SIDE + 1

    with_error = False
    try:
        game.apply_action(
            state,
            ProposeTrade(
                give={Resource.LUMBER: MAX_TRADE_OFFER_SIDE + 1},
                receive={Resource.ORE: 1},
            ),
        )
    except IllegalActionError:
        with_error = True
    assert with_error


def test_domestic_trade_only_with_current_player() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    state.phase = Phase.MAIN
    p = state.current_player
    other = (p + 1) % 3
    state.players[p].resources[Resource.LUMBER] = 2

    game.apply_action(
        state, ProposeTrade(give={Resource.LUMBER: 1}, receive={Resource.ORE: 1})
    )
    assert state.phase == Phase.AWAIT_TRADE_RESPONSE
    # only players other than the proposer are asked to respond
    assert p not in state.trade_responders
    assert other in state.trade_responders


def test_accept_trade_swaps_resources_and_reject_moves_to_next_responder() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    state.phase = Phase.MAIN
    p = state.current_player
    others = [i for i in range(3) if i != p]
    state.players[p].resources[Resource.LUMBER] = 1
    for o in others:
        state.players[o].resources[Resource.ORE] = 1

    game.apply_action(
        state, ProposeTrade(give={Resource.LUMBER: 1}, receive={Resource.ORE: 1})
    )
    first_responder = state.trade_responders[0]
    game.apply_action(state, RejectTrade())
    assert first_responder not in state.trade_responders
    assert state.phase == Phase.AWAIT_TRADE_RESPONSE

    second_responder = state.trade_responders[0]
    game.apply_action(state, AcceptTrade())
    assert state.phase == Phase.MAIN
    assert state.players[p].resources[Resource.ORE] == 1
    assert state.players[second_responder].resources[Resource.LUMBER] == 1
