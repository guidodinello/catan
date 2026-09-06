"""Development card rules: deck composition, one-per-turn limit, timing,
the victory-point exception, and progress-card effects.

Rule under test (Almanac pp. 7-10): exact deck of 14 knight / 2 road building
/ 2 year of plenty / 2 monopoly / 5 victory point = 25; at most one
knight-or-progress-card play per turn; a knight/progress card cannot be
played the turn it was bought, but knights/progress cards CAN be played
before rolling the dice; VP cards are exempt from both restrictions and can
win the game immediately, even the turn they were bought; buying is illegal
once the deck is empty; Road Building places 2 free roads; Year of Plenty
grants 2 bank resources (capped by supply); Monopoly takes a named resource
from every other player.
"""

from engine.actions import (
    BuyDevCard,
    MoveRobber,
    PlayKnight,
    PlayMonopoly,
    PlayRoadBuilding,
    PlayVictoryPoint,
    PlayYearOfPlenty,
)
from engine.board import GEOMETRY, Resource
from engine.game import (
    CatanGame,
    IllegalActionError,
    _play_monopoly,
    _play_year_of_plenty,
)
from engine.state import DEV_DECK_SIZE, DevCard, DevCardType, Phase


def test_dev_deck_has_25_cards_with_correct_composition() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    assert DEV_DECK_SIZE == 25
    assert len(state.dev_deck) == 25
    from collections import Counter

    counts = Counter(state.dev_deck)
    assert counts[DevCardType.KNIGHT] == 14
    assert counts[DevCardType.ROAD_BUILDING] == 2
    assert counts[DevCardType.YEAR_OF_PLENTY] == 2
    assert counts[DevCardType.MONOPOLY] == 2
    assert counts[DevCardType.VICTORY_POINT] == 5


def test_cannot_buy_dev_card_when_deck_empty() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    state.phase = Phase.MAIN
    state.dev_deck = []
    p = state.current_player
    for r, c in {Resource.ORE: 1, Resource.WOOL: 1, Resource.GRAIN: 1}.items():
        state.players[p].resources[r] = c

    legal = game.legal_actions(state)
    assert not any(isinstance(a, BuyDevCard) for a in legal)


def test_cannot_play_knight_bought_same_turn() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    state.phase = Phase.MAIN
    p = state.current_player
    state.players[p].dev_hand = [DevCard(DevCardType.KNIGHT, bought_this_turn=True)]

    legal = game.legal_actions(state)
    assert not any(isinstance(a, PlayKnight) for a in legal)

    with_error = False
    try:
        game.apply_action(state, PlayKnight())
    except IllegalActionError:
        with_error = True
    assert with_error


def test_knight_playable_before_rolling_and_returns_to_roll_phase() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    state.phase = Phase.ROLL
    p = state.current_player
    state.players[p].dev_hand = [DevCard(DevCardType.KNIGHT, bought_this_turn=False)]

    legal = game.legal_actions(state)
    assert any(isinstance(a, PlayKnight) for a in legal)
    game.apply_action(state, PlayKnight())
    assert state.phase == Phase.MOVE_ROBBER
    assert state.robber_return_phase == Phase.ROLL

    current_hex = state.board.robber_hex
    other_hex = next(h for h in GEOMETRY.land_hexes if h != current_hex)
    game.apply_action(state, MoveRobber(other_hex))
    assert state.phase == Phase.ROLL  # no opponents there -> straight back to ROLL


def test_at_most_one_knight_or_progress_card_per_turn() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    state.phase = Phase.MAIN
    p = state.current_player
    state.players[p].dev_hand = [
        DevCard(DevCardType.KNIGHT, bought_this_turn=False),
        DevCard(DevCardType.MONOPOLY, bought_this_turn=False),
    ]
    state.players[p].has_played_dev_card_this_turn = True

    legal = game.legal_actions(state)
    assert not any(
        isinstance(a, PlayKnight | PlayMonopoly | PlayRoadBuilding | PlayYearOfPlenty)
        for a in legal
    )


def test_victory_point_card_exempt_and_can_win_same_turn_bought() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    state.phase = Phase.MAIN
    p = state.current_player
    player = state.players[p]
    player.has_played_dev_card_this_turn = True  # already played a knight this turn
    player.dev_hand = [DevCard(DevCardType.VICTORY_POINT, bought_this_turn=True)]
    # give the player 9 VP from other sources so this VP card wins it
    player.settlement_vertices = set(range(9))

    legal = game.legal_actions(state)
    assert any(isinstance(a, PlayVictoryPoint) for a in legal)
    game.apply_action(state, PlayVictoryPoint())
    assert state.phase == Phase.GAME_OVER
    assert state.winner == p


def test_road_building_places_two_free_roads() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    state.phase = Phase.MAIN
    p = state.current_player
    player = state.players[p]
    # give the player a foothold: one settlement so roads have somewhere to
    # legally start from
    v = next(iter(GEOMETRY.land_hexes))
    start_vertex = GEOMETRY.hex_vertices[v][0]
    player.settlement_vertices.add(start_vertex)
    player.dev_hand = [DevCard(DevCardType.ROAD_BUILDING, bought_this_turn=False)]

    legal = [a for a in game.legal_actions(state) if isinstance(a, PlayRoadBuilding)]
    assert legal, "road building should offer at least one pair of placements"
    action = legal[0]
    resources_before = dict(player.resources)
    roads_before = player.roads_remaining
    game.apply_action(state, action)
    assert player.roads_remaining == roads_before - 2
    assert action.edge_id_1 in player.road_edges
    assert action.edge_id_2 in player.road_edges
    assert player.resources == resources_before  # free
    assert player.has_played_dev_card_this_turn


def test_year_of_plenty_grants_two_resources_capped_by_bank() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    state.phase = Phase.MAIN
    p = state.current_player
    player = state.players[p]
    player.dev_hand = [DevCard(DevCardType.YEAR_OF_PLENTY, bought_this_turn=False)]
    state.bank[Resource.ORE] = 1

    _play_year_of_plenty(state, p, Resource.ORE, Resource.ORE)
    assert player.resources[Resource.ORE] == 1  # capped, only 1 was available
    assert state.bank[Resource.ORE] == 0


def test_monopoly_takes_named_resource_from_all_other_players() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    state.phase = Phase.MAIN
    p = state.current_player
    player = state.players[p]
    player.dev_hand = [DevCard(DevCardType.MONOPOLY, bought_this_turn=False)]
    for i, pl in enumerate(state.players):
        if i != p:
            pl.resources[Resource.BRICK] = 2

    _play_monopoly(state, p, Resource.BRICK)
    assert player.resources[Resource.BRICK] == 4
    for i, pl in enumerate(state.players):
        if i != p:
            assert pl.resources[Resource.BRICK] == 0
