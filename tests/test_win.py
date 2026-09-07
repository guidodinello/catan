"""10+ VP win condition, checked only during the holder's own turn.

Rule under test (p. 5; Almanac p. 7, p. 14): if you have >=10 VP during your
turn the game ends immediately; reaching 10 on someone else's turn (e.g.
because Longest Road/Largest Army just moved to you) does not end the game
until your own next turn. VP sources: settlement 1, city 2, Longest Road 2,
Largest Army 2, VP card 1.
"""

from engine.board import Resource
from engine.game import (
    CatanGame,
    _buy_dev_card,
    _check_win,
    true_victory_points,
    victory_points,
)
from engine.state import DevCard, DevCardType, Phase


def test_victory_points_tally_all_sources() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    p = 0
    player = state.players[p]
    player.settlement_vertices = {1, 2, 3}  # 3
    player.city_vertices = {4}  # +2 = 5
    player.revealed_vp_cards = 1  # +1 = 6
    state.longest_road_owner = p  # +2 = 8
    state.largest_army_owner = p  # +2 = 10
    assert victory_points(state, p) == 10


def test_ten_vp_on_holders_turn_ends_the_game() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    state.current_player = 1
    state.players[1].settlement_vertices = set(range(10))
    state.phase = Phase.MAIN

    _check_win(state)
    assert state.phase == Phase.GAME_OVER
    assert state.winner == 1


def test_ten_vp_on_non_holders_turn_does_not_end_game() -> None:
    """Player 2 has 10 VP (e.g. via an award that just moved to them), but it
    is player 0's turn -- the game must not end yet."""
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    state.current_player = 0
    state.players[0].settlement_vertices = set()  # 0 VP for the turn player
    state.players[2].settlement_vertices = set(range(10))  # 10 VP, not their turn
    state.phase = Phase.MAIN

    _check_win(state)
    assert state.phase != Phase.GAME_OVER
    assert state.winner is None


def test_game_ends_at_start_of_the_actual_holders_turn() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    state.players[1].settlement_vertices = set(range(10))
    state.phase = Phase.MAIN
    state.current_player = 0

    from engine.game import _end_turn

    _end_turn(state)  # advances to player 1's turn
    assert state.current_player == 1
    assert state.phase == Phase.GAME_OVER
    assert state.winner == 1


def test_buying_the_winning_vp_card_ends_the_game_immediately() -> None:
    """9 public points + a bought VICTORY_POINT card should win on the spot,
    via _buy_dev_card's own _check_win -- no separate PlayVictoryPoint."""
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    p = 1
    player = state.players[p]
    player.settlement_vertices = {1, 2, 3}  # 3
    player.city_vertices = {4}  # +2 = 5
    player.revealed_vp_cards = 2  # +2 = 7
    state.longest_road_owner = p  # +2 = 9
    state.current_player = p
    state.phase = Phase.MAIN
    player.resources[Resource.ORE] = 1
    player.resources[Resource.WOOL] = 1
    player.resources[Resource.GRAIN] = 1
    # _buy_dev_card pops the last element -- stack it as the winning card.
    state.dev_deck.append(DevCardType.VICTORY_POINT)

    _buy_dev_card(state, p)

    assert state.phase == Phase.GAME_OVER
    assert state.winner == p


def test_hidden_vp_card_wins_at_start_of_the_holders_own_turn() -> None:
    """Hidden analogue of test_game_ends_at_start_of_the_actual_holders_turn:
    9 public + 1 unrevealed VP card must not end the game on someone else's
    turn, but must end it the instant it becomes the holder's own turn."""
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    holder = 1
    player = state.players[holder]
    player.settlement_vertices = set(range(9))  # 9 public
    player.dev_hand.append(
        DevCard(card_type=DevCardType.VICTORY_POINT, bought_this_turn=False)
    )
    state.phase = Phase.MAIN
    state.current_player = 0

    _check_win(state)
    assert state.phase != Phase.GAME_OVER  # not the holder's turn yet

    from engine.game import _end_turn

    _end_turn(state)  # advances to the holder's turn
    assert state.current_player == holder
    assert state.phase == Phase.GAME_OVER
    assert state.winner == holder


def test_hidden_vp_card_does_not_win_on_another_players_turn() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    state.current_player = 0
    state.players[0].settlement_vertices = set()  # 0 VP for the turn player
    holder = state.players[2]
    holder.settlement_vertices = set(range(9))  # 9 public
    holder.dev_hand.append(
        DevCard(card_type=DevCardType.VICTORY_POINT, bought_this_turn=False)
    )  # 10 true, not their turn
    state.phase = Phase.MAIN

    _check_win(state)
    assert state.phase != Phase.GAME_OVER
    assert state.winner is None


def test_win_reveals_the_winners_vp_cards() -> None:
    """On a win, the winner's VP cards move dev_hand -> revealed_vp_cards
    (so the public victory_points tally is accurate the instant the game
    ends), while a loser's hidden cards are left untouched."""
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    winner_idx = 1
    winner = state.players[winner_idx]
    winner.settlement_vertices = set(range(9))  # 9 public
    winner.dev_hand.append(
        DevCard(card_type=DevCardType.KNIGHT, bought_this_turn=False)
    )
    winner.dev_hand.append(
        DevCard(card_type=DevCardType.VICTORY_POINT, bought_this_turn=False)
    )  # 10 true

    loser = state.players[2]
    loser.dev_hand.append(
        DevCard(card_type=DevCardType.VICTORY_POINT, bought_this_turn=False)
    )

    state.current_player = winner_idx
    state.phase = Phase.MAIN

    _check_win(state)

    assert state.phase == Phase.GAME_OVER
    assert state.winner == winner_idx
    assert winner.revealed_vp_cards == 1
    assert [c.card_type for c in winner.dev_hand] == [DevCardType.KNIGHT]
    assert victory_points(state, winner_idx) == 10

    # The loser's hidden card is untouched -- only the winner is revealed.
    assert loser.revealed_vp_cards == 0
    assert len(loser.dev_hand) == 1


def test_true_victory_points_counts_a_card_bought_this_turn() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    p = 0
    player = state.players[p]
    player.settlement_vertices = {1}  # 1 public
    player.dev_hand.append(
        DevCard(card_type=DevCardType.VICTORY_POINT, bought_this_turn=True)
    )
    assert victory_points(state, p) == 1  # public tally excludes it
    assert true_victory_points(state, p) == 2  # true total counts it
