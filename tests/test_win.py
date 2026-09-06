"""10+ VP win condition, checked only during the holder's own turn.

Rule under test (p. 5; Almanac p. 7, p. 14): if you have >=10 VP during your
turn the game ends immediately; reaching 10 on someone else's turn (e.g.
because Longest Road/Largest Army just moved to you) does not end the game
until your own next turn. VP sources: settlement 1, city 2, Longest Road 2,
Largest Army 2, VP card 1.
"""

from engine.game import CatanGame, _check_win, victory_points
from engine.state import Phase


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
