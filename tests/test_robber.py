"""The seven, robber movement, and stealing.

Rule under test (p. 5; Almanac p. 11): rolling a 7 means no production; each
player with MORE THAN 7 resource cards (dev cards never count) must discard
floor(n/2); the robber must move to a different hex (desert included); after
moving, if 1+ opponents have a building on the new hex, steal exactly 1
random card from a chosen victim -- no steal if nobody has a building there,
and a victim with 0 cards yields nothing.
"""

from typing import Any

from engine.actions import Action, Discard, MoveRobber, RollDice, StealFrom
from engine.board import GEOMETRY, Resource
from engine.game import CatanGame, IllegalActionError
from engine.state import DevCard, DevCardType, GameState, Phase


class _FixedDiceRng:
    """Minimal stand-in for random.Random that always rolls a 7 (3+4)."""

    def __init__(self) -> None:
        self._calls = 0

    def randint(self, a: int, b: int) -> int:
        self._calls += 1
        return 3 if self._calls % 2 == 1 else 4

    def choice(self, seq: list[Any]) -> Any:
        return seq[0]

    def shuffle(self, seq: list[Any]) -> None:
        return None


def _force_seven(game: CatanGame, state: GameState) -> None:
    state.rng = _FixedDiceRng()  # type: ignore[assignment]  # test double, not a real Random
    game.apply_action(state, RollDice())


def _discard_options(actions: list[Action]) -> list[Discard]:
    return [a for a in actions if isinstance(a, Discard)]


def _move_robber_hexes(actions: list[Action]) -> list[tuple[int, int, int]]:
    return [a.hex_id for a in actions if isinstance(a, MoveRobber)]


def test_discard_threshold_counts_only_resource_cards() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    player = state.players[0]
    for r in Resource:
        player.resources[r] = 0
    player.resources[Resource.WOOL] = 5
    player.dev_hand = [
        DevCard(DevCardType.KNIGHT, bought_this_turn=False) for _ in range(4)
    ]
    # 5 resource cards + 4 dev cards: must NOT be forced to discard (<=7 resources)
    assert player.resource_card_count() == 5


def test_discard_required_is_floor_half_and_must_hold_more_than_seven() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    state.current_player = 0
    for i, count in enumerate([9, 7, 3]):
        player = state.players[i]
        for r in Resource:
            player.resources[r] = 0
        player.resources[Resource.WOOL] = count

    state.phase = Phase.ROLL
    _force_seven(game, state)

    assert state.phase == Phase.DISCARD
    assert set(state.pending_discards) == {0}  # only player with >7 cards
    assert state.discard_amounts[0] == 4  # floor(9/2)


def test_discard_action_enforces_exact_total_and_subset_of_hand() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    for r in Resource:
        state.players[0].resources[r] = 0
    state.players[0].resources[Resource.WOOL] = 9
    state.pending_discards = [0]
    state.discard_amounts = {0: 4}
    state.phase = Phase.DISCARD

    legal = _discard_options(game.legal_actions(state))
    assert all(sum(a.resources.values()) == 4 for a in legal)
    assert any(a.resources == {Resource.WOOL: 4} for a in legal)

    with_error = False
    try:
        game.apply_action(state, Discard(resources={Resource.WOOL: 3}))
    except IllegalActionError:
        with_error = True
    assert with_error

    game.apply_action(state, Discard(resources={Resource.WOOL: 4}))
    assert state.phase == Phase.MOVE_ROBBER
    assert state.players[0].resources[Resource.WOOL] == 5


def test_robber_must_move_to_a_different_hex() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    state.phase = Phase.MOVE_ROBBER
    current_hex = state.board.robber_hex

    legal = _move_robber_hexes(game.legal_actions(state))
    assert current_hex not in set(legal)

    with_error = False
    try:
        game.apply_action(state, MoveRobber(current_hex))
    except IllegalActionError:
        with_error = True
    assert with_error


def test_steal_only_from_players_with_buildings_on_new_hex() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    for p in state.players:
        p.settlement_vertices.clear()
        p.city_vertices.clear()
    target_hex = next(h for h in GEOMETRY.land_hexes if h != state.board.robber_hex)
    v = GEOMETRY.hex_vertices[target_hex][0]
    state.players[1].settlement_vertices.add(v)

    state.phase = Phase.MOVE_ROBBER
    state.robber_return_phase = Phase.MAIN
    state.current_player = 0
    game.apply_action(state, MoveRobber(target_hex))

    assert state.phase == Phase.STEAL
    legal = game.legal_actions(state)
    assert legal == [StealFrom(1)]


def test_no_steal_when_no_opponent_building_on_new_hex() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    for p in state.players:
        p.settlement_vertices.clear()
        p.city_vertices.clear()
    target_hex = next(h for h in GEOMETRY.land_hexes if h != state.board.robber_hex)

    state.phase = Phase.MOVE_ROBBER
    state.robber_return_phase = Phase.MAIN
    state.current_player = 0
    game.apply_action(state, MoveRobber(target_hex))

    assert state.phase == Phase.MAIN


def test_steal_transfers_exactly_one_card() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    for r in Resource:
        state.players[0].resources[r] = 0
        state.players[1].resources[r] = 0
    state.players[1].resources[Resource.ORE] = 3
    state.players[1].settlement_vertices.add(
        GEOMETRY.hex_vertices[state.board.robber_hex][0]
    )
    state.phase = Phase.STEAL
    state.robber_return_phase = Phase.MAIN
    state.current_player = 0

    game.apply_action(state, StealFrom(1))

    assert sum(state.players[0].resources.values()) == 1
    assert sum(state.players[1].resources.values()) == 2
    assert state.phase == Phase.MAIN


def test_steal_from_empty_handed_victim_is_a_no_op() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)
    for r in Resource:
        state.players[0].resources[r] = 0
        state.players[1].resources[r] = 0
    state.players[1].settlement_vertices.add(
        GEOMETRY.hex_vertices[state.board.robber_hex][0]
    )
    state.phase = Phase.STEAL
    state.robber_return_phase = Phase.MAIN
    state.current_player = 0

    game.apply_action(state, StealFrom(1))
    assert sum(state.players[0].resources.values()) == 0
