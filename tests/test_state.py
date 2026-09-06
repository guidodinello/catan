"""GameState.copy() completeness and seed+action-sequence determinism.

Rule under test: copy() must fork every mutable field (including the RNG
stream) rather than alias it, and replaying the same seed + action sequence
must always reach the same final state (dice, dev-deck order, and robber
steals all draw from state.rng, so this also guards RNG placement).
"""

import dataclasses

from engine.actions import BuyDevCard, EndTurn, RollDice
from engine.game import CatanGame
from engine.state import GameState, PlayerState


def _mutable_field_names(instance: GameState | PlayerState) -> list[str]:
    names = []
    for f in dataclasses.fields(instance):
        value = getattr(instance, f.name)
        if isinstance(value, list | dict | set):
            names.append(f.name)
    return names


def _mutate(value: object) -> None:
    if isinstance(value, list):
        value.append(value[0] if value else 0)
    elif isinstance(value, set):
        value.add(-999)
    elif isinstance(value, dict):
        for k in value:
            value[k] = value[k]
            break


def test_copy_forks_every_mutable_field_no_aliasing() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=7)
    copy = state.copy()

    for name in _mutable_field_names(state):
        original_value = getattr(state, name)
        copied_value = getattr(copy, name)
        assert original_value is not copied_value, f"GameState.{name} is aliased"

    # nested PlayerState fields
    for orig_player, copy_player in zip(state.players, copy.players, strict=True):
        assert orig_player is not copy_player
        for name in _mutable_field_names(orig_player):
            original_value = getattr(orig_player, name)
            copied_value = getattr(copy_player, name)
            assert original_value is not copied_value, f"PlayerState.{name} is aliased"


def test_mutating_copy_leaves_original_untouched() -> None:
    """Fresh copy per field so structural mutations (e.g. list.append) don't
    corrupt later assertions in this same test."""
    game = CatanGame(num_players=3)
    state = game.reset(seed=7)

    for name in _mutable_field_names(state):
        copy = state.copy()
        before = repr(getattr(state, name))
        try:
            _mutate(getattr(copy, name))
        except IndexError, StopIteration:
            continue
        after = repr(getattr(state, name))
        assert before == after, f"mutating copy.{name} affected the original"

    for player_idx in range(len(state.players)):
        for name in _mutable_field_names(state.players[player_idx]):
            copy = state.copy()
            before = repr(getattr(state.players[player_idx], name))
            try:
                _mutate(getattr(copy.players[player_idx], name))
            except IndexError, StopIteration:
                continue
            after = repr(getattr(state.players[player_idx], name))
            assert before == after, (
                f"mutating copy player's {name} affected the original"
            )


def test_copy_forks_rng_stream_not_aliased() -> None:
    """The clone must be an independent stream: advancing one must not
    advance or otherwise affect the other's internal state."""
    game = CatanGame(num_players=3)
    state = game.reset(seed=7)
    copy = state.copy()
    copy_state_before = copy.rng.getstate()

    assert state.rng is not copy.rng
    for _ in range(5):
        state.rng.random()
    assert copy.rng.getstate() == copy_state_before


def _sample_action_sequence(game: CatanGame, state: GameState) -> list:
    """Deterministically drive a few turns: setup, a roll, a dev-card buy."""
    actions_taken = []
    steps = 0
    while state.phase.name.startswith("SETUP") and steps < 200:
        legal = game.legal_actions(state)
        action = legal[0]
        game.apply_action(state, action)
        actions_taken.append(action)
        steps += 1

    # Roll dice for a few turns, buying a dev card when affordable.
    for _ in range(6):
        legal = game.legal_actions(state)
        roll_action = next((a for a in legal if isinstance(a, RollDice)), None)
        if roll_action is not None:
            game.apply_action(state, roll_action)
            actions_taken.append(roll_action)
        legal = game.legal_actions(state)
        buy_action = next((a for a in legal if isinstance(a, BuyDevCard)), None)
        if buy_action is not None:
            game.apply_action(state, buy_action)
            actions_taken.append(buy_action)
        legal = game.legal_actions(state)
        end_action = next((a for a in legal if isinstance(a, EndTurn)), None)
        if end_action is not None:
            game.apply_action(state, end_action)
            actions_taken.append(end_action)
    return actions_taken


def test_same_seed_and_actions_produce_identical_final_state() -> None:
    game_a = CatanGame(num_players=3)
    game_b = CatanGame(num_players=3)
    state_a = game_a.reset(seed=42)
    state_b = game_b.reset(seed=42)

    actions = _sample_action_sequence(game_a, state_a)
    for action in actions:
        game_b.apply_action(state_b, action)

    assert state_a.board.terrain == state_b.board.terrain
    assert state_a.board.tokens == state_b.board.tokens
    assert state_a.board.port_types == state_b.board.port_types
    assert state_a.board.robber_hex == state_b.board.robber_hex
    assert state_a.bank == state_b.bank
    assert state_a.dev_deck == state_b.dev_deck
    assert state_a.current_player == state_b.current_player
    assert state_a.phase == state_b.phase
    for pa, pb in zip(state_a.players, state_b.players, strict=True):
        assert pa.resources == pb.resources
        assert pa.settlement_vertices == pb.settlement_vertices
        assert pa.road_edges == pb.road_edges


def test_reset_is_reflectively_reachable() -> None:
    """Sanity check that PlayerState itself is a plain dataclass we can reflect over."""
    p = PlayerState(player_id=0)
    assert dataclasses.is_dataclass(p)
