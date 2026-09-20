"""Rule under test: the observation carries what this seat may legally see and
nothing more, in a fixed-width seat-relative layout.

The hidden-information tests are the point of this module. Leaking an
opponent's hand composition into the observation produces an agent whose win
rate looks excellent and means nothing, and it is invisible in review -- the
encoder would simply be reading a field it had no business reading. So the
invariance is pinned mechanically, with a negative control proving the test
can actually fail.

``engine/game.py`` states the rule this enforces: the gap between
``victory_points`` and ``true_victory_points`` *is* hidden information.
"""

from __future__ import annotations

import random

import pytest

pytest.importorskip("numpy")

import numpy as np  # noqa: E402

from agents.random_agent import RandomAgent  # noqa: E402
from engine.board import Resource  # noqa: E402
from engine.game import CatanGame  # noqa: E402
from engine.state import (  # noqa: E402
    DevCard,
    DevCardType,
    GameState,
    Phase,
    acting_player,
)
from rl.encoder import OBS_DIM, ObservationEncoder  # noqa: E402


def _mid_game_state(seed: int = 4) -> GameState:
    """A state a few hundred steps into a real random game.

    Driven by ``RandomAgent`` rather than a bare ``rng.choice`` so the
    open-ended ProposeTrade sentinel gets resolved into a real bundle -- the
    engine rejects the sentinel itself, by design.
    """
    game = CatanGame(num_players=4)
    state = game.reset(seed=seed)
    agent = RandomAgent(rng=random.Random(seed + 500))
    steps = 0
    while state.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        actor = acting_player(state)
        game.apply_action(
            state, agent.choose_action(state, game.legal_actions(state), actor)
        )
    while not game.is_terminal(state) and steps < 300:
        actor = acting_player(state)
        game.apply_action(
            state, agent.choose_action(state, game.legal_actions(state), actor)
        )
        steps += 1
    return state


def _encode(state: GameState, seat: int) -> np.ndarray:
    encoder = ObservationEncoder(4)
    encoder.reset(state)
    return encoder.encode(state, seat)


def test_observation_has_the_declared_shape_and_dtype() -> None:
    obs = _encode(_mid_game_state(), seat=0)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32


def test_observation_stays_inside_the_declared_unit_box() -> None:
    """The Box(low=0, high=1) declaration has to be honest, not aspirational."""
    for seed in range(6):
        state = _mid_game_state(seed)
        for seat in range(4):
            obs = _encode(state, seat)
            assert obs.min() >= 0.0, f"seed {seed} seat {seat} min {obs.min()}"
            assert obs.max() <= 1.0, f"seed {seed} seat {seat} max {obs.max()}"
            assert np.isfinite(obs).all()


def test_observation_is_identical_when_only_opponent_hand_composition_differs() -> None:
    """Same hand *size*, different cards: this seat cannot tell, so neither may
    the encoder."""
    state = _mid_game_state()
    seat = 0
    opponent = next(p for p in state.players if p.player_id != seat)
    for resource in Resource:
        opponent.resources[resource] = 0
    opponent.resources[Resource.LUMBER] = 4
    before = _encode(state, seat)

    for resource in Resource:
        opponent.resources[resource] = 0
    opponent.resources[Resource.ORE] = 3
    opponent.resources[Resource.GRAIN] = 1
    after = _encode(state, seat)

    np.testing.assert_array_equal(before, after)


def test_observation_is_identical_when_only_opponent_dev_cards_differ() -> None:
    state = _mid_game_state()
    seat = 0
    opponent = next(p for p in state.players if p.player_id != seat)
    opponent.dev_hand = [
        DevCard(card_type=DevCardType.KNIGHT, bought_this_turn=False),
        DevCard(card_type=DevCardType.KNIGHT, bought_this_turn=False),
    ]
    before = _encode(state, seat)

    opponent.dev_hand = [
        DevCard(card_type=DevCardType.VICTORY_POINT, bought_this_turn=False),
        DevCard(card_type=DevCardType.MONOPOLY, bought_this_turn=True),
    ]
    after = _encode(state, seat)

    np.testing.assert_array_equal(before, after)


def test_observation_is_identical_when_only_the_dev_deck_order_differs() -> None:
    """Deck *size* is public; the order is not."""
    state = _mid_game_state()
    before = _encode(state, seat=0)
    random.Random(1).shuffle(state.dev_deck)
    after = _encode(state, seat=0)
    np.testing.assert_array_equal(before, after)


def test_hidden_victory_point_cards_do_not_move_the_observation() -> None:
    """An unrevealed VP card is worth a point only to ``true_victory_points``,
    which no observation may ever reach."""
    state = _mid_game_state()
    seat = 0
    opponent = next(p for p in state.players if p.player_id != seat)
    opponent.dev_hand = [DevCard(card_type=DevCardType.KNIGHT, bought_this_turn=False)]
    before = _encode(state, seat)
    opponent.dev_hand = [
        DevCard(card_type=DevCardType.VICTORY_POINT, bought_this_turn=False)
    ]
    after = _encode(state, seat)
    np.testing.assert_array_equal(before, after)


def test_public_changes_do_move_the_observation() -> None:
    """Negative control: without this, every invariance test above could pass
    on an encoder that returned a constant."""
    state = _mid_game_state()
    seat = 0
    opponent = next(p for p in state.players if p.player_id != seat)
    before = _encode(state, seat)

    opponent.resources[Resource.LUMBER] += 5  # hand *size* is public
    assert not np.array_equal(before, _encode(state, seat))

    state.players[seat].revealed_vp_cards += 1  # revealed is public
    assert not np.array_equal(before, _encode(state, seat))


def test_own_hand_composition_is_visible_to_its_owner() -> None:
    """The flip side of the invariance: the learner does see its own cards."""
    state = _mid_game_state()
    seat = 0
    me = state.players[seat]
    for resource in Resource:
        me.resources[resource] = 0
    me.resources[Resource.LUMBER] = 3
    before = _encode(state, seat)
    for resource in Resource:
        me.resources[resource] = 0
    me.resources[Resource.ORE] = 3
    assert not np.array_equal(before, _encode(state, seat))


def test_the_learner_always_reads_itself_at_the_same_relative_slot() -> None:
    """Seat relativity: two seats holding identical positions encode alike.

    Built by rotating every player-indexed field of the state by one seat and
    checking the observation is unchanged when the viewing seat rotates too.
    """
    state = _mid_game_state(seed=8)
    rotated = state.copy()
    n = 4
    rotated.players = [state.players[(i - 1) % n].copy() for i in range(n)]
    for i, player in enumerate(rotated.players):
        player.player_id = i
    rotated.current_player = (state.current_player + 1) % n
    if state.longest_road_owner is not None:
        rotated.longest_road_owner = (state.longest_road_owner + 1) % n
    if state.largest_army_owner is not None:
        rotated.largest_army_owner = (state.largest_army_owner + 1) % n
    rotated.setup_sequence = [(s + 1) % n for s in state.setup_sequence]
    rotated.pending_discards = [(s + 1) % n for s in state.pending_discards]
    rotated.discard_amounts = {(s + 1) % n: v for s, v in state.discard_amounts.items()}
    rotated.trade_responders = [(s + 1) % n for s in state.trade_responders]
    if rotated.trade_offer is not None:
        rotated.trade_offer.proposer = (rotated.trade_offer.proposer + 1) % n

    np.testing.assert_array_equal(_encode(state, seat=0), _encode(rotated, seat=1))


def test_the_composition_buffer_is_visible_in_the_observation() -> None:
    """The mask says what may come next; the buffer block says what has already
    been committed to, which the value function needs and the mask does not
    convey."""
    from rl.action_space import EDGE_BASE, OPENER_ROAD_BUILDING

    state = _mid_game_state()
    encoder = ObservationEncoder(4)
    encoder.reset(state)
    empty = encoder.encode(state, 0)
    mid = encoder.encode(state, 0, buffer_prefix=(OPENER_ROAD_BUILDING, EDGE_BASE + 5))
    assert not np.array_equal(empty, mid)
