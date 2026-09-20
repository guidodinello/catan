"""Rule under test: the atom encoding is an exact, lossless re-spelling of
``legal_actions()`` -- no legal action becomes unreachable, and no reachable
atom path produces something the engine would reject.

This is the property the whole Phase 5 adapter rests on. README decision 10
says the engine is never bent to suit RL, which only holds if the encoding
layer can express what the engine actually offers. These tests are the proof,
and they are written against the real engine with no mocks, per the repo's
existing test conventions.

Stdlib-only on purpose (``rl.action_space`` imports no numpy), so this module
collects and runs in the default CI job as well as the RL one.
"""

from __future__ import annotations

import random
from enum import Enum

from engine.actions import (
    Action,
    Discard,
    PlayRoadBuilding,
    ProposeTrade,
)
from engine.board import Resource
from engine.game import CatanGame, _multiset_combinations, _road_building_pairs
from engine.state import DevCard, DevCardType, GameState, Phase, acting_player
from rl.action_space import (
    EDGE_BASE,
    N_ATOMS,
    OPENER_ROAD_BUILDING,
    TRADE_COUNTER,
    TRADE_PROPOSE,
    ActionComposer,
    atom_sequence,
    is_trade_proposal,
)

SEEDS = range(12)


def _key(action: Action) -> str:
    """Order-insensitive, sortable identity for an action.

    Actions are frozen dataclasses holding *mutable* dicts, so they are
    unhashable and cannot go in a set directly. Rendering to a string also
    sidesteps comparing heterogeneous field types when sorting.
    """
    parts: list[str] = [type(action).__name__]
    for field in getattr(action, "__slots__", ()):
        value = getattr(action, field)
        if isinstance(value, dict):
            rendered = ",".join(
                f"{r.name}:{c}"
                for r, c in sorted(value.items(), key=lambda kv: kv[0].name)
            )
        elif isinstance(value, Enum):
            rendered = value.name
        else:
            rendered = repr(value)
        parts.append(f"{field}={rendered}")
    return "|".join(parts)


def _reachable_actions(
    legal: list[Action], seat: int, num_players: int, *, allow_trades: bool = False
) -> list[Action]:
    """Every action reachable by walking unmasked atoms from an empty buffer."""
    found: list[Action] = []

    def walk(prefix: tuple[int, ...]) -> None:
        composer = ActionComposer(allow_trade_proposals=allow_trades)
        for atom in prefix:
            emitted = composer.push(atom, legal, seat, num_players)
            assert emitted is None, "prefix should not complete early"
        mask = composer.mask(legal, seat, num_players)
        for atom, allowed in enumerate(mask):
            if not allowed:
                continue
            probe = ActionComposer(allow_trade_proposals=allow_trades)
            for earlier in prefix:
                probe.push(earlier, legal, seat, num_players)
            emitted = probe.push(atom, legal, seat, num_players)
            if emitted is None:
                walk((*prefix, atom))
            else:
                found.append(emitted)

    walk(())
    return found


def _play_random_game(seed: int, *, step_budget: int = 4000) -> list[GameState]:
    """Play one full random game, yielding the state at every decision point."""
    game = CatanGame(num_players=4)
    state = game.reset(seed=seed)
    rng = random.Random(seed + 9_000)
    seen: list[GameState] = []
    steps = 0
    while not game.is_terminal(state) and steps < step_budget:
        seen.append(state.copy())
        legal = game.legal_actions(state)
        action = rng.choice(legal)
        if isinstance(action, ProposeTrade) and not action.give:
            action = Discard()  # never applied; skip the sentinel branch
            action = rng.choice([a for a in legal if not is_trade_proposal(a)])
        game.apply_action(state, action)
        steps += 1
    return seen


def test_every_legal_action_round_trips_through_its_atom_sequence() -> None:
    for seed in SEEDS:
        for state in _play_random_game(seed):
            game = CatanGame(num_players=4)
            legal = game.legal_actions(state)
            seat = acting_player(state)
            for action in legal:
                if is_trade_proposal(action):
                    continue
                composer = ActionComposer()
                emitted: Action | None = None
                for atom in atom_sequence(action, seat, 4):
                    emitted = composer.push(atom, legal, seat, 4)
                assert emitted is not None, f"{action!r} never completed"
                assert _key(emitted) == _key(action), (
                    f"{action!r} did not round-trip in {state.phase}"
                )
                assert not composer.in_progress


def test_atom_walk_reaches_exactly_the_legal_action_set() -> None:
    """The other direction: nothing reachable is illegal, nothing legal is lost."""
    for seed in SEEDS:
        for state in _play_random_game(seed):
            game = CatanGame(num_players=4)
            legal = game.legal_actions(state)
            seat = acting_player(state)
            expected = sorted(_key(a) for a in legal if not is_trade_proposal(a))
            reached = sorted(_key(a) for a in _reachable_actions(legal, seat, 4))
            assert reached == expected, f"mismatch in {state.phase}"


def test_mask_is_never_empty_while_the_engine_offers_a_non_trade_action() -> None:
    """Masking the trade sentinels off can never strand the learner.

    EndTurn in MAIN, RollDice in ROLL and RejectTrade in AWAIT_TRADE_RESPONSE
    are unconditionally legal, so removing the two open-ended trade
    affordances always leaves something to do.
    """
    for seed in SEEDS:
        for state in _play_random_game(seed):
            game = CatanGame(num_players=4)
            legal = game.legal_actions(state)
            if all(is_trade_proposal(a) for a in legal):
                continue
            composer = ActionComposer()
            mask = composer.mask(legal, acting_player(state), 4)
            assert any(mask), f"empty mask in {state.phase}"


def test_trade_sentinels_are_masked_off_by_default_and_reachable_when_enabled() -> None:
    state = _discard_free_main_state_with_cards()
    game = CatanGame(num_players=4)
    legal = game.legal_actions(state)
    assert any(isinstance(a, ProposeTrade) for a in legal), "fixture lost its sentinel"

    seat = acting_player(state)
    masked_off = ActionComposer().mask(legal, seat, 4)
    assert not masked_off[TRADE_PROPOSE]
    assert not masked_off[TRADE_COUNTER]

    enabled = ActionComposer(allow_trade_proposals=True).mask(legal, seat, 4)
    assert enabled[TRADE_PROPOSE]


def test_discard_composition_reproduces_the_engines_multiset_enumeration() -> None:
    """The 197-wide case: a 5-wide resource head must cover every combination."""
    state = _oversized_hand_discard_state()
    game = CatanGame(num_players=4)
    legal = game.legal_actions(state)
    assert all(isinstance(a, Discard) for a in legal)
    assert len(legal) > 100, f"fixture is too small to be interesting: {len(legal)}"

    seat = acting_player(state)
    hand = state.players[seat].resources
    required = state.discard_amounts[seat]
    expected = sorted(
        _key(Discard(resources=c)) for c in _multiset_combinations(hand, required)
    )
    reached = sorted(_key(a) for a in _reachable_actions(legal, seat, 4))
    assert reached == expected


def test_road_building_second_edge_mask_depends_on_the_first_edge() -> None:
    """The case ``MultiDiscrete`` could not express.

    ``_road_building_pairs`` recomputes legality with ``e1`` already placed, so
    the legal set for slot 2 genuinely differs per slot-1 choice. A head whose
    sub-masks are all computed before sampling would have to over-approximate.
    """
    state = _road_building_state()
    game = CatanGame(num_players=4)
    legal = game.legal_actions(state)
    pairs = _road_building_pairs(state, state.current_player)
    assert len(pairs) > 2, "fixture produced too few pairs to compare"

    seat = acting_player(state)
    slot_two_by_first: dict[int, frozenset[int]] = {}
    for first in {e1 for e1, _ in pairs}:
        composer = ActionComposer()
        assert composer.push(OPENER_ROAD_BUILDING, legal, seat, 4) is None
        assert composer.push(EDGE_BASE + first, legal, seat, 4) is None
        mask = composer.mask(legal, seat, 4)
        slot_two_by_first[first] = frozenset(
            atom - EDGE_BASE for atom, ok in enumerate(mask) if ok
        )
        expected = frozenset(e2 for e1, e2 in pairs if e1 == first)
        assert slot_two_by_first[first] == expected

    assert len(set(slot_two_by_first.values())) > 1, (
        "slot-2 mask was identical for every slot-1 choice -- the fixture does "
        "not actually exercise the dependency"
    )


def test_road_building_pairs_are_ordered_so_both_orderings_are_reachable() -> None:
    state = _road_building_state()
    game = CatanGame(num_players=4)
    legal = game.legal_actions(state)
    reached = {
        (a.edge_id_1, a.edge_id_2)
        for a in _reachable_actions(legal, acting_player(state), 4)
        if isinstance(a, PlayRoadBuilding)
    }
    assert reached == set(_road_building_pairs(state, state.current_player))


def test_atom_indices_stay_inside_the_declared_head_width() -> None:
    for seed in SEEDS:
        for state in _play_random_game(seed):
            game = CatanGame(num_players=4)
            seat = acting_player(state)
            for action in game.legal_actions(state):
                for atom in atom_sequence(action, seat, 4):
                    assert 0 <= atom < N_ATOMS


def test_steal_targets_are_encoded_relative_to_the_acting_seat() -> None:
    """Seat-relative so one policy transfers across seats."""
    from engine.actions import StealFrom

    for seat in range(4):
        for victim in range(4):
            if victim == seat:
                continue
            (atom,) = atom_sequence(StealFrom(victim), seat, 4)
            other_seat = (seat + 1) % 4
            other_victim = (victim + 1) % 4
            (other_atom,) = atom_sequence(StealFrom(other_victim), other_seat, 4)
            assert atom == other_atom


# ---------------------------------------------------------------------------
# Fixtures: states built by hand on the real engine, no mocks
# ---------------------------------------------------------------------------


def _advance_through_setup(game: CatanGame, state: GameState) -> GameState:
    rng = random.Random(7)
    while state.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        game.apply_action(state, rng.choice(game.legal_actions(state)))
    return state


def _discard_free_main_state_with_cards() -> GameState:
    game = CatanGame(num_players=4)
    state = game.reset(seed=3)
    _advance_through_setup(game, state)
    state.phase = Phase.MAIN
    player = state.players[state.current_player]
    player.resources[Resource.LUMBER] = 2
    player.resources[Resource.ORE] = 1
    return state


def _oversized_hand_discard_state() -> GameState:
    game = CatanGame(num_players=4)
    state = game.reset(seed=5)
    _advance_through_setup(game, state)
    victim = state.current_player
    player = state.players[victim]
    for resource in Resource:
        player.resources[resource] = 3
    player.resources[Resource.LUMBER] = 4
    state.phase = Phase.DISCARD
    state.pending_discards = [victim]
    state.discard_amounts = {victim: player.resource_card_count() // 2}
    return state


def _road_building_state() -> GameState:
    game = CatanGame(num_players=4)
    state = game.reset(seed=11)
    _advance_through_setup(game, state)
    state.phase = Phase.MAIN
    player = state.players[state.current_player]
    player.dev_hand.append(
        DevCard(card_type=DevCardType.ROAD_BUILDING, bought_this_turn=False)
    )
    player.has_played_dev_card_this_turn = False
    return state
