"""The decision-time search core (``agents/ismcts.py``): no torch, stub evaluator.

The invariants that matter for experiment 011: the search never peeks at
hidden information, leaves the live state untouched, is reproducible from its
seed, and actually lets values override a lopsided prior.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import pytest

from agents.heuristic import HeuristicAgent
from agents.ismcts import (
    ActionKey,
    SearchConfig,
    action_key,
    determinize,
    keyed_actions,
    search,
)
from engine.actions import (
    Action,
    EndTurn,
    PlaceCity,
    PlaceSettlement,
    PlayRoadBuilding,
)
from engine.game import CatanGame, true_victory_points, victory_points
from engine.state import DevCardType, GameState, Phase, acting_player

NUM_PLAYERS = 4
SEAT = 0


class StubEvaluator:
    """Priors from ``prior_of`` (uniform by default); value = public VP/10."""

    def __init__(self, prior_of=None, value_of=None) -> None:  # type: ignore[no-untyped-def]
        self.prior_of = prior_of
        self.value_of = value_of
        self.calls = 0

    def evaluate(
        self, state: GameState, seat: int, legal: Sequence[Action]
    ) -> tuple[dict[ActionKey, float], float]:
        self.calls += 1
        keyed = keyed_actions(legal, seat, len(state.players))
        raw = {
            k: (self.prior_of(a) if self.prior_of else 1.0) for k, a in keyed.items()
        }
        total = sum(raw.values())
        value = (
            self.value_of(state, seat)
            if self.value_of
            else victory_points(state, seat) / 10
            + 0.001 * state.players[seat].resource_card_count()
        )
        return {k: v / total for k, v in raw.items()}, value


def _play(seed: int, stop) -> tuple[CatanGame, GameState, list[Action]]:  # type: ignore[no-untyped-def]
    """Heuristic self-play until ``stop(state, legal, actor)`` holds."""
    game = CatanGame(NUM_PLAYERS)
    st = game.reset(seed=seed)
    agents = [HeuristicAgent() for _ in range(NUM_PLAYERS)]
    for _ in range(20_000):
        if game.is_terminal(st):
            break
        legal = game.legal_actions(st)
        actor = acting_player(st)
        if stop(st, legal, actor):
            return game, st, legal
        game.apply_action(st, agents[actor].choose_action(st, legal, actor))
    raise AssertionError("stop condition never reached")


def _main_decision(seed: int, min_actions: int = 4, after_steps: int = 200):  # type: ignore[no-untyped-def]
    count = 0

    def stop(st: GameState, legal: list[Action], actor: int) -> bool:
        nonlocal count
        count += 1
        return (
            count > after_steps
            and actor == SEAT
            and st.phase is Phase.MAIN
            and len(keyed_actions(legal, SEAT, NUM_PLAYERS)) >= min_actions
        )

    return _play(seed, stop)


def _public(st: GameState, seat: int) -> tuple:
    own = st.players[seat]
    return (
        tuple(sorted((r.name, n) for r, n in own.resources.items())),
        tuple(sorted((c.card_type.name, c.bought_this_turn) for c in own.dev_hand)),
        [p.resource_card_count() for p in st.players],
        [len(p.dev_hand) for p in st.players],
        [[c.bought_this_turn for c in p.dev_hand] for p in st.players],
        len(st.dev_deck),
        sorted((r.name, n) for r, n in st.bank.items()),
        [victory_points(st, i) for i in range(len(st.players))],
    )


def test_road_building_orderings_share_one_key() -> None:
    a = action_key(PlayRoadBuilding(edge_id_1=3, edge_id_2=9), 0, 4)
    b = action_key(PlayRoadBuilding(edge_id_1=9, edge_id_2=3), 0, 4)
    assert a == b


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_determinize_preserves_public_information(seed: int) -> None:
    game = CatanGame(NUM_PLAYERS)
    st = game.reset(seed=seed)
    agents = [HeuristicAgent() for _ in range(NUM_PLAYERS)]
    rng = random.Random(seed)
    for step in range(3000):
        if game.is_terminal(st):
            break
        legal = game.legal_actions(st)
        actor = acting_player(st)
        if step % 40 == 0:
            for seat in range(NUM_PLAYERS):
                d = determinize(st, seat, rng)
                assert _public(d, seat) == _public(st, seat)
                assert all(
                    true_victory_points(d, p) < 10
                    for p in range(NUM_PLAYERS)
                    if p != seat
                )
        game.apply_action(st, agents[actor].choose_action(st, legal, actor))


def test_determinize_does_not_touch_the_live_state() -> None:
    _, st, _ = _main_decision(5)
    before = (
        _public(st, SEAT),
        st.rng.getstate(),
        [list(p.dev_hand) for p in st.players],
    )
    hands = [dict(p.resources) for p in st.players]
    determinize(st, SEAT, random.Random(1))
    assert st.rng.getstate() == before[1]
    assert [dict(p.resources) for p in st.players] == hands
    assert _public(st, SEAT) == before[0]


def test_determinize_redeals_hidden_information() -> None:
    _, st, _ = _main_decision(5)
    rng = random.Random(3)
    seen = {
        tuple(
            tuple(sorted((r.name, n) for r, n in p.resources.items()))
            for p in d.players
        )
        for d in (determinize(st, SEAT, rng) for _ in range(20))
    }
    assert len(seen) > 1


def _hidden(s: GameState) -> list:
    return [
        (dict(p.resources), [c.card_type for c in p.dev_hand]) for p in s.players[1:]
    ]


def test_search_does_not_peek_and_leaves_state_untouched() -> None:
    game, st, legal = _main_decision(7)
    # Same public information, different hidden hands / dev cards / deck / rng.
    twin = determinize(st, SEAT, random.Random(12345))
    assert _public(twin, SEAT) == _public(st, SEAT)
    assert _hidden(twin) != _hidden(st) or twin.dev_deck != st.dev_deck

    cfg = SearchConfig(simulations=24)
    snapshot = (st.rng.getstate(), _public(st, SEAT), _hidden(st), list(st.dev_deck))
    a = search(
        st,
        SEAT,
        legal,
        config=cfg,
        evaluator=StubEvaluator(),
        opponent=HeuristicAgent(),
        rng=random.Random(99),
    )
    assert (
        st.rng.getstate(),
        _public(st, SEAT),
        _hidden(st),
        list(st.dev_deck),
    ) == snapshot
    b = search(
        twin,
        SEAT,
        game.legal_actions(twin),
        config=cfg,
        evaluator=StubEvaluator(),
        opponent=HeuristicAgent(),
        rng=random.Random(99),
    )
    assert (a.key, a.visits) == (b.key, b.visits)


def test_search_is_reproducible_and_returns_a_legal_action() -> None:
    _, st, legal = _main_decision(11)
    cfg = SearchConfig(simulations=16)
    runs = [
        search(
            st,
            SEAT,
            legal,
            config=cfg,
            evaluator=StubEvaluator(),
            opponent=HeuristicAgent(),
            rng=random.Random(5),
        )
        for _ in range(2)
    ]
    assert runs[0].visits == runs[1].visits
    assert runs[0].action in legal
    assert sum(runs[0].visits.values()) == 16


def _build_spot(seed: int):  # type: ignore[no-untyped-def]
    def stop(st: GameState, legal: list[Action], actor: int) -> bool:
        return (
            actor == SEAT
            and st.phase is Phase.MAIN
            and any(isinstance(a, PlaceCity | PlaceSettlement) for a in legal)
            # No opponent can win inside the horizon: a terminal -1 would widen
            # the tree-wide min-max range and (correctly) dwarf a 0.1 value gap.
            and all(true_victory_points(st, p) <= 4 for p in range(1, NUM_PLAYERS))
        )

    return _play(seed, stop)


def test_values_override_a_lopsided_prior() -> None:
    """Q is min-max normalized, so a ~0.1 value gap beats a 0.9 prior on EndTurn
    (the failure mode if Q were left raw against the exploration term)."""
    spot = None
    for seed in range(1, 40):
        try:
            spot = _build_spot(seed)
        except AssertionError:
            continue
        break
    if spot is None:
        pytest.skip("no build opportunity found")
    _, st, legal = spot
    builds = [a for a in legal if isinstance(a, PlaceCity | PlaceSettlement)]

    def prior(a: Action) -> float:
        return 90.0 if isinstance(a, EndTurn) else 1.0

    res = search(
        st,
        SEAT,
        legal,
        config=SearchConfig(simulations=200),
        evaluator=StubEvaluator(
            prior_of=prior, value_of=lambda s, p: victory_points(s, p) / 10
        ),
        opponent=HeuristicAgent(),
        rng=random.Random(2),
    )
    assert res.action in builds


def test_a_winning_move_is_found() -> None:
    for seed in range(1, 30):
        game = CatanGame(NUM_PLAYERS)
        st = game.reset(seed=seed)
        agents = [HeuristicAgent() for _ in range(NUM_PLAYERS)]
        prev = None
        while not game.is_terminal(st):
            legal = game.legal_actions(st)
            actor = acting_player(st)
            action = agents[actor].choose_action(st, legal, actor)
            prev = (st.copy(), legal, actor, action)
            game.apply_action(st, action)
        assert prev is not None
        before, legal, actor, action = prev
        if actor != st.winner or len(keyed_actions(legal, actor, NUM_PLAYERS)) < 2:
            continue
        res = search(
            before,
            actor,
            legal,
            config=SearchConfig(simulations=40),
            evaluator=StubEvaluator(value_of=lambda _s, _p: 0.0),
            opponent=HeuristicAgent(),
            rng=random.Random(1),
        )
        assert action_key(res.action, actor, NUM_PLAYERS) == action_key(
            action, actor, NUM_PLAYERS
        )
        return
    pytest.skip("no game ended on a multi-option winning move")


def test_dev_card_conservation_holds_for_unseen_pool() -> None:
    _, st, _ = _main_decision(9)
    d = determinize(st, SEAT, random.Random(0))
    total = sum(len(p.dev_hand) for p in d.players) + len(d.dev_deck)
    assert total == sum(len(p.dev_hand) for p in st.players) + len(st.dev_deck)
    assert all(isinstance(c, DevCardType) for c in d.dev_deck)
