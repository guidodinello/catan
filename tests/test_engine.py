"""Cross-cutting engine invariants: random full games, a Hypothesis
RuleBasedStateMachine over the phase machine, and a rollout benchmark.

Rule under test: legal_actions(state) is never empty before GAME_OVER; no
resource/piece count ever goes negative; total resource cards (bank + all
hands) stays constant at 95 throughout (bank-shortage aside -- production is
skipped, cards never vanish); total dev cards (deck + hands + played) stays
constant at 25. The step budget lives in the test driver, never inside
is_terminal().
"""

import random
from typing import Any

from hypothesis import settings
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from agents.random_agent import StratifiedRandomAgent
from engine.actions import Action
from engine.game import CatanGame
from engine.state import DEV_DECK_SIZE, GameState, Phase, acting_player

STEP_BUDGET = 4000
TOTAL_RESOURCE_CARDS = 95


def _sample_and_resolve_action(
    rng: random.Random, state: GameState, actions: list[Action]
) -> Action:
    """Sample and resolve one action via the shared stratified agent (see
    ``agents/random_agent.py`` -- the canonical sentinel-resolution logic
    lives there so this test and the Phase 3 rollout driver never drift).

    Passes the real acting player (``acting_player(state)``), not a
    hardcoded seat -- a ProposeTrade/CounterTrade sentinel is resolved from
    *that* player's hand, and the acting player is not always seat 0 (e.g.
    a trade responder, or a counterer, while state.current_player still
    names the original turn player).
    """
    actor = acting_player(state)
    return StratifiedRandomAgent(rng).choose_action(state, actions, actor)


def _total_resource_cards(state: GameState) -> int:
    total = sum(state.bank.values())
    for p in state.players:
        total += sum(p.resources.values())
    return total


def _total_dev_cards(state: GameState) -> int:
    total = len(state.dev_deck)
    for p in state.players:
        total += len(p.dev_hand)
        total += p.played_knights
        total += p.played_progress_count
        total += p.revealed_vp_cards
    return total


def _assert_invariants(game: CatanGame, state: GameState) -> None:
    if not game.is_terminal(state):
        assert game.legal_actions(state), f"empty legal_actions at phase {state.phase}"
    for p in state.players:
        assert p.settlements_remaining >= 0
        assert p.cities_remaining >= 0
        assert p.roads_remaining >= 0
        for count in p.resources.values():
            assert count >= 0
    for count in state.bank.values():
        assert count >= 0
    assert _total_resource_cards(state) == TOTAL_RESOURCE_CARDS
    assert _total_dev_cards(state) == DEV_DECK_SIZE


def test_random_games_never_violate_core_invariants() -> None:
    driver_rng = random.Random(0)
    for seed in range(5):
        game = CatanGame(num_players=3)
        state = game.reset(seed=seed)
        _assert_invariants(game, state)
        steps = 0
        while not game.is_terminal(state) and steps < STEP_BUDGET:
            action = _sample_and_resolve_action(
                driver_rng, state, game.legal_actions(state)
            )
            game.apply_action(state, action)
            _assert_invariants(game, state)
            steps += 1


# ---------------------------------------------------------------------------
# Hypothesis RuleBasedStateMachine over the phase machine.
# ---------------------------------------------------------------------------


class CatanStateMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.game = CatanGame(num_players=3)
        self.state = self.game.reset(seed=random.randint(0, 1_000_000))
        self.driver_rng = random.Random(1)
        self.steps = 0

    @rule()
    def step(self) -> None:
        if self.game.is_terminal(self.state) or self.steps >= STEP_BUDGET:
            return
        legal = self.game.legal_actions(self.state)
        assert legal, f"empty legal_actions at phase {self.state.phase}"
        action = _sample_and_resolve_action(self.driver_rng, self.state, legal)
        self.game.apply_action(self.state, action)
        self.steps += 1

    @invariant()
    def resources_and_pieces_never_negative(self) -> None:
        for p in self.state.players:
            assert p.settlements_remaining >= 0
            assert p.cities_remaining >= 0
            assert p.roads_remaining >= 0
            for count in p.resources.values():
                assert count >= 0
        for count in self.state.bank.values():
            assert count >= 0

    @invariant()
    def resource_and_dev_card_totals_are_conserved(self) -> None:
        assert _total_resource_cards(self.state) == TOTAL_RESOURCE_CARDS
        assert _total_dev_cards(self.state) == DEV_DECK_SIZE

    @invariant()
    def legal_actions_never_empty_before_game_over(self) -> None:
        if self.state.phase != Phase.GAME_OVER:
            assert self.game.legal_actions(self.state)


TestCatanStateMachine = CatanStateMachine.TestCase
TestCatanStateMachine.settings = settings(max_examples=30, stateful_step_count=200)


# ---------------------------------------------------------------------------
# Rollout benchmark (baseline for Phase 2, no hard threshold).
# ---------------------------------------------------------------------------


def _play_one_random_game(seed: int) -> None:
    driver_rng = random.Random(seed)
    game = CatanGame(num_players=3)
    state = game.reset(seed=seed)
    steps = 0
    while not game.is_terminal(state) and steps < STEP_BUDGET:
        action = _sample_and_resolve_action(
            driver_rng, state, game.legal_actions(state)
        )
        game.apply_action(state, action)
        steps += 1


def test_benchmark_one_random_game(benchmark: Any) -> None:
    benchmark(_play_one_random_game, 42)
