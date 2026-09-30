"""``TradingHeuristicAgent`` and the trade-aware benchmark support (catan #28)."""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

import pytest

from agents import Agent, HeuristicAgent, RandomAgent, TradingHeuristicAgent
from agents.trading_heuristic import (
    LEADER_VP_GUARD,
    MAX_PROPOSALS_PER_TURN,
    next_build_cost,
)
from engine.actions import (
    AcceptTrade,
    Action,
    CounterTrade,
    EndTurn,
    ProposeTrade,
    RejectTrade,
)
from engine.board import Resource
from engine.game import CITY_COST, CatanGame
from engine.state import GameState, Phase, TradeOffer
from experiments.benchmark import _result_name, run_arm
from experiments.rollout import GameRecord, run_game

L, W, G, B, ORE = (
    Resource.LUMBER,
    Resource.WOOL,
    Resource.GRAIN,
    Resource.BRICK,
    Resource.ORE,
)


type Maker = Callable[[random.Random], Agent]


def _factory(*makers: Maker) -> Callable[[int, int, int], list[Agent]]:
    def factory(num_players: int, engine_seed: int, driver_seed: int) -> list[Agent]:
        return [m(random.Random(driver_seed + i)) for i, m in enumerate(makers)]

    return factory


def _th(_rng: random.Random) -> Agent:
    return TradingHeuristicAgent()


def _h(_rng: random.Random) -> Agent:
    return HeuristicAgent()


def _rand(rng: random.Random) -> Agent:
    return RandomAgent(rng)


def _after_setup(seed: int = 3) -> tuple[CatanGame, GameState]:
    game = CatanGame(num_players=4)
    state = game.reset(seed=seed)
    agent = HeuristicAgent()
    while state.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        actor = state.current_player
        game.apply_action(
            state, agent.choose_action(state, game.legal_actions(state), actor)
        )
    return game, state


def _hand(state: GameState, player: int, **cards: int) -> None:
    hand = state.players[player].resources
    for r in Resource:
        hand[r] = 0
    for name, n in cards.items():
        hand[Resource[name]] = n


def _offering_state(
    give: dict[Resource, int],
    receive: dict[Resource, int],
    counter_of: int | None = None,
) -> tuple[CatanGame, GameState, int]:
    """Player 0 proposes to player 1, whose hand is 3 ore, 1 grain, 1 wool:
    one grain short of a city."""
    game, state = _after_setup()
    state.current_player = 0
    _hand(state, 1, ORE=3, GRAIN=1, WOOL=1)
    state.phase = Phase.AWAIT_TRADE_RESPONSE
    state.trade_offer = TradeOffer(0, give, receive, counter_of=counter_of)
    state.trade_responders = [1]
    return game, state, 1


def _respond(state: GameState, game: CatanGame, responder: int) -> Action:
    return TradingHeuristicAgent().choose_action(
        state, game.legal_actions(state), responder
    )


# -- Equivalence to HeuristicAgent apart from trading ---------------------


@pytest.mark.parametrize("seed", range(1, 6))
def test_lone_trader_among_heuristics_replays_all_heuristic(seed: int) -> None:
    """Heuristics always reject, so a lone trader can never complete a trade;
    its proposals leave state and RNG untouched, so the game is unchanged."""
    a = run_game(4, seed, seed, _factory(_th, _h, _h, _h))
    b = run_game(4, seed, seed, _factory(_h, _h, _h, _h))
    assert (a.winner, a.final_vp, a.vp_by_turn) == (b.winner, b.final_vp, b.vp_by_turn)


# -- Proposal bound and trade activity ------------------------------------


# Only trader seats are capped; RandomAgent proposes without limit.
def _max_proposals_in_a_turn(record: GameRecord) -> int:
    counts: dict[tuple[int, int], int] = {}
    for e in record.trade_events:
        if e.kind == "propose" and record.agent_names[e.actor] == "trading_heuristic":
            counts[(e.actor, e.turn)] = counts.get((e.actor, e.turn), 0) + 1
    return max(counts.values(), default=0)


@pytest.mark.parametrize("makers", [(_th,) * 4, (_th, _rand, _rand, _rand)])
def test_proposals_are_bounded_and_games_finish(makers: tuple[Maker, ...]) -> None:
    for seed in range(1, 5):
        record = run_game(4, seed, seed, _factory(*makers))
        assert record.winner is not None
        assert _max_proposals_in_a_turn(record) <= MAX_PROPOSALS_PER_TURN


def test_traders_complete_trades_with_each_other() -> None:
    records = [run_game(4, s, s, _factory(*[_th] * 4)) for s in range(1, 7)]
    accepts = sum(e.kind == "accept" for r in records for e in r.trade_events)
    assert accepts > 0


def test_proposal_budget_holds_on_a_static_state() -> None:
    game, state = _after_setup()
    state.phase = Phase.MAIN
    state.current_player = 0
    _hand(state, 0, ORE=3, GRAIN=1, LUMBER=2)
    agent = TradingHeuristicAgent()
    proposals = []
    for _ in range(6):
        action = agent.choose_action(state, game.legal_actions(state), 0)
        if isinstance(action, ProposeTrade):
            proposals.append(action)
        else:
            assert isinstance(action, EndTurn)
    assert 0 < len(proposals) <= MAX_PROPOSALS_PER_TURN
    assert len(
        {(tuple(p.give.items()), tuple(p.receive.items())) for p in proposals}
    ) == len(proposals)
    first = proposals[0]
    assert first.receive == {G: 1} and first.give == {L: 1}


def test_next_build_cost_works_for_a_responder() -> None:
    game, state, responder = _offering_state({G: 1}, {W: 1})
    assert next_build_cost(state, responder) == CITY_COST


# -- Accept rule ------------------------------------------------------------


def test_accepts_an_offer_that_cuts_its_shortfall() -> None:
    game, state, r = _offering_state({G: 1}, {W: 1})
    assert isinstance(_respond(state, game, r), AcceptTrade)


def test_accepts_a_counter_to_its_own_proposal() -> None:
    game, state, r = _offering_state({G: 1}, {W: 1}, counter_of=1)
    action = _respond(state, game, r)
    assert isinstance(action, AcceptTrade)


def test_rejects_an_offer_that_widens_its_shortfall() -> None:
    game, state, r = _offering_state({W: 1}, {G: 1})
    assert isinstance(_respond(state, game, r), RejectTrade)


def test_rejects_paying_more_cards_than_it_receives() -> None:
    game, state, r = _offering_state({G: 1}, {W: 1, ORE: 1})
    assert isinstance(_respond(state, game, r), RejectTrade)


def test_rejects_when_it_cannot_pay() -> None:
    game, state, r = _offering_state({G: 1}, {B: 1})
    assert isinstance(_respond(state, game, r), RejectTrade)


def test_rejects_a_proposer_near_winning() -> None:
    game, state, r = _offering_state({G: 1}, {W: 1})
    state.players[0].revealed_vp_cards = LEADER_VP_GUARD
    assert isinstance(_respond(state, game, r), RejectTrade)


def test_never_counters() -> None:
    for give, receive in (({G: 1}, {W: 1}), ({W: 1}, {G: 1}), ({G: 1}, {B: 1})):
        game, state, r = _offering_state(give, receive)
        assert any(isinstance(a, CounterTrade) for a in game.legal_actions(state))
        assert not isinstance(_respond(state, game, r), CounterTrade)


# -- Benchmark support -------------------------------------------------------


@pytest.mark.parametrize(
    "mode", ["trading_heuristic_2v2", "heuristic_vs_trading_heuristic"]
)
def test_trading_modes_report_trade_and_vp_stats(mode: str) -> None:
    payload = run_arm(mode, 4, 8, 1, 1, workers=1)
    stats = payload["trade_stats"]
    assert set(stats["by_role"]) == {"trading_heuristic", "heuristic"}
    assert stats["by_role"]["heuristic"]["proposals"] == 0
    assert stats["by_role"]["trading_heuristic"]["max_proposals_in_one_turn"] <= (
        MAX_PROPOSALS_PER_TURN
    )
    assert set(payload["vp_card_stats"]) == {"trading_heuristic", "heuristic"}
    assert payload["no_winner_games"] == 0


def test_rl_reject_mode_result_name_is_checkpoint_qualified() -> None:
    name = _result_name(
        "rl_reject_vs_trading_heuristic", 4, str(Path("x") / "ckpt_1.zip")
    )
    assert name == "benchmark_rl_reject_vs_trading_heuristic_p4_ckpt_1"


def test_rl_reject_mode_requires_a_checkpoint() -> None:
    with pytest.raises(ValueError, match="checkpoint"):
        run_arm("rl_reject_vs_trading_heuristic", 4, 4, 1, 1, workers=1)


def test_2v2_mode_is_four_player_only() -> None:
    with pytest.raises(ValueError, match="4-player"):
        run_arm("trading_heuristic_2v2", 3, 6, 1, 1, workers=1)
