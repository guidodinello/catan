"""Critic calibration harness (experiment 009, catan #37).

No checkpoint needed: the critic is an injected ``obs -> values`` stub, and the
rl seat is a ``TradingHeuristicAgent`` standing in under the ``rl_reject`` role
name (it accepts offers, so trade probes and forks actually fire).
"""

from __future__ import annotations

import numpy as np

from agents import CatanAgent, TradingHeuristicAgent
from engine.actions import ProposeTrade
from engine.board import Resource
from engine.game import CatanGame
from engine.state import Phase
from experiments import calibration_stats as cs
from experiments.calibration_analysis import analyze
from experiments.critic_calibration import (
    OFF_LIMITS,
    SEED_RANGES,
    Job,
    counterfactual_pair,
    play_and_record,
    stack_samples,
)
from experiments.rollout import run_game

NUM_PLAYERS = 4
RL_SEAT = 0


def _stub_value(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs)[:, :8].sum(axis=1) * 0.01


def _factory(num_players: int, engine_seed: int, driver_seed: int) -> list[CatanAgent]:
    return [
        TradingHeuristicAgent(name="rl_reject" if i == RL_SEAT else "trading_heuristic")
        for i in range(num_players)
    ]


def _job(engine_seed: int, *, forks: int = 0) -> Job:
    return Job("stub", None, NUM_PLAYERS, engine_seed, engine_seed, True, forks)


def test_recorder_and_forks_do_not_change_the_game() -> None:
    n_offers = 0
    for seed in (3, 4, 5, 6):
        plain = run_game(NUM_PLAYERS, seed, seed, _factory)
        sample = play_and_record(_job(seed, forks=2), _stub_value, _factory)
        assert sample["record"] == plain
        assert sample["won"] == (plain.winner == RL_SEAT)
        assert len(sample["states"]) > 0
        n_offers += len(sample["offers"])
    assert n_offers > 0, "the stand-in must have been offered trades to test the probe"


def test_counterfactual_pair_differs_only_in_the_traded_hands() -> None:
    game = CatanGame(num_players=NUM_PLAYERS)
    state = game.reset(seed=3)
    state.phase, state.current_player = Phase.MAIN, 1
    state.players[1].resources[Resource.LUMBER] = 2
    state.players[RL_SEAT].resources[Resource.BRICK] = 1
    game.apply_action(
        state,
        ProposeTrade(give={Resource.LUMBER: 1}, receive={Resource.BRICK: 1}),
    )
    frozen = state.copy()

    after, before = counterfactual_pair(game, state)

    for s in (after, before):
        assert s.phase is Phase.MAIN and s.trade_offer is None
        assert s.current_player == 1
    assert after.players[1].resources[Resource.LUMBER] == 1
    assert after.players[1].resources[Resource.BRICK] == 1
    assert after.players[RL_SEAT].resources[Resource.LUMBER] == 1
    assert after.players[RL_SEAT].resources[Resource.BRICK] == 0
    assert before.players[1].resources == frozen.players[1].resources
    # the source state is untouched
    assert state.phase is Phase.AWAIT_TRADE_RESPONSE
    assert state.trade_responders == frozen.trade_responders
    assert state.players[1].resources == frozen.players[1].resources


def test_pipeline_runs_end_to_end_on_stub_samples() -> None:
    samples = [
        play_and_record(_job(seed, forks=2), _stub_value, _factory)
        for seed in range(100, 112)
    ]
    rows = stack_samples(samples)
    assert len(rows["state_game"]) == len(rows["v"])
    assert rows["state_headline"].sum() == len(samples)
    result = analyze(rows, arm="vs_trading_heuristic")
    assert result["headline"]["n_games"] == len(samples)
    assert "verdicts" in result


def test_auc_and_spearman_on_known_inputs() -> None:
    y = np.array([0, 0, 1, 1])
    assert cs.auc(np.array([1.0, 2.0, 3.0, 4.0]), y) == 1.0
    assert cs.auc(np.array([4.0, 3.0, 2.0, 1.0]), y) == 0.0
    assert cs.auc(np.ones(4), y) == 0.5
    assert np.isnan(cs.auc(np.arange(3.0), np.zeros(3)))
    np.testing.assert_allclose(cs.rankdata(np.array([5.0, 5.0, 1.0])), [2.5, 2.5, 1.0])
    assert cs.spearman(np.arange(5.0), np.arange(5.0) ** 3) == 1.0
    assert cs.spearman(np.arange(5.0), -np.arange(5.0)) == -1.0


def test_stratified_auc_ignores_between_cell_differences() -> None:
    y = np.array([0, 1, 0, 1])
    cell = np.array([0, 0, 1, 1])
    # perfect within each cell, though cell 1 scores dominate cell 0 overall
    assert cs.stratified_auc(np.array([1.0, 2.0, 10.0, 11.0]), y, cell) == 1.0
    # constant within cells: no within-cell information
    assert cs.stratified_auc(np.array([1.0, 1.0, 9.0, 9.0]), y, cell) == 0.5


def test_logistic_recovers_a_separating_direction() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 1))
    y = (x[:, 0] + rng.normal(scale=0.5, size=500) > 0).astype(float)
    (xs,) = cs.standardize(x)
    w = cs.irls_logistic(xs, y)
    assert w[1] > 0
    assert cs.auc(cs.logistic_predict(xs, w), y) > 0.85


def test_seed_ranges_are_disjoint_from_used_boards_and_rotation_exact() -> None:
    spans = []
    for name, (start, n) in SEED_RANGES.items():
        assert n % 4 == 0, name
        spans.append((start, start + n - 1))
    for lo, hi in spans:
        for off_lo, off_hi in OFF_LIMITS:
            assert hi < off_lo or lo > off_hi
    spans.sort()
    for (_, hi), (lo, _) in zip(spans, spans[1:], strict=False):
        assert hi < lo
