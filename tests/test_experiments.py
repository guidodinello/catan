"""Phase 2 experiment-layer tests.

Fast smoke coverage for the rollout driver and mcstats utilities, plus the
load-bearing correctness check: the analytic ``vertex_production`` table
must match the actual engine's own ``_produce`` logic exactly (deterministic
per dice total -- no statistics needed, since production for a fixed total
has no randomness once ownership is fixed). If this fails, the whole
experiment layer's ground truth is wrong.

Full experiment runs (thousands of games) are not exercised here -- they are
manual, per the Phase 2 plan: ``uv run python -m experiments.exp_placement``.
"""

import random
import statistics

from engine.actions import PlaceSettlement
from engine.board import GEOMETRY, TERRAIN_RESOURCE
from engine.game import CatanGame, _produce
from experiments.features import dice_probability, vertex_production
from experiments.mcstats import (
    Accumulator,
    benjamini_hochberg,
    sample_size_clt,
    sample_size_hoeffding,
    two_proportion_sample_size,
    two_proportion_test,
    wilson_interval,
)
from experiments.rollout import ScriptedSetup, make_random_policy, run_game, run_many

STEP_BUDGET = 20_000


# ---------------------------------------------------------------------------
# Rollout driver smoke tests
# ---------------------------------------------------------------------------


def test_run_game_terminates_and_invariants_hold() -> None:
    for seed in range(5):
        record = run_game(
            4, engine_seed=seed, driver_seed=seed, step_budget=STEP_BUDGET
        )
        assert record.winner is not None, f"seed {seed} did not terminate"
        assert record.final_vp[record.winner] >= 10
        assert len(record.seat_order) == 4
        assert set(record.seat_order) == set(range(4))
        # VP-by-turn is recorded once per turn, always one row per player.
        for row in record.vp_by_turn:
            assert len(row) == 4
            for vp in row:
                assert vp >= 0


def test_scripted_setup_forces_the_treatment_vertex() -> None:
    game = CatanGame(num_players=4)
    state = game.reset(seed=1)
    legal_vertices = [
        a.vertex_id for a in game.legal_actions(state) if isinstance(a, PlaceSettlement)
    ]
    v = legal_vertices[0]
    record = run_game(
        4,
        engine_seed=1,
        driver_seed=2,
        scripted_setup=ScriptedSetup(seat=0, vertex_id=v),
        step_budget=STEP_BUDGET,
    )
    assert record.first_settlement_vertex[0] == v
    assert record.treatment_seat == 0
    assert record.treatment_vertex == v


def test_scripted_setup_rejects_an_illegal_vertex() -> None:
    """A vertex id outside the board's valid range is never a legal
    PlaceSettlement -- scripting it should raise, not silently substitute
    another vertex."""
    nonexistent_vertex = len(GEOMETRY.vertex_hexes) + 100
    try:
        run_game(
            4,
            engine_seed=1,
            driver_seed=1,
            scripted_setup=ScriptedSetup(seat=0, vertex_id=nonexistent_vertex),
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_run_many_is_deterministic_regardless_of_worker_count() -> None:
    pairs = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1)]
    sequential = run_many(3, pairs, workers=1, step_budget=STEP_BUDGET)
    parallel = run_many(3, pairs, workers=3, step_budget=STEP_BUDGET)
    for a, b in zip(sequential, parallel, strict=True):
        assert a.winner == b.winner
        assert a.turn_count == b.turn_count
        assert a.step_count == b.step_count
        assert a.final_vp == b.final_vp


def test_random_policy_also_terminates() -> None:
    record = run_game(
        3, engine_seed=7, driver_seed=7, policy_factory=make_random_policy
    )
    assert record.winner is not None


# ---------------------------------------------------------------------------
# The load-bearing check: analytic vertex_production vs. the engine's own
# _produce, exactly (deterministic -- no sampling needed).
# ---------------------------------------------------------------------------


def test_vertex_production_matches_engine_produce_exactly() -> None:
    game = CatanGame(num_players=3)
    state = game.reset(seed=1)

    # Pick a single-hex (boundary) vertex so one dice total maps to a clean,
    # unconfounded expectation -- vertex_production's amount=1 settlement
    # case, no robber interference (robber starts on the desert, which this
    # vertex's one hex is not, by construction below).
    def land_non_desert_hexes(vid: int) -> list:
        return [
            h
            for h in GEOMETRY.vertex_hexes[vid]
            if GEOMETRY.is_land(h) and state.board.terrain[h].name != "DESERT"
        ]

    v = next(
        vid
        for vid in range(len(GEOMETRY.vertex_hexes))
        if len(land_non_desert_hexes(vid)) == 1
    )
    state.players[0].settlement_vertices.add(v)
    hex_id = land_non_desert_hexes(v)[0]
    resource = TERRAIN_RESOURCE[state.board.terrain[hex_id]]
    token = state.board.tokens[hex_id]

    # For every possible dice total, _produce must grant exactly 1 card of
    # `resource` to player 0 iff total == token, and nothing otherwise.
    for total in range(2, 13):
        trial = state.copy()
        _produce(trial, total)
        gained = (
            trial.players[0].resources[resource] - state.players[0].resources[resource]
        )
        expected = 1 if total == token else 0
        assert gained == expected, f"total={total} token={token} gained={gained}"

    analytic = vertex_production(state.board, v)
    assert set(analytic) == {resource}
    assert analytic[resource] == dice_probability(token)


# ---------------------------------------------------------------------------
# mcstats unit tests
# ---------------------------------------------------------------------------


def test_wilson_interval_matches_known_reference_values() -> None:
    # 50/100 successes, 95% CI: standard textbook Wilson bounds ~ (0.404, 0.596).
    ci = wilson_interval(50, 100)
    assert round(ci.lower, 3) == 0.404
    assert round(ci.upper, 3) == 0.596
    assert ci.width > 0

    # A single 0/n case must stay within [0, 1] and not error (floating
    # point may leave `lower` a hair above exact zero).
    ci_zero = wilson_interval(0, 20)
    assert 0 <= ci_zero.lower < 1e-9
    assert 0 < ci_zero.upper < 1


def test_sample_size_formulas_match_the_plan_table() -> None:
    # From the Phase 2 plan's sample-size table: p=1/3, 80% power, alpha=0.05.
    assert two_proportion_sample_size(1 / 3, 0.05) == 1396
    assert two_proportion_sample_size(1 / 3, 0.02) == 8721
    assert two_proportion_sample_size(1 / 3, 0.01) == 34884

    assert sample_size_clt(0.05, 0.05) == 385
    assert sample_size_hoeffding(0.05, 0.05) == 738
    # Hoeffding (distribution-free) must never require fewer samples than
    # the CLT approximation for the same (eps, delta).
    assert sample_size_hoeffding(0.05, 0.05) >= sample_size_clt(0.05, 0.05)


def test_two_proportion_test_detects_no_difference_for_identical_arms() -> None:
    result = two_proportion_test(50, 100, 50, 100)
    assert result.z == 0.0
    assert result.p_value == 1.0


def test_benjamini_hochberg_flags_only_the_small_p_values() -> None:
    flags = benjamini_hochberg([0.001, 0.2, 0.03, 0.5], q=0.05)
    assert flags == [True, False, False, False]
    assert benjamini_hochberg([]) == []


def test_accumulator_matches_stdlib_statistics() -> None:
    rng = random.Random(0)
    data = [rng.uniform(0, 10) for _ in range(500)]
    acc: Accumulator[float] = Accumulator(value=lambda x: x)
    acc.update_all(data)
    assert acc.count == len(data)
    assert round(acc.mean, 9) == round(statistics.mean(data), 9)
    assert round(acc.variance, 6) == round(statistics.variance(data), 6)

    result = acc.result()
    ci = result.confidence_interval()
    assert ci.lower < result.mean < ci.upper


def test_accumulator_projects_arbitrary_sample_types() -> None:
    """Fixes the mmo-utils gap noted in docs/shared-ml-package.md: this
    Accumulator is not float-only."""

    class Sample:
        def __init__(self, value: float) -> None:
            self.value = value

    acc: Accumulator[Sample] = Accumulator(value=lambda s: s.value)
    acc.update_all(Sample(v) for v in [1.0, 2.0, 3.0])
    assert acc.mean == 2.0
