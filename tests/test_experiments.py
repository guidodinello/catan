"""Phase 2 experiment-layer tests.

Fast smoke coverage for the rollout driver, plus the load-bearing correctness
check: the analytic ``vertex_production`` table must match the actual
engine's own ``_produce`` logic exactly (deterministic per dice total -- no
statistics needed, since production for a fixed total has no randomness once
ownership is fixed). If this fails, the whole experiment layer's ground truth
is wrong.

mcstats unit tests moved to gamekit's own test suite
(``tests/test_mc.py``) along with the module itself -- see
``docs/shared-ml-package.md``.

Full experiment runs (thousands of games) are not exercised here -- they are
manual, per the Phase 2 plan: ``uv run python -m experiments.exp_placement``.
"""

import random

from agents.base import Agent
from agents.random_agent import RandomAgent
from engine.actions import PlaceSettlement
from engine.board import GEOMETRY, TERRAIN_RESOURCE
from engine.game import CatanGame, _produce
from experiments.features import dice_probability, vertex_production
from experiments.rollout import ScriptedSetup, run_game, run_many

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


def _random_agent_factory(
    num_players: int, engine_seed: int, driver_seed: int
) -> list[Agent]:
    rng = random.Random(driver_seed)
    return [RandomAgent(rng) for _ in range(num_players)]


def test_random_policy_also_terminates() -> None:
    record = run_game(
        3, engine_seed=7, driver_seed=7, agent_factory=_random_agent_factory
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
