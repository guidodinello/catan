"""Phase 3 benchmark harness: run named agent-lineup arms and report win
rates with seat rotation.

A **mode registry** maps a mode name to a role lineup (one role name per
seat slot) -- this replaces truco-py's 11 hardcoded ``--mode`` branches
(flagged in ``docs/shared-ml-package.md`` as the one non-reusable part of
``scripts/benchmark.py``).

**Seat rotation is mandatory, not a refinement.** README decision 12
established seat, not player id, as the unit of analysis (the starting
player is randomised). A heuristic-vs-random result is confounded unless
role-to-seat assignment rotates so each role occupies each seat an equal
number of times across an arm. The rotation offset is ``engine_seed % k``
(``k`` = lineup length), so it is exact only when a run's ``engine_seed``s
are ``k`` consecutive integers -- the convention every ``run_a1``/``run_a2``/
``run_tables``/this harness's own ``run_arm`` already follows. Per-seat RNG
streams are ``random.Random(hash((driver_seed, seat)))`` -- keyed by seat, so
swapping one seat's agent never perturbs another seat's draws.

Statistics reuse ``experiments/mcstats.py`` wholesale: ``wilson_interval``
for every proportion, ``two_proportion_test`` for the headline arm-vs-arm
claim. Output reuses ``exp_placement.py``'s stamping helpers so the header
format matches Phase 2's.

Not run in CI. Example:
    uv run python -m experiments.benchmark --mode heuristic_vs_random \\
        --games 10000 --players 4 --workers 8
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from typing import Any

from agents import Agent, HeuristicAgent, RandomAgent
from engine.game import CatanGame
from experiments.exp_placement import _git_commit, _write_result
from experiments.mcstats import two_proportion_test, wilson_interval
from experiments.rollout import GameRecord, run_many

MODE_LINEUPS: dict[str, Callable[[int], tuple[str, ...]]] = {
    "random_vs_random": lambda n: tuple(["random"] * n),
    "heuristic_vs_random": lambda n: ("heuristic",) + tuple(["random"] * (n - 1)),
    "heuristic_vs_heuristic": lambda n: tuple(["heuristic"] * n),
}


def _build_role(role: str, rng: random.Random) -> Agent:
    if role == "heuristic":
        return HeuristicAgent(name="heuristic")
    if role == "random":
        return RandomAgent(rng, name="random")
    raise ValueError(f"unknown role {role!r}")


def _recover_seat_order(num_players: int, engine_seed: int) -> tuple[int, ...]:
    """Recompute the same ``seat_order`` ``run_game`` will derive internally
    from ``engine_seed`` -- deterministic, via the public
    ``CatanGame.reset`` API, not a private accessor."""
    state = CatanGame(num_players=num_players).reset(seed=engine_seed)
    return tuple(state.setup_sequence[:num_players])


def _rotate(lineup: tuple[str, ...], engine_seed: int) -> tuple[str, ...]:
    r = engine_seed % len(lineup)
    return lineup[r:] + lineup[:r]


def benchmark_agent_factory(
    mode: str, num_players: int, engine_seed: int, driver_seed: int
) -> list[Agent]:
    """The module-level ``AgentFactory`` every benchmark arm uses -- picklable
    across ``ProcessPoolExecutor`` workers via ``functools.partial`` over
    ``mode`` (a plain string), never a closure over constructed agents."""
    lineup = MODE_LINEUPS[mode](num_players)
    seat_roles = _rotate(lineup, engine_seed)
    seat_order = _recover_seat_order(num_players, engine_seed)

    agents: list[Agent | None] = [None] * num_players
    for seat in range(num_players):
        seat_rng = random.Random(hash((driver_seed, seat)))
        player_id = seat_order[seat]
        agents[player_id] = _build_role(seat_roles[seat], seat_rng)
    assert all(a is not None for a in agents)
    return agents  # type: ignore[return-value]


def _cumulative_resources_through_turn(
    record: GameRecord, player: int, turn: int
) -> int:
    return sum(
        sum(e.gains.values())
        for e in record.production_events
        if e.player == player and e.turn <= turn
    )


def summarize_by_role(records: list[GameRecord]) -> dict[str, Any]:
    """Win rate + diagnostics per role name, aggregated across every seat the
    role occupied -- plus the rotation bookkeeping (role x seat counts) that
    proves the arm isn't confounded by an uneven seat assignment."""
    by_role_wins: dict[str, int] = defaultdict(int)
    by_role_n: dict[str, int] = defaultdict(int)
    by_role_resources: dict[str, list[int]] = defaultdict(list)
    seat_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for r in records:
        for seat, player_id in enumerate(r.seat_order):
            role = r.agent_names[player_id]
            by_role_n[role] += 1
            seat_counts[role][seat] += 1
            if r.winning_seat == seat:
                by_role_wins[role] += 1
            by_role_resources[role].append(
                _cumulative_resources_through_turn(r, player_id, 10)
            )

    summary: dict[str, Any] = {}
    for role, n in by_role_n.items():
        wins = by_role_wins[role]
        ci = wilson_interval(wins, n)
        resources = by_role_resources[role]
        summary[role] = {
            "n_seat_occupancies": n,
            "wins": wins,
            "win_rate": wins / n,
            "win_rate_wilson_ci": list(ci),
            "mean_resources_through_turn_10": sum(resources) / len(resources),
            "seat_occupancy_counts": dict(sorted(seat_counts[role].items())),
        }
    return summary


def summarize_by_seat(records: list[GameRecord]) -> dict[str, Any]:
    """Per-seat win rate, regardless of which role occupied it -- the gate 1
    check (RandomAgent vs RandomAgent -> 1/n per seat) needs this, since
    ``summarize_by_role`` collapses same-named roles across every seat they
    occupied."""
    n = len(records)
    num_players = records[0].num_players if records else 0
    summary: dict[str, Any] = {}
    for seat in range(num_players):
        wins = sum(1 for r in records if r.winning_seat == seat)
        ci = wilson_interval(wins, n)
        summary[str(seat)] = {
            "n_games": n,
            "wins": wins,
            "win_rate": wins / n,
            "win_rate_wilson_ci": list(ci),
            "expected_under_null": 1 / num_players if num_players else None,
        }
    return summary


def run_arm(
    mode: str,
    num_players: int,
    n_games: int,
    engine_seed_base: int,
    driver_seed_base: int,
    workers: int,
) -> dict[str, Any]:
    lineup = MODE_LINEUPS[mode](num_players)
    if n_games % len(lineup) != 0:
        raise ValueError(
            f"--games must be a multiple of the lineup length ({len(lineup)}) "
            "for exact seat rotation counts"
        )
    factory = partial(benchmark_agent_factory, mode)
    pairs = [(engine_seed_base + i, driver_seed_base + i) for i in range(n_games)]
    records = run_many(num_players, pairs, agent_factory=factory, workers=workers)

    by_role = summarize_by_role(records)
    by_seat = summarize_by_seat(records)
    roles = list(by_role)
    comparisons: dict[str, Any] = {}
    if len(roles) == 2:
        a, b = roles
        test = two_proportion_test(
            by_role[a]["wins"],
            by_role[a]["n_seat_occupancies"],
            by_role[b]["wins"],
            by_role[b]["n_seat_occupancies"],
        )
        comparisons[f"{a}_vs_{b}"] = {"z": test.z, "p_value": test.p_value}

    non_winner_exceeded_ten_rate = sum(
        1 for r in records if r.non_winner_exceeded_ten
    ) / len(records)

    return {
        "git_commit": _git_commit(),
        "generated_at": datetime.now(UTC).isoformat(),
        "experiment": "benchmark",
        "mode": mode,
        "num_players": num_players,
        "n_games": n_games,
        "engine_seed_base": engine_seed_base,
        "driver_seed_base": driver_seed_base,
        "by_role": by_role,
        "by_seat": by_seat,
        "comparisons": comparisons,
        "known_biases": [
            "PlayVictoryPoint is not gated by has_played_dev_card_this_turn -- "
            "every agent reveals VP cards immediately.",
            "Bank shortage (_produce) skips a resource when the bank cannot "
            "cover all claimants; heuristic agents produce more, so they hit "
            "this more often than random agents did in Phase 2.",
            "HeuristicAgent is hand-tuned against Phase 2's random-play "
            "results; a rule that helps against random opponents need not "
            "help against a good one. heuristic_vs_heuristic is the check.",
        ],
        "non_winner_exceeded_ten_rate": non_winner_exceeded_ten_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(MODE_LINEUPS), required=True)
    parser.add_argument("--games", type=int, default=10_000)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--engine-seed-base", type=int, default=1)
    parser.add_argument("--driver-seed-base", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    payload = run_arm(
        mode=args.mode,
        num_players=args.players,
        n_games=args.games,
        engine_seed_base=args.engine_seed_base,
        driver_seed_base=args.driver_seed_base,
        workers=args.workers,
    )
    path = _write_result(f"benchmark_{args.mode}_p{args.players}", payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
