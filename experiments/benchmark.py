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
streams are ``gamekit.seats.seat_rng(driver_seed, seat)`` -- keyed by seat, so
swapping one seat's agent never perturbs another seat's draws.

The rotation loop, the multiple-of-lineup-length guard, and the
rotation-balance validation now live in ``gamekit.benchmark.run_arm`` --
this module supplies the catan-specific role construction and the
Catan-only diagnostics (``mean_resources_through_turn_10``,
``known_biases``, ``non_winner_exceeded_ten_rate``) layered on top.
Statistics reuse ``gamekit.mc`` wholesale: ``wilson_interval`` for every
proportion (via ``gamekit.benchmark``'s summaries), ``two_proportion_test``
for the headline arm-vs-arm claim. Output reuses ``gamekit.results.stamp``
so the header format matches Phase 2's.

Not run in CI. Example:
    uv run python -m experiments.benchmark --mode heuristic_vs_random \\
        --games 10000 --players 4 --workers 8
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

from gamekit.benchmark import run_arm as _gk_run_arm
from gamekit.results import write_result
from gamekit.seats import rotate, seat_rng

from agents import CatanAgent, HeuristicAgent, RandomAgent
from engine.game import CatanGame
from experiments.rollout import GameRecord, run_many

RESULTS_DIR = Path(__file__).resolve().parent / "results"

MODE_LINEUPS: dict[str, Callable[[int], tuple[str, ...]]] = {
    "random_vs_random": lambda n: tuple(["random"] * n),
    "heuristic_vs_random": lambda n: ("heuristic",) + tuple(["random"] * (n - 1)),
    "heuristic_vs_heuristic": lambda n: tuple(["heuristic"] * n),
    "rl_vs_random": lambda n: ("rl",) + tuple(["random"] * (n - 1)),
    "rl_vs_heuristic": lambda n: ("rl",) + tuple(["heuristic"] * (n - 1)),
}

RL_MODES = frozenset({"rl_vs_random", "rl_vs_heuristic"})

RL_KNOWN_BIAS = (
    "The RL agent never *proposes* a domestic trade: rl/action_space.py "
    "reserves the ProposeTrade/CounterTrade atoms and masks them off, since "
    "the give/receive bundle product is not enumerable (README decision 5). "
    "It does still accept and reject offers, so it is not excluded from "
    "trading entirely -- but against 3 RandomAgents, which do propose, it "
    "gives up the initiative half of a rule random play uses constantly."
)


def _build_role(
    role: str, rng: random.Random, checkpoint: str | None = None
) -> CatanAgent:
    if role == "heuristic":
        return HeuristicAgent(name="heuristic")
    if role == "random":
        return RandomAgent(rng, name="random")
    if role == "rl":
        if checkpoint is None:
            raise ValueError("the rl_* modes require --checkpoint")
        from agents.rl_agent import RLAgent

        return RLAgent(checkpoint, name="rl", rng=rng)
    raise ValueError(f"unknown role {role!r}")


def _recover_seat_order(num_players: int, engine_seed: int) -> tuple[int, ...]:
    """Recompute the same ``seat_order`` ``run_game`` will derive internally
    from ``engine_seed`` -- deterministic, via the public
    ``CatanGame.reset`` API, not a private accessor."""
    state = CatanGame(num_players=num_players).reset(seed=engine_seed)
    return tuple(state.setup_sequence[:num_players])


def benchmark_agent_factory(
    mode: str,
    checkpoint: str | None,
    num_players: int,
    engine_seed: int,
    driver_seed: int,
) -> list[CatanAgent]:
    """The module-level ``AgentFactory`` every benchmark arm uses -- picklable
    across ``ProcessPoolExecutor`` workers via ``functools.partial`` over
    ``mode`` and ``checkpoint`` (both plain strings), never a closure over
    constructed agents. A checkpoint *path* crosses the process boundary, not
    a loaded model; ``agents.rl_agent`` caches the load per worker."""
    lineup = MODE_LINEUPS[mode](num_players)
    seat_roles = rotate(lineup, engine_seed)
    seat_order = _recover_seat_order(num_players, engine_seed)

    agents: list[CatanAgent | None] = [None] * num_players
    for seat in range(num_players):
        player_id = seat_order[seat]
        agents[player_id] = _build_role(
            seat_roles[seat], seat_rng(driver_seed, seat), checkpoint
        )
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


def _mean_resources_through_turn_10_by_role(
    records: Sequence[GameRecord],
) -> dict[str, float]:
    """The one ``summarize_by_role`` diagnostic that's Catan-specific
    (resource production), computed alongside gamekit's generic role/seat
    summaries rather than inside them."""
    by_role: dict[str, list[int]] = defaultdict(list)
    for r in records:
        for player_id in r.seat_order:
            role = r.agent_names[player_id]
            by_role[role].append(_cumulative_resources_through_turn(r, player_id, 10))
    return {role: sum(values) / len(values) for role, values in by_role.items()}


def run_arm(
    mode: str,
    num_players: int,
    n_games: int,
    engine_seed_base: int,
    driver_seed_base: int,
    workers: int,
    checkpoint: str | None = None,
) -> dict[str, Any]:
    if mode in RL_MODES and checkpoint is None:
        raise ValueError(f"mode {mode!r} requires a checkpoint")
    lineup = MODE_LINEUPS[mode](num_players)

    played: list[Sequence[GameRecord]] = []

    def play(pairs: Sequence[tuple[int, int]]) -> Sequence[GameRecord]:
        records = run_many(
            num_players,
            pairs,
            agent_factory=partial(benchmark_agent_factory, mode, checkpoint),
            workers=workers,
        )
        played.append(records)
        return records

    gk_payload = _gk_run_arm(
        lineup=lineup,
        num_seats=num_players,
        n_games=n_games,
        engine_seed_base=engine_seed_base,
        driver_seed_base=driver_seed_base,
        play=play,
        winning_seat=lambda r: r.winning_seat,
    )
    records = played[0]

    mean_resources = _mean_resources_through_turn_10_by_role(records)
    by_role = {
        role: {
            **summary,
            "mean_resources_through_turn_10": mean_resources[role],
        }
        for role, summary in gk_payload["by_role"].items()
    }
    # Re-insert to match the original field order (mean_resources_... before
    # seat_occupancy_counts) rather than the dict-merge order above.
    for summary in by_role.values():
        occupancy = summary.pop("seat_occupancy_counts")
        summary["seat_occupancy_counts"] = occupancy

    non_winner_exceeded_ten_rate = sum(
        1 for r in records if r.non_winner_exceeded_ten
    ) / len(records)

    return {
        "git_commit": gk_payload["git_commit"],
        "generated_at": gk_payload["generated_at"],
        "experiment": "benchmark",
        "mode": mode,
        "num_players": num_players,
        "n_games": n_games,
        "engine_seed_base": engine_seed_base,
        "driver_seed_base": driver_seed_base,
        "checkpoint": Path(checkpoint).name if checkpoint else None,
        "by_role": by_role,
        "by_seat": gk_payload["by_seat"],
        "comparisons": gk_payload["comparisons"],
        "known_biases": [
            "PlayVictoryPoint is not gated by has_played_dev_card_this_turn -- "
            "every agent reveals VP cards immediately. This no longer affects "
            "win timing (engine.game.true_victory_points/_check_win auto-wins "
            "on a hidden VP card the instant the true total reaches 10, "
            "independent of when/whether it's revealed); it's noted here only "
            "as a residual agent-behavior quirk, not a win-timing bias.",
            "Bank shortage (_produce) skips a resource when the bank cannot "
            "cover all claimants; heuristic agents produce more, so they hit "
            "this more often than random agents did in Phase 2.",
            "HeuristicAgent is hand-tuned against Phase 2's random-play "
            "results; a rule that helps against random opponents need not "
            "help against a good one. heuristic_vs_heuristic is the check.",
            *([RL_KNOWN_BIAS] if mode in RL_MODES else []),
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
    parser.add_argument("--checkpoint", default=None, help="required by the rl_* modes")
    args = parser.parse_args()

    payload = run_arm(
        mode=args.mode,
        num_players=args.players,
        n_games=args.games,
        engine_seed_base=args.engine_seed_base,
        driver_seed_base=args.driver_seed_base,
        workers=args.workers,
        checkpoint=args.checkpoint,
    )
    path = write_result(RESULTS_DIR, f"benchmark_{args.mode}_p{args.players}", payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
