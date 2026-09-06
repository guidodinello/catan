"""Starting-placement Monte Carlo experiments.

A1 -- per-vertex win-rate ranking on a fixed board (single engine seed, one
scripted seat, every candidate first-settlement vertex run out).

A2 -- feature-bucketed win-rate across random boards (many engine seeds, the
treatment seat's first settlement left to the policy, then bucketed post hoc
by ``experiments.features.vertex_features``).

Win rate is reported with a Wilson CI, but it is *not* the primary ranking
signal -- see the Phase 2 plan's "win rate is not the primary metric" note.
Policy-insensitive intermediate metrics (cumulative resources produced,
turns to 5 VP) are reported alongside every arm.

Run directly, e.g.:
    uv run python -m experiments.exp_placement a1 --games 2000
    uv run python -m experiments.exp_placement a2 --games 20000
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engine.board import NUM_VERTICES
from engine.game import CatanGame
from experiments.features import vertex_features
from experiments.mcstats import wilson_interval
from experiments.rollout import GameRecord, ScriptedSetup, run_game, run_many

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_GAMES = 10_000
DEFAULT_ENGINE_SEED = 1


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError, FileNotFoundError, OSError:
        return "unknown"


def _write_result(name: str, payload: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def _cumulative_resources_through_turn(
    record: GameRecord, player: int, turn: int
) -> int:
    return sum(
        sum(e.gains.values())
        for e in record.production_events
        if e.player == player and e.turn <= turn
    )


def _turns_to_n_vp(record: GameRecord, player: int, n: int) -> int | None:
    for turn, vps in enumerate(record.vp_by_turn, start=1):
        if vps[player] >= n:
            return turn
    return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_arm(records: list[GameRecord], seat: int) -> dict[str, Any]:
    """Summarize one arm's outcomes for the treatment ``seat``.

    The treatment seat's player id is looked up per-record via
    ``record.seat_order[seat]`` -- it is constant within an A1 arm (one
    ``engine_seed`` fixes ``setup_sequence``) but varies across records in
    an A2 bucket (many boards, many seeds).
    """
    n = len(records)
    wins = sum(1 for r in records if r.winning_seat == seat)
    win_ci = wilson_interval(wins, n) if n else None

    resources_10 = [
        float(_cumulative_resources_through_turn(r, r.seat_order[seat], 10))
        for r in records
    ]
    resources_20 = [
        float(_cumulative_resources_through_turn(r, r.seat_order[seat], 20))
        for r in records
    ]
    turns_to_5vp = [
        t
        for r in records
        if (t := _turns_to_n_vp(r, r.seat_order[seat], 5)) is not None
    ]

    return {
        "n_games": n,
        "wins": wins,
        "win_rate": wins / n if n else None,
        "win_rate_wilson_ci": list(win_ci) if win_ci is not None else None,
        "mean_resources_through_turn_10": _mean(resources_10),
        "mean_resources_through_turn_20": _mean(resources_20),
        "mean_turns_to_5vp": _mean([float(t) for t in turns_to_5vp]),
        "turns_to_5vp_reached_fraction": len(turns_to_5vp) / n if n else None,
    }


def run_a1(
    engine_seed: int,
    seat: int,
    num_players: int,
    n_games: int,
    workers: int,
    vertices: list[int] | None = None,
) -> dict[str, Any]:
    """Per-vertex win-rate ranking on one fixed board.

    ``seat == 0`` (the recommended default) sees the full, empty board, so
    every one of the 54 vertices is a legal first pick for every sample --
    no rejection sampling needed. For ``seat > 0``, earlier seats' setup
    picks are left to the policy and vary by ``driver_seed``, so a given
    candidate vertex may already be occupied/adjacent-occupied for some
    samples; those samples are skipped and counted in ``skipped_illegal``
    (this arm's effective n is implicitly conditioned on "vertex still
    available when this seat's turn came").
    """
    board = CatanGame(num_players=num_players).reset(seed=engine_seed).board
    candidates = vertices if vertices is not None else list(range(NUM_VERTICES))

    arms: dict[int, dict[str, Any]] = {}
    for v in candidates:
        setup = ScriptedSetup(seat=seat, vertex_id=v)
        if seat == 0:
            # Never illegal for the first picker -- batch through run_many
            # for full parallelism.
            pairs = [(engine_seed, d) for d in range(n_games)]
            records = run_many(
                num_players, pairs, scripted_setup=setup, workers=workers
            )
            skipped = 0
        else:
            # May be illegal depending on the (random) earlier picks for a
            # given driver seed; run sequentially and skip failures.
            records = []
            skipped = 0
            for d in range(n_games):
                try:
                    records.append(
                        run_game(num_players, engine_seed, d, scripted_setup=setup)
                    )
                except ValueError:
                    skipped += 1

        summary = summarize_arm(records, seat)
        summary["skipped_illegal"] = skipped
        summary["vertex_features"] = asdict(vertex_features(board, v))
        arms[v] = summary

    ranked = sorted(
        arms.items(),
        key=lambda kv: (kv[1]["win_rate"] is not None, kv[1]["win_rate"]),
        reverse=True,
    )

    return {
        "git_commit": _git_commit(),
        "generated_at": datetime.now(UTC).isoformat(),
        "experiment": "A1_fixed_board_vertex_ranking",
        "engine_seed": engine_seed,
        "seat": seat,
        "num_players": num_players,
        "n_games_per_arm": n_games,
        "policy": "stratified",
        "arms": {str(v): summary for v, summary in arms.items()},
        "ranked_vertices": [v for v, _ in ranked],
    }


def run_a2(
    n_games: int,
    num_players: int,
    seat: int,
    engine_seed_base: int,
    workers: int,
    pip_buckets: tuple[tuple[int, int], ...] = ((2, 6), (7, 9), (10, 20)),
) -> dict[str, Any]:
    """Feature-bucketed win-rate across random boards.

    The treatment seat's first settlement is left entirely to the policy
    (no scripting) -- ``vertex_features`` is looked up post hoc via
    ``first_settlement_vertex``, so buckets reflect what a policy actually
    chose, not an experimenter-forced placement.
    """
    pairs = [(engine_seed_base + i, engine_seed_base + i) for i in range(n_games)]
    records = run_many(num_players, pairs, workers=workers)

    per_record_features = []
    for r in records:
        player = r.seat_order[seat]
        v = r.first_settlement_vertex[seat]
        board = CatanGame(num_players=num_players).reset(seed=r.engine_seed).board
        per_record_features.append((r, player, vertex_features(board, v)))

    pip_labeled: dict[str, list[tuple[GameRecord, int]]] = defaultdict(list)
    resource_labeled: dict[int, list[tuple[GameRecord, int]]] = defaultdict(list)
    for r, player, feat in per_record_features:
        for lo, hi in pip_buckets:
            if lo <= feat.pip_sum <= hi:
                pip_labeled[f"{lo}-{hi}"].append((r, player))
                break
        resource_labeled[feat.distinct_resources].append((r, player))

    def bucket_summary(bucket: list[tuple[GameRecord, int]]) -> dict[str, Any]:
        recs = [r for r, _ in bucket]
        return summarize_arm(recs, seat)

    pooled_wins = sum(1 for r in records if r.winning_seat == seat)
    pooled_ci = wilson_interval(pooled_wins, len(records)) if records else None

    return {
        "git_commit": _git_commit(),
        "generated_at": datetime.now(UTC).isoformat(),
        "experiment": "A2_feature_bucketed_random_boards",
        "seat": seat,
        "num_players": num_players,
        "n_games": n_games,
        "engine_seed_base": engine_seed_base,
        "policy": "stratified",
        "pooled_win_rate_over_all_seats_sanity_check": {
            "n_games": len(records),
            "wins": pooled_wins,
            "win_rate": pooled_wins / len(records) if records else None,
            "win_rate_wilson_ci": list(pooled_ci) if pooled_ci is not None else None,
            "expected_under_null": 1 / num_players,
        },
        "by_pip_sum_bucket": {k: bucket_summary(v) for k, v in pip_labeled.items()},
        "by_distinct_resources": {
            str(k): bucket_summary(v) for k, v in sorted(resource_labeled.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("a1", help="per-vertex ranking on a fixed board")
    p1.add_argument("--engine-seed", type=int, default=DEFAULT_ENGINE_SEED)
    p1.add_argument("--seat", type=int, default=0)
    p1.add_argument("--players", type=int, default=4)
    p1.add_argument("--games", type=int, default=DEFAULT_GAMES)
    p1.add_argument("--workers", type=int, default=1)

    p2 = sub.add_parser("a2", help="feature-bucketed win rate across random boards")
    p2.add_argument("--seat", type=int, default=0)
    p2.add_argument("--players", type=int, default=4)
    p2.add_argument("--games", type=int, default=DEFAULT_GAMES)
    p2.add_argument("--engine-seed-base", type=int, default=DEFAULT_ENGINE_SEED)
    p2.add_argument("--workers", type=int, default=1)

    args = parser.parse_args()

    if args.command == "a1":
        payload = run_a1(
            engine_seed=args.engine_seed,
            seat=args.seat,
            num_players=args.players,
            n_games=args.games,
            workers=args.workers,
        )
        path = _write_result(
            f"a1_seed{args.engine_seed}_seat{args.seat}_p{args.players}", payload
        )
    else:
        payload = run_a2(
            n_games=args.games,
            num_players=args.players,
            seat=args.seat,
            engine_seed_base=args.engine_seed_base,
            workers=args.workers,
        )
        path = _write_result(f"a2_seat{args.seat}_p{args.players}", payload)

    print(f"wrote {path}")


if __name__ == "__main__":
    main()
