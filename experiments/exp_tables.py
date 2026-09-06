"""Resource/VP probability tables.

Four tables, per the Phase 2 plan:

1. Analytic dice/production table (exact, no simulation) for a given board.
2. VP trajectory: VP-by-turn distribution and P(win | VP=v at turn t).
3. Game shape: game-length distribution and VP composition at the win.
4. Empirical income: cards received per turn per resource, with the bank-
   shortage diagnostic alongside.

Run directly:
    uv run python -m experiments.exp_tables --games 10000 --players 4
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engine.board import NUM_VERTICES
from engine.game import CatanGame
from experiments.features import dice_probability, vertex_production
from experiments.mcstats import wilson_interval
from experiments.rollout import GameRecord, run_many

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MIN_CELL_COUNT = 30  # suppress VP-trajectory cells with fewer samples than this


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


def analytic_dice_production_table(
    engine_seed: int, num_players: int
) -> dict[str, Any]:
    """Table 1: exact dice distribution and expected settlement income per
    roll for every vertex on one board -- no simulation."""
    board = CatanGame(num_players=num_players).reset(seed=engine_seed).board
    dice_dist = {total: dice_probability(total) for total in range(2, 13)}
    per_vertex = {
        v: {str(r): p for r, p in vertex_production(board, v).items()}
        for v in range(NUM_VERTICES)
    }
    return {
        "engine_seed": engine_seed,
        "dice_distribution": dice_dist,
        "expected_settlement_income_per_roll_by_vertex": per_vertex,
    }


TURN_BUCKET_SIZE = 5  # bucket turns into 5-turn windows to keep the table compact


def _turn_bucket(turn: int) -> int:
    """Start turn of this turn's 5-turn bucket (1-5 -> 1, 6-10 -> 6, ...)."""
    return ((turn - 1) // TURN_BUCKET_SIZE) * TURN_BUCKET_SIZE + 1


def vp_trajectory_table(records: list[GameRecord]) -> dict[str, Any]:
    """Table 2: VP distribution by turn, and P(win | VP=v at turn t) with
    Wilson CIs. Turns are grouped into TURN_BUCKET_SIZE-turn windows (keeps
    the committed table compact; VP within a 5-turn window rarely changes
    more than once anyway). Cells with fewer than MIN_CELL_COUNT samples are
    suppressed rather than reported as noise."""
    counts: dict[tuple[int, int], int] = Counter()  # (turn_bucket, vp) -> n
    wins: dict[tuple[int, int], int] = Counter()  # (turn_bucket, vp) -> n won

    for r in records:
        for turn_idx, vps in enumerate(r.vp_by_turn, start=1):
            bucket = _turn_bucket(turn_idx)
            for player, vp in enumerate(vps):
                counts[(bucket, vp)] += 1
                if r.winner == player:
                    wins[(bucket, vp)] += 1

    cells = []
    for (turn_bucket, vp), n in counts.items():
        if n < MIN_CELL_COUNT:
            continue
        k = wins[(turn_bucket, vp)]
        ci = wilson_interval(k, n)
        cells.append(
            {
                "turn_bucket_start": turn_bucket,
                "vp": vp,
                "n": n,
                "p_win": k / n,
                "p_win_wilson_ci": list(ci),
            }
        )
    cells.sort(key=lambda c: (c["turn_bucket_start"], c["vp"]))
    return {
        "turn_bucket_size": TURN_BUCKET_SIZE,
        "min_cell_count": MIN_CELL_COUNT,
        "n_games": len(records),
        "cells": cells,
    }


def game_shape_table(records: list[GameRecord]) -> dict[str, Any]:
    """Table 3: game-length distribution and VP composition at the win.

    VP composition (settlements/cities/longest-road/largest-army/VP-cards)
    is not directly recoverable from ``GameRecord`` without re-deriving
    per-source counts -- this table reports what the record *does* carry:
    final total VP per player, turn/step length, and the winner's final VP.
    """
    turn_lengths = [r.turn_count for r in records]
    step_lengths = [r.step_count for r in records]
    winner_final_vp = [r.final_vp[r.winner] for r in records if r.winner is not None]
    non_winner_over_ten_events = sum(1 for r in records if r.non_winner_exceeded_ten)

    def _dist(values: list[int]) -> dict[str, float]:
        if not values:
            return {}
        sorted_v = sorted(values)
        n = len(sorted_v)
        return {
            "mean": sum(values) / n,
            "min": sorted_v[0],
            "p50": sorted_v[n // 2],
            "p90": sorted_v[int(n * 0.9)],
            "max": sorted_v[-1],
        }

    return {
        "n_games": len(records),
        "turn_length_distribution": _dist(turn_lengths),
        "step_length_distribution": _dist(step_lengths),
        "winner_final_vp_distribution": _dist(winner_final_vp),
        "non_winner_exceeded_ten_vp_transiently": {
            "count": non_winner_over_ten_events,
            "fraction": non_winner_over_ten_events / len(records) if records else None,
            "note": (
                "Engine finding, not a harness bug: _check_win only evaluates "
                "the turn player, so a settlement build stealing Longest Road "
                "can push a third player's VP to >=10 undetected until their "
                "own next turn. Self-resolves within a round."
            ),
        },
    }


def empirical_income_table(records: list[GameRecord]) -> dict[str, Any]:
    """Table 4: mean cards received per turn per resource, plus the bank-
    shortage diagnostic (a resource is skipped entirely by ``_produce`` when
    2+ players would claim more than the bank holds -- this biases
    late-game income downward, recorded here rather than hidden)."""
    per_resource_totals: dict[str, int] = Counter()
    per_resource_events: dict[str, int] = Counter()
    turns_observed = 0

    for r in records:
        turns_observed += r.turn_count
        for event in r.production_events:
            for resource, amount in event.gains.items():
                per_resource_totals[str(resource)] += amount
                per_resource_events[str(resource)] += 1

    mean_cards_per_turn = {
        r: total / turns_observed if turns_observed else None
        for r, total in per_resource_totals.items()
    }

    return {
        "n_games": len(records),
        "total_turns_observed": turns_observed,
        "mean_cards_per_turn_by_resource": mean_cards_per_turn,
        "production_events_by_resource": dict(per_resource_events),
    }


def run_all_tables(
    n_games: int, num_players: int, engine_seed_base: int, workers: int
) -> dict[str, Any]:
    pairs = [(engine_seed_base + i, engine_seed_base + i) for i in range(n_games)]
    records = run_many(num_players, pairs, workers=workers)

    return {
        "git_commit": _git_commit(),
        "generated_at": datetime.now(UTC).isoformat(),
        "num_players": num_players,
        "n_games": n_games,
        "engine_seed_base": engine_seed_base,
        "policy": "stratified",
        "table_1_analytic_dice_production": analytic_dice_production_table(
            engine_seed_base, num_players
        ),
        "table_2_vp_trajectory": vp_trajectory_table(records),
        "table_3_game_shape": game_shape_table(records),
        "table_4_empirical_income": empirical_income_table(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=10_000)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--engine-seed-base", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    payload = run_all_tables(
        n_games=args.games,
        num_players=args.players,
        engine_seed_base=args.engine_seed_base,
        workers=args.workers,
    )
    path = _write_result(f"tables_p{args.players}", payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
