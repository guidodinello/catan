"""Tabulate the persisted human-vs-bot games of experiment 007 into a stamped
result JSON (``docs/experiments/007-human-games.md``).

Reads the per-game records ``server/persistence.py`` writes to
``.catan-games/`` (one per human-involved game, gitignored), matches them to
the pre-registered ``SCHEDULE_007``, and reports human wins per bot kind with
Wilson CIs. The committed result embeds every scheduled game's row, so it is
the raw data too.

    python -m experiments.human_games --experiment 007

Pre-registered rules applied here (not tunable by flag):

- a game that was started but never finished counts as a **human loss**,
  and is also listed under ``abandoned``;
- a slot played more than once counts only its earliest attempt; the repeats
  are listed under ``deviations``;
- records outside the schedule (wrong seed, wrong human seat, mixed bot
  kinds) are excluded and listed under ``deviations``.

Writes nothing if there are no matching records. Exits non-zero if the rl
records were made with different checkpoints, since one result must name one
checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gamekit.mc import two_proportion_test, wilson_interval
from gamekit.results import stamp, write_result

from engine.game import CatanGame

RESULTS_DIR = Path(__file__).resolve().parent / "results"
GAMES_DIR = Path(__file__).resolve().parent.parent / ".catan-games"

BOT_KINDS = ("rl", "heuristic")
PAIRS = 12
ENGINE_SEED_BASE = 7000
NUM_PLAYERS = 4
CHANCE_RATE = 1 / NUM_PLAYERS


@dataclass(frozen=True, slots=True)
class Slot:
    game_no: int
    bot_kind: str
    engine_seed: int
    human_seat: int  # player id -- the GUI's "Seat i"
    turn_position: int  # 0 = moves first in setup; ``(k-1) mod 4``


def turn_order(engine_seed: int) -> tuple[int, ...]:
    """``turn_order(seed)[position] == player id`` -- the engine draws the
    setup order from the seed (same as ``benchmark._recover_seat_order``), so
    player id and turn position only coincide by luck.
    """
    state = CatanGame(num_players=NUM_PLAYERS).reset(seed=engine_seed)
    return tuple(state.setup_sequence[:NUM_PLAYERS])


def build_schedule() -> tuple[Slot, ...]:
    """Pair ``k`` (1..12): engine seed ``7000+k``; the human takes turn
    position ``(k-1) mod 4`` (so each position gets 3 pairs), i.e. whichever
    player id that seed puts there. Each pair is played once against each bot
    kind on the same board; the order within a pair alternates (rl first on
    odd pairs, heuristic first on even ones).
    """
    slots: list[Slot] = []
    for k in range(1, PAIRS + 1):
        order = BOT_KINDS if k % 2 == 1 else BOT_KINDS[::-1]
        position = (k - 1) % NUM_PLAYERS
        for kind in order:
            slots.append(
                Slot(
                    game_no=len(slots) + 1,
                    bot_kind=kind,
                    engine_seed=ENGINE_SEED_BASE + k,
                    human_seat=turn_order(ENGINE_SEED_BASE + k)[position],
                    turn_position=position,
                )
            )
    return tuple(slots)


SCHEDULE_007 = build_schedule()

KNOWN_BIASES = [
    "n=12 per bot kind: the Wilson CIs are very wide and the rl-vs-heuristic "
    "comparison has essentially no power -- descriptive only.",
    "Chance for a human seat in a 4-player game is 25%.",
    "One human whose skill is self-assessed (see the log); results say "
    "nothing about other humans.",
    "Domestic trading is disabled for every bot (they never propose and "
    "always reject) -- temporary, catan #28.",
    "An abandoned game counts as a human loss (pre-registered).",
    "The human learns over the 24 games; the pair order alternates to "
    "avoid confounding that with bot kind, but does not remove it.",
]


def _bot_kind(record: dict[str, Any]) -> str | None:
    kinds = {
        k for i, k in enumerate(record["seat_kinds"]) if i not in record["human_seats"]
    }
    return kinds.pop() if len(kinds) == 1 else None


def _interval(k: int, n: int) -> dict[str, float] | None:
    if n == 0:
        return None
    ci = wilson_interval(k, n)
    return {"lower": ci.lower, "upper": ci.upper}


def _verdict(k: int, n: int) -> str:
    ci = _interval(k, n)
    if ci is None:
        return "no games"
    if ci["lower"] > CHANCE_RATE:
        return "human clearly stronger than this bot"
    if ci["upper"] < CHANCE_RATE:
        return "bot stronger than this human"
    return "not distinguishable from chance"


def load_records(games_dir: Path, experiment: str) -> list[dict[str, Any]]:
    records = [json.loads(p.read_text()) for p in sorted(games_dir.glob("*.json"))]
    return [r for r in records if r.get("experiment") == experiment]


def _dice_totals(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Descriptive dice luck over finished games with a complete roll
    history: observed count per total vs. fair-dice expectation. ``None`` if
    no such game. Not part of the verdict.
    """
    games = [r for r in rows if r.get("dice_history_complete") and r["dice_counts"]]
    if not games:
        return None
    counts = {
        str(t): sum(r["dice_counts"][str(t)] for r in games) for t in range(2, 13)
    }
    n_rolls = sum(counts.values())
    return {
        "n_games": len(games),
        "n_rolls": n_rolls,
        "counts": counts,
        "expected": {str(t): n_rolls * (6 - abs(t - 7)) / 36 for t in range(2, 13)},
    }


def tabulate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Pure function of the records: the result payload minus the stamp."""
    by_slot = {(s.bot_kind, s.engine_seed): s for s in SCHEDULE_007}
    matched: dict[int, dict[str, Any]] = {}
    deviations: list[dict[str, Any]] = []
    for rec in sorted(records, key=lambda r: r["created_at"]):
        kind = _bot_kind(rec)
        slot = by_slot.get((kind, rec["engine_seed"])) if kind else None
        reason = None
        if slot is None:
            reason = "not a scheduled (bot kind, engine seed)"
        elif rec["human_seats"] != [slot.human_seat]:
            reason = f"human seat {rec['human_seats']}, scheduled {slot.human_seat}"
        elif slot.game_no in matched:
            reason = "slot already played; only the earliest attempt counts"
        if reason is not None:
            deviations.append({"game_id": rec["game_id"], "reason": reason})
            continue
        assert slot is not None
        matched[slot.game_no] = rec

    checkpoints = {
        json.dumps(r["rl_checkpoint"], sort_keys=True)
        for r in matched.values()
        if r["rl_checkpoint"] is not None
    }
    shas = {json.loads(c)["sha256_12"] for c in checkpoints}
    if len(shas) > 1:
        raise ValueError(
            f"rl games were made with different checkpoints: {sorted(shas)}"
        )
    checkpoint = json.loads(next(iter(checkpoints))) if checkpoints else None

    rows: list[dict[str, Any]] = []
    for slot in SCHEDULE_007:
        found = matched.get(slot.game_no)
        if found is None:
            rows.append({**asdict(slot), "status": "missing"})
            continue
        finished = found["status"] == "finished"
        human_won = finished and found["winner"] in found["human_seats"]
        rows.append(
            {
                **asdict(slot),
                "game_id": found["game_id"],
                "status": found["status"] if finished else "abandoned",
                "human_won": human_won,
                "winner": found.get("winner"),
                "winner_kind": found.get("winner_kind"),
                "vp_true": found.get("vp_true"),
                "turn_count": found["turn_count"] if finished else None,
                "trade_policy": found["trade_policy"],
                # Absent from records written before dice tracking, and from
                # games that never finished.
                "dice_counts": found.get("dice_counts"),
                "dice_history_complete": found.get("dice_history_complete", False),
            }
        )

    arms: dict[str, Any] = {}
    for kind in BOT_KINDS:
        arm = [r for r in rows if r["bot_kind"] == kind]
        played = [r for r in arm if r["status"] != "missing"]
        finished = [r for r in played if r["status"] == "finished"]
        wins = sum(1 for r in played if r["human_won"])
        by_seat = {}
        for position in range(NUM_PLAYERS):
            seat_games = [r for r in played if r["turn_position"] == position]
            by_seat[str(position)] = {
                "n": len(seat_games),
                "human_wins": sum(1 for r in seat_games if r["human_won"]),
            }
        human_vp = [r["vp_true"][r["human_seat"]] for r in finished]
        bot_wins = sum(
            1
            for r in finished
            if r["winner"] != r["human_seat"] and r["winner"] is not None
        )
        arms[kind] = {
            "planned": len(arm),
            "played": len(played),
            "finished": len(finished),
            "abandoned": sum(1 for r in played if r["status"] == "abandoned"),
            "missing": sum(1 for r in arm if r["status"] == "missing"),
            "human_wins": wins,
            "human_win_rate": wins / len(played) if played else None,
            "human_win_rate_wilson95": _interval(wins, len(played)),
            "verdict_vs_chance_25pct": _verdict(wins, len(played)),
            "human_wins_by_turn_position": by_seat,
            "mean_human_true_vp": sum(human_vp) / len(human_vp) if human_vp else None,
            "mean_turn_count": (
                sum(r["turn_count"] for r in finished) / len(finished)
                if finished
                else None
            ),
            "bot_wins_total": bot_wins,
        }

    comparison = None
    rl, heur = arms["rl"], arms["heuristic"]
    if rl["played"] and heur["played"]:
        test = two_proportion_test(
            rl["human_wins"], rl["played"], heur["human_wins"], heur["played"]
        )
        comparison = {
            "human_win_rate_vs_rl_minus_vs_heuristic": rl["human_win_rate"]
            - heur["human_win_rate"],
            "z": test.z,
            "p_value": test.p_value,
            "note": "descriptive only; essentially no power at n=12 per arm",
        }
    return {
        "num_players": NUM_PLAYERS,
        "rl_checkpoint": checkpoint,
        "trade_policy": "reject_all",
        "arms": arms,
        "dice_totals": _dice_totals(rows),
        "rl_vs_heuristic": comparison,
        "abandoned": [r["game_id"] for r in rows if r["status"] == "abandoned"],
        "missing_game_nos": [r["game_no"] for r in rows if r["status"] == "missing"],
        "deviations": deviations,
        "games": rows,
        "known_biases": KNOWN_BIASES,
    }


def result_name(experiment: str, checkpoint_stem: str | None) -> str:
    """Checkpoint-qualified, like ``benchmark._result_name``: two checkpoints
    must never overwrite each other's committed result."""
    base = f"human_games_{experiment}_p{NUM_PLAYERS}"
    return f"{base}_{checkpoint_stem}" if checkpoint_stem else base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--experiment", default="007")
    parser.add_argument("--games-dir", type=Path, default=GAMES_DIR)
    args = parser.parse_args()
    if args.experiment != "007":
        parser.error("only experiment 007 has a pre-registered schedule")

    records = load_records(args.games_dir, args.experiment)
    try:
        payload = tabulate(records)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    played = sum(a["played"] for a in payload["arms"].values())
    print(
        f"{len(records)} records, {played}/{len(SCHEDULE_007)} scheduled slots played, "
        f"{len(payload['deviations'])} deviations, "
        f"{len(payload['missing_game_nos'])} missing"
    )
    if played == 0:
        print("nothing to tabulate -- no result written", file=sys.stderr)
        return 1
    checkpoint = payload["rl_checkpoint"]
    path = write_result(
        RESULTS_DIR,
        result_name(args.experiment, checkpoint["stem"] if checkpoint else None),
        stamp("human_games", experiment_id=args.experiment, **payload),
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
