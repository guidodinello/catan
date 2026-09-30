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
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

from gamekit.benchmark import run_arm as _gk_run_arm
from gamekit.mc import wilson_interval
from gamekit.results import write_result
from gamekit.seats import rotate, seat_rng

from agents import CatanAgent, HeuristicAgent, RandomAgent, TradingHeuristicAgent
from engine.game import CatanGame
from experiments.rollout import GameRecord, run_many

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _trading_2v2_lineup(n: int) -> tuple[str, ...]:
    if n != 4:
        raise ValueError("trading_heuristic_2v2 is a 4-player mode")
    return ("trading_heuristic",) * 2 + ("heuristic",) * 2


MODE_LINEUPS: dict[str, Callable[[int], tuple[str, ...]]] = {
    "random_vs_random": lambda n: tuple(["random"] * n),
    "heuristic_vs_random": lambda n: ("heuristic",) + tuple(["random"] * (n - 1)),
    "heuristic_vs_heuristic": lambda n: tuple(["heuristic"] * n),
    "rl_vs_random": lambda n: ("rl",) + tuple(["random"] * (n - 1)),
    "rl_vs_heuristic": lambda n: ("rl",) + tuple(["heuristic"] * (n - 1)),
    # Trading-opponent modes (catan #28). HeuristicAgent never proposes and
    # always rejects, so a lone TradingHeuristic among 3 Heuristics can never
    # complete a trade (it replays heuristic_vs_heuristic exactly); the
    # meaningful lineups need >= 2 traders.
    "trading_heuristic_2v2": _trading_2v2_lineup,
    "heuristic_vs_trading_heuristic": lambda n: (
        ("heuristic",) + tuple(["trading_heuristic"] * (n - 1))
    ),
    "rl_reject_vs_trading_heuristic": lambda n: (
        ("rl_reject",) + tuple(["trading_heuristic"] * (n - 1))
    ),
}

RL_MODES = frozenset({"rl_vs_random", "rl_vs_heuristic"})

# Every mode that loads a checkpoint (result names are qualified by its stem).
CHECKPOINT_MODES = RL_MODES | {"rl_reject_vs_trading_heuristic"}

RL_REJECT_KNOWN_BIAS = (
    "The rl_reject seat is server.bots.RLSeatAgent, exactly as the web GUI "
    "builds it: the policy plays everything except domestic-trade responses, "
    "which always reject (its accept/reject atoms were never trained). It "
    "never proposes either (propose/counter atoms are masked), so this is the "
    "no-trading RL baseline at a table whose other seats do trade."
)

TRADING_KNOWN_BIAS = (
    "TradingHeuristicAgent (agents/trading_heuristic.py) proposes at most 2 "
    "trades a turn toward its next build target and accepts only offers that "
    "strictly cut its own shortfall to it; it never counters. HeuristicAgent "
    "and the rl_reject seat always reject, so in the mixed modes only "
    "trading_heuristic seats can complete a trade, and only with each other. "
    "Two seats of one role in a game cannot both win, so per-role win-rate "
    "intervals in a role with several seats treat dependent seat outcomes as "
    "independent."
)

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
    if role == "trading_heuristic":
        return TradingHeuristicAgent(name="trading_heuristic")
    if role == "random":
        return RandomAgent(rng, name="random")
    if role == "rl_reject":
        if checkpoint is None:
            raise ValueError("the rl_* modes require --checkpoint")
        # Reused as-is so the benchmark matches the GUI's forced-reject seat.
        # Imported lazily; server.bots imports no fastapi.
        from server.bots import RLSeatAgent

        agent = RLSeatAgent(Path(checkpoint), rng)
        agent.name = "rl_reject"  # must equal the role: record.agent_names keys by it
        return agent
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


def _trade_stats_by_role(records: Sequence[GameRecord]) -> dict[str, Any]:
    """Domestic-trade statistics per role, from ``GameRecord.trade_events``.

    ``proposals`` are fresh ``ProposeTrade``s (a proposal is offered to every
    other seat in turn and accepted by at most one, so the acceptance rate is
    per proposal). ``responses``/``accepted_as_responder`` count every
    accept/reject a role's seats made, counters to their own proposals
    included.
    """
    roles = sorted({name for r in records for name in r.agent_names})
    stats: dict[str, dict[str, Any]] = {
        role: {
            "proposals": 0,
            "counters": 0,
            "proposals_accepted": 0,
            "responses": 0,
            "accepted_as_responder": 0,
            "cards_given": 0,
            "cards_received": 0,
            "accepted_by_responder_role": Counter(),
            "max_proposals_in_one_turn": 0,
        }
        for role in roles
    }
    completed = 0
    for r in records:
        per_turn: Counter[tuple[int, int]] = Counter()
        for e in r.trade_events:
            actor_role = r.agent_names[e.actor]
            proposer_role = r.agent_names[e.proposer]
            if e.kind == "propose":
                stats[actor_role]["proposals"] += 1
                per_turn[(e.actor, e.turn)] += 1
            elif e.kind == "counter":
                stats[actor_role]["counters"] += 1
            else:
                stats[actor_role]["responses"] += 1
                if e.kind == "accept":
                    completed += 1
                    stats[actor_role]["accepted_as_responder"] += 1
                    stats[proposer_role]["cards_given"] += sum(e.give.values())
                    stats[proposer_role]["cards_received"] += sum(e.receive.values())
                    stats[actor_role]["cards_given"] += sum(e.receive.values())
                    stats[actor_role]["cards_received"] += sum(e.give.values())
                    if e.counter_of is None:
                        stats[proposer_role]["proposals_accepted"] += 1
                        stats[proposer_role]["accepted_by_responder_role"][
                            actor_role
                        ] += 1
        for (actor, _), n in per_turn.items():
            role = r.agent_names[actor]
            stats[role]["max_proposals_in_one_turn"] = max(
                stats[role]["max_proposals_in_one_turn"], n
            )
    for s in stats.values():
        n, k = s["proposals"], s["proposals_accepted"]
        s["acceptance_rate"] = k / n if n else None
        s["acceptance_rate_ci95"] = list(wilson_interval(k, n)) if n else None
        s["accepted_by_responder_role"] = dict(s["accepted_by_responder_role"])
    return {
        "n_games": len(records),
        "completed_trades": completed,
        "completed_trades_per_game": completed / len(records),
        "by_role": stats,
    }


def _vp_card_stats_by_role(records: Sequence[GameRecord]) -> dict[str, Any]:
    """How often each role chose ``PlayVictoryPoint`` when it was legal
    (catan #34: the action is dominated, so any positive rate is a leak)."""
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in records:
        for pid, name in enumerate(r.agent_names):
            totals[name][0] += r.vp_card_legal_decisions[pid]
            totals[name][1] += r.vp_card_plays[pid]
    return {
        role: {
            "legal_decisions": legal,
            "plays": plays,
            "plays_per_legal_decision": plays / legal if legal else None,
        }
        for role, (legal, plays) in sorted(totals.items())
    }


def run_arm(
    mode: str,
    num_players: int,
    n_games: int,
    engine_seed_base: int,
    driver_seed_base: int,
    workers: int,
    checkpoint: str | None = None,
) -> dict[str, Any]:
    if mode in CHECKPOINT_MODES and checkpoint is None:
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
        "trade_stats": _trade_stats_by_role(records),
        "vp_card_stats": _vp_card_stats_by_role(records),
        "no_winner_games": sum(1 for r in records if r.winner is None),
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
            *(
                [RL_REJECT_KNOWN_BIAS]
                if mode == "rl_reject_vs_trading_heuristic"
                else []
            ),
            *([TRADING_KNOWN_BIAS] if "trading_heuristic" in lineup else []),
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
    path = write_result(
        RESULTS_DIR, _result_name(args.mode, args.players, args.checkpoint), payload
    )
    print(f"wrote {path}")


def _result_name(mode: str, num_players: int, checkpoint: str | None) -> str:
    """The stamped result's file stem.

    For every non-RL mode this is unchanged: one committed file per mode
    tracks that role's canonical numbers over time (a rerun overwrites it, as
    intended -- benchmark_heuristic_vs_random_p4.json has always meant "the
    current heuristic_vs_random result", not "one run of it").

    For an RL mode, ``mode`` alone is ambiguous: two different checkpoints
    both produce ``rl_vs_random``. Without the checkpoint in the name, a
    second checkpoint's run silently overwrites the first's committed
    result -- which is exactly what happened running this PR's own
    self-play checkpoint through ``rl_vs_random`` and clobbering PR #18's
    committed 81.75% figure before this was added. So an RL mode's result
    is qualified by the checkpoint's stem, and each trained checkpoint gets
    its own permanent file.
    """
    if mode not in CHECKPOINT_MODES or checkpoint is None:
        return f"benchmark_{mode}_p{num_players}"
    return f"benchmark_{mode}_p{num_players}_{Path(checkpoint).stem}"


if __name__ == "__main__":
    main()
