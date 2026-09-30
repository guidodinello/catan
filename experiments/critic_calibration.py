"""Critic calibration check (experiment 009, catan #37).

Measures how well the critic V(s) of a trained checkpoint predicts the acting
seat's eventual win, and whether V's *change* under a hypothetical trade
predicts what accepting the trade does to the win rate (the number #38's
margin needs).

A ``ValueRecorder`` wraps the rl seat, so the game played is exactly the
benchmark game (same agents, same seats, same RNG streams; the recorder only
encodes copies and reads the critic). For every top-level rl decision it
records V and public-state features; for every trade offer the rl seat could
accept it builds two counterfactual states on ``state.copy()`` (accepted /
withdrawn) and records ΔV. For one uniformly sampled offer per game it also
plays the game on from both branches (``fork_outcome``), giving the
interventional win difference.

Not run in CI. Example:
    OMP_NUM_THREADS=1 .venv/bin/python -m experiments.critic_calibration \\
        --arm vs_trading_heuristic --games 200 --workers 6 --checkpoint <zip>
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import random
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path
from typing import Any

import numpy as np
from gamekit.benchmark import run_arm as _gk_run_arm
from gamekit.results import write_result

from agents import CatanAgent
from agents.trading_heuristic import next_build_cost, shortfall
from engine.actions import AcceptTrade, Action, RejectTrade
from engine.game import CatanGame, victory_points
from engine.state import GameState, Phase, acting_player
from experiments.benchmark import MODE_LINEUPS, benchmark_agent_factory
from experiments.rollout import GameRecord, run_game
from rl.encoder import ObservationEncoder

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ROOT = Path(__file__).resolve().parents[1]
NPZ_DIR = ROOT / "rl_runs" / "critic_calibration"

# Seat-rotated arms. Both lineups are the benchmark's own, so trajectories are
# the ones behind 006's 20.72% (vs heuristics) and 008's 18.9% (vs traders).
ARMS: dict[str, str] = {
    "vs_heuristic": "rl_vs_heuristic",
    "vs_trading_heuristic": "rl_reject_vs_trading_heuristic",
}
RL_ROLES = frozenset({"rl", "rl_reject"})

# Seed ranges. Each is consecutive with a length divisible by 4, so
# ``engine_seed % 4`` rotation is exact. OFF_LIMITS are ranges already used by
# an n=4000 benchmark (1..10000) or by 008's smoke runs (90001..100000).
SEED_RANGES: dict[str, tuple[int, int]] = {
    "smoke": (500_001, 200),
    "vs_heuristic": (1_000_001, 4000),
    "vs_trading_heuristic": (2_000_001, 4000),
    "reserved_for_38_tuning": (3_000_001, 4000),
}
OFF_LIMITS: tuple[tuple[int, int], ...] = ((1, 10_000), (90_001, 100_000))

HEADLINE_SALT = 0x9E3779B1
FORK_SALT = 0x85EBCA6B
FORK_STEP_BUDGET = 20_000
PHASES = list(Phase)

ValueFn = Callable[[np.ndarray], np.ndarray]
AgentFactory = Callable[[int, int, int], list[CatanAgent]]


@cache
def make_value_fn(checkpoint: str) -> ValueFn:
    """Batched critic ``(n, OBS_DIM) -> (n,)`` on CPU. The model comes from
    ``agents.rl_agent._load_model``'s per-process cache, so the recorder, the
    rl agent and every fork share one set of weights."""
    import torch

    from agents.rl_agent import _load_model

    policy = _load_model(checkpoint, "cpu").policy

    def value(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
            out = policy.predict_values(t).reshape(-1).cpu().numpy()
        return np.asarray(out, dtype=np.float64)

    return value


@dataclass(frozen=True, slots=True)
class Job:
    mode: str
    checkpoint: str | None
    num_players: int
    engine_seed: int
    driver_seed: int
    probe: bool  # record trade-offer counterfactuals
    fork_pairs: int  # accept/reject playout pairs at one sampled offer; 0 = none


def _pid_by_name(agents: Sequence[CatanAgent]) -> int:
    seats = [i for i, a in enumerate(agents) if a.name in RL_ROLES]
    if len(seats) != 1:
        raise RuntimeError(f"expected exactly one rl seat, got {seats}")
    return seats[0]


class ValueRecorder:
    """Transparent wrapper around the rl seat's agent: same actions, plus
    critic readings taken on copies. Draws from no game RNG."""

    def __init__(
        self,
        inner: CatanAgent,
        seat: int,
        value_fn: ValueFn,
        num_players: int,
        *,
        probe: bool,
        snapshot_agents: Callable[[], list[CatanAgent]] | None,
        rng: random.Random,
    ) -> None:
        self.name = inner.name
        self.inner = inner
        self.seat = seat
        self._value = value_fn
        self._n = num_players
        self._game = CatanGame(num_players=num_players)
        self._encoder = ObservationEncoder(num_players)
        self._board: Any = None
        self._probe = probe
        self._snapshot_agents = snapshot_agents
        self._rng = rng
        self._own_turns = 0
        self.states: list[tuple[float, ...]] = []
        self.offers: list[tuple[float, ...]] = []
        self.fork_snapshot: tuple[GameState, list[CatanAgent], int] | None = None

    def reset(self) -> None:
        self.inner.reset()

    def _obs(self, state: GameState) -> np.ndarray:
        """Observation of a *real* game state (resets the encoder's per-board
        statics when the board object changes). Counterfactual copies use
        ``self._encoder.encode`` directly: same board, no reset."""
        if state.board is not self._board:
            self._encoder.reset(state)
            self._board = state.board
        return self._encoder.encode(state, self.seat)

    def choose_action(
        self, state: GameState, legal_actions: list[Action], player_idx: int
    ) -> Action:
        if state.phase is Phase.AWAIT_TRADE_RESPONSE:
            if self._probe:
                self._probe_offer(state, legal_actions)
        else:
            self._record_state(state)
        return self.inner.choose_action(state, legal_actions, player_idx)

    def _hands_vp(self, state: GameState) -> tuple[int, int]:
        own = victory_points(state, self.seat)
        best_opp = max(
            victory_points(state, i) for i in range(self._n) if i != self.seat
        )
        return own, best_opp

    def _record_state(self, state: GameState) -> None:
        if state.phase is Phase.ROLL and state.current_player == self.seat:
            self._own_turns += 1
        v = float(self._value(self._obs(state)[None, :])[0])
        own, best_opp = self._hands_vp(state)
        self.states.append(
            (v, own, own - best_opp, self._own_turns, PHASES.index(state.phase))
        )

    def _probe_offer(self, state: GameState, legal: list[Action]) -> None:
        offer = state.trade_offer
        if (
            offer is None
            or offer.counter_of is not None
            or not any(isinstance(a, AcceptTrade) for a in legal)
        ):
            return
        seat, proposer = self.seat, offer.proposer
        after, before = counterfactual_pair(self._game, state)
        gift, pay = before.copy(), before.copy()
        for r, c in offer.give.items():
            gift.players[seat].resources[r] += c
        for r, c in offer.receive.items():
            pay.players[seat].resources[r] -= c
        obs = np.stack(
            [self._obs(state)]
            + [self._encoder.encode(s, seat) for s in (before, after, gift, pay)]
        )
        v_pending, v_before, v_after, v_gift, v_pay = self._value(obs)

        cost = next_build_cost(state, seat)
        hand = state.players[seat].resources
        if cost is None:
            drop = float("nan")
        else:
            drop = float(
                shortfall(hand, cost) - shortfall(after.players[seat].resources, cost)
            )
        self.offers.append(
            (
                float(v_after - v_before),
                float(v_gift - v_before),
                float(v_pay - v_before),
                float(v_pending - v_before),
                float(v_before),
                float(sum(offer.give.values())),
                float(sum(offer.receive.values())),
                drop,
                float(victory_points(state, proposer)),
                float(victory_points(state, seat)),
            )
        )
        if self._snapshot_agents is not None and self._rng.random() < 1 / len(
            self.offers
        ):
            unwrapped = [
                a.inner if isinstance(a, ValueRecorder) else a
                for a in self._snapshot_agents()
            ]
            self.fork_snapshot = (
                state.copy(),
                copy.deepcopy(unwrapped),
                len(self.offers) - 1,
            )


OFFER_FIELDS = (
    "dv",
    "dv_gift",
    "dv_pay",
    "v_pending_minus_before",
    "v_before",
    "cards_in",
    "cards_out",
    "shortfall_drop",
    "proposer_vp",
    "own_vp",
)
STATE_FIELDS = ("v", "vp_own", "lead", "turn", "phase")


def counterfactual_pair(
    game: CatanGame, state: GameState
) -> tuple[GameState, GameState]:
    """From a pending offer the acting responder may accept: ``(after, before)``,
    both on copies, both back in MAIN on the proposer's turn with the offer
    cleared. ``after``: the responder accepts. ``before``: it and every
    remaining responder reject (the offer is withdrawn). They differ only in
    the two traded hands; a violation raises."""
    offer = state.trade_offer
    if offer is None:
        raise ValueError("state has no pending trade offer")
    responder = acting_player(state)
    after, before = state.copy(), state.copy()
    game.apply_action(after, AcceptTrade())
    while before.phase is Phase.AWAIT_TRADE_RESPONSE:
        game.apply_action(before, RejectTrade())
    for s in (after, before):
        if s.phase is not Phase.MAIN or s.trade_offer is not None:
            raise RuntimeError("counterfactual did not return to a clean MAIN state")
    for i, (a, b) in enumerate(zip(after.players, before.players, strict=True)):
        sign = {offer.proposer: -1, responder: 1}.get(i, 0)
        for r in a.resources:
            moved = sign * (offer.give.get(r, 0) - offer.receive.get(r, 0))
            if a.resources[r] - b.resources[r] != moved:
                raise RuntimeError(
                    f"player {i} differs beyond the traded bundle in {r}"
                )
    return after, before


def fork_outcome(
    snapshot: tuple[GameState, list[CatanAgent], int],
    seat: int,
    pairs: int,
    engine_seed: int,
) -> dict[str, Any]:
    """Play the pending offer on from both branches, ``pairs`` times, with
    common random numbers (both branches reseeded with the same seed). The
    reject branch is the rl seat rejecting; later responders then act by their
    own policies, exactly as in the real game."""
    state0, agents0, offer_idx = snapshot
    game = CatanGame(num_players=len(state0.players))
    wins = {True: 0, False: 0}
    for k in range(pairs):
        seed = (engine_seed * FORK_SALT + k) & 0x7FFFFFFF
        for accept in (True, False):
            st = state0.copy()
            agents = copy.deepcopy(agents0)
            st.rng.seed(seed)
            game.apply_action(st, AcceptTrade() if accept else RejectTrade())
            steps = 0
            while not game.is_terminal(st) and steps < FORK_STEP_BUDGET:
                actor = acting_player(st)
                legal = game.legal_actions(st)
                game.apply_action(st, agents[actor].choose_action(st, legal, actor))
                steps += 1
            wins[accept] += int(st.winner == seat)
    return {
        "offer_idx": offer_idx,
        "pairs": pairs,
        "accept_wins": wins[True],
        "reject_wins": wins[False],
        "dwin": (wins[True] - wins[False]) / pairs,
    }


def play_and_record(
    job: Job, value_fn: ValueFn, agent_factory: AgentFactory
) -> dict[str, Any]:
    """One game with a recorder on the rl seat (plus the forks)."""
    holder: dict[str, Any] = {}

    def factory(n: int, engine_seed: int, driver_seed: int) -> list[CatanAgent]:
        agents = agent_factory(n, engine_seed, driver_seed)
        seat = _pid_by_name(agents)
        rec = ValueRecorder(
            agents[seat],
            seat,
            value_fn,
            n,
            probe=job.probe,
            snapshot_agents=(lambda: agents) if job.fork_pairs else None,
            rng=random.Random(driver_seed ^ FORK_SALT),
        )
        agents[seat] = rec
        holder["rec"], holder["seat"] = rec, seat
        return agents

    record = run_game(job.num_players, job.engine_seed, job.driver_seed, factory)
    rec: ValueRecorder = holder["rec"]
    seat: int = holder["seat"]

    states = np.asarray(rec.states, dtype=np.float64).reshape(-1, len(STATE_FIELDS))
    headline = random.Random(job.driver_seed ^ HEADLINE_SALT).randrange(len(states))
    fork = None
    if rec.fork_snapshot is not None:
        fork = fork_outcome(rec.fork_snapshot, seat, job.fork_pairs, job.engine_seed)
    return {
        "record": record,
        "seat": seat,
        "won": record.winner == seat,
        "final_vp": record.final_vp[seat],
        "states": states,
        "headline": headline,
        "offers": np.asarray(rec.offers, dtype=np.float64).reshape(
            -1, len(OFFER_FIELDS)
        ),
        "fork": fork,
        "module_paths": module_paths(),
    }


def module_paths() -> dict[str, str]:
    import agents
    import engine
    import experiments
    import rl

    return {
        m.__name__: str(Path(str(m.__file__)).resolve().parent)
        for m in (agents, engine, rl, experiments)
    }


def run_one(job: Job) -> dict[str, Any]:
    """Pool worker: the real benchmark lineup, the real checkpoint."""
    if job.checkpoint is None:
        raise ValueError("a checkpoint is required")
    return play_and_record(
        job,
        make_value_fn(job.checkpoint),
        partial(benchmark_agent_factory, job.mode, job.checkpoint),
    )


def _sha12(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def collect(
    arm: str,
    *,
    games: int,
    workers: int,
    engine_seed_base: int,
    driver_seed_base: int,
    checkpoint: str,
    probe_games: int,
    fork_pairs: int,
    num_players: int = 4,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the arm through gamekit's rotation harness; returns its payload and
    the per-game samples (in seed order)."""
    mode = ARMS[arm]
    lineup = MODE_LINEUPS[mode](num_players)
    samples: list[dict[str, Any]] = []

    def play(pairs: Sequence[tuple[int, int]]) -> Sequence[GameRecord]:
        jobs = [
            Job(
                mode,
                checkpoint,
                num_players,
                e,
                d,
                probe=arm == "vs_trading_heuristic",
                fork_pairs=fork_pairs
                if (arm == "vs_trading_heuristic" and i < probe_games)
                else 0,
            )
            for i, (e, d) in enumerate(pairs)
        ]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            samples.extend(pool.map(run_one, jobs, chunksize=4))  # ordered
        return [s["record"] for s in samples]

    payload = _gk_run_arm(
        lineup=lineup,
        num_seats=num_players,
        n_games=games,
        engine_seed_base=engine_seed_base,
        driver_seed_base=driver_seed_base,
        play=play,
        winning_seat=lambda r: r.winning_seat,
    )
    paths = {tuple(sorted(s["module_paths"].items())) for s in samples}
    bad = [p for tup in paths for _, p in tup if not Path(p).is_relative_to(ROOT)]
    if bad:
        raise RuntimeError(
            f"workers imported modules outside {ROOT}: {sorted(set(bad))}"
        )
    payload["module_paths"] = [dict(t) for t in paths]
    return payload, samples


def stack_samples(samples: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Flatten per-game samples into row arrays keyed by game index."""
    games = np.arange(len(samples))
    n_states = [len(s["states"]) for s in samples]
    n_offers = [len(s["offers"]) for s in samples]
    out: dict[str, np.ndarray] = {
        "state_game": np.repeat(games, n_states),
        "state_win": np.repeat([int(s["won"]) for s in samples], n_states),
        "state_final_vp": np.repeat([s["final_vp"] for s in samples], n_states),
        "state_headline": np.concatenate(
            [
                np.arange(n) == s["headline"]
                for n, s in zip(n_states, samples, strict=True)
            ]
        ),
        "offer_game": np.repeat(games, n_offers),
        "offer_win": np.repeat([int(s["won"]) for s in samples], n_offers),
    }
    for j, name in enumerate(STATE_FIELDS):
        out[name] = np.concatenate([s["states"][:, j] for s in samples])
    for j, name in enumerate(OFFER_FIELDS):
        out[f"offer_{name}"] = np.concatenate([s["offers"][:, j] for s in samples])
    forks = [(g, s["fork"]) for g, s in enumerate(samples) if s["fork"] is not None]
    out["fork_game"] = np.array([g for g, _ in forks], dtype=np.int64)
    out["fork_offer_idx"] = np.array([f["offer_idx"] for _, f in forks], dtype=np.int64)
    out["fork_dwin"] = np.array([f["dwin"] for _, f in forks], dtype=np.float64)
    out["fork_accept_wins"] = np.array(
        [f["accept_wins"] for _, f in forks], dtype=np.int64
    )
    out["fork_reject_wins"] = np.array(
        [f["reject_wins"] for _, f in forks], dtype=np.int64
    )
    # global offer row of each fork's sampled offer, to join ΔV with Δwin
    offset = (
        np.concatenate(([0], np.cumsum(n_offers)))[out["fork_game"]]
        if len(forks)
        else np.array([], dtype=np.int64)
    )
    out["fork_offer_row"] = offset + out["fork_offer_idx"]
    return out


def _result_name(arm: str, checkpoint: str) -> str:
    return f"critic_calibration_{arm}_p4_{Path(checkpoint).stem}"


def main() -> None:
    from experiments.calibration_analysis import analyze

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--engine-seed-base", type=int, default=None)
    parser.add_argument("--driver-seed-base", type=int, default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--probe-games",
        type=int,
        default=4000,
        help="games (from the first) that get a fork",
    )
    parser.add_argument("--forks-per-offer", type=int, default=4)
    parser.add_argument(
        "--tag", default="", help="result-name suffix (use 'smoke' for smoke runs)"
    )
    args = parser.parse_args()

    base, n_default = SEED_RANGES[args.arm]
    games = args.games or n_default
    engine_base = args.engine_seed_base or base
    driver_base = args.driver_seed_base or base
    if args.workers > 6:
        raise SystemExit("CPU cap for this experiment is 6 workers")

    t0 = time.perf_counter()
    payload, samples = collect(
        args.arm,
        games=games,
        workers=args.workers,
        engine_seed_base=engine_base,
        driver_seed_base=driver_base,
        checkpoint=args.checkpoint,
        probe_games=args.probe_games,
        fork_pairs=args.forks_per_offer,
    )
    rows = stack_samples(samples)
    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    stem = _result_name(args.arm, args.checkpoint) + (
        f"_{args.tag}" if args.tag else ""
    )
    npz = NPZ_DIR / f"{stem}.npz"
    np.savez_compressed(npz, **rows)  # type: ignore[arg-type]  # keys are ours

    result = {
        "git_commit": payload["git_commit"],
        "generated_at": payload["generated_at"],
        "experiment": "critic_calibration",
        "arm": args.arm,
        "mode": ARMS[args.arm],
        "n_games": games,
        "engine_seed_base": engine_base,
        "driver_seed_base": driver_base,
        "forks_per_offer": args.forks_per_offer,
        "probe_games": args.probe_games,
        "checkpoint": Path(args.checkpoint).name,
        "checkpoint_sha256_12": _sha12(args.checkpoint),
        "module_paths": payload["module_paths"],
        "by_role": payload["by_role"],
        "wall_seconds": time.perf_counter() - t0,
        "raw_rows_npz": str(npz.relative_to(ROOT)),
        "analysis": analyze(rows, arm=args.arm),
    }
    print(f"wrote {write_result(RESULTS_DIR, stem, result)}")


if __name__ == "__main__":
    main()
