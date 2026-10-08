"""Experiment 013: the first catan league on ``gamekit.league`` (4 seats).

Round robin over ``ROSTER`` (see ``docs/experiments/013-first-league.md``).
A pairing ``(a, b)`` fields the alternating lineup ``(a, b, a, b)`` in *setup
order* (rotated by ``engine_seed % 4`` inside ``gamekit.league``), so each
agent holds two of the four seats and "which agent's seat won" is the
pairwise outcome (note 022's N-seat reduction).

Two things differ from ``experiments.benchmark``:

* ``gamekit.league`` already rotates the lineup and hands it to ``play`` as
  ``ScheduledGame.lineup``; ``league_factory`` places it as given and must
  **not** rotate it again (``benchmark_agent_factory`` does rotate).
* Every RL seat is ``server.bots.RLSeatAgent`` (always rejects trade offers),
  so trade skill is not on this scale.

The full run is ONE serial ``run_league`` over the roster whose ``play``
callback fans a pairing's games out to a persistent process pool, so every
pairing file and ``league.json`` are written by a single process (no race,
unlike one ``run_league`` per pairing from a pool; gamekit#47 is still open).
``--pairs`` runs listed pairings as 2-agent leagues instead (timing smoke,
cross-machine check); pairing seeds and ``config_hash`` do not depend on the
roster, so those files equal the full run's.

Torch-free at import: RL agents are built lazily inside workers.

    uv run python -m experiments.league_013 run --n 4000 --workers 3 \\
        --ckpt-root <dir with the checkpoints> --results-dir <dir> --name league
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import random
import resource
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from gamekit.league import (
    ScheduledGame,
    load_pairings,
    pairing_seeds,
    run_league,
    summarize_league,
)
from gamekit.results import write_result
from gamekit.seats import seat_rng

from agents import CatanAgent, HeuristicAgent, RandomAgent, TradingHeuristicAgent
from experiments.benchmark import _recover_seat_order
from experiments.rollout import GameRecord, run_game

log = logging.getLogger("league_013")

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = Path(__file__).resolve().parent / "league_013_manifest.json"

NUM_SEATS = 4
NUM_PLAYERS = 4
ANCHOR = "heuristic"
ANCHOR_RATING = 0.0
PRIOR_DRAWS = 1.0
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 0
SEED = 20261013
TIMING_SEED = 20261014
BASELINES = ("heuristic", "random", "trading_heuristic")
PROGRESS_EVERY = 100
SEP = "__vs__"

# Engine-seed ranges already in use elsewhere (inclusive). Whole millions are
# reserved blocks (catan #38's 3000001+, 011/012's smoke 4000001+); fresh_012
# lives on the exp/46 branch. BC / training / in-loop eval boards are
# hash-spread over [0, 2**31) and cannot be listed.
USED_RANGES: tuple[tuple[int, int], ...] = (
    (1, 10_000),  # benchmark boards
    (90_001, 100_000),  # 008 smoke
    (500_001, 500_200),  # 009 smoke
    (1_000_001, 1_004_000),  # 009 arm H
    (2_000_001, 2_004_000),  # 009 arm T
    (3_000_001, 4_000_000),  # catan #38 (reserved block)
    (4_000_001, 5_000_000),  # 011 / 012 smoke
    (20_000_001, 20_004_000),  # fresh_012
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path = MANIFEST) -> dict[str, dict[str, str]]:
    manifest: dict[str, dict[str, str]] = json.loads(path.read_text())
    return manifest


def roster(manifest: dict[str, dict[str, str]]) -> list[str]:
    return sorted([*manifest, *BASELINES])


def verify_manifest(manifest: dict[str, dict[str, str]], ckpt_root: Path) -> None:
    """Refuse a missing or tampered checkpoint before any game is played."""
    for name, entry in manifest.items():
        path = ckpt_root / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"{name}: checkpoint {path} not found")
        got = _sha256(path)
        if got != entry["sha256"]:
            raise ValueError(
                f"{name}: sha256 {got[:12]} != manifest {entry['sha256'][:12]}"
            )


def engine_ranges(agents: Sequence[str], seed: int, n: int) -> list[tuple[int, int]]:
    """Every pairing's inclusive engine-seed range (hash-derived bases)."""
    out = []
    for a, b in itertools.combinations(sorted(agents), 2):
        base, _ = pairing_seeds(seed, a, b)
        out.append((base, base + n - 1))
    return out


def build_agent(
    name: str,
    ckpt_paths: dict[str, str],
    rng: random.Random,
) -> CatanAgent:
    """One agent by league name. RL agents always reject trade offers."""
    if name == "heuristic":
        return HeuristicAgent(name=name)
    if name == "trading_heuristic":
        return TradingHeuristicAgent(name=name)
    if name == "random":
        return RandomAgent(rng, name=name)
    from server.bots import RLSeatAgent  # lazy: pulls in numpy / torch

    agent = RLSeatAgent(Path(ckpt_paths[name]), rng)
    agent.name = name  # GameRecord.agent_names keys by it
    return agent


def league_factory(
    lineup: tuple[str, ...],
    ckpt_paths: dict[str, str],
    num_players: int,
    engine_seed: int,
    driver_seed: int,
) -> list[CatanAgent]:
    """``AgentFactory`` for one scheduled game. ``lineup`` is ALREADY rotated
    by ``gamekit.league``: ``lineup[seat]`` plays the seat that moves
    ``seat``-th in setup order. Placed as given -- no second rotation."""
    seat_order = _recover_seat_order(num_players, engine_seed)
    agents: list[CatanAgent | None] = [None] * num_players
    for seat in range(num_players):
        agents[seat_order[seat]] = build_agent(
            lineup[seat], ckpt_paths, seat_rng(driver_seed, seat)
        )
    assert all(a is not None for a in agents)
    return agents  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class GameOutcome:
    record: GameRecord
    worker_s: float
    peak_rss_mb: float


def _play_one(
    game: ScheduledGame,
    ckpt_paths: dict[str, str],
    step_budget: int,
) -> GameOutcome:
    t0 = time.process_time()
    record = run_game(
        NUM_PLAYERS,
        game.engine_seed,
        game.driver_seed,
        partial(league_factory, game.lineup, ckpt_paths),
        step_budget=step_budget,
    )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return GameOutcome(record, time.process_time() - t0, rss)


def _worker_init() -> None:
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


def record_digest(r: GameRecord) -> str:
    return hashlib.sha256(repr(r).encode()).hexdigest()[:16]


class Telemetry:
    """Per-pairing sidecar (outside the league dir, so ``load_pairings``
    ignores it): winners hash, record-digest hash, worker time, peak RSS."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def write(
        self, a: str, b: str, outcomes: Sequence[GameOutcome], wall_s: float
    ) -> None:
        winners = [o.record.winning_seat for o in outcomes]
        digests = [record_digest(o.record) for o in outcomes]
        write_result(
            self.directory,
            f"{a}{SEP}{b}",
            {
                "a": a,
                "b": b,
                "n": len(outcomes),
                "winners_sha256": hashlib.sha256(
                    json.dumps(winners).encode()
                ).hexdigest(),
                "digests_sha256": hashlib.sha256(
                    json.dumps(digests).encode()
                ).hexdigest(),
                "worker_s_total": sum(o.worker_s for o in outcomes),
                "worker_s_per_game": sum(o.worker_s for o in outcomes) / len(outcomes),
                "peak_rss_mb": max(o.peak_rss_mb for o in outcomes),
                "wall_s": wall_s,
                "ties": sum(w is None for w in winners),
            },
        )


def make_play(
    pool: ProcessPoolExecutor,
    ckpt_paths: dict[str, str],
    telemetry: Telemetry | None,
    step_budget: int,
) -> Callable[[Sequence[ScheduledGame]], list[GameRecord]]:
    def play(games: Sequence[ScheduledGame]) -> list[GameRecord]:
        t0 = time.perf_counter()
        a, b = sorted(set(games[0].lineup))
        futures = [pool.submit(_play_one, g, ckpt_paths, step_budget) for g in games]
        outcomes: list[GameOutcome] = []
        for i, f in enumerate(futures, 1):
            outcomes.append(f.result())
            if i % PROGRESS_EVERY == 0:
                log.info("%s%s%s %d/%d", a, SEP, b, i, len(games))
        wall = time.perf_counter() - t0
        winners = [o.record.winning_seat for o in outcomes]
        log.info(
            "%s%s%s done in %.0fs winners_sha256=%s",
            a,
            SEP,
            b,
            wall,
            hashlib.sha256(json.dumps(winners).encode()).hexdigest(),
        )
        if telemetry is not None:
            telemetry.write(a, b, outcomes, wall)
        return [o.record for o in outcomes]

    return play


def winning_seat(record: GameRecord) -> int | None:
    return record.winning_seat


@dataclass(frozen=True, slots=True)
class RunConfig:
    n: int
    seed: int
    workers: int
    ckpt_root: Path
    results_dir: Path
    name: str
    pairs: tuple[tuple[str, str], ...] | None = None
    step_budget: int = 20_000


def run(cfg: RunConfig) -> dict[str, Any]:
    if cfg.n <= 0 or cfg.n % NUM_SEATS:
        raise ValueError(f"n must be a positive multiple of {NUM_SEATS}")
    manifest = load_manifest()
    verify_manifest(manifest, cfg.ckpt_root)
    log.info("manifest verified: %d checkpoints", len(manifest))
    ckpt_paths = {k: str(cfg.ckpt_root / v["path"]) for k, v in manifest.items()}
    telemetry = Telemetry(cfg.results_dir / f"{cfg.name}-telemetry")
    summary: dict[str, Any] = {}
    with ProcessPoolExecutor(cfg.workers, initializer=_worker_init) as pool:
        play = make_play(pool, ckpt_paths, telemetry, cfg.step_budget)
        groups = (
            [roster(manifest)] if cfg.pairs is None else [list(p) for p in cfg.pairs]
        )
        for agents in groups:
            # Pairing seeds and config_hash are roster-independent, so a
            # 2-agent group writes the same pairing file as the full run.
            anchor = ANCHOR if ANCHOR in agents else sorted(agents)[0]
            summary = run_league(
                agents=agents,
                num_seats=NUM_SEATS,
                n_per_pairing=cfg.n,
                seed=cfg.seed,
                play=play,
                winning_seat=winning_seat,
                results_dir=cfg.results_dir,
                name=cfg.name,
                anchor=anchor,
                anchor_rating=ANCHOR_RATING,
                prior_draws=PRIOR_DRAWS,
                n_bootstrap=N_BOOTSTRAP if cfg.pairs is None else 10,
                bootstrap_seed=BOOTSTRAP_SEED,
            )
    return summary


def summarize(results_dir: Path, name: str) -> dict[str, Any]:
    league_dir = results_dir / name
    summary: dict[str, Any] = summarize_league(
        load_pairings(league_dir),
        anchor=ANCHOR,
        anchor_rating=ANCHOR_RATING,
        prior_draws=PRIOR_DRAWS,
        n_bootstrap=N_BOOTSTRAP,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    write_result(league_dir, "league", summary)
    return summary


def check_checkpoints(ckpt_root: Path) -> None:
    """Load every manifest checkpoint, check its spaces against the current
    encoder / action layout, and play one game with it."""
    from rl.action_space import N_ATOMS
    from rl.encoder import OBS_DIM

    manifest = load_manifest()
    verify_manifest(manifest, ckpt_root)
    for name, entry in manifest.items():
        path = ckpt_root / entry["path"]
        from agents.rl_agent import _load_model

        model = _load_model(str(path), "cpu")
        obs_shape = model.observation_space.shape
        assert obs_shape is not None
        obs_dim = int(obs_shape[0])
        n_actions = int(model.action_space.n)
        if (obs_dim, n_actions) != (OBS_DIM, N_ATOMS):
            raise ValueError(
                f"{name}: spaces ({obs_dim}, {n_actions}) != ({OBS_DIM}, {N_ATOMS})"
            )
        game = ScheduledGame(
            engine_seed=TIMING_SEED, driver_seed=TIMING_SEED, lineup=(name,) * 4
        )
        out = _play_one(game, {name: str(path)}, 20_000)
        log.info(
            "%s ok: spaces match, 1 game, winner seat %s, %.1fs",
            name,
            out.record.winning_seat,
            out.worker_s,
        )


def _parse_pairs(text: str | None) -> tuple[tuple[str, str], ...] | None:
    if not text:
        return None
    pairs = []
    for item in text.split(","):
        a, sep, b = item.partition(SEP)
        if not sep:
            raise SystemExit(f"--pairs entries look like a{SEP}b, got {item!r}")
        pairs.append((a, b))
    return tuple(pairs)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(prog="league_013")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for cmd in ("run", "summarize", "check-checkpoints"):
        p = sub.add_parser(cmd)
        p.add_argument("--results-dir", type=Path, default=Path("results/013"))
        p.add_argument("--name", default="league")
        p.add_argument("--ckpt-root", type=Path, default=ROOT)
        if cmd == "run":
            p.add_argument("--n", type=int, required=True)
            p.add_argument("--seed", type=int, default=SEED)
            p.add_argument("--workers", type=int, default=3)
            p.add_argument("--pairs", help=f"comma list of a{SEP}b to run alone")
    args = parser.parse_args(argv)
    if args.cmd == "run":
        run(
            RunConfig(
                n=args.n,
                seed=args.seed,
                workers=args.workers,
                ckpt_root=args.ckpt_root,
                results_dir=args.results_dir,
                name=args.name,
                pairs=_parse_pairs(args.pairs),
            )
        )
    elif args.cmd == "summarize":
        summarize(args.results_dir, args.name)
    else:
        check_checkpoints(args.ckpt_root)


if __name__ == "__main__":
    main()
