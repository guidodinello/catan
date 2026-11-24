"""Experiment 017: graph agents join the 013 league ladder (4 seats).

See ``docs/experiments/017-graph-league.md``. 013 rated 8 flat-MLP checkpoints
and 3 baselines on one anchored Bradley-Terry ladder (55 pairings, n = 4000,
seed 20261013). 017 adds 4 graph-encoder checkpoints with 26 "bridge" pairings
and refits all 55 + 26 pairing files jointly.

Nothing from 013 is replayed: gamekit derives a pairing's seeds and
``config_hash`` from ``(seed, a, b, num_seats, n)`` only, never from the roster,
so 013's committed files are valid 017 inputs as they stand. ``reuse_013``
re-checks that through ``gamekit.league.run_pairing`` (a finished file with a
different hash raises), and ``summarize`` refuses to fit without it.

Everything else (agent construction, the process pool, telemetry, the 4-seat
lineup placement) is ``experiments.league_013``'s, unchanged. RL seats always
reject trade offers (``server.bots.RLSeatAgent``), as in 013.

Torch-free at import: RL agents are built lazily inside workers.

    uv run python -m experiments.league_017 run --n 2000 --workers 3 \\
        --ckpt-root <dir with the checkpoints by basename> \\
        --results-dir <persistent dir> --name league
"""

from __future__ import annotations

import argparse
import itertools
import logging
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gamekit.league import (
    ScheduledGame,
    load_pairings,
    pairing_seeds,
    run_pairing,
    summarize_league,
)
from gamekit.results import write_result

from experiments import league_013 as lg13
from experiments.league_013 import (
    ANCHOR,
    ANCHOR_RATING,
    BOOTSTRAP_SEED,
    N_BOOTSTRAP,
    NUM_SEATS,
    PRIOR_DRAWS,
    ROOT,
    SEED,
    SEP,
    Telemetry,
    verify_manifest,
)

log = logging.getLogger("league_017")

MANIFEST = Path(__file__).resolve().parent / "league_017_manifest.json"
RESULTS_013 = ROOT / "experiments" / "results" / "league_013"
N_013 = 4000  # 013's n per pairing: the n its committed files were made with

# 013's timing seed (20261014) cannot be reused: one 017 pairing's engine range at
# that seed (graph_s4_12m__vs__graph_s4_3m, from 4,403,394) falls inside 011/012's
# reserved smoke block 4,000,001..5,000,000. The rule, fixed before any run: the
# smallest seed above 013's that passes tests/test_league_017.py's disjointness
# test. That is 20261015. Its ranges are disjoint from every used range, from
# 013's registered and timing ranges, from the registered seed's, and from each
# other. No win rate from it enters any result.
TIMING_SEED = 20261015

# The only seeds a 017 run may use: the registered one (shared with 013, so the
# 55 existing files stay valid) and the timing / cross-machine one.
ALLOWED_SEEDS = (SEED, TIMING_SEED)

# The five agents at the top of 013's ladder (Elo -90 .. +16): the cells nearest
# 50%, where a game carries the most information about a graph agent's rating.
BRIDGE = ("ent_decay_3m", "heuristic", "long_10m", "long_8m", "trading_heuristic")

# A pairing with more than this share of games without a winner stops the run
# for the owner (013 had 0 ties in 220,000 games).
MAX_TIE_SHARE = 0.01


def load_manifest(path: Path = MANIFEST) -> dict[str, dict[str, str]]:
    return lg13.load_manifest(path)


def roster_013() -> list[str]:
    """013's 11 agents, sorted."""
    return lg13.roster(lg13.load_manifest())


def new_agents() -> list[str]:
    return sorted(load_manifest())


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def pairings() -> tuple[tuple[str, str], ...]:
    """The 26 pairings 017 plays: every new agent against ``BRIDGE``, and the
    new agents against each other. Names sorted inside each pair (gamekit's
    file-name convention)."""
    new = new_agents()
    pairs = {_pair(n, b) for n in new for b in BRIDGE}
    pairs |= {_pair(a, b) for a, b in itertools.combinations(new, 2)}
    return tuple(sorted(pairs))


def pairing_ranges(seed: int, n: int) -> list[tuple[int, int]]:
    """Inclusive engine-seed range of every 017 pairing at ``seed`` and ``n``."""
    out = []
    for a, b in pairings():
        base, _ = pairing_seeds(seed, a, b)
        out.append((base, base + n - 1))
    return out


def ranges_013(seed: int, n: int) -> list[tuple[int, int]]:
    """Inclusive engine-seed range of every 013 pairing at ``seed`` and ``n``."""
    return lg13.engine_ranges(roster_013(), seed, n)


def _never_play(_games: Sequence[ScheduledGame]) -> list[Any]:
    raise RuntimeError("this pairing file is missing or was not reused as is")


def reuse_013(directory: Path = RESULTS_013) -> list[dict[str, Any]]:
    """The 55 finished 013 pairing payloads, each checked by gamekit itself.

    ``run_pairing`` returns a finished file whose ``config_hash`` matches
    ``(seed, a, b, num_seats, n)`` and raises ``ValueError`` on a different one;
    ``_never_play`` turns a missing file into an error instead of playing it.
    """
    payloads = []
    for a, b in itertools.combinations(roster_013(), 2):
        payloads.append(
            run_pairing(
                a,
                b,
                num_seats=NUM_SEATS,
                n_per_pairing=N_013,
                seed=SEED,
                play=_never_play,
                winning_seat=lg13.winning_seat,
                results_dir=directory.parent,
                name=directory.name,
            )
        )
    return payloads


def ties_of(payload: dict[str, Any]) -> int:
    cfg = payload["league_config"]
    wins = sum(payload["by_role"][x]["wins"] for x in (cfg["a"], cfg["b"]))
    return int(cfg["n_games"] - wins)


def check_ties(payload: dict[str, Any]) -> None:
    """Stop rule: too many games without a winner means stop and ask."""
    n = payload["league_config"]["n_games"]
    ties = ties_of(payload)
    if ties > MAX_TIE_SHARE * n:
        a, b = payload["league_config"]["a"], payload["league_config"]["b"]
        raise RuntimeError(
            f"{a}{SEP}{b}: {ties}/{n} games without a winner "
            f"(> {MAX_TIE_SHARE:.0%}); stop and ask the owner"
        )


def normalize_pairs(
    pairs: Sequence[tuple[str, str]] | None,
) -> tuple[tuple[str, str], ...]:
    registered = pairings()
    if pairs is None:
        return registered
    out = []
    for a, b in pairs:
        pair = _pair(a, b)
        if pair not in registered:
            raise ValueError(f"{a}{SEP}{b} is not one of the registered pairings")
        out.append(pair)
    return tuple(out)


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


def _checkpoint_entries(
    pairs: Sequence[tuple[str, str]],
) -> dict[str, dict[str, str]]:
    """Manifest entries (013's and 017's) for the RL agents these pairs use."""
    both = {**lg13.load_manifest(), **load_manifest()}
    used = {name for pair in pairs for name in pair}
    return {k: v for k, v in both.items() if k in used}


def run(cfg: RunConfig) -> list[dict[str, Any]]:
    if cfg.n <= 0 or cfg.n % NUM_SEATS:
        raise ValueError(f"n must be a positive multiple of {NUM_SEATS}")
    if cfg.seed not in ALLOWED_SEEDS:
        raise ValueError(f"seed must be one of {ALLOWED_SEEDS}, got {cfg.seed}")
    pairs = normalize_pairs(cfg.pairs)
    entries = _checkpoint_entries(pairs)
    verify_manifest(entries, cfg.ckpt_root)
    log.info(
        "manifest verified: %d checkpoints for %d pairings", len(entries), len(pairs)
    )
    ckpt_paths = {k: str(cfg.ckpt_root / v["path"]) for k, v in entries.items()}
    telemetry = Telemetry(cfg.results_dir / f"{cfg.name}-telemetry")
    payloads: list[dict[str, Any]] = []
    with ProcessPoolExecutor(cfg.workers, initializer=lg13._worker_init) as pool:
        play = lg13.make_play(pool, ckpt_paths, telemetry, cfg.step_budget)
        for a, b in pairs:
            payload = run_pairing(
                a,
                b,
                num_seats=NUM_SEATS,
                n_per_pairing=cfg.n,
                seed=cfg.seed,
                play=play,
                winning_seat=lg13.winning_seat,
                results_dir=cfg.results_dir,
                name=cfg.name,
            )
            payloads.append(payload)
            check_ties(payload)
    return payloads


def summarize(
    results_dir: Path, name: str, dir_013: Path = RESULTS_013
) -> dict[str, Any]:
    """Joint fit over 013's 55 files and 017's 26; writes ``league.json``."""
    league_dir = results_dir / name
    new = load_pairings(league_dir)
    got = {(p["league_config"]["a"], p["league_config"]["b"]) for p in new}
    missing = set(pairings()) - got
    if missing:
        raise ValueError(
            f"{len(missing)} registered pairings missing: {sorted(missing)}"
        )
    if extra := got - set(pairings()):
        raise ValueError(f"unregistered pairing files: {sorted(extra)}")
    cfgs = {(p["league_config"]["seed"], p["league_config"]["n_games"]) for p in new}
    if len(cfgs) != 1 or next(iter(cfgs))[0] != SEED:
        raise ValueError(f"017 files must share one n at seed {SEED}, got {cfgs}")
    summary: dict[str, Any] = summarize_league(
        [*reuse_013(dir_013), *new],
        anchor=ANCHOR,
        anchor_rating=ANCHOR_RATING,
        prior_draws=PRIOR_DRAWS,
        n_bootstrap=N_BOOTSTRAP,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    write_result(league_dir, "league", summary)
    return summary


def check_checkpoints(ckpt_root: Path) -> None:
    """Load every 017 checkpoint, check its spaces against the current encoder
    and action layout, and play one game with it. Reads no win rate."""
    from agents.rl_agent import _load_model
    from rl.action_space import N_ATOMS
    from rl.encoder import OBS_DIM

    manifest = load_manifest()
    verify_manifest(manifest, ckpt_root)
    for name, entry in manifest.items():
        path = ckpt_root / entry["path"]
        model = _load_model(str(path), "cpu")
        obs_shape = model.observation_space.shape
        assert obs_shape is not None
        spaces = (int(obs_shape[0]), int(model.action_space.n))
        if spaces != (OBS_DIM, N_ATOMS):
            raise ValueError(f"{name}: spaces {spaces} != ({OBS_DIM}, {N_ATOMS})")
        game = ScheduledGame(
            engine_seed=TIMING_SEED, driver_seed=TIMING_SEED, lineup=(name,) * 4
        )
        out = lg13._play_one(game, {name: str(path)}, 20_000)
        log.info(
            "%s ok: %s policy, spaces match, 1 game, %.1fs",
            name,
            type(model.policy).__name__,
            out.worker_s,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="league_017")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for cmd in ("run", "summarize", "check-checkpoints"):
        p = sub.add_parser(cmd)
        p.add_argument("--results-dir", type=Path, default=Path("results/017"))
        p.add_argument("--name", default="league")
        p.add_argument("--ckpt-root", type=Path, default=ROOT)
        if cmd == "run":
            p.add_argument("--n", type=int, required=True)
            p.add_argument("--seed", type=int, default=SEED)
            p.add_argument("--workers", type=int, default=3)
            p.add_argument("--pairs", help=f"comma list of a{SEP}b to run alone")
        if cmd == "summarize":
            p.add_argument("--dir-013", type=Path, default=RESULTS_013)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _build_parser().parse_args(argv)
    if args.cmd == "run":
        run(
            RunConfig(
                n=args.n,
                seed=args.seed,
                workers=args.workers,
                ckpt_root=args.ckpt_root,
                results_dir=args.results_dir,
                name=args.name,
                pairs=lg13._parse_pairs(args.pairs),
            )
        )
    elif args.cmd == "summarize":
        summarize(args.results_dir, args.name, args.dir_013)
    else:
        check_checkpoints(args.ckpt_root)


if __name__ == "__main__":
    main()
