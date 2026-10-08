"""Experiment 013 driver tests: the N-seat reduction on real catan games.

Torch-free (heuristic / random / trading_heuristic agents only), so these run
in CI. The RL seat is covered in ``tests/rl/test_league_013_rl.py``.
"""

import itertools
import json
import re
from pathlib import Path
from typing import Any

import pytest
from gamekit.league import ScheduledGame, load_pairings, pairing_seeds

from experiments import league_013 as lg
from experiments.benchmark import _recover_seat_order
from experiments.rollout import run_game

AGENTS = ("heuristic", "random")
N = 8  # a multiple of NUM_SEATS


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _run(
    tmp: Path,
    *,
    n: int = N,
    workers: int = 1,
    pairs: tuple[tuple[str, str], ...] | None = None,
    name: str = "t",
    **kw: Any,
) -> dict[str, Any]:
    cfg = lg.RunConfig(
        n=n,
        seed=lg.SEED,
        workers=workers,
        ckpt_root=tmp,
        results_dir=tmp / "out",
        name=name,
        pairs=pairs,
        **kw,
    )
    return lg.run(cfg)


@pytest.fixture
def baseline_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """A manifest with no checkpoints: a baseline-only roster, nothing to hash."""
    monkeypatch.setattr(lg, "load_manifest", lambda _path=lg.MANIFEST: {})


# --- placement -------------------------------------------------------------


def test_lineup_is_placed_in_setup_order_without_a_second_rotation() -> None:
    for engine_seed in (1, 2, 3, 4, 5):
        lineup = ("random", "heuristic", "random", "heuristic")  # BABA-like
        agents = lg.league_factory(lineup, {}, 4, engine_seed, 7)
        order = _recover_seat_order(4, engine_seed)
        # the player who moves seat-th in setup order is lineup[seat]
        assert [agents[order[s]].name for s in range(4)] == list(lineup)


def test_winning_seat_names_the_winning_agent() -> None:
    lineup = ("heuristic", "random", "heuristic", "random")
    for engine_seed in range(1, 9):
        record = run_game(
            4, engine_seed, 1, lambda n, e, d: lg.league_factory(lineup, {}, n, e, d)
        )
        if record.winner is None:
            continue
        assert record.winning_seat is not None
        assert record.agent_names[record.winner] == lineup[record.winning_seat]


# --- the league on real games ----------------------------------------------


def test_four_seat_reduction_ranks_heuristic_above_random_and_balances_seats(
    tmp_path: Path, baseline_only: None
) -> None:
    summary = _run(tmp_path, pairs=(AGENTS,), n=40)
    (pairing,) = load_pairings(tmp_path / "out" / "t")
    assert pairing["league_config"]["num_seats"] == 4
    # each agent holds each seat n/2 times per seat pair: occupancy is exact
    for role in AGENTS:
        seats = pairing["by_role"][role]["seat_occupancy_counts"]
        assert len(set(seats.values())) == 1
    wins_h = pairing["by_role"]["heuristic"]["wins"]
    wins_r = pairing["by_role"]["random"]["wins"]
    assert wins_h > wins_r
    assert wins_h + wins_r <= 40
    elo = {k: v["elo"] for k, v in summary["ratings"].items()}
    assert elo["heuristic"] == 0.0  # the anchor
    assert elo["random"] < 0


def test_a_step_budget_hit_is_a_tie_excluded_from_the_fit(
    tmp_path: Path, baseline_only: None
) -> None:
    # Nobody can win in 30 steps, so every game is a tie. The pairing file is
    # written, but a league with no decisive game cannot be rated (gamekit
    # refuses the fit): the error is raised after the file is on disk.
    with pytest.raises(ValueError, match="not connected"):
        _run(tmp_path, pairs=(AGENTS,), step_budget=30)
    out = tmp_path / "out" / "t"
    (pairing,) = load_pairings(out)
    wins = sum(r["wins"] for r in pairing["by_role"].values())
    assert wins == 0
    telemetry = json.loads(
        (tmp_path / "out" / "t-telemetry" / f"heuristic{lg.SEP}random.json").read_text()
    )
    assert telemetry["ties"] == N


def test_results_do_not_depend_on_worker_count(
    tmp_path: Path, baseline_only: None
) -> None:
    for w in (1, 2):
        _run(tmp_path, pairs=(AGENTS,), workers=w, name=f"w{w}")
    digests = [
        json.loads(
            (
                tmp_path / "out" / f"w{w}-telemetry" / f"heuristic{lg.SEP}random.json"
            ).read_text()
        )
        for w in (1, 2)
    ]
    assert digests[0]["digests_sha256"] == digests[1]["digests_sha256"]
    assert digests[0]["winners_sha256"] == digests[1]["winners_sha256"]


def test_a_two_agent_pairing_equals_the_full_roster_pairing(
    tmp_path: Path, baseline_only: None
) -> None:
    _run(tmp_path, name="full")  # roster = the 3 baselines, 3 pairings
    _run(tmp_path, pairs=(AGENTS,), name="solo")
    name = f"heuristic{lg.SEP}random.json"
    full = json.loads((tmp_path / "out" / "full" / name).read_text())
    solo = json.loads((tmp_path / "out" / "solo" / name).read_text())
    for key in ("league_config", "config_hash", "by_role", "by_seat"):
        assert full[key] == solo[key]
    assert len(load_pairings(tmp_path / "out" / "full")) == 3


def test_resume_skips_finished_pairings_and_refuses_a_different_n(
    tmp_path: Path, baseline_only: None
) -> None:
    _run(tmp_path, pairs=(AGENTS,))
    f = tmp_path / "out" / "t" / f"heuristic{lg.SEP}random.json"
    before = f.read_text()
    _run(tmp_path, pairs=(AGENTS,))
    assert f.read_text() == before
    with pytest.raises(ValueError, match="different config"):
        _run(tmp_path, pairs=(AGENTS,), n=N + 4)


def test_n_must_be_a_positive_multiple_of_four(tmp_path: Path) -> None:
    for bad in (0, 6, 10):
        with pytest.raises(ValueError, match="multiple of 4"):
            _run(tmp_path, n=bad)


# --- seeds -----------------------------------------------------------------


def test_league_seeds_are_disjoint_from_every_used_range_and_each_other() -> None:
    manifest = lg.load_manifest()
    names = lg.roster(manifest)
    assert len(names) == 11 and len(list(itertools.combinations(names, 2))) == 55
    n_max = 10_000  # any n we could pick is covered
    for seed in (lg.SEED, lg.TIMING_SEED):
        ranges = lg.engine_ranges(names, seed, n_max)
        driver = [
            (pairing_seeds(seed, a, b)[1],) * 2
            for a, b in itertools.combinations(sorted(names), 2)
        ]
        driver = [(lo, lo + n_max - 1) for lo, _ in driver]
        for r in (*ranges, *driver):
            assert not any(_overlaps(r, u) for u in lg.USED_RANGES), (seed, r)
    both = lg.engine_ranges(names, lg.SEED, n_max) + lg.engine_ranges(
        names, lg.TIMING_SEED, n_max
    )
    assert not any(_overlaps(x, y) for x, y in itertools.combinations(both, 2))
    assert lg.SEED != lg.TIMING_SEED


# --- manifest --------------------------------------------------------------


def test_committed_manifest_is_well_formed() -> None:
    manifest = lg.load_manifest()
    assert len(manifest) == 8
    for name, entry in manifest.items():
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), name
        assert "/" not in entry["path"] and lg.SEP not in name
    assert set(lg.BASELINES).isdisjoint(manifest)
    assert lg.ANCHOR in lg.BASELINES


def test_verify_manifest_refuses_tampered_or_missing(tmp_path: Path) -> None:
    ckpt = tmp_path / "c.zip"
    ckpt.write_bytes(b"abc")
    good = {"x": {"path": "c.zip", "sha256": lg._sha256(ckpt)}}
    lg.verify_manifest(good, tmp_path)
    ckpt.write_bytes(b"abd")
    with pytest.raises(ValueError, match="sha256"):
        lg.verify_manifest(good, tmp_path)
    with pytest.raises(FileNotFoundError):
        lg.verify_manifest({"y": {"path": "nope.zip", "sha256": "0" * 64}}, tmp_path)


def test_schedule_game_type_is_the_public_one() -> None:
    assert ScheduledGame.__module__.startswith("gamekit")
