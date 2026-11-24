"""Experiment 017 registration tests: roster, manifest, pairings, seeds.

Torch-free (nothing here builds an RL agent), so these run in CI. The 013
pairing files committed under ``experiments/results/league_013`` are inputs.
"""

import itertools
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from experiments import league_013 as lg13
from experiments import league_017 as lg

N_MAX = 10_000  # any n we could pick is covered


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _driver_ranges(seed: int, n: int) -> list[tuple[int, int]]:
    out = []
    for a, b in lg.pairings():
        _, base = lg.pairing_seeds(seed, a, b)
        out.append((base, base + n - 1))
    return out


# --- manifest and roster ---------------------------------------------------


def test_committed_manifest_is_well_formed() -> None:
    manifest = lg.load_manifest()
    assert len(manifest) == 4
    for name, entry in manifest.items():
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), name
        assert "/" not in entry["path"] and lg.SEP not in name
    old = lg13.load_manifest()
    assert set(manifest).isdisjoint(old) and set(manifest).isdisjoint(lg13.BASELINES)
    assert {e["sha256"] for e in manifest.values()}.isdisjoint(
        e["sha256"] for e in old.values()
    )


def test_pairings_are_the_26_registered_bridge_and_graph_pairs() -> None:
    pairs = lg.pairings()
    assert len(pairs) == len(set(pairs)) == 26
    new = set(lg.new_agents())
    assert all(a < b for a, b in pairs)
    assert set(lg.BRIDGE) <= set(lg13.roster(lg13.load_manifest()))
    for a, b in pairs:
        assert a in new or b in new  # a pairing among 013's agents is 013's, not ours
    bridge = [p for p in pairs if len(new & set(p)) == 1]
    assert len(bridge) == 20 and all(set(p) - new <= set(lg.BRIDGE) for p in bridge)
    assert len([p for p in pairs if set(p) <= new]) == 6


def test_win_graph_over_013_and_017_is_connected() -> None:
    agents = {*lg.roster_013(), *lg.new_agents()}
    edges = [*itertools.combinations(lg.roster_013(), 2), *lg.pairings()]
    seen, stack = set(), [lg.ANCHOR]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack += [y if x == node else x for x, y in edges if node in (x, y)]
    assert seen == agents


# --- seeds -----------------------------------------------------------------


def test_new_pairing_ranges_are_disjoint_from_everything_in_use() -> None:
    for seed in lg.ALLOWED_SEEDS:
        new = lg.pairing_ranges(seed, N_MAX)
        driver = _driver_ranges(seed, N_MAX)
        for r in (*new, *driver):
            assert not any(_overlaps(r, u) for u in lg13.USED_RANGES), (seed, r)
    registered = [
        *lg.pairing_ranges(lg.SEED, N_MAX),
        *lg.pairing_ranges(lg.TIMING_SEED, N_MAX),
    ]
    old = [*lg.ranges_013(lg.SEED, N_MAX), *lg.ranges_013(lg.TIMING_SEED, N_MAX)]
    assert len(registered) == 52 and len(old) == 110
    assert not any(_overlaps(x, y) for x, y in itertools.combinations(registered, 2))
    assert not any(_overlaps(x, y) for x in registered for y in old)


def test_registered_seed_is_013s_and_timing_seed_is_the_first_clean_one() -> None:
    assert lg.SEED == lg13.SEED == 20261013
    assert lg.TIMING_SEED not in (lg.SEED, lg13.TIMING_SEED)
    assert lg.ALLOWED_SEEDS == (lg.SEED, lg.TIMING_SEED)
    # 013's timing seed is skipped because one 017 range lands in 011/012's block.
    hits = [
        r
        for r in lg.pairing_ranges(lg13.TIMING_SEED, N_MAX)
        if any(_overlaps(r, u) for u in lg13.USED_RANGES)
    ]
    assert len(hits) == 1
    # The rule: the smallest seed above 013's that passes (the test above).
    assert lg.TIMING_SEED == lg13.TIMING_SEED + 1


# --- reuse of 013's committed files ---------------------------------------


def test_every_committed_013_file_is_valid_at_its_seed_and_n() -> None:
    payloads = lg.reuse_013()
    assert len(payloads) == 55
    assert {p["league_config"]["n_games"] for p in payloads} == {lg.N_013}
    assert {p["league_config"]["seed"] for p in payloads} == {lg.SEED}
    for p in payloads:
        a, b = p["league_config"]["a"], p["league_config"]["b"]
        assert (p["engine_seed_base"], p["driver_seed_base"]) == lg.pairing_seeds(
            lg.SEED, a, b
        )


def test_reuse_refuses_a_changed_config_or_a_missing_file(tmp_path: Path) -> None:
    copy = tmp_path / "league_013"
    shutil.copytree(lg.RESULTS_013, copy)
    assert len(lg.reuse_013(copy)) == 55
    target = copy / "bc_clone__vs__bc_ft_2m.json"
    payload: dict[str, Any] = json.loads(target.read_text())
    payload["config_hash"] = "0" * 12
    target.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="different config"):
        lg.reuse_013(copy)
    target.unlink()
    with pytest.raises(RuntimeError, match="missing"):
        lg.reuse_013(copy)


# --- run / summarize guards -----------------------------------------------


def _cfg(tmp: Path, **kw: Any) -> lg.RunConfig:
    base: dict[str, Any] = {
        "n": 8,
        "seed": lg.SEED,
        "workers": 1,
        "ckpt_root": tmp,
        "results_dir": tmp / "out",
        "name": "t",
    }
    return lg.RunConfig(**{**base, **kw})


def test_run_refuses_an_unregistered_pair(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not one of the registered"):
        lg.run(_cfg(tmp_path, pairs=(("heuristic", "random"),)))
    with pytest.raises(ValueError, match="not one of the registered"):
        lg.run(_cfg(tmp_path, pairs=(("bc_clone", "graph_s4_12m"),)))


def test_run_refuses_a_foreign_seed_and_a_bad_n(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="seed must be one of"):
        lg.run(_cfg(tmp_path, seed=1))
    with pytest.raises(ValueError, match="multiple of 4"):
        lg.run(_cfg(tmp_path, n=6))


def test_run_accepts_a_registered_pair_in_either_order(tmp_path: Path) -> None:
    pair = lg.pairings()[0]
    assert lg.normalize_pairs([pair[::-1]]) == (pair,)


def test_summarize_refuses_missing_pairings(tmp_path: Path) -> None:
    (tmp_path / "league").mkdir()
    with pytest.raises(ValueError, match="registered pairings missing"):
        lg.summarize(tmp_path, "league")


def _payload(wins_a: int, wins_b: int, n: int) -> dict[str, Any]:
    return {
        "league_config": {"a": "a", "b": "b", "n_games": n},
        "by_role": {"a": {"wins": wins_a}, "b": {"wins": wins_b}},
    }


def test_tie_stop_rule_fires_above_one_percent() -> None:
    lg.check_ties(_payload(1980, 1990, 4000))  # 30 ties = 0.75%
    lg.check_ties(_payload(2000, 1960, 4000))  # 40 ties = exactly 1%
    with pytest.raises(RuntimeError, match="stop and ask"):
        lg.check_ties(_payload(1990, 1960, 4000))  # 50 ties


def test_tie_rule_judges_a_pairing_once_not_again_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pair = lg.pairings()[0]
    tied = _payload(0, 0, 8)
    monkeypatch.setattr(lg, "verify_manifest", lambda *_a, **_k: None)
    monkeypatch.setattr(lg, "run_pairing", lambda *_a, **_k: tied)
    cfg = _cfg(tmp_path, pairs=(pair,))
    with pytest.raises(RuntimeError, match="stop and ask"):
        lg.run(cfg)  # played now, all ties: stops
    finished = tmp_path / "out" / "t" / f"{pair[0]}{lg.SEP}{pair[1]}.json"
    finished.parent.mkdir(parents=True)
    finished.write_text("{}")
    assert lg.run(cfg) == [tied]  # already on disk: resumes past it
