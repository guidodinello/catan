"""``experiments/human_games.py``: tabulating experiment 007's game records.

All records here are **synthetic test fixtures** built in ``tmp_path`` -- no
real human results exist yet. Rules under test: the schedule is the
pre-registered 24 slots; an unfinished game counts as a human loss; Wilson
CIs come from ``gamekit.mc``; off-schedule and repeated games are flagged and
excluded; one result never mixes checkpoints; the result name is
checkpoint-qualified and nothing is written when there is nothing to tabulate.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from gamekit.mc import wilson_interval

from experiments import human_games as hg

CKPT = {"path": "/x/ckpt.zip", "stem": "ckpt", "sha256_12": "aaaaaaaaaaaa"}


def _record(
    slot: hg.Slot,
    *,
    finished: bool = True,
    human_wins: bool = False,
    n: int = 0,
    checkpoint: dict[str, str] | None = CKPT,
) -> dict[str, Any]:
    kinds = ["human" if i == slot.human_seat else slot.bot_kind for i in range(4)]
    winner = slot.human_seat if human_wins else (slot.human_seat + 1) % 4
    rec: dict[str, Any] = {
        "game_id": f"g{slot.game_no}-{n}",
        "experiment": "007",
        "status": "finished" if finished else "started",
        "created_at": f"2026-10-01T00:{slot.game_no:02d}:{n:02d}+00:00",
        "num_players": 4,
        "seat_kinds": kinds,
        "human_seats": [slot.human_seat],
        "engine_seed": slot.engine_seed,
        "trade_policy": "reject_all",
        "rl_checkpoint": checkpoint if slot.bot_kind == "rl" else None,
        "turn_count": 50,
    }
    if finished:
        rec["winner"] = winner
        rec["winner_kind"] = kinds[winner]
        rec["vp_true"] = [10 if i == winner else 6 for i in range(4)]
    return rec


def test_schedule_is_the_preregistered_balanced_design() -> None:
    schedule = hg.SCHEDULE_007
    assert len(schedule) == 24
    assert [s.game_no for s in schedule] == list(range(1, 25))
    for kind in hg.BOT_KINDS:
        seats = [s.human_seat for s in schedule if s.bot_kind == kind]
        assert sorted(seats) == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    # paired boards: each seed appears once per bot kind, same seat
    for seed in {s.engine_seed for s in schedule}:
        pair = [s for s in schedule if s.engine_seed == seed]
        assert {s.bot_kind for s in pair} == set(hg.BOT_KINDS)
        assert len({s.human_seat for s in pair}) == 1
    assert schedule[0].bot_kind == "rl" and schedule[2].bot_kind == "heuristic"


def test_counts_wilson_and_abandoned_as_human_loss() -> None:
    rl_slots = [s for s in hg.SCHEDULE_007 if s.bot_kind == "rl"]
    records = [
        _record(rl_slots[0], human_wins=True),
        _record(rl_slots[1], human_wins=True),
        _record(rl_slots[2], human_wins=False),
        _record(rl_slots[3], finished=False),  # abandoned
    ]
    payload = hg.tabulate(records)
    arm = payload["arms"]["rl"]
    assert (arm["played"], arm["finished"], arm["abandoned"]) == (4, 3, 1)
    assert arm["human_wins"] == 2
    ci = wilson_interval(2, 4)
    assert arm["human_win_rate_wilson95"] == {"lower": ci.lower, "upper": ci.upper}
    assert arm["missing"] == 8
    assert payload["abandoned"] == [records[3]["game_id"]]
    assert payload["arms"]["heuristic"]["played"] == 0
    assert payload["arms"]["heuristic"]["human_win_rate_wilson95"] is None
    assert payload["rl_vs_heuristic"] is None


def test_verdict_is_relative_to_chance() -> None:
    assert hg._verdict(12, 12) == "human clearly stronger than this bot"
    assert hg._verdict(0, 12) == "bot stronger than this human"
    assert hg._verdict(3, 12) == "not distinguishable from chance"


def test_off_schedule_and_repeated_games_are_flagged_and_excluded() -> None:
    slot = hg.SCHEDULE_007[0]
    first = _record(slot, human_wins=True, n=1)
    repeat = _record(slot, human_wins=False, n=2)
    wrong_seed = {**_record(slot, n=3), "engine_seed": 1}
    wrong_seat = {**_record(slot, n=4), "human_seats": [3]}
    payload = hg.tabulate([repeat, wrong_seat, first, wrong_seed])
    assert payload["arms"]["rl"]["played"] == 1
    assert payload["arms"]["rl"]["human_wins"] == 1  # earliest attempt counts
    flagged = {d["game_id"] for d in payload["deviations"]}
    assert flagged == {repeat["game_id"], wrong_seat["game_id"], wrong_seed["game_id"]}


def test_mixed_rl_checkpoints_are_rejected() -> None:
    a, b = [s for s in hg.SCHEDULE_007 if s.bot_kind == "rl"][:2]
    other = {**CKPT, "sha256_12": "bbbbbbbbbbbb"}
    with pytest.raises(ValueError, match="different checkpoints"):
        hg.tabulate([_record(a), _record(b, checkpoint=other)])


def test_result_name_is_checkpoint_qualified() -> None:
    assert hg.result_name("007", "ckpt") == "human_games_007_p4_ckpt"
    assert hg.result_name("007", None) == "human_games_007_p4"


def test_main_writes_a_stamped_result_and_nothing_when_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = tmp_path / "results"
    monkeypatch.setattr(hg, "RESULTS_DIR", results)
    games = tmp_path / "games"
    games.mkdir()
    monkeypatch.setattr("sys.argv", ["human_games", "--games-dir", str(games)])
    assert hg.main() == 1
    assert not results.exists()

    slot = hg.SCHEDULE_007[0]
    (games / "a.json").write_text(json.dumps(_record(slot, human_wins=True)))
    assert hg.main() == 0
    written = json.loads((results / "human_games_007_p4_ckpt.json").read_text())
    assert written["experiment"] == "human_games"
    assert written["arms"]["rl"]["human_wins"] == 1
    assert len(written["games"]) == 24
