"""The ``rl`` seat's availability handling and the human-game records.

Rules under test: the server never imports torch unless an ``rl`` seat is
actually built; an unavailable ``rl`` seat is reported with a reason and
refused with a 400; every human-involved game leaves a JSON record that
starts as ``"started"`` and only becomes ``"finished"`` with an outcome;
``turn_count`` follows ``experiments/rollout.py``'s definition.
"""

import json
import random
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

import server.bots as bots_mod
import server.persistence as persistence_mod
from agents import HeuristicAgent
from experiments.rollout import run_game
from server.app import app
from server.bots import RLSeatUnavailableError, build_agent, rl_availability, step_bots
from server.sessions import create_session

client = TestClient(app)


@pytest.fixture(autouse=True)
def _session_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(persistence_mod, "SESSION_DIR", tmp_path / "sessions")
    return tmp_path


@pytest.fixture
def rl_unavailable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    missing = tmp_path / "nope.zip"
    monkeypatch.setenv(bots_mod.RL_CHECKPOINT_ENV, str(missing))
    return missing


def _create(seat_kinds: list[str], seed: int | None = 1) -> dict[str, Any]:
    response = client.post(
        "/api/games",
        json={"num_players": len(seat_kinds), "seat_kinds": seat_kinds, "seed": seed},
    )
    assert response.status_code == 200, response.text
    return cast(dict[str, Any], response.json())


def _record(games_dir: Path, game_id: str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((games_dir / f"{game_id}.json").read_text()))


def test_rl_availability_reports_a_missing_checkpoint(rl_unavailable: Path) -> None:
    ok, reason = rl_availability()
    if ok:  # pragma: no cover - checkpoint path was overridden to a missing file
        pytest.fail("a missing checkpoint must be unavailable")
    assert reason is not None
    assert str(rl_unavailable) in reason or "not installed" in reason


def test_rl_availability_reports_missing_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bots_mod.importlib.util, "find_spec", lambda name: None)
    ok, reason = rl_availability()
    assert not ok
    assert reason is not None and "rl-train" in reason


def test_build_agent_rl_raises_with_the_reason(rl_unavailable: Path) -> None:
    with pytest.raises(RLSeatUnavailableError):
        build_agent("rl", random.Random(0))


def test_seat_kinds_endpoint_marks_rl_unavailable_with_a_reason(
    rl_unavailable: Path,
) -> None:
    body = client.get("/api/seat_kinds").json()
    by_kind = {k["kind"]: k for k in body["kinds"]}
    assert by_kind["heuristic"]["available"] and by_kind["heuristic"]["reason"] is None
    assert not by_kind["rl"]["available"]
    assert by_kind["rl"]["reason"]
    assert body["rl_checkpoint"] is None


def test_creating_an_rl_game_when_unavailable_is_a_400(rl_unavailable: Path) -> None:
    response = client.post(
        "/api/games",
        json={"num_players": 3, "seat_kinds": ["human", "rl", "rl"], "seed": 1},
    )
    assert response.status_code == 400
    assert "rl seat unavailable" in response.json()["detail"]


def test_create_returns_the_engine_seed_and_draws_one_when_omitted() -> None:
    assert _create(["human", "heuristic", "heuristic"], seed=5)["engine_seed"] == 5
    assert isinstance(
        _create(["human", "heuristic", "heuristic"], seed=None)["engine_seed"], int
    )


def test_a_human_game_gets_a_started_record(_games_dir: Path) -> None:
    body = _create(["human", "heuristic", "heuristic"], seed=7)
    record = _record(_games_dir, body["game_id"])
    assert record["status"] == "started"
    assert record["seat_kinds"] == ["human", "heuristic", "heuristic"]
    assert record["human_seats"] == [0]
    assert record["engine_seed"] == 7
    assert record["driver_seed"] == body["driver_seed"]
    assert record["trade_policy"] == "reject_all"
    assert record["rl_checkpoint"] is None
    assert "winner" not in record


def test_a_bot_only_game_gets_no_record(_games_dir: Path) -> None:
    _create(["heuristic", "heuristic", "heuristic"], seed=1)
    assert list(_games_dir.glob("*.json")) == []


def test_a_deleted_unfinished_game_keeps_its_started_record(
    _games_dir: Path,
) -> None:
    game_id = _create(["human", "heuristic", "heuristic"], seed=1)["game_id"]
    assert client.delete(f"/api/games/{game_id}").status_code == 200
    assert _record(_games_dir, game_id)["status"] == "started"


def test_a_finished_game_record_has_the_outcome(_games_dir: Path) -> None:
    game_id = _create(["human", "heuristic", "heuristic"], seed=1)["game_id"]
    for _ in range(2000):
        state = client.get(f"/api/games/{game_id}/state").json()
        if state["winner"] is not None:
            break
        response = client.post(f"/api/games/{game_id}/action", json={"index": 0})
        assert response.status_code == 200, response.text
    record = _record(_games_dir, game_id)
    assert record["status"] == "finished"
    assert record["winner"] == state["winner"]
    assert record["winner_kind"] == record["seat_kinds"][state["winner"]]
    assert len(record["vp_true"]) == len(record["vp_public"]) == 3
    assert max(record["vp_true"]) >= 10
    assert record["turn_count"] > 0
    assert record["engine_seed"] == 1


def test_turn_count_matches_the_rollout_definition() -> None:
    def factory(num_players: int, engine_seed: int, driver_seed: int) -> list[Any]:
        return [HeuristicAgent() for _ in range(num_players)]

    expected = run_game(3, 4, 1, agent_factory=factory).turn_count
    agents: list[Any] = [HeuristicAgent() for _ in range(3)]
    _, session = create_session(3, agents, seed=4)
    step_bots(session)
    assert session.game.is_terminal(session.state)
    assert session.turn_count == expected


def test_a_snapshot_round_trips_turn_count() -> None:
    game_id = _create(["human", "heuristic", "heuristic"], seed=1)["game_id"]
    for _ in range(4):
        client.post(f"/api/games/{game_id}/action", json={"index": 0})
    from server.sessions import get_session

    before = get_session(game_id).turn_count
    restored = persistence_mod.load_all()[game_id]
    assert restored.turn_count == before


def test_server_import_and_non_rl_games_never_import_torch() -> None:
    # A subprocess: this process may already have torch imported by
    # tests/rl, so sys.modules here proves nothing.
    code = textwrap.dedent(
        """
        import sys, tempfile
        from pathlib import Path
        import server.persistence as p
        p.SESSION_DIR = Path(tempfile.mkdtemp())
        p.GAMES_DIR = Path(tempfile.mkdtemp())
        from fastapi.testclient import TestClient
        from server.app import app
        c = TestClient(app)
        r = c.post("/api/games", json={"num_players": 3,
            "seat_kinds": ["human", "heuristic", "heuristic"], "seed": 1})
        assert r.status_code == 200, r.text
        assert c.get("/api/seat_kinds").status_code == 200
        bad = sorted(m for m in ("torch", "sb3_contrib", "numpy", "rl")
                     if m in sys.modules)
        assert not bad, bad
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == 0, result.stderr
