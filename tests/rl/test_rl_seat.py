"""The web GUI's ``rl`` seat (``server/bots.py``) against a real, tiny
``MaskablePPO`` checkpoint: the model loads once per process and on one
thread, domestic-trade responses are forced to reject, and a game with rl
seats runs through the HTTP API. Needs the training stack; skipped without it.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("sb3_contrib")
torch = pytest.importorskip("torch")

from fastapi.testclient import TestClient  # noqa: E402

import server.bots as bots_mod  # noqa: E402
import server.persistence as persistence_mod  # noqa: E402
from agents import rl_agent as rl_agent_mod  # noqa: E402
from engine.actions import AcceptTrade, RejectTrade  # noqa: E402
from engine.game import CatanGame  # noqa: E402
from engine.state import Phase  # noqa: E402
from rl.train import TrainConfig, build_model, build_vec_env  # noqa: E402
from server.app import app  # noqa: E402


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("tiny-rl")
    cfg = TrainConfig(
        envs=1,
        opponents="heuristic",
        n_steps=64,
        batch_size=64,
        n_epochs=1,
        net_arch=[16],
        torch_threads=1,
        run_dir=tmp / "run",
    )
    vec_env = build_vec_env(cfg)
    model = build_model(cfg, vec_env)
    path = tmp / "tiny.zip"
    model.save(path)
    vec_env.close()
    return path


@pytest.fixture(autouse=True)
def _dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(persistence_mod, "SESSION_DIR", tmp_path / "sessions")


def test_trade_responses_are_rejected_without_consulting_the_policy(
    tiny_checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = bots_mod.build_agent("rl", random.Random(0), tiny_checkpoint)

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("the policy must not decide trade responses")

    monkeypatch.setattr(rl_agent_mod.RLAgent, "choose_action", boom)
    state = CatanGame(num_players=4).reset(seed=1)
    state.phase = Phase.AWAIT_TRADE_RESPONSE
    chosen = agent.choose_action(state, [AcceptTrade(), RejectTrade()], 1)
    assert chosen == RejectTrade()


def test_model_loads_once_per_process_on_one_thread(tiny_checkpoint: Path) -> None:
    rl_agent_mod._load_model.cache_clear()
    for seed in range(3):
        bots_mod.build_agent("rl", random.Random(seed), tiny_checkpoint)
    info = rl_agent_mod._load_model.cache_info()
    assert info.misses == 1
    assert info.hits >= 2
    assert torch.get_num_threads() == 1


def test_a_game_with_rl_seats_runs_over_http_and_records_the_checkpoint(
    tiny_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    _games_dir: Path,
) -> None:
    monkeypatch.setenv(bots_mod.RL_CHECKPOINT_ENV, str(tiny_checkpoint))
    client = TestClient(app)
    seats = ["human", "rl", "rl", "rl"]
    response = client.post(
        "/api/games", json={"num_players": 4, "seat_kinds": seats, "seed": 1}
    )
    assert response.status_code == 200, response.text
    game_id = response.json()["game_id"]
    for _ in range(40):
        state = client.get(f"/api/games/{game_id}/state").json()
        if state["winner"] is not None:
            break
        step = client.post(f"/api/games/{game_id}/action", json={"index": 0})
        assert step.status_code == 200, step.text
    record = (_games_dir / f"{game_id}.json").read_text()
    assert '"stem": "tiny"' in record
    assert '"seat_kinds"' in record

    kinds = client.get("/api/seat_kinds").json()
    assert next(k for k in kinds["kinds"] if k["kind"] == "rl")["available"]
    assert kinds["rl_checkpoint"]["stem"] == "tiny"
