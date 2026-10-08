"""Experiment 013: the RL seat in the league (needs torch; skipped in CI)."""

from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("gymnasium")
pytest.importorskip("sb3_contrib")

from experiments import league_013 as lg  # noqa: E402
from experiments.benchmark import _recover_seat_order  # noqa: E402
from experiments.rollout import run_game  # noqa: E402
from rl.train import TrainConfig, build_model, build_vec_env  # noqa: E402


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> str:
    tmp = tmp_path_factory.mktemp("tiny-league")
    cfg = TrainConfig(
        envs=1, opponents="heuristic", n_steps=64, batch_size=64, n_epochs=1,
        net_arch=[16], torch_threads=1, run_dir=tmp / "run",
    )  # fmt: skip
    vec_env = build_vec_env(cfg)
    model = build_model(cfg, vec_env)
    path = tmp / "tiny.zip"
    model.save(path)
    vec_env.close()
    return str(path)


def test_rl_seats_are_named_by_league_name_and_placed_in_setup_order(
    tiny_checkpoint: str,
) -> None:
    lineup = ("tiny", "heuristic", "tiny", "heuristic")
    agents = lg.league_factory(lineup, {"tiny": tiny_checkpoint}, 4, 3, 5)
    order = _recover_seat_order(4, 3)
    assert [agents[order[s]].name for s in range(4)] == list(lineup)
    assert all(a.trade_policy == "reject_all" for a in agents if a.name == "tiny")  # type: ignore[attr-defined]


def test_a_game_with_rl_seats_plays_to_a_named_winner(tiny_checkpoint: str) -> None:
    lineup = ("tiny", "random", "tiny", "random")
    record = run_game(
        4,
        3,
        5,
        lambda n, e, d: lg.league_factory(lineup, {"tiny": tiny_checkpoint}, n, e, d),
    )
    assert record.agent_names.count("tiny") == 2
    if record.winner is not None:
        assert record.agent_names[record.winner] == lineup[record.winning_seat]  # type: ignore[index]


def test_committed_manifest_hashes_match_the_real_files_when_present() -> None:
    root = Path("rl_runs")
    found = {p.name: p for p in root.rglob("*.zip")} if root.is_dir() else {}
    manifest = lg.load_manifest()
    checked = 0
    for entry in manifest.values():
        if entry["path"] in found:
            assert lg._sha256(found[entry["path"]]) == entry["sha256"]
            checked += 1
    if not checked:
        pytest.skip("no roster checkpoints on this machine")
