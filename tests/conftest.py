"""Suite-wide guard: no test may write game records into the repo."""

from pathlib import Path

import pytest

import server.persistence as persistence_mod


@pytest.fixture(autouse=True)
def _games_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    path = tmp_path_factory.mktemp("catan-games")
    monkeypatch.setattr(persistence_mod, "GAMES_DIR", path)
    return path
