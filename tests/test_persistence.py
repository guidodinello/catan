"""``server/persistence.py``: pickle-to-disk session snapshots.

Rule under test: a snapshot saved for a live session round-trips into a
still-playable ``GameSession`` (state equivalent, RNG stream continuous,
next legal action still applies); a snapshot whose format/shape/age no
longer matches the running code is discarded (logged), never resurrected
half-broken; and ``save_session``/``delete_snapshot`` track ``sessions.py``'s
own create/delete lifecycle one-to-one.

Every test monkeypatches ``SESSION_DIR`` to ``tmp_path`` so this suite never
touches the real ``.catan-sessions/``.
"""

from __future__ import annotations

import pickle
import random
import time
from pathlib import Path

import pytest

import server.persistence as persistence_mod
import server.sessions as sessions_mod
from agents import Agent, RandomAgent
from server.persistence import (
    _Snapshot,
    _snapshot_path,
    delete_snapshot,
    load_all,
    save_session,
    shape_fingerprint,
)
from server.sessions import GameSession, create_session


@pytest.fixture(autouse=True)
def _session_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(persistence_mod, "SESSION_DIR", tmp_path)
    return tmp_path


def _bot_session(seed: int = 1) -> tuple[str, GameSession]:
    agents: list[Agent | None] = [
        None,
        RandomAgent(random.Random(0), name="random"),
        None,
    ]
    return create_session(3, agents, seed=seed)


def test_save_then_load_all_round_trips_a_playable_session() -> None:
    game_id, session = _bot_session()
    save_session(game_id, session)

    loaded = load_all()

    assert set(loaded) == {game_id}
    restored = loaded[game_id]
    assert restored.state.phase == session.state.phase
    assert restored.seat_kind(0) == "human"
    assert restored.seat_kind(1) == "bot"
    # A usable game, not just an equal object: legal_actions still computes
    # and the first one still applies without error.
    legal = restored.game.legal_actions(restored.state)
    assert legal
    restored.game.apply_action(restored.state, legal[0])


def test_rng_stream_continues_after_a_round_trip() -> None:
    game_id, session = _bot_session()
    expected_next = [session.state.rng.random() for _ in range(5)]

    # Re-create identically and save *before* those draws happened, so the
    # loaded copy's RNG is at the same position the original was before the
    # draws above.
    game_id, session = _bot_session()
    save_session(game_id, session)
    restored = load_all()[game_id]

    actual_next = [restored.state.rng.random() for _ in range(5)]
    assert actual_next == expected_next


def test_fingerprint_mismatch_is_discarded_others_still_load(
    caplog: pytest.LogCaptureFixture,
) -> None:
    good_id, good_session = _bot_session()
    save_session(good_id, good_session)

    bad_id, bad_session = _bot_session(seed=2)
    snapshot = _Snapshot(
        format_version=persistence_mod.FORMAT_VERSION,
        fingerprint="stale-shape-fingerprint",
        saved_at=time.time(),
        num_players=3,
        seat_kinds=["human", "human", "human"],
        state=bad_session.state,
    )
    with _snapshot_path(bad_id).open("wb") as f:
        pickle.dump(snapshot, f)

    with caplog.at_level("WARNING"):
        loaded = load_all()

    assert set(loaded) == {good_id}
    assert any("shape has changed" in message for message in caplog.messages)


def test_expired_snapshot_is_dropped_on_load() -> None:
    game_id, session = _bot_session()
    save_session(game_id, session)
    path = _snapshot_path(game_id)
    with path.open("rb") as f:
        snapshot: _Snapshot = pickle.load(f)
    stale = _Snapshot(
        format_version=snapshot.format_version,
        fingerprint=snapshot.fingerprint,
        saved_at=time.time() - sessions_mod._TTL_SECONDS - 1,
        num_players=snapshot.num_players,
        seat_kinds=snapshot.seat_kinds,
        state=snapshot.state,
    )
    with path.open("wb") as f:
        pickle.dump(stale, f)

    assert load_all() == {}


def test_corrupt_snapshot_file_is_skipped_not_fatal(tmp_path: Path) -> None:
    good_id, good_session = _bot_session()
    save_session(good_id, good_session)
    (tmp_path / "garbage.pickle").write_bytes(b"not a pickle at all")

    loaded = load_all()

    assert set(loaded) == {good_id}


def test_delete_snapshot_removes_the_file_and_is_a_no_op_when_missing() -> None:
    game_id, session = _bot_session()
    save_session(game_id, session)
    assert _snapshot_path(game_id).exists()

    delete_snapshot(game_id)

    assert not _snapshot_path(game_id).exists()
    delete_snapshot(game_id)  # no error on a second, redundant delete


def test_shape_fingerprint_is_stable_across_calls() -> None:
    assert shape_fingerprint() == shape_fingerprint()


def test_load_all_returns_empty_when_directory_does_not_exist(tmp_path: Path) -> None:
    empty_dir = tmp_path / "does-not-exist"
    persistence_mod.SESSION_DIR = empty_dir
    assert load_all() == {}
