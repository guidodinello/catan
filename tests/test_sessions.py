"""``server/sessions.py``: the in-memory, multi-game session store.

Rule under test: ``create_session`` stores a playable game keyed by a fresh
``game_id`` (README decision 22 -- multiple concurrent games, no disk
persistence); ``get_session`` returns that same session and refreshes its
``last_touched``, but raises ``SessionNotFoundError`` for an unknown or
TTL-evicted id; two sessions never interfere with each other.
"""

import random
import time

import pytest

import server.sessions as sessions_mod
from agents import RandomAgent
from engine.state import Phase
from server.sessions import (
    GameSession,
    SessionNotFoundError,
    create_session,
    delete_session,
    get_session,
)


def test_create_session_returns_a_playable_game() -> None:
    game_id, session = create_session(3, [None, None, None], seed=1)
    assert isinstance(session, GameSession)
    assert session.state.phase == Phase.SETUP_SETTLEMENT
    assert get_session(game_id) is session


def test_create_session_rejects_agent_count_mismatch() -> None:
    with pytest.raises(ValueError, match="expected 3 agents"):
        create_session(3, [None, None], seed=1)


def test_seat_kind_reports_human_for_none_and_bot_otherwise() -> None:
    bot = RandomAgent(random.Random(0), name="random")
    _, session = create_session(3, [None, bot, None], seed=1)
    assert session.seat_kind(0) == "human"
    assert session.seat_kind(1) == "bot"
    assert session.seat_kind(2) == "human"


def test_get_session_raises_for_unknown_game_id() -> None:
    with pytest.raises(SessionNotFoundError):
        get_session("does-not-exist")


def test_get_session_refreshes_last_touched() -> None:
    game_id, session = create_session(3, [None, None, None], seed=1)
    session.last_touched = time.monotonic() - 10
    stale = session.last_touched
    get_session(game_id)
    assert session.last_touched > stale


def test_get_session_evicts_sessions_past_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sessions_mod, "_TTL_SECONDS", 0)
    game_id, session = create_session(3, [None, None, None], seed=1)
    session.last_touched = time.monotonic() - 1  # already older than TTL=0
    with pytest.raises(SessionNotFoundError):
        get_session(game_id)


def test_delete_session_removes_it() -> None:
    game_id, _ = create_session(3, [None, None, None], seed=1)
    delete_session(game_id)
    with pytest.raises(SessionNotFoundError):
        get_session(game_id)


def test_delete_session_is_a_no_op_for_an_unknown_id() -> None:
    delete_session("does-not-exist")  # must not raise


def test_multiple_concurrent_games_are_independent() -> None:
    id_a, session_a = create_session(3, [None, None, None], seed=1)
    id_b, session_b = create_session(4, [None, None, None, None], seed=2)
    assert id_a != id_b
    assert get_session(id_a) is session_a
    assert get_session(id_b) is session_b
    assert session_a.state is not session_b.state
    assert len(session_a.state.players) == 3
    assert len(session_b.state.players) == 4
