"""In-memory game-session store: multiple concurrent games (README decision
22), keyed by a random ``game_id``, with TTL eviction checked on access.

Deliberately has no dependency on ``server/bots.py``'s seat-kind vocabulary
(``"random"``/``"stratified_random"``/``"heuristic"``) -- ``create_session``
takes already-constructed ``Agent`` instances (``None`` for a human seat),
built by the caller. Two reasons: it keeps this module ignorant of which bot
classes exist, and it avoids a ``sessions.py``<->``bots.py`` import cycle,
since ``bots.py`` needs ``GameSession`` as a type for ``step_bots``.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Literal

from agents import Agent
from engine.game import CatanGame
from engine.state import GameState

_TTL_SECONDS = 2 * 60 * 60  # 2 hours -- a local dev/demo tool, not a service


class SessionNotFoundError(Exception):
    """Raised when a ``game_id`` has no active session (missing or evicted)."""


@dataclass(slots=True)
class GameSession:
    game: CatanGame
    state: GameState
    agents: list[Agent | None]  # one per player_id; ``None`` marks a human seat
    last_touched: float = field(default_factory=time.monotonic)

    def seat_kind(self, player_id: int) -> Literal["human", "bot"]:
        return "human" if self.agents[player_id] is None else "bot"


_SESSIONS: dict[str, GameSession] = {}


def _sweep_expired(now: float) -> None:
    expired = [
        game_id
        for game_id, session in _SESSIONS.items()
        if now - session.last_touched > _TTL_SECONDS
    ]
    for game_id in expired:
        del _SESSIONS[game_id]


def create_session(
    num_players: int, agents: list[Agent | None], seed: int | None = None
) -> tuple[str, GameSession]:
    """Start a new game and store it, returning its ``game_id``.

    ``agents`` must have one entry per player id (``None`` for a human seat).
    Does not run any bot turns -- the caller runs ``server.bots.step_bots``
    on the returned session afterward if the first seat to act is a bot.
    """
    if len(agents) != num_players:
        raise ValueError(f"expected {num_players} agents, got {len(agents)}")
    _sweep_expired(time.monotonic())
    game = CatanGame(num_players=num_players)
    state = game.reset(seed=seed)
    game_id = secrets.token_urlsafe(16)
    _SESSIONS[game_id] = GameSession(game=game, state=state, agents=agents)
    return game_id, _SESSIONS[game_id]


def get_session(game_id: str) -> GameSession:
    now = time.monotonic()
    _sweep_expired(now)
    session = _SESSIONS.get(game_id)
    if session is None:
        raise SessionNotFoundError(game_id)
    session.last_touched = now
    return session


def delete_session(game_id: str) -> None:
    _SESSIONS.pop(game_id, None)
