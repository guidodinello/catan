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


@dataclass(frozen=True, slots=True)
class DiceRoll:
    """One ``RollDice`` applied in a session: the turn it happened on (by
    ``turn_count``), who rolled, and the two dice. The engine keeps only the
    current roll, so this server-side list is the game's full roll history.
    """

    turn: int
    player_id: int
    d1: int
    d2: int

    @property
    def total(self) -> int:
        return self.d1 + self.d2


class SessionNotFoundError(Exception):
    """Raised when a ``game_id`` has no active session (missing or evicted)."""


@dataclass(slots=True)
class GameSession:
    game: CatanGame
    state: GameState
    agents: list[Agent | None]  # one per player_id; ``None`` marks a human seat
    last_touched: float = field(default_factory=time.monotonic)
    # Turns played so far, by ``experiments/rollout.py``'s definition (1 once
    # setup ends, +1 per ``EndTurn``) -- maintained by ``server/bots.py``.
    turn_count: int = 0
    # Path of the checkpoint any ``rl`` seat was built from (``None`` if there
    # is no rl seat) -- persisted so a restore rebuilds the same model.
    rl_checkpoint: str | None = None
    # Every ``RollDice`` applied so far, in order (``server/bots.py``'s
    # ``apply_to_session``). ``dice_history_complete`` is False when the game
    # was restored from a snapshot saved before this was tracked -- the rolls
    # before that point are unrecoverable, so counts are partial.
    dice_rolls: list[DiceRoll] = field(default_factory=list)
    dice_history_complete: bool = True

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
    num_players: int,
    agents: list[Agent | None],
    seed: int | None = None,
    rl_checkpoint: str | None = None,
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
    _SESSIONS[game_id] = GameSession(
        game=game, state=state, agents=agents, rl_checkpoint=rl_checkpoint
    )
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


def restore_sessions(sessions: dict[str, GameSession]) -> None:
    """Install sessions recovered from disk (``server/persistence.py``) into
    the in-memory store, at process startup only.

    A narrow setter rather than letting the caller poke ``_SESSIONS``
    directly -- this module owns the dict and ``_sweep_expired`` needs to
    stay consistent with whatever's in it. ``persistence.load_all`` has
    already dropped anything TTL-expired by wall clock, so no sweep is
    needed here; each restored session's ``last_touched`` is a fresh
    ``time.monotonic()`` set by the caller (see ``GameSession``'s
    docstring -- a persisted monotonic value would be meaningless after a
    restart).
    """
    _SESSIONS.update(sessions)
