"""On-disk session persistence: survives a backend restart (README decision
22, ``docs/backlog.md``'s "Sessions don't survive a backend restart").

The only module that knows both ``server/sessions.py`` and ``server/bots.py``
-- keeps ``sessions.py`` ignorant of the seat-kind vocabulary (its own
docstring) and avoids a ``sessions.py``<->``bots.py`` import cycle, same
reasoning ``bots.py``'s own docstring gives for its ``TYPE_CHECKING`` import.

**Format: pickle, gated by a shape fingerprint.** ``engine.state.GameState``
is plain dataclasses/dicts/lists/Enums plus a ``random.Random`` -- pickle
round-trips it exactly (including the RNG's stream position) for free. The
catch, measured rather than assumed:

- A dataclass field renamed or removed raises ``AttributeError`` **at load**
  (slots dataclasses have no ``__dict__`` to silently absorb an unknown key).
- A dataclass field added raises ``AttributeError`` on the **first read** of
  the new field -- loud, but deferred to a later request.
- An ``Enum`` member inserted mid-list (``Phase``, ``DevCardType``,
  ``Resource``, ``Terrain``, ``PortType`` all number via ``auto()``) is
  **silent corruption**: ``Enum.__reduce_ex__`` restores *by value*, so every
  member after the insertion renumbers and an old snapshot loads a
  *different*, wrong member with no exception at all.

``shape_fingerprint()`` covers both axes -- dataclass field names and enum
name->value maps -- so a real reshape is caught before it's ever silently
trusted, not just when it happens to raise. A fingerprint mismatch means the
snapshot is discarded (logged), which is exactly today's behavior for a lost
session -- never a worse outcome than not persisting at all.

**When: write-through after every mutation.** The restart this exists to
survive is the one a code change forces -- Ctrl+C or ``--reload`` -- and
that loop also covers crashes and ``kill -9``, which a shutdown-hook-only
design would not. A few-KB pickle per POST is free for a local dev tool.

**Bot RNG: not persisted.** Only ``seat_kinds`` (``agent.name``, or
``"human"`` for a human seat -- ``server/bots.py``'s own ``SeatKind``
vocabulary) are saved; ``build_agents`` rebuilds fresh agents with a newly
drawn ``driver_seed`` on load, exactly as a new game does. Per the backlog,
losing a bot's exact RNG position only costs perfect reproducibility, not
correctness -- and it keeps every class in ``agents/`` out of the save
format, so nothing there can invalidate a snapshot.

**Security note:** ``pickle.load`` runs arbitrary code for a maliciously
crafted file, but every snapshot here is written by this same process (never
accepted from a client or network peer) into a directory the server itself
owns -- the same trust boundary as any other file this process writes to its
own disk. The fingerprint/format-version/TTL checks below guard against
*stale or reshaped* data, not against a hostile file; that's an explicit,
accepted scope limit for this local dev/demo tool, not an oversight.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import pickle
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from engine.board import Board, PortType, Resource, Terrain
from engine.game import CatanGame
from engine.state import DevCard, GameState, Phase, PlayerState, TradeOffer

from .bots import SeatKind, build_agents
from .sessions import _TTL_SECONDS, GameSession

logger = logging.getLogger(__name__)

SESSION_DIR = Path(__file__).resolve().parent.parent / ".catan-sessions"
FORMAT_VERSION = 1

_FINGERPRINTED_DATACLASSES = (GameState, PlayerState, DevCard, TradeOffer, Board)
_FINGERPRINTED_ENUMS = (Phase, Resource, Terrain, PortType)
# DevCardType lives in engine.state but isn't imported above under that name
# to avoid a second import line -- imported directly where used instead.


def shape_fingerprint() -> str:
    """A string that changes if any persisted class's shape changes.

    Covers dataclass field names (a renamed/removed/added field) and enum
    member name->value maps (an ``auto()`` renumbering) -- see the module
    docstring for why the enum axis is the one that would otherwise corrupt
    silently rather than raise.
    """
    from engine.state import DevCardType

    parts: list[str] = []
    for cls in _FINGERPRINTED_DATACLASSES:
        names = tuple(f.name for f in dataclasses.fields(cls))
        parts.append(f"{cls.__name__}:{names}")
    for enum_cls in (*_FINGERPRINTED_ENUMS, DevCardType):
        members = tuple((m.name, m.value) for m in enum_cls)
        parts.append(f"{enum_cls.__name__}:{members}")
    return "|".join(parts)


@dataclass(frozen=True, slots=True)
class _Snapshot:
    """What actually gets pickled for one game -- ``GameSession`` plus the
    metadata needed to validate and rehydrate it.
    """

    format_version: int
    fingerprint: str
    saved_at: float  # wall clock (time.time()), not GameSession's monotonic
    num_players: int
    seat_kinds: list[SeatKind]  # "human", or agent.name for a bot seat
    state: GameState


def _snapshot_path(game_id: str) -> Path:
    return SESSION_DIR / f"{game_id}.pickle"


def _seat_kinds(session: GameSession) -> list[SeatKind]:
    # agent.name for a bot seat is already drawn from BotKind (see
    # server/bots.py's build_agent -- name always equals the kind it was
    # built from), so this cast is safe, not a widening.
    return [
        "human" if agent is None else cast(SeatKind, agent.name)
        for agent in session.agents
    ]


def save_session(game_id: str, session: GameSession) -> None:
    """Persist ``session`` to disk, atomically.

    Call after a mutation has fully completed (including any bot turns it
    triggered) -- never mid-mutation, or a restart could resume into a
    half-applied state.
    """
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = _Snapshot(
        format_version=FORMAT_VERSION,
        fingerprint=shape_fingerprint(),
        saved_at=time.time(),
        num_players=len(session.agents),
        seat_kinds=_seat_kinds(session),
        state=session.state,
    )
    path = _snapshot_path(game_id)
    tmp_path = path.with_suffix(".pickle.tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(snapshot, f)
    os.replace(tmp_path, path)  # atomic on POSIX -- no torn/partial file


def delete_snapshot(game_id: str) -> None:
    _snapshot_path(game_id).unlink(missing_ok=True)


def _load_one(path: Path) -> tuple[str, GameSession] | None:
    game_id = path.stem
    try:
        with path.open("rb") as f:
            snapshot: _Snapshot = pickle.load(f)
    except (
        pickle.UnpicklingError,
        AttributeError,
        ModuleNotFoundError,
        EOFError,
        ValueError,
    ) as exc:
        logger.warning("discarding unreadable session %r: %s", game_id, exc)
        return None

    if snapshot.format_version != FORMAT_VERSION:
        logger.warning(
            "discarding session %r: format_version %r != current %r",
            game_id,
            snapshot.format_version,
            FORMAT_VERSION,
        )
        return None
    if snapshot.fingerprint != shape_fingerprint():
        logger.warning(
            "discarding session %r: engine state shape has changed since it was saved",
            game_id,
        )
        return None
    if time.time() - snapshot.saved_at > _TTL_SECONDS:
        logger.info("discarding session %r: past its TTL while offline", game_id)
        return None

    game = CatanGame(num_players=snapshot.num_players)
    # A freshly drawn driver_seed, same as a new game's -- bot RNG streams
    # are deliberately not persisted (module docstring). build_agents must
    # see the whole seat_kinds list at once (not one seat at a time) so each
    # bot seat's RNG is keyed by its real seat index, not always seat 0.
    driver_seed = secrets.randbits(63)
    agents = build_agents(snapshot.seat_kinds, driver_seed)
    session = GameSession(game=game, state=snapshot.state, agents=agents)
    return game_id, session


def load_all() -> dict[str, GameSession]:
    """Recover every valid, non-expired session from ``SESSION_DIR``.

    Called once, at process startup. A bad, stale-shaped, or TTL-expired
    file is skipped with a logged reason rather than aborting startup --
    one damaged snapshot must never block every other game from resuming.
    """
    if not SESSION_DIR.is_dir():
        return {}
    sessions: dict[str, GameSession] = {}
    for path in sorted(SESSION_DIR.glob("*.pickle")):
        loaded = _load_one(path)
        if loaded is not None:
            game_id, session = loaded
            sessions[game_id] = session
    return sessions
