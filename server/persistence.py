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

**Adding session-level fields later.** ``_Snapshot`` is a slots dataclass, so
new fields must be *appended at the end* and given defaults: pickle restores
slots state positionally, and an old snapshot then simply leaves the trailing
new fields unset (measured on py3.14) instead of failing. Read them with
``getattr(snapshot, name, default)`` -- never bump ``FORMAT_VERSION`` for an
additive field, that would discard live games. ``dice_rolls`` (a
``server/sessions.py`` type, not engine state) is such a field, so it is
deliberately not in the shape fingerprint; a snapshot that predates it loads
with an empty history flagged ``dice_history_complete=False``.

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
import json
import logging
import os
import pickle
import secrets
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from engine.board import Board, PortType, Resource, Terrain
from engine.game import CatanGame, true_victory_points, victory_points
from engine.state import DevCard, GameState, Phase, PlayerState, TradeOffer

from .bots import (
    EXPERIMENT_ENV,
    TRADE_POLICY,
    RLSeatUnavailableError,
    SeatKind,
    build_agents,
    rl_checkpoint_id,
)
from .serialize import dice_counts
from .sessions import _TTL_SECONDS, DiceRoll, GameSession

logger = logging.getLogger(__name__)

SESSION_DIR = Path(__file__).resolve().parent.parent / ".catan-sessions"
# One JSON record per human-involved game (see ``write_game_record``);
# gitignored raw data, tabulated by ``experiments/human_games.py``.
GAMES_DIR = Path(__file__).resolve().parent.parent / ".catan-games"
FORMAT_VERSION = 2

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
    turn_count: int
    rl_checkpoint: str | None  # rebuilt with this, not the current env var
    # Appended last, with defaults -- see the module docstring.
    dice_rolls: tuple[DiceRoll, ...] = ()
    dice_history_complete: bool = True


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
        turn_count=session.turn_count,
        rl_checkpoint=session.rl_checkpoint,
        dice_rolls=tuple(session.dice_rolls),
        dice_history_complete=session.dice_history_complete,
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
    try:
        agents = build_agents(
            snapshot.seat_kinds,
            driver_seed,
            Path(snapshot.rl_checkpoint) if snapshot.rl_checkpoint else None,
        )
    except RLSeatUnavailableError as exc:
        logger.warning("skipping session %r: rl seat unavailable: %s", game_id, exc)
        return None
    # A snapshot saved before dice tracking has no such attribute at all.
    # Its rolls are unrecoverable, so its history is complete only if no
    # turn had been played yet.
    saved_rolls = getattr(snapshot, "dice_rolls", None)
    if saved_rolls is None:
        rolls: list[DiceRoll] = []
        history_complete = snapshot.turn_count == 0
        if not history_complete:
            logger.info(
                "session %r predates dice tracking: counting from turn %d",
                game_id,
                snapshot.turn_count,
            )
    else:
        rolls = list(saved_rolls)
        history_complete = getattr(snapshot, "dice_history_complete", True)
    session = GameSession(
        game=game,
        state=snapshot.state,
        agents=agents,
        turn_count=snapshot.turn_count,
        rl_checkpoint=snapshot.rl_checkpoint,
        dice_rolls=rolls,
        dice_history_complete=history_complete,
    )
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


def _record_path(game_id: str) -> Path:
    return GAMES_DIR / f"{game_id}.json"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return out.stdout.strip()
    except OSError, subprocess.CalledProcessError:
        return "unknown"


def write_game_record(
    game_id: str, session: GameSession, meta: dict[str, Any] | None = None
) -> None:
    """Write/refresh the on-disk record for a human-involved game.

    ``status`` is ``"started"`` until the game is terminal, then
    ``"finished"`` with the outcome. ``meta`` (``engine_seed``,
    ``driver_seed``, given at creation) is merged into any existing record,
    so later calls -- including after a restart -- keep it. Games with no
    human seat are not recorded. A game deleted before it finishes keeps its
    ``"started"`` record on purpose: that is what "abandoned" means to the
    tabulator.
    """
    if all(agent is not None for agent in session.agents):
        return
    path = _record_path(game_id)
    record: dict[str, Any] = {}
    if path.is_file():
        record = json.loads(path.read_text())
    if meta:
        record.update(meta)
    state = session.state
    kinds = _seat_kinds(session)
    now = datetime.now(UTC).isoformat()
    record.setdefault("game_id", game_id)
    record.setdefault("created_at", now)
    record.setdefault("git_commit", _git_commit())
    record.setdefault("experiment", os.environ.get(EXPERIMENT_ENV))
    record.update(
        num_players=len(kinds),
        seat_kinds=list(kinds),
        human_seats=[i for i, k in enumerate(kinds) if k == "human"],
        trade_policy=TRADE_POLICY,
        rl_checkpoint=(
            rl_checkpoint_id(session.rl_checkpoint) if session.rl_checkpoint else None
        ),
        turn_count=session.turn_count,
    )
    finished = session.game.is_terminal(state)
    record["status"] = "finished" if finished else "started"
    if finished:
        record.setdefault("finished_at", now)
        record["winner"] = state.winner
        record["winner_kind"] = None if state.winner is None else kinds[state.winner]
        record["vp_true"] = [true_victory_points(state, i) for i in range(len(kinds))]
        record["vp_public"] = [victory_points(state, i) for i in range(len(kinds))]
        record["dice_history_complete"] = session.dice_history_complete
        record["dice_counts"] = dice_counts(session.dice_rolls)
        record["dice_rolls"] = [
            [r.turn, r.player_id, r.d1, r.d2] for r in session.dice_rolls
        ]
    GAMES_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(record, indent=2))
    os.replace(tmp_path, path)
