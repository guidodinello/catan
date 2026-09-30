"""FastAPI HTTP layer wrapping the Catan engine.

Thin, per ``docs/plans/gui-web-frontend.md``: all rules/legality live in
``engine/``, all redaction/serialization in ``server/serialize.py``, all
session lifecycle in ``server/sessions.py``, all bot turn-stepping in
``server/bots.py``. This module only turns HTTP requests into calls against
those -- it never decides whether a move is legal itself.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from engine.actions import CounterTrade, PlayVictoryPoint, ProposeTrade
from engine.board import Resource
from engine.game import IllegalActionError
from engine.state import acting_player
from server.bots import (
    EXPERIMENT_ENV,
    RLSeatUnavailableError,
    SeatKind,
    apply_to_session,
    build_agents,
    rl_availability,
    rl_checkpoint_id,
    rl_checkpoint_path,
    step_bots,
)
from server.persistence import (
    delete_snapshot,
    load_all,
    save_session,
    write_game_record,
)
from server.serialize import (
    player_view,
    serialize_build_costs,
    serialize_dice_history,
    serialize_geometry,
    serialize_legal_actions,
    serialize_trail,
)
from server.sessions import (
    GameSession,
    SessionNotFoundError,
    create_session,
    delete_session,
    get_session,
    restore_sessions,
)
from server.static import mount_static


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Recover any games left over from before the last restart (see
    # server/persistence.py) -- eager, one-time scan of a handful of small
    # files, so get_session's hot path never has to touch disk. No shutdown
    # half needed: every mutation already writes through (server/app.py's
    # own save_session calls below), so there's nothing left to flush here.
    restore_sessions(load_all())
    yield


app = FastAPI(title="Catan", lifespan=_lifespan)


class CreateGameRequest(BaseModel):
    num_players: int
    seat_kinds: list[SeatKind]
    seed: int | None = None


class CreateGameResponse(BaseModel):
    game_id: str
    driver_seed: int
    engine_seed: int
    geometry: dict[str, Any]
    state: dict[str, Any]
    action_trail: list[dict[str, Any]]


class ActionRequest(BaseModel):
    index: int
    give: dict[str, int] | None = None
    receive: dict[str, int] | None = None


def _get_session_or_404(game_id: str) -> GameSession:
    try:
        return get_session(game_id)
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"no such game {game_id!r}"
        ) from exc


def _session_view(session: GameSession, viewer: int | None) -> dict[str, Any]:
    """``player_view`` plus the session's dice-roll history, which lives on
    the session rather than the engine state.
    """
    return {
        **player_view(session.state, viewer),
        "dice_history": serialize_dice_history(
            session.dice_rolls, session.dice_history_complete
        ),
    }


def _resource_bundle_from_wire(bundle: dict[str, int] | None) -> dict[Resource, int]:
    if not bundle:
        return {}
    try:
        return {Resource[name]: count for name, count in bundle.items()}
    except KeyError as exc:
        raise HTTPException(
            status_code=400, detail=f"unknown resource {exc.args[0]!r}"
        ) from exc


@app.post("/api/games", response_model=CreateGameResponse)
def create_game(request: CreateGameRequest) -> CreateGameResponse:
    if len(request.seat_kinds) != request.num_players:
        raise HTTPException(
            status_code=400,
            detail=(
                f"expected {request.num_players} seat kinds, "
                f"got {len(request.seat_kinds)}"
            ),
        )
    # The engine seed (`request.seed`) and the bot driver seed are
    # independent streams (README decision 17) -- reproducibility of a live
    # web game's engine rolls is the only thing the API exposes; bot choices
    # are not required to be reproducible, so the driver seed is always
    # freshly drawn (never accepted from the request). It's still returned
    # in the response -- purely observable, not settable -- so a client (or
    # a test, see docs/backlog.md's bit-identity cross-check) can replay this
    # exact game's bot choices through experiments.rollout.run_game. Note a
    # *resumed* game (server/persistence.py) draws a fresh driver_seed on
    # restore -- bot RNG is deliberately not persisted -- so that replay only
    # holds for the lifetime of one process.
    driver_seed = secrets.randbits(63)
    # game.reset(seed=None) draws an unrecoverable seed, and the game record
    # needs one -- so draw it here.
    engine_seed = request.seed if request.seed is not None else secrets.randbits(63)
    rl_path = rl_checkpoint_path() if "rl" in request.seat_kinds else None
    try:
        agents = build_agents(request.seat_kinds, driver_seed, rl_path)
    except RLSeatUnavailableError as exc:
        raise HTTPException(
            status_code=400, detail=f"rl seat unavailable: {exc}"
        ) from exc
    try:
        game_id, session = create_session(
            request.num_players,
            agents,
            seed=engine_seed,
            rl_checkpoint=str(rl_path) if rl_path else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    initial_trail = step_bots(session)
    # After step_bots, not before -- a save made inside create_session would
    # snapshot a state already stale by whatever bot turns just ran.
    save_session(game_id, session)
    write_game_record(
        game_id,
        session,
        {"engine_seed": engine_seed, "driver_seed": driver_seed},
    )
    return CreateGameResponse(
        game_id=game_id,
        driver_seed=driver_seed,
        engine_seed=engine_seed,
        geometry=serialize_geometry(),
        state=_session_view(session, viewer=None),
        action_trail=serialize_trail(initial_trail),
    )


@app.get("/api/games/{game_id}/state")
def get_state(game_id: str, viewer: int | None = None) -> dict[str, Any]:
    session = _get_session_or_404(game_id)
    return _session_view(session, viewer)


# VP cards count toward their holder's total automatically and stay hidden
# until game over, so "playing" one only leaks information. The engine and
# bots keep the action (decision 10 / #34); it is just never offered to a
# human. Indices stay the engine's, so the served list may have gaps.
HUMAN_HIDDEN_ACTION_KINDS = frozenset({"PlayVictoryPoint"})


@app.get("/api/games/{game_id}/legal_actions")
def get_legal_actions(game_id: str, viewer: int | None = None) -> list[dict[str, Any]]:
    session = _get_session_or_404(game_id)
    state = session.state
    if session.game.is_terminal(state):
        return []
    actor = acting_player(state)
    if viewer is not None and actor != viewer:
        return []
    if session.agents[actor] is not None:  # a bot seat -- nothing for a human to pick
        return []
    return [
        a
        for a in serialize_legal_actions(session.game.legal_actions(state))
        if a["kind"] not in HUMAN_HIDDEN_ACTION_KINDS
    ]


@app.post("/api/games/{game_id}/action")
def post_action(game_id: str, request: ActionRequest) -> dict[str, Any]:
    session = _get_session_or_404(game_id)
    game = session.game
    state = session.state
    if game.is_terminal(state):
        raise HTTPException(status_code=400, detail="game is already over")

    actor = acting_player(state)
    if session.agents[actor] is not None:
        # Unreachable in normal operation: step_bots always runs to a human
        # seat (or game over) before control returns to the client. Kept as
        # a defensive guard, not a documented API behavior.
        raise HTTPException(status_code=400, detail="it is a bot seat's turn")

    legal = game.legal_actions(state)
    if not 0 <= request.index < len(legal):
        raise HTTPException(
            status_code=400, detail=f"action index {request.index} out of range"
        )
    action = legal[request.index]
    if isinstance(action, PlayVictoryPoint):
        raise HTTPException(
            status_code=400,
            detail=(
                "PlayVictoryPoint is not offered to human players: VP cards "
                "count automatically and are revealed at game over"
            ),
        )

    if isinstance(action, ProposeTrade | CounterTrade):
        # Reconstruct the same class the sentinel was -- a counter must
        # never be silently downgraded to a fresh propose (or vice versa).
        action = type(action)(
            give=_resource_bundle_from_wire(request.give),
            receive=_resource_bundle_from_wire(request.receive),
        )

    # The human's own action belongs in the trail too, not just the bot
    # actions that follow it -- otherwise a human's own dice rolls (and the
    # resulting production) could never appear in a client-side "recent
    # rolls" view built from this trail, only bots'. Captured after the
    # ProposeTrade/CounterTrade sentinel was already resolved above, so this
    # reflects the real give/receive bundle. apply_and_record is the same
    # helper step_bots uses for bot actions, so dice_roll/production are
    # captured identically either way.
    try:
        human_entry = apply_to_session(session, actor, action)
    except IllegalActionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    trail = [human_entry, *step_bots(session)]
    save_session(game_id, session)
    if game.is_terminal(state):
        write_game_record(game_id, session)
    response = _session_view(session, viewer=actor)
    response["action_trail"] = serialize_trail(trail)
    return response


SEAT_KIND_LABELS: dict[str, str] = {
    "human": "Human (this browser)",
    "random": "Bot: Random",
    "stratified_random": "Bot: Stratified Random",
    "heuristic": "Bot: Heuristic",
    "rl": "Bot: RL",
}


@app.get("/api/seat_kinds")
def get_seat_kinds() -> dict[str, Any]:
    """Which seat kinds this server can build right now -- the ``rl`` kind
    depends on torch and a local (gitignored) checkpoint. Never imports
    torch.
    """
    rl_ok, rl_reason = rl_availability()
    kinds = [
        {
            "kind": kind,
            "label": label,
            "available": kind != "rl" or rl_ok,
            "reason": rl_reason if kind == "rl" and not rl_ok else None,
        }
        for kind, label in SEAT_KIND_LABELS.items()
    ]
    return {
        "kinds": kinds,
        "rl_checkpoint": (
            {
                k: v
                for k, v in rl_checkpoint_id(str(rl_checkpoint_path())).items()
                if k != "path"
            }
            if rl_ok
            else None
        ),
        "experiment": os.environ.get(EXPERIMENT_ENV),
    }


@app.get("/api/build_costs")
def get_build_costs() -> dict[str, dict[str, int]]:
    return serialize_build_costs()


@app.delete("/api/games/{game_id}")
def delete_game(game_id: str) -> dict[str, bool]:
    delete_session(game_id)
    delete_snapshot(game_id)  # else a restart would resurrect a deleted game
    return {"deleted": True}


# Must come after every /api/... route above: Starlette matches routes in
# registration order, so mounting the frontend's catch-all StaticFiles
# first would shadow the API routes instead of falling through to them.
mount_static(app)
