"""FastAPI HTTP layer wrapping the Catan engine.

Thin, per ``docs/plans/gui-web-frontend.md``: all rules/legality live in
``engine/``, all redaction/serialization in ``server/serialize.py``, all
session lifecycle in ``server/sessions.py``, all bot turn-stepping in
``server/bots.py``. This module only turns HTTP requests into calls against
those -- it never decides whether a move is legal itself.
"""

from __future__ import annotations

import secrets
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from engine.actions import ProposeTrade
from engine.board import Resource
from engine.game import IllegalActionError
from engine.state import acting_player
from server.bots import SeatKind, build_agents, step_bots
from server.serialize import player_view, serialize_geometry, serialize_legal_actions
from server.sessions import (
    GameSession,
    SessionNotFoundError,
    create_session,
    delete_session,
    get_session,
)

app = FastAPI(title="Catan")


class CreateGameRequest(BaseModel):
    num_players: int
    seat_kinds: list[SeatKind]
    seed: int | None = None


class CreateGameResponse(BaseModel):
    game_id: str
    geometry: dict[str, Any]
    state: dict[str, Any]


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
    # freshly drawn.
    driver_seed = secrets.randbits(63)
    agents = build_agents(request.seat_kinds, driver_seed)
    try:
        game_id, session = create_session(
            request.num_players, agents, seed=request.seed
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    step_bots(session)
    return CreateGameResponse(
        game_id=game_id,
        geometry=serialize_geometry(),
        state=player_view(session.state, viewer=None),
    )


@app.get("/api/games/{game_id}/state")
def get_state(game_id: str, viewer: int | None = None) -> dict[str, Any]:
    session = _get_session_or_404(game_id)
    return player_view(session.state, viewer)


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
    return serialize_legal_actions(session.game.legal_actions(state))


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

    if isinstance(action, ProposeTrade):
        action = ProposeTrade(
            give=_resource_bundle_from_wire(request.give),
            receive=_resource_bundle_from_wire(request.receive),
        )

    try:
        game.apply_action(state, action)
    except IllegalActionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    step_bots(session)
    return player_view(state, viewer=actor)


@app.delete("/api/games/{game_id}")
def delete_game(game_id: str) -> dict[str, bool]:
    delete_session(game_id)
    return {"deleted": True}
