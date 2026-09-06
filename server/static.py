"""Serves the built Svelte frontend (``web/dist/``, from ``npm run build``)
as static files, so one ``uv run uvicorn server.app:app`` serves both the
API and the UI -- no separate frontend dev server needed to actually play.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"

logger = logging.getLogger(__name__)


def mount_static(app: FastAPI) -> None:
    """Mount ``web/dist/`` at ``/``, if it has been built.

    Must be called after every ``/api/...`` route is already registered --
    Starlette matches routes in registration order, so an earlier call
    would let this catch-all mount shadow the API routes. A no-op (with a
    log warning) when ``web/dist/`` doesn't exist yet, so a fresh clone (or
    a test importing this module before ``npm run build`` has ever run)
    still starts an API-only server instead of crashing on import.
    """
    if not WEB_DIST.is_dir():
        logger.warning(
            "%s does not exist -- run `npm run build` in web/ to serve the "
            "frontend; the API alone is still available under /api",
            WEB_DIST,
        )
        return
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="static")
