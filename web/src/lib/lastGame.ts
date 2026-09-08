// Persists what's needed to resume a game after a hard page reload --
// server/app.py has no "rejoin" endpoint, so this is entirely a frontend
// mechanism: the last-used game_id/viewer, plus a cached copy of the board
// geometry. Caching geometry (rather than re-fetching it) is safe because
// server/serialize.py's serialize_geometry() is RNG-free and identical for
// every game (engine/board.py's fixed GEOMETRY singleton) -- a copy from
// any earlier game is valid for resuming any other game_id too. If no
// geometry has ever been cached (e.g. a fresh browser that's never created
// a game), resuming isn't possible without a server round-trip this app
// doesn't have; callers should treat loadCachedGeometry() returning null as
// "can't resume yet," not retry or guess.

import type { Geometry } from "./api";

const KEY_GAME_ID = "catan:lastGameId";
const KEY_VIEWER = "catan:lastViewer";
const KEY_GEOMETRY = "catan:cachedGeometry";

export interface LastGame {
  gameId: string;
  viewer: number | undefined;
}

export function loadLastGame(): LastGame {
  try {
    const gameId = localStorage.getItem(KEY_GAME_ID) ?? "";
    const viewerRaw = localStorage.getItem(KEY_VIEWER);
    const viewer = viewerRaw === null || viewerRaw === "" ? undefined : Number(viewerRaw);
    return { gameId, viewer };
  } catch {
    return { gameId: "", viewer: undefined };
  }
}

export function saveLastGame(gameId: string, viewer: number | undefined): void {
  try {
    localStorage.setItem(KEY_GAME_ID, gameId);
    localStorage.setItem(KEY_VIEWER, viewer === undefined ? "" : String(viewer));
  } catch {
    // best-effort only -- e.g. private browsing may reject writes
  }
}

export function loadCachedGeometry(): Geometry | null {
  try {
    const raw = localStorage.getItem(KEY_GEOMETRY);
    return raw === null ? null : (JSON.parse(raw) as Geometry);
  } catch {
    return null;
  }
}

export function saveCachedGeometry(geometry: Geometry): void {
  try {
    localStorage.setItem(KEY_GEOMETRY, JSON.stringify(geometry));
  } catch {
    // best-effort only
  }
}
