# GUI — Web frontend for the Catan engine

## Context

Phases 1–3 (`9ce94c7`, `c51e4bd`, `aa58a64`) delivered a rules engine, a
Monte Carlo analysis layer, and agents — but the only way to *play* a game is
`cli.py`'s text menu: hexes are `[L8 ]`-style bracketed cells, buildings are
flat `vertex 23: settlement (A)` listings, and a human picks actions by
choosing numeric indices into `legal_actions()` (`agents/human.py`). That is
enough to exercise the engine, not enough to actually enjoy a game or show it
to anyone else.

This plan adds a browser GUI: a Svelte + TypeScript frontend that renders
the real hex board and lets a human click to build/move-robber/trade, talking
to a small FastAPI backend that wraps `engine/game.py`. It does **not**
touch game rules — `CatanGame`/`GameState`/`engine/actions.py` stay exactly
as they are; the backend is a thin translation layer (session storage +
JSON serialization + redaction), matching the spirit of decision 8 ("this is
our own simulator") applied to the boundary too: the browser is a *client* of
the engine, never a second implementation of it.

`~/projects/fitted` was looked at as a possible source of reusable patterns
(a sibling project with an actual production web stack) and explicitly not
reused: its backend (Postgres/SQLAlchemy/Celery/Alembic/auth/Docker) doesn't
apply to a local in-memory research tool with no persistence and no users to
authenticate, and its frontend (React/TanStack/Tailwind/PWA) is a full
production SPA stack sized for a much bigger app than one board plus a side
panel. Noted here so a future reader doesn't wonder why an obvious sibling's
stack wasn't adopted — it was considered and rejected on fit, not overlooked.

### Roadmap placement (decided)

This lands as a **non-numbered parallel plan**, not a renumbered phase.
Phases 4–6 (shared ML extraction, RL, retrofit) are the ML/research track;
the GUI is a tooling/UX track that can proceed independently and doesn't
block or get blocked by them. README gets a new "Tooling" line alongside the
numbered roadmap, plus new Decisions entries (numbered continuing from 19).

### Scope boundaries (decided)

- **No new game logic.** Legality, resolution, and RNG stay in `engine/`.
  The backend never decides whether a move is legal — it calls
  `legal_actions`/`apply_action` and surfaces whatever they say, exactly like
  `cli.py` does today.
- **Svelte + Vite, not vanilla TypeScript or React.** The app's real
  complexity is reactive UI state — highlighting legal vertices/edges on the
  SVG board, swapping the side panel per game phase, opening/closing the
  trade form — all re-rendering on every poll. Plain vanilla TS was rejected:
  that's hand-rolled DOM diffing for state that changes on every tick, for no
  reason once a reactive layer is on the table. React was also rejected: its
  runtime and boilerplate are sized for a big SPA with many routes/screens,
  and its main value-add — a large ecosystem for that scale of app — isn't
  needed for one board plus a side panel. Svelte compiles to near-vanilla JS
  with a small runtime and low boilerplate, which fits this single-screen
  reactive app well without pulling in a heavy SPA stack. `fastapi`/`uvicorn`
  are this project's first-ever runtime dependencies (`pyproject.toml`'s
  `dependencies = []` today); the frontend gets its own `package.json` under
  `web/`, scaffolded via `npm create vite@latest -- --template svelte-ts`,
  kept out of the Python dependency graph entirely.
- **Multiple concurrent in-memory games, no disk persistence.** A
  `dict[game_id, GameSession]` in process memory. Restarting the server
  drops all games — acceptable for a local research/demo tool; revisit only
  if someone actually asks to resume a game across restarts (pickling a live
  `random.Random` is possible but couples the save format to `state.py`'s
  field layout, a real cost not worth paying speculatively).
- **Redact hidden information at the API boundary, not in the engine.**
  `GameState` deliberately has no `player_view()` (README decision 4); that
  stays true. Redaction is new code in the server layer only.
- **HumanAgent is not reused for the web driver.** It blocks on `input()`,
  which cannot work inside a request/response HTTP handler. The server gets
  its own request-driven loop; only the *type of thing a human is allowed to
  submit* (an element of `legal_actions()`, or a resolved `ProposeTrade`
  bundle) carries over from `agents/human.py`'s contract.
- **Agents (`RandomAgent`/`StratifiedRandomAgent`/`HeuristicAgent`) are
  reused as-is** for bot seats, via the same `AgentFactory` shape
  `experiments/rollout.py` already established — not a new factory
  abstraction.

---

## Architecture

```
web/                          # new: Svelte + Vite + TypeScript frontend
  src/
    lib/
      api.ts                 # typed fetch wrappers over the HTTP API
      geometry.ts             # axial/cube -> pixel layout math (pure functions)
      Board.svelte             # SVG rendering + click/highlight of hexes,
                                #   vertices, edges, robber, ports, buildings
      ActionPanel.svelte       # non-spatial legal actions as buttons, grouped
                                #   by kind (mirrors agents/human.py's menu)
      TradeForm.svelte          # ProposeTrade give/receive bundle form
      GameSetup.svelte          # new-game / seat-configuration screen
    App.svelte                 # top-level layout, polling loop, phase-driven
                                #   view switching (setup vs. in-game vs. game-over)
    main.ts                    # Svelte app bootstrap
  index.html
  package.json                 # svelte, vite, @sveltejs/vite-plugin-svelte, typescript
  svelte.config.js
  tsconfig.json
  vite.config.ts

server/                       # new: FastAPI backend wrapping engine/
  __init__.py
  app.py                      # FastAPI app, route handlers
  sessions.py                 # GameSession, in-memory session store, TTL eviction
  serialize.py                # GameState/Board/Action -> JSON-safe dict, redaction
  bots.py                     # auto-step loop for non-human seats (reuses AgentFactory)
  static.py                   # mounts web/dist/ as StaticFiles

engine/                       # unchanged
agents/                       # unchanged (consumed by server/bots.py)
```

### Why FastAPI

Type-hinted request/response bodies (matches this repo's `mypy --strict`-ish
config), automatic OpenAPI docs for free (useful once the action-selection
JSON shape below stabilizes), and `TestClient` gives the verification story
in-process without a live server — the same reason `pytest` already drives
the engine directly rather than through subprocesses.

### State serialization strategy

`GameState.rng: random.Random` is a live object — the reason there is no
naive `dataclasses.asdict`/JSON round-trip, and the reason the server, not
the browser, is the single authoritative holder of game state (README
decision 3's RNG-cloning rule already assumes exactly one owner). The
browser never receives a `GameState`; it receives a **view** produced by
`server/serialize.py`:

```python
def player_view(state: GameState, viewer: int | None) -> dict[str, Any]:
    """JSON-safe projection of GameState for one viewing seat.

    viewer=None (spectator/debug) sees every hand; a seat index sees its own
    hand and dev cards, and resource_count/dev_card_count only for others.
    """
```

This is exactly the redaction helper README decision 4 says the engine
deliberately doesn't have — it belongs here, one level up, same as that
decision anticipated. Board geometry (`GEOMETRY.land_hexes`,
`vertex_hexes`, `edge_vertices`, `port_locations`, etc.) is **RNG-free and
identical across every game** (`engine/board.py`'s module-level singleton),
so it is serialized **once**, on `POST /api/games`, and the frontend caches
it — per-turn state responses carry only per-game content (terrain/tokens/
robber/buildings/hands/phase), not the fixed topology.

### Legal-action protocol

`legal_actions(state)` returns frozen dataclass instances
(`engine/actions.py`) — TypeScript never constructs one. The server lists
them as an indexed, render-hinted array:

```json
[
  {"index": 0, "kind": "PlaceSettlement", "vertex_id": 23},
  {"index": 1, "kind": "PlaceRoad", "edge_id": 17},
  {"index": 7, "kind": "EndTurn"}
]
```

and the client posts back `{"index": 0}`. The one exception is
`ProposeTrade`: `legal_actions` offers a single open-ended sentinel
(README decisions 5/10/18) — `{"index": N, "kind": "ProposeTrade",
"open_ended": true}` — and the client's follow-up carries a constructed
bundle instead of just the index: `{"index": N, "give": {"LUMBER": 2},
"receive": {"ORE": 1}}`. The server validates the same way
`apply_action`/`agents/human.py`'s `_prompt_propose_trade` already do — no
new validation logic, just a body shape.

`acting_player(state)` (`engine/state.py`), not `state.current_player`, is
what the server polls to decide who acts next — the two diverge during
`Phase.AWAIT_TRADE_RESPONSE`, and getting this wrong would drive the wrong
seat's turn mid-trade, same trap `cli.py:111` already avoids.

### Board rendering

SVG, not canvas: `<polygon>` hexes and `<circle>`/small hit-target
`<rect>`s at vertices/edges give free hit-testing (`on:click`/`on:pointerenter`)
for highlighting legal build sites, which is exactly what action selection
needs. Axial→pixel conversion is pure layout math done once in
`lib/geometry.ts` from the `Cube = (q, r, s)` coordinates the server already
sends verbatim (`engine/board.py`'s `Cube` type) — no adjacency or legality
logic is reimplemented in the frontend; the server is the only place that
computes what's legal. `Board.svelte` renders that geometry declaratively
(an `{#each}` block per hex/vertex/edge keyed by id) and reacts to legal-
action/highlight state changing on every poll — the exact case Svelte's
reactivity is a good fit for, versus hand-rolling that diff in vanilla JS.

### Human interaction model

Hybrid, matching how the pieces naturally split:

- **Spatial actions** (`PlaceSettlement`, `PlaceRoad`, `PlaceCity`,
  `MoveRobber`, `StealFrom`) highlight the corresponding vertices/edges/
  hexes/player-tokens on the SVG board; clicking one posts that action's
  index.
- **Non-spatial actions** (`RollDice`, `BuyDevCard`, `PlayKnight`,
  `PlayRoadBuilding`, `PlayYearOfPlenty`, `PlayMonopoly`, `PlayVictoryPoint`,
  `TradeBank`, `TradePort`, `AcceptTrade`, `RejectTrade`, `EndTurn`) are
  buttons in `ActionPanel.svelte`, grouped like `agents/human.py`'s menu and
  swapped automatically as `state.phase` changes (a reactive `$:` binding on
  the phase, rather than manually tearing down/rebuilding a button list).
- **`ProposeTrade`** opens `TradeForm.svelte`, a small resource-bundle form
  (mirrors `_prompt_resource_bundle`'s give/receive-with-caps logic, just as
  HTML inputs instead of `input()` prompts) with its open/closed state as
  local component state.

### Bot turns

When `acting_player(state)` is a bot seat, the server does not wait for an
HTTP request to advance — `server/bots.py` runs a step loop (constructed via
the existing `AgentFactory` shape from `experiments/rollout.py`, not a new
abstraction) advancing through consecutive bot turns and stopping the moment
`acting_player` becomes human, or the game ends. The frontend polls
`GET /api/games/{id}/state` (short interval, e.g. 1s) so the board updates
as bots play, rather than needing a websocket for a first cut.

### Sessions

```python
@dataclass(slots=True)
class GameSession:
    game: CatanGame
    state: GameState
    seat_kind: list[Literal["human", "bot"]]   # per player_id
    agents: list[Agent | None]                 # bot agents, None for human seats
    last_touched: float                        # monotonic clock, for TTL eviction
```

`_SESSIONS: dict[str, GameSession]` at module scope in `server/sessions.py`,
keyed by a random `game_id` (`secrets.token_urlsafe`). A background sweep
(FastAPI startup task or a simple check-on-access) evicts sessions idle past
a TTL (e.g. 2 hours) — this is a local dev tool, not a production service,
so an in-memory cap plus TTL eviction is enough; no auth, no rate limiting.

---

## API design

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/games` | Create a game: `{num_players, seat_kinds: ["human","bot","bot"], seed?}` → `{game_id, geometry, state}` (geometry sent once) |
| `GET` | `/api/games/{id}/state?viewer={seat}` | Redacted state for one viewing seat (or `viewer` omitted → spectator/debug view, gated by a server flag — see open decisions) |
| `GET` | `/api/games/{id}/legal_actions?viewer={seat}` | Indexed, render-hinted legal actions for the acting seat; empty if it's not that seat's turn or a bot is acting |
| `POST` | `/api/games/{id}/action` | `{index, give?, receive?}` → applies via `apply_action`, then runs the bot step loop, returns the new state |
| `DELETE` | `/api/games/{id}` | Explicit cleanup (in addition to TTL eviction) |

All error paths surface `IllegalActionError` messages as HTTP 400 with the
exception text — same information `cli.py:124`'s `print(f"Illegal action:
{exc}")` already shows, just returned instead of printed.

---

## Files

### New: `server/serialize.py`
`player_view(state, viewer)`, `serialize_geometry()` (once, RNG-free),
`serialize_legal_actions(actions, board_geometry)` (adds render hints:
vertex/edge/hex ids already present on the action dataclasses need no
lookup; `ProposeTrade` gets `open_ended: true`). Pure functions, no FastAPI
imports — testable directly against `engine/` types.

### New: `server/sessions.py`
`GameSession` dataclass, `create_session(num_players, seat_kinds, seed) ->
(game_id, GameSession)`, `get_session(game_id) -> GameSession`, TTL sweep.

### New: `server/bots.py`
`step_bots(session: GameSession) -> None` — while `acting_player(session.state)`
is a bot seat and the game isn't terminal, call that seat's
`agent.choose_action` then `apply_action`. Reuses `agents.Agent` directly;
seat agents are constructed via a small `AgentFactory`-shaped function
mapping `seat_kinds` (`"random"`/`"stratified_random"`/`"heuristic"`) to
`agents/` classes, mirroring `experiments/benchmark.py`'s
`benchmark_agent_factory` pattern rather than inventing a new one.

### New: `server/app.py`
Route handlers per the API table above; thin — delegates to
`sessions.py`/`serialize.py`/`bots.py`, mounts `web/dist/` via
`fastapi.staticfiles.StaticFiles`.

### New: `web/` (Svelte + Vite + TypeScript)
Scaffolded via `npm create vite@latest -- --template svelte-ts`, then
trimmed to the component split in the Architecture section
(`Board.svelte`, `ActionPanel.svelte`, `TradeForm.svelte`, `GameSetup.svelte`,
`App.svelte`, `lib/api.ts`, `lib/geometry.ts`). `tsconfig.json`: `strict:
true`. `vite.config.ts`: `@sveltejs/vite-plugin-svelte`, `build.outDir:
"dist"`. Dev workflow: `npm run dev` for hot-reload against a running
`uvicorn` backend (Vite dev server proxies `/api` to it); `npm run build`
produces `web/dist/` for `server/static.py` to serve in production/demo mode.

### Modified: `pyproject.toml`
Add `fastapi` and `uvicorn` (or `uvicorn[standard]`) to `dependencies`
(first non-empty runtime dependency list); add a `catan-server` script
entry point alongside the existing `catan` CLI one.

### Modified: `README.md`
New "Tooling" line in the roadmap (this plan, marked complete once shipped);
Decisions entries 20+ recording: the redaction-at-the-boundary split, the
index-based legal-action protocol and its `ProposeTrade` exception, the
in-memory-only session scope, and the "no game logic in TypeScript" rule.

### New: `tests/test_server.py`
Drives the FastAPI app via `TestClient` — see Verification.

---

## Phased implementation steps

1. `server/serialize.py` + its tests — pure functions, no FastAPI
   dependency, easiest to get right and verify in isolation (round-trip a
   few real `GameState`s from `CatanGame.reset(seed=N)` and assert the
   redaction actually hides other seats' hands).
2. `server/sessions.py` + `server/bots.py` — session lifecycle and the bot
   step loop, tested directly (no HTTP yet) by driving `GameSession`
   objects the same way `experiments/rollout.py`'s tests drive `run_game`.
3. `server/app.py` — wire the routes; add `fastapi`/`uvicorn` to
   `pyproject.toml`; `tests/test_server.py` end-to-end via `TestClient`.
4. `web/` skeleton — scaffold via `npm create vite@latest -- --template
   svelte-ts`, then `lib/api.ts` typed fetch wrappers matching the API table
   exactly, `npm run dev` serving a blank page against a running backend.
5. `Board.svelte` — static SVG render of geometry + current state (no
   interactivity yet): hexes, tokens, robber, ports, buildings/roads by
   owner color, using `lib/geometry.ts` for axial→pixel layout. Verify
   visually against `cli.py`'s text output for the same seed.
6. Interactivity — legal-action highlighting and click-to-post on
   `Board.svelte`, `ActionPanel.svelte` for non-spatial actions,
   `TradeForm.svelte` for `ProposeTrade`.
7. `App.svelte` — polling loop, phase-driven view switching, bot-turn
   indication, game-over screen.
8. Polish pass: error display for rejected actions (mirrors `cli.py:124`),
   `GameSetup.svelte` for new-game / seat configuration; `npm run build` ->
   `web/dist/` wired into `server/static.py`.

---

## Open design decisions (resolved during planning)

- **Session scope:** multiple concurrent in-memory games, no persistence.
- **Redaction:** always redact per viewing seat; a spectator/debug view
  (full ground truth) is available but must be an explicit opt-in
  (`viewer` query param omitted + a server-side flag), never the default —
  avoids a DevTools tab silently seeing every hand.
- **Human interaction model:** hybrid board-clicks + action panel.
- **Roadmap placement:** non-numbered parallel plan (this document).
- **Frontend framework:** Svelte + Vite, not vanilla TS or React (see
  Scope boundaries above for the rejection reasons).
- **`~/projects/fitted`'s stack:** considered, not reused — mismatched on
  both ends (DB/auth/Celery backend for a stateless local tool; a
  production SPA stack for a one-screen app). See Context.

---

## Verification

1. `uv run pytest tests/test_server.py -v` — a full game driven purely
   through `TestClient` HTTP calls (`POST /api/games` → repeated
   `POST /api/games/{id}/action` using a `StratifiedRandomAgent`-equivalent
   client-side random choice among returned legal actions) reaches
   `is_terminal` and reports a winner.
2. **Bit-for-bit cross-check against the engine directly:** run the same
   `(engine_seed, driver_seed)` pair through `experiments/rollout.run_game`
   and through the HTTP path with an equivalent all-bot lineup; assert the
   same winner and the same final `victory_points` per seat — proves the
   HTTP layer didn't fork the rules, the same spirit as `test_agents.py`'s
   bit-identity regression test.
3. Redaction test: fetch `GET /api/games/{id}/state?viewer=0` and assert
   seat 1's `resources`/`dev_hand` keys are absent, only counts present.
4. `uv run ruff check . && uv run ruff format --check .` clean on
   `server/`; `web/`'s `npm run check` (`svelte-check` + `tsc --noEmit`)
   clean.
5. Manual smoke test: `uv run uvicorn server.app:app --reload`, open the
   page, play a full 3-player game (2 bots) end to end in the browser —
   place initial settlements/roads, roll, build, trigger a robber move and
   a steal, reach a win. This is the check that proves the feature works,
   not just that the API contract holds (per this repo's own testing
   philosophy: source/type-verified is not the same as verified).
