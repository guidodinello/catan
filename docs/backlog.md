# Backlog — future improvement ideas

Not committed to, not scheduled — a running list of ideas raised while using
the project, kept somewhere more durable than chat history. Contrast with
`docs/plans/`, which holds the implementation plan for work already approved
and built.

## GUI / UX

- ~~**Terrain and resource iconography.**~~ Done — hand-authored SVG icons
  per terrain/resource, in `web/src/lib/icons/`, looked up via
  `TERRAIN_ICON`/`RESOURCE_ICON` maps (`web/src/lib/resources.ts` is the
  single source of truth for the resource list).
- ~~**Port, settlement, and road iconography.**~~ Done — same map-based
  pattern extended to `PORT_ICON`, plus `SettlementIcon`/`CityIcon`/
  `RoadIcon`/`RobberIcon`. Settlement/city/road glyphs use `currentColor`
  so `Board.svelte` recolors them per owner; the robber glyph is fixed/
  neutral. All glyphs are `pointer-events: none` and layered on top of the
  existing invisible click/highlight targets, so click-to-select is
  unaffected. The glyph choice itself is intentionally a placeholder —
  the point of the map-based indirection is that swapping in real art
  later means editing the icon files, never touching `Board.svelte`.
- **Visible, paced bot turns.** `server/bots.py`'s `step_bots` resolves
  every consecutive bot action synchronously inside one HTTP response
  (`server/app.py`'s `post_action`), so from a human's perspective an
  entire multi-action bot turn (roll, build, trade, end turn, repeated for
  every bot seat) completes instantly between one click and the next --
  too fast to actually follow what happened. Raised during play; explicitly
  not scoped yet, needs a real design pass rather than a quick patch:
  - A flat artificial delay (e.g. `time.sleep` between actions in
    `step_bots`) only makes the response take longer -- the client still
    only ever sees the before/after state, not each intermediate action, so
    it doesn't actually solve "hard to see what happened."
  - Genuinely showing each bot action needs the backend to expose an
    action log or intermediate states per response (or a streaming
    mechanism), which the frontend then replays with a pause between
    steps -- likely via animation (a build/trade/dice-roll transition
    rather than an instant redraw), which is probably the right shape for
    this rather than a plain delay. This is real architecture work: what
    the wire format for a turn's action log looks like, how it interacts
    with the existing polling loop (`App.svelte`'s `POLL_INTERVAL_MS`) and
    the `isBusy`/"applying your move" state, and whether it's built once as
    a generic "replay this turn" capability or specific to bot turns.
- **Resume a game by ID.** `GameSetup.svelte` always starts a new game;
  there's no "rejoin an existing `game_id`" flow. Came up when switching
  from the rebuild-and-restart workflow to the Vite dev server — a hard
  page reload has no way to reconnect to the game already running
  server-side. Would need `game_id` persisted somewhere client-side
  (e.g. the URL or `localStorage`) and a `GameSetup.svelte` path that
  calls `getState`/`getLegalActions` instead of `createGame`.

## Engine

- **Counter-trades / free trade negotiation.** README decision 5
  deliberately scoped domestic trade down to propose → each other player
  accepts or rejects in turn, first accept wins, no counter-offers — "a
  documented scope simplification, not a rules claim (the real game allows
  free negotiation)." Surfaced again from actual play: an offer that gets
  rejected outright today might have been accepted with a different
  bundle, but there's no way to propose one back. This is a real engine
  gap, not a UI one — there's no `CounterTrade` action in `engine/actions.py`,
  and adding one means new negotiation state (who's countering whom, with
  what bundle, and how/when the original offer expires), not just a new
  action variant. Worth scoping properly before starting, since it touches
  `state.trade_offer`/`state.trade_responders` and every agent that
  currently handles `ProposeTrade`/`AcceptTrade`/`RejectTrade`
  (`agents/random_agent.py`, `agents/heuristic.py`, `agents/human.py`,
  `server/bots.py`).

## Known gaps from the GUI build (`docs/plans/gui-web-frontend.md`)

- ~~**`web/src/lib/geometry.test.ts` (Vitest) still isn't runnable.**~~
  Resolved — `npm install` succeeded on a later attempt (the network stall
  was transient after all, not permanent); all 8 geometry tests pass.
- **Only one human seat is playable per browser tab.** No in-tab hot-seat
  switching between multiple human seats — `GameSetup.svelte` states this
  directly in its UI copy rather than silently under-delivering, but true
  multi-human local play (as `cli.py` already supports) isn't built for the
  web GUI yet.
- **No bit-identity cross-check between the HTTP path and
  `experiments/rollout.run_game`.** Deferred during step 3 as real design
  work beyond what the phased plan called for (needs a `driver_seed` exposed
  through the API, plus reconciling `server/bots.py`'s per-seat agent shape
  with `rollout.py`'s `AgentFactory` shape). Would be a strong addition to
  the verification story if ever revisited.
