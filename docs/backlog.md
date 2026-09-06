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
  - Design thought (not scoped, not started): the smallest version of "an
    action log per response" wouldn't need full intermediate `GameState`
    snapshots to start being useful -- `post_action` already loops inside
    `step_bots` applying one `Action` at a time, so it could cheaply collect
    a `[{player_id, action_kind}, ...]` trail alongside the final state
    (no new redaction concerns, since `kind`/`player_id` are already public
    knowledge, same as any of today's `LegalAction.kind`s) and let the
    frontend render it as a scrolling "Player 2 rolled 8, Player 2 built a
    road, ..." log during `isBusy`, animated or not. That's a real
    `server/`+`App.svelte` change either way (not attempted here per this
    item's own stop condition), but it's a narrower first slice than full
    action replay/animation if this gets picked up later.
- **Dice roll history (last 5), not just the latest.** `HandSummary.svelte`
  only ever shows `state.dice_roll`, the single most recent roll —
  `engine/state.py`'s `GameState.dice_roll: tuple[int, int] | None` has no
  history, and neither does `server/serialize.py`'s wire format. Raised
  during play. Two real options, not a quick fix, and they're not
  equivalent:
  1. **Frontend-only rolling buffer.** `App.svelte` appends `dice_roll` to
     a capped `lastRolls` array each time `gameState` changes (poll tick or
     action response) and passes it to `HandSummary.svelte` instead of the
     single value. Hot-reload only, no backend change, doable immediately.
     **But it has a real, silent gap, not just a cosmetic one**: `step_bots`
     (`server/bots.py`) resolves every consecutive bot action synchronously
     inside one HTTP response, so a multi-bot-turn batch between two human
     turns can contain several actual die rolls, and the client only ever
     sees the *last* `gameState` in that batch — every roll before the final
     one in the same batch never reaches the frontend at all. This is the
     exact same root cause already written up under "Visible, paced bot
     turns" above. So a frontend-only buffer wouldn't be "the last 5 rolls,"
     it'd be "the last 5 rolls the client happened to observe" — silently
     wrong (looks complete, isn't) unless the gap is disclosed in the UI
     copy itself (e.g. a tooltip/caption noting bot-turn rolls between
     human turns may be skipped), not just a code comment nobody playing
     the game would ever see.
  2. **Server-tracked roll history.** Add an actual list to `GameState`
     (e.g. `dice_roll_history: list[tuple[int, int]]`, appended to
     wherever dice are rolled today, capped at some length) and expose it
     in `server/serialize.py`'s output. Correct and complete — captures
     every roll including mid-batch bot rolls — but it's an `engine/` +
     `server/` change, which means restarting the live `uvicorn` on `:8000`
     and losing the game in progress. Needs explicit go-ahead first, like
     the other backend-touching items here.
  - **Recommendation:** don't ship option 1 as a silent fix — the gap is
    exactly the kind of thing a player would notice and distrust ("wait, I
    swear there were more rolls than that") without ever being told why.
    Either (a) do option 2 whenever "Visible, paced bot turns" gets
    picked up, since both need the same underlying capability (a durable,
    per-action-or-per-roll record that survives a synchronous bot batch —
    solving one mostly hands you the other), or (b) if option 1 is wanted
    sooner as a stopgap, ship it with the gap explicitly disclosed in the
    UI, not hidden.
- ~~**Resume a game by ID.**~~ Done — `GameSetup.svelte` has a "Resume a
  game" section (game id + optional viewer seat, prefilled from
  `localStorage` via `web/src/lib/lastGame.ts`); `App.svelte`'s
  `resumeGame` calls `getState`/`getLegalActions` directly instead of
  `createGame`. One real constraint: `server/app.py` has no endpoint that
  returns board geometry alone (only `POST /api/games` does), so resuming
  reuses a *cached* geometry from any earlier game in this browser rather
  than re-fetching it — valid because `serialize_geometry()` is RNG-free
  and identical for every game, but it does mean resume only works once
  this browser has created/resumed at least one game before. A dedicated
  `GET /api/geometry` endpoint would remove that restriction, but that's a
  `server/` change, out of scope for this frontend-only pass.

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
  - Design thought (not scoped, not started): a smaller first cut than open
    negotiation might be "one counter-offer, then the original proposer's
    turn ends" (bounded, so it can't loop indefinitely and doesn't need an
    expiry mechanism) rather than jumping straight to real multi-round
    negotiation -- but even that bounded version still needs a real
    `CounterTrade` action, new `state` fields for who's countering whom
    with what, and every agent above updated to at least reject it
    sanely. Worth deciding the bounded-vs-open-ended shape explicitly
    before writing any of it, since they imply different state shapes.

## Known gaps from the GUI build (`docs/plans/gui-web-frontend.md`)

- ~~**`web/src/lib/geometry.test.ts` (Vitest) still isn't runnable.**~~
  Resolved — `npm install` succeeded on a later attempt (the network stall
  was transient after all, not permanent); all 8 geometry tests pass.
- ~~**Only one human seat is playable per browser tab.**~~ Done —
  `App.svelte` now tracks every seat `GameSetup.svelte`'s chosen
  `seat_kinds` marked `"human"` (`humanSeats`), and a "Viewing as" switcher
  appears whenever there's more than one. Switching re-fetches that seat's
  own redacted `state`/`legal_actions` and clears any UI state scoped to
  the previous viewer (open trade/dev-card forms, an in-progress
  road-building pick), so nothing stale leaks across the switch. It's
  pass-and-play, not per-seat secrecy within one screen — the UI copy says
  so. One known residual gap: after "Resume a game" (see above), only the
  single seat typed into the resume form is known, since a resumed game's
  full `seat_kinds` isn't available to the client — switching among a
  resumed game's *other* human seats would need a server-side way to learn
  the lineup, which is a `server/` change out of scope here.
- **No bit-identity cross-check between the HTTP path and
  `experiments/rollout.run_game`.** Deferred during step 3 as real design
  work beyond what the phased plan called for (needs a `driver_seed` exposed
  through the API, plus reconciling `server/bots.py`'s per-seat agent shape
  with `rollout.py`'s `AgentFactory` shape). Would be a strong addition to
  the verification story if ever revisited. Sharpening the first piece:
  `server/app.py`'s `create_game` already generates a `driver_seed =
  secrets.randbits(63)` per game (used to build each seat's `Agent` via
  `build_agents`) -- it's just discarded after use rather than returned in
  `CreateGameResponse`, so exposing it is a small, additive API change (a
  new response field), not a redesign. The harder half is still
  `rollout.py`'s `AgentFactory` shape vs. `server/bots.py`'s per-seat
  `Agent | None` list: reconciling those (or writing a small adapter
  between them) is the real work, not the seed plumbing.
