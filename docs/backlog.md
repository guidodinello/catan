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
- ~~**Visible, paced bot turns**~~ **+ ~~dice roll history~~.** Done —
  implemented together per this entry's own analysis that they share one
  root fix (a durable per-action record surviving a synchronous bot
  batch). `server/bots.py`'s `step_bots` now returns `list[TrailEntry]`
  (one entry per action it applied — no `GameState`/`PlayerState` change
  needed; `engine/state.py`'s `dice_roll` already held what was needed).
  `server/app.py`'s `post_action`/`create_game` fold the human's own
  action (if any) plus every bot action that followed into one
  `action_trail`, serialized via `server/serialize.py`'s new
  `serialize_trail` (factored out of `serialize_action`'s existing
  per-kind field extraction, so the two never drift on what a kind's
  fields are). One redaction call made along the way: `Discard.resources`
  is stripped from the public trail (nowhere else exposes which specific
  cards someone discarded, only aggregate amounts owed) — every other
  kind's fields are already public knowledge in the real game (a rolled
  die, a placed settlement, an announced `PlayMonopoly`/`PlayYearOfPlenty`
  resource, a `ProposeTrade` bundle...).
  `App.svelte` reveals a response's `action_trail` one entry at a time
  (`REPLAY_DELAY_MS`, capped log `LOG_CAP`) instead of jumping straight to
  it, reusing `isBusy`/"applying your move" for the whole paced reveal
  rather than a parallel state machine — a "Skip" button flushes the rest
  immediately (useful for a long bot batch, e.g. spectating an all-bot
  game). New `lib/actionLog.ts` (`describeTrailEntry`, `recentRolls` —
  both vitest-covered) feeds a new `ActivityLog.svelte` (a scrollable
  "Player 2 rolled 8, ..." feed) and `HandSummary.svelte`'s new "Recent
  rolls" strip, so dice history and paced bot-turn visibility both read
  from the exact same revealed-log array, not two parallel mechanisms.
  Scope note, stated honestly rather than overclaimed: there are no
  intermediate `GameState` snapshots, only intermediate *actions* — so the
  board/panel still jump straight to the final state immediately; only
  the *text log* is paced. A fuller "animate the board turn by turn"
  version (settlements fading in, a dice-roll animation, etc.) remains a
  distinct, larger follow-up if ever wanted, not something this delivers.
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
