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
  **Follow-on, same session:** each `RollDice` `TrailEntry` also carries
  `production: dict[int, dict[Resource, int]] | None` — who gained which
  resources from that roll. No `engine/` change needed here either:
  `engine.game._produce` computes and applies the gain but never returns
  it, so `server/bots.py`'s new `apply_and_record` (now shared by
  `step_bots` and `post_action`, replacing each building its own
  `TrailEntry` separately) reconstructs it by diffing every player's hand
  immediately before/after applying the roll. Not a new redaction concern:
  who gains what from a roll is fully determined by public information
  already (the board, everyone's settlements/cities) — any player at the
  table could work it out themselves, same as the roll itself. Rendered
  inline in `ActivityLog`'s line for that roll via `actionLog.ts`'s new
  `describeProduction` ("Player 1 +1 LUMBER, +1 GRAIN; Player 2 +1 GRAIN"),
  omitted entirely when a roll produced nothing (a 7, or no settled hex
  matches the number).
  **Second follow-on, same session — deal logging + reject collapsing:**
  `TrailEntry` also gained `trade_offer: TradeOffer | None`, set only for
  `AcceptTrade`/`RejectTrade` (neither carries fields of its own either, so
  the deal/offer being responded to would otherwise be invisible) —
  snapshotted from `state.trade_offer` in `apply_and_record` immediately
  before it applies the response and the engine clears it. Not a new
  redaction concern: a domestic trade offer is already public once
  proposed. `serialize_trail_entry` reuses a new shared
  `_serialize_trade_offer` helper (factored out of `player_view`'s
  existing inline dict, so the two shapes can't drift). `ActivityLog`'s
  `AcceptTrade` line now reads as an actual deal ("Player 2 traded with
  Player 0: gave 1 BRICK for 1 WOOL") instead of just "accepted the
  trade". Separately, `actionLog.ts`'s new `groupActivityEntries` collapses
  a run of consecutive `RejectTrade` entries that killed a trade outright
  (nobody accepted) into one "Everyone rejected the trade" line — a purely
  frontend, display-only grouping over the already-complete trail, no
  entries dropped; a run that's instead followed by an eventual
  `AcceptTrade` is left as individual lines, since someone did say yes.
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
- **A build-costs reference table.** Came up during play — no on-screen
  reminder of what a settlement/city/road/dev card actually costs, so it's
  guesswork or an alt-tab to the rulebook. The real numbers already live
  as named constants in `engine/game.py` (re-exported from
  `engine/__init__.py`): `ROAD_COST` (1 BRICK, 1 LUMBER), `SETTLEMENT_COST`
  (1 BRICK, 1 LUMBER, 1 WOOL, 1 GRAIN), `CITY_COST` (2 GRAIN, 3 ORE),
  `DEV_CARD_COST` (1 ORE, 1 WOOL, 1 GRAIN) — frontend-only, hot-reload:
  either a small static reference table/panel (reusing `RESOURCE_ICON`
  glyphs) with these four rows hand-copied into a `web/src/lib/`
  constant, or, to avoid a second source of truth that could drift from
  `engine/game.py` if a cost ever changes, a tiny dedicated
  `GET /api/build_costs`-style endpoint (a `server/` change, needing
  go-ahead + restart) that serializes the real constants directly. Worth
  deciding which before starting: hand-copied numbers are simpler and
  ship immediately, but they're a second source of truth for values that
  currently only exist once, in `engine/game.py`.

## Engine

- ~~**Bug: hidden Victory Point cards don't trigger an automatic win.**~~
  Done — `engine/game.py`'s `_check_win` now drives off a new
  `true_victory_points` (public tally plus every VP card still in
  `dev_hand`, revealed or not), not `victory_points` (which keeps its exact
  prior public-tally semantics; every existing caller/serializer is
  unchanged). `_buy_dev_card` gained a trailing `_check_win` too, since
  buying a card can now cross the true-VP threshold on its own, with no
  separate `PlayVictoryPoint`. On a win, the winner's VP cards are revealed
  (`dev_hand` -> `revealed_vp_cards`) in the very same step that sets
  `GAME_OVER` (a new `_reveal_victory_point_cards` helper), which is what
  keeps the fix contained to `engine/` alone: `server/serialize.py` needed
  **no change**, since it only ever serializes the public `victory_points`,
  and that number becomes accurate exactly when the game ends, never an
  instant before — an opponent's hand still reads fully redacted right up
  until then. Verified the redaction claim rather than assuming it: the
  other place VP information could leak is `_dev_card_play_actions`
  offering `PlayVictoryPoint` (which would betray a hidden card), but
  `server/app.py`'s `get_legal_actions` already returns `[]` unless
  `viewer == acting_player`, so that was never actually reachable.
  `true_victory_points` is deliberately not re-exported from
  `engine/__init__.py` — it's engine-internal on purpose, only `_check_win`
  calls it. Losers' hands stay redacted forever after `GAME_OVER` (a
  post-game full reveal, if ever wanted, is a separate product decision).
  `tests/golden_phase2_records.json` was deliberately **not** regenerated
  as part of this fix — see the commit/PR for the reported diff instead.
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

## Server / infrastructure

- **Sessions don't survive a backend restart.** `server/sessions.py` keeps
  every game in an in-memory `dict[str, GameSession]` on the one running
  `uvicorn` process — restarting it (needed for any `server/`/`engine/`
  code change) loses every game in progress, "Resume a game by ID"
  included (that feature only reconnects the *browser* to a game still
  running server-side; it has nothing to reconnect to once the process
  restarts). Raised during play.
  - **Games are pickle-friendly essentially for free.** `engine.state.GameState`
    (`board`, `players`, `dev_deck`, `bank`, `trade_offer`, etc., plus
    `rng: random.Random`) is plain dataclasses/dicts/lists/Enums — no
    exotic types blocking a round trip, and `random.Random` pickles its
    *exact* stream position, not just a reseed. `CatanGame` itself is
    stateless (`num_players` only), so it doesn't need persisting, just
    reconstructing fresh. `server/serialize.py` is **not** reusable for
    this — it's a one-way, lossy, per-viewer-*redacted* JSON projection
    built for the browser, never meant to round-trip back into a live game.
  - **The real catch:** pickle round-trips are tied to the *exact* class
    shape at dump time. If the very code change that triggered the
    restart reshapes `engine/state.py`'s dataclasses (add/remove/rename a
    field), the old pickle can fail to load, or worse, silently load into
    a stale-shaped object missing the new field. So this would reliably
    survive restarts for changes that don't reshape engine state (most
    `server/app.py` routes, `web/` changes, non-structural engine bug
    fixes) but not one that adds/removes/renames a `PlayerState`/`GameState`
    field. (The hidden-VP win-condition fix above turned out not to be an
    example of that: it added no new state field, since the win reveal
    reuses the existing `dev_hand`/`revealed_vp_cards` fields.) Also needs
    a decision on *when* to persist
    (every action = safest, extra I/O per request; only on a graceful
    shutdown hook = simpler, loses the game on a crash/kill) and whether
    bot `Agent` RNG streams are worth persisting too (skippable — losing
    them just means bots reseed fresh post-restart, not a correctness
    issue, only perfect reproducibility). Not attempted here; would need
    a real plan first given the fragility, same as other backend-touching
    items.
  - **Bigger picture, if this ever goes properly online (multiplayer over
    the internet, not just local dev):** pickle-to-disk stops being the
    right answer entirely, for two separate reasons, not one. (1) A
    single in-memory dict only works because there's one process — real
    online play needs multiple app instances behind a load balancer, so
    game state has to live somewhere shared (a DB, Redis, etc.), not
    process memory. (2) Pickle's fragility becomes a production risk
    rather than a dev inconvenience: locally, "pickled by yesterday's
    code" is a rare, self-inflicted event; in a real deployment, rows
    written across months of deploys can't all be assumed to match the
    current code's exact class shapes. A genuine online version would
    need `server/sessions.py`'s in-memory dict replaced by a real
    datastore with a *versioned/migratable* schema (JSON with a schema
    version field, or similar) — not raw Python pickle. That's a
    materially bigger rewrite than the local-restart pickle idea above,
    not an extension of it.

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
