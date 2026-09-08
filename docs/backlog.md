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
- ~~**A build-costs reference table.**~~ Done — chose the endpoint over
  hand-copying, deliberately against this repo's own precedent of
  hand-mirroring engine enums (`resources.ts`'s `RESOURCES`/
  `TERRAIN_RESOURCE`, `icons/index.ts`'s `TERRAIN_ICON`/`DEV_CARD_ICON`):
  those mirrors fail *loudly* (`server/serialize.py` sends `.name` on the
  wire, so a renamed enum member breaks rendering immediately), whereas a
  wrong hand-copied `CITY_COST` would fail *silently* — a wrong number
  shown forever, no type error, no failing test. That asymmetry is why
  SSOT wins here even though it costs a `uvicorn` restart (which, per the
  "sessions don't survive a restart" item below, drops in-progress games).
  New `server/serialize.py`'s `serialize_build_costs()` builds a
  `{"ROAD": {...}, "SETTLEMENT": {...}, "CITY": {...}, "DEV_CARD": {...}}`
  payload straight from the imported `ROAD_COST`/`SETTLEMENT_COST`/
  `CITY_COST`/`DEV_CARD_COST`, served by a new `GET /api/build_costs` in
  `server/app.py`. One real wrinkle, caught before writing the endpoint:
  the existing `_resources_to_dict` (used by `bank`/a hand's `resources`)
  indexes *all five* `Resource` members and `KeyError`s on a sparse dict
  like `ROAD_COST` (no WOOL/GRAIN/ORE keys at all) — so it couldn't be
  reused as-is, and a new `_sparse_resources_to_dict` handles the cost
  dicts' sparse shape instead (each entry only lists the resources that
  build kind actually costs; no `{"WOOL": 0}` noise). `web/src/lib/api.ts`
  gained a matching `getBuildCosts()`; `web/src/lib/BuildCosts.svelte` is a
  new panel modeled on `HandSummary.svelte`'s existing resource-row
  pattern (same `RESOURCE_ICON` 18px-svg-plus-count markup, reused rather
  than reinvented). `App.svelte` fetches it once `onMount` — not threaded
  through `startGame`/`resumeGame` — since build costs are static and
  viewer-independent, which covers "Resume a game" for free with no
  `lastGame.ts`-style caching needed; a failed fetch just leaves the panel
  hidden rather than blocking play. Scope note: `GET /api/geometry` (the
  still-open gap under "Resume a game by ID" above) was deliberately
  **not** bundled in here even though the restart was already being paid
  for — it's a distinct backlog item.

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
- ~~**Counter-trades / free trade negotiation.**~~ Done — bounded, exactly
  one counter-offer, decided and scoped rather than open-ended
  negotiation (see README decision 5 for the rationale). A new
  `CounterTrade` action (`engine/actions.py`) and one new `TradeOffer`
  field (`counter_of: int | None` — the original proposer's id, marking an
  offer as a counter) are the entire state addition; the bound is
  structural (`_trade_response_legal` never offers a further
  `CounterTrade` against an offer that already has `counter_of` set), so no
  expiry mechanism was needed. Countering drops the remaining original
  responders (`trade_responders` becomes `[original proposer]` only) and
  resolving the counter — accept or reject — returns to `Phase.MAIN` with
  `current_player` unchanged, so the original proposer's turn continues.
  Every call site flagged in the original note got updated:
  `agents/random_agent.py` (which also had a real pre-existing bug fixed
  along the way — `build_random_trade_offer` read
  `state.players[state.current_player]` instead of the acting player,
  wrong the moment the actor differs from the turn player, as it now can
  mid-negotiation), `agents/heuristic.py` (still rejects everything, never
  counters), `agents/human.py`, `server/bots.py`, `server/serialize.py`,
  `server/app.py`, `experiments/rollout.py`'s sentinel guard, and the
  Svelte layer (`TradeForm`, `TradeOfferBanner`, `ActionPanel`,
  `actionLog.ts`, and two real `App.svelte` bugs the new action type would
  otherwise have caused: the auto-reject effect firing while a counter was
  actually available, and `describeTradeOutcome` misreporting outcomes on
  both the human-as-proposer and human-as-counterer paths). Adding a third
  action type to `AWAIT_TRADE_RESPONSE`'s legal set changes
  `StratifiedRandomAgent`'s action-type draw, so
  `tests/golden_phase2_records.json` was regenerated (see the commit/PR
  for the reported before/after diff) rather than left as a documented
  exception — unlike the VP-card fix above, this change could not leave
  the bit-identity test green.

## Server / infrastructure

- ~~**Sessions don't survive a backend restart.**~~ Done —
  `server/persistence.py` pickles each `GameSession` to
  `.catan-sessions/<game_id>.pickle` after every mutating request
  (`server/app.py`'s `create_game`/`post_action`/`delete_game`) and a
  FastAPI lifespan handler reloads them at startup, so "Resume a game by
  ID" now has something to reconnect to after a restart. `GameState`
  pickled cleanly and cheaply as expected (5.4 KB for a fresh 3-player
  game; `random.Random`'s exact stream position round-trips). The one
  measured surprise: a `slots=True` dataclass field rename/removal/addition
  raises `AttributeError` (at load, or on first read of a new field) —
  loud, not silent as originally worried — but `Phase`/`DevCardType`/
  `Resource`/`Terrain`/`PortType`'s `auto()` numbering restores *by value*,
  so inserting an enum member mid-list silently loads a *different*,
  wrong member with no exception at all. A shape fingerprint (dataclass
  field names + every persisted enum's name→value map) gates every load
  to catch exactly that; a mismatch discards the snapshot (logged) rather
  than resurrecting a corrupted one. Persists write-through on every
  mutation, not just on a shutdown hook, since the restart this exists to
  survive is also a crash or `kill -9`, not only Ctrl+C. Bot `Agent` RNG
  streams are not persisted, as flagged as skippable — only each seat's
  kind is saved, and agents are rebuilt fresh (a new `driver_seed`) on
  load. See README decision 22 for the on-disk format.
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
