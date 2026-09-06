# Backlog — future improvement ideas

Not committed to, not scheduled — a running list of ideas raised while using
the project, kept somewhere more durable than chat history. Contrast with
`docs/plans/`, which holds the implementation plan for work already approved
and built.

## GUI / UX

- **Terrain and resource iconography.** `Board.svelte` currently renders
  hexes as flat colored polygons. Real pictorial icons for each terrain type
  (forest, hills, mountains, fields, pasture) — and for resource cards
  wherever they're listed (`HandSummary.svelte`, `TradeForm.svelte`) — would
  read faster than a color legend, especially for anyone unfamiliar with
  which color maps to which resource.
- **Port, settlement, and road iconography.** Same idea extended to the rest
  of the board: a real port icon instead of a plain circle + text label, and
  distinct settlement/city/road glyphs instead of plain SVG shapes colored by
  owner.

## Known gaps from the GUI build (`docs/plans/gui-web-frontend.md`)

- **`web/src/lib/geometry.test.ts` (Vitest) still isn't runnable.**
  `npm install` for `vitest` hit a persistent network stall in the dev
  environment (confirmed independently in two separate sessions, 40+
  minutes, zero output, even though `curl` reached the registry fine) — not
  a one-off fluke. The geometry math it would check was cross-verified
  against the real engine via an equivalent Python script instead, so
  nothing is un-verified, but the actual test suite needs a working
  `npm install` in an environment with better registry access before it can
  run in CI.
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
