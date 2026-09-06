# Resource / VP probability tables

`tables_p3.json` / `tables_p4.json` — 10,000 games each, engine_seed_base=
100000 (a different seed range from the A2 runs, so the two experiments'
game samples don't silently overlap), stratified random policy.

## Table 1 — analytic dice/production (exact, no simulation)

Sanity: `dice_distribution` reproduces the textbook 2d6 table exactly
(P(7)=1/6, P(2)=P(12)=1/36, etc.) — see the JSON's
`table_1_analytic_dice_production.dice_distribution`.

## Table 2 — VP trajectory / P(win | VP=v at turn t)

Turns are grouped into 5-turn buckets (`turn_bucket_size`) to keep the
committed table compact; ~1,200 cells survive the `min_cell_count=30`
suppression per player count. Face-validity spot check, 4-player table,
VP=9:

| turn bucket start | n | P(win \| VP=9 in this window) |
|---|---|---|
| 71 | 53 | 92.5% |
| 86 | 235 | 91.9% |
| 96 | 491 | 94.3% |

Reaching 9 VP overwhelmingly predicts winning, as expected (one more
settlement/city/dev-card away from 10) — this is the strongest available
face-validity check on the whole VP-trajectory table.

## Table 3 — game shape

4-player, 10,000 games:

- Turn-length distribution: mean 273, median 256, p90 410, max 1014.
- Winner's final VP: mean 10.06, median/p90 10, max 12 (dev-card VP or a
  same-turn multi-VP jump can overshoot 10 slightly before `_check_win`
  fires).
- **Engine finding, recorded not silently smoothed**: `non_winner_exceeded_
  ten_vp_transiently` fired in 22/10,000 games (0.22%) — confirms the
  documented gap where `_check_win` only evaluates the turn player, so a
  build that transfers Longest Road can push a third player to >=10 VP
  undetected until their own next turn. Self-resolves within a round; not a
  hang. Worth a look for Phase 3, not fixed here (Phase 2 is analysis-only).

## Table 4 — empirical income

Mean cards received per turn by resource (4-player): grain 0.42, wool 0.41,
lumber 0.41, ore 0.32, brick 0.30 — grain/wool/lumber ahead of ore/brick,
consistent with the terrain-count asymmetry (4 forest/4 pasture/4 field vs.
3 hill/3 mountain) even after robber blocking and bank-shortage effects.

## Caveats

- All figures are policy-conditional (stratified random) — see the Phase 2
  plan's central caveat.
- Table 4's bank-shortage diagnostic is folded into the mean (a shortage
  silently reduces a turn's recorded gain to 0 for the affected resource);
  it is not broken out as a separate count in this run. If late-game income
  bias needs to be isolated precisely, that's a targeted follow-up, not
  re-derivable from this table alone.
