# Phase 3 benchmark — the three acceptance gates

Six committed arms (`benchmark_<mode>_p<players>.json`): each of
`random_vs_random`, `heuristic_vs_random`, `heuristic_vs_heuristic` run for
both 3p and 4p. Seat rotation is exact in every arm (`--games` chosen as a
multiple of the lineup length) — `by_role.*.seat_occupancy_counts` shows
each role in each seat exactly `n/k` times, and `by_seat` breaks out win
rate per seat regardless of role. `engine_seed`/`driver_seed` are `k`
consecutive integers starting at 1 (`benchmark.py`'s rotation offset,
`engine_seed % k`, is only exact under that convention — see the module
docstring).

## Gate 1 — RandomAgent vs RandomAgent -> 1/n per seat

`random_vs_random`, 9,999 games (3p) / 10,000 games (4p):

- 3p: 33.33% [32.80%, 33.87%] — expected 33.33%.
- 4p: 25.00% [24.58%, 25.43%] — expected 25.00%.

Both dead center on the null, for every seat (rotation is a no-op here
since every seat holds the same role — this gate tests seat bookkeeping in
the driver, not the rotation logic itself). **Pass.**

## Gate 2 — HeuristicAgent vs RandomAgent -> heuristic clearly above 1/n

`heuristic_vs_random`, 3,999 games (3p) / 4,000 games (4p), 1 heuristic seat
rotated against 2 or 3 random seats:

- 3p: heuristic 94.25% [93.48%, 94.93%] vs. random (pooled) 2.88% [2.53%,
  3.27%]. Two-proportion test: z=-100.1, p≈0.
- 4p: heuristic 91.83% [90.94%, 92.63%] vs. random (pooled) 2.72% [2.45%,
  3.03%]. z=-112.7, p≈0.

Non-overlapping CIs by a wide margin in both player counts. **Pass** — and a
much larger effect than the "tens of points" the plan anticipated, which is
itself worth noting: a single fixed-priority heuristic (city > settlement >
dev card > knight-on-robber > monopoly/YoP-to-unblock > bank/port trade
that enables a build > road > end turn) beats uniform-random play by a wide
margin in base-game Catan. The intermediate diagnostic agrees: `mean_resources_through_turn_10` is
roughly double for heuristic in both player counts (3p: 6.52 vs. 3.18; 4p:
6.44 vs. 3.20) — the heuristic wins by producing more, not by exploiting an
engine quirk.

## Gate 3 — HeuristicAgent vs HeuristicAgent -> per-seat win rates

`heuristic_vs_heuristic`, 3,999 games (3p) / 4,000 games (4p), self-play:

- 3p by-seat: seat 0-2 all ~33.3%, overlapping CIs.
- 4p by-seat: seat 0 24.80% [23.49%, 26.16%], seat 1 25.87% [24.54%,
  27.25%], seat 2 24.55% [23.24%, 25.91%], seat 3 24.77% [23.46%, 26.14%].

**No detectable seat-0 first-pick advantage under heuristic self-play** at
this sample size — all four seats' CIs overlap heavily. Per the plan's own
framing this is an *ambiguous* result, not a bug signal, and it stands
given gate 1 passes and the rotation counts are exact: either the
first-pick edge measured under random play (`a1_seed1_seat0_p4.md`) doesn't
survive purposeful play at these seed ranges, or it's real but too small to
resolve at n=4,000 per-seat (~1,000 games/seat). Not resolved further here
— a targeted A/B at larger n is a follow-up, not required for this gate.
**Pass** (gate 1 held, rotation counts exact).

## `_check_win` re-measurement (README decision 15)

Phase 2 measured `non_winner_exceeded_ten` at 0.22% under stratified random
play (`tables.md`). Under `heuristic_vs_heuristic`:

- 3p: 0.150% (6/3,999)
- 4p: 0.125% (5/4,000)

The rate did **not** rise under purposeful play, contrary to the plan's own
speculation ("heuristics build roads purposefully, so Longest Road changes
hands more often") — if anything it's slightly lower, likely because
heuristic games are shorter on average with fewer late-game Longest-Road
swings per unit of play. Both figures stay in the same order of magnitude
as Phase 2's 0.22% and are low in absolute terms. **Decision: not fixed in
Phase 3** — the measured rate does not clear a "non-negligible" bar, so
`_check_win` is left evaluating only the turn player. Recorded here (and in
the README) rather than silently dropped, per the plan's instruction to
record the outcome either way.

## Known biases (repeated from the harness output)

- `PlayVictoryPoint` is not gated by `has_played_dev_card_this_turn` --
  every agent reveals VP cards immediately.
- Bank shortage (`_produce`) skips a resource when the bank cannot cover
  all claimants; heuristic agents produce more, so they hit this more often
  than random agents did in Phase 2.
- `HeuristicAgent` is hand-tuned against Phase 2's random-play results; a
  rule that helps against random opponents need not help against a good
  one. `heuristic_vs_heuristic` is the check on this, run above.
