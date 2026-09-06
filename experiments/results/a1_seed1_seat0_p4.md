# A1 — per-vertex win-rate ranking on a fixed board

`a1_seed1_seat0_p4.json` — engine_seed=1, seat=0 (first picker), 4 players,
800 games per arm (54 arms, one per vertex), stratified random policy.

Per-arm effect size δ≈800 games resolves roughly ±8pp at the 95% level per
the plan's sample-size table (interpolated between the 5pp/1400-game and
2pp/8700-game rows) — enough to see large, real separations, not enough to
resolve close calls between similarly-ranked vertices without pairwise
significance testing.

## Headline

- **Best vertex: 30** — win rate 62.3% (Wilson CI [58.8%, 65.5%]). Features:
  pip_sum=11, 3 distinct resources, no port.
- **Worst vertex: 8** — win rate 6.1% (CI [4.7%, 8.0%]). Features: pip_sum=0
  (touches only the desert, no other land hex), 1 hex.

This is a much larger and cleaner separation than the plan's "attenuation
under random play" caveat anticipated — the harness clearly resolves even
under a uniformly-random policy, at least for extreme vertices. The
intermediate metrics agree: top-ranked vertices also lead on
`mean_resources_through_turn_10`/`_20`, confirming the win-rate signal isn't
an artifact.

## How to read `arms`

Keyed by vertex id (stringified), each arm has `win_rate` +
`win_rate_wilson_ci` (headline, policy-conditional), plus
`mean_resources_through_turn_10/20`, `mean_turns_to_5vp`,
`turns_to_5vp_reached_fraction` (policy-insensitive-ish intermediate
metrics — still a random policy, but not a 1/n win indicator), and
`vertex_features` (the analytic site quality from `features.py`).

## Caveats

- Seat 0 only — the first picker sees the whole board; this ranking is not
  meaningful for other seats without re-running with `--seat`.
- Policy is `stratified` random, not any heuristic — see the Phase 2 plan's
  "win rate is not the primary metric" note.
- No multiple-comparison correction applied to the ranking itself (54 arms);
  treat the ranking as ordinal, not as 54 pairwise significance claims.
