# A2 — feature-bucketed win rate across random boards

`a2_seat0_p3.json` / `a2_seat0_p4.json` — seat=0, 10,000 games each,
engine_seed_base=1 (each game a distinct random board), stratified random
policy. No scripting: the treatment seat's first settlement is whatever the
policy actually picked; features are looked up post hoc.

## Sanity check (required by the plan): pooled win rate ≈ 1/n_players

| Players | Pooled win rate | Wilson CI | Expected (1/n) |
|---|---|---|---|
| 3 | 33.86% | [32.9%, 34.8%] | 33.3% |
| 4 | 25.26% | [24.4%, 26.1%] | 25.0% |

Both pass — no seat-assignment bug in the harness.

## Headline: win rate by distinct-resource count at the first settlement

| distinct resources | 3p win rate | 3p CI | 4p win rate | 4p CI |
|---|---|---|---|---|
| 0 | 11.4% | [7.3%, 17.3%] | 7.0% | [3.9%, 12.0%] |
| 1 | 21.6% | [20.4%, 23.0%] | 13.4% | [12.4%, 14.6%] |
| 2 | 37.7% | [36.2%, 39.3%] | 29.1% | [27.6%, 30.5%] |
| 3 | 50.5% | [48.4%, 52.6%] | 41.0% | [39.0%, 43.1%] |

Strictly monotonic in both player counts, with non-overlapping CIs at every
step — a clean, unambiguous face-validity result: touching more distinct
resources at your first settlement is worth a large amount of win
probability, even under uniformly-random play. `by_pip_sum_bucket` (see the
JSON) shows the same monotonic pattern for pip count.

## Caveats

- 3p and 4p reported separately, never pooled, per the plan.
- `distinct_resources=0` cells are thin (n=158) — CI is correspondingly
  wide; still clearly separated from the other buckets.
- Win rate is the headline metric here but is corroborated by
  `mean_resources_through_turn_10` in every bucket (see JSON) tracking the
  same ordering — not an artifact of a noisy 1/n indicator.
