# First catan league: 11 agents, 4 seats, anchored Bradley-Terry ratings (gamekit#38)

**Date:** pre-registered 2026-10-07 (before any measured or timing run on the HP)
**Note:** [gamekit#022 — League ratings](https://github.com/guidodinello/gamekit/blob/main/docs/research/022-league-ratings.md); issues [gamekit#38](https://github.com/guidodinello/gamekit/issues/38) (this is its catan first use) and [gamekit#47](https://github.com/guidodinello/gamekit/issues/47) (open, see Deviations). The truco first use is truco-py log 010.

**Status: pre-registered, not run.** Descriptive: no hypothesis test, no decision rule, no pass/fail threshold, and no verdict is scored for note 022.

## What this is for

Note 022 says one anchored Bradley-Terry/Elo rating from a seat-rotated round robin ranks agents better than win rates against one opponent. Truco 010 exercised the 2-seat case. Three things are not yet exercised and this log does: the **N>2 seat reduction** (4 seats, `(a,b,a,b)`), **ties** (a game with no winner), and **the prior at small n**.

## Questions (all descriptive)

1. Does the 2v2 ordering of `long_10m` against `heuristic` agree with the 1v3 benchmark ordering (006/011: 20.72% vs 3 heuristics, i.e. below parity)? Note 022: a disagreement is a finding about the game, not a bug.
2. How do the three 010 runs (`ent_ctrl_3m`, `ent_ramp_3m`, `ent_decay_3m`) order? 010 found no significant difference at 1v3.
3. Does the 006 trajectory order as `long_4m < long_8m < long_10m`?
4. Are there significant 3-cycles in the win matrix?
5. How many games end without a winner (step budget), and how does the prior behave in near-0% cells (anything vs `random`)?

## Design (frozen)

- **Roster, 11 agents, 55 pairings.** RL checkpoints, each identified by a full sha256 in `experiments/league_013_manifest.json`:

  | name | file | sha256_12 | origin |
  |---|---|---|---|
  | `bc_clone` | `catan_bc_clone.zip` | `538059e8dfb8` | 004 BC clone |
  | `bc_ft_2m` | `catan_bc_ft_2000000.zip` | `f91580529cb8` | 004 best / 006 start |
  | `long_4m` | `catan_bc_ft_long_4031616.zip` | `28299abbfb70` | 006 |
  | `long_8m` | `catan_bc_ft_long_8031616.zip` | `52df98c22a7a` | 006 |
  | `long_10m` | `catan_bc_ft_long_10031616.zip` | `e8001d3e81ba` | 006 final; the checkpoint 011/012 search over, **played without search** |
  | `ent_ctrl_3m` | `catan_ent_ctrl_3000000.zip` | `0c818cdb05be` | 010 control |
  | `ent_ramp_3m` | `catan_ent_ramp_3000000.zip` | `304236128704` | 010 ramp |
  | `ent_decay_3m` | `catan_ent_decay_3000000.zip` | `980ff1ae6d8d` | 010 decay |

  Baselines: `heuristic` (**the anchor**), `trading_heuristic`, `random`. No search agents (too slow for the HP).
- **Units.** Full 4-player games. A pairing `(a, b)` fields the lineup `(a, b, a, b)` in **setup order**, rotated by `engine_seed % 4` inside `gamekit.league` (so `abab` and `baba`, each exactly n/2 times). A win by either of A's seats is a win for A; a game with no winner is a **tie**, dropped from the fit and reported per pairing. The lineup is placed as given: `league_factory` does not rotate it again (`benchmark_agent_factory` does).
- **RL seats always reject trade offers** (`server.bots.RLSeatAgent`: policy for every decision except domestic-trade responses, which go to a heuristic that rejects; its accept/reject atoms were never trained, and it never proposes). **Consequence:** in pairings against RL, `trading_heuristic` can only trade with its own copy, so it has fewer trade partners than at a table of traders. Its ratings against RL agents reflect that, not trade skill against a trading opponent. Trade skill on this scale is deferred to a later league once catan #28 has a trading RL agent.
- **Other known biases.** (a) Copies of an agent at one table do not cooperate, except two `trading_heuristic` copies, which can trade with each other; that breaks note 022's premise in those cells, so the 2v2 gap there is not "how `a` does against three `b`s". (b) Note 022's reduction is a Luce-model derivation, not a sourced result; real catan has seat-order effects, and the rotation only ever produces `abab`/`baba`, so `aabb` seatings are never tested. (c) Per-role win-rate intervals in a pairing file treat dependent seat outcomes as independent; the league does not use them.
- **Ratings.** `gamekit.league.summarize_league(anchor="heuristic", anchor_rating=0, prior_draws=1, n_bootstrap=1000, bootstrap_seed=0, alpha=0.05)`. Elo is relative to `heuristic = 0`; its interval is `[0, 0]` by construction. State the roster and the anchor with every number.
- **Seeds.** League seed `20261013`, timing seed `20261014`. `gamekit.league` derives each pairing's engine/driver seed bases by hashing `(seed, a, b)` to 31 bits, so a "fresh range" is *asserted*, not reserved: `tests/test_league_013.py` checks that all 55 engine and 55 driver ranges of length 10,000 for both seeds are disjoint from every range in use (benchmark 1..10000, 008's 90001..100000, 009's 500001 / 1000001 / 2000001 arms, catan#38's whole 3,000,001..4,000,000 block, 011/012's whole 4,000,001..5,000,000 block, fresh_012 20,000,001..20,004,000) and from each other. BC data, training and in-loop eval boards are hash-spread over [0, 2^31) and cannot be included in that test. `experiments.league_013.USED_RANGES` and `engine_ranges()` export the numbers for later logs.
- **Descriptive status.** No claim is tested. The 6-vs-1 comparisons above are reported with the Wilson intervals of the matrix and the bootstrap intervals of the ratings, and nothing is called confirmed.

## How the run is executed

One serial `gamekit.league.run_league` over the 11 agents, whose `play` callback fans each pairing's games out to a persistent 3-worker process pool (nice 19). `run_league` writes every pairing file atomically and writes `league.json` once, with the real anchor and 1000 bootstrap resamples. gamekit#47 (a public `run_pairing` plus an atomic `league.json` write) is still open; this avoids truco 010's workaround (one `run_league` per pairing from a pool, whose `league.json` writes race). The cost is a barrier at the end of each pairing, negligible at n >= 2000. Roster-independence of pairing seeds and `config_hash` means a `--pairs` 2-agent run writes the same file as the full run (a test checks this); `--pairs` is used only for the timing smoke and the cross-machine check.

## n per pairing (rule, committed before any timing)

n is a multiple of 4 (`run_arm` requires it). Measure worker-seconds per game on the HP per pairing class (RL-vs-RL has 4 RL seats and dominates: 28 of 55 pairings), extrapolate to 55 pairings at 3 workers, and apply:

1. **n = 4000** if the extrapolated wall time is <= 48 h;
2. else **n = 2000** if it is <= 48 h;
3. else **stop and ask the owner** (drop agents or lower n); no third tier.

Timing is measured under whatever else runs on the HP (GitHub runners; truco 011 if running), at 1 worker on the free core and scaled to 3, and the measured conditions are disclosed. The measurement uses seed `20261014`, n = 8 per pairing, and reads no win rate.

## Cross-machine gate (before launch)

Pairings `long_10m__vs__heuristic`, `long_4m__vs__random`, `bc_ft_2m__vs__trading_heuristic` at n = 200, seed `20261013`, league name `xcheck`, run on the laptop and on the HP (torch cu128 vs CPU). **Pass** if wins, ties, `config_hash`, the sha256 of the per-game winner list and the sha256 of the per-game record digests are all identical. If it fails, the league is not launched. The xcheck files are deleted afterwards. Earlier evidence: truco 010 (C20 vs M20) and the catan HP determinism check of 2026-09-30 (200/200 games identical, 2228/2228 search decisions identical).

## Environment

To fill at run time: laptop and HP torch / python / numpy / sb3-contrib versions, gamekit commit (the lock moves from `dab51b1` to `05f271e` (docs-only on top of `cf879b3`; `src/` is unchanged since the league landed); the package version string stays 0.3.0, so the commit is recorded by hand), catan commit. Checkpoints were written after the last change to `rl/encoder.py` / `rl/action_space.py` (`e772d1e`, 2026-09-20); `experiments.league_013 check-checkpoints` verifies each checkpoint's observation and action space against the current encoder and plays one game before any league run. Engine rule fixes after a checkpoint's training date can still change what it learned for; that is not detectable here.

## Timing (HP smoke) — to fill

_Not measured yet._

## Result — to fill

_Not run yet._

## Verdict

Descriptive; none scored.

## Deviations / disclosures

- **One serial `run_league` with an inner pool, not truco's one-league-per-pairing pattern** (see above; the instruction was to copy truco's pattern if gamekit#47 was unavailable).
- The lock bump to gamekit `05f271e` adds only `src/gamekit/league/*` relative to the previously locked `dab51b1`.
