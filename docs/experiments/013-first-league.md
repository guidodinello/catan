# First catan league: 11 agents, 4 seats, anchored Bradley-Terry ratings (gamekit#38)

**Date:** pre-registered 2026-10-07 (before any measured or timing run on the HP)
**Note:** [gamekit#022 — League ratings](https://github.com/guidodinello/gamekit/blob/main/docs/research/022-league-ratings.md); issues [gamekit#38](https://github.com/guidodinello/gamekit/issues/38) (this is its catan first use) and [gamekit#47](https://github.com/guidodinello/gamekit/issues/47) (open, see Deviations). The truco first use is truco-py log 010.

**Status: run, 2026-10-08 to 2026-10-10 (results below).** Descriptive: no hypothesis test, no decision rule, no pass/fail threshold, and no verdict is scored for note 022.

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

Pairings `long_10m__vs__heuristic`, `long_4m__vs__random`, `bc_ft_2m__vs__trading_heuristic` at n = 200, **seed `20261014` (the timing seed, not the registered `20261013`)**, league name `xcheck`, run on the laptop and on the HP (torch cu128 vs CPU). **Pass** if wins, ties, `config_hash`, the sha256 of the per-game winner list and the sha256 of the per-game record digests are all identical. If it fails, the league is not launched. The xcheck files are deleted afterwards. **Why not the registered seed:** pairing seeds do not depend on n, so 3 x 200 games at `20261013` would be the first 200 games of three registered pairings, a peek at registered data (truco 011's amendment avoided the same thing). `20261014` is the timing seed: its ranges are asserted disjoint from every used range and from the registered seed's, and no win rate from it enters any result. Earlier evidence: truco 010 (C20 vs M20) and the catan HP determinism check of 2026-09-30 (200/200 games identical, 2228/2228 search decisions identical).

## Environment

To fill at run time: laptop and HP torch / python / numpy / sb3-contrib versions, gamekit commit (the lock moves from `dab51b1` to `05f271e` (docs-only on top of `cf879b3`; `src/` is unchanged since the league landed); the package version string stays 0.3.0, so the commit is recorded by hand), catan commit. Checkpoints were written after the last change to `rl/encoder.py` / `rl/action_space.py` (`e772d1e`, 2026-09-20); `experiments.league_013 check-checkpoints` verifies each checkpoint's observation and action space against the current encoder and plays one game before any league run. Engine rule fixes after a checkpoint's training date can still change what it learned for; that is not detectable here.

## Run log

- **Lock-bump replay (laptop, 2026-10-07, pgrep guard clear):** `experiments.benchmark --mode rl_vs_heuristic --games 200` (`catan_bc_ft_long_10031616`, boards 1..200, 2 workers) at the old lock (gamekit `dab51b1`, catan `100fd60`) and the new (gamekit `05f271e`, catan `433bf1e`). **`by_role` and `by_seat` identical** (heuristic 148/600 seat-wins, rl 52/200). The run overwrote a tracked results file in the main checkout; it was copied out and restored with `git checkout --`.
- **Cross-machine gate: partial pass (owner accepted, 2026-10-08).** Seed `20261014`, n = 200, league name `xcheck`, 3 pairings.
  - Laptop leg: 2 workers, 2026-10-07 (about 18-21 s per RL-baseline pairing). HP leg at `433bf1e`: 1 worker, nice 19, CPU-only torch, **run alongside truco 011's 3 workers**, 2026-10-08 00:55-01:14 UTC (346 s, 368 s, 381 s per pairing); 0 ties in every pairing.
  - **Matched:** the sha256 of the per-game winner list in all three pairings (`long_10m__vs__heuristic` `18daa064...`, `long_4m__vs__random` `b7f3f9f2...`, `bc_ft_2m__vs__trading_heuristic` `be72fcda...`; the laptop values come from its log lines, the first also from its telemetry file), and for `long_10m__vs__heuristic` the record-digest sha256 (`e73c68fb...`). Identical winner lists imply identical win counts; HP wins: heuristic 124 / long_10m 76; long_4m 195 / random 5; bc_ft_2m 58 / trading_heuristic 142.
  - **Not verified (lost):** the laptop's record-digest sha256 for the other two pairings, and its `config_hash` and pairing JSONs for all three. **Why lost:** the laptop leg wrote to the session scratchpad under `/tmp`, and the laptop rebooted overnight, which cleared `/tmp`. The criterion above is therefore not fully met; the owner accepted a partial pass. **Next time:** write cross-check outputs somewhere persistent (e.g. `~/projects/catan-eval-data-laptop/` or another directory outside `/tmp`) and copy the comparison fields into the Run log before any reboot. The xcheck files on the HP are deleted after this PR.
- **Launch:** truco 011 had finished (20:04 UTC) and the HP had no tmux sessions and no experiment processes (load 0.0). `launch.sh 433bf1e catan-013 -- experiments.league_013 run --n 4000 --workers 3 --seed 20261013 (default) --name league` at 2026-10-08 20:40:12 UTC, tmux `catan-013`, log `~/projects/catan-eval-data/logs/catan-013.log`, results `~/projects/catan-eval-data/out/013/league`. First progress lines: `bc_clone__vs__bc_ft_2m` 100/4000 at 20:42:08 and 200/4000 at 20:43:50, i.e. 1.02 s of wall per game, 3.05 worker-s per game at 3 workers. That pairing was the heaviest in the timing smoke (3.38 worker-s/game, the RL-RL maximum), so the launch rate is at or better than the smoke (0.90x of its own timing), with no shared-FPU slowdown visible yet; the ~36 h estimate and the stop rule (stop and ask if the rate is worse than the 1.3x headroom) stand, checked again at later pairings.
- **Rate checks during the run (owner-approved continue each time).** After 3 pairings (about 66 min each) a naive 55 x 66 min extrapolation gave ~60 h; that treats every pairing as RL-vs-RL. Per-class projection from the 3 finished RL-vs-RL pairings and the smoke's class ratios gave 46.5 h (all RL-vs-RL cost ~3.0 worker-s/game, 1.3x the smoke's class mean, because the smoke's n = 8 per pairing was noisy). After the first RL-vs-baseline pairing (37.8 min, 1.70 worker-s/game, as projected) the projection was 47.3 h, 0.7 h under the 48 h line. The stop rule was not triggered and n stayed 4000 for all 55 pairings.
- **Finished:** `exit 0` 2026-10-10 19:07:22 UTC, 46.5 h after launch (sum of pairing wall times 46.3 h); 55 pairing files plus `league.json`. Raw outputs and the log stay on the HP (`~/projects/catan-eval-data/{out/013,logs/catan-013.log}`, mirrored at `~/projects/catan-eval-data-laptop/013/`); this PR commits the 55 pairing files, `league.json` and the per-pairing telemetry sidecars under `experiments/results/league_013*/`.
- **Timing by pairing class (actual, 3 workers, nice 19, 4000 games per pairing, no ties):**

  | class | pairings | wall per pairing, mean (min-max) | worker-s / game, mean (min-max) | smoke (n = 8) worker-s / game |
  |---|---|---|---|---|
  | RL-RL | 28 | 63.9 min (59.2-69.4) | 2.87 (2.66-3.12) | 2.30 |
  | RL-baseline | 24 | 39.6 min (35.7-45.6) | 1.77 (1.60-2.04) | 1.30 |
  | baseline-baseline | 3 | 13.8 min (10.9-16.6) | 0.61 (0.48-0.74) | 0.50 |

  Actual / projected-from-smoke: 46.5 h vs 35.9 h, i.e. 1.29x (the headroom was 1.3x). Peak RSS 479 MB per worker. Load average sat at 3.0 with the three workers at ~100% of a core each; no GitHub runner job was seen at the checks made.
- **Concurrent load (disclosed).** Per the orchestrator, truco experiment 013 ran on the HP's fourth core from 2026-10-09 22:46 UTC (1 worker, nice 19); at the end of the run a `truco-050` tmux session was also present (created 2026-10-10 17:24 UTC; I did not characterize it). Wall times and worker-s/game may have been perturbed (shared-FPU modules), but results cannot be: games are deterministic functions of the seeds. No perturbation is visible in the timings: RL-RL pairings before vs after 22:46 UTC on Oct 9 averaged 64.1 min (18 pairings) vs 63.5 min (10), RL-baseline 39.5 (9) vs 39.6 (15).
- **Stamps re-derived from the files (2026-10-10):** 55 pairing files, 55 distinct `config_hash`es, each with `n_games` 4000, wins + ties = 4000 (0 ties in all 220,000 games), every seat occupied 2000 times by each agent (8000 per agent per pairing), and `seed` 20261013 in every `league_config`. `experiments.league_013 summarize` on a copy of the pairing files reproduced `league.json` exactly (agents, ratings and CIs, matrix, ties, cycles). The 55 telemetry sidecars agree with the pairing files on n and ties.

## Timing (HP smoke) and the chosen n

Measured on the HP 2026-10-08 (commit `764036b`, after the rule above was committed), seed `20261014`, n = 8 per pairing, all 55 pairings, **1 worker**, nice 19, `OMP_NUM_THREADS=1`, CPU-only torch. No win rate is read. Worker time is per-game CPU time (`time.process_time`) inside the worker.

| pairing class | pairings | worker-s / game (mean, min-max) | peak RSS | ties |
|---|---|---|---|---|
| baseline-baseline | 3 | 0.50 (0.34-0.61) | 479 MB | 0/24 |
| RL-baseline | 24 | 1.30 (0.92-2.05) | 479 MB | 0/192 |
| RL-RL | 28 | 2.30 (1.58-3.38) | 479 MB | 0/224 |

Sum over the 55 pairings: 97.0 worker-s per unit of n. At 3 workers (assuming linear scaling): **n = 4000 -> 35.9 h, n = 2000 -> 18.0 h.** The rule gives **n = 4000** (<= 48 h). Peak RSS 479 MB per worker (peak RSS is per process, measured in the worker; 3 workers about 1.5 GB of the 5 GB free).

Caveats, disclosed: the HP was **idle** during the smoke (load average about 0.5; no GitHub runner job and truco 011 was not running yet), so this is not measured under 011's load. The estimate assumes linear scaling to 3 workers, but the HP's 4 cores are 2 shared-FPU modules, so 3 workers will likely be slower than 3x (the 48 h line leaves about 1.3x of headroom); n = 8 per pairing is a small sample of game lengths (the per-class min-max spread above is the per-pairing noise). If the launch-time wall rate is worse than this by more than the headroom, the run is stopped and the owner is asked, not silently shortened. The launch waits until truco 011 finishes (owner coordinates).

## Result

Roster of 11, 55 pairings, n = 4000 per pairing (220,000 games), seed 20261013, 4-seat `(a,b,a,b)` reduction, anchor `heuristic = 0`, `prior_draws = 1`, 1000 bootstrap resamples (`bootstrap_seed = 0`), 95% intervals. **0 ties** in every pairing (no step-budget hits). Files: `experiments/results/league_013/` (`league.json` plus 55 pairing files), `experiments/results/league_013_telemetry/`.

**Ratings** (Elo relative to `heuristic = 0`; RL seats always reject trade offers, see Design):

| rank | agent | Elo | 95% bootstrap CI |
|---|---|---|---|
| 1 | `trading_heuristic` | +16.2 | [+10.6, +21.2] |
| 2 | `heuristic` (anchor) | 0 | [0, 0] by construction |
| 3 | `long_10m` | -62.9 | [-67.6, -57.8] |
| 4 | `long_8m` | -69.2 | [-74.4, -64.2] |
| 5 | `ent_decay_3m` | -89.9 | [-95.0, -84.6] |
| 6 | `ent_ctrl_3m` | -116.1 | [-121.0, -111.2] |
| 7 | `long_4m` | -123.4 | [-128.9, -118.4] |
| 8 | `ent_ramp_3m` | -132.3 | [-137.6, -127.2] |
| 9 | `bc_clone` | -161.1 | [-166.4, -156.1] |
| 10 | `bc_ft_2m` | -178.8 | [-183.8, -173.6] |
| 11 | `random` | -687.3 | [-696.2, -676.6] |

**Win matrix** (percent of decisive games the row agent wins in the 2v2 pairing, n = 4000 each; the per-cell Wilson 95% half-width is about 1.5 points at 50%; abbreviations: `trading`, `heur`, `L10m` = long_10m, `L8m`, `L4m`, `edecay` / `ectrl` / `eramp` = the 010 runs, `bcc` = bc_clone, `bcft` = bc_ft_2m, `rand`):

| row beats col (%) | trading | heur | L10m | L8m | edecay | ectrl | L4m | eramp | bcc | bcft | rand |
|---|---|---|---|---|---|---|---|---|---|---|---|
| trading | - | 52.0 | 60.3 | 61.5 | 66.6 | 69.7 | 67.3 | 70.1 | 75.3 | 72.2 | 99.95 |
| heur | 48.0 | - | 59.8 | 60.6 | 62.7 | 68.2 | 65.4 | 67.6 | 73.4 | 70.0 | 98.3 |
| L10m | 39.7 | 40.2 | - | 66.7 | 59.6 | 47.3 | 59.4 | 48.4 | 62.0 | 66.9 | 97.8 |
| L8m | 38.5 | 39.5 | 33.3 | - | 59.5 | 52.8 | 65.0 | 63.8 | 61.1 | 67.7 | 97.9 |
| edecay | 33.4 | 37.2 | 40.4 | 40.5 | - | 63.5 | 57.4 | 60.1 | 61.1 | 59.5 | 96.8 |
| ectrl | 30.3 | 31.8 | 52.6 | 47.2 | 36.4 | - | 52.9 | 51.0 | 52.4 | 61.3 | 96.9 |
| L4m | 32.7 | 34.6 | 40.6 | 35.0 | 42.6 | 47.1 | - | 54.8 | 57.6 | 61.9 | 95.5 |
| eramp | 29.9 | 32.4 | 51.6 | 36.1 | 39.9 | 48.9 | 45.2 | - | 52.4 | 57.8 | 95.5 |
| bcc | 24.6 | 26.7 | 38.0 | 38.9 | 38.9 | 47.6 | 42.4 | 47.5 | - | 51.6 | 93.7 |
| bcft | 27.9 | 30.0 | 33.1 | 32.3 | 40.5 | 38.7 | 38.1 | 42.2 | 48.4 | - | 94.8 |
| rand | 0.1 | 1.7 | 2.2 | 2.1 | 3.2 | 3.1 | 4.5 | 4.5 | 6.3 | 5.2 | - |

**Answers to the pre-registered questions (descriptive; nothing is called confirmed):**

1. **`long_10m` vs `heuristic`:** `heuristic` wins 59.8% of 2v2 games (long_10m 40.2%, Wilson [38.7, 41.8]); long_10m rates -62.9 [-67.6, -57.8]. This agrees in direction with the 1v3 benchmark (20.72% vs 25% parity: below parity). No RL checkpoint rates above `heuristic`. The two numbers are not on a common scale (1v3 win rate vs 2v2 Elo), so only the sign of the comparison is stated.
2. **010 runs:** `ent_decay_3m` (-89.9) > `ent_ctrl_3m` (-116.1) > `ent_ramp_3m` (-132.3), with non-overlapping bootstrap intervals; direct games decay over ctrl 63.5%, decay over ramp 60.1%, ctrl over ramp 51.0%. 010 found no significant difference at 1v3 with its sample; here decay is the best of the three, ramp the worst, but the ctrl-vs-ramp direct game is 51.0% (inside noise).
3. **006 trajectory:** `long_4m` (-123.4) < `long_8m` (-69.2) < `long_10m` (-62.9) holds in the ratings. The 8m and 10m intervals overlap (-74.4..-64.2 vs -67.6..-57.8), so the 8m-to-10m step is not resolved by the ratings, although `long_10m` beats `long_8m` head-to-head 66.7% (Wilson [65.2, 68.1]).
4. **3-cycles:** gamekit reports 5 significant 3-cycles, all through `long_10m`: ent_ctrl > long_10m > ent_decay > ent_ctrl; ent_ctrl > long_10m > long_8m > ent_ctrl; ent_decay > ent_ramp > long_10m > ent_decay; ent_ramp > long_10m > long_4m > ent_ramp; ent_ramp > long_10m > long_8m > ent_ramp. The largest Bradley-Terry residuals are the same cells: long_10m vs long_8m (observed 66.7%, fitted 50.9%, residual +0.158), long_10m vs ent_ramp (48.4% vs 59.9% fitted, -0.115), long_10m vs ent_ctrl (47.3% vs 57.6%, -0.102), ent_ctrl vs ent_decay (36.4% vs 46.2%, -0.098). `long_10m` beats the 006 family (4m, 8m) and `bc_*` by more than the single-scale fit predicts, and does worse against the 010 runs. Note 022's single-rating model does not capture this; one scalar per agent is a poor summary of the RL block.
5. **Ties and the prior:** 0 of 220,000 games ended without a winner. Near-0% cells: `random` wins between 2 and 252 of 4000 games in its 10 pairings (2/4000 against `trading_heuristic`, 69/4000 against `heuristic`). `random` rates -687.3 with an interval of width about 20, similar to the other agents; with `prior_draws = 1` and 4000 games per pairing the prior is negligible and nothing diverged. The ratings do not tell us whether the prior would matter at the smaller n; this league did not exercise that.

**Other observations (not pre-registered).** `trading_heuristic` rates +16.2 [+10.6, +21.2] and beats `heuristic` head-to-head 52.0% (Wilson roughly [50.5, 53.5]), but this includes the disclosed caveat: against RL agents it can trade only with its own copy (RL seats reject), and two `trading_heuristic` copies can trade with each other, which breaks note 022's no-cooperation premise in the pairings that involve it. So the +16 is not "trade skill against a trading opponent". `bc_ft_2m`, the 004 "best" and the 006 start, rates below its own start `bc_clone` (-178.8 vs -161.1; head-to-head `bc_clone` 51.6%): different seeds and the 4-seat game, not a finding about training.

## Verdict

Descriptive; none scored. No hypothesis was tested and no pass/fail threshold existed. The results above stand as the 4-seat, 4000-game first use of `gamekit.league`: the N>2 seat reduction ran end to end (balanced `abab`/`baba` occupancy in all 55 pairings), ties are exercised as a code path only (0 occurred), and the prior at small n was not exercised (n = 4000 throughout). The cross-machine gate was a partial pass (see Run log).

## Deviations / disclosures

- **One serial `run_league` with an inner pool, not truco's one-league-per-pairing pattern** (see above; the instruction was to copy truco's pattern if gamekit#47 was unavailable).
- The lock bump to gamekit `05f271e` adds only `src/gamekit/league/*` relative to the previously locked `dab51b1`.
- **Ties and the small-n prior were not exercised in the run itself.** 0 ties in 220,000 games and n = 4000 per pairing; the tie path is covered only by the CI test (all-ties pairing writes its file; the fit then refuses it).
- **Cross-machine gate: partial pass** (winner-list hashes matched for all three pairings; the laptop's record digests for two pairings and `config_hash` / pairing files for all three were lost to a reboot, see Run log). The league's own stamps were re-derived afterwards.
- **Timing was 1.29x the smoke estimate (46.5 h vs 35.9 h)** because the smoke's per-class means from n = 8 samples were noisy and 3 workers share 2 FPU modules; within the 48 h rule and its 1.3x headroom. The run overlapped truco experiment 013 (and later a `truco-050` session) on the HP.
