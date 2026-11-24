# Graph agents on the first league's Elo ladder: 4 graph checkpoints added to the 013 league (issue #125)

**Date:** pre-registered 2026-11-24 (before any measured run, replay, cross-check or timing smoke on the HP)
**Note:** [gamekit#022 — League ratings](https://github.com/guidodinello/gamekit/blob/main/docs/research/022-league-ratings.md). Issue [#125](https://github.com/guidodinello/catan/issues/125). Extends [013](013-first-league.md); the graph checkpoints come from [014](014-graph-encoder.md) and [016](016-graph-longer-training.md).

**Status: pre-registered, not run.** Descriptive: no hypothesis test, no decision rule, no pass/fail threshold on any rating. The only pass/fail items are the two pre-launch gates below, and they test the harness, not an agent.

## What this is for

013 put 8 flat-MLP checkpoints and 3 baselines on one anchored Bradley-Terry ladder (`heuristic = 0`). 014 and 016 then produced the board-aware graph agents, which win 54-70% against 3 `HeuristicAgent`s (1v3, [016](016-graph-longer-training.md)) where the best flat agent wins 20.7%. Those numbers are not on the 2v2 scale of the ladder, so no graph agent has a rating next to `long_10m`, `heuristic` or `trading_heuristic`. 017 adds four graph checkpoints so that all our trained agents sit on one anchored scale, with the same seat reduction, seeds and fit as 013.

## Questions (all descriptive)

1. Where do the graph agents rate relative to `heuristic` (the anchor) and `trading_heuristic`? In 013 no RL checkpoint rated above `heuristic`.
2. Within each fine-tune seed, does 3M order below 12M?
3. Do `graph_s3_12m` and `graph_s4_12m` separate? Their 1v3 intervals overlap in 016 (67.28% and 69.53%).
4. How do the graph agents do head-to-head against `long_10m`, the best flat agent?
5. Does the single-scalar Bradley-Terry fit hold with graph agents in, or do significant 3-cycles through them appear (013 found 5, all through `long_10m`)?
6. How far do 013's ratings move when the graph agents join the fit?

## Step 0 (before any design): the league path plays graph checkpoints

Question: can `server.bots.RLSeatAgent` -> `agents.rl_agent.RLAgent` (the league path) load and play graph-policy checkpoints? **Yes, with no code change.**

- Source: `RLAgent` loads through `agents.rl_agent._load_model` = `MaskablePPO.load`, which unpickles the `GraphMaskablePolicy` stored in the zip; inference goes through `model.predict`, and the observation encoder and the atom action space are the same for both architectures (`rl/graph_policy.py`: "the observation, action space and BC dataset are unchanged").
- Local smoke (laptop, 2026-11-24, `200dc6e`, one process, nice 19, `OMP_NUM_THREADS=1`, **no files written, no win rate read**): `_load_model` returned `GraphMaskablePolicy` for the graph zips and `MaskableActorCriticPolicy` for `long_10m`, all with observation dimension `OBS_DIM` and `N_ATOMS` actions; `RLSeatAgent` built on `catan016_graph_s4_12047424.zip` loaded; `league_013._play_one` then played full 4-seat games (4 per pairing, lineups alternated; ties were not tracked, only that each game returned a record).
- That smoke used 013's timing seed 20261014 and ad-hoc agent names, so its boards are not 017's (017's timing seed is 20261015, see Seeds). Worker CPU seconds per game (`process_time`, n = 4 per pairing, so noisy):

  | pairing | worker-s / game (mean, min-max) |
  |---|---|
  | graph s4 12M vs `heuristic` | 1.02 (0.58-1.76) |
  | `long_10m` vs `heuristic` | 0.38 (0.22-0.65) |
  | graph s4 12M vs graph s3 3M | 2.23 (1.35-3.61) |
  | `long_10m` vs graph s4 12M | 0.89 (0.57-1.53) |
  | `heuristic` vs `random` | 0.16 (0.06-0.26) |

- Also checked: `experiments.league_017 check-checkpoints` loads each of the four manifest zips, checks the spaces against the current encoder and plays one game with it (laptop, 2026-11-24, all four `ok`).
- **The reuse check below (Design) was also run read-only:** gamekit 0.5 reproduces the committed 013 file's seed bases and `config_hash`.

## Design (frozen)

**Extend, don't replay.** gamekit derives a pairing's seeds and `config_hash` from `(seed, a, b, num_seats, n)` only, never from the roster. So 017 keeps 013's registered seed `20261013`, n per pairing 4000 for 013's files, and 013's agent names, and **the 55 committed files under `experiments/results/league_013/` are used as they are**. Checked 2026-11-24 with gamekit 0.5.0: `pairing_seeds(20261013, "bc_clone", "bc_ft_2m")` = (1975542421, 984492537) and `config_hash` `d8027ec9c1f2`, identical to the committed file. `experiments.league_017.reuse_013` re-checks all 55 through `gamekit.league.run_pairing` (a finished file with a different hash raises; a missing file raises), and `tests/test_league_017.py` runs it in CI. Only the new pairings are played.

**New agents (4)**, full sha256 in `experiments/league_017_manifest.json`, checkpoints identified by file basename in one flat directory (as on the HP, `~/projects/catan-eval-data/checkpoints/`):

| name | file | sha256_12 | origin |
|---|---|---|---|
| `graph_s3_3m` | `catan014_ft_graph_s3_3000000.zip` | `6cb96ab7d856` | 014 fine-tune seed 3, the fixed 3M point |
| `graph_s4_3m` | `catan014_ft_graph_s4_3000000.zip` | `ed7c57e28f6e` | 014 fine-tune seed 4, the fixed 3M point |
| `graph_s3_12m` | `catan016_graph_s3_c_12047424.zip` | `f24c64b8ca28` | 016 "+9M" point, seed 3 (real 12,070,912 steps) |
| `graph_s4_12m` | `catan016_graph_s4_12047424.zip` | `6b7e921955d8` | 016 "+9M" point, seed 4 (real 12,189,696 steps); the current best in the README |

The names match the sha256 prefixes recorded in 014/016 Amendments. All four are played **without search**, deterministic, with `RLSeatAgent` (RL seats always reject trade offers), exactly as 013's RL agents.

**Dropped agents, and what that costs.** 016's +3M and +6M points (4 more graph agents) and 014's flat-MLP arm `catan014_ft_mlp_s3/s4_3000000` (2 more, 16% against 3 heuristics, the same family as the 013 flat agents) are not rated. Adding all 6 with full rows would cost about 137 h at n = 2000 on 3 workers (estimate, same method as below), which is not a reasonable run on a shared HP. The cost of dropping them: the ladder has no position for the intermediate training points (016 shows a plateau from 6M at 65-69% against 3 heuristics, so little is lost for the question) or for the 014 flat arm. A later league can add them the same way, because pairing seeds do not depend on the roster.

**Pairings: 26 bridge pairings, not full rows.**

- Each graph agent plays the five agents at the top of 013's ladder: `ent_decay_3m` (-89.9), `heuristic` (0, the anchor), `long_10m` (-62.9), `long_8m` (-69.2), `trading_heuristic` (+16.2). 20 pairings. The rungs closest to where the graph agents are expected, so each game carries the most information about their rating.
- The four graph agents play each other in a full round robin: 6 pairings.
- `summarize_league` needs a connected win graph, not a complete one (`gamekit.league.ratings._check_connected`), and the bridge graph is connected (tested). The ladder is the same single anchored fit, over 55 + 26 = 81 pairing files.
- **Trade-off against full rows** (each graph agent against all 11, 50 pairings): there are no graph cells against `random` or the bottom of the flat ladder (`bc_clone`, `bc_ft_2m`, `long_4m`, `ent_ctrl_3m`, `ent_ramp_3m`). Those cells would be near 90-100% and carry little information about a rating (013's `random` cells are the near-0% ones, and the prior matters most there). The graph agents' ratings are therefore determined through the top of the flat ladder, so an error in those rungs moves them; the bootstrap interval includes that resampling but not a model error. Full rows would cost about 1.7x more.

Everything else is 013's, unchanged: 4 seats, lineup `(a,b,a,b)` in setup order rotated by `engine_seed % 4` (`abab` and `baba`, each exactly n/2 times); a win by either of A's seats is a win for A; a game with no winner is a tie, dropped from the fit and reported per pairing. `gamekit.league.summarize_league(anchor="heuristic", anchor_rating=0, prior_draws=1, n_bootstrap=1000, bootstrap_seed=0, alpha=0.05)`. State the roster and the anchor with every number. 013's disclosed biases carry over (RL seats reject trades, so `trading_heuristic` can only trade with its own copy; copies at one table do not cooperate; `aabb` seatings are never tested; the reduction is a Luce-model derivation).

**Joint refit.** Adding agents changes every Elo value, 013's included. The 017 ladder **replaces** 013's; a 013 number and a 017 number are not compared one for one. Both tables are reported side by side, and Question 6 reports the shifts. 013's files have n = 4000 and 017's follow the n rule below (probably 2000); the fit works on counts, so the graph cells weigh less than 013's and their ratings carry wider intervals. `summarize` writes `league.json` into the 017 results dir from the 55 + 26 files; it refuses to fit unless all 26 registered pairings are present with one `n` at seed 20261013, and unless every 013 file passes `reuse_013`.

**Seeds.**

- **Registered seed:** `20261013`, 013's. Disjoint from every used range and from 013's own ranges (the 26 new engine and driver ranges, length 10,000, are asserted disjoint from `league_013.USED_RANGES`, from 013's 55 pairing ranges at both of 013's seeds, and from each other, by `tests/test_league_017.py`). The ranges are hash-derived, so as in 013 disjointness is asserted, not reserved; BC data, training and in-loop boards cannot be included in that test.
- **Timing and cross-check seed: `20261015`, not 013's `20261014`.** At `20261014` one new pairing's engine range (`graph_s4_12m__vs__graph_s4_3m`, from 4,403,394) falls inside 011/012's reserved smoke block 4,000,001..5,000,000, which the test caught. The rule, fixed before any run: the smallest seed above 013's timing seed that passes the full disjointness test. That is `20261015`. No win rate from it enters any result.
- **No peeking:** pairing seeds do not depend on n, so any game at `20261013` is one of the first games of a registered pairing. The cross-check therefore uses the timing seed. The replay gate plays only 013's already published games.

## n per pairing, wall-time budget, workers (rule, committed before any HP timing)

n is a multiple of 4. The 26 new pairings, with HP worker-s per game scaled from the Step 0 laptop numbers by the one ratio both machines have for the same pairing (`heuristic__vs__long_10m`: HP 1.63 in 013, laptop 0.38, **4.3x**); **estimates, to be replaced by the HP smoke**:

| class | pairings | est. HP worker-s / game |
|---|---|---|
| graph vs baseline (`heuristic`, `trading_heuristic`) | 8 | 4.4 |
| graph vs flat RL (`long_10m`, `long_8m`, `ent_decay_3m`) | 12 | 3.8 |
| graph vs graph | 6 | 9.6 |

Sum over the 26 pairings: about **138 worker-s per unit of n**. At 3 workers and linear scaling:

| n | 3 workers | 2 workers |
|---|---|---|
| 4000 | about 51 h | about 77 h |
| 2000 | about 26 h | about 38 h |

Not allowed to move the estimate silently: `trading_heuristic` pairings are probably heavier than `heuristic` ones (the trading code runs), graph inference on the HP's no-AVX2 torch kernels may scale worse than the 4.3x of the flat model, and 013's own run came out 1.29x its smoke estimate (3 workers on 2 shared-FPU modules).

**Rule (same shape as 013):**

1. **n = 4000** if the extrapolated wall time at the chosen worker count is <= 48 h;
2. else **n = 2000** if it is <= 48 h;
3. else **stop and ask the owner** (drop agents or lower n); no third tier.

The measurement is a **timing smoke on the HP that needs the owner's go**: seed `20261015`, n = 8 per pairing, all 26 pairings, 1 worker, nice 19, `OMP_NUM_THREADS=1`, CPU-only torch, reads no win rate; per-pairing worker time is `time.process_time` inside the worker, scaled to the worker count. The estimate above predicts n = 2000; at n = 2000 each pairing still has 1000 `abab` and 1000 `baba` games and a 95% half-width near 2.2 points at 50%.

**Workers: 3 (or 2), chosen from the HP's load before launch, never from results.** HP, read-only over ssh 2026-11-24: AMD A10-5745M, 4 cores (2 shared-FPU modules, 1 thread each), 6.6 GB RAM of which about 3.9 GB available, 108 GB free disk, torch capability DEFAULT (no AVX2). It was **loaded**: load average 5-6 on 4 cores, from GitHub Actions runners (`actions-runner-*`, `runners/hp-2`) and a `fitted` worker stack, no tmux session and no experiment process of ours. RAM is not the limit: one process with all 13 candidate models loaded peaked at 516 MB RSS (with the 8 flat models only, 403 MB; `_load_model` is an unbounded cache, so each persistent worker accumulates the models it meets), so 3 workers need about 1.5 GB. **CPU is the limit**: if the load is still above 4 at launch, use 2 workers (about 38 h at n = 2000, still under the line). Disclose the load and worker count with the run.

**Launch-time rate check:** after the first pairing of each class, project the total from the finished pairings. If the projection exceeds the chosen tier's 48 h by more than 1.3x headroom, **stop and ask**; do not shorten silently.

## Gates before launch (all need the owner's go; none runs in the PR that registers 017)

**Gate 1, replay (laptop).** The lock moved since 013 (`433bf1e`: gamekit 0.3 -> 0.5.0 and several dependency bumps), so before the committed 013 files are trusted as 017 inputs, replay one full RL-vs-baseline pairing of 013 at its registered seed: `heuristic__vs__long_10m`, `20261013`, n = 4000, `python -m experiments.league_013 run --n 4000 --seed 20261013 --workers 2 --pairs heuristic__vs__long_10m --ckpt-root <flat dir with all 8 013 zips> --results-dir <persistent dir> --name replay`. **Pass** if the pairing file's `config_hash`, `by_role` and `by_seat` equal the committed 013 file's, and the new telemetry sidecar's `winners_sha256` (`b95c852e...`) and `digests_sha256` (`7ccb7cab...`) equal 013's (`experiments/results/league_013_telemetry/heuristic__vs__long_10m.json`). These are already-published games, not a peek. About 13 min at 2 workers (013 mean 0.38 worker-s/game on the laptop). **Fail** -> the 013 files are not valid 017 inputs; a full replay does not fit the HP budget, so stop and ask. (`league_013 run` verifies all 8 of 013's checkpoint hashes; the laptop does not hold the three `ent_*` zips, so copy them from the HP's `checkpoints/` first and let `verify_manifest` check them.)

**Gate 2, cross-machine (laptop vs HP), now including graph models.** 013's gate covered flat models only. Graph message passing (masked mean aggregation) on the HP's no-AVX2 kernels is the new determinism risk. Pairings `graph_s4_12m__vs__heuristic`, `graph_s3_3m__vs__graph_s4_12m`, `graph_s3_12m__vs__long_10m` at n = 200, **seed `20261015`**, league name `xcheck`:

`python -m experiments.league_017 run --n 200 --seed 20261015 --workers 2 --pairs graph_s4_12m__vs__heuristic,graph_s3_3m__vs__graph_s4_12m,graph_s3_12m__vs__long_10m --ckpt-root <flat dir> --results-dir <persistent dir> --name xcheck`

- **Pass** if wins, ties, `config_hash`, the sha256 of the per-game winner list and the sha256 of the per-game record digests are all identical on both machines in all three pairings. **If it fails, the league is not launched.**
- **Outputs go to persistent directories, never `/tmp`** (013 lost its laptop leg's outputs to a reboot): laptop `~/projects/catan-eval-data-laptop/017/`, HP `~/projects/catan-eval-data/out/017/`. The comparison fields (all five, per pairing, both machines) are copied into this log's Run log **before** any reboot or cleanup.
- Before it, copy the four graph zips to the HP's `checkpoints/` and let `verify_manifest` check the sha256 (the HP holds only the 8 flat zips today); `python -m experiments.league_017 check-checkpoints --ckpt-root <HP checkpoints>` on the HP before the smoke.

## How the run is executed (after the gates and the owner's go)

Same pattern as 013, serial over pairings with one persistent process pool, via the HP's launcher:

```
~/projects/catan-eval-data/launch.sh <commit> catan-017 -- experiments.league_017 run --n <n> --workers <3|2> \
    --ckpt-root ~/projects/catan-eval-data/checkpoints --results-dir ~/projects/catan-eval-data/out/017 --name league
```

`launch.sh` runs under nice 19 in a detached tmux session named `catan-017`, logging to `~/projects/catan-eval-data/logs/catan-017.log` (persistent). Long-job rules: check `pgrep -f '[l]eague_01'` and `tmux ls` first and refuse a duplicate; a dead wrapper is not a dead job; stop only with the job's `tmux send-keys -t catan-017 C-c` / SIGTERM, never a broad `pkill -f`; results and logs stay in `catan-eval-data`, not `/tmp`. Resume is per pairing: a finished file with a matching `config_hash` is skipped, so a stop loses at most the current pairing. When it ends: `python -m experiments.league_017 summarize --results-dir <dir> --name league --dir-013 experiments/results/league_013`. The 26 pairing files, their telemetry sidecars and `league.json` are committed with the results PR.

**Stop rules during the run.** Stop and ask if (a) a pairing has more than 1% of games without a winner (`MAX_TIE_SHARE`, enforced after each pairing played in this invocation by `league_017.run`; a finished file skipped on resume is not judged again, so the owner's "continue" sticks; 013 had 0 in 220,000; the same rule applies to the n = 8 smoke and the n = 200 cross-check, where a single tie already exceeds 1% and stops the run, which is intended), (b) the projected time breaks the rate check above, (c) `verify_manifest` or `reuse_013` fails.

## Environment

To fill at run time: laptop and HP torch / python / numpy / sb3-contrib / gamekit versions, catan commit, HP load average and worker count, concurrent jobs. Local figures so far (laptop, 2026-11-24): the lock-pinned venv, gamekit 0.5.0.

## Run log

Not run. Nothing in this document is a measurement; the Step 0 numbers are timing only.

## Result

Not run.

## Deviations / disclosures

- **Timing seed differs from 013's** (`20261015` vs `20261014`) because of a real overlap with 011/012's smoke block; see Seeds.
- **Bridge pairings instead of full rows** and **four of ten candidate agents** are cost choices with their trade-offs stated above, not findings.
- **The cost numbers are estimates** from n = 4 laptop games times one HP/laptop ratio (4.3x), itself taken from a 4-game laptop sample for `heuristic__vs__long_10m` against 013's 4000-game HP mean; the HP smoke replaces them.
- The Step 0 smoke and the RSS measurement ran on the laptop before this document existed, outside any pre-registered pairing, with 013's timing seed; neither read a win rate.
