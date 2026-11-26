# Graph agents on the first league's Elo ladder: 4 graph checkpoints added to the 013 league (issue #125)

**Date:** pre-registered 2026-11-24 (before any measured run, replay, cross-check or timing smoke on the HP)
**Note:** [gamekit#022 — League ratings](https://github.com/guidodinello/gamekit/blob/main/docs/research/022-league-ratings.md). Issue [#125](https://github.com/guidodinello/catan/issues/125). Extends [013](013-first-league.md); the graph checkpoints come from [014](014-graph-encoder.md) and [016](016-graph-longer-training.md).

**Status: run 2026-11-24 to 2026-11-26 (UTC); see Run log and Result.** Everything from "What this is for" through "How the run is executed" is the pre-registered text, unchanged. Descriptive: no hypothesis test, no decision rule, no pass/fail threshold on any rating. The only pass/fail items are the two pre-launch gates below, and they test the harness, not an agent.

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

| | HP (league, HP cross-check leg, timing smoke) | laptop (replay gate, laptop cross-check leg) |
|---|---|---|
| catan commit | `7372ad7` | `7372ad7` (separate worktrees) |
| python | 3.14.4 | 3.14.5 |
| torch | 2.14.1+cpu (capability DEFAULT) | 2.14.0+cpu (capability AVX2) |
| numpy | 2.5.4 | 2.5.3 |
| sb3-contrib / stable-baselines3 | 2.9.2 / 2.9.2 | 2.9.0 / 2.9.0 |
| gamekit | 0.5.0 | 0.5.0 |

Versions were read on 2026-11-26, after the run. The HP column matches `uv.lock` (`launch.sh` runs `uv sync --frozen --extra rl-train`). **The laptop column does not:** both laptop gates ran with the main checkout's `.venv` (the launch command and the cross-check script name its interpreter), whose torch, numpy and sb3 are older patch releases than the lock; their install dates are from September, before the gates. gamekit was at the locked version. See Deviations.

`summarize` ran on the laptop at `d4117ca` (the commit stamped in `league.json`); the 26 pairing files are stamped `7372ad7`. `git diff 7372ad7 d4117ca -- uv.lock pyproject.toml experiments/ src/ tests/` is empty.

## Run log

- **Gate 1, replay: PASS** (laptop, 2026-11-24, `7372ad7`, `heuristic__vs__long_10m`, seed 20261013, n = 4000, **4 workers**, nice 10, 252 s). `config_hash`, `by_role` and `by_seat` equal the committed 013 file's (re-checked 2026-11-26), and both telemetry hashes equal 013's: `winners_sha256` `b95c852e511aae3e7ffcc34173c8ffc5d186d726967c17d5b5fc505a4433d210`, `digests_sha256` `7ccb7cab29f537f68703ec63abe0f7dc4383918ad270f31db587c75fc4082ecb`.
- **Gate 2, cross-machine: PASS** (seed 20261015, n = 200, league name `xcheck`, `7372ad7`; laptop 4 workers at nice 10, HP 1 worker; HP leg 21:22-22:27 UTC, 2026-11-24). All five fields are identical on both machines in all three pairings, and every pairing had 0 ties. The rows below are both machines' values. Wins are from the timing seed and enter no result.

  - `graph_s3_12m__vs__long_10m` (wall: laptop 27 s, HP 1061 s)
    - wins: graph_s3_12m 184, long_10m 16; ties 0; `config_hash` `3a7cf56b39c3`
    - `winners_sha256` `aed4f3018c783e8f5bbc685fc91969093ae1ba6d6a865ea72c0d8c969997abb9`
    - `digests_sha256` `e44e25f381c5e5aefaa2c7892c7801a1cfa295022286860563e324b2fa0bdfee`
  - `graph_s3_3m__vs__graph_s4_12m` (wall: laptop 44 s, HP 1819 s)
    - wins: graph_s4_12m 143, graph_s3_3m 57; ties 0; `config_hash` `7fc15934f624`
    - `winners_sha256` `6571ebf438918c5bd25b4c2979b8bcb8ddf8aae242482c42012c1fa0d8d03c96`
    - `digests_sha256` `e305fead81b2d3fb87085e1e604ae1581b0a286f5656979138839210e53c19da`
  - `graph_s4_12m__vs__heuristic` (wall: laptop 26 s, HP 1004 s)
    - wins: heuristic 25, graph_s4_12m 175; ties 0; `config_hash` `f9c27b6cd856`
    - `winners_sha256` `669765f86ad3f16e5554655bcdb84fc014521d5ee63441476e5e2f299fce73f6`
    - `digests_sha256` `e9e4c5f4da79a94edc62c1e33a339dd02cfba075e2761fed6e6d19da5822ad3d`
- **Timing smoke:** see below. It gave **n = 2000**.
- **Launch:** 2026-11-24 23:15:54 UTC on the HP, `launch.sh 7372ad7 catan-017 -- experiments.league_017 run --n 2000 --workers 2 --ckpt-root ~/projects/catan-eval-data/checkpoints --results-dir ~/projects/catan-eval-data/out/017 --name league` (seed 20261013, the default), tmux `catan-017`, nice 19, `OMP_NUM_THREADS=1`. **2 workers**, from the registered load clause: the HP load average was 6.1-7.7 at launch (above 4), with 3 GitHub runner workers active.
- **Rate checks** (telemetry only, no win rate read). The projection is the finished pairings' wall time plus each remaining pairing's smoke worker time times its class's actual/smoke ratio, at 2 workers. Stop-and-ask limit: 62.4 h.
  - 2026-11-25 about 15:32 UTC, 9 of 26 done, all three classes represented, 0 ties: 1.16x the smoke, **projected 43.7 h**. Load average at that check: 5.96 (1 min). The third class's first pairing had finished at 13:23 UTC, about 2 h earlier.
  - 2026-11-26 about 01:53 UTC, 14 of 26 done, 0 ties: 1.19x, projected 47.6 h.
  - Neither check triggered the stop rule.
- **Finished:** `exit 0` 2026-11-26 19:14:28 UTC, **44.0 h** after launch (sum of the 26 pairing wall times: 43.9 h). 26 pairing files and 26 telemetry sidecars.
- **Stamps re-derived from the files (2026-11-26):** 26 pairing files, 26 distinct `config_hash`es, each with `n_games` 2000, seed 20261013 and `git_commit` `7372ad7`; wins sum to 2000 in every file (**0 ties in 52,000 games**); each agent has 4000 seat occupancies per pairing, 1000 in each seat; telemetry agrees on n and ties. The 52 files on the laptop mirror had the same sha256 as the HP's. The committed copies differ from the HP's only by a final newline (the `end-of-file-fixer` hook, as for 013).
- **`summarize`:** `python -m experiments.league_017 summarize --results-dir experiments/results --name league_017 --dir-013 experiments/results/league_013`. A second run on a scratch copy of the 26 raw files gave the same `league.json` apart from `generated_at`.
- **Timing by pairing class (actual, 2 workers, nice 19, 2000 games per pairing):**

  | class | pairings | wall per pairing, mean (min-max) | worker-s / game, mean (min-max) | smoke worker-s / game, mean | ties |
  |---|---|---|---|---|---|
  | graph-baseline | 8 | 83.0 min (60.7-104.5) | 4.68 (3.63-5.71) | 4.69 | 0/16000 |
  | graph-flat | 12 | 94.3 min (66.9-122.7) | 5.27 (4.01-6.71) | 4.87 | 0/24000 |
  | graph-graph | 6 | 140.2 min (104.6-170.6) | 7.87 (6.27-9.22) | 7.19 | 0/12000 |

  Wall time was **1.14x** the smoke projection (44.0 h vs 38.6 h), within the 1.3x headroom and under 48 h. Summed worker time per unit of n was 147.9 s against the smoke's 139.0 s (1.06x). Peak RSS was 385 MB per worker.

### Timing (HP smoke) and the chosen n

HP, 2026-11-24 22:50-23:08 UTC, `7372ad7`, seed 20261015, n = 8 per pairing, all 26 pairings, 1 worker, nice 19, `OMP_NUM_THREADS=1`. No win rate was read. Worker time is per-game CPU time (`time.process_time`).

| pairing class | pairings | worker-s / game, mean (min-max) | peak RSS | ties |
|---|---|---|---|---|
| graph-baseline | 8 | 4.69 (3.45-5.92) | 450 MB | 0/64 |
| graph-flat | 12 | 4.87 (3.41-6.18) | 450 MB | 0/96 |
| graph-graph | 6 | 7.19 (5.90-9.52) | 445 MB | 0/48 |

Sum over the 26 pairings: 139.0 worker-s per unit of n (estimate before the smoke: 138). Linear scaling gives n = 4000 at 3 workers 51.5 h (77 h at 2) and n = 2000 at 3 workers 25.7 h (**38.6 h at 2**). n = 4000 is over 48 h at either worker count, so the rule gives **n = 2000**. Load was not constant during the smoke: the load average was about 1.2 in the first half and about 9.7 in the second, with 3 GitHub runner workers active.

## Result

Roster of 15, 81 pairings (013's 55 at n = 4000 plus 26 at n = 2000; 272,000 games, 52,000 of them new), seed 20261013, 4-seat `(a,b,a,b)` reduction, anchor `heuristic = 0`, `prior_draws = 1`, 1000 bootstrap resamples (`bootstrap_seed = 0`), 95% intervals. **0 ties** in all 81 pairings. Graph agents play without search; RL seats always reject trade offers. Files: `experiments/results/league_017/` (26 pairing files plus `league.json`), `experiments/results/league_017_telemetry/`.

**Ratings**, with 013's own fit (11 agents) alongside. The 017 ladder replaces 013's; "shift" is Question 6.

| rank | agent | 017 Elo | 017 95% CI | 013 rank | 013 Elo | 013 95% CI | shift |
|---|---|---|---|---|---|---|---|
| 1 | `graph_s4_12m` | +393.7 | [+384.5, +402.1] | - | - | - | - |
| 2 | `graph_s3_12m` | +382.7 | [+373.4, +391.3] | - | - | - | - |
| 3 | `graph_s3_3m` | +255.6 | [+247.4, +263.1] | - | - | - | - |
| 4 | `graph_s4_3m` | +193.5 | [+186.3, +200.8] | - | - | - | - |
| 5 | `trading_heuristic` | +15.0 | [+10.5, +19.7] | 1 | +16.2 | [+10.6, +21.2] | -1.1 |
| 6 | `heuristic` (anchor) | 0 | [0, 0] by construction | 2 | 0 | by construction | 0 |
| 7 | `long_10m` | -65.7 | [-70.3, -60.7] | 3 | -62.9 | [-67.6, -57.8] | -2.7 |
| 8 | `long_8m` | -71.4 | [-76.2, -66.7] | 4 | -69.2 | [-74.4, -64.2] | -2.2 |
| 9 | `ent_decay_3m` | -92.7 | [-97.3, -87.7] | 5 | -89.9 | [-95.0, -84.6] | -2.9 |
| 10 | `ent_ctrl_3m` | -118.0 | [-122.8, -112.8] | 6 | -116.1 | [-121.0, -111.2] | -1.9 |
| 11 | `long_4m` | -125.3 | [-130.0, -120.5] | 7 | -123.4 | [-128.9, -118.4] | -1.9 |
| 12 | `ent_ramp_3m` | -134.2 | [-139.1, -129.3] | 8 | -132.3 | [-137.6, -127.2] | -1.9 |
| 13 | `bc_clone` | -163.0 | [-167.6, -157.8] | 9 | -161.1 | [-166.4, -156.1] | -1.9 |
| 14 | `bc_ft_2m` | -180.7 | [-185.1, -175.3] | 10 | -178.8 | [-183.8, -173.6] | -1.9 |
| 15 | `random` | -689.3 | [-698.6, -678.2] | 11 | -687.3 | [-696.2, -676.6] | -1.9 |

**The 26 new cells.** Percent of games the **column** graph agent wins in its 2v2 pairing with the row opponent (n = 2000, Wilson 95%). Graph-vs-graph pairings appear twice, once from each side.

| opponent | `graph_s4_12m` | `graph_s3_12m` | `graph_s3_3m` | `graph_s4_3m` |
|---|---|---|---|---|
| `trading_heuristic` | 86.85 [85.3, 88.3] | 87.75 [86.2, 89.1] | 81.40 [79.6, 83.0] | 75.05 [73.1, 76.9] |
| `heuristic` | 87.25 [85.7, 88.6] | 88.10 [86.6, 89.4] | 80.95 [79.2, 82.6] | 76.25 [74.3, 78.1] |
| `long_10m` | 94.00 [92.9, 95.0] | 93.10 [91.9, 94.1] | 86.60 [85.0, 88.0] | 83.25 [81.6, 84.8] |
| `long_8m` | 93.30 [92.1, 94.3] | 92.80 [91.6, 93.9] | 87.85 [86.3, 89.2] | 82.90 [81.2, 84.5] |
| `ent_decay_3m` | 94.40 [93.3, 95.3] | 94.35 [93.3, 95.3] | 88.25 [86.8, 89.6] | 86.15 [84.6, 87.6] |
| `graph_s4_12m` | - | 47.45 [45.3, 49.6] | 27.35 [25.4, 29.3] | 22.75 [21.0, 24.6] |
| `graph_s3_12m` | 52.55 [50.4, 54.7] | - | 31.65 [29.6, 33.7] | 21.65 [19.9, 23.5] |
| `graph_s3_3m` | 72.65 [70.7, 74.6] | 68.35 [66.3, 70.4] | - | 38.90 [36.8, 41.1] |
| `graph_s4_3m` | 77.25 [75.4, 79.0] | 78.35 [76.5, 80.1] | 61.10 [58.9, 63.2] | - |

**Answers to the pre-registered questions (descriptive; nothing is called confirmed):**

1. **Against the anchor and `trading_heuristic`:** all four graph agents rate far above both, from +193.5 (`graph_s4_3m`) to +393.7 (`graph_s4_12m`), against +15.0 for `trading_heuristic`; their intervals do not overlap either baseline's. Head-to-head they win 76.25-88.10% against `heuristic` and 75.05-87.75% against `trading_heuristic`. They are the first RL agents on this ladder to rate above `heuristic`. For the two 12M agents the single-scale fit predicts a larger margin over both baselines than was observed (see Question 5), so the rating gap should not be read as an exact head-to-head prediction. 013's trading caveats carry over.
2. **3M vs 12M within a seed:** 3M orders below 12M in both. Seed 3: +255.6 [+247.4, +263.1] vs +382.7 [+373.4, +391.3]; `graph_s3_12m` wins 68.35% of the direct games. Seed 4: +193.5 [+186.3, +200.8] vs +393.7 [+384.5, +402.1]; `graph_s4_12m` wins 77.25%. The intervals do not overlap within either seed. At 3M, seed 3 is above seed 4 (direct 61.10%), the same order as 014's 1v3 benchmark.
3. **`graph_s3_12m` vs `graph_s4_12m`:** the ratings do not separate them: s4 is 11.0 points higher by point estimate and the marginal intervals overlap ([+384.5, +402.1] and [+373.4, +391.3]). The direct pairing leans to s4: s3 wins 47.45% [45.3, 49.6], so s4 wins 52.55%, an interval that excludes 50% by a small margin. That is one of 26 new cells, with no multiplicity correction, and no pre-registered test. Against the five shared flat and baseline opponents neither is ahead everywhere: s3's point estimate is higher against `heuristic` and `trading_heuristic`, s4's against the three flat agents, all by under 1 point. The overlap of marginal intervals is not a test of the difference; `league.json` has no interval for the difference.
4. **Against `long_10m`:** `graph_s4_12m` wins 94.00%, `graph_s3_12m` 93.10%, `graph_s3_3m` 86.60% and `graph_s4_3m` 83.25% (intervals in the table). On the ladder `long_10m` is at -65.7.
5. **Single-scalar fit:** gamekit reports 5 significant 3-cycles (every edge's Wilson interval entirely above 50%). They are **the same 5 as in 013**, all through `long_10m`; **none contains a graph agent**. A cycle can only be found among agents that all played each other, so the bridge design limits which graph triples could show one. No cycle does not mean the fit is exact: in 8 of the 26 new cells the fitted value lies outside the observed Wilson interval (the same check on 013's fit finds 35 of its 55 cells). The new cells' residuals are much smaller than 013's largest: mean absolute 1.3 points, maximum 3.8, against 15.8 for `long_10m` vs `long_8m`, which is still the largest in the joint fit.

| pairing (A vs B) | A observed % | A fitted % | residual (points) |
|---|---|---|---|
| `graph_s3_3m` vs `graph_s4_12m` | 27.35 | 31.1 | -3.8 |
| `graph_s3_12m` vs `graph_s4_3m` | 78.35 | 74.8 | +3.5 |
| `graph_s4_12m` vs `heuristic` | 87.25 | 90.6 | -3.4 |
| `graph_s4_12m` vs `trading_heuristic` | 86.85 | 89.8 | -3.0 |
| `ent_decay_3m` vs `graph_s4_3m` | 13.85 | 16.1 | -2.3 |
| `graph_s3_3m` vs `graph_s4_3m` | 61.10 | 58.8 | +2.3 |
| `graph_s3_12m` vs `heuristic` | 88.10 | 90.1 | -2.0 |
| `graph_s3_12m` vs `trading_heuristic` | 87.75 | 89.3 | -1.5 |

   In the four 12M-vs-baseline cells above, the fit overstates the 12M agent's win rate. Among the graph agents, it misses three round-robin cells by 2.3 to 3.8 points.
6. **Shifts of 013's ratings:** every non-anchor rating moved down relative to `heuristic`, by 1.1 points (`trading_heuristic`) to 2.9 (`ent_decay_3m`); `long_10m` moved -2.7, `long_8m` -2.2, `random` -1.9. The order of 013's 11 agents is unchanged, and each new point estimate lies inside that agent's 013 interval. These are shifts between two fits that share the 55 files, not new measurements of the flat agents.

## Verdict

Descriptive; none scored. No hypothesis was tested and no rating had a pass/fail threshold. Both pre-launch gates passed, with the laptop environment disclosure below. The four graph agents now sit on the anchored scale, above `trading_heuristic` and `heuristic`; their positions rest on the bridge pairings described in Design.

## Deviations / disclosures

- **Timing seed differs from 013's** (`20261015` vs `20261014`) because of a real overlap with 011/012's smoke block; see Seeds.
- **Bridge pairings instead of full rows** and **four of ten candidate agents** are cost choices with their trade-offs stated above, not findings.
- **The cost numbers are estimates** from n = 4 laptop games times one HP/laptop ratio (4.3x), itself taken from a 4-game laptop sample for `heuristic__vs__long_10m` against 013's 4000-game HP mean; the HP smoke replaces them.
- The Step 0 smoke and the RSS measurement ran on the laptop before this document existed, outside any pre-registered pairing, with 013's timing seed; neither read a win rate.
- **n = 2000** for the 26 new pairings (by the registered rule), against 4000 for 013's files, so the graph cells carry less weight and wider intervals.
- **2 workers instead of 3**, from the registered load clause (load 6.1-7.7 at launch). The HP is shared with GitHub Actions runners; load was not recorded continuously during the run. Concurrent load can change timings, not results, since games are deterministic functions of the seeds.
- **Worker counts in the gates differ from the registered commands:** the replay and the laptop cross-check leg used 4 workers (the commands say 2), and the HP cross-check leg used 1.
- **The laptop gates did not run on the locked torch, numpy and sb3** (see Environment). Gate 1 was meant to check the lock bump since 013; it did exercise gamekit 0.5.0, and it passed on torch 2.14.0 / numpy 2.5.3 / sb3 2.9.0, not on the locked 2.14.1 / 2.5.4 / 2.9.2. Indirect evidence for the locked versions: in Gate 2 the HP, on the locked versions, matched the laptop on all five fields in its three pairings (600 games). The replayed 013 pairing itself was not replayed on the locked versions.
- **The first rate check was later than "after the first pairing of each class"** by about 2 h, with 9 pairings finished.
- Timing: 44.0 h against the smoke's 38.6 h (1.14x).
