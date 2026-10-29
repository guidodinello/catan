# Training the graph encoder longer: 3M to 12M steps (issue #97)

**Date:** pre-registered 2026-10-25. No run, smoke or measurement exists before this file's merge; this PR is docs only.
**Note:** [gamekit#009 — Longer runs / resume when the curve has not bent](https://github.com/guidodinello/gamekit/blob/main/docs/research/009-longer-runs-and-resume.md). Statistics per [gamekit#005](https://github.com/guidodinello/gamekit/blob/main/docs/research/005-eval-statistics.md). Follows [014 — graph encoder](014-graph-encoder.md) (issue #42, PR #95) and [006 — longer run](006-longer-run.md), whose shape it copies.

**Status: pre-registered; training finished (Amendment 2); benchmarks not run.**

## Hypothesis

Log 014 fine-tuned the graph policy for only 3M steps (about 3.05M real steps) and reached 59.05% (graph s3) and 56.57% (graph s4) at n = 4000 on `fresh_014`. The previous best (006, flat) ran about 12M steps. The graph in-loop curves (n = 200, monitoring only, `reported` column of 014's Stage 2 table) were still rising when training stopped: graph s3 40.0% at 250k to 63.0% at 3M, graph s4 34.5% to 59.5%. A flat run plateaued at 20-21% from +6M on (006); whether the graph run keeps climbing is unknown, and a plateau-level claim needs a longer run (014, Limitations).

**H1:** continuing each 3M graph fine-tune to 12M total steps raises its win rate against 3 `HeuristicAgent` at n = 4000 on boards no earlier run has played.

**Design choice, stated as such:** the run *resumes* the existing 3M checkpoints rather than training a fresh 12M run. It costs a quarter of the compute and measures what the owner asked (does more training of this agent help), but it answers nothing about a from-scratch 12M schedule.

## Design (frozen at pre-registration)

- **Starts:** `rl_runs/exp014/selfplay/catan014_ft_graph_s3/catan014_ft_graph_s3_3000000.zip` (sha256 prefix `6cb96ab7d856`) and the same file for seed 4. Both zips record `num_timesteps` = **3,047,424**. Each is resumed with `--resume`; a resumed run's `--steps` is the number of *new* steps (`rl/train.py`, `_resume_step_bookkeeping`).
- **Runs: 2, one per 014 fine-tune seed (3 and 4)**, labels `catan016_graph_s3` and `catan016_graph_s4`, each `--steps 9000000` (cumulative about 12.05M). `--resume` copies the start zip into the new label's pool as the seed opponent (`_seed_selfplay_pool`), as 006 did with its new label. The resume seed (`--seed`) is the same as the original fine-tune's.
- **Flags, as 014 Stage 2** (`helpers/stage2.sh`; explicit `--learning-rate` and `--ent-coef` equal the pickled values, which `--resume` requires):

  ```
  --resume <3M zip> --init-rate <014's n=4000 rate of that checkpoint> \
  --baseline-mix 0.5 --ent-coef 0.01 --learning-rate 1e-4 \
  --envs 8 --n-steps 512 --device cuda \
  --eval-opponents heuristic --eval-every 250000 --eval-episodes 200 \
  --regression-margin 0.10 --regression-patience 1000 --steps 9000000
  ```

  `--init-rate` is 0.5905 for s3 and 0.5657 for s4 (014's `fresh_014` results); it only seeds `RegressionGuard`, and `--regression-patience 1000` disables the stop rule, so every run reaches its full step count.
- **Fixed measurement points (decided now):** the checkpoints written **+3M, +6M and +9M steps after the resume point** (cumulative about 6.05M, 9.05M, 12.05M), named by the trainer; their file names are recorded in the Amendment. Nothing else is benchmarked for the gate, in particular not the in-loop best, which is biased upward by selecting the maximum of noisy n = 200 evals.
- **Boards:** `fresh_016` = engine seeds **23,000,001 … 23,004,000** (n = 4000, divisible by 4 for exact seat rotation), `driver_seed` equal to the engine seed. A code change registers it in `experiments/search_eval.py` `SEED_RANGES` and `experiments/league_013.py` `USED_RANGES` with the existing disjointness test before any benchmark (see Docs-first protocol). Disjoint from every used range, including `fresh_012` (20,000,001+), `fresh_014` (21,000,001+), the reserved `fresh_015` block (22,000,001+, issue #96), the benchmark boards 1..10,000 and the in-loop eval range (`cfg.seed + 977 + step`, below about 25M).
- **Opponents:** 3 plain `HeuristicAgent`, seat-rotated by `engine_seed % 4`, `RLAgent(deterministic=True)` (the `rl_vs_heuristic` benchmark mode), as in 014.
- **Baseline:** the two 3M start checkpoints are **re-benchmarked on `fresh_016`**, so the comparison is paired on the same boards. 014's 59.05% / 56.57% came from `fresh_014`, and graph s3 was the better of two checkpoints there, so it is biased upward; the re-benchmark removes that bias from the gate.

## Pre-registered gate and verdict rule

α = 0.05, two-sided. Let `P3` be the two 3M checkpoints pooled (2 x 4000 games) and `P12` the two +9M (about 12M) checkpoints pooled, all on `fresh_016`. The gate **passes iff all three hold**:

1. `gamekit.mc.testing.two_proportion_test(P12, P3)`: **p < 0.05 with a positive difference**.
2. **Each seed improves:** s3 at +9M above s3 at 3M, and s4 at +9M above s4 at 3M (point estimates).
3. The pooled difference is **> 2.2 points** (the run-to-run spread 010 observed between two same-config fine-tunes, as in 014's S2).

At n = 8000 per pool and a rate near 0.58, the minimum detectable difference at 80% power is about 2.5 points, so clause 3 is the binding one.

| outcome | verdict | consequence |
|---|---|---|
| gate passes | **longer training helps** | the best +9M checkpoint is a candidate for the next "current best"; the 6M and 9M points describe how fast it rises |
| p < 0.05 but clause 2 or 3 fails | **inconclusive** | reported with all numbers; no claim of a gain |
| not significant, or the pooled difference is negative | **no gain from +9M at this config** | reported with both numbers; also covers a significant decrease |

Secondary, reported, not gated: every fixed point against the 3M baseline, with Benjamini-Hochberg-adjusted p; the 3M → 6M → 9M → 12M curve per seed (does it plateau?); board-paired exact McNemar for each seed's +9M vs its 3M (`experiments.search_eval.mcnemar_exact`, per-board outcomes by deterministic replay as in 014's `perboard.py`); per-rotation-position wins; the in-loop curves, labelled monitoring only (README §In-loop eval caveats); the +9M checkpoints against 006's 20.72% and 014's 59.05% (different boards, stated as such).

**The README "current best" line changes only** if a measured +9M checkpoint beats the then-current best on the same footing, i.e. on `fresh_016` against the re-benchmarked 3M baseline; a number from other boards does not move it.

## Docs-first protocol

1. This pre-registration merges first (docs only).
2. **Before any run**, a dated Amendment is committed to this file recording: the code merge SHA and a dedicated worktree (never the main checkout); the lock-pinned environment (`uv sync --frozen --dev --extra rl-train-cuda`, the CUDA venv on the HDD via `UV_PROJECT_ENVIRONMENT`, as in 014); the `fresh_016` registration PR and its disjointness test; the start zips' sha256 re-verified; and the resume sanity check. **Training code has changed since 014's run commit `7b353e3`** (`d27b6f9`: `CatanEnv` on gamekit's shared reset helpers with a gamekit bump, plus `uv.lock` updates), so the sanity check is required: a 4096-step resume smoke from a start zip on smoke boards, reading no win rate, confirming the effective `learning_rate`, `ent_coef`, `n_steps` and `num_timesteps` in the log match the start. If the new env changes training behavior for the graph policy, the log says so and the run is not started from a different commit without an amended pre-registration.
3. **Harness check:** the graph s3 3M start on `fresh_014` must reproduce **2362 / 4000** exactly (the benchmark is deterministic; 014's Stage 2 result), and flat `catan_bc_ft_long_10031616` on boards 1..4000 must give **829 / 4000** (logs 012 and 014). If either does not, stop and find out why before measuring anything.
4. Smokes use the smoke seed range only and read no win rate.

## Compute budget and wall time (laptop: RTX 4050 6 GB, 20 cores, 15 GB RAM, daytime only)

Estimates from 014's Stage 2 run log, which the first 250k-step chunk of the real run replaces: graph training ran at `train_fps` 271-287, 3 h 28 min and 3 h 33 min per 3M steps, of which 178.8 / 182.3 min were training and 28.2 / 28.4 min in-loop eval (n = 200, twice per 250k chunk).

| item | estimate |
|---|---|
| +9M steps, one seed | about 3 x 3.5 h = **10.5 h** (about 9 h training plus 1.4 h eval) |
| two seeds, sequential (one GPU) | **about 21 h** |
| fixed-point benchmarks (3M baselines + 3 points x 2 seeds = 8 checkpoints on `fresh_016`) | about 1 h at 6 workers (graph: 4-7 min each at 3-6 workers in 014) |
| **Total** | **about 22 h, 3 daytime days** |

- **`timeout`:** 014 used `timeout 8h` for a 3M-step job. One +9M job needs about 10.5 h, so it runs under `timeout 14h` (a 33% margin over the estimate), a change from the 8 h convention that is stated here and not discovered later. The laptop runs daytime only, so a run is stopped each evening with `tmux send-keys -t <name> C-c` and continued with `--resume` from its last 250k-step checkpoint under a new label (`..._b`, `..._c`), because `_final.zip` and checkpoint saves refuse to overwrite an existing file. Every stop and continuation, and the steps lost since the last 250k checkpoint, are disclosed. A continued run starts a new opponent pool from the checkpoint it resumes (the seed copy), unlike one uninterrupted run; the log states how many continuations occurred.
- **If a job is cut short** (timeout, crash), it is continued, not benchmarked at a different step count to rescue the gate.
- **Memory:** `--envs 8` plus the main process, as in 014; check `free -g` before launch.

## Run protocol

- **tmux + log:** each training job in a named session, `tmux new -d -s catan016-<arm>-s<seed> '<cmd> 2>&1 | tee <log>'`, a helper script modelled on 014's `stage2.sh`. No harness background shells.
- **Refuse duplicates:** before every launch `pgrep -af '[r]l.train|[r]l.bc|[e]xperiments.'`, `tmux ls`, `free -g`. A dead wrapper is not a dead job: check `pgrep` before relaunching.
- **Stop:** `tmux send-keys -t <name> C-c` (or SIGTERM to that session's process); never a `pkill -f` pattern.
- **Outputs:** `rl_runs/exp016/` on the HDD (`/media/guido/0DF7128F0DF7128F`), never `/tmp` or another tmpfs. Result JSONs go to `experiments/results/` named by checkpoint stem.
- **Order:** harness checks and the resume smoke, the s3 run, the s4 run, then the benchmarks. The benchmarks may start once both runs are finished (the 014 deviation of benchmarking during training is not repeated).

## Limitations, stated up front

- Two fine-tune seeds, one BC clone seed each: the gate is a consistency check, not a variance estimate.
- Resume, not a fresh 12M run (design choice above); self-play pool continuity differs if the run is continued under new labels.
- One opponent set (3 `HeuristicAgent`), 4-player, no trading. The training mix includes `HeuristicAgent` opponents (`--baseline-mix 0.5`), the same type as the benchmark; more training may raise the win rate against that type only, which is not a claim about other opponents.
- CUDA non-determinism: no bit-reproducibility of the training runs.
- 12M steps is the budget of 006; a plateau claim past it would need another run.

## Environment

To be filled by the Amendment (code SHA, lock-pinned versions, GPU driver). In-loop eval: seed scheme per README §In-loop eval caveats.

## Smoke / timing runs

None before this file's merge.

## Amendment 1 (2026-10-25): pre-run prep, before any training

Recorded before any `catan016_*` run exists. Nothing in the Design, gate or verdict rule changes.

**Code.** `fresh_015` (22,000,001-22,004,000) and `fresh_016` (23,000,001-23,004,000) are registered in `experiments/search_eval.py` `SEED_RANGES` and `experiments/league_013.py` `USED_RANGES` by PR #100, squash-merged as **`c3a8b97c3c84e7916d6c269f13188f3926af1041`**. The disjointness test is `test_fresh_seeds_are_disjoint_from_every_used_range` in `tests/rl/test_graph_policy.py` (the `fresh_014` test, now parametrized over `fresh_014`, `fresh_015`, `fresh_016`; each block is checked against every used range and against the other fresh blocks, with the in-loop bound `(0, 12_100_000)`). Runs use the dedicated worktree `/home/guido/projects/catan-016-run`, detached at that SHA, clean tree; never the main checkout.

**Environment (lock-pinned).** `UV_CACHE_DIR=<HDD>/uv-cache UV_PROJECT_ENVIRONMENT=/media/guido/0DF7128F0DF7128F/catan-016-venv uv sync --frozen --dev --extra rl-train-cuda` from the worktree (tmux `catan016-venv`, log `rl_runs/exp016/logs/venv.log`, exit 0). Read back: Python 3.14.5, torch 2.14.1+cu130 (CUDA 13.0, `cuda.is_available()` true), **gamekit 0.5.0**, sb3-contrib 2.9.0, stable-baselines3 2.9.0, numpy 2.5.4, gymnasium 1.4.0; GPU NVIDIA GeForce RTX 4050 Laptop (6 GB), driver 595.99.02. **Versus 014's run environment: gamekit 0.4.0 -> 0.5.0 and numpy 2.5.3 -> 2.5.4.**

**Start zips (sha256 re-verified).** `catan014_ft_graph_s3_3000000.zip` `6cb96ab7d856...` (matches the pre-registered prefix); `catan014_ft_graph_s4_3000000.zip` `ed7c57e28f6e...`.

**Resume sanity smoke.** Tmux `catan016-smoke`, `timeout 15m`, `--resume` the s3 3M zip, `--init-rate 0.5905`, the Design flags, `--seed 3`, `--steps 4096`, `--eval-every 10000000`, label `smoke016_graph_s3`, everything under `rl_runs/exp016_smoke/` (log `smoke.log`); no registered label was used. It exited 0. The trainer logged `resumed hyperparameters: learning_rate=0.0001 (lr_schedule(0)=0.0001 lr_schedule(1)=0.0001) ent_coef=0.01 n_steps=512 n_envs=8 num_timesteps=3047424`, i.e. all four match the start and the pre-registered flags, and it finished at `steps=3051520` (3,047,424 + 4,096), `train_fps=197`. **Disclosure:** the trainer always runs one in-loop eval when a run ends, so a 200-game win rate (`reported=55.5%`, `guard(fixed)=58.0%`) was printed in the smoke log. It was not used for any decision and is not a result: the boards are in-loop eval seeds, and the run is 4,096 steps. **What the smoke does not show:** it does not test whether the post-`d27b6f9` `CatanEnv` changes training behavior for the graph policy; it shows only that a resume loads, keeps the hyperparameters, and steps. The Design's rule stands: if a later comparison shows a behavior change, the log says so.

**Harness check (log 016 step 3).** Run from the 015 worktree (same SHA) with the CPU venv, `experiments.search_eval run --arm none`, 4 workers, `CUDA_VISIBLE_DEVICES` empty: graph s3 3M on `fresh_014` = **2362 / 4000** (expected 2362) and flat `catan_bc_ft_long_10031616` on boards 1..4000 = **829 / 4000** (expected 829). Both reproduce exactly (see Amendment 1 of log 015 for the commands).

**Runner.** `rl_runs/exp016/helpers/train016.sh` (sha256 `ad17a62e9635...`, on the HDD, not committed; modeled on `experiments/exp014/stage2.sh`): guards (bracketed `pgrep`, `tmux ls`, HDD mounted, 5 GB RAM wait), one tmux session per leg (`catan016-graph-s<seed>[-<suffix>]`), `tee` to `rl_runs/exp016/logs/<name>.log`, `EXIT=` line and `done/` marker, **`timeout 14h`** (as pre-registered), refuses a label that already has a run dir. Not launched. Continuation legs are `s3:b:<last 250k zip>:<remaining steps>`. A leg is stopped by SIGTERM to its own `rl.train` process (found by exact label), never `C-c` on the `tee`'d session or `pkill`.

## Amendment 2 (2026-10-29): training legs, step-count correction, overshoot cause

Recorded after all training finished and **before any benchmark**. Nothing in the Design, gate or verdict rule changes. The Design says the Amendment records the fixed points' file names and how many continuations occurred; both are below.

**Legs.** Runner `train016.sh`, logs in `rl_runs/exp016/logs/`, checkpoints in `rl_runs/exp016/selfplay/`. "Real" is the zip's own `num_timesteps` (see the correction below), read from the zip.

| leg (label) | start | `--steps` | ended | exit | last checkpoint (real `num_timesteps`) |
|---|---|---|---|---|---|
| `catan016_graph_s3` | 014 s3 3M zip (3,047,424) | 9,000,000 | killed 2026-10-27 22:11 (started 13:24) | **137** | `_10297424.zip` (10,412,032) |
| `catan016_graph_s3_b` | `s3_10297424` (10,412,032) | **1,750,000** | stopped about 23:23, no `EXIT` line (started 22:34) | none | `_b_10662032.zip` (10,665,984) |
| `catan016_graph_s3_c` | `s3_b_10662032` (10,665,984) | 1,381,440 | 2026-10-28 15:22 (started 13:47) | 0 | `_final.zip` (12,070,912) |
| `catan016_graph_s4` | 014 s4 3M zip (3,047,424) | 9,000,000 | 2026-10-29 01:33 (started 2026-10-28 15:23) | 0 | `_final.zip` (12,189,696) |

- **s3 leg 1: exit 137** (SIGKILL), a **global OOM kill, verified** in the kernel log of the previous boot (the laptop rebooted afterwards, so plain `journalctl -k` shows only the current boot; use `journalctl -k -b -1`). It records `oct 27 22:11:04 kernel: Out of memory: Killed process 70930 (python) ... oom-kill:constraint=CONSTRAINT_NONE,...,global_oom,task_memcg=/user.slice/user-1000.slice/session-c2.scope`, one second after a browser kill at 22:11:03 (`Killed process 3497 (WebExtensions)`). The leg's log stops at 22:11, matching. It is the only `python` kill in that boot; the same boot has other `global_oom` kills of browser processes (and Docker Desktop at 21:37) from 21:37 to 23:19, i.e. the 15 GB laptop was short of memory while the browser and concurrent agents shared it with the training. The leg had run about 8 h 47 min, inside the `timeout 14h`.
- **Leg b** was launched with `--steps 1750000`, computed from the **checkpoint file name** (`_10297424`: 12,047,424 - 10,297,424 = 1,750,000). The zip's real count was 10,412,032, so leg b would have finished at 12,162,032, 114,608 past the target. It was stopped before that. Its log has one line (22:53:10, `steps=10662032/12162032`, one 250k checkpoint) and no later entry; the stop time of about 23:23 is from the run notes and is not in any log. Leg b's checkpoint stays in the s3 lineage and leg c resumed from it.
- **Leg c** took its `--steps` from the corrected rule (12,047,424 - 10,665,984 = 1,381,440), hence `steps=.../12047424` in its log and a target met in the trainer's own accounting.
- **Work lost** to the two interruptions is the training after the last checkpoint of each leg: leg 1 after its 21:45 checkpoint (killed 22:11), leg b after its 22:53 checkpoint. Each is under one 250k chunk and was never saved; nothing from those partial chunks is in any checkpoint.
- **Continuations:** s3 ran in **3 legs (2 continuations)**, s4 in **1**. Each continuation starts a new self-play opponent pool from the checkpoint it resumes (the `_seed.zip` copy), unlike one uninterrupted run, as the Compute budget section warned. s3's three pools and s4's single pool are a difference between the two seeds that the gate does not account for.
- The `RegressionGuard` stop never fired (`--regression-patience 1000`), as designed.

**Step-count correction.** A checkpoint's file name is **not** its trained step count; the zip's `data` JSON holds the real one:

```
python3 -I -c 'import sys,zipfile,json; print(json.loads(zipfile.ZipFile(sys.argv[1]).read("data"))["num_timesteps"])' <zip>
```

A continuation's `--steps` is 12,047,424 minus that number. The runner's usage comment suggested file names; it now says this (runner change below).

**Why the final zips overshoot the 12,047,424 target (verified).** Final real counts: s3_c 12,070,912 (+23,488), s4 12,189,696 (+142,272). Cause, from `rl/train.py` and the pinned SB3 2.9.0 / sb3-contrib 2.9.0 source:

- `rl/train.py` (the `while done < target_step` loop) calls `model.learn(this_chunk, reset_num_timesteps=False)` and then advances **its own counter** `done += this_chunk`. Checkpoint names, the `steps=` log line and the loop bound all use `done`, never `model.num_timesteps`.
- `MaskablePPO.learn` loops `while self.num_timesteps < total_timesteps` and adds `env.num_envs` per environment step, so it only stops after a **whole rollout** of `n_steps x n_envs` = 512 x 8 = **4,096** steps. A 250,000-step chunk therefore trains ceil(250000 / 4096) x 4096 = 253,952 steps: **+3,952 per chunk**, which `done` does not see.
- `_resume_step_bookkeeping` restarts `done` from the zip's real `num_timesteps` on `--resume`, so the hidden surplus of earlier legs is carried into the next leg's names, while later chunks add their own again.

Every real count read from a zip matches this arithmetic exactly:

| zip | arithmetic | real |
|---|---|---|
| 014 start (both seeds) | 12 chunks x 253,952 | 3,047,424 (vs 3,000,000 nominal) |
| `s3_10297424` | 3,047,424 + 29 x 253,952 | 10,412,032 |
| `s3_b_10662032` | 10,412,032 + 253,952 | 10,665,984 |
| `s3_c_11165984` | 10,665,984 + 2 x 253,952 | 11,173,888 |
| `s3_c_final` | 10,665,984 + 5 x 253,952 + 33 x 4,096 (last chunk 131,440 -> 33 rollouts) | 12,070,912 |
| `s4_6047424` / `s4_9047424` | 3,047,424 + 12 / 24 x 253,952 | 6,094,848 / 9,142,272 |
| `s4_final` | 3,047,424 + 36 x 253,952 | 12,189,696 |

No step is untrained or logged as trained when it was not; the defect is that names, the `steps=` line and the loop bound count a nominal number. [006](006-longer-run.md) and [010](010-entropy-schedule.md) already noted a per-chunk overshoot of up to one rollout; 016 adds that it **accumulates over chunks and across resumed legs**, and that reading a file name as a step count produced leg b's wrong `--steps`. It also leaves the two seeds at slightly different real step counts at the 12M point (s3 +0.2%, s4 +1.2% over the target); the 6M and 9M points are matched (6,094,848 and 9,142,272 for both).

**This is a code issue, not fixed here.** The suggested fix is in `rl/train.py`: take `done` from `model.num_timesteps` after each `learn()` and loop on `model.num_timesteps < target_step` (or round each chunk to a multiple of `n_steps x n_envs`), with a test beside the `_resume_step_bookkeeping` tests. It changes checkpoint naming, which logs 006, 010, 014 and this one cite, so it should be its own change; this Amendment only records the behavior as run.

**Fixed measurement points (file names, as the Design asks).** The checkpoints the trainer wrote at nominal +3M, +6M and +9M. Real counts and sha256 prefixes read from the zips on 2026-10-29:

| seed | nominal | file | real `num_timesteps` | sha256 |
|---|---|---|---|---|
| s3 | +3M | `catan016_graph_s3/catan016_graph_s3_6047424.zip` | 6,094,848 | `e3c92c539efc` |
| s3 | +6M | `catan016_graph_s3/catan016_graph_s3_9047424.zip` | 9,142,272 | `e842d689fca3` |
| s3 | +9M | `catan016_graph_s3_c/catan016_graph_s3_c_12047424.zip` | 12,070,912 | `f24c64b8ca28` |
| s4 | +3M | `catan016_graph_s4/catan016_graph_s4_6047424.zip` | 6,094,848 | `915bf8543399` |
| s4 | +6M | `catan016_graph_s4/catan016_graph_s4_9047424.zip` | 9,142,272 | `079a6c2ec90a` |
| s4 | +9M | `catan016_graph_s4/catan016_graph_s4_12047424.zip` | 12,189,696 | `6b7e921955d8` |

The `+9M` point is the trainer-named `_12047424` checkpoint, not `_final.zip`: the two hold the same `num_timesteps` but their file hashes differ (`s3_c_final` `0406ef034288`, `s4_final` `7a0b91faac22`), and the Design names the checkpoint written at the point. The 3M baselines are the 014 start zips from Amendment 1. Nothing has been benchmarked.

**Best in-loop checkpoints (n = 200, monitoring only; not the pre-registered measure).** The in-loop `reported` rate is the maximum of noisy n = 200 evals and is biased upward (README, In-loop eval caveats); the Design says the gate does not use it. For the record: s3 `catan016_graph_s3_c_11165984.zip` 73.5% (147/200, real 11,173,888); s4 `catan016_graph_s4_11297424.zip` 75.0% (150/200, real 11,427,840). They are not benchmarked and not candidates for any claim.

**Runner change (comment only).** The usage comment of `rl_runs/exp016/helpers/train016.sh` now gives the continuation `<steps>` as 12,047,424 minus the zip's real `num_timesteps`, with the one-liner above, and notes the leg still lands past the target by a rollout. No executable line changed (checked with `diff`: only `#` lines differ; `bash -n` passes). sha256 `ad17a62e9635...` -> **`2b0439d8cc97...`**. The script is on the HDD and not committed, as in Amendment 1.

## Run log

Training ran 2026-10-27 to 2026-10-29; the legs, interruptions and counts are in Amendment 2. No benchmark has been run.

## Deviations / disclosures

- Written before any run, smoke or measurement.
- The `timeout 14h` for a +9M job replaces 014's `timeout 8h` for a 3M job (see Compute budget).
- s3 was trained in 3 legs and s4 in 1 (Amendment 2): one OOM kill (exit 137) and one stopped leg, each losing less than a 250k chunk of unsaved training.
- The real step counts of the +9M checkpoints are 12,070,912 (s3) and 12,189,696 (s4), not 12,047,424: `rl/train.py` counts nominal chunk steps while SB3 trains whole 4,096-step rollouts (Amendment 2).
- Leg b's `--steps` was computed from a checkpoint file name and was wrong; the leg was stopped before it overshot, and leg c used the corrected rule.
