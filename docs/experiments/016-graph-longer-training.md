# Training the graph encoder longer: 3M to 12M steps (issue #97)

**Date:** pre-registered 2026-10-25. No run, smoke or measurement exists before this file's merge; this PR is docs only.
**Note:** [gamekit#009 — Longer runs / resume when the curve has not bent](https://github.com/guidodinello/gamekit/blob/main/docs/research/009-longer-runs-and-resume.md). Statistics per [gamekit#005](https://github.com/guidodinello/gamekit/blob/main/docs/research/005-eval-statistics.md). Follows [014 — graph encoder](014-graph-encoder.md) (issue #42, PR #95) and [006 — longer run](006-longer-run.md), whose shape it copies.

**Status: pre-registered, not run.**

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

## Run log

Empty until the Amendment and the first launch.

## Deviations / disclosures

- Written before any run, smoke or measurement.
- The `timeout 14h` for a +9M job replaces 014's `timeout 8h` for a 3M job (see Compute budget).
