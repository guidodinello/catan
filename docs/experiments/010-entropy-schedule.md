# Entropy schedule for the BC fine-tune (issue #40)

**Date:** pre-registered 2026-09-29 (commit recorded in the result JSONs' `git_commit` is later; this file's own commit precedes every measured run)
**Note:** [gamekit#010 — Entropy schedule](https://github.com/guidodinello/gamekit/blob/main/docs/research/010-entropy-schedule.md), section 010b (plateau escape). Statistics per [gamekit#005](https://github.com/guidodinello/gamekit/blob/main/docs/research/005-eval-statistics.md). A gamekit follow-up (fill 010b's Result, link this log) is proposed in the PR, not edited there.

**Status: pre-registered, no measured run yet.**

## Hypothesis

004's fine-tune (constant `ent_coef=0.01`) oscillated for its whole 3M steps and 006 plateaued at ~20%. Note 010b asks whether a *schedule* beats the constant on a run that is not collapsing. The prior is low (catan log 002's diagnostic found no collapse). The warm-start ramp (0.005 -> 0.02 over 1M, then constant) keeps the clone intact early and raises exploration later. The cold-start decay (0.02 -> 0.005 over 3M) is the reference direction.

## Implementation (already committed, `8e09ce2`)

`--ent-schedule STEP:VALUE,...` (mutually exclusive with `--ent-coef`, which stays the constant shorthand). `rl/train.py:ent_coef_at` is piecewise-linear and clamped, keyed on cumulative `model.num_timesteps`; `_ent_schedule_callback` sets `model.ent_coef` in `_on_rollout_start` (before the update that consumes it, overwriting any pickled value, so `--resume` cannot keep a stale one) and records `train/ent_coef_effective`. Tests: `tests/rl/test_ent_schedule.py` (pure function, CLI, two `learn()` chunks with no restart, resume overwrite). Also: the bc-init load now overrides the clone's relative pickled `tensorboard_log` with `<run-dir>/tb`, so each arm has its own TB directory.

Facts that differ from the note, fixed here before running:

- **Rollout cadence is 4096 steps, not 8192**: `--envs 8` x `n_steps` 512 (004's shape). The schedule updates every 4096 steps.
- **Breakpoints are on real `num_timesteps`**, which overshoots nominal by up to a rollout: `<label>_3000000.zip` holds ~3,047,424. The ramp reaches 0.02 at real step 1,000,000; the decay arm is clamped at 0.005 for its last ~47k steps.
- **`train/ent_coef_effective` leads `train/entropy_loss` by one rollout at the same TB step** (SB3 dumps logs after collecting rollout k, having recorded the value set at its start, but before update k runs). Traces are reported with this in mind.

## Arms (seed 3, same BC clone, 3M steps, run sequentially)

| arm | label | `--ent-schedule` |
|---|---|---|
| control | `catan_ent_ctrl` | `0:0.01` (constant, same code path, also logs `ent_coef_effective`) |
| warm-start ramp | `catan_ent_ramp` | `0:0.005,1000000:0.02` |
| cold-start decay | `catan_ent_decay` | `0:0.02,3000000:0.005` |

**Why a fresh control instead of re-benchmarking 004's own 3M checkpoint** (the note allows that only if it equals the control config): it does not. 004 trained the PPO update on `torch+cpu`, these arms on CUDA; `rl/train.py` changed since (e.g. two in-loop evals per chunk, `95c59d8`); and no 004 tfevents survive, so 004 cannot supply the required `entropy_loss` trace. The control therefore runs through identical code/device as the other arms. 004's `catan_bc_ft_3000000.zip` is still benchmarked at n=4000 as a **secondary reference** (never benchmarked before; only its 2M checkpoint was), which also shows how far the same config drifts across code/device/run.

## Config (004's shape, from log 004 §Config)

```
PYTHONPATH=$PWD /home/guido/projects/catan/.venv-cuda/bin/python -m rl.train \
  --bc-init /home/guido/projects/catan/rl_runs/bc/catan_bc_clone.zip --bc-init-rate 0.10 \
  --selfplay-dir rl_runs/exp010/selfplay --run-dir rl_runs/exp010/<label> --label <label> \
  --baseline-mix 0.5 --ent-schedule <S> --learning-rate 1e-4 \
  --envs 8 --seed 3 --device cuda \
  --eval-opponents heuristic --eval-every 250000 --eval-episodes 200 \
  --regression-margin 0.10 --regression-patience 1000 --steps 3000000
```

Run from the worktree `../catan-40-entropy` (branch `exp/40-entropy-schedule`), `rl_runs/` there is gitignored and nothing under the main checkout's `rl_runs/` is written (the clone is read only). `PYTHONPATH=$PWD` is explicit because the venvs otherwise resolve `rl`/`agents`/`experiments` to the main checkout inside pool workers (found by a probe); with it, main process, fork workers and `ProcessPoolExecutor` workers all import from the worktree. PPO update on CUDA (torch 2.14.0+cu130, as in log 006); env workers and opponent inference on CPU; 8 env workers + main process, under the 12-core cap.

**Deviations from 004:** CUDA update; current in-loop eval scheme (README §In-loop eval caveats: guard + fresh-seed reported eval); `--regression-patience 1000` for all three arms equally, which only removes the stop rule (it does not change training) so no arm can stop before its fixed 3M checkpoint exists; per-arm run/TB dirs; fresh control (above).

**Aborted first start (disclosed).** A first start of the measured runs on 2026-09-29 ~23:09 was aborted at ~750k-1M steps of `catan_ent_ctrl` (last completed 250k chunk: 750,000) for a laptop shutdown. Its outputs were not used (the in-loop evals it printed are unread as results); they are archived, gitignored, under `rl_runs/exp010/aborted_2026-09-29/`. All arms were restarted from scratch (not `--resume`).

## Metric and gate

`experiments.benchmark --mode rl_vs_heuristic --games 4000 --players 4 --engine-seed-base 1 --driver-seed-base 1 --workers <=12` on each arm's **fixed `<label>_3000000.zip`** (not the in-loop best), seat-rotated, `RLAgent(deterministic=True)`, Wilson CIs, result JSONs named by checkpoint stem (`benchmark_rl_vs_heuristic_p4_catan_ent_<arm>_3000000.json`). `rl_vs_heuristic` is 3 plain `HeuristicAgent`s (`experiments/benchmark.py`). Same boards as 004/006.

**Gate.** An arm is *validated* if its rate is **higher** than the control's and `gamekit.mc.testing.two_proportion_test` (two-sided pooled z) gives **p < 0.025** (Bonferroni, two arms). If neither arm is validated, 010b is *rejected* for catan's BC fine-tune at this budget. An arm significantly *lower* than control is reported as such. The n=4000 minimum detectable difference is ~2.3 points.

**Secondary (not the gate):** 004's 3M checkpoint at n=4000 and control-vs-004; in-loop curves; `entropy_loss` and `ent_coef_effective` traces per arm (median over the first 100k steps, value at ~1M and ~2M, final), exported from each arm's tfevents.

## Limitations, stated up front

One seed per arm; CUDA non-determinism (no bit-reproducibility, so control-vs-004 is a spread estimate, not a replication); three arms plus one reference give one control draw, so a "win" of a few points can be run-to-run variation, which is what the 004-vs-control comparison is for; no test of entropy as the bottleneck (log 004 points at BC fidelity).

## Smoke runs (disclosed; no win rates read as results)

A 16,384-step CUDA smoke of `0:0.005,8192:0.02` (`--envs 2 --eval-episodes 4`, under `rl_runs/exp010_smoke/`) checked that the TB tag follows the schedule (0.005 -> 0.02, clamped after 8192), that the CUDA update runs, and the worktree import paths. Its 4-episode evals are meaningless and unused.

## Result

_Pending._
