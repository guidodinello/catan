# Entropy schedule for the BC fine-tune (issue #40)

**Date:** pre-registered 2026-09-29 (commit recorded in the result JSONs' `git_commit` is later; this file's own commit precedes every measured run)
**Note:** [gamekit#010 — Entropy schedule](https://github.com/guidodinello/gamekit/blob/main/docs/research/010-entropy-schedule.md), section 010b (plateau escape). Statistics per [gamekit#005](https://github.com/guidodinello/gamekit/blob/main/docs/research/005-eval-statistics.md). A gamekit follow-up (fill 010b's Result, link this log) is proposed in the PR, not edited there.

**Status: complete — 010b rejected for catan's BC fine-tune at this budget.** Pre-registration committed (`155fd7b`) before any measured run; the result JSONs carry `git_commit` `b20f501` (docs-only on top of the code in `8e09ce2`).

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

Runs 2026-09-30 (the from-scratch restart), sequential, `--envs 8`, CUDA update, seed 3. Wall clock per arm ~1 h 45 min (12:44 -> 14:30 -> 16:12 -> 17:55), 12 chunks each, `RegressionGuard` never stopped an arm (it was disabled by `--regression-patience 1000`; no "stopping early"). Each arm's `_3000000.zip` holds real `num_timesteps=3,047,424`. The benchmarks ran at `--workers 12` (no #37 job was running at that moment). The orchestrator had a #37 calibration job sharing the machine during training (load ~14 at the start); it does not change the training computation, only wall clock.

### n=4000 vs 3 `HeuristicAgent` at the fixed 3M checkpoint (the gate)

Seat-rotated, `engine_seed_base=1`, `driver_seed_base=1`, `RLAgent(deterministic=True)`. p-values are `gamekit.mc.testing.two_proportion_test` (two-sided) against the control's 652/4000.

| arm | checkpoint | wins/n | win rate | Wilson CI | vs control | p | gate (p<0.025 and higher) |
|---|---|---|---|---|---|---|---|
| control (0.01 const) | `catan_ent_ctrl_3000000` | 652/4000 | 16.30% | [15.19%, 17.48%] | -- | -- | -- |
| warm-start ramp | `catan_ent_ramp_3000000` | 694/4000 | 17.35% | [16.21%, 18.55%] | +1.05 pt | 0.209 | **no** |
| cold-start decay | `catan_ent_decay_3000000` | 695/4000 | 17.38% | [16.23%, 18.58%] | +1.07 pt | 0.199 | **no** |
| *reference: 004's own run* | `catan_bc_ft_3000000` | 564/4000 | 14.10% | [13.06%, 15.21%] | -2.20 pt | 0.006 | (not an arm) |

Ramp vs decay: p = 0.976. Result JSONs: `experiments/results/benchmark_rl_vs_heuristic_p4_{catan_ent_ctrl,catan_ent_ramp,catan_ent_decay,catan_bc_ft}_3000000.json`. Checkpoints not committed (`rl_runs/` gitignored).

### In-loop evals vs `HeuristicAgent`, n=200 (monitoring only; README §In-loop eval caveats)

`reported` (fresh seed per step) / `guard` (fixed seed). The seeds depend only on `cfg.seed` and `step`, so at a given step all three arms play the same 200 setups. The identical 18.5% (37/200) at step 3M in all three arms is a coincidence of the counts, not a bug (the n=4000 counts differ: 652/694/695).

| step | control | ramp | decay |
|---|---|---|---|
| 250k | 10.0 / 11.0 | 10.0 / 10.0 | 12.0 / 11.5 |
| 500k | 10.0 / 11.5 | 9.0 / 20.5 | 14.5 / 10.0 |
| 750k | 13.0 / 12.5 | 14.5 / 14.5 | 9.0 / 7.0 |
| 1.0M | 13.5 / 13.0 | 18.0 / 10.5 | 8.0 / 8.5 |
| 1.25M | 9.0 / 13.5 | 10.5 / 11.5 | 7.5 / 14.0 |
| 1.5M | 10.5 / 17.5 | 13.0 / 13.5 | 5.5 / 9.5 |
| 1.75M | 12.0 / 16.0 | 22.5 / 13.5 | 10.5 / 10.5 |
| 2.0M | 16.5 / 14.5 | 14.0 / 12.5 | 13.0 / 7.5 |
| 2.25M | 13.5 / 22.5 | 13.0 / 10.5 | 15.0 / 13.5 |
| 2.5M | 18.0 / 25.0 | 13.5 / 15.5 | 16.5 / 13.5 |
| 2.75M | 18.5 / 16.0 | 14.5 / 10.0 | 15.0 / 15.5 |
| 3.0M | 18.5 / 17.0 | 18.5 / 17.0 | 18.5 / 16.0 |

The in-loop curves are not evidence for either arm (±6-point CIs; the n=4000 numbers differ from the in-loop peaks in the same direction as 004 and 006).

### `entropy_loss` and `ent_coef_effective` traces

Full per-update series: `docs/experiments/010-traces.csv` (arm, tag, TB step, value; `train/entropy_loss`, `train/ent_coef_effective`, `train/approx_kl`). `entropy_loss` is the negative mean policy entropy, so a value closer to 0 is a more deterministic policy. Medians over the TB steps in each window (the `ent_coef_effective` at a TB step is the value set for the rollout that *started* one rollout earlier, i.e. it leads the `entropy_loss` logged at the same step by one 4096-step rollout):

| window | control: entropy_loss / ent_coef | ramp | decay |
|---|---|---|---|
| first 100k | -0.194 / 0.0100 | -0.191 / 0.0057 | -0.196 / 0.0198 |
| 0.9-1.1M | -0.164 / 0.0100 | -0.165 / 0.0199 | -0.204 / 0.0150 |
| 1.9-2.1M | -0.164 / 0.0100 | -0.193 / 0.0200 | -0.186 / 0.0100 |
| last ~100k | -0.153 / 0.0100 | -0.178 / 0.0200 | -0.157 / 0.0050 |

No collapse in any arm: `entropy_loss` stayed between about -0.15 and -0.20 throughout, and median `approx_kl` was 0.001-0.0025, consistent with log 002's diagnostic. The schedules did move entropy in the expected direction (the ramp arm holds about 0.03 more entropy than control from ~2M on; the decay arm is the most exploratory around 1M and converges toward control's level as the coefficient falls), but that moved the n=4000 win rate by only about one point.

## Verdict

**Rejected for catan's BC fine-tune at this 3M budget (gate not met by either arm).** Neither arm beat the control at p < 0.025: ramp +1.05 pt (p = 0.209), decay +1.07 pt (p = 0.199); both differences are below the ~2.3-point MDE. Not validated, and not significantly worse either, so the honest reading is "no detectable effect at this resolution", not "a schedule hurts".

What the data says beyond the letter of the gate (interpretation, not verdict):

- **Run-to-run spread is as large as the effect.** The control (16.30%) and 004's own 3M run (14.10%), same config modulo device/code, differ by 2.2 points (p = 0.006). A single seed cannot distinguish a one-point schedule effect from that spread; two arms each landing about one point above the control is suggestive of nothing on its own (both are also within the control's own run-to-run band).
- **The plateau is not obviously entropy-limited.** Entropy stayed healthy in every arm (no collapse), consistent with log 004's pointer that the bottleneck is BC fidelity on spatial decisions.
- **No comparison to 006's level.** All three arms are 3M steps, at about 16-17%, i.e. in the range of 004's best (16.2% at 2M) and well under 006's ~20% after 12M; this experiment did not test longer budgets. The README current-best line is unchanged (20.72%).

Limitations carried from the pre-registration: one seed per arm, CUDA non-determinism, one control draw. Confounds: the control also differs from 004 in device and code; the decay arm's last ~47k steps are clamped at 0.005; the in-loop eval scheme differs from 004's.

## Notes / follow-up

- Proposed gamekit follow-up (not edited here, see the PR body): fill note 010's 010b Result/verdict with this log's numbers and link it from the note's Result section; correct the note's 8,192-step rollout cadence to 4,096 for `--envs 8` x `n_steps` 512; and add that the "control = 004 itself" shortcut did not hold (004's 3M checkpoint benchmarks 2.2 points below a same-config rerun).
- If anyone wants a still-cheap follow-up: a second seed for control and the better arm would put a number on the run-to-run spread this single-seed design cannot resolve.
