# Longer run / resume from the best 004 checkpoint (PR #24)

**Date:** 2026-09-29
**Note:** [gamekit#009 — Longer runs / resume when the curve has not bent](https://github.com/guidodinello/gamekit/blob/main/docs/research/009-longer-runs-and-resume.md)

## Hypothesis

004's fine-tune oscillated in a 10.5-20.0% in-loop band for its whole 3M steps and its
`RegressionGuard` never fired, so "would more steps keep climbing?" was never answered. Note
009's refinement shows the n=200 in-loop eval cannot answer it either (MDE ~10 points, wider
than the band), so the measurement here is a small set of **fixed, pre-chosen checkpoints, each
benchmarked at n=4000** (MDE ~2.3 points), not the in-loop curve.

## Pre-registration (written and committed before the run started)

**Start:** `rl_runs/selfplay/catan_bc_ft/catan_bc_ft_2000000.zip`, whose real
`num_timesteps` is **2,031,616** (read from the zip). Its n=4000 result, 16.2% = 648/4000, is
the baseline; it came from separate benchmark games, so it is free of winner's-curse bias
relative to the later points.

**Fixed measurement points** (nominal checkpoint names, offset from the real start step;
each chunk's actual `num_timesteps` overshoots the nominal by up to one 4096-step rollout):
`catan_bc_ft_long_{4031616, 6031616, 8031616, 10031616, 12031616}.zip`, i.e. +2M, +4M, +6M,
+8M, +10M. Each is benchmarked vs 3 `HeuristicAgent`, n=4000, `engine_seed_base=1`,
`driver_seed_base=1`, seat-rotated, `RLAgent(deterministic=True)`.

**Verdict rule (gate = the last fixed point reached):**

- *validated*: it beats 648/4000 by `gamekit.mc.testing.two_proportion_test` with p < 0.05 and
  non-overlapping Wilson CIs, **and** the last three fixed points reached are non-decreasing
  in point estimate.
- *rejected* (for this config): it is not significant vs 648/4000 and its point estimate is
  under +2.3 points.
- *inconclusive*: anything else (e.g. significant but not monotone).
- Secondary, not the gate: the best in-loop checkpoint at n=4000 (winner's-curse biased);
  every fixed point vs 648/4000 with Benjamini-Hochberg-adjusted p.
- **If `RegressionGuard` stops the run:** benchmark every fixed point reached, plus the
  guard's best checkpoint and the last checkpoint written. The verdict applies to the last
  fixed point reached; if that is below +4M the verdict is *inconclusive (stopped early)*.

**Roadmap gate:** the Phase 5 checkbox is ticked only if some n=4000 CI lies entirely above 25%.

**Deviations from note 009's protocol** (all user-directed or stated up front): start at the 2M
checkpoint, not `_final`/3M; a new label (`catan_bc_ft_long`) with a pool seeded from that one
checkpoint, not 004's full pool; `--envs 16` with `--n-steps 256` (rollout stays 4096 = 8 × 512,
so update cadence matches 004; a bare 16 × 512 would halve updates per env step and confound
the result); up to 10M additional steps instead of 3M; `--device cuda` for the PPO update
(opponent inference stays on CPU).

**Guard behaviour, known in advance:** seeded at 16.2% with margin 0.10 it only trips below
6.2% until some in-loop eval exceeds 16.2%; after that it tracks (best - 10 points). 004's
in-loop dips reached 10.5%, so a stop after a lucky 21%+ in-loop eval is possible.

**Budget:** up to 8 h wall clock (`timeout 8h`) or 10M additional steps, whichever first;
early stop only via `RegressionGuard`.

## Config

```
UV_PROJECT_ENVIRONMENT=.venv-cuda nohup timeout 8h uv run python -m rl.train \
  --resume rl_runs/selfplay/catan_bc_ft/catan_bc_ft_2000000.zip \
  --init-rate 0.162 \
  --selfplay-dir rl_runs/selfplay --label catan_bc_ft_long \
  --baseline-mix 0.5 --ent-coef 0.01 --learning-rate 1e-4 --n-steps 256 \
  --envs 16 --seed 3 --device cuda \
  --eval-opponents heuristic --eval-every 250000 --eval-episodes 200 \
  --regression-margin 0.10 --regression-patience 2 --steps 10000000
```

The three prerequisites this run needed (all in `rl/train.py`, none in `engine/`):
`--init-rate` (alias `--bc-init-rate`) and `--init-best` seed `RegressionGuard` on `--resume`
as well as `--bc-init`; on `--resume`, an explicitly passed `--learning-rate` / `--ent-coef`
that differs from the checkpoint's pickled value is a `RuntimeError`, `--n-steps` is applied
over the pickled value, and the effective values are logged; every checkpoint save refuses to
overwrite an existing file. Tests: `tests/rl/test_resume.py`, `tests/rl/test_training_wiring.py`.

## Environment

- catan commit: `b34670a` (prerequisites); pre-registration committed as `255da78` before launch.
- torch `2.14.0+cu130` (`.venv-cuda`, `torch.cuda.is_available()` confirmed, RTX 4050 Laptop),
  sb3-contrib `2.9.0`, stable-baselines3 `2.9.0`. PPO update on **CUDA** (`--device cuda`);
  self-play opponent inference and in-loop eval on CPU (see [005](005-gpu-inference.md)).
- 20 logical cores; the run had the machine to itself (load average ~1 at launch).
  `--envs 16`, `train_fps` 889-902 throughout (004: ~599 at `--envs 8`).
- Wall clock: started 12:15:59, finished 15:42:07 -- **3 h 26 min** (185.7 min training +
  20.4 min in-loop eval), well inside the 8 h budget. Ran the full 10M additional steps;
  `RegressionGuard` never fired (`stopped_early=False`).
- The resumed checkpoint's real `num_timesteps` was 2,031,616 (not 2,000,000), and each
  250k chunk overshoots by up to one 4096-step rollout, so the nominal checkpoint names differ
  slightly from the model's true step count. 004's `catan_bc_ft/` directory was snapshotted
  before the run and is byte- and mtime-identical after it.

## Result

### In-loop eval vs `HeuristicAgent`, n=200 (every line, as it arrived)

| Nominal steps | Win rate | Guard best |
|---|---|---|
| 2,281,616 | 12.0% [8.2%, 17.2%] | 16.2% (seed) |
| 2,531,616 | 11.5% [7.8%, 16.7%] | 16.2% |
| 2,781,616 | 15.0% [10.7%, 20.6%] | 16.2% |
| 3,031,616 | 12.0% [8.2%, 17.2%] | 16.2% |
| 3,281,616 | 14.5% [10.3%, 20.0%] | 16.2% |
| 3,531,616 | 14.5% [10.3%, 20.0%] | 16.2% |
| 3,781,616 | 11.0% [7.4%, 16.1%] | 16.2% |
| **4,031,616 (+2M)** | 14.5% [10.3%, 20.0%] | 16.2% |
| 4,281,616 | 20.0% [15.0%, 26.1%] | 20.0% |
| 4,531,616 | 14.5% [10.3%, 20.0%] | 20.0% |
| 4,781,616 | 13.0% [9.0%, 18.4%] | 20.0% |
| 5,031,616 | 13.5% [9.4%, 18.9%] | 20.0% |
| 5,281,616 | 16.5% [12.0%, 22.3%] | 20.0% |
| 5,531,616 | 17.0% [12.4%, 22.8%] | 20.0% |
| 5,781,616 | 17.5% [12.9%, 23.4%] | 20.0% |
| **6,031,616 (+4M)** | 22.5% [17.3%, 28.8%] | 22.5% |
| 6,281,616 | 22.0% [16.8%, 28.2%] | 22.5% |
| 6,531,616 | 19.0% [14.2%, 25.0%] | 22.5% |
| 6,781,616 | 18.0% [13.3%, 23.9%] | 22.5% |
| 7,031,616 | 22.5% [17.3%, 28.8%] | 22.5% |
| 7,281,616 | 21.0% [15.9%, 27.2%] | 22.5% |
| 7,531,616 | 24.0% [18.6%, 30.4%] | 24.0% |
| 7,781,616 | 22.0% [16.8%, 28.2%] | 24.0% |
| **8,031,616 (+6M)** | 20.5% [15.5%, 26.6%] | 24.0% |
| 8,281,616 | 19.0% [14.2%, 25.0%] | 24.0% |
| 8,531,616 | 18.0% [13.3%, 23.9%] | 24.0% |
| 8,781,616 | 22.5% [17.3%, 28.8%] | 24.0% |
| 9,031,616 | 22.0% [16.8%, 28.2%] | 24.0% |
| 9,281,616 | 22.5% [17.3%, 28.8%] | 24.0% |
| 9,531,616 | 23.5% [18.2%, 29.8%] | 24.0% |
| 9,781,616 | 18.0% [13.3%, 23.9%] | 24.0% |
| **10,031,616 (+8M)** | 25.5% [20.0%, 32.0%] | 25.5% |
| 10,281,616 | 21.5% [16.4%, 27.7%] | 25.5% |
| 10,531,616 | 27.0% [21.3%, 33.5%] | 27.0% |
| 10,781,616 | 22.0% [16.8%, 28.2%] | 27.0% |
| 11,031,616 | 22.5% [17.3%, 28.8%] | 27.0% |
| 11,281,616 | 21.0% [15.9%, 27.2%] | 27.0% |
| 11,531,616 | 20.5% [15.5%, 26.6%] | 27.0% |
| 11,781,616 | 18.0% [13.3%, 23.9%] | 27.0% |
| **12,031,616 (+10M)** | 27.5% [21.8%, 34.1%] | 27.5% |

Bold rows are the pre-registered fixed points. The in-loop curve visibly rises after about
+2M (first 10 evals mean ~14%, the last 20 mean ~21%), but every point still carries a
+/-6-point CI, and the final eval's 27.5% is the run's highest reading -- see the n=4000
number for the same checkpoint below.

### Fixed-point benchmarks vs 3 `HeuristicAgent`, n=4000 (the measurement)

Protocol identical to 004: seat-rotated, `engine_seed_base=1`, `driver_seed_base=1`,
`RLAgent(deterministic=True)`. Baseline = 004's 2M checkpoint (= this run's start), 648/4000.
p-values are `gamekit.mc.testing.two_proportion_test` against 648/4000; the last column is
significance after `benjamini_hochberg` at q=0.05 across the five comparisons.

| Fixed point | Checkpoint | Wins/n | Win rate | Wilson CI | vs start | p | BH |
|---|---|---|---|---|---|---|---|
| start | `catan_bc_ft_2000000` (004) | 648/4000 | 16.20% | [15.09%, 17.37%] | -- | -- | -- |
| +2M | `..._long_4031616` | 680/4000 | 17.00% | [15.87%, 18.20%] | +0.80 pt | 0.336 | no |
| +4M | `..._long_6031616` | 761/4000 | 19.03% | [17.84%, 20.27%] | +2.82 pt | 9.1e-4 | yes |
| +6M | `..._long_8031616` | 824/4000 | 20.60% | [19.38%, 21.88%] | +4.40 pt | 3.8e-7 | yes |
| +8M | `..._long_10031616` | 829/4000 | **20.72%** | [19.50%, 22.01%] | +4.52 pt | 1.8e-7 | yes |
| **+10M (gate)** | `..._long_12031616` | 796/4000 | 19.90% | [18.69%, 21.17%] | +3.70 pt | 1.7e-5 | yes |

Result JSONs: `experiments/results/benchmark_rl_vs_heuristic_p4_catan_bc_ft_long_{4031616,6031616,8031616,10031616,12031616}.json`.
Checkpoints not committed (`rl_runs/` gitignored).

Pairwise among the last three points (+6M, +8M, +10M): 20.60% vs 20.72% p=0.89; 20.72% vs
19.90% p=0.36; 20.60% vs 19.90% p=0.44 -- none distinguishable.

### Secondary numbers

- In-loop best = guard best = the +10M checkpoint (27.5% in-loop) benchmarked at **19.90%**
  [18.69%, 21.17%] at n=4000 -- 7.6 points below its in-loop reading, the same winner's-curse
  gap 004 showed (20.0% in-loop, 16.2% at n=4000), larger here because the in-loop maximum was
  taken over 40 evals.
- Highest n=4000 fixed point (+8M, `catan_bc_ft_long_10031616`) vs 3 `RandomAgent`, n=4000:
  **94.6%** [93.86%, 95.26%] (3784/4000) -- `benchmark_rl_vs_random_p4_catan_bc_ft_long_10031616.json`
  (004: 93.5% [92.69%, 94.22%]; the intervals overlap slightly at 93.86-94.22).

## Verdict

**Inconclusive under the pre-registered rule; leaning "real gain, then plateau".**

Applying the rule exactly as written: the gate point (+10M, 19.90%) beats the start
(16.2%) with p=1.7e-5 and non-overlapping CIs ([18.69%, 21.17%] vs [15.09%, 17.37%]), so the
significance half of *validated* is met. The second half -- the last three fixed points
non-decreasing -- is **not**: +6M 20.60% -> +8M 20.72% -> +10M 19.90%, a 0.82-point dip at the
end. So the rule returns *inconclusive* (its example of exactly this case: significant but
not monotone), and its prescribed next step is another leg, not a new note.

What the data says beyond the letter of the rule, stated as interpretation not as verdict:
the curve did bend *upward* for a while -- +0.8, +2.8, +4.4, +4.5 points over the start at
+2M/+4M/+6M/+8M (all but +2M significant after BH correction) -- and then flattened.
The three points from +6M on (20.60%, 20.72%, 19.90%) are pairwise indistinguishable
(p >= 0.36), and the dip (0.82 points) is well under the protocol's 2.3-point MDE, so it is
noise-consistent with a plateau at about 20-21%. The improvement over 004's best (16.2%) is
real by this protocol: every point from +4M on has a CI entirely above 004's; but it is
still well short of the 25% gate (best CI upper bound 22.01%).

Confounds to keep in view: (1) `--envs 16`/`--n-steps 256` differs from 004's 8 x 512 even
though the rollout size (4096) matches; (2) one seed (3), one continuation -- there is no
control leg of "004 as-is for 10 more steps", so the gain cannot be separated from
run-to-run variation of a single trajectory; (3) the pool was seeded from one checkpoint
rather than 004's twelve, which may have changed the opponent mix early on.

**Roadmap:** gate not met (no n=4000 CI reaches above 25%); the Phase 5 checkbox stays
unchecked.

## Notes / follow-up

- The strict monotonicity clause turned a plateau into "inconclusive". If a follow-up note
  reuses this rule, it might phrase the trend condition as a test on slope or on
  "no later point significantly below an earlier one" rather than on point estimates that
  differ by less than the MDE.
- The in-loop eval calls `evaluate_winrate` with a fixed `seed=cfg.seed + 977` every time
  (`rl/train.py:evaluate`), so successive in-loop evals may re-play the same 200 opening
  positions -- not verified against `evaluate_winrate`'s internals. If true, the in-loop
  curve's smoothness and its overshoot vs n=4000 would partly come from one fixed sample of
  games, which is worth checking before anyone reads the in-loop curve as a learning curve.
- Natural next steps this result motivates:
  [gamekit#010](https://github.com/guidodinello/gamekit/blob/main/docs/research/010-entropy-schedule.md)
  and [#011](https://github.com/guidodinello/gamekit/blob/main/docs/research/011-kl-guard.md),
  since simply running longer flattened at ~20%; and the BC-fidelity work named in
  [004](004-bc-warm-start.md).
