# Longer run / resume from the best 004 checkpoint (PR #24)

**Date:** 2026-09-29 (started)
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

(Results, environment and verdict are filled in after the run.)
