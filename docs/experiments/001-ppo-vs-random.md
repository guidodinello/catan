# PR #18 — MaskablePPO vs 3 RandomAgents

**Date:** 2026-09-20
**Note:** [gamekit#003 — Discount horizon vs episode length](https://github.com/guidodinello/gamekit/blob/main/docs/research/003-discount-horizon.md)

## Config

```
uv run python -m rl.train \
  --steps 1000000 --envs 16 --seed 1 \
  --opponents random \
  --eval-every 500000 --eval-episodes 200 \
  --label catan_ppo --run-dir rl_runs
```

Non-default hyperparameters: `gamma=0.999` (default value, but the value
this note exists to justify — see [gamekit#003](https://github.com/guidodinello/gamekit/blob/main/docs/research/003-discount-horizon.md)),
`ent_coef=0.01`, `net_arch=[256, 256]`, `ShapedReward` (potential-based
shaping on public victory points). Stopped early at 1M of a planned 8M
steps: the 8k-step smoke run already showed learning (49%→70% at
60k/120k steps), and the first two real checkpoints were stable and
decisive (87.5% [82.2%, 91.4%] at 500k, 84.0% [78.3%, 88.4%] at 1M —
overlapping CIs, not a fluke). See [gamekit#009](https://github.com/guidodinello/gamekit/blob/main/docs/research/009-longer-runs-and-resume.md)
for the "was this actually a plateau" question this stop left untested.

## Environment

- catan commit: `e772d1e6a62f` (PR #17 merge; PR #18's own training and
  gate benchmark ran before that PR's commit landed)
- gamekit: `0.2.0` at run time (0.3.0's run-scoping fix, gamekit#23, had
  not yet shipped)
- torch `2.14.0+cpu`, sb3-contrib `2.9.0`, stable-baselines3 `2.9.0`

## Result

**81.75% [80.5%, 82.9%] win rate vs 3 `RandomAgent`s, 4p, n=4000**,
seat-rotated (z=-95.7 vs the 25% null, p≈0). Training throughput: 1636
train_fps at 1M steps (1444 at 500k, still warming up), ~10.2 min of
actual training wall-clock.

Result JSON: `experiments/results/benchmark_rl_vs_random_p4.json`
(checkpoint `catan_ppo_1000000.zip`, not committed — binary artifact,
`rl_runs/` is gitignored).

A real oversubscription bug was found and fixed running this gate
benchmark — see [gamekit#007](https://github.com/guidodinello/gamekit/blob/main/docs/research/007-inference-thread-oversubscription.md)
for the full writeup (`>300s` timeout → `7.81s`, bit-identical win counts
across worker counts after the fix).

## Verdict

**Validated.** The M1 gate (">25% vs random, Wilson CI excluding it") was
cleared by 55 points, not marginally. This is the base checkpoint both
self-play attempts ([002](002-selfplay-v1-baseline-mix-0.2.md),
[003](003-selfplay-v2-baseline-mix-0.5.md)) resume from.
