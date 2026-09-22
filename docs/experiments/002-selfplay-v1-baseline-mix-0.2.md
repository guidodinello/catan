# Self-play v1 — baseline_mix 0.2

**Date:** 2026-09-20
**Note:** [gamekit#001 — Self-play opponent mix vs a fixed baseline](https://github.com/guidodinello/gamekit/blob/main/docs/research/001-self-play-opponent-mix.md)

## Config

```
# Smoke (200k steps)
uv run python -m rl.train \
  --steps 200000 --envs 16 --seed 2 \
  --resume rl_runs/catan_ppo_1000000.zip \
  --selfplay-dir rl_runs/selfplay_smoke --label catan_selfplay_smoke \
  --baseline-mix 0.2 --ent-coef 0.02 \
  --eval-opponents heuristic --eval-every 50000 --eval-episodes 150 \
  --regression-margin 0.10 --regression-patience 2

# Continuation (resumed from the smoke's own final checkpoint)
uv run python -m rl.train \
  --steps 5000000 --envs 16 --seed 2 \
  --resume rl_runs/selfplay_smoke/catan_selfplay_smoke/catan_selfplay_smoke_final.zip \
  --selfplay-dir rl_runs/selfplay_smoke --label catan_selfplay_smoke \
  --baseline-mix 0.2 --ent-coef 0.02 \
  --eval-opponents heuristic --eval-every 250000 --eval-episodes 200 \
  --regression-margin 0.10 --regression-patience 2
```

`baseline_factory=HeuristicAgent`, resumed (warm-started) from
[001](001-ppo-vs-random.md)'s checkpoint rather than training self-play
from scratch. Stopped manually partway through the continuation's 5M-step
budget (killed at local step 2.5M) after the eval curve plateaued in the
6-9% band with no sustained climb.

Bookkeeping note: this run predates the `rl/train.py` fix for cumulative
step counting on resume (this PR's step 2). The continuation invocation's
own logged "steps=X/5000000" is **local to that invocation**, not
cumulative — the table below converts to cumulative steps on top of the
1M-step base for readability; the raw logs use the local numbering.

## Environment

- catan commit: `e7694f6a68c2` (PR #18 merge) — branch `phase5-rl-selfplay`
- gamekit: `0.2.0` at run time (before the `run_id` scoping fix landed
  mid-PR)
- torch `2.14.0+cpu`, sb3-contrib `2.9.0`, stable-baselines3 `2.9.0`

## Result

In-loop eval vs `HeuristicAgent` (n=150 for the smoke, n=200 for the
continuation), cumulative steps on top of the 1M-step base from
[001](001-ppo-vs-random.md):

| Cumulative steps | Win rate |
|---|---|
| 1.05M (smoke) | 3.3% [1.4%, 7.6%] |
| 1.10M (smoke) | 8.0% [4.6%, 13.5%] |
| 1.15M (smoke) | 4.7% [2.3%, 9.3%] |
| 1.20M (smoke, final) | 6.7% [3.7%, 11.8%] |
| 1.45M | 4.0% [2.0%, 7.7%] |
| 1.70M | 2.5% [1.1%, 5.7%] |
| 1.95M | 5.5% [3.1%, 9.6%] |
| 2.20M | 8.5% [5.4%, 13.2%] |
| 2.45M | 5.5% [3.1%, 9.6%] |
| 2.70M | 8.5% [5.4%, 13.2%] |
| 2.95M | 8.5% [5.4%, 13.2%] (third identical eval in a row) |
| 3.20M | 6.0% [3.5%, 10.2%] |
| 3.45M | 6.5% [3.8%, 10.8%] |
| 3.70M (final, run killed here) | 9.0% [5.8%, 13.8%] (new best) |

Total self-play steps actually run: 2.7M (200k smoke + 2.5M of the
continuation's 5M budget) on top of the 1M base = 3.7M total RL steps.
No authoritative (n=4000, seat-rotated) benchmark was run for v1 — it was
superseded by v2 before a gate run was justified, and its checkpoints
(`rl_runs/selfplay_smoke/`, gitignored, local-only) have since been
overwritten or discarded.

**TensorBoard diagnostic**, read from `rl_runs/tb/MaskablePPO_3` (the
continuation invocation) before deciding how to proceed, at the point of
the three-identical-eval plateau (cumulative 2.2M-2.95M):

| Metric | Early | Recent |
|---|---|---|
| `entropy_loss` | -0.287 | -0.168 |
| `approx_kl` | 0.004 | 0.002-0.003 |
| `clip_fraction` | 0.018-0.028 | 0.016-0.031 |
| `explained_variance` | 0.65-0.81 | 0.79-0.91 |
| `ep_rew_mean` | -0.42 to -0.71 | -0.31 to -0.46 |
| `ep_len_mean` | 207-226 | 208-226 |

None of truco-py's collapse signatures (entropy→~0, `ep_len_mean` 2→165,
`ep_rew_mean` unbounded blowup) are present — training was stable
throughout, it just wasn't winning.

## Verdict

**Rejected** (for this hyperparameter value): `baseline_mix=0.2` oscillates
in a 2.5-9% band vs `HeuristicAgent` through 3.7M total steps, far short
of the >25% gate, with healthy training dynamics — a genuine ceiling, not
a collapse. Directly informed the `baseline_mix=0.5` retry in
[003](003-selfplay-v2-baseline-mix-0.5.md), which reached a materially
higher band. See [gamekit#001](https://github.com/guidodinello/gamekit/blob/main/docs/research/001-self-play-opponent-mix.md)'s
counter-evidence section: truco-py collapsed at *both* 0.2 and 0.5, so the
mix value alone was never expected to be sufficient by itself — the
difference here is `RegressionGuard` actually watching the curve, which
neither of truco's runs had.
