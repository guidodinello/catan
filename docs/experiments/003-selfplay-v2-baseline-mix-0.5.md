# Self-play v2 — baseline_mix 0.5, ent_coef 0.02

**Date:** 2026-09-20 to 2026-09-21
**Note:** [gamekit#001 — Self-play opponent mix vs a fixed baseline](https://github.com/guidodinello/gamekit/blob/main/docs/research/001-self-play-opponent-mix.md)

## Config

```
# Leg 1 (fresh resume from the PR #18 base, killed by a session interruption
# at local step 4.5M of its own 5M budget -- not a deliberate stop)
uv run python -m rl.train \
  --steps 5000000 --envs 16 --seed 3 \
  --resume rl_runs/catan_ppo_1000000.zip \
  --selfplay-dir rl_runs/selfplay --label catan_selfplay_v2 \
  --baseline-mix 0.5 --ent-coef 0.02 \
  --eval-opponents heuristic --eval-every 250000 --eval-episodes 200 \
  --regression-margin 0.10 --regression-patience 2

# Leg 2 (resumed from leg 1's last checkpoint, ran to completion)
uv run python -m rl.train \
  --steps 2000000 --envs 16 --seed 3 \
  --resume rl_runs/selfplay/catan_selfplay_v2/catan_selfplay_v2_4500000.zip \
  --selfplay-dir rl_runs/selfplay --label catan_selfplay_v2 \
  --baseline-mix 0.5 --ent-coef 0.02 \
  --eval-opponents heuristic --eval-every 250000 --eval-episodes 200 \
  --regression-margin 0.10 --regression-patience 2
```

Retry of [002](002-selfplay-v1-baseline-mix-0.2.md) with `baseline_mix`
raised 0.2→0.5 and `ent_coef` raised 0.01→0.02, informed by a TensorBoard
diagnostic on v1's plateau (reproduced below) that ruled out collapse as
the cause. Resumed (warm-started) from the same [001](001-ppo-vs-random.md)
base checkpoint as v1, with a fresh, run-scoped self-play pool
(`rl_runs/selfplay/catan_selfplay_v2/`, seeded from the base checkpoint).

**Both legs ran on the pre-fix code** for cumulative step counting (this
PR's step 2 fix landed only after leg 2 finished, prompted by exactly the
collision this run produced). Each leg's own `done`/log line counts steps
*local to that invocation*, not cumulative -- and leg 2's early
checkpoints (`..._250000.zip` through `..._2000000.zip`) **silently
overwrote leg 1's checkpoints of the same name** (leg 1's `..._2250000.zip`
through `..._4500000.zip` survived, since those step numbers don't overlap
leg 2's own 250k-2M range). The table below reconstructs true cumulative
steps from the run logs; the checkpoint filenames on disk no longer agree
with what a naive reading of their numbers would suggest.

## Environment

- catan commit: `e7694f6a68c2` (PR #18 merge) — branch `phase5-rl-selfplay`
- gamekit: `0.3.0` (`run_id` scoping from gamekit#23 available and used —
  `run_id=cfg.label` — but the *step-counting* bug above is catan-side,
  not related to gamekit's fix)
- torch `2.14.0+cpu`, sb3-contrib `2.9.0`, stable-baselines3 `2.9.0`

## Result

In-loop eval vs `HeuristicAgent`, n=200. True cumulative steps = 1,000,000
(PR #18 base) + leg 1's own step count + leg 2's own step count (leg 2
resumed from wherever leg 1 stopped, 4,500,000):

| Leg | Local step | True cumulative | Win rate |
|---|---|---|---|
| 1 | 250,000 | 1,250,000 | 2.5% [1.1%, 5.7%] |
| 1 | 500,000 | 1,500,000 | 5.0% [2.7%, 9.0%] |
| 1 | 750,000 | 1,750,000 | 7.0% [4.2%, 11.4%] |
| 1 | 1,000,000 | 2,000,000 | 6.5% [3.8%, 10.8%] |
| 1 | 1,250,000 | 2,250,000 | 9.5% [6.2%, 14.4%] |
| 1 | 1,500,000 | 2,500,000 | 7.0% [4.2%, 11.4%] |
| 1 | 1,750,000 | 2,750,000 | 10.5% [7.0%, 15.5%] |
| 1 | 2,000,000 | 3,000,000 | 10.0% [6.6%, 14.9%] |
| 1 | 2,250,000 | 3,250,000 | 11.0% [7.4%, 16.1%] |
| 1 | 2,500,000 | 3,500,000 | 11.5% [7.8%, 16.7%] |
| 1 | 2,750,000 | 3,750,000 | 11.5% [7.8%, 16.7%] |
| 1 | 3,000,000 | 4,000,000 | 12.5% [8.6%, 17.8%] |
| 1 | 3,250,000 | 4,250,000 | **15.0% [10.7%, 20.6%]** (leg-1 peak) |
| 1 | 3,500,000 | 4,500,000 | 13.0% [9.0%, 18.4%] |
| 1 | 3,750,000 | 4,750,000 | 12.0% [8.2%, 17.2%] |
| 1 | 4,000,000 | 5,000,000 | 12.0% [8.2%, 17.2%] |
| 1 | 4,250,000 | 5,250,000 | 13.0% [9.0%, 18.4%] |
| 1 | 4,500,000 (leg 1 ends: session interruption) | 5,500,000 | 12.5% [8.6%, 17.8%] |
| 2 | 250,000 | 5,750,000 | 15.0% [10.7%, 20.6%] |
| 2 | 500,000 | **6,000,000** | **15.5% [11.1%, 21.2%]** (overall peak) |
| 2 | 750,000 | 6,250,000 | 14.0% [9.9%, 19.5%] |
| 2 | 1,000,000 | 6,500,000 | 12.5% [8.6%, 17.8%] |
| 2 | 1,250,000 | 6,750,000 | 10.5% [7.0%, 15.5%] |
| 2 | 1,500,000 | 7,000,000 | 12.5% [8.6%, 17.8%] |
| 2 | 1,750,000 | 7,250,000 | 15.0% [10.7%, 20.6%] |
| 2 | 2,000,000 (leg 2 ends: full budget) | 7,500,000 | 10.5% [7.0%, 15.5%] |

**Correction to the numbers reported live during this run**: mid-run
narration called the overall-peak checkpoint "5.0M cumulative" — that
figure omitted the 1,000,000-step base entirely (it was leg 1's own local
`4,500,000` plus leg 2's local `500,000` = 5,000,000, without adding the
base). The correct cumulative figure, used throughout this log, is
**6,000,000**.

`RegressionGuard` never triggered a stop in either leg — every dip stayed
within its 10-point margin of that leg's own best (each leg's guard starts
fresh; it does not carry a memory of the other leg's best, a known
limitation, see "Notes" below).

### Authoritative gate benchmark

Run against the overall-best checkpoint (leg 2's local `500000.zip`, true
cumulative 6,000,000 — on disk as
`rl_runs/selfplay/catan_selfplay_v2/catan_selfplay_v2_500000.zip`):

```
uv run python -m experiments.benchmark --mode rl_vs_heuristic --games 4000 \
  --players 4 --engine-seed-base 1 --driver-seed-base 1 --workers 16 \
  --checkpoint rl_runs/selfplay/catan_selfplay_v2/catan_selfplay_v2_500000.zip

uv run python -m experiments.benchmark --mode rl_vs_random --games 4000 \
  --players 4 --engine-seed-base 1 --driver-seed-base 1 --workers 16 \
  --checkpoint rl_runs/selfplay/catan_selfplay_v2/catan_selfplay_v2_500000.zip
```

| vs | n | Win rate | Wilson CI |
|---|---|---|---|
| 3 `HeuristicAgent` | 4000 | 12.075% | [11.10%, 13.12%] |
| 3 `RandomAgent` | 4000 | 90.025% | [89.06%, 90.92%] |

Result JSONs:
`experiments/results/benchmark_rl_vs_heuristic_p4_catan_selfplay_v2_500000.json`,
`experiments/results/benchmark_rl_vs_random_p4_catan_selfplay_v2_500000.json`.
Checkpoint not committed (binary artifact, `rl_runs/` gitignored).

Notably, self-play **improved** the vs-random number over
[001](001-ppo-vs-random.md)'s 81.75% — 6.5M additional steps of training,
even without moving the needle vs `HeuristicAgent`, generalized to a
stronger random-opponent policy.

## Verdict

**Learning confirmed; 25% gate not met.** `baseline_mix=0.5` clearly beat
v1's `0.2` (8.5-9% plateau vs a 10.5-15.5% band, non-overlapping at the
low end) — the diagnostic-informed adjustment worked. But the authoritative
n=4000 result, **12.075% [11.10%, 13.12%] vs `HeuristicAgent`**, is
decisively below the >25% gate with a tight interval; the in-loop
oscillation between 10.5% and 15.5% across the last ~2.5M steps of leg 2
shows a real ceiling for this configuration within the budget used, not a
still-climbing curve. See [gamekit#009](https://github.com/guidodinello/gamekit/blob/main/docs/research/009-longer-runs-and-resume.md)
for the open question of whether more steps at the same settings would
eventually break out.

## Notes / follow-up

- The step-counting bug this run exposed is fixed in `rl/train.py` as of
  this PR (`_resume_step_bookkeeping`), tested in
  `tests/rl/test_selfplay.py`. A future resumed run will not repeat this
  filename collision.
- `RegressionGuard`'s "best" tracking resets to a fresh instance per
  invocation (does not persist across a resume). Not fixed in this PR —
  it did not affect this run's outcome (no stop was ever warranted), but a
  resumed run currently cannot detect a regression relative to a *prior
  leg's* peak, only within its own. Worth a follow-up if self-play
  attempts keep needing multi-leg resumes.
- Untried directions this run's ceiling motivates:
  [gamekit#008](https://github.com/guidodinello/gamekit/blob/main/docs/research/008-board-aware-encoder.md)
  (board-aware encoder),
  [gamekit#010](https://github.com/guidodinello/gamekit/blob/main/docs/research/010-entropy-schedule.md)
  (entropy schedule instead of a fixed coefficient),
  [gamekit#011](https://github.com/guidodinello/gamekit/blob/main/docs/research/011-kl-guard.md)
  (KL guard vs the previous snapshot — deliberately deferred from this PR's
  scope as a higher-effort control).
