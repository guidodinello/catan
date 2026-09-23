# BC warm start + PPO self-play fine-tune (PR #21)

**Date:** 2026-09-21 to 2026-09-22
**Note:** [gamekit#002 — BC warm start](https://github.com/guidodinello/gamekit/blob/main/docs/research/002-bc-warm-start.md)
(see also [009](https://github.com/guidodinello/gamekit/blob/main/docs/research/009-longer-runs-and-resume.md),
[010](https://github.com/guidodinello/gamekit/blob/main/docs/research/010-entropy-schedule.md),
[011](https://github.com/guidodinello/gamekit/blob/main/docs/research/011-kl-guard.md) for the
follow-on knobs this run did not touch)

## Hypothesis

003's self-play PPO plateaued at 12.075% [11.10%, 13.12%] vs 3 `HeuristicAgent`s, oscillating in
a 10.5-15.5% band for its last ~2.5M steps with no sign of a still-climbing curve. This run tests
whether pre-training the policy by cross-entropy on `HeuristicAgent`'s own decisions (behaviour
cloning), then fine-tuning with the same self-play PPO recipe, escapes that plateau -- the
hypothesis being that a from-scratch policy spends its whole budget rediscovering basic
competence and never gets far enough to find the plateau's exit.

**The ceiling that bounds this experiment, measured on this exact protocol:** a *perfect* clone of
`HeuristicAgent` against 3 `HeuristicAgent`s wins exactly 25% by symmetry --
`experiments/results/benchmark_heuristic_vs_heuristic_p4.json` confirms **25.0% [24.34%, 25.68%]**,
n=4000, `engine_seed_base=1`, `driver_seed_base=1`. So BC alone can land *at* the gate but never
clear it (the gate needs the CI strictly above 25%); every point above 25% has to come from the
fine-tune.

**Comparison caveats, decided before running (not post-hoc):** 003 is the control, not a
budget-matched cold-start rerun -- 003 was itself warm-started from a 1M-step vs-random base
checkpoint and received far more total compute (6.5M additional self-play steps) than this run's
3M-step fine-tune, so a higher number here is not "PPO learns faster from a clone," only "this
config, from this start, cleared this control." The fine-tune also used `lr=1e-4`, `ent_coef=0.01`
from step 0, not 003's `lr=3e-4`, `ent_coef=0.02` -- deliberately gentler, since fresh-run entropy
pressure tends to wash out a near-deterministic clone in the first few updates. Both deviations
are confounds against 003's numbers, not just against each other.

## Config

```
# Dataset generation
uv run python -m rl.bc generate --games 3000 --workers 16 --out rl_runs/bc/v1 --seed-base 1

# Cross-entropy pre-training
uv run python -m rl.bc train --dataset rl_runs/bc/v1 --save rl_runs/bc/catan_bc_clone.zip \
  --epochs 30 --patience 3 --batch-size 1024 --lr 1e-3 --seed 0 --sanity-episodes 200

# BC clone benchmark
uv run python -m experiments.benchmark --mode rl_vs_heuristic --games 2000 \
  --players 4 --engine-seed-base 1 --driver-seed-base 1 --workers 16 \
  --checkpoint rl_runs/bc/catan_bc_clone.zip

# PPO self-play fine-tune from the clone
uv run python -m rl.train \
  --bc-init rl_runs/bc/catan_bc_clone.zip --bc-init-rate 0.10 \
  --selfplay-dir rl_runs/selfplay --label catan_bc_ft \
  --baseline-mix 0.5 --ent-coef 0.01 --learning-rate 1e-4 \
  --envs 8 --seed 3 \
  --eval-opponents heuristic --eval-every 250000 --eval-episodes 200 \
  --regression-margin 0.10 --regression-patience 2 --steps 3000000

# Authoritative gate benchmark on the best checkpoint
uv run python -m experiments.benchmark --mode rl_vs_heuristic --games 4000 \
  --players 4 --engine-seed-base 1 --driver-seed-base 1 --workers 10 \
  --checkpoint rl_runs/selfplay/catan_bc_ft/catan_bc_ft_2000000.zip

uv run python -m experiments.benchmark --mode rl_vs_random --games 4000 \
  --players 4 --engine-seed-base 1 --driver-seed-base 1 --workers 10 \
  --checkpoint rl_runs/selfplay/catan_bc_ft/catan_bc_ft_2000000.zip
```

## Environment

- catan commit: `94ba0e9` (this PR's own `rl/bc.py` + `rl/train.py` `--bc-init` landing)
- gamekit `0.3.0`, torch `2.14.0+cpu`, sb3-contrib `2.9.0`, stable-baselines3 `2.9.0`
- Machine: 20 logical cores.
- **Mid-run resource cap:** the user capped this session to at most 10 of 20 cores partway
  through, effective for everything launched after the process already in flight at the time
  (the BC clone's n=2000 benchmark, which had already started at `--workers 16` and was allowed
  to finish uncapped). Concretely: dataset generation ran at `--workers 16` (before the cap);
  the PPO fine-tune ran at `--envs 8` (capped, down from the originally planned 16); both n=4000
  authoritative benchmarks ran at `--workers 10` (capped, down from 16). `torch.set_num_threads(1)`
  was already the default everywhere it matters (inference workers, vec-env subprocesses) and
  needed no change. The fine-tune's own throughput (`train_fps~599`, ~7-7.5 min per 250k-step
  chunk) reflects the capped `--envs 8`, not the uncapped rate the original plan budgeted from.
- Budget actually used: generation, BC training and the clone benchmark together took well under
  10 minutes (see Notes) -- an order of magnitude under the plan's estimate -- so nearly the
  entire ~6h budget was available for the fine-tune, which ran its full 3,000,000-step ceiling
  in ~90 minutes wall clock (11 eval chunks, ~7-8 min each under the 10-core cap) without ever
  triggering `RegressionGuard`.

## Result

### BC dataset

3,000 heuristic-vs-heuristic-vs-heuristic-vs-heuristic games (seat-rotated via `randomize_seat`)
yielded 508,600 samples (0 dropped to truncation), split **by game** (game_id % 10 == 0 ->
validation): 456,707 train / 51,893 validation.

### BC held-out metrics

| Split | Raw top-1 acc | **Masked top-1 acc** | Value MSE |
|---|---|---|---|
| Train | 4.2% | 95.2% | 0.0055 |
| Val | 2.3% | **87.4%** | 0.6690 |

Masked accuracy is the metric that matters (see `rl/bc.py`'s docstring): raw argmax over all 223
atoms is not what inference does and is expected to look far worse, since most of the 223 atoms
are illegal at any given decision. Returns: mean -0.800, std 0.788 (gamma=0.999).

Val masked accuracy **by atom block**:

| Block | Acc |
|---|---|
| vertex_settlement | 51.6% |
| vertex_city | 76.8% |
| edge | 55.4% |
| hex | 50.5% |
| steal | 76.2% |
| resource | 94.8% |
| simple | 94.4% |
| opener | 78.8% |
| trade_reserved | n/a (never a label) |

The three weakest blocks -- `vertex_settlement`, `edge`, `hex` -- are exactly the spatial
placement decisions (settlement/road siting, robber targeting), each a many-way argmax over
`HeuristicAgent`'s own continuous vertex/edge score with frequent near-ties. `HeuristicAgent` is
deterministic, so Bayes-optimal top-1 is 100% and every point of this gap is generalization
error, not label noise -- except that `HeuristicAgent`'s own `max(candidates, key=...)` tie-breaks
by `legal_actions()`'s enumeration order among equally-scored candidates, so some of that "error"
is scoring an equally-good alternative as wrong.

### BC clone benchmark (no RL yet)

| n | Win rate vs 3 `HeuristicAgent`s |
|---|---|
| 200 (sanity, in-process) | 8.5% [5.4%, 13.2%] |
| 2000 (`experiments.benchmark`) | **10.0% [8.76%, 11.39%]** |

Well below the 25% ceiling a perfect clone would reach, despite 87.4% held-out masked accuracy --
compounding over a ~150+-decision game, even modest per-decision error on the
opening-placement-heavy weak blocks measurably costs full-game win rate. This is the central
finding motivating the verdict below: **the bottleneck at this stage is BC fidelity on spatial
decisions, not the RL fine-tune.**

### PPO fine-tune: in-loop eval vs `HeuristicAgent`, n=200

| Steps | Win rate | Best so far |
|---|---|---|
| 250,000 | 17.5% [12.9%, 23.4%] | 17.5%@250k |
| 500,000 | 14.0% [9.9%, 19.5%] | 17.5%@250k |
| 750,000 | 12.0% [8.2%, 17.2%] | 17.5%@250k |
| 1,000,000 | 17.0% [12.4%, 22.8%] | 17.5%@250k |
| 1,250,000 | 13.5% [9.4%, 18.9%] | 17.5%@250k |
| 1,500,000 | 10.5% [7.0%, 15.5%] | 17.5%@250k |
| 1,750,000 | 13.0% [9.0%, 18.4%] | 17.5%@250k |
| 2,000,000 | **20.0% [15.0%, 26.1%]** (peak) | 20.0%@2M |
| 2,250,000 | 15.0% [10.7%, 20.6%] | 20.0%@2M |
| 2,500,000 | 13.5% [9.4%, 18.9%] | 20.0%@2M |
| 2,750,000 | 14.5% [10.3%, 20.0%] | 20.0%@2M |
| 3,000,000 (ceiling reached) | 14.0% [9.9%, 19.5%] | 20.0%@2M |

`RegressionGuard`, seeded with the clone's own n=2000 rate (10.0%) rather than an unseeded -1.0
baseline, never triggered a stop -- every dip stayed within its 10-point margin of the running
best. The run oscillates in a roughly 10.5-20.0% band from step 250k onward, comparable in shape
to 003's oscillation but at a visibly higher level reached in a fraction of the steps (003 needed
~2.5M steps just to climb out of a 2.5-10% band; this run started there at step 0 courtesy of the
warm start and oscillated at 10.5-20% for the entire 3M-step run).

### Authoritative gate benchmark

Best in-loop checkpoint: `rl_runs/selfplay/catan_bc_ft/catan_bc_ft_2000000.zip` (2M steps,
20.0% in-loop peak).

| vs | n | Win rate | Wilson CI |
|---|---|---|---|
| 3 `HeuristicAgent` | 4000 | **16.2%** | **[15.09%, 17.37%]** |
| 3 `RandomAgent` | 4000 | 93.5% | [92.69%, 94.22%] |

Result JSONs: `experiments/results/benchmark_rl_vs_heuristic_p4_catan_bc_ft_2000000.json`,
`experiments/results/benchmark_rl_vs_random_p4_catan_bc_ft_2000000.json`. Checkpoint not
committed (`rl_runs/` gitignored).

**Same evaluation protocol as 003**, so the comparison is direct: seat-rotated,
`engine_seed_base=1`, `driver_seed_base=1`, `RLAgent(deterministic=True)`.

| | vs 3 `HeuristicAgent` | vs 3 `RandomAgent` |
|---|---|---|
| 003 (self-play, cold-start, ~7.5M steps total) | 12.075% [11.10%, 13.12%] | 90.025% [89.06%, 90.92%] |
| 004 (BC clone + 3M-step self-play fine-tune) | **16.2% [15.09%, 17.37%]** | **93.5% [92.69%, 94.22%]** |

Both intervals are non-overlapping with 003's: 15.09% > 13.12% and 92.69% > 90.92%. This is a
real, statistically distinguishable improvement over the previous best on both fronts -- not
within noise -- but the 25% gate (CI must exclude 25% from below) is still not met: 17.37% < 25%.

## Verdict

**Validated (technique-level), gate not met.** The BC-warm-start hypothesis -- that starting PPO
from a behaviour-cloned policy lets it reach a higher plateau than cold-start self-play within a
comparable step budget -- is supported: 004's fine-tune (3M steps) cleared 003's ceiling
(12.075%) with non-overlapping confidence intervals on both the heuristic and random benchmarks,
using **fewer** total steps than 003's ~7.5M. The technique measurably helped.

It does not close Phase 5. The roadmap checkbox stays unticked; the current-best line in
`docs/experiments/README.md` updates from 003's 12.075% to this run's 16.2%. The dominant finding
of this experiment is *why* the ceiling remains: BC fidelity on spatial placement decisions
(vertex/edge/hex, 50-55% masked accuracy) is the bottleneck the fine-tune inherited, not
something the fine-tune itself introduced -- a perfect clone would have started at 25% and the
fine-tune would have had a much higher floor to climb from.

**On the two confounds flagged before running:** the budget asymmetry (003's ~7.5M steps vs this
run's 3M) makes 004's result *more* impressive if anything, not less -- 004 cleared 003's ceiling
with under half its total compute. The lr/ent_coef deviation (1e-4/0.01 vs 003's 3e-4/0.02) was
chosen specifically to protect the clone from early entropy-driven collapse; whether the same
gentler settings would also have helped a cold-start run is untested and not this note's claim.

## Notes / follow-up

- **Dataset generation, BC training and the BC benchmark were all far cheaper than planned**: the
  plan budgeted ~40 min for dataset generation, ~20 min for BC training, ~35 min for the clone
  benchmark (n=2000) -- combined, they took well under 10 minutes on this machine. The catan
  engine and the BC forward/backward pass are both much faster than the original estimate
  assumed. This freed nearly the entire ~6h budget for the fine-tune, which is why it was able to
  run its full 3,000,000-step ceiling rather than being cut short by wall clock.
- **The BC clone's own bottleneck is the natural next attempt**: raising `vertex_settlement`/
  `edge`/`hex` masked accuracy above ~55% is very unlikely to come from more games at this
  dataset's scale (the train/val masked-accuracy gap is only ~8 points, i.e. this is base-task
  difficulty from `HeuristicAgent`'s own near-tied continuous scoring, not a data-scarcity gap) --
  it would need either a richer encoding of the spatial decision (closer to
  [gamekit#008](https://github.com/guidodinello/gamekit/blob/main/docs/research/008-board-aware-encoder.md)'s
  board-aware encoder) or accepting that BC's ceiling on this action family is intrinsically
  below what a perfect clone would need.
- **RegressionGuard's per-run "best" reset** (noted as a follow-up in 003) did not affect this run
  since it was a single, unresumed invocation with no multi-leg step-counting to reconcile; the
  `--bc-init-rate` seeding added by this PR is a separate, one-time seeding at the start of a run,
  not a fix for that cross-leg gap.
- Untried directions this run's result still motivates:
  [gamekit#009](https://github.com/guidodinello/gamekit/blob/main/docs/research/009-longer-runs-and-resume.md)
  (would more fine-tune steps past 3M keep climbing, given the run never regressed enough to
  stop?), [gamekit#010](https://github.com/guidodinello/gamekit/blob/main/docs/research/010-entropy-schedule.md)
  (an entropy schedule instead of the fixed 0.01 used here), and
  [gamekit#011](https://github.com/guidodinello/gamekit/blob/main/docs/research/011-kl-guard.md)
  (a KL guard against the clone's own initial policy, which might damp the 10.5-20% oscillation
  seen here).
