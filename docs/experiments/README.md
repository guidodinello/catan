# Experiments

Run logs for catan's Phase 5 (RL) work: the numbers, exact CLI
invocations, and result JSON paths for each attempt. The hypothesis and
literature behind each attempt live one layer up, in
[gamekit's `docs/research/`](https://github.com/guidodinello/gamekit/tree/main/docs/research)
— see that repo's README for the two-layer convention this directory
follows (a log here always links back to the gamekit note it tests; a note
there links forward to the log(s) that tested it).

## Index

| id | title | verdict | gamekit note |
|---|---|---|---|
| [001](001-ppo-vs-random.md) | MaskablePPO vs 3 RandomAgents (PR #18) | validated | [003](https://github.com/guidodinello/gamekit/blob/main/docs/research/003-discount-horizon.md) |
| [002](002-selfplay-v1-baseline-mix-0.2.md) | Self-play v1 — baseline_mix 0.2 | rejected (this value) | [001](https://github.com/guidodinello/gamekit/blob/main/docs/research/001-self-play-opponent-mix.md) |
| [003](003-selfplay-v2-baseline-mix-0.5.md) | Self-play v2 — baseline_mix 0.5, ent_coef 0.02 | learning confirmed, gate not met | [001](https://github.com/guidodinello/gamekit/blob/main/docs/research/001-self-play-opponent-mix.md) |
| [004](004-bc-warm-start.md) | BC warm start + PPO self-play fine-tune (PR #21) | validated (technique-level), gate not met | [002](https://github.com/guidodinello/gamekit/blob/main/docs/research/002-bc-warm-start.md) |
| [005](005-gpu-inference.md) | GPU (CUDA) for Phase 5 RL -- opponent-checkpoint inference, PPO update, BC training | opponent-checkpoint inference server **rejected** (regresses throughput); PPO update and BC training **adopted as opt-in** `--device cuda` | (candidate, not yet filed) "GPU inference servers don't automatically transfer across engines" |
| [006](006-longer-run.md) | Longer run / resume from 004's best checkpoint (+10M steps, PR #24) | inconclusive under the pre-registered rule (significant gain, +6M/+8M/+10M plateau at ~20-21%, last-three-monotone clause failed by 0.82 pt) | [009](https://github.com/guidodinello/gamekit/blob/main/docs/research/009-longer-runs-and-resume.md) |
| [007](007-human-games.md) | Human vs bots in the web GUI (issue #26): 12 games each vs `rl` and `heuristic` | pending (pre-registered, no games played yet) | [005](https://github.com/guidodinello/gamekit/blob/main/docs/research/005-eval-statistics.md) (statistics); new note proposed |
| [008](008-trading-opponents-baseline.md) | Trading opponents: `TradingHeuristicAgent`, trade-aware benchmark, baselines (issue #28 slice 1) | trading helps a heuristic (53.4% [51.85%, 54.94%] in 2 vs 2); RL forced-reject baseline vs 3 trading heuristics 18.9% [17.72%, 20.14%] | [005](https://github.com/guidodinello/gamekit/blob/main/docs/research/005-eval-statistics.md) (statistics); prerequisite for [012](https://github.com/guidodinello/gamekit/blob/main/docs/research/012-trade-heads.md) / [020](https://github.com/guidodinello/gamekit/blob/main/docs/research/020-modular-trade-agent.md) |
| [009](009-critic-calibration.md) | Critic calibration: does V(s) of `catan_bc_ft_long_10031616` rank positions, and does ΔV price trades? (issue #37) | V⁺ = V + own VP/10 ranks positions (headline AUC 0.83 [0.816, 0.847] vs 3 heuristic; 0.83 [0.814, 0.845] vs 3 trading heuristic) and is more than a VP counter (stratified AUC 0.67, held-out ΔAUC +0.04), but **#38 is not a go under the pre-registered rule**: ΔV ranks accepts only weakly (Spearman +0.04 [+0.002, +0.077]; Δwin(ΔV>0 vs ≤0) +0.6 pts [-0.03, +1.3]) and accepting the average offer is neutral (-0.09 pts) | [005](https://github.com/guidodinello/gamekit/blob/main/docs/research/005-eval-statistics.md) (statistics); prerequisite check for [020](https://github.com/guidodinello/gamekit/blob/main/docs/research/020-modular-trade-agent.md) / [021](https://github.com/guidodinello/gamekit/blob/main/docs/research/021-decision-time-search.md) |

## In-loop eval caveats

The in-loop eval (`rl/train.py:evaluate_in_loop`, n=200 by default) is a monitoring signal and
a checkpoint *shortlist*, never a result. Cite this section from a log's Environment section
("in-loop eval: seed scheme per README §In-loop eval caveats, catan commit `<sha>`").

- **Seed scheme (from issue #25's fix).** Every chunk runs two n=200 evals from one CPU model
  copy. The **guard** eval uses the fixed seed `cfg.seed + 977`; only `RegressionGuard`'s stop
  decision reads it. The **reported** eval uses `cfg.seed + 977 + step` (`step` = cumulative
  `num_timesteps`-based `done`, so a `--resume` leg never replays the previous leg's seeds);
  it alone picks `TrainResult.best_checkpoint` / `best_win_rate`. Both rates are logged
  (`reported=` and `guard(fixed)=`).
- **Experiments 001-006 used the fixed seed for everything**, including best-checkpoint
  selection: every in-loop eval replayed the same 200 setups. Their in-loop rates are biased
  toward checkpoints that suit those boards (004: 20.0% in-loop vs 16.2% at n=4000; 006: 27.5%
  vs 19.9%).
- **Why the guard stays fixed.** With independent draws at a flat true rate of 21%, the guard
  (margin 0.10, patience 2, seeded at 16.2%) falsely stops in ~24% of 40-eval runs and ~64% of
  93-eval runs (pure-Python simulation, iid n=200 binomials -- an upper bound, since a fixed
  set is only partly paired: dice and steals share the game RNG, so they drift once policies
  differ; board, dev deck, starting player, seat and opponent seeds stay fixed).
- **This does not close the winner's-curse gap.** The best reported rate is still the max over
  ~40 noisy n=200 evals, so it stays biased upward. **n=4000 confirmation of the chosen
  checkpoint remains mandatory.**
- **Cost.** The second eval doubles in-loop eval time: ~31 s per n=200 eval (006: 20.4 min over
  40 evals vs 185.7 min training), so about +20 min on a 10M-step run (~4% of an 8 h budget),
  or ~9% fewer training steps on a timeout-bound 8 h run.

## Current best (Phase 5, as of 2026-09-29)

**20.72% [19.50%, 22.01%] win rate vs 3 `HeuristicAgent`s**, n=4000,
seat-rotated — [006](006-longer-run.md), `catan_bc_ft_long_10031616` (the highest of the five
pre-registered n=4000 fixed points; the +10M checkpoint's 19.90% [18.69%, 21.17%] is
statistically indistinguishable from it). Non-overlapping with [004](004-bc-warm-start.md)'s
16.2% [15.09%, 17.37%]. Still below the Phase-5-done gate (>25% with the CI excluding it);
the roadmap checkbox stays unchecked. Beats 3 `RandomAgent`s at **94.6% [93.86%, 95.26%]**.
The 20.72% is the best of five fixed points, so it carries a small selection bias; every
fixed point from +4M on clears 004's interval regardless.
