# Critic calibration: does V(s) rank positions, and does ΔV price trades? (issue #37)

**Date:** pre-registered 2026-09-29; measured runs: pending
**Note:** [gamekit#005 — Eval statistics: Wilson intervals](https://github.com/guidodinello/gamekit/blob/main/docs/research/005-eval-statistics.md) (statistics). Prerequisite check for [020 — modular trade agent](https://github.com/guidodinello/gamekit/blob/main/docs/research/020-modular-trade-agent.md) (stage 1, value-priced responses = catan #38) and [021 — decision-time search](https://github.com/guidodinello/gamekit/blob/main/docs/research/021-decision-time-search.md) (critic as leaf evaluator). Neither note is *tested* by this log; it measures the evaluator both would use. A gamekit follow-up (link this log from 020 and 021) is proposed in the PR, not edited there.

**Status: pre-registered.** This section and the code were committed before any measured run. Smoke runs are disclosed below.

## Hypothesis

The critic of `catan_bc_ft_long_10031616` was trained on a shaped reward (potential-based public VP, gamma 0.999) and only on the rl seat's own on-policy decision states in trade-free games. Two things could make it a poor evaluator: it may be mostly a public-VP counter, and it has never seen a state produced by a trade. We measure, in order:

1. **Rank quality.** Does V(s) at the rl seat's decisions predict that seat's eventual win (AUC, rank correlation, reliability), against 3 `HeuristicAgent` and against 3 `TradingHeuristicAgent`?
2. **VP counter.** How much of that is public VP (own VP, lead over the best opponent) alone?
3. **Off-distribution probe.** At states where a `TradingHeuristicAgent` offer reached the rl seat, what is ΔV = V(after accepting) − V(offer withdrawn)? Does ΔV predict what accepting does to the win rate (paired playouts)? This is the evidence #38 needs for its margin.

### The scale of V (why two scores)

`rl/reward.py:ShapedReward` uses `phi = public VP / 10`, `phi(terminal) = 0`, gamma 0.999, and `rl/train.py` uses no reward normalisation (`VecNormalize`/scaling not used; checked at pre-registration). The shaping terms telescope, so V(s) estimates roughly `E[gamma^N · (±1)] − phi(s)`: it is **not a probability**, it is shrunk toward 0 by gamma^N, and raw V carries a *negative* own-VP term. We therefore report two scores everywhere:

- **V**: the raw critic output;
- **V⁺ = V + own public VP / 10**: the VP-adjusted score (public VP, matching `ShapedReward`, never hidden VP cards).

A domestic trade changes no VP, so phi cancels in ΔV and #38's margin can be stated on raw V. The naive map `p̂ = (V⁺ + 1) / 2` is shown against the empirical win rate in the reliability tables only to document that it is *not* calibrated.

## Design

**Rl seat and tables.** The benchmark's own lineups and seat rotation (`experiments.benchmark.benchmark_agent_factory`, `engine_seed % 4`): arm H = `rl_vs_heuristic`, arm T = `rl_reject_vs_trading_heuristic` (the seat exactly as the web GUI builds it: trade responses forced to reject, never proposes). `ValueRecorder` wraps the rl seat; it changes no action and draws from no game RNG (a unit test replays the same seeds with and without recorder and forks and requires identical `GameRecord`s). Critic readings use `policy.predict_values` on CPU with the same cached model as the agent.

**Checkpoint.** `rl_runs/selfplay/catan_bc_ft_long/catan_bc_ft_long_10031616.zip` (main checkout, read-only), sha256 prefix `e8001d3e81ba`.

**Seeds** (engine seed base = driver seed base, consecutive, length divisible by 4 so the rotation is exact):

| use | base | n |
|---|---|---|
| smoke / timing (disclosed) | 500001 | ≤200 |
| **arm H measured** | 1000001 | 4000 |
| **arm T measured** (+ probe + forks) | 2000001 | 4000 |
| reserved for #38 margin tuning (not used here) | 3000001 | — |

**Off-limits** (used by an n=4000 benchmark or 008's smoke): engine/driver seeds 1..10000 and 90001..100000. The 4000 benchmark boards (006, 008) are therefore never seen here, so later margin tuning on 3000001+ touches neither them nor these.

**Decision states.** Every rl decision that is not a trade response ("top-level decision", including setup). Recorded: V, own public VP, lead (own VP − best opponent's public VP), own-turn index (count of the seat's own ROLL decisions), phase. Outcome: whether the rl seat won (games without a winner count as a loss; the benchmarks have had none).

**Headline sample.** One uniformly random top-level decision per game (`random.Random(driver_seed ^ 0x9E3779B1)`): about 4000 independent states per arm. All-states analyses are secondary and use a game-clustered bootstrap.

**Offer probe (arm T, every offer).** At each `AWAIT_TRADE_RESPONSE` where the rl seat is the acting responder, may `AcceptTrade`, and the offer is fresh (not a counter), on copies (`state.copy()`, the engine's public `apply_action`; the engine is untouched):

- `after`: the rl seat accepts;
- `before`: the rl seat and every remaining responder reject (offer withdrawn);
- both end in MAIN on the proposer's turn with the offer cleared, and the harness raises unless they differ only in the two traded hands by exactly the bundle;
- ΔV = V(after) − V(before), both encoded from the rl seat's view. Also recorded: V at the raw pending state (to size the off-distribution shift), ΔV_gift (receive without paying), ΔV_pay (pay without receiving), cards in/out, drop in the seat's shortfall to its next build (`agents.trading_heuristic.next_build_cost`), proposer VP.

This construction is the exact evaluator #38 would use. Both states are opponent-turn states, which were never learner decision points in training.

**Paired fork (arm T, the interventional answer).** For one uniformly sampled offer per game (reservoir sampling, snapshot = `state.copy()` + `copy.deepcopy` of the unwrapped agents; the torch model is not an agent attribute and is shared through `_load_model`'s per-process cache), play the game on from both branches, **K = 16 pairs**, common random numbers (both branches of pair k reseeded with the same seed): the rl seat accepts vs rejects, then every later responder acts by its own policy, the same step budget. Δwin_i = mean over pairs of (win if accept − win if reject). The reject branch is a real reject (a later responder may still accept and trade with the proposer), whereas `before` for ΔV is "offer withdrawn": a known asymmetry of the design, noted in the result. Forks run in the worker after the main game ends and cannot change the recorded game.

## Pre-registered analysis and verdict rules

Frozen numbers: bootstrap B = 2000 for the headline and fork statistics (rows independent: one state / one offer per game), B = 200 for all-states statistics (resampling whole games), seed 20260929; turn strata by own-turn count with cut points **12 and 24** (the terciles of the smoke run, which read no win rate); leader-VP strata with cut points 5 and 7; reliability tables in decile bins of the score within each arm.

**Headline (one state per game):** AUC of V⁺ and of V against win with bootstrap 95% CIs; Spearman(V⁺, win), Spearman(V⁺, final own VP); reliability of V⁺ and V (mean score, win rate, Wilson 95% interval per decile; intervals are valid because the rows are independent); the `p̂ = (V⁺+1)/2` comparison.

**Is it a VP counter?** AUC of own VP alone and of lead alone; Spearman(V⁺, lead) and Spearman(V, own VP); **stratified AUC of V⁺ within (own VP × lead) cells**, pair-count weighted; AUC by turn band and by table-leader-VP band; held-out logistic regression (numpy IRLS, fitted on odd game indices, scored on even ones): win ~ lead + own VP + turn, with vs without V⁺, ΔAUC with a game-clustered bootstrap CI.

**Offer probe (arm T):** ΔV quantiles and share > 0; by trade shape (pays more cards than it gets vs not) and by shortfall-drop class; ΔV_gift / ΔV_pay as scale checks; V_pending − V_before.

**Fork (arm T):** Spearman(ΔV, Δwin) with bootstrap CI; mean Δwin per ΔV quintile with bootstrap CI; mean Δwin for ΔV > 0 minus for ΔV ≤ 0 with paired-sample bootstrap CI; an exploratory margin curve (share accepted and mean Δwin for accept-iff-ΔV > m), flagged as input to #38 only and not a tuned result. `two_proportion_test` is not used (the branches are paired).

**Verdict rules** (`verdicts` in the result JSON):

- **V ranks positions:** the lower bound of the headline AUC(V⁺) CI is > 0.5, in each arm.
- **V is more than a VP counter:** the lower bound of the stratified AUC (V⁺, own VP × lead) CI is > 0.5 **and** the held-out ΔAUC CI excludes 0. Otherwise the verdict text says "mostly a VP counter".
- **Go for #38 (value-priced responses):** the lower bound of the Spearman(ΔV, Δwin) CI is > 0 **and** the CI of mean Δwin(ΔV > 0) − mean Δwin(ΔV ≤ 0) excludes 0. Otherwise #38 is reported as no-go or inconclusive, with the margin curve as the reason to (not) try a margin anyway.
- The arms' rl win rates (Wilson intervals) are reported as a replication check against 006's 20.72% (vs heuristics) and 008's 18.9% (vs traders) on fresh boards; not a verdict.

## Config

```
OMP_NUM_THREADS=1 .venv/bin/python -m experiments.critic_calibration --arm vs_heuristic \
  --games 4000 --workers 6 --checkpoint rl_runs/selfplay/catan_bc_ft_long/catan_bc_ft_long_10031616.zip
OMP_NUM_THREADS=1 .venv/bin/python -m experiments.critic_calibration --arm vs_trading_heuristic \
  --games 4000 --workers 6 --probe-games 4000 --forks-per-offer 16 --checkpoint <same>
```

Run from the worktree `../catan-37-calib` with the main checkout's `.venv` (the editable install points at the main checkout; the run uses the worktree's modules through the working directory, and the result JSON's `module_paths` records where the workers actually imported `agents`, `engine`, `rl`, `experiments` from; the driver aborts if any is outside the worktree). 6 workers (CPU cap), one arm at a time. Result JSONs: `experiments/results/critic_calibration_<arm>_p4_catan_bc_ft_long_10031616.json`; raw per-state rows in `rl_runs/critic_calibration/*.npz` (gitignored).

## Smoke runs (disclosed; not results)

On seeds 500001+ (never used for anything else): a 24-game timing run of arm T, then 200 games of each arm with K = 4 forks. **Read only:** wall time, worker RSS, the module-path assertion, the offers per game, the sign check corr(V, own VP) vs corr(V⁺, own VP), and the turn terciles. **No AUC, win rate, ΔV or fork result was read**; the smoke result JSONs (which contain the analysis) were deleted unread and are not committed.

- Timing: arm T 35 s and arm H 8 s per 200 games on 6 workers (K = 4).
- Memory: peak 359 MB for one worker and 2.2 GB summed over all processes, so the forks share the model (no per-fork copy).
- Modules: all four resolved under the worktree in every worker.
- Sign check: corr(V, own VP) = −0.05 and −0.01, corr(V⁺, own VP) = +0.60 and +0.61 (arms T and H): the VP term is in V as the algebra says (raw V is nearly uncorrelated with VP because the two effects cancel), and V⁺ restores it.
- Volume: about 109-113 top-level decisions per game; arm T shows 14.7 offers per game reach the rl seat (195 of 200 games get a fork).
- Turn terciles: (12, 24) in arm T and (12, 25) in arm H; the frozen cut points above.

## Environment

catan commit: the result JSONs record `git_commit` (this log's pre-registration commit or later); existing `.venv` (torch CPU), `sb3-contrib`, gamekit as installed; checkpoint sha256 prefix `e8001d3e81ba`.

## Result

_Pending: filled in after the measured runs._

## Verdict

_Pending._

## Deviations / disclosures

- A first start of the measured runs on 2026-09-29 ~23:1x (from pre-registration commit `4652f9f`) was aborted at ~23:38 for a laptop shutdown. Progress at the abort: arm H had completed all 4000 games and written its result (never opened or read), and arm T had run ~23 minutes with no partial output (results are only written at the end of an arm). Its outputs were not used (moved unread to `rl_runs/critic_calibration/aborted_2026-09-29/`, gitignored); the runs were restarted from scratch.

## Notes / follow-up

- Feeds catan [#38](https://github.com/guidodinello/catan/issues/38) (`rl_value_trade`, margin tuning on the reserved seeds) and the decision-time search idea (gamekit 021).
- Caveats: the traders are the minimal 008 rule (4-7% of proposals accepted), so the offers are not a model of human trading; V is evaluated only from the rl seat's view; the critic is one checkpoint.
