# Critic calibration: does V(s) rank positions, and does ΔV price trades? (issue #37)

**Date:** pre-registered 2026-09-29 (commit `4652f9f`); measured runs 2026-09-30
**Note:** [gamekit#005 — Eval statistics: Wilson intervals](https://github.com/guidodinello/gamekit/blob/main/docs/research/005-eval-statistics.md) (statistics). Prerequisite check for [020 — modular trade agent](https://github.com/guidodinello/gamekit/blob/main/docs/research/020-modular-trade-agent.md) (stage 1, value-priced responses = catan #38) and [021 — decision-time search](https://github.com/guidodinello/gamekit/blob/main/docs/research/021-decision-time-search.md) (critic as leaf evaluator). Neither note is *tested* by this log; it measures the evaluator both would use. A gamekit follow-up (link this log from 020 and 021) is proposed in the PR, not edited there.

**Status: complete.** The pre-registration and the code were committed (`4652f9f`) before any measured run; the result JSONs carry `git_commit` `5ad44f0`. Smoke runs and an aborted first start are disclosed below.

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

Run from the worktree `../catan-37-calib` with the main checkout's `.venv` (the editable install points at the main checkout; the run uses the worktree's modules through the working directory, and the result JSON's `module_paths` records where the workers actually imported `agents`, `engine`, `rl`, `experiments` from; the driver aborts if any is outside the worktree). 6 workers as pre-registered (10 in the measured runs, see Deviations), one arm at a time. Result JSONs: `experiments/results/critic_calibration_<arm>_p4_catan_bc_ft_long_10031616.json`; raw per-state rows in `rl_runs/critic_calibration/*.npz` (gitignored).

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

n = 4000 games per arm, seats rotated, 10 workers (see Deviations). JSONs: `experiments/results/critic_calibration_vs_heuristic_p4_catan_bc_ft_long_10031616.json` (97 s) and `critic_calibration_vs_trading_heuristic_p4_catan_bc_ft_long_10031616.json` (1688 s), both stamped `git_commit` `5ad44f0` (pre-registration `4652f9f` plus the disclosed worker-cap commit), checkpoint sha `e8001d3e81ba`, every worker's modules imported from the worktree. Raw rows: `rl_runs/critic_calibration/critic_calibration_*_p4_catan_bc_ft_long_10031616.npz` (gitignored).

**Replication check (not a verdict).** The rl seat's win rate on the fresh boards: arm H **20.575% [19.35%, 21.86%]** (823/4000) vs 006's 20.72% (z = -0.17, p = 0.87); arm T (forced-reject rl seat) **17.40% [16.26%, 18.61%]** (696/4000) vs 008's 18.9% (756/4000; z = -1.74, p = 0.082, the intervals overlap). Both consistent with the earlier numbers.

### 1. Does V rank positions? (headline: one random decision per game, n = 4000 per arm, row-bootstrap B = 2000)

| metric | arm H (vs 3 heuristic) | arm T (vs 3 trading heuristic) |
|---|---|---|
| AUC of V⁺ vs win | 0.832 [0.816, 0.847] | 0.830 [0.814, 0.845] |
| AUC of raw V vs win | 0.768 [0.750, 0.786] | 0.760 [0.740, 0.780] |
| AUC of public lead alone | 0.777 [0.760, 0.794] | 0.774 [0.756, 0.792] |
| AUC of own public VP alone | 0.689 [0.666, 0.711] | 0.692 [0.670, 0.715] |
| Spearman(V⁺, win) | 0.465 [0.440, 0.489] | 0.433 [0.408, 0.457] |
| Spearman(V⁺, final own VP) | 0.644 [0.625, 0.664] | 0.619 [0.597, 0.639] |
| Spearman(V⁺, lead) | 0.695 [0.677, 0.713] | 0.710 [0.693, 0.727] |
| Spearman(raw V, own VP) | -0.146 [-0.178, -0.114] | -0.177 [-0.211, -0.144] |

Raw V is *negatively* rank-correlated with own VP (the shaped-reward algebra above: the `-phi(s)` term), so V⁺ is the score to use as a position evaluator: its AUC is 0.83 against 0.76-0.77 for raw V. V⁺ also beats the public lead alone (0.83 vs 0.77-0.78, intervals not overlapping).

**Reliability of V⁺** (headline sample, deciles within the arm; Wilson intervals are valid here, one state per game). The naive map p̂ = (V⁺+1)/2 is shown only for reference:

Arm H:

| decile | n | mean V⁺ | p̂ = (V⁺+1)/2 | win rate | Wilson 95% |
|---|---|---|---|---|---|
| 0 | 400 | -0.978 | 0.011 | 0.005 | [0.001, 0.018] |
| 1 | 400 | -0.886 | 0.057 | 0.035 | [0.021, 0.058] |
| 2 | 400 | -0.824 | 0.088 | 0.050 | [0.033, 0.076] |
| 3 | 400 | -0.773 | 0.114 | 0.085 | [0.061, 0.116] |
| 4 | 400 | -0.719 | 0.141 | 0.110 | [0.083, 0.144] |
| 5 | 400 | -0.655 | 0.173 | 0.168 | [0.134, 0.207] |
| 6 | 400 | -0.575 | 0.213 | 0.182 | [0.148, 0.223] |
| 7 | 400 | -0.467 | 0.266 | 0.263 | [0.222, 0.308] |
| 8 | 400 | -0.277 | 0.361 | 0.445 | [0.397, 0.494] |
| 9 | 400 | 0.204 | 0.602 | 0.715 | [0.669, 0.757] |

Arm T:

| decile | n | mean V⁺ | p̂ = (V⁺+1)/2 | win rate | Wilson 95% |
|---|---|---|---|---|---|
| 0 | 400 | -0.979 | 0.011 | 0.010 | [0.004, 0.025] |
| 1 | 400 | -0.891 | 0.055 | 0.025 | [0.014, 0.045] |
| 2 | 400 | -0.835 | 0.082 | 0.043 | [0.027, 0.067] |
| 3 | 400 | -0.783 | 0.109 | 0.052 | [0.035, 0.079] |
| 4 | 400 | -0.728 | 0.136 | 0.090 | [0.066, 0.122] |
| 5 | 400 | -0.671 | 0.165 | 0.120 | [0.092, 0.156] |
| 6 | 400 | -0.599 | 0.200 | 0.152 | [0.121, 0.191] |
| 7 | 400 | -0.496 | 0.252 | 0.247 | [0.208, 0.292] |
| 8 | 400 | -0.313 | 0.343 | 0.367 | [0.322, 0.416] |
| 9 | 400 | 0.183 | 0.592 | 0.632 | [0.584, 0.678] |

V⁺ is monotone in the win rate from ~1% to ~63%, and p̂ tracks it within ±0.11 (max deviation 0.113 in H, 0.056 in T; p̂ falls inside the Wilson interval in 7 of 10 deciles in H and 4 of 10 in T; it is low in the top two deciles in H and high in the middle deciles in T). Close, but systematically off: V⁺ is a good ranking score, not a probability.

### 2. Is it a VP counter? (all decisions, ~434,894 states in arm T; game-clustered bootstrap B = 200)

| metric | arm H | arm T |
|---|---|---|
| AUC of V⁺ over all states | 0.805 [0.794, 0.816] | 0.807 [0.795, 0.820] |
| **Stratified AUC of V⁺ within (own VP × lead) cells** | 0.665 [0.648, 0.681] | 0.673 [0.655, 0.692] |
| AUC of V⁺, early game (own turn < 12) | 0.687 [0.671, 0.705] | 0.699 [0.683, 0.717] |
| AUC of V⁺, mid game (own turn 12-23) | 0.823 [0.813, 0.837] | 0.828 [0.814, 0.842] |
| AUC of V⁺, late game (own turn ≥ 24) | 0.859 [0.843, 0.873] | 0.855 [0.839, 0.872] |
| AUC of V⁺, table leader < 5 VP | 0.685 [0.670, 0.701] | 0.691 [0.675, 0.710] |
| AUC of V⁺, table leader 5-6 VP | 0.807 [0.792, 0.824] | 0.816 [0.797, 0.834] |
| AUC of V⁺, table leader ≥ 7 VP | 0.879 [0.866, 0.891] | 0.879 [0.865, 0.894] |

(The stratified AUC of raw V is identical to V⁺'s by construction: own VP is constant within a cell.) Held-out logistic regression, fitted on odd game indices and scored on even ones:

| model | arm H AUC | arm T AUC |
|---|---|---|
| win ~ lead + own VP + turn | 0.778 | 0.761 |
| ... + V⁺ | 0.817 | 0.804 |
| ΔAUC (95% CI) | +0.039 [+0.030, +0.049] | +0.043 [+0.032, +0.055] |

### 3. Off-distribution probe: trade offers to the rl seat (arm T)

56,408 offers reached the rl seat in 3,856 of 4000 games (14.6 per game that had one). Every offer gives the rl seat at least as many cards as it pays (`TradingHeuristicAgent` asks 1 card for 1-2), so the pays-more split is empty and is not reported further.

- **ΔV = V(after accept) − V(offer withdrawn)** is tiny against V's own scale (V⁺ spans about -1 to +0.2): quantiles q05 / q25 / median / q75 / q95 = -0.0086 / -0.0025 / +0.00004 / +0.0035 / +0.0127, mean +0.00094, **51.1% positive**.
- Sanity scales point the right way: receiving the cards without paying raises V in 92.4% of offers (median +0.0077); paying without receiving raises it in only 7.9% (median -0.0075).
- ΔV follows the shortfall rule `TradingHeuristicAgent` uses: offers that cut the rl seat's shortfall to its next build (n = 8,973) have mean ΔV +0.0037 (74% positive); shortfall unchanged (n = 20,968) +0.0011 (54%); shortfall worse (n = 26,467) -0.0001 (41%).
- **Off-distribution shift.** The critic read at the raw pending-offer state (offer block encoded, never seen in training) differs from V(offer withdrawn) by a median +0.0041 with q05-q95 [-0.0458, +0.0267], a spread about 3x ΔV's own. The pending state is not a usable input; only the constructed `after`/`before` pair is (as pre-registered).

### 4. Does ΔV predict what accepting does? (paired forks, 3856 offers x 16 pairs, arm T)

Δwin = win rate if the rl seat accepts minus if it rejects, from paired playouts with common random numbers. Per-offer Δwin has SD 0.103; mean over offers -0.0009 (always-accept vs always-reject: 10,518 vs 10,576 wins in 61,696 playouts each), i.e. accepting the average offer is neutral.

| ΔV quintile | n | mean ΔV | mean Δwin | bootstrap 95% |
|---|---|---|---|---|
| 0 | 771 | -0.0072 | -0.0090 | [-0.0173, -0.0011] |
| 1 | 771 | -0.0020 | -0.0011 | [-0.0074, +0.0046] |
| 2 | 772 | -0.0001 | +0.0010 | [-0.0037, +0.0054] |
| 3 | 770 | +0.0020 | +0.0019 | [-0.0046, +0.0090] |
| 4 | 772 | +0.0103 | +0.0024 | [-0.0066, +0.0121] |

- **Spearman(ΔV, Δwin) = +0.041 [+0.002, +0.077]**: lower bound above 0, but only just.
- **Mean Δwin(ΔV > 0) − mean Δwin(ΔV ≤ 0) = +0.0062 [-0.0003, +0.0128]**: the interval includes 0.
- The pre-registered margin curve (accept iff ΔV > m; in `fork.margin_curve_exploratory` of the JSON) and a post-hoc bootstrap of its per-offer gain over always-reject (appendix) have a CI that includes 0 at every margin tried.

## Verdict

- **V ranks positions: yes**, in both arms. The lower bound of the headline AUC(V⁺) is 0.816 (H) and 0.814 (T), far above 0.5; V⁺ is also better than the public lead alone. Use V⁺, not raw V (raw V is negatively correlated with own VP). V⁺ is a ranking score, not a calibrated probability (p̂ = (V⁺+1)/2 tracks the win rate within ~0.11 but is systematically off).
- **V is more than a VP counter: yes by the pre-registered rule** (stratified AUC lower bound 0.648 H / 0.655 T > 0.5, and the held-out ΔAUC CI excludes 0: +0.039 [+0.030, +0.049] H, +0.043 [+0.032, +0.055] T). The qualifier matters: within a fixed public state (own VP × lead) V⁺'s AUC is only ~0.67, most of its ranking power is public-VP-related (Spearman(V⁺, lead) ≈ 0.70), and it is weaker early in the game (AUC ~0.69-0.70 in the first 12 own turns (< 12) vs ~0.86 late).
- **#38 (value-priced trade responses): not a go under the pre-registered rule** (`go_for_38 = False`). ΔV has the right sign: it follows the shortfall rule, the gift/pay sanity checks hold, and the worst-ΔV quintile loses win rate (-0.9 pts, CI excludes 0). But the two-part rule needs both conditions and the second fails: the Δwin difference between ΔV > 0 and ΔV ≤ 0 is +0.6 pts with an interval that includes 0, and Spearman(ΔV, Δwin) = +0.04 passes with a lower bound of only +0.002. The bigger finding is the size of the prize: accepting the average offer is neutral (-0.09 pts), and the best exploratory per-offer policy gain over always-reject is about +0.1 pt (every CI includes 0, upper bounds ≤ +0.4 pts). Whatever #38's value responder gains over its forced-reject bar (18.9% [17.72%, 20.14%], half-width ~1.2 pts) must therefore come from many offers compounding (14.6 offers per game), and this log does not measure that. **Recommendation (the decision is Guido's):** treat #38 as inconclusive with low expectations; if run, use a small non-negative margin on raw ΔV (the ΔV scale is ~0.01; the exploratory curve is flat from m ≈ -0.0003 to +0.008), tune it on the reserved seeds 3000001+, and expect an effect that n = 4000 may not resolve.
- **Gamekit 021 (critic as leaf evaluator):** V⁺ ranks states usefully (AUC ~0.83), but early states are weak (~0.69) and differences between sibling states (ΔV ~ 0.01) are small against V⁺'s range; evaluate on V⁺ and check a search's margins against that scale.

Caveats: the forked outcome is one offer per game with K = 16 pairs, so Δwin is noisy (SE 0.0017 on the mean); the reject branch lets later responders still trade, whereas `before` withdraws the offer; the offers come from one minimal trader rule; one checkpoint.

## Post-hoc additions (not pre-registered)

Computed from the npz after the pre-registered analysis; reported so the margin curve has intervals. Bootstrap over the 3856 forked offers, B = 2000, seed 20260929, per-offer gain `mean(Δwin · 1[ΔV > m])` over always-reject:

| margin m | share accepted | gain per offer | 95% CI |
|---|---|---|---|
| -0.0003 | 0.559 | +0.0013 | [-0.0011, +0.0038] |
| 0 | 0.483 | +0.0011 | [-0.0012, +0.0035] |
| +0.0015 | 0.315 | +0.0008 | [-0.0014, +0.0030] |
| +0.0030 | 0.246 | +0.0009 | [-0.0012, +0.0029] |
| +0.0050 | 0.182 | +0.0003 | [-0.0017, +0.0021] |
| +0.0080 | 0.107 | +0.0005 | [-0.0010, +0.0020] |
| always accept | 1 | -0.0009 | (see Result: Δwin overall) |

Also post-hoc: the replication z-tests in Result (`gamekit.mc.two_proportion_test`, descriptive) and the p̂ deviations quoted under the reliability tables.

## Deviations / disclosures

- A first start of the measured runs on 2026-09-29 ~23:1x (from pre-registration commit `4652f9f`) was aborted at ~23:38 for a laptop shutdown. Progress at the abort: arm H had completed all 4000 games and written its result (never opened or read), and arm T had run ~23 minutes with no partial output (results are only written at the end of an arm). Its outputs were not used (moved unread to `rl_runs/critic_calibration/aborted_2026-09-29/`, gitignored); the runs were restarted from scratch.

- **Workers 6 → 10 (2026-09-30).** The pre-registration and the aborted first start used `--workers 6` (the CPU cap at the time). The measured runs use `--workers 10` because the machine is free of other work except a ~9-core training run (20 cores, 15 GB RAM). This changes scheduling only: every game and every fork is seeded by its `(engine_seed, driver_seed)` and results are collected in seed order, so the numbers do not depend on the worker count. The harness's hard cap (`MAX_WORKERS`) was raised from 6 to 10 in the same commit; nothing else in the code or the frozen analysis changed. Every other flag is as pre-registered.

## Notes / follow-up

- Feeds catan [#38](https://github.com/guidodinello/catan/issues/38) (`rl_value_trade`, margin tuning on the reserved seeds) and the decision-time search idea (gamekit 021).
- Caveats: the traders are the minimal 008 rule (4-7% of proposals accepted), so the offers are not a model of human trading; V is evaluated only from the rl seat's view; the critic is one checkpoint.
