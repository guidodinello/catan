# Decision-time search over the current checkpoint (issue #41)

**Date:** pre-registered 2026-09-30 (commit `e58c76b`); measured runs 2026-09-30 / 2026-10-01
**Note:** [gamekit#021 — Decision-time search: ISMCTS with the trained policy/value network as priors](https://github.com/guidodinello/gamekit/blob/main/docs/research/021-decision-time-search.md). Statistics per [gamekit#005](https://github.com/guidodinello/gamekit/blob/main/docs/research/005-eval-statistics.md). Builds on [009 — critic calibration](009-critic-calibration.md), whose Verdict says how the critic may be used as a leaf. A gamekit follow-up (link this log from 021, record the design departures below) is proposed in the PR, not edited there.

**Status: complete.** The pre-registration and the code were committed (`e58c76b`) before any measured run; every chunk of every result carries that `git_commit`, checkpoint sha `e8001d3e81ba`, and module paths inside the worktree. The smoke/timing runs below read no win rate. A1 was stopped on purpose overnight and resumed (see Deviations).

## Hypothesis

A short determinized ISMCTS at decision time, using the trained policy as the move prior and its critic as the leaf evaluator (no retraining), raises the win rate of `catan_bc_ft_long_10031616` against 3 `HeuristicAgent`s above the no-search baseline (20.72% in 006), and possibly above the 25% Phase 5 gate. Tested as the same checkpoint with vs without search at a fixed per-move budget, reporting decision latency.

**How the critic is used (from 009's Verdict).** Leaves are scored by **V⁺ = V + own public VP / 10**, a *ranking score, not a probability*, weak early in the game (AUC ≈ 0.69 before own turn 12, ≈ 0.86 late), whose sibling differences (≈ 0.01) are small against its range (≈ −1 to +0.2).

## Two arms that answer different questions

The search must simulate the opponents somehow. They are not searched, only played by a fixed model:

- **`heur` — search with a known opponent model (an upper bound).** Opponents in the search are `HeuristicAgent`, the same kind of agent the benchmark pits it against. This shows what search can add when the opponent model is right; it is *not* a number that transfers to other opponents.
- **`self` — the transferable number (exploratory).** Opponents in the search are the checkpoint's own greedy policy, which assumes nothing about who is at the table. Smaller n because each simulation costs several network forwards per opponent move.

## Design (frozen)

Code: `agents/ismcts.py` (torch-free search core), `agents/rl_search.py` (`RLSearchAgent`, the policy/critic evaluator, the self-model opponent), benchmark role `rl_search` / mode `rl_search_vs_heuristic`, driver `experiments/search_eval.py`. `engine/` and `agents/heuristic.py` are untouched (README decision 10).

- **Tree.** Open-loop, single observer. A node is one of the rl seat's *top-level* decisions (every decision except a domestic-trade response); edges are composed actions (`ActionKey` = the atom sequence, with the two orderings of a Road Building pair merged). No game states are stored.
- **Simulation.** Each simulation starts from a fresh determinization of the live state, descends by PUCT over the currently legal children, and plays opponents through the opponent model until the rl seat's next top-level decision (its own trade responses are rejected; opponents here never propose).
- **Determinization (public information only).** The opponents' combined holding of each resource is `19 − bank − own`, dealt uniformly into their public hand sizes; the unseen dev-card multiset (full deck − own hand − publicly played knights/VP/progress counts, with the played progress types drawn uniformly among those the seat does not hold) is dealt into their public hand-slot counts (keeping each card's `bought_this_turn` flag) and the deck; a deal giving an opponent ≥ 10 true VP is redealt; the copy's RNG is reseeded from the agent's own `seat_rng`, so dice, steals and draws are sampled chance. This is weaker than card counting (it ignores, e.g., what a steal revealed). The module never reads an opponent's composition, the deck order or `state.rng`; `tests/test_ismcts.py` asserts this (two states identical in public information and different in hidden information give the same decision and visit counts).
- **Prior.** The product of the atom probabilities along a composed action's atom path, from one fused, batched policy+value forward over the trie of legal atom prefixes, renormalized over the legal actions.
- **Leaf (one rule for every leaf).** The rl seat's next top-level decision, valued by V⁺ from the empty-buffer observation; a terminal state is +1 if the rl seat won, else −1. Backup is single-seat.
- **Selection.** PUCT with `c_puct = 1.25`; Q min-max normalized over the tree (MuZero's `MinMaxStats`); an unvisited child is valued at the parent's mean normalized Q; no noise, no temperature. The move is the most-visited root child (ties: higher prior).
- **Not searched.** A decision with one legal composed action (53% of top-level decisions), and trade responses, go to the greedy `RLAgent`; with budget 0 the agent *is* `RLAgent` (tested: identical `GameRecord`s). Setup, discard, robber and steal decisions are searched.

**Departures from 021's sketch** (stated, not hidden): (1) backup is single-seat rather than per-seat multi-player UCT, because opponents are played by a fixed model rather than searched; (2) the tree is open-loop over the rl seat's decisions (opponent moves are simulated, not tree nodes); (3) Q is min-max normalized; (4) the first-play value of an unvisited child is the parent's mean normalized Q, where MuZero's pseudocode uses 0, and the exploration term uses `sqrt(N+1)`; `c_puct = 1.25` is MuZero's `pb_c_init` (its `log((N + 19652 + 1)/19652)` addend is < 0.007 at these visit counts), checked against the published pseudocode; (5) trades stay masked, as today.

**A property worth knowing before reading results.** When a rollout can end the game (an opponent near 10 VP), a terminal ±1 enters the tree-wide min-max range and compresses V⁺ differences between siblings, so in such endgame positions the prior dominates more. This is the normalization working as designed (a loss is worth more than any 0.01 gap), not a bug; it is why the unit test that checks "values override a lopsided prior" uses a position where no opponent can win inside the horizon.

## Smoke / timing runs (disclosed; not results)

On seeds **4000001–4000100** (48 for the last row), disjoint from every range used elsewhere (`tests/rl/test_search_eval.py` asserts this against 009's ranges and the off-limits 1..10000 and 90001..100000). `--timing-only`: the driver records **no outcome** for these games, so no win rate exists to read. Read: wall time, latency, simulations per second, worker RSS, override rate against the greedy action, skip rate, module paths, and the leaf-noise check below. 16 workers on the 20-core laptop, #40 finished.

| arm | S | worker-s/game | mean ms / searched decision | p95 | max | sims/s | override | peak RSS |
|---|---|---|---|---|---|---|---|---|
| none | – | 1.5 | – | – | – | – | – | 353 MB |
| heur | 16 | 4.0 | 66 | 129 | 323 | 243 | 8.6% | 358 MB |
| heur | 32 | 8.7 | 158 | 305 | 720 | 203 | 8.1% | 357 MB |
| heur | 64 | 20.0 | 360 | 671 | 1640 | 178 | 8.7% | 363 MB |
| heur | **128** | **43.2** | **794** | **1469** | **2727** | 161 | **9.9%** | 365 MB |
| heur | 256 | 93.2 | 1774 | 3188 | 5506 | 144 | 10.4% | 363 MB |
| self | 32 | 37.4 | 713 | 1617 | 4485 | 45 | 8.8% | 360 MB |
| self | 128 (48 games) | 378.9 | 7795 | 14780 | 27031 | 16 | 9.1% | 358 MB |

- 53% of the rl seat's top-level decisions have a single legal action and are not searched (`skip_rate`). The override rate is among the searched ones; at S = 128 it is 14.5% in MAIN, 6% at the robber, 12% at the first settlement, and 0% for the setup road.
- `state.copy()` costs about 20 µs; one isolated single-process game ran at about 2.5 ms per simulation, and 4–7 ms under 16 concurrent workers, so the latencies above are for a loaded machine (an upper-ish figure).
- Memory: one worker peaks at about 0.36 GB (the model is shared, the tree stores no states), so 16 workers take about 6 GB.
- **Leaf-noise check (the horizon asymmetry).** An `EndTurn` leaf is scored after three sampled opponent turns, a build leaf is scored at once. The SD of leaf values across determinizations at `EndTurn` edges is **0.061 (S=16) to 0.099 (S=256)** on average, against **0.024–0.028** for the SD of the sibling edges' mean values. So per-sample `EndTurn` noise is about 3.5× the spread between sibling moves, and only averaging over many simulations makes the comparison usable. This is reported, not fixed: the leaf rule above is frozen.
- The smoke run died once after `self_s32` (no traceback; the two remaining rows were rerun detached), and `experiments/search_eval.py` was refactored once while it ran (`_map_jobs` extracted; no logic change).

## Pre-registration

**Seeds.** The benchmark boards, as in 006: `engine_seed_base = driver_seed_base = 1`, consecutive seeds, seat-rotated by `engine_seed % 4`.

**Budget rule (applied to the table above).** Ladder S ∈ {16, 32, 64, 128, 256} simulations per searched decision. **S\*** is the largest S whose estimated n = 4000 wall time is ≤ 6 h at 10 workers, i.e. ≤ 60 worker-hours (conservative: with 16–18 workers it is ~3–4 h). Escalate to Guido instead of choosing if S = 16 does not fit, or if the override rate at S\* is below 2% of searched decisions.
- heur: 43.2 worker-s/game × 4000 = 48 worker-h at S = 128, 103 worker-h at S = 256 → **S\* = 128**. Override rate there: 9.9% → no escalation.
- self: 379 worker-s/game at S = 128 → 105 worker-h for n = 1000, over the cap; interpolating the measured S = 32 (37.4) and S = 128 (379) gives ≈ 119 worker-s at S = 64 (≈ 33 worker-h for n = 1000), which fits. So the `self` arm uses **S = 64**, disclosed: it is half the heur arm's budget.

**Arms and run order** (all vs 3 `HeuristicAgent`, `RLAgent` is the same checkpoint `rl_runs/selfplay/catan_bc_ft_long/catan_bc_ft_long_10031616.zip`, sha256 prefix `e8001d3e81ba`):

| id | arm | what | n (seeds) |
|---|---|---|---|
| A0 | `none` | `RLAgent`, no search (the driver's own rerun; must reproduce 006's 829/4000 exactly, it is deterministic) | 4000 (1..4000) |
| A1 | `heur`, S = 128 | **primary**: search with a known opponent model (upper bound) | 4000 (1..4000) |
| A2 | `self`, S = 64 | exploratory, **the transferable number** | 1000 (1..1000) |
| C1, C2 | `heur`, S = 32 and S = 64 | exploratory budget curve (with A1 at 128: three points) | 1000 (1..1000) |

If A0 does not give exactly 829/4000, stop and find out why before running anything else.

**Statistics (α = 0.05, two-sided).**
- **Primary:** A1 vs A0, `gamekit.mc.testing.two_proportion_test` (n = 4000 each). MDE at 80% power ≈ 2.5 points (p ≈ 0.207). Secondary: exact McNemar on the per-board discordant pairs (same boards), and Wilson intervals throughout.
- A2 and C1/C2 vs A0 on the same first 1000 boards (MDE ≈ 5.1 points at n = 1000): descriptive, with intervals, no gate claim.
- The 25% gate needs p̂ ≳ 26.4% at n = 4000 for the Wilson lower bound to clear 25%.

**Verdict rules.**
- **Search helps, with a known opponent model:** A1 beats A0 (p < 0.05 and a positive difference). Always worded as an upper bound.
- **Phase 5 gate:** A1's Wilson lower bound is above 25%. Reported with the known-opponent-model qualifier; whether it counts toward the gate is Guido's call.
- **Transferable:** A2 vs A0, reported as exploratory with its interval.
- Also reported: mean, p95 and max decision latency (per searched decision, and per top-level decision including skipped ones), simulations per second, override rate overall and by phase, skip rate, peak RSS, and the budget curve.
- 006's 20.72% is the best of five fixed points, so A0 (the same 4000 boards) carries its winner's-curse bias in the absolute level; 009 measured the same checkpoint at 20.575% [19.35, 21.86] on fresh boards. The A1 − A0 comparison is paired on the same boards, so the bias cancels in the difference but not in the level compared with 25%.

## Config

```
cd ../catan-41-search     # the worktree; never `uv run` here (no new venv)
export PYTHONPATH=$PWD OMP_NUM_THREADS=1
PY=/home/guido/projects/catan/.venv/bin/python
CK=/home/guido/projects/catan/rl_runs/selfplay/catan_bc_ft_long/catan_bc_ft_long_10031616.zip
$PY -m experiments.search_eval run --arm none --games 4000 --workers 16 --checkpoint $CK
$PY -m experiments.search_eval run --arm heur --sims 128 --games 4000 --workers 16 --checkpoint $CK
$PY -m experiments.search_eval run --arm self --sims 64  --games 1000 --workers 16 --checkpoint $CK
$PY -m experiments.search_eval run --arm heur --sims 32  --games 1000 --workers 16 --checkpoint $CK
$PY -m experiments.search_eval run --arm heur --sims 64  --games 1000 --workers 16 --checkpoint $CK
$PY -m experiments.search_eval analyze none heur_s128 self_s64 heur_s32 heur_s64 --baseline none --write search_eval_011
```

Games run in chunks of 200 seeds, each written atomically to `rl_runs/search_eval/<arm>/chunk_<base>.json` (gitignored); `run` skips finished chunks, so a laptop shutdown loses at most one chunk, and it refuses to resume under a different config and refuses a measured run from a dirty worktree. Every game is seeded by its `(engine_seed, driver_seed)`, so results do not depend on the worker count or on where a run was resumed. The driver records every worker's module paths and aborts if any is outside the worktree. Result JSON: `experiments/results/search_eval_011.json`.

## Environment

catan commit: the result JSON records `git_commit`; existing main-checkout `.venv` (torch CPU, `sb3-contrib`, gamekit as installed), run from the worktree `../catan-41-search` with `PYTHONPATH=$PWD`; 20 cores, 15 GB RAM, up to 16 workers (the 18-worker cap is for headroom), laptop used in the daytime only.

## Result

Result JSON: `experiments/results/search_eval_011.json` (all numbers below are from it; the raw per-chunk rows are in `rl_runs/search_eval/`, gitignored). 16 workers throughout; peak worker RSS 373 MB, no memory warning.

**A0 replication (pre-registered check).** The driver's no-search rerun gives **829/4000 = 20.72% [19.50, 22.01]**, exactly 006's number, so the harness, the seed/seat mapping and the checkpoint are the ones 006 measured.

### Win rates (3 `HeuristicAgent` opponents, seat-rotated, seeds 1..n)

| arm | simulations (S) | n | wins | win rate [Wilson 95%] | vs A0 on the same boards |
|---|---|---|---|---|---|
| A0 no search | 0 | 4000 | 829 | 20.72% [19.50, 22.01] | – |
| **A1 `heur`** (known opponent model, upper bound) | 128 | 4000 | 1385 | **34.62% [33.17, 36.11]** | **+13.9 pts**, z = 13.89, p < 1e-40 (two-proportion); exact McNemar p ≈ 3e-74 (770 boards won only with search, 214 only without) |
| **A2 `self`** (opponents = own policy; the transferable number, exploratory) | 64 | 1000 | 272 | 27.20% [24.53, 30.04] | +5.6 pts vs 21.6% (216/1000), z = 2.92, p = 0.0036; McNemar p = 6e-5 (124 vs 68) |
| C1 `heur` | 32 | 1000 | 263 | 26.30% [23.67, 29.12] | +4.7 pts, z = 2.46, p = 0.014 |
| C2 `heur` | 64 | 1000 | 298 | 29.80% [27.05, 32.71] | +8.2 pts, z = 4.20, p = 3e-5 |
| A1 on the first 1000 boards | 128 | 1000 | 353 | 35.30% [32.40, 38.31] | +13.7 pts, z = 6.79 |

- **Budget curve (`heur`, the same 1000 boards, 0 / 32 / 64 / 128 simulations): 21.6% → 26.3% → 29.8% → 35.3%.** Monotone, with no sign of saturation at the chosen S = 128. S = 128 vs S = 64 on the same boards is +5.5 pts (post-hoc, z = 2.62, p = 0.009, McNemar p = 2e-4).
- **Equal-budget opponent-model comparison (post-hoc, not pre-registered).** `heur` S = 64 (29.8%) vs `self` S = 64 (27.2%): −2.6 pts for not knowing the opponents, z = 1.29, p = 0.20 (McNemar p = 0.093). n = 1000 cannot resolve a difference of that size, so this neither confirms nor excludes a cost of the self-model.
- **The gain is not seat-specific.** A1 minus A0 by rotation position (`engine_seed % 4` = 0, 1, 2, 3): +12.8, +11.3, +16.9, +14.6 pts.

### Decision latency, throughput, overrides (per searched decision, 16 concurrent workers on a 20-core laptop, so an upper-ish figure)

| arm | S | mean ms | p95 | max | sims/s | override | worker-s/game |
|---|---|---|---|---|---|---|---|
| `heur` | 32 | 163 | 311 | 952 | 197 | 8.4% | 9.0 |
| `heur` | 64 | 371 | 694 | 3406 | 172 | 8.7% | 20.3 |
| **`heur`** | **128** | **845** | **1537** | **5129** | 151 | **9.3%** | **45.2** |
| `self` | 64 | 1806 | 3787 | 45066 | 35 | 8.6% | 95.8 |

- 53% of the rl seat's top-level decisions have one legal action and are not searched (53 searched decisions per game). Averaged over *all* top-level decisions, including the skipped ones, the cost is **395 ms (A1)** and **837 ms (A2)** per decision.
- A1 overrides the greedy action in 9.3% of searched decisions: MAIN 13.1%, first settlement 12.4%, discard 6.2%, robber 5.8%, roll 4.2%, steal 0.9%, setup road 0.0%. About one searched decision in eleven is enough for +13.9 points.
- Measured cost matched the smoke estimate (A1 45.2 vs 43.2 worker-s/game; A2 95.8 vs the interpolated 119). A1 took 3.3 h of chunk wall time, A2 1.8 h, the curve 0.5 h. Peak RSS stayed at 373 MB per worker.

## Verdict

- **Search helps with a known opponent model: yes by the pre-registered rule** (A1 beats A0, p < 0.05, positive difference): 34.62% [33.17, 36.11] vs 20.72% on the same 4000 boards, +13.9 pts. This is an **upper bound**: the opponents inside the search are the same deterministic `HeuristicAgent` that sits at the table, so the search predicts their moves exactly and is uncertain only about their hidden cards and the dice.
- **Phase 5 gate:** A1's Wilson lower bound (33.17%) is above 25%, so the rule is met *with the known-opponent-model qualifier*. Whether that counts toward closing the gate is Guido's call. The model-free number (20.72%) and the gate are unchanged.
- **Transferable number (exploratory): 27.2% [24.53, 30.04]**, +5.6 pts over no search on the same 1000 boards, at half the primary arm's budget and with opponents modelled by the agent's own policy. The effect is significant, but the interval's lower bound is below 25%, and the arm is exploratory with a ~5-point MDE; no gate claim is made. The curve suggests a larger budget would add (the heuristic-model arm gains 5.5 pts from S = 64 to S = 128), but the self-model arm was not run at S = 128.
- **Latency:** ~0.8 s per searched decision (p95 1.5 s) at S = 128 against a heuristic model, ~1.8 s (p95 3.8 s, max 45 s) with the self model at S = 64, on a loaded laptop. That is fine for an offline benchmark and slow for interactive play, which is why the agent is a benchmark role only and not a GUI seat.
- **Leakage check, stated plainly.** The result is large, so I looked for ways search could be reading hidden information. What rules it out: `tests/test_ismcts.py` shows two states with identical public information and different hidden hands, dev cards, deck order and RNG state give the same decision and visit counts, and the live state is untouched; the determinizer reads only public counts and the agent's own cards; chance is reseeded from the agent's own stream, never from `state.rng`; every action goes through the engine's `legal_actions` and `apply_action`. The same property is also tested end to end through `RLSearchAgent.choose_action` with the real evaluator, encoder and greedy comparison (`tests/rl/test_rl_search.py`, tiny checkpoint), and was run once against the real checkpoint at S = 16 on 13 mid-game decision states from the smoke seeds 4000001+ (games played by `HeuristicAgent`s; the others did not reach a qualifying decision): in all 13 the live state and a twin with redealt hidden information produced the same action and the same statistics, 12 of the twins having different opponent hands. What it does not rule out is a leak through a path those tests do not exercise. The size of the gain is, however, consistent with the structural explanation above (exact opponent prediction), and the self-model arm, which does not have that advantage, still gains 5.6 pts.
- **Where the number comes from, and what it does not say.** The comparison A1 vs A0 is paired on the same boards, so 006's winner's-curse bias (20.72% is the best of five fixed points) cancels in the difference; it does not cancel in the level compared with 25%. 009 measured the same checkpoint at 20.575% on fresh boards, and A1's margin is far larger than that gap. Nothing here says search would help against a different opponent; A2 is the only evidence on that, and it is exploratory.

Caveats: one checkpoint; the heuristic opponent model is deterministic and exact; the determinizer's belief is weaker than card counting (it ignores what a steal revealed); leaves of an `EndTurn` edge are ~3.5× noisier than the gaps between sibling moves (see the smoke section), so a larger budget mostly buys noise reduction; a terminal ±1 widens the tree-wide min-max range in endgames, so there the prior dominates more.

## Deviations / disclosures

- **A1 was stopped on purpose overnight and resumed.** On 2026-09-30 (`chunk_2201.json` was written at 21:21:53) the laptop was being shut down, so the driver was stopped right after `chunk_2201.json` was written (seeds 1–2400 done, 12 of 20 chunks, each verified as valid JSON with 200 games). It was resumed on 2026-10-01 (~13:37) from `chunk_2401` with the same command, at the same commit `e58c76b` and a clean tree; the driver skipped the 12 existing chunks. Every game is seeded by its `(engine_seed, driver_seed)`, so the numbers do not depend on where a run was resumed. At most a few seconds of the next chunk were lost.
- **Branch moved while the arms ran.** After the pre-registration commit, #40 (experiment 010) merged to main and conflicted with this branch's README index row. It was resolved in a separate worktree (merge commit `4a19c53`, then a test-only CI fix `548c227` that skips two numpy-dependent test files in the plain test job) so the measuring worktree stayed at `e58c76b`. All 55 measured chunk files carry `e58c76b`; the commits since differ only in tests, docs, and the analysis code below.
- **Analysis code added after the runs.** The `provenance` section of `search_eval analyze` (commits, config hashes, workers, module-path check per arm) was added after the measured runs, with the results commit. It reads the chunks only.
- **Workers.** 16 for every measured run (the cap is 18); #40 had finished, so the earlier 10-worker limit no longer applied.
- **Smoke.** One smoke script died once without a traceback (rerun detached), and the driver was refactored once while it ran, as already disclosed above.
- The `self` arm used S = 64, not S\*, by the pre-registered budget rule (disclosed in the pre-registration).

## Notes / follow-up

- Proposed gamekit change (not edited here): in note 021, set Status/Result to link this log, record the five design departures above, and state that the `heur` arm is an upper bound and the `self` arm the transferable number.
