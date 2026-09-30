# Trading opponents: baselines for trade learning (issue #28, slice 1)

**Date:** pre-registered 2026-09-29; n=4000 runs not yet started
**Note:** [gamekit#005 — Eval statistics: Wilson intervals](https://github.com/guidodinello/gamekit/blob/main/docs/research/005-eval-statistics.md) (statistics). Prerequisite for [012 — trade heads](https://github.com/guidodinello/gamekit/blob/main/docs/research/012-trade-heads.md) (end-to-end path) and [020 — modular trade agent](https://github.com/guidodinello/gamekit/blob/main/docs/research/020-modular-trade-agent.md) (modular path); see also [019](https://github.com/guidodinello/gamekit/blob/main/docs/research/019-human-catan-game-data.md). Neither note is *tested* by this log: it builds the opponents and records the baseline they need. A gamekit follow-up (link both notes here) is proposed in the PR, not edited there.

**Status: pre-registered, no n=4000 results.** Everything under [Result](#result) is empty on purpose. The only numbers that exist are the trade-activity smoke run below, which was used to check the opponent trades at all and looked at no win rates.

## Hypothesis

No bot in the repo proposes a trade, so the RL agent's accept/reject head has never been trained and no trade-learning stage can be measured. A minimal opponent that proposes toward its next build target and accepts offers that help it (`TradingHeuristicAgent`, `agents/trading_heuristic.py`, composing an untouched `HeuristicAgent`) gives (a) a table where trades complete, and (b) three baselines:

1. **Does trading help a heuristic?** Trading heuristics vs plain heuristics, 2 seats each.
2. **What does a lone non-trader lose at a trading table?** `HeuristicAgent` vs 3 trading heuristics.
3. **The number every future trade stage must beat:** the current best RL checkpoint, trade responses forced to reject exactly as the web GUI's `RLSeatAgent` does, vs 3 trading heuristics.

## The opponent (frozen before any n=4000 run)

Deterministic, no RNG, tie-breaks by `Resource` order. All other decisions are `HeuristicAgent`'s.

- **Target:** the build `HeuristicAgent` prefers next (city, settlement, road) that has a legal site.
- **Propose** (only when the heuristic would otherwise `EndTurn`): if within `MAX_SHORTFALL_TO_PROPOSE = 2` cards of the target and holding surplus, ask for 1 card of the scarcest needed resource, offering 1 (then 2) of the largest surplus. At most `MAX_PROPOSALS_PER_TURN = 2` per turn, never the same bundle twice in a turn. The engine has no per-turn limit (only 4 cards a side), so the cap lives in the agent.
- **Respond** (fresh offer or counter to its own proposal): accept iff its shortfall to the target strictly drops, it pays no more cards than it receives, and the proposer's public VP < `LEADER_VP_GUARD = 8`. Otherwise reject. Never counters.

`HeuristicAgent` and the forced-reject RL seat always reject, so only trading-heuristic seats can complete a trade, and only with each other.

## Pre-registration (committed before the n=4000 runs)

**Runs.** 4 players, `n=4000` games each, seat-rotated (`engine_seed % 4`), `engine_seed_base=1`, `driver_seed_base=1` (the same boards as `benchmark_rl_vs_heuristic_p4_catan_bc_ft_long_10031616.json`), `--workers 10` (half of 20 cores).

| arm | mode | lineup |
|---|---|---|
| A | `trading_heuristic_2v2` | 2 trading heuristic + 2 heuristic |
| B | `heuristic_vs_trading_heuristic` | 1 heuristic + 3 trading heuristic |
| C | `rl_reject_vs_trading_heuristic` | 1 rl (forced reject) + 3 trading heuristic |

**Why arm A is 2 vs 2.** The requested "1 trading heuristic vs 3 heuristics" cannot trade (all responders reject), and a rejected proposal changes neither state nor RNG, so it would replay `heuristic_vs_heuristic` exactly. That equivalence is a unit test (`tests/test_trading_heuristic.py::test_lone_trader_among_heuristics_replays_all_heuristic`) instead of a run.

**Checkpoint.** `rl_runs/selfplay/catan_bc_ft_long/catan_bc_ft_long_10031616.zip`, sha256 prefix `e8001d3e81ba`, `RLAgent(deterministic=True)`, CPU.

**Primary metrics and verdict rules** (Wilson 95% intervals via `gamekit.mc.wilson_interval`):

- **A:** the share of decided games won by a trading-heuristic seat. Trading helps if the lower bound is above 50%, hurts if the upper bound is below 50%, otherwise inconclusive. gamekit's per-role `two_proportion_test` in the JSON is descriptive only: seats of one role in one game are dependent.
- **B:** the heuristic's win rate vs the 25% chance rate.
- **C:** the RL agent's win rate. This is the baseline for every future trade stage. Compared descriptively (`two_proportion_test`) with 006's 20.72% vs 3 `HeuristicAgent` at the same boards: did trading opponents make the table harder?
- **Trade-aware floor:** an arm in which fewer than **0.5 completed trades per game** occur is declared *not trade-aware* and its number is not the bar for trade stages.
- Also reported for every arm: `trade_stats` (proposals, counters, acceptance rate with Wilson CI, responses, cards traded, completed trades per game, acceptances by responder role, max proposals by one player in one turn), `no_winner_games` (step-budget timeouts; the symptom if the proposal cap ever failed), and `vp_card_stats`.
- **`PlayVictoryPoint` (for #34):** for the RL seat and each other role, plays per legal decision. Report only; no threshold.

## Config

```
.venv/bin/python -m experiments.benchmark --mode trading_heuristic_2v2 --games 4000 --players 4 --workers 10
.venv/bin/python -m experiments.benchmark --mode heuristic_vs_trading_heuristic --games 4000 --players 4 --workers 10
.venv/bin/python -m experiments.benchmark --mode rl_reject_vs_trading_heuristic --games 4000 --players 4 --workers 10 \
  --checkpoint rl_runs/selfplay/catan_bc_ft_long/catan_bc_ft_long_10031616.zip
```

Result JSONs: `experiments/results/benchmark_<mode>_p4[_<checkpoint stem>].json` (the checkpoint stem is added for arm C).

## Smoke calibration (disclosed; not a result)

Before pre-registering, the opponent was run for 200 games per mode on **unrelated seeds (`engine_seed_base=90001`)** and only **trade statistics were read, no win rates**, to check that trades complete and to time the runs. The first version had a bug that produced **zero accepted trades in 1790 proposals** (responders evaluated their build target in the wrong phase); it was fixed, and the constants above were never tuned on win rates.

| mode (n=200) | completed trades / game | trading-heuristic proposals | proposals accepted | max proposals, one player, one turn | no-winner games |
|---|---|---|---|---|---|
| `trading_heuristic_2v2` | 0.70 | 3735 | 3.7% | 2 | 0 |
| `rl_reject_vs_trading_heuristic` | 1.62 | 5297 | 6.1% | 2 | 0 |

Both clear the 0.5 floor. A run of 200 games took 2-6 s.

## Environment

To be filled in at run time: catan commit of the runs, torch and sb3-contrib versions, wall clock per arm.

## Result

*Empty: the n=4000 runs have not been started.*

## Verdict

*Pending.*

## Notes / follow-up

- Trade-learning issue: [catan #28](https://github.com/guidodinello/catan/issues/28). This log is slice 1 (opponents, benchmark, baselines); no RL training and no unmasking of the trade heads.
- The web GUI does not offer a trading bot: `server/persistence.py` stamps `trade_policy="reject_all"` on every game record, which a trading bot would make false, and it would touch `web/` next to live experiment 007.
