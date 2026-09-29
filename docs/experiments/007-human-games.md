# Human vs bots: logged games in the web GUI (issue #26)

**Date:** pre-registered 2026-09-29; games not yet played
**Note:** [gamekit#005 — statistics](https://github.com/guidodinello/gamekit/tree/main/docs/research) (Wilson CIs, two-proportion test). No gamekit note covers "a human as a sanity check for a hand-written or learned agent"; a new one is proposed as a follow-up gamekit PR (see the PR for #26), not edited here.

**Status: pre-registered, no results.** Everything under [Result](#result) is empty on purpose; nothing here was simulated or estimated. The infrastructure that plays and logs the games is in the same PR.

## Hypothesis

Phase 5 has only machine-vs-machine numbers: the best RL agent (`catan_bc_ft_long_10031616`, [006](006-longer-run.md)) wins 94.6% of 4-player games vs 3 `RandomAgent`s and 20.7% vs 3 `HeuristicAgent`s at n=4000. Neither agent has been measured against a human. A few logged games are the cheapest way to learn whether either is actually good at Catan before spending on a board-aware encoder (gamekit note 008).

Question, per bot kind: **does one human beat 3 bots of that kind more often than the 25% chance rate of a 4-player game?** This is a sanity check, not a strength estimate. N=12 per bot kind gives very wide intervals (see [Power](#power)).

## Pre-registration (committed before any game is played)

**Line-up.** 4 players: one human plus 3 bots of a single kind, `rl` or `heuristic`. `rl` is `RLAgent(deterministic=True)` on the checkpoint below, CPU, one thread.

**N and schedule.** **N = 12 games per bot kind (24 games)**, fixed up front. There are 12 pairs; pair *k* uses engine seed `7000+k` and human seat `(k-1) mod 4`, and is played once against each bot kind (same board, same seat). So each human seat plays 3 games per bot kind, and boards are paired across kinds. The order inside a pair alternates (rl first on odd pairs, heuristic first on even ones), so the human's learning curve is not confounded with bot kind. The same table lives in `experiments/human_games.py` as `SCHEDULE_007`.

| # | bot kind | engine seed | human seat |
|---|---|---|---|
| 1 | rl | 7001 | 0 |
| 2 | heuristic | 7001 | 0 |
| 3 | heuristic | 7002 | 1 |
| 4 | rl | 7002 | 1 |
| 5 | rl | 7003 | 2 |
| 6 | heuristic | 7003 | 2 |
| 7 | heuristic | 7004 | 3 |
| 8 | rl | 7004 | 3 |
| 9 | rl | 7005 | 0 |
| 10 | heuristic | 7005 | 0 |
| 11 | heuristic | 7006 | 1 |
| 12 | rl | 7006 | 1 |
| 13 | rl | 7007 | 2 |
| 14 | heuristic | 7007 | 2 |
| 15 | heuristic | 7008 | 3 |
| 16 | rl | 7008 | 3 |
| 17 | rl | 7009 | 0 |
| 18 | heuristic | 7009 | 0 |
| 19 | heuristic | 7010 | 1 |
| 20 | rl | 7010 | 1 |
| 21 | rl | 7011 | 2 |
| 22 | heuristic | 7011 | 2 |
| 23 | heuristic | 7012 | 3 |
| 24 | rl | 7012 | 3 |

**Human, self-assessed level** (written by the human before game 1; not filled in by anyone else):

- Level (novice / casual / intermediate / experienced): **`<TO BE FILLED IN BEFORE GAME 1>`**
- Approximate lifetime games of Catan: **`<TO BE FILLED IN BEFORE GAME 1>`**
- Date filled in: **`<TO BE FILLED IN BEFORE GAME 1>`**

**Trades.** Domestic trading is disabled against both kinds: bots never propose and always reject offers. `HeuristicAgent` already behaves this way. For `rl`, `RLAgent` masks only the propose/counter sentinels, but its accept/reject atoms are live and untrained (no training opponent ever proposed a trade), so the `rl` seat delegates trade *responses* to `HeuristicAgent` (always reject) and uses the policy for everything else. That matches the n=4000 benchmark conditions. Each game record carries `trade_policy: "reject_all"`. Bank and port trades work normally. **This is temporary**: the plan is an RL agent that genuinely learns to trade ([catan #28](https://github.com/guidodinello/catan/issues/28), gamekit note 012), after which this experiment's trade condition no longer applies. The GUI says so too. (The issue text says the RL agent "always rejects (atoms masked)"; that is not accurate, hence the explicit override.)

**Abandonment.** Every game gets a `started` record when it is created. A game that never reaches `finished` (quit, deleted, crashed) **counts as a human loss** and is also listed separately. If a slot is played more than once, only the earliest attempt counts; repeats are reported as deviations.

**Tabulation.** `python -m experiments.human_games --experiment 007` reads the per-game records, matches them to the schedule, and writes `experiments/results/human_games_007_p4_<checkpoint stem>.json` (checkpoint-qualified, like the benchmark results) with Wilson 95% CIs (`gamekit.mc.wilson_interval`), per-seat counts, and every game's row. It refuses to mix rl games made with different checkpoint hashes, and writes nothing when there are no matching records.

**Verdict rule** (per bot kind, human wins *k* of the 12 games, 95% Wilson interval):

- lower bound > 25%: human clearly stronger than this bot;
- upper bound < 25%: bot stronger than this human;
- otherwise: not distinguishable from chance.

The rl-vs-heuristic comparison (`two_proportion_test`) is **descriptive only**.

### Power

At N=12 per kind the 95% Wilson interval is about 45-50 points wide at mid-range (e.g. 5/12 -> [19%, 68%]). The human needs at least **6 of 12** for the lower bound to clear 25% (6/12 -> [25.4%, 74.6%]); "bot stronger" needs 0 of 12 (upper bound 24.3%). The rl-vs-heuristic contrast has essentially no power: with 3/12 in one arm, the other needs about 8/12 to reach p<0.05 (8 vs 3 -> p=0.041). The result is a sanity check; a null result does not show the bots are equally strong.

### Known limitations

- One human, self-assessed skill; says nothing about other people.
- The human learns over 24 games; the alternating pair order only avoids confounding that with bot kind.
- No domestic trading (above).
- Games are not blind: the human knows which kind is seated (the setup screen shows it).
- Engine seeds are fixed and paired; results are for those 12 boards.

## Config

Checkpoint (gitignored, local only): `rl_runs/selfplay/catan_bc_ft_long/catan_bc_ft_long_10031616.zip`; override with `CATAN_RL_CHECKPOINT=/path/to.zip`. Server, from the repo root, using the existing `.venv` (torch and sb3-contrib already installed there):

```
cd web && npm run build && cd ..
CATAN_EXPERIMENT=007 .venv/bin/python -m uvicorn server.app:app
# open http://127.0.0.1:8000
```

For each schedule row: 4 players, seed = the row's engine seed, the human in the row's seat, the other three seats all `Bot: RL (...)` or all `Bot: Heuristic`. (`CATAN_EXPERIMENT=007` tags the game record; without it the game is not counted.) The game-over panel shows the engine seed for a check against the table. If `Bot: RL` is greyed out, the reason is printed under the seat list (torch missing or checkpoint not found).

Records go to `.catan-games/<game_id>.json` (gitignored). Tabulate with:

```
.venv/bin/python -m experiments.human_games --experiment 007
```

## Environment

To be filled in when the games are played: catan commit, torch/sb3-contrib versions, checkpoint `sha256_12` (recorded in every rl game record), human's level (above).

## Result

*Empty: no games have been played.* To be replaced by the tabulated numbers from `experiments/results/human_games_007_p4_catan_bc_ft_long_10031616.json`.

## Verdict

*Pending.* Applied per the rule above once all 24 slots are played (or explicitly reported as incomplete).

## Notes / follow-up

- Trade-learning agent: [catan #28](https://github.com/guidodinello/catan/issues/28), gamekit note 012. A fair human match *with* trading needs it.
- Proposed gamekit note (follow-up PR there): "human baseline as a sanity check for a hand-written heuristic / learned agent".
