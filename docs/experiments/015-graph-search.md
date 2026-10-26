# Decision-time search on the graph checkpoint (issue #96)

**Date:** pre-registered 2026-10-25. No measured game exists before this file's merge; the timing smoke below used `--timing-only` on smoke boards and recorded no outcome. This PR is docs only.
**Note:** [gamekit#021 — Decision-time search: ISMCTS with the trained policy/value network as priors](https://github.com/guidodinello/gamekit/blob/main/docs/research/021-decision-time-search.md). Statistics per [gamekit#005](https://github.com/guidodinello/gamekit/blob/main/docs/research/005-eval-statistics.md). Follows [012 — self search confirmation](012-self-search-confirmation.md) (issue #46) and [014 — graph encoder](014-graph-encoder.md) (issue #42, PR #95), and reuses 012's protocol.

**Status: pre-registered, not run.**

## Hypothesis

Log 012 showed that `self` search at S = 128 (opponents inside the search modelled by the checkpoint's own greedy policy) lifts the flat `catan_bc_ft_long_10031616` from 21.00% to 32.58% against 3 `HeuristicAgent` (+11.6 pts, n = 4000). Log 014 then produced a much stronger policy: `catan014_ft_graph_s3_3000000.zip` reached 59.05% [57.52, 60.56] at n = 4000 on `fresh_014`. **H1:** the same search still adds win rate on top of the graph policy.

This is not a given. The flat policy was weak, so search had many decisions to correct. On the 12 smoke games below, search overrode the greedy choice in 2.3% of top-level decisions, against 9.27% in 012 (a small sample, read as a warning and not as a result): a stronger prior may leave less to fix. The log can therefore end in "no detectable gain", and the verdict map below covers it.

## Design (frozen at pre-registration)

- **Checkpoint:** `rl_runs/exp014/selfplay/catan014_ft_graph_s3/catan014_ft_graph_s3_3000000.zip`, sha256 prefix **`6cb96ab7d856`** (the driver records and checks it). It is the graph s3 checkpoint that 014 named the current best; its 59.05% came from `fresh_014` and, as the better of two checkpoints, is biased upward, which is why B0 below re-measures it.
- **Search: unchanged from 012.** `git diff 694b12e HEAD -- agents/ engine/` (012's merge commit against main at pre-registration) is empty, and `experiments/search_eval.py` differs only by the `fresh_014` range registration. The parameters, as recorded by the smoke run's config: `simulations = 128`, `c_puct = 1.25`, `opponent = "self"`, `max_advance_steps = 5000`, `record_edge_stats = False`. The leaf value, the policy prior, the first-play value and the handling of one-legal-action and trade-response decisions are as in 012's Design section. The graph policy goes through the same `lean_predict_batch` / `rl.bc._forward` path (`extract_features`, `mlp_extractor`, `action_net`, `value_net`) that 014 pins for the graph arm; the smoke ran `self` search end to end on it.
- **Opponents at the table:** 3 plain `HeuristicAgent`, seat-rotated by `engine_seed % 4`, the benchmark's lineup (`rl_search_vs_heuristic` / `rl_vs_heuristic`).
- **Boards:** `fresh_015` = engine seeds **22,000,001 … 22,004,000** (n = 4000, divisible by 4 for exact rotation), `driver_seed` equal to the engine seed. Disjoint from every used range: the benchmark boards 1..10,000, 008/009/011 ranges, the reserved 3,000,001-4,999,999 blocks, `fresh_012` (20,000,001+), `fresh_014` (21,000,001+), the in-loop eval range (`cfg.seed + 977 + step`, below about 25M) and the reserved `fresh_016` block (23,000,001+, issue #97). The code change that registers `fresh_015` in `experiments/search_eval.py` `SEED_RANGES` and `experiments/league_013.py` `USED_RANGES`, with a disjointness test like `test_fresh_014_seeds_are_disjoint_from_every_used_range`, merges before any measured game (Docs-first protocol).
- **Arms, on the same 4000 boards:**

  | id | arm | what | n |
  |---|---|---|---|
  | B0 | `none` | `RLAgent`, greedy, no search | 4000 (22,000,001-22,004,000) |
  | B1 | `self`, S = 128 | `RLSearchAgent`, opponents modelled by the checkpoint's own greedy policy | 4000 (same boards) |

- **Pre-run harness checks (not part of the gate).** (1) The graph s3 3M checkpoint with the `none` arm on `fresh_014` must reproduce 014's **2362 / 4000** exactly (the arm is deterministic). (2) Flat `catan_bc_ft_long_10031616` with `none` on boards 1..4000 must give 012's **829 / 4000**. If either does not, stop and find out why before measuring anything. The sizes are small (a graph `none` benchmark of 4000 games took 4-7 min in 014).

## Pre-registered gate and verdict rule

α = 0.05, two-sided. The gate passes **iff** B1 beats B0 on the same boards: `gamekit.mc.testing.two_proportion_test(B1, B0)` with **p < 0.05 and a positive difference** (n = 4000 each; at a rate near 0.59 the minimum detectable difference at 80% power is about **3.1 points**). The test is the unpaired one, as in 012, which is conservative because the arms share boards.

There is deliberately no absolute-level clause: 012's "lower bound above 25%" was the Phase 5 gate for a weak policy, and the graph policy is already far above it.

| outcome | verdict | consequence |
|---|---|---|
| p < 0.05, positive difference | **search stacks on the graph policy** | `rl_search` on the graph checkpoint is a candidate for the strongest agent; the web GUI seat (#59) can reuse it with a smaller S |
| not significant | **no gain detectable at this size** | reported with the interval of the difference; the MDE of about 3.1 pts bounds what was ruled out |
| p < 0.05, negative difference | **search hurts the graph policy** | reported as such; the override table says where |

Secondary, reported, not gated: exact McNemar on the board-paired discordant games (`experiments.search_eval.mcnemar_exact`) and the per-rotation-position difference; B1 and B0 win rates with Wilson intervals; **B0's fresh-board level against 014's 59.05% [57.52, 60.56] on `fresh_014`** (the paired comparison is B1 − B0; the comparison with 59.05% is across different boards, which is why B0 is reported); latency per searched decision (mean, p95, max) and per top-level decision including the skipped ones; simulations per second; **override rate overall and by phase**; skip rate; peak RSS; worker-s per game. The driver computes the comparison (`search_eval analyze ... --baseline 015_none`); `gate()` there encodes 012's two-clause gate, so the analysis for this log reports the pooled difference and p from `compare()` and the gate here is applied by hand from those numbers (the Amendment records the command and output file).

## Chunking, stop and resume protocol

Exactly as 012: chunks of 200 seeds, every finished game journaled (append, flush, fsync), `touch rl_runs/search_eval/<label>/STOP` for a graceful stop (in-flight games finish; with 600 worker-s per game and 8 workers, a drain takes up to about 10 minutes), SIGTERM or Ctrl-C in tmux for an immediate stop (only games in flight are replayed), and the same command to resume. The driver refuses to continue under a different config hash, git commit, checkpoint or environment, refuses a measured run from a dirty tree, and logs every invocation in `<label>/invocations.jsonl`. A game is a pure function of `(engine_seed, driver_seed, config, checkpoint, code, environment)`; 012 tested this and nothing in the game code changed. Labels: `015_check_graph_none`, `015_check_flat_none`, `015_none`, `015_self_s128`.

Where it runs: a dedicated worktree at the code merge commit, with its own lock-pinned venv (a CPU venv is enough: `uv sync --frozen --dev --extra rl-train`), `PYTHONPATH=$PWD OMP_NUM_THREADS=1`, so a later `git pull` in the main checkout cannot change the commit or the packages mid-run. Outputs go under `rl_runs/search_eval/` (the HDD), never `/tmp`.

## Smoke / timing runs (disclosed; not results)

Run on 2026-10-25 from the 014 run worktree (`/home/guido/projects/catan-014-run`, detached at `7b353e3`, clean; its CUDA venv `catan-014-venv`: Python 3.14.5, torch 2.14.1+cu130, numpy 2.5.3, sb3-contrib 2.9.0, gamekit 0.4.0), CPU only (`CUDA_VISIBLE_DEVICES` empty), `OMP_NUM_THREADS=1`, in tmux `catan015-smoke` with `tee` to `rl_runs/exp015_smoke/smoke.log`. Before launch: no `rl.train` / `rl.bc` / `experiments.*` process and no tmux server, 8 GB available of 15 GB. `agents/` and `engine/` are identical between `7b353e3` and main (the only code difference there is `rl/env.py`, which search does not use). `--timing-only` records no outcome, so no win rate exists to read; smoke seeds 4,000,301-4,000,304 and 4,000,321-4,000,332 are inside the smoke range no measured run uses. Checkpoint: `catan014_ft_graph_s3_3000000.zip`, sha256 prefix `6cb96ab7d856`.

| | graph, S = 16 (load check) | graph, **S = 128** | [012 flat, S = 128, 16 workers] |
|---|---|---|---|
| games, workers | 4, 1 | **12, 12** | [48, 16] |
| wall | 179 s | 809 s (13.5 min) | [918 s] |
| worker-s per game, mean (median / min / max) | 44.0 (28.5 / 21.3 / 97.7) | **596.8** (610.2 / 420.2 / 808.2) | [248.4 smoke; 251.6 measured] |
| top-level decisions per game; searched | 106.5; 60.6% | 108.8; 55.0% | [112.5; 46.6%] |
| mean ms per searched decision | 454 | **9868** | [4796] |
| p95 / max ms | 817 / 1158 | 23,943 / 38,624 | [10,404 / 69,708] |
| simulations per second | 35.2 | **13.0** | [26.7] |
| override rate, searched decisions / all top-level | 2.7% / 1.6% | 4.2% / **2.3%** | [- / 9.27%] |
| peak worker RSS | 720 MB | **904 MB** | [378 MB] |

Override rate by phase at S = 128 (searched decisions, n, % overridden): MAIN 411 → 4.6%; MOVE_ROBBER 103 → 0.0%; ROLL 76 → 2.6%; STEAL 50 → 8.0%; DISCARD 30 → 10.0%; SETUP_SETTLEMENT 24 → 8.3%; SETUP_ROAD 24 → 0.0%. With 12 games these are counts, not rates to rely on.

What the smoke says, and its limits:
- A graph game costs about **2.4 times** a flat game under `self` S = 128 (597 against 252 worker-s), and simulations per second halve (13.0 against 26.7). That is far less than the 16x single-call inference gap of 014's Amendment 1, because the engine, the determinization and the heuristic opponents are a large part of a simulation.
- The 12-game mean has a sampling error of roughly ±6% (games range 420-808). The smoke ran 12 workers on 20 logical cores with a desktop session open (load average about 14), so I cannot isolate contention from the architecture; 012's smoke-to-measured agreement (248 against 252) suggests the contention effect is small, but that was a different machine state.
- **Memory is the binding limit.** 904 MB per worker × 12 = 10.8 GB; with 12 workers the machine showed about 4 GB available (of 15 GB), which is close to the swapping that froze this laptop during 012's smoke. The run therefore uses **8 workers** (about 7.2 GB), and 10 only if `free -g` shows at least 11 GB available at launch. The smoke was started with 12 workers and drained with a STOP file; all 12 games finished.

## Compute budget and wall time (laptop: 20 cores, 15 GB RAM, daytime only)

| item | estimate | basis |
|---|---|---|
| B1 `self` S = 128, n = 4000 | 4000 × 597 worker-s = 2.39 M worker-s = **663 worker-h**; **83 h at 8 workers**, 66 h at 10, 55 h at 12 | smoke mean (±6%) |
| in daytime legs of 6-8 h | **about 10-14 daytime days** at 8 workers | the laptop runs in the daytime only |
| B0 `none`, n = 4000 | well under 15 min | graph `none`: 4-7 min at 3-6 workers in 014 |
| harness checks | about 15 min | two `none` benchmarks |

The owner chose the full n = 4000 for this log on 2026-10-25, knowing this cost (alternatives considered: n = 2000 at about 28-42 h with an MDE near 4.3 pts, and n = 1000 at 2-3 days with an MDE near 6 pts). The result's own `worker_s_per_game` replaces the estimate. Because the run is chunked and journaled, the sample can be stopped at any chunk boundary, but **the pre-registration is for n = 4000 and an early stop is a disclosed deviation**; no interim win rate is looked at (the driver prints counts per chunk, which are not read for a decision).

## Docs-first protocol

1. This pre-registration merges first (docs only).
2. Before any measured game, a dated **Amendment** is committed to this file recording: the code merge SHA (with `fresh_015` registered and its disjointness test), the worktree and the lock-pinned venv versions read back, the checkpoint sha256 re-verified, and the two harness checks' results. The gate, arms, boards, S and n do not change after this merge.
3. Measured runs happen only from a clean worktree at that SHA, never from the main checkout.

## Run protocol

- **tmux + log:** every job in a named session, `tmux new -d -s <name> '<cmd> 2>&1 | tee <log>'` (`catan015-check`, `catan015-b0`, `catan015-b1`). No harness background shells.
- **Refuse duplicates:** before each launch `pgrep -af '[r]l.train|[r]l.bc|[e]xperiments.(benchmark|search_eval)'`, `tmux ls`, `free -g`; a dead wrapper is not a dead job, check `pgrep` before relaunching.
- **Stop:** the job's `STOP` file, or `tmux send-keys -t <name> C-c`; never a broad `pkill -f`.
- **Order:** harness checks, then B0, then B1.

## Config

Illustrative; the exact commands are recorded in the Amendment. From the worktree:

```
export PYTHONPATH=$PWD OMP_NUM_THREADS=1
PY=$PWD/.venv/bin/python
CK=/home/guido/projects/catan/rl_runs/exp014/selfplay/catan014_ft_graph_s3/catan014_ft_graph_s3_3000000.zip
$PY -m experiments.search_eval run --arm none --seed-range fresh_015 --games 4000 --workers 8 --checkpoint $CK --label 015_none
$PY -m experiments.search_eval run --arm self --sims 128 --seed-range fresh_015 --games 4000 --workers 8 --checkpoint $CK --label 015_self_s128
$PY -m experiments.search_eval analyze 015_none 015_self_s128 --baseline 015_none --write search_eval_015
```

## Limitations, stated up front

- One checkpoint (graph s3, the better of two 014 fine-tunes, so its no-search level is biased upward; B0 re-measures it), one opponent set (3 `HeuristicAgent`), 4-player, no trading.
- The result concerns `self` search at S = 128 only; a smaller S for interactive play (#59) is not measured here.
- The unpaired test is conservative for paired arms; McNemar is the paired secondary.
- The smoke is 12 games on one machine state; the cost estimate is ±6% sampling error plus an unknown contention effect.

## Environment

To be filled by the Amendment (code SHA, lock-pinned versions). The smoke's environment is above. In-loop eval caveats: not applicable (no training).

## Amendment 1 (2026-10-25): pre-run prep, before any measured game

Recorded before any `015_none` or `015_self_s128` game exists. Nothing in the Design, gate or verdict rule changes.

**Code.** `fresh_015` (22,000,001-22,004,000) and `fresh_016` are registered in `experiments/search_eval.py` `SEED_RANGES` and `experiments/league_013.py` `USED_RANGES` by PR #100, squash-merged as **`c3a8b97c3c84e7916d6c269f13188f3926af1041`**; the disjointness test is `test_fresh_seeds_are_disjoint_from_every_used_range` in `tests/rl/test_graph_policy.py` (parametrized over `fresh_014`/`fresh_015`/`fresh_016`). Measured runs use the worktree `/home/guido/projects/catan-015-run`, detached at that SHA, clean tree (still clean after the checks).

**Environment (lock-pinned).** `UV_CACHE_DIR=<HDD>/uv-cache UV_PROJECT_ENVIRONMENT=/media/guido/0DF7128F0DF7128F/catan-015-venv uv sync --frozen --dev --extra rl-train` (tmux `catan015-venv`, log `rl_runs/exp015/logs/venv.log`, exit 0). Read back: Python 3.14.5, torch 2.14.1+cpu, gamekit 0.5.0, sb3-contrib 2.9.0, stable-baselines3 2.9.0, numpy 2.5.4, gymnasium 1.4.0. The smoke above ran in 014's venv (gamekit 0.4.0, numpy 2.5.3); this one differs in those two packages.

**Checkpoint hashes (sha256 re-verified).** Graph s3 3M `6cb96ab7d856...` (matches); flat `catan_bc_ft_long_10031616.zip` `e8001d3e81ba...` (the checks' recorded `checkpoint_sha256_12`).

**Harness checks (not part of the gate).** Tmux `catan015-check`, one session, log `rl_runs/exp015/logs/check.log`, run in order with `PYTHONPATH=$PWD OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=` from the worktree, 4 workers (8 GB available at launch; 4 workers, about 0.4 GB peak RSS each, keep the 4 GB headroom), chunks of 200:

```
$PY -m experiments.search_eval run --arm none --seed-range fresh_014 --games 4000 --workers 4 --checkpoint <graph s3 3M zip> --label 015_check_graph_none --out-dir /home/guido/projects/catan/rl_runs/exp015
$PY -m experiments.search_eval run --arm none --seed-range measured --games 4000 --workers 4 --checkpoint <flat zip> --label 015_check_flat_none --out-dir /home/guido/projects/catan/rl_runs/exp015
```

| check | result | expected |
|---|---|---|
| graph s3 3M, `none`, `fresh_014` (seeds 21,000,001-21,004,000, 4000 unique) | **2362 / 4000** | 2362 |
| flat `catan_bc_ft_long_10031616`, `none`, boards 1..4000 (4000 unique) | **829 / 4000** | 829 |

Both exited 0 and reproduce exactly, so the harness is confirmed. Cost: 1362 worker-s (0.34 per game) for graph and 608 worker-s (0.15 per game) for flat. The invocation records and chunks are in `rl_runs/exp015/015_check_{graph,flat}_none/`.

**Deviation / note for the measured runs.** The checks pass `--out-dir` as an absolute path on the HDD, because the driver's `DEFAULT_OUT` is `ROOT/rl_runs/search_eval` and `ROOT` is the worktree, where no `rl_runs` link exists. The measured `015_none` and `015_self_s128` runs must pass `--out-dir` the same way (the Config block above omits it).

## Run log

Empty until the Amendment and the first launch.

## Deviations / disclosures

- Written before any measured game. The smoke read no win rate (`--timing-only`).
- The 12-worker smoke leg was stopped with a STOP file about 1 minute after launch, when memory showed about 4 GB available; the driver stops submitting new games and lets the games in flight finish, and with 12 workers on 12 games every game was already in flight, so all 12 completed (invocation reason `stop_file`, 12 games done). The load check (S = 16, 1 worker) ran first, in the same tmux session.
- Smoke stdout and chunks are in `rl_runs/exp015_smoke/` (gitignored, on the HDD).
