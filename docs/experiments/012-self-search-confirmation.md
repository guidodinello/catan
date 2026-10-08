# Confirming decision-time search with the self opponent model (issue #46)

**Date:** pre-registered 2026-10-07 (the commit that adds this file; no measured run exists before it); measured runs: see Deviations
**Note:** [gamekit#021 — Decision-time search: ISMCTS with the trained policy/value network as priors](https://github.com/guidodinello/gamekit/blob/main/docs/research/021-decision-time-search.md). Statistics per [gamekit#005](https://github.com/guidodinello/gamekit/blob/main/docs/research/005-eval-statistics.md). Follows [011 — decision-time search](011-decision-time-search.md) (issue #41, PR #45).

**Status: pre-registered, not yet run (timing smoke and stop/resume checks done).** The code and this log are committed before any measured game. The timing smoke below read no win rate (`--timing-only` records no outcome).

## Hypothesis

Log 011's transferable arm (A2: `RLSearchAgent`, opponents inside the search modelled by the checkpoint's own greedy policy, S = 64, n = 1000, exploratory) gave **27.2% [24.53, 30.04]** against 21.6% without search on the same boards (+5.6 pts, p = 0.004). Its lower bound was under 25% and it was exploratory. This log is the pre-registered confirmation at the budget the curve pointed to: the same checkpoint with `self` search at **S = 128**, n = 4000, on **boards no earlier experiment has seen**, against the same checkpoint without search on the same boards. The self model assumes nothing about who is at the table, so this is the search number that can transfer to other opponents (the `heur` arm of 011 is an upper bound).

## Design (frozen)

- **Checkpoint:** `rl_runs/selfplay/catan_bc_ft_long/catan_bc_ft_long_10031616.zip`, sha256 prefix `e8001d3e81ba` (the driver records and checks it).
- **Search: unchanged from 011.** `git diff 72cb62e origin/main -- agents/ engine/ rl/ experiments/` is empty at pre-registration (#45's merge commit against the current origin/main; the only changes to these directories in this PR are to `experiments/search_eval.py`, the driver, and its tests). The parameters, quoted from the code:
  - `agents/ismcts.py:59` — `SearchConfig`: `simulations=128` (the S* default), `c_puct=1.25`, `opponent="heuristic" | "self"`, `max_advance_steps=5000`, `record_edge_stats=False`; this run sets `opponent="self"`, `simulations=128`.
  - `agents/ismcts.py:278–281` — first-play value of an unvisited child is the parent's mean normalized Q (`fpu = self.minmax.norm(mean)`, `0.0` when the parent has no visits); exploration is `c_puct * sqrt(n_parent + 1)`.
  - `agents/rl_search.py:37,93` — leaf = `V⁺ = V + own public VP / VP_SCALE`, `VP_SCALE = 10.0`, from the empty-buffer observation; policy prior = product of the atom probabilities along the composed action's atom path.
  - Everything else is as described in 011's Design section (open-loop single-observer tree, public-information determinization, one-legal-action and trade-response decisions go to greedy `RLAgent`, trades masked).
- **Opponents at the table:** 3 `HeuristicAgent`, seat-rotated by `engine_seed % 4`, the benchmark's lineup (`rl_search_vs_heuristic` / `rl_vs_heuristic`).
- **Seeds.** `engine_seed = driver_seed`, consecutive, **`fresh_012` = 20,000,001 … 20,004,000** (n = 4000, divisible by 4 so the rotation is exact). Disjoint from every range used or reserved elsewhere, and `tests/rl/test_search_eval.py::test_fresh_012_seeds_are_disjoint_from_every_used_range` asserts it:

  | range | used by | overlaps? |
  |---|---|---|
  | 1 … 10,000 | the n = 4000 benchmark boards (004/006/008/010) and 011's measured arms | no |
  | 90,001 … 100,000 | 008's smoke | no |
  | 500,001 …, 1,000,001 …, 2,000,001 … (n = 200 / 4000 / 4000) | 009's smoke and measured arms | no |
  | 3,000,001 … 3,999,999 (the whole million block) | reserved for #38 | no |
  | 4,000,001 … 4,999,999 (the whole million block; 011's smoke is 4,000,001–4,000,400) | 011's smoke | no |
  | ≤ ≈ 12.05 M | in-loop evals of the 004/006/010 training runs: `cfg.seed + 977 + step` with `--seed 3` and `step` ≤ ≈ 12.04 M | no |

- **Arms, on the same 4000 boards:**

  | id | arm | what | n |
  |---|---|---|---|
  | B0 | `none` | `RLAgent`, greedy, no search | 4000 (20,000,001–20,004,000) |
  | B1 | `self`, S = 128 | `RLSearchAgent`, opponents modelled by the checkpoint's own greedy policy | 4000 (same boards) |

- **Pre-run harness check (not part of the gate).** The driver was rewritten for this run (journaled, stoppable; the game code is untouched) and the lock now pins torch 2.14.1 where 006/011 ran on 2.14.0. So before B0/B1 the driver reruns `none` on the benchmark boards 1..4000 (label `012_check_none`); it must reproduce 011's and 006's **829/4000 exactly** (the arm is deterministic). If it does not, stop and find out why before measuring anything.

## Pre-registered gate and verdict rule

α = 0.05, two-sided. The gate passes **iff both**:

1. B1's Wilson 95% lower bound is **above 25%** (the Phase 5 gate; the opponents inside the search are model-free). At n = 4000 this needs a point estimate of roughly 26.4% or more.
2. B1 beats B0 on the same boards: `gamekit.mc.testing.two_proportion_test(B1, B0)` with **p < 0.05 and a positive difference** (n = 4000 each; MDE at 80% power ≈ 2.5 pts).

Secondary (reported, does not change the verdict): exact McNemar on the board-paired discordant games, and the per-rotation-position difference. The driver computes the gate (`search_eval analyze … --gate`, `gate()` in the code, unit-tested). If only one condition holds the result is reported as such, with no gate claim.

**Reporting list:** B1 and B0 win rates with Wilson intervals; the difference, z, p, McNemar p and the discordant counts; latency per searched decision (mean, p95, max) and per top-level decision including the skipped ones; simulations per second; **override rate overall and by phase**; skip rate; peak RSS; worker-s per game; and **B0's fresh-board level against 006's 20.72% [19.50, 22.01]** (the best of five fixed points, so winner's-curse-biased upward) **and 009's 20.575% [19.35, 21.86]** (the same checkpoint on fresh boards). B1 − B0 is paired on the same boards; the comparison with the 25% gate level is not, which is why B0's level is reported.

## Chunking, stop and resume protocol

- **Chunks of 200 seeds** (20 per arm), each written atomically as `rl_runs/search_eval/<label>/chunk_<base>.json`; labels `012_check_none`, `012_none`, `012_self_s128`. Run order: check, then B0 (about 6 min), then B1.
- **Every finished game is journaled** (append + flush + fsync) to `chunk_<base>.partial.jsonl` and the journal is replaced by the chunk file when the chunk completes. One process pool per invocation keeps exactly one game per worker in flight across chunk boundaries.
- **Stop, graceful:** `touch rl_runs/search_eval/<label>/STOP` — the driver stops submitting, lets the games in flight finish (at most one game, minutes), consumes `STOP`, and exits with code 3. **Nothing is lost.** **Stop, immediate:** SIGTERM or SIGINT (Ctrl-C in tmux) to the driver kills the workers; only the games in flight are lost and they are replayed. A power-off or crash behaves like the immediate stop (journaled games survive; a truncated last journal line is dropped).
- **Resume:** run the same command again. The driver re-reads the journals, refuses to continue under a different config hash, git commit, checkpoint or **environment** (python, torch, numpy, sb3-contrib, gamekit versions are part of the hash), and refuses a measured run from a dirty tree or from a stale `STOP` file.
- **Why a resume gives the same numbers.** A game is a pure function of `(engine_seed, driver_seed, config, checkpoint, code, environment)`. This is tested, not assumed: `test_a_real_game_replays_identically_in_any_process_state` plays a real self-search game with a tiny checkpoint fresh, after a different game in the same process, and in a subprocess with a different `PYTHONHASHSEED`, and requires the same `GameRecord` digest, winner and search decisions. Each game stores that digest, so identity can also be checked on the real run.
- **Disclosure from data.** Every invocation (start, end, reason `completed` / `stop_file` / `signal` / `error`, workers, games done; a start with no end is an unclean exit) is logged in `<label>/invocations.jsonl`, and each game and chunk records its invocation id. `search_eval analyze` lists them in `provenance`, so the Deviations section below is generated from that file, not from memory.
- **Where it runs.** A dedicated worktree at the merge commit with its own lock-pinned venv (`uv sync --frozen --dev --extra rl-train`), `PYTHONPATH=$PWD OMP_NUM_THREADS=1`, so a later `git pull` in the main checkout cannot change the commit or the packages mid-run. The driver aborts if any worker imports a module from outside the worktree.

## Config

```
cd ../catan-46-run        # worktree at the merge commit; never run from the main checkout
export PYTHONPATH=$PWD OMP_NUM_THREADS=1
PY=$PWD/.venv/bin/python
CK=/home/guido/projects/catan/rl_runs/selfplay/catan_bc_ft_long/catan_bc_ft_long_10031616.zip
$PY -m experiments.search_eval run --arm none --games 4000 --workers W --checkpoint $CK --label 012_check_none
$PY -m experiments.search_eval run --arm none --seed-range fresh_012 --games 4000 --workers W --checkpoint $CK --label 012_none
$PY -m experiments.search_eval run --arm self --sims 128 --seed-range fresh_012 --games 4000 --workers W --checkpoint $CK --label 012_self_s128
$PY -m experiments.search_eval analyze 012_none 012_self_s128 --baseline 012_none --gate 012_self_s128 --write search_eval_012
```

`W` = **16** (memory-bound; see the smoke below). Every launch happens in a named tmux session after a `pgrep` check that no `experiments.search_eval` driver and no truco job is running, with free memory logged.

## Smoke / timing runs (disclosed; not results)

On seeds **4000101–4000148** (inside 011's smoke range 4,000,001–4,000,400, which no measured run uses; the driver refuses `--timing-only` runs outside the smoke range and measured runs inside it). `--timing-only` records **no outcome**, so no win rate exists to read. Run from the worktree `../catan-46` at commit `5e639fc` with its own lock-pinned venv (torch 2.14.1+cpu), in a tmux session, after `pgrep` guards showed no other `search_eval` driver and no truco job. At start: 11.0 GB available of 15.7 GB, AC power, battery 79%, load settling from an earlier incident (below).

**`self`, S = 128, 48 games, 16 workers** (wall 918 s, invocation `completed`; 011's own smoke row for the same arm in brackets):

| | this smoke (torch 2.14.1) | [011 smoke, torch 2.14.0] |
|---|---|---|
| worker-s per game, mean (median / max) | **248.4** (225.5 / 744.0) | [378.9] |
| mean ms per searched decision | 4529 | [7795] |
| p95 / max ms | 8966 / 19029 | [14780 / 27031] |
| simulations per second | 28.3 | [16] |
| override rate (searched decisions) | 9.2% | [9.1%] |
| skip rate (single-legal-action decisions) | 52.5% | [53%] |
| peak worker RSS | 358 MB | [358 MB] |

Override rate by phase (searched decisions, n): MAIN 12.6% (1458), MOVE_ROBBER 7.6% (383), SETUP_SETTLEMENT 6.3% (96), DISCARD 5.8% (121), ROLL 5.2% (230), STEAL 1.3% (235), SETUP_ROAD 0.0% (96). Leaf-noise check: SD across determinizations at `EndTurn` leaves 0.104 against 0.031 for the sibling edges' mean values (3.3×, as in 011).

- **Worker count: 16, chosen by memory.** 16 workers × 0.36 GB ≈ 5.7 GB; with Firefox and several Claude sessions open the laptop has about 10–11 GB available, and an earlier overload on this laptop (below) swapped and froze the desktop. An 18-worker leg was skipped for that reason (the owner's call): memory, not cores, is the binding limit.
- **ETA.** B1 is 4000 × 248.4 / 16 ≈ **17 h** of wall time at 16 workers with the streaming pool (no idle workers at chunk boundaries); the 48-game mean carries about ±13% sampling error (games range 106–744 worker-s), and the issue's 26 h figure came from 011's smoke (378.9 worker-s/game). I plan for **17–26 h, i.e. 3 to 4 daytime workdays** (at 6–8 h per day), plus about 12 minutes for the 829 check and B0 together (about 1.5 worker-s per game each). The result's own `worker_s_per_game` replaces this estimate.
- The smoke at S = 128 is 35% cheaper per game than 011's smoke row. Candidate reasons are the 2.14.1 build, different boards, and 011's smoke running while other jobs shared the laptop; I did not isolate the cause and the pre-registration does not depend on it.

**Stop / resume identity on the real checkpoint** (self, S = 16, 32 games on seeds 4000201–4000232, 8 workers, chunks of 16; `--timing-only`). Compared against an uninterrupted run, per game: `GameRecord` digest, winner field absent (timing-only), and every search decision except its wall time.

| leg | stop | exit | games before the stop | resumed games | result |
|---|---|---|---|---|---|
| STOP file | `touch STOP` after 25 s | 3 | 25 (drained in flight) | 7 | **identical** (32/32) |
| SIGTERM to the driver | `kill -TERM` after 25 s | 130 | 16 | 16 (14 + 2 journaled games kept) | **identical** (32/32); no orphaned worker or driver afterwards |

All 32 digests are distinct. The first version of the SIGTERM leg signalled a shell-function subshell instead of the driver, so the driver kept running and my resume ran on top of it on the same label; that leg's data was discarded and redone with the driver as the signalled process (the table row above). Journal recovery after a real power cut is covered below.

**Disclosed incidents (all on non-measured smoke seeds; every affected run was discarded and the 16-worker leg rerun from scratch).** Local times, 2026-10-07:

1. 19:35 — the first 16-worker attempt died after 6 s (the harness killed the detached child when its wrapper shell exited). Relaunched.
2. 19:35–19:49 — the next attempt overlapped the orchestrator's truco-py pytest run (about 19:43–19:48, one or two cores busy). I stopped it with SIGTERM (journal and invocation log record `signal`), and a restart at 19:49:22 resumed from its journal, so one game of the overlapped run sat in the new journal. A `pkill -f` pattern of mine matched my own shell during this clean-up. All of it discarded.
3. about 20:08 — **power loss** (battery): everything was killed, including the 19:49 run and its load sampler. The journal survived with all 29 fsynced games and no truncated line (`error` end event, 28 games done in that invocation); it is not usable as a timing sample (mixed provenance), so it was discarded. This is an unplanned test that journal resume survives a power cut.
4. 20:13–21:22 — **an orphaned driver**: the harness reaped my background wrapper, but the driver (16 workers) kept running for about 70 min, and my retries started more drivers on top of it. The doubled load swapped and froze the owner's desktop. The orchestrator SIGTERM'd it at 21:22. All of its output was discarded. Since then every timing job runs in tmux with `pgrep` duplicate and truco guards.
5. 21:40–21:41:29 — a truco job (`scripts.league_010`) ran for about a minute after the clean smoke had finished (21:39:04) and before the identity check started; my launch guard refused to start on top of it and the check started after it ended. No overlap with either measured smoke.

**"The machine had these runs to itself":** the clean 16-worker smoke (21:23:45–21:39:04) and the identity checks (21:41:38–21:45:39) ran with no other `search_eval` driver, no truco job and no pytest process (checked by `pgrep` at start and polled throughout), the orchestrator running nothing else, load average rising to 17–18 during the 16-worker smoke (16 workers plus desktop), AC power throughout, and free memory logged at start (11.0 GB, 10.8 GB). I have no per-process CPU trace of the desktop beyond that.

## Environment

Python 3.14.5, torch 2.14.1+cpu, numpy 2.5.3, sb3-contrib 2.9.0, gamekit 0.3.0 (from the lock at origin/main `efed88d`); the result JSON records the commit and these versions per chunk. 20 cores, 15 GB RAM; the laptop runs in the daytime only. In-loop eval caveats: not applicable (no training).

## Result

Pending. Result JSON will be `experiments/results/search_eval_012.json`.

## Verdict

Pending (pre-registered gate above).

## Deviations / disclosures

- Written before any measured run; the driver changes (journaled/stoppable resume, invocation log, digest, environment and search params in the config hash, the `fresh_012` range, the gate helper) are in the same PR as this log. The game code (`agents/`, `engine/`, `rl/`) is unchanged from 011.
- The stop/resume table (every invocation boundary) is added after the run from `provenance.invocations`.
