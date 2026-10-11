# Board-aware (graph) policy network vs the flat MLP (issue #42)

**Date:** pre-registered 2026-10-11. No run, smoke or measurement exists before this file's merge.
**Note:** [gamekit#008 — Board-aware encoder / spatial inductive bias](https://github.com/guidodinello/gamekit/blob/main/docs/research/008-board-aware-encoder.md). Statistics per [gamekit#005](https://github.com/guidodinello/gamekit/blob/main/docs/research/005-eval-statistics.md). Implementation plan: [docs/plans/graph-encoder.md](../plans/graph-encoder.md). Motivated by [004](004-bc-warm-start.md) (spatial BC accuracy) and [010](010-entropy-schedule.md) (run-to-run spread).

**Status: pre-registered; code merged (#71); Amendment 1 recorded; no measured run yet, no results.** Run log, Result and Verdict are empty on purpose. The smoke in Amendment 1 read no win rate. Stage 1 launches only on the owner's go.

## Hypothesis

Log 004's BC clone of `HeuristicAgent` reaches 51.6% / 55.4% / 50.5% masked accuracy on the `vertex_settlement` / `edge` / `hex` blocks and 76-95% elsewhere, with only an 8-point train/val gap. The suspected bottleneck behind the ~20-21% plateau (006) is the flat MLP's lack of board adjacency. **H1:** a graph policy network over the hex / vertex / edge incidence, same observation, same BC data and same budget, (S1) clones the spatial decisions measurably better, and (S2) if so, that gain survives the PPO fine-tune as a higher win rate against 3 `HeuristicAgent` at an equal 3M-step budget.

The comparison is graph vs flat **retrained under identical code and rules**, not graph vs the historical 004 / 006 checkpoints (different code, device and budget; log 010 showed the same config landing 2.2 points apart across reruns).

## Design (frozen at pre-registration)

- **Arms.** `flat`: stock `MaskablePPO("MlpPolicy", net_arch=[256,256])`, pi/vf disjoint. `graph`: `GraphMaskablePolicy` per the plan (d = 64, L = 3, two independent pi/vf trunks, pointer head, no node-ID embeddings). Same 1484-dim observation, same action space. Parameter counts of both arms are reported with the smoke.
- **BC data.** `rl_runs/bc/v1` (3000 heuristic games, 508,600 rows, `game_id` 0..2999, `--seed-base 1`, i.e. boards 1..3000, the same as log 004). sha256, computed read-only on 2026-10-11, files unchanged since 2026-09-21:

  | file | sha256 |
  |---|---|
  | `obs.npy` | `5b5d1f9f95ca1a13a3a89da6373ad95692484ec0418caa72718266681afc6ddc` |
  | `masks.npy` | `a51239612e017aa337e7b0a1c2ce6d5ca8c4bcd95a39db68c46b5d4542878126` |
  | `labels.npy` | `4fdfc4b3ac28386d7bca212a5a9d56a56b127f8579743e97817d62aa96bb03d4` |
  | `returns.npy` | `b176937fe238fa219fe733a7c16075613afce0ede1a79bc995301e06165668f3` |
  | `game_id.npy` | `8a4789053bc66cb1e3f88b3bb0231ca8a47b198b54d38591fc0fa5c9ae6969bb` |

- **Splits, by game (stage 1).** **val = `game_id % 10 == 0`, sealed: scored once per final model, never used for any choice.** **tune = `game_id % 10 == 1`**, used for early stopping and the learning-rate choice. train = the other 80%. (Log 004 early-stopped on val; this protocol deliberately does not.)
- **Hyperparameters, fixed now.** Graph d = 64, L = 3. Both arms: batch 1024, ≤ 30 epochs, patience 3 on the tune spatial accuracy, Adam, loss as in `rl/bc.py`. The only tuning is a learning-rate choice from **{1e-3, 3e-4}**, the same two values for both arms, picked by the highest tune spatial accuracy of a seed-0 run. No other search.
- **Boards for every measured win rate:** `fresh_014` = engine seeds **21,000,001 … 21,004,000** (n = 4000, divisible by 4 for exact seat rotation), `driver_seed` equal to the engine seed. The code PR adds the range to the seed registry with a test asserting disjointness from `experiments.league_013.USED_RANGES` (including `fresh_012` 20,000,001+, the benchmark boards 1..10,000 and the reserved blocks), from the BC boards 1..3000, and from the in-loop eval range (`cfg.seed + 977 + step`, ≤ ≈ 12.05 M). Nothing earlier has played these boards.
- **Opponents:** 3 plain `HeuristicAgent`, seat-rotated by `engine_seed % 4`, `RLAgent(deterministic=True)` (the `rl_vs_heuristic` benchmark mode).

### Stage 1: behaviour cloning (cheap)

1. lr sweep per arm on tune (seed 0), as above.
2. **3 BC seeds (0, 1, 2) per arm** at the chosen lr; the flat control is retrained, not reused.
3. Score each final model **once** on val.

**Metric S1:** val masked top-1 accuracy pooled over the three spatial label blocks (`vertex_settlement`, `edge`, `hex`) in `rl/bc.py`'s `ATOM_BLOCKS` (city vertices are in the other blocks and are reported, not gated).

**Gate S1 (passes iff both):**
1. mean(graph) − mean(flat) over the 3 seeds is **≥ +3.0 points**, and
2. **min(graph seeds) > max(flat seeds)** (seed-to-seed noise, not just sample noise: at ~10⁴ spatial val rows the binomial noise is about 0.5 point, the training-seed spread is unknown and this clause covers it).

Secondary, reported, not gated: per-block and overall masked accuracy; **tie-aware accuracy** (a prediction counts if it lies in the set of atoms tied at `HeuristicAgent`'s maximum score, if the code PR can make that set exact, otherwise dropped and said so); value MSE; each arm's seed-0 clone at n = 4000 on `fresh_014`.

### Stage 2: PPO fine-tune (only if S1 passes)

Seed-0 clone of each arm, at its chosen lr, fine-tuned in 010's shape:

```
--bc-init <clone> --bc-init-rate <clone's measured rate> \
--baseline-mix 0.5 --ent-coef 0.01 --learning-rate 1e-4 \
--envs 8 --n-steps 512 --device cuda \
--eval-opponents heuristic --eval-every 250000 --eval-episodes 200 \
--regression-margin 0.10 --regression-patience 1000 --steps 3000000
```

(`--regression-patience 1000` disables the stop rule for both arms, so every arm reaches its fixed 3M checkpoint, as in 010.) **2 training seeds per arm (3 and 4): 4 runs.** The fine-tune seed is the only difference between the two runs of an arm.

**Primary metric:** win rate vs 3 `HeuristicAgent` of the **fixed `<label>_3000000.zip`** checkpoint (not the in-loop best), n = 4000 on `fresh_014`, Wilson 95% intervals. Each of the 4 checkpoints is benchmarked once on the same 4000 boards.

**Gate S2 (passes iff all three):**
1. Pooled graph (2 × 4000 games) beats pooled flat (2 × 4000) under `gamekit.mc.testing.two_proportion_test`, **p < 0.05, two-sided, positive difference**.
2. **Both graph checkpoints are above both flat checkpoints** (seed consistency).
3. The pooled difference is **> 2.2 points** (the run-to-run spread observed in 010 between two same-config fine-tunes).

Why this replaces note 008's "non-overlapping CIs": overlapping intervals are not a defined test and the n = 4000 interval says nothing about training-seed variance. The pooled test plus the consistency clauses is the defined version; if the two readings disagree, the log reports both and the gate above decides.

Secondary, reported, not gated: board-paired exact McNemar for each graph-vs-flat checkpoint pair; per-rotation-position wins; in-loop curves (monitoring only, README §In-loop eval caveats); steps to first reach the flat arm's final level.

### Verdict map (decided now)

| outcome | verdict | consequence |
|---|---|---|
| S1 fails | **rejected at BC level**; stage 2 is not run | #42 is parked; report the accuracy table |
| S1 passes, S2 passes | **validated**; S2 is #42's headline | candidate for the next best checkpoint |
| S1 passes, S2 fails | **fidelity improved, not converted to strength at 3M** | rejected for this budget; reported with both numbers |

The README "current best" line changes only if a measured checkpoint beats 006's 20.72% on the same footing; a 3M-step arm is not budget-comparable with 006's ~12M, and the log says so rather than implying it. #42 closes on S2's outcome, not on S1's.

## Docs-first protocol

Unlike 012, there is no code SHA to freeze yet. So:

1. This pre-registration merges first. The code PR (graph policy, `--policy`, `--select-split`, `fresh_014`, tests) merges second.
2. **Before any measured run** a dated **Amendment** section is committed to this file recording: the code merge SHA; the lock-pinned environment versions from the worktree venv; the dataset hashes above re-verified; the smoke timings and parameter counts; and any change the smoke forces. The only change the smoke may force is **shrinking d / L** under the stop rule below; the gates, splits, seeds, metrics and boards do not change after this merge.
3. The **harness check** runs first: flat `catan_bc_ft_long_10031616` on benchmark boards 1..4000 must reproduce exactly **829/4000** (log 012's check; the arm is deterministic). If it does not, stop and find out why before measuring anything.
4. The smoke uses a throwaway dataset (about 50 games on smoke seeds, under `rl_runs/exp014_smoke/`), never `rl_runs/bc/v1`, so no stage-1 outcome is read early, and it reads no win rate.

## Compute budget and wall time (laptop: RTX 4050 6 GB, 20 cores, 15 GB RAM, daytime only)

Estimates from logs 004, 006 and 010; **the smoke replaces them**.

| item | estimate | basis |
|---|---|---|
| BC, flat (2-lr sweep + 3 seeds) | well under 1 h | 004: BC training took minutes |
| BC, graph (same) | ≤ ~3-5 h | unknown; set by the smoke (GPU s/epoch) |
| Fine-tune, flat × 2 | ~3.5 h | 010: ~1 h 45 min per 3M-step arm |
| Fine-tune, graph × 2 | ~7-14 h | assumes 2-4× the flat cost (CPU snapshot opponents dominate) |
| n = 4000 benchmarks (4 checkpoints, plus 2 BC clones) | well under 1 h | 012: B0 took ~6 min at 16 workers; graph inference is slower |
| **Total** | **≈ 15-23 h, 3-4 daytime days** | |

**Stop / amend rule.** If the smoke projects any single graph fine-tune above **8 h** (CPU single-sample graph latency of the opponent snapshots is the thing to watch), d / L shrink in the Amendment before launch. Every fine-tune runs under `timeout 8h`. If a run times out before 3M steps, that is a disclosed deviation, and the arm is not benchmarked at a different step count to rescue the gate.

Memory: size workers by free RAM, not cores (`free -g` before launch); `--envs 8` plus the main process, as in 010.

## Run protocol

- **Where.** A dedicated worktree at the code merge SHA with a lock-pinned venv (`uv sync --frozen --dev --extra rl-train`; the CUDA venv on the HDD via `UV_PROJECT_ENVIRONMENT`, as in the README), `PYTHONPATH=$PWD OMP_NUM_THREADS=1`. Never run from the main checkout: its `.venv` was stale at planning time (gamekit 0.3.0, torch 2.14.0, against a lock of 0.4.0 and 2.14.1).
- **tmux + log.** Every job (BC, each fine-tune, the benchmarks) in a named session, `tmux new -d -s <name> '<cmd> 2>&1 | tee <log>'`: `catan014-bc`, `catan014-ft-<arm>-s<seed>`, `catan014-bench`. No harness background shells for these.
- **Refuse duplicates.** Before each launch: `pgrep -af '[r]l.train|[r]l.bc|[e]xperiments.(benchmark|search_eval)'`, `tmux ls`, `free -g`. If anything is running, do not launch. A dead wrapper is not a dead job: check `pgrep` before relaunching.
- **Stop.** `tmux send-keys -t <name> C-c` (or SIGTERM to that session's process); never a `pkill -f` pattern. The fine-tune resumes with `--resume` from its last checkpoint.
- **Outputs.** `rl_runs/exp014/` on the HDD (a symlink to `/media/guido/0DF7128F0DF7128F/catan-rl_runs`, 380 GB free at planning): `bc/`, `selfplay/`, `logs/`, `bench/`. Never `/tmp`, `/dev/shm` or another tmpfs. Result JSONs go to `experiments/results/` named by checkpoint stem.
- **Sequential, one at a time**, in this order: harness check, BC sweeps and seeds (S1 verdict), then fine-tunes (flat s3, graph s3, flat s4, graph s4), then benchmarks.

## Limitations, stated up front

- One clone seed per arm feeds stage 2; the BC seed spread is measured in stage 1 but not propagated into the fine-tunes.
- 2 fine-tune seeds per arm: a modest estimate of the training-seed spread; the 2.2-point clause and the seed-consistency clause are the protection, not a power calculation.
- CUDA non-determinism: no bit-reproducibility of the fine-tunes.
- **Equivariance vs tie-breaking.** `HeuristicAgent` breaks score ties by enumeration (vertex-ID) order (log 004). The graph trunk has no node-ID embeddings and cannot learn that order, so plain masked accuracy partly penalises the property being tested. The tie-aware secondary exists for this; S1 is gated on the plain metric anyway, which makes the gate conservative for the graph arm.
- The budget is 3M fine-tune steps, not 006's ~12M; a plateau-level claim would need a longer run.
- One opponent set (3 heuristics), 4-player, no trading, as in 004 / 006 / 010 / 012.

## Config

Illustrative; the exact flags are fixed by the code PR and recorded in the Amendment. Run from the worktree, one tmux session per job.

```
cd ../catan-42-run                  # worktree at the code merge SHA
export PYTHONPATH=$PWD OMP_NUM_THREADS=1
PY=/home/guido/projects/catan/.venv-cuda/bin/python
# S1: per arm in {mlp,graph}, per seed in {0,1,2} (lr from the sweep)
$PY -m rl.bc train --dataset rl_runs/bc/v1 --policy <arm> --select-split tune \
  --save rl_runs/exp014/bc/<arm>_s<seed>.zip --epochs 30 --patience 3 --batch-size 1024 \
  --lr <chosen> --seed <seed> --device cuda
# S2: per arm, per fine-tune seed in {3,4} (see the flags above)
$PY -m rl.train --bc-init rl_runs/exp014/bc/<arm>_s0.zip ... \
  --selfplay-dir rl_runs/exp014/selfplay --run-dir rl_runs/exp014/<label> --label <label> ...
# primary metric, each of the 4 fixed 3M checkpoints
$PY -m experiments.benchmark --mode rl_vs_heuristic --games 4000 --players 4 \
  --engine-seed-base 21000001 --driver-seed-base 21000001 --workers <by free RAM> \
  --checkpoint rl_runs/exp014/selfplay/<label>/<label>_3000000.zip
```

## Environment

See [Amendment 1](#amendment-1-2026-10-11) (code SHA, lock-pinned versions, GPU driver). In-loop eval: seed scheme per README §In-loop eval caveats, catan commit `72f046e`.

## Smoke / timing runs

See [Amendment 1](#amendment-1-2026-10-11): two smoke runs (stale env, locked env), no win rate read.

## Amendment 1 (2026-10-11)

Committed before any measured run, as the [Docs-first protocol](#docs-first-protocol) requires. **The gates, splits, seeds, metrics, boards, lr sweep and stop rule above are unchanged.** What this records:

**Code.** Merged in #71 as **`72f046e9302176ff66a85bcac2e164f11275aff5`** (squash on `main`, on top of this log's merge `494ddff`). Measured runs use a dedicated worktree detached at that SHA (`/home/guido/projects/catan-014-run`, clean tree), `PYTHONPATH=$PWD OMP_NUM_THREADS=1`, never the main checkout.

**Environment (lock-pinned).** Built with `UV_CACHE_DIR=<HDD>/uv-cache UV_PROJECT_ENVIRONMENT=/media/guido/0DF7128F0DF7128F/catan-014-venv uv sync --frozen --dev --extra rl-train-cuda` from the worktree. Read back from that venv: Python 3.14.5, **torch 2.14.1+cu130** (CUDA 13.0, `cuda.is_available()` true), **gamekit 0.4.0**, sb3-contrib 2.9.0, stable-baselines3 2.9.0, numpy 2.5.3, gymnasium 1.4.0; GPU NVIDIA GeForce RTX 4050 Laptop (6 GB), driver 595.99.02. The older `.venv` / `.venv-cuda` in the main checkout are stale (gamekit 0.3.0, torch 2.14.0) and are not used for measured runs.

**BC dataset re-verified** (`rl_runs/bc/v1`, sha256 recomputed 2026-10-11, identical to the table in Design): `obs.npy` `5b5d1f9f95ca…6ddc`, `masks.npy` `a51239612e01…8126`, `labels.npy` `4fdfc4b3ac28…03d4`, `returns.npy` `b176937fe238…68f3`, `game_id.npy` `8a4789053bc6…69bb`. Full values are in Design; all five match.

**Smoke** (`experiments/exp014_smoke.py`, tmux `catan014-smoke` and `catan014-retime`, outputs under `rl_runs/exp014/smoke/` and `smoke_locked/`, no win rate read; the 4-game in-loop evals it prints are meaningless). Throwaway dataset of **200 games** (33,513 rows, 26,508 train) on smoke seeds from 4,000,501, **not the about 50 games** the Docs-first protocol said; the larger set keeps GPU warm-up from dominating a per-epoch figure. Seconds per epoch is the difference of a 1-epoch and a 3-epoch run after an untimed warm-up, extrapolated to a full-size train split (456,707 rows, an upper bound; the real split with the tuning split removed is about 407k rows).

| | flat, stale env (torch 2.14.0) | flat, **locked** (2.14.1) | graph, stale env | graph, **locked** |
|---|---|---|---|---|
| parameters, total (pi side / vf side) | 949,472 (503,263 / 446,209) | same | 473,373 (251,996 / 221,377) | same |
| BC s/epoch on 26.5k rows | 0.14 | 0.16 | 4.68 | **4.23** |
| BC s/epoch, extrapolated to 456,707 rows | 2.4 | 2.7 | 80.6 | **72.9** |
| CPU single-sample `lean_predict_batch`, median / p95 (ms) | 0.10 / 0.12 | 0.10 | 1.61 / 1.67 | **1.61** |
| CPU single-sample `model.predict`, median (ms) | 0.19 | 0.19 | 1.34 | **1.32** |
| PPO 16,384 steps, `--envs 8`: `train_fps` (steps/s incl. eval) | 515 (478) | 561 (534) | 256 (240) | **264 (248)** |

The first and third columns were measured on the stale env (code `e2287c2`, pre-merge); the second and fourth re-time the same driver on the locked env at `72f046e`. They agree within a few percent, so the stale-env figures stand and the locked-env ones are the reference. The 16k-step PPO figure is a very short run, so treat fps as about ±20%.

**Stop rule not triggered; d = 64 and L = 3 are kept.** Projection from the locked graph `train_fps` of 264: 3,047,424 steps is about 3.2 h of training. In-loop eval adds time the smoke only samples (8 graph episodes took about 6 s, so n = 200 twice per 250k-step chunk is about 5 min per chunk, about 1 h over 12 chunks, a rough figure). So roughly 4-4.5 h per graph fine-tune against the 8 h `timeout`, and about 1.6 h flat (log 010). Stage 1 BC for the graph arm is about 73 s per epoch, about 37 min for 30 epochs, so the 5 graph BC runs (2 lr-sweep + 3 seeds) are at most about 3 h. The compute table above stands; the fine-tune rows (graph x 2) land near its low end.

**Differences from the plan, disclosed.**
- The **tie-aware secondary metric is dropped**: the BC dataset has no per-decision score ties, so the tied set cannot be made exact without replaying every game. Not measured; the equivariance-vs-tie-break bias stays disclosed in Limitations.
- The graph context vector is **120 dims** (OWN, OPPONENT, GLOBAL, the buffer prefix and TRADE; `rl.graph_policy.CTX_DIM`); the 72-wide buffer edge mask is an edge-node feature instead.
- `rl.bc train` gained `--select-split tune` (early stopping on the tuning split's pooled spatial accuracy; val untouched until the final report) and a pooled `val_acc_spatial` metric; `--policy mlp` is the default and unchanged (a tiny flat BC run gives an identical parameter hash on `main` and the #71 branch).
- `fresh_014` (21,000,001-21,004,000) is registered in `experiments/search_eval.py` and `league_013.USED_RANGES`, with a disjointness test.

**Still to do before Stage 1** (nothing launched): the owner's go; the **harness check** (flat `catan_bc_ft_long_10031616` on boards 1..4000 must give exactly 829/4000); the `pgrep` / `tmux ls` / `free -g` guards at each launch.

## Run log

Empty.

## Result

Empty.

## Verdict

Empty.

## Deviations / disclosures

- Written before any code, smoke or measured run. The code PR and the Amendment follow this merge.
- The HDD behind `rl_runs` and `.venv-cuda` was unmounted at the start of the planning session and mounted during it; the dataset hashes above were computed read-only after the mount. Nothing was written to it.
