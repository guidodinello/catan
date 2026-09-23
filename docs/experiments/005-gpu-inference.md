# GPU (CUDA) for Phase 5 RL: measured, mostly rejected (issue #20)

**Date:** 2026-09-22
**Machine:** RTX 4050 Laptop (6GB, CUDA 13.2 driver), 20 logical cores, `/` 23GB free,
external HDD 455GB free at `/media/guido/0DF7128F0DF7128F`.

## Hypothesis

truco-py's centralized GPU inference server took self-play from ~164 FPS to ~1440 FPS
(8.8x) by batching opponent-checkpoint inference on the GPU instead of running it
locally, per CPU worker, one sample at a time. catan's self-play pool now holds many
checkpoints (up to a dozen-plus, sampled uniformly by `gamekit.rl.selfplay.OpponentPool`)
and each opponent decision may take several atom-level forward passes (the composition
buffer in `rl/action_space.py`), both of which raise the per-env inference rate beyond
truco's one-checkpoint case. The question: does the same design transfer here, and does
the GPU help anywhere else in the pipeline (BC pre-training, the PPO update)?

**Pre-registered decision rule** (written before implementing anything): for each
pipeline component, let `s` = its share of wall-clock time. Adopt the GPU only if
`s >= 20%` *and* a CPU control arm (a lean batched forward, bypassing `MaskablePPO.predict`'s
own overhead) doesn't already capture the win — i.e. the GPU must beat both naive CPU
*and* CPU-batched, not just naive CPU.

## Part A: profiling the CPU pipeline

Isolated profiling pools were copied from `rl_runs/selfplay/catan_bc_ft/` into
`rl_runs/gpu005/pool{1,4,14}/` (never the real self-play directories, to avoid
overwriting log 004's checkpoints) with a fresh `run_id` per pool
(`prof1`/`prof4`/`prof14`), sizes 1/4/12 distinct checkpoints. All commands below are
`rl.profile`, a new committed module (`rl/profile.py`) built for this issue.

### Worker-side split (single process, no PPO; BC clone as learner)

```
uv run python -m rl.profile worker --pool rl_runs/gpu005/pool{N} --label prof{N} \
  --learner rl_runs/bc/catan_bc_clone.zip --steps 20000
```

| Pool | baseline_mix | learner inf. | opponent inf. | heuristic | engine remainder | opp calls/learner-step | opp µs/call |
|---|---|---|---|---|---|---|---|
| pool1 (1 ckpt) | 0.5 | 18.7% | 25.2% | 10.4% | 45.7% | 1.42 | 217.6 |
| pool4 (4 ckpt) | 0.5 | 17.3% | 25.8% | 10.8% | 46.1% | 1.55 | 218.0 |
| pool14 (12 ckpt) | 0.5 | 17.6% | 25.8% | 9.6% | 47.0% | 1.53 | 220.6 |
| pool14 (12 ckpt) | **0.0 (worst case)** | 14.0% | **39.1%** | 0.0% | 46.9% | 2.90 | 217.0 |

Pool size does not change per-call latency (~218-220µs regardless of 1 vs 12 checkpoints)
-- the cost is dominated by `MaskablePPO.predict`'s own per-call overhead, not by which or
how many models are loaded. Opponent inference clears the 20% gate at both the default
`baseline_mix=0.5` (25.8%) and the worst case (39.1%).

### Main-process split (real 16-env MaskablePPO run, `rl.profile train`)

```
uv run python -m rl.profile train --pool rl_runs/gpu005/pool4 --label prof4 \
  --envs 16 --chunk 30000 --chunks 3
# steady-state (last chunk): fps=768 step_wait=26.8s forward=3.6s wall=39.0s
```

Reconciliation with log 004 (`--envs 8`, same pool): this profiler measured **570 fps**,
matching 004's reported `train_fps≈599` closely enough to trust the methodology.

At `--envs 16`: step_wait (env + opponent inference, from the main process's view) ≈
**68.7%** of wall time, learner forward ≈ **9.2%**, PPO update (the remainder) ≈ **22.1%**.
pool14 gave nearly identical numbers (756 fps steady-state) -- pool size doesn't move the
main-process split either, consistent with the worker-side finding above.

### Latency micro-bench (`rl.profile latency`), CPU vs CUDA

```
uv run python -m rl.profile latency --checkpoint rl_runs/bc/catan_bc_clone.zip --device cpu
.venv-cuda/bin/python -m rl.profile latency --checkpoint rl_runs/bc/catan_bc_clone.zip --device cuda
```

| | CPU | CUDA |
|---|---|---|
| `MaskablePPO.predict`, B=1 | 192.6 µs | **1497.9 µs** (GPU loses at B=1 -- kernel-launch latency dominates one sample) |
| lean forward (`agents.rl_agent.lean_predict_batch`), B=1 | 100.4 µs | 101.1 µs |
| lean forward, B=4 | 56.3 µs/sample | 21.2 µs/sample |
| lean forward, B=16 | 25.4 µs/sample | 5.8 µs/sample |
| lean forward, B=64 | 17.8 µs/sample | **1.4 µs/sample** |

Two findings here, both real: (1) bypassing SB3's `predict()` overhead (the lean forward)
is already ~2x faster than `predict()` at B=1 on CPU alone -- pure SB3 overhead, nothing to
do with hardware. (2) GPU batching is decisively cheaper *in isolation*: at B=64 the GPU is
~13x cheaper per sample than CPU. This isolated number is what motivated building the
server in Part C -- and what Part C's end-to-end measurement then contradicted.

## Part B: decision table (pre-registered, before implementing)

| Component | Gate (s≥20%) | Measured `s` | Initial verdict from Part A |
|---|---|---|---|
| Opponent-checkpoint inference | met | 25.8-39.1% | Proceed to build+measure end-to-end (Part C) |
| Learner inference (rollout, B=16 already) | **not met** | 9.2% | Skip -- below gate regardless of speed |
| PPO update | met (borderline) | 22.1% | Proceed to measure (Part C) |
| BC pre-training | n/a (separate process) | -- | Proceed to measure (Part C) |

## Part C: what was built, and what the end-to-end measurement actually showed

### C.1 Built

- `rl/inference_server.py`: `InferenceServer` -- a daemon thread in the main process
  (truco-py's own choice, for the reason its docstring gives: `SubprocVecEnv` blocks in
  `Connection.recv()` while workers step, releasing the GIL, so a thread doesn't contend
  with the vec-env's control flow). One duplex `Pipe` per env, created before
  `SubprocVecEnv(start_method="fork")` forks. Requests are grouped by checkpoint path and
  served with `agents.rl_agent.lean_predict_batch` (the lean forward from the micro-bench),
  `deterministic=True` to match `RLAgent`'s own default.
- `agents/rl_agent.py`: `lean_predict_batch` (shared by the server and `rl.profile`) and
  an optional `inference: InferenceHandle | None` on `RLAgent`, routing every atom decision
  through the server when set. Default (`None`) is byte-for-byte the pre-existing local
  path -- unchanged for `experiments/benchmark.py` and `server/bots.py`.
- `rl/train.py`: `--opponent-inference {none,cpu,cuda}` (default `none`, unchanged
  behavior) and `--device {cpu,cuda}` for the PPO model itself (`build_model`, `--resume`,
  `--bc-init` all honor it). In-loop eval always runs on a CPU copy of the policy
  regardless of `--device`, to avoid a single-sample-on-GPU latency trap (see the CUDA
  column above: B=1 predict on GPU is *slower* than CPU).
- `rl/bc.py`: `--device {cpu,cuda}` for `rl.bc train`.
- `tests/rl/test_inference_server.py`: equivalence tests (lean forward vs SB3 predict,
  batch-size invariance, server-through-pipe vs local `RLAgent`, concurrent multi-env
  requests, and a CUDA-gated test skipped without a CUDA device).

### C.2 Equivalence (verified before measuring speed)

`tests/rl/test_inference_server.py` (untrained but architecturally real checkpoints), all
pass in both the CPU `.venv` and the CUDA `.venv-cuda`:

- `lean_predict_batch` picks the identical atom as `model.predict(deterministic=True)`,
  row for row, on 32 random observation/mask pairs.
- Batching is size-invariant: one N-row call matches N separate one-row calls, bit for bit.
- `InferenceHandle.request` through the real pipe/thread machinery matches local `RLAgent`
  inference, including under concurrent requests from 6 simultaneous env threads.
- CUDA-vs-CPU: identical atoms on 64 rows (test skipped where no CUDA device exists).

Real-data check (not the tests above, a one-off script), 8192 rows sampled from
`rl_runs/bc/v1`'s `obs.npy`/`masks.npy` -- log 004's actual BC training data -- against the
real `catan_bc_clone.zip` checkpoint and a real self-play checkpoint
(`catan_bc_ft_2000000.zip`):

| Comparison | Agreement | Max \|Δlogit\| (finite entries) |
|---|---|---|
| SB3 `predict` (CPU) vs lean forward (CPU) | 100.0% | -- |
| SB3 `predict` (CPU) vs lean forward (CUDA) | 100.0% | -- |
| lean forward (CPU) vs lean forward (CUDA) | 100.0% | ~1.1-1.3e-5 |

The tiny logit delta is ordinary CPU/CUDA floating-point noise (different reduction
order), never enough to flip an argmax at these masked-logit magnitudes.

`rl.bc train --device cuda` on an identical tiny dataset (40 games, seed-fixed) produced
**identical to printed precision** `train_loss`/`val_masked_acc`/`val_value_mse` per epoch
to the CPU run -- only wall time differed.

### C.3 End-to-end FPS: opponent-checkpoint inference server (the headline measurement)

**The Amdahl ceiling, stated before the numbers:** opponent-checkpoint inference is 25.8%
of worker wall time at the default `baseline_mix=0.5` (39.1% at the `baseline_mix=0.0`
worst case, Part A). Even *eliminating that share's cost entirely* caps the possible
end-to-end speedup at `1/(1-0.258) ≈ 1.35x` (`1/(1-0.391) ≈ 1.64x` worst case). Truco-py's
reported 8.8x was never in reach here regardless of implementation quality -- catan's
engine is heavier, so opponent inference is a much smaller slice of the total.

```
uv run python -m rl.profile train --pool rl_runs/gpu005/pool{N} --label prof{N} \
  --envs 16 --chunk 20000 --chunks 2 --opponent-inference {cpu,cuda}
```

| Pool | opponent-inference=none (baseline) | =cpu (server) | mean group size |
|---|---|---|---|
| pool1 (1 ckpt, best case for batching) | 637 fps | 701 fps (+10%) | **1.28** |
| pool14 (12 ckpt) | 602-626 fps | 602 fps (~flat) | **1.04** |

**Root cause, found by instrumenting the server's own batch sizes (`InferenceServer.stats()`),
not by assumption:** `_collect_batch`'s call to `multiprocessing.connection.wait(conns,
timeout)` returns as soon as *any* connection is ready -- it is a max-wait, not a
collection window. Measured mean group size is **1.04-1.28**, i.e. this server almost
never actually batches, regardless of pool size. That single fact, not batch
fragmentation across distinct checkpoints, is the primary reason the Part A micro-benchmark's
promised speedup (up to 13x at B=64) never showed up end-to-end: there is effectively no
batch to speed up.

**A fix was attempted and reverted.** Rewriting `_collect_batch` to drain newly-ready
connections for the rest of the `_BATCH_WAIT_S` window (instead of returning on the first
signal) does turn the timeout into a real collection window -- but measured as a severe
regression, not a fix: when the next concurrent request is sparse (the common case here,
given the mean group size above), every decision now pays close to the full window as
pure added latency with nothing gained, collapsing effective server throughput to roughly
`1 / _BATCH_WAIT_S = 200` requests/second server-wide. A training run that completed a
20,000-step chunk in ~30s at 600+ fps never finished a single chunk under this version
even after several minutes. Reverted; not shipped. A correct version would need a short
grace period after the *first* arrival (well under 5ms) rather than a fixed deadline
applied regardless of what has already arrived -- untried here (see Notes).

**Verdict: reject as a training-throughput win, but not because centralized servers can't
transfer across engines -- this specific port never actually batched.** The correct,
narrower claim: `multiprocessing.connection.wait()`'s first-ready-returns semantics make a
single call to it an anti-pattern for a batching server, and that -- combined with an
Amdahl ceiling of ~1.35x even in the best case -- is why this design doesn't pay off here,
not a general verdict on GPU inference servers. The server is *not* a pure regression
either: pool1's +10% at mean group size 1.28 is a small, real IPC-and-lean-forward win
that beats the Amdahl-capped baseline, consistent with the mechanism, not a fluke;
pool14's ~flat result reflects group size collapsing to ~1.0 as more distinct checkpoints
compete for the same non-collecting `wait()` call.

**Benchmarks and the RL seat's own inference.** `rl.profile worker --baseline-mix 1.0`
(all three opponents `HeuristicAgent`, the BC clone driving the learner seat -- the shape
of `experiments.benchmark --mode rl_vs_heuristic`) measured the RL seat's own single-sample
CPU inference at 27.1% of worker time, comparable to heuristic scoring (27.2%). Since
single-sample `predict` measured *slower* on CUDA than CPU (Part A latency table), and a
benchmark worker plays one game at a time with no natural batch to form, GPU offers no
plausible win here either. A timed `--mode rl_vs_heuristic --games 400 --workers 16` run
took ~10s wall clock -- already fast, and not worth chasing further.

**Runtime code is kept, not deleted**, because it is correctness-verified (Part C.2) and
the underlying idea (batch across concurrent envs) is sound even though this
implementation doesn't realize it -- but `--opponent-inference` defaults to `none` and
this doc recommends leaving it there. `docs/experiments/README.md`'s index and
`pyproject.toml`'s torch comment both say so.

### C.4 End-to-end FPS: PPO update device (`--device cuda`, opponent inference untouched)

```
.venv-cuda/bin/python -m rl.profile train --pool rl_runs/gpu005/pool14 --label prof14 \
  --envs 16 --chunk 20000 --chunks 2 --device cuda
# steady-state: fps=713 step_wait=23.0s forward=2.2s wall=28.0s
```

The CPU pair needed for a like-for-like comparison (same pool, same chunk size, same
`opponent_inference="none"`): `step_wait=23.0s`, `forward=2.2s`, `wall=31.9s` (from Part
A's main-process split, pool4 -- step_wait/forward are pool-size-insensitive, confirmed
there). Since `step_wait` (env + opponent inference) is untouched by `--device` and
measured identical between the two runs (23.0s both), the difference is entirely in the
PPO update: `wall - step_wait - forward` is `31.9 - 23.0 - 2.2 = 6.7s` on CPU vs.
`28.0 - 23.0 - 2.2 = 2.8s` on CUDA -- **a 2.4x speedup on the update step itself**, clearing
the pre-registered `>=2x` rule. Because `step_wait` is identical across the pair, this
isolates the update-step effect cleanly rather than comparing two noisy end-to-end totals:
the 12% overall wall-time saving (`31.9s -> 28.0s`) matches Amdahl directly --
`21% x (1 - 1/2.4) ≈ 12%`, where 21% is the update's measured share of CPU wall time.
**Verdict: adopt as an opt-in, verified option** -- correct (Part C.2), no regression, a
real if modest end-to-end win limited by the update's ~21% share of wall time.

### C.5 BC pre-training device (`rl.bc train --device cuda`)

Isolated forward+backward+`optimizer.step()` benchmark, batch_size=1024, net_arch
`[256,256]` (BC's actual config), 50 batches after warmup:

| Device | ms/batch | batches/s |
|---|---|---|
| CPU | 39.45 | 25.3 |
| CUDA | **1.70** | **588.8** |

**~23x faster.** Log 004's full BC dataset (456,707 train samples, batch_size=1024) is
~446 batches/epoch; at 30 epochs that's ~13,380 batches -- **~529s (8.8 min) on CPU vs
~23s on CUDA**. The absolute saving is modest (004's whole BC pipeline -- generation,
training, clone benchmark -- took under 10 minutes combined), but the win is unambiguous,
free of the opponent-inference server's IPC problem (everything happens in one process,
no pipes), and bit-identical to the CPU result on a real dataset (Part C.2).
**Verdict: adopt as an opt-in `--device cuda` flag**, default stays `cpu` (no CUDA
dependency for routine BC runs).

## Overall verdict

| Component | Verdict |
|---|---|
| Opponent-checkpoint inference (self-play) | **Rejected**, capped at ~1.35x by Amdahl (25.8% share) regardless of implementation, and this implementation never even reached that: measured mean batch size 1.04-1.28 because `wait()` returns on the first ready connection, not a true collection window. A draining fix was tried and reverted (regressed throughput ~3x by adding full-window latency to every sparse request). Code kept as a documented, non-default option (`--opponent-inference {cpu,cuda}`); pool1's own +10% is real and consistent with the mechanism, not a fluke. |
| Learner inference (rollout) | Rejected outright by the Part A gate (9.2% < 20%) -- never built. |
| PPO update | **Adopted as opt-in** (`--device cuda`), a measured 2.4x on the update step itself (~12% end-to-end, matching Amdahl at its ~21% wall-time share), no regression. Default stays `cpu`. |
| BC pre-training | **Adopted as opt-in** (`rl.bc train --device cuda`), ~23x on the compute step, output identical to printed precision vs. CPU. Default stays `cpu`. |
| Benchmarks with RL opponents | Measured, not adopted: the RL seat's own single-sample CPU inference is 27.1% of worker time (`rl.profile worker --baseline-mix 1.0`), but single-sample `predict` is *slower* on CUDA (Part A), and a benchmark worker forms no natural batch. A 400-game `rl_vs_heuristic` run already completes in ~10s wall clock -- not worth pursuing further. |

**Packaging**: `pytorch-cpu` stays the default install (`rl-train` extra). A new
`rl-train-cuda` extra (same deps, `torch` sourced from `pytorch-cu130`) is opt-in,
declared conflicting with `rl-train` via `[tool.uv] conflicts` so the two can never
resolve into one environment, and CI's `uv sync --dev --extra rl` never touches it. The
CUDA venv lives on the external HDD (`.venv-cuda` symlink -> `/media/guido/.../catan-cuda-venv`,
gitignored), matching truco-py's own precedent, because the CUDA stack (~4GB of
`nvidia-*`/`triton` wheels) does not fit this machine's 23GB free on `/`. See README.md's
"Opting into CUDA" section for the exact commands.

## Commands (all of the above, verbatim)

```
# Isolated profiling pools (never the real self-play dirs)
mkdir -p rl_runs/gpu005/pool{1,4,14}/prof{1,4,14}
# ... cp specific rl_runs/selfplay/catan_bc_ft/*.zip checkpoints into each, renamed
# to prof{N}_<step>.zip (the glob rl.train.OpponentPool expects)

# Worker-side split
uv run python -m rl.profile worker --pool rl_runs/gpu005/pool14 --label prof14 \
  --learner rl_runs/bc/catan_bc_clone.zip --steps 20000 [--baseline-mix 0.0]

# Main-process split / end-to-end FPS
uv run python -m rl.profile train --pool rl_runs/gpu005/pool14 --label prof14 \
  --envs 16 --chunk 20000 --chunks 2 [--opponent-inference {cpu,cuda}] [--device cuda]

# Latency micro-bench
uv run python -m rl.profile latency --checkpoint rl_runs/bc/catan_bc_clone.zip --device cpu
.venv-cuda/bin/python -m rl.profile latency --checkpoint rl_runs/bc/catan_bc_clone.zip --device cuda

# Equivalence tests (both venvs)
uv run pytest tests/rl/test_inference_server.py tests/rl/test_cpu_eval_model.py -q
.venv-cuda/bin/python -m pytest tests/rl/test_inference_server.py -q

# Benchmark timing / RL-seat-inference approximation
uv run python -m rl.profile worker --pool rl_runs/gpu005/pool14 --label prof14 \
  --learner rl_runs/bc/catan_bc_clone.zip --steps 20000 --baseline-mix 1.0
uv run python -m experiments.benchmark --mode rl_vs_heuristic --games 400 --players 4 \
  --engine-seed-base 1 --driver-seed-base 1 --workers 16 \
  --checkpoint rl_runs/selfplay/catan_bc_ft/catan_bc_ft_2000000.zip
```

## Environment

- catan commit: this PR's own (`rl/gpu-inference-005` branch).
- gamekit `0.3.0`, torch `2.14.0+cpu` (`.venv`) / `2.14.0+cu130` (`.venv-cuda`),
  sb3-contrib `2.9.0`, stable-baselines3 `2.9.0`.
- Machine: RTX 4050 Laptop (6GB), CUDA 13.2 driver, 20 logical cores, shared with other
  interactive work during profiling (other coding-agent sessions were observed at 40-70%
  CPU each at points during this session) -- FPS numbers vary run-to-run by roughly
  ±15-20% for nominally identical configs even without that contention, and considerably
  more under it. Runs used for the C.3/C.4 comparisons were taken when `ps aux` showed no
  other heavy process competing, specifically to keep that noise out of the numbers
  reported there; the batch-size (`mean_group_size`) findings are contention-independent
  (a count, not a timing) and are the load-bearing evidence for the root-cause claim.

## Notes / follow-up

- **The real fix, not attempted here:** a grace period after the *first* arrival (e.g.
  0.5-1ms, well under `_BATCH_WAIT_S`) that extends only while new requests keep arriving,
  rather than either returning on the first signal (current, mean group size ~1) or always
  waiting out a fixed deadline regardless of arrivals (tried, regressed ~3x). This needs
  care to avoid reintroducing the same failure mode at a smaller time constant, and wasn't
  attempted given the ~1.35x Amdahl ceiling already capping the plausible upside.
- Even a perfectly-batching server is capped at ~1.35x end-to-end by Amdahl's law at this
  component's 25.8% wall-time share (Part A) -- worth stating explicitly, since it bounds
  how much any future fix here could be worth before attempting one.
- `experiments/benchmark.py`'s RL-vs-RL paths were not wired to `--opponent-inference` --
  measured instead via the worker-side `--baseline-mix 1.0` proxy and a timed run (Part
  C.3); if a future benchmark needs many concurrent RL-vs-RL games (larger n, more workers
  hitting the same checkpoints simultaneously), revisit with this doc's `stats()`-based
  method rather than assuming this verdict transfers.
- Candidate gamekit technique note (proposed in this PR's description, not written here):
  "compute the Amdahl ceiling before building a batching inference server, and instrument
  its actual batch size before trusting a micro-benchmark's promise -- `wait(conns,
  timeout)` returning on the first ready connection, not collecting a window, is an easy
  way to ship a server that never batches at all." catan's numbers (this doc) as the
  evidence; truco-py's opposite result as the contrast case worth understanding further --
  its own session notes don't record a batch-size measurement either, so the two results
  are not yet reconciled, only both honestly reported.
