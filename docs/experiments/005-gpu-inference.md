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

All pass, in both the CPU `.venv` and the CUDA `.venv-cuda`:

- `lean_predict_batch` picks the identical atom as `model.predict(deterministic=True)`,
  row for row, on 32 random observation/mask pairs.
- Batching is size-invariant: one N-row call matches N separate one-row calls, bit for bit.
- `InferenceHandle.request` through the real pipe/thread machinery matches local `RLAgent`
  inference, including under concurrent requests from 6 simultaneous env threads.
- CUDA-vs-CPU: identical atoms on 64 rows (test skipped where no CUDA device exists).
- `rl.bc train --device cuda` on an identical tiny dataset (40 games, seed-fixed) produced
  **bit-identical** `train_loss`/`val_masked_acc`/`val_value_mse` per epoch to the CPU run --
  only wall time differed.

### C.3 End-to-end FPS: opponent-checkpoint inference server (the headline measurement)

```
uv run python -m rl.profile train --pool rl_runs/gpu005/pool{N} --label prof{N} \
  --envs 16 --chunk 20000 --chunks 2 [--opponent-inference {cpu,cuda}] [--baseline-mix 0.0]
```

| Pool | baseline_mix | opponent-inference=none (baseline) | =cpu (server) | =cuda (server) |
|---|---|---|---|---|
| pool1 (1 ckpt, best case for batching) | 0.5 | 637 fps | 664 fps (+4%, noise-level) | -- |
| pool14 (12 ckpt) | 0.5 | 626 fps | **514 fps (-18%)** | **573 fps (-8%)** |
| pool14 (12 ckpt, worst case) | 0.0 | 573 fps | -- | **396 fps and falling (-31%+, chunk cut short)** |

**Verdict: reject.** The centralized inference server -- both CPU and CUDA variants --
does not net a throughput win, and regresses it once the pool holds more than a
single checkpoint. This directly contradicts the Part A micro-benchmark's promise (which
measured raw forward-pass compute only). Two things explain the gap:

1. **IPC round-trip cost.** Each atom decision now costs a pipe `send` + `recv` and a
   thread wake-up that didn't exist when the model lived in the worker process. That
   per-decision overhead is comparable to, or larger than, the compute this design saves.
2. **Batch fragmentation.** `OpponentPool.sample()` draws uniformly from every checkpoint
   in the pool -- 12 distinct paths in `pool14`. `_process_batch` groups by path, so a
   5ms collection window that might hold 16 concurrent requests fragments into up to 12
   groups of ~1, defeating the batching the whole design depends on. This gets *worse*,
   not better, as the pool grows -- the opposite of truco-py's single-checkpoint case,
   where every request groups into one batch by construction.

Both run in the expected direction: the server helps a little (within noise) at pool
size 1, and hurts more as the pool fragments requests across more checkpoints -- exactly
the mechanism above, not a fluke.

**Runtime code is kept, not deleted**, because it is correctness-verified and could still
help a different workload (fewer, more concurrent envs; a pool capped to one or two
checkpoints) -- but `--opponent-inference` defaults to `none` and this doc recommends
leaving it there. `docs/experiments/README.md`'s index and `pyproject.toml`'s torch
comment both say so.

### C.4 End-to-end FPS: PPO update device (`--device cuda`, opponent inference untouched)

```
.venv-cuda/bin/python -m rl.profile train --pool rl_runs/gpu005/pool14 --label prof14 \
  --envs 16 --chunk 20000 --chunks 2 --device cuda
# steady-state: fps=713 step_wait=23.0s forward=2.2s wall=28.0s
```

**713 fps** vs the `none`/`cpu` baseline's 626-768 fps range across repeated runs on this
shared, busy laptop -- inside that range, not clearly outside it. The isolated
forward+backward+`optimizer.step()` micro-benchmark below shows the underlying win is
real (23x); it is just small relative to the ~69% of wall time spent on
env+opponent-inference (`step_wait`), which `--device` does not touch. **Verdict: keep as
an opt-in, verified option** (correct, no regression, small-but-real improvement), not a
default -- the modest gain doesn't obviously outweigh keeping the simpler all-CPU path for
routine runs.

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
| Opponent-checkpoint inference (self-play) | **Rejected.** Centralized batched server (CPU or CUDA) regresses throughput once the pool holds more than ~1 checkpoint, due to IPC overhead and batch fragmentation across distinct checkpoints. Code kept as a documented, non-default option (`--opponent-inference {cpu,cuda}`). |
| Learner inference (rollout) | Rejected outright by the Part A gate (9.2% < 20%) -- never built. |
| PPO update | **Adopted as opt-in** (`--device cuda`), small-but-real, no regression. Default stays `cpu`. |
| BC pre-training | **Adopted as opt-in** (`rl.bc train --device cuda`), ~23x on the compute step, bit-identical output. Default stays `cpu`. |
| Benchmarks with RL opponents | Not separately measured -- `experiments/benchmark.py`'s RL-vs-X paths are unchanged by this PR (they never touch `--opponent-inference`); a candidate follow-up issue if a future benchmark needs many concurrent RL-vs-RL games. |

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
uv run pytest tests/rl/test_inference_server.py -q
.venv-cuda/bin/python -m pytest tests/rl/test_inference_server.py -q
```

## Environment

- catan commit: this PR's own (`rl/gpu-inference-005` branch).
- gamekit `0.3.0`, torch `2.14.0+cpu` (`.venv`) / `2.14.0+cu130` (`.venv-cuda`),
  sb3-contrib `2.9.0`, stable-baselines3 `2.9.0`.
- Machine: RTX 4050 Laptop (6GB), CUDA 13.2 driver, 20 logical cores, shared with other
  work during profiling -- FPS numbers vary run-to-run by roughly ±15-20% for nominally
  identical configs on this machine, which is why every comparison above is stated with
  that noise band in mind rather than as a single point estimate.

## Notes / follow-up

- The server's `_BATCH_WAIT_S=5ms` collection window and its threading design were not
  swept -- a shorter window, a process instead of a thread, or capping the pool to 1-2
  concurrent checkpoints (at the cost of self-play diversity) might change the verdict.
  Untried here because the measured direction (regression, worsening with pool size) was
  clear enough to not chase further within this issue's scope.
- `experiments/benchmark.py`'s RL-vs-RL paths were not wired to `--opponent-inference` or
  profiled separately; if a future benchmark needs many concurrent RL-vs-RL games, revisit
  with this doc's method rather than assuming the training-loop verdict transfers.
- Candidate gamekit technique note (proposed in this PR's description, not written here):
  "centralized GPU inference servers don't automatically transfer across engines -- check
  IPC overhead and opponent-pool fragmentation before porting one," with this doc's
  numbers as the catan-side evidence and truco-py's opposite result as the contrast case.
