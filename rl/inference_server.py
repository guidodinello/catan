"""Centralized batched inference server for self-play opponent checkpoints.

Modeled directly on truco-py's ``training/inference_server.py`` (issue #20),
adapted to two things that differ here: catan's ``OpponentPool`` samples
*uniformly over every checkpoint in the pool* (up to a dozen-plus, not one),
and its checkpoint agents make several single-atom decisions per engine
action (``rl.action_space``'s composition buffer), not one. Both raise the
per-env request rate this server has to absorb, which is exactly why
``docs/experiments/005-gpu-inference.md`` measured the design before
building it rather than porting it on the strength of truco's numbers alone.

**Why a thread, not a process** (truco-py's own choice): ``SubprocVecEnv``
is synchronous from the main process's point of view -- it blocks in
``Connection.recv()`` while workers step, which releases the GIL, so a
daemon thread in the main process can serve requests without contending
with the vec-env's own control flow for the interpreter lock. A second
process would need its own IPC hop to reach the main process's vec-env
loop, buying nothing here.

**Batching is why this exists at all -- and this port never actually
batches.** ``docs/experiments/005-gpu-inference.md`` measured
``agents.rl_agent.lean_predict_batch`` at ~100us/call regardless of batch
size on GPU (launch-latency-bound), while CPU cost scales with batch size,
so the whole design depends on how many concurrent requests land in one
grouped forward. ``connection.wait(conns, timeout)`` below returns as soon
as *any* connection is ready -- it is a max-wait, not a collection window --
and measured at a **mean group size of ~1.0-1.3** (i.e. essentially never
batching) even at 16 concurrent envs and as few as 1 distinct checkpoint.

A fix was attempted and reverted: draining newly-ready connections for the
rest of the ``_BATCH_WAIT_S`` window (instead of returning on the first
`wait()`) does turn the timeout into a real collection window, but measured
as a severe regression, not a fix -- when the *next* concurrent request is
sparse (the common case here), every single decision now pays close to the
full window as pure added latency with nothing to show for it, collapsing
effective server throughput to roughly ``1 / _BATCH_WAIT_S`` requests per
second server-wide. A correct version would need a short grace period after
the *first* arrival (well under ``_BATCH_WAIT_S``) rather than a fixed
deadline regardless of what has already arrived -- untried here; see
``docs/experiments/005-gpu-inference.md``'s notes.

**Determinism matches ``RLAgent``'s own default.** truco's server served
``deterministic=False``; this one serves ``deterministic=True`` (masked
argmax, which is what ``lean_predict_batch`` computes) so that switching an
opponent from local ``RLAgent`` inference to this server never changes what
it plays -- confirmed by ``tests/rl/test_inference_server.py``'s equivalence
test against ``model.predict(..., deterministic=True)``.

**Fork ordering.** Both duplex pipe ends must exist *before*
``SubprocVecEnv(start_method="fork")`` forks -- otherwise cloudpickle has to
serialize a live ``Connection`` across a process boundary that isn't a
plain fork, which is the exact failure truco-py hit with the forkserver
default. This module's constructor only creates the pipes and starts the
serving thread; ``rl.train.build_vec_env`` is responsible for constructing
the ``InferenceServer`` (and therefore all pipes) before calling
``SubprocVecEnv``, and workers must never import or touch CUDA themselves.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pipe
from multiprocessing.connection import Connection, wait
from typing import TYPE_CHECKING, cast

import numpy as np

from agents.rl_agent import lean_predict_batch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sb3_contrib import MaskablePPO

logger = logging.getLogger("rl.inference_server")

_BATCH_WAIT_S = 0.005  # seconds to collect concurrent requests before a batched forward


@dataclass(slots=True)
class InferenceHandle:
    """Picklable proxy held by each subprocess env's ``RLAgent`` opponents.

    Fork-inherits ``conn`` from the parent process (created before the
    fork, per this module's docstring) -- never itself pickled across a
    process boundary that isn't a plain fork.
    """

    conn: Connection

    def request(self, checkpoint: str, obs: np.ndarray, mask: np.ndarray) -> int:
        self.conn.send((checkpoint, obs, mask))
        return cast(int, self.conn.recv())


class InferenceServer:
    """Owns one ``MaskablePPO`` per distinct checkpoint path on ``device``,
    served by a daemon thread that batches concurrent requests across every
    env's pipe.

    Usage (see ``rl.train.build_vec_env``)::

        server = InferenceServer(n_envs=cfg.envs, device=cfg.opponent_inference)
        handles = [server.make_handle(rank) for rank in range(cfg.envs)]
        # ... build env_fns with each rank's handle,
        # then SubprocVecEnv(..., start_method="fork")
    """

    def __init__(self, n_envs: int, device: str) -> None:
        self._device = device
        self._models: dict[str, MaskablePPO] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        # Batch-size accounting: docs/experiments/005-gpu-inference.md's
        # whole verdict on this design turns on how big the per-checkpoint
        # groups `_process_batch` actually forms in practice, not on the
        # micro-benchmark's assumed batch sizes -- see `stats()`.
        self._total_requests = 0
        self._total_groups = 0

        self._server_conns: list[Connection] = []
        self._client_conns: list[Connection] = []
        for _ in range(n_envs):
            server_conn, client_conn = Pipe(duplex=True)
            self._server_conns.append(server_conn)
            self._client_conns.append(client_conn)

        self._thread = threading.Thread(
            target=self._serve, daemon=True, name="rl-inference-server"
        )
        self._thread.start()
        logger.info("InferenceServer started device=%s n_envs=%d", device, n_envs)

    def make_handle(self, env_rank: int) -> InferenceHandle:
        return InferenceHandle(conn=self._client_conns[env_rank])

    def stop(self) -> None:
        self._stop.set()

    def stats(self) -> tuple[int, int, float]:
        """``(total_requests, total_groups, mean_group_size)`` since this
        server started -- one call per ``_process_batch`` invocation grouped
        by checkpoint path, so ``mean_group_size`` is the actual batch size
        ``lean_predict_batch`` runs at, not the batch size a micro-benchmark
        assumed. A mean near 1 means requests are never really batching,
        regardless of how many env pipes exist."""
        with self._lock:
            requests, groups = self._total_requests, self._total_groups
        mean = requests / groups if groups else float("nan")
        return requests, groups, mean

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                self._process_batch(self._collect_batch())
            except Exception:
                logger.exception("InferenceServer: error in serve loop")

    def _collect_batch(self) -> list[tuple[Connection, str, np.ndarray, np.ndarray]]:
        """Whatever ``connection.wait`` returns within one ``_BATCH_WAIT_S``
        window.

        This is the shipped version, not the "collection window" redesign
        this module's docstring describes and rejects: that version drains
        newly-ready connections for the rest of the deadline instead of
        returning on the first signal, which does turn ``_BATCH_WAIT_S``
        into a real window -- but measured as a severe throughput
        regression (every request pays close to the full window as added
        latency when nothing else arrives, which is the common case here).
        This version has no artificial added latency at all: it returns the
        moment anything is ready, exactly like a bare ``wait()`` call would,
        at the cost of the mean group size measured in the module docstring.
        """
        ready = cast(list[Connection], wait(self._server_conns, timeout=_BATCH_WAIT_S))
        batch: list[tuple[Connection, str, np.ndarray, np.ndarray]] = []
        for conn in ready:
            try:
                checkpoint, obs, mask = conn.recv()
                batch.append((conn, checkpoint, obs, mask))
            except EOFError, OSError:
                pass  # subprocess exited -- skip
        return batch

    def _load(self, checkpoint: str) -> MaskablePPO:
        with self._lock:
            model = self._models.get(checkpoint)
        if model is not None:
            return model
        from sb3_contrib import MaskablePPO

        # Loaded outside the lock so a slow first load of a new checkpoint
        # doesn't block other checkpoints' already-loaded models from being
        # read by concurrent callers -- the dict write below is the only
        # part that needs the lock.
        model = MaskablePPO.load(checkpoint, device=self._device)
        with self._lock:
            self._models[checkpoint] = model
        logger.info(
            "InferenceServer: loaded %s -> %s (pool_size=%d)",
            checkpoint,
            self._device,
            len(self._models),
        )
        return model

    def _process_batch(
        self, batch: list[tuple[Connection, str, np.ndarray, np.ndarray]]
    ) -> None:
        if not batch:
            return
        groups: dict[str, list[tuple[Connection, str, np.ndarray, np.ndarray]]] = (
            defaultdict(list)
        )
        for item in batch:
            groups[item[1]].append(item)
        with self._lock:
            self._total_requests += len(batch)
            self._total_groups += len(groups)

        for checkpoint, items in groups.items():
            try:
                model = self._load(checkpoint)
                obs_arr = np.stack([item[2] for item in items])
                mask_arr = np.stack([item[3] for item in items])
                atoms = lean_predict_batch(
                    model.policy, obs_arr, mask_arr, self._device
                )
                for (conn, _, _, _), atom in zip(items, atoms, strict=True):
                    conn.send(int(atom))
            except Exception:
                logger.exception(
                    "InferenceServer: error serving %s -- sending fallback actions",
                    checkpoint,
                )
                for conn, _, _, mask in items:
                    legal = np.flatnonzero(mask)
                    conn.send(int(legal[0]) if len(legal) else 0)
