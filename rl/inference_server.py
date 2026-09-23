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

**Batching is why this exists at all.** ``docs/experiments/005-gpu-inference.md``
measured ``agents.rl_agent.lean_predict_batch`` at ~100us/call regardless
of batch size on GPU (launch-latency-bound), while CPU cost scales with
batch size -- so the win comes from however many concurrent requests
``connection.wait()`` collects inside one ``_BATCH_WAIT_S`` window, grouped
by checkpoint path so each distinct model gets one batched forward per
round, exactly as truco-py's server does.

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

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                ready = cast(
                    list[Connection], wait(self._server_conns, timeout=_BATCH_WAIT_S)
                )
                if not ready:
                    continue
                batch: list[tuple[Connection, str, np.ndarray, np.ndarray]] = []
                for conn in ready:
                    try:
                        checkpoint, obs, mask = conn.recv()
                        batch.append((conn, checkpoint, obs, mask))
                    except EOFError, OSError:
                        pass  # subprocess exited -- skip
                self._process_batch(batch)
            except Exception:
                logger.exception("InferenceServer: error in serve loop")

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
