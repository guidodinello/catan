"""Rule under test: routing an opponent's atom decisions through
``InferenceServer`` must never change what it plays -- the server's
``lean_predict_batch`` forward has to agree with ``RLAgent``'s local
``model.predict(..., deterministic=True)`` path, atom for atom, on real
observations and masks. If it didn't, self-play trained through the server
would be trained against a subtly different opponent pool than a benchmark
run that evaluates the same checkpoints locally.

Imports torch directly (via ``rl.train.build_model``/``sb3_contrib``), so
this whole module is skipped in the torch-free ``Tests (RL)`` CI job, same
as ``tests/rl/test_bc.py``'s training-side tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("gymnasium")

import numpy as np  # noqa: E402

torch = pytest.importorskip("torch")

from agents.rl_agent import lean_predict_batch  # noqa: E402
from rl.encoder import OBS_DIM  # noqa: E402
from rl.inference_server import InferenceServer  # noqa: E402
from rl.train import TrainConfig, build_model, build_vec_env  # noqa: E402


def _tiny_checkpoint(tmp_path: Path, seed: int = 0) -> Path:
    """A freshly-initialised (untrained, but architecturally real)
    MaskablePPO checkpoint -- mirrors ``tests/rl/test_bc.py``'s
    ``_tiny_model_and_vec_env`` fixture. Untrained weights are fine here:
    the property under test is "does the server reproduce local inference",
    not "is the policy any good"."""
    cfg = TrainConfig(envs=1, num_players=4, opponents="heuristic", seed=seed)
    vec_env = build_vec_env(cfg)
    model = build_model(cfg, vec_env)
    path = tmp_path / "tiny.zip"
    model.save(path)
    vec_env.close()
    return path


def _random_batch(n: int, n_atoms: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    obs = rng.random((n, OBS_DIM), dtype=np.float32)
    # A few legal atoms per row, never all-False -- mirrors a real mask's
    # shape (never every atom legal, never zero).
    mask = np.zeros((n, n_atoms), dtype=bool)
    for i in range(n):
        legal = rng.choice(n_atoms, size=rng.integers(2, 8), replace=False)
        mask[i, legal] = True
    return obs, mask


def test_lean_predict_batch_matches_sb3_predict(tmp_path: Path) -> None:
    """The server's own forward path (``lean_predict_batch``), run one row
    at a time, must pick the same atom as ``model.predict(deterministic=True)``
    -- pins the premise the server relies on before any batching or IPC is
    involved at all."""
    from sb3_contrib import MaskablePPO

    from rl.action_space import N_ATOMS

    checkpoint = _tiny_checkpoint(tmp_path)
    model = MaskablePPO.load(checkpoint, device="cpu")
    obs, mask = _random_batch(32, N_ATOMS, seed=1)

    lean_atoms = lean_predict_batch(model.policy, obs, mask, device="cpu")

    for i in range(len(obs)):
        sb3_atom, _ = model.predict(obs[i], action_masks=mask[i], deterministic=True)
        assert int(lean_atoms[i]) == int(sb3_atom), f"row {i} disagreed"


def test_lean_predict_batch_is_batch_size_invariant(tmp_path: Path) -> None:
    """Batching must not change the answer: one call with N rows must match
    N separate one-row calls -- the property ``InferenceServer`` depends on
    when it groups concurrent requests by checkpoint path into a single
    batched forward, whatever that batch size happens to be."""
    from sb3_contrib import MaskablePPO

    from rl.action_space import N_ATOMS

    checkpoint = _tiny_checkpoint(tmp_path)
    model = MaskablePPO.load(checkpoint, device="cpu")
    obs, mask = _random_batch(16, N_ATOMS, seed=2)

    batched = lean_predict_batch(model.policy, obs, mask, device="cpu")
    singles = np.array(
        [
            lean_predict_batch(model.policy, obs[i : i + 1], mask[i : i + 1], "cpu")[0]
            for i in range(len(obs))
        ]
    )
    np.testing.assert_array_equal(batched, singles)


def test_inference_server_matches_local_rl_agent(tmp_path: Path) -> None:
    """End-to-end equivalence through the real pipe/thread machinery:
    ``InferenceHandle.request`` (what ``RLAgent`` calls when wired to a
    server) must return the same atom ``RLAgent``'s own local ``model``
    property would have, for every row of a realistic batch -- the actual
    claim ``rl/inference_server.py``'s module docstring makes about
    ``deterministic=True`` parity."""
    from sb3_contrib import MaskablePPO

    from rl.action_space import N_ATOMS

    checkpoint = _tiny_checkpoint(tmp_path)
    local_model = MaskablePPO.load(checkpoint, device="cpu")
    obs, mask = _random_batch(20, N_ATOMS, seed=3)

    server = InferenceServer(n_envs=1, device="cpu")
    try:
        handle = server.make_handle(0)
        for i in range(len(obs)):
            server_atom = handle.request(str(checkpoint), obs[i], mask[i])
            local_atom, _ = local_model.predict(
                obs[i], action_masks=mask[i], deterministic=True
            )
            assert server_atom == int(local_atom), f"row {i} disagreed"
    finally:
        server.stop()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_inference_server_cuda_matches_cpu(tmp_path: Path) -> None:
    """The actual claim issue #20 needs verified: routing an opponent
    through the CUDA server must pick the identical atom the CPU path
    would, on real observations/masks -- not just "close", since a self-play
    opponent that plays differently on GPU than the checkpoint it was saved
    from would be a silent correctness regression, not a performance one.
    Skipped everywhere but the CUDA dev venv (this repo's default CI venv
    has no CUDA torch installed at all -- see pyproject.toml's rl-train-cuda
    extra)."""
    from sb3_contrib import MaskablePPO

    from rl.action_space import N_ATOMS

    checkpoint = _tiny_checkpoint(tmp_path)
    cpu_model = MaskablePPO.load(checkpoint, device="cpu")
    obs, mask = _random_batch(64, N_ATOMS, seed=5)

    server = InferenceServer(n_envs=1, device="cuda")
    try:
        handle = server.make_handle(0)
        for i in range(len(obs)):
            gpu_atom = handle.request(str(checkpoint), obs[i], mask[i])
            cpu_atom, _ = cpu_model.predict(
                obs[i], action_masks=mask[i], deterministic=True
            )
            assert gpu_atom == int(cpu_atom), f"row {i} disagreed"
    finally:
        server.stop()


def test_inference_server_batches_concurrent_requests(tmp_path: Path) -> None:
    """Requests sent from several handles at (near-)the same time land in
    one grouped batch and each gets back its own correct atom -- exercises
    ``_process_batch``'s grouping-by-path path, not just the single-request
    path the tests above cover."""
    import threading

    from sb3_contrib import MaskablePPO

    from rl.action_space import N_ATOMS

    checkpoint = _tiny_checkpoint(tmp_path)
    local_model = MaskablePPO.load(checkpoint, device="cpu")
    n_envs = 6
    obs, mask = _random_batch(n_envs, N_ATOMS, seed=4)

    server = InferenceServer(n_envs=n_envs, device="cpu")
    results: list[Any] = [None] * n_envs

    def _worker(rank: int) -> None:
        handle = server.make_handle(rank)
        results[rank] = handle.request(str(checkpoint), obs[rank], mask[rank])

    try:
        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n_envs)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        for i in range(n_envs):
            expected, _ = local_model.predict(
                obs[i], action_masks=mask[i], deterministic=True
            )
            assert results[i] == int(expected), f"env {i} disagreed"
    finally:
        server.stop()
