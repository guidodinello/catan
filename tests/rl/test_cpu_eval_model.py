"""Rule under test: in-loop eval must never perturb the training run's own
randomness. ``rl.train._cpu_eval_model`` exists specifically to keep
single-sample eval off a CUDA-device PPO model (measured slower than CPU at
batch size 1 -- ``docs/experiments/005-gpu-inference.md``).

It has to build a *fresh* ``MaskablePPO`` via ``rl.train.build_model``
(``copy.deepcopy(model.policy)`` was tried first and rejected: torch's tensor
``__deepcopy__`` only supports graph-leaf tensors, and a policy that has
actually been optimized isn't one -- confirmed empirically, a real
``RuntimeError`` from inside a real training run's first eval chunk).
``MaskablePPO.__init__`` -> ``_setup_model()`` calls stable-baselines3's
``set_random_seed``, which reseeds the *global* Python/numpy/torch RNGs as an
unconditional side effect (confirmed against the installed stable-baselines3
source) -- so every in-loop eval chunk during a real ``--device cuda`` run
would silently reset the training run's own env/opponent-pool/action-sampling
randomness, unless that reseed is undone. The fix saves the global RNG state
before calling ``build_model`` and restores it in a ``finally``, regardless of
having to construct a fresh model.

Imports torch directly, so this module is skipped in the torch-free
``Tests (RL)`` CI job, same as ``tests/rl/test_bc.py``'s training-side tests.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

pytest.importorskip("gymnasium")

import numpy as np  # noqa: E402

torch = pytest.importorskip("torch")

from rl.train import (  # noqa: E402
    TrainConfig,
    _cpu_eval_model,
    build_model,
    build_vec_env,
)


def _rng_snapshot() -> tuple[Any, Any, Any]:
    return random.getstate(), np.random.get_state(), torch.get_rng_state()


def test_cpu_eval_model_is_a_noop_on_cpu() -> None:
    """The documented no-op path: ``cfg.device == "cpu"`` returns the exact
    same model object, not a copy."""
    cfg = TrainConfig(envs=1, num_players=4, opponents="heuristic", device="cpu")
    vec_env = build_vec_env(cfg)
    model = build_model(cfg, vec_env)
    assert _cpu_eval_model(cfg, model) is model


def test_cpu_eval_model_never_reseeds_global_rng() -> None:
    """Pins the actual bug: calling ``_cpu_eval_model`` with
    ``cfg.device != "cpu"`` must leave the global Python/numpy/torch RNG
    state exactly as it found it, even though it builds a fresh
    ``MaskablePPO`` internally (via ``build_model``) to get a CPU-resident
    copy of the policy.

    No CUDA device is needed to exercise this: ``_cpu_eval_model`` branches
    on ``cfg.device`` alone, and its non-"cpu" branch builds the fresh model
    with ``device="cpu"`` regardless (only ``model.env`` and the loaded
    ``state_dict`` come from ``model`` itself) -- the exact same code path a
    genuine ``--device cuda`` run takes.
    """
    cfg = TrainConfig(envs=1, num_players=4, opponents="heuristic", device="cpu")
    vec_env = build_vec_env(cfg)
    model = build_model(cfg, vec_env)

    cuda_cfg = TrainConfig(envs=1, num_players=4, opponents="heuristic", device="cuda")
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    before = _rng_snapshot()

    _cpu_eval_model(cuda_cfg, model)

    after = _rng_snapshot()
    assert random.getstate() == before[0] == after[0]
    np.testing.assert_array_equal(before[1][1], after[1][1])
    assert torch.equal(before[2], after[2])
