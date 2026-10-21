"""Score saved sweep clones on the TUNE split only (game_id % 10 == 1).

Never touches val: the sweep dataset contains no val game."""

import sys
from pathlib import Path

from sb3_contrib import MaskablePPO

from rl.bc import DEFAULT_EVAL_CHUNK, _score_split, game_split_with_tune, load_dataset

ds = load_dataset(
    Path("/home/guido/projects/catan/rl_runs/exp014/bc_sweep_ds"), mmap=False
)
_, tune_idx, val_idx = game_split_with_tune(ds.game_id)
assert len(val_idx) == 0 and len(tune_idx) > 0
for path in sys.argv[1:]:
    policy = MaskablePPO.load(path, device="cpu").policy
    s = _score_split(policy, ds, tune_idx, DEFAULT_EVAL_CHUNK, "cpu")
    print(
        f"{Path(path).name}: tune_spatial_acc={s.spatial_acc * 100:.4f}% "
        f"(n_tune_rows={s.n})"
    )
