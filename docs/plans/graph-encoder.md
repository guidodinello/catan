# Board-aware (graph) policy network: implementation plan (issue #42)

**Status:** plan only; no code exists yet. The pre-registered experiment is [log 014](../experiments/014-graph-encoder.md). The code lands in a later PR, after the pre-registration merges.
**Note:** [gamekit#008 — Board-aware encoder](https://github.com/guidodinello/gamekit/blob/main/docs/research/008-board-aware-encoder.md).

## Why

Log 004's BC clone of `HeuristicAgent` reaches 51.6% / 55.4% / 50.5% masked accuracy on the `vertex_settlement` / `edge` / `hex` atom blocks, against 76-95% on the others. The train/val gap on those blocks is only about 8 points, so log 004 reads it as a representation limit rather than a data shortage. The flat `[256, 256]` MLP sees the board as a bag of per-entity features with no adjacency. A network that passes messages over the hex / vertex / edge incidence is the cheapest way to give it that structure.

## Starting facts (from the code at `main`)

- The policy is stock `MaskablePPO("MlpPolicy", net_arch=[256, 256])`, built in one place (`rl/train.py`, `build_model`). pi and vf have disjoint parameters (`tests/rl/test_bc.py`).
- Every consumer runs the chain `policy.extract_features → mlp_extractor → action_net / value_net`: `rl/bc.py:_forward`, `agents/rl_agent.py:lean_predict_batch`, `agents/rl_search.py` (`PolicyEvaluator`, `SelfModelOpponent`), `rl/inference_server.py`, `rl/profile.py`.
- The observation is a flat 1484-dim vector. Its per-entity blocks have exported offsets in `rl/encoder.py`: HEX 19×8, VERTEX 54×14, EDGE 72×5, then OWN, OPPONENT, GLOBAL, BUFFER (its last 72 dims are a per-edge chosen-edge mask), TRADE, and 24 reserved zero dims.
- Spatial atoms index entities one-to-one (`rl/action_space.py`): 0-53 settlement@vertex, 54-107 city@vertex, 108-179 road@edge, 180-198 robber@land-hex; 199-222 are non-spatial.
- Topology lives in `engine/board.py` as `GEOMETRY` (`vertex_hexes`, `vertex_edges`, `edge_vertices`, `hex_vertices`, `hex_edges`, …) as Python tuples. Land-hex order is `GEOMETRY.land_hexes`, the same order as `HEX_INDEX`.
- gamekit v0.4.0 has no torch or GNN code, and no GNN library is installed.

## Design

### Change the network, not the observation

Keep the 1484-dim obs, `OBS_DIM`, the env, the action space and the BC dataset exactly as they are. The graph network re-slices the flat obs with the offsets imported from `rl/encoder.py`. Consequences:

- "same BC data" is literal: `rl_runs/bc/v1` is reused;
- `league_013`'s `observation_space` check against `OBS_DIM` still holds;
- `RLAgent`, `RLSearchAgent` and the inference server need no change, because they only call the chain above.

### `rl/graph_policy.py` (pure torch, no PyG)

- **Static topology.** Index tensors built once from `GEOMETRY` and registered as buffers: hex→vertex (19×6), edge→vertex (72×2), plus padded vertex→edge and vertex→hex tables with validity masks.
- **Node features.** Hex 8, vertex 14, edge 5 + 1. The extra edge feature is the BUFFER's chosen-edge mask, moved from the global block to where it belongs.
- **Context vector.** OWN + OPPONENT + GLOBAL + the non-edge part of BUFFER + TRADE, through a small MLP, then broadcast into every node update. (120 dims in total, derived from the encoder's constants in `rl/graph_policy.py`.)
- **Trunk.** L = 3 rounds of message passing (hex→vertex, edge→vertex, vertex→edge, vertex→hex), gather + masked mean, d = 64, LayerNorm and residuals. No node-ID embeddings, so the trunk is equivariant to board symmetries.
- **Policy head (pointer style).** Settlement and city logits from per-vertex MLPs over `[h_v ‖ ctx]`, road logits from a per-edge MLP, robber logits from a per-hex MLP. The non-spatial atoms 199-222 come from an MLP over `[pooled nodes ‖ ctx]`. Concatenated in atom order to width 223.
- **Value head.** MLP over `[mean/max pooled nodes ‖ ctx]`.
- **Keeping the call chain intact.** `share_features_extractor=True` with the usual Flatten extractor. A custom `mlp_extractor` holds two independent trunks (pi and vf) and returns `latent_pi` (the flattened node embeddings plus ctx) and `latent_vf`. `action_net` is the pointer head and unflattens `latent_pi`. pi and vf stay disjoint, as in the flat arm, so the value loss does not shape the policy's features. A shared trunk would change two things at once (architecture and loss coupling) and is deliberately not used.
- **`GraphMaskablePolicy(MaskableActorCriticPolicy)`** overrides `_build_mlp_extractor` and builds the pointer `action_net`. Checkpoints load through `MaskablePPO.load` while the module is importable.

### Wiring

- `rl/train.py`: `--policy {mlp,graph}` (default `mlp`), `--gnn-dim`, `--gnn-layers`; `build_model` branches. `--bc-init` and `--resume` take the policy class from the checkpoint.
- `rl/bc.py train`: `--policy`, and a `--select-split` option so early stopping reads a *tuning* split instead of the validation split (today `train_bc` early-stops on val). A third split by game, `game_id % 10 == 1`, is carved from train; `game_id % 10 == 0` stays sealed.
- `rl/bc.py`: a pooled spatial accuracy (`vertex_settlement` + `edge` + `hex`) on val and, with `--select-split tune`, on the tuning split. **The tie-aware secondary metric is dropped:** the BC dataset stores observations, masks and labels but not `HeuristicAgent`'s per-decision score ties, so the tied set cannot be made exact without replaying every game. The tie-break bias stays disclosed (Risks) and un-measured.
- `experiments/search_eval.py`-style seed registry: add `fresh_014`.
- `experiments/benchmark.py` needs no change; it loads checkpoints through `RLAgent`.

## Tests (`tests/rl/test_graph_policy.py`, `importorskip("torch")`)

1. Topology tensors agree with `GEOMETRY`: every edge has 2 vertices, every land hex 6, incidence is symmetric.
2. **Slicing coverage:** every non-reserved obs dim is consumed exactly once (offsets imported, not hardcoded), and the per-node tensors equal the encoder's per-entity blocks on a real mid-game state. The BUFFER edge mask and the TRADE block are the easy ones to drop silently.
3. **Equivariance:** a 60° rotation of cube coordinates induces a vertex/edge/hex permutation; permuting the obs permutes the spatial logits the same way and leaves the value and non-spatial logits unchanged.
4. Output width is 223 and masked argmax never picks an illegal atom.
5. `rl.bc._forward` matches `model.predict` for the graph policy (mirrors `test_forward_matches_inference`).
6. pi and vf parameter sets are disjoint.
7. Save/load round trip, then `RLAgent` plays a full game and `RLSearchAgent` makes one searched decision at S = 2.
8. Tiny end-to-end `train_bc --policy graph` and a few-thousand-step `rl.train --policy graph`.
9. The same seed gives the same game digest.
10. `fresh_014` is disjoint from `league_013.USED_RANGES`, `fresh_012`, the BC boards 1..3000 and the in-loop eval range (about 12.05 M and below).

## Cheap smoke (minutes; reads no win rate)

On a **throwaway** dataset (200 games, about 33k rows, so GPU warm-up does not dominate a per-epoch figure; smoke seeds from 4,000,501, written under `rl_runs/exp014/smoke/`, never `rl_runs/bc/v1`), run BC for each arm (an untimed warm-up, then 1-epoch and 3-epoch runs whose time difference gives seconds per epoch) and record:

- BC seconds per epoch (GPU);
- **CPU single-sample `lean_predict_batch` latency**, because self-play opponents and the in-loop eval run on CPU (log 005);
- PPO fps on a 16k-step `--envs 8` run;
- parameter counts for both arms (the flat net is about 1 M).

It runs in a named tmux session with a log file. Its numbers go into the amendment in log 014.

## Harness check (after the code PR, before any measured run)

The flat `catan_bc_ft_long_10031616` checkpoint on benchmark boards 1..4000 must reproduce exactly **829/4000**, as in log 012. This catches any regression from touching `build_model`.

## Risks

- **Tie-breaks vs equivariance.** `HeuristicAgent` breaks score ties by enumeration (vertex-ID) order (log 004). An equivariant trunk cannot learn that, so plain masked accuracy partly penalises the property under test. Disclosed in log 014; the tie-aware metric is the mitigation. Node-ID embeddings would remove the bias but also the equivariance claim, so they are not used.
- **CPU latency.** Graph snapshots serve as self-play opponents on CPU. If the smoke projects a single graph fine-tune over 8 h, d / L shrink before launch (the stop rule in log 014).
- **Environment.** The local `.venv` was stale at planning time (gamekit 0.3.0, torch 2.14.0, against a lock of 0.4.0 and 2.14.1). The HDD behind `rl_runs` and `.venv-cuda` was unmounted at the start of the planning session and mounted later in it; `rl_runs/bc/v1` is intact (hashes in log 014). Measured runs go through a worktree with `uv sync --frozen`, not the main checkout's `.venv`.
