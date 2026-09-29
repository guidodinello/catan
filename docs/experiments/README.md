# Experiments

Run logs for catan's Phase 5 (RL) work: the numbers, exact CLI
invocations, and result JSON paths for each attempt. The hypothesis and
literature behind each attempt live one layer up, in
[gamekit's `docs/research/`](https://github.com/guidodinello/gamekit/tree/main/docs/research)
— see that repo's README for the two-layer convention this directory
follows (a log here always links back to the gamekit note it tests; a note
there links forward to the log(s) that tested it).

## Index

| id | title | verdict | gamekit note |
|---|---|---|---|
| [001](001-ppo-vs-random.md) | MaskablePPO vs 3 RandomAgents (PR #18) | validated | [003](https://github.com/guidodinello/gamekit/blob/main/docs/research/003-discount-horizon.md) |
| [002](002-selfplay-v1-baseline-mix-0.2.md) | Self-play v1 — baseline_mix 0.2 | rejected (this value) | [001](https://github.com/guidodinello/gamekit/blob/main/docs/research/001-self-play-opponent-mix.md) |
| [003](003-selfplay-v2-baseline-mix-0.5.md) | Self-play v2 — baseline_mix 0.5, ent_coef 0.02 | learning confirmed, gate not met | [001](https://github.com/guidodinello/gamekit/blob/main/docs/research/001-self-play-opponent-mix.md) |
| [004](004-bc-warm-start.md) | BC warm start + PPO self-play fine-tune (PR #21) | validated (technique-level), gate not met | [002](https://github.com/guidodinello/gamekit/blob/main/docs/research/002-bc-warm-start.md) |
| [005](005-gpu-inference.md) | GPU (CUDA) for Phase 5 RL -- opponent-checkpoint inference, PPO update, BC training | opponent-checkpoint inference server **rejected** (regresses throughput); PPO update and BC training **adopted as opt-in** `--device cuda` | (candidate, not yet filed) "GPU inference servers don't automatically transfer across engines" |
| [006](006-longer-run.md) | Longer run / resume from 004's best checkpoint (+10M steps, PR #24) | inconclusive under the pre-registered rule (significant gain, +6M/+8M/+10M plateau at ~20-21%, last-three-monotone clause failed by 0.82 pt) | [009](https://github.com/guidodinello/gamekit/blob/main/docs/research/009-longer-runs-and-resume.md) |

## Current best (Phase 5, as of 2026-09-29)

**20.72% [19.50%, 22.01%] win rate vs 3 `HeuristicAgent`s**, n=4000,
seat-rotated — [006](006-longer-run.md), `catan_bc_ft_long_10031616` (the highest of the five
pre-registered n=4000 fixed points; the +10M checkpoint's 19.90% [18.69%, 21.17%] is
statistically indistinguishable from it). Non-overlapping with [004](004-bc-warm-start.md)'s
16.2% [15.09%, 17.37%]. Still below the Phase-5-done gate (>25% with the CI excluding it);
the roadmap checkbox stays unchecked. Beats 3 `RandomAgent`s at **94.6% [93.86%, 95.26%]**.
The 20.72% is the best of five fixed points, so it carries a small selection bias; every
fixed point from +4M on clears 004's interval regardless.
