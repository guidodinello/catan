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

## Current best (Phase 5, as of 2026-09-21)

**12.075% [11.10%, 13.12%] win rate vs 3 `HeuristicAgent`s**, n=4000,
seat-rotated — [003](003-selfplay-v2-baseline-mix-0.5.md). Below the
Phase-5-done gate (>25% with the CI excluding it); the roadmap checkbox
stays unchecked until a future attempt clears it. Beats 3 `RandomAgent`s
at **90.025% [89.06%, 90.92%]** (improved from [001](001-ppo-vs-random.md)'s
81.75% by the additional self-play training, even without closing the gap
vs `HeuristicAgent`).
