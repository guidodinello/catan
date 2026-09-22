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
| [004](004-bc-warm-start.md) | BC warm start + PPO self-play fine-tune (PR #20) | validated (technique-level), gate not met | [002](https://github.com/guidodinello/gamekit/blob/main/docs/research/002-bc-warm-start.md) |

## Current best (Phase 5, as of 2026-09-22)

**16.2% [15.09%, 17.37%] win rate vs 3 `HeuristicAgent`s**, n=4000,
seat-rotated — [004](004-bc-warm-start.md). Below the Phase-5-done gate
(>25% with the CI excluding it); the roadmap checkbox stays unchecked until
a future attempt clears it. Beats 3 `RandomAgent`s at **93.5% [92.69%,
94.22%]** (improved from [003](003-selfplay-v2-baseline-mix-0.5.md)'s
90.025% [89.06%, 90.92%]). Both intervals are non-overlapping with 003's
12.075% [11.10%, 13.12%] vs `HeuristicAgent`s — a real improvement from
behaviour-cloning warm start, using fewer total steps than 003, even
without closing the gap to the gate.
