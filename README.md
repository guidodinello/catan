# Catan — Strategy Analysis & ML Research Platform

Study the best strategies in Catan by running statistical models over a
game simulator: Monte Carlo probability estimation, heuristic agents, and
reinforcement learning, following the pattern established in the Truco and
Roulette projects.

## Idea & Motivation

The goal is to *understand the game mathematically*, not just to play it
well by feel. Concretely: build a faithful Catan simulator, then answer
questions like:

- Which starting settlement/road placements maximize expected win rate?
- What is the win-probability of a given position (board + resources + VPs)?
- Which build orders and trading policies dominate under Monte Carlo rollouts?
- Can a learned agent beat strong heuristic agents in full games?

The workflow is always the same three stages:

1. **Build the game simulator** — rules, state, legal actions.
2. **Statistical analysis** — Monte Carlo simulations and experiments to
   approximate probabilities and optimal thresholds.
3. **ML / agents** — heuristic agents, MC rollout agents, and RL training
   that interact with the simulator.

## Relationship to Sibling Projects

This project follows the pattern already proven in the Truco/Roulette family:

| Project | Location | State |
|---|---|---|
| `truco` | `~/projects/truco` | Hand-probability simulator only (flowers, piezas, muestra). No game-play loop. Superseded by `truco-py`. |
| `roulette` | `~/projects/roulette` | Notebook with `BetStrategy` abstract class + heuristic strategies + balance simulation. |
| `truco-py` | `~/Desktop/fac/2026/mmo/truco-py` | **The reference pipeline**: full game engine, Agent protocol, VonNeumannAgent (online MC rollouts + EV cache), gymnasium env, MaskablePPO training, self-play, match benchmarking. Best result: 85.3% vs ThresholdAgent at 2M steps. |
| `catan` | this repo | Empty skeleton — this document is the proposed spec. |

`truco-py` is the template this project is designed after.

## Proposed Architecture

### Principle: independent game packages + one shared ML package

**No monorepo.** Each game stays an independent package that owns its
simulator (engine). A shared ML package contains only the pieces that are
genuinely reusable across games, extracted from `truco-py`:

| Shared package (to extract) | From `truco-py` | Verdict |
|---|---|---|
| `Agent` protocol — `choose_action(state, legal_actions, player_idx) -> Action` + `reset()` | `agents/base.py` (22 lines) | Generic already; only needs `State`/`Action` type parameters |
| Benchmark runner — match loop, team assignment, progress logging, result table | `scripts/benchmark.py` | Reusable except the hardcoded mode registry |
| Gym env wrapper structure — single-agent env, auto-stepped opponents, legal-action masking, self-play opponent resampling | `training/env.py` | Best *template*; content is game-specific |
| MC rollout agent pattern — online rollouts + persistent EV cache | `agents/von_neumann_agent.py` | Pattern only; cache keys are game-specific |

Everything else in `truco-py` (`engine/`, `state_encoder.py`, `reward.py`,
`policy.py`, `mc_tables.py`) is Truco-specific and gets rebuilt per game.

### Catan package layout (proposed)

```
catan/
├── engine/            # Game rules: board, hexes, vertices, edges, resources,
│                      #   dev cards, robber, trading, build phases
│   ├── board.py       #   random board generation + placement validation
│   ├── state.py       #   immutable-ish game state (mirrors GameState pattern)
│   ├── actions.py     #   action encoding/decoding (see design decision below)
│   └── game.py        #   reset / legal_actions / apply_action / is_terminal / winner
├── agents/            # RandomAgent, heuristic agents, (later) RLAgent
├── experiments/       # Monte Carlo experiments (starting placement, build order, …)
└── env/               # gymnasium wrapper for RL training
```

## Decisions

Resolved while building Phase 1. Full sourcing (official rules citations,
catanatron/prior-art comparisons) lives in `docs/research/rules-review.md` and
`docs/research/engineering-review.md`.

1. **Action space.** Structured, phase-dependent frozen dataclasses forming a
   tagged union (`engine/actions.py`), not a flat `IntEnum`. `legal_actions(state)`
   enumerates only what's legal right now, dispatching on the current phase.
   Matches catanatron, which also keeps structured actions in the engine and
   confines a flat/masked `Discrete` space to its separate gym subpackage. A
   flat encoding is Phase 5 work and is not built now.
2. **Engine API.** `reset(seed)`, `legal_actions(state)`, `apply_action(state, action)`,
   `is_terminal(state)`, `winner(state) -> int | None`, plus `acting_player(state)`
   (the player who must act now — distinct from `state.current_player`, the turn
   player, since they differ while awaiting a domestic-trade response).
   `get_rewards` is deferred to Phase 5; it's an RL affordance, not an engine one.
3. **RNG ownership.** A `random.Random` lives as a field on `GameState`, cloned
   (not aliased) by `copy()`. Every die roll, shuffle, and robber-steal draw goes
   through it, so two branches copied from the same state never share a stream.
4. **Observability.** State holds full ground truth; there is no
   `player_view()` / redaction helper. The CLI simply never prints another
   player's hand. Redaction is Phase 2+ scaffolding.
5. **Trading.** `ProposeTrade` / `AcceptTrade` / `RejectTrade`: an offer goes to
   all other players at once, first accept executes, no counter-offers — a
   documented scope simplification, not a rules claim (the real game allows
   free negotiation). The domain is bounded (single resource type per side,
   capped at 4 cards) to keep it enumerable for `legal_actions`; even bounded,
   domestic trade is a give-multiset × receive-multiset product, so **Phase 5
   will need a factored/hierarchical action head for it**, not a flat
   `Discrete` — catanatron's own flat space omits domestic trade entirely for
   this reason.
6. **Board setup.** Terrain (4 forest/4 pasture/4 field/3 hill/3 mountain/1
   desert) and number tokens (18, fixed distribution, one 2, one 12, no 7) are
   drawn from the game's fixed multisets, not sampled freely. The desert gets
   no token and starts holding the robber. Red numbers (6, 8) as one class must
   never sit on edge-adjacent hexes — enforced by rejection-resampling the
   token assignment (equivalent to the rulebook's "swap tokens" fix).
7. **Scope.** Base game only, 3–4 players, no expansions.
8. **Reference implementation.** `catanatron` was studied for interface design
   (see the engineering review) but not vendored or depended on — this is our
   own simulator, matching the hands-on pattern of the Truco projects.
9. **Benchmark sanity checks** (Random vs Random ≈ 50/50, heuristic agents beat
   random by a sane margin) are a **Phase 3** concern, once agents exist — not
   acted on in Phase 1 beyond a rollout-speed baseline (`pytest-benchmark`) for
   `apply_action`/full-game throughput, recorded now so a later regression is
   visible.

## Proposed Roadmap

- [x] **Phase 1 — Engine**: board generation, game state, legal actions,
      full rule enforcement, CLI playable game (human vs humans).
- [ ] **Phase 2 — Monte Carlo analysis**: starting placement win-probability
      experiments, resource/VP probability tables.
- [ ] **Phase 3 — Agents**: RandomAgent, heuristic agents (settlement
      placement heuristics, build-order policies), benchmark harness.
- [ ] **Phase 4 — Shared ML package extraction**: generic `Agent` protocol
      + benchmark runner extracted from `truco-py`, adopted by Catan.
- [ ] **Phase 5 — RL**: gymnasium env, state encoder, MaskablePPO training
      vs heuristic opponents, self-play with anti-collapse controls.
- [ ] **Phase 6 — Retrofit** `truco-py` / `roulette` onto the shared package
      (only if the extraction holds up).

## Status

Phase 1 complete: full base-game rules engine (`engine/`), a hot-seat CLI
(`cli.py`), and a test suite (`tests/`) covering board geometry, state
copy/determinism, every rule cluster, and a cross-cutting random-game +
property-based invariant check. Phase 2+ (Monte Carlo, agents, RL) not started.
