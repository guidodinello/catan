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
│   └── game.py        #   reset / legal_actions / apply_action / is_terminal / get_rewards
├── agents/            # RandomAgent, heuristic agents, (later) RLAgent
├── experiments/       # Monte Carlo experiments (starting placement, build order, …)
└── env/               # gymnasium wrapper for RL training
```

## Key Design Decisions & Open Questions

1. **Action space (the big one).** Truco encodes actions as a fixed global
   `Discrete(53)` + boolean masks. Catan's actions are *spatial and
   variable* (place settlement on any of ~54 vertices, road on any edge,
   trade offers, robber moves). A fixed enum does not scale. Options:
   - phase-dependent action encoding (legal target list per phase), or
   - a generated, masked discrete space sized for the board.
   This is the first design problem to solve in the engine.

2. **Complexity scope.** Catan is far bigger than Truco (647-line engine):
   random board generation, two placement phases, dice-roll resource
   production, robber blocking, port trading, dev cards (incl. victory
   point cards), longest road/largest army, hidden information (opponent
   hands, dev cards).

3. **Observability.** Opponent hands and dev cards are hidden. Decide early
   what the state exposed to agents contains (partial observation), and
   what the MC tables / baseline experiments can assume.

4. **Reference implementation.** `catanatron` (open-source Python Catan
   simulator with an env/agent architecture) is worth studying for
   interface design — but the plan is to build our own simulator, matching
   the hands-on pattern of the Truco projects.

5. **Benchmark sanity checks** (from `truco-py` experience): Random vs
   Random must be ~50/50; heuristic agents must beat random by a sane
   margin; match-level win rate is the goal metric, not per-move accuracy.

## Proposed Roadmap

- [ ] **Phase 1 — Engine**: board generation, game state, legal actions,
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

Empty skeleton — `main.py` hello-world, `pyproject.toml` (Python 3.14),
no commits. This README is the proposed spec; nothing implemented yet.
