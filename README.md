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
`docs/research/engineering-review.md`. The implementation plan approved for
each phase before it was built is kept in `docs/plans/`.

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
   free negotiation). The give/receive bundle itself is a **full multi-resource
   bundle on each side** — e.g. give 2 lumber + 1 brick for 1 ore + 1 grain — with
   no restriction to a single resource type; that would be a real usability
   regression on a rule domestic trade uses constantly. Because that bundle
   space is too large to usefully pre-enumerate, `legal_actions` offers a
   single open-ended `ProposeTrade` affordance rather than every possible
   instance (the same reasoning as decision 10 below); the caller (the CLI,
   prompting a human interactively; a test driver, constructing one at random)
   builds the actual bundle and `apply_action` validates it in full — both
   sides non-empty, no shared resource type, capped at 4 cards per side purely
   so an interactive builder doesn't need to prompt for absurd amounts. Even
   so, it's a give-multiset × receive-multiset product, so **Phase 5 will need
   a factored/hierarchical action head for it**, not a flat `Discrete` —
   catanatron's own flat space omits domestic trade entirely for this reason.
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
10. **The engine models the real rules exactly — full stop.** No rule is ever
    bounded, restricted, or approximated for the sake of a future flat/discrete
    RL action space; that tradeoff belongs entirely to a separate adapter/encoding
    layer built in Phase 5, never baked into the engine itself. Concretely: where
    a legal action's parameter domain is small enough to enumerate exhaustively
    (`PlayYearOfPlenty` — pick 2 of 5 resources — or `PlayMonopoly` — pick 1 of 5),
    `legal_actions` lists every parametrized instance. Where it isn't
    (`ProposeTrade`'s multi-resource bundles), `legal_actions` offers a single
    open-ended affordance and the real object is constructed and fully validated
    by the caller in `apply_action` — never narrowed to make it enumerable. This
    is the same principle catanatron applies by confining its flat/masked
    `Discrete` space to a separate gym subpackage (decision 1), just carried one
    level deeper: it governs not only *how* actions are encoded, but *what a
    legal action is allowed to represent* in the first place.

Resolved while building Phase 2 (full sourcing in the Phase 2 plan; results
in `experiments/results/`):

11. **Win rate is not the primary ranking metric.** Every rollout number is
    conditional on the policy that produced it (`experiments/rollout.py`'s
    `stratified`/`random` policies — deliberately *not* the Phase-3 `Agent`
    protocol). Every placement arm also records policy-insensitive
    intermediate metrics (cumulative resources produced, turns to 5 VP)
    alongside a Wilson-CI win rate, so a null result can't be misread as "no
    effect" when it might just be "random play can't express the effect."
    In practice the harness resolved cleanly even under random play (see
    `experiments/results/a1_seed1_seat0_p4.md`).
12. **Seat, not player id, is the unit of analysis for placement
    experiments.** `winner()` returns a player index and the starting
    player is randomized per game, so player-id win rates are ≈ uniform by
    construction. The treatment player is pinned to a setup-order **seat**
    (`ScriptedSetup` in `experiments/rollout.py`); "best first settlement"
    is only well-posed per seat, and seat 0 (first picker, full board) is
    the default.
13. **A local, stdlib-only `experiments/mcstats.py`**, not a dependency on
    the sibling `mmo-utils` package — see `docs/shared-ml-package.md`'s own
    verdict that this layer (`mc_tables.py`'s analog) should be rebuilt per
    game, not extracted. Revisit at Phase 4.
14. **Found while building Phase 2, fixed in Phase 1 territory (engine
    correctness, not scope creep):** `_bank_trade_actions`/`_port_trade_
    actions` listed a `TradeBank`/`TradePort` as legal without checking the
    bank actually held the requested `receive` resource, so a long random
    game could crash on an "illegal" action that `legal_actions()` itself
    had offered. Fixed with a one-line bank-availability check in each;
    regression tests added in `tests/test_trading.py`.
15. **Engine finding, recorded not fixed:** `_check_win` only evaluates the
    turn player, so a settlement build that transfers Longest Road to a
    third player can push their VP to >=10 undetected until their own next
    turn (self-resolves within a round, not a hang). Observed in 0.22% of
    10,000 4-player games — see `experiments/results/tables.md`. Left for
    Phase 3 to decide whether it's worth closing.

## Proposed Roadmap

- [x] **Phase 1 — Engine**: board generation, game state, legal actions,
      full rule enforcement, CLI playable game (human vs humans).
- [x] **Phase 2 — Monte Carlo analysis**: starting placement win-probability
      experiments, resource/VP probability tables. See `experiments/` and
      `experiments/results/`.
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
property-based invariant check.

Phase 2 complete: a rollout driver and MC statistics utilities
(`experiments/rollout.py`, `experiments/mcstats.py`, `experiments/
features.py`), two starting-placement experiments (`experiments/
exp_placement.py`: per-vertex ranking on a fixed board, and feature-bucketed
win rate across random boards) and four resource/VP tables
(`experiments/exp_tables.py`), plus `tests/test_experiments.py` (including a
deterministic analytic-vs-engine cross-check). Full-scale results (800-
10,000 games per arm, 3- and 4-player) are committed under
`experiments/results/` with hand-written summaries. Phase 3+ (agents, shared
ML package, RL) not started.
