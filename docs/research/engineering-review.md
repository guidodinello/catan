# Prior Art & Engineering Review — Catan Engine (Phase 1)

Curated prior-art reference for this repo's engine design, gathered while reviewing
`plans/build-phase-1-from-gleaming-fern.md`. Scope is **software engineering /
architecture only** — Catan rules correctness is reviewed separately.

Everything below is sourced. Where a claim comes from an auto-generated docs summary
rather than the source file, it is marked as such.

## Sources consulted

| Source | What it is | URL |
|---|---|---|
| `bcollazo/catanatron` | The reference open-source Python Catan simulator (README cites it). Engine + gym env + bots. | https://github.com/bcollazo/catanatron |
| Catanatron docs — Contributing | Maintainer's own perf/refactor notes | https://docs.catanatron.com/contributing |
| Catanatron docs — Gymnasium Interface | Env API + masking idiom | https://docs.catanatron.com/advanced/openapi |
| Catanatron API reference | `NodeRef`/`EdgeRef`, `Board`, coordinate system | https://catanatron.readthedocs.io/en/latest/catanatron.models.html |
| `henrycharlesworth/settlers_of_catan_RL` + write-up | Deep-RL Catan, custom Python simulator built specifically for forward search | https://github.com/henrycharlesworth/settlers_of_catan_RL / https://settlers-rl.github.io/ |
| JSettlers2 / OpenSettlers | Long-standing Java Catan engine, the classic academic Catan-AI substrate | https://github.com/nsp/OpenSettlers |
| Red Blob Games — Hexagonal Grids | Canonical reference for hex coordinate systems | https://www.redblobgames.com/grids/hexagons-v1/ |
| Red Blob Games — Parts of a hexagon grid | Corner/edge naming schemes | https://www.redblobgames.com/grids/parts/ |
| Gymnasium — Spaces / env API | `terminated` vs `truncated`, `Discrete` + mask | https://gymnasium.farama.org/api/spaces/ |
| Conditional Action Trees (Bamford & Ovalle, 2021) | Structured/factored action spaces for large combinatorial games | https://arxiv.org/pdf/2104.07294 |

Relevant catanatron paths (from the repo tree, `main` branch):
`catanatron/catanatron/models/map.py`, `models/board.py`, `models/coordinate_system.py`,
`models/tiles.py`, `models/actions.py`, `state.py`, `state_functions.py`,
`apply_action.py`, `gym/envs/catanatron_env.py`, `gym/envs/action_space.py`,
`tests/models/test_board.py`, `tests/test_game.py`, `tests/test_gym.py`.

---

## 1. Board geometry

### What prior art does

**Catanatron** uses a **cube coordinate system** (with `cube_to_axial()` /
`offset_to_cube()` helpers in `models/coordinate_system.py`), generating coordinates by
"expanding outward from a center tile on (0,0,0) with the given number of layers."
Its `BASE_MAP_TEMPLATE` in `models/map.py` explicitly distinguishes `LandTile` from
`Water` and `(Port, Direction)` entries — i.e. **catanatron has a water ring too**, and
ports live on it.

Node/edge canonicalisation is done in `get_nodes_and_edges()` by *sharing objects during
construction*: for each tile it looks up already-built neighbours, reuses their
node/edge objects, and only auto-increments a new integer id for slots still `None`.
Local references use `NodeRef` (NORTH, NORTHEAST, …) and `EdgeRef` (EAST, NORTHEAST, …)
enums; edges are stored as `(n1, n2)` tuples with `n1 < n2` — the same
canonical-sorted-pair idea the plan proposes for edges.

**Red Blob Games** ("Parts of a hexagon grid") uses a different, per-hex *ownership*
scheme: each corner is `q,r,N` / `q,r,S` and each edge `q,r,NE|NW|W`, with the article
stating plainly: *"The names and definitions here are arbitrary. This page isn't about
_the_ way to set up relationships, but _one_ way to set up relationships."*

### Verdict

**The plan's approach checks out.** A water ring is not an exotic trick — the reference
implementation ships one. The mechanism differs (catanatron canonicalises by object
sharing at construction time; the plan canonicalises by sorted hex-triple), but the ring
is what makes the plan's identity function *total*: without it a coastal vertex touches
1–2 land hexes and the triple isn't a triple. That is the actual requirement, and the
ring satisfies it with 18 extra dataless hexes.

The per-hex-owner scheme (redblobgames) is the standard alternative and avoids the ring,
but it costs a hand-written "which hex owns this corner" mapping and asymmetric
neighbour lookups. Sorted-triple + ring is the lower-code option. No change recommended.

### Residual risk

Integer node ids "assigned by sorting canonical keys" are *derived*, not declared. They
become a persistence contract the moment Phase 2 caches EVs by node id or Phase 5 encodes
a board tensor indexed by node id. A refactor of the canonical-key ordering silently
renumbers everything. Catanatron's ids are equally derived, but its ids are pinned by
`tests/models/test_board.py` fixtures. **Recommendation:** a golden test snapshotting the
54 canonical-key → id and 72 edge-key → id pairs.

---

## 2. Action representation

### What prior art does

**Catanatron keeps structured actions natively and flattens only at the RL boundary.**
`models/actions.py` defines an immutable namedtuple `Action(color, action_type, value)`
with a polymorphic `value`; `generate_playable_actions(state)` enumerates legal moves;
bots implement `decide(game, playable_actions)`. The flat space lives in a *separate
module in the gym subpackage*, `gym/envs/action_space.py`, as a sorted array with
`to_action_space(action) -> int` and `from_action_space(action_int) -> Action`. Masking
is applied by the caller: the docs' idiom is

```python
mask = np.zeros(env.action_space.n, dtype=np.float32)
mask[valid_actions] = 1
```

with SB3-contrib's `ActionMasker`.

### Verdict

**Deferring the flat encoding to Phase 5 is well-precedented and correct sequencing.**
The reference implementation draws exactly the same line, and the adapter is ~a hundred
lines in one file, not a cross-cutting rework. The plan's tagged-union frozen dataclasses
are, if anything, more type-safe than catanatron's polymorphic-`value` namedtuple.

### Residual risk — the trade action

A flat `Discrete(N)` mapping is only *total* if every action's value domain is finite and
small. Catanatron's flat space contains roll / discard / roads (per edge) / settlements
and cities (per node) / buy-dev / play-knight, YoP, road-building, monopoly / move-robber
(tile × victim) / maritime trades at 4:1, 3:1, 2:1 — and **no player-to-player trade at
all**. That omission is not an oversight; domestic trade is `give-multiset ×
receive-multiset × target-subset`, which does not enumerate.

The settlers-rl project hit the same wall and solved it with a **hierarchical multi-head
policy**: 13 base action types with per-head masking, and *recurrent resource heads* that
emit one resource at a time plus a "stop" token to build a trade proposal of variable
length. Conditional Action Trees (arXiv 2104.07294) is the general formulation of the
same idea.

**Implication for Phase 1:** the plan's `ProposeTrade(give, receive, targets)` is a good
*engine* action, but its domain must either be bounded now (e.g. cap offered/requested
totals, restrict targets to "all players") or Phase 5 must accept a factored head rather
than a flat `Discrete`. Whichever, record the decision in the README — it is cheap now
and expensive after the CLI and tests depend on the shape.

---

## 3. State design

### What prior art does

This is the area where catanatron's experience is most directly transferable, and it is
a **warning**. From the maintainer's own contributing guide, verbatim:

> "Continue refactoring the State to be more and more like a primitive `dict` or `array`.
> (Copies are much faster if State is just a native python object)."

and, in the same optimisation list:

> Move `.action_records` to the `Game` class so MCTS rollouts avoid copying unnecessary data

plus a note that Python `enum` objects are slow to hash and `RESOURCE` should become
integers. Catanatron's engine was restructured around `state.py` + `state_functions.py`
(free functions over a flat state) for exactly this reason. The settlers-rl author
likewise says they wrote their own Python simulator rather than reuse JSettlers
specifically to control "efficient state saving/restoration for forward search."

### Verdict

Mutable state + explicit `copy()` + `apply_action(state, action) -> state` is the
*right* family — it is what catanatron converged on, and it beats immutable/persistent
structures in Python, where structural sharing costs more in allocation and attribute
indirection than it saves. `slots=True` helps. **No architectural change recommended.**

### Residual risks (both concrete, both cheap to fix in Phase 1)

1. **Nothing tests `copy()` completeness.** A hand-written `copy()` over a `slots=True`
   dataclass is a landmine: add a field, forget one line, and two rollout branches
   silently alias the same list. This is the single most dangerous defect class for
   Phase 2, and it produces *plausible wrong numbers*, not crashes.
   **Fix:** a reflective test that walks `dataclasses.fields()` / `__slots__` and asserts
   (a) every mutable field in the copy is a distinct object, and (b) mutating each field
   of the copy leaves the original unchanged. This matches the repo's own
   `debugging-patterns.md` "try the type system / be exhaustive" guidance.
2. **Where does the RNG live?** The plan never says. If it is a module global or an
   engine attribute, rollout branches share a stream and `copy()` does not fork it —
   reproducibility and independence both break. If it lives on `GameState`, `copy()` must
   clone it (`random.Random` instances are copyable; `numpy.random.Generator` needs
   care). This must be decided in Phase 1 because dice, dev-deck shuffle and robber steal
   all draw from it.

---

## 4. Engine API surface

### What prior art does

- **Catanatron engine:** `generate_playable_actions(state)`, `game.execute(action)`,
  bot `decide(game, playable_actions) -> Action`.
- **Catanatron gym:** `gymnasium.make("catanatron/Catanatron-v0")`; `reset()` →
  `(observation, info)` with `info["valid_actions"]`; `step(action)` →
  `(observation, reward, terminated, truncated, info)`.
- **This repo's README** cites `truco-py`'s `Agent` protocol:
  `choose_action(state, legal_actions, player_idx) -> Action` + `reset()`.

### Verdict

`reset / legal_actions / apply_action / is_terminal / winner` lines up cleanly with all
three; deferring `get_rewards` is fine, since reward shaping is an RL-phase concern and
catanatron computes reward in the *env*, not the engine.

### Residual risks

1. **Actor ≠ current player.** With `AWAIT_TRADE_RESPONSE` in the phase machine, the
   player who must act is not the turn player. The README's target `Agent` protocol takes
   an explicit `player_idx`, so the engine needs a first-class "whose turn is it to act"
   accessor distinct from "whose turn is it". Retrofitting that distinction later touches
   every phase handler *and* every env wrapper.
2. **`terminated` vs `truncated`.** Gymnasium separates them because conflating them
   corrupts value bootstrapping. Keep the plan's step budget in the *test harness*, never
   inside `is_terminal()`, and define `winner() -> int | None` unambiguously (None must
   mean "no winner yet", with no second meaning).

---

## 5. Testing strategy

### What prior art does

Catanatron's suite splits by concern (`tests/models/test_board.py`,
`tests/models/test_actions.py`, `tests/test_game.py`, `tests/test_gym.py`) — the same
"one module per rule cluster" shape the plan proposes. Its contributing guide treats
**performance as a tested property**, recommending `pyinstrument`, `cProfile` + `snakeviz`,
and `pytest --benchmark-compare`. The project's headline claim is throughput —
"run thousands of Catan games per minute between different bots."

### Verdict

Geometry-invariants-first, one module per rule cluster, a full-random-game cross-cutting
invariant test, and a seeded determinism check are all sound and match prior art.

### Gaps

1. **No rollout-speed measurement.** Phase 2 *is* Monte Carlo and the reference bar is
   thousands of games/minute. Without a `pytest-benchmark` case now (games/sec for random
   self-play; µs for `copy()` and `apply_action`), a 10× regression is discovered only
   after every rule is baked in. Add a loose-threshold benchmark, not a strict one.
2. **Determinism test is too narrow.** `reset(seed=42)` twice → identical boards only
   covers board generation. The property that matters for MC/RL is *same seed + same
   action sequence → identical final state*, which additionally covers dice, dev-deck
   order and robber steals. Cheap to write, and it is the test that catches an
   RNG-placement mistake (§3).
3. **Copy-completeness test** — see §3.
4. **Property-based testing: targeted, not generic.** For a fixed 19/54/72 structure,
   exhaustive assertion is strictly *stronger* than generative sampling, so Hypothesis
   buys nothing on geometry. Where it earns its place is a Hypothesis
   `RuleBasedStateMachine` over the *phase machine*: generate legal action sequences and
   re-assert the cross-cutting invariants after each step, with automatic shrinking of any
   failing sequence. That is a materially better debugging experience than a random-game
   loop that fails at step 4,000.
5. **Golden id snapshot** — see §1.

---

## 6. Tooling / layout

Flat layout (`engine/`, `tests/`, `cli.py`) matches the README's proposed package layout
and `truco-py`; uv + pytest + ruff + mypy + pre-commit + CI is unremarkable and fine.
Adding `agents/`, `experiments/`, `env/` alongside `engine/` is exactly the README's own
diagram and will not become awkward on its own.

The one concrete future trigger is **Phase 4** (extracting a shared ML package consumed by
both `catan` and `truco-py`), which wants this project to be a proper installable
distribution with an unambiguous import root. That is a packaging change at Phase 4, not a
reason to deviate from the README now. Worth adding to the plan: `pytest-benchmark` (§5)
and `hypothesis` (§5.4) as dev dependencies.

---

## Summary table

| Area | Status | Action |
|---|---|---|
| Water ring + sorted-triple vertex ids | Sound, matches catanatron | none |
| Derived integer node/edge ids | Unpinned contract | golden snapshot test |
| Structured actions now, flat encoding at Phase 5 | Sound, matches catanatron | none |
| `ProposeTrade` domain | Unbounded → not flattenable | bound it, or record "factored head at Phase 5" |
| Mutable state + `copy()` | Right family; matches catanatron's own refactor direction | none |
| `copy()` completeness | Untested; silent aliasing risk | reflective copy test |
| RNG ownership | Unspecified | decide in Phase 1; must live on / be cloned with state |
| Engine API names | Aligned with catanatron / gymnasium / `truco-py` | none |
| Actor vs turn player | Implicit | explicit accessor |
| Step budget / `is_terminal` | Risk of conflation | budget stays in tests |
| Determinism test | Board-only | extend to seed + action sequence → final state |
| Rollout benchmark | Missing | `pytest-benchmark`, loose threshold |
| Property-based testing | Missing | Hypothesis `RuleBasedStateMachine` over phases |
| Flat layout | Fine | revisit at Phase 4 packaging |
