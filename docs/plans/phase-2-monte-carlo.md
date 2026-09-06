# Phase 2 — Monte Carlo analysis on the Catan engine

## Context

Phase 1 (merged in `9ce94c7`) delivered a complete base-game rules engine
(`engine/`), a hot-seat CLI (`cli.py`), and 11 test modules. The README's
roadmap puts Phase 2 next: *"Monte Carlo analysis: starting placement
win-probability experiments, resource/VP probability tables."*

This phase consumes the engine only through its public API
(`reset` / `legal_actions` / `apply_action` / `is_terminal` / `winner` /
`acting_player`) via random rollouts. It answers two questions:

1. **Which starting placements are good, and by how much?**
2. **What do the resource-income and victory-point distributions actually
   look like over a game?**

### Scope boundaries (decided, not up for re-litigation)

- Lives in **this repo**, in `experiments/`, per the README's proposed layout.
  Not a separate repo, not the not-yet-existing shared ML package —
  `docs/shared-ml-package.md` explicitly classifies `mc_tables.py` (truco-py's
  analog of this work) as *"rebuild per game, do NOT extract"*.
- **No Phase 3+ material**: no `Agent` protocol, no `agents/`, no gymnasium
  env, no reward function, no state encoder.
- **No engine performance work.** Measured below: it's fast enough.

### What was measured before planning

Run on the current `main`, 3-player games, both the type-stratified sampler
from `tests/test_engine.py` and a flat-uniform sampler:

| Sampler | Players | Games finished | Mean steps | Wall clock/game |
|---|---|---|---|---|
| type-stratified | 3 | 20/20 | 1,528 | 0.03 s |
| flat uniform | 3 | 20/20 | 1,809 | 0.02 s |
| flat uniform | 4 | 20/20 | 1,999 | 0.03 s |

**Uniform-random play terminates reliably.** No step budget is needed for
correctness (one is still kept as a safety valve). ~30 ms/game single-core
means 35,000 games ≈ 18 minutes on one core, minutes across a pool. This
de-risks the entire phase and sets the sample-size budget below.

---

## The central design decision: win rate is not the primary metric

Every rollout number is conditional on the policy that produced it. Under
uniform-random play, players do not convert good production into wins — so
the win-rate difference between an excellent and a mediocre starting vertex
is **attenuated toward zero**. It is entirely possible to burn 35,000 games
per arm and land inside the noise, with no way to distinguish *"this vertex
isn't better"* from *"random play cannot express the advantage."*

So every placement arm records **policy-insensitive intermediate metrics**
on the same rollouts:

- cumulative resources produced by the treatment player through turns 10/20/30
- turns-to-first-city, turns-to-5th-VP
- realized production vs. the analytic pip expectation for that vertex

These have far better signal-to-noise than a 1/n win indicator and carry
into Phase 3 unchanged. **Win rate is reported as the headline with an
honest Wilson CI; the intermediate metrics are what actually rank
placements.** Every committed result file is stamped with the policy name so
no number is ever read as policy-free.

---

## Files

### New: `experiments/mcstats.py` — local generic MC utilities

Stdlib-only (`math`, `statistics`). Mirrors the *names* used in
`~/Desktop/fac/2026/mmo/mmo-utils` so a later swap is mechanical:

- `ConfidenceInterval(lower, upper)` with a `.width` property
- `MCResult(mean, variance, n)` with `.confidence_interval(alpha)`
- `wilson_interval(successes, n, alpha)` — **used for all proportions**;
  the normal approximation is not used anywhere
- `sample_size_clt(eps, delta)`, `sample_size_hoeffding(eps, delta)`
- `two_proportion_sample_size(p, delta, power, alpha)`
- `two_proportion_test(k1, n1, k2, n2)` → z, p-value
- `benjamini_hochberg(p_values, q)` for multi-arm comparisons
- `Accumulator` — Welford streaming mean/variance, generic over the sample
  type (not `float`-only; that is one of the gaps `mmo-utils/DESIGN.md`
  records against itself)

**Why local, not `mmo-utils`:** that package has documented gaps (two
parallel sampling models, `float`-only `Accumulator`, no CI for custom
accumulators, no variance reduction) and lives on a different filesystem
tree. A cross-tree dependency for Phase 2 is premature. **Revisit in
Phase 4** — when the shared-package extraction happens, this module is the
natural thing to either delete in favour of a fixed `mmo-utils`, or to
contribute upstream. Noted here so it isn't forgotten.

### New: `experiments/rollout.py` — the rollout driver

- `Policy = Callable[[GameState, list[Action]], Action]` — a **plain
  callable alias**. Not a class, no `reset()`, no registry. This is
  deliberately *not* the Phase-3 `Agent` protocol and will be replaced by
  it; keeping it a bare callable prevents Phase 3 material from leaking in
  under another name.
- `random_policy` (flat uniform) and `stratified_policy` (sample action
  *type* first, then a member — the sampler from `tests/test_engine.py`).
- `build_random_trade_offer(rng, state)` — the canonical resolver for the
  `ProposeTrade` sentinel (`engine/game.py:908` returns an empty
  give/receive affordance that is never directly appliable).
  **`tests/test_engine.py` is refactored to import this instead of keeping
  its own copy** — SSOT for the sentinel-resolution rule.
- `GameRecord` (frozen dataclass): winner, winning seat, turn count, step
  count, per-player final VP, per-player VP-by-turn trace, per-player
  cumulative resources produced by resource and turn, bank-shortage event
  count, max VP observed by any player, both seeds.
- `run_game(num_players, engine_seed, driver_seed, policies, scripted_setup)
  -> GameRecord`, and `run_many(...)` parallelising with
  `concurrent.futures.ProcessPoolExecutor`.

### New: `experiments/features.py` — analytic tables and vertex features

No simulation. Pure combinatorics over `engine.board.GEOMETRY`:

- `dice_probability(total)` — exact 2d6 distribution; `pips(token)`
- `vertex_production(board, vertex_id) -> dict[Resource, float]` — expected
  cards per roll, robber-aware
- `vertex_features(board, vertex_id)` — pip sum, distinct resource count,
  port type at the vertex, desert adjacency, hex count (3 vs. 2 for coastal)

This is the analytic ground truth the empirical income tables are validated
against.

### New: `experiments/exp_placement.py` — the two placement experiments

**A1 — per-vertex ranking on a fixed board.** Fix an engine seed (hence
board, dev-deck order, and starting player) and a **setup seat**. For each
candidate first-settlement vertex, run N games where the treatment seat is
scripted to take that vertex (and a scripted-legal adjacent road) and every
other decision by every player is the policy. Report per-vertex win rate
with Wilson CIs plus the intermediate metrics, ranked.

**A2 — feature-bucket win rates across random boards.** Sample random
boards; the treatment seat takes a uniformly-random legal first settlement;
record its `vertex_features`, play out, record the outcome. Report win rate
and intermediate metrics **bucketed by pip sum and by distinct-resource
count**, each with a Wilson CI.

Bucketed tables, not logistic regression — no numpy dependency, more
interpretable, and honest about the fact that the features are confounded
with each other. A regression is an optional Phase-3 follow-up.

### New: `experiments/exp_tables.py` — resource/VP probability tables

Concretely, four tables:

1. **Analytic dice/production table** (exact, from `features.py`): P(roll=k),
   pips per token, expected cards/roll for every vertex on a given board.
2. **VP trajectory**: distribution of VP by turn number, and
   **P(win | VP = v at turn t)** as a matrix with Wilson CIs, cells below a
   minimum count suppressed rather than reported as noise.
3. **Game shape**: game-length distribution (turns and steps), and
   VP composition at the win (settlements / cities / longest road /
   largest army / VP cards).
4. **Empirical income**: cards received per turn per resource, with the
   bank-shortage diagnostic alongside.

### New: `experiments/results/` — committed outputs

JSON + CSV. Every file stamped with: git commit, engine seed base, driver
seed base, `n_games`, `n_players`, policy name, and a wall-clock timestamp.
Each is accompanied by a short `.md` summary written by hand.

### New: `tests/test_experiments.py`

- Fast smoke: ~20 games end-to-end, asserting `GameRecord` invariants
  (VP trace monotonic per player except for longest-road/largest-army
  transfers; winner's final VP ≥ 10).
- `mcstats` unit tests: Wilson interval against known values, sample-size
  formulas, Welford against `statistics.variance`.
- **The load-bearing one — analytic vs. empirical cross-check:** empirical
  per-vertex resource income over many rolls must match
  `vertex_production()` within a CI, modulo robber blocking and bank
  shortage. If this fails, the harness is wrong. This is the strongest
  correctness check available on the whole experiment layer, so it is a
  test, not a remark.

Full experiment runs are **not** in CI — they are manual
(`uv run python -m experiments.exp_placement --games 20000`).

### Modified: `engine/game.py` + `engine/__init__.py`

Make `_victory_points` public as `victory_points(state, player_idx)` and
export it. Read-only query, zero rule change. Exactly one accessor, matching
the engine's own win condition (it counts `revealed_vp_cards`, not
unrevealed cards sitting in `dev_hand`); any "hidden VP" notion is computed
in `experiments/`, not added to the engine.

### Modified: `pyproject.toml`

Add `experiments` to `[tool.hatch.build.targets.wheel] packages`. Verified:
the repo root is already on `sys.path` under uv's editable install (`tests`
and `docs` import as namespace packages even from `/tmp`), so this is not
required for local imports — but it is required for a correctly-built wheel,
and the guidelines forbid working around packaging with `sys.path`
manipulation.

### Modified: `README.md`, `CHANGELOG.md`

Tick Phase 2, add a Decisions entry for the design choices below.

---

## Statistical methodology (stated, not gestured at)

**Sample sizes.** Two-proportion comparison at p ≈ 1/3, 80% power,
α = 0.05: `n ≈ 3.49/δ²` per arm.

| Effect δ to detect | n per arm | Wall clock, 1 core |
|---|---|---|
| 5 pp | ~1,400 | ~45 s |
| 2 pp | ~8,700 | ~4.5 min |
| 1 pp | ~35,000 | ~18 min |

A single-arm Wilson half-width of ±0.01 at p ≈ 1/3 needs ~8,500 games.
**Defaults: 10,000 games per arm**, configurable via CLI flag.

**Multiple comparisons.** A1 ranks up to 54 vertices. Report CIs and a rank
ordering by default; apply Benjamini-Hochberg whenever a *pairwise
significance claim* is made.

**Variance reduction — what is and isn't claimed.** `state.rng` (engine) and
the driver RNG (policy) are already separate streams. Holding the engine
seed fixed across arms gives an identical board, dev-deck order, and
starting player — the **dominant variance component, eliminated for free**.
Dice stay aligned across arms only until the first divergence that draws
from `state.rng` (`_steal_apply` at `engine/game.py:521` draws the stolen
card). So: **board-variance elimination is claimed; a fully paired design is
not.**

**Seat, not player id.** `winner()` returns a player *index*, and
`_resolve_starting_player` (`engine/game.py:108`) randomizes who starts, so
player-id win rates are ≈ uniform by construction and meaningless. The
treatment player is pinned to a **setup-order seat**
(`state.setup_sequence`). This also means "the best first settlement" is
only well-posed *per seat* — the first picker sees the whole board, seat 3
does not. **Seat is an explicit experiment parameter**, defaulting to seat 0.

**Player count.** `n_players` is not incidental — setup depth and vertex
scarcity differ materially. **Default 4**; A1/A2 are run and reported for 3
and 4 separately, never pooled.

**Determinism under parallelism.** Every game is identified by
`(engine_seed, driver_seed)` derived from a single base seed, both recorded
in the `GameRecord`. Results aggregate by **sorted seed, never completion
order**; no RNG is ever shared across workers. **Acceptance criterion:
output is bit-identical regardless of pool size**, and any single game is
replayable from its recorded seed pair.

---

## Known biases and engine findings to record, not paper over

- **Bank shortage.** `_produce` (`engine/game.py:433`) skips a resource
  entirely when the bank cannot cover 2+ claimants. Correct rule, but it
  biases late-game income tables downward. Recorded as a diagnostic column
  so the bias is visible.
- **VP-card reveal is a policy artifact.** `PlayVictoryPoint` is offered
  whenever a VP card is held and is not gated by
  `has_played_dev_card_this_turn`, so a random policy reveals VP cards
  almost immediately. Under this policy the public/total VP distinction
  collapses. That is a property of the policy, not of Catan — stated on the
  VP tables.
- **VP can transiently exceed 10 without a win being detected.**
  `_check_win` (`engine/game.py:215`) only evaluates
  `state.current_player`. Building a settlement can break an opponent's road
  and transfer Longest Road to a *third* player, who may cross 10 undetected
  until their own turn begins. It self-resolves within a round, so it is not
  a hang — but the harness records max-VP-observed and **flags any
  non-winner above 10 as an engine finding to report**, rather than
  silently smoothing it into the tables.

---

## Verification

1. `uv run ruff format --check . && uv run ruff check . && uv run mypy .`
2. `uv run pytest` — the existing 11 modules stay green (notably
   `tests/test_engine.py` after its refactor to import the shared sentinel
   resolver), plus the new `tests/test_experiments.py`.
3. **Analytic cross-check passes** — empirical per-vertex income matches
   `vertex_production()` within CI. This is the primary correctness signal.
4. **Determinism check:** run `exp_placement` with `--workers 1` and
   `--workers 8` at a small n; assert the JSON outputs are byte-identical.
5. **Sanity check on the headline number:** across A2, the pooled win rate
   over all seats must be 1/n_players within its Wilson CI. If it isn't,
   the harness has a seat-assignment bug.
6. **Face-validity check:** the top pip-sum bucket in A2 must beat the
   bottom bucket on *cumulative resources produced* with a clearly separated
   CI. If the intermediate metrics can't separate an obvious effect, the
   experiment cannot support any claim about win rate either.
7. Full runs executed and their outputs committed under
   `experiments/results/` with hand-written `.md` summaries.
