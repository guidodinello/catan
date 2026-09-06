# Phase 3 — Agents, heuristics, and the benchmark harness

## Context

Phase 1 (`9ce94c7`) delivered the rules engine; Phase 2 (`c51e4bd`) delivered
the Monte Carlo layer (`experiments/`) and committed real results under
`experiments/results/`. The README roadmap puts Phase 3 next: *"RandomAgent,
heuristic agents (settlement placement heuristics, build-order policies),
benchmark harness."*

Two things in the repo are explicitly waiting on this phase:

- **README decision 9** — "Benchmark sanity checks (Random vs Random ≈ 50/50,
  heuristic agents beat random by a sane margin) are a **Phase 3** concern,
  once agents exist."
- **`experiments/rollout.py`'s own docstring** — `Policy` is "a plain callable
  alias, not the Phase-3 `Agent` protocol… deliberately, so Phase 3 material
  doesn't creep into Phase 2 under another name." Phase 3 is where that
  placeholder gets replaced.

The phase produces the first agents that actually *play* Catan, and the first
numbers in this repo that are about agent skill rather than about the board.

### Scope boundaries (decided)

- **Random + heuristic agents only.** No MC-rollout agent (truco-py's
  `VonNeumannAgent` analog). It is not in the README roadmap for this phase,
  and it roughly doubles the work — Catan games run ~1,800 steps against
  truco's short hands, so an online-rollout agent needs its own performance
  budget. Deferred to Phase 5 territory, or a separate Phase 3.5.
- **No Phase 4 material.** The `Agent` protocol is written here in the shape
  Phase 4 will extract, but nothing is moved to a shared package and no
  cross-tree dependency is added — same reasoning the Phase 2 plan applied to
  `mcstats.py` vs. `mmo-utils`.
- **No Phase 5 material.** No gymnasium env, no state encoder, no reward
  function, no flat action space.
- **Agents do not propose domestic trades.** `ProposeTrade` is an open-ended
  sentinel (README decisions 5 and 10); a heuristic that negotiates is a
  research project of its own. Bank/port trades are in scope. Stated as a
  boundary, not hidden.

### The inversion to state out loud

Phase 2's decision 11 says *"win rate is not the primary metric"* — correct
there, because every number was conditional on uniform-random play.

**In Phase 3 win rate is the primary metric.** That is what a benchmark *is*:
the thing being measured is now the policy itself, so a policy-conditional
number is the point rather than the caveat. The `GameRecord` intermediate
metrics (cumulative resources, turns-to-5-VP) survive as *diagnostics* — they
explain *why* an agent wins, and they are what catches a heuristic that wins by
exploiting an engine quirk rather than by playing well. This is not a reversal
of decision 11; it is decision 11's precondition finally being met. It gets its
own README Decisions entry so it doesn't read as a contradiction.

---

## The load-bearing change: `rollout.py` goes from one policy to per-seat agents

`run_game` (`experiments/rollout.py:159`) today takes **one** `policy_factory`
applied to every seat and **one** `driver_rng`. Every Phase 3 headline number
requires *heterogeneous* seats (heuristic in one seat, random in the others),
so this signature is the centre of the phase.

**RNG ownership moves into the agent.** The driver derives no per-seat streams
and holds no policy RNG. Each agent owns whatever randomness it needs. This one
decision resolves both constraints at once:

- **Benchmark arms stay comparable** — each seat is constructed with its own
  independently-seeded `random.Random`, so swapping the agent in seat 0 cannot
  perturb seat 1's *driver* draws. Without this, a heuristic-vs-random arm and
  a random-vs-random arm are not on the same footing. Carrying Phase 2's
  scruple forward: this claims **driver-stream independence only**. The engine
  stream (`state.rng`) is shared and its consumption depends on play — dice,
  shuffles, and the `_steal_apply` draw (`game.py:517`) — so the design is
  **not** fully paired across arms, and the write-up must not read as if it
  were.
- **Phase 2's committed results stay replayable** — the old code called
  `make_random_policy(driver_rng)` *once* and used it for every actor, i.e. all
  seats already shared a single stream. Handing **the same `random.Random`
  instance** to all n `RandomAgent`s reproduces that draw sequence exactly. No
  compatibility flag, no mode switch — just a different way of constructing the
  agents. Every file under `experiments/results/` stays reproducible from its
  recorded `(engine_seed, driver_seed)` pair.

**Everything crossing the process pool must stay picklable.** `run_many` passes
its factory through `pool.submit`, and it works today only because
`make_random_policy`/`make_stratified_policy` are module-level functions. Agent
*instances*, closures, lambdas, and `partial` over locals all break
`workers > 1`. So the driver takes a **module-level factory**, never a list of
constructed agents — a hard constraint on the API shape, not a style choice.

**The factory signature — the detail the whole phase hangs off.** It must
receive the game's full identity so it can derive *both* the per-seat RNG
streams *and* the seat rotation deterministically:

```python
AgentFactory = Callable[[int, int, int], list[Agent]]
#   (num_players, engine_seed, driver_seed) -> one agent per PLAYER ID
```

- Per-seat streams: `random.Random(hash((driver_seed, seat)))` or equivalent —
  independent, reproducible, and swapping one seat's agent leaves the others'
  draws untouched.
- Seat rotation (benchmark only): the factory permutes which lineup slot lands
  in which seat as a deterministic function of `(engine_seed, driver_seed)`, so
  across an arm each agent occupies each seat an equal number of times. The
  permutation used is recoverable from `seat_order` + `agent_names` on the
  `GameRecord`, so the "each agent in each seat n/k times" bookkeeping is
  directly assertable.
- The Phase 2 default factory ignores rotation and hands the *same*
  `random.Random` instance to all n `StratifiedRandomAgent`s — that is what
  preserves bit-identity (below).

Also preserved from the current driver:

- **`acting_player(state)` is the agent index, not `state.current_player`**
  (`rollout.py:196` already does this right — they differ while a domestic
  trade is awaiting a response; `state.py:172`). The rewrite must not quietly
  regress to `current_player`.
- `build_random_trade_offer` correctly uses `state.current_player` — only the
  turn player proposes.

Two pre-existing layering facts this touches:

- `tests/test_engine.py:21` imports `make_stratified_policy` **and**
  `build_random_trade_offer` from `experiments.rollout`. Both move; the test's
  imports move with them.
- `MAX_TRADE_OFFER_SIDE` is defined twice — `engine/game.py:853` (authoritative,
  enforced in `_validate_trade_shape`) and duplicated as a literal at
  `rollout.py:29`. Values agree today. Since the sentinel resolver is being
  moved anyway, collapse it to the engine's definition. SSOT, one line.

---

## Files

### New: `agents/base.py` — the `Agent` protocol

```python
@runtime_checkable
class Agent(Protocol):
    name: str
    def choose_action(
        self, state: GameState, legal_actions: list[Action], player_idx: int
    ) -> Action: ...
    def reset(self) -> None: ...
```

**Method names and argument order are deliberately identical to truco-py's
`agents/base.py`.** Phase 4's extraction plan lists that file as extractable
because it is "already generic"; keeping the call surface identical means the
extraction is a rename-and-parameterise, not a redesign. (`name` is an
addition, needed for result stamping — a widening, not a divergence.)

**Sentinel handling — the agent resolves it, not the driver.** An agent that
returns the empty `ProposeTrade` sentinel returns something `apply_action`
cannot accept. Two designs keep the protocol generic (a third — growing the
protocol a bundle-construction method — would make it Catan-shaped and break
Phase 4's extraction, so it is rejected):

- `RandomAgent`/`StratifiedRandomAgent` resolve it themselves via
  `build_random_trade_offer` (the existing SSOT resolver).
- `HeuristicAgent` filters the sentinel out of its candidate set — it does not
  propose trades, per the scope boundary above.

The driver then **fails fast** on an unresolved sentinel with a clear error,
rather than letting a malformed action reach `apply_action`. Boundary
validation, per the fail-fast guideline.

### New: `agents/random_agent.py`

`RandomAgent(rng, name=...)` and `StratifiedRandomAgent(rng, name=...)` — the
class forms of `make_random_policy` / `make_stratified_policy`
(`rollout.py:131`, `:143`). Logic moves; behaviour is unchanged, which is what
makes the bit-identity regression test below meaningful. The old
`make_*_policy` factories and the `Policy` alias are **deleted**, not kept as
aliases — SSOT.

**These two agents must preserve the RNG call order and count exactly: no
filtering, no sorting, no reordering of `legal_actions`.**
`make_stratified_policy` draws `rng.choice(list(by_type.keys()))` — which
depends on dict insertion order, i.e. on the order `legal_actions` returned —
then `rng.choice(by_type[action_type])`, then whatever the sentinel resolver
draws. Bit-identity dies silently if `HeuristicAgent`'s
filter-the-sentinel-out idiom leaks in here, or if someone copies `cli.py:169`'s
`sorted(by_type, key=lambda t: t.__name__)`. This constraint belongs in the
module docstring, not just in this plan.

### New: `agents/heuristic.py` — `HeuristicAgent`

**Every rule below is derived from a committed Phase 2 result, not invented.**
That was the point of sequencing Phase 2 first; each rule cites its source.

- **Setup settlement** — score each legal vertex with
  `experiments.features.vertex_features()` (do *not* recompute pip sums) and
  pick the max of `pip_sum + W_DISTINCT * distinct_resources`.
  Source: `experiments/results/a2_seat0.md` — win rate by distinct-resource
  count is strictly monotonic with non-overlapping CIs at every step
  (4p: 7.0% → 13.4% → 29.1% → 41.0%), and `by_pip_sum_bucket` in the JSON shows
  the same pattern. `W_DISTINCT` is a named module constant calibrated so the
  ordering reproduces `a1_seed1_seat0_p4.md`'s top arms (vertex 30:
  pip_sum=11, 3 distinct — beating pure-pip-sum picks).
- **Setup road** — the adjacent edge pointing at the best-scoring unoccupied
  neighbouring vertex, same scoring function.
- **Build order** — a fixed preference: `PlaceCity` > `PlaceSettlement` >
  `BuyDevCard` > `PlaceRoad`, with `RollDice` taken whenever legal and
  `EndTurn` only when nothing else is preferred. Settlement/city sites ranked
  by the same `vertex_features` score.
- **Robber** — `MoveRobber` to the land hex maximising blocked opponent pips,
  never one of its own; `StealFrom` the player with the largest hand.
- **Discard** — shed from the largest resource pile first.
- **Bank/port trades** — taken only when the trade completes an affordable
  build this turn; otherwise skipped. (`TradeBank`/`TradePort` are already
  bank-availability-checked in the engine per README decision 14.)
- **Dev cards** — `PlayKnight` when the robber sits on one of its own producing
  hexes; `PlayVictoryPoint` always (it is unconditionally offered — see the
  known bias below); `PlayMonopoly`/`PlayYearOfPlenty` on the resource that
  most unblocks the preferred build.

Where a rule has a tunable, it is a named module constant, not a magic number.

**The heuristic reads the `legal_actions` list it was handed; it does not reach
into engine privates.** `engine/game.py` has tempting helpers
(`_settlement_legal_vertices`, `_road_legal_edges`, `_city_legal_vertices`,
`_longest_road_length`, `_player_port_rates`, `_steal_candidates`) but every
one of them is already reflected in the actions the driver passes in. Phase 2
set the precedent for promoting a private only when genuinely needed
(`_victory_points` → `victory_points`); the only promotion this phase actually
needs is **exporting the build costs** — `ROAD_COST`, `SETTLEMENT_COST`,
`CITY_COST`, `DEV_CARD_COST` (`engine/game.py:80-91`) are module constants that
`engine/__init__.py` does not re-export, and the bank/port-trade rule needs
them to know what "completes an affordable build" means.

### New: `agents/human.py` — `HumanAgent`

`cli.py:165`'s `choose_action(legal_actions, state, actor)` is already the
protocol's shape with the arguments transposed. `HumanAgent` wraps it (moving
the prompting helpers `_prompt_int`, `_prompt_propose_trade`, `_format_action`
along with it if that reads better; `cli.py` keeps only rendering + the game
loop). `HumanAgent` is the one agent allowed to return a real `ProposeTrade` —
`_prompt_propose_trade` already constructs a full bundle.

### Modified: `cli.py`

`run_hotseat_game(num_players, agents, seed)` takes a seat→agent list. `main()`
prompts for the number of human seats and fills the rest with `HeuristicAgent`,
so the agents are actually playable and inspectable rather than only visible as
aggregate numbers. All-human seats reproduce today's behaviour exactly.

### Modified: `experiments/rollout.py`

- `Policy`, `make_random_policy`, `make_stratified_policy`, and
  `_resolve_trade_sentinel` are removed (they move to `agents/`).
- `run_game(num_players, engine_seed, driver_seed, agent_factory, ...)` and
  `run_many(...)` take the module-level `AgentFactory` described above —
  `(num_players, engine_seed, driver_seed) -> one agent per player id`. It must
  stay a module-level function to remain picklable for `ProcessPoolExecutor`;
  the current `policy_factory` is under the same constraint, so nothing new.
- The Phase 2 default factory (`n` `StratifiedRandomAgent`s **sharing one
  `random.Random`**, no rotation) is what keeps `exp_placement.py` /
  `exp_tables.py` bit-identical. Their call sites are updated to the new
  keyword; nothing else in them changes.
- `GameRecord` gains `agent_names: tuple[str, ...]` (indexed by player id).
  Phase 2 stamped a single policy name on the result file; with heterogeneous
  seats, *who played where* must live on each record.
- `run_many`'s ordering guarantee is unchanged and its acceptance criterion is
  **strengthened**: output is bit-identical regardless of pool size **and**
  independent of which seats hold which agent (given per-seat RNGs).

### New: `experiments/benchmark.py` — the harness

- **A mode registry** — `dict[str, Callable[[int], list[Agent]]]` mapping a
  mode name to a seat lineup. Explicitly replaces truco-py's 11 hardcoded
  `--mode` branches, which `docs/shared-ml-package.md` flags as the one part of
  `scripts/benchmark.py` that is not reusable.
- **Seat rotation is mandatory.** README decision 12 established that the
  starting player is randomised and seat, not player id, is the unit of
  analysis. A heuristic-vs-random result is confounded unless the agent→seat
  assignment is rotated so each agent occupies each seat an equal number of
  times. This is the direct analog of truco-py's team assignment, and it is a
  correctness requirement, not a refinement.
- **Statistics reuse `experiments/mcstats.py` wholesale** — `wilson_interval`
  for every reported proportion (never the normal approximation),
  `two_proportion_test` for any A-vs-B claim, `benjamini_hochberg` when more
  than two agents are compared, `Accumulator` for the diagnostic means. The
  stats layer is already written and tested; do not write a second one.
- **Output** to `experiments/results/`, reusing `exp_placement.py`'s existing
  stamping helpers (`_git_commit`, `_write_result` at `exp_placement.py:42`,
  `:56`) so the header format stays identical, extended with the per-seat agent
  names.
- CLI: `uv run python -m experiments.benchmark --mode heuristic_vs_random
  --games 10000 --players 4 --workers 8`. Not run in CI.

### Modified: `engine/game.py` (conditional — measure first)

**README decision 15**, explicitly left to Phase 3: `_check_win`
(`engine/game.py:215`) evaluates only `state.current_player`, so a settlement
build that transfers Longest Road to a *third* player can leave them at ≥10 VP
undetected until their own turn. Observed in **0.22% of 10,000 4-player games**
under random play (`experiments/results/tables.md`).

Phase 3 is the first phase where **winner attribution is the headline number**,
so a mis-attributed winner is now a bug in the primary metric rather than a
footnote. Procedure:

1. Re-measure `non_winner_exceeded_ten` (already recorded on every
   `GameRecord`) under `HeuristicAgent`. The rate may well **rise** — heuristics
   build roads purposefully, so Longest Road changes hands more often.
2. If it is non-negligible, change `_check_win` to evaluate all players. This
   is a genuine rule fix (the rulebook's win condition is not turn-scoped), so
   it comes with a regression test in `tests/test_win.py` constructing the
   Longest-Road-transfer position directly.
3. Record the measured rate and the decision in the README Decisions list
   either way.

### Modified: `pyproject.toml`, `README.md`, `CHANGELOG.md`

`pyproject.toml`: add `agents` to `[tool.hatch.build.targets.wheel] packages`
(same reason Phase 2 added `experiments` — no `sys.path` workarounds) **and** to
`[tool.ruff.lint.isort] known-first-party`, which currently lists only
`engine, experiments`. Target is `py314` with zero runtime dependencies; the
repo already uses 3.14-only syntax (PEP 695 generics in `mcstats.py`, PEP 758
except groups in `exp_*.py`), so `agents/` may too.

`README.md`: tick Phase 3, update the Status section, and add Decisions entries
for the win-rate inversion, agent-owned RNG, the no-domestic-trade boundary,
and the `_check_win` outcome. (Note while editing: decision 13 cites
`docs/shared-ml-package.md` as if it were in this repo — it actually lives at
`~/projects/docs/shared-ml-package.md`. Fix the reference in passing.)

### New: `tests/test_agents.py`

Follows the existing conventions: flat in `tests/`, no `conftest.py` (there is
none in the repo), plain `def test_*() -> None` with a docstring saying why the
assertion matters, scenarios arranged by driving the real engine.

- `isinstance(RandomAgent(...), Agent)` — the `@runtime_checkable` protocol
  check for every agent.
- **Legality sweep**: over N random games, every action every agent returns is
  in the `legal_actions` list it was handed, and is directly appliable (no
  unresolved sentinel).
- **Determinism**: same seeds → identical `GameRecord`.
- **The load-bearing one — bit-identity against Phase 2.** `run_game` with n
  `StratifiedRandomAgent`s sharing one RNG must produce a `GameRecord` byte-for-byte
  equal (modulo the new `agent_names` field) to the pre-refactor driver on the
  same seed pair. Assert against a small set of golden records captured from
  `main` before the refactor lands. This is a far stronger check than "the tests
  still pass" — it proves the migration preserved semantics and that every
  committed Phase 2 result is still replayable.
- **Heuristic beats random on the diagnostics**, not just on win rate: at a
  small n, `HeuristicAgent`'s `mean_resources_through_turn_10` must exceed
  `RandomAgent`'s with separated CIs. A heuristic that wins without producing
  more is winning by an engine quirk.

---

## Methodology

**Step 0a, before a single line is deleted: capture the golden records.** The
bit-identity test compares against pre-refactor `GameRecord`s, so they must be
serialised from `main` **while `make_stratified_policy` still exists**. Doing
this first is not optional sequencing — once the baseline is gone the strongest
check in the phase is unrecoverable.

**Step 0b, before the harness is written: measure per-decision cost.** Phase 2
opened by measuring ~30 ms/game under random play and used it to set the sample
budget. `HeuristicAgent` scores every legal vertex on every setup decision and
ranks build options every turn — it is strictly more expensive per step, and
the factor is unknown. Measure games/second for `heuristic_vs_random` at n=50
*first*, then set the default `--games` from it. If the slowdown is large
enough to make 10,000-game arms impractical, that is a design input (cache
`vertex_features` per board — it is robber-independent by construction, see
`features.py:72`), not something to discover halfway through a run.

**Sample sizes** reuse Phase 2's table (`n ≈ 3.49/δ²` per arm at 80% power,
α=0.05). A heuristic-vs-random effect should be tens of points, not single
digits — so the binding constraint is the *sanity* arms, not the headline one.
Default 10,000 games per arm, subject to Step 0.

**The three acceptance gates** (README decision 9, made concrete):

1. **`RandomAgent` vs `RandomAgent` → `1/n` per seat**, within Wilson CI, for
   both 3p and 4p. Phase 2's A2 already measured seat 0 at 33.86% [32.9, 34.8]
   (3p) and 25.26% (4p), so this is a known-good target. Note rotation is a
   *no-op* here — every seat holds the same agent — so this gate tests seat
   bookkeeping in the driver, not the rotation logic. **Rotation is verified
   separately**, by asserting from the recorded `seat_order` + `agent_names`
   that each lineup slot occupied each seat `n/k` times.
2. **`HeuristicAgent` vs `RandomAgent` → heuristic clearly above `1/n`** with a
   non-overlapping CI. If this fails, the heuristic is not a heuristic.
3. **`HeuristicAgent` vs `HeuristicAgent` → per-seat win rates.** Unlike gate 1,
   a deviation here is **ambiguous and must not be treated as a red light.**
   Seat 0 picks first and sees the whole board (`a1_seed1_seat0_p4.md`'s own
   caveat); random play cannot express that advantage (decision 11), but
   purposeful play plausibly can. So given gate 1 passing and the rotation
   assertion above holding, a seat-0 edge under heuristic play is **evidence
   about Catan — a genuine, reportable first-pick advantage** — not a harness
   bug. It gets written up as a finding. Only a deviation that survives with
   gate 1 *failing*, or with the rotation counts wrong, indicates a bug.

**3p and 4p reported separately, never pooled** — Phase 2's rule, unchanged.

---

## Known biases to record, not paper over

- **`PlayVictoryPoint` is not gated by `has_played_dev_card_this_turn`**, so
  every agent reveals VP cards immediately and the public/hidden VP distinction
  collapses. Already documented for Phase 2; it now also means no agent can
  hide a winning VP card, which is a real strategic option in the actual game.
  Stated on the benchmark output.
- **Bank shortage** (`_produce`, `engine/game.py:433`) skips a resource when
  the bank cannot cover all claimants. Heuristic agents produce more, so they
  hit this *more often* than random agents did — the diagnostic column carries
  over to the benchmark output.
- **The heuristic is hand-tuned against Phase 2's own results**, which came
  from random play. A rule that helps against random opponents need not help
  against a good one. Gate 3 (heuristic vs. heuristic) is the check on this,
  and the limitation is stated in the result summary.

---

## Verification

1. `uv run ruff format --check . && uv run ruff check . && uv run mypy .`
   (CI runs only ruff + pytest — mypy is local/pre-commit only, so it must be
   run by hand before pushing.)
2. `uv run pytest` — all 12 existing modules stay green (notably
   `tests/test_experiments.py` and `tests/test_engine.py`, which both reach
   into `rollout.py`), plus `tests/test_agents.py`.
3. **Bit-identity regression passes** — the shared-RNG random lineup reproduces
   pre-refactor `GameRecord`s exactly. Primary correctness signal for the
   migration.
4. **Re-run one Phase 2 experiment at small n and diff against the committed
   JSON** — end-to-end proof that `experiments/results/` is still reproducible.
5. **Determinism and rotation checks**: `--workers 1` vs `--workers 8` at small
   n produces byte-identical benchmark JSON; swapping which seats hold which
   agent does not perturb the other seats' driver draws; and the recorded
   `seat_order` + `agent_names` show each lineup slot in each seat `n/k` times.
6. **The three acceptance gates above pass** at full sample size.
7. `_check_win` rate re-measured under heuristics; fixed with a regression test
   if warranted; outcome recorded in README Decisions either way.
8. `uv run python cli.py` — play a hot-seat game as one human against two
   `HeuristicAgent`s and watch it make sane openings. Source-verified is not
   verified; this is the one check that the heuristic *looks* like it is
   playing Catan.
9. Full benchmark runs executed for 3p and 4p, committed under
   `experiments/results/` with a hand-written `.md` summary, matching Phase 2's
   format.
