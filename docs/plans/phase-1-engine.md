# Phase 1 — Catan Engine (board, state, actions, rules, CLI)

_Revision 2 — folds in `docs/research/rules-review.md` (Catan Game Rules & Almanac,
5th ed.) and `docs/research/engineering-review.md` (catanatron / settlers-rl /
gymnasium prior art). Changes from rev 1 are marked **[R#]** (rules review) and
**[E#]** (engineering review)._

## Context

`~/projects/catan` is an empty skeleton: `main.py` hello-world, a `pyproject.toml`
(Python 3.14), a README that is a **proposed spec**, and **no commits**. The README
lays out a 6-phase roadmap toward Monte Carlo analysis, heuristic agents and RL,
modelled on the sibling `truco-py` pipeline.

This task builds **Phase 1 only** — the base-game rules engine and a hot-seat CLI.
Phases 2–6 (Monte Carlo, agents, shared ML package, RL, retrofit) are explicitly
deferred: no `agents/`, no `env/`, no `experiments/`, no reward function, no state
encoder, no flat action-index space.

The README's "Key Design Decisions & Open Questions" section is rewritten in place
as a **Decisions** section, so the open questions stop being open.

## Decisions (to be written into README.md)

1. **Action space** — structured, phase-dependent frozen dataclasses forming a tagged
   union, not a flat `IntEnum`. `legal_actions(state) -> list[Action]` enumerates only
   what is legal now. Resolves README open question #1 via the "phase-dependent
   encoding" option. A flat masked discrete space is Phase 5 and is **not** built now.
   This mirrors catanatron, which keeps structured actions in the engine and confines
   its flat space to `gym/envs/action_space.py`.
2. **Engine API** — `reset(seed)`, `legal_actions(state)`, `apply_action(state, action)`,
   `is_terminal(state)`, `winner(state) -> int | None`, plus `acting_player(state) -> int`
   **[E7]**. `get_rewards` is deferred to Phase 5 (user-confirmed).
   `winner()` has exactly one meaning: `None` = no winner yet. **[E8]** No step budget
   ever enters `is_terminal()` — truncation lives in the test harness only, mirroring
   gymnasium's `terminated`/`truncated` split.
3. **Acting player ≠ turn player [E7]** — during `AWAIT_TRADE_RESPONSE` the player who
   must act is not the turn player. `state.current_player` (whose turn) and
   `acting_player(state)` (who must move now) are distinct from day one, because the
   README's target `Agent.choose_action(state, legal_actions, player_idx)` needs it and
   retrofitting it later touches every phase handler.
4. **RNG ownership [E1]** — a `random.Random` instance lives as a **field on
   `GameState`**, never a module global or engine attribute, and `copy()` clones it
   (`copy.deepcopy` of a `random.Random` preserves and forks internal state). Dice,
   dev-deck shuffle, board generation and robber steals all draw from it. Without this,
   Phase 2 rollout branches share a stream — reproducibility and branch independence
   both break, and they break by producing *plausible wrong numbers*.
5. **Observability** — state holds full ground truth; there is **no** `player_view()` /
   redaction helper (Phase 2+ scaffolding). The CLI simply does not print other players'
   hands or unplayed dev cards. Resolves open question #3.
6. **Trading — bounded domain [E4]** (user-confirmed shape, domain bounded here):
   `ProposeTrade(give: ResourceBundle, receive: ResourceBundle, ...)` +
   `AcceptTrade` / `RejectTrade`; first accept executes; no counter-offers.
   **The domain is bounded now** rather than left open:
   - offers go to **all other players**, not an arbitrary target subset — this removes
     the `2^(n-1)` subset explosion entirely and matches how hot-seat play works;
   - each side is capped at **4 cards total**;
   - both sides must be non-empty and must not share a resource type **[R8]**.

   Even bounded, this is a give-multiset × receive-multiset product. Recorded in the
   Decisions section: **Phase 5 will need a factored/hierarchical action head for
   trade** (as in `settlers_of_catan_RL`'s recurrent resource heads / Conditional
   Action Trees), *not* a flat `Discrete`. Catanatron's flat space omits domestic trade
   altogether for exactly this reason.
   Also recorded: "first accept, no counter-offers" is a **documented scope
   simplification**, not a rules claim — the real game permits free negotiation.
7. **Board setup [R9]** — terrain and number tokens are drawn from the game's **fixed
   multisets**, not sampled freely: 19 terrain hexes (4 forest, 4 pasture, 4 field,
   3 hill, 3 mountain, 1 desert) and **18** number tokens (one 2, one 12, no 7,
   standard distribution for the rest). The **desert receives no token** and the robber
   **starts in the desert**. Red numbers (6 and 8) as a single class must not sit on
   **edge-adjacent** hexes — 6-6, 6-8 and 8-8 all forbidden — enforced by
   rejection-resample (the rulebook says "swap tokens"; same validity constraint,
   different method).
8. **Scope** — base game only, 3–4 players, no expansions.
9. **Reference implementation** — `catanatron` is studied (see the engineering review)
   but not vendored or depended on; we build our own. Resolves open question #4.
10. Open question #5 (benchmark sanity checks: random-vs-random ≈ 50/50 etc.) is a
    **Phase 3** concern and is restated as such rather than acted on.

## Tooling (mirrors `~/projects/truco`, the more polished sibling)

`roulette` has no tooling at all, so `truco` is the template:

- **uv** + `pyproject.toml`. Keep Python 3.14 (already pinned in `.python-version`).
- **Flat layout** (`engine/`, `tests/`, `cli.py`) — matches the README's proposed
  layout and `truco-py`; `truco`'s src-layout would contradict the README. Revisit at
  Phase 4 when a shared package needs an unambiguous import root.
- **pytest**, `testpaths = ["tests"]`, plain annotated functions, module docstring
  stating the rule under test (matches `truco/tests/test_hand.py`).
- **ruff** (`line-length = 88`) + **mypy** (truco's near-strict flag list), using the
  modern `[tool.ruff.lint]` table rather than truco's deprecated top-level `select`,
  and PEP 735 `[dependency-groups]`.
- **pre-commit** with truco's three repos (pre-commit-hooks, ruff, mirrors-mypy).
- **CI** copied from `truco/.github/workflows/ci.yml` (lint job + test job, uv-based).
- Dev deps additionally include **`pytest-benchmark`** and **`hypothesis`** **[E3, E9]**.

## Board geometry

Trap: a coastal vertex touches only 1–2 *land* hexes, so "vertex = sorted triple of
adjacent land hexes" produces non-canonical identities and silently corrupts adjacency
— which then breaks the distance rule, road connectivity and longest road.

**Chosen fix: generate a water ring.** Land = the 19 axial hexes with
`max(|q|, |r|, |q+r|) <= 2`; add the radius-3 ring (18 water hexes) purely as geometry.
Every vertex is then the intersection of exactly 3 hexes. Catanatron's
`BASE_MAP_TEMPLATE` ships a water ring too, so this is the referenced design, not a
novel one.

- **Coordinates**: axial `(q, r)`, cube derived where needed.
- **Vertex identity**: canonical sorted 3-tuple of the hex coords meeting at it.
  Enumerate the 6 corners of each of the 19 land hexes, dedupe → **54 vertices**.
- **Edge identity**: canonical sorted pair of vertex keys (catanatron uses the same
  sorted-pair idea); enumerate the 6 edges of each land hex, dedupe → **72 edges**.
- **Integer IDs**: assigned by sorting canonical keys — deterministic, no hand-typed
  adjacency tables.
- **Ports**: the only hand-typed table, 9 coastal edges given as `(hex, corner_index)`
  pairs resolved through the same canonicalization. 4 generic 3:1 + 5 specific 2:1
  (one per resource), types shuffled per game.

Two geometry tests, written **first**:

1. **Invariants** — 19 land hexes, 54 vertices, 72 edges; every vertex has 2–3 vertex
   neighbours; every edge exactly 2 endpoints; 9 ports, each on a coastal edge with
   2 distinct vertices.
2. **Golden id snapshot [E6]** — pin all 54 canonical-key → id and 72 edge-key → id
   pairs. These ids become a persistence contract the moment Phase 2 caches EVs by
   vertex id or Phase 5 encodes a board tensor; an incidental reordering would silently
   misalign caches with no failing test. Catanatron pins its ids the same way in
   `tests/models/test_board.py`.

Hypothesis is deliberately **not** used on geometry **[E9]** — for a fixed 19/54/72
structure, exhaustive assertion is strictly stronger than generative sampling.

## Files

```
engine/__init__.py     explicit `from .x import Y as Y` re-exports
engine/board.py        axial geometry, water ring, vertex/edge derivation, port table,
                       terrain/token multisets, generation + 6/8-adjacency resample
engine/state.py        Phase enum, PlayerState, GameState (mutable dataclass, rng
                       field, copy()), acting_player()
engine/actions.py      frozen-dataclass tagged union of all action types
engine/game.py         CatanGame: reset / legal_actions / apply_action / is_terminal /
                       winner, with _{phase}_legal / _{phase}_apply helpers
                       (mirrors truco-py's naming convention)
cli.py                 hot-seat human-vs-humans loop + ASCII board render
main.py                rewired to launch the CLI
tests/                 one module per rule cluster (below)
```

`GameState` is a mutable `@dataclass(slots=True)` with an explicit `copy()`;
`apply_action` mutates and returns it, mirroring `truco-py/engine/game.py`. This is the
family catanatron itself converged on ("copies are much faster if State is just a
native python object"); no persistent/immutable structures.

**Phase state machine**: `SETUP_SETTLEMENT`, `SETUP_ROAD`, `ROLL`, `DISCARD`,
`MOVE_ROBBER`, `STEAL`, `MAIN`, `AWAIT_TRADE_RESPONSE`, `GAME_OVER`.
`legal_actions` is a dispatch on `state.phase`.

**[R2] Pre-roll dev cards**: `ROLL` also offers `PlayDevCard`. A knight played pre-roll
routes `MOVE_ROBBER → STEAL → back to ROLL`, not to `MAIN`. A `robber_return_phase`
field on the state records where to return, since the same robber sub-machine is
entered from `ROLL` (pre-roll knight), from a 7 (`DISCARD`), and from `MAIN`
(post-roll knight).

## Rule enforcement — the testable units

Each bullet is one rule, one or more pytest cases. **[R#]** marks items added or
corrected by the rules review.

**Setup** — starting player by dice roll; round 1 clockwise, round 2
counterclockwise (snake 1,2,3,4,4,3,2,1); second settlement grants its adjacent hexes'
resources; setup road must touch the settlement just placed; distance rule applies in
setup too.

**Placement** — distance rule (no settlement adjacent to an existing settlement/city,
any owner); settlement must connect to your own road (outside setup); road must connect
to your own road/settlement/city and may **not** extend through an opponent's
settlement/city; only one road per path; city only upgrades your own settlement.
**[R5] Upgrading to a city returns the settlement piece to your supply** — the
5-settlement cap is concurrent, not lifetime, and `legal_actions` must offer settlement
builds again after an upgrade.

**Production** — settlement 1, city 2 per matching hex; robber-blocked hex pays
nothing; **bank shortage** evaluated **per resource type independently**: if the bank
cannot pay everyone owed that resource, nobody gets it — *unless exactly one* player is
owed, who then takes what remains (the excess is lost). Other resource types are
unaffected.

**Seven** — **[R6] the discard threshold counts resource cards only**; dev cards in
hand are never counted or discarded. Every player with >7 resource cards discards
`floor(n/2)`. Robber must move to a *different* hex. Steal one random card from a
player with a building on the new hex; choose among multiple victims; no steal if none
or all are empty.

**Dev cards** — **[R1] deck of exactly 25: 14 knight, 5 victory point, 2 road
building, 2 year of plenty, 2 monopoly.** Max one knight *or* one progress card per
turn. Cannot play a card bought this turn. **[R2] playable at any time during your
turn, including before the roll.** **[R3] VP cards are exempt from both restrictions**
— any number may be revealed on your turn, including cards bought that same turn, and
including to win immediately. **[R7] cannot buy when the deck is empty.** Dev cards are
never returned to the supply and **[R8] cannot be traded or given away**.
Progress effects: Road Building = 2 free roads under normal building rules;
Year of Plenty = any 2 resources from the supply, usable the same turn;
Monopoly = all other players hand over every card of one named resource.

**Awards** — **[R10] Longest Road is the longest simple path in the player's road
graph** (no edge reused, branches not summed), ≥5 segments; first to reach it takes the
card; a strictly **longer** road steals it; an opponent settlement on an unoccupied
intersection **breaks** a path. **[R4] The card can become unowned**: if the incumbent
loses the lead and two or more other players tie for the new longest, or if nobody has
≥5, the card is **set aside** and returns only when a single player is uniquely
longest. (The incumbent *does* keep it when merely tied while still holding the lead.)
Largest Army ≥3 knights, strictly greater to steal.

**Supply limits** — 19 cards per resource in the bank; per player 5 settlements,
4 cities, 15 roads (settlements recyclable per [R5]).

**Trading** — 4:1 bank always; 3:1 generic and 2:1 specific ports only with a
building on a port vertex (a 2:1 port grants no better-than-4:1 rate on other
resources). Domestic trade per Decision 6, with **[R8]** legality constraints:
both sides non-empty (no gifts), no like-for-like resource on both sides, no dev
cards, and only the **current turn's player** may be a party to a trade.

**Win** — 10+ VP, checked on the holder's own turn only. VP sources: settlement 1,
city 2, Longest Road 2, Largest Army 2, VP card 1. An award moving to another player on
*their* turn must not end the game for someone else.

## Testing plan

One module per rule cluster, plus these cross-cutting tests:

- **`copy()` completeness [E2]** — reflective test walking `dataclasses.fields()`:
  every mutable field in the copy is a distinct object, and mutating each field of the
  copy leaves the original unchanged. A hand-written `copy()` over a `slots=True`
  dataclass is a landmine — add a field, forget a line, and two rollout branches
  silently alias. This is the highest-value test in the suite for Phase 2.
- **Determinism [E5]** — not just "same seed → same board": **same seed + same action
  sequence → identical final state**, covering dice, dev-deck order and robber steals.
  This is the test that catches an RNG-placement mistake.
- **Random-game invariants** — drive full random games to completion with a step budget
  *in the harness* **[E8]**: `legal_actions` never empty before `GAME_OVER`, counts
  never negative, bank + hands conserve cards.
- **Hypothesis `RuleBasedStateMachine` [E9]** over the phase machine — generate legal
  action sequences, re-assert the invariants after each step, and get automatic
  shrinking. Materially better debugging than a random loop that fails at step 4,000.
- **Rollout benchmark [E3]** — `pytest-benchmark` with a **loose** threshold: games/sec
  for random self-play, µs for `copy()` and `apply_action`. Phase 2 *is* Monte Carlo
  (catanatron's bar is "thousands of games per minute"); without a baseline recorded
  now, a 10× regression only surfaces after every rule is baked in.

## CLI

Text-menu hot seat: print public state + the current player's private hand, number the
legal actions, read an index, apply. Board render is a fixed ASCII template with slots
substituted (resource letter, number token, robber marker, player initials on
vertices/edges) — a template string, deliberately not a layout engine.

## Git / workflow

1. Initial commit on `main` with the existing skeleton plus `docs/research/` — no PR.
2. Branch `feat/phase-1-engine` for all engine work.
3. **No remote is created yet.** When one is needed, ask which GitHub account should own
   it — personal `guidodinello` vs work `gdinlightit` — before creating anything.
   Personal repos use the direnv `.envrc` + `GH_TOKEN` switch from the global guidelines.
4. Run tests, commit, then **check in with the user before opening any PR or merging**.

## Verification

- `uv run pytest` — full suite green; geometry invariants and golden id snapshot first.
- `uv run pytest --benchmark-only` — records the rollout baseline.
- `uv run ruff check . && uv run ruff format --check . && uv run mypy .`
- `uv run python main.py` — play a real hot-seat game end to end: setup, a few turns, a
  pre-roll knight, a 7-roll with discard + robber + steal, a port trade, a domestic
  trade, a city upgrade followed by a new settlement build **[R5]**, and a win.
  Source-clean is not done; the CLI gets driven by hand once.
