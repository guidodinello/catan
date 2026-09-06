"""Rollout driver: play out full games against a policy, recording a
``GameRecord`` per game for downstream Monte Carlo analysis.

``Policy`` is a **plain callable alias**, not a class and not the Phase-3
``Agent`` protocol (no ``reset()``, no registry). Keeping it a bare callable
is deliberate: it prevents Phase 3 material (agents/, RL) from creeping into
Phase 2 under another name. ``make_random_policy``/``make_stratified_policy``
are the only two policies Phase 2 needs.

Turn/production/VP bookkeeping is derived only from actions this module
itself chooses to apply and from public ``GameState``/``victory_points``
reads -- no private engine internals are touched.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

from engine.actions import Action, EndTurn, PlaceSettlement, ProposeTrade, RollDice
from engine.board import Cube, Resource
from engine.game import CatanGame, victory_points
from engine.state import WINNING_VICTORY_POINTS, GameState, Phase, acting_player

STEP_BUDGET_DEFAULT = 20_000
MAX_TRADE_OFFER_SIDE = 4

Policy = Callable[[GameState, list[Action]], Action]


@dataclass(frozen=True, slots=True)
class ScriptedSetup:
    """Force one seat's first-round setup settlement to a specific vertex;
    every other decision (that seat's setup road, every other seat's setup,
    and the whole rest of the game) is left to the policy.

    ``seat`` indexes into the game's setup order (``seat 0`` picks first),
    not a player id -- see the seat-vs-player-id note in the Phase 2 plan.
    """

    seat: int
    vertex_id: int


@dataclass(frozen=True, slots=True)
class ProductionEvent:
    """One player's resource gain from one non-7 dice roll."""

    turn: int
    player: int
    robber_hex: Cube
    gains: dict[Resource, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GameRecord:
    engine_seed: int
    driver_seed: int
    num_players: int
    seat_order: tuple[int, ...]  # seat_order[seat] == player id
    winner: int | None
    winning_seat: int | None
    turn_count: int
    step_count: int
    final_vp: tuple[int, ...]  # indexed by player id
    vp_by_turn: tuple[tuple[int, ...], ...]  # vp_by_turn[turn - 1][player id]
    production_events: tuple[ProductionEvent, ...]
    max_vp_observed: int
    max_vp_observed_by: int
    non_winner_exceeded_ten: bool
    first_settlement_vertex: tuple[int, ...]  # indexed by seat, first-round pick
    treatment_seat: int | None = None
    treatment_vertex: int | None = None


def _resolve_trade_sentinel(
    rng: random.Random, state: GameState, action: Action
) -> Action:
    """Resolve the ``ProposeTrade`` sentinel (empty give/receive -- see
    ``engine/game.py``'s ``_propose_trade_actions`` docstring) into a real
    multi-resource bundle, mirroring what an interactive builder (the CLI)
    or a test driver would construct.
    """
    if not (
        isinstance(action, ProposeTrade) and not action.give and not action.receive
    ):
        return action
    return build_random_trade_offer(rng, state)


def build_random_trade_offer(rng: random.Random, state: GameState) -> ProposeTrade:
    """Construct a real multi-resource bundle for the ``ProposeTrade``
    sentinel: a give side drawn from the proposer's actual hand (1..
    ``MAX_TRADE_OFFER_SIDE`` cards across however many resource types the
    random draw picks), and a receive side of any resource types not
    already on the give side.

    This is the single canonical sentinel resolver for the repo --
    ``tests/test_engine.py`` imports it rather than keeping its own copy.
    """
    player = state.players[state.current_player]
    available = [r for r in Resource if player.resources[r] > 0]
    rng.shuffle(available)
    give: dict[Resource, int] = {}
    remaining = min(MAX_TRADE_OFFER_SIDE, player.resource_card_count())
    for r in available:
        if remaining <= 0:
            break
        take = rng.randint(1, min(player.resources[r], remaining))
        give[r] = take
        remaining -= take

    receive_pool = [r for r in Resource if r not in give]
    rng.shuffle(receive_pool)
    receive: dict[Resource, int] = {}
    remaining = MAX_TRADE_OFFER_SIDE
    for r in receive_pool:
        if remaining <= 0:
            break
        take = rng.randint(1, remaining)
        receive[r] = take
        remaining -= take
        if rng.random() < 0.5:
            break
    return ProposeTrade(give=give, receive=receive)


def make_random_policy(rng: random.Random) -> Policy:
    """Flat-uniform policy: sample one action uniformly from all legal
    actions. TradeBank/TradePort/PlaceSettlement etc. dominate whenever they
    have many members for a state -- see ``make_stratified_policy`` for a
    policy that avoids this."""

    def policy(state: GameState, actions: list[Action]) -> Action:
        return _resolve_trade_sentinel(rng, state, rng.choice(actions))

    return policy


def make_stratified_policy(rng: random.Random) -> Policy:
    """Sample the action *type* uniformly first, then a member of that type.

    Prevents a numerous action type (a huge TradeBank/TradePort/placement
    list for one state) from dominating every step."""

    def policy(state: GameState, actions: list[Action]) -> Action:
        by_type: dict[type, list[Action]] = defaultdict(list)
        for a in actions:
            by_type[type(a)].append(a)
        action_type = rng.choice(list(by_type.keys()))
        return _resolve_trade_sentinel(rng, state, rng.choice(by_type[action_type]))

    return policy


def run_game(
    num_players: int,
    engine_seed: int,
    driver_seed: int,
    policy_factory: Callable[[random.Random], Policy] = make_stratified_policy,
    scripted_setup: ScriptedSetup | None = None,
    step_budget: int = STEP_BUDGET_DEFAULT,
) -> GameRecord:
    """Play out one full game and return its ``GameRecord``.

    ``engine_seed`` drives ``GameState.rng`` (board, dev-deck order, dice,
    steals -- everything ``reset``/``apply_action`` touch internally).
    ``driver_seed`` drives only the policy's own choices among legal
    actions, via a separate RNG -- so holding ``engine_seed`` fixed across
    two calls with different ``driver_seed``s isolates the board/dice/deck
    from the policy, and vice versa.
    """
    game = CatanGame(num_players=num_players)
    state = game.reset(seed=engine_seed)
    driver_rng = random.Random(driver_seed)
    policy = policy_factory(driver_rng)
    seat_order = tuple(state.setup_sequence[:num_players])

    def snapshot_vp() -> tuple[int, ...]:
        return tuple(victory_points(state, i) for i in range(num_players))

    turn_count = 0
    step_count = 0
    vp_by_turn: list[tuple[int, ...]] = []
    production_events: list[ProductionEvent] = []
    max_vp_observed = 0
    max_vp_observed_by = -1
    non_winner_exceeded_ten = False
    first_settlement_vertex: list[int | None] = [None] * num_players

    while not game.is_terminal(state) and step_count < step_budget:
        legal = game.legal_actions(state)
        actor = acting_player(state)

        if (
            scripted_setup is not None
            and state.phase == Phase.SETUP_SETTLEMENT
            and state.setup_position < num_players
            and actor == seat_order[scripted_setup.seat]
        ):
            action: Action = PlaceSettlement(scripted_setup.vertex_id)
            if action not in legal:
                raise ValueError(
                    f"scripted vertex {scripted_setup.vertex_id} is illegal for "
                    f"seat {scripted_setup.seat} on engine_seed={engine_seed}"
                )
        else:
            action = policy(state, legal)

        pre_phase = state.phase
        pre_setup_position = state.setup_position
        pre_resources: list[dict[Resource, int]] | None = None
        if isinstance(action, RollDice):
            pre_resources = [dict(p.resources) for p in state.players]

        game.apply_action(state, action)
        step_count += 1

        if (
            isinstance(action, PlaceSettlement)
            and pre_phase == Phase.SETUP_SETTLEMENT
            and pre_setup_position < num_players
        ):
            seat = seat_order.index(actor)
            first_settlement_vertex[seat] = action.vertex_id

        if pre_resources is not None and state.dice_roll is not None:
            total = sum(state.dice_roll)
            if total != 7:
                for i, (before, player) in enumerate(
                    zip(pre_resources, state.players, strict=True)
                ):
                    gains = {
                        r: player.resources[r] - before[r]
                        for r in Resource
                        if player.resources[r] - before[r] > 0
                    }
                    if gains:
                        production_events.append(
                            ProductionEvent(
                                turn=max(turn_count, 1),
                                player=i,
                                robber_hex=state.board.robber_hex,
                                gains=gains,
                            )
                        )

        setup_just_ended = pre_phase == Phase.SETUP_ROAD and state.phase == Phase.ROLL
        if setup_just_ended:
            turn_count = 1
            vp_by_turn.append(snapshot_vp())
        elif isinstance(action, EndTurn):
            turn_count += 1
            vp_by_turn.append(snapshot_vp())

        current_vps = snapshot_vp()
        for i, vp in enumerate(current_vps):
            if vp > max_vp_observed:
                max_vp_observed, max_vp_observed_by = vp, i
            if vp >= WINNING_VICTORY_POINTS and i != state.current_player:
                non_winner_exceeded_ten = True

    winner = game.winner(state)
    winning_seat = seat_order.index(winner) if winner is not None else None

    return GameRecord(
        engine_seed=engine_seed,
        driver_seed=driver_seed,
        num_players=num_players,
        seat_order=seat_order,
        winner=winner,
        winning_seat=winning_seat,
        turn_count=turn_count,
        step_count=step_count,
        final_vp=snapshot_vp(),
        vp_by_turn=tuple(vp_by_turn),
        production_events=tuple(production_events),
        max_vp_observed=max_vp_observed,
        max_vp_observed_by=max_vp_observed_by,
        non_winner_exceeded_ten=non_winner_exceeded_ten,
        first_settlement_vertex=tuple(
            v if v is not None else -1 for v in first_settlement_vertex
        ),
        treatment_seat=scripted_setup.seat if scripted_setup is not None else None,
        treatment_vertex=(
            scripted_setup.vertex_id if scripted_setup is not None else None
        ),
    )


def run_many(
    num_players: int,
    seed_pairs: Sequence[tuple[int, int]],
    policy_factory: Callable[[random.Random], Policy] = make_stratified_policy,
    scripted_setup: ScriptedSetup | None = None,
    step_budget: int = STEP_BUDGET_DEFAULT,
    workers: int = 1,
) -> list[GameRecord]:
    """Run one game per ``(engine_seed, driver_seed)`` pair.

    Returned records are always ordered by ``seed_pairs``, never by
    completion order -- output is bit-identical regardless of ``workers``.
    """
    if workers <= 1:
        return [
            run_game(num_players, e, d, policy_factory, scripted_setup, step_budget)
            for e, d in seed_pairs
        ]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                run_game, num_players, e, d, policy_factory, scripted_setup, step_budget
            ): (e, d)
            for e, d in seed_pairs
        }
        by_pair: dict[tuple[int, int], GameRecord] = {}
        for fut in as_completed(futures):
            by_pair[futures[fut]] = fut.result()
    return [by_pair[pair] for pair in seed_pairs]
