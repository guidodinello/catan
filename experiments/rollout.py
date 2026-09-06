"""Rollout driver: play out full games against per-seat agents, recording a
``GameRecord`` per game for downstream Monte Carlo analysis.

Seats are heterogeneous as of Phase 3: ``run_game``/``run_many`` take an
``AgentFactory`` -- a module-level callable, not a list of constructed
agents, so results stay picklable across ``ProcessPoolExecutor`` workers --
that builds one ``agents.Agent`` per player id. RNG ownership lives in the
agents themselves; this driver derives no per-seat streams and holds no
policy RNG of its own.

Turn/production/VP bookkeeping is derived only from actions this module
itself chooses to apply and from public ``GameState``/``victory_points``
reads -- no private engine internals are touched.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

from agents import Agent, StratifiedRandomAgent
from engine.actions import Action, EndTurn, PlaceSettlement, ProposeTrade, RollDice
from engine.board import Cube, Resource
from engine.game import CatanGame, IllegalActionError, victory_points
from engine.state import WINNING_VICTORY_POINTS, Phase, acting_player

STEP_BUDGET_DEFAULT = 20_000

AgentFactory = Callable[[int, int, int], list[Agent]]
#   (num_players, engine_seed, driver_seed) -> one agent per PLAYER ID


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
    agent_names: tuple[str, ...] = ()  # indexed by player id
    treatment_seat: int | None = None
    treatment_vertex: int | None = None


def default_agent_factory(
    num_players: int, engine_seed: int, driver_seed: int
) -> list[Agent]:
    """Phase 2's default: ``num_players`` ``StratifiedRandomAgent``s sharing
    **one** ``random.Random`` instance, no seat rotation.

    Reproduces Phase 2's exact draw sequence -- the old code called
    ``make_stratified_policy(driver_rng)`` once and used it for every actor,
    i.e. all seats already shared a single stream. Handing the same
    ``random.Random`` to every agent here is what keeps
    ``experiments/results/`` replayable from its recorded
    ``(engine_seed, driver_seed)`` pair.
    """
    rng = random.Random(driver_seed)
    return [StratifiedRandomAgent(rng) for _ in range(num_players)]


def run_game(
    num_players: int,
    engine_seed: int,
    driver_seed: int,
    agent_factory: AgentFactory = default_agent_factory,
    scripted_setup: ScriptedSetup | None = None,
    step_budget: int = STEP_BUDGET_DEFAULT,
) -> GameRecord:
    """Play out one full game and return its ``GameRecord``.

    ``engine_seed`` drives ``GameState.rng`` (board, dev-deck order, dice,
    steals -- everything ``reset``/``apply_action`` touch internally).
    ``driver_seed`` is passed to ``agent_factory``, which derives whatever
    per-seat RNG streams its agents need -- this driver holds no policy RNG
    of its own, so agents own all of their own randomness.
    """
    game = CatanGame(num_players=num_players)
    state = game.reset(seed=engine_seed)
    agent_list = agent_factory(num_players, engine_seed, driver_seed)
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
            action = agent_list[actor].choose_action(state, legal, actor)
            if (
                isinstance(action, ProposeTrade)
                and not action.give
                and not action.receive
            ):
                raise IllegalActionError(
                    f"agent {agent_list[actor].name!r} (player {actor}) returned "
                    "the unresolved ProposeTrade sentinel -- agents must resolve "
                    "it themselves (build_random_trade_offer) or never offer it "
                    "(filter it out of legal_actions)"
                )

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
        agent_names=tuple(a.name for a in agent_list),
        treatment_seat=scripted_setup.seat if scripted_setup is not None else None,
        treatment_vertex=(
            scripted_setup.vertex_id if scripted_setup is not None else None
        ),
    )


def run_many(
    num_players: int,
    seed_pairs: Sequence[tuple[int, int]],
    agent_factory: AgentFactory = default_agent_factory,
    scripted_setup: ScriptedSetup | None = None,
    step_budget: int = STEP_BUDGET_DEFAULT,
    workers: int = 1,
) -> list[GameRecord]:
    """Run one game per ``(engine_seed, driver_seed)`` pair.

    Returned records are always ordered by ``seed_pairs``, never by
    completion order -- output is bit-identical regardless of ``workers``,
    and independent of which seats hold which agent (given per-seat RNGs).
    """
    if workers <= 1:
        return [
            run_game(num_players, e, d, agent_factory, scripted_setup, step_budget)
            for e, d in seed_pairs
        ]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                run_game, num_players, e, d, agent_factory, scripted_setup, step_budget
            ): (e, d)
            for e, d in seed_pairs
        }
        by_pair: dict[tuple[int, int], GameRecord] = {}
        for fut in as_completed(futures):
            by_pair[futures[fut]] = fut.result()
    return [by_pair[pair] for pair in seed_pairs]
