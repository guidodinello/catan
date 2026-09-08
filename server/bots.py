"""Seat-kind -> ``Agent`` factory, and the auto-step loop for bot seats.

Reuses ``agents.Agent`` instances directly (``RandomAgent``/
``StratifiedRandomAgent``/``HeuristicAgent``) via a small per-seat factory
mirroring ``experiments/benchmark.py``'s ``benchmark_agent_factory`` pattern:
each bot seat owns its own independently-seeded ``random.Random``
(``random.Random(hash((driver_seed, seat)))``), so no two bot seats share a
stream (README decision 17). Unlike that benchmark harness, there is no
seat-rotation/lineup-registry logic here -- rotation exists to keep *many*
games statistically comparable across arms; a single live web game has no
arm to be fair against, so ``seat_kinds`` is just per-seat ground truth.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from agents import Agent, HeuristicAgent, RandomAgent, StratifiedRandomAgent
from engine.actions import AcceptTrade, Action, CounterTrade, RejectTrade, RollDice
from engine.board import Resource
from engine.game import CatanGame
from engine.state import GameState, TradeOffer, acting_player

if TYPE_CHECKING:
    from .sessions import GameSession

BotKind = Literal["random", "stratified_random", "heuristic"]
SeatKind = Literal["human", "random", "stratified_random", "heuristic"]


def build_agent(kind: BotKind, rng: random.Random) -> Agent:
    """One bot seat's agent -- mirrors ``experiments/benchmark.py``'s
    ``_build_role``, extended with ``stratified_random``.
    """
    if kind == "random":
        return RandomAgent(rng, name="random")
    if kind == "stratified_random":
        return StratifiedRandomAgent(rng, name="stratified_random")
    if kind == "heuristic":
        return HeuristicAgent(name="heuristic")
    raise ValueError(f"unknown bot kind {kind!r}")


def build_agents(seat_kinds: list[SeatKind], driver_seed: int) -> list[Agent | None]:
    """One agent per seat (``None`` for a human seat).

    Each bot seat gets its own independently-seeded RNG, keyed by seat, so
    swapping one seat's agent never perturbs another seat's draws.
    """
    agents: list[Agent | None] = []
    for seat, kind in enumerate(seat_kinds):
        if kind == "human":
            agents.append(None)
            continue
        seat_rng = random.Random(hash((driver_seed, seat)))
        agents.append(build_agent(kind, seat_rng))
    return agents


@dataclass(frozen=True, slots=True)
class TrailEntry:
    """One action applied while advancing bot turns (or the human action
    that triggered them) -- everything ``server/serialize.py``'s
    ``serialize_trail`` needs to describe it publicly.

    ``dice_roll`` and ``production`` are only ever set when ``action`` is a
    ``RollDice`` -- ``RollDice`` itself carries no fields, so neither the
    actual roll nor who gained what from it is otherwise visible from the
    action alone. Both are captured at the moment this specific action was
    applied (not at the end of a whole bot-turn batch), so multiple bot
    rolls within one batch each keep their own correct values. Exposing
    ``production`` isn't a new redaction concern: who gains which resources
    from a roll is fully determined by public information already (the
    board, everyone's settlements/cities) -- any player at the table could
    compute it themselves, same as the dice roll itself.

    ``trade_offer`` is only ever set when ``action`` is ``AcceptTrade``,
    ``RejectTrade``, or ``CounterTrade`` -- none of the first two carry
    fields either, so the deal (or rejected offer) being responded to would
    otherwise be invisible; a ``CounterTrade`` carries its own new bundle but
    not the *original* offer it is responding to, which is the same gap.
    Also not a new redaction concern: a domestic trade offer is already
    public once proposed (``player_view``'s own docstring).
    """

    player_id: int
    action: Action
    dice_roll: tuple[int, int] | None = None
    production: dict[int, dict[Resource, int]] | None = None
    trade_offer: TradeOffer | None = None


def _production_diff(
    resources_before: dict[int, dict[Resource, int]], state: GameState
) -> dict[int, dict[Resource, int]]:
    """Per-player resource gains from a just-applied ``RollDice``, found by
    diffing hands before/after -- ``engine.game``'s ``_produce`` computes
    this internally but doesn't return it, so this reconstructs it from the
    outside rather than requiring an ``engine/`` change.
    """
    result: dict[int, dict[Resource, int]] = {}
    for player in state.players:
        before = resources_before[player.player_id]
        gained = {
            r: player.resources[r] - before.get(r, 0)
            for r in Resource
            if player.resources[r] - before.get(r, 0) > 0
        }
        if gained:
            result[player.player_id] = gained
    return result


def apply_and_record(
    state: GameState, game: CatanGame, actor: int, action: Action
) -> TrailEntry:
    """Apply ``action`` to ``state`` and return the ``TrailEntry`` describing
    it -- shared by ``step_bots`` (for bot actions) and ``server/app.py``'s
    ``post_action`` (for the human's own action), so the two never diverge
    on how ``dice_roll``/``production``/``trade_offer`` get captured.
    """
    resources_before = (
        {p.player_id: dict(p.resources) for p in state.players}
        if isinstance(action, RollDice)
        else None
    )
    # AcceptTrade/RejectTrade both clear (or partially clear) state.trade_offer
    # as a side effect -- snapshot it beforehand so the caller can still see
    # what deal was being responded to.
    trade_offer_before = (
        state.trade_offer
        if isinstance(action, AcceptTrade | RejectTrade | CounterTrade)
        else None
    )
    game.apply_action(state, action)
    if resources_before is not None:
        return TrailEntry(
            player_id=actor,
            action=action,
            dice_roll=state.dice_roll,
            production=_production_diff(resources_before, state),
        )
    if trade_offer_before is not None:
        return TrailEntry(
            player_id=actor, action=action, trade_offer=trade_offer_before
        )
    return TrailEntry(player_id=actor, action=action)


def step_bots(session: GameSession) -> list[TrailEntry]:
    """Advance ``session.state`` through consecutive bot turns, in place.

    Stops the moment ``acting_player`` is a human seat (``None`` in
    ``session.agents``) or the game ends -- the server's own request/
    response loop takes over from there. Returns every action applied along
    the way, in order, for ``server/app.py`` to fold into the response's
    action trail (see ``docs/plans`` discussion -- this only ever captures
    *bot* actions; the human action that triggered this call is the
    caller's own responsibility to record).
    """
    game = session.game
    state = session.state
    trail: list[TrailEntry] = []
    while not game.is_terminal(state):
        actor = acting_player(state)
        agent = session.agents[actor]
        if agent is None:
            return trail
        legal = game.legal_actions(state)
        action = agent.choose_action(state, legal, actor)
        trail.append(apply_and_record(state, game, actor, action))
    return trail
