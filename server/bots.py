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
from typing import TYPE_CHECKING, Literal

from agents import Agent, HeuristicAgent, RandomAgent, StratifiedRandomAgent
from engine.state import acting_player

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


def step_bots(session: GameSession) -> None:
    """Advance ``session.state`` through consecutive bot turns, in place.

    Stops the moment ``acting_player`` is a human seat (``None`` in
    ``session.agents``) or the game ends -- the server's own request/
    response loop takes over from there.
    """
    game = session.game
    state = session.state
    while not game.is_terminal(state):
        actor = acting_player(state)
        agent = session.agents[actor]
        if agent is None:
            return
        legal = game.legal_actions(state)
        action = agent.choose_action(state, legal, actor)
        game.apply_action(state, action)
