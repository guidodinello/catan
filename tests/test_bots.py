"""``server/bots.py``: seat-kind -> ``Agent`` factory and the bot step loop.

Rule under test: ``build_agents`` maps ``"human"`` to ``None`` and every bot
kind to the matching ``agents/`` class with its own independent RNG stream
(README decision 17); ``step_bots`` advances ``session.state`` through
consecutive bot turns and stops the instant the acting seat is human or the
game ends, never returning the unresolved ``ProposeTrade`` sentinel
(README decision 18 -- bot agents already resolve it themselves).
"""

import random

import pytest

from agents import HeuristicAgent, RandomAgent, StratifiedRandomAgent
from engine.actions import RollDice
from engine.state import acting_player
from server.bots import build_agent, build_agents, step_bots
from server.sessions import create_session


def test_build_agent_constructs_the_matching_class_and_name() -> None:
    rng = random.Random(0)
    random_agent = build_agent("random", rng)
    assert isinstance(random_agent, RandomAgent)
    assert random_agent.name == "random"

    stratified_agent = build_agent("stratified_random", rng)
    assert isinstance(stratified_agent, StratifiedRandomAgent)
    assert stratified_agent.name == "stratified_random"

    heuristic_agent = build_agent("heuristic", rng)
    assert isinstance(heuristic_agent, HeuristicAgent)
    assert heuristic_agent.name == "heuristic"


def test_build_agent_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown bot kind"):
        build_agent("not-a-kind", random.Random(0))  # type: ignore[arg-type]


def test_build_agents_marks_human_seats_as_none() -> None:
    agents = build_agents(["human", "random", "heuristic"], driver_seed=1)
    assert agents[0] is None
    assert isinstance(agents[1], RandomAgent)
    assert isinstance(agents[2], HeuristicAgent)


def test_build_agents_gives_each_bot_seat_an_independent_rng_stream() -> None:
    agents = build_agents(["random", "random"], driver_seed=1)
    first, second = agents
    assert isinstance(first, RandomAgent)
    assert isinstance(second, RandomAgent)
    assert first.rng is not second.rng


def test_step_bots_is_a_no_op_when_the_acting_seat_is_human() -> None:
    _, session = create_session(3, [None, None, None], seed=1)
    phase_before = session.state.phase
    actor_before = acting_player(session.state)
    step_bots(session)
    assert session.state.phase == phase_before
    assert acting_player(session.state) == actor_before


def test_step_bots_stops_at_a_human_seat_or_game_over() -> None:
    agents = build_agents(["human", "random", "random"], driver_seed=1)
    _, session = create_session(3, agents, seed=1)
    step_bots(session)
    state = session.state
    actor = acting_player(state)
    assert session.game.is_terminal(state) or session.agents[actor] is None


def test_step_bots_runs_an_all_bot_game_to_completion() -> None:
    agents = build_agents(["heuristic", "heuristic", "heuristic"], driver_seed=1)
    _, session = create_session(3, agents, seed=1)
    step_bots(session)
    assert session.game.is_terminal(session.state)
    assert session.game.winner(session.state) is not None


def test_step_bots_returns_an_empty_trail_when_the_acting_seat_is_human() -> None:
    _, session = create_session(3, [None, None, None], seed=1)
    assert step_bots(session) == []


def test_step_bots_trail_has_one_entry_per_applied_action_all_from_bot_seats() -> None:
    # Bot seat first, so it's the acting seat from the very start of setup --
    # human at seat 0 would leave step_bots nothing to do before returning.
    agents = build_agents(["heuristic", "human", "human"], driver_seed=1)
    _, session = create_session(3, agents, seed=1)
    trail = step_bots(session)
    assert len(trail) > 0
    for entry in trail:
        assert session.agents[entry.player_id] is not None  # never a human seat


def test_step_bots_trail_dice_roll_is_set_only_on_roll_dice_entries() -> None:
    agents = build_agents(["heuristic", "heuristic", "heuristic"], driver_seed=1)
    _, session = create_session(3, agents, seed=1)
    trail = step_bots(session)
    roll_entries = [e for e in trail if isinstance(e.action, RollDice)]
    assert roll_entries  # a full game has at least one roll
    for entry in trail:
        if isinstance(entry.action, RollDice):
            assert entry.dice_roll is not None
            assert 1 <= entry.dice_roll[0] <= 6
            assert 1 <= entry.dice_roll[1] <= 6
        else:
            assert entry.dice_roll is None
