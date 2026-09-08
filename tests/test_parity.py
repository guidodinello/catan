"""Bit-identity cross-check between the HTTP path (``server/app.py`` +
``server/bots.py``) and the batch driver (``experiments/rollout.run_game``).

Per ``docs/backlog.md``'s "No bit-identity cross-check..." item: nothing
today would catch the two drivers silently diverging (a different
action-selection order, a stray extra RNG draw, an off-by-one in the bot
loop) -- only the batch side is guarded, by
``tests/test_agents.py::test_bit_identity_against_pre_refactor_golden_records``.

This module plays the *same* all-bot game twice -- once entirely through one
``POST /api/games`` call (``create_game`` runs ``step_bots`` to completion
when every seat is a bot), once through ``run_game`` fed the same
``engine_seed`` and the ``driver_seed`` the HTTP response now exposes -- and
asserts the two resulting raw ``GameState``s and action sequences match.

Deliberately not compared against ``player_view``: that function *redacts*
(hides dev-card hands, the remaining ``dev_deck``), exactly the fields a
driver divergence would corrupt most quietly. This compares raw state.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

import server.persistence as persistence_mod
from agents import Agent
from engine.actions import Action
from engine.state import GameState
from experiments.rollout import AgentFactory, run_game
from server.app import app
from server.bots import SeatKind, TrailEntry, build_agents
from server.serialize import serialize_trail_entry
from server.sessions import get_session

client = TestClient(app)


@pytest.fixture(autouse=True)
def _session_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Same reasoning as tests/test_server.py's fixture of the same name --
    # keep this module's games out of the real .catan-sessions/.
    monkeypatch.setattr(persistence_mod, "SESSION_DIR", tmp_path)
    return tmp_path


class _RecordingAgent:
    """Wraps a real ``Agent``, appending every ``(actor, action)`` it
    produces to a single log shared across *all* seats' recorders, and
    remembering the ``state`` reference it's handed.

    The shared log (rather than a per-instance one) is what makes the
    recorded sequence globally ordered: only one seat's agent is ever
    consulted per step, so a per-instance log would just be that seat's
    actions in isolation, with no way to recover interleaving across seats.

    ``game.apply_action`` mutates ``state`` in place (both
    ``experiments/rollout.py``'s loop and ``server/bots.py``'s
    ``apply_and_record`` read post-action fields like ``state.dice_roll``
    off the same object right after calling it) -- so the reference
    captured on this agent's very first ``choose_action`` call *is* the
    final state once ``run_game`` returns. With no ``scripted_setup`` in
    play, every action ``run_game``'s loop applies comes from some agent,
    so the shared log ends up as the complete action sequence for the whole
    game.
    """

    def __init__(self, inner: Agent, shared_log: list[tuple[int, Action]]) -> None:
        self.inner = inner
        self.name = inner.name
        self._shared_log = shared_log
        self.seen_state: GameState | None = None

    def choose_action(
        self, state: GameState, legal_actions: list[Action], player_idx: int
    ) -> Action:
        self.seen_state = state
        action = self.inner.choose_action(state, legal_actions, player_idx)
        self._shared_log.append((player_idx, action))
        return action

    def reset(self) -> None:
        self.inner.reset()


def _http_parity_factory(
    seat_kinds: list[SeatKind],
    recorders: list[_RecordingAgent],
    shared_log: list[tuple[int, Action]],
) -> AgentFactory:
    """Adapts ``server.bots.build_agents``' ``list[Agent | None]`` shape to
    ``rollout.AgentFactory``'s ``list[Agent]`` shape -- the backlog's
    "harder half".

    Calls ``build_agents`` itself rather than re-deriving its per-seat
    ``random.Random(hash((driver_seed, seat)))`` seeding rule: re-implementing
    that here would make the test tautological -- it would stay green even
    if ``build_agents``' own seeding ever changed. ``recorders``/``shared_log``
    are populated in place so the caller can inspect them after ``run_game``
    returns.
    """

    def factory(num_players: int, engine_seed: int, driver_seed: int) -> list[Agent]:
        del engine_seed  # unused: build_agents only needs seat_kinds + driver_seed
        assert num_players == len(seat_kinds)
        agents = build_agents(seat_kinds, driver_seed)
        assert all(a is not None for a in agents), (
            "parity factory is for all-bot lineups only"
        )
        wrapped = [_RecordingAgent(cast(Agent, a), shared_log) for a in agents]
        recorders[:] = wrapped
        return cast(list[Agent], wrapped)

    return factory


def _state_projection(state: GameState) -> dict[str, Any]:
    """Everything in ``GameState`` that identifies the game, minus the
    ``rng`` field itself (compared separately via ``getstate()``, since
    ``random.Random`` has no ``__eq__`` and isn't otherwise comparable).

    ``GameState``/``PlayerState``/``Board`` are plain ``@dataclass(slots=True)``
    of dicts/lists/sets/Enums -- ``dataclasses.asdict`` decomposes them fully.
    """
    return {
        "board": dataclasses.asdict(state.board),
        "players": [dataclasses.asdict(p) for p in state.players],
        "current_player": state.current_player,
        "phase": state.phase,
        "dev_deck": list(state.dev_deck),
        "bank": dict(state.bank),
        "robber_return_phase": state.robber_return_phase,
        "pending_discards": list(state.pending_discards),
        "discard_amounts": dict(state.discard_amounts),
        "trade_offer": state.trade_offer,
        "trade_responders": list(state.trade_responders),
        "longest_road_owner": state.longest_road_owner,
        "largest_army_owner": state.largest_army_owner,
        "setup_sequence": list(state.setup_sequence),
        "setup_position": state.setup_position,
        "last_settlement_vertex": state.last_settlement_vertex,
        "winner": state.winner,
        "dice_roll": state.dice_roll,
    }


# Keys apply_and_record derives from before/after diffs, which an
# agent-level proxy can't observe -- stripped from both trails before
# comparing. This is sufficient, not a hole: all three are fully determined
# by the action sequence plus the engine RNG stream, and the state
# projection above (esp. rng.getstate()) pins both of those independently.
_DIFF_DERIVED_TRAIL_KEYS = ("dice_roll", "production", "trade_offer")


def _strip_diff_derived(trail: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {k: v for k, v in entry.items() if k not in _DIFF_DERIVED_TRAIL_KEYS}
        for entry in trail
    ]


def _rollout_trail(log: list[tuple[int, Action]]) -> list[dict[str, Any]]:
    entries = [
        serialize_trail_entry(TrailEntry(player_id=player_id, action=action))
        for player_id, action in log
    ]
    return _strip_diff_derived(entries)


def _run_parity_case(num_players: int, seat_kinds: list[SeatKind], seed: int) -> None:
    response = client.post(
        "/api/games",
        json={"num_players": num_players, "seat_kinds": seat_kinds, "seed": seed},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    game_id = body["game_id"]
    driver_seed = body["driver_seed"]
    http_trail = _strip_diff_derived(body["action_trail"])

    http_state = get_session(game_id).state

    recorders: list[_RecordingAgent] = []
    shared_log: list[tuple[int, Action]] = []
    record = run_game(
        num_players,
        engine_seed=seed,
        driver_seed=driver_seed,
        agent_factory=_http_parity_factory(seat_kinds, recorders, shared_log),
    )
    # Guard against a false pass: a step_budget truncation must fail as
    # itself, not masquerade as a parity mismatch (step_bots has no such
    # budget, so only run_game's side could hit it).
    assert record.winner is not None, "rollout side hit its step_budget"

    rollout_state = next(r.seen_state for r in recorders if r.seen_state is not None)

    assert _state_projection(http_state) == _state_projection(rollout_state)
    assert http_state.rng.getstate() == rollout_state.rng.getstate()

    assert _rollout_trail(shared_log) == http_trail


def test_parity_all_stratified_random_three_players() -> None:
    _run_parity_case(
        3, ["stratified_random", "stratified_random", "stratified_random"], seed=7
    )


def test_parity_all_stratified_random_four_players() -> None:
    _run_parity_case(
        4,
        [
            "stratified_random",
            "stratified_random",
            "stratified_random",
            "stratified_random",
        ],
        seed=42,
    )


def test_parity_mixed_bot_lineup() -> None:
    # heuristic seats alone can't exercise driver_seed (HeuristicAgent
    # ignores the rng build_agent hands it) -- at least one stratified_random
    # or random seat is required so a broken driver_seed plumbing actually
    # fails this test instead of passing vacuously.
    _run_parity_case(3, ["stratified_random", "random", "heuristic"], seed=99)
