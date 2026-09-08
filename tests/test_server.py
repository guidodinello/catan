"""``server/app.py``: the FastAPI HTTP layer, driven entirely through
``fastapi.testclient.TestClient`` -- no live server needed (``docs/plans/
gui-web-frontend.md``'s verification story).

Rule under test: the HTTP API is a thin wrapper -- it never decides
legality itself, redacts hidden information per viewer, and a game played
purely through HTTP calls reaches the same kind of outcome
(``is_terminal``/``winner``) a direct ``engine.game.CatanGame`` game would.
"""

import random
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

import server.persistence as persistence_mod
from agents import RandomAgent
from engine.board import Resource
from engine.game import CITY_COST, DEV_CARD_COST, ROAD_COST, SETTLEMENT_COST
from engine.state import Phase
from server.app import app
from server.sessions import get_session

client = TestClient(app)


@pytest.fixture(autouse=True)
def _session_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Every route in this module goes through server/app.py's real
    # save_session/delete_snapshot calls -- keep them out of the real
    # .catan-sessions/ directory.
    monkeypatch.setattr(persistence_mod, "SESSION_DIR", tmp_path)
    return tmp_path


def _create_game(
    num_players: int, seat_kinds: list[str], seed: int | None = None
) -> dict[str, Any]:
    response = client.post(
        "/api/games",
        json={"num_players": num_players, "seat_kinds": seat_kinds, "seed": seed},
    )
    assert response.status_code == 200, response.text
    return cast(dict[str, Any], response.json())


def test_create_game_returns_game_id_geometry_and_initial_state() -> None:
    body = _create_game(3, ["human", "human", "human"], seed=1)
    assert "game_id" in body
    assert isinstance(body["driver_seed"], int)
    assert body["geometry"]["port_locations"]
    assert len(body["geometry"]["vertices"]) == 54
    assert len(body["geometry"]["edges"]) == 72
    assert body["state"]["phase"] == Phase.SETUP_SETTLEMENT.name


def test_build_costs_reflects_the_real_engine_constants() -> None:
    # Asserted against the imported constants, not literals -- a test
    # hardcoding e.g. {"BRICK": 1, "LUMBER": 1} would reintroduce the very
    # second source of truth this endpoint exists to avoid.
    body = client.get("/api/build_costs").json()
    assert body.keys() == {"ROAD", "SETTLEMENT", "CITY", "DEV_CARD"}
    assert body["ROAD"] == {r.name: n for r, n in ROAD_COST.items()}
    assert body["SETTLEMENT"] == {r.name: n for r, n in SETTLEMENT_COST.items()}
    assert body["CITY"] == {r.name: n for r, n in CITY_COST.items()}
    assert body["DEV_CARD"] == {r.name: n for r, n in DEV_CARD_COST.items()}
    # Sparse, not all-five-with-zeros -- ROAD_COST has no WOOL/GRAIN/ORE
    # entry at all, and the payload must not silently grow one.
    assert "WOOL" not in body["ROAD"]
    assert "ORE" not in body["ROAD"]


def test_create_game_rejects_seat_kind_count_mismatch() -> None:
    response = client.post(
        "/api/games",
        json={"num_players": 3, "seat_kinds": ["human", "human"], "seed": 1},
    )
    assert response.status_code == 400


def test_create_game_rejects_an_unknown_seat_kind() -> None:
    response = client.post(
        "/api/games",
        json={
            "num_players": 3,
            "seat_kinds": ["human", "human", "not-a-kind"],
            "seed": 1,
        },
    )
    assert response.status_code == 422  # pydantic Literal validation


def test_create_game_with_a_leading_bot_seat_auto_steps_it() -> None:
    # All-bot lineups run the entire game to completion inside the create
    # call itself, since step_bots never finds a human seat to stop at.
    body = _create_game(3, ["heuristic", "heuristic", "heuristic"], seed=1)
    assert body["state"]["phase"] == Phase.GAME_OVER.name
    assert body["state"]["winner"] is not None


def test_create_game_action_trail_is_empty_when_seat_0_is_human() -> None:
    body = _create_game(3, ["human", "heuristic", "heuristic"], seed=1)
    assert body["action_trail"] == []


def test_create_game_with_a_leading_bot_seat_returns_a_non_empty_trail() -> None:
    body = _create_game(3, ["heuristic", "human", "human"], seed=1)
    assert len(body["action_trail"]) > 0
    for entry in body["action_trail"]:
        assert "player_id" in entry
        assert "kind" in entry
        assert "index" not in entry  # trail entries aren't legal-action selectors


def test_get_state_redacts_non_viewer_hands() -> None:
    body = _create_game(4, ["human", "human", "human", "human"], seed=1)
    game_id = body["game_id"]

    response = client.get(f"/api/games/{game_id}/state", params={"viewer": 0})
    assert response.status_code == 200
    state = response.json()
    assert "resources" in state["players"][0]
    for other in state["players"][1:]:
        assert "resources" not in other
        assert "resource_count" in other


def test_get_state_spectator_view_reveals_every_hand() -> None:
    body = _create_game(3, ["human", "human", "human"], seed=1)
    response = client.get(f"/api/games/{body['game_id']}/state")
    state = response.json()
    for entry in state["players"]:
        assert "resources" in entry


def test_get_state_unknown_game_returns_404() -> None:
    response = client.get("/api/games/does-not-exist/state")
    assert response.status_code == 404


def test_get_legal_actions_for_the_acting_human_seat() -> None:
    body = _create_game(3, ["human", "human", "human"], seed=1)
    game_id = body["game_id"]
    actor = body["state"]["acting_player"]

    response = client.get(
        f"/api/games/{game_id}/legal_actions", params={"viewer": actor}
    )
    assert response.status_code == 200
    actions = response.json()
    assert len(actions) > 0
    assert all(a["kind"] == "PlaceSettlement" for a in actions)


def test_get_legal_actions_is_empty_for_a_non_acting_viewer() -> None:
    body = _create_game(3, ["human", "human", "human"], seed=1)
    game_id = body["game_id"]
    actor = body["state"]["acting_player"]
    other_seat = (actor + 1) % 3

    response = client.get(
        f"/api/games/{game_id}/legal_actions", params={"viewer": other_seat}
    )
    assert response.status_code == 200
    assert response.json() == []


def test_post_action_applies_it_and_advances_the_state() -> None:
    body = _create_game(3, ["human", "human", "human"], seed=1)
    game_id = body["game_id"]
    actor = body["state"]["acting_player"]

    legal = client.get(
        f"/api/games/{game_id}/legal_actions", params={"viewer": actor}
    ).json()
    placed_vertex = legal[0]["vertex_id"]

    response = client.post(f"/api/games/{game_id}/action", json={"index": 0})
    assert response.status_code == 200
    new_state = response.json()
    assert placed_vertex in new_state["players"][actor]["settlement_vertices"]


def test_post_action_trail_includes_the_humans_own_action_first() -> None:
    body = _create_game(3, ["human", "human", "human"], seed=1)
    game_id = body["game_id"]
    actor = body["state"]["acting_player"]
    legal = client.get(
        f"/api/games/{game_id}/legal_actions", params={"viewer": actor}
    ).json()
    placed_vertex = legal[0]["vertex_id"]

    response = client.post(f"/api/games/{game_id}/action", json={"index": 0})
    trail = response.json()["action_trail"]
    assert trail[0] == {
        "player_id": actor,
        "kind": "PlaceSettlement",
        "vertex_id": placed_vertex,
    }


def test_post_action_trail_includes_bot_actions_that_follow() -> None:
    # A human seat's very first action (a setup placement) doesn't hand off
    # to any bot yet -- setup visits every seat once each before MAIN. Drive
    # through both setup rounds so at least one human action is immediately
    # followed by consecutive bot turns, and check that response's trail.
    body = _create_game(3, ["human", "heuristic", "heuristic"], seed=1)
    game_id = body["game_id"]

    trail_kinds_seen: list[str] = []
    for _ in range(50):
        state = client.get(f"/api/games/{game_id}/state").json()
        if state["phase"] not in ("SETUP_SETTLEMENT", "SETUP_ROAD"):
            break
        legal = client.get(f"/api/games/{game_id}/legal_actions").json()
        if not legal:
            break
        response = client.post(f"/api/games/{game_id}/action", json={"index": 0})
        assert response.status_code == 200, response.text
        trail = response.json()["action_trail"]
        trail_kinds_seen += [e["kind"] for e in trail]
        if len(trail) > 1:
            # Found a response whose trail has more than just the human's
            # own entry -- i.e. bot actions were folded in too.
            assert any(e["player_id"] != 0 for e in trail), (
                f"expected a bot entry in {trail!r}"
            )
            return
    raise AssertionError(
        f"never saw a multi-entry trail; kinds seen: {trail_kinds_seen!r}"
    )


def test_post_action_trail_dice_roll_is_captured() -> None:
    body = _create_game(3, ["human", "heuristic", "heuristic"], seed=1)
    game_id = body["game_id"]

    # Drive through setup (2 rounds x settlement+road) to reach ROLL.
    for _ in range(20):
        state = client.get(f"/api/games/{game_id}/state").json()
        if state["phase"] == "ROLL" and state["acting_player"] == 0:
            break
        legal = client.get(f"/api/games/{game_id}/legal_actions").json()
        if not legal:
            continue
        client.post(f"/api/games/{game_id}/action", json={"index": 0})
    else:
        raise AssertionError("never reached seat 0's ROLL phase")

    legal = client.get(f"/api/games/{game_id}/legal_actions").json()
    roll_index = next(a["index"] for a in legal if a["kind"] == "RollDice")
    response = client.post(f"/api/games/{game_id}/action", json={"index": roll_index})
    trail = response.json()["action_trail"]
    human_roll = trail[0]
    assert human_roll["kind"] == "RollDice"
    assert "dice_roll" in human_roll
    d1, d2 = human_roll["dice_roll"]
    assert 1 <= d1 <= 6 and 1 <= d2 <= 6


def test_post_action_trail_roll_dice_can_include_production() -> None:
    body = _create_game(3, ["human", "heuristic", "heuristic"], seed=1)
    game_id = body["game_id"]

    found_production = False
    for _ in range(300):
        state = client.get(f"/api/games/{game_id}/state").json()
        if state["phase"] == "GAME_OVER":
            break
        legal = client.get(f"/api/games/{game_id}/legal_actions").json()
        if not legal:
            continue
        response = client.post(f"/api/games/{game_id}/action", json={"index": 0})
        assert response.status_code == 200, response.text
        for entry in response.json()["action_trail"]:
            if entry["kind"] != "RollDice":
                continue
            if "production" not in entry:
                continue  # a real roll that happened to produce nothing
            found_production = True
            for player_id, gains in entry["production"].items():
                assert int(player_id) in (0, 1, 2)
                assert gains, "an entry present in production must be non-empty"
                for resource, amount in gains.items():
                    assert resource in ("LUMBER", "WOOL", "GRAIN", "BRICK", "ORE")
                    assert amount > 0
    assert found_production, "never observed a RollDice entry with production"


def test_post_action_out_of_range_index_returns_400() -> None:
    body = _create_game(3, ["human", "human", "human"], seed=1)
    response = client.post(
        f"/api/games/{body['game_id']}/action", json={"index": 99999}
    )
    assert response.status_code == 400


def test_post_action_on_an_unknown_game_returns_404() -> None:
    response = client.post("/api/games/does-not-exist/action", json={"index": 0})
    assert response.status_code == 404


def test_post_action_rejects_a_bot_seats_turn_defensively() -> None:
    body = _create_game(3, ["human", "human", "human"], seed=1)
    game_id = body["game_id"]
    actor = body["state"]["acting_player"]
    session = get_session(game_id)
    session.agents[actor] = RandomAgent(random.Random(0), name="random")

    response = client.post(f"/api/games/{game_id}/action", json={"index": 0})
    assert response.status_code == 400


def test_post_action_propose_trade_sentinel_requires_a_bundle() -> None:
    body = _create_game(3, ["human", "human", "human"], seed=1)
    game_id = body["game_id"]
    session = get_session(game_id)
    # Jump straight to MAIN, as tests/test_trading.py does.
    session.state.phase = Phase.MAIN
    # _propose_trade_actions only offers the sentinel to a player holding
    # at least one resource card.
    session.state.players[session.state.current_player].resources[Resource.ORE] = 1

    legal = client.get(f"/api/games/{game_id}/legal_actions").json()
    trade_entries = [a for a in legal if a["kind"] == "ProposeTrade"]
    assert len(trade_entries) == 1
    assert trade_entries[0]["open_ended"] is True
    trade_index = trade_entries[0]["index"]

    # No bundle supplied -> the sentinel's empty give/receive is rejected by
    # apply_action's own validation, surfaced as 400.
    response = client.post(f"/api/games/{game_id}/action", json={"index": trade_index})
    assert response.status_code == 400


def test_post_action_propose_trade_with_a_bundle_succeeds() -> None:
    body = _create_game(3, ["human", "human", "human"], seed=1)
    game_id = body["game_id"]
    session = get_session(game_id)
    session.state.phase = Phase.MAIN
    actor = session.state.current_player
    session.state.players[actor].resources[Resource.LUMBER] = 2

    legal = client.get(f"/api/games/{game_id}/legal_actions").json()
    trade_index = next(a["index"] for a in legal if a["kind"] == "ProposeTrade")

    response = client.post(
        f"/api/games/{game_id}/action",
        json={"index": trade_index, "give": {"LUMBER": 2}, "receive": {"ORE": 1}},
    )
    assert response.status_code == 200
    new_state = response.json()
    assert new_state["trade_offer"] == {
        "proposer": actor,
        "give": {"LUMBER": 2},
        "receive": {"ORE": 1},
        "counter_of": None,
    }


def test_post_action_trail_reject_trade_includes_the_offer() -> None:
    body = _create_game(3, ["human", "human", "human"], seed=1)
    game_id = body["game_id"]
    session = get_session(game_id)
    session.state.phase = Phase.MAIN
    proposer = session.state.current_player
    session.state.players[proposer].resources[Resource.LUMBER] = 2

    legal = client.get(f"/api/games/{game_id}/legal_actions").json()
    trade_index = next(a["index"] for a in legal if a["kind"] == "ProposeTrade")
    client.post(
        f"/api/games/{game_id}/action",
        json={"index": trade_index, "give": {"LUMBER": 2}, "receive": {"ORE": 1}},
    )

    responder_legal = client.get(f"/api/games/{game_id}/legal_actions").json()
    reject_index = next(
        a["index"] for a in responder_legal if a["kind"] == "RejectTrade"
    )
    response = client.post(f"/api/games/{game_id}/action", json={"index": reject_index})
    reject_entry = response.json()["action_trail"][0]
    assert reject_entry["kind"] == "RejectTrade"
    assert reject_entry["trade_offer"] == {
        "proposer": proposer,
        "give": {"LUMBER": 2},
        "receive": {"ORE": 1},
        "counter_of": None,
    }


def test_post_action_trail_accept_trade_includes_the_offer() -> None:
    body = _create_game(3, ["human", "human", "human"], seed=1)
    game_id = body["game_id"]
    session = get_session(game_id)
    session.state.phase = Phase.MAIN
    proposer = session.state.current_player
    session.state.players[proposer].resources[Resource.LUMBER] = 2

    legal = client.get(f"/api/games/{game_id}/legal_actions").json()
    trade_index = next(a["index"] for a in legal if a["kind"] == "ProposeTrade")
    client.post(
        f"/api/games/{game_id}/action",
        json={"index": trade_index, "give": {"LUMBER": 2}, "receive": {"ORE": 1}},
    )

    responder = session.state.trade_responders[0]
    session.state.players[responder].resources[Resource.ORE] = 1
    responder_legal = client.get(f"/api/games/{game_id}/legal_actions").json()
    accept_index = next(
        a["index"] for a in responder_legal if a["kind"] == "AcceptTrade"
    )
    response = client.post(f"/api/games/{game_id}/action", json={"index": accept_index})
    accept_entry = response.json()["action_trail"][0]
    assert accept_entry["kind"] == "AcceptTrade"
    assert accept_entry["trade_offer"] == {
        "proposer": proposer,
        "give": {"LUMBER": 2},
        "receive": {"ORE": 1},
        "counter_of": None,
    }


def test_post_action_counter_trade_sentinel_requires_a_bundle() -> None:
    body = _create_game(3, ["human", "human", "human"], seed=1)
    game_id = body["game_id"]
    session = get_session(game_id)
    session.state.phase = Phase.MAIN
    proposer = session.state.current_player
    session.state.players[proposer].resources[Resource.LUMBER] = 2

    legal = client.get(f"/api/games/{game_id}/legal_actions").json()
    trade_index = next(a["index"] for a in legal if a["kind"] == "ProposeTrade")
    client.post(
        f"/api/games/{game_id}/action",
        json={"index": trade_index, "give": {"LUMBER": 2}, "receive": {"ORE": 1}},
    )

    responder = session.state.trade_responders[0]
    session.state.players[responder].resources[Resource.ORE] = 1
    responder_legal = client.get(f"/api/games/{game_id}/legal_actions").json()
    counter_entries = [a for a in responder_legal if a["kind"] == "CounterTrade"]
    assert len(counter_entries) == 1
    assert counter_entries[0]["open_ended"] is True
    counter_index = counter_entries[0]["index"]

    response = client.post(
        f"/api/games/{game_id}/action", json={"index": counter_index}
    )
    assert response.status_code == 400


def test_post_action_counter_trade_with_a_bundle_succeeds() -> None:
    body = _create_game(3, ["human", "human", "human"], seed=1)
    game_id = body["game_id"]
    session = get_session(game_id)
    session.state.phase = Phase.MAIN
    proposer = session.state.current_player
    session.state.players[proposer].resources[Resource.LUMBER] = 2

    legal = client.get(f"/api/games/{game_id}/legal_actions").json()
    trade_index = next(a["index"] for a in legal if a["kind"] == "ProposeTrade")
    client.post(
        f"/api/games/{game_id}/action",
        json={"index": trade_index, "give": {"LUMBER": 2}, "receive": {"ORE": 1}},
    )

    responder = session.state.trade_responders[0]
    session.state.players[responder].resources[Resource.ORE] = 1
    responder_legal = client.get(f"/api/games/{game_id}/legal_actions").json()
    counter_index = next(
        a["index"] for a in responder_legal if a["kind"] == "CounterTrade"
    )

    response = client.post(
        f"/api/games/{game_id}/action",
        json={"index": counter_index, "give": {"ORE": 1}, "receive": {"LUMBER": 1}},
    )
    assert response.status_code == 200
    new_state = response.json()
    assert new_state["phase"] == Phase.AWAIT_TRADE_RESPONSE.name
    assert new_state["trade_offer"] == {
        "proposer": responder,
        "give": {"ORE": 1},
        "receive": {"LUMBER": 1},
        "counter_of": proposer,
    }
    assert new_state["trade_responders"] == [proposer]


def test_delete_game_removes_the_session() -> None:
    body = _create_game(3, ["human", "human", "human"], seed=1)
    game_id = body["game_id"]
    response = client.delete(f"/api/games/{game_id}")
    assert response.status_code == 200
    assert client.get(f"/api/games/{game_id}/state").status_code == 404


def test_full_game_via_http_with_one_human_seat_reaches_a_winner() -> None:
    body = _create_game(3, ["human", "heuristic", "heuristic"], seed=1)
    game_id = body["game_id"]

    for _ in range(2000):
        state = client.get(f"/api/games/{game_id}/state").json()
        if state["phase"] == Phase.GAME_OVER.name:
            break
        legal = client.get(f"/api/games/{game_id}/legal_actions").json()
        assert legal, "human seat is acting but has no legal actions"
        response = client.post(f"/api/games/{game_id}/action", json={"index": 0})
        assert response.status_code == 200, response.text
    else:
        raise AssertionError("game did not finish within the iteration budget")

    final_state = client.get(f"/api/games/{game_id}/state").json()
    assert final_state["phase"] == Phase.GAME_OVER.name
    assert final_state["winner"] is not None


def test_create_and_post_action_write_a_session_snapshot(tmp_path: Path) -> None:
    body = _create_game(3, ["human", "human", "human"], seed=1)
    game_id = body["game_id"]
    assert (tmp_path / f"{game_id}.pickle").exists()

    legal = client.get(f"/api/games/{game_id}/legal_actions").json()
    assert legal
    client.post(f"/api/games/{game_id}/action", json={"index": 0})
    # Still present (and re-written) after a real mutation, not just at
    # creation time.
    assert (tmp_path / f"{game_id}.pickle").exists()


def test_delete_game_also_removes_its_snapshot(tmp_path: Path) -> None:
    body = _create_game(3, ["human", "human", "human"], seed=1)
    game_id = body["game_id"]
    assert (tmp_path / f"{game_id}.pickle").exists()

    client.delete(f"/api/games/{game_id}")

    assert not (tmp_path / f"{game_id}.pickle").exists()


def test_a_game_survives_an_app_restart() -> None:
    # The end-to-end payoff this whole feature exists for: a game created
    # against one FastAPI app instance is resumable, by the same game_id,
    # against a brand new instance -- simulating the backend restart a
    # server/ code change forces (docs/backlog.md's "Sessions don't survive
    # a backend restart").
    with TestClient(app) as restart_client:
        body = restart_client.post(
            "/api/games",
            json={
                "num_players": 3,
                "seat_kinds": ["human", "heuristic", "heuristic"],
                "seed": 1,
            },
        ).json()
        game_id = body["game_id"]
        before = restart_client.get(f"/api/games/{game_id}/state").json()

    # A fresh TestClient over the same `app` object re-runs the lifespan
    # startup handler, mirroring a fresh `uvicorn server.app:app` process
    # picking the same .catan-sessions/ directory back up.
    with TestClient(app) as restart_client:
        after = restart_client.get(f"/api/games/{game_id}/state").json()
        assert after["phase"] == before["phase"]
        assert after["current_player"] == before["current_player"]

        legal = restart_client.get(f"/api/games/{game_id}/legal_actions").json()
        assert legal or after["phase"] != before["phase"]
