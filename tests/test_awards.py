"""Longest Road and Largest Army award tracking.

Rule under test (Almanac p. 9): Longest Road is the longest SIMPLE PATH (no
edge reused, branches not summed) of >=5 segments; first to reach it takes
the card; a strictly longer road by another player steals it; an opponent
settlement on an unoccupied intersection along the path breaks it; the card
can become unowned when the incumbent drops out of the lead and 2+ others
tie for the new max, or when nobody has >=5. Largest Army (p. 5; Almanac p.
8): first to 3+ played knights takes it; a strictly greater count steals it;
ties simply mean the incumbent keeps it.
"""

from engine.board import GEOMETRY
from engine.game import (
    CatanGame,
    _longest_road_length,
    _update_largest_army,
    _update_longest_road,
)
from engine.state import GameState, Phase


def _fresh_state(num_players: int = 3) -> tuple[CatanGame, GameState]:
    game = CatanGame(num_players=num_players)
    state = game.reset(seed=1)
    for p in state.players:
        p.settlement_vertices.clear()
        p.city_vertices.clear()
        p.road_edges.clear()
    state.phase = Phase.MAIN
    return game, state


def _path_edges(start_vertex: int, length: int) -> list[int]:
    """Walk a simple path of `length` edges outward from start_vertex,
    avoiding revisiting a vertex, using whatever geometry is available."""
    edges: list[int] = []
    visited = {start_vertex}
    current = start_vertex
    for _ in range(length):
        for e in GEOMETRY.vertex_edges[current]:
            other = [v for v in GEOMETRY.edge_vertices[e] if v != current][0]
            if other not in visited and e not in edges:
                edges.append(e)
                visited.add(other)
                current = other
                break
        else:
            raise RuntimeError("could not extend path further on this board")
    return edges


def test_longest_road_requires_at_least_five_segments() -> None:
    game, state = _fresh_state()
    start = 0
    edges = _path_edges(start, 4)
    state.players[0].road_edges.update(edges)
    _update_longest_road(state)
    assert state.longest_road_owner is None

    fifth = _path_edges(start, 5)
    state.players[0].road_edges.clear()
    state.players[0].road_edges.update(fifth)
    _update_longest_road(state)
    assert state.longest_road_owner == 0


def test_branching_road_counts_longest_simple_path_not_sum_of_branches() -> None:
    """3-edge path + a 2-edge branch off one of its vertices sums to 5, but
    the longest *simple* path through the resulting tree is only 4 edges."""
    game, state = _fresh_state()
    main_path = _path_edges(0, 3)
    main_vertices = [0]
    current_v = 0
    for e in main_path:
        current_v = [v for v in GEOMETRY.edge_vertices[e] if v != current_v][0]
        main_vertices.append(current_v)
    interior_vertices = main_vertices[1:-1]  # exclude both endpoints
    main_vertices_set = set(main_vertices)

    branch_edges: list[int] | None = None
    for hub in interior_vertices:
        used_vertices = set(main_vertices_set)
        candidate: list[int] = []
        current = hub
        ok = True
        for _ in range(2):
            found = False
            for e in GEOMETRY.vertex_edges[current]:
                if e in main_path or e in candidate:
                    continue
                other = [v for v in GEOMETRY.edge_vertices[e] if v != current][0]
                if other not in used_vertices:
                    candidate.append(e)
                    used_vertices.add(other)
                    current = other
                    found = True
                    break
            if not found:
                ok = False
                break
        if ok:
            branch_edges = candidate
            break

    assert branch_edges is not None, "could not find a 2-edge branch on this board"
    state.players[0].road_edges.update(main_path)
    state.players[0].road_edges.update(branch_edges)

    length = _longest_road_length(state, 0)
    assert length == 4  # not 5 (3 + 2 summed)


def test_strictly_longer_road_steals_the_card() -> None:
    game, state = _fresh_state()
    state.players[0].road_edges.update(_path_edges(0, 5))
    _update_longest_road(state)
    assert state.longest_road_owner == 0

    # player 1 builds a strictly longer road elsewhere
    other_start = next(
        v for v in range(54) if v not in {0} | GEOMETRY.vertex_neighbors[0]
    )
    state.players[1].road_edges.update(_path_edges(other_start, 6))
    _update_longest_road(state)
    assert state.longest_road_owner == 1


def test_incumbent_keeps_card_on_tie() -> None:
    game, state = _fresh_state()
    state.players[0].road_edges.update(_path_edges(0, 5))
    _update_longest_road(state)
    assert state.longest_road_owner == 0

    other_start = next(
        v for v in range(54) if v not in {0} | GEOMETRY.vertex_neighbors[0]
    )
    state.players[1].road_edges.update(_path_edges(other_start, 5))
    _update_longest_road(state)
    assert state.longest_road_owner == 0  # tie: incumbent keeps it


def test_card_becomes_unowned_when_incumbent_drops_and_others_tie() -> None:
    game, state = _fresh_state()
    chain = _path_edges(0, 6)
    state.players[0].road_edges.update(chain)
    _update_longest_road(state)
    assert state.longest_road_owner == 0

    # break player 0's road down below the new max by removing the middle
    # edge of the chain, splitting it into two pieces each shorter than 5
    state.players[0].road_edges.discard(chain[len(chain) // 2])
    assert _longest_road_length(state, 0) < 5

    other_start_1 = next(
        v for v in range(54) if v not in {0} | GEOMETRY.vertex_neighbors[0]
    )
    state.players[1].road_edges.update(_path_edges(other_start_1, 5))

    other_start_2 = next(
        v
        for v in range(54)
        if v
        not in (
            {0, other_start_1}
            | GEOMETRY.vertex_neighbors[0]
            | GEOMETRY.vertex_neighbors[other_start_1]
        )
    )
    state.players[2].road_edges.update(_path_edges(other_start_2, 5))

    _update_longest_road(state)
    assert state.longest_road_owner is None


def test_largest_army_requires_three_knights_and_strict_steal() -> None:
    game, state = _fresh_state()
    state.players[0].played_knights = 2
    _update_largest_army(state)
    assert state.largest_army_owner is None

    state.players[0].played_knights = 3
    _update_largest_army(state)
    assert state.largest_army_owner == 0

    state.players[1].played_knights = 3
    _update_largest_army(state)
    assert state.largest_army_owner == 0  # tie: incumbent keeps it

    state.players[1].played_knights = 4
    _update_largest_army(state)
    assert state.largest_army_owner == 1  # strictly greater steals it
