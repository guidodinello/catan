"""CatanGame: reset / legal_actions / apply_action / is_terminal / winner.

Dispatches on ``state.phase`` to private ``_{phase}_legal`` / ``_{phase}_apply``
helpers. ``apply_action`` mutates the given state in place and returns it.

Scope simplifications (documented, not rules bugs):
  - Dev cards may only be *bought* during MAIN (post-roll); they may be
    *played* during ROLL or MAIN per the official "any time on your turn" rule.
  - PlayRoadBuilding is only offered when two sequential legal placements
    exist and the player has >=2 road pieces left (rather than allowing a
    single free road when only one placement/piece is available).

Architectural principle: the engine models the real rules exactly, with no
compromises made for the sake of a future flat/enumerable RL action space.
Where a legal action's parameter domain is small (PlayYearOfPlenty: pick 2 of
5 resources), legal_actions() pre-enumerates every parametrized instance.
Where it is not (ProposeTrade: any non-empty multi-resource give/receive
bundle), legal_actions() offers a single open-ended affordance and the actual
bundle is constructed by the caller and fully validated in apply_action --
never bounded to a single resource type or otherwise restricted beyond the
real rules just to make it enumerable. Any bounding/flattening needed for a
future discrete action space (Phase 5) belongs in a separate adapter/encoding
layer, never baked into the engine as a rules compromise -- the same
principle catanatron applies by confining its flat space to a separate gym
subpackage.
"""

from __future__ import annotations

import random
from collections import Counter

from .actions import (
    AcceptTrade,
    Action,
    BuyDevCard,
    Discard,
    EndTurn,
    MoveRobber,
    PlaceCity,
    PlaceRoad,
    PlaceSettlement,
    PlayKnight,
    PlayMonopoly,
    PlayRoadBuilding,
    PlayVictoryPoint,
    PlayYearOfPlenty,
    ProposeTrade,
    RejectTrade,
    RollDice,
    StealFrom,
    TradeBank,
    TradePort,
)
from .board import (
    GEOMETRY,
    NUM_EDGES,
    NUM_VERTICES,
    PORT_RESOURCE,
    TERRAIN_RESOURCE,
    PortType,
    Resource,
    Terrain,
    generate_board,
)
from .state import (
    BANK_RESOURCE_COUNT,
    DEV_DECK_COUNTS,
    LARGEST_ARMY_MIN_KNIGHTS,
    LONGEST_ROAD_MIN_LENGTH,
    MAX_HAND_BEFORE_DISCARD,
    WINNING_VICTORY_POINTS,
    DevCard,
    DevCardType,
    GameState,
    Phase,
    PlayerState,
    TradeOffer,
)

ROAD_COST: dict[Resource, int] = {Resource.BRICK: 1, Resource.LUMBER: 1}
SETTLEMENT_COST: dict[Resource, int] = {
    Resource.BRICK: 1,
    Resource.LUMBER: 1,
    Resource.WOOL: 1,
    Resource.GRAIN: 1,
}
CITY_COST: dict[Resource, int] = {Resource.GRAIN: 2, Resource.ORE: 3}
DEV_CARD_COST: dict[Resource, int] = {
    Resource.ORE: 1,
    Resource.WOOL: 1,
    Resource.GRAIN: 1,
}

MIN_PLAYERS = 3
MAX_PLAYERS = 4


class IllegalActionError(Exception):
    """Raised when an action fails a rule check during apply_action."""


# ---------------------------------------------------------------------------
# Reset / setup
# ---------------------------------------------------------------------------


def _resolve_starting_player(rng: random.Random, num_players: int) -> int:
    """Highest combined 2d6 roll starts; ties reroll among the tied players."""
    candidates = list(range(num_players))
    while len(candidates) > 1:
        rolls = {p: rng.randint(1, 6) + rng.randint(1, 6) for p in candidates}
        best = max(rolls.values())
        candidates = [p for p in candidates if rolls[p] == best]
    return candidates[0]


class CatanGame:
    def __init__(self, num_players: int = 4) -> None:
        if not (MIN_PLAYERS <= num_players <= MAX_PLAYERS):
            raise ValueError("base-game Catan supports 3 or 4 players")
        self.num_players = num_players

    def reset(self, seed: int | None = None) -> GameState:
        rng = random.Random(seed)
        board = generate_board(rng)
        players = [PlayerState(player_id=i) for i in range(self.num_players)]

        dev_deck: list[DevCardType] = []
        for card_type, count in DEV_DECK_COUNTS.items():
            dev_deck.extend([card_type] * count)
        rng.shuffle(dev_deck)

        bank = {r: BANK_RESOURCE_COUNT for r in Resource}

        starting_player = _resolve_starting_player(rng, self.num_players)
        order = [
            (starting_player + i) % self.num_players for i in range(self.num_players)
        ]
        setup_sequence = order + list(reversed(order))

        return GameState(
            board=board,
            players=players,
            current_player=setup_sequence[0],
            phase=Phase.SETUP_SETTLEMENT,
            dev_deck=dev_deck,
            bank=bank,
            rng=rng,
            setup_sequence=setup_sequence,
            setup_position=0,
        )

    def legal_actions(self, state: GameState) -> list[Action]:
        handler = _LEGAL_HANDLERS.get(state.phase)
        if handler is None:
            return []
        return handler(state)

    def apply_action(self, state: GameState, action: Action) -> GameState:
        handler = _APPLY_HANDLERS.get(state.phase)
        if handler is None:
            raise IllegalActionError(f"cannot apply an action in phase {state.phase}")
        return handler(state, action)

    def is_terminal(self, state: GameState) -> bool:
        return state.phase == Phase.GAME_OVER

    def winner(self, state: GameState) -> int | None:
        return state.winner


# ---------------------------------------------------------------------------
# Shared lookups
# ---------------------------------------------------------------------------


def _vertex_owner(state: GameState, vertex_id: int) -> int | None:
    for i, player in enumerate(state.players):
        if vertex_id in player.settlement_vertices or vertex_id in player.city_vertices:
            return i
    return None


def _edge_owner(state: GameState, edge_id: int) -> int | None:
    for i, player in enumerate(state.players):
        if edge_id in player.road_edges:
            return i
    return None


def _has_resources(player: PlayerState, cost: dict[Resource, int]) -> bool:
    return all(player.resources.get(r, 0) >= c for r, c in cost.items())


def _pay(state: GameState, player: PlayerState, cost: dict[Resource, int]) -> None:
    if not _has_resources(player, cost):
        raise IllegalActionError("insufficient resources")
    for r, c in cost.items():
        player.resources[r] -= c
        state.bank[r] += c


def victory_points(state: GameState, player_idx: int) -> int:
    """*Public* victory-point tally for ``player_idx``: revealed VP dev cards
    only, deliberately excluding unrevealed cards still sitting in ``dev_hand``
    (see ``PlayerState.revealed_vp_cards``).

    This is the number every player at the table can see, and the only one safe
    to serialize for *any* viewer -- it is what ``server/serialize.py`` publishes
    per seat and what ``experiments/`` records as the observable score series.
    It is **not** the win condition: see ``true_victory_points``.
    """
    player = state.players[player_idx]
    vp = len(player.settlement_vertices) + 2 * len(player.city_vertices)
    vp += player.revealed_vp_cards
    if state.longest_road_owner == player_idx:
        vp += 2
    if state.largest_army_owner == player_idx:
        vp += 2
    return vp


def true_victory_points(state: GameState, player_idx: int) -> int:
    """The *real* victory-point total: the public tally plus every VP dev card
    still in hand, revealed or not -- the engine's actual win condition.

    Per the real rule, the instant a player's true total reaches 10 they win
    immediately and automatically, mid-turn, on whatever action crossed the
    threshold; revealing a VP card is showing proof, not a game action with its
    own timing. A card bought this very turn counts, matching
    ``_dev_card_play_actions``, which offers ``PlayVictoryPoint`` regardless of
    ``bought_this_turn``.

    Kept separate from ``victory_points`` (rather than replacing it) because
    the difference between the two *is* hidden information: this number must
    never reach a viewer who isn't the card's owner. It is intentionally not
    re-exported from ``engine/__init__.py`` -- ``server/`` and ``web/`` have no
    business reading it, and only ``_check_win`` calls it.
    """
    player = state.players[player_idx]
    hidden_vp = sum(
        1 for c in player.dev_hand if c.card_type is DevCardType.VICTORY_POINT
    )
    return victory_points(state, player_idx) + hidden_vp


def _reveal_victory_point_cards(state: GameState, player_idx: int) -> None:
    """Move every VP dev card out of ``player_idx``'s hand into the public
    ``revealed_vp_cards`` count -- one card out, one count in, so the dev-card
    conservation invariant holds. Called only when the game ends, on the winner
    alone: a loser's hand stays hidden, same as the real game.
    """
    player = state.players[player_idx]
    remaining = [
        c for c in player.dev_hand if c.card_type is not DevCardType.VICTORY_POINT
    ]
    player.revealed_vp_cards += len(player.dev_hand) - len(remaining)
    player.dev_hand[:] = remaining


def _check_win(state: GameState) -> None:
    """End the game if the turn player's *true* total has reached 10.

    Mutates on a win beyond the phase/winner fields: the winner's VP dev cards
    are revealed (``_reveal_victory_point_cards``) in the same transition that
    sets GAME_OVER, so the public ``victory_points`` tally becomes accurate
    exactly when the game ends -- never an instant before, which is what keeps
    ``server/serialize.py``'s opponent-hand redaction honest.
    """
    if state.phase == Phase.GAME_OVER:
        return
    if true_victory_points(state, state.current_player) >= WINNING_VICTORY_POINTS:
        state.winner = state.current_player
        state.phase = Phase.GAME_OVER
        _reveal_victory_point_cards(state, state.current_player)


# ---------------------------------------------------------------------------
# Placement legality
# ---------------------------------------------------------------------------


def _distance_rule_ok(state: GameState, vertex_id: int) -> bool:
    if _vertex_owner(state, vertex_id) is not None:
        return False
    return all(
        _vertex_owner(state, n) is None for n in GEOMETRY.vertex_neighbors[vertex_id]
    )


def _vertex_connects_for_road(
    state: GameState, player_idx: int, vertex_id: int
) -> bool:
    owner = _vertex_owner(state, vertex_id)
    if owner is not None:
        return owner == player_idx
    return any(
        _edge_owner(state, e) == player_idx for e in GEOMETRY.vertex_edges[vertex_id]
    )


def _road_legal_edges(state: GameState, player_idx: int) -> list[int]:
    legal = []
    for e in range(NUM_EDGES):
        if _edge_owner(state, e) is not None:
            continue
        v1, v2 = GEOMETRY.edge_vertices[e]
        if _vertex_connects_for_road(
            state, player_idx, v1
        ) or _vertex_connects_for_road(state, player_idx, v2):
            legal.append(e)
    return legal


def _settlement_legal_vertices(state: GameState, player_idx: int) -> list[int]:
    legal = []
    for v in range(NUM_VERTICES):
        if not _distance_rule_ok(state, v):
            continue
        if any(_edge_owner(state, e) == player_idx for e in GEOMETRY.vertex_edges[v]):
            legal.append(v)
    return legal


def _city_legal_vertices(state: GameState, player_idx: int) -> list[int]:
    player = state.players[player_idx]
    if player.cities_remaining <= 0:
        return []
    return list(player.settlement_vertices)


# ---------------------------------------------------------------------------
# Longest road / largest army
# ---------------------------------------------------------------------------


def _longest_road_length(state: GameState, player_idx: int) -> int:
    player = state.players[player_idx]
    if not player.road_edges:
        return 0
    adjacency: dict[int, list[tuple[int, int]]] = {}
    for e in player.road_edges:
        v1, v2 = GEOMETRY.edge_vertices[e]
        adjacency.setdefault(v1, []).append((e, v2))
        adjacency.setdefault(v2, []).append((e, v1))

    best = 0

    def dfs(vertex: int, visited_edges: set[int]) -> None:
        nonlocal best
        best = max(best, len(visited_edges))
        owner = _vertex_owner(state, vertex)
        if owner is not None and owner != player_idx:
            return
        for edge_id, next_vertex in adjacency.get(vertex, []):
            if edge_id in visited_edges:
                continue
            visited_edges.add(edge_id)
            dfs(next_vertex, visited_edges)
            visited_edges.remove(edge_id)

    for start_vertex in adjacency:
        dfs(start_vertex, set())
    return best


def _update_longest_road(state: GameState) -> None:
    lengths = {i: _longest_road_length(state, i) for i in range(len(state.players))}
    qualifying = {i: n for i, n in lengths.items() if n >= LONGEST_ROAD_MIN_LENGTH}
    incumbent = state.longest_road_owner
    if incumbent is not None and incumbent in qualifying:
        max_len = max(qualifying.values())
        if qualifying[incumbent] == max_len:
            return  # incumbent keeps the card, ties included
    if not qualifying:
        state.longest_road_owner = None
        return
    max_len = max(qualifying.values())
    leaders = [i for i, n in qualifying.items() if n == max_len]
    state.longest_road_owner = leaders[0] if len(leaders) == 1 else None


def _update_largest_army(state: GameState) -> None:
    counts = {i: p.played_knights for i, p in enumerate(state.players)}
    qualifying = {i: c for i, c in counts.items() if c >= LARGEST_ARMY_MIN_KNIGHTS}
    incumbent = state.largest_army_owner
    if incumbent is not None and incumbent in qualifying:
        max_count = max(qualifying.values())
        if qualifying[incumbent] == max_count:
            return
    if not qualifying:
        state.largest_army_owner = None
        return
    max_count = max(qualifying.values())
    leaders = [i for i, c in qualifying.items() if c == max_count]
    if len(leaders) == 1:
        state.largest_army_owner = leaders[0]


# ---------------------------------------------------------------------------
# Setup phase
# ---------------------------------------------------------------------------


def _setup_settlement_legal(state: GameState) -> list[Action]:
    return [
        PlaceSettlement(v) for v in range(NUM_VERTICES) if _distance_rule_ok(state, v)
    ]


def _setup_settlement_apply(state: GameState, action: Action) -> GameState:
    if not isinstance(action, PlaceSettlement):
        raise IllegalActionError(f"illegal action during setup settlement: {action!r}")
    v = action.vertex_id
    if not _distance_rule_ok(state, v):
        raise IllegalActionError("distance rule violated")
    player = state.players[state.current_player]
    player.settlement_vertices.add(v)
    player.settlements_remaining -= 1
    state.last_settlement_vertex = v

    is_second_settlement = state.setup_position >= len(state.players)
    if is_second_settlement:
        for h in GEOMETRY.vertex_hexes[v]:
            if GEOMETRY.is_land(h) and state.board.terrain[h] is not Terrain.DESERT:
                resource = TERRAIN_RESOURCE[state.board.terrain[h]]
                player.resources[resource] += 1
                state.bank[resource] -= 1

    state.phase = Phase.SETUP_ROAD
    return state


def _setup_road_legal(state: GameState) -> list[Action]:
    v = state.last_settlement_vertex
    assert v is not None
    return [
        PlaceRoad(e) for e in GEOMETRY.vertex_edges[v] if _edge_owner(state, e) is None
    ]


def _setup_road_apply(state: GameState, action: Action) -> GameState:
    if not isinstance(action, PlaceRoad):
        raise IllegalActionError(f"illegal action during setup road: {action!r}")
    e = action.edge_id
    v = state.last_settlement_vertex
    assert v is not None
    if _edge_owner(state, e) is not None:
        raise IllegalActionError("edge already has a road")
    if v not in GEOMETRY.edge_vertices[e]:
        raise IllegalActionError("setup road must touch the settlement just placed")

    player = state.players[state.current_player]
    player.road_edges.add(e)
    player.roads_remaining -= 1
    state.setup_position += 1

    if state.setup_position < len(state.setup_sequence):
        state.current_player = state.setup_sequence[state.setup_position]
        state.phase = Phase.SETUP_SETTLEMENT
    else:
        state.current_player = state.setup_sequence[0]
        state.phase = Phase.ROLL
    return state


# ---------------------------------------------------------------------------
# Roll / production / seven / robber / steal
# ---------------------------------------------------------------------------


def _produce(state: GameState, total: int) -> None:
    gains: dict[Resource, dict[int, int]] = {r: {} for r in Resource}
    for h, token in state.board.tokens.items():
        if token != total or h == state.board.robber_hex:
            continue
        terrain = state.board.terrain[h]
        if terrain is Terrain.DESERT:
            continue
        resource = TERRAIN_RESOURCE[terrain]
        for v in GEOMETRY.hex_vertices[h]:
            owner = _vertex_owner(state, v)
            if owner is None:
                continue
            amount = 2 if v in state.players[owner].city_vertices else 1
            gains[resource][owner] = gains[resource].get(owner, 0) + amount

    for resource, per_player in gains.items():
        if not per_player:
            continue
        total_needed = sum(per_player.values())
        available = state.bank[resource]
        if available >= total_needed:
            for owner, amount in per_player.items():
                state.players[owner].resources[resource] += amount
            state.bank[resource] -= total_needed
        elif len(per_player) == 1:
            (owner, amount) = next(iter(per_player.items()))
            grant = min(amount, available)
            state.players[owner].resources[resource] += grant
            state.bank[resource] -= grant
        # else: shortage affects 2+ players -> nobody receives this resource


def _enter_discard_or_robber(state: GameState) -> None:
    pending = [
        i
        for i, p in enumerate(state.players)
        if p.resource_card_count() > MAX_HAND_BEFORE_DISCARD
    ]
    if pending:
        state.pending_discards = pending
        state.discard_amounts = {
            i: state.players[i].resource_card_count() // 2 for i in pending
        }
        state.phase = Phase.DISCARD
    else:
        state.phase = Phase.MOVE_ROBBER


def _roll_dice(state: GameState) -> None:
    d1 = state.rng.randint(1, 6)
    d2 = state.rng.randint(1, 6)
    state.dice_roll = (d1, d2)
    total = d1 + d2
    if total == 7:
        state.robber_return_phase = Phase.MAIN
        _enter_discard_or_robber(state)
    else:
        _produce(state, total)
        state.phase = Phase.MAIN


def _steal_candidates(state: GameState) -> list[int]:
    hex_vertices = GEOMETRY.hex_vertices[state.board.robber_hex]
    candidates = []
    for i, p in enumerate(state.players):
        if i == state.current_player:
            continue
        if any(
            v in p.settlement_vertices or v in p.city_vertices for v in hex_vertices
        ):
            candidates.append(i)
    return candidates


def _move_robber_legal(state: GameState) -> list[Action]:
    return [MoveRobber(h) for h in GEOMETRY.land_hexes if h != state.board.robber_hex]


def _move_robber_apply(state: GameState, action: Action) -> GameState:
    if not isinstance(action, MoveRobber):
        raise IllegalActionError(f"illegal action during move-robber: {action!r}")
    if action.hex_id == state.board.robber_hex or not GEOMETRY.is_land(action.hex_id):
        raise IllegalActionError("robber must move to a different land hex")
    state.board.robber_hex = action.hex_id
    if _steal_candidates(state):
        state.phase = Phase.STEAL
    else:
        state.phase = state.robber_return_phase
    return state


def _steal_legal(state: GameState) -> list[Action]:
    return [StealFrom(i) for i in _steal_candidates(state)]


def _steal_apply(state: GameState, action: Action) -> GameState:
    if not isinstance(action, StealFrom):
        raise IllegalActionError(f"illegal action during steal: {action!r}")
    if action.player_idx not in _steal_candidates(state):
        raise IllegalActionError("no such steal candidate")
    victim = state.players[action.player_idx]
    if victim.resource_card_count() > 0:
        pool = [r for r, c in victim.resources.items() for _ in range(c)]
        chosen = state.rng.choice(pool)
        victim.resources[chosen] -= 1
        state.players[state.current_player].resources[chosen] += 1
    state.phase = state.robber_return_phase
    return state


def _discard_legal(state: GameState) -> list[Action]:
    player_idx = state.pending_discards[0]
    required = state.discard_amounts[player_idx]
    hand = state.players[player_idx].resources
    return [Discard(resources=c) for c in _multiset_combinations(hand, required)]


def _multiset_combinations(
    hand: dict[Resource, int], required: int
) -> list[dict[Resource, int]]:
    resources = [r for r, c in hand.items() if c > 0]
    results: list[dict[Resource, int]] = []

    def rec(i: int, remaining: int, current: dict[Resource, int]) -> None:
        if remaining == 0:
            results.append(dict(current))
            return
        if i == len(resources):
            return
        r = resources[i]
        max_take = min(hand[r], remaining)
        for take in range(max_take + 1):
            if take:
                current[r] = take
            rec(i + 1, remaining - take, current)
            if take:
                del current[r]

    rec(0, required, {})
    return results


def _discard_apply(state: GameState, action: Action) -> GameState:
    if not isinstance(action, Discard):
        raise IllegalActionError(f"illegal action during discard: {action!r}")
    player_idx = state.pending_discards[0]
    required = state.discard_amounts[player_idx]
    total = sum(action.resources.values())
    if total != required:
        raise IllegalActionError(f"must discard exactly {required} cards")
    player = state.players[player_idx]
    for r, c in action.resources.items():
        if c > player.resources[r]:
            raise IllegalActionError("cannot discard more cards than held")
        player.resources[r] -= c
        state.bank[r] += c
    state.pending_discards.pop(0)
    del state.discard_amounts[player_idx]
    if not state.pending_discards:
        state.phase = Phase.MOVE_ROBBER
    return state


def _roll_legal(state: GameState) -> list[Action]:
    actions: list[Action] = [RollDice()]
    actions += _dev_card_play_actions(state, state.current_player)
    return actions


def _roll_apply(state: GameState, action: Action) -> GameState:
    p_idx = state.current_player
    if isinstance(action, RollDice):
        _roll_dice(state)
    elif isinstance(action, PlayKnight):
        _play_knight(state, p_idx, Phase.ROLL)
    elif isinstance(action, PlayRoadBuilding):
        _play_road_building(state, p_idx, action.edge_id_1, action.edge_id_2)
    elif isinstance(action, PlayYearOfPlenty):
        _play_year_of_plenty(state, p_idx, action.resource_1, action.resource_2)
    elif isinstance(action, PlayMonopoly):
        _play_monopoly(state, p_idx, action.resource)
    elif isinstance(action, PlayVictoryPoint):
        _play_victory_point(state, p_idx)
    else:
        raise IllegalActionError(f"illegal action in ROLL: {action!r}")
    return state


# ---------------------------------------------------------------------------
# Dev cards
# ---------------------------------------------------------------------------


def _dev_card_playable(player: PlayerState, card_type: DevCardType) -> bool:
    return any(
        c.card_type == card_type and not c.bought_this_turn for c in player.dev_hand
    )


def _consume_dev_card(player: PlayerState, card_type: DevCardType) -> None:
    for i, c in enumerate(player.dev_hand):
        if c.card_type == card_type and not c.bought_this_turn:
            del player.dev_hand[i]
            return
    raise IllegalActionError(f"no playable {card_type} card")


def _road_building_pairs(state: GameState, player_idx: int) -> list[tuple[int, int]]:
    player = state.players[player_idx]
    pairs: list[tuple[int, int]] = []
    for e1 in _road_legal_edges(state, player_idx):
        player.road_edges.add(e1)
        try:
            for e2 in _road_legal_edges(state, player_idx):
                if e2 != e1:
                    pairs.append((e1, e2))
        finally:
            player.road_edges.discard(e1)
    return pairs


def _year_of_plenty_options() -> list[tuple[Resource, Resource]]:
    resources = list(Resource)
    return [
        (resources[i], resources[j])
        for i in range(len(resources))
        for j in range(i, len(resources))
    ]


def _dev_card_play_actions(state: GameState, player_idx: int) -> list[Action]:
    player = state.players[player_idx]
    actions: list[Action] = []
    if not player.has_played_dev_card_this_turn:
        if _dev_card_playable(player, DevCardType.KNIGHT):
            actions.append(PlayKnight())
        if (
            _dev_card_playable(player, DevCardType.ROAD_BUILDING)
            and player.roads_remaining >= 2
        ):
            actions += [
                PlayRoadBuilding(e1, e2)
                for e1, e2 in _road_building_pairs(state, player_idx)
            ]
        if _dev_card_playable(player, DevCardType.YEAR_OF_PLENTY):
            actions += [
                PlayYearOfPlenty(r1, r2) for r1, r2 in _year_of_plenty_options()
            ]
        if _dev_card_playable(player, DevCardType.MONOPOLY):
            actions += [PlayMonopoly(r) for r in Resource]
    if any(c.card_type == DevCardType.VICTORY_POINT for c in player.dev_hand):
        actions.append(PlayVictoryPoint())
    return actions


def _buy_dev_card(state: GameState, player_idx: int) -> None:
    if not state.dev_deck:
        raise IllegalActionError("dev card deck is empty")
    player = state.players[player_idx]
    _pay(state, player, DEV_CARD_COST)
    card_type = state.dev_deck.pop()
    player.dev_hand.append(DevCard(card_type=card_type, bought_this_turn=True))
    # A bought VICTORY_POINT card can itself cross the true-VP threshold --
    # unlike every other card type, buying it changes the *true* (though not
    # the public) tally, so this is a new win-check site (see docs/backlog.md).
    _check_win(state)


def _play_knight(state: GameState, player_idx: int, return_phase: Phase) -> None:
    player = state.players[player_idx]
    if player.has_played_dev_card_this_turn:
        raise IllegalActionError("already played a dev card this turn")
    _consume_dev_card(player, DevCardType.KNIGHT)
    player.played_knights += 1
    player.has_played_dev_card_this_turn = True
    state.robber_return_phase = return_phase
    _update_largest_army(state)
    _check_win(state)
    if state.phase != Phase.GAME_OVER:
        state.phase = Phase.MOVE_ROBBER


def _play_road_building(state: GameState, player_idx: int, e1: int, e2: int) -> None:
    player = state.players[player_idx]
    if player.has_played_dev_card_this_turn:
        raise IllegalActionError("already played a dev card this turn")
    if player.roads_remaining < 2:
        raise IllegalActionError("not enough road pieces remaining")
    if e1 not in _road_legal_edges(state, player_idx):
        raise IllegalActionError("illegal first road placement")
    _consume_dev_card(player, DevCardType.ROAD_BUILDING)
    player.road_edges.add(e1)
    player.roads_remaining -= 1
    if e2 not in _road_legal_edges(state, player_idx):
        player.road_edges.discard(e1)
        player.roads_remaining += 1
        player.dev_hand.append(
            DevCard(DevCardType.ROAD_BUILDING, bought_this_turn=False)
        )
        raise IllegalActionError("illegal second road placement")
    player.road_edges.add(e2)
    player.roads_remaining -= 1
    player.has_played_dev_card_this_turn = True
    player.played_progress_count += 1
    _update_longest_road(state)
    _check_win(state)


def _play_year_of_plenty(
    state: GameState, player_idx: int, r1: Resource, r2: Resource
) -> None:
    player = state.players[player_idx]
    if player.has_played_dev_card_this_turn:
        raise IllegalActionError("already played a dev card this turn")
    _consume_dev_card(player, DevCardType.YEAR_OF_PLENTY)
    for resource, count in Counter([r1, r2]).items():
        grant = min(count, state.bank[resource])
        player.resources[resource] += grant
        state.bank[resource] -= grant
    player.has_played_dev_card_this_turn = True
    player.played_progress_count += 1


def _play_monopoly(state: GameState, player_idx: int, resource: Resource) -> None:
    player = state.players[player_idx]
    if player.has_played_dev_card_this_turn:
        raise IllegalActionError("already played a dev card this turn")
    _consume_dev_card(player, DevCardType.MONOPOLY)
    for i, other in enumerate(state.players):
        if i == player_idx:
            continue
        taken = other.resources[resource]
        other.resources[resource] = 0
        player.resources[resource] += taken
    player.has_played_dev_card_this_turn = True
    player.played_progress_count += 1


def _play_victory_point(state: GameState, player_idx: int) -> None:
    player = state.players[player_idx]
    for i, c in enumerate(player.dev_hand):
        if c.card_type == DevCardType.VICTORY_POINT:
            del player.dev_hand[i]
            break
    else:
        raise IllegalActionError("no victory point card to reveal")
    player.revealed_vp_cards += 1
    _check_win(state)


# ---------------------------------------------------------------------------
# Trading
# ---------------------------------------------------------------------------


def _validate_trade_shape(
    give: dict[Resource, int], receive: dict[Resource, int]
) -> None:
    if not give or not receive:
        raise IllegalActionError("trade must be non-empty on both sides")
    if set(give) & set(receive):
        raise IllegalActionError("cannot trade like-for-like resources")


def _execute_trade_with_bank(
    state: GameState,
    player_idx: int,
    give: dict[Resource, int],
    receive: dict[Resource, int],
) -> None:
    player = state.players[player_idx]
    if not _has_resources(player, give):
        raise IllegalActionError("insufficient resources to trade")
    for r, c in receive.items():
        if state.bank.get(r, 0) < c:
            raise IllegalActionError("bank cannot cover this trade")
    for r, c in give.items():
        player.resources[r] -= c
        state.bank[r] += c
    for r, c in receive.items():
        player.resources[r] += c
        state.bank[r] -= c


def _apply_trade_bank(
    state: GameState,
    player_idx: int,
    give: dict[Resource, int],
    receive: dict[Resource, int],
) -> None:
    _validate_trade_shape(give, receive)
    if sum(give.values()) != 4 * sum(receive.values()):
        raise IllegalActionError("bank trade must be at a 4:1 ratio")
    _execute_trade_with_bank(state, player_idx, give, receive)


def _player_port_rates(state: GameState, player_idx: int) -> dict[Resource, int]:
    player = state.players[player_idx]
    owned_vertices = player.settlement_vertices | player.city_vertices
    rates: dict[Resource, int] = {}
    has_generic = False
    for edge_id, port_type in state.board.port_types.items():
        v1, v2 = GEOMETRY.edge_vertices[edge_id]
        if v1 in owned_vertices or v2 in owned_vertices:
            if port_type is PortType.GENERIC:
                has_generic = True
            else:
                rates[PORT_RESOURCE[port_type]] = 2
    if has_generic:
        for r in Resource:
            rates.setdefault(r, 3)
    return rates


def _apply_trade_port(
    state: GameState,
    player_idx: int,
    give: dict[Resource, int],
    receive: dict[Resource, int],
) -> None:
    _validate_trade_shape(give, receive)
    rates = _player_port_rates(state, player_idx)
    credits = 0
    for r, c in give.items():
        rate = rates.get(r)
        if rate is None:
            raise IllegalActionError(f"no port access for {r}")
        if c % rate != 0:
            raise IllegalActionError("give amount must be a multiple of the port rate")
        credits += c // rate
    if credits != sum(receive.values()):
        raise IllegalActionError("port trade give/receive amounts do not match")
    _execute_trade_with_bank(state, player_idx, give, receive)


MAX_TRADE_OFFER_SIDE = 4


def _propose_trade(
    state: GameState,
    player_idx: int,
    give: dict[Resource, int],
    receive: dict[Resource, int],
) -> None:
    """Propose a domestic trade. ``give``/``receive`` are full multi-resource
    bundles (e.g. give 2 lumber + 1 brick for 1 ore) -- real Catan trades are
    routinely multi-resource, so no single-resource-type restriction is
    applied. The only cap is a per-side card count, kept small purely so an
    interactive builder (the CLI) doesn't need to prompt for absurd amounts;
    it is not a rules restriction and does not force a single resource type.
    """
    _validate_trade_shape(give, receive)
    if sum(give.values()) > MAX_TRADE_OFFER_SIDE:
        raise IllegalActionError(f"give side is capped at {MAX_TRADE_OFFER_SIDE} cards")
    if sum(receive.values()) > MAX_TRADE_OFFER_SIDE:
        raise IllegalActionError(
            f"receive side is capped at {MAX_TRADE_OFFER_SIDE} cards"
        )
    if not _has_resources(state.players[player_idx], give):
        raise IllegalActionError("cannot offer resources you do not have")
    state.trade_offer = TradeOffer(
        proposer=player_idx, give=dict(give), receive=dict(receive)
    )
    state.trade_responders = [i for i in range(len(state.players)) if i != player_idx]
    state.phase = Phase.AWAIT_TRADE_RESPONSE


def _bank_trade_actions(state: GameState, player_idx: int) -> list[Action]:
    player = state.players[player_idx]
    actions: list[Action] = []
    for give_r in Resource:
        if player.resources[give_r] >= 4:
            actions += [
                TradeBank(give={give_r: 4}, receive={receive_r: 1})
                for receive_r in Resource
                if receive_r != give_r and state.bank[receive_r] >= 1
            ]
    return actions


def _port_trade_actions(state: GameState, player_idx: int) -> list[Action]:
    player = state.players[player_idx]
    rates = _player_port_rates(state, player_idx)
    actions: list[Action] = []
    for give_r, rate in rates.items():
        if player.resources[give_r] >= rate:
            actions += [
                TradePort(give={give_r: rate}, receive={receive_r: 1})
                for receive_r in Resource
                if receive_r != give_r and state.bank[receive_r] >= 1
            ]
    return actions


def _propose_trade_actions(state: GameState, player_idx: int) -> list[Action]:
    """A single open-ended affordance, not a pre-enumeration.

    The give/receive bundle space (any non-empty multi-resource multiset on
    each side, up to MAX_TRADE_OFFER_SIDE cards) is too large to usefully
    pre-enumerate -- and unlike PlayYearOfPlenty's tiny domain, enumerating it
    would bury a human player under thousands of near-duplicate menu entries.
    This sentinel (empty give/receive) signals "you may propose a trade";
    the caller (the CLI, prompting a human; or a test driver, constructing
    one at random) builds the actual bundle and passes a fully-specified
    ProposeTrade to apply_action, which validates it in full -- the sentinel
    itself always fails validation (both sides must be non-empty) so it can
    never be applied unmodified.
    """
    player = state.players[player_idx]
    if player.resource_card_count() == 0:
        return []
    return [ProposeTrade(give={}, receive={})]


def _trade_response_legal(state: GameState) -> list[Action]:
    responder = state.trade_responders[0]
    actions: list[Action] = [RejectTrade()]
    offer = state.trade_offer
    if offer is not None and _has_resources(state.players[responder], offer.receive):
        actions.append(AcceptTrade())
    return actions


def _trade_response_apply(state: GameState, action: Action) -> GameState:
    responder = state.trade_responders[0]
    offer = state.trade_offer
    if isinstance(action, AcceptTrade):
        if offer is None:
            raise IllegalActionError("no trade offer pending")
        proposer_player = state.players[offer.proposer]
        responder_player = state.players[responder]
        if not _has_resources(responder_player, offer.receive):
            raise IllegalActionError("responder lacks the requested resources")
        for r, c in offer.give.items():
            proposer_player.resources[r] -= c
            responder_player.resources[r] += c
        for r, c in offer.receive.items():
            responder_player.resources[r] -= c
            proposer_player.resources[r] += c
        state.trade_offer = None
        state.trade_responders = []
        state.phase = Phase.MAIN
    elif isinstance(action, RejectTrade):
        state.trade_responders.pop(0)
        if not state.trade_responders:
            state.trade_offer = None
            state.phase = Phase.MAIN
    else:
        raise IllegalActionError(
            f"illegal action while awaiting trade response: {action!r}"
        )
    return state


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def _build_settlement(state: GameState, player_idx: int, vertex_id: int) -> None:
    player = state.players[player_idx]
    if vertex_id not in _settlement_legal_vertices(state, player_idx):
        raise IllegalActionError("illegal settlement placement")
    if player.settlements_remaining <= 0:
        raise IllegalActionError("no settlement pieces remaining")
    _pay(state, player, SETTLEMENT_COST)
    player.settlement_vertices.add(vertex_id)
    player.settlements_remaining -= 1
    _update_longest_road(state)
    _check_win(state)


def _build_road(state: GameState, player_idx: int, edge_id: int) -> None:
    player = state.players[player_idx]
    if edge_id not in _road_legal_edges(state, player_idx):
        raise IllegalActionError("illegal road placement")
    if player.roads_remaining <= 0:
        raise IllegalActionError("no road pieces remaining")
    _pay(state, player, ROAD_COST)
    player.road_edges.add(edge_id)
    player.roads_remaining -= 1
    _update_longest_road(state)
    _check_win(state)


def _build_city(state: GameState, player_idx: int, vertex_id: int) -> None:
    player = state.players[player_idx]
    if vertex_id not in player.settlement_vertices:
        raise IllegalActionError("a city must upgrade the builder's own settlement")
    if player.cities_remaining <= 0:
        raise IllegalActionError("no city pieces remaining")
    _pay(state, player, CITY_COST)
    player.settlement_vertices.discard(vertex_id)
    player.settlements_remaining += 1
    player.city_vertices.add(vertex_id)
    player.cities_remaining -= 1
    _check_win(state)


def _end_turn(state: GameState) -> None:
    n = len(state.players)
    state.current_player = (state.current_player + 1) % n
    new_player = state.players[state.current_player]
    for c in new_player.dev_hand:
        c.bought_this_turn = False
    new_player.has_played_dev_card_this_turn = False
    state.dice_roll = None
    state.phase = Phase.ROLL
    _check_win(state)


def _main_legal(state: GameState) -> list[Action]:
    p_idx = state.current_player
    player = state.players[p_idx]
    actions: list[Action] = [EndTurn()]

    if player.settlements_remaining > 0 and _has_resources(player, SETTLEMENT_COST):
        actions += [
            PlaceSettlement(v) for v in _settlement_legal_vertices(state, p_idx)
        ]
    if player.roads_remaining > 0 and _has_resources(player, ROAD_COST):
        actions += [PlaceRoad(e) for e in _road_legal_edges(state, p_idx)]
    if player.cities_remaining > 0 and _has_resources(player, CITY_COST):
        actions += [PlaceCity(v) for v in _city_legal_vertices(state, p_idx)]
    if state.dev_deck and _has_resources(player, DEV_CARD_COST):
        actions.append(BuyDevCard())

    actions += _dev_card_play_actions(state, p_idx)
    actions += _bank_trade_actions(state, p_idx)
    actions += _port_trade_actions(state, p_idx)
    actions += _propose_trade_actions(state, p_idx)
    return actions


def _main_apply(state: GameState, action: Action) -> GameState:
    p_idx = state.current_player
    if isinstance(action, EndTurn):
        _end_turn(state)
    elif isinstance(action, PlaceSettlement):
        _build_settlement(state, p_idx, action.vertex_id)
    elif isinstance(action, PlaceRoad):
        _build_road(state, p_idx, action.edge_id)
    elif isinstance(action, PlaceCity):
        _build_city(state, p_idx, action.vertex_id)
    elif isinstance(action, BuyDevCard):
        _buy_dev_card(state, p_idx)
    elif isinstance(action, PlayKnight):
        _play_knight(state, p_idx, Phase.MAIN)
    elif isinstance(action, PlayRoadBuilding):
        _play_road_building(state, p_idx, action.edge_id_1, action.edge_id_2)
    elif isinstance(action, PlayYearOfPlenty):
        _play_year_of_plenty(state, p_idx, action.resource_1, action.resource_2)
    elif isinstance(action, PlayMonopoly):
        _play_monopoly(state, p_idx, action.resource)
    elif isinstance(action, PlayVictoryPoint):
        _play_victory_point(state, p_idx)
    elif isinstance(action, TradeBank):
        _apply_trade_bank(state, p_idx, action.give, action.receive)
    elif isinstance(action, TradePort):
        _apply_trade_port(state, p_idx, action.give, action.receive)
    elif isinstance(action, ProposeTrade):
        _propose_trade(state, p_idx, action.give, action.receive)
    else:
        raise IllegalActionError(f"illegal action in MAIN: {action!r}")
    return state


_LEGAL_HANDLERS = {
    Phase.SETUP_SETTLEMENT: _setup_settlement_legal,
    Phase.SETUP_ROAD: _setup_road_legal,
    Phase.ROLL: _roll_legal,
    Phase.DISCARD: _discard_legal,
    Phase.MOVE_ROBBER: _move_robber_legal,
    Phase.STEAL: _steal_legal,
    Phase.MAIN: _main_legal,
    Phase.AWAIT_TRADE_RESPONSE: _trade_response_legal,
}

_APPLY_HANDLERS = {
    Phase.SETUP_SETTLEMENT: _setup_settlement_apply,
    Phase.SETUP_ROAD: _setup_road_apply,
    Phase.ROLL: _roll_apply,
    Phase.DISCARD: _discard_apply,
    Phase.MOVE_ROBBER: _move_robber_apply,
    Phase.STEAL: _steal_apply,
    Phase.MAIN: _main_apply,
    Phase.AWAIT_TRADE_RESPONSE: _trade_response_apply,
}
