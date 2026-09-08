"""``HeuristicAgent`` -- a fixed-policy agent whose rules are each derived
from a committed Phase 2 result, not invented. See ``docs/plans/phase-3-agents.md``
for the source citation behind every rule below.

Reads only the ``legal_actions`` list it is handed plus public ``GameState``
fields (``PlayerState.settlement_vertices`` etc.) -- it does not call any
underscore-prefixed engine helper. The one exception is a "would this action
enable a build" check (used for bank/port trades): it runs the real engine on
a ``state.copy()`` and inspects the resulting public ``legal_actions()``,
which is the public API doing the work, not a private accessor.

Agents do not propose domestic trades (README decisions 5, 10) -- the
``ProposeTrade`` sentinel is filtered out of every candidate set. When
another agent proposes a trade to this one, it always rejects -- and never
counters either, even though a ``CounterTrade`` sentinel may be on offer:
negotiating a trade is out of scope for a fixed heuristic (see the Phase 3
plan's scope boundary).
"""

from __future__ import annotations

from engine.actions import (
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
    PlayVictoryPoint,
    PlayYearOfPlenty,
    ProposeTrade,
    RejectTrade,
    RollDice,
    StealFrom,
    TradeBank,
    TradePort,
)
from engine.board import GEOMETRY, Board, Cube, Resource
from engine.game import CITY_COST, ROAD_COST, SETTLEMENT_COST, CatanGame
from engine.state import GameState, Phase, PlayerState
from experiments.features import pips, vertex_features

# Calibrated so the vertex score's ordering reproduces
# experiments/results/a1_seed1_seat0_p4.md's top arms (vertex 30: pip_sum=11,
# 3 distinct resources -- outranking vertex 34's pip_sum=12, 3 distinct is
# noise at n=800, but the score correctly ranks vertex 30 above vertex 32's
# pip_sum=11, 2 distinct, which the data supports: 0.62 vs 0.41 win rate).
W_DISTINCT = 2.0


def _vertex_owner(state: GameState, vertex_id: int) -> int | None:
    """Public-field reimplementation of ``engine.game._vertex_owner`` --
    not a call into the private helper, just reading public
    ``PlayerState.settlement_vertices``/``city_vertices``."""
    for i, player in enumerate(state.players):
        if vertex_id in player.settlement_vertices or vertex_id in player.city_vertices:
            return i
    return None


def _vertex_score(board: Board, vertex_id: int) -> float:
    f = vertex_features(board, vertex_id)
    return f.pip_sum + W_DISTINCT * f.distinct_resources


def _shortfall(
    player: PlayerState, cost: dict[Resource, int], resource: Resource
) -> int:
    return max(0, cost.get(resource, 0) - player.resources.get(resource, 0))


class HeuristicAgent:
    def __init__(self, name: str = "heuristic") -> None:
        self.name = name

    def reset(self) -> None:
        pass

    def choose_action(
        self, state: GameState, legal_actions: list[Action], player_idx: int
    ) -> Action:
        phase = state.phase
        if phase is Phase.SETUP_SETTLEMENT:
            return self._setup_settlement(state, legal_actions)
        if phase is Phase.SETUP_ROAD:
            return self._setup_road(state, legal_actions)
        if phase is Phase.DISCARD:
            return self._discard(state, legal_actions, player_idx)
        if phase is Phase.MOVE_ROBBER:
            return self._move_robber(state, legal_actions, player_idx)
        if phase is Phase.STEAL:
            return self._steal(state, legal_actions)
        if phase is Phase.ROLL:
            return self._roll_phase(state, legal_actions, player_idx)
        if phase is Phase.MAIN:
            return self._main_phase(state, legal_actions, player_idx)
        # AWAIT_TRADE_RESPONSE: the heuristic never proposes, and never
        # negotiates as a responder either -- always reject.
        for a in legal_actions:
            if isinstance(a, RejectTrade):
                return a
        return legal_actions[0]

    # -- Setup ----------------------------------------------------------

    def _setup_settlement(
        self, state: GameState, legal_actions: list[Action]
    ) -> Action:
        candidates = [a for a in legal_actions if isinstance(a, PlaceSettlement)]
        return max(candidates, key=lambda a: _vertex_score(state.board, a.vertex_id))

    def _setup_road(self, state: GameState, legal_actions: list[Action]) -> Action:
        origin = state.last_settlement_vertex
        assert origin is not None

        def target_score(edge_id: int) -> float:
            v1, v2 = GEOMETRY.edge_vertices[edge_id]
            other = v2 if v1 == origin else v1
            return _vertex_score(state.board, other)

        candidates = [a for a in legal_actions if isinstance(a, PlaceRoad)]
        return max(candidates, key=lambda a: target_score(a.edge_id))

    # -- Discard / robber / steal ----------------------------------------

    def _discard(
        self, state: GameState, legal_actions: list[Action], player_idx: int
    ) -> Action:
        player = state.players[player_idx]
        candidates = [a for a in legal_actions if isinstance(a, Discard)]
        # Every candidate discards the same total; recover it from any one.
        required = sum(candidates[0].resources.values())

        hand = dict(player.resources)
        target: dict[Resource, int] = {}
        remaining = required
        while remaining > 0:
            r = max(hand, key=lambda k: hand[k])
            if hand[r] <= 0:
                break
            take = min(hand[r], remaining)
            target[r] = target.get(r, 0) + take
            hand[r] -= take
            remaining -= take

        for a in candidates:
            if a.resources == target:
                return a
        # Fallback: shouldn't happen (the greedy bundle is always a member
        # of the pre-enumerated combinations), but never crash on it.
        return candidates[0]

    def _move_robber(
        self, state: GameState, legal_actions: list[Action], player_idx: int
    ) -> Action:
        own_vertices = (
            state.players[player_idx].settlement_vertices
            | state.players[player_idx].city_vertices
        )

        def blocked_value(hex_id: Cube) -> float:
            if any(v in own_vertices for v in GEOMETRY.hex_vertices[hex_id]):
                return -1.0
            token = state.board.tokens.get(hex_id)
            if token is None:
                return 0.0
            value = 0.0
            for v in GEOMETRY.hex_vertices[hex_id]:
                owner = _vertex_owner(state, v)
                if owner is None or owner == player_idx:
                    continue
                is_city = v in state.players[owner].city_vertices
                value += pips(token) * (2 if is_city else 1)
            return value

        candidates = [a for a in legal_actions if isinstance(a, MoveRobber)]
        return max(candidates, key=lambda a: blocked_value(a.hex_id))

    def _steal(self, state: GameState, legal_actions: list[Action]) -> Action:
        candidates = [a for a in legal_actions if isinstance(a, StealFrom)]
        return max(
            candidates,
            key=lambda a: state.players[a.player_idx].resource_card_count(),
        )

    # -- Roll / main ------------------------------------------------------

    def _robber_on_own_producing_hex(self, state: GameState, player_idx: int) -> bool:
        player = state.players[player_idx]
        owned = player.settlement_vertices | player.city_vertices
        return any(
            v in owned for v in GEOMETRY.hex_vertices.get(state.board.robber_hex, ())
        )

    def _roll_phase(
        self, state: GameState, legal_actions: list[Action], player_idx: int
    ) -> Action:
        for a in legal_actions:
            if isinstance(a, PlayVictoryPoint):
                return a
        if self._robber_on_own_producing_hex(state, player_idx):
            for a in legal_actions:
                if isinstance(a, PlayKnight):
                    return a
        for a in legal_actions:
            if isinstance(a, RollDice):
                return a
        return legal_actions[0]

    def _has_available_site(
        self,
        state: GameState,
        player_idx: int,
        action_cls: type,
        cost: dict[Resource, int],
    ) -> bool:
        """Whether a legal placement site exists, ignoring affordability --
        determined by temporarily granting the cost on a state *copy* and
        re-checking the real ``legal_actions()``, never by calling a private
        site-enumeration helper directly."""
        trial = state.copy()
        player = trial.players[player_idx]
        for r, c in cost.items():
            player.resources[r] = max(player.resources[r], c)
        game = CatanGame(num_players=len(trial.players))
        return any(isinstance(a, action_cls) for a in game.legal_actions(trial))

    def _preferred_build_cost(
        self, state: GameState, player_idx: int
    ) -> dict[Resource, int] | None:
        if self._has_available_site(state, player_idx, PlaceCity, CITY_COST):
            return CITY_COST
        if self._has_available_site(
            state, player_idx, PlaceSettlement, SETTLEMENT_COST
        ):
            return SETTLEMENT_COST
        if self._has_available_site(state, player_idx, PlaceRoad, ROAD_COST):
            return ROAD_COST
        return None

    def _trade_enables_a_build(
        self, state: GameState, player_idx: int, action: Action
    ) -> bool:
        trial = state.copy()
        game = CatanGame(num_players=len(trial.players))
        game.apply_action(trial, action)
        trial_legal = game.legal_actions(trial)
        return any(
            isinstance(a, PlaceCity | PlaceSettlement | PlaceRoad) for a in trial_legal
        )

    def _main_phase(
        self, state: GameState, legal_actions: list[Action], player_idx: int
    ) -> Action:
        player = state.players[player_idx]

        for a in legal_actions:
            if isinstance(a, PlayVictoryPoint):
                return a

        city_actions = [a for a in legal_actions if isinstance(a, PlaceCity)]
        if city_actions:
            return max(
                city_actions, key=lambda a: _vertex_score(state.board, a.vertex_id)
            )

        settlement_actions = [
            a for a in legal_actions if isinstance(a, PlaceSettlement)
        ]
        if settlement_actions:
            return max(
                settlement_actions,
                key=lambda a: _vertex_score(state.board, a.vertex_id),
            )

        for a in legal_actions:
            if isinstance(a, BuyDevCard):
                return a

        if self._robber_on_own_producing_hex(state, player_idx):
            for a in legal_actions:
                if isinstance(a, PlayKnight):
                    return a

        target_cost = self._preferred_build_cost(state, player_idx)
        if target_cost is not None:
            monopoly_actions = [a for a in legal_actions if isinstance(a, PlayMonopoly)]
            if monopoly_actions:
                best = max(
                    monopoly_actions,
                    key=lambda a: _shortfall(player, target_cost, a.resource),
                )
                if _shortfall(player, target_cost, best.resource) > 0:
                    return best

            yop_actions = [a for a in legal_actions if isinstance(a, PlayYearOfPlenty)]
            if yop_actions:

                def yop_score(a: PlayYearOfPlenty) -> int:
                    return _shortfall(player, target_cost, a.resource_1) + _shortfall(
                        player, target_cost, a.resource_2
                    )

                best_yop = max(yop_actions, key=yop_score)
                if yop_score(best_yop) > 0:
                    return best_yop

            trade_actions = [
                a for a in legal_actions if isinstance(a, TradeBank | TradePort)
            ]
            for a in trade_actions:
                if self._trade_enables_a_build(state, player_idx, a):
                    return a

        road_actions = [a for a in legal_actions if isinstance(a, PlaceRoad)]
        if road_actions:

            def road_score(edge_id: int) -> float:
                v1, v2 = GEOMETRY.edge_vertices[edge_id]
                candidates = [
                    _vertex_score(state.board, v)
                    for v in (v1, v2)
                    if _vertex_owner(state, v) is None
                ]
                if not candidates:
                    return (
                        _vertex_score(state.board, v1) + _vertex_score(state.board, v2)
                    ) / 2
                return max(candidates)

            return max(road_actions, key=lambda a: road_score(a.edge_id))

        for a in legal_actions:
            if isinstance(a, EndTurn):
                return a

        # Never proposes a trade -- filtered out, per the scope boundary.
        for a in legal_actions:
            if isinstance(a, ProposeTrade):
                continue
            return a
        raise AssertionError("no non-ProposeTrade legal action available in MAIN")
