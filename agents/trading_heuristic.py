"""``TradingHeuristicAgent`` -- ``HeuristicAgent`` plus domestic trading.

Catan #28's prerequisite: every existing bot never proposes a trade and always
rejects offers, so nothing in the repo ever put a trade in front of the RL
agent. This agent is the cheapest opponent that does.

It **delegates every decision to a ``HeuristicAgent``** except two, and does
not modify ``HeuristicAgent`` (a live arm of experiment 007):

* **Proposing.** In MAIN, when the heuristic would just ``EndTurn``, it may
  instead propose a trade toward its next build target: ask for 1 card of the
  resource it is most short of, offering 1 (then 2) cards of the resource it
  has the most surplus of. Only when it is at most ``MAX_SHORTFALL_TO_PROPOSE``
  cards from affording the target, at most ``MAX_PROPOSALS_PER_TURN`` times a
  turn, and never the same bundle twice in a turn. The engine has no
  per-turn proposal limit (only 4 cards per side), so this bound lives here.
* **Responding.** Accept an offer (fresh, or a counter to its own proposal)
  iff it strictly reduces its shortfall to the next build target, it does not
  pay more cards than it receives, and the proposer is not within reach of
  winning (public VP < ``LEADER_VP_GUARD``). Otherwise reject. It never
  counters.

Fully deterministic: it draws from no RNG (ties break by ``Resource`` enum
order), so results depend only on the game state.
"""

from __future__ import annotations

from engine.actions import (
    AcceptTrade,
    Action,
    EndTurn,
    PlaceCity,
    PlaceRoad,
    PlaceSettlement,
    ProposeTrade,
    RejectTrade,
)
from engine.board import Resource
from engine.game import (
    CITY_COST,
    ROAD_COST,
    SETTLEMENT_COST,
    CatanGame,
    victory_points,
)
from engine.state import GameState, Phase

from .heuristic import HeuristicAgent

MAX_PROPOSALS_PER_TURN = 2
MAX_SHORTFALL_TO_PROPOSE = 2
LEADER_VP_GUARD = 8

_RESOURCE_ORDER = {r: i for i, r in enumerate(Resource)}

Bundle = tuple[tuple[Resource, int], ...]


def next_build_cost(state: GameState, player_idx: int) -> dict[Resource, int] | None:
    """The cost of the build ``HeuristicAgent`` prefers next (city, then
    settlement, then road) among those with a legal site, found through the
    public API: grant the cost on a state copy and look at ``legal_actions``."""
    for action_cls, cost in (
        (PlaceCity, CITY_COST),
        (PlaceSettlement, SETTLEMENT_COST),
        (PlaceRoad, ROAD_COST),
    ):
        trial = state.copy()
        # Evaluate as if it were this player's own MAIN phase, also when they
        # are only a trade responder (phase AWAIT_TRADE_RESPONSE offers no
        # build actions).
        trial.phase = Phase.MAIN
        trial.current_player = player_idx
        trial.trade_offer = None
        trial.trade_responders = []
        hand = trial.players[player_idx].resources
        for r, c in cost.items():
            hand[r] = max(hand[r], c)
        game = CatanGame(num_players=len(trial.players))
        if any(isinstance(a, action_cls) for a in game.legal_actions(trial)):
            return cost
    return None


def shortfall(hand: dict[Resource, int], cost: dict[Resource, int]) -> int:
    return sum(max(0, c - hand.get(r, 0)) for r, c in cost.items())


def _bundle(cards: dict[Resource, int]) -> Bundle:
    return tuple(sorted(cards.items(), key=lambda kv: _RESOURCE_ORDER[kv[0]]))


class TradingHeuristicAgent:
    def __init__(self, name: str = "trading_heuristic") -> None:
        self.name = name
        self._base = HeuristicAgent(name=name)
        self._proposals_this_turn = 0
        self._tried_this_turn: set[tuple[Bundle, Bundle]] = set()

    def reset(self) -> None:
        self._base.reset()
        self._new_turn()

    def _new_turn(self) -> None:
        self._proposals_this_turn = 0
        self._tried_this_turn = set()

    def choose_action(
        self, state: GameState, legal_actions: list[Action], player_idx: int
    ) -> Action:
        if state.phase is Phase.ROLL and player_idx == state.current_player:
            self._new_turn()
        if state.phase is Phase.AWAIT_TRADE_RESPONSE:
            return self._respond(state, legal_actions, player_idx)
        action = self._base.choose_action(state, legal_actions, player_idx)
        if state.phase is Phase.MAIN and isinstance(action, EndTurn):
            proposal = self._propose(state, legal_actions, player_idx)
            if proposal is not None:
                return proposal
        return action

    def _propose(
        self, state: GameState, legal_actions: list[Action], player_idx: int
    ) -> ProposeTrade | None:
        if self._proposals_this_turn >= MAX_PROPOSALS_PER_TURN:
            return None
        if not any(isinstance(a, ProposeTrade) for a in legal_actions):
            return None
        cost = next_build_cost(state, player_idx)
        if cost is None:
            return None
        hand = state.players[player_idx].resources
        gap = shortfall(hand, cost)
        if not 0 < gap <= MAX_SHORTFALL_TO_PROPOSE:
            return None
        short = {r: max(0, c - hand[r]) for r, c in cost.items()}
        surplus = {r: hand[r] - cost.get(r, 0) for r in Resource}
        want = max(
            (r for r in Resource if short.get(r, 0) > 0),
            key=lambda r: (short[r], -_RESOURCE_ORDER[r]),
        )
        spare = [r for r in Resource if surplus[r] > 0]
        if not spare:
            return None
        offer_r = max(spare, key=lambda r: (surplus[r], -_RESOURCE_ORDER[r]))
        for n in (1, 2):
            if surplus[offer_r] < n:
                break
            give, receive = {offer_r: n}, {want: 1}
            key = (_bundle(give), _bundle(receive))
            if key in self._tried_this_turn:
                continue
            self._tried_this_turn.add(key)
            self._proposals_this_turn += 1
            return ProposeTrade(give=give, receive=receive)
        return None

    def _respond(
        self, state: GameState, legal_actions: list[Action], player_idx: int
    ) -> Action:
        reject = next(a for a in legal_actions if isinstance(a, RejectTrade))
        offer = state.trade_offer
        accept = next((a for a in legal_actions if isinstance(a, AcceptTrade)), None)
        if offer is None or accept is None:
            return reject
        if victory_points(state, offer.proposer) >= LEADER_VP_GUARD:
            return reject
        if sum(offer.receive.values()) > sum(offer.give.values()):
            return reject
        cost = next_build_cost(state, player_idx)
        if cost is None:
            return reject
        hand = state.players[player_idx].resources
        after = dict(hand)
        for r, c in offer.give.items():
            after[r] += c
        for r, c in offer.receive.items():
            after[r] -= c
        if shortfall(after, cost) < shortfall(hand, cost):
            return accept
        return reject
