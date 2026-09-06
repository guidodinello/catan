"""``RandomAgent`` and ``StratifiedRandomAgent`` -- the class forms of Phase
2's ``make_random_policy``/``make_stratified_policy``. Logic moves verbatim;
behaviour is unchanged, which is what makes the bit-identity regression test
in ``tests/test_agents.py`` meaningful.

**RNG call order and count must be preserved exactly: no filtering, no
sorting, no reordering of ``legal_actions``.** ``StratifiedRandomAgent`` draws
``rng.choice(list(by_type.keys()))`` -- which depends on dict insertion
order, i.e. on the order ``legal_actions`` was handed in -- then
``rng.choice(by_type[action_type])``, then whatever the sentinel resolver
draws. Bit-identity dies silently if a filter-the-sentinel-out idiom (as used
by ``HeuristicAgent``) leaks in here, or if ``legal_actions`` is sorted before
sampling.
"""

from __future__ import annotations

import random
from collections import defaultdict

from engine.actions import Action, ProposeTrade
from engine.game import MAX_TRADE_OFFER_SIDE
from engine.state import GameState


def build_random_trade_offer(rng: random.Random, state: GameState) -> ProposeTrade:
    """Construct a real multi-resource bundle for the ``ProposeTrade``
    sentinel: a give side drawn from the proposer's actual hand (1..
    ``MAX_TRADE_OFFER_SIDE`` cards across however many resource types the
    random draw picks), and a receive side of any resource types not
    already on the give side.

    This is the single canonical sentinel resolver for the repo --
    ``tests/test_engine.py`` imports the agent that uses it rather than
    keeping its own copy.
    """
    player = state.players[state.current_player]
    available = [r for r in player.resources if player.resources[r] > 0]
    rng.shuffle(available)
    give: dict = {}
    remaining = min(MAX_TRADE_OFFER_SIDE, player.resource_card_count())
    for r in available:
        if remaining <= 0:
            break
        take = rng.randint(1, min(player.resources[r], remaining))
        give[r] = take
        remaining -= take

    receive_pool = [r for r in player.resources if r not in give]
    rng.shuffle(receive_pool)
    receive: dict = {}
    remaining = MAX_TRADE_OFFER_SIDE
    for r in receive_pool:
        if remaining <= 0:
            break
        take = rng.randint(1, remaining)
        receive[r] = take
        remaining -= take
        if rng.random() < 0.5:
            break
    return ProposeTrade(give=give, receive=receive)


def _resolve_trade_sentinel(
    rng: random.Random, state: GameState, action: Action
) -> Action:
    """Resolve the ``ProposeTrade`` sentinel (empty give/receive -- see
    ``engine/game.py``'s ``_propose_trade_actions`` docstring) into a real
    multi-resource bundle, mirroring what an interactive builder (the CLI)
    or a test driver would construct.
    """
    if not (
        isinstance(action, ProposeTrade) and not action.give and not action.receive
    ):
        return action
    return build_random_trade_offer(rng, state)


class RandomAgent:
    """Flat-uniform policy: sample one action uniformly from all legal
    actions. TradeBank/TradePort/PlaceSettlement etc. dominate whenever they
    have many members for a state -- see ``StratifiedRandomAgent`` for an
    agent that avoids this."""

    def __init__(self, rng: random.Random, name: str = "random") -> None:
        self.rng = rng
        self.name = name

    def choose_action(
        self, state: GameState, legal_actions: list[Action], player_idx: int
    ) -> Action:
        return _resolve_trade_sentinel(self.rng, state, self.rng.choice(legal_actions))

    def reset(self) -> None:
        pass


class StratifiedRandomAgent:
    """Sample the action *type* uniformly first, then a member of that type.

    Prevents a numerous action type (a huge TradeBank/TradePort/placement
    list for one state) from dominating every step."""

    def __init__(self, rng: random.Random, name: str = "stratified_random") -> None:
        self.rng = rng
        self.name = name

    def choose_action(
        self, state: GameState, legal_actions: list[Action], player_idx: int
    ) -> Action:
        by_type: dict[type, list[Action]] = defaultdict(list)
        for a in legal_actions:
            by_type[type(a)].append(a)
        action_type = self.rng.choice(list(by_type.keys()))
        return _resolve_trade_sentinel(
            self.rng, state, self.rng.choice(by_type[action_type])
        )

    def reset(self) -> None:
        pass
