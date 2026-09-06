"""``HumanAgent`` -- wraps the interactive prompting from Phase 1's
``cli.py`` in the ``Agent`` protocol. The one agent allowed to return a real
``ProposeTrade``: ``_prompt_propose_trade`` already constructs a full bundle,
so there is no sentinel to resolve.
"""

from __future__ import annotations

from collections import defaultdict

from engine.actions import Action, ProposeTrade
from engine.board import Resource
from engine.game import MAX_TRADE_OFFER_SIDE
from engine.state import GameState, PlayerState


def _format_action(action: Action) -> str:
    name = type(action).__name__
    fields = getattr(action, "__dataclass_fields__", {})
    if not fields:
        return name
    parts = [f"{f}={getattr(action, f)!r}" for f in fields]
    return f"{name}({', '.join(parts)})"


def _prompt_int(prompt: str, valid: range | list[int]) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            choice = int(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if choice in valid:
            return choice
        print("Out of range, try again.")


def _prompt_resource_bundle(side_label: str, hand: dict | None) -> dict:
    """Prompt for a multi-resource bundle, one count per resource type.

    ``hand`` caps each resource by what the player actually holds (the give
    side); ``None`` means uncapped except by the per-side card total (the
    receive side -- the proposer need not know what any other player holds).
    """
    print(f"  {side_label} (enter a count for each resource, 0 to skip):")
    bundle: dict = {}
    total = 0
    for r in Resource:
        remaining = MAX_TRADE_OFFER_SIDE - total
        if remaining <= 0:
            break
        cap = remaining if hand is None else min(hand.get(r, 0), remaining)
        if cap <= 0:
            continue
        count = _prompt_int(f"    {r.name.title()} (0-{cap}): ", range(cap + 1))
        if count > 0:
            bundle[r] = count
            total += count
    return bundle


def _prompt_propose_trade(player: PlayerState) -> ProposeTrade:
    print(f"\nPropose a trade (up to {MAX_TRADE_OFFER_SIDE} cards per side):")
    while True:
        give = _prompt_resource_bundle("You give", player.resources)
        receive = _prompt_resource_bundle("You receive", None)
        if not give or not receive:
            print("Both sides must be non-empty -- try again.")
            continue
        if set(give) & set(receive):
            print("Cannot trade like-for-like resources -- try again.")
            continue
        return ProposeTrade(give=give, receive=receive)


class HumanAgent:
    def __init__(self, name: str = "human") -> None:
        self.name = name

    def reset(self) -> None:
        pass

    def choose_action(
        self, state: GameState, legal_actions: list[Action], player_idx: int
    ) -> Action:
        by_type: dict[type, list[Action]] = defaultdict(list)
        for a in legal_actions:
            by_type[type(a)].append(a)
        types = sorted(by_type, key=lambda t: t.__name__)

        print("\nAvailable actions:")
        for i, t in enumerate(types):
            count = len(by_type[t])
            suffix = f" ({count} options)" if count > 1 else ""
            print(f"  {i}: {t.__name__}{suffix}")

        idx = _prompt_int("Choose an action type: ", range(len(types)))
        action_type = types[idx]
        if action_type is ProposeTrade:
            return _prompt_propose_trade(state.players[player_idx])

        options = by_type[action_type]
        if len(options) == 1:
            return options[0]

        print("Choose a specific action:")
        for i, a in enumerate(options):
            print(f"  {i}: {_format_action(a)}")
        choice = _prompt_int("Choice: ", range(len(options)))
        return options[choice]
