"""``gamekit.rl.protocols.TurnBasedGame`` adapter over ``engine.game.CatanGame``.

``CatanGame`` already exposes ``reset``/``legal_actions``/``apply_action``/
``is_terminal`` with matching names and signatures. The only member the
protocol wants that it lacks is ``acting_player``, which exists as a free
function in ``engine.state`` because the seat that must decide next is not
always the turn seat (the robber-discard queue and trade responses both
interrupt). Binding it here is the whole adapter -- no engine change.
"""

from __future__ import annotations

from engine.actions import Action
from engine.game import CatanGame
from engine.state import GameState, acting_player


class CatanTurnBasedGame:
    """Satisfies ``gamekit.rl.protocols.TurnBasedGame[GameState, Action]``.

    ``apply_action`` mutates ``state`` in place and returns that same object,
    inherited from ``CatanGame``. That is load-bearing, not incidental:
    ``gamekit.rl.driver.advance_until_learner`` rebinds a local and returns an
    ``int``, so an engine that returned a *new* state would silently drop
    every opponent action.
    """

    def __init__(self, num_players: int = 4) -> None:
        self._game = CatanGame(num_players=num_players)
        self.num_players = num_players

    def reset(self, seed: int | None = None) -> GameState:
        return self._game.reset(seed=seed)

    def legal_actions(self, state: GameState) -> list[Action]:
        return self._game.legal_actions(state)

    def apply_action(self, state: GameState, action: Action) -> GameState:
        return self._game.apply_action(state, action)

    def is_terminal(self, state: GameState) -> bool:
        return self._game.is_terminal(state)

    def acting_player(self, state: GameState) -> int:
        return acting_player(state)
