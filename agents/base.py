"""The ``Agent`` protocol.

Method names and argument order are deliberately identical to truco-py's
``agents/base.py`` (``~/Desktop/fac/2026/mmo/truco-py/agents/base.py``): Phase
4's extraction plan lists that shape as already generic, so keeping the call
surface identical means the extraction is a rename-and-parameterise, not a
redesign. ``name`` is an addition, needed for result stamping -- a widening,
not a divergence.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from engine.actions import Action
from engine.state import GameState


@runtime_checkable
class Agent(Protocol):
    name: str

    def choose_action(
        self, state: GameState, legal_actions: list[Action], player_idx: int
    ) -> Action: ...

    def reset(self) -> None: ...
