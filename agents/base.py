"""The ``Agent`` protocol.

Extracted to ``gamekit.agent.Agent[StateT, ActionT]`` -- see
``docs/shared-ml-package.md``. ``CatanAgent`` is the parameterised alias this
codebase's annotations use; ``isinstance`` checks stay against the bare
``Agent`` re-exported below (a subscripted generic Protocol raises
``TypeError`` from ``isinstance``), which is exactly what
``tests/test_agents.py`` already does.
"""

from __future__ import annotations

from gamekit import Agent as Agent

from engine.actions import Action
from engine.state import GameState

type CatanAgent = Agent[GameState, Action]
