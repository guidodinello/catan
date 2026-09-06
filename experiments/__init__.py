"""Phase 2: Monte Carlo analysis on top of the engine.

Consumes ``engine`` only through its public rollout API (``reset`` /
``legal_actions`` / ``apply_action`` / ``is_terminal`` / ``winner`` /
``acting_player`` / ``victory_points``). No RL/agent-protocol material lives
here -- that is Phase 3+.
"""
