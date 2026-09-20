"""RL adapter layer for catan (Phase 5).

Nothing here changes the engine. Per README decision 10, every bound or
approximation made for the sake of a learnable action space lives in this
package; ``engine/`` continues to model the real rules exactly.

Only ``rl.env`` and ``rl.encoder`` need the ``rl`` extra (gymnasium + numpy);
``rl.adapter``, ``rl.action_space`` and ``rl.reward`` are stdlib-only. Nothing
is re-exported from this module for that reason -- a re-export would make
every submodule unimportable without the extra, the same reasoning
``gamekit.rl.__init__`` documents.
"""
