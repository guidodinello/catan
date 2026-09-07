# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Phase 3: agents layer (`agents/`) -- the `Agent` protocol, `RandomAgent`/
  `StratifiedRandomAgent` (moved verbatim from `experiments/rollout.py`),
  `HeuristicAgent` (fixed build-order + site-scoring policy, each rule cited
  to a Phase 2 result), and `HumanAgent` (moved from `cli.py`). Rewrote
  `experiments/rollout.py` around a module-level `AgentFactory` so seats can
  hold heterogeneous agents, with a bit-identity regression test proving
  every Phase 2 result stays replayable. Added `experiments/benchmark.py`, a
  benchmark harness with a mode registry and mandatory seat rotation; all
  three acceptance gates pass at full scale, committed under
  `experiments/results/benchmark_*.json` with a hand-written summary.
  `cli.py` now builds a human/heuristic agent lineup instead of a fixed
  human-only loop.
- Phase 2: Monte Carlo analysis layer (`experiments/`) -- a rollout driver,
  local MC statistics utilities, analytic dice/production tables, starting-
  placement experiments (fixed-board vertex ranking + feature-bucketed
  random-board win rates), and four resource/VP probability tables, plus
  tests and committed full-scale results (`experiments/results/`).
- Initial project bootstrap with dev-standards baseline

### Fixed
- Hidden Victory Point dev cards now trigger the automatic win the instant a
  player's *true* total (public score plus every VP card in `dev_hand`,
  revealed or not) reaches 10 -- matching the real rule, where revealing a VP
  card is proof, not a distinct timed action. Previously the engine's win
  check (`engine.game._check_win`) only ever counted `revealed_vp_cards`, so a
  player at 9 public points plus a hidden VP card had to separately submit
  `PlayVictoryPoint` before winning; buying a winning VP card outright didn't
  trigger a win check at all. New `engine.game.true_victory_points` (engine-
  internal, not re-exported) drives `_check_win`; `victory_points` keeps its
  exact prior public-tally semantics and every existing caller/serializer is
  unchanged. On a win, the winner's VP cards are revealed
  (`dev_hand` -> `revealed_vp_cards`) in the same step the game ends, so no
  `server/serialize.py` change was needed -- an opponent's hand still reads
  fully redacted right up until the instant the game actually ends. Raised
  during play; see `docs/backlog.md`.
- `_bank_trade_actions`/`_port_trade_actions` could list a `TradeBank`/
  `TradePort` as legal without the bank actually holding the requested
  resource, crashing `apply_action` on an action `legal_actions()` itself
  had offered. Found while building the Phase 2 rollout harness.

## [0.1.0] - YYYY-MM-DD

### Added
- Project scaffolded
