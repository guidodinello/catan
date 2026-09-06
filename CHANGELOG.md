# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Phase 2: Monte Carlo analysis layer (`experiments/`) -- a rollout driver,
  local MC statistics utilities, analytic dice/production tables, starting-
  placement experiments (fixed-board vertex ranking + feature-bucketed
  random-board win rates), and four resource/VP probability tables, plus
  tests and committed full-scale results (`experiments/results/`).
- Initial project bootstrap with dev-standards baseline

### Fixed
- `_bank_trade_actions`/`_port_trade_actions` could list a `TradeBank`/
  `TradePort` as legal without the bank actually holding the requested
  resource, crashing `apply_action` on an action `legal_actions()` itself
  had offered. Found while building the Phase 2 rollout harness.

## [0.1.0] - YYYY-MM-DD

### Added
- Project scaffolded
