"""Rule under test: the training entrypoint's configuration and evaluation
plumbing are correct *without* running training.

``rl/train.py`` defers its sb3-contrib and torch imports into the functions
that need them, specifically so this module can check the parts that are easy
to get quietly wrong. That is not hypothetical: the first version of
``build_parser`` read its defaults off ``TrainConfig`` the class rather than an
instance, and because ``@dataclass(slots=True)`` replaces class attributes with
slot descriptors, every default silently became a ``member_descriptor``. It
surfaced only mid-rollout as ``unsupported operand type(s) for *:
'member_descriptor' and 'float'``.

Nothing here imports torch, so it runs in the ``Tests (RL)`` CI job.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

pytest.importorskip("gymnasium")

import numpy as np  # noqa: E402

from agents.heuristic import HeuristicAgent  # noqa: E402
from agents.random_agent import (  # noqa: E402
    RandomAgent,
    StratifiedRandomAgent,
)
from experiments.benchmark import (  # noqa: E402
    MODE_LINEUPS,
    RL_MODES,
    _build_role,
    _result_name,
    run_arm,
)
from rl.evaluate import (  # noqa: E402
    WinRate,
    evaluate_winrate,
    random_predictor,
    summarize,
)
from rl.train import (  # noqa: E402
    OPPONENT_KINDS,
    TrainConfig,
    build_opponent,
    build_parser,
)


def test_train_config_defaults_are_values_not_slot_descriptors() -> None:
    """The regression test for the bug described in this module's docstring."""
    cfg = TrainConfig()
    assert isinstance(cfg.gamma, float)
    assert isinstance(cfg.steps, int)
    assert isinstance(cfg.envs, int)
    assert isinstance(cfg.ent_coef, float)
    assert isinstance(cfg.eval_every, int)
    assert isinstance(cfg.eval_episodes, int)
    assert isinstance(cfg.torch_threads, int)
    assert isinstance(cfg.label, str)
    assert isinstance(cfg.run_dir, Path)


def test_parser_defaults_are_usable_numbers() -> None:
    args = build_parser().parse_args([])
    assert args.gamma == TrainConfig().gamma
    assert args.steps == TrainConfig().steps
    assert isinstance(args.gamma, float)
    assert isinstance(args.steps, int)


def test_learning_rate_is_exposed_on_the_cli() -> None:
    """Needed for a BC fine-tune at a gentler lr than the TrainConfig default
    (e.g. 1e-4 instead of 3e-4) -- previously only settable via TrainConfig,
    not the CLI a real run actually invokes."""
    args = build_parser().parse_args(["--learning-rate", "1e-4"])
    assert args.learning_rate == pytest.approx(1e-4)


def test_bc_init_and_resume_are_mutually_exclusive() -> None:
    """A warm *policy* (--bc-init) and a resumed *run* (--resume) are
    different things -- see rl/train.py's _resume_step_bookkeeping -- so
    passing both is refused at the CLI rather than silently picking one."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--resume", "a.zip", "--bc-init", "b.zip"])


def test_bc_init_rate_defaults_to_an_unseeded_guard() -> None:
    """Absent --bc-init (a from-scratch or --resume run), RegressionGuard
    must not be seeded by a stale default rate."""
    args = build_parser().parse_args([])
    assert args.bc_init is None
    assert args.bc_init_rate == pytest.approx(-1.0)


def test_gamma_defaults_high_enough_for_catan_episode_lengths() -> None:
    """A 4p episode is ~450 learner decisions. truco-py's 0.99 would give a
    ~100-step effective horizon and bury the terminal signal."""
    assert TrainConfig().gamma >= 0.999


def test_build_opponent_covers_every_advertised_kind_and_rejects_others() -> None:
    for kind in OPPONENT_KINDS:
        agent = build_opponent(kind, random.Random(0))
        assert agent.name == kind
    with pytest.raises(ValueError, match="unknown opponent kind"):
        build_opponent("wizard", random.Random(0))


def test_opponent_vocabulary_matches_the_benchmark_harness() -> None:
    """One vocabulary of opponent names across train, benchmark and server."""
    for kind in OPPONENT_KINDS:
        assert _build_role(kind, random.Random(0)).name == kind


def test_wilson_summary_reports_the_interval_not_just_the_point_estimate() -> None:
    result = summarize(50, 200)
    assert result.rate == pytest.approx(0.25)
    assert result.ci_low < 0.25 < result.ci_high
    assert "25.0%" in str(result)
    assert "50/200" in str(result)


def test_the_gate_is_the_interval_and_not_the_point_estimate() -> None:
    """A 27% point estimate at n=200 says nothing about a 25% null."""
    noisy = summarize(54, 200)  # 27%, interval straddles 25%
    assert noisy.rate > 0.25
    assert not noisy.beats(0.25)

    convincing = summarize(2800, 4000)  # 70% at n=4000
    assert convincing.beats(0.25)


def test_summarize_refuses_an_empty_evaluation_rather_than_reporting_zero() -> None:
    """A 0-episode eval is a bug in the caller, not a 0% win rate."""
    with pytest.raises(ValueError, match="n must be positive"):
        summarize(0, 0)


def test_uniform_atom_play_matches_stratified_random_not_flat_random() -> None:
    """Sampling uniformly over *atoms* is not the same policy as sampling
    uniformly over *actions*, and the gap matters for reading the gate.

    An atom draw weights action *kinds* roughly evenly instead of weighting
    each concrete instance evenly, which is what ``StratifiedRandomAgent``
    does deliberately. So an untrained atom policy already beats three flat
    ``RandomAgent``s -- measured 32.3% at n=300, above the 25% seat share --
    while sitting right at the seat share against three stratified agents.

    The consequence: "beats 25%" is a weak statement about a policy on this
    action space. The untrained atom policy is the honest reference.
    """
    versus_flat = evaluate_winrate(
        random_predictor(seed=3),
        opponent_factory=lambda rng: RandomAgent(rng),
        num_players=4,
        n_episodes=150,
        seed=3,
    )
    versus_stratified = evaluate_winrate(
        random_predictor(seed=3),
        opponent_factory=lambda rng: StratifiedRandomAgent(rng),
        num_players=4,
        n_episodes=150,
        seed=3,
    )
    assert isinstance(versus_flat, WinRate)
    assert versus_flat.rate > versus_stratified.rate
    assert versus_stratified.ci_low < 0.25 < versus_stratified.ci_high


def test_evaluate_winrate_against_heuristics_is_far_below_the_seat_share() -> None:
    """The other end of the scale: uniform play should lose badly to three
    HeuristicAgents, which win 91.9% [91.0, 92.7] against randoms."""
    result = evaluate_winrate(
        random_predictor(seed=4),
        opponent_factory=lambda rng: HeuristicAgent(),
        num_players=4,
        n_episodes=80,
        seed=4,
    )
    assert result.ci_high < 0.25


def test_a_scripted_predictor_is_actually_consulted_every_step() -> None:
    calls = 0

    def counting(obs: np.ndarray, mask: np.ndarray) -> int:
        nonlocal calls
        calls += 1
        return int(np.flatnonzero(mask)[0])

    evaluate_winrate(
        counting,
        opponent_factory=lambda rng: RandomAgent(rng),
        num_players=4,
        n_episodes=2,
        seed=5,
    )
    assert calls > 100


def test_the_rl_benchmark_modes_place_one_learner_against_the_field() -> None:
    for mode in RL_MODES:
        assert mode in MODE_LINEUPS
        for players in (3, 4):
            lineup = MODE_LINEUPS[mode](players)
            assert len(lineup) == players
            assert lineup[0] == "rl"
            assert lineup.count("rl") == 1


def test_the_rl_modes_refuse_to_run_without_a_checkpoint() -> None:
    """Failing loudly here beats silently benchmarking an untrained policy."""
    with pytest.raises(ValueError, match="requires a checkpoint"):
        run_arm(
            mode="rl_vs_random",
            num_players=4,
            n_games=4,
            engine_seed_base=1,
            driver_seed_base=1,
            workers=1,
        )
    with pytest.raises(ValueError, match="require --checkpoint"):
        _build_role("rl", random.Random(0))


def test_result_name_is_unchanged_for_every_non_rl_mode() -> None:
    """One committed file per non-RL mode always means 'the current result
    for that mode' -- unaffected by anything checkpoint-related."""
    for mode in MODE_LINEUPS:
        if mode in RL_MODES:
            continue
        assert _result_name(mode, 4, None) == f"benchmark_{mode}_p4"
        assert _result_name(mode, 4, "whatever.zip") == f"benchmark_{mode}_p4"


def test_result_name_is_unqualified_for_an_rl_mode_with_no_checkpoint() -> None:
    assert _result_name("rl_vs_random", 4, None) == "benchmark_rl_vs_random_p4"


def test_result_name_is_qualified_by_checkpoint_for_an_rl_mode() -> None:
    """The regression test for a real incident: running a second checkpoint's
    rl_vs_random benchmark silently overwrote PR #18's committed 81.75%
    result, because the mode name alone doesn't say which checkpoint
    produced it."""
    name = _result_name(
        "rl_vs_random", 4, "rl_runs/selfplay/catan_selfplay_v2_500000.zip"
    )
    assert name == "benchmark_rl_vs_random_p4_catan_selfplay_v2_500000"


def test_result_name_never_collides_across_two_different_checkpoints() -> None:
    a = _result_name("rl_vs_heuristic", 4, "ckpt_1000000.zip")
    b = _result_name("rl_vs_heuristic", 4, "ckpt_5000000.zip")
    assert a != b


def test_the_non_rl_modes_still_build_without_a_checkpoint() -> None:
    """The added parameter must not have broken the Phase 3 arms."""
    from experiments.benchmark import benchmark_agent_factory

    agents = benchmark_agent_factory("heuristic_vs_random", None, 4, 1, 1)
    assert len(agents) == 4
    assert sorted(a.name for a in agents) == [
        "heuristic",
        "random",
        "random",
        "random",
    ]
