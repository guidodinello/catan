"""Rule under test: ``CatanEnv`` is a well-behaved gymnasium env over the real
engine, and its composition buffer never lets an illegal action reach
``apply_action``.

Nothing else in the repo exercises this adapter -- ``experiments/benchmark.py``
drives the engine directly and never imports ``rl/`` -- so a regression here
would otherwise surface only as a training run that quietly learns nothing.
That is the same gap truco-py's ``tests/test_env.py`` was written to cover.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

pytest.importorskip("gymnasium")

import numpy as np  # noqa: E402
from gamekit.rl.protocols import RewardFn  # noqa: E402
from gamekit.rl.selfplay import OpponentPool  # noqa: E402
from gymnasium import spaces  # noqa: E402

from agents.heuristic import HeuristicAgent  # noqa: E402
from agents.random_agent import RandomAgent  # noqa: E402
from engine.actions import Action  # noqa: E402
from engine.state import GameState, Phase  # noqa: E402
from rl.action_space import N_ATOMS, TRADE_COUNTER, TRADE_PROPOSE  # noqa: E402
from rl.adapter import CatanTurnBasedGame  # noqa: E402
from rl.encoder import OBS_DIM  # noqa: E402
from rl.env import CatanEnv, make_env  # noqa: E402
from rl.reward import ShapedReward, SparseReward  # noqa: E402


def _sample_legal(env: CatanEnv, rng: random.Random) -> np.int64:
    mask = env.action_masks()
    legal = np.flatnonzero(mask)
    assert legal.size > 0, "env offered no legal atom on a live episode"
    return np.int64(rng.choice(legal.tolist()))


def _play_episode(env: CatanEnv, rng: random.Random, max_steps: int = 20_000) -> int:
    env.reset(seed=rng.randint(0, 2**31 - 1))
    for step in range(max_steps):
        _, _, terminated, truncated, _ = env.step(_sample_legal(env, rng))
        if terminated or truncated:
            return step + 1
    raise AssertionError("episode did not finish within max_steps")


def test_reset_returns_a_correctly_shaped_observation_and_empty_info() -> None:
    env = make_env(seed=1)
    obs, info = env.reset(seed=1)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert info == {}


def test_action_and_observation_spaces_match_the_declared_widths() -> None:
    env = make_env(seed=1)
    assert isinstance(env.action_space, spaces.Discrete)
    assert int(env.action_space.n) == N_ATOMS
    assert env.observation_space.shape == (OBS_DIM,)


def test_action_masks_are_boolean_and_never_empty_on_a_live_episode() -> None:
    env = make_env(seed=2)
    env.reset(seed=2)
    rng = random.Random(2)
    for _ in range(200):
        mask = env.action_masks()
        assert mask.shape == (N_ATOMS,)
        assert mask.dtype == np.bool_
        assert mask.sum() > 0
        _, _, terminated, truncated, _ = env.step(_sample_legal(env, rng))
        if terminated or truncated:
            break


def test_step_returns_the_gymnasium_five_tuple_with_the_state_in_info() -> None:
    env = make_env(seed=3)
    env.reset(seed=3)
    obs, reward, terminated, truncated, info = env.step(
        _sample_legal(env, random.Random(3))
    )
    assert obs.shape == (OBS_DIM,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info["state"], GameState)


def test_a_full_episode_terminates_and_its_final_observation_is_all_zeros() -> None:
    env = make_env(seed=4)
    rng = random.Random(4)
    env.reset(seed=4)
    while True:
        obs, _, terminated, truncated, _ = env.step(_sample_legal(env, rng))
        if terminated or truncated:
            break
    assert terminated, "random play should reach a winner well inside the budget"
    assert np.array_equal(obs, np.zeros(OBS_DIM, dtype=np.float32))


def test_episodes_finish_from_several_seeds() -> None:
    env = make_env(seed=5)
    rng = random.Random(5)
    for _ in range(3):
        assert _play_episode(env, rng) > 0


def test_a_mid_composition_step_yields_no_reward_and_does_not_end_the_episode() -> None:
    """Filling in half an action applies nothing to the engine, so it is not a
    transition -- otherwise ``ShapedReward``'s potential term would accrue
    ``(gamma - 1) * phi`` per keystroke instead of telescoping."""
    env = make_env(seed=6, reward=ShapedReward())
    rng = random.Random(6)
    env.reset(seed=6)
    saw_partial = False
    for _ in range(4000):
        before = env.state
        _, reward, terminated, truncated, _ = env.step(_sample_legal(env, rng))
        if env._composer.in_progress:
            saw_partial = True
            assert reward == 0.0
            assert not terminated and not truncated
            assert env.state is before
        if terminated or truncated:
            env.reset(seed=rng.randint(0, 2**31 - 1))
    assert saw_partial, "never exercised a multi-atom action"


def test_an_illegal_atom_is_coerced_by_default_and_raised_on_request() -> None:
    env = make_env(seed=7)
    env.reset(seed=7)
    illegal = int(np.flatnonzero(~env.action_masks())[0])
    env.step(np.int64(illegal))  # coerced, no exception

    strict = make_env(seed=7, on_illegal="raise")
    strict.reset(seed=7)
    illegal = int(np.flatnonzero(~strict.action_masks())[0])
    with pytest.raises(ValueError, match="illegal atom"):
        strict.step(np.int64(illegal))


def test_the_reserved_trade_atoms_are_never_offered() -> None:
    env = make_env(seed=8)
    rng = random.Random(8)
    env.reset(seed=8)
    for _ in range(1500):
        mask = env.action_masks()
        assert not mask[TRADE_PROPOSE]
        assert not mask[TRADE_COUNTER]
        _, _, terminated, truncated, _ = env.step(_sample_legal(env, rng))
        if terminated or truncated:
            env.reset(seed=rng.randint(0, 2**31 - 1))


def test_the_learner_only_ever_acts_on_its_own_seat() -> None:
    from engine.state import acting_player

    env = make_env(seed=9)
    rng = random.Random(9)
    env.reset(seed=9)
    for _ in range(1000):
        state = env.state
        assert state is not None
        assert acting_player(state) == env.learner_seat
        _, _, terminated, truncated, _ = env.step(_sample_legal(env, rng))
        if terminated or truncated:
            env.reset(seed=rng.randint(0, 2**31 - 1))


def test_the_learner_is_woken_for_out_of_turn_discards_and_trade_responses() -> None:
    """``acting_player`` is the seam, not ``current_player`` -- the discard
    queue and trade responses both interrupt the turn player."""
    env = make_env(seed=10)
    rng = random.Random(10)
    seen: set[Phase] = set()
    for _ in range(6):
        env.reset(seed=rng.randint(0, 2**31 - 1))
        for _ in range(3000):
            state = env.state
            assert state is not None
            seen.add(state.phase)
            _, _, terminated, truncated, _ = env.step(_sample_legal(env, rng))
            if terminated or truncated:
                break
        if {Phase.DISCARD, Phase.AWAIT_TRADE_RESPONSE} <= seen:
            break
    assert Phase.DISCARD in seen
    assert Phase.AWAIT_TRADE_RESPONSE in seen


def test_randomize_seat_off_pins_the_learner_to_seat_zero() -> None:
    env = make_env(seed=11, randomize_seat=False)
    for seed in range(5):
        env.reset(seed=seed)
        assert env.learner_seat == 0


def test_randomize_seat_on_visits_every_seat() -> None:
    env = make_env(seed=12)
    seats = set()
    for seed in range(40):
        env.reset(seed=seed)
        seats.add(env.learner_seat)
    assert seats == {0, 1, 2, 3}


def test_every_opponent_seat_is_actually_driven() -> None:
    class CountingAgent:
        def __init__(self, inner: RandomAgent) -> None:
            self.inner = inner
            self.name = "counting"
            self.calls = 0

        def choose_action(
            self, state: GameState, legal_actions: list[Action], player_idx: int
        ) -> Action:
            self.calls += 1
            return self.inner.choose_action(state, legal_actions, player_idx)

        def reset(self) -> None:
            self.inner.reset()

    agents = [CountingAgent(RandomAgent(rng=random.Random(i))) for i in range(3)]
    env = CatanEnv(num_players=4, agents=agents, seed=13)
    _play_episode(env, random.Random(13))
    assert all(a.calls > 0 for a in agents)


def test_the_constructor_rejects_contradictory_or_incomplete_wiring() -> None:
    agents = [RandomAgent(rng=random.Random(i)) for i in range(3)]
    with pytest.raises(ValueError, match="exactly one of"):
        CatanEnv(num_players=4)
    with pytest.raises(ValueError, match="num_players - 1"):
        CatanEnv(num_players=4, agents=agents[:2])
    with pytest.raises(ValueError, match="on_illegal"):
        CatanEnv(num_players=4, agents=agents, on_illegal="explode")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="3 or 4 players"):
        CatanEnv(num_players=2, agents=agents)


def test_three_player_games_work_too() -> None:
    env = make_env(num_players=3, seed=14)
    assert _play_episode(env, random.Random(14)) > 0


def test_heuristic_opponents_drive_an_episode_to_completion() -> None:
    env = CatanEnv(num_players=4, agents=[HeuristicAgent() for _ in range(3)], seed=15)
    assert _play_episode(env, random.Random(15)) > 0


def test_self_play_mode_falls_back_to_the_baseline_with_an_empty_pool(
    tmp_path: Path,
) -> None:
    pool: OpponentPool[GameState, Action] = OpponentPool(
        tmp_path,
        "catan_selfplay_*.zip",
        load_opponent=lambda path: HeuristicAgent(),
        baseline_factory=HeuristicAgent,
        baseline_mix=0.2,
    )
    env = CatanEnv(num_players=4, opponent_pool=pool, seed=16)
    assert _play_episode(env, random.Random(16)) > 0


def test_sparse_reward_pays_exactly_once_at_the_end_of_the_episode() -> None:
    env = make_env(seed=17, reward=SparseReward())
    rng = random.Random(17)
    env.reset(seed=17)
    rewards = []
    while True:
        _, reward, terminated, truncated, _ = env.step(_sample_legal(env, rng))
        rewards.append(reward)
        if terminated or truncated:
            break
    assert all(r == 0.0 for r in rewards[:-1])
    assert rewards[-1] in (1.0, -1.0)


def test_shaped_reward_telescopes_to_the_sparse_outcome_over_an_episode() -> None:
    """Potential-based shaping with phi(terminal) = 0 must sum to the sparse
    return plus ``-phi(start)`` at gamma = 1 -- the property that makes it
    policy-invariant."""
    reward = ShapedReward(gamma=1.0)
    env = make_env(seed=18, reward=reward)
    rng = random.Random(18)
    env.reset(seed=18)
    start_potential = reward._previous_potential
    total = 0.0
    while True:
        _, r, terminated, truncated, _ = env.step(_sample_legal(env, rng))
        total += r
        if terminated or truncated:
            break
    outcome = (
        1.0
        if terminated
        and env.state is not None
        and (env.state.winner == env.learner_seat)
        else -1.0
    )
    assert total == pytest.approx(outcome - start_potential)


def test_the_step_budget_truncates_instead_of_running_forever() -> None:
    env = make_env(seed=19, step_budget=50)
    rng = random.Random(19)
    env.reset(seed=19)
    for _ in range(5000):
        _, _, terminated, truncated, _ = env.step(_sample_legal(env, rng))
        if terminated or truncated:
            break
    assert truncated


def test_step_before_reset_and_after_termination_both_fail_loudly() -> None:
    env = make_env(seed=20)
    with pytest.raises(RuntimeError, match="call reset"):
        env.step(np.int64(0))
    rng = random.Random(20)
    _play_episode(env, rng)
    with pytest.raises(RuntimeError, match="already-terminal"):
        env.step(np.int64(0))


def test_the_adapter_mutates_state_in_place_as_the_driver_requires() -> None:
    """``advance_until_learner`` rebinds a local and returns an ``int``; an
    engine that returned a fresh state would silently drop opponent actions."""
    game = CatanTurnBasedGame(num_players=4)
    state = game.reset(seed=21)
    returned = game.apply_action(state, game.legal_actions(state)[0])
    assert returned is state


def test_the_reward_functions_satisfy_the_gamekit_protocol() -> None:
    assert isinstance(SparseReward(), RewardFn)
    assert isinstance(ShapedReward(), RewardFn)
