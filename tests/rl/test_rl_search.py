"""``RLSearchAgent`` against a real, tiny ``MaskablePPO`` checkpoint: the fused
forward reproduces the policy's own probabilities, a zero budget is exactly
``RLAgent``, and a searched game runs end to end. Needs the training stack;
skipped without it."""

from __future__ import annotations

import dataclasses
import random
from typing import Any, cast

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("sb3_contrib")
torch = pytest.importorskip("torch")

from agents.heuristic import HeuristicAgent  # noqa: E402
from agents.ismcts import SearchConfig, keyed_actions  # noqa: E402
from agents.rl_agent import RLAgent, _load_model  # noqa: E402
from agents.rl_search import PolicyEvaluator, RLSearchAgent  # noqa: E402
from engine.game import CatanGame, victory_points  # noqa: E402
from engine.state import Phase, acting_player  # noqa: E402
from experiments.benchmark import benchmark_agent_factory  # noqa: E402
from experiments.rollout import run_game  # noqa: E402
from rl.action_space import ActionComposer  # noqa: E402
from rl.encoder import ObservationEncoder  # noqa: E402
from rl.train import TrainConfig, build_model, build_vec_env  # noqa: E402


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> str:
    tmp = tmp_path_factory.mktemp("tiny-search")
    cfg = TrainConfig(
        envs=1, opponents="heuristic", n_steps=64, batch_size=64, n_epochs=1,
        net_arch=[16], torch_threads=1, run_dir=tmp / "run",
    )  # fmt: skip
    vec_env = build_vec_env(cfg)
    model = build_model(cfg, vec_env)
    path = tmp / "tiny.zip"
    model.save(path)
    vec_env.close()
    return str(path)


def _decision_state(seed: int = 3):  # type: ignore[no-untyped-def]
    game = CatanGame(4)
    st = game.reset(seed=seed)
    agents = [HeuristicAgent() for _ in range(4)]
    for _ in range(300):
        actor = acting_player(st)
        legal = game.legal_actions(st)
        if st.phase is Phase.MAIN and len(keyed_actions(legal, actor, 4)) > 3:
            return game, st, legal, actor
        game.apply_action(st, agents[actor].choose_action(st, legal, actor))
    raise AssertionError("no MAIN decision reached")


def test_priors_sum_to_one_and_match_the_policy_distribution(
    tiny_checkpoint: str,
) -> None:
    _, st, legal, seat = _decision_state()
    ev = PolicyEvaluator(tiny_checkpoint, 4)
    ev.ensure_board(st)
    priors, value = ev.evaluate(st, seat, legal)
    assert sum(priors.values()) == pytest.approx(1.0)
    assert set(priors) == set(keyed_actions(legal, seat, 4))

    # A single-atom action's prior is proportional to the policy's own atom
    # probability at the empty prefix (the trie only renormalises over legal).
    comp = ActionComposer()
    mask = np.array(comp.mask(legal, seat, 4), dtype=bool)
    obs = torch.as_tensor(ev.encoder.encode(st, seat)[None], dtype=torch.float32)
    dist = ev.policy.get_distribution(obs, action_masks=mask[None])
    probs = cast(Any, dist.distribution).probs[0].detach().numpy()
    singles = {k[0]: v for k, v in priors.items() if len(k) == 1}
    ratio = {a: priors[(a,)] / probs[a] for a in singles if probs[a] > 1e-9}
    assert max(ratio.values()) == pytest.approx(min(ratio.values()), rel=1e-4)

    v_raw = ev.policy.predict_values(obs).reshape(-1)[0].item()
    assert value == pytest.approx(v_raw + victory_points(st, seat) / 10, abs=1e-5)


def test_zero_budget_is_exactly_the_greedy_agent(tiny_checkpoint: str) -> None:
    def factory(cls):  # type: ignore[no-untyped-def]
        def make(n: int, e: int, d: int):  # type: ignore[no-untyped-def]
            agents = [HeuristicAgent() for _ in range(n)]
            agents[e % n] = cls(tiny_checkpoint, rng=random.Random(d))
            return agents

        return make

    def zero(path, rng):  # type: ignore[no-untyped-def]
        return RLSearchAgent(path, config=SearchConfig(simulations=0), rng=rng)

    def search_factory(n: int, e: int, d: int):  # type: ignore[no-untyped-def]
        return [
            zero(tiny_checkpoint, random.Random(d)) if i == e % n else HeuristicAgent()
            for i in range(n)
        ]

    for seed in (1, 2):
        a = run_game(4, seed, seed, factory(RLAgent))
        b = run_game(4, seed, seed, search_factory)
        assert dataclasses.replace(a, agent_names=()) == dataclasses.replace(
            b, agent_names=()
        )


def test_a_searched_game_runs_and_logs_its_decisions(tiny_checkpoint: str) -> None:
    holder: list = []

    def factory(n: int, e: int, d: int):  # type: ignore[no-untyped-def]
        agents = benchmark_agent_factory(
            "rl_search_vs_heuristic",
            tiny_checkpoint,
            n,
            e,
            d,
            SearchConfig(simulations=4),
        )
        holder.extend(agents)
        return agents

    record = run_game(4, 1, 1, factory)
    assert record.winner is not None or record.winning_seat is None
    searcher = next(a for a in holder if hasattr(a, "log"))
    assert searcher.log.decisions
    assert any(d.simulations == 4 for d in searcher.log.decisions)
    assert all(d.ms >= 0 for d in searcher.log.decisions)
    _load_model.cache_clear()


def test_encoder_reset_is_per_board(tiny_checkpoint: str) -> None:
    ev = PolicyEvaluator(tiny_checkpoint, 4)
    _, st, _, _ = _decision_state()
    ev.ensure_board(st)
    first = ev._board
    ev.ensure_board(st)
    assert ev._board is first
    assert isinstance(ev.encoder, ObservationEncoder)
