"""Open-loop, single-observer ISMCTS over the rl seat's own decisions.

Torch-free on purpose: the policy/critic (``agents/rl_search.py``) and the
opponent model plug in through the ``Evaluator`` protocol and the ordinary
``Agent`` protocol, so this module is testable in CI with stubs.

Design (experiment 011, gamekit note 021):

* **Tree.** A node is one of the observer's *top-level* decisions (every
  decision except a domestic-trade response), identified by the path of the
  observer's own action keys. No game states are stored.
* **Determinization.** Each simulation starts from a fresh ``determinize``d
  copy of the live state: opponents' hands and dev cards are re-dealt from
  *public information only*, the RNG is reseeded, so dice/steals/draws are
  sampled chance. Opponents then act through a fixed ``OpponentModel``; they
  are not searched, which is why backup is single-seat (a departure from
  021's per-seat multi-player UCT).
* **Leaf.** One rule for every leaf: the observer's next top-level decision
  state, valued by the evaluator (V+ = V + own public VP/10, experiment 009).
  A terminal state is +1 if the observer won, else -1.
* **Selection.** PUCT over the currently legal children with Q min-max
  normalized over the tree (009 found sibling gaps ~0.01 against a range of
  ~1.2; without normalization the prior term would swamp Q) and a first-play
  urgency equal to the parent's mean normalized Q.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from engine.actions import Action, RejectTrade
from engine.board import Resource
from engine.game import CatanGame, true_victory_points
from engine.state import (
    BANK_RESOURCE_COUNT,
    DEV_DECK_COUNTS,
    PROGRESS_CARD_TYPES,
    DevCard,
    DevCardType,
    GameState,
    Phase,
    acting_player,
)
from rl.action_space import OPENER_ROAD_BUILDING, atom_sequence, is_trade_proposal

ActionKey = tuple[int, ...]

MAX_REDEALS = 50
WIN_VALUE = 1.0
LOSS_VALUE = -1.0


@dataclass(frozen=True, slots=True)
class SearchConfig:
    simulations: int = 128  # S* of experiment 011 (see its budget rule)
    c_puct: float = 1.25
    opponent: str = "heuristic"  # "heuristic" | "self"
    max_advance_steps: int = 5000
    record_edge_stats: bool = False


class Evaluator(Protocol):
    def evaluate(
        self, state: GameState, seat: int, legal: Sequence[Action]
    ) -> tuple[dict[ActionKey, float], float]:
        """Priors (summing to 1) over the action keys of ``legal`` and the
        leaf value of ``state`` from ``seat``'s view."""
        ...


class OpponentModel(Protocol):
    def choose_action(
        self, state: GameState, legal_actions: list[Action], player_idx: int
    ) -> Action: ...


def action_key(action: Action, seat: int, num_players: int) -> ActionKey:
    """Hashable identity of a composed action. The two orderings of a
    Road Building pair reach the same state, so they share one key."""
    seq = atom_sequence(action, seat, num_players)
    if seq[0] == OPENER_ROAD_BUILDING:
        return (seq[0], *sorted(seq[1:]))
    return seq


def keyed_actions(
    legal: Sequence[Action], seat: int, num_players: int
) -> dict[ActionKey, Action]:
    """One representative legal action per key; trade sentinels dropped."""
    out: dict[ActionKey, Action] = {}
    for a in legal:
        if is_trade_proposal(a):
            continue
        out.setdefault(action_key(a, seat, num_players), a)
    return out


# --- Determinization ---------------------------------------------------------


def determinize(state: GameState, seat: int, rng: random.Random) -> GameState:
    """A copy of ``state`` whose hidden information is re-dealt from public
    information only.

    Read from ``state``: the observer's own hand and dev cards; per opponent,
    only ``resource_card_count()``, ``len(dev_hand)`` and each dev card's
    ``bought_this_turn`` flag (a purchase is seen by the whole table); bank
    counts; ``len(dev_deck)``; the public played-card counters. Never read:
    an opponent's resource composition, dev card types, the deck order, or
    ``state.rng``.

    * Resources: each resource has ``BANK_RESOURCE_COUNT`` cards across bank
      and hands, so the opponents jointly hold ``19 - bank - own`` of it; that
      pool is shuffled and dealt by the public hand sizes.
    * Dev cards: the unseen multiset is the full deck minus the observer's own
      hand minus cards publicly played. ``played_progress_count`` is a total
      without types, so which progress cards were played is drawn uniformly
      from those the observer does not hold. The multiset is shuffled and dealt
      into the opponents' hand slots, the rest is the deck.
    * A deal that hands an opponent >= 10 true VP is redealt (that game would
      already be over).
    """
    s = state.copy()
    own = s.players[seat]
    opponents = [p for p in s.players if p.player_id != seat]

    sizes = [p.resource_card_count() for p in opponents]
    res_pool: list[Resource] = []
    for r in Resource:
        res_pool += [r] * (BANK_RESOURCE_COUNT - s.bank[r] - own.resources[r])
    if len(res_pool) != sum(sizes):
        raise ValueError(
            f"resource conservation broken: pool {len(res_pool)} != hands {sum(sizes)}"
        )

    held = Counter(c.card_type for c in own.dev_hand)
    progress_unheld: list[DevCardType] = []
    for t in sorted(PROGRESS_CARD_TYPES, key=lambda t: t.value):
        progress_unheld += [t] * (DEV_DECK_COUNTS[t] - held[t])
    played_progress = sum(p.played_progress_count for p in s.players)
    played_idx = set(rng.sample(range(len(progress_unheld)), played_progress))
    dev_pool: list[DevCardType] = [
        t for i, t in enumerate(progress_unheld) if i not in played_idx
    ]
    knights = (
        DEV_DECK_COUNTS[DevCardType.KNIGHT]
        - held[DevCardType.KNIGHT]
        - sum(p.played_knights for p in s.players)
    )
    vps = (
        DEV_DECK_COUNTS[DevCardType.VICTORY_POINT]
        - held[DevCardType.VICTORY_POINT]
        - sum(p.revealed_vp_cards for p in s.players)
    )
    dev_pool += [DevCardType.KNIGHT] * knights + [DevCardType.VICTORY_POINT] * vps
    slots = [[c.bought_this_turn for c in p.dev_hand] for p in opponents]
    if len(dev_pool) != sum(map(len, slots)) + len(s.dev_deck):
        raise ValueError(
            f"dev-card conservation broken: unseen {len(dev_pool)} != "
            f"hands {sum(map(len, slots))} + deck {len(s.dev_deck)}"
        )
    deck_size = len(s.dev_deck)

    for _ in range(MAX_REDEALS):
        rng.shuffle(res_pool)
        rng.shuffle(dev_pool)
        i = j = 0
        for p, size, flags in zip(opponents, sizes, slots, strict=True):
            hand = Counter(res_pool[i : i + size])
            i += size
            p.resources = {r: hand.get(r, 0) for r in Resource}
            p.dev_hand = [
                DevCard(card_type=t, bought_this_turn=f)
                for t, f in zip(dev_pool[j : j + len(flags)], flags, strict=True)
            ]
            j += len(flags)
        s.dev_deck = dev_pool[j : j + deck_size]
        if all(true_victory_points(s, p.player_id) < 10 for p in opponents):
            break
    s.rng = random.Random(rng.getrandbits(64))
    return s


# --- Search ------------------------------------------------------------------


class _Edge:
    __slots__ = ("child", "prior", "value_sum", "visits")

    def __init__(self, prior: float) -> None:
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.child: _Node | None = None


class _Node:
    __slots__ = ("edges",)

    def __init__(self) -> None:
        self.edges: dict[ActionKey, _Edge] | None = None


class _MinMax:
    __slots__ = ("hi", "lo")

    def __init__(self) -> None:
        self.lo = math.inf
        self.hi = -math.inf

    def update(self, v: float) -> None:
        self.lo = min(self.lo, v)
        self.hi = max(self.hi, v)

    def norm(self, v: float) -> float:
        return (v - self.lo) / (self.hi - self.lo) if self.hi > self.lo else 0.0


@dataclass(frozen=True, slots=True)
class SearchResult:
    key: ActionKey
    action: Action
    visits: dict[ActionKey, int]
    # (key, n, mean, sd) of leaf values per root edge; only with
    # ``record_edge_stats`` (the smoke run's leaf-noise check).
    edge_stats: tuple[tuple[ActionKey, int, float, float], ...] = ()


class _Search:
    def __init__(
        self,
        state: GameState,
        seat: int,
        config: SearchConfig,
        evaluator: Evaluator,
        opponent: OpponentModel,
        rng: random.Random,
    ) -> None:
        self.state = state
        self.seat = seat
        self.cfg = config
        self.evaluator = evaluator
        self.opponent = opponent
        self.rng = rng
        self.n = len(state.players)
        self.game = CatanGame(num_players=self.n)
        self.minmax = _MinMax()
        self.root = _Node()
        self.leaf_values: dict[ActionKey, list[float]] = {}

    # -- one simulation --------------------------------------------------

    def _expand(
        self, node: _Node, st: GameState, legal: list[Action]
    ) -> tuple[bool, float]:
        priors, value = self.evaluator.evaluate(st, self.seat, legal)
        fresh = node.edges is None
        if node.edges is None:
            node.edges = {}
        for k, p in priors.items():
            if k not in node.edges:
                node.edges[k] = _Edge(p)
        return fresh, value

    def _select(self, node: _Node, keys: Sequence[ActionKey]) -> ActionKey:
        assert node.edges is not None
        edges = [(k, node.edges[k]) for k in keys]
        total_p = sum(e.prior for _, e in edges) or 1.0
        n_parent = sum(e.visits for _, e in edges)
        visited = [e for _, e in edges if e.visits]
        if visited:
            mean = sum(e.value_sum for e in visited) / sum(e.visits for e in visited)
            fpu = self.minmax.norm(mean)
        else:
            fpu = 0.0
        scale = self.cfg.c_puct * math.sqrt(n_parent + 1)
        best = keys[0]
        best_score = (-math.inf, -math.inf)
        for k, e in edges:
            q = self.minmax.norm(e.value_sum / e.visits) if e.visits else fpu
            score = (q + scale * (e.prior / total_p) / (1 + e.visits), e.prior)
            if score > best_score:
                best, best_score = k, score
        return best

    def _advance(self, st: GameState) -> float | None:
        """Play opponents (and the observer's forced trade rejects) until the
        observer's next top-level decision. Returns the terminal value, or
        ``None`` when a decision point was reached."""
        for _ in range(self.cfg.max_advance_steps):
            if self.game.is_terminal(st):
                return WIN_VALUE if st.winner == self.seat else LOSS_VALUE
            actor = acting_player(st)
            legal = self.game.legal_actions(st)
            action: Action
            if actor == self.seat:
                if st.phase is not Phase.AWAIT_TRADE_RESPONSE:
                    return None
                action = next(a for a in legal if isinstance(a, RejectTrade))
            else:
                action = self.opponent.choose_action(st, legal, actor)
            self.game.apply_action(st, action)
        raise RuntimeError("search rollout exceeded max_advance_steps")

    def _backup(self, path: list[tuple[ActionKey, _Edge]], value: float) -> None:
        for _, e in path:
            e.visits += 1
            e.value_sum += value
            self.minmax.update(e.value_sum / e.visits)
        if self.cfg.record_edge_stats and path:
            self.leaf_values.setdefault(path[0][0], []).append(value)

    def simulate(self) -> None:
        st = determinize(self.state, self.seat, self.rng)
        node = self.root
        path: list[tuple[ActionKey, _Edge]] = []
        while True:
            legal = self.game.legal_actions(st)
            keyed = keyed_actions(legal, self.seat, self.n)
            if node.edges is None or any(k not in node.edges for k in keyed):
                fresh, value = self._expand(node, st, legal)
                if fresh:
                    self._backup(path, value)
                    return
            assert node.edges is not None
            key = self._select(node, list(keyed))
            path.append((key, node.edges[key]))
            self.game.apply_action(st, keyed[key])
            terminal = self._advance(st)
            if terminal is not None:
                self._backup(path, terminal)
                return
            edge = node.edges[key]
            if edge.child is None:
                edge.child = _Node()
            node = edge.child

    def run(self, real_keyed: dict[ActionKey, Action]) -> SearchResult:
        # Expand the root from a first determinization (not a simulation).
        st = determinize(self.state, self.seat, self.rng)
        self._expand(self.root, st, self.game.legal_actions(st))
        for _ in range(self.cfg.simulations):
            self.simulate()
        root_edges = self.root.edges
        assert root_edges is not None
        best = max(
            (k for k in real_keyed if k in root_edges),
            key=lambda k: (root_edges[k].visits, root_edges[k].prior),
        )
        stats: tuple[tuple[ActionKey, int, float, float], ...] = ()
        if self.cfg.record_edge_stats:
            stats = tuple(
                (k, len(v), _mean(v), _sd(v)) for k, v in self.leaf_values.items()
            )
        return SearchResult(
            key=best,
            action=real_keyed[best],
            visits={k: e.visits for k, e in root_edges.items()},
            edge_stats=stats,
        )


def _mean(v: Sequence[float]) -> float:
    return sum(v) / len(v)


def _sd(v: Sequence[float]) -> float:
    if len(v) < 2:
        return 0.0
    m = _mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def search(
    state: GameState,
    seat: int,
    legal_actions: Sequence[Action],
    *,
    config: SearchConfig,
    evaluator: Evaluator,
    opponent: OpponentModel,
    rng: random.Random,
) -> SearchResult:
    """Run ``config.simulations`` simulations from ``state`` (left untouched)
    and return the most-visited root action."""
    real_keyed = keyed_actions(legal_actions, seat, len(state.players))
    if not real_keyed:
        raise RuntimeError("no encodable legal action to search")
    return _Search(state, seat, config, evaluator, opponent, rng).run(real_keyed)
