"""Factored action space: a flat atom head plus a composition buffer.

catan's legal-action set is neither small nor fixed-shape. Measured over full
random games: ``Discard`` reaches **197** legal multisets in one decision, and
``_road_building_pairs`` emits *ordered* pairs, making the ROLL phase **106**
wide. Flattening either is hopeless, and ``gamekit.rl.env.SingleAgentEnv``
says so in its own docstring -- naming catan as the reason it does not serve
this case.

The encoding here is one flat ``Discrete`` head over **atoms** (a vertex, an
edge, a resource, ...), where a multi-parameter action is spelled as a short
sequence of atoms consumed over consecutive env steps. A 3-card discard is
three picks from a 5-wide resource head, covering all 197 multisets exactly.

Why not ``MultiDiscrete``: sb3-contrib computes every sub-head's mask *before*
sampling, so the second road-building edge could not be masked on the first
one's choice -- and it must be, since ``_road_building_pairs`` recomputes
legality with ``e1`` already placed. Sequential atoms re-derive the mask from
the real engine at each sub-step, so legality stays exact.

**Masks are derived from ``legal_actions()``, never from re-implemented rules.**
``mask()`` decomposes every currently legal action into its atom sequence and
asks which atoms can extend the buffered prefix. Nothing here knows what a
settlement costs. That is what makes the round-trip exact in both directions
and keeps decision 10 honest: the engine remains the single source of legality.

Seat-relative encoding: ``StealFrom`` carries an absolute player index, but the
atom is the offset from the acting seat, so one policy transfers across seats
(the observation encoder is seat-relative for the same reason).
"""

from __future__ import annotations

from engine.actions import (
    AcceptTrade,
    Action,
    BuyDevCard,
    CounterTrade,
    Discard,
    EndTurn,
    MoveRobber,
    PlaceCity,
    PlaceRoad,
    PlaceSettlement,
    PlayKnight,
    PlayMonopoly,
    PlayRoadBuilding,
    PlayVictoryPoint,
    PlayYearOfPlenty,
    ProposeTrade,
    RejectTrade,
    RollDice,
    StealFrom,
    TradeBank,
    TradePort,
)
from engine.board import GEOMETRY, NUM_EDGES, NUM_LAND_HEXES, NUM_VERTICES, Resource
from engine.game import MAX_PLAYERS

RESOURCES: tuple[Resource, ...] = tuple(Resource)
RESOURCE_INDEX: dict[Resource, int] = {r: i for i, r in enumerate(RESOURCES)}
HEX_INDEX: dict[tuple[int, int, int], int] = {
    h: i for i, h in enumerate(GEOMETRY.land_hexes)
}

# --- Atom block layout -----------------------------------------------------
# Blocks are contiguous so an atom index decodes to (block, offset) by range.
VERTEX_SETTLEMENT_BASE = 0
VERTEX_CITY_BASE = VERTEX_SETTLEMENT_BASE + NUM_VERTICES  # 54
EDGE_BASE = VERTEX_CITY_BASE + NUM_VERTICES  # 108
HEX_BASE = EDGE_BASE + NUM_EDGES  # 180
STEAL_BASE = HEX_BASE + NUM_LAND_HEXES  # 199
RESOURCE_BASE = STEAL_BASE + MAX_PLAYERS  # 203
SIMPLE_BASE = RESOURCE_BASE + len(RESOURCES)  # 208

# Parameterless actions, one atom each.
SIMPLE_ROLL_DICE = SIMPLE_BASE + 0
SIMPLE_BUY_DEV_CARD = SIMPLE_BASE + 1
SIMPLE_PLAY_KNIGHT = SIMPLE_BASE + 2
SIMPLE_PLAY_VICTORY_POINT = SIMPLE_BASE + 3
SIMPLE_END_TURN = SIMPLE_BASE + 4
SIMPLE_ACCEPT_TRADE = SIMPLE_BASE + 5
SIMPLE_REJECT_TRADE = SIMPLE_BASE + 6
N_SIMPLE = 8  # one spare, so adding a parameterless action keeps N_ATOMS

# Openers: the first atom of a multi-parameter action. They exist to
# disambiguate the shared RESOURCE/EDGE blocks -- Monopoly and Year of Plenty
# are both legal in MAIN, so a bare resource pick would be ambiguous.
OPENER_BASE = SIMPLE_BASE + N_SIMPLE  # 216
OPENER_ROAD_BUILDING = OPENER_BASE + 0
OPENER_MONOPOLY = OPENER_BASE + 1
OPENER_YEAR_OF_PLENTY = OPENER_BASE + 2
OPENER_TRADE_BANK = OPENER_BASE + 3
OPENER_TRADE_PORT = OPENER_BASE + 4
N_OPENERS = 5

# Reserved for the factored domestic-trade heads. Masked off for now (see
# ``allow_trade_proposals``); the slots exist so enabling them later is a mask
# change rather than a reshaping of the policy head.
TRADE_BASE = OPENER_BASE + N_OPENERS  # 221
TRADE_PROPOSE = TRADE_BASE + 0
TRADE_COUNTER = TRADE_BASE + 1
N_TRADE_RESERVED = 2

N_ATOMS = TRADE_BASE + N_TRADE_RESERVED  # 223


class ActionEncodingError(Exception):
    """Raised when an action cannot be expressed in the atom space."""


def _resource_atoms(bundle: dict[Resource, int]) -> tuple[int, ...]:
    """Resource multiset as atoms in canonical (enum) order.

    Canonical order matters: without it the same discard multiset would have
    one atom path per permutation, splitting the policy's probability mass
    across paths that mean the same thing.
    """
    atoms: list[int] = []
    for resource in RESOURCES:
        atoms.extend(
            [RESOURCE_BASE + RESOURCE_INDEX[resource]] * bundle.get(resource, 0)
        )
    return tuple(atoms)


def atom_sequence(action: Action, seat: int, num_players: int) -> tuple[int, ...]:
    """The atom sequence spelling ``action`` from ``seat``'s perspective."""
    match action:
        case PlaceSettlement(vertex_id=v):
            return (VERTEX_SETTLEMENT_BASE + v,)
        case PlaceCity(vertex_id=v):
            return (VERTEX_CITY_BASE + v,)
        case PlaceRoad(edge_id=e):
            return (EDGE_BASE + e,)
        case MoveRobber(hex_id=h):
            return (HEX_BASE + HEX_INDEX[h],)
        case StealFrom(player_idx=p):
            return (STEAL_BASE + (p - seat) % num_players,)
        case RollDice():
            return (SIMPLE_ROLL_DICE,)
        case BuyDevCard():
            return (SIMPLE_BUY_DEV_CARD,)
        case PlayKnight():
            return (SIMPLE_PLAY_KNIGHT,)
        case PlayVictoryPoint():
            return (SIMPLE_PLAY_VICTORY_POINT,)
        case EndTurn():
            return (SIMPLE_END_TURN,)
        case AcceptTrade():
            return (SIMPLE_ACCEPT_TRADE,)
        case RejectTrade():
            return (SIMPLE_REJECT_TRADE,)
        case Discard(resources=bundle):
            return _resource_atoms(bundle)
        case PlayRoadBuilding(edge_id_1=e1, edge_id_2=e2):
            return (OPENER_ROAD_BUILDING, EDGE_BASE + e1, EDGE_BASE + e2)
        case PlayMonopoly(resource=r):
            return (OPENER_MONOPOLY, RESOURCE_BASE + RESOURCE_INDEX[r])
        case PlayYearOfPlenty(resource_1=r1, resource_2=r2):
            return (
                OPENER_YEAR_OF_PLENTY,
                RESOURCE_BASE + RESOURCE_INDEX[r1],
                RESOURCE_BASE + RESOURCE_INDEX[r2],
            )
        case TradeBank(give=give, receive=receive):
            return (
                OPENER_TRADE_BANK,
                *_resource_atoms(give),
                *_resource_atoms(receive),
            )
        case TradePort(give=give, receive=receive):
            return (
                OPENER_TRADE_PORT,
                *_resource_atoms(give),
                *_resource_atoms(receive),
            )
        case ProposeTrade():
            return (TRADE_PROPOSE,)
        case CounterTrade():
            return (TRADE_COUNTER,)
        case _:  # pragma: no cover - the Action union is closed
            raise ActionEncodingError(f"no atom encoding for {action!r}")


def is_trade_proposal(action: Action) -> bool:
    """Whether ``action`` is one of the two open-ended trade sentinels.

    These are the actions ``legal_actions`` cannot pre-enumerate (README
    decision 5): the real bundle is built by the caller and validated in
    ``apply_action``. They are reserved rather than encoded for now.
    """
    return isinstance(action, ProposeTrade | CounterTrade)


class ActionComposer:
    """Accumulates atoms until they spell one complete engine action.

    While a sequence is partially filled nothing has been applied to the
    engine, so the acting seat cannot change mid-composition and no opponent
    can interleave.
    """

    def __init__(self, *, allow_trade_proposals: bool = False) -> None:
        self.allow_trade_proposals = allow_trade_proposals
        self._prefix: tuple[int, ...] = ()

    @property
    def prefix(self) -> tuple[int, ...]:
        return self._prefix

    @property
    def in_progress(self) -> bool:
        return bool(self._prefix)

    def reset(self) -> None:
        self._prefix = ()

    def _candidates(
        self, legal_actions: list[Action], seat: int, num_players: int
    ) -> list[tuple[tuple[int, ...], Action]]:
        """Legal actions whose atom sequence extends the buffered prefix."""
        depth = len(self._prefix)
        out: list[tuple[tuple[int, ...], Action]] = []
        for action in legal_actions:
            if not self.allow_trade_proposals and is_trade_proposal(action):
                continue
            seq = atom_sequence(action, seat, num_players)
            if len(seq) > depth and seq[:depth] == self._prefix:
                out.append((seq, action))
        return out

    def mask(
        self, legal_actions: list[Action], seat: int, num_players: int
    ) -> list[bool]:
        """Atoms that can legally extend the buffered prefix."""
        mask = [False] * N_ATOMS
        depth = len(self._prefix)
        for seq, _ in self._candidates(legal_actions, seat, num_players):
            mask[seq[depth]] = True
        return mask

    def push(
        self, atom: int, legal_actions: list[Action], seat: int, num_players: int
    ) -> Action | None:
        """Consume one atom.

        Returns the completed action when ``atom`` finishes a sequence (and
        clears the buffer), or ``None`` when more atoms are still needed.
        """
        candidates = self._candidates(legal_actions, seat, num_players)
        depth = len(self._prefix)
        extended = self._prefix + (atom,)
        matches = [(seq, a) for seq, a in candidates if seq[depth] == atom]
        if not matches:
            raise ActionEncodingError(
                f"atom {atom} does not extend prefix {self._prefix}"
            )
        complete = [a for seq, a in matches if len(seq) == len(extended)]
        if complete:
            if len(matches) != len(complete):
                raise ActionEncodingError(
                    f"ambiguous prefix {extended}: completes one action and "
                    f"extends another"
                )
            self._prefix = ()
            return complete[0]
        self._prefix = extended
        return None
