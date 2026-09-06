"""Hot-seat CLI for the Catan rules engine: human seats via ``HumanAgent``,
the rest filled with ``HeuristicAgent`` so a solo player has real opponents.

Text-menu loop: render the public board plus the acting player's own private
hand, then defer the actual choice to whichever ``agents.Agent`` occupies
that seat, apply it, and repeat until the game ends.
"""

from __future__ import annotations

from collections import defaultdict

from agents import Agent, HeuristicAgent, HumanAgent
from engine.board import GEOMETRY, PortType, Resource, Terrain
from engine.game import CatanGame
from engine.state import DevCardType, GameState, acting_player

TERRAIN_LETTERS: dict[Terrain, str] = {
    Terrain.FOREST: "L",
    Terrain.PASTURE: "W",
    Terrain.FIELD: "G",
    Terrain.HILL: "B",
    Terrain.MOUNTAIN: "O",
    Terrain.DESERT: "D",
}

RESOURCE_LETTERS: dict[Resource, str] = {
    Resource.LUMBER: "Lumber",
    Resource.WOOL: "Wool",
    Resource.GRAIN: "Grain",
    Resource.BRICK: "Brick",
    Resource.ORE: "Ore",
}

PLAYER_INITIALS = "ABCDEFGH"


def _player_label(idx: int) -> str:
    return PLAYER_INITIALS[idx]


def render_board(state: GameState) -> None:
    print("\n=== Board ===")
    hexes_by_row: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for h in GEOMETRY.land_hexes:
        hexes_by_row[h[1]].append(h)  # group by axial r

    for r in sorted(hexes_by_row):
        row = sorted(hexes_by_row[r], key=lambda h: h[0])
        cells = []
        for h in row:
            terrain = state.board.terrain[h]
            letter = TERRAIN_LETTERS[terrain]
            token = state.board.tokens.get(h, "--")
            robber = "*" if h == state.board.robber_hex else " "
            cells.append(f"[{letter}{token}{robber}]")
        indent = "  " * abs(r)
        print(indent + " ".join(cells))

    print("\nSettlements/Cities:")
    for i, player in enumerate(state.players):
        for v in sorted(player.settlement_vertices):
            print(f"  vertex {v}: settlement ({_player_label(i)})")
        for v in sorted(player.city_vertices):
            print(f"  vertex {v}: city ({_player_label(i)})")

    print("Roads:")
    for i, player in enumerate(state.players):
        for e in sorted(player.road_edges):
            print(f"  edge {e}: road ({_player_label(i)})")

    print("Ports:")
    for edge_id, port_type in state.board.port_types.items():
        label = (
            "3:1 generic" if port_type is PortType.GENERIC else f"2:1 {port_type.name}"
        )
        print(f"  edge {edge_id}: {label}")

    if state.board.robber_hex is not None:
        print(f"Robber on hex {state.board.robber_hex}")


def render_hand(state: GameState, player_idx: int) -> None:
    player = state.players[player_idx]
    print(f"\n=== Player {_player_label(player_idx)}'s hand (private) ===")
    for r in Resource:
        print(f"  {RESOURCE_LETTERS[r]}: {player.resources[r]}")
    dev_counts: dict[DevCardType, int] = defaultdict(int)
    for c in player.dev_hand:
        dev_counts[c.card_type] += 1
    if dev_counts:
        print("  Dev cards:")
        for card_type, count in dev_counts.items():
            print(f"    {card_type.name}: {count}")
    print(
        f"  Pieces remaining: settlements={player.settlements_remaining} "
        f"cities={player.cities_remaining} roads={player.roads_remaining}"
    )
    print(f"  Knights played: {player.played_knights}")


def run_hotseat_game(
    num_players: int, agents: list[Agent], seed: int | None = None
) -> None:
    if len(agents) != num_players:
        raise ValueError(f"expected {num_players} agents, got {len(agents)}")
    game = CatanGame(num_players=num_players)
    state = game.reset(seed=seed)

    while not game.is_terminal(state):
        actor = acting_player(state)
        if isinstance(agents[actor], HumanAgent):
            render_board(state)
            render_hand(state, actor)
            print(
                f"\nPhase: {state.phase.name} -- Player {_player_label(actor)} to act"
            )

        legal = game.legal_actions(state)
        action = agents[actor].choose_action(state, legal, actor)
        try:
            game.apply_action(state, action)
        except Exception as exc:  # noqa: BLE001 -- surface any rule violation to the player
            print(f"Illegal action: {exc}")

    render_board(state)
    winner = game.winner(state)
    print(
        f"\n=== Game over! Player {_player_label(winner)} wins! ==="
        if winner is not None
        else "Game over."
    )


def _prompt_int(prompt: str, valid: range | list[int]) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            choice = int(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if choice in valid:
            return choice
        print("Out of range, try again.")


def main() -> None:
    print("Catan hot-seat CLI")
    num_players = _prompt_int("Number of players (3 or 4): ", [3, 4])
    num_humans = _prompt_int(
        f"Number of human seats (0-{num_players}): ", range(num_players + 1)
    )
    agents: list[Agent] = [HumanAgent() for _ in range(num_humans)]
    agents += [HeuristicAgent() for _ in range(num_players - num_humans)]
    run_hotseat_game(num_players=num_players, agents=agents)


if __name__ == "__main__":
    main()
