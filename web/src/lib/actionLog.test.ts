import { describe, expect, test } from "vitest";
import type { TrailEntry } from "./api";
import { describeTrailEntry, recentRolls } from "./actionLog";

describe("describeTrailEntry", () => {
  test("formats a RollDice entry with the actual dice values", () => {
    const entry: TrailEntry = { player_id: 2, kind: "RollDice", dice_roll: [4, 3] };
    expect(describeTrailEntry(entry)).toBe("Player 2 rolled 4 + 3 = 7");
  });

  test("formats a RollDice entry with no dice_roll defensively", () => {
    const entry: TrailEntry = { player_id: 0, kind: "RollDice" };
    expect(describeTrailEntry(entry)).toBe("Player 0 rolled the dice");
  });

  test("never mentions specific resources for a Discard entry", () => {
    const entry: TrailEntry = { player_id: 1, kind: "Discard" };
    const line = describeTrailEntry(entry);
    expect(line).toBe("Player 1 discarded resources");
    expect(line).not.toMatch(/LUMBER|WOOL|GRAIN|BRICK|ORE/);
  });

  test("includes public fields for spatial and dev-card actions", () => {
    expect(
      describeTrailEntry({ player_id: 0, kind: "StealFrom", player_idx: 2 }),
    ).toBe("Player 0 stole from Player 2");
    expect(
      describeTrailEntry({ player_id: 0, kind: "PlayMonopoly", resource: "ORE" }),
    ).toBe("Player 0 played Monopoly on ORE");
    expect(
      describeTrailEntry({
        player_id: 0,
        kind: "PlayYearOfPlenty",
        resource_1: "WOOL",
        resource_2: "GRAIN",
      }),
    ).toBe("Player 0 played Year of Plenty (WOOL, GRAIN)");
  });

  test("falls back to a generic line for an unrecognized kind", () => {
    expect(describeTrailEntry({ player_id: 0, kind: "SomeNewKind" })).toBe(
      "Player 0: SomeNewKind",
    );
  });
});

describe("recentRolls", () => {
  const entries: TrailEntry[] = [
    { player_id: 0, kind: "PlaceRoad", edge_id: 1 },
    { player_id: 0, kind: "RollDice", dice_roll: [1, 2] },
    { player_id: 1, kind: "PlaceSettlement", vertex_id: 3 },
    { player_id: 1, kind: "RollDice", dice_roll: [3, 4] },
    { player_id: 2, kind: "RollDice", dice_roll: [5, 6] },
  ];

  test("filters to only RollDice entries", () => {
    const rolls = recentRolls(entries);
    expect(rolls).toHaveLength(3);
    expect(rolls.every((e) => e.kind === "RollDice")).toBe(true);
  });

  test("caps to the last n, preserving order", () => {
    const rolls = recentRolls(entries, 2);
    expect(rolls.map((e) => e.dice_roll)).toEqual([
      [3, 4],
      [5, 6],
    ]);
  });

  test("returns an empty array when there are no rolls", () => {
    expect(recentRolls([{ player_id: 0, kind: "EndTurn" }])).toEqual([]);
  });
});
