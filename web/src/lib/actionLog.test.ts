import { describe, expect, test } from "vitest";
import type { TrailEntry } from "./api";
import { describeTrailEntry, groupActivityEntries, recentRolls } from "./actionLog";

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

  test("appends production for a RollDice entry that gained resources", () => {
    const entry: TrailEntry = {
      player_id: 0,
      kind: "RollDice",
      dice_roll: [3, 4],
      production: { "1": { LUMBER: 1, GRAIN: 1 }, "2": { GRAIN: 1 } },
    };
    expect(describeTrailEntry(entry)).toBe(
      "Player 0 rolled 3 + 4 = 7 (Player 1 +1 LUMBER, +1 GRAIN; Player 2 +1 GRAIN)",
    );
  });

  test("omits the production suffix when nothing was gained", () => {
    const entry: TrailEntry = { player_id: 0, kind: "RollDice", dice_roll: [3, 4] };
    expect(describeTrailEntry(entry)).toBe("Player 0 rolled 3 + 4 = 7");
  });

  test("describes an accepted trade as a deal, from the responder's perspective", () => {
    const entry: TrailEntry = {
      player_id: 2,
      kind: "AcceptTrade",
      trade_offer: { proposer: 0, give: { WOOL: 1 }, receive: { BRICK: 1 } },
    };
    // Proposer (0) gave WOOL and got BRICK; responder (2, "p") is the one
    // who actually handed over BRICK and received WOOL.
    expect(describeTrailEntry(entry)).toBe(
      "Player 2 traded with Player 0: gave 1 BRICK for 1 WOOL",
    );
  });

  test("falls back to a generic accepted-trade line with no trade_offer", () => {
    expect(describeTrailEntry({ player_id: 2, kind: "AcceptTrade" })).toBe(
      "Player 2 accepted the trade",
    );
  });
});

describe("groupActivityEntries", () => {
  test("collapses a multi-reject run that killed the trade into one line", () => {
    const entries: TrailEntry[] = [
      { player_id: 0, kind: "ProposeTrade", give: { WOOL: 1 }, receive: { BRICK: 1 } },
      { player_id: 1, kind: "RejectTrade" },
      { player_id: 2, kind: "RejectTrade" },
      { player_id: 0, kind: "EndTurn" },
    ];
    const lines = groupActivityEntries(entries);
    expect(lines.map((l) => l.text)).toEqual([
      "Player 0 proposed a trade",
      "Everyone rejected the trade (Player 1, Player 2)",
      "Player 0 ended their turn",
    ]);
    expect(lines[1].playerIds).toEqual([1, 2]);
  });

  test("leaves rejects uncollapsed when someone eventually accepts", () => {
    const entries: TrailEntry[] = [
      { player_id: 1, kind: "RejectTrade" },
      {
        player_id: 2,
        kind: "AcceptTrade",
        trade_offer: { proposer: 0, give: { WOOL: 1 }, receive: { BRICK: 1 } },
      },
    ];
    const lines = groupActivityEntries(entries);
    expect(lines).toHaveLength(2);
    expect(lines[0].text).toBe("Player 1 rejected the trade");
    expect(lines[1].text).toContain("traded with Player 0");
  });

  test("leaves a single rejection as its own line, uncollapsed", () => {
    const entries: TrailEntry[] = [{ player_id: 1, kind: "RejectTrade" }];
    const lines = groupActivityEntries(entries);
    expect(lines).toEqual([
      { key: "0", text: "Player 1 rejected the trade", playerIds: [1] },
    ]);
  });

  test("passes non-reject entries through unchanged", () => {
    const entries: TrailEntry[] = [
      { player_id: 0, kind: "RollDice", dice_roll: [3, 4] },
      { player_id: 1, kind: "EndTurn" },
    ];
    const lines = groupActivityEntries(entries);
    expect(lines.map((l) => l.text)).toEqual([
      "Player 0 rolled 3 + 4 = 7",
      "Player 1 ended their turn",
    ]);
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
