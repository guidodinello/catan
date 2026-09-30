import { describe, expect, test } from "vitest";
import type { DiceRollRecord } from "./api";
import { countTotals, countsByPlayer, expectedCounts, visibleRolls } from "./diceStats";

const roll = (player_id: number, a: number, b: number, turn = 1): DiceRollRecord => ({
  turn,
  player_id,
  dice: [a, b],
});

describe("countTotals", () => {
  test("is all zeros for no rolls", () => {
    expect(countTotals([])).toEqual(new Array(11).fill(0));
  });

  test("buckets by die sum, index 0 = total 2 and index 10 = total 12", () => {
    const counts = countTotals([roll(0, 1, 1), roll(1, 6, 6), roll(0, 6, 6), roll(2, 3, 4)]);
    expect(counts[0]).toBe(1);
    expect(counts[10]).toBe(2);
    expect(counts[5]).toBe(1);
    expect(counts.reduce((a, b) => a + b, 0)).toBe(4);
  });

  test("handles a game of nothing but sevens", () => {
    const counts = countTotals(Array.from({ length: 5 }, () => roll(0, 2, 5)));
    expect(counts[5]).toBe(5);
    expect(counts.filter((c) => c > 0)).toHaveLength(1);
  });
});

describe("expectedCounts", () => {
  test("sums to the number of rolls", () => {
    const sum = expectedCounts(72).reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(72);
  });

  test("peaks at 7 with the 1-2-3-4-5-6-5-4-3-2-1 triangle over 36", () => {
    const expected = expectedCounts(36);
    expect(expected).toEqual([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]);
  });

  test("is zero for zero rolls", () => {
    expect(expectedCounts(0).every((v) => v === 0)).toBe(true);
  });
});

describe("countsByPlayer", () => {
  test("per-player rows sum to the overall totals", () => {
    const rolls = [roll(0, 3, 4), roll(1, 3, 4), roll(1, 1, 1), roll(2, 6, 6), roll(0, 2, 2)];
    const byPlayer = countsByPlayer(rolls, 3);
    const totals = countTotals(rolls);
    totals.forEach((total, i) => {
      expect(byPlayer.reduce((sum, row) => sum + row[i], 0)).toBe(total);
    });
    expect(byPlayer[1][5]).toBe(1);
  });

  test("gives an all-zero row to a player who never rolled", () => {
    const byPlayer = countsByPlayer([roll(0, 1, 2)], 3);
    expect(byPlayer[2]).toEqual(new Array(11).fill(0));
  });
});

describe("visibleRolls", () => {
  const history = {
    complete: true,
    rolls: [roll(0, 1, 1), roll(1, 2, 2), roll(2, 3, 3)],
  };

  test("returns everything when nothing is pending", () => {
    expect(visibleRolls(history, 0)).toHaveLength(3);
  });

  test("drops the newest rolls that are still pending reveal", () => {
    expect(visibleRolls(history, 2).map((r) => r.player_id)).toEqual([0]);
  });

  test("never goes negative when more rolls are pending than exist", () => {
    expect(visibleRolls(history, 10)).toEqual([]);
  });

  test("is empty when the server sent no history", () => {
    expect(visibleRolls(undefined, 0)).toEqual([]);
  });
});
