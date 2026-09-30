// Pure helpers behind the dice histogram (DiceHistogram.svelte): counting
// rolls per total, the fair-dice expectation, and trimming rolls the paced
// activity log hasn't revealed yet. No Svelte, so trivially testable.

import type { DiceHistoryView, DiceRollRecord } from "./api";

export const DICE_TOTALS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] as const;

export function rollTotal(roll: DiceRollRecord): number {
  return roll.dice[0] + roll.dice[1];
}

// counts[i] is how many rolls totalled DICE_TOTALS[i].
export function countTotals(rolls: DiceRollRecord[]): number[] {
  const counts = DICE_TOTALS.map(() => 0);
  for (const roll of rolls) counts[rollTotal(roll) - 2]++;
  return counts;
}

// Fair 2d6: P(total) = (6 - |total - 7|) / 36, the 7-peaked triangle.
export function expectedCounts(nRolls: number): number[] {
  return DICE_TOTALS.map((t) => (nRolls * (6 - Math.abs(t - 7))) / 36);
}

// byPlayer[p][i] is how many of player p's rolls totalled DICE_TOTALS[i].
export function countsByPlayer(rolls: DiceRollRecord[], numPlayers: number): number[][] {
  const byPlayer = Array.from({ length: numPlayers }, () => DICE_TOTALS.map(() => 0));
  for (const roll of rolls) byPlayer[roll.player_id][rollTotal(roll) - 2]++;
  return byPlayer;
}

// gameState runs ahead of the paced activity log (App.svelte's pendingTrail),
// so the newest `pendingRollCount` rolls haven't been "shown" yet -- drop
// them so the histogram never spoils a roll the log hasn't revealed.
export function visibleRolls(
  history: DiceHistoryView | undefined,
  pendingRollCount: number,
): DiceRollRecord[] {
  if (!history) return [];
  return history.rolls.slice(0, Math.max(0, history.rolls.length - pendingRollCount));
}
