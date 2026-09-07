// Pure helpers over TrailEntry (server/serialize.py's serialize_trail_entry
// output) -- turning an already-applied action into a readable line, and
// picking out the dice-roll subset for a "recent rolls" view. No Svelte
// here, so these are trivially testable and reusable by both
// ActivityLog.svelte and HandSummary.svelte.

import type { TrailEntry } from "./api";

// engine/actions.py's per-kind field names, read directly off `entry`
// (loosely typed like LegalAction -- see api.ts) since a trail entry's
// shape varies by `kind` the same way a legal action's does.
export function describeTrailEntry(entry: TrailEntry): string {
  const p = `Player ${entry.player_id}`;
  switch (entry.kind) {
    case "PlaceSettlement":
      return `${p} built a settlement`;
    case "PlaceCity":
      return `${p} built a city`;
    case "PlaceRoad":
      return `${p} built a road`;
    case "RollDice": {
      if (!entry.dice_roll) return `${p} rolled the dice`;
      const [d1, d2] = entry.dice_roll;
      return `${p} rolled ${d1} + ${d2} = ${d1 + d2}`;
    }
    case "Discard":
      // resources deliberately redacted server-side -- see
      // server/serialize.py's _TRAIL_REDACTED_FIELDS.
      return `${p} discarded resources`;
    case "MoveRobber":
      return `${p} moved the robber`;
    case "StealFrom":
      return `${p} stole from Player ${entry.player_idx}`;
    case "BuyDevCard":
      return `${p} bought a development card`;
    case "PlayKnight":
      return `${p} played Knight`;
    case "PlayRoadBuilding":
      return `${p} played Road Building`;
    case "PlayYearOfPlenty":
      return `${p} played Year of Plenty (${entry.resource_1}, ${entry.resource_2})`;
    case "PlayMonopoly":
      return `${p} played Monopoly on ${entry.resource}`;
    case "PlayVictoryPoint":
      return `${p} revealed a Victory Point card`;
    case "TradeBank":
      return `${p} traded with the bank`;
    case "TradePort":
      return `${p} traded at a port`;
    case "ProposeTrade":
      return `${p} proposed a trade`;
    case "AcceptTrade":
      return `${p} accepted the trade`;
    case "RejectTrade":
      return `${p} rejected the trade`;
    case "EndTurn":
      return `${p} ended their turn`;
    default:
      return `${p}: ${entry.kind}`;
  }
}

export function recentRolls(entries: TrailEntry[], n = 5): TrailEntry[] {
  return entries.filter((e) => e.kind === "RollDice").slice(-n);
}
