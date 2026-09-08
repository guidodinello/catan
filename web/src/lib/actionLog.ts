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
      const base = `${p} rolled ${d1} + ${d2} = ${d1 + d2}`;
      return entry.production ? `${base} (${describeProduction(entry.production)})` : base;
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
    case "AcceptTrade": {
      if (!entry.trade_offer) return `${p} accepted the trade`;
      const { proposer, give, receive } = entry.trade_offer;
      // give/receive are from the *proposer*'s perspective (same convention
      // as TradeOfferBanner.svelte) -- the responder (p) is who actually
      // gave `receive` and got `give` in return.
      return `${p} traded with Player ${proposer}: gave ${formatBundle(receive)} for ${formatBundle(give)}`;
    }
    case "RejectTrade":
      return `${p} rejected the trade`;
    case "CounterTrade": {
      // give/receive here are the *counterer*'s own bundle (the action's own
      // fields, entry.give/entry.receive) -- distinct from entry.trade_offer,
      // which is the original offer being countered, used only to name who.
      const original = entry.trade_offer?.proposer;
      const give = entry.give as Record<string, number> | undefined;
      const receive = entry.receive as Record<string, number> | undefined;
      if (original === undefined || !give || !receive) {
        return `${p} countered with a different offer`;
      }
      return `${p} countered Player ${original}'s offer: gives ${formatBundle(give)}, wants ${formatBundle(receive)}`;
    }
    case "EndTurn":
      return `${p} ended their turn`;
    default:
      return `${p}: ${entry.kind}`;
  }
}

export function recentRolls(entries: TrailEntry[], n = 5): TrailEntry[] {
  return entries.filter((e) => e.kind === "RollDice").slice(-n);
}

// "1 WOOL, 2 GRAIN" -- a resource bundle (TradeOfferView's give/receive,
// same shape everywhere a bundle appears on the wire) as a short line.
function formatBundle(bundle: Record<string, number>): string {
  return Object.entries(bundle)
    .filter(([, count]) => count > 0)
    .map(([resource, count]) => `${count} ${resource}`)
    .join(", ");
}

export interface ActivityLine {
  key: string;
  text: string;
  playerIds: number[];
}

// Collapses a run of consecutive RejectTrade entries that ends the trade
// outright (nobody accepted) into one "everyone rejected" line -- a run
// that's instead followed by an AcceptTrade is left as individual lines,
// since someone did eventually say yes and each rejection leading up to
// that is more informative on its own. A run of exactly one RejectTrade
// (nothing to collapse) is also left alone. This is purely a display
// concern over an already-complete, correct trail -- no entries are
// dropped, only merged for readability. A rejected CounterTrade produces a
// run of exactly one RejectTrade (the original proposer's response to the
// counter), which the "nothing to collapse" case already leaves as its own
// line -- no special-casing needed for counters here.
export function groupActivityEntries(entries: TrailEntry[]): ActivityLine[] {
  const lines: ActivityLine[] = [];
  let i = 0;
  let key = 0;
  while (i < entries.length) {
    if (entries[i].kind !== "RejectTrade") {
      lines.push({ key: `${key++}`, text: describeTrailEntry(entries[i]), playerIds: [entries[i].player_id] });
      i++;
      continue;
    }
    const runStart = i;
    while (i < entries.length && entries[i].kind === "RejectTrade") i++;
    const run = entries.slice(runStart, i);
    const followedByAccept = i < entries.length && entries[i].kind === "AcceptTrade";
    if (run.length > 1 && !followedByAccept) {
      lines.push({
        key: `${key++}`,
        text: `Everyone rejected the trade (${run.map((e) => `Player ${e.player_id}`).join(", ")})`,
        playerIds: run.map((e) => e.player_id),
      });
    } else {
      for (const entry of run) {
        lines.push({ key: `${key++}`, text: describeTrailEntry(entry), playerIds: [entry.player_id] });
      }
    }
  }
  return lines;
}

// "Player 0 +1 WOOL, +1 GRAIN; Player 2 +1 GRAIN" -- who gained what from a
// roll (server/serialize.py's serialize_trail_entry's `production`, itself
// derived from server/bots.py's apply_and_record diffing hands before/
// after). Not new redaction exposure: this is fully derivable by any
// player from public information (the board, everyone's buildings), same
// as the dice roll itself.
function describeProduction(production: Record<string, Record<string, number>>): string {
  return Object.entries(production)
    .map(([playerId, gains]) => {
      const gainsText = Object.entries(gains)
        .map(([resource, amount]) => `+${amount} ${resource}`)
        .join(", ");
      return `Player ${playerId} ${gainsText}`;
    })
    .join("; ");
}
