// Pure filters over the real, server-enumerated PlayRoadBuilding legal
// actions (engine/game.py's _road_building_pairs) -- no client-side
// adjacency logic is invented here, only lookups into what the server
// already offered.

import type { LegalAction } from "./api";

// _road_building_pairs enumerates both (e1, e2) and (e2, e1) whenever both
// edges are individually legal, so every distinct edge_id_1 across the
// PlayRoadBuilding instances is a valid "first road" pick.
export function firstEdgeCandidates(legalActions: LegalAction[]): Set<number> {
  const set = new Set<number>();
  for (const a of legalActions) {
    if (a.kind === "PlayRoadBuilding") set.add(a.edge_id_1 as number);
  }
  return set;
}

// Once the first edge is picked, each PlayRoadBuilding instance with that
// edge_id_1 pins down an exact second edge (and thus an exact action index)
// -- no need to also match edge_id_2, since both orderings were already
// enumerated separately.
export function secondEdgeCandidates(
  legalActions: LegalAction[],
  firstEdge: number,
): Map<number, number> {
  const map = new Map<number, number>();
  for (const a of legalActions) {
    if (a.kind === "PlayRoadBuilding" && a.edge_id_1 === firstEdge) {
      map.set(a.edge_id_2 as number, a.index);
    }
  }
  return map;
}
