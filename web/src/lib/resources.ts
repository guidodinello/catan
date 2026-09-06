// Single source of truth for the 5 resource types and the terrain that
// produces each -- previously duplicated (with a "keep in sync" comment) in
// both HandSummary.svelte and TradeForm.svelte.

// engine/board.py's Resource enum, in its declared order.
export const RESOURCES = ["LUMBER", "WOOL", "GRAIN", "BRICK", "ORE"] as const;
export type Resource = (typeof RESOURCES)[number];

// engine/board.py's Terrain -> Resource production. DESERT produces
// nothing, hence the nullable value.
export const TERRAIN_RESOURCE: Record<string, Resource | null> = {
  FOREST: "LUMBER",
  PASTURE: "WOOL",
  FIELD: "GRAIN",
  HILL: "BRICK",
  MOUNTAIN: "ORE",
  DESERT: null,
};
