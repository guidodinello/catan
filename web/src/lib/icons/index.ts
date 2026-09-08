// Map lookups from terrain/resource name to icon component, so callers do a
// map lookup instead of an {#if}/{:else if} chain.

import type { Component } from "svelte";
import type { Resource } from "../resources";

import LumberIcon from "./LumberIcon.svelte";
import WoolIcon from "./WoolIcon.svelte";
import GrainIcon from "./GrainIcon.svelte";
import BrickIcon from "./BrickIcon.svelte";
import OreIcon from "./OreIcon.svelte";
import DesertIcon from "./DesertIcon.svelte";
import PortGenericIcon from "./PortGenericIcon.svelte";
import KnightIcon from "./KnightIcon.svelte";
import RoadBuildingIcon from "./RoadBuildingIcon.svelte";
import YearOfPlentyIcon from "./YearOfPlentyIcon.svelte";
import MonopolyIcon from "./MonopolyIcon.svelte";
import VictoryPointIcon from "./VictoryPointIcon.svelte";

export const RESOURCE_ICON: Record<Resource, Component> = {
  LUMBER: LumberIcon,
  WOOL: WoolIcon,
  GRAIN: GrainIcon,
  BRICK: BrickIcon,
  ORE: OreIcon,
};

// engine/board.py's Terrain enum names -- includes DESERT, which has no
// resource icon of its own.
export const TERRAIN_ICON: Record<string, Component> = {
  FOREST: LumberIcon,
  PASTURE: WoolIcon,
  FIELD: GrainIcon,
  HILL: BrickIcon,
  MOUNTAIN: OreIcon,
  DESERT: DesertIcon,
};

// engine/board.py's PortType enum names -- a 2:1 port per resource, plus the
// GENERIC (3:1) port, which has no resource of its own. Keyed by plain
// string (like TERRAIN_ICON) since server/serialize.py sends `.name` and the
// wire type (BoardPortEntry.port_type in api.ts) is just `string`.
export const PORT_ICON: Record<string, Component> = {
  ...RESOURCE_ICON,
  GENERIC: PortGenericIcon,
};

// Owner-tintable glyphs (currentColor -- see SettlementIcon.svelte), always
// the same component per kind, so no lookup map is needed; export directly.
export { default as SettlementIcon } from "./SettlementIcon.svelte";
export { default as CityIcon } from "./CityIcon.svelte";
export { default as RoadIcon } from "./RoadIcon.svelte";

// Neutral (non-tinted) glyph.
export { default as RobberIcon } from "./RobberIcon.svelte";

// engine/state.py's DevCardType enum names -- the 5 real card faces, used
// for a fully-revealed hand (the viewer's own unplayed dev_hand) and for
// individually-unambiguous played counters (played_knights,
// revealed_vp_cards). Never used to guess an opponent's unplayed card, or
// which progress type was played -- see DevCardBackIcon/ProgressCardIcon.
export const DEV_CARD_ICON: Record<string, Component> = {
  KNIGHT: KnightIcon,
  ROAD_BUILDING: RoadBuildingIcon,
  YEAR_OF_PLENTY: YearOfPlentyIcon,
  MONOPOLY: MonopolyIcon,
  VICTORY_POINT: VictoryPointIcon,
};

// Also exported directly -- DevCardStrip.svelte's "Played" row uses these
// two individually (a played Knight or a revealed VP card is never
// type-ambiguous), without going through the DevCardType-keyed map above.
export { KnightIcon, VictoryPointIcon };

// Generic face-down cover for another seat's unplayed hand (real subtype
// unknown) and generic face-up cover for a played progress card (subtype
// not tracked by the engine for any seat, including its own owner).
export { default as DevCardBackIcon } from "./DevCardBackIcon.svelte";
export { default as ProgressCardIcon } from "./ProgressCardIcon.svelte";
