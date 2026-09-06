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
