<script lang="ts">
  import type { Geometry, GameStateView, LegalAction } from "./api";
  import {
    edgeEndpoints,
    hexCorners,
    hexToPixel,
    pointsAttribute,
    vertexPixels,
  } from "./geometry";
  import {
    TERRAIN_ICON,
    PORT_ICON,
    SettlementIcon,
    CityIcon,
    RoadIcon,
    RobberIcon,
  } from "./icons";

  interface Props {
    geometry: Geometry;
    state: GameStateView;
    legalActions?: LegalAction[];
    onSelect?: (index: number) => void;
  }

  const { geometry, state, legalActions = [], onSelect }: Props = $props();

  const SIZE = 50;

  // engine/board.py's Terrain enum names (server/serialize.py's
  // _serialize_board sends `.name`).
  const TERRAIN_COLOR: Record<string, string> = {
    FOREST: "#2d6a4f",
    PASTURE: "#95d5b2",
    FIELD: "#e9c46a",
    HILL: "#bc6c25",
    MOUNTAIN: "#6c757d",
    DESERT: "#e0d8b0",
  };

  // Up to 4 players (engine/game.py's MIN_PLAYERS..MAX_PLAYERS).
  const PLAYER_COLOR = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261"];

  const pixelsByVertexId = $derived(vertexPixels(geometry, SIZE));
  const edgeById = $derived(new Map(geometry.edges.map((e) => [e.edge_id, e])));

  const terrainByHexKey = $derived(
    new Map(state.board.terrain.map((t) => [t.hex.join(","), t.terrain])),
  );
  const tokenByHexKey = $derived(
    new Map(state.board.tokens.map((t) => [t.hex.join(","), t.token])),
  );
  const portByEdgeId = $derived(
    new Map(state.board.port_types.map((p) => [p.edge_id, p.port_type])),
  );
  const robberKey = $derived(state.board.robber_hex.join(","));

  const ownerOfVertex = $derived.by(() => {
    const map = new Map<number, { color: string; city: boolean }>();
    for (const player of state.players) {
      for (const v of player.settlement_vertices) {
        map.set(v, { color: PLAYER_COLOR[player.player_id], city: false });
      }
      for (const v of player.city_vertices) {
        map.set(v, { color: PLAYER_COLOR[player.player_id], city: true });
      }
    }
    return map;
  });

  const ownerOfEdge = $derived.by(() => {
    const map = new Map<number, string>();
    for (const player of state.players) {
      for (const e of player.road_edges) {
        map.set(e, PLAYER_COLOR[player.player_id]);
      }
    }
    return map;
  });

  // Spatial legal actions -- engine/actions.py's PlaceSettlement/PlaceCity
  // (vertex_id), PlaceRoad (edge_id), MoveRobber (hex_id), StealFrom
  // (player_idx) -- keyed by the board location they highlight, per the
  // plan's hybrid interaction model (docs/plans/gui-web-frontend.md).
  // Every other legal-action kind belongs to ActionPanel.svelte instead.
  const settlementSites = $derived.by(() => {
    const map = new Map<number, number>(); // vertex_id -> action index
    for (const a of legalActions) {
      if (a.kind === "PlaceSettlement") map.set(a.vertex_id as number, a.index);
    }
    return map;
  });

  const citySites = $derived.by(() => {
    const map = new Map<number, number>(); // vertex_id -> action index
    for (const a of legalActions) {
      if (a.kind === "PlaceCity") map.set(a.vertex_id as number, a.index);
    }
    return map;
  });

  const roadSites = $derived.by(() => {
    const map = new Map<number, number>(); // edge_id -> action index
    for (const a of legalActions) {
      if (a.kind === "PlaceRoad") map.set(a.edge_id as number, a.index);
    }
    return map;
  });

  const robberSites = $derived.by(() => {
    const map = new Map<string, number>(); // hex key -> action index
    for (const a of legalActions) {
      if (a.kind === "MoveRobber") {
        map.set((a.hex_id as [number, number, number]).join(","), a.index);
      }
    }
    return map;
  });

  // StealFrom targets a player, not a location -- highlighted as that
  // player's existing buildings ("player-tokens" per the plan doc).
  const stealTargetVertices = $derived.by(() => {
    const map = new Map<number, number>(); // vertex_id -> action index
    const indexByPlayer = new Map<number, number>();
    for (const a of legalActions) {
      if (a.kind === "StealFrom") indexByPlayer.set(a.player_idx as number, a.index);
    }
    if (indexByPlayer.size === 0) return map;
    for (const player of state.players) {
      const index = indexByPlayer.get(player.player_id);
      if (index === undefined) continue;
      for (const v of [...player.settlement_vertices, ...player.city_vertices]) {
        map.set(v, index);
      }
    }
    return map;
  });

  function activateOnKey(index: number, event: KeyboardEvent) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onSelect?.(index);
    }
  }

  // Port labels ("2:1 LUMBER") are anchored below their edge-midpoint
  // marker (see the port <text> below, y = mid.y + SIZE * 0.35) and can be
  // fairly wide -- well outside the hex-corner vertices alone, which is
  // why a viewBox padded only around vertex positions clipped labels on
  // boundary ports. Pad around each port label's actual anchor point
  // instead of guessing a single global margin.
  const portLabelPoints = $derived.by(() =>
    geometry.port_locations.flatMap((portEdgeId) => {
      const edge = edgeById.get(portEdgeId);
      if (!edge) return [];
      const [p1, p2] = edgeEndpoints(edge, pixelsByVertexId);
      return [{ x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 + SIZE * 0.35 }];
    }),
  );

  const viewBox = $derived.by(() => {
    const xs = [...pixelsByVertexId.map((p) => p.x), ...portLabelPoints.map((p) => p.x)];
    const ys = [...pixelsByVertexId.map((p) => p.y), ...portLabelPoints.map((p) => p.y)];
    const margin = SIZE;
    // Port label text extends roughly this far past its anchor point
    // (longest label is "2:1 LUMBER" at font-size SIZE * 0.14).
    const labelHalfWidth = SIZE * 0.55;
    const labelHeight = SIZE * 0.2;
    const minX = Math.min(...xs) - margin - labelHalfWidth;
    const maxX = Math.max(...xs) + margin + labelHalfWidth;
    const minY = Math.min(...ys) - margin;
    const maxY = Math.max(...ys) + margin + labelHeight;
    return `${minX} ${minY} ${maxX - minX} ${maxY - minY}`;
  });
</script>

<svg viewBox={viewBox} role="img" aria-label="Catan board">
  {#each geometry.hexes as hex (hex.hex.join(","))}
    {@const key = hex.hex.join(",")}
    {@const terrain = terrainByHexKey.get(key)}
    {@const token = tokenByHexKey.get(key)}
    {@const center = hexToPixel(hex.hex, SIZE)}
    {@const robberIndex = robberSites.get(key)}
    <polygon
      points={pointsAttribute(hexCorners(hex.vertex_ids, pixelsByVertexId))}
      fill={terrain ? TERRAIN_COLOR[terrain] : "#cccccc"}
      stroke="#1d2d1f"
      stroke-width="1"
    />
    {#if terrain && TERRAIN_ICON[terrain]}
      {@const Icon = TERRAIN_ICON[terrain]}
      <g
        class="terrain-icon"
        transform="translate({center.x - SIZE * 0.25}, {center.y - SIZE * 0.72}) scale({SIZE * 0.021})"
        aria-hidden="true"
      >
        <Icon />
      </g>
    {/if}
    {#if token !== undefined}
      <circle cx={center.x} cy={center.y} r={SIZE * 0.28} fill="white" stroke="#333" />
      <text
        x={center.x}
        y={center.y}
        text-anchor="middle"
        dominant-baseline="middle"
        font-size={SIZE * 0.28}
      >
        {token}
      </text>
    {/if}
    {#if key === robberKey}
      {@const robberScale = SIZE * 0.016}
      <g
        class="board-glyph"
        transform="translate({center.x - 12 * robberScale}, {center.y -
          SIZE * 0.05 -
          12 * robberScale}) scale({robberScale})"
      >
        <RobberIcon />
      </g>
    {/if}
    {#if robberIndex !== undefined}
      <polygon
        points={pointsAttribute(hexCorners(hex.vertex_ids, pixelsByVertexId))}
        class="highlight-hex"
        role="button"
        tabindex="0"
        onclick={() => onSelect?.(robberIndex)}
        onkeydown={(e) => activateOnKey(robberIndex, e)}
      />
    {/if}
  {/each}

  {#each geometry.port_locations as portEdgeId (portEdgeId)}
    {@const edge = edgeById.get(portEdgeId)}
    {#if edge}
      {@const [p1, p2] = edgeEndpoints(edge, pixelsByVertexId)}
      {@const mid = { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 }}
      {@const portType = portByEdgeId.get(portEdgeId)}
      {@const PortIcon = portType ? PORT_ICON[portType] : undefined}
      <circle cx={mid.x} cy={mid.y} r={SIZE * 0.15} fill="#a8dadc" stroke="#1d3557" />
      {#if PortIcon}
        {@const portScale = SIZE * 0.011}
        <g
          class="board-glyph"
          transform="translate({mid.x - 12 * portScale}, {mid.y - 12 * portScale}) scale({portScale})"
        >
          <PortIcon />
        </g>
      {/if}
      <text
        x={mid.x}
        y={mid.y + SIZE * 0.35}
        text-anchor="middle"
        dominant-baseline="middle"
        font-size={SIZE * 0.14}
      >
        {portType === "GENERIC" ? "3:1" : `2:1 ${portType}`}
      </text>
    {/if}
  {/each}

  {#each geometry.edges as edge (edge.edge_id)}
    {@const color = ownerOfEdge.get(edge.edge_id)}
    {@const roadIndex = roadSites.get(edge.edge_id)}
    {@const [p1, p2] = edgeEndpoints(edge, pixelsByVertexId)}
    {#if color}
      {@const dx = p2.x - p1.x}
      {@const dy = p2.y - p1.y}
      {@const angleDeg = (Math.atan2(dy, dx) * 180) / Math.PI}
      {@const roadLength = Math.hypot(dx, dy)}
      {@const roadMid = { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 }}
      {@const lengthScale = roadLength / 24}
      {@const thicknessScale = (SIZE * 0.12) / 8}
      <g
        class="board-glyph"
        style="color: {color}"
        transform="translate({roadMid.x}, {roadMid.y}) rotate({angleDeg}) scale({lengthScale}, {thicknessScale}) translate(-12, -4)"
      >
        <RoadIcon />
      </g>
    {/if}
    {#if roadIndex !== undefined}
      <line
        x1={p1.x}
        y1={p1.y}
        x2={p2.x}
        y2={p2.y}
        class="highlight-edge"
        stroke-width={SIZE * 0.12}
        stroke-linecap="round"
        role="button"
        tabindex="0"
        onclick={() => onSelect?.(roadIndex)}
        onkeydown={(e) => activateOnKey(roadIndex, e)}
      />
    {/if}
  {/each}

  {#each pixelsByVertexId as point, vertexId (vertexId)}
    {@const owner = ownerOfVertex.get(vertexId)}
    {@const settlementIndex = settlementSites.get(vertexId)}
    {@const cityIndex = citySites.get(vertexId)}
    {@const stealIndex = stealTargetVertices.get(vertexId)}
    {#if owner}
      {@const buildingScale = owner.city ? SIZE * 0.0183 : SIZE * 0.0117}
      <g
        class="board-glyph"
        style="color: {owner.color}"
        transform="translate({point.x - 12 * buildingScale}, {point.y -
          12 * buildingScale}) scale({buildingScale})"
      >
        {#if owner.city}
          <CityIcon />
        {:else}
          <SettlementIcon />
        {/if}
      </g>
    {/if}
    {#if settlementIndex !== undefined}
      <circle
        cx={point.x}
        cy={point.y}
        r={SIZE * 0.16}
        class="highlight-vertex"
        role="button"
        tabindex="0"
        onclick={() => onSelect?.(settlementIndex)}
        onkeydown={(e) => activateOnKey(settlementIndex, e)}
      />
    {/if}
    {#if cityIndex !== undefined}
      <circle
        cx={point.x}
        cy={point.y}
        r={SIZE * 0.24}
        class="highlight-vertex"
        role="button"
        tabindex="0"
        onclick={() => onSelect?.(cityIndex)}
        onkeydown={(e) => activateOnKey(cityIndex, e)}
      />
    {/if}
    {#if stealIndex !== undefined}
      <circle
        cx={point.x}
        cy={point.y}
        r={SIZE * 0.28}
        class="highlight-vertex steal"
        role="button"
        tabindex="0"
        onclick={() => onSelect?.(stealIndex)}
        onkeydown={(e) => activateOnKey(stealIndex, e)}
      />
    {/if}
  {/each}
</svg>

<style>
  svg {
    width: 100%;
    height: auto;
    max-height: 80vh;
  }

  .terrain-icon,
  .board-glyph {
    pointer-events: none;
  }

  .highlight-hex,
  .highlight-edge,
  .highlight-vertex {
    cursor: pointer;
    fill: color-mix(in srgb, #ffd60a 35%, transparent);
    stroke: #ffd60a;
    stroke-width: 2;
    stroke-dasharray: 4 3;
  }

  .highlight-edge {
    fill: none;
  }

  .highlight-vertex.steal {
    stroke: #d90429;
  }

  .highlight-hex:hover,
  .highlight-edge:hover,
  .highlight-vertex:hover {
    fill: color-mix(in srgb, #ffd60a 60%, transparent);
  }
</style>
