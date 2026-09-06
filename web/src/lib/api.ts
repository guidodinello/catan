// Typed fetch wrappers over server/app.py's HTTP API.
//
// These types mirror server/app.py's pydantic models and server/serialize.py's
// output shapes directly (not the plan doc's prose, which predates the
// implementation) -- keep them in sync with those two files, not with each
// other's memory of what they say.

const API_BASE = "/api";

export type SeatKind = "human" | "random" | "stratified_random" | "heuristic";

export interface CreateGameRequest {
  num_players: number;
  seat_kinds: SeatKind[];
  seed?: number | null;
}

export interface GeometryHex {
  hex: [number, number, number];
  vertex_ids: number[];
  edge_ids: number[];
}

export interface GeometryVertex {
  vertex_id: number;
  hexes: [number, number, number][];
  neighbors: number[];
  edges: number[];
}

export interface GeometryEdge {
  edge_id: number;
  vertices: [number, number];
  hexes: [number, number, number][];
}

// server/serialize.py's serialize_geometry(): the fixed, RNG-free board
// topology, sent once per game (engine/board.py's GEOMETRY singleton).
export interface Geometry {
  land_hexes: [number, number, number][];
  water_hexes: [number, number, number][];
  hexes: GeometryHex[];
  vertices: GeometryVertex[];
  edges: GeometryEdge[];
  port_locations: number[];
}

export interface BoardTerrainEntry {
  hex: [number, number, number];
  terrain: string;
}

export interface BoardTokenEntry {
  hex: [number, number, number];
  token: number;
}

export interface BoardPortEntry {
  edge_id: number;
  port_type: string;
}

// server/serialize.py's _serialize_board(): per-game, randomized content.
export interface BoardView {
  terrain: BoardTerrainEntry[];
  tokens: BoardTokenEntry[];
  port_types: BoardPortEntry[];
  robber_hex: [number, number, number];
}

export interface DevCardEntry {
  card_type: string;
  bought_this_turn: boolean;
}

// server/serialize.py's _serialize_player(): fields common to every seat,
// plus either the revealed hand (own seat / spectator) or redacted counts
// (another seat) -- never both, per player_view()'s redaction rule.
export interface PlayerView {
  player_id: number;
  settlement_vertices: number[];
  city_vertices: number[];
  road_edges: number[];
  settlements_remaining: number;
  cities_remaining: number;
  roads_remaining: number;
  played_knights: number;
  played_progress_count: number;
  revealed_vp_cards: number;
  has_played_dev_card_this_turn: boolean;
  victory_points: number;
  // Present only when this seat's hand is revealed to the viewer.
  resources?: Record<string, number>;
  dev_hand?: DevCardEntry[];
  // Present only when this seat's hand is redacted.
  resource_count?: number;
  dev_card_count?: number;
}

export interface DiscardAmountEntry {
  player_id: number;
  amount: number;
}

export interface TradeOfferView {
  proposer: number;
  give: Record<string, number>;
  receive: Record<string, number>;
}

// server/serialize.py's player_view(): the redacted per-turn game state.
export interface GameStateView {
  phase: string;
  current_player: number;
  acting_player: number;
  dice_roll: [number, number] | null;
  bank: Record<string, number>;
  board: BoardView;
  players: PlayerView[];
  longest_road_owner: number | null;
  largest_army_owner: number | null;
  pending_discards: number[];
  discard_amounts: DiscardAmountEntry[];
  trade_offer: TradeOfferView | null;
  trade_responders: number[];
  winner: number | null;
}

export interface CreateGameResponse {
  game_id: string;
  geometry: Geometry;
  state: GameStateView;
}

// server/serialize.py's serialize_action(): one legal action, indexed and
// render-hinted. Field shape varies per `kind` (vertex_id, edge_id, hex_id,
// resource fields, ...), so it's a loosely-typed record rather than a
// per-kind union -- the frontend only ever echoes `index` (plus, for
// ProposeTrade, a constructed bundle) back to the server.
export interface LegalAction {
  index: number;
  kind: string;
  open_ended?: boolean;
  [field: string]: unknown;
}

export interface ActionRequest {
  index: number;
  give?: Record<string, number> | null;
  receive?: Record<string, number> | null;
}

async function parseJsonOrThrow<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const detail = await response
      .json()
      .then((body: unknown) =>
        typeof body === "object" && body !== null && "detail" in body
          ? String((body as { detail: unknown }).detail)
          : response.statusText,
      )
      .catch(() => response.statusText);
    throw new Error(`${response.status} ${response.url}: ${detail}`);
  }
  return response.json() as Promise<T>;
}

export function createGame(
  request: CreateGameRequest,
): Promise<CreateGameResponse> {
  return fetch(`${API_BASE}/games`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  }).then((r) => parseJsonOrThrow<CreateGameResponse>(r));
}

export function getState(
  gameId: string,
  viewer?: number,
): Promise<GameStateView> {
  const query = viewer === undefined ? "" : `?viewer=${viewer}`;
  return fetch(`${API_BASE}/games/${gameId}/state${query}`).then((r) =>
    parseJsonOrThrow<GameStateView>(r),
  );
}

export function getLegalActions(
  gameId: string,
  viewer?: number,
): Promise<LegalAction[]> {
  const query = viewer === undefined ? "" : `?viewer=${viewer}`;
  return fetch(`${API_BASE}/games/${gameId}/legal_actions${query}`).then((r) =>
    parseJsonOrThrow<LegalAction[]>(r),
  );
}

export function postAction(
  gameId: string,
  request: ActionRequest,
): Promise<GameStateView> {
  return fetch(`${API_BASE}/games/${gameId}/action`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  }).then((r) => parseJsonOrThrow<GameStateView>(r));
}

export function deleteGame(gameId: string): Promise<{ deleted: boolean }> {
  return fetch(`${API_BASE}/games/${gameId}`, { method: "DELETE" }).then((r) =>
    parseJsonOrThrow<{ deleted: boolean }>(r),
  );
}
