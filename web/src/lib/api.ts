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
  // The original proposer's id, when this offer is a counter-offer; null
  // for a fresh ProposeTrade.
  counter_of: number | null;
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

// server/serialize.py's serialize_trail_entry(): one already-applied action
// (the human's own, or a bot's, in application order) -- player_id + kind +
// that kind's own fields, loosely typed like LegalAction since the field
// shape varies per kind. Unlike LegalAction, never has `index` (a trail
// entry isn't a selectable option) and `dice_roll`/`production` are present
// only on a RollDice entry (RollDice itself carries no fields of its own).
// `production`'s keys are player_id as a *string* -- JSON object keys
// always are -- and it's present only when it's non-empty (a 7, or a roll
// matching no settled hex, produces nothing). `trade_offer` is present only
// on AcceptTrade/RejectTrade/CounterTrade -- for the first two, neither
// carries fields either, so the deal (or rejected offer) being responded to
// would otherwise be invisible; for CounterTrade it's the *original* offer
// being countered, distinct from the entry's own give/receive fields.
export interface TrailEntry {
  player_id: number;
  kind: string;
  dice_roll?: [number, number];
  production?: Record<string, Record<string, number>>;
  trade_offer?: TradeOfferView;
  [field: string]: unknown;
}

export interface CreateGameResponse {
  game_id: string;
  geometry: Geometry;
  state: GameStateView;
  action_trail: TrailEntry[];
}

// server/app.py's post_action(): the redacted state plus the trail of
// every action applied while producing it (the human's own action first,
// then any consecutive bot turns server/bots.py's step_bots resolved
// synchronously -- see lib/actionLog.ts for how the frontend paces
// revealing these instead of just jumping straight to the final state).
export interface ActionResponse extends GameStateView {
  action_trail: TrailEntry[];
}

// server/serialize.py's serialize_action(): one legal action, indexed and
// render-hinted. Field shape varies per `kind` (vertex_id, edge_id, hex_id,
// resource fields, ...), so it's a loosely-typed record rather than a
// per-kind union -- the frontend only ever echoes `index` (plus, for
// ProposeTrade/CounterTrade, a constructed bundle) back to the server.
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
): Promise<ActionResponse> {
  return fetch(`${API_BASE}/games/${gameId}/action`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  }).then((r) => parseJsonOrThrow<ActionResponse>(r));
}

export function deleteGame(gameId: string): Promise<{ deleted: boolean }> {
  return fetch(`${API_BASE}/games/${gameId}`, { method: "DELETE" }).then((r) =>
    parseJsonOrThrow<{ deleted: boolean }>(r),
  );
}

// server/serialize.py's serialize_build_costs(): the fixed, RNG-free,
// viewer-independent resource cost of each build kind (ROAD/SETTLEMENT/
// CITY/DEV_CARD), sparse -- a kind's dict only has the resources it
// actually costs. Static and session-independent, unlike Geometry (which
// only comes back from createGame), so callers fetch it once per app
// mount rather than caching it per-game.
export type BuildCosts = Record<string, Record<string, number>>;

export function getBuildCosts(): Promise<BuildCosts> {
  return fetch(`${API_BASE}/build_costs`).then((r) =>
    parseJsonOrThrow<BuildCosts>(r),
  );
}
