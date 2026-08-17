import {
  draftAuthorityStatus,
  hasDescriptiveDraftAuthority,
  type DraftAuthorityStatus,
} from "./pack";
import type {
  MatchSummary,
  PackManifest,
  PlayerChampionRecord,
  PlayerRating,
  PlayerRecord,
  ProfileGame,
  TeamRating,
  TeamRecord,
} from "./pack";
import { supabaseConfig } from "./serverPack";
import type {
  RegionalView,
  StructuralSimilarity,
  TierBucket,
  TierRow,
  TierScope,
} from "./tierBoard";

export const QUERY_API_SCHEMA = "scryglass:query-api:v1" as const;
export const PUBLIC_RESPONSE_MAX_BYTES = 500 * 1024;
export const PUBLIC_DRAFT_SCORE_RESULT_SCHEMA = "scryglass:public-draft-score-result:v1" as const;
export const PROMOTED_DRAFT_RESULTS_SCHEMA = "scryglass:promoted-draft-results:v1" as const;

export type PublicDraftScoreResult = {
  schema_version: typeof PUBLIC_DRAFT_SCORE_RESULT_SCHEMA;
  authority: "promoted";
  release_id: string;
  model_version: string;
  receipt_sha256: string;
  evidence_window: { start: string; end: string };
  match_win_probability: { Blue: number; Red: number };
  controlled_draft_score: {
    model_units: number;
    edge_percentage_points: number;
    stronger_draft: "Blue" | "Red" | "Even";
    explanation: string;
    method: "role_matched_champion_swap";
    intervention_receipt_sha256: string;
    isolated_blue_draft_probability: number;
    fixed_strength_blue_win_probability: number;
  };
  side_recommendation: "Blue" | "Red";
};

export type PromotedDraftResultsPayload = {
  schema_version: typeof PROMOTED_DRAFT_RESULTS_SCHEMA;
  authority: "promoted";
  release_id: string;
  model_version: string;
  receipt_sha256: string;
  results: Record<string, PublicDraftScoreResult>;
};

const SHA256_PATTERN = /^[a-f0-9]{64}$/;
const exactKeys = (value: object, keys: readonly string[]) => {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  return actual.length === expected.length && actual.every((key, index) => key === expected[index]);
};

/** Validate one promoted public result before any product surface renders it. */
export function validatePromotedDraftScoreResult(
  value: unknown,
  manifest: PackManifest,
): asserts value is PublicDraftScoreResult {
  validateDraftResponse(value, manifest);
  if (!value || typeof value !== "object") throw new Error("Promoted Draft result is malformed");
  const result = value as Partial<PublicDraftScoreResult>;
  if (
    !exactKeys(value, [
      "schema_version", "authority", "release_id", "model_version", "receipt_sha256",
      "evidence_window", "match_win_probability", "controlled_draft_score", "side_recommendation",
    ])
    || result.schema_version !== PUBLIC_DRAFT_SCORE_RESULT_SCHEMA
    || result.authority !== "promoted"
    || result.release_id !== manifest.pack_id
    || typeof result.model_version !== "string"
    || !result.model_version.trim()
    || typeof result.receipt_sha256 !== "string"
    || !SHA256_PATTERN.test(result.receipt_sha256)
    || result.receipt_sha256 !== manifest.draft_authority?.receipt_sha256
    || result.model_version !== manifest.draft_authority?.model_version
  ) {
    throw new Error("Promoted Draft result is not release-bound");
  }
  const blue = result.match_win_probability?.Blue;
  const red = result.match_win_probability?.Red;
  if (
    !result.match_win_probability
    || !exactKeys(result.match_win_probability, ["Blue", "Red"])
    || typeof blue !== "number"
    || typeof red !== "number"
    || !Number.isFinite(blue)
    || !Number.isFinite(red)
    || blue < 0
    || blue > 1
    || red < 0
    || red > 1
    || Math.abs(blue + red - 1) > 1e-9
  ) {
    throw new Error("Promoted Draft probabilities are invalid");
  }
  const score = result.controlled_draft_score;
  if (
    !score
    || !exactKeys(score, [
      "model_units", "edge_percentage_points", "stronger_draft", "explanation", "method",
      "intervention_receipt_sha256", "isolated_blue_draft_probability",
      "fixed_strength_blue_win_probability",
    ])
    || typeof score.model_units !== "number"
    || !Number.isFinite(score.model_units)
    || typeof score.edge_percentage_points !== "number"
    || !Number.isFinite(score.edge_percentage_points)
    || Math.abs(score.edge_percentage_points) > 100
    || !["Blue", "Red", "Even"].includes(String(score.stronger_draft))
    || typeof score.explanation !== "string"
    || !score.explanation.trim()
    || score.method !== "role_matched_champion_swap"
    || typeof score.intervention_receipt_sha256 !== "string"
    || !SHA256_PATTERN.test(score.intervention_receipt_sha256)
    || typeof score.isolated_blue_draft_probability !== "number"
    || !Number.isFinite(score.isolated_blue_draft_probability)
    || score.isolated_blue_draft_probability < 0
    || score.isolated_blue_draft_probability > 1
    || typeof score.fixed_strength_blue_win_probability !== "number"
    || !Number.isFinite(score.fixed_strength_blue_win_probability)
    || score.fixed_strength_blue_win_probability < 0
    || score.fixed_strength_blue_win_probability > 1
  ) {
    throw new Error("Controlled Draft Score is invalid");
  }
  const expectedDraftSide = score.model_units > 0
    ? "Blue"
    : score.model_units < 0
      ? "Red"
      : "Even";
  if (
    score.stronger_draft !== expectedDraftSide
    || ((score.model_units === 0) !== (score.edge_percentage_points === 0))
    || score.model_units * score.edge_percentage_points < 0
  ) {
    throw new Error("Controlled Draft Score direction is inconsistent");
  }
  if (result.side_recommendation !== (blue >= red ? "Blue" : "Red")) {
    throw new Error("Public side recommendation conflicts with probability");
  }
  const window = result.evidence_window;
  if (
    !window
    || !exactKeys(window, ["start", "end"])
    || typeof window.start !== "string"
    || typeof window.end !== "string"
    || !Number.isFinite(Date.parse(window.start))
    || !Number.isFinite(Date.parse(window.end))
    || Date.parse(window.start) >= Date.parse(window.end)
  ) {
    throw new Error("Promoted Draft evidence window is invalid");
  }
}

/** Validate the complete promoted result asset before exposing any row. */
export function validatePromotedDraftResultsPayload(
  value: unknown,
  manifest: PackManifest,
): asserts value is PromotedDraftResultsPayload {
  if (!value || typeof value !== "object") throw new Error("Promoted Draft result asset is malformed");
  const payload = value as Partial<PromotedDraftResultsPayload>;
  if (
    !exactKeys(value, ["schema_version", "authority", "release_id", "model_version", "receipt_sha256", "results"])
    || payload.schema_version !== PROMOTED_DRAFT_RESULTS_SCHEMA
    || payload.authority !== "promoted"
    || payload.release_id !== manifest.pack_id
    || payload.model_version !== manifest.draft_authority?.model_version
    || payload.receipt_sha256 !== manifest.draft_authority?.receipt_sha256
    || !payload.results
    || typeof payload.results !== "object"
    || Array.isArray(payload.results)
    || Object.keys(payload.results).length === 0
    || Object.keys(payload.results).length > 10_000
  ) {
    throw new Error("Promoted Draft result asset is not release-bound");
  }
  for (const [gameUid, result] of Object.entries(payload.results)) {
    if (!gameUid) throw new Error("Promoted Draft result game ID is empty");
    validatePromotedDraftScoreResult(result, manifest);
  }
}
const RPC_TIMEOUT_MS = 5_000;
const PUBLIC_ROW_LIMIT = 20;
const PUBLIC_RATINGS_ROW_LIMIT = 100;
const PLAYER_PROFILE_CACHE_MAX_ENTRIES = 256;
const PLAYER_PROFILE_CACHE_MAX_BYTES = 8 * 1024 * 1024;
const PLAYER_PROFILE_CACHE_MAX_ENTRY_BYTES = 256 * 1024;
const PLAYER_PROFILE_CACHE_TTL_MS = 30_000;

type PlayerProfileCacheEntry = {
  promise: Promise<PlayerProfileQuery>;
  expiresAt: number;
  sizeBytes: number;
};

const playerProfileCache = new Map<string, PlayerProfileCacheEntry>();
let playerProfileCacheBytes = 0;

export type QueryApiEnvelope<T> = {
  schema_version: typeof QUERY_API_SCHEMA;
  release_id: string;
  authority?: DraftAuthorityStatus;
  rows: T[];
  limit: number;
  offset: number;
  total: number;
};

export type PlayerChampionQueryEnvelope = QueryApiEnvelope<PlayerChampionQueryRow> & {
  median_reference?: number | null;
  mean_reference?: number | null;
};

export type PlayerChampionOrder =
  | "best" | "worst"
  | "games_desc" | "games_asc"
  | "rating_desc" | "rating_asc"
  | "win_rate_desc" | "win_rate_asc"
  | "median" | "mean";

export type RatingQueryRow = {
  row_key: string;
  name: string;
  role: string | null;
  team: string | null;
  league: string | null;
  tier: string | null;
  active: boolean;
  rating: number | null;
  adjusted_rating: number | null;
  games: number;
  wins: number | null;
  win_rate: number | null;
  movement: number | null;
  grade_a_games?: number;
  grade_games?: number;
  payload: {
    rating?: PlayerRating | TeamRating;
    record?: PlayerRecord | TeamRecord;
    weekly?: Record<string, unknown>;
    metadata?: Record<string, unknown>;
    grade_a_games?: number;
    grade_games?: number;
    recent_form?: number | null;
  };
};

export type RatingFacets = {
  schema_version: typeof QUERY_API_SCHEMA;
  release_id: string;
  leagues: string[];
  tiers: string[];
  roles: string[];
  min_games: number;
  max_games: number;
};

export type PlayerChampionQueryRow = {
  player_id: string;
  player: string;
  champion_id: string;
  champion: string;
  role: string | null;
  team: string | null;
  league: string | null;
  tier: string | null;
  active: boolean;
  rating: number | null;
  adjusted_rating: number | null;
  games: number;
  wins: number;
  win_rate: number | null;
  score: number | null;
  tier_score?: number | null;
  median_distance?: number | null;
  mean_distance?: number | null;
  payload: {
    record?: PlayerChampionRecord;
    champion_image_url?: string | null;
  };
};

export type MatchQueryRow = {
  game_id: string;
  played_at: string;
  year: number;
  league: string | null;
  tier: string | null;
  blue_team: string;
  red_team: string;
  blue_win: 0 | 1;
  champions: string[];
  payload: ProfileGame & { schema_version?: string };
};

export type MatchQueryEnvelope = QueryApiEnvelope<MatchQueryRow> & {
  champion_images: Record<string, string>;
};

export type TeamDraftMetric = {
  draft_edge: number;
  games: number;
  positive_edge_rate?: number | null;
  scope?: "whole_archive" | "profile_window" | string | null;
};

export type PlayerDraftMetric = {
  best_available_rate: number | null;
  games: number;
  pick_contribution?: number | null;
  pool_definition?: string | null;
  ban_coverage?: number | null;
  scope?: "whole_archive" | "profile_window" | string | null;
};

export type MatchFacets = {
  schema_version: typeof QUERY_API_SCHEMA;
  release_id: string;
  tiers: string[];
  years: number[];
  months: string[];
  teams: string[];
  leagues: string[];
};

export type PlayerProfileQuery = {
  schema_version: typeof QUERY_API_SCHEMA;
  release_id: string;
  authority?: DraftAuthorityStatus;
  draft_metric?: PlayerDraftMetric | null;
  row: RatingQueryRow | null;
  team_row: RatingQueryRow | null;
  champions: PlayerChampionQueryRow[];
  recent_games: MatchQueryRow[];
  champion_images: Record<string, string>;
  standing: {
    tier_rank: number;
    tier_total: number;
    role_rank: number;
    role_total: number;
  } | null;
};

export type TeamProfileRosterRow = RatingQueryRow & {
  role_rank: number;
  role_total: number;
};

export type TeamProfileQuery = {
  schema_version: typeof QUERY_API_SCHEMA;
  release_id: string;
  authority?: DraftAuthorityStatus;
  draft_metric?: TeamDraftMetric | null;
  row: RatingQueryRow | null;
  roster: TeamProfileRosterRow[];
  recent_games: MatchQueryRow[];
  champion_images: Record<string, string>;
  standing: { tier_rank: number; tier_total: number } | null;
};

export type QueryEntities = {
  schema_version: typeof QUERY_API_SCHEMA;
  release_id: string;
  players: string[];
  teams: string[];
  champions: string[];
  leagues: string[];
  aliases: Array<{
    kind: "player" | "team" | "champion";
    alias: string;
    name: string;
  }>;
};

export type TierQueryRow = {
  row_key: string;
  kind: "champion" | "player" | "team";
  name: string;
  patch: string | null;
  region: string | null;
  league: string | null;
  tier: string | null;
  role: string | null;
  rank: number;
  score: number | null;
  played_maps: number;
  payload: Record<string, unknown>;
};

export type ChampionAggregateQueryRow = {
  champion_id: string;
  champion: string;
  games: number;
  wins: number;
  win_rate: number | null;
  players: number;
};

export type TierFacetScope = {
  scope_id: string;
  patch: string;
  role: string;
  row_count: number;
  regions?: RegionalView[];
  response_matrix_available: boolean;
};

export type TierFacets = {
  schema_version: typeof QUERY_API_SCHEMA;
  release_id: string;
  options: {
    patches: string[];
    roles: string[];
    regions: string[];
    leagues: string[];
    tiers: string[];
    tier_buckets: TierBucket[];
  };
  scopes: TierFacetScope[];
};

export type TierScopeQuery = {
  schema_version: typeof QUERY_API_SCHEMA;
  release_id: string;
  scope: TierScope | null;
  rows: TierRow[];
  structural_similarity: StructuralSimilarity | null;
  champion_images: Record<string, string>;
};

type RpcOptions = {
  manifest: PackManifest;
  cache?: RequestCache;
  signal?: AbortSignal;
};

function safeArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string" && item.length <= 100)
    : [];
}

function safeIntegerArray(value: unknown): number[] {
  return Array.isArray(value)
    ? value.filter((item): item is number => Number.isInteger(item))
    : [];
}

export function boundedRowLimit(
  value: number | undefined,
  fallback: number,
  maximum = PUBLIC_ROW_LIMIT,
): number {
  if (!Number.isInteger(value)) return fallback;
  return Math.min(Math.max(Number(value), 1), maximum);
}

async function readBoundedJson(response: Response): Promise<unknown> {
  const declared = Number(response.headers.get("content-length"));
  if (Number.isFinite(declared) && declared > PUBLIC_RESPONSE_MAX_BYTES) {
    throw new Error("Public query response exceeds 500 KB");
  }
  const reader = response.body?.getReader();
  if (!reader) throw new Error("Public query response has no body");
  const chunks: Uint8Array[] = [];
  let bytes = 0;
  while (true) {
    const chunk = await reader.read();
    if (chunk.done) break;
    bytes += chunk.value.byteLength;
    if (bytes > PUBLIC_RESPONSE_MAX_BYTES) {
      await reader.cancel("public query response exceeds 500 KB");
      throw new Error("Public query response exceeds 500 KB");
    }
    chunks.push(chunk.value);
  }
  const raw = new Uint8Array(bytes);
  let offset = 0;
  for (const chunk of chunks) {
    raw.set(chunk, offset);
    offset += chunk.byteLength;
  }
  try {
    return JSON.parse(new TextDecoder().decode(raw)) as unknown;
  } catch {
    throw new Error("Public query response is invalid JSON");
  }
}

function unwrapRpc(value: unknown): unknown {
  if (Array.isArray(value) && value.length === 1) {
    const row = value[0];
    if (row && typeof row === "object") {
      const entries = Object.entries(row as Record<string, unknown>);
      if (entries.length === 1 && entries[0][1] && typeof entries[0][1] === "object") {
        return entries[0][1];
      }
    }
  }
  return value;
}

const DRAFT_PROBABILITY_KEYS = new Set([
  "draft_probability",
  "draft_win_share",
  "average_win_share",
  "match_win_probability",
  "p_blue",
  "p_red",
  "probability",
]);

const PERMANENTLY_FORBIDDEN_PUBLIC_KEYS = new Set([
  "bet",
  "betting",
  "ev",
  "expected_value",
  "fair_odds",
  "odds",
  "stake",
  "wager",
]);

const DRAFT_RECOMMENDATION_KEYS = new Set([
  "recommendation",
  "recommended_side",
  "side_recommendation",
]);

function containsDraftProbability(value: unknown): boolean {
  if (Array.isArray(value)) return value.some(containsDraftProbability);
  if (!value || typeof value !== "object") return false;
  return Object.entries(value as Record<string, unknown>).some(([key, child]) => (
    DRAFT_PROBABILITY_KEYS.has(key.toLowerCase()) || containsDraftProbability(child)
  ));
}

function containsAnyKey(value: unknown, keys: ReadonlySet<string>): boolean {
  if (Array.isArray(value)) return value.some((entry) => containsAnyKey(entry, keys));
  if (!value || typeof value !== "object") return false;
  return Object.entries(value as Record<string, unknown>).some(([key, child]) => (
    keys.has(key.toLowerCase()) || containsAnyKey(child, keys)
  ));
}

function containsKey(value: unknown, keyName: string): boolean {
  if (Array.isArray(value)) return value.some((entry) => containsKey(entry, keyName));
  if (!value || typeof value !== "object") return false;
  return Object.entries(value as Record<string, unknown>).some(([key, child]) => (
    key.toLowerCase() === keyName || containsKey(child, keyName)
  ));
}

function responseAuthority(value: Record<string, unknown>): DraftAuthorityStatus | null {
  const direct = value.authority;
  if (direct === "unavailable" || direct === "descriptive" || direct === "promoted") return direct;
  const nested = value.draft_authority;
  if (!nested || typeof nested !== "object") return null;
  const record = nested as Record<string, unknown>;
  const authority = record.authority ?? record.status;
  return authority === "unavailable" || authority === "descriptive" || authority === "promoted"
    ? authority
    : null;
}

function validateDraftResponse(value: unknown, manifest: PackManifest): void {
  if (!value || typeof value !== "object") return;
  const record = value as Record<string, unknown>;
  const authority = responseAuthority(record);
  if (containsAnyKey(value, PERMANENTLY_FORBIDDEN_PUBLIC_KEYS)) {
    throw new Error("Public Draft response contains a permanently forbidden betting field");
  }
  if (authority !== "promoted" && containsDraftProbability(value)) {
    throw new Error("Public Draft response contains probability fields without promoted authority");
  }
  if (authority !== "promoted" && containsAnyKey(value, DRAFT_RECOMMENDATION_KEYS)) {
    throw new Error("Public Draft response contains a recommendation without promoted authority");
  }
  const declared = draftAuthorityStatus(manifest);
  if (
    authority
    && authority !== declared
    && !(authority === "descriptive" && hasDescriptiveDraftAuthority(manifest))
  ) {
    throw new Error("Public Draft response has an unbound authority");
  }
  if (containsKey(value, "draft_metric") || containsKey(value, "draft_pool") || containsKey(value, "draft_contribution")) {
    if (authority !== "descriptive" || !hasDescriptiveDraftAuthority(manifest)) {
      throw new Error("Public Draft response is missing descriptive release authority");
    }
  }
}

/** Validate the public Draft fields before a query response reaches a page. */
export function validatePublicDraftResponse(value: unknown, manifest: PackManifest): void {
  validateDraftResponse(value, manifest);
}

async function publicRpc<T>(
  name: string,
  parameters: Record<string, unknown>,
  { manifest, cache = "no-store", signal }: RpcOptions,
): Promise<T> {
  if (!queryApiAvailable(manifest)) throw new Error("The bounded public query API is unavailable");
  const config = supabaseConfig();
  if (!config) throw new Error("Supabase public data is not configured");
  const response = await fetch(`${config.url}/rest/v1/rpc/${name}`, {
    method: "POST",
    headers: {
      Accept: "application/json",
      apikey: config.publishableKey,
      "content-type": "application/json",
    },
    body: JSON.stringify(parameters),
    cache,
    signal: signal
      ? AbortSignal.any([signal, AbortSignal.timeout(RPC_TIMEOUT_MS)])
      : AbortSignal.timeout(RPC_TIMEOUT_MS),
  });
  if (!response.ok) throw new Error(`Public query ${name} returned ${response.status}`);
  const value = unwrapRpc(await readBoundedJson(response));
  if (!value || typeof value !== "object") throw new Error(`Public query ${name} is malformed`);
  const envelope = value as { schema_version?: unknown; release_id?: unknown };
  if (envelope.schema_version !== QUERY_API_SCHEMA || envelope.release_id !== manifest.pack_id) {
    throw new Error(`Public query ${name} has a different release`);
  }
  validateDraftResponse(value, manifest);
  return value as T;
}

export function queryApiAvailable(manifest: PackManifest): boolean {
  return manifest.query_api?.schema_version === QUERY_API_SCHEMA
    && manifest.query_api.status === "available";
}

export async function getRatings(
  manifest: PackManifest,
  input: {
    kind: "players" | "teams";
    leagues?: string[];
    tiers?: string[];
    roles?: string[];
    active?: boolean;
    names?: string[];
    search?: string;
    teams?: string[];
    order?:
      | "rating_desc" | "rating_asc"
      | "movement_desc"
      | "games_desc" | "games_asc"
      | "wins_desc" | "wins_asc"
      | "win_rate_desc" | "win_rate_asc"
      | "grade_a_desc" | "grade_a_asc"
      | "name_asc";
    minGames?: number;
    limit?: number;
    offset?: number;
  },
  signal?: AbortSignal,
): Promise<QueryApiEnvelope<RatingQueryRow>> {
  const maximumRows = input.names?.length ? PUBLIC_ROW_LIMIT : PUBLIC_RATINGS_ROW_LIMIT;
  return publicRpc("get_scryglass_ratings", {
    p_kind: input.kind,
    p_leagues: input.leagues?.length ? input.leagues : null,
    p_tiers: input.tiers?.length ? input.tiers : null,
    p_roles: input.roles?.length ? input.roles : null,
    p_teams: input.teams?.length ? input.teams : null,
    p_active: input.active ?? null,
    p_names: input.names?.length ? input.names.slice(0, PUBLIC_ROW_LIMIT) : null,
    p_search: input.search?.trim() || null,
    p_order: input.order ?? "rating_desc",
    p_min_games: input.minGames ?? 0,
    p_limit: boundedRowLimit(input.limit, maximumRows, maximumRows),
    p_offset: input.offset ?? 0,
  }, { manifest, signal });
}

export async function getRatingFacets(
  manifest: PackManifest,
  kind: "players" | "teams",
  tiers: string[] = [],
  signal?: AbortSignal,
): Promise<RatingFacets> {
  const result = await publicRpc<RatingFacets>("get_scryglass_rating_facets", {
    p_kind: kind,
    p_tiers: tiers.length ? tiers : null,
  }, { manifest, signal });
  return {
    ...result,
    leagues: safeArray(result.leagues),
    tiers: safeArray(result.tiers),
    roles: safeArray(result.roles),
  };
}

export function getPlayerProfile(
  manifest: PackManifest,
  name: string,
  signal?: AbortSignal,
): Promise<PlayerProfileQuery> {
  if (signal) {
    return publicRpc("get_scryglass_player_profile", { p_name: name }, { manifest, signal });
  }
  const normalizedName = name.normalize("NFKC").trim().toLocaleLowerCase();
  const cacheKey = `${manifest.pack_id}:${normalizedName}`;
  const now = Date.now();
  const cached = playerProfileCache.get(cacheKey);
  if (cached) {
    if (cached.expiresAt > now) {
      playerProfileCache.delete(cacheKey);
      playerProfileCache.set(cacheKey, cached);
      return cached.promise;
    }
    removePlayerProfileCacheEntry(cacheKey, cached);
  }
  const pending = publicRpc<PlayerProfileQuery>(
    "get_scryglass_player_profile",
    { p_name: name },
    { manifest },
  );
  const entry: PlayerProfileCacheEntry = {
    promise: pending,
    expiresAt: now + PLAYER_PROFILE_CACHE_TTL_MS,
    sizeBytes: PLAYER_PROFILE_CACHE_MAX_ENTRY_BYTES,
  };
  playerProfileCache.set(cacheKey, entry);
  playerProfileCacheBytes += entry.sizeBytes;
  void pending.then(
    (value) => finalizePlayerProfileCacheEntry(cacheKey, entry, value),
    () => removePlayerProfileCacheEntry(cacheKey, entry),
  );
  trimPlayerProfileCache();
  return pending;
}

function removePlayerProfileCacheEntry(key: string, entry: PlayerProfileCacheEntry): void {
  if (playerProfileCache.get(key) !== entry) return;
  playerProfileCache.delete(key);
  playerProfileCacheBytes = Math.max(0, playerProfileCacheBytes - entry.sizeBytes);
}

function finalizePlayerProfileCacheEntry(
  key: string,
  entry: PlayerProfileCacheEntry,
  value: PlayerProfileQuery,
): void {
  if (playerProfileCache.get(key) !== entry) return;
  if (entry.expiresAt <= Date.now()) {
    removePlayerProfileCacheEntry(key, entry);
    return;
  }
  let sizeBytes: number;
  try {
    sizeBytes = new TextEncoder().encode(JSON.stringify(value)).byteLength;
  } catch {
    removePlayerProfileCacheEntry(key, entry);
    return;
  }
  if (sizeBytes > PLAYER_PROFILE_CACHE_MAX_ENTRY_BYTES) {
    removePlayerProfileCacheEntry(key, entry);
    return;
  }
  playerProfileCacheBytes += sizeBytes - entry.sizeBytes;
  entry.sizeBytes = sizeBytes;
  trimPlayerProfileCache();
}

function trimPlayerProfileCache(): void {
  while (
    playerProfileCache.size > PLAYER_PROFILE_CACHE_MAX_ENTRIES
    || playerProfileCacheBytes > PLAYER_PROFILE_CACHE_MAX_BYTES
  ) {
    const oldest = playerProfileCache.entries().next().value as [string, PlayerProfileCacheEntry] | undefined;
    if (!oldest) break;
    removePlayerProfileCacheEntry(oldest[0], oldest[1]);
  }
}

export function getTeamProfile(
  manifest: PackManifest,
  name: string,
  signal?: AbortSignal,
): Promise<TeamProfileQuery> {
  return publicRpc("get_scryglass_team_profile", { p_name: name }, { manifest, signal });
}

export function getMatches(
  manifest: PackManifest,
  input: {
    leagues?: string[];
    tiers?: string[];
    team?: string;
    champion?: string;
    years?: number[];
    from?: string;
    to?: string;
    before?: string;
    limit?: number;
    offset?: number;
  } = {},
  signal?: AbortSignal,
): Promise<MatchQueryEnvelope> {
  return publicRpc("get_scryglass_matches", {
    p_leagues: input.leagues?.length ? input.leagues : null,
    p_tiers: input.tiers?.length ? input.tiers : null,
    p_team: input.team?.trim() || null,
    p_champion: input.champion?.trim() || null,
    p_years: input.years?.length ? input.years : null,
    p_from: input.from ?? null,
    p_to: input.to ?? null,
    p_before: input.before ?? null,
    p_limit: boundedRowLimit(input.limit, PUBLIC_ROW_LIMIT),
    p_offset: input.offset ?? 0,
  }, { manifest, signal });
}

export async function getMatchFacets(
  manifest: PackManifest,
  input: {
    tiers?: string[];
    years?: number[];
    from?: string;
    to?: string;
    team?: string;
  } = {},
  signal?: AbortSignal,
): Promise<MatchFacets> {
  const result = await publicRpc<MatchFacets>("get_scryglass_match_facets", {
    p_tiers: input.tiers?.length ? input.tiers : null,
    p_years: input.years?.length ? input.years : null,
    p_from: input.from ?? null,
    p_to: input.to ?? null,
    p_team: input.team?.trim() || null,
  }, { manifest, signal });
  return {
    ...result,
    tiers: safeArray(result.tiers),
    years: safeIntegerArray(result.years),
    months: safeArray(result.months),
    teams: safeArray(result.teams),
    leagues: safeArray(result.leagues),
  };
}

export function getMatch(manifest: PackManifest, gameId: string, signal?: AbortSignal): Promise<{
  schema_version: typeof QUERY_API_SCHEMA;
  release_id: string;
  row: MatchQueryRow | null;
  champion_images: Record<string, string>;
}> {
  return publicRpc("get_scryglass_match", { p_game_id: gameId }, { manifest, signal });
}

export function getPlayerChampions(
  manifest: PackManifest,
  input: {
    player?: string;
    champion?: string;
    leagues?: string[];
    tiers?: string[];
    roles?: string[];
    teams?: string[];
    active?: boolean;
    minGames?: number;
    order?: PlayerChampionOrder;
    limit?: number;
    offset?: number;
  } = {},
  signal?: AbortSignal,
): Promise<PlayerChampionQueryEnvelope> {
  return publicRpc("get_scryglass_player_champions", {
    p_player: input.player?.trim() || null,
    p_champion: input.champion?.trim() || null,
    p_leagues: input.leagues?.length ? input.leagues : null,
    p_tiers: input.tiers?.length ? input.tiers : null,
    p_roles: input.roles?.length ? input.roles : null,
    p_teams: input.teams?.length ? input.teams : null,
    p_active: input.active ?? null,
    p_min_games: input.minGames ?? 5,
    p_order: input.order ?? "best",
    p_limit: boundedRowLimit(input.limit, PUBLIC_ROW_LIMIT),
    p_offset: input.offset ?? 0,
  }, { manifest, signal });
}

export function getQueryEntities(manifest: PackManifest, signal?: AbortSignal): Promise<QueryEntities> {
  return publicRpc("get_scryglass_query_entities", {}, { manifest, signal });
}

export function getTierRows(
  manifest: PackManifest,
  input: {
    kind: "champion" | "player" | "team";
    patches?: string[];
    regions?: string[];
    leagues?: string[];
    tiers?: string[];
    roles?: string[];
    search?: string;
    minGames?: number;
    order?: "rank_asc" | "rank_desc" | "score_desc" | "score_asc" | "name_asc";
    limit?: number;
    offset?: number;
  },
  signal?: AbortSignal,
): Promise<QueryApiEnvelope<TierQueryRow>> {
  return publicRpc("get_scryglass_tier_rows", {
    p_kind: input.kind,
    p_patches: input.patches?.length ? input.patches : null,
    p_regions: input.regions?.length ? input.regions : null,
    p_leagues: input.leagues?.length ? input.leagues : null,
    p_tiers: input.tiers?.length ? input.tiers : null,
    p_roles: input.roles?.length ? input.roles : null,
    p_search: input.search?.trim() || null,
    p_min_games: input.minGames ?? 0,
    p_order: input.order ?? "rank_asc",
    p_limit: boundedRowLimit(input.limit, PUBLIC_ROW_LIMIT),
    p_offset: input.offset ?? 0,
  }, { manifest, signal });
}

export async function getTierFacets(manifest: PackManifest, signal?: AbortSignal): Promise<TierFacets> {
  const result = await publicRpc<TierFacets>("get_scryglass_tier_facets", {}, { manifest, signal });
  return {
    ...result,
    options: {
      patches: safeArray(result.options?.patches),
      roles: safeArray(result.options?.roles),
      regions: safeArray(result.options?.regions),
      leagues: safeArray(result.options?.leagues),
      tiers: safeArray(result.options?.tiers),
      tier_buckets: safeArray(result.options?.tier_buckets) as TierBucket[],
    },
    scopes: Array.isArray(result.scopes) ? result.scopes : [],
  };
}

export function getTierScope(
  manifest: PackManifest,
  input: {
    patch: string;
    role?: string;
    region?: string;
    league?: string;
    tier?: string;
    similarityLimit?: number;
  },
  signal?: AbortSignal,
): Promise<TierScopeQuery> {
  return publicRpc("get_scryglass_tier_scope", {
    p_patch: input.patch.trim(),
    p_role: input.role?.trim() || null,
    p_region: input.region?.trim() || null,
    p_league: input.league?.trim() || null,
    p_tier: input.tier?.trim() || null,
    p_similarity_limit: Number.isInteger(input.similarityLimit)
      ? Math.min(100, Math.max(0, Number(input.similarityLimit)))
      : 100,
  }, { manifest, signal });
}

export function getChampionAggregates(
  manifest: PackManifest,
  input: {
    leagues?: string[];
    tiers?: string[];
    roles?: string[];
    active?: boolean;
    minGames?: number;
    order?: "games_desc" | "games_asc" | "win_rate_desc" | "win_rate_asc";
    limit?: number;
    offset?: number;
  } = {},
  signal?: AbortSignal,
): Promise<QueryApiEnvelope<ChampionAggregateQueryRow>> {
  return publicRpc("get_scryglass_champions", {
    p_leagues: input.leagues?.length ? input.leagues : null,
    p_tiers: input.tiers?.length ? input.tiers : null,
    p_roles: input.roles?.length ? input.roles : null,
    p_active: input.active ?? null,
    p_min_games: input.minGames ?? 100,
    p_order: input.order ?? "games_desc",
    p_limit: boundedRowLimit(input.limit, PUBLIC_ROW_LIMIT),
    p_offset: input.offset ?? 0,
  }, { manifest, signal });
}

export function matchSummary(row: MatchQueryRow): MatchSummary {
  return {
    game_id: row.game_id,
    date: row.played_at,
    league: row.league ?? "Unknown competition",
    competition_tier: row.tier,
    blue_team: row.blue_team,
    red_team: row.red_team,
    blue_win: row.blue_win,
    champions: row.champions,
    grades_available: row.payload.players?.filter(
      (player) => player.grade?.status === "available",
    ).length ?? 0,
  };
}
