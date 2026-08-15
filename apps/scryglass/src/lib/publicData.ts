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
const RPC_TIMEOUT_MS = 5_000;
const PUBLIC_ROW_LIMIT = 20;
const PUBLIC_RATINGS_ROW_LIMIT = 100;
const PLAYER_PROFILE_CACHE_MAX_ENTRIES = 256;
const playerProfileCache = new Map<string, Promise<PlayerProfileQuery>>();

export type QueryApiEnvelope<T> = {
  schema_version: typeof QUERY_API_SCHEMA;
  release_id: string;
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
  const cached = playerProfileCache.get(cacheKey);
  if (cached) {
    playerProfileCache.delete(cacheKey);
    playerProfileCache.set(cacheKey, cached);
    return cached;
  }
  const pending = publicRpc<PlayerProfileQuery>(
    "get_scryglass_player_profile",
    { p_name: name },
    { manifest },
  ).catch((error) => {
    if (playerProfileCache.get(cacheKey) === pending) playerProfileCache.delete(cacheKey);
    throw error;
  });
  playerProfileCache.set(cacheKey, pending);
  while (playerProfileCache.size > PLAYER_PROFILE_CACHE_MAX_ENTRIES) {
    const oldest = playerProfileCache.keys().next().value;
    if (oldest === undefined) break;
    playerProfileCache.delete(oldest);
  }
  return pending;
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
