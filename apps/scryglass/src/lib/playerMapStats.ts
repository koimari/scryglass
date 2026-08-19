/** Reader for features/player_map_stats.json, the profile Stats section source.
 *
 * The artifact is display-only and optional, so every reader here fails closed
 * to `null` rather than throwing: a release published without the asset must
 * still render both profiles, minus the Stats section.
 *
 * `players` and `teams` are arrays carrying a `name` field rather than
 * name-keyed objects. The publication guard
 * `public.scryglass_json_has_draft_fields` inspects JSON object *keys*, so a
 * name-keyed map would submit player and team names to the banned-key list.
 * Keep the array shape.
 */

import { readChatJson } from "@/lib/chatApi";
import type { PackManifest } from "@/lib/pack";
import { readPackJson } from "@/lib/serverPack";

export const PLAYER_MAP_STATS_PATH = "features/player_map_stats.json";
export const PLAYER_MAP_STATS_SCHEMA_VERSION = "scryglass:player-map-stats:v1";

/** One published map. Every metric is nullable: absent is never zero. */
export type MapStatsGame = {
  game_id: string;
  date: string;
  league: string | null;
  opponent: string | null;
  champion?: string | null;
  position?: string | null;
  win: boolean;
  cs_per_min?: number | null;
  gold_per_min: number | null;
  gold_share_pct?: number | null;
  damage_per_min: number | null;
  damage_share_pct?: number | null;
  kda?: number | null;
  kills?: number | null;
  deaths?: number | null;
  game_length_min: number | null;
};

/** Aggregate header over exactly the `games` rows beneath it, plus those rows. */
export type MapStatsEntry = {
  name: string;
  maps: number;
  wins: number;
  losses: number;
  win_rate: number | null;
  cs_per_min?: number | null;
  gold_per_min: number | null;
  gold_share_pct?: number | null;
  damage_per_min: number | null;
  damage_share_pct?: number | null;
  kda?: number | null;
  kills?: number | null;
  deaths?: number | null;
  game_length_min?: number | null;
  games: MapStatsGame[];
};

export type PlayerMapStats = {
  schema_version: string;
  window_days: number;
  map_limit: number;
  players: MapStatsEntry[];
  teams: MapStatsEntry[];
};

export type MapStatsKind = "players" | "teams";

export type MapStatsLookup = {
  entry: MapStatsEntry;
  window_days: number;
  map_limit: number;
};

function isEntry(value: unknown): value is MapStatsEntry {
  if (!value || typeof value !== "object") return false;
  const record = value as Record<string, unknown>;
  return typeof record.name === "string"
    && record.name.length > 0
    && Number.isFinite(record.maps)
    && Array.isArray(record.games);
}

/** Accept only a payload that declares the exact schema this reader knows. */
export function parsePlayerMapStats(value: unknown): PlayerMapStats | null {
  if (!value || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  if (record.schema_version !== PLAYER_MAP_STATS_SCHEMA_VERSION) return null;
  if (!Array.isArray(record.players) || !Array.isArray(record.teams)) return null;
  const windowDays = Number(record.window_days);
  const mapLimit = Number(record.map_limit);
  if (!Number.isFinite(windowDays) || !Number.isFinite(mapLimit)) return null;
  return {
    schema_version: PLAYER_MAP_STATS_SCHEMA_VERSION,
    window_days: windowDays,
    map_limit: mapLimit,
    players: record.players.filter(isEntry),
    teams: record.teams.filter(isEntry),
  };
}

/** Exact name first, so source casing never decides a collision. */
export function findMapStatsEntry(
  entries: readonly MapStatsEntry[],
  name: string,
): MapStatsEntry | null {
  const wanted = name.trim();
  if (!wanted) return null;
  const exact = entries.find((entry) => entry.name === wanted);
  if (exact) return exact;
  const lower = wanted.toLowerCase();
  return entries.find((entry) => entry.name.toLowerCase() === lower) ?? null;
}

/** Read the optional asset for a server-rendered profile, never throwing. */
export async function readMapStatsEntry(
  manifest: PackManifest,
  kind: MapStatsKind,
  name: string,
): Promise<MapStatsLookup | null> {
  const payload = await readPackJson<unknown>(manifest, PLAYER_MAP_STATS_PATH)
    .then(parsePlayerMapStats)
    .catch(() => null);
  if (!payload) return null;
  const entry = findMapStatsEntry(payload[kind], name);
  if (!entry) return null;
  return { entry, window_days: payload.window_days, map_limit: payload.map_limit };
}

/** Read the optional asset for an /api/chat route. Throws only on transport. */
export async function lookupMapStats(
  kind: MapStatsKind,
  name: string,
  signal?: AbortSignal,
): Promise<MapStatsLookup | null> {
  const payload = parsePlayerMapStats(
    await readChatJson<unknown>(PLAYER_MAP_STATS_PATH, signal),
  );
  if (!payload) return null;
  const entry = findMapStatsEntry(payload[kind], name);
  if (!entry) return null;
  return { entry, window_days: payload.window_days, map_limit: payload.map_limit };
}
