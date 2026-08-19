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
import { teamQueryAliases } from "@/lib/pack";
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

function isNullableNumber(value: unknown): boolean {
  return value == null || Number.isFinite(value);
}

function isGame(value: unknown): value is MapStatsGame {
  if (!value || typeof value !== "object") return false;
  const game = value as Record<string, unknown>;
  return typeof game.game_id === "string"
    && game.game_id.length > 0
    && typeof game.date === "string"
    && game.date.length > 0
    && typeof game.win === "boolean"
    && (game.league == null || typeof game.league === "string")
    && (game.opponent == null || typeof game.opponent === "string")
    && isNullableNumber(game.cs_per_min)
    && isNullableNumber(game.gold_per_min)
    && isNullableNumber(game.gold_share_pct)
    && isNullableNumber(game.damage_per_min)
    && isNullableNumber(game.damage_share_pct)
    && isNullableNumber(game.kda)
    && isNullableNumber(game.game_length_min);
}

function isEntry(value: unknown): value is MapStatsEntry {
  if (!value || typeof value !== "object") return false;
  const record = value as Record<string, unknown>;
  if (typeof record.name !== "string" || record.name.length === 0) return false;
  if (!Number.isFinite(record.maps) || !Number.isFinite(record.wins) || !Number.isFinite(record.losses)) return false;
  if (!isNullableNumber(record.win_rate)) return false;
  if (!Array.isArray(record.games)) return false;
  // Fail closed on the WHOLE entry when any row is malformed. The aggregate
  // header is documented as the mean over exactly the rows beneath it, so
  // silently dropping a bad row would publish a header disagreeing with its
  // own table; and a null row previously crashed the profile at sort time.
  return record.games.every(isGame);
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
  kind: MapStatsKind = "players",
): MapStatsEntry | null {
  const wanted = name.trim();
  if (!wanted) return null;
  // Teams are commonly queried by alias (DK, KC, MKOI); resolve through the
  // existing alias contract so those do not return a false not-found.
  const candidates = kind === "teams" ? teamQueryAliases(wanted) : [wanted];
  for (const candidate of candidates) {
    const exact = entries.find((entry) => entry.name === candidate);
    if (exact) return exact;
  }
  for (const candidate of candidates) {
    const lower = candidate.toLowerCase();
    const match = entries.find((entry) => entry.name.toLowerCase() === lower);
    if (match) return match;
  }
  return null;
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
  const entry = findMapStatsEntry(payload[kind], name, kind);
  if (!entry) return null;
  return { entry, window_days: payload.window_days, map_limit: payload.map_limit };
}

/** Read the optional asset for an /api/chat route. Throws only on transport. */
export async function lookupMapStats(
  kind: MapStatsKind,
  name: string,
): Promise<MapStatsLookup | null> {
  // Deliberately does NOT forward the route's deadline signal: the
  // signal-bearing branch of readPackJson bypasses the per-release cache, so
  // every request would download, hash and parse the whole multi-megabyte
  // artifact to answer for one subject. The cached branch serves repeats from
  // memory keyed by release; the route's overall deadline still bounds the
  // response through secureChatRoute's race.
  const payload = parsePlayerMapStats(
    await readChatJson<unknown>(PLAYER_MAP_STATS_PATH),
  );
  if (!payload) return null;
  const entry = findMapStatsEntry(payload[kind], name, kind);
  if (!entry) return null;
  return { entry, window_days: payload.window_days, map_limit: payload.map_limit };
}
