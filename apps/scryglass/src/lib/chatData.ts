/** Support-chat data layer. Active query-API releases use bounded RPCs. Older
 * rollout releases keep the immutable asset reader as a temporary fallback. */

import { readChatJson } from "@/lib/chatApi";
import {
  getMatches,
  getRatings,
  getTeamProfile,
  queryApiAvailable,
  type MatchQueryRow,
  type RatingQueryRow,
} from "@/lib/publicData";
import { readPackJson, readPackManifest } from "@/lib/serverPack";

export type LeaderboardRow = {
  name: string;
  rating: number | null;
  role: string | null;
  team: string | null;
  league: string | null;
  tier: string | null;
  games: number;
  wins: number | null;
  win_rate: number | null;
  grade_a_games: number;
  grade_games: number;
  recent_form: number | null;
};

type ProfileStats = {
  roleByPlayer: Map<string, string>;
  gradeAByPlayer: Map<string, number>;
  gradeGamesByPlayer: Map<string, number>;
  teamByPlayer: Map<string, string>;
  leagueByPlayer: Map<string, string>;
};

type ProfileStatsGame = {
  players: Array<{
    player?: string;
    role?: string;
    grade?: { status?: string; grade?: string };
  }>;
};

let profileStatsCache: { releaseId: string; promise: Promise<ProfileStats> } | null = null;

function roleKey(role: string | null | undefined): string | null {
  const value = String(role ?? "").trim().toLowerCase();
  if (value === "jungle") return "jng";
  if (value === "support") return "sup";
  if (value === "mid" || value === "middle") return "mid";
  if (value === "bot" || value === "adc" || value === "marksman") return "bot";
  if (value === "top") return "top";
  return value || null;
}

function boundedLimit(limit: number): number {
  return Math.min(Math.max(Number.isInteger(limit) ? limit : 10, 1), 20);
}

function rowFromQueryApi(row: RatingQueryRow): LeaderboardRow {
  return {
    name: row.name,
    rating: typeof row.rating === "number" ? row.rating : null,
    role: roleKey(row.role),
    team: row.team,
    league: row.league,
    tier: row.tier,
    games: Number(row.games) || 0,
    wins: typeof row.wins === "number" ? row.wins : null,
    win_rate: typeof row.win_rate === "number" ? row.win_rate : null,
    grade_a_games: Number(row.grade_a_games ?? row.payload?.grade_a_games ?? 0) || 0,
    grade_games: Number(row.grade_games ?? row.payload?.grade_games ?? 0) || 0,
    recent_form: typeof row.payload?.recent_form === "number" ? row.payload.recent_form : null,
  };
}

function matchFromQueryApi(row: MatchQueryRow): ChatMatch {
  return {
    game_id: row.game_id,
    date: row.played_at,
    league: row.league ?? "",
    blue_team: row.blue_team,
    red_team: row.red_team,
    blue_win: row.blue_win,
    champions: row.champions,
  };
}

function profileStatsFromPayload(payload: { games: Record<string, ProfileStatsGame> }): ProfileStats {
  const roleByPlayer = new Map<string, string>();
  const gradeAByPlayer = new Map<string, number>();
  const gradeGamesByPlayer = new Map<string, number>();
  const teamByPlayer = new Map<string, string>();
  const leagueByPlayer = new Map<string, string>();
  const roleCounts = new Map<string, Map<string, number>>();
  for (const game of Object.values(payload.games ?? {})) {
    for (const player of game.players ?? []) {
      const name = String(player.player ?? "").trim();
      if (!name) continue;
      const role = roleKey(player.role);
      if (role) {
        const counts = roleCounts.get(name) ?? new Map<string, number>();
        counts.set(role, (counts.get(role) ?? 0) + 1);
        roleCounts.set(name, counts);
      }
      const grade = player.grade;
      if (grade?.status === "available" && typeof grade.grade === "string") {
        gradeGamesByPlayer.set(name, (gradeGamesByPlayer.get(name) ?? 0) + 1);
        if (grade.grade.toUpperCase() === "A") {
          gradeAByPlayer.set(name, (gradeAByPlayer.get(name) ?? 0) + 1);
        }
      }
    }
  }
  for (const [name, counts] of roleCounts) {
    let best = "";
    let bestCount = 0;
    for (const [role, count] of counts) {
      if (count > bestCount) {
        best = role;
        bestCount = count;
      }
    }
    if (best) roleByPlayer.set(name, best);
  }
  return { roleByPlayer, gradeAByPlayer, gradeGamesByPlayer, teamByPlayer, leagueByPlayer };
}

async function scanProfileStats(signal?: AbortSignal): Promise<ProfileStats> {
  const manifest = await readPackManifest(signal);
  const load = () => readPackJson<{ games: Record<string, ProfileStatsGame> }>(
    manifest,
    "features/profile_records.json",
    signal,
  ).then(profileStatsFromPayload);
  if (signal) return load();
  if (profileStatsCache?.releaseId !== manifest.pack_id) {
    const releaseId = manifest.pack_id;
    const promise = load().catch((error) => {
      if (profileStatsCache?.releaseId === releaseId) profileStatsCache = null;
      throw error;
    });
    profileStatsCache = { releaseId, promise };
  }
  return profileStatsCache.promise;
}

type RatingsSnapshot = Array<{
  player?: string;
  mu_total?: number;
  n_maps?: number;
  last_team?: string;
  home_league?: string;
}>;

type TeamRatingsSnapshot = Array<{
  team?: string;
  mu_total?: number;
  home_league?: string;
}>;

type PlayerRecords = Record<string, { wins?: number; games?: number; wr?: number; current_team?: string; current_league?: string; current_tier?: string }>;
type TeamRecords = Record<string, { wins?: number; games?: number; wr?: number; leagues?: string[]; current_tier?: string }>;

export function findPlayerRecord(records: PlayerRecords, name: string): PlayerRecords[string] {
  const exact = records[name];
  if (exact) return exact;
  const lower = name.toLowerCase();
  const directLower = records[lower];
  if (directLower) return directLower;
  const matchingKey = Object.keys(records).find((key) => key.toLowerCase() === lower);
  return matchingKey ? records[matchingKey] : {};
}

async function loadPlayerRatings(signal?: AbortSignal): Promise<RatingsSnapshot> {
  return readChatJson<RatingsSnapshot>("features/player_ratings_snapshot.json", signal);
}

async function loadPlayerRecords(signal?: AbortSignal): Promise<PlayerRecords> {
  return readChatJson<PlayerRecords>("features/player_records.json", signal);
}

export async function leaderboardRows(
  category: string,
  role: string | null,
  limit: number,
  tier: string | null = null,
  signal?: AbortSignal,
): Promise<LeaderboardRow[]> {
  const manifest = await readPackManifest(signal);
  if (queryApiAvailable(manifest)) {
    const safeLimit = boundedLimit(limit);
    const response = await getRatings(manifest, {
      kind: category === "teams" ? "teams" : "players",
      tiers: tier ? [tier.toLowerCase()] : [],
      roles: role && category !== "teams" ? [roleKey(role) ?? role] : [],
      active: true,
      order: category === "a_grades"
        ? "grade_a_desc"
        : category === "win_rate"
          ? "win_rate_desc"
          : "rating_desc",
      minGames: category === "win_rate" ? 10 : 0,
      limit: safeLimit,
      offset: 0,
    }, signal);
    return response.rows.map(rowFromQueryApi).slice(0, safeLimit);
  }
  // Preferred: the precomputed artifact.
  try {
    const payload = await readChatJson<{
      top?: Record<string, unknown[]>;
      teams?: Array<Record<string, unknown>>;
      indexes?: { players?: Record<string, unknown>; teams?: Record<string, unknown> };
    }>("features/leaderboards.json", signal);
    const rows = category === "teams"
      ? (payload.teams ?? [])
      : category === "a_grades"
        ? (payload.top?.a_grades ?? [])
        : category === "win_rate"
          ? (payload.top?.win_rate ?? [])
          : role
            ? ((payload.top?.rating_by_role as Record<string, unknown[]> | undefined)?.[role] ?? [])
            : (payload.top?.rating ?? []);
    const mapped = (rows as Array<Record<string, unknown>>).map((row) => ({
      name: String(row.name ?? row.player ?? row.team ?? ""),
      rating: typeof row.rating === "number" ? row.rating : null,
      role: typeof row.role === "string" ? row.role : null,
      team: typeof row.team === "string" ? row.team : null,
      league: typeof row.league === "string" ? row.league : null,
      tier: typeof (row.tier ?? row.current_tier) === "string" ? String(row.tier ?? row.current_tier) : null,
      games: Number(row.games ?? row.n_maps ?? 0) || 0,
      wins: typeof row.wins === "number" ? row.wins : null,
      win_rate: typeof row.win_rate === "number" ? row.win_rate : null,
      grade_a_games: Number(row.grade_a_games ?? 0) || 0,
      grade_games: Number(row.grade_games ?? 0) || 0,
      recent_form: typeof row.recent_form === "number" ? row.recent_form : null,
    }));
    return (tier ? mapped.filter((row) => row.tier === tier) : mapped).slice(0, limit);
  } catch {
    // Fallback: compute from the always-present assets.
    if (category === "teams") {
      const [ratings, records] = await Promise.all([
        readChatJson<TeamRatingsSnapshot>("features/ratings_snapshot.json", signal),
        readChatJson<TeamRecords>("features/team_records.json", signal),
      ]);
      return ratings
        .map((entry) => {
          const name = String(entry.team ?? "");
          const record = records[name] ?? {};
          return {
            name,
            rating: typeof entry.mu_total === "number" ? entry.mu_total : null,
            role: null,
            team: name,
            league: String(entry.home_league ?? "") || null,
            tier: String(record.current_tier ?? "") || null,
            games: Number(record.games ?? 0) || 0,
            wins: Number(record.wins ?? null) || null,
            win_rate: typeof record.wr === "number" ? record.wr : null,
            grade_a_games: 0,
            grade_games: 0,
            recent_form: null,
          };
        })
        .filter((row) => !tier || row.tier === tier)
        .sort((a, b) => (b.rating ?? 0) - (a.rating ?? 0))
        .slice(0, limit);
    }
    const [ratings, records, stats] = await Promise.all([
      loadPlayerRatings(signal),
      loadPlayerRecords(signal),
      scanProfileStats(signal),
    ]);
    const rows: LeaderboardRow[] = ratings.map((entry) => {
      const name = String(entry.player ?? "");
      const record = findPlayerRecord(records, name);
      const role = stats.roleByPlayer.get(name) ?? null;
      const wins = typeof record.wins === "number" ? record.wins : null;
      const games = Number(record.games ?? entry.n_maps ?? 0) || 0;
      const winRate = typeof record.wr === "number" ? record.wr : (games > 0 && wins != null ? wins / games : null);
      return {
        name,
        rating: typeof entry.mu_total === "number" ? entry.mu_total : null,
        role,
        team: String(entry.last_team ?? "") || null,
        league: String(entry.home_league ?? "") || null,
        tier: String(record.current_tier ?? "") || null,
        games,
        wins,
        win_rate: winRate,
        grade_a_games: stats.gradeAByPlayer.get(name) ?? 0,
        grade_games: stats.gradeGamesByPlayer.get(name) ?? 0,
        recent_form: null,
      };
    });
    const sorted = category === "a_grades"
      ? rows.sort((a, b) => b.grade_a_games - a.grade_a_games)
      : category === "win_rate"
        ? rows.filter((row) => row.games >= 10).sort((a, b) => (b.win_rate ?? 0) - (a.win_rate ?? 0))
        : rows.sort((a, b) => (b.rating ?? 0) - (a.rating ?? 0));
    const filtered = sorted.filter((row) => (!role || row.role === role) && (!tier || row.tier === tier));
    return filtered.slice(0, limit);
  }
}

export async function lookupPlayer(name: string, signal?: AbortSignal): Promise<LeaderboardRow | null> {
  const manifest = await readPackManifest(signal);
  if (queryApiAvailable(manifest)) {
    const result = await getRatings(manifest, {
      kind: "players",
      names: [name],
      limit: 1,
      offset: 0,
    }, signal);
    return result.rows[0] ? rowFromQueryApi(result.rows[0]) : null;
  }
  const lower = name.trim().toLowerCase();
  try {
    const payload = await readChatJson<{ players?: Record<string, LeaderboardRow & { player?: string }> }>("features/leaderboards.json", signal);
    const players = payload.players ?? {};
    const key = Object.keys(players).find((candidate) => candidate.toLowerCase() === lower)
      ?? Object.keys(players).find((candidate) => candidate.toLowerCase().includes(lower));
    if (key) {
      const row = players[key];
      return { name: key, rating: row.rating ?? null, role: row.role ?? null, team: row.team ?? null, league: row.league ?? null, tier: row.tier ?? null, games: row.games ?? 0, wins: row.wins ?? null, win_rate: row.win_rate ?? null, grade_a_games: row.grade_a_games ?? 0, grade_games: row.grade_games ?? 0, recent_form: row.recent_form ?? null };
    }
    return null;
  } catch {
    const [ratings, records, stats] = await Promise.all([
      loadPlayerRatings(signal),
      loadPlayerRecords(signal),
      scanProfileStats(signal),
    ]);
    const entry = ratings.find((item) => String(item.player ?? "").toLowerCase() === lower)
      ?? ratings.find((item) => String(item.player ?? "").toLowerCase().includes(lower));
    if (!entry) return null;
    const name = String(entry.player ?? "");
    const record = findPlayerRecord(records, name);
    const games = Number(record.games ?? entry.n_maps ?? 0) || 0;
    const wins = typeof record.wins === "number" ? record.wins : null;
    return {
      name,
      rating: typeof entry.mu_total === "number" ? entry.mu_total : null,
      role: stats.roleByPlayer.get(name) ?? null,
      team: String(entry.last_team ?? "") || null,
      league: String(entry.home_league ?? "") || null,
      tier: String(record.current_tier ?? "") || null,
      games,
      wins,
      win_rate: typeof record.wr === "number" ? record.wr : games > 0 && wins != null ? wins / games : null,
      grade_a_games: stats.gradeAByPlayer.get(name) ?? 0,
      grade_games: stats.gradeGamesByPlayer.get(name) ?? 0,
      recent_form: null,
    };
  }
}

export type TeamLookup = {
  team: string;
  rating: number | null;
  league: string | null;
  games: number;
  wins: number | null;
  win_rate: number | null;
  recent: Array<{ date: string; opponent: string; side: string; won: boolean; game_id: string }>;
};

export type ChatMatch = {
  game_id: string;
  date: string;
  league: string;
  blue_team: string;
  red_team: string;
  blue_win: number;
  champions: string[];
};

export async function loadChatMatches(input: {
  team?: string;
  league?: string;
  champion?: string;
  limit?: number;
} = {}, signal?: AbortSignal): Promise<ChatMatch[]> {
  const manifest = await readPackManifest(signal);
  if (queryApiAvailable(manifest)) {
    const response = await getMatches(manifest, {
      team: input.team,
      champion: input.champion,
      leagues: input.league ? [input.league] : [],
      limit: boundedLimit(input.limit ?? 20),
      offset: 0,
    }, signal);
    return response.rows.map(matchFromQueryApi);
  }
  try {
    const index = await readChatJson<{ games: ChatMatch[] }>("features/match_index.json", signal);
    return index.games ?? [];
  } catch {
    type ProfileGame = Omit<ChatMatch, "champions"> & { players?: Array<{ champion?: string }> };
    const profiles = await readChatJson<{ games: Record<string, ProfileGame> }>("features/profile_records.json", signal);
    return Object.values(profiles.games ?? {}).map((game) => ({
      game_id: String(game.game_id ?? ""),
      date: String(game.date ?? ""),
      league: String(game.league ?? ""),
      blue_team: String(game.blue_team ?? ""),
      red_team: String(game.red_team ?? ""),
      blue_win: Number(game.blue_win ?? 0),
      champions: (game.players ?? []).map((player) => String(player.champion ?? "")).filter(Boolean),
    }));
  }
}

export function filterChatMatchesByTeam(games: ChatMatch[], query: string): ChatMatch[] {
  const lower = query.trim().toLowerCase();
  const exact = games.filter(
    (game) => game.blue_team.toLowerCase() === lower || game.red_team.toLowerCase() === lower,
  );
  if (exact.length) return exact;
  return games.filter(
    (game) => game.blue_team.toLowerCase().includes(lower) || game.red_team.toLowerCase().includes(lower),
  );
}

export async function lookupTeam(name: string, signal?: AbortSignal): Promise<TeamLookup | null> {
  const manifest = await readPackManifest(signal);
  if (queryApiAvailable(manifest)) {
    const profile = await getTeamProfile(manifest, name, signal);
    if (!profile.row) return null;
    const row = profile.row;
    const team = row.name;
    const recent = profile.recent_games.slice(0, 5).map((game) => {
      const isBlue = game.blue_team.toLowerCase() === team.toLowerCase();
      return {
        date: game.played_at,
        opponent: isBlue ? game.red_team : game.blue_team,
        side: isBlue ? "Blue" : "Red",
        won: isBlue ? game.blue_win === 1 : game.blue_win === 0,
        game_id: game.game_id,
      };
    });
    return {
      team,
      rating: typeof row.rating === "number" ? row.rating : null,
      league: row.league,
      games: Number(row.games) || 0,
      wins: typeof row.wins === "number" ? row.wins : null,
      win_rate: typeof row.win_rate === "number" ? row.win_rate : null,
      recent,
    };
  }
  const lower = name.trim().toLowerCase();
  try {
    const payload = await readChatJson<{ teams?: Array<Record<string, unknown>> }>("features/leaderboards.json", signal);
    const team = (payload.teams ?? []).find((entry) => String(entry.team ?? "").toLowerCase() === lower)
      ?? (payload.teams ?? []).find((entry) => String(entry.team ?? "").toLowerCase().includes(lower));
    if (team) {
      return {
        team: String(team.team ?? ""),
        rating: typeof team.rating === "number" ? team.rating : null,
        league: typeof team.league === "string" ? team.league : null,
        games: Number(team.games ?? 0) || 0,
        wins: typeof team.wins === "number" ? team.wins : null,
        win_rate: typeof team.win_rate === "number" ? team.win_rate : null,
        recent: Array.isArray(team.recent) ? (team.recent as Array<{ date: string; opponent: string; side: string; won: boolean; game_id: string }>) : [],
      };
    }
    return null;
  } catch {
    const [ratings, records, matchGames] = await Promise.all([
      readChatJson<TeamRatingsSnapshot>("features/ratings_snapshot.json", signal),
      readChatJson<TeamRecords>("features/team_records.json", signal),
      loadChatMatches({}, signal),
    ]);
    const ratingEntry = ratings.find((entry) => String(entry.team ?? "").toLowerCase() === lower)
      ?? ratings.find((entry) => String(entry.team ?? "").toLowerCase().includes(lower));
    const recordKey = Object.keys(records).find((key) => key.toLowerCase() === lower)
      ?? Object.keys(records).find((key) => key.toLowerCase().includes(lower));
    const record = recordKey ? records[recordKey] : undefined;
    if (!ratingEntry && !record) return null;
    const team = String(ratingEntry?.team ?? recordKey ?? "");
    const games = Number(record?.games ?? 0) || 0;
    const wins = typeof record?.wins === "number" ? record.wins : null;
    const recent = matchGames
      .filter((game) => String(game.blue_team ?? "").toLowerCase() === lower || String(game.red_team ?? "").toLowerCase() === lower)
      .sort((a, b) => String(b.date ?? "").localeCompare(String(a.date ?? "")))
      .slice(0, 5)
      .map((game) => {
        const isBlue = String(game.blue_team ?? "").toLowerCase() === lower;
        const won = isBlue ? game.blue_win === 1 : game.blue_win === 0;
        return {
          date: String(game.date ?? ""),
          opponent: isBlue ? String(game.red_team ?? "") : String(game.blue_team ?? ""),
          side: isBlue ? "Blue" : "Red",
          won,
          game_id: String(game.game_id ?? ""),
        };
      });
    return {
      team,
      rating: typeof ratingEntry?.mu_total === "number" ? ratingEntry.mu_total : null,
      league: String(ratingEntry?.home_league ?? "") || null,
      games,
      wins,
      win_rate: typeof record?.wr === "number" ? record.wr : games > 0 && wins != null ? wins / games : null,
      recent,
    };
  }
}
