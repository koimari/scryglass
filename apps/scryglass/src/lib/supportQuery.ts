import { readChatJson } from "@/lib/chatApi";

export type QueryDataset = "players" | "player_champions";
export type QueryOperation = "rank" | "compare";
export type QueryDirection = "asc" | "desc";
export type QueryField =
  | "name"
  | "role"
  | "team"
  | "league"
  | "tier"
  | "champion"
  | "active"
  | "rating"
  | "games"
  | "wins"
  | "win_rate"
  | "grade_a_games"
  | "champion_games"
  | "champion_wins"
  | "champion_win_rate"
  | "champion_score"
  | "champion_median_distance"
  | "champion_mean_distance";
export type QueryOperator = "eq" | "in" | "gte" | "lte";

export type QueryFilter = {
  field: QueryField;
  op: QueryOperator;
  value: string | number | string[];
};

export type QueryPlan = {
  version: 1;
  dataset: QueryDataset;
  operation: QueryOperation;
  filters: QueryFilter[];
  orderBy: Array<{ field: QueryField; direction: QueryDirection }>;
  limit: number;
};

export type QueryPlayerRow = {
  name: string;
  role: string | null;
  team: string | null;
  league: string | null;
  tier: string | null;
  active: number | null;
  rating: number | null;
  games: number;
  wins: number | null;
  win_rate: number | null;
  grade_a_games: number;
  champion: string | null;
  champion_games: number | null;
  champion_wins: number | null;
  champion_win_rate: number | null;
  champion_score: number | null;
  champion_median_distance: number | null;
  champion_mean_distance: number | null;
};

export type SupportQueryIndex = {
  players: QueryPlayerRow[];
  playerChampions: QueryPlayerRow[];
  entities: {
    players: string[];
    champions: string[];
    teams: string[];
    leagues: string[];
  };
};

export type QueryAnswer = {
  headline: string;
  basis: string;
  caveat: string | null;
};

export type QueryExecutionResult = {
  kind: "player_query";
  plan: QueryPlan;
  answer: QueryAnswer;
  rows: QueryPlayerRow[];
  proof: {
    sources: string[];
    resultCount: number;
  };
};

type RatingAsset = Array<{
  player?: string;
  mu_total?: number;
  n_maps?: number;
  last_team?: string;
  home_league?: string;
  evidence_active?: number | null;
}>;

type PlayerRecord = {
  wins?: number;
  games?: number;
  wr?: number;
  current_team?: string;
  current_league?: string;
  current_tier?: string;
  primary_role?: string;
  roles?: string[];
};

type PlayerRecordsAsset = Record<string, PlayerRecord>;

type ChampionRecord = {
  champion?: string;
  games?: number;
  wins?: number;
  wr?: number;
};

type PlayerChampionAsset = Record<string, ChampionRecord[]>;

type ProfileAsset = {
  games?: Record<string, {
    players?: Array<{
      player?: string;
      grade?: { status?: string; grade?: string };
    }>;
  }>;
};

const DATASET_FIELDS: Record<QueryDataset, Set<QueryField>> = {
  players: new Set([
    "name", "role", "team", "league", "tier", "active", "rating", "games", "wins",
    "win_rate", "grade_a_games",
  ]),
  player_champions: new Set([
    "name", "role", "team", "league", "tier", "champion", "active", "rating", "games",
    "wins", "win_rate", "grade_a_games", "champion_games", "champion_wins",
    "champion_win_rate", "champion_score", "champion_median_distance", "champion_mean_distance",
  ]),
};

const STRING_FIELDS = new Set<QueryField>(["name", "role", "team", "league", "tier", "champion"]);
const NUMBER_FIELDS = new Set<QueryField>([
  "active", "rating", "games", "wins", "win_rate", "grade_a_games", "champion_games",
  "champion_wins", "champion_win_rate", "champion_score",
  "champion_median_distance", "champion_mean_distance",
]);
const PLAN_KEYS = new Set(["version", "dataset", "operation", "filters", "orderBy", "limit"]);
const FILTER_KEYS = new Set(["field", "op", "value"]);
const ORDER_KEYS = new Set(["field", "direction"]);
const QUERY_SOURCES = [
  "features/player_ratings_snapshot.json",
  "features/player_records.json",
  "features/player_champion_records.json",
  "features/profile_records.json",
];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasOnlyKeys(value: Record<string, unknown>, allowed: Set<string>): boolean {
  return Object.keys(value).every((key) => allowed.has(key));
}

function finiteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function normalizeEntity(value: string): string {
  return value
    .normalize("NFKD")
    .toLowerCase()
    .replace(/[’']/g, "")
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .replace(/\s+/g, " ");
}

function roleKey(value: string | null | undefined): string | null {
  const role = normalizeEntity(value ?? "");
  if (role === "jungle" || role === "jungler" || role === "jng") return "jng";
  if (role === "support" || role === "sup") return "sup";
  if (role === "mid" || role === "middle" || role === "mid laner") return "mid";
  if (role === "bot" || role === "adc" || role === "marksman" || role === "bot laner") return "bot";
  if (role === "top" || role === "top laner") return "top";
  return role || null;
}

function numberOrNull(value: unknown): number | null {
  return finiteNumber(value) ? value : null;
}

function findByNormalizedKey<T>(records: Record<string, T>, name: string): T | undefined {
  const direct = records[name];
  if (direct) return direct;
  const target = normalizeEntity(name);
  const keys = Object.keys(records).filter((candidate) => normalizeEntity(candidate) === target);
  return keys.length === 1 ? records[keys[0]] : undefined;
}

export function wilsonLowerBound(wins: number, games: number, z = 1.96): number | null {
  if (!Number.isInteger(wins) || !Number.isInteger(games) || games <= 0 || wins < 0 || wins > games) return null;
  const proportion = wins / games;
  const zSquared = z * z;
  const denominator = 1 + zSquared / games;
  const centre = proportion + zSquared / (2 * games);
  const margin = z * Math.sqrt((proportion * (1 - proportion) + zSquared / (4 * games)) / games);
  return Math.max(0, (centre - margin) / denominator);
}

function gradeACounts(profile: ProfileAsset): Map<string, number> {
  const counts = new Map<string, number>();
  for (const game of Object.values(profile.games ?? {})) {
    for (const player of game.players ?? []) {
      if (player.grade?.status !== "available" || player.grade.grade?.toUpperCase() !== "A") continue;
      const name = String(player.player ?? "").trim();
      if (!name) continue;
      const key = normalizeEntity(name);
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
  }
  return counts;
}

export function buildSupportQueryIndex(assets: {
  ratings: RatingAsset;
  records: PlayerRecordsAsset;
  champions: PlayerChampionAsset;
  profiles?: ProfileAsset;
}): SupportQueryIndex {
  const gradeAByName = gradeACounts(assets.profiles ?? {});
  const playerSources: Array<{ name: string; rating: RatingAsset[number] | undefined }> = [];
  const ratingNames = new Set<string>();
  for (const rating of assets.ratings) {
    const name = String(rating.player ?? "").trim();
    if (!name) continue;
    playerSources.push({ name, rating });
    ratingNames.add(name);
  }
  for (const name of Object.keys(assets.records)) {
    const canonicalName = name.trim();
    if (canonicalName && !ratingNames.has(canonicalName)) {
      playerSources.push({ name: canonicalName, rating: undefined });
    }
  }

  const players: QueryPlayerRow[] = [];
  for (const source of playerSources) {
    const canonicalName = source.name;
    const key = normalizeEntity(canonicalName);
    const record = findByNormalizedKey(assets.records, canonicalName) ?? {};
    const rating = source.rating;
    const games = Number(record.games ?? rating?.n_maps ?? 0) || 0;
    const wins = finiteNumber(record.wins) ? record.wins : null;
    const winRate = finiteNumber(record.wr)
      ? record.wr
      : games > 0 && wins != null
        ? wins / games
        : null;
    players.push({
      name: canonicalName,
      role: roleKey(record.primary_role ?? record.roles?.[0]),
      team: String(record.current_team ?? rating?.last_team ?? "").trim() || null,
      league: String(record.current_league ?? rating?.home_league ?? "").trim() || null,
      tier: String(record.current_tier ?? "").trim().toLowerCase() || null,
      active: finiteNumber(rating?.evidence_active) ? rating.evidence_active : null,
      rating: numberOrNull(rating?.mu_total),
      games,
      wins,
      win_rate: winRate,
      grade_a_games: gradeAByName.get(key) ?? 0,
      champion: null,
      champion_games: null,
      champion_wins: null,
      champion_win_rate: null,
      champion_score: null,
      champion_median_distance: null,
      champion_mean_distance: null,
    });
  }

  const playerByExactName = new Map(players.map((player) => [player.name, player]));
  const playersByNormalizedName = new Map<string, QueryPlayerRow[]>();
  for (const player of players) {
    const key = normalizeEntity(player.name);
    playersByNormalizedName.set(key, [...(playersByNormalizedName.get(key) ?? []), player]);
  }
  const playerChampions: QueryPlayerRow[] = [];
  for (const [playerName, records] of Object.entries(assets.champions)) {
    const normalizedPlayers = playersByNormalizedName.get(normalizeEntity(playerName)) ?? [];
    const player = playerByExactName.get(playerName)
      ?? (normalizedPlayers.length === 1 ? normalizedPlayers[0] : undefined);
    if (!player || !Array.isArray(records)) continue;
    for (const record of records) {
      const champion = String(record.champion ?? "").trim();
      const championGames = Number(record.games ?? 0) || 0;
      const championWins = Number(record.wins ?? 0) || 0;
      if (!champion || championGames <= 0) continue;
      const championWinRate = finiteNumber(record.wr) ? record.wr : championWins / championGames;
      playerChampions.push({
        ...player,
        champion,
        champion_games: championGames,
        champion_wins: championWins,
        champion_win_rate: championWinRate,
        champion_score: wilsonLowerBound(championWins, championGames),
        champion_median_distance: null,
        champion_mean_distance: null,
      });
    }
  }

  const uniqueSorted = (values: Array<string | null>): string[] =>
    [...new Set(values.filter((value): value is string => Boolean(value)))]
      .sort((left, right) => left.localeCompare(right));
  const uniqueNormalizedSorted = (values: Array<string | null>): string[] => {
    const byKey = new Map<string, string>();
    for (const value of values) {
      if (!value) continue;
      const key = normalizeEntity(value);
      if (!byKey.has(key)) byKey.set(key, value);
    }
    return [...byKey.values()].sort((left, right) => left.localeCompare(right));
  };

  return {
    players,
    playerChampions,
    entities: {
      players: uniqueNormalizedSorted(players.map((player) => player.name)),
      champions: uniqueSorted(playerChampions.map((row) => row.champion)),
      teams: uniqueSorted(players.map((player) => player.team)),
      leagues: uniqueSorted(players.map((player) => player.league)),
    },
  };
}

let queryIndexPromise: Promise<SupportQueryIndex> | null = null;

export function loadSupportQueryIndex(): Promise<SupportQueryIndex> {
  if (!queryIndexPromise) {
    queryIndexPromise = Promise.all([
      readChatJson<RatingAsset>(QUERY_SOURCES[0]),
      readChatJson<PlayerRecordsAsset>(QUERY_SOURCES[1]),
      readChatJson<PlayerChampionAsset>(QUERY_SOURCES[2]),
      readChatJson<ProfileAsset>(QUERY_SOURCES[3]),
    ])
      .then(([ratings, records, champions, profiles]) =>
        buildSupportQueryIndex({ ratings, records, champions, profiles }))
      .catch((error) => {
        queryIndexPromise = null;
        throw error;
      });
  }
  return queryIndexPromise;
}

export function parseQueryPlan(value: unknown): { ok: true; plan: QueryPlan } | { ok: false; reason: string } {
  if (!isRecord(value) || !hasOnlyKeys(value, PLAN_KEYS)) {
    return { ok: false, reason: "The query plan must contain only approved fields." };
  }
  if (value.version !== 1 || (value.dataset !== "players" && value.dataset !== "player_champions")) {
    return { ok: false, reason: "The query plan version or dataset is unsupported." };
  }
  if (value.operation !== "rank" && value.operation !== "compare") {
    return { ok: false, reason: "The query operation is unsupported." };
  }
  if (!Array.isArray(value.filters) || value.filters.length > 12) {
    return { ok: false, reason: "The query filters are invalid." };
  }
  const filters: QueryFilter[] = [];
  for (const rawFilter of value.filters) {
    if (!isRecord(rawFilter) || !hasOnlyKeys(rawFilter, FILTER_KEYS)) {
      return { ok: false, reason: "A query filter contains unsupported fields." };
    }
    const field = rawFilter.field as QueryField;
    const op = rawFilter.op as QueryOperator;
    if (!DATASET_FIELDS[value.dataset].has(field)) {
      return { ok: false, reason: `Field ${String(rawFilter.field)} is unavailable for ${value.dataset}.` };
    }
    if (!(["eq", "in", "gte", "lte"] as string[]).includes(op)) {
      return { ok: false, reason: "A query filter operator is unsupported." };
    }
    if (STRING_FIELDS.has(field)) {
      const valid = op === "in"
        ? Array.isArray(rawFilter.value) && rawFilter.value.length > 0 && rawFilter.value.length <= 20
          && rawFilter.value.every((entry) => typeof entry === "string" && entry.trim().length > 0)
        : op === "eq" && typeof rawFilter.value === "string" && rawFilter.value.trim().length > 0;
      if (!valid) return { ok: false, reason: `Filter ${field} has an invalid value or operator.` };
    } else if (!NUMBER_FIELDS.has(field) || !finiteNumber(rawFilter.value) || !["eq", "gte", "lte"].includes(op)) {
      return { ok: false, reason: `Filter ${field} has an invalid value or operator.` };
    }
    filters.push({ field, op, value: rawFilter.value as string | number | string[] });
  }
  if (!Array.isArray(value.orderBy) || value.orderBy.length < 1 || value.orderBy.length > 3) {
    return { ok: false, reason: "The query order is invalid." };
  }
  const orderBy: QueryPlan["orderBy"] = [];
  for (const rawOrder of value.orderBy) {
    if (!isRecord(rawOrder) || !hasOnlyKeys(rawOrder, ORDER_KEYS)) {
      return { ok: false, reason: "A query order contains unsupported fields." };
    }
    const field = rawOrder.field as QueryField;
    const direction = rawOrder.direction as QueryDirection;
    if (!DATASET_FIELDS[value.dataset].has(field) || !NUMBER_FIELDS.has(field) || !["asc", "desc"].includes(direction)) {
      return { ok: false, reason: "A query order is unsupported." };
    }
    orderBy.push({ field, direction });
  }
  if (!Number.isInteger(value.limit) || Number(value.limit) < 1 || Number(value.limit) > 20) {
    return { ok: false, reason: "The query result limit must be between 1 and 20." };
  }
  const plan: QueryPlan = {
    version: 1,
    dataset: value.dataset,
    operation: value.operation,
    filters,
    orderBy,
    limit: Number(value.limit),
  };
  if (plan.operation === "compare") {
    const names = plan.filters.find((filter) => filter.field === "name" && filter.op === "in")?.value;
    if (!Array.isArray(names) || names.length < 2) {
      return { ok: false, reason: "A comparison requires at least two resolved player names." };
    }
  }
  return { ok: true, plan };
}

function mentionedEntities(question: string, values: string[]): string[] {
  const withoutPossessives = question.replace(/([\p{L}\p{N}])(?:[’']s|s[’'])\b/gu, "$1");
  const normalizedQuestion = ` ${normalizeEntity(withoutPossessives)} `;
  return values
    .map((value) => ({ value, key: normalizeEntity(value) }))
    .filter(({ key }) => key.length >= 2 && normalizedQuestion.includes(` ${key} `))
    .sort((left, right) => right.key.length - left.key.length)
    .filter((entry, index, entries) =>
      !entries.slice(0, index).some((prior) => prior.key.includes(entry.key)))
    .map(({ value }) => value);
}

function resolvePlayerTarget(
  value: string,
  index: SupportQueryIndex,
  team: string | null,
): { ok: true; name: string } | { ok: false; reason: string } {
  const target = normalizeEntity(value
    .replace(/\?+$/g, "")
    .replace(/\s+(?:rating|ratings|win rate|wr|games|maps)$/i, "")
    .replace(/^(?:the\s+)?player\s+/i, "")
    .trim());
  let candidates = index.players.filter((player) => normalizeEntity(player.name) === target);
  if (team) {
    candidates = candidates.filter((player) => normalizeEntity(player.team ?? "") === normalizeEntity(team));
  }
  if (candidates.length === 1) return { ok: true, name: candidates[0].name };
  if (candidates.length > 1) {
    return { ok: false, reason: "That player handle is ambiguous. Add the current team to the question." };
  }
  return { ok: false, reason: "I could not resolve that player name in the active public release." };
}

function namedMetricTarget(question: string): string | null {
  const stripped = question.trim()
    .replace(/^(?:what(?:'s| is)|show me|give me|tell me|rank|list)\s+/i, "")
    .replace(/\?+$/g, "")
    .trim();
  const possessive = /^(.+?)[’']s\s+/.exec(stripped);
  const metricOf = /^(?:rating|win rate|games|maps|a grades?)\s+of\s+(.+)$/i.exec(stripped);
  const trailingMetric = /^(.+?)\s+(?:rating|win rate|games|maps|a grades?)$/i.exec(stripped);
  const subjectFor = /\b(?:champions?|performance)\s+(?:for|of)\s+(.+)$/i.exec(stripped);
  const target = possessive?.[1] ?? metricOf?.[1] ?? trailingMetric?.[1] ?? subjectFor?.[1] ?? null;
  if (!target || /\b(best|highest|top|most|average)\b/i.test(target)) return null;
  if (/^(?:the\s+)?(?:patch|current patch|overall|general|all time)$/i.test(target.trim())) return null;
  return target;
}

function comparisonTargetNames(question: string): [string, string] | null {
  const text = question.trim().replace(/\?+$/g, "");
  const patterns = [
    /\bbetween\s+(.+?)\s+(?:and|or|vs\.?|versus)\s+(.+)$/i,
    /\bcompare\s+(.+?)\s+(?:and|or|vs\.?|versus)\s+(.+)$/i,
    /\b(?:better|higher)(?:\s+(?:rating|rated|win rate|wr))?[,;:\s]+(.+?)\s+(?:and|or|vs\.?|versus)\s+(.+)$/i,
  ];
  for (const pattern of patterns) {
    const match = pattern.exec(text);
    if (match) return [match[1].trim(), match[2].trim()];
  }
  return null;
}

function detectRole(question: string): string | null {
  const normalized = normalizeEntity(question);
  if (/\b(jungle|jungler|jng)\b/.test(normalized)) return "jng";
  if (/\b(support|sup)\b/.test(normalized)) return "sup";
  if (/\b(mid|middle|mid laner)\b/.test(normalized)) return "mid";
  if (/\b(bot|adc|marksman|bot laner)\b/.test(normalized)) return "bot";
  if (/\b(top|top laner)\b/.test(normalized)) return "top";
  return null;
}

function detectMinimumGames(question: string): number | null {
  const match = /(?:at least|minimum|min\.?|over)\s+(\d+)\s+(?:games|maps)/i.exec(question)
    ?? /(\d+)\s*\+\s*(?:games|maps)/i.exec(question);
  if (!match) return null;
  const value = Number.parseInt(match[1], 10);
  return Number.isFinite(value) && value > 0 ? Math.min(value, 10_000) : null;
}

function detectLimit(question: string, fallback = 5): number {
  const match = /\b(?:top|best|bottom|worst)\s+(\d+)\b/i.exec(question);
  if (!match) return fallback;
  return Math.min(Math.max(Number.parseInt(match[1], 10) || fallback, 1), 20);
}

function metricFor(
  question: string,
  championQuery: boolean,
  hasNamedChampion: boolean,
  hasNamedPlayer: boolean,
): QueryField {
  const normalized = normalizeEntity(question);
  if (/\ba grade|grade a\b/.test(normalized)) return "grade_a_games";
  if (championQuery && /\bmedian\b|\bmiddle(?:most)?\b|\bmidpoint\b/.test(normalized)) return "champion_median_distance";
  if (championQuery && /\baverage performance\b|\bmean performance\b/.test(normalized)) return "champion_mean_distance";
  if (/\bwin rate|\bwr\b/.test(normalized)) return championQuery ? "champion_win_rate" : "win_rate";
  if (/\bmost games|\bmost maps|\bexperience|\bexperienced\b/.test(normalized)) {
    return championQuery ? "champion_games" : "games";
  }
  if (/\brating|\brated\b/.test(normalized)) return "rating";
  if (championQuery && hasNamedPlayer && !hasNamedChampion) return "champion_win_rate";
  return championQuery ? "champion_score" : "rating";
}

function directionFor(question: string): QueryDirection {
  if (/\bmedian\b|\bmiddle(?:most)?\b|\bmidpoint\b|\bmean\b|\baverage performance\b|\bclosest to\b/i.test(question)) return "asc";
  if (/\b(?:best|highest|top)\b.*\bto\b.*\b(?:worst|lowest|bottom)\b/i.test(question)) return "desc";
  if (/\b(?:worst|lowest|bottom)\b.*\bto\b.*\b(?:best|highest|top)\b/i.test(question)) return "asc";
  const directionalText = question.replace(/\bat least\b/gi, "");
  return /\b(worst|lowest|bottom|least)\b/i.test(directionalText) ? "asc" : "desc";
}

function medianValue(values: number[]): number | null {
  if (!values.length) return null;
  const middle = Math.floor(values.length / 2);
  return values.length % 2 === 1
    ? values[middle]
    : (values[middle - 1] + values[middle]) / 2;
}

export function planPlayerQuestion(
  question: string,
  index: SupportQueryIndex,
): { ok: true; plan: QueryPlan } | { ok: false; reason: string } {
  const text = question.trim();
  if (text.length < 3 || text.length > 500) return { ok: false, reason: "The player question has an invalid length." };
  let players = mentionedEntities(text, index.entities.players);
  const champions = mentionedEntities(text, index.entities.champions);
  const teams = mentionedEntities(text, index.entities.teams);
  const leagues = mentionedEntities(text, index.entities.leagues);
  const namedTarget = namedMetricTarget(text);
  if (namedTarget) {
    const resolved = resolvePlayerTarget(namedTarget, index, teams[0] ?? null);
    if (!resolved.ok) return resolved;
    players = [resolved.name];
  }
  const champion = champions[0] ?? null;
  const championQuery = Boolean(champion) || /\bchampions?\b/i.test(text);
  const metric = metricFor(text, championQuery, Boolean(champion), players.length > 0);
  const direction = directionFor(text);
  const comparisonLanguage = /\b(compare|between|versus|vs|better|higher)\b|\sor\s/i.test(text);
  const comparisonTargets = comparisonLanguage ? comparisonTargetNames(text) : null;
  if (comparisonTargets) {
    const resolved = comparisonTargets.map((target) => resolvePlayerTarget(target, index, null));
    if (resolved.some((entry) => !entry.ok)) {
      return { ok: false, reason: "I could not resolve two published player names for that comparison." };
    }
    players = resolved.map((entry) => entry.ok ? entry.name : "");
  }
  const operation: QueryOperation = comparisonLanguage || players.length >= 2 ? "compare" : "rank";
  if (comparisonLanguage && players.length < 2) {
    return { ok: false, reason: "I could not resolve two published player names for that comparison." };
  }

  const dataset: QueryDataset = championQuery ? "player_champions" : "players";
  const filters: QueryFilter[] = [];
  if (players.length) filters.push({ field: "name", op: players.length > 1 ? "in" : "eq", value: players.length > 1 ? players : players[0] });
  if (champion) filters.push({ field: "champion", op: "eq", value: champion });
  if (operation === "rank" && players.length === 0) filters.push({ field: "active", op: "eq", value: 1 });
  const role = detectRole(text);
  if (role) filters.push({ field: "role", op: "eq", value: role });
  if (teams[0]) filters.push({ field: "team", op: "eq", value: teams[0] });
  if (leagues[0]) filters.push({ field: "league", op: "eq", value: leagues[0] });
  const tierMatch = /\btier\s*([123])\b/i.exec(text);
  if (tierMatch) {
    filters.push({ field: "tier", op: "eq", value: `tier${tierMatch[1]}` });
  } else if (operation === "rank" && players.length === 0 && !/\ball tiers\b/i.test(text)) {
    filters.push({ field: "tier", op: "eq", value: "tier1" });
  }
  const minimumGames = detectMinimumGames(text);
  if (minimumGames != null) {
    filters.push({ field: championQuery ? "champion_games" : "games", op: "gte", value: minimumGames });
  } else if (championQuery) {
    filters.push({ field: "champion_games", op: "gte", value: 5 });
  }

  const rawPlan: QueryPlan = {
    version: 1,
    dataset,
    operation,
    filters,
    orderBy: [
      { field: metric, direction },
      { field: championQuery ? "champion_games" : "games", direction: "desc" },
    ],
    limit: operation === "compare"
      ? Math.min(players.length, 20)
      : detectLimit(text, championQuery && players.length === 1 ? 20 : 5),
  };
  return parseQueryPlan(rawPlan);
}

function rowValue(row: QueryPlayerRow, field: QueryField): string | number | null {
  return row[field as keyof QueryPlayerRow] as string | number | null;
}

function matchesFilter(row: QueryPlayerRow, filter: QueryFilter): boolean {
  const value = rowValue(row, filter.field);
  if (value == null) return false;
  if (filter.op === "in" && Array.isArray(filter.value)) {
    if (filter.field === "name") return typeof value === "string" && filter.value.includes(value);
    const candidates = filter.value.map((entry) => normalizeEntity(entry));
    return typeof value === "string" && candidates.includes(normalizeEntity(value));
  }
  if (filter.op === "eq") {
    if (filter.field === "name" && typeof filter.value === "string" && typeof value === "string") {
      return value === filter.value;
    }
    return typeof filter.value === "string" && typeof value === "string"
      ? normalizeEntity(value) === normalizeEntity(filter.value)
      : value === filter.value;
  }
  if (typeof value !== "number" || typeof filter.value !== "number") return false;
  return filter.op === "gte" ? value >= filter.value : value <= filter.value;
}

function compareRows(left: QueryPlayerRow, right: QueryPlayerRow, orderBy: QueryPlan["orderBy"]): number {
  for (const order of orderBy) {
    const leftValue = rowValue(left, order.field);
    const rightValue = rowValue(right, order.field);
    if (leftValue == null && rightValue == null) continue;
    if (leftValue == null) return 1;
    if (rightValue == null) return -1;
    const comparison = typeof leftValue === "number" && typeof rightValue === "number"
      ? leftValue - rightValue
      : String(leftValue).localeCompare(String(rightValue));
    if (comparison !== 0) return order.direction === "desc" ? -comparison : comparison;
  }
  return left.name.localeCompare(right.name);
}

function filterDescription(plan: QueryPlan): string {
  const descriptions: string[] = [];
  for (const filter of plan.filters) {
    if (filter.field === "name") continue;
    if (filter.field === "active") descriptions.push("active competitors");
    else if (filter.field === "tier" && typeof filter.value === "string") descriptions.push(filter.value.replace("tier", "Tier "));
    else if (filter.field === "role") descriptions.push(`${String(filter.value)} role`);
    else if (filter.field === "league") descriptions.push(String(filter.value));
    else if (filter.field === "team") descriptions.push(String(filter.value));
    else if (filter.field === "champion") descriptions.push(`${String(filter.value)} games`);
    else if (filter.op === "gte") descriptions.push(`at least ${String(filter.value)} ${filter.field.replaceAll("_", " ")}`);
  }
  return descriptions.length ? descriptions.join(", ") : "the active public release";
}

function answerFor(
  plan: QueryPlan,
  rows: QueryPlayerRow[],
  total: number,
  allRows: QueryPlayerRow[] = rows,
): QueryAnswer {
  if (!rows.length) {
    return {
      headline: "No published player rows match those constraints.",
      basis: `Applied ${filterDescription(plan)}.`,
      caveat: "Unavailable values were excluded from the requested ordering.",
    };
  }
  const top = rows[0];
  const metric = plan.orderBy[0].field;
  const direction = plan.orderBy[0].direction;
  const rankWord = direction === "asc" ? "lowest" : "highest";
  const value = rowValue(top, metric);
  const rounded = typeof value === "number" ? Math.round(value) : null;
  let headline: string;
  if (plan.operation === "compare") {
    const runnerUp = rows[1];
    const runnerValue = runnerUp ? rowValue(runnerUp, metric) : null;
    const difference = typeof value === "number" && typeof runnerValue === "number"
      ? Math.abs(value - runnerValue)
      : null;
    if (!runnerUp || difference == null) {
      headline = "Published values are unavailable for a complete comparison.";
    } else if (difference < Number.EPSILON) {
      headline = `${top.name} and ${runnerUp.name} have the same published ${metric.replaceAll("_", " ")}.`;
    } else if (metric === "win_rate" || metric === "champion_win_rate" || metric === "champion_score") {
      const percentagePoints = Math.round(difference * 100);
      headline = `${top.name} ranks higher than ${runnerUp.name} by ${percentagePoints} percentage ${percentagePoints === 1 ? "point" : "points"} on ${metric.replaceAll("_", " ")}.`;
    } else {
      const points = Math.round(difference);
      headline = `${top.name} ranks higher than ${runnerUp.name} by ${points} ${metric.replaceAll("_", " ")} ${points === 1 ? "point" : "points"}.`;
    }
  } else if (metric === "champion_median_distance" || metric === "champion_mean_distance") {
    const values = allRows
      .map((row) => row.champion_win_rate)
      .filter((candidate): candidate is number => candidate != null && Number.isFinite(candidate))
      .sort((left, right) => left - right);
    const reference = values.length
      ? metric === "champion_median_distance"
        ? medianValue(values)
        : values.reduce((sum, candidate) => sum + candidate, 0) / values.length
      : null;
    const label = metric === "champion_median_distance" ? "median" : "mean";
    headline = reference == null
      ? `${top.name}'s ${label}-performance champion is ${top.champion}.`
      : `${top.name}'s ${label}-performance champion is ${top.champion} at ${Math.round(Number(top.champion_win_rate) * 100)}%, closest to the ${label} champion win rate of ${Math.round(reference * 100)}%.`;
  } else if (plan.dataset === "player_champions" && plan.filters.some((filter) => filter.field === "name" && filter.op === "eq")) {
    if (metric === "champion_win_rate") {
      headline = `${top.name}'s ${rankWord} published champion win rate is ${Math.round(Number(value) * 100)}% on ${top.champion} across ${top.champion_games ?? "—"} games.`;
    } else if (metric === "champion_games") {
      headline = `${top.name} has the ${direction === "asc" ? "fewest" : "most"} published games on ${top.champion}, with ${rounded ?? "—"}.`;
    } else {
      headline = `${top.name}'s ${rankWord} matching ${metric.replaceAll("_", " ")} is on ${top.champion}.`;
    }
  } else if (metric === "champion_score") {
    headline = `${top.name} ranks ${direction === "asc" ? "last" : "first"} for ${top.champion} under the evidence rule.`;
  } else if (metric === "champion_win_rate" || metric === "win_rate") {
    headline = `${top.name} has the ${rankWord} matching win rate at ${Math.round(Number(value) * 100)}%.`;
  } else if (metric === "champion_games" || metric === "games" || metric === "grade_a_games") {
    headline = `${top.name} ranks ${direction === "asc" ? "last" : "first"} with ${rounded ?? "—"} ${metric.replaceAll("_", " ")}.`;
  } else {
    headline = `${top.name} has the ${rankWord} matching published rating at ${rounded ?? "—"}.`;
  }
  const championEvidence = plan.dataset === "player_champions";
  const subject = championEvidence ? "matching player-champion records" : "matching players";
  return {
    headline,
    basis: championEvidence && metric === "champion_score"
      ? `Ranked ${total} ${subject} by the 95% Wilson lower bound on champion win rate; ${filterDescription(plan)}.`
      : `Ranked ${total} ${subject} by ${metric === "champion_median_distance"
        ? "distance from the median champion win rate"
        : metric === "champion_mean_distance"
          ? "distance from the mean champion win rate"
          : metric.replaceAll("_", " ")}; ${filterDescription(plan)}.`,
    caveat: championEvidence
      ? "This is a descriptive player-champion record, not a champion-specific rating or causal estimate."
      : null,
  };
}

export function executeQueryPlan(planInput: unknown, index: SupportQueryIndex): QueryExecutionResult {
  const parsed = parseQueryPlan(planInput);
  if (!parsed.ok) throw new Error(parsed.reason);
  const plan = parsed.plan;
  const source = plan.dataset === "player_champions" ? index.playerChampions : index.players;
  const baseMatching = source.filter((row) => plan.filters.every((filter) => matchesFilter(row, filter)));
  const derivedField = plan.orderBy[0].field;
  const matching = derivedField === "champion_median_distance" || derivedField === "champion_mean_distance"
    ? (() => {
      const values = baseMatching
        .map((row) => row.champion_win_rate)
        .filter((candidate): candidate is number => candidate != null && Number.isFinite(candidate))
        .sort((left, right) => left - right);
      const reference = values.length
        ? derivedField === "champion_median_distance"
          ? medianValue(values)
          : values.reduce((sum, candidate) => sum + candidate, 0) / values.length
        : null;
      return baseMatching.map((row) => ({
        ...row,
        champion_median_distance: derivedField === "champion_median_distance" && reference != null && row.champion_win_rate != null
          ? Math.abs(row.champion_win_rate - reference)
          : null,
        champion_mean_distance: derivedField === "champion_mean_distance" && reference != null && row.champion_win_rate != null
          ? Math.abs(row.champion_win_rate - reference)
          : null,
      }));
    })()
    : baseMatching;
  const rows = [...matching]
    .sort((left, right) => compareRows(left, right, plan.orderBy))
    .slice(0, plan.limit);
  return {
    kind: "player_query",
    plan,
    answer: answerFor(plan, rows, matching.length, matching),
    rows,
    proof: { sources: QUERY_SOURCES, resultCount: matching.length },
  };
}
