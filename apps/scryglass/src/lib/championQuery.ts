import type { QueryPlayerRow, SupportQueryIndex } from "./supportQuery";

export type ChampionQueryDirection = "asc" | "desc";
export type ChampionQueryMetric = "tier" | "win_rate" | "games";
export type ChampionTier = "tier1" | "tier2" | "tier3" | "all";

export type ChampionQueryRow = {
  champion: string;
  role: string | null;
  tier_bucket: string | null;
  rank: number | null;
  patch: string | null;
  games: number | null;
  wins: number | null;
  win_rate: number | null;
  players: number | null;
};

export type ChampionQueryResult = {
  kind: "champion_query";
  metric: ChampionQueryMetric;
  direction: ChampionQueryDirection;
  tier: ChampionTier;
  role: string | null;
  minimumGames: number;
  answer: {
    headline: string;
    basis: string;
    caveat: string;
  };
  rows: ChampionQueryRow[];
  proof: {
    sources: string[];
    resultCount: number;
  };
};

type Aggregate = {
  champion: string;
  games: number;
  wins: number;
  players: Set<string>;
};

export type PublishedTierBoard = {
  options?: { patches?: string[] };
  rows: Array<{
    champion?: string;
    role?: string;
    patch?: string;
    rank?: number;
    tier_bucket?: string;
    played_maps?: number;
  }>;
};

const SOURCES = [
  "features/player_champion_records.json",
  "features/player_records.json",
  "features/player_ratings_snapshot.json",
];
const TIER_SOURCE = "rankings/tierlists.json";

function normalized(value: string): string {
  return value
    .normalize("NFKD")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .replace(/\s+/g, " ");
}

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function roleFor(value: string | null): string | null {
  const role = normalized(value ?? "");
  if (role === "jungle" || role === "jungler" || role === "jng") return "jng";
  if (role === "support" || role === "sup") return "sup";
  if (role === "mid" || role === "middle" || role === "mid laner") return "mid";
  if (role === "bot" || role === "adc" || role === "marksman" || role === "bot laner") return "bot";
  if (role === "top" || role === "top laner") return "top";
  return role || null;
}

function detectRole(question: string): string | null {
  const text = normalized(question);
  if (/\b(jungle|jungler|jng)\b/.test(text)) return "jng";
  if (/\b(support|sup)\b/.test(text)) return "sup";
  if (/\b(mid|middle|mid laner)\b/.test(text)) return "mid";
  if (/\b(bot|adc|marksman|bot laner)\b/.test(text)) return "bot";
  if (/\b(top|top laner)\b/.test(text)) return "top";
  return null;
}

function detectTier(question: string): ChampionTier {
  if (/\ball tiers\b/i.test(question)) return "all";
  const match = /\btier\s*([123])\b/i.exec(question);
  return match ? `tier${match[1]}` as Exclude<ChampionTier, "all"> : "tier1";
}

function detectMinimumGames(question: string): number | null {
  const match = /(?:at least|minimum|min\.?|over)\s+(\d+)\s+(?:games?|maps?)/i.exec(question)
    ?? /(\d+)\s*\+\s+(?:games?|maps?)/i.exec(question);
  if (!match) return null;
  const value = Number.parseInt(match[1], 10);
  return Number.isFinite(value) && value > 0 ? Math.min(value, 100_000) : null;
}

function detectLimit(question: string): number {
  const match = /\b(?:top|best|bottom|worst)\s+(\d+)\b/i.exec(question);
  if (!match) return 20;
  return Math.min(Math.max(Number.parseInt(match[1], 10) || 20, 1), 20);
}

function detectDirection(question: string): ChampionQueryDirection {
  if (/\b(?:best|highest|top)\b.*\bto\b.*\b(?:worst|lowest|bottom)\b/i.test(question)) return "desc";
  if (/\b(?:worst|lowest|bottom)\b.*\bto\b.*\b(?:best|highest|top)\b/i.test(question)) return "asc";
  return /\b(worst|lowest|bottom|least)\b/i.test(question.replace(/\bat least\b/gi, "")) ? "asc" : "desc";
}

function detectMetric(question: string): ChampionQueryMetric {
  if (/\bmost games?\b|\bmost maps?\b|\bmost played\b|\bhighest pick\b/i.test(question)) return "games";
  if (/\bwin\s*rate\b|\bwr\b|\bhighest win\b|\blowest win\b/i.test(question)) return "win_rate";
  return "tier";
}

function rowForAggregate(aggregate: Aggregate): ChampionQueryRow {
  return {
    champion: aggregate.champion,
    role: null,
    tier_bucket: null,
    rank: null,
    patch: null,
    games: aggregate.games,
    wins: aggregate.wins,
    win_rate: aggregate.wins / aggregate.games,
    players: aggregate.players.size,
  };
}

function eligibleRows(index: SupportQueryIndex, tier: ChampionTier, role: string | null): QueryPlayerRow[] {
  return index.playerChampions.filter((row) => (
    row.active === 1
    && (tier === "all" || row.tier === tier)
    && (!role || roleFor(row.role) === role)
    && row.champion != null
    && finite(row.champion_games)
    && finite(row.champion_wins)
  ));
}

function patchSort(left: string, right: string): number {
  const leftParts = left.split(/[.-]/).map((part) => Number.parseInt(part, 10));
  const rightParts = right.split(/[.-]/).map((part) => Number.parseInt(part, 10));
  return (leftParts[0] ?? 0) - (rightParts[0] ?? 0)
    || (leftParts[1] ?? 0) - (rightParts[1] ?? 0)
    || left.localeCompare(right);
}

function queryTierBoard(
  question: string,
  tier: ChampionTier,
  role: string | null,
  direction: ChampionQueryDirection,
  minimumGames: number,
  board: PublishedTierBoard | undefined,
): ChampionQueryResult {
  const base = {
    kind: "champion_query" as const,
    metric: "tier" as const,
    direction,
    tier,
    role,
    minimumGames,
  };
  if (!board?.rows?.length) {
    return {
      ...base,
      answer: {
        headline: "Tier lists are unavailable for the current release.",
        basis: "The published patch tier list could not be loaded.",
        caveat: "Best and worst champion rankings use the published patch tier list, not aggregate win rate.",
      },
      rows: [],
      proof: { sources: [TIER_SOURCE], resultCount: 0 },
    };
  }

  const patches = (board.options?.patches ?? [...new Set(board.rows.map((row) => row.patch).filter((patch): patch is string => Boolean(patch)))])
    .sort(patchSort);
  const patch = patches.at(-1) ?? null;
  if (!patch) {
    return {
      ...base,
      answer: {
        headline: "Tier lists are unavailable for the current release.",
        basis: "The published patch tier list has no usable patch identifier.",
        caveat: "Best and worst champion rankings use the published patch tier list, not aggregate win rate.",
      },
      rows: [],
      proof: { sources: [TIER_SOURCE], resultCount: 0 },
    };
  }

  const rows = board.rows
    .filter((row) => {
      const playedMaps = finite(row.played_maps) ? row.played_maps : 0;
      return row.patch === patch
        && Boolean(row.champion)
        && finite(row.rank)
        && (!role || roleFor(row.role ?? null) === role)
        && playedMaps >= minimumGames;
    })
    .sort((left, right) => {
      const rankOrder = (left.rank as number) - (right.rank as number);
      if (rankOrder !== 0) return direction === "desc" ? rankOrder : -rankOrder;
      const mapsOrder = (right.played_maps ?? 0) - (left.played_maps ?? 0);
      if (mapsOrder !== 0) return mapsOrder;
      return String(left.champion).localeCompare(String(right.champion));
    })
    .map((row): ChampionQueryRow => ({
      champion: String(row.champion),
      role: row.role ?? null,
      tier_bucket: row.tier_bucket ?? null,
      rank: row.rank as number,
      patch,
      games: finite(row.played_maps) ? row.played_maps : null,
      wins: null,
      win_rate: null,
      players: null,
    }));
  const visibleRows = rows.slice(0, detectLimit(question));
  const first = visibleRows[0];
  const rankWord = direction === "desc" ? "first" : "last";
  const roleLabel = role ? ` for the ${role} role` : "";
  const headline = first
    ? `${first.champion}${first.role ? ` (${first.role})` : ""} ranks ${rankWord} on the published ${patch} champion tier list.`
    : "No published tier-list rows match those constraints.";

  return {
    ...base,
    answer: {
      headline,
      basis: `Ranked ${rows.length} published champion-role entr${rows.length === 1 ? "y" : "ies"}${roleLabel} by tier-list rank for patch ${patch}, with at least ${minimumGames} played ${minimumGames === 1 ? "map" : "maps"}.`,
      caveat: "This is the published patch tier-list order. It is not a champion win rate or a player-champion record.",
    },
    rows: visibleRows,
    proof: { sources: [TIER_SOURCE], resultCount: rows.length },
  };
}

export function queryChampions(index: SupportQueryIndex, question: string, tierBoard?: PublishedTierBoard): ChampionQueryResult {
  const tier = detectTier(question);
  const role = detectRole(question);
  const metric = detectMetric(question);
  const direction = detectDirection(question);
  const minimumGames = detectMinimumGames(question) ?? (metric === "tier" ? 1 : 100);
  if (metric === "tier") return queryTierBoard(question, tier, role, direction, minimumGames, tierBoard);
  const aggregates = new Map<string, Aggregate>();

  for (const row of eligibleRows(index, tier, role)) {
    const champion = row.champion as string;
    const key = normalized(champion);
    const aggregate = aggregates.get(key) ?? {
      champion,
      games: 0,
      wins: 0,
      players: new Set<string>(),
    };
    aggregate.games += row.champion_games as number;
    aggregate.wins += row.champion_wins as number;
    aggregate.players.add(normalized(row.name));
    aggregates.set(key, aggregate);
  }

  const rows = Array.from(aggregates.values())
    .filter((aggregate) => aggregate.games >= minimumGames)
    .map(rowForAggregate)
    .sort((left, right) => {
      const leftValue = metric === "games" ? (left.games ?? 0) : (left.win_rate ?? 0);
      const rightValue = metric === "games" ? (right.games ?? 0) : (right.win_rate ?? 0);
      if (leftValue !== rightValue) return direction === "desc" ? rightValue - leftValue : leftValue - rightValue;
      if ((left.games ?? 0) !== (right.games ?? 0)) return (right.games ?? 0) - (left.games ?? 0);
      return left.champion.localeCompare(right.champion);
    });

  const visibleRows = rows.slice(0, detectLimit(question));
  const first = visibleRows[0];
  const rankWord = direction === "asc" ? "lowest" : "highest";
  const metricLabel = metric === "games" ? "published games" : "published win rate";
  const headline = first
    ? metric === "games"
      ? `${first.champion} has the ${rankWord} published game count at ${first.games} games.`
      : `${first.champion} has the ${rankWord} published ${tier === "all" ? "overall" : tier.replace("tier", "Tier ")} win rate at ${Math.round((first.win_rate ?? 0) * 100)}% across ${first.games} games.`
    : "No published champion rows match those constraints.";

  return {
    kind: "champion_query",
    metric,
    direction,
    tier,
    role,
    minimumGames,
    answer: {
      headline,
      basis: `Ranked ${rows.length} champion${rows.length === 1 ? "" : "s"} by ${metricLabel} from active ${tier === "all" ? "all-tier" : tier.replace("tier", "Tier ")} player-champion records${role ? ` for the ${role} role` : ""}, with at least ${minimumGames} games per champion.`,
      caveat: "This is an aggregate descriptive win rate. It does not isolate champion strength from player, team, role, opponent, or draft context.",
    },
    rows: visibleRows,
    proof: { sources: SOURCES, resultCount: rows.length },
  };
}
