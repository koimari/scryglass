import type { QueryPlayerRow, SupportQueryIndex } from "./supportQuery";

export type ChampionQueryDirection = "asc" | "desc";
export type ChampionQueryMetric = "win_rate" | "games";
export type ChampionTier = "tier1" | "tier2" | "tier3" | "all";

export type ChampionQueryRow = {
  champion: string;
  games: number;
  wins: number;
  win_rate: number;
  players: number;
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

const SOURCES = [
  "features/player_champion_records.json",
  "features/player_records.json",
  "features/player_ratings_snapshot.json",
];

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

function detectMinimumGames(question: string): number {
  const match = /(?:at least|minimum|min\.?|over)\s+(\d+)\s+(?:games?|maps?)/i.exec(question)
    ?? /(\d+)\s*\+\s+(?:games?|maps?)/i.exec(question);
  if (!match) return 100;
  const value = Number.parseInt(match[1], 10);
  return Number.isFinite(value) && value > 0 ? Math.min(value, 100_000) : 100;
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
  return /\bmost games?\b|\bmost maps?\b|\bmost played\b|\bhighest pick\b/i.test(question) ? "games" : "win_rate";
}

function rowForAggregate(aggregate: Aggregate): ChampionQueryRow {
  return {
    champion: aggregate.champion,
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

export function queryChampions(index: SupportQueryIndex, question: string): ChampionQueryResult {
  const tier = detectTier(question);
  const role = detectRole(question);
  const metric = detectMetric(question);
  const direction = detectDirection(question);
  const minimumGames = detectMinimumGames(question);
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
      const leftValue = metric === "games" ? left.games : left.win_rate;
      const rightValue = metric === "games" ? right.games : right.win_rate;
      if (leftValue !== rightValue) return direction === "desc" ? rightValue - leftValue : leftValue - rightValue;
      if (left.games !== right.games) return right.games - left.games;
      return left.champion.localeCompare(right.champion);
    });

  const visibleRows = rows.slice(0, detectLimit(question));
  const first = visibleRows[0];
  const rankWord = direction === "asc" ? "lowest" : "highest";
  const metricLabel = metric === "games" ? "published games" : "published win rate";
  const headline = first
    ? metric === "games"
      ? `${first.champion} has the ${rankWord} published game count at ${first.games} games.`
      : `${first.champion} has the ${rankWord} published ${tier === "all" ? "overall" : tier.replace("tier", "Tier ")} win rate at ${Math.round(first.win_rate * 100)}% across ${first.games} games.`
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
