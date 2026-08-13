import { chatError, chatJson, clean, readChatJson, searchParams, secureChatRoute } from "@/lib/chatApi";
import { getChampionAggregates, getTierRows, queryApiAvailable } from "@/lib/publicData";
import { loadSupportQueryIndex } from "@/lib/supportQuery";
import { queryChampions, type PublishedTierBoard } from "@/lib/championQuery";
import { readPackManifest } from "@/lib/serverPack";

export const runtime = "nodejs";

function normalized(value: string): string {
  return value.normalize("NFKD").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

function roleFromQuestion(question: string): string | null {
  const text = normalized(question);
  if (/\b(jungle|jungler|jng)\b/.test(text)) return "jng";
  if (/\b(support|sup)\b/.test(text)) return "sup";
  if (/\b(mid|middle|mid laner)\b/.test(text)) return "mid";
  if (/\b(bot|adc|marksman|bot laner)\b/.test(text)) return "bot";
  if (/\b(top|top laner)\b/.test(text)) return "top";
  return null;
}

function minimumGames(question: string, fallback: number): number {
  const match = /(?:at least|minimum|min\.?|over)\s+(\d+)\s+(?:games?|maps?)/i.exec(question)
    ?? /(\d+)\s*\+\s*(?:games?|maps?)/i.exec(question);
  return match ? Math.min(Math.max(Number.parseInt(match[1], 10) || fallback, 1), 100_000) : fallback;
}

function resultLimit(question: string): number {
  const match = /\b(?:top|best|bottom|worst)\s+(\d+)\b/i.exec(question);
  return Math.min(Math.max(match ? Number.parseInt(match[1], 10) || 20 : 20, 1), 20);
}

function worstFirst(question: string): boolean {
  return /\b(worst|lowest|bottom|least)\b/i.test(question.replace(/\bat least\b/gi, ""));
}

function tierScope(question: string): "tier1" | "tier2" | "tier3" | "all" {
  if (/\ball tiers\b/i.test(question)) return "all";
  const match = /\btier\s*([123])\b/i.exec(question);
  return match ? `tier${match[1]}` as "tier1" | "tier2" | "tier3" : "tier1";
}

async function queryBoundedChampions(question: string, signal?: AbortSignal) {
  const manifest = await readPackManifest(signal);
  const index = await loadSupportQueryIndex(signal);
  const text = ` ${normalized(question)} `;
  const leagues = index.entities.leagues.filter((league) => text.includes(` ${normalized(league)} `));
  const regions = ["Americas", "EMEA", "Asia", "International"]
    .filter((region) => text.includes(` ${normalized(region)} `));
  const role = roleFromQuestion(question);
  const tier = tierScope(question);
  const direction = worstFirst(question) ? "asc" as const : "desc" as const;
  const limit = resultLimit(question);
  const explicitGames = /\bmost games?\b|\bmost maps?\b|\bmost played\b|\bhighest pick\b/i.test(question);
  const explicitWinRate = /\bwin\s*rate\b|\bwr\b|\bhighest win\b|\blowest win\b/i.test(question);

  if (!explicitGames && !explicitWinRate) {
    const minGames = minimumGames(question, 1);
    const result = await getTierRows(manifest, {
      kind: "champion",
      regions,
      leagues,
      tiers: tier === "all" ? [] : [tier],
      roles: role ? [role] : [],
      minGames,
      order: direction === "desc" ? "rank_asc" : "rank_desc",
      limit,
      offset: 0,
    }, signal);
    const rows = result.rows.map((row) => ({
      champion: row.name,
      role: row.role,
      tier_bucket: typeof row.payload.tier_bucket === "string" ? row.payload.tier_bucket : null,
      rank: row.rank,
      patch: row.patch,
      games: row.played_maps,
      wins: null,
      win_rate: null,
      players: null,
    }));
    const first = rows[0];
    return {
      kind: "champion_query",
      metric: "tier",
      direction,
      tier,
      role,
      minimumGames: minGames,
      answer: {
        headline: first
          ? `${first.champion}${first.role ? ` (${first.role})` : ""} ranks ${direction === "desc" ? "first" : "last"} on the published ${first.patch ?? "current"} champion tier list.`
          : "No published tier-list rows match those constraints.",
        basis: `Ranked ${result.total} published champion-role entries by tier-list rank with at least ${minGames} played ${minGames === 1 ? "map" : "maps"}.`,
        caveat: "This is the published patch tier-list order. It is not a champion win rate or a player-champion record.",
      },
      rows,
      proof: { sources: ["rpc:get_scryglass_tier_rows"], resultCount: result.total },
    };
  }

  const metric = explicitGames ? "games" as const : "win_rate" as const;
  const minGames = minimumGames(question, 100);
  const result = await getChampionAggregates(manifest, {
    leagues,
    tiers: tier === "all" ? [] : [tier],
    roles: role ? [role] : [],
    active: true,
    minGames,
    order: metric === "games"
      ? direction === "desc" ? "games_desc" : "games_asc"
      : direction === "desc" ? "win_rate_desc" : "win_rate_asc",
    limit,
    offset: 0,
  }, signal);
  const rows = result.rows.map((row) => ({
    champion: row.champion,
    role: null,
    tier_bucket: null,
    rank: null,
    patch: null,
    games: row.games,
    wins: row.wins,
    win_rate: row.win_rate,
    players: row.players,
  }));
  const first = rows[0];
  return {
    kind: "champion_query",
    metric,
    direction,
    tier,
    role,
    minimumGames: minGames,
    answer: {
      headline: first
        ? metric === "games"
          ? `${first.champion} has the ${direction === "desc" ? "highest" : "lowest"} published game count at ${first.games} games.`
          : `${first.champion} has the ${direction === "desc" ? "highest" : "lowest"} published win rate at ${Math.round((first.win_rate ?? 0) * 100)}% across ${first.games} games.`
        : "No published champion rows match those constraints.",
      basis: `Ranked ${result.total} champions by published ${metric === "games" ? "games" : "win rate"}, with at least ${minGames} games per champion.`,
      caveat: "This is an aggregate descriptive record. It does not isolate champion strength from player, team, role, opponent, or draft context.",
    },
    rows,
    proof: { sources: ["rpc:get_scryglass_champions"], resultCount: result.total },
  };
}

async function get(request: Request) {
  const question = clean(searchParams(request).get("q"));
  if (!question) return chatError("A champion question is required.", 422);
  try {
    const manifest = await readPackManifest(request.signal);
    if (queryApiAvailable(manifest)) {
      return chatJson(await queryBoundedChampions(question, request.signal));
    }
    const [index, tierBoard] = await Promise.all([
      loadSupportQueryIndex(request.signal),
      readChatJson<PublishedTierBoard>("rankings/tierlists.json", request.signal),
    ]);
    return chatJson(queryChampions(index, question, tierBoard));
  } catch {
    return chatError("Champion rankings are unavailable for the active release.", 422);
  }
}

export const GET = secureChatRoute(get);
