/** Support-chat router: natural language -> validated read-only tool calls. */

export type ToolName =
  | "query_players"
  | "query_champions"
  | "query_drafts"
  | "leaderboards"
  | "player"
  | "compare_players"
  | "team"
  | "matches"
  | "tier"
  | "schedule"
  | "methodology"
  | "navigation";

export type ToolCall = { tool: ToolName; args: Record<string, string> };
export type RouteResult = { call: ToolCall } | { explanation: string };

export type ToolSpec = {
  name: ToolName;
  description: string;
  args: Array<{ name: string; description: string }>;
};

export const TOOLS: ToolSpec[] = [
  { name: "query_players", description: "General player-data query. Use for named player comparisons, filtered player rankings, ratings, win rates, experience, A grades, and best-player-on-champion questions. The server resolves entities and executes a validated query plan against published data.", args: [{ name: "q", description: "the complete original player question" }] },
  { name: "query_champions", description: "Query the published champion tier list for best, worst, top, bottom, or role-filtered champion questions. Use the descriptive player-champion records only when the question explicitly asks for win rate or games. The response states the patch, tier-list scope, and sample floor.", args: [{ name: "q", description: "the complete original champion question" }] },
  { name: "query_drafts", description: "Rank teams by descriptive draft edge or compare two named teams over complete published drafts. Use for best, worst, top, bottom, named-team, between-team, versus-team, or league-specific draft questions. The response states the scope, profile window, and sample floor.", args: [{ name: "q", description: "the complete original team draft question" }] },
  { name: "leaderboards", description: "Top players by A-grade games, rating, or win rate; top teams; or draft rankings. Team draft rankings use descriptive model-unit edge. Player rankings use best-available rate from published bans and tier pools. Filters support role and competitive tier.", args: [{ name: "category", description: "a_grades | rating | win_rate | teams | teams_draft | players_draft" }, { name: "role", description: "top | jng | mid | bot | sup (optional)" }, { name: "tier", description: "tier1 | tier2 | tier3 (optional; use tier1 by default)" }, { name: "limit", description: "number of results (optional)" }] },
  { name: "player", description: "Player profile: rating, role, team, grades, win rate, recent form.", args: [{ name: "name", description: "player name" }] },
  { name: "compare_players", description: "Compare the ratings of two named players and answer which rating is higher.", args: [{ name: "player1", description: "first player name" }, { name: "player2", description: "second player name" }] },
  { name: "team", description: "Team profile: rating, record, recent results.", args: [{ name: "name", description: "team name" }] },
  { name: "matches", description: "Recent completed matches, optionally filtered by team, league, or champion.", args: [{ name: "team", description: "team name (optional)" }, { name: "league", description: "league code such as LEC or LCK (optional)" }, { name: "champion", description: "champion name (optional)" }, { name: "limit", description: "number of matches (optional)" }] },
  { name: "tier", description: "Patch-wide champion tier list, optionally per role.", args: [{ name: "role", description: "top | jng | mid | bot | sup (optional)" }, { name: "patch", description: "public patch such as 26.15 (optional)" }] },
  { name: "schedule", description: "Upcoming fixtures, optionally for a league.", args: [{ name: "league", description: "league or tournament (optional)" }] },
  { name: "methodology", description: "Explain ratings, grades, tier lists, descriptive draft edge, matches, or schedules.", args: [{ name: "topic", description: "ratings | grades | tiers | draft | matches | schedule | all" }] },
  { name: "navigation", description: "What pages exist on the site and where to find something.", args: [] },
];

const TOOL_NAMES = new Set<string>(TOOLS.map((tool) => tool.name));
export const SUPPORT_QUESTION_MAX_CHARS = 500;
const SUPPORT_ARGUMENT_MAX_CHARS = 100;
const SUPPORT_TOOL_TIMEOUT_MS = 5_000;
// The planner query routes carry a 10s server budget (CHAT_QUERY_ROUTE_TIMEOUT_MS
// in chatApi.ts) because their non-trivial questions chain several sequential
// network legs. A 5s client abort would cancel exactly the slow-but-successful
// case that budget exists to recover, so the client allows the server budget
// plus transport overhead for these tools only.
const SUPPORT_PLANNER_TOOLS = new Set(["query_players", "query_champions", "query_drafts"]);
const SUPPORT_PLANNER_TOOL_TIMEOUT_MS = 12_000;

// --- Deterministic fallback router -----------------------------------------

function normalize(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9\s]/g, " ").replace(/\s+/g, " ").trim();
}

function matchName(text: string, candidates: string[]): string | null {
  const lower = normalize(text);
  for (const candidate of candidates) {
    const key = normalize(candidate);
    if (key && lower.includes(key)) return candidate;
  }
  return null;
}

const COMMON_LEAGUES = ["LEC", "LCK", "LPL", "LCS", "CBLOL", "PCS", "VCS", "LJL", "LTA", "EMEA", "WORLDS", "MSI", "EWC", "AMERICAS"];
const COMMON_CHAMPIONS = ["akali", "syndra", "ori", "yasuo", "jinx", "thresh", "leona", "ahri", "zed", "lux", "kaisa", "ezreal", "renekton", "wukong", "vi", "viego", "skarner", "naafiri"];
const COMMON_TEAMS = ["T1", "G2", "Fnatic", "Karmine", "Team Liquid", "Gen.G", "Hanwha", "Bilibili", "Top Esports", "DRX", "KC", "MKOI", "GAM", "PSG", "FNC", "C9", "100 Thieves", "Cloud9"];

function detectLeague(text: string): string | null {
  const lower = text.toUpperCase();
  for (const league of COMMON_LEAGUES) {
    if (new RegExp(`\\b${league}\\b`).test(lower)) return league;
  }
  return null;
}

function detectRole(text: string): string | null {
  const normalized = normalize(text);
  if (/jng|jungl/.test(normalized)) return "jng";
  if (/support|sup\b/.test(normalized)) return "sup";
  if (/mid/.test(normalized)) return "mid";
  if (/top/.test(normalized)) return "top";
  if (/bot|adc|marksman/.test(normalized)) return "bot";
  return null;
}

function detectChampion(text: string): string | null {
  const normalized = normalize(text);
  for (const champion of COMMON_CHAMPIONS) {
    if (normalized.includes(champion)) return champion;
  }
  return null;
}

function looksLikePlayerQuestion(text: string): boolean {
  const normalized = normalize(text);
  return /(who is|player|profile|rating of|most|best|top player)/.test(normalized);
}

function cleanRatingTarget(value: string): string | null {
  const target = value.trim().replace(/(?:'s|s')$/i, "").trim();
  if (target.length < 2 || /\d/.test(target) || /^(the|a|an|player|rating)$/.test(target)) return null;
  return target;
}

function ratingTarget(text: string): string | null {
  const patterns = [
    /^(?:what(?:'s| is)\s+)?(.+?)\s+rating\s*\??$/i,
    /^(?:what(?:'s| is)|how is)\s+(.+?)\s+rated\s*\??$/i,
    /^rating\s+of\s+(.+?)\s*\??$/i,
  ];
  for (const pattern of patterns) {
    const match = pattern.exec(text.trim());
    const target = match ? cleanRatingTarget(match[1]) : null;
    if (target) return target;
  }
  return null;
}

function deterministicDataRoute(text: string): RouteResult | null {
  const lower = text.toLowerCase();
  const draftRanking = /\bdraft\b/.test(lower)
    && /\b(?:best|worst|better|worse|highest|lowest|top|bottom|score|scores|points?|pts|rank|ranking|rankings|between|compare|versus|vs\.?|team|teams)\b/.test(lower)
    && !/\b(?:how does|how is|what does|explain|methodology|work|computed|mean)\b/.test(lower);
  if (draftRanking) {
    const archiveList = /\b(?:best|worst|highest|lowest|top|bottom|rank|ranking|rankings|leaderboard|list|strongest|weakest)\b/.test(lower)
      && !/\b(?:between|compare|versus|vs\.?|vs |against)\b/.test(lower);
    if (archiveList) {
      const teamsOnly = /\bteams?\b/.test(lower) && !/\bplayers?\b/.test(lower);
      return { call: { tool: "leaderboards", args: { category: teamsOnly ? "teams_draft" : "players_draft", tier: "tier1", limit: "10" } } };
    }
    return { call: { tool: "query_drafts", args: { q: text.trim() } } };
  }
  if (
    /[’']s\s+(?:best|worst|top|bottom|highest|lowest|most|least|median|average|middle)\b.*\bchampions?\b/i.test(text)
    || /\b(?:rank|show|list)\b.*[’']s\s+champions?\b/i.test(text)
    || (
      /\b(?:best|worst|top|bottom|highest|lowest|most|least|median|average|middle)\b.*\bchampions?\b.*\b(?:for|of)\b/i.test(text)
      && !/\b(?:patch|overall|general|all time)\b/i.test(text)
    )
  ) {
    return { call: { tool: "query_players", args: { q: text.trim() } } };
  }
  if (
    /\bchampions?\b/.test(lower)
    && /\b(?:best|worst|top|bottom|highest|lowest|most|least|median|average|middle|rank|ranking|played)\b/.test(lower)
    && !/\b(?:this patch|in patch|current patch)\b/.test(lower)
  ) {
    return { call: { tool: "query_champions", args: { q: text.trim() } } };
  }
  return null;
}

function comparisonTargets(text: string): [string, string] | null {
  const patterns = [
    /better\s+rating\s*[,;:]?\s*(.+?)\s+(?:or|vs\.?|versus)\s+(.+?)\s*\??$/i,
    /compare\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+?)(?:\s+ratings?)?\s*\??$/i,
  ];
  for (const pattern of patterns) {
    const match = pattern.exec(text.trim());
    if (!match) continue;
    const first = cleanRatingTarget(match[1]);
    const second = cleanRatingTarget(match[2]);
    if (first && second) return [first, second];
  }
  return null;
}

export function fallbackRoute(text: string): RouteResult {
  const lower = text.toLowerCase();

  const deterministic = deterministicDataRoute(text);
  if (deterministic) return deterministic;

  const generalPlayerQuery = /(player|laner|jungler|support|adc|rating|rated|win rate|\bwr\b|a grade|best .* on|best .* player|better|compare|between|(?:best|top|highest).*\b(?:mid|top|jungle|bot|support)\b|tier\s*[123].*\b(?:mid|top|jungle|bot|support)\b)/.test(lower)
    && !/(rating of \d|rated \d|with a rating)/.test(lower)
    && !/(recent match|recent game|schedule|fixture|when does|methodology|how does|what does|explain|how do|\bwork\b|how is .* computed|tier list|best .* champion this patch)/.test(lower);
  if (generalPlayerQuery) {
    return { call: { tool: "query_players", args: { q: text.trim() } } };
  }

  const comparedPlayers = comparisonTargets(text);
  if (comparedPlayers) {
    return { call: { tool: "compare_players", args: { player1: comparedPlayers[0], player2: comparedPlayers[1] } } };
  }

  const namedRating = ratingTarget(text);
  if (namedRating) {
    const team = matchName(namedRating, COMMON_TEAMS);
    return team
      ? { call: { tool: "team", args: { name: team } } }
      : { call: { tool: "player", args: { name: namedRating } } };
  }

  // methodology / explanation questions
  if (/(how does|how is|what does|explain|how do|methodology|work\b)/.test(lower)) {
    const topic = /draft|win share|composition/.test(lower)
      ? "draft"
      : /grade/.test(lower)
        ? "grades"
        : /tier/.test(lower)
          ? "tiers"
          : /schedule|fixture|next game/.test(lower)
            ? "schedule"
            : /match|game/.test(lower)
              ? "matches"
              : /rating|elo/.test(lower)
                ? "ratings"
                : "all";
    return { call: { tool: "methodology", args: { topic } } };
  }

  // navigation
  if (/(where|find|navigate|pages|section|support|help me find)/.test(lower)) {
    return { call: { tool: "navigation", args: {} } };
  }

  // schedule
  if (/(when does|next game|upcoming|fixture|schedule|next match|playing next)/.test(lower)) {
    const league = detectLeague(text);
    return { call: { tool: "schedule", args: league ? { league } : {} } };
  }

  // Player-role rankings. "Mid laner" and similar phrases refer to players.
  if (/(best|top|highest|ranked)/.test(lower) && /(laner|player|jungler|support|adc)/.test(lower)) {
    const role = detectRole(text);
    return { call: { tool: "leaderboards", args: { category: "rating", tier: "tier1", limit: "5", ...(role ? { role } : {}) } } };
  }

  // Champion tier lists.
  if (/(tier|tier list|best .* (champion|pick).* (this patch|in patch)|patch .* best .* (champion|pick))/.test(lower)) {
    const role = detectRole(text);
    return { call: { tool: "tier", args: role ? { role } : {} } };
  }

  // leaderboards (aggregates: most A grades, best rating, win rate, rating lookup)
  if (/(most|top|best|leaderboard|highest|ranked|who has)/.test(lower) && /(grade|rating|win rate|wr|player)/.test(lower)) {
    const category = /(a grade|grade a|grades)/.test(lower)
      ? "a_grades"
      : /(win rate|wr)/.test(lower)
        ? "win_rate"
        : "rating";
    const role = detectRole(text);
    return { call: { tool: "leaderboards", args: { category, tier: "tier1", ...(role ? { role } : {}) } } };
  }
  // "top N <entity>" / "best <entity>" -> leaderboards with the right category
  const topMatch = /(?:top|best|highest)\s+(\d+)?\s*(team|teams|player|players|champion|champions)/.exec(lower);
  if (topMatch) {
    const limit = topMatch[1] ? Math.min(Math.max(parseInt(topMatch[1], 10) || 1, 1), 25) : 5;
    const entity = topMatch[2];
    const role = entity.includes("jungl") ? "jng" : entity.includes("mid") ? "mid" : entity.includes("support") ? "sup" : entity.includes("top laner") ? "top" : entity.includes("adc") || entity.includes("bot") ? "bot" : null;
    if (entity.startsWith("team")) {
      return { call: { tool: "leaderboards", args: { category: "teams", tier: "tier1", limit: String(limit) } } };
    }
    const args: Record<string, string> = { category: "rating", tier: "tier1", limit: String(limit) };
    if (role) args.role = role;
    return { call: { tool: "leaderboards", args } };
  }

  // with-a-rating-of-<value> lookups -> per-role leaderboard index
  if (/(with a rating|rating of \d|rated \d|rating is \d)/.test(lower)) {
    const category = "rating";
    const role = detectRole(text);
    return { call: { tool: "leaderboards", args: { category, tier: "tier1", ...(role ? { role } : {}) } } };
  }

  // matches by league/champion/team
  const league = detectLeague(text);
  const champion = detectChampion(text);
  if (league || champion || /(matches|games|results|played|recent)/.test(lower)) {
    const args: Record<string, string> = {};
    if (league) args.league = league;
    if (champion) args.champion = champion;
    const team = matchName(text, COMMON_TEAMS);
    if (team) args.team = team;
    args.limit = "10";
    return { call: { tool: "matches", args } };
  }

  // team profile questions
  if (/team|profile|tell me about/.test(lower)) {
    const team = matchName(text, COMMON_TEAMS);
    if (team) return { call: { tool: "team", args: { name: team } } };
  }

  // player questions (fallback default for identity questions)
  if (looksLikePlayerQuestion(text)) {
    const name = text.replace(/^(who is|player|profile of|tell me about|rating of)\s+/i, "").trim();
    return name ? { call: { tool: "player", args: { name } } } : { call: { tool: "leaderboards", args: { category: "rating" } } };
  }

  return { explanation: "I can help with players, champions, teams, matches, ratings, tier lists, schedules, methodology, and Draft Score availability. Try: \"what is the worst champion in general?\", \"show me T1's recent matches\", or \"is Draft Score available?\"." };
}

/** Route a question through the deterministic, auditable query router. */
export async function routeQuestion(text: string): Promise<RouteResult> {
  const question = text.trim();
  if (!question) return { explanation: "Ask me anything about players, teams, matches, ratings, tiers, schedules, or methodology." };
  if (Array.from(question).length > SUPPORT_QUESTION_MAX_CHARS) {
    return { explanation: `Questions can contain up to ${SUPPORT_QUESTION_MAX_CHARS} characters.` };
  }
  const deterministic = deterministicDataRoute(question);
  if (deterministic) return deterministic;
  return fallbackRoute(question);
}

// --- Execution --------------------------------------------------------------

export async function executeTool(call: ToolCall): Promise<unknown> {
  if (!TOOL_NAMES.has(call.tool)) throw new Error("The chat tool is unsupported");
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(call.args)) {
    const maximum = key === "q" ? SUPPORT_QUESTION_MAX_CHARS : SUPPORT_ARGUMENT_MAX_CHARS;
    if (Array.from(value).length > maximum) throw new Error("A chat tool argument is too long");
    if (value) params.set(key, value);
  }
  const toolTimeoutMs = SUPPORT_PLANNER_TOOLS.has(call.tool)
    ? SUPPORT_PLANNER_TOOL_TIMEOUT_MS
    : SUPPORT_TOOL_TIMEOUT_MS;
  const response = await fetch(`/api/chat/${call.tool}?${params.toString()}`, {
    headers: { Accept: "application/json" },
    signal: AbortSignal.timeout(toolTimeoutMs),
  });
  const payload = (await response.json()) as { ok: boolean; data?: unknown; reason?: string };
  if (!response.ok || !payload.ok) {
    throw new Error(payload.reason ?? `Chat tool ${call.tool} failed`);
  }
  return payload.data;
}
