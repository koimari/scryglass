/** Support-chat router: natural language -> tool calls, executed against the
 * real /api/chat endpoints. The model (Cactus/Needle WASM) only EMITS tool
 * calls; this module executes them and the UI renders executed results.
 *
 * Two implementations behind one interface:
 *   - needleRoute: the on-device Cactus/Needle WASM adapter (OpenAI-compatible
 *     tool calling), loaded async with graceful degradation.
 *   - fallbackRoute: a deterministic keyword/regex router that always works.
 */

export type ToolName =
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
  { name: "leaderboards", description: "Top players by A-grade games, rating, or win rate; optionally filtered by role and competitive tier.", args: [{ name: "category", description: "a_grades | rating | win_rate | teams" }, { name: "role", description: "top | jng | mid | bot | sup (optional)" }, { name: "tier", description: "tier1 | tier2 | tier3 (optional; use tier1 by default)" }, { name: "limit", description: "number of results (optional)" }] },
  { name: "player", description: "Player profile: rating, role, team, grades, win rate, recent form.", args: [{ name: "name", description: "player name" }] },
  { name: "compare_players", description: "Compare the ratings of two named players and answer which rating is higher.", args: [{ name: "player1", description: "first player name" }, { name: "player2", description: "second player name" }] },
  { name: "team", description: "Team profile: rating, record, recent results.", args: [{ name: "name", description: "team name" }] },
  { name: "matches", description: "Recent completed matches, optionally filtered by team, league, or champion.", args: [{ name: "team", description: "team name (optional)" }, { name: "league", description: "league code such as LEC or LCK (optional)" }, { name: "champion", description: "champion name (optional)" }, { name: "limit", description: "number of matches (optional)" }] },
  { name: "tier", description: "Patch-wide champion tier list, optionally per role.", args: [{ name: "role", description: "top | jng | mid | bot | sup (optional)" }, { name: "patch", description: "patch such as 16.15 (optional)" }] },
  { name: "schedule", description: "Upcoming fixtures, optionally for a league.", args: [{ name: "league", description: "league or tournament (optional)" }] },
  { name: "methodology", description: "Explain ratings, grades, tier lists, the draft win share, matches, or schedules.", args: [{ name: "topic", description: "ratings | grades | tiers | draft | matches | schedule | all" }] },
  { name: "navigation", description: "What pages exist on the site and where to find something.", args: [] },
];

const TOOL_NAMES = new Set<string>(TOOLS.map((tool) => tool.name));

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

  return { explanation: "I can help with players, teams, matches, ratings, tier lists, schedules, and methodology. Try: \"who is the player with the most A grade games?\", \"show me T1's recent matches\", or \"how does the draft win share work?\"." };
}

// --- Cactus / Needle WASM adapter ------------------------------------------

/** Config for the on-device model; wire these to the hosted engine + weights. */
export const NEEDLE_CONFIG = {
  engineUrl: "/needle/engine.wasm", // Cactus engine WASM build
  modelUrl: "/needle/needle-model.bin", // Needle 2 weights (14 MB)
  enabled: true,
};

let needleState: "unloaded" | "loading" | "ready" | "failed" = "unloaded";
let needleModule: { call: (prompt: string) => Promise<string> } | null = null;

async function loadNeedleModule(): Promise<{ call: (prompt: string) => Promise<string> } | null> {
  // Dynamic import keeps the WASM engine out of the main bundle until first use.
  // The engine exposes an OpenAI-compatible chat/tool-calling entry point.
  const client = await import("./needleClient").then(
    (mod) => mod.createNeedleClient(NEEDLE_CONFIG),
    () => null,
  );
  return client;
}

export async function needleRoute(text: string): Promise<RouteResult | null> {
  if (!NEEDLE_CONFIG.enabled) return null;
  if (needleState === "unloaded") {
    needleState = "loading";
    try {
      needleModule = await loadNeedleModule();
      needleState = needleModule ? "ready" : "failed";
    } catch {
      needleState = "failed";
    }
  }
  if (needleState !== "ready" || !needleModule) return null;
  const schema = TOOLS.map(
    (tool) =>
      `${tool.name}: ${tool.description}${
        tool.args.length ? ` args(${tool.args.map((arg) => `${arg.name} = ${arg.description}`).join(", ")})` : " (no args)"
      }`,
  ).join("\n");
  const prompt = [
    "You route a question about the Scryglass League of Legends data site to exactly one tool.",
    "TOOLS:\n" + schema,
    "Respond with ONLY a JSON object: {\"tool\": \"<name>\", \"args\": {\"<arg>\": \"<value>\"}}.",
    "If the question is off-topic or ambiguous, respond {\"explanation\": \"...\"}.",
    "QUESTION: " + text,
  ].join("\n");
  const raw = await needleModule.call(prompt);
  try {
    const parsed = JSON.parse(raw) as { tool?: string; args?: Record<string, string>; explanation?: string };
    if (parsed.explanation) return { explanation: parsed.explanation };
    if (parsed.tool && TOOL_NAMES.has(parsed.tool)) {
      return { call: { tool: parsed.tool as ToolName, args: parsed.args ?? {} } };
    }
    return null;
  } catch {
    return null;
  }
}

/** Route a question: prefer the on-device model, fall back deterministically. */
export async function routeQuestion(text: string): Promise<RouteResult> {
  if (!text.trim()) return { explanation: "Ask me anything about players, teams, matches, ratings, tiers, schedules, or methodology." };
  const routed = await needleRoute(text);
  if (routed) return routed;
  return fallbackRoute(text);
}

// --- Execution --------------------------------------------------------------

export async function executeTool(call: ToolCall): Promise<unknown> {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(call.args)) {
    if (value) params.set(key, value);
  }
  const response = await fetch(`/api/chat/${call.tool}?${params.toString()}`);
  const payload = (await response.json()) as { ok: boolean; data?: unknown; reason?: string };
  if (!response.ok || !payload.ok) {
    throw new Error(payload.reason ?? `Chat tool ${call.tool} failed`);
  }
  return payload.data;
}
