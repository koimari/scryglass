/** Pack manifest helpers (client + server). */

export type PackFile = {
  path: string;
  relative?: string;
  rows: number | null;
  cols: number | null;
  bytes: number;
  sha256: string;
  columns?: string[] | null;
};

export type PackManifest = {
  pack_id: string;
  schema_version: string;
  created_utc: string;
  data_as_of?: string | null;
  recent_activity_window_days?: number;
  current_tournament_as_of?: string | null;
  current_tournaments?: Record<string, string>;
  membership_note?: string;
  filters: {
    years: number[];
    leagues: string;
    leagues_note?: string;
  };
  identity?: {
    taxonomy_version?: string;
    team_key?: string;
    league_source?: string;
    deprecated_leagues?: Record<string, string>;
  };
  attribution: string;
  excluded: string[];
  base_url: string | null;
  total_bytes: number;
  total_files: number;
  files: PackFile[];
};

export type TeamRating = {
  team: string;
  team_key?: string;
  mu_total: number;
  mu_regional: number;
  mu_meta: number;
  sigma: number;
  rating_p10?: number;
  n_series?: number;
  n_maps?: number;
  international_series?: number;
  home_league?: string;
  model?: string;
};

export type PlayerRating = {
  player: string;
  mu_total: number;
  mu_regional: number;
  mu_meta: number;
  sigma: number;
  n_maps: number;
  last_team: string | null;
};

export type PlayerWeeklyRank = {
  rank: number;
  delta: number | null;
};

export type PlayerWeeklyRanks = {
  as_of: string | null;
  previous_as_of: string | null;
  by_player: Record<string, Partial<Record<CompetitionTier | "all", PlayerWeeklyRank>>>;
};

export type PlayerMetadata = {
  country?: string | null;
  country_code?: string | null;
  flag?: string | null;
};

export type EloCalibration = {
  team: { intercept: number; coef: number; temperature_400?: number };
  player: { intercept: number; coef: number; temperature_400?: number };
};

export type TeamRecord = {
  leagues: string[];
  source_leagues?: string[];
  primary: string | null;
  intl: boolean;
  interregional?: boolean;
  current_league?: string | null;
  current_tier?: CompetitionTier | null;
  current_team?: string | null;
  current_date?: string | null;
  current_tournament?: string | null;
  wins: number;
  games: number;
  wr: number | null;
  by_league?: Record<string, { wins: number; games: number; wr: number | null }>;
  by_tier?: Record<string, { wins: number; games: number; wr: number | null }>;
  by_tournament?: Record<string, { wins: number; games: number; wr: number | null }>;
};

export type PlayerRecord = {
  wins: number;
  games: number;
  wr: number | null;
  leagues?: string[];
  primary?: string | null;
  intl?: boolean;
  interregional?: boolean;
  current_league?: string | null;
  current_tier?: CompetitionTier | null;
  current_team?: string | null;
  current_date?: string | null;
  current_tournament?: string | null;
};

export type CompetitionTier = "tier1" | "tier2" | "tier3";

export const TIER_FILTERS = [
  { value: "TIER1", label: "Tier 1", description: "LCK, LPL, LEC, LCS, CBLOL, and LCP" },
  { value: "TIER2", label: "Tier 2", description: "Established regional and challenger circuits" },
  { value: "TIER3", label: "Tier 3", description: "Other domestic and developmental circuits" },
] as const;

/** Dual Elo σ floors from ratings_meta / player_ratings_meta. */
export const TEAM_SIGMA_MIN = 25;
export const PLAYER_SIGMA_MIN = 28;

export const INTL_LEAGUES = [
  "MSI",
  "EWC",
  "FST",
  "WORLDS",
  "IWC",
  "MSC",
  "EM",
  "ASIA MASTER",
  "ASIA MASTERS",
] as const;

export const MAJOR_REGIONAL_LEAGUES = ["LCK", "LPL", "LEC", "LCS", "CBLOL", "LCP"] as const;

export const SECONDARY_REGIONAL_LEAGUES = ["PCS", "VCS", "LJL", "TCL"] as const;

export const REGION_LEAGUES = [
  "LCK",
  "LPL",
  "LEC",
  "LCS",
  "CBLOL",
  "PCS",
  "VCS",
  "LJL",
  "LCP",
  "TCL",
] as const;

export const INTERREGIONAL_LEAGUES = ["AMERICAS"] as const;

export type TrustKind = "settled" | "thin" | "very_thin";

export type TrustInfo = {
  kind: TrustKind;
  label: string;
  headroom: number;
  sigma: number;
  floor: number;
  layman: string;
};

export async function loadManifest(origin = ""): Promise<PackManifest> {
  const res = await fetch(`${origin}/packs/manifest.json`, { cache: "no-store" });
  if (!res.ok) throw new Error(`manifest ${res.status}`);
  return res.json();
}

export function packUrl(manifest: PackManifest, relativePath: string): string {
  const base = (manifest.base_url || `/packs/${manifest.pack_id}`).replace(/\/$/, "");
  return `${base}/${relativePath.replace(/^\//, "")}`;
}

/** logit = a + b*(mu_diff/400); p = sigmoid(logit) for favorite when mu_diff>0 vs even foe */
export function eloToWinProb(
  mu: number,
  foeMu: number,
  cal: { intercept: number; coef: number },
): number {
  const muDiff = mu - foeMu;
  const logit = cal.intercept + cal.coef * (muDiff / 400);
  return 1 / (1 + Math.exp(-logit));
}

export function formatMb(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function teamSlug(name: string): string {
  return encodeURIComponent(name.trim());
}

export function playerSlug(name: string): string {
  return encodeURIComponent(name.trim());
}

export function packUpdatedLabel(manifest: PackManifest): string {
  const raw = manifest.created_utc || "";
  const d = raw.slice(0, 10);
  if (/^\d{4}-\d{2}-\d{2}$/.test(d)) {
    const [y, m, day] = d.split("-");
    return `${day}/${m}/${y}`;
  }
  // pack_id vYYYY.MM.DD or the freshness-aware vYYYY.MM.DD.HHMM form
  const m = /^v(\d{4})\.(\d{2})\.(\d{2})(?:\.\d{4})?$/.exec(manifest.pack_id);
  if (m) return `${m[3]}/${m[2]}/${m[1]}`;
  return manifest.pack_id;
}

function dateMs(value: string | null | undefined): number | null {
  if (!value) return null;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

/** True only when a record has a dated observation inside the pack's guard window. */
export function recordIsRecent(
  rec: { current_date?: string | null } | undefined,
  dataAsOf: string | null | undefined,
  windowDays = 90,
): boolean {
  const observedMs = dateMs(rec?.current_date);
  const asOfMs = dateMs(dataAsOf);
  if (observedMs == null || asOfMs == null) return false;
  const ageDays = (asOfMs - observedMs) / 86_400_000;
  return ageDays >= 0 && ageDays <= Math.max(0, windowDays);
}

/** Soft ranking: penalize high-σ so thin ladders don't outrank settled orgs. */
export function softMu(mu: number, sigma: number, floor = TEAM_SIGMA_MIN): number {
  return mu - Math.max(0, sigma - floor);
}

/** Conservative posterior display value for the hierarchical public ladder. */
export function adjustedRating(
  rating: Pick<TeamRating, "mu_total" | "sigma" | "rating_p10">,
  floor = TEAM_SIGMA_MIN,
): number {
  return Number.isFinite(rating.rating_p10) ? Number(rating.rating_p10) : softMu(rating.mu_total, rating.sigma, floor);
}

export function trustInfo(sigma: number, floor: number, games?: number | null): TrustInfo {
  const headroom = Math.max(0, sigma - floor);
  let kind: TrustKind = "settled";
  if (headroom > 20) kind = "very_thin";
  else if (headroom > 0.5) kind = "thin";
  const label = kind === "settled" ? "Settled" : kind === "thin" ? "Thin" : "Very thin";
  const gamesBit =
    games != null && games > 0 ? ` · ${games} game${games === 1 ? "" : "s"} in sample` : "";
  const layman =
    kind === "settled"
      ? `Rating is as settled as this model allows${gamesBit}.`
      : kind === "thin"
        ? `Still moving — fewer informative games than a settled org${gamesBit}.`
        : `Very thin sample — treat the number gently${gamesBit}.`;
  return { kind, label, headroom, sigma, floor, layman };
}

export function formatWr(wr: number | null | undefined): string {
  if (wr == null || !Number.isFinite(wr)) return "—";
  return `${(100 * wr).toFixed(1)}%`;
}

export function formatTrustCell(info: TrustInfo): string {
  return info.label;
}

/** Common team aliases for fuzzy ladder / filter matching. */
const TEAM_ALIASES: Record<string, string> = {
  kc: "Karmine Corp",
  "karmine corp": "Karmine Corp",
  dk: "Dplus Kia",
  "dplus kia": "Dplus Kia",
  mkoi: "Movistar KOI",
  koi: "Movistar KOI",
  "movistar koi": "Movistar KOI",
  t1: "T1",
  gen: "Gen.G",
  geng: "Gen.G",
  "gen.g": "Gen.G",
  hle: "Hanwha Life Esports",
  "hanwha life": "Hanwha Life Esports",
  g2: "G2 Esports",
  fnc: "Fnatic",
  c9: "Cloud9",
  tl: "Team Liquid",
  "100t": "100 Thieves",
  "100 thieves": "100 Thieves",
  blg: "Bilibili Gaming",
  jdg: "JD Gaming",
  tes: "Top Esports",
  lng: "LNG Esports",
  we: "Team WE",
  ig: "Invictus Gaming",
  rng: "Royal Never Give Up",
  dwg: "Dplus Kia",
  drx: "DRX",
  ktz: "KT Rolster",
  kt: "KT Rolster",
  nsf: "Nongshim RedForce",
  ns: "Nongshim RedForce",
  bro: "OKSavingsBank BRION",
  fox: "BNK FEARX",
  fly: "FlyQuest",
  sr: "Shopify Rebellion",
  dig: "Dignitas",
  lrx: "LOUD",
  pai: "paiN Gaming",
  fur: "FURIA",
  red: "RED Canids",
  vibe: "Vivo Keyd Stars",
};

function normKey(name: string): string {
  return (name || "")
    .normalize("NFKC")
    .trim()
    .toLowerCase()
    .replace(/[’`]/g, "'")
    .replace(/\s+/g, " ");
}

export function expandTeamQuery(q: string): string[] {
  const raw = q.trim();
  if (!raw) return [];
  const key = normKey(raw);
  const canon = TEAM_ALIASES[key];
  const out = [raw.toLowerCase()];
  if (canon) out.push(canon.toLowerCase(), normKey(canon));
  out.push(key);
  return [...new Set(out)];
}

export function teamMatchesQuery(team: string, q: string): boolean {
  const needles = expandTeamQuery(q);
  if (!needles.length) return true;
  const hay = team.toLowerCase();
  const hayKey = normKey(team);
  return needles.some((n) => hay.includes(n) || hayKey.includes(n) || n.includes(hayKey));
}

export function playerMatchesQuery(player: string, lastTeam: string | null, q: string): boolean {
  const needle = q.trim().toLowerCase();
  if (!needle) return true;
  if (player.toLowerCase().includes(needle)) return true;
  if (lastTeam && teamMatchesQuery(lastTeam, q)) return true;
  return false;
}

export function isIntlLeague(league: string): boolean {
  const u = league.toUpperCase();
  return INTL_LEAGUES.some((event) => u === event || u.includes(event)) || u.includes("WORLD") || u.includes("FIRST STAND");
}

export function recordMatchesLeagues(
  rec: {
    leagues?: string[];
    intl?: boolean;
    interregional?: boolean;
    primary?: string | null;
    current_league?: string | null;
    current_tier?: CompetitionTier | null;
    current_date?: string | null;
    current_tournament?: string | null;
  } | undefined,
  selected: string[],
  options?: {
    dataAsOf?: string | null;
    recentActivityWindowDays?: number;
    currentTournaments?: Record<string, string>;
  },
): boolean {
  if (!selected.length) return true;
  if (!rec) return false;
  if (
    options?.dataAsOf &&
    !recordIsRecent(rec, options.dataAsOf, options.recentActivityWindowDays ?? 90)
  ) {
    return false;
  }
  const tierSet = new Set(selected.filter((scope) => scope.startsWith("TIER")));
  const regionalSet = new Set(selected.filter((scope) => REGION_LEAGUES.includes(scope as (typeof REGION_LEAGUES)[number])));
  const crossRegionSet = new Set(selected.filter((scope) => INTERREGIONAL_LEAGUES.includes(scope as (typeof INTERREGIONAL_LEAGUES)[number])));
  const internationalSet = new Set(selected.filter((scope) => scope === "INTL" || isIntlLeague(scope)));
  const currentLeague = rec.current_league ?? rec.primary;
  const currentTier = rec.current_tier?.toUpperCase();

  // Filters in one group are alternatives; filters across groups narrow the
  // same current affiliation/evidence record. This prevents selecting Tier 1
  // from being silently overridden by a Tier 2 regional chip.
  const tierMatches = !tierSet.size || (!!currentTier && tierSet.has(currentTier));
  const regionalMatches = !regionalSet.size || (!!currentLeague && regionalSet.has(currentLeague));
  const expectedTournament = currentLeague ? options?.currentTournaments?.[currentLeague] : undefined;
  const currentTournamentMatches = !expectedTournament || rec.current_tournament === expectedTournament;
  const crossRegionMatches = !crossRegionSet.size && !internationalSet.size
    ? true
    : !crossRegionSet.size || (!!rec.interregional && crossRegionSet.has("AMERICAS"));
  const internationalMatches = !internationalSet.size
    ? true
    : (internationalSet.has("INTL") && (rec.intl || (rec.leagues || []).some(isIntlLeague))) ||
      [...internationalSet].some((event) => event !== "INTL" && (rec.leagues || []).some((league) => league === event));
  const domesticScopeMatches = (!tierSet.size && !regionalSet.size) || currentTournamentMatches;
  return tierMatches && regionalMatches && domesticScopeMatches && crossRegionMatches && internationalMatches;
}

function scopedTournamentRow(
  rec: TeamRecord,
  league: string,
  currentTournaments?: Record<string, string>,
): { wins: number; games: number; wr: number | null } | null {
  const expected = currentTournaments?.[league];
  if (!expected) return rec.by_league?.[league] ?? null;
  if (rec.current_league !== league || rec.current_tournament !== expected) return null;
  return rec.by_tournament?.[`${league}|${expected}`] ?? null;
}

export function scopedTeamWr(
  rec: TeamRecord | undefined,
  selected: string[],
  options?: { currentTournaments?: Record<string, string> },
): number | null {
  if (!rec) return null;
  if (!selected.length) return rec.wr;
  const tierSelected = selected.filter((scope) => scope.startsWith("TIER")).map((scope) => scope.toLowerCase().replace("tier", ""));
  const regionalSelected = selected.filter((scope) => REGION_LEAGUES.includes(scope as (typeof REGION_LEAGUES)[number]));
  const internationalSelected = selected.filter((scope) => scope === "INTL" || isIntlLeague(scope));
  if (regionalSelected.length) {
    let wins = 0;
    let games = 0;
    for (const league of regionalSelected) {
      const row = scopedTournamentRow(rec, league, options?.currentTournaments);
      if (!row) continue;
      wins += row.wins;
      games += row.games;
    }
    return games ? wins / games : null;
  }
  if (internationalSelected.length) {
    const by = rec.by_league || {};
    let wins = 0;
    let games = 0;
    for (const [league, row] of Object.entries(by)) {
      const wanted = internationalSelected.includes("INTL") || internationalSelected.includes(league);
      if (!wanted || !isIntlLeague(league)) continue;
      wins += row.wins;
      games += row.games;
    }
    return games ? wins / games : null;
  }
  if (tierSelected.length && rec.by_tier) {
    const expected = rec.current_league ? options?.currentTournaments?.[rec.current_league] : undefined;
    if (expected) {
      const row = scopedTournamentRow(rec, rec.current_league!, options?.currentTournaments);
      if (!row) return null;
      return row.games ? row.wins / row.games : null;
    }
    let wins = 0;
    let games = 0;
    for (const tier of tierSelected) {
      const row = rec.by_tier[tier];
      if (!row) continue;
      wins += row.wins;
      games += row.games;
    }
    return games ? wins / games : null;
  }
  let wins = 0;
  let games = 0;
  const by = rec.by_league || {};
  for (const [lg, row] of Object.entries(by)) {
    const want =
      (selected.includes("INTL") && isIntlLeague(lg)) || selected.includes(lg);
    if (!want) continue;
    wins += row.wins;
    games += row.games;
  }
  if (!games) return null;
  return wins / games;
}
