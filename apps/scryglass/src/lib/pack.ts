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
  ratings?: {
    source_mode?: string;
    source_as_of?: string;
    window_years?: number[];
    map_rows?: number;
    team_rating_rows?: number;
    player_rating_rows?: number;
    claim_ceiling?: string;
  };
  ingest?: Record<string, unknown>;
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

export type TeamWeeklyRank = {
  rank: number;
  delta: number | null;
};

export type TeamWeeklyRanks = {
  as_of: string | null;
  previous_as_of: string | null;
  current_through?: string | null;
  by_team: Record<string, TeamWeeklyRank>;
};

export type PlayerRating = {
  player: string;
  mu_total: number;
  mu_regional: number;
  mu_meta: number;
  sigma: number;
  n_maps: number;
  last_team: string | null;
  evidence_interval_width?: number | null;
  evidence_precision_ratio?: number | null;
  evidence_stability?: number | null;
  evidence_freshness_days?: number | null;
  evidence_support_coverage?: number | null;
  evidence_fallback?: number | null;
  evidence_active?: number | null;
  evidence_disconnected?: number | null;
  evidence_ood?: number | null;
  evidence_state?: string | null;
};

export type PlayerWeeklyRank = {
  rank: number;
  delta: number | null;
};

export type PlayerWeeklyRanks = {
  as_of: string | null;
  previous_as_of: string | null;
  current_through?: string | null;
  by_player: Record<string, Partial<Record<CompetitionTier | "all", PlayerWeeklyRank>>>;
};

export type PlayerMetadata = {
  country?: string | null;
  country_code?: string | null;
  flag?: string | null;
};

export type PlayerChampionRecord = {
  champion: string;
  champion_image_url?: string | null;
  games: number;
  wins: number;
  losses: number;
  wr: number | null;
  kills: number | null;
  deaths: number | null;
  assists: number | null;
};

export type ProfileParticipant = {
  player: string;
  side: "Blue" | "Red";
  role: string;
  champion: string | null;
  kills: number | null;
  deaths: number | null;
  assists: number | null;
};

export type ProfileGame = {
  game_id: string;
  date: string;
  league: string;
  blue_team: string;
  red_team: string;
  blue_win: 0 | 1;
  players: ProfileParticipant[];
};

export type ProfileRecords = {
  schema_version: "scryglass:profile-records:v1";
  window_days: number;
  champion_images: Record<string, string>;
  games: Record<string, ProfileGame>;
  players: Record<string, string[]>;
  teams: Record<string, string[]>;
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
  wins: number;
  games: number;
  wr: number | null;
  by_league?: Record<string, { wins: number; games: number; wr: number | null }>;
  by_tier?: Record<string, { wins: number; games: number; wr: number | null }>;
};

export type PlayerRecord = {
  wins: number;
  games: number;
  wr: number | null;
  blue_games?: number;
  blue_wins?: number;
  blue_wr?: number | null;
  red_games?: number;
  red_wins?: number;
  red_wr?: number | null;
  roles?: string[];
  primary_role?: string | null;
  leagues?: string[];
  primary?: string | null;
  intl?: boolean;
  interregional?: boolean;
  current_league?: string | null;
  current_tier?: CompetitionTier | null;
  current_team?: string | null;
  current_date?: string | null;
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

export function packSourceUpdatedLabel(manifest: PackManifest): string | null {
  const raw = manifest.ratings?.source_as_of;
  if (!raw) return null;
  const value = new Date(raw);
  if (Number.isNaN(value.getTime())) return null;
  return value.toLocaleString("en-GB", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "UTC",
    timeZoneName: "short",
  });
}

/** Soft ranking: penalize high-σ so thin ladders don't outrank settled orgs. */
export function softMu(mu: number, sigma: number, floor = TEAM_SIGMA_MIN): number {
  return mu - Math.max(0, sigma - floor);
}

/** Keep the public player payload small while preserving its evidence contract. */
export function compactPlayerRatings(players: PlayerRating[]): PlayerRating[] {
  return players
    .filter((player) => (player.n_maps ?? 0) >= 5)
    .filter(
      (player) =>
        player.evidence_disconnected !== 1 &&
        player.evidence_state?.toLowerCase() !== "disconnected",
    )
    .map((player) => ({
      player: player.player,
      mu_total: player.mu_total,
      mu_regional: player.mu_regional,
      mu_meta: player.mu_meta,
      sigma: player.sigma,
      n_maps: player.n_maps,
      last_team: player.last_team,
      evidence_interval_width: player.evidence_interval_width,
      evidence_precision_ratio: player.evidence_precision_ratio,
      evidence_stability: player.evidence_stability,
      evidence_freshness_days: player.evidence_freshness_days,
      evidence_support_coverage: player.evidence_support_coverage,
      evidence_fallback: player.evidence_fallback,
      evidence_active: player.evidence_active,
      evidence_disconnected: player.evidence_disconnected,
      evidence_ood: player.evidence_ood,
      evidence_state: player.evidence_state,
    }))
    .sort(
      (a, b) =>
        softMu(b.mu_total, b.sigma, PLAYER_SIGMA_MIN) -
        softMu(a.mu_total, a.sigma, PLAYER_SIGMA_MIN),
    );
}

/** Conservative posterior display value for the hierarchical public ladder. */
export function adjustedRating(
  rating: Pick<TeamRating, "mu_total" | "sigma" | "rating_p10">,
  floor = TEAM_SIGMA_MIN,
): number {
  return Number.isFinite(rating.rating_p10) ? Number(rating.rating_p10) : softMu(rating.mu_total, rating.sigma, floor);
}

export function formatWr(wr: number | null | undefined): string {
  if (wr == null || !Number.isFinite(wr)) return "—";
  return `${(100 * wr).toFixed(1)}%`;
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
  } | undefined,
  selected: string[],
): boolean {
  if (!selected.length) return true;
  if (!rec) return false;
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
  const crossRegionMatches = !crossRegionSet.size && !internationalSet.size
    ? true
    : !crossRegionSet.size || (!!rec.interregional && crossRegionSet.has("AMERICAS"));
  const internationalMatches = !internationalSet.size
    ? true
    : (internationalSet.has("INTL") && (rec.intl || (rec.leagues || []).some(isIntlLeague))) ||
      [...internationalSet].some((event) => event !== "INTL" && (rec.leagues || []).some((league) => league === event));
  return tierMatches && regionalMatches && crossRegionMatches && internationalMatches;
}

export function scopedTeamWr(
  rec: TeamRecord | undefined,
  selected: string[],
): number | null {
  if (!rec) return null;
  if (!selected.length) return rec.wr;
  const tierSelected = selected.filter((scope) => scope.startsWith("TIER")).map((scope) => scope.toLowerCase().replace("tier", ""));
  const regionalSelected = selected.filter((scope) => REGION_LEAGUES.includes(scope as (typeof REGION_LEAGUES)[number]));
  const internationalSelected = selected.filter((scope) => scope === "INTL" || isIntlLeague(scope));
  if (regionalSelected.length) {
    const by = rec.by_league || {};
    let wins = 0;
    let games = 0;
    for (const league of regionalSelected) {
      const row = by[league];
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
