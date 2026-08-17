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

export type CompositionSignalAudit = {
  schema_version?: string;
  model_version?: string;
  included_terms?: string[];
  excluded_terms?: string[];
  training_order?: string;
  status?: "available" | "limited" | "unavailable";
  published_status?: "available" | "limited" | "unavailable";
  target_games?: number;
  available_games?: number;
  limited_games?: number;
  unavailable_games?: number;
  published_games?: number;
  published_available_games?: number;
  published_limited_games?: number;
  published_unavailable_games?: number;
  fit_through?: string | null;
  source_as_of?: string | null;
  source_identity_sha256?: string | null;
  canonical_game_identity_sha256?: string | null;
  worker_commit?: string | null;
  cache_hits?: number;
  min_support_games?: number;
  regularization_c?: number;
  calibration_tolerance?: {
    slope?: number;
    intercept?: number;
  };
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
  data_backend?: "supabase" | "local";
  tier?: {
    status?: "available" | "unavailable";
    as_of?: string | null;
  };
  composition_signal?: CompositionSignalAudit;
  draft_pool?: {
    scope?: string;
    games?: number;
    complete_bans?: number;
    complete_pick_order?: number;
    quality_games?: number;
    quality_picks?: number;
    coverage?: number;
  };
  draft_authority?: {
    schema_version: "scryglass:draft-authority:v1";
    /**
     * `status` is retained for release compatibility. New releases should
     * also write `authority`. The web gate accepts the explicit authority
     * value when both fields are present and requires them to agree.
     */
    status: "unavailable" | "descriptive" | "promoted";
    authority?: "unavailable" | "descriptive" | "promoted";
    release_id: string;
    model_version: string | null;
    artifact_sha256?: string | null;
    receipt_sha256: string | null;
    issued_utc?: string | null;
    estimand?: string | null;
    probability_authority?: boolean;
    recommendation_authority?: boolean;
    betting_authority?: boolean;
    reason?: string | null;
    descriptive_authority?: {
      schema_version: "scryglass:draft-authority:v1";
      status: "descriptive";
      authority: "descriptive";
      release_id: string;
      model_version: string;
      artifact_sha256: string;
      receipt_sha256: string;
      issued_utc: string;
      estimand: "composition_only";
      probability_authority: false;
      recommendation_authority: false;
      betting_authority: false;
    };
  };
  query_api?: {
    schema_version?: "scryglass:query-api:v1";
    status?: "available" | "unavailable";
    projection?: {
      path?: string;
      bytes?: number;
      sha256?: string;
    };
    datasets?: Record<string, {
      schema_version?: string;
      dataset?: string;
      rows?: number;
      bytes?: number;
      sha256?: string;
    }>;
  };
  ratings?: {
    source_mode?: string;
    source_as_of?: string;
    source_game_count?: number;
    source_identity_sha256?: string;
    window_years?: number[];
    map_rows?: number;
    team_rating_rows?: number;
    player_rating_rows?: number;
    claim_ceiling?: string;
    momentum?: {
      schema_version?: string;
      window_games?: number;
      scale?: number;
      scale_unit?: string;
      status?: string;
      authority?: Record<string, boolean>;
      selected?: Record<string, unknown>;
      registered?: Record<string, unknown>;
      active?: Record<string, unknown>;
      candidate?: Record<string, unknown>;
      promotion?: Record<string, unknown>;
    };
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
  mu_base_total?: number;
  mu_effective?: number;
  momentum_residual?: number;
  mu_regional: number;
  mu_meta: number;
  sigma: number;
  rating_p10?: number;
  n_series?: number;
  n_maps?: number;
  international_series?: number;
  home_league?: string;
  model?: string;
  /**
   * Exact match-ready five from the v2 Team Rating publication contract
   * (issue #47).  Present only when an authorized exact-roster artifact is
   * published for this org; the descriptive snapshot rows never fill this.
   */
  exact_roster?: {
    roster_id: string;
    model_scope: "regional" | "global";
    players: { player_id: string; display_name: string; role: string }[];
    roster_effective_at: string;
    roster_as_of: string;
    roster_receipt_sha256: string;
    evidence_state: string;
  } | null;
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
  mu_base_total?: number;
  mu_effective?: number;
  momentum_residual?: number;
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

export type PlayerRankComparison = {
  as_of: string;
  rank: number | null;
  delta: number | null;
};

export type PlayerPositionDeltas = Partial<
  Record<"1m" | "3m" | "12m", PlayerRankComparison>
>;

export type PlayerWeeklyRank = {
  rank: number;
  delta: number | null;
  position_deltas?: PlayerPositionDeltas;
};

export type PlayerWeeklyRanks = {
  as_of: string | null;
  previous_as_of: string | null;
  current_through?: string | null;
  position_delta_as_of?: Partial<Record<"1m" | "3m" | "12m", string>>;
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
  team_kills?: number | null;
  cs?: number | null;
  cs_per_minute?: number | null;
  damage_per_minute?: number | null;
  damage_share?: number | null;
  gold?: number | null;
  gold_diff_at_10?: number | null;
  vision_score?: number | null;
  wards_placed?: number | null;
  grade?: ProfileGrade;
};

export type ProfileTeamStats = {
  kills?: number | null;
  gold?: number | null;
  dragons?: number | null;
  heralds?: number | null;
  void_grubs?: number | null;
  barons?: number | null;
  atakhans?: number | null;
  towers?: number | null;
  inhibitors?: number | null;
};

export type ProfileGrade =
  | {
      status: "available";
      grade: string;
      score: number;
      baseline_games: number;
      self_baseline_games: number;
      components: {
        self: number;
        team: number;
        opponent: number;
        league_role: number;
      };
    }
  | { status: "unavailable"; reason: string };

export type DraftContribution = {
  schema_version: "scryglass:draft-descriptive-signal:v1";
  status: "available" | "limited" | "unavailable";
  model_version: string;
  fit_through: string | null;
  archetype_interaction_source?: {
    id: string;
    status: string;
    lcc_atoms: "excluded";
    reason: string;
  };
  blue: {
    signal: number | null;
    prior_role_games: number;
    components?: {
      base: number | null;
      archetype_interactions: number | null;
      ally_synergy: number | null;
      enemy_counter: number | null;
      same_role: number | null;
    };
  };
  red: {
    signal: number | null;
    prior_role_games: number;
    components?: {
      base: number | null;
      archetype_interactions: number | null;
      ally_synergy: number | null;
      enemy_counter: number | null;
      same_role: number | null;
    };
  };
  edge_components?: {
    base: number | null;
    archetype_interactions: number | null;
    ally_synergy: number | null;
    enemy_counter: number | null;
    same_role: number | null;
    total: number | null;
  };
  player_comfort?: {
    status: "available" | "limited" | "unavailable" | string;
    contribution: number | null;
    source?: string | null;
    reason?: string | null;
  };
  picks: Array<{
    side: "Blue" | "Red";
    role: string;
    champion: string;
    contribution: number | null;
    prior_role_games: number;
    evidence_status: "available" | "atom_estimate" | "role_estimate" | "limited" | "unavailable";
    best_available?: boolean | null;
    tier_rank?: number | null;
    available_count?: number | null;
    components?: {
      base?: number | null;
      archetype_interactions?: number | null;
      ally_synergy?: number | null;
      enemy_counter?: number | null;
      same_role?: number | null;
      total?: number | null;
    };
  }>;
  note: string;
  reason?: string;
};

export type DraftPool = {
  schema_version: "scryglass:draft-pool:v1";
  status: "complete" | "limited" | "unavailable";
  source: "oracle-elixir" | "published-tier-list" | string;
  basis?: string;
  patch: string | null;
  bans: { Blue: string[]; Red: string[] };
  picked: Array<{
    side: "Blue" | "Red";
    role: string | null;
    champion: string;
    order: number | null;
    best_available?: boolean | null;
    tier_rank?: number | null;
    available_count?: number | null;
  }>;
  unpicked: string[];
  evaluated_picks?: number;
  reason?: string | null;
};

export type ProfileGame = {
  game_id: string;
  date: string;
  league: string;
  competition_tier?: string | null;
  patch?: string | null;
  blue_team: string;
  red_team: string;
  blue_win: 0 | 1;
  duration_seconds?: number | null;
  team_stats?: Partial<Record<"Blue" | "Red", ProfileTeamStats>>;
  players: ProfileParticipant[];
  draft_pool?: DraftPool;
  draft_contribution?: DraftContribution;
};

export type ProfileRecords = {
  schema_version: "scryglass:profile-records:v1" | "scryglass:profile-records:v2" | "scryglass:profile-records:v3";
  grade_contract?: "scryglass:player-map-grade:v1" | "scryglass:player-map-grade:v2";
  window_days: number;
  champion_images: Record<string, string>;
  games: Record<string, ProfileGame>;
  players: Record<string, string[]>;
  teams: Record<string, string[]>;
  draft_pool_audit?: {
    schema_version?: string;
    source?: string;
    games?: number;
    complete_bans?: number;
    complete_pick_order?: number;
    quality_games?: number;
    quality_picks?: number;
    coverage?: number;
  };
};

export type MatchSummary = {
  game_id: string;
  date: string;
  league: string;
  competition_tier?: string | null;
  blue_team: string;
  red_team: string;
  blue_win: 0 | 1;
  champions: string[];
  grades_available: number;
};

export type MatchIndex = {
  schema_version: "scryglass:match-index:v1";
  years: number[];
  games: MatchSummary[];
};

export type MatchRecords = {
  schema_version: "scryglass:match-records:v1";
  year: number;
  games: Record<string, ProfileGame>;
};

export type ScheduleSeries = {
  series_id: string;
  start_utc: string;
  has_time: boolean;
  status: "scheduled" | "live";
  team1: string;
  team2: string;
  best_of: number | null;
  tournament: string;
  overview_page: string;
  tournament_url: string | null;
  stage: string | null;
  region: "Americas" | "EMEA" | "Asia" | "International" | "Other";
  level: string | null;
};

export type ScheduleTournament = {
  name: string;
  overview_page: string;
  url: string;
  start_date: string;
  end_date: string;
  region: ScheduleSeries["region"];
  league: string | null;
  level: string | null;
  official: boolean;
  status: "upcoming" | "current" | "past";
};

export type PublicSchedule = {
  schema_version: "scryglass:public-schedule:v1";
  source: "Leaguepedia Cargo";
  source_url: string;
  as_of: string;
  refresh_status: "fresh" | "cached";
  upcoming: ScheduleSeries[];
  tournaments: ScheduleTournament[];
};

export function recentProfileGames(
  records: ProfileRecords,
  limit = 100,
): ProfileGame[] {
  if (!Number.isInteger(limit) || limit < 1) return [];
  return Object.values(records.games)
    .sort((a, b) => {
      const byDate = Date.parse(b.date) - Date.parse(a.date);
      return byDate || a.game_id.localeCompare(b.game_id);
    })
    .slice(0, limit);
}

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
  const clean = relativePath.replace(/^\/+/, "");
  if (!clean || clean.split("/").some((part) => !part || part === "." || part === "..")) {
    throw new Error("pack path is invalid");
  }
  if (manifest.data_backend === "supabase") {
    return `/api/assets/${encodeURIComponent(manifest.pack_id)}/${encodeURIComponent(clean)}`;
  }
  return `/packs/${encodeURIComponent(manifest.pack_id)}/${clean}`;
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

/** Prefer the source's exact casing when route names collide case-insensitively. */
export function findPlayerByRouteName(
  players: PlayerRating[],
  routeName: string,
): PlayerRating | undefined {
  const exact = players.find((player) => player.player === routeName);
  return exact ?? players.find((player) => player.player.toLowerCase() === routeName.toLowerCase());
}

/** Match a display identity to a record without making source casing part of the join. */
export function findRecordByName<T>(
  records: Record<string, T>,
  name: string,
): T | undefined {
  const exact = records[name];
  if (exact) return exact;
  const wanted = normKey(name);
  const entry = Object.entries(records).find(([candidate]) => normKey(candidate) === wanted);
  return entry?.[1];
}

export type DraftAuthorityStatus = "unavailable" | "descriptive" | "promoted";

const DRAFT_RECEIPT_SHA256 = /^[a-f0-9]{64}$/;
const DRAFT_ISSUED_UTC = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$/;

function declaredDraftAuthority(
  manifest: PackManifest,
): DraftAuthorityStatus | null {
  const authority = manifest.draft_authority;
  if (!authority || authority.schema_version !== "scryglass:draft-authority:v1") return null;
  const status = authority.authority ?? authority.status;
  if (authority.authority && authority.status !== authority.authority) return null;
  if (status !== "unavailable" && status !== "descriptive" && status !== "promoted") return null;
  return status;
}

/**
 * Validate the release binding that is safe for the browser to verify.
 *
 * The browser can verify the manifest release ID and the receipt digest shape.
 * It cannot verify the receipt contents. Probability authority therefore
 * remains behind the independent verifier used by the private release gate.
 */
function hasBoundDraftReceipt(manifest: PackManifest): boolean {
  const authority = manifest.draft_authority;
  return Boolean(
    authority
    && authority.release_id === manifest.pack_id
    && typeof authority.model_version === "string"
    && authority.model_version.trim().length > 0
    && typeof authority.artifact_sha256 === "string"
    && DRAFT_RECEIPT_SHA256.test(authority.artifact_sha256)
    && typeof authority.receipt_sha256 === "string"
    && DRAFT_RECEIPT_SHA256.test(authority.receipt_sha256)
    && typeof authority.issued_utc === "string"
    && DRAFT_ISSUED_UTC.test(authority.issued_utc)
    && Number.isFinite(Date.parse(authority.issued_utc))
  );
}

/** Return the manifest's verified public Draft state. */
export function draftAuthorityStatus(manifest: PackManifest): DraftAuthorityStatus {
  const status = declaredDraftAuthority(manifest);
  if (!status || status === "unavailable") return "unavailable";
  if (status === "promoted") {
    const authority = manifest.draft_authority;
    if (
      authority?.authority !== "promoted"
      || authority.estimand !== "prematch_map_win_probability_with_controlled_draft_intervention"
      || authority.probability_authority !== true
      || authority.recommendation_authority !== true
      || authority.betting_authority !== false
    ) return "unavailable";
  }
  return hasBoundDraftReceipt(manifest) ? status : "unavailable";
}

/** Descriptive Draft fields may render after their release-bound receipt passes. */
export function hasDescriptiveDraftAuthority(manifest: PackManifest): boolean {
  const status = draftAuthorityStatus(manifest);
  if (status === "descriptive") return true;
  const nested = manifest.draft_authority?.descriptive_authority;
  return Boolean(
    status === "promoted"
    && nested?.schema_version === "scryglass:draft-authority:v1"
    && nested.status === "descriptive"
    && nested.authority === "descriptive"
    && nested.release_id === manifest.pack_id
    && nested.estimand === "composition_only"
    && nested.model_version.trim()
    && DRAFT_RECEIPT_SHA256.test(nested.artifact_sha256)
    && DRAFT_RECEIPT_SHA256.test(nested.receipt_sha256)
    && DRAFT_ISSUED_UTC.test(nested.issued_utc)
    && Number.isFinite(Date.parse(nested.issued_utc))
    && nested.probability_authority === false
    && nested.recommendation_authority === false
    && nested.betting_authority === false
  );
}

/** Accept only the active manifest's complete release-bound promoted receipt. */
export function hasPromotedDraftAuthority(manifest: PackManifest): boolean {
  return draftAuthorityStatus(manifest) === "promoted";
}

export function hasPublishedDraftAuthority(manifest: PackManifest): boolean {
  return hasDescriptiveDraftAuthority(manifest) || hasPromotedDraftAuthority(manifest);
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

/** Public ladders contain current competitors. Historical rows remain available to direct profiles. */
export function isActiveRating(
  rating: Pick<TeamRating | PlayerRating, "evidence_active">,
): boolean {
  return rating.evidence_active === 1;
}

function championResultScore(record: PlayerChampionRecord): number {
  if (record.games <= 0) return 0;
  const rate = record.wins / record.games;
  const z = 1.2815515655446004;
  const zSquared = z * z;
  const denominator = 1 + zSquared / record.games;
  const centre = rate + zSquared / (2 * record.games);
  const spread = z * Math.sqrt(
    (rate * (1 - rate) + zSquared / (4 * record.games)) / record.games,
  );
  return (centre - spread) / denominator;
}

/** Rank champion results while discounting tiny samples. */
export function bestChampionRecords(
  records: PlayerChampionRecord[],
  limit = 5,
): PlayerChampionRecord[] {
  if (!Number.isInteger(limit) || limit < 1) return [];
  return [...records]
    .filter((record) => record.games > 0)
    .sort((a, b) => {
      const byScore = championResultScore(b) - championResultScore(a);
      if (byScore) return byScore;
      const byGames = b.games - a.games;
      if (byGames) return byGames;
      return a.champion.localeCompare(b.champion);
    })
    .slice(0, limit);
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
    .map((player) => {
      const compact: PlayerRating = {
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
      };
      if (player.mu_base_total !== undefined) compact.mu_base_total = player.mu_base_total;
      if (player.mu_effective !== undefined) compact.mu_effective = player.mu_effective;
      if (player.momentum_residual !== undefined) {
        compact.momentum_residual = player.momentum_residual;
      }
      return compact;
    })
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

/** Return the canonical team name and its known query aliases. */
export function teamQueryAliases(team: string): string[] {
  const canonical = TEAM_ALIASES[normKey(team)] ?? team;
  const canonicalKey = normKey(canonical);
  const aliases = Object.entries(TEAM_ALIASES)
    .filter(([, value]) => normKey(value) === canonicalKey)
    .map(([alias]) => alias);
  return [...new Set([team, canonical, ...aliases])];
}

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
  const tierSelected = selected
    .filter((scope) => scope.startsWith("TIER"))
    .map((scope) => scope.toLowerCase());
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
