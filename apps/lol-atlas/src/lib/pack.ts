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
  model_pack_id?: string | null;
  schema_version: string;
  created_utc: string;
  data_as_of?: string | null;
  recent_activity_window_days?: number;
  current_tournament_as_of?: string | null;
  current_tournaments?: Record<string, string>;
  membership_registry?: {
    snapshot_id?: string | null;
    authority?: string | null;
    checked_at?: string | null;
    review_due_at?: string | null;
    sources?: Record<string, string>;
    participants_by_league?: Record<string, string[]>;
  };
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
  source_summary?: {
    schema_version?: number;
    sources?: Record<string, unknown>;
    canonicalization?: Record<string, unknown>;
    attribution?: string;
  };
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
  sigma: number;
  rating_p05?: number;
  n_series?: number;
  n_maps?: number;
  international_series?: number;
  home_league?: string;
  model?: string;
  model_version?: string;
  comparison_component_id?: string;
  comparison_component_size?: number;
  cross_component_rankable?: boolean;
};

export type TeamRatingsMeta = {
  model?: string | null;
  as_of?: string | null;
  n_series?: number | null;
  n_maps?: number | null;
  model_id?: string | null;
  model_version?: string | null;
  config?: {
    min_sigma?: number | null;
    unbridged_league_sigma?: number | null;
    bridge_target_series?: number | null;
    team_prior_sigma?: number | null;
    team_variance_per_day?: number | null;
  };
  uncertainty?: {
    field?: string | null;
    lower_tail_probability?: number | null;
    one_sided_coverage?: number | null;
    z?: number | null;
    formula?: string | null;
    interpretation?: string | null;
    sigma_kind?: string | null;
    coverage_claim?: boolean | null;
  };
  comparison_components?: {
    count?: number | null;
    cross_component_rankable?: boolean | null;
    policy?: string | null;
  };
};

export type PlayerRatingsMeta = {
  n_maps?: number | null;
  n_input_maps?: number | null;
  n_identity_eligible_maps?: number | null;
  identity_eligible_map_rate?: number | null;
  n_players?: number | null;
  n_unique_outcome_exposure_players?: number | null;
  n_shared_outcome_history_players?: number | null;
  outcome_ordering_verified?: boolean | null;
  individual_skill_estimand?: boolean | null;
  config?: {
    sigma_min?: number | null;
  };
};

export type PlayerRating = {
  player: string;
  mu_total: number;
  mu_regional: number;
  mu_meta: number;
  sigma: number;
  n_maps: number;
  last_team: string | null;
  outcome_exposure_group_id?: string | null;
  outcome_exposure_group_size?: number | null;
  outcome_separately_identified?: boolean | null;
  outcome_identifiability_label?: string | null;
  outcome_identical_players?: string[] | null;
  n_distinct_lineups?: number | null;
  n_distinct_teams?: number | null;
};

export type PlayerPerformanceRating = {
  model_id: string;
  model_hash: string;
  player_id: string;
  player_name: string;
  role: "top" | "jng" | "mid" | "bot" | "sup";
  last_team_key: string;
  last_observed_league: string | null;
  last_observed_date: string | null;
  fit_through: string;
  effective_sample_maps: number;
  performance_mean: number;
  performance_sd: number;
  lower_bound: number;
  rank: number;
  uncertainty_method: string;
  estimand: string;
  publication_status: "validated_narrow_descriptive_view";
};

export type PlayerPerformanceMetrics = {
  rows: number;
  rmse: number;
  mae: number;
  r2: number;
  spearman: number;
  zero_baseline_rmse: number;
  relative_rmse_lift: number;
};

export type PlayerPerformanceRMSEContrast = {
  rows: number;
  calendar_day_blocks: number;
  candidate_rmse: number;
  baseline_rmse: number;
  relative_rmse_lift: number;
  ci_low: number;
  ci_high: number;
  confidence_level: number;
  bootstrap_replicates: number;
  resampling_unit: string;
};

export type PlayerPerformanceMeta = {
  artifact_schema_version: string;
  model_family: string;
  display_name: string;
  publication_status: string;
  model_id: string;
  model_hash: string;
  model_hash_scope: string;
  grain: string;
  estimand: string;
  roles: string[];
  effective_sample: {
    eligible_role_matchups: number;
    stable_identity_matchups: number;
    published_player_role_rows: number;
    test_player_rows: number;
  };
  fit_through: string;
  test_window: { start: string; end: string };
  uncertainty: {
    methods: string[];
    conservative_z: number;
    lower_bound: string;
    interpretation: string;
  };
  test_metrics: PlayerPerformanceMetrics;
  context_only_test_metrics: PlayerPerformanceMetrics;
  player_incremental_test_contrast: PlayerPerformanceRMSEContrast;
  non_estimands: string[];
  limitations: string[];
  research_anchors: string[];
  ranking: {
    scope: string;
    score: string;
    ties: string;
  };
};

export type PlayerPerformanceValidation = {
  artifact_schema_version: string;
  model_id: string;
  model_hash: string;
  model_hash_scope: string;
  evaluation_target: string;
  estimand: string;
  non_estimands: string[];
  roles: string[];
  effective_sample: {
    eligible_role_matchups: number;
    stable_identity_matchups: number;
    test_player_rows: number;
  };
  test_gate_passed: boolean;
  split_boundaries: {
    train_start: string;
    train_end: string;
    validation_start: string;
    validation_end: string;
    test_start: string;
    test_end: string;
  };
  test_metrics: PlayerPerformanceMetrics;
  test_context_baseline_metrics: PlayerPerformanceMetrics;
  player_incremental_test_rmse_lift: number;
  player_incremental_test_contrast: PlayerPerformanceRMSEContrast;
  large_prediction_ledger_exported: boolean;
};

export type PlayerPerformanceContract = {
  valid: boolean;
  reason: string | null;
  modelId: string | null;
  modelHash: string | null;
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
  membership_as_of?: string | null;
  membership_source?: string | null;
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
  last_observed_team?: string | null;
  last_observed_league?: string | null;
  last_observed_date?: string | null;
  current_league?: string | null;
  current_tier?: CompetitionTier | null;
  current_team?: string | null;
  current_date?: string | null;
  current_tournament?: string | null;
  current_affiliation_basis?: "observed_current_tournament_map" | null;
  membership_as_of?: string | null;
  membership_source?: string | null;
};

export type CompetitionTier = "tier1" | "tier2" | "tier3";

export type CurrentMembershipContext = {
  valid: boolean;
  authority: string | null;
  asOf: string | null;
  checkedAt: string | null;
  reviewDueAt: string | null;
  currentTournaments: Record<string, string>;
};

export type VerifiedPlayerAffiliation = {
  team: string;
  league: string;
  tier: CompetitionTier;
  tournament: string;
  observedAt: string;
  membershipAsOf: string;
  source: string;
};

export type VerifiedTeamAffiliation = {
  team: string;
  league: string;
  tier: CompetitionTier;
  tournament: string;
  membershipAsOf: string;
  source: string;
};

export type TeamRatingContract = {
  model: "series_dynamic_bt";
  minSigma: number;
  conservativeZ: number;
  lowerTailProbability: number;
  oneSidedCoverage: number;
  boundLabel: string;
};

export type TeamEvidenceInfo = {
  available: boolean;
  label: string;
  layman: string;
  sigma: number;
  minimumSigma: number | null;
  spreadAboveMinimum: number | null;
};

export type PlayerIdentifiabilityInfo = {
  status: "identified" | "shared" | "unknown";
  label: string;
  layman: string;
  individuallyOrderable: boolean;
};

export const TIER_FILTERS = [
  { value: "TIER1", label: "Tier 1", description: "LCK, LPL, LEC, LCS, CBLOL, and LCP" },
  { value: "TIER2", label: "Tier 2", description: "Established regional and challenger circuits" },
  { value: "TIER3", label: "Tier 3", description: "Other domestic and developmental circuits" },
] as const;

/** Legacy defaults for the sequential Dual Elo benchmark and Player Dual Elo. */
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

const PLAYER_PERFORMANCE_ROLES = new Set(["top", "jng", "mid", "bot", "sup"]);
const PLAYER_PERFORMANCE_ESTIMAND_PREFIX =
  "Descriptive role-relative 15-minute resource performance";

/**
 * Fail-closed contract for the separately published early-resource view.
 *
 * This deliberately does not accept Player Dual Elo rows as a fallback.
 */
export function playerPerformanceContract(
  rows: PlayerPerformanceRating[] | null | undefined,
  meta: PlayerPerformanceMeta | null | undefined,
  validation: PlayerPerformanceValidation | null | undefined,
): PlayerPerformanceContract {
  const fail = (reason: string): PlayerPerformanceContract => ({
    valid: false,
    reason,
    modelId: null,
    modelHash: null,
  });
  if (!rows || !meta || !validation) {
    return fail("snapshot, metadata, or validation artifact is absent");
  }
  if (rows.length === 0) return fail("snapshot has no player-role rows");
  if (
    meta.artifact_schema_version !== "1.0.0" ||
    validation.artifact_schema_version !== meta.artifact_schema_version ||
    meta.model_family !== "role_relative_15_minute_resource_performance" ||
    meta.display_name !== "15-minute resource performance" ||
    meta.publication_status !== "validated_narrow_descriptive_view"
  ) {
    return fail("model identity or artifact schema is not recognized");
  }
  if (
    !meta.model_id ||
    !/^[a-f0-9]{64}$/.test(meta.model_hash) ||
    validation.model_id !== meta.model_id ||
    validation.model_hash !== meta.model_hash ||
    validation.model_hash_scope !== meta.model_hash_scope ||
    validation.estimand !== meta.estimand ||
    validation.non_estimands.join("\u0000") !== meta.non_estimands.join("\u0000") ||
    !validation.test_gate_passed
  ) {
    return fail("model hash does not reconcile or the chronological gate failed");
  }
  if (
    !meta.estimand.startsWith(PLAYER_PERFORMANCE_ESTIMAND_PREFIX) ||
    !meta.non_estimands.includes("causal player skill") ||
    !meta.non_estimands.includes("match-win probability") ||
    !meta.non_estimands.includes("win contribution")
  ) {
    return fail("estimand or required non-estimands are missing");
  }
  const z = Number(meta.uncertainty?.conservative_z);
  if (
    !Number.isFinite(z) ||
    z <= 0 ||
    !Number.isFinite(meta.test_metrics?.rmse) ||
    (meta.test_metrics?.rows ?? 0) <= 0 ||
    validation.large_prediction_ledger_exported !== false
  ) {
    return fail("uncertainty, test metrics, or compact-export contract is invalid");
  }
  if (
    meta.effective_sample.published_player_role_rows !== rows.length ||
    validation.effective_sample.eligible_role_matchups !==
      meta.effective_sample.eligible_role_matchups ||
    validation.effective_sample.stable_identity_matchups !==
      meta.effective_sample.stable_identity_matchups ||
    validation.effective_sample.test_player_rows !==
      meta.effective_sample.test_player_rows ||
    validation.test_metrics.rows !== meta.test_metrics.rows ||
    validation.test_metrics.rmse !== meta.test_metrics.rmse ||
    validation.player_incremental_test_contrast.bootstrap_replicates !== 5_000 ||
    validation.player_incremental_test_contrast.resampling_unit !== "calendar_day" ||
    validation.player_incremental_test_contrast.bootstrap_replicates !==
      meta.player_incremental_test_contrast.bootstrap_replicates ||
    validation.player_incremental_test_contrast.ci_low !==
      meta.player_incremental_test_contrast.ci_low ||
    validation.player_incremental_test_contrast.ci_high !==
      meta.player_incremental_test_contrast.ci_high ||
    validation.split_boundaries.validation_end !== meta.fit_through
  ) {
    return fail("effective sample, test metrics, or fit-through fields disagree");
  }

  const seen = new Set<string>();
  const byRole = new Map<string, PlayerPerformanceRating[]>();
  for (const row of rows) {
    const key = `${row.player_id}\u0000${row.role}`;
    const expectedBound = row.performance_mean - z * row.performance_sd;
    const tolerance = Math.max(1e-10, Math.abs(expectedBound) * 1e-10);
    if (
      seen.has(key) ||
      !row.player_id ||
      !row.player_name ||
      !PLAYER_PERFORMANCE_ROLES.has(row.role) ||
      row.model_id !== meta.model_id ||
      row.model_hash !== meta.model_hash ||
      row.publication_status !== "validated_narrow_descriptive_view" ||
      row.estimand !== meta.estimand ||
      !Number.isFinite(row.performance_mean) ||
      !Number.isFinite(row.performance_sd) ||
      row.performance_sd < 0 ||
      !Number.isFinite(row.lower_bound) ||
      Math.abs(row.lower_bound - expectedBound) > tolerance ||
      !Number.isInteger(row.effective_sample_maps) ||
      row.effective_sample_maps <= 0 ||
      !Number.isInteger(row.rank) ||
      row.rank <= 0 ||
      !row.uncertainty_method ||
      !Number.isFinite(Date.parse(row.fit_through)) ||
      row.fit_through !== meta.fit_through
    ) {
      return fail("one or more snapshot rows violate the model contract");
    }
    seen.add(key);
    const roleRows = byRole.get(row.role) ?? [];
    roleRows.push(row);
    byRole.set(row.role, roleRows);
  }
  for (const roleRows of byRole.values()) {
    const orderedValues = [...roleRows]
      .map((row) => row.lower_bound)
      .sort((a, b) => b - a);
    for (const row of roleRows) {
      const expectedRank =
        orderedValues.findIndex((value) => value === row.lower_bound) + 1;
      if (row.rank !== expectedRank) {
        return fail("exact tied values do not share competition rank");
      }
    }
  }
  return {
    valid: true,
    reason: null,
    modelId: meta.model_id,
    modelHash: meta.model_hash,
  };
}

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

export function packDataThroughLabel(manifest: PackManifest): string {
  const raw = manifest.data_as_of || "";
  const date = raw.slice(0, 10);
  if (/^\d{4}-\d{2}-\d{2}$/.test(date)) {
    const [year, month, day] = date.split("-");
    return `${day}/${month}/${year}`;
  }
  return "not declared";
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

/** Legacy soft-rating helper. Hierarchical public surfaces must use teamBoundRating. */
export function adjustedRating(
  rating: Pick<TeamRating, "mu_total" | "sigma" | "rating_p05">,
  floor = TEAM_SIGMA_MIN,
): number {
  return Number.isFinite(rating.rating_p05)
    ? Number(rating.rating_p05)
    : softMu(rating.mu_total, rating.sigma, floor);
}

// Abramowitz-Stegun 7.1.26; sufficient for public quantile labels and tests.
function normalCdf(value: number): number {
  const x = Math.abs(value) / Math.sqrt(2);
  const t = 1 / (1 + 0.3275911 * x);
  const erf =
    1 -
    (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) *
      t +
      0.254829592) *
      t *
      Math.exp(-x * x));
  return 0.5 * (1 + (value < 0 ? -erf : erf));
}

/**
 * Validate the public dynamic series Bradley-Terry uncertainty contract.
 *
 * A missing/legacy metadata file is not permission to reinterpret rating_p05
 * using the sequential Dual Elo floor.
 */
export function teamRatingContract(
  meta: TeamRatingsMeta | null | undefined,
): TeamRatingContract | null {
  const minSigma = 0;
  const conservativeZ = Number(meta?.uncertainty?.z);
  if (
    meta?.model !== "series_dynamic_bt" ||
    !meta?.model_version ||
    !Number.isFinite(conservativeZ) ||
    conservativeZ <= 0 ||
    meta?.uncertainty?.field !== "rating_p05" ||
    meta?.uncertainty?.formula !== "rating_p05 = mu_total - z * sigma" ||
    meta?.uncertainty?.sigma_kind !==
      "diagonal_filter_approximation_sd" ||
    meta?.uncertainty?.coverage_claim !== false ||
    meta?.comparison_components?.cross_component_rankable !== false
  ) {
    return null;
  }
  const lowerTailProbability = normalCdf(-conservativeZ);
  const oneSidedCoverage = 1 - lowerTailProbability;
  return {
    model: "series_dynamic_bt",
    minSigma,
    conservativeZ,
    lowerTailProbability,
    oneSidedCoverage,
    boundLabel: "Uncertainty-adjusted rating",
  };
}

/** Return the published bound only when the row and metadata agree algebraically. */
export function teamBoundRating(
  rating: Pick<TeamRating, "model" | "mu_total" | "sigma" | "rating_p05">,
  contract: TeamRatingContract | null,
): number | null {
  if (
    !contract ||
    rating.model !== contract.model ||
    !Number.isFinite(rating.mu_total) ||
    !Number.isFinite(rating.sigma) ||
    rating.sigma < contract.minSigma ||
    !Number.isFinite(rating.rating_p05)
  ) {
    return null;
  }
  const expected = rating.mu_total - contract.conservativeZ * rating.sigma;
  const published = Number(rating.rating_p05);
  const tolerance = Math.max(1e-7, Math.abs(expected) * 1e-10);
  return Math.abs(published - expected) <= tolerance ? published : null;
}

export function teamEvidenceInfo(
  sigma: number,
  contract: TeamRatingContract | null,
  games?: number | null,
): TeamEvidenceInfo {
  if (!contract || !Number.isFinite(sigma) || sigma < contract.minSigma) {
    return {
      available: false,
      label: "Uncertainty unavailable",
      layman:
        "The pack does not provide a valid dynamic series Bradley–Terry uncertainty contract for this row.",
      sigma,
      minimumSigma: contract?.minSigma ?? null,
      spreadAboveMinimum: null,
    };
  }
  const spread = sigma;
  const label = `Approx. σ ${sigma.toFixed(1)}`;
  const gamesBit =
    games != null && games > 0 ? ` The record table contains ${games} maps.` : "";
  return {
    available: true,
    label,
    layman:
      `${label}. This is the dynamic filter's diagonal Gaussian approximation, ` +
      `including uncertainty growth during inactivity. It is not an empirically ` +
      `calibrated coverage interval.${gamesBit}`,
    sigma,
    minimumSigma: contract.minSigma,
    spreadAboveMinimum: spread,
  };
}

export function playerSigmaFloor(
  meta: PlayerRatingsMeta | null | undefined,
): number | null {
  const floor = Number(meta?.config?.sigma_min);
  return Number.isFinite(floor) && floor > 0 ? floor : null;
}

export function playerAdjustedRating(
  rating: Pick<PlayerRating, "mu_total" | "sigma">,
  sigmaFloor: number | null,
): number | null {
  if (
    sigmaFloor == null ||
    !Number.isFinite(rating.mu_total) ||
    !Number.isFinite(rating.sigma) ||
    rating.sigma < sigmaFloor
  ) {
    return null;
  }
  return softMu(rating.mu_total, rating.sigma, sigmaFloor);
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

export function playerIdentifiabilityInfo(
  rating: Pick<
    PlayerRating,
    | "outcome_separately_identified"
    | "outcome_exposure_group_id"
    | "outcome_exposure_group_size"
    | "outcome_identical_players"
  >,
): PlayerIdentifiabilityInfo {
  const size = Number(rating.outcome_exposure_group_size);
  if (
    rating.outcome_separately_identified === true &&
    Number.isInteger(size) &&
    size === 1 &&
    Boolean(rating.outcome_exposure_group_id)
  ) {
    return {
      status: "identified",
      label: "Distinct outcome exposure",
      layman:
        "This player has a distinct signed map-exposure history in the published team-result data. That alone does not identify individual skill or support a public rank.",
      individuallyOrderable: true,
    };
  }
  if (
    rating.outcome_separately_identified === false &&
    Number.isInteger(size) &&
    size >= 2 &&
    Boolean(rating.outcome_exposure_group_id)
  ) {
    const peers = (rating.outcome_identical_players ?? []).filter(Boolean);
    return {
      status: "shared",
      label: `Shared outcome cohort (n=${size})`,
      layman:
        `Team-result data give this player the same signed map exposure as ${
          peers.join(", ") || "the other members of this cohort"
        }. The model cannot defend an individual ordering inside the cohort.`,
      individuallyOrderable: false,
    };
  }
  return {
    status: "unknown",
    label: "Outcome identifiability unverified",
    layman:
      "This pack does not contain the signed map-exposure metadata needed to tell whether team results identify this player separately from teammates. Individual rank claims are withheld.",
    individuallyOrderable: false,
  };
}

export function playerOutcomeOrderingVerified(
  meta: PlayerRatingsMeta | null | undefined,
  ratings: PlayerRating[],
): boolean {
  return (
    meta?.outcome_ordering_verified === true &&
    meta?.individual_skill_estimand === true &&
    ratings.length > 0 &&
    ratings.every(
      (rating) => playerIdentifiabilityInfo(rating).individuallyOrderable,
    )
  );
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
  "cloud9 kia": "Cloud9",
  tl: "Team Liquid",
  tlaw: "Team Liquid",
  "team liquid alienware": "Team Liquid",
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
  drx: "KIWOOM DRX",
  krx: "KIWOOM DRX",
  ktz: "KT Rolster",
  kt: "KT Rolster",
  nsf: "Nongshim RedForce",
  ns: "Nongshim RedForce",
  bro: "HANJIN BRION",
  fox: "BNK FEARX",
  fly: "FlyQuest",
  sr: "Shopify Rebellion",
  dig: "Dignitas",
  lrx: "LOUD",
  pai: "paiN Gaming",
  fur: "FURIA",
  red: "RED Kalunga",
  navi: "Natus Vincere",
  shft: "Shifters",
  dsg: "Disguised",
  sen: "Sentinels",
  dcg: "Relove Deep Cross Gaming",
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

export function canonicalTeamDisplay(value: string): string {
  const raw = value.trim();
  if (!raw) return "";
  return TEAM_ALIASES[normKey(raw)] ?? raw;
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
    membershipRegistryValid?: boolean;
    membershipContext?: CurrentMembershipContext;
  },
): boolean {
  if (!selected.length) return true;
  if (!rec) return false;
  const tierSet = new Set(selected.filter((scope) => scope.startsWith("TIER")));
  const regionalSet = new Set(selected.filter((scope) => REGION_LEAGUES.includes(scope as (typeof REGION_LEAGUES)[number])));
  const crossRegionSet = new Set(selected.filter((scope) => INTERREGIONAL_LEAGUES.includes(scope as (typeof INTERREGIONAL_LEAGUES)[number])));
  const internationalSet = new Set(selected.filter((scope) => scope === "INTL" || isIntlLeague(scope)));
  const verifiedTeam = options?.membershipContext
    ? verifiedTeamAffiliation(rec as TeamRecord, options.membershipContext)
    : null;
  const hasVerifiedCurrentRecord =
    verifiedTeam != null || options?.membershipRegistryValid === true;
  const currentLeague = verifiedTeam?.league ??
    (hasVerifiedCurrentRecord ? rec.current_league : null);
  const currentTier = (
    verifiedTeam?.tier ??
    (hasVerifiedCurrentRecord ? rec.current_tier : null)
  )?.toUpperCase();

  // Filters in one group are alternatives; filters across groups narrow the
  // same current affiliation/evidence record. This prevents selecting Tier 1
  // from being silently overridden by a Tier 2 regional chip.
  const tierMatches = !tierSet.size || (!!currentTier && tierSet.has(currentTier));
  const regionalMatches = !regionalSet.size || (!!currentLeague && regionalSet.has(currentLeague));
  const currentTournaments =
    options?.currentTournaments ??
    options?.membershipContext?.currentTournaments;
  const expectedTournament = currentLeague
    ? currentTournaments?.[currentLeague]
    : undefined;
  const currentTournamentMatches =
    hasVerifiedCurrentRecord &&
    !!expectedTournament &&
    rec.current_tournament === expectedTournament;
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

export function membershipRegistryIsCurrent(
  manifest: PackManifest,
  asOf: string | Date = new Date(),
): boolean {
  const registry = manifest.membership_registry;
  if (
    registry?.authority !== "Riot Games LoL Esports" ||
    !registry.snapshot_id ||
    !registry.checked_at ||
    !registry.review_due_at ||
    !manifest.current_tournament_as_of ||
    !manifest.current_tournaments ||
    !Object.keys(manifest.current_tournaments).length
  ) {
    return false;
  }
  const checked = Date.parse(registry.checked_at);
  const reviewDue = Date.parse(registry.review_due_at);
  const observed = Date.parse(manifest.current_tournament_as_of);
  const query = new Date(asOf).getTime();
  return (
    Number.isFinite(checked) &&
    Number.isFinite(reviewDue) &&
    Number.isFinite(observed) &&
    checked >= observed &&
    checked <= reviewDue &&
    query >= observed &&
    query <= reviewDue
  );
}

export function currentMembershipContext(
  manifest: PackManifest,
  asOf: string | Date = new Date(),
): CurrentMembershipContext {
  return {
    valid: membershipRegistryIsCurrent(manifest, asOf),
    authority: manifest.membership_registry?.authority ?? null,
    asOf: manifest.current_tournament_as_of ?? null,
    checkedAt: manifest.membership_registry?.checked_at ?? null,
    reviewDueAt: manifest.membership_registry?.review_due_at ?? null,
    currentTournaments: manifest.current_tournaments ?? {},
  };
}

/**
 * Treat a player affiliation as current only when the authoritative registry
 * and the row-level participation proof agree. Legacy current_team fields are
 * historical hints, not present-day membership.
 */
export function verifiedPlayerAffiliation(
  rec: PlayerRecord | undefined,
  context: CurrentMembershipContext,
): VerifiedPlayerAffiliation | null {
  if (
    !context.valid ||
    !rec?.current_team ||
    !rec.current_league ||
    !rec.current_tier ||
    !rec.current_tournament ||
    !rec.current_date ||
    rec.current_affiliation_basis !== "observed_current_tournament_map" ||
    !rec.membership_as_of ||
    !rec.membership_source ||
    !context.authority ||
    !context.asOf ||
    rec.membership_source !== context.authority ||
    rec.membership_as_of !== context.asOf ||
    context.currentTournaments[rec.current_league] !== rec.current_tournament
  ) {
    return null;
  }
  const observedAt = dateMs(rec.current_date);
  const membershipAt = dateMs(rec.membership_as_of);
  const reviewDueAt = dateMs(context.reviewDueAt);
  if (
    observedAt == null ||
    membershipAt == null ||
    reviewDueAt == null ||
    observedAt > reviewDueAt ||
    membershipAt > reviewDueAt
  ) {
    return null;
  }
  return {
    team: rec.current_team,
    league: rec.current_league,
    tier: rec.current_tier,
    tournament: rec.current_tournament,
    observedAt: rec.current_date,
    membershipAsOf: rec.membership_as_of,
    source: rec.membership_source,
  };
}

export function verifiedTeamAffiliation(
  rec: TeamRecord | undefined,
  context: CurrentMembershipContext,
): VerifiedTeamAffiliation | null {
  if (
    !context.valid ||
    !rec?.current_team ||
    !rec.current_league ||
    !rec.current_tier ||
    !rec.current_tournament ||
    !rec.membership_as_of ||
    !rec.membership_source ||
    !context.authority ||
    !context.asOf ||
    rec.membership_source !== context.authority ||
    rec.membership_as_of !== context.asOf ||
    context.currentTournaments[rec.current_league] !== rec.current_tournament
  ) {
    return null;
  }
  return {
    team: rec.current_team,
    league: rec.current_league,
    tier: rec.current_tier,
    tournament: rec.current_tournament,
    membershipAsOf: rec.membership_as_of,
    source: rec.membership_source,
  };
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
  const crossRegionSelected = selected.filter((scope) =>
    INTERREGIONAL_LEAGUES.includes(scope as (typeof INTERREGIONAL_LEAGUES)[number]),
  );
  const internationalSelected = selected.filter((scope) => scope === "INTL" || isIntlLeague(scope));
  const domesticSelected = tierSelected.length > 0 || regionalSelected.length > 0;
  if (
    crossRegionSelected.length > 0 ||
    (domesticSelected && internationalSelected.length > 0)
  ) {
    // The artifact does not expose a joint map-level denominator for these
    // mixed eligibility scopes. Returning one component's WR would mislabel it.
    return null;
  }
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
