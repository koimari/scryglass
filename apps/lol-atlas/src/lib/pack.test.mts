import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  canonicalTeamDisplay,
  currentMembershipContext,
  membershipRegistryIsCurrent,
  playerIdentifiabilityInfo,
  playerOutcomeOrderingVerified,
  playerPerformanceContract,
  recordMatchesLeagues,
  scopedTeamWr,
  teamBoundRating,
  teamRatingContract,
  verifiedPlayerAffiliation,
  verifiedTeamAffiliation,
  type PackManifest,
  type PlayerRating,
  type PlayerPerformanceMeta,
  type PlayerPerformanceRating,
  type PlayerPerformanceValidation,
  type PlayerRecord,
  type TeamRating,
  type TeamRecord,
} from "./pack.ts";

const currentTournaments = { LPL: "LPL - Split 3 2026" };

test("team aliases preserve canonical display casing", () => {
  assert.equal(canonicalTeamDisplay("koi"), "Movistar KOI");
  assert.equal(canonicalTeamDisplay("drx"), "KIWOOM DRX");
  assert.equal(canonicalTeamDisplay("bro"), "HANJIN BRION");
  assert.equal(canonicalTeamDisplay("fox"), "BNK FEARX");
  assert.equal(canonicalTeamDisplay("Unknown Club"), "Unknown Club");
});

function teamRecord(currentTournament: string | null): TeamRecord {
  return {
    leagues: ["LPL"],
    primary: "LPL",
    intl: false,
    current_league: "LPL",
    current_tier: "tier1",
    current_team: "Bilibili Gaming",
    current_date: "2026-07-25",
    current_tournament: currentTournament,
    membership_as_of: "2026-07-26T10:00:00Z",
    membership_source: "Riot Games LoL Esports",
    wins: 8,
    games: 10,
    wr: 0.8,
    by_league: { LPL: { wins: 8, games: 10, wr: 0.8 } },
    by_tier: { tier1: { wins: 8, games: 10, wr: 0.8 } },
    by_tournament: {
      "LPL|LPL - Split 3 2026": { wins: 2, games: 3, wr: 0.6667 },
    },
  };
}

test("regional membership requires the pack-declared current tournament", () => {
  assert.equal(
    recordMatchesLeagues(teamRecord("LPL - Split 3 2026"), ["LPL"], {
      dataAsOf: "2026-07-26",
      currentTournaments,
      membershipRegistryValid: true,
    }),
    true,
  );
  assert.equal(
    recordMatchesLeagues(teamRecord("LPL - Split 2 2026"), ["LPL"], {
      dataAsOf: "2026-07-26",
      currentTournaments,
      membershipRegistryValid: true,
    }),
    false,
  );
  assert.equal(
    recordMatchesLeagues(teamRecord(null), ["LPL"], {
      dataAsOf: "2026-07-26",
      currentTournaments,
      membershipRegistryValid: true,
    }),
    false,
  );
});

test("scoped win rate uses current tournament observations when available", () => {
  const record = teamRecord("LPL - Split 3 2026");
  assert.equal(scopedTeamWr(record, ["LPL"], { currentTournaments }), 2 / 3);
  assert.equal(scopedTeamWr(record, ["TIER1"], { currentTournaments }), 2 / 3);
});

test("mixed domestic and international scopes withhold a misleading WR denominator", () => {
  const record = {
    ...teamRecord("LPL - Split 3 2026"),
    intl: true,
    leagues: ["LPL", "MSI"],
    by_league: {
      LPL: { wins: 8, games: 10, wr: 0.8 },
      MSI: { wins: 3, games: 5, wr: 0.6 },
    },
  };
  assert.equal(
    scopedTeamWr(record, ["LPL", "MSI"], { currentTournaments }),
    null,
  );
  assert.equal(
    scopedTeamWr(record, ["TIER1", "INTL"], { currentTournaments }),
    null,
  );
});

test("domestic current membership fails closed without a valid registry", () => {
  const record = teamRecord(null);
  assert.equal(
    recordMatchesLeagues(record, ["LPL"], {
      dataAsOf: "2026-07-26",
      currentTournaments,
      membershipRegistryValid: true,
    }),
    false,
  );
  assert.equal(
    recordMatchesLeagues(record, ["LPL"], {
      dataAsOf: "2026-07-26",
      currentTournaments: {},
      membershipRegistryValid: false,
    }),
    false,
  );
});

test("membership registry expires at its declared review deadline", () => {
  const manifest = {
    pack_id: "test",
    schema_version: "1",
    created_utc: "2026-07-26T12:00:00Z",
    current_tournament_as_of: "2026-07-26T10:00:00Z",
    current_tournaments: currentTournaments,
    membership_registry: {
      snapshot_id: "registry-test",
      authority: "Riot Games LoL Esports",
      checked_at: "2026-07-26T12:00:00Z",
      review_due_at: "2026-08-02T23:59:59Z",
    },
    filters: { years: [2026], leagues: "all" },
    attribution: "test",
    excluded: [],
    base_url: null,
    total_bytes: 0,
    total_files: 0,
    files: [],
  } satisfies PackManifest;
  assert.equal(
    membershipRegistryIsCurrent(manifest, "2026-07-27T00:00:00Z"),
    true,
  );
  assert.equal(
    membershipRegistryIsCurrent(manifest, "2026-08-03T00:00:00Z"),
    false,
  );
});

function currentManifest(): PackManifest {
  return {
    pack_id: "test",
    schema_version: "1",
    created_utc: "2026-07-26T12:00:00Z",
    current_tournament_as_of: "2026-07-26T10:00:00Z",
    current_tournaments: currentTournaments,
    membership_registry: {
      snapshot_id: "registry-test",
      authority: "Riot Games LoL Esports",
      checked_at: "2026-07-26T12:00:00Z",
      review_due_at: "2026-08-02T23:59:59Z",
    },
    filters: { years: [2026], leagues: "all" },
    attribution: "test",
    excluded: [],
    base_url: null,
    total_bytes: 0,
    total_files: 0,
    files: [],
  };
}

test("legacy membership fields cannot become current affiliation", () => {
  const context = currentMembershipContext(
    currentManifest(),
    "2026-07-27T00:00:00Z",
  );
  const legacyPlayer: PlayerRecord = {
    wins: 2,
    games: 3,
    wr: 2 / 3,
    current_team: "Bilibili Gaming",
    current_league: "LPL",
    current_tier: "tier1",
    current_tournament: "LPL - Split 3 2026",
    current_date: "2026-07-25",
  };
  assert.equal(verifiedPlayerAffiliation(legacyPlayer, context), null);

  const verifiedPlayer: PlayerRecord = {
    ...legacyPlayer,
    current_affiliation_basis: "observed_current_tournament_map",
    membership_as_of: "2026-07-26T10:00:00Z",
    membership_source: "Riot Games LoL Esports",
  };
  assert.deepEqual(verifiedPlayerAffiliation(verifiedPlayer, context), {
    team: "Bilibili Gaming",
    league: "LPL",
    tier: "tier1",
    tournament: "LPL - Split 3 2026",
    observedAt: "2026-07-25",
    membershipAsOf: "2026-07-26T10:00:00Z",
    source: "Riot Games LoL Esports",
  });

  const legacyTeam = teamRecord("LPL - Split 3 2026");
  delete legacyTeam.membership_source;
  assert.equal(verifiedTeamAffiliation(legacyTeam, context), null);
  assert.equal(
    recordMatchesLeagues(legacyTeam, ["LPL"], {
      currentTournaments,
      membershipContext: context,
    }),
    false,
  );
  assert.equal(
    verifiedTeamAffiliation(teamRecord("LPL - Split 3 2026"), context)?.league,
    "LPL",
  );
});

test("dynamic series team bound requires matching model, quantile metadata, and row algebra", () => {
  const contract = teamRatingContract({
    model: "series_dynamic_bt",
    model_version: "series_dynamic_bt:test:test",
    uncertainty: {
      field: "rating_p05",
      z: 1.6448536269514722,
      formula: "rating_p05 = mu_total - z * sigma",
      sigma_kind: "diagonal_filter_approximation_sd",
      coverage_claim: false,
    },
    comparison_components: { cross_component_rankable: false },
  });
  assert.ok(contract);
  assert.equal(contract.boundLabel, "Uncertainty-adjusted rating");
  assert.ok(Math.abs(contract.oneSidedCoverage - 0.95) < 1e-6);

  const rating: TeamRating = {
    team: "Gen.G",
    mu_total: 1600,
    sigma: 30,
    rating_p05: 1600 - 1.6448536269514722 * 30,
    model: "series_dynamic_bt",
  };
  assert.equal(teamBoundRating(rating, contract), rating.rating_p05);
  assert.equal(teamBoundRating({ ...rating, model: "dual_elo" }, contract), null);
  assert.equal(teamBoundRating({ ...rating, rating_p05: 1555 }, contract), null);
  assert.equal(teamRatingContract({ model: "series_dynamic_bt", config: {} }), null);
});

test("missing player identifiability metadata is unknown, never individual evidence", () => {
  const legacy: PlayerRating = {
    player: "Player",
    mu_total: 1500,
    mu_regional: 1500,
    mu_meta: 0,
    sigma: 30,
    n_maps: 100,
    last_team: "Team",
  };
  assert.equal(playerIdentifiabilityInfo(legacy).status, "unknown");
  assert.equal(playerIdentifiabilityInfo(legacy).individuallyOrderable, false);

  const shared = {
    ...legacy,
    outcome_separately_identified: false,
    outcome_exposure_group_id: "group-1",
    outcome_exposure_group_size: 2,
    outcome_identical_players: ["Teammate"],
  };
  assert.equal(playerIdentifiabilityInfo(shared).status, "shared");
  assert.equal(playerIdentifiabilityInfo(shared).individuallyOrderable, false);
});

test("player outcome ordering requires an explicit individual-skill contract", () => {
  const distinct: PlayerRating = {
    player: "Distinct",
    mu_total: 1500,
    mu_regional: 1500,
    mu_meta: 0,
    sigma: 30,
    n_maps: 100,
    last_team: "Team",
    outcome_separately_identified: true,
    outcome_exposure_group_id: "unique",
    outcome_exposure_group_size: 1,
  };
  assert.equal(
    playerOutcomeOrderingVerified(
      {
        outcome_ordering_verified: false,
        individual_skill_estimand: false,
      },
      [distinct],
    ),
    false,
  );
  assert.equal(
    playerOutcomeOrderingVerified(
      {
        outcome_ordering_verified: true,
        individual_skill_estimand: true,
      },
      [distinct],
    ),
    true,
  );
  assert.equal(
    playerOutcomeOrderingVerified(
      {
        outcome_ordering_verified: true,
        individual_skill_estimand: true,
      },
      [
        {
          ...distinct,
          outcome_separately_identified: false,
          outcome_exposure_group_id: "shared",
          outcome_exposure_group_size: 2,
        },
      ],
    ),
    false,
  );
});

function performanceArtifacts(): {
  rows: PlayerPerformanceRating[];
  meta: PlayerPerformanceMeta;
  validation: PlayerPerformanceValidation;
} {
  const modelHash = "a".repeat(64);
  const estimand =
    "Descriptive role-relative 15-minute resource performance: test fixture.";
  const incrementalContrast = {
    rows: 40,
    calendar_day_blocks: 10,
    candidate_rmse: 1,
    baseline_rmse: 1.03,
    relative_rmse_lift: 0.03,
    ci_low: 0.01,
    ci_high: 0.05,
    confidence_level: 0.95,
    bootstrap_replicates: 5_000,
    resampling_unit: "calendar_day",
  };
  const meta: PlayerPerformanceMeta = {
    artifact_schema_version: "1.0.0",
    model_family: "role_relative_15_minute_resource_performance",
    display_name: "15-minute resource performance",
    publication_status: "validated_narrow_descriptive_view",
    model_id: "player-performance-v1-aaaaaaaaaaaa",
    model_hash: modelHash,
    model_hash_scope: "public snapshot, config, splits, and metrics",
    grain: "one stable player ID by canonical role",
    estimand,
    roles: ["top", "jng", "mid", "bot", "sup"],
    effective_sample: {
      eligible_role_matchups: 100,
      stable_identity_matchups: 98,
      published_player_role_rows: 3,
      test_player_rows: 40,
    },
    fit_through: "2026-04-21T00:00:00Z",
    test_window: {
      start: "2026-04-22T00:00:00Z",
      end: "2026-07-01T00:00:00Z",
    },
    uncertainty: {
      methods: ["exact_penalized_hessian_diagonal"],
      conservative_z: 1.645,
      lower_bound: "mean minus 1.645 standard deviations",
      interpretation: "local coefficient uncertainty",
    },
    test_metrics: {
      rows: 40,
      rmse: 1,
      mae: 0.8,
      r2: 0.1,
      spearman: 0.2,
      zero_baseline_rmse: 1.08,
      relative_rmse_lift: 0.074,
    },
    context_only_test_metrics: {
      rows: 40,
      rmse: 1.03,
      mae: 0.82,
      r2: 0.08,
      spearman: 0.18,
      zero_baseline_rmse: 1.08,
      relative_rmse_lift: 0.046,
    },
    player_incremental_test_contrast: incrementalContrast,
    non_estimands: [
      "causal player skill",
      "match-win probability",
      "win contribution",
    ],
    limitations: ["Narrow early-resource target."],
    research_anchors: ["SIDO: arXiv:2403.04873"],
    ranking: {
      scope: "within canonical role",
      score: "lower_bound",
      ties: "minimum competition rank on exact unrounded values",
    },
  };
  const base = {
    model_id: meta.model_id,
    model_hash: modelHash,
    role: "top" as const,
    last_team_key: "team",
    last_observed_league: "LCK",
    last_observed_date: "2026-04-20T00:00:00Z",
    fit_through: meta.fit_through,
    performance_mean: 0.4,
    performance_sd: 0.1,
    lower_bound: 0.4 - 1.645 * 0.1,
    uncertainty_method: "exact_penalized_hessian_diagonal",
    estimand,
    publication_status: "validated_narrow_descriptive_view" as const,
  };
  const rows: PlayerPerformanceRating[] = [
    {
      ...base,
      player_id: "p1",
      player_name: "One",
      effective_sample_maps: 50,
      rank: 1,
    },
    {
      ...base,
      player_id: "p2",
      player_name: "Two",
      effective_sample_maps: 48,
      rank: 1,
    },
    {
      ...base,
      player_id: "p3",
      player_name: "Three",
      performance_mean: 0.2,
      lower_bound: 0.2 - 1.645 * 0.1,
      effective_sample_maps: 42,
      rank: 3,
    },
  ];
  const validation: PlayerPerformanceValidation = {
    artifact_schema_version: meta.artifact_schema_version,
    model_id: meta.model_id,
    model_hash: modelHash,
    model_hash_scope: meta.model_hash_scope,
    evaluation_target: "held-out role-relative 15-minute resource performance",
    estimand,
    non_estimands: meta.non_estimands,
    roles: meta.roles,
    effective_sample: {
      eligible_role_matchups: 100,
      stable_identity_matchups: 98,
      test_player_rows: 40,
    },
    test_gate_passed: true,
    split_boundaries: {
      train_start: "2025-01-01T00:00:00Z",
      train_end: "2026-02-01T00:00:00Z",
      validation_start: "2026-02-02T00:00:00Z",
      validation_end: meta.fit_through,
      test_start: "2026-04-22T00:00:00Z",
      test_end: "2026-07-01T00:00:00Z",
    },
    test_metrics: meta.test_metrics,
    test_context_baseline_metrics: meta.context_only_test_metrics,
    player_incremental_test_rmse_lift: 0.03,
    player_incremental_test_contrast: incrementalContrast,
    large_prediction_ledger_exported: false,
  };
  return { rows, meta, validation };
}

test("player performance contract preserves exact tied ranks", () => {
  const { rows, meta, validation } = performanceArtifacts();
  assert.equal(playerPerformanceContract(rows, meta, validation).valid, true);
});

test("player performance fails closed on absence, failed gate, or false rank", () => {
  const { rows, meta, validation } = performanceArtifacts();
  assert.equal(playerPerformanceContract(null, meta, validation).valid, false);
  assert.equal(
    playerPerformanceContract(rows, meta, {
      ...validation,
      test_gate_passed: false,
    }).valid,
    false,
  );
  assert.equal(
    playerPerformanceContract(
      rows.map((row, index) => (index === 1 ? { ...row, rank: 2 } : row)),
      meta,
      validation,
    ).valid,
    false,
  );
  assert.equal(
    playerPerformanceContract(
      rows,
      {
        ...meta,
        player_incremental_test_contrast: {
          ...meta.player_incremental_test_contrast,
          bootstrap_replicates: 400,
        },
      },
      {
        ...validation,
        player_incremental_test_contrast: {
          ...validation.player_incremental_test_contrast,
          bootstrap_replicates: 400,
        },
      },
    ).valid,
    false,
  );
});

test("ratings surfaces keep model and fallback semantics explicit", () => {
  const sources = [
    "../app/elo/page.tsx",
    "../components/EloLadders.tsx",
    "../components/TeamEloDetail.tsx",
    "../components/PlayerEloDetail.tsx",
  ].map((relative) =>
    readFileSync(new URL(relative, import.meta.url), "utf8"),
  );
  const combined = sources.join("\n");
  assert.match(combined, /Hierarchical Bradley–Terry/i);
  assert.match(combined, /Player Dual Elo/);
  assert.doesNotMatch(combined, /Dual Elo ladders/);
  assert.doesNotMatch(combined, /one-sided 90% conservative/);
  assert.doesNotMatch(combined, /league-aware/i);
  assert.match(
    sources[1],
    /const displayName = affiliation\?\.team \?\? t\.team/,
  );
  assert.match(
    sources[1],
    /teamMatchesQuery\(affiliation\?\.team \?\? "", q\)/,
  );
});
