import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import {
  g1RosterEvidenceAvailable,
  loadTerminalModelArtifact,
  renderTerminalContract,
  scoreTerminalDraft,
  terminalInputId,
  terminalDraftFromSides,
  validateTerminalL2AuthorityRecord,
} from "./draftTerminalScore";
import type {
  TerminalAction,
  TerminalAssignment,
  TerminalContract,
  TerminalModelArtifact,
  TerminalSide,
} from "./draftTerminalScore";
import { canonicalDraftServingAuthorityAvailable, scoreCanonicalTerminalDraft } from "./draftTerminalServer";

const repoRoot = path.resolve(process.cwd(), "../..");
const modelPath = path.join(repoRoot, "data/lol/v2/models/draft-terminal/terminal-model-development-v1.json");
const fixturePath = path.join(repoRoot, "data/lol/v2/models/draft-terminal/terminal-replay-fixture.json");
const neutralModelPath = path.join(repoRoot, "data/lol/v2/models/draft-terminal/terminal-model-neutral-development-v1.json");
const neutralFixturePath = path.join(repoRoot, "data/lol/v2/models/draft-terminal/terminal-neutral-development-replay-fixture.json");

type ReplayFixture = {
  draft: {
    side_a: Record<string, string>;
    side_b: Record<string, string>;
    event_start: string;
    source_available_at: string;
    source_record_id: string;
    source_payload_sha256: string;
    source_rights_status: "reviewed" | "unknown";
    mode: "neutral" | "contextual";
    actions?: TerminalAction[];
    final_assignments?: TerminalAssignment[];
  };
  model_artifact_sha256: string;
  expected_development: Record<string, unknown>;
};

function loadFixtureAt(filePath: string): ReplayFixture {
  return JSON.parse(readFileSync(filePath, "utf8")) as ReplayFixture;
}

function loadFixture() {
  return loadFixtureAt(fixturePath);
}

function g1Evidence(sourceRecordId = "source:g1-roster-fixture") {
  const payload = {
    schema_version: "scryglass:g1-roster-payload:v1",
    source_record_id: sourceRecordId,
    event_start: "2026-07-01T12:00:00Z",
    available_at: "2026-07-01T08:00:00Z",
    retrieved_at: "2026-07-01T09:00:00Z",
    rights_status: "reviewed",
    rosters: {
      A: {
        roster_id: "roster:A",
        starters: [
          { role: "top", player_id: "player:a-top" },
          { role: "jungle", player_id: "player:a-jungle" },
          { role: "mid", player_id: "player:a-mid" },
          { role: "bot", player_id: "player:a-bot" },
          { role: "support", player_id: "player:a-support" },
        ],
      },
      B: {
        roster_id: "roster:B",
        starters: [
          { role: "top", player_id: "player:b-top" },
          { role: "jungle", player_id: "player:b-jungle" },
          { role: "mid", player_id: "player:b-mid" },
          { role: "bot", player_id: "player:b-bot" },
          { role: "support", player_id: "player:b-support" },
        ],
      },
    },
  } as const;
  const raw = Buffer.from(JSON.stringify(payload), "utf8");
  return {
    event_start: payload.event_start,
    source_record_id: sourceRecordId,
    source_available_at: payload.available_at,
    source_retrieved_at: payload.retrieved_at,
    source_rights_status: "reviewed" as const,
    source_payload_sha256: createHash("sha256").update(raw).digest("hex"),
    source_payload_base64: raw.toString("base64"),
    roster_a_id: "roster:A",
    roster_b_id: "roster:B",
    starters_a: [...payload.rosters.A.starters],
    starters_b: [...payload.rosters.B.starters],
  };
}

function completeHistory(fixture: ReplayFixture) {
  const actions: TerminalAction[] = [];
  const finalAssignments: TerminalAssignment[] = [];
  let slot = 1;
  for (const role of ["top", "jungle", "mid", "bot", "support"] as const) {
    for (const [canonicalSide, champions] of [["A", fixture.draft.side_a], ["B", fixture.draft.side_b]] as const) {
      const actionId = `scryglass:contract-fixture:action:${slot}`;
      actions.push({ action_id: actionId, slot, kind: "pick", canonical_side: canonicalSide, champion_id: champions[role], role_set: [role] });
      finalAssignments.push({ action_id: actionId, canonical_side: canonicalSide, champion_id: champions[role], role });
      slot += 1;
    }
  }
  return { actions, finalAssignments };
}

function authorizedReceipt(model: TerminalModelArtifact) {
  return {
    schema_version: "draft-terminal-promotion-receipt-v1" as const,
    status: "approved",
    model_version: model.modelVersion,
    artifact_sha256: model.artifactSha256,
    l2_contract_sha256: "5".repeat(64),
    development_evaluation_sha256: "f".repeat(64),
    candidate_registry_sha256: "1".repeat(64),
    calibration_transform_sha256: "2".repeat(64),
    reliability_artifact_sha256: "3".repeat(64),
    replay_parity_evidence_sha256: "4".repeat(64),
    independent_authority_record_sha256: "9".repeat(64),
    independent_l2_authority: true as const,
    final_temporal_holdout_sealed: true as const,
    public_probability_authorized: true,
    replay_parity_verified: true as const,
    reliability_gate_passed: true as const,
    contextual_g1_authority: "not_applicable" as const,
    authority_record_id: "test-only:contract-renderer",
    issued_at: "2026-07-01T13:00:00Z",
  };
}

function authorityRecordFor(receipt: ReturnType<typeof authorizedReceipt>) {
  return {
    schema_version: "scryglass:draft-terminal-l2-authority-record:v1",
    status: "approved",
    authority_record_id: receipt.authority_record_id,
    issued_at: receipt.issued_at,
    independent_reviewer_id: "test-only:independent-reviewer",
    model_artifact_sha256: receipt.artifact_sha256,
    candidate_registry_sha256: receipt.candidate_registry_sha256,
    development_evaluation_sha256: receipt.development_evaluation_sha256,
    l2_contract_sha256: receipt.l2_contract_sha256,
    calibration_transform_sha256: receipt.calibration_transform_sha256,
    reliability_artifact_sha256: receipt.reliability_artifact_sha256,
    replay_parity_evidence_sha256: receipt.replay_parity_evidence_sha256,
    source_snapshot_sha256: "8".repeat(64),
    independent_l2_authority: true,
    sealed_outer_temporal_holdout_decision: "passed",
    source_snapshot: {
      availability_status: "verified_preevent",
      participant_cluster_status: "team_or_series_available",
      series_grouped: true,
    },
    holdouts: {
      future_patch: "passed",
      league: "passed",
      international_event_or_meta: "passed",
      roster_change: "not_required_for_neutral",
      sparse_or_new_champion: "passed",
    },
    reliability: {
      validation_gate_passed: true,
      probability_wording_approved: true,
      baseline_support_verified: true,
      dependence_support_verified: true,
      interval_coverage_verified: true,
    },
    claim_ceiling: {
      descriptive_pre_map_association: true,
      causal_draft_effect: false,
      recommendation: false,
      betting: false,
    },
  };
}

function contractFor(fixture: ReplayFixture, model: TerminalModelArtifact): TerminalContract {
  return {
    season_id: "scryglass:season:dev-2026",
    competition_scope_id: "scryglass:competition-scope:dev",
    competition_scope_kind: "regional_league",
    patch_id: "26.14",
    protocol_id: "scryglass:protocol:dev",
    event_id: null,
    side_mapping: {
      side_a_game_side: "blue",
      side_b_game_side: "red",
      side_a_draft_order: "first",
      side_b_draft_order: "second",
      mapping_source_id: "scryglass:source:protocol",
      available_at: fixture.draft.source_available_at,
      mapping_basis: "observed",
    },
    source_record: {
      source_id: "scryglass:source:drafts",
      source_record_id: fixture.draft.source_record_id,
      source_revision_id: "scryglass:source-revision:1",
      supersedes_source_revision_id: null,
      observed_at: fixture.draft.source_available_at,
      available_at: fixture.draft.source_available_at,
      action_order_source: "observed",
    },
    protocol_validation: {
      status: "validated",
      validator_id: "scryglass:protocol-validator:fixture",
      validator_sha256: "6".repeat(64),
      available_at: fixture.draft.source_available_at,
      action_order_verified: true,
      pick_ban_counts_verified: true,
      canonical_side_mapping_verified: true,
    },
    role_constraint_revisions: [],
    assignment_revisions: [],
    evidence: {
      source_context_coverage: {
        coverage_spec_id: "scryglass:coverage:terminal",
        supported_source_family_ids: ["scryglass:source-family:drafts"],
        supported_context_ids: ["scryglass:context:neutral"],
        missing_required_context_ids: [],
        identity_terms_status: "not_applicable",
        bridge_path_status: "not_applicable",
        fallback_levels: [],
        coverage_gaps: [],
      },
    },
    reliability: {
      label: "limited",
      validation_stratum_id: "scryglass:stratum:terminal",
      stratum_match_status: "matched",
      stratum_mapping_sha256: "b".repeat(64),
      benchmark_version: "2.0.0",
      baseline_id: "scryglass:baseline:terminal",
      probability_wording_approved: true,
      validation_gate_passed: true,
      out_of_distribution: false,
      out_of_distribution_flags: [],
      log_loss: 0.69,
      baseline_log_loss: 0.70,
      log_loss_skill: 0.01,
      brier_score: 0.24,
      baseline_brier_score: 0.25,
      brier_skill: 0.01,
      calibration_intercept: 0,
      calibration_slope: 1,
      empirical_interval_coverage: 0.95,
      nominal_interval_coverage: 0.95,
      sample_count: 100,
      cluster_count: 50,
    },
    calibration_id: "scryglass:calibration:terminal-dev",
    provenance: {
      schema_version: "2.0.0",
      model_version: model.modelVersion,
      as_of: model.modelAsOf,
      prediction_id: "scryglass:prediction:fixture",
      mode: "forecast",
      created_at: "2026-07-01T11:30:00Z",
      event_start: fixture.draft.event_start,
      availability_replayed: true,
      sealed_before_event_start: true,
      input_snapshot_id: "scryglass:input:fixture",
      estimator_id: "scryglass:estimator:terminal",
      calibration_id: "scryglass:calibration:terminal-dev",
      probability_transform: {
        transform_sha256: "c".repeat(64),
        probability_domain: "open_0_1",
        monotonicity: "nondecreasing",
        complement_symmetry_verified: true,
        open_support_verified: true,
        transform_proof_sha256: "d".repeat(64),
      },
      required_input_status: "complete",
      freshness_checks: [{ input_id: "scryglass:source:drafts", source_updated_at: fixture.draft.source_available_at, limit_seconds: 3600, fresh: true }],
      input_conflicts: [],
      fallback_levels: [],
      out_of_distribution_flags: [],
      output_sha256: "e".repeat(64),
      immutable: true,
    },
  };
}

test("TypeScript terminal replay matches the shared Python fixture", () => {
  const fixture = loadFixture();
  const raw = readFileSync(modelPath);
  const model = loadTerminalModelArtifact(raw, { expectedArtifactSha256: fixture.model_artifact_sha256 });
  const draft = terminalDraftFromSides({
    sideA: fixture.draft.side_a as TerminalSide,
    sideB: fixture.draft.side_b as TerminalSide,
    eventStart: fixture.draft.event_start,
    sourceAvailableAt: fixture.draft.source_available_at,
    sourceRecordId: fixture.draft.source_record_id,
    sourcePayloadSha256: fixture.draft.source_payload_sha256,
    sourceRightsStatus: fixture.draft.source_rights_status,
    mode: fixture.draft.mode,
    actions: fixture.draft.actions as TerminalAction[] | undefined,
    finalAssignments: fixture.draft.final_assignments as TerminalAssignment[] | undefined,
  });
  assert.deepEqual(scoreTerminalDraft(draft, model, { development: true }), fixture.expected_development);
});

test("TypeScript replays the refit neutral development artifact", () => {
  const fixture = loadFixtureAt(neutralFixturePath);
  const raw = readFileSync(neutralModelPath);
  const model = loadTerminalModelArtifact(raw, { expectedArtifactSha256: fixture.model_artifact_sha256 });
  const draft = terminalDraftFromSides({
    sideA: fixture.draft.side_a as TerminalSide,
    sideB: fixture.draft.side_b as TerminalSide,
    eventStart: fixture.draft.event_start,
    sourceAvailableAt: fixture.draft.source_available_at,
    sourceRecordId: fixture.draft.source_record_id,
    sourcePayloadSha256: fixture.draft.source_payload_sha256,
    sourceRightsStatus: fixture.draft.source_rights_status,
    mode: fixture.draft.mode,
    actions: fixture.draft.actions as TerminalAction[] | undefined,
    finalAssignments: fixture.draft.final_assignments as TerminalAssignment[] | undefined,
  });
  assert.deepEqual(scoreTerminalDraft(draft, model, { development: true }), fixture.expected_development);
});

test("TypeScript terminal scorer preserves side-swap complement symmetry", () => {
  const fixture = loadFixtureAt(neutralFixturePath);
  const model = loadTerminalModelArtifact(readFileSync(neutralModelPath), { expectedArtifactSha256: fixture.model_artifact_sha256 });
  const build = (sideA: Record<string, string>, sideB: Record<string, string>) => terminalDraftFromSides({
    sideA: sideA as TerminalSide,
    sideB: sideB as TerminalSide,
    eventStart: fixture.draft.event_start,
    sourceAvailableAt: fixture.draft.source_available_at,
    sourceRecordId: fixture.draft.source_record_id,
    sourcePayloadSha256: fixture.draft.source_payload_sha256,
    sourceRightsStatus: fixture.draft.source_rights_status,
    mode: "neutral",
  });
  const first = scoreTerminalDraft(build(fixture.draft.side_a, fixture.draft.side_b), model, { development: true });
  const swapped = scoreTerminalDraft(build(fixture.draft.side_b, fixture.draft.side_a), model, { development: true });
  assert.equal(first.status, "development_only");
  assert.equal(swapped.status, "development_only");
  assert.ok(Math.abs(
    (swapped.standardized_map_win_probability_a as number) - (1 - (first.standardized_map_win_probability_a as number)),
  ) <= 1e-12);
  assert.ok(Math.abs(
    (swapped.interval_95 as Record<string, number>).lower - (1 - (first.interval_95 as Record<string, number>).upper),
  ) <= 1e-12);
  assert.ok(Math.abs(
    (swapped.interval_95 as Record<string, number>).upper - (1 - (first.interval_95 as Record<string, number>).lower),
  ) <= 1e-12);
});

test("TypeScript terminal public boundary remains unavailable without promotion", () => {
  assert.equal(canonicalDraftServingAuthorityAvailable(), false);
  const fixture = loadFixture();
  const raw = readFileSync(modelPath);
  const model = loadTerminalModelArtifact(raw, { expectedArtifactSha256: fixture.model_artifact_sha256 });
  const draft = terminalDraftFromSides({
    sideA: fixture.draft.side_a as TerminalSide,
    sideB: fixture.draft.side_b as TerminalSide,
    eventStart: fixture.draft.event_start,
    sourceAvailableAt: fixture.draft.source_available_at,
    sourceRecordId: fixture.draft.source_record_id,
    sourcePayloadSha256: fixture.draft.source_payload_sha256,
    sourceRightsStatus: fixture.draft.source_rights_status,
    mode: fixture.draft.mode,
  });
  const result = scoreTerminalDraft(draft, model);
  assert.equal(result.status, "unavailable");
  assert.equal((result.error as Record<string, unknown>).code, "model_not_promoted");
  assert.equal("score_a" in result, false);
});

test("TypeScript model as-of must be strictly before event start", () => {
  const fixture = loadFixture();
  const model = loadTerminalModelArtifact(readFileSync(modelPath), { expectedArtifactSha256: fixture.model_artifact_sha256 });
  const atEventStart = { ...model, modelAsOf: fixture.draft.event_start };
  const draft = terminalDraftFromSides({
    sideA: fixture.draft.side_a as TerminalSide,
    sideB: fixture.draft.side_b as TerminalSide,
    eventStart: fixture.draft.event_start,
    sourceAvailableAt: fixture.draft.source_available_at,
    sourceRecordId: fixture.draft.source_record_id,
    sourcePayloadSha256: fixture.draft.source_payload_sha256,
    sourceRightsStatus: fixture.draft.source_rights_status,
    mode: fixture.draft.mode,
  });
  const result = scoreTerminalDraft(draft, atEventStart, { development: true });
  assert.equal(result.status, "unavailable");
  assert.equal((result.error as Record<string, unknown>).code, "prediction_time_violation");
});

test("TypeScript promoted-mode input requires legal final role assignments", () => {
  const fixture = loadFixture();
  const roles = ["top", "jungle", "mid", "bot", "support"] as const;
  const actions: TerminalAction[] = [];
  const finalAssignments: TerminalAssignment[] = [];
  let slot = 1;
  for (const role of roles) {
    for (const [canonicalSide, champions] of [["A", fixture.draft.side_a], ["B", fixture.draft.side_b]] as const) {
      const actionId = `scryglass:action:${slot}`;
      actions.push({ action_id: actionId, slot, kind: "pick", canonical_side: canonicalSide, champion_id: champions[role], role_set: [role] });
      finalAssignments.push({ action_id: actionId, canonical_side: canonicalSide, champion_id: champions[role], role });
      slot += 1;
    }
  }
  const complete = terminalDraftFromSides({
    sideA: fixture.draft.side_a as TerminalSide,
    sideB: fixture.draft.side_b as TerminalSide,
    eventStart: fixture.draft.event_start,
    sourceAvailableAt: fixture.draft.source_available_at,
    sourceRecordId: fixture.draft.source_record_id,
    sourcePayloadSha256: fixture.draft.source_payload_sha256,
    sourceRightsStatus: fixture.draft.source_rights_status,
    actions,
    finalAssignments,
  });
  assert.equal(complete.actions.length, 10);
  const promotedModel = loadTerminalModelArtifact(readFileSync(modelPath), { expectedArtifactSha256: fixture.model_artifact_sha256, authorizesPrediction: true });
  const promoted = scoreTerminalDraft(complete, promotedModel, {
    promotionReceipt: {
      schema_version: "draft-terminal-promotion-receipt-v1",
      status: "approved",
      model_version: promotedModel.modelVersion,
      artifact_sha256: promotedModel.artifactSha256,
      l2_contract_sha256: "5".repeat(64),
      development_evaluation_sha256: "f".repeat(64),
      candidate_registry_sha256: "1".repeat(64),
      calibration_transform_sha256: "2".repeat(64),
      reliability_artifact_sha256: "3".repeat(64),
      replay_parity_evidence_sha256: "4".repeat(64),
      independent_authority_record_sha256: "9".repeat(64),
      independent_l2_authority: true,
      final_temporal_holdout_sealed: true,
      public_probability_authorized: true,
      replay_parity_verified: true,
      reliability_gate_passed: true,
      contextual_g1_authority: "not_applicable",
      authority_record_id: "test-only:mechanics-receipt",
      issued_at: "2026-07-01T13:00:00Z",
    },
    promotionBindings: {
      development_evaluation_sha256: "f".repeat(64),
      candidate_registry_sha256: "1".repeat(64),
      l2_contract_sha256: "5".repeat(64),
      calibration_transform_sha256: "2".repeat(64),
      reliability_artifact_sha256: "3".repeat(64),
      replay_parity_evidence_sha256: "4".repeat(64),
      independent_authority_record_sha256: "9".repeat(64),
      authority_record_id: "test-only:mechanics-receipt",
    },
    protocolValidation: {
      status: "validated",
      validator_id: "scryglass:protocol-validator:fixture",
      validator_sha256: "6".repeat(64),
      available_at: "2026-07-01T11:00:00Z",
      action_order_verified: true,
      pick_ban_counts_verified: true,
      canonical_side_mapping_verified: true,
    },
  });
  assert.equal(promoted.status, "ok");
  assert.equal(promoted.input_id, "e0724c9c7c415e6f1ef9d7dcf75c796c406848d9c081db7d877f6bab3a9d4c0b");
  assert.throws(() => terminalDraftFromSides({
    sideA: fixture.draft.side_a as TerminalSide,
    sideB: fixture.draft.side_b as TerminalSide,
    eventStart: fixture.draft.event_start,
    sourceAvailableAt: fixture.draft.source_available_at,
    sourceRecordId: fixture.draft.source_record_id,
    sourcePayloadSha256: fixture.draft.source_payload_sha256,
    sourceRightsStatus: fixture.draft.source_rights_status,
    actions,
    finalAssignments: [...finalAssignments.slice(0, -1), { ...finalAssignments.at(-1)!, role: "top" }],
  }), /not legal/);
  assert.throws(() => terminalDraftFromSides({
    sideA: fixture.draft.side_a as TerminalSide,
    sideB: fixture.draft.side_b as TerminalSide,
    eventStart: fixture.draft.event_start,
    sourceAvailableAt: fixture.draft.source_available_at,
    sourceRecordId: fixture.draft.source_record_id,
    sourcePayloadSha256: fixture.draft.source_payload_sha256,
    sourceRightsStatus: fixture.draft.source_rights_status,
    actions: [...actions.slice(0, -1), { ...actions.at(-1)!, kind: "draft" as never }],
    finalAssignments,
  }), /pick or ban/);
  assert.throws(() => terminalDraftFromSides({
    sideA: fixture.draft.side_a as TerminalSide,
    sideB: fixture.draft.side_b as TerminalSide,
    eventStart: fixture.draft.event_start,
    sourceAvailableAt: fixture.draft.source_available_at,
    sourceRecordId: fixture.draft.source_record_id,
    sourcePayloadSha256: fixture.draft.source_payload_sha256,
    sourceRightsStatus: fixture.draft.source_rights_status,
    actions: [{ ...actions[0], slot: 1.5 }, ...actions.slice(1)],
    finalAssignments,
  }), /contiguous and ordered/);
  assert.throws(() => terminalDraftFromSides({
    sideA: { ...fixture.draft.side_a, top: "Camille" } as TerminalSide,
    sideB: fixture.draft.side_b as TerminalSide,
    eventStart: fixture.draft.event_start,
    sourceAvailableAt: fixture.draft.source_available_at,
    sourceRecordId: fixture.draft.source_record_id,
    sourcePayloadSha256: fixture.draft.source_payload_sha256,
    sourceRightsStatus: fixture.draft.source_rights_status,
    actions,
    finalAssignments,
  }), /side composition/);
});

test("TypeScript contract renderer mirrors the authorized Python serving boundary", () => {
  const fixture = loadFixture();
  const model = loadTerminalModelArtifact(readFileSync(modelPath), {
    expectedArtifactSha256: fixture.model_artifact_sha256,
    authorizesPrediction: true,
  });
  const { actions, finalAssignments } = completeHistory(fixture);
  const draft = terminalDraftFromSides({
    sideA: fixture.draft.side_a as TerminalSide,
    sideB: fixture.draft.side_b as TerminalSide,
    eventStart: fixture.draft.event_start,
    sourceAvailableAt: fixture.draft.source_available_at,
    sourceRecordId: fixture.draft.source_record_id,
    sourcePayloadSha256: fixture.draft.source_payload_sha256,
    sourceRightsStatus: fixture.draft.source_rights_status,
    actions,
    finalAssignments,
  });
  const receipt = authorizedReceipt(model);
  const bindings = {
    development_evaluation_sha256: receipt.development_evaluation_sha256,
    candidate_registry_sha256: receipt.candidate_registry_sha256,
    l2_contract_sha256: receipt.l2_contract_sha256,
    calibration_transform_sha256: receipt.calibration_transform_sha256,
    reliability_artifact_sha256: receipt.reliability_artifact_sha256,
    replay_parity_evidence_sha256: receipt.replay_parity_evidence_sha256,
    independent_authority_record_sha256: receipt.independent_authority_record_sha256,
    authority_record_id: receipt.authority_record_id,
  };
  const contract = contractFor(fixture, model);
  const rendered = renderTerminalContract(draft, model, {
    promotionReceipt: receipt,
    promotionBindings: bindings,
    contract,
  });
  assert.equal(rendered.status, "ok");
  assert.equal(rendered.identity_mode, "neutral");
  assert.equal(rendered.identity_intentionally_omitted, true);
  assert.equal(rendered.as_of, model.modelAsOf);
  assert.equal((rendered.provenance as Record<string, unknown>).created_at, "2026-07-01T11:30:00Z");
  assert.equal((rendered.score_a as number) + (rendered.score_b as number), 100);
  assert.equal("claim_ceiling" in rendered, false);
  assert.equal("error" in rendered, false);
});

test("TypeScript promotion binds receipt calibration and replay hashes to independent authority", () => {
  const fixture = loadFixture();
  const model = loadTerminalModelArtifact(readFileSync(modelPath), {
    expectedArtifactSha256: fixture.model_artifact_sha256,
    authorizesPrediction: true,
  });
  const authority = validateTerminalL2AuthorityRecord(authorityRecordFor(authorizedReceipt(model)), {
    model_artifact_sha256: model.artifactSha256,
    candidate_registry_sha256: "1".repeat(64),
    development_evaluation_sha256: "f".repeat(64),
    l2_contract_sha256: "5".repeat(64),
  });
  assert.equal(authority.calibration_transform_sha256, "2".repeat(64));
  const { actions, finalAssignments } = completeHistory(fixture);
  const draft = terminalDraftFromSides({
    sideA: fixture.draft.side_a as TerminalSide,
    sideB: fixture.draft.side_b as TerminalSide,
    eventStart: fixture.draft.event_start,
    sourceAvailableAt: fixture.draft.source_available_at,
    sourceRecordId: fixture.draft.source_record_id,
    sourcePayloadSha256: fixture.draft.source_payload_sha256,
    sourceRightsStatus: fixture.draft.source_rights_status,
    actions,
    finalAssignments,
  });
  const receipt = authorizedReceipt(model);
  const bindings = {
    development_evaluation_sha256: receipt.development_evaluation_sha256,
    candidate_registry_sha256: receipt.candidate_registry_sha256,
    l2_contract_sha256: receipt.l2_contract_sha256,
    calibration_transform_sha256: authority.calibration_transform_sha256,
    reliability_artifact_sha256: authority.reliability_artifact_sha256,
    replay_parity_evidence_sha256: authority.replay_parity_evidence_sha256,
    independent_authority_record_sha256: receipt.independent_authority_record_sha256,
    authority_record_id: receipt.authority_record_id,
  };
  const unavailable = scoreTerminalDraft(draft, model, {
    promotionReceipt: { ...receipt, calibration_transform_sha256: "a".repeat(64) },
    promotionBindings: bindings,
    protocolValidation: {
      status: "validated",
      validator_id: "scryglass:protocol-validator:fixture",
      validator_sha256: "6".repeat(64),
      available_at: fixture.draft.source_available_at,
      action_order_verified: true,
      pick_ban_counts_verified: true,
      canonical_side_mapping_verified: true,
    },
  });
  assert.equal(unavailable.status, "unavailable");
  assert.equal((unavailable.error as Record<string, unknown>).code, "model_not_promoted");
  const mismatchedRecord = scoreTerminalDraft(draft, model, {
    promotionReceipt: { ...receipt, authority_record_id: "test-only:other-authority" },
    promotionBindings: bindings,
    protocolValidation: {
      status: "validated",
      validator_id: "scryglass:protocol-validator:fixture",
      validator_sha256: "6".repeat(64),
      available_at: fixture.draft.source_available_at,
      action_order_verified: true,
      pick_ban_counts_verified: true,
      canonical_side_mapping_verified: true,
    },
  });
  assert.equal(mismatchedRecord.status, "unavailable");
  assert.equal((mismatchedRecord.error as Record<string, unknown>).code, "model_not_promoted");
});

test("TypeScript contract renderer stays unavailable when the match contract is absent", () => {
  const fixture = loadFixture();
  const model = loadTerminalModelArtifact(readFileSync(modelPath), {
    expectedArtifactSha256: fixture.model_artifact_sha256,
    authorizesPrediction: true,
  });
  const { actions, finalAssignments } = completeHistory(fixture);
  const draft = terminalDraftFromSides({
    sideA: fixture.draft.side_a as TerminalSide,
    sideB: fixture.draft.side_b as TerminalSide,
    eventStart: fixture.draft.event_start,
    sourceAvailableAt: fixture.draft.source_available_at,
    sourceRecordId: fixture.draft.source_record_id,
    sourcePayloadSha256: fixture.draft.source_payload_sha256,
    sourceRightsStatus: fixture.draft.source_rights_status,
    actions,
    finalAssignments,
  });
  const receipt = authorizedReceipt(model);
  const result = renderTerminalContract(draft, model, {
    promotionReceipt: receipt,
    promotionBindings: {
      development_evaluation_sha256: receipt.development_evaluation_sha256,
      candidate_registry_sha256: receipt.candidate_registry_sha256,
      l2_contract_sha256: receipt.l2_contract_sha256,
      calibration_transform_sha256: receipt.calibration_transform_sha256,
      reliability_artifact_sha256: receipt.reliability_artifact_sha256,
      replay_parity_evidence_sha256: receipt.replay_parity_evidence_sha256,
      independent_authority_record_sha256: receipt.independent_authority_record_sha256,
      authority_record_id: receipt.authority_record_id,
    },
    protocolValidation: {
      status: "validated",
      validator_id: "scryglass:protocol-validator:fixture",
      validator_sha256: "6".repeat(64),
      available_at: fixture.draft.source_available_at,
      action_order_verified: true,
      pick_ban_counts_verified: true,
      canonical_side_mapping_verified: true,
    },
  });
  assert.equal(result.status, "unavailable");
  assert.equal((result.error as Record<string, unknown>).code, "missing_required_input");
  assert.equal("score_a" in result, false);
  assert.deepEqual((result.error as Record<string, unknown>).missing_fields, [
    "season_id",
    "competition_scope_id",
    "competition_scope_kind",
    "patch_id",
    "protocol_id",
    "side_mapping",
    "source_record",
    "protocol_validation",
    "role_constraint_revisions",
    "assignment_revisions",
    "evidence",
    "reliability",
    "calibration_id",
    "provenance",
  ]);
});

test("TypeScript terminal input identity binds source availability and payload hash", () => {
  const fixture = loadFixture();
  const build = (sourceAvailableAt: string, sourcePayloadSha256: string) => terminalDraftFromSides({
    sideA: fixture.draft.side_a as TerminalSide,
    sideB: fixture.draft.side_b as TerminalSide,
    eventStart: fixture.draft.event_start,
    sourceAvailableAt,
    sourceRecordId: fixture.draft.source_record_id,
    sourcePayloadSha256,
    sourceRightsStatus: fixture.draft.source_rights_status,
    mode: fixture.draft.mode,
  });
  const baseline = terminalInputId(build(fixture.draft.source_available_at, fixture.draft.source_payload_sha256));
  assert.notEqual(terminalInputId(build("2026-07-01T10:59:59Z", fixture.draft.source_payload_sha256)), baseline);
  assert.notEqual(terminalInputId(build(fixture.draft.source_available_at, "b".repeat(64))), baseline);
});

test("TypeScript terminal input hash matches Python ASCII canonicalization", () => {
  const fixture = loadFixture();
  const draft = terminalDraftFromSides({
    sideA: fixture.draft.side_a as TerminalSide,
    sideB: fixture.draft.side_b as TerminalSide,
    eventStart: fixture.draft.event_start,
    sourceAvailableAt: fixture.draft.source_available_at,
    sourceRecordId: "source:á",
    sourcePayloadSha256: fixture.draft.source_payload_sha256,
    sourceRightsStatus: fixture.draft.source_rights_status,
    mode: fixture.draft.mode,
  });
  assert.equal(terminalInputId(draft), "b8a769fdfed02f3d5fdbf8c55ab595e45b0a96ee1e25d276bab87ca7b96cad69");
});

test("TypeScript contextual input separates verified G1 evidence from model approval", () => {
  const fixture = loadFixture();
  const evidence = g1Evidence();
  assert.equal(g1RosterEvidenceAvailable(evidence, fixture.draft.event_start), true);
  assert.equal(g1RosterEvidenceAvailable({ ...evidence, source_available_at: "2026-07-01T12:00:01Z" }, fixture.draft.event_start), false);
  assert.equal(g1RosterEvidenceAvailable({ ...evidence, source_available_at: fixture.draft.event_start }, fixture.draft.event_start), false);
  assert.equal(g1RosterEvidenceAvailable({ ...evidence, source_payload_sha256: "a".repeat(64) }, fixture.draft.event_start), false);
  assert.equal(g1RosterEvidenceAvailable({ ...evidence, source_payload_base64: "" }, fixture.draft.event_start), false);
  const contextDraft = terminalDraftFromSides({
    sideA: fixture.draft.side_a as TerminalSide,
    sideB: fixture.draft.side_b as TerminalSide,
    eventStart: fixture.draft.event_start,
    sourceAvailableAt: fixture.draft.source_available_at,
    sourceRecordId: fixture.draft.source_record_id,
    sourcePayloadSha256: fixture.draft.source_payload_sha256,
    sourceRightsStatus: fixture.draft.source_rights_status,
    mode: "contextual",
    rosterEvidence: evidence,
  });
  const model = loadTerminalModelArtifact(readFileSync(modelPath), { expectedArtifactSha256: fixture.model_artifact_sha256, authorizesPrediction: true });
  const result = scoreTerminalDraft(contextDraft, model);
  assert.equal(result.status, "unavailable");
  assert.equal((result.error as Record<string, unknown>).code, "model_not_promoted");
  assert.deepEqual((result.error as Record<string, unknown>).missing_fields, ["contextual_fit_model", "player_champion_response", "team_policy_response"]);
});

test("TypeScript canonical server adapter preserves contextual G1 mode", () => {
  const fixture = loadFixture();
  const evidence = g1Evidence("source:g1-server-fixture");
  const result = scoreCanonicalTerminalDraft({
    sideA: fixture.draft.side_a as TerminalSide,
    sideB: fixture.draft.side_b as TerminalSide,
    eventStart: fixture.draft.event_start,
    sourceAvailableAt: fixture.draft.source_available_at,
    sourceRecordId: fixture.draft.source_record_id,
    sourcePayloadSha256: fixture.draft.source_payload_sha256,
    sourceRightsStatus: fixture.draft.source_rights_status,
    mode: "contextual",
    rosterEvidence: evidence,
  });
  assert.equal(result.status, "unavailable");
  assert.equal((result.error as Record<string, unknown>).code, "model_not_promoted");
  assert.deepEqual((result.error as Record<string, unknown>).missing_fields, [
    "contextual_fit_model",
    "player_champion_response",
    "team_policy_response",
  ]);
});
