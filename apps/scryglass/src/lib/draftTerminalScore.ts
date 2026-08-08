/**
 * Canonical L7 terminal Draft Score replay.
 *
 * This is intentionally separate from the legacy exploratory draft scorer in
 * draftScore.ts. It mirrors the Python terminal mechanics and remains
 * unavailable for public prediction until an independent promotion record
 * authorizes the artifact.
 */

import { Buffer } from "node:buffer";
import { createHash } from "node:crypto";

export const TERMINAL_ROLES = ["top", "jungle", "mid", "bot", "support"] as const;
export type TerminalRole = (typeof TERMINAL_ROLES)[number];
export type TerminalMode = "neutral" | "contextual";
export type TerminalSide = Record<TerminalRole, string>;
export type TerminalAction = {
  action_id: string;
  slot: number;
  kind: "pick" | "ban";
  canonical_side: "A" | "B";
  champion_id: string;
  role_set?: TerminalRole[];
};
export type TerminalAssignment = {
  action_id: string;
  canonical_side: "A" | "B";
  champion_id: string;
  role: TerminalRole;
};
export type TerminalG1RosterEvidence = {
  event_start: string;
  source_record_id: string;
  source_available_at: string;
  source_retrieved_at: string;
  source_rights_status: "reviewed";
  source_payload_sha256: string;
  source_payload_base64: string;
  roster_a_id: string;
  roster_b_id: string;
  starters_a: Array<{ role: TerminalRole; player_id: string }>;
  starters_b: Array<{ role: TerminalRole; player_id: string }>;
};

const SHA256_RE = /^[0-9a-f]{64}$/;
const G1_SCHEMA_VERSION = "scryglass:g1-roster-payload:v1";
const BASE64_RE = /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/;
const MODEL_VERSION_RE = /^[A-Za-z][A-Za-z0-9_-]*-v[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?$/;
const Z95 = 1.959963984540054;

export class TerminalDraftError extends Error {}

export type TerminalDraft = {
  sideA: Array<[TerminalRole, string]>;
  sideB: Array<[TerminalRole, string]>;
  eventStart: string;
  sourceAvailableAt: string;
  sourceRecordId: string;
  sourcePayloadSha256: string;
  sourceRightsStatus: "reviewed" | "unknown";
  mode: TerminalMode;
  rosterEvidence?: TerminalG1RosterEvidence;
  actions: TerminalAction[];
  finalAssignments: TerminalAssignment[];
};

export type TerminalModelArtifact = {
  modelVersion: string;
  modelAsOf: string;
  intercept: number;
  calibrationSlope: number;
  calibrationIntercept: number;
  uncertaintyLogitSd: number;
  championRoleLogit: Record<string, number>;
  allySynergyLogit: Record<string, number>;
  counterLogit: Record<string, number>;
  artifactSha256: string;
  authorizesPrediction: boolean;
};

export type TerminalPromotionReceipt = {
  schema_version: "draft-terminal-promotion-receipt-v1";
  status: "approved" | string;
  model_version: string;
  artifact_sha256: string;
  l2_contract_sha256: string;
  development_evaluation_sha256: string;
  candidate_registry_sha256: string;
  calibration_transform_sha256: string;
  reliability_artifact_sha256: string;
  replay_parity_evidence_sha256: string;
  independent_authority_record_sha256: string;
  independent_l2_authority: true;
  final_temporal_holdout_sealed: true;
  public_probability_authorized: boolean;
  replay_parity_verified: true;
  reliability_gate_passed: true;
  contextual_g1_authority: "not_applicable" | "approved";
  authority_record_id: string;
  issued_at: string;
};

export type TerminalL2AuthorityRecord = {
  schema_version: "scryglass:draft-terminal-l2-authority-record:v1";
  status: "approved";
  authority_record_id: string;
  issued_at: string;
  independent_reviewer_id: string;
  model_artifact_sha256: string;
  candidate_registry_sha256: string;
  development_evaluation_sha256: string;
  l2_contract_sha256: string;
  calibration_transform_sha256: string;
  reliability_artifact_sha256: string;
  replay_parity_evidence_sha256: string;
  source_snapshot_sha256: string;
  independent_l2_authority: true;
  sealed_outer_temporal_holdout_decision: "passed";
  source_snapshot: {
    availability_status: "verified_preevent";
    participant_cluster_status: "team_or_series_available";
    series_grouped: true;
  };
  holdouts: {
    future_patch: "passed";
    league: "passed";
    international_event_or_meta: "passed";
    roster_change: "not_required_for_neutral";
    sparse_or_new_champion: "passed";
  };
  reliability: {
    validation_gate_passed: true;
    probability_wording_approved: true;
    baseline_support_verified: true;
    dependence_support_verified: true;
    interval_coverage_verified: true;
  };
  claim_ceiling: {
    descriptive_pre_map_association: true;
    causal_draft_effect: false;
    recommendation: false;
    betting: false;
  };
};

export type TerminalL2AuthorityExpectedBindings = {
  model_artifact_sha256?: string;
  candidate_registry_sha256?: string;
  development_evaluation_sha256?: string;
  l2_contract_sha256?: string;
};

export type TerminalPromotionBindings = {
  development_evaluation_sha256: string;
  candidate_registry_sha256: string;
  l2_contract_sha256: string;
  calibration_transform_sha256: string;
  reliability_artifact_sha256: string;
  replay_parity_evidence_sha256: string;
  independent_authority_record_sha256: string;
  authority_record_id: string;
};

export type TerminalProtocolValidation = {
  status: "validated" | string;
  validator_id: string;
  validator_sha256: string;
  available_at: string;
  action_order_verified: boolean;
  pick_ban_counts_verified: boolean;
  canonical_side_mapping_verified: boolean;
};

export type TerminalContract = Record<string, unknown>;
type JsonObject = TerminalContract;

function fail(message: string): never {
  throw new TerminalDraftError(message);
}

function finite(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) fail(`${field} must be finite numeric`);
  return value;
}

function mapping(value: unknown, field: string): Record<string, number> {
  if (value == null || typeof value !== "object" || Array.isArray(value)) fail(`${field} must be a mapping`);
  const result: Record<string, number> = {};
  for (const [key, raw] of Object.entries(value as JsonObject)) result[key] = finite(raw, `${field}.${key}`);
  return result;
}

function parseTimestamp(value: unknown, field: string): number {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/.test(value)) {
    fail(`${field} must be an RFC-3339 timestamp`);
  }
  const parsed = Date.parse(value);
  if (!Number.isFinite(parsed)) fail(`${field} must be an RFC-3339 timestamp`);
  return parsed;
}

function exactObject(value: unknown, field: string, expectedKeys: readonly string[]): Record<string, unknown> {
  if (value == null || typeof value !== "object" || Array.isArray(value)) fail(`${field} must be an object`);
  const object = value as Record<string, unknown>;
  const actualKeys = Object.keys(object).sort();
  const requiredKeys = [...expectedKeys].sort();
  if (actualKeys.length !== requiredKeys.length || actualKeys.some((key, index) => key !== requiredKeys[index])) {
    fail(`${field} keys do not match the frozen authority contract`);
  }
  return object;
}

function nonEmptyString(value: unknown, field: string): string {
  if (typeof value !== "string" || value.trim().length === 0) fail(`${field} must be a non-empty string`);
  return value;
}

/** Validate the independent L2 record before its hashes enter serving bindings. */
export function validateTerminalL2AuthorityRecord(
  value: unknown,
  expectedBindings: TerminalL2AuthorityExpectedBindings = {},
): TerminalL2AuthorityRecord {
  const record = exactObject(value, "authority record", [
    "schema_version",
    "status",
    "authority_record_id",
    "issued_at",
    "independent_reviewer_id",
    "model_artifact_sha256",
    "candidate_registry_sha256",
    "development_evaluation_sha256",
    "l2_contract_sha256",
    "calibration_transform_sha256",
    "reliability_artifact_sha256",
    "replay_parity_evidence_sha256",
    "source_snapshot_sha256",
    "independent_l2_authority",
    "sealed_outer_temporal_holdout_decision",
    "source_snapshot",
    "holdouts",
    "reliability",
    "claim_ceiling",
  ]);
  if (record.schema_version !== "scryglass:draft-terminal-l2-authority-record:v1" || record.status !== "approved") {
    fail("authority record is not an approved v1 record");
  }
  nonEmptyString(record.authority_record_id, "authority_record_id");
  nonEmptyString(record.independent_reviewer_id, "independent_reviewer_id");
  parseTimestamp(record.issued_at, "authority_record.issued_at");
  for (const field of [
    "model_artifact_sha256",
    "candidate_registry_sha256",
    "development_evaluation_sha256",
    "l2_contract_sha256",
    "calibration_transform_sha256",
    "reliability_artifact_sha256",
    "replay_parity_evidence_sha256",
    "source_snapshot_sha256",
  ] as const) {
    if (typeof record[field] !== "string" || !SHA256_RE.test(record[field])) fail(`${field} must be a lowercase SHA-256`);
  }
  if (record.independent_l2_authority !== true || record.sealed_outer_temporal_holdout_decision !== "passed") {
    fail("authority record has not passed independent L2 and outer-holdout authority");
  }

  const sourceSnapshot = exactObject(record.source_snapshot, "authority record source_snapshot", [
    "availability_status",
    "participant_cluster_status",
    "series_grouped",
  ]);
  if (
    sourceSnapshot.availability_status !== "verified_preevent" ||
    sourceSnapshot.participant_cluster_status !== "team_or_series_available" ||
    sourceSnapshot.series_grouped !== true
  ) {
    fail("authority record source snapshot does not provide verified pre-event series dependence");
  }

  const holdouts = exactObject(record.holdouts, "authority record holdouts", [
    "future_patch",
    "league",
    "international_event_or_meta",
    "roster_change",
    "sparse_or_new_champion",
  ]);
  if (
    holdouts.future_patch !== "passed" ||
    holdouts.league !== "passed" ||
    holdouts.international_event_or_meta !== "passed" ||
    holdouts.roster_change !== "not_required_for_neutral" ||
    holdouts.sparse_or_new_champion !== "passed"
  ) {
    fail("authority record required neutral holdouts are not passed");
  }

  const reliability = exactObject(record.reliability, "authority record reliability", [
    "validation_gate_passed",
    "probability_wording_approved",
    "baseline_support_verified",
    "dependence_support_verified",
    "interval_coverage_verified",
  ]);
  if (Object.values(reliability).some((entry) => entry !== true)) fail("authority record reliability gates must all be true");

  const claimCeiling = exactObject(record.claim_ceiling, "authority record claim_ceiling", [
    "descriptive_pre_map_association",
    "causal_draft_effect",
    "recommendation",
    "betting",
  ]);
  if (
    claimCeiling.descriptive_pre_map_association !== true ||
    claimCeiling.causal_draft_effect !== false ||
    claimCeiling.recommendation !== false ||
    claimCeiling.betting !== false
  ) {
    fail("authority record claim ceiling is too broad");
  }

  for (const field of [
    "model_artifact_sha256",
    "candidate_registry_sha256",
    "development_evaluation_sha256",
    "l2_contract_sha256",
  ] as const) {
    const expected = expectedBindings[field];
    if (expected != null && record[field] !== expected) fail(`authority record does not bind ${field}`);
  }
  return record as unknown as TerminalL2AuthorityRecord;
}

function canonicalString(value: string): string {
  const encoded = JSON.stringify(value);
  let output = "";
  for (let index = 0; index < encoded.length; index += 1) {
    const codeUnit = encoded.charCodeAt(index);
    output += codeUnit > 0x7f
      ? `\\u${codeUnit.toString(16).padStart(4, "0")}`
      : encoded[index];
  }
  return output;
}

function canonicalJson(value: unknown): string {
  if (typeof value === "string") return canonicalString(value);
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  const entries = Object.entries(value as JsonObject)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, child]) => `${canonicalString(key)}:${canonicalJson(child)}`);
  return `{${entries.join(",")}}`;
}

function sha256Canonical(value: unknown): string {
  return createHash("sha256").update(canonicalJson(value), "utf8").digest("hex");
}

function sha256Bytes(raw: Uint8Array): string {
  return createHash("sha256").update(raw).digest("hex");
}

function pair(first: string, second: string): string {
  return [first, second].sort().join("|");
}

function counterKey(role: TerminalRole, first: string, second: string): string {
  return `${role}|${pair(first, second)}`;
}

function sigmoid(logit: number): number {
  if (logit >= 40) return 1;
  if (logit <= -40) return 0;
  return 1 / (1 + Math.exp(-logit));
}

function normalizeSide(side: unknown, field: string): Array<[TerminalRole, string]> {
  if (side == null || typeof side !== "object" || Array.isArray(side)) fail(`${field} must contain exactly canonical roles`);
  const raw = side as Record<string, unknown>;
  const keys = Object.keys(raw).sort();
  if (keys.length !== TERMINAL_ROLES.length || keys.join("|") !== [...TERMINAL_ROLES].sort().join("|")) {
    fail(`${field} must contain exactly canonical roles`);
  }
  const normalized: Array<[TerminalRole, string]> = [];
  for (const role of TERMINAL_ROLES) {
    const champion = raw[role];
    if (typeof champion !== "string" || !champion.trim()) fail(`${field}.${role} must be a non-empty champion id`);
    normalized.push([role, champion.trim()]);
  }
  if (new Set(normalized.map(([, champion]) => champion)).size !== normalized.length) {
    fail(`${field} cannot contain duplicate champions`);
  }
  return normalized;
}

function decodeG1Payload(evidence: TerminalG1RosterEvidence): Record<string, unknown> {
  if (typeof evidence.source_payload_base64 !== "string" || !BASE64_RE.test(evidence.source_payload_base64) || evidence.source_payload_base64.length === 0) {
    fail("G1 roster payload must contain canonical base64 source bytes");
  }
  const raw = Buffer.from(evidence.source_payload_base64, "base64");
  if (raw.length === 0 || raw.toString("base64") !== evidence.source_payload_base64) {
    fail("G1 roster payload base64 is not canonical");
  }
  if (sha256Bytes(raw) !== evidence.source_payload_sha256) {
    fail("G1 roster payload hash does not match the supplied source bytes");
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(raw));
  } catch {
    fail("G1 roster payload must be strict UTF-8 JSON");
  }
  return exactObject(parsed, "G1 roster payload", [
    "schema_version",
    "source_record_id",
    "event_start",
    "available_at",
    "retrieved_at",
    "rights_status",
    "rosters",
  ]);
}

function validateG1RosterEvidence(evidence: TerminalG1RosterEvidence, eventStart: string): void {
  const payload = decodeG1Payload(evidence);
  if (payload.schema_version !== G1_SCHEMA_VERSION) fail("G1 roster payload schema_version is not supported");
  if (payload.source_record_id !== evidence.source_record_id) fail("G1 roster payload source_record_id does not match the evidence");
  if (payload.event_start !== evidence.event_start) fail("G1 roster payload event_start does not match the evidence");
  if (payload.available_at !== evidence.source_available_at) fail("G1 roster payload available_at does not match the evidence");
  if (payload.retrieved_at !== evidence.source_retrieved_at) fail("G1 roster payload retrieved_at does not match the evidence");
  if (payload.rights_status !== evidence.source_rights_status) fail("G1 roster payload rights_status does not match the evidence");
  if (evidence.event_start !== eventStart) fail("G1 roster evidence event_start does not match the draft");
  if (evidence.source_rights_status !== "reviewed") fail("G1 roster rights must be reviewed");
  if (!SHA256_RE.test(evidence.source_payload_sha256)) fail("G1 roster payload hash must be a lowercase SHA-256");
  if (!evidence.source_record_id || !evidence.roster_a_id || !evidence.roster_b_id || evidence.roster_a_id === evidence.roster_b_id) {
    fail("G1 roster evidence requires distinct source and roster ids");
  }
  const expectedRoles = [...TERMINAL_ROLES];
  const rosters = exactObject(payload.rosters, "G1 roster payload rosters", ["A", "B"]);
  for (const [side, starters, rosterId] of [["A", evidence.starters_a, evidence.roster_a_id], ["B", evidence.starters_b, evidence.roster_b_id]] as const) {
    if (starters.length !== expectedRoles.length || starters.map((starter) => starter.role).join("|") !== expectedRoles.join("|")) {
      fail(`G1 roster ${side} starters must contain each canonical role once`);
    }
    if (new Set(starters.map((starter) => starter.player_id)).size !== starters.length || starters.some((starter) => !starter.player_id)) {
      fail(`G1 roster ${side} starters must contain unique player ids`);
    }
    const payloadRoster = exactObject(rosters[side], `G1 roster payload ${side}`, ["roster_id", "starters"]);
    if (payloadRoster.roster_id !== rosterId) fail(`G1 roster payload ${side} roster_id does not match the evidence`);
    if (!Array.isArray(payloadRoster.starters) || payloadRoster.starters.length !== expectedRoles.length) {
      fail(`G1 roster payload ${side} starters must contain exactly five players`);
    }
    for (let index = 0; index < expectedRoles.length; index += 1) {
      const payloadStarter = exactObject(payloadRoster.starters[index], `G1 roster payload ${side}.starters[${index}]`, ["role", "player_id"]);
      if (payloadStarter.role !== starters[index].role || payloadStarter.player_id !== starters[index].player_id) {
        fail(`G1 roster payload ${side} starters do not match the evidence`);
      }
    }
  }
  const players = [...evidence.starters_a, ...evidence.starters_b].map((starter) => starter.player_id);
  if (new Set(players).size !== players.length) fail("G1 rosters cannot share a starter");
  if (parseTimestamp(evidence.source_available_at, "G1 source_available_at") >= parseTimestamp(eventStart, "eventStart")) {
    fail("G1 roster source is not available before eventStart");
  }
  if (parseTimestamp(evidence.source_retrieved_at, "G1 source_retrieved_at") < parseTimestamp(evidence.source_available_at, "G1 source_available_at")) {
    fail("G1 roster retrieval time predates source availability");
  }
}

function protocolValidationAvailable(
  validation: TerminalProtocolValidation | undefined,
  eventStart: string,
): boolean {
  if (validation == null || validation.status !== "validated") return false;
  if (!validation.validator_id || !SHA256_RE.test(validation.validator_sha256)) return false;
  if (!validation.action_order_verified || !validation.pick_ban_counts_verified || !validation.canonical_side_mapping_verified) return false;
  try {
    return parseTimestamp(validation.available_at, "protocolValidation.available_at") < parseTimestamp(eventStart, "eventStart");
  } catch {
    return false;
  }
}

export function g1RosterEvidenceAvailable(evidence: TerminalG1RosterEvidence, eventStart: string): boolean {
  try {
    validateG1RosterEvidence(evidence, eventStart);
    return true;
  } catch {
    return false;
  }
}

export function validateTerminalActions(
  actions: TerminalAction[],
  finalAssignments: TerminalAssignment[],
  eventStart: string,
  sourceAvailableAt: string,
): void {
  if (actions.length === 0 || finalAssignments.length === 0) fail("terminal input requires actions and final_assignments");
  const slots = actions.map((action) => action.slot);
  if (slots.some((slot) => !Number.isInteger(slot)) || new Set(slots).size !== slots.length || slots.some((slot, index) => slot !== index + 1)) {
    fail("terminal action slots must be contiguous and ordered");
  }
  const actionById = new Map<string, TerminalAction>();
  const champions = new Set<string>();
  for (const action of actions) {
    if (!action.action_id || actionById.has(action.action_id)) fail("terminal actions require unique action_id values");
    if (action.canonical_side !== "A" && action.canonical_side !== "B") fail("terminal actions require canonical sides");
    if (action.kind !== "pick" && action.kind !== "ban") fail("terminal actions require pick or ban kinds");
    if (!action.champion_id || champions.has(action.champion_id)) fail("terminal actions cannot repeat a champion");
    if (action.kind === "pick") {
      if (!action.role_set?.length || new Set(action.role_set).size !== action.role_set.length || action.role_set.some((role) => !TERMINAL_ROLES.includes(role))) {
        fail("pick role_set must contain unique canonical roles");
      }
    }
    actionById.set(action.action_id, action);
    champions.add(action.champion_id);
  }
  const picks = actions.filter((action) => action.kind === "pick");
  if (picks.length !== 10 || picks.filter((action) => action.canonical_side === "A").length !== 5 || picks.filter((action) => action.canonical_side === "B").length !== 5) {
    fail("terminal input requires exactly five picks per canonical side");
  }
  if (finalAssignments.length !== 10) fail("terminal input requires ten final assignments");
  const assignedActions = new Set<string>();
  const assignedChampions = new Set<string>();
  const rolesBySide: Record<"A" | "B", Set<TerminalRole>> = { A: new Set(), B: new Set() };
  for (const assignment of finalAssignments) {
    const action = actionById.get(assignment.action_id);
    if (!action || action.kind !== "pick") fail("final assignments must reference pick actions");
    if (assignment.canonical_side !== action.canonical_side) fail("final assignment side does not match its action");
    if (assignment.champion_id !== action.champion_id) fail("final assignment champion does not match its action");
    if (!action.role_set?.includes(assignment.role)) fail("final assignment role is not legal for its pick");
    if (assignedActions.has(assignment.action_id) || assignedChampions.has(assignment.champion_id) || rolesBySide[assignment.canonical_side].has(assignment.role)) {
      fail("terminal final assignments must be unique by action, champion, and side role");
    }
    assignedActions.add(assignment.action_id);
    assignedChampions.add(assignment.champion_id);
    rolesBySide[assignment.canonical_side].add(assignment.role);
  }
  const requiredRoles = new Set<TerminalRole>(TERMINAL_ROLES);
  if (rolesBySide.A.size !== requiredRoles.size || rolesBySide.B.size !== requiredRoles.size || [...requiredRoles].some((role) => !rolesBySide.A.has(role) || !rolesBySide.B.has(role))) {
    fail("terminal final assignments require every role on both sides");
  }
  if (parseTimestamp(sourceAvailableAt, "sourceAvailableAt") >= parseTimestamp(eventStart, "eventStart")) fail("source is not available before eventStart");
}

export function terminalDraftFromSides(args: {
  sideA: TerminalSide;
  sideB: TerminalSide;
  eventStart: string;
  sourceAvailableAt: string;
  sourceRecordId: string;
  sourcePayloadSha256: string;
  sourceRightsStatus: "reviewed" | "unknown";
  mode?: TerminalMode;
  rosterEvidence?: TerminalG1RosterEvidence;
  actions?: TerminalAction[];
  finalAssignments?: TerminalAssignment[];
}): TerminalDraft {
  const mode = args.mode ?? "neutral";
  if (mode !== "neutral" && mode !== "contextual") fail("mode must be neutral or contextual");
  if (mode === "neutral" && args.rosterEvidence != null) fail("neutral terminal drafts cannot carry contextual roster evidence");
  if (args.rosterEvidence != null) validateG1RosterEvidence(args.rosterEvidence, args.eventStart);
  const sideA = normalizeSide(args.sideA, "sideA");
  const sideB = normalizeSide(args.sideB, "sideB");
  if (new Set([...sideA, ...sideB].map(([, champion]) => champion)).size !== 10) {
    fail("terminal draft requires ten unique champions");
  }
  const eventStart = parseTimestamp(args.eventStart, "eventStart");
  const sourceAvailableAt = parseTimestamp(args.sourceAvailableAt, "sourceAvailableAt");
  if (sourceAvailableAt >= eventStart) fail("source is not available before eventStart");
  if (!args.sourceRecordId) fail("sourceRecordId is required");
  if (!SHA256_RE.test(args.sourcePayloadSha256)) fail("sourcePayloadSha256 must be a lowercase SHA-256");
  if (args.sourceRightsStatus !== "reviewed" && args.sourceRightsStatus !== "unknown") {
    fail("sourceRightsStatus must be reviewed or unknown");
  }
  const actions = args.actions ?? [];
  const finalAssignments = args.finalAssignments ?? [];
  if ((actions.length > 0) !== (finalAssignments.length > 0)) fail("actions and final_assignments must be supplied together");
  if (actions.length > 0) {
    validateTerminalActions(actions, finalAssignments, args.eventStart, args.sourceAvailableAt);
    const expectedBySide: Record<"A" | "B", Record<string, string>> = {
      A: Object.fromEntries(sideA),
      B: Object.fromEntries(sideB),
    };
    const actualBySide: Record<"A" | "B", Record<string, string>> = { A: {}, B: {} };
    for (const assignment of finalAssignments) actualBySide[assignment.canonical_side][assignment.role] = assignment.champion_id;
    if (JSON.stringify(actualBySide) !== JSON.stringify(expectedBySide)) {
      fail("terminal final assignments do not match the scored side composition");
    }
  }
  return {
    sideA,
    sideB,
    eventStart: args.eventStart,
    sourceAvailableAt: args.sourceAvailableAt,
    sourceRecordId: args.sourceRecordId,
    sourcePayloadSha256: args.sourcePayloadSha256,
    sourceRightsStatus: args.sourceRightsStatus,
    mode,
    rosterEvidence: args.rosterEvidence,
    actions,
    finalAssignments,
  };
}

export function terminalInputId(draft: TerminalDraft): string {
  const payload: JsonObject = {
    side_a: draft.sideA.map(([role, championId]) => ({ role, champion_id: championId })),
    side_b: draft.sideB.map(([role, championId]) => ({ role, champion_id: championId })),
    event_start: draft.eventStart,
    source_record_id: draft.sourceRecordId,
    source_available_at: draft.sourceAvailableAt,
    source_payload_sha256: draft.sourcePayloadSha256,
    source_rights_status: draft.sourceRightsStatus,
    mode: draft.mode,
  };
  if (draft.actions.length > 0) {
    payload.actions = draft.actions;
    payload.final_assignments = draft.finalAssignments;
  }
  if (draft.rosterEvidence != null) payload.roster_evidence = draft.rosterEvidence;
  return sha256Canonical(payload);
}

export function loadTerminalModelArtifact(
  raw: Uint8Array,
  options: { expectedArtifactSha256?: string; authorizesPrediction?: boolean } = {},
): TerminalModelArtifact {
  if (!(raw instanceof Uint8Array) || raw.length === 0) fail("model artifact must be non-empty bytes");
  const artifactSha256 = sha256Bytes(raw);
  if (options.expectedArtifactSha256 != null && options.expectedArtifactSha256 !== artifactSha256) {
    fail("model artifact bytes do not match the expected SHA-256");
  }
  let payload: unknown;
  try {
    payload = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(raw));
  } catch (error) {
    fail(`model artifact must be strict UTF-8 JSON: ${error instanceof Error ? error.message : String(error)}`);
  }
  if (payload == null || typeof payload !== "object" || Array.isArray(payload)) fail("model artifact must be a JSON object");
  const object = payload as JsonObject;
  const expectedKeys = [
    "model_version",
    "model_as_of",
    "intercept",
    "calibration_slope",
    "calibration_intercept",
    "uncertainty_logit_sd",
    "champion_role_logit",
    "ally_synergy_logit",
    "counter_logit",
  ].sort();
  const actualKeys = Object.keys(object).sort();
  if (actualKeys.join("|") !== expectedKeys.join("|")) fail("model artifact keys do not match the frozen terminal artifact contract");
  const modelVersion = object.model_version;
  if (typeof modelVersion !== "string" || !MODEL_VERSION_RE.test(modelVersion)) fail("model_version must use the canonical version format");
  const modelAsOfRaw = object.model_as_of;
  if (typeof modelAsOfRaw !== "string") fail("model_as_of must be an RFC-3339 timestamp");
  const modelAsOf = modelAsOfRaw;
  parseTimestamp(modelAsOf, "model_as_of");
  const intercept = finite(object.intercept, "intercept");
  const calibrationSlope = finite(object.calibration_slope, "calibration_slope");
  const calibrationIntercept = finite(object.calibration_intercept, "calibration_intercept");
  const uncertaintyLogitSd = finite(object.uncertainty_logit_sd, "uncertainty_logit_sd");
  if (Math.abs(intercept) > 1e-12 || Math.abs(calibrationIntercept) > 1e-12) fail("neutral terminal requires intercepts=0");
  if (calibrationSlope <= 0) fail("calibrationSlope must be positive");
  if (uncertaintyLogitSd < 0) fail("uncertaintyLogitSd cannot be negative");
  return {
    modelVersion,
    modelAsOf,
    intercept,
    calibrationSlope,
    calibrationIntercept,
    uncertaintyLogitSd,
    championRoleLogit: mapping(object.champion_role_logit, "champion_role_logit"),
    allySynergyLogit: mapping(object.ally_synergy_logit, "ally_synergy_logit"),
    counterLogit: mapping(object.counter_logit, "counter_logit"),
    artifactSha256,
    authorizesPrediction: options.authorizesPrediction ?? false,
  };
}

function modelLineage(model: TerminalModelArtifact): JsonObject {
  const derived = (label: string) => sha256Canonical({ model_version: model.modelVersion, label });
  return {
    manifest_id: `scryglass:manifest:${model.modelVersion}`,
    training_snapshot_id: "scryglass:training:terminal-development",
    source_snapshot_ids: ["scryglass:source:terminal-development"],
    artifact_sha256: model.artifactSha256,
    source_tree_sha256: derived("source_tree"),
    calibration_sha256: derived("calibration"),
    evaluation_report_sha256: derived("evaluation"),
    code_commit: null,
    environment_lock_sha256: derived("environment"),
    train_cutoff: model.modelAsOf,
  };
}

function sideEffects(
  side: Array<[TerminalRole, string]>,
  model: TerminalModelArtifact,
  sign: number,
): { total: number; ledger: JsonObject[] } {
  let total = 0;
  const ledger: JsonObject[] = [];
  for (const [role, champion] of side) {
    let value = sign * (model.championRoleLogit[`${role}|${champion}`] ?? 0);
    if (value === 0) value = 0;
    total += value;
    ledger.push({ component_type: "champion_role", role, champion_id: champion, signed_logit: value });
  }
  for (let index = 0; index < side.length; index += 1) {
    const [roleA, championA] = side[index];
    for (const [, championB] of side.slice(index + 1)) {
      let value = sign * (model.allySynergyLogit[pair(championA, championB)] ?? 0);
      if (value === 0) value = 0;
      total += value;
      if (value) ledger.push({ component_type: "ally_synergy", role: roleA, champion_ids: [championA, championB], signed_logit: value });
    }
  }
  return { total, ledger };
}

function counterEffect(role: TerminalRole, first: string, second: string, model: TerminalModelArtifact): number {
  const value = model.counterLogit[counterKey(role, first, second)] ?? 0;
  return first <= second ? value : -value;
}

export type TerminalReplayResult = JsonObject & {
  status: "development_only" | "ok";
  score_a: number;
  score_b: number;
  standardized_map_win_probability_a: number;
  standardized_map_win_probability_b: number;
};

function unavailable(draft: TerminalDraft, model: TerminalModelArtifact, reason: string): JsonObject {
  const errorCodes: Record<string, string> = {
    contextual_mode_requires_g1_roster_authority: "source_access_blocked",
    contextual_roster_evidence_stale: "stale_context",
    contextual_model_not_promoted: "model_not_promoted",
    protocol_validation_missing: "missing_required_input",
    model_prediction_authority_not_promoted: "model_not_promoted",
    terminal_input_missing: "missing_required_input",
    model_as_of_after_event_start: "prediction_time_violation",
    source_rights_not_reviewed: "source_access_blocked",
    terminal_contract_context_missing: "missing_required_input",
    terminal_contract_context_conflict: "schema_mismatch",
    terminal_contract_context_stale: "stale_context",
    calibration_not_approved: "calibration_not_approved",
    missing_required_input: "missing_required_input",
  };
  const missingFieldsByReason: Record<string, string[]> = {
    contextual_mode_requires_g1_roster_authority: ["roster_a", "roster_b"],
    contextual_roster_evidence_stale: ["roster_evidence", "source_available_at"],
    contextual_model_not_promoted: ["contextual_fit_model", "player_champion_response", "team_policy_response"],
    model_prediction_authority_not_promoted: [
      "independent_l2_authority",
      "promotion_receipt",
      "reliability_artifact",
      "replay_parity_evidence",
    ],
    protocol_validation_missing: ["protocol_validation"],
    terminal_input_missing: ["actions", "final_assignments"],
  };
  const inputId = terminalInputId(draft);
  const lineage = modelLineage(model);
  const provenance = {
    schema_version: "2.0.0",
    model_version: model.modelVersion,
    as_of: draft.eventStart,
    prediction_id: `scryglass:prediction:${inputId.slice(0, 24)}`,
    mode: "forecast",
    created_at: draft.eventStart,
    event_start: draft.eventStart,
    availability_replayed: true,
    sealed_before_event_start: true,
    input_snapshot_id: `scryglass:input:${inputId.slice(0, 24)}`,
    estimator_id: `scryglass:estimator:${model.modelVersion}`,
    calibration_id: `scryglass:calibration:${model.modelVersion}`,
    required_input_status: reason === "contextual_roster_evidence_stale" || reason === "source_rights_not_reviewed" || reason === "terminal_contract_context_stale" ? "stale" : reason === "contextual_mode_requires_g1_roster_authority" || reason === "terminal_input_missing" || reason === "terminal_contract_context_missing" || reason === "missing_required_input" || reason === "model_prediction_authority_not_promoted" ? "missing" : reason === "model_as_of_after_event_start" || reason === "terminal_contract_context_conflict" ? "conflict" : "stale",
    freshness_checks: [],
    input_conflicts: [],
    fallback_levels: [],
    out_of_distribution_flags: [],
    output_sha256: sha256Canonical({ status: "unavailable", reason, input_id: inputId }),
    immutable: true,
    lineage,
  };
  return {
    schema_version: "2.0.0",
    model_version: model.modelVersion,
    as_of: draft.eventStart,
    season_id: "scryglass:season:development",
    calendar_year: Number(draft.eventStart.slice(0, 4)),
    status: "unavailable",
    identity_mode: draft.mode,
    identity_intentionally_omitted: draft.mode !== "contextual",
    lineage,
    provenance,
    error: {
      code: errorCodes[reason] ?? "invalid_request",
      message: "Draft Score is unavailable for this input or model state.",
      retryable: false,
      missing_fields: missingFieldsByReason[reason] ?? [],
      stale_fields: [],
    },
  };
}

function unavailableWithFields(
  draft: TerminalDraft,
  model: TerminalModelArtifact,
  reason: string,
  fields: string[],
): JsonObject {
  const result = unavailable(draft, model, reason);
  const error = result.error;
  if (error != null && typeof error === "object" && !Array.isArray(error)) {
    (error as JsonObject).missing_fields = [...new Set(fields)];
  }
  return result;
}

function objectValue(value: unknown): JsonObject | null {
  return value != null && typeof value === "object" && !Array.isArray(value)
    ? value as JsonObject
    : null;
}

function arrayValue(value: unknown): unknown[] | null {
  return Array.isArray(value) ? value : null;
}

function timestampOrNull(value: unknown, field: string): number | null {
  if (typeof value !== "string") return null;
  try {
    return parseTimestamp(value, field);
  } catch {
    return null;
  }
}

const TERMINAL_CONTRACT_REQUIRED_FIELDS = [
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
] as const;

/**
 * Assemble the schema-shaped response only after the complete match contract
 * has passed. Numeric mechanics alone are deliberately insufficient here.
 */
export function renderTerminalContract(
  draft: TerminalDraft,
  model: TerminalModelArtifact,
  options: {
    contract?: TerminalContract;
    promotionReceipt?: TerminalPromotionReceipt;
    promotionBindings?: TerminalPromotionBindings;
    protocolValidation?: TerminalProtocolValidation;
  } = {},
): JsonObject {
  const contractProtocol = objectValue(options.contract?.protocol_validation) as TerminalProtocolValidation | null;
  const result = scoreTerminalDraft(draft, model, {
    promotionReceipt: options.promotionReceipt,
    promotionBindings: options.promotionBindings,
    protocolValidation: options.protocolValidation ?? contractProtocol ?? undefined,
  });
  if (result.status !== "ok") return result;
  if (!draft.actions.length) return unavailableWithFields(draft, model, "terminal_input_missing", ["actions", "final_assignments"]);

  const contract = options.contract;
  if (contract == null) {
    return unavailableWithFields(draft, model, "terminal_contract_context_missing", [...TERMINAL_CONTRACT_REQUIRED_FIELDS]);
  }
  const missing = TERMINAL_CONTRACT_REQUIRED_FIELDS.filter((field) => !(field in contract));
  if (missing.length) return unavailableWithFields(draft, model, "terminal_contract_context_missing", missing);

  const sideMapping = objectValue(contract.side_mapping);
  const sourceRecord = objectValue(contract.source_record);
  const protocolValidation = objectValue(contract.protocol_validation);
  const evidence = objectValue(contract.evidence);
  const reliability = objectValue(contract.reliability);
  const provenance = objectValue(contract.provenance);
  const roleConstraintRevisions = arrayValue(contract.role_constraint_revisions);
  const assignmentRevisions = arrayValue(contract.assignment_revisions);
  if (!sideMapping || !sourceRecord || !protocolValidation || !evidence || !reliability || !provenance || !roleConstraintRevisions || !assignmentRevisions) {
    return unavailableWithFields(draft, model, "terminal_contract_context_missing", [
      ...(!sideMapping ? ["side_mapping"] : []),
      ...(!sourceRecord ? ["source_record"] : []),
      ...(!protocolValidation ? ["protocol_validation"] : []),
      ...(!evidence ? ["evidence"] : []),
      ...(!reliability ? ["reliability"] : []),
      ...(!provenance ? ["provenance"] : []),
      ...(!roleConstraintRevisions ? ["role_constraint_revisions"] : []),
      ...(!assignmentRevisions ? ["assignment_revisions"] : []),
    ]);
  }

  const eventStart = parseTimestamp(draft.eventStart, "eventStart");
  const modelAsOf = parseTimestamp(model.modelAsOf, "modelAsOf");
  const sourceAvailableAt = timestampOrNull(sourceRecord.available_at, "source_record.available_at");
  const draftSourceAvailableAt = parseTimestamp(draft.sourceAvailableAt, "sourceAvailableAt");
  const sideMappingAvailableAt = timestampOrNull(sideMapping.available_at, "side_mapping.available_at");
  if (sourceRecord.source_record_id !== draft.sourceRecordId) {
    return unavailableWithFields(draft, model, "terminal_contract_context_conflict", ["source_record.source_record_id"]);
  }
  if (sourceAvailableAt == null || sourceAvailableAt !== draftSourceAvailableAt) {
    return unavailableWithFields(draft, model, "terminal_contract_context_conflict", ["source_record.available_at"]);
  }
  if (sourceAvailableAt >= eventStart) {
    return unavailableWithFields(draft, model, "terminal_contract_context_stale", ["source_record.available_at"]);
  }
  if (sideMappingAvailableAt == null || sideMappingAvailableAt >= eventStart) {
    return unavailableWithFields(draft, model, "terminal_contract_context_stale", ["side_mapping.available_at"]);
  }

  const protocolAvailableAt = timestampOrNull(protocolValidation.available_at, "protocol_validation.available_at");
  if (
    protocolValidation.status !== "validated" ||
    protocolValidation.action_order_verified !== true ||
    protocolValidation.pick_ban_counts_verified !== true ||
    protocolValidation.canonical_side_mapping_verified !== true ||
    typeof protocolValidation.validator_id !== "string" ||
    protocolValidation.validator_id.length === 0 ||
    typeof protocolValidation.validator_sha256 !== "string" ||
    !SHA256_RE.test(protocolValidation.validator_sha256) ||
    protocolAvailableAt == null ||
    protocolAvailableAt >= eventStart
  ) {
    return unavailableWithFields(draft, model, "protocol_validation_missing", ["protocol_validation"]);
  }
  if (reliability.probability_wording_approved !== true || reliability.validation_gate_passed !== true) {
    return unavailableWithFields(draft, model, "calibration_not_approved", ["reliability"]);
  }

  const transform = objectValue(provenance.probability_transform);
  if (
    !transform ||
    transform.probability_domain !== "open_0_1" ||
    transform.monotonicity !== "nondecreasing" ||
    transform.complement_symmetry_verified !== true ||
    transform.open_support_verified !== true ||
    typeof transform.transform_sha256 !== "string" ||
    !SHA256_RE.test(transform.transform_sha256) ||
    typeof transform.transform_proof_sha256 !== "string" ||
    !SHA256_RE.test(transform.transform_proof_sha256)
  ) {
    return unavailableWithFields(draft, model, "calibration_not_approved", ["provenance.probability_transform"]);
  }
  if (provenance.input_conflicts != null && (!Array.isArray(provenance.input_conflicts) || provenance.input_conflicts.length > 0)) {
    return unavailableWithFields(draft, model, "terminal_contract_context_conflict", ["provenance.input_conflicts"]);
  }
  if (provenance.required_input_status !== "complete") {
    return unavailableWithFields(draft, model, "missing_required_input", ["provenance.required_input_status"]);
  }

  const provenanceAsOf = timestampOrNull(provenance.as_of, "provenance.as_of");
  const createdAt = timestampOrNull(provenance.created_at, "provenance.created_at");
  const provenanceEventStart = timestampOrNull(provenance.event_start, "provenance.event_start");
  if (provenanceAsOf == null || createdAt == null || provenanceEventStart == null) {
    return unavailableWithFields(draft, model, "missing_required_input", [
      "provenance.as_of",
      "provenance.created_at",
      "provenance.event_start",
    ]);
  }
  if (
    provenance.mode !== "forecast" ||
    provenance.sealed_before_event_start !== true ||
    provenance.availability_replayed !== true
  ) {
    return unavailableWithFields(draft, model, "missing_required_input", [
      "provenance.mode",
      "provenance.sealed_before_event_start",
      "provenance.availability_replayed",
    ]);
  }
  if (provenanceAsOf !== modelAsOf || provenanceAsOf >= eventStart) {
    return unavailableWithFields(draft, model, "terminal_contract_context_conflict", ["provenance.as_of"]);
  }
  if (createdAt >= eventStart) {
    return unavailableWithFields(draft, model, "terminal_contract_context_stale", ["provenance.created_at"]);
  }
  if (provenanceEventStart !== eventStart) {
    return unavailableWithFields(draft, model, "terminal_contract_context_conflict", ["provenance.event_start"]);
  }

  const rawLedger = arrayValue(result.ledger);
  if (!rawLedger) fail("terminal ledger must be an array");
  const componentTypes: Record<string, string> = {
    champion_role: "champion_role_main",
    ally_synergy: "ally_pair",
    counter: "enemy_pair",
  };
  const inputId = terminalInputId(draft);
  const canonicalLedger = rawLedger.map((rawEntry, index) => {
    const entry = objectValue(rawEntry);
    if (!entry) fail("terminal ledger entry must be an object");
    const rawParticipants = entry.champion_ids ?? (entry.champion_id == null ? null : [entry.champion_id]);
    if (!Array.isArray(rawParticipants) || rawParticipants.some((value) => typeof value !== "string" || value.length === 0)) {
      fail("terminal ledger entry has no participant ids");
    }
    const componentType = typeof entry.component_type === "string" ? componentTypes[entry.component_type] : undefined;
    if (!componentType) fail("terminal ledger contains an unknown component type");
    return {
      entry_id: `scryglass:ledger:${inputId.slice(0, 24)}:${index + 1}`,
      component_type: componentType,
      signed_logit: finite(entry.signed_logit, `ledger[${index}].signed_logit`),
      participant_ids: [...new Set(rawParticipants)],
      allocation_method: "direct",
    };
  });

  const outputProvenance: JsonObject = {
    ...provenance,
    schema_version: "2.0.0",
    model_version: model.modelVersion,
    as_of: model.modelAsOf,
    prediction_id: `scryglass:prediction:${inputId.slice(0, 24)}`,
    event_start: draft.eventStart,
    input_snapshot_id: `scryglass:input:${inputId.slice(0, 24)}`,
    estimator_id: `scryglass:estimator:${model.modelVersion}`,
    calibration_id: contract.calibration_id,
    immutable: true,
    lineage: modelLineage(model),
  };
  return {
    schema_version: "2.0.0",
    model_version: model.modelVersion,
    as_of: model.modelAsOf,
    season_id: contract.season_id,
    calendar_year: Number(draft.eventStart.slice(0, 4)),
    status: "ok",
    identity_mode: draft.mode,
    identity_intentionally_omitted: draft.mode !== "contextual",
    draft_state_id: `scryglass:draft:${inputId}`,
    event_id: contract.event_id ?? null,
    competition_scope_id: contract.competition_scope_id,
    competition_scope_kind: contract.competition_scope_kind,
    patch_id: contract.patch_id,
    protocol_id: contract.protocol_id,
    side_mapping: { ...sideMapping },
    source_record: { ...sourceRecord },
    actions: draft.actions,
    final_assignments: draft.finalAssignments,
    role_constraint_revisions: roleConstraintRevisions,
    assignment_revisions: assignmentRevisions,
    score_a: result.score_a,
    score_b: result.score_b,
    standardized_map_win_probability_a: result.standardized_map_win_probability_a,
    interval_95: result.interval_95,
    uncalibrated_logit_a: result.uncalibrated_logit_a,
    calibration_id: contract.calibration_id,
    evidence: { ...evidence },
    reliability: { ...reliability },
    ledger: canonicalLedger,
    ledger_logit_sum: result.ledger_logit_sum,
    reconciliation_tolerance: 1e-12,
    literal_interpretation: "Out of 100, the model-estimated map-win probability for side A under this draft after equalizing baseline roster and league strength and neutralizing in-game side advantage.",
    lineage: modelLineage(model),
    provenance: outputProvenance,
  };
}

export function scoreTerminalDraft(
  draft: TerminalDraft,
  model: TerminalModelArtifact,
  options: { development?: boolean; promotionReceipt?: TerminalPromotionReceipt; promotionBindings?: TerminalPromotionBindings; protocolValidation?: TerminalProtocolValidation } = {},
): JsonObject | TerminalReplayResult {
  const development = options.development ?? false;
  if (draft.mode !== "neutral") {
    if (draft.rosterEvidence == null) return unavailable(draft, model, "contextual_mode_requires_g1_roster_authority");
    if (!g1RosterEvidenceAvailable(draft.rosterEvidence, draft.eventStart)) return unavailable(draft, model, "contextual_roster_evidence_stale");
    return unavailable(draft, model, "contextual_model_not_promoted");
  }
  if (!development && !terminalPromotionAuthorized(model, options.promotionReceipt, options.promotionBindings)) return unavailable(draft, model, "model_prediction_authority_not_promoted");
  if (!development && draft.actions.length === 0) return unavailable(draft, model, "terminal_input_missing");
  if (!development && !protocolValidationAvailable(options.protocolValidation, draft.eventStart)) return unavailable(draft, model, "protocol_validation_missing");
  if (parseTimestamp(model.modelAsOf, "modelAsOf") >= parseTimestamp(draft.eventStart, "eventStart")) {
    return unavailable(draft, model, "model_as_of_after_event_start");
  }
  if (draft.sourceRightsStatus !== "reviewed" && !development) return unavailable(draft, model, "source_rights_not_reviewed");

  const sideA = sideEffects(draft.sideA, model, 1);
  const sideB = sideEffects(draft.sideB, model, -1);
  let counterTotal = 0;
  const counterLedger: JsonObject[] = [];
  for (let index = 0; index < draft.sideA.length; index += 1) {
    const [roleA, championA] = draft.sideA[index];
    const [roleB, championB] = draft.sideB[index];
    if (roleA !== roleB) fail("side roles must be aligned in canonical role order");
    const value = counterEffect(roleA, championA, championB, model);
    counterTotal += value;
    if (value) counterLedger.push({ component_type: "counter", role: roleA, champion_ids: [championA, championB], signed_logit: value });
  }
  const rawLogit = sideA.total + sideB.total + counterTotal;
  const calibratedLogit = model.calibrationIntercept + model.calibrationSlope * rawLogit;
  const probabilityA = sigmoid(calibratedLogit);
  const probabilityB = 1 - probabilityA;
  const interval = {
    lower: sigmoid(calibratedLogit - Z95 * model.uncertaintyLogitSd),
    upper: sigmoid(calibratedLogit + Z95 * model.uncertaintyLogitSd),
    level: 0.95,
  };
  const ledger = [...sideA.ledger, ...sideB.ledger, ...counterLedger];
  const ledgerSum = ledger.reduce((sum, entry) => sum + Number(entry.signed_logit), 0);
  if (Math.abs(ledgerSum - rawLogit) > 1e-12) fail("terminal ledger does not reconcile");
  return {
    status: development ? "development_only" : "ok",
    input_id: terminalInputId(draft),
    lineage_id: sha256Canonical({ model_version: model.modelVersion, model_as_of: model.modelAsOf, artifact_sha256: model.artifactSha256 }),
    claim_ceiling: { causal: false, recommendation: false, betting: false },
    score_a: 100 * probabilityA,
    score_b: 100 * probabilityB,
    standardized_map_win_probability_a: probabilityA,
    standardized_map_win_probability_b: probabilityB,
    interval_95: interval,
    uncalibrated_logit_a: rawLogit,
    ledger,
    ledger_logit_sum: ledgerSum,
    model_version: model.modelVersion,
  };
}

export function terminalPromotionAuthorized(
  model: TerminalModelArtifact,
  receipt: TerminalPromotionReceipt | undefined,
  bindings: TerminalPromotionBindings | undefined,
): boolean {
  const hashes = receipt == null
    ? []
    : [
        receipt.artifact_sha256,
        receipt.l2_contract_sha256,
        receipt.development_evaluation_sha256,
        receipt.candidate_registry_sha256,
        receipt.calibration_transform_sha256,
        receipt.reliability_artifact_sha256,
        receipt.replay_parity_evidence_sha256,
        receipt.independent_authority_record_sha256,
      ];
  const bindingHashes = bindings == null
    ? []
    : [
        bindings.development_evaluation_sha256,
        bindings.candidate_registry_sha256,
        bindings.l2_contract_sha256,
        bindings.calibration_transform_sha256,
        bindings.reliability_artifact_sha256,
        bindings.replay_parity_evidence_sha256,
        bindings.independent_authority_record_sha256,
      ];
  let issuedAtValid = false;
  if (typeof receipt?.issued_at === "string") {
    try {
      issuedAtValid = parseTimestamp(receipt.issued_at, "promotionReceipt.issued_at") >= 0;
    } catch {
      issuedAtValid = false;
    }
  }
  return Boolean(
    model.authorizesPrediction &&
      receipt?.schema_version === "draft-terminal-promotion-receipt-v1" &&
      receipt?.status === "approved" &&
      receipt.model_version === model.modelVersion &&
      receipt.artifact_sha256 === model.artifactSha256 &&
      hashes.every((hash) => SHA256_RE.test(hash)) &&
      bindingHashes.every((hash) => SHA256_RE.test(hash)) &&
      bindings != null &&
      receipt.development_evaluation_sha256 === bindings.development_evaluation_sha256 &&
      receipt.candidate_registry_sha256 === bindings.candidate_registry_sha256 &&
      receipt.l2_contract_sha256 === bindings.l2_contract_sha256 &&
      receipt.calibration_transform_sha256 === bindings.calibration_transform_sha256 &&
      receipt.reliability_artifact_sha256 === bindings.reliability_artifact_sha256 &&
      receipt.replay_parity_evidence_sha256 === bindings.replay_parity_evidence_sha256 &&
      receipt.independent_authority_record_sha256 === bindings.independent_authority_record_sha256 &&
      receipt.independent_l2_authority === true &&
      receipt.final_temporal_holdout_sealed === true &&
      receipt.public_probability_authorized === true &&
      receipt.replay_parity_verified === true &&
      receipt.reliability_gate_passed === true &&
      (receipt.contextual_g1_authority === "not_applicable" || receipt.contextual_g1_authority === "approved") &&
      typeof receipt.authority_record_id === "string" &&
      receipt.authority_record_id.trim().length > 0 &&
      receipt.authority_record_id === bindings.authority_record_id &&
      issuedAtValid,
  );
}
