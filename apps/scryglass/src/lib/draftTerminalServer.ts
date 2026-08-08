import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import {
  loadTerminalModelArtifact,
  renderTerminalContract,
  terminalDraftFromSides,
  terminalPromotionAuthorized,
  validateTerminalL2AuthorityRecord,
  type TerminalAction,
  type TerminalAssignment,
  type TerminalContract,
  type TerminalG1RosterEvidence,
  type TerminalL2AuthorityRecord,
  type TerminalModelArtifact,
  type TerminalPromotionBindings,
  type TerminalPromotionReceipt,
  type TerminalProtocolValidation,
  type TerminalSide,
} from "./draftTerminalScore";

const REPO_ROOT = path.resolve(process.cwd(), "../..");
const MODEL_LOCATOR = "data/lol/v2/models/draft-terminal/terminal-model-neutral-development-v1.json";
const RECEIPT_LOCATOR = "data/lol/v2/models/draft-terminal/draft-terminal-promotion-receipt.json";
const AUTHORITY_LOCATOR = "data/lol/v2/models/draft-terminal/draft-terminal-l2-authority-record.json";
const EVALUATION_LOCATOR = "data/lol/v2/models/draft-terminal/development-evaluation-summary.json";
const REGISTRY_LOCATOR = "data/lol/v2/models/draft-terminal/draft-terminal-candidate-registry.json";
const CONTRACT_LOCATOR = "data/lol/v2/models/draft-terminal/draft-terminal-l2-evaluation-contract.json";

export type CanonicalTerminalDraftRequest = {
  sideA: TerminalSide;
  sideB: TerminalSide;
  eventStart: string;
  sourceAvailableAt: string;
  sourceRecordId: string;
  sourcePayloadSha256: string;
  sourceRightsStatus: "reviewed" | "unknown";
  actions?: TerminalAction[];
  finalAssignments?: TerminalAssignment[];
  protocolValidation?: TerminalProtocolValidation;
  contract?: TerminalContract;
  mode?: "neutral" | "contextual";
  rosterEvidence?: TerminalG1RosterEvidence;
};

export type CanonicalNeutralDraftRequest = Omit<CanonicalTerminalDraftRequest, "mode" | "rosterEvidence">;

function sha256(raw: Uint8Array): string {
  return createHash("sha256").update(raw).digest("hex");
}

function absolute(locator: string): string {
  return path.join(REPO_ROOT, locator);
}

function promotionArtifactsDeclarePublicEligibility(registryRaw: Uint8Array, evaluationRaw: Uint8Array): boolean {
  void registryRaw;
  void evaluationRaw;
  // Public Scryglass is non-betting. Private component authority is replayed
  // only by the Python v2 boundary and can never promote this public route.
  return false;
}

function loadServingContext(): {
  model: TerminalModelArtifact;
  promotionReceipt?: TerminalPromotionReceipt;
  promotionBindings?: TerminalPromotionBindings;
} {
  const modelRaw = readFileSync(absolute(MODEL_LOCATOR));
  const modelHash = sha256(modelRaw);
  const receiptPath = absolute(RECEIPT_LOCATOR);
  const authorityPath = absolute(AUTHORITY_LOCATOR);
  const evaluationPath = absolute(EVALUATION_LOCATOR);
  const registryPath = absolute(REGISTRY_LOCATOR);
  const contractPath = absolute(CONTRACT_LOCATOR);
  const allPromotionFilesPresent = [receiptPath, authorityPath, evaluationPath, registryPath, contractPath].every(existsSync);
  if (!allPromotionFilesPresent) {
    return { model: loadTerminalModelArtifact(modelRaw, { expectedArtifactSha256: modelHash }) };
  }
  const evaluationSha256 = sha256(readFileSync(evaluationPath));
  const candidateRegistrySha256 = sha256(readFileSync(registryPath));
  const l2ContractSha256 = sha256(readFileSync(contractPath));
  const evaluationRaw = readFileSync(evaluationPath);
  const registryRaw = readFileSync(registryPath);
  if (!promotionArtifactsDeclarePublicEligibility(registryRaw, evaluationRaw)) {
    return { model: loadTerminalModelArtifact(modelRaw, { expectedArtifactSha256: modelHash }) };
  }
  const authorityRaw = readFileSync(authorityPath);
  try {
    const authority = validateTerminalL2AuthorityRecord(
      JSON.parse(authorityRaw.toString("utf8")) as TerminalL2AuthorityRecord,
      {
        model_artifact_sha256: modelHash,
        development_evaluation_sha256: evaluationSha256,
        candidate_registry_sha256: candidateRegistrySha256,
        l2_contract_sha256: l2ContractSha256,
      },
    );
    const receipt = JSON.parse(readFileSync(receiptPath, "utf8")) as TerminalPromotionReceipt;
    if (receipt.authority_record_id !== authority.authority_record_id) {
      throw new Error("promotion receipt authority_record_id does not match the independent authority record");
    }
    const model = loadTerminalModelArtifact(modelRaw, { expectedArtifactSha256: modelHash, authorizesPrediction: true });
    return {
      model,
      promotionReceipt: receipt,
      promotionBindings: {
        development_evaluation_sha256: evaluationSha256,
        candidate_registry_sha256: candidateRegistrySha256,
        l2_contract_sha256: l2ContractSha256,
        calibration_transform_sha256: authority.calibration_transform_sha256,
        reliability_artifact_sha256: authority.reliability_artifact_sha256,
        replay_parity_evidence_sha256: authority.replay_parity_evidence_sha256,
        independent_authority_record_sha256: sha256(authorityRaw),
        authority_record_id: authority.authority_record_id,
      },
    };
  } catch {
    return { model: loadTerminalModelArtifact(modelRaw, { expectedArtifactSha256: modelHash }) };
  }
}

/**
 * The canonical public route may open only when the complete promotion bundle
 * is present and the same receipt/bindings check used for scoring succeeds.
 * Missing, malformed, or unapproved evidence returns false.
 */
export function canonicalDraftServingAuthorityAvailable(): boolean {
  try {
    const context = loadServingContext();
    return Boolean(
      context.model.authorizesPrediction &&
      context.promotionReceipt &&
      context.promotionBindings &&
      terminalPromotionAuthorized(context.model, context.promotionReceipt, context.promotionBindings),
    );
  } catch {
    return false;
  }
}

export function scoreCanonicalTerminalDraft(request: CanonicalTerminalDraftRequest) {
  const context = loadServingContext();
  const draft = terminalDraftFromSides({
    sideA: request.sideA,
    sideB: request.sideB,
    eventStart: request.eventStart,
    sourceAvailableAt: request.sourceAvailableAt,
    sourceRecordId: request.sourceRecordId,
    sourcePayloadSha256: request.sourcePayloadSha256,
    sourceRightsStatus: request.sourceRightsStatus,
    mode: request.mode ?? "neutral",
    rosterEvidence: request.rosterEvidence,
    actions: request.actions,
    finalAssignments: request.finalAssignments,
  });
  return renderTerminalContract(draft, context.model, {
    promotionReceipt: context.promotionReceipt,
    promotionBindings: context.promotionBindings,
    protocolValidation: request.protocolValidation,
    contract: request.contract,
  });
}

export function scoreCanonicalNeutralDraft(request: CanonicalNeutralDraftRequest) {
  return scoreCanonicalTerminalDraft({ ...request, mode: "neutral" });
}
