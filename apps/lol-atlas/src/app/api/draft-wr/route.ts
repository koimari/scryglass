import { NextResponse } from "next/server";
import { compositionRuntimeMetadata } from "@/lib/draftComposition";
import {
  draftProbabilityGateEvidence,
  draftRuntimeBindingEvidence,
  type DraftRuntimeBindingEvidence,
} from "@/lib/modelValidation";
import {
  exposePatchContracts,
  patchContractFromSource,
} from "@/lib/patch";
import { readPackJson, readPackManifest } from "@/lib/serverPack";

export const runtime = "nodejs";

export function draftWrWithheldPayload(
  evidence: unknown = null,
  binding: DraftRuntimeBindingEvidence | null = null,
) {
  const verifiedEvidence = draftProbabilityGateEvidence(evidence);
  const metadata = compositionRuntimeMetadata();
  const currentPatch = patchContractFromSource(
    metadata?.latest_observed_patch,
  );
  return {
    error:
      "Draft win probability is withheld because the composition probability pipeline failed its chronological promotion gate.",
    status: "withheld_failed_chronological_gate",
    gate_id: "draft-probability-chronological-2026-07-27",
    estimand:
      "pre-map blue win probability from ten uniquely role-assigned champions",
    evidence: verifiedEvidence && binding
      ? {
          artifact_schema_version: verifiedEvidence.schemaVersion,
          artifact_created_utc: verifiedEvidence.artifactCreatedUtc,
          artifact_sha256: verifiedEvidence.artifactSha256,
          model_code_sha256: verifiedEvidence.modelCodeSha256,
          training_population_sha256:
            verifiedEvidence.trainingPopulationSha256,
          population_maps: verifiedEvidence.populationMaps,
          final_test_maps: verifiedEvidence.finalTestMaps,
          final_test_start: verifiedEvidence.finalTestStart,
          final_test_end: verifiedEvidence.finalTestEnd,
          composition_log_loss: verifiedEvidence.compositionLogLoss,
          overall_base_rate_log_loss: verifiedEvidence.overallBaseRateLogLoss,
          composition_brier: verifiedEvidence.compositionBrier,
          overall_base_rate_brier: verifiedEvidence.overallBaseRateBrier,
          decision: "withheld",
          active_pack_id: binding.packId,
          runtime_binding_status: binding.status,
        }
      : null,
    evidence_status: verifiedEvidence && binding
      ? "verified_immutable_pack_artifact"
      : "unavailable_or_invalid",
    patch_selection: {
      request_field: "public_patch",
      contract: "public 25.x -> source 15.x; public 26.x -> source 16.x",
      current_patch: currentPatch,
    },
    candidate_model_metadata: metadata
      ? exposePatchContracts(metadata)
      : null,
    served_probability_model: null,
    note:
      "The interactive Sandbox exposes an experimental policy value, not a calibrated win probability.",
  } as const;
}

async function currentGateEvidence(): Promise<{
  artifact: unknown;
  binding: DraftRuntimeBindingEvidence;
} | null> {
  try {
    const manifest = await readPackManifest();
    const artifact = await readPackJson(
      manifest,
      "models/model_validation_2026-07-27.json",
    );
    const binding = draftRuntimeBindingEvidence(
      manifest,
      artifact,
      compositionRuntimeMetadata(),
    );
    return binding ? { artifact, binding } : null;
  } catch {
    return null;
  }
}

export async function GET() {
  const current = await currentGateEvidence();
  return NextResponse.json(
    draftWrWithheldPayload(current?.artifact, current?.binding ?? null),
    { status: 503 },
  );
}

export async function POST() {
  const current = await currentGateEvidence();
  return NextResponse.json(
    draftWrWithheldPayload(current?.artifact, current?.binding ?? null),
    { status: 503 },
  );
}
