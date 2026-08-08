import { NextResponse } from "next/server";
import {
  canonicalPublicPredictiveDraftsEnabled,
  DRAFT_UNAVAILABLE_RESPONSE,
  DRAFT_INTERNAL_ERROR_RESPONSE,
} from "@/lib/publicDraftGate";
import {
  scoreCanonicalTerminalDraft,
  type CanonicalTerminalDraftRequest,
} from "@/lib/draftTerminalServer";

export const runtime = "nodejs";

const unavailable = () =>
  NextResponse.json(DRAFT_UNAVAILABLE_RESPONSE, { status: 503 });

function hasBaseDraftInput(body: Body): boolean {
  return (
    body.side_a != null &&
    body.side_b != null &&
    typeof body.event_start === "string" &&
    typeof body.source_available_at === "string" &&
    typeof body.source_record_id === "string" &&
    typeof body.source_payload_sha256 === "string" &&
    body.source_rights_status != null
  );
}

type Body = {
  identity_mode?: "neutral" | "contextual";
  side_a?: Record<string, string>;
  side_b?: Record<string, string>;
  event_start?: string;
  source_available_at?: string;
  source_record_id?: string;
  source_payload_sha256?: string;
  source_rights_status?: "reviewed" | "unknown";
  actions?: unknown[];
  final_assignments?: unknown[];
  protocol_validation?: Record<string, unknown>;
  contract?: Record<string, unknown>;
  roster_evidence?: Record<string, unknown>;
};

/** Canonical versioned terminal endpoint; it remains fail-closed until promotion. */
export async function POST(req: Request) {
  try {
    const body = (await req.json()) as Body;
    if (!canonicalPublicPredictiveDraftsEnabled() && !hasBaseDraftInput(body)) return unavailable();
    const result = scoreCanonicalTerminalDraft({
      sideA: body.side_a as CanonicalTerminalDraftRequest["sideA"],
      sideB: body.side_b as CanonicalTerminalDraftRequest["sideB"],
      eventStart: body.event_start as string,
      sourceAvailableAt: body.source_available_at as string,
      sourceRecordId: body.source_record_id as string,
      sourcePayloadSha256: body.source_payload_sha256 as string,
      sourceRightsStatus: body.source_rights_status as "reviewed" | "unknown",
      actions: body.actions as CanonicalTerminalDraftRequest["actions"],
      finalAssignments: body.final_assignments as CanonicalTerminalDraftRequest["finalAssignments"],
      protocolValidation: body.protocol_validation as CanonicalTerminalDraftRequest["protocolValidation"],
      contract: body.contract,
      mode: body.identity_mode,
      rosterEvidence: body.roster_evidence as CanonicalTerminalDraftRequest["rosterEvidence"],
    });
    return NextResponse.json(result, { status: result.status === "unavailable" ? 503 : 200 });
  } catch (error) {
    void error;
    return NextResponse.json(DRAFT_INTERNAL_ERROR_RESPONSE, { status: 503 });
  }
}
