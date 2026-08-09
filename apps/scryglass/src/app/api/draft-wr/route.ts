import { NextResponse } from "next/server";
import {
  DRAFT_UNAVAILABLE_RESPONSE,
  publicPredictiveDraftsEnabled,
} from "@/lib/publicDraftGate";
import {
  scoreCanonicalNeutralDraft,
  type CanonicalNeutralDraftRequest,
} from "@/lib/draftTerminalServer";

export const runtime = "nodejs";

const unavailable = () =>
  NextResponse.json(
    DRAFT_UNAVAILABLE_RESPONSE,
    { status: 503 },
  );

type Body = {
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
};

export async function POST(req: Request) {
  if (!publicPredictiveDraftsEnabled()) return unavailable();
  try {
    const body = (await req.json()) as Body;
    const result = scoreCanonicalNeutralDraft({
      sideA: body.side_a as CanonicalNeutralDraftRequest["sideA"],
      sideB: body.side_b as CanonicalNeutralDraftRequest["sideB"],
      eventStart: body.event_start as string,
      sourceAvailableAt: body.source_available_at as string,
      sourceRecordId: body.source_record_id as string,
      sourcePayloadSha256: body.source_payload_sha256 as string,
      sourceRightsStatus: body.source_rights_status as "reviewed" | "unknown",
      actions: body.actions as CanonicalNeutralDraftRequest["actions"],
      finalAssignments: body.final_assignments as CanonicalNeutralDraftRequest["finalAssignments"],
      protocolValidation: body.protocol_validation as CanonicalNeutralDraftRequest["protocolValidation"],
      contract: body.contract,
    });
    return NextResponse.json(result, { status: result.status === "unavailable" ? 503 : 200 });
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : String(e) },
      { status: 400 },
    );
  }
}
