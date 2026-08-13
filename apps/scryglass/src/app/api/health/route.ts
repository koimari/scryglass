import { NextResponse } from "next/server";
import { validDiagnosticSecret } from "@/lib/dataPublish";
import { sameTimestamp } from "@/lib/health";
import {
  readPublicRefreshHealth,
  readPublicTierList,
  readRemotePackManifest,
} from "@/lib/serverPack";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

async function readTierState(manifest: Awaited<ReturnType<typeof readRemotePackManifest>>) {
  if (manifest.tier?.status === "available") {
    return {
      status: "available",
      as_of: manifest.tier.as_of ?? null,
    };
  }
  const payload = await readPublicTierList<Record<string, unknown>>();
  return {
    status: payload.status === "available" ? "available" : "unavailable",
    as_of: typeof payload.as_of === "string" ? payload.as_of : null,
  };
}

export async function GET(request: Request) {
  try {
    const [manifest, refresh] = await Promise.all([
      readRemotePackManifest(),
      readPublicRefreshHealth(),
    ]);
    let tier = { status: "unavailable", as_of: null as string | null };
    try {
      tier = await readTierState(manifest);
    } catch {
      // Ratings remain useful while a later tier authority cycle is pending.
    }
    const releaseAligned = !refresh?.active_release_id
      || refresh.active_release_id === manifest.pack_id
      || refresh.refresh_status === "running";
    const sourceAsOf = manifest.ratings?.source_as_of ?? null;
    const sourceAligned = !refresh?.source_as_of
      || sameTimestamp(refresh.source_as_of, sourceAsOf)
      || refresh.refresh_status === "running";
    const status = refresh
      && refresh.status === "ok"
      && tier.status === "available"
      && releaseAligned
      && sourceAligned
      && !refresh.stale
      ? "ok"
      : "partial";
    const checkedAt = new Date().toISOString();
    const publicHealth = {
      status,
      checked_at: checkedAt,
      last_success_at: refresh?.last_success_at ?? null,
      stale: refresh?.stale ?? false,
    };
    const diagnostic = validDiagnosticSecret(
      request.headers.get("authorization"),
      process.env.SCRYGLASS_DIAGNOSTIC_TOKEN,
      process.env.SCRYGLASS_DATA_PUBLISH_TOKEN,
    );
    return NextResponse.json(
      diagnostic
        ? {
            ...publicHealth,
            diagnostics: {
              release_id: manifest.pack_id,
              pack_created_utc: manifest.created_utc ?? null,
              source_as_of: sourceAsOf,
              tier,
              refresh_status: refresh?.refresh_status ?? "unknown",
              worker_commit: refresh?.worker_commit ?? null,
              run_id: refresh?.last_run_id ?? null,
            },
          }
        : publicHealth,
      { headers: { "Cache-Control": "private, no-store" } },
    );
  } catch {
    return NextResponse.json(
      { status: "error", checked_at: new Date().toISOString() },
      { status: 503, headers: { "Cache-Control": "private, no-store" } },
    );
  }
}
