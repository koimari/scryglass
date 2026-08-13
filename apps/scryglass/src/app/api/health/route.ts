import { NextResponse } from "next/server";
import { validDiagnosticSecret } from "@/lib/dataPublish";
import { sameTimestamp } from "@/lib/health";
import {
  readPublicRefreshHealth,
  readPrivateRefreshHealth,
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
    const diagnostic = validDiagnosticSecret(
      request.headers.get("authorization"),
      process.env.SCRYGLASS_DIAGNOSTIC_TOKEN,
    );
    const [manifest, refresh, privateRefresh] = await Promise.all([
      readRemotePackManifest(),
      readPublicRefreshHealth(),
      diagnostic ? readPrivateRefreshHealth() : Promise.resolve(null),
    ]);
    let tier = { status: "unavailable", as_of: null as string | null };
    try {
      tier = await readTierState(manifest);
    } catch {
      // Ratings remain useful while a later tier authority cycle is pending.
    }
    const releaseAligned = refresh?.active_release_id === manifest.pack_id;
    const sourceAsOf = manifest.ratings?.source_as_of ?? null;
    const sourceAligned = !refresh?.source_as_of
      || sameTimestamp(refresh.source_as_of, sourceAsOf);
    const status = refresh
      && refresh.status === "ok"
      && refresh.refresh_status === "idle"
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
    return NextResponse.json(
      diagnostic
        ? {
            ...publicHealth,
            diagnostics: {
              release_id: manifest.pack_id,
              pack_created_utc: manifest.created_utc ?? null,
              source_as_of: sourceAsOf,
              tier,
              refresh_status: privateRefresh?.refresh_status ?? "unknown",
              worker_commit: privateRefresh?.worker_commit ?? null,
              run_id: privateRefresh?.last_run_id ?? null,
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
