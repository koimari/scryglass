import { NextResponse } from "next/server";
import { readPublicTierList, readRemotePackManifest } from "@/lib/serverPack";

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

export async function GET() {
  try {
    const manifest = await readRemotePackManifest();
    let tier = { status: "unavailable", as_of: null as string | null };
    try {
      tier = await readTierState(manifest);
    } catch {
      // Ratings remain useful while a later tier authority cycle is pending.
    }
    return NextResponse.json({
      status: tier.status === "available" ? "ok" : "partial",
      pack_id: manifest.pack_id,
      pack_created_utc: manifest.created_utc ?? null,
      source_as_of: manifest.ratings?.source_as_of ?? null,
      tier,
    });
  } catch {
    return NextResponse.json({ status: "error" }, { status: 503 });
  }
}
