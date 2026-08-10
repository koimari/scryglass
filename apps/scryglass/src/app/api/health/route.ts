import { NextResponse } from "next/server";
import { readRemotePackManifest } from "@/lib/serverPack";

const DEFAULT_BLOB_ROOT = "https://97gks2fobqkgppwx.public.blob.vercel-storage.com";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

async function readTierState() {
  const configured = process.env.SCRYGLASS_TIERLIST_INDEX_URL?.trim();
  const blobRoot = process.env.LIVE_BLOB_BASE_URL?.trim() || DEFAULT_BLOB_ROOT;
  const url = configured || `${blobRoot.replace(/\/$/, "")}/rankings/tierlists.json`;
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`tier list ${response.status}`);
  const payload = (await response.json()) as Record<string, unknown>;
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
      tier = await readTierState();
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
