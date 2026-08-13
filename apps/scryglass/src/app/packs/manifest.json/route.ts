import { NextResponse } from "next/server";
import { publicPackManifest, readRemotePackManifest } from "@/lib/serverPack";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const manifest = publicPackManifest(await readRemotePackManifest());
    return NextResponse.json(manifest, {
      headers: {
        "Cache-Control": "public, max-age=0, must-revalidate",
        "X-Scryglass-Release": manifest.release_id,
      },
    });
  } catch {
    return NextResponse.json(
      { status: "unavailable" },
      { status: 503, headers: { "Cache-Control": "private, no-store" } },
    );
  }
}
