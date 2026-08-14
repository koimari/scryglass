import { NextResponse } from "next/server";
import { e2eLocalPackRoot, readPackJson } from "@/lib/serverPack";
import { readLocalManifest } from "@/lib/serverPack";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  if (e2eLocalPackRoot()) {
    const manifest = await readLocalManifest();
    return NextResponse.json(await readPackJson(manifest, "rankings/tierlists.json"), {
      headers: { "Cache-Control": "private, no-store" },
    });
  }
  return NextResponse.json(
    {
      status: "retired",
      reason: "The legacy tier-list artifact is retired. Use the bounded /tiers view.",
    },
    {
      status: 410,
      headers: {
        "Cache-Control": "private, no-store",
        Link: "</tiers>; rel=\"alternate\"",
      },
    },
  );
}
