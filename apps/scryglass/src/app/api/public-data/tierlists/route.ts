import { NextResponse } from "next/server";
import { readPublicTierList } from "@/lib/serverPack";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const payload = await readPublicTierList<Record<string, unknown>>();
    return NextResponse.json(payload, {
      headers: {
        "Cache-Control": "public, max-age=0, s-maxage=21600, stale-while-revalidate=3600",
      },
    });
  } catch {
    return NextResponse.json(
      { status: "unavailable", reason: "The current tier-list release is unavailable." },
      { status: 503 },
    );
  }
}
