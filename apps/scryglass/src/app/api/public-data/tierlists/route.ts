import { NextResponse } from "next/server";
import { publicTierListDownloadUrl } from "@/lib/serverPack";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const url = await publicTierListDownloadUrl();
    return NextResponse.redirect(url, {
      status: 307,
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
