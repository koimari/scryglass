import { NextResponse } from "next/server";
import { publicTierListViewDownloadUrl } from "@/lib/serverPack";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  try {
    const url = new URL(await publicTierListViewDownloadUrl("full"), request.url);
    return NextResponse.redirect(url, {
      status: 307,
      headers: {
        "Cache-Control": "public, max-age=0, s-maxage=21600, stale-while-revalidate=3600",
      },
    });
  } catch {
    return NextResponse.json(
      { status: "unavailable" },
      { status: 503, headers: { "Cache-Control": "private, no-store" } },
    );
  }
}
