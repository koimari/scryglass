import { NextResponse } from "next/server";
import { publicTierListViewDownloadUrl } from "@/lib/serverPack";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  try {
    const url = new URL(await publicTierListViewDownloadUrl("latest"), request.url);
    return NextResponse.redirect(url, {
      status: 307,
      headers: {
        "Cache-Control": "public, max-age=0, must-revalidate",
      },
    });
  } catch {
    return NextResponse.json(
      { status: "unavailable" },
      { status: 503, headers: { "Cache-Control": "private, no-store" } },
    );
  }
}
