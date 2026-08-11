import { NextResponse } from "next/server";
import { publicTierListViewDownloadUrl } from "@/lib/serverPack";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const view = new URL(request.url).searchParams.get("view") === "latest" ? "latest" : "full";
  try {
    const url = await publicTierListViewDownloadUrl(view);
    return NextResponse.redirect(url, {
      status: 307,
      headers: {
        "Cache-Control": "public, max-age=0, s-maxage=21600, stale-while-revalidate=3600",
      },
    });
  } catch {
    if (view === "latest") {
      try {
        const url = await publicTierListViewDownloadUrl("full");
        return NextResponse.redirect(url, {
          status: 307,
          headers: {
            "Cache-Control": "public, max-age=0, s-maxage=60, stale-while-revalidate=60",
          },
        });
      } catch {
        // Use the shared unavailable response below.
      }
    }
    return NextResponse.json(
      { status: "unavailable", reason: "The current tier-list release is unavailable." },
      { status: 503 },
    );
  }
}
