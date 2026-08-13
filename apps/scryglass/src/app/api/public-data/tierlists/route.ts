import { NextResponse } from "next/server";
import {
  e2eLocalPackRoot,
  publicTierListViewDownloadUrl,
  readPublicTierList,
} from "@/lib/serverPack";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const view = new URL(request.url).searchParams.get("view") === "latest" ? "latest" : "full";
  try {
    if (e2eLocalPackRoot()) {
      return NextResponse.json(await readPublicTierList(), {
        headers: { "Cache-Control": "private, no-store" },
      });
    }
    const url = new URL(await publicTierListViewDownloadUrl(view), request.url);
    return NextResponse.redirect(url, {
      status: 307,
      headers: {
        "Cache-Control": "public, max-age=0, must-revalidate",
      },
    });
  } catch {
    if (view === "latest") {
      try {
        const url = new URL(await publicTierListViewDownloadUrl("full"), request.url);
        return NextResponse.redirect(url, {
          status: 307,
          headers: {
            "Cache-Control": "public, max-age=0, must-revalidate",
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
