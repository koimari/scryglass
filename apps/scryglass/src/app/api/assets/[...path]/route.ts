import { NextResponse } from "next/server";
import { supabaseConfig } from "@/lib/serverPack";

export const runtime = "nodejs";
export const revalidate = 3600;

/** Vercel-side proxy for Supabase Storage pack assets.
 * The pack assets are immutable per release, so the response is CDN-cached
 * (s-maxage) and the Next data cache (force-cache + revalidate) absorbs the
 * refetches — Supabase egress becomes one fetch per release instead of one
 * per cache miss. */
export async function GET(
  request: Request,
  context: { params: Promise<{ path: string[] }> },
) {
  const { path } = await context.params;
  const config = supabaseConfig();
  if (!config) {
    return NextResponse.json({ error: "Supabase is not configured" }, { status: 500 });
  }
  const safe = path.map((part) => encodeURIComponent(part)).join("/");
  const url = `${config.url}/storage/v1/object/public/scryglass-public/${safe}`;
  let response: Response;
  try {
    response = await fetch(url, {
      headers: { apikey: config.publishableKey },
      cache: "force-cache",
    });
  } catch {
    return NextResponse.json({ error: "Supabase Storage unreachable" }, { status: 502 });
  }
  if (!response.ok) {
    return NextResponse.json({ error: `asset ${response.status}` }, { status: response.status });
  }
  // Stream the body: Vercel Functions cap buffered responses at 4.5MB and
  // the pack assets are 20-47MB.
  return new NextResponse(response.body, {
    headers: {
      "Content-Type": "application/json",
      "Cache-Control": "public, max-age=3600, s-maxage=86400, stale-while-revalidate=604800",
    },
  });
}
