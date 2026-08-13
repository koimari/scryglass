import { NextResponse } from "next/server";
import { supabaseConfig } from "@/lib/serverPack";

export const runtime = "nodejs";
export const revalidate = 3600;

/** Vercel-side proxy for Supabase public pack assets (storage + DB rows).
 * The pack assets are immutable per release, so responses are CDN-cached
 * (s-maxage) and the Next data cache absorbs refetches — Supabase egress
 * becomes one fetch per release instead of one per cache miss. */
export async function GET(
  request: Request,
  context: { params: Promise<{ path: string[] }> },
) {
  const { path } = await context.params;
  const config = supabaseConfig();
  if (!config) {
    return NextResponse.json({ error: "Supabase is not configured" }, { status: 500 });
  }
  const [releaseId, ...rest] = path;
  if (!releaseId || rest.length === 0) {
    return NextResponse.json({ error: "expected /api/assets/<release_id>/<asset path>" }, { status: 400 });
  }
  const release = encodeURIComponent(releaseId);
  const assetPath = encodeURIComponent(rest.join("/"));

  // 1) Storage-backed assets (objects live under <release_id>/<path>).
  const storagePath = [releaseId, ...rest].map((part) => encodeURIComponent(part)).join("/");
  const storageUrl = `${config.url}/storage/v1/object/public/scryglass-public/${storagePath}`;
  let response: Response | null = null;
  try {
    const storageResponse = await fetch(storageUrl, {
      headers: { apikey: config.publishableKey },
      cache: "force-cache",
    });
    if (storageResponse.ok) response = storageResponse;
  } catch {
    response = null;
  }

  // 2) DB-row (inline) assets: fall back to the public assets table.
  if (!response) {
    try {
      const rowResponse = await fetch(
        `${config.url}/rest/v1/scryglass_public_assets?release_id=eq.${release}&path=eq.${assetPath}&select=body&limit=1`,
        {
          headers: { apikey: config.publishableKey },
          cache: "force-cache",
        },
      );
      if (rowResponse.ok) {
        const rows = (await rowResponse.json()) as Array<{ body?: unknown }>;
        const body = rows[0]?.body;
        if (body !== undefined && body !== null) {
          return new NextResponse(JSON.stringify(body), {
            headers: {
              "Content-Type": "application/json",
              "Cache-Control": "public, max-age=3600, s-maxage=86400, stale-while-revalidate=604800",
            },
          });
        }
      }
    } catch {
      // fall through to the 404
    }
  }

  if (!response) {
    return NextResponse.json({ error: `asset ${404}` }, { status: 404 });
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
