import { NextResponse } from "next/server";
import {
  fetchVerifiedStorageAsset,
  readActivePublicAsset,
  type ActivePublicAsset,
} from "@/lib/serverPack";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const ACTIVE_RELEASE_CACHE = "public, max-age=0, must-revalidate";

function unavailable(status = 404): NextResponse {
  return NextResponse.json(
    { error: "asset unavailable" },
    { status, headers: { "Cache-Control": "private, no-store" } },
  );
}

function assetHeaders(asset: ActivePublicAsset): HeadersInit {
  return {
    "Access-Control-Allow-Origin": "*",
    "Cache-Control": ACTIVE_RELEASE_CACHE,
    "Content-Length": String(asset.bytes),
    "Content-Type": asset.contentType,
    "Cross-Origin-Resource-Policy": "cross-origin",
    ETag: `"${asset.sha256}"`,
    "X-Scryglass-Release": asset.releaseId,
  };
}

/** Serve only assets that belong to the active, manifest-bound release. */
export async function GET(
  _request: Request,
  context: { params: Promise<{ path: string[] }> },
) {
  const segments = (await context.params).path;
  const [releaseId, ...rest] = segments;
  if (!releaseId || rest.length === 0 || segments.length > 4) return unavailable();

  let asset: ActivePublicAsset | null;
  try {
    asset = await readActivePublicAsset(releaseId, rest.join("/"));
  } catch {
    return unavailable();
  }
  if (!asset) return unavailable();
  try {
    const verified = await fetchVerifiedStorageAsset(asset);
    return new NextResponse(verified.body, { headers: assetHeaders(asset) });
  } catch {
    return unavailable(502);
  }
}
