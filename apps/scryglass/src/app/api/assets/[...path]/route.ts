import { createHash } from "node:crypto";
import { NextResponse } from "next/server";
import {
  fetchVerifiedStorageAsset,
  readActivePublicAsset,
  readVerifiedAssetBytes,
  type ActivePublicAsset,
} from "@/lib/serverPack";

export const runtime = "nodejs";
export const revalidate = 3600;

const IMMUTABLE_CACHE = "public, max-age=3600, s-maxage=86400, stale-while-revalidate=604800";

function unavailable(status = 404): NextResponse {
  return NextResponse.json(
    { error: "asset unavailable" },
    { status, headers: { "Cache-Control": "private, no-store" } },
  );
}

function verifiedStream(body: ReadableStream<Uint8Array>, asset: ActivePublicAsset): ReadableStream<Uint8Array> {
  const reader = body.getReader();
  const digest = createHash("sha256");
  let bytes = 0;
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        const chunk = await reader.read();
        if (!chunk.done) {
          bytes += chunk.value.byteLength;
          if (bytes > asset.bytes) {
            await reader.cancel("asset size mismatch");
            controller.error(new Error("Public asset integrity check failed"));
            return;
          }
          digest.update(chunk.value);
          controller.enqueue(chunk.value);
          return;
        }
        if (bytes !== asset.bytes || digest.digest("hex") !== asset.sha256) {
          controller.error(new Error("Public asset integrity check failed"));
          return;
        }
        controller.close();
      } catch (error) {
        controller.error(error);
      }
    },
    cancel(reason) {
      return reader.cancel(reason);
    },
  });
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
    const body = asset.storagePath
      ? verifiedStream((await fetchVerifiedStorageAsset(asset)).body!, asset)
      : await readVerifiedAssetBytes(asset);
    const responseBody = body instanceof Uint8Array ? Uint8Array.from(body).buffer : body;
    return new NextResponse(responseBody, {
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": IMMUTABLE_CACHE,
        "Content-Length": String(asset.bytes),
        "Content-Type": asset.contentType,
        "Cross-Origin-Resource-Policy": "cross-origin",
        ETag: `"${asset.sha256}"`,
      },
    });
  } catch {
    return unavailable(502);
  }
}
