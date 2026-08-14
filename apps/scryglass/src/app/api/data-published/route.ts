import { revalidatePath, revalidateTag } from "next/cache";
import { NextResponse } from "next/server";
import {
  PUBLISH_RATE_LIMIT,
  rateLimitResponse,
  readJsonBody,
} from "@/lib/chatApi";
import { validPublishSecret, validReleaseId } from "@/lib/dataPublish";
import { REVALIDATED_TARGETS } from "@/lib/revalidationTargets";
import { PACK_MANIFEST_CACHE_TAG, readRemotePackManifest } from "@/lib/serverPack";

export const runtime = "nodejs";

// Next 16's runtime supports the `/route` cache target for App Router route
// handlers, while its public type still lists only page and layout.
const revalidateReleasePath = revalidatePath as unknown as (
  path: string,
  type: "layout" | "page" | "route",
) => void;

export async function POST(request: Request) {
  const limited = rateLimitResponse(request, "publish", PUBLISH_RATE_LIMIT);
  if (limited) return limited;
  if (!validPublishSecret(request.headers.get("authorization"), process.env.SCRYGLASS_DATA_PUBLISH_TOKEN)) {
    return NextResponse.json(
      { error: "unauthorized" },
      { status: 401, headers: { "Cache-Control": "private, no-store" } },
    );
  }
  const bodyResult = await readJsonBody(request);
  if (!bodyResult.ok) return bodyResult.response;
  const body = bodyResult.value as { release_id?: unknown } | null;
  if (!validReleaseId(body?.release_id)) {
    return NextResponse.json(
      { error: "valid release_id is required" },
      { status: 422, headers: { "Cache-Control": "private, no-store" } },
    );
  }
  revalidateTag(PACK_MANIFEST_CACHE_TAG, { expire: 0 });
  for (const target of REVALIDATED_TARGETS) revalidateReleasePath(target.path, target.type);
  const manifest = await readRemotePackManifest();
  const servedReleaseId = manifest.pack_id;
  const matches = servedReleaseId === body.release_id;
  return NextResponse.json(
    {
      revalidated: true,
      requested_release_id: body.release_id,
      served_release_id: servedReleaseId,
      matches,
    },
    {
      status: matches ? 200 : 409,
      headers: { "Cache-Control": "private, no-store" },
    },
  );
}
