import { revalidatePath, revalidateTag } from "next/cache";
import { NextResponse } from "next/server";
import { validPublishSecret, validReleaseId } from "@/lib/dataPublish";
import { PACK_MANIFEST_CACHE_TAG, readRemotePackManifest } from "@/lib/serverPack";

export const runtime = "nodejs";

export async function POST(request: Request) {
  if (!validPublishSecret(request.headers.get("authorization"), process.env.SCRYGLASS_DATA_PUBLISH_TOKEN)) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }
  const body = await request.json().catch(() => null) as { release_id?: unknown } | null;
  if (!validReleaseId(body?.release_id)) {
    return NextResponse.json({ error: "valid release_id is required" }, { status: 400 });
  }
  revalidateTag(PACK_MANIFEST_CACHE_TAG, { expire: 0 });
  revalidatePath("/elo");
  revalidatePath("/matches");
  revalidatePath("/tiers");
  revalidatePath("/api/public-data/tierlists");
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
    { status: matches ? 200 : 409 },
  );
}
