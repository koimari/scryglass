import { revalidatePath, revalidateTag } from "next/cache";
import { NextResponse } from "next/server";
import { validPublishSecret } from "@/lib/dataPublish";
import { PACK_MANIFEST_CACHE_TAG } from "@/lib/serverPack";

export const runtime = "nodejs";

export async function POST(request: Request) {
  if (!validPublishSecret(request.headers.get("authorization"), process.env.SCRYGLASS_DATA_PUBLISH_TOKEN)) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }
  revalidateTag(PACK_MANIFEST_CACHE_TAG, { expire: 0 });
  revalidatePath("/elo");
  revalidatePath("/matches");
  revalidatePath("/tiers");
  revalidatePath("/api/public-data/tierlists");
  return NextResponse.json({ revalidated: true });
}
