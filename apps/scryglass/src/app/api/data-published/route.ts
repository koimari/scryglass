import { revalidatePath, revalidateTag } from "next/cache";
import { NextResponse } from "next/server";
import {
  PUBLISH_RATE_LIMIT,
  rateLimitResponse,
  readJsonBody,
} from "@/lib/chatApi";
import { validPublishSecret, validReleaseId } from "@/lib/dataPublish";
import { PACK_MANIFEST_CACHE_TAG, readRemotePackManifest } from "@/lib/serverPack";

export const runtime = "nodejs";

const REVALIDATED_PATHS = [
  "/",
  "/elo",
  "/matches",
  "/tiers",
  "/chat",
  "/methodology",
  "/packs/manifest.json",
  "/rankings/tierlists.json",
  "/rankings/tierlists-latest.json",
  "/api/public-data/tierlists",
  "/api/chat/compare_players",
  "/api/chat/leaderboards",
  "/api/chat/matches",
  "/api/chat/methodology",
  "/api/chat/navigation",
  "/api/chat/player",
  "/api/chat/query_champions",
  "/api/chat/query_drafts",
  "/api/chat/query_players",
  "/api/chat/schedule",
  "/api/chat/team",
  "/api/chat/tier",
] as const;

const REVALIDATED_PATTERNS = [
  "/elo/player/[player]",
  "/elo/team/[team]",
  "/matches/[game]",
  "/api/assets/[...path]",
] as const;

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
  for (const path of REVALIDATED_PATHS) revalidatePath(path);
  for (const pattern of REVALIDATED_PATTERNS) revalidatePath(pattern, "page");
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
