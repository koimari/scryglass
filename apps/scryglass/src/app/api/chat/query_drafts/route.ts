import { chatError, chatJson, clean, searchParams, secureChatRoute } from "@/lib/chatApi";
import { queryTeamDraftScores } from "@/lib/draftQuery";
import { hasDescriptiveDraftAuthority, type ProfileRecords } from "@/lib/pack";
import { readPackJson, readPackManifest } from "@/lib/serverPack";

export const runtime = "nodejs";

async function get(request: Request, signal: AbortSignal) {
  const question = clean(searchParams(request).get("q"));
  if (!question) return chatError("A team draft-score question is required.", 422);
  const manifest = await readPackManifest(signal);
  if (!hasDescriptiveDraftAuthority(manifest)) {
    return chatJson({
      schema_version: "scryglass:draft-api:v1",
      release_id: manifest.pack_id,
      authority: "unavailable",
      reason: manifest.draft_authority?.reason ?? "A release-bound descriptive Draft receipt is unavailable.",
    });
  }
  const records = await readPackJson<ProfileRecords>(manifest, "features/profile_records.json", signal);
  const result = queryTeamDraftScores(records, question);
  return chatJson({
    schema_version: "scryglass:draft-api:v1",
    release_id: manifest.pack_id,
    authority: "descriptive",
    ...result,
  });
}

export const GET = secureChatRoute(get);
