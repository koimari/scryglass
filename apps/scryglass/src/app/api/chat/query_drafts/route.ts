import { chatError, chatJson, clean, readChatJson, searchParams, secureChatRoute } from "@/lib/chatApi";
import { queryTeamDraftScores } from "@/lib/draftQuery";
import { hasPromotedDraftAuthority, type ProfileRecords } from "@/lib/pack";
import { readPackManifest } from "@/lib/serverPack";

export const runtime = "nodejs";

async function get(request: Request) {
  const question = clean(searchParams(request).get("q"));
  if (!question) return chatError("A team draft-score question is required.", 422);
  try {
    const manifest = await readPackManifest();
    if (!hasPromotedDraftAuthority(manifest)) {
      return chatJson({
        schema_version: "scryglass:draft-api:v1",
        release_id: manifest.pack_id,
        authority: "unavailable",
      });
    }
    const records = await readChatJson<ProfileRecords>("features/profile_records.json");
    return chatJson({
      schema_version: "scryglass:draft-api:v1",
      release_id: manifest.pack_id,
      authority: "promoted",
      result: queryTeamDraftScores(records, question),
    });
  } catch {
    return chatError("Team draft scores are unavailable for the active release.", 422);
  }
}

export const GET = secureChatRoute(get);
