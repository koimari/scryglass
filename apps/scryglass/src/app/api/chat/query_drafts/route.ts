import { chatError, chatJson, clean, readChatJson, searchParams } from "@/lib/chatApi";
import { queryTeamDraftScores } from "@/lib/draftQuery";
import type { ProfileRecords } from "@/lib/pack";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const question = clean(searchParams(request).get("q"));
  if (!question) return chatError("A team draft-score question is required.", 400);
  try {
    const records = await readChatJson<ProfileRecords>("features/profile_records.json");
    return chatJson(queryTeamDraftScores(records, question));
  } catch (error) {
    return chatError(error instanceof Error ? error.message : "Team draft scores are unavailable.", 422);
  }
}
