import { chatError, chatJson, clean, readChatJson, searchParams } from "@/lib/chatApi";
import { loadSupportQueryIndex } from "@/lib/supportQuery";
import { queryChampions, type PublishedTierBoard } from "@/lib/championQuery";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const question = clean(searchParams(request).get("q"));
  if (!question) return chatError("A champion question is required.", 400);
  try {
    const [index, tierBoard] = await Promise.all([
      loadSupportQueryIndex(),
      readChatJson<PublishedTierBoard>("rankings/tierlists.json"),
    ]);
    return chatJson(queryChampions(index, question, tierBoard));
  } catch (error) {
    return chatError(error instanceof Error ? error.message : "Champion rankings are unavailable.", 422);
  }
}
