import { chatError, chatJson, clean, readChatJson, searchParams, secureChatRoute } from "@/lib/chatApi";
import { loadSupportQueryIndex } from "@/lib/supportQuery";
import { queryChampions, type PublishedTierBoard } from "@/lib/championQuery";

export const runtime = "nodejs";

async function get(request: Request) {
  const question = clean(searchParams(request).get("q"));
  if (!question) return chatError("A champion question is required.", 422);
  try {
    const [index, tierBoard] = await Promise.all([
      loadSupportQueryIndex(),
      readChatJson<PublishedTierBoard>("rankings/tierlists.json"),
    ]);
    return chatJson(queryChampions(index, question, tierBoard));
  } catch {
    return chatError("Champion rankings are unavailable for the active release.", 422);
  }
}

export const GET = secureChatRoute(get);
