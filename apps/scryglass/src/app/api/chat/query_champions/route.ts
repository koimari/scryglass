import { chatError, chatJson, clean, searchParams } from "@/lib/chatApi";
import { loadSupportQueryIndex } from "@/lib/supportQuery";
import { queryChampions } from "@/lib/championQuery";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const question = clean(searchParams(request).get("q"));
  if (!question) return chatError("A champion question is required.", 400);
  try {
    const index = await loadSupportQueryIndex();
    return chatJson(queryChampions(index, question));
  } catch (error) {
    return chatError(error instanceof Error ? error.message : "Champion rankings are unavailable.", 422);
  }
}
