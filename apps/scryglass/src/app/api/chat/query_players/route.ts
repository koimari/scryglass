import { chatError, chatJson, clean, searchParams } from "@/lib/chatApi";
import {
  executeQueryPlan,
  loadSupportQueryIndex,
  parseQueryPlan,
  planPlayerQuestion,
} from "@/lib/supportQuery";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const question = clean(searchParams(request).get("q"));
  if (!question) return chatError("A player question is required.", 400);
  try {
    const index = await loadSupportQueryIndex();
    const planned = planPlayerQuestion(question, index);
    if (!planned.ok) return chatError(planned.reason, 422);
    return chatJson(executeQueryPlan(planned.plan, index));
  } catch (error) {
    return chatError(error instanceof Error ? error.message : "The player query is unavailable.", 422);
  }
}

export async function POST(request: Request) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return chatError("The query-plan payload is invalid JSON.", 400);
  }
  const candidate = typeof body === "object" && body !== null && "plan" in body
    ? (body as { plan: unknown }).plan
    : body;
  const parsed = parseQueryPlan(candidate);
  if (!parsed.ok) return chatError(parsed.reason, 422);
  try {
    const index = await loadSupportQueryIndex();
    return chatJson(executeQueryPlan(parsed.plan, index));
  } catch {
    return chatError("The player query is unavailable for the active release.", 422);
  }
}
