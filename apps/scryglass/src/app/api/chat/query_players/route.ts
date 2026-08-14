import { chatError, chatJson, clean, readJsonBody, searchParams, secureChatRoute } from "@/lib/chatApi";
import {
  executePublishedQueryPlan,
  loadSupportQueryIndex,
  parseQueryPlan,
  planPlayerQuestion,
} from "@/lib/supportQuery";

export const runtime = "nodejs";

async function get(request: Request, signal: AbortSignal) {
  const question = clean(searchParams(request).get("q"));
  if (!question) return chatError("A player question is required.", 422);
  try {
    const index = await loadSupportQueryIndex(signal);
    const planned = planPlayerQuestion(question, index);
    if (!planned.ok) return chatError("The player question could not be resolved.", 422);
    return chatJson(await executePublishedQueryPlan(planned.plan, signal));
  } catch (error) {
    return chatError(error instanceof Error ? error.message : "The player query is unavailable.", 422);
  }
}

async function post(request: Request, signal: AbortSignal) {
  const parsedBody = await readJsonBody(request);
  if (!parsedBody.ok) return parsedBody.response;
  const body = parsedBody.value;
  const candidate = typeof body === "object" && body !== null && "plan" in body
    ? (body as { plan: unknown }).plan
    : body;
  const parsed = parseQueryPlan(candidate);
  if (!parsed.ok) return chatError("The query-plan payload is invalid.", 422);
  try {
    return chatJson(await executePublishedQueryPlan(parsed.plan, signal));
  } catch {
    return chatError("The player query is unavailable for the active release.", 422);
  }
}

export const GET = secureChatRoute(get);
export const POST = secureChatRoute(post);
