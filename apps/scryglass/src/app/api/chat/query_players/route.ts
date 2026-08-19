import { chatError, chatJson, clean, readJsonBody, searchParams, secureChatRoute, CHAT_QUERY_ROUTE_TIMEOUT_MS, chatQueryFailureStatus } from "@/lib/chatApi";
import {
  executePublishedQueryPlan,
  loadSupportQueryIndex,
  parseQueryPlan,
  planPlayerQuestion,
} from "@/lib/supportQuery";

export const runtime = "nodejs";

const PLAYER_QUERY_WORDS = new Set([
  "active", "best", "champion", "compare", "find", "for", "games", "league",
  "list", "player", "players", "rank", "rating", "role", "show", "team",
  "tier", "what", "who", "with",
]);

function isSimplePlayerName(value: string): boolean {
  if (!value || value.length > 100 || /[?!,:;]/u.test(value)) return false;
  const words = value.split(/\s+/u).filter(Boolean);
  return words.length > 0 && words.length <= 4
    && words.every((word) => !PLAYER_QUERY_WORDS.has(word.toLocaleLowerCase()));
}

async function get(request: Request, signal: AbortSignal) {
  const question = clean(searchParams(request).get("q"));
  if (!question) return chatError("A player question is required.", 422);
  try {
    if (isSimplePlayerName(question)) {
      return chatJson(await executePublishedQueryPlan({
        version: 1,
        dataset: "players",
        operation: "rank",
        filters: [{ field: "name", op: "eq", value: question }],
        orderBy: [{ field: "rating", direction: "desc" }],
        limit: 20,
      }, signal));
    }
    const index = await loadSupportQueryIndex(signal);
    const planned = planPlayerQuestion(question, index);
    if (!planned.ok) return chatError("The player question could not be resolved.", 422);
    return chatJson(await executePublishedQueryPlan(planned.plan, signal));
  } catch (error) {
    return chatError(error instanceof Error ? error.message : "The player query is unavailable.", chatQueryFailureStatus(error));
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

export const GET = secureChatRoute(get, CHAT_QUERY_ROUTE_TIMEOUT_MS);
export const POST = secureChatRoute(post, CHAT_QUERY_ROUTE_TIMEOUT_MS);
