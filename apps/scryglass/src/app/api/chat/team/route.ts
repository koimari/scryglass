import { chatError, chatJson, clean, searchParams, secureChatRoute } from "@/lib/chatApi";
import { lookupTeam } from "@/lib/chatData";

export const runtime = "nodejs";

async function get(request: Request) {
  const name = clean(searchParams(request).get("name"));
  if (!name) return chatError("A team name is required.", 422);
  const team = await lookupTeam(name, request.signal);
  if (!team) return chatError("A matching team was not found.");
  return chatJson(team);
}

export const GET = secureChatRoute(get);
