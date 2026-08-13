import { chatError, chatJson, clean, searchParams, secureChatRoute } from "@/lib/chatApi";
import { lookupPlayer } from "@/lib/chatData";

export const runtime = "nodejs";

async function get(request: Request, signal: AbortSignal) {
  const name = clean(searchParams(request).get("name"));
  if (!name) return chatError("A player name is required.", 422);
  const player = await lookupPlayer(name, signal);
  if (!player) return chatError("A matching player was not found.");
  return chatJson(player);
}

export const GET = secureChatRoute(get);
