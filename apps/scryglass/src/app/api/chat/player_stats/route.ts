import { chatError, chatJson, clean, searchParams, secureChatRoute } from "@/lib/chatApi";
import { lookupMapStats } from "@/lib/playerMapStats";

export const runtime = "nodejs";

async function get(request: Request, signal: AbortSignal) {
  const name = clean(searchParams(request).get("name"));
  if (!name) return chatError("A player name is required.", 422);
  const stats = await lookupMapStats("players", name);
  if (!stats) return chatError("Per-map statistics are not published for that player.");
  return chatJson({
    kind: "player",
    window_days: stats.window_days,
    map_limit: stats.map_limit,
    ...stats.entry,
  });
}

export const GET = secureChatRoute(get);
