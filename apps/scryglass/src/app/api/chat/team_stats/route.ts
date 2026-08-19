import { chatError, chatJson, clean, searchParams, secureChatRoute } from "@/lib/chatApi";
import { lookupMapStats } from "@/lib/playerMapStats";

export const runtime = "nodejs";

async function get(request: Request, signal: AbortSignal) {
  const name = clean(searchParams(request).get("name"));
  if (!name) return chatError("A team name is required.", 422);
  const stats = await lookupMapStats("teams", name);
  if (!stats) return chatError("Per-map statistics are not published for that team.");
  return chatJson({
    kind: "team",
    window_days: stats.window_days,
    map_limit: stats.map_limit,
    ...stats.entry,
  });
}

export const GET = secureChatRoute(get);
