import { chatJson, clean, searchParams } from "@/lib/chatApi";
import { leaderboardRows } from "@/lib/chatData";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const params = searchParams(request);
  const category = clean(params.get("category")) || "rating";
  const role = clean(params.get("role")) || null;
  const tier = clean(params.get("tier")) || null;
  const limit = Math.min(Math.max(parseInt(params.get("limit") ?? "10", 10) || 10, 1), 50);
  const rows = await leaderboardRows(category, role, limit, tier);
  return chatJson({ category, role, tier, limit: rows.length, rows });
}
