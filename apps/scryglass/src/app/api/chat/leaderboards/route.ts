import { chatJson, clean, searchParams } from "@/lib/chatApi";
import { leaderboardRows } from "@/lib/chatData";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const params = searchParams(request);
  const category = clean(params.get("category")) || "rating";
  const role = clean(params.get("role")) || null;
  const limit = Math.min(Math.max(parseInt(params.get("limit") ?? "10", 10) || 10, 1), 50);
  const rows = await leaderboardRows(category, role, limit);
  return chatJson({ category, role, limit: rows.length, rows });
}
