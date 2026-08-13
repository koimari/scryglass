import { chatError, chatJson, clean, readChatJson, searchParams } from "@/lib/chatApi";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const params = searchParams(request);
  const category = clean(params.get("category")) || "rating";
  const role = clean(params.get("role"));
  const limit = Math.min(Math.max(parseInt(params.get("limit") ?? "10", 10) || 10, 1), 50);
  try {
    const payload = await readChatJson<{
      top: {
        a_grades: unknown[];
        rating: unknown[];
        win_rate: unknown[];
        rating_by_role: Record<string, unknown[]>;
      };
      teams: unknown[];
    }>("features/leaderboards.json");
    let rows: unknown[] = [];
    if (category === "a_grades") rows = payload.top.a_grades;
    else if (category === "win_rate") rows = payload.top.win_rate;
    else if (category === "rating" && role) rows = payload.top.rating_by_role[role] ?? [];
    else rows = payload.top.rating;
    if (category === "teams") rows = payload.teams;
    return chatJson({ category, role, limit: rows.length, rows: rows.slice(0, limit) });
  } catch {
    return chatError("Leaderboards are unavailable for the current release.");
  }
}
