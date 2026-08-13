import { chatJson, clean, readChatJson, searchParams } from "@/lib/chatApi";
import { leaderboardRows } from "@/lib/chatData";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const params = searchParams(request);
  const category = clean(params.get("category")) || "rating";
  const role = clean(params.get("role")) || null;
  const tier = clean(params.get("tier")) || null;
  const limit = Math.min(Math.max(parseInt(params.get("limit") ?? "10", 10) || 10, 1), 50);
  let rows = await leaderboardRows(category, role, limit, tier);
  if (category === "teams_draft" || category === "players_draft") {
    // Whole-archive draft leaderboards. Teams use draft win share. Players use
    // the share of scored drafts where their pick led their side.
    try {
      const payload = await readChatJson<{ teams_draft: Array<{ team?: string; player?: string; games?: number; draft_win_share?: number; draft_score?: number; best_pick_rate?: number | null; role?: string | null }>; players_draft: Array<{ team?: string; player?: string; games?: number; draft_win_share?: number; draft_score?: number; best_pick_rate?: number | null; role?: string | null }> }>("features/leaderboards.json");
      const source = category === "teams_draft" ? payload.teams_draft : payload.players_draft;
      rows = source.slice(0, limit).map((row) => ({
        name: String(row.team ?? row.player ?? ""),
        rating: null,
        role: row.role ?? null,
        team: row.team ?? null,
        league: null,
        tier: null,
        games: row.games ?? 0,
        wins: null,
        win_rate: null,
        grade_a_games: 0,
        grade_games: 0,
        recent_form: category === "players_draft"
          ? row.best_pick_rate ?? null
          : row.draft_win_share ?? null,
      }));
    } catch {
      rows = [];
    }
  }
  return chatJson({ category, role, tier, limit: rows.length, rows });
}
