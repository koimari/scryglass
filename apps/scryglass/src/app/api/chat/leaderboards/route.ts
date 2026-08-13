import { chatJson, clean, readChatJson, searchParams } from "@/lib/chatApi";
import { leaderboardRows } from "@/lib/chatData";
import { draftRankingsFromProfile, filterDraftRankings } from "@/lib/draftRankings";
import type { ProfileRecords } from "@/lib/pack";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const params = searchParams(request);
  const category = clean(params.get("category")) || "rating";
  const role = clean(params.get("role")) || null;
  const tier = clean(params.get("tier")) || null;
  const limit = Math.min(Math.max(parseInt(params.get("limit") ?? "10", 10) || 10, 1), 50);
  let rows = await leaderboardRows(category, role, limit, tier);
  if (category === "teams_draft" || category === "players_draft") {
    // Use the same profile evidence as player and team pages. The optional
    // leaderboard artifact may contain one row per league or tier.
    try {
      const records = await readChatJson<ProfileRecords>("features/profile_records.json");
      const rankings = filterDraftRankings(
        draftRankingsFromProfile(records),
        { leagues: tier ? [tier] : [], role: role ?? undefined, minGames: 5 },
      );
      if (category === "teams_draft") {
        rows = rankings.teams.slice(0, limit).map((row) => ({
          name: row.team,
          rating: null,
          role: null,
          team: row.team,
          league: row.league ?? null,
          tier: row.tier ?? null,
          games: row.games,
          wins: null,
          win_rate: null,
          grade_a_games: 0,
          grade_games: 0,
          recent_form: row.draft_win_share,
        }));
      } else {
        rows = rankings.players.slice(0, limit).map((row) => ({
          name: row.player,
          rating: null,
          role: row.role ?? null,
          team: row.team ?? null,
          league: row.league ?? null,
          tier: row.tier ?? null,
          games: row.games,
          wins: null,
          win_rate: null,
          grade_a_games: 0,
          grade_games: 0,
          recent_form: row.best_available_rate ?? null,
        }));
      }
    } catch {
      rows = [];
    }
  }
  return chatJson({ category, role, tier, limit: rows.length, rows });
}
