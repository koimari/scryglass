import { chatJson, clean, searchParams, secureChatRoute } from "@/lib/chatApi";
import { leaderboardRows } from "@/lib/chatData";
import { draftRankingsFromProfile, filterDraftRankings } from "@/lib/draftRankings";
import { draftAuthorityStatus, type ProfileRecords } from "@/lib/pack";
import { readPackJson, readPackManifest } from "@/lib/serverPack";

export const runtime = "nodejs";

async function get(request: Request, signal: AbortSignal) {
  const params = searchParams(request);
  const category = clean(params.get("category")) || "rating";
  const role = clean(params.get("role")) || null;
  const tier = clean(params.get("tier")) || null;
  const limit = Math.min(Math.max(parseInt(params.get("limit") ?? "10", 10) || 10, 1), 20);
  if (category === "teams_draft" || category === "players_draft") {
    const manifest = await readPackManifest(signal);
    const authority = draftAuthorityStatus(manifest);
    if (authority !== "descriptive") {
      return chatJson({
        schema_version: "scryglass:draft-api:v1",
        release_id: manifest.pack_id,
        authority: "unavailable",
        category,
        role,
        tier,
        limit: 0,
        rows: [],
      });
    }
    const records = await readPackJson<ProfileRecords>(manifest, "features/profile_records.json", signal);
    const rankings = filterDraftRankings(draftRankingsFromProfile(records), {
      leagues: [],
      role: category === "players_draft" ? role ?? undefined : undefined,
      minGames: 5,
    });
    const rows = category === "teams_draft"
      ? rankings.teams
        .filter((row) => !tier || row.tier === tier.toLowerCase())
        .slice(0, limit)
        .map((row) => ({
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
          recent_form: null,
          draft_edge: row.draft_edge,
          positive_edge_rate: row.positive_edge_rate ?? null,
        }))
      : rankings.players
        .filter((row) => !tier || row.tier === tier.toLowerCase())
        .slice(0, limit)
        .map((row) => ({
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
          recent_form: null,
          best_available_rate: row.best_available_rate ?? null,
        }));
    return chatJson({
      schema_version: "scryglass:draft-api:v1",
      release_id: manifest.pack_id,
      authority,
      category,
      role,
      tier,
      limit: rows.length,
      rows,
    });
  }
  const rows = await leaderboardRows(category, role, limit, tier, signal);
  return chatJson({
    category,
    role,
    tier,
    limit: rows.length,
    rows,
  });
}

export const GET = secureChatRoute(get);
