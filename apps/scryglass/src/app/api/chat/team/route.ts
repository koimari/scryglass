import { chatError, chatJson, clean, readChatJson, searchParams } from "@/lib/chatApi";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const name = clean(searchParams(request).get("name"));
  if (!name) return chatError("A team name is required.", 400);
  try {
    const leaderboards = await readChatJson<{
      teams: Array<{
        team: string;
        team_key: string | null;
        rating: number | null;
        league: string | null;
        games: number;
        wins: number;
        win_rate: number | null;
        recent: Array<{ date: string; opponent: string; side: string; won: boolean; game_id: string }>;
      }>;
      indexes: { teams: Record<string, { team_key?: string | null }> };
    }>("features/leaderboards.json");
    const lower = name.toLowerCase();
    const team = leaderboards.teams.find((entry) => entry.team.toLowerCase() === lower)
      ?? leaderboards.teams.find((entry) => entry.team.toLowerCase().includes(lower));
    if (!team) return chatError(`No team found for "${name}".`);
    return chatJson(team);
  } catch {
    return chatError("Team lookup is unavailable for the current release.");
  }
}
