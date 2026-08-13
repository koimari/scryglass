import { chatError, chatJson, clean, readChatJson, searchParams } from "@/lib/chatApi";

export const runtime = "nodejs";

type MatchIndexGame = {
  game_id: string;
  date: string;
  league: string;
  competition_tier?: string | null;
  blue_team: string;
  red_team: string;
  blue_win: number;
  champions: string[];
};

export async function GET(request: Request) {
  const params = searchParams(request);
  const team = clean(params.get("team"));
  const league = clean(params.get("league"));
  const champion = clean(params.get("champion"));
  const limit = Math.min(Math.max(parseInt(params.get("limit") ?? "10", 10) || 10, 1), 50);
  if (!team && !league && !champion) {
    return chatError("One of team, league, or champion is required.", 400);
  }
  try {
    const index = await readChatJson<{ games: MatchIndexGame[] }>("features/match_index.json");
    let games = index.games;
    if (team) {
      const lower = team.toLowerCase();
      games = games.filter(
        (game) =>
          game.blue_team.toLowerCase().includes(lower) || game.red_team.toLowerCase().includes(lower),
      );
    }
    if (league) {
      const lower = league.toLowerCase();
      games = games.filter((game) => game.league.toLowerCase() === lower);
    }
    if (champion) {
      const lower = champion.toLowerCase();
      games = games.filter((game) =>
        (game.champions ?? []).some((pick) => pick.toLowerCase() === lower),
      );
    }
    const sorted = [...games].sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : 0));
    return chatJson({ count: sorted.length, matches: sorted.slice(0, limit) });
  } catch {
    return chatError("Match search is unavailable for the current release.");
  }
}
