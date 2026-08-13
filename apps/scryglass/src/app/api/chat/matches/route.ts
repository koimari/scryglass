import { chatError, chatJson, clean, searchParams } from "@/lib/chatApi";
import { filterChatMatchesByTeam, loadChatMatches } from "@/lib/chatData";

export const runtime = "nodejs";

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
    let games = await loadChatMatches();
    if (team) {
      games = filterChatMatchesByTeam(games, team);
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
