import { chatError, chatJson, clean, searchParams, secureChatRoute } from "@/lib/chatApi";
import { filterChatMatchesByTeam, loadChatMatches } from "@/lib/chatData";
import { queryApiAvailable } from "@/lib/publicData";
import { readPackManifest } from "@/lib/serverPack";

export const runtime = "nodejs";

async function get(request: Request) {
  const params = searchParams(request);
  const team = clean(params.get("team"));
  const league = clean(params.get("league"));
  const champion = clean(params.get("champion"));
  const limit = Math.min(Math.max(parseInt(params.get("limit") ?? "10", 10) || 10, 1), 20);
  if (!team && !league && !champion) {
    return chatError("One of team, league, or champion is required.", 422);
  }
  try {
    const bounded = queryApiAvailable(await readPackManifest(request.signal));
    let games = await loadChatMatches({
      team: team || undefined,
      league: league || undefined,
      champion: champion || undefined,
      limit,
    }, request.signal);
    if (!bounded && team) {
      games = filterChatMatchesByTeam(games, team);
    }
    if (!bounded && league) {
      const lower = league.toLowerCase();
      games = games.filter((game) => game.league.toLowerCase() === lower);
    }
    if (!bounded && champion) {
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

export const GET = secureChatRoute(get);
