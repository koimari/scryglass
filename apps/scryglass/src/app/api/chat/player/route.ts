import { chatError, chatJson, clean, readChatJson, searchParams } from "@/lib/chatApi";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const name = clean(searchParams(request).get("name"));
  if (!name) return chatError("A player name is required.", 400);
  try {
    const leaderboards = await readChatJson<{
      players: Record<string, {
        rating: number | null;
        role: string | null;
        team: string | null;
        league: string | null;
        grade_a_games: number;
        grade_games: number;
        games: number;
        win_rate: number | null;
        recent_form: number | null;
      }>;
      indexes: { players: Record<string, { role?: string | null; team?: string | null }> };
    }>("features/leaderboards.json");
    const lower = name.toLowerCase();
    const player = Object.keys(leaderboards.players).find(
      (candidate) => candidate.toLowerCase() === lower,
    ) ?? Object.keys(leaderboards.players).find((candidate) =>
      candidate.toLowerCase().includes(lower),
    );
    if (!player) return chatError(`No player found for "${name}".`);
    return chatJson({ player, ...leaderboards.players[player] });
  } catch {
    return chatError("Player lookup is unavailable for the current release.");
  }
}
