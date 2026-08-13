import { chatError, chatJson, clean, readChatJson, searchParams } from "@/lib/chatApi";

export const runtime = "nodejs";

type Upcoming = {
  series_id: string;
  start_utc: string;
  has_time: boolean;
  status?: string;
  team1: string;
  team2: string;
  best_of?: number;
  tournament?: string;
  league?: string;
};

export async function GET(request: Request) {
  const league = clean(searchParams(request).get("league"));
  const limit = Math.min(Math.max(parseInt(searchParams(request).get("limit") ?? "10", 10) || 10, 1), 50);
  try {
    const schedule = await readChatJson<{ upcoming: Upcoming[] }>("features/schedule.json");
    let upcoming = schedule.upcoming ?? [];
    if (league) {
      const lower = league.toLowerCase();
      upcoming = upcoming.filter(
        (entry) =>
          (entry.league ?? "").toLowerCase() === lower ||
          (entry.tournament ?? "").toLowerCase().includes(lower),
      );
    }
    const sorted = [...upcoming].sort((a, b) => (a.start_utc < b.start_utc ? -1 : 1));
    return chatJson({ count: sorted.length, upcoming: sorted.slice(0, limit) });
  } catch {
    return chatError("The schedule is unavailable for the current release.");
  }
}
