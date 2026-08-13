import { chatError, chatJson, clean, searchParams } from "@/lib/chatApi";
import { lookupTeam } from "@/lib/chatData";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const name = clean(searchParams(request).get("name"));
  if (!name) return chatError("A team name is required.", 400);
  const team = await lookupTeam(name);
  if (!team) return chatError(`No team found for "${name}".`);
  return chatJson(team);
}
