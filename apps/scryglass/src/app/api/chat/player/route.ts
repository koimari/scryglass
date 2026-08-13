import { chatError, chatJson, clean, searchParams } from "@/lib/chatApi";
import { lookupPlayer } from "@/lib/chatData";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const name = clean(searchParams(request).get("name"));
  if (!name) return chatError("A player name is required.", 400);
  const player = await lookupPlayer(name);
  if (!player) return chatError(`No player found for "${name}".`);
  return chatJson(player);
}
