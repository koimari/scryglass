import { chatError, chatJson, clean, searchParams, secureChatRoute } from "@/lib/chatApi";
import { lookupPlayer } from "@/lib/chatData";

export const runtime = "nodejs";

async function get(request: Request) {
  const params = searchParams(request);
  const firstName = clean(params.get("player1"));
  const secondName = clean(params.get("player2"));
  if (!firstName || !secondName) return chatError("Two player names are required.", 422);

  const [first, second] = await Promise.all([lookupPlayer(firstName), lookupPlayer(secondName)]);
  if (!first || !second) return chatError("A matching player was not found.");

  const comparable = first.rating != null && second.rating != null;
  const difference = comparable ? Math.abs(first.rating! - second.rating!) : null;
  const better = comparable
    ? first.rating! === second.rating!
      ? null
      : first.rating! > second.rating!
        ? first.name
        : second.name
    : null;

  return chatJson({ players: [first, second], better, difference });
}

export const GET = secureChatRoute(get);
