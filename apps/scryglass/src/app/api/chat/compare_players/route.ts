import { chatError, chatJson, clean, searchParams } from "@/lib/chatApi";
import { lookupPlayer } from "@/lib/chatData";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const params = searchParams(request);
  const firstName = clean(params.get("player1"));
  const secondName = clean(params.get("player2"));
  if (!firstName || !secondName) return chatError("Two player names are required.", 400);

  const [first, second] = await Promise.all([lookupPlayer(firstName), lookupPlayer(secondName)]);
  if (!first) return chatError(`No player found for "${firstName}".`);
  if (!second) return chatError(`No player found for "${secondName}".`);

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
