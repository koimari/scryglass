import { redirect } from "next/navigation";

/** /elo/player/<name>/stats — the address a person types for the Stats tab.
 * The Stats section renders on the profile page itself; this route exists so
 * the natural URL resolves instead of returning a 404. */
export default async function PlayerStatsPage({
  params,
}: {
  params: Promise<{ player: string }>;
}) {
  const { player } = await params;
  redirect(`/elo/player/${encodeURIComponent(decodeURIComponent(player))}#stats`);
}
