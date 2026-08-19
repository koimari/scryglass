import { redirect } from "next/navigation";

/** /elo/team/<name>/stats — the address a person types for the Stats tab.
 * The Stats section renders on the profile page itself; this route exists so
 * the natural URL resolves instead of returning a 404. */
export default async function TeamStatsPage({
  params,
}: {
  params: Promise<{ team: string }>;
}) {
  const { team } = await params;
  redirect(`/elo/team/${encodeURIComponent(decodeURIComponent(team))}#stats`);
}
