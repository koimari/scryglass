import { notFound } from "next/navigation";
import { TeamEloDetail } from "@/components/TeamEloDetail";
import type { PlayerRating, TeamRating, TeamRecord } from "@/lib/pack";
import { readPackJson, readPackManifest } from "@/lib/serverPack";

type Props = { params: Promise<{ team: string }> };

export default async function TeamEloPage({ params }: Props) {
  const { team: raw } = await params;
  const teamName = decodeURIComponent(raw);
  const man = await readPackManifest();
  const teams = await readPackJson<TeamRating[]>(man, "features/ratings_snapshot.json");
  const players = await readPackJson<PlayerRating[]>(man, "features/player_ratings_snapshot.json");
  let teamRecords: Record<string, TeamRecord> = {};
  try {
    teamRecords = await readPackJson(man, "features/team_records.json");
  } catch {
    teamRecords = {};
  }
  const team = teams.find((t) => t.team.toLowerCase() === teamName.toLowerCase());
  if (!team) notFound();

  const roster = players.filter(
    (p) => (p.last_team || "").toLowerCase() === team.team.toLowerCase(),
  );
  const baseUrl = man.base_url || `/packs/${man.pack_id}`;

  return (
    <TeamEloDetail
      team={team}
      roster={roster}
      record={teamRecords[team.team]}
      baseUrl={baseUrl}
      years={man.filters.years}
      manifest={man}
    />
  );
}
