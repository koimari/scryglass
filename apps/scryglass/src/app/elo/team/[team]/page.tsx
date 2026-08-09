import { notFound } from "next/navigation";
import { TeamRatingProfile } from "@/components/RatingProfiles";
import type { PlayerRating, PlayerRecord, TeamRating, TeamRecord } from "@/lib/pack";
import { compactPlayerRatings } from "@/lib/pack";
import { readPackJson, readPackManifest } from "@/lib/serverPack";

export const revalidate = 21_600;

type Props = { params: Promise<{ team: string }> };

export default async function TeamEloPage({ params }: Props) {
  const { team: raw } = await params;
  const teamName = decodeURIComponent(raw);
  const man = await readPackManifest();
  const teams = await readPackJson<TeamRating[]>(man, "features/ratings_snapshot.json");
  const playerRows = await readPackJson<PlayerRating[]>(man, "features/player_ratings_snapshot.json");
  const players = compactPlayerRatings(playerRows);
  let teamRecords: Record<string, TeamRecord> = {};
  let playerRecords: Record<string, PlayerRecord> = {};
  try {
    teamRecords = await readPackJson(man, "features/team_records.json");
  } catch {
    teamRecords = {};
  }
  try {
    playerRecords = await readPackJson(man, "features/player_records.json");
  } catch {
    playerRecords = {};
  }
  const team = teams.find((t) => t.team.toLowerCase() === teamName.toLowerCase());
  if (!team) notFound();

  const roster = players.filter(
    (p) => (p.last_team || "").toLowerCase() === team.team.toLowerCase(),
  );
  return (
    <TeamRatingProfile
      team={team}
      roster={roster}
      record={teamRecords[team.team]}
      playerRecords={playerRecords}
      manifest={man}
    />
  );
}
