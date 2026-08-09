import { notFound } from "next/navigation";
import { PlayerRatingProfile } from "@/components/RatingProfiles";
import type {
  PlayerChampionRecord,
  PlayerRating,
  PlayerRecord,
  TeamRating,
} from "@/lib/pack";
import { compactPlayerRatings } from "@/lib/pack";
import { readPackJson, readPackManifest } from "@/lib/serverPack";

export const revalidate = 21_600;

type Props = { params: Promise<{ player: string }> };

export default async function PlayerEloPage({ params }: Props) {
  const { player: raw } = await params;
  const name = decodeURIComponent(raw);
  const man = await readPackManifest();
  const playerRows = await readPackJson<PlayerRating[]>(man, "features/player_ratings_snapshot.json");
  const players = compactPlayerRatings(playerRows);
  const teams = await readPackJson<TeamRating[]>(man, "features/ratings_snapshot.json");
  let playerRecords: Record<string, PlayerRecord> = {};
  let playerChampions: Record<string, PlayerChampionRecord[]> = {};
  try {
    playerRecords = await readPackJson(man, "features/player_records.json");
  } catch {
    playerRecords = {};
  }
  try {
    playerChampions = await readPackJson(man, "features/player_champion_records.json");
  } catch {
    playerChampions = {};
  }

  const player = players.find((p) => p.player.toLowerCase() === name.toLowerCase());
  if (!player) notFound();

  const rec = playerRecords[player.player];
  const currentTeam = rec?.current_team ?? player.last_team;
  const team = currentTeam
    ? teams.find((t) => t.team.toLowerCase() === currentTeam.toLowerCase())
    : null;

  return (
    <PlayerRatingProfile
      player={player}
      champions={playerChampions[player.player] ?? []}
      record={rec}
      team={team}
      manifest={man}
    />
  );
}
