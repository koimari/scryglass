import { notFound } from "next/navigation";
import { PlayerRatingProfile } from "@/components/RatingProfiles";
import type {
  PlayerChampionRecord,
  PlayerRating,
  PlayerRecord,
  ProfileGame,
  ProfileRecords,
  TeamRating,
} from "@/lib/pack";
import { compactPlayerRatings, findPlayerByRouteName, PLAYER_SIGMA_MIN, softMu } from "@/lib/pack";
import { playerPortrait } from "@/lib/playerPortraits";
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
  let profileRecords: ProfileRecords | null = null;
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
  try {
    profileRecords = await readPackJson(man, "features/profile_records.json");
  } catch {
    profileRecords = null;
  }

  const player = findPlayerByRouteName(players, name);
  if (!player) notFound();

  const rec = playerRecords[player.player];
  const currentTeam = rec?.current_team ?? player.last_team;
  const team = currentTeam
    ? teams.find((t) => t.team.toLowerCase() === currentTeam.toLowerCase())
    : null;
  const tier = rec?.current_tier;
  const role = rec?.primary_role;
  const eligible = players
    .filter((candidate) => candidate.n_maps >= 20 && candidate.evidence_active !== 0)
    .filter((candidate) => playerRecords[candidate.player]?.current_tier === tier)
    .sort(
      (a, b) =>
        softMu(b.mu_total, b.sigma, PLAYER_SIGMA_MIN) -
        softMu(a.mu_total, a.sigma, PLAYER_SIGMA_MIN),
    );
  const roleEligible = eligible.filter(
    (candidate) => playerRecords[candidate.player]?.primary_role === role,
  );
  const gameIds = profileRecords?.players[player.player] ?? [];
  const recentGames = gameIds
    .map((gameId) => profileRecords?.games[gameId])
    .filter((game): game is ProfileGame => Boolean(game));

  return (
    <PlayerRatingProfile
      player={player}
      portrait={playerPortrait(player.player, currentTeam)}
      champions={playerChampions[player.player] ?? []}
      record={rec}
      team={team}
      standing={{
        tierRank: eligible.findIndex((candidate) => candidate.player === player.player) + 1,
        tierTotal: eligible.length,
        roleRank: roleEligible.findIndex((candidate) => candidate.player === player.player) + 1,
        roleTotal: roleEligible.length,
      }}
      recentGames={recentGames}
      championImages={profileRecords?.champion_images ?? {}}
      manifest={man}
    />
  );
}
