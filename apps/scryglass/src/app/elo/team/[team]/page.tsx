import { notFound } from "next/navigation";
import { TeamRatingProfile, type TeamRosterEntry } from "@/components/RatingProfiles";
import type { PlayerRating, PlayerRecord, ProfileGame, ProfileRecords, TeamRating, TeamRecord } from "@/lib/pack";
import { adjustedRating, compactPlayerRatings, PLAYER_SIGMA_MIN, softMu, TEAM_SIGMA_MIN } from "@/lib/pack";
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
  let profileRecords: ProfileRecords | null = null;
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
  try {
    profileRecords = await readPackJson(man, "features/profile_records.json");
  } catch {
    profileRecords = null;
  }
  const team = teams.find((t) => t.team.toLowerCase() === teamName.toLowerCase());
  if (!team) notFound();

  const record = teamRecords[team.team];
  const gameIds = profileRecords?.teams[team.team] ?? [];
  const recentGames = gameIds
    .map((gameId) => profileRecords?.games[gameId])
    .filter((game): game is ProfileGame => Boolean(game));
  const latestGame = recentGames[0];
  const latestSide = latestGame?.blue_team.toLowerCase() === team.team.toLowerCase()
    ? "Blue"
    : latestGame?.red_team.toLowerCase() === team.team.toLowerCase()
      ? "Red"
      : null;
  const ratingByPlayer = new Map(players.map((player) => [player.player.toLowerCase(), player]));
  const rawRatingByPlayer = new Map(playerRows.map((player) => [player.player.toLowerCase(), player]));
  const roster: TeamRosterEntry[] = latestGame && latestSide
    ? latestGame.players
      .filter((participant) => participant.side === latestSide)
      .map((participant) => ({
        player: participant.player,
        role: participant.role,
        rating: ratingByPlayer.get(participant.player.toLowerCase()) ?? null,
        ratingNote: (rawRatingByPlayer.get(participant.player.toLowerCase())?.n_maps ?? 0) < 5
          ? "Rating needs 5 maps"
          : "Rating unavailable",
      }))
    : players
      .filter((player) =>
        (playerRecords[player.player]?.current_team ?? player.last_team ?? "").toLowerCase() === team.team.toLowerCase(),
      )
      .map((player) => ({
        player: player.player,
        role: playerRecords[player.player]?.primary_role ?? "",
        rating: player,
        ratingNote: "Role rank unavailable",
      }));
  const tierTeams = teams
    .filter((candidate) => teamRecords[candidate.team]?.current_tier === record?.current_tier)
    .sort((a, b) => adjustedRating(b, TEAM_SIGMA_MIN) - adjustedRating(a, TEAM_SIGMA_MIN));
  const roleRanks: Record<string, { rank: number; total: number }> = {};
  for (const player of roster) {
    if (!player.rating) continue;
    const playerRecord = playerRecords[player.player];
    const peers = players
      .filter((candidate) => candidate.n_maps >= 20 && candidate.evidence_active !== 0)
      .filter((candidate) => {
        const candidateRecord = playerRecords[candidate.player];
        return candidateRecord?.current_tier === (record?.current_tier ?? playerRecord?.current_tier)
          && candidateRecord?.primary_role === player.role;
      })
      .sort(
        (a, b) =>
          softMu(b.mu_total, b.sigma, PLAYER_SIGMA_MIN) -
          softMu(a.mu_total, a.sigma, PLAYER_SIGMA_MIN),
      );
    roleRanks[player.player] = {
      rank: peers.findIndex((candidate) => candidate.player === player.player) + 1,
      total: peers.length,
    };
  }
  return (
    <TeamRatingProfile
      team={team}
      roster={roster}
      record={record}
      playerRecords={playerRecords}
      standing={{
        tierRank: tierTeams.findIndex((candidate) => candidate.team === team.team) + 1,
        tierTotal: tierTeams.length,
      }}
      roleRanks={roleRanks}
      recentGames={recentGames}
      championImages={profileRecords?.champion_images ?? {}}
      manifest={man}
    />
  );
}
