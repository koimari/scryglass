import { notFound } from "next/navigation";
import { TeamRatingProfile, type TeamRosterEntry } from "@/components/RatingProfiles";
import type { PlayerRating, PlayerRecord, ProfileGame, ProfileRecords, TeamRating, TeamRecord } from "@/lib/pack";
import {
  adjustedRating,
  compactPlayerRatings,
  findPlayerByRouteName,
  hasPublishedDraftAuthority,
  isActiveRating,
  PLAYER_SIGMA_MIN,
  softMu,
  TEAM_SIGMA_MIN,
} from "@/lib/pack";
import { draftRankingsFromProfile, filterDraftRankings } from "@/lib/draftRankings";
import { readMapStatsEntry } from "@/lib/playerMapStats";
import { getTeamProfile, queryApiAvailable } from "@/lib/publicData";
import { readPackJson, readPackManifest } from "@/lib/serverPack";

export const revalidate = 21_600;

type Props = { params: Promise<{ team: string }> };

export default async function TeamEloPage({ params }: Props) {
  const { team: raw } = await params;
  const teamName = decodeURIComponent(raw);
  const man = await readPackManifest();
  if (queryApiAvailable(man)) {
    const profile = await getTeamProfile(man, teamName);
    if (!profile.row) notFound();
    const team = profile.row.payload.rating as TeamRating | undefined;
    if (!team) throw new Error("The public team profile has no rating payload");
    const record = profile.row.payload.record as TeamRecord | undefined;
    const playerRecords: Record<string, PlayerRecord> = {};
    const roster: TeamRosterEntry[] = profile.roster.map((row) => {
      const rating = row.payload.rating as PlayerRating | undefined;
      const playerRecord = row.payload.record as PlayerRecord | undefined;
      if (playerRecord) playerRecords[row.name] = playerRecord;
      return {
        player: row.name,
        role: row.role ?? playerRecord?.primary_role ?? "",
        rating: rating ?? null,
        ratingNote: rating ? undefined : "Rating unavailable",
      };
    });
    const roleRanks = Object.fromEntries(profile.roster.map((row) => [
      row.name,
      { rank: row.role_rank, total: row.role_total },
    ]));
    const standing = profile.standing ?? { tier_rank: 0, tier_total: 0 };
    const mapStats = await readMapStatsEntry(man, "teams", team.team);
    return (
      <TeamRatingProfile
        team={team}
        roster={roster}
        record={record}
        playerRecords={playerRecords}
        standing={{ tierRank: standing.tier_rank, tierTotal: standing.tier_total }}
        roleRanks={roleRanks}
        recentGames={profile.recent_games.map((row) => row.payload)}
        championImages={profile.champion_images}
        manifest={man}
        draftMetric={hasPublishedDraftAuthority(man) && profile.draft_metric?.draft_edge != null ? {
          draftEdge: profile.draft_metric.draft_edge,
          positiveEdgeRate: profile.draft_metric.positive_edge_rate ?? null,
          games: profile.draft_metric.games,
          scope: profile.draft_metric.scope === "whole_archive" ? "whole_archive" : "profile_window",
        } : null}
        mapStats={mapStats ? {
          entry: mapStats.entry,
          windowDays: mapStats.window_days,
          mapLimit: mapStats.map_limit,
        } : null}
      />
    );
  }
  const [teams, playerRows, teamRecords, playerRecords, profileRecords] = await Promise.all([
    readPackJson<TeamRating[]>(man, "features/ratings_snapshot.json"),
    readPackJson<PlayerRating[]>(man, "features/player_ratings_snapshot.json"),
    readPackJson<Record<string, TeamRecord>>(man, "features/team_records.json")
      .catch((): Record<string, TeamRecord> => ({})),
    readPackJson<Record<string, PlayerRecord>>(man, "features/player_records.json")
      .catch((): Record<string, PlayerRecord> => ({})),
    readPackJson<ProfileRecords>(man, "features/profile_records.json").catch(() => null),
  ]);
  const players = compactPlayerRatings(playerRows);
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
  const ratingFor = (name: string) => findPlayerByRouteName(players, name);
  const rawRatingFor = (name: string) => findPlayerByRouteName(playerRows, name);
  const publishedRoster = team.exact_roster?.players.length === 5
    ? team.exact_roster.players
    : null;
  const roster: TeamRosterEntry[] = publishedRoster
    ? publishedRoster.map((publishedPlayer) => {
      const displayName = publishedPlayer.display_name || publishedPlayer.player_id;
      const rating = ratingFor(displayName)
        ?? ratingFor(publishedPlayer.player_id)
        ?? null;
      const rawRating = rawRatingFor(displayName)
        ?? rawRatingFor(publishedPlayer.player_id);
      return {
        player: displayName,
        role: publishedPlayer.role,
        rating,
        ratingNote: (rawRating?.n_maps ?? 0) < 5
          ? "Rating needs 5 games"
          : "Rating unavailable",
      };
    })
    : latestGame && latestSide
    ? latestGame.players
      .filter((participant) => participant.side === latestSide)
      .map((participant) => ({
        player: participant.player,
        role: participant.role,
        rating: ratingFor(participant.player) ?? null,
        ratingNote: (rawRatingFor(participant.player)?.n_maps ?? 0) < 5
          ? "Rating needs 5 games"
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
    .filter(isActiveRating)
    .filter((candidate) => teamRecords[candidate.team]?.current_tier === record?.current_tier)
    .sort((a, b) => adjustedRating(b, TEAM_SIGMA_MIN) - adjustedRating(a, TEAM_SIGMA_MIN));
  const roleRanks: Record<string, { rank: number; total: number }> = {};
  for (const player of roster) {
    if (!player.rating) continue;
    const playerRecord = playerRecords[player.player];
    const peers = players
      .filter((candidate) => candidate.n_maps >= 20 && isActiveRating(candidate))
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
  const draftProfile = hasPublishedDraftAuthority(man) && profileRecords
    ? filterDraftRankings(draftRankingsFromProfile(profileRecords), { leagues: [], minGames: 5 })
    : null;
  const draftMetric = draftProfile?.teams.find(
    (row) => row.team.toLowerCase() === team.team.toLowerCase(),
  );
  const mapStats = await readMapStatsEntry(man, "teams", team.team);
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
      draftMetric={draftMetric && draftProfile ? {
        draftEdge: draftMetric.draft_edge,
        positiveEdgeRate: draftMetric.positive_edge_rate ?? null,
        games: draftMetric.games,
        scope: draftProfile.scope,
      } : null}
      mapStats={mapStats ? {
        entry: mapStats.entry,
        windowDays: mapStats.window_days,
        mapLimit: mapStats.map_limit,
      } : null}
    />
  );
}
