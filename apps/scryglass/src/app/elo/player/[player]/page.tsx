import { notFound } from "next/navigation";
import { PlayerRatingProfile } from "@/components/RatingProfiles";
import type {
  PlayerChampionRecord,
  PlayerRating,
  PlayerWeeklyRanks,
  PlayerRecord,
  ProfileGame,
  ProfileRecords,
  TeamRating,
} from "@/lib/pack";
import { compactPlayerRatings, findPlayerByRouteName, hasPromotedDraftAuthority, isActiveRating, PLAYER_SIGMA_MIN, softMu } from "@/lib/pack";
import { draftRankingsFromProfile, filterDraftRankings } from "@/lib/draftRankings";
import { playerPortrait } from "@/lib/playerPortraits";
import { playerPositionDeltas } from "@/lib/playerMovement";
import { getPlayerProfile, queryApiAvailable } from "@/lib/publicData";
import { readPackJson, readPackManifest } from "@/lib/serverPack";

export const revalidate = 21_600;

type Props = { params: Promise<{ player: string }> };

export default async function PlayerEloPage({ params }: Props) {
  const { player: raw } = await params;
  const name = decodeURIComponent(raw);
  const man = await readPackManifest();
  if (queryApiAvailable(man)) {
    const profile = await getPlayerProfile(man, name);
    if (!profile.row) notFound();
    const player = profile.row.payload.rating as PlayerRating | undefined;
    if (!player) throw new Error("The public player profile has no rating payload");
    const record = profile.row.payload.record as PlayerRecord | undefined;
    const team = profile.team_row?.payload.rating as TeamRating | undefined;
    const champions = profile.champions.map((row) => ({
      ...(row.payload.record ?? {}),
      champion: row.champion,
      champion_image_url: row.payload.champion_image_url ?? null,
      games: row.games,
      wins: row.wins,
      losses: Math.max(0, row.games - row.wins),
      wr: row.win_rate,
      kills: row.payload.record?.kills ?? null,
      deaths: row.payload.record?.deaths ?? null,
      assists: row.payload.record?.assists ?? null,
    }));
    const standing = profile.standing ?? {
      tier_rank: 0,
      tier_total: 0,
      role_rank: 0,
      role_total: 0,
    };
    const positionDeltas = playerPositionDeltas(
      profile.row.payload.weekly,
      record?.current_tier ?? profile.row.tier,
    );
    return (
      <PlayerRatingProfile
        player={player}
        portrait={playerPortrait(player.player, record?.current_team ?? player.last_team)}
        champions={champions}
        record={record}
        team={team ?? null}
        standing={{
          tierRank: standing.tier_rank,
          tierTotal: standing.tier_total,
          roleRank: standing.role_rank,
          roleTotal: standing.role_total,
        }}
        positionDeltas={positionDeltas}
        recentGames={profile.recent_games.map((row) => row.payload)}
        championImages={profile.champion_images}
        manifest={man}
        draftMetric={null}
      />
    );
  }
  const [playerRows, teams, playerRecords, playerChampions, profileRecords, weeklyRanks] = await Promise.all([
    readPackJson<PlayerRating[]>(man, "features/player_ratings_snapshot.json"),
    readPackJson<TeamRating[]>(man, "features/ratings_snapshot.json"),
    readPackJson<Record<string, PlayerRecord>>(man, "features/player_records.json")
      .catch((): Record<string, PlayerRecord> => ({})),
    readPackJson<Record<string, PlayerChampionRecord[]>>(man, "features/player_champion_records.json")
      .catch((): Record<string, PlayerChampionRecord[]> => ({})),
    readPackJson<ProfileRecords>(man, "features/profile_records.json").catch(() => null),
    readPackJson<PlayerWeeklyRanks>(man, "features/player_weekly_ranks.json").catch(() => null),
  ]);
  const players = compactPlayerRatings(playerRows);

  const player = findPlayerByRouteName(players, name);
  if (!player) notFound();

  const rec = playerRecords[player.player];
  const currentTeam = rec?.current_team ?? player.last_team;
  const team = currentTeam
    ? teams.find((t) => t.team.toLowerCase() === currentTeam.toLowerCase())
    : null;
  const tier = rec?.current_tier;
  const role = rec?.primary_role;
  const rankScope = tier === "tier1" || tier === "tier2" || tier === "tier3" ? tier : "all";
  const eligible = players
    .filter((candidate) => candidate.n_maps >= 20 && isActiveRating(candidate))
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
  const draftProfile = hasPromotedDraftAuthority(man) && profileRecords
    ? filterDraftRankings(draftRankingsFromProfile(profileRecords), { leagues: [], minGames: 5 })
    : null;
  const draftMetric = draftProfile?.players.find(
    (row) => row.player.toLowerCase() === player.player.toLowerCase(),
  );

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
      positionDeltas={weeklyRanks?.by_player[player.player]?.[rankScope]?.position_deltas}
      recentGames={recentGames}
      championImages={profileRecords?.champion_images ?? {}}
      manifest={man}
      draftMetric={draftMetric?.best_available_rate != null && draftProfile ? {
        bestAvailableRate: draftMetric.best_available_rate,
        games: draftMetric.games,
        scope: draftProfile.scope,
      } : null}
    />
  );
}
