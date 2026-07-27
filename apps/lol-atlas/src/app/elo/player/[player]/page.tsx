import { notFound } from "next/navigation";
import { PlayerEloDetail } from "@/components/PlayerEloDetail";
import type {
  PlayerPerformanceMeta,
  PlayerPerformanceRating,
  PlayerPerformanceValidation,
  PlayerRating,
  PlayerRatingsMeta,
  PlayerRecord,
  TeamRating,
} from "@/lib/pack";
import {
  currentMembershipContext,
  playerPerformanceContract,
  playerOutcomeOrderingVerified,
  verifiedPlayerAffiliation,
} from "@/lib/pack";
import { readPackJson, readPackManifest } from "@/lib/serverPack";

export const dynamic = "force-dynamic";

type Props = { params: Promise<{ player: string }> };

export default async function PlayerEloPage({ params }: Props) {
  const { player: raw } = await params;
  const name = decodeURIComponent(raw);
  const man = await readPackManifest();
  const players = await readPackJson<PlayerRating[]>(man, "features/player_ratings_snapshot.json");
  const teams = await readPackJson<TeamRating[]>(man, "features/ratings_snapshot.json");
  let playerRecords: Record<string, PlayerRecord> = {};
  let playerRatingsMeta: PlayerRatingsMeta | null = null;
  let playerPerformanceRows: PlayerPerformanceRating[] | null = null;
  let playerPerformanceMeta: PlayerPerformanceMeta | null = null;
  let playerPerformanceValidation: PlayerPerformanceValidation | null = null;
  try {
    playerRecords = await readPackJson(man, "features/player_records.json");
  } catch {
    playerRecords = {};
  }
  try {
    playerRatingsMeta = await readPackJson<PlayerRatingsMeta>(
      man,
      "features/player_ratings_meta.json",
    );
  } catch {
    playerRatingsMeta = null;
  }
  try {
    [playerPerformanceRows, playerPerformanceMeta, playerPerformanceValidation] =
      await Promise.all([
        readPackJson<PlayerPerformanceRating[]>(
          man,
          "features/player_performance_snapshot.json",
        ),
        readPackJson<PlayerPerformanceMeta>(
          man,
          "features/player_performance_meta.json",
        ),
        readPackJson<PlayerPerformanceValidation>(
          man,
          "features/player_performance_validation.json",
        ),
      ]);
  } catch {
    playerPerformanceRows = null;
    playerPerformanceMeta = null;
    playerPerformanceValidation = null;
  }

  const player = players.find((p) => p.player.toLowerCase() === name.toLowerCase());
  if (!player) notFound();

  const rec = playerRecords[player.player];
  const membershipContext = currentMembershipContext(man);
  const playerOrderingVerified = playerOutcomeOrderingVerified(
    playerRatingsMeta,
    players,
  );
  const currentAffiliation = verifiedPlayerAffiliation(rec, membershipContext);
  const performanceContract = playerPerformanceContract(
    playerPerformanceRows,
    playerPerformanceMeta,
    playerPerformanceValidation,
  );
  const playerPerformance = performanceContract.valid
    ? (playerPerformanceRows ?? []).filter(
        (row) => row.player_name.toLowerCase() === player.player.toLowerCase(),
      )
    : null;
  const team = currentAffiliation
    ? teams.find(
        (candidate) =>
          candidate.team.toLowerCase() === currentAffiliation.team.toLowerCase(),
      )
    : null;

  const baseUrl = man.base_url || `/packs/${man.pack_id}`;

  return (
    <PlayerEloDetail
      player={player}
      record={rec}
      team={team}
      baseUrl={baseUrl}
      years={man.filters.years}
      manifest={man}
      currentAffiliation={currentAffiliation}
      membershipContext={membershipContext}
      playerRatingsMeta={playerRatingsMeta}
      playerOrderingVerified={playerOrderingVerified}
      playerPerformance={playerPerformance}
      playerPerformanceUnavailableReason={
        performanceContract.valid ? null : performanceContract.reason
      }
    />
  );
}
