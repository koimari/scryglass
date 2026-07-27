import { Suspense } from "react";
import { EloLadders } from "@/components/EloLadders";
import type {
  PlayerMetadata,
  PlayerPerformanceMeta,
  PlayerPerformanceRating,
  PlayerPerformanceValidation,
  PlayerRating,
  PlayerRatingsMeta,
  PlayerRecord,
  PlayerWeeklyRanks,
  TeamRating,
  TeamRatingsMeta,
  TeamRecord,
} from "@/lib/pack";
import {
  currentMembershipContext,
  isIntlLeague,
  packUpdatedLabel,
  packDataThroughLabel,
  packUrl,
  playerOutcomeOrderingVerified,
  playerPerformanceContract,
} from "@/lib/pack";
import { readPackJson, readPackManifest } from "@/lib/serverPack";

// Ratings are refreshed independently of the app deployment and are served
// from the current Blob pack at request time.
export const dynamic = "force-dynamic";

function thinPlayers(players: PlayerRating[]): PlayerRating[] {
  return players
    .filter((p) => (p.n_maps ?? 0) >= 5)
    .map((p) => ({
      ...p,
    }));
}

export default async function EloPage() {
  const man = await readPackManifest();
  const teams = await readPackJson<TeamRating[]>(man, "features/ratings_snapshot.json");
  const playersRaw = await readPackJson<PlayerRating[]>(man, "features/player_ratings_snapshot.json");
  const players = thinPlayers(playersRaw);

  let teamRecords: Record<string, TeamRecord> = {};
  let playerRecords: Record<string, PlayerRecord> = {};
  let teamRatingsMeta: TeamRatingsMeta | null = null;
  let playerRatingsMeta: PlayerRatingsMeta | null = null;
  let playerPerformanceRows: PlayerPerformanceRating[] | null = null;
  let playerPerformanceMeta: PlayerPerformanceMeta | null = null;
  let playerPerformanceValidation: PlayerPerformanceValidation | null = null;
  let playerWeeklyRanks: PlayerWeeklyRanks = { as_of: null, previous_as_of: null, by_player: {} };
  let playerMetadata: Record<string, PlayerMetadata> = {};
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
    teamRatingsMeta = await readPackJson<TeamRatingsMeta>(man, "features/ratings_meta.json");
  } catch {
    teamRatingsMeta = null;
  }
  try {
    playerRatingsMeta = await readPackJson<PlayerRatingsMeta>(
      man,
      "features/player_ratings_meta.json",
    );
  } catch {
    playerRatingsMeta = null;
  }
  const playerOrderingVerified = playerOutcomeOrderingVerified(
    playerRatingsMeta,
    playersRaw,
  );
  if (playerOrderingVerified) {
    try {
      playerWeeklyRanks = await readPackJson<PlayerWeeklyRanks>(
        man,
        "features/player_weekly_ranks.json",
      );
    } catch {
      playerWeeklyRanks = { as_of: null, previous_as_of: null, by_player: {} };
    }
  }
  try {
    playerMetadata = await readPackJson<Record<string, PlayerMetadata>>(man, "features/player_metadata.json");
  } catch {
    playerMetadata = {};
  }
  try {
    [
      playerPerformanceRows,
      playerPerformanceMeta,
      playerPerformanceValidation,
    ] = await Promise.all([
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
  const performanceContract = playerPerformanceContract(
    playerPerformanceRows,
    playerPerformanceMeta,
    playerPerformanceValidation,
  );

  const membershipContext = currentMembershipContext(man);
  const leagueSet = new Set<string>(
    membershipContext.valid
      ? Object.keys(man.current_tournaments ?? {})
      : [],
  );
  for (const record of Object.values(teamRecords)) {
    for (const league of record.leagues ?? []) {
      if (isIntlLeague(league) || league === "AMERICAS") {
        leagueSet.add(league);
      }
    }
  }
  const availableLeagues = [...leagueSet].sort();

  return (
    <div className="space-y-6">
      <header className="page-header">
        <p className="blog-kicker">Ratings · team and player models</p>
        <h1 className="font-display mt-2 text-3xl">Ratings</h1>
        <p className="lede">
          Team ratings use a time-varying, pre-series Bradley–Terry model. Player Dual Elo remains a shared
          team-outcome signal, while the separately named 15-minute resource-performance view
          describes role-relative early resources. Neither is relabeled as general player skill.
        </p>
        <div className="micro-log mt-4">
          <span>
            <strong>Pack published</strong> {packUpdatedLabel(man)}
          </span>
          <span>
            <strong>Data through</strong> {packDataThroughLabel(man)}
          </span>
          <span>
            <strong>Pack</strong> {man.pack_id}
          </span>
          <span>
            <strong>Orgs</strong> {teams.length}
          </span>
          <span>
            <a className="row-link" href={packUrl(man, "features/ratings_snapshot.json")}>
              Snapshot JSON
            </a>
          </span>
        </div>
      </header>
      <Suspense fallback={<div className="skeleton-block" aria-hidden />}>
        <EloLadders
          teams={teams}
          players={players}
          teamRecords={teamRecords}
          playerRecords={playerRecords}
          playerWeeklyRanks={playerWeeklyRanks}
          playerMetadata={playerMetadata}
          availableLeagues={availableLeagues}
          dataAsOf={man.data_as_of ?? man.created_utc}
          recentActivityWindowDays={man.recent_activity_window_days ?? 90}
          currentTournaments={man.current_tournaments ?? {}}
          membershipContext={membershipContext}
          teamRatingsMeta={teamRatingsMeta}
          playerRatingsMeta={playerRatingsMeta}
          playerOrderingVerified={playerOrderingVerified}
          playerPerformanceRows={
            performanceContract.valid ? playerPerformanceRows ?? [] : null
          }
          playerPerformanceMeta={
            performanceContract.valid ? playerPerformanceMeta : null
          }
          playerPerformanceValidation={
            performanceContract.valid ? playerPerformanceValidation : null
          }
          playerPerformanceUnavailableReason={
            performanceContract.valid ? null : performanceContract.reason
          }
        />
      </Suspense>
    </div>
  );
}
