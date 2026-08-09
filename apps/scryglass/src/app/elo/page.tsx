import { Suspense } from "react";
import Link from "next/link";
import { EloLadders } from "@/components/EloLadders";
import type {
  PlayerMetadata,
  PlayerRating,
  PlayerRecord,
  PlayerWeeklyRanks,
  TeamRating,
  TeamRecord,
  TeamWeeklyRanks,
} from "@/lib/pack";
import { packSourceUpdatedLabel, packUpdatedLabel, softMu } from "@/lib/pack";
import { readPackJson, readPackManifest } from "@/lib/serverPack";
import styles from "./EloPage.module.css";

// Ratings use the current validated local pack. A local sync can replace this
// pack without rebuilding the application.
export const revalidate = 21_600;

function thinPlayers(players: PlayerRating[]): PlayerRating[] {
  return players
    .filter((p) => (p.n_maps ?? 0) >= 5)
    .map((p) => ({
      player: p.player,
      mu_total: p.mu_total,
      mu_regional: p.mu_regional,
      mu_meta: p.mu_meta,
      sigma: p.sigma,
      n_maps: p.n_maps,
      last_team: p.last_team,
    }))
    .sort(
      (a, b) =>
        softMu(b.mu_total, b.sigma, 28) - softMu(a.mu_total, a.sigma, 28),
    );
}

export default async function EloPage() {
  const man = await readPackManifest();
  const sourceUpdated = packSourceUpdatedLabel(man);
  const teams = await readPackJson<TeamRating[]>(man, "features/ratings_snapshot.json");
  const playersRaw = await readPackJson<PlayerRating[]>(man, "features/player_ratings_snapshot.json");
  const players = thinPlayers(playersRaw);

  let teamRecords: Record<string, TeamRecord> = {};
  let teamWeeklyRanks: TeamWeeklyRanks = { as_of: null, previous_as_of: null, by_team: {} };
  let playerRecords: Record<string, PlayerRecord> = {};
  let playerWeeklyRanks: PlayerWeeklyRanks = { as_of: null, previous_as_of: null, by_player: {} };
  let playerMetadata: Record<string, PlayerMetadata> = {};
  try {
    teamRecords = await readPackJson(man, "features/team_records.json");
  } catch {
    teamRecords = {};
  }
  try {
    teamWeeklyRanks = await readPackJson<TeamWeeklyRanks>(man, "features/team_weekly_ranks.json");
  } catch {
    teamWeeklyRanks = { as_of: null, previous_as_of: null, by_team: {} };
  }
  try {
    playerRecords = await readPackJson(man, "features/player_records.json");
  } catch {
    playerRecords = {};
  }
  try {
    playerWeeklyRanks = await readPackJson<PlayerWeeklyRanks>(man, "features/player_weekly_ranks.json");
  } catch {
    playerWeeklyRanks = { as_of: null, previous_as_of: null, by_player: {} };
  }
  try {
    playerMetadata = await readPackJson<Record<string, PlayerMetadata>>(man, "features/player_metadata.json");
  } catch {
    playerMetadata = {};
  }

  const leagueSet = new Set<string>();
  for (const rec of Object.values(teamRecords)) {
    for (const lg of rec.leagues || []) leagueSet.add(lg);
  }
  const availableLeagues = [...leagueSet].sort();

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1>Team and player ratings</h1>
          <p>
            Current team and player strength from completed professional games.
            Adjusted rating discounts uncertain estimates.
          </p>
        </div>
        <div className={styles.provenance} aria-label="Ratings provenance">
          <span>
            Updated <time dateTime={man.created_utc}>{packUpdatedLabel(man)}</time>
          </span>
          {sourceUpdated ? <span>Source through {sourceUpdated}</span> : null}
          <Link href="/methodology">Method</Link>
        </div>
      </header>
      <Suspense fallback={<div className="skeleton-block" aria-hidden />}>
        <EloLadders
          teams={teams}
          players={players}
          teamRecords={teamRecords}
          teamWeeklyRanks={teamWeeklyRanks}
          playerRecords={playerRecords}
          playerWeeklyRanks={playerWeeklyRanks}
          playerMetadata={playerMetadata}
          availableLeagues={availableLeagues}
        />
      </Suspense>
    </div>
  );
}
