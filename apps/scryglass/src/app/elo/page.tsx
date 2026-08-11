import { Suspense } from "react";
import Link from "next/link";
import { SignalRatings } from "@/components/SignalRatings";
import type {
  PlayerMetadata,
  PlayerRating,
  PlayerRecord,
  PlayerWeeklyRanks,
  TeamRating,
  TeamRecord,
  TeamWeeklyRanks,
  ProfileRecords,
} from "@/lib/pack";
import { compactPlayerRatings, packSourceUpdatedLabel, packUpdatedLabel } from "@/lib/pack";
import { readPackJson, readPackManifest } from "@/lib/serverPack";
import styles from "./EloPage.module.css";

// Ratings use the current validated local pack. A local sync can replace this
// pack without rebuilding the application.
export const revalidate = 21_600;

export default async function EloPage() {
  const man = await readPackManifest();
  const sourceUpdated = packSourceUpdatedLabel(man);
  const teams = await readPackJson<TeamRating[]>(man, "features/ratings_snapshot.json");
  const playersRaw = await readPackJson<PlayerRating[]>(man, "features/player_ratings_snapshot.json");
  const players = compactPlayerRatings(playersRaw);

  let teamRecords: Record<string, TeamRecord> = {};
  let teamWeeklyRanks: TeamWeeklyRanks = { as_of: null, previous_as_of: null, by_team: {} };
  let playerRecords: Record<string, PlayerRecord> = {};
  let playerWeeklyRanks: PlayerWeeklyRanks = { as_of: null, previous_as_of: null, by_player: {} };
  let playerMetadata: Record<string, PlayerMetadata> = {};
  let profileRecords: ProfileRecords | null = null;
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
  try {
    profileRecords = await readPackJson<ProfileRecords>(man, "features/profile_records.json");
  } catch {
    profileRecords = null;
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
        <SignalRatings
          teams={teams}
          players={players}
          teamRecords={teamRecords}
          teamWeeklyRanks={teamWeeklyRanks}
          playerRecords={playerRecords}
          playerWeeklyRanks={playerWeeklyRanks}
          playerMetadata={playerMetadata}
          availableLeagues={availableLeagues}
          profileRecords={profileRecords}
        />
      </Suspense>
    </div>
  );
}
