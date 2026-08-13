import { Suspense } from "react";
import Link from "next/link";
import { SignalRatings, type DraftPlayerRow, type DraftTeamRow } from "@/components/SignalRatings";
import type {
  CompetitionTier,
  PlayerChampionRecord,
  PlayerMetadata,
  PlayerRating,
  PlayerRecord,
  PlayerWeeklyRanks,
  ProfileGame,
  ProfileRecords,
  TeamRating,
  TeamRecord,
  TeamWeeklyRanks,
} from "@/lib/pack";
import {
  bestChampionRecords,
  compactPlayerRatings,
  isActiveRating,
  packSourceUpdatedLabel,
  packUpdatedLabel,
} from "@/lib/pack";
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
  let playerChampionRecords: Record<string, PlayerChampionRecord[]> = {};
  let profileRecords: ProfileRecords | null = null;
  let draftTeams: DraftTeamRow[] = [];
  let draftPlayers: DraftPlayerRow[] = [];
  try {
    const leaderboards = await readPackJson<{ teams_draft?: DraftTeamRow[]; players_draft?: DraftPlayerRow[] }>(man, "features/leaderboards.json");
    draftTeams = leaderboards.teams_draft ?? [];
    draftPlayers = leaderboards.players_draft ?? [];
  } catch {
    // draft rankings are optional
  }
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
    playerChampionRecords = await readPackJson<Record<string, PlayerChampionRecord[]>>(
      man,
      "features/player_champion_records.json",
    );
  } catch {
    playerChampionRecords = {};
  }
  try {
    profileRecords = await readPackJson<ProfileRecords>(man, "features/profile_records.json");
  } catch {
    profileRecords = null;
  }

  const activeTeamNames = new Set(
    teams.filter(isActiveRating).map((team) => team.team),
  );
  const leagueSets: Record<CompetitionTier, Set<string>> = {
    tier1: new Set(),
    tier2: new Set(),
    tier3: new Set(),
  };
  for (const [team, rec] of Object.entries(teamRecords)) {
    if (!activeTeamNames.has(team) || !rec.current_tier || !rec.current_league) continue;
    leagueSets[rec.current_tier].add(rec.current_league);
  }
  const availableLeaguesByTier: Record<CompetitionTier, string[]> = {
    tier1: [...leagueSets.tier1].sort(),
    tier2: [...leagueSets.tier2].sort(),
    tier3: [...leagueSets.tier3].sort(),
  };

  const championsByPlayer = new Map(
    Object.entries(playerChampionRecords).map(([player, records]) => [
      player.toLowerCase(),
      bestChampionRecords(records, 5),
    ]),
  );
  const playerChampionPicks = Object.fromEntries(
    players.map((player) => [
      player.player,
      (championsByPlayer.get(player.player.toLowerCase()) ?? []).map((record) => ({
        champion: record.champion,
        label: `${record.champion}: ${record.wins}-${record.losses} in ${record.games} games`,
      })),
    ]),
  );
  const roleOrder = ["top", "jungle", "mid", "bot", "support"];
  const teamChampionPicks = Object.fromEntries(
    teams.map((team) => {
      const games = (profileRecords?.teams[team.team] ?? [])
        .map((gameId) => profileRecords?.games[gameId])
        .filter((game): game is ProfileGame => Boolean(game))
        .sort((a, b) => Date.parse(b.date) - Date.parse(a.date));
      const latest = games[0];
      if (!latest) return [team.team, []];
      const side = latest.blue_team.toLowerCase() === team.team.toLowerCase()
        ? "Blue"
        : "Red";
      const picks = latest.players
        .filter((participant) => participant.side === side)
        .sort(
          (a, b) =>
            roleOrder.indexOf(a.role.toLowerCase()) -
            roleOrder.indexOf(b.role.toLowerCase()),
        )
        .slice(0, 5)
        .map((participant) => {
          const best = championsByPlayer.get(participant.player.toLowerCase())?.[0];
          const champion = best?.champion ?? participant.champion;
          return champion
            ? {
                champion,
                label: `${participant.role}: ${participant.player} · ${champion}${
                  best ? ` · ${best.wins}-${best.losses} in ${best.games} games` : ""
                }`,
              }
            : null;
        })
        .filter((pick): pick is { champion: string; label: string } => Boolean(pick));
      return [team.team, picks];
    }),
  );
  const teamRecentForms = Object.fromEntries(
    teams.map((team) => {
      const form = (profileRecords?.teams[team.team] ?? [])
        .map((gameId) => profileRecords?.games[gameId])
        .filter((game): game is ProfileGame => Boolean(game))
        .sort((a, b) => Date.parse(b.date) - Date.parse(a.date))
        .map((game) => {
          const blue = game.blue_team.toLowerCase() === team.team.toLowerCase();
          return blue ? game.blue_win === 1 : game.blue_win === 0;
        })
        .slice(0, 5);
      return [team.team, form];
    }),
  );
  const playerRecentForms = Object.fromEntries(
    players.map((player) => {
      const form = (profileRecords?.players[player.player] ?? [])
        .map((gameId) => profileRecords?.games[gameId])
        .filter((game): game is ProfileGame => Boolean(game))
        .sort((a, b) => Date.parse(b.date) - Date.parse(a.date))
        .flatMap((game) => {
          const participant = game.players.find(
            (entry) => entry.player.toLowerCase() === player.player.toLowerCase(),
          );
          if (!participant) return [];
          return [participant.side === "Blue" ? game.blue_win === 1 : game.blue_win === 0];
        })
        .slice(0, 5);
      return [player.player, form];
    }),
  );

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
      <details className={styles.ratingExplainer}>
        <summary>How is the rating calculated?</summary>
        <div>
          <p><strong>Teams:</strong> series results, opponent strength, and uncertainty. Beating a strong team carries more information than beating a weak one.</p>
          <p><strong>Players:</strong> game results with that player in the five-person lineup, adjusted for the opposing lineup and uncertainty. Individual statistics affect game grades, not the rating.</p>
          <Link href="/methodology#player-ratings">Read the full method →</Link>
        </div>
      </details>
      <Suspense fallback={<div className="skeleton-block" aria-hidden />}>
        <SignalRatings
          draftTeams={draftTeams}
          draftPlayers={draftPlayers}
          teams={teams}
          players={players}
          teamRecords={teamRecords}
          teamWeeklyRanks={teamWeeklyRanks}
          playerRecords={playerRecords}
          playerWeeklyRanks={playerWeeklyRanks}
          playerMetadata={playerMetadata}
          availableLeaguesByTier={availableLeaguesByTier}
          championImages={profileRecords?.champion_images ?? {}}
          playerChampionPicks={playerChampionPicks}
          recentForms={{ teams: teamRecentForms, players: playerRecentForms }}
          teamChampionPicks={teamChampionPicks}
        />
      </Suspense>
    </div>
  );
}
