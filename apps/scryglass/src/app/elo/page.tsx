import type { Metadata } from "next";
import Link from "next/link";
import {
  SignalRatings,
  type PlayerRatingView,
  type PlayerRecordView,
  type RatingsTab,
  type TeamRatingView,
  type TeamRecordView,
} from "@/components/SignalRatings";
import type {
  CompetitionTier,
  PlayerRating,
  PlayerRecord,
  PlayerWeeklyRanks,
  TeamRating,
  TeamRecord,
  TeamWeeklyRanks,
} from "@/lib/pack";
import {
  compactPlayerRatings,
  findRecordByName,
  hasPromotedDraftAuthority,
  isActiveRating,
  packSourceUpdatedLabel,
  packUpdatedLabel,
  recordMatchesLeagues,
} from "@/lib/pack";
import type { DraftPlayerRow, DraftRankingsScope, DraftTeamRow } from "@/lib/draftRankings";
import {
  getRatingFacets,
  getRatings,
  queryApiAvailable,
  type RatingQueryRow,
} from "@/lib/publicData";
import { readPackJson, readPackManifest } from "@/lib/serverPack";
import styles from "./EloPage.module.css";

export const revalidate = 21_600;

export const metadata: Metadata = {
  title: "Team And Player Ratings",
  description: "Current professional League of Legends team and player ratings with evidence-aware filters.",
};

type PageProps = {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
};

function first(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}

function selectedTab(value: string | undefined): RatingsTab {
  return value === "players" || value === "draft" ? value : "teams";
}

function selectedScopes(value: string | undefined): string[] {
  if (value === "ALL") return [];
  const scopes = value?.split(",").map((item) => item.trim()).filter(Boolean) ?? [];
  return scopes.length ? scopes : ["TIER1"];
}

function integer(value: string | undefined, fallback: number, minimum: number, maximum: number): number {
  const parsed = Number.parseInt(value ?? "", 10);
  return Number.isInteger(parsed) ? Math.min(maximum, Math.max(minimum, parsed)) : fallback;
}

function querySort(value: string | undefined): "rating_desc" | "movement_desc" | "games_desc" | "name_asc" {
  if (value === "movement") return "movement_desc";
  if (value === "games") return "games_desc";
  if (value === "name") return "name_asc";
  return "rating_desc";
}

function ratingFromQuery(row: RatingQueryRow): PlayerRating | TeamRating {
  const rating = row.payload.rating;
  if (!rating || typeof rating !== "object") throw new Error("The public rating row has no rating payload");
  return rating;
}

function compactTeamRating(row: TeamRating): TeamRatingView {
  return {
    team: row.team,
    mu_total: row.mu_total,
    sigma: row.sigma,
    rating_p10: row.rating_p10,
    n_maps: row.n_maps,
    evidence_active: row.evidence_active,
  };
}

function compactPlayerRating(row: PlayerRating): PlayerRatingView {
  return {
    player: row.player,
    mu_total: row.mu_total,
    sigma: row.sigma,
    n_maps: row.n_maps,
    last_team: row.last_team,
    evidence_active: row.evidence_active,
  };
}

function compactTeamRecord(row: TeamRecord): TeamRecordView {
  return {
    leagues: row.leagues ?? [],
    primary: row.primary ?? null,
    intl: Boolean(row.intl),
    interregional: Boolean(row.interregional),
    current_league: row.current_league ?? null,
    current_tier: row.current_tier ?? null,
    current_team: row.current_team ?? null,
    games: row.games,
    wr: row.wr,
    by_league: row.by_league,
    by_tier: row.by_tier,
  };
}

function compactPlayerRecord(row: PlayerRecord): PlayerRecordView {
  return {
    games: row.games,
    wr: row.wr,
    primary_role: row.primary_role ?? row.roles?.[0] ?? null,
    current_league: row.current_league ?? null,
    current_tier: row.current_tier ?? null,
    current_team: row.current_team ?? null,
  };
}

function recordsFor<T, R>(
  names: string[],
  records: Record<string, T>,
  compact: (row: T) => R,
): Record<string, R> {
  return Object.fromEntries(names.flatMap((name) => {
    const record = findRecordByName(records, name);
    return record ? [[name, compact(record)] as const] : [];
  }));
}

function availableLeagues(records: Record<string, TeamRecord>): Record<CompetitionTier, string[]> {
  const sets: Record<CompetitionTier, Set<string>> = {
    tier1: new Set(),
    tier2: new Set(),
    tier3: new Set(),
  };
  for (const record of Object.values(records)) {
    if (record.current_tier && record.current_league) {
      sets[record.current_tier].add(record.current_league);
    }
  }
  return {
    tier1: [...sets.tier1].sort(),
    tier2: [...sets.tier2].sort(),
    tier3: [...sets.tier3].sort(),
  };
}

export default async function EloPage({ searchParams }: PageProps) {
  const query = await searchParams;
  const tab = selectedTab(first(query.tab));
  const scopes = selectedScopes(first(query.leagues));
  const manifest = await readPackManifest();
  const sourceUpdated = packSourceUpdatedLabel(manifest);
  const draftAuthorized = hasPromotedDraftAuthority(manifest);
  const boundedQueries = queryApiAvailable(manifest);
  const page = integer(first(query.page), 1, 1, 101);
  const pageSize = 100;
  let total = 0;

  let teams: TeamRatingView[] = [];
  let players: PlayerRatingView[] = [];
  let teamRecords: Record<string, TeamRecordView> = {};
  let playerRecords: Record<string, PlayerRecordView> = {};
  let movementByName: Record<string, number | null> = {};
  let draftTeams: DraftTeamRow[] = [];
  let draftPlayers: DraftPlayerRow[] = [];
  let draftScope: DraftRankingsScope = "whole_archive";
  let draftEvidenceGames: number | null = null;

  let allTeamRecords: Record<string, TeamRecord> = {};
  let availableLeaguesByTier: Record<CompetitionTier, string[]> | null = null;

  if (tab === "teams" && boundedQueries) {
    const selectedTiers = scopes.filter((scope) => scope.startsWith("TIER"));
    const selectedLeagues = scopes.filter((scope) => !scope.startsWith("TIER"));
    const [result, tier1, tier2, tier3] = await Promise.all([
      getRatings(manifest, {
        kind: "teams",
        leagues: selectedLeagues,
        tiers: selectedTiers.map((tier) => tier.toLowerCase()),
        active: true,
        search: first(query.q),
        order: querySort(first(query.sort)),
        limit: pageSize,
        offset: (page - 1) * pageSize,
      }),
      getRatingFacets(manifest, "teams", ["tier1"]),
      getRatingFacets(manifest, "teams", ["tier2"]),
      getRatingFacets(manifest, "teams", ["tier3"]),
    ]);
    total = result.total;
    teams = result.rows.map((row) => compactTeamRating(ratingFromQuery(row) as TeamRating));
    teamRecords = Object.fromEntries(result.rows.flatMap((row) => {
      const record = row.payload.record as TeamRecord | undefined;
      return record ? [[row.name, compactTeamRecord(record)] as const] : [];
    }));
    movementByName = Object.fromEntries(result.rows.map((row) => [row.name, row.movement]));
    allTeamRecords = Object.fromEntries(result.rows.flatMap((row) => {
      const record = row.payload.record as TeamRecord | undefined;
      return record ? [[row.name, record] as const] : [];
    }));
    availableLeaguesByTier = {
      tier1: tier1.leagues,
      tier2: tier2.leagues,
      tier3: tier3.leagues,
    };
  } else if (tab === "teams") {
    const [ratingRows, records, ranks] = await Promise.all([
      readPackJson<TeamRating[]>(manifest, "features/ratings_snapshot.json"),
      readPackJson<Record<string, TeamRecord>>(manifest, "features/team_records.json"),
      readPackJson<TeamWeeklyRanks>(manifest, "features/team_weekly_ranks.json").catch(() => null),
    ]);
    allTeamRecords = records;
    teams = ratingRows
      .filter(isActiveRating)
      .filter((team) => recordMatchesLeagues(findRecordByName(allTeamRecords, team.team), scopes))
      .map(compactTeamRating);
    teamRecords = recordsFor(teams.map((team) => team.team), allTeamRecords, compactTeamRecord);
    if (ranks) {
      movementByName = Object.fromEntries(teams.flatMap((team) => {
        const delta = ranks.by_team[team.team]?.delta;
        return delta == null ? [] : [[team.team, delta] as const];
      }));
    }
    total = teams.length;
  }

  if (tab === "players" && boundedQueries) {
    const selectedTiers = scopes.filter((scope) => scope.startsWith("TIER"));
    const selectedLeagues = scopes.filter((scope) => !scope.startsWith("TIER"));
    const [result, tier1, tier2, tier3] = await Promise.all([
      getRatings(manifest, {
        kind: "players",
        leagues: selectedLeagues,
        tiers: selectedTiers.map((tier) => tier.toLowerCase()),
        roles: first(query.role) ? [first(query.role)!] : [],
        active: true,
        search: first(query.q),
        order: querySort(first(query.sort)),
        minGames: integer(first(query.min), 20, 5, 10_000),
        limit: pageSize,
        offset: (page - 1) * pageSize,
      }),
      getRatingFacets(manifest, "players", ["tier1"]),
      getRatingFacets(manifest, "players", ["tier2"]),
      getRatingFacets(manifest, "players", ["tier3"]),
    ]);
    total = result.total;
    players = result.rows.map((row) => compactPlayerRating(ratingFromQuery(row) as PlayerRating));
    playerRecords = Object.fromEntries(result.rows.flatMap((row) => {
      const record = row.payload.record as PlayerRecord | undefined;
      return record ? [[row.name, compactPlayerRecord(record)] as const] : [];
    }));
    movementByName = Object.fromEntries(result.rows.map((row) => [row.name, row.movement]));
    availableLeaguesByTier = {
      tier1: tier1.leagues,
      tier2: tier2.leagues,
      tier3: tier3.leagues,
    };
  } else if (tab === "players") {
    const [ratingRows, allPlayerRecords, records, ranks] = await Promise.all([
      readPackJson<PlayerRating[]>(manifest, "features/player_ratings_snapshot.json"),
      readPackJson<Record<string, PlayerRecord>>(manifest, "features/player_records.json"),
      readPackJson<Record<string, TeamRecord>>(manifest, "features/team_records.json"),
      readPackJson<PlayerWeeklyRanks>(manifest, "features/player_weekly_ranks.json").catch(() => null),
    ]);
    allTeamRecords = records;
    players = compactPlayerRatings(ratingRows)
      .filter(isActiveRating)
      .filter((player) => recordMatchesLeagues(findRecordByName(allPlayerRecords, player.player), scopes))
      .map(compactPlayerRating);
    playerRecords = recordsFor(players.map((player) => player.player), allPlayerRecords, compactPlayerRecord);
    if (ranks) {
      movementByName = Object.fromEntries(players.flatMap((player) => {
        const tier = playerRecords[player.player]?.current_tier ?? "all";
        const rank = ranks.by_player[player.player]?.[tier] ?? ranks.by_player[player.player]?.all;
        return rank?.delta == null ? [] : [[player.player, rank.delta] as const];
      }));
    }
    total = players.length;
  }

  if (tab === "draft" && draftAuthorized) {
    try {
      const [leaderboards, records] = await Promise.all([
        readPackJson<{ teams_draft?: DraftTeamRow[]; players_draft?: DraftPlayerRow[] }>(
          manifest,
          "features/leaderboards.json",
        ),
        readPackJson<Record<string, TeamRecord>>(manifest, "features/team_records.json"),
      ]);
      allTeamRecords = records;
      draftTeams = leaderboards.teams_draft ?? [];
      draftPlayers = leaderboards.players_draft ?? [];
      draftScope = "whole_archive";
      draftEvidenceGames = manifest.draft_pool?.quality_games ?? manifest.draft_pool?.games ?? null;
    } catch {
      draftTeams = [];
      draftPlayers = [];
    }
  }

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
            Updated <time dateTime={manifest.created_utc}>{packUpdatedLabel(manifest)}</time>
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
      <SignalRatings
        key={tab}
        loadedTab={tab}
        draftAuthorized={draftAuthorized}
        draftUnavailableReason={manifest.draft_authority?.reason ?? "Draft Score is waiting for an independent promotion receipt."}
        draftTeams={draftTeams}
        draftPlayers={draftPlayers}
        draftScope={draftScope}
        draftWindowDays={null}
        draftEvidenceGames={draftEvidenceGames}
        teams={teams}
        players={players}
        teamRecords={teamRecords}
        playerRecords={playerRecords}
        movementByName={movementByName}
        availableLeaguesByTier={availableLeaguesByTier ?? availableLeagues(allTeamRecords)}
        total={total}
        initialPage={page}
        pageSize={pageSize}
        serverFiltered={boundedQueries && tab !== "draft"}
      />
    </div>
  );
}
