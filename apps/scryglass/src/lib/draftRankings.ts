import type { ProfileGame, ProfileRecords } from "./pack";

export type DraftTeamRow = {
  team: string;
  games: number;
  draft_win_share: number;
  draft_edge: number;
  league?: string | null;
  tier?: string | null;
};

export type DraftPlayerRow = {
  player: string;
  games: number;
  draft_score: number;
  role?: string | null;
  team?: string | null;
  league?: string | null;
  tier?: string | null;
};

export type DraftRankingsScope = "whole_archive" | "profile_window";

export type DraftRankings = {
  teams: DraftTeamRow[];
  players: DraftPlayerRow[];
  scope: DraftRankingsScope;
  evidenceGames: number;
};

export type DraftRankingFilters = {
  leagues: string[];
  role?: string;
  minGames?: number;
};

type TeamAggregate = {
  team: string;
  tier: string | null;
  league: string | null;
  scores: number[];
  winShares: number[];
};

type PlayerAggregate = {
  player: string;
  tier: string | null;
  league: string | null;
  scores: number[];
  roles: Map<string, number>;
  teams: Map<string, number>;
};

const ROLE_ALIASES: Record<string, string> = {
  top: "top",
  jungle: "jungle",
  jng: "jungle",
  mid: "mid",
  bot: "bot",
  adc: "bot",
  support: "support",
  sup: "support",
};

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function roleKey(value: string | null | undefined): string {
  return ROLE_ALIASES[value?.trim().toLowerCase() ?? ""] ?? value?.trim().toLowerCase() ?? "";
}

function scopeKey(identity: string, tier: string | null, league: string | null): string {
  return `${identity}\u0000${tier ?? ""}\u0000${league ?? ""}`;
}

function addTeamEvidence(
  aggregates: Map<string, TeamAggregate>,
  team: string,
  tier: string | null,
  league: string | null,
  score: number,
  winShare: number,
): void {
  const key = scopeKey(team, tier, league);
  const aggregate = aggregates.get(key) ?? { team, tier, league, scores: [], winShares: [] };
  aggregate.scores.push(score);
  aggregate.winShares.push(winShare);
  aggregates.set(key, aggregate);
}

function addPlayerEvidence(
  aggregates: Map<string, PlayerAggregate>,
  player: string,
  tier: string | null,
  league: string | null,
  score: number,
  role: string,
  team: string,
): void {
  const key = scopeKey(player, tier, league);
  const aggregate = aggregates.get(key) ?? {
    player,
    tier,
    league,
    scores: [],
    roles: new Map<string, number>(),
    teams: new Map<string, number>(),
  };
  aggregate.scores.push(score);
  if (role) aggregate.roles.set(role, (aggregate.roles.get(role) ?? 0) + 1);
  if (team) aggregate.teams.set(team, (aggregate.teams.get(team) ?? 0) + 1);
  aggregates.set(key, aggregate);
}

function mostCommon(counts: Map<string, number>): string | null {
  return [...counts.entries()].sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))[0]?.[0] ?? null;
}

function participantForPick(game: ProfileGame, side: "Blue" | "Red", role: string) {
  return game.players.find(
    (participant) => participant.side === side && roleKey(participant.role) === role,
  );
}

function average(values: number[]): number {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function round(value: number): number {
  return Number(value.toFixed(4));
}

function winShare(edge: number): number {
  return 1 / (1 + Math.exp(-edge));
}

function inferScope(records: ProfileRecords): DraftRankingsScope {
  if (records.window_days <= 0) return "whole_archive";
  const timestamps = Object.values(records.games)
    .map((game) => Date.parse(game.date))
    .filter(Number.isFinite);
  if (!timestamps.length) return "profile_window";
  const spanDays = (Math.max(...timestamps) - Math.min(...timestamps)) / 86_400_000;
  return spanDays > records.window_days + 2 ? "whole_archive" : "profile_window";
}

function rowMatchesScope(row: { league?: string | null; tier?: string | null }, leagues: string[]): boolean {
  if (!leagues.length || (!row.league && !row.tier)) return true;
  const tierFilters = leagues
    .filter((value) => value.toUpperCase().startsWith("TIER"))
    .map((value) => value.toLowerCase());
  const leagueFilters = leagues.filter((value) => !value.toUpperCase().startsWith("TIER"));
  if (tierFilters.length && row.tier && !tierFilters.includes(row.tier.toLowerCase())) return false;
  if (tierFilters.length && !row.tier && row.league) return false;
  if (leagueFilters.length && (!row.league || !leagueFilters.some((value) => value.toLowerCase() === row.league?.toLowerCase()))) return false;
  return true;
}

function aggregateTeamRows(rows: DraftTeamRow[]): DraftTeamRow[] {
  const aggregates = new Map<string, { games: number; edge: number; winShare: number }>();
  for (const row of rows) {
    const aggregate = aggregates.get(row.team) ?? { games: 0, edge: 0, winShare: 0 };
    aggregate.games += row.games;
    aggregate.edge += row.draft_edge * row.games;
    aggregate.winShare += row.draft_win_share * row.games;
    aggregates.set(row.team, aggregate);
  }
  return [...aggregates.entries()]
    .map(([team, aggregate]) => ({
      team,
      games: aggregate.games,
      draft_win_share: round(aggregate.winShare / aggregate.games),
      draft_edge: round(aggregate.edge / aggregate.games),
    }))
    .sort((left, right) => right.draft_win_share - left.draft_win_share || right.draft_edge - left.draft_edge || right.games - left.games || left.team.localeCompare(right.team));
}

function aggregatePlayerRows(rows: DraftPlayerRow[]): DraftPlayerRow[] {
  const aggregates = new Map<string, { games: number; score: number; roles: Map<string, number>; teams: Map<string, number> }>();
  for (const row of rows) {
    const aggregate = aggregates.get(row.player) ?? { games: 0, score: 0, roles: new Map(), teams: new Map() };
    aggregate.games += row.games;
    aggregate.score += row.draft_score * row.games;
    if (row.role) aggregate.roles.set(row.role, (aggregate.roles.get(row.role) ?? 0) + row.games);
    if (row.team) aggregate.teams.set(row.team, (aggregate.teams.get(row.team) ?? 0) + row.games);
    aggregates.set(row.player, aggregate);
  }
  return [...aggregates.entries()]
    .map(([player, aggregate]) => ({
      player,
      games: aggregate.games,
      draft_score: round(aggregate.score / aggregate.games),
      role: mostCommon(aggregate.roles),
      team: mostCommon(aggregate.teams),
    }))
    .sort((left, right) => right.draft_score - left.draft_score || right.games - left.games || left.player.localeCompare(right.player));
}

export function filterDraftRankings(rankings: DraftRankings, filters: DraftRankingFilters): DraftRankings {
  const minGames = Math.max(5, filters.minGames ?? 5);
  const scopedTeams = rankings.teams.filter((row) => rowMatchesScope(row, filters.leagues));
  const scopedPlayers = rankings.players
    .filter((row) => rowMatchesScope(row, filters.leagues))
    .filter((row) => !filters.role || roleKey(row.role) === roleKey(filters.role));
  return {
    ...rankings,
    teams: aggregateTeamRows(scopedTeams).filter((row) => row.games >= minGames),
    players: aggregatePlayerRows(scopedPlayers).filter((row) => row.games >= minGames),
  };
}

/**
 * Build compact, scope-aware rankings from published profile evidence.
 * A team row stores the per-game average of the descriptive draft win share.
 * Player rows keep pick contribution because one pick is not a team win
 * probability.
 */
export function draftRankingsFromProfile(records: ProfileRecords): DraftRankings {
  const teamAggregates = new Map<string, TeamAggregate>();
  const playerAggregates = new Map<string, PlayerAggregate>();
  let evidenceGames = 0;

  for (const game of Object.values(records.games)) {
    const contribution = game.draft_contribution;
    if (!contribution || (contribution.status !== "available" && contribution.status !== "limited")) continue;

    const blueSignal = contribution.blue.signal;
    const redSignal = contribution.red.signal;
    const tier = game.competition_tier?.trim().toLowerCase() || null;
    const league = game.league?.trim() || null;
    if (finite(blueSignal) && finite(redSignal)) {
      const edge = blueSignal - redSignal;
      const blueShare = winShare(edge);
      addTeamEvidence(teamAggregates, game.blue_team, tier, league, edge, blueShare);
      addTeamEvidence(teamAggregates, game.red_team, tier, league, -edge, 1 - blueShare);
      evidenceGames += 1;
    }

    for (const pick of contribution.picks) {
      if (!finite(pick.contribution)) continue;
      const side = pick.side;
      const role = roleKey(pick.role);
      const participant = participantForPick(game, side, role);
      const player = participant?.player.trim();
      if (!participant || !player) continue;
      addPlayerEvidence(
        playerAggregates,
        player,
        tier,
        league,
        pick.contribution,
        roleKey(participant.role) || role,
        side === "Blue" ? game.blue_team : game.red_team,
      );
    }
  }

  const teams = [...teamAggregates.values()]
    .filter((aggregate) => aggregate.scores.length >= 5)
    .map((aggregate) => ({
      team: aggregate.team,
      games: aggregate.scores.length,
      draft_win_share: round(average(aggregate.winShares)),
      draft_edge: round(average(aggregate.scores)),
      league: aggregate.league,
      tier: aggregate.tier,
    }))
    .sort((left, right) => right.draft_win_share - left.draft_win_share || right.draft_edge - left.draft_edge || right.games - left.games || left.team.localeCompare(right.team));

  const players = [...playerAggregates.values()]
    .filter((aggregate) => aggregate.scores.length >= 5)
    .map((aggregate) => ({
      player: aggregate.player,
      games: aggregate.scores.length,
      draft_score: round(average(aggregate.scores)),
      role: mostCommon(aggregate.roles),
      team: mostCommon(aggregate.teams),
      league: aggregate.league,
      tier: aggregate.tier,
    }))
    .sort((left, right) => right.draft_score - left.draft_score || right.games - left.games || left.player.localeCompare(right.player));

  return {
    teams,
    players,
    evidenceGames,
    scope: inferScope(records),
  };
}
