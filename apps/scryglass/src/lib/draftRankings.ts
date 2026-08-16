import type { ProfileGame, ProfileRecords } from "./pack";

export type DraftTeamRow = {
  team: string;
  games: number;
  draft_edge: number;
  /** Share of complete games where this team had a positive draft edge. */
  positive_edge_rate?: number | null;
  league?: string | null;
  tier?: string | null;
};

export type DraftPlayerRow = {
  player: string;
  games: number;
  draft_score: number | null;
  /** Share of evaluated picks that were the highest-ranked available pick. */
  best_available_rate?: number | null;
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
  positiveEdges: number;
};

type PlayerAggregate = {
  player: string;
  tier: string | null;
  league: string | null;
  scores: number[];
  bestAvailablePicks: number;
  evaluatedPicks: number;
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
): void {
  const key = scopeKey(team, tier, league);
  const aggregate = aggregates.get(key) ?? { team, tier, league, scores: [], positiveEdges: 0 };
  aggregate.scores.push(score);
  if (score > 0) aggregate.positiveEdges += 1;
  aggregates.set(key, aggregate);
}

function addPlayerEvidence(
  aggregates: Map<string, PlayerAggregate>,
  player: string,
  tier: string | null,
  league: string | null,
  score: number | null,
  bestPick: boolean | null,
  role: string,
  team: string,
): void {
  const key = scopeKey(player, tier, league);
  const aggregate = aggregates.get(key) ?? {
    player,
    tier,
    league,
    scores: [],
    bestAvailablePicks: 0,
    evaluatedPicks: 0,
    roles: new Map<string, number>(),
    teams: new Map<string, number>(),
  };
  if (finite(score)) aggregate.scores.push(score);
  if (bestPick !== null) {
    aggregate.evaluatedPicks += 1;
    if (bestPick) aggregate.bestAvailablePicks += 1;
  }
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
  const aggregates = new Map<string, { games: number; edge: number; positiveEdges: number }>();
  for (const row of rows) {
    const aggregate = aggregates.get(row.team) ?? { games: 0, edge: 0, positiveEdges: 0 };
    aggregate.games += row.games;
    aggregate.edge += row.draft_edge * row.games;
    if (row.positive_edge_rate != null) aggregate.positiveEdges += row.positive_edge_rate * row.games;
    aggregates.set(row.team, aggregate);
  }
  return [...aggregates.entries()]
    .map(([team, aggregate]) => ({
      team,
      games: aggregate.games,
      draft_edge: round(aggregate.edge / aggregate.games),
      positive_edge_rate: aggregate.games > 0 ? round(aggregate.positiveEdges / aggregate.games) : null,
    }))
    .sort((left, right) => right.draft_edge - left.draft_edge || right.games - left.games || left.team.localeCompare(right.team));
}

function aggregatePlayerRows(rows: DraftPlayerRow[]): DraftPlayerRow[] {
  const aggregates = new Map<string, {
    games: number;
    score: number;
    scoreGames: number;
    bestAvailablePicks: number;
    evaluatedPicks: number;
    roles: Map<string, number>;
    teams: Map<string, number>;
  }>();
  for (const row of rows) {
    const aggregate = aggregates.get(row.player) ?? {
      games: 0,
      score: 0,
      scoreGames: 0,
      bestAvailablePicks: 0,
      evaluatedPicks: 0,
      roles: new Map(),
      teams: new Map(),
    };
    aggregate.games += row.games;
    if (finite(row.draft_score)) {
      aggregate.score += row.draft_score * row.games;
      aggregate.scoreGames += row.games;
    }
    if (finite(row.best_available_rate)) {
      aggregate.bestAvailablePicks += row.best_available_rate * row.games;
      aggregate.evaluatedPicks += row.games;
    }
    if (row.role) aggregate.roles.set(row.role, (aggregate.roles.get(row.role) ?? 0) + row.games);
    if (row.team) aggregate.teams.set(row.team, (aggregate.teams.get(row.team) ?? 0) + row.games);
    aggregates.set(row.player, aggregate);
  }
  return [...aggregates.entries()]
    .map(([player, aggregate]) => ({
      player,
      games: aggregate.games,
      draft_score: aggregate.scoreGames ? round(aggregate.score / aggregate.scoreGames) : null,
      best_available_rate: aggregate.evaluatedPicks ? round(aggregate.bestAvailablePicks / aggregate.evaluatedPicks) : null,
      role: mostCommon(aggregate.roles),
      team: mostCommon(aggregate.teams),
    }))
    .sort((left, right) => (
      (right.best_available_rate ?? -Infinity) - (left.best_available_rate ?? -Infinity)
      || right.games - left.games
      || (right.draft_score ?? -Infinity) - (left.draft_score ?? -Infinity)
      || left.player.localeCompare(right.player)
    ));
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
 * Return true only when all pre-game draft facts needed for a descriptive
 * score are present. Missing facts stay unavailable at the UI boundary.
 */
export function hasCompleteDraftEvidence(game: ProfileGame): boolean {
  const pool = game.draft_pool;
  if (!pool || pool.status !== "complete" || !pool.patch || !game.competition_tier?.trim()) return false;
  const bans = [...(pool.bans?.Blue ?? []), ...(pool.bans?.Red ?? [])];
  const picked = pool.picked ?? [];
  const orders = picked.map((pick) => pick.order);
  const numericOrders = orders.filter((order): order is number => typeof order === "number" && Number.isInteger(order));
  const normalizeRole = (role: string | null): string => ({ jungle: "jungle", jng: "jungle", adc: "bot", bot: "bot", sup: "support", support: "support" }[role?.trim().toLowerCase() ?? ""] ?? role?.trim().toLowerCase() ?? "");
  const completeSides = (["Blue", "Red"] as const).every((side) => {
    const sidePicks = picked.filter((pick) => pick.side === side);
    const roles = sidePicks.map((pick) => normalizeRole(pick.role));
    return sidePicks.length === 5
      && roles.every((role) => ["top", "jungle", "mid", "bot", "support"].includes(role))
      && new Set(roles).size === 5;
  });
  return (
    bans.length === 10
    && new Set(bans).size === 10
    && picked.length === 10
    && new Set(picked.map((pick) => pick.champion)).size === 10
    && numericOrders.length === 10
    && numericOrders.every((order) => order >= 1 && order <= 10)
    && new Set(numericOrders).size === 10
    && completeSides
  );
}

/**
 * Build compact, scope-aware rankings from published profile evidence.
 * A team row stores the per-game mean descriptive edge in model units.
 * Player rows publish a best-available rate alongside the underlying
 * contribution. The rate is attached to each pick only after the public
 * record has complete bans, pick order, patch identity, and tier evidence.
 */
export function draftRankingsFromProfile(records: ProfileRecords): DraftRankings {
  const teamAggregates = new Map<string, TeamAggregate>();
  const playerAggregates = new Map<string, PlayerAggregate>();
  let evidenceGames = 0;

  for (const game of Object.values(records.games)) {
    const contribution = game.draft_contribution;
    const poolPicks = game.draft_pool?.picked ?? [];
    if (!hasCompleteDraftEvidence(game) || contribution?.status !== "available") continue;

    const blueSignal = contribution?.blue.signal;
    const redSignal = contribution?.red.signal;
    const tier = game.competition_tier?.trim().toLowerCase() || null;
    const league = game.league?.trim() || null;
    if (finite(blueSignal) && finite(redSignal)) {
      const edge = blueSignal - redSignal;
      addTeamEvidence(teamAggregates, game.blue_team, tier, league, edge);
      addTeamEvidence(teamAggregates, game.red_team, tier, league, -edge);
      evidenceGames += 1;
    }

    const contributionPicks = contribution?.picks ?? [];
    const contributionFor = (side: "Blue" | "Red", role: string, champion: string) => contributionPicks.find(
      (pick) => pick.side === side && roleKey(pick.role) === role && pick.champion === champion,
    );
    const evaluatedPoolPicks = poolPicks.filter((pick) => typeof pick.best_available === "boolean");
    const qualityPicks = evaluatedPoolPicks.length ? evaluatedPoolPicks : contributionPicks.filter(
      (pick) => typeof pick.best_available === "boolean",
    );
    for (const pick of qualityPicks) {
      const side = pick.side;
      const role = roleKey(pick.role);
      const participant = participantForPick(game, side, role);
      const player = participant?.player.trim();
      if (!participant || !player) continue;
      const contributionPick = "contribution" in pick
        ? pick
        : contributionFor(side, role, pick.champion);
      addPlayerEvidence(
        playerAggregates,
        player,
        tier,
        league,
        contributionPick?.contribution ?? null,
        typeof pick.best_available === "boolean" ? pick.best_available : null,
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
      draft_edge: round(average(aggregate.scores)),
      positive_edge_rate: round(aggregate.positiveEdges / aggregate.scores.length),
      league: aggregate.league,
      tier: aggregate.tier,
    }))
    .sort((left, right) => right.draft_edge - left.draft_edge || right.games - left.games || left.team.localeCompare(right.team));

  const players = [...playerAggregates.values()]
    .filter((aggregate) => aggregate.evaluatedPicks >= 5)
    .map((aggregate) => ({
      player: aggregate.player,
      games: aggregate.evaluatedPicks,
      draft_score: aggregate.scores.length ? round(average(aggregate.scores)) : null,
      best_available_rate: round(aggregate.bestAvailablePicks / aggregate.evaluatedPicks),
      role: mostCommon(aggregate.roles),
      team: mostCommon(aggregate.teams),
      league: aggregate.league,
      tier: aggregate.tier,
    }))
    .sort((left, right) => (
      (right.best_available_rate ?? -Infinity) - (left.best_available_rate ?? -Infinity)
      || right.games - left.games
      || (right.draft_score ?? -Infinity) - (left.draft_score ?? -Infinity)
      || left.player.localeCompare(right.player)
    ));

  return {
    teams,
    players,
    evidenceGames,
    scope: inferScope(records),
  };
}
