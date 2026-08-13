import type { ProfileGame, ProfileRecords } from "./pack";

export type DraftTeamRow = {
  team: string;
  games: number;
  draft_edge: number;
};

export type DraftPlayerRow = {
  player: string;
  games: number;
  draft_score: number;
  role?: string | null;
  team?: string | null;
};

export type DraftRankingsScope = "whole_archive" | "profile_window";

export type DraftRankings = {
  teams: DraftTeamRow[];
  players: DraftPlayerRow[];
  scope: DraftRankingsScope;
  evidenceGames: number;
};

type TeamAggregate = {
  scores: number[];
};

type PlayerAggregate = {
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

function addTeamScore(aggregates: Map<string, TeamAggregate>, team: string, score: number): void {
  const aggregate = aggregates.get(team) ?? { scores: [] };
  aggregate.scores.push(score);
  aggregates.set(team, aggregate);
}

function addCount(counts: Map<string, number>, value: string): void {
  if (!value) return;
  counts.set(value, (counts.get(value) ?? 0) + 1);
}

function mostCommon(counts: Map<string, number>): string | null {
  return [...counts.entries()].sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))[0]?.[0] ?? null;
}

function participantForPick(game: ProfileGame, side: "Blue" | "Red", role: string) {
  return game.players.find(
    (participant) => participant.side === side && roleKey(participant.role) === role,
  );
}

/**
 * Build the same descriptive rankings as features/leaderboards.json when an
 * older release has published profile evidence but no leaderboard asset yet.
 * Only finite, accepted composition signals enter the rankings.
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
    if (finite(blueSignal) && finite(redSignal)) {
      addTeamScore(teamAggregates, game.blue_team, blueSignal - redSignal);
      addTeamScore(teamAggregates, game.red_team, redSignal - blueSignal);
      evidenceGames += 1;
    }

    for (const pick of contribution.picks) {
      if (!finite(pick.contribution)) continue;
      const side = pick.side;
      const role = roleKey(pick.role);
      const participant = participantForPick(game, side, role);
      const player = participant?.player.trim();
      if (!participant || !player) continue;
      const aggregate = playerAggregates.get(player) ?? {
        scores: [],
        roles: new Map<string, number>(),
        teams: new Map<string, number>(),
      };
      aggregate.scores.push(pick.contribution);
      addCount(aggregate.roles, roleKey(participant.role) || role);
      addCount(aggregate.teams, side === "Blue" ? game.blue_team : game.red_team);
      playerAggregates.set(player, aggregate);
    }
  }

  const teams = [...teamAggregates.entries()]
    .filter(([, aggregate]) => aggregate.scores.length >= 5)
    .map(([team, aggregate]) => ({
      team,
      games: aggregate.scores.length,
      draft_edge: Number((aggregate.scores.reduce((sum, score) => sum + score, 0) / aggregate.scores.length).toFixed(4)),
    }))
    .sort((left, right) => right.draft_edge - left.draft_edge || right.games - left.games || left.team.localeCompare(right.team))
    .slice(0, 50);

  const players = [...playerAggregates.entries()]
    .filter(([, aggregate]) => aggregate.scores.length >= 5)
    .map(([player, aggregate]) => ({
      player,
      games: aggregate.scores.length,
      draft_score: Number((aggregate.scores.reduce((sum, score) => sum + score, 0) / aggregate.scores.length).toFixed(4)),
      role: mostCommon(aggregate.roles),
      team: mostCommon(aggregate.teams),
    }))
    .sort((left, right) => right.draft_score - left.draft_score || right.games - left.games || left.player.localeCompare(right.player))
    .slice(0, 50);

  return {
    teams,
    players,
    evidenceGames,
    scope: records.window_days > 0 ? "profile_window" : "whole_archive",
  };
}
