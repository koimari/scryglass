import { teamQueryAliases, type ProfileRecords } from "@/lib/pack";

export type TeamDraftRow = {
  team: string;
  average_score: number;
  average_win_share: number;
  games: number;
  best_score: number;
  worst_score: number;
};

export type TeamDraftComparison = {
  teams: [TeamDraftRow, TeamDraftRow];
  winner: string | null;
  direction: "higher" | "lower";
  difference: number;
  win_share_gap: number;
};

export type TeamDraftQueryResult = {
  kind: "team_draft_query" | "team_draft_comparison";
  direction: "asc" | "desc";
  tier: "tier1" | "tier2" | "tier3" | "all";
  answer: {
    headline: string;
    basis: string;
    caveat: string;
  };
  rows: TeamDraftRow[];
  comparison?: TeamDraftComparison;
  proof: {
    sources: string[];
    resultCount: number;
  };
};

type DraftAggregate = {
  scores: number[];
  winShares: number[];
};

const SOURCE = "features/profile_records.json";

function normalize(value: string): string {
  return value
    .normalize("NFKD")
    .toLowerCase()
    .replace(/([\p{L}\p{N}])(?:[’']s|s[’'])\b/gu, "$1")
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .replace(/\s+/g, " ");
}

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function detectLimit(question: string): number {
  const match = /\b(?:top|best|bottom|worst)\s+(\d+)\b/i.exec(question);
  if (match) return Math.min(Math.max(Number.parseInt(match[1], 10) || 20, 1), 100);
  return /\ball\b/i.test(question) ? 100 : 20;
}

function detectMinimumGames(question: string): number | null {
  const match = /(?:at least|minimum|min\.?|over)\s+(\d+)\s+(?:drafts?|games?|maps?)/i.exec(question)
    ?? /(\d+)\s*\+\s*(?:drafts?|games?|maps?)/i.exec(question);
  if (!match) return null;
  const value = Number.parseInt(match[1], 10);
  return Number.isFinite(value) && value > 0 ? Math.min(value, 10_000) : null;
}

function detectTier(question: string): TeamDraftQueryResult["tier"] {
  if (/\ball tiers\b/i.test(question)) return "all";
  const match = /\btier\s*([123])\b/i.exec(question);
  return match ? `tier${match[1]}` as "tier1" | "tier2" | "tier3" : "tier1";
}

type TeamMention = {
  position: number;
  specificity: number;
};

function teamMention(question: string, team: string): TeamMention | null {
  const normalizedQuestion = ` ${normalize(question)} `;
  const canonicalKey = normalize(team);
  if (canonicalKey.length >= 2) {
    const position = normalizedQuestion.indexOf(` ${canonicalKey} `);
    if (position >= 0) return { position, specificity: canonicalKey.length + 1000 };
  }
  const aliases = teamQueryAliases(team)
    .map((alias) => normalize(alias))
    .filter((alias) => alias.length >= 2 && alias !== canonicalKey);
  const matches = aliases
    .map((alias) => ({ alias, position: normalizedQuestion.indexOf(` ${alias} `) }))
    .filter(({ position }) => position >= 0)
    .sort((left, right) => right.alias.length - left.alias.length);
  const match = matches[0];
  return match ? { position: match.position, specificity: match.alias.length } : null;
}

function namedTeam(question: string, teams: string[]): string | null {
  const matches = teams
    .map((team) => ({ team, mention: teamMention(question, team) }))
    .filter((entry): entry is { team: string; mention: TeamMention } => Boolean(entry.mention))
    .sort((left, right) => left.mention.position - right.mention.position || right.mention.specificity - left.mention.specificity);
  return matches[0]?.team ?? null;
}

function namedTeams(question: string, teams: string[]): string[] {
  return teams
    .map((team) => ({ team, mention: teamMention(question, team) }))
    .filter((entry): entry is { team: string; mention: TeamMention } => Boolean(entry.mention))
    .sort((left, right) => left.mention.position - right.mention.position || right.mention.specificity - left.mention.specificity)
    .map(({ team }) => team)
    .filter((team, index, matches) => matches.indexOf(team) === index)
    .slice(0, 2);
}

function asksForComparison(question: string): boolean {
  return /\b(?:between|compare|versus|vs\.?|or)\b/i.test(question)
    || /\b(?:better|worse|higher|lower)\b/i.test(question);
}

function addDraft(aggregates: Map<string, DraftAggregate>, team: string, score: number, winShare: number): void {
  const aggregate = aggregates.get(team) ?? { scores: [], winShares: [] };
  aggregate.scores.push(score);
  aggregate.winShares.push(winShare);
  aggregates.set(team, aggregate);
}

function average(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function percentagePointGap(first: number, second: number): string {
  return `${Math.abs(Math.round(first * 100) - Math.round(second * 100))} percentage-point`;
}

function toDraftRow(team: string, aggregate: DraftAggregate): TeamDraftRow {
  return {
    team,
    average_score: average(aggregate.scores),
    average_win_share: average(aggregate.winShares),
    games: aggregate.scores.length,
    best_score: Math.max(...aggregate.scores),
    worst_score: Math.min(...aggregate.scores),
  };
}

function collectDraftAggregates(records: ProfileRecords, tier: TeamDraftQueryResult["tier"]): Map<string, DraftAggregate> {
  const aggregates = new Map<string, DraftAggregate>();
  for (const game of Object.values(records.games)) {
    const contribution = game.draft_contribution;
    const blue = contribution?.blue.signal;
    const red = contribution?.red.signal;
    if (
      (tier !== "all" && game.competition_tier !== tier)
      || contribution?.status !== "available"
      || !finite(blue)
      || !finite(red)
    ) continue;
    const blueWinShare = 1 / (1 + Math.exp(-(blue - red)));
    addDraft(aggregates, game.blue_team, blue, blueWinShare);
    addDraft(aggregates, game.red_team, red, 1 - blueWinShare);
  }
  return aggregates;
}

function comparisonAnswer(
  first: TeamDraftRow | undefined,
  second: TeamDraftRow | undefined,
  firstTeam: string,
  secondTeam: string,
  tier: TeamDraftQueryResult["tier"],
  minimumGames: number,
  windowDays: number,
  direction: "higher" | "lower",
): TeamDraftQueryResult {
  const tierLabel = tier === "all" ? "all tiers" : tier.replace("tier", "Tier ");
  const basis = `Compared ${firstTeam} and ${secondTeam} in ${tierLabel} by average published draft win share, using at least ${minimumGames} available ${minimumGames === 1 ? "draft" : "drafts"} in the active ${windowDays}-day profile window. “Historical” here means this profile window, not all seasons.`;
  const caveat = "Draft win share is the estimated pre-game probability from the picks. It is not the team's match win rate, a stable team rating, or a prediction guarantee.";

  if (!first || !second) {
    return {
      kind: "team_draft_comparison",
      direction: direction === "higher" ? "desc" : "asc",
      tier,
      answer: {
        headline: `The published draft records do not support a complete historical comparison between ${firstTeam} and ${secondTeam}.`,
        basis,
        caveat,
      },
      rows: [first, second].filter((row): row is TeamDraftRow => Boolean(row)),
      proof: { sources: [SOURCE], resultCount: 0 },
    };
  }

  const winShareDifference = first.average_win_share - second.average_win_share;
  const difference = first.average_score - second.average_score;
  const winner = winShareDifference === 0
    ? null
    : direction === "higher"
      ? winShareDifference > 0 ? first.team : second.team
      : winShareDifference < 0 ? first.team : second.team;
  const shareLeader = winner ? (winner === first.team ? first : second) : null;
  const shareOther = winner ? (winner === first.team ? second : first) : null;
  const relation = shareLeader && shareOther
    ? `${shareLeader.team} has the ${direction} average published draft win share in the active ${windowDays}-day profile window at ${percent(shareLeader.average_win_share)} across ${shareLeader.games} games, versus ${percent(shareOther.average_win_share)} for ${shareOther.team} across ${shareOther.games} games. That is a ${percentagePointGap(shareLeader.average_win_share, shareOther.average_win_share)} edge for ${shareLeader.team}.`
    : `${first.team} and ${second.team} have the same average published draft win share in the active ${windowDays}-day profile window at ${percent(first.average_win_share)} across ${first.games} and ${second.games} games.`;

  return {
    kind: "team_draft_comparison",
    direction: direction === "higher" ? "desc" : "asc",
    tier,
    answer: { headline: relation, basis, caveat },
    rows: [first, second],
    comparison: {
      teams: [first, second],
      winner,
      direction,
      difference: Math.abs(difference),
      win_share_gap: Math.abs(winShareDifference),
    },
    proof: { sources: [SOURCE], resultCount: 2 },
  };
}

export function queryTeamDraftScores(records: ProfileRecords, question: string): TeamDraftQueryResult {
  const directionalText = question.replace(/\bat least\b/gi, "");
  const direction: "asc" | "desc" = /\b(?:best|highest|top)\b.*\bto\b.*\b(?:worst|lowest|bottom)\b/i.test(question)
    ? "desc"
    : /\b(?:worst|lowest|bottom)\b.*\bto\b.*\b(?:best|highest|top)\b/i.test(question)
      ? "asc"
      : /\b(worst|lowest|bottom|least)\b/i.test(directionalText) ? "asc" : "desc";
  const tier = detectTier(question);
  const teams = Array.from(new Set(Object.values(records.games).flatMap((game) => [game.blue_team, game.red_team])));
  const comparisonNames = asksForComparison(question) ? namedTeams(question, teams) : [];
  const team = comparisonNames.length < 2 ? namedTeam(question, teams) : null;
  const minimumGames = detectMinimumGames(question) ?? (comparisonNames.length >= 2 ? 3 : team ? 1 : 3);
  const aggregates = collectDraftAggregates(records, tier);

  if (comparisonNames.length >= 2) {
    const [firstTeam, secondTeam] = comparisonNames;
    const first = aggregates.get(firstTeam);
    const second = aggregates.get(secondTeam);
    return comparisonAnswer(
      first && first.scores.length >= minimumGames ? toDraftRow(firstTeam, first) : undefined,
      second && second.scores.length >= minimumGames ? toDraftRow(secondTeam, second) : undefined,
      firstTeam,
      secondTeam,
      tier,
      minimumGames,
      records.window_days,
      /\b(?:worse|lower)\b/i.test(question) ? "lower" : "higher",
    );
  }

  const rows = Array.from(aggregates.entries())
    .filter(([name, aggregate]) => (!team || name === team) && aggregate.scores.length >= minimumGames)
    .map(([name, aggregate]) => toDraftRow(name, aggregate))
    .sort((left, right) => {
      const shareOrder = left.average_win_share - right.average_win_share;
      if (shareOrder !== 0) return direction === "asc" ? shareOrder : -shareOrder;
      const scoreOrder = left.average_score - right.average_score;
      if (scoreOrder !== 0) return direction === "asc" ? scoreOrder : -scoreOrder;
      if (left.games !== right.games) return right.games - left.games;
      return left.team.localeCompare(right.team);
    });

  const visibleRows = rows.slice(0, detectLimit(question));
  const first = visibleRows[0];
  const rankWord = direction === "asc" ? "lowest" : "highest";
  const headline = first
    ? team
      ? `${first.team}'s average published draft win share is ${percent(first.average_win_share)} across ${first.games} games.`
      : `${first.team} has the ${rankWord} average published draft win share at ${percent(first.average_win_share)} across ${first.games} games.`
    : team
      ? `No available draft scores meet the requested sample for ${team}.`
      : "Team draft score rankings are unavailable for the active release.";

  return {
    kind: "team_draft_query",
    direction,
    tier,
    answer: {
      headline,
      basis: `Ranked ${rows.length} ${rows.length === 1 ? "team" : "teams"} by average published draft win share in ${tier === "all" ? "all tiers" : tier.replace("tier", "Tier ")} with at least ${minimumGames} available ${minimumGames === 1 ? "draft" : "drafts"} in the active ${records.window_days}-day profile window.`,
      caveat: "Draft win share is the estimated pre-game probability from the picks. It is not the team's match win rate or a stable team rating.",
    },
    rows: visibleRows,
    proof: { sources: [SOURCE], resultCount: rows.length },
  };
}
