import {
  INTL_LEAGUES,
  REGION_LEAGUES,
  SECONDARY_REGIONAL_LEAGUES,
  teamQueryAliases,
  type ProfileRecords,
} from "@/lib/pack";
import { hasCompleteDraftEvidence } from "./draftRankings";

type DraftTier = "tier1" | "tier2" | "tier3" | "international" | "all";

export type TeamDraftRow = {
  team: string;
  average_edge: number;
  games: number;
  best_edge: number;
  worst_edge: number;
  positive_edge_rate: number | null;
};

export type TeamDraftComparison = {
  teams: [TeamDraftRow, TeamDraftRow];
  winner: string | null;
  direction: "higher" | "lower";
  difference: number;
};

export type TeamDraftQueryResult = {
  kind: "team_draft_query" | "team_draft_comparison";
  direction: "asc" | "desc";
  tier: DraftTier;
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
  positiveEdges: number;
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

function inferLeagueTier(records: ProfileRecords, league: string): Exclude<DraftTier, "all"> {
  const normalizedLeague = normalize(league);
  if (INTL_LEAGUES.some((candidate) => normalize(candidate) === normalizedLeague)) return "international";
  if (SECONDARY_REGIONAL_LEAGUES.some((candidate) => normalize(candidate) === normalizedLeague)) return "tier2";
  const observedTier = Object.values(records.games)
    .find((game) => normalize(game.league) === normalizedLeague && game.competition_tier)?.competition_tier;
  if (observedTier === "tier2" || observedTier === "tier3" || observedTier === "international") return observedTier;
  return "tier1";
}

function detectTier(question: string, records: ProfileRecords, league: string | null): TeamDraftQueryResult["tier"] {
  if (/\ball tiers\b/i.test(question)) return "all";
  const match = /\btier\s*([123])\b/i.exec(question);
  if (match) return `tier${match[1]}` as "tier1" | "tier2" | "tier3";
  if (/\binternational\b/i.test(question)) return "international";
  return league ? inferLeagueTier(records, league) : "tier1";
}

function detectLeague(question: string, availableLeagues: string[]): string | null {
  const normalizedQuestion = ` ${normalize(question)} `;
  const aliases: Array<{ canonical: string; names: string[] }> = [
    { canonical: "LEC", names: ["emea", "europe", "european", "eu"] },
    { canonical: "LCS", names: ["americas", "north america", "north american", "na"] },
  ];
  for (const alias of aliases) {
    if (alias.names.some((name) => normalizedQuestion.includes(` ${normalize(name)} `))) return alias.canonical;
  }
  const candidates = [...new Set([
    ...availableLeagues,
    ...REGION_LEAGUES,
    ...INTL_LEAGUES,
    "LTA",
  ])]
    .filter((league) => normalize(league).length > 0)
    .sort((left, right) => normalize(right).length - normalize(left).length);
  return candidates.find((league) => normalizedQuestion.includes(` ${normalize(league)} `)) ?? null;
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

function addDraft(aggregates: Map<string, DraftAggregate>, team: string, score: number): void {
  const aggregate = aggregates.get(team) ?? { scores: [], positiveEdges: 0 };
  aggregate.scores.push(score);
  if (score > 0) aggregate.positiveEdges += 1;
  aggregates.set(team, aggregate);
}

function average(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function edge(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)} model units`;
}

function edgeGap(first: number, second: number): string {
  return `${Math.abs(first - second).toFixed(2)} model-unit`;
}

function scopeLabel(tier: TeamDraftQueryResult["tier"], league: string | null): string {
  const tierLabel = tier === "all" ? "all tiers" : tier.replace("tier", "Tier ");
  return league ? `${league}, ${tierLabel}` : tierLabel;
}

function toDraftRow(team: string, aggregate: DraftAggregate): TeamDraftRow {
  return {
    team,
    average_edge: average(aggregate.scores),
    games: aggregate.scores.length,
    best_edge: Math.max(...aggregate.scores),
    worst_edge: Math.min(...aggregate.scores),
    positive_edge_rate: aggregate.scores.length
      ? aggregate.positiveEdges / aggregate.scores.length
      : null,
  };
}

function collectDraftAggregates(
  records: ProfileRecords,
  tier: TeamDraftQueryResult["tier"],
  league: string | null,
): Map<string, DraftAggregate> {
  const aggregates = new Map<string, DraftAggregate>();
  for (const game of Object.values(records.games)) {
    const contribution = game.draft_contribution;
    const blue = contribution?.blue.signal;
    const red = contribution?.red.signal;
    if (
      !hasCompleteDraftEvidence(game)
      ||
      (tier !== "all" && game.competition_tier !== tier)
      || (league && normalize(game.league) !== normalize(league))
      || contribution?.status !== "available"
      || !finite(blue)
      || !finite(red)
    ) continue;
    const draftEdge = blue - red;
    addDraft(aggregates, game.blue_team, draftEdge);
    addDraft(aggregates, game.red_team, -draftEdge);
  }
  return aggregates;
}

function comparisonAnswer(
  first: TeamDraftRow | undefined,
  second: TeamDraftRow | undefined,
  firstTeam: string,
  secondTeam: string,
  tier: TeamDraftQueryResult["tier"],
  league: string | null,
  minimumGames: number,
  windowDays: number,
  direction: "higher" | "lower",
): TeamDraftQueryResult {
  const scope = scopeLabel(tier, league);
  const basis = `Compared ${firstTeam} and ${secondTeam} in ${scope} by average descriptive draft edge, using at least ${minimumGames} complete ${minimumGames === 1 ? "draft" : "drafts"} in the active ${windowDays}-day profile window. Historical here means this profile window, not all seasons.`;
  const caveat = "Draft edge is a descriptive composition signal in model units. It is separate from match results, team rating, and calibrated probability.";

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

  const difference = first.average_edge - second.average_edge;
  const winner = difference === 0
    ? null
    : direction === "higher"
      ? difference > 0 ? first.team : second.team
      : difference < 0 ? first.team : second.team;
  const edgeLeader = winner ? (winner === first.team ? first : second) : null;
  const edgeOther = winner ? (winner === first.team ? second : first) : null;
  const leaguePhrase = league ? ` in ${league}` : "";
  const relation = edgeLeader && edgeOther
    ? `${edgeLeader.team} has the ${direction} average descriptive draft edge${leaguePhrase} in the active ${windowDays}-day profile window at ${edge(edgeLeader.average_edge)} across ${edgeLeader.games} games, versus ${edge(edgeOther.average_edge)} for ${edgeOther.team} across ${edgeOther.games} games. The gap is ${edgeGap(edgeLeader.average_edge, edgeOther.average_edge)} for ${edgeLeader.team}.`
    : `${first.team} and ${second.team} have the same average descriptive draft edge${leaguePhrase} in the active ${windowDays}-day profile window at ${edge(first.average_edge)} across ${first.games} and ${second.games} games.`;

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
  const availableLeagues = Array.from(new Set(Object.values(records.games).map((game) => game.league)));
  const league = detectLeague(question, availableLeagues);
  const tier = detectTier(question, records, league);
  const teams = Array.from(new Set(Object.values(records.games).flatMap((game) => [game.blue_team, game.red_team])));
  const comparisonNames = asksForComparison(question) ? namedTeams(question, teams) : [];
  const team = comparisonNames.length < 2 ? namedTeam(question, teams) : null;
  const minimumGames = detectMinimumGames(question) ?? (comparisonNames.length >= 2 ? 3 : team ? 1 : 3);
  const aggregates = collectDraftAggregates(records, tier, league);

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
      league,
      minimumGames,
      records.window_days,
      /\b(?:worse|lower)\b/i.test(question) ? "lower" : "higher",
    );
  }

  const rows = Array.from(aggregates.entries())
    .filter(([name, aggregate]) => (!team || name === team) && aggregate.scores.length >= minimumGames)
    .map(([name, aggregate]) => toDraftRow(name, aggregate))
    .sort((left, right) => {
      const edgeOrder = left.average_edge - right.average_edge;
      if (edgeOrder !== 0) return direction === "asc" ? edgeOrder : -edgeOrder;
      if (left.games !== right.games) return right.games - left.games;
      return left.team.localeCompare(right.team);
    });

  const visibleRows = rows.slice(0, detectLimit(question));
  const first = visibleRows[0];
  const rankWord = direction === "asc" ? "lowest" : "highest";
  const scope = scopeLabel(tier, league);
  const leaguePhrase = league ? ` in ${league}` : "";
  const headline = first
    ? team
      ? `${first.team}'s average descriptive draft edge is ${edge(first.average_edge)} in ${scope} across ${first.games} games.`
      : `${first.team} has the ${rankWord} average descriptive draft edge${leaguePhrase} at ${edge(first.average_edge)} across ${first.games} games.`
    : team
      ? `No available draft scores meet the requested sample for ${team}${league ? ` in ${league}` : ""}.`
      : "Team draft score rankings are unavailable for the active release.";

  return {
    kind: "team_draft_query",
    direction,
    tier,
    answer: {
      headline,
      basis: `Ranked ${rows.length} ${rows.length === 1 ? "team" : "teams"} by average descriptive draft edge in ${scope} with at least ${minimumGames} complete ${minimumGames === 1 ? "draft" : "drafts"} in the active ${records.window_days}-day profile window.`,
      caveat: "Draft edge is a descriptive composition signal in model units. It is separate from match results, team rating, and calibrated probability.",
    },
    rows: visibleRows,
    proof: { sources: [SOURCE], resultCount: rows.length },
  };
}
