import type { MatchSummary } from "./pack";

export type MatchResultFilters = {
  level: string;
  year: string;
  month: string;
  team: string;
  league: string;
};

const TIER_ONE_LEAGUES = new Set(["LCK", "LPL", "LEC", "LCS", "CBLOL", "LCP"]);
const INTERNATIONAL_LEAGUES = new Set(["MSI", "EWC", "FST", "WORLDS", "IWC", "MSC", "EM", "ASIA MASTER", "ASIA MASTERS"]);

export function currentMatchDefaults(now = new Date()): Pick<MatchResultFilters, "level" | "year" | "month"> {
  const year = String(now.getUTCFullYear());
  const month = `${year}-${String(now.getUTCMonth() + 1).padStart(2, "0")}`;
  return { level: "tier1", year, month };
}

export function matchCompetitionLevel(game: MatchSummary): string {
  const explicit = String(game.competition_tier ?? "").trim().toLowerCase();
  if (explicit) return explicit;
  const league = game.league.trim().toUpperCase();
  if (TIER_ONE_LEAGUES.has(league)) return "tier1";
  if (INTERNATIONAL_LEAGUES.has(league)) return "international";
  return "tier3";
}

export function matchIncludesTeam(game: MatchSummary, team: string): boolean {
  if (!team) return true;
  const target = team.trim().toLocaleLowerCase();
  return game.blue_team.toLocaleLowerCase() === target || game.red_team.toLocaleLowerCase() === target;
}

export function filterMatchResults(games: MatchSummary[], filters: MatchResultFilters): MatchSummary[] {
  return games.filter((game) => {
    if (filters.level && matchCompetitionLevel(game) !== filters.level) return false;
    if (filters.year && !game.date.startsWith(`${filters.year}-`)) return false;
    if (filters.month && !game.date.startsWith(`${filters.month}-`)) return false;
    if (!matchIncludesTeam(game, filters.team)) return false;
    if (filters.league && game.league !== filters.league) return false;
    return true;
  });
}

export function matchTeamHref(team: string): string {
  const params = new URLSearchParams({ section: "results", team });
  return `/matches?${params.toString()}`;
}
