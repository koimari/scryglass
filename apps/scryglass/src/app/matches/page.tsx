import type { Metadata } from "next";
import { Suspense } from "react";
import { SignalMatches, type MatchResultState } from "@/components/SignalMatches";
import { currentMatchDefaults } from "@/lib/matchFilters";
import {
  packSourceUpdatedLabel,
  recentProfileGames,
  type MatchIndex,
  type MatchSummary,
  type PublicSchedule,
  type ProfileRecords,
} from "@/lib/pack";
import {
  getMatchFacets,
  getMatches,
  matchSummary,
  queryApiAvailable,
  type MatchFacets,
} from "@/lib/publicData";
import { readPackJson, readPackManifest } from "@/lib/serverPack";
import styles from "./MatchesPage.module.css";

// The active release is a runtime dependency. A build must stay independent
// from a retired or temporarily unavailable data object.
export const dynamic = "force-dynamic";
export const revalidate = 21_600;

export const metadata: Metadata = {
  title: "Matches — Scryglass",
  description: "Upcoming professional series and completed League of Legends games in Scryglass.",
};

type PageProps = {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
};

const PAGE_SIZE = 20;

function first(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}

function pageNumber(value: string | undefined): number {
  const parsed = Number.parseInt(value ?? "", 10);
  return Number.isInteger(parsed) ? Math.min(501, Math.max(1, parsed)) : 1;
}

function boundedText(value: string | undefined): string {
  const trimmed = value?.trim() ?? "";
  return trimmed.length <= 100 ? trimmed : "";
}

function selectedValue(value: string | undefined, options: string[], fallback: string): string {
  if (value === "ALL") return "";
  if (value && options.includes(value)) return value;
  if (value) return "";
  if (!fallback) return "";
  if (options.includes(fallback)) return fallback;
  return options[0] ?? "";
}

function monthRange(month: string): { from?: string; to?: string } {
  if (!/^\d{4}-\d{2}$/.test(month)) return {};
  const [year, monthNumber] = month.split("-").map(Number);
  const next = new Date(Date.UTC(year, monthNumber, 1)).toISOString().slice(0, 10);
  return { from: `${month}-01`, to: next };
}

export default async function MatchesPage({ searchParams }: PageProps) {
  const query = await searchParams;
  const manifest = await readPackManifest();
  const boundedQueries = queryApiAvailable(manifest);
  let facets: MatchFacets | null = null;
  let filters: MatchResultState | null = null;
  let page = 1;
  let total = 0;
  let championImages: Record<string, string> = {};
  const profiles = boundedQueries
    ? null
    : await readPackJson<ProfileRecords>(manifest, "features/profile_records.json");
  let games: MatchSummary[];
  if (boundedQueries) {
    facets = await getMatchFacets(manifest);
    const defaults = currentMatchDefaults();
    const requestedYear = first(query.year);
    const yearOptions = facets.years.map(String).sort().reverse();
    const level = selectedValue(first(query.level), facets.tiers, defaults.level);
    const year = selectedValue(requestedYear, yearOptions, defaults.year);
    const monthFallback = requestedYear === undefined
      ? (facets.months.includes(defaults.month)
          ? defaults.month
          : facets.months.filter((item) => !year || item.startsWith(`${year}-`)).sort().reverse()[0] ?? "")
      : "";
    const month = selectedValue(first(query.month), facets.months, monthFallback);
    const team = boundedText(first(query.team));
    const league = selectedValue(first(query.tournament), facets.leagues, "");
    page = pageNumber(first(query.page));
    filters = { level, year, month, team, league };
    const range = monthRange(month);
    const matchInput = {
      leagues: league ? [league] : [],
      tiers: level ? [level] : [],
      team,
      years: year ? [Number(year)] : [],
      from: range.from,
      to: range.to,
      limit: PAGE_SIZE,
      offset: (page - 1) * PAGE_SIZE,
    };
    let result = await getMatches(manifest, matchInput);
    const lastPage = Math.max(1, Math.ceil(result.total / PAGE_SIZE));
    if (page > lastPage) {
      page = lastPage;
      result = await getMatches(manifest, { ...matchInput, offset: (page - 1) * PAGE_SIZE });
    }
    total = result.total;
    championImages = result.champion_images ?? {};
    games = result.rows.map(matchSummary);
  } else try {
    const archive = await readPackJson<MatchIndex>(manifest, "features/match_index.json");
    games = archive.games;
  } catch {
    games = recentProfileGames(profiles!, 100).map((game) => ({
      game_id: game.game_id,
      date: game.date,
      league: game.league,
      competition_tier: game.competition_tier,
      blue_team: game.blue_team,
      red_team: game.red_team,
      blue_win: game.blue_win,
      champions: game.players.flatMap((player) => player.champion ? [player.champion] : []),
      grades_available: game.players.filter((player) => player.grade?.status === "available").length,
    }));
  }
  if (!boundedQueries) {
    total = games.length;
    championImages = profiles!.champion_images;
  }
  let schedule: PublicSchedule | null = null;
  try {
    schedule = await readPackJson<PublicSchedule>(manifest, "features/schedule.json");
  } catch {
    schedule = null;
  }

  return (
    <div className={styles.page} data-scryglass-release={manifest.pack_id}>
      <header className={styles.header}>
        <div>
          <h1>Matches</h1>
          <p>Upcoming professional series, plus accepted Oracle&apos;s Elixir games since January 2025. Open a result for its roster, champions, KDA, and player grades.</p>
        </div>
        <div className={styles.freshness}>
          <span>Source through</span>
          <strong>{packSourceUpdatedLabel(manifest)}</strong>
        </div>
      </header>

      <Suspense fallback={<p>Loading results…</p>}>
        <SignalMatches
          key={filters ? `${filters.level}|${filters.year}|${filters.month}|${filters.team}|${filters.league}|${page}` : "legacy"}
          games={games}
          championImages={championImages}
          schedule={schedule}
          facets={facets}
          filters={filters}
          page={page}
          pageSize={boundedQueries ? PAGE_SIZE : games.length || PAGE_SIZE}
          serverFiltered={boundedQueries}
          total={total}
        />
      </Suspense>
    </div>
  );
}
