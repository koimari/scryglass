"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import {
  currentMatchDefaults,
  filterMatchResults,
  matchCompetitionLevel,
  matchIncludesTeam,
} from "@/lib/matchFilters";
import {
  teamSlug,
  type MatchSummary,
  type PublicSchedule,
  type ScheduleSeries,
  type ScheduleTournament,
} from "@/lib/pack";
import type { MatchFacets } from "@/lib/publicData";
import { TeamMark } from "./TeamMark";
import styles from "./SignalMatches.module.css";

type MainView = "upcoming" | "results" | "tournaments";
type ResultView = "gallery" | "timeline";

export type MatchResultState = {
  level: string;
  year: string;
  month: string;
  team: string;
  league: string;
};

const DAY_FORMATTER = new Intl.DateTimeFormat("en-GB", { weekday: "long", day: "2-digit", month: "long", timeZone: "UTC" });
const SHORT_DAY_FORMATTER = new Intl.DateTimeFormat("en-GB", { day: "2-digit", month: "short", year: "numeric", timeZone: "UTC" });
const TIME_FORMATTER = new Intl.DateTimeFormat("en-GB", { hour: "2-digit", minute: "2-digit", hour12: false, timeZone: "UTC" });
const MONTH_FORMATTER = new Intl.DateTimeFormat("en-GB", { month: "long", year: "numeric", timeZone: "UTC" });
const INITIAL_VISIBLE = 31;
const LOAD_STEP = 60;
const LEVEL_LABELS: Record<string, string> = {
  tier1: "Tier 1",
  tier2: "Tier 2",
  tier3: "Tier 3",
  international: "International",
  interregional: "Interregional",
};
const LEVEL_ORDER = ["tier1", "international", "interregional", "tier2", "tier3"];

function dayLabel(value: string): string {
  return SHORT_DAY_FORMATTER.format(new Date(value));
}

function timeLabel(value: string): string {
  return TIME_FORMATTER.format(new Date(value));
}

function dateRange(tournament: ScheduleTournament): string {
  const start = dayLabel(`${tournament.start_date}T12:00:00Z`);
  const end = dayLabel(`${tournament.end_date}T12:00:00Z`);
  return start === end ? start : `${start} – ${end}`;
}

function countdownLabel(start: string): string {
  const milliseconds = Date.parse(start) - Date.now();
  if (milliseconds <= 0) return milliseconds > -21_600_000 ? "Live window" : "Started";
  const minutes = Math.max(1, Math.round(milliseconds / 60_000));
  if (minutes < 60) return `in ${minutes}m`;
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  if (hours < 24) return `in ${hours}h${remainingMinutes ? ` ${remainingMinutes}m` : ""}`;
  const days = Math.floor(hours / 24);
  return `in ${days}d ${hours % 24}h`;
}

function ScheduleClock({ series }: { series: ScheduleSeries }) {
  const [countdown, setCountdown] = useState(series.has_time ? "Scheduled" : "Time pending");
  useEffect(() => {
    if (!series.has_time) return;
    const update = () => setCountdown(countdownLabel(series.start_utc));
    update();
    const timer = window.setInterval(update, 60_000);
    return () => window.clearInterval(timer);
  }, [series.has_time, series.start_utc]);
  return <span className={series.status === "live" ? styles.live : ""}>{countdown}</span>;
}

function RegionFilter({ regions, value, onChange }: { regions: string[]; value: string; onChange: (region: string) => void }) {
  return (
    <div className={styles.regionFilter} aria-label="Region filter">
      {["", ...regions].map((region) => (
        <button key={region || "all"} type="button" className={value === region ? styles.active : ""} aria-pressed={value === region} onClick={() => onChange(region)}>
          {region || "All regions"}
        </button>
      ))}
    </div>
  );
}

function UpcomingSeries({ series }: { series: ScheduleSeries }) {
  return (
    <article className={styles.seriesRow}>
      <div className={styles.seriesTime}>
        <time dateTime={series.start_utc}>{series.has_time ? `${timeLabel(series.start_utc)} UTC` : "Time pending"}</time>
        <ScheduleClock series={series} />
      </div>
      <div className={styles.seriesTeams} aria-label={`${series.team1} versus ${series.team2}`}>
        <span><strong>{series.team1}</strong><TeamMark team={series.team1} size="medium" /></span>
        <em>vs</em>
        <span><TeamMark team={series.team2} size="medium" /><strong>{series.team2}</strong></span>
      </div>
      <div className={styles.seriesMeta}>
        <strong>{series.best_of ? `Best of ${series.best_of}` : "Series"}</strong>
        <span>{series.stage || series.region}</span>
      </div>
    </article>
  );
}

function UpcomingView({ schedule }: { schedule: PublicSchedule }) {
  const [region, setRegion] = useState("");
  const [visibleCount, setVisibleCount] = useState(40);
  const regions = useMemo(() => [...new Set(schedule.upcoming.map((series) => series.region))].filter((item) => item !== "Other").sort(), [schedule.upcoming]);
  const filtered = useMemo(() => schedule.upcoming.filter((series) => !region || series.region === region), [region, schedule.upcoming]);
  const visible = useMemo(() => filtered.slice(0, visibleCount), [filtered, visibleCount]);
  const byDay = useMemo(() => {
    const days = new Map<string, Map<string, ScheduleSeries[]>>();
    for (const series of visible) {
      const day = series.start_utc.slice(0, 10);
      const tournaments = days.get(day) ?? new Map<string, ScheduleSeries[]>();
      tournaments.set(series.tournament, [...(tournaments.get(series.tournament) ?? []), series]);
      days.set(day, tournaments);
    }
    return [...days.entries()];
  }, [visible]);

  return (
    <section className={styles.scheduleSurface} aria-label="Upcoming matches">
      <header className={styles.surfaceHeader}>
        <div><h2>Upcoming matches</h2><p>Series times use UTC.</p></div>
        <RegionFilter regions={regions} value={region} onChange={(value) => { setRegion(value); setVisibleCount(40); }} />
      </header>
      {byDay.length ? byDay.map(([day, tournaments]) => (
        <section className={styles.scheduleDay} key={day}>
          <header><time dateTime={day}>{DAY_FORMATTER.format(new Date(`${day}T12:00:00Z`))}</time><span>{[...tournaments.values()].flat().length} series</span></header>
          <div className={styles.dayTournaments}>
            {[...tournaments.entries()].map(([name, series]) => (
              <section className={styles.tournamentSchedule} key={name}>
                <header>
                  <div><h3>{name}</h3><span>{series[0]?.region}</span></div>
                  {series[0]?.tournament_url ? <a href={series[0].tournament_url} target="_blank" rel="noreferrer">Event page</a> : null}
                </header>
                <div>{series.map((item) => <UpcomingSeries key={item.series_id} series={item} />)}</div>
              </section>
            ))}
          </div>
        </section>
      )) : <p className={styles.empty}>No scheduled series match this region.</p>}
      {filtered.length > visible.length ? <button className={styles.more} type="button" onClick={() => setVisibleCount((current) => Math.min(filtered.length, current + 40))}>Show next {Math.min(40, filtered.length - visible.length)} series</button> : null}
      <p className={styles.sourceNote}>Future fixtures: Leaguepedia · cached every six hours</p>
    </section>
  );
}

function gameChampions(game: MatchSummary, limit: number): Array<{ champion: string; image: string | null }> {
  return game.champions.slice(0, limit).map((champion) => ({ champion, image: null }));
}

function gameCount(value: number): string {
  return `${value} ${value === 1 ? "game" : "games"}`;
}

function ChampionLine({ game, images, limit = 10 }: { game: MatchSummary; images: Record<string, string>; limit?: number }) {
  const champions = gameChampions(game, limit);
  return (
    <span className={styles.championLine} role="group" aria-label={`Champions: ${champions.map((item) => item.champion).join(", ")}`}>
      {champions.map((item) => images[item.champion] ? (
        // CommunityDragon supplies these portraits through the published pack.
        // eslint-disable-next-line @next/next/no-img-element
        <img key={`${item.champion}-${images[item.champion]}`} src={images[item.champion]} alt={item.champion} loading="lazy" />
      ) : null)}
    </span>
  );
}

function MatchCard({ game, images, featured = false }: { game: MatchSummary; images: Record<string, string>; featured?: boolean }) {
  const blueWon = game.blue_win === 1;
  return (
    <article className={`${styles.matchCard} ${featured ? styles.matchCardFeatured : ""}`}>
      <header><span>{game.league}</span><time dateTime={game.date}>{dayLabel(game.date)} · {timeLabel(game.date)} UTC</time></header>
      <div className={styles.cardTeams}>
        <Link href={`/elo/team/${teamSlug(game.blue_team)}`} className={blueWon ? styles.winner : ""}><span className={styles.teamIdentity}><TeamMark team={game.blue_team} size={featured ? "medium" : "small"} /><strong>{game.blue_team}</strong></span><b>{blueWon ? "W" : "L"}</b></Link>
        <Link href={`/elo/team/${teamSlug(game.red_team)}`} className={!blueWon ? styles.winner : ""}><span className={styles.teamIdentity}><TeamMark team={game.red_team} size={featured ? "medium" : "small"} /><strong>{game.red_team}</strong></span><b>{!blueWon ? "W" : "L"}</b></Link>
      </div>
      <ChampionLine game={game} images={images} limit={featured ? 10 : 5} />
      <footer><span>{game.grades_available ? `${game.grades_available}/10 grades` : "Grades unavailable"}</span><Link href={`/matches/${encodeURIComponent(game.game_id)}`}>Open game →</Link></footer>
    </article>
  );
}

function ResultsView({
  games,
  championImages,
  facets,
  filters: initialFilters,
  page,
  pageSize,
  serverFiltered,
  total,
}: {
  games: MatchSummary[];
  championImages: Record<string, string>;
  facets: MatchFacets | null;
  filters: MatchResultState | null;
  page: number;
  pageSize: number;
  serverFiltered: boolean;
  total: number;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const defaults = useMemo(() => currentMatchDefaults(), []);
  const [view, setView] = useState<ResultView>(searchParams.get("layout") === "timeline" ? "timeline" : "gallery");
  const [level, setLevel] = useState(initialFilters?.level ?? searchParams.get("level") ?? defaults.level);
  const [league, setLeague] = useState(initialFilters?.league ?? searchParams.get("tournament") ?? "");
  const [year, setYear] = useState(initialFilters?.year ?? searchParams.get("year") ?? defaults.year);
  const [month, setMonth] = useState(initialFilters?.month ?? searchParams.get("month") ?? defaults.month);
  const [team, setTeam] = useState(initialFilters?.team ?? searchParams.get("team") ?? "");
  const [visibleCount, setVisibleCount] = useState(INITIAL_VISIBLE);
  const [requestedPage, setRequestedPage] = useState(page);
  const localLevels = useMemo(() => [...new Set(games.map(matchCompetitionLevel))].sort((left, right) => {
    const leftIndex = LEVEL_ORDER.indexOf(left);
    const rightIndex = LEVEL_ORDER.indexOf(right);
    return (leftIndex < 0 ? 99 : leftIndex) - (rightIndex < 0 ? 99 : rightIndex);
  }), [games]);
  const levels = serverFiltered ? facets?.tiers ?? [] : localLevels;
  const levelGames = useMemo(() => level ? games.filter((game) => matchCompetitionLevel(game) === level) : games, [games, level]);
  const localYears = useMemo(() => [...new Set(levelGames.map((game) => game.date.slice(0, 4)))].sort().reverse(), [levelGames]);
  const years = serverFiltered ? (facets?.years ?? []).map(String).sort().reverse() : localYears;
  const yearGames = useMemo(() => year ? levelGames.filter((game) => game.date.startsWith(`${year}-`)) : levelGames, [levelGames, year]);
  const localMonths = useMemo(() => [...new Set(yearGames.map((game) => game.date.slice(0, 7)))].sort().reverse(), [yearGames]);
  const months = serverFiltered
    ? (facets?.months ?? []).filter((item) => !year || item.startsWith(`${year}-`)).sort().reverse()
    : localMonths;
  const monthGames = useMemo(() => month ? yearGames.filter((game) => game.date.startsWith(`${month}-`)) : yearGames, [yearGames, month]);
  const localTeams = useMemo(() => [...new Set(monthGames.flatMap((game) => [game.blue_team, game.red_team]))].sort(), [monthGames]);
  const teams = serverFiltered ? facets?.teams ?? [] : localTeams;
  const teamGames = useMemo(() => team ? monthGames.filter((game) => matchIncludesTeam(game, team)) : monthGames, [monthGames, team]);
  const localLeagues = useMemo(() => [...new Set(teamGames.map((game) => game.league))].sort(), [teamGames]);
  const leagues = serverFiltered ? facets?.leagues ?? [] : localLeagues;
  const filtered = useMemo(
    () => serverFiltered ? games : filterMatchResults(games, { level, year, month, team, league }),
    [games, level, year, month, team, league, serverFiltered],
  );
  const visibleGames = useMemo(
    () => serverFiltered ? filtered : filtered.slice(0, visibleCount),
    [filtered, serverFiltered, visibleCount],
  );
  const byDay = useMemo(() => {
    const groups = new Map<string, MatchSummary[]>();
    for (const game of visibleGames) groups.set(game.date.slice(0, 10), [...(groups.get(game.date.slice(0, 10)) ?? []), game]);
    return [...groups.entries()];
  }, [visibleGames]);
  const resetVisible = () => {
    setVisibleCount(INITIAL_VISIBLE);
    setRequestedPage(1);
  };

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    params.set("section", "results");
    params.set("layout", view);
    if (level) params.set("level", level); else if (serverFiltered) params.set("level", "ALL"); else params.delete("level");
    if (year) params.set("year", year); else if (serverFiltered) params.set("year", "ALL"); else params.delete("year");
    if (month) params.set("month", month); else if (serverFiltered) params.set("month", "ALL"); else params.delete("month");
    if (team) params.set("team", team); else params.delete("team");
    if (league) params.set("tournament", league); else params.delete("tournament");
    if (serverFiltered && requestedPage > 1) params.set("page", String(requestedPage)); else params.delete("page");
    const suffix = params.toString();
    const next = `${pathname}${suffix ? `?${suffix}` : ""}`;
    const current = `${pathname}${searchParams.toString() ? `?${searchParams.toString()}` : ""}`;
    if (next === current) return;
    if (serverFiltered) router.replace(next, { scroll: false });
    else window.history.replaceState(window.history.state, "", next);
  }, [league, level, month, pathname, requestedPage, router, searchParams, serverFiltered, team, view, year]);

  const resultCount = serverFiltered ? total : filtered.length;
  const pageCount = Math.max(1, Math.ceil(resultCount / pageSize));

  return (
    <section className={styles.resultsSurface} aria-label="Completed match results">
      <section className={styles.controls}>
        <div className={styles.views} aria-label="Result layout">
          {(["gallery", "timeline"] as const).map((value) => <button key={value} type="button" className={view === value ? styles.active : ""} aria-pressed={view === value} onClick={() => { setView(value); resetVisible(); }}>{value.charAt(0).toUpperCase() + value.slice(1)}</button>)}
        </div>
        <div className={styles.filters}>
          <label><span>Level</span><select value={level} onChange={(event) => { setLevel(event.target.value); setTeam(""); setLeague(""); resetVisible(); }}><option value="">All levels</option>{levels.map((item) => <option key={item} value={item}>{LEVEL_LABELS[item] ?? item}</option>)}</select></label>
          <label><span>Year</span><select value={year} onChange={(event) => { setYear(event.target.value); setMonth(""); setTeam(""); setLeague(""); resetVisible(); }}><option value="">2025–present</option>{years.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
          <label><span>Month</span><select value={month} onChange={(event) => { setMonth(event.target.value); setTeam(""); setLeague(""); resetVisible(); }}><option value="">All months</option>{months.map((item) => <option key={item} value={item}>{MONTH_FORMATTER.format(new Date(`${item}-01T12:00:00Z`))}</option>)}</select></label>
          <label><span>Team</span><select value={team} onChange={(event) => { setTeam(event.target.value); setLeague(""); resetVisible(); }}><option value="">All teams</option>{teams.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
          <label><span>Tournament</span><select value={league} onChange={(event) => { setLeague(event.target.value); resetVisible(); }}><option value="">All tournaments</option>{leagues.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
        </div>
      </section>
      {filtered.length ? (
        <>
          <section className={styles.summaryLine} aria-label="Match summary">
            <p><span>Accepted games</span><strong>{resultCount}</strong></p>
            <p><span>Latest result</span><strong>{dayLabel(filtered[0].date)}</strong></p>
            <p><span>Tournament</span><strong>{filtered[0].league}</strong></p>
          </section>
          {view === "gallery" ? <section className={styles.gallery} aria-label="Latest match gallery"><MatchCard game={visibleGames[0]} images={championImages} featured />{visibleGames.slice(1).map((game) => <MatchCard key={game.game_id} game={game} images={championImages} />)}</section> : null}
          {view === "timeline" ? <div className={styles.timeline}>{byDay.map(([day, dayGames]) => <section key={day} className={styles.resultDay}><header><time dateTime={day}>{dayLabel(`${day}T12:00:00Z`)}</time><span>{gameCount(dayGames.length)}</span></header><div>{dayGames.map((game) => <MatchCard key={game.game_id} game={game} images={championImages} />)}</div></section>)}</div> : null}
          {!serverFiltered && filtered.length > visibleGames.length ? <button className={styles.more} type="button" onClick={() => setVisibleCount((current) => Math.min(filtered.length, current + LOAD_STEP))}>Show next {Math.min(LOAD_STEP, filtered.length - visibleGames.length)} games</button> : null}
          {!serverFiltered && visibleGames.length > INITIAL_VISIBLE ? <button className={styles.more} type="button" onClick={resetVisible}>Back to latest {INITIAL_VISIBLE}</button> : null}
          {serverFiltered && pageCount > 1 ? (
            <nav className={styles.pagination} aria-label="Match result pages">
              <button type="button" disabled={requestedPage <= 1} onClick={() => setRequestedPage((current) => Math.max(1, current - 1))}>Previous</button>
              <span>Page {requestedPage} of {pageCount}</span>
              <button type="button" disabled={requestedPage >= pageCount} onClick={() => setRequestedPage((current) => Math.min(pageCount, current + 1))}>Next</button>
            </nav>
          ) : null}
        </>
      ) : <p className={styles.empty}>No accepted games match these filters.</p>}
    </section>
  );
}

function TournamentsView({ schedule }: { schedule: PublicSchedule }) {
  const [region, setRegion] = useState("");
  const regions = useMemo(() => [...new Set(schedule.tournaments.map((item) => item.region))].filter((item) => item !== "Other").sort(), [schedule.tournaments]);
  const filtered = useMemo(() => schedule.tournaments.filter((item) => !region || item.region === region), [region, schedule.tournaments]);
  const groups: Array<{ status: ScheduleTournament["status"]; label: string }> = [
    { status: "current", label: "Current" },
    { status: "upcoming", label: "Upcoming" },
    { status: "past", label: "Recently completed" },
  ];
  return (
    <section className={styles.tournamentDirectory} aria-label="Tournament directory">
      <header className={styles.surfaceHeader}>
        <div><h2>Tournaments</h2><p>Current and scheduled competition.</p></div>
        <RegionFilter regions={regions} value={region} onChange={setRegion} />
      </header>
      {groups.map((group) => {
        const rows = filtered.filter((item) => item.status === group.status);
        if (!rows.length) return null;
        return (
          <section className={styles.tournamentGroup} key={group.status}>
            <header><h3>{group.label}</h3><span>{rows.length}</span></header>
            <div>{rows.map((item) => <a className={styles.tournamentRow} key={item.overview_page} href={item.url} target="_blank" rel="noreferrer"><span className={styles.tournamentDates}>{dateRange(item)}</span><strong>{item.name}</strong><span>{item.region}</span><span>{item.level || "Open"}</span><b>Open</b></a>)}</div>
          </section>
        );
      })}
      <p className={styles.sourceNote}>Tournament dates: Leaguepedia · cached every six hours</p>
    </section>
  );
}

export function SignalMatches({
  games,
  championImages,
  schedule,
  facets = null,
  filters = null,
  page = 1,
  pageSize = INITIAL_VISIBLE,
  serverFiltered = false,
  total = games.length,
}: {
  games: MatchSummary[];
  championImages: Record<string, string>;
  schedule: PublicSchedule | null;
  facets?: MatchFacets | null;
  filters?: MatchResultState | null;
  page?: number;
  pageSize?: number;
  serverFiltered?: boolean;
  total?: number;
}) {
  const hasSchedule = Boolean(schedule?.upcoming.length || schedule?.tournaments.length);
  const searchParams = useSearchParams();
  const requestedView = searchParams.get("section");
  const [view, setView] = useState<MainView>(requestedView === "upcoming" || requestedView === "tournaments" ? requestedView : "results");
  const tabs: Array<{ value: MainView; label: string; count: number }> = [
    { value: "upcoming", label: "Upcoming", count: schedule?.upcoming.length ?? 0 },
    { value: "results", label: "Results", count: serverFiltered ? total : games.length },
    { value: "tournaments", label: "Tournaments", count: schedule?.tournaments.filter((item) => item.status !== "past").length ?? 0 },
  ];

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    params.set("section", view);
    window.history.replaceState(window.history.state, "", `${window.location.pathname}?${params.toString()}`);
  }, [view]);

  const changeView = (value: MainView) => {
    if (value === view) return;
    setView(value);
  };

  return (
    <div className={styles.root}>
      <nav className={styles.mainTabs} aria-label="Match sections">
        {tabs.map((tab) => <button key={tab.value} type="button" disabled={tab.value !== "results" && !hasSchedule} className={view === tab.value ? styles.active : ""} aria-pressed={view === tab.value} onClick={() => changeView(tab.value)}><span>{tab.label}</span><b>{tab.count}</b></button>)}
      </nav>
      {view === "upcoming" && schedule ? <UpcomingView schedule={schedule} /> : null}
      {view === "results" ? <ResultsView games={games} championImages={championImages} facets={facets} filters={filters} page={page} pageSize={pageSize} serverFiltered={serverFiltered} total={total} /> : null}
      {view === "tournaments" && schedule ? <TournamentsView schedule={schedule} /> : null}
      {!hasSchedule && view !== "results" ? <p className={styles.empty}>The next schedule refresh will restore this view.</p> : null}
    </div>
  );
}
