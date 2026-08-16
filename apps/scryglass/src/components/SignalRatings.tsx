"use client";

import Link from "next/link";
import { useEffect, useMemo, useState, type KeyboardEvent } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import {
  adjustedRating,
  formatWr,
  isActiveRating,
  playerMatchesQuery,
  playerSlug,
  recordMatchesLeagues,
  softMu,
  TEAM_SIGMA_MIN,
  PLAYER_SIGMA_MIN,
  teamMatchesQuery,
  teamSlug,
  TIER_FILTERS,
  type PlayerRating,
  type PlayerRecord,
  type TeamRating,
  type TeamRecord,
  type CompetitionTier,
} from "@/lib/pack";
import { filterDraftRankings, type DraftPlayerRow, type DraftRankingsScope, type DraftTeamRow } from "@/lib/draftRankings";
import { DataBars, type DataBarRow } from "./DataBars";
import { TeamMark } from "./TeamMark";
import styles from "./SignalRatings.module.css";

export type { DraftPlayerRow, DraftTeamRow } from "@/lib/draftRankings";

export type TeamRatingView = Pick<
  TeamRating,
  | "team"
  | "mu_total"
  | "mu_base_total"
  | "mu_effective"
  | "momentum_residual"
  | "sigma"
  | "rating_p10"
  | "n_maps"
  | "evidence_active"
>;
export type PlayerRatingView = Pick<
  PlayerRating,
  | "player"
  | "mu_total"
  | "mu_base_total"
  | "mu_effective"
  | "momentum_residual"
  | "sigma"
  | "n_maps"
  | "last_team"
  | "evidence_active"
>;
export type TeamRecordView = Pick<
  TeamRecord,
  "leagues" | "primary" | "intl" | "interregional" | "current_league" | "current_tier" | "current_team" | "games" | "wr" | "by_league" | "by_tier"
>;
export type PlayerRecordView = Pick<
  PlayerRecord,
  "games" | "wr" | "primary_role" | "current_league" | "current_tier" | "current_team"
>;

type Props = {
  loadedTab: RatingsTab;
  draftAuthorized: boolean;
  draftUnavailableReason: string;
  draftTeams: DraftTeamRow[];
  draftPlayers: DraftPlayerRow[];
  draftScope: DraftRankingsScope;
  draftWindowDays: number | null;
  draftEvidenceGames: number | null;
  teams: TeamRatingView[];
  players: PlayerRatingView[];
  teamRecords: Record<string, TeamRecordView>;
  playerRecords: Record<string, PlayerRecordView>;
  movementByName: Record<string, number | null>;
  showInactive: boolean;
  inactiveTotal: number;
  availableLeaguesByTier: Record<CompetitionTier, string[]>;
  total: number;
  initialPage: number;
  pageSize: number;
  serverFiltered: boolean;
};

export type RatingsTab = "teams" | "players" | "draft";
type Sort = "rating" | "movement" | "games" | "name";
const TAB_ORDER: RatingsTab[] = ["teams", "players", "draft"];

const PLAYER_ROLES = [
  ["top", "Top"],
  ["jungle", "Jungle"],
  ["mid", "Mid"],
  ["bot", "Bot"],
  ["support", "Support"],
] as const;

function formatTier(tier: string | null | undefined): string {
  if (tier === "tier1") return "Tier 1";
  if (tier === "tier2") return "Tier 2";
  if (tier === "tier3") return "Tier 3";
  return "All levels";
}

function formatAffiliation(tier: string | null | undefined, league: string | null | undefined): string {
  const tierLabel = formatTier(tier);
  return league ? `${tierLabel} · ${league}` : tierLabel;
}

function roleLabel(role: string | null | undefined): string {
  return PLAYER_ROLES.find(([value]) => value === role)?.[1] ?? "Unknown role";
}

function rankDeltaLabel(delta: number | null | undefined): string {
  if (delta == null) return "Pending";
  if (delta === 0) return "0";
  return delta > 0 ? `+${delta}` : `${delta}`;
}

function parseLeagues(value: string | null): string[] {
  return value?.split(",").map((item) => item.trim()).filter(Boolean) ?? [];
}

export function SignalRatings({
  loadedTab,
  draftAuthorized,
  draftUnavailableReason,
  draftTeams,
  draftPlayers,
  draftScope,
  draftWindowDays,
  draftEvidenceGames,
  teams,
  players,
  teamRecords,
  playerRecords,
  movementByName,
  showInactive,
  inactiveTotal,
  availableLeaguesByTier,
  total,
  initialPage,
  pageSize,
  serverFiltered,
}: Props) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [tab, setTab] = useState<RatingsTab>((searchParams.get("tab") as RatingsTab) === "players" || (searchParams.get("tab") as RatingsTab) === "draft" ? (searchParams.get("tab") as RatingsTab) : "teams");
  const [query, setQuery] = useState(searchParams.get("q") ?? "");
  const [leagues, setLeagues] = useState<string[]>(() => {
    if (searchParams.get("leagues") === "ALL") return [];
    const parsed = parseLeagues(searchParams.get("leagues"));
    return parsed.length ? parsed : ["TIER1"];
  });
  const [role, setRole] = useState(searchParams.get("role") ?? searchParams.get("draftRole") ?? "");
  const [minGames, setMinGames] = useState(Math.max(5, Number(searchParams.get("min") ?? 20)));
  const [draftMinGames, setDraftMinGames] = useState(Math.max(5, Number(searchParams.get("draftMin") ?? 5)));
  const [sort, setSort] = useState<Sort>((searchParams.get("sort") as Sort) || "rating");
  const [expanded, setExpanded] = useState(() => tab === "teams" && serverFiltered);
  const [selected, setSelected] = useState("");
  const [page, setPage] = useState(initialPage);
  const [includeInactive, setIncludeInactive] = useState(showInactive);

  useEffect(() => {
    if (tab !== loadedTab) return;
    const params = new URLSearchParams();
    if (tab !== "teams") params.set("tab", tab);
    if (query.trim()) params.set("q", query.trim());
    if (!leagues.length) params.set("leagues", "ALL");
    else if (!(leagues.length === 1 && leagues[0] === "TIER1")) params.set("leagues", leagues.join(","));
    if ((tab === "players" || tab === "draft") && role) params.set(tab === "draft" ? "draftRole" : "role", role);
    if (tab === "players" && minGames !== 20) params.set("min", String(minGames));
    if (tab === "draft" && draftMinGames !== 5) params.set("draftMin", String(draftMinGames));
    if (includeInactive) params.set("active", "all");
    if (sort !== "rating") params.set("sort", sort);
    if (serverFiltered && page > 1) params.set("page", String(page));
    const suffix = params.toString();
    const next = suffix ? `${pathname}?${suffix}` : pathname;
    const current = `${pathname}${searchParams.toString() ? `?${searchParams.toString()}` : ""}`;
    if (next !== current) router.replace(next, { scroll: false });
  }, [tab, loadedTab, query, leagues, role, minGames, draftMinGames, sort, page, includeInactive, serverFiltered, pathname, router, searchParams]);

  const teamDelta = (team: string) => movementByName[team];
  const playerDelta = (player: string) => movementByName[player];

  const filteredTeams = useMemo(() => {
    const list = teams.filter((team) => (includeInactive || isActiveRating(team)) && teamMatchesQuery(team.team, query) && recordMatchesLeagues(teamRecords[team.team], leagues));
    return [...list].sort((a, b) => {
      if (sort === "name") return a.team.localeCompare(b.team);
      if (sort === "games") return (b.n_maps ?? 0) - (a.n_maps ?? 0);
      if (sort === "movement") return (teamDelta(b.team) ?? -999) - (teamDelta(a.team) ?? -999);
      return adjustedRating(b, TEAM_SIGMA_MIN) - adjustedRating(a, TEAM_SIGMA_MIN);
    });
  // Movement maps are immutable inputs for this render.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [teams, teamRecords, leagues, query, sort, movementByName, includeInactive]);

  const filteredPlayers = useMemo(() => {
    const list = players
      .filter((player) => includeInactive || isActiveRating(player))
      .filter((player) => player.n_maps >= minGames)
      .filter((player) => {
        const record = playerRecords[player.player];
        return !role || record?.primary_role === role;
      })
      .filter((player) => playerMatchesQuery(player.player, playerRecords[player.player]?.current_team ?? player.last_team, query))
      .filter((player) => {
        const record = playerRecords[player.player];
        const team = record?.current_team ?? player.last_team;
        return recordMatchesLeagues(record, leagues) || recordMatchesLeagues(team ? teamRecords[team] : undefined, leagues);
      });
    return [...list].sort((a, b) => {
      if (sort === "name") return a.player.localeCompare(b.player);
      if (sort === "games") return b.n_maps - a.n_maps;
      if (sort === "movement") return (playerDelta(b.player) ?? -999) - (playerDelta(a.player) ?? -999);
      return softMu(b.mu_total, b.sigma, PLAYER_SIGMA_MIN) - softMu(a.mu_total, a.sigma, PLAYER_SIGMA_MIN);
    });
  // Movement maps are immutable inputs for this render.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [players, playerRecords, teamRecords, leagues, query, role, minGames, sort, movementByName, includeInactive]);

  const draftRows = useMemo(() => {
    const rows = filterDraftRankings(
      {
      teams: draftTeams,
      players: draftPlayers,
      scope: draftScope,
      evidenceGames: draftEvidenceGames ?? 0,
      },
      { leagues, role: role || undefined, minGames: draftMinGames },
    );
    return {
      ...rows,
      teams: rows.teams.filter((row) => teamMatchesQuery(row.team, query)),
      players: rows.players.filter((row) => playerMatchesQuery(row.player, row.team ?? null, query)),
    };
  }, [draftTeams, draftPlayers, draftScope, draftEvidenceGames, leagues, role, draftMinGames, query]);

  const entities = tab === "teams" ? filteredTeams : filteredPlayers;
  const entityName = (entity: TeamRatingView | PlayerRatingView) => tab === "teams" ? (entity as TeamRatingView).team : (entity as PlayerRatingView).player;
  const ratingOf = (entity: TeamRatingView | PlayerRatingView) => tab === "teams"
    ? adjustedRating(entity as TeamRatingView, TEAM_SIGMA_MIN)
    : softMu((entity as PlayerRatingView).mu_total, (entity as PlayerRatingView).sigma, PLAYER_SIGMA_MIN);
  const deltaOf = (name: string) => tab === "teams" ? teamDelta(name) : playerDelta(name);

  const featured = entities.find((entity) => entityName(entity) === selected) ?? entities[0];
  const featuredName = featured ? entityName(featured) : "";
  const featuredRecord = featuredName ? (tab === "teams" ? teamRecords[featuredName] : playerRecords[featuredName]) : undefined;
  const visible = expanded ? entities : entities.slice(0, 18);
  const movers = [...entities]
    .filter((entity) => deltaOf(entityName(entity)) != null)
    .sort((a, b) => (deltaOf(entityName(b)) ?? 0) - (deltaOf(entityName(a)) ?? 0))
    .slice(0, 5);
  const selectedTiers = leagues.filter((league) => league.startsWith("TIER"));
  const selectedLeagues = leagues.filter((league) => !league.startsWith("TIER"));
  const selectedTierKey = selectedTiers.length === 1
    ? selectedTiers[0].toLowerCase() as CompetitionTier
    : null;
  const leagueGroups = (selectedTierKey
    ? TIER_FILTERS.filter((tier) => tier.value.toLowerCase() === selectedTierKey)
    : TIER_FILTERS
  ).map((tier) => ({
    ...tier,
    leagues: availableLeaguesByTier[tier.value.toLowerCase() as CompetitionTier],
  })).filter((group) => group.leagues.length);
  const scope = [...selectedTiers.map((tier) => TIER_FILTERS.find((item) => item.value === tier)?.label ?? tier), ...selectedLeagues].join(" · ") || "All levels";
  const topScore = featured ? ratingOf(entities[0]) : null;
  const featuredTeam = tab === "teams"
    ? featuredName
    : (featuredRecord as PlayerRecordView | undefined)?.current_team ?? (featured as PlayerRatingView | undefined)?.last_team;

  const setTier = (tier: string | null) => {
    setLeagues((current) => {
      const selectedCurrentLeagues = current.filter((league) => !league.startsWith("TIER"));
      if (!tier) return selectedCurrentLeagues;
      const allowed = new Set(availableLeaguesByTier[tier.toLowerCase() as CompetitionTier]);
      return [...selectedCurrentLeagues.filter((league) => allowed.has(league)), tier];
    });
    setPage(1);
    setExpanded(false);
  };

  const toggleLeague = (league: string) => {
    setLeagues((current) => current.includes(league) ? current.filter((item) => item !== league) : [...current, league]);
    setPage(1);
    setExpanded(false);
  };

  const changeTab = (value: RatingsTab) => {
    if (value === tab) return;
    setTab(value);
    setExpanded(false);
    setSelected("");
    setPage(1);
    const params = new URLSearchParams(searchParams.toString());
    if (value === "teams") params.delete("tab");
    else params.set("tab", value);
    if (value !== "players") params.delete("role");
    if (value !== "draft") params.delete("draftRole");
    const suffix = params.toString();
    router.push(suffix ? `${pathname}?${suffix}` : pathname, { scroll: false });
  };

  const moveTabFocus = (event: KeyboardEvent<HTMLButtonElement>, current: RatingsTab) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const index = TAB_ORDER.indexOf(current);
    const next = event.key === "Home"
      ? TAB_ORDER[0]
      : event.key === "End"
        ? TAB_ORDER[TAB_ORDER.length - 1]
        : TAB_ORDER[(index + (event.key === "ArrowRight" ? 1 : -1) + TAB_ORDER.length) % TAB_ORDER.length];
    changeTab(next);
    requestAnimationFrame(() => document.getElementById(`ratings-tab-${next}`)?.focus());
  };

  const resetFilters = () => {
    setQuery("");
    setLeagues(["TIER1"]);
    setRole("");
    setMinGames(20);
    setDraftMinGames(5);
    setSort("rating");
    setIncludeInactive(false);
    setExpanded(false);
    setSelected("");
    setPage(1);
  };

  const draftScopeLabel = draftScope === "whole_archive"
    ? "Whole accepted archive"
    : draftWindowDays ? `Published ${draftWindowDays}-day window` : "Published profile window";
  const draftEvidenceLabel = draftEvidenceGames ? `${draftEvidenceGames.toLocaleString()} games with draft evidence` : "Published draft evidence";
  const formatDraftEdge = (value: number) => `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
  const formatBestAvailableRate = (value: number | null | undefined) => value == null ? "—" : `${Math.round(value * 100)}%`;
  const bestAvailableRateBarWidth = (value: number | null | undefined) => value == null ? 0 : Math.min(100, value * 100);

  const ratingChartRows: DataBarRow[] = entities.slice(0, 8).map((entity) => {
    const name = entityName(entity);
    const record = tab === "teams" ? teamRecords[name] : playerRecords[name];
    const team = tab === "teams" ? name : (record as PlayerRecordView | undefined)?.current_team ?? (entity as PlayerRatingView).last_team;
    const value = ratingOf(entity);
    const detail = tab === "teams"
      ? `${(record as TeamRecordView | undefined)?.games ?? entity.n_maps ?? 0} games · ${formatAffiliation((record as TeamRecordView | undefined)?.current_tier, (record as TeamRecordView | undefined)?.current_league ?? (record as TeamRecordView | undefined)?.primary)}`
      : `${(entity as PlayerRatingView).n_maps} games · ${roleLabel((record as PlayerRecordView | undefined)?.primary_role)}`;
    return {
      id: name,
      label: name,
      href: tab === "teams" ? `/elo/team/${teamSlug(name)}` : `/elo/player/${playerSlug(name)}`,
      value,
      valueLabel: value.toFixed(0),
      detail,
      mark: <TeamMark team={team} />,
      tone: "neutral",
    };
  });
  const ratingValues = ratingChartRows.map((row) => row.value);
  const ratingDomain = ratingValues.length ? {
    min: Math.floor((Math.min(...ratingValues) - 25) / 50) * 50,
    max: Math.ceil((Math.max(...ratingValues) + 25) / 50) * 50,
  } : undefined;
  const draftTeamChartRows: DataBarRow[] = draftRows.teams.slice(0, 8).map((row) => ({
    id: `team-${row.team}-${row.league ?? "all"}-${row.tier ?? "all"}`,
    label: row.team,
    href: `/elo/team/${teamSlug(row.team)}`,
    value: row.draft_edge,
    valueLabel: formatDraftEdge(row.draft_edge),
    detail: `${row.games} games${row.league ? ` · ${row.league}` : ""}${row.positive_edge_rate == null ? "" : ` · ${Math.round(row.positive_edge_rate * 100)}% positive edge`}`,
    mark: <TeamMark team={row.team} />,
    tone: row.draft_edge >= 0 ? "positive" : "negative",
  }));
  const draftEdgeValues = draftTeamChartRows.map((row) => row.value);
  const draftEdgeDomain = draftEdgeValues.length ? {
    min: Math.min(0, Math.floor(Math.min(...draftEdgeValues) * 10) / 10),
    max: Math.max(0.1, Math.ceil(Math.max(...draftEdgeValues) * 10) / 10),
  } : undefined;
  const draftPlayerChartRows: DataBarRow[] = draftRows.players
    .slice(0, 8)
    .filter((row): row is typeof row & { best_available_rate: number } => row.best_available_rate != null)
    .map((row) => ({
      id: `player-${row.player}-${row.league ?? "all"}-${row.tier ?? "all"}`,
      label: row.player,
      href: `/elo/player/${playerSlug(row.player)}`,
      value: row.best_available_rate,
      valueLabel: formatBestAvailableRate(row.best_available_rate),
      detail: `${row.games} picks${row.role ? ` · ${roleLabel(row.role)}` : ""}`,
      mark: <TeamMark team={row.team} />,
      tone: "positive" as const,
    }));

  return (
    <div className={styles.root}>
      <section className={styles.controls} aria-label="Rating controls">
        <div className={styles.controlMain}>
          <div className={styles.tabs} role="tablist" aria-label="Rating type">
            {TAB_ORDER.map((value) => (
              <button
                id={`ratings-tab-${value}`}
                key={value}
                type="button"
                role="tab"
                className={tab === value ? styles.tabActive : ""}
                aria-selected={tab === value}
                aria-controls="ratings-tab-panel"
                tabIndex={tab === value ? 0 : -1}
                onClick={() => changeTab(value)}
                onKeyDown={(event) => moveTabFocus(event, value)}
              >
                {value === "teams" ? "Teams" : value === "players" ? "Players" : "Draft"}
              </button>
            ))}
          </div>
          {tab === "draft" && !draftAuthorized ? null : <label className={styles.search}><span>Search</span><input type="search" maxLength={100} value={query} onChange={(event) => { setQuery(event.target.value); setPage(1); }} placeholder={tab === "teams" ? "Team or alias" : tab === "players" ? "Player or team" : "Team or player"} /></label>}
          {(tab === "players" || tab === "draft") && (tab !== "draft" || draftAuthorized) ? (
            <label><span>{tab === "draft" ? "Player role" : "Role"}</span><select value={role} onChange={(event) => { setRole(event.target.value); setPage(1); }}><option value="">All roles</option>{PLAYER_ROLES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
          ) : null}
          {tab === "draft" ? null : <label><span>Sort</span><select value={sort} onChange={(event) => { setSort(event.target.value as Sort); setPage(1); }}><option value="rating">Rating</option><option value="movement">Movement</option><option value="games">Games</option><option value="name">Name</option></select></label>}
          {tab === "players" ? <label className={styles.minGames}><span>Min games</span><input type="number" min={5} max={10000} value={minGames} onChange={(event) => { setMinGames(Math.min(10000, Math.max(5, Number(event.target.value) || 5))); setPage(1); }} /></label> : null}
          {tab === "draft" && draftAuthorized ? <label className={styles.minGames}><span>Min games / picks</span><input type="number" min={5} max={10000} value={draftMinGames} onChange={(event) => setDraftMinGames(Math.min(10000, Math.max(5, Number(event.target.value) || 5)))} /></label> : null}
          {tab !== "draft" ? <label className={styles.includeInactive}><input type="checkbox" checked={includeInactive} onChange={(event) => { setIncludeInactive(event.target.checked); setPage(1); setExpanded(false); }} /><span>Include inactive</span></label> : null}
        </div>
        {tab === "draft" && !draftAuthorized ? null : <div className={styles.scopeRow} role="group" aria-label="Competition level">
          <span>Level</span>
          <button type="button" aria-pressed={!selectedTiers.length} className={!selectedTiers.length ? styles.scopeActive : ""} onClick={() => setTier(null)}>All</button>
          {TIER_FILTERS.map((tier) => <button key={tier.value} type="button" aria-pressed={leagues.includes(tier.value)} className={leagues.includes(tier.value) ? styles.scopeActive : ""} onClick={() => setTier(tier.value)}>{tier.label}</button>)}
          <details className={styles.leagueFilter}><summary>Regions / leagues {selectedLeagues.length ? `(${selectedLeagues.length})` : ""}</summary><div data-native-scroll>{leagueGroups.map((group) => <section key={group.value}><strong>{group.label}</strong><div>{group.leagues.map((league) => <button key={league} type="button" aria-pressed={leagues.includes(league)} className={leagues.includes(league) ? styles.scopeActive : ""} onClick={() => toggleLeague(league)}>{league}</button>)}</div></section>)}</div></details>
        </div>}
      </section>

      <div id="ratings-tab-panel" role="tabpanel" aria-labelledby={`ratings-tab-${tab}`} aria-busy={loadedTab !== tab}>
      {loadedTab !== tab ? (
        <section className={styles.statusState} role="status" aria-live="polite">
          <h2>Loading {tab}</h2>
          <p>Reading the selected rating set.</p>
        </section>
      ) : tab === "draft" && !draftAuthorized ? (
        <section className={styles.statusState} aria-labelledby="draft-unavailable-title">
          <p className={styles.statusKicker}>Release gate</p>
          <h2 id="draft-unavailable-title">Draft Score is unavailable</h2>
          <p>Public Draft Score stays closed until an independent promotion receipt authorizes the model output.</p>
          <p>{draftUnavailableReason}</p>
          <Link href="/methodology#composition-signal">See the research boundary →</Link>
        </section>
      ) : tab === "draft" ? (
        draftRows.teams.length || draftRows.players.length ? <>
          <p className={styles.draftNote}>{draftScopeLabel} · {draftEvidenceLabel}. Team rows use average descriptive draft edge in model units. Player rows use the best-available rate: the share of evaluated picks that were the highest-ranked unbanned champion available for the role.</p>
          <section className={styles.draftVisuals} aria-label="Draft visualizations">
            <DataBars
              title="Team draft edge"
              description="Average descriptive edge from complete drafts"
              rows={draftTeamChartRows}
              domain={draftEdgeDomain}
              baseline={0}
              baselineLabel="0 even"
              axisLeft="lower"
              axisRight="higher"
            />
            <DataBars
              title="Player best-available rate"
              description="Share of picks that matched the highest-ranked unbanned option"
              rows={draftPlayerChartRows}
              domain={{ min: 0, max: 1 }}
              axisMiddle="50%"
              axisLeft="0%"
              axisRight="100%"
            />
          </section>
          <section className={styles.draftSection} aria-label="Draft rankings">
              <div className={styles.draftColumn}>
                <header><div><h2>Teams by draft</h2><p>Average descriptive draft edge · model units</p></div><b>{draftRows.teams.length} teams</b></header>
                {draftRows.teams.length ? <ol className={styles.draftList}>
                  {(expanded ? draftRows.teams : draftRows.teams.slice(0, 18)).map((row, index) => <li key={`${row.team}-${row.league ?? "all"}-${row.tier ?? "all"}`}>
                    <span className={styles.cardRank}>{String(index + 1).padStart(2, "0")}</span>
                    <span className={styles.draftIdentity}><TeamMark team={row.team} /><Link className="row-link" href={`/elo/team/${teamSlug(row.team)}`}>{row.team}</Link></span>
                    <span className={styles.draftMetric}><b title="Average descriptive draft edge in model units">{formatDraftEdge(row.draft_edge)}</b><span className={styles.draftBarTrack} aria-hidden="true"><span className={row.draft_edge >= 0 ? styles.draftBarPositive : styles.draftBarNegative} style={{ width: `${Math.min(100, Math.abs(row.draft_edge) * 24)}%` }} /></span></span>
                    <small>{row.games} games{row.positive_edge_rate == null ? "" : ` · ${Math.round(row.positive_edge_rate * 100)}% positive edge`}</small>
                  </li>)}
                </ol> : <p className={styles.empty}>No team rows meet the evidence floor.</p>}
                {draftRows.teams.length > 18 ? <button type="button" className={styles.showAll} onClick={() => setExpanded((value) => !value)}>{expanded ? "Show fewer" : `Show all ${draftRows.teams.length} teams`}</button> : null}
              </div>
              <div className={styles.draftColumn}>
                <header><div><h2>Players by draft</h2><p>Best-available rate · published bans and tier pool</p></div><b>{draftRows.players.length} players</b></header>
                {draftRows.players.length ? <ol className={styles.draftList}>
                  {(expanded ? draftRows.players : draftRows.players.slice(0, 18)).map((row, index) => <li key={`${row.player}-${row.league ?? "all"}-${row.tier ?? "all"}`}>
                    <span className={styles.cardRank}>{String(index + 1).padStart(2, "0")}</span>
                    <span className={styles.draftIdentity}><TeamMark team={row.team} /><Link className="row-link" href={`/elo/player/${playerSlug(row.player)}`}>{row.player}</Link></span>
                    <span className={styles.draftMetric}><b title="Best-available rate: share of evaluated picks that were the highest-ranked unbanned champion available for the role">{formatBestAvailableRate(row.best_available_rate)}</b><span className={styles.draftBarTrack} aria-hidden="true"><span className={styles.draftBarPositive} style={{ width: `${bestAvailableRateBarWidth(row.best_available_rate)}%` }} /></span></span>
                    <small>{row.games} picks{row.role ? ` · ${roleLabel(row.role)}` : ""}</small>
                  </li>)}
                </ol> : <p className={styles.empty}>No player rows meet the evidence floor.</p>}
                {draftRows.players.length > 18 ? <button type="button" className={styles.showAll} onClick={() => setExpanded((value) => !value)}>{expanded ? "Show fewer" : `Show all ${draftRows.players.length} players`}</button> : null}
              </div>
          </section>
        </> : <section className={styles.statusState}>
          <h2>No draft rows match</h2>
          <p>Try a broader level, league, role, or evidence floor.</p>
          <button type="button" onClick={resetFilters}>Reset filters</button>
        </section>
      ) : featured ? (
        <>
          <section className={styles.summaryLine} aria-label="Rating summary" aria-live="polite">
            <p><span>{includeInactive ? "All records" : tab === "teams" ? "Active teams" : "Active players"}</span><strong>{serverFiltered ? total : entities.length} {tab}</strong><small>{includeInactive ? "active + inactive" : inactiveTotal ? `${inactiveTotal} inactive excluded` : "active ratings"}</small></p>
            <p><span>Leader</span><strong>{entityName(entities[0])}</strong><small>{topScore?.toFixed(0)}</small></p>
            <p><span>Scope</span><strong>{scope}</strong></p>
          </section>

          <section className={styles.focusBoard}>
            <aside className={styles.rankingRail}>
              <header><span>Leaders</span><b>→</b></header>
              {entities.slice(0, 5).map((entity, index) => {
                const name = entityName(entity);
                const entityTeam = tab === "teams" ? name : playerRecords[name]?.current_team ?? (entity as PlayerRatingView).last_team;
                return <button type="button" key={name} className={name === featuredName ? styles.railActive : ""} onClick={() => setSelected(name)}><span>{String(index + 1).padStart(2, "0")}</span><span className={styles.railIdentity}><TeamMark team={entityTeam} /><strong>{name}</strong></span><b>{ratingOf(entity).toFixed(0)}</b></button>;
              })}
            </aside>

            <article className={styles.featured}>
              <header>
                <div className={styles.featureIdentity}><TeamMark team={featuredTeam} size="large" /><div><h2>{featuredName}</h2><p>{tab === "teams" ? formatAffiliation((featuredRecord as TeamRecordView | undefined)?.current_tier, (featuredRecord as TeamRecordView | undefined)?.current_league ?? (featuredRecord as TeamRecordView | undefined)?.primary) : `${(featuredRecord as PlayerRecordView | undefined)?.current_team ?? (featured as PlayerRatingView).last_team ?? "Independent"} · ${roleLabel((featuredRecord as PlayerRecordView | undefined)?.primary_role)}`}</p></div></div>
                <dl><div><dt>Rating</dt><dd>{ratingOf(featured).toFixed(0)}</dd></div><div><dt>Movement</dt><dd className={(deltaOf(featuredName) ?? 0) > 0 ? styles.positive : (deltaOf(featuredName) ?? 0) < 0 ? styles.negative : ""}>{rankDeltaLabel(deltaOf(featuredName))}</dd></div></dl>
              </header>
              <div className={styles.featureBody}>
                <div className={styles.featureSnapshot}>
                  <div><span>Evidence</span><strong>Published and active</strong></div>
                  <div><span>Sample</span><strong>{tab === "teams" ? (featuredRecord as TeamRecordView | undefined)?.games ?? featured.n_maps ?? 0 : (featured as PlayerRatingView).n_maps} games</strong></div>
                </div>
              </div>
              <footer>
                <span>Evidence window</span>
                <span>{tab === "teams" ? `${(featuredRecord as TeamRecordView | undefined)?.games ?? featured.n_maps ?? 0} games · ${formatWr((featuredRecord as TeamRecordView | undefined)?.wr)}` : `${(featured as PlayerRatingView).n_maps} games · ${formatWr((featuredRecord as PlayerRecordView | undefined)?.wr)}`}</span>
                <Link href={tab === "teams" ? `/elo/team/${teamSlug(featuredName)}` : `/elo/player/${playerSlug(featuredName)}`}>Open profile →</Link>
              </footer>
            </article>

            <aside className={styles.rankingRail}>
              <header><span>Movers</span><b>Δ</b></header>
              {(movers.length ? movers : entities.slice(0, 5)).map((entity) => {
                const name = entityName(entity);
                const entityTeam = tab === "teams" ? name : playerRecords[name]?.current_team ?? (entity as PlayerRatingView).last_team;
                return <button type="button" key={name} className={name === featuredName ? styles.railActive : ""} onClick={() => setSelected(name)}><span className={(deltaOf(name) ?? 0) > 0 ? styles.positive : (deltaOf(name) ?? 0) < 0 ? styles.negative : ""}>{rankDeltaLabel(deltaOf(name))}</span><span className={styles.railIdentity}><TeamMark team={entityTeam} /><strong>{name}</strong></span><b>{ratingOf(entity).toFixed(0)}</b></button>;
              })}
            </aside>
          </section>

          <DataBars
            title={`${tab === "teams" ? "Team" : "Player"} adjusted rating`}
            description={`Top ${ratingChartRows.length} ${tab} in ${scope.toLowerCase()} by uncertainty-adjusted rating`}
            rows={ratingChartRows}
            domain={ratingDomain}
            axisLeft={ratingDomain ? `${ratingDomain.min}` : "—"}
            axisRight={ratingDomain ? `${ratingDomain.max}` : "—"}
          />

          <section className={styles.gallerySection}>
            <header><h2>{tab === "teams" ? "Team ratings" : "Player ratings"}</h2><p>{visible.length} of {serverFiltered ? total : entities.length} shown · {scope}</p></header>
            <div className={styles.gallery}>
              {visible.map((entity, index) => {
                const name = entityName(entity);
                const record = tab === "teams" ? teamRecords[name] : playerRecords[name];
                const entityTeam = tab === "teams" ? name : (record as PlayerRecordView | undefined)?.current_team ?? (entity as PlayerRatingView).last_team;
                return (
                  <Link className={styles.ratingCard} href={tab === "teams" ? `/elo/team/${teamSlug(name)}` : `/elo/player/${playerSlug(name)}`} key={name}>
                    <span className={styles.cardRank}>{String(index + 1).padStart(2, "0")}</span>
                    <div className={styles.cardIdentity}><TeamMark team={entityTeam} size="medium" /><div><h3>{name}</h3><p>{tab === "teams" ? formatAffiliation((record as TeamRecordView | undefined)?.current_tier, (record as TeamRecordView | undefined)?.current_league ?? (record as TeamRecordView | undefined)?.primary) : `${(record as PlayerRecordView | undefined)?.current_team ?? (entity as PlayerRatingView).last_team ?? "Independent"} · ${roleLabel((record as PlayerRecordView | undefined)?.primary_role)}`}</p></div></div>
                    <strong>{ratingOf(entity).toFixed(0)}</strong>
                    <span className={(deltaOf(name) ?? 0) > 0 ? styles.positive : (deltaOf(name) ?? 0) < 0 ? styles.negative : ""}>{rankDeltaLabel(deltaOf(name))}</span>
                    <div className={styles.cardVisuals}><span>Evidence window</span><span>{tab === "teams" ? (record as TeamRecordView | undefined)?.games ?? entity.n_maps ?? 0 : (entity as PlayerRatingView).n_maps} games</span></div>
                  </Link>
                );
              })}
            </div>
            {entities.length > 18 ? <button type="button" className={styles.showAll} onClick={() => setExpanded((current) => !current)}>{expanded ? "Show first 18" : `Show all ${entities.length}`}</button> : null}
            {serverFiltered && total > pageSize ? <nav className={styles.pagination} aria-label="Rating pages"><button type="button" disabled={page <= 1} onClick={() => { setPage((value) => Math.max(1, value - 1)); setExpanded(false); }}>Previous</button><span>Page {page} of {Math.ceil(total / pageSize)}</span><button type="button" disabled={page * pageSize >= total} onClick={() => { setPage((value) => value + 1); setExpanded(false); }}>Next</button></nav> : null}
          </section>
        </>
      ) : <section className={styles.statusState}>
        <h2>No {tab} match</h2>
        <p>Try a broader level, league, role, or game floor.</p>
        <button type="button" onClick={resetFilters}>Reset filters</button>
      </section>}
      </div>
    </div>
  );
}
