"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
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
  type PlayerMetadata,
  type PlayerRating,
  type PlayerRecord,
  type PlayerWeeklyRanks,
  type TeamRating,
  type TeamRecord,
  type TeamWeeklyRanks,
  type CompetitionTier,
} from "@/lib/pack";
import { evidenceFields, evidenceInfo, formatEvidenceCell } from "@/lib/evidence";
import { filterDraftRankings, type DraftPlayerRow, type DraftRankingsScope, type DraftTeamRow } from "@/lib/draftRankings";
import { DataBars, type DataBarRow } from "./DataBars";
import { TeamMark } from "./TeamMark";
import styles from "./SignalRatings.module.css";

export type { DraftPlayerRow, DraftTeamRow } from "@/lib/draftRankings";

type Props = {
  draftTeams: DraftTeamRow[];
  draftPlayers: DraftPlayerRow[];
  draftScope: DraftRankingsScope;
  draftWindowDays: number | null;
  draftEvidenceGames: number | null;
  teams: TeamRating[];
  players: PlayerRating[];
  teamRecords: Record<string, TeamRecord>;
  teamWeeklyRanks: TeamWeeklyRanks;
  playerRecords: Record<string, PlayerRecord>;
  playerWeeklyRanks: PlayerWeeklyRanks;
  playerMetadata: Record<string, PlayerMetadata>;
  availableLeaguesByTier: Record<CompetitionTier, string[]>;
  championImages: Record<string, string>;
  playerChampionPicks: Record<string, ChampionPick[]>;
  recentForms: { teams: Record<string, boolean[]>; players: Record<string, boolean[]> };
  teamChampionPicks: Record<string, ChampionPick[]>;
};

type ChampionPick = {
  champion: string;
  label: string;
};

type Tab = "teams" | "players" | "draft";
type Sort = "rating" | "movement" | "games" | "name";

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
  if (delta == null || delta === 0) return "—";
  return delta > 0 ? `+${delta}` : `${delta}`;
}

function parseLeagues(value: string | null): string[] {
  return value?.split(",").map((item) => item.trim()).filter(Boolean) ?? [];
}

function FormStrip({ form }: { form: boolean[] }) {
  if (!form.length) return <span className={styles.formEmpty}>Awaiting recent games</span>;
  return (
    <span className={styles.formStrip} aria-label={`Recent form: ${form.map((won) => won ? "win" : "loss").join(", ")}`}>
      {form.map((won, index) => <i key={index} className={won ? styles.formWin : styles.formLoss}>{won ? "W" : "L"}</i>)}
    </span>
  );
}

function ChampionStrip({ picks, images }: { picks: ChampionPick[]; images: Record<string, string> }) {
  const slots = Array.from({ length: 5 }, (_, index) => picks[index] ?? null);
  return (
    <span className={styles.championStrip} aria-label={`Best champions: ${picks.map((pick) => pick.champion).join(", ") || "unavailable"}`}>
      {slots.map((pick, index) => pick && images[pick.champion] ? (
        // CommunityDragon supplies the champion portraits in the published pack.
        // eslint-disable-next-line @next/next/no-img-element
        <img key={`${pick.champion}-${index}`} src={images[pick.champion]} alt={pick.champion} title={pick.label} loading="lazy" />
      ) : <i key={`empty-${index}`} className={styles.championEmpty} aria-label="Champion record unavailable" />)}
    </span>
  );
}

export function SignalRatings({
  draftTeams,
  draftPlayers,
  draftScope,
  draftWindowDays,
  draftEvidenceGames,
  teams,
  players,
  teamRecords,
  teamWeeklyRanks,
  playerRecords,
  playerWeeklyRanks,
  playerMetadata,
  availableLeaguesByTier,
  championImages,
  playerChampionPicks,
  recentForms,
  teamChampionPicks,
}: Props) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [tab, setTab] = useState<Tab>((searchParams.get("tab") as Tab) === "players" || (searchParams.get("tab") as Tab) === "draft" ? (searchParams.get("tab") as Tab) : "teams");
  const [query, setQuery] = useState(searchParams.get("q") ?? "");
  const [leagues, setLeagues] = useState<string[]>(() => {
    const parsed = parseLeagues(searchParams.get("leagues"));
    return parsed.length ? parsed : ["TIER1"];
  });
  const [role, setRole] = useState(searchParams.get("role") ?? searchParams.get("draftRole") ?? "");
  const [minGames, setMinGames] = useState(Math.max(5, Number(searchParams.get("min") ?? 20)));
  const [draftMinGames, setDraftMinGames] = useState(Math.max(5, Number(searchParams.get("draftMin") ?? 5)));
  const [sort, setSort] = useState<Sort>((searchParams.get("sort") as Sort) || "rating");
  const [expanded, setExpanded] = useState(false);
  const [selected, setSelected] = useState("");

  useEffect(() => {
    const params = new URLSearchParams();
    if (tab !== "teams") params.set("tab", tab);
    if (query.trim()) params.set("q", query.trim());
    if (leagues.length) params.set("leagues", leagues.join(","));
    if ((tab === "players" || tab === "draft") && role) params.set(tab === "draft" ? "draftRole" : "role", role);
    if (tab === "players" && minGames !== 20) params.set("min", String(minGames));
    if (tab === "draft" && draftMinGames !== 5) params.set("draftMin", String(draftMinGames));
    if (sort !== "rating") params.set("sort", sort);
    const suffix = params.toString();
    router.replace(suffix ? `${pathname}?${suffix}` : pathname, { scroll: false });
  }, [tab, query, leagues, role, minGames, draftMinGames, sort, pathname, router]);

  const teamDelta = (team: string) => teamWeeklyRanks.by_team[team]?.delta;
  const playerDelta = (player: string) => {
    const record = playerRecords[player];
    const tier = record?.current_tier ?? "all";
    return playerWeeklyRanks.by_player[player]?.[tier]?.delta ?? playerWeeklyRanks.by_player[player]?.all?.delta;
  };

  const filteredTeams = useMemo(() => {
    const list = teams.filter((team) => isActiveRating(team) && teamMatchesQuery(team.team, query) && recordMatchesLeagues(teamRecords[team.team], leagues));
    return [...list].sort((a, b) => {
      if (sort === "name") return a.team.localeCompare(b.team);
      if (sort === "games") return (b.n_maps ?? 0) - (a.n_maps ?? 0);
      if (sort === "movement") return (teamDelta(b.team) ?? -999) - (teamDelta(a.team) ?? -999);
      return adjustedRating(b, TEAM_SIGMA_MIN) - adjustedRating(a, TEAM_SIGMA_MIN);
    });
  // Weekly rank maps are immutable inputs for this render.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [teams, teamRecords, leagues, query, sort, teamWeeklyRanks]);

  const filteredPlayers = useMemo(() => {
    const list = players
      .filter(isActiveRating)
      .filter((player) => player.n_maps >= minGames)
      .filter((player) => {
        const record = playerRecords[player.player];
        return !role || (record?.primary_role ?? record?.roles?.[0]) === role;
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
  // Weekly rank maps are immutable inputs for this render.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [players, playerRecords, teamRecords, leagues, query, role, minGames, sort, playerWeeklyRanks]);

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
  const entityName = (entity: TeamRating | PlayerRating) => tab === "teams" ? (entity as TeamRating).team : (entity as PlayerRating).player;
  const ratingOf = (entity: TeamRating | PlayerRating) => tab === "teams"
    ? adjustedRating(entity as TeamRating, TEAM_SIGMA_MIN)
    : softMu((entity as PlayerRating).mu_total, (entity as PlayerRating).sigma, PLAYER_SIGMA_MIN);
  const deltaOf = (name: string) => tab === "teams" ? teamDelta(name) : playerDelta(name);

  const featured = entities.find((entity) => entityName(entity) === selected) ?? entities[0];
  const featuredName = featured ? entityName(featured) : "";
  const featuredRecord = featuredName ? (tab === "teams" ? teamRecords[featuredName] : playerRecords[featuredName]) : undefined;
  const featuredForm = recentForms[tab === "draft" ? "players" : tab][featuredName] ?? [];
  const featuredTrust = featured
    ? evidenceInfo(
        evidenceFields(featured as unknown as Record<string, unknown>),
        featured.sigma,
        tab === "teams" ? (featuredRecord as TeamRecord | undefined)?.games : (featured as PlayerRating).n_maps,
      )
    : null;
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
    : (featuredRecord as PlayerRecord | undefined)?.current_team ?? (featured as PlayerRating | undefined)?.last_team;

  const setTier = (tier: string | null) => {
    setLeagues((current) => {
      const selectedCurrentLeagues = current.filter((league) => !league.startsWith("TIER"));
      if (!tier) return selectedCurrentLeagues;
      const allowed = new Set(availableLeaguesByTier[tier.toLowerCase() as CompetitionTier]);
      return [...selectedCurrentLeagues.filter((league) => allowed.has(league)), tier];
    });
    setExpanded(false);
  };

  const toggleLeague = (league: string) => {
    setLeagues((current) => current.includes(league) ? current.filter((item) => item !== league) : [...current, league]);
    setExpanded(false);
  };

  const changeTab = (value: Tab) => {
    if (value === tab) return;
    setTab(value);
    setExpanded(false);
    setSelected("");
  };

  const draftScopeLabel = draftScope === "whole_archive"
    ? "Whole accepted archive"
    : draftWindowDays ? `Published ${draftWindowDays}-day window` : "Published profile window";
  const draftEvidenceLabel = draftEvidenceGames ? `${draftEvidenceGames.toLocaleString()} games with draft evidence` : "Published draft evidence";
  const formatDraftWinShare = (value: number) => `${Math.round(value * 100)}%`;
  const formatBestAvailableRate = (value: number | null | undefined) => value == null ? "—" : `${Math.round(value * 100)}%`;
  const draftShareBarWidth = (value: number) => Math.min(100, Math.abs(value - 0.5) * 300);
  const bestAvailableRateBarWidth = (value: number | null | undefined) => value == null ? 0 : Math.min(100, value * 100);

  const ratingChartRows: DataBarRow[] = entities.slice(0, 8).map((entity) => {
    const name = entityName(entity);
    const record = tab === "teams" ? teamRecords[name] : playerRecords[name];
    const team = tab === "teams" ? name : (record as PlayerRecord | undefined)?.current_team ?? (entity as PlayerRating).last_team;
    const value = ratingOf(entity);
    const detail = tab === "teams"
      ? `${(record as TeamRecord | undefined)?.games ?? entity.n_maps ?? 0} games · ${formatAffiliation((record as TeamRecord | undefined)?.current_tier, (record as TeamRecord | undefined)?.current_league ?? (record as TeamRecord | undefined)?.primary)}`
      : `${(entity as PlayerRating).n_maps} games · ${roleLabel((record as PlayerRecord | undefined)?.primary_role)}`;
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
    value: row.draft_win_share,
    valueLabel: formatDraftWinShare(row.draft_win_share),
    detail: `${row.games} games${row.league ? ` · ${row.league}` : ""}`,
    mark: <TeamMark team={row.team} />,
    tone: row.draft_win_share >= 0.5 ? "positive" : "negative",
  }));
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
          <div className={styles.tabs} aria-label="Rating type">
            {(["teams", "players", "draft"] as const).map((value) => (
              <button key={value} type="button" className={tab === value ? styles.tabActive : ""} aria-pressed={tab === value} onClick={() => changeTab(value)}>
                {value === "teams" ? "Teams" : value === "players" ? "Players" : "Draft"}
              </button>
            ))}
          </div>
          <label className={styles.search}><span>Search</span><input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder={tab === "teams" ? "Team or alias" : tab === "players" ? "Player or team" : "Team or player"} /></label>
          {tab === "players" || tab === "draft" ? (
            <label><span>{tab === "draft" ? "Player role" : "Role"}</span><select value={role} onChange={(event) => setRole(event.target.value)}><option value="">All roles</option>{PLAYER_ROLES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
          ) : null}
          {tab === "draft" ? null : <label><span>Sort</span><select value={sort} onChange={(event) => setSort(event.target.value as Sort)}><option value="rating">Rating</option><option value="movement">Movement</option><option value="games">Games</option><option value="name">Name</option></select></label>}
          {tab === "players" ? <label className={styles.minGames}><span>Min games</span><input type="number" min={5} value={minGames} onChange={(event) => setMinGames(Math.max(5, Number(event.target.value) || 5))} /></label> : null}
          {tab === "draft" ? <label className={styles.minGames}><span>Min games / picks</span><input type="number" min={5} value={draftMinGames} onChange={(event) => setDraftMinGames(Math.max(5, Number(event.target.value) || 5))} /></label> : null}
        </div>
        <div className={styles.scopeRow} role="group" aria-label="Competition level">
          <span>Level</span>
          <button type="button" className={!selectedTiers.length ? styles.scopeActive : ""} onClick={() => setTier(null)}>All</button>
          {TIER_FILTERS.map((tier) => <button key={tier.value} type="button" className={leagues.includes(tier.value) ? styles.scopeActive : ""} onClick={() => setTier(tier.value)}>{tier.label}</button>)}
          <details className={styles.leagueFilter}><summary>Regions / leagues {selectedLeagues.length ? `(${selectedLeagues.length})` : ""}</summary><div data-native-scroll>{leagueGroups.map((group) => <section key={group.value}><strong>{group.label}</strong><div>{group.leagues.map((league) => <button key={league} type="button" className={leagues.includes(league) ? styles.scopeActive : ""} onClick={() => toggleLeague(league)}>{league}</button>)}</div></section>)}</div></details>
        </div>
      </section>

      {tab === "draft" ? (
        draftRows.teams.length || draftRows.players.length ? <>
          <p className={styles.draftNote}>{draftScopeLabel} · {draftEvidenceLabel}. Teams use average published draft win share. Player rows use the best-available rate: the share of evaluated picks that were the highest-ranked unbanned champion available for the role.</p>
          <section className={styles.draftVisuals} aria-label="Draft visualizations">
            <DataBars
              title="Team draft win share"
              description="Average published estimate from the scored draft"
              rows={draftTeamChartRows}
              domain={{ min: 0, max: 1 }}
              baseline={0.5}
              baselineLabel="50% even"
              axisLeft="0%"
              axisRight="100%"
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
                <header><div><h2>Teams by draft</h2><p>Average published draft win share · per-game estimate</p></div><b>{draftRows.teams.length} teams</b></header>
                {draftRows.teams.length ? <ol className={styles.draftList}>
                  {(expanded ? draftRows.teams : draftRows.teams.slice(0, 18)).map((row, index) => <li key={`${row.team}-${row.league ?? "all"}-${row.tier ?? "all"}`}>
                    <span className={styles.cardRank}>{String(index + 1).padStart(2, "0")}</span>
                    <span className={styles.draftIdentity}><TeamMark team={row.team} /><Link className="row-link" href={`/elo/team/${teamSlug(row.team)}`}>{row.team}</Link></span>
                    <span className={styles.draftMetric}><b title="Average published draft win share">{formatDraftWinShare(row.draft_win_share)}</b><span className={styles.draftBarTrack} aria-hidden="true"><span className={row.draft_win_share >= 0.5 ? styles.draftBarPositive : styles.draftBarNegative} style={{ width: `${draftShareBarWidth(row.draft_win_share)}%` }} /></span></span>
                    <small>{row.games} games</small>
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
        </> : <p className={styles.empty}>No published draft evidence meets these filters.</p>
      ) : featured ? (
        <>
          <section className={styles.summaryLine} aria-label="Rating summary">
            <p><span>Ranked</span><strong>{entities.length} {tab}</strong></p>
            <p><span>Leader</span><strong>{entityName(entities[0])}</strong><small>{topScore?.toFixed(0)}</small></p>
            <p><span>Scope</span><strong>{scope}</strong></p>
          </section>

          <section className={styles.focusBoard}>
            <aside className={styles.rankingRail}>
              <header><span>Leaders</span><b>→</b></header>
              {entities.slice(0, 5).map((entity, index) => {
                const name = entityName(entity);
                const entityTeam = tab === "teams" ? name : playerRecords[name]?.current_team ?? (entity as PlayerRating).last_team;
                return <button type="button" key={name} className={name === featuredName ? styles.railActive : ""} onClick={() => setSelected(name)}><span>{String(index + 1).padStart(2, "0")}</span><span className={styles.railIdentity}><TeamMark team={entityTeam} /><strong>{name}</strong></span><b>{ratingOf(entity).toFixed(0)}</b></button>;
              })}
            </aside>

            <article className={styles.featured}>
              <header>
                <div className={styles.featureIdentity}><TeamMark team={featuredTeam} size="large" /><div><h2>{tab === "players" && playerMetadata[featuredName]?.flag ? `${playerMetadata[featuredName].flag} ` : ""}{featuredName}</h2><p>{tab === "teams" ? formatAffiliation((featuredRecord as TeamRecord | undefined)?.current_tier, (featuredRecord as TeamRecord | undefined)?.current_league ?? (featuredRecord as TeamRecord | undefined)?.primary) : `${(featuredRecord as PlayerRecord | undefined)?.current_team ?? (featured as PlayerRating).last_team ?? "Independent"} · ${roleLabel((featuredRecord as PlayerRecord | undefined)?.primary_role)}`}</p></div></div>
                <dl><div><dt>Rating</dt><dd>{ratingOf(featured).toFixed(0)}</dd></div><div><dt>Movement</dt><dd className={(deltaOf(featuredName) ?? 0) > 0 ? styles.positive : (deltaOf(featuredName) ?? 0) < 0 ? styles.negative : ""}>{rankDeltaLabel(deltaOf(featuredName))}</dd></div></dl>
              </header>
              <div className={styles.featureBody}>
                <div className={styles.featureForm}><span>Best champions</span><ChampionStrip picks={(tab === "teams" ? teamChampionPicks : playerChampionPicks)[featuredName] ?? []} images={championImages} /><span>Recent form</span><FormStrip form={featuredForm} /></div>
              </div>
              <footer>
                <span>{featuredTrust ? formatEvidenceCell(featuredTrust) : "Confidence unavailable"}</span>
                <span>{tab === "teams" ? `${(featuredRecord as TeamRecord | undefined)?.games ?? featured.n_maps ?? 0} games · ${formatWr((featuredRecord as TeamRecord | undefined)?.wr)}` : `${(featured as PlayerRating).n_maps} games · ${formatWr((featuredRecord as PlayerRecord | undefined)?.wr)}`}</span>
                <Link href={tab === "teams" ? `/elo/team/${teamSlug(featuredName)}` : `/elo/player/${playerSlug(featuredName)}`}>Open profile →</Link>
              </footer>
            </article>

            <aside className={styles.rankingRail}>
              <header><span>Movers</span><b>Δ</b></header>
              {(movers.length ? movers : entities.slice(0, 5)).map((entity) => {
                const name = entityName(entity);
                const entityTeam = tab === "teams" ? name : playerRecords[name]?.current_team ?? (entity as PlayerRating).last_team;
                return <button type="button" key={name} className={name === featuredName ? styles.railActive : ""} onClick={() => setSelected(name)}><span className={(deltaOf(name) ?? 0) > 0 ? styles.positive : (deltaOf(name) ?? 0) < 0 ? styles.negative : ""}>{rankDeltaLabel(deltaOf(name))}</span><span className={styles.railIdentity}><TeamMark team={entityTeam} /><strong>{name}</strong></span><b>{ratingOf(entity).toFixed(0)}</b></button>;
              })}
            </aside>
          </section>

          <DataBars
            title={`${tab === "teams" ? "Team" : "Player"} rating field`}
            description={`Top ${ratingChartRows.length} ${tab} in ${scope.toLowerCase()} by adjusted rating`}
            rows={ratingChartRows}
            domain={ratingDomain}
            axisLeft={ratingDomain ? `${ratingDomain.min}` : "—"}
            axisRight={ratingDomain ? `${ratingDomain.max}` : "—"}
          />

          <section className={styles.gallerySection}>
            <header><h2>{tab === "teams" ? "Team ratings" : "Player ratings"}</h2><p>{entities.length} shown · {scope}</p></header>
            <div className={styles.gallery}>
              {visible.map((entity, index) => {
                const name = entityName(entity);
                const record = tab === "teams" ? teamRecords[name] : playerRecords[name];
                const form = recentForms[tab][name] ?? [];
                const entityTeam = tab === "teams" ? name : (record as PlayerRecord | undefined)?.current_team ?? (entity as PlayerRating).last_team;
                return (
                  <Link className={styles.ratingCard} href={tab === "teams" ? `/elo/team/${teamSlug(name)}` : `/elo/player/${playerSlug(name)}`} key={name}>
                    <span className={styles.cardRank}>{String(index + 1).padStart(2, "0")}</span>
                    <div className={styles.cardIdentity}><TeamMark team={entityTeam} size="medium" /><div><h3>{tab === "players" && playerMetadata[name]?.flag ? `${playerMetadata[name].flag} ` : ""}{name}</h3><p>{tab === "teams" ? formatAffiliation((record as TeamRecord | undefined)?.current_tier, (record as TeamRecord | undefined)?.current_league ?? (record as TeamRecord | undefined)?.primary) : `${(record as PlayerRecord | undefined)?.current_team ?? (entity as PlayerRating).last_team ?? "Independent"} · ${roleLabel((record as PlayerRecord | undefined)?.primary_role)}`}</p></div></div>
                    <strong>{ratingOf(entity).toFixed(0)}</strong>
                    <span className={(deltaOf(name) ?? 0) > 0 ? styles.positive : (deltaOf(name) ?? 0) < 0 ? styles.negative : ""}>{rankDeltaLabel(deltaOf(name))}</span>
                    <div className={styles.cardVisuals}><ChampionStrip picks={(tab === "teams" ? teamChampionPicks : playerChampionPicks)[name] ?? []} images={championImages} /><FormStrip form={form} /></div>
                  </Link>
                );
              })}
            </div>
            {entities.length > 18 ? <button type="button" className={styles.showAll} onClick={() => setExpanded((current) => !current)}>{expanded ? "Show first 18" : `Show all ${entities.length}`}</button> : null}
          </section>
        </>
      ) : <p className={styles.empty}>No {tab} match these filters.</p>}
    </div>
  );
}
