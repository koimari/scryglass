"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import type {
  PlayerMetadata,
  PlayerRating,
  PlayerRecord,
  PlayerWeeklyRanks,
  TeamRating,
  TeamRecord,
} from "@/lib/pack";
import {
  adjustedRating,
  formatTrustCell,
  formatWr,
  INTL_LEAGUES,
  INTERREGIONAL_LEAGUES,
  MAJOR_REGIONAL_LEAGUES,
  PLAYER_SIGMA_MIN,
  playerMatchesQuery,
  playerSlug,
  recordMatchesLeagues,
  REGION_LEAGUES,
  SECONDARY_REGIONAL_LEAGUES,
  scopedTeamWr,
  softMu,
  TIER_FILTERS,
  TEAM_SIGMA_MIN,
  teamMatchesQuery,
  teamSlug,
  trustInfo,
} from "@/lib/pack";

type Props = {
  teams: TeamRating[];
  players: PlayerRating[];
  teamRecords: Record<string, TeamRecord>;
  playerRecords: Record<string, PlayerRecord>;
  playerWeeklyRanks: PlayerWeeklyRanks;
  playerMetadata: Record<string, PlayerMetadata>;
  availableLeagues: string[];
  dataAsOf: string;
  recentActivityWindowDays: number;
  currentTournaments: Record<string, string>;
};

type TeamCol = "team" | "league" | "soft" | "mu" | "meta" | "trust" | "wr";
type PlayerCol = "player" | "last_team" | "league" | "soft" | "mu" | "trust" | "games";
type Dir = "asc" | "desc";

const CHIP_ORDER = [...REGION_LEAGUES, ...INTERREGIONAL_LEAGUES, "INTL", ...INTL_LEAGUES];

function formatTier(tier: string | null | undefined): string {
  if (tier === "tier1") return "Tier 1";
  if (tier === "tier2") return "Tier 2";
  if (tier === "tier3") return "Tier 3";
  return "—";
}

function formatAffiliation(tier: string | null | undefined, league: string | null | undefined): string {
  if (!league) return "—";
  const tierLabel = formatTier(tier);
  return tierLabel === "—" ? league : `${tierLabel} · ${league}`;
}

function formatScope(scope: string): string {
  return TIER_FILTERS.find((tier) => tier.value === scope)?.label ?? scope;
}

function rankDeltaLabel(delta: number | null | undefined): string {
  if (delta == null || delta === 0) return "—";
  return delta > 0 ? `↑${delta}` : `↓${Math.abs(delta)}`;
}

function rankDeltaClass(delta: number | null | undefined): string {
  if (delta == null || delta === 0) return "text-[var(--ink-faint)]";
  return delta > 0 ? "text-[var(--accent-ink)]" : "text-[var(--danger)]";
}

function SortTh({
  label,
  col,
  active,
  dir,
  align = "left",
  title,
  onSort,
}: {
  label: string;
  col: string;
  active: boolean;
  dir: Dir;
  align?: "left" | "num";
  title?: string;
  onSort: (col: string) => void;
}) {
  return (
    <th
      className={align === "num" ? "num" : undefined}
      scope="col"
      aria-sort={active ? (dir === "asc" ? "ascending" : "descending") : "none"}
    >
      <button
        type="button"
        className={`sort-th ${active ? "is-active" : ""}`}
        onClick={() => onSort(col)}
        title={title}
      >
        {label}
        <span className="sort-ind" aria-hidden>
          {active ? (dir === "asc" ? " ↑" : " ↓") : ""}
        </span>
      </button>
    </th>
  );
}

function parseLeagues(raw: string | null): string[] {
  if (!raw) return [];
  return raw
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

export function EloLadders({
  teams,
  players,
  teamRecords,
  playerRecords,
  playerWeeklyRanks,
  playerMetadata,
  availableLeagues,
  dataAsOf,
  recentActivityWindowDays,
  currentTournaments,
}: Props) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const [tab, setTab] = useState<"teams" | "players">(
    searchParams.get("tab") === "players" ? "players" : "teams",
  );
  const [q, setQ] = useState(searchParams.get("q") || "");
  const [leagues, setLeagues] = useState<string[]>(() => {
    const parsed = parseLeagues(searchParams.get("leagues"));
    return parsed.length ? parsed : ["TIER1"];
  });
  const [minGames, setMinGames] = useState(Math.max(5, Number(searchParams.get("min") || 20)));
  const [expanded, setExpanded] = useState(false);
  const [teamCol, setTeamCol] = useState<TeamCol>("soft");
  const [teamDir, setTeamDir] = useState<Dir>("desc");
  const [playerCol, setPlayerCol] = useState<PlayerCol>("soft");
  const [playerDir, setPlayerDir] = useState<Dir>("desc");

  const chips = useMemo(() => {
    const present = new Set(availableLeagues);
    const core = [...REGION_LEAGUES];
    const shown = [
      ...core.filter((L) => availableLeagues.includes(L) || present.has(L)),
      ...REGION_LEAGUES.filter((L) => !(core as readonly string[]).includes(L) && availableLeagues.includes(L)),
      ...INTERREGIONAL_LEAGUES.filter((L) => availableLeagues.includes(L)),
      "INTL",
      ...INTL_LEAGUES.filter((L) => availableLeagues.includes(L)),
    ];
    return shown.length > 1 ? [...new Set(shown)] : [...CHIP_ORDER];
  }, [availableLeagues]);

  useEffect(() => {
    const params = new URLSearchParams();
    if (tab !== "teams") params.set("tab", tab);
    if (q.trim()) params.set("q", q.trim());
    if (leagues.length) params.set("leagues", leagues.join(","));
    if (tab === "players" && minGames !== 20) params.set("min", String(minGames));
    const qs = params.toString();
    router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
  }, [tab, q, leagues, minGames, pathname, router]);

  const toggleLeague = (lg: string) => {
    setLeagues((prev) => (prev.includes(lg) ? prev.filter((x) => x !== lg) : [...prev, lg]));
    setExpanded(false);
  };

  const onTeamSort = useCallback(
    (col: string) => {
      const c = col as TeamCol;
      if (c === teamCol) setTeamDir((d) => (d === "desc" ? "asc" : "desc"));
      else {
        setTeamCol(c);
        setTeamDir(c === "team" || c === "league" ? "asc" : "desc");
      }
    },
    [teamCol],
  );

  const onPlayerSort = useCallback(
    (col: string) => {
      const c = col as PlayerCol;
      if (c === playerCol) setPlayerDir((d) => (d === "desc" ? "asc" : "desc"));
      else {
        setPlayerCol(c);
        setPlayerDir(
          c === "player" || c === "last_team" || c === "league" ? "asc" : "desc",
        );
      }
    },
    [playerCol],
  );

  const sortedTeams = useMemo(() => {
    let list = teams.filter((t) => {
      if (!teamMatchesQuery(t.team, q)) return false;
      return recordMatchesLeagues(teamRecords[t.team], leagues, {
        dataAsOf,
        recentActivityWindowDays,
        currentTournaments,
      });
    });
    const sign = teamDir === "asc" ? 1 : -1;
    list = [...list].sort((a, b) => {
      let cmp = 0;
      const ra = teamRecords[a.team];
      const rb = teamRecords[b.team];
      switch (teamCol) {
        case "team":
          cmp = a.team.localeCompare(b.team);
          break;
        case "league":
          cmp = (ra?.primary || "").localeCompare(rb?.primary || "");
          break;
        case "soft":
          cmp =
            adjustedRating(a, TEAM_SIGMA_MIN) - adjustedRating(b, TEAM_SIGMA_MIN);
          break;
        case "mu":
          cmp = a.mu_total - b.mu_total;
          break;
        case "meta":
          cmp = a.mu_meta - b.mu_meta;
          break;
        case "trust":
          cmp =
            Math.max(0, a.sigma - TEAM_SIGMA_MIN) - Math.max(0, b.sigma - TEAM_SIGMA_MIN);
          break;
        case "wr": {
          const wa = scopedTeamWr(ra, leagues, { currentTournaments }) ?? -1;
          const wb = scopedTeamWr(rb, leagues, { currentTournaments }) ?? -1;
          cmp = wa - wb;
          break;
        }
      }
      return cmp !== 0 ? sign * cmp : a.team.localeCompare(b.team);
    });
    return list;
  }, [teams, q, leagues, teamRecords, teamCol, teamDir, dataAsOf, recentActivityWindowDays, currentTournaments]);

  const sortedPlayers = useMemo(() => {
    let list = players
      .filter((p) => (p.n_maps ?? 0) >= minGames)
      .filter((p) => {
        const currentTeam = playerRecords[p.player]?.current_team ?? p.last_team;
        return playerMatchesQuery(p.player, currentTeam, q);
      })
      .filter((p) => {
        const rec = playerRecords[p.player];
        const currentTeam = rec?.current_team ?? p.last_team;
        const fromTeam = currentTeam ? teamRecords[currentTeam] : undefined;
        return rec
          ? recordMatchesLeagues(rec, leagues, { dataAsOf, recentActivityWindowDays, currentTournaments })
          : recordMatchesLeagues(fromTeam, leagues, { dataAsOf, recentActivityWindowDays, currentTournaments }) || (!fromTeam && !leagues.length);
      });
    // if leagues selected and no player record match via team
    if (leagues.length) {
      list = list.filter((p) => {
        const rec = playerRecords[p.player];
        const currentTeam = rec?.current_team ?? p.last_team;
        const fromTeam = currentTeam ? teamRecords[currentTeam] : undefined;
        return rec
          ? recordMatchesLeagues(rec, leagues, { dataAsOf, recentActivityWindowDays, currentTournaments })
          : recordMatchesLeagues(fromTeam, leagues, { dataAsOf, recentActivityWindowDays, currentTournaments });
      });
    }
    const sign = playerDir === "asc" ? 1 : -1;
    list = [...list].sort((a, b) => {
      let cmp = 0;
      const ra = playerRecords[a.player];
      const rb = playerRecords[b.player];
      switch (playerCol) {
        case "player":
          cmp = a.player.localeCompare(b.player);
          break;
        case "last_team":
          cmp = (a.last_team || "").localeCompare(b.last_team || "");
          break;
        case "league":
          cmp = (ra?.primary || "").localeCompare(rb?.primary || "");
          break;
        case "soft":
          cmp =
            softMu(a.mu_total, a.sigma, PLAYER_SIGMA_MIN) -
            softMu(b.mu_total, b.sigma, PLAYER_SIGMA_MIN);
          break;
        case "mu":
          cmp = a.mu_total - b.mu_total;
          break;
        case "trust":
          cmp =
            Math.max(0, a.sigma - PLAYER_SIGMA_MIN) -
            Math.max(0, b.sigma - PLAYER_SIGMA_MIN);
          break;
        case "games":
          cmp = (a.n_maps ?? 0) - (b.n_maps ?? 0);
          break;
      }
      return cmp !== 0 ? sign * cmp : a.player.localeCompare(b.player);
    });
    return list;
  }, [
    players,
    q,
    minGames,
    leagues,
    playerRecords,
    teamRecords,
    playerCol,
    playerDir,
    dataAsOf,
    recentActivityWindowDays,
    currentTournaments,
  ]);

  const visibleTeams = expanded ? sortedTeams : sortedTeams.slice(0, 20);
  const visiblePlayers = expanded ? sortedPlayers : sortedPlayers.slice(0, 20);
  const intlSet = new Set<string>(["INTL", ...INTL_LEAGUES]);
  const scopeSummary = leagues.length ? leagues.map(formatScope).join(" + ") : "All competitive tiers and event scopes";
  const rankScopeFor = (tier: string | null | undefined): "all" | "tier1" | "tier2" | "tier3" => {
    const selectedTiers = leagues.filter((scope) => scope.startsWith("TIER")).map((scope) => scope.toLowerCase()) as Array<"tier1" | "tier2" | "tier3">;
    if (selectedTiers.length === 1) return selectedTiers[0];
    if (tier === "tier1" || tier === "tier2" || tier === "tier3") return tier;
    return "all";
  };
  const playerRankDelta = (player: string, tier: string | null | undefined): number | null | undefined => {
    const scope = rankScopeFor(tier);
    return playerWeeklyRanks.by_player[player]?.[scope]?.delta;
  };
  const liveSummary =
    tab === "teams"
      ? `${sortedTeams.length} teams shown in ${scopeSummary}, sorted by ${teamCol === "soft" ? "adjusted rating" : teamCol}`
      : `${sortedPlayers.length} players shown in ${scopeSummary}, sorted by ${playerCol === "soft" ? "adjusted rating" : playerCol}`;

  return (
    <div className="space-y-6">
      <p className="text-sm text-[var(--ink-muted)] max-w-[62ch]">
        Scope the ladder by current competitive tier, regional league, or international event. Selected
        chips define the visible comparison sample and its win rate; each row keeps its published Dual Elo rating.
      </p>
      <p className="method-note max-w-[62ch]">
          Scope: <strong>{scopeSummary}</strong>. Scoped views require an observation within the last {recentActivityWindowDays} days
          of the pack data ({dataAsOf.slice(0, 10)}) and, where labeled, participation in the pack’s current
          tournament for that league. This is a recency and pack-membership guard, not an official registry.
          Adjusted rating = raw rating minus rating spread above the settled floor.
      </p>
      {tab === "players" && (
        <p className="method-note max-w-[62ch]">
          Rank Δ compares with the previous Sunday at 00:00 UTC; flags use Leaguepedia country metadata when available.
        </p>
      )}
      <p className="sr-only" aria-live="polite">
        {liveSummary}
      </p>

      <div className="filter-bar">
        <div className="flex gap-1">
          <button
            type="button"
            className={tab === "teams" ? "btn-primary" : "status-pill ghost"}
            onClick={() => {
              setTab("teams");
              setExpanded(false);
            }}
          >
            Teams
          </button>
          <button
            type="button"
            className={tab === "players" ? "btn-primary" : "status-pill ghost"}
            onClick={() => {
              setTab("players");
              setExpanded(false);
            }}
          >
            Players
          </button>
        </div>
        <label className="field grow">
          <span>Search</span>
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder={tab === "teams" ? "Team or alias (G2, KC…)" : "Player or team"}
          />
        </label>
        {tab === "players" && (
          <label className="field">
            <span>Min games</span>
            <input
              type="number"
              min={5}
              value={minGames}
              onChange={(e) => setMinGames(Math.max(5, Number(e.target.value) || 5))}
            />
          </label>
        )}
      </div>

      <div className="league-filter" role="group" aria-label="Rating scope filters">
        <div className="scope-filter-head">
          <span className="chip-group-label">Scope filters</span>
          <span className="scope-filter-help">Tier sets current competitive level; leagues and events narrow within it. Groups combine with AND.</span>
          <button
            type="button"
            className={`chip ${leagues.length === 0 ? "is-on" : ""}`}
            onClick={() => setLeagues([])}
          >
            All scopes
          </button>
        </div>
        <div className="chip-group chip-group-tier">
          <span className="chip-group-label">Competitive tier</span>
          {TIER_FILTERS.map((tier) => (
            <button
              key={tier.value}
              type="button"
              className={`chip ${leagues.includes(tier.value) ? "is-on" : ""}`}
              onClick={() => toggleLeague(tier.value)}
              title={tier.description}
            >
              {tier.label}
            </button>
          ))}
        </div>
        <div className="chip-group">
          <span className="chip-group-label">Major regional</span>
          {chips
            .filter((lg) => (MAJOR_REGIONAL_LEAGUES as readonly string[]).includes(lg))
            .map((lg) => (
              <button
                key={lg}
                type="button"
                className={`chip ${leagues.includes(lg) ? "is-on" : ""}`}
                onClick={() => toggleLeague(lg)}
              >
                {lg}
              </button>
            ))}
        </div>
        <div className="chip-group">
          <span className="chip-group-label">Other regional</span>
          {chips
            .filter((lg) => (SECONDARY_REGIONAL_LEAGUES as readonly string[]).includes(lg))
            .map((lg) => (
              <button
                key={lg}
                type="button"
                className={`chip ${leagues.includes(lg) ? "is-on" : ""}`}
                onClick={() => toggleLeague(lg)}
              >
                {lg}
              </button>
          ))}
        </div>
        <div className="chip-group">
          <span className="chip-group-label">Cross-region</span>
          {chips
            .filter((lg) => INTERREGIONAL_LEAGUES.includes(lg as (typeof INTERREGIONAL_LEAGUES)[number]))
            .map((lg) => (
              <button
                key={lg}
                type="button"
                className={`chip ${leagues.includes(lg) ? "is-on" : ""}`}
                onClick={() => toggleLeague(lg)}
              >
                {lg}
              </button>
            ))}
        </div>
        <div className="chip-group">
          <span className="chip-group-label">International events</span>
          {chips
            .filter((lg) => intlSet.has(lg) || (INTL_LEAGUES as readonly string[]).includes(lg))
            .map((lg) => (
              <button
                key={lg}
                type="button"
                className={`chip ${leagues.includes(lg) ? "is-on" : ""}`}
                onClick={() => toggleLeague(lg)}
              >
                {lg}
              </button>
            ))}
        </div>
      </div>

      {tab === "teams" ? (
        <>
          <div className="table-scroll elo-desktop" tabIndex={0} aria-label="Team ratings table; scroll horizontally if needed">
            <table className="data-table">
              <caption className="sr-only">
                Team ratings in the current pack. Default order is adjusted rating.
              </caption>
              <thead>
                <tr>
                  <th scope="col">#</th>
                  <SortTh label="Team" col="team" active={teamCol === "team"} dir={teamDir} onSort={onTeamSort} />
                  <SortTh
                    label="League"
                    col="league"
                    active={teamCol === "league"}
                    dir={teamDir}
                    title="Current competition affiliation and tier; international appearances stay on the profile."
                    onSort={onTeamSort}
                  />
                  <SortTh
                    label="Adjusted rating"
                    col="soft"
                    active={teamCol === "soft"}
                    dir={teamDir}
                    align="num"
                    title="One-sided 90% conservative rating bound; wider when the team lacks an international bridge. Default sort. See Method."
                    onSort={onTeamSort}
                  />
                  <SortTh
                    label="Raw rating"
                    col="mu"
                    active={teamCol === "mu"}
                    dir={teamDir}
                    align="num"
                    title="Full rating: regional plus international components."
                    onSort={onTeamSort}
                  />
                  <SortTh
                    label="International"
                    col="meta"
                    active={teamCol === "meta"}
                    dir={teamDir}
                    align="num"
                    title="International component from MSI, EWC, Worlds, or First Stand."
                    onSort={onTeamSort}
                  />
                  <SortTh
                    label="Evidence"
                    col="trust"
                    active={teamCol === "trust"}
                    dir={teamDir}
                    align="num"
                    title="Settled means the estimate reached its floor; thin means it is still moving."
                    onSort={onTeamSort}
                  />
                  <SortTh
                    label="Win rate"
                    col="wr"
                    active={teamCol === "wr"}
                    dir={teamDir}
                    align="num"
                    title="Empirical win rate in the pack for the active league filter."
                    onSort={onTeamSort}
                  />
                </tr>
              </thead>
              <tbody>
                {visibleTeams.map((t, i) => {
                  const rec = teamRecords[t.team];
                  const trust = trustInfo(t.sigma, TEAM_SIGMA_MIN, rec?.games);
                  const wr = scopedTeamWr(rec, leagues, { currentTournaments });
                  return (
                    <tr
                      key={t.team}
                      className="row-click"
                      onClick={() => router.push(`/elo/team/${teamSlug(t.team)}`)}
                    >
                      <td className="font-mono text-[var(--ink-muted)]">{i + 1}</td>
                      <td className="font-medium">
                        <Link
                          href={`/elo/team/${teamSlug(t.team)}`}
                          className="row-link"
                          onClick={(e) => e.stopPropagation()}
                        >
                          {t.team}
                        </Link>
                      </td>
                      <td>
                        {formatAffiliation(rec?.current_tier, rec?.current_league ?? rec?.primary)}
                        {rec?.current_tournament ? <span className="block text-xs text-[var(--ink-faint)]">{rec.current_tournament}</span> : null}
                      </td>
                      <td className="num">{adjustedRating(t, TEAM_SIGMA_MIN).toFixed(1)}</td>
                      <td className="num">{t.mu_total.toFixed(1)}</td>
                      <td className="num">{t.mu_meta.toFixed(1)}</td>
                      <td className="num" title={trust.layman}>
                        {formatTrustCell(trust)}
                      </td>
                      <td className="num">{formatWr(wr)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <ul className="elo-cards elo-mobile">
            {visibleTeams.map((t, i) => {
              const rec = teamRecords[t.team];
              const trust = trustInfo(t.sigma, TEAM_SIGMA_MIN, rec?.games);
              return (
                <li key={t.team}>
                  <Link href={`/elo/team/${teamSlug(t.team)}`} className="elo-card">
                    <span className="elo-card-rank">#{i + 1}</span>
                    <span className="elo-card-title">{t.team}</span>
                    <span className="elo-card-meta">
                      {formatAffiliation(rec?.current_tier, rec?.current_league ?? rec?.primary)} · Evidence {formatTrustCell(trust)} · Win rate{" "}
                      {formatWr(scopedTeamWr(rec, leagues, { currentTournaments }))}
                    </span>
                    <span className="elo-card-rating">
                      {adjustedRating(t, TEAM_SIGMA_MIN).toFixed(1)}
                    </span>
                  </Link>
                </li>
              );
            })}
          </ul>

          {sortedTeams.length === 0 && (
            <p className="empty-hint">
              No teams match these chips
              {q.trim() ? ` and “${q.trim()}”` : ""}. Clear a filter or try another alias.
            </p>
          )}
          {sortedTeams.length > 20 && (
            <p className="empty-hint flex flex-wrap items-center gap-3">
              Showing {visibleTeams.length} of {sortedTeams.length}.
              <button type="button" className="btn-primary" onClick={() => setExpanded((x) => !x)}>
                {expanded ? "Collapse" : "Expand"}
              </button>
            </p>
          )}
        </>
      ) : (
        <>
          <div className="table-scroll elo-desktop" tabIndex={0} aria-label="Player ratings table; scroll horizontally if needed">
            <table className="data-table">
              <caption className="sr-only">
                Player ratings in the current pack. Default order is adjusted rating.
              </caption>
              <thead>
                <tr>
                  <th scope="col">#</th>
                  <th scope="col" title="Rank movement since the previous Sunday 00:00 UTC">Δ</th>
                  <SortTh label="Player" col="player" active={playerCol === "player"} dir={playerDir} onSort={onPlayerSort} />
                  <SortTh label="Team" col="last_team" active={playerCol === "last_team"} dir={playerDir} onSort={onPlayerSort} />
                  <SortTh label="League" col="league" active={playerCol === "league"} dir={playerDir} onSort={onPlayerSort} />
                  <SortTh
                    label="Adjusted rating"
                    col="soft"
                    active={playerCol === "soft"}
                    dir={playerDir}
                    align="num"
                    title="Raw rating with a soft penalty when evidence is still thin."
                    onSort={onPlayerSort}
                  />
                  <SortTh label="Raw rating" col="mu" active={playerCol === "mu"} dir={playerDir} align="num" onSort={onPlayerSort} />
                  <SortTh
                    label="Evidence"
                    col="trust"
                    active={playerCol === "trust"}
                    dir={playerDir}
                    align="num"
                    title="Settled is the evidence floor; thin means the estimate is still moving."
                    onSort={onPlayerSort}
                  />
                  <SortTh label="Games" col="games" active={playerCol === "games"} dir={playerDir} align="num" onSort={onPlayerSort} />
                </tr>
              </thead>
              <tbody>
                {visiblePlayers.map((p, i) => {
                  const rec = playerRecords[p.player];
                  const trust = trustInfo(p.sigma, PLAYER_SIGMA_MIN, p.n_maps);
                  const fromTeam = p.last_team ? teamRecords[p.last_team] : undefined;
                  const league = rec?.current_league ?? rec?.primary ?? fromTeam?.current_league ?? fromTeam?.primary;
                  const currentTeam = rec?.current_team ?? p.last_team;
                  const rankDelta = playerRankDelta(p.player, rec?.current_tier ?? fromTeam?.current_tier);
                  const metadata = playerMetadata[p.player];
                  return (
                    <tr
                      key={p.player}
                      className="row-click"
                      onClick={() => router.push(`/elo/player/${playerSlug(p.player)}`)}
                    >
                      <td className="font-mono text-[var(--ink-muted)]">{i + 1}</td>
                      <td className={`font-mono ${rankDeltaClass(rankDelta)}`} title="Change since the previous Sunday 00:00 UTC">
                        {rankDeltaLabel(rankDelta)}
                      </td>
                      <td className="font-medium">
                        <Link
                          href={`/elo/player/${playerSlug(p.player)}`}
                          className="row-link"
                          onClick={(e) => e.stopPropagation()}
                        >
                          {metadata?.flag ? (
                            <span className="player-country" title={metadata.country || undefined} aria-label={metadata.country || undefined}>
                              {metadata.flag}
                            </span>
                          ) : null}
                          {p.player}
                        </Link>
                      </td>
                      <td>
                        {currentTeam ? (
                          <Link
                            href={`/elo/team/${teamSlug(currentTeam)}`}
                            className="row-link"
                            onClick={(e) => e.stopPropagation()}
                          >
                            {currentTeam}
                          </Link>
                        ) : (
                          "—"
                        )}
                      </td>
                      <td>
                        {formatAffiliation(rec?.current_tier ?? fromTeam?.current_tier, league)}
                        {rec?.current_tournament ? <span className="block text-xs text-[var(--ink-faint)]">{rec.current_tournament}</span> : null}
                      </td>
                      <td className="num">
                        {softMu(p.mu_total, p.sigma, PLAYER_SIGMA_MIN).toFixed(1)}
                      </td>
                      <td className="num">{p.mu_total.toFixed(1)}</td>
                      <td className="num" title={trust.layman}>
                        {formatTrustCell(trust)}
                      </td>
                      <td className="num">{p.n_maps}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <ul className="elo-cards elo-mobile">
            {visiblePlayers.map((p, i) => {
              const rec = playerRecords[p.player];
              const fromTeam = p.last_team ? teamRecords[p.last_team] : undefined;
              const trust = trustInfo(p.sigma, PLAYER_SIGMA_MIN, p.n_maps);
              const currentTeam = rec?.current_team ?? p.last_team;
              const rankDelta = playerRankDelta(p.player, rec?.current_tier ?? fromTeam?.current_tier);
              const metadata = playerMetadata[p.player];
              return (
                <li key={p.player}>
                  <Link href={`/elo/player/${playerSlug(p.player)}`} className="elo-card">
                    <span className="elo-card-rank">#{i + 1} <span className={rankDeltaClass(rankDelta)}>{rankDeltaLabel(rankDelta)}</span></span>
                    <span className="elo-card-title">
                      {metadata?.flag ? <span className="player-country" title={metadata.country || undefined}>{metadata.flag}</span> : null}
                      {p.player}
                    </span>
                    <span className="elo-card-meta">
                      {currentTeam ?? "—"} · {formatAffiliation(rec?.current_tier ?? fromTeam?.current_tier, rec?.current_league ?? rec?.primary ?? fromTeam?.primary)} · Evidence {formatTrustCell(trust)} · {p.n_maps} games
                    </span>
                    <span className="elo-card-rating">
                      {softMu(p.mu_total, p.sigma, PLAYER_SIGMA_MIN).toFixed(1)}
                    </span>
                  </Link>
                </li>
              );
            })}
          </ul>

          {sortedPlayers.length === 0 && (
            <p className="empty-hint">
              No players match these chips
              {q.trim() ? ` and “${q.trim()}”` : ""}. Try a different scope or clear a filter.
            </p>
          )}
          {sortedPlayers.length > 20 && (
            <p className="empty-hint flex flex-wrap items-center gap-3">
              Showing {visiblePlayers.length} of {sortedPlayers.length}.
              <button type="button" className="btn-primary" onClick={() => setExpanded((x) => !x)}>
                {expanded ? "Collapse" : "Expand"}
              </button>
            </p>
          )}
        </>
      )}
    </div>
  );
}
