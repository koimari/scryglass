"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import type { PlayerRating, PlayerRecord, TeamRating, TeamRecord } from "@/lib/pack";
import {
  formatTrustCell,
  formatWr,
  INTL_LEAGUES,
  PLAYER_SIGMA_MIN,
  playerMatchesQuery,
  playerSlug,
  recordMatchesLeagues,
  REGION_LEAGUES,
  scopedTeamWr,
  softMu,
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
  availableLeagues: string[];
};

type TeamCol = "team" | "league" | "soft" | "mu" | "meta" | "trust" | "wr";
type PlayerCol = "player" | "last_team" | "league" | "soft" | "mu" | "trust" | "games";
type Dir = "asc" | "desc";

const CHIP_ORDER = [...REGION_LEAGUES, "INTL", ...INTL_LEAGUES];

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
  availableLeagues,
}: Props) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const [tab, setTab] = useState<"teams" | "players">(
    searchParams.get("tab") === "players" ? "players" : "teams",
  );
  const [q, setQ] = useState(searchParams.get("q") || "");
  const [leagues, setLeagues] = useState<string[]>(() => parseLeagues(searchParams.get("leagues")));
  const [minGames, setMinGames] = useState(Number(searchParams.get("min") || 20));
  const [expanded, setExpanded] = useState(false);
  const [teamCol, setTeamCol] = useState<TeamCol>("soft");
  const [teamDir, setTeamDir] = useState<Dir>("desc");
  const [playerCol, setPlayerCol] = useState<PlayerCol>("soft");
  const [playerDir, setPlayerDir] = useState<Dir>("desc");

  const chips = useMemo(() => {
    const present = new Set(availableLeagues);
    const core = ["LCK", "LPL", "LEC", "LCS", "LTA", "CBLOL", "PCS", "VCS"];
    const shown = [
      ...core.filter((L) => availableLeagues.includes(L) || present.has(L)),
      ...REGION_LEAGUES.filter((L) => !(core as readonly string[]).includes(L) && availableLeagues.includes(L)),
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
      return recordMatchesLeagues(teamRecords[t.team], leagues);
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
            softMu(a.mu_total, a.sigma, TEAM_SIGMA_MIN) -
            softMu(b.mu_total, b.sigma, TEAM_SIGMA_MIN);
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
          const wa = scopedTeamWr(ra, leagues) ?? -1;
          const wb = scopedTeamWr(rb, leagues) ?? -1;
          cmp = wa - wb;
          break;
        }
      }
      return sign * cmp;
    });
    return list;
  }, [teams, q, leagues, teamRecords, teamCol, teamDir]);

  const sortedPlayers = useMemo(() => {
    let list = players
      .filter((p) => (p.n_maps ?? 0) >= minGames)
      .filter((p) => playerMatchesQuery(p.player, p.last_team, q))
      .filter((p) => {
        const rec = playerRecords[p.player];
        const fromTeam = p.last_team ? teamRecords[p.last_team] : undefined;
        return (
          recordMatchesLeagues(rec, leagues) ||
          recordMatchesLeagues(fromTeam, leagues) ||
          (!rec && !fromTeam && !leagues.length)
        );
      });
    // if leagues selected and no player record match via team
    if (leagues.length) {
      list = list.filter((p) => {
        const rec = playerRecords[p.player];
        const fromTeam = p.last_team ? teamRecords[p.last_team] : undefined;
        return recordMatchesLeagues(rec, leagues) || recordMatchesLeagues(fromTeam, leagues);
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
      return sign * cmp;
    });
    return list;
  }, [players, q, minGames, leagues, playerRecords, teamRecords, playerCol, playerDir]);

  const visibleTeams = expanded ? sortedTeams : sortedTeams.slice(0, 20);
  const visiblePlayers = expanded ? sortedPlayers : sortedPlayers.slice(0, 20);

  return (
    <div className="space-y-6">
      <p className="text-sm text-[var(--ink-muted)] max-w-[62ch]">
        One Dual Elo ladder for every region. League chips only filter who appears — they do not
        re-fit the numbers. Default sort is league-aware rating, so a thin spike cannot outrun a
        settled org.
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
              min={0}
              value={minGames}
              onChange={(e) => setMinGames(Number(e.target.value) || 0)}
            />
          </label>
        )}
      </div>

      <div className="league-chips" role="group" aria-label="League filter">
        <button
          type="button"
          className={`chip ${leagues.length === 0 ? "is-on" : ""}`}
          onClick={() => setLeagues([])}
        >
          All
        </button>
        {chips.map((lg) => (
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

      {tab === "teams" ? (
        <>
          <div className="table-scroll elo-desktop">
            <table className="data-table">
              <thead>
                <tr>
                  <th>#</th>
                  <SortTh label="Team" col="team" active={teamCol === "team"} dir={teamDir} onSort={onTeamSort} />
                  <SortTh
                    label="League"
                    col="league"
                    active={teamCol === "league"}
                    dir={teamDir}
                    title="Primary regional league in the pack (INTL tournaments listed on the profile)."
                    onSort={onTeamSort}
                  />
                  <SortTh
                    label="League-aware"
                    col="soft"
                    active={teamCol === "soft"}
                    dir={teamDir}
                    align="num"
                    title="Raw Elo with a soft penalty when Trust is still Thin. Default sort. See Method."
                    onSort={onTeamSort}
                  />
                  <SortTh
                    label="Raw Elo"
                    col="mu"
                    active={teamCol === "mu"}
                    dir={teamDir}
                    align="num"
                    title="Full Dual Elo rating (regional + international)."
                    onSort={onTeamSort}
                  />
                  <SortTh
                    label="Intl."
                    col="meta"
                    active={teamCol === "meta"}
                    dir={teamDir}
                    align="num"
                    title="International component (MSI / EWC / Worlds / FST)."
                    onSort={onTeamSort}
                  />
                  <SortTh
                    label="Trust"
                    col="trust"
                    active={teamCol === "trust"}
                    dir={teamDir}
                    align="num"
                    title="Settled means the rating hit its floor. Thin means it is still moving. See Method."
                    onSort={onTeamSort}
                  />
                  <SortTh
                    label="WR"
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
                  const wr = scopedTeamWr(rec, leagues);
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
                      <td>{rec?.primary ?? "—"}</td>
                      <td className="num">{softMu(t.mu_total, t.sigma, TEAM_SIGMA_MIN).toFixed(1)}</td>
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
                      {rec?.primary ?? "—"} · {formatTrustCell(trust)} · WR{" "}
                      {formatWr(scopedTeamWr(rec, leagues))}
                    </span>
                    <span className="elo-card-rating">
                      {softMu(t.mu_total, t.sigma, TEAM_SIGMA_MIN).toFixed(1)}
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
          <div className="table-scroll elo-desktop">
            <table className="data-table">
              <thead>
                <tr>
                  <th>#</th>
                  <SortTh label="Player" col="player" active={playerCol === "player"} dir={playerDir} onSort={onPlayerSort} />
                  <SortTh label="Team" col="last_team" active={playerCol === "last_team"} dir={playerDir} onSort={onPlayerSort} />
                  <SortTh label="League" col="league" active={playerCol === "league"} dir={playerDir} onSort={onPlayerSort} />
                  <SortTh
                    label="League-aware"
                    col="soft"
                    active={playerCol === "soft"}
                    dir={playerDir}
                    align="num"
                    title="Raw Elo with a soft penalty when Trust is still Thin."
                    onSort={onPlayerSort}
                  />
                  <SortTh label="Raw Elo" col="mu" active={playerCol === "mu"} dir={playerDir} align="num" onSort={onPlayerSort} />
                  <SortTh
                    label="Trust"
                    col="trust"
                    active={playerCol === "trust"}
                    dir={playerDir}
                    align="num"
                    title="Settled is the floor (often 28 for players). Thin means still moving."
                    onSort={onPlayerSort}
                  />
                  <SortTh label="Games" col="games" active={playerCol === "games"} dir={playerDir} align="num" onSort={onPlayerSort} />
                </tr>
              </thead>
              <tbody>
                {visiblePlayers.map((p, i) => {
                  const rec = playerRecords[p.player];
                  const trust = trustInfo(p.sigma, PLAYER_SIGMA_MIN, p.n_maps);
                  const league = rec?.primary ?? teamRecords[p.last_team || ""]?.primary ?? "—";
                  return (
                    <tr
                      key={p.player}
                      className="row-click"
                      onClick={() => router.push(`/elo/player/${playerSlug(p.player)}`)}
                    >
                      <td className="font-mono text-[var(--ink-muted)]">{i + 1}</td>
                      <td className="font-medium">
                        <Link
                          href={`/elo/player/${playerSlug(p.player)}`}
                          className="row-link"
                          onClick={(e) => e.stopPropagation()}
                        >
                          {p.player}
                        </Link>
                      </td>
                      <td>
                        {p.last_team ? (
                          <Link
                            href={`/elo/team/${teamSlug(p.last_team)}`}
                            className="row-link"
                            onClick={(e) => e.stopPropagation()}
                          >
                            {p.last_team}
                          </Link>
                        ) : (
                          "—"
                        )}
                      </td>
                      <td>{league}</td>
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
              const trust = trustInfo(p.sigma, PLAYER_SIGMA_MIN, p.n_maps);
              return (
                <li key={p.player}>
                  <Link href={`/elo/player/${playerSlug(p.player)}`} className="elo-card">
                    <span className="elo-card-rank">#{i + 1}</span>
                    <span className="elo-card-title">{p.player}</span>
                    <span className="elo-card-meta">
                      {p.last_team ?? "—"} · {formatTrustCell(trust)} · {p.n_maps} games
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
              {q.trim() ? ` and “${q.trim()}”` : ""}. Lower min games or clear a filter.
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
