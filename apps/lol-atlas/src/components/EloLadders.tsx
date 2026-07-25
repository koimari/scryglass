"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { listMajorTeams } from "@/lib/duck";
import type { PlayerRating, TeamRating } from "@/lib/pack";
import { eloToWinProb, softMu, teamSlug, type EloCalibration } from "@/lib/pack";

type Props = {
  teams: TeamRating[];
  players: PlayerRating[];
  calibration: EloCalibration | null;
  baseUrl: string;
  years: number[];
};

type Scope = "major" | "all";
type TeamCol = "team" | "soft" | "mu" | "meta" | "sigma" | "wr";
type PlayerCol = "player" | "last_team" | "soft" | "mu" | "sigma" | "maps";
type Dir = "asc" | "desc";

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
    <th className={align === "num" ? "num" : undefined}>
      <button
        type="button"
        className={`sort-th ${active ? "is-active" : ""}`}
        onClick={() => onSort(col)}
        title={title}
        aria-sort={active ? (dir === "asc" ? "ascending" : "descending") : "none"}
      >
        {label}
        <span className="sort-ind" aria-hidden>
          {active ? (dir === "asc" ? " ↑" : " ↓") : ""}
        </span>
      </button>
    </th>
  );
}

export function EloLadders({ teams, players, calibration, baseUrl, years }: Props) {
  const [tab, setTab] = useState<"teams" | "players">("teams");
  const [q, setQ] = useState("");
  const [minMaps, setMinMaps] = useState(20);
  const [scope, setScope] = useState<Scope>("major");
  const [teamCol, setTeamCol] = useState<TeamCol>("soft");
  const [teamDir, setTeamDir] = useState<Dir>("desc");
  const [playerCol, setPlayerCol] = useState<PlayerCol>("soft");
  const [playerDir, setPlayerDir] = useState<Dir>("desc");
  const [majors, setMajors] = useState<Set<string> | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const s = await listMajorTeams(baseUrl, years);
        if (!cancelled) setMajors(s);
      } catch {
        if (!cancelled) setMajors(new Set());
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [baseUrl, years]);

  const meanTeam =
    teams.reduce((s, t) => s + t.mu_total, 0) / Math.max(teams.length, 1);

  const onTeamSort = useCallback((col: string) => {
    const c = col as TeamCol;
    if (c === teamCol) {
      setTeamDir((d) => (d === "desc" ? "asc" : "desc"));
    } else {
      setTeamCol(c);
      setTeamDir(c === "team" ? "asc" : "desc");
    }
  }, [teamCol]);

  const onPlayerSort = useCallback((col: string) => {
    const c = col as PlayerCol;
    if (c === playerCol) {
      setPlayerDir((d) => (d === "desc" ? "asc" : "desc"));
    } else {
      setPlayerCol(c);
      setPlayerDir(c === "player" || c === "last_team" ? "asc" : "desc");
    }
  }, [playerCol]);

  const sortedTeams = useMemo(() => {
    const needle = q.trim().toLowerCase();
    let list = [...teams];
    if (scope === "major" && majors) {
      list = list.filter((t) => majors.has(t.team));
    }
    list = list.filter((t) => !needle || t.team.toLowerCase().includes(needle));
    const sign = teamDir === "asc" ? 1 : -1;
    list.sort((a, b) => {
      let cmp = 0;
      switch (teamCol) {
        case "team":
          cmp = a.team.localeCompare(b.team);
          break;
        case "soft":
          cmp = softMu(a.mu_total, a.sigma) - softMu(b.mu_total, b.sigma);
          break;
        case "mu":
          cmp = a.mu_total - b.mu_total;
          break;
        case "meta":
          cmp = a.mu_meta - b.mu_meta;
          break;
        case "sigma":
          cmp = a.sigma - b.sigma;
          break;
        case "wr": {
          const pa =
            calibration != null ? eloToWinProb(a.mu_total, meanTeam, calibration.team) : 0;
          const pb =
            calibration != null ? eloToWinProb(b.mu_total, meanTeam, calibration.team) : 0;
          cmp = pa - pb;
          break;
        }
      }
      return sign * cmp;
    });
    return list;
  }, [teams, q, scope, majors, teamCol, teamDir, calibration, meanTeam]);

  const sortedPlayers = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const list = [...players]
      .filter((p) => (p.n_maps ?? 0) >= minMaps)
      .filter(
        (p) =>
          !needle ||
          p.player.toLowerCase().includes(needle) ||
          (p.last_team || "").toLowerCase().includes(needle),
      );
    const sign = playerDir === "asc" ? 1 : -1;
    list.sort((a, b) => {
      let cmp = 0;
      switch (playerCol) {
        case "player":
          cmp = a.player.localeCompare(b.player);
          break;
        case "last_team":
          cmp = (a.last_team || "").localeCompare(b.last_team || "");
          break;
        case "soft":
          cmp = softMu(a.mu_total, a.sigma) - softMu(b.mu_total, b.sigma);
          break;
        case "mu":
          cmp = a.mu_total - b.mu_total;
          break;
        case "sigma":
          cmp = a.sigma - b.sigma;
          break;
        case "maps":
          cmp = (a.n_maps ?? 0) - (b.n_maps ?? 0);
          break;
      }
      return sign * cmp;
    });
    return list;
  }, [players, q, minMaps, playerCol, playerDir]);

  return (
    <div className="space-y-6">
      <p className="text-sm text-[var(--ink-muted)] max-w-[68ch]">
        Dual Elo is one shared ladder across regions. Thin leagues can post a high raw rating with
        large uncertainty — default sort is Adj. rating ↓ and Major scope. Click a column header to
        sort (again to flip direction). Hover a header for the short definition.
      </p>

      <div className="filter-bar">
        <div className="flex gap-1">
          <button
            type="button"
            className={tab === "teams" ? "btn-primary" : "status-pill ghost"}
            onClick={() => setTab("teams")}
          >
            Teams
          </button>
          <button
            type="button"
            className={tab === "players" ? "btn-primary" : "status-pill ghost"}
            onClick={() => setTab("players")}
          >
            Players
          </button>
        </div>
        {tab === "teams" && (
          <div className="flex gap-1">
            <button
              type="button"
              className={scope === "major" ? "btn-primary" : "status-pill ghost"}
              onClick={() => setScope("major")}
            >
              Major
            </button>
            <button
              type="button"
              className={scope === "all" ? "btn-primary" : "status-pill ghost"}
              onClick={() => setScope("all")}
            >
              All
            </button>
          </div>
        )}
        <label className="field grow">
          <span>Search</span>
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder={tab === "teams" ? "Team name" : "Player or team"}
          />
        </label>
        {tab === "players" && (
          <label className="field">
            <span>Min maps</span>
            <input
              type="number"
              min={0}
              value={minMaps}
              onChange={(e) => setMinMaps(Number(e.target.value) || 0)}
            />
          </label>
        )}
      </div>

      {tab === "teams" ? (
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>#</th>
                <SortTh
                  label="Team"
                  col="team"
                  active={teamCol === "team"}
                  dir={teamDir}
                  onSort={onTeamSort}
                />
                <SortTh
                  label="Adj. rating"
                  col="soft"
                  active={teamCol === "soft"}
                  dir={teamDir}
                  align="num"
                  title="Uncertainty-adjusted rating: raw − max(0, uncertainty − 25). Default ladder sort."
                  onSort={onTeamSort}
                />
                <SortTh
                  label="Raw rating"
                  col="mu"
                  active={teamCol === "mu"}
                  dir={teamDir}
                  align="num"
                  title="Full Dual Elo μ (regional + international)."
                  onSort={onTeamSort}
                />
                <SortTh
                  label="Intl."
                  col="meta"
                  active={teamCol === "meta"}
                  dir={teamDir}
                  align="num"
                  title="International component (MSI / EWC / Worlds, etc.)."
                  onSort={onTeamSort}
                />
                <SortTh
                  label="Uncertainty"
                  col="sigma"
                  active={teamCol === "sigma"}
                  dir={teamDir}
                  align="num"
                  title="σ — higher means fewer informative maps / less settled rating."
                  onSort={onTeamSort}
                />
                <SortTh
                  label="Est. WR"
                  col="wr"
                  active={teamCol === "wr"}
                  dir={teamDir}
                  align="num"
                  title="Estimated win rate vs the average team on this ladder (Elo→WR calibration)."
                  onSort={onTeamSort}
                />
              </tr>
            </thead>
            <tbody>
              {sortedTeams.map((t, i) => {
                const p =
                  calibration != null
                    ? eloToWinProb(t.mu_total, meanTeam, calibration.team)
                    : null;
                const soft = softMu(t.mu_total, t.sigma);
                return (
                  <tr key={t.team}>
                    <td className="font-mono text-[var(--ink-muted)]">{i + 1}</td>
                    <td className="font-medium">
                      <Link href={`/elo/team/${teamSlug(t.team)}`} className="row-link">
                        {t.team}
                      </Link>
                    </td>
                    <td className="num">{soft.toFixed(1)}</td>
                    <td className="num">{t.mu_total.toFixed(1)}</td>
                    <td className="num">{t.mu_meta.toFixed(1)}</td>
                    <td className="num">{t.sigma.toFixed(1)}</td>
                    <td className="num">{p != null ? `${(100 * p).toFixed(1)}%` : "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {scope === "major" && majors && (
            <p className="empty-hint">
              Showing {sortedTeams.length} major-circuit teams
              {majors.size ? ` · ${majors.size} orgs seen in pack majors` : ""}.
            </p>
          )}
        </div>
      ) : (
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>#</th>
                <SortTh
                  label="Player"
                  col="player"
                  active={playerCol === "player"}
                  dir={playerDir}
                  onSort={onPlayerSort}
                />
                <SortTh
                  label="Team"
                  col="last_team"
                  active={playerCol === "last_team"}
                  dir={playerDir}
                  title="Most recent team on the player snapshot."
                  onSort={onPlayerSort}
                />
                <SortTh
                  label="Adj. rating"
                  col="soft"
                  active={playerCol === "soft"}
                  dir={playerDir}
                  align="num"
                  title="Uncertainty-adjusted rating: raw − max(0, uncertainty − 25)."
                  onSort={onPlayerSort}
                />
                <SortTh
                  label="Raw rating"
                  col="mu"
                  active={playerCol === "mu"}
                  dir={playerDir}
                  align="num"
                  title="Full player Dual Elo μ."
                  onSort={onPlayerSort}
                />
                <SortTh
                  label="Uncertainty"
                  col="sigma"
                  active={playerCol === "sigma"}
                  dir={playerDir}
                  align="num"
                  title="σ — higher means less settled."
                  onSort={onPlayerSort}
                />
                <SortTh
                  label="Maps"
                  col="maps"
                  active={playerCol === "maps"}
                  dir={playerDir}
                  align="num"
                  title="Maps counted in the player Dual Elo sample."
                  onSort={onPlayerSort}
                />
              </tr>
            </thead>
            <tbody>
              {sortedPlayers.slice(0, 200).map((p, i) => (
                <tr key={p.player}>
                  <td className="font-mono text-[var(--ink-muted)]">{i + 1}</td>
                  <td className="font-medium">{p.player}</td>
                  <td>
                    {p.last_team ? (
                      <Link href={`/elo/team/${teamSlug(p.last_team)}`} className="row-link">
                        {p.last_team}
                      </Link>
                    ) : (
                      "—"
                    )}
                  </td>
                  <td className="num">{softMu(p.mu_total, p.sigma).toFixed(1)}</td>
                  <td className="num">{p.mu_total.toFixed(1)}</td>
                  <td className="num">{p.sigma.toFixed(1)}</td>
                  <td className="num">{p.n_maps}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {sortedPlayers.length > 200 && (
            <p className="empty-hint">
              Showing top 200 of {sortedPlayers.length} (raise search / min maps to narrow).
            </p>
          )}
        </div>
      )}
    </div>
  );
}
