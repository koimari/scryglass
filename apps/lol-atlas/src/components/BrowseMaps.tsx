"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { listLeagues, queryMaps, type QueryRow } from "@/lib/duck";
import { formatClock, formatGold } from "@/lib/format";

type Props = { baseUrl: string; years: number[] };

function encodeGameId(id: string): string {
  return encodeURIComponent(id);
}

export function BrowseMaps({ baseUrl, years }: Props) {
  const yearDefault = years.includes(2026) ? 2026 : years[years.length - 1] ?? 2026;
  const [year, setYear] = useState(yearDefault);
  const [league, setLeague] = useState("");
  const [team, setTeam] = useState("");
  const [teamQuery, setTeamQuery] = useState("");
  const [leagues, setLeagues] = useState<string[]>([]);
  const [rows, setRows] = useState<QueryRow[]>([]);
  const [status, setStatus] = useState("Idle");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const t = window.setTimeout(() => setTeamQuery(team.trim()), 300);
    return () => window.clearTimeout(t);
  }, [team]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        setStatus("Loading leagues…");
        const L = await listLeagues(baseUrl, year);
        if (!cancelled) {
          setLeagues(L);
          setStatus("Ready");
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
          setStatus("Error");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [baseUrl, year]);

  const run = useCallback(async () => {
    setError(null);
    setStatus("Querying maps parquet…");
    try {
      const data = await queryMaps(baseUrl, year, {
        league: league || undefined,
        team: teamQuery || undefined,
        limit: 100,
      });
      setRows(data);
      setStatus(`${data.length} maps`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("Error");
    }
  }, [baseUrl, year, league, teamQuery]);

  useEffect(() => {
    void run();
  }, [run]);

  return (
    <div className="space-y-4">
      <div className="filter-bar">
        <label className="field">
          <span>Year</span>
          <select value={year} onChange={(e) => setYear(Number(e.target.value))}>
            {years.map((y) => (
              <option key={y} value={y}>
                {y}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>League</span>
          <select value={league} onChange={(e) => setLeague(e.target.value)}>
            <option value="">All</option>
            {leagues.map((L) => (
              <option key={L} value={L}>
                {L}
              </option>
            ))}
          </select>
        </label>
        <label className="field grow">
          <span>Team contains</span>
          <input
            value={team}
            onChange={(e) => setTeam(e.target.value)}
            placeholder="G2 / Karmine / VKS"
            onKeyDown={(e) => {
              if (e.key === "Enter") void run();
            }}
          />
        </label>
        <button type="button" className="btn-primary" onClick={() => void run()}>
          Refresh
        </button>
        <span className="status-hint">{status}</span>
      </div>
      {error && <p className="error-banner">{error}</p>}
      <div className="table-scroll">
        <table className="data-table maps-table">
          <thead>
            <tr>
              <th>Date</th>
              <th>League</th>
              <th>Blue</th>
              <th>Red</th>
              <th className="num">Kills</th>
              <th className="num">Gold</th>
              <th className="num">DRG</th>
              <th className="num">GRB</th>
              <th className="num">TWR</th>
              <th className="num">HLD</th>
              <th className="num">BAR</th>
              <th className="num">GD@15</th>
              <th className="num">Len</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const id = String(r.oe_gameid ?? r.game_uid ?? "");
              const blueW = Number(r.blue_result) === 1;
              const gd =
                r.blue_golddiffat15 != null && Number.isFinite(Number(r.blue_golddiffat15))
                  ? Number(r.blue_golddiffat15)
                  : null;
              return (
                <tr key={id}>
                  <td className="font-mono text-xs">{String(r.date ?? "").slice(0, 10)}</td>
                  <td>{String(r.league ?? "")}</td>
                  <td className={blueW ? "winner-cell" : undefined}>
                    {String(r.blue_teamname ?? "")}
                  </td>
                  <td className={!blueW ? "winner-cell" : undefined}>
                    {String(r.red_teamname ?? "")}
                  </td>
                  <td className="num font-mono">
                    {String(r.blue_teamkills ?? 0)}–{String(r.red_teamkills ?? 0)}
                  </td>
                  <td className="num font-mono">
                    {formatGold(r.blue_totalgold)} / {formatGold(r.red_totalgold)}
                  </td>
                  <td className="num font-mono">
                    {String(r.blue_dragons ?? "—")}–{String(r.red_dragons ?? "—")}
                  </td>
                  <td className="num font-mono">
                    {String(r.blue_void_grubs ?? "—")}–{String(r.red_void_grubs ?? "—")}
                  </td>
                  <td className="num font-mono">
                    {String(r.blue_towers ?? "—")}–{String(r.red_towers ?? "—")}
                  </td>
                  <td className="num font-mono">
                    {String(r.blue_heralds ?? "—")}–{String(r.red_heralds ?? "—")}
                  </td>
                  <td className="num font-mono">
                    {String(r.blue_barons ?? "—")}–{String(r.red_barons ?? "—")}
                  </td>
                  <td
                    className={`num font-mono ${
                      gd == null ? "" : gd >= 0 ? "gd-blue" : "gd-red"
                    }`}
                  >
                    {gd == null ? "—" : `${gd >= 0 ? "+" : ""}${Math.round(gd)}`}
                  </td>
                  <td className="num font-mono text-xs">
                    {formatClock(r.gamelength, r.length_min)}
                  </td>
                  <td>
                    <Link
                      href={`/browse/match/${encodeGameId(id)}?year=${year}`}
                      className="row-link"
                    >
                      Board
                    </Link>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {rows.length === 0 && !error && status !== "Querying maps parquet…" && (
          <p className="empty-hint">No maps match these filters.</p>
        )}
      </div>
    </div>
  );
}
