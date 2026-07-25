"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  formatClock,
  listLeagues,
  loadMatchBundle,
  loadMatchModelPrior,
  queryMaps,
  type MatchModelPrior,
  type QueryRow,
} from "@/lib/duck";
import { MatchScoreboard } from "./MatchScoreboard";
import { ModelChecklist } from "./ModelChecklist";
import { useDraftWr } from "./DraftWrPanel";

function H2HBoardPanel({
  map,
  players,
  prior,
  priorLoading,
  year,
}: {
  map: QueryRow;
  players: QueryRow[];
  prior: MatchModelPrior | null;
  priorLoading: boolean;
  year: number;
}) {
  const eloDiff = prior?.muDiff ?? null;
  const { draft } = useDraftWr(map, players, eloDiff);
  return (
    <>
      <MatchScoreboard map={map} players={players} draftPctBlue={draft?.p_blue_draft ?? null} />
      <ModelChecklist
        map={map}
        players={players}
        prior={prior}
        loading={priorLoading}
        eloDiff={eloDiff}
      />
      <p className="mt-3 text-xs text-[var(--ink-muted)]">
        <Link
          href={`/browse/match/${encodeURIComponent(String(map.oe_gameid))}?year=${year}`}
          className="row-link"
        >
          Open match page
        </Link>
      </p>
    </>
  );
}

type Props = { baseUrl: string; years: number[] };

export function HeadToHead({ baseUrl, years }: Props) {
  const yearDefault = years.includes(2026) ? 2026 : years[years.length - 1] ?? 2026;
  const [year, setYear] = useState(yearDefault);
  const [league, setLeague] = useState("");
  const [teamA, setTeamA] = useState("");
  const [teamB, setTeamB] = useState("");
  const [leagues, setLeagues] = useState<string[]>([]);
  const [rows, setRows] = useState<QueryRow[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [board, setBoard] = useState<{
    map: QueryRow;
    players: QueryRow[];
  } | null>(null);
  const [prior, setPrior] = useState<MatchModelPrior | null>(null);
  const [priorLoading, setPriorLoading] = useState(false);
  const [status, setStatus] = useState("Idle");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const L = await listLeagues(baseUrl, year);
        if (!cancelled) setLeagues(L);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [baseUrl, year]);

  const record = useMemo(() => {
    let a = 0;
    let b = 0;
    const aNeedle = teamA.trim().toLowerCase();
    const bNeedle = teamB.trim().toLowerCase();
    if (!aNeedle || !bNeedle) return { a, b };
    for (const r of rows) {
      const blue = String(r.blue_teamname ?? "").toLowerCase();
      const red = String(r.red_teamname ?? "").toLowerCase();
      const blueIsA = blue.includes(aNeedle);
      const redIsA = red.includes(aNeedle);
      const blueIsB = blue.includes(bNeedle);
      const redIsB = red.includes(bNeedle);
      // Prefer exact-side pairing; skip ambiguous double-matches
      const aOnBlue = blueIsA && redIsB && !(blueIsB || redIsA);
      const aOnRed = redIsA && blueIsB && !(redIsB || blueIsA);
      // Looser fallback when needles are unique enough
      const looseBlueA = blueIsA && redIsB;
      const looseRedA = redIsA && blueIsB;
      const blueWin = Number(r.blue_result) === 1;
      if (aOnBlue || (!aOnRed && looseBlueA && !(blueIsB && redIsA))) {
        if (blueWin) a += 1;
        else b += 1;
      } else if (aOnRed || looseRedA) {
        if (blueWin) b += 1;
        else a += 1;
      }
    }
    return { a, b };
  }, [rows, teamA, teamB]);

  const run = useCallback(async () => {
    if (!teamA.trim() || !teamB.trim()) {
      setError("Enter both team names for head-to-head.");
      return;
    }
    setError(null);
    setBoard(null);
    setSelected(null);
    setStatus("Searching maps…");
    try {
      const data = await queryMaps(baseUrl, year, {
        league: league || undefined,
        teamA: teamA.trim(),
        teamB: teamB.trim(),
        limit: 80,
      });
      setRows(data);
      setStatus(`${data.length} meetings`);
      if (data[0]) {
        const id = String(data[0].oe_gameid);
        setSelected(id);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("Error");
    }
  }, [baseUrl, year, league, teamA, teamB]);

  useEffect(() => {
    if (!selected) return;
    let cancelled = false;
    (async () => {
      setStatus("Loading scoreboard…");
      setPrior(null);
      try {
        const bundle = await loadMatchBundle(baseUrl, [year], selected);
        if (cancelled) return;
        if (!bundle) {
          setError("No player rows for this map.");
          setBoard(null);
          return;
        }
        setBoard({ map: bundle.map, players: bundle.players });
        setStatus("Ready");
        setPriorLoading(true);
        const p = await loadMatchModelPrior(baseUrl, bundle.year, bundle.map);
        if (!cancelled) {
          setPrior(p);
          setPriorLoading(false);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
          setStatus("Error");
          setPriorLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selected, baseUrl, year]);

  return (
    <div className="space-y-6">
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
          <span>Team A</span>
          <input
            value={teamA}
            onChange={(e) => setTeamA(e.target.value)}
            placeholder="VKS Academy"
            onKeyDown={(e) => {
              if (e.key === "Enter") void run();
            }}
          />
        </label>
        <label className="field grow">
          <span>Team B</span>
          <input
            value={teamB}
            onChange={(e) => setTeamB(e.target.value)}
            placeholder="KaBuM"
            onKeyDown={(e) => {
              if (e.key === "Enter") void run();
            }}
          />
        </label>
        <button type="button" className="btn-primary" onClick={() => void run()}>
          Find H2H
        </button>
        <span className="status-hint">{status}</span>
      </div>

      {error && <p className="error-banner">{error}</p>}

      {rows.length > 0 && (
        <p className="h2h-record font-mono">
          {teamA.trim() || "A"} {record.a}–{record.b} {teamB.trim() || "B"}
          <span className="muted"> · {rows.length} maps in pack</span>
        </p>
      )}

      <div className="h2h-layout">
        <div className="h2h-list table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>Date</th>
                <th>Blue</th>
                <th>Red</th>
                <th className="num">K</th>
                <th className="num">Len</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const id = String(r.oe_gameid ?? "");
                const active = id === selected;
                return (
                  <tr
                    key={id}
                    className={active ? "is-selected" : undefined}
                    tabIndex={0}
                    role="button"
                    aria-pressed={active}
                    onClick={() => setSelected(id)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        setSelected(id);
                      }
                    }}
                    style={{ cursor: "pointer" }}
                  >
                    <td className="font-mono text-xs">{String(r.date ?? "").slice(0, 10)}</td>
                    <td>{String(r.blue_teamname ?? "")}</td>
                    <td>{String(r.red_teamname ?? "")}</td>
                    <td className="num font-mono">
                      {String(r.blue_teamkills ?? 0)}–{String(r.red_teamkills ?? 0)}
                    </td>
                    <td className="num font-mono text-xs">
                      {formatClock(r.gamelength, r.length_min)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {rows.length === 0 && !error && (
            <p className="empty-hint">Enter two teams and find meetings in the pack years.</p>
          )}
        </div>
        <div className="h2h-board">
          {board ? (
            <H2HBoardPanel
              map={board.map}
              players={board.players}
              prior={prior}
              priorLoading={priorLoading}
              year={year}
            />
          ) : (
            <p className="empty-hint">Select a meeting to open the Leaguepedia-style scoreboard.</p>
          )}
        </div>
      </div>
    </div>
  );
}
