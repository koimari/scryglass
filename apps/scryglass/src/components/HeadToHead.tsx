"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import {
  groupMapsIntoSeries,
  listLeagues,
  loadMatchBundle,
  listTeams,
  queryMaps,
  queryMapsYears,
  type QueryRow,
  type SeriesCard,
} from "@/lib/duck";
import { expandTeamQuery, teamSlug } from "@/lib/pack";
import { MatchScoreboard } from "./MatchScoreboard";
import styles from "./HeadToHead.module.css";

function H2HBoardPanel({
  map,
  players,
  year,
}: {
  map: QueryRow;
  players: QueryRow[];
  year: number;
}) {
  return (
    <>
      <div className="micro-log mb-3">
        <span className="muted">Draft estimate withheld in this public MVP.</span>
      </div>
      <MatchScoreboard map={map} players={players} />
      <p className="mt-3 text-xs text-[var(--ink-muted)]">
        <Link
          href={`/browse/match/${encodeURIComponent(String(map.oe_gameid))}?year=${year}`}
          className="row-link"
        >
          Open match
        </Link>
      </p>
    </>
  );
}

type Props = { baseUrl: string; years: number[] };
type SortCol = "date" | "kills" | "len";
type Dir = "asc" | "desc";

export function HeadToHead({ baseUrl, years }: Props) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const yearDefault = years.includes(2026) ? 2026 : years[years.length - 1] ?? 2026;
  const [allTime, setAllTime] = useState(searchParams.get("scope") === "all");
  const [year, setYear] = useState(Number(searchParams.get("year") || yearDefault));
  const [league, setLeague] = useState(searchParams.get("league") || "");
  const [teamA, setTeamA] = useState(searchParams.get("a") || "");
  const [teamB, setTeamB] = useState(searchParams.get("b") || "");
  const [leagues, setLeagues] = useState<string[]>([]);
  const [teamDirectory, setTeamDirectory] = useState<string[]>([]);
  const [series, setSeries] = useState<SeriesCard[]>([]);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [board, setBoard] = useState<{ map: QueryRow; players: QueryRow[]; year: number } | null>(
    null,
  );
  const [sortCol, setSortCol] = useState<SortCol>("date");
  const [sortDir, setSortDir] = useState<Dir>("desc");
  const [status, setStatus] = useState("Enter two teams to compare their series.");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const autoRunKey = useRef<string | null>(null);

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

  useEffect(() => {
    let cancelled = false;
    listTeams(baseUrl, years)
      .then((names) => {
        if (!cancelled) setTeamDirectory(names);
      })
      .catch(() => {
        if (!cancelled) setTeamDirectory([]);
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, years]);

  useEffect(() => {
    const params = new URLSearchParams();
    if (teamA.trim()) params.set("a", teamA.trim());
    if (teamB.trim()) params.set("b", teamB.trim());
    if (league) params.set("league", league);
    if (allTime) params.set("scope", "all");
    else params.set("year", String(year));
    router.replace(`${pathname}?${params.toString()}`, { scroll: false });
  }, [teamA, teamB, league, year, allTime, pathname, router]);

  const record = useMemo(() => {
    let a = 0;
    let b = 0;
    const aN = teamA.trim().toLowerCase();
    for (const s of series) {
      if (s.teamA.toLowerCase().includes(aN) || s.teamB.toLowerCase().includes(aN)) {
        // map wins onto A/B labels
        const aIsTeamA = s.teamA.toLowerCase().includes(aN);
        if (aIsTeamA) {
          a += s.winsA;
          b += s.winsB;
        } else {
          a += s.winsB;
          b += s.winsA;
        }
      }
    }
    return { a, b };
  }, [series, teamA]);

  const sortedSeries = useMemo(() => {
    const list = [...series];
    const sign = sortDir === "asc" ? 1 : -1;
    list.sort((x, y) => {
      if (sortCol === "date") return sign * x.date.localeCompare(y.date);
      if (sortCol === "kills") {
        const kx = x.games.reduce(
          (s, g) => s + (Number(g.blue_teamkills) || 0) + (Number(g.red_teamkills) || 0),
          0,
        );
        const ky = y.games.reduce(
          (s, g) => s + (Number(g.blue_teamkills) || 0) + (Number(g.red_teamkills) || 0),
          0,
        );
        return sign * (kx - ky);
      }
      const lx = x.games.reduce((s, g) => s + (Number(g.length_min) || 0), 0);
      const ly = y.games.reduce((s, g) => s + (Number(g.length_min) || 0), 0);
      return sign * (lx - ly);
    });
    return list;
  }, [series, sortCol, sortDir]);

  const teamCandidates = (raw: string, sampleNames: string[]): string[] => {
    const needles = expandTeamQuery(raw);
    return [...new Set(sampleNames.filter((n) =>
      needles.some((nd) => n.toLowerCase().includes(nd) || nd.includes(n.toLowerCase())),
    ))];
  };

  const resolveExact = (raw: string, candidates: string[]): string | null => {
    if (candidates.length === 1) return candidates[0];
    return candidates.find((u) => u.toLowerCase() === raw.trim().toLowerCase()) ?? null;
  };

  const run = useCallback(async () => {
    if (!teamA.trim() || !teamB.trim()) {
      setError("Enter two teams.");
      return;
    }
    setError(null);
    setBoard(null);
    setSelectedKey(null);
    setLoading(true);
    setStatus("Searching series…");
    try {
      const aQ = teamA.trim();
      const bQ = teamB.trim();
      const variants = (q: string) => [...new Set(expandTeamQuery(q))].slice(0, 2);
      const pairs = variants(aQ).flatMap((a) => variants(bQ).map((b) => [a, b] as const));
      const datasets = await Promise.all(
        pairs.map(([a, b]) =>
          allTime
            ? queryMapsYears(baseUrl, years, {
                teamA: a,
                teamB: b,
                league: league || undefined,
                limit: 200,
              })
            : queryMaps(baseUrl, year, {
                league: league || undefined,
                teamA: a,
                teamB: b,
                limit: 120,
              }),
        ),
      );
      const data = [
        ...new Map(
          datasets
            .flat()
            .map((row) => [String(row.game_uid ?? row.oe_gameid ?? `${row.date}-${row.game}`), row] as const),
        ).values(),
      ];

      const names = [
        ...new Set(
          data.flatMap((r) => [String(r.blue_teamname ?? ""), String(r.red_teamname ?? "")]),
        ),
      ].filter(Boolean);
      const candidatesA = teamCandidates(aQ, names);
      const candidatesB = teamCandidates(bQ, names);
      const exactA = resolveExact(aQ, candidatesA);
      const exactB = resolveExact(bQ, candidatesB);
      if (!exactA || !exactB) {
        const missing = [
          candidatesA.length ? null : aQ,
          candidatesB.length ? null : bQ,
        ].filter(Boolean);
        setError(
          missing.length
            ? `No matching team found for ${missing.join(" and ")}. Try the full org name or an alias.`
            : `Multiple teams match ${aQ} or ${bQ}. Pick a full org name from the suggestions.`,
        );
        setSeries([]);
        setStatus("Idle");
        return;
      }

      const filtered = data.filter((r) => {
        const blue = String(r.blue_teamname);
        const red = String(r.red_teamname);
        return (
          (blue === exactA && red === exactB) || (blue === exactB && red === exactA)
        );
      });
      const grouped = groupMapsIntoSeries(
        filtered.map((r) => ({
          ...r,
          _year: r._year ?? year,
        })),
      );
      setSeries(grouped);
      setStatus(`${grouped.length} series · ${filtered.length} games`);
      if (grouped[0]) setSelectedKey(grouped[0].key);
    } catch (e) {
      setError(e instanceof Error ? `${e.message} Try again.` : String(e));
      setStatus("Error");
    } finally {
      setLoading(false);
    }
  }, [baseUrl, year, years, league, teamA, teamB, allTime]);

  useEffect(() => {
    const a = searchParams.get("a")?.trim() ?? "";
    const b = searchParams.get("b")?.trim() ?? "";
    if (!a || !b) return;
    const key = `${a}|${b}|${searchParams.get("year") ?? ""}|${searchParams.get("scope") ?? ""}`;
    if (autoRunKey.current === key) return;
    autoRunKey.current = key;
    void run();
  }, [run, searchParams]);

  useEffect(() => {
    if (!selectedKey) return;
    const s = series.find((x) => x.key === selectedKey);
    if (!s?.games[0]) return;
    const id = String(s.games[0].oe_gameid);
    const y = s.year || year;
    let cancelled = false;
    (async () => {
      setStatus("Loading game…");
      try {
        const bundle = await loadMatchBundle(baseUrl, allTime ? years : [y], id);
        if (cancelled) return;
        if (!bundle) {
          setError("No player data is available for this game.");
          setBoard(null);
          return;
        }
        setBoard({ map: bundle.map, players: bundle.players, year: bundle.year });
        setStatus("Ready");
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
  }, [selectedKey, series, baseUrl, year, years, allTime]);

  const onSort = (c: SortCol) => {
    if (c === sortCol) setSortDir((d) => (d === "desc" ? "asc" : "desc"));
    else {
      setSortCol(c);
      setSortDir(c === "date" ? "desc" : "desc");
    }
  };

  return (
    <div className={styles.root}>
      <div className="filter-bar">
        <label className="field">
          <span>Scope</span>
          <select
            value={allTime ? "all" : "year"}
            onChange={(e) => setAllTime(e.target.value === "all")}
          >
            <option value="year">Current year</option>
            <option value="all">All time (pack)</option>
          </select>
        </label>
        {!allTime && (
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
        )}
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
            placeholder="G2"
            list="h2h-teams"
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
            placeholder="FNC"
            list="h2h-teams"
            onKeyDown={(e) => {
              if (e.key === "Enter") void run();
            }}
          />
        </label>
        <datalist id="h2h-teams">
          {[...new Set([...teamDirectory, ...series.flatMap((s) => [s.teamA, s.teamB])])].map((t) => (
            <option key={t} value={t} />
          ))}
        </datalist>
        <button type="button" className="btn-primary" onClick={() => void run()}>
          Compare
        </button>
        {teamA.trim() && teamB.trim() ? (
          <Link
            className="status-pill ghost"
            href={`/browse?teams=${encodeURIComponent(teamA.trim())}|${encodeURIComponent(teamB.trim())}`}
          >
            Open in Matches
          </Link>
        ) : null}
      </div>

      {loading && <div className="skeleton-block" />}
      {error && <p className="error-banner">{error}</p>}
      {(loading || series.length > 0) && <p className="status-hint">{status}</p>}

      {series.length > 0 && (
        <>
          <p className={styles.record}>
            <Link href={`/elo/team/${teamSlug(teamA.trim())}`} className="row-link">
              {teamA.trim() || "A"}
            </Link>{" "}
            {record.a}–{record.b}{" "}
            <Link href={`/elo/team/${teamSlug(teamB.trim())}`} className="row-link">
              {teamB.trim() || "B"}
            </Link>
            <span className="muted">
              {" "}
              · {series.length} series · {series.reduce((n, s) => n + s.games.length, 0)} games
            </span>
          </p>
        </>
      )}

      {series.length === 0 && !loading && !error ? (
        <section className={styles.empty} aria-labelledby="h2h-empty-title">
          <h2 id="h2h-empty-title">No series selected</h2>
          <p>Enter two team names above to compare their completed series.</p>
        </section>
      ) : (
      <div className="h2h-layout">
        <div className="h2h-list table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>
                  <button type="button" className="sort-th" onClick={() => onSort("date")}>
                    Date{sortCol === "date" ? (sortDir === "asc" ? " ↑" : " ↓") : ""}
                  </button>
                </th>
                <th>Series</th>
                <th className="num">
                  <button type="button" className="sort-th" onClick={() => onSort("kills")}>
                    Kills{sortCol === "kills" ? (sortDir === "asc" ? " ↑" : " ↓") : ""}
                  </button>
                </th>
                <th className="num">
                  <button type="button" className="sort-th" onClick={() => onSort("len")}>
                    Len{sortCol === "len" ? (sortDir === "asc" ? " ↑" : " ↓") : ""}
                  </button>
                </th>
              </tr>
            </thead>
            <tbody>
              {sortedSeries.map((s) => {
                const active = s.key === selectedKey;
                const kills = s.games.reduce(
                  (n, g) => n + (Number(g.blue_teamkills) || 0) + (Number(g.red_teamkills) || 0),
                  0,
                );
                const len = s.games.reduce((n, g) => n + (Number(g.length_min) || 0), 0);
                return (
                  <tr
                    key={s.key}
                    className={active ? "is-selected" : undefined}
                  >
                    <td className="font-mono text-xs">{s.date}</td>
                    <td>
                      <button
                        type="button"
                        className={styles.seriesButton}
                        aria-pressed={active}
                        onClick={() => setSelectedKey(s.key)}
                      >
                        Bo{s.bestOf} · {s.teamA} {s.winsA}–{s.winsB} {s.teamB}
                        <span className="muted text-xs block">{s.league}</span>
                      </button>
                    </td>
                    <td className="num font-mono">{kills}</td>
                    <td className="num font-mono text-xs">{len.toFixed(0)}m</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="h2h-board">
          {board ? (
            <H2HBoardPanel map={board.map} players={board.players} year={board.year} />
          ) : (
            <p className="empty-hint">
              Select a series to inspect a game. Model checks are available on the match page.
            </p>
          )}
        </div>
      </div>
      )}
    </div>
  );
}
