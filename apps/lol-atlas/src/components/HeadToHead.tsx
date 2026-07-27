"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import {
  groupMapsIntoSeries,
  formatSeriesLabel,
  formatSeriesScore,
  isQuarantinedSeriesRow,
  listLeagues,
  loadMatchBundle,
  listTeams,
  queryMaps,
  queryMapsYears,
  resolveMapWinnerSide,
  sumKnownNumbers,
  type QueryRow,
  type SeriesCard,
} from "@/lib/duck";
import { expandTeamQuery, teamSlug } from "@/lib/pack";
import { MatchScoreboard } from "./MatchScoreboard";

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
        <span className="muted">
          <strong>Draft forecast</strong> withheld · current model failed the chronological benchmark
        </span>
      </div>
      <MatchScoreboard
        map={map}
        players={players}
        draftPctBlue={null}
      />
      <p className="mt-3 text-xs text-[var(--ink-muted)]">
        <Link
          href={`/browse/match/${encodeURIComponent(
            String(map.oe_gameid ?? map.game_uid),
          )}?year=${year}`}
          className="row-link"
        >
          Open match page (model checklist)
        </Link>
      </p>
    </>
  );
}

type Props = { baseUrl: string; years: number[] };
type SortCol = "date" | "kills" | "len";
type Dir = "asc" | "desc";

function seriesTotal(series: SeriesCard, fields: string[]): number | null {
  const perGame = series.games.map((game) =>
    sumKnownNumbers(fields.map((field) => game[field])),
  );
  return perGame.every((value): value is number => value != null)
    ? perGame.reduce((total, value) => total + value, 0)
    : null;
}

function compareNullable(a: number | null, b: number | null, direction: Dir): number {
  if (a == null && b == null) return 0;
  if (a == null) return 1;
  if (b == null) return -1;
  return (direction === "asc" ? 1 : -1) * (a - b);
}

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
  const [resolvedPair, setResolvedPair] = useState<{ a: string; b: string } | null>(
    null,
  );
  const [leagues, setLeagues] = useState<string[]>([]);
  const [teamDirectory, setTeamDirectory] = useState<string[]>([]);
  const [series, setSeries] = useState<SeriesCard[]>([]);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [board, setBoard] = useState<{ map: QueryRow; players: QueryRow[]; year: number } | null>(
    null,
  );
  const [sortCol, setSortCol] = useState<SortCol>("date");
  const [sortDir, setSortDir] = useState<Dir>("desc");
  const [status, setStatus] = useState("Idle");
  const [resultDisclosure, setResultDisclosure] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const autoRunKey = useRef<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const L = await listLeagues(baseUrl, year);
        if (!cancelled) setLeagues(L);
      } catch {
        if (!cancelled) setError("Could not load league filters from the public pack.");
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
    let mapA = 0;
    let mapB = 0;
    let unknownMaps = 0;
    let seriesA = 0;
    let seriesB = 0;
    let unknownSeries = 0;
    const aName = resolvedPair?.a ?? teamA.trim();
    for (const s of series) {
      for (const map of s.games) {
        const winnerSide = resolveMapWinnerSide(map);
        if (!winnerSide) {
          unknownMaps += 1;
          continue;
        }
        const winner = String(
          winnerSide === "blue" ? map.blue_teamname : map.red_teamname,
        );
        if (winner === aName) mapA += 1;
        else mapB += 1;
      }
      if (s.recordKind !== "canonical_series") continue;
      if (s.winsA == null || s.winsB == null || s.winsA === s.winsB) {
        unknownSeries += 1;
        continue;
      }
      const winner = s.winsA > s.winsB ? s.teamA : s.teamB;
      if (winner === aName) seriesA += 1;
      else seriesB += 1;
    }
    return {
      mapA,
      mapB,
      knownMaps: mapA + mapB,
      unknownMaps,
      seriesA,
      seriesB,
      knownSeries: seriesA + seriesB,
      unknownSeries,
    };
  }, [series, teamA, resolvedPair]);

  const sortedSeries = useMemo(() => {
    const list = [...series];
    const sign = sortDir === "asc" ? 1 : -1;
    list.sort((x, y) => {
      if (sortCol === "date") return sign * x.date.localeCompare(y.date);
      if (sortCol === "kills") {
        return compareNullable(
          seriesTotal(x, ["blue_teamkills", "red_teamkills"]),
          seriesTotal(y, ["blue_teamkills", "red_teamkills"]),
          sortDir,
        );
      }
      return compareNullable(
        seriesTotal(x, ["length_min"]),
        seriesTotal(y, ["length_min"]),
        sortDir,
      );
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
      setError("Need two teams — even rivals deserve a proper introduction.");
      return;
    }
    setError(null);
    setResultDisclosure(null);
    setBoard(null);
    setSelectedKey(null);
    setResolvedPair(null);
    setLoading(true);
    setStatus("Searching series…");
    try {
      const aQ = teamA.trim();
      const bQ = teamB.trim();
      const variants = (q: string) => [...new Set(expandTeamQuery(q))].slice(0, 2);
      const pairs = variants(aQ).flatMap((a) => variants(bQ).map((b) => [a, b] as const));
      const queryLimit = allTime ? 200 : 120;
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
      setResolvedPair({ a: exactA, b: exactB });

      const filtered = data.filter((r) => {
        const blue = String(r.blue_teamname);
        const red = String(r.red_teamname);
        return (
          (blue === exactA && red === exactB) || (blue === exactB && red === exactA)
        );
      });
      const unavailableOutcomes = filtered.filter(
        (row) => resolveMapWinnerSide(row) == null,
      ).length;
      const quarantined = filtered.filter(isQuarantinedSeriesRow).length;
      const displayable = filtered.filter((row) => !isQuarantinedSeriesRow(row));
      const capped = datasets.some((dataset) => dataset.length >= queryLimit);
      const grouped = groupMapsIntoSeries(
        displayable.map((r) => ({
          ...r,
          _year: r._year ?? year,
        })),
      );
      const canonicalSeries = grouped.filter(
        (item) => item.recordKind === "canonical_series",
      ).length;
      const unverifiedGroups = grouped.length - canonicalSeries;
      setSeries(grouped);
      setStatus(
        [
          canonicalSeries ? `${canonicalSeries} canonical series` : null,
          unverifiedGroups
            ? `${unverifiedGroups} unverified map group${
                unverifiedGroups === 1 ? "" : "s"
              }`
            : null,
          `${displayable.length} maps`,
          capped ? `latest ${queryLimit} rows per alias query` : null,
        ]
          .filter(Boolean)
          .join(" · "),
      );
      const omissions = [
        unavailableOutcomes
          ? `${unavailableOutcomes} map${unavailableOutcomes === 1 ? "" : "s"} retained with outcome unavailable`
          : null,
        quarantined
          ? `${quarantined} map${quarantined === 1 ? "" : "s"} omitted: canonical series quarantined`
          : null,
        capped ? "Records and counts describe the returned sample, not a complete total" : null,
      ].filter(Boolean);
      setResultDisclosure(omissions.length ? omissions.join(" · ") : null);
      if (grouped[0]) setSelectedKey(grouped[0].key);
    } catch {
      setError("Could not load this head-to-head sample from the public pack. Try again.");
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
    const id = String(s.games[0].oe_gameid ?? s.games[0].game_uid);
    const y = s.year || year;
    let cancelled = false;
    (async () => {
      setStatus("Loading board…");
      try {
        const bundle = await loadMatchBundle(baseUrl, allTime ? years : [y], id);
        if (cancelled) return;
        if (!bundle) {
          setError("This map was not found in the selected public pack years.");
          setBoard(null);
          return;
        }
        setBoard({ map: bundle.map, players: bundle.players, year: bundle.year });
        setStatus("Ready");
      } catch {
        if (!cancelled) {
          setError("Could not load this match board from the public pack.");
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
    <div className="space-y-6">
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
          Find H2H
        </button>
        {teamA.trim() && teamB.trim() ? (
          <Link
            className="status-pill ghost"
            href={`/browse?teams=${encodeURIComponent(teamA.trim())}|${encodeURIComponent(teamB.trim())}`}
          >
            Open Matches
          </Link>
        ) : null}
      </div>

      {loading && <div className="skeleton-block" />}
      {error && <p className="error-banner">{error}</p>}
      <p className="status-hint">{status}</p>
      {resultDisclosure && <p className="status-hint">{resultDisclosure}</p>}

      {series.length > 0 && (
        <>
          <p className="h2h-record font-mono">
            Map record ·{" "}
            <Link
              href={`/elo/team/${teamSlug(resolvedPair?.a ?? teamA.trim())}`}
              className="row-link"
            >
              {(resolvedPair?.a ?? teamA.trim()) || "A"}
            </Link>{" "}
            {record.mapA}–{record.mapB}{" "}
            <Link
              href={`/elo/team/${teamSlug(resolvedPair?.b ?? teamB.trim())}`}
              className="row-link"
            >
              {(resolvedPair?.b ?? teamB.trim()) || "B"}
            </Link>
            <span className="muted">
              {" "}· n={record.knownMaps}
              {record.unknownMaps ? ` · ${record.unknownMaps} unknown` : ""}
            </span>
          </p>
          <p className="h2h-record font-mono">
            Canonical series record · {record.seriesA}–{record.seriesB}
            <span className="muted">
              {" "}· n={record.knownSeries}
              {record.unknownSeries
                ? ` · ${record.unknownSeries} outcome${record.unknownSeries === 1 ? "" : "s"} unknown`
                : ""}
              {" "}· unverified map groups excluded
            </span>
          </p>
        </>
      )}

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
                <th>Record</th>
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
                const kills = seriesTotal(s, ["blue_teamkills", "red_teamkills"]);
                const len = seriesTotal(s, ["length_min"]);
                return (
                  <tr
                    key={s.key}
                    className={active ? "is-selected" : undefined}
                    tabIndex={0}
                    role="button"
                    onClick={() => setSelectedKey(s.key)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        setSelectedKey(s.key);
                      }
                    }}
                    style={{ cursor: "pointer" }}
                  >
                    <td className="font-mono text-xs">{s.date}</td>
                    <td>
                      {formatSeriesLabel(s)} · {s.teamA} {formatSeriesScore(s)}{" "}
                      {s.teamB}
                      <div className="muted text-xs">{s.league}</div>
                    </td>
                    <td className="num font-mono">
                      {kills == null ? "—" : Math.round(kills)}
                    </td>
                    <td className="num font-mono text-xs">
                      {len == null ? "—" : `${len.toFixed(0)}m`}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {series.length === 0 && !error && !loading && (
            <p className="empty-hint">
              Enter two teams and find meetings. Tip: aliases like KC work once they resolve cleanly.
            </p>
          )}
        </div>
        <div className="h2h-board">
          {board ? (
            <H2HBoardPanel map={board.map} players={board.players} year={board.year} />
          ) : (
            <p className="empty-hint">
              Click a record to open the board. The checklist lives on the match page — H2H stays
              about the rivalry.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
