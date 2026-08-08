"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import {
  groupMapsIntoSeries,
  listLeagues,
  queryMaps,
  formatGameDate,
  type QueryRow,
  type SeriesCard,
} from "@/lib/duck";
import styles from "./BrowseMaps.module.css";
import { expandTeamQuery, teamSlug } from "@/lib/pack";

type Props = {
  baseUrl: string;
  years: number[];
};

function blueWon(m: QueryRow): boolean {
  if (m.y_blue_win != null) return Number(m.y_blue_win) >= 0.5;
  return m.blue_result === 1 || m.blue_result === true || m.blue_result === "1";
}

function TagInput({
  tags,
  onChange,
  placeholder,
}: {
  tags: string[];
  onChange: (tags: string[]) => void;
  placeholder: string;
}) {
  const [draft, setDraft] = useState("");
  const commit = (raw: string) => {
    const parts = raw
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    if (!parts.length) return;
    const next = [...tags];
    for (const p of parts) {
      // Use original token if not alias; else title-case from known aliases via first match in expand list
      const pretty = (() => {
        const needles = expandTeamQuery(p);
        // Find a needle that looks like a full name
        const full = needles.find((n) => n.includes(" ") || n.length > 4);
        if (full && full !== p.toLowerCase()) {
          // recover casing from common pattern - capitalize words
          return full.replace(/\b\w/g, (c) => c.toUpperCase()).replace(/\bOf\b/, "of");
        }
        return p;
      })();
      if (!next.some((t) => t.toLowerCase() === pretty.toLowerCase())) next.push(pretty);
    }
    onChange(next);
    setDraft("");
  };

  return (
    <div className="tag-field">
      <div className="tag-list">
        {tags.map((t) => (
          <span key={t} className="tag-chip">
            {t}
            <button
              type="button"
              aria-label={`Remove ${t}`}
              onClick={() => onChange(tags.filter((x) => x !== t))}
            >
              ×
            </button>
          </span>
        ))}
      </div>
      <input
        value={draft}
        placeholder={placeholder}
        onChange={(e) => {
          const v = e.target.value;
          if (v.includes(",")) commit(v);
          else setDraft(v);
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === ",") {
            e.preventDefault();
            commit(draft);
          }
          if (e.key === "Backspace" && !draft && tags.length) {
            onChange(tags.slice(0, -1));
          }
        }}
      />
    </div>
  );
}

function SeriesTile({
  s,
  open,
  onToggle,
}: {
  s: SeriesCard;
  open: boolean;
  onToggle: () => void;
}) {
  const sourceLabel = s.source === "grid" ? "GRID freshness" : s.source === "mixed" ? "OE + GRID" : null;
  return (
    <article className={`series-card ${open ? "is-open" : ""}`}>
      <button
        type="button"
        className="series-card-head"
        aria-expanded={open}
        onClick={onToggle}
      >
        <div className="series-card-main">
          <p className="series-kicker">
            {s.date} · {s.league}
            {s.patch ? ` · ${s.patch}` : ""} · Bo{s.bestOf}
            {sourceLabel ? ` · ${sourceLabel}` : ""}
          </p>
          <h3 className="series-title">
            <Link
              href={`/elo/team/${teamSlug(s.teamA)}`}
              className="row-link"
              onClick={(e) => e.stopPropagation()}
            >
              {s.teamA}
            </Link>
            <span className="series-score">
              {s.winsA}–{s.winsB}
            </span>
            <Link
              href={`/elo/team/${teamSlug(s.teamB)}`}
              className="row-link"
              onClick={(e) => e.stopPropagation()}
            >
              {s.teamB}
            </Link>
          </h3>
        </div>
        <span className="series-claim muted">
          {s.games.length} game{s.games.length === 1 ? "" : "s"}
        </span>
      </button>
      {open && (
        <ul className="series-games">
          {s.games.map((g) => {
            const id = String(g.oe_gameid ?? g.game_uid);
            const year = s.year || Number(formatGameDate(g.date).slice(0, 4));
            const bw = blueWon(g);
            const winner = bw ? String(g.blue_teamname) : String(g.red_teamname);
            return (
              <li key={id}>
                <Link href={`/browse/match/${encodeURIComponent(id)}?year=${year}`}>
                  <span className="font-mono">G{String(g.game ?? "?")}</span>
                  <span className={bw ? "text-[var(--side-blue)]" : "text-[var(--side-red)]"}>
                    {winner}
                  </span>
                  <span className="muted">
                    {Number(g.blue_teamkills) || 0}–{Number(g.red_teamkills) || 0} ·{" "}
                    {Number(g.length_min)?.toFixed?.(0) ?? "—"}m
                  </span>
                  <span className="row-link">Board »</span>
                </Link>
              </li>
            );
          })}
        </ul>
      )}
    </article>
  );
}

export function BrowseMatches({ baseUrl, years }: Props) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const yearDefault = years[years.length - 1] ?? 2026;
  const [year, setYear] = useState(Number(searchParams.get("year") || yearDefault));
  const [league, setLeague] = useState(searchParams.get("league") || "");
  const [patch, setPatch] = useState(searchParams.get("patch") || "");
  const [side, setSide] = useState<"" | "blue" | "red">(
    (searchParams.get("side") as "" | "blue" | "red") || "",
  );
  const [tags, setTags] = useState<string[]>(() =>
    (searchParams.get("teams") || "")
      .split("|")
      .map((s) => s.trim())
      .filter(Boolean),
  );
  const [leagues, setLeagues] = useState<string[]>([]);
  const [series, setSeries] = useState<SeriesCard[]>([]);
  const [page, setPage] = useState(0);
  const [openKey, setOpenKey] = useState<string | null>(null);
  const [status, setStatus] = useState("Idle");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [hasLoaded, setHasLoaded] = useState(false);

  const pageSize = 12;

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const L = await listLeagues(baseUrl, year);
        if (!cancelled) setLeagues(L);
      } catch {
        if (!cancelled) setLeagues([]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [baseUrl, year]);

  useEffect(() => {
    const params = new URLSearchParams();
    params.set("year", String(year));
    if (league) params.set("league", league);
    if (patch) params.set("patch", patch);
    if (side) params.set("side", side);
    if (tags.length) params.set("teams", tags.join("|"));
    router.replace(`${pathname}?${params.toString()}`, { scroll: false });
  }, [year, league, patch, side, tags, pathname, router]);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    setHasLoaded(false);
    setStatus(`Loading ${year} matches…`);
    setPage(0);
    try {
      const data = await queryMaps(baseUrl, year, {
        league: league || undefined,
        teams: tags.length ? tags : undefined,
        patch: patch || undefined,
        side: side || undefined,
        limit: 400,
      });
      const grouped = groupMapsIntoSeries(data.map((r) => ({ ...r, _year: year })));
      setSeries(grouped);
      if (grouped[0]) setOpenKey(grouped[0].key);
      setStatus(
        grouped.length
          ? `${grouped.length} series · ${data.length} games`
          : "No matches found for these filters.",
      );
      setHasLoaded(true);
    } catch {
      setError("Could not load matches for this selection. Try again or choose another year.");
      setSeries([]);
      setStatus("Could not load matches.");
      setHasLoaded(true);
    } finally {
      setLoading(false);
    }
  }, [baseUrl, year, league, tags, patch, side]);

  useEffect(() => {
    const t = setTimeout(() => {
      void run();
    }, 200);
    return () => clearTimeout(t);
  }, [run]);

  const pageRows = useMemo(() => {
    const start = page * pageSize;
    return series.slice(start, start + pageSize);
  }, [series, page]);

  const pageCount = Math.max(1, Math.ceil(series.length / pageSize));

  return (
    <div className={styles.root}>
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
        <label className="field">
          <span>Patch</span>
          <input
            value={patch}
            onChange={(e) => setPatch(e.target.value)}
            placeholder="16.1"
          />
        </label>
        <label className="field">
          <span>Side</span>
          <select value={side} onChange={(e) => setSide(e.target.value as "" | "blue" | "red")}>
            <option value="">Any</option>
            <option value="blue">Blue (tagged team)</option>
            <option value="red">Red (tagged team)</option>
          </select>
        </label>
        <label className="field grow">
          <span>Teams</span>
          <TagInput
            tags={tags}
            onChange={setTags}
            placeholder="G2, then comma → chip"
          />
        </label>
      </div>

      {loading && <div className="skeleton-block" aria-label={`Loading ${year} matches`} />}
      {error && (
        <div className="error-banner">
          <p>{error}</p>
          <button type="button" className="status-pill ghost mt-2" onClick={() => void run()}>
            Try again
          </button>
        </div>
      )}
      {!loading && <p className="status-hint" aria-live="polite">{status}</p>}

      <div className="series-gallery">
        {pageRows.map((s) => (
          <SeriesTile
            key={s.key}
            s={s}
            open={openKey === s.key}
            onToggle={() => setOpenKey((k) => (k === s.key ? null : s.key))}
          />
        ))}
      </div>

      {!loading && hasLoaded && series.length === 0 && !error && (
        <p className="empty-hint">
          No matches found for these filters. Clear a chip or try another league.
        </p>
      )}

      {series.length > pageSize && (
        <div className="pager">
          <button
            type="button"
            className="status-pill ghost"
            disabled={page <= 0}
            onClick={() => setPage((p) => Math.max(0, p - 1))}
          >
            Prev
          </button>
          <span className="status-hint">
            Page {page + 1} / {pageCount}
          </span>
          <button
            type="button"
            className="status-pill ghost"
            disabled={page + 1 >= pageCount}
            onClick={() => setPage((p) => p + 1)}
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}

/** @deprecated name kept for imports */
export { BrowseMatches as BrowseMaps };
