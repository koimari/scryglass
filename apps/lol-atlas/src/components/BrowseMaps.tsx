"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import {
  groupMapsIntoSeries,
  formatCompletionSource,
  formatSeriesLabel,
  formatSeriesScore,
  finiteNumberOrNull,
  isQuarantinedSeriesRow,
  listLeagues,
  queryFavoriteHitRate,
  queryMaps,
  formatGameDate,
  resolveMapWinnerSide,
  type FavoriteHitRateResult,
  type SeriesCard,
} from "@/lib/duck";
import { canonicalTeamDisplay, teamSlug } from "@/lib/pack";

type Props = {
  baseUrl: string;
  years: number[];
};

function formatNullableCount(value: unknown): string {
  const parsed = finiteNumberOrNull(value);
  return parsed == null ? "—" : String(Math.round(parsed));
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
      const pretty = canonicalTeamDisplay(p);
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
  const completionLabel =
    formatCompletionSource(s.completionSource) ??
    (s.completionSource === "score_to_format_validation"
      ? "Canonical score-to-format validation"
      : null);
  return (
    <article className={`series-card ${open ? "is-open" : ""}`}>
      <button type="button" className="series-card-head" onClick={onToggle}>
        <div className="series-card-main">
          <p className="series-kicker">
            {s.date} · {s.league}
            {s.patch ? ` · ${s.patch}` : ""} ·{" "}
            {formatSeriesLabel(s)}
            {sourceLabel ? ` · ${sourceLabel}` : ""}
            {completionLabel ? ` · ${completionLabel}` : ""}
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
              {formatSeriesScore(s)}
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
            const winnerSide = resolveMapWinnerSide(g);
            const winner =
              winnerSide === "blue"
                ? String(g.blue_teamname)
                : winnerSide === "red"
                  ? String(g.red_teamname)
                  : "Winner unavailable";
            const length = finiteNumberOrNull(g.length_min);
            return (
              <li key={id}>
                <Link href={`/browse/match/${encodeURIComponent(id)}?year=${year}`}>
                  <span className="font-mono">G{String(g.game ?? "?")}</span>
                  <span
                    className={
                      winnerSide === "blue"
                        ? "text-[var(--side-blue)]"
                        : winnerSide === "red"
                          ? "text-[var(--side-red)]"
                          : "muted"
                    }
                  >
                    {winner}
                  </span>
                  <span className="muted">
                    {formatNullableCount(g.blue_teamkills)}–
                    {formatNullableCount(g.red_teamkills)} ·{" "}
                    {length == null ? "—" : length.toFixed(0)}m
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
  const [favoriteRate, setFavoriteRate] = useState<FavoriteHitRateResult | null>(null);
  const [resultDisclosure, setResultDisclosure] = useState<string | null>(null);

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
      const unavailableOutcomes = data.filter((row) => resolveMapWinnerSide(row) == null).length;
      const quarantined = data.filter(isQuarantinedSeriesRow).length;
      const displayable = data.filter((row) => !isQuarantinedSeriesRow(row));
      const grouped = groupMapsIntoSeries(
        displayable.map((r) => ({ ...r, _year: year })),
      );
      const canonicalSeries = grouped.filter(
        (record) => record.recordKind === "canonical_series",
      ).length;
      const unverifiedGroups = grouped.length - canonicalSeries;
      const capped = data.length >= 400;
      setSeries(grouped);
      if (grouped[0]) setOpenKey(grouped[0].key);
      setStatus(
        grouped.length
          ? [
              canonicalSeries
                ? `${canonicalSeries} canonical series`
                : null,
              unverifiedGroups
                ? `${unverifiedGroups} unverified map group${
                    unverifiedGroups === 1 ? "" : "s"
                  }`
                : null,
              `${displayable.length} maps`,
              capped ? "latest 400 rows returned" : null,
            ]
              .filter(Boolean)
              .join(" · ")
          : "No matches found for these filters.",
      );
      const omissions = [
        unavailableOutcomes
          ? `${unavailableOutcomes} map${unavailableOutcomes === 1 ? "" : "s"} retained with outcome unavailable`
          : null,
        quarantined
          ? `${quarantined} map${quarantined === 1 ? "" : "s"} omitted: canonical series quarantined`
          : null,
        capped ? "Counts describe the latest returned sample, not a complete total" : null,
      ].filter(Boolean);
      setResultDisclosure(omissions.length ? omissions.join(" · ") : null);
      const rate = await queryFavoriteHitRate(baseUrl, year);
      setFavoriteRate(rate);
      setHasLoaded(true);
    } catch {
      setError("Could not load matches for this selection. Try again or choose another year.");
      setSeries([]);
      setResultDisclosure(null);
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
    <div className="space-y-6">
      <div className="model-acc-strip">
        <div>
          <strong>Model check · favorite hit rate</strong>{" "}
          {favoriteRate?.status === "ok" ? (
            <>
              {(100 * favoriteRate.rate).toFixed(1)}% · {favoriteRate.hits}/
              {favoriteRate.n} eligible games in {year}
            </>
          ) : favoriteRate?.status === "sample_empty" ? (
            "No eligible rated games in this year"
          ) : favoriteRate?.status === "error" ? (
            favoriteRate.code === "integrity_failed"
              ? "Unavailable · rating/map integrity check failed"
              : "Unavailable · pack query failed"
          ) : (
            "Loading…"
          )}
        </div>
        <div className="muted text-sm">
          Threshold diagnostic only: whether the pre-match rating favorite won. It is not a proper
          probability score; this pack does not attach log loss, Brier score, or calibration results
          here. Draft forecast remains withheld.
        </div>
      </div>

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
      {!loading && resultDisclosure && <p className="status-hint">{resultDisclosure}</p>}

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
