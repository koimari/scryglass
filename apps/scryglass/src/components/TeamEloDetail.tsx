"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  groupMapsIntoSeries,
  queryMapsYears,
  queryRosterChampStats,
  type ChampAgg,
  type SeriesCard,
} from "@/lib/duck";
import { champIconUrl, formatGold } from "@/lib/format";
import type { PlayerRating, TeamRating, TeamRecord } from "@/lib/pack";
import { evidenceFields, evidenceInfo, formatEvidenceCell } from "@/lib/evidence";
import {
  adjustedRating,
  formatWr,
  packUpdatedLabel,
  PLAYER_SIGMA_MIN,
  playerSlug,
  softMu,
  TEAM_SIGMA_MIN,
  teamSlug,
  type PackManifest,
} from "@/lib/pack";
import profileStyles from "./ProfileHeader.module.css";

type Props = {
  team: TeamRating;
  roster: PlayerRating[];
  record?: TeamRecord;
  baseUrl: string;
  years: number[];
  manifest: PackManifest;
};

type ChampCol = "n" | "wr" | "kda" | "gold" | "dpm" | "cs";

function ChampTable({
  champs,
  limit,
}: {
  champs: ChampAgg[];
  limit: number | null;
}) {
  const [col, setCol] = useState<ChampCol>("n");
  const [dir, setDir] = useState<"asc" | "desc">("desc");
  const sorted = useMemo(() => {
    const list = [...champs];
    const sign = dir === "asc" ? 1 : -1;
    list.sort((a, b) => {
      const kda = (c: ChampAgg) => (c.kills + c.assists) / Math.max(c.deaths, 1);
      switch (col) {
        case "wr":
          return sign * (a.wr - b.wr);
        case "kda":
          return sign * (kda(a) - kda(b));
        case "gold":
          return sign * (a.gold - b.gold);
        case "dpm":
          return sign * ((a.dpm ?? 0) - (b.dpm ?? 0));
        case "cs":
          return sign * ((a.cs ?? 0) - (b.cs ?? 0));
        default:
          return sign * (a.n - b.n);
      }
    });
    return limit == null ? list : list.slice(0, limit);
  }, [champs, col, dir, limit]);

  const th = (label: string, c: ChampCol) => (
    <th className="num">
      <button
        type="button"
        className={`sort-th ${col === c ? "is-active" : ""}`}
        onClick={() => {
          if (col === c) setDir((d) => (d === "desc" ? "asc" : "desc"));
          else {
            setCol(c);
            setDir("desc");
          }
        }}
      >
        {label}
      </button>
    </th>
  );

  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead>
          <tr>
            <th>Champion</th>
            {th("n", "n")}
            {th("KDA", "kda")}
            {th("Gold", "gold")}
            {th("DPM", "dpm")}
            {th("CS", "cs")}
            {th("WR", "wr")}
          </tr>
        </thead>
        <tbody>
          {sorted.map((c) => {
            const src = champIconUrl(c.champion);
            return (
              <tr key={c.champion}>
                <td>
                  <span className="inline-flex items-center gap-2">
                    {src ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={src} alt="" width={22} height={22} className="champ-thumb" />
                    ) : null}
                    {c.champion}
                  </span>
                </td>
                <td className="num">{c.n}</td>
                <td className="num">
                  {c.kills.toFixed(1)}/{c.deaths.toFixed(1)}/{c.assists.toFixed(1)}
                </td>
                <td className="num">{formatGold(c.gold)}</td>
                <td className="num">{c.dpm != null ? c.dpm.toFixed(0) : "—"}</td>
                <td className="num">{c.cs != null ? c.cs.toFixed(0) : "—"}</td>
                <td className="num">{(100 * c.wr).toFixed(0)}%</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function exportRosterCsv(team: string, roster: PlayerRating[]) {
  const lines = [
    "player,raw_rating,adjusted_rating,evidence_sigma,games,last_team",
    ...roster.map(
      (p) =>
        `"${p.player}",${p.mu_total.toFixed(2)},${softMu(p.mu_total, p.sigma, PLAYER_SIGMA_MIN).toFixed(2)},${p.sigma.toFixed(2)},${p.n_maps},"${p.last_team ?? ""}"`,
    ),
  ];
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${team.replace(/\s+/g, "_")}_roster.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

export function TeamEloDetail({ team, roster, record, baseUrl, years, manifest }: Props) {
  const topRatedPlayers = useMemo(
    () =>
      [...roster]
        .sort(
          (a, b) =>
            softMu(b.mu_total, b.sigma, PLAYER_SIGMA_MIN) -
            softMu(a.mu_total, a.sigma, PLAYER_SIGMA_MIN),
        )
        .slice(0, 5),
    [roster],
  );
  const otherPlayers = useMemo(() => {
    const topRatedSet = new Set(topRatedPlayers.map((p) => p.player));
    return [...roster]
      .filter((p) => !topRatedSet.has(p.player))
      .sort(
        (a, b) =>
          softMu(b.mu_total, b.sigma, PLAYER_SIGMA_MIN) -
          softMu(a.mu_total, a.sigma, PLAYER_SIGMA_MIN),
      );
  }, [roster, topRatedPlayers]);

  const [showSubs, setShowSubs] = useState(false);
  const [byPlayer, setByPlayer] = useState<Record<string, ChampAgg[]> | null>(null);
  const [champExpand, setChampExpand] = useState<Record<string, boolean>>({});
  const [series, setSeries] = useState<SeriesCard[]>([]);
  const [banPick, setBanPick] = useState<{ bans: string[]; picks: string[] } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [seriesLoaded, setSeriesLoaded] = useState(false);
  const [seriesError, setSeriesError] = useState<string | null>(null);
  const [seriesRetry, setSeriesRetry] = useState(0);
  const trust = evidenceInfo(evidenceFields(team as unknown as Record<string, unknown>), team.sigma, record?.games);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setSeriesError(null);
      setSeriesLoaded(false);
      try {
        const names = roster.map((p) => p.player);
        const rows = await queryRosterChampStats(baseUrl, years, names, 40);
        if (!cancelled) {
          setByPlayer(rows);
        }
      } catch (e) {
        if (!cancelled) {
          setErr(e instanceof Error ? e.message : String(e));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [baseUrl, years, roster]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const maps = await queryMapsYears(baseUrl, years, { team: team.team, limit: 80 });
        const grouped = groupMapsIntoSeries(maps);
        if (cancelled) return;
        setSeries(grouped.slice(0, 8));
        setSeriesLoaded(true);

        const banCount = new Map<string, number>();
        const pickCount = new Map<string, number>();
        const patches = [...new Set(maps.map((m) => String(m.patch ?? "")))].filter(Boolean);
        patches.sort();
        const recentPatches = new Set(patches.slice(-3));

        for (const m of maps.slice(0, 25)) {
          if (recentPatches.has(String(m.patch ?? ""))) {
            const side = String(m.blue_teamname) === team.team ? "blue" : "red";
            for (let i = 1; i <= 5; i++) {
              const b = String(m[`${side}_ban${i}`] ?? "");
              const p = String(m[`${side}_pick${i}`] ?? "");
              if (b) banCount.set(b, (banCount.get(b) ?? 0) + 1);
              if (p) pickCount.set(p, (pickCount.get(p) ?? 0) + 1);
            }
          }
        }
        const top = (m: Map<string, number>, n: number) =>
          [...m.entries()]
            .sort((a, b) => b[1] - a[1])
            .slice(0, n)
            .map(([k]) => k);
        setBanPick({ bans: top(banCount, 8), picks: top(pickCount, 8) });
      } catch {
        if (!cancelled) {
          setSeries([]);
          setSeriesError("Could not load recent series. Try again when the pack is available.");
          setSeriesLoaded(true);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [baseUrl, years, team.team, seriesRetry]);

  const wl = record
    ? `${record.wins}–${record.games - record.wins}`
    : "—";

  return (
    <div className="profile-page team-profile-page space-y-6 print-area">
      <p className="text-xs text-[var(--ink-muted)]">
        <Link href="/elo" className="row-link">
          ← Ratings
        </Link>
      </p>
      <header className={profileStyles.header}>
        <div className={profileStyles.identity}>
          <p className={profileStyles.scope}>
            Team · {record?.primary ?? "Dual Elo"}
            {record?.intl ? " · International" : ""}
          </p>
          <h1>{team.team}</h1>
          <p className={profileStyles.summary}>
            Organization strength in the current pack. This descriptive benchmark is not an
            exact match-time roster estimate; the adjusted rating accounts for uncertainty.
          </p>
        </div>
        <div className={profileStyles.metrics}>
          <span className={profileStyles.primary}>
            <strong>Adjusted rating</strong>{" "}
            <em>{adjustedRating(team, TEAM_SIGMA_MIN).toFixed(1)}</em>
          </span>
          <span>
            <strong>Raw rating</strong> {team.mu_total.toFixed(1)}
          </span>
          <span title={trust.layman}>
            <strong>Evidence</strong> {formatEvidenceCell(trust)}
          </span>
          <span>
            <strong>Games</strong> {record?.games ?? "—"}
          </span>
          <span>
            <strong>W–L</strong> {wl}
          </span>
          <span>
            <strong>WR</strong> {formatWr(record?.wr)}
          </span>
          <span>
            <strong>Updated</strong> {packUpdatedLabel(manifest)}
          </span>
        </div>
        <div className={profileStyles.actions}>
          <Link
            className="btn-primary"
            href={`/browse/head-to-head?a=${encodeURIComponent(team.team)}`}
          >
            Head-to-head
          </Link>
          <button
            type="button"
            className="status-pill ghost"
            onClick={() => exportRosterCsv(team.team, roster)}
          >
            Export CSV
          </button>
          <button type="button" className="status-pill ghost" onClick={() => window.print()}>
            Print / PDF
          </button>
        </div>
      </header>

      <section className="space-y-4">
        <h2 className="font-display text-xl">Players by adjusted rating</h2>
        <p className="text-sm muted">
          The top five current-snapshot players appear first. This order reflects rating evidence;
          the pack does not encode starter or substitute roles.
        </p>
        {roster.length === 0 ? (
          <p className="empty-hint">
            No players currently tagged to this org in the snapshot. Rosters move — try Matches for
            recent lineups.
          </p>
        ) : (
          topRatedPlayers.map((p) => {
            const champs = byPlayer?.[p.player] ?? [];
            const expanded = champExpand[p.player];
            const pTrust = evidenceInfo(evidenceFields(p as unknown as Record<string, unknown>), p.sigma, p.n_maps);
            return (
              <section key={p.player} className="border-t border-[var(--line)] pt-4 space-y-3">
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <h3 className="font-display text-lg">
                    <Link href={`/elo/player/${playerSlug(p.player)}`} className="row-link">
                      {p.player}
                    </Link>
                  </h3>
                  <div className="micro-log">
                    <span>
                      <strong>Raw rating</strong> {p.mu_total.toFixed(1)}
                    </span>
                    <span>
                      <strong>Adjusted rating</strong>{" "}
                      {softMu(p.mu_total, p.sigma, PLAYER_SIGMA_MIN).toFixed(1)}
                    </span>
                    <span title={pTrust.layman}>
                      <strong>Evidence</strong> {formatEvidenceCell(pTrust)}
                    </span>
                    <span>
                      <strong>Games</strong> {p.n_maps}
                    </span>
                  </div>
                </div>
                {err && <p className="error-banner">{err}</p>}
                {!byPlayer && !err && <div className="skeleton-block short" aria-label="Loading champion rows" />}
                {byPlayer && champs.length === 0 && !err && (
                  <p className="empty-hint">Champion rows are unavailable in this pack.</p>
                )}
                {champs.length > 0 && (
                  <>
                    <ChampTable champs={champs} limit={expanded ? null : 3} />
                    {champs.length > 3 && (
                      <button
                        type="button"
                        className="status-pill ghost"
                        onClick={() =>
                          setChampExpand((s) => ({ ...s, [p.player]: !s[p.player] }))
                        }
                      >
                        {expanded ? "Collapse champs" : "Expand champs"}
                      </button>
                    )}
                  </>
                )}
              </section>
            );
          })
        )}

        {otherPlayers.length > 0 && (
          <div className="pt-2">
            <button
              type="button"
              className="status-pill ghost"
              onClick={() => setShowSubs((x) => !x)}
            >
              {showSubs ? "Hide remaining players" : `Show remaining players (${otherPlayers.length})`}
            </button>
            {showSubs &&
              otherPlayers.map((p) => (
                <p key={p.player} className="mt-2 text-sm">
                  <span className="status-pill ghost" style={{ marginRight: 8 }}>
                    Additional snapshot player
                  </span>
                  <Link href={`/elo/player/${playerSlug(p.player)}`} className="row-link">
                    {p.player}
                  </Link>{" "}
                  <span className="muted">
                    · {softMu(p.mu_total, p.sigma, PLAYER_SIGMA_MIN).toFixed(1)} · {p.n_maps} games
                  </span>
                </p>
              ))}
          </div>
        )}
      </section>

      {banPick && (banPick.bans.length > 0 || banPick.picks.length > 0) && (
        <section className="space-y-2 border-t border-[var(--line)] pt-4">
          <h2 className="font-display text-xl">Ban / pick (last 3 patches)</h2>
          <p className="text-sm">
            <strong>Bans</strong> {banPick.bans.join(", ") || "—"}
          </p>
          <p className="text-sm">
            <strong>Picks</strong> {banPick.picks.join(", ") || "—"}
          </p>
        </section>
      )}

      <section className="space-y-3 border-t border-[var(--line)] pt-4">
        <h2 className="font-display text-xl">Recent series</h2>
        {!seriesLoaded ? (
          <div className="skeleton-block short" aria-label="Loading recent series" />
        ) : seriesError ? (
          <div className="space-y-2">
            <p className="error-banner">{seriesError}</p>
            <button type="button" className="status-pill ghost" onClick={() => setSeriesRetry((x) => x + 1)}>
              Try again
            </button>
          </div>
        ) : series.length === 0 ? (
          <p className="empty-hint">No recent series found in the selected pack.</p>
        ) : (
          <ul className="space-y-2">
            {series.map((s) => (
              <li key={s.key} className="text-sm">
                <span className="font-mono muted">{s.date}</span> · {s.league} ·{" "}
                {s.bestOf ? `Bo${s.bestOf}` : "Incomplete series"}{" "}
                <Link href={`/elo/team/${teamSlug(s.teamA)}`} className="row-link">
                  {s.teamA}
                </Link>{" "}
                {s.winsA}–{s.winsB}{" "}
                <Link href={`/elo/team/${teamSlug(s.teamB)}`} className="row-link">
                  {s.teamB}
                </Link>
              </li>
            ))}
          </ul>
        )}
        <Link className="row-link" href={`/browse?teams=${encodeURIComponent(team.team)}`}>
          Open in Matches »
        </Link>
      </section>
    </div>
  );
}
