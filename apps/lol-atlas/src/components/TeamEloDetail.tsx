"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  groupMapsIntoSeries,
  formatSeriesLabel,
  formatSeriesScore,
  isQuarantinedSeriesRow,
  queryMapsYears,
  queryRosterChampStats,
  type ChampAgg,
  type SeriesCard,
} from "@/lib/duck";
import { champIconUrl, normalizePatchVersion } from "@/lib/format";
import type {
  CurrentMembershipContext,
  PlayerRating,
  PlayerRatingsMeta,
  TeamRating,
  TeamRatingsMeta,
  TeamRecord,
  VerifiedTeamAffiliation,
} from "@/lib/pack";
import {
  formatWr,
  packUpdatedLabel,
  packDataThroughLabel,
  playerAdjustedRating,
  playerIdentifiabilityInfo,
  playerSigmaFloor,
  playerSlug,
  teamBoundRating,
  teamEvidenceInfo,
  teamRatingContract,
  teamSlug,
  trustInfo,
  type PackManifest,
} from "@/lib/pack";

type Props = {
  team: TeamRating;
  roster: PlayerRating[];
  record?: TeamRecord;
  baseUrl: string;
  years: number[];
  manifest: PackManifest;
  membershipContext: CurrentMembershipContext;
  teamAffiliation: VerifiedTeamAffiliation | null;
  teamRatingsMeta: TeamRatingsMeta | null;
  playerRatingsMeta: PlayerRatingsMeta | null;
  playerOrderingVerified: boolean;
};

type ChampCol = "n" | "wr" | "dpm" | "cs";
type CountedChampion = { champion: string; count: number };
type BanPickSummary = {
  bans: CountedChampion[];
  picks: CountedChampion[];
  patches: string[];
  banMaps: number;
  pickMaps: number;
};

function compareNullableMetric(
  a: number | null,
  b: number | null,
  sign: 1 | -1,
): number {
  if (a == null && b == null) return 0;
  if (a == null) return 1;
  if (b == null) return -1;
  return sign * (a - b);
}

export function latestNormalizedPatches(
  values: unknown[],
  limit = 3,
): string[] {
  return [...new Set(values.map(normalizePatchVersion).filter(
    (patch): patch is string => patch != null,
  ))]
    .sort((a, b) => {
      const [aMajor, aMinor] = a.split(".").map(Number);
      const [bMajor, bMinor] = b.split(".").map(Number);
      return aMajor - bMajor || aMinor - bMinor;
    })
    .slice(-Math.max(0, limit));
}

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
    const sign: 1 | -1 = dir === "asc" ? 1 : -1;
    list.sort((a, b) => {
      switch (col) {
        case "wr":
          return sign * (a.wr - b.wr);
        case "dpm":
          return compareNullableMetric(a.dpm, b.dpm, sign);
        case "cs":
          return compareNullableMetric(a.cs, b.cs, sign);
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
                  {c.dpm != null ? `${c.dpm.toFixed(0)} (n=${c.dpmN})` : "—"}
                </td>
                <td className="num">
                  {c.cs != null ? `${c.cs.toFixed(0)} (n=${c.csN})` : "—"}
                </td>
                <td className="num">{(100 * c.wr).toFixed(0)}%</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function exportParticipantCsv(
  team: string,
  roster: PlayerRating[],
  playerFloor: number | null,
) {
  const lines = [
    "player,raw_team_outcome_signal,uncertainty_adjusted_team_outcome_signal,evidence_sigma,games,last_observed_team,outcome_identifiability",
    ...roster.map(
      (p) => {
        const adjusted = playerAdjustedRating(p, playerFloor);
        return `"${p.player}",${p.mu_total.toFixed(2)},${adjusted?.toFixed(2) ?? ""},${p.sigma.toFixed(2)},${p.n_maps},"${p.last_team ?? ""}","${playerIdentifiabilityInfo(p).status}"`;
      },
    ),
  ];
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${team.replace(/\s+/g, "_")}_current_tournament_participants.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

export function TeamEloDetail({
  team,
  roster,
  record,
  baseUrl,
  years,
  manifest,
  membershipContext,
  teamAffiliation,
  teamRatingsMeta,
  playerRatingsMeta,
  playerOrderingVerified,
}: Props) {
  const teamContract = teamRatingContract(teamRatingsMeta);
  const teamBound = teamBoundRating(team, teamContract);
  const teamEvidence = teamEvidenceInfo(team.sigma, teamContract, record?.games);
  const playerFloor = playerSigmaFloor(playerRatingsMeta);
  const participantsByName = useMemo(
    () => [...roster].sort((a, b) => a.player.localeCompare(b.player)),
    [roster],
  );

  const [byPlayer, setByPlayer] = useState<Record<string, ChampAgg[]> | null>(null);
  const [champExpand, setChampExpand] = useState<Record<string, boolean>>({});
  const [series, setSeries] = useState<SeriesCard[]>([]);
  const [draftEdge, setDraftEdge] = useState<number | null>(null);
  const [banPick, setBanPick] = useState<BanPickSummary | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [seriesLoaded, setSeriesLoaded] = useState(false);
  const [seriesError, setSeriesError] = useState<string | null>(null);
  const [seriesRetry, setSeriesRetry] = useState(0);

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
      } catch {
        if (!cancelled) {
          setErr("Champion aggregates are unavailable from the public pack.");
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
        const displayableMaps = maps.filter(
          (map) => !isQuarantinedSeriesRow(map),
        );
        const grouped = groupMapsIntoSeries(displayableMaps);
        if (cancelled) return;
        setSeries(grouped.slice(0, 8));
        setSeriesLoaded(true);

        // Draft edge over recent games with full picks
        const edges: number[] = [];
        const banCount = new Map<string, number>();
        const pickCount = new Map<string, number>();
        const patches = latestNormalizedPatches(
          displayableMaps.map((map) => map.patch),
        );
        const recentPatches = new Set(patches);
        let banMaps = 0;
        let pickMaps = 0;

        for (const [mapIndex, m] of displayableMaps.entries()) {
          const blue = [1, 2, 3, 4, 5].map((i) => String(m[`blue_pick${i}`] ?? "")).filter(Boolean);
          const red = [1, 2, 3, 4, 5].map((i) => String(m[`red_pick${i}`] ?? "")).filter(Boolean);
          if (
            mapIndex < 25 &&
            blue.length === 5 &&
            red.length === 5 &&
            edges.length < 12
          ) {
            try {
              const res = await fetch("/api/draft-wr", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  blue,
                  red,
                  league: String(m.league ?? ""),
                  patch: String(m.patch ?? ""),
                }),
              });
              if (res.ok) {
                const ds = (await res.json()) as {
                  draft_edge: number;
                };
                const isBlue = String(m.blue_teamname) === team.team;
                edges.push(isBlue ? ds.draft_edge : -ds.draft_edge);
              }
            } catch {
              /* ignore */
            }
          }
          const normalizedPatch = normalizePatchVersion(m.patch);
          if (normalizedPatch && recentPatches.has(normalizedPatch)) {
            const side = String(m.blue_teamname) === team.team ? "blue" : "red";
            const mapBans: string[] = [];
            const mapPicks: string[] = [];
            for (let i = 1; i <= 5; i++) {
              const b = String(m[`${side}_ban${i}`] ?? "");
              const p = String(m[`${side}_pick${i}`] ?? "");
              if (b) mapBans.push(b);
              if (p) mapPicks.push(p);
            }
            if (mapBans.length) banMaps += 1;
            if (mapPicks.length) pickMaps += 1;
            for (const champion of mapBans) {
              banCount.set(champion, (banCount.get(champion) ?? 0) + 1);
            }
            for (const champion of mapPicks) {
              pickCount.set(champion, (pickCount.get(champion) ?? 0) + 1);
            }
          }
        }
        setDraftEdge(
          edges.length
            ? edges.reduce((a, b) => a + b, 0) / edges.length
            : null,
        );
        const top = (m: Map<string, number>, n: number) =>
          [...m.entries()]
            .sort((a, b) => b[1] - a[1])
            .slice(0, n)
            .map(([champion, count]) => ({ champion, count }));
        setBanPick({
          bans: top(banCount, 8),
          picks: top(pickCount, 8),
          patches,
          banMaps,
          pickMaps,
        });
      } catch {
        if (!cancelled) {
          setSeries([]);
          setSeriesError("Could not load recent records. Try again when the pack is available.");
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
    <div className="space-y-6 print-area">
      <p className="text-xs text-[var(--ink-muted)]">
        <Link href="/elo" className="row-link">
          ← Ratings
        </Link>
      </p>
      <header className="page-header">
        <p className="blog-kicker">
          Team · Hierarchical Bradley–Terry
          {record?.primary ? ` · historical home ${record.primary}` : ""}
          {record?.intl ? " · INTL" : ""}
        </p>
        <h1 className="font-display mt-2 text-3xl">{team.team}</h1>
        <p className="lede text-sm">
          Team strength from canonical series outcomes.{" "}
          {teamAffiliation
            ? `Participants below have an observed map in ${teamAffiliation.tournament}; this is not a claim about contracts, starters, or substitutes.`
            : "Current membership and participants are withheld because this pack does not provide a complete, current registry proof."}
        </p>
        {teamAffiliation ? (
          <p className="text-xs muted">
            Current scope: {teamAffiliation.tier.toUpperCase()} · {teamAffiliation.league} ·{" "}
            {teamAffiliation.tournament}. Registry checked{" "}
            {membershipContext.checkedAt?.slice(0, 10) ?? "on an unspecified date"}; next review due{" "}
            {membershipContext.reviewDueAt?.slice(0, 10) ?? "on an unspecified date"}.
          </p>
        ) : null}
        <div className="micro-log mt-4">
          <span>
            <strong>Raw rating</strong> {team.mu_total.toFixed(1)}
          </span>
          <span>
            <strong>{teamContract?.boundLabel ?? "Conservative bound"}</strong>{" "}
            {teamBound != null ? teamBound.toFixed(1) : "unavailable"}
          </span>
          <span title={teamEvidence.layman}>
            <strong>Uncertainty</strong> {teamEvidence.label}
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
          {draftEdge != null && (
            <span>
              <strong>Draft score edge</strong> {draftEdge >= 0 ? "+" : ""}
              {draftEdge.toFixed(1)}
            </span>
          )}
          <span>
            <strong>Pack published</strong> {packUpdatedLabel(manifest)}
          </span>
          <span>
            <strong>Data through</strong> {packDataThroughLabel(manifest)}
          </span>
        </div>
        <div className="filter-bar mt-4">
          <Link
            className="btn-primary"
            href={`/browse/head-to-head?a=${encodeURIComponent(team.team)}`}
          >
            Head-to-head
          </Link>
          <button
            type="button"
            className="status-pill ghost"
            onClick={() => exportParticipantCsv(team.team, participantsByName, playerFloor)}
          >
            Export CSV
          </button>
          <button type="button" className="status-pill ghost" onClick={() => window.print()}>
            Print / PDF
          </button>
        </div>
      </header>

      <section className="space-y-4">
        <h2 className="font-display text-xl">Current-tournament participants · name order</h2>
        <p className="text-sm muted">
          A player appears here only after an observed map for this team in the active registered
          tournament. The pack does not encode contract, starter, or substitute status. Players
          are shown alphabetically; Player Dual Elo values are shared team-outcome signals and
          are not used to rank this roster.
        </p>
        <p className="method-note">
          Champion K/D/A and gold averages are withheld until the pack publishes per-metric
          coverage denominators; missing source values must not be counted as zero.
        </p>
        {!playerOrderingVerified && roster.length > 0 ? (
          <p className="error-banner">
            Individual ordering is withheld because team outcomes do not identify individual
            skill in this model.
          </p>
        ) : null}
        {roster.length === 0 ? (
          <p className="empty-hint">
            No current-tournament participant is verified for this team in the pack. Try Matches for
            historical lineups.
          </p>
        ) : (
          participantsByName.map((p) => {
            const champs = byPlayer?.[p.player] ?? [];
            const expanded = champExpand[p.player];
            const pTrust =
              playerFloor != null ? trustInfo(p.sigma, playerFloor, p.n_maps) : null;
            const identifiability = playerIdentifiabilityInfo(p);
            const adjusted = playerAdjustedRating(p, playerFloor);
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
                      <strong>Raw team-outcome signal</strong> {p.mu_total.toFixed(1)}
                    </span>
                    <span>
                      <strong>Uncertainty-adjusted team-outcome signal</strong>{" "}
                      {adjusted != null ? adjusted.toFixed(1) : "unavailable"}
                    </span>
                    <span title={`${identifiability.layman}${pTrust ? ` ${pTrust.layman}` : ""}`}>
                      <strong>Evidence</strong> {identifiability.label}
                      {identifiability.status === "identified" && pTrust
                        ? ` · ${pTrust.label}`
                        : ""}
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
      </section>

      {banPick && (banPick.bans.length > 0 || banPick.picks.length > 0) && (
        <section className="space-y-2 border-t border-[var(--line)] pt-4">
          <h2 className="font-display text-xl">
            Ban / pick (latest {banPick.patches.length} normalized patches)
          </h2>
          <p className="method-note">
            Patch scope: {banPick.patches.join(", ") || "unavailable"}. Counts use maps with that
            metric present as the denominator.
          </p>
          <p className="text-sm">
            <strong>Bans</strong>{" "}
            {banPick.bans
              .map(({ champion, count }) => `${champion} ${count}/${banPick.banMaps}`)
              .join(", ") || "—"}
          </p>
          <p className="text-sm">
            <strong>Picks</strong>{" "}
            {banPick.picks
              .map(({ champion, count }) => `${champion} ${count}/${banPick.pickMaps}`)
              .join(", ") || "—"}
          </p>
        </section>
      )}

      <section className="space-y-3 border-t border-[var(--line)] pt-4">
        <h2 className="font-display text-xl">Recent records</h2>
        {!seriesLoaded ? (
          <div className="skeleton-block short" aria-label="Loading recent records" />
        ) : seriesError ? (
          <div className="space-y-2">
            <p className="error-banner">{seriesError}</p>
            <button type="button" className="status-pill ghost" onClick={() => setSeriesRetry((x) => x + 1)}>
              Try again
            </button>
          </div>
        ) : series.length === 0 ? (
          <p className="empty-hint">No recent records found in the selected pack.</p>
        ) : (
          <ul className="space-y-2">
            {series.map((s) => (
              <li key={s.key} className="text-sm">
                <span className="font-mono muted">{s.date}</span> · {s.league} ·{" "}
                {formatSeriesLabel(s)}{" "}
                <Link href={`/elo/team/${teamSlug(s.teamA)}`} className="row-link">
                  {s.teamA}
                </Link>{" "}
                {formatSeriesScore(s)}{" "}
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
