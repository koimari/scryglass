"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  groupMapsIntoSeries,
  formatSeriesLabel,
  formatSeriesScore,
  isQuarantinedSeriesRow,
  queryMapsYears,
  queryPlayerChampStats,
  queryPlayerRole,
  queryPlayersForGame,
  type ChampAgg,
  type QueryRow,
  type SeriesCard,
} from "@/lib/duck";
import { champIconUrl } from "@/lib/format";
import type {
  CurrentMembershipContext,
  PackManifest,
  PlayerRating,
  PlayerRatingsMeta,
  PlayerRecord,
  TeamRating,
  VerifiedPlayerAffiliation,
} from "@/lib/pack";
import {
  formatWr,
  packUpdatedLabel,
  packDataThroughLabel,
  playerAdjustedRating,
  playerIdentifiabilityInfo,
  playerSigmaFloor,
  teamSlug,
  trustInfo,
} from "@/lib/pack";

type Props = {
  player: PlayerRating;
  record?: PlayerRecord;
  team?: TeamRating | null;
  baseUrl: string;
  years: number[];
  manifest: PackManifest;
  currentAffiliation: VerifiedPlayerAffiliation | null;
  membershipContext: CurrentMembershipContext;
  playerRatingsMeta: PlayerRatingsMeta | null;
  playerOrderingVerified: boolean;
};

type ChampCol = "n" | "wr" | "dpm" | "cs";

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

export function PlayerEloDetail({
  player,
  record,
  team,
  baseUrl,
  years,
  manifest,
  currentAffiliation,
  membershipContext,
  playerRatingsMeta,
  playerOrderingVerified,
}: Props) {
  const [champs, setChamps] = useState<ChampAgg[] | null>(null);
  const [expandChamps, setExpandChamps] = useState(false);
  const [col, setCol] = useState<ChampCol>("n");
  const [dir, setDir] = useState<"asc" | "desc">("desc");
  const [series, setSeries] = useState<SeriesCard[]>([]);
  const [yearFilter, setYearFilter] = useState<number | "all">("all");
  const [leagueFilter, setLeagueFilter] = useState("");
  const [sideWr, setSideWr] = useState<{
    blue: number | null;
    blueN: number;
    red: number | null;
    redN: number;
    checked: number;
  }>({ blue: null, blueN: 0, red: null, redN: 0, checked: 0 });
  const [err, setErr] = useState<string | null>(null);
  const [seriesLoaded, setSeriesLoaded] = useState(false);
  const [role, setRole] = useState<string | null>(null);

  const playerFloor = useMemo(
    () => playerSigmaFloor(playerRatingsMeta),
    [playerRatingsMeta],
  );
  const trust =
    playerFloor != null ? trustInfo(player.sigma, playerFloor, player.n_maps) : null;
  const adjusted = playerAdjustedRating(player, playerFloor);
  const identifiability = playerIdentifiabilityInfo(player);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const rows = await queryPlayerChampStats(baseUrl, years, player.player, 40);
        if (!cancelled) setChamps(rows);
      } catch {
        if (!cancelled) {
          setErr("Champion aggregates are unavailable from the public pack.");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [baseUrl, years, player.player]);

  useEffect(() => {
    let cancelled = false;
    queryPlayerRole(baseUrl, years, player.player).then((value) => {
      if (!cancelled) setRole(value);
    });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, years, player.player]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const historyTeam = record?.last_observed_team ?? player.last_team;
      if (!historyTeam) {
        setSeries([]);
        setSeriesLoaded(true);
        return;
      }
      try {
        const ys = yearFilter === "all" ? years : [yearFilter];
        const maps = await queryMapsYears(baseUrl, ys, {
          team: historyTeam,
          league: leagueFilter || undefined,
          limit: 100,
        });
        // keep games where player appears
        const withPlayer: QueryRow[] = [];
        let blueW = 0;
        let blueN = 0;
        let redW = 0;
        let redN = 0;
        for (const m of maps.slice(0, 40)) {
          const year = Number(m._year ?? String(m.date).slice(0, 4));
          const id = String(m.oe_gameid ?? "");
          try {
            const plist = await queryPlayersForGame(baseUrl, year, id);
            const me = plist.find(
              (p) => String(p.playername).toLowerCase() === player.player.toLowerCase(),
            );
            if (!me) continue;
            if (isQuarantinedSeriesRow(m)) continue;
            withPlayer.push(m);
            const side = String(me.side).toLowerCase();
            const result =
              me.result === 1 || me.result === "1"
                ? 1
                : me.result === 0 || me.result === "0"
                  ? 0
                  : null;
            if (result == null) continue;
            if (side === "blue") {
              blueN += 1;
              if (result === 1) blueW += 1;
            } else if (side === "red") {
              redN += 1;
              if (result === 1) redW += 1;
            }
          } catch {
            /* skip */
          }
        }
        if (!cancelled) {
          setSeries(groupMapsIntoSeries(withPlayer).slice(0, 10));
          setSideWr({
            blue: blueN ? blueW / blueN : null,
            blueN,
            red: redN ? redW / redN : null,
            redN,
            checked: withPlayer.length,
          });
          setSeriesLoaded(true);
        }
      } catch {
        if (!cancelled) {
          setErr("Recent player records are unavailable from the public pack.");
          setSeriesLoaded(true);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [
    baseUrl,
    years,
    player.player,
    player.last_team,
    record?.last_observed_team,
    yearFilter,
    leagueFilter,
  ]);

  const sortedChamps = useMemo(() => {
    if (!champs) return [];
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
    return expandChamps ? list : list.slice(0, 3);
  }, [champs, col, dir, expandChamps]);

  const leagues = record?.leagues ?? [];
  const roleLabel = role
    ? ({
        top: "Top",
        jungle: "Jungle",
        jng: "Jungle",
        mid: "Mid",
        bot: "Bot",
        adc: "Bot",
        sup: "Support",
      }[role.toLowerCase()] ?? role)
    : null;
  const teamLabel =
    currentAffiliation?.team ??
    record?.last_observed_team ??
    player.last_team ??
    null;

  return (
    <div className="space-y-6">
      <p className="text-xs text-[var(--ink-muted)]">
        <Link href="/elo?tab=players" className="row-link">
          ← Players
        </Link>
        {currentAffiliation && (
          <>
            {" · "}
            <Link href={`/elo/team/${teamSlug(currentAffiliation.team)}`} className="row-link">
              {currentAffiliation.team}
            </Link>
          </>
        )}
      </p>

      <header className="page-header">
        <p className="blog-kicker">
          Player · Player Dual Elo
          {record?.primary ? ` · historical primary ${record.primary}` : ""}
          {record?.intl ? " · INTL" : ""}
        </p>
        <h1 className="font-display mt-2 text-3xl">{player.player}</h1>
        <p className="text-sm muted">
          {currentAffiliation ? "Current-tournament team" : "Last observed team"}{" "}
          <strong className="text-[var(--ink)]">{teamLabel ?? "unverified"}</strong>
          {roleLabel ? (
            <>
              {" · "}Role <strong className="text-[var(--ink)]">{roleLabel}</strong>
            </>
          ) : null}
        </p>
        {currentAffiliation ? (
          <p className="text-xs muted">
            Based on an observed map in {currentAffiliation.tournament} on{" "}
            {currentAffiliation.observedAt.slice(0, 10)} and registry source{" "}
            {currentAffiliation.source}; this does not assert contract or starter status. Registry
            review due {membershipContext.reviewDueAt?.slice(0, 10) ?? "unspecified"}.
          </p>
        ) : record?.last_observed_team ? (
          <p className="text-xs muted">
            Last observed with {record.last_observed_team} on{" "}
            {String(record.last_observed_date ?? "").slice(0, 10) || "an undated historical map"};
            no current tournament affiliation is verified.
          </p>
        ) : null}
        <p className="lede text-sm" title={`${identifiability.layman}${trust ? ` ${trust.layman}` : ""}`}>
          {identifiability.layman}
          {trust ? ` ${trust.layman}` : " Player Dual Elo sigma metadata is unavailable."}
        </p>
        {!playerOrderingVerified ? (
          <p className="error-banner">
            Individual ordering and peer comparisons are withheld because team outcomes do
            not identify individual skill in this model.
          </p>
        ) : null}
        <p className="method-note">
          This profile reports the player&apos;s shared team-outcome signal without a peer
          median, difference, or rank.
        </p>
        <div className="micro-log mt-4">
          <span>
            <strong>Raw team-outcome signal</strong> {player.mu_total.toFixed(1)}
          </span>
          <span>
            <strong>Uncertainty-adjusted team-outcome signal</strong>{" "}
            {adjusted != null ? adjusted.toFixed(1) : "unavailable"}
          </span>
          <span>
            <strong>Evidence</strong>{" "}
            {identifiability.label}
            {identifiability.status === "identified" && trust ? ` · ${trust.label}` : ""}
          </span>
          <span>
            <strong>Games</strong> {player.n_maps}
          </span>
          <span>
            <strong>Regional outcome component</strong> {player.mu_regional.toFixed(1)}
          </span>
          <span>
            <strong>International-transfer component</strong> {player.mu_meta.toFixed(1)}
          </span>
          <span>
            <strong>WR</strong> {formatWr(record?.wr)}
          </span>
          <span>
            <strong>Recent last-team blue WR (n={sideWr.blueN})</strong>{" "}
            {formatWr(sideWr.blue)}
          </span>
          <span>
            <strong>Recent last-team red WR (n={sideWr.redN})</strong>{" "}
            {formatWr(sideWr.red)}
          </span>
          <span>
            <strong>Pack published</strong> {packUpdatedLabel(manifest)}
          </span>
          <span>
            <strong>Data through</strong> {packDataThroughLabel(manifest)}
          </span>
        </div>

      </header>

      <section className="space-y-3">
        <h2 className="font-display text-xl">Champions</h2>
        {err && <p className="error-banner">{err}</p>}
        {!champs && !err && <div className="skeleton-block short" />}
        {champs && champs.length === 0 && (
          <p className="empty-hint">No champion rows in pack years for this player.</p>
        )}
        {champs && champs.length > 0 && (
          <>
            <div className="table-scroll">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Champion</th>
                    {(
                      [
                        ["n", "n"],
                        ["dpm", "DPM"],
                        ["cs", "CS"],
                        ["wr", "WR"],
                      ] as const
                    ).map(([c, label]) => (
                      <th key={c} className="num">
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
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sortedChamps.map((c) => {
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
            {champs.length > 3 && (
              <button
                type="button"
                className="status-pill ghost"
                onClick={() => setExpandChamps((x) => !x)}
              >
                {expandChamps ? "Collapse" : "Expand"}
              </button>
            )}
            <p className="method-note">
              DPM and CS show their own coverage n. Missing K/D/A and gold values remain missing
              in the aggregate rather than being counted as zero.
            </p>
          </>
        )}
      </section>

      <section className="space-y-3 border-t border-[var(--line)] pt-4">
        <div className="filter-bar">
          <h2 className="font-display text-xl grow">
            Recent records · last observed team
          </h2>
          <label className="field">
            <span>Year</span>
            <select
              value={yearFilter === "all" ? "all" : String(yearFilter)}
              onChange={(e) =>
                setYearFilter(e.target.value === "all" ? "all" : Number(e.target.value))
              }
            >
              <option value="all">All pack years</option>
              {years.map((y) => (
                <option key={y} value={y}>
                  {y}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>League</span>
            <select value={leagueFilter} onChange={(e) => setLeagueFilter(e.target.value)}>
              <option value="">All</option>
              {leagues.map((L) => (
                <option key={L} value={L}>
                  {L}
                </option>
              ))}
            </select>
          </label>
        </div>
        <p className="method-note">
          This is a bounded last-observed-team lookup, not the player’s complete multi-team
          history: up to 40 player-confirmed maps are checked and up to 10 records are shown.
          Current side record coverage: n={sideWr.checked}.
        </p>
        {!seriesLoaded ? (
          <div className="skeleton-block short" aria-label="Loading recent records" />
        ) : series.length === 0 ? (
          <p className="empty-hint">
            No recent records found with this player in the selected pack scope.
          </p>
        ) : (
          <ul className="space-y-2">
            {series.map((s) => {
              const g0 = s.games[0];
              const id = String(g0?.oe_gameid ?? "");
              return (
                <li key={s.key} className="text-sm">
                  <span className="font-mono muted">{s.date}</span> · {s.league} ·{" "}
                  {formatSeriesLabel(s)}{" "}
                  {s.teamA} {formatSeriesScore(s)} {s.teamB}{" "}
                  {id && (
                    <Link
                      href={`/browse/match/${encodeURIComponent(id)}?year=${s.year}`}
                      className="row-link"
                    >
                      Board »
                    </Link>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </section>

      {team && (
        <p className="text-sm muted">
          Current org raw hierarchical team rating {team.mu_total.toFixed(1)} ·{" "}
          <Link href={`/elo/team/${teamSlug(team.team)}`} className="row-link">
            Team page
          </Link>
        </p>
      )}
    </div>
  );
}
