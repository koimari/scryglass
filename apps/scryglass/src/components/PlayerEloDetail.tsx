"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  groupMapsIntoSeries,
  queryMapsYears,
  queryPlayerChampStats,
  queryPlayerRole,
  queryPlayersForGame,
  type ChampAgg,
  type QueryRow,
  type SeriesCard,
} from "@/lib/duck";
import { champIconUrl, formatGold } from "@/lib/format";
import type { PackManifest, PlayerRating, PlayerRecord, TeamRating } from "@/lib/pack";
import {
  formatTrustCell,
  formatWr,
  packUpdatedLabel,
  PLAYER_SIGMA_MIN,
  softMu,
  teamSlug,
  trustInfo,
} from "@/lib/pack";
import profileStyles from "./ProfileHeader.module.css";

type Props = {
  player: PlayerRating;
  record?: PlayerRecord;
  team?: TeamRating | null;
  peers: PlayerRating[];
  intlPeers: PlayerRating[];
  baseUrl: string;
  years: number[];
  manifest: PackManifest;
  medianModeDefault?: "league" | "intl";
};

type ChampCol = "n" | "wr" | "kda" | "gold" | "dpm" | "cs";

function median(nums: number[]): number | null {
  if (!nums.length) return null;
  const s = [...nums].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

export function PlayerEloDetail({
  player,
  record,
  team,
  peers,
  intlPeers,
  baseUrl,
  years,
  manifest,
}: Props) {
  const [champs, setChamps] = useState<ChampAgg[] | null>(null);
  const [expandChamps, setExpandChamps] = useState(false);
  const [col, setCol] = useState<ChampCol>("n");
  const [dir, setDir] = useState<"asc" | "desc">("desc");
  const [series, setSeries] = useState<SeriesCard[]>([]);
  const [yearFilter, setYearFilter] = useState<number | "all">("all");
  const [leagueFilter, setLeagueFilter] = useState("");
  const [compare, setCompare] = useState<"league" | "intl">("league");
  const [sideWr, setSideWr] = useState<{ blue: number | null; red: number | null }>({
    blue: null,
    red: null,
  });
  const [err, setErr] = useState<string | null>(null);
  const [seriesLoaded, setSeriesLoaded] = useState(false);
  const [role, setRole] = useState<string | null>(null);

  const trust = trustInfo(player.sigma, PLAYER_SIGMA_MIN, player.n_maps);
  const leagueAware = softMu(player.mu_total, player.sigma, PLAYER_SIGMA_MIN);

  const peerMedian = useMemo(() => {
    const pool =
      compare === "intl"
        ? intlPeers
        : peers;
    // crude: same last_team primary league via peers list already scoped by page
    const vals = pool.map((p) => softMu(p.mu_total, p.sigma, PLAYER_SIGMA_MIN));
    return median(vals);
  }, [intlPeers, peers, compare]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const rows = await queryPlayerChampStats(baseUrl, years, player.player, 40);
        if (!cancelled) setChamps(rows);
      } catch (e) {
        if (!cancelled) setErr(e instanceof Error ? e.message : String(e));
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
      if (!player.last_team) {
        setSeries([]);
        setSeriesLoaded(true);
        return;
      }
      try {
        const ys = yearFilter === "all" ? years : [yearFilter];
        const maps = await queryMapsYears(baseUrl, ys, {
          team: player.last_team,
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
            withPlayer.push(m);
            const side = String(me.side);
            const won = Number(me.result) === 1;
            if (side === "Blue") {
              blueN += 1;
              if (won) blueW += 1;
            } else {
              redN += 1;
              if (won) redW += 1;
            }
          } catch {
            /* skip */
          }
        }
        if (!cancelled) {
          setSeries(groupMapsIntoSeries(withPlayer).slice(0, 10));
          setSideWr({
            blue: blueN ? blueW / blueN : null,
            red: redN ? redW / redN : null,
          });
          setSeriesLoaded(true);
        }
      } catch (e) {
        if (!cancelled) {
          setErr(e instanceof Error ? e.message : String(e));
          setSeriesLoaded(true);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [baseUrl, years, player.player, player.last_team, yearFilter, leagueFilter]);

  const sortedChamps = useMemo(() => {
    if (!champs) return [];
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
  const teamLabel = team?.team ?? player.last_team ?? null;

  return (
    <div className="profile-page player-profile-page space-y-6">
      <p className="text-xs text-[var(--ink-muted)]">
        <Link href="/elo?tab=players" className="row-link">
          ← Players
        </Link>
        {player.last_team && (
          <>
            {" · "}
            <Link href={`/elo/team/${teamSlug(player.last_team)}`} className="row-link">
              {player.last_team}
            </Link>
          </>
        )}
      </p>

      <header className={profileStyles.header}>
        <div className={profileStyles.identity}>
          <p className={profileStyles.scope}>
            Player · {record?.primary ?? "Dual Elo"}
            {record?.intl ? " · International" : ""}
          </p>
          <h1>{player.player}</h1>
          <p className={profileStyles.affiliation}>
            Current team <strong>{teamLabel ?? "—"}</strong>
            {roleLabel ? (
              <>
                {" · "}Role <strong>{roleLabel}</strong>
              </>
            ) : null}
          </p>
          <p className={profileStyles.summary} title={trust.layman}>
            {trust.layman}
          </p>
        </div>
        <div className={profileStyles.metrics}>
          <span className={profileStyles.primary}>
            <strong>Adjusted rating</strong> <em>{leagueAware.toFixed(1)}</em>
          </span>
          <span>
            <strong>Raw rating</strong> {player.mu_total.toFixed(1)}
          </span>
          <span>
            <strong>Evidence</strong> {formatTrustCell(trust)}
          </span>
          <span>
            <strong>Games</strong> {player.n_maps}
          </span>
          <span>
            <strong>Regional Elo</strong> {player.mu_regional.toFixed(1)}
          </span>
          <span>
            <strong>International Elo</strong> {player.mu_meta.toFixed(1)}
          </span>
          <span>
            <strong>WR</strong> {formatWr(record?.wr)}
          </span>
          <span>
            <strong>Blue WR</strong> {formatWr(sideWr.blue)}
          </span>
          <span>
            <strong>Red WR</strong> {formatWr(sideWr.red)}
          </span>
          <span>
            <strong>Updated</strong> {packUpdatedLabel(manifest)}
          </span>
        </div>

        <div className={`${profileStyles.actions} ${profileStyles.comparison}`}>
          <label className="field">
            <span>Compare with</span>
            <select
              value={compare}
              onChange={(e) => setCompare(e.target.value as "league" | "intl")}
            >
              <option value="league">League median</option>
              <option value="intl">International median</option>
            </select>
          </label>
          <div className="status-hint">
            {peerMedian != null
              ? `${compare === "intl" ? "International" : record?.primary ?? "League"} median ${peerMedian.toFixed(1)} · difference ${(leagueAware - peerMedian).toFixed(1)} Elo`
              : "Comparison median unavailable"}
          </div>
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
                        ["kda", "KDA"],
                        ["gold", "Gold"],
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
            {champs.length > 3 && (
              <button
                type="button"
                className="status-pill ghost"
                onClick={() => setExpandChamps((x) => !x)}
              >
                {expandChamps ? "Collapse" : "Expand"}
              </button>
            )}
          </>
        )}
      </section>

      <section className="space-y-3 border-t border-[var(--line)] pt-4">
        <div className="filter-bar">
          <h2 className="font-display text-xl grow">Recent series</h2>
          <label className="field">
            <span>Year</span>
            <select
              value={yearFilter === "all" ? "all" : String(yearFilter)}
              onChange={(e) =>
                setYearFilter(e.target.value === "all" ? "all" : Number(e.target.value))
              }
            >
              <option value="all">All</option>
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
        {!seriesLoaded ? (
          <div className="skeleton-block short" aria-label="Loading recent series" />
        ) : series.length === 0 ? (
          <p className="empty-hint">
            No recent series found with this player in the selected pack scope.
          </p>
        ) : (
          <ul className="space-y-2">
            {series.map((s) => {
              const g0 = s.games[0];
              const id = String(g0?.oe_gameid ?? "");
              return (
                <li key={s.key} className="text-sm">
                  <span className="font-mono muted">{s.date}</span> · {s.league} · Bo{s.bestOf}{" "}
                  {s.teamA} {s.winsA}–{s.winsB} {s.teamB}{" "}
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
          Current org rating {softMu(team.mu_total, team.sigma).toFixed(1)} league-aware ·{" "}
          <Link href={`/elo/team/${teamSlug(team.team)}`} className="row-link">
            Team page
          </Link>
        </p>
      )}
    </div>
  );
}
