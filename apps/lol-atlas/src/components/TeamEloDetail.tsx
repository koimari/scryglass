"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { champIconUrl, formatGold, queryPlayerChampStats, type ChampAgg } from "@/lib/duck";
import type { PlayerRating, TeamRating } from "@/lib/pack";
import { softMu } from "@/lib/pack";

type Props = {
  team: TeamRating;
  roster: PlayerRating[];
  baseUrl: string;
  years: number[];
};

function ChampRow({ c }: { c: ChampAgg }) {
  const src = champIconUrl(c.champion);
  return (
    <tr>
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
}

function PlayerBlock({
  player,
  baseUrl,
  years,
}: {
  player: PlayerRating;
  baseUrl: string;
  years: number[];
}) {
  const [champs, setChamps] = useState<ChampAgg[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const rows = await queryPlayerChampStats(baseUrl, years, player.player, 5);
        if (!cancelled) setChamps(rows);
      } catch (e) {
        if (!cancelled) setErr(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [baseUrl, years, player.player]);

  return (
    <section className="border-t border-[var(--line)] pt-4 space-y-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="font-display text-lg">{player.player}</h3>
        <div className="micro-log">
          <span>
            <strong>raw</strong> {player.mu_total.toFixed(1)}
          </span>
          <span>
            <strong>adj.</strong> {softMu(player.mu_total, player.sigma).toFixed(1)}
          </span>
          <span>
            <strong>uncertainty</strong> {player.sigma.toFixed(1)}
          </span>
          <span>
            <strong>maps</strong> {player.n_maps}
          </span>
        </div>
      </div>
      {err && <p className="error-banner">{err}</p>}
      {!champs && !err && <p className="status-hint">Loading top champions…</p>}
      {champs && champs.length === 0 && (
        <p className="empty-hint">No champion rows in pack years for this player.</p>
      )}
      {champs && champs.length > 0 && (
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>Champion</th>
                <th className="num">n</th>
                <th className="num">KDA</th>
                <th className="num">Gold</th>
                <th className="num">DPM</th>
                <th className="num">CS</th>
                <th className="num">WR</th>
              </tr>
            </thead>
            <tbody>
              {champs.map((c) => (
                <ChampRow key={c.champion} c={c} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

export function TeamEloDetail({ team, roster, baseUrl, years }: Props) {
  const sorted = useMemo(
    () => [...roster].sort((a, b) => softMu(b.mu_total, b.sigma) - softMu(a.mu_total, a.sigma)),
    [roster],
  );

  return (
    <div className="space-y-6">
      <p className="text-xs text-[var(--ink-muted)]">
        <Link href="/elo" className="row-link">
          ← Elo ladders
        </Link>
      </p>
      <header className="page-header">
        <p className="blog-kicker">Team · Dual Elo</p>
        <h1 className="font-display mt-2 text-3xl">{team.team}</h1>
        <div className="micro-log mt-4">
          <span>
            <strong>raw</strong> {team.mu_total.toFixed(1)}
          </span>
          <span>
            <strong>adj.</strong> {softMu(team.mu_total, team.sigma).toFixed(1)}
          </span>
          <span>
            <strong>regional</strong> {team.mu_regional.toFixed(1)}
          </span>
          <span>
            <strong>intl.</strong> {team.mu_meta.toFixed(1)}
          </span>
          <span>
            <strong>uncertainty</strong> {team.sigma.toFixed(1)}
          </span>
        </div>
      </header>

      {sorted.length === 0 ? (
        <p className="empty-hint">
          No players with last_team matching this org in the player snapshot.
        </p>
      ) : (
        sorted.map((p) => (
          <PlayerBlock key={p.player} player={p} baseUrl={baseUrl} years={years} />
        ))
      )}
    </div>
  );
}
