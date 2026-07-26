"use client";

import { useEffect, useMemo, useState } from "react";
import {
  probabilityLabel,
  relativeLiveTime,
  secondsLabel,
  type LiveIndex,
  type LivePointer,
  type LiveSnapshot,
} from "@/lib/liveClient";

type Props = {
  initialIndex: LiveIndex | null;
  initialSnapshots: Record<string, LiveSnapshot>;
  liveIndexUrl: string;
};

function statusLabel(snapshot: LiveSnapshot): string {
  if (snapshot.status === "finished") return "Finished";
  if (snapshot.status === "unavailable") return "State unavailable";
  if (snapshot.evaluation.status === "preliminary") return "Preliminary estimate";
  if (snapshot.evaluation.status.includes("out-of-calibration")) return "Estimate withheld";
  return "State received";
}

function numericValue(value: unknown): number | null {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) ? number : null;
}

function extractExternal(text: string): { pBlue: number | null; pRed: number | null; raw: Record<string, unknown> } {
  const simple = text.trim().match(/^(\d+(?:\.\d+)?)\s*[/:]\s*(\d+(?:\.\d+)?)$/);
  if (simple) {
    return { pBlue: Number(simple[1]) / 100, pRed: Number(simple[2]) / 100, raw: {} };
  }
  const parsed = JSON.parse(text) as Record<string, unknown>;
  const evaluation = (parsed.evaluation as Record<string, unknown> | undefined) || parsed;
  const normalise = (value: unknown) => {
    const number = numericValue(value);
    return number == null ? null : number > 1 ? number / 100 : number;
  };
  const pBlue = normalise(evaluation.p_blue ?? evaluation.blue_probability ?? evaluation.blueWinProbability);
  const pRed = normalise(evaluation.p_red ?? evaluation.red_probability ?? evaluation.redWinProbability);
  return { pBlue, pRed: pRed ?? (pBlue == null ? null : 1 - pBlue), raw: parsed };
}

function CompareModel({ snapshot }: { snapshot: LiveSnapshot }) {
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [external, setExternal] = useState<{ pBlue: number | null; pRed: number | null; raw: Record<string, unknown> } | null>(null);
  const evaluation = snapshot.evaluation;
  const compare = () => {
    try {
      setExternal(extractExternal(text));
      setError(null);
    } catch {
      setExternal(null);
      setError("Paste JSON with p_blue / p_red, or use a simple pair such as 57/43.");
    }
  };
  const rows: Array<[string, string, string]> = [
    ["Series", snapshot.series_id, String(external?.raw.series_id ?? external?.raw.seriesId ?? "not supplied")],
    ["Game clock", secondsLabel(snapshot.game_state.clock_seconds), String(external?.raw.game_time ?? external?.raw.gameTime ?? "not supplied")],
    ["Blue team", snapshot.teams.blue.name, String(external?.raw.blue_team ?? "not supplied")],
    ["Red team", snapshot.teams.red.name, String(external?.raw.red_team ?? "not supplied")],
  ];
  return (
    <section className="live-compare" aria-labelledby="compare-heading">
      <div>
        <p className="blog-kicker">Audit surface</p>
        <h2 id="compare-heading" className="font-display text-lg">Compare another model</h2>
        <p className="live-muted">
          This compares outputs and the fields supplied by the other model. It does not decide which model is correct when the inputs differ.
        </p>
      </div>
      <label className="live-input-label" htmlFor="external-model">External output</label>
      <textarea
        id="external-model"
        className="live-textarea"
        value={text}
        onChange={(event) => setText(event.target.value)}
        placeholder={'{"p_blue": 0.57, "p_red": 0.43, "game_time": "31:00"}'}
        rows={5}
      />
      <button type="button" className="btn-primary live-compare-button" onClick={compare}>Run comparison</button>
      {error && <p className="error-banner">{error}</p>}
      {external && (
        <div className="live-compare-result">
          <div className="live-compare-probabilities">
            <span>Scryglass <strong>{probabilityLabel(evaluation.p_blue)}</strong></span>
            <span>Other model <strong>{probabilityLabel(external.pBlue)}</strong></span>
            <span>Difference <strong>{external.pBlue != null && evaluation.p_blue != null ? `${external.pBlue > evaluation.p_blue ? "+" : ""}${Math.round((external.pBlue - evaluation.p_blue) * 100)} pp` : "not comparable"}</strong></span>
          </div>
          <div className="table-scroll">
            <table className="data-table">
              <thead><tr><th>Input</th><th>Scryglass</th><th>Other model</th></tr></thead>
              <tbody>{rows.map(([label, ours, theirs]) => <tr key={label}><td>{label}</td><td>{String(ours)}</td><td>{String(theirs)}</td></tr>)}</tbody>
            </table>
          </div>
          <p className="live-footnote">Next diagnostic step: supply the other model&apos;s rating source, gold field, objective fields, and calibration horizon.</p>
        </div>
      )}
    </section>
  );
}

function ContributionList({ snapshot }: { snapshot: LiveSnapshot }) {
  const contributions = snapshot.evaluation.contributions || [];
  const max = Math.max(1, ...contributions.map((item) => Math.abs(item.delta_pp)));
  if (!contributions.length) {
    return <p className="live-muted">Contributions will appear when the estimate has enough verified fields to run.</p>;
  }
  return (
    <div className="live-contributions">
      {contributions.slice(0, 8).map((item) => (
        <div className="live-contribution" key={item.key}>
          <div className="live-contribution-label"><span>{item.label}</span><strong className={item.delta_pp >= 0 ? "is-blue" : "is-red"}>{item.delta_pp >= 0 ? "+" : ""}{item.delta_pp.toFixed(2)} pp</strong></div>
          <div className="live-contribution-track"><span className={item.delta_pp >= 0 ? "is-blue" : "is-red"} style={{ width: `${Math.max(8, (Math.abs(item.delta_pp) / max) * 100)}%` }} /></div>
          <small>{item.source || "derived"}</small>
        </div>
      ))}
    </div>
  );
}

function TeamSide({ side, snapshot }: { side: "blue" | "red"; snapshot: LiveSnapshot }) {
  const team = snapshot.teams[side];
  return (
    <section className={`live-team live-team-${side}`} aria-label={`${side} side`}>
      <div className="live-team-heading"><span>{side} side</span><strong>{team.score == null ? "—" : team.score}</strong></div>
      <h2 className="live-team-name">{team.name}</h2>
      <div className="live-roster">
        {team.players.length ? team.players.map((player, index) => (
          <div className="live-player" key={`${player.name}-${index}`}>
            <span className="live-role">{player.role || "—"}</span>
            <span>{player.name}</span>
            <strong>{player.champion || "Champion pending"}</strong>
          </div>
        )) : <p className="live-muted">Roster state not supplied.</p>}
      </div>
    </section>
  );
}

function SnapshotBoard({ snapshot, now }: { snapshot: LiveSnapshot; now: number }) {
  const pBlue = snapshot.evaluation.p_blue;
  const pRed = snapshot.evaluation.p_red;
  const blueGold = snapshot.game_state.gold_by_side.blue;
  const redGold = snapshot.game_state.gold_by_side.red;
  const goldDiff = blueGold != null && redGold != null ? blueGold - redGold : null;
  const blueKills = snapshot.game_state.kills_by_side.blue;
  const redKills = snapshot.game_state.kills_by_side.red;
  const stale = now > 0 && now - Date.parse(snapshot.emitted_utc) > 20_000;
  return (
    <>
      <section className="live-board anim-fade-up">
        <div className="live-board-topline">
          <div><span className={`status-pill ${stale ? "live-status-stale" : ""}`}>{stale ? "Feed stale" : statusLabel(snapshot)}</span><span className="live-source">GRID Series Events · verified state</span></div>
          <div className="live-micro">{snapshot.tournament || "Professional series"} · game {snapshot.game_number ?? "—"}</div>
        </div>
        <div className="live-teams"><TeamSide side="blue" snapshot={snapshot} /><div className="live-center-score"><span className="live-clock">{secondsLabel(snapshot.game_state.clock_seconds)}</span><strong>{blueKills ?? "—"} <em>–</em> {redKills ?? "—"}</strong><small>{goldDiff == null ? "Gold unavailable" : `${goldDiff >= 0 ? "+" : ""}${Math.round(goldDiff).toLocaleString()} gold · blue relative`}</small></div><TeamSide side="red" snapshot={snapshot} /></div>
        <div className="live-probability-section">
          <div className="live-probability-heading"><span>Game-state win chance</span><strong>{pBlue == null || pRed == null ? "Estimate withheld" : `${probabilityLabel(pBlue)} blue · ${probabilityLabel(pRed)} red`}</strong></div>
          {pBlue != null && pRed != null ? <div className="live-probability-bar" aria-label={`Blue ${probabilityLabel(pBlue)}, red ${probabilityLabel(pRed)}`}><span className="is-blue" style={{ width: `${pBlue * 100}%` }}>{probabilityLabel(pBlue)}</span><span className="is-red" style={{ width: `${pRed * 100}%` }}>{probabilityLabel(pRed)}</span></div> : <div className="live-withheld"><strong>{snapshot.evaluation.status.includes("out-of-calibration") ? "31-minute and later states are withheld by the current calibration." : "The state is visible, but a probability is not emitted until required fields are present."}</strong></div>}
          <p className="live-footnote">Conditional map-win estimate · preliminary model · not an Elo update or a wagering line.</p>
        </div>
      </section>
      <div className="live-detail-grid">
        <section className="live-detail"><p className="blog-kicker">Explanation</p><h2 className="font-display text-lg">Why this number moved</h2><ContributionList snapshot={snapshot} /></section>
        <section className="live-detail"><p className="blog-kicker">Trust ledger</p><h2 className="font-display text-lg">What the feed supplied</h2><dl className="live-ledger"><div><dt>Model</dt><dd>{snapshot.evaluation.model}</dd></div><div><dt>Clock</dt><dd>{snapshot.evaluation.minute == null ? "missing" : `${snapshot.evaluation.minute.toFixed(1)} min`}</dd></div><div><dt>Ratings</dt><dd>{snapshot.provenance.rating_pack_id || "not supplied"}</dd></div><div><dt>Missing</dt><dd>{snapshot.evaluation.missing.length ? snapshot.evaluation.missing.join(", ") : "none"}</dd></div></dl>{snapshot.evaluation.warnings.length > 0 && <ul className="live-warnings">{snapshot.evaluation.warnings.slice(0, 3).map((warning) => <li key={warning}>{warning}</li>)}</ul>}</section>
      </div>
      <CompareModel snapshot={snapshot} />
    </>
  );
}

export function LiveBoard({ initialIndex, initialSnapshots, liveIndexUrl }: Props) {
  const [index, setIndex] = useState(initialIndex);
  const [snapshots, setSnapshots] = useState(initialSnapshots);
  const [selected, setSelected] = useState(initialIndex?.series[0]?.series_id || Object.keys(initialSnapshots)[0] || "");
  const [now, setNow] = useState(0);
  const selectedSnapshot = snapshots[selected] || Object.values(snapshots)[0];

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 5_000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!index?.series?.length) return;
    let cancelled = false;
    const refresh = async () => {
      const freshIndex = await fetch(liveIndexUrl, { cache: "no-store" }).then((response) => response.ok ? response.json() as Promise<LiveIndex> : null).catch(() => null);
      if (!freshIndex || cancelled) return;
      const freshPairs = await Promise.all(freshIndex.series.map(async (pointer: LivePointer) => {
        const latestPath = pointer.latest_path || `live/series/${pointer.series_id}/latest.json`;
        const response = await fetch(pointer.latest_url || `/${latestPath.replace(/^\//, "")}`, { cache: "no-store" });
        if (!response.ok) return null;
        const latest = await response.json() as LivePointer;
        const snapshotResponse = await fetch(latest.snapshot_url || `/${latest.snapshot_path.replace(/^\//, "")}`, { cache: "no-store" });
        return snapshotResponse.ok ? [pointer.series_id, await snapshotResponse.json() as LiveSnapshot] as const : null;
      }));
      if (!cancelled) {
        setIndex(freshIndex);
        setSnapshots(Object.fromEntries(freshPairs.filter((pair): pair is [string, LiveSnapshot] => Boolean(pair))));
      }
    };
    const timer = window.setInterval(refresh, 5_000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [index?.series?.length, liveIndexUrl]);

  const pointers = useMemo(() => index?.series || [], [index]);
  if (!selectedSnapshot) {
    return <section className="live-empty"><p className="blog-kicker">Live · GRID Series Events</p><h2 className="font-display text-xl">No verified live feed is published yet.</h2><p>When the worker discovers an entitled professional series, it will write a verified snapshot here. The page does not connect to GRID directly.</p><div className="live-empty-log"><span>Feed <strong>waiting</strong></span><span>API key <strong>server-side only</strong></span><span>Ratings <strong>unchanged during live play</strong></span></div></section>;
  }
  return <div className="live-page"><header className="page-header"><p className="blog-kicker">Live · Verified game state</p><h1 className="font-display mt-2 text-3xl">What is happening now?</h1><p className="lede">Low-latency GRID state, a provisional game-state win chance, and an audit trail for comparing another model&apos;s answer.</p><div className="micro-log mt-4"><span><strong>Last verified state</strong> {relativeLiveTime(selectedSnapshot.emitted_utc)}</span><span><strong>Series</strong> {pointers.length}</span><span><strong>Refresh</strong> every 5s</span></div></header>{pointers.length > 1 && <nav className="live-series-tabs" aria-label="Live series">{pointers.map((pointer) => <button type="button" className={pointer.series_id === selected ? "is-selected" : ""} key={pointer.series_id} onClick={() => setSelected(pointer.series_id)}>{pointer.teams?.blue?.name || "Blue"} <span>vs</span> {pointer.teams?.red?.name || "Red"}</button>)}</nav>}<SnapshotBoard snapshot={selectedSnapshot} now={now} /></div>;
}
