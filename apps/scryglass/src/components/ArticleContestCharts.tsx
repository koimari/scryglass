"use client";

import { useMemo, useState } from "react";
import { articlePStarAtGoldB, contestBarPct, PSTAR_FX } from "@/lib/pstar";

export type ArticleCurvePoint = {
  p_win_fight: number;
  ev_contest_pp: number;
  ev_leave_pp: number;
  edge_contest_minus_leave_pp: number;
  verdict: string;
};

export type LeaveFarmRow = {
  label: string;
  leave_farm_gold: number;
  p_star_at_parity: number;
  p_star_at_parity_pct: number;
  p_star_at_B_plus_1183: number;
  p_star_at_B_plus_1183_pct: number;
};

export type GoldBRow = {
  B_gold: number;
  leave_farm_gold: number;
  objective_gold: number;
  p_star: number;
  p_star_pct: number;
};

export type FormulaHtml = {
  pStar: string;
  winProb: string;
  params: string;
};

type Props = {
  curve: ArticleCurvePoint[];
  pStar: number;
  byLeaveFarm: LeaveFarmRow[];
  byGoldB: GoldBRow[];
  formulaHtml: FormulaHtml;
};

const FARM_ALLOW = new Set(["no_farm", "one_wave", "two_waves"]);

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function interpEdge(curve: ArticleCurvePoint[], p: number) {
  const sorted = [...curve].sort((x, y) => x.p_win_fight - y.p_win_fight);
  if (p <= sorted[0].p_win_fight) return sorted[0];
  if (p >= sorted[sorted.length - 1].p_win_fight) return sorted[sorted.length - 1];
  for (let i = 0; i < sorted.length - 1; i++) {
    const a = sorted[i];
    const b = sorted[i + 1];
    if (p >= a.p_win_fight && p <= b.p_win_fight) {
      const t = (p - a.p_win_fight) / (b.p_win_fight - a.p_win_fight);
      const edge = lerp(a.edge_contest_minus_leave_pp, b.edge_contest_minus_leave_pp, t);
      const evC = lerp(a.ev_contest_pp, b.ev_contest_pp, t);
      const evL = lerp(a.ev_leave_pp, b.ev_leave_pp, t);
      return {
        p_win_fight: p,
        ev_contest_pp: evC,
        ev_leave_pp: evL,
        edge_contest_minus_leave_pp: edge,
        verdict: edge >= 0 ? "Contest preferred" : "Leave preferred",
      };
    }
  }
  return sorted[0];
}

export function ArticleContestCharts({
  curve,
  pStar,
  byLeaveFarm,
  byGoldB,
  formulaHtml,
}: Props) {
  const farms = useMemo(
    () => byLeaveFarm.filter((r) => FARM_ALLOW.has(r.label)),
    [byLeaveFarm],
  );
  const [p, setP] = useState(0.5);
  const [farmLabel, setFarmLabel] = useState("two_waves");
  const [bGold, setBGold] = useState(0);
  const point = useMemo(() => interpEdge(curve, p), [curve, p]);
  const contestBarAtB = useMemo(() => articlePStarAtGoldB(bGold), [bGold]);

  const W = 640;
  const H = 280;
  const pad = { l: 48, r: 16, t: 20, b: 36 };
  const innerW = W - pad.l - pad.r;
  const innerH = H - pad.t - pad.b;

  const xs = curve.map((c) => c.p_win_fight);
  const ys = curve.map((c) => c.edge_contest_minus_leave_pp);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys, 0) - 1;
  const yMax = Math.max(...ys, 0) + 1;

  const xScale = (v: number) => pad.l + ((v - xMin) / (xMax - xMin)) * innerW;
  const yScale = (v: number) => pad.t + ((yMax - v) / (yMax - yMin)) * innerH;

  const pathD = curve
    .map(
      (c, i) =>
        `${i === 0 ? "M" : "L"} ${xScale(c.p_win_fight)} ${yScale(c.edge_contest_minus_leave_pp)}`,
    )
    .join(" ");

  const zeroY = yScale(0);
  const starX = xScale(pStar);
  const scrubX = xScale(p);
  const scrubY = yScale(point.edge_contest_minus_leave_pp);

  const farm = farms.find((r) => r.label === farmLabel) || farms.find((r) => r.label === "two_waves");
  const goldSorted = [...byGoldB].sort((a, b) => a.B_gold - b.B_gold);

  const bMin = Math.min(-2500, ...goldSorted.map((r) => r.B_gold));
  const bMax = Math.max(4000, ...goldSorted.map((r) => r.B_gold));
  const curvePts: { B: number; pct: number }[] = [];
  for (let B = bMin; B <= bMax; B += 125) {
    const ps = articlePStarAtGoldB(B);
    if (ps != null) curvePts.push({ B, pct: 100 * ps });
  }

  const barPct = contestBarPct(pStar).toFixed(1);
  const edgeTicks = [-5, 0, 5].filter((v) => v >= yMin && v <= yMax);

  return (
    <div className="space-y-8">
      <div className="space-y-3">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h2 className="font-display text-xl">Leave vs contest by fight-win chance</h2>
            <p className="mt-1 max-w-xl text-sm text-[var(--ink-muted)]">
              Two-wave leave-farm reference. At a coin-flip fight leave still looks better. The
              contest bar — the fight-win chance where the two options tie — is{" "}
              <span className="font-mono text-[var(--ink)]">{barPct}%</span>.
            </p>
          </div>
          <div className="min-w-[12rem] text-right">
            <p className="text-xs uppercase tracking-wide text-[var(--ink-muted)]">
              Your chance to win the fight
            </p>
            <p className="font-display text-3xl tabular-nums">{(100 * p).toFixed(0)}%</p>
            <p
              className={`font-mono text-sm ${point.edge_contest_minus_leave_pp >= 0 ? "text-[var(--secondary)]" : "text-[var(--danger)]"}`}
            >
              {point.edge_contest_minus_leave_pp >= 0 ? "+" : ""}
              {point.edge_contest_minus_leave_pp.toFixed(2)}pp · {point.verdict}
            </p>
          </div>
        </div>

        <input
          type="range"
          min={15}
          max={75}
          step={1}
          value={Math.round(p * 100)}
          onChange={(e) => setP(Number(e.target.value) / 100)}
          className="w-full accent-[var(--accent)]"
          aria-label="Chance you win the fight"
        />

        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="w-full max-w-3xl"
          role="img"
          aria-label="Edge versus chance you win the fight"
        >
          <line x1={pad.l} y1={zeroY} x2={W - pad.r} y2={zeroY} stroke="var(--line)" strokeWidth={1} />
          {edgeTicks.map((v) => (
            <g key={v}>
              <line
                x1={pad.l - 4}
                y1={yScale(v)}
                x2={pad.l}
                y2={yScale(v)}
                stroke="var(--ink-muted)"
                strokeWidth={1}
              />
              <text
                x={pad.l - 6}
                y={yScale(v) + 3}
                textAnchor="end"
                className="fill-[var(--ink-muted)]"
                style={{ fontSize: 11 }}
              >
                {v > 0 ? `+${v}` : v}pp
              </text>
            </g>
          ))}
          <text
            x={W - pad.r}
            y={pad.t + 12}
            textAnchor="end"
            className="fill-[var(--secondary)]"
            style={{ fontSize: 11 }}
          >
            contest better ↑
          </text>
          <text
            x={W - pad.r}
            y={H - pad.b - 8}
            textAnchor="end"
            className="fill-[var(--danger)]"
            style={{ fontSize: 11 }}
          >
            leave better ↓
          </text>
          <path
            d={pathD}
            fill="none"
            stroke="var(--accent)"
            strokeWidth={2.25}
            pathLength={1}
            strokeDasharray="1"
            className="chart-draw"
          />
          <line
            x1={starX}
            y1={pad.t}
            x2={starX}
            y2={H - pad.b}
            stroke="var(--ink)"
            strokeWidth={1}
            strokeDasharray="4 4"
            opacity={0.55}
          />
          <circle cx={scrubX} cy={scrubY} r={6} fill="var(--canvas)" stroke="var(--accent)" strokeWidth={2} />
          <text x={starX + 4} y={pad.t + 12} className="fill-[var(--ink-muted)]" style={{ fontSize: 12 }}>
            bar {barPct}%
          </text>
          <text x={pad.l} y={H - 8} className="fill-[var(--ink-muted)]" style={{ fontSize: 12 }}>
            chance you win the fight →
          </text>
        </svg>
      </div>

      <div className="grid gap-8 lg:grid-cols-2">
        <div className="space-y-3">
          <h3 className="font-display text-lg">Contest bar by leave-farm package</h3>
          <div className="flex flex-wrap gap-2">
            {farms.map((r) => (
              <button
                key={r.label}
                type="button"
                onClick={() => setFarmLabel(r.label)}
                className={`px-3 py-1.5 text-xs uppercase tracking-wide ${
                  farmLabel === r.label
                    ? "bg-[var(--accent)] text-[var(--canvas)]"
                    : "border border-[var(--line)] text-[var(--ink-muted)] hover:text-[var(--ink)]"
                }`}
              >
                {r.label.replace(/_/g, " ")}
              </button>
            ))}
          </div>
          {farm && (
            <div className="space-y-1 text-sm text-[var(--ink-muted)]">
              <p>
                Leave farm gold:{" "}
                <span className="font-mono text-[var(--ink)]">{farm.leave_farm_gold.toFixed(0)}g</span>
              </p>
              <p>
                Contest bar at even gold:{" "}
                <span className="font-mono text-[var(--ink)]">{farm.p_star_at_parity_pct.toFixed(1)}%</span>
              </p>
              <p>
                Contest bar when ahead +1183g:{" "}
                <span className="font-mono text-[var(--ink)]">
                  {farm.p_star_at_B_plus_1183_pct.toFixed(1)}%
                </span>
              </p>
            </div>
          )}
          <svg viewBox="0 0 400 160" className="w-full" aria-label="Contest bar by leave farm">
            {farms.map((r, i) => {
              const x = 60 + i * 100;
              const h = (r.p_star_at_parity_pct / 80) * 120;
              const active = r.label === farmLabel;
              return (
                <g key={r.label}>
                  <rect
                    x={x}
                    y={120 - h}
                    width={48}
                    height={h}
                    fill={active ? "var(--accent)" : "var(--line)"}
                  />
                  <text
                    x={x + 24}
                    y={140}
                    textAnchor="middle"
                    style={{ fontSize: 10 }}
                    className="fill-[var(--ink-muted)]"
                  >
                    {r.label.replace(/_/g, " ").slice(0, 10)}
                  </text>
                  <text
                    x={x + 24}
                    y={120 - h - 6}
                    textAnchor="middle"
                    style={{ fontSize: 11 }}
                    className="fill-[var(--ink)]"
                  >
                    {r.p_star_at_parity_pct.toFixed(0)}%
                  </text>
                </g>
              );
            })}
            <line
              x1={30}
              y1={120 * (1 - Number(barPct) / 80)}
              x2={380}
              y2={120 * (1 - Number(barPct) / 80)}
              stroke="var(--ink)"
              strokeDasharray="3 3"
              opacity={0.4}
            />
            <text x={385} y={120 * (1 - Number(barPct) / 80) + 3} style={{ fontSize: 11 }} className="fill-[var(--ink-muted)]">
              {barPct}%
            </text>
          </svg>
        </div>

        <div className="space-y-3">
          <h3 className="font-display text-lg">Contest bar vs your gold lead</h3>
          <div className="rounded-[var(--radius-sm)] border border-[var(--line)] bg-[var(--surface)] px-3 py-3 space-y-2 text-sm text-[var(--ink-muted)]">
            <p className="text-[var(--ink)]">
              The contest bar is the fight-win chance where contesting and leaving are equal.
              Below it, leave looks better; above it, contest can be worth it.
            </p>
            <p>
              This chart uses the two-wave leave package (~{PSTAR_FX.leaveFarmTwoWave.toFixed(0)}g
              farm), ~{PSTAR_FX.objectiveGold.toFixed(0)}g for the objective, and a
              ±{PSTAR_FX.winKill}g fight swing. Map-win probability comes from gold lead at 10
              minutes.
            </p>
            <details className="pt-1">
              <summary className="cursor-pointer text-[var(--accent-ink)] text-xs uppercase tracking-wide">
                Show equations (optional)
              </summary>
              <div className="mt-2 space-y-1 text-[var(--ink)]">
                <div
                  className="my-2 overflow-x-auto [&_.katex]:text-[1.05em]"
                  dangerouslySetInnerHTML={{ __html: formulaHtml.pStar }}
                />
                <div
                  className="my-2 overflow-x-auto [&_.katex]:text-[1.05em]"
                  dangerouslySetInnerHTML={{ __html: formulaHtml.winProb }}
                />
                <div
                  className="my-1 overflow-x-auto text-[var(--ink-muted)] [&_.katex]:text-[0.92em]"
                  dangerouslySetInnerHTML={{ __html: formulaHtml.params }}
                />
              </div>
            </details>
          </div>
          <label className="field">
            <span>What if your gold lead is</span>
            <input
              type="number"
              step={50}
              value={bGold}
              onChange={(e) => setBGold(Number(e.target.value) || 0)}
            />
          </label>
          <p className="text-sm">
            <span className="status-pill">Contest bar</span>{" "}
            <span className="font-display text-2xl tabular-nums ml-2">
              {contestBarAtB != null ? `${(100 * contestBarAtB).toFixed(1)}%` : "—"}
            </span>
            <span className="text-[var(--ink-muted)] text-xs ml-2">
              at gold lead {bGold >= 0 ? "+" : ""}
              {bGold}g
            </span>
          </p>
          <svg viewBox="0 0 400 180" className="w-full" aria-label="Contest bar versus gold lead">
            {(() => {
              const yMin = 48;
              const yMax = 78;
              const xsScale = (v: number) => 40 + ((v - bMin) / (bMax - bMin)) * 340;
              const ysScale = (v: number) => 20 + ((yMax - v) / (yMax - yMin)) * 130;
              const d = curvePts
                .map((r, i) => `${i === 0 ? "M" : "L"} ${xsScale(r.B)} ${ysScale(r.pct)}`)
                .join(" ");
              const scrub =
                contestBarAtB != null
                  ? { x: xsScale(bGold), y: ysScale(100 * contestBarAtB) }
                  : null;
              return (
                <>
                  <path d={d} fill="none" stroke="var(--accent)" strokeWidth={2} />
                  {goldSorted.map((r) => (
                    <circle
                      key={r.B_gold}
                      cx={xsScale(r.B_gold)}
                      cy={ysScale(r.p_star_pct)}
                      r={3.5}
                      fill="var(--ink)"
                    />
                  ))}
                  {scrub && (
                    <circle
                      cx={scrub.x}
                      cy={scrub.y}
                      r={5}
                      fill="var(--canvas)"
                      stroke="var(--accent)"
                      strokeWidth={2}
                    />
                  )}
                  <text x={40} y={170} style={{ fontSize: 10 }} className="fill-[var(--ink-muted)]">
                    gold lead →
                  </text>
                </>
              );
            })()}
          </svg>
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Gold lead</th>
                  <th className="num">Contest bar</th>
                </tr>
              </thead>
              <tbody>
                {goldSorted.map((r) => (
                  <tr key={r.B_gold}>
                    <td className="font-mono">
                      {r.B_gold >= 0 ? "+" : ""}
                      {r.B_gold}
                    </td>
                    <td className="num">{r.p_star_pct.toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
