"use client";

import { useEffect, useState } from "react";
import type { QueryRow } from "@/lib/duck";
import { resolveDraftLineup } from "@/lib/draftLineup";
import { sortPlayersByRole } from "@/lib/format";

export type DraftWrResult = {
  p_blue_draft: number;
  draft_score_blue: number;
  draft_score_red: number;
  wr_bump_pp: number;
  confidence: number;
  draft_edge: number;
  raw?: {
    p_blue: number;
    score_blue: number;
    score_red: number;
    edge: number;
    source: string;
  };
  contextualized?: {
    p_blue: number;
    score_blue: number;
    score_red: number;
    edge: number;
    source: string;
  } | null;
  strength?: {
    team_elo_diff: number | null;
    player_elo_diff: number | null;
    source: string;
  };
  uncertainty?: {
    edge_se_logit: number;
    p_blue_95: [number, number];
    method: string;
  };
  explanation?: {
    edge: number;
    composition_edge: number;
    side_advantage: number;
    champions: Array<{
      champion: string;
      side: "blue" | "red";
      role: string;
      direct_effect: number;
      team_synergy: number;
      enemy_interaction: number;
      edge_contribution: number;
      uncertainty_logit: number;
      evidence: { games: number; shrinkage: number; label: string; uncertainty_logit: number };
    }>;
    reconciles: boolean;
    attribution: string;
  };
  contextualized_uncertainty?: {
    p_blue_95: [number, number];
    method: string;
  } | null;
};

type Props = {
  map: QueryRow;
  players: QueryRow[];
  strength?: DraftStrengthInput;
};

export type DraftStrengthInput = {
  teamEloDiff?: number | null;
  playerEloDiff?: number | null;
  source?: string | null;
};

function rolesFromPlayers(players: QueryRow[], side: "Blue" | "Red"): string[] | null {
  const ordered = sortPlayersByRole(players.filter((p) => String(p.side) === side));
  if (ordered.length < 5) return null;
  return ordered.map((p) => String(p.position).toLowerCase());
}

export function useDraftWr(
  map: QueryRow | null,
  players: QueryRow[],
  strength?: DraftStrengthInput,
): { draft: DraftWrResult | null; loading: boolean; error: string | null } {
  const [draft, setDraft] = useState<DraftWrResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    if (!map) return;
    const lineup = resolveDraftLineup(map, players);
    if (!lineup) {
      queueMicrotask(() => {
        if (cancelled) return;
        setDraft(null);
        setError("Draft rating unavailable · champion lineup missing from source");
        setLoading(false);
      });
      return () => {
        cancelled = true;
      };
    }
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch("/api/draft-wr", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            blue: lineup.blue,
            red: lineup.red,
            league: map.league ? String(map.league) : null,
            patch: map.patch ? String(map.patch) : null,
            team_elo_diff: strength?.teamEloDiff ?? null,
            player_elo_diff: strength?.playerEloDiff ?? null,
            strength_source: strength?.source ?? null,
            blue_team: map.blue_teamname ? String(map.blue_teamname) : null,
            red_team: map.red_teamname ? String(map.red_teamname) : null,
            blue_players: players
              .filter((p) => String(p.side) === "Blue")
              .map((p) => String(p.playername ?? ""))
              .filter(Boolean),
            red_players: players
              .filter((p) => String(p.side) === "Red")
              .map((p) => String(p.playername ?? ""))
              .filter(Boolean),
            blue_roles: lineup.blueRoles ?? rolesFromPlayers(players, "Blue"),
            red_roles: lineup.redRoles ?? rolesFromPlayers(players, "Red"),
          }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || `draft-wr ${res.status}`);
        if (!cancelled) {
          setDraft({
            p_blue_draft: Number(data.p_blue_draft),
            draft_score_blue: Number(data.draft_score_blue),
            draft_score_red: Number(data.draft_score_red),
            wr_bump_pp: Number(data.wr_bump_pp),
            confidence: Number(data.confidence),
            draft_edge: Number(data.draft_edge),
            raw: data.raw,
            contextualized: data.contextualized,
            strength: data.strength,
            uncertainty: data.uncertainty,
            explanation: data.explanation,
            contextualized_uncertainty: data.contextualized_uncertainty,
          });
        }
      } catch (e) {
        if (!cancelled) {
          setDraft(null);
          setError(e instanceof Error ? e.message : String(e));
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [map, players, strength]);

  return { draft, loading, error };
}

function teamWon(map: QueryRow, side: "blue" | "red"): boolean {
  const key = side === "blue" ? "blue_result" : "red_result";
  const yKey = side === "blue" ? "y_blue_win" : "y_red_win";
  const v = map[key] ?? map[yKey];
  if (v === true || v === 1 || v === "1") return true;
  if (typeof v === "number" && Number.isFinite(v) && v >= 0.5) return true;
  if (typeof v === "string" && ["true", "w", "win", "victory"].includes(v.toLowerCase())) {
    return true;
  }
  const opp = side === "blue" ? map.red_result : map.blue_result;
  if (opp === false || opp === 0 || opp === "0") return true;
  return false;
}

function actualWinnerName(map: QueryRow): string {
  if (teamWon(map, "blue")) return String(map.blue_teamname ?? "Blue");
  if (teamWon(map, "red")) return String(map.red_teamname ?? "Red");
  return String(map.blue_teamname ?? "Blue");
}

function winnerFromPlayers(map: QueryRow, players: QueryRow[]): string {
  if (players.length) {
    const blue = players.find((p) => String(p.side) === "Blue");
    const red = players.find((p) => String(p.side) === "Red");
    if (blue && Number(blue.result) === 1) return String(map.blue_teamname ?? "Blue");
    if (red && Number(red.result) === 1) return String(map.red_teamname ?? "Red");
    if (blue && (blue.result === true || blue.result === "1")) {
      return String(map.blue_teamname ?? "Blue");
    }
    if (red && (red.result === true || red.result === "1")) {
      return String(map.red_teamname ?? "Red");
    }
  }
  return actualWinnerName(map);
}

/** Inline draft WR strip for scoreboard / checklist. */
export function DraftWrPanel({ map, players, strength }: Props) {
  const { draft, loading, error } = useDraftWr(map, players, strength);
  const blueName = String(map.blue_teamname ?? "Blue");
  const redName = String(map.red_teamname ?? "Red");
  const winner = winnerFromPlayers(map, players);
  const contextual = draft?.contextualized ?? null;
  const primaryP = contextual?.p_blue ?? draft?.p_blue_draft ?? null;
  const draftFav =
    primaryP == null ? null : primaryP >= 0.5 ? blueName : redName;
  const hit =
    draftFav == null ? null : draftFav.toLowerCase() === winner.toLowerCase();

  return (
    <div className="check-row draft-wr-row">
      <span>Draft WR:</span>
      {loading && <span className="status-hint">…</span>}
      {error && !loading && (
        <span className="text-[var(--danger)]">{error}</span>
      )}
      {draft && !loading && (
        <>
          <span className="font-mono">
            {contextual && <span className="text-[var(--ink)]">context </span>}
            <span className="text-[var(--side-blue)]">
              {(100 * primaryP!).toFixed(1)}%
            </span>
            {" / "}
            <span className="text-[var(--side-red)]">
              {(100 * (1 - primaryP!)).toFixed(1)}%
            </span>
          </span>
          {contextual && draft.raw && (
            <span className="font-mono text-[var(--ink-muted)]">
              raw composition {(100 * draft.raw.p_blue).toFixed(1)}% / {(100 * (1 - draft.raw.p_blue)).toFixed(1)}%
            </span>
          )}
          <span className="font-mono text-[var(--ink-muted)]">
            score {(contextual?.score_blue ?? draft.draft_score_blue).toFixed(0)}–
            {(contextual?.score_red ?? draft.draft_score_red).toFixed(0)}
            {draft.wr_bump_pp != null
              ? ` · composition ${draft.wr_bump_pp >= 0 ? "+" : ""}${draft.wr_bump_pp.toFixed(1)}pp`
              : ""}
          </span>
          <span className="text-[var(--ink-muted)]">
            fav {draftFav}
            {" · actual "}
            {winner}
          </span>
          {hit == null ? (
            <span className="check-mark na">—</span>
          ) : hit ? (
            <span className="check-mark ok">✓</span>
          ) : (
            <span className="check-mark bad">✗</span>
          )}
          {draft.strength && (
            <span className="w-full text-xs text-[var(--ink-muted)]">
              strength: team {draft.strength.team_elo_diff == null ? "—" : `${draft.strength.team_elo_diff >= 0 ? "+" : ""}${draft.strength.team_elo_diff.toFixed(0)}`} Elo
              {draft.strength.player_elo_diff != null
                ? ` · players ${draft.strength.player_elo_diff >= 0 ? "+" : ""}${draft.strength.player_elo_diff.toFixed(0)} Elo`
                : " · player rating unavailable"}
              {` · ${draft.strength.source}`}
            </span>
          )}
          {draft.explanation && (
            <details className="mt-2 w-full">
              <summary className="cursor-pointer text-[var(--ink-muted)]">
                Full composition explanation · {draft.explanation.reconciles ? "ledger reconciles" : "ledger unavailable"}
              </summary>
              <div className="mt-2 text-xs text-[var(--ink-muted)]">
                <p>
                  Each pick has a role-aware direct estimate, its share of own-team synergy, and its
                  share of interactions with all five enemies. Pair effects are split between both
                  champions; no single H2H pair is treated as the whole explanation.
                </p>
                {draft.uncertainty && (
                  <p className="mt-1">
                    Raw composition range: {(100 * draft.uncertainty.p_blue_95[0]).toFixed(1)}–
                    {(100 * draft.uncertainty.p_blue_95[1]).toFixed(1)}% blue.
                  </p>
                )}
                {draft.contextualized_uncertainty && (
                  <p className="mt-1">
                    Context range, conditional on the supplied strength inputs: {(
                      100 * draft.contextualized_uncertainty.p_blue_95[0]
                    ).toFixed(1)}–{(
                      100 * draft.contextualized_uncertainty.p_blue_95[1]
                    ).toFixed(1)}% blue.
                  </p>
                )}
                <div className="table-scroll mt-2">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Champion</th>
                        <th>Direct</th>
                        <th>Own team</th>
                        <th>All enemies</th>
                        <th>Evidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {draft.explanation.champions.map((row) => {
                        const signed = (value: number) => `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
                        return (
                          <tr key={`${row.side}-${row.role}-${row.champion}`}>
                            <td>
                              <span className={row.side === "blue" ? "text-[var(--side-blue)]" : "text-[var(--side-red)]"}>
                                {row.champion}
                              </span>{" "}
                              <span className="text-[var(--ink-muted)]">({row.role})</span>
                            </td>
                            <td className="num font-mono">{signed(row.direct_effect)}</td>
                            <td className="num font-mono">{signed(row.team_synergy)}</td>
                            <td className="num font-mono">{signed(row.enemy_interaction)}</td>
                            <td>{row.evidence.label} · {row.evidence.games} games</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <p className="mt-1">Side advantage is reported separately from composition contributions.</p>
              </div>
            </details>
          )}
        </>
      )}
    </div>
  );
}
