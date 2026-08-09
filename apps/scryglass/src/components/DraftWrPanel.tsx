"use client";

import type { QueryRow } from "@/lib/duck";
import { DRAFT_UNAVAILABLE_MESSAGE } from "@/lib/publicDraftGate";

export type DraftWrResult = {
  p_blue_draft: number;
  draft_score_blue: number;
  draft_score_red: number;
  wr_bump_pp: number;
  confidence: number;
  draft_edge: number;
};

type Props = {
  map: QueryRow;
  players: QueryRow[];
  eloDiff?: number | null;
};

export function useDraftWr(
  map: QueryRow | null,
  players: QueryRow[],
  eloDiff?: number | null,
): { draft: DraftWrResult | null; loading: boolean; error: string | null } {
  void map;
  void players;
  void eloDiff;
  return {
    draft: null,
    loading: false,
    error: DRAFT_UNAVAILABLE_MESSAGE,
  };
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

/** Public claim gate for historic draft displays. */
export function DraftWrPanel({ map, players, eloDiff }: Props) {
  const { draft, loading, error } = useDraftWr(map, players, eloDiff);
  const blueName = String(map.blue_teamname ?? "Blue");
  const redName = String(map.red_teamname ?? "Red");
  const winner = winnerFromPlayers(map, players);
  const draftFav =
    draft == null ? null : draft.p_blue_draft >= 0.5 ? blueName : redName;
  const hit =
    draftFav == null ? null : draftFav.toLowerCase() === winner.toLowerCase();

  return (
    <div className="check-row draft-wr-row">
      <span>Draft WR:</span>
      {loading && <span className="status-hint">…</span>}
      {error && !loading && (
        <span className={error === DRAFT_UNAVAILABLE_MESSAGE ? "draft-unavailable" : "text-[var(--danger)]"}>
          {error}
        </span>
      )}
      {draft && !loading && (
        <>
          <span className="font-mono">
            <span className="text-[var(--side-blue)]">
              {(100 * draft.p_blue_draft).toFixed(1)}%
            </span>
            {" / "}
            <span className="text-[var(--side-red)]">
              {(100 * (1 - draft.p_blue_draft)).toFixed(1)}%
            </span>
          </span>
          <span className="font-mono text-[var(--ink-muted)]">
            score {draft.draft_score_blue.toFixed(0)}–{draft.draft_score_red.toFixed(0)}
            {draft.wr_bump_pp != null
              ? ` · bump ${draft.wr_bump_pp >= 0 ? "+" : ""}${draft.wr_bump_pp.toFixed(1)}pp`
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
        </>
      )}
    </div>
  );
}
