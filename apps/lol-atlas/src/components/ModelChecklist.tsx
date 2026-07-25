"use client";

import type { MatchModelPrior, QueryRow } from "@/lib/duck";
import { DraftWrPanel } from "./DraftWrPanel";

type Props = {
  map: QueryRow;
  players?: QueryRow[];
  prior: MatchModelPrior | null;
  loading?: boolean;
  eloDiff?: number | null;
};

function Mark({ ok, missing }: { ok: boolean | null; missing?: boolean }) {
  if (missing || ok == null) return <span className="check-mark na">—</span>;
  return ok ? (
    <span className="check-mark ok" aria-label="correct">
      ✓
    </span>
  ) : (
    <span className="check-mark bad" aria-label="miss">
      ✗
    </span>
  );
}

function resolveWinner(map: QueryRow, players: QueryRow[]): string {
  for (const p of players) {
    if (Number(p.result) === 1 || p.result === true || p.result === "1") {
      return String(p.side) === "Blue"
        ? String(map.blue_teamname ?? "Blue")
        : String(map.red_teamname ?? "Red");
    }
  }
  const br = map.blue_result ?? map.y_blue_win;
  if (br === true || br === 1 || br === "1" || (typeof br === "number" && br >= 0.5)) {
    return String(map.blue_teamname ?? "Blue");
  }
  const rr = map.red_result;
  if (rr === true || rr === 1 || rr === "1" || (typeof rr === "number" && rr >= 0.5)) {
    return String(map.red_teamname ?? "Red");
  }
  if (rr === false || rr === 0 || rr === "0") return String(map.blue_teamname ?? "Blue");
  return String(map.red_teamname ?? "Red");
}

export function ModelChecklist({ map, players = [], prior, loading, eloDiff }: Props) {
  const actualWinner = resolveWinner(map, players);
  const actualKills =
    map.total_kills != null
      ? Number(map.total_kills)
      : (Number(map.blue_teamkills) || 0) + (Number(map.red_teamkills) || 0);

  const line = prior?.killsLine ?? null;
  const modelOver = line != null && prior?.expectedKills != null ? prior.expectedKills > line : null;
  const actualOver = line != null ? actualKills > line : null;
  const killsOk =
    modelOver != null && actualOver != null ? modelOver === actualOver : null;

  const fav = prior?.expectedFavorite ?? null;
  const winnerOk =
    fav != null ? fav.toLowerCase() === actualWinner.toLowerCase() : null;
  const pBlue = prior?.pBlueWin;

  if (loading) {
    return (
      <div className="model-checklist">
        <h3>Model vs actual</h3>
        <p className="status-hint">Loading Dual Elo prior…</p>
      </div>
    );
  }

  return (
    <div className="model-checklist">
      <h3>Model vs actual</h3>
      <DraftWrPanel map={map} players={players} eloDiff={eloDiff} />
      <div className="check-row">
        <span>Winner (Elo):</span>
        <span className="font-medium">{fav ?? "—"}</span>
        {pBlue != null && (
          <span className="font-mono text-xs text-[var(--ink-muted)]">
            p(blue)={(100 * pBlue).toFixed(1)}%
          </span>
        )}
        <span className="text-[var(--ink-muted)]">actual {actualWinner}</span>
        <Mark ok={winnerOk} missing={fav == null} />
      </div>
      <div className="check-row">
        <span>Expected kills:</span>
        <span className="font-mono">
          {prior?.expectedKills != null ? prior.expectedKills.toFixed(1) : "—"}
        </span>
        <span>
          {line != null ? (
            <>
              {modelOver ? "Over" : "Under"} {line}
            </>
          ) : (
            "—"
          )}
        </span>
        <span className="text-[var(--ink-muted)]">
          actual {actualKills}
          {line != null ? ` · ${actualOver ? "Over" : "Under"} ${line}` : ""}
        </span>
        <Mark ok={killsOk} missing={line == null} />
      </div>
      <p className="text-xs text-[var(--ink-muted)]">{prior?.sourceNote}</p>
    </div>
  );
}
