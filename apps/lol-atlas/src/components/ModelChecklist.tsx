"use client";

import {
  resolveMapWinnerSide,
  sumKnownNumbers,
  type MatchModelPrior,
  type QueryRow,
} from "@/lib/duck";
import { DraftWrPanel, type DraftStrengthInput } from "./DraftWrPanel";

type Props = {
  map: QueryRow;
  players?: QueryRow[];
  prior: MatchModelPrior | null;
  loading?: boolean;
  strength?: DraftStrengthInput;
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

export function ModelChecklist({ map, players = [], prior, loading, strength }: Props) {
  const winnerSide = resolveMapWinnerSide(map, players);
  const actualWinner =
    winnerSide === "blue"
      ? String(map.blue_teamname ?? "Blue")
      : winnerSide === "red"
        ? String(map.red_teamname ?? "Red")
        : null;
  const actualKills =
    map.total_kills != null
      ? sumKnownNumbers([map.total_kills])
      : sumKnownNumbers([map.blue_teamkills, map.red_teamkills]);

  const line = prior?.killsLine ?? null;
  const modelOver = line != null && prior?.expectedKills != null ? prior.expectedKills > line : null;
  const actualOver = line != null && actualKills != null ? actualKills > line : null;
  const killsOk =
    modelOver != null && actualOver != null ? modelOver === actualOver : null;

  const fav = prior?.expectedFavorite ?? null;
  const winnerOk =
    fav != null && actualWinner != null
      ? fav.toLowerCase() === actualWinner.toLowerCase()
      : null;
  const pBlue = prior?.pBlueWin;

  if (loading) {
    return (
      <div className="model-checklist">
        <h3>Pregame checks</h3>
        <p className="status-hint">Loading pre-match rating prior…</p>
      </div>
    );
  }

  return (
    <div className="model-checklist">
      <h3>Pregame checks</h3>
      <DraftWrPanel map={map} players={players} strength={strength} />
      <div className="check-row">
        <span>Favorite (rating):</span>
        <span className="font-medium">{fav ?? "—"}</span>
        {pBlue != null && (
          <span className="font-mono text-xs text-[var(--ink-muted)]">
            p(blue)={(100 * pBlue).toFixed(1)}%
          </span>
        )}
        <span className="text-[var(--ink-muted)]">actual {actualWinner ?? "—"}</span>
        <Mark ok={winnerOk} missing={fav == null || actualWinner == null} />
      </div>
      <div className="check-row">
        <span>Prior-date league mean:</span>
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
          actual {actualKills ?? "—"}
          {line != null && actualOver != null
            ? ` · ${actualOver ? "Over" : "Under"} ${line}`
            : ""}
        </span>
        <Mark ok={killsOk} missing={line == null || actualKills == null} />
      </div>
      <p className="text-xs text-[var(--ink-muted)]">
        Kill benchmark sample: {prior?.expectedKillsN ?? 0} same-league maps from this pack year,
        strictly before the match date. It is retrospective analysis run with a chronological
        cutoff, not a trained forecast.
      </p>
      <p className="text-xs text-[var(--ink-muted)]">{prior?.sourceNote}</p>
    </div>
  );
}
