/**
 * Validated evidence states for public ratings (issue #46).
 *
 * Mirrors `lol_kills/ratings/evidence.py` exactly: the public UI no longer
 * calls a rating "Settled" because sigma reached a fixed floor.  The evidence
 * state is derived from the validated contract fields stored in the pack:
 *
 * - interval width (two-sided 95% interval, display points)
 * - relative precision ((tightest_sigma / sigma)^2 within the same scope)
 * - stability (per-game / weekly posterior displacement)
 * - freshness (days since the row's most recent game vs pack source as-of)
 * - support coverage (sample support vs the coverage target)
 * - fallback / active / disconnected / ood flags
 *
 * Sigma and map count remain separate diagnostics.  Settled requires
 * strictly greater-than-95% relative precision, known bounded stability,
 * fresh inputs, full support coverage, and active eligibility, with no
 * fallback, disconnection, or out-of-distribution flag.  Everything else
 * fails closed to an explicit state; a row without the required fields is
 * "unsupported" and can never be settled.
 */

export type EvidenceFields = {
  evidence_interval_width?: number | null;
  evidence_precision_ratio?: number | null;
  evidence_stability?: number | null;
  evidence_freshness_days?: number | null;
  evidence_support_coverage?: number | null;
  evidence_fallback?: number | null;
  evidence_active?: number | null;
  evidence_disconnected?: number | null;
  evidence_ood?: number | null;
  evidence_state?: string | null;
};

export type EvidenceState =
  | "settled"
  | "observed"
  | "thin"
  | "stale"
  | "inactive"
  | "disconnected"
  | "wide_interval"
  | "fallback"
  | "ood"
  | "unsupported";

export type EvidenceInfo = {
  state: EvidenceState;
  label: string;
  headroom: number | null;
  sigma: number | null;
  games: number | null;
  layman: string;
  detail: string;
};

// Exact contract thresholds — must stay in sync with
// lol_kills/ratings/evidence.py and the adversarial tests.
export const EVIDENCE_CONTRACT = {
  freshDays: 14,
  activeDays: 60,
  staleDays: 90,
  wideIntervalWidth: 200,
  settledPrecisionRatio: 0.95,
  observedPrecisionRatio: 0.8,
  settledStability: 6,
} as const;

const LABELS: Record<EvidenceState, string> = {
  settled: "Settled",
  observed: "Observed",
  thin: "Thin",
  stale: "Stale",
  inactive: "Inactive",
  disconnected: "Disconnected",
  wide_interval: "Wide interval",
  fallback: "Fallback",
  ood: "Out of distribution",
  unsupported: "Unsupported",
};

const LAYMAN: Record<EvidenceState, string> = {
  settled:
    "Evidence basis is met: tight interval, stable and fresh inputs, full support coverage, active eligibility.",
  observed:
    "Evidence basis is partial: interval is reasonably tight and inputs are supported, but not enough for Settled.",
  thin: "Still moving — fewer informative games than a settled row; treat the number gently.",
  stale: "No recent games in the source window; the estimate may be outdated.",
  inactive: "Row is not currently active (no games in the active window or no roster anchor).",
  disconnected: "No supported league anchor; the row cannot be ranked on league evidence.",
  wide_interval: "Interval is too wide for a stable label; treat the number as approximate.",
  fallback: "Rests on a fallback/neutral prior rather than observed games.",
  ood: "Row lies outside the supported distribution; result is unavailable.",
  unsupported: "This pack predates the evidence contract or the row lacks required evidence fields.",
};

function number(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  return value;
}

function flag(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isInteger(value)) return null;
  return value === 0 || value === 1 ? value : null;
}

/** Derive the fail-closed evidence state from the validated contract. */
export function evidenceState(fields: EvidenceFields): EvidenceState {
  const intervalWidth = number(fields.evidence_interval_width);
  const precision = number(fields.evidence_precision_ratio);
  const stability = number(fields.evidence_stability);
  const freshness = number(fields.evidence_freshness_days);
  const coverage = number(fields.evidence_support_coverage);
  const fallback = flag(fields.evidence_fallback);
  const active = flag(fields.evidence_active);
  const disconnected = flag(fields.evidence_disconnected);
  const ood = flag(fields.evidence_ood);

  if (
    intervalWidth === null ||
    precision === null ||
    coverage === null ||
    fallback === null ||
    active === null ||
    disconnected === null ||
    ood === null
  ) {
    return "unsupported";
  }
  if (ood === 1) return "ood";
  if (disconnected === 1) return "disconnected";
  if (fallback === 1) return "fallback";
  if (freshness === null || freshness > EVIDENCE_CONTRACT.staleDays) return "stale";
  if (active !== 1) return "inactive";
  if (intervalWidth > EVIDENCE_CONTRACT.wideIntervalWidth) return "wide_interval";
  if (
    precision > EVIDENCE_CONTRACT.settledPrecisionRatio &&
    stability !== null &&
    stability <= EVIDENCE_CONTRACT.settledStability &&
    freshness <= EVIDENCE_CONTRACT.freshDays &&
    coverage >= 1 &&
    active === 1
  ) {
    return "settled";
  }
  if (precision > EVIDENCE_CONTRACT.observedPrecisionRatio && stability !== null && coverage >= 0.5) {
    return "observed";
  }
  return "thin";
}

export function evidenceInfo(
  fields: EvidenceFields,
  sigma: number | null | undefined,
  games: number | null | undefined,
): EvidenceInfo {
  const state = evidenceState(fields);
  const interval = number(fields.evidence_interval_width);
  const headroom =
    interval !== null && sigma != null && Number.isFinite(sigma) ? Math.max(0, interval) : null;
  const detail = [
    `basis: ${state}`,
    interval !== null ? `95% interval ${interval.toFixed(0)} pt` : "interval unknown",
    fields.evidence_precision_ratio != null
      ? `precision ${(100 * fields.evidence_precision_ratio).toFixed(1)}%`
      : "precision unknown",
    fields.evidence_freshness_days != null
      ? `${Math.round(fields.evidence_freshness_days)}d since last game`
      : "freshness unknown",
    fields.evidence_support_coverage != null
      ? `support ${(100 * fields.evidence_support_coverage).toFixed(0)}%`
      : "support unknown",
    fields.evidence_stability != null
      ? `stability ${fields.evidence_stability.toFixed(1)}`
      : "stability unknown",
  ].join(" · ");
  return {
    state,
    label: LABELS[state],
    headroom,
    sigma: sigma ?? null,
    games: games ?? null,
    layman: LAYMAN[state],
    detail,
  };
}

/** Read evidence fields from a pack row, tolerating legacy packs. */
export function evidenceFields(row: Record<string, unknown>): EvidenceFields {
  return {
    evidence_interval_width: row.evidence_interval_width as number | null | undefined,
    evidence_precision_ratio: row.evidence_precision_ratio as number | null | undefined,
    evidence_stability: row.evidence_stability as number | null | undefined,
    evidence_freshness_days: row.evidence_freshness_days as number | null | undefined,
    evidence_support_coverage: row.evidence_support_coverage as number | null | undefined,
    evidence_fallback: row.evidence_fallback as number | null | undefined,
    evidence_active: row.evidence_active as number | null | undefined,
    evidence_disconnected: row.evidence_disconnected as number | null | undefined,
    evidence_ood: row.evidence_ood as number | null | undefined,
    evidence_state: row.evidence_state as string | null | undefined,
  };
}

export function formatEvidenceCell(info: EvidenceInfo): string {
  return info.label;
}
