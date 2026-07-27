export type { LiveIndex, LivePointer, LiveSnapshot } from "./live";

function publicBlobRoot(): string | null {
  return process.env.NEXT_PUBLIC_LIVE_BLOB_BASE_URL?.replace(/\/$/, "") || null;
}

export function liveAssetUrl(relativePath: string): string {
  const root = publicBlobRoot();
  return root ? `${root}/${relativePath.replace(/^\//, "")}` : `/${relativePath.replace(/^\//, "")}`;
}

export function secondsLabel(seconds?: number | null): string {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  const total = Math.max(0, Math.round(seconds));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

export function relativeLiveTime(raw?: string | null): string {
  if (!raw) return "time unavailable";
  const timestamp = Date.parse(raw);
  if (!Number.isFinite(timestamp)) return "time unavailable";
  const seconds = Math.max(0, Math.round((Date.now() - timestamp) / 1000));
  if (seconds < 10) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.round(seconds / 60);
  return `${minutes}m ago`;
}

export function probabilityLabel(value?: number | null): string {
  return value == null ||
    !Number.isFinite(value) ||
    value < 0 ||
    value > 1
    ? "—"
    : `${Math.round(value * 100)}%`;
}

function probabilityValue(value: unknown): number | null {
  const parsed = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(parsed) || parsed < 0 || parsed > 100) return null;
  const probability = parsed > 1 ? parsed / 100 : parsed;
  return probability >= 0 && probability <= 1 ? probability : null;
}

export function validProbabilityPair(
  pBlue: number | null | undefined,
  pRed: number | null | undefined,
): boolean {
  return (
    pBlue != null &&
    pRed != null &&
    Number.isFinite(pBlue) &&
    Number.isFinite(pRed) &&
    pBlue >= 0 &&
    pBlue <= 1 &&
    pRed >= 0 &&
    pRed <= 1 &&
    Math.abs(pBlue + pRed - 1) <= 1e-6
  );
}

export function parseExternalProbability(text: string): {
  pBlue: number;
  pRed: number;
  raw: Record<string, unknown>;
} {
  const simple = text
    .trim()
    .match(/^(\d+(?:\.\d+)?)\s*[/:]\s*(\d+(?:\.\d+)?)$/);
  const raw = simple
    ? {}
    : (JSON.parse(text) as Record<string, unknown>);
  const evaluation =
    !simple &&
    raw.evaluation != null &&
    typeof raw.evaluation === "object" &&
    !Array.isArray(raw.evaluation)
      ? (raw.evaluation as Record<string, unknown>)
      : raw;
  const pBlue = probabilityValue(
    simple
      ? simple[1]
      : evaluation.p_blue ??
          evaluation.blue_probability ??
          evaluation.blueWinProbability,
  );
  const suppliedRed = probabilityValue(
    simple
      ? simple[2]
      : evaluation.p_red ??
          evaluation.red_probability ??
          evaluation.redWinProbability,
  );
  if (pBlue == null) {
    throw new Error("blue probability is missing or outside [0,1]");
  }
  const pRed = suppliedRed ?? 1 - pBlue;
  if (!validProbabilityPair(pBlue, pRed)) {
    throw new Error("blue and red probabilities must be complementary");
  }
  return { pBlue, pRed, raw };
}
