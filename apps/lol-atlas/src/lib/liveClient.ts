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
  return value == null || !Number.isFinite(value) ? "—" : `${Math.round(value * 100)}%`;
}
