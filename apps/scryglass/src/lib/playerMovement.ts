import type { PlayerPositionDeltas } from "./pack";

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function positionDeltas(value: unknown): PlayerPositionDeltas | undefined {
  if (!isRecord(value) || !isRecord(value.position_deltas)) return undefined;
  return value.position_deltas as PlayerPositionDeltas;
}

/**
 * Profile payloads keep movement under the player's current tier. Older local
 * packs placed it directly under weekly, so retain that compatibility path.
 */
export function playerPositionDeltas(
  weekly: unknown,
  currentTier: string | null | undefined,
): PlayerPositionDeltas | undefined {
  if (!isRecord(weekly)) return undefined;
  const scoped = currentTier ? positionDeltas(weekly[currentTier]) : undefined;
  if (scoped) return scoped;
  const byTier = isRecord(weekly.by_tier) && currentTier
    ? positionDeltas(weekly.by_tier[currentTier])
    : undefined;
  return byTier ?? positionDeltas(weekly);
}
