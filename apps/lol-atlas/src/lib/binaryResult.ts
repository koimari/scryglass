export function parseBinaryResult(value: unknown): 0 | 1 | null {
  if (value === true) return 1;
  if (value === false) return 0;

  if (typeof value === "bigint") {
    const normalized = value.toString();
    if (normalized === "1") return 1;
    if (normalized === "0") return 0;
    return null;
  }

  if (typeof value === "number") {
    if (!Number.isFinite(value) || !Number.isInteger(value)) return null;
    return value === 1 ? 1 : value === 0 ? 0 : null;
  }

  if (typeof value === "string") {
    const text = value.trim().toLowerCase();
    if (!text) return null;
    if (text === "1" || text === "true" || text === "t" || text === "yes" || text === "y") {
      return 1;
    }
    if (text === "0" || text === "false" || text === "f" || text === "no" || text === "n") {
      return 0;
    }
  }

  return null;
}
