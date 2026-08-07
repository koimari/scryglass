import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

const REPO_ROOT = path.resolve(process.cwd(), "../..");
const INDEX_LOCATOR = "data/lol/v2/tierlists/index-v1.json";

type CellMeta = {
  artifact_id: string;
  scope_kind: string;
  league: string | null;
  event_kind: string | null;
  competition_tier: string | null;
  role: string;
  patch_id: string;
  as_of: string;
  locator: string;
  raw_sha256: string;
  row_count: number;
  status: string;
  fail_closed_status: string;
};

type CellPayload = {
  artifact_id: string;
  artifact_sha256: string;
  as_of: string;
  patch_id: string;
  role: string;
  status: string;
  rows: Array<{
    champion_id: string;
    champion_name: string;
    tier_value: number;
    verified_appearance_count: number;
    counterability_status: string;
  }>;
};

export type TierRow = {
  scope_id: string;
  league: string | null;
  event_kind: string | null;
  competition_tier: string | null;
  role: string;
  patch: string;
  champion: string;
  champion_id: string;
  tier_value_pp: number;
  rank: number;
  tier_bucket: "S" | "A" | "B" | "C" | "D";
  played_maps: number;
  counterability_status: string;
};

export type TierlistView = {
  status: "available";
  generated_at: string;
  development_only: boolean;
  cells_available: number;
  cells_total: number;
  options: {
    regions: string[];
    leagues: string[];
    event_kinds: string[];
    competition_tiers: string[];
    roles: string[];
    patches: string[];
  };
  rows: TierRow[];
};

export const TIERLIST_UNAVAILABLE: Record<string, unknown> = {
  status: "unavailable",
  reason: "tier-list index or cells missing or not canonical; development-only surface",
};

function sha256Hex(raw: string): string {
  return createHash("sha256").update(raw).digest("hex");
}

function canonical(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map((item) => canonical(item)).join(",")}]`;
  }
  if (value !== null && typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>).sort(([a], [b]) =>
      a < b ? -1 : a > b ? 1 : 0,
    );
    return `{${entries.map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(",")}}`;
  }
  if (typeof value === "number") {
    return String(value);
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  if (value === null) {
    return "null";
  }
  return JSON.stringify(String(value));
}

const BUCKETS = ["S", "A", "B", "C", "D"] as const;

export function loadTierlistView(): TierlistView | null {
  const absolute = path.join(REPO_ROOT, INDEX_LOCATOR);
  if (!existsSync(absolute)) return null;
  let index: { artifact_sha256: string; generated_at: string; cells: CellMeta[]; options: TierlistView["options"] };
  try {
    index = JSON.parse(readFileSync(absolute, "utf-8"));
  } catch {
    return null;
  }
  const { artifact_sha256: submitted, ...unsigned } = index;
  if (sha256Hex(canonical(unsigned)) !== submitted) return null;

  const rows: TierRow[] = [];
  let cellsAvailable = 0;
  for (const cell of index.cells) {
    if (cell.status !== "development_only") continue;
    const cellPath = path.join(REPO_ROOT, cell.locator);
    if (!existsSync(cellPath)) return null;
    const raw = readFileSync(cellPath, "utf-8");
    if (sha256Hex(raw) !== cell.raw_sha256) return null;
    let payload: CellPayload;
    try {
      payload = JSON.parse(raw);
    } catch {
      return null;
    }
    cellsAvailable += 1;
    const sorted = [...payload.rows].sort((a, b) => b.tier_value - a.tier_value);
    const total = Math.max(1, sorted.length);
    sorted.forEach((row, index_) => {
      const rank = index_ + 1;
      const quantile = (rank - 1) / total;
      const bucket = BUCKETS[Math.min(BUCKETS.length - 1, Math.floor(quantile * BUCKETS.length))];
      rows.push({
        scope_id: cell.artifact_id,
        league: cell.league,
        event_kind: cell.event_kind,
        competition_tier: cell.competition_tier,
        role: cell.role,
        patch: cell.patch_id,
        champion: row.champion_name,
        champion_id: row.champion_id,
        tier_value_pp: row.tier_value,
        rank,
        tier_bucket: bucket,
        played_maps: row.verified_appearance_count,
        counterability_status: row.counterability_status,
      });
    });
  }
  return {
    status: "available",
    generated_at: index.generated_at,
    development_only: true,
    cells_available: cellsAvailable,
    cells_total: index.cells.length,
    options: index.options,
    rows,
  };
}

const INTERNATIONAL_KINDS: Record<string, string[]> = {
  msi: ["msi"],
  ewc: ["ewc"],
  worlds: ["worlds"],
  international: ["msi", "ewc", "worlds"],
};
const REGION_LEAGUES: Record<string, string[]> = {
  europe: ["LEC"],
  americas: ["LCS"],
  asia: ["LCK", "LPL", "PCS", "VCS", "LJL"],
};

export type TierlistQuery = {
  region?: string;
  league?: string;
  international?: string;
  competition_tier?: string;
  role?: string;
  patch?: string;
  played_maps_min?: number;
};

export function filterTierlist(view: TierlistView, query: TierlistQuery): TierRow[] {
  const allowedKinds = query.international ? INTERNATIONAL_KINDS[query.international] : null;
  return view.rows.filter((row) => {
    if (query.region === "international") {
      if (!row.event_kind) return false;
    } else if (query.region) {
      const leagues = REGION_LEAGUES[query.region];
      if (!leagues || !row.league || !leagues.includes(row.league)) return false;
    }
    if (query.league && row.league !== query.league) return false;
    if (allowedKinds && (!row.event_kind || !allowedKinds.includes(row.event_kind))) return false;
    if (query.competition_tier && row.competition_tier !== query.competition_tier) return false;
    if (query.role && row.role !== query.role) return false;
    if (query.patch && row.patch !== query.patch) return false;
    const min = query.played_maps_min ?? 1;
    if (row.played_maps < min) return false;
    return true;
  });
}
