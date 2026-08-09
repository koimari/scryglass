import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

const TIERLIST_ROOT = path.join(process.cwd(), "public", "v2", "tierlists");
const PRODUCTION_INDEX_LOCATOR = "production/index-v1.json";
const SHA256_RE = /^[0-9a-f]{64}$/;
const TIERLIST_CACHE_TTL_MS = 60_000;

let tierlistCache: {
  key: string;
  expiresAt: number;
  view: TierlistView;
} | null = null;
let tierlistLoadPromise: Promise<TierlistView | null> | null = null;
let tierlistLoadKey = "";

const ROLES = ["top", "jungle", "mid", "bot", "support"] as const;
const SOURCE_MODES = ["oe_only", "oe_plus_grid"] as const;

const DEFAULT_OPTIONS: TierlistOptions = {
  leagues: [
    "AC",
    "AL",
    "AMERICAS",
    "ASI",
    "CBLOL",
    "CCWS",
    "CD",
    "CT",
    "DCUP",
    "EBL",
    "EM",
    "HC",
    "HLL",
    "HM",
    "HW",
    "IC",
    "KESPA",
    "KESPA CUP",
    "LAS",
    "LCKC",
    "LCP",
    "LCO",
    "LEC",
    "LCS",
    "LCK",
    "LPL",
    "LFL",
    "LFL2",
    "LIT",
    "LJL",
    "LLA",
    "LDL",
    "LTA",
    "LPLOL",
    "LRN",
    "LRS",
    "LVP SL",
    "NACL",
    "NEXO",
    "NL",
    "NLC",
    "PCS",
    "PRM",
    "PRMP",
    "RL",
    "ROL",
    "TCL",
    "VCS",
  ],
  event_kinds: ["asia_master", "em", "ewc", "fst", "msi", "worlds"],
  competition_tiers: ["tier1", "tier2", "tier3", "international", "interregional"],
  roles: [...ROLES],
  patches: [],
  tier_buckets: ["Z Blind", "Z Counter", "S Blind", "S Counter", "A", "B", "C", "D"],
};

type TierBucket = "Z Blind" | "Z Counter" | "S Blind" | "S Counter" | "A" | "B" | "C" | "D";
type SourceMode = (typeof SOURCE_MODES)[number];

type TierlistOptions = {
  leagues: string[];
  event_kinds: string[];
  competition_tiers: string[];
  roles: string[];
  patches: string[];
  tier_buckets: TierBucket[];
};

type CellMeta = {
  artifact_id: string;
  scope_kind: string;
  scope_id?: string;
  region?: string | null;
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

type CellRow = {
  champion_id: string;
  champion_name: string;
  tier_value: number;
  verified_appearance_count: number;
  counterability_status: string;
  counterability?: number | null;
  matchup_maps?: number;
  matchup_opponents?: number;
  blind_score_pp?: number | null;
  counter_score?: number | null;
  expected_counter_breadth?: number | null;
  countered_opponent_count?: number | null;
  countered_opponent_share?: number | null;
  tier_bucket?: TierBucket;
  rank?: number;
  rating?: number;
  previous_rating?: number | null;
  rating_delta?: number | null;
  previous_rank?: number | null;
  rank_delta?: number | null;
  movement?: "up" | "down" | "flat" | "new";
  champion_image_url?: string | null;
};

type CellPayload = {
  artifact_id: string;
  artifact_sha256: string;
  artifact_kind?: string;
  as_of: string;
  patch_id: string;
  role: string;
  status: string;
  development_only?: boolean;
  publication_eligible?: boolean;
  rows: CellRow[];
};

type TierlistIndex = {
  schema_version?: string;
  artifact_kind?: string;
  artifact_sha256: string;
  source_mode: SourceMode;
  generated_at: string;
  as_of?: string;
  development_only: boolean;
  production_eligible?: boolean;
  publication_eligible?: boolean;
  cells: CellMeta[];
  options: TierlistOptions;
  claim_ceiling?: Record<string, unknown>;
  base_url?: string | null;
};

export type TierlistScope = {
  scope_id: string;
  scope_kind: string;
  region: string | null;
  league: string | null;
  event_kind: string | null;
  competition_tier: string | null;
  role: string;
  patch: string;
  as_of: string;
  status: "production" | "unavailable";
  row_count: number;
  fail_closed_status: string;
};

export type TierRow = {
  scope_id: string;
  region: string | null;
  league: string | null;
  event_kind: string | null;
  competition_tier: string | null;
  role: string;
  patch: string;
  as_of: string;
  champion: string;
  champion_id: string;
  champion_image_url: string | null;
  tier_value_pp: number;
  rating: number;
  rating_delta: number | null;
  rank: number;
  previous_rank: number | null;
  rank_delta: number | null;
  movement: "up" | "down" | "flat" | "new";
  tier_bucket: TierBucket;
  played_maps: number;
  counterability_status: string;
  counterability: number | null;
  matchup_maps: number;
  matchup_opponents: number;
  blind_score_pp: number | null;
  counter_score: number | null;
  expected_counter_breadth: number | null;
  countered_opponent_count: number | null;
  countered_opponent_share: number | null;
};

export type TierlistView = {
  status: "available";
  api_version: "tierlist-v2";
  generated_at: string;
  as_of: string;
  development_only: false;
  publication_eligible: true;
  cells_available: number;
  cells_total: number;
  options: TierlistOptions;
  scopes: TierlistScope[];
  rows: TierRow[];
  provenance: {
    index_sha256: string;
    source: string;
    source_mode: SourceMode;
    freshness: "oe_daily_export" | "oe_with_same_day_grid_bridge";
    claim_ceiling: Record<string, unknown>;
  };
};

export const TIERLIST_UNAVAILABLE: Record<string, unknown> = {
  api_version: "tierlist-v2",
  status: "unavailable",
  reason: "approved production tier-list index is missing or failed integrity checks",
  development_only: false,
  publication_eligible: false,
  options: DEFAULT_OPTIONS,
};

export class TierlistQueryError extends Error {
  readonly code = "invalid_query";
}

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
  if (typeof value === "number") return String(value);
  if (typeof value === "boolean") return value ? "true" : "false";
  if (value === null) return "null";
  return JSON.stringify(String(value));
}

function isSha256(value: unknown): value is string {
  return typeof value === "string" && SHA256_RE.test(value);
}

function uniqueSorted(values: unknown, fallback: string[]): string[] {
  if (!Array.isArray(values)) return [...fallback];
  return [...new Set(values.filter((value): value is string => typeof value === "string" && value.length > 0))].sort();
}

function normalizeOptions(index: TierlistIndex): TierlistOptions {
  return {
    leagues: uniqueSorted(index.options?.leagues, DEFAULT_OPTIONS.leagues),
    event_kinds: uniqueSorted(index.options?.event_kinds, DEFAULT_OPTIONS.event_kinds),
    competition_tiers: uniqueSorted(index.options?.competition_tiers, DEFAULT_OPTIONS.competition_tiers),
    roles: [...ROLES].filter((role) => index.options?.roles?.includes(role)),
    patches: uniqueSorted(index.options?.patches, DEFAULT_OPTIONS.patches),
    tier_buckets: (Array.isArray(index.options?.tier_buckets)
      ? index.options.tier_buckets
      : DEFAULT_OPTIONS.tier_buckets
    ).filter((bucket): bucket is TierBucket => DEFAULT_OPTIONS.tier_buckets.includes(bucket as TierBucket)),
  };
}

function localIndexPath(): string {
  return path.join(TIERLIST_ROOT, PRODUCTION_INDEX_LOCATOR);
}

function safeLocalCellPath(locator: string): string | null {
  const basename = path.basename(locator);
  if (!basename || basename !== locator.split("/").at(-1)) return null;
  return path.join(TIERLIST_ROOT, "production", "cells", basename);
}

function parseIndex(raw: string): TierlistIndex | null {
  try {
    const index = JSON.parse(raw) as TierlistIndex;
    if (!index || typeof index !== "object") return null;
    if (!isSha256(index.artifact_sha256)) return null;
    const unsigned: Record<string, unknown> = { ...index };
    delete unsigned.artifact_sha256;
    if (sha256Hex(canonical(unsigned)) !== index.artifact_sha256) return null;
    if (index.artifact_kind !== "tier_list_index_production") return null;
    if (!SOURCE_MODES.includes(index.source_mode)) return null;
    if (index.development_only !== false || index.publication_eligible !== true) return null;
    if (index.production_eligible !== true) return null;
    if (!Array.isArray(index.cells) || index.cells.length === 0) return null;
    if (!index.options || typeof index.options !== "object") return null;
    const options = normalizeOptions(index);
    if (options.roles.length !== ROLES.length) return null;
    if (options.leagues.length === 0 || options.competition_tiers.length === 0) return null;
    const roleCoverage = new Map<string, Set<string>>();
    for (const cell of index.cells) {
      if (!ROLES.includes(cell.role as (typeof ROLES)[number])) return null;
      if (!cell.scope_id || !cell.patch_id || !cell.locator) return null;
      const key = `${cell.scope_id}|${cell.patch_id}`;
      const roles = roleCoverage.get(key) ?? new Set<string>();
      roles.add(cell.role);
      roleCoverage.set(key, roles);
    }
    if ([...roleCoverage.values()].some((roles) => roles.size !== ROLES.length)) return null;
    return { ...index, options };
  } catch {
    return null;
  }
}

async function fetchText(url: string): Promise<string | null> {
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) return null;
    return await response.text();
  } catch {
    return null;
  }
}

function resolveRemoteLocator(locator: string, indexUrl: string, baseUrl: string | null | undefined): string {
  if (/^https?:\/\//.test(locator)) return locator;
  const base = baseUrl ? new URL(baseUrl, indexUrl) : new URL("./", indexUrl);
  return new URL(locator.replace(/^\/+/, ""), base).toString();
}

function championImageUrl(championId: string): string | null {
  const match = /^riot:champion:(\d+)$/.exec(championId);
  if (!match) return null;
  return `https://cdn.communitydragon.org/latest/champion/${match[1]}/square`;
}

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function integerOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isInteger(value) && value >= 1 ? value : null;
}

function movementFor(rankDelta: number | null): "up" | "down" | "flat" | "new" {
  if (rankDelta === null) return "new";
  if (rankDelta > 0) return "up";
  if (rankDelta < 0) return "down";
  return "flat";
}

function bucketFor(rank: number, total: number): TierBucket {
  const fallback = ["A", "B", "C", "D"] as const;
  const quantile = (rank - 1) / Math.max(1, total);
  return fallback[Math.min(fallback.length - 1, Math.floor(quantile * fallback.length))];
}

function parseCell(raw: string, meta: CellMeta): CellPayload | null {
  try {
    if (!isSha256(meta.raw_sha256) || createHash("sha256").update(raw).digest("hex") !== meta.raw_sha256) {
      return null;
    }
    const payload = JSON.parse(raw) as CellPayload;
    if (!payload || typeof payload !== "object" || !Array.isArray(payload.rows)) return null;
    if (!isSha256(payload.artifact_sha256)) return null;
    const unsigned: Record<string, unknown> = { ...payload };
    delete unsigned.artifact_sha256;
    if (sha256Hex(canonical(unsigned)) !== payload.artifact_sha256) return null;
    if (payload.status !== meta.status) return null;
    if (payload.artifact_id !== meta.artifact_id) return null;
    if (payload.patch_id !== meta.patch_id || payload.role !== meta.role) return null;
    if (payload.development_only !== false || payload.publication_eligible !== true) return null;
    if (payload.status === "production" && payload.artifact_kind !== "tier_list_production") return null;
    return payload;
  } catch {
    return null;
  }
}

function scopeFromMeta(meta: CellMeta): TierlistScope {
  return {
    scope_id: meta.scope_id ?? meta.artifact_id,
    scope_kind: meta.scope_kind,
    region: meta.region ?? null,
    league: meta.league,
    event_kind: meta.event_kind,
    competition_tier: meta.competition_tier,
    role: meta.role,
    patch: meta.patch_id,
    as_of: meta.as_of,
    status: meta.status === "production" ? "production" : "unavailable",
    row_count: meta.row_count,
    fail_closed_status: meta.fail_closed_status,
  };
}

async function loadTierlistViewUncached(configuredIndexUrl: string): Promise<TierlistView | null> {
  let index: TierlistIndex | null = null;
  let indexUrl: string | null = null;
  if (configuredIndexUrl) {
    indexUrl = configuredIndexUrl;
    index = parseIndex(await fetchText(configuredIndexUrl) ?? "");
  } else if (existsSync(localIndexPath())) {
    index = parseIndex(readFileSync(localIndexPath(), "utf-8"));
  }
  if (!index) return null;

  const rows: TierRow[] = [];
  const scopes = index.cells.map(scopeFromMeta);
  let cellsAvailable = 0;
  const rawCells = await Promise.all(
    index.cells.map(async (cell) => {
      if (indexUrl) return fetchText(resolveRemoteLocator(cell.locator, indexUrl, index.base_url));
      const local = safeLocalCellPath(cell.locator);
      return local && existsSync(local) ? readFileSync(local, "utf-8") : null;
    }),
  );
  for (const [cellIndex, cell] of index.cells.entries()) {
    const raw = rawCells[cellIndex];
    if (raw === null) return null;
    const payload = parseCell(raw, cell);
    if (!payload) return null;
    if (cell.status !== "production") continue;
    cellsAvailable += 1;
    const sorted = [...payload.rows].sort((a, b) => {
      const ar = finiteNumber(a.rating) ?? a.tier_value;
      const br = finiteNumber(b.rating) ?? b.tier_value;
      return br - ar || a.champion_name.localeCompare(b.champion_name);
    });
    const total = Math.max(1, sorted.length);
    sorted.forEach((row, index_) => {
      const rank = integerOrNull(row.rank) ?? index_ + 1;
      const previousRank = integerOrNull(row.previous_rank);
      const rankDelta = finiteNumber(row.rank_delta) ?? (previousRank === null ? null : previousRank - rank);
      rows.push({
        scope_id: cell.scope_id ?? cell.artifact_id,
        region: cell.region ?? null,
        league: cell.league,
        event_kind: cell.event_kind,
        competition_tier: cell.competition_tier,
        role: cell.role,
        patch: cell.patch_id,
        as_of: cell.as_of,
        champion: row.champion_name,
        champion_id: row.champion_id,
        champion_image_url: row.champion_image_url ?? championImageUrl(row.champion_id),
        tier_value_pp: row.tier_value,
        rating: finiteNumber(row.rating) ?? row.tier_value,
        rating_delta: finiteNumber(row.rating_delta),
        rank,
        previous_rank: previousRank,
        rank_delta: rankDelta,
        movement: row.movement ?? movementFor(rankDelta),
        tier_bucket: row.tier_bucket ?? bucketFor(rank, total),
        played_maps: row.verified_appearance_count,
        counterability_status: row.counterability_status,
        counterability: finiteNumber(row.counterability),
        matchup_maps: finiteNumber(row.matchup_maps) ?? 0,
        matchup_opponents: finiteNumber(row.matchup_opponents) ?? 0,
        blind_score_pp: finiteNumber(row.blind_score_pp),
        counter_score: finiteNumber(row.counter_score),
        expected_counter_breadth: finiteNumber(row.expected_counter_breadth),
        countered_opponent_count: finiteNumber(row.countered_opponent_count),
        countered_opponent_share: finiteNumber(row.countered_opponent_share),
      });
    });
  }

  return {
    status: "available",
    api_version: "tierlist-v2",
    generated_at: index.generated_at,
    as_of: index.as_of ?? index.generated_at,
    development_only: false,
    publication_eligible: true,
    cells_available: cellsAvailable,
    cells_total: index.cells.length,
    options: index.options,
    scopes,
    rows,
    provenance: {
      index_sha256: index.artifact_sha256,
      source: indexUrl ?? "repo-production-artifact",
      source_mode: index.source_mode,
      freshness:
        index.source_mode === "oe_only"
          ? "oe_daily_export"
          : "oe_with_same_day_grid_bridge",
      claim_ceiling: index.claim_ceiling ?? {},
    },
  };
}

/** Load only the approved production artifact. Development cells never enter this API. */
export async function loadTierlistView(): Promise<TierlistView | null> {
  const configuredIndexUrl = process.env.SCRYGLASS_TIERLIST_INDEX_URL?.trim() ?? "";
  const cacheKey = configuredIndexUrl || localIndexPath();
  const now = Date.now();
  if (tierlistCache && tierlistCache.key === cacheKey && tierlistCache.expiresAt > now) {
    return tierlistCache.view;
  }
  if (tierlistLoadPromise && tierlistLoadKey === cacheKey) return tierlistLoadPromise;

  tierlistLoadKey = cacheKey;
  const promise = loadTierlistViewUncached(configuredIndexUrl)
    .then((view) => {
      if (view) {
        tierlistCache = { key: cacheKey, expiresAt: Date.now() + TIERLIST_CACHE_TTL_MS, view };
      }
      return view;
    })
    .finally(() => {
      if (tierlistLoadPromise === promise) {
        tierlistLoadPromise = null;
        tierlistLoadKey = "";
      }
    });
  tierlistLoadPromise = promise;
  return promise;
}

const INTERNATIONAL_KINDS: Record<string, string[]> = {
  asia_master: ["asia_master"],
  em: ["em"],
  msi: ["msi"],
  ewc: ["ewc"],
  fst: ["fst"],
  worlds: ["worlds"],
  international: ["asia_master", "em", "msi", "ewc", "fst", "worlds"],
};

export type TierlistQuery = {
  league?: string;
  international?: string;
  competition_tier?: string;
  role?: string;
  patch?: string;
  played_maps_min?: number;
};

function checkQueryValue(value: string | undefined, allowed: string[], label: string): void {
  if (value && !allowed.includes(value)) {
    throw new TierlistQueryError(`unknown ${label}: ${value}`);
  }
}

export function filterTierlist(view: TierlistView, query: TierlistQuery): TierRow[] {
  checkQueryValue(query.league, view.options.leagues, "league");
  checkQueryValue(query.competition_tier, view.options.competition_tiers, "competition_tier");
  checkQueryValue(query.role, view.options.roles, "role");
  checkQueryValue(query.patch, view.options.patches, "patch");
  if (query.international) {
    checkQueryValue(query.international, Object.keys(INTERNATIONAL_KINDS), "international");
  }
  if (query.league && query.international) {
    throw new TierlistQueryError("league and international event cannot be selected together");
  }
  if (query.played_maps_min !== undefined && (!Number.isInteger(query.played_maps_min) || query.played_maps_min < 1)) {
    throw new TierlistQueryError("played_maps_min must be at least 1");
  }
  const allowedKinds = query.international ? INTERNATIONAL_KINDS[query.international] : null;
  const min = query.played_maps_min ?? 1;
  return view.rows.filter((row) => {
    if (query.league && row.league !== query.league) return false;
    if (allowedKinds && (!row.event_kind || !allowedKinds.includes(row.event_kind))) return false;
    if (query.competition_tier && row.competition_tier !== query.competition_tier) return false;
    if (query.role && row.role !== query.role) return false;
    if (query.patch && row.patch !== query.patch) return false;
    return row.played_maps >= min;
  });
}
