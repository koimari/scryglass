export const TIER_ROLE_ORDER = ["top", "jungle", "mid", "bot", "support"] as const;

export type TierBoardMode = "first_pick" | "blind" | "counter" | "responses" | "regions";

export type TierBucket = "Z Blind" | "Z Counter" | "S Blind" | "S Counter" | "A" | "B" | "C" | "D";

export type MatchupProfile = {
  champion: string;
  champion_id: string;
  model_edge_pp: number;
  posterior_interval_pp: { low: number; high: number };
  posterior_positive_probability?: number;
  effective_maps: number;
  series_count: number;
  evidence_status: "supported" | "limited";
  evidence_source?: string;
};

export type TierRow = {
  scope_id: string;
  role: string;
  patch: string;
  champion: string;
  champion_id: string;
  champion_image_url: string | null;
  rank: number;
  rank_delta: number | null;
  movement: "up" | "down" | "flat" | "new";
  tier_bucket: TierBucket;
  played_maps: number;
  tier_value_pp?: number | null;
  counterability_status: "available" | "unavailable" | string;
  matchup_maps: number;
  matchup_opponents: number;
  blind_score_pp?: number | null;
  counter_score?: number | null;
  countered_opponent_count?: number | null;
  countered_opponent_share?: number | null;
  expected_counter_breadth: number | null;
  matchup_profile?: MatchupProfile[];
};

export type RegionalRow = {
  champion: string;
  champion_id: string;
  regional_rank: number;
  global_rank: number;
  strength_score_pp: number;
  played_maps: number;
  sample_status: "thin" | "observed" | string;
};

export type RegionalView = {
  id: string;
  label: string;
  maps: number;
  basis: string;
  rows: RegionalRow[];
};

export type TierScope = {
  scope_id: string;
  scope_kind: "patch";
  role: string;
  patch: string;
  as_of: string;
  status: "production" | "unavailable";
  row_count: number;
  regional_views?: RegionalView[];
};

export function signedPp(value: number | null | undefined): string | null {
  if (value === null || value === undefined || !Number.isFinite(value)) return null;
  return `${value >= 0 ? "+" : ""}${value.toFixed(1)} pp`;
}

export function firstPickMetric(row: TierRow | undefined): string {
  if (!row) return "No ranked pick";
  return signedPp(row.tier_value_pp) ?? row.tier_bucket;
}

export function rowsForMode(rows: TierRow[], mode: TierBoardMode): TierRow[] {
  if (mode === "blind") {
    return rows.filter((row) => signedPp(row.blind_score_pp) !== null);
  }
  if (mode === "counter") {
    return rows.filter(
      (row) => row.countered_opponent_count !== null && row.countered_opponent_count !== undefined,
    );
  }
  return rows;
}

export function regionalOptions(scopes: TierScope[], patch: string): Array<{ id: string; label: string }> {
  const options = new Map<string, string>();
  for (const scope of scopes) {
    if (scope.patch !== patch) continue;
    for (const view of scope.regional_views ?? []) options.set(view.id, view.label);
  }
  return [...options.entries()]
    .map(([id, label]) => ({ id, label }))
    .sort((left, right) => left.label.localeCompare(right.label));
}

export function regionalViewForRole(
  scopes: TierScope[],
  patch: string,
  role: string,
  regionId: string,
): RegionalView | undefined {
  return scopes
    .find((scope) => scope.patch === patch && scope.role === role)
    ?.regional_views?.find((view) => view.id === regionId);
}
