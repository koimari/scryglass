export const TIER_ROLE_ORDER = ["top", "jungle", "mid", "bot", "support"] as const;

export type TierBoardMode = "first_pick" | "blind" | "counter" | "responses" | "unpicked";
export type TierRankedMode = Exclude<TierBoardMode, "responses" | "unpicked">;

export type TierBucket = "Z Blind" | "Z Counter" | "S Blind" | "S Counter" | "A" | "B" | "C" | "D";

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

export type ResponseMatrix = {
  champions: Array<{ champion: string; champion_id: string }>;
  edge_pp: Array<Array<number | null>>;
  interval_low_pp: Array<Array<number | null>>;
  interval_high_pp: Array<Array<number | null>>;
  evidence: Array<Array<"supported" | "limited" | null>>;
  effective_maps: Array<Array<number | null>>;
  basis?: Array<Array<"observed_pair_plus_model" | "atom_and_strength_inferred" | "strength_only_inferred" | null>>;
  grade_thresholds_pp?: { S: number; A: number; B: number; C: number };
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
  response_matrix?: ResponseMatrix;
};

export type StructuralTrait = {
  dimension: string;
  label: string;
};

export type StructuralChampion = {
  champion_id: string;
  champion: string;
  champion_image_url: string | null;
  positions: string[];
  roles: string[];
  profile_status: "family_only" | "atom_detail";
  traits: StructuralTrait[];
};

export type StructuralSimilarity = {
  schema_version: "scryglass:champion-structural-similarity:v1";
  source_atom_bridge_sha256: string;
  minimum_similarity: number;
  weights: Record<string, number>;
  champions: StructuralChampion[];
  similarity: number[][];
};

export type ViableCandidate = {
  candidate: StructuralChampion;
  reference: StructuralChampion;
  similarity: number;
  sharedRoles: string[];
  sharedTraits: StructuralTrait[];
};

export function signedPercentagePoints(value: number | null | undefined): string | null {
  if (value === null || value === undefined || !Number.isFinite(value)) return null;
  return `${value >= 0 ? "+" : ""}${value.toFixed(1)} percentage points`;
}

export function firstPickMetric(row: TierRow | undefined): string {
  if (!row) return "No ranked pick";
  return signedPercentagePoints(row.tier_value_pp) ?? row.tier_bucket;
}

export function rowsForMode(rows: TierRow[], mode: TierRankedMode): TierRow[] {
  if (mode === "blind") {
    return rows.filter((row) => signedPercentagePoints(row.blind_score_pp) !== null);
  }
  if (mode === "counter") {
    return rows.filter(
      (row) => row.countered_opponent_count !== null && row.countered_opponent_count !== undefined,
    );
  }
  return rows;
}

export function filterRowsByMinimumGames(rows: TierRow[], minimumGames: number): TierRow[] {
  const threshold = Number.isFinite(minimumGames) ? Math.max(1, Math.floor(minimumGames)) : 1;
  return rows.filter((row) => row.played_maps >= threshold);
}

export function viableCandidates(
  library: StructuralSimilarity | undefined,
  playedPatchRoleRows: TierRow[],
  visibleReferenceRows: TierRow[],
  role: string,
  selectedReferenceId = "",
): ViableCandidate[] {
  if (!library || !role || !library.champions.length) return [];
  const profileIndex = new Map(library.champions.map((profile, index) => [profile.champion_id, index]));
  const patchWidePickedIds = new Set(playedPatchRoleRows.map((row) => row.champion_id));
  const visibleReferenceIds = new Set(visibleReferenceRows.map((row) => row.champion_id));
  const referenceIds = selectedReferenceId
    ? (visibleReferenceIds.has(selectedReferenceId) ? [selectedReferenceId] : [])
    : [...visibleReferenceIds];
  const references = referenceIds.flatMap((championId) => {
    const index = profileIndex.get(championId);
    return index === undefined ? [] : [{ profile: library.champions[index], index }];
  });
  if (!references.length) return [];

  const candidates: ViableCandidate[] = [];
  library.champions.forEach((candidate, candidateIndex) => {
    if (!candidate.positions.includes(role) || patchWidePickedIds.has(candidate.champion_id)) return;
    const best = references
      .map((reference) => ({
        reference: reference.profile,
        similarity: library.similarity[candidateIndex]?.[reference.index],
      }))
      .filter((item): item is { reference: StructuralChampion; similarity: number } => (
        typeof item.similarity === "number" && Number.isFinite(item.similarity)
      ))
      .sort((left, right) => right.similarity - left.similarity)[0];
    if (!best || best.similarity < library.minimum_similarity) return;
    const referenceRoles = new Set(best.reference.roles);
    const referenceTraits = new Map(best.reference.traits.map((trait) => [trait.dimension, trait.label]));
    candidates.push({
      candidate,
      reference: best.reference,
      similarity: best.similarity,
      sharedRoles: candidate.roles.filter((value) => referenceRoles.has(value)),
      sharedTraits: candidate.traits.filter((trait) => referenceTraits.get(trait.dimension) === trait.label),
    });
  });
  return candidates.sort((left, right) => (
    right.similarity - left.similarity || left.candidate.champion.localeCompare(right.candidate.champion)
  ));
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

export function filterRowsByRegion(
  rows: TierRow[],
  scopes: TierScope[],
  patch: string,
  regionId: string,
): TierRow[] {
  if (!regionId) return rows;
  const allowedByRole = new Map<string, Set<string>>();
  for (const role of TIER_ROLE_ORDER) {
    const view = regionalViewForRole(scopes, patch, role, regionId);
    allowedByRole.set(role, new Set((view?.rows ?? []).map((row) => row.champion_id)));
  }
  return rows.filter((row) => allowedByRole.get(row.role)?.has(row.champion_id));
}

export type MatchupGrade = "S" | "A" | "B" | "C" | "D";

export function matchupGrade(edgePp: number): MatchupGrade {
  if (edgePp >= 7.5) return "S";
  if (edgePp >= 3.0) return "A";
  if (edgePp >= -3.0) return "B";
  if (edgePp >= -7.5) return "C";
  return "D";
}
