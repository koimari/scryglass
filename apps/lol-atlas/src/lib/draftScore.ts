/**
 * League-calibrated Draft Score (v3) — TypeScript port of lol_kills.draft_score.
 * Loads a slim runtime artifact (no Python / no warehouse at request time).
 */
import { readFileSync } from "fs";
import path from "path";
import {
  CompositionRuntimeUnavailableError,
  DraftIntegrityInputError,
  compositionDraftCatalog,
  compositionDraftScore,
  compositionPartialDraftScore,
  compositionRuntimeMetadata,
  normalizeCompositionPatch,
} from "./draftComposition";
import {
  DRAFT_PICK_ORDER,
  DRAFT_POLICY_MIN_ROLE_GAMES,
  DRAFT_ROLES,
} from "./draftRules";

const DEFAULT_TEMP = 1.4;
const BLUE_SIDE_BONUS = 0.03;
const PRIOR_N = 25;

export { DRAFT_PICK_ORDER, DRAFT_ROLES };
export type DraftRole = (typeof DRAFT_ROLES)[number];
export type DraftSide = "blue" | "red";

const LEAGUE_TIER: Record<string, string> = {
  LCK: "tier1",
  LPL: "tier1",
  LEC: "west",
  LCS: "west",
  LTA: "west",
  "LTA N": "west",
  "LTA S": "west",
  CBLOL: "americas",
  AMERICAS: "americas",
  PCS: "asia_reg",
  VCS: "asia_reg",
  LJL: "asia_reg",
  LCP: "asia_reg",
  TCL: "asia_reg",
  MSI: "intl",
  EWC: "intl",
  FST: "intl",
  Worlds: "intl",
};

const CHAMP_ALIASES: Record<string, string> = {
  kaisa: "Kai'Sa",
  "kai'sa": "Kai'Sa",
  mel: "Mel",
  jarvan: "Jarvan IV",
  "jarvan iv": "Jarvan IV",
  j4: "Jarvan IV",
  "monkey king": "Wukong",
  wukong: "Wukong",
  ksante: "K'Sante",
  "k'sante": "K'Sante",
  renata: "Renata Glasc",
  nunu: "Nunu & Willump",
  "dr mundo": "Dr. Mundo",
  mundo: "Dr. Mundo",
  reksai: "Rek'Sai",
  belveth: "Bel'Veth",
  kogmaw: "Kog'Maw",
  chogath: "Cho'Gath",
  velkoz: "Vel'Koz",
  khazix: "Kha'Zix",
  mf: "Miss Fortune",
  "miss fortune": "Miss Fortune",
  tf: "Twisted Fate",
  lee: "Lee Sin",
  xin: "Xin Zhao",
  locke: "Locke",
  "corvin locke": "Locke",
};

type RolePair = { logit: number; n?: number };

type DraftRuntime = {
  version: number;
  kill_beta: Record<string, number>;
  win_delta: Record<string, number>;
  champ_game_counts: Record<string, number>;
  role_pairs: Record<string, RolePair>;
  calibration: {
    by_league?: Record<string, CalRow>;
    by_tier?: Record<string, CalRow>;
    joint_logistic_global?: CalRow;
    residual_ridge?: { coef_dp_per_logit?: number };
  };
};

type CalRow = {
  coef_draft?: number;
  coef_elo?: number;
  intercept?: number;
};

type Scale = {
  source: string;
  temp: number;
  coef_elo: number;
  intercept: number;
  dp_per_logit: number;
};

export type DraftScoreInput = {
  blue: string[];
  red: string[];
  league?: string | null;
  patch?: string | null;
  elo_diff?: number | null;
  team_elo_diff?: number | null;
  player_elo_diff?: number | null;
  blue_team?: string | null;
  red_team?: string | null;
  blue_players?: string[] | null;
  red_players?: string[] | null;
  strength_source?: string | null;
  strength_as_of?: string | null;
  strength_model_id?: string | null;
  blue_roles?: string[] | null;
  red_roles?: string[] | null;
};

export type DraftScoreView = {
  p_blue: number;
  score_blue: number;
  score_red: number;
  edge: number;
  source: string;
};

export type DraftScoreResult = {
  draft_score_blue: number;
  draft_score_red: number;
  draft_edge: number;
  confidence: number;
  p_blue_draft: number;
  raw: DraftScoreView;
  contextualized: DraftScoreView | null;
  strength: {
    team_elo_diff: number | null;
    player_elo_diff: number | null;
    source: string;
  };
  wr_bump_pp: number;
  posterior_width: number;
  calibration: {
    league: string | null | undefined;
    patch?: string | null;
    source: string;
    intercept?: number;
    slope?: number;
    neutral_blue_baseline?: number;
    temperature?: number;
    legacy_temp?: number;
    p_blue_legacy_1_4?: number;
    p_blue_with_strength: number | null;
    dp_per_logit?: number;
    normalized_patch?: string | null;
    patch_status?: "exact" | "pooled_missing" | "pooled_unsupported";
    league_status?: "exact" | "pooled_global";
  };
  components: {
    win_logit_blue: number;
    win_logit_red: number;
    pair_logit: number;
    win_edge: number;
    pace_shift_blue: number;
    pace_shift_red: number;
    pace_total_shift: number;
    known_frac_blue: number;
    known_frac_red: number;
    main_logit?: number;
    synergy_logit?: number;
    opposition_logit?: number;
    low_rank_logit?: number;
    composition_edge?: number;
    model_edge?: number;
    side_advantage_logit?: number;
  };
  blue: string[];
  red: string[];
  note: string;
  uncertainty?: {
    edge_se_logit: number;
    p_blue_95: [number, number];
    method: string;
  };
  contextualized_uncertainty?: {
    p_blue_95: [number, number];
    method: string;
  } | null;
  explanation?: {
    edge: number;
    composition_edge: number;
    side_advantage: number;
    champions: Array<{
      champion: string;
      side: "blue" | "red";
      role: string;
      direct_effect: number;
      team_synergy: number;
      enemy_interaction: number;
      edge_contribution: number;
      uncertainty_logit: number;
      evidence: {
        role_champion_maps: {
          observed_terms: number;
          possible_terms: number;
          minimum_maps: number;
          median_maps: number;
          label: string;
        };
        ally_synergy_pairs: {
          observed_terms: number;
          possible_terms: number;
          minimum_maps: number;
          median_maps: number;
          label: string;
        };
        enemy_interaction_pairs: {
          observed_terms: number;
          possible_terms: number;
          minimum_maps: number;
          median_maps: number;
          label: string;
        };
        uncertainty_logit: number;
      };
    }>;
    reconciles: boolean;
    attribution: string;
  };
  model?: {
    runtime_version: number;
    artifact_sha256: string;
    model_code_sha256: string;
    training_population_sha256: string;
    trained_through: string;
    normalized_patch: string | null;
    patch_status: "exact" | "pooled_missing" | "pooled_unsupported";
    league: string;
    league_status: "exact" | "pooled_global";
  } | null;
};

export type DraftChampion = {
  name: string;
  roles: DraftRole[];
  games: number;
  role_games?: Partial<Record<DraftRole, number>>;
};

export type DraftAction = {
  side: DraftSide;
  champion: string;
  role?: DraftRole | null;
};

export type DraftCandidateRole = DraftRole | "open" | "any";

export type DraftRecommendation = {
  champion: string;
  role: DraftRole | null;
  immediate_value: number;
  projected_value: number;
  delta_points: number;
  sample_games: number;
  evidence: "Settled" | "Observed" | "Thin" | "Unseen role";
  lookahead_plies: number;
  principal_variation: DraftAction[];
};

export type DraftTimelineRow = DraftAction & {
  pick_number: number;
  projected_value: number;
  delta_points: number;
};

export type DraftSandboxResult = {
  value_kind: "experimental_composition_policy_value";
  probability_status: "withheld_failed_chronological_gate";
  candidate_role_policy: "supported_pro_roles_minimum_maps";
  perspective: DraftSide;
  recommendation_side: DraftSide;
  next_side: DraftSide;
  candidate_role: DraftCandidateRole;
  current: {
    projected_value: number;
    audit: {
      probability_pipeline_gate: "failed";
      release_runtime_binding: "not_checked" | "matched";
      calibration: "not_applicable_to_policy_value";
    };
  };
  timeline: DraftTimelineRow[];
  recommendations: DraftRecommendation[];
  open_roles: DraftRole[];
  search: {
    exhaustive: false;
    root_legal_actions: number;
    root_evaluated_actions: number;
    root_beam_width: number;
    future_beam_width: number;
    max_followup_plies: number;
    minimum_role_maps: number;
  };
  note: string;
  model_context: DraftScoreResult["model"] | null;
};

export function validateDraftSandboxState(
  actions: DraftAction[],
  nextSide: DraftSide,
  requireVerifiedRoles = true,
): void {
  if (actions.length > DRAFT_PICK_ORDER.length) {
    throw new Error("a draft can contain at most ten picks");
  }
  const selected = new Set<string>();
  const occupied = {
    blue: new Set<DraftRole>(),
    red: new Set<DraftRole>(),
  };
  actions.forEach((action, index) => {
    if (action.side !== DRAFT_PICK_ORDER[index]) {
      throw new Error(
        `pick ${index + 1} must belong to ${DRAFT_PICK_ORDER[index]} side`,
      );
    }
    if (!action.role && requireVerifiedRoles) {
      throw new Error(`pick ${index + 1} needs one verified role`);
    }
    if (action.role && !DRAFT_ROLES.includes(action.role)) {
      throw new Error(`pick ${index + 1} needs one verified role`);
    }
    if (action.role && occupied[action.side].has(action.role)) {
      throw new Error(
        `${action.side} side cannot assign two champions to ${action.role}`,
      );
    }
    if (action.role) occupied[action.side].add(action.role);
    const champion = normKey(normalizeDraftChampion(action.champion));
    if (selected.has(champion)) {
      throw new Error(`${action.champion} is already selected`);
    }
    selected.add(champion);
  });
  const expected = DRAFT_PICK_ORDER[actions.length];
  if (expected && nextSide !== expected) {
    throw new Error(
      `next side must be ${expected} after ${actions.length} picks`,
    );
  }
}

let cached: DraftRuntime | null = null;
let catalogCached: DraftChampion[] | null = null;

function resolveExplicitStrength(input: DraftScoreInput): DraftScoreInput {
  const teamDiff = input.team_elo_diff ?? input.elo_diff ?? null;
  const playerDiff = input.player_elo_diff ?? null;
  for (const [label, value] of [
    ["team_elo_diff", teamDiff],
    ["player_elo_diff", playerDiff],
  ] as const) {
    if (value != null && !Number.isFinite(Number(value))) {
      throw new DraftIntegrityInputError(`${label} must be finite`);
    }
  }
  if (teamDiff == null && playerDiff == null) {
    return {
      ...input,
      team_elo_diff: null,
      player_elo_diff: null,
    };
  }
  if (
    !input.strength_source?.trim() ||
    !input.strength_as_of?.trim() ||
    !input.strength_model_id?.trim()
  ) {
    throw new DraftIntegrityInputError(
      "contextual strength requires an explicit source, as-of time, and model ID",
    );
  }
  const asOf = Date.parse(input.strength_as_of);
  if (!Number.isFinite(asOf)) {
    throw new DraftIntegrityInputError(
      "strength_as_of must be a valid timestamp",
    );
  }
  return {
    ...input,
    team_elo_diff: teamDiff,
    player_elo_diff: playerDiff,
  };
}

function loadRuntime(): DraftRuntime {
  if (cached) return cached;
  const file = path.join(process.cwd(), "data", "draft", "runtime.json");
  cached = JSON.parse(readFileSync(file, "utf8")) as DraftRuntime;
  return cached;
}

function normKey(name: string): string {
  return (name || "")
    .normalize("NFKC")
    .trim()
    .toLowerCase()
    .replace(/[’`]/g, "'")
    .replace(/\s+/g, " ");
}

export function normalizeDraftChampion(name: string): string {
  const key = normKey(name);
  if (key in CHAMP_ALIASES) return CHAMP_ALIASES[key];
  return (name || "").trim();
}

function sigmoid(x: number): number {
  if (x >= 30) return 1;
  if (x <= -30) return 0;
  return 1 / (1 + Math.exp(-x));
}

function clip(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

function posteriorWeight(count: number, priorN = PRIOR_N): number {
  return count / (count + priorN);
}

function draftTemperature(league: string | null | undefined, rt: DraftRuntime): Scale {
  const cal = rt.calibration || {};
  const lg = (league || "").toUpperCase().trim();
  const byLg = cal.by_league || {};
  const byTier = cal.by_tier || {};
  const joint = cal.joint_logistic_global || {};
  const residual = cal.residual_ridge || {};
  const dp = Number(residual.coef_dp_per_logit ?? 0.23);

  if (lg && byLg[lg]?.coef_draft != null) {
    const row = byLg[lg];
    return {
      source: `league:${lg}`,
      temp: Number(row.coef_draft),
      coef_elo: Number(row.coef_elo ?? 0),
      intercept: Number(row.intercept ?? 0),
      dp_per_logit: dp,
    };
  }
  const tier = LEAGUE_TIER[lg];
  if (tier && byTier[tier]?.coef_draft != null) {
    const row = byTier[tier];
    return {
      source: `tier:${tier}`,
      temp: Number(row.coef_draft),
      coef_elo: Number(row.coef_elo ?? 0),
      intercept: Number(row.intercept ?? 0),
      dp_per_logit: dp,
    };
  }
  if (joint.coef_draft != null) {
    return {
      source: "global_joint",
      temp: Number(joint.coef_draft),
      coef_elo: Number(joint.coef_elo ?? 0),
      intercept: Number(joint.intercept ?? 0),
      dp_per_logit: dp,
    };
  }
  return {
    source: "legacy_1.4",
    temp: DEFAULT_TEMP,
    coef_elo: 0,
    intercept: 0,
    dp_per_logit: 0.23,
  };
}

function scoreSide(
  champsIn: string[],
  rt: DraftRuntime,
  sidePrior: number,
): {
  champs: string[];
  win_logit: number;
  pace_shift: number;
  known_frac: number;
  confidence: number;
  posterior_width: number;
} {
  const champs = champsIn.map(normalizeDraftChampion);
  let winLogit = sidePrior;
  let pace = 0;
  let known = 0;
  const confParts: number[] = [];
  for (const c of champs) {
    const cnt = rt.champ_game_counts[c] ?? 0;
    const w = posteriorWeight(cnt);
    confParts.push(w);
    if (c in rt.win_delta || c in rt.kill_beta) known += 1;
    winLogit += w * (rt.win_delta[c] ?? 0);
    pace += w * (rt.kill_beta[c] ?? 0);
  }
  const n = Math.max(champs.length, 1);
  const knownFrac = known / n;
  const conf = clip(0.5 * knownFrac + 0.5 * (confParts.reduce((a, b) => a + b, 0) / n), 0.05, 0.98);
  const width =
    champs.length === 0
      ? 1
      : confParts.length
        ? confParts.map((w) => 1 - w).reduce((a, b) => a + b, 0) / champs.length
        : 1;
  return {
    champs,
    win_logit: winLogit,
    pace_shift: pace,
    known_frac: knownFrac,
    confidence: conf,
    posterior_width: width,
  };
}

function normRole(r: string): string {
  const s = String(r || "").toLowerCase();
  if (s.startsWith("jng") || s === "jungle" || s.startsWith("jungler")) return "jng";
  if (s.startsWith("bot") || s === "adc" || s.startsWith("bottom")) return "bot";
  if (s.startsWith("sup") || s === "utility" || s.startsWith("support")) return "sup";
  if (s.startsWith("mid")) return "mid";
  if (s.startsWith("top")) return "top";
  return s.slice(0, 3);
}

function isDraftRole(value: string): value is DraftRole {
  return DRAFT_ROLES.includes(value as DraftRole);
}

function roleMap(
  champs: string[],
  roles: string[] | null | undefined,
): Map<DraftRole, string> {
  const output = new Map<DraftRole, string>();
  // Map-level pick arrays are draft-order, not role-order. Only calculate
  // matchup terms when the caller supplies an explicit role for each champion.
  if (!roles?.length || roles.length !== champs.length) return output;
  champs.forEach((champion, index) => {
    const role = normRole(roles[index] || "");
    if (isDraftRole(role) && !output.has(role)) output.set(role, champion);
  });
  return output;
}

function rolePairScore(
  blue: string[],
  red: string[],
  blueRoles: string[] | null | undefined,
  redRoles: string[] | null | undefined,
  rt: DraftRuntime,
): number {
  const blueByRole = roleMap(blue, blueRoles);
  const redByRole = roleMap(red, redRoles);
  let pairLogit = 0;
  for (const role of DRAFT_ROLES) {
    const blueChampion = blueByRole.get(role);
    const redChampion = redByRole.get(role);
    if (!blueChampion || !redChampion) continue;
    const row = rt.role_pairs[`${role}|${blueChampion}|${redChampion}`];
    if (row) pairLogit += 0.35 * Number(row.logit);
  }
  return pairLogit;
}

export function draftScore(input: DraftScoreInput): DraftScoreResult {
  const enriched = resolveExplicitStrength(input);
  if (enriched.blue.length === 5 && enriched.red.length === 5) {
    return compositionDraftScore(enriched);
  }
  if (enriched.blue.length === 5 || enriched.red.length === 5) {
    throw new DraftIntegrityInputError(
      "complete scoring requires five champions on both sides",
    );
  }
  const rt = loadRuntime();
  const scale = draftTemperature(enriched.league, rt);
  const b = scoreSide(enriched.blue, rt, BLUE_SIDE_BONUS);
  const r = scoreSide(enriched.red, rt, 0);

  const pairLogit = rolePairScore(
    b.champs,
    r.champs,
    enriched.blue_roles,
    enriched.red_roles,
    rt,
  );

  const winEdge = b.win_logit - r.win_logit + pairLogit;
  const temp = scale.temp;
  const pBlueRaw = sigmoid(winEdge * temp);
  let conf = Math.min(b.confidence, r.confidence);
  const width = 0.5 * (b.posterior_width + r.posterior_width);
  conf = clip(conf * (1 - 0.5 * width), 0.05, 0.98);
  const pShrunk = 0.5 + (pBlueRaw - 0.5) * conf;
  const scoreBlue = 100 * pShrunk;
  const scoreRed = 100 - scoreBlue;
  const wrBumpPp = 100 * scale.dp_per_logit * winEdge * conf;

  let pWithStrength: number | null = null;
  if (enriched.team_elo_diff != null && scale.source !== "legacy_1.4") {
    pWithStrength = sigmoid(
      scale.intercept + scale.coef_elo * (Number(enriched.team_elo_diff) / 400) + temp * winEdge,
    );
  }

  const legacyP = sigmoid(winEdge * DEFAULT_TEMP);
  const legacyShrunk = 0.5 + (legacyP - 0.5) * conf;

  return {
    draft_score_blue: round(scoreBlue, 2),
    draft_score_red: round(scoreRed, 2),
    draft_edge: round(scoreBlue - scoreRed, 2),
    confidence: round(conf, 3),
    p_blue_draft: round(pShrunk, 4),
    raw: {
      p_blue: round(pShrunk, 4),
      score_blue: round(scoreBlue, 2),
      score_red: round(100 - scoreBlue, 2),
      edge: round(scoreBlue - (100 - scoreBlue), 2),
      source: "legacy partial-draft model (partial boards only)",
    },
    contextualized:
      pWithStrength == null
        ? null
        : {
            p_blue: round(pWithStrength, 4),
            score_blue: round(100 * pWithStrength, 2),
            score_red: round(100 * (1 - pWithStrength), 2),
            edge: round(100 * (2 * pWithStrength - 1), 2),
            source: enriched.strength_source ?? "pre-match team strength",
          },
    strength: {
      team_elo_diff: enriched.team_elo_diff ?? enriched.elo_diff ?? null,
      player_elo_diff: enriched.player_elo_diff ?? null,
      source: enriched.strength_source ?? (enriched.team_elo_diff != null ? "explicit pre-match strength" : "unavailable"),
    },
    wr_bump_pp: round(wrBumpPp, 2),
    posterior_width: round(width, 3),
    calibration: {
      league: enriched.league,
      patch: enriched.patch,
      source: scale.source,
      temperature: round(temp, 4),
      legacy_temp: DEFAULT_TEMP,
      p_blue_legacy_1_4: round(legacyShrunk, 4),
      p_blue_with_strength: pWithStrength != null ? round(pWithStrength, 4) : null,
      dp_per_logit: round(scale.dp_per_logit, 4),
    },
    contextualized_uncertainty: null,
    components: {
      win_logit_blue: round(b.win_logit, 4),
      win_logit_red: round(r.win_logit, 4),
      pair_logit: round(pairLogit, 4),
      win_edge: round(winEdge, 4),
      pace_shift_blue: round(b.pace_shift, 3),
      pace_shift_red: round(r.pace_shift, 3),
      pace_total_shift: round(b.pace_shift + r.pace_shift, 3),
      known_frac_blue: round(b.known_frac, 3),
      known_frac_red: round(r.known_frac, 3),
    },
    blue: b.champs,
    red: r.champs,
    note:
      "Legacy partial-draft comparison. Complete five-versus-five boards never use this fallback.",
    model: null,
  };
}

export function draftCatalog(): DraftChampion[] {
  if (catalogCached) return catalogCached;
  const compositionCatalog = compositionDraftCatalog();
  if (!compositionCatalog?.length) {
    throw new CompositionRuntimeUnavailableError();
  }
  catalogCached = compositionCatalog.map((row) => ({
    name: row.name,
    roles: row.roles.filter(
      (role): role is DraftRole =>
        isDraftRole(role) &&
        Number(row.role_games[role] ?? 0) >=
          DRAFT_POLICY_MIN_ROLE_GAMES,
    ),
    games: row.games,
    role_games: Object.fromEntries(
      Object.entries(row.role_games)
        .filter(([role]) => isDraftRole(role))
        .map(([role, games]) => [role, Number(games) || 0]),
    ) as Partial<Record<DraftRole, number>>,
  }));
  return catalogCached;
}

function sideModelValue(score: DraftScoreResult, side: DraftSide): number {
  return side === "blue" ? score.p_blue_draft : 1 - score.p_blue_draft;
}

function scoreActions(
  actions: DraftAction[],
  league: string | null | undefined,
  patch: string | null | undefined,
  eloDiff: number | null | undefined,
  requireRoleAware: boolean,
): DraftScoreResult {
  const blue = actions.filter((action) => action.side === "blue");
  const red = actions.filter((action) => action.side === "red");
  const input: DraftScoreInput = {
    blue: blue.map((action) => action.champion),
    red: red.map((action) => action.champion),
    league,
    patch,
    elo_diff: eloDiff,
  };
  if (requireRoleAware) {
    input.blue_roles = blue.map((action) => action.role || "");
    input.red_roles = red.map((action) => action.role || "");
    return compositionPartialDraftScore(input);
  }
  return draftScore(input);
}

function draftStateKey(
  actions: DraftAction[],
  league: string | null | undefined,
  patch: string | null | undefined,
  eloDiff: number | null | undefined,
): string {
  const sideKey = (side: DraftSide) =>
    actions
      .filter((action) => action.side === side)
      .map(
        (action) =>
          `${action.role ?? "?"}:${normKey(action.champion)}`,
      )
      .sort()
      .join(",");
  return `${league ?? ""};${patch ?? ""};${eloDiff ?? ""};b=${sideKey("blue")};r=${sideKey("red")}`;
}

function evidenceLabel(games: number): DraftRecommendation["evidence"] {
  if (games <= 0) return "Unseen role";
  if (games >= 200) return "Settled";
  if (games >= 75) return "Observed";
  return "Thin";
}

type PolicyValue = {
  value: number;
  line: DraftAction[];
};

const SANDBOX_LOOKAHEAD_PLIES = 2;
const SANDBOX_BEAM_WIDTH = 8;
const SANDBOX_ROOT_BEAM_MAX = 32;

function requireDraftSandboxPatch(
  value: string | null | undefined,
): string {
  if (typeof value !== "string" || !value.trim()) {
    throw new DraftIntegrityInputError(
      "Sandbox patch is required; select an observed patch context",
    );
  }
  const patch = normalizeCompositionPatch(value);
  if (!patch) {
    throw new DraftIntegrityInputError(
      "Sandbox patch must use a numeric major.minor form",
    );
  }
  const metadata = compositionRuntimeMetadata();
  if (!metadata) throw new CompositionRuntimeUnavailableError();
  if (!metadata.analysis_patches.includes(patch)) {
    throw new DraftIntegrityInputError(
      `Sandbox patch ${patch} is outside the observed model artifact`,
    );
  }
  return patch;
}

function legalFutureActions(
  actions: DraftAction[],
  excluded: Set<string>,
  catalog: DraftChampion[],
): DraftAction[] {
  const side = DRAFT_PICK_ORDER[actions.length];
  if (!side) return [];
  const selected = new Set(actions.map((action) => normKey(action.champion)));
  const occupied = new Set(
    actions
      .filter((action) => action.side === side)
      .map((action) => action.role)
      .filter((role): role is DraftRole => Boolean(role)),
  );
  const openRoles = DRAFT_ROLES.filter((role) => !occupied.has(role));
  const legal: DraftAction[] = [];
  for (const candidate of catalog) {
    if (selected.has(normKey(candidate.name)) || excluded.has(normKey(candidate.name))) {
      continue;
    }
    for (const role of candidate.roles) {
      if (openRoles.includes(role)) {
        legal.push({ side, champion: candidate.name, role });
      }
    }
  }
  return legal;
}

function policyValue(
  actions: DraftAction[],
  rootSide: DraftSide,
  depth: number,
  excluded: Set<string>,
  catalog: DraftChampion[],
  league: string | null | undefined,
  patch: string | null | undefined,
  eloDiff: number | null | undefined,
  roleAwareBoard: boolean,
  memo: Map<string, PolicyValue>,
  scoreMemo: Map<string, number>,
): PolicyValue {
  const terminal = (): PolicyValue => ({
    value: cachedSideValue(
      actions,
      rootSide,
      league,
      patch,
      eloDiff,
      roleAwareBoard,
      scoreMemo,
    ),
    line: [],
  });
  if (depth <= 0 || actions.length >= DRAFT_PICK_ORDER.length) return terminal();

  const key = [
    rootSide,
    depth,
    actions
      .map((action) => `${action.side}:${action.role}:${normKey(action.champion)}`)
      .join("|"),
  ].join(";");
  const cachedValue = memo.get(key);
  if (cachedValue) return cachedValue;

  const sideToAct = DRAFT_PICK_ORDER[actions.length];
  const legal = legalFutureActions(actions, excluded, catalog);
  if (!sideToAct || !legal.length) return terminal();

  const ranked = legal
    .map((action) => {
      const next = [...actions, action];
      return {
        action,
        immediate: cachedSideValue(
          next,
          rootSide,
          league,
          patch,
          eloDiff,
          roleAwareBoard,
          scoreMemo,
        ),
      };
    })
    .sort((a, b) => {
      const difference =
        sideToAct === rootSide
          ? b.immediate - a.immediate
          : a.immediate - b.immediate;
      return (
        difference ||
        a.action.champion.localeCompare(b.action.champion) ||
        String(a.action.role).localeCompare(String(b.action.role))
      );
    })
    .slice(0, SANDBOX_BEAM_WIDTH);

  if (depth === 1 && ranked.length) {
    const resolved = {
      value: ranked[0].immediate,
      line: [ranked[0].action],
    };
    memo.set(key, resolved);
    return resolved;
  }

  let best: PolicyValue | null = null;
  for (const candidate of ranked) {
    const child = policyValue(
      [...actions, candidate.action],
      rootSide,
      depth - 1,
      excluded,
      catalog,
      league,
      patch,
      eloDiff,
      roleAwareBoard,
      memo,
      scoreMemo,
    );
    const result = {
      value: child.value,
      line: [candidate.action, ...child.line],
    };
    if (
      !best ||
      (sideToAct === rootSide
        ? result.value > best.value
        : result.value < best.value)
    ) {
      best = result;
    }
  }
  const resolved = best ?? terminal();
  memo.set(key, resolved);
  return resolved;
}

function cachedSideValue(
  actions: DraftAction[],
  side: DraftSide,
  league: string | null | undefined,
  patch: string | null | undefined,
  eloDiff: number | null | undefined,
  requireRoleAware: boolean,
  cache: Map<string, number>,
): number {
  const key = `${side};${draftStateKey(
    actions,
    league,
    patch,
    eloDiff,
  )};scope=${requireRoleAware ? "roles" : "roles-free"}`;
  const cachedValue = cache.get(key);
  if (cachedValue != null) return cachedValue;
  const value = sideModelValue(
    scoreActions(actions, league, patch, eloDiff, requireRoleAware),
    side,
  );
  cache.set(key, value);
  return value;
}

export function analyzeDraftSandbox(input: {
  actions: DraftAction[];
  perspective: DraftSide;
  next_side: DraftSide;
  candidate_role?: DraftCandidateRole;
  excluded?: string[];
  league?: string | null;
  patch?: string | null;
  elo_diff?: number | null;
  limit?: number;
}): DraftSandboxResult {
  const actions = input.actions.map((action) => ({
    ...action,
    champion: normalizeDraftChampion(action.champion),
  }));
  const roleAwareBoard = actions.every((action) => isDraftRole(action.role || ""));
  validateDraftSandboxState(actions, input.next_side, roleAwareBoard);
  const patch = requireDraftSandboxPatch(input.patch);
  const perspective = input.perspective;
  const currentScore = scoreActions(
    actions,
    input.league,
    patch,
    input.elo_diff,
    roleAwareBoard,
  );
  const currentValue = sideModelValue(currentScore, perspective);
  const currentRecommendationValue = sideModelValue(currentScore, input.next_side);
  const timeline: DraftTimelineRow[] = [];
  const prefix: DraftAction[] = [];
  let previousValue = 0.5;

  actions.forEach((action, index) => {
    prefix.push(action);
    const score = scoreActions(
      prefix,
      input.league,
      patch,
      input.elo_diff,
      roleAwareBoard,
    );
    const value = sideModelValue(score, perspective);
    timeline.push({
      ...action,
      pick_number: index + 1,
      projected_value: round(value, 4),
      delta_points: round(100 * (value - previousValue), 2),
    });
    previousValue = value;
  });

  const occupied = new Set(
    actions
      .filter((action) => action.side === input.next_side)
      .map((action) => action.role)
      .filter((role): role is DraftRole => Boolean(role)),
  );
  const openRoles = DRAFT_ROLES.filter((role) => !occupied.has(role));
  const selected = new Set(actions.map((action) => normKey(action.champion)));
  const excluded = new Set((input.excluded || []).map((champion) => normKey(normalizeDraftChampion(champion))));
  const candidateRole = input.candidate_role ?? "open";
  const catalog = draftCatalog();
  const recommendations: DraftRecommendation[] = [];
  const policyMemo = new Map<string, PolicyValue>();
  const scoreMemo = new Map<string, number>();
  const requestedLimit = clip(input.limit ?? 12, 1, 200);
  const legalRootActions: Array<{
    action: DraftAction;
    immediate: number;
    sampleGames: number;
  }> = [];

  for (const candidate of catalog) {
    if (selected.has(normKey(candidate.name)) || excluded.has(normKey(candidate.name))) continue;
    let roles: DraftRole[];
    if (candidateRole === "any") {
      roles = candidate.roles.filter((role) => openRoles.includes(role));
      if (!roles.length) continue;
    } else if (candidateRole === "open") {
      roles = candidate.roles.filter((role) => openRoles.includes(role));
      if (!roles.length) continue;
    } else {
      if (!openRoles.includes(candidateRole)) continue;
      if (!candidate.roles.includes(candidateRole)) continue;
      roles = [candidateRole];
    }

    for (const role of roles) {
      const candidateAction: DraftAction = {
        side: input.next_side,
        champion: candidate.name,
        role,
      };
      const branch = [...actions, candidateAction];
      legalRootActions.push({
        action: candidateAction,
        immediate: cachedSideValue(
          branch,
          input.next_side,
          input.league,
          patch,
          input.elo_diff,
          roleAwareBoard,
          scoreMemo,
        ),
        sampleGames: Number(candidate.role_games?.[role] ?? 0),
      });
    }
  }

  const rootBeamWidth = Math.min(
    legalRootActions.length,
    SANDBOX_ROOT_BEAM_MAX,
  );
  const rootBeam = legalRootActions
    .sort(
      (a, b) =>
        b.immediate - a.immediate ||
        b.sampleGames - a.sampleGames ||
        a.action.champion.localeCompare(b.action.champion) ||
        String(a.action.role).localeCompare(String(b.action.role)),
    )
    .slice(0, rootBeamWidth);
  const bestByChampion = new Map<string, DraftRecommendation>();

  for (const seed of rootBeam) {
    const branch = [...actions, seed.action];
    const immediate = cachedSideValue(
      branch,
      input.next_side,
      input.league,
      patch,
      input.elo_diff,
      roleAwareBoard,
      scoreMemo,
    );
    const remainingPicks = DRAFT_PICK_ORDER.length - branch.length;
    const lookaheadPlies = Math.min(
      SANDBOX_LOOKAHEAD_PLIES,
      remainingPicks,
    );
    const policy = policyValue(
      branch,
      input.next_side,
      lookaheadPlies,
      excluded,
      catalog,
      input.league,
      patch,
      input.elo_diff,
      roleAwareBoard,
      policyMemo,
      scoreMemo,
    );
    const row: DraftRecommendation = {
      champion: seed.action.champion,
      role: seed.action.role ?? null,
      immediate_value: round(immediate, 4),
      projected_value: round(policy.value, 4),
      delta_points: round(
        100 * (policy.value - currentRecommendationValue),
        2,
      ),
      sample_games: seed.sampleGames,
      evidence: evidenceLabel(seed.sampleGames),
      lookahead_plies: lookaheadPlies,
      principal_variation: policy.line,
    };
    const currentBest = bestByChampion.get(row.champion);
    if (
      !currentBest ||
      row.projected_value > currentBest.projected_value ||
      (row.projected_value === currentBest.projected_value &&
        row.sample_games > currentBest.sample_games)
    ) {
      bestByChampion.set(row.champion, row);
    }
  }
  recommendations.push(...bestByChampion.values());

  recommendations.sort(
    (a, b) =>
      b.projected_value - a.projected_value ||
      b.sample_games - a.sample_games ||
      a.champion.localeCompare(b.champion),
  );

  return {
    value_kind: "experimental_composition_policy_value",
    probability_status: "withheld_failed_chronological_gate",
    candidate_role_policy: "supported_pro_roles_minimum_maps",
    perspective,
    recommendation_side: input.next_side,
    next_side: input.next_side,
    candidate_role: candidateRole,
    current: {
      projected_value: round(currentValue, 4),
      audit: {
        probability_pipeline_gate: "failed",
        release_runtime_binding: "not_checked",
        calibration: "not_applicable_to_policy_value",
      },
    },
    timeline,
    recommendations: recommendations.slice(0, requestedLimit),
    open_roles: openRoles,
    search: {
      exhaustive: false,
      root_legal_actions: legalRootActions.length,
      root_evaluated_actions: rootBeam.length,
      root_beam_width: rootBeamWidth,
      future_beam_width: SANDBOX_BEAM_WIDTH,
      max_followup_plies: SANDBOX_LOOKAHEAD_PLIES,
      minimum_role_maps: DRAFT_POLICY_MIN_ROLE_GAMES,
    },
    note:
      `Experimental beam-minimax composition policy value. ${
        roleAwareBoard
          ? "Role constraints and role-aware pair effects are applied to all placed actions."
          : "Current board includes unresolved roles; role-agnostic partial-draft fallback is used for scoring and recommendations."
      } Candidate and look-ahead policy picks require at least ${DRAFT_POLICY_MIN_ROLE_GAMES} pro maps in the champion-role pair, including when the manual board allows an unsupported role what-if. The root beam retains the strongest immediate legal actions, then re-evaluates each through up to two subsequent legal pro-role picks; the acting side maximizes its value and the opponent minimizes it. This is a bounded, non-exhaustive policy search. The composition probability pipeline failed its chronological promotion gate, so no Sandbox value is a win probability or a solved best response.`,
    model_context: currentScore.model,
  };
}

function round(x: number, digits: number): number {
  const p = 10 ** digits;
  return Math.round(x * p) / p;
}
