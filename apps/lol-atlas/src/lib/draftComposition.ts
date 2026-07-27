/** Browser/runtime scorer for the full-composition draft artifact. */
import { readFileSync } from "fs";
import path from "path";
import { gunzipSync } from "zlib";
import type { DraftScoreInput, DraftScoreResult } from "./draftScore";

type FeatureSpec = {
  coef: number;
  n?: number;
  se: number;
};

type LowRank = {
  status: "disabled";
  rank: 0;
  champions: [];
  left: [];
  right: [];
  reason: string;
};

type StrengthCalibrationSource = {
  artifact: string;
  artifact_sha256: string | null;
  artifact_version: number | null;
};

type StrengthCalibrationUnavailable = {
  schema_version: "1.0.0";
  status: "unavailable";
  reason: string;
  source: StrengthCalibrationSource;
};

type StrengthCalibrationAvailable = {
  schema_version: "1.0.0";
  status: "available";
  calibration_id: string;
  fit_cutoff: string;
  holdout_start: string;
  source: {
    artifact: string;
    artifact_sha256: string;
    artifact_version: number;
  };
  team: { model_id: string; intercept: number; coef: number };
  player: { model_id: string; intercept: number; coef: number };
  blend: {
    model_id: string;
    intercept: number;
    coef_team: number;
    coef_player: number;
  };
};

type CompositionRuntime = {
  version: number;
  model_code_sha256: string;
  training_population_sha256: string;
  numerical_environment: {
    python: string;
    packages: {
      numpy: string;
      pandas: string;
      scipy: string;
      "scikit-learn": string;
    };
  };
  estimand: string;
  intercept: number;
  intercept_se: number;
  feature_specs: Record<string, FeatureSpec>;
  role_champion_counts: Record<string, number>;
  components: string[];
  prior_n: number;
  low_rank: LowRank;
  calibration: {
    intercept: number;
    slope: number;
    covariance: [[number, number], [number, number]];
  };
  calibration_source: string;
  n_games_fit: number;
  n_games_total: number;
  date_min: string;
  date_max: string;
  min_support?: number | null;
  recency_half_life_days?: number | null;
  validation: {
    time_holdout?: Record<string, number>;
    future_patch_holdout?: Record<string, number>;
    future_patch?: string[];
    league_holdout?: Record<string, number>;
    league?: string;
  };
  uncertainty: {
    schema_version: "1.0.0";
    method: string;
    active_terms: string[];
    low_rank_status: "disabled";
  };
  limitations: string[];
  artifact_sha256: string;
  strength_calibration:
    | StrengthCalibrationUnavailable
    | StrengthCalibrationAvailable;
};

export type CompositionRuntimeMetadata = {
  runtime_status: "available";
  version: number;
  estimand: string;
  n_games_fit: number;
  n_games_total: number;
  date_min: string;
  date_max: string;
  calibration_source: string;
  artifact_sha256: string;
  model_code_sha256: string;
  training_population_sha256: string;
  numerical_environment: CompositionRuntime["numerical_environment"];
  validation: CompositionRuntime["validation"];
  limitations: string[];
  strength_calibration_status: "available" | "unavailable";
  latest_observed_patch: string | null;
  observed_holdout_patches: string[];
  analysis_patches: string[];
  supported_patches: string[];
  supported_leagues: string[];
};

type Pick = { role: string; champion: string };
export type CompositionCatalogRow = {
  name: string;
  roles: string[];
  games: number;
  role_games: Record<string, number>;
};
type ExplanationRow = {
  champion: string;
  side: "blue" | "red";
  role: string;
  direct_effect: number;
  team_synergy: number;
  enemy_interaction: number;
  edge_contribution: number;
  uncertainty_logit: number;
  evidence: {
    role_champion_maps: SupportSummary;
    ally_synergy_pairs: SupportSummary;
    enemy_interaction_pairs: SupportSummary;
    uncertainty_logit: number;
  };
  allySupport: number[];
  enemySupport: number[];
};

type SupportSummary = {
  observed_terms: number;
  possible_terms: number;
  minimum_maps: number;
  median_maps: number;
  label: string;
};

export class CompositionRuntimeUnavailableError extends Error {
  readonly code = "composition_runtime_unavailable";

  constructor() {
    super("The versioned composition runtime is unavailable.");
    this.name = "CompositionRuntimeUnavailableError";
  }
}

export class DraftIntegrityInputError extends Error {
  readonly code = "invalid_draft_integrity_input";

  constructor(message: string) {
    super(message);
    this.name = "DraftIntegrityInputError";
  }
}

export class StrengthCalibrationUnavailableError extends Error {
  readonly code = "strength_calibration_unavailable";

  constructor(reason: string) {
    super(`Contextual strength calibration is unavailable: ${reason}`);
    this.name = "StrengthCalibrationUnavailableError";
  }
}

const ROLES = ["top", "jng", "mid", "bot", "sup"];
const ALIASES: Record<string, string> = {
  kaisa: "Kai'Sa",
  "kai'sa": "Kai'Sa",
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
  khazix: "Kha'Zix",
  mf: "Miss Fortune",
  "miss fortune": "Miss Fortune",
  tf: "Twisted Fate",
  lee: "Lee Sin",
  xin: "Xin Zhao",
};

type RuntimeState =
  | { status: "available"; runtime: CompositionRuntime }
  | { status: "unavailable"; reason: "missing" | "invalid" };

let cachedState: RuntimeState | undefined;
let cachedMetadata: CompositionRuntimeMetadata | null | undefined;

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isValidTimestamp(value: unknown): value is string {
  return (
    typeof value === "string" &&
    value.trim().length > 0 &&
    Number.isFinite(Date.parse(value))
  );
}

function isSha256(value: unknown): value is string {
  return typeof value === "string" && /^[a-f0-9]{64}$/i.test(value);
}

/**
 * Validate the complete, versioned runtime contract before any score is
 * calculated. Issues are suitable for tests and server logs, not public APIs.
 */
export function compositionRuntimeSchemaIssues(value: unknown): string[] {
  if (!isRecord(value)) return ["artifact must be an object"];
  const issues: string[] = [];
  const requireNumber = (key: string) => {
    if (!isFiniteNumber(value[key])) issues.push(`${key} must be finite`);
  };
  const requireString = (key: string) => {
    if (typeof value[key] !== "string" || !String(value[key]).trim()) {
      issues.push(`${key} must be a non-empty string`);
    }
  };

  if (value.version !== 2) issues.push("version must equal 2");
  for (const key of ["model_code_sha256", "training_population_sha256"]) {
    requireString(key);
    if (typeof value[key] === "string" && !isSha256(value[key])) {
      issues.push(`${key} must be a 64-character hexadecimal digest`);
    }
  }
  requireString("estimand");
  requireNumber("intercept");
  requireNumber("intercept_se");
  if (isFiniteNumber(value.intercept_se) && value.intercept_se < 0) {
    issues.push("intercept_se must be non-negative");
  }
  requireNumber("prior_n");
  requireNumber("n_games_fit");
  requireNumber("n_games_total");
  requireString("date_min");
  requireString("date_max");
  requireString("calibration_source");
  requireString("artifact_sha256");
  if (
    typeof value.artifact_sha256 === "string" &&
    !isSha256(value.artifact_sha256)
  ) {
    issues.push("artifact_sha256 must be a 64-character hexadecimal digest");
  }
  if (!isRecord(value.numerical_environment)) {
    issues.push("numerical_environment is required");
  } else {
    if (
      typeof value.numerical_environment.python !== "string" ||
      !value.numerical_environment.python.trim()
    ) {
      issues.push("numerical_environment.python must be a non-empty string");
    }
    const packages = value.numerical_environment.packages;
    if (!isRecord(packages)) {
      issues.push("numerical_environment.packages is required");
    } else {
      for (const key of ["numpy", "pandas", "scipy", "scikit-learn"]) {
        if (typeof packages[key] !== "string" || !packages[key].trim()) {
          issues.push(
            `numerical_environment.packages.${key} must be a non-empty string`,
          );
        }
      }
    }
  }
  const components = Array.isArray(value.components)
    ? value.components
    : [];
  if (
    !["main", "synergy", "opposition"].every((component) =>
      components.includes(component),
    )
  ) {
    issues.push("components must include main, synergy, and opposition");
  }
  if (
    !Array.isArray(value.limitations) ||
    !value.limitations.length ||
    value.limitations.some(
      (item) => typeof item !== "string" || !item.trim(),
    )
  ) {
    issues.push("limitations must contain public model limitations");
  }
  if (!isRecord(value.validation) || !isRecord(value.validation.time_holdout)) {
    issues.push("validation.time_holdout is required");
  }

  if (!isRecord(value.calibration)) {
    issues.push("calibration is required");
  } else {
    for (const key of ["intercept", "slope"]) {
      if (!isFiniteNumber(value.calibration[key])) {
        issues.push(`calibration.${key} must be finite`);
      }
    }
    const covariance = value.calibration.covariance;
    if (
      !Array.isArray(covariance) ||
      covariance.length !== 2 ||
      covariance.some(
        (row) =>
          !Array.isArray(row) ||
          row.length !== 2 ||
          row.some((entry) => !isFiniteNumber(entry)),
      )
    ) {
      issues.push("calibration.covariance must be a finite 2x2 matrix");
    } else if (covariance[0][0] < 0 || covariance[1][1] < 0) {
      issues.push("calibration.covariance diagonal must be non-negative");
    }
  }

  if (!isRecord(value.low_rank)) {
    issues.push("low_rank is required");
  } else {
    if (value.low_rank.status !== "disabled" || value.low_rank.rank !== 0) {
      issues.push("low_rank must be explicitly disabled with rank 0");
    }
    for (const key of ["champions", "left", "right"]) {
      if (!Array.isArray(value.low_rank[key]) || value.low_rank[key].length) {
        issues.push(`disabled low_rank.${key} must be empty`);
      }
    }
    if (
      typeof value.low_rank.reason !== "string" ||
      !value.low_rank.reason.trim()
    ) {
      issues.push("low_rank.reason must explain why it is disabled");
    }
  }

  if (!isRecord(value.uncertainty)) {
    issues.push("uncertainty metadata is required");
  } else {
    if (value.uncertainty.schema_version !== "1.0.0") {
      issues.push("uncertainty.schema_version must equal 1.0.0");
    }
    if (
      typeof value.uncertainty.method !== "string" ||
      !value.uncertainty.method.trim()
    ) {
      issues.push("uncertainty.method must be a non-empty string");
    }
    if (
      !Array.isArray(value.uncertainty.active_terms) ||
      !value.uncertainty.active_terms.length ||
      value.uncertainty.active_terms.some(
        (term) => typeof term !== "string" || !term.trim(),
      )
    ) {
      issues.push("uncertainty.active_terms must be non-empty");
    }
    if (value.uncertainty.low_rank_status !== "disabled") {
      issues.push("uncertainty.low_rank_status must equal disabled");
    }
  }

  if (!isRecord(value.strength_calibration)) {
    issues.push("strength_calibration is required");
  } else {
    const strength = value.strength_calibration;
    const source = isRecord(strength.source) ? strength.source : null;
    if (strength.schema_version !== "1.0.0") {
      issues.push("strength_calibration.schema_version must equal 1.0.0");
    }
    if (!source) {
      issues.push("strength_calibration.source is required");
    } else {
      if (
        typeof source.artifact !== "string" ||
        !source.artifact.trim()
      ) {
        issues.push(
          "strength_calibration.source.artifact must be a non-empty string",
        );
      }
      if (
        source.artifact_sha256 != null &&
        !isSha256(source.artifact_sha256)
      ) {
        issues.push(
          "strength_calibration.source.artifact_sha256 must be a SHA-256 digest",
        );
      }
      if (
        source.artifact_version != null &&
        (!Number.isInteger(source.artifact_version) ||
          Number(source.artifact_version) < 1)
      ) {
        issues.push(
          "strength_calibration.source.artifact_version must be a positive integer",
        );
      }
    }
    if (strength.status === "unavailable") {
      if (typeof strength.reason !== "string" || !strength.reason.trim()) {
        issues.push(
          "unavailable strength_calibration.reason must be a non-empty string",
        );
      }
      for (const key of ["team", "player", "blend"]) {
        if (key in strength) {
          issues.push(
            `unavailable strength_calibration cannot contain ${key} coefficients`,
          );
        }
      }
    } else if (strength.status === "available") {
      for (const key of [
        "calibration_id",
        "fit_cutoff",
        "holdout_start",
      ]) {
        if (typeof strength[key] !== "string" || !strength[key].trim()) {
          issues.push(`strength_calibration.${key} must be a non-empty string`);
        }
      }
      if (
        !isValidTimestamp(strength.fit_cutoff) ||
        !isValidTimestamp(strength.holdout_start) ||
        Date.parse(String(strength.fit_cutoff)) >=
          Date.parse(String(strength.holdout_start))
      ) {
        issues.push(
          "strength_calibration fit_cutoff must precede holdout_start",
        );
      }
      if (
        !source ||
        !isSha256(source.artifact_sha256) ||
        !Number.isInteger(source.artifact_version) ||
        Number(source.artifact_version) < 2
      ) {
        issues.push(
          "available strength_calibration requires a versioned immutable source",
        );
      }
      for (const [blockName, coefficientNames] of [
        ["team", ["intercept", "coef"]],
        ["player", ["intercept", "coef"]],
        ["blend", ["intercept", "coef_team", "coef_player"]],
      ] as const) {
        const block = isRecord(strength[blockName])
          ? strength[blockName]
          : null;
        if (!block) {
          issues.push(`strength_calibration.${blockName} is required`);
          continue;
        }
        if (
          typeof block.model_id !== "string" ||
          !block.model_id.trim()
        ) {
          issues.push(
            `strength_calibration.${blockName}.model_id must be a non-empty string`,
          );
        }
        for (const coefficientName of coefficientNames) {
          if (!isFiniteNumber(block[coefficientName])) {
            issues.push(
              `strength_calibration.${blockName}.${coefficientName} must be finite`,
            );
          }
        }
      }
    } else {
      issues.push(
        "strength_calibration.status must be available or unavailable",
      );
    }
  }

  if (
    !isRecord(value.feature_specs) ||
    !Object.keys(value.feature_specs).length
  ) {
    issues.push("feature_specs must be a non-empty object");
  } else {
    for (const [key, spec] of Object.entries(value.feature_specs)) {
      if (!isRecord(spec) || !isFiniteNumber(spec.coef)) {
        issues.push(`feature_specs.${key}.coef must be finite`);
        break;
      }
      if (spec.n != null && !isFiniteNumber(spec.n)) {
        issues.push(`feature_specs.${key}.n must be finite when present`);
        break;
      }
      if (!isFiniteNumber(spec.se) || spec.se < 0) {
        issues.push(
          `feature_specs.${key}.se must be finite and non-negative`,
        );
        break;
      }
    }
  }

  if (
    !isRecord(value.role_champion_counts) ||
    !Object.keys(value.role_champion_counts).length
  ) {
    issues.push("role_champion_counts must be a non-empty object");
  } else if (
    Object.values(value.role_champion_counts).some(
      (count) => !isFiniteNumber(count) || count < 0,
    )
  ) {
    issues.push("role_champion_counts values must be finite and non-negative");
  }
  return issues;
}

function runtimeState(): RuntimeState {
  if (cachedState) return cachedState;
  let parsed: unknown;
  try {
    const file = path.join(
      process.cwd(),
      "data",
      "draft",
      "composition_runtime.json",
    );
    parsed = JSON.parse(readFileSync(file, "utf8"));
  } catch {
    try {
      const packed = readFileSync(
        path.join(
          process.cwd(),
          "data",
          "draft",
          "composition_runtime.json.gz.b64",
        ),
        "utf8",
      );
      parsed = JSON.parse(
        gunzipSync(Buffer.from(packed, "base64")).toString("utf8"),
      );
    } catch {
      cachedState = { status: "unavailable", reason: "missing" };
      return cachedState;
    }
  }
  const issues = compositionRuntimeSchemaIssues(parsed);
  if (issues.length) {
    console.error("[draft-composition] invalid runtime", { issues });
    cachedState = { status: "unavailable", reason: "invalid" };
    return cachedState;
  }
  cachedState = {
    status: "available",
    runtime: parsed as unknown as CompositionRuntime,
  };
  return cachedState;
}

function loadRuntime(): CompositionRuntime | null {
  const state = runtimeState();
  return state.status === "available" ? state.runtime : null;
}

export function compositionRuntimeMetadata(): CompositionRuntimeMetadata | null {
  if (cachedMetadata !== undefined) return cachedMetadata;
  const runtime = loadRuntime();
  if (!runtime) {
    cachedMetadata = null;
    return cachedMetadata;
  }
  const supportedPatches = [
    ...new Set(
      Object.keys(runtime.feature_specs ?? {})
        .filter((key) => key.startsWith("patch|"))
        .map((key) => key.split("|")[1])
        .filter(Boolean),
    ),
  ].sort((a, b) => Number(a) - Number(b));
  const supportedLeagues = [
    ...new Set(
      Object.keys(runtime.feature_specs)
        .filter((key) => key.startsWith("league|"))
        .map((key) => key.split("|")[1])
        .filter(Boolean),
    ),
  ].sort();
  const observedHoldoutPatches = Array.isArray(runtime.validation.future_patch)
    ? runtime.validation.future_patch
        .filter(
          (value): value is string =>
            typeof value === "string" &&
            normalizeCompositionPatch(value) === value,
        )
        .sort((a, b) => Number(a) - Number(b))
    : [];
  const analysisPatches = [
    ...new Set([...supportedPatches, ...observedHoldoutPatches]),
  ].sort((a, b) => Number(a) - Number(b));
  cachedMetadata = {
    runtime_status: "available",
    version: runtime.version,
    estimand: runtime.estimand,
    n_games_fit: runtime.n_games_fit,
    n_games_total: runtime.n_games_total,
    date_min: runtime.date_min,
    date_max: runtime.date_max,
    calibration_source: runtime.calibration_source,
    artifact_sha256: runtime.artifact_sha256,
    model_code_sha256: runtime.model_code_sha256,
    training_population_sha256: runtime.training_population_sha256,
    numerical_environment: runtime.numerical_environment,
    validation: runtime.validation,
    limitations: runtime.limitations,
    strength_calibration_status: runtime.strength_calibration.status,
    latest_observed_patch:
      analysisPatches[analysisPatches.length - 1] ?? null,
    observed_holdout_patches: observedHoldoutPatches,
    analysis_patches: analysisPatches,
    supported_patches: supportedPatches,
    supported_leagues: supportedLeagues,
  };
  return cachedMetadata;
}

export function compositionDraftCatalog(): CompositionCatalogRow[] | null {
  const runtime = loadRuntime();
  if (!runtime) return null;
  const byChampion = new Map<string, CompositionCatalogRow>();
  for (const [key, rawGames] of Object.entries(runtime.role_champion_counts ?? {})) {
    const separator = key.indexOf("|");
    if (separator < 0) continue;
    const role = key.slice(0, separator);
    const name = key.slice(separator + 1);
    if (!ROLES.includes(role) || !name) continue;
    const games = Math.max(0, Number(rawGames) || 0);
    const row = byChampion.get(name) ?? {
      name,
      roles: [],
      games: 0,
      role_games: {},
    };
    row.role_games[role] = games;
    row.games += games;
    if (games > 0 && !row.roles.includes(role)) row.roles.push(role);
    byChampion.set(name, row);
  }
  return [...byChampion.values()]
    .map((row) => ({
      ...row,
      roles: ROLES.filter((role) => row.roles.includes(role)),
    }))
    .sort((a, b) => a.name.localeCompare(b.name));
}

function normKey(value: string): string {
  return (value || "")
    .normalize("NFKC")
    .trim()
    .toLowerCase()
    .replace(/[’`]/g, "'")
    .replace(/\s+/g, " ");
}

function normalizeChamp(value: string): string {
  const key = normKey(value);
  return ALIASES[key] ?? value.trim();
}

function normRole(value: string): string {
  const key = normKey(value);
  if (key.startsWith("jng") || key.startsWith("jung")) return "jng";
  if (key.startsWith("bot") || key.startsWith("adc") || key.startsWith("bottom")) return "bot";
  if (key.startsWith("sup") || key.startsWith("util")) return "sup";
  if (key.startsWith("mid")) return "mid";
  if (key.startsWith("top")) return "top";
  return key.slice(0, 3);
}

export function normalizeCompositionPatch(
  value?: string | null,
): string | null {
  if (!value) return null;
  const match = String(value)
    .trim()
    .match(/^(\d+)\.(\d+)(?:\.\d+)*$/);
  if (!match) return null;
  const major = Number(match[1]);
  const minor = Number(match[2]);
  if (
    !Number.isSafeInteger(major) ||
    !Number.isSafeInteger(minor) ||
    major < 0 ||
    minor < 0 ||
    minor > 99
  ) {
    return null;
  }
  return `${major}.${String(minor).padStart(2, "0")}`;
}

function pair(a: string, b: string): [string, string] {
  return a <= b ? [a, b] : [b, a];
}

function oppositionKey(a: string, b: string): [string, number] {
  const [left, right] = pair(a, b);
  if (a === b) return [`opposition|${left}|${right}`, 0];
  return [`opposition|${left}|${right}`, a === left && b === right ? 1 : -1];
}

function sigmoid(value: number): number {
  if (value >= 30) return 1;
  if (value <= -30) return 0;
  return 1 / (1 + Math.exp(-value));
}

function round(value: number, digits = 6): number {
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function evidenceLabel(games: number): string {
  if (games >= 100) return "well supported";
  if (games >= 30) return "supported";
  if (games >= 10) return "thin";
  return "very thin";
}

export function compositionDraftScore(input: DraftScoreInput): DraftScoreResult {
  return scoreComposition(input, false);
}

export function compositionPartialDraftScore(
  input: DraftScoreInput,
): DraftScoreResult {
  return scoreComposition(input, true);
}

function scoreComposition(
  input: DraftScoreInput,
  partial: boolean,
): DraftScoreResult {
  const runtime = loadRuntime();
  if (!runtime) throw new CompositionRuntimeUnavailableError();
  if (!partial && (input.blue.length !== 5 || input.red.length !== 5)) {
    throw new DraftIntegrityInputError("need 5 picks per side");
  }
  if (input.blue.length > 5 || input.red.length > 5) {
    throw new DraftIntegrityInputError(
      "each side can contain at most 5 picks",
    );
  }

  const resolveRoles = (
    champions: string[],
    supplied: string[] | null | undefined,
    side: "blue" | "red",
  ): string[] => {
    if (!supplied || supplied.length !== champions.length) {
      throw new DraftIntegrityInputError(
        `${side} side needs one authoritative role per champion`,
      );
    }
    const roles = supplied.map(normRole);
    if (roles.some((role) => !ROLES.includes(role))) {
      throw new DraftIntegrityInputError(
        `${side} side contains an invalid role`,
      );
    }
    if (new Set(roles).size !== roles.length) {
      throw new DraftIntegrityInputError(
        `${side} side cannot assign the same role twice`,
      );
    }
    return roles;
  };
  const blueRoles = resolveRoles(input.blue, input.blue_roles, "blue");
  const redRoles = resolveRoles(input.red, input.red_roles, "red");
  const blue: Pick[] = input.blue.map((champion, i) => ({
    role: blueRoles[i],
    champion: normalizeChamp(champion),
  }));
  const red: Pick[] = input.red.map((champion, i) => ({
    role: redRoles[i],
    champion: normalizeChamp(champion),
  }));
  const catalogNames = new Map(
    Object.keys(runtime.role_champion_counts).map((key) => {
      const champion = key.slice(key.indexOf("|") + 1);
      return [normKey(champion), champion];
    }),
  );
  for (const pick of [...blue, ...red]) {
    const canonical = catalogNames.get(normKey(pick.champion));
    if (!canonical) {
      throw new DraftIntegrityInputError(
        `unknown champion: ${pick.champion}`,
      );
    }
    pick.champion = canonical;
  }
  const uniqueChampions = new Set(
    [...blue, ...red].map((pick) => normKey(pick.champion)),
  );
  if (uniqueChampions.size !== blue.length + red.length) {
    throw new DraftIntegrityInputError(
      "a champion cannot be selected more than once",
    );
  }
  const league = String(input.league ?? "UNKNOWN").trim().toUpperCase() || "UNKNOWN";
  const patch = normalizeCompositionPatch(input.patch);
  if (input.patch && !patch) {
    throw new DraftIntegrityInputError("patch must use a numeric major.minor form");
  }
  const metadata = compositionRuntimeMetadata();
  if (!metadata) throw new CompositionRuntimeUnavailableError();
  const patchStatus = patch
    ? metadata.supported_patches.includes(patch)
      ? "exact"
      : "pooled_unsupported"
    : "pooled_missing";
  const leagueStatus = metadata.supported_leagues.includes(league)
    ? "exact"
    : "pooled_global";
  const specs = runtime.feature_specs ?? {};
  const components = new Set(runtime.components ?? []);
  const rows: ExplanationRow[] = [];

  const rowFor = (side: "blue" | "red", pick: Pick): ExplanationRow => {
    const sign = side === "blue" ? 1 : -1;
    let direct = 0;
    let variance = 0;
    for (const key of [
      `main|${pick.role}|${pick.champion}`,
      `league|${league}|${pick.role}|${pick.champion}`,
      ...(patch ? [`patch|${patch}|${pick.role}|${pick.champion}`] : []),
    ]) {
      const spec = specs[key];
      if (!spec) continue;
      direct += sign * spec.coef;
      variance += spec.se ** 2;
    }
    return {
      champion: pick.champion,
      side,
      role: pick.role,
      direct_effect: direct,
      team_synergy: 0,
      enemy_interaction: 0,
      edge_contribution: direct,
      uncertainty_logit: Math.sqrt(variance),
      evidence: {
        role_champion_maps: {
          observed_terms: 1,
          possible_terms: 1,
          minimum_maps: Number(
            runtime.role_champion_counts[
              `${pick.role}|${pick.champion}`
            ] ?? 0,
          ),
          median_maps: Number(
            runtime.role_champion_counts[
              `${pick.role}|${pick.champion}`
            ] ?? 0,
          ),
          label: "very thin",
        },
        ally_synergy_pairs: {
          observed_terms: 0,
          possible_terms: 0,
          minimum_maps: 0,
          median_maps: 0,
          label: "no selected pairs",
        },
        enemy_interaction_pairs: {
          observed_terms: 0,
          possible_terms: 0,
          minimum_maps: 0,
          median_maps: 0,
          label: "no selected pairs",
        },
        uncertainty_logit: 0,
      },
      allySupport: [],
      enemySupport: [],
    };
  };

  blue.forEach((pick) => rows.push(rowFor("blue", pick)));
  red.forEach((pick) => rows.push(rowFor("red", pick)));
  const findRow = (side: "blue" | "red", pick: Pick) => rows.find((row) => row.side === side && row.role === pick.role && row.champion === pick.champion)!;

  const mainLogit = rows.reduce((sum, row) => sum + row.direct_effect, 0);
  let synergyLogit = 0;
  let oppositionLogit = 0;
  const lowRankLogit = 0;
  let edgeVariance = rows.reduce((sum, row) => sum + row.uncertainty_logit ** 2, 0);

  if (components.has("synergy")) {
    for (const [side, picks] of [["blue", blue], ["red", red]] as const) {
      const sign = side === "blue" ? 1 : -1;
      for (let i = 0; i < picks.length; i += 1) {
        for (let j = i + 1; j < picks.length; j += 1) {
          const [left, right] = pair(picks[i].champion, picks[j].champion);
          const spec = specs[`synergy|${left}|${right}`];
          const support = Math.max(0, Number(spec?.n ?? 0));
          findRow(side, picks[i]).allySupport.push(support);
          findRow(side, picks[j]).allySupport.push(support);
          if (!spec) continue;
          const value = sign * spec.coef;
          synergyLogit += value;
          for (const pick of [picks[i], picks[j]]) {
            const row = findRow(side, pick);
            row.team_synergy += value / 2;
            row.edge_contribution += value / 2;
            row.uncertainty_logit = Math.sqrt(
              row.uncertainty_logit ** 2 + (spec.se / 2) ** 2,
            );
          }
          edgeVariance += spec.se ** 2;
        }
      }
    }
  }

  for (const bluePick of blue) {
    for (const redPick of red) {
      let value = 0;
      if (components.has("opposition")) {
        const [key, orientation] = oppositionKey(bluePick.champion, redPick.champion);
        const spec = specs[key];
        const support = Math.max(0, Number(spec?.n ?? 0));
        findRow("blue", bluePick).enemySupport.push(support);
        findRow("red", redPick).enemySupport.push(support);
        if (spec && orientation !== 0) {
          value = orientation * spec.coef;
          oppositionLogit += value;
          edgeVariance += spec.se ** 2;
          const blueRow = findRow("blue", bluePick);
          const redRow = findRow("red", redPick);
          blueRow.enemy_interaction += value / 2;
          redRow.enemy_interaction += value / 2;
          blueRow.edge_contribution += value / 2;
          redRow.edge_contribution += value / 2;
          blueRow.uncertainty_logit = Math.sqrt(
            blueRow.uncertainty_logit ** 2 + (spec.se / 2) ** 2,
          );
          redRow.uncertainty_logit = Math.sqrt(
            redRow.uncertainty_logit ** 2 + (spec.se / 2) ** 2,
          );
        }
      }
    }
  }

  const compositionEdge = mainLogit + synergyLogit + oppositionLogit + lowRankLogit;
  const sideAdvantage = partial ? 0 : runtime.intercept;
  const modelEdge = sideAdvantage + compositionEdge;
  const calibrationIntercept = partial ? 0 : runtime.calibration.intercept;
  const calibrationSlope = partial ? 1 : runtime.calibration.slope;
  // Complete-draft probability is calibrated from the same full linear
  // predictor used to fit the calibration curve. Composition-edge
  // antisymmetry remains explicit, while the blue-side baseline correctly
  // stays attached to blue under a composition-only swap.
  const calibrated =
    calibrationIntercept + calibrationSlope * modelEdge;
  const calibrationCovariance = partial
    ? ([[0, 0], [0, 0]] as [[number, number], [number, number]])
    : runtime.calibration.covariance;
  const modelEdgeVariance =
    edgeVariance + (partial ? 0 : runtime.intercept_se ** 2);
  const calibrationParameterVariance =
    calibrationCovariance[0][0] +
    2 * modelEdge * calibrationCovariance[0][1] +
    modelEdge ** 2 * calibrationCovariance[1][1];
  const calibratedVariance =
    calibrationSlope ** 2 * modelEdgeVariance +
    calibrationParameterVariance;
  const edgeSe = Math.sqrt(Math.max(calibratedVariance, 1e-12));
  const pBlue = sigmoid(calibrated);
  const pLo = sigmoid(calibrated - 1.96 * edgeSe);
  const pHi = sigmoid(calibrated + 1.96 * edgeSe);
  const neutralBlueBaseline = sigmoid(
    calibrationIntercept + calibrationSlope * sideAdvantage,
  );
  const strength = runtime.strength_calibration;
  const teamDiff = partial ? null : input.team_elo_diff ?? input.elo_diff ?? null;
  const playerDiff = partial ? null : input.player_elo_diff ?? null;
  if (
    (teamDiff != null || playerDiff != null) &&
    strength.status !== "available"
  ) {
    throw new StrengthCalibrationUnavailableError(strength.reason);
  }
  const availableStrength =
    strength.status === "available" ? strength : null;
  const teamP =
    teamDiff == null
      ? null
      : sigmoid(
          availableStrength!.team.intercept +
            (availableStrength!.team.coef * Number(teamDiff)) / 400,
        );
  const playerP =
    playerDiff == null
      ? null
      : sigmoid(
          availableStrength!.player.intercept +
            (availableStrength!.player.coef * Number(playerDiff)) / 400,
        );
  let strengthLogit: number | null = null;
  if (teamP != null && playerP != null) {
    strengthLogit =
      availableStrength!.blend.intercept +
      availableStrength!.blend.coef_team * teamP +
      availableStrength!.blend.coef_player * playerP;
  } else if (teamP != null) {
    strengthLogit = Math.log(teamP / Math.max(1 - teamP, 1e-12));
  } else if (playerP != null) {
    strengthLogit = Math.log(playerP / Math.max(1 - playerP, 1e-12));
  }
  const pWithStrength =
    strengthLogit == null ? null : sigmoid(strengthLogit + calibrationSlope * compositionEdge);

  const supportSummary = (values: number[]): SupportSummary => {
    if (!values.length) {
      return {
        observed_terms: 0,
        possible_terms: 0,
        minimum_maps: 0,
        median_maps: 0,
        label: "no selected pairs",
      };
    }
    const sorted = [...values].sort((a, b) => a - b);
    const midpoint = Math.floor(sorted.length / 2);
    const median =
      sorted.length % 2
        ? sorted[midpoint]
        : (sorted[midpoint - 1] + sorted[midpoint]) / 2;
    const observed = sorted.filter((count) => count > 0).length;
    const minimum = sorted[0];
    return {
      observed_terms: observed,
      possible_terms: sorted.length,
      minimum_maps: round(minimum, 2),
      median_maps: round(median, 2),
      label:
        observed < sorted.length
          ? "contains unseen pair terms"
          : evidenceLabel(minimum),
    };
  };

  for (const row of rows) {
    const games = row.evidence.role_champion_maps.minimum_maps;
    row.evidence.role_champion_maps.label = evidenceLabel(games);
    row.evidence.ally_synergy_pairs = supportSummary(row.allySupport);
    row.evidence.enemy_interaction_pairs = supportSummary(row.enemySupport);
    row.evidence.uncertainty_logit = round(row.uncertainty_logit, 4);
    row.direct_effect = round(row.direct_effect);
    row.team_synergy = round(row.team_synergy);
    row.enemy_interaction = round(row.enemy_interaction);
    row.edge_contribution = round(row.edge_contribution);
    delete (row as Partial<ExplanationRow>).allySupport;
    delete (row as Partial<ExplanationRow>).enemySupport;
  }
  const contributionSum = rows.reduce((sum, row) => sum + row.edge_contribution, 0);
  const reconciles = Math.abs(contributionSum - compositionEdge) < 1e-5;
  const filledSeats = blue.length + red.length;
  const confidenceBase = 1 / (1 + 2 * edgeSe);
  const confidence = partial
    ? Math.min(0.98, Math.max(0.05, (filledSeats / 10) * confidenceBase))
    : Math.min(0.98, Math.max(0.05, confidenceBase));
  const knownFraction = (picks: Pick[]) =>
    picks.length
      ? picks.filter(
          (pick) =>
            Number(runtime.role_champion_counts[`${pick.role}|${pick.champion}`] ?? 0) > 0,
        ).length / picks.length
      : 0;

  return {
    draft_score_blue: round(100 * pBlue, 2),
    draft_score_red: round(100 * (1 - pBlue), 2),
    draft_edge: round(100 * (2 * pBlue - 1), 2),
    confidence: round(confidence, 3),
    p_blue_draft: round(pBlue, 4),
    raw: {
      p_blue: round(pBlue, 4),
      score_blue: round(100 * pBlue, 2),
      score_red: round(100 * (1 - pBlue), 2),
      edge: round(100 * (2 * pBlue - 1), 2),
      source: partial
        ? "selected composition terms only; unfilled roles neutralized"
        : "composition only; no roster/player strength",
    },
    contextualized:
      pWithStrength == null
        ? null
        : {
            p_blue: round(pWithStrength, 4),
            score_blue: round(100 * pWithStrength, 2),
            score_red: round(100 * (1 - pWithStrength), 2),
            edge: round(100 * (2 * pWithStrength - 1), 2),
            source: input.strength_source ?? "pre-match team + player strength",
          },
    strength: {
      team_elo_diff: teamDiff == null ? null : round(Number(teamDiff), 2),
      player_elo_diff: playerDiff == null ? null : round(Number(playerDiff), 2),
      source: input.strength_source ?? (strengthLogit == null ? "unavailable" : "explicit pre-match strength"),
    },
    wr_bump_pp: round(100 * (pBlue - neutralBlueBaseline), 2),
    posterior_width: round(edgeSe, 4),
    uncertainty: {
      edge_se_logit: round(edgeSe, 4),
      p_blue_95: [round(pLo, 4), round(pHi, 4)],
      method: runtime.uncertainty.method,
    },
    contextualized_uncertainty: null,
    calibration: {
      league: input.league,
      patch: input.patch,
      source: partial
        ? "partial-draft composition utility (uncalibrated)"
        : runtime.calibration_source,
      intercept: round(calibrationIntercept, 4),
      slope: round(calibrationSlope, 4),
      neutral_blue_baseline: round(neutralBlueBaseline, 4),
      p_blue_with_strength: pWithStrength == null ? null : round(pWithStrength, 4),
      normalized_patch: patch,
      patch_status: patchStatus,
      league_status: leagueStatus,
    },
    components: {
      main_logit: round(mainLogit),
      synergy_logit: round(synergyLogit),
      opposition_logit: round(oppositionLogit),
      low_rank_logit: round(lowRankLogit),
      composition_edge: round(compositionEdge),
      model_edge: round(modelEdge),
      side_advantage_logit: round(sideAdvantage),
      win_logit_blue: round(mainLogit),
      win_logit_red: 0,
      pair_logit: round(synergyLogit + oppositionLogit + lowRankLogit),
      win_edge: round(compositionEdge),
      pace_shift_blue: 0,
      pace_shift_red: 0,
      pace_total_shift: 0,
      known_frac_blue: knownFraction(blue),
      known_frac_red: knownFraction(red),
    },
    explanation: {
      edge: round(contributionSum + sideAdvantage),
      composition_edge: round(contributionSum),
      side_advantage: round(sideAdvantage),
      champions: rows,
      reconciles,
      attribution: "symmetric pair allocation: each synergy/opposition pair is split equally across its two champions",
    },
    blue: blue.map((pick) => pick.champion),
    red: red.map((pick) => pick.champion),
    model: {
      runtime_version: runtime.version,
      artifact_sha256: runtime.artifact_sha256,
      model_code_sha256: runtime.model_code_sha256,
      training_population_sha256: runtime.training_population_sha256,
      trained_through: runtime.date_max,
      normalized_patch: patch,
      patch_status: patchStatus,
      league,
      league_status: leagueStatus,
    },
    note: partial
      ? "Partial full-composition counterfactual: role-aware direct effects plus synergy and opposition among selected champions. Unfilled seats are neutralized; the displayed share is not a calibrated live win probability."
      : "Full-composition draft model: role-aware direct effects, within-team synergy, and all 25 explicit enemy interactions. Low-rank residuals are disabled until their uncertainty is estimable; strength is reported separately when supplied.",
  };
}
