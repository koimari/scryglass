/** Browser/runtime scorer for the full-composition draft artifact. */
import { readFileSync } from "fs";
import path from "path";
import { gunzipSync } from "zlib";
import type { DraftScoreInput, DraftScoreResult } from "./draftScore";

type FeatureSpec = {
  coef: number;
  n?: number;
  se?: number;
};

type LowRank = {
  rank?: number;
  champions?: string[];
  left?: number[][];
  right?: number[][];
};

type CompositionRuntime = {
  version: number;
  estimand?: string;
  intercept: number;
  feature_specs: Record<string, FeatureSpec>;
  role_champion_counts: Record<string, number>;
  components: string[];
  prior_n?: number;
  low_rank?: LowRank;
  calibration?: { intercept?: number; slope?: number };
  calibration_source?: string;
  strength_calibration?: {
    team_intercept?: number;
    team_coef?: number;
    player_intercept?: number;
    player_coef?: number;
    blend_intercept?: number;
    blend_coef_team?: number;
    blend_coef_player?: number;
    draft_coef?: number;
  };
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
    games: number;
    shrinkage: number;
    label: string;
    uncertainty_logit: number;
  };
};

const ROLES = ["top", "jng", "mid", "bot", "sup"];
const DEFAULT_PRIOR_N = 25;
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

let cached: CompositionRuntime | null | undefined;

function loadRuntime(): CompositionRuntime | null {
  if (cached !== undefined) return cached;
  try {
    const file = path.join(process.cwd(), "data", "draft", "composition_runtime.json");
    cached = JSON.parse(readFileSync(file, "utf8")) as CompositionRuntime;
  } catch {
    try {
      const packed = readFileSync(
        path.join(process.cwd(), "data", "draft", "composition_runtime.json.gz.b64"),
        "utf8",
      );
      cached = JSON.parse(gunzipSync(Buffer.from(packed, "base64")).toString("utf8")) as CompositionRuntime;
    } catch {
      cached = null;
    }
  }
  return cached;
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

function normalizePatch(value?: string | null): string {
  if (!value) return "unknown";
  const match = String(value).trim().match(/^(\d+)(?:\.(\d+))?/);
  if (!match) return String(value).trim();
  const minor = String(match[2] ?? "0").padEnd(2, "0");
  return `${Number(match[1])}.${minor}`;
}

function pair(a: string, b: string): [string, string] {
  return a <= b ? [a, b] : [b, a];
}

function oppositionKey(a: string, b: string): [string, number] {
  const [left, right] = pair(a, b);
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

function lowRankValue(lowRank: LowRank | undefined, blue: string, red: string): number {
  if (!lowRank?.rank || lowRank.rank <= 0) return 0;
  const champions = lowRank.champions ?? [];
  const i = champions.indexOf(blue);
  const j = champions.indexOf(red);
  if (i < 0 || j < 0) return 0;
  const left = lowRank.left ?? [];
  const right = lowRank.right ?? [];
  const dot = (a: number[], b: number[]) => a.reduce((sum, value, k) => sum + value * (b[k] ?? 0), 0);
  return 0.5 * (dot(left[i] ?? [], right[j] ?? []) - dot(left[j] ?? [], right[i] ?? []));
}

export function compositionDraftScore(input: DraftScoreInput): DraftScoreResult | null {
  return scoreComposition(input, false);
}

export function compositionPartialDraftScore(input: DraftScoreInput): DraftScoreResult | null {
  return scoreComposition(input, true);
}

function scoreComposition(
  input: DraftScoreInput,
  partial: boolean,
): DraftScoreResult | null {
  const runtime = loadRuntime();
  if (!runtime) return null;
  if (!partial && (input.blue.length !== 5 || input.red.length !== 5)) {
    throw new Error("need 5 picks per side");
  }
  if (input.blue.length > 5 || input.red.length > 5) {
    throw new Error("each side can contain at most 5 picks");
  }

  const resolveRoles = (
    champions: string[],
    supplied: string[] | null | undefined,
  ): string[] | null => {
    const source = supplied ?? (!partial && champions.length === 5 ? ROLES : null);
    if (!source || source.length !== champions.length) return null;
    const roles = source.map(normRole);
    if (roles.some((role) => !ROLES.includes(role))) return null;
    if (new Set(roles).size !== roles.length) return null;
    return roles;
  };
  const blueRoles = resolveRoles(input.blue, input.blue_roles);
  const redRoles = resolveRoles(input.red, input.red_roles);
  if (!blueRoles || !redRoles) return null;
  const blue: Pick[] = input.blue.map((champion, i) => ({ role: blueRoles[i], champion: normalizeChamp(champion) }));
  const red: Pick[] = input.red.map((champion, i) => ({ role: redRoles[i], champion: normalizeChamp(champion) }));
  const league = String(input.league ?? "UNKNOWN").trim().toUpperCase() || "UNKNOWN";
  const patch = normalizePatch(input.patch);
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
      `patch|${patch}|${pick.role}|${pick.champion}`,
    ]) {
      const spec = specs[key];
      if (!spec) continue;
      direct += sign * Number(spec.coef ?? 0);
      variance += Number(spec.se ?? 0) ** 2;
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
        games: Number(runtime.role_champion_counts[`${pick.role}|${pick.champion}`] ?? 0),
        shrinkage: 0,
        label: "very thin",
        uncertainty_logit: 0,
      },
    };
  };

  blue.forEach((pick) => rows.push(rowFor("blue", pick)));
  red.forEach((pick) => rows.push(rowFor("red", pick)));
  const findRow = (side: "blue" | "red", pick: Pick) => rows.find((row) => row.side === side && row.role === pick.role && row.champion === pick.champion)!;

  const mainLogit = rows.reduce((sum, row) => sum + row.direct_effect, 0);
  let synergyLogit = 0;
  let oppositionLogit = 0;
  let lowRankLogit = 0;
  let edgeVariance = rows.reduce((sum, row) => sum + row.uncertainty_logit ** 2, 0);

  if (components.has("synergy")) {
    for (const [side, picks] of [["blue", blue], ["red", red]] as const) {
      const sign = side === "blue" ? 1 : -1;
      for (let i = 0; i < picks.length; i += 1) {
        for (let j = i + 1; j < picks.length; j += 1) {
          const [left, right] = pair(picks[i].champion, picks[j].champion);
          const spec = specs[`synergy|${left}|${right}`];
          if (!spec) continue;
          const value = sign * Number(spec.coef ?? 0);
          synergyLogit += value;
          for (const pick of [picks[i], picks[j]]) {
            const row = findRow(side, pick);
            row.team_synergy += value / 2;
            row.edge_contribution += value / 2;
            row.uncertainty_logit = Math.sqrt(row.uncertainty_logit ** 2 + (Number(spec.se ?? 0) / 2) ** 2);
          }
          edgeVariance += Number(spec.se ?? 0) ** 2;
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
        if (spec) {
          value = orientation * Number(spec.coef ?? 0);
          oppositionLogit += value;
          edgeVariance += Number(spec.se ?? 0) ** 2;
          const blueRow = findRow("blue", bluePick);
          const redRow = findRow("red", redPick);
          blueRow.enemy_interaction += value / 2;
          redRow.enemy_interaction += value / 2;
          blueRow.edge_contribution += value / 2;
          redRow.edge_contribution += value / 2;
          blueRow.uncertainty_logit = Math.sqrt(blueRow.uncertainty_logit ** 2 + (Number(spec.se ?? 0) / 2) ** 2);
          redRow.uncertainty_logit = Math.sqrt(redRow.uncertainty_logit ** 2 + (Number(spec.se ?? 0) / 2) ** 2);
        }
      }
      const lowRank = lowRankValue(runtime.low_rank, bluePick.champion, redPick.champion);
      lowRankLogit += lowRank;
      if (lowRank) {
        const blueRow = findRow("blue", bluePick);
        const redRow = findRow("red", redPick);
        blueRow.enemy_interaction += lowRank / 2;
        redRow.enemy_interaction += lowRank / 2;
        blueRow.edge_contribution += lowRank / 2;
        redRow.edge_contribution += lowRank / 2;
      }
    }
  }

  const compositionEdge = mainLogit + synergyLogit + oppositionLogit + lowRankLogit;
  const sideAdvantage = partial ? 0 : Number(runtime.intercept ?? 0);
  const modelEdge = sideAdvantage + compositionEdge;
  const edgeSe = Math.sqrt(Math.max(edgeVariance, 1e-12));
  const calibrationIntercept = partial ? 0 : Number(runtime.calibration?.intercept ?? 0);
  const calibrationSlope = partial ? 1 : Number(runtime.calibration?.slope ?? 1);
  const calibrated = calibrationIntercept + calibrationSlope * modelEdge;
  const pBlue = sigmoid(calibrated);
  const pLo = sigmoid(calibrationIntercept + calibrationSlope * (modelEdge - 1.96 * edgeSe));
  const pHi = sigmoid(calibrationIntercept + calibrationSlope * (modelEdge + 1.96 * edgeSe));
  const strength = runtime.strength_calibration ?? {};
  const teamDiff = partial ? null : input.team_elo_diff ?? input.elo_diff ?? null;
  const playerDiff = partial ? null : input.player_elo_diff ?? null;
  const teamP =
    teamDiff == null
      ? null
      : sigmoid(
          Number(strength.team_intercept ?? 0.14729) +
            Number(strength.team_coef ?? 2.37625) * Number(teamDiff) / 400,
        );
  const playerP =
    playerDiff == null
      ? null
      : sigmoid(
          Number(strength.player_intercept ?? 0.13166) +
            Number(strength.player_coef ?? 3.64257) * Number(playerDiff) / 400,
        );
  let strengthLogit: number | null = null;
  if (teamP != null && playerP != null) {
    strengthLogit =
      Number(strength.blend_intercept ?? -2.47489) +
      Number(strength.blend_coef_team ?? 2.84763) * teamP +
      Number(strength.blend_coef_player ?? 2.07485) * playerP;
  } else if (teamP != null) {
    strengthLogit = Math.log(teamP / Math.max(1 - teamP, 1e-12));
  } else if (playerP != null) {
    strengthLogit = Math.log(playerP / Math.max(1 - playerP, 1e-12));
  }
  const pWithStrength =
    strengthLogit == null ? null : sigmoid(strengthLogit + calibrationSlope * compositionEdge);
  const contextualizedRange =
    strengthLogit == null
      ? null
      : ([
          sigmoid(strengthLogit + calibrationSlope * (compositionEdge - 1.96 * edgeSe)),
          sigmoid(strengthLogit + calibrationSlope * (compositionEdge + 1.96 * edgeSe)),
        ] as [number, number]);

  for (const row of rows) {
    const games = row.evidence.games;
    const prior = Number(runtime.prior_n ?? DEFAULT_PRIOR_N);
    row.evidence.shrinkage = games / (games + prior);
    row.evidence.label = evidenceLabel(games);
    row.evidence.uncertainty_logit = round(row.uncertainty_logit, 4);
    row.direct_effect = round(row.direct_effect);
    row.team_synergy = round(row.team_synergy);
    row.enemy_interaction = round(row.enemy_interaction);
    row.edge_contribution = round(row.edge_contribution);
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
    wr_bump_pp: round(100 * (pBlue - 0.5), 2),
    posterior_width: round(edgeSe, 4),
    uncertainty: {
      edge_se_logit: round(edgeSe, 4),
      p_blue_95: [round(pLo, 4), round(pHi, 4)],
      method: "diagonal Laplace approximation; coefficient correlations are not represented",
    },
    contextualized_uncertainty: contextualizedRange
      ? {
          p_blue_95: [round(contextualizedRange[0], 4), round(contextualizedRange[1], 4)],
          method: "draft-only Laplace range conditional on supplied roster/player strength",
        }
      : null,
    calibration: {
      league: input.league,
      patch: input.patch,
      source: partial
        ? "partial-draft composition utility (uncalibrated)"
        : runtime.calibration_source ?? "time-heldout calibration slice",
      intercept: round(calibrationIntercept, 4),
      slope: round(calibrationSlope, 4),
      p_blue_with_strength: pWithStrength == null ? null : round(pWithStrength, 4),
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
    note: partial
      ? "Partial full-composition counterfactual: role-aware direct effects plus synergy and opposition among selected champions. Unfilled seats are neutralized; the displayed share is not a calibrated live win probability."
      : "Full-composition draft model: role-aware direct effects, within-team synergy, all 25 enemy interactions, and sparse low-rank residual. Strength is reported separately when supplied.",
  };
}
