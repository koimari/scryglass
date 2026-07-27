/** Fail-closed validator for the one public Void Grubs article artifact. */

export const GRUBS_ARTICLE_SCHEMA_VERSION = "scryglass.grubs.article.v1";
export const GRUBS_ARTICLE_PUBLICATION_ID =
  "void-grubs-contest-or-leave.patch-26.11";
export const GRUBS_MECHANICS_PATCH = "26.11+";

const GOLD10_INTERCEPT = 0.1611182873782888;
const GOLD10_COEF = 0.000666860223609559;
const GRUB_CASH_GOLD = 90;
const TOUCH_TRUE_DAMAGE = 256;
const FIRST_PLATE_HP = 900;
const FIRST_PLATE_GOLD = 120;
const BRIEF_TOUCH_PROGRESS_GOLD_EQUIVALENT = Number(
  ((TOUCH_TRUE_DAMAGE / FIRST_PLATE_HP) * FIRST_PLATE_GOLD).toFixed(2),
);
const OBJECTIVE_GOLD =
  GRUB_CASH_GOLD + BRIEF_TOUCH_PROGRESS_GOLD_EQUIVALENT;
const LEAVE_FARM_TWO_WAVE = 241.33;
const WIN_KILL = 600;
const LOSS_KILL = -600;

const CURVE_PROBABILITIES = [
  0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.75,
] as const;
const LEAVE_FARM_PACKAGES = [
  ["no_farm", 0],
  ["one_wave", 120.67],
  ["two_waves", LEAVE_FARM_TWO_WAVE],
] as const;
const GOLD_LEADS = [0, -500, -1000, -2000, 500, 1000, 1183, 1200] as const;

function sigmoid(z: number): number {
  const x = Math.max(-35, Math.min(35, z));
  return 1 / (1 + Math.exp(-x));
}

/** Side-neutral associational map-win conversion at own-team gold lead g. */
export function sideNeutralWinProb(gold: number): number {
  const linear = GOLD10_COEF * gold;
  return (
    0.5 *
    (sigmoid(GOLD10_INTERCEPT + linear) +
      sigmoid(-GOLD10_INTERCEPT + linear))
  );
}

/**
 * Fight-win probability where the fixed contest and leave sensitivities tie.
 * This is not an identified live-action policy.
 */
export function articlePStarAtGoldB(
  B: number,
  leaveFarmGold = LEAVE_FARM_TWO_WAVE,
  objectiveGold = OBJECTIVE_GOLD,
): number | null {
  const pLeave = sideNeutralWinProb(B + leaveFarmGold - objectiveGold);
  const pWin = sideNeutralWinProb(B + objectiveGold + WIN_KILL);
  const pLoss = sideNeutralWinProb(B - objectiveGold + LOSS_KILL);
  const denominator = pWin - pLoss;
  if (denominator <= 1e-12) return null;
  const root = (pLeave - pLoss) / denominator;
  return root >= 0 && root <= 1 ? root : null;
}

function coefTex(x: number): string {
  const [coefficient, exponent] = x.toExponential(3).split("e");
  return `${coefficient}\\times 10^{${Number(exponent)}}`;
}

export const PSTAR_TEX = {
  pStar:
    "\\mathrm{contest\\,bar}(\\mathrm{gold\\,lead})=\\frac{P_{\\mathrm{leave}}-P_{\\mathrm{loss}}}{P_{\\mathrm{win}}-P_{\\mathrm{loss}}}",
  winProb:
    "P(\\mathrm{gold})=\\tfrac{1}{2}\\bigl[\\sigma(a+b\\cdot\\mathrm{gold})+\\sigma(-a+b\\cdot\\mathrm{gold})\\bigr]",
  params: `\\mathrm{farm}=${LEAVE_FARM_TWO_WAVE}\\,\\mathrm{g},\\; \\mathrm{objective\\ equivalent}=${OBJECTIVE_GOLD}\\,\\mathrm{g},\\; \\text{fight swing }\\pm ${WIN_KILL}\\,\\mathrm{g},\\; a=${GOLD10_INTERCEPT.toFixed(4)},\\; b=${coefTex(GOLD10_COEF)}`,
} as const;

export const PSTAR_FX = {
  intercept: GOLD10_INTERCEPT,
  coef: GOLD10_COEF,
  objectiveGold: OBJECTIVE_GOLD,
  leaveFarmTwoWave: LEAVE_FARM_TWO_WAVE,
  winKill: WIN_KILL,
  lossKill: LOSS_KILL,
} as const;

export function contestBarPct(pStar: number): number {
  return 100 * pStar;
}

export type ArticleCurvePoint = {
  p_win_fight: number;
  ev_contest_pp: number;
  ev_leave_pp: number;
  edge_contest_minus_leave_pp: number;
  model_preference: "LEAVE" | "CONTEST";
};

export type LeaveFarmRow = {
  label: string;
  leave_farm_gold: number;
  p_star_at_parity: number;
  p_star_at_parity_pct: number;
  p_star_at_B_plus_1183: number;
  p_star_at_B_plus_1183_pct: number;
};

export type GoldBRow = {
  B_gold: number;
  leave_farm_gold: number;
  objective_gold: number;
  p_star: number;
  p_star_pct: number;
};

export type ValidatedArticleEv = {
  schema_version: string;
  publication_id: string;
  estimand: string;
  units: string;
  mechanics: {
    patch: string;
    grub_cash_gold: number;
    brief_touch_seconds: number;
    touch_true_damage: number;
    first_plate_hp: number;
    first_plate_gold: number;
    brief_touch_progress_gold_equivalent: number;
    objective_gold_equivalent: number;
    valuation: string;
    hunger_mite_included: boolean;
  };
  model: {
    conversion: string;
    intercept: number;
    gold_coefficient: number;
    fight_swing_gold: number;
    secure_if_win: number;
    secure_if_lose: number;
  };
  reference_knobs: {
    baseline_gold: number;
    objective_gold_equivalent: number;
    two_wave_leave_farm_gold: number;
    fight_swing_gold: number;
  };
  p_star: number;
  p_star_pct: number;
  edge_at_50_pp: number;
  interpretation: string;
  curve: ArticleCurvePoint[];
  by_leave_farm_F: LeaveFarmRow[];
  by_precontest_gold_B_two_wave_leave: GoldBRow[];
  limitations: string[];
};

export type ArticlePublicationPin = {
  schemaVersion: string;
  publicationId: string;
  mechanicsPatch: string;
  contestBarPct: number;
  atFiftyEdgePp: number;
};

export type ArticlePublicationValidation =
  | {
      ok: true;
      article: ValidatedArticleEv;
      atFifty: ArticleCurvePoint;
    }
  | { ok: false; reason: string };

type UnknownRecord = Record<string, unknown>;

const TOP_LEVEL_KEYS = [
  "schema_version",
  "publication_id",
  "estimand",
  "units",
  "mechanics",
  "model",
  "reference_knobs",
  "p_star",
  "p_star_pct",
  "edge_at_50_pp",
  "interpretation",
  "curve",
  "by_leave_farm_F",
  "by_precontest_gold_B_two_wave_leave",
  "limitations",
] as const;
const MECHANICS_KEYS = [
  "patch",
  "grub_cash_gold",
  "brief_touch_seconds",
  "touch_true_damage",
  "first_plate_hp",
  "first_plate_gold",
  "brief_touch_progress_gold_equivalent",
  "objective_gold_equivalent",
  "valuation",
  "hunger_mite_included",
] as const;
const MODEL_KEYS = [
  "conversion",
  "intercept",
  "gold_coefficient",
  "fight_swing_gold",
  "secure_if_win",
  "secure_if_lose",
] as const;
const REFERENCE_KEYS = [
  "baseline_gold",
  "objective_gold_equivalent",
  "two_wave_leave_farm_gold",
  "fight_swing_gold",
] as const;
const CURVE_KEYS = [
  "p_win_fight",
  "ev_contest_pp",
  "ev_leave_pp",
  "edge_contest_minus_leave_pp",
  "model_preference",
] as const;
const FARM_KEYS = [
  "label",
  "leave_farm_gold",
  "p_star_at_parity",
  "p_star_at_parity_pct",
  "p_star_at_B_plus_1183",
  "p_star_at_B_plus_1183_pct",
] as const;
const GOLD_KEYS = [
  "B_gold",
  "leave_farm_gold",
  "objective_gold",
  "p_star",
  "p_star_pct",
] as const;

function record(value: unknown): UnknownRecord | null {
  return value != null && typeof value === "object" && !Array.isArray(value)
    ? (value as UnknownRecord)
    : null;
}

function exactKeys(
  value: UnknownRecord,
  expected: readonly string[],
): boolean {
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  return (
    actual.length === wanted.length &&
    actual.every((key, index) => key === wanted[index])
  );
}

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function close(a: number, b: number, tolerance: number): boolean {
  return Math.abs(a - b) <= tolerance;
}

function rounded(value: number, digits: number): number {
  const scale = 10 ** digits;
  const result = Math.round((value + Number.EPSILON) * scale) / scale;
  return Object.is(result, -0) ? 0 : result;
}

function expectedCurvePoint(pWinFight: number): ArticleCurvePoint {
  const baseline = sideNeutralWinProb(0);
  const pWin = sideNeutralWinProb(OBJECTIVE_GOLD + WIN_KILL);
  const pLoss = sideNeutralWinProb(-OBJECTIVE_GOLD + LOSS_KILL);
  const pLeave = sideNeutralWinProb(LEAVE_FARM_TWO_WAVE - OBJECTIVE_GOLD);
  const contestPp =
    100 * (pWinFight * pWin + (1 - pWinFight) * pLoss - baseline);
  const leavePp = 100 * (pLeave - baseline);
  const edgePp = contestPp - leavePp;
  return {
    p_win_fight: pWinFight,
    ev_contest_pp: rounded(contestPp, 2),
    ev_leave_pp: rounded(leavePp, 2),
    edge_contest_minus_leave_pp: rounded(edgePp, 2),
    model_preference: edgePp >= 0 ? "CONTEST" : "LEAVE",
  };
}

function parseCurve(value: unknown): ArticleCurvePoint[] | null {
  if (!Array.isArray(value) || value.length !== CURVE_PROBABILITIES.length) {
    return null;
  }
  const rows: ArticleCurvePoint[] = [];
  for (let index = 0; index < value.length; index += 1) {
    const row = record(value[index]);
    const expected = expectedCurvePoint(CURVE_PROBABILITIES[index]);
    if (
      !row ||
      !exactKeys(row, CURVE_KEYS) ||
      !finite(row.p_win_fight) ||
      !finite(row.ev_contest_pp) ||
      !finite(row.ev_leave_pp) ||
      !finite(row.edge_contest_minus_leave_pp) ||
      (row.model_preference !== "LEAVE" &&
        row.model_preference !== "CONTEST") ||
      row.p_win_fight !== expected.p_win_fight ||
      row.ev_contest_pp !== expected.ev_contest_pp ||
      row.ev_leave_pp !== expected.ev_leave_pp ||
      row.edge_contest_minus_leave_pp !==
        expected.edge_contest_minus_leave_pp ||
      row.model_preference !== expected.model_preference
    ) {
      return null;
    }
    rows.push(row as ArticleCurvePoint);
  }
  return rows;
}

function parseLeaveFarm(value: unknown): LeaveFarmRow[] | null {
  if (!Array.isArray(value) || value.length !== LEAVE_FARM_PACKAGES.length) {
    return null;
  }
  const rows: LeaveFarmRow[] = [];
  for (let index = 0; index < value.length; index += 1) {
    const row = record(value[index]);
    const [label, farmGold] = LEAVE_FARM_PACKAGES[index];
    const parity = articlePStarAtGoldB(0, farmGold);
    const ahead = articlePStarAtGoldB(1183, farmGold);
    if (
      !row ||
      !exactKeys(row, FARM_KEYS) ||
      parity == null ||
      ahead == null ||
      row.label !== label ||
      row.leave_farm_gold !== farmGold ||
      row.p_star_at_parity !== rounded(parity, 6) ||
      row.p_star_at_parity_pct !== rounded(100 * parity, 2) ||
      row.p_star_at_B_plus_1183 !== rounded(ahead, 6) ||
      row.p_star_at_B_plus_1183_pct !== rounded(100 * ahead, 2)
    ) {
      return null;
    }
    rows.push(row as unknown as LeaveFarmRow);
  }
  return rows;
}

function parseGoldRows(value: unknown): GoldBRow[] | null {
  if (!Array.isArray(value) || value.length !== GOLD_LEADS.length) return null;
  const rows: GoldBRow[] = [];
  for (let index = 0; index < value.length; index += 1) {
    const row = record(value[index]);
    const goldLead = GOLD_LEADS[index];
    const pStar = articlePStarAtGoldB(goldLead);
    if (
      !row ||
      !exactKeys(row, GOLD_KEYS) ||
      pStar == null ||
      row.B_gold !== goldLead ||
      row.leave_farm_gold !== LEAVE_FARM_TWO_WAVE ||
      row.objective_gold !== OBJECTIVE_GOLD ||
      row.p_star !== rounded(pStar, 6) ||
      row.p_star_pct !== rounded(100 * pStar, 2)
    ) {
      return null;
    }
    rows.push(row as unknown as GoldBRow);
  }
  return rows;
}

/**
 * Validate schema, Patch 26.11+ mechanics, every displayed calculation, and
 * the editorial headline before the app renders any numerical conclusion.
 */
export function validateArticlePublication(
  articleValue: unknown,
  pin: ArticlePublicationPin,
): ArticlePublicationValidation {
  const article = record(articleValue);
  const mechanics = record(article?.mechanics);
  const model = record(article?.model);
  const reference = record(article?.reference_knobs);
  if (
    !article ||
    !exactKeys(article, TOP_LEVEL_KEYS) ||
    article.schema_version !== pin.schemaVersion ||
    article.schema_version !== GRUBS_ARTICLE_SCHEMA_VERSION ||
    article.publication_id !== pin.publicationId ||
    article.publication_id !== GRUBS_ARTICLE_PUBLICATION_ID ||
    article.estimand !== "article_opportunity_cost_sensitivity" ||
    typeof article.units !== "string" ||
    !article.units.includes("associational map-win") ||
    !mechanics ||
    !exactKeys(mechanics, MECHANICS_KEYS) ||
    mechanics.patch !== pin.mechanicsPatch ||
    mechanics.patch !== GRUBS_MECHANICS_PATCH ||
    mechanics.grub_cash_gold !== GRUB_CASH_GOLD ||
    mechanics.brief_touch_seconds !== 8 ||
    mechanics.touch_true_damage !== TOUCH_TRUE_DAMAGE ||
    mechanics.first_plate_hp !== FIRST_PLATE_HP ||
    mechanics.first_plate_gold !== FIRST_PLATE_GOLD ||
    mechanics.brief_touch_progress_gold_equivalent !==
      BRIEF_TOUCH_PROGRESS_GOLD_EQUIVALENT ||
    mechanics.objective_gold_equivalent !== OBJECTIVE_GOLD ||
    mechanics.valuation !==
      "upper_bound_plate_progress_equivalent_not_guaranteed_gold" ||
    mechanics.hunger_mite_included !== false ||
    !model ||
    !exactKeys(model, MODEL_KEYS) ||
    model.conversion !== "side_neutral_gold10_associational_logit" ||
    model.intercept !== GOLD10_INTERCEPT ||
    model.gold_coefficient !== GOLD10_COEF ||
    model.fight_swing_gold !== WIN_KILL ||
    model.secure_if_win !== 1 ||
    model.secure_if_lose !== 0 ||
    !reference ||
    !exactKeys(reference, REFERENCE_KEYS) ||
    reference.baseline_gold !== 0 ||
    reference.objective_gold_equivalent !== OBJECTIVE_GOLD ||
    reference.two_wave_leave_farm_gold !== LEAVE_FARM_TWO_WAVE ||
    reference.fight_swing_gold !== WIN_KILL ||
    !finite(article.p_star) ||
    !finite(article.p_star_pct) ||
    !finite(article.edge_at_50_pp) ||
    typeof article.interpretation !== "string" ||
    !article.interpretation.includes("not an identified action threshold") ||
    !Array.isArray(article.limitations) ||
    article.limitations.length !== 4 ||
    !article.limitations.every(
      (item) => typeof item === "string" && item.trim().length > 0,
    )
  ) {
    return { ok: false, reason: "article_schema_or_mechanics_invalid" };
  }

  const serialized = JSON.stringify(article);
  if (/leave[_ -]?mix|oe_sister|breakeven_p_win_fight|~24%/i.test(serialized)) {
    return { ok: false, reason: "article_auxiliary_estimand_present" };
  }

  const curve = parseCurve(article.curve);
  const leaveFarm = parseLeaveFarm(article.by_leave_farm_F);
  const goldRows = parseGoldRows(
    article.by_precontest_gold_B_two_wave_leave,
  );
  if (!curve || !leaveFarm || !goldRows) {
    return { ok: false, reason: "article_derived_rows_invalid" };
  }

  const expectedPStar = articlePStarAtGoldB(0);
  const atFifty = curve.find((row) => row.p_win_fight === 0.5);
  if (
    expectedPStar == null ||
    !atFifty ||
    !close(article.p_star, rounded(expectedPStar, 12), 5e-13) ||
    article.p_star_pct !== rounded(100 * expectedPStar, 2) ||
    article.edge_at_50_pp !== atFifty.edge_contest_minus_leave_pp ||
    atFifty.model_preference !== "LEAVE" ||
    article.p_star_pct !== pin.contestBarPct ||
    article.edge_at_50_pp !== pin.atFiftyEdgePp
  ) {
    return { ok: false, reason: "article_headline_parity_failed" };
  }

  return {
    ok: true,
    article: {
      ...(article as unknown as ValidatedArticleEv),
      mechanics: mechanics as unknown as ValidatedArticleEv["mechanics"],
      model: model as unknown as ValidatedArticleEv["model"],
      reference_knobs:
        reference as unknown as ValidatedArticleEv["reference_knobs"],
      curve,
      by_leave_farm_F: leaveFarm,
      by_precontest_gold_B_two_wave_leave: goldRows,
      limitations: article.limitations as string[],
    },
    atFifty,
  };
}
