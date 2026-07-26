/**
 * League-calibrated Draft Score (v3) — TypeScript port of lol_kills.draft_score.
 * Loads a slim runtime artifact (no Python / no warehouse at request time).
 */
import { readFileSync } from "fs";
import path from "path";

const DEFAULT_TEMP = 1.4;
const BLUE_SIDE_BONUS = 0.03;
const PRIOR_N = 25;

const LEAGUE_TIER: Record<string, string> = {
  LCK: "tier1",
  LPL: "tier1",
  LEC: "west",
  LCS: "west",
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
  elo_diff?: number | null;
  blue_roles?: string[] | null;
  red_roles?: string[] | null;
};

export type DraftScoreResult = {
  draft_score_blue: number;
  draft_score_red: number;
  draft_edge: number;
  confidence: number;
  p_blue_draft: number;
  wr_bump_pp: number;
  posterior_width: number;
  calibration: {
    league: string | null | undefined;
    source: string;
    temperature: number;
    legacy_temp: number;
    p_blue_legacy_1_4: number;
    p_blue_with_strength: number | null;
    dp_per_logit: number;
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
  };
  blue: string[];
  red: string[];
  note: string;
};

let cached: DraftRuntime | null = null;

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

function normalizeChamp(name: string): string {
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
  const champs = champsIn.map(normalizeChamp);
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

export function draftScore(input: DraftScoreInput): DraftScoreResult {
  const rt = loadRuntime();
  const scale = draftTemperature(input.league, rt);
  const b = scoreSide(input.blue, rt, BLUE_SIDE_BONUS);
  const r = scoreSide(input.red, rt, 0);

  let pairLogit = 0;
  const roles = (input.blue_roles ?? ["top", "jng", "mid", "bot", "sup"]).map(normRole);
  if (b.champs.length === 5 && r.champs.length === 5) {
    for (let i = 0; i < 5; i++) {
      const key = `${roles[i]}|${b.champs[i]}|${r.champs[i]}`;
      const row = rt.role_pairs[key];
      if (row) pairLogit += 0.35 * Number(row.logit);
    }
  }

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
  if (input.elo_diff != null && scale.source !== "legacy_1.4") {
    pWithStrength = sigmoid(
      scale.intercept + scale.coef_elo * (Number(input.elo_diff) / 400) + temp * winEdge,
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
    wr_bump_pp: round(wrBumpPp, 2),
    posterior_width: round(width, 3),
    calibration: {
      league: input.league,
      source: scale.source,
      temperature: round(temp, 4),
      legacy_temp: DEFAULT_TEMP,
      p_blue_legacy_1_4: round(legacyShrunk, 4),
      p_blue_with_strength: pWithStrength != null ? round(pWithStrength, 4) : null,
      dp_per_logit: round(scale.dp_per_logit, 4),
    },
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
      "Draft Score v3: league-calibrated WR temperature from OE study; one input among several on a board. wr_bump_pp ≈ Elo-controlled ΔP from residual ridge × edge × conf.",
  };
}

function round(x: number, digits: number): number {
  const p = 10 ** digits;
  return Math.round(x * p) / p;
}
