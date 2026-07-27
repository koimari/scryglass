/**
 * League-calibrated Draft Score (v3) — TypeScript port of lol_kills.draft_score.
 * Loads a slim runtime artifact (no Python / no warehouse at request time).
 */
import { readFileSync } from "fs";
import path from "path";
import {
  compositionDraftCatalog,
  compositionDraftScore,
  compositionPartialDraftScore,
} from "./draftComposition";

const DEFAULT_TEMP = 1.4;
const BLUE_SIDE_BONUS = 0.03;
const PRIOR_N = 25;

export const DRAFT_ROLES = ["top", "jng", "mid", "bot", "sup"] as const;
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
    temperature?: number;
    legacy_temp?: number;
    p_blue_legacy_1_4?: number;
    p_blue_with_strength: number | null;
    dp_per_logit?: number;
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
        games: number;
        shrinkage: number;
        label: string;
        uncertainty_logit: number;
      };
    }>;
    reconciles: boolean;
    attribution: string;
  };
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
  projected_wr: number;
  delta_pp: number;
  confidence: number;
  sample_games: number;
  evidence: "Settled" | "Observed" | "Thin" | "Unseen role";
};

export type DraftTimelineRow = DraftAction & {
  pick_number: number;
  projected_wr: number;
  delta_pp: number;
  confidence: number;
};

export type DraftSandboxResult = {
  perspective: DraftSide;
  recommendation_side: DraftSide;
  next_side: DraftSide;
  candidate_role: DraftCandidateRole;
  current: {
    projected_wr: number;
    confidence: number;
    score: DraftScoreResult;
  };
  timeline: DraftTimelineRow[];
  recommendations: DraftRecommendation[];
  open_roles: DraftRole[];
  note: string;
};

let cached: DraftRuntime | null = null;
let catalogCached: DraftChampion[] | null = null;

type StrengthRow = {
  team?: string;
  player?: string;
  mu_total: number;
  sigma: number;
  last_team?: string | null;
};

type StrengthPack = { teams: StrengthRow[]; players: StrengthRow[] };
let strengthPack: StrengthPack | null | undefined;

function loadStrengthPack(): StrengthPack | null {
  if (strengthPack !== undefined) return strengthPack;
  try {
    const latest = JSON.parse(
      readFileSync(path.join(process.cwd(), "public", "packs", "latest.json"), "utf8"),
    ) as { pack_id?: string };
    const packId = latest.pack_id;
    if (!packId) throw new Error("latest pack id missing");
    const root = path.join(process.cwd(), "public", "packs", packId, "features");
    strengthPack = {
      teams: JSON.parse(readFileSync(path.join(root, "ratings_snapshot.json"), "utf8")) as StrengthRow[],
      players: JSON.parse(
        readFileSync(path.join(root, "player_ratings_snapshot.json"), "utf8"),
      ) as StrengthRow[],
    };
  } catch {
    strengthPack = null;
  }
  return strengthPack;
}

function adjustedRating(row: StrengthRow, floor: number): number {
  return Number(row.mu_total) - Math.max(0, Number(row.sigma) - floor);
}

function teamMatch(name: string, candidate: string): boolean {
  const a = normKey(name);
  const b = normKey(candidate);
  return a === b || a.includes(b) || b.includes(a);
}

function resolveStrength(input: DraftScoreInput): DraftScoreInput {
  const teamDiff = input.team_elo_diff ?? input.elo_diff;
  const playerDiff = input.player_elo_diff;
  if (playerDiff != null || !input.blue_team || !input.red_team) {
    return {
      ...input,
      team_elo_diff: teamDiff,
      strength_source:
        input.strength_source ?? (teamDiff != null || playerDiff != null ? "explicit pre-match strength" : null),
    };
  }
  const pack = loadStrengthPack();
  if (!pack) return input;
  const blueRow = pack.teams.find((row) => teamMatch(input.blue_team!, String(row.team ?? "")));
  const redRow = pack.teams.find((row) => teamMatch(input.red_team!, String(row.team ?? "")));
  if (!blueRow || !redRow) return input;
  const resolved: DraftScoreInput = {
    ...input,
    team_elo_diff: teamDiff ?? adjustedRating(blueRow, 25) - adjustedRating(redRow, 25),
    strength_source: teamDiff != null ? "pre-match team + current roster proxy" : "current pack adjusted team + roster proxy",
  };
  if (input.blue_players?.length && input.red_players?.length) {
    const bluePlayers = pack.players.filter((row) => input.blue_players!.some((p) => teamMatch(p, String(row.player ?? ""))));
    const redPlayers = pack.players.filter((row) => input.red_players!.some((p) => teamMatch(p, String(row.player ?? ""))));
    if (bluePlayers.length && redPlayers.length) {
      const b = bluePlayers.reduce((sum, row) => sum + adjustedRating(row, 28), 0) / bluePlayers.length;
      const r = redPlayers.reduce((sum, row) => sum + adjustedRating(row, 28), 0) / redPlayers.length;
      resolved.player_elo_diff = b - r;
      resolved.strength_source = "explicit roster/player ratings + adjusted team rating";
    }
  } else {
    const roster = (team: string) => {
      const exact = pack.players.filter((row) => normKey(String(row.last_team ?? "")) === normKey(team));
      const candidates = exact.length
        ? exact
        : pack.players.filter((row) => teamMatch(team, String(row.last_team ?? "")));
      return candidates
        .sort((a, b) => adjustedRating(b, 28) - adjustedRating(a, 28))
        .slice(0, 5);
    };
    const bluePlayers = roster(input.blue_team!);
    const redPlayers = roster(input.red_team!);
    if (bluePlayers.length && redPlayers.length) {
      const b = bluePlayers.reduce((sum, row) => sum + adjustedRating(row, 28), 0) / bluePlayers.length;
      const r = redPlayers.reduce((sum, row) => sum + adjustedRating(row, 28), 0) / redPlayers.length;
      resolved.player_elo_diff = b - r;
    }
  }
  return resolved;
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
  const enriched = resolveStrength(input);
  if (enriched.blue.length === 5 && enriched.red.length === 5) {
    const composition = compositionDraftScore(enriched);
    if (composition) return composition;
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
      source: "legacy partial-draft fallback",
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
      "Legacy partial-draft fallback: league-calibrated Draft Score comparison. Complete five-versus-five boards use the full-composition runtime when available.",
  };
}

export function draftCatalog(): DraftChampion[] {
  if (catalogCached) return catalogCached;
  const compositionCatalog = compositionDraftCatalog();
  if (compositionCatalog?.length) {
    catalogCached = compositionCatalog.map((row) => ({
      name: row.name,
      roles: row.roles.filter(isDraftRole),
      games: row.games,
      role_games: Object.fromEntries(
        Object.entries(row.role_games)
          .filter(([role]) => isDraftRole(role))
          .map(([role, games]) => [role, Number(games) || 0]),
      ) as Partial<Record<DraftRole, number>>,
    }));
    return catalogCached;
  }
  const rt = loadRuntime();
  const roleSets = new Map<string, Set<DraftRole>>();
  for (const key of Object.keys(rt.role_pairs)) {
    const [roleRaw, blueChampion, redChampion] = key.split("|");
    if (!isDraftRole(roleRaw)) continue;
    for (const champion of [blueChampion, redChampion]) {
      const roles = roleSets.get(champion) ?? new Set<DraftRole>();
      roles.add(roleRaw);
      roleSets.set(champion, roles);
    }
  }
  catalogCached = Object.keys(rt.champ_game_counts)
    .sort((a, b) => a.localeCompare(b))
    .map((name) => ({
      name,
      roles: DRAFT_ROLES.filter((role) => roleSets.get(name)?.has(role)),
      games: Number(rt.champ_game_counts[name] ?? 0),
    }));
  return catalogCached;
}

function sideProbability(score: DraftScoreResult, side: DraftSide): number {
  return side === "blue" ? score.p_blue_draft : 1 - score.p_blue_draft;
}

function scoreActions(
  actions: DraftAction[],
  league: string | null | undefined,
  eloDiff: number | null | undefined,
): DraftScoreResult {
  const blue = actions.filter((action) => action.side === "blue");
  const red = actions.filter((action) => action.side === "red");
  const input: DraftScoreInput = {
    blue: blue.map((action) => action.champion),
    red: red.map((action) => action.champion),
    league,
    elo_diff: eloDiff,
    blue_roles: blue.map((action) => action.role || ""),
    red_roles: red.map((action) => action.role || ""),
  };
  return compositionPartialDraftScore(input) ?? draftScore(input);
}

function evidenceLabel(games: number): DraftRecommendation["evidence"] {
  if (games <= 0) return "Unseen role";
  if (games >= 200) return "Settled";
  if (games >= 75) return "Observed";
  return "Thin";
}

export function analyzeDraftSandbox(input: {
  actions: DraftAction[];
  perspective: DraftSide;
  next_side: DraftSide;
  candidate_role?: DraftCandidateRole;
  excluded?: string[];
  league?: string | null;
  elo_diff?: number | null;
  limit?: number;
}): DraftSandboxResult {
  const actions = input.actions.map((action) => ({
    ...action,
    champion: normalizeDraftChampion(action.champion),
  }));
  const perspective = input.perspective;
  const currentScore = scoreActions(actions, input.league, input.elo_diff);
  const currentProbability = sideProbability(currentScore, perspective);
  const currentRecommendationProbability = sideProbability(currentScore, input.next_side);
  const timeline: DraftTimelineRow[] = [];
  const prefix: DraftAction[] = [];
  let previousProbability = 0.5;

  actions.forEach((action, index) => {
    prefix.push(action);
    const score = scoreActions(prefix, input.league, input.elo_diff);
    const probability = sideProbability(score, perspective);
    timeline.push({
      ...action,
      pick_number: index + 1,
      projected_wr: round(probability, 4),
      delta_pp: round(100 * (probability - previousProbability), 2),
      confidence: score.confidence,
    });
    previousProbability = probability;
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

  for (const candidate of catalog) {
    if (selected.has(normKey(candidate.name)) || excluded.has(normKey(candidate.name))) continue;
    let roles: Array<DraftRole | null>;
    if (candidateRole === "any") {
      roles = openRoles;
    } else if (candidateRole === "open") {
      roles = candidate.roles.filter((role) => openRoles.includes(role));
      if (!roles.length) continue;
    } else {
      if (!openRoles.includes(candidateRole)) continue;
      roles = [candidateRole];
    }

    let best: DraftRecommendation | null = null;
    for (const role of roles) {
      const score = scoreActions(
        [...actions, { side: input.next_side, champion: candidate.name, role }],
        input.league,
        input.elo_diff,
      );
      const projected = sideProbability(score, input.next_side);
      const row: DraftRecommendation = {
        champion: candidate.name,
        role,
        projected_wr: round(projected, 4),
        delta_pp: round(100 * (projected - currentRecommendationProbability), 2),
        confidence: score.confidence,
        sample_games: role ? Number(candidate.role_games?.[role] ?? 0) : candidate.games,
        evidence: evidenceLabel(
          role ? Number(candidate.role_games?.[role] ?? 0) : candidate.games,
        ),
      };
      if (!best || row.projected_wr > best.projected_wr) best = row;
    }
    if (best) recommendations.push(best);
  }

  recommendations.sort(
    (a, b) =>
      b.projected_wr - a.projected_wr ||
      b.sample_games - a.sample_games ||
      a.champion.localeCompare(b.champion),
  );

  return {
    perspective,
    recommendation_side: input.next_side,
    next_side: input.next_side,
    candidate_role: candidateRole,
    current: {
      projected_wr: round(currentProbability, 4),
      confidence: currentScore.confidence,
      score: currentScore,
    },
    timeline,
    recommendations: recommendations.slice(0, clip(input.limit ?? 12, 1, 200)),
    open_roles: openRoles,
    note:
      "Partial full-composition counterfactual. Rankings use role-aware direct effects plus learned ally synergy and enemy interactions among selected champions. Unfilled seats are neutralized, so pick-to-pick changes are not calibrated live win probabilities.",
  };
}

function round(x: number, digits: number): number {
  const p = 10 ** digits;
  return Math.round(x * p) / p;
}
