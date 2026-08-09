/**
 * League-calibrated Draft Score (v3) — TypeScript port of lol_kills.draft_score.
 * Loads a slim runtime artifact (no Python / no warehouse at request time).
 */
import { readFileSync } from "fs";
import path from "path";

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
  as_of?: string;
  n_maps?: number;
  kill_beta: Record<string, number>;
  win_delta: Record<string, number>;
  champ_game_counts: Record<string, number>;
  champion_roster?: string[];
  champion_roles?: Record<string, DraftRole[]>;
  champion_effects?: Record<string, RolePair>;
  role_effects?: Record<string, RolePair>;
  champion_role_counts?: Record<string, Partial<Record<DraftRole, number>>>;
  champion_archetypes?: Record<string, string[]>;
  ally_synergy?: Record<string, RolePair>;
  archetype_synergy?: Record<string, RolePair>;
  counter_pairs?: Record<string, RolePair>;
  archetype_counters?: Record<string, RolePair>;
  role_pairs: Record<string, RolePair>;
  calibration: {
    by_league?: Record<string, CalRow>;
    by_tier?: Record<string, CalRow>;
    joint_logistic_global?: CalRow;
    residual_ridge?: { coef_dp_per_logit?: number };
  };
  elo_calibration?: {
    team?: { intercept?: number; coef?: number };
    player?: { intercept?: number; coef?: number };
    strength_blend?: {
      intercept?: number;
      coef_team?: number;
      coef_player?: number;
    };
  };
  evaluation?: {
    split?: string;
    selected_alpha?: number;
    selected_half_life_days?: number;
    interaction_gates?: {
      role_residual?: number;
      ally_synergy?: number;
      cross_counter?: number;
      composition_counter?: number;
      same_role?: number;
    };
    baseline_holdout?: Record<string, number>;
    interaction_holdout?: Record<string, number>;
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
  team_elo_diff?: number | null;
  player_elo_diff?: number | null;
  blue_roles?: string[] | null;
  red_roles?: string[] | null;
  player_context?: DraftPlayerContext | null;
};

export type DraftScoreResult = {
  draft_score_blue: number;
  draft_score_red: number;
  draft_edge: number;
  confidence: number;
  p_blue_draft: number;
  p_blue_combined: number;
  wr_bump_pp: number;
  posterior_width: number;
  calibration: {
    league: string | null | undefined;
    source: string;
    temperature: number;
    legacy_temp: number;
    p_blue_legacy_1_4: number;
    p_blue_with_strength: number | null;
    strength_source: string;
    p_team_strength: number | null;
    p_player_strength: number | null;
    dp_per_logit: number;
  };
  components: {
    win_logit_blue: number;
    win_logit_red: number;
    synergy_logit_blue: number;
    synergy_logit_red: number;
    counter_logit: number;
    pair_logit: number;
    player_logit_blue: number;
    player_logit_red: number;
    strength_logit: number;
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

export type DraftChampion = {
  name: string;
  roles: DraftRole[];
  games: number;
  role_games: Partial<Record<DraftRole, number>>;
  modeled: boolean;
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
  evidence: "Settled" | "Observed" | "Thin" | "Neutral prior";
  player: string | null;
  player_games: number;
  components: {
    champion: number;
    synergy: number;
    counters: number;
    lane: number;
    comfort: number;
  };
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
  model: {
    as_of: string | null;
    maps: number | null;
    recency_half_life_days: number | null;
    interaction_gates: {
      role: number;
      synergy: number;
      counters: number;
      composition: number;
      lane: number;
    };
    baseline_brier: number | null;
    interaction_brier: number | null;
  };
  note: string;
};

export type DraftMastery = {
  logit: number;
  n: number;
  effective_n?: number;
};

export type DraftContextPlayer = {
  player: string;
  team: string;
  role: DraftRole | null;
  rating: number;
  raw_rating: number;
  sigma: number;
  n_maps: number;
  mastery: Record<string, DraftMastery>;
};

export type DraftContextTeam = {
  team: string;
  league: string | null;
  tier: "tier1" | "tier2" | "tier3" | null;
  rating: number;
  raw_rating: number;
  sigma: number;
  roster: DraftContextPlayer[];
};

export type DraftContext = {
  version: number;
  as_of: string;
  teams: DraftContextTeam[];
  players: Record<string, DraftContextPlayer>;
  note: string;
};

export type DraftPlayerContext = Partial<
  Record<DraftSide, Partial<Record<DraftRole, DraftContextPlayer>>>
>;

let cached: DraftRuntime | null = null;
let catalogCached: DraftChampion[] | null = null;
let contextCached: DraftContext | null = null;

function loadRuntime(): DraftRuntime {
  if (cached) return cached;
  const file = path.join(process.cwd(), "data", "draft", "runtime.json");
  cached = JSON.parse(readFileSync(file, "utf8")) as DraftRuntime;
  return cached;
}

export function draftContext(): DraftContext {
  if (contextCached) return contextCached;
  const file = path.join(process.cwd(), "data", "draft", "context.json");
  contextCached = JSON.parse(readFileSync(file, "utf8")) as DraftContext;
  return contextCached;
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
  rolesIn: string[] | null | undefined,
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
  for (const [index, c] of champs.entries()) {
    const cnt = rt.champ_game_counts[c] ?? 0;
    const w = posteriorWeight(cnt);
    confParts.push(w);
    if (c in rt.win_delta || c in rt.kill_beta) known += 1;
    winLogit += w * (rt.win_delta[c] ?? 0);
    const role = normRole(rolesIn?.[index] ?? "");
    if (isDraftRole(role)) {
      const roleCount = rt.champion_role_counts?.[c]?.[role] ?? 0;
      const roleWeight = posteriorWeight(roleCount, 18);
      winLogit +=
        roleWeight *
        Number(rt.role_effects?.[`${role}|${c}`]?.logit ?? 0);
    }
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

function synergyKey(first: string, second: string): string {
  return [first, second].sort((a, b) => a.localeCompare(b)).join("|");
}

function championArchetypes(champion: string, rt: DraftRuntime): string[] {
  return rt.champion_archetypes?.[champion] ?? [];
}

function interactionScore(
  blue: string[],
  red: string[],
  blueRoles: string[] | null | undefined,
  redRoles: string[] | null | undefined,
  rt: DraftRuntime,
  playerContext?: DraftPlayerContext | null,
): {
  synergyBlue: number;
  synergyRed: number;
  counters: number;
  lane: number;
  playerBlue: number;
  playerRed: number;
} {
  const blueByRole = roleMap(blue, blueRoles);
  const redByRole = roleMap(red, redRoles);
  let synergyBlue = 0;
  let synergyRed = 0;
  let counters = 0;
  let lane = 0;
  let playerBlue = 0;
  let playerRed = 0;

  for (const [first, second] of blue.flatMap((champion, index) =>
    blue.slice(index + 1).map((other) => [champion, other] as const),
  )) {
    synergyBlue += Number(rt.ally_synergy?.[synergyKey(first, second)]?.logit ?? 0);
    for (const firstTag of championArchetypes(first, rt)) {
      for (const secondTag of championArchetypes(second, rt)) {
        synergyBlue += Number(
          rt.archetype_synergy?.[synergyKey(firstTag, secondTag)]?.logit ?? 0,
        );
      }
    }
  }
  for (const [first, second] of red.flatMap((champion, index) =>
    red.slice(index + 1).map((other) => [champion, other] as const),
  )) {
    synergyRed += Number(rt.ally_synergy?.[synergyKey(first, second)]?.logit ?? 0);
    for (const firstTag of championArchetypes(first, rt)) {
      for (const secondTag of championArchetypes(second, rt)) {
        synergyRed += Number(
          rt.archetype_synergy?.[synergyKey(firstTag, secondTag)]?.logit ?? 0,
        );
      }
    }
  }
  for (const blueChampion of blue) {
    for (const redChampion of red) {
      counters += Number(
        rt.counter_pairs?.[`${blueChampion}|${redChampion}`]?.logit ?? 0,
      );
      for (const blueTag of championArchetypes(blueChampion, rt)) {
        for (const redTag of championArchetypes(redChampion, rt)) {
          counters += Number(
            rt.archetype_counters?.[`${blueTag}|${redTag}`]?.logit ?? 0,
          );
        }
      }
    }
  }
  for (const role of DRAFT_ROLES) {
    const blueChampion = blueByRole.get(role);
    const redChampion = redByRole.get(role);
    if (blueChampion && redChampion) {
      const row = rt.role_pairs[`${role}|${blueChampion}|${redChampion}`];
      if (row) lane += Number(row.logit);
    }
    if (blueChampion) {
      playerBlue += Number(
        playerContext?.blue?.[role]?.mastery?.[blueChampion]?.logit ?? 0,
      );
    }
    if (redChampion) {
      playerRed += Number(
        playerContext?.red?.[role]?.mastery?.[redChampion]?.logit ?? 0,
      );
    }
  }
  return {
    synergyBlue,
    synergyRed,
    counters,
    lane,
    playerBlue,
    playerRed,
  };
}

function calibratedStrength(
  rt: DraftRuntime,
  teamDiff: number | null | undefined,
  playerDiff: number | null | undefined,
): {
  logit: number;
  source: string;
  pTeam: number | null;
  pPlayer: number | null;
} {
  const calibration = rt.elo_calibration ?? {};
  const team = calibration.team;
  const player = calibration.player;
  const pTeam =
    teamDiff != null && team?.coef != null
      ? sigmoid(Number(team.intercept ?? 0) + Number(team.coef) * (teamDiff / 400))
      : null;
  const pPlayer =
    playerDiff != null && player?.coef != null
      ? sigmoid(
          Number(player.intercept ?? 0) + Number(player.coef) * (playerDiff / 400),
        )
      : null;
  if (pTeam != null && pPlayer != null) {
    const blend = calibration.strength_blend;
    const probability =
      blend?.coef_team != null && blend?.coef_player != null
        ? sigmoid(
            Number(blend.intercept ?? 0) +
              Number(blend.coef_team) * pTeam +
              Number(blend.coef_player) * pPlayer,
          )
        : 0.6 * pTeam + 0.4 * pPlayer;
    return {
      logit: Math.log(probability / (1 - probability)),
      source: "team + lineup Elo",
      pTeam,
      pPlayer,
    };
  }
  const probability = pTeam ?? pPlayer;
  if (probability != null) {
    return {
      logit: Math.log(probability / (1 - probability)),
      source: pTeam != null ? "team Elo" : "lineup Elo",
      pTeam,
      pPlayer,
    };
  }
  return { logit: 0, source: "even-strength assumption", pTeam, pPlayer };
}

export function draftScore(input: DraftScoreInput): DraftScoreResult {
  const rt = loadRuntime();
  const scale = draftTemperature(input.league, rt);
  const b = scoreSide(input.blue, input.blue_roles, rt, BLUE_SIDE_BONUS);
  const r = scoreSide(input.red, input.red_roles, rt, 0);

  const interactions = interactionScore(
    b.champs,
    r.champs,
    input.blue_roles,
    input.red_roles,
    rt,
    input.player_context,
  );

  const winEdge =
    b.win_logit -
    r.win_logit +
    interactions.synergyBlue -
    interactions.synergyRed +
    interactions.counters +
    interactions.lane +
    interactions.playerBlue -
    interactions.playerRed;
  const temp = scale.temp;
  const pBlueRaw = sigmoid(winEdge * temp);
  let conf = Math.min(b.confidence, r.confidence);
  const width = 0.5 * (b.posterior_width + r.posterior_width);
  conf = clip(conf * (1 - 0.5 * width), 0.05, 0.98);
  const completion = clip((b.champs.length + r.champs.length) / 10, 0, 1);
  conf = clip(conf * (0.25 + 0.75 * completion), 0.05, 0.98);
  const pShrunk = 0.5 + (pBlueRaw - 0.5) * conf;
  const scoreBlue = 100 * pShrunk;
  const scoreRed = 100 - scoreBlue;
  const wrBumpPp = 100 * scale.dp_per_logit * winEdge * conf;

  const legacyTeamDiff =
    input.team_elo_diff == null ? input.elo_diff : input.team_elo_diff;
  const strength = calibratedStrength(rt, legacyTeamDiff, input.player_elo_diff);
  const pWithStrength =
    strength.source === "even-strength assumption"
      ? null
      : sigmoid(strength.logit + temp * winEdge * conf);
  const pCombined = pWithStrength ?? pShrunk;

  const legacyP = sigmoid(winEdge * DEFAULT_TEMP);
  const legacyShrunk = 0.5 + (legacyP - 0.5) * conf;

  return {
    draft_score_blue: round(scoreBlue, 2),
    draft_score_red: round(scoreRed, 2),
    draft_edge: round(scoreBlue - scoreRed, 2),
    confidence: round(conf, 3),
    p_blue_draft: round(pShrunk, 4),
    p_blue_combined: round(pCombined, 4),
    wr_bump_pp: round(wrBumpPp, 2),
    posterior_width: round(width, 3),
    calibration: {
      league: input.league,
      source: scale.source,
      temperature: round(temp, 4),
      legacy_temp: DEFAULT_TEMP,
      p_blue_legacy_1_4: round(legacyShrunk, 4),
      p_blue_with_strength: pWithStrength != null ? round(pWithStrength, 4) : null,
      strength_source: strength.source,
      p_team_strength: strength.pTeam != null ? round(strength.pTeam, 4) : null,
      p_player_strength: strength.pPlayer != null ? round(strength.pPlayer, 4) : null,
      dp_per_logit: round(scale.dp_per_logit, 4),
    },
    components: {
      win_logit_blue: round(b.win_logit, 4),
      win_logit_red: round(r.win_logit, 4),
      synergy_logit_blue: round(interactions.synergyBlue, 4),
      synergy_logit_red: round(interactions.synergyRed, 4),
      counter_logit: round(interactions.counters, 4),
      pair_logit: round(interactions.lane, 4),
      player_logit_blue: round(interactions.playerBlue, 4),
      player_logit_red: round(interactions.playerRed, 4),
      strength_logit: round(strength.logit, 4),
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
      "Draft recommendation v4: regularized champion, allied synergy, enemy counter, same-role, and player-comfort terms; team and lineup Elo enter through an independently calibrated strength prior.",
  };
}

export function draftCatalog(): DraftChampion[] {
  if (catalogCached) return catalogCached;
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
  for (const [champion, roles] of Object.entries(rt.champion_roles ?? {})) {
    roleSets.set(champion, new Set(roles.filter(isDraftRole)));
  }
  const roster = new Set([
    ...(rt.champion_roster ?? []),
    ...Object.keys(rt.champ_game_counts),
    ...Object.keys(rt.win_delta),
    ...roleSets.keys(),
  ]);
  catalogCached = [...roster]
    .sort((a, b) => a.localeCompare(b))
    .map((name) => ({
      name,
      roles: DRAFT_ROLES.filter((role) => roleSets.get(name)?.has(role)),
      games: Number(rt.champ_game_counts[name] ?? 0),
      role_games: Object.fromEntries(
        DRAFT_ROLES.map((role) => [
          role,
          Number(rt.champion_role_counts?.[name]?.[role] ?? 0),
        ]),
      ),
      modeled: Number(rt.champ_game_counts[name] ?? 0) > 0,
    }));
  return catalogCached;
}

function sideProbability(score: DraftScoreResult, side: DraftSide): number {
  return side === "blue" ? score.p_blue_combined : 1 - score.p_blue_combined;
}

function scoreActions(
  actions: DraftAction[],
  league: string | null | undefined,
  teamEloDiff: number | null | undefined,
  playerEloDiff: number | null | undefined,
  playerContext: DraftPlayerContext | null | undefined,
): DraftScoreResult {
  const blue = actions.filter((action) => action.side === "blue");
  const red = actions.filter((action) => action.side === "red");
  return draftScore({
    blue: blue.map((action) => action.champion),
    red: red.map((action) => action.champion),
    league,
    team_elo_diff: teamEloDiff,
    player_elo_diff: playerEloDiff,
    blue_roles: blue.map((action) => action.role || ""),
    red_roles: red.map((action) => action.role || ""),
    player_context: playerContext,
  });
}

function evidenceLabel(games: number): DraftRecommendation["evidence"] {
  if (games <= 0) return "Neutral prior";
  if (games >= 200) return "Settled";
  if (games >= 75) return "Observed";
  return "Thin";
}

function recommendationComponents(
  current: DraftScoreResult,
  candidate: DraftScoreResult,
  side: DraftSide,
): DraftRecommendation["components"] {
  const direction = side === "blue" ? 1 : -1;
  return {
    champion: round(
      direction *
        ((candidate.components.win_logit_blue - candidate.components.win_logit_red) -
          (current.components.win_logit_blue - current.components.win_logit_red)),
      4,
    ),
    synergy: round(
      direction *
        ((candidate.components.synergy_logit_blue -
          candidate.components.synergy_logit_red) -
          (current.components.synergy_logit_blue - current.components.synergy_logit_red)),
      4,
    ),
    counters: round(
      direction *
        (candidate.components.counter_logit - current.components.counter_logit),
      4,
    ),
    lane: round(
      direction * (candidate.components.pair_logit - current.components.pair_logit),
      4,
    ),
    comfort: round(
      direction *
        ((candidate.components.player_logit_blue - candidate.components.player_logit_red) -
          (current.components.player_logit_blue - current.components.player_logit_red)),
      4,
    ),
  };
}

export function analyzeDraftSandbox(input: {
  actions: DraftAction[];
  perspective: DraftSide;
  next_side: DraftSide;
  candidate_role?: DraftCandidateRole;
  excluded?: string[];
  league?: string | null;
  elo_diff?: number | null;
  team_elo_diff?: number | null;
  player_elo_diff?: number | null;
  player_context?: DraftPlayerContext | null;
  limit?: number;
}): DraftSandboxResult {
  const runtime = loadRuntime();
  const actions = input.actions.map((action) => ({
    ...action,
    champion: normalizeDraftChampion(action.champion),
  }));
  const perspective = input.perspective;
  const teamEloDiff =
    input.team_elo_diff == null ? input.elo_diff : input.team_elo_diff;
  const currentScore = scoreActions(
    actions,
    input.league,
    teamEloDiff,
    input.player_elo_diff,
    input.player_context,
  );
  const currentProbability = sideProbability(currentScore, perspective);
  const currentRecommendationProbability = sideProbability(currentScore, input.next_side);
  const timeline: DraftTimelineRow[] = [];
  const prefix: DraftAction[] = [];
  let previousProbability = sideProbability(
    scoreActions(
      [],
      input.league,
      teamEloDiff,
      input.player_elo_diff,
      input.player_context,
    ),
    perspective,
  );

  actions.forEach((action, index) => {
    prefix.push(action);
    const score = scoreActions(
      prefix,
      input.league,
      teamEloDiff,
      input.player_elo_diff,
      input.player_context,
    );
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
      roles = candidate.roles.length ? candidate.roles : [null];
    } else if (candidateRole === "open") {
      roles = candidate.roles.filter((role) => openRoles.includes(role));
      if (!roles.length) continue;
    } else {
      if (!candidate.roles.includes(candidateRole)) continue;
      roles = [candidateRole];
    }

    let best: DraftRecommendation | null = null;
    for (const role of roles) {
      const score = scoreActions(
        [...actions, { side: input.next_side, champion: candidate.name, role }],
        input.league,
        teamEloDiff,
        input.player_elo_diff,
        input.player_context,
      );
      const projected = sideProbability(score, input.next_side);
      const player =
        role != null
          ? input.player_context?.[input.next_side]?.[role] ?? null
          : null;
      const mastery = player?.mastery?.[candidate.name];
      const row: DraftRecommendation = {
        champion: candidate.name,
        role,
        projected_wr: round(projected, 4),
        delta_pp: round(100 * (projected - currentRecommendationProbability), 2),
        confidence: score.confidence,
        sample_games:
          role != null
            ? Number(candidate.role_games[role] ?? 0)
            : candidate.games,
        evidence: evidenceLabel(
          role != null
            ? Number(candidate.role_games[role] ?? 0)
            : candidate.games,
        ),
        player: player?.player ?? null,
        player_games: mastery?.n ?? 0,
        components: recommendationComponents(currentScore, score, input.next_side),
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
    recommendations: recommendations.slice(0, clip(input.limit ?? 12, 1, 30)),
    open_roles: openRoles,
    model: {
      as_of: runtime.as_of ?? null,
      maps: runtime.n_maps ?? null,
      recency_half_life_days:
        runtime.evaluation?.selected_half_life_days ?? null,
      interaction_gates: {
        role:
          runtime.evaluation?.interaction_gates?.role_residual ?? 0,
        synergy:
          runtime.evaluation?.interaction_gates?.ally_synergy ?? 0,
        counters:
          runtime.evaluation?.interaction_gates?.cross_counter ?? 0,
        composition:
          runtime.evaluation?.interaction_gates?.composition_counter ?? 0,
        lane:
          runtime.evaluation?.interaction_gates?.same_role ?? 0,
      },
      baseline_brier:
        runtime.evaluation?.baseline_holdout?.brier ?? null,
      interaction_brier:
        runtime.evaluation?.interaction_holdout?.brier ?? null,
    },
    note:
      "Partial-draft counterfactual. Unfilled seats contribute the fitted average. Recommendations combine regularized champion strength, allied synergy, enemy counters, same-role evidence, player comfort, and calibrated team/lineup Elo; unsupported terms remain at a neutral prior.",
  };
}

function round(x: number, digits: number): number {
  const p = 10 ** digits;
  return Math.round(x * p) / p;
}
