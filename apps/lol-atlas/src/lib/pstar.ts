/** Article opportunity-cost p* from side-neutral gold@10 logit (grubs study). */

const GOLD10_INTERCEPT = 0.1611182873782888;
const GOLD10_COEF = 0.000666860223609559;
const OBJECTIVE_GOLD = 115.6;
const LEAVE_FARM_TWO_WAVE = 241.33;
const WIN_KILL = 600;
const LOSS_KILL = -600;

function sigmoid(z: number): number {
  const x = Math.max(-35, Math.min(35, z));
  return 1 / (1 + Math.exp(-x));
}

/** Side-neutral map-win probability at own-team gold lead g. */
export function sideNeutralWinProb(gold: number): number {
  const linear = GOLD10_COEF * gold;
  return 0.5 * (sigmoid(GOLD10_INTERCEPT + linear) + sigmoid(-GOLD10_INTERCEPT + linear));
}

/**
 * Indifference fight-win p* for two-wave leave vs diagonal secure contest.
 * p* = (P_leave − P_loss) / (P_win − P_loss)
 */
export function articlePStarAtGoldB(
  B: number,
  leaveFarmGold = LEAVE_FARM_TWO_WAVE,
  objectiveGold = OBJECTIVE_GOLD,
): number | null {
  const pLeave = sideNeutralWinProb(B + leaveFarmGold - objectiveGold);
  const pWin = sideNeutralWinProb(B + objectiveGold + WIN_KILL);
  const pLoss = sideNeutralWinProb(B - objectiveGold + LOSS_KILL);
  const denom = pWin - pLoss;
  if (denom <= 1e-12) return null;
  const root = (pLeave - pLoss) / denom;
  if (root < 0 || root > 1) return null;
  return root;
}

/** KaTeX source for optional equation fold (display). Public UI says “contest bar”. */
function coefTex(x: number): string {
  const [coeff, exp] = x.toExponential(3).split("e");
  return `${coeff}\\times 10^{${Number(exp)}}`;
}

export const PSTAR_TEX = {
  pStar:
    "\\mathrm{contest\\,bar}(\\mathrm{gold\\,lead})=\\frac{P_{\\mathrm{leave}}-P_{\\mathrm{loss}}}{P_{\\mathrm{win}}-P_{\\mathrm{loss}}}",
  winProb:
    "P(\\mathrm{gold})=\\tfrac{1}{2}\\bigl[\\sigma(a+b\\cdot\\mathrm{gold})+\\sigma(-a+b\\cdot\\mathrm{gold})\\bigr]",
  params: `\\mathrm{farm}=${LEAVE_FARM_TWO_WAVE}\\,\\mathrm{g},\\; \\mathrm{objective}=${OBJECTIVE_GOLD}\\,\\mathrm{g},\\; \\text{fight swing }\\pm ${WIN_KILL}\\,\\mathrm{g},\\; a=${GOLD10_INTERCEPT.toFixed(4)},\\; b=${coefTex(GOLD10_COEF)}`,
} as const;

export const PSTAR_FX = {
  intercept: GOLD10_INTERCEPT,
  coef: GOLD10_COEF,
  objectiveGold: OBJECTIVE_GOLD,
  leaveFarmTwoWave: LEAVE_FARM_TWO_WAVE,
  winKill: WIN_KILL,
  lossKill: LOSS_KILL,
};

/** Contest-bar percent for charts (0–100). */
export function contestBarPct(pStar: number): number {
  return 100 * pStar;
}
