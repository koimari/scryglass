#!/usr/bin/env python3
"""
Intrinsic void-grub package value (LS estimand) — not take-regime WR.

Question answered
-----------------
Holding *fight outcome* aside: how many map-WR points does the mechanical
grub package itself buy (90g + 195 XP + Touch of the Void), and for what
fight-win odds does contesting beat "do nothing and gift the camp"?

This is deliberately *not* the OE association of who ended with 3–0
(+3.7pp after gold@10). That estimand confounds selection + tempo conversion.

Identification
--------------
1. Mechanical constants (wiki / LS sandbox language).
2. Map golddiff@10 and xpdiff@10 → P(map win) with era logits (OE, max 3 grubs).
3. Burn → gold-equivalent under explicit siege assumptions (sensitivity).
4. Contest EV: compare −Δp_obj (leave/gift) vs lottery over fight swings.

  python3 -m lol_kills.research.grubs_intrinsic_value
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from lol_kills.etl.paths import MODELS_DIR
from lol_kills.research.grubs_contest_study import engineer_v2, load_maps

OUT_JSON = MODELS_DIR / "grubs_intrinsic_value.json"
OUT_MD = MODELS_DIR / "grubs_intrinsic_value_summary.md"  # auto stub; full prose in *_paper.md
OUT_PAPER = MODELS_DIR / "grubs_intrinsic_value_paper.md"

# Prespecified competition scope for the paper. Regional, academy, challenger,
# and developmental circuits are excluded from the calibration population.
PAPER_LEAGUES = ("LCS", "LCK", "LEC", "LPL", "CBLOL")

# ---------------------------------------------------------------------------
# Mechanical package — wiki constants (Voidgrub camp / Touch / Turret / XP)
# ---------------------------------------------------------------------------

GOLD_PER_GRUB = 30  # local gold; global gold = 0 (wiki V26.01)
XP_PER_GRUB = 65
N_GRUBS_FULL = 3
PACKAGE_GOLD = GOLD_PER_GRUB * N_GRUBS_FULL  # 90
PACKAGE_XP = XP_PER_GRUB * N_GRUBS_FULL  # 195
XP_RADIUS = 2000  # units; epic XP shared equally among alive allies in radius
XP_TO_LEVEL_7 = 880  # wiki XP for level 7→8 (not 6→7); label carefully in prose
MODEL_SCALE = 1000.0  # fit gold/XP logits in 1,000-unit increments for stability

# Touch of the Void (wiki; V26.11 tick schedule).
# Basic attacks against structures apply a 4s burn that deals true damage every
# 0.5s; subsequent structure attacks refresh duration (they do not stack
# overlapping full burn packets). Tick damage applies only if the triggering
# instance can damage the structure. Summoned Hunger Voidmites apply/refresh
# the summoner's Touch. Melee/ranged tick tables below are post-26.11;
# pre-26.11 melee@3 was 12 true / 0.5s.
TOTV_TICK_MELEE_3 = 12  # pre-26.11
TOTV_TICK_RANGED_3 = 6  # pre-26.11
TOTV_TICK_MELEE_BY_STACK_POST_26_11 = (0, 4, 12, 16)
TOTV_TICK_RANGED_BY_STACK_POST_26_11 = (0, 2, 6, 8)
TOTV_TICK_MELEE_3_POST_26_11 = TOTV_TICK_MELEE_BY_STACK_POST_26_11[3]
TOTV_TICK_INTERVAL = 0.5
TOTV_BURN_DURATION = 4.0
TOTV_TICKS_PER_CYCLE = int(TOTV_BURN_DURATION / TOTV_TICK_INTERVAL)  # 8
# Hunger of the Void: granted at 3 Touch stacks; while in combat with a
# targetable (non-invulnerable) enemy structure, summon 1 allied Voidmite
# (15s cooldown). Mite HP mirrors a melee minion post-26.11; mite AAs
# apply/refresh Touch and do not disable turret Reinforced Armor.
HUNGER_MITE_CD = 15.0
HUNGER_TOUCH_STACKS_REQUIRED = 3

# Turret Plating — wiki Turret / Patch 26.1 (outer still standing in grub window).
# Destroying a plate grants 120 local gold. Plates are no longer removed at 14:00.
# Outer turret base HP = 9000. Plate claim thresholds at 10/25/45/70/100% missing HP.
# Sources: https://wiki.leagueoflegends.com/en-us/Turret
PLATE_GOLD = 120.0
OUTER_TURRET_HP = 9000.0
PLATE_MISSING_HP_FRACTIONS = (0.10, 0.25, 0.45, 0.70, 1.00)
# Exact HP widths on a 9000-HP outer (10/25/45/70/100% missing thresholds).
PLATE_HP_BANDS = (900.0, 1350.0, 1800.0, 2250.0, 2700.0)
PLATE_HP_FIRST = PLATE_HP_BANDS[0]  # 900 HP → first outer plate
PLATE_HP = PLATE_HP_FIRST  # burn→gold uses first outer plate (grub→push window)
PLATE_GOLD_FULL_OUTER = PLATE_GOLD * len(PLATE_HP_BANDS)  # 600g if all five claimed
# LS sandbox: burn ≈ 1 AA / 4s; midgame full-turret save ≈ 1.5 AA
LS_AA_PER_CYCLE = 1.0
LS_MID_AA_SAVED_FULL_TURRET = 1.5
# Representative AA damage on turret (level-7 / early-mid)
AA_DMG_EARLY = 90.0
AA_DMG_MID = 150.0


# ---------------------------------------------------------------------------
# Minion wave gold / XP — wiki Minion / Patch 26.1, conditioned on grub clock
# ---------------------------------------------------------------------------
# Void Grubs: spawn 08:00, despawn 14:45 (14:55 if in combat). While they are
# alive and no outer has fallen, leave-farm is outer-lane waves + outer plates.
#
# Fixed bounties (V26.01): melee 20g / 62 XP, caster 14g / 31 XP,
# siege (cannon) base 50g / 75 XP with cannon gold +1 per 90s upgrade.
# Pre-14:00 composition: always 3 melee + 3 caster; cannon every 3rd wave.
# At 14:00+: wave every 25s; on cannon waves one fewer melee; cannon every 2nd.
# Sources: https://wiki.leagueoflegends.com/en-us/Minion
#          https://wiki.leagueoflegends.com/en-us/Siege_minion
MELEE_MINION_GOLD = 20.0
CASTER_MINION_GOLD = 14.0
CANNON_MINION_GOLD_BASE = 50.0
CANNON_GOLD_UPGRADE_INTERVAL_S = 90.0
MELEE_MINION_XP = 62.0
CASTER_MINION_XP = 31.0
CANNON_MINION_XP = 75.0
WAVE_GOLD_NO_CANNON = 3 * MELEE_MINION_GOLD + 3 * CASTER_MINION_GOLD  # 102
WAVE_XP_NO_CANNON = 3 * MELEE_MINION_XP + 3 * CASTER_MINION_XP  # 279
WAVE_XP_CANNON = WAVE_XP_NO_CANNON + CANNON_MINION_XP  # 354
WAVE_XP_AVG_PRE14 = WAVE_XP_NO_CANNON + CANNON_MINION_XP / 3.0  # 304
# Wiki table row at 0:30 uses base cannon 50 → 102 + 50/3 = 118.6̅
WAVE_GOLD_AVG_WIKI_0_30 = WAVE_GOLD_NO_CANNON + CANNON_MINION_GOLD_BASE / 3.0
# Article leave ladder uses the grub-window expected wave (pre-14 composition)
# evaluated at 10:00, the mid-grub reference clock.
GRUB_SPAWN_S = 8 * 60.0
GRUB_DESPAWN_S = 14 * 60.0 + 45.0
GRUB_REF_CLOCK_S = 10 * 60.0
WAVE_PERIOD_S_PRE14 = 30.0


def cannon_minion_gold(game_time_s: float) -> float:
    """Siege/cannon gold bounty at game time: 50 + ⌊t / 90⌋ (V26.01)."""
    if game_time_s < 0:
        raise ValueError("game_time_s must be non-negative")
    upgrades = int(game_time_s // CANNON_GOLD_UPGRADE_INTERVAL_S)
    return CANNON_MINION_GOLD_BASE + float(upgrades)


def wave_gold_no_cannon() -> float:
    return WAVE_GOLD_NO_CANNON


def wave_gold_cannon(game_time_s: float) -> float:
    """Full cannon wave under pre-14 composition (3 melee + 3 caster + 1 cannon)."""
    return WAVE_GOLD_NO_CANNON + cannon_minion_gold(game_time_s)


def wave_gold_expected_pre14(game_time_s: float) -> float:
    """E[wave gold] for t < 14:00: cannon every third wave, full 3+3 composition."""
    if game_time_s >= 14 * 60.0:
        raise ValueError("pre-14 expected-wave formula is only valid before 14:00")
    return WAVE_GOLD_NO_CANNON + cannon_minion_gold(game_time_s) / 3.0


def wave_gold_expected_at_14_00() -> float:
    """Wiki Minion table at 14:00: ((8/3)×20)+(3×14)+(59/3) = 115.

    At 14:00, cannon waves drop one melee and cannon gold is 59.
    """
    return (8.0 / 3.0) * MELEE_MINION_GOLD + 3.0 * CASTER_MINION_GOLD + 59.0 / 3.0


# Headline leave-farm unit = grub-era expected wave at the 10:00 reference clock.
WAVE_GOLD_EARLY = wave_gold_expected_pre14(GRUB_REF_CLOCK_S)  # 102 + 56/3 = 120.6̅
WAVE_GOLD_AVG_EARLY = WAVE_GOLD_EARLY  # alias used throughout
CANNON_MINION_GOLD_EARLY = cannon_minion_gold(GRUB_REF_CLOCK_S)  # 56 at 10:00
WAVE_GOLD_CANNON_AT_REF = wave_gold_cannon(GRUB_REF_CLOCK_S)  # 158
WAVE_GOLD_ONE_LANE = WAVE_GOLD_EARLY
JUNGLE_CAMP_GOLD = 100.0  # not in headline leave hypothesis
KILL_BOUNTY_EARLY = 300.0  # early solo kill bounty baseline


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _fit_stable_logistic(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Fit a logistic calibration and reject non-finite results.

    Some macOS Accelerate builds emit spurious floating-point status warnings
    during otherwise finite BLAS products.  Inputs are explicitly finite and
    scaled; suppress only those status flags around the library fit, then make
    the actual invariant explicit rather than allowing a bad fit through.
    """
    clf = LogisticRegression(C=1e6, max_iter=4000, solver="lbfgs")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\..*")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            clf.fit(X, y.astype(int))
    if not np.isfinite(clf.intercept_).all() or not np.isfinite(clf.coef_).all():
        raise RuntimeError("Non-finite logistic calibration; inspect source inputs.")
    return clf


def fit_logit(
    x: np.ndarray, y: np.ndarray, abs_cap: float
) -> tuple[float, float, int]:
    """Univariate logistic; returns intercept, coef, n_fit."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y) & (np.abs(x) <= abs_cap)
    # Raw gold/XP units make the nearly-unpenalized optimizer ill-conditioned.
    # Fit in 1,000-unit increments, then return the coefficient per original
    # unit so every downstream scenario remains expressed in gold/XP.
    clf = _fit_stable_logistic((x[m] / MODEL_SCALE).reshape(-1, 1), y[m])
    return (
        float(clf.intercept_[0]),
        float(clf.coef_[0][0] / MODEL_SCALE),
        int(m.sum()),
    )


def fit_joint(
    g: np.ndarray, x: np.ndarray, y: np.ndarray, g_cap: float, x_cap: float
) -> tuple[float, float, float, int]:
    g = np.asarray(g, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    m = (
        np.isfinite(g)
        & np.isfinite(x)
        & np.isfinite(y)
        & (np.abs(g) <= g_cap)
        & (np.abs(x) <= x_cap)
    )
    Z = np.column_stack([g[m] / MODEL_SCALE, x[m] / MODEL_SCALE])
    clf = _fit_stable_logistic(Z, y[m])
    return (
        float(clf.intercept_[0]),
        float(clf.coef_[0][0] / MODEL_SCALE),
        float(clf.coef_[0][1] / MODEL_SCALE),
        int(m.sum()),
    )


def cross_validated_gold_diagnostics(
    x: np.ndarray,
    y: np.ndarray,
    *,
    abs_cap: float,
    folds: int = 10,
) -> dict[str, float]:
    """Deterministic out-of-fold diagnostics for the primary gold@10 logit."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y) & (np.abs(x) <= abs_cap)
    X = (x[m] / MODEL_SCALE).reshape(-1, 1)
    yy = y[m].astype(int)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=20260719)
    pred = np.empty(len(yy), dtype=float)
    for train, test in cv.split(X, yy):
        model = _fit_stable_logistic(X[train], yy[train])
        pred[test] = model.predict_proba(X[test])[:, 1]
    clipped = np.clip(pred, 1e-6, 1.0 - 1e-6)
    logit_pred = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    recal = _fit_stable_logistic(logit_pred, yy)
    prevalence = float(yy.mean())
    return {
        "folds": int(folds),
        "n": int(len(yy)),
        "prevalence": prevalence,
        "auc": float(roc_auc_score(yy, pred)),
        "brier": float(brier_score_loss(yy, pred)),
        "null_brier": float(
            brier_score_loss(yy, np.full(len(yy), prevalence, dtype=float))
        ),
        "log_loss": float(log_loss(yy, pred)),
        "calibration_intercept": float(recal.intercept_[0]),
        "calibration_slope": float(recal.coef_[0][0]),
        "kind": "10-fold stratified out-of-fold diagnostics; fixed seed 20260719",
    }


def delta_pp(intercept: float, coef: float, base: float, bump: float) -> float:
    p0 = _side_neutral_probability(intercept, coef, base)
    p1 = _side_neutral_probability(intercept, coef, base + bump)
    return (p1 - p0) * 100.0


def joint_delta_pp(
    intercept: float,
    c_g: float,
    c_x: float,
    g0: float,
    x0: float,
    dg: float,
    dx: float,
) -> float:
    eta0 = c_g * g0 + c_x * x0
    eta1 = c_g * (g0 + dg) + c_x * (x0 + dx)
    p0 = 0.5 * (_sigmoid(intercept + eta0) + _sigmoid(-intercept + eta0))
    p1 = 0.5 * (_sigmoid(intercept + eta1) + _sigmoid(-intercept + eta1))
    return (p1 - p0) * 100.0


def _side_neutral_probability(intercept: float, coef: float, gold: float) -> float:
    """Average blue- and red-side own-team probabilities at the same gold lead."""
    linear = coef * gold
    return 0.5 * (
        _sigmoid(intercept + linear) + _sigmoid(-intercept + linear)
    )


def _wald_covariance(X: np.ndarray, intercept: float, coef: np.ndarray) -> np.ndarray:
    """Observed-information covariance for an unpenalized logistic approximation.

    This is intentionally reported as *sampling* uncertainty only.  It does not
    turn the associational calibration, the mechanical conversion, or the contest
    priors into a causal estimate.
    """
    beta = np.concatenate([[float(intercept)], np.asarray(coef, dtype=float)])
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        z = X @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))
        w = p * (1.0 - p)
        info = X.T @ (w[:, None] * X)
    return np.linalg.pinv(info)


def univariate_delta_sampling_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    abs_cap: float,
    intercept: float,
    coef: float,
    base: float,
    bump: float,
) -> dict[str, float]:
    """Wald 95% CI for a logistic probability-difference calibration."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y) & (np.abs(x) <= abs_cap)
    xv = x[m] / MODEL_SCALE
    coef_scaled = coef * MODEL_SCALE
    base_scaled = base / MODEL_SCALE
    bump_scaled = bump / MODEL_SCALE
    X = np.column_stack([np.ones(len(xv)), xv])
    cov = _wald_covariance(X, intercept, np.asarray([coef_scaled]))
    def neutral_value_grad(x_scaled: float) -> tuple[float, np.ndarray]:
        p_blue = _sigmoid(intercept + coef_scaled * x_scaled)
        p_red = _sigmoid(-intercept + coef_scaled * x_scaled)
        value = 0.5 * (p_blue + p_red)
        grad = 0.5 * np.array(
            [
                p_blue * (1.0 - p_blue) - p_red * (1.0 - p_red),
                x_scaled
                * (p_blue * (1.0 - p_blue) + p_red * (1.0 - p_red)),
            ]
        )
        return value, grad

    p0, grad0 = neutral_value_grad(base_scaled)
    p1, grad1 = neutral_value_grad(base_scaled + bump_scaled)
    grad = 100.0 * (grad1 - grad0)
    se = float(np.sqrt(max(0.0, grad @ cov @ grad)))
    estimate = delta_pp(intercept, coef, base, bump)
    return {
        "estimate_pp": estimate,
        "se_pp": se,
        "ci95_low_pp": estimate - 1.96 * se,
        "ci95_high_pp": estimate + 1.96 * se,
        "n": int(m.sum()),
        "kind": "Wald sampling interval for the fitted association only",
    }


def joint_delta_sampling_ci(
    gold: np.ndarray,
    xp: np.ndarray,
    y: np.ndarray,
    *,
    gold_cap: float,
    xp_cap: float,
    intercept: float,
    coef_gold: float,
    coef_xp: float,
    gold_base: float,
    xp_base: float,
    gold_bump: float,
    xp_bump: float,
) -> dict[str, float]:
    """Wald 95% CI for a two-variable logistic probability difference."""
    gold = np.asarray(gold, dtype=float).reshape(-1)
    xp = np.asarray(xp, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    m = (
        np.isfinite(gold)
        & np.isfinite(xp)
        & np.isfinite(y)
        & (np.abs(gold) <= gold_cap)
        & (np.abs(xp) <= xp_cap)
    )
    X = np.column_stack(
        [np.ones(int(m.sum())), gold[m] / MODEL_SCALE, xp[m] / MODEL_SCALE]
    )
    coef_scaled = np.asarray([coef_gold, coef_xp]) * MODEL_SCALE
    gold_base_scaled = gold_base / MODEL_SCALE
    xp_base_scaled = xp_base / MODEL_SCALE
    gold_bump_scaled = gold_bump / MODEL_SCALE
    xp_bump_scaled = xp_bump / MODEL_SCALE
    cov = _wald_covariance(X, intercept, coef_scaled)
    def neutral_value_grad(g_scaled: float, x_scaled: float) -> tuple[float, np.ndarray]:
        linear = coef_scaled[0] * g_scaled + coef_scaled[1] * x_scaled
        p_blue = _sigmoid(intercept + linear)
        p_red = _sigmoid(-intercept + linear)
        value = 0.5 * (p_blue + p_red)
        common = p_blue * (1.0 - p_blue) + p_red * (1.0 - p_red)
        grad = 0.5 * np.array(
            [
                p_blue * (1.0 - p_blue) - p_red * (1.0 - p_red),
                g_scaled * common,
                x_scaled * common,
            ]
        )
        return value, grad

    p0, grad0 = neutral_value_grad(gold_base_scaled, xp_base_scaled)
    p1, grad1 = neutral_value_grad(
        gold_base_scaled + gold_bump_scaled,
        xp_base_scaled + xp_bump_scaled,
    )
    grad = 100.0 * (grad1 - grad0)
    se = float(np.sqrt(max(0.0, grad @ cov @ grad)))
    estimate = joint_delta_pp(
        intercept, coef_gold, coef_xp, gold_base, xp_base, gold_bump, xp_bump
    )
    return {
        "estimate_pp": estimate,
        "se_pp": se,
        "ci95_low_pp": estimate - 1.96 * se,
        "ci95_high_pp": estimate + 1.96 * se,
        "n": int(m.sum()),
        "kind": "Wald sampling interval for the fitted association only",
    }


def totv_damage_per_cycle(tick: float) -> float:
    return tick * TOTV_TICKS_PER_CYCLE


def burn_gold_equivalent(
    *,
    tick_melee: float,
    siege_seconds: float,
    plate_gold: float = PLATE_GOLD,
    plate_hp: float = PLATE_HP_FIRST,
    hunger_mite_refreshes: int = 0,
) -> dict[str, float]:
    """
    Touch burn → *undiscounted plate-progress equivalent* using Turret Plating.

    Default plate band = first outer plate (900 HP / 120g). Hunger mites that
    apply/refresh Touch add extra full burn cycles (upper sensitivity).  This
    quantity is not paid gold: it becomes gold only if the team later converts
    the additional structure damage into a plate before plates expire.
    """
    dps = tick_melee / TOTV_TICK_INTERVAL
    total_dmg = dps * siege_seconds
    # Each mite refresh ≈ one extra AA-applied burn cycle while sieging
    total_dmg += hunger_mite_refreshes * totv_damage_per_cycle(tick_melee)
    plates = total_dmg / plate_hp
    gold = plates * plate_gold
    return {
        "tick_melee": tick_melee,
        "dps_true": dps,
        "siege_seconds": siege_seconds,
        "hunger_mite_refreshes": hunger_mite_refreshes,
        "total_true_damage": total_dmg,
        "plates_equivalent": plates,
        "undiscounted_plate_progress_g": gold,
        # Kept for backward compatibility with older JSON consumers.  Never
        # describe this as guaranteed or immediately received gold.
        "gold_equivalent": gold,
        "plate_gold": plate_gold,
        "plate_hp": plate_hp,
        "gold_per_true_damage": plate_gold / plate_hp,
        "source": (
            "Undiscounted first-plate progress ceiling: 120g/plate; outer 9000 HP; "
            "first plate at 10% missing = 900 HP (Patch 26.1). Not paid gold unless converted."
        ),
    }


def worked_siege_example() -> dict[str, Any]:
    """Discrete mechanics check for level-15 Zaahen sieging with Touch.

    Champion state is level 15 wiki base stats: base AD 116.06, base AS 0.625,
    bonus AS from levels 33.16%. Items (SR flats): Doran's Blade (+10 AD),
    Trinity Force (+36 AD, +30% AS, Spellblade 200% base AD vs structures),
    Hexdrinker (+25 AD), Mercury's Treads, Sundered Sky (+45 AD), and
    Caulfield's Warhammer (+20 AD). Structure DPS uses total AD autos +
    Trinity Spellblade only. Excluded vs structures: Sundered Sky Lightshield
    Strike, Hexdrinker Lifeline, Cultivation of War / Determination (requires
    champion damage), The Darkin Glaive double-strike / AA resets, Hunger
    mites, minions. Touch: basic attacks apply a 4s burn ticking every 0.5s;
    refreshes maintain one burn rather than stacking full packets.
    """
    champion = "Zaahen"
    level = 15
    items = [
        "Doran's Blade",
        "Trinity Force",
        "Hexdrinker",
        "Mercury's Treads",
        "Sundered Sky",
        "Caulfield's Warhammer",
    ]
    item_ad = {
        "Doran's Blade": 10.0,
        "Trinity Force": 36.0,
        "Hexdrinker": 25.0,
        "Mercury's Treads": 0.0,
        "Sundered Sky": 45.0,
        "Caulfield's Warhammer": 20.0,
    }
    structure_hp = 5000.0
    structure_armor = 60.0
    # Wiki level-15 base panel (no items / no Determination).
    base_ad = 116.06
    base_attack_speed = 0.625
    level_bonus_as = 0.3316
    trinity_bonus_as = 0.30
    total_ad = base_ad + sum(item_ad.values())
    attack_speed = base_attack_speed * (1.0 + level_bonus_as + trinity_bonus_as)
    spellblade_ratio = 2.0
    spellblade_cooldown = 1.5

    armor_multiplier = 100.0 / (100.0 + structure_armor)
    attack_period = 1.0 / attack_speed
    normal_attack_damage = total_ad * armor_multiplier
    spellblade_bonus_damage = spellblade_ratio * base_ad * armor_multiplier
    proc_every_attacks = int(math.ceil(spellblade_cooldown / attack_period))

    rows: list[dict[str, float | int | str]] = []
    for stacks, tick_damage in enumerate(TOTV_TICK_MELEE_BY_STACK_POST_26_11):
        hp = structure_hp
        attacks = 0
        next_attack = 0.0
        next_tick = TOTV_TICK_INTERVAL if tick_damage else math.inf
        active_until = -math.inf
        kill_time = math.nan
        kill_event = ""
        touch_true_damage = 0.0
        zaahen_physical_damage = 0.0

        while hp > 0 and attacks < 1000:
            if next_attack <= next_tick + 1e-12:
                t = next_attack
                attacks += 1
                proc = (attacks - 1) % proc_every_attacks == 0
                raw = normal_attack_damage + (spellblade_bonus_damage if proc else 0.0)
                applied = min(raw, hp)
                hp -= applied
                zaahen_physical_damage += applied
                if tick_damage:
                    active_until = t + TOTV_BURN_DURATION
                next_attack = attacks * attack_period
                if hp <= 0:
                    kill_time = t
                    kill_event = "attack"
                    break
            else:
                t = next_tick
                if t <= active_until + 1e-12:
                    applied = min(float(tick_damage), hp)
                    hp -= applied
                    touch_true_damage += applied
                next_tick += TOTV_TICK_INTERVAL
                if hp <= 0:
                    kill_time = t
                    kill_event = "Touch tick"
                    break

        if not math.isfinite(kill_time):
            raise RuntimeError("Worked siege example failed to terminate.")
        # Without Touch, Zaahen alone must cover the full structure HP.
        zaahen_without_touch = structure_hp
        rows.append(
            {
                "stacks": stacks,
                "tick_damage": int(tick_damage),
                "maintained_true_dps": float(tick_damage / TOTV_TICK_INTERVAL),
                "destruction_time_s": float(kill_time),
                "attacks": int(attacks),
                "kill_event": kill_event,
                "touch_true_damage": float(touch_true_damage),
                "zaahen_physical_damage": float(zaahen_physical_damage),
                "zaahen_without_touch_damage": float(zaahen_without_touch),
                "touch_vs_zaahen_without_touch_pct": (
                    100.0 * float(touch_true_damage) / zaahen_without_touch
                ),
            }
        )

    baseline_time = float(rows[0]["destruction_time_s"])
    baseline_attacks = int(rows[0]["attacks"])
    for row in rows:
        row["time_saved_s"] = baseline_time - float(row["destruction_time_s"])
        row["attacks_saved"] = baseline_attacks - int(row["attacks"])
        row["time_reduction_pct"] = (
            100.0 * float(row["time_saved_s"]) / baseline_time
        )

    three = rows[3]
    return {
        "champion": champion,
        "level": level,
        "items": items,
        "item_ad": item_ad,
        "structure_hp": structure_hp,
        "structure_armor": structure_armor,
        "total_ad": total_ad,
        "base_ad": base_ad,
        "base_attack_speed": base_attack_speed,
        "level_bonus_as": level_bonus_as,
        "trinity_bonus_as": trinity_bonus_as,
        "attack_speed": attack_speed,
        "attack_period_s": attack_period,
        "armor_multiplier": armor_multiplier,
        "spellblade_base_ad_ratio": spellblade_ratio,
        "spellblade_cooldown_s": spellblade_cooldown,
        "spellblade_proc_every_attacks": proc_every_attacks,
        "normal_attack_damage": normal_attack_damage,
        "spellblade_bonus_damage": spellblade_bonus_damage,
        "rows": rows,
        "three_stack_time_s": float(three["destruction_time_s"]),
        "three_stack_attacks": int(three["attacks"]),
        "zero_stack_time_s": baseline_time,
        "zero_stack_attacks": baseline_attacks,
        "three_stack_time_saved_s": float(three["time_saved_s"]),
        "three_stack_attacks_saved": int(three["attacks_saved"]),
        "three_stack_time_reduction_pct": float(three["time_reduction_pct"]),
        "structure_damage_included": [
            "total AD autoattacks",
            "Trinity Force Spellblade (200% level-15 base AD; structures)",
            "Touch of the Void true burn (refresh, not stacked packets)",
        ],
        "structure_damage_excluded": [
            "Sundered Sky Lightshield Strike (champions only)",
            "Hexdrinker Lifeline (shield)",
            "Cultivation of War / Determination (champion damage)",
            "The Darkin Glaive double-strike and AA resets",
            "Hunger of the Void Voidmites",
        ],
        "touch_rules": {
            "source": "https://wiki.leagueoflegends.com/en-us/Touch_of_the_Void",
            "apply": "basic attacks against structures",
            "duration_s": TOTV_BURN_DURATION,
            "tick_interval_s": TOTV_TICK_INTERVAL,
            "melee_tick_by_stack_post_26_11": list(TOTV_TICK_MELEE_BY_STACK_POST_26_11),
            "ranged_tick_by_stack_post_26_11": list(TOTV_TICK_RANGED_BY_STACK_POST_26_11),
            "refresh": (
                "subsequent structure attacks refresh duration; "
                "do not stack overlapping full burns"
            ),
            "trigger_gate": (
                "applies only if the triggering instance can damage the structure"
            ),
        },
        "hunger_rules": {
            "source": "https://wiki.leagueoflegends.com/en-us/Hunger_of_the_Void",
            "granted_at_touch_stacks": HUNGER_TOUCH_STACKS_REQUIRED,
            "summon_cooldown_s": HUNGER_MITE_CD,
            "mites_apply_touch": True,
            "omitted_from_this_clock": True,
        },
        "scope": (
            "Level-15 Zaahen deterministic mechanics check only; excludes "
            "Bulwark, backdoor protection, regeneration, minions, allies, "
            "Hunger mites, Determination, Q double-strikes, latency, and "
            "animation timing."
        ),
    }



def xp_local_share_table() -> dict[str, Any]:
    """Wiki: 65 XP/grub, radius 2000, split equally among alive allies in range."""
    rows = []
    for n in (1, 2, 3):
        each = PACKAGE_XP / n
        rows.append(
            {
                "allies_in_radius": n,
                "xp_each": each,
                "team_xpdiff_sum": PACKAGE_XP,  # OE team xpdiff is a sum
                "levels_each_toward_7_to_8": each / XP_TO_LEVEL_7,
                "note": (
                    "Solo jungler soak"
                    if n == 1
                    else ("Duo river" if n == 2 else "Trio contest")
                ),
            }
        )
    return {
        "xp_per_grub": XP_PER_GRUB,
        "package_xp": PACKAGE_XP,
        "radius": XP_RADIUS,
        "xp_to_level_7_to_8": XP_TO_LEVEL_7,
        "package_as_fraction_of_level": PACKAGE_XP / XP_TO_LEVEL_7,
        "shares": rows,
        "notes": (
            "Epic XP is local and shared; team xpdiff@10 still rises by the full "
            "bounty sum. Concentration (solo jg vs trio) changes who spikes a "
            "level, not the OE team-sum accounting."
        ),
    }


def ls_burn_gold_equivalent() -> dict[str, Any]:
    """LS sandbox language → gold: 1 AA/cycle early; 1.5 AA saved mid on full turret."""
    early = LS_AA_PER_CYCLE * AA_DMG_EARLY  # damage per 4s cycle
    # Continuous early siege for 20s ≈ 5 cycles (LS-style upper human siege)
    early_20s = early * (20.0 / TOTV_BURN_DURATION)
    mid_save = LS_MID_AA_SAVED_FULL_TURRET * AA_DMG_MID
    g_per_dmg = PLATE_GOLD / PLATE_HP_FIRST
    return {
        "early_dmg_per_cycle": early,
        "early_dmg_20s_siege": early_20s,
        "early_gold_20s": early_20s * g_per_dmg,
        "mid_dmg_saved_full_turret": mid_save,
        "mid_gold_saved_full_turret": mid_save * g_per_dmg,
        "assumptions": {
            "aa_dmg_early": AA_DMG_EARLY,
            "aa_dmg_mid": AA_DMG_MID,
            "aa_per_cycle": LS_AA_PER_CYCLE,
            "mid_aa_saved_full_turret": LS_MID_AA_SAVED_FULL_TURRET,
            "plate_gold": PLATE_GOLD,
            "plate_hp_first_outer": PLATE_HP_FIRST,
            "outer_turret_hp": OUTER_TURRET_HP,
        },
    }


def _pp(x: float) -> float:
    return round(float(x), 2)


def river_outcome_matrix(
    scrap_pp: float,
    farm_pp: float,
    kill_win_pp: float,
    kill_lose_pp: float,
) -> dict[str, Any]:
    """
    Prisoner's-dilemma-style 2x2 of terminal river outcomes (map WR pp vs even).

    Rows: fight result for your team.
    Columns: whether YOUR team secures the camp (not the enemy).
    Leave (no fight) is the outside option: gift scrap, keep farm.
    """
    k_lose = abs(float(kill_lose_pp))
    scrap = float(scrap_pp)
    farm = float(farm_pp)
    k_win = float(kill_win_pp)

    leave = farm - scrap  # gift camp, optional farm
    # Lose TF, you do NOT get grubs (enemy takes / you gift): deaths + gift scrap
    lose_no_grubs = -(k_lose + scrap)
    # Lose TF, you still GET grubs (rare: secure then lose, or steal after deaths)
    lose_but_grubs = -k_lose + scrap
    # Win TF, but the opponent secures the camp (for example, a smite loss).
    win_no_grubs = k_win - scrap
    # Win TF and secure camp (usual contest success)
    win_and_grubs = k_win + scrap

    leave_label = (
        "Leave river (no TF): gift camp, farm=0"
        if abs(farm) < 1e-9
        else "Leave river (no TF): farm waves/jungle, gift camp"
    )
    return {
        "units": "map WR pp vs even gold@10",
        "outside_option_leave": {
            "label": leave_label,
            "pp": _pp(leave),
            "components": {"farm_pp": _pp(farm), "gift_scrap_pp": _pp(-scrap)},
        },
        "matrix": {
            "lose_tf_no_grubs_pp": _pp(lose_no_grubs),
            "lose_tf_but_you_get_grubs_pp": _pp(lose_but_grubs),
            "win_tf_no_grubs_pp": _pp(win_no_grubs),
            "win_tf_and_grubs_pp": _pp(win_and_grubs),
        },
        "vs_leave": {
            "lose_tf_no_grubs_pp": _pp(lose_no_grubs - leave),
            "lose_tf_but_you_get_grubs_pp": _pp(lose_but_grubs - leave),
            "win_tf_no_grubs_pp": _pp(win_no_grubs - leave),
            "win_tf_and_grubs_pp": _pp(win_and_grubs - leave),
        },
        "notes": (
            "Columns are always about YOUR camp secure, not the enemy's. "
            "Lose+no grubs = usual collapse (deaths + gift). "
            "Lose+grubs = rare (you secure then lose, or steal after deaths). "
            "Win+no grubs = smite loss. Win+grubs = usual contest success."
        ),
    }


def river_outcome_terminal_matrix(
    intercept: float,
    coef: float,
    *,
    objective_gold: float,
    leave_farm_gold: float,
    leave_plate_probability: float = 0.0,
    plate_gold: float = PLATE_GOLD,
    win_kill_gold: float = 600.0,
    loss_kill_gold: float = -600.0,
) -> dict[str, Any]:
    """Contingent river decision matrix from complete terminal gold states.

    This is deliberately not a formal Prisoner's Dilemma: there is no
    simultaneous two-player normal form or dominant-strategy claim.  It is a
    payoff-board presentation of the fight result × camp-secure contingencies,
    measured against an explicit no-fight outside option.
    """
    def q(gold: float) -> float:
        return _side_neutral_probability(intercept, coef, gold)

    p_plate = float(np.clip(leave_plate_probability, 0.0, 1.0))
    q_even = q(0.0)
    q_leave = (
        (1.0 - p_plate) * q(leave_farm_gold - objective_gold)
        + p_plate * q(leave_farm_gold + plate_gold - objective_gold)
    )
    states = {
        "lose_tf_no_grubs": -objective_gold + loss_kill_gold,
        "lose_tf_but_you_get_grubs": objective_gold + loss_kill_gold,
        "win_tf_no_grubs": -objective_gold + win_kill_gold,
        "win_tf_and_grubs": objective_gold + win_kill_gold,
    }
    probs = {name: q(gold) for name, gold in states.items()}
    matrix = {f"{name}_pp": _pp((prob - q_even) * 100.0) for name, prob in probs.items()}
    leave_pp = _pp((q_leave - q_even) * 100.0)
    return {
        "units": "map WR pp versus zero team gold differential at 10 minutes",
        "outside_option_leave": {
            "label": "Leave river: farm, concede grubs, avoid the fight",
            "pp": leave_pp,
            "terminal_gold": {
                "no_plate": leave_farm_gold - objective_gold,
                "with_plate": leave_farm_gold + plate_gold - objective_gold,
                "plate_probability": p_plate,
            },
        },
        "matrix": matrix,
        "vs_leave": {
            key: _pp(value - leave_pp) for key, value in matrix.items()
        },
        "terminal_gold_states": states,
        "notes": (
            "Every cell is evaluated as a complete terminal gold state before "
            "the fitted gold logit is applied. Rows are fight outcomes; columns "
            "are whether your team secured the camp."
        ),
    }


def contest_ev_table(
    delta_obj_pp: float,
    fight_loss_pp: float,
    fight_win_extra_pp: float = 0.0,
    farm_pp: float = 0.0,
) -> tuple[list[dict[str, Any]], Optional[float]]:
    """
    Diagonal lottery (secure if fight win): leave gifts scrap (+ optional farm);
    contest win = scrap + kill-win; contest lose = deaths + gift.
    fight_loss_pp should already include Δobj.
    """
    leave = -delta_obj_pp + farm_pp
    rows = []
    for p in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.75]:
        ev = p * (delta_obj_pp + fight_win_extra_pp) + (1.0 - p) * (-fight_loss_pp)
        edge = ev - leave
        rows.append(
            {
                "p_win_fight": p,
                "ev_contest_pp": _pp(ev),
                "ev_leave_pp": _pp(leave),
                "edge_contest_minus_leave_pp": _pp(edge),
                "verdict": (
                    "CONTEST"
                    if edge > 0.05
                    else ("AVOID" if edge < -0.05 else "TOSS-UP")
                ),
            }
        )
    denom = delta_obj_pp + fight_win_extra_pp + fight_loss_pp
    numer = fight_loss_pp - delta_obj_pp + farm_pp
    be = numer / denom if denom > 1e-12 else None
    return rows, (None if be is None else float(be))


def contest_ev_terminal_states(
    intercept: float,
    coef: float,
    *,
    baseline_gold: float = 0.0,
    objective_gold: float,
    leave_farm_gold: float,
    leave_plate_probability: float = 0.0,
    plate_gold: float = PLATE_GOLD,
    win_kill_gold: float = 600.0,
    loss_kill_gold: float = -600.0,
    p_secure_if_win: float = 1.0,
    p_secure_if_lose: float = 0.0,
) -> tuple[list[dict[str, Any]], Optional[float], dict[str, float]]:
    """Evaluate all four fight-result x camp-secure terminal states.

    ``p_secure_if_win`` and ``p_secure_if_lose`` determine how the four cells
    are mixed conditional on the fight result.  In this exhaustive two-team
    state model, failure by the focal team to secure means the opponent secures
    the camp.  The historical diagonal model is the explicit reference branch
    (1, 0), not an implicit assumption.
    """
    def prob(gold: float) -> float:
        return _side_neutral_probability(intercept, coef, gold)

    p_plate = float(np.clip(leave_plate_probability, 0.0, 1.0))
    baseline = float(baseline_gold)
    p_leave = (1.0 - p_plate) * prob(baseline + leave_farm_gold - objective_gold)
    p_leave += p_plate * prob(
        baseline + leave_farm_gold + plate_gold - objective_gold
    )
    s_win = float(np.clip(p_secure_if_win, 0.0, 1.0))
    s_lose = float(np.clip(p_secure_if_lose, 0.0, 1.0))
    p_win_secure = prob(baseline + objective_gold + win_kill_gold)
    p_win_no_secure = prob(baseline - objective_gold + win_kill_gold)
    p_lose_secure = prob(baseline + objective_gold + loss_kill_gold)
    p_lose_opponent_secure = prob(baseline - objective_gold + loss_kill_gold)
    p_win = s_win * p_win_secure + (1.0 - s_win) * p_win_no_secure
    p_loss = s_lose * p_lose_secure + (1.0 - s_lose) * p_lose_opponent_secure
    p_even = prob(baseline)
    rows = []
    for p_fight in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.75]:
        p_contest = p_fight * p_win + (1.0 - p_fight) * p_loss
        edge = (p_contest - p_leave) * 100.0
        rows.append({
            "p_win_fight": p_fight,
            "ev_contest_pp": _pp((p_contest - p_even) * 100.0),
            "ev_leave_pp": _pp((p_leave - p_even) * 100.0),
            "edge_contest_minus_leave_pp": _pp(edge),
            "verdict": "CONTEST" if edge > 0.05 else ("AVOID" if edge < -0.05 else "TOSS-UP"),
        })
    denom = p_win - p_loss
    breakeven = None
    if denom > 1e-12:
        algebraic_root = (p_leave - p_loss) / denom
        if 0.0 <= algebraic_root <= 1.0:
            breakeven = algebraic_root
    return rows, breakeven, {
        "leave_probability": p_leave,
        "contest_win_probability": p_win,
        "contest_loss_probability": p_loss,
        "win_secure_probability": p_win_secure,
        "win_no_secure_probability": p_win_no_secure,
        "lose_secure_probability": p_lose_secure,
        "lose_opponent_secure_probability": p_lose_opponent_secure,
        # Alias kept for one regen cycle so old readers do not KeyError mid-migration.
        "lose_concede_probability": p_lose_opponent_secure,
        "p_secure_if_win": s_win,
        "p_secure_if_lose": s_lose,
        "baseline_gold": baseline,
        "objective_gold": objective_gold,
        "leave_farm_gold": leave_farm_gold,
        "leave_plate_probability": p_plate,
        "win_kill_gold": win_kill_gold,
        "loss_kill_gold": loss_kill_gold,
    }


def contest_pstar_sampling_ci_from_covariance(
    intercept: float,
    coef: float,
    covariance_scaled: np.ndarray,
    *,
    baseline_gold: float = 0.0,
    objective_gold: float,
    leave_farm_gold: float,
    win_kill_gold: float,
    loss_kill_gold: float,
    p_secure_if_win: float,
    p_secure_if_lose: float,
) -> dict[str, float]:
    """Delta-method interval for p* under the map-independent fit covariance."""

    def evaluate(theta: np.ndarray) -> float:
        _, pstar, _ = contest_ev_terminal_states(
            float(theta[0]),
            float(theta[1]) / MODEL_SCALE,
            baseline_gold=baseline_gold,
            objective_gold=objective_gold,
            leave_farm_gold=leave_farm_gold,
            win_kill_gold=win_kill_gold,
            loss_kill_gold=loss_kill_gold,
            p_secure_if_win=p_secure_if_win,
            p_secure_if_lose=p_secure_if_lose,
        )
        if pstar is None:
            raise RuntimeError("Undefined p* while propagating calibration uncertainty")
        return float(pstar)

    theta = np.asarray([intercept, coef * MODEL_SCALE], dtype=float)
    estimate = evaluate(theta)
    grad = np.zeros(2, dtype=float)
    for j in range(2):
        step = 1e-5 * max(1.0, abs(float(theta[j])))
        plus = theta.copy()
        minus = theta.copy()
        plus[j] += step
        minus[j] -= step
        grad[j] = (evaluate(plus) - evaluate(minus)) / (2.0 * step)
    cov = np.asarray(covariance_scaled, dtype=float)
    se = float(np.sqrt(max(0.0, grad @ cov @ grad)))
    return {
        "estimate": estimate,
        "se": se,
        "ci95_low": estimate - 1.96 * se,
        "ci95_high": estimate + 1.96 * se,
        "kind": (
            "Delta-method interval using the map-level independence working covariance; "
            "does not include clustering or scenario-parameter uncertainty"
        ),
    }


def contest_certainty_atlas(
    intercept: float,
    coef: float,
    *,
    touch_gold: float,
    covariance_scaled: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Break-even confidence study over rewards, leave states, and all four capture corners."""
    packages = [
        ("cash_only", "90g cash only", PACKAGE_GOLD),
        ("cash_plus_touch", "90g cash + brief Touch ceiling", PACKAGE_GOLD + touch_gold),
    ]
    leaves = [
        ("concede_only", "No farm recovered", 0.0),
        ("one_wave", "One grub-era average wave", WAVE_GOLD_EARLY),
        ("two_waves", "Two grub-era average waves", 2.0 * WAVE_GOLD_EARLY),
        ("two_waves_plate", "Two waves + one outer plate", 2.0 * WAVE_GOLD_EARLY + PLATE_GOLD),
        ("three_waves_plate", "Three waves + one outer plate", 3.0 * WAVE_GOLD_EARLY + PLATE_GOLD),
    ]
    capture_branches = [
        ("secure_if_win", "Secure if fight won", 1.0, 0.0),
        ("always_secure", "Secure after either fight result", 1.0, 1.0),
        ("never_secure", "Do not secure after either result", 0.0, 0.0),
        ("secure_if_lose", "Secure if fight lost", 0.0, 1.0),
    ]
    package_rows: dict[str, Any] = {}
    for key, label, objective_gold in packages:
        branches = []
        for branch_key, branch_label, s_win, s_lose in capture_branches:
            cells = []
            for leave_key, leave_label, leave_gold in leaves:
                _, pstar, terminal = contest_ev_terminal_states(
                    intercept,
                    coef,
                    objective_gold=objective_gold,
                    leave_farm_gold=leave_gold,
                    win_kill_gold=600.0,
                    loss_kill_gold=-600.0,
                    p_secure_if_win=s_win,
                    p_secure_if_lose=s_lose,
                )
                interval = None
                if covariance_scaled is not None:
                    interval = contest_pstar_sampling_ci_from_covariance(
                        intercept,
                        coef,
                        np.asarray(covariance_scaled, dtype=float),
                        objective_gold=objective_gold,
                        leave_farm_gold=leave_gold,
                        win_kill_gold=600.0,
                        loss_kill_gold=-600.0,
                        p_secure_if_win=s_win,
                        p_secure_if_lose=s_lose,
                    )
                cells.append(
                    {
                        "key": leave_key,
                        "label": leave_label,
                        "leave_gold": round(leave_gold, 2),
                        "breakeven_p_win_fight": pstar,
                        "map_level_sampling_interval": interval,
                        "terminal_states": terminal,
                    }
                )
            branches.append(
                {
                    "key": branch_key,
                    "label": branch_label,
                    "p_secure_if_win": s_win,
                    "p_secure_if_lose": s_lose,
                    "cells": cells,
                }
            )
        # Backward-compatible alias: the reference diagonal branch.
        cells = branches[0]["cells"]
        package_rows[key] = {
            "label": label,
            "objective_gold": round(objective_gold, 2),
            "cells": cells,
            "capture_branches": branches,
        }
    return {
        "fight_exchange": "contest win +600g / contest loss -600g (symmetric two-kill sensitivity)",
        "packages": package_rows,
        "leave_states": [
            {"key": key, "label": label, "leave_gold": round(gold, 2)}
            for key, label, gold in leaves
        ],
        "notes": (
            "For each state and capture branch, p* is the fight-win probability at "
            "which the four-cell contest mixture equals the explicit leave state. "
            "The four deterministic capture corners bound the continuous "
            "(P(secure|win), P(secure|lose)) unit square."
        ),
    }


def contest_ev_mixture_table(
    scrap_pp: float,
    kill_win_pp: float,
    kill_lose_pp: float,
    farm_pp: float = 0.0,
    *,
    p_secure_if_win: float = 0.85,
    p_secure_if_lose: float = 0.05,
) -> tuple[list[dict[str, Any]], Optional[float], dict[str, float]]:
    """
    Contest EV averaging over the 2×2 (fight × camp secure), not only the diagonal.
    Defaults: occasional smite loss on won fights; rare secure-then-collapse.
    """
    scrap = float(scrap_pp)
    k_win = float(kill_win_pp)
    k_lose = abs(float(kill_lose_pp))
    farm = float(farm_pp)
    lose_no = -(k_lose + scrap)
    lose_yes = -k_lose + scrap
    win_no = k_win - scrap
    win_yes = k_win + scrap
    leave = -scrap + farm
    e_win = p_secure_if_win * win_yes + (1.0 - p_secure_if_win) * win_no
    e_lose = p_secure_if_lose * lose_yes + (1.0 - p_secure_if_lose) * lose_no
    cells = {
        "lose_tf_no_grubs_pp": lose_no,
        "lose_tf_but_you_get_grubs_pp": lose_yes,
        "win_tf_no_grubs_pp": win_no,
        "win_tf_and_grubs_pp": win_yes,
        "e_payoff_if_win_pp": e_win,
        "e_payoff_if_lose_pp": e_lose,
        "p_secure_if_win": p_secure_if_win,
        "p_secure_if_lose": p_secure_if_lose,
    }
    rows = []
    for p in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.75]:
        ev = p * e_win + (1.0 - p) * e_lose
        edge = ev - leave
        rows.append(
            {
                "p_win_fight": p,
                "ev_contest_pp": _pp(ev),
                "ev_leave_pp": _pp(leave),
                "edge_contest_minus_leave_pp": _pp(edge),
                "verdict": (
                    "CONTEST"
                    if edge > 0.05
                    else ("AVOID" if edge < -0.05 else "TOSS-UP")
                ),
            }
        )
    denom = e_win - e_lose
    be = (leave - e_lose) / denom if abs(denom) > 1e-12 else None
    return rows, (None if be is None else float(be)), cells


# ---------------------------------------------------------------------------
# Leave-window farm (core leave hypothesis) — wiki gold → gold@10 WR logit
# ---------------------------------------------------------------------------
# Hypothesis: while gifting grubs, N laners (1–3) each clear one grub-era
# expected wave (pre-14 composition at the 10:00 reference clock), optionally
# with P(take an outer plate at 120g). Wave/plate constants are defined above.

# Early component basket (wiki item costs) — median next spike around grub timing.
# Sources: Pickaxe / Recurve Bow / Caulfield's / B.F. Sword wiki pages.
EARLY_COMPONENT_COSTS = {
    "recurve_bow": 700.0,
    "pickaxe": 875.0,
    "caulfields_warhammer": 1050.0,
    "bf_sword": 1300.0,
}
# Modal early AD spike (Pickaxe); also report basket median.
MEDIAN_NEXT_COMPONENT_GOLD = float(np.median(list(EARLY_COMPONENT_COSTS.values())))
MODAL_NEXT_COMPONENT_GOLD = EARLY_COMPONENT_COSTS["pickaxe"]
WAVE_PERIOD_S = WAVE_PERIOD_S_PRE14  # minion wave cadence before 14:00
LANER_GPM_EARLY = WAVE_GOLD_AVG_EARLY * (60.0 / WAVE_PERIOD_S)
CONTEST_DELAY_MIN = 1.25  # ~75s river trip / clear / walk-back out of lane


def item_completion_pace(
    b0: float,
    b1: float,
    *,
    farm_gold: float,
    scrap_gold: float = PACKAGE_GOLD,
    n_laners: int = 2,
    opp_miss_fraction: float = 1.0,
    net_kills_win: int = 2,
    net_kills_lose: int = -2,
    delay_min: float = CONTEST_DELAY_MIN,
    horizons_min: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> dict[str, Any]:
    """
    Item-completion / gold-pace layer for leave vs take and fight win vs lose.

    Leave team farms waves; take/contest team spends the window on river and
    misses that CS. Relative golddiff under gift = own farm + opp missed farm
    − scrap. Map that into fractions of the next early component and minutes
    of laner GPM, then project the gap over the next 1/2/3 minutes while the
    river side is still delayed.

    Headline leave EV still counts only own farm (contest forgoes the same F);
    dual-tempo (own + opp miss) is reported here and as an EV sensitivity.
    """
    costs = dict(EARLY_COMPONENT_COSTS)
    median_c = MEDIAN_NEXT_COMPONENT_GOLD
    modal_c = MODAL_NEXT_COMPONENT_GOLD
    gpm_laner = LANER_GPM_EARLY
    gpm_team = n_laners * gpm_laner
    f_own = float(farm_gold)
    f_opp = float(farm_gold) * float(opp_miss_fraction)  # same window, N on river
    scrap_g = float(scrap_gold)

    # Gift path (no TF): leave farms, take spends window on camp.
    leave_g = f_own
    take_g = scrap_g - f_opp
    rel_leave_vs_take = leave_g - take_g  # = F + F_opp − scrap

    def _item_view(gold: float) -> dict[str, Any]:
        return {
            "gold": round(gold, 2),
            "frac_modal_pickaxe_875": round(gold / modal_c, 3),
            "frac_median_component": round(gold / median_c, 3),
            "minutes_solo_laner_gpm": round(gold / gpm_laner, 3),
            "minutes_team_gpm": round(gold / gpm_team, 3) if gpm_team > 0 else None,
            "wr_pp_at_even": _pp(delta_pp(b0, b1, 0.0, gold)),
        }

    # Fight path: both miss lane farm → farm cancels; kills + scrap drive gap.
    kill_win_g = net_kills_win * KILL_BOUNTY_EARLY
    kill_lose_g = net_kills_lose * KILL_BOUNTY_EARLY  # negative
    win_net = kill_win_g + scrap_g  # farm cancel
    lose_net = kill_lose_g - scrap_g  # deaths + gift
    win_minus_lose = win_net - lose_net  # = |2*kills|*300 + 2*scrap

    # Horizons: while take/contest is delayed, leave keeps full team GPM.
    horizon_rows = []
    for h in horizons_min:
        extra = gpm_team * min(float(h), float(delay_min))
        # After delay_min both farm again → gap stops growing from CS alone.
        gap = rel_leave_vs_take + extra
        horizon_rows.append(
            {
                "horizon_min": h,
                "extra_farm_while_opp_delayed_g": round(extra, 2),
                "relative_gold_leave_minus_take": round(gap, 2),
                **{k: v for k, v in _item_view(gap).items() if k != "gold"},
                "gold": round(gap, 2),
            }
        )

    dual_farm_g = f_own + f_opp  # leave golddiff attribution if opp miss counted
    return {
        "wiki_sources": [
            "https://wiki.leagueoflegends.com/en-us/Pickaxe",
            "https://wiki.leagueoflegends.com/en-us/Recurve_Bow",
            "https://wiki.leagueoflegends.com/en-us/Caulfield%27s_Warhammer",
            "https://wiki.leagueoflegends.com/en-us/B._F._Sword",
            "https://wiki.leagueoflegends.com/en-us/Minion",
        ],
        "hypothesis": (
            "Contest/take burns lane tempo: missed waves slow next-item completion "
            "for 1–3 minutes. Leave farm gold is not only a one-shot WR bump — it is "
            "item-spike pace. Relative leave vs take = own farm + opp missed farm − scrap."
        ),
        "constants": {
            "early_component_costs_g": costs,
            "median_next_component_g": median_c,
            "modal_next_component_g": modal_c,
            "modal_item": "pickaxe",
            "wave_period_s": WAVE_PERIOD_S,
            "laner_gpm_early": round(gpm_laner, 4),
            "contest_delay_min_default": delay_min,
            "n_laners": n_laners,
            "opp_miss_fraction": opp_miss_fraction,
            "scrap_gold": scrap_g,
            "farm_gold_own": round(f_own, 4),
            "farm_gold_opp_missed": round(f_opp, 4),
        },
        "gift_path_no_tf": {
            "leave_team_gold": _item_view(leave_g),
            "take_team_gold": _item_view(take_g),
            "relative_leave_minus_take": _item_view(rel_leave_vs_take),
            "note": (
                "Take gets scrap but misses F_opp waves. Leave gets F and does not "
                "pay river time. Relative gap ≫ own-farm-only leave EV gold."
            ),
        },
        "fight_path_both_contest": {
            "win_team_net_gold": _item_view(win_net),
            "lose_team_net_gold": _item_view(lose_net),
            "relative_win_minus_lose": _item_view(win_minus_lose),
            "note": (
                "Both miss lane farm → CS cancels. Item gap ≈ kill gold swing + 2× scrap. "
                "Winning the fight still buys a large item-spike lead; losing is catastrophic."
            ),
        },
        "horizons_after_window_min": horizon_rows,
        "dual_tempo_leave_gold": {
            "gold": round(dual_farm_g, 4),
            "wr_pp_at_even": _pp(delta_pp(b0, b1, 0.0, dual_farm_g)),
            "components": {
                "own_farm_g": round(f_own, 4),
                "opp_missed_farm_g": round(f_opp, 4),
            },
            "note": (
                "Sensitivity for leave EV: count opp missed waves in leave golddiff. "
                "Headline leave EV still uses own farm only (contest already forgoes F)."
            ),
        },
        "read": (
            f"Gift relative gap ≈ {rel_leave_vs_take:.0f}g "
            f"({rel_leave_vs_take / modal_c:.2f}× Pickaxe, "
            f"{rel_leave_vs_take / gpm_laner:.1f} solo-laner minutes). "
            f"Fight win−lose ≈ {win_minus_lose:.0f}g "
            f"({win_minus_lose / modal_c:.2f}× Pickaxe). "
            "Take/contest side is behind on item pace even when they 'win' free scrap."
        ),
    }


def leave_farm_scenarios(b0: float, b1: float) -> dict[str, Any]:
    """
    Leave gold for the giving team: N laners × 1 wiki early wave, ± E[plate].

    Gold constants from LoL wiki (V26.01 minion table + Turret Plating 120g).
    """
    out: dict[str, Any] = {
        "wiki_sources": [
            "https://wiki.leagueoflegends.com/en-us/Minion",
            "https://wiki.leagueoflegends.com/en-us/Turret",
        ],
        "constants": {
            "melee_minion_gold": MELEE_MINION_GOLD,
            "caster_minion_gold": CASTER_MINION_GOLD,
            "cannon_minion_gold_early": CANNON_MINION_GOLD_EARLY,
            "wave_gold_no_cannon": WAVE_GOLD_NO_CANNON,
            "wave_gold_avg_early_wiki_0_30": WAVE_GOLD_AVG_WIKI_0_30,
            "wave_gold_grub_era_at_10_00": WAVE_GOLD_EARLY,
            "plate_gold_local": PLATE_GOLD,
            "note": (
                "Wave = grub-era E[wave] at 10:00 = 102 + 56/3 = 120.6̅g "
                "(pre-14 composition; cannon gold 50+⌊t/90⌋). "
                "Plate = 120g local outer plating (plates persist past 14:00)."
            ),
        },
        "wave_gold_early": WAVE_GOLD_EARLY,
        "plate_gold": PLATE_GOLD,
        "scenarios": {},
        "hypothesis": (
            "If the giving team farms 1–3 laner-waves (wiki early wave gold) "
            "during the window and has a chance at a plate (120g), "
            "what map-WR pp does that buy vs contesting the river?"
        ),
        "headline_key": "two_laners_one_wave",
    }
    specs = [
        ("one_laner_one_wave", 1, 0.0, "1 laner × 1 grub-era average wave"),
        ("two_laners_one_wave", 2, 0.0, "2 laners × 1 grub-era average wave each"),
        ("three_laners_one_wave", 3, 0.0, "3 laners × 1 grub-era average wave each"),
        ("one_laner_wave_plus_plate", 1, 1.0, "1 laner × wave + one outer plate"),
        ("two_laners_wave_plus_plate", 2, 1.0, "2 laners × wave + one outer plate"),
        ("three_laners_wave_plus_plate", 3, 1.0, "3 laners × wave + one outer plate"),
        # lower bound: non-cannon wave only (102g)
        ("two_laners_nocannon_wave", 2, 0.0, "2 laners × non-cannon wave (102g each)"),
    ]
    for key, n_laners, p_plate, label in specs:
        per_wave = WAVE_GOLD_NO_CANNON if "nocannon" in key else WAVE_GOLD_EARLY
        waves_g = n_laners * per_wave
        plate_ev_g = p_plate * PLATE_GOLD
        gold = waves_g + plate_ev_g
        out["scenarios"][key] = {
            "label": label,
            "n_laners": n_laners,
            "waves_each": 1,
            "wave_gold_each": per_wave,
            "p_plate": p_plate,
            "gold": round(gold, 4),
            "components": {
                "waves_g": round(waves_g, 4),
                "plate_ev_g": round(plate_ev_g, 4),
            },
            "wr_pp_at_even": _pp(delta_pp(b0, b1, 0.0, gold)),
            "source": "wiki Minion avg early wave / Turret Plating 120g",
        }
    return out


# Legacy jungle packages kept only as optional sensitivity (not wiki-headline)
FARM_PACKAGES = {
    "low_one_wave": {
        "label": "1 laner × grub-era average wave (120.67g)",
        "gold": WAVE_GOLD_EARLY,
        "components": {"waves_g": WAVE_GOLD_EARLY, "jungle_g": 0.0},
    },
    "preferred_waves_plus_camp": {
        "label": "LEGACY: 2× wave + jungle camp (not headline)",
        "gold": 2 * WAVE_GOLD_EARLY + JUNGLE_CAMP_GOLD,
        "components": {
            "waves_g": 2 * WAVE_GOLD_EARLY,
            "jungle_g": JUNGLE_CAMP_GOLD,
        },
    },
    "high_three_waves_two_camps": {
        "label": "LEGACY: 3× wave + 2 camps (not headline)",
        "gold": 3 * WAVE_GOLD_EARLY + 2 * JUNGLE_CAMP_GOLD,
        "components": {
            "waves_g": 3 * WAVE_GOLD_EARLY,
            "jungle_g": 2 * JUNGLE_CAMP_GOLD,
        },
    },
}


def kill_net_gold_table(b0: float, b1: float) -> list[dict[str, Any]]:
    """Net kill differential for the contesting side → gold → WR pp at even."""
    rows = []
    for net_kills in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
        g = net_kills * KILL_BOUNTY_EARLY
        rows.append(
            {
                "net_kills_for_contester": net_kills,
                "gold": g,
                "wr_pp_at_even": _pp(delta_pp(b0, b1, 0.0, g)),
            }
        )
    return rows


def farm_opportunity_table(b0: float, b1: float) -> dict[str, Any]:
    out = {}
    for key, pkg in FARM_PACKAGES.items():
        out[key] = {
            **pkg,
            "wr_pp_at_even": _pp(delta_pp(b0, b1, 0.0, pkg["gold"])),
        }
    return out


def empirical_fight_swing_pp(era: pd.DataFrame, gold_logit: tuple[float, float]) -> dict[str, Any]:
    """
    Empirical gold swings @15 among maps that were near-even @10.
    Used as fight-loss severity priors (not filmed contests).
    """
    b0, b1 = gold_logit
    even = era[era.gold10.notna() & (era.gold10.abs() <= 300)].copy()
    even = even[even.gold15.notna()]
    dg = even.gold15 - even.gold10

    def wr_delta_for_gold_path(lo: float, hi: float) -> dict[str, Any]:
        g = even[(dg >= lo) & (dg < hi)]
        if len(g) < 80:
            return {"n": int(len(g)), "mean_dg": None, "emp_wr": None, "logit_pp_at_mean": None}
        mean_dg = float(dg[(dg >= lo) & (dg < hi)].mean())
        # blue-centric: negative path = blue lost tempo
        emp = float(g.y_blue_win.mean())
        # map mean path through gold@10 logit as if applied at even
        pp = delta_pp(b0, b1, 0.0, mean_dg)
        return {
            "n": int(len(g)),
            "mean_dg": mean_dg,
            "emp_wr": emp,
            "logit_mapped_pp_vs_even": pp,
        }

    # Losing a river: ~−800 to −1500g over 10→15 is a useful band
    bands = {
        "mild_loss_dg": wr_delta_for_gold_path(-800, -400),
        "medium_loss_dg": wr_delta_for_gold_path(-1500, -800),
        "severe_loss_dg": wr_delta_for_gold_path(-2500, -1500),
        "mild_gain_dg": wr_delta_for_gold_path(400, 800),
        "medium_gain_dg": wr_delta_for_gold_path(800, 1500),
    }
    # Structural kill package: 2 deaths × 300g
    deaths_2 = delta_pp(b0, b1, 0.0, -600.0)
    deaths_3 = delta_pp(b0, b1, 0.0, -900.0)
    return {
        "near_even_n": int(len(even)),
        "bands_gold10_to_15": bands,
        "structural_2_deaths_neg600g_pp": deaths_2,
        "structural_3_deaths_neg900g_pp": deaths_3,
    }


def contaminated_association_pp(era: pd.DataFrame) -> dict[str, Any]:
    """The NOT-intrinsic estimand — for contrast only."""
    d = era.dropna(subset=["gold10", "xp10", "y_blue_win", "blue_all3"]).copy()
    d["kills10_diff"] = d.get("kills10_diff", 0).fillna(0)
    y = d.y_blue_win.astype(int).values
    # headline controls
    X = np.column_stack(
        [
            d.blue_all3.values,
            d.gold10.values / MODEL_SCALE,
            d.kills10_diff.values,
            d.xp10.values / MODEL_SCALE,
        ]
    )
    m = np.isfinite(X).all(axis=1)
    clf = _fit_stable_logistic(X[m], y[m])
    # unique effect at means
    means = X[m].mean(axis=0)
    def pred(all3: float) -> float:
        z = means.copy()
        z[0] = all3
        return float(_sigmoid(float(clf.intercept_[0] + z @ clf.coef_[0])))

    dpp = (pred(1.0) - pred(0.0)) * 100.0
    return {
        "label": "OE association of blue 3–0 | gold/kills/xp @10 — NOT intrinsic package value",
        "unique_dpp": dpp,
        "n": int(m.sum()),
    }


def build_report() -> dict[str, Any]:
    print("[intrinsic] loading OE maps…")
    raw = load_maps()
    df = engineer_v2(raw)
    # The direct 90g / 195 XP package is the 2026 reward design.  Do not pool
    # 2025 maps, which had materially different rewards and strategic incentives.
    in_paper_leagues = df.league.isin(PAPER_LEAGUES)
    era = df[
        (df.oe_year >= 2026)
        & in_paper_leagues
        & df.era_3grub
    ].copy()
    print(f"[intrinsic] 2026 reward-era maps n={len(era)}")

    # Explicit sample-accounting funnel. engineer_v2 returns one row per
    # unique map, so these counts are map counts rather than doubled team rows.
    raw_2026 = df[(df.oe_year >= 2026) & in_paper_leagues].copy()
    gold10_numeric = pd.to_numeric(era.gold10, errors="coerce").to_numpy(dtype=float)
    outcome_numeric = pd.to_numeric(era.y_blue_win, errors="coerce").to_numpy(dtype=float)
    gold10_finite = np.isfinite(gold10_numeric)
    outcome_finite = np.isfinite(outcome_numeric)
    missing_gold10 = ~gold10_finite
    missing_outcome_after_gold = gold10_finite & ~outcome_finite
    outside_gold10_cap = (
        gold10_finite & outcome_finite & (np.abs(gold10_numeric) > 3000)
    )
    included_gold10 = (
        gold10_finite & outcome_finite & (np.abs(gold10_numeric) <= 3000)
    )

    g10 = era.gold10.values
    x10 = era.xp10.values
    y = era.y_blue_win.values

    b0, b1, n_g = fit_logit(g10, y, abs_cap=3000)
    x0, x1, n_x = fit_logit(x10, y, abs_cap=2000)
    j0, jg, jx, n_j = fit_joint(g10, x10, y, g_cap=3000, x_cap=2000)
    gold_cv = cross_validated_gold_diagnostics(g10, y, abs_cap=3000, folds=10)
    gold_fit_X = np.column_stack(
        [np.ones(int(included_gold10.sum())), gold10_numeric[included_gold10] / MODEL_SCALE]
    )
    gold_fit_covariance_scaled = _wald_covariance(
        gold_fit_X, b0, np.asarray([b1 * MODEL_SCALE])
    )
    if int(included_gold10.sum()) != int(n_g):
        raise RuntimeError(
            "Gold@10 sample-audit count does not match the fitted sample: "
            f"audit={included_gold10.sum()}, fit={n_g}"
        )

    # Gold@15 slope (for burn mapped through mid gold)
    g15 = era.gold15.values
    c0, c1, n_15 = fit_logit(g15, y, abs_cap=4000)

    gold_only = {
        "at_even": delta_pp(b0, b1, 0.0, PACKAGE_GOLD),
        "at_behind_1k": delta_pp(b0, b1, -1000.0, PACKAGE_GOLD),
        "at_ahead_1k": delta_pp(b0, b1, 1000.0, PACKAGE_GOLD),
        "pp_per_100g_at_even": delta_pp(b0, b1, 0.0, 100.0),
    }
    xp_only = {
        "at_even": delta_pp(x0, x1, 0.0, PACKAGE_XP),
        "fraction_of_level7": PACKAGE_XP / XP_TO_LEVEL_7,
    }
    joint_pkg = {
        "at_even_gold_and_xp": joint_delta_pp(j0, jg, jx, 0.0, 0.0, PACKAGE_GOLD, PACKAGE_XP),
        "at_even_gold_only_in_joint": joint_delta_pp(j0, jg, jx, 0.0, 0.0, PACKAGE_GOLD, 0.0),
        "at_even_xp_only_in_joint": joint_delta_pp(j0, jg, jx, 0.0, 0.0, 0.0, PACKAGE_XP),
        "at_behind_1k": joint_delta_pp(j0, jg, jx, -1000.0, 0.0, PACKAGE_GOLD, PACKAGE_XP),
    }

    # Burn scenarios (Patch 26.1 plate: 120g / 900 HP first outer plate)
    burn_wiki = {
        "pre_26_11_3stack": burn_gold_equivalent(
            tick_melee=TOTV_TICK_MELEE_3, siege_seconds=20.0
        ),
        "post_26_11_3stack": burn_gold_equivalent(
            tick_melee=TOTV_TICK_MELEE_3_POST_26_11, siege_seconds=20.0
        ),
        "pre_26_11_brief_8s": burn_gold_equivalent(
            tick_melee=TOTV_TICK_MELEE_3, siege_seconds=8.0
        ),
        "post_26_11_brief_8s": burn_gold_equivalent(
            tick_melee=TOTV_TICK_MELEE_3_POST_26_11, siege_seconds=8.0
        ),
        # Upper: 20s siege + one Hunger mite refresh (3-stack Hunger, 15s CD)
        "pre_26_11_20s_plus_hunger_mite": burn_gold_equivalent(
            tick_melee=TOTV_TICK_MELEE_3,
            siege_seconds=20.0,
            hunger_mite_refreshes=1,
        ),
    }
    for k, v in burn_wiki.items():
        v["wr_pp_via_gold10_logit"] = delta_pp(b0, b1, 0.0, v["gold_equivalent"])
        v["wr_pp_via_gold15_logit"] = delta_pp(c0, c1, 0.0, v["gold_equivalent"])

    ls_burn = ls_burn_gold_equivalent()
    ls_burn["early_20s_wr_pp"] = delta_pp(b0, b1, 0.0, ls_burn["early_gold_20s"])
    ls_burn["mid_full_turret_wr_pp"] = delta_pp(b0, b1, 0.0, ls_burn["mid_gold_saved_full_turret"])

    xp_local = xp_local_share_table()
    # Duo-river XP mapped as team-sum 195 (OE accounting); also report solo soak same sum.
    # Preferred XP add-on for "central" stays full joint; narrative uses share table.

    # Package totals (bounds) — preferred = scrap you get with no fight: local gold + short Touch.
    burn_pp = burn_wiki["pre_26_11_3stack"]["wr_pp_via_gold10_logit"]
    burn_brief_pp = burn_wiki["pre_26_11_brief_8s"]["wr_pp_via_gold10_logit"]
    burn_brief_post_pp = burn_wiki["post_26_11_brief_8s"]["wr_pp_via_gold10_logit"]
    burn_hunger_pp = burn_wiki["pre_26_11_20s_plus_hunger_mite"]["wr_pp_via_gold10_logit"]
    ls_burn_pp = ls_burn["early_20s_wr_pp"]
    # The extra structure damage is a conditional pressure ceiling, not cash.
    # Compute it as one probability difference rather than adding two nonlinear
    # logits.  The pre-26.11 figure is retained for the historical sample; the
    # post-26.11 version is the current-mechanics ceiling.
    preferred = delta_pp(
        b0, b1, 0.0, PACKAGE_GOLD + burn_wiki["pre_26_11_brief_8s"]["gold_equivalent"]
    )
    preferred_post = delta_pp(
        b0, b1, 0.0, PACKAGE_GOLD + burn_wiki["post_26_11_brief_8s"]["gold_equivalent"]
    )
    bounds = {
        "lower_gold_only_pp": gold_only["at_even"],
        "preferred_gold_plus_brief_burn_pp": preferred,
        "post_26_11_cash_plus_brief_pressure_ceiling_pp": preferred_post,
        "central_gold_plus_xp_joint_pp": joint_pkg["at_even_gold_and_xp"],
        "upper_joint_plus_wiki_burn_20s_pp": joint_delta_pp(
            j0, jg, jx, 0.0, 0.0,
            PACKAGE_GOLD + burn_wiki["pre_26_11_3stack"]["gold_equivalent"], PACKAGE_XP,
        ),
        "upper_joint_plus_hunger_mite_pp": joint_delta_pp(
            j0, jg, jx, 0.0, 0.0,
            PACKAGE_GOLD + burn_wiki["pre_26_11_20s_plus_hunger_mite"]["gold_equivalent"], PACKAGE_XP,
        ),
        "upper_joint_plus_ls_burn_20s_pp": joint_pkg["at_even_gold_and_xp"] + ls_burn_pp,
        "ls_style_mid_turret_burn_alone_pp": ls_burn["mid_full_turret_wr_pp"],
        "notes": (
            "Cash lower bound = 90g local. The 8s Touch term is an undiscounted "
            "first-plate-progress ceiling, not paid gold. Central adds joint team XP "
            "(+195 team xpdiff; river-local share). Upper adds 20s continuous Touch; "
            "Hunger-mite upper adds one refresh cycle."
        ),
    }

    sampling_intervals = {
        "cash_90g": univariate_delta_sampling_ci(
            g10, y, abs_cap=3000, intercept=b0, coef=b1, base=0.0, bump=PACKAGE_GOLD
        ),
        "cash_plus_pre26_11_brief_pressure_ceiling": univariate_delta_sampling_ci(
            g10, y, abs_cap=3000, intercept=b0, coef=b1, base=0.0,
            bump=PACKAGE_GOLD + burn_wiki["pre_26_11_brief_8s"]["gold_equivalent"],
        ),
        "cash_plus_post26_11_brief_pressure_ceiling": univariate_delta_sampling_ci(
            g10, y, abs_cap=3000, intercept=b0, coef=b1, base=0.0,
            bump=PACKAGE_GOLD + burn_wiki["post_26_11_brief_8s"]["gold_equivalent"],
        ),
        "joint_cash_xp": joint_delta_sampling_ci(
            g10, x10, y, gold_cap=3000, xp_cap=2000, intercept=j0,
            coef_gold=jg, coef_xp=jx, gold_base=0.0, xp_base=0.0,
            gold_bump=PACKAGE_GOLD, xp_bump=PACKAGE_XP,
        ),
    }

    swings = empirical_fight_swing_pp(era, (b0, b1))
    farm = farm_opportunity_table(b0, b1)
    kills = kill_net_gold_table(b0, b1)
    farm_pref_pp = float(farm["preferred_waves_plus_camp"]["wr_pp_at_even"])
    # Median TF assumption: net +2 kills on win / net -2 on loss (300g bounty each)
    kill_win_pp = float(next(r["wr_pp_at_even"] for r in kills if r["net_kills_for_contester"] == 2))
    kill_lose_pp = abs(
        float(next(r["wr_pp_at_even"] for r in kills if r["net_kills_for_contester"] == -2))
    )

    delta_obj = bounds["preferred_gold_plus_brief_burn_pp"]
    delta_obj_joint = bounds["central_gold_plus_xp_joint_pp"]
    delta_obj_upper = bounds["upper_joint_plus_wiki_burn_20s_pp"]
    death2 = abs(swings["structural_2_deaths_neg600g_pp"])
    death3 = abs(swings["structural_3_deaths_neg900g_pp"])
    # Prefer kill-table death magnitude (same -600g) for consistency with win-side kills
    loss_2 = kill_lose_pp + delta_obj
    loss_3 = death3 + delta_obj

    curves = {}
    specs = [
        # name, loss, obj, win_extra, farm
        ("baseline_2deaths_gift_no_farm", loss_2, delta_obj, 0.0, 0.0),
        # Headline (v4): farm demoted — kill lottery only, no invented leave-farm
        ("headline_nofarm_2kills_sym", loss_2, delta_obj, kill_win_pp, 0.0),
        ("preferred_farm_2kills_sym", loss_2, delta_obj, kill_win_pp, farm_pref_pp),
        ("preferred_farm_only_no_kill_bonus", loss_2, delta_obj, 0.0, farm_pref_pp),
        ("high_farm_2kills_sym", loss_2, delta_obj, kill_win_pp, float(farm["high_three_waves_two_camps"]["wr_pp_at_even"])),
        ("joint_obj_nofarm_2kills", death2 + delta_obj_joint, delta_obj_joint, kill_win_pp, 0.0),
        ("joint_obj_farm_2kills", death2 + delta_obj_joint, delta_obj_joint, kill_win_pp, farm_pref_pp),
        ("upper_obj_farm_2kills", death2 + delta_obj_upper, delta_obj_upper, kill_win_pp, farm_pref_pp),
        ("preferred_3deaths_farm_2kills", loss_3, delta_obj, kill_win_pp, farm_pref_pp),
        ("headline_nofarm_3deaths_2kills", death3 + delta_obj, delta_obj, kill_win_pp, 0.0),
    ]
    for name, loss, obj, win_x, farm_x in specs:
        rows, be = contest_ev_table(
            obj, loss, fight_win_extra_pp=win_x, farm_pp=farm_x
        )
        curves[name] = {
            "delta_obj_pp": _pp(obj),
            "fight_loss_pp": _pp(loss),
            "fight_win_extra_pp": _pp(win_x),
            "farm_pp": _pp(farm_x),
            "breakeven_p_win_fight": be,
            "curve": rows,
        }

    mix_rows, mix_be, mix_cells = contest_ev_mixture_table(
        delta_obj, kill_win_pp, kill_lose_pp, farm_pp=0.0,
        p_secure_if_win=0.85, p_secure_if_lose=0.05,
    )
    curves["headline_nofarm_2kills_mixture"] = {
        "delta_obj_pp": _pp(delta_obj),
        "fight_loss_pp": _pp(loss_2),
        "fight_win_extra_pp": _pp(kill_win_pp),
        "farm_pp": 0.0,
        "breakeven_p_win_fight": mix_be,
        "curve": mix_rows,
        "mixture": {k: _pp(v) if isinstance(v, float) and k.endswith("_pp") else v for k, v in mix_cells.items()},
        "note": (
            "Averages over 2×2: P(secure|win)=0.85 (smite loss), "
            "P(secure|lose)=0.05 (rare secure-then-collapse). Farm=0."
        ),
    }

    # Backward-compatible aliases
    curves["preferred_obj_vs_2deaths_plus_gift"] = curves["preferred_farm_2kills_sym"]
    curves["preferred_obj_vs_3deaths_plus_gift"] = curves["preferred_3deaths_farm_2kills"]
    curves["joint_obj_vs_2deaths_plus_gift"] = curves["joint_obj_farm_2kills"]
    curves["upper_obj_vs_2deaths_plus_gift"] = curves["upper_obj_farm_2kills"]

    head = curves["headline_nofarm_2kills_sym"]
    p25 = head["curve"]
    row25 = next(r for r in p25 if abs(r["p_win_fight"] - 0.25) < 1e-9)
    fight_loss_2d = loss_2
    mix25 = next(r for r in mix_rows if abs(r["p_win_fight"] - 0.25) < 1e-9)
    farm25 = next(
        r
        for r in curves["preferred_farm_2kills_sym"]["curve"]
        if abs(r["p_win_fight"] - 0.25) < 1e-9
    )

    contaminated = contaminated_association_pp(era)
    recon = sister_study_reconciliation()

    out: dict[str, Any] = {
        "version": 4,
        "title": "Void Grubs scrap proxy and contest sensitivity (associational)",
        "estimand": (
            "Associational scrap proxy via gold@10/xp@10 logits under mechanical bumps — "
            "not an experimental grant with combat held fixed. "
            "T_full = {90g local, 195 XP in radius 2000, Touch@3 (+ Hunger mite)}. "
            "T_pref (headline) = {90g local, brief 8s Touch→first-plate gold}; "
            "XP enters joint/upper bounds, not the preferred headline. "
            "Headline contest EV: farm=0, median ±2 kill prior, diagonal secure; "
            "leave-farm and 2×2 mixture are sensitivities. "
            "p(fight-win) is an exogenous narrative input."
        ),
        "sample": {
            "n_raw_2026_maps": int(len(raw_2026)),
            "n_era_3camp": int(len(era)),
            "n_exactly_three_grubs": int((era.grub_sum == 3).sum()),
            "n_fewer_than_three_grubs": int((era.grub_sum < 3).sum()),
            "paper_leagues": list(PAPER_LEAGUES),
            "n_leagues": int(era.league.nunique(dropna=True)),
            "n_fit_leagues": int(era.loc[included_gold10, "league"].nunique(dropna=True)),
            "n_gold10_missing": int(missing_gold10.sum()),
            "gold10_missing_leagues": sorted(
                str(v) for v in era.loc[missing_gold10, "league"].dropna().unique()
            ),
            "n_outcome_missing_after_gold": int(missing_outcome_after_gold.sum()),
            "n_gold10_outside_cap": int(outside_gold10_cap.sum()),
            "filter": "OE 16.xx / Patch 26.1+ reward era; LCS/LCK/LEC/LPL/CBLOL",
            "date_min": str(pd.to_datetime(era.date).min().date()) if len(era) else None,
            "date_max": str(pd.to_datetime(era.date).max().date()) if len(era) else None,
            "n_fit_gold10": int(n_g),
            "n_fit_xp10": int(n_x),
            "n_fit_joint": int(n_j),
            "missingness_note": (
                f"Complete-case logit fits use n_gold={n_g}, n_xp={n_x}, n_joint={n_j} "
                f"vs era n={len(era)} (caps |gold|≤3000, |xp|≤2000)."
            ),
        },
        "mechanical_package": {
            "gold_local": PACKAGE_GOLD,
            "gold": PACKAGE_GOLD,
            "xp": PACKAGE_XP,
            "xp_radius": XP_RADIUS,
            "xp_to_level_7_to_8": XP_TO_LEVEL_7,
            "xp_local_share": xp_local,
            "totv_tick_melee_3stack_pre_26_11": TOTV_TICK_MELEE_3,
            "totv_tick_melee_3stack_post_26_11": TOTV_TICK_MELEE_3_POST_26_11,
            "totv_tick_melee_by_stack_post_26_11": list(TOTV_TICK_MELEE_BY_STACK_POST_26_11),
            "totv_tick_ranged_by_stack_post_26_11": list(TOTV_TICK_RANGED_BY_STACK_POST_26_11),
            "totv_damage_per_aa_cycle_melee": totv_damage_per_cycle(TOTV_TICK_MELEE_3),
            "totv_damage_per_aa_cycle_ranged": totv_damage_per_cycle(TOTV_TICK_RANGED_3),
            "totv_burn_duration_s": TOTV_BURN_DURATION,
            "totv_tick_interval_s": TOTV_TICK_INTERVAL,
            "hunger_mite_cd_s": HUNGER_MITE_CD,
            "hunger_at_3_stacks": True,
            "hunger_touch_stacks_required": HUNGER_TOUCH_STACKS_REQUIRED,
            "plate_gold": PLATE_GOLD,
            "outer_turret_hp": OUTER_TURRET_HP,
            "plate_hp_first_outer": PLATE_HP_FIRST,
            "plate_hp_bands_outer": list(PLATE_HP_BANDS),
            "plate_missing_hp_fractions": list(PLATE_MISSING_HP_FRACTIONS),
            "plate_gold_full_outer": PLATE_GOLD_FULL_OUTER,
            "wiki_economy": {
                "reference_clock": "grub_window_pre14_at_10_00",
                "grub_spawn_s": GRUB_SPAWN_S,
                "grub_despawn_s": GRUB_DESPAWN_S,
                "reference_clock_s": GRUB_REF_CLOCK_S,
                "melee_gold": MELEE_MINION_GOLD,
                "caster_gold": CASTER_MINION_GOLD,
                "cannon_gold_base": CANNON_MINION_GOLD_BASE,
                "cannon_gold_at_reference": CANNON_MINION_GOLD_EARLY,
                "cannon_gold_rule": "50 + floor(t_seconds / 90)",
                "wave_gold_no_cannon": WAVE_GOLD_NO_CANNON,
                "wave_gold_cannon_at_reference": WAVE_GOLD_CANNON_AT_REF,
                "wave_gold_expected_at_reference": WAVE_GOLD_EARLY,
                "wave_gold_avg_wiki_0_30": WAVE_GOLD_AVG_WIKI_0_30,
                "wave_gold_expected_at_08_00": wave_gold_expected_pre14(GRUB_SPAWN_S),
                "wave_gold_expected_at_14_00_table": wave_gold_expected_at_14_00(),
                "wave_xp_no_cannon": WAVE_XP_NO_CANNON,
                "wave_xp_cannon": WAVE_XP_CANNON,
                "wave_xp_expected_pre14": WAVE_XP_AVG_PRE14,
                "plate_gold_local": PLATE_GOLD,
                "plates_persist_past_14_00": True,
                "outer_assumed_standing": True,
                "sources": [
                    "https://wiki.leagueoflegends.com/en-us/Minion",
                    "https://wiki.leagueoflegends.com/en-us/Siege_minion",
                    "https://wiki.leagueoflegends.com/en-us/Turret",
                ],
            },
            "worked_siege_example": worked_siege_example(),
            "wiki_sources": [
                "https://wiki.leagueoflegends.com/en-us/Voidgrub_camp",
                "https://wiki.leagueoflegends.com/en-us/Touch_of_the_Void",
                "https://wiki.leagueoflegends.com/en-us/Hunger_of_the_Void",
                "https://wiki.leagueoflegends.com/en-us/Minion",
                "https://wiki.leagueoflegends.com/en-us/Turret",
                "https://wiki.leagueoflegends.com/en-us/Experience_(champion)",
            ],
        },
        "logits": {
            "gold10": {
                "intercept": b0,
                "coef": b1,
                "n": n_g,
                "cap": 3000,
                "map_level_independence_covariance_scaled": gold_fit_covariance_scaled.tolist(),
            },
            "xp10": {"intercept": x0, "coef": x1, "n": n_x, "cap": 2000},
            "joint_gold_xp": {"intercept": j0, "coef_gold": jg, "coef_xp": jx, "n": n_j},
            "gold15": {"intercept": c0, "coef": c1, "n": n_15, "cap": 4000},
        },
        "model_diagnostics": {"gold10_10fold": gold_cv},
        "component_wr_pp": {
            "gold_only": gold_only,
            "xp_only_univariate": xp_only,
            "joint_gold_xp": joint_pkg,
            "xp_local_share": xp_local,
        },
        "sampling_intervals": sampling_intervals,
        "burn": {"wiki_scenarios": burn_wiki, "ls_sandbox": ls_burn},
        "intrinsic_bounds_pp_at_even": {
            **bounds,
            "notes": (
                "Preferred = T_pref = 90g local scrap + 8s melee Touch→first-plate gold "
                "(NOT full T; XP is in joint/upper). "
                "Maps are associational unit conversions via gold@10/xp@10 logits."
            ),
        },
        "opportunity_gold": {
            "window_note": (
                "Structural sensitivity only — not OE-logged. Demoted from headline EV. "
                "Leave farm packages (waves + jungle) during ~60-90s contest window."
            ),
            "packages": farm,
            "kill_bounty_early": KILL_BOUNTY_EARLY,
            "kill_net_table": kills,
            "median_tf_assumption": {
                "net_kills_on_win": 2,
                "net_kills_on_lose": -2,
                "win_pp": kill_win_pp,
                "lose_pp": kill_lose_pp,
                "note": "Structural prior (300g bounty × net kills), not filmed median TF.",
            },
        },
        "river_outcome_matrix": river_outcome_matrix(
            delta_obj, 0.0, kill_win_pp, kill_lose_pp
        ),
        "contaminated_association_for_contrast": contaminated,
        "fight_swing_priors": swings,
        "contest_ev": curves,
        "headline_key": "headline_nofarm_2kills_sym",
        "sister_study_reconciliation": recon,
        "ls_furia_scenario": {
            "description": (
                "Headline (v4): leave gifts Δ_pref with farm=0; contest forgoes farm; "
                "win = Δ_pref + net +2 kills; lose = net −2 kills + gift (diagonal secure). "
                "Leave-farm and 2×2 mixture are sensitivities. p is exogenous."
            ),
            "delta_obj_pp_preferred": _pp(delta_obj),
            "farm_pp": 0.0,
            "fight_win_extra_pp": _pp(kill_win_pp),
            "fight_loss_pp": _pp(fight_loss_2d),
            "at_p_025": row25,
            "breakeven_p": head["breakeven_p_win_fight"],
            "mixture_at_p_025": mix25,
            "mixture_breakeven_p": mix_be,
            "farm_sensitivity_at_p_025": farm25,
            "farm_sensitivity_breakeven_p": curves["preferred_farm_2kills_sym"][
                "breakeven_p_win_fight"
            ],
            "verdict": (
                f"Under priors (T_pref, farm=0, ±2 kills, diagonal): "
                f"{'LEAVE' if row25['edge_contest_minus_leave_pp'] < 0 else 'CONTEST'} "
                f"at p=0.25"
            ),
        },
        "answers": _answers_v4(
            preferred=preferred,
            bounds=bounds,
            farm=farm,
            farm_pref_pp=farm_pref_pp,
            kill_win_pp=kill_win_pp,
            kill_lose_pp=kill_lose_pp,
            fight_loss_2d=fight_loss_2d,
            row25=row25,
            head=head,
            mix25=mix25,
            mix_be=mix_be,
            farm25=farm25,
            contaminated=contaminated,
            recon=recon,
            delta_obj=delta_obj,
        ),
    }
    return out


def sister_study_reconciliation() -> dict[str, Any]:
    """Contrast structural scrap EV vs OE take-regime contest numbers (different estimand)."""
    path = MODELS_DIR / "grubs_decision_numbers.json"
    if not path.exists():
        return {"available": False, "note": "grubs_decision_numbers.json missing"}
    sib = json.loads(path.read_text())
    row25 = None
    for r in sib.get("ev_curve", []):
        if abs(float(r.get("p_win_fight", -1)) - 0.25) < 1e-9:
            row25 = r
            break
    be = sib.get("breakeven_p_win_fight", {})
    return {
        "available": True,
        "source": "grubs_decision_numbers.json",
        "estimand": (
            "OE take-regime contest EV among maps (who ended with all3 / leave_mix / split) "
            "— selection + tempo, NOT the mechanical scrap proxy in this note."
        ),
        "at_p_025_edge_vs_leave_mix_pp": (
            None if row25 is None else row25.get("edge_vs_leave_mix_pp")
        ),
        "at_p_025_verdict_vs_leave": (
            None if row25 is None else row25.get("verdict_vs_leave")
        ),
        "breakeven_p_vs_leave_mix": be.get("vs_leave_mix"),
        "breakeven_p_vs_split": be.get("vs_split"),
        "do_not_coach_from_either_alone": True,
        "note": (
            "Sister study at p=0.25 is near toss-up vs leave_mix (~0pp / p*~0.26). "
            "This note's structural scrap lottery can show large negative edges under "
            "death+gift priors. Different objects — report both; do not collapse."
        ),
    }


def _answers_v4(
    *,
    preferred: float,
    bounds: dict,
    farm: dict,
    farm_pref_pp: float,
    kill_win_pp: float,
    kill_lose_pp: float,
    fight_loss_2d: float,
    row25: dict,
    head: dict,
    mix25: dict,
    mix_be: Optional[float],
    farm25: dict,
    contaminated: dict,
    recon: dict,
    delta_obj: float,
) -> dict[str, str]:
    mix_be_s = f"{mix_be:.2f}" if mix_be is not None else "n/a"
    sib = ""
    if recon.get("available"):
        sib = (
            f" Sister OE contest EV at p=0.25 vs leave_mix ≈ "
            f"{recon.get('at_p_025_edge_vs_leave_mix_pp')}pp "
            f"({recon.get('at_p_025_verdict_vs_leave')}) — different estimand."
        )
    return {
        "what_is_the_scrap": (
            "Full mechanical package T_full = 90g local + 195 XP (radius 2000) + Touch@3 "
            "(Hunger mite). Headline price T_pref excludes XP: 90g + brief 8s Touch→first plate "
            f"≈ {_pp(preferred):+.2f}pp via gold@10 logit (associational proxy, not a free grant). "
            f"Joint gold+XP bound ≈ {_pp(bounds['central_gold_plus_xp_joint_pp']):+.2f}pp."
        ),
        "how_large_is_intrinsic_grub_value": (
            f"T_pref (90g + 8s Touch): {_pp(preferred):+.2f}pp. "
            f"Gold alone: {_pp(bounds['lower_gold_only_pp']):+.2f}pp. "
            f"Joint gold+XP: {_pp(bounds['central_gold_plus_xp_joint_pp']):+.2f}pp. "
            f"Upper (joint + 20s burn): {_pp(bounds['upper_joint_plus_wiki_burn_20s_pp']):+.2f}pp. "
            f"LS mid-turret burn alone: {_pp(bounds['ls_style_mid_turret_burn_alone_pp']):+.2f}pp."
        ),
        "leave_farm_gold": (
            f"Sensitivity only (not headline): preferred leave-window farm = "
            f"{farm['preferred_waves_plus_camp']['gold']:.0f}g → {_pp(farm_pref_pp):+.2f}pp. "
            "Not OE-logged; inventing this gold inflates leave EV."
        ),
        "tf_kill_gold": (
            f"Early kill bounty {KILL_BOUNTY_EARLY:.0f}g prior (assists ignored). "
            f"Median TF net +2 / −2 → {_pp(kill_win_pp):+.2f}pp / {_pp(-kill_lose_pp):+.2f}pp."
        ),
        "is_ls_right": (
            "On the Furia call under these priors (don't take a clearly unfavored river to deny "
            f"T_pref ≈ {_pp(preferred):+.2f}pp): the sensitivity still favors leave at p=0.25. "
            "On literal '0.10pp': no — that is rhetorical undersell. "
            "This is not an identified causal scrap effect."
        ),
        "is_it_decimal_like_ls": (
            f"LS's '0.10pp' is rhetorical. T_pref proxy ≈ {_pp(preferred):+.2f}pp — "
            f"small vs −2 kills ({_pp(-kill_lose_pp):+.2f}pp), and not take-regime "
            f"({_pp(contaminated['unique_dpp']):+.2f}pp)."
        ),
        "should_you_take_25_75_to_deny": (
            f"Under headline priors (T_pref={_pp(delta_obj):+.2f}pp, farm=0, "
            f"+2/−2 kills, diagonal): at p=25% edge vs leave = "
            f"{_pp(row25['edge_contest_minus_leave_pp']):+.2f}pp → "
            f"{'LEAVE' if row25['edge_contest_minus_leave_pp'] < 0 else 'CONTEST'}; "
            f"p* ≈ {head['breakeven_p_win_fight']:.2f}. "
            f"2×2 mixture: edge {_pp(mix25['edge_contest_minus_leave_pp']):+.2f}pp, "
            f"p* ≈ {mix_be_s}. "
            f"With invented leave-farm: edge {_pp(farm25['edge_contest_minus_leave_pp']):+.2f}pp "
            f"(sensitivity).{sib}"
        ),
        "why_not_plus_3_7pp": (
            f"The {_pp(contaminated['unique_dpp']):+.2f}pp OE association after gold@10 "
            "is selection + tempo of teams that took 3–0, not the scrap proxy."
        ),
        "certainty_disclaimer": (
            "Do not read LEAVE as a scientific law. It is a sensitivity result under stated "
            "priors (T_pref excludes XP; farm demoted; exogenous p; diagonal or mixture). "
            "Sister OE contest work is a different estimand — do not coach from either alone."
        ),
    }


def enrich_report_v4(d: dict[str, Any]) -> dict[str, Any]:
    """
    v6 policy: leave-farm headline (wiki wave + plate EV) + item-pace layer
    (component costs, leave−take relative gold, fight win−lose gap, 1/2/3 min
    horizons). Dual-tempo leave is sensitivity. Recomputes without OE refit.
    """
    b = d["intrinsic_bounds_pp_at_even"]
    preferred = float(b["preferred_gold_plus_brief_burn_pp"])
    delta_obj = preferred
    delta_obj_joint = float(b["central_gold_plus_xp_joint_pp"])
    delta_obj_upper = float(b["upper_joint_plus_wiki_burn_20s_pp"])
    logits = d["logits"]["gold10"]
    b0, b1 = float(logits["intercept"]), float(logits["coef"])
    leave = leave_farm_scenarios(b0, b1)
    head_farm_key = leave["headline_key"]
    farm_headline = leave["scenarios"][head_farm_key]
    farm_pp = float(farm_headline["wr_pp_at_even"])
    preferred_obj_gold = PACKAGE_GOLD + float(
        d["burn"]["wiki_scenarios"]["pre_26_11_brief_8s"]["undiscounted_plate_progress_g"]
    )
    certainty_atlas = contest_certainty_atlas(
        b0,
        b1,
        touch_gold=preferred_obj_gold - PACKAGE_GOLD,
        covariance_scaled=np.asarray(
            logits["map_level_independence_covariance_scaled"], dtype=float
        ),
    )

    def state_curve(
        *,
        objective_gold: float = preferred_obj_gold,
        farm_gold: float = 0.0,
        p_plate: float = 0.0,
        win_kill_gold: float = 600.0,
        loss_kill_gold: float = -600.0,
    ) -> tuple[list[dict[str, Any]], Optional[float], dict[str, float]]:
        return contest_ev_terminal_states(
            b0, b1,
            objective_gold=objective_gold,
            leave_farm_gold=farm_gold,
            leave_plate_probability=p_plate,
            win_kill_gold=win_kill_gold,
            loss_kill_gold=loss_kill_gold,
        )

    # Keep legacy packages mapped with new wave gold for any old keys
    farm_legacy = farm_opportunity_table(b0, b1)

    kill_win_pp = float(d["opportunity_gold"]["median_tf_assumption"]["win_pp"])
    kill_lose_pp = abs(float(d["opportunity_gold"]["median_tf_assumption"]["lose_pp"]))
    # refresh kill table under same logit (unchanged bounty)
    kills = kill_net_gold_table(b0, b1)
    kill_win_pp = float(next(r["wr_pp_at_even"] for r in kills if r["net_kills_for_contester"] == 2))
    kill_lose_pp = abs(float(next(r["wr_pp_at_even"] for r in kills if r["net_kills_for_contester"] == -2)))

    death2 = abs(float(d["fight_swing_priors"]["structural_2_deaths_neg600g_pp"]))
    death3 = abs(float(d["fight_swing_priors"]["structural_3_deaths_neg900g_pp"]))
    loss_2 = kill_lose_pp + delta_obj
    loss_3 = death3 + delta_obj

    curves: dict[str, Any] = {}
    # Core leave-farm axis
    for skey, sc in leave["scenarios"].items():
        name = f"leave_{skey}_2kills"
        rows, be, terminal = state_curve(
            farm_gold=float(sc["components"]["waves_g"]),
            p_plate=float(sc["p_plate"]),
        )
        curves[name] = {
            "delta_obj_pp": _pp(delta_obj),
            "fight_loss_pp": _pp(loss_2),
            "fight_win_extra_pp": _pp(kill_win_pp),
            "farm_pp": _pp(sc["wr_pp_at_even"]),
            "farm_gold": sc["gold"],
            "farm_label": sc["label"],
            "leave_scenario_key": skey,
            "terminal_states": terminal,
            "breakeven_p_win_fight": be,
            "curve": rows,
        }

    specs = [
        ("baseline_gift_no_farm_no_kills", 0.0, 0.0, 0.0, 0.0),
        ("contrast_nofarm_2kills", 0.0, 0.0, 600.0, -600.0),
        ("headline_farm_3deaths_2kills", float(farm_headline["components"]["waves_g"]), float(farm_headline["p_plate"]), 600.0, -900.0),
    ]
    for name, farm_g, plate_p, win_g, loss_g in specs:
        rows, be, terminal = state_curve(
            farm_gold=farm_g, p_plate=plate_p,
            win_kill_gold=win_g, loss_kill_gold=loss_g,
        )
        curves[name] = {
            "delta_obj_pp": _pp(delta_obj),
            "fight_loss_pp": _pp(loss_2),
            "fight_win_extra_pp": _pp(kill_win_pp),
            "farm_pp": _pp(farm_pp),
            "breakeven_p_win_fight": be,
            "curve": rows,
            "terminal_states": terminal,
        }

    head_key = f"leave_{head_farm_key}_2kills"
    head = curves[head_key]

    # Item-pace / dual-tempo: leave own farm + opp missed waves while taking river.
    n_head = int(farm_headline.get("n_laners") or 2)
    item_pace = item_completion_pace(
        b0,
        b1,
        farm_gold=float(farm_headline["gold"]),
        scrap_gold=PACKAGE_GOLD,
        n_laners=n_head,
    )
    dual_farm_pp = float(item_pace["dual_tempo_leave_gold"]["wr_pp_at_even"])
    dual_rows, dual_be, dual_terminal = state_curve(
        farm_gold=float(item_pace["dual_tempo_leave_gold"]["gold"])
    )
    curves["leave_dual_tempo_own_plus_opp_miss_2kills"] = {
        "delta_obj_pp": _pp(delta_obj),
        "fight_loss_pp": _pp(loss_2),
        "fight_win_extra_pp": _pp(kill_win_pp),
        "farm_pp": _pp(dual_farm_pp),
        "farm_gold": item_pace["dual_tempo_leave_gold"]["gold"],
        "farm_label": (
            f"dual tempo: own {farm_headline['gold']:.1f}g + opp miss "
            f"{item_pace['constants']['farm_gold_opp_missed']:.1f}g"
        ),
        "breakeven_p_win_fight": dual_be,
        "curve": dual_rows,
        "terminal_states": dual_terminal,
        "note": (
            "Sensitivity: leave golddiff counts opponent missed waves while they take. "
            "Not headline (avoids double-counting vs contest-forgoes-F framing)."
        ),
    }

    mix_rows, mix_be, mix_cells = head["curve"], head["breakeven_p_win_fight"], head["terminal_states"]
    curves["headline_leave_farm_2kills_mixture"] = {
        "delta_obj_pp": _pp(delta_obj),
        "fight_loss_pp": _pp(loss_2),
        "fight_win_extra_pp": _pp(kill_win_pp),
        "farm_pp": _pp(farm_pp),
        "breakeven_p_win_fight": mix_be,
        "curve": mix_rows,
        "mixture": {
            k: (_pp(v) if isinstance(v, float) and str(k).endswith("_pp") else v)
            for k, v in mix_cells.items()
        },
        "note": "2×2 mixture with headline leave-farm (wiki waves + plate EV).",
    }

    # aliases for older PDF paths
    curves["preferred_farm_2kills_sym"] = head
    curves["preferred_obj_vs_2deaths_plus_gift"] = head
    curves["headline_nofarm_2kills_sym"] = curves["contrast_nofarm_2kills"]
    curves["preferred_3deaths_farm_2kills"] = curves["headline_farm_3deaths_2kills"]
    curves["baseline_2deaths_gift_no_farm"] = curves["baseline_gift_no_farm_no_kills"]
    curves["preferred_farm_only_no_kill_bonus"] = {
        **{k: v for k, v in head.items() if k != "curve"},
        "fight_win_extra_pp": 0.0,
        "curve": state_curve(
            farm_gold=float(farm_headline["components"]["waves_g"]),
            p_plate=float(farm_headline["p_plate"]),
            win_kill_gold=0.0,
        )[0],
        "breakeven_p_win_fight": state_curve(
            farm_gold=float(farm_headline["components"]["waves_g"]),
            p_plate=float(farm_headline["p_plate"]),
            win_kill_gold=0.0,
        )[1],
    }
    # 1/2/3 laner no-plate shortcuts for table
    for n, sk in [(1, "one_laner_one_wave"), (2, "two_laners_one_wave"), (3, "three_laners_one_wave")]:
        curves[f"leave_{n}_laner_wave_2kills"] = curves[f"leave_{sk}_2kills"]

    row25 = next(r for r in head["curve"] if abs(r["p_win_fight"] - 0.25) < 1e-9)
    mix25 = next(r for r in mix_rows if abs(r["p_win_fight"] - 0.25) < 1e-9)
    nofarm25 = next(
        r for r in curves["contrast_nofarm_2kills"]["curve"] if abs(r["p_win_fight"] - 0.25) < 1e-9
    )
    dual25 = next(r for r in dual_rows if abs(r["p_win_fight"] - 0.25) < 1e-9)
    recon = sister_study_reconciliation()
    contaminated = d["contaminated_association_for_contrast"]

    d["version"] = 7
    d["title"] = "Void Grubs: cash reward, conditional pressure, and contest sensitivity"
    d["estimand"] = (
        "Associational calibration of the 90g cash reward and conditional Touch pressure "
        "ceiling via a gold@10 logit, "
        "versus leave-window farm: N∈{1,2,3} laners × grub-era average wave gold (120.67g) "
        "plus E[plate]=p×120g. Headline leave = 2 laners + 25% plate (own farm only). "
        "Item-pace layer: median/modal early component costs, relative leave−take gold "
        "(own farm + opp missed waves − scrap), fight win−lose item gap, and 1/2/3 min "
        "horizons while river side is delayed. Dual-tempo leave is sensitivity only."
    )
    d["leave_farm"] = leave
    d["opportunity_gold"] = {
        "window_note": (
            "HEADLINE leave hypothesis: wiki early-wave gold × N laners + optional plate EV. "
            "Mapped through gold@10 logit. Jungle camps are NOT in the headline."
        ),
        "leave_farm": leave,
        "packages": farm_legacy,  # legacy only
        "kill_bounty_early": KILL_BOUNTY_EARLY,
        "kill_net_table": kills,
        "median_tf_assumption": {
            "net_kills_on_win": 2,
            "net_kills_on_lose": -2,
            "win_pp": kill_win_pp,
            "lose_pp": kill_lose_pp,
            "note": "Structural prior (300g × net kills), not filmed median TF.",
        },
    }
    d["contest_ev"] = curves
    d["headline_key"] = head_key
    d["certainty_atlas"] = certainty_atlas
    d["item_pace"] = item_pace
    d["sister_study_reconciliation"] = recon
    d["river_outcome_matrix"] = river_outcome_terminal_matrix(
        b0,
        b1,
        objective_gold=preferred_obj_gold,
        leave_farm_gold=float(farm_headline["components"]["waves_g"]),
        leave_plate_probability=float(farm_headline["p_plate"]),
        win_kill_gold=600.0,
        loss_kill_gold=-600.0,
    )
    d["ls_furia_scenario"] = {
        "description": (
            f"Illustrative leave scenario: {farm_headline['label']} "
            f"({farm_headline['gold']:.2f}g → {farm_pp:+.2f}pp wiki→logit); "
            "contest forgoes farm; win = scrap + +2 kills; lose = −2 kills + gift. "
            f"Item-pace: gift relative leave−take ≈ "
            f"{item_pace['gift_path_no_tf']['relative_leave_minus_take']['gold']:.0f}g "
            f"({item_pace['gift_path_no_tf']['relative_leave_minus_take']['frac_modal_pickaxe_875']:.2f}× Pickaxe)."
        ),
        "delta_obj_pp_preferred": _pp(delta_obj),
        "farm_pp": _pp(farm_pp),
        "farm_gold": farm_headline["gold"],
        "farm_label": farm_headline["label"],
        "fight_win_extra_pp": _pp(kill_win_pp),
        "fight_loss_pp": _pp(loss_2),
        "at_p_025": row25,
        "breakeven_p": head["breakeven_p_win_fight"],
        "mixture_at_p_025": mix25,
        "mixture_breakeven_p": mix_be,
        "nofarm_contrast_at_p_025": nofarm25,
        "nofarm_contrast_breakeven_p": curves["contrast_nofarm_2kills"]["breakeven_p_win_fight"],
        "dual_tempo_at_p_025": dual25,
        "dual_tempo_breakeven_p": dual_be,
        "dual_tempo_farm_pp": _pp(dual_farm_pp),
        "verdict": (
            f"Under the illustrative priors (cash plus conditional pressure, wiki leave-farm, ±2 kills, diagonal): "
            f"{'LEAVE' if row25['edge_contest_minus_leave_pp'] < 0 else 'CONTEST'} "
            f"at p=0.25"
        ),
    }
    # answers
    laner_lines = []
    for sk in ("one_laner_one_wave", "two_laners_one_wave", "three_laners_one_wave"):
        sc = leave["scenarios"][sk]
        evk = curves[f"leave_{sk}_2kills"]
        e25 = next(r["edge_contest_minus_leave_pp"] for r in evk["curve"] if r["p_win_fight"] == 0.25)
        laner_lines.append(
            f"{sc['n_laners']} laner(s): {sc['gold']:.2f}g → {sc['wr_pp_at_even']:+.2f}pp; "
            f"edge@25%={e25:+.2f}pp; p*≈{evk['breakeven_p_win_fight']:.2f}"
        )
    d["answers"] = {
        "what_is_the_scrap": (
            f"T_pref ≈ {_pp(preferred):+.2f}pp (90g + 8s Touch→plate). "
            "Full T includes 195 XP in joint/upper bounds."
        ),
        "leave_farm_hypothesis": (
            leave["hypothesis"] + " "
            + leave["constants"]["note"] + " "
            + " | ".join(laner_lines)
        ),
        "how_large_is_intrinsic_grub_value": (
            f"T_pref {_pp(preferred):+.2f}pp; gold alone {_pp(b['lower_gold_only_pp']):+.2f}pp; "
            f"joint {_pp(b['central_gold_plus_xp_joint_pp']):+.2f}pp."
        ),
        "leave_farm_gold": (
            f"Headline: {farm_headline['label']} = {farm_headline['gold']:.2f}g → "
            f"{farm_pp:+.2f}pp (grub-era wave 120.67g × 2 + 0.25×120g plate)."
        ),
        "tf_kill_gold": (
            f"Kill bounty {KILL_BOUNTY_EARLY:.0f}g prior. +2/−2 → "
            f"{kill_win_pp:+.2f}pp / {-kill_lose_pp:+.2f}pp."
        ),
        "should_you_take_25_75_to_deny": (
            f"Headline leave-farm ({farm_headline['gold']:.0f}g): at p=25% edge = "
            f"{row25['edge_contest_minus_leave_pp']:+.2f}pp → "
            f"{'LEAVE' if row25['edge_contest_minus_leave_pp'] < 0 else 'CONTEST'}; "
            f"p*≈{head['breakeven_p_win_fight']:.2f}. "
            f"Mixture: {mix25['edge_contest_minus_leave_pp']:+.2f}pp. "
            f"If somehow zero farm: {nofarm25['edge_contest_minus_leave_pp']:+.2f}pp "
            f"(p*≈{curves['contrast_nofarm_2kills']['breakeven_p_win_fight']:.2f}). "
            f"Dual-tempo (own+opp miss): {dual25['edge_contest_minus_leave_pp']:+.2f}pp "
            f"(p*≈{dual_be:.2f})."
        ),
        "item_pace": item_pace["read"],
        "is_ls_right": (
            "On Furia 25/75 under headline leave-farm (wiki waves ± plate): "
            f"{'leave still preferred' if row25['edge_contest_minus_leave_pp'] < 0 else 'contest closer'}. "
            f"T_pref ≈ {_pp(preferred):+.2f}pp is not literal 0.10pp. "
            "Item-pace: gifting scrap can still leave the take side behind on next-item "
            "completion for ~1–3 min because they missed waves."
        ),
        "is_it_decimal_like_ls": (
            f"T_pref ≈ {_pp(preferred):+.2f}pp — not 0.10. "
            f"Take-regime {_pp(contaminated['unique_dpp']):+.2f}pp is a different estimand."
        ),
        "why_not_plus_3_7pp": (
            f"OE 3–0 association {_pp(contaminated['unique_dpp']):+.2f}pp is selection+tempo, not scrap."
        ),
        "certainty_disclaimer": (
            "Leave farm uses wiki gold constants + gold@10 associational map; "
            "p_plate and N laners are scenario inputs; p(fight) is exogenous. "
            "Item costs and contest delay are wiki/structural priors, not OE-logged recalls."
        ),
    }
    b["notes"] = (
        "Cash lower bound = 90g local. The 8s Touch quantity is an undiscounted "
        "first-plate-progress ceiling, not paid gold; 26.11 changes its tick rate. "
        "Leave farm is a scenario input, not an OE-observed counterfactual."
    )
    sample = d.setdefault("sample", {})
    sample["n_fit_gold10"] = logits.get("n")
    sample["n_fit_xp10"] = d.get("logits", {}).get("xp10", {}).get("n")
    sample["n_fit_joint"] = d.get("logits", {}).get("joint_gold_xp", {}).get("n")
    mech = d.setdefault("mechanical_package", {})
    mech["hunger_at_3_stacks"] = True
    mech["xp_to_level_7_to_8"] = XP_TO_LEVEL_7
    mech["wave_gold_avg_early_wiki"] = WAVE_GOLD_AVG_EARLY
    mech["wave_gold_no_cannon"] = WAVE_GOLD_NO_CANNON
    mech["plate_gold"] = PLATE_GOLD
    return d



def render_paper(d: dict[str, Any]) -> str:
    """Working paper regenerated from JSON (source of truth). Never hand-edit numbers here."""
    b = d["intrinsic_bounds_pp_at_even"]
    c = d["contaminated_association_for_contrast"]
    ls = d["ls_furia_scenario"]
    mech = d["mechanical_package"]
    gold = d["component_wr_pp"]["gold_only"]
    joint = d["component_wr_pp"]["joint_gold_xp"]
    head_key = d.get("headline_key", "headline_nofarm_2kills_sym")
    head = d["contest_ev"].get(head_key) or d["contest_ev"]["headline_nofarm_2kills_sym"]
    mix = d["contest_ev"].get("headline_nofarm_2kills_mixture", {})
    recon = d.get("sister_study_reconciliation", {})
    ans = d.get("answers", {})
    xp_lvl = mech.get("xp_to_level_7_to_8", mech.get("xp_to_level_7", XP_TO_LEVEL_7))
    plate = mech.get("plate_gold", PLATE_GOLD)
    plate_hp = mech.get("plate_hp_first_outer", PLATE_HP_FIRST)

    lines = [
        "# Void Grubs scrap proxy and contest sensitivity",
        "",
        f"**Version {d.get('version', '?')}** · Associational unit conversion · OE professional maps",
        "",
        "> Source of truth: `grubs_intrinsic_value.json`. This file is regenerated from JSON.",
        f"> Plate constants: **{plate:g}g / plate**, first outer plate **{plate_hp:g} HP** (wiki).",
        "",
        "---",
        "",
        "## Abstract",
        "",
        "We separate (i) an **associational scrap proxy** for mechanical bumps "
        "(90g, optional XP, Touch→plate gold) mapped through gold@10 / xp@10 logits "
        "from (ii) the **OE take-regime association** of ending 3–0 after early controls.",
        f"Era sample n={d['sample']['n_era_3camp']:,}; "
        f"gold@10 fit n={d['sample'].get('n_fit_gold10', d['logits']['gold10']['n']):,}.",
        f"+90g at even → **{gold['at_even']:.2f}pp**. "
        f"**T_pref** (90g + 8s Touch) → **{b['preferred_gold_plus_brief_burn_pp']:.2f}pp**; "
        f"joint gold+XP → **{joint['at_even_gold_and_xp']:.2f}pp**; "
        f"take-regime contrast → **{c['unique_dpp']:.2f}pp** (not scrap).",
        f"Headline leave-farm: at p=0.25 edge "
        f"**{ls['at_p_025']['edge_contest_minus_leave_pp']:+.2f}pp**, "
        f"p*≈**{ls['breakeven_p']:.2f}**. "
        f"Item-pace / dual-tempo and 2×2 mixture are sensitivities.",
        "",
        "## Estimand (honest)",
        "",
        d.get("estimand", ""),
        "",
        "**T_full** = {90g, 195 XP radius 2000, Touch@3 + Hunger mite}.",
        "**T_pref** (headline) = {90g, brief Touch} — XP is in joint/upper only.",
        "",
        "## Item completion pace",
        "",
        (d.get("item_pace") or {}).get("read", "(re-enrich JSON for item_pace)"),
        "",
        "## Mechanical package",
        "",
        "| Component | Value |",
        "|-----------|------:|",
        f"| Gold (3 × {GOLD_PER_GRUB}) | {mech['gold']} |",
        f"| XP (3 × {XP_PER_GRUB}) | {mech['xp']} |",
        f"| XP toward level 7→8 ({xp_lvl}) | {mech['xp']}/{xp_lvl} = {mech['xp']/xp_lvl:.3f} |",
        f"| Touch tick melee @3 (pre-26.11) | {mech['totv_tick_melee_3stack_pre_26_11']} true / 0.5s |",
        f"| Hunger at 3 stacks | {mech.get('hunger_at_3_stacks', True)} |",
        f"| Plate gold / first-plate HP | {plate:g} / {plate_hp:g} |",
        "",
        "## Results",
        "",
        "| Bound | pp | Contents |",
        "|-------|---:|----------|",
        f"| Lower | **{b['lower_gold_only_pp']:.2f}** | 90g only |",
        f"| T_pref | **{b['preferred_gold_plus_brief_burn_pp']:.2f}** | 90g + 8s Touch |",
        f"| Joint | **{b['central_gold_plus_xp_joint_pp']:.2f}** | 90g + 195 XP |",
        f"| Upper | **{b['upper_joint_plus_wiki_burn_20s_pp']:.2f}** | joint + 20s Touch |",
        f"| Take-regime | **{c['unique_dpp']:.2f}** | blue 3–0 | controls@10 |",
        "",
        "### Headline contest sensitivity",
        "",
        f"- Δ T_pref = **{ls['delta_obj_pp_preferred']:.2f}pp**; farm headline = **0**",
        f"- Loss prior (2 deaths + gift) = **{ls['fight_loss_pp']:.2f}pp**; win +2 kills = **{ls['fight_win_extra_pp']:.2f}pp**",
        f"- At p=0.25: edge **{ls['at_p_025']['edge_contest_minus_leave_pp']:+.2f}pp**; p*≈**{head['breakeven_p_win_fight']:.2f}**",
    ]
    if mix:
        m25 = ls.get("mixture_at_p_025", {})
        lines += [
            f"- 2×2 mixture (P(secure|win)=0.85, P(secure|lose)=0.05): "
            f"edge **{m25.get('edge_contest_minus_leave_pp', 'n/a')}pp**; "
            f"p*≈**{ls.get('mixture_breakeven_p', mix.get('breakeven_p_win_fight'))}**",
        ]
    f25 = ls.get("farm_sensitivity_at_p_025", {})
    if f25:
        fg = d["opportunity_gold"]["packages"]["preferred_waves_plus_camp"]["gold"]
        lines += [
            f"- Leave-farm sensitivity (+{fg:.0f}g): "
            f"edge **{f25.get('edge_contest_minus_leave_pp')}pp** (not headline)",
        ]
    lines += [
        "",
        "### Sister OE contest study (different estimand)",
        "",
        recon.get("note", "n/a"),
    ]
    if recon.get("available"):
        lines.append(
            f"At p=0.25 vs leave_mix: **{recon.get('at_p_025_edge_vs_leave_mix_pp')}pp** "
            f"({recon.get('at_p_025_verdict_vs_leave')}); "
            f"p*_leave_mix≈**{recon.get('breakeven_p_vs_leave_mix')}**."
        )
    lines += ["", "## Answers", ""]
    for k, v in ans.items():
        lines += [f"**{k}.** {v}", ""]
    lines += [
        "## Limitations",
        "",
        "- Gold/XP→WR maps are associational, not experimental grants; gold@10 is post-spawn.",
        "- T_pref excludes XP by definition; do not quote it as the price of T_full.",
        "- Farm and kill nets are structural priors; assist gold and smite are stylized in the mixture.",
        "- Exogenous p; one-objective / one-fight lottery.",
        "- Do not coach from this note or the sister OE contest EV alone.",
        "",
        "Reproduce:",
        "```bash",
        "python3 -m lol_kills.research.grubs_intrinsic_value",
        "python3 -m lol_kills.research.grubs_intrinsic_pdf",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    # This report is a data product, not a hand-maintained narrative.  Always
    # refit so a corrected patch parser or refreshed OE snapshot cannot leave a
    # stale numerical paper behind.
    out = enrich_report_v4(build_report())
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    summary = render_paper(out)
    header = (
        "# Intrinsic value — numeric summary (auto-generated)\n\n"
        f"> Full working paper: `{OUT_PAPER.name}` (also regenerated from JSON)\n\n"
        "---\n\n"
    )
    OUT_MD.write_text(header + summary)
    OUT_PAPER.write_text(summary)
    print(f"[intrinsic] wrote {OUT_JSON}")
    print(f"[intrinsic] wrote {OUT_MD}")
    print(f"[intrinsic] wrote {OUT_PAPER} (synced from JSON)")
    b = out["intrinsic_bounds_pp_at_even"]
    ls = out["ls_furia_scenario"]
    print(
        f"[intrinsic] T_pref={b['preferred_gold_plus_brief_burn_pp']:.3f}pp "
        f"gold={b['lower_gold_only_pp']:.3f}pp "
        f"upper={b['upper_joint_plus_wiki_burn_20s_pp']:.3f}pp "
        f"contaminated={out['contaminated_association_for_contrast']['unique_dpp']:.3f}pp"
    )
    print(
        f"[intrinsic] headline p=25% edge={ls['at_p_025']['edge_contest_minus_leave_pp']:.2f}pp "
        f"→ {ls['verdict']} | p*={ls['breakeven_p']:.3f}"
    )


if __name__ == "__main__":
    main()
