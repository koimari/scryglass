#!/usr/bin/env python3
"""
Composite draft score by clock bucket (@10 / @15 / @20 / @25) + beatdown roles.

Mike Flores — "Who's the Beatdown?": the side with more early damage must press;
the side with late inevitability must weather and win late. Misassignment = loss.

This is the proper multi-horizon draft score used by the board/live stack:
  - classic champ win_edge (Draft Score v3) as base
  - phase-fitted beatdown / inevitability adjustments from OE
  - per-bucket P(blue) curve + who is the beatdown / control

  from lol_kills.draft_phase_score import draft_score_composite, draft_edge_at_minute
"""

from __future__ import annotations

import json

from lol_kills.draft_archetypes import draft_archetype_features
from lol_kills.draft_score import draft_score, sigmoid
from lol_kills.etl.paths import MODELS_DIR

COEFS_PATH = MODELS_DIR / "draft_phase_beatdown.json"
BUCKETS = (10, 15, 20, 25)

# Flores axes — early damage vs late inevitability
BEATDOWN_WEIGHTS = {
    "early_snowball": 1.00,
    "assassin": 0.85,
    "roam": 0.45,
    "engage": 0.35,
    "skirmisher": 0.25,
    "poke_siege": 0.20,
}
INEVITABILITY_WEIGHTS = {
    "scaling_late": 1.00,
    "hypercarry_adc": 0.90,
    "control_mage": 0.75,
    "teamfight_aoe": 0.55,
    "peel_enchanter": 0.35,
    "splitpush": 0.30,
}

# Fallback phase blend if fit artifact missing (early → late)
_FALLBACK_BLEND = {
    10: {"beatdown": 0.55, "inev": 0.10, "base": 0.90},
    15: {"beatdown": 0.40, "inev": 0.25, "base": 0.90},
    20: {"beatdown": 0.22, "inev": 0.45, "base": 0.90},
    25: {"beatdown": 0.10, "inev": 0.60, "base": 0.90},
}


def _axis(feats: dict[str, float], weights: dict[str, float], side: str) -> float:
    total = 0.0
    for arch, w in weights.items():
        if side == "diff":
            total += w * float(feats.get(f"arch_{arch}_diff", 0.0))
        else:
            total += w * float(feats.get(f"arch_{arch}_{side}", 0.0))
    return total


def _load_coefs() -> dict:
    if not COEFS_PATH.exists():
        return {}
    try:
        return json.loads(COEFS_PATH.read_text())
    except Exception:
        return {}


def _side_powers(blue: list[str], red: list[str]) -> dict:
    feats = draft_archetype_features(blue, red)
    return {
        "feats": feats,
        "beatdown_blue": _axis(feats, BEATDOWN_WEIGHTS, "blue"),
        "beatdown_red": _axis(feats, BEATDOWN_WEIGHTS, "red"),
        "beatdown_diff": _axis(feats, BEATDOWN_WEIGHTS, "diff"),
        "inev_blue": _axis(feats, INEVITABILITY_WEIGHTS, "blue"),
        "inev_red": _axis(feats, INEVITABILITY_WEIGHTS, "red"),
        "inev_diff": _axis(feats, INEVITABILITY_WEIGHTS, "diff"),
    }


def assign_roles(powers: dict) -> dict:
    """
    Flores assignment from relative early damage vs late inevitability.

    Beatdown = side with higher beatdown power (early damage).
    Control  = the other side (must have a path via inevitability, or it's a bad seat).
    """
    bd_b, bd_r = powers["beatdown_blue"], powers["beatdown_red"]
    in_b, in_r = powers["inev_blue"], powers["inev_red"]
    blue_beatdown = bd_b >= bd_r
    beatdown_side = "blue" if blue_beatdown else "red"
    control_side = "red" if blue_beatdown else "blue"

    # Gap magnitudes
    early_gap = abs(bd_b - bd_r)
    late_gap = abs(in_b - in_r)

    # Bad seat: assigned beatdown but opponent has way more inevitability AND
    # you don't actually lead early damage by much → forced wrong role.
    beatdown_inev = in_b if blue_beatdown else in_r
    control_inev = in_r if blue_beatdown else in_b
    beatdown_early = bd_b if blue_beatdown else bd_r
    control_early = bd_r if blue_beatdown else bd_b

    misassign_risk = 0.0
    if beatdown_early < control_early + 0.15 and control_inev > beatdown_inev + 0.4:
        misassign_risk = min(1.0, 0.35 + 0.2 * (control_inev - beatdown_inev))
    # Mirror / mud: both axes tiny
    if early_gap < 0.25 and late_gap < 0.25:
        plan = "mirror_mud"
    elif blue_beatdown:
        plan = "blue_beatdown_red_control"
    else:
        plan = "red_beatdown_blue_control"

    return {
        "beatdown_side": beatdown_side,
        "control_side": control_side,
        "blue_is_beatdown": blue_beatdown,
        "early_gap": round(early_gap, 3),
        "late_gap": round(late_gap, 3),
        "misassign_risk": round(misassign_risk, 3),
        "plan": plan,
        "rule": "Misassignment of Role = Game Loss (Flores)",
    }


def _bucket_logit(
    *,
    t: int,
    base_edge: float,
    powers: dict,
    elo_z: float,
    coefs: dict,
    conf: float,
) -> dict:
    """
    Pure-draft logit at clock bucket t.

    Uses OE gold@t path + win|gold leftovers (not final-win-only — that is
    identical across buckets). Flores: beatdown loads into gold early;
    inevitability shows up in win|gold (and late).
    """
    b = (coefs.get("buckets") or {}).get(str(t)) or {}
    gold_fit = b.get("gold_k") or {}
    wg = b.get("win_given_gold") or {}
    ahead_fit = b.get("ahead") or {}

    bd = powers["beatdown_diff"]
    inev = powers["inev_diff"]

    if gold_fit.get("coef") and wg.get("coef"):
        gcoef = dict(zip(gold_fit.get("feature_names") or ["elo_z", "draft_win_logit", "beatdown_diff", "inev_diff"], gold_fit["coef"]))
        wcoef = dict(zip(wg.get("feature_names") or ["elo_z", "gold_k", "draft_win_logit", "beatdown_diff", "inev_diff"], wg["coef"]))
        # Expected gold (k) from draft axes only
        gold_hat = (
            float(gcoef.get("draft_win_logit", 0.0)) * base_edge
            + float(gcoef.get("beatdown_diff", 0.0)) * bd
            + float(gcoef.get("inev_diff", 0.0)) * inev
        )
        draft_logit = (
            float(wcoef.get("gold_k", 0.55)) * gold_hat
            + float(wcoef.get("draft_win_logit", 0.25)) * base_edge
            + float(wcoef.get("beatdown_diff", 0.0)) * bd
            + float(wcoef.get("inev_diff", 0.0)) * inev
        )
        with_elo_gold = gold_hat + float(gcoef.get("elo_z", 0.0)) * elo_z
        with_elo = (
            float(wg.get("intercept") or 0.0)
            + float(wcoef.get("elo_z", 0.0)) * elo_z
            + float(wcoef.get("gold_k", 0.55)) * with_elo_gold
            + float(wcoef.get("draft_win_logit", 0.25)) * base_edge
            + float(wcoef.get("beatdown_diff", 0.0)) * bd
            + float(wcoef.get("inev_diff", 0.0)) * inev
        )
        method = "oe_gold_path"
        ahead_p = None
        if ahead_fit.get("coef"):
            acoef = dict(zip(ahead_fit.get("feature_names") or list(gcoef), ahead_fit["coef"]))
            ahead_logit = (
                float(ahead_fit.get("intercept") or 0.0)
                + float(acoef.get("draft_win_logit", 0.0)) * base_edge
                + float(acoef.get("beatdown_diff", 0.0)) * bd
                + float(acoef.get("inev_diff", 0.0)) * inev
            )
            ahead_p = sigmoid(ahead_logit * conf)
    else:
        blend = _FALLBACK_BLEND[t]
        draft_logit = (
            blend["base"] * base_edge
            + blend["beatdown"] * bd
            + blend["inev"] * inev
        )
        with_elo = draft_logit + 1.1 * elo_z
        method = "fallback_blend"
        gold_hat = 0.0
        ahead_p = None

    draft_logit *= conf
    p = sigmoid(draft_logit)
    return {
        "minute": t,
        "draft_logit": round(draft_logit, 4),
        "p_blue": round(p, 4),
        "draft_score_blue": round(100.0 * p, 2),
        "draft_score_red": round(100.0 * (1.0 - p), 2),
        "draft_edge": round(100.0 * p - 50.0, 2),
        "win_edge": round(draft_logit, 4),
        "p_blue_with_elo": round(sigmoid(with_elo), 4),
        "expected_gold_k_from_draft": round(gold_hat, 3),
        "p_ahead_from_draft": round(ahead_p, 4) if ahead_p is not None else None,
        "method": method,
    }


def draft_score_composite(
    blue: list[str],
    red: list[str],
    *,
    league: str | None = None,
    elo_diff: float | None = None,
) -> dict:
    """
    Composite draft score: classic v3 + phase buckets + beatdown plan.

    Primary `p_blue_draft` / scores = confidence-weighted mean of @10/@15/@20/@25
    pure-draft probabilities (Elo stripped). Also returns the classic scalar.
    """
    classic = draft_score(blue, red, league=league, elo_diff=elo_diff)
    base_edge = float(classic["components"]["win_edge"])
    conf = float(classic["confidence"])
    powers = _side_powers(blue, red)
    roles = assign_roles(powers)
    coefs = _load_coefs()
    elo_z = float(elo_diff or 0.0) / 400.0

    buckets = {}
    for t in BUCKETS:
        buckets[str(t)] = _bucket_logit(
            t=t,
            base_edge=base_edge,
            powers=powers,
            elo_z=elo_z,
            coefs=coefs,
            conf=conf,
        )

    # Composite = average of bucket pure-draft p (equal weight; OE horizons)
    ps = [buckets[str(t)]["p_blue"] for t in BUCKETS]
    p_comp = sum(ps) / len(ps)
    # Emphasize role clarity: if beatdown is clear, tilt composite toward early bucket
    if roles["early_gap"] >= 0.6:
        p_comp = 0.40 * buckets["10"]["p_blue"] + 0.30 * buckets["15"]["p_blue"] + 0.20 * buckets["20"]["p_blue"] + 0.10 * buckets["25"]["p_blue"]
    elif roles["late_gap"] >= 0.6:
        p_comp = 0.10 * buckets["10"]["p_blue"] + 0.20 * buckets["15"]["p_blue"] + 0.30 * buckets["20"]["p_blue"] + 0.40 * buckets["25"]["p_blue"]

    score_blue = 100.0 * p_comp
    # Curve narrative
    early = buckets["10"]["p_blue"]
    late = buckets["25"]["p_blue"]
    if early - late >= 0.03:
        curve = "frontloaded"  # prefers early
    elif late - early >= 0.03:
        curve = "backloaded"
    else:
        curve = "flat"

    beatdown_name = "blue" if roles["blue_is_beatdown"] else "red"
    control_name = "red" if roles["blue_is_beatdown"] else "blue"
    advice = (
        f"{beatdown_name.upper()} is the beatdown — must convert gold/towers/dragons by ~15–20. "
        f"{control_name.upper()} is control/inevitability — survive early, win through late spikes. "
        f"Curve={curve}."
    )
    if roles["misassign_risk"] >= 0.35:
        advice += " High misassign risk: early tools are weak vs opponent inevitability."

    return {
        "draft_score_blue": round(score_blue, 2),
        "draft_score_red": round(100.0 - score_blue, 2),
        "draft_edge": round(score_blue - (100.0 - score_blue), 2),
        "p_blue_draft": round(p_comp, 4),
        "confidence": classic["confidence"],
        "wr_bump_pp": classic.get("wr_bump_pp"),
        "curve": curve,
        "buckets": buckets,
        "beatdown": {
            **roles,
            "powers": {
                "beatdown_blue": round(powers["beatdown_blue"], 3),
                "beatdown_red": round(powers["beatdown_red"], 3),
                "inev_blue": round(powers["inev_blue"], 3),
                "inev_red": round(powers["inev_red"], 3),
                "beatdown_diff": round(powers["beatdown_diff"], 3),
                "inev_diff": round(powers["inev_diff"], 3),
            },
            "advice": advice,
            "citation": "Mike Flores — Who's the Beatdown? (1999)",
        },
        "classic": {
            "draft_score_blue": classic["draft_score_blue"],
            "draft_score_red": classic["draft_score_red"],
            "p_blue_draft": classic["p_blue_draft"],
            "win_edge": classic["components"]["win_edge"],
            "pace_total_shift": classic["components"]["pace_total_shift"],
        },
        "phase_curve": classic.get("phase_curve"),
        "components": {
            **classic["components"],
            "composite_p_blue": round(p_comp, 4),
            "beatdown_diff": round(powers["beatdown_diff"], 4),
            "inev_diff": round(powers["inev_diff"], 4),
        },
        "calibration": classic.get("calibration"),
        "blue": classic.get("blue"),
        "red": classic.get("red"),
        "note": (
            "Composite Draft Score v4: mean of OE-fitted @10/@15/@20/@25 pure-draft "
            "probabilities + Flores beatdown/control assignment. Not a standalone bet."
        ),
    }


def draft_edge_at_minute(composite: dict, minute: float) -> float:
    """Pick win_edge for live_win from nearest clock bucket."""
    buckets = composite.get("buckets") or {}
    if not buckets:
        return float((composite.get("components") or {}).get("win_edge") or 0.0)
    # If before 10, use 10; after 25, use 25
    if minute < 10:
        t = 10
    elif minute >= 25:
        t = 25
    else:
        # snap down to last passed bucket for "as of now"
        passed = [b for b in BUCKETS if b <= minute]
        t = passed[-1] if passed else 10
    return float(buckets[str(t)]["win_edge"])


def nearest_bucket(minute: float) -> int:
    if minute < 12.5:
        return 10
    if minute < 17.5:
        return 15
    if minute < 22.5:
        return 20
    return 25


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blue", required=True, help="Comma-separated blue champs")
    ap.add_argument("--red", required=True, help="Comma-separated red champs")
    ap.add_argument("--league", default="EWC")
    ap.add_argument("--elo-diff", type=float, default=None)
    args = ap.parse_args()
    blue = [c.strip() for c in args.blue.split(",") if c.strip()]
    red = [c.strip() for c in args.red.split(",") if c.strip()]
    print(json.dumps(draft_score_composite(blue, red, league=args.league, elo_diff=args.elo_diff), indent=2))


if __name__ == "__main__":
    main()
