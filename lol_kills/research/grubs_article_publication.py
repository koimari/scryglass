#!/usr/bin/env python3
"""Build the one public Void Grubs article artifact.

This module is intentionally self-contained and deterministic. It is the only
writer for ``grubs_article_contest_ev.json``; broader grubs studies may consume
its calculations, but they must not publish a second article-shaped artifact.

The article estimand is a stylized opportunity-cost comparison converted
through a side-neutral gold-at-10 associational logit. It is not an identified
shotcalling policy or a causal map-win estimate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from lol_kills.etl.paths import MODELS_DIR

SCHEMA_VERSION = "scryglass.grubs.article.v1"
PUBLICATION_ID = "void-grubs-contest-or-leave.patch-26.11"
MECHANICS_PATCH = "26.11+"

# Frozen article conversion. These coefficients are an associational
# side-neutral gold@10 calibration, not a causal or draft-true win probability.
GOLD10_INTERCEPT = 0.1611182873782888
GOLD10_COEF = 0.000666860223609559

# Patch 26.11+ current-mechanics reference.
GRUB_CASH_GOLD = 90.0
BRIEF_TOUCH_SECONDS = 8
TOUCH_TRUE_DAMAGE = 256
FIRST_PLATE_HP = 900
FIRST_PLATE_GOLD = 120
BRIEF_TOUCH_PROGRESS_GOLD_EQUIVALENT = round(
    TOUCH_TRUE_DAMAGE / FIRST_PLATE_HP * FIRST_PLATE_GOLD,
    2,
)
OBJECTIVE_GOLD_EQUIVALENT = round(
    GRUB_CASH_GOLD + BRIEF_TOUCH_PROGRESS_GOLD_EQUIVALENT,
    2,
)

TWO_WAVE_LEAVE_FARM_GOLD = 241.33
FIGHT_SWING_GOLD = 600.0
SECURE_IF_WIN = 1.0
SECURE_IF_LOSE = 0.0

DEFAULT_OUTPUT = MODELS_DIR / "grubs_article_contest_ev.json"
CURVE_PROBABILITIES = (
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.60,
    0.70,
    0.75,
)
LEAVE_FARM_PACKAGES = (
    ("no_farm", 0.0),
    ("one_wave", 120.67),
    ("two_waves", TWO_WAVE_LEAVE_FARM_GOLD),
)
GOLD_LEADS = (
    0.0,
    -500.0,
    -1000.0,
    -2000.0,
    500.0,
    1000.0,
    1183.0,
    1200.0,
)


def _sigmoid(value: float) -> float:
    bounded = max(-35.0, min(35.0, float(value)))
    return 1.0 / (1.0 + math.exp(-bounded))


def side_neutral_win_probability(gold: float) -> float:
    """Associational map-win conversion at an own-team gold lead."""

    linear = GOLD10_COEF * float(gold)
    return 0.5 * (
        _sigmoid(GOLD10_INTERCEPT + linear)
        + _sigmoid(-GOLD10_INTERCEPT + linear)
    )


def article_p_star_at_gold(
    baseline_gold: float,
    *,
    leave_farm_gold: float = TWO_WAVE_LEAVE_FARM_GOLD,
    objective_gold: float = OBJECTIVE_GOLD_EQUIVALENT,
) -> float:
    """Return the fight-win probability where contest and leave EV are equal."""

    p_leave = side_neutral_win_probability(
        baseline_gold + leave_farm_gold - objective_gold
    )
    p_win = side_neutral_win_probability(
        baseline_gold + objective_gold + FIGHT_SWING_GOLD
    )
    p_loss = side_neutral_win_probability(
        baseline_gold - objective_gold - FIGHT_SWING_GOLD
    )
    denominator = p_win - p_loss
    if denominator <= 1e-12:
        raise ValueError("article contest states do not identify a finite p_star")
    root = (p_leave - p_loss) / denominator
    if not 0.0 <= root <= 1.0:
        raise ValueError(f"article p_star falls outside probability units: {root}")
    return root


def _round(value: float, digits: int) -> float:
    rounded = round(float(value), digits)
    return 0.0 if rounded == 0.0 else rounded


def _curve_point(p_win_fight: float) -> dict[str, Any]:
    baseline = side_neutral_win_probability(0.0)
    p_win = side_neutral_win_probability(
        OBJECTIVE_GOLD_EQUIVALENT + FIGHT_SWING_GOLD
    )
    p_loss = side_neutral_win_probability(
        -OBJECTIVE_GOLD_EQUIVALENT - FIGHT_SWING_GOLD
    )
    p_leave = side_neutral_win_probability(
        TWO_WAVE_LEAVE_FARM_GOLD - OBJECTIVE_GOLD_EQUIVALENT
    )
    contest_pp = 100.0 * (
        p_win_fight * p_win + (1.0 - p_win_fight) * p_loss - baseline
    )
    leave_pp = 100.0 * (p_leave - baseline)
    edge_pp = contest_pp - leave_pp
    return {
        "p_win_fight": p_win_fight,
        "ev_contest_pp": _round(contest_pp, 2),
        "ev_leave_pp": _round(leave_pp, 2),
        "edge_contest_minus_leave_pp": _round(edge_pp, 2),
        "model_preference": "CONTEST" if edge_pp >= 0.0 else "LEAVE",
    }


def _leave_farm_row(label: str, leave_farm_gold: float) -> dict[str, Any]:
    parity = article_p_star_at_gold(
        0.0,
        leave_farm_gold=leave_farm_gold,
    )
    ahead = article_p_star_at_gold(
        1183.0,
        leave_farm_gold=leave_farm_gold,
    )
    return {
        "label": label,
        "leave_farm_gold": leave_farm_gold,
        "p_star_at_parity": _round(parity, 6),
        "p_star_at_parity_pct": _round(100.0 * parity, 2),
        "p_star_at_B_plus_1183": _round(ahead, 6),
        "p_star_at_B_plus_1183_pct": _round(100.0 * ahead, 2),
    }


def _gold_lead_row(baseline_gold: float) -> dict[str, Any]:
    p_star = article_p_star_at_gold(baseline_gold)
    return {
        "B_gold": baseline_gold,
        "leave_farm_gold": TWO_WAVE_LEAVE_FARM_GOLD,
        "objective_gold": OBJECTIVE_GOLD_EQUIVALENT,
        "p_star": _round(p_star, 6),
        "p_star_pct": _round(100.0 * p_star, 2),
    }


def _build_unchecked() -> dict[str, Any]:
    p_star = article_p_star_at_gold(0.0)
    curve = [_curve_point(p) for p in CURVE_PROBABILITIES]
    at_fifty = next(row for row in curve if row["p_win_fight"] == 0.5)
    return {
        "schema_version": SCHEMA_VERSION,
        "publication_id": PUBLICATION_ID,
        "estimand": "article_opportunity_cost_sensitivity",
        "units": (
            "edge values are percentage points of associational map-win "
            "probability; p_star values are probabilities"
        ),
        "mechanics": {
            "patch": MECHANICS_PATCH,
            "grub_cash_gold": GRUB_CASH_GOLD,
            "brief_touch_seconds": BRIEF_TOUCH_SECONDS,
            "touch_true_damage": TOUCH_TRUE_DAMAGE,
            "first_plate_hp": FIRST_PLATE_HP,
            "first_plate_gold": FIRST_PLATE_GOLD,
            "brief_touch_progress_gold_equivalent": (
                BRIEF_TOUCH_PROGRESS_GOLD_EQUIVALENT
            ),
            "objective_gold_equivalent": OBJECTIVE_GOLD_EQUIVALENT,
            "valuation": (
                "upper_bound_plate_progress_equivalent_not_guaranteed_gold"
            ),
            "hunger_mite_included": False,
        },
        "model": {
            "conversion": "side_neutral_gold10_associational_logit",
            "intercept": GOLD10_INTERCEPT,
            "gold_coefficient": GOLD10_COEF,
            "fight_swing_gold": FIGHT_SWING_GOLD,
            "secure_if_win": SECURE_IF_WIN,
            "secure_if_lose": SECURE_IF_LOSE,
        },
        "reference_knobs": {
            "baseline_gold": 0.0,
            "objective_gold_equivalent": OBJECTIVE_GOLD_EQUIVALENT,
            "two_wave_leave_farm_gold": TWO_WAVE_LEAVE_FARM_GOLD,
            "fight_swing_gold": FIGHT_SWING_GOLD,
        },
        "p_star": _round(p_star, 12),
        "p_star_pct": _round(100.0 * p_star, 2),
        "edge_at_50_pp": at_fifty["edge_contest_minus_leave_pp"],
        "interpretation": (
            "Within this fixed sensitivity comparison, contest and two-wave "
            "leave have equal modeled value at p_star. This is not an "
            "identified action threshold or universal shotcalling rule."
        ),
        "curve": curve,
        "by_leave_farm_F": [
            _leave_farm_row(label, gold)
            for label, gold in LEAVE_FARM_PACKAGES
        ],
        "by_precontest_gold_B_two_wave_leave": [
            _gold_lead_row(gold) for gold in GOLD_LEADS
        ],
        "limitations": [
            (
                "Gold-at-10 to map-win is an associational conversion, not a "
                "causal effect."
            ),
            (
                "Fight-win probability is exogenous rather than estimated from "
                "champions, position, vision, cooldowns, or player form."
            ),
            (
                "The 34.13g Touch term is an upper-bound plate-progress "
                "equivalent, not guaranteed immediate gold."
            ),
            (
                "The comparison does not identify a live action policy and "
                "should not be used as a universal contest threshold."
            ),
        ],
    }


def _first_mismatch(
    actual: Any,
    expected: Any,
    path: str = "$",
) -> str | None:
    if type(actual) is not type(expected):
        return (
            f"{path}: expected {type(expected).__name__}, "
            f"got {type(actual).__name__}"
        )
    if isinstance(expected, dict):
        actual_keys = set(actual)
        expected_keys = set(expected)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            return (
                f"{path}: schema keys differ "
                f"(missing={missing}, extra={extra})"
            )
        for key in expected:
            mismatch = _first_mismatch(
                actual[key],
                expected[key],
                f"{path}.{key}",
            )
            if mismatch:
                return mismatch
        return None
    if isinstance(expected, list):
        if len(actual) != len(expected):
            return f"{path}: expected {len(expected)} rows, got {len(actual)}"
        for index, expected_item in enumerate(expected):
            mismatch = _first_mismatch(
                actual[index],
                expected_item,
                f"{path}[{index}]",
            )
            if mismatch:
                return mismatch
        return None
    if actual != expected:
        return f"{path}: expected {expected!r}, got {actual!r}"
    return None


def validate_article_publication(document: Mapping[str, Any]) -> None:
    """Fail closed unless schema, mechanics, and every derived value match."""

    expected = _build_unchecked()
    mismatch = _first_mismatch(dict(document), expected)
    if mismatch:
        raise ValueError(f"invalid grubs article publication: {mismatch}")


def build_article_publication() -> dict[str, Any]:
    document = _build_unchecked()
    validate_article_publication(document)
    return document


def canonical_json(document: Mapping[str, Any]) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def write_article_publication(output: Path = DEFAULT_OUTPUT) -> Path:
    """Atomically write the canonical artifact after strict self-validation."""

    document = build_article_publication()
    payload = canonical_json(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def _check_existing(output: Path) -> None:
    loaded = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("grubs article publication must be a JSON object")
    validate_article_publication(loaded)
    expected = canonical_json(loaded)
    actual = output.read_text(encoding="utf-8")
    if actual != expected:
        raise ValueError(
            "grubs article publication is semantically valid but not canonical JSON"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the existing output without rewriting it",
    )
    args = parser.parse_args(argv)
    if args.check:
        _check_existing(args.output)
        print(f"[grubs_article_publication] valid {args.output}")
        return 0
    output = write_article_publication(args.output)
    print(f"[grubs_article_publication] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
