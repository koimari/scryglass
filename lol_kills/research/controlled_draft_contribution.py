"""Isolate a Draft Score contribution from paired model predictions.

The caller must score two versions of the same pre-match row:

* the observed draft;
* a role-matched champion swap between Blue and Red.

Global team strength, player strength, momentum, competition, date, and side
must be identical in both rows. Draft-dependent champion, player-champion,
ally, enemy, patch, and phase features may change after the swap.
"""

from __future__ import annotations

import math
import hashlib
import json
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "scryglass:controlled-draft-contribution:v1"
INTERVENTION_RECEIPT_SCHEMA = "scryglass:role-matched-draft-swap:v1"
SIDES = ("Blue", "Red")
ROLES = ("top", "jng", "mid", "bot", "sup")
FIXED_FIELDS = (
    "game_uid",
    "date",
    "side",
    "position",
    "playername",
    "teamname",
    "league",
)


class ControlledDraftContributionError(ValueError):
    """Raised when a paired Draft intervention is invalid."""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _normalized_slots(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> dict[tuple[str, str], dict[str, str]]:
    if len(rows) != 10:
        raise ControlledDraftContributionError(f"{label} draft must have ten rows")
    slots: dict[tuple[str, str], dict[str, str]] = {}
    for raw in rows:
        row = {field: str(raw.get(field, "")).strip() for field in FIXED_FIELDS}
        row["champion"] = str(raw.get("champion", "")).strip()
        side = row["side"].title()
        role = row["position"].lower()
        role = {"jungle": "jng", "support": "sup"}.get(role, role)
        row["side"] = side
        row["position"] = role
        if side not in SIDES or role not in ROLES:
            raise ControlledDraftContributionError(f"{label} draft has an invalid slot")
        if any(not row[field] for field in (*FIXED_FIELDS, "champion")):
            raise ControlledDraftContributionError(f"{label} draft has an empty field")
        key = (side, role)
        if key in slots:
            raise ControlledDraftContributionError(f"{label} draft repeats a slot")
        slots[key] = row
    expected = {(side, role) for side in SIDES for role in ROLES}
    if set(slots) != expected:
        raise ControlledDraftContributionError(f"{label} draft has incomplete slots")
    champions = [row["champion"].casefold() for row in slots.values()]
    if len(set(champions)) != 10:
        raise ControlledDraftContributionError(f"{label} draft repeats a champion")
    return slots


def validate_role_matched_champion_swap(
    *,
    observed_rows: Sequence[Mapping[str, Any]],
    swapped_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Prove that a paired input changes only role-matched champions."""

    observed = _normalized_slots(observed_rows, label="observed")
    swapped = _normalized_slots(swapped_rows, label="swapped")
    for key, observed_row in observed.items():
        swapped_row = swapped[key]
        if any(observed_row[field] != swapped_row[field] for field in FIXED_FIELDS):
            raise ControlledDraftContributionError(
                "paired draft changed a fixed control field"
            )
        side, role = key
        other_side = "Red" if side == "Blue" else "Blue"
        if swapped_row["champion"] != observed[(other_side, role)]["champion"]:
            raise ControlledDraftContributionError(
                "paired draft is not an exact role-matched champion swap"
            )

    ordered_observed = [
        observed[(side, role)] for side in SIDES for role in ROLES
    ]
    ordered_swapped = [swapped[(side, role)] for side in SIDES for role in ROLES]
    fixed_controls = [
        {field: row[field] for field in FIXED_FIELDS}
        for row in ordered_observed
    ]
    receipt = {
        "schema_version": INTERVENTION_RECEIPT_SCHEMA,
        "method": "role_matched_champion_swap",
        "observed_rows_sha256": _canonical_sha256(ordered_observed),
        "swapped_rows_sha256": _canonical_sha256(ordered_swapped),
        "fixed_controls_sha256": _canonical_sha256(fixed_controls),
        "fixed_fields": list(FIXED_FIELDS),
        "changed_field": "champion",
        "slots": len(ordered_observed),
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def isolate_controlled_draft_contribution(
    *,
    observed_blue_win_probability: float,
    swapped_draft_blue_win_probability: float,
) -> dict[str, Any]:
    """Return the antisymmetric Draft contribution from a paired intervention.

    The average logit is the fixed-strength expectation. Half of the logit
    difference is the Draft contribution. Adding the contribution reconstructs
    the observed prediction. Subtracting it reconstructs the swapped-draft
    prediction.
    """

    probabilities = (
        observed_blue_win_probability,
        swapped_draft_blue_win_probability,
    )
    if not all(math.isfinite(value) for value in probabilities):
        raise ControlledDraftContributionError("paired probabilities are not finite")
    if not all(0.0 < value < 1.0 for value in probabilities):
        raise ControlledDraftContributionError(
            "paired probabilities must be strictly between zero and one"
        )

    observed_logit = _logit(observed_blue_win_probability)
    swapped_logit = _logit(swapped_draft_blue_win_probability)
    strength_logit = 0.5 * (observed_logit + swapped_logit)
    draft_logit = 0.5 * (observed_logit - swapped_logit)
    isolated_probability = _sigmoid(draft_logit)
    edge_percentage_points = 100.0 * (isolated_probability - 0.5)

    return {
        "schema_version": SCHEMA_VERSION,
        "method": "role_matched_champion_swap",
        "model_units": draft_logit,
        "edge_percentage_points": edge_percentage_points,
        "stronger_draft": (
            "Blue" if draft_logit > 0.0 else "Red" if draft_logit < 0.0 else "Even"
        ),
        "isolated_blue_draft_probability": isolated_probability,
        "fixed_strength_blue_win_probability": _sigmoid(strength_logit),
        "reconstructed_observed_blue_win_probability": _sigmoid(
            strength_logit + draft_logit
        ),
        "reconstructed_swapped_blue_win_probability": _sigmoid(
            strength_logit - draft_logit
        ),
        "controls_held_fixed": [
            "team_rating",
            "player_rating",
            "rating_uncertainty",
            "momentum",
            "competition_context",
            "match_context",
            "blue_side",
        ],
        "draft_features_allowed_to_change": [
            "champion_atoms",
            "player_champion_atoms",
            "ally_atom_interactions",
            "enemy_atom_interactions",
            "patch_atom_history",
            "prematch_phase_curve",
        ],
    }
