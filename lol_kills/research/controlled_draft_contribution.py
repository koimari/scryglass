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
from typing import Any


SCHEMA_VERSION = "scryglass:controlled-draft-contribution:v1"


class ControlledDraftContributionError(ValueError):
    """Raised when a paired Draft intervention is invalid."""


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
