"""Tests for the L5 policy / lineup-synergy estimand opener (v1)."""

from __future__ import annotations

import json
import math

import pytest

from lol_kills.v2.champions.atoms.consume import AtomBridge
from lol_kills.v2.ratings.team.estimands_v1 import (
    COMPOSITION_DIMENSION_COUNT,
    EstimandError,
    composition_vector,
    identification_audit,
    lineup_synergy_estimand,
    opened_estimands,
    policy_weight_estimand,
)
from lol_kills.v2.ratings.team.model import aggregate_team_rating

ROLES = ("top", "jungle", "mid", "bot", "support")

DISTINCT_ROSTER = {
    "top": "riot:champion:266",      # Aatrox
    "jungle": "riot:champion:64",    # Lee Sin
    "mid": "riot:champion:115",      # Ziggs
    "bot": "riot:champion:222",      # Jinx
    "support": "riot:champion:412",  # Thresh
}
FLAT_RESOURCES = {role: 0.2 for role in ROLES}
REF_WEIGHTS = {role: 0.2 for role in ROLES}
RESOURCE_MIX = {"top": 0.22, "jungle": 0.18, "mid": 0.24, "bot": 0.26, "support": 0.10}
PLAYER_SPAN = {"top": 0.3, "jungle": 0.1, "mid": 0.5, "bot": 0.4, "support": 0.2}


@pytest.fixture(scope="module")
def bridge() -> AtomBridge:
    return AtomBridge.load()


def test_composition_vector_shape_and_family_order(bridge) -> None:
    vector = composition_vector(bridge, "riot:champion:266")
    assert len(vector) == COMPOSITION_DIMENSION_COUNT
    assert all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in vector)
    # families sum to 1 (with zero-fill), type mix sums to 1
    assert abs(sum(vector[:6]) - 1.0) < 1e-6
    assert abs(vector[6] + vector[7] - 1.0) < 1e-6


def test_composition_vector_unknown_champion_fails_closed(bridge) -> None:
    with pytest.raises(EstimandError):
        composition_vector(bridge, "riot:champion:99999")


def test_policy_weights_shrink_toward_reference(bridge) -> None:
    policy = policy_weight_estimand(RESOURCE_MIX, REF_WEIGHTS)
    weights = policy["weights"]
    assert set(weights) == set(ROLES)
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert weights["bot"] > weights["support"]
    assert policy["within_roster_variation"] > 0.0


def test_lineup_synergy_is_orthogonalized_and_shrunken(bridge) -> None:
    policy = policy_weight_estimand(RESOURCE_MIX, REF_WEIGHTS)
    synergy = lineup_synergy_estimand(
        bridge, DISTINCT_ROSTER, policy["weights"], PLAYER_SPAN
    )
    assert math.isfinite(synergy["gamma_hat"])
    assert synergy["gamma_sd"] > 0.0
    assert synergy["posterior_dependence"] < 0.5
    assert len(synergy["gamma_interval_95"]) == 2


def test_strong_case_opens_estimands_and_weak_case_fails_closed(bridge) -> None:
    strong = opened_estimands(
        bridge, DISTINCT_ROSTER, RESOURCE_MIX, REF_WEIGHTS, PLAYER_SPAN
    )
    assert strong is not None
    assert strong["audit"]["verdict"] == "strong"
    assert strong["policy"]["available"] is True
    assert strong["lineup_synergy"]["available"] is True
    assert math.isfinite(strong["lineup_synergy"]["gamma_hat"])

    weak = opened_estimands(
        bridge, {role: "riot:champion:266" for role in ROLES},
        FLAT_RESOURCES, REF_WEIGHTS, {role: 0.2 for role in ROLES},
    )
    assert weak is None  # fail closed: no fabrication of separate facts


def test_identification_audit_gates_report(bridge) -> None:
    policy = policy_weight_estimand(RESOURCE_MIX, REF_WEIGHTS)
    synergy = lineup_synergy_estimand(bridge, DISTINCT_ROSTER, policy["weights"], PLAYER_SPAN)
    audit = identification_audit(bridge, DISTINCT_ROSTER, policy, synergy)
    assert audit["verdict"] in {"strong", "weak"}
    assert set(audit["gates"]) == {
        "within_roster_policy_variation",
        "orthogonalization_residual",
        "posterior_dependence",
        "source_removal",
        "design_rank",
    }


def test_aggregate_team_rating_wiring_opens_components(bridge) -> None:
    """End-to-end: estimand_inputs -> TeamRating carries the components."""
    roster = {
        "active": True,
        "ambiguous": False,
        "as_of": "2026-08-01T00:00:00Z",
        "custom": False,
        "effective_at": "2026-08-01T00:00:00Z",
        "fresh": True,
        "hypothetical": False,
        "league_id": "SYN-LEC",
        "official": True,
        "organization_id": "SYN-ORG-A",
        "substitute": False,
        "players": [
            {"player_id": f"p-{i}", "role": role, "league_id": "SYN-LEC",
             "scope": "regional", "posterior_mean": mean, "active": True,
             "as_of": "2026-08-01T00:00:00Z"}
            for i, (role, mean) in enumerate(
                zip(ROLES, (0.3, 0.1, 0.5, 0.4, 0.2)))
        ],
    }
    covariance = [[0.01 if i == j else 0.0 for j in range(5)] for i in range(5)]
    from lol_kills.v2.ratings.team.model import ExactRoster, aggregate_team_rating as aggregate
    exact = ExactRoster.from_mapping(roster)
    opened = aggregate_team_rating(
        exact, covariance, scope="regional",
        estimand_inputs={
            "bridge": bridge,
            "roster_champions": DISTINCT_ROSTER,
            "resource_share": RESOURCE_MIX,
            "reference_weights": REF_WEIGHTS,
            "player_span": PLAYER_SPAN,
        },
    ).to_dict()
    assert opened["lineup_synergy_component"] is not None
    assert opened["policy_component"] is not None
    assert opened["estimand"]["gamma_q"] is not None
    assert opened["estimand"]["policy_q"] is not None
    assert opened["component_availability"]["lineup_synergy"]["available"] is True
    assert opened["component_availability"]["policy"]["available"] is True
    assert opened["reference_convention"]["status"] == "estimated_with_uncertainty"
    # claim ceilings stay fail-closed
    assert opened["rank_eligibility"] is False
    assert opened["development_only"] is True

    closed = aggregate_team_rating(
        exact, covariance, scope="regional"
    ).to_dict()
    assert closed["lineup_synergy_component"] is None
    assert closed["policy_component"] is None
    assert closed["component_availability"]["lineup_synergy"]["available"] is False
