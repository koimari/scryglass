"""Build a descriptive champion similarity library from the validated atom bridge.

The score measures shared role, function, and mechanic structure. It does not
estimate performance, win probability, or draft value.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from lol_kills.v2.champions.atoms.consume import AtomBridge
from lol_kills.v2.champions.atoms.schema import DIMENSIONS


SCHEMA_VERSION = "scryglass:champion-structural-similarity:v1"
MINIMUM_SIMILARITY = 0.80
WEIGHTS = {
    "ontology": 0.45,
    "roles": 0.25,
    "attributes": 0.20,
    "families": 0.10,
}
POSITION_TO_ROLE = {
    "TOP": "top",
    "JUNGLE": "jungle",
    "MIDDLE": "mid",
    "BOTTOM": "bot",
    "SUPPORT": "support",
}
ATTRIBUTE_SCALES = {
    "abilityReliance": 100.0,
    "control": 3.0,
    "damage": 3.0,
    "difficulty": 3.0,
    "mobility": 3.0,
    "toughness": 3.0,
    "utility": 3.0,
}


class StructuralSimilarityError(ValueError):
    """Raised when a validated bridge profile cannot form a safe library."""


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, value))


def _jaccard(left: set[str], right: set[str]) -> float | None:
    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


def _ontology_similarity(left: Mapping[str, Any], right: Mapping[str, Any]) -> float | None:
    scores: list[float] = []
    for dimension in DIMENSIONS:
        left_dimension = left.get(dimension)
        right_dimension = right.get(dimension)
        if not isinstance(left_dimension, Mapping) or not isinstance(right_dimension, Mapping):
            continue
        left_labels = left_dimension.get("labels")
        right_labels = right_dimension.get("labels")
        if not isinstance(left_labels, Mapping) or not isinstance(right_labels, Mapping):
            continue
        labels = set(left_labels) | set(right_labels)
        if not labels:
            continue
        coefficient = sum(
            math.sqrt(max(0.0, float(left_labels.get(label, 0.0))) * max(0.0, float(right_labels.get(label, 0.0))))
            for label in labels
        )
        scores.append(_bounded(coefficient))
    return sum(scores) / len(scores) if scores else None


def _attribute_similarity(left: Mapping[str, Any], right: Mapping[str, Any]) -> float | None:
    scores: list[float] = []
    for name, scale in ATTRIBUTE_SCALES.items():
        left_value = left.get(name)
        right_value = right.get(name)
        if isinstance(left_value, bool) or isinstance(right_value, bool):
            continue
        if not isinstance(left_value, (int, float)) or not isinstance(right_value, (int, float)):
            continue
        scores.append(1.0 - _bounded(abs(float(left_value) - float(right_value)) / scale))
    return sum(scores) / len(scores) if scores else None


def _family_set(profile: Mapping[str, Any]) -> set[str]:
    presence = profile.get("family_presence")
    if not isinstance(presence, Mapping):
        return set()
    return {str(name) for name, included in presence.items() if included is True}


def champion_structural_similarity(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    """Return a symmetric 0..1 structural similarity for two bridge profiles."""

    components = {
        "ontology": _ontology_similarity(
            left.get("ontology_prior") if isinstance(left.get("ontology_prior"), Mapping) else {},
            right.get("ontology_prior") if isinstance(right.get("ontology_prior"), Mapping) else {},
        ),
        "roles": _jaccard(
            {str(value) for value in left.get("lcc_roles", [])},
            {str(value) for value in right.get("lcc_roles", [])},
        ),
        "attributes": _attribute_similarity(
            left.get("lcc_attribute_ratings") if isinstance(left.get("lcc_attribute_ratings"), Mapping) else {},
            right.get("lcc_attribute_ratings") if isinstance(right.get("lcc_attribute_ratings"), Mapping) else {},
        ),
        "families": _jaccard(_family_set(left), _family_set(right)),
    }
    available = [(name, value) for name, value in components.items() if value is not None]
    if not available:
        raise StructuralSimilarityError("champion profiles have no comparable structural fields")
    weight_total = sum(WEIGHTS[name] for name, _ in available)
    return round(sum(WEIGHTS[name] * float(value) for name, value in available) / weight_total, 4)


def _dominant_traits(profile: Mapping[str, Any]) -> list[dict[str, str]]:
    prior = profile.get("ontology_prior")
    if not isinstance(prior, Mapping):
        return []
    traits: list[dict[str, str]] = []
    for dimension in DIMENSIONS:
        value = prior.get(dimension)
        labels = value.get("labels") if isinstance(value, Mapping) else None
        if not isinstance(labels, Mapping) or not labels:
            continue
        label = max(labels, key=lambda item: float(labels[item]))
        traits.append({"dimension": dimension, "label": str(label)})
    return traits


def _public_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    positions = [POSITION_TO_ROLE[value] for value in profile.get("lcc_positions", []) if value in POSITION_TO_ROLE]
    status = profile.get("profile_status")
    if status not in {"family_only", "atom_detail"}:
        raise StructuralSimilarityError("champion profile has an unsupported evidence status")
    return {
        "champion_id": str(profile["champion_id"]),
        "champion": str(profile["display_name"]),
        "positions": positions,
        "roles": [str(value).replace("_", " ").title() for value in profile.get("lcc_roles", [])],
        "profile_status": status,
        "traits": _dominant_traits(profile),
    }


def build_structural_similarity(bridge: AtomBridge) -> dict[str, Any]:
    """Build one compact, global similarity matrix for the public tier artifact."""

    source_profiles: list[Mapping[str, Any]] = []
    public_profiles: list[dict[str, Any]] = []
    for champion_id in bridge.champion_ids():
        profile = bridge.profile(champion_id)
        if not isinstance(profile, Mapping):
            raise StructuralSimilarityError(f"missing bridge profile for {champion_id}")
        source_profiles.append(profile)
        public_profiles.append(_public_profile(profile))

    matrix = [
        [champion_structural_similarity(left, right) for right in source_profiles]
        for left in source_profiles
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "source_atom_bridge_sha256": bridge.artifact_sha256,
        "minimum_similarity": MINIMUM_SIMILARITY,
        "weights": dict(WEIGHTS),
        "champions": public_profiles,
        "similarity": matrix,
    }
