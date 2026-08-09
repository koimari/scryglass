"""Fail-closed atom features for champion tier-list matchup models.

This module is an adapter over the validated LCC ``AtomBridge``.  It does
not fit ratings and it does not turn atom evidence into an empirical win
rate.  It emits a fixed, versioned feature vector that a downstream model
can use with explicit availability masks.

The feature registry is deliberately static.  Family presence uses the
registered LCC family order.  LCC attribute ratings use fixed ranges.  Each
ontology simplex omits its first registered label as the deterministic
reference category.  The omitted category is not replaced by zero.

An explicit patch request requires an exact, time-safe snapshot mapping.  A
bridge's current patch marker is provenance for the loaded snapshot.  It is
not an inferred historical mapping.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
import re
from collections.abc import Mapping
from typing import Any

from lol_kills.v2.champions.atoms.consume import AtomBridge
from lol_kills.v2.champions.atoms.schema import (
    BRIDGE_SCHEMA_ID,
    BRIDGE_VERSION,
    CHAMPION_ATOM_FAMILIES,
    DIMENSION_LABELS,
    canonical_sha256,
)


ATOM_FEATURE_SCHEMA_ID = "scryglass.tierlists.atom-matchup-features.v1"
ATOM_FEATURE_SCHEMA_VERSION = "atom-matchup-features-v1"

# These names and ranges are part of the public feature contract.  The LCC
# source uses 0..100 for ability reliance and 0..3 for the other ratings.
LCC_ATTRIBUTE_RANGES: tuple[tuple[str, float, float], ...] = (
    ("abilityReliance", 0.0, 100.0),
    ("control", 0.0, 3.0),
    ("damage", 0.0, 3.0),
    ("difficulty", 0.0, 3.0),
    ("mobility", 0.0, 3.0),
    ("toughness", 0.0, 3.0),
    ("utility", 0.0, 3.0),
)

# The first label is the reference category.  DIMENSION_LABELS is a tuple in
# the validated bridge schema, so this choice is deterministic and stable.
ONTOLOGY_REFERENCE_LABELS: dict[str, str] = {
    dimension: labels[0] for dimension, labels in DIMENSION_LABELS.items()
}


def _family_feature_name(family: str) -> str:
    return f"family_presence.{family}"


def _attribute_feature_name(attribute: str) -> str:
    return f"lcc_attribute.{attribute}"


def _ontology_feature_name(dimension: str, label: str) -> str:
    return f"ontology.{dimension}.{label}"


FAMILY_FEATURE_NAMES: tuple[str, ...] = tuple(
    _family_feature_name(family) for family in CHAMPION_ATOM_FAMILIES
)
ATTRIBUTE_FEATURE_NAMES: tuple[str, ...] = tuple(
    _attribute_feature_name(attribute) for attribute, _lower, _upper in LCC_ATTRIBUTE_RANGES
)
ONTOLOGY_FEATURE_NAMES: tuple[str, ...] = tuple(
    _ontology_feature_name(dimension, label)
    for dimension, labels in DIMENSION_LABELS.items()
    for label in labels
    if label != ONTOLOGY_REFERENCE_LABELS[dimension]
)
FEATURE_ORDER: tuple[str, ...] = (
    *FAMILY_FEATURE_NAMES,
    *ATTRIBUTE_FEATURE_NAMES,
    *ONTOLOGY_FEATURE_NAMES,
)

_PATCH_RE = re.compile(r"^\d{1,2}\.\d{1,2}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class AtomMatchupFeatureError(ValueError):
    """Raised when atom features cannot be resolved safely."""


@dataclass(frozen=True)
class ExactAtomSnapshotMapping:
    """Evidence that a bridge snapshot is exact and safe for one patch.

    ``snapshot_patch`` must equal ``requested_patch``.  The bridge digest
    binds the mapping to the validated bridge instance used by the resolver.
    ``time_safe`` is an assertion from the snapshot producer and must be
    explicitly true.
    """

    requested_patch: str
    snapshot_patch: str
    snapshot_as_of: str
    bridge_artifact_sha256: str
    time_safe: bool = True


def _schema_payload() -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    for family in CHAMPION_ATOM_FAMILIES:
        features.append(
            {
                "name": _family_feature_name(family),
                "source": "family_presence",
                "source_key": family,
                "value_type": "binary",
            }
        )
    for attribute, lower, upper in LCC_ATTRIBUTE_RANGES:
        features.append(
            {
                "name": _attribute_feature_name(attribute),
                "source": "lcc_attribute_ratings",
                "source_key": attribute,
                "value_type": "normalized_float",
                "normalization": {"lower": lower, "upper": upper},
            }
        )
    for dimension, labels in DIMENSION_LABELS.items():
        reference = ONTOLOGY_REFERENCE_LABELS[dimension]
        for label in labels:
            if label == reference:
                continue
            features.append(
                {
                    "name": _ontology_feature_name(dimension, label),
                    "source": "ontology_prior",
                    "source_key": f"{dimension}.{label}",
                    "value_type": "probability",
                    "reference_label": reference,
                }
            )
    return {
        "schema_id": ATOM_FEATURE_SCHEMA_ID,
        "version": ATOM_FEATURE_SCHEMA_VERSION,
        "feature_order": list(FEATURE_ORDER),
        "features": features,
        "availability": {
            "representation": "parallel boolean map keyed by feature_order",
            "missing_value": None,
            "missing_policy": "fail_closed_no_zero_imputation",
        },
        "ontology_reference_labels": dict(ONTOLOGY_REFERENCE_LABELS),
        "pair": {
            "operation": "left_minus_right",
            "canonical_key": "lexicographically_sorted_champion_ids",
            "orientation": "+1 for canonical order, -1 for reversed order",
        },
        "patch_policy": {
            "explicit_request": "requires_exact_time_safe_snapshot_mapping",
            "patch_match": "requested_patch_equals_snapshot_patch",
            "bridge_binding": "mapping_bridge_artifact_sha256_equals_loaded_bridge",
        },
    }


_FEATURE_SCHEMA_PAYLOAD = _schema_payload()
FEATURE_SCHEMA_SHA256 = canonical_sha256(_FEATURE_SCHEMA_PAYLOAD)


def feature_schema() -> dict[str, Any]:
    """Return the immutable feature registry as a detached dictionary."""

    return deepcopy(_FEATURE_SCHEMA_PAYLOAD)


def _require_finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AtomMatchupFeatureError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise AtomMatchupFeatureError(f"{label} must be finite")
    return number


def _require_utc_timestamp(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AtomMatchupFeatureError(f"{label} must be a UTC timestamp")
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AtomMatchupFeatureError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise AtomMatchupFeatureError(f"{label} must include a UTC offset")
    return text


def _mapping_dict(
    mapping: ExactAtomSnapshotMapping | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(mapping, ExactAtomSnapshotMapping):
        return asdict(mapping)
    if not isinstance(mapping, Mapping):
        raise AtomMatchupFeatureError("snapshot mapping must be an object")
    return dict(mapping)


class AtomMatchupFeatureResolver:
    """Resolve fixed champion and ordered pair atom features."""

    def __init__(self, bridge: AtomBridge) -> None:
        if not isinstance(bridge, AtomBridge):
            raise AtomMatchupFeatureError("bridge must be a validated AtomBridge")
        self.bridge = bridge

    @property
    def feature_schema_sha256(self) -> str:
        return FEATURE_SCHEMA_SHA256

    @property
    def snapshot_patch(self) -> str | None:
        provenance = self.bridge.provenance
        patch = provenance.get("data_patch")
        return patch if isinstance(patch, str) and patch else None

    def feature_schema(self) -> dict[str, Any]:
        """Return the registered schema and its deterministic hash."""

        schema = feature_schema()
        schema["schema_sha256"] = self.feature_schema_sha256
        return schema

    def provenance(self) -> dict[str, Any]:
        """Return bridge-bound provenance for every resolved vector."""

        return {
            "feature_schema_id": ATOM_FEATURE_SCHEMA_ID,
            "feature_schema_version": ATOM_FEATURE_SCHEMA_VERSION,
            "feature_schema_sha256": self.feature_schema_sha256,
            "source_kind": "validated_atom_bridge",
            "bridge_schema_id": BRIDGE_SCHEMA_ID,
            "bridge_version": BRIDGE_VERSION,
            "bridge_artifact_sha256": self.bridge.artifact_sha256,
            "bridge_generated_at": self.bridge.generated_at,
            "bridge_provenance": deepcopy(dict(self.bridge.provenance)),
        }

    def resolve(
        self,
        champion_id: str,
        *,
        requested_patch: str | None = None,
        snapshot_mapping: ExactAtomSnapshotMapping | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resolve one champion vector with explicit availability masks."""

        self._validate_patch_request(requested_patch, snapshot_mapping)
        profile = self.bridge.profile(champion_id)
        if profile is None:
            raise AtomMatchupFeatureError(f"bridge has no champion profile for {champion_id}")

        values: dict[str, float | None] = {name: None for name in FEATURE_ORDER}
        availability: dict[str, bool] = {name: False for name in FEATURE_ORDER}

        self._resolve_family_presence(profile, values, availability)
        self._resolve_attributes(profile, values, availability)
        self._resolve_ontology(profile, values, availability)

        return {
            "champion_id": champion_id,
            "features": values,
            "availability": availability,
            "feature_order": list(FEATURE_ORDER),
            "feature_schema_sha256": self.feature_schema_sha256,
            "snapshot": {
                "bridge_artifact_sha256": self.bridge.artifact_sha256,
                "bridge_generated_at": self.bridge.generated_at,
                "bridge_data_patch": self.snapshot_patch,
                "requested_patch": requested_patch,
            },
            "provenance": self.provenance(),
        }

    # Alias with a descriptive name for downstream callers.
    resolve_champion = resolve

    def resolve_pair(
        self,
        left_champion_id: str,
        right_champion_id: str,
        *,
        requested_patch: str | None = None,
        snapshot_mapping: ExactAtomSnapshotMapping | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resolve antisymmetric features for an ordered champion pair.

        The pair key is canonical and independent of argument order.  The
        signed feature values are always ``left - right``.  Therefore the
        reverse argument order keeps the canonical key and reverses every
        available feature sign.
        """

        left = self.resolve(
            left_champion_id,
            requested_patch=requested_patch,
            snapshot_mapping=snapshot_mapping,
        )
        right = self.resolve(
            right_champion_id,
            requested_patch=requested_patch,
            snapshot_mapping=snapshot_mapping,
        )

        canonical_ids = tuple(sorted((left_champion_id, right_champion_id)))
        orientation = 1 if (left_champion_id, right_champion_id) == canonical_ids else -1
        values: dict[str, float | None] = {}
        availability: dict[str, bool] = {}
        for name in FEATURE_ORDER:
            available = bool(left["availability"][name] and right["availability"][name])
            availability[name] = available
            if not available:
                values[name] = None
                continue
            left_value = left["features"][name]
            right_value = right["features"][name]
            if left_value is None or right_value is None:
                raise AtomMatchupFeatureError(
                    f"feature mask/value mismatch for pair feature {name}"
                )
            values[name] = float(left_value) - float(right_value)

        return {
            "left_champion_id": left_champion_id,
            "right_champion_id": right_champion_id,
            "canonical_pair": {
                "champion_ids": list(canonical_ids),
                "key": f"{canonical_ids[0]}|{canonical_ids[1]}",
                "orientation": orientation,
            },
            "features": values,
            "availability": availability,
            "feature_order": list(FEATURE_ORDER),
            "feature_schema_sha256": self.feature_schema_sha256,
            "snapshot": left["snapshot"],
            "provenance": self.provenance(),
        }

    def _validate_patch_request(
        self,
        requested_patch: str | None,
        snapshot_mapping: ExactAtomSnapshotMapping | Mapping[str, Any] | None,
    ) -> None:
        if requested_patch is None:
            if snapshot_mapping is not None:
                raise AtomMatchupFeatureError(
                    "snapshot mapping requires an explicit requested_patch"
                )
            return
        if not isinstance(requested_patch, str) or not _PATCH_RE.fullmatch(requested_patch):
            raise AtomMatchupFeatureError(f"invalid requested patch: {requested_patch!r}")
        if snapshot_mapping is None:
            raise AtomMatchupFeatureError(
                "exact time-safe atom snapshot mapping is required for an explicit patch"
            )

        mapping = _mapping_dict(snapshot_mapping)
        required = {
            "requested_patch",
            "snapshot_patch",
            "snapshot_as_of",
            "bridge_artifact_sha256",
            "time_safe",
        }
        missing = sorted(required - set(mapping))
        if missing:
            raise AtomMatchupFeatureError(
                f"snapshot mapping is missing required field(s): {', '.join(missing)}"
            )
        if mapping["requested_patch"] != requested_patch:
            raise AtomMatchupFeatureError("snapshot mapping does not match requested patch")
        if mapping["snapshot_patch"] != requested_patch:
            raise AtomMatchupFeatureError(
                "snapshot patch must exactly equal requested patch"
            )
        if mapping["time_safe"] is not True:
            raise AtomMatchupFeatureError("snapshot mapping is not time-safe")
        if mapping["bridge_artifact_sha256"] != self.bridge.artifact_sha256:
            raise AtomMatchupFeatureError(
                "snapshot mapping is bound to a different atom bridge artifact"
            )
        digest = mapping["bridge_artifact_sha256"]
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise AtomMatchupFeatureError("snapshot mapping bridge digest is invalid")
        _require_utc_timestamp(mapping["snapshot_as_of"], "snapshot_as_of")

    @staticmethod
    def _resolve_family_presence(
        profile: Mapping[str, Any],
        values: dict[str, float | None],
        availability: dict[str, bool],
    ) -> None:
        presence = profile.get("family_presence")
        if not isinstance(presence, Mapping):
            return
        for family in CHAMPION_ATOM_FAMILIES:
            if family not in presence:
                continue
            value = presence[family]
            if not isinstance(value, bool):
                raise AtomMatchupFeatureError(
                    f"family_presence.{family} must be boolean when present"
                )
            name = _family_feature_name(family)
            values[name] = 1.0 if value else 0.0
            availability[name] = True

    @staticmethod
    def _resolve_attributes(
        profile: Mapping[str, Any],
        values: dict[str, float | None],
        availability: dict[str, bool],
    ) -> None:
        ratings = profile.get("lcc_attribute_ratings")
        if not isinstance(ratings, Mapping):
            return
        for attribute, lower, upper in LCC_ATTRIBUTE_RANGES:
            if attribute not in ratings:
                continue
            number = _require_finite_number(
                ratings[attribute], f"lcc_attribute_ratings.{attribute}"
            )
            if number < lower or number > upper:
                raise AtomMatchupFeatureError(
                    f"lcc_attribute_ratings.{attribute} is outside [{lower}, {upper}]"
                )
            name = _attribute_feature_name(attribute)
            values[name] = (number - lower) / (upper - lower)
            availability[name] = True

    @staticmethod
    def _resolve_ontology(
        profile: Mapping[str, Any],
        values: dict[str, float | None],
        availability: dict[str, bool],
    ) -> None:
        prior = profile.get("ontology_prior")
        if not isinstance(prior, Mapping):
            return
        for dimension, labels in DIMENSION_LABELS.items():
            entry = prior.get(dimension)
            if not isinstance(entry, Mapping) or entry.get("status") != "available":
                continue
            supplied = entry.get("labels")
            if not isinstance(supplied, Mapping):
                continue
            # A simplex is available only when every registered probability is
            # present.  A partial simplex would otherwise hide missing data.
            if set(supplied) != set(labels):
                continue
            probabilities: dict[str, float] = {}
            for label in labels:
                probability = _require_finite_number(
                    supplied[label], f"ontology_prior.{dimension}.{label}"
                )
                if probability < 0.0 or probability > 1.0:
                    raise AtomMatchupFeatureError(
                        f"ontology_prior.{dimension}.{label} is outside [0, 1]"
                    )
                probabilities[label] = probability
            if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-3):
                raise AtomMatchupFeatureError(
                    f"ontology_prior.{dimension} probabilities do not form a simplex"
                )
            reference = ONTOLOGY_REFERENCE_LABELS[dimension]
            for label in labels:
                if label == reference:
                    continue
                name = _ontology_feature_name(dimension, label)
                values[name] = probabilities[label]
                availability[name] = True


__all__ = [
    "ATOM_FEATURE_SCHEMA_ID",
    "ATOM_FEATURE_SCHEMA_VERSION",
    "ATTRIBUTE_FEATURE_NAMES",
    "AtomMatchupFeatureError",
    "AtomMatchupFeatureResolver",
    "ExactAtomSnapshotMapping",
    "FEATURE_ORDER",
    "FEATURE_SCHEMA_SHA256",
    "FAMILY_FEATURE_NAMES",
    "LCC_ATTRIBUTE_RANGES",
    "ONTOLOGY_FEATURE_NAMES",
    "ONTOLOGY_REFERENCE_LABELS",
    "feature_schema",
]
