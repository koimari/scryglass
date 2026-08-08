"""Bounded, lineage-bearing Champion Representation v2 builder.

This module is intentionally not wired into the draft model.  It composes the
existing ontology vector with two optional, frozen empirical snapshots while
keeping all three layers structurally independent.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .catalog import ChampionOntology, canonical_sha256
from .schema import DIMENSION_LABEL_ORDER, ROLES


class ChampionRepresentationError(ValueError):
    """Raised when a representation contract or snapshot fails closed."""


ROLE_ORDER = ("top", "jungle", "mid", "bot", "support")
KIT_SEMANTIC_FEATURE_ORDER = tuple(
    f"{dimension}.{label}" for dimension, label in DIMENSION_LABEL_ORDER
)
RESPONSE_FEATURE_ORDER = ("empirical_residual_mean",)
REPRESENTATION_SCHEMA_ID = "scryglass.champion-representation.v2"
RESPONSE_SNAPSHOT_SCHEMA_ID = "scryglass.champion-response-snapshot.v1"
EMBEDDING_SNAPSHOT_SCHEMA_ID = (
    "scryglass.champion-learned-residual-embedding-snapshot.v1"
)
CONTRACT_SCHEMA_ID = "scryglass.champion-representation-contract.v2"
ALLOWED_EMBEDDING_DIMENSIONS = (2, 4)
DEFAULT_EMBEDDING_DIMENSION = 2
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "lol"
    / "v2"
    / "champions"
    / "champion-representation-contract-v2.json"
)

_RESPONSE_SNAPSHOT_FIELDS = {
    "schema_id",
    "snapshot_id",
    "snapshot_as_of",
    "feature_order",
    "source_sha256",
    "cells",
    "snapshot_sha256",
}
_RESPONSE_CELL_FIELDS = {
    "champion_id",
    "patch_id",
    "role",
    "league_id",
    "status",
    "values",
    "uncertainty",
    "evidence",
}


def _require_object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ChampionRepresentationError(f"{label} must be an object")
    return value


def _require_exact_fields(
    value: Mapping[str, Any],
    required: set[str],
    allowed: set[str],
    label: str,
) -> None:
    missing = required - set(value)
    extra = set(value) - allowed
    if missing:
        raise ChampionRepresentationError(
            f"{label} missing required field(s): {', '.join(sorted(missing))}"
        )
    if extra:
        raise ChampionRepresentationError(
            f"{label} has unknown field(s): {', '.join(sorted(extra))}"
        )


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ChampionRepresentationError(f"{label} must be a non-empty string")
    return value


def _require_hash(value: object, label: str) -> str:
    text = _require_string(value, label)
    if not HASH_RE.fullmatch(text):
        raise ChampionRepresentationError(f"{label} must be a lowercase sha256")
    return text


def _require_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ChampionRepresentationError(f"{label} must be numeric, not boolean")
    number = float(value)
    if not math.isfinite(number):
        raise ChampionRepresentationError(f"{label} must be finite")
    return number


def _require_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ChampionRepresentationError(f"{label} must be an integer greater than zero")
    return value


def _parse_timestamp(value: object, label: str) -> datetime:
    text = _require_string(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ChampionRepresentationError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ChampionRepresentationError(f"{label} must be timezone-aware")
    return parsed


def _snapshot_hash(snapshot: Mapping[str, Any]) -> str:
    unsigned = dict(snapshot)
    submitted = _require_hash(
        unsigned.pop("snapshot_sha256", None),
        "snapshot_sha256",
    )
    digest = canonical_sha256(unsigned)
    if submitted != digest:
        raise ChampionRepresentationError("snapshot_sha256 does not match canonical payload")
    return digest


def load_representation_contract(
    path: Path = DEFAULT_CONTRACT_PATH,
) -> dict[str, Any]:
    """Load and strictly validate the pinned representation contract."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            contract = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ChampionRepresentationError(f"cannot load representation contract: {path}") from exc
    if not isinstance(contract, dict):
        raise ChampionRepresentationError("representation contract must be an object")

    expected = {
        "schema_id": CONTRACT_SCHEMA_ID,
        "representation_schema_id": REPRESENTATION_SCHEMA_ID,
        "response_snapshot_schema_id": RESPONSE_SNAPSHOT_SCHEMA_ID,
        "embedding_snapshot_schema_id": EMBEDDING_SNAPSHOT_SCHEMA_ID,
        "role_order": list(ROLE_ORDER),
        "kit_semantic_feature_order": list(KIT_SEMANTIC_FEATURE_ORDER),
        "response_feature_order": list(RESPONSE_FEATURE_ORDER),
        "embedding": {
            "allowed_dimensions": list(ALLOWED_EMBEDDING_DIMENSIONS),
            "activation_status": "disabled_pending_independent_evidence_registry",
            "allowed_snapshot_cell_statuses": ["learned"],
            "allowed_output_statuses": ["unavailable"],
            "axis_order_rule": "fixed_snapshot_order",
            "derivation": "forbidden_in_representation_builder",
        },
        "response": {
            "allowed_snapshot_cell_statuses": ["observed"],
            "allowed_output_statuses": [
                "observed",
                "unavailable",
                "blocked_missing_semantic_anchor",
            ],
            "snapshot_sha256_required": True,
            "content_addressing_confers_predictive_authority": False,
            "sigma_location": "uncertainty",
            "observation_count_location": "evidence",
        },
        "kit_semantic": {
            "allowed_output_statuses": ["exact", "fallback_prior", "unavailable"],
            "exact_requested_patch_rule": "resolved_patch_id == requested_patch_id",
            "fallback_availability": "available_prior_not_exact",
        },
        "exact_cell_rule": ["champion_id", "patch_id", "role", "league_id"],
        "fallback_rules": {
            "kit_semantic": "ontology_explicit_only_never_cross_role",
            "response": "none",
            "learned_residual_embedding": "none",
        },
        "cutoff_rules": {
            "response": "max_event_at <= snapshot_as_of <= requested_as_of",
            "learned_residual_embedding": (
                "training_cutoff <= snapshot_as_of <= requested_as_of"
            ),
        },
        "canonical_hash": {
            "algorithm": "sha256",
            "serialization": "json_sort_keys_compact_ascii",
            "self_reference": "exclude_representation_sha256",
        },
        "layer_order": [
            "kit_semantic",
            "response",
            "learned_residual_embedding",
        ],
    }
    if contract != expected:
        raise ChampionRepresentationError(
            "unknown or modified champion representation contract"
        )
    if tuple(ROLES) != ROLE_ORDER:
        raise ChampionRepresentationError("ontology role order is incompatible with contract")
    if len(KIT_SEMANTIC_FEATURE_ORDER) != 48:
        raise ChampionRepresentationError("kit semantic order must contain exactly 48 features")
    return contract


def _cell_key(cell: Mapping[str, Any], label: str) -> tuple[str, str, str, str]:
    champion_id = _require_string(cell.get("champion_id"), f"{label}.champion_id")
    patch_id = _require_string(cell.get("patch_id"), f"{label}.patch_id")
    role = _require_string(cell.get("role"), f"{label}.role")
    league_id = _require_string(cell.get("league_id"), f"{label}.league_id")
    if role not in ROLE_ORDER:
        raise ChampionRepresentationError(f"{label}.role is unknown: {role}")
    return champion_id, patch_id, role, league_id


def _validate_response_snapshot(
    snapshot: Mapping[str, Any] | None,
    requested_as_of: datetime,
) -> tuple[dict[tuple[str, str, str, str], dict[str, Any]], dict[str, Any] | None]:
    if snapshot is None:
        return {}, {
            "schema_id": RESPONSE_SNAPSHOT_SCHEMA_ID,
            "snapshot_id": None,
            "snapshot_as_of": None,
            "source_sha256": None,
            "snapshot_sha256": None,
            "content_addressed": False,
            "predictive_authority_status": "unavailable",
            "predictive_eligible": False,
        }
    snapshot = _require_object(snapshot, "response snapshot")
    _require_exact_fields(
        snapshot,
        _RESPONSE_SNAPSHOT_FIELDS,
        _RESPONSE_SNAPSHOT_FIELDS,
        "response snapshot",
    )
    if snapshot["schema_id"] != RESPONSE_SNAPSHOT_SCHEMA_ID:
        raise ChampionRepresentationError("unknown response snapshot schema_id")
    _require_string(snapshot["snapshot_id"], "response snapshot.snapshot_id")
    if snapshot["feature_order"] != list(RESPONSE_FEATURE_ORDER):
        raise ChampionRepresentationError("response snapshot feature_order mismatch")
    _require_hash(snapshot["source_sha256"], "response snapshot.source_sha256")
    snapshot_as_of = _parse_timestamp(
        snapshot["snapshot_as_of"], "response snapshot.snapshot_as_of"
    )
    if snapshot_as_of > requested_as_of:
        raise ChampionRepresentationError("response snapshot is newer than requested_as_of")
    if not isinstance(snapshot["cells"], list):
        raise ChampionRepresentationError("response snapshot.cells must be a list")

    cells: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for index, raw_cell in enumerate(snapshot["cells"]):
        label = f"response snapshot.cells[{index}]"
        cell = _require_object(raw_cell, label)
        _require_exact_fields(cell, _RESPONSE_CELL_FIELDS, _RESPONSE_CELL_FIELDS, label)
        key = _cell_key(cell, label)
        if key in cells:
            raise ChampionRepresentationError(f"duplicate response exact cell: {key}")
        if cell["status"] != "observed":
            raise ChampionRepresentationError(f"{label}.status must be observed")
        if not isinstance(cell["values"], list) or len(cell["values"]) != 1:
            raise ChampionRepresentationError(f"{label}.values must follow response feature_order")
        values = [_require_number(cell["values"][0], f"{label}.values[0]")]
        uncertainty = _require_object(cell["uncertainty"], f"{label}.uncertainty")
        _require_exact_fields(
            uncertainty, {"sigma"}, {"sigma"}, f"{label}.uncertainty"
        )
        sigma = _require_number(uncertainty["sigma"], f"{label}.uncertainty.sigma")
        if sigma <= 0.0:
            raise ChampionRepresentationError(
                f"{label}.uncertainty.sigma must be greater than zero"
            )
        evidence = _require_object(cell["evidence"], f"{label}.evidence")
        _require_exact_fields(
            evidence,
            {"observation_count", "max_event_at"},
            {"observation_count", "max_event_at"},
            f"{label}.evidence",
        )
        count = _require_positive_int(
            evidence["observation_count"], f"{label}.evidence.observation_count"
        )
        max_event_at = _parse_timestamp(
            evidence["max_event_at"], f"{label}.evidence.max_event_at"
        )
        if max_event_at > snapshot_as_of:
            raise ChampionRepresentationError(
                f"{label}.evidence.max_event_at exceeds snapshot_as_of"
            )
        cells[key] = {
            **dict(cell),
            "values": values,
            "uncertainty": {"sigma": sigma},
            "evidence": {
                "observation_count": count,
                "max_event_at": evidence["max_event_at"],
            },
        }
    digest = _snapshot_hash(snapshot)
    lineage = {
        "schema_id": snapshot["schema_id"],
        "snapshot_id": snapshot["snapshot_id"],
        "snapshot_as_of": snapshot["snapshot_as_of"],
        "source_sha256": snapshot["source_sha256"],
        "snapshot_sha256": digest,
        "content_addressed": True,
        "predictive_authority_status": "unavailable",
        "predictive_eligible": False,
    }
    return cells, lineage


def _validate_embedding_snapshot(
    snapshot: Mapping[str, Any] | None,
    requested_as_of: datetime,
    requested_dimension: int,
) -> tuple[
    dict[tuple[str, str, str, str], dict[str, Any]],
    dict[str, Any] | None,
    int,
    list[str],
]:
    if (
        type(requested_dimension) is not int
        or requested_dimension not in ALLOWED_EMBEDDING_DIMENSIONS
    ):
        raise ChampionRepresentationError("embedding_dimension must be 2 or 4")
    if snapshot is not None:
        raise ChampionRepresentationError(
            "learned embedding snapshots are disabled pending an independent evidence registry"
        )
    axis_order = [f"residual_axis_{i + 1}" for i in range(requested_dimension)]
    lineage = {
        "activation_status": "disabled_pending_independent_evidence_registry",
        "snapshot_id": None,
        "predictive_authority_status": "unavailable",
        "predictive_eligible": False,
    }
    return {}, lineage, requested_dimension, axis_order


def _kit_semantic_layer(
    feature: Mapping[str, Any],
    requested_patch_id: str,
) -> dict[str, Any]:
    values = feature.get("vector")
    if not isinstance(values, list) or len(values) != len(KIT_SEMANTIC_FEATURE_ORDER):
        raise ChampionRepresentationError("ontology returned incompatible semantic feature order")
    semantic_values = [
        _require_number(value, f"kit_semantic.values[{index}]")
        for index, value in enumerate(values)
    ]
    coverage = _require_object(feature.get("ontology_coverage"), "ontology_coverage")
    available = bool(coverage.get("has_ontology") and coverage.get("has_role_profile"))
    exact_requested_patch = bool(
        available and feature.get("resolved_patch_id") == requested_patch_id
    )
    status = (
        "exact"
        if exact_requested_patch
        else "fallback_prior"
        if available
        else "unavailable"
    )
    if not available:
        semantic_values = [0.0] * len(KIT_SEMANTIC_FEATURE_ORDER)
    uncertainty_map = feature.get("dimension_uncertainty", {})
    uncertainty_values: list[float | None] = []
    for dimension, label in DIMENSION_LABEL_ORDER:
        value = (
            uncertainty_map.get(dimension, {}).get(label)
            if isinstance(uncertainty_map, Mapping)
            else None
        )
        uncertainty_values.append(
            _require_number(value, f"kit_semantic.uncertainty.{dimension}.{label}")
            if value is not None
            else None
        )
    return {
        "available": available,
        "status": status,
        "exact_requested_patch": exact_requested_patch,
        "feature_order": list(KIT_SEMANTIC_FEATURE_ORDER),
        "values": semantic_values,
        "uncertainty": uncertainty_values,
        "sources": list(feature.get("source_ids", [])),
        "review_summary": dict(feature.get("review_summary", {})),
        "review_coverage": dict(feature.get("review_coverage", {})),
        "fallback_level": feature.get("fallback_level"),
        "resolved_patch_id": feature.get("resolved_patch_id"),
        "ontology_coverage": dict(coverage),
        "issues": list(feature.get("issues", [])),
    }


def _response_layer(
    cell: Mapping[str, Any] | None,
    lineage: Mapping[str, Any] | None,
    *,
    semantic_available: bool,
) -> dict[str, Any]:
    common = {
        "content_addressed": bool(lineage and lineage.get("content_addressed")),
        "predictive_authority_status": "unavailable",
        "predictive_eligible": False,
    }
    if cell is not None and not semantic_available:
        return {
            **common,
            "available": False,
            "status": "blocked_missing_semantic_anchor",
            "feature_order": list(RESPONSE_FEATURE_ORDER),
            "values": [0.0],
            "uncertainty": {"sigma": None},
            "evidence": {"observation_count": 0, "max_event_at": None},
            "snapshot_id": lineage["snapshot_id"] if lineage else None,
        }
    if cell is None:
        return {
            **common,
            "available": False,
            "status": "unavailable",
            "feature_order": list(RESPONSE_FEATURE_ORDER),
            "values": [0.0],
            "uncertainty": {"sigma": None},
            "evidence": {"observation_count": 0, "max_event_at": None},
            "snapshot_id": lineage["snapshot_id"] if lineage else None,
        }
    return {
        **common,
        "available": True,
        "status": "observed",
        "feature_order": list(RESPONSE_FEATURE_ORDER),
        "values": list(cell["values"]),
        "uncertainty": dict(cell["uncertainty"]),
        "evidence": dict(cell["evidence"]),
        "snapshot_id": lineage["snapshot_id"] if lineage else None,
    }


def _embedding_layer(
    lineage: Mapping[str, Any] | None,
    dimension: int,
    axis_order: list[str],
) -> dict[str, Any]:
    return {
        "available": False,
        "status": "unavailable",
        "dimension": dimension,
        "axis_order": axis_order,
        "vector": [0.0] * dimension,
        "reliability": None,
        "time_safe": None,
        "cross_fitted": None,
        "training_cutoff": None,
        "snapshot_id": lineage["snapshot_id"] if lineage else None,
        "activation_status": "disabled_pending_independent_evidence_registry",
        "predictive_authority_status": "unavailable",
        "predictive_eligible": False,
    }


def build_champion_representation_v2(
    *,
    ontology: ChampionOntology,
    champion_id: str,
    patch_id: str,
    league_id: str,
    requested_as_of: str,
    response_snapshot: Mapping[str, Any] | None = None,
    embedding_snapshot: Mapping[str, Any] | None = None,
    embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
) -> dict[str, Any]:
    """Build all five requested-role representations without empirical fallback."""

    if not isinstance(ontology, ChampionOntology):
        raise ChampionRepresentationError("ontology must be a ChampionOntology")
    _require_string(champion_id, "champion_id")
    _require_string(patch_id, "patch_id")
    _require_string(league_id, "league_id")
    requested_dt = _parse_timestamp(requested_as_of, "requested_as_of")
    for label, timestamp in (
        ("ontology.ontology_as_of", ontology.ontology_as_of),
        ("ontology.source_as_of", ontology.source_as_of),
        ("ontology.review_as_of", ontology.as_of),
    ):
        if _parse_timestamp(timestamp, label) > requested_dt:
            raise ChampionRepresentationError(f"{label} exceeds requested_as_of")
    contract = load_representation_contract(contract_path)
    contract_sha256 = canonical_sha256(contract)
    response_cells, response_lineage = _validate_response_snapshot(
        response_snapshot, requested_dt
    )
    _, embedding_lineage, dimension, axis_order = (
        _validate_embedding_snapshot(
            embedding_snapshot, requested_dt, embedding_dimension
        )
    )

    roles: list[dict[str, Any]] = []
    for role in ROLE_ORDER:
        feature = ontology.build_feature_vector(
            champion_id=champion_id,
            role=role,
            patch_id=patch_id,
            league_id=league_id,
        )
        key = (champion_id, patch_id, role, league_id)
        semantic_layer = _kit_semantic_layer(feature, patch_id)
        roles.append(
            {
                "role": role,
                "kit_semantic": semantic_layer,
                "response": _response_layer(
                    response_cells.get(key),
                    response_lineage,
                    semantic_available=semantic_layer["available"],
                ),
                "learned_residual_embedding": _embedding_layer(
                    embedding_lineage,
                    dimension,
                    axis_order,
                ),
            }
        )

    payload: dict[str, Any] = {
        "schema_id": REPRESENTATION_SCHEMA_ID,
        "champion_id": champion_id,
        "requested_patch_id": patch_id,
        "league_id": league_id,
        "requested_as_of": requested_as_of,
        "role_order": list(ROLE_ORDER),
        "roles": roles,
        "lineage": {
            "contract_sha256": contract_sha256,
            "ontology": {
                "snapshot_id": ontology.snapshot_id,
                "ontology_sha256": ontology.ontology_snapshot_hash,
                "source_metadata_sha256": ontology.source_metadata_sha256,
                "reviews_sha256": ontology.review_snapshot_hash,
                "ontology_as_of": ontology.ontology_as_of,
                "source_as_of": ontology.source_as_of,
                "review_as_of": ontology.as_of,
            },
            "response": response_lineage,
            "learned_residual_embedding": embedding_lineage,
        },
    }
    payload["representation_sha256"] = canonical_sha256(payload)
    return payload


def validate_representation_sha256(payload: Mapping[str, Any]) -> bool:
    """Check only the self-excluding checksum, not schema, provenance, or authority."""

    if not isinstance(payload, Mapping):
        return False
    unsigned = dict(payload)
    submitted = unsigned.pop("representation_sha256", None)
    return isinstance(submitted, str) and submitted == canonical_sha256(unsigned)


__all__ = [
    "ALLOWED_EMBEDDING_DIMENSIONS",
    "ChampionRepresentationError",
    "DEFAULT_CONTRACT_PATH",
    "EMBEDDING_SNAPSHOT_SCHEMA_ID",
    "KIT_SEMANTIC_FEATURE_ORDER",
    "REPRESENTATION_SCHEMA_ID",
    "RESPONSE_FEATURE_ORDER",
    "RESPONSE_SNAPSHOT_SCHEMA_ID",
    "ROLE_ORDER",
    "build_champion_representation_v2",
    "load_representation_contract",
    "validate_representation_sha256",
]
