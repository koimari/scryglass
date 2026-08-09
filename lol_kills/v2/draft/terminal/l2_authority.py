"""Strict loader for an external independent L2 Draft Score authority record.

The repository contains no such record today.  This module only defines the
shape and exact-byte checks required before a promotion receipt may name one.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

from lol_kills.v2.data.common import parse_rfc3339


L2_AUTHORITY_SCHEMA_VERSION = "scryglass:draft-terminal-l2-authority-record:v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HASH_FIELDS = {
    "model_artifact_sha256",
    "candidate_registry_sha256",
    "development_evaluation_sha256",
    "l2_contract_sha256",
    "calibration_transform_sha256",
    "reliability_artifact_sha256",
    "replay_parity_evidence_sha256",
    "source_snapshot_sha256",
}
_REQUIRED_KEYS = {
    "schema_version",
    "status",
    "authority_record_id",
    "issued_at",
    "independent_reviewer_id",
    *_HASH_FIELDS,
    "independent_l2_authority",
    "sealed_outer_temporal_holdout_decision",
    "source_snapshot",
    "holdouts",
    "reliability",
    "claim_ceiling",
}


class L2AuthorityRecordError(ValueError):
    """Raised when an external L2 authority record is not admissible."""


def _strict_json(raw: bytes) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise L2AuthorityRecordError(f"authority record contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except L2AuthorityRecordError:
        raise
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise L2AuthorityRecordError("authority record must be strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise L2AuthorityRecordError("authority record must be a JSON object")
    return payload


def _require_hash(record: Mapping[str, Any], field: str) -> None:
    if not isinstance(record.get(field), str) or not _SHA256_RE.fullmatch(record[field]):
        raise L2AuthorityRecordError(f"{field} must be a lowercase SHA-256")


def _require_mapping(record: Mapping[str, Any], field: str, keys: set[str]) -> Mapping[str, Any]:
    value = record.get(field)
    if not isinstance(value, Mapping) or set(value) != keys:
        raise L2AuthorityRecordError(f"{field} keys do not match the frozen authority contract")
    return value


def validate_l2_authority_record(
    record: Mapping[str, Any],
    *,
    expected_bindings: Mapping[str, str] | None = None,
) -> None:
    """Validate structure, passed gates, and optional exact local bindings."""

    if not isinstance(record, Mapping):
        raise L2AuthorityRecordError("authority record must be a mapping")
    if set(record) != _REQUIRED_KEYS:
        missing = sorted(_REQUIRED_KEYS - set(record))
        extra = sorted(set(record) - _REQUIRED_KEYS)
        detail = []
        if missing:
            detail.append(f"missing {','.join(missing)}")
        if extra:
            detail.append(f"unexpected {','.join(extra)}")
        raise L2AuthorityRecordError("authority record keys do not match the frozen contract (" + "; ".join(detail) + ")")
    if record["schema_version"] != L2_AUTHORITY_SCHEMA_VERSION or record["status"] != "approved":
        raise L2AuthorityRecordError("authority record is not an approved v1 record")
    for field in ("authority_record_id", "independent_reviewer_id"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise L2AuthorityRecordError(f"{field} must be a non-empty string")
    try:
        parse_rfc3339(record["issued_at"])
    except (TypeError, ValueError) as exc:
        raise L2AuthorityRecordError("issued_at must be an RFC-3339 timestamp") from exc
    for field in _HASH_FIELDS:
        _require_hash(record, field)
    if record["independent_l2_authority"] is not True:
        raise L2AuthorityRecordError("independent_l2_authority must be true")
    if record["sealed_outer_temporal_holdout_decision"] != "passed":
        raise L2AuthorityRecordError("sealed_outer_temporal_holdout_decision must be passed")
    source_snapshot = _require_mapping(
        record,
        "source_snapshot",
        {"availability_status", "participant_cluster_status", "series_grouped"},
    )
    if source_snapshot["availability_status"] != "verified_preevent":
        raise L2AuthorityRecordError("source snapshot availability is not verified pre-event")
    if source_snapshot["participant_cluster_status"] != "team_or_series_available" or source_snapshot["series_grouped"] is not True:
        raise L2AuthorityRecordError("source snapshot does not provide the required neutral dependence structure")
    holdouts = _require_mapping(
        record,
        "holdouts",
        {"future_patch", "league", "international_event_or_meta", "roster_change", "sparse_or_new_champion"},
    )
    for field, value in holdouts.items():
        if field == "roster_change":
            if value != "not_required_for_neutral":
                raise L2AuthorityRecordError("roster_change holdout is not applicable to the neutral authority scope")
        elif value != "passed":
            raise L2AuthorityRecordError("all required neutral holdouts must be passed")
    reliability = _require_mapping(
        record,
        "reliability",
        {"validation_gate_passed", "probability_wording_approved", "baseline_support_verified", "dependence_support_verified", "interval_coverage_verified"},
    )
    if any(value is not True for value in reliability.values()):
        raise L2AuthorityRecordError("all reliability gates must be true")
    claim_ceiling = _require_mapping(
        record,
        "claim_ceiling",
        {"descriptive_pre_map_association", "causal_draft_effect", "recommendation", "betting"},
    )
    if claim_ceiling["descriptive_pre_map_association"] is not True or any(
        claim_ceiling[field] is not False for field in ("causal_draft_effect", "recommendation", "betting")
    ):
        raise L2AuthorityRecordError("authority record claim ceiling is too broad")
    if expected_bindings is not None:
        for field in (
            "candidate_registry_sha256",
            "development_evaluation_sha256",
            "l2_contract_sha256",
        ):
            if record[field] != expected_bindings.get(field):
                raise L2AuthorityRecordError(f"authority record does not bind {field}")
        expected_model = expected_bindings.get("model_artifact_sha256")
        if expected_model is not None and record["model_artifact_sha256"] != expected_model:
            raise L2AuthorityRecordError("authority record does not bind model_artifact_sha256")


def authority_record_payload_sha256(raw: bytes) -> str:
    if not isinstance(raw, bytes) or not raw:
        raise L2AuthorityRecordError("authority record bytes must be non-empty")
    return hashlib.sha256(raw).hexdigest()


def load_l2_authority_record(raw: bytes) -> dict[str, Any]:
    record = _strict_json(raw)
    validate_l2_authority_record(record)
    return record


__all__ = [
    "L2_AUTHORITY_SCHEMA_VERSION",
    "L2AuthorityRecordError",
    "authority_record_payload_sha256",
    "load_l2_authority_record",
    "validate_l2_authority_record",
]
