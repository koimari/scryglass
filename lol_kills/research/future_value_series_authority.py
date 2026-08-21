"""Fail-closed audit for whole-series identity in future-value evidence.

The accepted future-value census binds map identities and source-file hashes.
It does not bind a source-observed series identifier.  This module records
that boundary in a small, immutable audit.  It also records the older proxy
artifact and its limited Leaguepedia checks without promoting either one.

The audit is outcome-free.  It does not infer a series from map order, team
names, dates, tournament labels, or a proxy cluster.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import json
import re

from lol_kills.v2.tierlists.accepted_census import canonical_game_ids, identity_sha256


SCHEMA_VERSION = "scryglass:future-value-series-authority-audit:v1"
TARGET_PROXY_MAP_COUNT = 2_788
AUTHORITY = {
    "research_only": True,
    "authoritative_series": False,
    "tournament_boundary": False,
    "public": False,
    "promotion": False,
    "deployment": False,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_CROSSWALK_RECEIPT_SCHEMA_VERSION = (
    "scryglass:verified-oe-leaguepedia-series-crosswalk-receipt:v1"
)
_CROSSWALK_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "authority",
        "artifact",
        "crosswalk_sha256",
        "source_receipt_sha256",
        "source_identity_sha256",
        "accepted_game_count",
        "accepted_game_identity_sha256",
        "assignment_count",
        "assignment_sha256",
        "mapped_game_count",
        "mapped_game_identity_sha256",
        "mapped_game_ids",
        "receipt_sha256",
    }
)


class SeriesAuthorityAuditError(ValueError):
    """Raised when an audit receipt is malformed or has changed."""


def canonical_json_bytes(value: object) -> bytes:
    """Return the repository JSON form used for content hashes."""

    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    """Hash a JSON value after canonicalization."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_record(path: Path | str, *, locator: str | None = None) -> dict[str, Any]:
    """Describe a local artifact without retaining its contents."""

    artifact_path = Path(path)
    raw = artifact_path.read_bytes()
    return {
        "bytes": len(raw),
        "locator": locator or artifact_path.as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _hash_matches(payload: Mapping[str, Any], field: str) -> bool:
    claimed = payload.get(field)
    if not isinstance(claimed, str) or _SHA256_RE.fullmatch(claimed) is None:
        return False
    body = dict(payload)
    body.pop(field, None)
    return canonical_sha256(body) == claimed.lower()


def _valid_file_record(record: Mapping[str, Any] | None) -> bool:
    if not isinstance(record, Mapping):
        return False
    value = str(record.get("sha256") or "")
    locator = record.get("path") or record.get("locator")
    if not isinstance(locator, str) or not locator.strip():
        return False
    return (
        _SHA256_RE.fullmatch(value) is not None
        and isinstance(record.get("bytes"), int)
        and not isinstance(record.get("bytes"), bool)
        and int(record["bytes"]) > 0
    )


def _read_verified_json_file(
    record: Mapping[str, Any] | None,
    *,
    label: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Read and verify one caller-described JSON artifact.

    A locator is evidence only after it resolves to a regular, non-symlink
    file.  The bytes and digest in the caller record must match the bytes read
    from that file.  The returned record is the normalized on-disk binding.
    """

    if not _valid_file_record(record):
        return None, None
    assert record is not None
    raw_locator = record.get("path") or record.get("locator")
    if not isinstance(raw_locator, str) or not raw_locator.strip():
        return None, None
    candidate = Path(raw_locator).expanduser()
    if candidate.is_symlink():
        return None, None
    try:
        path = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return None, None
    if path.is_symlink() or not path.is_file():
        return None, None
    try:
        raw = path.read_bytes()
    except OSError:
        return None, None
    digest = hashlib.sha256(raw).hexdigest()
    if int(record["bytes"]) != len(raw) or str(record["sha256"]).lower() != digest:
        return None, None
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, None
    if not isinstance(parsed, Mapping):
        return None, None
    return dict(parsed), {
        "path": str(path),
        "bytes": len(raw),
        "sha256": digest,
        "label": label,
    }


def _payloads_match(left: Mapping[str, Any] | None, right: Mapping[str, Any] | None) -> bool:
    if left is None or right is None:
        return False
    try:
        return canonical_sha256(left) == canonical_sha256(right)
    except (TypeError, ValueError):
        return False


def _crosswalk_payload_is_verified(payload: Mapping[str, Any] | None) -> bool:
    """Run the crosswalk module's complete structural verifier."""

    if payload is None:
        return False
    try:
        from lol_kills.research.oe_leaguepedia_series_crosswalk import (
            CrosswalkError,
            verify_crosswalk,
        )

        verify_crosswalk(payload)
    except (CrosswalkError, ImportError, TypeError, ValueError):
        return False
    return True


def _crosswalk_receipt_is_verified(
    payload: Mapping[str, Any] | None,
    *,
    expected_file_sha256: str | None,
    verified_file: Mapping[str, Any] | None,
) -> bool:
    """Verify the independent receipt schema, digest, and file hash."""

    if payload is None or set(payload) != set(_CROSSWALK_RECEIPT_FIELDS):
        return False
    if payload.get("schema_version") != _CROSSWALK_RECEIPT_SCHEMA_VERSION:
        return False
    if payload.get("status") != "verified_research_only":
        return False
    authority = payload.get("authority")
    if not isinstance(authority, Mapping):
        return False
    if set(authority) != {
        "research_only",
        "public",
        "authoritative_series",
        "promotion",
        "deployment",
    }:
        return False
    if (
        authority.get("research_only") is not True
        or authority.get("public") is not False
        or authority.get("promotion") is not False
        or authority.get("deployment") is not False
        or not isinstance(authority.get("authoritative_series"), bool)
    ):
        return False
    if not _hash_matches(payload, "receipt_sha256"):
        return False
    expected = str(expected_file_sha256 or "").lower()
    if _SHA256_RE.fullmatch(expected) is None or verified_file is None:
        return False
    return str(verified_file.get("sha256") or "").lower() == expected


def _as_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _as_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _contains_key(value: Any, wanted: set[str]) -> bool:
    if isinstance(value, Mapping):
        if any(str(key).lower() in wanted for key in value):
            return True
        return any(_contains_key(item, wanted) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_key(item, wanted) for item in value)
    return False


def _canonical_ids(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        values = tuple(str(item) for item in value)
        canonical = tuple(canonical_game_ids(values))
    except (TypeError, ValueError):
        return None
    return canonical if values == canonical else None


def _copy_counts(value: Any, keys: Sequence[str]) -> dict[str, int | None]:
    mapping = value if isinstance(value, Mapping) else {}
    return {key: _as_count(mapping.get(key)) for key in keys}


def _crosswalk_summary(
    crosswalk: Mapping[str, Any] | None,
    crosswalk_receipt: Mapping[str, Any] | None,
    *,
    source_receipt: Mapping[str, Any],
    accepted_ids: tuple[str, ...] | None,
    crosswalk_artifact_file: Mapping[str, Any] | None,
    crosswalk_receipt_file: Mapping[str, Any] | None,
    expected_crosswalk_receipt_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Summarize an optional bridge without treating it as source authority.

    A complete bridge must carry a full accepted census, an independent
    receipt, an outcome-free assignment for every map, and the source hashes.
    The normal audit has no bridge.  This path keeps future reruns explicit.
    """

    artifact_present = isinstance(crosswalk, Mapping)
    receipt_present = isinstance(crosswalk_receipt, Mapping)
    accepted_count = len(accepted_ids) if accepted_ids is not None else None
    assignments = crosswalk.get("assignments") if artifact_present else None
    assignment_rows = (
        list(assignments)
        if isinstance(assignments, Sequence)
        and not isinstance(assignments, (str, bytes, bytearray))
        else []
    )
    assignment_ids = [
        str(row.get("oe_game_id"))
        for row in assignment_rows
        if isinstance(row, Mapping) and row.get("oe_game_id") is not None
    ]
    assignment_ids_are_full = (
        accepted_ids is not None
        and len(assignment_ids) == len(set(assignment_ids))
        and tuple(sorted(assignment_ids)) == tuple(sorted(accepted_ids))
    )
    assignment_outcome_free = all(
        isinstance(row, Mapping) and row.get("outcome_used") is False
        for row in assignment_rows
    )
    source_binding = crosswalk.get("source_binding") if artifact_present else None
    coverage = crosswalk.get("coverage") if artifact_present else None
    receipt_source_hash = (
        crosswalk_receipt.get("source_receipt_sha256")
        if receipt_present
        else None
    )
    source_hash = source_receipt.get("receipt_sha256")
    crosswalk_source_identity = (
        crosswalk_receipt.get("source_identity_sha256") if receipt_present else None
    )
    accepted_source_identity = source_receipt.get("source_identity_sha256")
    crosswalk_accepted_count = (
        _as_count(crosswalk_receipt.get("accepted_game_count"))
        if receipt_present
        else None
    )
    source_census_matches = (
        receipt_source_hash == source_hash
        and crosswalk_source_identity == accepted_source_identity
        and crosswalk_accepted_count == accepted_count
    )
    artifact_payload, verified_artifact_file = _read_verified_json_file(
        crosswalk_artifact_file,
        label="leaguepedia crosswalk artifact",
    )
    receipt_payload, verified_receipt_file = _read_verified_json_file(
        crosswalk_receipt_file,
        label="leaguepedia crosswalk receipt",
    )
    crosswalk_artifact_file_verified = verified_artifact_file is not None
    crosswalk_receipt_file_verified = verified_receipt_file is not None
    crosswalk_artifact_payload_matches = _payloads_match(crosswalk, artifact_payload)
    crosswalk_receipt_payload_matches = _payloads_match(
        crosswalk_receipt, receipt_payload
    )
    crosswalk_artifact_schema_verified = _crosswalk_payload_is_verified(
        artifact_payload
    )
    crosswalk_receipt_schema_verified = _crosswalk_receipt_is_verified(
        receipt_payload,
        expected_file_sha256=expected_crosswalk_receipt_file_sha256,
        verified_file=verified_receipt_file,
    )
    crosswalk_hash_valid = (
        artifact_present
        and _hash_matches(crosswalk, "crosswalk_sha256")
        and artifact_payload is not None
        and _hash_matches(artifact_payload, "crosswalk_sha256")
    )
    crosswalk_receipt_hash_valid = (
        receipt_present
        and _hash_matches(crosswalk_receipt, "receipt_sha256")
        and receipt_payload is not None
        and _hash_matches(receipt_payload, "receipt_sha256")
    )
    receipt_crosswalk_hash_matches = (
        artifact_present
        and receipt_present
        and crosswalk_receipt.get("crosswalk_sha256")
        == crosswalk.get("crosswalk_sha256")
        and isinstance(artifact_payload, Mapping)
        and isinstance(receipt_payload, Mapping)
        and receipt_payload.get("crosswalk_sha256")
        == artifact_payload.get("crosswalk_sha256")
    )
    receipt_artifact_binding_matches = False
    receipt_artifact = receipt_payload.get("artifact") if receipt_payload else None
    if isinstance(receipt_artifact, Mapping) and verified_artifact_file is not None:
        artifact_path_matches = False
        raw_receipt_path = receipt_artifact.get("path")
        if isinstance(raw_receipt_path, str) and raw_receipt_path.strip():
            try:
                artifact_path_matches = (
                    Path(raw_receipt_path).expanduser().resolve(strict=True)
                    == Path(str(verified_artifact_file["path"]))
                )
            except (OSError, RuntimeError):
                artifact_path_matches = False
        receipt_artifact_binding_matches = (
            artifact_path_matches
            and set(receipt_artifact) == {"path", "bytes", "sha256"}
            and receipt_artifact.get("bytes")
            == verified_artifact_file.get("bytes")
            and receipt_artifact.get("sha256") == verified_artifact_file.get("sha256")
        )
    mapped_identity_hash = (
        identity_sha256(tuple(sorted(assignment_ids))) if assignment_ids else None
    )
    receipt_mapped_identity_matches = (
        receipt_present
        and crosswalk_receipt.get("mapped_game_identity_sha256") == mapped_identity_hash
    )
    accepted_identity_hash = identity_sha256(accepted_ids) if accepted_ids else None
    receipt_accepted_identity_matches = (
        receipt_present
        and crosswalk_receipt.get("accepted_game_identity_sha256")
        == accepted_identity_hash
    )
    artifact_assignment_hash = (
        crosswalk.get("assignment_sha256") if artifact_present else None
    )
    receipt_assignment_binding_matches = (
        receipt_present
        and crosswalk_receipt.get("assignment_count") == len(assignment_ids)
        and crosswalk_receipt.get("mapped_game_count") == len(assignment_ids)
        and crosswalk_receipt.get("assignment_sha256") == artifact_assignment_hash
        and crosswalk_receipt.get("mapped_game_ids") == sorted(assignment_ids)
    )
    bridge_authority = crosswalk.get("authority") if artifact_present else None
    receipt_authority = (
        crosswalk_receipt.get("authority") if receipt_present else None
    )
    artifact_authoritative_series = (
        isinstance(bridge_authority, Mapping)
        and bridge_authority.get("authoritative_series") is True
    )
    receipt_authoritative_series = (
        isinstance(receipt_authority, Mapping)
        and receipt_authority.get("authoritative_series") is True
    )
    bridge_authority_safe = (
        isinstance(bridge_authority, Mapping)
        and bridge_authority.get("research_only") is True
        and bridge_authority.get("public") is False
        and bridge_authority.get("promotion") is False
        and bridge_authority.get("deployment") is False
    )
    receipt_authority_safe = (
        isinstance(receipt_authority, Mapping)
        and receipt_authority.get("research_only") is True
        and receipt_authority.get("public") is False
        and receipt_authority.get("promotion") is False
        and receipt_authority.get("deployment") is False
    )
    full_source_binding = (
        isinstance(source_binding, Mapping)
        and source_binding.get("receipt_sha256") == source_hash
        and source_binding.get("selected_is_full_accepted_census") is True
        and accepted_ids is not None
        and tuple(source_binding.get("accepted_game_ids") or ()) == accepted_ids
        and tuple(source_binding.get("selected_game_ids") or ()) == accepted_ids
    )
    full_coverage = (
        isinstance(coverage, Mapping)
        and coverage.get("mapped_is_full_accepted_census") is True
        and _as_count(coverage.get("mapped_game_count")) == accepted_count
    )
    complete = (
        artifact_present
        and receipt_present
        and crosswalk.get("status") == "complete_authoritative_coverage"
        and crosswalk_receipt.get("status") == "verified_research_only"
        and crosswalk_hash_valid
        and crosswalk_receipt_hash_valid
        and receipt_crosswalk_hash_matches
        and receipt_artifact_binding_matches
        and receipt_mapped_identity_matches
        and receipt_accepted_identity_matches
        and receipt_assignment_binding_matches
        and receipt_source_hash == source_hash
        and full_source_binding
        and full_coverage
        and assignment_ids_are_full
        and assignment_outcome_free
        and bridge_authority_safe
        and receipt_authority_safe
        and artifact_authoritative_series
        and receipt_authoritative_series
        and crosswalk_artifact_file_verified
        and crosswalk_receipt_file_verified
        and crosswalk_artifact_payload_matches
        and crosswalk_receipt_payload_matches
        and crosswalk_artifact_schema_verified
        and crosswalk_receipt_schema_verified
    )
    return {
        "status": "verified_full_coverage_candidate" if complete else "unavailable",
        "artifact_present": artifact_present,
        "receipt_present": receipt_present,
        "artifact_file": dict(crosswalk_artifact_file)
        if isinstance(crosswalk_artifact_file, Mapping)
        else None,
        "receipt_file": dict(crosswalk_receipt_file)
        if isinstance(crosswalk_receipt_file, Mapping)
        else None,
        "durable_in_repository": False,
        "artifact_status": (
            _as_string(crosswalk.get("status")) if artifact_present else None
        ),
        "receipt_status": (
            _as_string(crosswalk_receipt.get("status")) if receipt_present else None
        ),
        "accepted_game_count": accepted_count,
        "crosswalk_accepted_game_count": crosswalk_accepted_count,
        "crosswalk_source_receipt_sha256": receipt_source_hash,
        "accepted_source_receipt_sha256": source_hash,
        "crosswalk_sha256": (
            _as_string(crosswalk.get("crosswalk_sha256"))
            if artifact_present
            else None
        ),
        "crosswalk_receipt_sha256": (
            _as_string(crosswalk_receipt.get("receipt_sha256"))
            if receipt_present
            else None
        ),
        "crosswalk_source_identity_sha256": crosswalk_source_identity,
        "accepted_source_identity_sha256": accepted_source_identity,
        "source_census_matches": source_census_matches,
        "assigned_game_count": len(assignment_ids),
        "assignment_ids_are_full_accepted_census": assignment_ids_are_full,
        "assignment_outcome_free": assignment_outcome_free,
        "source_binding_is_full": full_source_binding,
        "coverage_is_full": full_coverage,
        "source_receipt_sha256_matches": receipt_source_hash == source_hash,
        "crosswalk_hash_valid": crosswalk_hash_valid,
        "crosswalk_receipt_hash_valid": crosswalk_receipt_hash_valid,
        "crosswalk_artifact_file_verified": crosswalk_artifact_file_verified,
        "crosswalk_receipt_file_verified": crosswalk_receipt_file_verified,
        "crosswalk_artifact_payload_matches": crosswalk_artifact_payload_matches,
        "crosswalk_receipt_payload_matches": crosswalk_receipt_payload_matches,
        "crosswalk_artifact_schema_verified": crosswalk_artifact_schema_verified,
        "crosswalk_receipt_schema_verified": crosswalk_receipt_schema_verified,
        "expected_crosswalk_receipt_file_sha256": (
            str(expected_crosswalk_receipt_file_sha256).lower()
            if expected_crosswalk_receipt_file_sha256 is not None
            else None
        ),
        "receipt_crosswalk_hash_matches": receipt_crosswalk_hash_matches,
        "receipt_artifact_binding_matches": receipt_artifact_binding_matches,
        "receipt_mapped_identity_matches": receipt_mapped_identity_matches,
        "receipt_accepted_identity_matches": receipt_accepted_identity_matches,
        "receipt_assignment_binding_matches": receipt_assignment_binding_matches,
        "bridge_authority_safe": bridge_authority_safe,
        "receipt_authority_safe": receipt_authority_safe,
        "artifact_authoritative_series": artifact_authoritative_series,
        "receipt_authoritative_series": receipt_authoritative_series,
        "authoritative_series_flags_match": (
            artifact_authoritative_series and receipt_authoritative_series
        ),
        "authoritative_for_accepted_census": complete,
    }


def build_series_authority_audit(
    *,
    source_receipt: Mapping[str, Any],
    accepted_census: Mapping[str, Any],
    phase_evaluation: Mapping[str, Any],
    proxy_artifact: Mapping[str, Any] | None,
    source_receipt_artifact: Mapping[str, Any],
    accepted_census_artifact: Mapping[str, Any],
    phase_evaluation_artifact: Mapping[str, Any],
    proxy_artifact_file: Mapping[str, Any] | None,
    requested_proxy_map_count: int = TARGET_PROXY_MAP_COUNT,
    leaguepedia_crosswalk: Mapping[str, Any] | None = None,
    leaguepedia_crosswalk_receipt: Mapping[str, Any] | None = None,
    leaguepedia_crosswalk_artifact_file: Mapping[str, Any] | None = None,
    leaguepedia_crosswalk_receipt_file: Mapping[str, Any] | None = None,
    leaguepedia_crosswalk_expected_receipt_file_sha256: str | None = None,
    variant_bundle: Mapping[str, Any] | None = None,
    variant_bundle_file: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a source-bound whole-series authority audit.

    The function reports missing authority as data.  It never assigns a
    series identifier when the accepted source lacks one.
    """

    blockers: list[str] = []
    source_hash = _as_string(source_receipt.get("receipt_sha256"))
    source_identity = _as_string(source_receipt.get("source_identity_sha256"))
    source_ids = _canonical_ids(source_receipt.get("accepted_game_ids"))
    census_ids = _canonical_ids(accepted_census.get("game_ids"))
    source_receipt_hash_valid = _hash_matches(source_receipt, "receipt_sha256")
    if not source_receipt_hash_valid:
        blockers.append("accepted_source_receipt_hash_invalid")
    if not source_hash or not source_identity or source_ids is None:
        blockers.append("accepted_source_receipt_census_invalid")

    census_binding = {
        "game_count": _as_count(accepted_census.get("game_count")),
        "game_identity_sha256": (
            identity_sha256(census_ids) if census_ids is not None else None
        ),
        "source_identity_sha256": _as_string(
            accepted_census.get("source_identity_sha256")
        ),
        "matches_source_census": (
            census_ids is not None
            and source_ids is not None
            and census_ids == source_ids
            and _as_count(accepted_census.get("game_count")) == len(source_ids)
            and accepted_census.get("source_identity_sha256") == source_identity
        ),
        "contains_series_assignment": _contains_key(
            accepted_census, {"series_id", "whole_series_id", "match_id"}
        ),
        "contains_tournament_field": _contains_key(accepted_census, {"tournament"}),
    }
    if not census_binding["matches_source_census"]:
        blockers.append("accepted_census_source_binding_invalid")
    if not isinstance(source_receipt.get("series_identity"), Mapping):
        blockers.append("accepted_source_receipt_has_no_series_identity")
    if not census_binding["contains_series_assignment"]:
        blockers.append("accepted_census_has_no_series_assignment")

    source_tournament_fields = {
        "series_identity": isinstance(source_receipt.get("series_identity"), Mapping),
        "series_id": _contains_key(source_receipt, {"series_id", "whole_series_id"}),
        "match_id": _contains_key(source_receipt, {"match_id"}),
        "tournament": _contains_key(source_receipt, {"tournament"}),
        "lp_game_id": _contains_key(source_receipt, {"lp_game_id"}),
        "source_lp": _contains_key(source_receipt, {"source_lp"}),
        "lp_matched": _contains_key(source_receipt, {"lp_matched"}),
    }
    accepted_tournament_fields = {
        "series_id": census_binding["contains_series_assignment"],
        "tournament": census_binding["contains_tournament_field"],
    }

    phase_series = phase_evaluation.get("series_identity")
    phase_series = phase_series if isinstance(phase_series, Mapping) else {}
    phase_authoritative = phase_evaluation.get("authoritative_series_identity") is True
    phase_authoritative = phase_authoritative and phase_series.get("authoritative") is True
    phase_source_binding = {
        "source_game_count": _as_count(phase_evaluation.get("source_game_count")),
        "source_identity_sha256": _as_string(
            phase_evaluation.get("source_identity_sha256")
        ),
        "source_receipt_sha256": _as_string(
            phase_evaluation.get("source_receipt_sha256")
        ),
        "matches_source_receipt": (
            _as_count(phase_evaluation.get("source_game_count"))
            == _as_count(source_receipt.get("source_game_count"))
            and phase_evaluation.get("source_identity_sha256") == source_identity
            and phase_evaluation.get("source_receipt_sha256") == source_hash
        ),
    }
    if not phase_source_binding["matches_source_receipt"]:
        blockers.append("phase_evaluation_source_binding_invalid")
    if not phase_authoritative:
        blockers.append("phase_evaluation_declares_series_identity_non_authoritative")
    phase_collisions = phase_series.get("possible_collisions")
    phase_collisions = phase_collisions if isinstance(phase_collisions, Mapping) else {}
    phase_collision_counts = _copy_counts(
        phase_collisions,
        ("clusters", "rows", "cross_date_clusters", "cross_date_rows"),
    )
    if any(value not in (None, 0) for value in phase_collision_counts.values()):
        blockers.append("phase_evaluation_reports_possible_cross_date_collisions")

    proxy_summary: dict[str, Any]
    if isinstance(proxy_artifact, Mapping):
        proxy_source = proxy_artifact.get("source")
        proxy_source = proxy_source if isinstance(proxy_source, Mapping) else {}
        proxy_maps = proxy_source.get("maps")
        proxy_maps = proxy_maps if isinstance(proxy_maps, Mapping) else {}
        proxy_oracle = proxy_artifact.get("oracle_audit")
        proxy_oracle = proxy_oracle if isinstance(proxy_oracle, Mapping) else {}
        proxy_cluster_arithmetic = proxy_artifact.get("cluster_arithmetic")
        proxy_cluster_arithmetic = (
            proxy_cluster_arithmetic
            if isinstance(proxy_cluster_arithmetic, Mapping)
            else {}
        )
        accepted_maps = source_receipt.get("source_files")
        accepted_maps = accepted_maps if isinstance(accepted_maps, Mapping) else {}
        accepted_maps_record = accepted_maps.get("maps")
        accepted_maps_record = (
            accepted_maps_record if isinstance(accepted_maps_record, Mapping) else {}
        )
        proxy_raw_hash = _as_string(proxy_maps.get("raw_sha256"))
        accepted_maps_hash = _as_string(accepted_maps_record.get("sha256"))
        proxy_summary = {
            "present": True,
            "authoritative_series_identity": proxy_artifact.get(
                "authoritative_series_identity"
            ) is True,
            "artifact_hash_valid": _hash_matches(proxy_artifact, "artifact_sha256"),
            "artifact_sha256": _as_string(proxy_artifact.get("artifact_sha256")),
            "assigned_maps": _as_count(
                (proxy_artifact.get("eligibility") or {}).get("assigned_maps")
                if isinstance(proxy_artifact.get("eligibility"), Mapping)
                else None
            ),
            "source_rows": _as_count(proxy_maps.get("rows")),
            "source_raw_sha256": proxy_raw_hash,
            "accepted_source_maps_sha256": accepted_maps_hash,
            "source_hash_matches_accepted_maps": (
                proxy_raw_hash is not None and proxy_raw_hash == accepted_maps_hash
            ),
            "declared_columns": list(proxy_maps.get("columns_read") or [])
            if isinstance(proxy_maps.get("columns_read"), Sequence)
            and not isinstance(proxy_maps.get("columns_read"), (str, bytes, bytearray))
            else [],
            "oracle_audit": {
                str(key): dict(value)
                for key, value in proxy_oracle.items()
                if isinstance(value, Mapping)
            },
            "cluster_count": _as_count(proxy_cluster_arithmetic.get("dependence_clusters")),
            "claim_ceiling": _as_string(proxy_artifact.get("claim_ceiling")),
        }
        if proxy_summary["authoritative_series_identity"]:
            blockers.append("legacy_proxy_claims_authoritative_series_identity")
        if proxy_summary["claim_ceiling"] and "never" in proxy_summary["claim_ceiling"].lower():
            blockers.append("legacy_proxy_claim_ceiling_forbids_authoritative_series_id")
        if not proxy_summary["source_hash_matches_accepted_maps"]:
            blockers.append("legacy_proxy_source_hash_differs_from_accepted_source_maps")
    else:
        proxy_summary = {"present": False}
        blockers.append("legacy_series_proxy_artifact_missing")

    crosswalk_summary = _crosswalk_summary(
        leaguepedia_crosswalk,
        leaguepedia_crosswalk_receipt,
        source_receipt=source_receipt,
        accepted_ids=source_ids,
        crosswalk_artifact_file=leaguepedia_crosswalk_artifact_file,
        crosswalk_receipt_file=leaguepedia_crosswalk_receipt_file,
        expected_crosswalk_receipt_file_sha256=(
            leaguepedia_crosswalk_expected_receipt_file_sha256
        ),
    )
    if not crosswalk_summary["artifact_present"]:
        blockers.append("leaguepedia_oe_crosswalk_artifact_missing")
    if not crosswalk_summary["receipt_present"]:
        blockers.append("leaguepedia_oe_crosswalk_receipt_missing")
    if not crosswalk_summary["authoritative_for_accepted_census"]:
        blockers.append("leaguepedia_oe_bridge_is_not_full_source_bound_coverage")
    if (
        crosswalk_summary["artifact_present"]
        and not crosswalk_summary["crosswalk_artifact_file_verified"]
    ):
        blockers.append("leaguepedia_crosswalk_artifact_file_unverified")
    if (
        crosswalk_summary["receipt_present"]
        and not crosswalk_summary["crosswalk_receipt_file_verified"]
    ):
        blockers.append("leaguepedia_crosswalk_receipt_file_unverified")
    if (
        crosswalk_summary["artifact_present"]
        and crosswalk_summary["receipt_present"]
        and not crosswalk_summary["authoritative_series_flags_match"]
    ):
        blockers.append("leaguepedia_oe_bridge_authoritative_series_flag_missing")
    if (
        crosswalk_summary["artifact_present"]
        and crosswalk_summary["receipt_present"]
        and not crosswalk_summary["source_census_matches"]
    ):
        blockers.append("leaguepedia_crosswalk_binds_different_source_census")

    bundle_source = variant_bundle.get("source") if isinstance(variant_bundle, Mapping) else None
    bundle_source = bundle_source if isinstance(bundle_source, Mapping) else {}
    bundle_partition = bundle_source.get("series_partition")
    bundle_partition = bundle_partition if isinstance(bundle_partition, Mapping) else {}
    bundle_source_hash = _as_string(bundle_source.get("source_receipt_sha256"))
    bundle_identity = _as_string(bundle_source.get("source_identity_sha256"))
    bundle_count = _as_count(bundle_source.get("source_game_count"))
    bundle_source_matches = (
        bundle_source_hash == source_hash
        and bundle_identity == source_identity
        and bundle_count == _as_count(source_receipt.get("source_game_count"))
    )
    bundle_summary = {
        "present": isinstance(variant_bundle, Mapping),
        "artifact_file": dict(variant_bundle_file)
        if isinstance(variant_bundle_file, Mapping)
        else None,
        "artifact_hash_valid": (
            _hash_matches(variant_bundle, "bundle_sha256")
            if isinstance(variant_bundle, Mapping)
            else False
        ),
        "source_receipt_sha256": bundle_source_hash,
        "source_identity_sha256": bundle_identity,
        "source_game_count": bundle_count,
        "source_census_matches_accepted": bundle_source_matches,
        "retained_proxy_game_count": _as_count(
            bundle_partition.get("retained_proxy_game_count")
        ),
        "retained_proxy_cluster_count": _as_count(
            bundle_partition.get("retained_proxy_cluster_count")
        ),
        "mapped_game_count": _as_count(bundle_partition.get("mapped_game_count")),
        "mapped_series_count": _as_count(bundle_partition.get("mapped_series_count")),
        "mapped_series_authoritative": bundle_partition.get("mapped_series_authoritative") is True,
        "partial_series_blocker": bundle_partition.get("partial_series_blocker") is True,
        "status": _as_string(variant_bundle.get("status"))
        if isinstance(variant_bundle, Mapping)
        else None,
    }
    if bundle_summary["present"] and not bundle_source_matches:
        blockers.append("variant_bundle_binds_different_source_census")

    proxy_cohort = {
        "requested_map_count": requested_proxy_map_count,
        "requested_count_is_currently_bound": False,
        "observed_counts": {
            "phase_exact_id_proxy_rows": _as_count(
                (phase_series.get("source_counts") or {}).get("exact_id_proxy")
                if isinstance(phase_series.get("source_counts"), Mapping)
                else None
            ),
            "phase_team_tournament_proxy_rows": _as_count(
                (phase_series.get("source_counts") or {}).get("team_tournament_proxy")
                if isinstance(phase_series.get("source_counts"), Mapping)
                else None
            ),
            "legacy_proxy_assigned_maps": proxy_summary.get("assigned_maps"),
            "legacy_lpl_gameid_audit_maps": _as_count(
                (proxy_summary.get("oracle_audit") or {})
                .get("lpl_gameid_prefix_and_url_bmid", {})
                .get("maps")
                if isinstance(proxy_summary.get("oracle_audit"), Mapping)
                else None
            ),
            "legacy_leaguepedia_game_id_audit_maps": _as_count(
                (proxy_summary.get("oracle_audit") or {})
                .get("leaguepedia_game_id", {})
                .get("maps")
                if isinstance(proxy_summary.get("oracle_audit"), Mapping)
                else None
            ),
            "external_variant_bundle_retained_proxy_maps": bundle_summary[
                "retained_proxy_game_count"
            ],
        },
        "external_variant_bundle": bundle_summary,
        "verification": (
            "The requested cohort is present only in the external variant bundle. "
            "Its source census must match the accepted receipt before it can support evaluation."
            if bundle_summary["retained_proxy_game_count"] == requested_proxy_map_count
            else (
                "The requested cohort count is not present as a bound field in the "
                "accepted source receipt, accepted census, phase evaluation, or "
                "legacy proxy artifact."
            )
        ),
    }
    if bundle_summary["retained_proxy_game_count"] != requested_proxy_map_count:
        blockers.append("requested_proxy_cohort_not_identified_in_frozen_artifacts")
    elif not bundle_summary["source_census_matches_accepted"]:
        blockers.append("requested_proxy_cohort_is_bound_to_different_source_census")

    tournament_boundary = {
        "status": "blocked",
        "source_receipt_fields": source_tournament_fields,
        "accepted_census_fields": accepted_tournament_fields,
        "phase_partition_mentions_tournament": "tournament"
        in set(phase_series.get("cross_model_partition", {}).get("phase_key_fields", []))
        if isinstance(phase_series.get("cross_model_partition"), Mapping)
        else False,
        "source_bound_tournament_values": False,
        "reason": "The frozen source receipt and accepted census contain map IDs only.",
    }
    blockers.append("tournament_assignment_not_source_bound")

    artifact_records = {
        "source_receipt": dict(source_receipt_artifact),
        "accepted_census": dict(accepted_census_artifact),
        "phase_evaluation": dict(phase_evaluation_artifact),
        "legacy_proxy": dict(proxy_artifact_file)
        if isinstance(proxy_artifact_file, Mapping)
        else None,
    }
    for record_name, record in artifact_records.items():
        if record is not None and not _valid_file_record(record):
            blockers.append(f"{record_name}_artifact_record_invalid")
    for record_name, record in (
        ("leaguepedia_crosswalk", leaguepedia_crosswalk_artifact_file),
        ("leaguepedia_crosswalk_receipt", leaguepedia_crosswalk_receipt_file),
    ):
        if record is not None and not _valid_file_record(record):
            blockers.append(f"{record_name}_artifact_record_invalid")
    if variant_bundle_file is not None and not _valid_file_record(variant_bundle_file):
        blockers.append("variant_bundle_artifact_record_invalid")

    audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked_research_only",
        "authority": dict(AUTHORITY),
        "decision": {
            "fail_closed": True,
            "can_assign_authoritative_series": False,
            "can_populate_tournament_boundary": False,
            "can_promote_tier_evaluation": False,
        },
        "source": {
            "source_as_of": _as_string(source_receipt.get("source_as_of")),
            "source_game_count": _as_count(source_receipt.get("source_game_count")),
            "source_identity_sha256": source_identity,
            "source_receipt_sha256": source_hash,
            "source_receipt_hash_valid": source_receipt_hash_valid,
            "source_receipt_artifact": dict(source_receipt_artifact),
            "accepted_census_artifact": dict(accepted_census_artifact),
            "accepted_census_binding": census_binding,
        },
        "series_authority": {
            "status": "blocked",
            "authoritative_non_grid_series_id": {
                "status": "unavailable",
                "source_fields": source_tournament_fields,
                "accepted_census_assignment_present": census_binding[
                    "contains_series_assignment"
                ],
                "phase_declares_authoritative": phase_authoritative,
            },
            "phase_evaluation": {
                "artifact": dict(phase_evaluation_artifact),
                "source_binding": phase_source_binding,
                "status": _as_string(phase_series.get("status")),
                "authoritative": phase_authoritative,
                "source_counts": _copy_counts(
                    phase_series.get("source_counts"),
                    ("exact_id_proxy", "team_tournament_proxy"),
                ),
                "cluster_counts": _copy_counts(
                    phase_series.get("cluster_counts"),
                    ("exact_id_proxy", "team_tournament_proxy"),
                ),
                "possible_collisions": phase_collision_counts,
            },
            "leaguepedia_oe_bridge": crosswalk_summary,
            "legacy_proxy": proxy_summary,
        },
        "tournament_boundary": tournament_boundary,
        "proxy_cohort": proxy_cohort,
        "blockers": sorted(set(blockers)),
        "artifact_records": artifact_records,
    }
    audit["receipt_sha256"] = canonical_sha256(audit)
    return audit


def verify_series_authority_audit(
    audit: Mapping[str, Any], *, expected_receipt_sha256: str | None = None
) -> None:
    """Verify the immutable audit hash and fail-closed decision."""

    if not isinstance(audit, Mapping):
        raise SeriesAuthorityAuditError("series authority audit is required")
    if audit.get("schema_version") != SCHEMA_VERSION:
        raise SeriesAuthorityAuditError("series authority audit schema is invalid")
    if audit.get("status") != "blocked_research_only":
        raise SeriesAuthorityAuditError("series authority audit status is invalid")
    if dict(audit.get("authority") or {}) != AUTHORITY:
        raise SeriesAuthorityAuditError("series authority audit authority is invalid")
    claimed = audit.get("receipt_sha256")
    if not isinstance(claimed, str) or _SHA256_RE.fullmatch(claimed) is None:
        raise SeriesAuthorityAuditError("series authority audit hash is invalid")
    body = dict(audit)
    body.pop("receipt_sha256", None)
    if canonical_sha256(body) != claimed.lower():
        raise SeriesAuthorityAuditError("series authority audit hash does not match payload")
    if expected_receipt_sha256 is not None and claimed.lower() != str(
        expected_receipt_sha256
    ).lower():
        raise SeriesAuthorityAuditError("series authority audit hash differs from expected")
    decision = audit.get("decision")
    if not isinstance(decision, Mapping) or decision.get("fail_closed") is not True:
        raise SeriesAuthorityAuditError("series authority audit is not fail closed")
    for field in (
        "can_assign_authoritative_series",
        "can_populate_tournament_boundary",
        "can_promote_tier_evaluation",
    ):
        if decision.get(field) is not False:
            raise SeriesAuthorityAuditError(f"series authority audit decision is unsafe: {field}")
    blockers = audit.get("blockers")
    if not isinstance(blockers, list) or not blockers:
        raise SeriesAuthorityAuditError("series authority audit has no blockers")
    if not isinstance(audit.get("artifact_records"), Mapping):
        raise SeriesAuthorityAuditError("series authority audit artifact records are missing")
    for name, record in audit["artifact_records"].items():
        if record is not None and not _valid_file_record(record):
            raise SeriesAuthorityAuditError(
                f"series authority audit artifact record is invalid: {name}"
            )
    series_authority = audit.get("series_authority")
    bridge = (
        series_authority.get("leaguepedia_oe_bridge")
        if isinstance(series_authority, Mapping)
        else None
    )
    if isinstance(bridge, Mapping):
        if bridge.get("authoritative_for_accepted_census") is True and (
            bridge.get("artifact_authoritative_series") is not True
            or bridge.get("receipt_authoritative_series") is not True
        ):
            raise SeriesAuthorityAuditError(
                "series authority audit bridge lacks dual authoritative-series flags"
            )
        if bridge.get("authoritative_for_accepted_census") is True:
            if (
                bridge.get("crosswalk_artifact_file_verified") is not True
                or bridge.get("crosswalk_receipt_file_verified") is not True
                or bridge.get("crosswalk_artifact_payload_matches") is not True
                or bridge.get("crosswalk_receipt_payload_matches") is not True
                or bridge.get("crosswalk_artifact_schema_verified") is not True
                or bridge.get("crosswalk_receipt_schema_verified") is not True
            ):
                raise SeriesAuthorityAuditError(
                    "series authority audit bridge files are not byte-bound"
                )
            artifact_payload, artifact_file = _read_verified_json_file(
                bridge.get("artifact_file"),
                label="leaguepedia crosswalk artifact",
            )
            receipt_payload, receipt_file = _read_verified_json_file(
                bridge.get("receipt_file"),
                label="leaguepedia crosswalk receipt",
            )
            if artifact_payload is None or receipt_payload is None:
                raise SeriesAuthorityAuditError(
                    "series authority audit bridge files cannot be read"
                )
            if not _crosswalk_payload_is_verified(artifact_payload):
                raise SeriesAuthorityAuditError(
                    "series authority audit crosswalk artifact is invalid"
                )
            if not _crosswalk_receipt_is_verified(
                receipt_payload,
                expected_file_sha256=bridge.get(
                    "expected_crosswalk_receipt_file_sha256"
                ),
                verified_file=receipt_file,
            ):
                raise SeriesAuthorityAuditError(
                    "series authority audit crosswalk receipt is invalid"
                )
            if (
                artifact_payload.get("crosswalk_sha256")
                != bridge.get("crosswalk_sha256")
                or receipt_payload.get("receipt_sha256")
                != bridge.get("crosswalk_receipt_sha256")
                or receipt_payload.get("crosswalk_sha256")
                != artifact_payload.get("crosswalk_sha256")
            ):
                raise SeriesAuthorityAuditError(
                    "series authority audit bridge payload binding changed"
                )
            receipt_artifact = receipt_payload.get("artifact")
            if (
                not isinstance(receipt_artifact, Mapping)
                or artifact_file is None
                or receipt_artifact.get("bytes") != artifact_file.get("bytes")
                or receipt_artifact.get("sha256") != artifact_file.get("sha256")
            ):
                raise SeriesAuthorityAuditError(
                    "series authority audit bridge artifact binding changed"
                )
        for name in ("artifact_file", "receipt_file"):
            record = bridge.get(name)
            if record is not None and not _valid_file_record(record):
                raise SeriesAuthorityAuditError(
                    f"series authority audit bridge record is invalid: {name}"
                )
    cohort = audit.get("proxy_cohort")
    external_bundle = (
        cohort.get("external_variant_bundle")
        if isinstance(cohort, Mapping)
        else None
    )
    if isinstance(external_bundle, Mapping):
        record = external_bundle.get("artifact_file")
        if record is not None and not _valid_file_record(record):
            raise SeriesAuthorityAuditError(
                "series authority audit variant bundle record is invalid"
            )


__all__ = [
    "AUTHORITY",
    "SCHEMA_VERSION",
    "TARGET_PROXY_MAP_COUNT",
    "SeriesAuthorityAuditError",
    "build_series_authority_audit",
    "canonical_json_bytes",
    "canonical_sha256",
    "file_record",
    "verify_series_authority_audit",
]
