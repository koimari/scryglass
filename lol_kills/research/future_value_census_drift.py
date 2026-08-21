"""Record exact local bridge evidence for a source-census drift.

The current accepted source census and the Leaguepedia bridge source census
are different snapshots.  This module records the rows that occur only in the
bridge snapshot.  It verifies their source rows and outcome-free series
assignments without extending the current census or opening a promotion gate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import re

from lol_kills.research.future_value_series_authority import (
    canonical_json_bytes,
    canonical_sha256,
    file_record,
)
from lol_kills.research.oe_leaguepedia_series_crosswalk import verify_crosswalk
from lol_kills.v2.tierlists.accepted_census import canonical_game_ids, identity_sha256


SCHEMA_VERSION = "scryglass:future-value-census-drift-audit:v1"
STATUS = "verified_local_bridge_census_drift_fail_closed"
DRIFT_GAME_COUNT = 8
AUTHORITY = {
    "authoritative_series": False,
    "deployment": False,
    "promotion": False,
    "public": False,
    "research_only": True,
    "tournament_boundary": False,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


class CensusDriftAuditError(ValueError):
    """Raised when a census-drift audit is malformed or cannot be verified."""


def _string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _sha256(value: Any) -> str | None:
    candidate = _string(value)
    return candidate.lower() if candidate and _SHA256_RE.fullmatch(candidate) else None


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _record_is_valid(record: Any) -> bool:
    if not isinstance(record, Mapping):
        return False
    return (
        _sha256(record.get("sha256")) is not None
        and isinstance(record.get("bytes"), int)
        and not isinstance(record.get("bytes"), bool)
        and record["bytes"] > 0
        and _string(record.get("locator")) is not None
    )


def _self_hash_is_valid(payload: Mapping[str, Any], field: str) -> bool:
    claimed = _sha256(payload.get(field))
    if claimed is None:
        return False
    body = dict(payload)
    body.pop(field, None)
    return canonical_sha256(body) == claimed


def _source_receipt_census(receipt: Mapping[str, Any], *, label: str) -> tuple[str, ...]:
    ids = receipt.get("accepted_game_ids")
    if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes, bytearray)):
        raise CensusDriftAuditError(f"{label} accepted game IDs are missing")
    canonical = canonical_game_ids(ids)
    if tuple(str(item) for item in ids) != canonical:
        raise CensusDriftAuditError(f"{label} accepted game IDs are not canonical")
    if _count(receipt.get("source_game_count")) != len(canonical):
        raise CensusDriftAuditError(f"{label} source game count is invalid")
    if _sha256(receipt.get("source_identity_sha256")) != identity_sha256(canonical):
        raise CensusDriftAuditError(f"{label} source identity is invalid")
    if not _self_hash_is_valid(receipt, "receipt_sha256"):
        raise CensusDriftAuditError(f"{label} source receipt hash is invalid")
    return canonical


def _assignment_map(crosswalk: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = crosswalk.get("assignments")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise CensusDriftAuditError("crosswalk assignments are missing")
    assignments: dict[str, dict[str, Any]] = {}
    for value in rows:
        if not isinstance(value, Mapping):
            raise CensusDriftAuditError("crosswalk assignment is invalid")
        game_id = _string(value.get("oe_game_id"))
        series_id = _string(value.get("series_id"))
        if game_id is None or series_id is None or game_id in assignments:
            raise CensusDriftAuditError("crosswalk assignment identity is invalid")
        if value.get("outcome_used") is not False:
            raise CensusDriftAuditError("crosswalk assignment uses an outcome")
        assignments[game_id] = dict(value)
    return assignments


def _source_row_map(rows: Sequence[Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for value in rows:
        if not isinstance(value, Mapping):
            raise CensusDriftAuditError("bridge source row is invalid")
        game_id = _string(value.get("gameid"))
        if game_id is None or game_id in result:
            raise CensusDriftAuditError("bridge source row identity is invalid")
        result[game_id] = dict(value)
    return result


def _team_keys(row: Mapping[str, Any]) -> tuple[str, ...]:
    values = row.get("team_keys")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise CensusDriftAuditError("bridge source team keys are missing")
    canonical = tuple(sorted(str(value) for value in values if str(value).strip()))
    if len(canonical) != 2 or len(set(canonical)) != 2:
        raise CensusDriftAuditError("bridge source team keys are invalid")
    return canonical


def _bridge_binding(
    *,
    crosswalk: Mapping[str, Any],
    crosswalk_receipt: Mapping[str, Any],
    external_source: Mapping[str, Any],
    external_ids: tuple[str, ...],
    assignments: Mapping[str, Mapping[str, Any]],
    crosswalk_artifact: Mapping[str, Any],
    crosswalk_receipt_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    source_binding = crosswalk.get("source_binding")
    source_binding = source_binding if isinstance(source_binding, Mapping) else {}
    receipt_artifact = crosswalk_receipt.get("artifact")
    receipt_artifact = receipt_artifact if isinstance(receipt_artifact, Mapping) else {}
    assignment_ids = tuple(sorted(assignments))
    expected_source_receipt = _sha256(external_source.get("receipt_sha256"))
    expected_source_identity = _sha256(external_source.get("source_identity_sha256"))
    expected_crosswalk_hash = _sha256(crosswalk.get("crosswalk_sha256"))
    expected_receipt_hash = _sha256(crosswalk_receipt.get("receipt_sha256"))
    receipt_artifact_matches = (
        receipt_artifact.get("bytes") == crosswalk_artifact.get("bytes")
        and receipt_artifact.get("sha256") == crosswalk_artifact.get("sha256")
    )
    return {
        "crosswalk_sha256": expected_crosswalk_hash,
        "crosswalk_hash_valid": _self_hash_is_valid(crosswalk, "crosswalk_sha256"),
        "crosswalk_receipt_sha256": expected_receipt_hash,
        "crosswalk_receipt_hash_valid": _self_hash_is_valid(
            crosswalk_receipt, "receipt_sha256"
        ),
        "crosswalk_receipt_binds_crosswalk": (
            crosswalk_receipt.get("crosswalk_sha256") == crosswalk.get("crosswalk_sha256")
        ),
        "crosswalk_receipt_binds_artifact": receipt_artifact_matches,
        "crosswalk_receipt_binds_assignment_ids": (
            crosswalk_receipt.get("mapped_game_identity_sha256")
            == identity_sha256(assignment_ids)
        ),
        "source_receipt_sha256": expected_source_receipt,
        "source_identity_sha256": expected_source_identity,
        "accepted_game_count": len(external_ids),
        "source_binding_matches": (
            _count(source_binding.get("accepted_game_count")) == len(external_ids)
            and tuple(source_binding.get("accepted_game_ids") or ()) == external_ids
            and source_binding.get("accepted_game_identity_sha256")
            == expected_source_identity
            and source_binding.get("receipt_sha256") == expected_source_receipt
            and crosswalk_receipt.get("accepted_game_count") == len(external_ids)
            and crosswalk_receipt.get("source_identity_sha256") == expected_source_identity
            and crosswalk_receipt.get("source_receipt_sha256") == expected_source_receipt
        ),
        "assignments_outcome_free": all(
            row.get("outcome_used") is False for row in assignments.values()
        ),
        "mapped_game_count": len(assignment_ids),
        "mapped_series_count": len(
            {str(row["series_id"]) for row in assignments.values()}
        ),
        "mapped_is_full_external_census": assignment_ids == external_ids,
    }


def build_census_drift_audit(
    *,
    current_source: Mapping[str, Any],
    external_source: Mapping[str, Any],
    bridge_oe_rows: Sequence[Mapping[str, Any]],
    crosswalk: Mapping[str, Any],
    crosswalk_receipt: Mapping[str, Any],
    current_source_artifact: Mapping[str, Any],
    external_source_artifact: Mapping[str, Any],
    bridge_oe_artifact: Mapping[str, Any],
    crosswalk_artifact: Mapping[str, Any],
    crosswalk_receipt_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an immutable, fail-closed receipt for external-only map rows."""

    current_ids = _source_receipt_census(current_source, label="current source")
    external_ids = _source_receipt_census(external_source, label="external source")
    if not _record_is_valid(current_source_artifact):
        raise CensusDriftAuditError("current source artifact record is invalid")
    for label, record in (
        ("external source", external_source_artifact),
        ("bridge OE", bridge_oe_artifact),
        ("crosswalk", crosswalk_artifact),
        ("crosswalk receipt", crosswalk_receipt_artifact),
    ):
        if not _record_is_valid(record):
            raise CensusDriftAuditError(f"{label} artifact record is invalid")

    verify_crosswalk(crosswalk)
    if not _self_hash_is_valid(crosswalk_receipt, "receipt_sha256"):
        raise CensusDriftAuditError("crosswalk receipt hash is invalid")
    assignments = _assignment_map(crosswalk)
    source_rows = _source_row_map(bridge_oe_rows)
    external_only = tuple(sorted(set(external_ids) - set(current_ids)))
    current_only = tuple(sorted(set(current_ids) - set(external_ids)))
    if len(external_only) != DRIFT_GAME_COUNT or current_only:
        raise CensusDriftAuditError("the expected eight-map census drift changed")
    bridge_binding = _bridge_binding(
        crosswalk=crosswalk,
        crosswalk_receipt=crosswalk_receipt,
        external_source=external_source,
        external_ids=external_ids,
        assignments=assignments,
        crosswalk_artifact=crosswalk_artifact,
        crosswalk_receipt_artifact=crosswalk_receipt_artifact,
    )
    if not bridge_binding["source_binding_matches"]:
        raise CensusDriftAuditError("crosswalk source binding is invalid")
    if not bridge_binding["assignments_outcome_free"]:
        raise CensusDriftAuditError("crosswalk assignments are not outcome free")

    drift_rows: list[dict[str, Any]] = []
    series_ids: set[str] = set()
    for game_id in external_only:
        source_row = source_rows.get(game_id)
        assignment = assignments.get(game_id)
        if source_row is None:
            raise CensusDriftAuditError(f"bridge source row is missing: {game_id}")
        if assignment is None:
            raise CensusDriftAuditError(f"bridge assignment is missing: {game_id}")
        source_date = _string(source_row.get("date"))
        source_league = _string(source_row.get("league"))
        source_patch = _string(source_row.get("patch"))
        source_teams = _team_keys(source_row)
        series_id = _string(assignment.get("series_id"))
        scoreboard_id = _string(assignment.get("scoreboard_game_id"))
        scoreboard_prefix = _string(assignment.get("scoreboard_game_id_prefix"))
        assignment_timestamp = _string(assignment.get("oe_timestamp"))
        assignment_league = _string(assignment.get("source_league"))
        assignment_patch = _string(assignment.get("source_patch"))
        normalized_team_set = assignment.get("normalized_team_set")
        normalized_team_set = (
            tuple(sorted(str(value) for value in normalized_team_set))
            if isinstance(normalized_team_set, Sequence)
            and not isinstance(normalized_team_set, (str, bytes, bytearray))
            else ()
        )
        order = assignment.get("scoreboard_game_order")
        checks = {
            "source_game_id_exact": source_row.get("gameid") == game_id,
            "source_timestamp_equals_assignment": (
                source_date is not None and source_date == assignment_timestamp
            ),
            "source_league_equals_assignment": (
                source_league is not None and source_league == assignment_league
            ),
            "source_patch_equals_assignment": (
                source_patch is not None and source_patch == assignment_patch
            ),
            "source_team_set_equals_assignment": source_teams == normalized_team_set,
            "scoreboard_prefix_equals_series": (
                series_id is not None and scoreboard_prefix == series_id
            ),
            "scoreboard_game_present": scoreboard_id is not None,
            "series_id_present": series_id is not None,
            "scoreboard_order_positive": isinstance(order, int)
            and not isinstance(order, bool)
            and order > 0,
            "outcome_free": assignment.get("outcome_used") is False,
        }
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise CensusDriftAuditError(
                f"bridge assignment checks failed for {game_id}: {', '.join(failed)}"
            )
        series_ids.add(str(series_id))
        drift_rows.append(
            {
                "game_id": game_id,
                "source_row": {
                    "date": source_date,
                    "gameid": game_id,
                    "league": source_league,
                    "patch": source_patch,
                    "team_keys": list(source_row.get("team_keys") or []),
                    "teams": list(source_row.get("teams") or []),
                    "tournament": source_row.get("tournament"),
                },
                "bridge_assignment": dict(assignment),
                "checks": checks,
                "series_id": series_id,
                "source_date": source_date,
                "source_tournament": source_row.get("tournament"),
                "competition_overview_page": (
                    ((assignment.get("evidence") or {}).get("competition") or {}).get(
                        "overview_page"
                    )
                    if isinstance(assignment.get("evidence"), Mapping)
                    else None
                ),
            }
        )

    blockers = [
        "external_bridge_source_receipt_differs_from_current_accepted_receipt",
        "drift_rows_are_not_in_current_accepted_census",
        "current_census_series_gate_remains_closed",
        "tournament_assignment_not_source_bound",
    ]
    audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "authority": dict(AUTHORITY),
        "decision": {
            "fail_closed": True,
            "drift_rows_series_assignments_verified": True,
            "drift_rows_tournament_fields_verified": False,
            "current_census_series_gate_closed": False,
            "can_promote_tier_evaluation": False,
        },
        "current_source": {
            "source_as_of": _string(current_source.get("source_as_of")),
            "source_game_count": len(current_ids),
            "source_identity_sha256": _sha256(current_source.get("source_identity_sha256")),
            "source_receipt_sha256": _sha256(current_source.get("receipt_sha256")),
            "artifact": dict(current_source_artifact),
        },
        "external_bridge_source": {
            "source_as_of": _string(external_source.get("source_as_of")),
            "source_game_count": len(external_ids),
            "source_identity_sha256": _sha256(external_source.get("source_identity_sha256")),
            "source_receipt_sha256": _sha256(external_source.get("receipt_sha256")),
            "artifact": dict(external_source_artifact),
            "bridge_oe_rows_artifact": dict(bridge_oe_artifact),
        },
        "census_diff": {
            "current_game_count": len(current_ids),
            "external_game_count": len(external_ids),
            "overlap_game_count": len(set(current_ids) & set(external_ids)),
            "external_only_game_count": len(external_only),
            "external_only_game_ids": list(external_only),
            "external_only_identity_sha256": identity_sha256(external_only),
            "current_only_game_count": len(current_only),
            "current_only_game_ids": list(current_only),
            "current_only_identity_sha256": identity_sha256(current_only),
        },
        "bridge_binding": {
            **bridge_binding,
            "crosswalk_artifact": dict(crosswalk_artifact),
            "crosswalk_receipt_artifact": dict(crosswalk_receipt_artifact),
        },
        "drift_rows": drift_rows,
        "series_summary": {
            "verified_series_count": len(series_ids),
            "verified_series_ids": sorted(series_ids),
            "source_tournament_non_null_count": sum(
                row["source_tournament"] is not None for row in drift_rows
            ),
            "competition_overview_pages": sorted(
                {
                    str(row["competition_overview_page"])
                    for row in drift_rows
                    if row["competition_overview_page"]
                }
            ),
        },
        "tournament_boundary": {
            "status": "blocked_current_census",
            "source_bound_tournament_values": False,
            "drift_source_tournament_non_null_count": 0,
            "reason": "The local bridge supplies competition evidence, while the source rows have no tournament field.",
        },
        "blockers": blockers,
    }
    audit["receipt_sha256"] = canonical_sha256(audit)
    return audit


def verify_census_drift_audit(audit: Mapping[str, Any]) -> None:
    """Verify the immutable hash and fail-closed census-drift decision."""

    if not isinstance(audit, Mapping):
        raise CensusDriftAuditError("census-drift audit is required")
    if audit.get("schema_version") != SCHEMA_VERSION:
        raise CensusDriftAuditError("census-drift audit schema is invalid")
    if audit.get("status") != STATUS:
        raise CensusDriftAuditError("census-drift audit status is invalid")
    if dict(audit.get("authority") or {}) != AUTHORITY:
        raise CensusDriftAuditError("census-drift audit authority is invalid")
    claimed = _sha256(audit.get("receipt_sha256"))
    body = dict(audit)
    body.pop("receipt_sha256", None)
    if claimed is None or canonical_sha256(body) != claimed:
        raise CensusDriftAuditError("census-drift audit hash does not match payload")
    decision = audit.get("decision")
    if not isinstance(decision, Mapping):
        raise CensusDriftAuditError("census-drift audit decision is missing")
    if decision.get("fail_closed") is not True:
        raise CensusDriftAuditError("census-drift audit is not fail closed")
    if decision.get("drift_rows_series_assignments_verified") is not True:
        raise CensusDriftAuditError("census-drift rows are not verified")
    for field in (
        "drift_rows_tournament_fields_verified",
        "current_census_series_gate_closed",
        "can_promote_tier_evaluation",
    ):
        if decision.get(field) is not False:
            raise CensusDriftAuditError(f"census-drift decision is unsafe: {field}")
    diff = audit.get("census_diff")
    if not isinstance(diff, Mapping):
        raise CensusDriftAuditError("census-drift difference is missing")
    ids = diff.get("external_only_game_ids")
    if (
        not isinstance(ids, list)
        or len(ids) != DRIFT_GAME_COUNT
        or ids != sorted(ids)
        or len(set(ids)) != len(ids)
        or diff.get("external_only_game_count") != DRIFT_GAME_COUNT
        or diff.get("current_only_game_count") != 0
        or diff.get("current_only_game_ids") != []
    ):
        raise CensusDriftAuditError("census-drift IDs are invalid")
    rows = audit.get("drift_rows")
    if not isinstance(rows, list) or [row.get("game_id") for row in rows] != ids:
        raise CensusDriftAuditError("census-drift row evidence is invalid")
    for row in rows:
        if not isinstance(row, Mapping):
            raise CensusDriftAuditError("census-drift row is invalid")
        checks = row.get("checks")
        if not isinstance(checks, Mapping) or not checks or not all(checks.values()):
            raise CensusDriftAuditError("census-drift row checks are not verified")
        if _string(row.get("series_id")) is None or _string(row.get("source_date")) is None:
            raise CensusDriftAuditError("census-drift row assignment is incomplete")
        assignment = row.get("bridge_assignment")
        if not isinstance(assignment, Mapping) or assignment.get("outcome_used") is not False:
            raise CensusDriftAuditError("census-drift row assignment is not outcome free")
    series_summary = audit.get("series_summary")
    if not isinstance(series_summary, Mapping):
        raise CensusDriftAuditError("census-drift series summary is missing")
    if series_summary.get("verified_series_count") != len(
        {str(row["series_id"]) for row in rows}
    ):
        raise CensusDriftAuditError("census-drift series count is invalid")
    for name, record in (
        ("current_source", audit.get("current_source")),
        ("external_bridge_source", audit.get("external_bridge_source")),
    ):
        if not isinstance(record, Mapping):
            raise CensusDriftAuditError("census-drift source binding is missing")
        for key in ("artifact",):
            if not _record_is_valid(record.get(key)):
                raise CensusDriftAuditError(f"census-drift source artifact is invalid: {key}")
    external = audit.get("external_bridge_source")
    if not isinstance(external, Mapping):
        raise CensusDriftAuditError("census-drift external source binding is missing")
    for key in ("bridge_oe_rows_artifact",):
        if not _record_is_valid(external.get(key)):
            raise CensusDriftAuditError(f"census-drift bridge artifact is invalid: {key}")
    binding = audit.get("bridge_binding")
    if not isinstance(binding, Mapping):
        raise CensusDriftAuditError("census-drift bridge binding is missing")
    for key in ("crosswalk_artifact", "crosswalk_receipt_artifact"):
        if not _record_is_valid(binding.get(key)):
            raise CensusDriftAuditError(f"census-drift bridge record is invalid: {key}")
    if binding.get("crosswalk_hash_valid") is not True:
        raise CensusDriftAuditError("census-drift crosswalk hash is invalid")
    if binding.get("crosswalk_receipt_hash_valid") is not True:
        raise CensusDriftAuditError("census-drift crosswalk receipt hash is invalid")
    if binding.get("crosswalk_receipt_binds_crosswalk") is not True:
        raise CensusDriftAuditError("census-drift crosswalk receipt binding is invalid")
    if binding.get("crosswalk_receipt_binds_artifact") is not True:
        raise CensusDriftAuditError("census-drift crosswalk artifact binding is invalid")
    if binding.get("crosswalk_receipt_binds_assignment_ids") is not True:
        raise CensusDriftAuditError("census-drift crosswalk assignment binding is invalid")
    if binding.get("source_binding_matches") is not True:
        raise CensusDriftAuditError("census-drift source binding is invalid")
    if binding.get("assignments_outcome_free") is not True:
        raise CensusDriftAuditError("census-drift bridge assignments are not outcome free")


__all__ = [
    "AUTHORITY",
    "CensusDriftAuditError",
    "DRIFT_GAME_COUNT",
    "SCHEMA_VERSION",
    "STATUS",
    "build_census_drift_audit",
    "verify_census_drift_audit",
]
