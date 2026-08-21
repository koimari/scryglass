"""Source-bound comparison evidence for future-value snapshots.

The comparison uses the common verified finite ID universe.  It keeps the
full-snapshot rank result explicitly incomparable when the current and future
universes differ.  The report is research-only and has no public authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "scryglass:future-value-snapshot-comparison:v1"
RANK_UNIVERSE = "common_verified_finite_ids"
RANK_DIRECTION = "descending_value_rank_1_highest"
FULL_RANK_STATUS = "incomparable"
IDENTITY_BLOCKER = "current_rating_player_team_identity_missing_for_rank_diffs"
CURRENT_RATING_INPUT_BINDING_SCHEMA = (
    "scryglass:future-value-current-rating-input-binding:v1"
)


class SnapshotComparisonError(ValueError):
    """The source-bound snapshot comparison cannot be trusted."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_text(value: Any, *, label: str) -> str:
    text = str(value or "").lower()
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise SnapshotComparisonError(f"{label} is not a SHA-256 digest")
    return text


def _finite(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise SnapshotComparisonError(f"{label} is not finite") from error
    if not math.isfinite(result):
        raise SnapshotComparisonError(f"{label} is not finite")
    return result


def _payload_receipt_hash(
    payload: Mapping[str, Any], *, label: str, trailing_newline: bool
) -> str:
    claimed = _hash_text(
        payload.get("receipt_sha256"), label=f"{label} receipt_sha256"
    )
    unsigned = dict(payload)
    unsigned.pop("receipt_sha256", None)
    canonical = _canonical_json_bytes(unsigned)
    if trailing_newline:
        canonical += b"\n"
    actual = _sha256_bytes(canonical)
    if actual != claimed:
        raise SnapshotComparisonError(f"{label} receipt self-hash changed")
    return claimed


def _identity_digest(identity: str, values: Iterable[Any]) -> str:
    ids = sorted({str(value) for value in values if str(value).strip()})
    return _sha256_bytes(_canonical_json_bytes({"identity": identity, "ids": ids}))


def _paired_digest(
    rows: Sequence[Mapping[str, Any]],
    *,
    identity: str,
    current_value: str,
    future_value: str,
) -> str:
    normalized: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: str(item.get(identity) or "")):
        key = str(row.get(identity) or "")
        if not key:
            raise SnapshotComparisonError(f"{identity} comparison row has no identity")
        normalized.append(
            {
                identity: key,
                "current_rank": int(row["current_rank"]),
                "future_rank": int(row["future_rank"]),
                "rank_delta": int(row["rank_delta"]),
                "current_value": _finite(row["current_value"], label="current_value"),
                "future_value": _finite(row["future_value"], label="future_value"),
            }
        )
    return _sha256_bytes(_canonical_json_bytes(normalized))


def _authority_payload(authority: Mapping[str, Any] | None) -> dict[str, bool]:
    source = dict(authority or {})
    expected_false = (
        "public_player_rating",
        "public_team_rating",
        "public_probability",
        "promotion",
        "merge",
        "deployment",
        "odds",
        "expected_value",
        "recommendation",
        "betting",
    )
    if source.get("research_only") is not True:
        raise SnapshotComparisonError("snapshot comparison requires research-only authority")
    if any(source.get(field) is True for field in expected_false):
        raise SnapshotComparisonError("snapshot comparison authority is public")
    return {
        "research_only": True,
        **{field: False for field in expected_false},
    }


def _validate_current_snapshot_trust_root(
    trust_root: Mapping[str, Any] | None,
    *,
    current_payload: Mapping[str, Any],
    current_receipt_sha256: str,
    current_receipt_file_sha256: str | None,
    source_receipt_sha256: str,
    player: Mapping[str, Any],
    team: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind common IDs to the independently verified current snapshots.

    Rank artifacts carry their own future-source receipt binding.  Their
    common-ID digest alone does not prove that the current side used the
    pinned rating snapshots.  The comparison therefore requires the binding
    emitted after those snapshots were verified.  It keeps the full ID lists
    in the verification input and emits only their digests.
    """

    if not isinstance(trust_root, Mapping):
        raise SnapshotComparisonError("current snapshot trust root is required")
    if trust_root.get("schema_version") != CURRENT_RATING_INPUT_BINDING_SCHEMA:
        raise SnapshotComparisonError("current snapshot trust root schema changed")
    receipt = trust_root.get("receipt")
    if not isinstance(receipt, Mapping):
        raise SnapshotComparisonError("current snapshot trust root receipt is missing")
    if receipt.get("receipt_sha256") != current_receipt_sha256:
        raise SnapshotComparisonError("current snapshot trust root receipt changed")
    if current_receipt_file_sha256 is None:
        raise SnapshotComparisonError("current snapshot receipt file binding is required")
    expected_file_sha256 = _hash_text(
        current_receipt_file_sha256, label="current receipt file"
    )
    if receipt.get("sha256") != expected_file_sha256:
        raise SnapshotComparisonError("current snapshot trust root file binding changed")
    if receipt.get("schema_version") != current_payload.get("schema_version"):
        raise SnapshotComparisonError("current snapshot trust root receipt schema changed")
    if trust_root.get("source_receipt_sha256") != source_receipt_sha256:
        raise SnapshotComparisonError("current snapshot trust root source receipt changed")
    if trust_root.get("source_identity_sha256") != current_payload.get(
        "source_identity_sha256"
    ):
        raise SnapshotComparisonError("current snapshot trust root source identity changed")
    if trust_root.get("source_as_of") != current_payload.get("source_as_of"):
        raise SnapshotComparisonError("current snapshot trust root source timestamp changed")
    if int(trust_root.get("source_game_count") or -1) != int(
        current_payload.get("source_game_count") or -2
    ):
        raise SnapshotComparisonError("current snapshot trust root source game count changed")

    snapshots = trust_root.get("snapshots")
    current_snapshots = current_payload.get("snapshots")
    if not isinstance(snapshots, Mapping) or not isinstance(current_snapshots, Mapping):
        raise SnapshotComparisonError("current snapshot trust root snapshots are missing")
    if set(snapshots) != {"player", "team"}:
        raise SnapshotComparisonError("current snapshot trust root snapshots changed")

    validated: dict[str, Any] = {
        "schema_version": CURRENT_RATING_INPUT_BINDING_SCHEMA,
        "receipt_sha256": current_receipt_sha256,
        "file_sha256": expected_file_sha256,
        "source_receipt_sha256": source_receipt_sha256,
        "source_identity_sha256": str(
            current_payload.get("source_identity_sha256") or ""
        ),
        "snapshots": {},
    }
    for scope, identity, comparison in (
        ("player", "player_id", player),
        ("team", "team_id", team),
    ):
        root_record = snapshots.get(scope)
        current_record = current_snapshots.get(scope)
        if not isinstance(root_record, Mapping) or not isinstance(
            current_record, Mapping
        ):
            raise SnapshotComparisonError(
                f"current {scope} snapshot trust binding is missing"
            )
        if root_record.get("identity_column") != identity:
            raise SnapshotComparisonError(
                f"current {scope} snapshot trust identity changed"
            )
        if root_record.get("value_column") != comparison["current_value_field"]:
            raise SnapshotComparisonError(
                f"current {scope} snapshot trust value field changed"
            )
        raw_ids = root_record.get("identity_ids")
        if not isinstance(raw_ids, list):
            raise SnapshotComparisonError(
                f"current {scope} snapshot trust IDs are missing"
            )
        ids = [str(value) for value in raw_ids if str(value).strip()]
        if len(ids) != len(raw_ids) or len(set(ids)) != len(ids):
            raise SnapshotComparisonError(
                f"current {scope} snapshot trust IDs are ambiguous"
            )
        identity_sha256 = _identity_digest(identity, ids)
        if root_record.get("identity_sha256") != identity_sha256:
            raise SnapshotComparisonError(
                f"current {scope} snapshot trust identity digest changed"
            )
        if int(root_record.get("verified_rows") or -1) != len(ids):
            raise SnapshotComparisonError(
                f"current {scope} snapshot trust count changed"
            )
        if int(root_record.get("finite_rows") or -1) != len(ids):
            raise SnapshotComparisonError(
                f"current {scope} snapshot trust finite count changed"
            )
        if int(current_record.get("verified_rows") or -1) != len(ids):
            raise SnapshotComparisonError(
                f"current {scope} snapshot receipt count changed"
            )
        if current_record.get("identity_column") not in (None, identity):
            raise SnapshotComparisonError(
                f"current {scope} snapshot receipt identity changed"
            )
        if current_record.get("value_column") != comparison["current_value_field"]:
            raise SnapshotComparisonError(
                f"current {scope} snapshot receipt value field changed"
            )
        common_ids = comparison.get("common_ids")
        if not isinstance(common_ids, list) or not set(common_ids).issubset(set(ids)):
            raise SnapshotComparisonError(
                f"{scope} rank IDs are not bound to current snapshot trust root"
            )
        if int(comparison["current_finite_rows"]) != len(ids):
            raise SnapshotComparisonError(
                f"{scope} current finite count is not source-bound"
            )
        snapshot_file_sha256 = _hash_text(
            root_record.get("sha256"), label=f"current {scope} snapshot file"
        )
        current_file_sha256 = _hash_text(
            current_record.get("sha256"), label=f"current {scope} receipt file"
        )
        if current_file_sha256 != snapshot_file_sha256:
            raise SnapshotComparisonError(
                f"current {scope} snapshot file binding changed"
            )
        validated["snapshots"][scope] = {
            "identity_column": identity,
            "value_column": comparison["current_value_field"],
            "verified_rows": len(ids),
            "finite_rows": len(ids),
            "identity_sha256": identity_sha256,
            "file_sha256": snapshot_file_sha256,
        }
    return validated


def _validate_rank_artifact(
    artifact: Mapping[str, Any],
    *,
    identity: str,
    current_value: str,
    future_value: str,
    current_snapshot_rows: int,
    future_source_receipt_sha256: str,
) -> dict[str, Any]:
    rows = artifact.get("rows")
    if not isinstance(rows, list):
        raise SnapshotComparisonError(f"{identity} rank artifact rows are missing")
    coverage = artifact.get("rank_coverage")
    if not isinstance(coverage, Mapping):
        raise SnapshotComparisonError(f"{identity} rank coverage is missing")
    if artifact.get("source_receipt_sha256") != future_source_receipt_sha256:
        raise SnapshotComparisonError(f"{identity} rank artifact source receipt changed")
    if coverage.get("rank_universe") != RANK_UNIVERSE:
        raise SnapshotComparisonError(f"{identity} rank universe changed")
    if coverage.get("rank_direction") != RANK_DIRECTION:
        raise SnapshotComparisonError(f"{identity} rank direction changed")
    if coverage.get("current_value_field") != current_value:
        raise SnapshotComparisonError(f"{identity} current value field changed")
    if coverage.get("future_value_field") != future_value:
        raise SnapshotComparisonError(f"{identity} future value field changed")
    if not isinstance(coverage.get("full_snapshot_ranks"), Mapping):
        raise SnapshotComparisonError(f"{identity} full rank contract is missing")
    full = dict(coverage["full_snapshot_ranks"])
    if full.get("status") != FULL_RANK_STATUS:
        raise SnapshotComparisonError(f"{identity} full rank status is not incomparable")
    if full.get("current_universe_size") != int(coverage.get("finite_current_rows") or -1):
        raise SnapshotComparisonError(f"{identity} full current universe is not bound")
    if full.get("future_universe_size") != int(coverage.get("finite_future_rows") or -1):
        raise SnapshotComparisonError(f"{identity} full future universe is not bound")
    if full.get("current_value_field") != current_value or full.get("future_value_field") != future_value:
        raise SnapshotComparisonError(f"{identity} full rank fields changed")
    if int(coverage.get("current_rows") or -1) != int(current_snapshot_rows):
        raise SnapshotComparisonError(f"{identity} current snapshot row count changed")
    ids: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise SnapshotComparisonError(f"{identity} rank row is invalid")
        key = str(row.get(identity) or "")
        if not key:
            raise SnapshotComparisonError(f"{identity} rank row identity is missing")
        ids.append(key)
        current_rank = int(row.get("current_rank"))
        future_rank = int(row.get("future_rank"))
        if current_rank < 1 or future_rank < 1:
            raise SnapshotComparisonError(f"{identity} rank is invalid")
        if int(row.get("rank_delta")) != current_rank - future_rank:
            raise SnapshotComparisonError(f"{identity} rank delta changed")
        _finite(row.get("current_value"), label=f"{identity} current value")
        _finite(row.get("future_value"), label=f"{identity} future value")
    if len(set(ids)) != len(ids):
        raise SnapshotComparisonError(f"{identity} rank rows contain duplicate IDs")
    sorted_ids = sorted(ids)
    paired_digest = _paired_digest(
        rows,
        identity=identity,
        current_value=current_value,
        future_value=future_value,
    )
    identity_sha256 = _identity_digest(identity, sorted_ids)
    for field, expected in (
        ("common_universe_size", len(sorted_ids)),
        ("matched_rows", len(sorted_ids)),
        ("common_identity_sha256", identity_sha256),
        ("identity_sha256", identity_sha256),
        ("paired_row_digest_sha256", paired_digest),
        ("paired_row_digest", paired_digest),
    ):
        if coverage.get(field) != expected:
            raise SnapshotComparisonError(f"{identity} rank coverage changed: {field}")
    future_rows = int(coverage.get("future_rows") or -1)
    unmatched_rows = int(coverage.get("unmatched_rows") or -1)
    if future_rows < len(sorted_ids) or unmatched_rows != future_rows - len(sorted_ids):
        raise SnapshotComparisonError(f"{identity} future join counts changed")
    expected_join_rate = float(len(sorted_ids) / future_rows) if future_rows else None
    actual_join_rate = coverage.get("join_rate")
    if expected_join_rate is None:
        if actual_join_rate is not None:
            raise SnapshotComparisonError(f"{identity} empty join rate changed")
    elif not math.isclose(float(actual_join_rate), expected_join_rate, rel_tol=0.0, abs_tol=1e-15):
        raise SnapshotComparisonError(f"{identity} join rate changed")
    return {
        "identity_column": identity,
        "current_value_field": current_value,
        "future_value_field": future_value,
        "rank_universe": RANK_UNIVERSE,
        "rank_direction": RANK_DIRECTION,
        "common_ids": sorted_ids,
        "common_universe_size": len(sorted_ids),
        "common_identity_sha256": identity_sha256,
        "paired_row_digest_sha256": paired_digest,
        "current_snapshot_rows": int(current_snapshot_rows),
        "current_finite_rows": int(coverage["finite_current_rows"]),
        "future_snapshot_rows": future_rows,
        "future_finite_rows": int(coverage["finite_future_rows"]),
        "matched_rows": len(sorted_ids),
        "unmatched_rows": unmatched_rows,
        "join_rate": expected_join_rate,
        "full_snapshot_rank_status": FULL_RANK_STATUS,
        "full_snapshot_ranks": full,
        "artifact_rows": len(rows),
        "status": str(coverage.get("status") or "partial"),
    }


def build_snapshot_comparison_report(
    *,
    current_receipt: Mapping[str, Any],
    future_receipt: Mapping[str, Any],
    player_rank_diff_artifact: Mapping[str, Any],
    team_rank_diff_artifact: Mapping[str, Any],
    current_receipt_file_sha256: str | None = None,
    future_receipt_file_sha256: str | None = None,
    player_rank_diff_file_sha256: str | None = None,
    team_rank_diff_file_sha256: str | None = None,
    expected_source_receipt_sha256: str | None = None,
    current_snapshot_trust_root: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify and summarize one current-versus-future snapshot comparison."""

    current_payload = dict(current_receipt)
    future_payload = dict(future_receipt)
    current_hash = _payload_receipt_hash(
        current_payload, label="current snapshot", trailing_newline=True
    )
    future_hash = _payload_receipt_hash(
        future_payload, label="future snapshot", trailing_newline=False
    )
    if current_payload.get("schema_version") != "scryglass:current-rating-snapshot-receipt:v1":
        raise SnapshotComparisonError("current snapshot receipt schema changed")
    if future_payload.get("schema_version") != "scryglass:future-value-snapshot-receipt:v1":
        raise SnapshotComparisonError("future snapshot receipt schema changed")
    authority = _authority_payload(future_payload.get("authority"))
    source = future_payload.get("source")
    if not isinstance(source, Mapping):
        raise SnapshotComparisonError("future snapshot source binding is missing")
    source_hash = _hash_text(source.get("source_receipt_sha256"), label="source receipt")
    if expected_source_receipt_sha256 is not None and source_hash != _hash_text(
        expected_source_receipt_sha256, label="expected source receipt"
    ):
        raise SnapshotComparisonError("source receipt changed")
    if current_payload.get("source_receipt_sha256") != source_hash:
        raise SnapshotComparisonError("current snapshot source receipt changed")
    if current_payload.get("source_identity_sha256") != source.get("source_identity_sha256"):
        raise SnapshotComparisonError("snapshot source identity changed")
    if int(current_payload.get("source_game_count") or -1) != int(source.get("source_game_count") or -2):
        raise SnapshotComparisonError("snapshot source game count changed")
    current_files = {
        "receipt_sha256": current_hash,
        "file_sha256": None
        if current_receipt_file_sha256 is None
        else _hash_text(current_receipt_file_sha256, label="current receipt file"),
    }
    future_files = {
        "receipt_sha256": future_hash,
        "file_sha256": None
        if future_receipt_file_sha256 is None
        else _hash_text(future_receipt_file_sha256, label="future receipt file"),
    }
    snapshots = current_payload.get("snapshots")
    if not isinstance(snapshots, Mapping):
        raise SnapshotComparisonError("current snapshot bindings are missing")
    player_record = snapshots.get("player")
    team_record = snapshots.get("team")
    if not isinstance(player_record, Mapping) or not isinstance(team_record, Mapping):
        raise SnapshotComparisonError("current player/team snapshot bindings are missing")
    player = _validate_rank_artifact(
        player_rank_diff_artifact,
        identity="player_id",
        current_value=str(player_record.get("value_column") or ""),
        future_value="future_player_value_logit",
        current_snapshot_rows=int(player_record.get("verified_rows") or -1),
        future_source_receipt_sha256=source_hash,
    )
    team = _validate_rank_artifact(
        team_rank_diff_artifact,
        identity="team_id",
        current_value=str(team_record.get("value_column") or ""),
        future_value="future_team_value_logit",
        current_snapshot_rows=int(team_record.get("verified_rows") or -1),
        future_source_receipt_sha256=source_hash,
    )
    current_trust = _validate_current_snapshot_trust_root(
        current_snapshot_trust_root,
        current_payload=current_payload,
        current_receipt_sha256=current_hash,
        current_receipt_file_sha256=current_receipt_file_sha256,
        source_receipt_sha256=source_hash,
        player=player,
        team=team,
    )
    raw_blockers: list[str] = []
    for candidate in (
        future_payload.get("blockers"),
        future_payload.get("fit", {}).get("blockers")
        if isinstance(future_payload.get("fit"), Mapping)
        else None,
    ):
        if isinstance(candidate, list):
            raw_blockers.extend(str(value) for value in candidate)
    inherited = sorted(
        {
            str(value)
            for value in raw_blockers
            if str(value) in {IDENTITY_BLOCKER, "current_player_team_rating_comparison_missing"}
        }
    )
    model_authorization_blockers = [
        value for value in inherited if value == IDENTITY_BLOCKER
    ]
    independent_join = {
        "status": "verified",
        "source_bound": True,
        "rank_universe": RANK_UNIVERSE,
        "rank_direction": RANK_DIRECTION,
        "player_rows": int(player["matched_rows"]),
        "team_rows": int(team["matched_rows"]),
        "rank_pairs": {
            "player": int(player["matched_rows"]),
            "team": int(team["matched_rows"]),
        },
        "player_rank_pairs": int(player["matched_rows"]),
        "team_rank_pairs": int(team["matched_rows"]),
        "common_verified_finite_ids": {
            "player": int(player["common_universe_size"]),
            "team": int(team["common_universe_size"]),
        },
        "join_rates": {
            "player": player["join_rate"],
            "team": team["join_rate"],
        },
        "full_snapshot_rank_status": {
            "player": player["full_snapshot_rank_status"],
            "team": team["full_snapshot_rank_status"],
        },
        "current_value_fields": {
            "player": player["current_value_field"],
            "team": team["current_value_field"],
        },
        "future_value_fields": {
            "player": player["future_value_field"],
            "team": team["future_value_field"],
        },
        "player": {
            "rank_pairs": int(player["matched_rows"]),
            "common_verified_finite_ids": int(player["common_universe_size"]),
            "common_identity_sha256": player["common_identity_sha256"],
            "join_rate": player["join_rate"],
            "full_snapshot_rank_status": player["full_snapshot_rank_status"],
            "current_finite_rows": player["current_finite_rows"],
            "future_finite_rows": player["future_finite_rows"],
            "future_rows": player["future_snapshot_rows"],
            "unmatched_rows": player["unmatched_rows"],
        },
        "team": {
            "rank_pairs": int(team["matched_rows"]),
            "common_verified_finite_ids": int(team["common_universe_size"]),
            "common_identity_sha256": team["common_identity_sha256"],
            "join_rate": team["join_rate"],
            "full_snapshot_rank_status": team["full_snapshot_rank_status"],
            "current_finite_rows": team["current_finite_rows"],
            "future_finite_rows": team["future_finite_rows"],
            "future_rows": team["future_snapshot_rows"],
            "unmatched_rows": team["unmatched_rows"],
        },
        "current_snapshot_trust_root": current_trust,
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "research_only_partial",
        "authority": authority,
        "source": {
            "source_receipt_sha256": source_hash,
            "source_identity_sha256": str(source.get("source_identity_sha256") or ""),
            "source_as_of": str(source.get("source_as_of") or ""),
            "source_game_count": int(source.get("source_game_count") or -1),
        },
        "receipts": {
            "current": current_files,
            "future": future_files,
            "future_rank_diff_artifacts": {
                "player_file_sha256": None
                if player_rank_diff_file_sha256 is None
                else _hash_text(
                    player_rank_diff_file_sha256, label="player rank diff file"
                ),
                "team_file_sha256": None
                if team_rank_diff_file_sha256 is None
                else _hash_text(
                    team_rank_diff_file_sha256, label="team rank diff file"
                ),
            },
            "future_model_receipt_sha256": future_payload.get("model", {}).get("receipt_sha256")
            if isinstance(future_payload.get("model"), Mapping)
            else None,
        },
        "snapshot_comparisons": {"player": player, "team": team},
        "full_snapshot_rank_status": FULL_RANK_STATUS,
        "independent_join": independent_join,
        "inherited_authorization_blockers": inherited,
        "model_authorization_blocker": {
            "status": "stale_inherited" if model_authorization_blockers else "none",
            "label": (
                "stale_inherited_for_model_authorization"
                if model_authorization_blockers
                else "none"
            ),
            "stale": bool(model_authorization_blockers),
            "inherited": bool(model_authorization_blockers),
            "scope": "model_authorization",
            "blockers": model_authorization_blockers,
        },
        "blocker_context": {
            "status": "stale_inherited" if model_authorization_blockers else "none",
            "independent_join_status": "verified",
            "full_snapshot_rank_status": FULL_RANK_STATUS,
            "identity_blocker_is_join_failure": False,
            "model_authorization_status": (
                "stale_inherited" if model_authorization_blockers else "none"
            ),
        },
        "blockers": sorted(set(raw_blockers)),
    }
    report["report_sha256"] = _sha256_bytes(_canonical_json_bytes(report))
    return report


__all__ = [
    "CURRENT_RATING_INPUT_BINDING_SCHEMA",
    "FULL_RANK_STATUS",
    "IDENTITY_BLOCKER",
    "RANK_DIRECTION",
    "RANK_UNIVERSE",
    "SCHEMA_VERSION",
    "SnapshotComparisonError",
    "build_snapshot_comparison_report",
]
