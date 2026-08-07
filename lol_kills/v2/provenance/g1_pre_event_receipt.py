"""Non-authorizing G1 pre-event interface receipt.

This receipt binds the accepted non-final LPL snapshot to the outcome-free v3
completed-draft feature replay.  It is an identity handoff for downstream
development modules, not a source-authority upgrade: observed map lineups are
never relabelled as pre-event roster authority, and target/final-holdout rows
are never opened.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from lol_kills.v2.data.source_tree import resolve_repository_file


ROOT = Path(__file__).resolve().parents[3]
BASE_MANIFEST = ROOT / "data/lol/v2/snapshots/real-v1/lpl-private-development-manifest.json"
FEATURE_MANIFEST = ROOT / "data/lol/v2/snapshots/real-v1/lpl-private-draft-features-v3-manifest.json"
FEATURE_ROWS = ROOT / "data/lol/v2/snapshots/real-v1/lpl-private-draft-features-v3-rows.jsonl"
TRANSFORM = ROOT / "lol_kills/v2/data/g1_draft_features.py"
RECEIPT_PATH = ROOT / "data/lol/v2/snapshots/real-v1/g1-pre-event-interface-receipt.json"
SCHEMA_PATH = ROOT / "data/lol/v2/snapshots/real-v1/g1-pre-event-interface-receipt.schema.json"

SCHEMA_VERSION = "scryglass:g1-pre-event-interface-receipt:v1"
ARTIFACT_KIND = "G1_PRE_EVENT_INTERFACE_RECEIPT"
STATUS = "PRIVATE_NONAUTHORIZING_PRE_EVENT_INPUT"

BASE_MANIFEST_RAW_SHA256 = "dca3c7b8fb5c6bc6bc4ebf8779448bbbb0728a592d85461bee479c5f39d608e1"
BASE_MANIFEST_CANONICAL_SHA256 = "3af87fffb2b32fd95aeb920409abe0254fa158b3dc7f079650b3472731d4ff72"
BASE_ROWS_RAW_SHA256 = "4ed79abb0b2471a666ab5643b91edf33c2fdde19e361c456aa589d2e9a4df846"
FEATURE_MANIFEST_RAW_SHA256 = "9777180fcd57fc55e3b84da39533bfcaf5a8216b727656070b9d49c7f801cf45"
FEATURE_MANIFEST_CANONICAL_SHA256 = "c0ea9d72e4d5fc7a1e4e372563b73ed365ee5eb9e0fb53b000c071c58c9a6395"
FEATURE_ROWS_RAW_SHA256 = "e742631e1c12fb1af7148468a0d595ff6cf23e816af4edb20af162a04a6a9680"
FEATURE_ROWS_CANONICAL_SHA256 = "52d59dd0c41a212f7eb07b6f6132841f3c152f28324308b376042f8e262c141d"
TRANSFORM_RAW_SHA256 = "95a7bd8cf903b6f44c6a6c553250989d261dd4f82809432dd03fbe3aa6a5b4a3"
TRANSFORM_CONFIG_SHA256 = "5feac1493a5495fff561ccc39e830f855f4c9ba0508c1007eb4549e9df2b2e27"

_FORBIDDEN_KEYS = {
    "target",
    "outcome",
    "result",
    "winner",
    "map_winner",
    "y_blue_win",
    "y_red_win",
    "y_total_kills",
}


class G1PreEventReceiptError(ValueError):
    """Raised when the pre-event interface cannot be replayed exactly."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise G1PreEventReceiptError("noncanonical G1 pre-event payload") from error


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _raw_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _safe_path(path: Path, *, label: str) -> Path:
    try:
        relative = path.relative_to(ROOT)
        resolved = resolve_repository_file(ROOT, relative.as_posix())
    except Exception as error:
        raise G1PreEventReceiptError(f"{label} path is unsafe") from error
    metadata = os.lstat(resolved)
    if metadata.st_nlink != 1:
        raise G1PreEventReceiptError(f"{label} must be an unaliased regular file")
    return resolved


def _read_pinned(path: Path, expected_raw_sha256: str, *, label: str) -> bytes:
    safe = _safe_path(path, label=label)
    raw = safe.read_bytes()
    if _raw_sha256(raw) != expected_raw_sha256:
        raise G1PreEventReceiptError(f"{label} raw sha256 mismatch")
    return raw


def _load_manifest(path: Path, expected_raw: str, expected_canonical: str, *, label: str) -> dict[str, Any]:
    raw = _read_pinned(path, expected_raw, label=label)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise G1PreEventReceiptError(f"{label} is not JSON") from error
    if not isinstance(payload, dict):
        raise G1PreEventReceiptError(f"{label} must be an object")
    if raw != _canonical_bytes(payload) + b"\n":
        raise G1PreEventReceiptError(f"{label} is not canonical")
    claimed = payload.get("manifest_sha256")
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    if claimed != expected_canonical or _sha256(unsigned) != expected_canonical:
        raise G1PreEventReceiptError(f"{label} canonical digest mismatch")
    return payload


def _assert_outcome_free(value: Any, *, path: str = "row") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise G1PreEventReceiptError(f"non-string feature key at {path}")
            normalized = key.casefold()
            if normalized in _FORBIDDEN_KEYS or normalized.startswith("y_"):
                raise G1PreEventReceiptError(f"outcome key present at {path}.{key}")
            _assert_outcome_free(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_outcome_free(child, path=f"{path}[{index}]")


def _load_feature_rows(path: Path = FEATURE_ROWS, expected_raw: str = FEATURE_ROWS_RAW_SHA256) -> list[dict[str, Any]]:
    raw = _read_pinned(path, expected_raw, label="v3 feature rows")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise G1PreEventReceiptError(f"v3 feature row {line_number} is not JSON") from error
        if not isinstance(row, dict):
            raise G1PreEventReceiptError(f"v3 feature row {line_number} must be an object")
        _assert_outcome_free(row, path=f"row[{line_number}]")
        rows.append(row)
    if _sha256(rows) != FEATURE_ROWS_CANONICAL_SHA256:
        raise G1PreEventReceiptError("v3 feature rows canonical digest mismatch")
    if len(rows) != 1226 or sum(len(row.get("picks", ())) for row in rows) != 12260:
        raise G1PreEventReceiptError("v3 feature row/pick count drift")
    return rows


def _blockers() -> list[dict[str, str]]:
    return [
        {"code": "CURRENT_ROSTER_AUTHORITY_UNAVAILABLE", "scope": "all_rows", "claim_effect": "PRE_EVENT_AND_CURRENT_ROSTER_CLAIMS_BLOCKED"},
        {"code": "AUTHORITATIVE_SERIES_CROSSWALK_UNBOUND", "scope": "all_rows", "claim_effect": "PROVIDER_SERIES_IDENTITY_CLAIMS_BLOCKED"},
        {"code": "HISTORICAL_INGEST_RECEIPT_UNAVAILABLE", "scope": "all_rows", "claim_effect": "LIVE_FORECAST_AND_AS_OF_AVAILABILITY_CLAIMS_BLOCKED"},
        {"code": "GRID_HISTORICAL_PAYLOAD_NOT_AUTHORIZED_OR_DOWNLOADED", "scope": "grid", "claim_effect": "GRID_ROW_COVERAGE_AND_COMPLETENESS_CLAIMS_BLOCKED"},
        {"code": "G1_UNIFIED_BENCHMARK_AUTHORITY_BUNDLE_UNAVAILABLE", "scope": "real-v1", "claim_effect": "SOURCE_BOUND_BENCHMARK_TRANSITION_BLOCKED"},
        {"code": "G1_018_BASELINES_TYPED_UNAVAILABLE", "scope": "real-v1", "claim_effect": "REQUIRED_COMPARISON_BLOCKED_UNTIL_RESOLVED"},
        {"code": "FINAL_HOLDOUT_SEALED", "scope": "real-v1", "claim_effect": "FINAL_HOLDOUT_RESULT_CLAIMS_BLOCKED"},
    ]


def build_receipt(
    *,
    base_manifest_path: Path = BASE_MANIFEST,
    feature_manifest_path: Path = FEATURE_MANIFEST,
    feature_rows_path: Path = FEATURE_ROWS,
    transform_path: Path = TRANSFORM,
) -> dict[str, Any]:
    base = _load_manifest(base_manifest_path, BASE_MANIFEST_RAW_SHA256, BASE_MANIFEST_CANONICAL_SHA256, label="base G1 manifest")
    feature = _load_manifest(feature_manifest_path, FEATURE_MANIFEST_RAW_SHA256, FEATURE_MANIFEST_CANONICAL_SHA256, label="v3 feature manifest")
    rows = _load_feature_rows(feature_rows_path)
    transform = _safe_path(transform_path, label="feature transform")
    transform_raw = transform.read_bytes()
    if _raw_sha256(transform_raw) != TRANSFORM_RAW_SHA256:
        raise G1PreEventReceiptError("feature transform raw sha256 mismatch")
    config = feature.get("transform", {}).get("config")
    if not isinstance(config, Mapping) or feature.get("transform", {}).get("config_sha256") != TRANSFORM_CONFIG_SHA256 or _sha256(config) != TRANSFORM_CONFIG_SHA256:
        raise G1PreEventReceiptError("feature transform config identity mismatch")
    if base.get("final_holdout") != {"accessed": False, "cutoff_local_naive": "2026-06-01T00:00:00", "status": "SEALED_UNREAD"}:
        raise G1PreEventReceiptError("base final holdout boundary changed")
    if feature.get("final_holdout") != {"accessed": False, "included": False, "status": "SEALED_UNREAD"}:
        raise G1PreEventReceiptError("feature final holdout boundary changed")
    if feature.get("base_g1", {}).get("manifest_canonical_sha256") != BASE_MANIFEST_CANONICAL_SHA256 or feature.get("base_g1", {}).get("rows_raw_sha256") != BASE_ROWS_RAW_SHA256:
        raise G1PreEventReceiptError("feature/base G1 identity mismatch")
    if feature.get("claim_ceiling", {}).get("private_model_fit") is not True or feature.get("claim_ceiling", {}).get("private_rank_selection") is not True:
        raise G1PreEventReceiptError("feature private-use claim ceiling changed")
    if feature.get("claim_ceiling", {}).get("prediction") is not False or feature.get("claim_ceiling", {}).get("final_holdout") is not False:
        raise G1PreEventReceiptError("feature claim ceiling expanded")
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "status": STATUS,
        "source_snapshot": {
            "manifest_locator": str(base_manifest_path.relative_to(ROOT)),
            "manifest_raw_sha256": BASE_MANIFEST_RAW_SHA256,
            "manifest_sha256": BASE_MANIFEST_CANONICAL_SHA256,
            "rows_locator": base["rows_locator"],
            "rows_raw_sha256": BASE_ROWS_RAW_SHA256,
            "source_scope": base["source_scope"],
            "map_count": base["coverage"]["map_count"],
            "source_series_family_count": base["coverage"]["source_series_family_count"],
            "partition_counts": base["coverage"]["partition_counts"],
            "split_payload_sha256": base["target_authority"]["split_payload_sha256"],
        },
        "feature_snapshot": {
            "manifest_locator": str(feature_manifest_path.relative_to(ROOT)),
            "manifest_raw_sha256": FEATURE_MANIFEST_RAW_SHA256,
            "manifest_sha256": FEATURE_MANIFEST_CANONICAL_SHA256,
            "rows_locator": feature["rows_locator"],
            "rows_raw_sha256": FEATURE_ROWS_RAW_SHA256,
            "rows_canonical_sha256": FEATURE_ROWS_CANONICAL_SHA256,
            "feature_row_count": feature["coverage"]["feature_row_count"],
            "pick_count": feature["coverage"]["pick_count"],
            "transform": {
                "locator": str(transform_path.relative_to(ROOT)),
                "raw_sha256": TRANSFORM_RAW_SHA256,
                "config_sha256": TRANSFORM_CONFIG_SHA256,
            },
        },
        "row_interface": {
            "map_id": "source_game_id",
            "series_id": "source_series_id",
            "event_time": "source_local_event_start",
            "partition": "partition",
            "lineups": "observed_lineups",
            "availability": "COMPLETED_DRAFT_AVAILABLE_AT_OR_BEFORE_EVENT_START",
            "roster_authority": "OBSERVED_MAP_PARTICIPANTS_NOT_PRE_EVENT_ROSTER_AUTHORITY",
            "target_included": False,
        },
        "final_holdout": {"status": "SEALED_UNREAD", "accessed": False, "included": False},
        "rights_status": "PRIVATE_REVIEWED",
        "claim_ceiling": {
            "private_model_fit_feature_input": True,
            "private_rank_selection_feature_input": True,
            "current_roster": False,
            "pre_event_roster": False,
            "forecast": False,
            "prediction": False,
            "production": False,
            "publication": False,
            "promotion": False,
            "sota": False,
            "final_holdout": False,
        },
        "typed_blockers": _blockers(),
    }
    body["receipt_sha256"] = _sha256(body)
    return body


def validate_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise G1PreEventReceiptError("G1 pre-event receipt must be an object")
    claimed = payload.get("receipt_sha256")
    unsigned = dict(payload)
    unsigned.pop("receipt_sha256", None)
    if not isinstance(claimed, str) or _sha256(unsigned) != claimed:
        raise G1PreEventReceiptError("G1 pre-event receipt digest is invalid")
    expected = build_receipt()
    if dict(payload) != expected:
        raise G1PreEventReceiptError("G1 pre-event receipt differs from replayed evidence")
    return expected


def canonical_receipt_bytes(payload: Mapping[str, Any]) -> bytes:
    validate_receipt(payload)
    return _canonical_bytes(dict(payload)) + b"\n"


__all__ = [
    "BASE_MANIFEST_CANONICAL_SHA256",
    "BASE_MANIFEST_RAW_SHA256",
    "BASE_ROWS_RAW_SHA256",
    "FEATURE_MANIFEST_CANONICAL_SHA256",
    "FEATURE_MANIFEST_RAW_SHA256",
    "FEATURE_ROWS_CANONICAL_SHA256",
    "FEATURE_ROWS_RAW_SHA256",
    "G1PreEventReceiptError",
    "RECEIPT_PATH",
    "SCHEMA_PATH",
    "SCHEMA_VERSION",
    "build_receipt",
    "canonical_receipt_bytes",
    "validate_receipt",
]
