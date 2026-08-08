"""Fail-closed G1 source inventory and authority-boundary receipt.

This module does not create source authority.  It reopens already accepted
private LPL artifacts by independently pinned raw bytes, records the frozen
benchmark boundary, and reduces the GRID capability catalog to provenance
metadata.  Missing roster/series/historical-ingest evidence remains typed
unavailable.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from ..data.source_tree import canonical_source_tree_sha256, resolve_repository_file


SCHEMA_VERSION = "scryglass:g1-source-authority-inventory:v1"
RECEIPT_DISPOSITION = "SOURCE_INVENTORY_BOUND_AUTHORITY_UNAVAILABLE"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GRID_CATALOG = (
    Path.home()
    / ".codex"
    / "skills"
    / "query-grid-research"
    / "assets"
    / "grid-capability-catalog.v1.json"
)

SOURCE_CONTRACT_ALLOWLIST = (
    "lol_kills/v2/data/rosters.py",
    "lol_kills/v2/data/series.py",
    "lol_kills/v2/data/source_tree.py",
    "lol_kills/v2/provenance/g1_source_authority.py",
)

_PINNED_REPOSITORY_ARTIFACTS = {
    "private_lpl_snapshot_manifest": {
        "locator": "data/lol/v2/snapshots/real-v1/lpl-private-development-manifest.json",
        "raw_sha256": "dca3c7b8fb5c6bc6bc4ebf8779448bbbb0728a592d85461bee479c5f39d608e1",
    },
    "private_lpl_snapshot_rows": {
        "locator": "data/lol/v2/snapshots/real-v1/lpl-private-development-rows.jsonl",
        "raw_sha256": "4ed79abb0b2471a666ab5643b91edf33c2fdde19e361c456aa589d2e9a4df846",
    },
    "completed_draft_manifest": {
        "locator": "data/lol/v2/snapshots/real-v1/lpl-private-draft-features-manifest.json",
        "raw_sha256": "c806505ac5dfb9eabf00921ed5176f6d295af84bf577179fab3ccf68c216690f",
    },
    "completed_draft_rows": {
        "locator": "data/lol/v2/snapshots/real-v1/lpl-private-draft-features-rows.jsonl",
        "raw_sha256": "e742631e1c12fb1af7148468a0d595ff6cf23e816af4edb20af162a04a6a9680",
    },
    "completed_draft_review": {
        "locator": "data/lol/v2/snapshots/real-v1/lpl-private-draft-features-review.json",
        "raw_sha256": "eb8bde9730421469520a60383282d2810904fffc5896f24263135a0b96a079fb",
    },
    "benchmark_contract_manifest": {
        "locator": "data/lol/v2/evaluation/real-v1/contract-manifest.json",
        "raw_sha256": "22695be602fae2ef116779b502d9495937b60e8fc7af00edceb20c991928471c",
    },
    "benchmark_contract": {
        "locator": "data/lol/v2/evaluation/real-v1/benchmark-contract.json",
        "raw_sha256": "b77fe451105d6e216b71928ad2381117c1ba5d0a5bce30b0b658414ab8559128",
    },
    "benchmark_candidate_registry": {
        "locator": "data/lol/v2/evaluation/real-v1/candidate-registry.json",
        "raw_sha256": "02dc4bd730fb6162918eb5fc565d795dde0da1745f88b4439219254da6b42697",
    },
}

_EXPECTED_GRID_CAPABILITIES = {
    "central_metadata": "confirmed",
    "identity_crosswalk_queries": "confirmed",
    "live_or_final_series_state_snapshot": "confirmed",
    "historical_file_listing": "confirmed",
    "historical_file_download": "not_tested",
    "series_events_websocket": "locally_observed_and_configured",
}

_BLOCKERS = (
    {
        "code": "CURRENT_ROSTER_AUTHORITY_UNAVAILABLE",
        "scope": "all_rows",
        "claim_effect": "PRE_EVENT_AND_CURRENT_ROSTER_CLAIMS_BLOCKED",
    },
    {
        "code": "AUTHORITATIVE_SERIES_CROSSWALK_UNBOUND",
        "scope": "all_rows",
        "claim_effect": "PROVIDER_SERIES_IDENTITY_CLAIMS_BLOCKED",
    },
    {
        "code": "HISTORICAL_INGEST_RECEIPT_UNAVAILABLE",
        "scope": "all_rows",
        "claim_effect": "LIVE_FORECAST_AND_AS_OF_AVAILABILITY_CLAIMS_BLOCKED",
    },
    {
        "code": "GRID_HISTORICAL_PAYLOAD_NOT_AUTHORIZED_OR_DOWNLOADED",
        "scope": "grid",
        "claim_effect": "GRID_ROW_COVERAGE_AND_COMPLETENESS_CLAIMS_BLOCKED",
    },
    {
        "code": "G1_UNIFIED_BENCHMARK_AUTHORITY_BUNDLE_UNAVAILABLE",
        "scope": "real-v1",
        "claim_effect": "SOURCE_BOUND_BENCHMARK_TRANSITION_BLOCKED",
    },
    {
        "code": "FINAL_HOLDOUT_SEALED",
        "scope": "real-v1",
        "claim_effect": "FINAL_HOLDOUT_RESULT_CLAIMS_BLOCKED",
    },
)


class G1SourceAuthorityError(ValueError):
    """Raised when the inventory would overstate or lose its source boundary."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_repo_bytes(root: Path, locator: str) -> bytes:
    path = resolve_repository_file(root, locator)
    metadata = os.lstat(path)
    if metadata.st_nlink != 1:
        raise G1SourceAuthorityError(f"repository artifact must not be hard-linked: {locator}")
    return path.read_bytes()


def _read_pinned_repo_bytes(root: Path, name: str) -> bytes:
    binding = _PINNED_REPOSITORY_ARTIFACTS[name]
    raw = _read_repo_bytes(root, binding["locator"])
    if _sha256_bytes(raw) != binding["raw_sha256"]:
        raise G1SourceAuthorityError(f"pinned repository artifact drifted: {name}")
    return raw


def _decode_object(raw: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise G1SourceAuthorityError(f"{name} must be UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise G1SourceAuthorityError(f"{name} must be a JSON object")
    return value


def _validate_embedded_digest(value: Mapping[str, Any], field: str, name: str) -> str:
    claimed = value.get(field)
    if not _is_sha256(claimed):
        raise G1SourceAuthorityError(f"{name} lacks a valid {field}")
    unsigned = dict(value)
    unsigned.pop(field)
    if _canonical_sha256(unsigned) != claimed:
        raise G1SourceAuthorityError(f"{name} {field} does not match canonical content")
    return claimed


def _read_external_regular(path: Path) -> bytes:
    absolute = path.expanduser().absolute()
    current = Path(absolute.anchor)
    for index, component in enumerate(absolute.parts[1:]):
        current = current / component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as error:
            raise G1SourceAuthorityError("GRID capability catalog is missing") from error
        if os.path.islink(current):
            raise G1SourceAuthorityError("GRID capability catalog path must not contain a symlink")
        if index < len(absolute.parts) - 2 and not os.path.isdir(current):
            raise G1SourceAuthorityError("GRID capability catalog parent must be a directory")
    if not os.path.isfile(absolute) or os.lstat(absolute).st_nlink != 1:
        raise G1SourceAuthorityError("GRID capability catalog must be an unaliased regular file")
    return absolute.read_bytes()


def _grid_catalog_provenance(catalog_path: Path) -> dict[str, Any]:
    raw = _read_external_regular(catalog_path)
    catalog = _decode_object(raw, "GRID capability catalog")
    if catalog.get("schema_version") != "scryglass.grid.capability-catalog.v1":
        raise G1SourceAuthorityError("unexpected GRID capability catalog schema")
    if catalog.get("catalog_version") != "1.0.0":
        raise G1SourceAuthorityError("unexpected GRID capability catalog version")
    claimed = catalog.get("catalog_sha256")
    unsigned = dict(catalog)
    unsigned.pop("catalog_sha256", None)
    if not _is_sha256(claimed) or _canonical_sha256(unsigned) != claimed:
        raise G1SourceAuthorityError("GRID capability catalog digest is invalid")

    capabilities_raw = catalog.get("capabilities")
    if not isinstance(capabilities_raw, list):
        raise G1SourceAuthorityError("GRID catalog capabilities must be a list")
    capability_status: dict[str, str] = {}
    for row in capabilities_raw:
        if not isinstance(row, Mapping):
            raise G1SourceAuthorityError("GRID catalog capability entry must be an object")
        capability = row.get("capability")
        status = row.get("status")
        if not isinstance(capability, str) or not isinstance(status, str):
            raise G1SourceAuthorityError("GRID catalog capability identity/status is invalid")
        if capability in capability_status:
            raise G1SourceAuthorityError(f"duplicate GRID capability: {capability}")
        capability_status[capability] = status
    for capability, expected_status in _EXPECTED_GRID_CAPABILITIES.items():
        if capability_status.get(capability) != expected_status:
            raise G1SourceAuthorityError(
                f"GRID capability is unavailable or changed: {capability}"
            )

    provenance = catalog.get("provenance")
    if not isinstance(provenance, Mapping):
        raise G1SourceAuthorityError("GRID catalog provenance is missing")
    if provenance.get("credentials_serialized") is not False:
        raise G1SourceAuthorityError("GRID catalog serialized credentials")
    if provenance.get("match_files_downloaded") is not False:
        raise G1SourceAuthorityError("GRID catalog discovery downloaded match files")
    if provenance.get("websocket_connections_opened") is not False:
        raise G1SourceAuthorityError("GRID catalog discovery opened a websocket")

    scope = catalog.get("scope")
    if not isinstance(scope, Mapping):
        raise G1SourceAuthorityError("GRID catalog scope is missing")
    for field in (
        "market_edge_claim_authority",
        "model_authority",
        "publication",
        "redistribution",
    ):
        if scope.get(field) is not False:
            raise G1SourceAuthorityError(f"GRID catalog scope unexpectedly authorizes {field}")

    file_probe = catalog.get("file_listing_probe")
    if not isinstance(file_probe, Mapping):
        raise G1SourceAuthorityError("GRID file-listing probe metadata is missing")
    if file_probe.get("status") != "confirmed":
        raise G1SourceAuthorityError("GRID file listing is not confirmed")
    if file_probe.get("download_attempted") is not False:
        raise G1SourceAuthorityError("GRID file-listing probe attempted a download")
    if file_probe.get("signed_urls_retained") is not False:
        raise G1SourceAuthorityError("GRID file-listing probe retained signed URLs")

    endpoints = catalog.get("endpoints")
    if not isinstance(endpoints, list):
        raise G1SourceAuthorityError("GRID catalog endpoints must be a list")
    endpoint_schema_sha256: dict[str, str] = {}
    for endpoint in endpoints:
        if not isinstance(endpoint, Mapping):
            raise G1SourceAuthorityError("GRID catalog endpoint must be an object")
        endpoint_id = endpoint.get("endpoint_id")
        schema_sha256 = endpoint.get("schema_sha256")
        introspection = endpoint.get("introspection")
        if not isinstance(endpoint_id, str) or not _is_sha256(schema_sha256):
            raise G1SourceAuthorityError("GRID endpoint identity/schema digest is invalid")
        if endpoint.get("authenticated") is not True or endpoint.get("read_only_discovery") is not True:
            raise G1SourceAuthorityError("GRID endpoint was not authenticated read-only discovery")
        if _canonical_sha256(introspection) != schema_sha256:
            raise G1SourceAuthorityError(f"GRID endpoint schema digest drifted: {endpoint_id}")
        endpoint_schema_sha256[endpoint_id] = schema_sha256
    if set(endpoint_schema_sha256) != {"central_data", "series_state"}:
        raise G1SourceAuthorityError("GRID endpoint inventory changed")

    generated_at = catalog.get("generated_at")
    if not isinstance(generated_at, str) or not generated_at.endswith("Z"):
        raise G1SourceAuthorityError("GRID catalog generated_at must be UTC")
    return {
        "logical_locator": "query-grid-research/assets/grid-capability-catalog.v1.json",
        "external_to_repository": True,
        "schema_version": catalog["schema_version"],
        "catalog_version": catalog["catalog_version"],
        "generated_at": generated_at,
        "catalog_sha256": claimed,
        "raw_sha256": _sha256_bytes(raw),
        "endpoint_schema_sha256": dict(sorted(endpoint_schema_sha256.items())),
        "capability_status": {
            capability: capability_status[capability]
            for capability in sorted(_EXPECTED_GRID_CAPABILITIES)
        },
        "file_listing": {
            "status": "confirmed",
            "download_attempted": False,
            "signed_urls_retained": False,
            "payload_completeness_authority": False,
        },
        "use_boundary": "PROVENANCE_METADATA_ONLY_NO_QUERY_NO_DOWNLOAD",
    }


def _accepted_source_inventory(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    development_raw = _read_pinned_repo_bytes(root, "private_lpl_snapshot_manifest")
    development = _decode_object(development_raw, "private LPL snapshot manifest")
    development_canonical = _validate_embedded_digest(
        development, "manifest_sha256", "private LPL snapshot manifest"
    )
    if development.get("schema_version") != "scryglass:real-v1-lpl-private-g2-input:v1":
        raise G1SourceAuthorityError("private LPL snapshot schema changed")
    if development.get("source_scope") != "LPL_2025_2026_PRE_2026_06_01":
        raise G1SourceAuthorityError("private LPL snapshot source scope changed")
    if development.get("final_holdout") != {
        "accessed": False,
        "cutoff_local_naive": "2026-06-01T00:00:00",
        "status": "SEALED_UNREAD",
    }:
        raise G1SourceAuthorityError("private LPL final-holdout boundary changed")
    expected_blocked = {
        "pre_event_roster_authority",
        "historical_ingest_availability",
        "forecast",
        "prediction",
        "production",
        "publication",
        "promotion",
        "sota",
        "final_holdout_result",
    }
    claim_scope = development.get("claim_scope")
    if not isinstance(claim_scope, Mapping) or set(claim_scope.get("blocked_claims", ())) != expected_blocked:
        raise G1SourceAuthorityError("private LPL snapshot claim blockers changed")
    if claim_scope.get("available_claims") != ["private_model_fit", "private_rank_selection"]:
        raise G1SourceAuthorityError("private LPL snapshot authority changed")

    development_rows_raw = _read_pinned_repo_bytes(root, "private_lpl_snapshot_rows")
    if development.get("rows_sha256") != _sha256_bytes(development_rows_raw):
        raise G1SourceAuthorityError("private LPL snapshot rows no longer match its manifest")

    draft_raw = _read_pinned_repo_bytes(root, "completed_draft_manifest")
    draft = _decode_object(draft_raw, "completed-draft manifest")
    draft_canonical = _validate_embedded_digest(
        draft, "manifest_sha256", "completed-draft manifest"
    )
    if draft.get("schema_version") != "scryglass:g1-lpl-completed-draft-features:v1":
        raise G1SourceAuthorityError("completed-draft manifest schema changed")
    if draft.get("final_holdout") != {
        "accessed": False,
        "included": False,
        "status": "SEALED_UNREAD",
    }:
        raise G1SourceAuthorityError("completed-draft final-holdout boundary changed")
    availability = draft.get("availability")
    if not isinstance(availability, Mapping) or set(availability.get("not_authorized", ())) != {
        "historical_ingest",
        "current",
        "live",
        "forecast",
        "prediction",
    }:
        raise G1SourceAuthorityError("completed-draft availability boundary changed")

    draft_rows_raw = _read_pinned_repo_bytes(root, "completed_draft_rows")
    if draft.get("rows_raw_sha256") != _sha256_bytes(draft_rows_raw):
        raise G1SourceAuthorityError("completed-draft rows no longer match their manifest")

    review_raw = _read_pinned_repo_bytes(root, "completed_draft_review")
    review = _decode_object(review_raw, "completed-draft review")
    review_canonical = _validate_embedded_digest(
        review, "review_sha256", "completed-draft review"
    )
    if review.get("disposition") != "ACCEPT_PRIVATE_COMPLETED_DRAFT_FEATURE_SLICE_V1":
        raise G1SourceAuthorityError("completed-draft review is not accepted")
    execution = review.get("execution_boundary")
    if not isinstance(execution, Mapping) or execution.get("final_holdout_accessed") is not False:
        raise G1SourceAuthorityError("completed-draft review crossed the final holdout")
    if execution.get("model_fit_executed") is not False or execution.get("rank_selection_executed") is not False:
        raise G1SourceAuthorityError("completed-draft review unexpectedly executed modeling")

    inventory = [
        {
            "kind": "PRIVATE_RETROSPECTIVE_LPL_SNAPSHOT",
            "manifest": {
                **_PINNED_REPOSITORY_ARTIFACTS["private_lpl_snapshot_manifest"],
                "canonical_sha256": development_canonical,
            },
            "rows": dict(_PINNED_REPOSITORY_ARTIFACTS["private_lpl_snapshot_rows"]),
            "map_count": development.get("canonical_selected_target_row_count"),
            "source_series_family_count": development.get("coverage", {}).get(
                "source_series_family_count"
            ),
            "existing_authority": ["private_model_fit", "private_rank_selection"],
            "authority_origin": "accepted_manifest_only",
        },
        {
            "kind": "PRIVATE_COMPLETED_DRAFT_FEATURE_SLICE",
            "manifest": {
                **_PINNED_REPOSITORY_ARTIFACTS["completed_draft_manifest"],
                "canonical_sha256": draft_canonical,
            },
            "rows": dict(_PINNED_REPOSITORY_ARTIFACTS["completed_draft_rows"]),
            "review": {
                **_PINNED_REPOSITORY_ARTIFACTS["completed_draft_review"],
                "canonical_sha256": review_canonical,
                "disposition": review["disposition"],
            },
            "map_count": draft.get("coverage", {}).get("accepted_map_count"),
            "existing_authority": [
                "private_model_fit_feature_input",
                "private_rank_selection_feature_input",
            ],
            "authority_origin": "independent_accepted_review_only",
        },
    ]
    semantic_boundary = {
        "observed_map_participants": "RETROSPECTIVE_OBSERVED_LINEUPS_ONLY",
        "pre_event_roster_authority": False,
        "current_roster_authority": False,
        "source_local_series_family": "DESCRIPTIVE_GROUPING_ONLY",
        "authoritative_provider_series_crosswalk": False,
        "historical_ingest_authority": False,
    }
    return inventory, semantic_boundary


def _benchmark_boundary(root: Path) -> dict[str, Any]:
    manifest_raw = _read_pinned_repo_bytes(root, "benchmark_contract_manifest")
    manifest = _decode_object(manifest_raw, "benchmark contract manifest")
    if manifest.get("manifest_id") != "scryglass:real-benchmark-manifest:v1.4":
        raise G1SourceAuthorityError("benchmark contract manifest identity changed")
    if manifest.get("status") != "FROZEN_CANDIDATE_PENDING_INDEPENDENT_REVIEW":
        raise G1SourceAuthorityError("benchmark contract manifest status changed")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise G1SourceAuthorityError("benchmark contract manifest file inventory is missing")

    benchmark_raw = _read_pinned_repo_bytes(root, "benchmark_contract")
    benchmark_binding = files.get("benchmark-contract.json")
    if not isinstance(benchmark_binding, Mapping) or benchmark_binding.get("raw_sha256") != _sha256_bytes(benchmark_raw):
        raise G1SourceAuthorityError("benchmark contract is not bound by its manifest")

    registry_raw = _read_pinned_repo_bytes(root, "benchmark_candidate_registry")
    registry_binding = files.get("candidate-registry.json")
    if not isinstance(registry_binding, Mapping) or registry_binding.get("raw_sha256") != _sha256_bytes(registry_raw):
        raise G1SourceAuthorityError("candidate registry is not bound by the benchmark manifest")
    registry = _decode_object(registry_raw, "benchmark candidate registry")
    handoff = registry.get("g1_unified_authority_handoff")
    if not isinstance(handoff, Mapping):
        raise G1SourceAuthorityError("benchmark candidate registry lacks the G1 handoff")
    if handoff.get("status") != "NOT_IMPLEMENTED_IN_V1_4":
        raise G1SourceAuthorityError("benchmark G1 handoff status unexpectedly changed")
    if handoff.get("transition") != (
        "PROHIBITED_IN_V1_4_REQUIRES_G1_UNIFIED_AUTHORITY_BUNDLE"
    ):
        raise G1SourceAuthorityError("benchmark source-bound transition unexpectedly changed")
    return {
        "contract_manifest": dict(
            _PINNED_REPOSITORY_ARTIFACTS["benchmark_contract_manifest"]
        ),
        "benchmark_contract": dict(_PINNED_REPOSITORY_ARTIFACTS["benchmark_contract"]),
        "candidate_registry": dict(
            _PINNED_REPOSITORY_ARTIFACTS["benchmark_candidate_registry"]
        ),
        "status": manifest["status"],
        "g1_unified_authority_handoff_status": handoff["status"],
        "source_bound_transition_authorized": False,
        "required_boundary": handoff.get("required_boundary"),
    }


def build_g1_source_authority_receipt(
    *,
    repo_root: Path = REPO_ROOT,
    catalog_path: Path = DEFAULT_GRID_CATALOG,
) -> dict[str, Any]:
    """Build a deterministic inventory receipt without expanding authority."""

    root = repo_root.resolve()
    inventory, semantic_boundary = _accepted_source_inventory(root)
    benchmark = _benchmark_boundary(root)
    catalog = _grid_catalog_provenance(catalog_path)

    code_files = [
        {
            "locator": locator,
            "raw_sha256": _sha256_bytes(_read_repo_bytes(root, locator)),
            "authority_effect": "NONE_CONTENT_ADDRESSING_ONLY",
        }
        for locator in SOURCE_CONTRACT_ALLOWLIST
    ]
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "receipt_as_of": catalog["generated_at"],
        "disposition": RECEIPT_DISPOSITION,
        "content_addressing_confers_authority": False,
        "source_inventory": inventory,
        "semantic_boundary": semantic_boundary,
        "source_contract_tree": {
            "allowlist": list(SOURCE_CONTRACT_ALLOWLIST),
            "files": code_files,
            "canonical_source_tree_sha256": canonical_source_tree_sha256(
                root, SOURCE_CONTRACT_ALLOWLIST
            ),
            "authority_effect": "NONE_CONTRACT_IDENTITY_ONLY",
        },
        "grid_catalog_provenance": catalog,
        "benchmark_boundary": benchmark,
        "authority_effect": {
            "receipt_authorizes": ["source_inventory_preflight"],
            "receipt_reopens_existing_authority_only": [
                "private_model_fit",
                "private_rank_selection",
                "private_model_fit_feature_input",
                "private_rank_selection_feature_input",
            ],
            "receipt_does_not_authorize": [
                "current_roster",
                "pre_event_roster",
                "authoritative_provider_series_crosswalk",
                "historical_ingest",
                "grid_payload_download",
                "grid_payload_completeness",
                "benchmark_source_bound_transition",
                "forecast",
                "prediction",
                "production",
                "publication",
                "promotion",
                "sota",
                "final_holdout_result",
            ],
        },
        "final_holdout": {
            "status": "SEALED_UNREAD",
            "accessed": False,
            "included": False,
        },
        "typed_blockers": [dict(blocker) for blocker in _BLOCKERS],
        "claim_ceiling": {
            "source_inventory_preflight": True,
            "existing_private_fit_authority_expanded": False,
            "current_roster": False,
            "pre_event_roster": False,
            "authoritative_provider_series_crosswalk": False,
            "historical_ingest": False,
            "grid_row_coverage": False,
            "benchmark_source_bound_transition": False,
            "forecast": False,
            "prediction": False,
            "production": False,
            "publication": False,
            "promotion": False,
            "sota": False,
            "final_holdout": False,
        },
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    return body


def validate_g1_source_authority_receipt(
    payload: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    catalog_path: Path = DEFAULT_GRID_CATALOG,
) -> dict[str, Any]:
    """Reopen every pinned input and reject any receipt or authority mutation."""

    if not isinstance(payload, Mapping):
        raise G1SourceAuthorityError("G1 source authority receipt must be an object")
    claimed = payload.get("receipt_sha256")
    unsigned = dict(payload)
    unsigned.pop("receipt_sha256", None)
    if not _is_sha256(claimed) or _canonical_sha256(unsigned) != claimed:
        raise G1SourceAuthorityError("G1 source authority receipt digest is invalid")
    expected = build_g1_source_authority_receipt(
        repo_root=repo_root,
        catalog_path=catalog_path,
    )
    if dict(payload) != expected:
        raise G1SourceAuthorityError(
            "G1 source authority receipt differs from reopened pinned evidence"
        )
    return expected


def canonical_receipt_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return canonical newline-terminated receipt bytes after digest validation."""

    claimed = payload.get("receipt_sha256")
    unsigned = dict(payload)
    unsigned.pop("receipt_sha256", None)
    if not _is_sha256(claimed) or _canonical_sha256(unsigned) != claimed:
        raise G1SourceAuthorityError("G1 source authority receipt digest is invalid")
    return _canonical_bytes(dict(payload)) + b"\n"


__all__ = [
    "DEFAULT_GRID_CATALOG",
    "G1SourceAuthorityError",
    "RECEIPT_DISPOSITION",
    "REPO_ROOT",
    "SCHEMA_VERSION",
    "SOURCE_CONTRACT_ALLOWLIST",
    "build_g1_source_authority_receipt",
    "canonical_receipt_bytes",
    "validate_g1_source_authority_receipt",
]
