"""Authenticated, non-authorizing G1-018 baseline status materialization.

The frozen real-v1 benchmark package is intentionally source-independent and
prohibits executable source-bound transitions.  This module does not weaken
that boundary.  It authenticates the accepted non-final LPL development
snapshot and materializes the only honest current result: every required
baseline remains typed-unavailable and no score, probability, execution
receipt, or promotion evidence exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from lol_kills.v2.evaluation import benchmark_contract as bc


ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT_MANIFEST_LOCATOR = (
    "data/lol/v2/snapshots/real-v1/lpl-private-development-manifest.json"
)
SNAPSHOT_ROWS_LOCATOR = (
    "data/lol/v2/snapshots/real-v1/lpl-private-development-rows.jsonl"
)
BASELINE_REGISTRY_LOCATOR = (
    "data/lol/v2/evaluation/real-v1/baseline-registry.json"
)

_ACCEPTED_MANIFEST_SHA256 = (
    "3af87fffb2b32fd95aeb920409abe0254fa158b3dc7f079650b3472731d4ff72"
)
_ACCEPTED_ROWS_SHA256 = (
    "4ed79abb0b2471a666ab5643b91edf33c2fdde19e361c456aa589d2e9a4df846"
)
_ACCEPTED_TARGET_ROWS_SHA256 = (
    "4c332fa4e6cb155341bcffd83bd0ee1be2e04f3b5950b8a7745931253dd8bd2d"
)
_ACCEPTED_SPLIT_PAYLOAD_SHA256 = (
    "469c8d2c568a6a4480db277bf41f7eacf72964e33997f0a4e1f53f60285cd3e4"
)
_EXPECTED_PARTITION_COUNTS = {
    "DEVELOPMENT": 214,
    "TRAIN": 805,
    "VALIDATION": 207,
}
_EXPECTED_OUTPUT_COUNTS = {
    "partial_draft_score": 18,
    "player_rating": 7,
    "team_rating": 7,
    "terminal_draft_score": 29,
}
_CLAIM_CEILING = {
    "baseline_execution": False,
    "baseline_score": False,
    "final_holdout": False,
    "prediction": False,
    "promotion": False,
    "publication": False,
    "sota": False,
}


def _read_repo_file(root: Path, locator: str, *, purpose: str) -> bytes:
    return bc._read_regular_under_root(  # noqa: SLF001 - shared integrity primitive
        Path(root),
        locator,
        purpose=purpose,
    )


def _load_snapshot(root: Path) -> tuple[dict[str, Any], bytes, bytes]:
    manifest_raw = _read_repo_file(
        root,
        SNAPSHOT_MANIFEST_LOCATOR,
        purpose="G1-018 accepted snapshot manifest",
    )
    manifest = bc._parse_json_bytes(  # noqa: SLF001 - duplicate-key rejection
        manifest_raw,
        label="G1-018 accepted snapshot manifest",
    )
    rows_raw = _read_repo_file(
        root,
        SNAPSHOT_ROWS_LOCATOR,
        purpose="G1-018 accepted snapshot rows",
    )
    unsigned = dict(manifest)
    manifest_sha256 = unsigned.pop("manifest_sha256", None)
    bc._require(  # noqa: SLF001 - use the benchmark's typed failure
        manifest_sha256 == bc.stable_digest(unsigned)
        and manifest_sha256 == _ACCEPTED_MANIFEST_SHA256,
        "G1-018 source manifest is not the accepted snapshot",
        code="G1_BASELINE_SNAPSHOT_MISMATCH",
    )
    bc._require(
        manifest.get("schema_version")
        == "scryglass:real-v1-lpl-private-g2-input:v1"
        and manifest.get("source_scope") == "LPL_2025_2026_PRE_2026_06_01"
        and manifest.get("rows_locator") == SNAPSHOT_ROWS_LOCATOR,
        "G1-018 source scope or row locator changed",
        code="G1_BASELINE_SNAPSHOT_MISMATCH",
    )
    bc._require(
        manifest.get("rows_sha256") == _ACCEPTED_ROWS_SHA256
        and bc.raw_digest(rows_raw) == _ACCEPTED_ROWS_SHA256,
        "G1-018 source rows do not match the accepted snapshot",
        code="G1_BASELINE_SNAPSHOT_MISMATCH",
    )
    bc._require(
        manifest.get("canonical_selected_target_rows_sha256")
        == _ACCEPTED_TARGET_ROWS_SHA256
        and manifest.get("target_authority", {}).get("split_payload_sha256")
        == _ACCEPTED_SPLIT_PAYLOAD_SHA256,
        "G1-018 target or split authority changed",
        code="G1_BASELINE_SNAPSHOT_MISMATCH",
    )
    bc._require(
        manifest.get("final_holdout")
        == {
            "accessed": False,
            "cutoff_local_naive": "2026-06-01T00:00:00",
            "status": "SEALED_UNREAD",
        },
        "G1-018 final holdout is not sealed and unread",
        code="G1_BASELINE_SNAPSHOT_MISMATCH",
    )
    bc._require(
        manifest.get("coverage", {}).get("target_partition_counts")
        == _EXPECTED_PARTITION_COUNTS
        and manifest.get("coverage", {}).get("map_count") == 1226
        and len(rows_raw.splitlines()) == 1226,
        "G1-018 non-final partition coverage changed",
        code="G1_BASELINE_SNAPSHOT_MISMATCH",
    )
    bc._require(
        manifest.get("claim_scope", {}).get("state")
        == "PRIVATE_RETROSPECTIVE_MODEL_FIT_AND_RANK_SELECTION_AVAILABLE"
        and set(manifest.get("claim_scope", {}).get("available_claims", ()))
        == {"private_model_fit", "private_rank_selection"},
        "G1-018 private retrospective authority changed",
        code="G1_BASELINE_SNAPSHOT_MISMATCH",
    )
    bc._require(
        manifest.get("g1_018_baseline_binding")
        == {
            "claim_effect": "REQUIRED_COMPARISON_BLOCKED_UNTIL_RESOLVED",
            "reason_code": "REQUIRED_BASELINES_NOT_EXECUTED_ON_FROZEN_REAL_ROWS",
            "status": "TYPED_UNAVAILABLE",
        },
        "G1-018 typed-unavailable baseline status changed",
        code="G1_BASELINE_SNAPSHOT_MISMATCH",
    )
    return manifest, manifest_raw, rows_raw


def _load_baseline_registry(
    root: Path,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    package_root = Path(root) / bc.REAL_V1_ROOT
    bc.validate_real_v1(package_root, repo_root=Path(root))
    registry_raw = _read_repo_file(
        root,
        BASELINE_REGISTRY_LOCATOR,
        purpose="G1-018 frozen baseline registry",
    )
    registry = bc._parse_json_bytes(  # noqa: SLF001 - duplicate-key rejection
        registry_raw,
        label="G1-018 frozen baseline registry",
    )
    prebindings = {
        item.get("kind"): item
        for item in manifest.get("g0_prebinding_contracts", ())
        if isinstance(item, Mapping)
    }
    baseline_prebinding = prebindings.get("baseline_registry")
    bc._require(
        isinstance(baseline_prebinding, Mapping)
        and baseline_prebinding.get("locator") == BASELINE_REGISTRY_LOCATOR
        and baseline_prebinding.get("raw_sha256") == bc.raw_digest(registry_raw),
        "accepted G1 snapshot does not bind the frozen baseline registry",
        code="G1_BASELINE_REGISTRY_MISMATCH",
    )
    bc.validate_baseline_registry(registry, repo_root=Path(root))
    return registry, registry_raw


def _bundle_without_digest(root: Path) -> dict[str, Any]:
    manifest, manifest_raw, rows_raw = _load_snapshot(Path(root))
    registry, registry_raw = _load_baseline_registry(Path(root), manifest)
    output_counts: dict[str, int] = {}
    baseline_statuses: list[dict[str, Any]] = []
    for entry in registry["baselines"]:
        output_id = entry["comparison_output"]
        output_counts[output_id] = output_counts.get(output_id, 0) + 1
        baseline_statuses.append(
            {
                "applies_to_draft_depths": list(
                    entry["applies_to_draft_depths"]
                ),
                "comparison_output": output_id,
                "id": entry["id"],
                "status": entry["status"],
                "unavailable": dict(entry["unavailable"]),
            }
        )
    bc._require(
        output_counts == _EXPECTED_OUTPUT_COUNTS
        and len(baseline_statuses) == 61
        and all(
            item["status"] == "TYPED_UNAVAILABLE"
            for item in baseline_statuses
        ),
        "G1-018 required baseline inventory changed",
        code="G1_BASELINE_REGISTRY_MISMATCH",
    )
    return {
        "artifact_kind": "G1_018_TYPED_UNAVAILABLE_BASELINE_STATUS",
        "authority_status": "NONAUTHORIZING_STATUS_ONLY",
        "baseline_inventory": {
            "acceptance_status": registry["acceptance"]["status"],
            "baseline_count": len(baseline_statuses),
            "baselines": baseline_statuses,
            "output_counts": output_counts,
            "registry_locator": BASELINE_REGISTRY_LOCATOR,
            "registry_raw_sha256": bc.raw_digest(registry_raw),
            "registry_semantic_sha256": bc.stable_digest(registry),
            "source_dependent_execution_bindings_status": registry[
                "acceptance"
            ]["source_dependent_execution_bindings_status"],
        },
        "claim_ceiling": dict(_CLAIM_CEILING),
        "final_labels_read": False,
        "schema_version": "scryglass:g1-018-baseline-status:v1",
        "source_rows_decoded": False,
        "source_snapshot": {
            "allowed_actions": [
                "private_model_fit",
                "private_rank_selection",
            ],
            "final_holdout_status": "SEALED_UNREAD",
            "manifest_locator": SNAPSHOT_MANIFEST_LOCATOR,
            "manifest_raw_sha256": bc.raw_digest(manifest_raw),
            "manifest_sha256": manifest["manifest_sha256"],
            "map_count": manifest["coverage"]["map_count"],
            "partition_counts": dict(
                manifest["coverage"]["target_partition_counts"]
            ),
            "rows_locator": SNAPSHOT_ROWS_LOCATOR,
            "rows_raw_sha256": bc.raw_digest(rows_raw),
            "selected_target_rows_sha256": (
                manifest["canonical_selected_target_rows_sha256"]
            ),
            "source_scope": manifest["source_scope"],
            "split_payload_sha256": manifest["target_authority"][
                "split_payload_sha256"
            ],
        },
        "status": "TYPED_UNAVAILABLE",
        "unavailable": {
            "claim_effect": "REQUIRED_COMPARISON_BLOCKED_UNTIL_RESOLVED",
            "reason_code": (
                "REQUIRED_BASELINES_NOT_EXECUTED_ON_FROZEN_REAL_ROWS"
            ),
            "resolution_task": "G1-018",
        },
    }


def materialize_g1_018_baseline_status(
    root: Path = ROOT,
) -> dict[str, Any]:
    """Return the authenticated non-authorizing baseline status bundle."""

    payload = _bundle_without_digest(Path(root))
    return {**payload, "bundle_sha256": bc.stable_digest(payload)}


def validate_g1_018_baseline_status(
    bundle: Mapping[str, Any],
    root: Path = ROOT,
) -> str:
    """Re-derive and exactly validate a G1-018 baseline status bundle."""

    expected = materialize_g1_018_baseline_status(Path(root))
    bc._require(
        isinstance(bundle, Mapping) and dict(bundle) == expected,
        "G1-018 baseline status bundle is not the exact re-derived payload",
        code="G1_BASELINE_STATUS_MISMATCH",
    )
    return expected["bundle_sha256"]
