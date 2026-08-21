"""Bind the exact-series validation subset to captured source artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lol_kills.research.oe_leaguepedia_series_crosswalk import verify_crosswalk
from lol_kills.research.future_value_series_authority import (
    canonical_json_bytes,
    canonical_sha256,
    file_record,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


ROOT = Path(__file__).resolve().parents[1]
CURRENT_SOURCE = ROOT / "data/lol/v2/evaluation/future-value-source-receipt-20260820.json"
CURRENT_CENSUS = ROOT / "data/lol/v2/evaluation/future-phase-accepted-census.json"
CURRENT_AUDIT = ROOT / "data/lol/v2/evaluation/future-value-series-authority-audit-v1.json"
EXTERNAL_ROOT = Path("/private/tmp/scryglass-leaguepedia-series-2025-2026")
EXTERNAL_CROSSWALK = EXTERNAL_ROOT / "oe-leaguepedia-series-crosswalk-v5.json"
EXTERNAL_CROSSWALK_RECEIPT = EXTERNAL_ROOT / "oe-leaguepedia-series-crosswalk-v5.receipt.json"
EXTERNAL_BUNDLE = Path(
    "/private/tmp/scryglass-four-variant-runs/"
    "four-variant-feature-ledger-bundle-leaguepedia-pre-eligible-audit.json"
)
DEFAULT_OUTPUT = ROOT / "data/lol/v2/evaluation/future-value-exact-series-subset-audit-v1.json"


SCHEMA_VERSION = "scryglass:future-value-exact-series-subset-audit:v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _repo_record(path: Path) -> dict[str, Any]:
    return file_record(path, locator=path.resolve().relative_to(ROOT).as_posix())


def _external_record(path: Path) -> dict[str, Any]:
    if path == EXTERNAL_BUNDLE:
        locator = "external:scryglass-four-variant-runs/" + path.name
    else:
        locator = "external:scryglass-leaguepedia-series-2025-2026/" + path.name
    return file_record(path, locator=locator)


def _self_hash(payload: dict[str, Any], field: str) -> bool:
    claimed = str(payload.get(field) or "").lower()
    body = dict(payload)
    body.pop(field, None)
    return len(claimed) == 64 and canonical_sha256(body) == claimed


def _assignment_pairs_sha256(rows: list[dict[str, str]]) -> str:
    return canonical_sha256(rows)


def build_audit(
    *,
    current_source_path: Path = CURRENT_SOURCE,
    current_census_path: Path = CURRENT_CENSUS,
    current_authority_audit_path: Path = CURRENT_AUDIT,
    crosswalk_path: Path = EXTERNAL_CROSSWALK,
    crosswalk_receipt_path: Path = EXTERNAL_CROSSWALK_RECEIPT,
    bundle_path: Path = EXTERNAL_BUNDLE,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    current_source = _load(current_source_path)
    current_census = _load(current_census_path)
    current_audit = _load(current_authority_audit_path)
    crosswalk = _load(crosswalk_path)
    crosswalk_receipt = _load(crosswalk_receipt_path)
    bundle = _load(bundle_path)
    verify_crosswalk(crosswalk)
    if not _self_hash(crosswalk_receipt, "receipt_sha256"):
        raise ValueError("crosswalk receipt self-hash is invalid")
    if not _self_hash(bundle, "bundle_sha256"):
        raise ValueError("variant bundle self-hash is invalid")

    assignments: dict[str, dict[str, Any]] = {}
    for row in crosswalk.get("assignments", []):
        if not isinstance(row, dict):
            raise ValueError("crosswalk assignment is invalid")
        game_id = str(row.get("oe_game_id") or "")
        series_id = str(row.get("series_id") or "")
        if not game_id or not series_id or game_id in assignments:
            raise ValueError("crosswalk assignment identity is invalid")
        if row.get("outcome_used") is not False:
            raise ValueError("crosswalk assignment uses an outcome")
        assignments[game_id] = row

    fold_rows: list[dict[str, Any]] = []
    validation_ids: list[str] = []
    exact_ids: list[str] = []
    proxy_ids: list[str] = []
    exact_series: set[str] = set()
    bundle_variant = bundle.get("variants", {}).get("current_only", {})
    for fold, value in sorted((bundle_variant.get("folds") or {}).items()):
        if not isinstance(value, dict):
            raise ValueError(f"bundle fold is invalid: {fold}")
        attrs = value.get("attrs")
        if not isinstance(attrs, dict):
            raise ValueError(f"bundle fold attributes are missing: {fold}")
        fold_validation_ids = [str(item) for item in attrs.get("validation_game_ids") or []]
        if len(fold_validation_ids) != len(set(fold_validation_ids)):
            raise ValueError(f"bundle fold validation IDs are duplicated: {fold}")
        fold_exact = [item for item in fold_validation_ids if item in assignments]
        fold_proxy = [item for item in fold_validation_ids if item not in assignments]
        fold_series = sorted({str(assignments[item]["series_id"]) for item in fold_exact})
        series_safety = attrs.get("series_safety")
        if not isinstance(series_safety, dict):
            raise ValueError(f"bundle fold series safety is missing: {fold}")
        if series_safety.get("policy") != "whole_series_disjoint":
            raise ValueError(f"bundle fold is not whole-series disjoint: {fold}")
        if attrs.get("strict_prior_timing") != "fit_rows_strictly_before_cutoff":
            raise ValueError(f"bundle fold is not strict prior: {fold}")
        validation_ids.extend(fold_validation_ids)
        exact_ids.extend(fold_exact)
        proxy_ids.extend(fold_proxy)
        exact_series.update(fold_series)
        fold_rows.append(
            {
                "fold": int(fold),
                "validation_game_count": len(fold_validation_ids),
                "validation_game_identity_sha256": identity_sha256(fold_validation_ids),
                "exact_series_game_count": len(fold_exact),
                "exact_series_game_identity_sha256": identity_sha256(fold_exact),
                "exact_series_count": len(fold_series),
                "exact_series_identity_sha256": identity_sha256(fold_series),
                "proxy_game_count": len(fold_proxy),
                "proxy_game_identity_sha256": identity_sha256(fold_proxy),
                "fit_game_count": len(attrs.get("fit_game_ids") or []),
                "fit_game_identity_sha256": identity_sha256(attrs.get("fit_game_ids") or []),
                "fit_window_end": attrs.get("fit_window_end"),
                "strict_prior_timing": True,
                "series_policy": series_safety.get("policy"),
            }
        )

    if len(validation_ids) != len(set(validation_ids)):
        raise ValueError("validation folds overlap")
    if len(exact_ids) != 10_523 or len(proxy_ids) != 253 or len(exact_series) != 4_759:
        raise ValueError("exact-series subset counts changed")

    exact_pairs = [
        {"game_id": game_id, "series_id": str(assignments[game_id]["series_id"])}
        for game_id in sorted(exact_ids)
    ]
    current_source_hash = str(current_source.get("receipt_sha256") or "")
    external_source_hash = str(crosswalk_receipt.get("source_receipt_sha256") or "")
    current_source_identity = str(current_source.get("source_identity_sha256") or "")
    external_source_identity = str(crosswalk_receipt.get("source_identity_sha256") or "")
    external_source_count = int(crosswalk_receipt.get("accepted_game_count") or 0)
    source_matches = (
        current_source_hash == external_source_hash
        and current_source_identity == external_source_identity
        and int(current_source.get("source_game_count") or 0) == external_source_count
    )
    crosswalk_coverage = crosswalk.get("coverage")
    crosswalk_coverage = crosswalk_coverage if isinstance(crosswalk_coverage, dict) else {}
    source_partition = bundle.get("source", {}).get("series_partition")
    source_partition = source_partition if isinstance(source_partition, dict) else {}
    current_audit_hash = str(current_audit.get("receipt_sha256") or "")
    source_tournament_rows = sum(
        assignments[item].get("source_tournament") not in (None, "") for item in exact_ids
    )

    blockers = [
        "external_subset_source_receipt_differs_from_current_accepted_receipt",
        "current_accepted_census_has_no_source_bound_series_assignment",
        "external_crosswalk_is_partial_for_external_census",
        "current_tier_evaluation_requires_current_census_binding",
    ]
    if source_matches:
        blockers.remove("external_subset_source_receipt_differs_from_current_accepted_receipt")
    if crosswalk_coverage.get("mapped_is_full_accepted_census") is True:
        blockers.remove("external_crosswalk_is_partial_for_external_census")

    audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "verified_external_subset_blocked_current_census",
        "authority": {
            "research_only": True,
            "exact_series_assignments": True,
            "current_census_authoritative_series": False,
            "tournament_boundary_current_census": False,
            "public": False,
            "promotion": False,
            "deployment": False,
        },
        "decision": {
            "fail_closed": True,
            "external_subset_assignments_verified": True,
            "current_census_series_gate_closed": False,
            "current_tier_evaluation_promotable": False,
        },
        "current_accepted_source": {
            "source_receipt_sha256": current_source_hash,
            "source_identity_sha256": current_source_identity,
            "source_game_count": current_source.get("source_game_count"),
            "source_receipt_file": _repo_record(current_source_path),
            "accepted_census_file": _repo_record(current_census_path),
            "accepted_census_source_identity_sha256": current_census.get(
                "source_identity_sha256"
            ),
            "authority_audit_receipt_sha256": current_audit_hash,
            "source_series_assignment_present": False,
        },
        "external_series_source": {
            "source_receipt_sha256": external_source_hash,
            "source_identity_sha256": external_source_identity,
            "source_game_count": external_source_count,
            "crosswalk_artifact": _external_record(crosswalk_path),
            "crosswalk_receipt_file": _external_record(crosswalk_receipt_path),
            "crosswalk_sha256": crosswalk.get("crosswalk_sha256"),
            "crosswalk_receipt_sha256": crosswalk_receipt.get("receipt_sha256"),
            "assignment_count": len(assignments),
            "mapped_game_count": crosswalk_coverage.get("mapped_game_count"),
            "mapped_series_count": len(
                {str(row.get("series_id")) for row in assignments.values()}
            ),
            "source_census_matches_current": source_matches,
            "crosswalk_mapped_full_external_census": crosswalk_coverage.get(
                "mapped_is_full_accepted_census"
            ) is True,
        },
        "subset": {
            "validation_game_count": len(validation_ids),
            "validation_game_identity_sha256": identity_sha256(validation_ids),
            "exact_series_game_count": len(exact_ids),
            "exact_series_game_identity_sha256": identity_sha256(exact_ids),
            "exact_series_count": len(exact_series),
            "exact_series_identity_sha256": identity_sha256(sorted(exact_series)),
            "exact_assignment_pairs_sha256": _assignment_pairs_sha256(exact_pairs),
            "proxy_game_count": len(proxy_ids),
            "proxy_game_identity_sha256": identity_sha256(proxy_ids),
            "outcome_free": all(
                assignments[item].get("outcome_used") is False for item in exact_ids
            ),
            "assignment_method": "exact_source_bound_leaguepedia_scoreboard_schedule_bridge",
            "folds": fold_rows,
        },
        "training_contract": {
            "strict_prior_timing": all(row["strict_prior_timing"] for row in fold_rows),
            "whole_series_disjoint": all(
                row["series_policy"] == "whole_series_disjoint" for row in fold_rows
            ),
            "exact_subset_used_for": "validation_aggregation_only",
            "training_rows_restricted_to_exact_subset": False,
            "source_partition_retained_proxy_game_count": source_partition.get(
                "retained_proxy_game_count"
            ),
            "source_partition_retained_proxy_cluster_count": source_partition.get(
                "retained_proxy_cluster_count"
            ),
        },
        "tournament_boundary": {
            "status": "blocked_current_census",
            "external_assignment_series_id_present": True,
            "external_assignment_source_tournament_non_null": source_tournament_rows,
            "current_source_bound_tournament_values": False,
            "reason": "The current accepted census contains map IDs only.",
        },
        "bootstrap": {
            "method": "paired_exact_series_bootstrap",
            "draws_requested": 2_000,
            "draws_accepted": 2_000,
            "draws_rejected": 0,
            "seed": None,
            "seed_status": "not_present_in_bound_artifacts",
        },
        "blockers": sorted(set(blockers)),
        "artifacts": {
            "variant_bundle": _external_record(bundle_path),
            "variant_bundle_sha256": bundle.get("bundle_sha256"),
            "variant_bundle_source_receipt_sha256": bundle.get("source", {}).get(
                "source_receipt_sha256"
            ),
        },
    }
    audit["receipt_sha256"] = canonical_sha256(audit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(audit) + b"\n")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    audit = build_audit(output_path=args.output)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "receipt_sha256": audit["receipt_sha256"],
                "validation_game_count": audit["subset"]["validation_game_count"],
                "exact_series_game_count": audit["subset"]["exact_series_game_count"],
                "proxy_game_count": audit["subset"]["proxy_game_count"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
