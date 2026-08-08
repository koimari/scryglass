"""Outcome-free 2026 support-sufficiency gate for the rank assay.

This module consumes only aggregate coverage diagnostics from the pinned
private result.  It does not load targets, nuisance probabilities, fit rows,
predictions, or the final temporal holdout.
"""

from __future__ import annotations

import hashlib
import json
import numbers
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_ID = "scryglass.representation-rank-2026-support-gate.v1"
SOURCE_SCHEMA_ID = "scryglass.representation-rank-private-result.v1"
SOURCE_LOCATOR = (
    "data/lol/warehouse/private_v2/draft-interactions/"
    "representation-rank-private-result.json"
)
SOURCE_RAW_SHA256 = (
    "be59be06c9cdbbeb2d1065f6ddd6c2a6fd84da716e7be73ab12ef025f60a847a"
)
SOURCE_ARTIFACT_SHA256 = (
    "3f909cb0f283b1804d424155545684a74c207c13b4d74359126c7338d69c6c78"
)
SOURCE_PROJECTION_SHA256 = (
    "b11c689a743c3d617ecb2773e96ae59bd53444e9cfdec973a7072c302c6a5fc0"
)
OUTPUT_LOCATOR = (
    "data/lol/v2/models/draft-interactions/"
    "representation-rank-2026-support-gate.json"
)

DECISION_BLOCKS = (
    ("development", "2026-01"),
    ("development", "2026-02"),
    ("development", "2026-03"),
    ("validation", "2026-04"),
    ("validation", "2026-05"),
)
DIAGNOSTIC_ONLY_BLOCKS = (
    ("train", "2025-04"),
    ("train", "2025-05"),
    ("train", "2025-06"),
    ("train", "2025-07"),
    ("train", "2025-08"),
    ("train", "2025-09"),
    ("development", "2025-10"),
)
EXPECTED_SOURCE_BLOCKS = (*DIAGNOSTIC_ONLY_BLOCKS, *DECISION_BLOCKS)
COUNT_FIELDS = ("maps", "eligible_maps", "clusters", "eligible_clusters")

OVERALL_MAP_NUMERATOR = 4
OVERALL_MAP_DENOMINATOR = 5
OVERALL_CLUSTER_NUMERATOR = 4
OVERALL_CLUSTER_DENOMINATOR = 5
MONTH_MAP_NUMERATOR = 2
MONTH_MAP_DENOMINATOR = 3
MONTH_MINIMUM_ELIGIBLE_CLUSTERS = 15
LEAGUE_MINIMUM_MAPS = 30
LEAGUE_MINIMUM_CLUSTERS = 10
LEAGUE_MAP_NUMERATOR = 3
LEAGUE_MAP_DENOMINATOR = 4


class SupportGateError(ValueError):
    """Raised when source identity or aggregate coverage is not exact."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical JSON byte representation used for hashing."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def with_artifact_sha256(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(unsigned)
    payload["artifact_sha256"] = canonical_sha256(unsigned)
    return payload


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _count(value: Any, *, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or int(value) < 0
    ):
        raise SupportGateError(f"{label} must be a nonnegative integer")
    return int(value)


def _counts(row: Mapping[str, Any], *, label: str) -> dict[str, int]:
    values = {
        field: _count(row.get(field), label=f"{label}.{field}")
        for field in COUNT_FIELDS
    }
    if (
        values["eligible_maps"] > values["maps"]
        or values["eligible_clusters"] > values["clusters"]
    ):
        raise SupportGateError(f"{label} eligible count exceeds total")
    return values


def _fraction_passed(
    *,
    eligible: int,
    total: int,
    numerator: int,
    denominator: int,
) -> bool:
    """Compare eligible / total >= numerator / denominator exactly."""
    return eligible * denominator >= total * numerator


def _validate_block(
    row: Mapping[str, Any],
    *,
    expected_split: str,
    expected_month: str,
) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise SupportGateError("coverage block must be an object")
    expected_keys = {
        "split",
        "calendar_month",
        "passed",
        *COUNT_FIELDS,
        "month",
        "leagues",
        "membership_sha256",
    }
    if set(row) != expected_keys:
        raise SupportGateError("coverage block source schema changed")
    if (
        row.get("split") != expected_split
        or row.get("calendar_month") != expected_month
    ):
        raise SupportGateError("coverage block identity or order changed")
    if not _is_sha256(row.get("membership_sha256")):
        raise SupportGateError("coverage membership identity invalid")

    overall = _counts(row, label=f"{expected_split}/{expected_month}")
    month = row.get("month")
    if (
        not isinstance(month, Mapping)
        or set(month) != {"calendar_month", *COUNT_FIELDS}
        or month.get("calendar_month") != expected_month
        or _counts(month, label=f"{expected_split}/{expected_month}.month")
        != overall
    ):
        raise SupportGateError("month aggregate differs from block aggregate")

    leagues = row.get("leagues")
    if not isinstance(leagues, list) or not leagues:
        raise SupportGateError("coverage leagues must be a nonempty array")
    league_rows: list[dict[str, Any]] = []
    observed_names: list[str] = []
    league_sums = {field: 0 for field in COUNT_FIELDS}
    for index, league_row in enumerate(leagues):
        if not isinstance(league_row, Mapping):
            raise SupportGateError("coverage league must be an object")
        if set(league_row) != {"league", *COUNT_FIELDS}:
            raise SupportGateError("coverage league source schema changed")
        league = league_row.get("league")
        if not isinstance(league, str) or not league:
            raise SupportGateError("coverage league identity invalid")
        counts = _counts(
            league_row,
            label=f"{expected_split}/{expected_month}.leagues[{index}]",
        )
        observed_names.append(league)
        for field in COUNT_FIELDS:
            league_sums[field] += counts[field]
        league_rows.append({"league": league, **counts})
    if observed_names != sorted(set(observed_names)):
        raise SupportGateError("coverage league order or uniqueness changed")
    if league_sums != overall:
        raise SupportGateError("league aggregates do not sum to block counts")

    return {
        "split": expected_split,
        "calendar_month": expected_month,
        **overall,
        "month": {"calendar_month": expected_month, **overall},
        "leagues": league_rows,
        "membership_sha256": row["membership_sha256"],
    }


def _authorized_source_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build the exact outcome-free projection whose digest is pinned."""
    if not isinstance(payload, Mapping):
        raise SupportGateError("source payload must be an object")
    projection = {
        "schema_id": payload.get("schema_id"),
        "artifact_sha256": payload.get("artifact_sha256"),
        "aggregate_only": payload.get("aggregate_only"),
        "final_target_loaded": payload.get("final_target_loaded"),
        "coverage_diagnostics": payload.get("coverage_diagnostics"),
    }
    if (
        projection["schema_id"] != SOURCE_SCHEMA_ID
        or projection["artifact_sha256"] != SOURCE_ARTIFACT_SHA256
        or projection["aggregate_only"] is not True
        or projection["final_target_loaded"] is not False
    ):
        raise SupportGateError("pinned aggregate source identity changed")
    rows = projection["coverage_diagnostics"]
    if not isinstance(rows, list):
        raise SupportGateError("coverage diagnostics must be an array")
    observed = tuple(
        (row.get("split"), row.get("calendar_month"))
        if isinstance(row, Mapping)
        else (None, None)
        for row in rows
    )
    if any(split == "final" for split, _ in observed):
        raise SupportGateError("final split is prohibited")
    if observed != EXPECTED_SOURCE_BLOCKS:
        raise SupportGateError(
            "source must contain the exact ordered nonfinal frozen blocks"
        )
    validated = [
        _validate_block(row, expected_split=split, expected_month=month)
        for row, (split, month) in zip(rows, EXPECTED_SOURCE_BLOCKS)
    ]
    return {
        **{key: projection[key] for key in projection if key != "coverage_diagnostics"},
        "coverage_diagnostics": validated,
    }


def project_pinned_source(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate and return only the authorized outcome-free projection."""
    projection = _authorized_source_projection(payload)
    if canonical_sha256(projection) != SOURCE_PROJECTION_SHA256:
        raise SupportGateError("pinned authorized source projection changed")
    return projection


def load_pinned_source_projection(path: Path | str) -> dict[str, Any]:
    """Verify raw bytes, then return the authorized aggregate projection."""
    source_path = Path(path)
    raw = source_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_RAW_SHA256:
        raise SupportGateError("pinned source raw-file SHA-256 changed")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SupportGateError("pinned source is not valid JSON") from exc
    return project_pinned_source(payload)


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        field: sum(_count(row.get(field), label=field) for row in rows)
        for field in COUNT_FIELDS
    }


def _overall_result(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = _aggregate(rows)
    map_passed = _fraction_passed(
        eligible=counts["eligible_maps"],
        total=counts["maps"],
        numerator=OVERALL_MAP_NUMERATOR,
        denominator=OVERALL_MAP_DENOMINATOR,
    )
    cluster_passed = _fraction_passed(
        eligible=counts["eligible_clusters"],
        total=counts["clusters"],
        numerator=OVERALL_CLUSTER_NUMERATOR,
        denominator=OVERALL_CLUSTER_DENOMINATOR,
    )
    return {
        **counts,
        "component_pass": {
            "eligible_maps_at_least_four_fifths": map_passed,
            "eligible_clusters_at_least_four_fifths": cluster_passed,
        },
        "passed": map_passed and cluster_passed,
    }


def _month_result(row: Mapping[str, Any]) -> dict[str, Any]:
    counts = {field: int(row[field]) for field in COUNT_FIELDS}
    map_passed = _fraction_passed(
        eligible=counts["eligible_maps"],
        total=counts["maps"],
        numerator=MONTH_MAP_NUMERATOR,
        denominator=MONTH_MAP_DENOMINATOR,
    )
    cluster_passed = (
        counts["eligible_clusters"] >= MONTH_MINIMUM_ELIGIBLE_CLUSTERS
    )
    return {
        "split": row["split"],
        "calendar_month": row["calendar_month"],
        **counts,
        "membership_sha256": row["membership_sha256"],
        "component_pass": {
            "eligible_maps_at_least_two_thirds": map_passed,
            "eligible_clusters_at_least_15": cluster_passed,
        },
        "passed": map_passed and cluster_passed,
    }


def _league_results(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pooled: dict[str, dict[str, int]] = {}
    for row in rows:
        for league_row in row["leagues"]:
            league = league_row["league"]
            totals = pooled.setdefault(
                league, {field: 0 for field in COUNT_FIELDS}
            )
            for field in COUNT_FIELDS:
                totals[field] += int(league_row[field])

    output: list[dict[str, Any]] = []
    for league in sorted(pooled):
        counts = pooled[league]
        maps_minimum = counts["maps"] >= LEAGUE_MINIMUM_MAPS
        clusters_minimum = counts["clusters"] >= LEAGUE_MINIMUM_CLUSTERS
        required = maps_minimum and clusters_minimum
        fraction_passed = _fraction_passed(
            eligible=counts["eligible_maps"],
            total=counts["maps"],
            numerator=LEAGUE_MAP_NUMERATOR,
            denominator=LEAGUE_MAP_DENOMINATOR,
        )
        output.append(
            {
                "league": league,
                **counts,
                "required": required,
                "component_pass": {
                    "pooled_maps_at_least_30": maps_minimum,
                    "pooled_clusters_at_least_10": clusters_minimum,
                    "eligible_maps_at_least_three_fourths": (
                        fraction_passed if required else None
                    ),
                },
                "passed": fraction_passed if required else True,
            }
        )
    return output


def _build_from_projection(source: Mapping[str, Any]) -> dict[str, Any]:
    decision_rows = source["coverage_diagnostics"][len(DIAGNOSTIC_ONLY_BLOCKS):]
    overall = _overall_result(decision_rows)
    months = [_month_result(row) for row in decision_rows]
    leagues = _league_results(decision_rows)
    passed = (
        overall["passed"]
        and all(row["passed"] for row in months)
        and all(row["passed"] for row in leagues)
    )
    unsigned = {
        "schema_id": SCHEMA_ID,
        "development_only": True,
        "outcome_free": True,
        "aggregate_only": True,
        "source_identity": {
            "locator": SOURCE_LOCATOR,
            "raw_sha256": SOURCE_RAW_SHA256,
            "authorized_projection_sha256": SOURCE_PROJECTION_SHA256,
            "schema_id": source["schema_id"],
            "artifact_sha256": source["artifact_sha256"],
            "aggregate_only": source["aggregate_only"],
            "final_target_loaded": source["final_target_loaded"],
        },
        "policy": {
            "scope": (
                "frozen January-May 2026 nonfinal window only: development "
                "2026-01/02/03 and validation 2026-04/05"
            ),
            "authorized_source_projection": {
                "sha256": SOURCE_PROJECTION_SHA256,
                "top_level_fields": [
                    "schema_id",
                    "artifact_sha256",
                    "aggregate_only",
                    "final_target_loaded",
                    "coverage_diagnostics",
                ],
                "coverage_fields": [
                    "split",
                    "calendar_month",
                    "maps",
                    "eligible_maps",
                    "clusters",
                    "eligible_clusters",
                    "month",
                    "leagues",
                    "membership_sha256",
                ],
                "month_fields": [
                    "calendar_month",
                    "maps",
                    "eligible_maps",
                    "clusters",
                    "eligible_clusters",
                ],
                "league_fields": [
                    "league",
                    "maps",
                    "eligible_maps",
                    "clusters",
                    "eligible_clusters",
                ],
                "stored_pass_field": "validated_source_schema_but_not_projected",
                "outcome_fit_prediction_fields": "not_projected_not_hashed",
            },
            "decision_blocks": [
                {"split": split, "calendar_month": month}
                for split, month in DECISION_BLOCKS
            ],
            "diagnostic_only_2025_blocks": [
                {
                    "split": split,
                    "calendar_month": month,
                    "decision_role": "diagnostic_only_nonblocking_excluded",
                }
                for split, month in DIAGNOSTIC_ONLY_BLOCKS
            ],
            "final_split": "prohibited_not_loaded",
            "overall": {
                "eligible_maps": "at_least_4_of_5",
                "eligible_clusters": "at_least_4_of_5",
            },
            "month": {
                "eligible_maps": "at_least_2_of_3",
                "eligible_clusters": "at_least_15",
            },
            "league": {
                "pooling": "sum_each_league_across_all_five_decision_blocks",
                "required_if_pooled_maps_at_least": LEAGUE_MINIMUM_MAPS,
                "required_if_pooled_clusters_at_least": (
                    LEAGUE_MINIMUM_CLUSTERS
                ),
                "eligible_maps": "at_least_3_of_4_when_required",
            },
            "cluster_aggregation": (
                "sum_of_block_local_counts_not_unique_year_deduplication"
            ),
            "decision_arithmetic": (
                "integer_cross_products_and_integer_thresholds_no_rounding"
            ),
            "stored_source_pass_booleans": "ignored_not_trusted",
        },
        "overall": overall,
        "months": months,
        "leagues": leagues,
        "terminal_status": "PASS" if passed else "FAIL",
        "claim_ceiling": {
            "statement": (
                "support sufficiency only; no rank, model-fit, prediction, "
                "publication, production, Reliability, or SOTA authority"
            ),
            "rank_authority": False,
            "model_fit_authority": False,
            "prediction_authority": False,
            "publication_authority": False,
            "production_authority": False,
            "reliability_authority": False,
            "sota_claim_authority": False,
        },
    }
    return with_artifact_sha256(unsigned)


def build_support_gate(source_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build and self-hash the deterministic support-sufficiency artifact."""
    return _build_from_projection(project_pinned_source(source_payload))


def validate_support_gate(
    artifact: Mapping[str, Any],
    source_payload: Mapping[str, Any],
) -> None:
    """Recompute the entire artifact and require byte-semantic equality."""
    if not isinstance(artifact, Mapping):
        raise SupportGateError("support gate artifact must be an object")
    claimed = artifact.get("artifact_sha256")
    if not _is_sha256(claimed):
        raise SupportGateError("support gate artifact SHA-256 invalid")
    unsigned = dict(artifact)
    unsigned.pop("artifact_sha256", None)
    if canonical_sha256(unsigned) != claimed:
        raise SupportGateError("support gate artifact hash changed")
    expected = _build_from_projection(project_pinned_source(source_payload))
    if dict(artifact) != expected:
        raise SupportGateError("support gate differs from deterministic replay")


def write_support_gate(
    *,
    source_path: Path | str = SOURCE_LOCATOR,
    output_path: Path | str = OUTPUT_LOCATOR,
) -> dict[str, Any]:
    source = load_pinned_source_projection(source_path)
    artifact = _build_from_projection(source)
    if artifact != _build_from_projection(source):
        raise SupportGateError("support gate replay is nondeterministic")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(canonical_json_bytes(artifact) + b"\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return artifact


def main() -> None:
    write_support_gate()


if __name__ == "__main__":
    main()
