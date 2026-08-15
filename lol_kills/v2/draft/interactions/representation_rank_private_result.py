"""Strict aggregate-only terminal result for the private rank assay."""

from __future__ import annotations

import json
import math
import numbers
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import representation_rank_assay as assay


SCHEMA_ID = "scryglass.representation-rank-private-result.v1"
FIT_PLAN_COLUMNS = (
    "sequence",
    "stage",
    "split",
    "calendar_month",
    "family",
    "fit_role",
    "lambda_ally",
    "lambda_enemy",
    "width",
    "execution_status",
    "maps",
    "clusters",
    "membership_sha256",
    "objective",
    "max_gradient",
    "converged_starts",
    "stability_rms",
    "log_loss_total",
    "brier_total",
)
EXECUTION_STATUSES = frozenset({"passed", "failed", "not_run", "aliased"})
TOP_LEVEL_KEYS = frozenset(
    {
        "schema_id",
        "artifact_sha256",
        "aggregate_only",
        "development_only",
        "run_status",
        "fallback",
        "reason_code",
        "reason_context",
        "selected_model",
        "selected_width",
        "contract_artifact_sha256",
        "contract_review_core_sha256",
        "runner_review_permit_raw_sha256",
        "source_identity_sha256",
        "runtime_identity_sha256",
        "feature_domain_sha256",
        "target_domain_sha256",
        "fit_availability_domain_sha256",
        "population",
        "penalties",
        "fit_counts",
        "stage_status",
        "coverage_diagnostics",
        "development_diagnostics",
        "validation_diagnostics",
        "fit_plan",
        "final_target_loaded",
        "publication_authority",
        "production_authority",
        "reliability_authority",
        "promotion_authority",
        "sota_claim_authority",
    }
)
REASON_CODES = frozenset(
    {
        "coverage_gate_failed",
        "fit_unavailable",
        "scoring_unavailable",
        "penalty_selection_failed",
        "development_gate_failed",
        "validation_gate_failed",
    }
)
STAGE_REASON_CODES = {
    "inner": frozenset(
        {
            "coverage_gate_failed",
            "fit_unavailable",
            "scoring_unavailable",
            "penalty_selection_failed",
        }
    ),
    "development": frozenset(
        {
            "coverage_gate_failed",
            "fit_unavailable",
            "scoring_unavailable",
            "development_gate_failed",
        }
    ),
    "validation": frozenset(
        {
            "coverage_gate_failed",
            "fit_unavailable",
            "scoring_unavailable",
            "validation_gate_failed",
        }
    ),
}
STAGES = ("inner", "development", "validation")
FAMILIES = ("ally", "enemy", "joint")
ALL_MONTHS = (
    *assay.INNER_MONTHS,
    *(row[0] for row in assay.ELIGIBLE_GATE_BLOCKS["development"]),
    *(row[0] for row in assay.ELIGIBLE_GATE_BLOCKS["validation"]),
)
METRICS = ("log_loss", "brier", "calibration")
COMPARATORS = ("M0", "M8")
POPULATION_SUMMARY_FIELDS = (
    "maps",
    "clusters",
    "membership_sha256",
)
OPTIMIZATION_SUMMARY_FIELDS = (
    "objective",
    "max_gradient",
    "converged_starts",
    "stability_rms",
)
SCORING_SUMMARY_FIELDS = (
    "log_loss_total",
    "brier_total",
)
SUMMARY_FIELDS = (
    *POPULATION_SUMMARY_FIELDS,
    *OPTIMIZATION_SUMMARY_FIELDS,
    *SCORING_SUMMARY_FIELDS,
)
ALIAS_EQUAL_FIELDS = (
    "lambda_ally",
    "lambda_enemy",
    "width",
    *SUMMARY_FIELDS,
)
STAGE_SLICES = {
    "inner": (0, 36),
    "development": (36, 52),
    "validation": (52, 56),
}
FORBIDDEN_KEYS = frozenset(
    {
        "game_id",
        "game_ids",
        "y",
        "target",
        "targets",
        "p0",
        "p_blue_win_m0",
        "p_selected",
        "p_blue_win_selected",
        "prediction",
        "predictions",
        "final_material",
        "path",
        "locator",
        "parquet_locator",
        "columns",
        "rows",
    }
)


class PrivateResultError(ValueError):
    """Raised when an aggregate terminal result is not exact."""


def empty_fit_plan() -> list[dict[str, Any]]:
    """Return the complete frozen 56-slot execution ledger."""
    rows: list[dict[str, Any]] = []

    def append(
        *,
        stage: str,
        split: str,
        month: str,
        family: str,
        fit_role: str,
        lambda_ally: float | None,
        lambda_enemy: float | None,
        width: int | None,
    ) -> None:
        rows.append(
            {
                "sequence": len(rows) + 1,
                "stage": stage,
                "split": split,
                "calendar_month": month,
                "family": family,
                "fit_role": fit_role,
                "lambda_ally": lambda_ally,
                "lambda_enemy": lambda_enemy,
                "width": width,
                "execution_status": "not_run",
                "maps": None,
                "clusters": None,
                "membership_sha256": None,
                "objective": None,
                "max_gradient": None,
                "converged_starts": None,
                "stability_rms": None,
                "log_loss_total": None,
                "brier_total": None,
            }
        )

    for family in ("ally", "enemy"):
        for penalty in assay.PENALTY_GRID:
            for month in assay.INNER_MONTHS:
                append(
                    stage="inner",
                    split="train",
                    month=month,
                    family=family,
                    fit_role=f"{family}_penalty",
                    lambda_ally=float(penalty) if family == "ally" else 1.0,
                    lambda_enemy=float(penalty) if family == "enemy" else 1.0,
                    width=8,
                )
    for month, _, _ in assay.ELIGIBLE_GATE_BLOCKS["development"]:
        for width in assay.WIDTHS:
            append(
                stage="development",
                split="development",
                month=month,
                family="joint",
                fit_role="candidate_width",
                lambda_ally=None,
                lambda_enemy=None,
                width=width,
            )
    for month, _, _ in assay.ELIGIBLE_GATE_BLOCKS["validation"]:
        append(
            stage="validation",
            split="validation",
            month=month,
            family="joint",
            fit_role="locked_width",
            lambda_ally=None,
            lambda_enemy=None,
            width=None,
        )
        append(
            stage="validation",
            split="validation",
            month=month,
            family="joint",
            fit_role="M8_reference",
            lambda_ally=None,
            lambda_enemy=None,
            width=8,
        )
    if len(rows) != 56:
        raise AssertionError("frozen fit plan must contain 56 slots")
    return rows


def _walk_forbidden(value: Any) -> None:
    if isinstance(value, Mapping):
        if FORBIDDEN_KEYS & set(value):
            raise PrivateResultError("row-level or target material is prohibited")
        for nested in value.values():
            _walk_forbidden(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _walk_forbidden(nested)


def _optional_finite(value: Any, *, nonnegative: bool = False) -> bool:
    if value is None:
        return True
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    number = float(value)
    return math.isfinite(number) and (not nonnegative or number >= 0)


def _required_finite(value: Any) -> bool:
    return value is not None and _optional_finite(value)


def _exact_integral(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, numbers.Integral)


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def reason_context(
    *,
    stage: str,
    sequence: int | None = None,
    calendar_month: str | None = None,
    family: str | None = None,
    width: int | None = None,
) -> dict[str, Any]:
    value = {
        "stage": stage,
        "sequence": sequence,
        "calendar_month": calendar_month,
        "family": family,
        "width": width,
    }
    _validate_reason_context(value)
    return value


def _validate_reason_context(value: Mapping[str, Any]) -> None:
    if set(value) != {
        "stage",
        "sequence",
        "calendar_month",
        "family",
        "width",
    }:
        raise PrivateResultError("reason context schema changed")
    if (
        value["stage"] not in STAGES
        or value["calendar_month"] not in {None, *ALL_MONTHS}
        or value["family"] not in {None, *FAMILIES}
        or (
            value["width"] is not None
            and (
                not _exact_integral(value["width"])
                or int(value["width"]) not in assay.WIDTHS
            )
        )
    ):
        raise PrivateResultError("reason context enum invalid")
    sequence = value["sequence"]
    if sequence is not None and (
        not _exact_integral(sequence)
        or not 1 <= int(sequence) <= 56
    ):
        raise PrivateResultError("reason context sequence invalid")


def build_coverage_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expected = [
        *[("train", month) for month in assay.INNER_MONTHS],
        *[
            ("development", row[0])
            for row in assay.ELIGIBLE_GATE_BLOCKS["development"]
        ],
        *[
            ("validation", row[0])
            for row in assay.ELIGIBLE_GATE_BLOCKS["validation"]
        ],
    ]
    if len(rows) != len(expected):
        raise PrivateResultError("coverage diagnostic cardinality changed")
    output: list[dict[str, Any]] = []
    for row, (split, month) in zip(rows, expected):
        if set(row) != {
            "split",
            "calendar_month",
            "passed",
            "maps",
            "eligible_maps",
            "clusters",
            "eligible_clusters",
            "month",
            "leagues",
            "membership_sha256",
        }:
            raise PrivateResultError("coverage diagnostic schema changed")
        copied = dict(row)
        if copied["split"] != split or copied["calendar_month"] != month:
            raise PrivateResultError("coverage diagnostic order changed")
        if not isinstance(copied["passed"], bool):
            raise PrivateResultError("coverage gate flag invalid")
        for key in ("maps", "eligible_maps", "clusters", "eligible_clusters"):
            value = copied[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Integral)
                or int(value) < 0
            ):
                raise PrivateResultError("coverage count invalid")
        if (
            copied["eligible_maps"] > copied["maps"]
            or copied["eligible_clusters"] > copied["clusters"]
            or not _sha256(copied["membership_sha256"])
        ):
            raise PrivateResultError("coverage aggregate invalid")
        month_row = copied["month"]
        if not isinstance(month_row, Mapping) or set(month_row) != {
            "calendar_month",
            "maps",
            "eligible_maps",
            "clusters",
            "eligible_clusters",
        }:
            raise PrivateResultError("coverage month schema changed")
        if month_row["calendar_month"] != month:
            raise PrivateResultError("coverage month identity changed")
        for key in ("maps", "eligible_maps", "clusters", "eligible_clusters"):
            value = month_row[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Integral)
                or int(value) < 0
            ):
                raise PrivateResultError("coverage month count invalid")
        leagues = copied["leagues"]
        if not isinstance(leagues, list) or not 1 <= len(leagues) <= 32:
            raise PrivateResultError("coverage league cardinality invalid")
        observed_leagues: list[str] = []
        for league_row in leagues:
            if not isinstance(league_row, Mapping) or set(league_row) != {
                "league",
                "maps",
                "eligible_maps",
                "clusters",
                "eligible_clusters",
            }:
                raise PrivateResultError("coverage league schema changed")
            league = league_row["league"]
            if (
                not isinstance(league, str)
                or not 1 <= len(league) <= 64
            ):
                raise PrivateResultError("coverage league identity invalid")
            observed_leagues.append(league)
            for key in (
                "maps",
                "eligible_maps",
                "clusters",
                "eligible_clusters",
            ):
                value = league_row[key]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, numbers.Integral)
                    or int(value) < 0
                ):
                    raise PrivateResultError("coverage league count invalid")
        if observed_leagues != sorted(set(observed_leagues)):
            raise PrivateResultError("coverage league order changed")
        try:
            derived_passed = assay.coverage_gate_decision(
                overall={
                    key: copied[key]
                    for key in (
                        "maps",
                        "eligible_maps",
                        "clusters",
                        "eligible_clusters",
                    )
                },
                month_rows=(month_row,),
                league_rows=leagues,
            )
        except assay.RepresentationRankAssayError as exc:
            raise PrivateResultError("coverage decision inputs invalid") from exc
        if copied["passed"] is not derived_passed:
            raise PrivateResultError("coverage pass differs from frozen gates")
        output.append(copied)
    return output


def derive_population_identity(
    *,
    split: str,
    ordered_month_blocks: Sequence[tuple[str, int]],
) -> dict[str, Any]:
    """Derive the exact ordered aggregate population identity."""
    expected_blocks = {
        "development": len(assay.ELIGIBLE_GATE_BLOCKS["development"]),
        "validation": len(assay.ELIGIBLE_GATE_BLOCKS["validation"]),
    }
    if (
        split not in expected_blocks
        or len(ordered_month_blocks) != expected_blocks[split]
        or any(
            not _sha256(membership_sha256)
            or not _exact_integral(maps)
            or int(maps) <= 0
            for membership_sha256, maps in ordered_month_blocks
        )
    ):
        raise PrivateResultError("population month blocks invalid")
    maps = sum(int(block_maps) for _, block_maps in ordered_month_blocks)
    membership_sha256 = assay.canonical_sha256(
        {
            "split": split,
            "ordered_month_membership_sha256": [
                block_membership
                for block_membership, _ in ordered_month_blocks
            ],
            "maps": maps,
        }
    )
    return {
        "maps": maps,
        "membership_sha256": membership_sha256,
    }


def _gate_candidate(
    *,
    role: str,
    width: int | None,
    raw: Mapping[str, Any],
    months: Sequence[str],
) -> dict[str, Any]:
    if set(raw) != {"passed", "comparators"} or not isinstance(
        raw["passed"], bool
    ):
        raise PrivateResultError("selector candidate schema changed")
    comparators = raw["comparators"]
    if not isinstance(comparators, Mapping) or set(comparators) != set(
        COMPARATORS
    ):
        raise PrivateResultError("selector comparators changed")
    comparator_rows = []
    for comparator in COMPARATORS:
        source = comparators[comparator]
        if not isinstance(source, Mapping) or set(source) != {
            *METRICS,
            "blocks",
        }:
            raise PrivateResultError("selector comparator schema changed")
        metric_rows = []
        for metric in METRICS:
            gate = source[metric]
            if not isinstance(gate, Mapping) or set(gate) != {
                "upper",
                "limit",
                "strict",
                "passed",
            }:
                raise PrivateResultError("selector metric schema changed")
            if (
                not _required_finite(gate["upper"])
                or not _required_finite(gate["limit"])
                or not isinstance(gate["strict"], bool)
                or not isinstance(gate["passed"], bool)
            ):
                raise PrivateResultError("selector metric value invalid")
            try:
                decision = assay.metric_gate_decision(
                    comparator=comparator,
                    metric=metric,
                    upper=gate["upper"],
                )
            except assay.RepresentationRankAssayError as exc:
                raise PrivateResultError(
                    "selector metric decision input invalid"
                ) from exc
            if (
                float(gate["limit"]) != float(decision["limit"])
                or gate["strict"] is not decision["strict"]
                or gate["passed"] is not decision["passed"]
            ):
                raise PrivateResultError(
                    "selector metric differs from frozen decision"
                )
            metric_rows.append(
                {
                    "metric": metric,
                    "upper": float(gate["upper"]),
                    "limit": float(decision["limit"]),
                    "strict": decision["strict"],
                    "passed": decision["passed"],
                }
            )
        blocks = source["blocks"]
        if not isinstance(blocks, Mapping) or set(blocks) != set(months):
            raise PrivateResultError("selector block membership changed")
        block_rows = []
        for month in months:
            gate = blocks[month]
            if not isinstance(gate, Mapping) or set(gate) != {
                "delta",
                "passed",
            }:
                raise PrivateResultError("selector block schema changed")
            if not _required_finite(gate["delta"]) or not isinstance(
                gate["passed"], bool
            ):
                raise PrivateResultError("selector block value invalid")
            try:
                block_passed = assay.block_gate_decision(gate["delta"])
            except assay.RepresentationRankAssayError as exc:
                raise PrivateResultError(
                    "selector block decision input invalid"
                ) from exc
            if gate["passed"] is not block_passed:
                raise PrivateResultError(
                    "selector block differs from frozen decision"
                )
            block_rows.append(
                {
                    "calendar_month": month,
                    "delta": float(gate["delta"]),
                    "passed": block_passed,
                }
            )
        comparator_rows.append(
            {
                "comparator": comparator,
                "metrics": metric_rows,
                "blocks": block_rows,
            }
        )
    derived_passed = all(
        gate["passed"]
        for comparator_row in comparator_rows
        for group in ("metrics", "blocks")
        for gate in comparator_row[group]
    )
    if raw["passed"] is not derived_passed:
        raise PrivateResultError(
            "selector candidate differs from frozen decisions"
        )
    return {
        "candidate_role": role,
        "width": width,
        "passed": derived_passed,
        "comparators": comparator_rows,
    }


def build_development_diagnostics(
    raw: Mapping[str, Any] | None,
    *,
    status: str,
    selected_width: int | None,
) -> dict[str, Any]:
    if status not in {"passed", "failed", "not_run"}:
        raise PrivateResultError("development diagnostic status invalid")
    candidates: list[dict[str, Any]] = []
    if status in {"passed", "failed"} and raw is not None:
        selected_index = (
            assay.WIDTHS.index(selected_width)
            if (
                not isinstance(selected_width, bool)
                and selected_width in assay.WIDTHS
            )
            else None
        )
        raw_widths = raw.get("widths") if isinstance(raw, Mapping) else None
        expected_widths = (
            assay.WIDTHS[: selected_index + 1]
            if status == "passed" and selected_index is not None
            else (
                ()
                if isinstance(raw_widths, Mapping) and not raw_widths
                else assay.WIDTHS
            )
        )
        if (
            not isinstance(raw, Mapping)
            or set(raw) != {"M8_prerequisite", "widths", "locked_width"}
            or raw["locked_width"] != selected_width
            or (status == "passed" and selected_index is None)
            or (status == "failed" and selected_width is not None)
            or not isinstance(raw["widths"], Mapping)
            or tuple(raw["widths"]) != tuple(
                str(width) for width in expected_widths
            )
        ):
            raise PrivateResultError("development selector output changed")
        months = tuple(
            row[0] for row in assay.ELIGIBLE_GATE_BLOCKS["development"]
        )
        candidates.append(
            _gate_candidate(
                role="M8_prerequisite",
                width=8,
                raw=raw["M8_prerequisite"],
                months=months,
            )
        )
        candidates.extend(
            _gate_candidate(
                role="candidate_width",
                width=width,
                raw=raw["widths"][str(width)],
                months=months,
            )
            for width in expected_widths
        )
    elif (
        status == "passed"
        or raw is not None
        or selected_width is not None
    ):
        raise PrivateResultError("unrun development diagnostics contain values")
    output = {
        "status": status,
        "bootstrap": {
            "replicates": assay.PRIMARY_REPLICATES,
            "seed": assay.DEVELOPMENT_SEED,
            "endpoint_1_indexed": assay.DEVELOPMENT_ENDPOINT,
        },
        "selected_width": selected_width,
        "candidates": candidates,
    }
    _validate_development_diagnostics(output)
    return output


def _validate_development_diagnostics(value: Mapping[str, Any]) -> None:
    if set(value) != {"status", "bootstrap", "selected_width", "candidates"}:
        raise PrivateResultError("development diagnostic schema changed")
    if value["status"] not in {"passed", "failed", "not_run"}:
        raise PrivateResultError("development diagnostic status invalid")
    if value["bootstrap"] != {
        "replicates": assay.PRIMARY_REPLICATES,
        "seed": assay.DEVELOPMENT_SEED,
        "endpoint_1_indexed": assay.DEVELOPMENT_ENDPOINT,
    }:
        raise PrivateResultError("development bootstrap identity changed")
    candidates = value["candidates"]
    if not isinstance(candidates, list) or len(candidates) > 5:
        raise PrivateResultError("development candidate cardinality invalid")
    if value["status"] == "passed":
        selected_width = value["selected_width"]
        selected_index = (
            assay.WIDTHS.index(selected_width)
            if (
                not isinstance(selected_width, bool)
                and selected_width in assay.WIDTHS
            )
            else None
        )
        if selected_index is None or len(candidates) != selected_index + 2:
            raise PrivateResultError("development accepted diagnostics incomplete")
        expected = [
            ("M8_prerequisite", 8),
            *[
                ("candidate_width", width)
                for width in assay.WIDTHS[: selected_index + 1]
            ],
        ]
        for row, (role, width) in zip(candidates, expected):
            _validate_candidate_dto(
                row,
                role=role,
                width=width,
                months=tuple(
                    item[0]
                    for item in assay.ELIGIBLE_GATE_BLOCKS["development"]
                ),
            )
        if (
            candidates[0]["passed"] is not True
            or candidates[-1]["passed"] is not True
            or any(
                row["passed"] is not False for row in candidates[1:-1]
            )
        ):
            raise PrivateResultError(
                "development width is not the smallest passing candidate"
            )
    elif value["status"] == "failed":
        if value["selected_width"] is not None or len(candidates) not in {
            0,
            1,
            5,
        }:
            raise PrivateResultError("development failure diagnostics invalid")
        if candidates:
            expected = [
                ("M8_prerequisite", 8),
                *(
                    []
                    if len(candidates) == 1
                    else [
                        ("candidate_width", width)
                        for width in assay.WIDTHS
                    ]
                ),
            ]
            for row, (role, width) in zip(candidates, expected):
                _validate_candidate_dto(
                    row,
                    role=role,
                    width=width,
                    months=tuple(
                        item[0]
                        for item in assay.ELIGIBLE_GATE_BLOCKS[
                            "development"
                        ]
                    ),
                )
            if (
                len(candidates) == 1
                and candidates[0]["passed"] is not False
            ) or (
                len(candidates) == 5
                and (
                    candidates[0]["passed"] is not True
                    or any(
                        row["passed"] is not False
                        for row in candidates[1:]
                    )
                )
            ):
                raise PrivateResultError(
                    "development failure differs from frozen gates"
                )
    elif value["selected_width"] is not None or candidates:
        raise PrivateResultError("unrun development diagnostics are nonempty")


def build_validation_diagnostics(
    raw: Mapping[str, Any] | None,
    *,
    status: str,
    selected_width: int | None,
) -> dict[str, Any]:
    if status not in {"passed", "failed", "not_run"}:
        raise PrivateResultError("validation diagnostic status invalid")
    candidates: list[dict[str, Any]] = []
    if status in {"passed", "failed"} and raw is not None:
        if (
            not isinstance(raw, Mapping)
            or set(raw) != {"passed", "locked_width", "M8", "locked"}
            or raw["passed"] is not (status == "passed")
            or raw["locked_width"] != selected_width
            or not _exact_integral(selected_width)
            or int(selected_width) not in assay.WIDTHS
        ):
            raise PrivateResultError("validation selector output changed")
        months = tuple(
            row[0] for row in assay.ELIGIBLE_GATE_BLOCKS["validation"]
        )
        candidates = [
            _gate_candidate(
                role="M8_reference",
                width=8,
                raw=raw["M8"],
                months=months,
            ),
            _gate_candidate(
                role="locked_width",
                width=selected_width,
                raw=raw["locked"],
                months=months,
            ),
        ]
    elif (
        status == "passed"
        or raw is not None
        or selected_width is not None
    ):
        raise PrivateResultError("unrun validation diagnostics contain values")
    output = {
        "status": status,
        "bootstrap": {
            "replicates": assay.PRIMARY_REPLICATES,
            "seed": assay.VALIDATION_SEED,
            "endpoint_1_indexed": assay.VALIDATION_ENDPOINT,
        },
        "selected_width": selected_width,
        "candidates": candidates,
    }
    _validate_validation_diagnostics(output)
    return output


def _validate_validation_diagnostics(value: Mapping[str, Any]) -> None:
    if set(value) != {"status", "bootstrap", "selected_width", "candidates"}:
        raise PrivateResultError("validation diagnostic schema changed")
    if value["status"] not in {"passed", "failed", "not_run"}:
        raise PrivateResultError("validation diagnostic status invalid")
    if value["bootstrap"] != {
        "replicates": assay.PRIMARY_REPLICATES,
        "seed": assay.VALIDATION_SEED,
        "endpoint_1_indexed": assay.VALIDATION_ENDPOINT,
    }:
        raise PrivateResultError("validation bootstrap identity changed")
    candidates = value["candidates"]
    if not isinstance(candidates, list) or len(candidates) > 2:
        raise PrivateResultError("validation candidate cardinality invalid")
    if value["status"] == "passed":
        width = value["selected_width"]
        if (
            not _exact_integral(width)
            or int(width) not in assay.WIDTHS
            or len(candidates) != 2
        ):
            raise PrivateResultError("validation accepted diagnostics incomplete")
        _validate_candidate_dto(
            candidates[0],
            role="M8_reference",
            width=8,
            months=tuple(
                item[0] for item in assay.ELIGIBLE_GATE_BLOCKS["validation"]
            ),
        )
        _validate_candidate_dto(
            candidates[1],
            role="locked_width",
            width=width,
            months=tuple(
                item[0] for item in assay.ELIGIBLE_GATE_BLOCKS["validation"]
            ),
        )
        if any(row["passed"] is not True for row in candidates):
            raise PrivateResultError(
                "validation accepted a failing candidate"
            )
    elif value["status"] == "failed":
        width = value["selected_width"]
        if not candidates:
            if width is not None:
                raise PrivateResultError(
                    "empty validation failure has a selected width"
                )
        elif (
            not _exact_integral(width)
            or int(width) not in assay.WIDTHS
            or len(candidates) != 2
        ):
            raise PrivateResultError(
                "validation failure diagnostics incomplete"
            )
        else:
            _validate_candidate_dto(
                candidates[0],
                role="M8_reference",
                width=8,
                months=tuple(
                    item[0]
                    for item in assay.ELIGIBLE_GATE_BLOCKS["validation"]
                ),
            )
            _validate_candidate_dto(
                candidates[1],
                role="locked_width",
                width=int(width),
                months=tuple(
                    item[0]
                    for item in assay.ELIGIBLE_GATE_BLOCKS["validation"]
                ),
            )
            if all(row["passed"] is True for row in candidates):
                raise PrivateResultError(
                    "validation failure has no recomputed failing candidate"
                )
    elif value["selected_width"] is not None or candidates:
        raise PrivateResultError("unrun validation diagnostics are nonempty")


def _validate_candidate_dto(
    value: Mapping[str, Any],
    *,
    role: str,
    width: int,
    months: Sequence[str],
) -> None:
    if set(value) != {"candidate_role", "width", "passed", "comparators"}:
        raise PrivateResultError("candidate DTO schema changed")
    if (
        value["candidate_role"] != role
        or not _exact_integral(value["width"])
        or int(value["width"]) != width
        or not isinstance(value["passed"], bool)
    ):
        raise PrivateResultError("candidate DTO identity invalid")
    comparators = value["comparators"]
    if not isinstance(comparators, list) or len(comparators) != 2:
        raise PrivateResultError("candidate comparator cardinality invalid")
    derived_candidate_passed = True
    for row, comparator in zip(comparators, COMPARATORS):
        if set(row) != {"comparator", "metrics", "blocks"}:
            raise PrivateResultError("comparator DTO schema changed")
        if row["comparator"] != comparator:
            raise PrivateResultError("comparator DTO order changed")
        metrics = row["metrics"]
        blocks = row["blocks"]
        if (
            not isinstance(metrics, list)
            or len(metrics) != 3
            or not isinstance(blocks, list)
            or len(blocks) != len(months)
        ):
            raise PrivateResultError("gate DTO cardinality invalid")
        for gate, metric in zip(metrics, METRICS):
            if set(gate) != {
                "metric",
                "upper",
                "limit",
                "strict",
                "passed",
            }:
                raise PrivateResultError("metric DTO schema changed")
            if (
                gate["metric"] != metric
                or not _required_finite(gate["upper"])
                or not _required_finite(gate["limit"])
                or not isinstance(gate["strict"], bool)
                or not isinstance(gate["passed"], bool)
            ):
                raise PrivateResultError("metric DTO invalid")
            try:
                decision = assay.metric_gate_decision(
                    comparator=comparator,
                    metric=metric,
                    upper=gate["upper"],
                )
            except assay.RepresentationRankAssayError as exc:
                raise PrivateResultError(
                    "metric DTO decision input invalid"
                ) from exc
            if (
                float(gate["limit"]) != float(decision["limit"])
                or gate["strict"] is not decision["strict"]
                or gate["passed"] is not decision["passed"]
            ):
                raise PrivateResultError(
                    "metric DTO differs from frozen decision"
                )
            derived_candidate_passed = (
                derived_candidate_passed and bool(decision["passed"])
            )
        for gate, month in zip(blocks, months):
            if set(gate) != {"calendar_month", "delta", "passed"}:
                raise PrivateResultError("block DTO schema changed")
            if (
                gate["calendar_month"] != month
                or not _required_finite(gate["delta"])
                or not isinstance(gate["passed"], bool)
            ):
                raise PrivateResultError("block DTO invalid")
            try:
                block_passed = assay.block_gate_decision(gate["delta"])
            except assay.RepresentationRankAssayError as exc:
                raise PrivateResultError(
                    "block DTO decision input invalid"
                ) from exc
            if gate["passed"] is not block_passed:
                raise PrivateResultError(
                    "block DTO differs from frozen decision"
                )
            derived_candidate_passed = (
                derived_candidate_passed and block_passed
            )
    if value["passed"] is not derived_candidate_passed:
        raise PrivateResultError(
            "candidate DTO differs from frozen decisions"
        )


def _population_summary_complete(row: Mapping[str, Any]) -> bool:
    maps = row["maps"]
    clusters = row["clusters"]
    return (
        _exact_integral(maps)
        and int(maps) > 0
        and _exact_integral(clusters)
        and 0 < int(clusters) <= int(maps)
        and _sha256(row["membership_sha256"])
    )


def _optimization_summary_complete(row: Mapping[str, Any]) -> bool:
    if (
        not _required_finite(row["objective"])
        or float(row["objective"]) < 0
    ):
        return False
    try:
        return assay.optimization_gate_decision(
            converged_starts=row["converged_starts"],
            max_gradient=row["max_gradient"],
            stability_rms=row["stability_rms"],
        )
    except assay.RepresentationRankAssayError:
        return False


def _scoring_summary_complete(row: Mapping[str, Any]) -> bool:
    return all(
        _required_finite(row[key]) and float(row[key]) >= 0
        for key in SCORING_SUMMARY_FIELDS
    )


def _failed_summary_shape(row: Mapping[str, Any]) -> str | None:
    if not _population_summary_complete(row):
        return None
    optimization = _optimization_summary_complete(row)
    optimization_empty = all(
        row[key] is None for key in OPTIMIZATION_SUMMARY_FIELDS
    )
    scoring_empty = all(row[key] is None for key in SCORING_SUMMARY_FIELDS)
    if optimization_empty and scoring_empty:
        return "fit"
    if optimization and scoring_empty:
        return "scoring"
    return None


def _validate_fit_plan(
    rows: Sequence[Mapping[str, Any]],
    *,
    coverage: Sequence[Mapping[str, Any]],
) -> None:
    if not isinstance(rows, list) or len(rows) != 56:
        raise PrivateResultError("fit plan must contain exactly 56 slots")
    coverage_by_block = {
        (row["split"], row["calendar_month"]): row for row in coverage
    }
    frozen = empty_fit_plan()
    for index, (row, expected) in enumerate(zip(rows, frozen), start=1):
        if (
            not isinstance(row, Mapping)
            or set(row) != set(FIT_PLAN_COLUMNS)
            or not _exact_integral(row.get("sequence"))
            or int(row["sequence"]) != index
        ):
            raise PrivateResultError("fit-plan schema or sequence changed")
        for key in (
            "stage",
            "split",
            "calendar_month",
            "family",
            "fit_role",
        ):
            if row[key] != expected[key]:
                raise PrivateResultError("fit-plan frozen order changed")
        if index <= 36:
            for key in ("lambda_ally", "lambda_enemy", "width"):
                if row[key] != expected[key]:
                    raise PrivateResultError("inner fit-plan parameters changed")
        elif index <= 52 and row["width"] != expected["width"]:
            raise PrivateResultError("development width plan changed")
        elif index > 52:
            if row["fit_role"] == "M8_reference" and row["width"] != 8:
                raise PrivateResultError("validation M8 width changed")
            if row["fit_role"] == "locked_width" and row["width"] not in {
                None,
                *assay.WIDTHS,
            }:
                raise PrivateResultError("validation locked width invalid")
        if row["execution_status"] not in EXECUTION_STATUSES:
            raise PrivateResultError("fit execution status invalid")
        for key in ("lambda_ally", "lambda_enemy"):
            if not _optional_finite(row[key], nonnegative=True):
                raise PrivateResultError("fit penalty invalid")
        if row["width"] is not None and (
            not _exact_integral(row["width"])
            or int(row["width"]) not in assay.WIDTHS
        ):
            raise PrivateResultError("fit width invalid")
        for key in (
            "objective",
            "max_gradient",
            "stability_rms",
            "log_loss_total",
            "brier_total",
        ):
            if not _optional_finite(row[key], nonnegative=True):
                raise PrivateResultError("fit summary invalid")
        for key in ("maps", "clusters", "converged_starts"):
            value = row[key]
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, numbers.Integral)
                or int(value) < 0
            ):
                raise PrivateResultError("fit count invalid")
        digest = row["membership_sha256"]
        if digest is not None and not _sha256(digest):
            raise PrivateResultError("fit membership digest invalid")
        status = row["execution_status"]
        if status == "not_run":
            if any(row[key] is not None for key in SUMMARY_FIELDS):
                raise PrivateResultError("not-run fit contains summaries")
            continue
        if (
            not _required_finite(row["lambda_ally"])
            or float(row["lambda_ally"]) <= 0
            or not _required_finite(row["lambda_enemy"])
            or float(row["lambda_enemy"]) <= 0
            or not _exact_integral(row["width"])
            or int(row["width"]) not in assay.WIDTHS
        ):
            raise PrivateResultError("executed fit identity is incomplete")
        block = coverage_by_block.get((row["split"], row["calendar_month"]))
        if (
            block is None
            or row["maps"] != block["eligible_maps"]
            or row["clusters"] != block["eligible_clusters"]
            or row["membership_sha256"] != block["membership_sha256"]
        ):
            raise PrivateResultError(
                "fit population summary differs from coverage block"
            )
        if status in {"passed", "aliased"}:
            if (
                not _population_summary_complete(row)
                or not _optimization_summary_complete(row)
                or not _scoring_summary_complete(row)
            ):
                raise PrivateResultError("completed fit summary is incomplete")
        elif status == "failed" and _failed_summary_shape(row) is None:
            raise PrivateResultError("failed fit summary shape is invalid")
        if status == "aliased":
            if (
                index not in {54, 56}
                or row["fit_role"] != "M8_reference"
                or int(row["width"]) != 8
            ):
                raise PrivateResultError("only validation M8 may be aliased")
            source = rows[index - 2]
            if (
                source["execution_status"] != "passed"
                or source["fit_role"] != "locked_width"
                or source["calendar_month"] != row["calendar_month"]
                or any(source[key] != row[key] for key in ALIAS_EQUAL_FIELDS)
            ):
                raise PrivateResultError(
                    "validation M8 alias differs from locked-width source"
                )


def _expected_completed_status(sequence: int, *, locked_width: int | None) -> str:
    if locked_width == 8 and sequence in {54, 56}:
        return "aliased"
    return "passed"


def _recompute_inner_penalties(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Select both penalties from the exact completed 36-row inner ledger."""
    if len(rows) < 36 or any(
        row["execution_status"] != "passed" for row in rows[:36]
    ):
        raise PrivateResultError("inner penalty ledger is incomplete")
    selected: dict[str, float] = {}
    for family in ("ally", "enemy"):
        family_rows = []
        for row in rows[:36]:
            if row["family"] != family:
                continue
            family_rows.append(
                {
                    "family": family,
                    "lambda": (
                        row["lambda_ally"]
                        if family == "ally"
                        else row["lambda_enemy"]
                    ),
                    "width": row["width"],
                    "calendar_month": row["calendar_month"],
                    "split": row["split"],
                    "maps": row["maps"],
                    "clusters": row["clusters"],
                    "membership_sha256": row["membership_sha256"],
                    "log_loss_total": row["log_loss_total"],
                    "brier_total": row["brier_total"],
                    "strictly_earlier_fit": True,
                    "cluster_atomic": True,
                }
            )
        selected[f"lambda_{family}"] = float(
            assay.select_separate_penalty(family_rows, family=family)
        )
    return selected


def _validate_terminal_ledger_semantics(
    payload: Mapping[str, Any],
    *,
    coverage: Sequence[Mapping[str, Any]],
) -> None:
    rows = payload["fit_plan"]
    stages = payload["stage_status"]
    actual = payload["fit_counts"]["actual"]
    failed_stages = [
        stage for stage in STAGES if stages[stage]["status"] == "failed"
    ]
    if payload["run_status"] == "accepted":
        locked_width = int(payload["selected_width"])
        expected = [
            _expected_completed_status(sequence, locked_width=locked_width)
            for sequence in range(1, 57)
        ]
        if (
            failed_stages
            or [stages[stage]["status"] for stage in STAGES]
            != ["passed", "passed", "passed"]
            or [row["execution_status"] for row in rows] != expected
            or actual != expected.count("passed")
            or any(not row["passed"] for row in coverage)
        ):
            raise PrivateResultError("accepted execution ledger is incomplete")
        effective_width = locked_width
    else:
        if len(failed_stages) != 1:
            raise PrivateResultError(
                "inconclusive result must contain exactly one failed stage"
            )
        failed_stage = failed_stages[0]
        reason_code = payload["reason_code"]
        context = payload["reason_context"]
        if (
            stages[failed_stage]["reason_code"] != reason_code
            or context["stage"] != failed_stage
            or reason_code not in STAGE_REASON_CODES[failed_stage]
        ):
            raise PrivateResultError(
                "terminal, stage, and context reasons disagree"
            )
        if failed_stage in {"development", "validation"}:
            selector = payload[f"{failed_stage}_diagnostics"]
            has_failure_diagnostics = bool(selector["candidates"])
            expected_failure_diagnostics = (
                reason_code == f"{failed_stage}_gate_failed"
            )
            if has_failure_diagnostics is not expected_failure_diagnostics:
                raise PrivateResultError(
                    "stage failure diagnostics differ from terminal reason"
                )
        if reason_code == "coverage_gate_failed":
            first_failed = next(
                (row for row in coverage if row["passed"] is False),
                None,
            )
            first_failed_stage = (
                "inner"
                if first_failed is not None and first_failed["split"] == "train"
                else first_failed["split"]
                if first_failed is not None
                else None
            )
            if (
                first_failed is None
                or first_failed_stage != failed_stage
                or [stages[stage]["status"] for stage in STAGES]
                != [
                    "failed" if stage == failed_stage else "not_run"
                    for stage in STAGES
                ]
                or context
                != {
                    "stage": failed_stage,
                    "sequence": None,
                    "calendar_month": first_failed["calendar_month"],
                    "family": None,
                    "width": None,
                }
                or actual != 0
                or any(row["execution_status"] != "not_run" for row in rows)
            ):
                raise PrivateResultError(
                    "coverage preflight terminal chronology invalid"
                )
            return
        if any(not row["passed"] for row in coverage):
            raise PrivateResultError(
                "non-coverage terminal contains a failed coverage block"
            )
        failed_stage_index = STAGES.index(failed_stage)
        if [stages[stage]["status"] for stage in STAGES] != [
            "passed" if index < failed_stage_index
            else "failed" if index == failed_stage_index
            else "not_run"
            for index, stage in enumerate(STAGES)
        ]:
            raise PrivateResultError("inconclusive stage chronology invalid")
        development = payload["development_diagnostics"]
        effective_width = (
            int(development["selected_width"])
            if development["status"] == "passed"
            else None
        )
        if reason_code in {"fit_unavailable", "scoring_unavailable"}:
            sequence = context["sequence"]
            if not _exact_integral(sequence):
                raise PrivateResultError("cell failure lacks a sequence")
            sequence = int(sequence)
            failed_row = rows[sequence - 1]
            expected_shape = (
                "fit" if reason_code == "fit_unavailable" else "scoring"
            )
            if (
                failed_row["execution_status"] != "failed"
                or failed_row["stage"] != failed_stage
                or _failed_summary_shape(failed_row) != expected_shape
                or context
                != {
                    "stage": failed_stage,
                    "sequence": sequence,
                    "calendar_month": failed_row["calendar_month"],
                    "family": failed_row["family"],
                    "width": failed_row["width"],
                }
                or (
                    effective_width == 8
                    and sequence in {54, 56}
                )
            ):
                raise PrivateResultError("failed-cell identity or shape invalid")
            expected = [
                (
                    _expected_completed_status(
                        row_sequence,
                        locked_width=effective_width,
                    )
                    if row_sequence < sequence
                    else "failed"
                    if row_sequence == sequence
                    else "not_run"
                )
                for row_sequence in range(1, 57)
            ]
        else:
            boundary = {
                "penalty_selection_failed": 36,
                "development_gate_failed": 52,
                "validation_gate_failed": 56,
            }.get(reason_code)
            if (
                boundary is None
                or context
                != {
                    "stage": failed_stage,
                    "sequence": None,
                    "calendar_month": None,
                    "family": None,
                    "width": None,
                }
            ):
                raise PrivateResultError("stage gate terminal identity invalid")
            expected = [
                (
                    _expected_completed_status(
                        sequence,
                        locked_width=effective_width,
                    )
                    if sequence <= boundary
                    else "not_run"
                )
                for sequence in range(1, 57)
            ]
        if [row["execution_status"] for row in rows] != expected:
            raise PrivateResultError("fit ledger is not a completed prefix")
        if actual != expected.count("passed"):
            raise PrivateResultError("actual fit count differs from prefix")
    inner_complete = all(
        row["execution_status"] == "passed" for row in rows[:36]
    )
    recomputed_penalties: dict[str, float] | None = None
    penalty_selection_error: assay.RepresentationRankAssayError | None = None
    if inner_complete:
        try:
            recomputed_penalties = _recompute_inner_penalties(rows)
        except assay.RepresentationRankAssayError as exc:
            penalty_selection_error = exc
    penalties = payload["penalties"]
    if payload["reason_code"] == "penalty_selection_failed":
        if (
            not inner_complete
            or penalty_selection_error is None
            or penalties != {"lambda_ally": None, "lambda_enemy": None}
        ):
            raise PrivateResultError(
                "penalty terminal is not the first recomputed failure"
            )
    elif inner_complete:
        if (
            penalty_selection_error is not None
            or recomputed_penalties is None
            or penalties != recomputed_penalties
        ):
            raise PrivateResultError(
                "selected penalties differ from the inner ledger"
            )
    elif penalties != {"lambda_ally": None, "lambda_enemy": None}:
        raise PrivateResultError(
            "incomplete inner ledger contains selected penalties"
        )
    if effective_width is not None:
        for sequence in (53, 55):
            row = rows[sequence - 1]
            if (
                row["execution_status"] != "not_run"
                and row["width"] != effective_width
            ):
                raise PrivateResultError(
                    "validation locked fit differs from selected width"
                )
    executed_later_stage_rows = [
        row
        for row in rows[36:]
        if row["execution_status"] != "not_run"
    ]
    if executed_later_stage_rows:
        if (
            penalties["lambda_ally"] not in assay.PENALTY_GRID
            or penalties["lambda_enemy"] not in assay.PENALTY_GRID
            or any(
                row["lambda_ally"] != penalties["lambda_ally"]
                or row["lambda_enemy"] != penalties["lambda_enemy"]
                for row in executed_later_stage_rows
            )
        ):
            raise PrivateResultError(
                "later-stage fit penalties differ from selected penalties"
            )


def validate_private_result(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping) or set(payload) != TOP_LEVEL_KEYS:
        raise PrivateResultError("private result schema changed")
    unsigned = dict(payload)
    claimed = unsigned.pop("artifact_sha256")
    try:
        expected_sha256 = assay.canonical_sha256(unsigned)
    except (TypeError, ValueError) as exc:
        raise PrivateResultError("private result is not canonical JSON") from exc
    if claimed != expected_sha256:
        raise PrivateResultError("private result artifact hash changed")
    _walk_forbidden(payload)
    if (
        payload["schema_id"] != SCHEMA_ID
        or payload["aggregate_only"] is not True
        or payload["development_only"] is not True
        or payload["run_status"] not in {"accepted", "inconclusive"}
        or payload["fallback"] not in {"none", "M0"}
        or payload["selected_model"] not in {"latent_candidate", "M0"}
    ):
        raise PrivateResultError("private result terminal state invalid")
    for key in (
        "contract_artifact_sha256",
        "contract_review_core_sha256",
        "runner_review_permit_raw_sha256",
        "source_identity_sha256",
        "runtime_identity_sha256",
        "feature_domain_sha256",
        "target_domain_sha256",
        "fit_availability_domain_sha256",
    ):
        if not _sha256(payload[key]):
            raise PrivateResultError("private result identity invalid")
    false_fields = (
        "final_target_loaded",
        "publication_authority",
        "production_authority",
        "reliability_authority",
        "promotion_authority",
        "sota_claim_authority",
    )
    if any(payload[key] is not False for key in false_fields):
        raise PrivateResultError("private result authority ceiling exceeded")
    coverage = payload["coverage_diagnostics"]
    if not isinstance(coverage, list):
        raise PrivateResultError("coverage diagnostics must be an array")
    coverage = build_coverage_diagnostics(coverage)
    population = payload["population"]
    if not isinstance(population, Mapping) or set(population) != {
        "development_maps",
        "development_membership_sha256",
        "validation_maps",
        "validation_membership_sha256",
        "combined_maps",
        "combined_membership_sha256",
    }:
        raise PrivateResultError("private population schema changed")
    development = derive_population_identity(
        split="development",
        ordered_month_blocks=tuple(
            (row["membership_sha256"], row["eligible_maps"])
            for row in coverage
            if row["split"] == "development"
        ),
    )
    validation = derive_population_identity(
        split="validation",
        ordered_month_blocks=tuple(
            (row["membership_sha256"], row["eligible_maps"])
            for row in coverage
            if row["split"] == "validation"
        ),
    )
    combined_maps = development["maps"] + validation["maps"]
    combined_membership_sha256 = assay.canonical_sha256(
        {
            "development_membership_sha256": development[
                "membership_sha256"
            ],
            "validation_membership_sha256": validation[
                "membership_sha256"
            ],
            "maps": combined_maps,
        }
    )
    expected_population = {
        "development_maps": development["maps"],
        "development_membership_sha256": development[
            "membership_sha256"
        ],
        "validation_maps": validation["maps"],
        "validation_membership_sha256": validation["membership_sha256"],
        "combined_maps": combined_maps,
        "combined_membership_sha256": combined_membership_sha256,
    }
    if (
        development["maps"] != 981
        or validation["maps"] != 1084
        or combined_maps != 2065
        or population != expected_population
    ):
        raise PrivateResultError("private population identity invalid")
    penalties = payload["penalties"]
    if not isinstance(penalties, Mapping) or set(penalties) != {
        "lambda_ally",
        "lambda_enemy",
    }:
        raise PrivateResultError("penalty result schema changed")
    if any(
        value is not None
        and (
            isinstance(value, bool)
            or not _optional_finite(value, nonnegative=True)
        )
        for value in penalties.values()
    ):
        raise PrivateResultError("penalty result value invalid")
    if payload["run_status"] == "accepted":
        if (
            payload["fallback"] != "none"
            or payload["selected_model"] != "latent_candidate"
            or not _exact_integral(payload["selected_width"])
            or int(payload["selected_width"]) not in assay.WIDTHS
            or payload["reason_code"] is not None
            or payload["reason_context"] is not None
            or any(
                float(penalties[key]) not in assay.PENALTY_GRID
                for key in penalties
            )
        ):
            raise PrivateResultError("accepted result state invalid")
    else:
        if (
            payload["fallback"] != "M0"
            or payload["selected_model"] != "M0"
            or payload["selected_width"] is not None
            or payload["reason_code"] not in REASON_CODES
            or not isinstance(payload["reason_context"], Mapping)
        ):
            raise PrivateResultError("fallback result state invalid")
        _validate_reason_context(payload["reason_context"])
    counts = payload["fit_counts"]
    if (
        not isinstance(counts, Mapping)
        or set(counts) != {"actual", "planned_slots"}
        or not _exact_integral(counts["planned_slots"])
        or int(counts["planned_slots"]) != 56
    ):
        raise PrivateResultError("fit count schema changed")
    if (
        isinstance(counts["actual"], bool)
        or not isinstance(counts["actual"], numbers.Integral)
        or not 0 <= int(counts["actual"]) <= 56
    ):
        raise PrivateResultError("actual fit count invalid")
    if (
        not isinstance(payload["stage_status"], Mapping)
        or set(payload["stage_status"])
        != {"inner", "development", "validation"}
    ):
        raise PrivateResultError("stage status schema changed")
    for stage, value in payload["stage_status"].items():
        if (
            not isinstance(value, Mapping)
            or set(value) != {"status", "reason_code"}
            or value["status"] not in {
            "passed",
            "failed",
            "not_run",
            }
        ):
            raise PrivateResultError("stage status invalid")
        if (
            value["status"] == "failed"
            and value["reason_code"] not in REASON_CODES
        ) or (
            value["status"] != "failed" and value["reason_code"] is not None
        ):
            raise PrivateResultError("stage reason code invalid")
    if not isinstance(payload["development_diagnostics"], Mapping):
        raise PrivateResultError("development diagnostics must be an object")
    _validate_development_diagnostics(payload["development_diagnostics"])
    if not isinstance(payload["validation_diagnostics"], Mapping):
        raise PrivateResultError("validation diagnostics must be an object")
    _validate_validation_diagnostics(payload["validation_diagnostics"])
    if (
        payload["development_diagnostics"]["status"]
        != payload["stage_status"]["development"]["status"]
        or payload["validation_diagnostics"]["status"]
        != payload["stage_status"]["validation"]["status"]
    ):
        raise PrivateResultError("selector diagnostics differ from stage status")
    if payload["run_status"] == "accepted" and (
        payload["development_diagnostics"]["selected_width"]
        != payload["selected_width"]
        or payload["validation_diagnostics"]["selected_width"]
        != payload["selected_width"]
    ):
        raise PrivateResultError("selector diagnostics differ from locked width")
    _validate_fit_plan(payload["fit_plan"], coverage=coverage)
    passed_fits = sum(
        row["execution_status"] == "passed" for row in payload["fit_plan"]
    )
    if counts["actual"] != passed_fits:
        raise PrivateResultError("actual fit count differs from complete ledger")
    _validate_terminal_ledger_semantics(payload, coverage=coverage)


def with_artifact_sha256(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(unsigned)
    payload.pop("artifact_sha256", None)
    try:
        digest = assay.canonical_sha256(payload)
    except (TypeError, ValueError) as exc:
        raise PrivateResultError("private result is not canonical JSON") from exc
    return {**payload, "artifact_sha256": digest}


def write_private_result(payload: Mapping[str, Any], *, path: Path) -> None:
    validate_private_result(payload)
    raw = assay.canonical_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise PrivateResultError("private result temporary path already exists")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def verify_private_result(*, path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PrivateResultError("private result is not a regular file")
    raw = path.read_bytes()
    payload = json.loads(raw)
    if raw != assay.canonical_bytes(payload):
        raise PrivateResultError("private result is not canonical JSON")
    validate_private_result(payload)
    return payload
