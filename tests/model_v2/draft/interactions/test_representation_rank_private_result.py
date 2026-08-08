from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from lol_kills.v2.draft.interactions import representation_rank_assay as assay
from lol_kills.v2.draft.interactions import representation_rank_private_result as result


def _coverage_diagnostics() -> list[dict]:
    rows = [
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
    development_counts = dict(
        zip(
            (
                row[0]
                for row in assay.ELIGIBLE_GATE_BLOCKS["development"]
            ),
            (128, 230, 421, 202),
        )
    )
    validation_counts = dict(
        zip(
            (
                row[0]
                for row in assay.ELIGIBLE_GATE_BLOCKS["validation"]
            ),
            (515, 569),
        )
    )
    output = []
    for split, month in rows:
        count = (
            development_counts[month]
            if split == "development"
            else validation_counts[month]
            if split == "validation"
            else 300
        )
        output.append(
            {
            "split": split,
            "calendar_month": month,
            "passed": True,
            "maps": count,
            "eligible_maps": count,
            "clusters": count,
            "eligible_clusters": count,
            "month": {
                "calendar_month": month,
                "maps": count,
                "eligible_maps": count,
                "clusters": count,
                "eligible_clusters": count,
            },
            "leagues": [
                {
                    "league": "SYNTHETIC",
                    "maps": count,
                    "eligible_maps": count,
                    "clusters": count,
                    "eligible_clusters": count,
                }
            ],
            "membership_sha256": assay.canonical_sha256([split, month]),
            }
        )
    return output


def _gate_result(months: tuple[str, ...], *, passed: bool = True) -> dict:
    comparators: dict[str, dict] = {}
    for comparator in ("M0", "M8"):
        gates: dict[str, object] = {}
        for metric in ("log_loss", "brier", "calibration"):
            upper = (
                0.0
                if not passed and comparator == "M0" and metric == "log_loss"
                else -0.01
                if metric == "log_loss"
                else 0.0
            )
            decision = assay.metric_gate_decision(
                comparator=comparator,
                metric=metric,
                upper=upper,
            )
            gates[metric] = {
                "upper": upper,
                **decision,
            }
        gates["blocks"] = {
            month: {
                "delta": 0.0,
                "passed": assay.block_gate_decision(0.0),
            }
            for month in months
        }
        comparators[comparator] = gates
    derived = all(
        gate["passed"]
        for comparator in comparators.values()
        for name, gate in comparator.items()
        if name != "blocks"
    ) and all(
        gate["passed"]
        for comparator in comparators.values()
        for gate in comparator["blocks"].values()
    )
    assert derived is passed
    return {"passed": derived, "comparators": comparators}


def _inconclusive_payload() -> dict:
    coverage = _coverage_diagnostics()
    coverage[0]["maps"] = 10_000
    coverage[0]["clusters"] = 10_000
    coverage[0]["month"]["maps"] = 10_000
    coverage[0]["month"]["clusters"] = 10_000
    coverage[0]["leagues"][0]["maps"] = 10_000
    coverage[0]["leagues"][0]["clusters"] = 10_000
    coverage[0]["passed"] = False
    development = result.derive_population_identity(
        split="development",
        ordered_month_blocks=tuple(
            (row["membership_sha256"], row["eligible_maps"])
            for row in coverage
            if row["split"] == "development"
        ),
    )
    validation = result.derive_population_identity(
        split="validation",
        ordered_month_blocks=tuple(
            (row["membership_sha256"], row["eligible_maps"])
            for row in coverage
            if row["split"] == "validation"
        ),
    )
    combined_maps = development["maps"] + validation["maps"]
    unsigned = {
        "schema_id": result.SCHEMA_ID,
        "aggregate_only": True,
        "development_only": True,
        "run_status": "inconclusive",
        "fallback": "M0",
        "reason_code": "coverage_gate_failed",
        "reason_context": {
            "stage": "inner",
            "sequence": None,
            "calendar_month": coverage[0]["calendar_month"],
            "family": None,
            "width": None,
        },
        "selected_model": "M0",
        "selected_width": None,
        "contract_artifact_sha256": "a" * 64,
        "contract_review_core_sha256": "b" * 64,
        "runner_review_permit_raw_sha256": "c" * 64,
        "source_identity_sha256": "d" * 64,
        "runtime_identity_sha256": "e" * 64,
        "feature_domain_sha256": "f" * 64,
        "target_domain_sha256": "0" * 64,
        "fit_availability_domain_sha256": "1" * 64,
        "population": {
            "development_maps": development["maps"],
            "development_membership_sha256": development[
                "membership_sha256"
            ],
            "validation_maps": validation["maps"],
            "validation_membership_sha256": validation[
                "membership_sha256"
            ],
            "combined_maps": combined_maps,
            "combined_membership_sha256": assay.canonical_sha256(
                {
                    "development_membership_sha256": development[
                        "membership_sha256"
                    ],
                    "validation_membership_sha256": validation[
                        "membership_sha256"
                    ],
                    "maps": combined_maps,
                }
            ),
        },
        "penalties": {"lambda_ally": None, "lambda_enemy": None},
        "fit_counts": {"actual": 0, "planned_slots": 56},
        "stage_status": {
            "inner": {
                "status": "failed",
                "reason_code": "coverage_gate_failed",
            },
            "development": {"status": "not_run", "reason_code": None},
            "validation": {"status": "not_run", "reason_code": None},
        },
        "coverage_diagnostics": coverage,
        "development_diagnostics": result.build_development_diagnostics(
            None,
            status="not_run",
            selected_width=None,
        ),
        "validation_diagnostics": result.build_validation_diagnostics(
            None,
            status="not_run",
            selected_width=None,
        ),
        "fit_plan": result.empty_fit_plan(),
        "final_target_loaded": False,
        "publication_authority": False,
        "production_authority": False,
        "reliability_authority": False,
        "promotion_authority": False,
        "sota_claim_authority": False,
    }
    return result.with_artifact_sha256(unsigned)


def _accepted_payload(*, locked_width: int = 2) -> dict:
    payload = _inconclusive_payload()
    development_months = tuple(
        row[0] for row in assay.ELIGIBLE_GATE_BLOCKS["development"]
    )
    validation_months = tuple(
        row[0] for row in assay.ELIGIBLE_GATE_BLOCKS["validation"]
    )
    plan = result.empty_fit_plan()
    coverage_by_block = {
        (row["split"], row["calendar_month"]): row
        for row in payload["coverage_diagnostics"]
    }
    for row in payload["coverage_diagnostics"]:
        row["maps"] = row["eligible_maps"]
        row["clusters"] = row["eligible_clusters"]
        row["month"]["maps"] = row["month"]["eligible_maps"]
        row["month"]["clusters"] = row["month"]["eligible_clusters"]
        for league in row["leagues"]:
            league["maps"] = league["eligible_maps"]
            league["clusters"] = league["eligible_clusters"]
        row["passed"] = True
    for row in plan:
        coverage = coverage_by_block[(row["split"], row["calendar_month"])]
        if row["stage"] != "inner":
            row["lambda_ally"] = 1.0
            row["lambda_enemy"] = 1.0
        if row["fit_role"] == "locked_width":
            row["width"] = locked_width
        row.update(
            {
                "execution_status": "passed",
                "maps": coverage["eligible_maps"],
                "clusters": coverage["eligible_clusters"],
                "membership_sha256": coverage["membership_sha256"],
                "objective": 0.5,
                "max_gradient": 1e-7,
                "converged_starts": 3,
                "stability_rms": 1e-4,
                "log_loss_total": 5.0,
                "brier_total": 2.5,
            }
        )
    if locked_width == 8:
        for alias_index in (53, 55):
            source = plan[alias_index - 1]
            alias = plan[alias_index]
            alias["execution_status"] = "aliased"
            for key in result.ALIAS_EQUAL_FIELDS:
                alias[key] = source[key]
    payload.update(
        {
            "run_status": "accepted",
            "fallback": "none",
            "reason_code": None,
            "reason_context": None,
            "selected_model": "latent_candidate",
            "selected_width": locked_width,
            "penalties": {"lambda_ally": 1.0, "lambda_enemy": 1.0},
            "fit_counts": {
                "actual": 54 if locked_width == 8 else 56,
                "planned_slots": 56,
            },
            "stage_status": {
                stage: {"status": "passed", "reason_code": None}
                for stage in ("inner", "development", "validation")
            },
            "development_diagnostics": result.build_development_diagnostics(
                {
                    "M8_prerequisite": _gate_result(development_months),
                    "widths": {
                        str(width): _gate_result(
                            development_months,
                            passed=width == locked_width,
                        )
                        for width in assay.WIDTHS[
                            : assay.WIDTHS.index(locked_width) + 1
                        ]
                    },
                    "locked_width": locked_width,
                },
                status="passed",
                selected_width=locked_width,
            ),
            "validation_diagnostics": result.build_validation_diagnostics(
                {
                    "passed": True,
                    "locked_width": locked_width,
                    "M8": _gate_result(validation_months),
                    "locked": _gate_result(validation_months),
                },
                status="passed",
                selected_width=locked_width,
            ),
            "fit_plan": plan,
        }
    )
    return result.with_artifact_sha256(payload)


def _clear_summaries(row: dict) -> None:
    for key in result.SUMMARY_FIELDS:
        row[key] = None


def _inconclusive_cell_payload(
    *,
    sequence: int,
    reason_code: str = "fit_unavailable",
    locked_width: int = 2,
) -> dict:
    payload = deepcopy(_accepted_payload(locked_width=locked_width))
    failed_row = payload["fit_plan"][sequence - 1]
    failed_stage = failed_row["stage"]
    payload.update(
        {
            "run_status": "inconclusive",
            "fallback": "M0",
            "reason_code": reason_code,
            "reason_context": {
                "stage": failed_stage,
                "sequence": sequence,
                "calendar_month": failed_row["calendar_month"],
                "family": failed_row["family"],
                "width": failed_row["width"],
            },
            "selected_model": "M0",
            "selected_width": None,
        }
    )
    failed_stage_index = result.STAGES.index(failed_stage)
    payload["stage_status"] = {
        stage: {
            "status": (
                "passed"
                if index < failed_stage_index
                else "failed"
                if index == failed_stage_index
                else "not_run"
            ),
            "reason_code": reason_code if index == failed_stage_index else None,
        }
        for index, stage in enumerate(result.STAGES)
    }
    if failed_stage == "inner":
        for coverage in payload["coverage_diagnostics"][:6]:
            for container in (
                coverage,
                coverage["month"],
                coverage["leagues"][0],
            ):
                container["maps"] = 200
                container["eligible_maps"] = 200
                container["clusters"] = 200
                container["eligible_clusters"] = 200
        for row in payload["fit_plan"][:36]:
            row["maps"] = 200
            row["clusters"] = 200
        payload["penalties"] = {
            "lambda_ally": None,
            "lambda_enemy": None,
        }
        payload["development_diagnostics"] = (
            result.build_development_diagnostics(
                None,
                status="not_run",
                selected_width=None,
            )
        )
        payload["validation_diagnostics"] = result.build_validation_diagnostics(
            None,
            status="not_run",
            selected_width=None,
        )
    elif failed_stage == "development":
        payload["development_diagnostics"] = (
            result.build_development_diagnostics(
                {
                    "M8_prerequisite": _gate_result(development_months := tuple(
                        row[0]
                        for row in assay.ELIGIBLE_GATE_BLOCKS[
                            "development"
                        ]
                    )),
                    "widths": {
                        str(width): _gate_result(
                            development_months,
                            passed=False,
                        )
                        for width in assay.WIDTHS
                    },
                    "locked_width": None,
                },
                status="failed",
                selected_width=None,
            )
        )
        payload["validation_diagnostics"] = result.build_validation_diagnostics(
            None,
            status="not_run",
            selected_width=None,
        )
    else:
        validation_months = tuple(
            row[0] for row in assay.ELIGIBLE_GATE_BLOCKS["validation"]
        )
        payload["validation_diagnostics"] = result.build_validation_diagnostics(
            {
                "passed": False,
                "locked_width": locked_width,
                "M8": _gate_result(validation_months),
                "locked": _gate_result(
                    validation_months,
                    passed=False,
                ),
            },
            status="failed",
            selected_width=locked_width,
        )
    for row in payload["fit_plan"][sequence:]:
        row["execution_status"] = "not_run"
        _clear_summaries(row)
    failed_row["execution_status"] = "failed"
    for key in result.SCORING_SUMMARY_FIELDS:
        failed_row[key] = None
    if reason_code == "fit_unavailable":
        for key in result.OPTIMIZATION_SUMMARY_FIELDS:
            failed_row[key] = None
    payload["fit_counts"]["actual"] = sum(
        row["execution_status"] == "passed"
        for row in payload["fit_plan"]
    )
    return result.with_artifact_sha256(payload)


def _gate_failure_payload(
    reason_code: str,
    *,
    locked_width: int = 2,
) -> dict:
    payload = deepcopy(_accepted_payload(locked_width=locked_width))
    failed_stage, boundary = {
        "penalty_selection_failed": ("inner", 36),
        "development_gate_failed": ("development", 52),
        "validation_gate_failed": ("validation", 56),
    }[reason_code]
    failed_stage_index = result.STAGES.index(failed_stage)
    payload.update(
        {
            "run_status": "inconclusive",
            "fallback": "M0",
            "reason_code": reason_code,
            "reason_context": {
                "stage": failed_stage,
                "sequence": None,
                "calendar_month": None,
                "family": None,
                "width": None,
            },
            "selected_model": "M0",
            "selected_width": None,
            "stage_status": {
                stage: {
                    "status": (
                        "passed"
                        if index < failed_stage_index
                        else "failed"
                        if index == failed_stage_index
                        else "not_run"
                    ),
                    "reason_code": (
                        reason_code if index == failed_stage_index else None
                    ),
                }
                for index, stage in enumerate(result.STAGES)
            },
        }
    )
    for row in payload["fit_plan"][boundary:]:
        row["execution_status"] = "not_run"
        _clear_summaries(row)
    if failed_stage == "inner":
        payload["development_diagnostics"] = (
            result.build_development_diagnostics(
                None,
                status="not_run",
                selected_width=None,
            )
        )
        payload["validation_diagnostics"] = result.build_validation_diagnostics(
            None,
            status="not_run",
            selected_width=None,
        )
    elif failed_stage == "development":
        payload["development_diagnostics"] = (
            result.build_development_diagnostics(
                None,
                status="failed",
                selected_width=None,
            )
        )
        payload["validation_diagnostics"] = result.build_validation_diagnostics(
            None,
            status="not_run",
            selected_width=None,
        )
    else:
        payload["validation_diagnostics"] = result.build_validation_diagnostics(
            None,
            status="failed",
            selected_width=None,
        )
    payload["fit_counts"]["actual"] = sum(
        row["execution_status"] == "passed"
        for row in payload["fit_plan"]
    )
    return result.with_artifact_sha256(payload)


def test_frozen_fit_plan_has_exact_56_slot_order() -> None:
    rows = result.empty_fit_plan()
    assert len(rows) == 56
    assert [row["sequence"] for row in rows] == list(range(1, 57))
    assert [row["stage"] for row in rows[:36]] == ["inner"] * 36
    assert [row["stage"] for row in rows[36:52]] == ["development"] * 16
    assert [row["stage"] for row in rows[52:]] == ["validation"] * 4
    assert [
        (row["family"], row["lambda_ally"], row["lambda_enemy"], row["width"])
        for row in rows[:18:6]
    ] == [
        ("ally", 0.01, 1.0, 8),
        ("ally", 0.1, 1.0, 8),
        ("ally", 1.0, 1.0, 8),
    ]
    assert [
        (row["fit_role"], row["calendar_month"]) for row in rows[52:]
    ] == [
        ("locked_width", "2026-04"),
        ("M8_reference", "2026-04"),
        ("locked_width", "2026-05"),
        ("M8_reference", "2026-05"),
    ]


def test_aggregate_result_round_trip_tamper_and_byte_determinism(
    tmp_path: Path,
) -> None:
    payload = _inconclusive_payload()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    result.write_private_result(payload, path=first)
    result.write_private_result(payload, path=second)
    assert first.read_bytes() == second.read_bytes() == assay.canonical_bytes(
        payload
    )
    assert result.verify_private_result(path=first) == payload
    changed = json.loads(first.read_bytes())
    changed["fit_counts"]["actual"] = 1
    first.write_bytes(assay.canonical_bytes(changed))
    with pytest.raises(result.PrivateResultError, match="artifact hash"):
        result.verify_private_result(path=first)


def test_atomic_writer_io_failure_leaves_no_terminal_file(tmp_path: Path) -> None:
    payload = _inconclusive_payload()
    path = tmp_path / "terminal.json"
    temporary = tmp_path / ".terminal.json.tmp"
    temporary.write_bytes(b"occupied")
    with pytest.raises(result.PrivateResultError, match="temporary"):
        result.write_private_result(payload, path=path)
    assert not path.exists()
    assert temporary.read_bytes() == b"occupied"


@pytest.mark.parametrize(
    "mutation",
    (
        "game_id",
        "prediction",
        "column",
        "final",
        "partial_plan",
        "path",
    ),
)
def test_result_rejects_row_level_arbitrary_or_partial_material(
    mutation: str,
) -> None:
    payload = _inconclusive_payload()
    if mutation == "game_id":
        payload["development_diagnostics"] = {"game_id": "forbidden"}
    elif mutation == "prediction":
        payload["validation_diagnostics"] = {"predictions": [0.5]}
    elif mutation == "column":
        payload["unexpected"] = True
    elif mutation == "final":
        payload["final_target_loaded"] = True
    elif mutation == "partial_plan":
        payload["fit_plan"] = payload["fit_plan"][:-1]
    else:
        payload["development_diagnostics"] = {
            "parquet_locator": "/tmp/arbitrary.parquet"
        }
    payload = result.with_artifact_sha256(payload)
    with pytest.raises(result.PrivateResultError):
        result.validate_private_result(payload)


@pytest.mark.parametrize(
    "key,value",
    (
        ("row_ids", ["g1"]),
        ("y_blue_win", [1]),
        ("probabilities", [0.5]),
        ("final_rows", [{"sealed": True}]),
        ("output_file", "/tmp/leak.json"),
    ),
)
def test_rehashed_confirmed_nested_exploits_are_rejected(
    key: str,
    value: object,
) -> None:
    payload = _accepted_payload()
    payload["development_diagnostics"]["candidates"][0]["comparators"][0][
        "metrics"
    ][0][key] = value
    payload = result.with_artifact_sha256(payload)
    with pytest.raises(result.PrivateResultError):
        result.validate_private_result(payload)


@pytest.mark.parametrize(
    "path,key,value",
    (
        ((), "unknown", True),
        (("population",), "row_id", "g1"),
        (("penalties",), "target", 1),
        (("fit_counts",), "prediction", 0.5),
        (("stage_status",), "final", True),
        (("stage_status", "inner"), "locator", "/tmp/private"),
        (("coverage_diagnostics", 0), "game_ids", ["g1"]),
        (("development_diagnostics",), "rows", []),
        (("development_diagnostics", "bootstrap"), "path", "/tmp/private"),
        (
            ("development_diagnostics", "candidates", 0),
            "row_ids",
            ["g1"],
        ),
        (
            (
                "development_diagnostics",
                "candidates",
                0,
                "comparators",
                0,
            ),
            "game_id",
            "g1",
        ),
        (
            (
                "development_diagnostics",
                "candidates",
                0,
                "comparators",
                0,
                "metrics",
                0,
            ),
            "probability",
            0.5,
        ),
        (
            (
                "development_diagnostics",
                "candidates",
                0,
                "comparators",
                0,
                "blocks",
                0,
            ),
            "final_rows",
            [],
        ),
        (("validation_diagnostics",), "output_file", "/tmp/result"),
        (("validation_diagnostics", "bootstrap"), "unknown", 1),
        (
            ("validation_diagnostics", "candidates", 0),
            "y_blue_win",
            [1],
        ),
        (
            (
                "validation_diagnostics",
                "candidates",
                0,
                "comparators",
                0,
            ),
            "target",
            1,
        ),
        (
            (
                "validation_diagnostics",
                "candidates",
                0,
                "comparators",
                0,
                "metrics",
                0,
            ),
            "predictions",
            [0.5],
        ),
        (
            (
                "validation_diagnostics",
                "candidates",
                0,
                "comparators",
                0,
                "blocks",
                0,
            ),
            "file",
            "/tmp/result",
        ),
        (("fit_plan", 0), "source_mapping", {"locator": "/tmp/source"}),
    ),
)
def test_rehashed_unknown_key_is_rejected_at_every_recursive_extension(
    path: tuple[object, ...],
    key: str,
    value: object,
) -> None:
    payload = deepcopy(_accepted_payload())
    location: object = payload
    for part in path:
        location = location[part]  # type: ignore[index]
    location[key] = value  # type: ignore[index]
    payload = result.with_artifact_sha256(payload)
    with pytest.raises(result.PrivateResultError):
        result.validate_private_result(payload)


@pytest.mark.parametrize(
    "mutation",
    (
        "coverage_boolean_count",
        "development_boolean_width",
        "validation_string_width",
        "penalty_text",
        "nonhex_identity",
        "metric_string",
        "block_nonfinite",
        "fit_nonfinite",
        "coverage_excess",
        "development_excess",
        "validation_excess",
        "fit_plan_excess",
    ),
)
def test_rehashed_type_cardinality_and_finite_mutations_reject(
    mutation: str,
) -> None:
    payload = deepcopy(_accepted_payload())
    if mutation == "coverage_boolean_count":
        payload["coverage_diagnostics"][0]["maps"] = True
    elif mutation == "development_boolean_width":
        payload["development_diagnostics"]["selected_width"] = True
    elif mutation == "validation_string_width":
        payload["validation_diagnostics"]["selected_width"] = "2"
    elif mutation == "penalty_text":
        payload["penalties"]["lambda_ally"] = "/tmp/private-target.parquet"
    elif mutation == "nonhex_identity":
        payload["source_identity_sha256"] = "private-game-id".ljust(64, "x")
    elif mutation == "metric_string":
        payload["development_diagnostics"]["candidates"][0]["comparators"][0][
            "metrics"
        ][0]["upper"] = "0.0"
    elif mutation == "block_nonfinite":
        payload["validation_diagnostics"]["candidates"][0]["comparators"][0][
            "blocks"
        ][0]["delta"] = float("nan")
    elif mutation == "fit_nonfinite":
        payload["fit_plan"][0]["objective"] = float("inf")
    elif mutation == "coverage_excess":
        payload["coverage_diagnostics"].append(
            deepcopy(payload["coverage_diagnostics"][-1])
        )
    elif mutation == "development_excess":
        payload["development_diagnostics"]["candidates"].extend(
            deepcopy(payload["development_diagnostics"]["candidates"][:3])
        )
    elif mutation == "validation_excess":
        payload["validation_diagnostics"]["candidates"].append(
            deepcopy(payload["validation_diagnostics"]["candidates"][0])
        )
    else:
        payload["fit_plan"].append(deepcopy(payload["fit_plan"][-1]))
    if mutation in {"block_nonfinite", "fit_nonfinite"}:
        with pytest.raises(result.PrivateResultError, match="canonical JSON"):
            result.with_artifact_sha256(payload)
        return
    payload = result.with_artifact_sha256(payload)
    with pytest.raises(result.PrivateResultError):
        result.validate_private_result(payload)


@pytest.mark.parametrize(
    "location",
    ("terminal", "stage"),
)
def test_rehashed_free_text_reason_or_identifier_leakage_is_rejected(
    location: str,
) -> None:
    payload = _inconclusive_payload()
    leaked = "fit failed for game_id=private-123 at /tmp/secret.parquet"
    if location == "terminal":
        payload["reason_code"] = leaked
    else:
        payload["stage_status"]["inner"]["reason_code"] = leaked
    payload = result.with_artifact_sha256(payload)
    with pytest.raises(result.PrivateResultError):
        result.validate_private_result(payload)


def test_rehashed_reason_context_extension_and_raw_source_mapping_reject() -> None:
    payload = _inconclusive_payload()
    payload["reason_context"]["output_file"] = "/tmp/private-result.json"
    payload = result.with_artifact_sha256(payload)
    with pytest.raises(result.PrivateResultError):
        result.validate_private_result(payload)


@pytest.mark.parametrize("field", result.SUMMARY_FIELDS)
def test_rehashed_accepted_passed_slot_requires_every_summary(
    field: str,
) -> None:
    payload = _accepted_payload()
    payload["fit_plan"][0][field] = None
    payload = result.with_artifact_sha256(payload)
    with pytest.raises(result.PrivateResultError):
        result.validate_private_result(payload)


def test_rehashed_inconclusive_passed_and_alias_slots_cannot_be_empty() -> None:
    payload = _inconclusive_cell_payload(sequence=37)
    _clear_summaries(payload["fit_plan"][0])
    payload = result.with_artifact_sha256(payload)
    with pytest.raises(result.PrivateResultError):
        result.validate_private_result(payload)

    payload = _gate_failure_payload(
        "validation_gate_failed",
        locked_width=8,
    )
    _clear_summaries(payload["fit_plan"][53])
    payload = result.with_artifact_sha256(payload)
    with pytest.raises(result.PrivateResultError):
        result.validate_private_result(payload)


@pytest.mark.parametrize(
    "mutation",
    (
        "empty_alias",
        "synthetic_alias",
        "wrong_source_alias",
        "source_not_passed",
        "wrong_penalty",
        "alias_at_locked_slot",
        "alias_when_width_not_8",
    ),
)
def test_rehashed_alias_requires_exact_prior_locked_width_source(
    mutation: str,
) -> None:
    payload = _accepted_payload(locked_width=8)
    alias = payload["fit_plan"][53]
    source = payload["fit_plan"][52]
    if mutation == "empty_alias":
        _clear_summaries(alias)
    elif mutation == "synthetic_alias":
        alias["objective"] += 0.01
    elif mutation == "wrong_source_alias":
        alias["membership_sha256"] = payload["fit_plan"][54][
            "membership_sha256"
        ]
    elif mutation == "source_not_passed":
        source["execution_status"] = "not_run"
        _clear_summaries(source)
        payload["fit_counts"]["actual"] -= 1
    elif mutation == "wrong_penalty":
        alias["lambda_ally"] = 0.1
    elif mutation == "alias_at_locked_slot":
        source["execution_status"] = "aliased"
        payload["fit_counts"]["actual"] -= 1
    else:
        payload = _accepted_payload(locked_width=2)
        source = payload["fit_plan"][52]
        alias = payload["fit_plan"][53]
        alias["execution_status"] = "aliased"
        for key in result.SUMMARY_FIELDS:
            alias[key] = source[key]
        payload["fit_counts"]["actual"] -= 1
    payload = result.with_artifact_sha256(payload)
    with pytest.raises(result.PrivateResultError):
        result.validate_private_result(payload)


@pytest.mark.parametrize(
    "mutation",
    (
        "multiple_failures",
        "passed_after_failure",
        "not_run_before_failure",
        "failed_after_not_run",
        "failed_summary_mixture",
        "scoring_missing_optimization",
        "scoring_has_partial_loss",
    ),
)
def test_rehashed_cell_failure_requires_one_exact_completed_prefix(
    mutation: str,
) -> None:
    if mutation.startswith("scoring"):
        payload = _inconclusive_cell_payload(
            sequence=37,
            reason_code="scoring_unavailable",
        )
    else:
        payload = _inconclusive_cell_payload(sequence=37)
    accepted = _accepted_payload()
    if mutation == "multiple_failures":
        row = payload["fit_plan"][37]
        row["execution_status"] = "failed"
        for key in result.POPULATION_SUMMARY_FIELDS:
            row[key] = accepted["fit_plan"][37][key]
    elif mutation == "passed_after_failure":
        payload["fit_plan"][37] = deepcopy(accepted["fit_plan"][37])
        payload["fit_counts"]["actual"] += 1
    elif mutation in {"not_run_before_failure", "failed_after_not_run"}:
        row = payload["fit_plan"][35]
        row["execution_status"] = "not_run"
        _clear_summaries(row)
        payload["fit_counts"]["actual"] -= 1
    elif mutation == "failed_summary_mixture":
        payload["fit_plan"][36]["objective"] = 0.5
    elif mutation == "scoring_missing_optimization":
        payload["fit_plan"][36]["objective"] = None
    else:
        payload["fit_plan"][36]["log_loss_total"] = 1.0
    payload = result.with_artifact_sha256(payload)
    with pytest.raises(result.PrivateResultError):
        result.validate_private_result(payload)


@pytest.mark.parametrize(
    "mutation",
    (
        "terminal_reason_mismatch",
        "stage_reason_mismatch",
        "context_stage_mismatch",
        "multiple_failed_stages",
        "prior_stage_not_run",
        "later_stage_passed",
        "count_fudged",
        "context_sequence_wrong",
        "context_row_identity_wrong",
    ),
)
def test_rehashed_inconclusive_stage_reason_and_count_semantics_reject(
    mutation: str,
) -> None:
    payload = _inconclusive_cell_payload(sequence=53)
    if mutation == "terminal_reason_mismatch":
        payload["reason_code"] = "scoring_unavailable"
    elif mutation == "stage_reason_mismatch":
        payload["stage_status"]["validation"]["reason_code"] = (
            "scoring_unavailable"
        )
    elif mutation == "context_stage_mismatch":
        payload["reason_context"]["stage"] = "development"
    elif mutation == "multiple_failed_stages":
        payload["stage_status"]["development"] = {
            "status": "failed",
            "reason_code": "development_gate_failed",
        }
        payload["development_diagnostics"] = (
            result.build_development_diagnostics(
                None,
                status="failed",
                selected_width=None,
            )
        )
    elif mutation == "prior_stage_not_run":
        payload["stage_status"]["development"] = {
            "status": "not_run",
            "reason_code": None,
        }
        payload["development_diagnostics"] = (
            result.build_development_diagnostics(
                None,
                status="not_run",
                selected_width=None,
            )
        )
    elif mutation == "later_stage_passed":
        payload = _inconclusive_cell_payload(sequence=37)
        payload["stage_status"]["validation"] = {
            "status": "passed",
            "reason_code": None,
        }
    elif mutation == "count_fudged":
        payload["fit_counts"]["actual"] += 1
    elif mutation == "context_sequence_wrong":
        payload["reason_context"]["sequence"] = 54
    else:
        payload["reason_context"]["calendar_month"] = "2026-05"
    payload = result.with_artifact_sha256(payload)
    with pytest.raises(result.PrivateResultError):
        result.validate_private_result(payload)


@pytest.mark.parametrize(
    "mutation",
    (
        "failed_fit_row",
        "passed_other_stage",
        "wrong_context_month",
        "count_fudged",
        "not_first_failed_block",
    ),
)
def test_rehashed_coverage_preflight_has_one_exact_zero_fit_state(
    mutation: str,
) -> None:
    payload = _inconclusive_payload()
    if mutation == "failed_fit_row":
        row = payload["fit_plan"][0]
        row["execution_status"] = "failed"
        coverage = payload["coverage_diagnostics"][0]
        row["maps"] = coverage["eligible_maps"]
        row["clusters"] = coverage["eligible_clusters"]
        row["membership_sha256"] = coverage["membership_sha256"]
    elif mutation == "passed_other_stage":
        payload["stage_status"]["development"] = {
            "status": "passed",
            "reason_code": None,
        }
        payload["development_diagnostics"]["status"] = "passed"
    elif mutation == "wrong_context_month":
        payload["reason_context"]["calendar_month"] = assay.INNER_MONTHS[1]
    elif mutation == "count_fudged":
        payload["fit_counts"]["actual"] = 1
    else:
        payload["coverage_diagnostics"][1]["passed"] = False
        payload["coverage_diagnostics"][0]["passed"] = True
    payload = result.with_artifact_sha256(payload)
    with pytest.raises(result.PrivateResultError):
        result.validate_private_result(payload)


@pytest.mark.parametrize("failed", (False, True))
def test_rehashed_fit_population_must_equal_coverage_block(
    failed: bool,
) -> None:
    payload = (
        _inconclusive_cell_payload(sequence=37)
        if failed
        else _accepted_payload()
    )
    sequence = 37 if failed else 1
    payload["fit_plan"][sequence - 1]["maps"] -= 1
    payload = result.with_artifact_sha256(payload)
    with pytest.raises(result.PrivateResultError):
        result.validate_private_result(payload)


@pytest.mark.parametrize(
    "reason_code",
    (
        "penalty_selection_failed",
        "development_gate_failed",
        "validation_gate_failed",
    ),
)
def test_rehashed_gate_failure_cannot_invent_a_failed_fit(
    reason_code: str,
) -> None:
    payload = _gate_failure_payload(reason_code)
    boundary = {
        "penalty_selection_failed": 36,
        "development_gate_failed": 52,
        "validation_gate_failed": 56,
    }[reason_code]
    row = payload["fit_plan"][boundary - 1]
    row["execution_status"] = "failed"
    payload["fit_counts"]["actual"] -= 1
    payload = result.with_artifact_sha256(payload)
    with pytest.raises(result.PrivateResultError):
        result.validate_private_result(payload)


def test_rehashed_later_fit_penalties_bind_to_selected_penalties() -> None:
    payload = _accepted_payload()
    payload["fit_plan"][36]["lambda_ally"] = 0.1
    payload = result.with_artifact_sha256(payload)
    with pytest.raises(result.PrivateResultError):
        result.validate_private_result(payload)


def test_rehashed_development_rejects_passing_smaller_width() -> None:
    payload = _accepted_payload(locked_width=2)
    width_one = payload["development_diagnostics"]["candidates"][1]
    metric = width_one["comparators"][0]["metrics"][0]
    metric.update(
        {
            "upper": -0.01,
            **assay.metric_gate_decision(
                comparator="M0",
                metric="log_loss",
                upper=-0.01,
            ),
        }
    )
    width_one["passed"] = True
    payload = result.with_artifact_sha256(payload)
    with pytest.raises(
        result.PrivateResultError,
        match="smallest passing",
    ):
        result.validate_private_result(payload)


@pytest.mark.parametrize(
    ("upper", "stored_passed"),
    (
        (1.0, True),
        (0.0, True),
    ),
)
def test_rehashed_metric_decision_rejects_arithmetic_or_strict_contradiction(
    upper: float,
    stored_passed: bool,
) -> None:
    payload = _accepted_payload()
    metric = payload["development_diagnostics"]["candidates"][0][
        "comparators"
    ][0]["metrics"][0]
    metric["upper"] = upper
    metric["passed"] = stored_passed
    payload = result.with_artifact_sha256(payload)
    with pytest.raises(
        result.PrivateResultError,
        match="frozen decision",
    ):
        result.validate_private_result(payload)


def test_rehashed_accepted_validation_requires_both_candidates_to_pass() -> None:
    payload = _accepted_payload()
    locked = payload["validation_diagnostics"]["candidates"][1]
    metric = locked["comparators"][0]["metrics"][0]
    metric.update(
        {
            "upper": 0.0,
            **assay.metric_gate_decision(
                comparator="M0",
                metric="log_loss",
                upper=0.0,
            ),
        }
    )
    locked["passed"] = False
    payload = result.with_artifact_sha256(payload)
    with pytest.raises(
        result.PrivateResultError,
        match="validation accepted",
    ):
        result.validate_private_result(payload)


@pytest.mark.parametrize("starts", (2, 3))
def test_completed_fit_accepts_exact_optimizer_boundaries(starts: int) -> None:
    payload = _accepted_payload()
    payload["fit_plan"][0].update(
        {
            "converged_starts": starts,
            "max_gradient": assay.GRADIENT_TOLERANCE,
            "stability_rms": assay.STABILITY_RMS_TOLERANCE,
        }
    )
    payload = result.with_artifact_sha256(payload)
    result.validate_private_result(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("converged_starts", 1),
        ("converged_starts", 4),
        ("max_gradient", assay.GRADIENT_TOLERANCE + 1e-12),
        ("stability_rms", assay.STABILITY_RMS_TOLERANCE + 1e-12),
    ),
)
def test_rehashed_completed_fit_rejects_optimizer_gate_violation(
    field: str,
    value: float,
) -> None:
    payload = _accepted_payload()
    payload["fit_plan"][0][field] = value
    payload = result.with_artifact_sha256(payload)
    with pytest.raises(result.PrivateResultError, match="completed fit"):
        result.validate_private_result(payload)


def test_penalty_selection_is_recomputed_from_all_inner_rows() -> None:
    payload = _accepted_payload()
    for row in payload["fit_plan"][:18]:
        if row["lambda_ally"] == 0.1:
            row["log_loss_total"] = 0.1
    payload["penalties"]["lambda_ally"] = 0.1
    for row in payload["fit_plan"][36:]:
        row["lambda_ally"] = 0.1
    payload = result.with_artifact_sha256(payload)
    result.validate_private_result(payload)

    contradictory = deepcopy(payload)
    contradictory["penalties"]["lambda_ally"] = 1.0
    for row in contradictory["fit_plan"][36:]:
        row["lambda_ally"] = 1.0
    contradictory = result.with_artifact_sha256(contradictory)
    with pytest.raises(
        result.PrivateResultError,
        match="inner ledger",
    ):
        result.validate_private_result(contradictory)


def test_penalty_tie_uses_frozen_largest_penalty_rule() -> None:
    payload = _accepted_payload()
    assert result._recompute_inner_penalties(payload["fit_plan"]) == {
        "lambda_ally": 1.0,
        "lambda_enemy": 1.0,
    }
    result.validate_private_result(payload)


def test_rehashed_coverage_cannot_mark_300_of_10000_maps_passing() -> None:
    payload = _accepted_payload()
    coverage = payload["coverage_diagnostics"][0]
    for container in (
        coverage,
        coverage["month"],
        coverage["leagues"][0],
    ):
        container["maps"] = 10_000
    coverage["passed"] = True
    payload = result.with_artifact_sha256(payload)
    with pytest.raises(
        result.PrivateResultError,
        match="coverage pass",
    ):
        result.validate_private_result(payload)


def test_rehashed_coverage_recomputes_conditional_league_failure() -> None:
    payload = _accepted_payload()
    coverage = payload["coverage_diagnostics"][0]
    coverage.update(
        {
            "maps": 300,
            "eligible_maps": 240,
            "clusters": 250,
            "eligible_clusters": 200,
        }
    )
    coverage["month"].update(
        {
            "maps": 300,
            "eligible_maps": 240,
            "clusters": 250,
            "eligible_clusters": 200,
        }
    )
    coverage["leagues"] = [
        {
            "league": "A",
            "maps": 40,
            "eligible_maps": 29,
            "clusters": 10,
            "eligible_clusters": 8,
        },
        {
            "league": "B",
            "maps": 260,
            "eligible_maps": 211,
            "clusters": 240,
            "eligible_clusters": 192,
        },
    ]
    coverage["passed"] = True
    payload = result.with_artifact_sha256(payload)
    with pytest.raises(
        result.PrivateResultError,
        match="coverage pass",
    ):
        result.validate_private_result(payload)


def test_penalty_failure_requires_actual_recomputed_selection_failure() -> None:
    payload = _gate_failure_payload("penalty_selection_failed")
    for coverage in payload["coverage_diagnostics"][:6]:
        for container in (
            coverage,
            coverage["month"],
            coverage["leagues"][0],
        ):
            container.update(
                {
                    "maps": 300,
                    "eligible_maps": 300,
                    "clusters": 300,
                    "eligible_clusters": 300,
                }
            )
    for row in payload["fit_plan"][:36]:
        row["maps"] = 300
        row["clusters"] = 300
    payload = result.with_artifact_sha256(payload)
    with pytest.raises(
        result.PrivateResultError,
        match="first recomputed failure",
    ):
        result.validate_private_result(payload)


@pytest.mark.parametrize(
    "field",
    (
        "development_maps",
        "development_membership_sha256",
        "validation_maps",
        "validation_membership_sha256",
        "combined_maps",
        "combined_membership_sha256",
    ),
)
def test_rehashed_population_field_binds_to_ordered_coverage(
    field: str,
) -> None:
    payload = _accepted_payload()
    if field.endswith("_maps"):
        payload["population"][field] += 1
    else:
        payload["population"][field] = assay.canonical_sha256(
            ["independent-replacement", field]
        )
    payload = result.with_artifact_sha256(payload)
    with pytest.raises(result.PrivateResultError):
        result.validate_private_result(payload)


@pytest.mark.parametrize("mutation", ("order", "membership"))
def test_rehashed_population_rejects_coverage_order_or_membership_drift(
    mutation: str,
) -> None:
    payload = _accepted_payload()
    if mutation == "order":
        payload["coverage_diagnostics"][6:8] = reversed(
            payload["coverage_diagnostics"][6:8]
        )
    else:
        payload["coverage_diagnostics"][6]["membership_sha256"] = (
            assay.canonical_sha256(["replacement-development-membership"])
        )
    payload = result.with_artifact_sha256(payload)
    with pytest.raises(result.PrivateResultError):
        result.validate_private_result(payload)

    payload = _accepted_payload()
    payload["source_identity"] = {
        "raw_source_mapping": {
            "locator": "/tmp/private-target.parquet",
            "columns": ["game_id", "y_blue_win"],
        }
    }
    payload = result.with_artifact_sha256(payload)
    with pytest.raises(result.PrivateResultError):
        result.validate_private_result(payload)
