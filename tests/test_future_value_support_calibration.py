from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone

import pytest

from lol_kills.research.future_value_uncertainty import (
    FutureValueUncertaintyError,
    apply_strict_prior_support_calibration,
    build_strict_prior_support_calibration,
    verify_support_calibration_artifact,
)


SOURCE = {
    "source_as_of": "2026-08-20T00:00:00Z",
    "source_game_count": 60,
    "source_identity_sha256": "a" * 64,
}


def _folds(rows_per_fold: int = 20) -> list[dict[str, object]]:
    folds: list[dict[str, object]] = []
    for fold_number in range(3):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=30 * fold_number)
        rows: list[dict[str, object]] = []
        for row_number in range(rows_per_fold):
            date = start + timedelta(days=row_number)
            rows.append(
                {
                    "game_id": f"g-{fold_number}-{row_number}",
                    "series_id": f"s-{fold_number}-{row_number}",
                    "date": date.isoformat().replace("+00:00", "Z"),
                    "minimum_effective_support": float(row_number % 10),
                    "prediction_probability": 0.25 + 0.5 * ((row_number % 4) / 3.0),
                    "target": float((row_number + fold_number) % 2),
                }
            )
        folds.append(
            {
                "fold": fold_number + 1,
                "train_end": (start - timedelta(days=1)).isoformat().replace("+00:00", "Z"),
                "validation_start": start.isoformat().replace("+00:00", "Z"),
                "validation_end": (start + timedelta(days=rows_per_fold - 1)).isoformat().replace(
                    "+00:00", "Z"
                ),
                "rows": rows,
            }
        )
    return folds


def _artifact(**kwargs: object) -> dict[str, object]:
    options = {
        "minimum_training_rows": 10,
        "minimum_bin_rows": 2,
        "minimum_bins": 2,
        "maximum_bins": 5,
    }
    options.update(kwargs)
    return build_strict_prior_support_calibration(
        _folds(),
        source_receipt=SOURCE,
        variant="future_player_form",
        **options,
    )


def test_support_calibration_is_strict_prior_monotonic_and_receipted() -> None:
    artifact = _artifact()

    assert artifact["status"] == "research_only"
    assert artifact["coverage"]["complete_enough"] is True  # type: ignore[index]
    assert artifact["folds"][0]["status"] == "blocked"  # type: ignore[index]
    assert "calibration_prior_validation_folds_missing" in artifact["blockers"]  # type: ignore[operator]
    assert verify_support_calibration_artifact(artifact)

    mapping = artifact["folds"][1]["mapping"]  # type: ignore[index]
    fitted = [float(row["fitted_residual"]) for row in mapping["bins"]]  # type: ignore[index]
    assert all(right <= left + 1e-12 for left, right in zip(fitted, fitted[1:]))
    values = apply_strict_prior_support_calibration(
        artifact,
        [0.0, 4.0, 9.0],
        fold_id=2,
        expected_source_receipt_sha256=artifact["source"]["source_receipt_sha256"],  # type: ignore[index]
        expected_variant="future_player_form",
    )
    assert values.name == "calibrated_support_uncertainty"
    assert values.notna().all()


def test_first_fold_and_insufficient_prior_support_fail_closed() -> None:
    artifact = _artifact(minimum_training_rows=100)
    assert artifact["folds"][0]["status"] == "blocked"  # type: ignore[index]
    assert all(
        fold["status"] == "blocked" for fold in artifact["folds"]  # type: ignore[index]
    )
    with pytest.raises(FutureValueUncertaintyError, match="no verified prior mapping"):
        apply_strict_prior_support_calibration(artifact, [1.0], fold_id=2)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("future_date", "outside its validation window"),
        ("source_drift", "source drift"),
        ("game_overlap", "game IDs overlap"),
        ("series_overlap", "series IDs overlap"),
    ],
)
def test_source_date_and_overlap_mutations_fail_closed(mutation: str, message: str) -> None:
    folds = _folds()
    if mutation == "future_date":
        folds[1]["rows"][0]["date"] = "2026-04-01T00:00:00Z"  # type: ignore[index]
    elif mutation == "source_drift":
        folds[1]["source_receipt_sha256"] = "b" * 64
    elif mutation == "game_overlap":
        folds[1]["rows"][0]["game_id"] = folds[0]["rows"][0]["game_id"]  # type: ignore[index]
    else:
        folds[1]["rows"][0]["series_id"] = folds[0]["rows"][0]["series_id"]  # type: ignore[index]
    with pytest.raises(FutureValueUncertaintyError, match=message):
        build_strict_prior_support_calibration(
            folds,
            source_receipt=SOURCE,
            variant="future_player_form",
            minimum_training_rows=10,
            minimum_bin_rows=2,
            minimum_bins=2,
            maximum_bins=5,
        )


def test_mutated_artifact_rows_and_receipt_source_are_rejected() -> None:
    artifact = _artifact()
    mutated = copy.deepcopy(artifact)
    mutated["rows"][0]["support"] = 999.0  # type: ignore[index]
    with pytest.raises(FutureValueUncertaintyError, match="artifact hash"):
        verify_support_calibration_artifact(mutated)

    with pytest.raises(FutureValueUncertaintyError, match="expected source"):
        verify_support_calibration_artifact(artifact, expected_source_receipt_sha256="b" * 64)


def test_absolute_logit_residual_target_is_explicit() -> None:
    artifact = _artifact(target_kind="absolute_logit_residual")
    assert artifact["target"]["kind"] == "absolute_logit_residual"  # type: ignore[index]
    assert "proper scoring residual" not in artifact["target"]["description"]  # type: ignore[index]
    assert verify_support_calibration_artifact(artifact)
