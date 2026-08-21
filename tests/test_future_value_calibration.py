from __future__ import annotations

import numpy as np
import pytest

from lol_kills.research.future_value_rating import (
    FutureValueSourceError,
    _apply_strict_prior_calibration,
    _fit_strict_prior_calibration,
)


SOURCE_HASH = "a" * 64


def test_first_outer_fold_uses_identity_and_records_prior_blocker() -> None:
    calibration = _fit_strict_prior_calibration(
        [],
        [],
        source_receipt_sha256=SOURCE_HASH,
        current_fold=1,
        current_validation_game_ids=("g-current",),
        current_validation_start="2026-01-10T00:00:00Z",
    )

    assert calibration["mode"] == "identity"
    assert calibration["slope"] == 1.0
    assert calibration["strict_prior"] is True
    assert calibration["uses_current_validation"] is False
    assert calibration["blockers"] == ["calibration_prior_validation_folds_missing"]
    assert calibration["optimizer_evidence"]["status"] == "identity"


def test_later_fold_fits_positive_slope_from_prior_rows_only() -> None:
    calibration = _fit_strict_prior_calibration(
        (-2.0, 2.0, -1.0, 1.0),
        (0, 1, 0, 1),
        source_receipt_sha256=SOURCE_HASH,
        current_fold=2,
        current_validation_game_ids=("g-current",),
        current_validation_start="2026-01-20T00:00:00Z",
        prior_fold_numbers=(1, 1, 1, 1),
        prior_game_ids=("g1", "g2", "g3", "g4"),
        prior_validation_ends=("2026-01-19T00:00:00Z",),
    )

    assert calibration["mode"] == "fitted"
    assert calibration["slope"] > 0.0
    assert calibration["optimizer_evidence"]["success"] is True
    assert calibration["optimizer_evidence"]["zero_intercept"] is True
    assert calibration["optimizer_evidence"]["positive_slope"] is True
    assert calibration["fit_game_identity_sha256"]
    assert calibration["source_receipt_sha256"] == SOURCE_HASH
    assert calibration["blockers"] == []


def test_calibration_rejects_current_fold_and_preserves_side_complement() -> None:
    with pytest.raises(FutureValueSourceError, match="current or a future fold"):
        _fit_strict_prior_calibration(
            (-1.0, 1.0),
            (0, 1),
            source_receipt_sha256=SOURCE_HASH,
            current_fold=2,
            current_validation_game_ids=("g-current",),
            current_validation_start="2026-01-20T00:00:00Z",
            prior_fold_numbers=(1, 2),
            prior_game_ids=("g1", "g2"),
            prior_validation_ends=("2026-01-19T00:00:00Z",),
        )

    calibration = {
        "mode": "fitted",
        "slope": 1.7,
    }
    logits = np.asarray([-2.0, -0.25, 0.25, 2.0])
    raw = 1.0 / (1.0 + np.exp(-logits))
    calibrated_logit, calibrated_probability = _apply_strict_prior_calibration(
        logits,
        raw,
        calibration,
    )
    _, swapped_probability = _apply_strict_prior_calibration(
        -logits,
        1.0 - raw,
        calibration,
    )

    assert np.isfinite(calibrated_logit).all()
    np.testing.assert_allclose(
        swapped_probability.to_numpy(),
        1.0 - calibrated_probability.to_numpy(),
        rtol=0.0,
        atol=1e-15,
    )
