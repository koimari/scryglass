from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from lol_kills.research.future_value_rating import (
    FutureValueSourceError,
    _apply_strict_prior_calibration,
    _fit_strict_prior_calibration,
    _normalise_strict_prior_calibration_folds,
)


SOURCE_HASH = "a" * 64


def _prior_frame() -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    frame = pd.DataFrame(
        [
            ("prior-1", "series-prior", "2025-12-01T00:00:00Z", 1),
            ("prior-2", "series-prior", "2025-12-01T01:00:00Z", 0),
            ("current-1", "series-current", "2026-01-01T00:00:00Z", 1),
        ],
        columns=["game_id", "series_id", "date", "target"],
    )
    prior_rows = [
        {
            "game_id": str(game_id),
            "series_id": str(series_id),
            "date": date,
            "raw_logit": 1.0 if int(target) else -1.0,
            "target": int(target),
        }
        for game_id, series_id, date, target in frame.iloc[:2].itertuples(
            index=False, name=None
        )
    ]
    return frame, prior_rows, [
        {
            "fold": 1,
            "validation_start": "2026-01-01T00:00:00Z",
            "validation_end": "2026-01-01T00:00:00Z",
            "validation_game_ids": ["current-1"],
            "validation_series_ids": ["series-current"],
        }
    ]


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


def test_first_outer_fold_fits_from_explicit_prior_calibration_fold() -> None:
    calibration = _fit_strict_prior_calibration(
        (-2.0, 2.0, -1.0, 1.0),
        (0, 1, 0, 1),
        source_receipt_sha256=SOURCE_HASH,
        current_fold=1,
        current_validation_game_ids=("g-current",),
        current_validation_start="2026-01-01T00:00:00Z",
        prior_fold_numbers=(0, 0, 0, 0),
        prior_game_ids=("g-prior-1", "g-prior-2", "g-prior-3", "g-prior-4"),
        prior_validation_ends=("2025-12-31T00:00:00Z",),
    )

    assert calibration["mode"] == "fitted"
    assert calibration["current_fold"] == 1
    assert calibration["prior_fold_numbers"] == [0]
    assert calibration["fit_rows"] == 4
    assert calibration["blockers"] == []


def test_prior_calibration_rows_bind_source_and_complete_series() -> None:
    frame, prior_rows, evaluation_folds = _prior_frame()
    model_binding = {
        "source_receipt_sha256": SOURCE_HASH,
        "variant": "future_player_form",
        "fit_window_end": "2025-11-30T00:00:00Z",
        "fit_game_identity_sha256": "b" * 64,
        "validation_game_identity_sha256": "c" * 64,
        "parameter_sha256": "d" * 64,
        "prediction_ledger_row_count": len(prior_rows),
        "prediction_ledger_rows_sha256": hashlib.sha256(
            json.dumps(
                prior_rows,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "model_receipt": {"path": "/tmp/model-receipt.json", "bytes": 1, "sha256": "e" * 64},
        "model_artifact": {"path": "/tmp/model-artifact.json", "bytes": 1, "sha256": "f" * 64},
        "prediction_ledger": {"path": "/tmp/prediction-ledger.json", "bytes": 1, "sha256": "0" * 64},
        "code": {
            "commit": "1" * 40,
            "files": [{"path": "/tmp/future-value-rating.py", "bytes": 1, "sha256": "2" * 64}],
        },
    }
    folds = [
        {
            "fold": 0,
            "source_receipt_sha256": SOURCE_HASH,
            "variant": "future_player_form",
            "train_end": "2025-11-30T00:00:00Z",
            "validation_start": "2025-12-01T00:00:00Z",
            "validation_end": "2025-12-01T01:00:00Z",
            "out_of_sample": True,
            "whole_series": True,
            "model_binding": model_binding,
            "rows": prior_rows,
        }
    ]

    normalized = _normalise_strict_prior_calibration_folds(
        folds,
        map_frame=frame,
        source_receipt_sha256=SOURCE_HASH,
        variant="future_player_form",
        evaluation_folds=evaluation_folds,
    )

    assert normalized[0]["row_count"] == 2
    assert normalized[0]["game_identity_sha256"]
    assert normalized[0]["rows"][0]["raw_probability"] == pytest.approx(
        0.7310585786300049
    )

    incomplete = [dict(folds[0], rows=prior_rows[:1])]
    with pytest.raises(FutureValueSourceError, match="complete series"):
        _normalise_strict_prior_calibration_folds(
            incomplete,
            map_frame=frame,
            source_receipt_sha256=SOURCE_HASH,
            variant="future_player_form",
            evaluation_folds=evaluation_folds,
        )


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
