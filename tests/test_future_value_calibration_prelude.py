from __future__ import annotations

import inspect

import pandas as pd
import pytest

from benchmarks.build_future_value_calibration_prelude import (
    PreludeError,
    _strict_prior_model_frame,
    _validate_outer_evaluation_cutoff,
    build_prelude,
)


def _fold(validation_end: str) -> dict[str, object]:
    return {"validation_end": validation_end}


def test_prelude_cutoff_requires_strictly_earlier_validation_end() -> None:
    with pytest.raises(PreludeError, match="strictly earlier"):
        _validate_outer_evaluation_cutoff(
            _fold("2025-05-09T17:54:05Z"),
            outer_evaluation_start="2025-05-09T17:54:05Z",
        )

    with pytest.raises(PreludeError, match="strictly earlier"):
        _validate_outer_evaluation_cutoff(
            _fold("2025-05-10T00:00:00Z"),
            outer_evaluation_start="2025-05-09T17:54:05Z",
        )


def test_prelude_cutoff_returns_normalized_utc_timestamp() -> None:
    result = _validate_outer_evaluation_cutoff(
        _fold("2025-05-09T17:54:04Z"),
        outer_evaluation_start=pd.Timestamp("2025-05-09T13:54:05-04:00"),
    )

    assert result == pd.Timestamp("2025-05-09T17:54:05Z")


def test_prelude_cutoff_rejects_timezone_free_input() -> None:
    with pytest.raises(PreludeError, match="UTC timezone"):
        _validate_outer_evaluation_cutoff(
            _fold("2025-05-09T17:54:04Z"),
            outer_evaluation_start="2025-05-09T17:54:05",
        )


def test_builder_requires_explicit_outer_evaluation_start() -> None:
    parameter = inspect.signature(build_prelude).parameters["outer_evaluation_start"]

    assert parameter.default is inspect.Parameter.empty


def test_prelude_folds_use_only_rows_before_outer_evaluation() -> None:
    frame = pd.DataFrame(
        {
            "game_id": ["prior", "boundary", "future"],
            "date": [
                "2025-05-09T17:54:04Z",
                "2025-05-09T17:54:05Z",
                "2025-05-10T00:00:00Z",
            ],
        }
    )

    prior, cutoff = _strict_prior_model_frame(
        frame,
        outer_evaluation_start="2025-05-09T17:54:05Z",
    )

    assert tuple(prior["game_id"]) == ("prior",)
    assert cutoff == pd.Timestamp("2025-05-09T17:54:05Z")


def test_prelude_strict_prior_frame_rejects_missing_dates() -> None:
    frame = pd.DataFrame({"game_id": ["bad"], "date": [None]})

    with pytest.raises(PreludeError, match="invalid game date"):
        _strict_prior_model_frame(
            frame,
            outer_evaluation_start="2025-05-09T17:54:05Z",
        )
