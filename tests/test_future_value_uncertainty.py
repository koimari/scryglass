from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from lol_kills.research import future_value_uncertainty as uncertainty
from lol_kills.research.future_value_uncertainty import (
    FutureValueUncertaintyError,
    bootstrap_future_value_uncertainty,
    cluster_bootstrap_weights,
    verify_uncertainty_receipt,
)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    training_rows: list[dict[str, object]] = []
    for series_index in range(12):
        for map_index, target in enumerate((0, 1)):
            training_rows.append(
                {
                    "game_id": f"g-{series_index}-{map_index}",
                    "series_id": f"series-{series_index}",
                    "x": float((series_index - 5.5) / 5.0 + (0.2 if target else -0.2)),
                    "target": target,
                }
            )
    validation = pd.DataFrame(
        [
            {
                "game_id": "v-1",
                "series_id": "validation-series",
                "x": 0.5,
                "support_status": "complete",
                "imputation_status": "not_needed",
                "target": 1,
            },
            {
                "game_id": "v-2",
                "series_id": "validation-series",
                "x": np.nan,
                "support_status": "sparse",
                "imputation_status": "fixed_fold_local",
                "target": 0,
            },
        ]
    )
    return pd.DataFrame(training_rows), validation


def _run(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    **kwargs: object,
) -> dict[str, object]:
    return bootstrap_future_value_uncertainty(
        train,
        validation,
        feature_names=("x",),
        selected_c=0.3,
        imputation_values=(0.0,),
        scales=(1.0,),
        source_receipt={"source_as_of": "2026-08-20T00:00:00Z", "source_game_count": len(train)},
        calibration={"method": "fixed_identity", "version": 1},
        requested_draws=10,
        minimum_accepted_draws=10,
        seed=461,
        **kwargs,
    )


def test_cluster_weights_keep_each_series_whole() -> None:
    weights, selected = cluster_bootstrap_weights(
        ["b", "a", "a", "c", "b", "c"],
        seed=461,
    )

    assert len(selected) == 3
    assert weights[1] == weights[2]
    assert weights[0] == weights[4]
    assert weights[3] == weights[5]
    assert set(weights).issubset({0.0, 1.0, 2.0, 3.0})


def test_cluster_bootstrap_is_deterministic() -> None:
    train, validation = _frames()

    first = _run(train, validation)
    second = _run(train, validation)

    assert first == second
    assert first["receipt"]["draws"]["accepted"] == 10  # type: ignore[index]


def test_default_draw_request_and_acceptance_gate_are_conservative() -> None:
    assert uncertainty.DEFAULT_REQUESTED_DRAWS == 2000
    assert uncertainty._required_accepted_draws(2000, None) == 1980
    assert uncertainty._required_accepted_draws(2000, 1) == 1980
    assert uncertainty._required_accepted_draws(10, None) == 10


def test_outer_validation_series_cannot_enter_draw_population() -> None:
    train, validation = _frames()
    validation.loc[:, "series_id"] = "series-0"

    with pytest.raises(FutureValueUncertaintyError, match="outer validation series"):
        _run(train, validation)


def test_convergence_failure_blocks_the_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    train, validation = _frames()

    def fail(*args: object, **kwargs: object) -> tuple[np.ndarray, dict[str, object]]:
        raise FutureValueUncertaintyError("forced non-converged fit")

    monkeypatch.setattr(uncertainty, "_fit_zero_intercept_logistic", fail)
    with pytest.raises(FutureValueUncertaintyError, match="accepted draws"):
        _run(train, validation, point_coefficients=(0.2,))


def test_each_draw_and_interval_has_exact_side_swap_complement() -> None:
    train, validation = _frames()
    artifact = _run(train, validation)

    for record in artifact["draws"]["records"]:  # type: ignore[index]
        assert record["status"] == "accepted"
        assert record["side_swap_max_complement_error"] <= 1e-12
    for row in artifact["intervals"]["rows"]:  # type: ignore[index]
        interval = row["probability_interval"]
        swapped = row["side_swap_probability_interval"]
        assert swapped["lower"] == pytest.approx(1.0 - interval["upper"])
        assert swapped["median"] == pytest.approx(1.0 - interval["median"])
        assert swapped["upper"] == pytest.approx(1.0 - interval["lower"])


def test_support_and_imputation_labels_are_preserved_without_target_ledger() -> None:
    train, validation = _frames()
    artifact = _run(train, validation)

    rows = artifact["pre_event_uncertainty_ledger"]["rows"]  # type: ignore[index]
    assert rows[0]["support_status"] == "complete"
    assert rows[1]["support_status"] == "sparse"
    assert rows[1]["imputation_status"] == "fixed_fold_local"
    assert "target" not in artifact["pre_event_uncertainty_ledger"]["columns"]  # type: ignore[index]
    assert all("target" not in row for row in rows)


def test_receipt_binds_source_fold_series_transform_calibration_and_draws() -> None:
    train, validation = _frames()
    artifact = _run(train, validation)

    assert verify_uncertainty_receipt(artifact)
    assert all(value is False for value in artifact["authority_flags"].values())  # type: ignore[index]
    changed = copy.deepcopy(artifact)
    changed["receipt"]["series"]["train_series_ids"][0] = "forged"  # type: ignore[index]
    with pytest.raises(FutureValueUncertaintyError, match="receipt hash"):
        verify_uncertainty_receipt(changed)
