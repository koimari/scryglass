from __future__ import annotations

import copy
from typing import Any, cast

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
                    "date": pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(days=series_index),
                    "x": float((series_index - 5.5) / 5.0 + (0.2 if target else -0.2)),
                    "target": target,
                }
            )
    validation = pd.DataFrame(
        [
            {
                "game_id": "v-1",
                "series_id": "validation-series",
                "date": pd.Timestamp("2026-02-01T00:00:00Z"),
                "x": 0.5,
                "support_status": "complete",
                "imputation_status": "not_needed",
                "target": 1,
            },
            {
                "game_id": "v-2",
                "series_id": "validation-series",
                "date": pd.Timestamp("2026-02-02T00:00:00Z"),
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
    *,
    calibration: dict[str, object] | None = None,
    requested_draws: int = 1000,
    **kwargs: Any,
) -> dict[str, Any]:
    return bootstrap_future_value_uncertainty(
        train,
        validation,
        feature_names=("x",),
        selected_c=0.3,
        imputation_values=(0.0,),
        scales=(1.0,),
        source_receipt={"source_as_of": "2026-08-20T00:00:00Z", "source_game_count": len(train)},
        calibration=calibration or {"method": "fixed_identity", "version": 1},
        requested_draws=requested_draws,
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


def test_positive_weight_draw_requires_both_target_classes() -> None:
    target = np.asarray([0, 1, 0, 1])
    assert uncertainty._positive_weight_rows_have_both_target_classes(
        target, np.asarray([1.0, 1.0, 0.0, 0.0])
    )
    assert not uncertainty._positive_weight_rows_have_both_target_classes(
        target, np.asarray([1.0, 0.0, 1.0, 0.0])
    )


def test_cluster_bootstrap_is_deterministic() -> None:
    train, validation = _frames()

    first = _run(train, validation)
    second = _run(train, validation)

    assert first == second
    assert first["receipt"]["draws"]["accepted"] == 1000  # type: ignore[index]


def test_default_draw_request_and_acceptance_gate_are_conservative() -> None:
    assert uncertainty.DEFAULT_REQUESTED_DRAWS == 2000
    assert uncertainty._required_accepted_draws(2000) == 1980
    with pytest.raises(FutureValueUncertaintyError, match="at least 1000"):
        uncertainty._required_accepted_draws(999)


def test_requested_draw_floor_cannot_be_bypassed() -> None:
    train, validation = _frames()
    with pytest.raises(FutureValueUncertaintyError, match="at least 1000"):
        _run(train, validation, requested_draws=999)


def test_fixed_zero_intercept_calibration_slope_is_applied() -> None:
    train, validation = _frames()
    artifact = _run(
        train,
        validation,
        calibration={"method": "scalar_zero_intercept", "slope": 2.0},
        point_coefficients=(0.4,),
    )

    row = artifact["pre_event_uncertainty_ledger"]["rows"][0]  # type: ignore[index]
    assert row["point_logit"] == pytest.approx(0.4)
    assert row["point_probability"] == pytest.approx(1.0 / (1.0 + np.exp(-0.4)))
    assert artifact["receipt"]["calibration_slope"] == 2.0  # type: ignore[index]
    assert artifact["receipt"]["calibration"]["intercept"] == 0.0  # type: ignore[index]


@pytest.mark.parametrize(
    "calibration",
    [
        {"method": "isotonic", "bins": [0.2, 0.8]},
        {"method": "scalar_zero_intercept", "slope": 1.0, "intercept": 0.2},
        {"method": "identity", "slope": 2.0},
    ],
)
def test_unsupported_or_non_identity_calibration_is_rejected(
    calibration: dict[str, object],
) -> None:
    with pytest.raises(FutureValueUncertaintyError, match="calibration"):
        uncertainty._fixed_calibration(calibration)


def test_outer_validation_series_cannot_enter_draw_population() -> None:
    train, validation = _frames()
    validation.loc[:, "series_id"] = "series-0"

    with pytest.raises(FutureValueUncertaintyError, match="outer validation series"):
        _run(train, validation)


def test_outer_validation_date_boundary_is_strict_and_receipted() -> None:
    train, validation = _frames()
    artifact = _run(train, validation)
    assert artifact["receipt"]["fold"]["strict_date_boundary"] is True  # type: ignore[index]
    assert artifact["receipt"]["fold"]["train_date_max"] < artifact["receipt"]["fold"]["outer_validation_date_min"]  # type: ignore[index]

    same_day = validation.copy()
    same_day.loc[:, "date"] = train["date"].max()
    with pytest.raises(FutureValueUncertaintyError, match="strictly after"):
        _run(train, same_day)


def test_row_level_game_series_assignment_is_receipted() -> None:
    train, validation = _frames()
    artifact = _run(train, validation)
    series = artifact["receipt"]["series"]  # type: ignore[index]
    assert series["train_game_series_assignment_sha256"]
    assert series["outer_validation_game_series_assignment_sha256"]
    assert uncertainty.verify_uncertainty_receipt(artifact)

    forged = copy.deepcopy(artifact)
    forged["receipt"]["series"]["train_game_series_assignments"][0]["series_id"] = "forged"  # type: ignore[index]
    with pytest.raises(FutureValueUncertaintyError, match="receipt hash"):
        uncertainty.verify_uncertainty_receipt(forged)

    forged["receipt"]["series_sha256"] = uncertainty._sha256_json(forged["receipt"]["series"])  # type: ignore[index]
    receipt_payload = dict(cast(dict[str, Any], forged["receipt"]))
    receipt_payload.pop("receipt_sha256", None)
    forged["receipt"]["receipt_sha256"] = uncertainty._sha256_json(receipt_payload)  # type: ignore[index]
    with pytest.raises(FutureValueUncertaintyError, match="series assignment"):
        uncertainty.verify_uncertainty_receipt(forged)


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
    assert all(
        value is False
        for value in cast(dict[str, bool], artifact["authority"]).values()
    )  # type: ignore[index]
    changed = copy.deepcopy(artifact)
    changed["receipt"]["series"]["train_series_ids"][0] = "forged"  # type: ignore[index]
    with pytest.raises(FutureValueUncertaintyError, match="receipt hash"):
        verify_uncertainty_receipt(changed)


def test_verifier_rejects_target_insertion_duplicate_ledger_and_authority() -> None:
    train, validation = _frames()
    artifact = _run(train, validation)

    target_forged = copy.deepcopy(artifact)
    target_forged["pre_event_uncertainty_ledger"]["rows"][0]["target"] = 1  # type: ignore[index]
    target_forged["pre_event_uncertainty_ledger"]["sha256"] = uncertainty._sha256_json(  # type: ignore[index]
        target_forged["pre_event_uncertainty_ledger"]["rows"]  # type: ignore[index]
    )
    target_forged["receipt"]["ledger_sha256"] = target_forged["pre_event_uncertainty_ledger"]["sha256"]  # type: ignore[index]
    target_receipt_payload = dict(cast(dict[str, Any], target_forged["receipt"]))
    target_receipt_payload.pop("receipt_sha256", None)
    target_forged["receipt"]["receipt_sha256"] = uncertainty._sha256_json(target_receipt_payload)  # type: ignore[index]
    with pytest.raises(FutureValueUncertaintyError, match="target|receipt"):
        verify_uncertainty_receipt(target_forged)

    alias_forged = copy.deepcopy(artifact)
    alias_forged["ledger"] = alias_forged["pre_event_uncertainty_ledger"]
    with pytest.raises(FutureValueUncertaintyError, match="duplicate|receipt"):
        verify_uncertainty_receipt(alias_forged)

    authority_forged = copy.deepcopy(artifact)
    authority_forged["authority"]["odds"] = True  # type: ignore[index]
    authority_forged["receipt"]["authority"]["odds"] = True  # type: ignore[index]
    authority_receipt_payload = dict(cast(dict[str, Any], authority_forged["receipt"]))
    authority_receipt_payload.pop("receipt_sha256", None)
    authority_forged["receipt"]["receipt_sha256"] = uncertainty._sha256_json(authority_receipt_payload)  # type: ignore[index]
    with pytest.raises(FutureValueUncertaintyError, match="authority|receipt"):
        verify_uncertainty_receipt(authority_forged)
