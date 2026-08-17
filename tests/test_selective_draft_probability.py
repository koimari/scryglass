from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lol_kills import draft_recommendation as draft_recommendation_module
from lol_kills.research import evaluate_selective_draft_holdout as evaluator_module
from lol_kills.research import selective_draft_probability as selective_module
from lol_kills.research import selective_draft_constituents as constituent_module
from lol_kills.research import public_draft_score_promotion as promotion_module
from lol_kills.research import prepare_selective_draft_holdout_sources as source_module
from lol_kills.research import seal_selective_draft_holdout as sealer_module
from lol_kills.research import selective_draft_holdout_inventory as inventory_module
from lol_kills.research import verify_selective_draft_promotion as verifier_module
from lol_kills.research.selective_draft_probability import (
    CONFIDENCE_COLUMNS,
    PREDICTORS,
    SelectiveDraftProbabilityError,
    apply_selective_candidate,
    apply_side_symmetric_calibration,
    canonical_sha256,
    evaluate_frozen_candidate_holdout,
    evaluate_rolling_selective_probability,
    evaluate_selective_probability,
    fit_side_symmetric_calibration,
    fit_ridge_confidence,
    fit_selective_candidate,
    file_sha256,
    load_evaluation_frame,
    predict_ridge_confidence,
)


def _frame() -> pd.DataFrame:
    dates = pd.date_range("2025-07-01", periods=1600, freq="8h", tz="UTC")
    probability = np.linspace(0.2, 0.8, len(dates))
    frame = pd.DataFrame(
        {
            "game_uid": [f"g-{index}" for index in range(len(dates))],
            "series_id": [f"s-{index // 3}" for index in range(len(dates))],
            "date": dates,
            "league": "LEC",
            "source_patch": "16.1",
            "y": np.arange(len(dates)) % 2,
            "ensemble_probability": probability,
        }
    )
    for index, name in enumerate(PREDICTORS):
        frame[name] = np.clip(probability + (index - 1.5) * 0.01, 0.01, 0.99)
    frame["disagreement"] = frame[list(PREDICTORS)].std(axis=1, ddof=0)
    frame["prediction_range"] = frame[list(PREDICTORS)].max(axis=1) - frame[list(PREDICTORS)].min(axis=1)
    frame["ensemble_margin"] = abs(frame["ensemble_probability"] - 0.5)
    for name in PREDICTORS:
        frame[f"margin_{name}"] = abs(frame[name] - 0.5)
    for index, left in enumerate(PREDICTORS):
        for right in PREDICTORS[index + 1 :]:
            frame[f"distance_{left}_{right}"] = abs(frame[left] - frame[right])
    for name in CONFIDENCE_COLUMNS[-9:]:
        frame[name] = 1.0
    return frame


def test_confidence_features_are_side_symmetric() -> None:
    frame = _frame()
    mirrored = frame.copy()
    for name in PREDICTORS:
        mirrored[name] = 1.0 - mirrored[name]
    mirrored["ensemble_probability"] = 1.0 - mirrored["ensemble_probability"]
    mirrored["ensemble_margin"] = abs(mirrored["ensemble_probability"] - 0.5)
    for name in PREDICTORS:
        mirrored[f"margin_{name}"] = abs(mirrored[name] - 0.5)
    for index, left in enumerate(PREDICTORS):
        for right in PREDICTORS[index + 1 :]:
            mirrored[f"distance_{left}_{right}"] = abs(mirrored[left] - mirrored[right])
    np.testing.assert_allclose(frame[list(CONFIDENCE_COLUMNS)], mirrored[list(CONFIDENCE_COLUMNS)])


def test_probability_calibration_is_side_symmetric() -> None:
    probabilities = np.asarray([0.12, 0.31, 0.5, 0.74, 0.91])
    outcomes = np.asarray([0, 0, 1, 1, 1])
    slope = fit_side_symmetric_calibration(outcomes, probabilities)
    calibrated = apply_side_symmetric_calibration(probabilities, slope)
    mirrored = apply_side_symmetric_calibration(1.0 - probabilities, slope)

    assert 0.2 <= slope <= 3.0
    np.testing.assert_allclose(calibrated, 1.0 - mirrored, atol=1e-12)
    assert calibrated[2] == pytest.approx(0.5)


def test_evaluation_never_trains_on_later_outcomes() -> None:
    frame = _frame()
    report, predictions = evaluate_selective_probability(
        frame,
        selection_start="2025-07-01T00:00:00Z",
        selection_end="2025-12-01T00:00:00Z",
        evaluation_end="2026-06-01T00:00:00Z",
        minimum_auc=0.49,
    )
    changed = frame.copy()
    changed.loc[changed["date"] >= pd.Timestamp("2025-12-01", tz="UTC"), "y"] = 1 - changed.loc[
        changed["date"] >= pd.Timestamp("2025-12-01", tz="UTC"), "y"
    ]
    changed_report, changed_predictions = evaluate_selective_probability(
        changed,
        selection_start="2025-07-01T00:00:00Z",
        selection_end="2025-12-01T00:00:00Z",
        evaluation_end="2026-06-01T00:00:00Z",
        minimum_auc=0.49,
    )
    assert len(predictions) == report["evaluation"]["eligible_rows"]
    assert report["authority"] == "research_only"
    assert changed_report["selection"] == report["selection"]
    np.testing.assert_allclose(predictions["confidence_score"], changed_predictions["confidence_score"])
    assert predictions["probability_authorized"].equals(changed_predictions["probability_authorized"])


def test_loader_rejects_mismatched_outcome(tmp_path: Path) -> None:
    matrix = _frame().iloc[:20].copy()
    matrix_path = tmp_path / "matrix.parquet"
    matrix.to_parquet(matrix_path, index=False)
    paths = {}
    for name in PREDICTORS:
        prediction = matrix[["game_uid", "y"]].copy()
        prediction["p"] = matrix[name]
        if name == "identity":
            prediction.loc[0, "y"] = 1 - prediction.loc[0, "y"]
        path = tmp_path / f"{name}.parquet"
        prediction.to_parquet(path, index=False)
        paths[name] = path
    with pytest.raises(SelectiveDraftProbabilityError, match="outcomes do not match"):
        load_evaluation_frame(matrix_path=matrix_path, prediction_paths=paths)


def test_loader_accepts_prediction_artifacts_without_outcomes(tmp_path: Path) -> None:
    matrix = _frame().iloc[:20].copy()
    matrix_path = tmp_path / "matrix.parquet"
    matrix.to_parquet(matrix_path, index=False)
    paths = {}
    for name in PREDICTORS:
        path = tmp_path / f"{name}.parquet"
        matrix[["game_uid", name]].rename(columns={name: "p"}).to_parquet(
            path, index=False
        )
        paths[name] = path

    loaded = load_evaluation_frame(
        matrix_path=matrix_path, prediction_paths=paths
    )
    assert loaded["y"].equals(matrix["y"])


def test_confidence_model_round_trips_and_rolling_replay_is_prior_only() -> None:
    frame = _frame()
    features = frame[list(CONFIDENCE_COLUMNS)].to_numpy(dtype=float)
    target = np.square(frame["y"] - frame["ensemble_probability"]).to_numpy()
    model = fit_ridge_confidence(features[:500], target[:500], alpha=10.0)
    first = predict_ridge_confidence(model, features[500:600])
    second = predict_ridge_confidence(dict(model), features[500:600])
    np.testing.assert_allclose(first, second)

    report, selected = evaluate_rolling_selective_probability(
        frame,
        fold_edges=(
            "2025-07-01T00:00:00Z",
            "2025-11-01T00:00:00Z",
            "2026-03-01T00:00:00Z",
            "2026-07-01T00:00:00Z",
            "2026-11-01T00:00:00Z",
        ),
        target_coverage=0.9,
    )
    assert report["authority"] == "research_only"
    assert report["selected_rows"] == len(selected)
    assert [fold["training_rows"] for fold in report["folds"]] == sorted(
        fold["training_rows"] for fold in report["folds"]
    )


def test_frozen_candidate_applies_without_outcomes_and_rejects_tampering() -> None:
    frame = _frame()
    hashes = {name: "a" * 64 for name in ("matrix", *PREDICTORS)}
    artifact = fit_selective_candidate(
        frame,
        evidence_end="2027-01-01T00:00:00Z",
        input_sha256=hashes,
    )
    inference = frame.iloc[-20:].drop(columns="y")
    output = apply_selective_candidate(artifact, inference)
    tampered_features = inference.copy()
    tampered_features["ensemble_margin"] = 999.0
    repaired = apply_selective_candidate(artifact, tampered_features)

    assert artifact["authority"] == "research_only"
    assert artifact["public_probability"] is False
    assert len(output) == 20
    assert output["ensemble_probability"].between(0, 1).all()
    np.testing.assert_allclose(
        output["confidence_score"], repaired["confidence_score"]
    )

    tampered = dict(artifact)
    tampered["public_probability"] = True
    with pytest.raises(SelectiveDraftProbabilityError, match="receipt"):
        apply_selective_candidate(tampered, inference)

    with pytest.raises(SelectiveDraftProbabilityError, match="weights"):
        fit_selective_candidate(
            frame,
            evidence_end="2027-01-01T00:00:00Z",
            input_sha256=hashes,
            predictor_weights={name: 0.1 for name in PREDICTORS},
        )


def test_holdout_gate_stays_closed_until_the_sample_is_large_enough() -> None:
    frame = _frame()
    hashes = {name: "b" * 64 for name in ("matrix", *PREDICTORS)}
    artifact = fit_selective_candidate(
        frame,
        evidence_end="2026-11-15T00:00:00Z",
        input_sha256=hashes,
    )
    sealed_frame = frame.copy()
    sealed_frame["y"] = sealed_frame["y"].astype(object)
    sealed_frame.loc[
        sealed_frame["date"] >= pd.Timestamp("2026-11-15T00:00:00Z"), "y"
    ] = "sealed-outcome"
    with pytest.raises(
        SelectiveDraftProbabilityError, match="outcomes remain sealed"
    ):
        evaluate_frozen_candidate_holdout(
            artifact,
            sealed_frame,
            holdout_start="2026-11-15T00:00:00Z",
            holdout_end="2027-01-01T00:00:00Z",
        )

    with pytest.raises(SelectiveDraftProbabilityError, match="overlaps"):
        evaluate_frozen_candidate_holdout(
            artifact,
            frame,
            holdout_start="2026-11-14T00:00:00Z",
            holdout_end="2027-01-01T00:00:00Z",
        )


def test_frozen_holdout_gates_cannot_be_weakened() -> None:
    frame = _frame()
    hashes = {name: "b" * 64 for name in ("matrix", *PREDICTORS)}
    artifact = fit_selective_candidate(
        frame,
        evidence_end="2026-11-15T00:00:00Z",
        input_sha256=hashes,
    )

    with pytest.raises(SelectiveDraftProbabilityError, match="cannot be weakened"):
        evaluate_frozen_candidate_holdout(
            artifact,
            frame,
            holdout_start="2026-11-15T00:00:00Z",
            holdout_end="2027-01-01T00:00:00Z",
            minimum_selected_rows=1,
        )


def test_latest_protocol_binds_the_frozen_candidate_and_implementation() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / "data/lol/v2/evaluation/public-draft-score-promotion-protocol-v34.json"
    candidate_path = root / "data/lol/v2/evaluation/public-draft-score-selective-candidate-v34.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    receipt = candidate.pop("receipt_sha256")

    assert receipt == canonical_sha256(candidate)
    assert protocol["next_holdout"]["candidate_receipt_sha256"] == receipt
    assert protocol["next_holdout"]["candidate_artifact_sha256"] == file_sha256(
        candidate_path
    )
    assert protocol["iteration"]["implementation_sha256"] == file_sha256(
        Path(selective_module.__file__)
    )
    assert protocol["iteration"][
        "constituent_implementation_sha256"
    ] == file_sha256(Path(constituent_module.__file__))
    assert protocol["iteration"][
        "quantum_implementation_sha256"
    ] == file_sha256(Path(promotion_module.__file__))
    assert protocol["iteration"][
        "draft_builder_implementation_sha256"
    ] == file_sha256(Path(draft_recommendation_module.__file__))
    assert protocol["iteration"][
        "holdout_source_preparer_sha256"
    ] == file_sha256(Path(source_module.__file__))
    assert protocol["iteration"][
        "holdout_evaluator_sha256"
    ] == file_sha256(Path(evaluator_module.__file__))
    assert protocol["iteration"][
        "promotion_verifier_sha256"
    ] == file_sha256(Path(verifier_module.__file__))
    assert protocol["iteration"]["holdout_sealer_sha256"] == file_sha256(
        Path(sealer_module.__file__)
    )
    assert protocol["iteration"]["holdout_inventory_sha256"] == file_sha256(
        Path(inventory_module.__file__)
    )
    assert protocol["quantum_reproducibility"] == {
        "protocol_file_sha256": constituent_module.V24_QUANTUM_PROTOCOL_FILE_SHA256,
        "protocol_resolved_sha256": constituent_module.V24_QUANTUM_PROTOCOL_RESOLVED_SHA256,
        "feature_groups_sha256": constituent_module.V24_QUANTUM_FEATURE_GROUPS_SHA256,
        "reference_matrix_sha256": "0f9de19c3849c4d4454a833391626257552ee498e35e243e0309a5ac2a0ebf1c",
        "reference_player_source_sha256": "8d0285c6991b53cc3535653b99faa7f46bc55e2b204751f274835df6dfbcfb95",
        "reference_prediction_file_sha256": "c3dba387fc375a9c1cec6d27e8751e856934a6f2e6390af94277ba6062b1e1d8",
        "parity_rows": 69,
        "parity_max_absolute_probability_delta": 0.0,
        "parity_receipt_sha256": "60c001c53461368f3cf794146909d23dd6994784e98627a98eb15c0c0fda13dc",
    }
    assert protocol["authority"]["public_probability"].startswith("unavailable")
