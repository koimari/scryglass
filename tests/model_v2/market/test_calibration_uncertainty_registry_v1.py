from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from lol_kills.v2.market import calibration_uncertainty_registry_v1 as registry


def _binding() -> dict:
    return {
        "probability_pipeline_readiness": {
            "locator": "data/readiness.json",
            "raw_sha256": "1" * 64,
            "artifact_sha256": "2" * 64,
            "runtime_identity": {
                "python_implementation": "CPython",
                "python_version": "3.13.5",
                "numpy_version": "2.3.1",
                "scipy_version": "1.16.0",
            },
        },
        "phase_one_result": {
            "locator": "data/result.json",
            "raw_sha256": "3" * 64,
            "artifact_sha256": "4" * 64,
            "independent_evaluation_registry_raw_sha256": "5" * 64,
            "independent_evaluation_registry_id": "phase-one-review-1",
        },
        "recalibration": {
            "locator": "data/recalibration.json",
            "raw_sha256": "6" * 64,
            "artifact_sha256": "7" * 64,
            "fitted_at_utc": "2026-09-01T12:00:00+00:00",
            "models": {"ratings_plus_draft": {}, "ratings_only": {}},
        },
        "uncertainty_verification": {
            "locator": "data/verification.json",
            "raw_sha256": "8" * 64,
            "artifact_sha256": "9" * 64,
            "target_prediction_locator": "data/prediction.json",
            "target_prediction_artifact_sha256": "a" * 64,
            "target_prediction_captured_at_utc": "2026-09-01T13:00:00+00:00",
            "target_excluded_from_phase_one": True,
            "target_must_be_excluded_from_phase_two": True,
            "resamples": 2_000,
            "master_seed": 20_260_805,
            "draws_sha256": "b" * 64,
            "probability_interval_blue": [0.4, 0.6],
        },
        "fast_uncertainty_verification": {
            "locator": "data/fast-verification.json",
            "raw_sha256": "f" * 64,
            "artifact_sha256": "0" * 64,
            "rating_bootstrap_locator": "data/rating-bootstrap.json",
            "rating_bootstrap_artifact_sha256": "1" * 64,
            "terminal_draws_sha256": "b" * 64,
            "all_2000_draw_records_equal_frozen_slow_path": True,
        },
        "implementation": {
            "recalibration_source_locator": "recalibration.py",
            "recalibration_source_raw_sha256": "c" * 64,
            "uncertainty_source_locator": "uncertainty.py",
            "uncertainty_source_raw_sha256": "d" * 64,
            "event_rating_bootstrap_source_locator": "rating-bootstrap.py",
            "event_rating_bootstrap_source_raw_sha256": "2" * 64,
            "fast_uncertainty_source_locator": "fast-uncertainty.py",
            "fast_uncertainty_source_raw_sha256": "3" * 64,
            "registry_source_locator": "registry.py",
            "registry_source_raw_sha256": "e" * 64,
        },
    }


def _reviews() -> list[dict]:
    return [
        {
            "review_scope": scope,
            "reviewer_id": f"independent-{index}",
            "reviewed_at_utc": f"2026-09-01T14:0{index}:00+00:00",
            "attestation": dict(attestation),
        }
        for index, (scope, attestation) in enumerate(
            registry.REVIEW_SCOPES.items(), start=1
        )
    ]


def _receipt() -> dict:
    return registry.registry_template(
        registry_id="calibration-uncertainty-review-1",
        registered_at_utc="2026-09-01T15:00:00+00:00",
        reviews=_reviews(),
        expected_binding=_binding(),
    )


def test_registry_requires_two_independent_exact_reproductions() -> None:
    receipt = _receipt()
    checked = registry.validate_calibration_uncertainty_registry(
        receipt, expected_binding=_binding()
    )
    assert checked["decision"]["recalibration_independently_registered"] is True
    assert checked["decision"][
        "uncertainty_implementation_independently_registered"
    ] is True
    assert checked["decision"]["phase_two_opened"] is False
    assert checked["authority"]["phase_two_opening_authority"] is False
    assert checked["authority"]["betting_authority"] is False


def test_registry_rejects_same_reviewer_or_incomplete_replay() -> None:
    same_reviewer = _receipt()
    same_reviewer["reviews"][1]["reviewer_id"] = same_reviewer["reviews"][0][
        "reviewer_id"
    ]
    with pytest.raises(
        registry.CalibrationUncertaintyRegistryError,
        match="not independent",
    ):
        registry.validate_calibration_uncertainty_registry(
            same_reviewer, expected_binding=_binding()
        )

    incomplete = _receipt()
    scope = incomplete["reviews"][1]["review_scope"]
    incomplete["reviews"][1]["attestation"] = dict(
        incomplete["reviews"][1]["attestation"]
    )
    incomplete["reviews"][1]["attestation"][
        "all_2000_seeded_series_resamples_and_sample_digests_replayed"
    ] = False
    assert scope == "FULL_PIPELINE_UNCERTAINTY_REPRODUCTION"
    with pytest.raises(
        registry.CalibrationUncertaintyRegistryError,
        match="attestation",
    ):
        registry.validate_calibration_uncertainty_registry(
            incomplete, expected_binding=_binding()
        )


def test_registry_rejects_binding_or_chronology_change() -> None:
    changed = _receipt()
    changed["binding"]["uncertainty_verification"]["draws_sha256"] = "f" * 64
    with pytest.raises(
        registry.CalibrationUncertaintyRegistryError,
        match="binding changed",
    ):
        registry.validate_calibration_uncertainty_registry(
            changed, expected_binding=_binding()
        )

    early = _receipt()
    early["registered_at_utc"] = "2026-09-01T12:30:00+00:00"
    with pytest.raises(
        registry.CalibrationUncertaintyRegistryError,
        match="chronology",
    ):
        registry.validate_calibration_uncertainty_registry(
            early, expected_binding=_binding()
        )


def test_loader_requires_exact_external_raw_pin(tmp_path: Path) -> None:
    receipt = _receipt()
    raw = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    path = tmp_path / "registry.json"
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    loaded = registry.load_pinned_calibration_uncertainty_registry(
        path=path,
        external_sha256=digest,
        expected_binding=_binding(),
    )
    assert loaded["recalibration_independently_registered"] is True
    assert loaded["phase_two_opening_authorized"] is False
    with pytest.raises(
        registry.CalibrationUncertaintyRegistryError,
        match="external pin",
    ):
        registry.load_pinned_calibration_uncertainty_registry(
            path=path,
            external_sha256="0" * 64,
            expected_binding=_binding(),
        )
