from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from lol_kills.v2.market import full_pipeline_uncertainty_v1 as uncertainty


ROOT = Path(__file__).resolve().parents[3]


def test_stream_seeds_and_series_samples_are_deterministic_and_distinct() -> None:
    first = uncertainty._sample_indices(
        20, draw_id=7, stream="ratings-development"
    )
    assert first == uncertainty._sample_indices(
        20, draw_id=7, stream="ratings-development"
    )
    assert first != uncertainty._sample_indices(
        20, draw_id=7, stream="phase-one-recalibration"
    )
    assert uncertainty._seed(7, "ratings-development") != uncertainty._seed(
        8, "ratings-development"
    )


def test_phase_one_resampling_keeps_complete_series() -> None:
    rows = [
        {"series_id": "a", "event_id": "a1"},
        {"series_id": "a", "event_id": "a2"},
        {"series_id": "b", "event_id": "b1"},
    ]
    sampled = uncertainty._phase_one_sample(rows, [0, 0])
    assert [row["event_id"] for row in sampled] == ["a1", "a2", "a1", "a2"]


def test_terminal_draft_requires_exact_fresh_refit_event_roster_patch_and_time() -> None:
    target = {
        "captured_at_utc": "2026-08-02T12:00:00+00:00",
        "event": {
            "event_id": "event-1",
            "league": "LCS",
            "patch": "26.15",
            "blue_organization_id": "blue",
            "blue_organization_name": "Blue",
            "red_organization_id": "red",
            "red_organization_name": "Red",
        },
    }
    target_ratings = {
        "event": {"event_start_utc": "2026-08-02T18:00:00+00:00"},
        "input_receipts": {
            "roster": {"raw_sha256": "a" * 64, "canonical_sha256": "b" * 64},
            "patch": {"raw_sha256": "c" * 64},
        },
    }
    refit = {
        "built_at_utc": "2026-08-02T11:00:00+00:00",
        "event": {
            "event_id": "event-1",
            "event_start_utc": "2026-08-02T18:00:00+00:00",
            "league": "LCS",
            "patch": "26.15",
            "blue_organization_id": "blue",
            "red_organization_id": "red",
        },
        "input_receipts": {
            "roster_raw_sha256": "a" * 64,
            "roster_canonical_sha256": "b" * 64,
            "patch_raw_sha256": "c" * 64,
        },
    }
    prepared = {
        "refit": refit,
        "roster": {
            "teams": [
                {"organization_name": "Blue"},
                {"organization_name": "Red"},
            ]
        },
    }
    uncertainty._assert_target_refit_binding(
        target=target,
        target_ratings=target_ratings,
        refit_prepared=prepared,
    )

    wrong_roster = deepcopy(target_ratings)
    wrong_roster["input_receipts"]["roster"]["raw_sha256"] = "d" * 64
    with pytest.raises(
        uncertainty.FullPipelineUncertaintyError,
        match="receipts differ",
    ):
        uncertainty._assert_target_refit_binding(
            target=target,
            target_ratings=wrong_roster,
            refit_prepared=prepared,
        )

    late_refit = deepcopy(prepared)
    late_refit["refit"]["built_at_utc"] = "2026-08-02T12:00:01+00:00"
    with pytest.raises(
        uncertainty.FullPipelineUncertaintyError,
        match="receipts differ",
    ):
        uncertainty._assert_target_refit_binding(
            target=target,
            target_ratings=target_ratings,
            refit_prepared=late_refit,
        )


def artifact(monkeypatch: pytest.MonkeyPatch) -> dict:
    populations = {
        "ratings_development_series": 5,
        "draft_development_train_series": 4,
        "draft_development_calibration_series": 2,
        "phase_one_recalibration_series": 3,
    }
    stream_bindings = {
        "ratings_development": (
            "ratings-development",
            "ratings_development_series",
        ),
        "draft_development_train": (
            "draft-development-train",
            "draft_development_train_series",
        ),
        "draft_development_calibration": (
            "draft-development-calibration",
            "draft_development_calibration_series",
        ),
        "phase_one_recalibration": (
            "phase-one-recalibration",
            "phase_one_recalibration_series",
        ),
    }
    rating_probability = 0.60
    draft_logit = 0.10
    raw_combined = uncertainty._sigmoid(
        uncertainty._logit(rating_probability) + draft_logit
    )
    draws = []
    for draw_id in range(uncertainty.RESAMPLES):
        draws.append(
            {
                "draw_id": draw_id,
                "seeds": {
                    name: uncertainty._seed(draw_id, stream)
                    for name, (stream, _population) in stream_bindings.items()
                },
                "sample_digests": {
                    name: uncertainty._sample_digest(
                        uncertainty._sample_indices(
                            populations[population],
                            draw_id=draw_id,
                            stream=stream,
                        )
                    )
                    for name, (stream, population) in stream_bindings.items()
                },
                "refit": {
                    "rating_probability_blue": rating_probability,
                    "draft_scaled_logit_blue": draft_logit,
                    "draft": {
                        "feature_count": 12,
                        "calibration_slope": 1.0,
                        "optimizer_iterations": 3,
                        "optimizer_gradient_max_abs": 1e-9,
                    },
                    "raw_combined_probability_blue": raw_combined,
                    "combined_recalibration_intercept": 0.0,
                    "combined_recalibration_slope": 1.0,
                    "rating_only_recalibration_intercept": 0.0,
                    "rating_only_recalibration_slope": 1.0,
                },
                "probability_blue": raw_combined,
            }
        )
    probabilities = np.asarray([draw["probability_blue"] for draw in draws])
    interval = [
        float(np.quantile(probabilities, uncertainty.PERCENTILE_INTERVAL[0])),
        float(np.quantile(probabilities, uncertainty.PERCENTILE_INTERVAL[1])),
    ]
    point_rating = rating_probability
    point_draft_logit = draft_logit
    point_raw = raw_combined
    target_event = {
        "event_id": "event-1",
        "series_id": "series-1",
        "game_number": 1,
        "league": "LCS",
        "patch": "26.17",
        "blue_organization_id": "blue",
        "blue_organization_name": "Blue",
        "red_organization_id": "red",
        "red_organization_name": "Red",
    }
    target = {
        "artifact_sha256": "2" * 64,
        "captured_at_utc": "2026-08-02T12:00:00+00:00",
        "event": target_event,
        "draft_index": {"scaled_logit_blue": point_draft_logit},
    }
    target_ratings = {"input_receipts": {}, "event": {}}
    refit = {
        "artifact_sha256": "7" * 64,
        "source_snapshot": {
            "locator": "data/lol/v2/snapshots/rating-deployment/test/manifest.json",
            "raw_sha256": "8" * 64,
            "artifact_sha256": "9" * 64,
        },
        "input_receipts": {
            "roster_raw_sha256": "a" * 64,
            "roster_canonical_sha256": "b" * 64,
            "patch_raw_sha256": "c" * 64,
        },
        "phase_one_pass": {
            "result_locator": "data/lol/v2/evaluation/match-winner-market-v1/phase-one/evaluations/run-1.json",
            "result_raw_sha256": "3" * 64,
            "result_artifact_sha256": "4" * 64,
        },
    }
    refit_prepared = {
        "refit": refit,
        "input_data": type("Input", (), {"development_series": [1, 2, 3, 4, 5]})(),
    }
    result_raw = b"phase-result"
    recalibration_raw = b"recalibration"
    refit["phase_one_pass"]["result_raw_sha256"] = uncertainty._sha256_bytes(
        result_raw
    )
    monkeypatch.setattr(
        uncertainty,
        "_target",
        lambda _root, _locator: (b"target", target, target_ratings, {}),
    )
    monkeypatch.setattr(
        uncertainty,
        "_fresh_refit",
        lambda _root, locator, _environment: (
            locator,
            b"refit",
            refit,
            refit_prepared,
        ),
    )
    monkeypatch.setattr(
        uncertainty, "_assert_target_refit_binding", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        uncertainty,
        "_fresh_point_calculation",
        lambda **_kwargs: (point_rating, point_draft_logit, point_raw),
    )
    monkeypatch.setattr(
        uncertainty.evaluation,
        "_read_regular",
        lambda _root, locator, _label: (
            result_raw if locator.endswith("run-1.json") else recalibration_raw
        ),
    )
    monkeypatch.setattr(
        uncertainty.evaluation, "_strict_object", lambda _raw, _label: {}
    )
    monkeypatch.setattr(
        uncertainty.evaluation,
        "validate_phase_one_evaluation_result",
        lambda _value: {"artifact_sha256": "4" * 64, "phase_one_models_passed": True},
    )
    monkeypatch.setattr(
        uncertainty.recalibration,
        "validate_phase_one_recalibration_artifact",
        lambda _value: {
            "artifact_sha256": "6" * 64,
            "inputs": {"phase_one_result_artifact_sha256": "4" * 64},
            "models": {
                "ratings_plus_draft": {"intercept": 0.0, "slope": 1.0}
            },
        },
    )
    payload = {
        "schema_version": uncertainty.SCHEMA_VERSION,
        "result_state": uncertainty.RESULT_STATE,
        "event": {
            **target_event,
            "target_prediction_locator": "data/lol/v2/evaluation/draft-terminal-v1/predictions/event-1.json",
            "target_prediction_raw_sha256": uncertainty._sha256_bytes(b"target"),
            "target_prediction_artifact_sha256": "2" * 64,
        },
        "inputs": {
            "phase_one_result_locator": "data/lol/v2/evaluation/match-winner-market-v1/phase-one/evaluations/run-1.json",
            "phase_one_result_raw_sha256": uncertainty._sha256_bytes(result_raw),
            "phase_one_result_artifact_sha256": "4" * 64,
            "recalibration_artifact_locator": "data/lol/v2/evaluation/match-winner-market-v1/recalibration/phase-one.json",
            "recalibration_artifact_raw_sha256": uncertainty._sha256_bytes(
                recalibration_raw
            ),
            "recalibration_artifact_sha256": "6" * 64,
            "rating_refit_locator": "data/lol/v2/evaluation/rating-deployment/refits/event-1.json",
            "rating_refit_raw_sha256": uncertainty._sha256_bytes(b"refit"),
            "rating_refit_artifact_sha256": "7" * 64,
            "rating_source_snapshot_locator": refit["source_snapshot"]["locator"],
            "rating_source_snapshot_raw_sha256": "8" * 64,
            "rating_source_snapshot_artifact_sha256": "9" * 64,
            "rating_roster_raw_sha256": "a" * 64,
            "rating_roster_canonical_sha256": "b" * 64,
            "rating_patch_raw_sha256": "c" * 64,
        },
        "bootstrap_contract": {
            "method": "series_cluster_bootstrap_full_prediction_pipeline",
            "confidence_level": 0.95,
            "resamples": uncertainty.RESAMPLES,
            "master_seed": uncertainty.MASTER_SEED,
            "percentile_interval": list(uncertainty.PERCENTILE_INTERVAL),
            "populations": populations,
            "ratings_development_resampling": "series_with_replacement_preserve_chronological_order",
            "draft_development_resampling": "train_and_calibration_series_resampled_separately_with_replacement",
            "phase_one_recalibration_resampling": "series_with_replacement",
            "candidate_and_hyperparameters_fixed": True,
            "ratings_state_refit_in_each_resample": True,
            "draft_terms_refit_in_each_resample": True,
            "phase_one_recalibration_refit_in_each_resample": True,
            "phase_one_stored_predictions_used_for_recalibration_refit": True,
            "target_event_rating_and_draft_predictions_refit_in_each_resample": True,
            "fresh_post_validation_refit_exactly_bound": True,
            "fresh_point_rating_replayed_from_same_source_and_roster": True,
            "target_event_outcome_or_market_price_used": False,
            "failure_or_nonconvergence_action": "event_probability_unavailable",
        },
        "point_calculation": {
            "rating_probability_blue": point_rating,
            "draft_scaled_logit_blue": point_draft_logit,
            "raw_probability_blue": point_raw,
            "recalibration_intercept": 0.0,
            "recalibration_slope": 1.0,
            "probability_blue": point_raw,
        },
        "uncertainty": {
            "draws": draws,
            "draws_sha256": uncertainty._canonical_sha256(draws),
            "probability_interval_blue": interval,
            "opposing_probability_interval_red": [
                1.0 - interval[1],
                1.0 - interval[0],
            ],
            "interval_is_epistemic": True,
            "interval_is_not_a_guarantee_of_binary_outcome_coverage": True,
        },
        "source_locks": uncertainty._source_locks(ROOT),
        "qualification": {
            "phase_one_models_independently_passed": True,
            "recalibration_artifact_present": True,
            "fresh_post_validation_rating_refit_validated": True,
            "fresh_rating_source_roster_patch_and_point_replayed": True,
            "recalibration_independently_registered": False,
            "uncertainty_independently_registered": False,
            "phase_two_opening_registered": False,
            "target_event_outcome_present": False,
            "target_event_outcome_accessed": False,
            "market_price_used": False,
        },
        "authority": uncertainty.AUTHORITY,
        "claim_ceiling": uncertainty.CLAIM_CEILING,
    }
    payload["artifact_sha256"] = uncertainty._canonical_sha256(payload)
    return payload


def test_uncertainty_artifact_replays_every_seed_sample_and_probability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = artifact(monkeypatch)
    checked = uncertainty.validate_event_uncertainty_candidate(
        value, root=ROOT, environment={}
    )
    assert len(checked["uncertainty"]["draws"]) == 2000
    assert all(authority is False for authority in checked["authority"].values())

    changed = deepcopy(value)
    changed["uncertainty"]["draws"][10]["seeds"]["ratings_development"] += 1
    changed["uncertainty"]["draws_sha256"] = uncertainty._canonical_sha256(
        changed["uncertainty"]["draws"]
    )
    changed["artifact_sha256"] = uncertainty._canonical_sha256(
        {key: item for key, item in changed.items() if key != "artifact_sha256"}
    )
    with pytest.raises(uncertainty.FullPipelineUncertaintyError, match="seed changed"):
        uncertainty.validate_event_uncertainty_candidate(
            changed, root=ROOT, environment={}
        )

    point_tamper = deepcopy(value)
    point_tamper["point_calculation"]["rating_probability_blue"] = 0.61
    point_tamper["point_calculation"]["raw_probability_blue"] = (
        uncertainty._sigmoid(
            uncertainty._logit(0.61)
            + point_tamper["point_calculation"]["draft_scaled_logit_blue"]
        )
    )
    point_tamper["point_calculation"]["probability_blue"] = point_tamper[
        "point_calculation"
    ]["raw_probability_blue"]
    point_tamper["artifact_sha256"] = uncertainty._canonical_sha256(
        {
            key: item
            for key, item in point_tamper.items()
            if key != "artifact_sha256"
        }
    )
    with pytest.raises(
        uncertainty.FullPipelineUncertaintyError,
        match="exact fresh-refit replay",
    ):
        uncertainty.validate_event_uncertainty_candidate(
            point_tamper, root=ROOT, environment={}
        )

    refit_tamper = deepcopy(value)
    refit_tamper["inputs"]["rating_refit_artifact_sha256"] = "d" * 64
    refit_tamper["artifact_sha256"] = uncertainty._canonical_sha256(
        {
            key: item
            for key, item in refit_tamper.items()
            if key != "artifact_sha256"
        }
    )
    with pytest.raises(
        uncertainty.FullPipelineUncertaintyError,
        match="fresh-refit file binding changed",
    ):
        uncertainty.validate_event_uncertainty_candidate(
            refit_tamper, root=ROOT, environment={}
        )
