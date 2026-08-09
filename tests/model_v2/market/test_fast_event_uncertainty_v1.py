from __future__ import annotations

from copy import deepcopy

import pytest

from lol_kills.v2.market import fast_event_uncertainty_v1 as fast
from lol_kills.v2.market import full_pipeline_uncertainty_v1 as frozen


def test_fast_draw_reuses_exact_frozen_rating_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rating_draws = [
        {
            "draw_id": draw_id,
            "seed": frozen._seed(draw_id, "ratings-development"),
            "sample_digest": frozen._sample_digest(
                frozen._sample_indices(
                    3, draw_id=draw_id, stream="ratings-development"
                )
            ),
            "rating_probability_blue": 0.55,
        }
        for draw_id in range(frozen.RESAMPLES)
    ]
    prepared = {
        "rating": {"draws": rating_draws},
        "draft_train_order": ["train-a", "train-b"],
        "draft_calibration_order": ["cal-a", "cal-b"],
        "phase_one_rows": [
            {
                "series_id": "series-a",
                "blue_win": 1,
                "ratings_plus_draft": 0.6,
                "ratings_only": 0.55,
            },
            {
                "series_id": "series-b",
                "blue_win": 0,
                "ratings_plus_draft": 0.4,
                "ratings_only": 0.45,
            },
        ],
        "draft_rows": [object()],
        "draft_baseline_logits": {},
        "target_metadata": {},
        "draft_grouped": {},
    }
    monkeypatch.setattr(
        fast.frozen,
        "_draft_target_scaled_logit",
        lambda **_kwargs: (
            0.2,
            {
                "feature_count": 1,
                "calibration_slope": 1.0,
                "optimizer_iterations": 1,
                "optimizer_gradient_max_abs": 0.0,
            },
        ),
    )
    monkeypatch.setattr(
        fast.recalibration,
        "fit_bounded_recalibration",
        lambda _probabilities, _labels: {"intercept": 0.0, "slope": 1.0},
    )
    draw = fast._fast_draw(prepared, 17)
    assert draw["seeds"]["ratings_development"] == rating_draws[17]["seed"]
    assert (
        draw["sample_digests"]["ratings_development"]
        == rating_draws[17]["sample_digest"]
    )
    assert draw["refit"]["rating_probability_blue"] == 0.55
    assert draw["refit"]["draft_scaled_logit_blue"] == 0.2


def _artifact(monkeypatch: pytest.MonkeyPatch) -> dict:
    rating_draws = [
        {
            "draw_id": draw_id,
            "seed": frozen._seed(draw_id, "ratings-development"),
            "sample_digest": str(draw_id),
            "rating_probability_blue": 0.5,
        }
        for draw_id in range(frozen.RESAMPLES)
    ]
    terminal_draws = [
        {
            "draw_id": draw_id,
            "seeds": {"ratings_development": rating_draws[draw_id]["seed"]},
            "sample_digests": {
                "ratings_development": rating_draws[draw_id]["sample_digest"]
            },
            "refit": {"rating_probability_blue": 0.5},
        }
        for draw_id in range(frozen.RESAMPLES)
    ]
    rating = {
        "artifact_sha256": "1" * 64,
        "draws_sha256": "2" * 64,
        "draws": rating_draws,
        "inputs": {
            "rating_refit_locator": "data/lol/v2/evaluation/rating-deployment/refits/test.json",
            "rating_refit_raw_sha256": "3" * 64,
            "rating_refit_artifact_sha256": "6" * 64,
        },
        "point_rating_probability_blue": 0.5,
        "event": {"event_start_utc": "2026-09-01T18:00:00+00:00"},
    }
    candidate = {
        "event": {"target_prediction_locator": "prediction.json"},
        "inputs": {
            "recalibration_artifact_locator": "recalibration.json",
            "rating_refit_locator": "data/lol/v2/evaluation/rating-deployment/refits/test.json",
            "rating_refit_raw_sha256": "3" * 64,
            "rating_refit_artifact_sha256": "6" * 64,
        },
        "point_calculation": {"rating_probability_blue": 0.5},
        "uncertainty": {"draws_sha256": "4" * 64, "draws": terminal_draws},
    }
    target = {
        "captured_at_utc": "2026-09-01T14:00:00+00:00",
        "input_receipts": {},
    }
    monkeypatch.setattr(
        fast.frozen,
        "validate_event_uncertainty_candidate",
        lambda candidate_value, **_kwargs: candidate_value,
    )
    monkeypatch.setattr(
        fast.evaluation,
        "_read_regular",
        lambda *_args, **_kwargs: b"recalibration",
    )
    monkeypatch.setattr(
        fast.recalibration,
        "validate_phase_one_recalibration_artifact",
        lambda _value: {
            "models": {"ratings_only": {"intercept": 0.0, "slope": 1.0}}
        },
    )
    monkeypatch.setattr(
        fast.evaluation,
        "_strict_object",
        lambda _raw, _label: {},
    )
    monkeypatch.setattr(
        fast,
        "_rating_artifact",
        lambda _root, _locator, _environment: ("rating.json", b"rating", rating),
    )
    monkeypatch.setattr(
        fast.frozen,
        "_target",
        lambda _root, _locator: (b"target", target, {}, {}),
    )
    monkeypatch.setattr(
        fast.frozen,
        "_fresh_refit",
        lambda _root, _locator, _environment: (
            "data/lol/v2/evaluation/rating-deployment/refits/test.json",
            b"refit",
            {},
            {},
        ),
    )
    monkeypatch.setattr(
        fast.frozen, "_assert_target_refit_binding", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        fast,
        "_source_locks",
        lambda _root: [{"locator": "fast.py", "bytes": 1, "raw_sha256": "5" * 64}],
    )
    payload = {
        "schema_version": fast.SCHEMA_VERSION,
        "result_state": fast.RESULT_STATE,
        "built_at_utc": "2026-09-01T15:00:00+00:00",
        "clock_attestation": {
            "source": "system_utc_clock_sampled_after_all_terminal_draws",
            "observed_wall_clock_utc": "2026-09-01T15:00:00+00:00",
            "user_supplied_timestamp_allowed": False,
            "all_draws_completed_before_observation": True,
        },
        "frozen_contract_candidate": candidate,
        "evaluation_comparator": {
            "model": "recalibrated_rating_only",
            "raw_probability_blue": 0.5,
            "recalibration_intercept": 0.0,
            "recalibration_slope": 1.0,
            "probability_blue": 0.5,
            "phase_two_market_price_used": False,
            "target_event_outcome_used": False,
        },
        "decomposition": {
            "rating_bootstrap_locator": "rating.json",
            "rating_bootstrap_raw_sha256": fast._sha256_bytes(b"rating"),
            "rating_bootstrap_artifact_sha256": "1" * 64,
            "rating_draws_sha256": "2" * 64,
            "terminal_draws_sha256": "4" * 64,
            "same_draw_ids_seeds_and_rating_probabilities_verified": True,
            "exact_slow_path_parity_independently_registered": False,
        },
        "source_locks": fast._source_locks(None),
        "authority": dict(fast.AUTHORITY),
        "claim_ceiling": fast.CLAIM_CEILING,
    }
    payload["artifact_sha256"] = fast._canonical_sha256(payload)
    return payload


def _resign(payload: dict) -> None:
    payload["artifact_sha256"] = fast._canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )


def test_fast_artifact_rejects_rating_terminal_draw_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _artifact(monkeypatch)
    checked = fast.validate_fast_event_uncertainty_v1(payload)
    assert checked["authority"]["probability_authority"] is False

    changed = deepcopy(payload)
    changed["frozen_contract_candidate"]["uncertainty"]["draws"][9]["refit"][
        "rating_probability_blue"
    ] = 0.6
    _resign(changed)
    with pytest.raises(
        fast.FastEventUncertaintyError,
        match="draw parity changed",
    ):
        fast.validate_fast_event_uncertainty_v1(changed)
