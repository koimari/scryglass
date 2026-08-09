from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from lol_kills.v2.market import event_rating_bootstrap_v1 as rating_bootstrap
from lol_kills.v2.market import full_pipeline_uncertainty_v1 as frozen


def _artifact(monkeypatch: pytest.MonkeyPatch) -> dict:
    population = 3
    event = {
        "event_id": "verification-event",
        "event_start_utc": "2026-09-01T18:00:00+00:00",
    }
    refit = {
        "built_at_utc": "2026-09-01T12:00:00+00:00",
        "artifact_sha256": "6" * 64,
        "event": event,
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
            "result_locator": "result.json",
            "result_raw_sha256": rating_bootstrap._sha256_bytes(
                b"phase-one-result"
            ),
            "result_artifact_sha256": "3" * 64,
        },
    }
    result = {
        "artifact_sha256": "3" * 64,
        "phase_one_models_passed": True,
    }
    monkeypatch.setattr(
        rating_bootstrap.evaluation,
        "_read_regular",
        lambda _root, locator, _label: (
            b"phase-one-result" if locator == "result.json" else b"unexpected"
        ),
    )
    monkeypatch.setattr(
        rating_bootstrap.evaluation,
        "_strict_object",
        lambda _raw, _label: {},
    )
    monkeypatch.setattr(
        rating_bootstrap.evaluation,
        "validate_phase_one_evaluation_result",
        lambda _payload: result,
    )
    monkeypatch.setattr(
        rating_bootstrap,
        "_prepare",
        lambda _root, _locator, _environment: {
            "locator": "data/lol/v2/evaluation/rating-deployment/refits/test.json",
            "raw": b"refit",
            "refit": refit,
            "replay": {
                "input_data": type(
                    "Input", (), {"development_series": list(range(population))}
                )()
            },
            "point_probability_blue": 0.55,
        },
    )
    monkeypatch.setattr(
        rating_bootstrap,
        "_source_locks",
        lambda _root: [{"locator": "source.py", "bytes": 1, "raw_sha256": "7" * 64}],
    )
    draws = []
    for draw_id in range(frozen.RESAMPLES):
        indices = frozen._sample_indices(
            population, draw_id=draw_id, stream="ratings-development"
        )
        draws.append(
            {
                "draw_id": draw_id,
                "seed": frozen._seed(draw_id, "ratings-development"),
                "sample_digest": frozen._sample_digest(indices),
                "rating_probability_blue": 0.4 + (draw_id % 100) / 500.0,
            }
        )
    payload = {
        "schema_version": rating_bootstrap.SCHEMA_VERSION,
        "result_state": rating_bootstrap.RESULT_STATE,
        "built_at_utc": "2026-09-01T15:00:00+00:00",
        "clock_attestation": {
            "source": "system_utc_clock_sampled_after_all_rating_draws",
            "observed_wall_clock_utc": "2026-09-01T15:00:00+00:00",
            "user_supplied_timestamp_allowed": False,
            "all_draws_completed_before_observation": True,
        },
        "event": event,
        "inputs": {
            "phase_one_result_locator": "result.json",
            "phase_one_result_raw_sha256": rating_bootstrap._sha256_bytes(
                b"phase-one-result"
            ),
            "phase_one_result_artifact_sha256": "3" * 64,
            "rating_refit_locator": "data/lol/v2/evaluation/rating-deployment/refits/test.json",
            "rating_refit_raw_sha256": rating_bootstrap._sha256_bytes(b"refit"),
            "rating_refit_artifact_sha256": "6" * 64,
            "rating_source_snapshot_locator": refit["source_snapshot"]["locator"],
            "rating_source_snapshot_raw_sha256": "8" * 64,
            "rating_source_snapshot_artifact_sha256": "9" * 64,
            "rating_roster_raw_sha256": "a" * 64,
            "rating_roster_canonical_sha256": "b" * 64,
            "rating_patch_raw_sha256": "c" * 64,
        },
        "bootstrap_contract": {
            "method": "fresh_post_validation_full_pipeline_rating_leg",
            "resamples": frozen.RESAMPLES,
            "master_seed": frozen.MASTER_SEED,
            "stream": "ratings-development",
            "development_series": population,
            "series_with_replacement_preserve_chronological_order": True,
            "rating_state_refit_in_each_draw": True,
            "target_event_outcome_or_market_price_used": False,
        },
        "point_rating_probability_blue": 0.55,
        "draws": draws,
        "draws_sha256": rating_bootstrap._canonical_sha256(draws),
        "source_locks": rating_bootstrap._source_locks(Path(".")),
        "qualification": {
            "phase_one_models_independently_passed": True,
            "fresh_refit_created_after_registered_phase_one_pass": True,
            "fresh_source_roster_patch_and_point_replayed": True,
            "terminal_draft_used": False,
            "target_event_outcome_present": False,
            "target_event_outcome_accessed": False,
            "market_price_used": False,
            "independently_registered": False,
        },
        "authority": dict(rating_bootstrap.AUTHORITY),
        "claim_ceiling": rating_bootstrap.CLAIM_CEILING,
    }
    payload["artifact_sha256"] = rating_bootstrap._canonical_sha256(payload)
    return payload


def _resign(payload: dict) -> None:
    payload["artifact_sha256"] = rating_bootstrap._canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )


def test_rating_leg_replays_frozen_draw_ids_and_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _artifact(monkeypatch)
    checked = rating_bootstrap.validate_event_rating_bootstrap_v1(payload)
    assert len(checked["draws"]) == 2_000
    assert checked["draws"][71]["seed"] == frozen._seed(
        71, "ratings-development"
    )
    assert checked["authority"]["probability_authority"] is False


def test_rating_leg_rejects_seed_probability_and_source_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = deepcopy(_artifact(monkeypatch))
    seed["draws"][0]["seed"] += 1
    seed["draws_sha256"] = rating_bootstrap._canonical_sha256(seed["draws"])
    _resign(seed)
    with pytest.raises(
        rating_bootstrap.EventRatingBootstrapError,
        match="does not replay",
    ):
        rating_bootstrap.validate_event_rating_bootstrap_v1(seed)

    probability = deepcopy(_artifact(monkeypatch))
    probability["draws"][1]["rating_probability_blue"] = 1.0
    probability["draws_sha256"] = rating_bootstrap._canonical_sha256(
        probability["draws"]
    )
    _resign(probability)
    with pytest.raises(
        rating_bootstrap.EventRatingBootstrapError,
        match="does not replay",
    ):
        rating_bootstrap.validate_event_rating_bootstrap_v1(probability)

    source = deepcopy(_artifact(monkeypatch))
    source["source_locks"][0]["raw_sha256"] = "0" * 64
    _resign(source)
    with pytest.raises(
        rating_bootstrap.EventRatingBootstrapError,
        match="source lock changed",
    ):
        rating_bootstrap.validate_event_rating_bootstrap_v1(source)

    point = deepcopy(_artifact(monkeypatch))
    point["point_rating_probability_blue"] = 0.56
    _resign(point)
    with pytest.raises(
        rating_bootstrap.EventRatingBootstrapError,
        match="point rating probability binding changed",
    ):
        rating_bootstrap.validate_event_rating_bootstrap_v1(point)
