from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from lol_kills.v2.market import probability_pipeline_readiness_v1 as readiness
from lol_kills.v2.market import (
    probability_pipeline_readiness_registry_v1 as registry,
)


LOCKED_AT = datetime(2026, 8, 2, 5, 0, tzinfo=timezone.utc)


def test_probability_pipeline_contract_remains_bounded_and_non_authorizing() -> None:
    contract = readiness._contract()
    assert contract["uncertainty"]["resamples"] == 2_000
    assert contract["uncertainty"]["ratings_state_refit_in_each_resample"] is True
    assert contract["uncertainty"]["draft_terms_refit_in_each_resample"] is True
    assert (
        contract["uncertainty"]["phase_one_recalibration_refit_in_each_resample"]
        is True
    )
    assert (
        contract["uncertainty"]
        ["phase_one_stored_predictions_used_for_recalibration_refit"]
        is True
    )
    fresh_refit = contract["fresh_post_validation_rating_refit"]
    assert fresh_refit["required_before_every_phase_two_event_prediction"] is True
    assert fresh_refit["strict_target_event_cutoff"] is True
    assert fresh_refit["maximum_data_age_seconds"] == 14 * 24 * 60 * 60
    assert fresh_refit["cross_team_covariance_retained"] is True
    assert fresh_refit["unidentified_synergy_and_policy_remain_null"] is True
    assert (
        fresh_refit["full_pipeline_uncertainty_binding_status"]
        == "wired_replayed_and_independently_reviewable"
    )
    assert contract["uncertainty"]["fresh_post_validation_refit_exactly_bound"] is True
    assert contract["uncertainty"]["slow_and_fast_paths_share_exact_rating_draws"] is True
    assert all(value is False for value in readiness.AUTHORITY.values())


def test_readiness_stays_closed_on_expired_evaluation_dependency(
    historical_capture_root: Path,
) -> None:
    with pytest.raises(
        readiness.ProbabilityPipelineReadinessError,
        match="dependency is invalid",
    ):
        readiness.build_probability_pipeline_readiness_v1(
            root=historical_capture_root,
            clock=lambda: LOCKED_AT,
        )


def test_readiness_rejects_boundary_lock(historical_capture_root: Path) -> None:
    with pytest.raises(
        readiness.ProbabilityPipelineReadinessError,
        match="not frozen pre-boundary",
    ):
        readiness.build_probability_pipeline_readiness_v1(
            root=historical_capture_root,
            clock=lambda: datetime(2026, 8, 3, 0, 0, tzinfo=timezone.utc),
        )


def test_readiness_write_is_no_clobber(tmp_path: Path) -> None:
    receipt = {"status": "BLOCKED_EXPIRED_DEPENDENCY"}
    path = tmp_path / "readiness.json"
    raw_sha256 = readiness.write_no_clobber(path, receipt)
    assert len(raw_sha256) == 64
    assert json.loads(path.read_text()) == receipt
    with pytest.raises(
        readiness.ProbabilityPipelineReadinessError,
        match="refusing to replace",
    ):
        readiness.write_no_clobber(path, receipt)


def test_registered_readiness_stays_closed_on_expired_dependency(
    historical_capture_root: Path,
) -> None:
    with pytest.raises(
        registry.RegisteredProbabilityPipelineReadinessError,
        match="invalid",
    ):
        registry.validate_registered_probability_pipeline_readiness_v1(
            root=historical_capture_root
        )
