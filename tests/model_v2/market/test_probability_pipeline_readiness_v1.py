from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from lol_kills.v2.market import probability_pipeline_readiness_v1 as readiness
from lol_kills.v2.market import (
    probability_pipeline_readiness_registry_v1 as registry,
)


ROOT = Path(".").resolve()
LOCKED_AT = datetime(2026, 8, 2, 5, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def receipt() -> dict:
    return readiness.build_probability_pipeline_readiness_v1(
        root=ROOT,
        clock=lambda: LOCKED_AT,
    )


def _resign(payload: dict) -> None:
    payload["artifact_sha256"] = readiness._canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )


def test_readiness_freezes_real_full_pipeline_and_no_authority(receipt: dict) -> None:
    checked = readiness.validate_probability_pipeline_readiness_v1(
        receipt, root=ROOT
    )
    contract = checked["probability_pipeline_contract"]
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
    assert all(value == 0 for key, value in checked["locked_empty_state"].items() if key.endswith("artifacts") or key.endswith("outputs") or key.endswith("cohorts"))
    assert checked["locked_empty_state"]["outcomes_accessed"] is False
    assert all(value is False for value in checked["authority"].values())
    assert all(value is None for value in checked["decision_outputs"].values())


def test_readiness_rejects_boundary_lock() -> None:
    with pytest.raises(
        readiness.ProbabilityPipelineReadinessError,
        match="not frozen pre-boundary",
    ):
        readiness.build_probability_pipeline_readiness_v1(
            root=ROOT,
            clock=lambda: datetime(2026, 8, 3, 0, 0, tzinfo=timezone.utc),
        )


def test_readiness_rejects_forged_contract_source_and_authority(
    receipt: dict,
) -> None:
    forged_contract = deepcopy(receipt)
    forged_contract["probability_pipeline_contract"]["uncertainty"][
        "resamples"
    ] = 1
    _resign(forged_contract)
    with pytest.raises(
        readiness.ProbabilityPipelineReadinessError,
        match="contract changed",
    ):
        readiness.validate_probability_pipeline_readiness_v1(
            forged_contract, root=ROOT
        )

    forged_source = deepcopy(receipt)
    forged_source["source_locks"][0]["raw_sha256"] = "0" * 64
    _resign(forged_source)
    with pytest.raises(
        readiness.ProbabilityPipelineReadinessError,
        match="source lock changed",
    ):
        readiness.validate_probability_pipeline_readiness_v1(
            forged_source, root=ROOT
        )

    forged_authority = deepcopy(receipt)
    forged_authority["authority"]["betting_authority"] = True
    _resign(forged_authority)
    with pytest.raises(
        readiness.ProbabilityPipelineReadinessError,
        match="exceeds authority",
    ):
        readiness.validate_probability_pipeline_readiness_v1(
            forged_authority, root=ROOT
        )


def test_readiness_write_is_no_clobber(tmp_path: Path, receipt: dict) -> None:
    path = tmp_path / "readiness.json"
    raw_sha256 = readiness.write_no_clobber(path, receipt)
    assert len(raw_sha256) == 64
    assert json.loads(path.read_text()) == receipt
    with pytest.raises(
        readiness.ProbabilityPipelineReadinessError,
        match="refusing to replace",
    ):
        readiness.write_no_clobber(path, receipt)


def test_registered_readiness_is_exactly_hash_pinned() -> None:
    checked = registry.validate_registered_probability_pipeline_readiness_v1(
        root=ROOT
    )
    assert (
        checked["artifact_sha256"]
        == registry.REGISTERED_READINESS_ARTIFACT_SHA256
    )
    assert checked["locked_at_utc"] == registry.REGISTERED_READINESS_LOCKED_AT_UTC
    assert checked["locked_empty_state"]["outcomes_accessed"] is False
    assert all(value is False for value in checked["authority"].values())
