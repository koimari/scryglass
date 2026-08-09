from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from lol_kills.v2.market import phase_two_outcome_opening_v1 as opening


def _authority(expected: dict) -> dict:
    return {
        "schema_version": opening.SCHEMA_VERSION,
        "authority_id": "authority-1",
        "status": "APPROVED",
        "scope": "ONE_TIME_PHASE_TWO_MARKET_EVALUATION_ONLY",
        "reviewed_at_utc": "2026-10-02T12:00:00+00:00",
        "reviews": [
            {
                "review_scope": scope,
                "reviewer_id": f"reviewer-{index}",
                "reviewed_at_utc": "2026-10-02T11:00:00+00:00",
                "attestation": attestation,
            }
            for index, (scope, attestation) in enumerate(
                opening.REVIEW_SCOPES.items(), start=1
            )
        ],
        "bindings": expected,
        "sealed_outcomes": {
            "cohort_locator": (
                "data/lol/v2/evaluation/match-winner-market-v1/phase-two/"
                "outcomes-v1/cohort.json"
            ),
            "cohort_raw_sha256": "a" * 64,
            "custodian_id": "custodian-1",
            "sealed_at_utc": "2026-10-02T10:00:00+00:00",
            "custodian_attestation": {
                "digest_created_without_disclosing_outcomes_to_model_or_capture_authors_or_reviewers": True,
                "cohort_bytes_immutable_after_digest": True,
                "cohort_exactly_matches_every_map_in_the_registered_snapshot": True,
                "no_manual_post_outcome_exclusion_or_replacement": True,
            },
        },
        "one_time_run": {
            "run_id": "run-1",
            "opening_marker_locator": (
                "data/lol/v2/evaluation/match-winner-market-v1/phase-two/"
                "outcome-opening-markers-v1/run-1.json"
            ),
            "authorized_output_locator": (
                "data/lol/v2/evaluation/match-winner-market-v1/phase-two/"
                "results-v1/run-1.json"
            ),
            "marker_written_before_outcome_read": True,
            "atomic_no_clobber_output_required": True,
            "partial_result_publication_prohibited": True,
            "second_opening_or_replacement_cohort_prohibited": True,
        },
        "authority": dict(opening.AUTHORITY),
        "claim_ceiling": opening.CLAIM_CEILING,
    }


def test_opening_requires_two_distinct_reviewers() -> None:
    expected = {"snapshot": {"artifact_sha256": "1" * 64}}
    receipt = _authority(expected)
    assert opening.validate_outcome_opening_authority_v1(
        receipt, expected_bindings=expected
    ) == receipt
    forged = deepcopy(receipt)
    forged["reviews"][1]["reviewer_id"] = forged["reviews"][0]["reviewer_id"]
    with pytest.raises(
        opening.PhaseTwoOutcomeOpeningError,
        match="not independent",
    ):
        opening.validate_outcome_opening_authority_v1(
            forged, expected_bindings=expected
        )


def test_marker_is_persisted_before_first_outcome_read(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {
        "first_support_met_snapshot_registry": {
            "snapshot_binding": {"snapshot_artifact_sha256": "1" * 64}
        }
    }
    authority = _authority(expected)
    marker_path = tmp_path / authority["one_time_run"]["opening_marker_locator"]
    outcome_raw = b'{"sealed":true}\n'
    authority["sealed_outcomes"]["cohort_raw_sha256"] = opening._sha256(
        outcome_raw
    )
    monkeypatch.setattr(opening, "current_expected_bindings", lambda **_kwargs: expected)
    monkeypatch.setattr(
        opening,
        "load_pinned_outcome_opening_authority_v1",
        lambda **_kwargs: (b"authority-bytes", authority),
    )

    def first_read(_root, _locator, _label):
        assert marker_path.is_file()
        return outcome_raw

    monkeypatch.setattr(opening, "_regular", first_read)
    monkeypatch.setattr(
        opening.evaluation,
        "evaluate_phase_two_v1",
        lambda **_kwargs: {"result": "terminal"},
    )
    result = opening.run_authorized_phase_two_evaluation_v1(
        snapshot_locator="unused.json",
        root=tmp_path,
        environment={opening.EXTERNAL_SHA256_ENV: "b" * 64},
        clock=lambda: datetime(2026, 10, 2, 13, 0, tzinfo=timezone.utc),
    )
    assert result == {"result": "terminal"}
    assert marker_path.is_file()
    assert (
        tmp_path / authority["one_time_run"]["authorized_output_locator"]
    ).is_file()
