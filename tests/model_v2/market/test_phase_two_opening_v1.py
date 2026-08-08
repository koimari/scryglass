from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from lol_kills.v2.market import phase_two_opening_v1 as opening


def _bindings() -> dict:
    return {
        "phase_one_evaluation": {"raw_sha256": "1" * 64},
        "calibration_uncertainty": {"raw_sha256": "2" * 64},
        "bookmaker_terms": {"raw_sha256": "3" * 64},
        "quote_adapter": {"registry_sha256": "4" * 64},
        "phase_two_collection_readiness": {"artifact_sha256": "5" * 64},
    }


def _authority() -> dict:
    return {
        "schema_version": opening.SCHEMA_VERSION,
        "authority_id": "phase-two-opening-1",
        "status": "OUTCOME_FREE_PHASE_TWO_COLLECTION_APPROVED",
        "issued_at_utc": "2026-09-01T15:00:00+00:00",
        "reviews": [
            {
                "review_scope": scope,
                "reviewer_id": f"reviewer-{index}",
                "reviewed_at_utc": f"2026-09-01T14:0{index}:00+00:00",
                "attestation": dict(attestation),
            }
            for index, (scope, attestation) in enumerate(
                opening.REVIEW_SCOPES.items(), start=1
            )
        ],
        "bindings": _bindings(),
        "one_time": {
            "opening_marker_locator": opening.MARKER_LOCATOR.as_posix(),
            "marker_written_before_first_phase_two_probability_or_quote": True,
            "second_opening_or_marker_replacement_prohibited": True,
            "crash_after_marker_does_not_authorize_reopening": True,
        },
        "decision": {
            "outcome_free_phase_two_collection_approved": True,
            "phase_two_outcomes_opened": False,
            "probability_authorized": False,
            "quote_authorized": False,
            "betting_authorized": False,
        },
        "authority": dict(opening.AUTHORITY),
        "claim_ceiling": opening.CLAIM_CEILING,
    }


def test_opening_requires_two_independent_scope_complete_reviews() -> None:
    authority = _authority()
    checked = opening.validate_opening_authority(
        authority, expected_bindings=_bindings()
    )
    assert checked["authority"]["outcome_free_phase_two_collection_authority"] is True
    assert checked["authority"]["probability_authority"] is False
    assert checked["authority"]["betting_authority"] is False

    same = deepcopy(authority)
    same["reviews"][1]["reviewer_id"] = same["reviews"][0]["reviewer_id"]
    with pytest.raises(opening.PhaseTwoOpeningError, match="not independent"):
        opening.validate_opening_authority(same, expected_bindings=_bindings())


def test_marker_binds_exact_external_authority_and_is_no_clobber(
    tmp_path: Path,
) -> None:
    authority = _authority()
    raw = (json.dumps(authority, indent=2, sort_keys=True) + "\n").encode()
    opened = datetime(2026, 9, 1, 16, 0, tzinfo=timezone.utc)
    marker = opening._marker_payload(
        raw_authority=raw, authority=authority, opened_at=opened
    )
    checked = opening.validate_opening_marker(
        marker, raw_authority=raw, authority=authority
    )
    assert checked["outcomes_accessed"] is False
    assert checked["betting_authorized"] is False
    path = tmp_path / "marker.json"
    digest = opening.write_no_clobber(path, marker)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(opening.PhaseTwoOpeningError, match="refusing to replace"):
        opening.write_no_clobber(path, marker)


def test_marker_rejects_reopening_or_probability_claim() -> None:
    authority = _authority()
    raw = json.dumps(authority).encode()
    marker = opening._marker_payload(
        raw_authority=raw,
        authority=authority,
        opened_at=datetime(2026, 9, 1, 16, 0, tzinfo=timezone.utc),
    )
    forged = deepcopy(marker)
    forged["probability_authorized"] = True
    with pytest.raises(opening.PhaseTwoOpeningError, match="marker changed"):
        opening.validate_opening_marker(
            forged, raw_authority=raw, authority=authority
        )


@pytest.mark.parametrize(
    "locator",
    [
        opening.EVENT_PROBABILITY_PREFIX / "probability.json",
        opening.EVENT_PLAN_PREFIX / "plan.json",
        opening.QUOTE_FAILURE_PREFIX / "failure.json",
        opening.ATTEMPT_COMPLETION_PREFIX / "completion.json",
        opening.STOPPING_SNAPSHOT_PREFIX / "snapshot.json",
        opening.STOPPING_SNAPSHOT_REGISTRY_LOCATOR,
        opening.BETANO_QUOTE_PREFIX / "quote.json",
        opening.QUALIFIED_BETANO_QUOTE_PREFIX / "qualified.json",
        opening.BETANO_QUOTE_REGISTRY_LOCATOR,
    ],
)
def test_opening_rejects_any_predating_v2_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, locator: Path
) -> None:
    authority = _authority()
    raw = json.dumps(authority).encode()
    path = tmp_path / locator
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    monkeypatch.setattr(opening, "current_expected_bindings", lambda **_kwargs: _bindings())
    monkeypatch.setattr(
        opening,
        "load_pinned_opening_authority",
        lambda **_kwargs: (raw, authority),
    )
    with pytest.raises(opening.PhaseTwoOpeningError, match="predate opening"):
        opening.consume_phase_two_opening(
            root=tmp_path,
            environment={opening.EXTERNAL_SHA256_ENV: "1" * 64},
            clock=lambda: datetime(2026, 9, 1, 16, 0, tzinfo=timezone.utc),
        )
