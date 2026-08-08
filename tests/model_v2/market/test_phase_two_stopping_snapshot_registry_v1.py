from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from lol_kills.v2.market import phase_two_stopping_snapshot_registry_v1 as registry


def _binding() -> dict:
    return {
        "snapshot_locator": "snapshot.json",
        "snapshot_raw_sha256": "1" * 64,
        "snapshot_artifact_sha256": "2" * 64,
        "entries_sha256": "3" * 64,
        "captured_at_utc": "2026-10-01T15:00:00+00:00",
        "eligible_quoted_maps": 500,
        "otherwise_eligible_maps": 600,
        "eligible_series": 125,
        "quote_coverage": 500 / 600,
        "shadow_policy_qualifying_maps": 100,
        "support_met": True,
        "outcomes_accessed": False,
    }


def _payload() -> dict:
    return {
        "schema_version": registry.SCHEMA_VERSION,
        "registry_id": "phase-two-snapshot-registry-1",
        "status": "FIRST_SUPPORT_MET_PHASE_TWO_SNAPSHOT_REGISTERED_OUTCOMES_UNOPENED",
        "issued_at_utc": "2026-10-01T16:00:00+00:00",
        "independent_review": {
            "reviewer_id": "independent-snapshot-reviewer",
            "reviewed_at_utc": "2026-10-01T15:30:00+00:00",
            "attestation": dict(registry.REVIEW_ATTESTATION),
        },
        "snapshot_binding": _binding(),
        "decision": {
            "first_support_met_snapshot_independently_registered": True,
            "phase_two_outcomes_opened": False,
            "evaluation_authorized": False,
            "betting_authorized": False,
        },
        "authority": dict(registry.AUTHORITY),
        "claim_ceiling": registry.CLAIM_CEILING,
    }


def test_registry_grants_snapshot_identity_only() -> None:
    checked = registry.validate_phase_two_snapshot_registry_v1(
        _payload(), expected_binding=_binding()
    )
    assert checked["authority"]["phase_two_snapshot_identity_authority"] is True
    assert checked["authority"]["phase_two_outcome_opening_authority"] is False
    assert checked["authority"]["betting_authority"] is False


def test_registry_rejects_review_tamper_and_external_pin(tmp_path) -> None:
    forged = deepcopy(_payload())
    forged["independent_review"]["attestation"][
        "this_is_the_first_snapshot_meeting_all_minima"
    ] = False
    with pytest.raises(registry.PhaseTwoSnapshotRegistryError, match="incomplete"):
        registry.validate_phase_two_snapshot_registry_v1(
            forged, expected_binding=_binding()
        )

    raw = (json.dumps(_payload(), indent=2, sort_keys=True) + "\n").encode()
    path = tmp_path / "registry.json"
    path.write_bytes(raw)
    loaded = registry.load_pinned_phase_two_snapshot_registry_v1(
        path=path,
        external_sha256=hashlib.sha256(raw).hexdigest(),
        expected_binding=_binding(),
    )
    assert loaded["phase_two_snapshot_identity_authority"] is True
    assert loaded["betting_authorized"] is False
