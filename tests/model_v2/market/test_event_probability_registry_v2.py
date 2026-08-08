from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from lol_kills.v2.market import event_probability_registry_v2 as registry


def _entries() -> list[dict]:
    return [
        {
            "event_id": "event-1",
            "series_id": "series-1",
            "game_number": 1,
            "league": "LCS",
            "patch": "26.17",
            "roster_change_stratum": "UNCHANGED",
            "sparse_or_new_champion_map": False,
            "market_type": "match_winner",
            "selection": "winner:blue",
            "opposing_selection": "winner:red",
            "captured_at_utc": "2026-09-01T15:00:00+00:00",
            "receipt_locator": "data/probability.json",
            "receipt_raw_sha256": "1" * 64,
            "receipt_artifact_sha256": "2" * 64,
            "receipt_sha256": "3" * 64,
            "fast_uncertainty_artifact_sha256": "4" * 64,
            "draws_sha256": "5" * 64,
            "probability": 0.8,
            "rating_only_probability": 0.7,
            "probability_interval": [0.2, 0.4],
            "point_inside_percentile_interval": False,
        }
    ]


def _receipt() -> dict:
    return {
        "schema_version": registry.SCHEMA_VERSION,
        "registry_id": "probability-registry-1",
        "status": "EVENT_PROBABILITY_IDENTITIES_REGISTERED",
        "issued_at_utc": "2026-09-01T16:00:00+00:00",
        "independent_review": {
            "reviewer_id": "independent-probability-reviewer",
            "reviewed_at_utc": "2026-09-01T15:30:00+00:00",
            "attestation": dict(registry.REVIEW_ATTESTATION),
        },
        "entries": _entries(),
        "decision": {
            "event_probability_receipts_independently_registered": True,
            "registered_receipts": 1,
            "probability_accuracy_authorized": False,
            "betting_authorized": False,
        },
        "authority": dict(registry.AUTHORITY),
        "claim_ceiling": registry.CLAIM_CEILING,
    }


def test_registry_preserves_noncontaining_interval_and_grants_identity_only() -> None:
    checked = registry.validate_event_probability_registry_v2(
        _receipt(), expected=_entries()
    )
    assert checked["entries"][0]["point_inside_percentile_interval"] is False
    assert checked["authority"]["event_probability_identity_authority"] is True
    assert checked["authority"]["probability_accuracy_authority"] is False
    assert checked["authority"]["betting_authority"] is False


def test_registry_rejects_receipt_or_review_tamper() -> None:
    changed = _receipt()
    changed["entries"][0]["probability"] = 0.9
    with pytest.raises(
        registry.EventProbabilityRegistryV2Error,
        match="entries changed",
    ):
        registry.validate_event_probability_registry_v2(
            changed, expected=_entries()
        )
    review = _receipt()
    review["independent_review"]["attestation"] = dict(
        review["independent_review"]["attestation"]
    )
    review["independent_review"]["attestation"][
        "percentile_interval_point_containment_not_required_or_falsified"
    ] = False
    with pytest.raises(
        registry.EventProbabilityRegistryV2Error,
        match="review is incomplete",
    ):
        registry.validate_event_probability_registry_v2(
            review, expected=_entries()
        )


def test_registry_loader_requires_exact_raw_external_pin(tmp_path: Path) -> None:
    receipt = _receipt()
    raw = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    path = tmp_path / "registry.json"
    path.write_bytes(raw)
    loaded = registry.load_pinned_event_probability_registry_v2(
        path=path,
        external_sha256=hashlib.sha256(raw).hexdigest(),
        expected=_entries(),
    )
    assert loaded["event_probability_identity_authority"] is True
    assert loaded["betting_authorized"] is False
    with pytest.raises(
        registry.EventProbabilityRegistryV2Error,
        match="external pin",
    ):
        registry.load_pinned_event_probability_registry_v2(
            path=path,
            external_sha256="0" * 64,
            expected=_entries(),
        )
