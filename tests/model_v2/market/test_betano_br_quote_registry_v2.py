from __future__ import annotations

from copy import deepcopy

import pytest

from lol_kills.v2.market import betano_br_quote_registry_v2 as registry


def _entry() -> dict:
    return {
        "event_id": "event-1", "series_id": "series-1", "game_number": 1,
        "league": "LCS", "market_type": "match_winner",
        "patch": "26.17", "roster_change_stratum": "UNCHANGED",
        "sparse_or_new_champion_map": False,
        "selection": "winner:blue", "opposing_selection": "winner:red",
        "qualification_locator": "qualified.json",
        "qualification_raw_sha256": "1" * 64,
        "qualification_artifact_sha256": "2" * 64,
        "event_plan_locator": "plan.json", "event_plan_raw_sha256": "9" * 64,
        "event_plan_artifact_sha256": "a" * 64,
        "quote_locator": "quote.json", "quote_raw_sha256": "3" * 64,
        "quote_artifact_sha256": "4" * 64,
        "generic_quote_receipt_sha256": "5" * 64,
        "event_probability_artifact_sha256": "6" * 64,
        "map_start_locator": "start.json", "map_start_raw_sha256": "7" * 64,
        "map_start_artifact_sha256": "8" * 64,
        "quote_response_received_at_utc": "2026-09-01T15:00:02+00:00",
        "actual_map_start_utc": "2026-09-01T15:00:10+00:00",
        "response_to_start_seconds": 8.0,
    }


def _payload() -> dict:
    entries = [_entry()]
    return {
        "schema_version": registry.SCHEMA_VERSION,
        "registry_id": "quote-registry-1",
        "status": "QUALIFIED_BETANO_QUOTE_IDENTITIES_REGISTERED",
        "issued_at_utc": "2026-09-01T16:00:00+00:00",
        "independent_review": {
            "reviewer_id": "independent-market-reviewer",
            "reviewed_at_utc": "2026-09-01T15:30:00+00:00",
            "attestation": dict(registry.REVIEW_ATTESTATION),
        },
        "entries": entries,
        "decision": {
            "qualified_quote_receipts_independently_registered": True,
            "registered_quotes": 1,
            "odds_accuracy_authorized": False,
            "betting_authorized": False,
        },
        "authority": dict(registry.AUTHORITY),
        "claim_ceiling": registry.CLAIM_CEILING,
    }


def test_registry_grants_only_qualified_quote_identity() -> None:
    checked = registry.validate_betano_quote_registry_v2(
        _payload(), expected=[_entry()]
    )
    assert checked["authority"]["quote_identity_authority"] is True
    assert checked["authority"]["odds_accuracy_authority"] is False
    assert checked["authority"]["betting_authority"] is False


def test_registry_rejects_timing_or_authority_forgery() -> None:
    late = deepcopy(_payload())
    late["entries"][0]["response_to_start_seconds"] = 4.999
    with pytest.raises(registry.BetanoQuoteRegistryV2Error):
        registry.validate_betano_quote_registry_v2(late, expected=late["entries"])

    authority = deepcopy(_payload())
    authority["authority"]["betting_authority"] = True
    with pytest.raises(
        registry.BetanoQuoteRegistryV2Error, match="exceeds authority"
    ):
        registry.validate_betano_quote_registry_v2(
            authority, expected=[_entry()]
        )
