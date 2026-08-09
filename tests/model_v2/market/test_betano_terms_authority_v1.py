from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from lol_kills.v2.market import betano_terms_authority_v1 as terms


def _install_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        terms,
        "validate_registered_match_winner_future_protocol_v1",
        lambda **_kwargs: {
            "settlement_contract": {"settlement_rule_id": "settlement-v1"}
        },
    )
    monkeypatch.setattr(
        terms,
        "validate_registered_betano_terms_snapshot_v1",
        lambda **_kwargs: {
            "coverage": {"complete_bookmaker_terms_snapshot": False}
        },
    )


def _receipt(tmp_path: Path) -> dict:
    evidence_path = tmp_path / terms.EVIDENCE_PREFIX / "rules.html"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_raw = b"independent Betano rules evidence"
    evidence_path.write_bytes(evidence_raw)
    source_id = "betano-rules-2026-09-01"
    rule_names = {
        "market_label",
        "winning_selection",
        "non_started_map",
        "same_day_resumption",
        "postponement",
        "cancellation",
        "remake_or_restart",
        "forfeit_walkover_or_disqualification",
        "void_refund_cash_odds_treatment",
        "ambiguous_or_conflicting_result",
    }
    return {
        "schema_version": terms.SCHEMA_VERSION,
        "registry_id": "betano-terms-independent-1",
        "status": "COMPLETE_TERMS_INDEPENDENTLY_REGISTERED",
        "registered_at_utc": "2026-09-01T15:00:00+00:00",
        "scope": "betano_brazil_league_of_legends_single_map_winner_cash_odds",
        "protocol_binding": {
            "artifact_sha256": terms.REGISTERED_PROTOCOL_ARTIFACT_SHA256,
            "settlement_contract_sha256": terms.REGISTERED_SETTLEMENT_CONTRACT_SHA256,
            "settlement_rule_id": "settlement-v1",
        },
        "incomplete_public_snapshot_binding": {
            "locator": terms.REGISTERED_SNAPSHOT_LOCATOR.as_posix(),
            "raw_sha256": terms.REGISTERED_SNAPSHOT_RAW_SHA256,
            "artifact_sha256": terms.REGISTERED_SNAPSHOT_ARTIFACT_SHA256,
            "acknowledged_incomplete": True,
        },
        "complete_terms_evidence": [
            {
                "source_id": source_id,
                "source_url": "https://www.betano.bet.br/help/rules/",
                "locator": (terms.EVIDENCE_PREFIX / "rules.html").as_posix(),
                "raw_sha256": hashlib.sha256(evidence_raw).hexdigest(),
                "captured_at_utc": "2026-09-01T12:00:00+00:00",
                "effective_at_utc": "2026-08-01T00:00:00+00:00",
                "language": "pt-BR",
                "access_method": "fresh_unauthenticated_browser",
                "account_or_credentials_embedded": False,
                "coverage": sorted(terms.REQUIRED_COVERAGE),
            }
        ],
        "resolved_settlement_rules": {
            **{name: f"resolved {name}" for name in rule_names},
            "manual_post_outcome_override_permitted": False,
            "supporting_source_ids_by_rule": {
                name: [source_id] for name in rule_names
            },
        },
        "reviews": [
            {
                "review_scope": scope,
                "reviewer_id": f"reviewer-{index}",
                "reviewed_at_utc": f"2026-09-01T14:0{index}:00+00:00",
                "attestation": dict(attestation),
            }
            for index, (scope, attestation) in enumerate(
                terms.REVIEW_SCOPES.items(), start=1
            )
        ],
        "decision": {
            "complete_bookmaker_terms_snapshot": True,
            "independent_alignment_review_present": True,
            "settlement_contract_resolved": True,
            "phase_two_opened": False,
            "betting_authorized": False,
        },
        "authority": dict(terms.AUTHORITY),
        "claim_ceiling": terms.CLAIM_CEILING,
    }


def test_complete_terms_require_evidence_and_two_reviews(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_dependencies(monkeypatch)
    receipt = _receipt(tmp_path)
    checked = terms.validate_betano_terms_authority_v1(receipt, root=tmp_path)
    assert checked["decision"]["settlement_contract_resolved"] is True
    assert checked["authority"]["phase_two_opening_authority"] is False
    assert checked["authority"]["betting_authority"] is False


def test_complete_terms_reject_missing_coverage_or_rule_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_dependencies(monkeypatch)
    missing = _receipt(tmp_path)
    missing["complete_terms_evidence"][0]["coverage"].pop()
    with pytest.raises(terms.BetanoTermsAuthorityError, match="incomplete"):
        terms.validate_betano_terms_authority_v1(missing, root=tmp_path)

    unsupported = _receipt(tmp_path)
    unsupported["resolved_settlement_rules"]["supporting_source_ids_by_rule"][
        "cancellation"
    ] = []
    with pytest.raises(terms.BetanoTermsAuthorityError, match="mapping changed"):
        terms.validate_betano_terms_authority_v1(unsupported, root=tmp_path)


def test_complete_terms_reject_same_reviewer_or_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_dependencies(monkeypatch)
    same = _receipt(tmp_path)
    same["reviews"][1]["reviewer_id"] = same["reviews"][0]["reviewer_id"]
    with pytest.raises(terms.BetanoTermsAuthorityError, match="not independent"):
        terms.validate_betano_terms_authority_v1(same, root=tmp_path)

    unsafe = _receipt(tmp_path)
    unsafe["complete_terms_evidence"][0]["account_or_credentials_embedded"] = True
    with pytest.raises(terms.BetanoTermsAuthorityError, match="safety"):
        terms.validate_betano_terms_authority_v1(unsafe, root=tmp_path)


def test_complete_terms_loader_requires_external_raw_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_dependencies(monkeypatch)
    receipt = _receipt(tmp_path)
    raw = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    path = tmp_path / "registry.json"
    path.write_bytes(raw)
    loaded = terms.load_pinned_betano_terms_authority_v1(
        path=path,
        external_sha256=hashlib.sha256(raw).hexdigest(),
        root=tmp_path,
    )
    assert loaded["settlement_contract_resolved"] is True
    with pytest.raises(terms.BetanoTermsAuthorityError, match="external pin"):
        terms.load_pinned_betano_terms_authority_v1(
            path=path,
            external_sha256="0" * 64,
            root=tmp_path,
        )
