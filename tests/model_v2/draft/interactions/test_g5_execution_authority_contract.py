from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path

import pytest

from lol_kills.v2.draft.interactions.g5_exploratory import execution_authority_contract as authority


RUNNER_CORE = "a" * 64


def _permit(run_id: str = "run-1") -> dict:
    value = {
        "schema_id": authority.PERMIT_SCHEMA, "permit_id": "permit-1", "run_id": run_id, "nonce": "nonce-1", "issued_at": "2026-07-30T00:00:00Z", "expires_at": "2026-07-31T00:00:00Z",
        "authority_contract_sha256": authority.AUTHORITY_CONTRACT_SHA256, "reviewed_runner_core_sha256": RUNNER_CORE,
        "prefit": {"contract": authority.PREFIT_CONTRACT_SHA256, "core": authority.PREFIT_CORE_SHA256, "review": authority.PREFIT_REVIEW_SHA256, "source_identities_sha256": authority.FROZEN_SOURCE_IDENTITIES_SHA256},
        "root_authority": dict(authority.ROOT_AUTHORITY),
        "reviewer": {"identity": "KOI_MARI", "independent_from_evidence_generator": True, "scope": authority.REVIEWER_SCOPE},
        "authorization": authority._authorization(),
        "paths": {"claim_locator": authority.CLAIM_LOCATOR, "result_locator": authority.RESULT_LOCATOR, "result_locator_sha256": authority.RESULT_LOCATOR_SHA256},
        "state": "AUTHORIZED_ONCE",
    }
    value["permit_sha256"] = authority.sha256(value)
    return value


def _claim(permit: dict, raw_sha: str) -> dict:
    value = {
        "schema_id": authority.CLAIM_SCHEMA, "permit_id": permit["permit_id"], "run_id": permit["run_id"], "nonce": permit["nonce"],
        "permit_raw_sha256": raw_sha, "permit_canonical_sha256": permit["permit_sha256"],
        "authority_contract_sha256": authority.AUTHORITY_CONTRACT_SHA256, "reviewed_runner_core_sha256": RUNNER_CORE,
        "result_locator": authority.RESULT_LOCATOR, "result_locator_sha256": authority.RESULT_LOCATOR_SHA256, "state": "CLAIMED_CONSUMED",
    }
    value["claim_sha256"] = authority.sha256(value)
    return value


def test_contract_is_tiny_non_authorizing_and_declares_exact_future_api() -> None:
    contract = authority.authority_contract()
    unsigned = dict(contract); claimed = unsigned.pop("artifact_sha256")
    assert claimed == authority.sha256(unsigned)
    assert contract["state"] == "SCHEMA_FROZEN_NO_EXECUTION_AUTHORITY"
    assert contract["root_authority"] == authority.ROOT_AUTHORITY
    assert contract["prefit"]["source_identities"] == authority.FROZEN_SOURCE_IDENTITIES
    assert authority.FROZEN_SOURCE_IDENTITIES_SHA256 == authority.sha256(authority.FROZEN_SOURCE_IDENTITIES)
    assert contract["pending_runner_core"] == authority.PENDING_REVIEWED_RUNNER_CORE
    assert contract["claim_ceiling"]["execution_authorization"] is False
    assert not hasattr(authority, "authenticate_immutable_permit") and not hasattr(authority, "claim_single_use_run")
    assert not (authority.ROOT / authority.PERMIT_LOCATOR).exists()


def test_canonical_raw_pinned_permit_and_claim_validate_in_memory_only() -> None:
    permit = _permit()
    raw = authority.canonical_bytes(permit) + b"\n"
    parsed = authority.validate_permit_bytes(raw, expected_raw_sha256=hashlib.sha256(raw).hexdigest(), expected_runner_core_sha256=RUNNER_CORE, expected_run_id="run-1")
    authority.validate_claim_payload(_claim(parsed, hashlib.sha256(raw).hexdigest()), permit=parsed, permit_raw_sha256=hashlib.sha256(raw).hexdigest(), expected_runner_core_sha256=RUNNER_CORE)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.pop("nonce"),
        lambda p: p.update({"extra": True}),
        lambda p: p.update({"authority_contract_sha256": "0" * 64}),
        lambda p: p.update({"reviewed_runner_core_sha256": "0" * 64}),
        lambda p: p.update({"run_id": "other-run"}),
        lambda p: p.update({"state": "CLAIMED_CONSUMED"}),
        lambda p: p["authorization"].update({"final_holdout": True}),
        lambda p: p["authorization"].update({"validation_evaluations": 2}),
        lambda p: p["paths"].update({"claim_locator": "../claim.json"}),
        lambda p: p["prefit"].update({"source_identities": "0" * 64}),
        lambda p: p["root_authority"].update({"identity": "forged"}),
        lambda p: p["reviewer"].update({"independent_from_evidence_generator": False}),
    ],
)
def test_hostile_permit_mutations_fail_even_if_self_rehashed(mutation) -> None:
    permit = _permit(); mutation(permit); permit.pop("permit_sha256", None); permit["permit_sha256"] = authority.sha256(permit)
    with pytest.raises(authority.ExecutionAuthorityContractError):
        authority.validate_permit_payload(permit, expected_runner_core_sha256=RUNNER_CORE, expected_run_id="run-1")


def test_claim_binds_exact_permit_nonce_runner_and_result_path() -> None:
    permit = _permit(); raw = authority.canonical_bytes(permit) + b"\n"; raw_sha = hashlib.sha256(raw).hexdigest()
    claim = _claim(permit, raw_sha)
    authority.validate_claim_payload(claim, permit=permit, permit_raw_sha256=raw_sha, expected_runner_core_sha256=RUNNER_CORE)
    for key, bad in (("nonce", "bad"), ("result_locator", "../result"), ("reviewed_runner_core_sha256", "0" * 64)):
        altered = deepcopy(claim); altered[key] = bad; altered.pop("claim_sha256"); altered["claim_sha256"] = authority.sha256(altered)
        with pytest.raises(authority.ExecutionAuthorityContractError):
            authority.validate_claim_payload(altered, permit=permit, permit_raw_sha256=raw_sha, expected_runner_core_sha256=RUNNER_CORE)


@pytest.mark.parametrize(
    ("issued", "expires", "now", "accepted"),
    [
        ("2026-07-30T00:00:00Z", "2026-07-31T00:00:00Z", datetime(2026, 7, 30, 0, 0, 0, tzinfo=timezone.utc), True),
        ("2026-07-30T00:00:00Z", "2026-07-31T00:00:00Z", datetime(2026, 7, 31, 0, 0, 0, tzinfo=timezone.utc), False),
        ("2026-07-30T00:00:00Z", "2026-07-31T00:00:00Z", datetime(2026, 7, 29, 23, 59, 59, tzinfo=timezone.utc), False),
        ("2026-07-31T00:00:00Z", "2026-07-30T00:00:00Z", datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc), False),
        ("2026-07-30T00:00:00+00:00", "2026-07-31T00:00:00Z", datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc), False),
        ("malformed", "2026-07-31T00:00:00Z", datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc), False),
    ],
)
def test_permit_time_window_is_canonical_utc_and_half_open(issued, expires, now, accepted) -> None:
    permit = _permit(); permit["issued_at"] = issued; permit["expires_at"] = expires; permit.pop("permit_sha256"); permit["permit_sha256"] = authority.sha256(permit)
    if accepted:
        authority.validate_permit_time_window(permit, now_utc=now)
    else:
        with pytest.raises(authority.ExecutionAuthorityContractError):
            authority.validate_permit_time_window(permit, now_utc=now)
    with pytest.raises(authority.ExecutionAuthorityContractError):
        authority.validate_permit_time_window(_permit(), now_utc=datetime(2026, 7, 30, 12, 0, 0))


def test_fixed_paths_reject_traversal_symlink_hardlink_and_existing_consumed_claim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(authority, "ROOT", tmp_path)
    parent = tmp_path / authority.NAMESPACE; parent.mkdir(parents=True)
    with pytest.raises(authority.ExecutionAuthorityContractError): authority.fixed_path("../claim.json")
    claim = tmp_path / authority.CLAIM_LOCATOR
    target = parent / "target"; target.write_text("x")
    claim.symlink_to(target)
    with pytest.raises(authority.ExecutionAuthorityContractError): authority.fixed_path(authority.CLAIM_LOCATOR)
    claim.unlink(); os.link(target, claim)
    with pytest.raises(authority.ExecutionAuthorityContractError): authority.fixed_path(authority.CLAIM_LOCATOR)
    claim.unlink(); claim.write_text("consumed")
    with pytest.raises(authority.ExecutionAuthorityContractError, match="consumed"):
        authority.validate_claim_target_before_create()


def test_claim_semantics_require_exclusive_before_read_and_crash_stays_consumed() -> None:
    semantics = authority.authority_contract()["claim_semantics"]
    assert "before any protected read" in semantics and "O_CREAT|O_EXCL" in semantics
    assert "any existing claim is consumed" in semantics and "post-claim crash is terminal incomplete" in semantics
    assert "not empirical no-winner" in semantics
