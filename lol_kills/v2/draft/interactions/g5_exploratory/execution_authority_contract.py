"""Tiny non-authorizing schema for a future reviewer-owned G5 permit.

No authority implementation, permit, claim, protected read, fit, or result is
created here.  These are pure validators and fixed names only.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[5]
NAMESPACE = "data/lol/v2/models/draft-interactions/g5-exploratory"
PERMIT_LOCATOR = f"{NAMESPACE}/execution-permit.json"
CLAIM_LOCATOR = f"{NAMESPACE}/execution-claim.json"
RESULT_LOCATOR = f"{NAMESPACE}/execution-result.json"
PERMIT_SCHEMA = "scryglass:g5-execution-permit:v1"
CLAIM_SCHEMA = "scryglass:g5-execution-claim:v1"
PENDING_REVIEWED_RUNNER_CORE = "PENDING_REVIEWED_RUNNER_CORE"
PREFIT_CONTRACT_SHA256 = "993a9e8e6184e8f2e2b7c1eed244f28ed6eb5d749f067979881d35e715a3a1f0"
PREFIT_CORE_SHA256 = "25df39ad248fed2565aed7f501b935f1992020fa5441ccff3b6e6ee99cf15ab8"
PREFIT_REVIEW_SHA256 = "f869b509abe2ba17bee66eff9a44d72f4ca6422d6ded78f1459e3976180d21ec"
FROZEN_SOURCE_IDENTITIES = {
    "g1_base": {"manifest": "3af87fffb2b32fd95aeb920409abe0254fa158b3dc7f079650b3472731d4ff72", "rows": "4ed79abb0b2471a666ab5643b91edf33c2fdde19e361c456aa589d2e9a4df846", "selected_target": "4c332fa4e6cb155341bcffd83bd0ee1be2e04f3b5950b8a7745931253dd8bd2d", "split": "1695cee14ad6b4221526ec6187206b8c61a560a00005d2f799f808ed901ee014"},
    "g1_features": {"manifest_raw": "4ba9653e634c703ca6ef4379461833276811b4571e4f998f2bd6683d5e060efb", "manifest_canonical": "7e559054ac3f1bd79f1821121c17b778927736f6a3a52c85b48b5d3d0460189c", "rows_raw": "e742631e1c12fb1af7148468a0d595ff6cf23e816af4edb20af162a04a6a9680", "rows_canonical": "52d59dd0c41a212f7eb07b6f6132841f3c152f28324308b376042f8e262c141d", "review_raw": "eb8bde9730421469520a60383282d2810904fffc5896f24263135a0b96a079fb", "review_canonical": "a73ba02cd14083f702fd96fde9df6d616c4d0fec81b21d7ae0bc98c128ce517e"},
    "g2": {"runner_raw": "eff9232fbc53bc3ffff6e57285aea2647ff52ade945985db63449b995e651706", "model_raw": "abf80300e4b740bfdb11b9fb33533034c12e934b97df59b2100b9dd56cdfdf8c", "artifact_raw": "b0d8276fd164735db0abd9b2353c7e10168c599e5607e4db3d15cd12bd9d7b50", "artifact_canonical": "35e8831fb4d39fd60ec7f8f59b934ff5571f788ec8dc1151c78661b67ab6d4fd"},
    "clusters": {"proxy_raw": "d63ee58bb93015e0c0427c7aac584b098884a1567da32849e4ed1993e54dae48", "artifact_raw": "5d89bcc3029d3fa912d76af9c702888f76f4183933dd1d1e2e660b8f2a8bdd2a", "artifact_canonical": "e456d267797a23dae94f8ecc9a31ca91593d48e830b6f334191f0c025bf19ada"},
}
ROOT_AUTHORITY = {"identity": "KOI_MARI", "target_authority": "private_retrospective_oe_target_v1", "scope": "private_model_fit_and_private_rank_selection_only"}
REVIEWER_SCOPE = "private_retrospective_g5_execution_permit_review"


class ExecutionAuthorityContractError(ValueError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


FROZEN_SOURCE_IDENTITIES_SHA256 = sha256(FROZEN_SOURCE_IDENTITIES)


def _sha(value: Any, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ExecutionAuthorityContractError(f"{label} must be lowercase sha256")


def _text(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value or len(value) > 160 or any(char in value for char in "/\\\x00"):
        raise ExecutionAuthorityContractError(f"{label} invalid")


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ExecutionAuthorityContractError(f"{label} exact field set mismatch")


def fixed_path(locator: str) -> Path:
    if locator not in {PERMIT_LOCATOR, CLAIM_LOCATOR, RESULT_LOCATOR}:
        raise ExecutionAuthorityContractError("unpermitted fixed locator")
    path = ROOT / locator
    try:
        relative = path.absolute().relative_to(ROOT.absolute())
    except ValueError as error:
        raise ExecutionAuthorityContractError("fixed locator outside repository") from error
    current = ROOT.absolute()
    for part in relative.parts[:-1]:
        current = current / part
        meta = os.lstat(current)
        if stat.S_ISLNK(meta.st_mode) or not stat.S_ISDIR(meta.st_mode):
            raise ExecutionAuthorityContractError("unsafe fixed locator parent")
    if path.exists():
        meta = os.lstat(path)
        if stat.S_ISLNK(meta.st_mode) or not stat.S_ISREG(meta.st_mode) or meta.st_nlink != 1:
            raise ExecutionAuthorityContractError("unsafe fixed locator leaf")
    return path


def authority_contract() -> dict[str, Any]:
    """Frozen schema identity, not a permit and not an authorization."""

    payload = {
        "schema_id": "scryglass:g5-execution-authority-contract:v1",
        "state": "SCHEMA_FROZEN_NO_EXECUTION_AUTHORITY",
        "future_api": ["AUTHORITY_CONTRACT_SHA256", "authenticate_immutable_permit(...):must_call_validate_permit_time_window", "claim_single_use_run(...)"],
        "fixed_locators": {"permit": PERMIT_LOCATOR, "claim": CLAIM_LOCATOR, "result": RESULT_LOCATOR},
        "root_authority": ROOT_AUTHORITY,
        "pending_runner_core": PENDING_REVIEWED_RUNNER_CORE,
        "prefit": {"contract": PREFIT_CONTRACT_SHA256, "core": PREFIT_CORE_SHA256, "review": PREFIT_REVIEW_SHA256, "source_identities": FROZEN_SOURCE_IDENTITIES, "source_identities_sha256": FROZEN_SOURCE_IDENTITIES_SHA256},
        "permit_fields": ["schema_id", "permit_id", "run_id", "nonce", "issued_at", "expires_at", "authority_contract_sha256", "reviewed_runner_core_sha256", "prefit", "root_authority", "reviewer", "authorization", "paths", "state", "permit_sha256"],
        "claim_fields": ["schema_id", "permit_id", "run_id", "nonce", "permit_raw_sha256", "permit_canonical_sha256", "authority_contract_sha256", "reviewed_runner_core_sha256", "result_locator", "result_locator_sha256", "state", "claim_sha256"],
        "claim_semantics": "before any protected read: component-safe fixed claim path, O_CREAT|O_EXCL, write, fsync file, fsync parent; any existing claim is consumed; never delete/reset/reuse; post-claim crash is terminal incomplete, not empirical no-winner",
        "claim_ceiling": {"schema_only": True, "execution_authorization": False, "protected_reads": 0, "model_fit": False, "private_rank_selection": False, "final_holdout": False, "public": False, "prediction": False, "forecast": False, "publication": False, "promotion": False},
    }
    payload["artifact_sha256"] = sha256(payload)
    return payload


AUTHORITY_CONTRACT_SHA256 = authority_contract()["artifact_sha256"]
RESULT_LOCATOR_SHA256 = sha256({"result_locator": RESULT_LOCATOR})


def _authorization() -> dict[str, Any]:
    return {"private_model_fit": True, "private_rank_selection": True, "validation_evaluations": 1, "final_holdout": False, "public": False, "prediction": False, "forecast": False, "publication": False, "promotion": False, "sota": False, "reliability": False, "current": False, "live": False}


def _utc_rfc3339(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or len(value) != 20 or not value.endswith("Z"):
        raise ExecutionAuthorityContractError(f"{label} must be canonical UTC RFC3339")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as error:
        raise ExecutionAuthorityContractError(f"{label} must be canonical UTC RFC3339") from error


def validate_permit_time_window(value: Mapping[str, Any], *, now_utc: datetime) -> None:
    """Pure authentication prerequisite; current time must be UTC-aware."""

    issued = _utc_rfc3339(value.get("issued_at"), "issued_at")
    expires = _utc_rfc3339(value.get("expires_at"), "expires_at")
    if issued >= expires:
        raise ExecutionAuthorityContractError("permit issued_at must precede expires_at")
    if not isinstance(now_utc, datetime) or now_utc.tzinfo is None or now_utc.utcoffset() != timezone.utc.utcoffset(now_utc):
        raise ExecutionAuthorityContractError("now_utc must be UTC-aware")
    if not issued <= now_utc < expires:
        raise ExecutionAuthorityContractError("permit is not currently valid")


def validate_permit_payload(value: Mapping[str, Any], *, expected_runner_core_sha256: str, expected_run_id: str | None = None) -> None:
    """Pure shape check; a future raw-byte-pinned authority must call it."""

    _exact(value, {"schema_id", "permit_id", "run_id", "nonce", "issued_at", "expires_at", "authority_contract_sha256", "reviewed_runner_core_sha256", "prefit", "root_authority", "reviewer", "authorization", "paths", "state", "permit_sha256"}, "permit")
    unsigned = dict(value); claimed = unsigned.pop("permit_sha256")
    if value.get("schema_id") != PERMIT_SCHEMA or value.get("state") != "AUTHORIZED_ONCE" or claimed != sha256(unsigned):
        raise ExecutionAuthorityContractError("permit schema/state/self hash mismatch")
    _sha(expected_runner_core_sha256, "expected runner core")
    for field in ("permit_id", "run_id", "nonce"):
        _text(value.get(field), field)
    issued = _utc_rfc3339(value.get("issued_at"), "issued_at")
    expires = _utc_rfc3339(value.get("expires_at"), "expires_at")
    if issued >= expires:
        raise ExecutionAuthorityContractError("permit issued_at must precede expires_at")
    if expected_run_id is not None and value["run_id"] != expected_run_id:
        raise ExecutionAuthorityContractError("permit run id mismatch")
    if value.get("authority_contract_sha256") != AUTHORITY_CONTRACT_SHA256 or value.get("reviewed_runner_core_sha256") != expected_runner_core_sha256:
        raise ExecutionAuthorityContractError("permit contract/runner binding mismatch")
    if value.get("prefit") != {"contract": PREFIT_CONTRACT_SHA256, "core": PREFIT_CORE_SHA256, "review": PREFIT_REVIEW_SHA256, "source_identities_sha256": FROZEN_SOURCE_IDENTITIES_SHA256} or value.get("root_authority") != ROOT_AUTHORITY:
        raise ExecutionAuthorityContractError("permit prefit/root authority mismatch")
    reviewer = value.get("reviewer")
    if not isinstance(reviewer, Mapping) or set(reviewer) != {"identity", "independent_from_evidence_generator", "scope"} or not isinstance(reviewer.get("identity"), str) or reviewer.get("independent_from_evidence_generator") is not True or reviewer.get("scope") != REVIEWER_SCOPE:
        raise ExecutionAuthorityContractError("permit reviewer mismatch")
    if value.get("authorization") != _authorization() or value.get("paths") != {"claim_locator": CLAIM_LOCATOR, "result_locator": RESULT_LOCATOR, "result_locator_sha256": RESULT_LOCATOR_SHA256}:
        raise ExecutionAuthorityContractError("permit scope/path mismatch")
    fixed_path(CLAIM_LOCATOR); fixed_path(RESULT_LOCATOR)


def validate_permit_bytes(raw: bytes, *, expected_raw_sha256: str, expected_runner_core_sha256: str, expected_run_id: str | None = None) -> dict[str, Any]:
    """Byte check for a future reviewer-held raw permit pin; no file is read."""

    _sha(expected_raw_sha256, "expected permit raw")
    if hashlib.sha256(raw).hexdigest() != expected_raw_sha256:
        raise ExecutionAuthorityContractError("permit raw pin mismatch")
    try: value = json.loads(raw)
    except json.JSONDecodeError as error: raise ExecutionAuthorityContractError("permit is not JSON") from error
    if raw != canonical_bytes(value) + b"\n":
        raise ExecutionAuthorityContractError("permit not canonical newline JSON")
    validate_permit_payload(value, expected_runner_core_sha256=expected_runner_core_sha256, expected_run_id=expected_run_id)
    return value


def validate_claim_payload(value: Mapping[str, Any], *, permit: Mapping[str, Any], permit_raw_sha256: str, expected_runner_core_sha256: str) -> None:
    """Pure consumed-marker validation; this contract never creates a claim."""

    _exact(value, {"schema_id", "permit_id", "run_id", "nonce", "permit_raw_sha256", "permit_canonical_sha256", "authority_contract_sha256", "reviewed_runner_core_sha256", "result_locator", "result_locator_sha256", "state", "claim_sha256"}, "claim")
    unsigned = dict(value); claimed = unsigned.pop("claim_sha256")
    if value.get("schema_id") != CLAIM_SCHEMA or value.get("state") != "CLAIMED_CONSUMED" or claimed != sha256(unsigned):
        raise ExecutionAuthorityContractError("claim schema/state/self hash mismatch")
    _sha(permit_raw_sha256, "permit raw")
    if value.get("permit_id") != permit.get("permit_id") or value.get("run_id") != permit.get("run_id") or value.get("nonce") != permit.get("nonce") or value.get("permit_raw_sha256") != permit_raw_sha256 or value.get("permit_canonical_sha256") != permit.get("permit_sha256") or value.get("authority_contract_sha256") != AUTHORITY_CONTRACT_SHA256 or value.get("reviewed_runner_core_sha256") != expected_runner_core_sha256 or value.get("result_locator") != RESULT_LOCATOR or value.get("result_locator_sha256") != RESULT_LOCATOR_SHA256:
        raise ExecutionAuthorityContractError("claim runtime binding mismatch")
    fixed_path(RESULT_LOCATOR)


def validate_claim_target_before_create() -> Path:
    """A future claim writer must call this before O_CREAT|O_EXCL; no write."""

    path = fixed_path(CLAIM_LOCATOR)
    if path.exists():
        raise ExecutionAuthorityContractError("claim already consumed")
    return path
