"""Validate an independent review of the contract reconciliation candidate.

This registry can establish that the candidate was independently reviewed.  It
still cannot replace the active trust root: activation is a separate no-clobber
operation that must bind this externally pinned registry.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

from lol_kills.v2.data.source_tree import canonical_source_tree_sha256

from . import contract_validation as validation
from .contract_reconciliation_v1 import (
    DEFAULT_OUTPUT as CANDIDATE_LOCATOR,
    REVIEW_REGISTRY_ENV as EXTERNAL_SHA256_ENV,
    REVIEW_REGISTRY_LOCATOR as REGISTRY_LOCATOR,
    ContractReconciliationError,
    validate_contract_reconciliation_candidate_v1,
)
from .types import canonical_sha256


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/evaluation/contract_reconciliation_review_v1.py"
SCHEMA_VERSION = "scryglass:contract-validation-reconciliation-review:v1"
CANDIDATE_RAW_SHA256 = (
    "1eed5424830ad934e490aeac8a084cf48ffa5c3897bde410e4cd5d7db92f3785"
)
CANDIDATE_ARTIFACT_SHA256 = (
    "0ec4602e874be495463a673544d06320417d7ccfb7feceefc365e7a6d7abb0be"
)
PRIOR_TREE_EVIDENCE_ROOT = PurePosixPath(
    "data/lol/private_contract_authority/evidence/prior-contract-tree-v1"
)
REPLAY_EVIDENCE_PREFIX = PurePosixPath(
    "data/lol/private_contract_authority/evidence/contract-reconciliation-v1"
)
REVIEW_SCOPES = {
    "SCHEMA_SEMANTICS": {
        "reviewer_independent_of_contract_and_validation_code_authors": True,
        "exact_prior_and_candidate_contract_tree_bytes_verified": True,
        "all_schema_semantic_changes_reviewed": True,
        "all_examples_and_mutations_replayed_against_candidate_anchors": True,
        "no_weaker_probability_rating_draft_or_provenance_semantics_found": True,
        "review_not_generated_by_the_evaluated_system": True,
    },
    "FAIL_CLOSED_AUTHORITY": {
        "reviewer_independent_of_schema_semantics_reviewer_and_code_authors": True,
        "unavailable_and_research_only_outputs_remain_fail_closed": True,
        "production_model_authority_remains_false": True,
        "public_non_betting_boundary_remains_explicit": True,
        "candidate_and_registry_do_not_self_activate": True,
        "review_not_generated_by_the_evaluated_system": True,
    },
}
AUTHORITY = {
    "contract_reconciliation_review_authority": True,
    "contract_trust_root_activation_authority": False,
    "model_validation_authority": False,
    "player_rating_authority": False,
    "team_rating_authority": False,
    "draft_validation_authority": False,
    "probability_authority": False,
    "odds_authority": False,
    "expected_value_authority": False,
    "recommendation_authority": False,
    "betting_authority": False,
}
CLAIM_CEILING = (
    "Independent review of exact contract reconciliation evidence only. The "
    "active trust root is unchanged and no model, rating, Draft Score, "
    "probability, odds, expected value, recommendation, or betting authority "
    "is granted."
)
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractReconciliationReviewError(RuntimeError):
    """The independent reconciliation registry or evidence failed closed."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise ContractReconciliationReviewError(
            f"{field} must be a lowercase SHA-256"
        )
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractReconciliationReviewError(f"{field} must be nonempty")
    return value.strip()


def _time(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractReconciliationReviewError(
            f"{field} must be RFC-3339"
        ) from exc
    if parsed.tzinfo is None:
        raise ContractReconciliationReviewError(f"{field} lacks timezone")
    return parsed.astimezone(timezone.utc)


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ContractReconciliationReviewError(
                f"duplicate JSON key: {key}"
            )
        result[key] = value
    return result


def _object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except ContractReconciliationReviewError:
        raise
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ContractReconciliationReviewError(
            f"{label} is not strict JSON"
        ) from exc
    if not isinstance(value, dict):
        raise ContractReconciliationReviewError(f"{label} must be an object")
    return value


def _candidate(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = root / CANDIDATE_LOCATOR
    if path.is_symlink() or not path.is_file():
        raise ContractReconciliationReviewError("candidate is unavailable")
    raw = path.read_bytes()
    if _sha256(raw) != CANDIDATE_RAW_SHA256:
        raise ContractReconciliationReviewError("candidate raw hash changed")
    try:
        candidate = validate_contract_reconciliation_candidate_v1(
            _object(raw, "candidate"), root=root
        )
    except ContractReconciliationError as exc:
        raise ContractReconciliationReviewError(str(exc)) from exc
    if candidate.get("artifact_sha256") != CANDIDATE_ARTIFACT_SHA256:
        raise ContractReconciliationReviewError("candidate artifact hash changed")
    binding = {
        "locator": CANDIDATE_LOCATOR.as_posix(),
        "raw_sha256": CANDIDATE_RAW_SHA256,
        "artifact_sha256": CANDIDATE_ARTIFACT_SHA256,
        "created_at_utc": candidate["created_at_utc"],
        "prior_contract_tree_sha256": candidate["active_trust_root"][
            "contract_tree_sha256"
        ],
        "candidate_contract_tree_sha256": candidate["current_contracts"][
            "contract_tree_sha256"
        ],
    }
    return candidate, binding


def _evidence_locator(value: Any, field: str) -> str:
    text = _text(value, field)
    locator = PurePosixPath(text)
    if (
        locator.is_absolute()
        or any(part in {"", ".", ".."} for part in locator.parts)
        or tuple(locator.parts[: len(REPLAY_EVIDENCE_PREFIX.parts)])
        != REPLAY_EVIDENCE_PREFIX.parts
    ):
        raise ContractReconciliationReviewError(
            f"{field} is outside the reconciliation evidence prefix"
        )
    return locator.as_posix()


def _expected_semantic_coverage(root: Path) -> dict[str, Any]:
    schemas = {
        name: json.loads(
            (root / validation.CONTRACT_ROOT / name).read_text(encoding="utf-8")
        )
        for name in validation.SCHEMA_FILES
    }
    invariants, mutations = validation._collect_extensions(schemas)
    structural = sorted(
        mutation_id
        for mutation_id, (_, fixture) in mutations.items()
        if fixture.get("expected_schema_failure") is True
    )
    return {
        "outputs": sorted(validation.OUTPUT_SCHEMAS),
        "invariant_ids": sorted(invariants),
        "mutation_ids": sorted(mutations),
        "structural_mutation_ids": structural,
        "invariant_pass_count": len(invariants) * 2,
        "mutation_pass_count": len(mutations),
        "structural_pass_count": len(structural),
        "all_pass": True,
    }


def validate_contract_reconciliation_review_v1(
    payload: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ContractReconciliationReviewError("review registry must be an object")
    value = dict(payload)
    expected_keys = {
        "schema_version",
        "registry_id",
        "status",
        "registered_at_utc",
        "candidate_binding",
        "prior_contract_tree_evidence",
        "semantic_replay_evidence",
        "reviews",
        "decision",
        "authority",
        "claim_ceiling",
    }
    if set(value) != expected_keys:
        raise ContractReconciliationReviewError("review registry fields changed")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != "INDEPENDENT_REVIEW_REGISTERED_PENDING_ACTIVATION"
    ):
        raise ContractReconciliationReviewError("review registry identity changed")
    _text(value.get("registry_id"), "registry_id")
    registered = _time(value.get("registered_at_utc"), "registered_at_utc")
    candidate, binding = _candidate(root)
    if value.get("candidate_binding") != binding:
        raise ContractReconciliationReviewError("candidate binding changed")
    if _time(binding["created_at_utc"], "candidate.created_at") > registered:
        raise ContractReconciliationReviewError("registry predates candidate")

    prior = value.get("prior_contract_tree_evidence")
    if not isinstance(prior, Mapping) or dict(prior) != {
        "root_locator": PRIOR_TREE_EVIDENCE_ROOT.as_posix(),
        "contract_tree_sha256": validation.CONTRACT_TREE_SHA256,
        "allowlist": list(validation.CONTRACT_SOURCE_TREE_ALLOWLIST),
    }:
        raise ContractReconciliationReviewError(
            "prior contract-tree evidence binding changed"
        )
    prior_root = root / PRIOR_TREE_EVIDENCE_ROOT
    try:
        prior_digest = canonical_source_tree_sha256(
            prior_root, validation.CONTRACT_SOURCE_TREE_ALLOWLIST
        )
    except (OSError, ValueError) as exc:
        raise ContractReconciliationReviewError(
            "prior contract-tree evidence is incomplete"
        ) from exc
    if prior_digest != validation.CONTRACT_TREE_SHA256:
        raise ContractReconciliationReviewError(
            "prior contract-tree evidence hash changed"
        )

    semantic = value.get("semantic_replay_evidence")
    if not isinstance(semantic, Mapping) or set(semantic) != {
        "locator",
        "raw_sha256",
        "report_sha256",
    }:
        raise ContractReconciliationReviewError(
            "semantic replay evidence structure changed"
        )
    semantic_locator = _evidence_locator(
        semantic.get("locator"), "semantic_replay_evidence.locator"
    )
    semantic_path = root / semantic_locator
    if semantic_path.is_symlink() or not semantic_path.is_file():
        raise ContractReconciliationReviewError(
            "semantic replay evidence is unavailable"
        )
    semantic_raw = semantic_path.read_bytes()
    if _sha256(semantic_raw) != _sha(
        semantic.get("raw_sha256"), "semantic_replay_evidence.raw_sha256"
    ):
        raise ContractReconciliationReviewError(
            "semantic replay evidence raw hash changed"
        )
    report = _object(semantic_raw, "semantic replay evidence")
    report_hash = report.pop("report_sha256", None)
    if report_hash != _sha(
        semantic.get("report_sha256"), "semantic_replay_evidence.report_sha256"
    ) or report_hash != canonical_sha256(report):
        raise ContractReconciliationReviewError(
            "semantic replay report hash changed"
        )
    if set(report) != {
        "schema_version",
        "report_id",
        "executed_at_utc",
        "candidate_binding",
        "runner_provenance",
        "coverage",
        "authority",
    }:
        raise ContractReconciliationReviewError(
            "semantic replay report fields changed"
        )
    if report.get("schema_version") != (
        "scryglass:contract-validation-candidate-semantic-replay:v1"
    ) or report.get("candidate_binding") != binding:
        raise ContractReconciliationReviewError(
            "semantic replay report identity changed"
        )
    executed = _time(report.get("executed_at_utc"), "semantic replay executed_at")
    if executed > registered or executed < _time(
        binding["created_at_utc"], "candidate.created_at"
    ):
        raise ContractReconciliationReviewError(
            "semantic replay chronology changed"
        )
    _text(report.get("report_id"), "semantic replay report_id")
    runner = report.get("runner_provenance")
    if not isinstance(runner, Mapping) or set(runner) != {
        "implementation_id",
        "source_locator",
        "source_raw_sha256",
        "environment_lock_locator",
        "environment_lock_raw_sha256",
        "generated_by_evaluated_system",
    }:
        raise ContractReconciliationReviewError(
            "semantic replay runner provenance changed"
        )
    for field in ("source_locator", "environment_lock_locator"):
        locator = _evidence_locator(runner.get(field), f"runner.{field}")
        path = root / locator
        if path.is_symlink() or not path.is_file():
            raise ContractReconciliationReviewError(
                f"semantic replay runner {field} is unavailable"
            )
        if _sha256(path.read_bytes()) != _sha(
            runner.get(field.replace("locator", "raw_sha256")),
            f"runner.{field}.raw_sha256",
        ):
            raise ContractReconciliationReviewError(
                f"semantic replay runner {field} hash changed"
            )
    if (
        runner.get("generated_by_evaluated_system") is not False
        or report.get("coverage") != _expected_semantic_coverage(root)
        or report.get("authority") != {
            "production_model_authority": False,
            "probability_authority": False,
            "betting_authority": False,
        }
    ):
        raise ContractReconciliationReviewError(
            "semantic replay coverage or authority changed"
        )

    reviews = value.get("reviews")
    if not isinstance(reviews, list) or len(reviews) != 2:
        raise ContractReconciliationReviewError("two independent reviews required")
    reviewers: set[str] = set()
    scopes: set[str] = set()
    for review in reviews:
        if not isinstance(review, Mapping) or set(review) != {
            "scope",
            "reviewer_id",
            "reviewed_at_utc",
            "attestation",
        }:
            raise ContractReconciliationReviewError("review structure changed")
        scope = _text(review.get("scope"), "review.scope")
        reviewer = _text(review.get("reviewer_id"), "review.reviewer_id")
        reviewed = _time(review.get("reviewed_at_utc"), "review.reviewed_at_utc")
        if reviewed > registered or review.get("attestation") != REVIEW_SCOPES.get(scope):
            raise ContractReconciliationReviewError("review attestation changed")
        scopes.add(scope)
        reviewers.add(reviewer)
    if scopes != set(REVIEW_SCOPES) or len(reviewers) != 2:
        raise ContractReconciliationReviewError(
            "review scopes or reviewer independence changed"
        )
    if value.get("decision") != {
        "candidate_independently_reviewed": True,
        "prior_contract_tree_replayed": True,
        "candidate_semantic_harness_passed": True,
        "active_trust_root_changed": False,
        "separate_activation_required": True,
    }:
        raise ContractReconciliationReviewError("review decision changed")
    if value.get("authority") != AUTHORITY or value.get("claim_ceiling") != CLAIM_CEILING:
        raise ContractReconciliationReviewError("review authority boundary changed")
    return value


def load_pinned_contract_reconciliation_review_v1(
    *, root: Path = ROOT, environment: Mapping[str, str]
) -> dict[str, Any]:
    digest = _sha(environment.get(EXTERNAL_SHA256_ENV), "external review digest")
    path = root / REGISTRY_LOCATOR
    if path.is_symlink() or not path.is_file():
        raise ContractReconciliationReviewError("review registry is unavailable")
    raw = path.read_bytes()
    if _sha256(raw) != digest:
        raise ContractReconciliationReviewError(
            "review registry does not match external pin"
        )
    receipt = validate_contract_reconciliation_review_v1(
        _object(raw, "review registry"), root=root
    )
    return {
        "status": "independently_reviewed_pending_activation",
        "receipt": receipt,
        "receipt_raw_sha256": digest,
        "contract_trust_root_active": False,
        "model_authorized": False,
        "betting_authorized": False,
    }


__all__ = [
    "EXTERNAL_SHA256_ENV",
    "REGISTRY_LOCATOR",
    "ContractReconciliationReviewError",
    "load_pinned_contract_reconciliation_review_v1",
    "validate_contract_reconciliation_review_v1",
]
