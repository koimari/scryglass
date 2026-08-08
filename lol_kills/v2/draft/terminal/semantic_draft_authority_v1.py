"""Semantic authority for the private terminal Draft Score component.

The old L2 record binds development artifacts but does not semantically replay
the jointly evaluated future result.  This short-lived authority is the missing
deployment layer.  It authorizes only the equal-strength terminal-Draft
component; an event probability must separately bind exact approved ratings and
pass the market-probability authority stack.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from lol_kills.v2.market import phase_one_evaluation_registry_v1 as registry
from lol_kills.v2.market import phase_one_evaluation_v1 as evaluation

from . import future_prediction_ledger as draft_ledger
from .model import TerminalDraftError, TerminalModel


ROOT = Path(__file__).resolve().parents[4]
SOURCE_LOCATOR = "lol_kills/v2/draft/terminal/semantic_draft_authority_v1.py"
SCHEMA_VERSION = "scryglass:semantic-terminal-draft-authority:v1"
AUTHORITY_LOCATOR = Path(
    "data/lol/private_draft_authority/semantic-terminal-draft-authority-v1.json"
)
EXTERNAL_SHA256_ENV = "SCRYGLASS_SEMANTIC_TERMINAL_DRAFT_AUTHORITY_SHA256"
MODEL_PREFIX = PurePosixPath("data/lol/v2/models/draft-terminal")
PRODUCTION_SOURCE_LOCATORS = (
    "lol_kills/v2/draft/terminal/model.py",
    "lol_kills/v2/draft/terminal/promotion.py",
    "apps/lol-atlas/src/lib/draftTerminalScore.ts",
    "apps/lol-atlas/src/lib/draftTerminalServer.ts",
)
REVIEW_SCOPES = {
    "TERMINAL_DRAFT_MODEL_DEPLOYMENT": {
        "reviewer_independent_of_candidate_evaluator_outcome_and_production_code_authors": True,
        "exact_joint_future_result_and_terminal_draft_gates_verified": True,
        "approved_model_lineage_and_equal_strength_estimand_replayed": True,
        "no_post_outcome_model_threshold_calibration_or_cohort_change_found": True,
        "review_not_generated_by_the_evaluated_system": True,
    },
    "TERMINAL_DRAFT_RUNTIME_PARITY": {
        "reviewer_independent_of_python_typescript_and_parity_artifact_authors": True,
        "exact_registered_python_typescript_parity_artifact_replayed": True,
        "serving_boundary_withholds_public_and_event_probability_authority": True,
        "combined_probability_requires_separately_authorized_exact_rating_bytes": True,
        "review_not_generated_by_the_evaluated_system": True,
    },
}
DEPLOYMENT_POLICY = {
    "estimand": "equal_strength_terminal_composition_index",
    "terminal_pick_ban_and_role_assignment_required": True,
    "exact_source_identity_timing_rights_and_patch_required": True,
    "posterior_interval_required": True,
    "neutral_score_is_not_event_win_probability": True,
    "combined_probability_requires_exact_semantically_authorized_rating_receipt": True,
    "combined_probability_requires_separate_market_probability_authority": True,
    "public_serving_permitted": False,
}
AUTHORITY = {
    "private_terminal_draft_component_authority": True,
    "private_equal_strength_score_authority": True,
    "private_event_probability_authority": False,
    "public_probability_authority": False,
    "fair_odds_authority": False,
    "expected_value_authority": False,
    "recommendation_authority": False,
    "transaction_authority": False,
    "stake_authority": False,
    "betting_authority": False,
}
CLAIM_CEILING = (
    "Private equal-strength terminal-Draft component only. It is not an event "
    "win probability, causal draft effect, fair price, expected value, betting "
    "recommendation, stake, transaction, or public betting output. A combined "
    "event probability must bind exact separately authorized rating bytes and "
    "pass the independent market-probability authority stack."
)


class SemanticDraftAuthorityError(RuntimeError):
    """Future evaluation, parity, deployment review, artifact, or pin failed."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise SemanticDraftAuthorityError(f"{field} must be a lowercase SHA-256")
    return value


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SemanticDraftAuthorityError(f"{field} must be nonempty")
    return value.strip()


def _timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise SemanticDraftAuthorityError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise SemanticDraftAuthorityError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SemanticDraftAuthorityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SemanticDraftAuthorityError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise SemanticDraftAuthorityError(f"{label} must contain an object")
    return value


def _source_record(root: Path, locator: str) -> dict[str, Any]:
    path = root / locator
    if path.is_symlink() or not path.is_file():
        raise SemanticDraftAuthorityError(f"production source unavailable: {locator}")
    return {
        "locator": locator,
        "bytes": path.stat().st_size,
        "raw_sha256": evaluation._sha256_path(path),
    }


def _evaluated_draft_model_binding(
    snapshot: Mapping[str, Any], *, root: Path
) -> dict[str, Any]:
    entries = (snapshot.get("draft_ledger_candidate") or {}).get("entries")
    if not isinstance(entries, list) or not entries:
        raise SemanticDraftAuthorityError(
            "phase-one snapshot has no evaluated Draft predictions"
        )
    evaluated_models: dict[
        tuple[str, str, str, str, str], dict[str, Any]
    ] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise SemanticDraftAuthorityError(
                "phase-one Draft ledger entry is malformed"
            )
        prediction_raw = evaluation._read_regular(
            root, entry["prediction_locator"], "evaluated Draft prediction"
        )
        prediction = draft_ledger.validate_draft_prediction_receipt(
            evaluation._strict_object(
                prediction_raw, "evaluated Draft prediction"
            ),
            root=root,
        )
        model = prediction["model"]
        identity = (
            model["artifact_locator"],
            model["artifact_raw_sha256"],
            model["model_version"],
            model["candidate_id"],
            model["variant_id"],
        )
        evaluated_models[identity] = dict(model)
    if len(evaluated_models) != 1:
        raise SemanticDraftAuthorityError(
            "phase-one Draft cohort does not bind exactly one model identity"
        )
    model = next(iter(evaluated_models.values()))
    return {
        "locator": model["artifact_locator"],
        "raw_sha256": model["artifact_raw_sha256"],
        "model_version": model["model_version"],
        "candidate_id": model["candidate_id"],
        "variant_id": model["variant_id"],
        "prediction_receipts": len(entries),
        "same_exact_model_used_for_every_evaluated_prediction": True,
    }


def current_expected_bindings(
    *, root: Path = ROOT, environment: Mapping[str, str] = os.environ
) -> dict[str, Any]:
    path = root / registry.REGISTRY_LOCATOR
    digest = environment.get(registry.EXTERNAL_SHA256_ENV)
    if path.is_symlink() or not path.is_file() or not digest:
        raise SemanticDraftAuthorityError("phase-one evaluation registry unavailable")
    registry_payload = _object(path.read_bytes(), "phase-one evaluation registry")
    result_locator = (registry_payload.get("result_binding") or {}).get(
        "result_locator"
    )
    if not isinstance(result_locator, str):
        raise SemanticDraftAuthorityError("phase-one result locator missing")
    try:
        binding = registry.expected_result_binding(
            result_locator=result_locator, root=root
        )
        registered = registry.load_pinned_evaluation_registry(
            path=path,
            external_sha256=digest,
            expected_binding=binding,
        )
        result_raw = evaluation._read_regular(
            root, result_locator, "phase-one evaluation result"
        )
        result = evaluation.validate_phase_one_evaluation_result(
            evaluation._strict_object(result_raw, "phase-one evaluation result")
        )
    except Exception as exc:
        raise SemanticDraftAuthorityError(
            "phase-one evaluation registry or result is invalid"
        ) from exc
    draft = result.get("draft_evaluation") or {}
    if (
        registered.get("phase_one_models_independently_passed") is not True
        or result.get("phase_one_models_passed") is not True
        or draft.get("primary_gate_passed") is not True
        or draft.get("subgroup_nonharm_gate_passed") is not True
        or draft.get("reliability_gate_passed") is not True
        or draft.get("passed") is not True
    ):
        raise SemanticDraftAuthorityError(
            "independently registered future Draft gates did not pass"
        )
    inputs = result["inputs"]
    snapshot_raw, snapshot = evaluation._snapshot(
        root, inputs["snapshot_locator"]
    )
    if (
        _sha256(snapshot_raw) != inputs["snapshot_raw_sha256"]
        or snapshot["artifact_sha256"] != inputs["snapshot_artifact_sha256"]
    ):
        raise SemanticDraftAuthorityError(
            "phase-one snapshot does not match the registered result"
        )
    evaluated_model = _evaluated_draft_model_binding(snapshot, root=root)
    parity_sha = _sha(
        draft.get("typescript_parity_artifact_sha256"),
        "typescript_parity_artifact_sha256",
    )
    return {
        "phase_one_evaluation": {
            "registry_locator": registry.REGISTRY_LOCATOR.as_posix(),
            "registry_raw_sha256": registered["receipt_raw_sha256"],
            "registry_id": registered["receipt"]["registry_id"],
            "result_locator": result_locator,
            "result_raw_sha256": binding["result_raw_sha256"],
            "result_artifact_sha256": result["artifact_sha256"],
            "run_id": result["run_id"],
            "snapshot_artifact_sha256": inputs["snapshot_artifact_sha256"],
            "parity_locator": inputs["parity_locator"],
            "parity_raw_sha256": inputs["parity_raw_sha256"],
            "parity_artifact_sha256": parity_sha,
            "maps": inputs["maps"],
            "series": inputs["series"],
            "draft_primary_gate_passed": True,
            "draft_subgroup_nonharm_gate_passed": True,
            "draft_reliability_and_runtime_parity_gate_passed": True,
            "draft_future_evaluation_independently_passed": True,
        },
        "evaluated_draft_model": evaluated_model,
        "production_sources": [
            _source_record(root, locator)
            for locator in (SOURCE_LOCATOR, *PRODUCTION_SOURCE_LOCATORS)
        ],
        "reviewer_ids_excluded_from_deployment_authority": sorted(
            review["reviewer_id"]
            for review in registered["receipt"]["reviews"]
        ),
    }


def _deployment_model(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {
        "locator",
        "raw_sha256",
        "artifact_sha256",
        "model_version",
    }:
        raise SemanticDraftAuthorityError("deployment model structure changed")
    locator = PurePosixPath(_nonempty(value.get("locator"), "model.locator"))
    if (
        locator.is_absolute()
        or any(part in {"", ".", ".."} for part in locator.parts)
        or tuple(locator.parts[: len(MODEL_PREFIX.parts)]) != MODEL_PREFIX.parts
        or locator.suffix != ".json"
    ):
        raise SemanticDraftAuthorityError("deployment model locator is unsafe")
    return {
        "locator": locator.as_posix(),
        "raw_sha256": _sha(value.get("raw_sha256"), "model.raw_sha256"),
        "artifact_sha256": _sha(
            value.get("artifact_sha256"), "model.artifact_sha256"
        ),
        "model_version": _nonempty(value.get("model_version"), "model.model_version"),
    }


def validate_semantic_draft_authority_v1(
    payload: Mapping[str, Any], *, expected_bindings: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise SemanticDraftAuthorityError("Draft authority must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version",
        "authority_id",
        "status",
        "scope",
        "issued_at_utc",
        "valid_until_utc",
        "reviews",
        "bindings",
        "deployment_model",
        "deployment_policy",
        "authority",
        "claim_ceiling",
    }:
        raise SemanticDraftAuthorityError("Draft authority fields are not exact")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != "APPROVED"
        or value.get("scope") != "PRIVATE_TERMINAL_DRAFT_COMPONENT_ONLY"
    ):
        raise SemanticDraftAuthorityError("Draft authority identity changed")
    _nonempty(value.get("authority_id"), "authority_id")
    issued = _timestamp(value.get("issued_at_utc"), "issued_at_utc")
    valid_until = _timestamp(value.get("valid_until_utc"), "valid_until_utc")
    if valid_until <= issued or (valid_until - issued).total_seconds() > 30 * 86400:
        raise SemanticDraftAuthorityError("Draft authority validity window changed")
    if value.get("bindings") != dict(expected_bindings):
        raise SemanticDraftAuthorityError("Draft authority bindings changed")
    model = _deployment_model(value.get("deployment_model"))
    evaluated_model = expected_bindings.get("evaluated_draft_model") or {}
    if model != {
        "locator": evaluated_model.get("locator"),
        "raw_sha256": evaluated_model.get("raw_sha256"),
        "artifact_sha256": evaluated_model.get("raw_sha256"),
        "model_version": evaluated_model.get("model_version"),
    }:
        raise SemanticDraftAuthorityError(
            "deployment model is not the exact future-evaluated Draft model"
        )
    reviews = value.get("reviews")
    if not isinstance(reviews, list) or len(reviews) != 2:
        raise SemanticDraftAuthorityError("two Draft deployment reviews are required")
    excluded = set(
        expected_bindings["reviewer_ids_excluded_from_deployment_authority"]
    )
    reviewers: set[str] = set()
    scopes: set[str] = set()
    for review in reviews:
        if not isinstance(review, Mapping) or set(review) != {
            "review_scope",
            "reviewer_id",
            "reviewed_at_utc",
            "attestation",
        }:
            raise SemanticDraftAuthorityError("Draft review structure changed")
        scope = _nonempty(review.get("review_scope"), "review_scope")
        reviewer = _nonempty(review.get("reviewer_id"), "reviewer_id")
        if (
            scope not in REVIEW_SCOPES
            or reviewer in excluded
            or review.get("attestation") != REVIEW_SCOPES[scope]
            or _timestamp(review.get("reviewed_at_utc"), "reviewed_at_utc") > issued
        ):
            raise SemanticDraftAuthorityError("Draft review is not independent")
        reviewers.add(reviewer)
        scopes.add(scope)
    if len(reviewers) != 2 or scopes != set(REVIEW_SCOPES):
        raise SemanticDraftAuthorityError("Draft reviews are not scope-complete")
    if value.get("deployment_policy") != DEPLOYMENT_POLICY:
        raise SemanticDraftAuthorityError("Draft deployment policy changed")
    if value.get("authority") != AUTHORITY or value.get("claim_ceiling") != CLAIM_CEILING:
        raise SemanticDraftAuthorityError("Draft authority exceeds scope")
    return {**value, "deployment_model": model}


def load_active_semantic_draft_authority_v1(
    *,
    root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    digest = environment.get(EXTERNAL_SHA256_ENV)
    if not digest:
        raise SemanticDraftAuthorityError("external Draft-authority pin missing")
    _sha(digest, "external Draft-authority pin")
    path = root / AUTHORITY_LOCATOR
    if path.is_symlink() or not path.is_file():
        raise SemanticDraftAuthorityError("semantic Draft authority unavailable")
    raw = path.read_bytes()
    if _sha256(raw) != digest:
        raise SemanticDraftAuthorityError("Draft authority external pin changed")
    expected = current_expected_bindings(root=root, environment=environment)
    receipt = validate_semantic_draft_authority_v1(
        _object(raw, "semantic Draft authority"), expected_bindings=expected
    )
    model_ref = receipt["deployment_model"]
    model_path = root / model_ref["locator"]
    if model_path.is_symlink() or not model_path.is_file():
        raise SemanticDraftAuthorityError("approved Draft model unavailable")
    model_raw = model_path.read_bytes()
    if (
        _sha256(model_raw) != model_ref["raw_sha256"]
        or model_ref["artifact_sha256"] != model_ref["raw_sha256"]
    ):
        raise SemanticDraftAuthorityError("approved Draft model changed")
    try:
        model = TerminalModel.from_artifact_bytes(
            model_raw,
            expected_artifact_sha256=model_ref["artifact_sha256"],
        )
    except TerminalDraftError as exc:
        raise SemanticDraftAuthorityError(
            "approved Draft model changed"
        ) from exc
    if model.model_version != model_ref["model_version"]:
        raise SemanticDraftAuthorityError("approved Draft model identity changed")
    observed = as_of or datetime.now(timezone.utc)
    if observed.tzinfo is None:
        raise SemanticDraftAuthorityError("authority clock must be timezone-aware")
    observed = observed.astimezone(timezone.utc)
    if not (
        _timestamp(receipt["issued_at_utc"], "issued_at_utc")
        <= observed
        <= _timestamp(receipt["valid_until_utc"], "valid_until_utc")
    ):
        raise SemanticDraftAuthorityError("semantic Draft authority is not active")
    return {
        "receipt": receipt,
        "receipt_raw_sha256": digest,
        "bindings": expected,
        "deployment_model": model_ref,
        "private_terminal_draft_component_authorized": True,
        "private_equal_strength_score_authorized": True,
        "private_event_probability_authorized": False,
        "public_probability_authorized": False,
        "betting_authorized": False,
    }


__all__ = [
    "AUTHORITY",
    "AUTHORITY_LOCATOR",
    "CLAIM_CEILING",
    "DEPLOYMENT_POLICY",
    "EXTERNAL_SHA256_ENV",
    "PRODUCTION_SOURCE_LOCATORS",
    "REVIEW_SCOPES",
    "SCHEMA_VERSION",
    "SOURCE_LOCATOR",
    "SemanticDraftAuthorityError",
    "current_expected_bindings",
    "load_active_semantic_draft_authority_v1",
    "validate_semantic_draft_authority_v1",
]
