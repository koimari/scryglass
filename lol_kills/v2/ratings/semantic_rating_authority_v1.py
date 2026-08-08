"""Semantic deployment authority for private player and team ratings.

An event-rating registry proves identity, timing, and exact bytes.  It does not
by itself prove that those bytes descend from the independently reviewed future
evaluation.  This short-lived authority supplies that missing semantic link.

The authority is deliberately rating-component-only.  It cannot authorize a
match probability, odds, expected value, recommendation, stake, or transaction.
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
from lol_kills.v2.ratings.player import (
    multileague_v3_prediction_ledger as ratings_ledger,
)


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/ratings/semantic_rating_authority_v1.py"
SCHEMA_VERSION = "scryglass:semantic-private-rating-authority:v1"
AUTHORITY_LOCATOR = Path(
    "data/lol/private_rating_authority/semantic-rating-authority-v1.json"
)
EXTERNAL_SHA256_ENV = "SCRYGLASS_SEMANTIC_PRIVATE_RATING_AUTHORITY_SHA256"
PRODUCTION_SOURCE_LOCATORS = (
    "lol_kills/private_rating_authority.py",
    "lol_kills/pregame_roster_capture.py",
    "lol_kills/v2/ratings/player/multileague_v3_prediction_ledger.py",
)
ARTIFACT_KINDS = (
    "source_snapshot",
    "player_model",
    "team_model",
    "evaluation",
    "reliability",
    "uncertainty",
)
REVIEW_SCOPES = {
    "RATING_MODEL_DEPLOYMENT": {
        "reviewer_independent_of_model_candidate_evaluator_outcome_and_production_code_authors": True,
        "exact_independently_registered_future_result_and_model_lineage_verified": True,
        "player_and_team_estimands_replayed_from_the_approved_artifacts": True,
        "no_post_outcome_model_threshold_estimand_or_uncertainty_change_found": True,
        "review_not_generated_by_the_evaluated_system": True,
    },
    "RATING_DATA_AND_ROSTER_DEPLOYMENT": {
        "reviewer_independent_of_source_roster_rating_and_event_registry_producers": True,
        "source_identity_role_roster_freshness_and_temporal_paths_verified": True,
        "unidentified_team_components_remain_unavailable_not_zero": True,
        "event_receipts_must_bind_exact_approved_deployment_artifacts": True,
        "review_not_generated_by_the_evaluated_system": True,
    },
}
DEPLOYMENT_POLICY = {
    "scope": "private_pre_event_exact_roster_rating_components_only",
    "maximum_data_age_seconds": 14 * 24 * 60 * 60,
    "required_roles": ["top", "jungle", "mid", "bot", "support"],
    "required_team_components": [
        "player_aggregate",
        "organization_residual",
        "lineup_synergy",
        "team_policy",
    ],
    "unidentified_components_must_be_null": True,
    "posterior_interval_required": True,
    "event_roster_registry_required": True,
    "event_rating_registry_required": True,
}
AUTHORITY = {
    "private_player_rating_authority": True,
    "private_team_rating_authority": True,
    "private_rating_component_input_authority": True,
    "match_probability_authority": False,
    "fair_odds_authority": False,
    "expected_value_authority": False,
    "recommendation_authority": False,
    "public_rating_authority": False,
    "transaction_authority": False,
    "stake_authority": False,
    "betting_authority": False,
}
CLAIM_CEILING = (
    "Short-lived private player/team rating-component authority only. Every "
    "event must separately replay exact pre-event roster, freshness, registry, "
    "and approved artifact bindings. This authority does not produce or "
    "authorize probability, odds, expected value, a recommendation, a stake, "
    "a transaction, public betting content, or a claim of certainty."
)


class SemanticRatingAuthorityError(RuntimeError):
    """Future evaluation, deployment evidence, review, or pin failed closed."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_sha256(value: Any) -> str:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SemanticRatingAuthorityError(
            "rating runtime lineage is not canonical"
        ) from exc
    return _sha256(raw)


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise SemanticRatingAuthorityError(f"{field} must be a lowercase SHA-256")
    return value


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SemanticRatingAuthorityError(f"{field} must be nonempty")
    return value.strip()


def _timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise SemanticRatingAuthorityError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise SemanticRatingAuthorityError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SemanticRatingAuthorityError(f"duplicate JSON key: {key}")
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
        raise SemanticRatingAuthorityError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise SemanticRatingAuthorityError(f"{label} must contain an object")
    return value


def _source_record(root: Path, locator: str) -> dict[str, Any]:
    path = root / locator
    if path.is_symlink() or not path.is_file():
        raise SemanticRatingAuthorityError(f"production source unavailable: {locator}")
    return {
        "locator": locator,
        "bytes": path.stat().st_size,
        "raw_sha256": evaluation._sha256_path(path),
    }


def _evaluated_rating_runtime_binding(
    snapshot: Mapping[str, Any], *, root: Path
) -> dict[str, Any]:
    entries = (snapshot.get("ratings_ledger_candidate") or {}).get("entries")
    if not isinstance(entries, list) or not entries:
        raise SemanticRatingAuthorityError(
            "phase-one snapshot has no evaluated rating predictions"
        )
    shared_lineages: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise SemanticRatingAuthorityError(
                "phase-one ratings ledger entry is malformed"
            )
        raw = evaluation._read_regular(
            root, entry["receipt_locator"], "evaluated ratings prediction"
        )
        receipt = ratings_ledger.validate_pre_event_prediction_receipt(
            evaluation._strict_object(raw, "evaluated ratings prediction"),
            root=root,
        )
        source_locks = receipt["source_locks"]
        runtime_records = [
            record
            for record in source_locks
            if isinstance(record, Mapping)
            and record.get("locator") == ratings_ledger.SOURCE_LOCATOR
        ]
        if len(runtime_records) != 1:
            raise SemanticRatingAuthorityError(
                "evaluated rating receipt lacks one exact runtime source"
            )
        lineage = {
            "protocol": receipt["protocol"],
            "source_snapshot": receipt["source_snapshot"],
            "source_preflight": receipt["source_preflight"],
            "source_locks": source_locks,
            "model_ids": sorted(receipt["evaluation_predictions"]),
            "runtime_source": dict(runtime_records[0]),
        }
        shared_lineages[_canonical_sha256(lineage)] = lineage
    if len(shared_lineages) != 1:
        raise SemanticRatingAuthorityError(
            "phase-one ratings cohort does not bind exactly one runtime lineage"
        )
    lineage_sha256, lineage = next(iter(shared_lineages.items()))
    runtime = lineage["runtime_source"]
    return {
        "runtime_locator": runtime["locator"],
        "runtime_raw_sha256": runtime["raw_sha256"],
        "runtime_bytes": runtime["bytes"],
        "lineage_sha256": lineage_sha256,
        "protocol": lineage["protocol"],
        "source_snapshot": lineage["source_snapshot"],
        "source_preflight": lineage["source_preflight"],
        "source_locks_sha256": _canonical_sha256(lineage["source_locks"]),
        "model_ids": lineage["model_ids"],
        "prediction_receipts": len(entries),
        "same_exact_runtime_lineage_used_for_every_evaluated_prediction": True,
        "player_and_team_outputs_are_joint_views_of_one_runtime": True,
    }


def current_expected_bindings(
    *, root: Path = ROOT, environment: Mapping[str, str] = os.environ
) -> dict[str, Any]:
    """Replay the exact independently registered joint future evaluation."""

    path = root / registry.REGISTRY_LOCATOR
    digest = environment.get(registry.EXTERNAL_SHA256_ENV)
    if path.is_symlink() or not path.is_file() or not digest:
        raise SemanticRatingAuthorityError("phase-one evaluation registry unavailable")
    registry_payload = _object(path.read_bytes(), "phase-one evaluation registry")
    result_locator = (registry_payload.get("result_binding") or {}).get(
        "result_locator"
    )
    if not isinstance(result_locator, str):
        raise SemanticRatingAuthorityError("phase-one result locator missing")
    try:
        binding = registry.expected_result_binding(
            result_locator=result_locator,
            root=root,
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
        raise SemanticRatingAuthorityError(
            "phase-one evaluation registry or result is invalid"
        ) from exc
    ratings = result.get("ratings_evaluation") or {}
    if (
        registered.get("phase_one_models_independently_passed") is not True
        or result.get("phase_one_models_passed") is not True
        or ratings.get("primary_gate_passed") is not True
        or ratings.get("reliability_gate_passed") is not True
        or ratings.get("passed") is not True
    ):
        raise SemanticRatingAuthorityError(
            "independently registered future rating gates did not pass"
        )
    result_inputs = result["inputs"]
    snapshot_raw, snapshot = evaluation._snapshot(
        root, result_inputs["snapshot_locator"]
    )
    if (
        _sha256(snapshot_raw) != result_inputs["snapshot_raw_sha256"]
        or snapshot["artifact_sha256"]
        != result_inputs["snapshot_artifact_sha256"]
    ):
        raise SemanticRatingAuthorityError(
            "phase-one snapshot does not match the registered result"
        )
    evaluated_runtime = _evaluated_rating_runtime_binding(snapshot, root=root)
    result_reviews = registered["receipt"]["reviews"]
    return {
        "phase_one_evaluation": {
            "registry_locator": registry.REGISTRY_LOCATOR.as_posix(),
            "registry_raw_sha256": registered["receipt_raw_sha256"],
            "registry_id": registered["receipt"]["registry_id"],
            "result_locator": result_locator,
            "result_raw_sha256": binding["result_raw_sha256"],
            "result_artifact_sha256": result["artifact_sha256"],
            "run_id": result["run_id"],
            "snapshot_locator": result_inputs["snapshot_locator"],
            "snapshot_raw_sha256": result_inputs["snapshot_raw_sha256"],
            "snapshot_artifact_sha256": result_inputs[
                "snapshot_artifact_sha256"
            ],
            "maps": result_inputs["maps"],
            "series": result_inputs["series"],
            "ratings_primary_gate_passed": True,
            "ratings_reliability_gate_passed": True,
            "ratings_future_evaluation_independently_passed": True,
        },
        "evaluated_rating_runtime": evaluated_runtime,
        "production_sources": [
            _source_record(root, locator)
            for locator in (SOURCE_LOCATOR, *PRODUCTION_SOURCE_LOCATORS)
        ],
        "reviewer_ids_excluded_from_deployment_authority": sorted(
            review["reviewer_id"] for review in result_reviews
        ),
    }


def _artifact_reference(value: Any, field: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"locator", "raw_sha256"}:
        raise SemanticRatingAuthorityError(f"{field} structure changed")
    locator = PurePosixPath(_nonempty(value.get("locator"), f"{field}.locator"))
    if locator.is_absolute() or any(part in {"", ".", ".."} for part in locator.parts):
        raise SemanticRatingAuthorityError(f"{field}.locator is unsafe")
    return {
        "locator": locator.as_posix(),
        "raw_sha256": _sha(value.get("raw_sha256"), f"{field}.raw_sha256"),
    }


def _validate_deployment_artifacts(
    value: Any, *, expected_bindings: Mapping[str, Any]
) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or set(value) != set(ARTIFACT_KINDS):
        raise SemanticRatingAuthorityError("deployment artifact inventory changed")
    artifacts = {
        kind: _artifact_reference(value[kind], f"deployment_artifacts.{kind}")
        for kind in ARTIFACT_KINDS
    }
    phase_one = expected_bindings.get("phase_one_evaluation") or {}
    expected_snapshot = {
        "locator": phase_one.get("snapshot_locator"),
        "raw_sha256": phase_one.get("snapshot_raw_sha256"),
    }
    expected_result = {
        "locator": phase_one.get("result_locator"),
        "raw_sha256": phase_one.get("result_raw_sha256"),
    }
    if artifacts["source_snapshot"] != expected_snapshot:
        raise SemanticRatingAuthorityError(
            "deployment source snapshot is not the evaluated snapshot"
        )
    for kind in ("evaluation", "reliability", "uncertainty"):
        if artifacts[kind] != expected_result:
            raise SemanticRatingAuthorityError(
                f"deployment {kind} is not the independently registered result"
            )
    runtime = expected_bindings.get("evaluated_rating_runtime") or {}
    expected_runtime = {
        "locator": runtime.get("runtime_locator"),
        "raw_sha256": runtime.get("runtime_raw_sha256"),
    }
    for kind in ("player_model", "team_model"):
        if artifacts[kind] != expected_runtime:
            raise SemanticRatingAuthorityError(
                f"deployment {kind} is not the exact future-evaluated runtime"
            )
    return artifacts


def validate_semantic_rating_authority_v1(
    payload: Mapping[str, Any], *, expected_bindings: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise SemanticRatingAuthorityError("rating authority must be an object")
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
        "deployment_artifacts",
        "deployment_policy",
        "authority",
        "claim_ceiling",
    }:
        raise SemanticRatingAuthorityError("rating authority fields are not exact")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != "APPROVED"
        or value.get("scope") != "PRIVATE_PLAYER_TEAM_RATINGS_ONLY"
    ):
        raise SemanticRatingAuthorityError("rating authority identity changed")
    _nonempty(value.get("authority_id"), "authority_id")
    issued = _timestamp(value.get("issued_at_utc"), "issued_at_utc")
    valid_until = _timestamp(value.get("valid_until_utc"), "valid_until_utc")
    if valid_until <= issued or (valid_until - issued).total_seconds() > 30 * 86400:
        raise SemanticRatingAuthorityError("rating authority validity window changed")
    if value.get("bindings") != dict(expected_bindings):
        raise SemanticRatingAuthorityError("rating authority bindings changed")
    artifacts = _validate_deployment_artifacts(
        value.get("deployment_artifacts"), expected_bindings=expected_bindings
    )
    reviews = value.get("reviews")
    if not isinstance(reviews, list) or len(reviews) != 2:
        raise SemanticRatingAuthorityError("two deployment reviews are required")
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
            raise SemanticRatingAuthorityError("deployment review structure changed")
        scope = _nonempty(review.get("review_scope"), "review_scope")
        reviewer = _nonempty(review.get("reviewer_id"), "reviewer_id")
        if (
            scope not in REVIEW_SCOPES
            or reviewer in excluded
            or review.get("attestation") != REVIEW_SCOPES[scope]
            or _timestamp(review.get("reviewed_at_utc"), "reviewed_at_utc") > issued
        ):
            raise SemanticRatingAuthorityError(
                "rating deployment review is not independent"
            )
        reviewers.add(reviewer)
        scopes.add(scope)
    if len(reviewers) != 2 or scopes != set(REVIEW_SCOPES):
        raise SemanticRatingAuthorityError(
            "rating deployment reviews are not scope-complete"
        )
    if value.get("deployment_policy") != DEPLOYMENT_POLICY:
        raise SemanticRatingAuthorityError("rating deployment policy changed")
    if value.get("authority") != AUTHORITY or value.get("claim_ceiling") != CLAIM_CEILING:
        raise SemanticRatingAuthorityError("rating authority exceeds scope")
    return {**value, "deployment_artifacts": artifacts}


def load_active_semantic_rating_authority_v1(
    *,
    root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    digest = environment.get(EXTERNAL_SHA256_ENV)
    if not digest:
        raise SemanticRatingAuthorityError("external rating-authority pin missing")
    _sha(digest, "external rating-authority pin")
    path = root / AUTHORITY_LOCATOR
    if path.is_symlink() or not path.is_file():
        raise SemanticRatingAuthorityError("semantic rating authority unavailable")
    raw = path.read_bytes()
    if _sha256(raw) != digest:
        raise SemanticRatingAuthorityError("rating authority external pin changed")
    expected = current_expected_bindings(root=root, environment=environment)
    receipt = validate_semantic_rating_authority_v1(
        _object(raw, "semantic rating authority"), expected_bindings=expected
    )
    for kind, reference in receipt["deployment_artifacts"].items():
        artifact_path = root / reference["locator"]
        if artifact_path.is_symlink() or not artifact_path.is_file():
            raise SemanticRatingAuthorityError(
                f"approved deployment artifact unavailable: {kind}"
            )
        if _sha256(artifact_path.read_bytes()) != reference["raw_sha256"]:
            raise SemanticRatingAuthorityError(
                f"approved deployment artifact changed: {kind}"
            )
    observed = as_of or datetime.now(timezone.utc)
    if observed.tzinfo is None:
        raise SemanticRatingAuthorityError("authority clock must be timezone-aware")
    observed = observed.astimezone(timezone.utc)
    if not (
        _timestamp(receipt["issued_at_utc"], "issued_at_utc")
        <= observed
        <= _timestamp(receipt["valid_until_utc"], "valid_until_utc")
    ):
        raise SemanticRatingAuthorityError("semantic rating authority is not active")
    return {
        "receipt": receipt,
        "receipt_raw_sha256": digest,
        "bindings": expected,
        "deployment_artifacts": receipt["deployment_artifacts"],
        "private_player_rating_authorized": True,
        "private_team_rating_authorized": True,
        "private_rating_component_input_authorized": True,
        "match_probability_authorized": False,
        "betting_authorized": False,
    }


__all__ = [
    "ARTIFACT_KINDS",
    "AUTHORITY",
    "AUTHORITY_LOCATOR",
    "CLAIM_CEILING",
    "DEPLOYMENT_POLICY",
    "EXTERNAL_SHA256_ENV",
    "PRODUCTION_SOURCE_LOCATORS",
    "REVIEW_SCOPES",
    "SCHEMA_VERSION",
    "SOURCE_LOCATOR",
    "SemanticRatingAuthorityError",
    "current_expected_bindings",
    "load_active_semantic_rating_authority_v1",
    "validate_semantic_rating_authority_v1",
]
