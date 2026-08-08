"""Independent registration of the fitted recalibration and bootstrap identity.

The registry can be issued only after the terminal phase-one pass, an exact
recalibration replay, and a 2,000-draw verification run on a fresh event that
is explicitly excluded from both phase one and phase two.  Its externally
pinned bytes permit a separate phase-two opening review; they do not open
phase two or authorize an event probability, price, recommendation, or wager.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import full_pipeline_uncertainty_v1 as uncertainty
from . import fast_event_uncertainty_v1 as fast_uncertainty
from . import event_rating_bootstrap_v1 as rating_bootstrap
from . import phase_one_evaluation_v1 as evaluation
from . import phase_one_recalibration_v1 as recalibration
from .probability_pipeline_readiness_registry_v1 import (
    REGISTERED_READINESS_ARTIFACT_SHA256,
    REGISTERED_READINESS_LOCATOR,
    REGISTERED_READINESS_RAW_SHA256,
    validate_registered_probability_pipeline_readiness_v1,
)


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/market/calibration_uncertainty_registry_v1.py"
SCHEMA_VERSION = (
    "scryglass:phase-one-recalibration-uncertainty-independent-registry:v1"
)
REGISTRY_LOCATOR = Path(
    "data/lol/private_market_authority/"
    "phase-one-recalibration-uncertainty-registry-v1.json"
)
EXTERNAL_SHA256_ENV = "SCRYGLASS_PRIVATE_MATCH_WINNER_CALIBRATION_SHA256"
REVIEW_SCOPES = {
    "RECALIBRATION_REPRODUCTION": {
        "reviewer_independent_of_model_authors_candidate_selectors_evaluators_and_outcome_custodians": True,
        "phase_one_pass_registry_and_probability_pipeline_readiness_hashes_verified": True,
        "complete_phase_one_map_cohort_and_stored_predictions_verified": True,
        "combined_and_rating_only_bounded_fits_recomputed_exactly": True,
        "optimizer_bounds_tolerances_convergence_and_runtime_identity_verified": True,
        "no_candidate_hyperparameter_threshold_or_cohort_reselection_found": True,
        "review_not_generated_by_the_evaluated_system": True,
    },
    "FULL_PIPELINE_UNCERTAINTY_REPRODUCTION": {
        "reviewer_independent_of_model_authors_candidate_selectors_evaluators_and_outcome_custodians": True,
        "probability_pipeline_source_and_runtime_hashes_verified": True,
        "all_2000_seeded_series_resamples_and_sample_digests_replayed": True,
        "rating_state_draft_terms_and_recalibration_refit_each_draw_verified": True,
        "draw_probabilities_hash_and_percentile_interval_reproduced_exactly": True,
        "all_2000_fast_and_frozen_slow_draw_records_match_exactly": True,
        "pre_event_rating_leg_and_terminal_completion_source_hashes_verified": True,
        "verification_target_excluded_from_phase_one_and_phase_two": True,
        "verification_target_outcome_and_market_price_not_accessed": True,
        "review_not_generated_by_the_evaluated_system": True,
    },
}
AUTHORITY = {
    "recalibration_identity_authority": True,
    "uncertainty_implementation_identity_authority": True,
    "verification_event_probability_authority": False,
    "phase_two_opening_authority": False,
    "event_probability_authority": False,
    "odds_authority": False,
    "expected_value_authority": False,
    "recommendation_authority": False,
    "transaction_authority": False,
    "betting_authority": False,
}
CLAIM_CEILING = (
    "Independent identity registration for one fitted phase-one recalibration "
    "and the exact 2,000-draw uncertainty implementation only. The verification "
    "event is excluded from phase two. Separate phase-two opening, event "
    "probability, quote, settlement, market, and betting authority remain required."
)


class CalibrationUncertaintyRegistryError(RuntimeError):
    """The independent registry or one of its exact bindings failed closed."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CalibrationUncertaintyRegistryError(
            f"{field} must be a lowercase SHA-256"
        )
    return value


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CalibrationUncertaintyRegistryError(f"{field} must be nonempty")
    return value.strip()


def _timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise CalibrationUncertaintyRegistryError(
            f"{field} must be RFC-3339"
        ) from exc
    if parsed.tzinfo is None:
        raise CalibrationUncertaintyRegistryError(
            f"{field} must include a timezone"
        )
    return parsed.astimezone(timezone.utc)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CalibrationUncertaintyRegistryError(
                f"duplicate JSON key: {key}"
            )
        result[key] = value
    return result


def _object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalibrationUncertaintyRegistryError(
            f"{label} is not strict JSON"
        ) from exc
    if not isinstance(value, dict):
        raise CalibrationUncertaintyRegistryError(
            f"{label} must contain an object"
        )
    return value


def _read(root: Path, locator: str, label: str) -> bytes:
    try:
        return evaluation._read_regular(root, locator, label)
    except Exception as exc:
        raise CalibrationUncertaintyRegistryError(str(exc)) from exc


def _recomputed_models(
    *, result: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    rows, _snapshot_raw, snapshot, _outcome_raw, _outcomes = (
        recalibration._phase_one_rows(result=result, root=root)
    )
    labels = [int(row["blue_win"]) for row in rows]
    combined = recalibration.fit_bounded_recalibration(
        [float(row["ratings_plus_draft"]) for row in rows], labels
    )
    rating_only = recalibration.fit_bounded_recalibration(
        [float(row["ratings_only"]) for row in rows], labels
    )
    return {
        "ratings_plus_draft": {
            "raw_probability_field": "ratings_plus_draft.p_blue",
            **combined,
        },
        "ratings_only": {
            "raw_probability_field": "ratings_only.p_blue",
            **rating_only,
        },
    }, snapshot


def expected_registration_binding(
    *,
    recalibration_artifact_locator: str,
    verification_uncertainty_locator: str,
    verification_fast_uncertainty_locator: str,
    root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
) -> dict[str, Any]:
    readiness = validate_registered_probability_pipeline_readiness_v1(root=root)
    recalibration_locator = evaluation._locator(
        recalibration_artifact_locator,
        recalibration.OUTPUT_PREFIX,
        "recalibration_artifact_locator",
    )
    recalibration_raw = _read(
        root, recalibration_locator, "phase-one recalibration artifact"
    )
    try:
        calibrated = recalibration.validate_phase_one_recalibration_artifact(
            _object(recalibration_raw, "phase-one recalibration artifact")
        )
    except Exception as exc:
        raise CalibrationUncertaintyRegistryError(
            "phase-one recalibration artifact is invalid"
        ) from exc
    result_locator = str(calibrated["inputs"]["phase_one_result_locator"])
    try:
        registered_pass, result, result_raw = recalibration._registered_pass(
            result_locator=result_locator,
            root=root,
            environment=environment,
        )
    except Exception as exc:
        raise CalibrationUncertaintyRegistryError(
            "phase-one evaluation pass is not independently registered"
        ) from exc
    if (
        result.get("phase_one_models_passed") is not True
        or _sha256(result_raw)
        != calibrated["inputs"]["phase_one_result_raw_sha256"]
        or result["artifact_sha256"]
        != calibrated["inputs"]["phase_one_result_artifact_sha256"]
    ):
        raise CalibrationUncertaintyRegistryError(
            "recalibration is not bound to a passing phase-one result"
        )
    recomputed, snapshot = _recomputed_models(result=result, root=root)
    if calibrated.get("models") != recomputed:
        raise CalibrationUncertaintyRegistryError(
            "phase-one recalibration does not exactly recompute"
        )

    verification_locator = evaluation._locator(
        verification_uncertainty_locator,
        uncertainty.OUTPUT_PREFIX,
        "verification_uncertainty_locator",
    )
    verification_raw = _read(
        root, verification_locator, "uncertainty verification artifact"
    )
    try:
        verification = uncertainty.validate_event_uncertainty_candidate(
            _object(verification_raw, "uncertainty verification artifact"),
            root=root,
            environment=environment,
        )
    except Exception as exc:
        raise CalibrationUncertaintyRegistryError(
            "uncertainty verification artifact is invalid"
        ) from exc
    if (
        verification["inputs"]["phase_one_result_locator"] != result_locator
        or verification["inputs"]["phase_one_result_raw_sha256"]
        != _sha256(result_raw)
        or verification["inputs"]["recalibration_artifact_locator"]
        != recalibration_locator
        or verification["inputs"]["recalibration_artifact_raw_sha256"]
        != _sha256(recalibration_raw)
        or verification["bootstrap_contract"]["resamples"]
        != uncertainty.RESAMPLES
    ):
        raise CalibrationUncertaintyRegistryError(
            "uncertainty verification inputs changed"
        )
    phase_one_predictions = {
        str(item["prediction_locator"])
        for item in snapshot["draft_ledger_candidate"]["entries"]
    }
    target_locator = str(verification["event"]["target_prediction_locator"])
    if target_locator in phase_one_predictions:
        raise CalibrationUncertaintyRegistryError(
            "uncertainty verification target belongs to phase one"
        )
    target_raw, target, _ratings, _metadata = uncertainty._target(
        root, target_locator
    )
    if (
        _sha256(target_raw)
        != verification["event"]["target_prediction_raw_sha256"]
        or target["artifact_sha256"]
        != verification["event"]["target_prediction_artifact_sha256"]
    ):
        raise CalibrationUncertaintyRegistryError(
            "uncertainty verification target binding changed"
        )
    fitted_at = _timestamp(calibrated["fitted_at_utc"], "recalibration.fitted_at")
    target_captured = _timestamp(
        target["captured_at_utc"], "verification_target.captured_at"
    )
    if target_captured <= fitted_at:
        raise CalibrationUncertaintyRegistryError(
            "uncertainty verification target predates recalibration"
        )
    fast_locator = evaluation._locator(
        verification_fast_uncertainty_locator,
        fast_uncertainty.OUTPUT_PREFIX,
        "verification_fast_uncertainty_locator",
    )
    fast_raw = _read(
        root, fast_locator, "fast uncertainty verification artifact"
    )
    try:
        fast_verification = fast_uncertainty.validate_fast_event_uncertainty_v1(
            _object(fast_raw, "fast uncertainty verification artifact"),
            root=root,
            environment=environment,
        )
    except Exception as exc:
        raise CalibrationUncertaintyRegistryError(
            "fast uncertainty verification artifact is invalid"
        ) from exc
    fast_candidate = fast_verification["frozen_contract_candidate"]
    if (
        fast_candidate["event"] != verification["event"]
        or fast_candidate["inputs"] != verification["inputs"]
        or fast_candidate["point_calculation"]
        != verification["point_calculation"]
        or fast_candidate["uncertainty"] != verification["uncertainty"]
    ):
        raise CalibrationUncertaintyRegistryError(
            "fast uncertainty does not exactly match all frozen slow draws"
        )
    source_path = root / SOURCE_LOCATOR
    if source_path.is_symlink() or not source_path.is_file():
        raise CalibrationUncertaintyRegistryError(
            "calibration/uncertainty registry source is unavailable"
        )
    return {
        "probability_pipeline_readiness": {
            "locator": REGISTERED_READINESS_LOCATOR.as_posix(),
            "raw_sha256": REGISTERED_READINESS_RAW_SHA256,
            "artifact_sha256": REGISTERED_READINESS_ARTIFACT_SHA256,
            "runtime_identity": readiness["runtime_identity"],
        },
        "phase_one_result": {
            "locator": result_locator,
            "raw_sha256": _sha256(result_raw),
            "artifact_sha256": result["artifact_sha256"],
            "independent_evaluation_registry_raw_sha256": registered_pass[
                "receipt_raw_sha256"
            ],
            "independent_evaluation_registry_id": registered_pass["receipt"][
                "registry_id"
            ],
        },
        "recalibration": {
            "locator": recalibration_locator,
            "raw_sha256": _sha256(recalibration_raw),
            "artifact_sha256": calibrated["artifact_sha256"],
            "fitted_at_utc": calibrated["fitted_at_utc"],
            "models": calibrated["models"],
        },
        "uncertainty_verification": {
            "locator": verification_locator,
            "raw_sha256": _sha256(verification_raw),
            "artifact_sha256": verification["artifact_sha256"],
            "target_prediction_locator": target_locator,
            "target_prediction_artifact_sha256": target["artifact_sha256"],
            "target_prediction_captured_at_utc": target["captured_at_utc"],
            "target_excluded_from_phase_one": True,
            "target_must_be_excluded_from_phase_two": True,
            "resamples": uncertainty.RESAMPLES,
            "master_seed": uncertainty.MASTER_SEED,
            "draws_sha256": verification["uncertainty"]["draws_sha256"],
            "probability_interval_blue": verification["uncertainty"][
                "probability_interval_blue"
            ],
        },
        "fast_uncertainty_verification": {
            "locator": fast_locator,
            "raw_sha256": _sha256(fast_raw),
            "artifact_sha256": fast_verification["artifact_sha256"],
            "rating_bootstrap_locator": fast_verification["decomposition"][
                "rating_bootstrap_locator"
            ],
            "rating_bootstrap_artifact_sha256": fast_verification[
                "decomposition"
            ]["rating_bootstrap_artifact_sha256"],
            "terminal_draws_sha256": fast_verification["decomposition"][
                "terminal_draws_sha256"
            ],
            "all_2000_draw_records_equal_frozen_slow_path": True,
        },
        "implementation": {
            "recalibration_source_locator": recalibration.SOURCE_LOCATOR,
            "recalibration_source_raw_sha256": evaluation._sha256_path(
                root / recalibration.SOURCE_LOCATOR
            ),
            "uncertainty_source_locator": uncertainty.SOURCE_LOCATOR,
            "uncertainty_source_raw_sha256": evaluation._sha256_path(
                root / uncertainty.SOURCE_LOCATOR
            ),
            "event_rating_bootstrap_source_locator": rating_bootstrap.SOURCE_LOCATOR,
            "event_rating_bootstrap_source_raw_sha256": evaluation._sha256_path(
                root / rating_bootstrap.SOURCE_LOCATOR
            ),
            "fast_uncertainty_source_locator": fast_uncertainty.SOURCE_LOCATOR,
            "fast_uncertainty_source_raw_sha256": evaluation._sha256_path(
                root / fast_uncertainty.SOURCE_LOCATOR
            ),
            "registry_source_locator": SOURCE_LOCATOR,
            "registry_source_raw_sha256": _sha256(source_path.read_bytes()),
        },
    }


def validate_calibration_uncertainty_registry(
    payload: Mapping[str, Any], *, expected_binding: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise CalibrationUncertaintyRegistryError("registry must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version",
        "registry_id",
        "status",
        "registered_at_utc",
        "reviews",
        "binding",
        "decision",
        "authority",
        "claim_ceiling",
    }:
        raise CalibrationUncertaintyRegistryError("registry fields are not exact")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise CalibrationUncertaintyRegistryError("registry schema changed")
    _nonempty(value.get("registry_id"), "registry_id")
    registered_at = _timestamp(value.get("registered_at_utc"), "registered_at")
    if value.get("status") != "REGISTERED_FOR_SEPARATE_PHASE_TWO_OPENING":
        raise CalibrationUncertaintyRegistryError("registry status changed")
    binding = dict(expected_binding)
    if value.get("binding") != binding:
        raise CalibrationUncertaintyRegistryError("registry binding changed")
    for section in binding.values():
        if isinstance(section, Mapping):
            for key, item in section.items():
                if key.endswith("sha256"):
                    _sha(item, f"binding.{key}")
    fitted_at = _timestamp(
        binding["recalibration"]["fitted_at_utc"], "binding.recalibration.fitted_at"
    )
    target_captured = _timestamp(
        binding["uncertainty_verification"][
            "target_prediction_captured_at_utc"
        ],
        "binding.uncertainty_verification.target_captured_at",
    )
    if registered_at < target_captured or target_captured <= fitted_at:
        raise CalibrationUncertaintyRegistryError(
            "registry chronology changed"
        )
    reviews = value.get("reviews")
    if not isinstance(reviews, list) or len(reviews) != len(REVIEW_SCOPES):
        raise CalibrationUncertaintyRegistryError(
            "two independent specialized reviews are required"
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
            raise CalibrationUncertaintyRegistryError(
                "review structure changed"
            )
        scope = _nonempty(review.get("review_scope"), "review_scope")
        reviewer = _nonempty(review.get("reviewer_id"), "reviewer_id")
        reviewed_at = _timestamp(review.get("reviewed_at_utc"), "reviewed_at")
        if (
            scope not in REVIEW_SCOPES
            or review.get("attestation") != REVIEW_SCOPES[scope]
            or reviewed_at < target_captured
            or reviewed_at > registered_at
        ):
            raise CalibrationUncertaintyRegistryError(
                "review attestation or chronology changed"
            )
        reviewers.add(reviewer)
        scopes.add(scope)
    if len(reviewers) != 2 or scopes != set(REVIEW_SCOPES):
        raise CalibrationUncertaintyRegistryError(
            "reviews are not independent and scope-complete"
        )
    if value.get("decision") != {
        "recalibration_independently_registered": True,
        "uncertainty_implementation_independently_registered": True,
        "verification_target_excluded_from_phase_two": True,
        "phase_two_available_for_separate_opening_review": True,
        "phase_two_opened": False,
        "event_probability_authorized": False,
        "betting_authorized": False,
    }:
        raise CalibrationUncertaintyRegistryError("registry decision changed")
    if value.get("authority") != AUTHORITY:
        raise CalibrationUncertaintyRegistryError("registry exceeds authority")
    if value.get("claim_ceiling") != CLAIM_CEILING:
        raise CalibrationUncertaintyRegistryError("claim ceiling changed")
    return value


def load_pinned_calibration_uncertainty_registry(
    *,
    path: Path,
    external_sha256: str,
    expected_binding: Mapping[str, Any],
) -> dict[str, Any]:
    expected_digest = _sha(external_sha256, "external registry digest")
    if path.is_symlink() or not path.is_file():
        raise CalibrationUncertaintyRegistryError("registry is unavailable")
    raw = path.read_bytes()
    if _sha256(raw) != expected_digest:
        raise CalibrationUncertaintyRegistryError(
            "registry does not match its external pin"
        )
    receipt = validate_calibration_uncertainty_registry(
        _object(raw, "calibration/uncertainty registry"),
        expected_binding=expected_binding,
    )
    return {
        "status": "registered_for_separate_phase_two_opening",
        "receipt": receipt,
        "receipt_raw_sha256": expected_digest,
        "recalibration_independently_registered": True,
        "uncertainty_implementation_independently_registered": True,
        "phase_two_opening_authorized": False,
        "event_probability_authorized": False,
        "betting_authorized": False,
    }


def registry_template(
    *,
    registry_id: str,
    registered_at_utc: str,
    reviews: Sequence[Mapping[str, Any]],
    expected_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble a review-complete candidate that still needs an external pin."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "registry_id": _nonempty(registry_id, "registry_id"),
        "status": "REGISTERED_FOR_SEPARATE_PHASE_TWO_OPENING",
        "registered_at_utc": _timestamp(
            registered_at_utc, "registered_at_utc"
        ).isoformat(),
        "reviews": [dict(item) for item in reviews],
        "binding": dict(expected_binding),
        "decision": {
            "recalibration_independently_registered": True,
            "uncertainty_implementation_independently_registered": True,
            "verification_target_excluded_from_phase_two": True,
            "phase_two_available_for_separate_opening_review": True,
            "phase_two_opened": False,
            "event_probability_authorized": False,
            "betting_authorized": False,
        },
        "authority": dict(AUTHORITY),
        "claim_ceiling": CLAIM_CEILING,
    }
    return validate_calibration_uncertainty_registry(
        payload, expected_binding=expected_binding
    )


__all__ = [
    "AUTHORITY",
    "EXTERNAL_SHA256_ENV",
    "REGISTRY_LOCATOR",
    "REVIEW_SCOPES",
    "SCHEMA_VERSION",
    "SOURCE_LOCATOR",
    "CalibrationUncertaintyRegistryError",
    "expected_registration_binding",
    "load_pinned_calibration_uncertainty_registry",
    "registry_template",
    "validate_calibration_uncertainty_registry",
]
