"""Freeze post-pass recalibration and full-pipeline uncertainty pre-boundary.

This receipt fixes implementation identity and exact resampling semantics before
the first phase-one map may enter the sealed cohort.  It neither opens outcomes
nor approves a fitted recalibration, an uncertainty artifact, phase two, a
probability, a price comparison, or a wager.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import platform
import tempfile
from typing import Any, Callable, Mapping

import numpy as np
import scipy

from lol_kills.v2.ratings.player.multileague_v3_future_protocol import (
    FUTURE_SEALED_START,
)
from lol_kills.v2.ratings.player import post_validation_refit_v1 as rating_refit

from . import full_pipeline_uncertainty_v1 as uncertainty
from . import event_rating_bootstrap_v1 as rating_bootstrap
from . import fast_event_uncertainty_v1 as fast_uncertainty
from . import phase_one_evaluation_v1 as evaluation
from . import phase_one_evaluation_registry_v1 as evaluation_registry
from . import phase_one_recalibration_v1 as recalibration
from .match_winner_future_protocol_registry_v1 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256 as MARKET_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_LOCATOR as MARKET_PROTOCOL_LOCATOR,
    REGISTERED_PROTOCOL_RAW_SHA256 as MARKET_PROTOCOL_RAW_SHA256,
    validate_registered_match_winner_future_protocol_v1,
)
from .phase_one_evaluation_readiness_registry_v1 import (
    REGISTERED_READINESS_ARTIFACT_SHA256 as EVALUATION_READINESS_ARTIFACT_SHA256,
    REGISTERED_READINESS_LOCATOR as EVALUATION_READINESS_LOCATOR,
    REGISTERED_READINESS_RAW_SHA256 as EVALUATION_READINESS_RAW_SHA256,
    validate_registered_phase_one_evaluation_readiness_v1,
)


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/market/probability_pipeline_readiness_v1.py"
SCHEMA_VERSION = "scryglass:probability-pipeline-readiness:v1"
RESULT_STATE = (
    "POST_PASS_RECALIBRATION_AND_FULL_PIPELINE_UNCERTAINTY_"
    "IMPLEMENTATION_FROZEN_PRE_BOUNDARY"
)
DEFAULT_OUTPUT = Path(
    "data/lol/v2/evaluation/match-winner-market-v1/phase-one/"
    "probability-pipeline-readiness-v1.json"
)
RECALIBRATION_UNCERTAINTY_REGISTRY_LOCATOR = Path(
    "data/lol/private_market_authority/"
    "phase-one-recalibration-uncertainty-registry-v1.json"
)
PHASE_TWO_OPENING_LOCATOR = Path(
    "data/lol/private_market_authority/phase-two-opening-authority-v1.json"
)
SOURCE_LOCK_PREFIX = (
    SOURCE_LOCATOR,
    "lol_kills/v2/market/phase_one_evaluation_readiness_registry_v1.py",
    rating_refit.SOURCE_LOCATOR,
)
AUTHORITY = {
    "phase_one_outcome_opening_authority": False,
    "recalibration_identity_authority": False,
    "uncertainty_identity_authority": False,
    "phase_two_opening_authority": False,
    "probability_authority": False,
    "odds_authority": False,
    "expected_value_authority": False,
    "recommendation_authority": False,
    "betting_authority": False,
}
CLAIM_CEILING = (
    "Pre-boundary implementation freeze only. No outcome was accessed and no "
    "fitted recalibration, uncertainty result, phase-two opening, probability, "
    "odds, expected value, recommendation, transaction, or betting authority "
    "is present."
)


class ProbabilityPipelineReadinessError(RuntimeError):
    """The implementation freeze, dependency, or empty state drifted."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProbabilityPipelineReadinessError(
            "readiness value is not canonical"
        ) from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProbabilityPipelineReadinessError(
            f"{field} must be RFC-3339"
        ) from exc
    if parsed.tzinfo is None:
        raise ProbabilityPipelineReadinessError(
            f"{field} must include a timezone"
        )
    return parsed.astimezone(timezone.utc)


def _source_record(root: Path, locator: str) -> dict[str, Any]:
    path = root / locator
    if path.is_symlink() or not path.is_file():
        raise ProbabilityPipelineReadinessError(
            f"source is unavailable: {locator}"
        )
    return {
        "locator": locator,
        "bytes": path.stat().st_size,
        "raw_sha256": _sha256_path(path),
    }


def _source_locks(root: Path) -> list[dict[str, Any]]:
    locators = [*SOURCE_LOCK_PREFIX]
    for record in fast_uncertainty._source_locks(root):
        locator = str(record["locator"])
        if locator not in locators:
            locators.append(locator)
    return [_source_record(root, locator) for locator in locators]


def _dependencies(root: Path) -> dict[str, Any]:
    try:
        evaluation_readiness = (
            validate_registered_phase_one_evaluation_readiness_v1(root=root)
        )
        market_protocol = validate_registered_match_winner_future_protocol_v1(
            root=root
        )
    except Exception as exc:
        raise ProbabilityPipelineReadinessError(
            "registered probability-pipeline dependency is invalid"
        ) from exc
    return {
        "phase_one_evaluation_readiness": {
            "locator": EVALUATION_READINESS_LOCATOR.as_posix(),
            "raw_sha256": EVALUATION_READINESS_RAW_SHA256,
            "artifact_sha256": EVALUATION_READINESS_ARTIFACT_SHA256,
            "result_state": evaluation_readiness["result_state"],
        },
        "match_winner_future_protocol": {
            "locator": MARKET_PROTOCOL_LOCATOR.as_posix(),
            "raw_sha256": MARKET_PROTOCOL_RAW_SHA256,
            "artifact_sha256": MARKET_PROTOCOL_ARTIFACT_SHA256,
            "result_state": market_protocol["result_state"],
        },
    }


def _runtime_identity() -> dict[str, str]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }


def _contract() -> dict[str, Any]:
    signatures = {
        "fit_bounded_recalibration": list(
            inspect.signature(recalibration.fit_bounded_recalibration).parameters
        ),
        "build_phase_one_recalibration": list(
            inspect.signature(recalibration.build_phase_one_recalibration).parameters
        ),
        "validate_phase_one_recalibration": list(
            inspect.signature(
                recalibration.validate_phase_one_recalibration_artifact
            ).parameters
        ),
        "build_event_uncertainty": list(
            inspect.signature(
                uncertainty.build_event_uncertainty_candidate
            ).parameters
        ),
        "validate_event_uncertainty": list(
            inspect.signature(
                uncertainty.validate_event_uncertainty_candidate
            ).parameters
        ),
        "build_event_rating_bootstrap": list(
            inspect.signature(
                rating_bootstrap.build_event_rating_bootstrap_v1
            ).parameters
        ),
        "validate_event_rating_bootstrap": list(
            inspect.signature(
                rating_bootstrap.validate_event_rating_bootstrap_v1
            ).parameters
        ),
        "build_fast_event_uncertainty": list(
            inspect.signature(
                fast_uncertainty.build_fast_event_uncertainty_v1
            ).parameters
        ),
        "validate_fast_event_uncertainty": list(
            inspect.signature(
                fast_uncertainty.validate_fast_event_uncertainty_v1
            ).parameters
        ),
        "build_fresh_rating_source_snapshot": list(
            inspect.signature(
                rating_refit.build_source_snapshot_manifest_v1
            ).parameters
        ),
        "validate_fresh_rating_source_snapshot": list(
            inspect.signature(
                rating_refit.validate_source_snapshot_manifest_v1
            ).parameters
        ),
        "build_post_validation_rating_refit": list(
            inspect.signature(
                rating_refit.build_post_validation_refit_v1
            ).parameters
        ),
        "validate_post_validation_rating_refit": list(
            inspect.signature(
                rating_refit.validate_post_validation_refit_v1
            ).parameters
        ),
    }
    expected = {
        "fit_bounded_recalibration": ["probabilities", "outcomes"],
        "build_phase_one_recalibration": [
            "phase_one_result_locator",
            "root",
            "environment",
            "clock",
        ],
        "validate_phase_one_recalibration": ["payload"],
        "build_event_uncertainty": [
            "phase_one_result_locator",
            "recalibration_artifact_locator",
            "target_prediction_locator",
            "rating_refit_locator",
            "workers",
            "root",
            "environment",
        ],
        "validate_event_uncertainty": ["payload", "root", "environment"],
        "build_event_rating_bootstrap": [
            "phase_one_result_locator",
            "rating_refit_locator",
            "workers",
            "root",
            "environment",
            "clock",
        ],
        "validate_event_rating_bootstrap": [
            "payload",
            "root",
            "environment",
        ],
        "build_fast_event_uncertainty": [
            "phase_one_result_locator",
            "recalibration_artifact_locator",
            "target_prediction_locator",
            "rating_bootstrap_locator",
            "workers",
            "root",
            "environment",
            "clock",
        ],
        "validate_fast_event_uncertainty": [
            "payload",
            "root",
            "environment",
        ],
        "build_fresh_rating_source_snapshot": [
            "snapshot_id",
            "maps_locator",
            "players_locator",
            "root",
            "clock",
        ],
        "validate_fresh_rating_source_snapshot": ["payload", "root"],
        "build_post_validation_rating_refit": [
            "phase_one_result_locator",
            "source_snapshot_locator",
            "roster_receipt_raw",
            "patch_receipt_raw",
            "root",
            "environment",
            "clock",
        ],
        "validate_post_validation_rating_refit": [
            "payload",
            "root",
            "environment",
        ],
    }
    if signatures != expected:
        raise ProbabilityPipelineReadinessError(
            "probability-pipeline signatures changed"
        )
    return {
        "schemas": {
            "phase_one_recalibration": recalibration.SCHEMA_VERSION,
            "event_full_pipeline_uncertainty": uncertainty.SCHEMA_VERSION,
            "event_rating_bootstrap": rating_bootstrap.SCHEMA_VERSION,
            "event_full_pipeline_uncertainty_fast": fast_uncertainty.SCHEMA_VERSION,
            "fresh_rating_source_snapshot": (
                rating_refit.SOURCE_SNAPSHOT_SCHEMA_VERSION
            ),
            "post_validation_event_rating_refit": rating_refit.SCHEMA_VERSION,
        },
        "builder_parameters": signatures,
        "recalibration": recalibration._optimization_contract(),
        "uncertainty": {
            "method": "series_cluster_bootstrap_full_prediction_pipeline",
            "confidence_level": 0.95,
            "resamples": uncertainty.RESAMPLES,
            "master_seed": uncertainty.MASTER_SEED,
            "percentile_interval": list(uncertainty.PERCENTILE_INTERVAL),
            "ratings_development_resampling": (
                "series_with_replacement_preserve_chronological_order"
            ),
            "draft_development_resampling": (
                "train_and_calibration_series_resampled_separately_with_replacement"
            ),
            "phase_one_recalibration_resampling": "series_with_replacement",
            "candidate_and_hyperparameters_fixed": True,
            "ratings_state_refit_in_each_resample": True,
            "draft_terms_refit_in_each_resample": True,
            "phase_one_recalibration_refit_in_each_resample": True,
            "phase_one_stored_predictions_used_for_recalibration_refit": True,
            "target_event_rating_and_draft_predictions_refit_in_each_resample": True,
            "fresh_post_validation_refit_exactly_bound": True,
            "fresh_point_rating_replayed_from_same_source_and_roster": True,
            "slow_and_fast_paths_share_exact_rating_draws": True,
            "target_event_outcome_or_market_price_used": False,
            "failure_or_nonconvergence_action": "event_probability_unavailable",
            "interval_is_epistemic_not_binary_outcome_guarantee": True,
        },
        "fresh_post_validation_rating_refit": {
            "required_before_every_phase_two_event_prediction": True,
            "model_family_and_hyperparameters_fixed_by_phase_one": True,
            "independently_registered_phase_one_pass_required": True,
            "immutable_exact_source_bytes_required": True,
            "strict_target_event_cutoff": True,
            "availability_embargo_hours": (
                rating_refit.AVAILABILITY_EMBARGO_HOURS
            ),
            "maximum_data_age_seconds": (
                rating_refit.MAXIMUM_DATA_AGE_SECONDS
            ),
            "exact_pre_event_roster_and_patch_receipts_required": True,
            "cross_team_covariance_retained": True,
            "unidentified_synergy_and_policy_remain_null": True,
            "match_probability_or_betting_authority": False,
            "full_pipeline_uncertainty_binding_status": (
                "wired_replayed_and_independently_reviewable"
            ),
        },
        "post_pass_sequence": [
            "independently_registered_phase_one_model_pass",
            "fit_exact_bounded_recalibration_once",
            "capture_immutable_fresh_pre_event_rating_source_bytes",
            "refit_locked_rating_family_before_target_event",
            "build_exact_slow_and_fast_fresh_refit_uncertainty_verification",
            "independently_register_recalibration_and_uncertainty_identity",
            "independently_open_disjoint_phase_two",
            "capture_event_specific_prediction_and_uncertainty_before_quote",
        ],
        "phase_one_pass_itself_authorizes_phase_two": False,
        "implementation_freeze_itself_authorizes_probability_or_betting": False,
    }


def _count_json(root: Path, prefix: PurePosixPath) -> int:
    path = root / Path(prefix)
    if not path.exists():
        return 0
    if path.is_symlink() or not path.is_dir():
        raise ProbabilityPipelineReadinessError(
            "probability-pipeline artifact directory is aliased"
        )
    return sum(1 for item in path.rglob("*.json") if item.is_file())


def _empty_state(root: Path) -> dict[str, Any]:
    state = {
        "phase_one_outcome_cohorts": _count_json(root, evaluation.OUTCOME_PREFIX),
        "phase_one_evaluation_outputs": _count_json(root, evaluation.OUTPUT_PREFIX),
        "recalibration_artifacts": _count_json(
            root, recalibration.OUTPUT_PREFIX
        ),
        "event_uncertainty_artifacts": _count_json(
            root, uncertainty.OUTPUT_PREFIX
        ),
        "phase_one_evaluation_registry_present": (
            root / evaluation_registry.REGISTRY_LOCATOR
        ).is_file(),
        "recalibration_uncertainty_registry_present": (
            root / RECALIBRATION_UNCERTAINTY_REGISTRY_LOCATOR
        ).is_file(),
        "phase_two_opening_present": (
            root / PHASE_TWO_OPENING_LOCATOR
        ).is_file(),
        "phase_two_started": False,
        "outcomes_accessed": False,
    }
    nonzero = (
        state["phase_one_outcome_cohorts"],
        state["phase_one_evaluation_outputs"],
        state["recalibration_artifacts"],
        state["event_uncertainty_artifacts"],
    )
    present = (
        state["phase_one_evaluation_registry_present"],
        state["recalibration_uncertainty_registry_present"],
        state["phase_two_opening_present"],
    )
    if any(nonzero) or any(present):
        raise ProbabilityPipelineReadinessError(
            "probability pipeline must be frozen before outcomes or post-pass artifacts"
        )
    return state

def build_probability_pipeline_readiness_v1(
    *,
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    locked = clock()
    if not isinstance(locked, datetime) or locked.tzinfo is None:
        raise ProbabilityPipelineReadinessError(
            "readiness clock must be timezone-aware"
        )
    locked = locked.astimezone(timezone.utc)
    if locked >= FUTURE_SEALED_START.replace(tzinfo=timezone.utc):
        raise ProbabilityPipelineReadinessError(
            "probability pipeline was not frozen pre-boundary"
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "locked_at_utc": locked.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_inside_builder",
            "observed_wall_clock_utc": locked.isoformat(),
            "user_supplied_timestamp_allowed": False,
            "lock_time_not_after_builder_observation": True,
        },
        "dependencies": _dependencies(root),
        "runtime_identity": _runtime_identity(),
        "probability_pipeline_contract": _contract(),
        "locked_empty_state": _empty_state(root),
        "source_locks": _source_locks(root),
        "decision_outputs": {
            "phase_one_model_pass": None,
            "recalibration_parameters": None,
            "uncertainty_interval": None,
            "phase_two_opening_authority": None,
            "match_probability": None,
            "fair_odds": None,
            "expected_value": None,
            "bet_recommendation": None,
        },
        "authority": dict(AUTHORITY),
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_probability_pipeline_readiness_v1(payload, root=root)


def validate_probability_pipeline_readiness_v1(
    payload: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ProbabilityPipelineReadinessError("readiness must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version",
        "result_state",
        "locked_at_utc",
        "clock_attestation",
        "dependencies",
        "runtime_identity",
        "probability_pipeline_contract",
        "locked_empty_state",
        "source_locks",
        "decision_outputs",
        "authority",
        "claim_ceiling",
        "artifact_sha256",
    }:
        raise ProbabilityPipelineReadinessError("readiness structure changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise ProbabilityPipelineReadinessError("readiness hash changed")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("result_state") != RESULT_STATE
    ):
        raise ProbabilityPipelineReadinessError("readiness identity changed")
    locked = _timestamp(value.get("locked_at_utc"), "locked_at_utc")
    if locked >= FUTURE_SEALED_START.replace(tzinfo=timezone.utc):
        raise ProbabilityPipelineReadinessError(
            "readiness was not frozen pre-boundary"
        )
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_inside_builder",
        "observed_wall_clock_utc": locked.isoformat(),
        "user_supplied_timestamp_allowed": False,
        "lock_time_not_after_builder_observation": True,
    }:
        raise ProbabilityPipelineReadinessError("readiness clock changed")
    if value.get("dependencies") != _dependencies(root):
        raise ProbabilityPipelineReadinessError("readiness dependencies changed")
    if value.get("runtime_identity") != _runtime_identity():
        raise ProbabilityPipelineReadinessError("readiness runtime changed")
    if value.get("probability_pipeline_contract") != _contract():
        raise ProbabilityPipelineReadinessError(
            "probability-pipeline contract changed"
        )
    expected_empty = {
        "phase_one_outcome_cohorts": 0,
        "phase_one_evaluation_outputs": 0,
        "recalibration_artifacts": 0,
        "event_uncertainty_artifacts": 0,
        "phase_one_evaluation_registry_present": False,
        "recalibration_uncertainty_registry_present": False,
        "phase_two_opening_present": False,
        "phase_two_started": False,
        "outcomes_accessed": False,
    }
    if value.get("locked_empty_state") != expected_empty:
        raise ProbabilityPipelineReadinessError("locked empty state changed")
    if value.get("source_locks") != _source_locks(root):
        raise ProbabilityPipelineReadinessError("source lock changed")
    if value.get("decision_outputs") != {
        "phase_one_model_pass": None,
        "recalibration_parameters": None,
        "uncertainty_interval": None,
        "phase_two_opening_authority": None,
        "match_probability": None,
        "fair_odds": None,
        "expected_value": None,
        "bet_recommendation": None,
    }:
        raise ProbabilityPipelineReadinessError("decision outputs changed")
    if value.get("authority") != AUTHORITY:
        raise ProbabilityPipelineReadinessError("readiness exceeds authority")
    if value.get("claim_ceiling") != CLAIM_CEILING:
        raise ProbabilityPipelineReadinessError("claim ceiling changed")
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ProbabilityPipelineReadinessError(
            f"refusing to replace readiness: {path}"
        )
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ProbabilityPipelineReadinessError(
                f"refusing to replace readiness: {path}"
            ) from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "DEFAULT_OUTPUT",
    "RESULT_STATE",
    "SCHEMA_VERSION",
    "SOURCE_LOCATOR",
    "ProbabilityPipelineReadinessError",
    "build_probability_pipeline_readiness_v1",
    "validate_probability_pipeline_readiness_v1",
    "write_no_clobber",
]
