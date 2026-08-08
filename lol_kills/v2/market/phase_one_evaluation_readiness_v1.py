"""Freeze phase-one evaluation and one-time opening machinery pre-boundary."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

from lol_kills.v2.ratings.player.multileague_v3_future_protocol import (
    FUTURE_SEALED_START,
)

from . import phase_one_evaluation_v1 as evaluation
from . import phase_one_evaluation_registry_v1 as evaluation_registry
from . import phase_one_opening_v1 as opening
from .match_winner_future_protocol_registry_v1 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256 as MARKET_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_LOCATOR as MARKET_PROTOCOL_LOCATOR,
    REGISTERED_PROTOCOL_RAW_SHA256 as MARKET_PROTOCOL_RAW_SHA256,
    validate_registered_match_winner_future_protocol_v1,
)
from .phase_one_collection_readiness_registry_v1 import (
    REGISTERED_READINESS_ARTIFACT_SHA256 as COLLECTION_READINESS_ARTIFACT_SHA256,
    REGISTERED_READINESS_LOCATOR as COLLECTION_READINESS_LOCATOR,
    REGISTERED_READINESS_RAW_SHA256 as COLLECTION_READINESS_RAW_SHA256,
    validate_registered_phase_one_collection_readiness_v1,
)


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/market/phase_one_evaluation_readiness_v1.py"
SCHEMA_VERSION = "scryglass:phase-one-evaluation-readiness:v1"
RESULT_STATE = "ONE_TIME_PHASE_ONE_EVALUATION_IMPLEMENTATION_FROZEN_PRE_BOUNDARY"
DEFAULT_OUTPUT = Path(
    "data/lol/v2/evaluation/match-winner-market-v1/phase-one/"
    "evaluation-readiness-v1.json"
)
SOURCE_LOCKS = (
    SOURCE_LOCATOR,
    evaluation.SOURCE_LOCATOR,
    opening.SOURCE_LOCATOR,
    evaluation_registry.SOURCE_LOCATOR,
    evaluation.TYPESCRIPT_PARITY_LOCATOR,
    evaluation.TYPESCRIPT_SCORER_LOCATOR,
    "lol_kills/v2/market/phase_one_collection_v1.py",
    "lol_kills/v2/ratings/player/multileague_v3_prediction_ledger.py",
    "lol_kills/v2/draft/terminal/future_prediction_ledger.py",
)
CLAIM_CEILING = (
    "Pre-boundary implementation freeze only. No future outcomes, model pass, "
    "rating, probability, odds, expected value, recommendation, transaction, "
    "or betting authority is present."
)


class PhaseOneEvaluationReadinessError(RuntimeError):
    """The evaluation implementation or its pre-boundary freeze drifted."""


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
        raise PhaseOneEvaluationReadinessError("readiness value is not canonical") from exc


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
        raise PhaseOneEvaluationReadinessError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise PhaseOneEvaluationReadinessError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _source_record(root: Path, locator: str) -> dict[str, Any]:
    path = root / locator
    if path.is_symlink() or not path.is_file():
        raise PhaseOneEvaluationReadinessError(f"source is unavailable: {locator}")
    return {
        "locator": locator,
        "bytes": path.stat().st_size,
        "raw_sha256": _sha256_path(path),
    }


def _dependencies(root: Path) -> dict[str, Any]:
    collection = validate_registered_phase_one_collection_readiness_v1(root=root)
    market = validate_registered_match_winner_future_protocol_v1(root=root)
    return {
        "phase_one_collection_readiness": {
            "locator": COLLECTION_READINESS_LOCATOR.as_posix(),
            "raw_sha256": COLLECTION_READINESS_RAW_SHA256,
            "artifact_sha256": COLLECTION_READINESS_ARTIFACT_SHA256,
            "result_state": collection["result_state"],
        },
        "match_winner_future_protocol": {
            "locator": MARKET_PROTOCOL_LOCATOR.as_posix(),
            "raw_sha256": MARKET_PROTOCOL_RAW_SHA256,
            "artifact_sha256": MARKET_PROTOCOL_ARTIFACT_SHA256,
            "result_state": market["result_state"],
        },
    }


def _contract() -> dict[str, Any]:
    signatures = {
        "build_parity": list(
            inspect.signature(evaluation.build_draft_replay_parity_registry).parameters
        ),
        "evaluate": list(inspect.signature(evaluation.evaluate_phase_one).parameters),
        "validate_outcomes": list(
            inspect.signature(evaluation.validate_outcome_cohort).parameters
        ),
        "expected_bindings": list(
            inspect.signature(opening.current_expected_bindings).parameters
        ),
        "run_authorized": list(
            inspect.signature(opening.run_authorized_phase_one_evaluation).parameters
        ),
    }
    expected_signatures = {
        "build_parity": [
            "snapshot_locator",
            "typescript_replay_raw",
            "parity_locator",
            "root",
            "clock",
        ],
        "evaluate": [
            "snapshot_locator",
            "parity_locator",
            "outcome_cohort_raw",
            "outcome_cohort_locator",
            "opening_authority_binding",
            "run_id",
            "root",
            "clock",
        ],
        "validate_outcomes": ["payload", "snapshot", "root"],
        "expected_bindings": ["snapshot_locator", "parity_locator", "root"],
        "run_authorized": [
            "snapshot_locator",
            "parity_locator",
            "root",
            "environment",
            "clock",
        ],
    }
    if signatures != expected_signatures:
        raise PhaseOneEvaluationReadinessError("evaluation signatures changed")
    return {
        "schemas": {
            "parity_registry": evaluation.PARITY_SCHEMA_VERSION,
            "typescript_parity_replay": evaluation.PARITY_REPLAY_SCHEMA_VERSION,
            "sealed_outcome_cohort": evaluation.OUTCOME_SCHEMA_VERSION,
            "evaluation_result": evaluation.RESULT_SCHEMA_VERSION,
            "opening_authority": opening.SCHEMA_VERSION,
            "independent_evaluation_registry": evaluation_registry.SCHEMA_VERSION,
        },
        "builder_parameters": signatures,
        "bootstrap": {
            "ratings_replicates": evaluation.RATINGS_BOOTSTRAP_REPLICATES,
            "ratings_seed": evaluation.RATINGS_BOOTSTRAP_SEED,
            "draft_replicates": evaluation.DRAFT_BOOTSTRAP_REPLICATES,
            "draft_seed": evaluation.DRAFT_BOOTSTRAP_SEED,
            "confidence_interval": list(evaluation.CONFIDENCE_INTERVAL),
            "whole_series_resampling": True,
            "map_weighted_point_metrics": True,
        },
        "entity_network_dependence_sensitivity": {
            "method": (
                "shared_series_participant_or_organization_network_hac_"
                "sandwich_for_mean_paired_loss_delta"
            ),
            "dependency_rule": (
                "maps_may_covary_when_they_share_series_id_any_exact_player_id_"
                "or_any_exact_organization_id"
            ),
            "critical_value": evaluation.ENTITY_NETWORK_HAC_CRITICAL_VALUE,
            "minimum_series": evaluation.ENTITY_NETWORK_HAC_MINIMUM_SERIES,
            "minimum_participants": (
                evaluation.ENTITY_NETWORK_HAC_MINIMUM_PARTICIPANTS
            ),
            "ratings_required_strata": [
                "overall",
                *(f"league:{league}" for league in evaluation.DOMESTIC_LEAGUES),
                "roster_change",
            ],
            "draft_required_strata": ["overall"],
            "both_metric_upper_95_bounds_must_be_nonpositive": True,
            "supplements_not_replaces_whole_series_bootstrap": True,
            "exact_ten_player_and_two_organization_ids_required": True,
            "locked_before_future_outcomes": True,
        },
        "ratings_reliability_gate_repair": {
            "reason": "v3_protocol_required_a_locked_gate_but_omitted_its_exact_definition",
            "locked_before_outcomes": True,
            "calibration_intercept_percentile_interval_must_include_zero": True,
            "calibration_slope_percentile_interval_must_include_one": True,
            "candidate_minus_each_comparator_ece_upper_95_maximum": evaluation.ECE_DELTA_UPPER_MAXIMUM,
            "ece_equal_frequency_bins": evaluation.ECE_BINS,
        },
        "draft_gates": {
            "overall_both_metric_points_nonpositive": True,
            "overall_both_upper_95_bounds_nonpositive": True,
            "at_least_one_overall_upper_95_strictly_negative": True,
            "each_domestic_league_each_patch_and_international_point_nonharm": True,
            "calibration_intercept_interval_includes_zero": True,
            "calibration_slope_interval_includes_one": True,
            "ece_delta_upper_95_maximum": evaluation.ECE_DELTA_UPPER_MAXIMUM,
            "exact_python_typescript_parity_tolerance": evaluation.PARITY_TOLERANCE,
            "entity_network_dependence_sensitivity_must_pass": True,
        },
        "outcome_contract": {
            "exact_snapshot_event_coverage_required": True,
            "outcome_evidence_bytes_hash_checked": True,
            "winning_side_explicit_blue_or_red": True,
            "source_observation_strictly_after_actual_map_start": True,
            "manual_post_outcome_exclusion_permitted": False,
        },
        "opening_contract": {
            "two_distinct_scope_complete_independent_reviews_required": True,
            "authority_raw_sha256_must_be_pinned_out_of_band": True,
            "opening_marker_written_before_first_outcome_read": True,
            "marker_or_output_presence_blocks_second_opening": True,
            "atomic_no_clobber_output": True,
            "crash_after_marker_does_not_authorize_reopening": True,
        },
        "selection_or_tuning_after_opening_permitted": False,
        "phase_one_pass_itself_authorizes_phase_two": False,
        "phase_one_pass_itself_is_probability_or_betting_authority": False,
    }


def _count_json(root: Path, prefix: object) -> int:
    path = root / Path(str(prefix))
    if not path.exists():
        return 0
    if path.is_symlink() or not path.is_dir():
        raise PhaseOneEvaluationReadinessError("evaluation directory is aliased")
    return sum(1 for item in path.rglob("*.json") if item.is_file())


def _empty_state(root: Path) -> dict[str, Any]:
    state = {
        "parity_registries": _count_json(root, evaluation.PARITY_PREFIX),
        "outcome_cohorts": _count_json(root, evaluation.OUTCOME_PREFIX),
        "outcome_evidence": _count_json(root, evaluation.OUTCOME_EVIDENCE_PREFIX),
        "opening_markers": _count_json(root, opening.OPENING_MARKER_PREFIX),
        "evaluation_outputs": _count_json(root, evaluation.OUTPUT_PREFIX),
        "opening_authority_present": (root / opening.AUTHORITY_LOCATOR).is_file(),
        "outcomes_accessed": False,
    }
    if any(state[key] != 0 for key in (
        "parity_registries",
        "outcome_cohorts",
        "outcome_evidence",
        "opening_markers",
        "evaluation_outputs",
    )) or state["opening_authority_present"] is not False:
        raise PhaseOneEvaluationReadinessError(
            "evaluation readiness must be frozen before phase-one artifacts exist"
        )
    return state


def build_phase_one_evaluation_readiness_v1(
    *,
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    locked = clock()
    if not isinstance(locked, datetime) or locked.tzinfo is None:
        raise PhaseOneEvaluationReadinessError("readiness clock must be timezone-aware")
    locked = locked.astimezone(timezone.utc)
    if locked >= FUTURE_SEALED_START.replace(tzinfo=timezone.utc):
        raise PhaseOneEvaluationReadinessError("readiness was not frozen pre-boundary")
    dependencies = _dependencies(root)
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
        "dependencies": dependencies,
        "evaluation_contract": _contract(),
        "locked_empty_state": _empty_state(root),
        "source_locks": [_source_record(root, locator) for locator in SOURCE_LOCKS],
        "decision_outputs": {
            "phase_one_opening_authority": None,
            "ratings_validation_authority": None,
            "draft_validation_authority": None,
            "phase_two_opening_authority": None,
            "match_probability": None,
            "fair_odds": None,
            "expected_value": None,
            "bet_recommendation": None,
        },
        "authority": {name: False for name in evaluation.AUTHORITY_KEYS},
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_phase_one_evaluation_readiness_v1(payload, root=root)


def validate_phase_one_evaluation_readiness_v1(
    payload: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PhaseOneEvaluationReadinessError("readiness must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version",
        "result_state",
        "locked_at_utc",
        "clock_attestation",
        "dependencies",
        "evaluation_contract",
        "locked_empty_state",
        "source_locks",
        "decision_outputs",
        "authority",
        "claim_ceiling",
        "artifact_sha256",
    }:
        raise PhaseOneEvaluationReadinessError("readiness structure changed")
    if value.get("schema_version") != SCHEMA_VERSION or value.get("result_state") != RESULT_STATE:
        raise PhaseOneEvaluationReadinessError("readiness identity changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise PhaseOneEvaluationReadinessError("readiness hash changed")
    locked = _timestamp(value.get("locked_at_utc"), "locked_at_utc")
    if locked >= FUTURE_SEALED_START.replace(tzinfo=timezone.utc):
        raise PhaseOneEvaluationReadinessError("readiness was not frozen pre-boundary")
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_inside_builder",
        "observed_wall_clock_utc": locked.isoformat(),
        "user_supplied_timestamp_allowed": False,
        "lock_time_not_after_builder_observation": True,
    }:
        raise PhaseOneEvaluationReadinessError("readiness clock changed")
    if value.get("dependencies") != _dependencies(root):
        raise PhaseOneEvaluationReadinessError("readiness dependencies changed")
    if value.get("evaluation_contract") != _contract():
        raise PhaseOneEvaluationReadinessError("evaluation contract changed")
    if value.get("locked_empty_state") != {
        "parity_registries": 0,
        "outcome_cohorts": 0,
        "outcome_evidence": 0,
        "opening_markers": 0,
        "evaluation_outputs": 0,
        "opening_authority_present": False,
        "outcomes_accessed": False,
    }:
        raise PhaseOneEvaluationReadinessError("locked empty state changed")
    if value.get("source_locks") != [_source_record(root, locator) for locator in SOURCE_LOCKS]:
        raise PhaseOneEvaluationReadinessError("evaluation source lock changed")
    if value.get("decision_outputs") != {
        "phase_one_opening_authority": None,
        "ratings_validation_authority": None,
        "draft_validation_authority": None,
        "phase_two_opening_authority": None,
        "match_probability": None,
        "fair_odds": None,
        "expected_value": None,
        "bet_recommendation": None,
    }:
        raise PhaseOneEvaluationReadinessError("readiness decision outputs changed")
    authority = value.get("authority") or {}
    if set(authority) != set(evaluation.AUTHORITY_KEYS) or any(authority.values()):
        raise PhaseOneEvaluationReadinessError("readiness exceeds authority")
    if value.get("claim_ceiling") != CLAIM_CEILING:
        raise PhaseOneEvaluationReadinessError("readiness claim ceiling changed")
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PhaseOneEvaluationReadinessError(f"refusing to replace readiness: {path}")
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
            raise PhaseOneEvaluationReadinessError(
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
    "PhaseOneEvaluationReadinessError",
    "build_phase_one_evaluation_readiness_v1",
    "validate_phase_one_evaluation_readiness_v1",
    "write_no_clobber",
]
