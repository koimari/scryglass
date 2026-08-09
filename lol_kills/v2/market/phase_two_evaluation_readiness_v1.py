"""Freeze the complete phase-two evaluation path before collection or outcomes."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

from . import match_winner_future_protocol_v1 as protocol_source
from . import phase_two_evaluation_result_registry_v1 as result_registry
from . import phase_two_evaluation_v1 as evaluation
from . import phase_two_outcome_opening_v1 as opening
from . import phase_two_stopping_snapshot_registry_v1 as snapshot_registry
from . import phase_two_stopping_snapshot_v1 as snapshot
from .match_winner_future_protocol_registry_v1 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_LOCATOR,
    REGISTERED_PROTOCOL_RAW_SHA256,
    validate_registered_match_winner_future_protocol_v1,
)


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/market/phase_two_evaluation_readiness_v1.py"
SCHEMA_VERSION = "scryglass:phase-two-evaluation-readiness:v1"
RESULT_STATE = "PHASE_TWO_EVALUATION_IMPLEMENTATION_FROZEN_OUTCOMES_UNOPENED"
DEFAULT_OUTPUT = Path(
    "data/lol/v2/evaluation/match-winner-market-v1/phase-two/"
    "evaluation-readiness-v1.json"
)
SOURCE_LOCKS = (
    SOURCE_LOCATOR,
    evaluation.SOURCE_LOCATOR,
    opening.SOURCE_LOCATOR,
    result_registry.SOURCE_LOCATOR,
    snapshot.SOURCE_LOCATOR,
    snapshot_registry.SOURCE_LOCATOR,
    protocol_source.SOURCE_LOCATOR,
    "lol_kills/v2/market/phase_two_evaluation_readiness_registry_v1.py",
    "lol_kills/v2/market/semantic_market_authority_v1.py",
    "lol_kills/v2/market/production_event_probability_v1.py",
    "lol_kills/v2/market/production_betano_quote_v1.py",
    "lol_kills/v2/market/semantic_match_winner_decision_v1.py",
)
AUTHORITY = {
    "phase_two_outcome_opening_authority": False,
    "phase_two_evaluation_authority": False,
    "phase_two_evaluation_identity_authority": False,
    "probability_authority": False,
    "fair_odds_authority": False,
    "expected_value_authority": False,
    "recommendation_authority": False,
    "transaction_authority": False,
    "stake_authority": False,
    "betting_authority": False,
}
CLAIM_CEILING = (
    "Outcome-free implementation freeze for the phase-two evaluator, one-time "
    "opening, and independent result registry. It grants no opening, result, "
    "probability, odds, EV, stake, transaction, or betting authority."
)


class PhaseTwoEvaluationReadinessError(RuntimeError):
    """An evaluation dependency, source, contract, or empty state drifted."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value, allow_nan=False, ensure_ascii=False,
            separators=(",", ":"), sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PhaseTwoEvaluationReadinessError(
            "evaluation readiness is not canonical"
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
        raise PhaseTwoEvaluationReadinessError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise PhaseTwoEvaluationReadinessError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _source_record(root: Path, locator: str) -> dict[str, Any]:
    path = root / locator
    if path.is_symlink() or not path.is_file():
        raise PhaseTwoEvaluationReadinessError(
            f"evaluation source is unavailable: {locator}"
        )
    return {
        "locator": locator,
        "bytes": path.stat().st_size,
        "raw_sha256": _sha256_path(path),
    }


def _dependencies(
    root: Path, environment: Mapping[str, str] = os.environ
) -> dict[str, Any]:
    protocol = validate_registered_match_winner_future_protocol_v1(root=root)
    try:
        from .phase_two_collection_readiness_registry_v1 import (
            EXTERNAL_SHA256_ENV as COLLECTION_EXTERNAL_SHA256_ENV,
            REGISTRY_LOCATOR as COLLECTION_REGISTRY_LOCATOR,
            expected_readiness_binding as expected_collection_binding,
            load_pinned_phase_two_collection_readiness_registry_v1,
        )
        collection_digest = environment.get(COLLECTION_EXTERNAL_SHA256_ENV)
        if not collection_digest:
            raise PhaseTwoEvaluationReadinessError(
                "phase-two collection readiness registry pin is missing"
            )
        collection_binding = expected_collection_binding(
            root=root, environment=environment
        )
        collection = load_pinned_phase_two_collection_readiness_registry_v1(
            path=root / COLLECTION_REGISTRY_LOCATOR,
            external_sha256=collection_digest,
            expected_binding=collection_binding,
        )
    except Exception as exc:
        raise PhaseTwoEvaluationReadinessError(
            "registered phase-two collection readiness is unavailable"
        ) from exc
    return {
        "phase_two_collection_readiness": {
            "registry_locator": COLLECTION_REGISTRY_LOCATOR.as_posix(),
            "registry_raw_sha256": collection["receipt_raw_sha256"],
            "registry_id": collection["receipt"]["registry_id"],
            "readiness_binding": collection_binding,
        },
        "match_winner_future_protocol": {
            "locator": REGISTERED_PROTOCOL_LOCATOR.as_posix(),
            "raw_sha256": REGISTERED_PROTOCOL_RAW_SHA256,
            "artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
            "result_state": protocol["result_state"],
        },
    }


def _contract(root: Path = ROOT) -> dict[str, Any]:
    signatures = {
        "validate_outcomes": list(
            inspect.signature(evaluation.validate_outcome_cohort_v1).parameters
        ),
        "evaluate": list(
            inspect.signature(evaluation.evaluate_phase_two_v1).parameters
        ),
        "expected_opening_bindings": list(
            inspect.signature(opening.current_expected_bindings).parameters
        ),
        "run_authorized_evaluation": list(
            inspect.signature(
                opening.run_authorized_phase_two_evaluation_v1
            ).parameters
        ),
        "expected_result_binding": list(
            inspect.signature(result_registry.expected_result_binding).parameters
        ),
    }
    expected = {
        "validate_outcomes": ["payload", "snapshot", "root"],
        "evaluate": [
            "snapshot_locator", "outcome_cohort_raw", "outcome_cohort_locator",
            "opening_authority_binding", "run_id", "root", "environment",
            "clock",
        ],
        "expected_opening_bindings": ["snapshot_locator", "root", "environment"],
        "run_authorized_evaluation": [
            "snapshot_locator", "root", "environment", "clock"
        ],
        "expected_result_binding": [
            "result_locator", "root", "environment"
        ],
    }
    if signatures != expected:
        raise PhaseTwoEvaluationReadinessError(
            "phase-two evaluation signatures changed"
        )
    protocol = validate_registered_match_winner_future_protocol_v1(root=root)
    return {
        "schemas": {
            "sealed_outcome_cohort": evaluation.OUTCOME_SCHEMA_VERSION,
            "evaluation_result": evaluation.RESULT_SCHEMA_VERSION,
            "opening_authority": opening.SCHEMA_VERSION,
            "opening_marker": opening.MARKER_SCHEMA_VERSION,
            "independent_evaluation_registry": result_registry.SCHEMA_VERSION,
        },
        "builder_parameters": signatures,
        "evaluation_contract": protocol["evaluation"],
        "bootstrap": {
            "method": "paired_series_cluster_bootstrap",
            "replicates": evaluation.BOOTSTRAP_REPLICATES,
            "seed": evaluation.BOOTSTRAP_SEED,
            "confidence_interval": list(evaluation.CONFIDENCE_INTERVAL),
            "whole_series_resampling": True,
            "map_weighted_point_metrics": True,
        },
        "outcome_contract": {
            "exact_registered_snapshot_coverage_required": True,
            "winning_side_explicit_blue_or_red": True,
            "authoritative_map_start_identity_required": True,
            "source_observation_strictly_after_map_start": True,
            "outcome_evidence_bytes_and_sha256_checked": True,
            "manual_post_outcome_exclusion_permitted": False,
        },
        "capture_contract": {
            "coverage_denominator_includes_every_completed_prospective_plan": True,
            "prediction_to_response_p95_uses_every_received_quote": True,
            "quote_after_map_start_count_is_exact_not_five_second_proxy": True,
            "quote_within_five_seconds_before_start_reported_separately": True,
            "extraction_and_binding_mismatches_are_terminal_gate_failures": True,
        },
        "opening_contract": {
            "externally_pinned_first_support_met_snapshot_required": True,
            "two_distinct_scope_complete_independent_reviews_required": True,
            "authority_raw_sha256_pinned_out_of_band": True,
            "marker_written_before_first_outcome_read": True,
            "any_marker_or_result_blocks_second_opening": True,
            "atomic_no_clobber_result": True,
            "crash_after_marker_does_not_authorize_reopening": True,
        },
        "result_registration_contract": {
            "exact_full_evaluation_replay_required": True,
            "two_distinct_model_and_market_result_reviews_required": True,
            "external_registry_pin_required": True,
            "registered_pass_itself_is_not_probability_or_betting_authority": True,
            "registered_failure_is_terminal": True,
        },
        "selection_or_tuning_after_opening_permitted": False,
        "all_predeclared_gates_required": True,
    }


def _count_json(root: Path, prefix: object) -> int:
    path = root / Path(str(prefix))
    if not path.exists():
        return 0
    if path.is_symlink() or not path.is_dir():
        raise PhaseTwoEvaluationReadinessError(
            "phase-two evaluation directory is aliased"
        )
    return sum(1 for item in path.rglob("*.json") if item.is_file())


def _empty_state(root: Path) -> dict[str, Any]:
    state = {
        "outcome_cohorts": _count_json(root, evaluation.OUTCOME_PREFIX),
        "outcome_evidence": _count_json(root, evaluation.OUTCOME_EVIDENCE_PREFIX),
        "opening_markers": _count_json(root, opening.OPENING_MARKER_PREFIX),
        "evaluation_results": _count_json(root, evaluation.RESULT_PREFIX),
        "outcome_opening_authority_present": (
            root / opening.AUTHORITY_LOCATOR
        ).is_file(),
        "evaluation_registry_present": (
            root / result_registry.REGISTRY_LOCATOR
        ).is_file(),
        "outcomes_accessed": False,
    }
    if (
        any(
            state[key] != 0
            for key in (
                "outcome_cohorts", "outcome_evidence", "opening_markers",
                "evaluation_results",
            )
        )
        or state["outcome_opening_authority_present"] is not False
        or state["evaluation_registry_present"] is not False
    ):
        raise PhaseTwoEvaluationReadinessError(
            "evaluation readiness must be frozen before outcome artifacts exist"
        )
    return state


def build_phase_two_evaluation_readiness_v1(
    *, root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    locked = clock()
    if not isinstance(locked, datetime) or locked.tzinfo is None:
        raise PhaseTwoEvaluationReadinessError(
            "evaluation readiness clock must be timezone-aware"
        )
    locked = locked.astimezone(timezone.utc)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "locked_at_utc": locked.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_inside_builder",
            "observed_wall_clock_utc": locked.isoformat(),
            "user_supplied_timestamp_allowed": False,
            "outcomes_absent_and_unaccessed_at_lock": True,
        },
        "dependencies": _dependencies(root, environment),
        "evaluation_contract": _contract(root),
        "locked_empty_state": _empty_state(root),
        "source_locks": [_source_record(root, item) for item in SOURCE_LOCKS],
        "decision_outputs": {
            "phase_two_outcome_opening_authority": None,
            "phase_two_evaluation_result": None,
            "phase_two_evaluation_registration": None,
            "match_probability": None,
            "fair_odds": None,
            "expected_value": None,
            "stake": None,
            "bet_recommendation": None,
        },
        "authority": dict(AUTHORITY),
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_phase_two_evaluation_readiness_v1(
        payload, root=root, environment=environment
    )


def validate_phase_two_evaluation_readiness_v1(
    payload: Mapping[str, Any], *, root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PhaseTwoEvaluationReadinessError("readiness must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version", "result_state", "locked_at_utc", "clock_attestation",
        "dependencies", "evaluation_contract", "locked_empty_state",
        "source_locks", "decision_outputs", "authority", "claim_ceiling",
        "artifact_sha256",
    }:
        raise PhaseTwoEvaluationReadinessError("readiness structure changed")
    if value.get("schema_version") != SCHEMA_VERSION or value.get("result_state") != RESULT_STATE:
        raise PhaseTwoEvaluationReadinessError("readiness identity changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise PhaseTwoEvaluationReadinessError("readiness hash changed")
    locked = _timestamp(value.get("locked_at_utc"), "locked_at_utc")
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_inside_builder",
        "observed_wall_clock_utc": locked.isoformat(),
        "user_supplied_timestamp_allowed": False,
        "outcomes_absent_and_unaccessed_at_lock": True,
    }:
        raise PhaseTwoEvaluationReadinessError("readiness clock changed")
    if value.get("dependencies") != _dependencies(root, environment):
        raise PhaseTwoEvaluationReadinessError("readiness dependencies changed")
    if value.get("evaluation_contract") != _contract(root):
        raise PhaseTwoEvaluationReadinessError("evaluation contract changed")
    if value.get("locked_empty_state") != {
        "outcome_cohorts": 0,
        "outcome_evidence": 0,
        "opening_markers": 0,
        "evaluation_results": 0,
        "outcome_opening_authority_present": False,
        "evaluation_registry_present": False,
        "outcomes_accessed": False,
    }:
        raise PhaseTwoEvaluationReadinessError("locked empty state changed")
    if value.get("source_locks") != [
        _source_record(root, item) for item in SOURCE_LOCKS
    ]:
        raise PhaseTwoEvaluationReadinessError("evaluation source lock changed")
    if value.get("decision_outputs") != {
        "phase_two_outcome_opening_authority": None,
        "phase_two_evaluation_result": None,
        "phase_two_evaluation_registration": None,
        "match_probability": None,
        "fair_odds": None,
        "expected_value": None,
        "stake": None,
        "bet_recommendation": None,
    }:
        raise PhaseTwoEvaluationReadinessError("decision outputs changed")
    if value.get("authority") != AUTHORITY or value.get("claim_ceiling") != CLAIM_CEILING:
        raise PhaseTwoEvaluationReadinessError("readiness exceeds authority")
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PhaseTwoEvaluationReadinessError(
            f"refusing to replace evaluation readiness: {path}"
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
            raise PhaseTwoEvaluationReadinessError(
                f"refusing to replace evaluation readiness: {path}"
            ) from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "DEFAULT_OUTPUT", "RESULT_STATE", "SCHEMA_VERSION", "SOURCE_LOCATOR",
    "PhaseTwoEvaluationReadinessError",
    "build_phase_two_evaluation_readiness_v1",
    "validate_phase_two_evaluation_readiness_v1", "write_no_clobber",
]
