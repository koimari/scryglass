"""Fit the frozen phase-one recalibration only after an independent model pass."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from . import phase_one_evaluation_v1 as evaluation
from . import phase_one_evaluation_registry_v1 as evaluation_registry
from .match_winner_future_protocol_registry_v1 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_LOCATOR,
    REGISTERED_PROTOCOL_RAW_SHA256,
    validate_registered_match_winner_future_protocol_v1,
)


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/market/phase_one_recalibration_v1.py"
SCHEMA_VERSION = "scryglass:phase-one-bounded-recalibration:v1"
RESULT_STATE = "PHASE_ONE_RECALIBRATION_FIT_NON_AUTHORIZING"
OUTPUT_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/match-winner-market-v1/recalibration"
)
DEFAULT_OUTPUT = Path(OUTPUT_PREFIX / "phase-one-recalibration-v1.json")
RAW_PROBABILITY_CLIP = (1e-6, 0.999999)
INTERCEPT_BOUNDS = (-2.0, 2.0)
SLOPE_BOUNDS = (0.25, 4.0)
INITIAL_PARAMETERS = (0.0, 1.0)
MAXIMUM_ITERATIONS = 10_000
OPTIMIZER_FTOL = 1e-12
OPTIMIZER_GTOL = 1e-8
AUTHORITY = {
    "calibration_identity_authority": False,
    "phase_two_opening_authority": False,
    "probability_authority": False,
    "odds_authority": False,
    "expected_value_authority": False,
    "recommendation_authority": False,
    "betting_authority": False,
}
CLAIM_CEILING = (
    "Deterministic phase-one bounded recalibration candidate only. Independent "
    "calibration and uncertainty registration, phase-two opening, event-specific "
    "probability registration, quote, settlement, and market authority remain required."
)


class PhaseOneRecalibrationError(RuntimeError):
    """The phase-one pass, inputs, optimizer, or recalibration artifact failed closed."""


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
        raise PhaseOneRecalibrationError("recalibration value is not canonical") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _clock_sample(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise PhaseOneRecalibrationError("recalibration clock must be timezone-aware")
    return value.astimezone(timezone.utc)


def _optimization_contract() -> dict[str, Any]:
    return {
        "method": "bounded_logistic_recalibration",
        "formula": "sigmoid(intercept+slope*logit(clipped_raw_probability))",
        "raw_probability_clip": list(RAW_PROBABILITY_CLIP),
        "objective": "unweighted_map_log_loss",
        "intercept_bounds": list(INTERCEPT_BOUNDS),
        "slope_bounds": list(SLOPE_BOUNDS),
        "initial_parameters": {
            "intercept": INITIAL_PARAMETERS[0],
            "slope": INITIAL_PARAMETERS[1],
        },
        "optimizer": "scipy.optimize.minimize:L-BFGS-B",
        "optimizer_ftol": OPTIMIZER_FTOL,
        "optimizer_gtol": OPTIMIZER_GTOL,
        "maximum_iterations": MAXIMUM_ITERATIONS,
        "rating_only_comparator_recalibrated_by_identical_procedure": True,
        "phase_two_refit_or_online_update_permitted": False,
    }


def _validate_protocol(root: Path) -> dict[str, Any]:
    try:
        protocol = validate_registered_match_winner_future_protocol_v1(root=root)
    except Exception as exc:
        raise PhaseOneRecalibrationError("market protocol is invalid") from exc
    if protocol.get("recalibration") != {
        "status": "NOT_YET_FIT",
        "fit_only_after_phase_one_passes": True,
        "fit_dataset": "complete_independently_opened_phase_one_draft_cohort",
        "raw_probability": "frozen_rating_plus_terminal_draft_contextual_probability",
        **_optimization_contract(),
        "parameters_and_implementation_sha256_frozen_before_phase_two": True,
    }:
        raise PhaseOneRecalibrationError("registered recalibration protocol changed")
    return protocol


def _logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(
        probabilities, RAW_PROBABILITY_CLIP[0], RAW_PROBABILITY_CLIP[1]
    )
    return np.log(clipped / (1.0 - clipped))


def _objective_and_gradient(
    parameters: np.ndarray,
    *,
    raw_logits: np.ndarray,
    outcomes: np.ndarray,
) -> tuple[float, np.ndarray]:
    intercept = float(parameters[0])
    slope = float(parameters[1])
    linear = np.clip(intercept + slope * raw_logits, -40.0, 40.0)
    probabilities = 1.0 / (1.0 + np.exp(-linear))
    probabilities = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
    loss = float(
        -np.mean(
            outcomes * np.log(probabilities)
            + (1.0 - outcomes) * np.log1p(-probabilities)
        )
    )
    residual = probabilities - outcomes
    gradient = np.asarray(
        [float(np.mean(residual)), float(np.mean(residual * raw_logits))],
        dtype=float,
    )
    if not math.isfinite(loss) or not np.all(np.isfinite(gradient)):
        raise PhaseOneRecalibrationError("recalibration objective is non-finite")
    return loss, gradient


def fit_bounded_recalibration(
    probabilities: Sequence[float], outcomes: Sequence[int]
) -> dict[str, Any]:
    """Fit the exact preregistered transform with no configurable thresholds."""

    if len(probabilities) != len(outcomes) or not probabilities:
        raise PhaseOneRecalibrationError("recalibration inputs are empty or misaligned")
    raw = np.asarray(probabilities, dtype=float)
    labels = np.asarray(outcomes, dtype=float)
    if (
        not np.all(np.isfinite(raw))
        or np.any(raw <= 0.0)
        or np.any(raw >= 1.0)
        or not set(labels.tolist()).issubset({0.0, 1.0})
        or np.unique(labels).size < 2
    ):
        raise PhaseOneRecalibrationError("recalibration probability or outcome domain changed")
    raw_logits = _logits(raw)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        return _objective_and_gradient(
            parameters, raw_logits=raw_logits, outcomes=labels
        )

    result = minimize(
        objective,
        np.asarray(INITIAL_PARAMETERS, dtype=float),
        method="L-BFGS-B",
        jac=True,
        bounds=(INTERCEPT_BOUNDS, SLOPE_BOUNDS),
        options={
            "maxiter": MAXIMUM_ITERATIONS,
            "ftol": OPTIMIZER_FTOL,
            "gtol": OPTIMIZER_GTOL,
        },
    )
    parameters = np.asarray(result.x, dtype=float)
    final_loss, final_gradient = objective(parameters)
    if (
        result.success is not True
        or result.status != 0
        or parameters.shape != (2,)
        or not np.all(np.isfinite(parameters))
        or not INTERCEPT_BOUNDS[0] <= parameters[0] <= INTERCEPT_BOUNDS[1]
        or not SLOPE_BOUNDS[0] <= parameters[1] <= SLOPE_BOUNDS[1]
        or not math.isfinite(final_loss)
    ):
        raise PhaseOneRecalibrationError(
            f"bounded recalibration did not converge: {result.message}"
        )
    identity_loss, _ = objective(np.asarray(INITIAL_PARAMETERS, dtype=float))
    return {
        "intercept": float(parameters[0]),
        "slope": float(parameters[1]),
        "map_log_loss": final_loss,
        "identity_map_log_loss": identity_loss,
        "map_log_loss_delta_vs_identity": final_loss - identity_loss,
        "convergence": {
            "success": True,
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "final_gradient_max_abs": float(np.max(np.abs(final_gradient))),
        },
    }


def _registered_pass(
    *, result_locator: str, root: Path, environment: Mapping[str, str]
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    digest = environment.get(evaluation_registry.EXTERNAL_SHA256_ENV)
    if not digest:
        raise PhaseOneRecalibrationError(
            "independent phase-one evaluation registry digest is missing"
        )
    try:
        binding = evaluation_registry.expected_result_binding(
            result_locator=result_locator, root=root
        )
        registered = evaluation_registry.load_pinned_evaluation_registry(
            path=root / evaluation_registry.REGISTRY_LOCATOR,
            external_sha256=digest,
            expected_binding=binding,
        )
    except evaluation_registry.PhaseOneEvaluationRegistryError as exc:
        raise PhaseOneRecalibrationError(
            "independent phase-one evaluation registry is invalid"
        ) from exc
    if registered.get("phase_one_models_independently_passed") is not True:
        raise PhaseOneRecalibrationError(
            "phase-one models did not independently pass"
        )
    result_raw = evaluation._read_regular(root, result_locator, "phase-one result")
    result = evaluation.validate_phase_one_evaluation_result(
        evaluation._strict_object(result_raw, "phase-one result")
    )
    return registered, result, result_raw


def _phase_one_rows(
    *, result: Mapping[str, Any], root: Path
) -> tuple[list[dict[str, Any]], bytes, Mapping[str, Any], bytes, Mapping[str, Any]]:
    inputs = result["inputs"]
    snapshot_raw, snapshot = evaluation._snapshot(
        root, inputs["snapshot_locator"]
    )
    if (
        _sha256_bytes(snapshot_raw) != inputs["snapshot_raw_sha256"]
        or snapshot["artifact_sha256"] != inputs["snapshot_artifact_sha256"]
    ):
        raise PhaseOneRecalibrationError("phase-one snapshot binding changed")
    outcome_raw = evaluation._read_regular(
        root, inputs["outcome_cohort_locator"], "phase-one outcome cohort"
    )
    if _sha256_bytes(outcome_raw) != inputs["outcome_cohort_raw_sha256"]:
        raise PhaseOneRecalibrationError("phase-one outcome cohort hash changed")
    outcomes = evaluation.validate_outcome_cohort(
        evaluation._strict_object(outcome_raw, "phase-one outcome cohort"),
        snapshot=snapshot,
        root=root,
    )
    if outcomes["artifact_sha256"] != inputs["outcome_cohort_artifact_sha256"]:
        raise PhaseOneRecalibrationError("phase-one outcome artifact changed")
    rows = evaluation._evaluation_rows(
        snapshot=snapshot, outcomes=outcomes, root=root
    )
    return rows, snapshot_raw, snapshot, outcome_raw, outcomes


def build_phase_one_recalibration(
    *,
    phase_one_result_locator: str,
    root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    fitted_at = _clock_sample(clock)
    protocol = _validate_protocol(root)
    registered, result, result_raw = _registered_pass(
        result_locator=phase_one_result_locator,
        root=root,
        environment=environment,
    )
    rows, snapshot_raw, snapshot, outcome_raw, outcomes = _phase_one_rows(
        result=result, root=root
    )
    labels = [int(row["blue_win"]) for row in rows]
    combined = fit_bounded_recalibration(
        [float(row["ratings_plus_draft"]) for row in rows], labels
    )
    rating_only = fit_bounded_recalibration(
        [float(row["ratings_only"]) for row in rows], labels
    )
    registry_receipt = registered["receipt"]
    registry_raw = (root / evaluation_registry.REGISTRY_LOCATOR).read_bytes()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "fitted_at_utc": fitted_at.isoformat(),
        "inputs": {
            "phase_one_result_locator": phase_one_result_locator,
            "phase_one_result_raw_sha256": _sha256_bytes(result_raw),
            "phase_one_result_artifact_sha256": result["artifact_sha256"],
            "phase_one_evaluation_registry_locator": evaluation_registry.REGISTRY_LOCATOR.as_posix(),
            "phase_one_evaluation_registry_raw_sha256": _sha256_bytes(registry_raw),
            "phase_one_evaluation_registry_id": registry_receipt["registry_id"],
            "joint_snapshot_locator": result["inputs"]["snapshot_locator"],
            "joint_snapshot_raw_sha256": _sha256_bytes(snapshot_raw),
            "joint_snapshot_artifact_sha256": snapshot["artifact_sha256"],
            "outcome_cohort_locator": result["inputs"]["outcome_cohort_locator"],
            "outcome_cohort_raw_sha256": _sha256_bytes(outcome_raw),
            "outcome_cohort_artifact_sha256": outcomes["artifact_sha256"],
            "maps": len(rows),
            "series": len({row["series_id"] for row in rows}),
        },
        "protocol": {
            "locator": REGISTERED_PROTOCOL_LOCATOR.as_posix(),
            "raw_sha256": REGISTERED_PROTOCOL_RAW_SHA256,
            "artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
            "recalibration_contract": protocol["recalibration"],
        },
        "optimization_contract": _optimization_contract(),
        "models": {
            "ratings_plus_draft": {
                "raw_probability_field": "ratings_plus_draft.p_blue",
                **combined,
            },
            "ratings_only": {
                "raw_probability_field": "ratings_only.p_blue",
                **rating_only,
            },
        },
        "qualification": {
            "phase_one_evaluation_independently_registered": True,
            "phase_one_models_independently_passed": True,
            "complete_phase_one_draft_cohort_used": True,
            "unweighted_map_objective_used": True,
            "candidate_or_hyperparameter_reselection_performed": False,
            "market_price_used": False,
            "phase_two_event_outcome_used": False,
            "phase_two_started": False,
            "independently_registered": False,
        },
        "authority": dict(AUTHORITY),
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_phase_one_recalibration_artifact(payload)


def validate_phase_one_recalibration_artifact(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PhaseOneRecalibrationError("recalibration artifact must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version",
        "result_state",
        "fitted_at_utc",
        "inputs",
        "protocol",
        "optimization_contract",
        "models",
        "qualification",
        "authority",
        "claim_ceiling",
        "artifact_sha256",
    }:
        raise PhaseOneRecalibrationError("recalibration artifact structure changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise PhaseOneRecalibrationError("recalibration artifact hash changed")
    if value.get("schema_version") != SCHEMA_VERSION or value.get("result_state") != RESULT_STATE:
        raise PhaseOneRecalibrationError("recalibration artifact identity changed")
    evaluation._timestamp(value.get("fitted_at_utc"), "fitted_at_utc")
    inputs = value.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "phase_one_result_locator",
        "phase_one_result_raw_sha256",
        "phase_one_result_artifact_sha256",
        "phase_one_evaluation_registry_locator",
        "phase_one_evaluation_registry_raw_sha256",
        "phase_one_evaluation_registry_id",
        "joint_snapshot_locator",
        "joint_snapshot_raw_sha256",
        "joint_snapshot_artifact_sha256",
        "outcome_cohort_locator",
        "outcome_cohort_raw_sha256",
        "outcome_cohort_artifact_sha256",
        "maps",
        "series",
    }:
        raise PhaseOneRecalibrationError("recalibration inputs changed")
    for key, item in inputs.items():
        if key.endswith("sha256"):
            evaluation._sha(item, f"inputs.{key}")
    if inputs.get("phase_one_evaluation_registry_locator") != evaluation_registry.REGISTRY_LOCATOR.as_posix():
        raise PhaseOneRecalibrationError("evaluation registry locator changed")
    for key in ("maps", "series"):
        item = inputs.get(key)
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise PhaseOneRecalibrationError("recalibration sample size changed")
    protocol = value.get("protocol")
    if not isinstance(protocol, Mapping) or set(protocol) != {
        "locator",
        "raw_sha256",
        "artifact_sha256",
        "recalibration_contract",
    } or protocol.get("locator") != REGISTERED_PROTOCOL_LOCATOR.as_posix() or protocol.get("raw_sha256") != REGISTERED_PROTOCOL_RAW_SHA256 or protocol.get("artifact_sha256") != REGISTERED_PROTOCOL_ARTIFACT_SHA256:
        raise PhaseOneRecalibrationError("recalibration protocol binding changed")
    if value.get("optimization_contract") != _optimization_contract():
        raise PhaseOneRecalibrationError("optimization contract changed")
    models = value.get("models")
    if not isinstance(models, Mapping) or set(models) != {"ratings_plus_draft", "ratings_only"}:
        raise PhaseOneRecalibrationError("recalibration model inventory changed")
    expected_fields = {
        "raw_probability_field",
        "intercept",
        "slope",
        "map_log_loss",
        "identity_map_log_loss",
        "map_log_loss_delta_vs_identity",
        "convergence",
    }
    for name, report in models.items():
        if not isinstance(report, Mapping) or set(report) != expected_fields:
            raise PhaseOneRecalibrationError("recalibration model report changed")
        intercept = evaluation._number(report.get("intercept"), f"{name}.intercept")
        slope = evaluation._number(report.get("slope"), f"{name}.slope")
        loss = evaluation._number(report.get("map_log_loss"), f"{name}.loss")
        identity = evaluation._number(
            report.get("identity_map_log_loss"), f"{name}.identity_loss"
        )
        delta = evaluation._number(
            report.get("map_log_loss_delta_vs_identity"), f"{name}.delta"
        )
        if (
            not INTERCEPT_BOUNDS[0] <= intercept <= INTERCEPT_BOUNDS[1]
            or not SLOPE_BOUNDS[0] <= slope <= SLOPE_BOUNDS[1]
            or loss < 0.0
            or identity < 0.0
            or not math.isclose(delta, loss - identity, abs_tol=1e-15)
            or not isinstance(report.get("convergence"), Mapping)
            or report["convergence"].get("success") is not True
            or report["convergence"].get("status") != 0
        ):
            raise PhaseOneRecalibrationError("recalibration model result is invalid")
    if value.get("qualification") != {
        "phase_one_evaluation_independently_registered": True,
        "phase_one_models_independently_passed": True,
        "complete_phase_one_draft_cohort_used": True,
        "unweighted_map_objective_used": True,
        "candidate_or_hyperparameter_reselection_performed": False,
        "market_price_used": False,
        "phase_two_event_outcome_used": False,
        "phase_two_started": False,
        "independently_registered": False,
    }:
        raise PhaseOneRecalibrationError("recalibration qualification changed")
    if value.get("authority") != AUTHORITY or value.get("claim_ceiling") != CLAIM_CEILING:
        raise PhaseOneRecalibrationError("recalibration artifact exceeds authority")
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PhaseOneRecalibrationError(f"refusing to replace recalibration: {path}")
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
            raise PhaseOneRecalibrationError(
                f"refusing to replace recalibration: {path}"
            ) from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return _sha256_bytes(raw)


__all__ = [
    "DEFAULT_OUTPUT",
    "SCHEMA_VERSION",
    "SOURCE_LOCATOR",
    "PhaseOneRecalibrationError",
    "build_phase_one_recalibration",
    "fit_bounded_recalibration",
    "validate_phase_one_recalibration_artifact",
    "write_no_clobber",
]
