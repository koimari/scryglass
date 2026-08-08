"""Narrow private Player Rating development on the accepted G2 adapter input.

The runner deliberately consumes only ``load_accepted_lpl_private_player_rating_input``.
It neither opens G1 files nor infers/reorders origin membership.  This is a
private, rolling/prequential development experiment, not a production rating.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Callable, Mapping

from .model import DISPLAY_LOGIT_SCALE, posterior_predictive_expected_result
from .real_v1_adapter import (
    ACCEPTED_G1_PINS,
    CLAIM_CEILING as ADAPTER_CLAIM_CEILING,
    MapObservation,
    PrivatePlayerRatingInput,
    load_accepted_lpl_private_player_rating_input,
)


# ``state.mean`` is the natural-logit latent used by the predictive model.
# The public Elo-compatible display therefore uses 400 / ln(10), not 400
# points per raw latent unit.  Keep the conversion shared with the synthetic
# Player Rating mechanics so the private handoff cannot silently change the
# meaning of a displayed rating.
DISPLAY_ANCHOR = 1500.0
DISPLAY_SCALE = DISPLAY_LOGIT_SCALE
# Frozen v2 display conversion retained only for replaying the historical
# descriptive last-observed Team table.  It is not used by the v3 runner.
LEGACY_DISPLAY_SCALE_V2 = 400.0
SCHEMA_VERSION = "scryglass:player-real-v1-private-development:v3"
# The v2 artifact remains a frozen input to the descriptive last-observed Team
# table.  It is not the current Player development output, so compatibility is
# explicit instead of silently accepting arbitrary historical schemas.
LEGACY_SCHEMA_VERSION_V2 = "scryglass:player-real-v1-private-development:v2"
FROZEN_PROCESS_VARIANCE_PER_DAY = 0.0005
FROZEN_MEAN_REVERSION_HALF_LIFE_DAYS = 120.0


class PrivateDevelopmentError(ValueError):
    """A private rating development calculation failed closed."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PrivateDevelopmentError("canonical artifact contains a non-finite value") from error


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _finite(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise PrivateDevelopmentError(f"{name} must be finite")
    return float(value)


def _plugin_sigmoid(value: float) -> float:
    """Stable plug-in sigmoid for the ADF score/curvature only."""

    if not math.isfinite(value):
        raise PrivateDevelopmentError("non-finite plug-in logit")
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    numerator = math.exp(value)
    return numerator / (1.0 + numerator)


def _time(value: str) -> datetime:
    if not isinstance(value, str) or value.endswith("Z"):
        raise PrivateDevelopmentError("adapter timestamp must be source-local naive ISO-8601")
    try:
        result = datetime.fromisoformat(value)
    except ValueError as error:
        raise PrivateDevelopmentError("adapter timestamp is invalid") from error
    if result.tzinfo is not None:
        raise PrivateDevelopmentError("adapter timestamp must not assert a timezone")
    return result


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    kind: str
    half_life_days: float | None
    process_variance_per_day: float

    def payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "half_life_days": self.half_life_days,
            "process_variance_per_day": self.process_variance_per_day,
            "reset_policy": "NO_RESET",
            "update": "BERNOULLI_LOGISTIC_DIAGONAL_ADF_RANK_ONE",
        }


CANDIDATES = (
    Candidate("static_baseline", "STATIC", None, 0.0),
    Candidate("random_walk_no_reset", "RANDOM_WALK", None, FROZEN_PROCESS_VARIANCE_PER_DAY),
    Candidate("mean_reversion_no_reset", "MEAN_REVERSION", FROZEN_MEAN_REVERSION_HALF_LIFE_DAYS, FROZEN_PROCESS_VARIANCE_PER_DAY),
)


@dataclass
class PlayerState:
    mean: float = 0.0
    variance: float = 1.0
    at: datetime | None = None

    def transition(self, target: datetime, candidate: Candidate) -> "PlayerState":
        if self.at is None:
            return PlayerState(self.mean, self.variance, target)
        days = (target - self.at).total_seconds() / 86400.0
        if days < 0.0:
            raise PrivateDevelopmentError("adapter supplied a future origin")
        mean, variance = self.mean, self.variance
        if candidate.kind == "RANDOM_WALK":
            variance += candidate.process_variance_per_day * days
        elif candidate.kind == "MEAN_REVERSION":
            if candidate.half_life_days is None or candidate.half_life_days <= 0.0:
                raise PrivateDevelopmentError("invalid mean-reversion half life")
            phi = math.exp(-math.log(2.0) * days / candidate.half_life_days)
            mean = phi * mean
            variance = phi * phi * variance + candidate.process_variance_per_day * days
        elif candidate.kind != "STATIC":
            raise PrivateDevelopmentError("unknown candidate kind")
        if not math.isfinite(mean) or not math.isfinite(variance) or variance <= 0.0:
            raise PrivateDevelopmentError("non-finite or non-PSD transition")
        return PlayerState(mean, variance, target)


def _map_parts(observation: MapObservation) -> tuple[datetime, tuple[str, ...], tuple[str, ...], int]:
    at = _time(observation.source_local_event_start)
    if observation.blue_win not in (0, 1):
        raise PrivateDevelopmentError("adapter outcome must be binary")
    blue = tuple(item.source_player_id for item in observation.player_observations if item.game_side == "blue")
    red = tuple(item.source_player_id for item in observation.player_observations if item.game_side == "red")
    if len(blue) != 5 or len(red) != 5 or len(set(blue + red)) != 10:
        raise PrivateDevelopmentError("adapter must supply exact distinct five-player lineups")
    return at, blue, red, observation.blue_win


def _assert_psd(states: Mapping[str, PlayerState]) -> None:
    for identifier, state in states.items():
        if not math.isfinite(state.mean) or not math.isfinite(state.variance) or state.variance <= 0.0:
            raise PrivateDevelopmentError(f"non-finite or non-PSD diagonal posterior: {identifier}")


def _predict(states: Mapping[str, PlayerState], observation: MapObservation, candidate: Candidate) -> tuple[float, float, dict[str, PlayerState]]:
    at, blue, red, _ = _map_parts(observation)
    transitioned = {identifier: states.get(identifier, PlayerState()).transition(at, candidate) for identifier in blue + red}
    # x_i is +1/5 for blue and -1/5 for red.  This is the complete feature
    # vector for the intentionally player-only baseline.
    mean = sum(transitioned[item].mean for item in blue) / 5.0 - sum(transitioned[item].mean for item in red) / 5.0
    variance = sum(transitioned[item].variance for item in blue + red) / 25.0
    _finite(mean, "predictive mean")
    if variance < 0.0 or not math.isfinite(variance):
        raise PrivateDevelopmentError("non-finite or non-PSD predictive variance")
    # This is the existing validated logistic-normal integral, rather than a
    # plug-in sigmoid of the posterior mean.
    probability = posterior_predictive_expected_result(mean, variance)
    return probability, math.sqrt(variance), transitioned


def _update(states: dict[str, PlayerState], observation: MapObservation, candidate: Candidate) -> None:
    at, blue, red, outcome = _map_parts(observation)
    transitioned = {identifier: states.get(identifier, PlayerState()).transition(at, candidate) for identifier in blue + red}
    eta = sum(transitioned[item].mean for item in blue) / 5.0 - sum(transitioned[item].mean for item in red) / 5.0
    # The ADF/Laplace score and Hessian are derivatives of the conditional
    # Bernoulli likelihood at eta.  Logistic-normal integration is reserved
    # for scored forecasts, where posterior uncertainty belongs.
    probability = _plugin_sigmoid(eta)
    residual = float(outcome) - probability
    # Bernoulli-logistic diagonal assumed-density filtering / one-step
    # rank-one Laplace update.  Every coordinate has x_i in {+0.2, -0.2}.
    weights = {identifier: 0.2 for identifier in blue}
    weights.update({identifier: -0.2 for identifier in red})
    curvature = probability * (1.0 - probability)
    x_sigma_x = sum(transitioned[identifier].variance * weight * weight for identifier, weight in weights.items())
    denominator = 1.0 + curvature * x_sigma_x
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise PrivateDevelopmentError("non-finite ADF/Laplace denominator")
    for identifier, weight in weights.items():
        prior = transitioned[identifier]
        mean = prior.mean + prior.variance * weight * residual / denominator
        variance = prior.variance - curvature * (prior.variance * weight) ** 2 / denominator
        if not math.isfinite(mean) or not math.isfinite(variance) or variance <= 0.0:
            raise PrivateDevelopmentError("non-finite or non-PSD ADF/Laplace update")
        states[identifier] = PlayerState(mean, variance, prior.at)
    _assert_psd(states)


def _folds_by_id(input_data: PrivatePlayerRatingInput) -> dict[str, tuple[MapObservation, ...]]:
    expected = ("TRAIN", "DEVELOPMENT", "VALIDATION")
    actual = tuple(fold.fold_id for fold in input_data.folds)
    if actual != expected:
        raise PrivateDevelopmentError("accepted adapter folds are not the fixed train/development/validation order")
    for fold in input_data.folds:
        maps = [item.source_game_id for item in fold.map_observations]
        origins = [{"source_game_id": item.source_game_id, "ordered_origin_map_ids": list(item.ordered_origin_map_ids)} for item in fold.map_observations]
        if _sha256(_canonical_bytes(maps)) != fold.ordered_map_ids_sha256:
            raise PrivateDevelopmentError("accepted adapter fold map digest mismatch")
        if _sha256(_canonical_bytes(origins)) != fold.ordered_origin_identities_sha256:
            raise PrivateDevelopmentError("accepted adapter fold origin digest mismatch")
    return {fold.fold_id: fold.map_observations for fold in input_data.folds}


def _all_maps(input_data: PrivatePlayerRatingInput) -> dict[str, MapObservation]:
    maps = [item for fold in input_data.folds for item in fold.map_observations]
    if len(maps) != input_data.map_count or len({item.source_game_id for item in maps}) != len(maps):
        raise PrivateDevelopmentError("adapter map identity/count mismatch")
    return {item.source_game_id: item for item in maps}


def _state_for_exact_origins(
    *, maps: Mapping[str, MapObservation], origin_ids: tuple[str, ...], origin_sha256: str,
    target: MapObservation, candidate: Candidate,
) -> dict[str, PlayerState]:
    if _sha256(_canonical_bytes(list(origin_ids))) != origin_sha256:
        raise PrivateDevelopmentError("adapter ordered origin digest mismatch")
    if len(set(origin_ids)) != len(origin_ids):
        raise PrivateDevelopmentError("adapter ordered origins are repeated")
    states: dict[str, PlayerState] = {}
    target_at = _time(target.source_local_event_start)
    # Do not sort, filter, supplement, or otherwise reinterpret the frozen
    # ordered origin ledger.  The adapter already checked its exact boundary.
    for identifier in origin_ids:
        observation = maps.get(identifier)
        if observation is None:
            raise PrivateDevelopmentError("adapter ordered origin is missing")
        if observation.source_series_id == target.source_series_id or (target_at - _time(observation.source_local_event_start)).total_seconds() <= 48.0 * 3600.0:
            raise PrivateDevelopmentError("adapter origin violates 48-hour/source-series boundary")
        _update(states, observation, candidate)
    _assert_psd(states)
    return states


def _metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        return {"n": 0, "log_loss": None, "brier": None, "calibration": None}
    losses, briers = [], []
    bins = [{"count": 0, "probability": 0.0, "outcome": 0.0} for _ in range(5)]
    for item in predictions:
        probability, outcome = _finite(item["probability"], "probability"), item["y"]
        if outcome not in (0, 1) or not 0.0 < probability < 1.0:
            raise PrivateDevelopmentError("invalid evaluation prediction")
        losses.append(-(outcome * math.log(probability) + (1 - outcome) * math.log(1.0 - probability)))
        briers.append((probability - outcome) ** 2)
        bucket = min(4, int(probability * 5.0))
        bins[bucket]["count"] += 1
        bins[bucket]["probability"] += probability
        bins[bucket]["outcome"] += outcome
    calibration = []
    ece = 0.0
    for bucket in bins:
        count = bucket["count"]
        mean_p = None if not count else bucket["probability"] / count
        observed = None if not count else bucket["outcome"] / count
        if count:
            ece += abs(mean_p - observed) * count / len(predictions)
        calibration.append({"count": count, "mean_probability": mean_p, "observed_rate": observed})
    return {"n": len(predictions), "log_loss": sum(losses) / len(losses), "brier": sum(briers) / len(briers), "calibration": {"kind": "five_equal_width_bins", "ece": ece, "bins": calibration}}


def _posterior_payload(states: Mapping[str, PlayerState]) -> list[dict[str, Any]]:
    _assert_psd(states)
    return [
        {"player_id": identifier, "posterior_mean": DISPLAY_ANCHOR + DISPLAY_SCALE * state.mean, "posterior_uncertainty": DISPLAY_SCALE * math.sqrt(state.variance)}
        for identifier, state in sorted(states.items(), key=lambda item: (-item[1].mean, item[0]))
    ]


def _candidate_prequential_evaluations(input_data: PrivatePlayerRatingInput, candidate: Candidate) -> tuple[dict[str, tuple[dict[str, Any], str, str]], dict[str, PlayerState], MapObservation]:
    """One exact-prefix replay: no origin reordering or repeated fitting."""

    maps = _all_maps(input_data)
    ordered = [item for fold in input_data.folds for item in fold.map_observations]
    applied: list[str] = []
    states: dict[str, PlayerState] = {}
    predictions: dict[str, list[dict[str, Any]]] = {"DEVELOPMENT": [], "VALIDATION": []}
    ledgers: dict[str, list[dict[str, Any]]] = {"DEVELOPMENT": [], "VALIDATION": []}
    last_validation_states: dict[str, PlayerState] | None = None
    last_validation: MapObservation | None = None
    for observation in ordered:
        origins = list(observation.ordered_origin_map_ids)
        if _sha256(_canonical_bytes(origins)) != observation.ordered_origin_sha256:
            raise PrivateDevelopmentError("adapter ordered origin digest mismatch")
        if origins[:len(applied)] != applied:
            raise PrivateDevelopmentError("accepted exact origin ledger is not a monotone prefix")
        target_at = _time(observation.source_local_event_start)
        # Recheck every declared origin against this target, including the
        # previously applied prefix.  An injected/forged adapter object may
        # otherwise make an old origin illegal for a later target.
        for identifier in origins:
            origin = maps.get(identifier)
            if origin is None:
                raise PrivateDevelopmentError("adapter ordered origin is missing")
            if origin.source_series_id == observation.source_series_id or (target_at - _time(origin.source_local_event_start)).total_seconds() <= 48.0 * 3600.0:
                raise PrivateDevelopmentError("adapter origin violates 48-hour/source-series boundary")
        for identifier in origins[len(applied):]:
            origin = maps.get(identifier)
            if origin is None:
                raise PrivateDevelopmentError("adapter ordered origin is missing")
            _update(states, origin, candidate)
            applied.append(identifier)
        probability, uncertainty, _ = _predict(states, observation, candidate)
        if observation.fold_id in predictions:
            predictions[observation.fold_id].append({"source_game_id": observation.source_game_id, "probability": probability, "uncertainty": uncertainty, "y": observation.blue_win})
            ledgers[observation.fold_id].append({"source_game_id": observation.source_game_id, "ordered_origin_map_ids": origins, "ordered_origin_sha256": observation.ordered_origin_sha256})
        if observation.fold_id == "VALIDATION":
            last_validation_states = dict(states)
            last_validation = observation
    if last_validation_states is None or last_validation is None:
        raise PrivateDevelopmentError("accepted adapter has no validation target")
    reports = {fold: (_metrics(predictions[fold]), _sha256(_canonical_bytes(ledgers[fold])), _sha256(_canonical_bytes(predictions[fold]))) for fold in predictions}
    return reports, last_validation_states, last_validation


def _assert_accepted_pins(input_data: PrivatePlayerRatingInput) -> None:
    actual = (input_data.manifest_sha256, input_data.rows_sha256, input_data.selected_target_sha256, input_data.split_payload_sha256)
    expected = (ACCEPTED_G1_PINS.manifest_sha256, ACCEPTED_G1_PINS.rows_sha256, ACCEPTED_G1_PINS.selected_target_sha256, ACCEPTED_G1_PINS.split_payload_sha256)
    if actual != expected:
        raise PrivateDevelopmentError("adapter input does not match code-held accepted G1 pins")


def build_private_development_artifact(
    *, input_loader: Callable[[], PrivatePlayerRatingInput] = load_accepted_lpl_private_player_rating_input,
) -> dict[str, Any]:
    """Evaluate fixed candidates; selection is development-only, validation is a gate."""

    input_data = input_loader()
    _assert_accepted_pins(input_data)
    if dict(input_data.claim_ceiling) != dict(ADAPTER_CLAIM_CEILING):
        raise PrivateDevelopmentError("adapter claim ceiling does not preserve the private boundary")
    folds = _folds_by_id(input_data)
    recomputed_player_observations = sum(len(observation.player_observations) for fold in input_data.folds for observation in fold.map_observations)
    if recomputed_player_observations != input_data.player_observation_count:
        raise PrivateDevelopmentError("adapter player observation count mismatch")
    config = {
        "display": {"anchor": DISPLAY_ANCHOR, "scale": DISPLAY_SCALE},
        "candidates": [candidate.payload() for candidate in CANDIDATES],
        "evaluation": {"kind": "ROLLING_PREQUENTIAL_EXACT_ADAPTER_ORIGINS", "selection_fold": "DEVELOPMENT", "external_validation_fold": "VALIDATION", "posterior_predictive": "existing_validated_logistic_normal_integral"},
        # This rule is declared in the executable config before candidate
        # metrics are computed: a development winner may not be retained if it
        # loses to the static baseline on either external validation metric.
        "decision_rule": {"development_tie_break": ["log_loss", "brier", "candidate_id"], "validation_gate": "candidate_log_loss_and_brier_must_not_exceed_static_baseline"},
    }
    results = []
    candidate_end_states: dict[str, tuple[dict[str, PlayerState], MapObservation]] = {}
    for candidate in CANDIDATES:
        reports, validation_states, validation_latest = _candidate_prequential_evaluations(input_data, candidate)
        development, development_origins, development_predictions = reports["DEVELOPMENT"]
        validation, validation_origins, validation_predictions = reports["VALIDATION"]
        candidate_end_states[candidate.candidate_id] = (validation_states, validation_latest)
        finite = all(value is None or math.isfinite(value) for report in (development, validation) for value in (report["log_loss"], report["brier"], None if report["calibration"] is None else report["calibration"]["ece"]))
        results.append({"candidate": candidate.payload(), "development": development, "validation": validation, "fold_origin_sha256": {"TRAIN": next(fold.ordered_origin_identities_sha256 for fold in input_data.folds if fold.fold_id == "TRAIN"), "DEVELOPMENT": development_origins, "VALIDATION": validation_origins}, "fold_prediction_sha256": {"DEVELOPMENT": development_predictions, "VALIDATION": validation_predictions}, "diagnostics": {"converged": True, "convergence_kind": "ONE_STEP_ADF_LAPLACE_NO_ITERATIVE_OPTIMIZER", "finite": finite, "covariance": "DIAGONAL_ASSUMED_DENSITY_PSD", "auxiliary_resource_channels": "ABSENT"}})
    selectable = [item for item in results if item["development"]["n"] and item["diagnostics"]["finite"]]
    development_winner = min(selectable, key=lambda item: (item["development"]["log_loss"], item["development"]["brier"], item["candidate"]["candidate_id"])) if selectable else None
    static = next(item for item in results if item["candidate"]["candidate_id"] == "static_baseline")
    external_ok = development_winner is not None and development_winner["validation"]["log_loss"] <= static["validation"]["log_loss"] and development_winner["validation"]["brier"] <= static["validation"]["brier"]
    selected_id = development_winner["candidate"]["candidate_id"] if external_ok else None
    selected = next((candidate for candidate in CANDIDATES if candidate.candidate_id == selected_id), None)
    development_candidate = next((candidate for candidate in CANDIDATES if development_winner is not None and candidate.candidate_id == development_winner["candidate"]["candidate_id"]), None)
    development_states, validation_latest = (None, None) if development_candidate is None else candidate_end_states[development_candidate.candidate_id]
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "result_state": "DEVELOPMENT_CANDIDATE" if selected else "NO_WINNER",
        "decision": {"development_winner_candidate_id": None if development_winner is None else development_winner["candidate"]["candidate_id"], "external_validation_gate_passed": external_ok, "selected_candidate_id": selected_id},
        "private_scope": {"authorizes": ["private_model_fit", "private_rank_selection"], "blocked": ["forecast", "prediction", "production", "publication", "promotion", "sota", "final_holdout"]},
        "adapter_input_pins": {"manifest_sha256": input_data.manifest_sha256, "rows_sha256": input_data.rows_sha256, "selected_target_sha256": input_data.selected_target_sha256, "split_payload_sha256": input_data.split_payload_sha256, "map_count": input_data.map_count, "player_observation_count": input_data.player_observation_count, "fold_map_digests": {fold.fold_id: fold.ordered_map_ids_sha256 for fold in input_data.folds}, "fold_origin_digests": {fold.fold_id: fold.ordered_origin_identities_sha256 for fold in input_data.folds}},
        "config": config,
        "config_sha256": _sha256(_canonical_bytes(config)),
        "candidate_results": results,
        "development_winner_posterior_ratings": None if development_candidate is None else {"candidate_id": development_candidate.candidate_id, "as_of_source_game_id": validation_latest.source_game_id, "ordered_origin_sha256": validation_latest.ordered_origin_sha256, "validation_gate_passed": external_ok, "ratings": _posterior_payload(development_states)},
        "posterior_ratings": [] if selected is None else _posterior_payload(candidate_end_states[selected.candidate_id][0]),
        "output_checks": {"all_finite": True, "covariance": "DIAGONAL_ASSUMED_DENSITY_PSD", "display_scale": {"anchor": DISPLAY_ANCHOR, "scale": DISPLAY_SCALE}},
    }
    artifact["artifact_sha256"] = _sha256(_canonical_bytes(artifact))
    return artifact


def _validate_private_development_artifact(
    artifact: Mapping[str, Any], *, expected_schema_version: str = SCHEMA_VERSION
) -> dict[str, Any]:
    if not isinstance(artifact, Mapping):
        raise PrivateDevelopmentError("private artifact must be an object")
    unsigned = dict(artifact)
    claimed = unsigned.pop("artifact_sha256", None)
    if claimed != _sha256(_canonical_bytes(unsigned)):
        raise PrivateDevelopmentError("private artifact digest drift")
    if artifact.get("schema_version") != expected_schema_version or artifact.get("result_state") not in {"DEVELOPMENT_CANDIDATE", "NO_WINNER"}:
        raise PrivateDevelopmentError("private artifact state is invalid")
    if artifact.get("private_scope", {}).get("authorizes") != ["private_model_fit", "private_rank_selection"]:
        raise PrivateDevelopmentError("private artifact exceeds accepted scope")
    checks = artifact.get("output_checks")
    if not isinstance(checks, Mapping) or checks.get("all_finite") is not True or checks.get("covariance") != "DIAGONAL_ASSUMED_DENSITY_PSD":
        raise PrivateDevelopmentError("private artifact finite/PSD output checks are invalid")
    for candidate in artifact.get("candidate_results", []):
        diagnostics = candidate.get("diagnostics", {}) if isinstance(candidate, Mapping) else {}
        if diagnostics.get("converged") is not True or diagnostics.get("finite") is not True or diagnostics.get("covariance") != "DIAGONAL_ASSUMED_DENSITY_PSD":
            raise PrivateDevelopmentError("private artifact convergence/finite/PSD diagnostics are invalid")
    for value in artifact.get("posterior_ratings", []):
        _finite(value.get("posterior_mean"), "posterior_mean")
        if _finite(value.get("posterior_uncertainty"), "posterior_uncertainty") < 0.0:
            raise PrivateDevelopmentError("posterior uncertainty must be non-negative")
    development_posterior = artifact.get("development_winner_posterior_ratings")
    if development_posterior is not None:
        if not isinstance(development_posterior, Mapping) or not isinstance(development_posterior.get("ratings"), list):
            raise PrivateDevelopmentError("development-winner posterior is invalid")
        for value in development_posterior["ratings"]:
            _finite(value.get("posterior_mean"), "development posterior_mean")
            if _finite(value.get("posterior_uncertainty"), "development posterior_uncertainty") < 0.0:
                raise PrivateDevelopmentError("development posterior uncertainty must be non-negative")
    return dict(artifact)


def verify_private_development_artifact(artifact: Mapping[str, Any], *, expected_artifact_sha256: str) -> dict[str, Any]:
    """Acceptance verifier: external pin is mandatory; self-hash is not authority."""

    verified = _validate_private_development_artifact(artifact)
    if verified["artifact_sha256"] != expected_artifact_sha256:
        raise PrivateDevelopmentError("private artifact does not match independently pinned digest")
    return verified


def verify_legacy_private_development_artifact_v2(
    artifact: Mapping[str, Any], *, expected_artifact_sha256: str
) -> dict[str, Any]:
    """Verify the frozen v2 mechanics input used by descriptive Team evidence.

    This compatibility path is intentionally named and schema-pinned.  New
    Player development callers must use :func:`verify_private_development_artifact`,
    which accepts only the v3 display-scale contract.
    """

    verified = _validate_private_development_artifact(
        artifact, expected_schema_version=LEGACY_SCHEMA_VERSION_V2
    )
    if verified["artifact_sha256"] != expected_artifact_sha256:
        raise PrivateDevelopmentError("legacy private artifact does not match independently pinned digest")
    return verified


def write_private_development_artifact(artifact: Mapping[str, Any], path: Path) -> str:
    data = _canonical_bytes(_validate_private_development_artifact(artifact)) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise PrivateDevelopmentError("artifact output must not be a symlink, non-regular file, or hardlink")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return _sha256(data)
