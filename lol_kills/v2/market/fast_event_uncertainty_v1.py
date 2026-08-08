"""Fast terminal-Draft completion of the frozen full-pipeline bootstrap.

The expensive rating refits are loaded from an exact pre-event artifact.  This
module then refits Draft terms and phase-one recalibration for the same 2,000
draw IDs and seeds.  Its nested frozen-contract candidate is validated by the
original pre-boundary validator.  Independent slow/fast draw parity on a fresh
verification target is still required before registration.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Callable, Mapping

import numpy as np

from lol_kills.v2.draft.terminal import future_prediction_ledger as draft_ledger
from lol_kills.v2.draft.terminal.development_evaluation import (
    pre_event_team_elo_logits,
)
from lol_kills.v2.draft.terminal.development_snapshot import (
    load_development_snapshot,
)

from . import event_rating_bootstrap_v1 as rating_bootstrap
from . import full_pipeline_uncertainty_v1 as frozen
from . import phase_one_evaluation_v1 as evaluation
from . import phase_one_recalibration_v1 as recalibration


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/market/fast_event_uncertainty_v1.py"
SCHEMA_VERSION = "scryglass:event-full-pipeline-uncertainty-fast:v2"
RESULT_STATE = "FAST_EVENT_UNCERTAINTY_CAPTURED_NON_AUTHORIZING"
OUTPUT_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/match-winner-market-v1/event-uncertainty-fast"
)
AUTHORITY = {
    "fast_decomposition_identity_authority": False,
    "uncertainty_identity_authority": False,
    "probability_authority": False,
    "odds_authority": False,
    "expected_value_authority": False,
    "recommendation_authority": False,
    "betting_authority": False,
}
CLAIM_CEILING = (
    "Operational decomposition of the fresh-refit frozen 2,000-draw uncertainty candidate "
    "only. Exact slow-path parity, independent registration, phase-two opening, "
    "event-probability registration, quote, settlement, and market authority "
    "remain required."
)


class FastEventUncertaintyError(RuntimeError):
    """A target binding, refit, decomposition, or frozen replay failed closed."""


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
        raise FastEventUncertaintyError("fast uncertainty is not canonical") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise FastEventUncertaintyError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise FastEventUncertaintyError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _clock_sample(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise FastEventUncertaintyError("fast uncertainty clock must be timezone-aware")
    return value.astimezone(timezone.utc)


def _rating_artifact(
    root: Path,
    locator_value: str,
    environment: Mapping[str, str],
) -> tuple[str, bytes, dict[str, Any]]:
    locator = evaluation._locator(
        locator_value, rating_bootstrap.OUTPUT_PREFIX, "rating_bootstrap_locator"
    )
    raw = evaluation._read_regular(root, locator, "event rating bootstrap")
    try:
        value = rating_bootstrap.validate_event_rating_bootstrap_v1(
            evaluation._strict_object(raw, "event rating bootstrap"),
            root=root,
            environment=environment,
        )
    except Exception as exc:
        raise FastEventUncertaintyError("event rating bootstrap is invalid") from exc
    return locator, raw, value


def _prepare(
    *,
    phase_one_result_locator: str,
    target_prediction_locator: str,
    rating_bootstrap_locator: str,
    root: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    result_raw = evaluation._read_regular(
        root, phase_one_result_locator, "phase-one result"
    )
    result = evaluation.validate_phase_one_evaluation_result(
        evaluation._strict_object(result_raw, "phase-one result")
    )
    if result.get("phase_one_models_passed") is not True:
        raise FastEventUncertaintyError("phase-one models did not pass")
    phase_rows, _snapshot_raw, _snapshot, _outcome_raw, _outcomes = (
        recalibration._phase_one_rows(result=result, root=root)
    )
    target_raw, target, target_ratings, target_metadata = frozen._target(
        root, target_prediction_locator
    )
    rating_locator, rating_raw, rating = _rating_artifact(
        root, rating_bootstrap_locator, environment
    )
    refit_locator, refit_raw, refit, refit_prepared = frozen._fresh_refit(
        root, rating["inputs"]["rating_refit_locator"], environment
    )
    frozen._assert_target_refit_binding(
        target=target,
        target_ratings=target_ratings,
        refit_prepared=refit_prepared,
    )
    if (
        refit_locator != rating["inputs"]["rating_refit_locator"]
        or _sha256_bytes(refit_raw) != rating["inputs"]["rating_refit_raw_sha256"]
        or refit["artifact_sha256"]
        != rating["inputs"]["rating_refit_artifact_sha256"]
        or refit["event"] != rating["event"]
        or rating["inputs"]["phase_one_result_raw_sha256"]
        != _sha256_bytes(result_raw)
    ):
        raise FastEventUncertaintyError(
            "terminal Draft and pre-event rating bootstrap differ"
        )
    draft_rows, _draft_source = load_development_snapshot(root)
    draft_order, draft_grouped = frozen._cluster_partition(draft_rows)
    calibration_count = max(20, len(draft_order) // 10)
    train_order = draft_order[:-calibration_count]
    calibration_order = draft_order[-calibration_count:]
    if not train_order or not calibration_order:
        raise FastEventUncertaintyError("Draft bootstrap partition is empty")
    return {
        "result_raw": result_raw,
        "result": result,
        "phase_one_rows": phase_rows,
        "target_raw": target_raw,
        "target": target,
        "target_metadata": target_metadata,
        "rating_locator": rating_locator,
        "rating_raw": rating_raw,
        "rating": rating,
        "rating_refit": refit,
        "rating_refit_prepared": refit_prepared,
        "draft_rows": draft_rows,
        "draft_baseline_logits": pre_event_team_elo_logits(draft_rows),
        "draft_train_order": train_order,
        "draft_calibration_order": calibration_order,
        "draft_grouped": draft_grouped,
    }


def _fast_draw(prepared: Mapping[str, Any], draw_id: int) -> dict[str, Any]:
    rating_draw = prepared["rating"]["draws"][draw_id]
    train_indices = frozen._sample_indices(
        len(prepared["draft_train_order"]),
        draw_id=draw_id,
        stream="draft-development-train",
    )
    calibration_indices = frozen._sample_indices(
        len(prepared["draft_calibration_order"]),
        draw_id=draw_id,
        stream="draft-development-calibration",
    )
    phase_series = sorted(
        {str(row["series_id"]) for row in prepared["phase_one_rows"]}
    )
    phase_indices = frozen._sample_indices(
        len(phase_series), draw_id=draw_id, stream="phase-one-recalibration"
    )
    draft_logit, draft_diagnostics = frozen._draft_target_scaled_logit(
        rows=prepared["draft_rows"],
        baseline_logits=prepared["draft_baseline_logits"],
        metadata=prepared["target_metadata"],
        train_order=prepared["draft_train_order"],
        calibration_order=prepared["draft_calibration_order"],
        grouped=prepared["draft_grouped"],
        train_indices=train_indices,
        calibration_indices=calibration_indices,
    )
    rating_probability = float(rating_draw["rating_probability_blue"])
    raw_combined = frozen._sigmoid(
        frozen._logit(rating_probability) + draft_logit
    )
    phase_sample = frozen._phase_one_sample(
        prepared["phase_one_rows"], phase_indices
    )
    labels = [int(row["blue_win"]) for row in phase_sample]
    combined_fit = recalibration.fit_bounded_recalibration(
        [float(row["ratings_plus_draft"]) for row in phase_sample], labels
    )
    rating_fit = recalibration.fit_bounded_recalibration(
        [float(row["ratings_only"]) for row in phase_sample], labels
    )
    probability = frozen._apply_calibration(
        raw_combined, combined_fit["intercept"], combined_fit["slope"]
    )
    return {
        "draw_id": draw_id,
        "seeds": {
            "ratings_development": rating_draw["seed"],
            "draft_development_train": frozen._seed(
                draw_id, "draft-development-train"
            ),
            "draft_development_calibration": frozen._seed(
                draw_id, "draft-development-calibration"
            ),
            "phase_one_recalibration": frozen._seed(
                draw_id, "phase-one-recalibration"
            ),
        },
        "sample_digests": {
            "ratings_development": rating_draw["sample_digest"],
            "draft_development_train": frozen._sample_digest(train_indices),
            "draft_development_calibration": frozen._sample_digest(
                calibration_indices
            ),
            "phase_one_recalibration": frozen._sample_digest(phase_indices),
        },
        "refit": {
            "rating_probability_blue": rating_probability,
            "draft_scaled_logit_blue": draft_logit,
            "draft": draft_diagnostics,
            "raw_combined_probability_blue": raw_combined,
            "combined_recalibration_intercept": combined_fit["intercept"],
            "combined_recalibration_slope": combined_fit["slope"],
            "rating_only_recalibration_intercept": rating_fit["intercept"],
            "rating_only_recalibration_slope": rating_fit["slope"],
        },
        "probability_blue": probability,
    }


_WORKER_PREPARED: dict[str, Any] | None = None


def _worker_init(config: Mapping[str, Any]) -> None:
    global _WORKER_PREPARED
    _WORKER_PREPARED = _prepare(
        phase_one_result_locator=config["phase_one_result_locator"],
        target_prediction_locator=config["target_prediction_locator"],
        rating_bootstrap_locator=config["rating_bootstrap_locator"],
        root=Path(config["root"]),
        environment=config["environment"],
    )


def _worker_draw(draw_id: int) -> dict[str, Any]:
    if _WORKER_PREPARED is None:
        raise FastEventUncertaintyError("fast uncertainty worker is uninitialized")
    return _fast_draw(_WORKER_PREPARED, draw_id)


def _source_locks(root: Path) -> list[dict[str, Any]]:
    locators = [SOURCE_LOCATOR, rating_bootstrap.SOURCE_LOCATOR]
    for record in frozen._source_locks(root):
        locator = str(record["locator"])
        if locator not in locators:
            locators.append(locator)
    return [evaluation._source_record(root, locator) for locator in locators]


def _frozen_candidate(
    *,
    prepared: Mapping[str, Any],
    calibration: Mapping[str, Any],
    recalibration_locator: str,
    recalibration_raw: bytes,
    target_prediction_locator: str,
    phase_one_result_locator: str,
    draws: list[dict[str, Any]],
    root: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    target = prepared["target"]
    probabilities = np.asarray(
        [float(draw["probability_blue"]) for draw in draws], dtype=float
    )
    point_rating = float(prepared["rating"]["point_rating_probability_blue"])
    point_draft_logit = float(target["draft_index"]["scaled_logit_blue"])
    point_raw = frozen._sigmoid(
        frozen._logit(point_rating) + point_draft_logit
    )
    point_model = calibration["models"]["ratings_plus_draft"]
    point = frozen._apply_calibration(
        point_raw, float(point_model["intercept"]), float(point_model["slope"])
    )
    interval = [
        float(np.quantile(probabilities, frozen.PERCENTILE_INTERVAL[0])),
        float(np.quantile(probabilities, frozen.PERCENTILE_INTERVAL[1])),
    ]
    rating_population = prepared["rating"]["bootstrap_contract"][
        "development_series"
    ]
    candidate: dict[str, Any] = {
        "schema_version": frozen.SCHEMA_VERSION,
        "result_state": frozen.RESULT_STATE,
        "event": {
            **target["event"],
            "target_prediction_locator": target_prediction_locator,
            "target_prediction_raw_sha256": _sha256_bytes(
                prepared["target_raw"]
            ),
            "target_prediction_artifact_sha256": target["artifact_sha256"],
        },
        "inputs": {
            "phase_one_result_locator": phase_one_result_locator,
            "phase_one_result_raw_sha256": _sha256_bytes(
                prepared["result_raw"]
            ),
            "phase_one_result_artifact_sha256": prepared["result"][
                "artifact_sha256"
            ],
            "recalibration_artifact_locator": recalibration_locator,
            "recalibration_artifact_raw_sha256": _sha256_bytes(
                recalibration_raw
            ),
            "recalibration_artifact_sha256": calibration["artifact_sha256"],
            "rating_refit_locator": prepared["rating"]["inputs"][
                "rating_refit_locator"
            ],
            "rating_refit_raw_sha256": prepared["rating"]["inputs"][
                "rating_refit_raw_sha256"
            ],
            "rating_refit_artifact_sha256": prepared["rating"]["inputs"][
                "rating_refit_artifact_sha256"
            ],
            "rating_source_snapshot_locator": prepared["rating"]["inputs"][
                "rating_source_snapshot_locator"
            ],
            "rating_source_snapshot_raw_sha256": prepared["rating"]["inputs"][
                "rating_source_snapshot_raw_sha256"
            ],
            "rating_source_snapshot_artifact_sha256": prepared["rating"]["inputs"][
                "rating_source_snapshot_artifact_sha256"
            ],
            "rating_roster_raw_sha256": prepared["rating"]["inputs"][
                "rating_roster_raw_sha256"
            ],
            "rating_roster_canonical_sha256": prepared["rating"]["inputs"][
                "rating_roster_canonical_sha256"
            ],
            "rating_patch_raw_sha256": prepared["rating"]["inputs"][
                "rating_patch_raw_sha256"
            ],
        },
        "bootstrap_contract": {
            "method": "series_cluster_bootstrap_full_prediction_pipeline",
            "confidence_level": 0.95,
            "resamples": frozen.RESAMPLES,
            "master_seed": frozen.MASTER_SEED,
            "percentile_interval": list(frozen.PERCENTILE_INTERVAL),
            "populations": {
                "ratings_development_series": rating_population,
                "draft_development_train_series": len(
                    prepared["draft_train_order"]
                ),
                "draft_development_calibration_series": len(
                    prepared["draft_calibration_order"]
                ),
                "phase_one_recalibration_series": len(
                    {
                        str(row["series_id"])
                        for row in prepared["phase_one_rows"]
                    }
                ),
            },
            "ratings_development_resampling": "series_with_replacement_preserve_chronological_order",
            "draft_development_resampling": "train_and_calibration_series_resampled_separately_with_replacement",
            "phase_one_recalibration_resampling": "series_with_replacement",
            "candidate_and_hyperparameters_fixed": True,
            "ratings_state_refit_in_each_resample": True,
            "draft_terms_refit_in_each_resample": True,
            "phase_one_recalibration_refit_in_each_resample": True,
            "phase_one_stored_predictions_used_for_recalibration_refit": True,
            "target_event_rating_and_draft_predictions_refit_in_each_resample": True,
            "fresh_post_validation_refit_exactly_bound": True,
            "fresh_point_rating_replayed_from_same_source_and_roster": True,
            "target_event_outcome_or_market_price_used": False,
            "failure_or_nonconvergence_action": "event_probability_unavailable",
        },
        "point_calculation": {
            "rating_probability_blue": point_rating,
            "draft_scaled_logit_blue": point_draft_logit,
            "raw_probability_blue": point_raw,
            "recalibration_intercept": point_model["intercept"],
            "recalibration_slope": point_model["slope"],
            "probability_blue": point,
        },
        "uncertainty": {
            "draws": draws,
            "draws_sha256": frozen._canonical_sha256(draws),
            "probability_interval_blue": interval,
            "opposing_probability_interval_red": [
                1.0 - interval[1],
                1.0 - interval[0],
            ],
            "interval_is_epistemic": True,
            "interval_is_not_a_guarantee_of_binary_outcome_coverage": True,
        },
        "source_locks": frozen._source_locks(root),
        "qualification": {
            "phase_one_models_independently_passed": True,
            "recalibration_artifact_present": True,
            "fresh_post_validation_rating_refit_validated": True,
            "fresh_rating_source_roster_patch_and_point_replayed": True,
            "recalibration_independently_registered": False,
            "uncertainty_independently_registered": False,
            "phase_two_opening_registered": False,
            "target_event_outcome_present": False,
            "target_event_outcome_accessed": False,
            "market_price_used": False,
        },
        "authority": dict(frozen.AUTHORITY),
        "claim_ceiling": frozen.CLAIM_CEILING,
    }
    candidate["artifact_sha256"] = frozen._canonical_sha256(candidate)
    return frozen.validate_event_uncertainty_candidate(
        candidate, root=root, environment=environment
    )


def build_fast_event_uncertainty_v1(
    *,
    phase_one_result_locator: str,
    recalibration_artifact_locator: str,
    target_prediction_locator: str,
    rating_bootstrap_locator: str,
    workers: int,
    root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise FastEventUncertaintyError("workers must be a positive integer")
    recalibration_raw = evaluation._read_regular(
        root, recalibration_artifact_locator, "recalibration artifact"
    )
    calibration = recalibration.validate_phase_one_recalibration_artifact(
        evaluation._strict_object(recalibration_raw, "recalibration artifact")
    )
    recalibration._registered_pass(
        result_locator=phase_one_result_locator,
        root=root,
        environment=environment,
    )
    prepared = _prepare(
        phase_one_result_locator=phase_one_result_locator,
        target_prediction_locator=target_prediction_locator,
        rating_bootstrap_locator=rating_bootstrap_locator,
        root=root,
        environment=environment,
    )
    if (
        calibration["inputs"]["phase_one_result_artifact_sha256"]
        != prepared["result"]["artifact_sha256"]
    ):
        raise FastEventUncertaintyError("recalibration and phase-one result differ")
    config = {
        "root": str(root.resolve()),
        "phase_one_result_locator": phase_one_result_locator,
        "target_prediction_locator": target_prediction_locator,
        "rating_bootstrap_locator": rating_bootstrap_locator,
        "environment": dict(environment),
    }
    if workers == 1:
        draws = [_fast_draw(prepared, draw_id) for draw_id in range(frozen.RESAMPLES)]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_worker_init,
            initargs=(config,),
        ) as pool:
            draws = list(
                pool.map(_worker_draw, range(frozen.RESAMPLES), chunksize=1)
            )
    draws.sort(key=lambda item: item["draw_id"])
    frozen_candidate = _frozen_candidate(
        prepared=prepared,
        calibration=calibration,
        recalibration_locator=recalibration_artifact_locator,
        recalibration_raw=recalibration_raw,
        target_prediction_locator=target_prediction_locator,
        phase_one_result_locator=phase_one_result_locator,
        draws=draws,
        root=root,
        environment=environment,
    )
    rating_only_raw = float(prepared["rating"]["point_rating_probability_blue"])
    rating_only_model = calibration["models"]["ratings_only"]
    rating_only_probability = frozen._apply_calibration(
        rating_only_raw,
        float(rating_only_model["intercept"]),
        float(rating_only_model["slope"]),
    )
    built_at = _clock_sample(clock)
    target_captured = _timestamp(
        prepared["target"]["captured_at_utc"], "target.captured_at"
    )
    event_start = _timestamp(
        prepared["rating"]["event"]["event_start_utc"], "event.start"
    )
    if built_at < target_captured or built_at >= event_start:
        raise FastEventUncertaintyError(
            "fast uncertainty was not completed before the event"
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "built_at_utc": built_at.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_after_all_terminal_draws",
            "observed_wall_clock_utc": built_at.isoformat(),
            "user_supplied_timestamp_allowed": False,
            "all_draws_completed_before_observation": True,
        },
        "frozen_contract_candidate": frozen_candidate,
        "evaluation_comparator": {
            "model": "recalibrated_rating_only",
            "raw_probability_blue": rating_only_raw,
            "recalibration_intercept": rating_only_model["intercept"],
            "recalibration_slope": rating_only_model["slope"],
            "probability_blue": rating_only_probability,
            "phase_two_market_price_used": False,
            "target_event_outcome_used": False,
        },
        "decomposition": {
            "rating_bootstrap_locator": prepared["rating_locator"],
            "rating_bootstrap_raw_sha256": _sha256_bytes(
                prepared["rating_raw"]
            ),
            "rating_bootstrap_artifact_sha256": prepared["rating"][
                "artifact_sha256"
            ],
            "rating_draws_sha256": prepared["rating"]["draws_sha256"],
            "terminal_draws_sha256": frozen_candidate["uncertainty"][
                "draws_sha256"
            ],
            "same_draw_ids_seeds_and_rating_probabilities_verified": True,
            "exact_slow_path_parity_independently_registered": False,
        },
        "source_locks": _source_locks(root),
        "authority": dict(AUTHORITY),
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_fast_event_uncertainty_v1(
        payload, root=root, environment=environment
    )


def validate_fast_event_uncertainty_v1(
    payload: Mapping[str, Any],
    *,
    root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise FastEventUncertaintyError("fast uncertainty must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version",
        "result_state",
        "built_at_utc",
        "clock_attestation",
        "frozen_contract_candidate",
        "evaluation_comparator",
        "decomposition",
        "source_locks",
        "authority",
        "claim_ceiling",
        "artifact_sha256",
    }:
        raise FastEventUncertaintyError("fast uncertainty structure changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise FastEventUncertaintyError("fast uncertainty hash changed")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("result_state") != RESULT_STATE
    ):
        raise FastEventUncertaintyError("fast uncertainty identity changed")
    built_at = _timestamp(value.get("built_at_utc"), "built_at_utc")
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_after_all_terminal_draws",
        "observed_wall_clock_utc": built_at.isoformat(),
        "user_supplied_timestamp_allowed": False,
        "all_draws_completed_before_observation": True,
    }:
        raise FastEventUncertaintyError("fast uncertainty clock changed")
    try:
        candidate = frozen.validate_event_uncertainty_candidate(
            value.get("frozen_contract_candidate"),
            root=root,
            environment=environment,
        )
    except Exception as exc:
        raise FastEventUncertaintyError(
            "nested frozen uncertainty candidate is invalid"
        ) from exc
    recalibration_locator = candidate["inputs"]["recalibration_artifact_locator"]
    recalibration_raw = evaluation._read_regular(
        root, recalibration_locator, "recalibration artifact"
    )
    calibration = recalibration.validate_phase_one_recalibration_artifact(
        evaluation._strict_object(recalibration_raw, "recalibration artifact")
    )
    decomposition = value.get("decomposition")
    if not isinstance(decomposition, Mapping) or set(decomposition) != {
        "rating_bootstrap_locator",
        "rating_bootstrap_raw_sha256",
        "rating_bootstrap_artifact_sha256",
        "rating_draws_sha256",
        "terminal_draws_sha256",
        "same_draw_ids_seeds_and_rating_probabilities_verified",
        "exact_slow_path_parity_independently_registered",
    }:
        raise FastEventUncertaintyError("fast decomposition changed")
    rating_locator, rating_raw, rating = _rating_artifact(
        root,
        str(decomposition["rating_bootstrap_locator"]),
        environment,
    )
    if (
        decomposition["rating_bootstrap_locator"] != rating_locator
        or decomposition["rating_bootstrap_raw_sha256"]
        != _sha256_bytes(rating_raw)
        or decomposition["rating_bootstrap_artifact_sha256"]
        != rating["artifact_sha256"]
        or decomposition["rating_draws_sha256"] != rating["draws_sha256"]
        or decomposition["terminal_draws_sha256"]
        != candidate["uncertainty"]["draws_sha256"]
        or decomposition[
            "same_draw_ids_seeds_and_rating_probabilities_verified"
        ]
        is not True
        or decomposition["exact_slow_path_parity_independently_registered"]
        is not False
    ):
        raise FastEventUncertaintyError("fast decomposition binding changed")
    for draw_id, (rating_draw, terminal_draw) in enumerate(
        zip(rating["draws"], candidate["uncertainty"]["draws"])
    ):
        if (
            rating_draw["draw_id"] != draw_id
            or terminal_draw["draw_id"] != draw_id
            or terminal_draw["seeds"]["ratings_development"]
            != rating_draw["seed"]
            or terminal_draw["sample_digests"]["ratings_development"]
            != rating_draw["sample_digest"]
            or terminal_draw["refit"]["rating_probability_blue"]
            != rating_draw["rating_probability_blue"]
        ):
            raise FastEventUncertaintyError("rating/terminal draw parity changed")
    target_locator = candidate["event"]["target_prediction_locator"]
    _target_raw, target, target_ratings, _metadata = frozen._target(
        root, target_locator
    )
    _refit_locator, _refit_raw, _refit, refit_prepared = frozen._fresh_refit(
        root, rating["inputs"]["rating_refit_locator"], environment
    )
    frozen._assert_target_refit_binding(
        target=target,
        target_ratings=target_ratings,
        refit_prepared=refit_prepared,
    )
    rating_only_raw = float(rating["point_rating_probability_blue"])
    rating_only_model = calibration["models"]["ratings_only"]
    expected_comparator = {
        "model": "recalibrated_rating_only",
        "raw_probability_blue": rating_only_raw,
        "recalibration_intercept": rating_only_model["intercept"],
        "recalibration_slope": rating_only_model["slope"],
        "probability_blue": frozen._apply_calibration(
            rating_only_raw,
            float(rating_only_model["intercept"]),
            float(rating_only_model["slope"]),
        ),
        "phase_two_market_price_used": False,
        "target_event_outcome_used": False,
    }
    if value.get("evaluation_comparator") != expected_comparator:
        raise FastEventUncertaintyError("rating-only evaluation comparator changed")
    if (
        candidate["inputs"]["rating_refit_locator"]
        != rating["inputs"]["rating_refit_locator"]
        or candidate["inputs"]["rating_refit_raw_sha256"]
        != rating["inputs"]["rating_refit_raw_sha256"]
        or candidate["inputs"]["rating_refit_artifact_sha256"]
        != rating["inputs"]["rating_refit_artifact_sha256"]
        or candidate["point_calculation"]["rating_probability_blue"]
        != rating["point_rating_probability_blue"]
        or built_at < _timestamp(target["captured_at_utc"], "target.captured_at")
        or built_at
        >= _timestamp(rating["event"]["event_start_utc"], "event.start")
    ):
        raise FastEventUncertaintyError("fast target chronology or binding changed")
    if value.get("source_locks") != _source_locks(root):
        raise FastEventUncertaintyError("fast uncertainty source lock changed")
    if value.get("authority") != AUTHORITY or value.get("claim_ceiling") != CLAIM_CEILING:
        raise FastEventUncertaintyError("fast uncertainty exceeds authority")
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FastEventUncertaintyError(f"refusing to replace fast uncertainty: {path}")
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
            raise FastEventUncertaintyError(
                f"refusing to replace fast uncertainty: {path}"
            ) from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return _sha256_bytes(raw)


__all__ = [
    "OUTPUT_PREFIX",
    "SCHEMA_VERSION",
    "SOURCE_LOCATOR",
    "FastEventUncertaintyError",
    "build_fast_event_uncertainty_v1",
    "validate_fast_event_uncertainty_v1",
    "write_no_clobber",
]
