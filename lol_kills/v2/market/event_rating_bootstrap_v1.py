"""Precompute the expensive rating leg of the frozen 2,000-draw bootstrap.

This runs after an independently registered phase-one pass and after a fresh
pre-event ratings receipt exists, but before terminal draft and market quote.
It uses the exact draw IDs, seeds, series samples, and rating refit function of
``full_pipeline_uncertainty_v1``.  The artifact is calculation evidence only;
exact parity with the frozen full pipeline and independent registration remain
mandatory.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Callable, Mapping

from . import full_pipeline_uncertainty_v1 as frozen
from . import phase_one_evaluation_v1 as evaluation
from . import phase_one_recalibration_v1 as recalibration


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/market/event_rating_bootstrap_v1.py"
SCHEMA_VERSION = "scryglass:event-rating-bootstrap:v2"
RESULT_STATE = "EVENT_RATING_BOOTSTRAP_PRECOMPUTED_NON_AUTHORIZING"
OUTPUT_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/match-winner-market-v1/event-rating-bootstrap"
)
AUTHORITY = {
    "rating_bootstrap_identity_authority": False,
    "uncertainty_identity_authority": False,
    "probability_authority": False,
    "odds_authority": False,
    "expected_value_authority": False,
    "recommendation_authority": False,
    "betting_authority": False,
}
CLAIM_CEILING = (
    "Fresh post-validation pre-event rating-refit leg of the frozen full-pipeline bootstrap only. "
    "Terminal Draft, recalibration, exact slow-path parity, independent "
    "registration, phase-two opening, quote, and market authority remain required."
)


class EventRatingBootstrapError(RuntimeError):
    """The pre-event target, phase-one pass, refit, or artifact failed closed."""


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
        raise EventRatingBootstrapError("rating bootstrap is not canonical") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise EventRatingBootstrapError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise EventRatingBootstrapError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _clock_sample(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise EventRatingBootstrapError("rating bootstrap clock must be timezone-aware")
    return value.astimezone(timezone.utc)


def _phase_one_snapshot(
    *, result_locator: str, root: Path, environment: Mapping[str, str]
) -> tuple[bytes, dict[str, Any], Mapping[str, Any]]:
    try:
        _registry, result, result_raw = recalibration._registered_pass(
            result_locator=result_locator,
            root=root,
            environment=environment,
        )
        _rows, _snapshot_raw, snapshot, _outcome_raw, _outcomes = (
            recalibration._phase_one_rows(result=result, root=root)
        )
    except Exception as exc:
        raise EventRatingBootstrapError(
            "independently registered phase-one pass is invalid"
        ) from exc
    return result_raw, result, snapshot


def _source_locks(root: Path) -> list[dict[str, Any]]:
    locators = [SOURCE_LOCATOR]
    for record in frozen._source_locks(root):
        locator = str(record["locator"])
        if locator not in locators:
            locators.append(locator)
    return [evaluation._source_record(root, locator) for locator in locators]


def _prepare(
    root: Path,
    rating_refit_locator: str,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    locator, raw, refit, replay = frozen._fresh_refit(
        root, rating_refit_locator, environment
    )
    try:
        point_probability = frozen.rating_refit.point_rating_probability_v1(replay)
    except Exception as exc:
        raise EventRatingBootstrapError(
            "fresh point rating probability failed"
        ) from exc
    return {
        "locator": locator,
        "raw": raw,
        "refit": refit,
        "replay": replay,
        "point_probability_blue": point_probability,
    }


def _draw(prepared: Mapping[str, Any], draw_id: int) -> dict[str, Any]:
    population = len(prepared["replay"]["input_data"].development_series)
    indices = frozen._sample_indices(
        population, draw_id=draw_id, stream="ratings-development"
    )
    probability = frozen._rating_target_probability(
        rating_refit_prepared=prepared["replay"],
        sampled_indices=indices,
    )
    return {
        "draw_id": draw_id,
        "seed": frozen._seed(draw_id, "ratings-development"),
        "sample_digest": frozen._sample_digest(indices),
        "rating_probability_blue": probability,
    }


_WORKER_PREPARED: dict[str, Any] | None = None


def _worker_init(config: Mapping[str, Any]) -> None:
    global _WORKER_PREPARED
    _WORKER_PREPARED = _prepare(
        Path(config["root"]),
        config["rating_refit_locator"],
        config["environment"],
    )


def _worker_draw(draw_id: int) -> dict[str, Any]:
    if _WORKER_PREPARED is None:
        raise EventRatingBootstrapError("rating bootstrap worker is uninitialized")
    return _draw(_WORKER_PREPARED, draw_id)


def build_event_rating_bootstrap_v1(
    *,
    phase_one_result_locator: str,
    rating_refit_locator: str,
    workers: int,
    root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise EventRatingBootstrapError("workers must be a positive integer")
    result_raw, result, _snapshot = _phase_one_snapshot(
        result_locator=phase_one_result_locator,
        root=root,
        environment=environment,
    )
    prepared = _prepare(root, rating_refit_locator, environment)
    refit = prepared["refit"]
    if (
        refit["phase_one_pass"]["result_locator"] != phase_one_result_locator
        or refit["phase_one_pass"]["result_raw_sha256"]
        != _sha256_bytes(result_raw)
        or refit["phase_one_pass"]["result_artifact_sha256"]
        != result["artifact_sha256"]
    ):
        raise EventRatingBootstrapError(
            "rating refit is not bound to the exact registered phase-one pass"
        )
    config = {
        "root": str(root.resolve()),
        "rating_refit_locator": prepared["locator"],
        "environment": dict(environment),
    }
    if workers == 1:
        draws = [_draw(prepared, draw_id) for draw_id in range(frozen.RESAMPLES)]
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
    built_at = _clock_sample(clock)
    captured_at = _timestamp(refit["built_at_utc"], "rating_refit.built_at")
    event_start = _timestamp(refit["event"]["event_start_utc"], "event.start")
    if built_at < captured_at or built_at >= event_start:
        raise EventRatingBootstrapError(
            "rating bootstrap was not completed before the event"
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "built_at_utc": built_at.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_after_all_rating_draws",
            "observed_wall_clock_utc": built_at.isoformat(),
            "user_supplied_timestamp_allowed": False,
            "all_draws_completed_before_observation": True,
        },
        "event": dict(refit["event"]),
        "inputs": {
            "phase_one_result_locator": phase_one_result_locator,
            "phase_one_result_raw_sha256": _sha256_bytes(result_raw),
            "phase_one_result_artifact_sha256": result["artifact_sha256"],
            "rating_refit_locator": prepared["locator"],
            "rating_refit_raw_sha256": _sha256_bytes(prepared["raw"]),
            "rating_refit_artifact_sha256": refit["artifact_sha256"],
            "rating_source_snapshot_locator": refit["source_snapshot"]["locator"],
            "rating_source_snapshot_raw_sha256": refit["source_snapshot"][
                "raw_sha256"
            ],
            "rating_source_snapshot_artifact_sha256": refit["source_snapshot"][
                "artifact_sha256"
            ],
            "rating_roster_raw_sha256": refit["input_receipts"][
                "roster_raw_sha256"
            ],
            "rating_roster_canonical_sha256": refit["input_receipts"][
                "roster_canonical_sha256"
            ],
            "rating_patch_raw_sha256": refit["input_receipts"][
                "patch_raw_sha256"
            ],
        },
        "bootstrap_contract": {
            "method": "fresh_post_validation_full_pipeline_rating_leg",
            "resamples": frozen.RESAMPLES,
            "master_seed": frozen.MASTER_SEED,
            "stream": "ratings-development",
            "development_series": len(
                prepared["replay"]["input_data"].development_series
            ),
            "series_with_replacement_preserve_chronological_order": True,
            "rating_state_refit_in_each_draw": True,
            "target_event_outcome_or_market_price_used": False,
        },
        "point_rating_probability_blue": prepared["point_probability_blue"],
        "draws": draws,
        "draws_sha256": _canonical_sha256(draws),
        "source_locks": _source_locks(root),
        "qualification": {
            "phase_one_models_independently_passed": True,
            "fresh_refit_created_after_registered_phase_one_pass": True,
            "fresh_source_roster_patch_and_point_replayed": True,
            "terminal_draft_used": False,
            "target_event_outcome_present": False,
            "target_event_outcome_accessed": False,
            "market_price_used": False,
            "independently_registered": False,
        },
        "authority": dict(AUTHORITY),
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_event_rating_bootstrap_v1(
        payload, root=root, environment=environment
    )


def validate_event_rating_bootstrap_v1(
    payload: Mapping[str, Any],
    *,
    root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise EventRatingBootstrapError("rating bootstrap must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version",
        "result_state",
        "built_at_utc",
        "clock_attestation",
        "event",
        "inputs",
        "bootstrap_contract",
        "point_rating_probability_blue",
        "draws",
        "draws_sha256",
        "source_locks",
        "qualification",
        "authority",
        "claim_ceiling",
        "artifact_sha256",
    }:
        raise EventRatingBootstrapError("rating bootstrap structure changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise EventRatingBootstrapError("rating bootstrap hash changed")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("result_state") != RESULT_STATE
    ):
        raise EventRatingBootstrapError("rating bootstrap identity changed")
    built_at = _timestamp(value.get("built_at_utc"), "built_at_utc")
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_after_all_rating_draws",
        "observed_wall_clock_utc": built_at.isoformat(),
        "user_supplied_timestamp_allowed": False,
        "all_draws_completed_before_observation": True,
    }:
        raise EventRatingBootstrapError("rating bootstrap clock changed")
    inputs = value.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "phase_one_result_locator",
        "phase_one_result_raw_sha256",
        "phase_one_result_artifact_sha256",
        "rating_refit_locator",
        "rating_refit_raw_sha256",
        "rating_refit_artifact_sha256",
        "rating_source_snapshot_locator",
        "rating_source_snapshot_raw_sha256",
        "rating_source_snapshot_artifact_sha256",
        "rating_roster_raw_sha256",
        "rating_roster_canonical_sha256",
        "rating_patch_raw_sha256",
    }:
        raise EventRatingBootstrapError("rating bootstrap inputs changed")
    for key, item in inputs.items():
        if key.endswith("sha256"):
            evaluation._sha(item, f"inputs.{key}")
    result_raw = evaluation._read_regular(
        root, inputs["phase_one_result_locator"], "phase-one result"
    )
    result = evaluation.validate_phase_one_evaluation_result(
        evaluation._strict_object(result_raw, "phase-one result")
    )
    prepared = _prepare(root, str(inputs["rating_refit_locator"]), environment)
    refit = prepared["refit"]
    if (
        _sha256_bytes(result_raw) != inputs["phase_one_result_raw_sha256"]
        or result["artifact_sha256"]
        != inputs["phase_one_result_artifact_sha256"]
        or result.get("phase_one_models_passed") is not True
        or prepared["locator"] != inputs["rating_refit_locator"]
        or _sha256_bytes(prepared["raw"])
        != inputs["rating_refit_raw_sha256"]
        or refit["artifact_sha256"] != inputs["rating_refit_artifact_sha256"]
        or refit["source_snapshot"]["locator"]
        != inputs["rating_source_snapshot_locator"]
        or refit["source_snapshot"]["raw_sha256"]
        != inputs["rating_source_snapshot_raw_sha256"]
        or refit["source_snapshot"]["artifact_sha256"]
        != inputs["rating_source_snapshot_artifact_sha256"]
        or refit["input_receipts"]["roster_raw_sha256"]
        != inputs["rating_roster_raw_sha256"]
        or refit["input_receipts"]["roster_canonical_sha256"]
        != inputs["rating_roster_canonical_sha256"]
        or refit["input_receipts"]["patch_raw_sha256"]
        != inputs["rating_patch_raw_sha256"]
        or refit["phase_one_pass"]["result_locator"]
        != inputs["phase_one_result_locator"]
        or refit["phase_one_pass"]["result_raw_sha256"]
        != inputs["phase_one_result_raw_sha256"]
        or refit["phase_one_pass"]["result_artifact_sha256"]
        != inputs["phase_one_result_artifact_sha256"]
        or value.get("event") != refit["event"]
        or built_at < _timestamp(refit["built_at_utc"], "rating_refit.built_at")
        or built_at >= _timestamp(refit["event"]["event_start_utc"], "event.start")
    ):
        raise EventRatingBootstrapError("rating bootstrap file binding changed")
    contract = value.get("bootstrap_contract")
    if not isinstance(contract, Mapping) or contract != {
        "method": "fresh_post_validation_full_pipeline_rating_leg",
        "resamples": frozen.RESAMPLES,
        "master_seed": frozen.MASTER_SEED,
        "stream": "ratings-development",
        "development_series": contract.get("development_series"),
        "series_with_replacement_preserve_chronological_order": True,
        "rating_state_refit_in_each_draw": True,
        "target_event_outcome_or_market_price_used": False,
    }:
        raise EventRatingBootstrapError("rating bootstrap contract changed")
    population = contract.get("development_series")
    if isinstance(population, bool) or not isinstance(population, int) or population <= 0:
        raise EventRatingBootstrapError("rating bootstrap population changed")
    if population != len(prepared["replay"]["input_data"].development_series):
        raise EventRatingBootstrapError(
            "fresh rating bootstrap population binding changed"
        )
    point_probability = value.get("point_rating_probability_blue")
    if (
        isinstance(point_probability, bool)
        or not isinstance(point_probability, (int, float))
        or not 0.0 < float(point_probability) < 1.0
        or float(point_probability) != prepared["point_probability_blue"]
    ):
        raise EventRatingBootstrapError(
            "fresh point rating probability binding changed"
        )
    draws = value.get("draws")
    if (
        not isinstance(draws, list)
        or len(draws) != frozen.RESAMPLES
        or value.get("draws_sha256") != _canonical_sha256(draws)
    ):
        raise EventRatingBootstrapError("rating bootstrap draw inventory changed")
    for draw_id, draw in enumerate(draws):
        if not isinstance(draw, Mapping) or set(draw) != {
            "draw_id",
            "seed",
            "sample_digest",
            "rating_probability_blue",
        }:
            raise EventRatingBootstrapError("rating bootstrap draw changed")
        indices = frozen._sample_indices(
            population, draw_id=draw_id, stream="ratings-development"
        )
        probability = draw.get("rating_probability_blue")
        if (
            draw.get("draw_id") != draw_id
            or draw.get("seed") != frozen._seed(draw_id, "ratings-development")
            or draw.get("sample_digest") != frozen._sample_digest(indices)
            or isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(float(probability))
            or not 0.0 < float(probability) < 1.0
        ):
            raise EventRatingBootstrapError("rating bootstrap draw does not replay")
    if value.get("source_locks") != _source_locks(root):
        raise EventRatingBootstrapError("rating bootstrap source lock changed")
    if value.get("qualification") != {
        "phase_one_models_independently_passed": True,
        "fresh_refit_created_after_registered_phase_one_pass": True,
        "fresh_source_roster_patch_and_point_replayed": True,
        "terminal_draft_used": False,
        "target_event_outcome_present": False,
        "target_event_outcome_accessed": False,
        "market_price_used": False,
        "independently_registered": False,
    }:
        raise EventRatingBootstrapError("rating bootstrap qualification changed")
    if value.get("authority") != AUTHORITY or value.get("claim_ceiling") != CLAIM_CEILING:
        raise EventRatingBootstrapError("rating bootstrap exceeds authority")
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise EventRatingBootstrapError(f"refusing to replace rating bootstrap: {path}")
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
            raise EventRatingBootstrapError(
                f"refusing to replace rating bootstrap: {path}"
            ) from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return _sha256_bytes(raw)


__all__ = [
    "OUTPUT_PREFIX",
    "SCHEMA_VERSION",
    "SOURCE_LOCATOR",
    "EventRatingBootstrapError",
    "build_event_rating_bootstrap_v1",
    "validate_event_rating_bootstrap_v1",
    "write_no_clobber",
]
