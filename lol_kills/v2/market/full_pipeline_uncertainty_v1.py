"""Generate event-level epistemic draws by refitting the frozen prediction pipeline.

Each draw independently resamples development series for the player/team rating
state, development series for terminal-Draft terms, and phase-one series for
the bounded recalibration.  It never reads the target event outcome or a market
price.  The resulting artifact is non-authorizing until independently
registered with the exact recalibration, phase-two, prediction, and source
bindings.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from lol_kills.v2.draft.terminal import future_prediction_ledger as draft_ledger
from lol_kills.v2.draft.terminal.development_artifact_v3 import (
    SELECTED_CANDIDATE_ID,
    SELECTED_RIDGE_STRENGTH,
    _calibration as fit_draft_calibration,
)
from lol_kills.v2.draft.terminal.development_evaluation import (
    pre_event_team_elo_logits,
)
from lol_kills.v2.draft.terminal.development_evaluation_v2 import fit_penalized
from lol_kills.v2.draft.terminal.development_snapshot import (
    load_development_snapshot,
)
from lol_kills.v2.draft.terminal.model import TerminalModel
from lol_kills.v2.ratings.player import post_validation_refit_v1 as rating_refit

from . import phase_one_evaluation_v1 as evaluation
from . import phase_one_recalibration_v1 as recalibration


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/market/full_pipeline_uncertainty_v1.py"
SCHEMA_VERSION = "scryglass:event-full-pipeline-uncertainty:v2"
RESULT_STATE = "EVENT_FULL_PIPELINE_BOOTSTRAP_CAPTURED_NON_AUTHORIZING"
OUTPUT_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/match-winner-market-v1/event-uncertainty"
)
RESAMPLES = 2_000
MASTER_SEED = 20_260_805
PERCENTILE_INTERVAL = (0.025, 0.975)
AUTHORITY = {
    "uncertainty_identity_authority": False,
    "probability_authority": False,
    "odds_authority": False,
    "expected_value_authority": False,
    "recommendation_authority": False,
    "betting_authority": False,
}
CLAIM_CEILING = (
    "Event-specific fresh-refit full-pipeline bootstrap calculation only. Independent "
    "recalibration, uncertainty, phase-two, probability, quote, settlement, "
    "and market authority remain required."
)


class FullPipelineUncertaintyError(RuntimeError):
    """A refit, resample, target binding, or uncertainty artifact failed closed."""


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
        raise FullPipelineUncertaintyError("uncertainty value is not canonical") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _seed(draw_id: int, stream: str) -> int:
    raw = f"{MASTER_SEED}|{draw_id}|{stream}".encode("ascii")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16) % (2**32)


def _sample_indices(size: int, *, draw_id: int, stream: str) -> list[int]:
    if size <= 0:
        raise FullPipelineUncertaintyError("bootstrap population is empty")
    rng = np.random.default_rng(_seed(draw_id, stream))
    return rng.integers(0, size, size=size).astype(int).tolist()


def _sample_digest(indices: Sequence[int]) -> str:
    return _canonical_sha256(list(indices))


def _sigmoid(value: float) -> float:
    if value >= 40.0:
        return 1.0 - 1e-15
    if value <= -40.0:
        return 1e-15
    return 1.0 / (1.0 + math.exp(-value))


def _logit(probability: float) -> float:
    clipped = min(max(probability, 1e-6), 0.999999)
    return math.log(clipped / (1.0 - clipped))


def _apply_calibration(probability: float, intercept: float, slope: float) -> float:
    return _sigmoid(intercept + slope * _logit(probability))


def _fresh_refit(
    root: Path,
    locator_value: str,
    environment: Mapping[str, str],
) -> tuple[str, bytes, dict[str, Any], dict[str, Any]]:
    try:
        locator, raw, value = rating_refit.load_post_validation_refit_v1(
            locator_value,
            root=root,
            environment=environment,
        )
        prepared = rating_refit.prepare_probability_replay_v1(
            value,
            root=root,
            environment=environment,
        )
    except Exception as exc:
        raise FullPipelineUncertaintyError(
            "fresh post-validation rating refit is invalid"
        ) from exc
    return locator, raw, value, prepared


def _target(
    root: Path, target_prediction_locator: str
) -> tuple[bytes, dict[str, Any], dict[str, Any], dict[str, Any]]:
    locator = PurePosixPath(target_prediction_locator)
    if (
        locator.is_absolute()
        or any(part in {"", ".", ".."} for part in locator.parts)
        or tuple(locator.parts[: len(draft_ledger.PREDICTION_PREFIX.parts)])
        != draft_ledger.PREDICTION_PREFIX.parts
        or locator.suffix != ".json"
    ):
        raise FullPipelineUncertaintyError("target prediction locator is invalid")
    raw = evaluation._read_regular(root, locator.as_posix(), "target Draft prediction")
    try:
        prediction = draft_ledger.validate_draft_prediction_receipt(
            evaluation._strict_object(raw, "target Draft prediction"), root=root
        )
    except Exception as exc:
        raise FullPipelineUncertaintyError("target Draft prediction is invalid") from exc
    ratings = prediction["input_receipts"]["ratings_prediction"]["value"]
    metadata_value = prediction["input_receipts"]["draft_metadata"]["value"]
    source_raw = draft_ledger._decode_source_payload(
        prediction["input_receipts"]["draft_source_payload"], "draft source"
    )
    metadata = draft_ledger._validate_draft_metadata(
        metadata_value, source_payload_raw=source_raw
    )
    return raw, prediction, ratings, metadata


def _rating_target_probability(
    *,
    rating_refit_prepared: Mapping[str, Any],
    sampled_indices: Sequence[int],
) -> float:
    try:
        return rating_refit.sampled_rating_probability_v1(
            rating_refit_prepared, sampled_indices
        )
    except Exception as exc:
        raise FullPipelineUncertaintyError(
            "fresh bootstrap rating probability failed"
        ) from exc


def _assert_target_refit_binding(
    *,
    target: Mapping[str, Any],
    target_ratings: Mapping[str, Any],
    refit_prepared: Mapping[str, Any],
) -> None:
    refit = refit_prepared["refit"]
    roster = refit_prepared["roster"]
    event = target["event"]
    refit_event = refit["event"]
    exact = {
        "event_id": refit_event["event_id"],
        "league": refit_event["league"],
        "patch": refit_event["patch"],
        "blue_organization_id": refit_event["blue_organization_id"],
        "red_organization_id": refit_event["red_organization_id"],
    }
    if any(event[key] != expected for key, expected in exact.items()):
        raise FullPipelineUncertaintyError(
            "terminal Draft and fresh rating refit event identities differ"
        )
    target_inputs = target_ratings["input_receipts"]
    refit_inputs = refit["input_receipts"]
    if (
        target_inputs["roster"]["raw_sha256"]
        != refit_inputs["roster_raw_sha256"]
        or target_inputs["patch"]["raw_sha256"]
        != refit_inputs["patch_raw_sha256"]
        or target_inputs["roster"]["canonical_sha256"]
        != refit_inputs["roster_canonical_sha256"]
        or target_ratings["event"]["event_start_utc"]
        != refit_event["event_start_utc"]
        or roster["teams"][0]["organization_name"]
        != event["blue_organization_name"]
        or roster["teams"][1]["organization_name"]
        != event["red_organization_name"]
        or evaluation._timestamp(refit["built_at_utc"], "rating refit built_at")
        > evaluation._timestamp(target["captured_at_utc"], "target captured_at")
    ):
        raise FullPipelineUncertaintyError(
            "terminal Draft and fresh rating refit receipts differ"
        )


def _fresh_point_calculation(
    *,
    target: Mapping[str, Any],
    refit_prepared: Mapping[str, Any],
) -> tuple[float, float, float]:
    try:
        rating_probability = rating_refit.point_rating_probability_v1(
            refit_prepared
        )
    except Exception as exc:
        raise FullPipelineUncertaintyError(
            "fresh point rating probability failed"
        ) from exc
    draft_logit = float(target["draft_index"]["scaled_logit_blue"])
    raw_combined = _sigmoid(_logit(rating_probability) + draft_logit)
    return rating_probability, draft_logit, raw_combined


def _cluster_partition(rows: Sequence[Any]) -> tuple[list[str], dict[str, list[Any]]]:
    latest: dict[str, Any] = {}
    grouped: dict[str, list[Any]] = {}
    for row in rows:
        grouped.setdefault(row.dependence_cluster_id, []).append(row)
        latest[row.dependence_cluster_id] = max(
            latest.get(row.dependence_cluster_id, row.date), row.date
        )
    order = [
        cluster
        for cluster, _date in sorted(latest.items(), key=lambda item: (item[1], item[0]))
    ]
    return order, grouped


def _sample_cluster_rows(
    order: Sequence[str],
    grouped: Mapping[str, Sequence[Any]],
    indices: Sequence[int],
) -> list[Any]:
    # Bootstrap multiplicity is retained while chronological cluster order is
    # restored without the quadratic ``order.index`` lookup per draw.
    return [
        row
        for index in sorted(indices)
        for row in grouped[order[index]]
    ]


def _draft_target_scaled_logit(
    *,
    rows: Sequence[Any],
    baseline_logits: Mapping[str, float],
    metadata: Mapping[str, Any],
    train_order: Sequence[str],
    calibration_order: Sequence[str],
    grouped: Mapping[str, Sequence[Any]],
    train_indices: Sequence[int],
    calibration_indices: Sequence[int],
) -> tuple[float, dict[str, Any]]:
    train = _sample_cluster_rows(train_order, grouped, train_indices)
    calibration_rows = _sample_cluster_rows(
        calibration_order, grouped, calibration_indices
    )
    fit = fit_penalized(
        train,
        SELECTED_CANDIDATE_ID,
        SELECTED_RIDGE_STRENGTH,
        baseline_logits,
    )
    from lol_kills.v2.draft.terminal.development_evaluation_v2 import (
        baseline_adjusted_logits,
    )

    calibration_logits = baseline_adjusted_logits(
        calibration_rows, fit, baseline_logits
    )
    _method, scale, _reports = fit_draft_calibration(
        calibration_logits, [row.label_a for row in calibration_rows]
    )
    champion: dict[str, float] = {}
    ally: dict[str, float] = {}
    counter: dict[str, float] = {}
    for feature, coefficient in zip(fit.vocabulary, fit.beta):
        value = float(coefficient)
        if abs(value) <= 1e-15:
            continue
        if feature.startswith("main|"):
            _, role, champion_id = feature.split("|", 2)
            champion[f"{role}|{champion_id}"] = value
        elif feature.startswith("ally|"):
            _, first, second = feature.split("|", 2)
            ally[f"{first}|{second}"] = value
        elif feature.startswith("counter|"):
            _, role, first, second = feature.split("|", 3)
            counter[f"{role}|{first}|{second}"] = value
        else:
            raise FullPipelineUncertaintyError("bootstrap Draft feature is unknown")
    model = TerminalModel(
        model_version="draft-terminal-bootstrap-v1.0.0",
        model_as_of=max(row.date for row in rows).isoformat(),
        intercept=0.0,
        calibration_slope=scale,
        calibration_intercept=0.0,
        uncertainty_logit_sd=0.0,
        champion_role_logit=champion,
        ally_synergy_logit=ally,
        counter_logit=counter,
        artifact_sha256="0" * 64,
        authorizes_prediction=False,
    )
    composition = draft_ledger._score_composition(metadata, model)
    return float(composition["scaled_logit_blue"]), {
        "feature_count": len(fit.vocabulary),
        "calibration_slope": scale,
        "optimizer_iterations": fit.optimizer_iterations,
        "optimizer_gradient_max_abs": fit.optimizer_gradient_max_abs,
    }


def _phase_one_sample(
    rows: Sequence[Mapping[str, Any]], indices: Sequence[int]
) -> list[Mapping[str, Any]]:
    series_order = sorted({str(row["series_id"]) for row in rows})
    grouped = {
        series_id: [row for row in rows if row["series_id"] == series_id]
        for series_id in series_order
    }
    return [row for index in indices for row in grouped[series_order[index]]]


def _draw_from_prepared(prepared: Mapping[str, Any], draw_id: int) -> dict[str, Any]:
    rating_indices = _sample_indices(
        len(prepared["rating_refit_prepared"]["input_data"].development_series),
        draw_id=draw_id,
        stream="ratings-development",
    )
    draft_train_indices = _sample_indices(
        len(prepared["draft_train_order"]),
        draw_id=draw_id,
        stream="draft-development-train",
    )
    draft_calibration_indices = _sample_indices(
        len(prepared["draft_calibration_order"]),
        draw_id=draw_id,
        stream="draft-development-calibration",
    )
    phase_one_series = sorted(
        {str(row["series_id"]) for row in prepared["phase_one_rows"]}
    )
    phase_one_indices = _sample_indices(
        len(phase_one_series), draw_id=draw_id, stream="phase-one-recalibration"
    )
    rating_probability = _rating_target_probability(
        rating_refit_prepared=prepared["rating_refit_prepared"],
        sampled_indices=rating_indices,
    )
    draft_logit, draft_diagnostics = _draft_target_scaled_logit(
        rows=prepared["draft_rows"],
        baseline_logits=prepared["draft_baseline_logits"],
        metadata=prepared["target_metadata"],
        train_order=prepared["draft_train_order"],
        calibration_order=prepared["draft_calibration_order"],
        grouped=prepared["draft_grouped"],
        train_indices=draft_train_indices,
        calibration_indices=draft_calibration_indices,
    )
    raw_combined = _sigmoid(_logit(rating_probability) + draft_logit)
    phase_one_sample = _phase_one_sample(
        prepared["phase_one_rows"], phase_one_indices
    )
    labels = [int(row["blue_win"]) for row in phase_one_sample]
    combined_fit = recalibration.fit_bounded_recalibration(
        [float(row["ratings_plus_draft"]) for row in phase_one_sample], labels
    )
    rating_fit = recalibration.fit_bounded_recalibration(
        [float(row["ratings_only"]) for row in phase_one_sample], labels
    )
    probability = _apply_calibration(
        raw_combined, combined_fit["intercept"], combined_fit["slope"]
    )
    return {
        "draw_id": draw_id,
        "seeds": {
            "ratings_development": _seed(draw_id, "ratings-development"),
            "draft_development_train": _seed(draw_id, "draft-development-train"),
            "draft_development_calibration": _seed(
                draw_id, "draft-development-calibration"
            ),
            "phase_one_recalibration": _seed(draw_id, "phase-one-recalibration"),
        },
        "sample_digests": {
            "ratings_development": _sample_digest(rating_indices),
            "draft_development_train": _sample_digest(draft_train_indices),
            "draft_development_calibration": _sample_digest(
                draft_calibration_indices
            ),
            "phase_one_recalibration": _sample_digest(phase_one_indices),
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


def _prepare(
    *,
    phase_one_result_locator: str,
    target_prediction_locator: str,
    rating_refit_locator: str,
    root: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    result_raw = evaluation._read_regular(
        root, phase_one_result_locator, "phase-one result"
    )
    result = evaluation.validate_phase_one_evaluation_result(
        evaluation._strict_object(result_raw, "phase-one result")
    )
    if result["phase_one_models_passed"] is not True:
        raise FullPipelineUncertaintyError("phase-one models did not pass")
    phase_one_rows, _snapshot_raw, _snapshot, _outcome_raw, _outcomes = (
        recalibration._phase_one_rows(result=result, root=root)
    )
    _target_raw, target_prediction, target_ratings, target_metadata = _target(
        root, target_prediction_locator
    )
    refit_locator, refit_raw, refit, refit_prepared = _fresh_refit(
        root, rating_refit_locator, environment
    )
    _assert_target_refit_binding(
        target=target_prediction,
        target_ratings=target_ratings,
        refit_prepared=refit_prepared,
    )
    draft_rows, _draft_source = load_development_snapshot(root)
    draft_order, draft_grouped = _cluster_partition(draft_rows)
    calibration_count = max(20, len(draft_order) // 10)
    draft_train_order = draft_order[:-calibration_count]
    draft_calibration_order = draft_order[-calibration_count:]
    if not draft_train_order or not draft_calibration_order:
        raise FullPipelineUncertaintyError("Draft bootstrap partition is empty")
    return {
        "rating_refit_locator": refit_locator,
        "rating_refit_raw": refit_raw,
        "rating_refit": refit,
        "rating_refit_prepared": refit_prepared,
        "target_prediction": target_prediction,
        "target_metadata": target_metadata,
        "phase_one_rows": phase_one_rows,
        "draft_rows": draft_rows,
        "draft_baseline_logits": pre_event_team_elo_logits(draft_rows),
        "draft_train_order": draft_train_order,
        "draft_calibration_order": draft_calibration_order,
        "draft_grouped": draft_grouped,
    }


def _worker_init(config: Mapping[str, Any]) -> None:
    global _WORKER_PREPARED
    _WORKER_PREPARED = _prepare(
        phase_one_result_locator=config["phase_one_result_locator"],
        target_prediction_locator=config["target_prediction_locator"],
        rating_refit_locator=config["rating_refit_locator"],
        root=Path(config["root"]),
        environment=config["environment"],
    )


def _worker_draw(draw_id: int) -> dict[str, Any]:
    if _WORKER_PREPARED is None:
        raise FullPipelineUncertaintyError("uncertainty worker was not initialized")
    return _draw_from_prepared(_WORKER_PREPARED, draw_id)


def _source_locks(root: Path) -> list[dict[str, Any]]:
    locators = (
        SOURCE_LOCATOR,
        recalibration.SOURCE_LOCATOR,
        evaluation.SOURCE_LOCATOR,
        "lol_kills/v2/market/phase_one_evaluation_registry_v1.py",
        "lol_kills/v2/market/match_winner_future_protocol_registry_v1.py",
        rating_refit.SOURCE_LOCATOR,
        "lol_kills/v2/ratings/player/multileague_v3_prediction_ledger.py",
        "lol_kills/v2/ratings/player/multileague_runner.py",
        "lol_kills/v2/ratings/player/multileague_v2_runner.py",
        "lol_kills/v2/ratings/player/multileague_development.py",
        "lol_kills/v2/ratings/player/multileague_v3_registry_v3.py",
        "lol_kills/v2/draft/terminal/future_prediction_ledger.py",
        "lol_kills/v2/draft/terminal/development_artifact_v3.py",
        "lol_kills/v2/draft/terminal/development_evaluation.py",
        "lol_kills/v2/draft/terminal/development_evaluation_v2.py",
        "lol_kills/v2/draft/terminal/development_snapshot.py",
        "lol_kills/v2/draft/terminal/model.py",
    )
    return [evaluation._source_record(root, locator) for locator in locators]


def build_event_uncertainty_candidate(
    *,
    phase_one_result_locator: str,
    recalibration_artifact_locator: str,
    target_prediction_locator: str,
    rating_refit_locator: str,
    workers: int,
    root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
) -> dict[str, Any]:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise FullPipelineUncertaintyError("workers must be a positive integer")
    recalibration_raw = evaluation._read_regular(
        root, recalibration_artifact_locator, "recalibration artifact"
    )
    calibration = recalibration.validate_phase_one_recalibration_artifact(
        evaluation._strict_object(recalibration_raw, "recalibration artifact")
    )
    result_raw = evaluation._read_regular(
        root, phase_one_result_locator, "phase-one result"
    )
    result = evaluation.validate_phase_one_evaluation_result(
        evaluation._strict_object(result_raw, "phase-one result")
    )
    recalibration._registered_pass(
        result_locator=phase_one_result_locator,
        root=root,
        environment=environment,
    )
    if (
        calibration["inputs"]["phase_one_result_artifact_sha256"]
        != result["artifact_sha256"]
    ):
        raise FullPipelineUncertaintyError(
            "recalibration and phase-one result differ"
        )
    target_raw, target_prediction, _ratings, _metadata = _target(
        root, target_prediction_locator
    )
    config = {
        "root": str(root.resolve()),
        "phase_one_result_locator": phase_one_result_locator,
        "target_prediction_locator": target_prediction_locator,
        "rating_refit_locator": rating_refit_locator,
        "environment": dict(environment),
    }
    prepared = _prepare(
        phase_one_result_locator=phase_one_result_locator,
        target_prediction_locator=target_prediction_locator,
        rating_refit_locator=rating_refit_locator,
        root=root,
        environment=environment,
    )
    refit_phase = prepared["rating_refit"]["phase_one_pass"]
    if (
        refit_phase["result_locator"] != phase_one_result_locator
        or refit_phase["result_raw_sha256"] != _sha256_bytes(result_raw)
        or refit_phase["result_artifact_sha256"] != result["artifact_sha256"]
    ):
        raise FullPipelineUncertaintyError(
            "fresh rating refit and phase-one result differ"
        )
    if workers == 1:
        draws = [_draw_from_prepared(prepared, draw_id) for draw_id in range(RESAMPLES)]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_worker_init,
            initargs=(config,),
        ) as pool:
            draws = list(pool.map(_worker_draw, range(RESAMPLES), chunksize=1))
    draws.sort(key=lambda item: item["draw_id"])
    if [item["draw_id"] for item in draws] != list(range(RESAMPLES)):
        raise FullPipelineUncertaintyError("bootstrap draw inventory changed")
    probabilities = np.asarray(
        [float(item["probability_blue"]) for item in draws], dtype=float
    )
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0.0) or np.any(probabilities >= 1.0):
        raise FullPipelineUncertaintyError("bootstrap produced an invalid probability")
    point_rating, point_draft_logit, point_raw = _fresh_point_calculation(
        target=target_prediction,
        refit_prepared=prepared["rating_refit_prepared"],
    )
    point_model = calibration["models"]["ratings_plus_draft"]
    point = _apply_calibration(
        point_raw, float(point_model["intercept"]), float(point_model["slope"])
    )
    interval = [
        float(np.quantile(probabilities, PERCENTILE_INTERVAL[0])),
        float(np.quantile(probabilities, PERCENTILE_INTERVAL[1])),
    ]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "event": {
            **target_prediction["event"],
            "target_prediction_locator": target_prediction_locator,
            "target_prediction_raw_sha256": _sha256_bytes(target_raw),
            "target_prediction_artifact_sha256": target_prediction["artifact_sha256"],
        },
        "inputs": {
            "phase_one_result_locator": phase_one_result_locator,
            "phase_one_result_raw_sha256": _sha256_bytes(result_raw),
            "phase_one_result_artifact_sha256": result["artifact_sha256"],
            "recalibration_artifact_locator": recalibration_artifact_locator,
            "recalibration_artifact_raw_sha256": _sha256_bytes(recalibration_raw),
            "recalibration_artifact_sha256": calibration["artifact_sha256"],
            "rating_refit_locator": prepared["rating_refit_locator"],
            "rating_refit_raw_sha256": _sha256_bytes(
                prepared["rating_refit_raw"]
            ),
            "rating_refit_artifact_sha256": prepared["rating_refit"][
                "artifact_sha256"
            ],
            "rating_source_snapshot_locator": prepared["rating_refit"][
                "source_snapshot"
            ]["locator"],
            "rating_source_snapshot_raw_sha256": prepared["rating_refit"][
                "source_snapshot"
            ]["raw_sha256"],
            "rating_source_snapshot_artifact_sha256": prepared["rating_refit"][
                "source_snapshot"
            ]["artifact_sha256"],
            "rating_roster_raw_sha256": prepared["rating_refit"][
                "input_receipts"
            ]["roster_raw_sha256"],
            "rating_roster_canonical_sha256": prepared["rating_refit"][
                "input_receipts"
            ]["roster_canonical_sha256"],
            "rating_patch_raw_sha256": prepared["rating_refit"][
                "input_receipts"
            ]["patch_raw_sha256"],
        },
        "bootstrap_contract": {
            "method": "series_cluster_bootstrap_full_prediction_pipeline",
            "confidence_level": 0.95,
            "resamples": RESAMPLES,
            "master_seed": MASTER_SEED,
            "percentile_interval": list(PERCENTILE_INTERVAL),
            "populations": {
                "ratings_development_series": len(
                    prepared["rating_refit_prepared"]["input_data"].development_series
                ),
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
            "draws_sha256": _canonical_sha256(draws),
            "probability_interval_blue": interval,
            "opposing_probability_interval_red": [1.0 - interval[1], 1.0 - interval[0]],
            "interval_is_epistemic": True,
            "interval_is_not_a_guarantee_of_binary_outcome_coverage": True,
        },
        "source_locks": _source_locks(root),
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
        "authority": dict(AUTHORITY),
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_event_uncertainty_candidate(
        payload, root=root, environment=environment
    )


def validate_event_uncertainty_candidate(
    payload: Mapping[str, Any],
    *,
    root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise FullPipelineUncertaintyError("uncertainty artifact must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version",
        "result_state",
        "event",
        "inputs",
        "bootstrap_contract",
        "point_calculation",
        "uncertainty",
        "source_locks",
        "qualification",
        "authority",
        "claim_ceiling",
        "artifact_sha256",
    }:
        raise FullPipelineUncertaintyError("uncertainty artifact structure changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise FullPipelineUncertaintyError("uncertainty artifact hash changed")
    if value.get("schema_version") != SCHEMA_VERSION or value.get("result_state") != RESULT_STATE:
        raise FullPipelineUncertaintyError("uncertainty artifact identity changed")
    event = value.get("event")
    expected_event_keys = {
        "event_id",
        "series_id",
        "game_number",
        "league",
        "patch",
        "blue_organization_id",
        "blue_organization_name",
        "red_organization_id",
        "red_organization_name",
        "target_prediction_locator",
        "target_prediction_raw_sha256",
        "target_prediction_artifact_sha256",
    }
    if not isinstance(event, Mapping) or set(event) != expected_event_keys:
        raise FullPipelineUncertaintyError("uncertainty event binding changed")
    target_locator = PurePosixPath(str(event["target_prediction_locator"]))
    if (
        target_locator.is_absolute()
        or tuple(
            target_locator.parts[: len(draft_ledger.PREDICTION_PREFIX.parts)]
        )
        != draft_ledger.PREDICTION_PREFIX.parts
        or target_locator.suffix != ".json"
    ):
        raise FullPipelineUncertaintyError("uncertainty target locator changed")
    evaluation._sha(event["target_prediction_raw_sha256"], "target raw sha")
    evaluation._sha(
        event["target_prediction_artifact_sha256"], "target artifact sha"
    )
    inputs = value.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "phase_one_result_locator",
        "phase_one_result_raw_sha256",
        "phase_one_result_artifact_sha256",
        "recalibration_artifact_locator",
        "recalibration_artifact_raw_sha256",
        "recalibration_artifact_sha256",
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
        raise FullPipelineUncertaintyError("uncertainty input binding changed")
    for key, item in inputs.items():
        if key.endswith("sha256"):
            evaluation._sha(item, f"inputs.{key}")
    target_raw, target, target_ratings, _target_metadata = _target(
        root, str(event["target_prediction_locator"])
    )
    refit_locator, refit_raw, refit, refit_prepared = _fresh_refit(
        root, str(inputs["rating_refit_locator"]), environment
    )
    _assert_target_refit_binding(
        target=target,
        target_ratings=target_ratings,
        refit_prepared=refit_prepared,
    )
    result_raw = evaluation._read_regular(
        root, str(inputs["phase_one_result_locator"]), "phase-one result"
    )
    result = evaluation.validate_phase_one_evaluation_result(
        evaluation._strict_object(result_raw, "phase-one result")
    )
    recalibration_raw = evaluation._read_regular(
        root,
        str(inputs["recalibration_artifact_locator"]),
        "recalibration artifact",
    )
    calibration = recalibration.validate_phase_one_recalibration_artifact(
        evaluation._strict_object(recalibration_raw, "recalibration artifact")
    )
    if (
        _sha256_bytes(target_raw) != event["target_prediction_raw_sha256"]
        or target["artifact_sha256"]
        != event["target_prediction_artifact_sha256"]
        or dict(event)
        != {
            **target["event"],
            "target_prediction_locator": event["target_prediction_locator"],
            "target_prediction_raw_sha256": _sha256_bytes(target_raw),
            "target_prediction_artifact_sha256": target["artifact_sha256"],
        }
        or _sha256_bytes(result_raw) != inputs["phase_one_result_raw_sha256"]
        or result["artifact_sha256"]
        != inputs["phase_one_result_artifact_sha256"]
        or result.get("phase_one_models_passed") is not True
        or _sha256_bytes(recalibration_raw)
        != inputs["recalibration_artifact_raw_sha256"]
        or calibration["artifact_sha256"]
        != inputs["recalibration_artifact_sha256"]
        or calibration["inputs"]["phase_one_result_artifact_sha256"]
        != result["artifact_sha256"]
        or refit_locator != inputs["rating_refit_locator"]
        or _sha256_bytes(refit_raw) != inputs["rating_refit_raw_sha256"]
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
    ):
        raise FullPipelineUncertaintyError(
            "uncertainty fresh-refit file binding changed"
        )
    contract = value.get("bootstrap_contract")
    if not isinstance(contract, Mapping) or set(contract) != {
        "method",
        "confidence_level",
        "resamples",
        "master_seed",
        "percentile_interval",
        "populations",
        "ratings_development_resampling",
        "draft_development_resampling",
        "phase_one_recalibration_resampling",
        "candidate_and_hyperparameters_fixed",
        "ratings_state_refit_in_each_resample",
        "draft_terms_refit_in_each_resample",
        "phase_one_recalibration_refit_in_each_resample",
        "phase_one_stored_predictions_used_for_recalibration_refit",
        "target_event_rating_and_draft_predictions_refit_in_each_resample",
        "fresh_post_validation_refit_exactly_bound",
        "fresh_point_rating_replayed_from_same_source_and_roster",
        "target_event_outcome_or_market_price_used",
        "failure_or_nonconvergence_action",
    } or contract.get("method") != "series_cluster_bootstrap_full_prediction_pipeline" or contract.get("confidence_level") != 0.95 or contract.get("resamples") != RESAMPLES or contract.get("master_seed") != MASTER_SEED or contract.get("percentile_interval") != list(PERCENTILE_INTERVAL) or any(
        contract.get(field) is not True
        for field in (
            "candidate_and_hyperparameters_fixed",
            "ratings_state_refit_in_each_resample",
            "draft_terms_refit_in_each_resample",
            "phase_one_recalibration_refit_in_each_resample",
            "phase_one_stored_predictions_used_for_recalibration_refit",
            "target_event_rating_and_draft_predictions_refit_in_each_resample",
            "fresh_post_validation_refit_exactly_bound",
            "fresh_point_rating_replayed_from_same_source_and_roster",
        )
    ) or contract.get("target_event_outcome_or_market_price_used") is not False:
        raise FullPipelineUncertaintyError("bootstrap contract changed")
    if (
        contract.get("ratings_development_resampling")
        != "series_with_replacement_preserve_chronological_order"
        or contract.get("draft_development_resampling")
        != "train_and_calibration_series_resampled_separately_with_replacement"
        or contract.get("phase_one_recalibration_resampling")
        != "series_with_replacement"
        or contract.get("failure_or_nonconvergence_action")
        != "event_probability_unavailable"
    ):
        raise FullPipelineUncertaintyError("bootstrap resampling semantics changed")
    populations = contract.get("populations")
    population_keys = {
        "ratings_development_series",
        "draft_development_train_series",
        "draft_development_calibration_series",
        "phase_one_recalibration_series",
    }
    if (
        not isinstance(populations, Mapping)
        or set(populations) != population_keys
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in populations.values()
        )
    ):
        raise FullPipelineUncertaintyError("bootstrap populations changed")
    if populations["ratings_development_series"] != len(
        refit_prepared["input_data"].development_series
    ):
        raise FullPipelineUncertaintyError(
            "fresh rating bootstrap population binding changed"
        )
    point = value.get("point_calculation")
    if not isinstance(point, Mapping) or set(point) != {
        "rating_probability_blue",
        "draft_scaled_logit_blue",
        "raw_probability_blue",
        "recalibration_intercept",
        "recalibration_slope",
        "probability_blue",
    }:
        raise FullPipelineUncertaintyError("point calculation changed")
    rating_point = evaluation._number(
        point["rating_probability_blue"], "point rating probability"
    )
    draft_point = evaluation._number(
        point["draft_scaled_logit_blue"], "point Draft logit"
    )
    raw_point = evaluation._number(point["raw_probability_blue"], "raw point")
    intercept = evaluation._number(
        point["recalibration_intercept"], "point intercept"
    )
    slope = evaluation._number(point["recalibration_slope"], "point slope")
    probability_point = evaluation._number(
        point["probability_blue"], "point probability"
    )
    if (
        not 0.0 < rating_point < 1.0
        or not 0.0 < raw_point < 1.0
        or not math.isclose(
            raw_point,
            _sigmoid(_logit(rating_point) + draft_point),
            abs_tol=1e-15,
        )
        or not recalibration.INTERCEPT_BOUNDS[0]
        <= intercept
        <= recalibration.INTERCEPT_BOUNDS[1]
        or not recalibration.SLOPE_BOUNDS[0]
        <= slope
        <= recalibration.SLOPE_BOUNDS[1]
        or not math.isclose(
            probability_point,
            _apply_calibration(raw_point, intercept, slope),
            abs_tol=1e-15,
        )
    ):
        raise FullPipelineUncertaintyError("point calculation does not replay")
    expected_rating_point, expected_draft_point, expected_raw_point = (
        _fresh_point_calculation(target=target, refit_prepared=refit_prepared)
    )
    point_model = calibration["models"]["ratings_plus_draft"]
    if (
        rating_point != expected_rating_point
        or draft_point != expected_draft_point
        or raw_point != expected_raw_point
        or intercept != point_model["intercept"]
        or slope != point_model["slope"]
    ):
        raise FullPipelineUncertaintyError(
            "point calculation is not the exact fresh-refit replay"
        )
    uncertainty = value.get("uncertainty")
    if not isinstance(uncertainty, Mapping) or set(uncertainty) != {
        "draws",
        "draws_sha256",
        "probability_interval_blue",
        "opposing_probability_interval_red",
        "interval_is_epistemic",
        "interval_is_not_a_guarantee_of_binary_outcome_coverage",
    }:
        raise FullPipelineUncertaintyError("uncertainty result changed")
    draws = uncertainty.get("draws")
    if not isinstance(draws, list) or len(draws) != RESAMPLES or [item.get("draw_id") for item in draws if isinstance(item, Mapping)] != list(range(RESAMPLES)):
        raise FullPipelineUncertaintyError("uncertainty draw inventory changed")
    if uncertainty.get("draws_sha256") != _canonical_sha256(draws):
        raise FullPipelineUncertaintyError("uncertainty draw hash changed")
    expected_draw_keys = {
        "draw_id",
        "seeds",
        "sample_digests",
        "refit",
        "probability_blue",
    }
    stream_bindings = {
        "ratings_development": (
            "ratings-development",
            "ratings_development_series",
        ),
        "draft_development_train": (
            "draft-development-train",
            "draft_development_train_series",
        ),
        "draft_development_calibration": (
            "draft-development-calibration",
            "draft_development_calibration_series",
        ),
        "phase_one_recalibration": (
            "phase-one-recalibration",
            "phase_one_recalibration_series",
        ),
    }
    probabilities_list: list[float] = []
    for draw_id, draw in enumerate(draws):
        if not isinstance(draw, Mapping) or set(draw) != expected_draw_keys:
            raise FullPipelineUncertaintyError("uncertainty draw structure changed")
        seeds = draw.get("seeds")
        digests = draw.get("sample_digests")
        if (
            not isinstance(seeds, Mapping)
            or set(seeds) != set(stream_bindings)
            or not isinstance(digests, Mapping)
            or set(digests) != set(stream_bindings)
        ):
            raise FullPipelineUncertaintyError("uncertainty sampling record changed")
        for name, (stream, population_name) in stream_bindings.items():
            if seeds[name] != _seed(draw_id, stream):
                raise FullPipelineUncertaintyError("uncertainty draw seed changed")
            expected_indices = _sample_indices(
                populations[population_name], draw_id=draw_id, stream=stream
            )
            if digests[name] != _sample_digest(expected_indices):
                raise FullPipelineUncertaintyError(
                    "uncertainty sample digest changed"
                )
        refit = draw.get("refit")
        if not isinstance(refit, Mapping) or set(refit) != {
            "rating_probability_blue",
            "draft_scaled_logit_blue",
            "draft",
            "raw_combined_probability_blue",
            "combined_recalibration_intercept",
            "combined_recalibration_slope",
            "rating_only_recalibration_intercept",
            "rating_only_recalibration_slope",
        }:
            raise FullPipelineUncertaintyError("uncertainty refit record changed")
        rating_probability = evaluation._number(
            refit["rating_probability_blue"], "draw rating probability"
        )
        draft_logit = evaluation._number(
            refit["draft_scaled_logit_blue"], "draw Draft logit"
        )
        raw_combined = evaluation._number(
            refit["raw_combined_probability_blue"], "draw raw combined"
        )
        combined_intercept = evaluation._number(
            refit["combined_recalibration_intercept"], "draw intercept"
        )
        combined_slope = evaluation._number(
            refit["combined_recalibration_slope"], "draw slope"
        )
        rating_intercept = evaluation._number(
            refit["rating_only_recalibration_intercept"], "draw rating intercept"
        )
        rating_slope = evaluation._number(
            refit["rating_only_recalibration_slope"], "draw rating slope"
        )
        probability = evaluation._number(
            draw["probability_blue"], "draw probability"
        )
        if (
            not 0.0 < rating_probability < 1.0
            or not 0.0 < raw_combined < 1.0
            or not 0.0 < probability < 1.0
            or not math.isclose(
                raw_combined,
                _sigmoid(_logit(rating_probability) + draft_logit),
                abs_tol=1e-15,
            )
            or not math.isclose(
                probability,
                _apply_calibration(
                    raw_combined, combined_intercept, combined_slope
                ),
                abs_tol=1e-15,
            )
            or not recalibration.INTERCEPT_BOUNDS[0]
            <= combined_intercept
            <= recalibration.INTERCEPT_BOUNDS[1]
            or not recalibration.SLOPE_BOUNDS[0]
            <= combined_slope
            <= recalibration.SLOPE_BOUNDS[1]
            or not recalibration.INTERCEPT_BOUNDS[0]
            <= rating_intercept
            <= recalibration.INTERCEPT_BOUNDS[1]
            or not recalibration.SLOPE_BOUNDS[0]
            <= rating_slope
            <= recalibration.SLOPE_BOUNDS[1]
        ):
            raise FullPipelineUncertaintyError("uncertainty draw does not replay")
        draft = refit.get("draft")
        if not isinstance(draft, Mapping) or set(draft) != {
            "feature_count",
            "calibration_slope",
            "optimizer_iterations",
            "optimizer_gradient_max_abs",
        }:
            raise FullPipelineUncertaintyError("Draft refit diagnostics changed")
        probabilities_list.append(probability)
    probabilities = np.asarray(probabilities_list)
    expected_interval = [
        float(np.quantile(probabilities, PERCENTILE_INTERVAL[0])),
        float(np.quantile(probabilities, PERCENTILE_INTERVAL[1])),
    ]
    if uncertainty.get("probability_interval_blue") != expected_interval or uncertainty.get("opposing_probability_interval_red") != [1.0 - expected_interval[1], 1.0 - expected_interval[0]] or uncertainty.get("interval_is_epistemic") is not True or uncertainty.get("interval_is_not_a_guarantee_of_binary_outcome_coverage") is not True:
        raise FullPipelineUncertaintyError("uncertainty interval does not replay")
    if value.get("source_locks") != _source_locks(root):
        raise FullPipelineUncertaintyError("uncertainty source lock changed")
    if value.get("qualification") != {
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
    }:
        raise FullPipelineUncertaintyError("uncertainty qualification changed")
    if value.get("authority") != AUTHORITY or value.get("claim_ceiling") != CLAIM_CEILING:
        raise FullPipelineUncertaintyError("uncertainty artifact exceeds authority")
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FullPipelineUncertaintyError(f"refusing to replace uncertainty: {path}")
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
            raise FullPipelineUncertaintyError(
                f"refusing to replace uncertainty: {path}"
            ) from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return _sha256_bytes(raw)


__all__ = [
    "MASTER_SEED",
    "RESAMPLES",
    "SCHEMA_VERSION",
    "SOURCE_LOCATOR",
    "FullPipelineUncertaintyError",
    "build_event_uncertainty_candidate",
    "validate_event_uncertainty_candidate",
    "write_no_clobber",
]
