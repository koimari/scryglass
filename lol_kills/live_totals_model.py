#!/usr/bin/env python3
"""Leakage-safe live total-kills evaluation and fail-closed runtime pricing."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
import tempfile
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAPS_PATH = ROOT / "data" / "lol" / "warehouse" / "parquet" / "maps.parquet"
WAREHOUSE_REFRESH_MANIFEST_PATH = (
    ROOT / "data" / "lol" / "warehouse" / "parquet" / "refresh_meta.json"
)
SOURCE_SNAPSHOT_ROOT = (
    ROOT / "data" / "lol" / "warehouse" / "snapshots" / "live_totals"
)
OUT_PATH = ROOT / "data" / "lol" / "models" / "live_totals_model_v2.json"

SCHEMA_VERSION = "scryglass.live-total-kills.v2"
LEAGUES = ("LCK", "LPL", "LEC", "LCS", "CBLOL")
CHECKPOINTS = (10, 15, 20, 25)
SPLIT_FRACTIONS = (0.50, 0.15, 0.15, 0.20)
FRESHNESS_LIMIT_DAYS = 14
MIN_CALIBRATION_GAMES = 40
MIN_CALIBRATION_SERIES = 20
MIN_TEST_GAMES = 40
MIN_PATCH_TEST_GAMES = 25
MAX_CDF_ERROR = 0.10
MIN_SELECTION_RMSE_IMPROVEMENT = 0.005
RIDGE_LAMBDA = 10.0
DEPENDENCE_INTERVAL_CONFIDENCE = 0.95
DEPENDENCE_INTERVAL_METHOD = "series_cluster_weighted_hoeffding"

FAMILY_ORDER = (
    "league",
    "team_pace",
    "head_to_head",
    "current_kills",
    "gold_difference",
    "completed_draft",
    "champion_scaling",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _root_locator(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _copy_content_addressed(source: Path, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or _sha256(destination) != expected_sha256:
            raise FileExistsError(f"content-addressed snapshot conflict: {destination}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as source_stream, os.fdopen(descriptor, "wb") as output:
            shutil.copyfileobj(source_stream, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        if _sha256(temporary) != expected_sha256:
            raise ValueError(f"staged source snapshot hash mismatch: {source}")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if _sha256(destination) != expected_sha256:
                raise
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_idempotent_no_clobber(path: Path, payload: dict[str, Any]) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    expected = hashlib.sha256(encoded).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise FileExistsError(f"refusing to replace existing snapshot manifest: {path}")
        return expected
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise
    finally:
        temporary.unlink(missing_ok=True)
    return expected


def snapshot_source_package(
    source_path: Path = MAPS_PATH,
    refresh_manifest_path: Path = WAREHOUSE_REFRESH_MANIFEST_PATH,
    snapshot_root: Path = SOURCE_SNAPSHOT_ROOT,
) -> dict[str, Any]:
    """Freeze exact model input plus its non-authorizing warehouse provenance."""

    source_sha256 = _sha256(source_path)
    try:
        refresh_manifest = json.loads(refresh_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("warehouse refresh manifest could not be parsed") from exc
    if not isinstance(refresh_manifest, dict):
        raise ValueError("warehouse refresh manifest root must be an object")
    if refresh_manifest.get("schema_version") != "scryglass:warehouse-refresh-manifest:v2":
        raise ValueError("warehouse refresh manifest schema is not supported")
    claimed = refresh_manifest.get("manifest_canonical_sha256")
    unsigned_refresh = dict(refresh_manifest)
    unsigned_refresh.pop("manifest_canonical_sha256", None)
    if not isinstance(claimed, str) or claimed != _canonical_sha256(unsigned_refresh):
        raise ValueError("warehouse refresh manifest canonical hash mismatch")
    maps_record = ((refresh_manifest.get("outputs") or {}).get("maps") or {})
    if maps_record.get("raw_sha256") != source_sha256:
        raise ValueError("warehouse refresh manifest does not bind the maps source")
    authority = refresh_manifest.get("authority") or {}
    if any(
        authority.get(name) is not False
        for name in (
            "model_validation_authority",
            "probability_authority",
            "recommendation_authority",
            "betting_authority",
        )
    ):
        raise ValueError("warehouse refresh manifest exceeds its authority ceiling")

    package = snapshot_root / source_sha256
    snapshot_maps = package / "maps.parquet"
    snapshot_refresh = package / "warehouse-refresh-manifest.json"
    _copy_content_addressed(source_path, snapshot_maps, source_sha256)
    refresh_sha256 = _sha256(refresh_manifest_path)
    _copy_content_addressed(refresh_manifest_path, snapshot_refresh, refresh_sha256)
    manifest = {
        "schema_version": "scryglass:live-total-kills-source-snapshot:v1",
        "created_from_refresh_at_utc": refresh_manifest.get("refreshed_at"),
        "source": {
            "locator": _root_locator(snapshot_maps),
            "bytes": int(snapshot_maps.stat().st_size),
            "raw_sha256": source_sha256,
        },
        "warehouse_refresh_manifest": {
            "locator": _root_locator(snapshot_refresh),
            "bytes": int(snapshot_refresh.stat().st_size),
            "raw_sha256": refresh_sha256,
            "canonical_sha256": claimed,
        },
        "authority": {
            "replayable_source_provenance": True,
            "model_validation_authority": False,
            "probability_authority": False,
            "recommendation_authority": False,
            "betting_authority": False,
        },
        "claim_ceiling": (
            "This immutable package binds model input bytes and warehouse provenance "
            "only; it does not validate a model, probability, recommendation, or wager."
        ),
    }
    manifest["snapshot_canonical_sha256"] = _canonical_sha256(manifest)
    snapshot_manifest = package / "snapshot-manifest.json"
    manifest_raw_sha256 = _write_json_idempotent_no_clobber(snapshot_manifest, manifest)
    return {
        "maps_path": snapshot_maps,
        "maps_raw_sha256": source_sha256,
        "manifest_path": snapshot_manifest,
        "manifest_raw_sha256": manifest_raw_sha256,
        "manifest_canonical_sha256": manifest["snapshot_canonical_sha256"],
    }


def validate_source_snapshot_manifest(
    source_path: Path, manifest_path: Path
) -> dict[str, Any]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("source snapshot manifest could not be parsed") from exc
    if not isinstance(manifest, dict):
        raise ValueError("source snapshot manifest root must be an object")
    if manifest.get("schema_version") != "scryglass:live-total-kills-source-snapshot:v1":
        raise ValueError("source snapshot manifest schema is not supported")
    claimed = manifest.get("snapshot_canonical_sha256")
    unsigned = dict(manifest)
    unsigned.pop("snapshot_canonical_sha256", None)
    if not isinstance(claimed, str) or claimed != _canonical_sha256(unsigned):
        raise ValueError("source snapshot manifest canonical hash mismatch")
    source = manifest.get("source") or {}
    if source.get("raw_sha256") != _sha256(source_path):
        raise ValueError("source snapshot manifest does not bind model source bytes")
    authority = manifest.get("authority") or {}
    if any(
        authority.get(name) is not False
        for name in (
            "model_validation_authority",
            "probability_authority",
            "recommendation_authority",
            "betting_authority",
        )
    ):
        raise ValueError("source snapshot manifest exceeds its authority ceiling")
    return manifest


def write_artifact_no_clobber(path: Path, payload: dict[str, Any]) -> str:
    """Publish one immutable artifact without replacing an existing file."""

    encoded = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to replace existing artifact: {path}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return hashlib.sha256(encoded).hexdigest()


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_patch(value: Any) -> str:
    text = str(value or "").strip()
    parts = text.split(".")
    if len(parts) == 2 and len(parts[0]) == 2 and len(parts[1]) == 1:
        return f"{parts[0]}.{parts[1]}0"
    return text


def _series_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            _parse_time(row["date"]).strftime("%Y-%m-%d"),
            row["league"],
            *sorted((row["blue_team"], row["red_team"])),
        ]
    )


def load_maps(path: Path = MAPS_PATH) -> list[dict[str, Any]]:
    wanted = [
        "game_uid",
        "date",
        "league",
        "patch",
        "blue_team",
        "red_team",
        "total_kills",
        *[f"blue_pick{i}" for i in range(1, 6)],
        *[f"red_pick{i}" for i in range(1, 6)],
        *[
            f"blue_{metric}at{checkpoint}"
            for checkpoint in CHECKPOINTS
            for metric in ("kills", "opp_kills", "golddiff")
        ],
    ]
    frame = pd.read_parquet(path, columns=wanted)
    frame = frame[frame["league"].astype(str).str.upper().isin(LEAGUES)].copy()
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        required = [record.get("date"), record.get("total_kills")]
        if any(pd.isna(value) for value in required):
            continue
        champions = []
        for side in ("blue", "red"):
            for index in range(1, 6):
                value = record.get(f"{side}_pick{index}")
                champions.append("" if pd.isna(value) else str(value).strip())
        if any(not champion for champion in champions) or len(set(champions)) != 10:
            continue
        checkpoints: dict[str, dict[str, float]] = {}
        for checkpoint in CHECKPOINTS:
            kills = record.get(f"blue_killsat{checkpoint}")
            opposing_kills = record.get(f"blue_opp_killsat{checkpoint}")
            gold_difference = record.get(f"blue_golddiffat{checkpoint}")
            if any(pd.isna(value) for value in (kills, opposing_kills, gold_difference)):
                continue
            current_kills = float(kills) + float(opposing_kills)
            if current_kills > float(record["total_kills"]):
                continue
            checkpoints[str(checkpoint)] = {
                "current_kills": current_kills,
                "gold_difference": float(gold_difference),
            }
        if not checkpoints:
            continue
        date = pd.Timestamp(record["date"])
        if date.tzinfo is not None:
            date = date.tz_convert("UTC").tz_localize(None)
        row = {
            "game_id": str(record["game_uid"]),
            "date": date.isoformat(),
            "league": str(record["league"]).upper(),
            "patch": _normalize_patch(record.get("patch")),
            "blue_team": str(record.get("blue_team") or "").strip(),
            "red_team": str(record.get("red_team") or "").strip(),
            "champions": sorted(champions),
            "total_kills": float(record["total_kills"]),
            "checkpoints": checkpoints,
        }
        if row["blue_team"] and row["red_team"]:
            row["series_id"] = _series_key(row)
            rows.append(row)
    if not rows:
        raise ValueError(f"no complete live total-kills rows in {path}")
    return rows


def split_series(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_series[row["series_id"]].append(row)
    ordered = sorted(
        by_series.items(),
        key=lambda item: (
            max(_parse_time(row["date"]) for row in item[1]),
            item[0],
        ),
    )
    if len(ordered) < 20:
        raise ValueError("at least 20 series are required")
    n = len(ordered)
    train_end = int(n * SPLIT_FRACTIONS[0])
    selection_end = int(n * sum(SPLIT_FRACTIONS[:2]))
    calibration_end = int(n * sum(SPLIT_FRACTIONS[:3]))

    def flatten(items: list[tuple[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
        return [row for _, series_rows in items for row in series_rows]

    return {
        "train": flatten(ordered[:train_end]),
        "selection": flatten(ordered[train_end:selection_end]),
        "calibration": flatten(ordered[selection_end:calibration_end]),
        "test": flatten(ordered[calibration_end:]),
    }


def attach_pregame_priors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create deployment-realistic priors using only already completed series."""
    by_series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_series[row["series_id"]].append(row)
    ordered = sorted(
        by_series.items(),
        key=lambda item: (
            max(_parse_time(row["date"]) for row in item[1]),
            item[0],
        ),
    )
    global_history: list[float] = []
    league_history: dict[str, list[float]] = defaultdict(list)
    team_history: dict[str, list[float]] = defaultdict(list)
    pair_history: dict[tuple[str, str], list[float]] = defaultdict(list)
    output: list[dict[str, Any]] = []
    for _, series_rows in ordered:
        enriched = []
        for row in series_rows:
            league_values = league_history[row["league"]]
            fallback = median(league_values or global_history or [28.0])
            team_values = [
                team_history[row["blue_team"]],
                team_history[row["red_team"]],
            ]
            team_medians = [median(values) if values else fallback for values in team_values]
            pair = tuple(sorted((row["blue_team"], row["red_team"])))
            h2h_values = pair_history[pair]
            enriched.append(
                {
                    **row,
                    "team_pace_median": sum(team_medians) / 2.0,
                    "h2h_median": median(h2h_values) if h2h_values else fallback,
                    "h2h_missing": 0.0 if h2h_values else 1.0,
                    "team_history_min_n": min(len(values) for values in team_values),
                    "h2h_n": len(h2h_values),
                }
            )
        output.extend(enriched)
        for row in series_rows:
            total = float(row["total_kills"])
            global_history.append(total)
            league_history[row["league"]].append(total)
            team_history[row["blue_team"]].append(total)
            team_history[row["red_team"]].append(total)
            pair_history[tuple(sorted((row["blue_team"], row["red_team"])))].append(total)
    return output


def expand_checkpoints(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        for checkpoint_text, state in row["checkpoints"].items():
            checkpoint = int(checkpoint_text)
            current_kills = float(state["current_kills"])
            examples.append(
                {
                    **{key: value for key, value in row.items() if key != "checkpoints"},
                    "checkpoint": checkpoint,
                    "current_kills": current_kills,
                    "gold_difference": float(state["gold_difference"]),
                    "remaining_kills": float(row["total_kills"]) - current_kills,
                }
            )
    return examples


def champion_vocabulary(rows: list[dict[str, Any]], minimum_games: int = 20) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        for champion in set(row["champions"]):
            counts[champion] += 1
    return sorted(champion for champion, count in counts.items() if count >= minimum_games)


def _raw_features(
    example: dict[str, Any],
    families: tuple[str, ...],
    champions: list[str],
) -> dict[str, float]:
    features = {
        f"checkpoint:{checkpoint}": float(example["checkpoint"] == checkpoint)
        for checkpoint in CHECKPOINTS
    }
    if "league" in families:
        features.update(
            {
                f"league:{league}": float(example["league"] == league)
                for league in LEAGUES
            }
        )
    if "team_pace" in families:
        features["team_pace_median"] = float(example["team_pace_median"])
    if "head_to_head" in families:
        features["h2h_median"] = float(example["h2h_median"])
        features["h2h_missing"] = float(example["h2h_missing"])
    if "current_kills" in families:
        features["current_kills"] = float(example["current_kills"])
    if "gold_difference" in families:
        features["absolute_gold_difference"] = abs(float(example["gold_difference"]))
    present = set(example["champions"])
    if "completed_draft" in families or "champion_scaling" in families:
        features.update(
            {f"champion:{champion}": float(champion in present) for champion in champions}
        )
    if "champion_scaling" in families:
        scaled_time = (float(example["checkpoint"]) - 17.5) / 7.5
        features.update(
            {
                f"champion_time:{champion}": (
                    scaled_time if champion in present else 0.0
                )
                for champion in champions
            }
        )
    return features


def fit_ridge(
    examples: list[dict[str, Any]],
    families: tuple[str, ...],
    champions: list[str],
    ridge_lambda: float = RIDGE_LAMBDA,
) -> dict[str, Any]:
    raw = [_raw_features(example, families, champions) for example in examples]
    feature_names = sorted(
        name
        for name in raw[0]
        if any(abs(float(row[name])) > 1e-12 for row in raw)
    )
    numeric = {
        "team_pace_median",
        "h2h_median",
        "current_kills",
        "absolute_gold_difference",
    }
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in feature_names:
        values = np.asarray([row[name] for row in raw], dtype=float)
        if name in numeric:
            centers[name] = float(values.mean())
            scale = float(values.std())
            scales[name] = scale if scale > 1e-9 else 1.0
        else:
            centers[name] = 0.0
            scales[name] = 1.0
    matrix = np.asarray(
        [
            [1.0]
            + [
                (row[name] - centers[name]) / scales[name]
                for name in feature_names
            ]
            for row in raw
        ],
        dtype=float,
    )
    target = np.asarray([example["remaining_kills"] for example in examples], dtype=float)
    penalty = np.eye(matrix.shape[1]) * ridge_lambda
    penalty[0, 0] = 0.0
    cross_product = np.einsum("ni,nj->ij", matrix, matrix)
    cross_target = np.einsum("ni,n->i", matrix, target)
    coefficients = np.linalg.solve(cross_product + penalty, cross_target)
    return {
        "families": list(families),
        "champions": champions,
        "feature_names": feature_names,
        "centers": centers,
        "scales": scales,
        "coefficients": [round(float(value), 10) for value in coefficients],
        "ridge_lambda": ridge_lambda,
    }


def predict_remaining(model: dict[str, Any], example: dict[str, Any]) -> float:
    raw = _raw_features(
        example,
        tuple(model["families"]),
        list(model["champions"]),
    )
    vector = [1.0] + [
        (raw[name] - float(model["centers"][name]))
        / float(model["scales"][name])
        for name in model["feature_names"]
    ]
    value = sum(
        coefficient * feature
        for coefficient, feature in zip(model["coefficients"], vector)
    )
    return max(0.0, float(value))


def _metrics(model: dict[str, Any], examples: list[dict[str, Any]]) -> dict[str, float]:
    errors = [
        predict_remaining(model, example) - float(example["remaining_kills"])
        for example in examples
    ]
    return {
        "n": len(errors),
        "rmse": round(math.sqrt(sum(error * error for error in errors) / len(errors)), 4),
        "mae": round(sum(abs(error) for error in errors) / len(errors), 4),
    }


def _group_mean_rmse(
    train: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> float:
    global_mean = sum(float(row["remaining_kills"]) for row in train) / len(train)
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in train:
        groups[tuple(row[field] for field in fields)].append(float(row["remaining_kills"]))
    means = {key: sum(values) / len(values) for key, values in groups.items()}
    errors = [
        means.get(tuple(row[field] for field in fields), global_mean)
        - float(row["remaining_kills"])
        for row in evaluation
    ]
    return round(math.sqrt(sum(error * error for error in errors) / len(errors)), 4)


def select_families(
    train: list[dict[str, Any]],
    selection: list[dict[str, Any]],
    champions: list[str],
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    selected: tuple[str, ...] = ()
    model = fit_ridge(train, selected, champions)
    current_rmse = _metrics(model, selection)["rmse"]
    constant_rmse = _group_mean_rmse(train, selection, ())
    checkpoint_rmse = _group_mean_rmse(train, selection, ("checkpoint",))
    league_checkpoint_rmse = _group_mean_rmse(
        train, selection, ("league", "checkpoint")
    )
    report = [
        {
            "candidate": "constant_baseline",
            "selection_rmse": constant_rmse,
            "accepted": True,
        },
        {
            "candidate": "time_checkpoint_strata",
            "selection_rmse": checkpoint_rmse,
            "relative_rmse_improvement": round(
                (constant_rmse - checkpoint_rmse) / constant_rmse, 6
            ),
            "accepted": checkpoint_rmse < constant_rmse,
        },
        {
            "candidate": "league_within_checkpoint_strata",
            "selection_rmse": league_checkpoint_rmse,
            "relative_rmse_improvement": round(
                (checkpoint_rmse - league_checkpoint_rmse) / checkpoint_rmse, 6
            ),
            "accepted": league_checkpoint_rmse < checkpoint_rmse,
        },
        {
            "candidate": "checkpoint_ridge_baseline",
            "families": [],
            "selection_rmse": current_rmse,
            "accepted": True,
        }
    ]
    for family in FAMILY_ORDER:
        additions = (
            ("completed_draft", "champion_scaling")
            if family == "champion_scaling" and "completed_draft" not in selected
            else (family,)
        )
        candidate_families = tuple(dict.fromkeys((*selected, *additions)))
        candidate_model = fit_ridge(train, candidate_families, champions)
        candidate_rmse = _metrics(candidate_model, selection)["rmse"]
        improvement = (current_rmse - candidate_rmse) / current_rmse
        accepted = improvement >= MIN_SELECTION_RMSE_IMPROVEMENT
        report.append(
            {
                "candidate": family,
                "families": list(candidate_families),
                "selection_rmse": candidate_rmse,
                "relative_rmse_improvement": round(improvement, 6),
                "minimum_improvement": MIN_SELECTION_RMSE_IMPROVEMENT,
                "accepted": accepted,
            }
        )
        if accepted:
            selected = candidate_families
            current_rmse = candidate_rmse
    return selected, report


def _nearest_rank(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def _cdf_report(calibration: list[float], test: list[float]) -> dict[str, Any]:
    points = []
    for nominal in (0.10, 0.25, 0.50, 0.75, 0.90):
        threshold = _nearest_rank(calibration, nominal)
        observed = sum(value <= threshold for value in test) / len(test)
        points.append(
            {
                "nominal": nominal,
                "observed": round(observed, 4),
                "absolute_error": round(abs(observed - nominal), 4),
            }
        )
    maximum = max(point["absolute_error"] for point in points)
    return {
        "points": points,
        "max_absolute_error": round(maximum, 4),
        "threshold": MAX_CDF_ERROR,
        "passed": maximum <= MAX_CDF_ERROR,
    }


def _series_cluster_cdf(
    clusters: list[list[float]],
    cutoff: float,
    *,
    minimum_series: int | None = None,
    confidence: float = DEPENDENCE_INTERVAL_CONFIDENCE,
) -> dict[str, Any]:
    """Estimate a residual CDF with arbitrary dependence inside each series.

    Each series is treated as one independent bounded observation whose value
    is its within-series share of residuals at or below ``cutoff``.  Weighted
    Hoeffding bounds use the series' map shares, so repeating maps inside the
    same series does not pretend to add independent evidence.
    """
    minimum_series = (
        MIN_CALIBRATION_SERIES if minimum_series is None else minimum_series
    )
    blockers: list[str] = []
    clean: list[list[float]] = []
    for cluster in clusters:
        if not isinstance(cluster, list) or not cluster:
            blockers.append("calibration_series_cluster_invalid")
            continue
        try:
            values = [float(value) for value in cluster]
        except (TypeError, ValueError):
            blockers.append("calibration_series_residual_invalid")
            continue
        if any(not math.isfinite(value) for value in values):
            blockers.append("calibration_series_residual_invalid")
            continue
        clean.append(values)
    if len(clean) < minimum_series:
        blockers.append(
            f"insufficient_calibration_series:{len(clean)}<{minimum_series}"
        )
    total_n = sum(len(cluster) for cluster in clean)
    if total_n <= 0:
        blockers.append("calibration_residuals_unavailable")
    if not 0.0 < confidence < 1.0:
        blockers.append("dependence_interval_confidence_invalid")
    if blockers:
        return {
            "status": "unavailable",
            "blockers": sorted(set(blockers)),
            "method": DEPENDENCE_INTERVAL_METHOD,
            "confidence": confidence,
            "calibration_series_n": len(clean),
            "calibration_games_n": total_n,
            "effective_series_n": None,
            "probability": None,
            "interval": None,
        }

    under_n = sum(
        sum(value <= cutoff for value in cluster) for cluster in clean
    )
    raw_probability = under_n / total_n
    probability = (under_n + 0.5) / (total_n + 1.0)
    weights = [len(cluster) / total_n for cluster in clean]
    squared_weight_sum = sum(weight * weight for weight in weights)
    alpha = 1.0 - confidence
    half_width = math.sqrt(
        0.5 * squared_weight_sum * math.log(2.0 / alpha)
    )
    lower = min(probability, max(0.0, raw_probability - half_width))
    upper = max(probability, min(1.0, raw_probability + half_width))
    return {
        "status": "available",
        "blockers": [],
        "method": DEPENDENCE_INTERVAL_METHOD,
        "confidence": confidence,
        "assumptions": [
            "independent_calibration_series",
            "arbitrary_within_series_dependence",
        ],
        "calibration_series_n": len(clean),
        "calibration_games_n": total_n,
        "effective_series_n": round(1.0 / squared_weight_sum, 6),
        "probability": round(probability, 6),
        "interval": [round(lower, 6), round(upper, 6)],
    }


def _split_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    series = sorted({row["series_id"] for row in rows})
    dates = [_parse_time(row["date"]) for row in rows]
    return {
        "games": len(rows),
        "series": len(series),
        "start": min(dates).isoformat(),
        "end": max(dates).isoformat(),
        "series_ids_sha256": hashlib.sha256("\n".join(series).encode()).hexdigest(),
    }


def _current_priors(rows: list[dict[str, Any]]) -> dict[str, Any]:
    league: dict[str, list[float]] = defaultdict(list)
    teams: dict[str, list[float]] = defaultdict(list)
    pairs: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        total = float(row["total_kills"])
        league[row["league"]].append(total)
        teams[row["blue_team"]].append(total)
        teams[row["red_team"]].append(total)
        pairs[tuple(sorted((row["blue_team"], row["red_team"])))].append(total)
    return {
        "league_median": {key: median(values) for key, values in sorted(league.items())},
        "teams": {
            key: {"n": len(values), "median": median(values)}
            for key, values in sorted(teams.items())
        },
        "pairs": {
            "|".join(key): {"n": len(values), "median": median(values)}
            for key, values in sorted(pairs.items())
        },
    }


def build_artifact(
    rows: list[dict[str, Any]],
    *,
    source_path: Path,
    built_at: str,
    source_manifest_path: Path | None = None,
) -> dict[str, Any]:
    enriched = attach_pregame_priors(rows)
    split = split_series(enriched)
    examples = {name: expand_checkpoints(part) for name, part in split.items()}
    champions = champion_vocabulary(split["train"])
    selected, ablations = select_families(
        examples["train"], examples["selection"], champions
    )
    fit_rows = examples["train"] + examples["selection"]
    model = fit_ridge(fit_rows, selected, champions)
    baseline = fit_ridge(fit_rows, (), champions)

    calibration_residuals: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    calibration_residual_clusters: dict[
        str, dict[str, dict[str, list[float]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for example in examples["calibration"]:
        residual = float(example["remaining_kills"]) - predict_remaining(model, example)
        checkpoint = str(example["checkpoint"])
        league = example["league"]
        calibration_residuals[checkpoint][league].append(residual)
        series_sha256 = hashlib.sha256(
            str(example["series_id"]).encode("utf-8")
        ).hexdigest()
        calibration_residual_clusters[checkpoint][league][series_sha256].append(
            residual
        )

    windows: dict[str, dict[str, Any]] = defaultdict(dict)
    patch_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    for checkpoint in CHECKPOINTS:
        for league in LEAGUES:
            test_rows = [
                example
                for example in examples["test"]
                if example["checkpoint"] == checkpoint and example["league"] == league
            ]
            calibration = calibration_residuals[str(checkpoint)][league]
            calibration_series_n = len(
                calibration_residual_clusters[str(checkpoint)][league]
            )
            if not test_rows:
                continue
            selected_metrics = _metrics(model, test_rows)
            baseline_metrics = _metrics(baseline, test_rows)
            test_residuals = [
                float(example["remaining_kills"]) - predict_remaining(model, example)
                for example in test_rows
            ]
            cdf = (
                _cdf_report(calibration, test_residuals)
                if calibration and test_residuals
                else {"passed": False, "max_absolute_error": None, "points": []}
            )
            supported = (
                len(calibration) >= MIN_CALIBRATION_GAMES
                and calibration_series_n >= MIN_CALIBRATION_SERIES
                and len(test_rows) >= MIN_TEST_GAMES
                and selected_metrics["rmse"] <= baseline_metrics["rmse"]
                and cdf["passed"]
            )
            blockers = []
            if len(calibration) < MIN_CALIBRATION_GAMES:
                blockers.append("insufficient_calibration_games")
            if calibration_series_n < MIN_CALIBRATION_SERIES:
                blockers.append("insufficient_calibration_series")
            if len(test_rows) < MIN_TEST_GAMES:
                blockers.append("insufficient_test_games")
            if selected_metrics["rmse"] > baseline_metrics["rmse"]:
                blockers.append("heldout_rmse_does_not_beat_checkpoint_baseline")
            if not cdf["passed"]:
                blockers.append("heldout_cdf_calibration_failed")
            windows[str(checkpoint)][league] = {
                "status": "supported" if supported else "unavailable",
                "calibration_n": len(calibration),
                "calibration_series_n": calibration_series_n,
                "test_n": len(test_rows),
                "model": selected_metrics,
                "baseline": baseline_metrics,
                "cdf_calibration": cdf,
                "blockers": blockers,
            }
            for example in test_rows:
                patch_counts[str(checkpoint)][league][example["patch"]] += 1

    cutoffs = {
        league: max(
            _parse_time(row["date"]) for row in rows if row["league"] == league
        ).isoformat()
        for league in LEAGUES
        if any(row["league"] == league for row in rows)
    }
    source_record: dict[str, Any] = {
        "path": _root_locator(source_path),
        "bytes": source_path.stat().st_size,
        "sha256": _sha256(source_path),
    }
    if source_manifest_path is not None:
        validate_source_snapshot_manifest(source_path, source_manifest_path)
        source_record["snapshot_manifest"] = {
            "path": _root_locator(source_manifest_path),
            "bytes": source_manifest_path.stat().st_size,
            "sha256": _sha256(source_manifest_path),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "built_at": _parse_time(built_at).isoformat(),
            "source": source_record,
            "games": len(rows),
            "series": len({row["series_id"] for row in rows}),
            "usable_games_by_league": {
                league: sum(row["league"] == league for row in rows)
                for league in LEAGUES
                if any(row["league"] == league for row in rows)
            },
            "data_cutoff_by_league": cutoffs,
            "unavailable_leagues": {
                league: "no complete draft plus checkpoint kills and gold rows"
                for league in LEAGUES
                if league not in cutoffs
            },
            "checkpoints_observed": list(CHECKPOINTS),
            "interpolation_authorized": False,
        },
        "protocol": {
            "split": "chronological_series_train_selection_calibration_test",
            "fractions": list(SPLIT_FRACTIONS),
            "preprocessing": "training_only",
            "pregame_priors": "prequential_prior_series_only",
            "target": "remaining_kills_after_checkpoint",
            "selection_rule": "forward_family_addition_on_selection_rmse",
            "calibration": (
                "league_checkpoint_empirical_residual_cdf_with_"
                "series_cluster_weighted_hoeffding_95_interval"
            ),
            "dependence_assumption": (
                "independent_series_with_arbitrary_within_series_dependence"
            ),
            "splits": {name: _split_manifest(part) for name, part in split.items()},
        },
        "feature_selection": {
            "selected_families": list(selected),
            "ablations": ablations,
            "objectives": {
                "status": "unavailable",
                "reason": "no leakage-safe checkpoint objective counts or timestamps in source",
            },
        },
        "model": model,
        "baseline_model": baseline,
        "calibration_residuals": {
            checkpoint: {
                league: [round(value, 6) for value in sorted(values)]
                for league, values in sorted(by_league.items())
            }
            for checkpoint, by_league in sorted(calibration_residuals.items())
        },
        "calibration_residual_clusters": {
            checkpoint: {
                league: {
                    "series_n": len(by_series),
                    "games_n": sum(len(values) for values in by_series.values()),
                    "clusters": [
                        {
                            "series_id_sha256": series_sha256,
                            "residuals": [round(value, 6) for value in sorted(values)],
                        }
                        for series_sha256, values in sorted(by_series.items())
                    ],
                }
                for league, by_series in sorted(by_league.items())
            }
            for checkpoint, by_league in sorted(
                calibration_residual_clusters.items()
            )
        },
        "windows": {key: dict(value) for key, value in sorted(windows.items())},
        "test_patch_counts": {
            checkpoint: {
                league: dict(sorted(counts.items()))
                for league, counts in sorted(by_league.items())
            }
            for checkpoint, by_league in sorted(patch_counts.items())
        },
        "runtime_priors": _current_priors(rows),
        "authority": {
            "validated_minutes_are_exact_checkpoints_only": True,
            "supported_windows": [
                {"minute": int(checkpoint), "league": league}
                for checkpoint, by_league in sorted(windows.items())
                for league, report in sorted(by_league.items())
                if report["status"] == "supported"
            ],
            "content_addressing_confers_authority": False,
            "betting_decision_authorized": False,
            "dependence_interval": {
                "status": "development_only",
                "method": DEPENDENCE_INTERVAL_METHOD,
                "confidence": DEPENDENCE_INTERVAL_CONFIDENCE,
                "requires_independent_market_authority": True,
            },
        },
    }


def _runtime_example(
    payload: dict[str, Any],
    *,
    league: str,
    blue_team: str,
    red_team: str,
    champions: list[str],
    minute: int,
    current_kills: int,
    gold_difference: float,
) -> dict[str, Any]:
    priors = payload["runtime_priors"]
    league_median = float(priors["league_median"][league])
    blue = priors["teams"].get(blue_team)
    red = priors["teams"].get(red_team)
    pair = priors["pairs"].get("|".join(sorted((blue_team, red_team))))
    return {
        "league": league,
        "checkpoint": minute,
        "current_kills": current_kills,
        "gold_difference": gold_difference,
        "champions": champions,
        "team_pace_median": (
            float(blue["median"] if blue else league_median)
            + float(red["median"] if red else league_median)
        )
        / 2.0,
        "h2h_median": float(pair["median"] if pair else league_median),
        "h2h_missing": 0.0 if pair else 1.0,
    }


def runtime_eligibility(
    payload: dict[str, Any],
    *,
    league: str,
    blue_team: str,
    red_team: str,
    champions: list[str],
    minute: float,
    current_kills: int | None,
    gold_difference: float | None,
    patch: str | None,
    as_of: datetime,
) -> dict[str, Any]:
    blockers: list[str] = []
    exact_minute = int(minute) if float(minute).is_integer() else None
    if exact_minute not in CHECKPOINTS:
        blockers.append(f"minute_not_validated:{minute:g}")
    window = (
        ((payload.get("windows") or {}).get(str(exact_minute)) or {}).get(league)
        if exact_minute is not None
        else None
    )
    if not window or window.get("status") != "supported":
        blockers.append(f"league_checkpoint_not_supported:{league}:{minute:g}")
    cutoff_text = (payload.get("meta") or {}).get("data_cutoff_by_league", {}).get(league)
    age_days = None
    if not cutoff_text:
        blockers.append(f"league_data_cutoff_missing:{league}")
    else:
        age_days = (
            as_of.astimezone(timezone.utc) - _parse_time(cutoff_text)
        ).total_seconds() / 86400.0
        if age_days > FRESHNESS_LIMIT_DAYS:
            blockers.append("data_stale")
        elif age_days < 0:
            blockers.append("as_of_precedes_data_cutoff")
    patch_text = _normalize_patch(patch)
    if not patch_text:
        blockers.append("competition_patch_unverified")
    elif exact_minute is not None:
        patch_n = int(
            (
                ((payload.get("test_patch_counts") or {}).get(str(exact_minute)) or {})
                .get(league, {})
            ).get(patch_text, 0)
        )
        if patch_n < MIN_PATCH_TEST_GAMES:
            blockers.append(f"exact_patch_holdout_unavailable:{patch_text}")
    families = set((payload.get("model") or {}).get("families") or [])
    if current_kills is None:
        blockers.append("current_kills_missing")
    elif current_kills < 0:
        blockers.append("current_kills_invalid")
    if "gold_difference" in families and gold_difference is None:
        blockers.append("gold_difference_missing")
    elif gold_difference is not None and not math.isfinite(float(gold_difference)):
        blockers.append("gold_difference_invalid")
    known_champions = set((payload.get("model") or {}).get("champions") or [])
    if {"completed_draft", "champion_scaling"} & families:
        if len(champions) != 10 or len(set(champions)) != 10:
            blockers.append("completed_draft_identity_invalid")
        unknown = sorted(set(champions) - known_champions)
        if unknown:
            blockers.append("unknown_champions:" + ",".join(unknown))
    else:
        unknown = []
    known_teams = set((payload.get("runtime_priors") or {}).get("teams") or {})
    if {"team_pace", "head_to_head"} & families:
        if blue_team not in known_teams or red_team not in known_teams:
            blockers.append("team_identity_unavailable")
    return {
        "status": "supported" if not blockers else "unavailable",
        "blockers": blockers,
        "minute": minute,
        "league": league,
        "patch": patch_text or None,
        "data_age_days": round(age_days, 3) if age_days is not None else None,
        "freshness_limit_days": FRESHNESS_LIMIT_DAYS,
        "unknown_champions": unknown,
    }


def price_live_totals(
    payload: dict[str, Any],
    *,
    league: str,
    blue_team: str,
    red_team: str,
    champions: list[str],
    minute: float,
    current_kills: int | None,
    gold_difference: float | None,
    patch: str | None,
    as_of: datetime,
    lines: list[dict[str, Any]],
) -> dict[str, Any]:
    eligibility = runtime_eligibility(
        payload,
        league=league,
        blue_team=blue_team,
        red_team=red_team,
        champions=champions,
        minute=minute,
        current_kills=current_kills,
        gold_difference=gold_difference,
        patch=patch,
        as_of=as_of,
    )
    withheld_lines = [
        {
            "line": float(market["line"]),
            "under_probability": None,
            "over_probability": None,
            "under_probability_interval": None,
            "over_probability_interval": None,
            "under_edge_pp": None,
            "under_expected_return": None,
            "classification": "WITHHELD",
        }
        for market in lines
    ]
    if eligibility["status"] != "supported":
        return {
            "eligibility": eligibility,
            "projected_mean": None,
            "lines": withheld_lines,
        }
    checkpoint = int(minute)
    example = _runtime_example(
        payload,
        league=league,
        blue_team=blue_team,
        red_team=red_team,
        champions=champions,
        minute=checkpoint,
        current_kills=int(current_kills),
        gold_difference=float(gold_difference),
    )
    remaining = predict_remaining(payload["model"], example)
    projected_mean = int(current_kills) + remaining
    residual_values = (
        ((payload.get("calibration_residuals") or {}).get(str(checkpoint)) or {})
        .get(league)
    )
    try:
        residuals = (
            sorted(float(value) for value in residual_values)
            if isinstance(residual_values, list)
            else []
        )
    except (TypeError, ValueError):
        residuals = []
    if not residuals or any(not math.isfinite(value) for value in residuals):
        unavailable = {
            **eligibility,
            "status": "unavailable",
            "blockers": sorted(
                set(
                    [
                        *eligibility.get("blockers", []),
                        "calibration_residuals_unavailable",
                    ]
                )
            ),
        }
        return {
            "eligibility": unavailable,
            "projected_mean": None,
            "uncertainty": {
                "status": "unavailable",
                "method": DEPENDENCE_INTERVAL_METHOD,
                "confidence": DEPENDENCE_INTERVAL_CONFIDENCE,
                "blockers": ["calibration_residuals_unavailable"],
            },
            "lines": withheld_lines,
        }
    cluster_entry = (
        ((payload.get("calibration_residual_clusters") or {}).get(str(checkpoint)) or {})
        .get(league)
    )
    cluster_blockers: list[str] = []
    clusters: list[list[float]] = []
    if not isinstance(cluster_entry, dict):
        cluster_blockers.append("series_cluster_calibration_missing")
    else:
        records = cluster_entry.get("clusters")
        if not isinstance(records, list) or not records:
            cluster_blockers.append("series_cluster_calibration_missing")
        else:
            seen: set[str] = set()
            for record in records:
                if not isinstance(record, dict):
                    cluster_blockers.append("series_cluster_record_invalid")
                    continue
                series_sha256 = record.get("series_id_sha256")
                values = record.get("residuals")
                if (
                    not isinstance(series_sha256, str)
                    or len(series_sha256) != 64
                    or any(character not in "0123456789abcdef" for character in series_sha256)
                    or series_sha256 in seen
                    or not isinstance(values, list)
                    or not values
                ):
                    cluster_blockers.append("series_cluster_record_invalid")
                    continue
                try:
                    normalized_values = [float(value) for value in values]
                except (TypeError, ValueError):
                    cluster_blockers.append("series_cluster_record_invalid")
                    continue
                if any(not math.isfinite(value) for value in normalized_values):
                    cluster_blockers.append("series_cluster_record_invalid")
                    continue
                seen.add(series_sha256)
                clusters.append(normalized_values)
            flattened = sorted(
                round(float(value), 6)
                for cluster in clusters
                for value in cluster
            )
            expected = sorted(round(float(value), 6) for value in residuals)
            if flattened != expected:
                cluster_blockers.append("series_cluster_residual_mismatch")
            if cluster_entry.get("series_n") != len(clusters):
                cluster_blockers.append("series_cluster_count_mismatch")
            if cluster_entry.get("games_n") != len(flattened):
                cluster_blockers.append("series_cluster_count_mismatch")
    priced = []
    line_uncertainty: list[dict[str, Any]] = []
    for market in lines:
        line = float(market["line"])
        if abs(line * 2 - round(line * 2)) > 1e-9 or line.is_integer():
            raise ValueError("live total-kills pricing requires half-kill lines")
        cutoff = math.floor(line) - projected_mean
        under_n = bisect.bisect_right(residuals, cutoff)
        under_probability = (under_n + 0.5) / (len(residuals) + 1.0)
        uncertainty = (
            {
                "status": "unavailable",
                "blockers": sorted(set(cluster_blockers)),
                "method": DEPENDENCE_INTERVAL_METHOD,
                "confidence": DEPENDENCE_INTERVAL_CONFIDENCE,
                "calibration_series_n": 0,
                "calibration_games_n": len(residuals),
                "effective_series_n": None,
                "probability": None,
                "interval": None,
            }
            if cluster_blockers
            else _series_cluster_cdf(clusters, cutoff)
        )
        interval = uncertainty.get("interval")
        if uncertainty.get("probability") is not None:
            under_probability = float(uncertainty["probability"])
        under_interval = list(interval) if interval is not None else None
        over_interval = (
            [round(1.0 - interval[1], 6), round(1.0 - interval[0], 6)]
            if interval is not None
            else None
        )
        line_uncertainty.append(uncertainty)
        priced.append(
            {
                "line": line,
                "under_probability": round(under_probability, 6),
                "over_probability": round(1.0 - under_probability, 6),
                "under_probability_interval": under_interval,
                "over_probability_interval": over_interval,
                "under_edge_pp": None,
                "under_expected_return": None,
                "classification": "WITHHELD",
                "claim_ceiling": "research_diagnostic_only",
                "uncertainty": uncertainty,
            }
        )
    return {
        "eligibility": eligibility,
        "projected_mean": round(projected_mean, 4),
        "uncertainty": {
            "status": (
                "available"
                if line_uncertainty
                and all(item.get("status") == "available" for item in line_uncertainty)
                else "unavailable"
            ),
            "method": DEPENDENCE_INTERVAL_METHOD,
            "confidence": DEPENDENCE_INTERVAL_CONFIDENCE,
            "blockers": sorted(
                {
                    blocker
                    for item in line_uncertainty
                    for blocker in item.get("blockers") or []
                }
            ),
        },
        "lines": priced,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--built-at", required=True)
    parser.add_argument("--source", type=Path, default=MAPS_PATH)
    parser.add_argument("--source-manifest", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()
    rows = load_maps(args.source)
    payload = build_artifact(
        rows,
        source_path=args.source,
        source_manifest_path=args.source_manifest,
        built_at=args.built_at,
    )
    raw_sha256 = write_artifact_no_clobber(args.out, payload)
    print(
        json.dumps(
            {
                "output": str(args.out),
                "raw_sha256": raw_sha256,
                "games": payload["meta"]["games"],
                "selected_families": payload["feature_selection"]["selected_families"],
                "supported_windows": payload["authority"]["supported_windows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
