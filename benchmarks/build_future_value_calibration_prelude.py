"""Build one source-bound chronological calibration prelude.

The prelude is a research-only artifact.  It fits each named rating variant
on the first whole-series training fold and records only that fold's held-out
raw logits.  The rows are suitable as strict-prior inputs to the later
three-fold evaluation.

The caller must provide the later evaluation's UTC start cutoff with
``--outer-evaluation-start``.  The prelude validation interval must end
strictly before that cutoff.  This keeps a prelude from entering the
evaluation interval by accident.

The command keeps producer artifacts outside the repository.  The receipt
binds their paths and hashes, the frozen source, the series crosswalk, and
the model producer code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
import pandas as pd

from benchmarks.future_value_four_variant_bundle import (
    _accepted_map_frame,
    _build_inner_fold_artifacts,
    _derive_inner_fold_spec,
)
from lol_kills.research.future_value_rating import (
    RATING_VARIANT_ORDER,
    _map_model_frame,
    bind_verified_leaguepedia_series_crosswalk,
    build_time_decayed_prior_player_form,
    chronological_whole_series_folds,
    fit_future_value_model,
    rating_variant_config_receipt,
)
from lol_kills.research.future_value_training import CALIBRATION_PRIOR_SCHEMA_VERSION
from lol_kills.v2.tierlists.accepted_census import identity_sha256


SCHEMA_VERSION = CALIBRATION_PRIOR_SCHEMA_VERSION
VARIANT_NAMES = tuple(variant.value for variant in RATING_VARIANT_ORDER)


class PreludeError(RuntimeError):
    """The calibration prelude cannot be built safely."""


def _utc_timestamp(value: object, label: str) -> pd.Timestamp:
    """Parse one explicit UTC timestamp used by the prelude contract."""

    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise PreludeError(f"{label} is invalid") from error
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise PreludeError(f"{label} must include a UTC timezone")
    return timestamp.tz_convert("UTC")


def _validate_outer_evaluation_cutoff(
    prelude_fold: Mapping[str, Any],
    *,
    outer_evaluation_start: object,
) -> pd.Timestamp:
    """Require the prelude validation interval to precede evaluation."""

    evaluation_start = _utc_timestamp(
        outer_evaluation_start,
        "outer evaluation start cutoff",
    )
    validation_end = _utc_timestamp(
        prelude_fold.get("validation_end"),
        "prelude validation end",
    )
    if validation_end >= evaluation_start:
        raise PreludeError(
            "prelude validation end must be strictly earlier than the outer "
            "evaluation start cutoff"
        )
    return evaluation_start


def _strict_prior_model_frame(
    model_frame: pd.DataFrame,
    *,
    outer_evaluation_start: object,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Return the model rows that exist before the outer evaluation starts."""

    evaluation_start = _utc_timestamp(
        outer_evaluation_start,
        "outer evaluation start cutoff",
    )
    dates = pd.to_datetime(model_frame["date"], errors="coerce", utc=True)
    if dates.isna().any():
        raise PreludeError("prelude model frame has an invalid game date")
    prior = model_frame.loc[dates.lt(evaluation_start)].copy()
    if prior.empty:
        raise PreludeError("prelude has no strict-prior model rows")
    if not bool(pd.to_datetime(prior["date"], utc=True).lt(evaluation_start).all()):
        raise PreludeError("prelude model frame is not strictly prior")
    return prior, evaluation_start


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_path(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise PreludeError(f"file is missing or unsafe: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha_path(path)}


def _write_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise PreludeError(f"output already exists: {path}")
    path.write_bytes(_canonical(value))
    return _file_record(path)


def _code_binding() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip() if result.returncode == 0 else ""
    if not commit or any(character not in "0123456789abcdef" for character in commit.lower()):
        raise PreludeError("prelude code commit cannot be bound")
    paths = (
        repo_root / "lol_kills/research/future_value_rating.py",
        repo_root / "lol_kills/research/future_value_training.py",
        repo_root / "lol_kills/research/future_value_uncertainty.py",
        Path(__file__).resolve(),
    )
    records = [_file_record(path) for path in paths]
    return {"commit": commit, "files": records}


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PreludeError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreludeError(f"{label} cannot be read: {path}") from error
    if not isinstance(value, dict):
        raise PreludeError(f"{label} is not an object")
    return value


def _empty_root(path: Path) -> Path:
    path = path.resolve()
    if path.exists() and path.is_symlink():
        raise PreludeError(f"producer output root is a symlink: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir() or path.is_symlink() or any(path.iterdir()):
        raise PreludeError(f"producer output root must be empty: {path}")
    return path


def _iso_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        normalized: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (np.generic,)):
                value = value.item()
            normalized[key] = value
        result.append(normalized)
    return result


def _record_frame(record: Mapping[str, Any]) -> pd.DataFrame:
    rows = record.get("rows")
    attrs = record.get("attrs")
    if not isinstance(rows, list) or not isinstance(attrs, Mapping):
        raise PreludeError("producer ledger record is incomplete")
    frame = pd.DataFrame(rows)
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame.attrs = dict(attrs)
    return frame


def _outer_as_inner(fold: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt an outer fold to the existing producer helper contract."""

    train_ids = tuple(sorted(str(value) for value in fold["train_game_ids"]))
    validation_ids = tuple(sorted(str(value) for value in fold["validation_game_ids"]))
    return {
        "fold": 1,
        "outer_fold": 1,
        "inner_fold": 0,
        "fold_window_end": str(fold["validation_start"]),
        "fit_window_end": str(fold["validation_start"]),
        "validation_start": str(fold["validation_start"]),
        "validation_end": str(fold["validation_end"]),
        "train_game_ids": list(train_ids),
        "validation_game_ids": list(validation_ids),
        "outer_train_game_ids": list(train_ids),
        "outer_train_identity_sha256": identity_sha256(train_ids),
        "outer_validation_game_ids": list(validation_ids),
        "outer_validation_identity_sha256": identity_sha256(validation_ids),
        "boundary_excluded_game_ids": [],
        "boundary_excluded_game_count": 0,
        "boundary_excluded_identity_sha256": identity_sha256(()),
        "overlap_audit": dict(fold.get("overlap_audit") or {}),
    }


def _build_producer_records(
    *,
    source_receipt_path: Path,
    source_receipt: Mapping[str, Any],
    maps: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    identity: pd.DataFrame,
    fold_number: int,
    fold: Mapping[str, Any],
    output_root: Path,
    series_by_game: Mapping[str, str],
    series_partition_source: str,
    series_partition_receipt_file_sha256: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    return _build_inner_fold_artifacts(
        source_receipt_path=source_receipt_path,
        source_receipt=source_receipt,
        maps=maps,
        players=players,
        teams=teams,
        identity=identity,
        fold_number=fold_number,
        inner_fold=fold,
        output_root=output_root,
        series_by_game=series_by_game,
        series_partition_source=series_partition_source,
        series_partition_receipt_file_sha256=series_partition_receipt_file_sha256,
    )


def _build_variant_rows(
    *,
    variant_key: str,
    source_receipt: Mapping[str, Any],
    source_receipt_sha256: str,
    crosswalk_receipt_file_sha256: str,
    model_frame: pd.DataFrame,
    players: pd.DataFrame,
    fold: Mapping[str, Any],
    outer_record: Mapping[str, Any],
    inner_record: Mapping[str, Any],
    artifact_root: Path,
    code_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_ids = tuple(sorted(str(value) for value in fold["train_game_ids"]))
    validation_ids = tuple(sorted(str(value) for value in fold["validation_game_ids"]))
    fold_ids = tuple(sorted((*train_ids, *validation_ids)))
    fold_frame = model_frame[model_frame["game_id"].astype(str).isin(fold_ids)].copy()
    if tuple(sorted(fold_frame["game_id"].astype(str))) != fold_ids:
        raise PreludeError("prelude fold source rows are incomplete")
    form = build_time_decayed_prior_player_form(fold_frame, players)
    outer_ledger = _record_frame(outer_record)
    inner_ledger = _record_frame(inner_record)
    variant = next(
        value for value in RATING_VARIANT_ORDER if value.value == variant_key
    )
    model, design = fit_future_value_model(
        fold_frame,
        form,
        train_game_ids=train_ids,
        fit_window_end=fold["validation_start"],
        source_receipt=source_receipt,
        crosswalk_receipt_file_sha256=crosswalk_receipt_file_sha256,
        verified_model_frame=fold_frame,
        variant=variant,
        feature_ledger=outer_ledger,
        inner_feature_ledger=inner_ledger,
    )
    validation = design[design["game_id"].astype(str).isin(validation_ids)].copy()
    raw_logit = model.predict_logit(validation)
    raw_probability = model.predict_probability(validation)
    target = validation["target"].astype(float)
    if not (raw_logit.notna() & raw_probability.notna() & target.notna()).all():
        raise PreludeError(f"{variant_key} prelude predictions are incomplete")
    rows: list[dict[str, Any]] = []
    for index in validation.index:
        logit = float(raw_logit.loc[index])
        probability = float(raw_probability.loc[index])
        support = validation.loc[index, "player_form_minimum_effective_support"]
        support_value = None if pd.isna(support) else float(support)
        row = {
            "game_id": str(validation.loc[index, "game_id"]),
            "series_id": str(validation.loc[index, "series_id"]),
            "date": pd.Timestamp(validation.loc[index, "date"])
            .tz_convert("UTC")
            .isoformat()
            .replace("+00:00", "Z"),
            "raw_logit": logit,
            "raw_probability": probability,
            "target": int(target.loc[index]),
        }
        if support_value is not None and np.isfinite(support_value):
            row["support"] = support_value
        rows.append(row)
    rows.sort(key=lambda row: (str(row["date"]), str(row["game_id"])))
    variant_root = artifact_root / variant_key
    variant_root.mkdir(parents=True, exist_ok=True)
    if any(variant_root.iterdir()):
        raise PreludeError(f"prelude variant artifact directory is not empty: {variant_root}")
    parameter_receipt = model.parameter_receipt()
    parameter_sha256 = str(parameter_receipt["parameter_sha256"])
    model_receipt_payload: dict[str, Any] = {
        "schema_version": "scryglass:future-value-calibration-prelude-model-receipt:v1",
        "source_receipt_sha256": source_receipt_sha256,
        "variant": variant_key,
        "fold": 0,
        "fit_window_end": model.fit_window_end,
        "fit_game_ids": list(model.fit_game_ids),
        "fit_game_identity_sha256": identity_sha256(model.fit_game_ids),
        "validation_game_count": len(validation_ids),
        "validation_game_identity_sha256": identity_sha256(validation_ids),
        "parameter_sha256": parameter_sha256,
        "code": dict(code_binding),
    }
    model_receipt_payload["receipt_sha256"] = _sha_bytes(_canonical(model_receipt_payload))
    model_receipt_record = _write_json(
        variant_root / "model-receipt.json", model_receipt_payload
    )
    model_artifact_payload: dict[str, Any] = {
        "schema_version": "scryglass:future-value-calibration-prelude-model-artifact:v1",
        "source_receipt_sha256": source_receipt_sha256,
        "variant": variant_key,
        "fold": 0,
        "parameter_sha256": parameter_sha256,
        "parameters": parameter_receipt,
        "code": dict(code_binding),
    }
    model_artifact_payload["artifact_sha256"] = _sha_bytes(_canonical(model_artifact_payload))
    model_artifact_record = _write_json(
        variant_root / "model-artifact.json", model_artifact_payload
    )
    prediction_ledger_payload: dict[str, Any] = {
        "schema_version": "scryglass:future-value-calibration-prelude-prediction-ledger:v1",
        "source_receipt_sha256": source_receipt_sha256,
        "variant": variant_key,
        "fold": 0,
        "rows": rows,
        "row_count": len(rows),
        "rows_sha256": _sha_bytes(_canonical(rows)),
    }
    prediction_ledger_payload["ledger_sha256"] = _sha_bytes(
        _canonical(prediction_ledger_payload)
    )
    prediction_ledger_record = _write_json(
        variant_root / "prediction-ledger.json", prediction_ledger_payload
    )
    model_binding = {
        "source_receipt_sha256": source_receipt_sha256,
        "variant": variant_key,
        "fit_window_end": model.fit_window_end,
        "fit_game_identity_sha256": identity_sha256(model.fit_game_ids),
        "validation_game_identity_sha256": identity_sha256(validation_ids),
        "parameter_sha256": parameter_sha256,
        "prediction_ledger_row_count": len(rows),
        "prediction_ledger_rows_sha256": _sha_bytes(_canonical(rows)),
        "model_receipt": model_receipt_record,
        "model_artifact": model_artifact_record,
        "prediction_ledger": prediction_ledger_record,
        "code": dict(code_binding),
    }
    payload = {
        "fold": 0,
        "train_end": str(fold["train_end"]),
        "validation_start": str(fold["validation_start"]),
        "validation_end": str(fold["validation_end"]),
        "source_receipt_sha256": source_receipt_sha256,
        "variant": variant_key,
        "out_of_sample": True,
        "whole_series": True,
        "train_game_count": len(train_ids),
        "train_game_identity_sha256": identity_sha256(train_ids),
        "validation_game_count": len(validation_ids),
        "validation_game_identity_sha256": identity_sha256(validation_ids),
        "rows": _iso_rows(rows),
        "row_count": len(rows),
        "game_identity_sha256": identity_sha256(tuple(row["game_id"] for row in rows)),
        "rows_sha256": _sha_bytes(_canonical(rows)),
        "model": {
            "parameter_sha256": parameter_sha256,
            "fit_game_count": len(model.fit_game_ids),
            "fit_game_identity_sha256": identity_sha256(model.fit_game_ids),
            "fit_window_end": model.fit_window_end,
            "variant_receipt": rating_variant_config_receipt(variant),
            "regularization_selection": dict(model.regularization_selection),
            "feature_ledger_binding": dict(model.feature_ledger_binding or {}),
        },
        "model_binding": model_binding,
    }
    evidence = {
        "variant": variant_key,
        "validation_rows": len(rows),
        "validation_identity_sha256": identity_sha256(validation_ids),
        "model_parameter_sha256": payload["model"]["parameter_sha256"],
        "selected_c": payload["model"]["regularization_selection"].get("selected_c"),
    }
    return payload, evidence


def build_prelude(
    *,
    source_root: Path,
    source_receipt_path: Path,
    crosswalk_path: Path,
    crosswalk_receipt_path: Path,
    crosswalk_receipt_file_sha256: str,
    producer_root: Path,
    outer_evaluation_start: str | pd.Timestamp,
    fold_count: int = 4,
) -> dict[str, Any]:
    source_receipt = _load_json(source_receipt_path, "source receipt")
    maps = pd.read_parquet(source_root / "maps.parquet")
    players = pd.read_parquet(source_root / "oe_player_games.parquet")
    teams = pd.read_parquet(source_root / "oe_team_games.parquet")
    accepted_maps = _accepted_map_frame(maps, source_receipt=source_receipt)
    bound_maps = bind_verified_leaguepedia_series_crosswalk(
        accepted_maps,
        crosswalk_path=crosswalk_path,
        receipt_path=crosswalk_receipt_path,
        source_receipt=source_receipt,
        expected_receipt_file_sha256=crosswalk_receipt_file_sha256,
    )
    model_frame = _map_model_frame(
        bound_maps,
        verified_source_receipt=source_receipt,
        verified_source_receipt_sha256=str(source_receipt["receipt_sha256"]),
        verified_crosswalk_receipt_file_sha256=crosswalk_receipt_file_sha256,
    )
    eligible_ids = tuple(sorted(str(value) for value in source_receipt["model_eligible_game_ids"]))
    model_frame = model_frame[model_frame["game_id"].astype(str).isin(eligible_ids)].copy()
    if tuple(sorted(model_frame["game_id"].astype(str))) != eligible_ids:
        raise PreludeError("prelude model frame does not match the accepted eligible census")
    prelude_model_frame, evaluation_start = _strict_prior_model_frame(
        model_frame,
        outer_evaluation_start=outer_evaluation_start,
    )
    folds = chronological_whole_series_folds(
        prelude_model_frame,
        n_folds=fold_count,
        verified_model_frame=prelude_model_frame,
    )
    if not folds or int(folds[0]["fold"]) != 1:
        raise PreludeError("prelude chronological fold is missing")
    prelude_fold = folds[0]
    validated_evaluation_start = _validate_outer_evaluation_cutoff(
        prelude_fold,
        outer_evaluation_start=evaluation_start,
    )
    identity = model_frame[["game_id", "date", "series_id"]].copy()
    series_by_game = dict(
        zip(identity["game_id"].astype(str), identity["series_id"].astype(str))
    )
    series_source = str(
        "verified_leaguepedia_series_crosswalk"
    )
    eligible_series = model_frame["series_id"].astype("string").str.strip()
    if eligible_series.isna().any() or not bool(
        eligible_series.str.startswith("leaguepedia:").all()
    ):
        raise PreludeError("prelude series partition contains a conservative proxy")
    producer_root = _empty_root(producer_root)
    outer_root = _empty_root(producer_root / "outer")
    inner_root = _empty_root(producer_root / "inner")
    model_root = _empty_root(producer_root / "models")
    code_binding = _code_binding()
    outer_spec = _outer_as_inner(prelude_fold)
    outer_records, outer_receipt = _build_producer_records(
        source_receipt_path=source_receipt_path,
        source_receipt=source_receipt,
        maps=accepted_maps,
        players=players,
        teams=teams,
        identity=identity,
        fold_number=0,
        fold=outer_spec,
        output_root=outer_root,
        series_by_game=series_by_game,
        series_partition_source=series_source,
        series_partition_receipt_file_sha256=crosswalk_receipt_file_sha256,
    )
    inner_spec = _derive_inner_fold_spec(
        prelude_model_frame,
        outer_fold=1,
        outer_train_ids=tuple(str(value) for value in prelude_fold["train_game_ids"]),
        outer_validation_ids=tuple(
            str(value) for value in prelude_fold["validation_game_ids"]
        ),
    )
    inner_records, inner_receipt = _build_producer_records(
        source_receipt_path=source_receipt_path,
        source_receipt=source_receipt,
        maps=accepted_maps,
        players=players,
        teams=teams,
        identity=identity,
        fold_number=1,
        fold=inner_spec,
        output_root=inner_root,
        series_by_game=series_by_game,
        series_partition_source=series_source,
        series_partition_receipt_file_sha256=crosswalk_receipt_file_sha256,
    )
    variants: dict[str, dict[str, Any]] = {}
    evidence: list[dict[str, Any]] = []
    for variant_key in VARIANT_NAMES:
        payload, detail = _build_variant_rows(
            variant_key=variant_key,
            source_receipt=source_receipt,
            source_receipt_sha256=str(source_receipt["receipt_sha256"]),
            crosswalk_receipt_file_sha256=crosswalk_receipt_file_sha256,
            model_frame=prelude_model_frame,
            players=players,
            fold=prelude_fold,
            outer_record=outer_records[variant_key],
            inner_record=inner_records[variant_key],
            artifact_root=model_root,
            code_binding=code_binding,
        )
        variants[variant_key] = {"folds": [payload]}
        evidence.append(detail)
    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "research_only",
        "source_receipt_sha256": str(source_receipt["receipt_sha256"]),
        "source_receipt_file": _file_record(source_receipt_path),
        "source": {
            "source_as_of": source_receipt["source_as_of"],
            "source_game_count": source_receipt["source_game_count"],
            "source_identity_sha256": source_receipt["source_identity_sha256"],
            "model_eligible_game_count": len(eligible_ids),
            "model_eligible_identity_sha256": identity_sha256(eligible_ids),
            "model_eligible_game_ids": list(eligible_ids),
            "source_root": str(source_root),
            "source_files": {
                label: _file_record(source_root / filename)
                for label, filename in (
                    ("maps", "maps.parquet"),
                    ("players", "oe_player_games.parquet"),
                    ("teams", "oe_team_games.parquet"),
                )
            },
        },
        "series_partition": {
            "artifact": _file_record(crosswalk_path),
            "receipt": _file_record(crosswalk_receipt_path),
            "expected_receipt_file_sha256": crosswalk_receipt_file_sha256,
            "source": series_source,
            "eligible_identity_sha256": identity_sha256(eligible_ids),
        },
        "fold_protocol": {
            "fold_count": int(fold_count),
            "prelude_fold": 0,
            "outer_evaluation_start": validated_evaluation_start.isoformat().replace(
                "+00:00", "Z"
            ),
            "validation_interval": {
                "train_end": str(prelude_fold["train_end"]),
                "validation_start": str(prelude_fold["validation_start"]),
                "validation_end": str(prelude_fold["validation_end"]),
            },
            "train_game_count": len(prelude_fold["train_game_ids"]),
            "train_identity_sha256": identity_sha256(prelude_fold["train_game_ids"]),
            "validation_game_count": len(prelude_fold["validation_game_ids"]),
            "validation_identity_sha256": identity_sha256(
                prelude_fold["validation_game_ids"]
            ),
            "train_series_count": len(prelude_fold["train_series_ids"]),
            "validation_series_count": len(prelude_fold["validation_series_ids"]),
            "overlap_audit": dict(prelude_fold["overlap_audit"]),
            "out_of_sample": True,
            "whole_series": True,
        },
        "producer_artifacts": {
            "root": str(producer_root),
            "outer_receipt": outer_receipt,
            "inner_receipt": inner_receipt,
            "files": [
                _file_record(path)
                for path in sorted(producer_root.rglob("*"))
                if path.is_file() and not path.is_symlink()
            ],
        },
        "variants": variants,
        "variant_configs": {
            variant.value: rating_variant_config_receipt(variant)
            for variant in RATING_VARIANT_ORDER
        },
        "evidence": evidence,
        "authority": {
            "research_only": True,
            "public_player_rating": False,
            "public_team_rating": False,
            "public_probability": False,
            "promotion": False,
            "deployment": False,
            "betting": False,
        },
    }
    output["receipt_sha256"] = _sha_bytes(_canonical(output))
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--crosswalk", required=True, type=Path)
    parser.add_argument("--crosswalk-receipt", required=True, type=Path)
    parser.add_argument("--crosswalk-receipt-file-sha256", required=True)
    parser.add_argument("--producer-root", required=True, type=Path)
    parser.add_argument(
        "--outer-evaluation-start",
        required=True,
        help="UTC ISO-8601 start cutoff for the later outer evaluation",
    )
    parser.add_argument("--fold-count", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.fold_count < 2:
        raise PreludeError("fold count must leave a training interval before validation")
    output_path = args.output.resolve()
    if output_path.exists() or output_path.is_symlink():
        raise PreludeError(f"output already exists: {output_path}")
    if not output_path.parent.is_dir() or output_path.parent.is_symlink():
        raise PreludeError(f"output parent is unsafe: {output_path.parent}")
    payload = build_prelude(
        source_root=args.source_root.resolve(),
        source_receipt_path=args.source_receipt.resolve(),
        crosswalk_path=args.crosswalk.resolve(),
        crosswalk_receipt_path=args.crosswalk_receipt.resolve(),
        crosswalk_receipt_file_sha256=str(args.crosswalk_receipt_file_sha256),
        producer_root=args.producer_root.resolve(),
        outer_evaluation_start=str(args.outer_evaluation_start),
        fold_count=int(args.fold_count),
    )
    output_path.write_bytes(_canonical(payload))
    print(
        json.dumps(
            {
                "output": str(output_path),
                "bytes": output_path.stat().st_size,
                "sha256": _sha_path(output_path),
                "receipt_sha256": payload["receipt_sha256"],
                "variants": list(payload["variants"]),
                "prelude_rows": {
                    key: len(value["folds"][0]["rows"])
                    for key, value in payload["variants"].items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
