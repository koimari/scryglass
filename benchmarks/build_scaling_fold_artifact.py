"""Build one producer-owned fold-local scaling artifact and receipt set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import pandas as pd

from lol_kills.research.atomized_rf_composite import build_scaling_feature_ledger
from lol_kills.research.future_value_rating import (
    SCALING_CURVE_SIGNED_MAP_FEATURES,
    _frame_game_ids,
    _map_model_frame,
    bind_verified_leaguepedia_series_crosswalk,
    build_rating_feature_producer_manifest,
    write_rating_feature_producer_receipt,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"input is missing or unsafe: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"input is not a JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"output already exists: {path}")
    path.write_text(
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _series_assignment_sha256(frame: pd.DataFrame) -> str:
    rows = sorted(
        (
            {"game_id": str(game_id), "series_id": str(series_id)}
            for game_id, series_id in zip(frame["game_id"], frame["series_id"])
        ),
        key=lambda row: row["game_id"],
    )
    if len(rows) != len({row["game_id"] for row in rows}):
        raise RuntimeError("series assignment contains duplicate game IDs")
    return hashlib.sha256(_canonical_bytes(rows)).hexdigest()


def _accepted_map_frame(
    maps: pd.DataFrame,
    *,
    source_receipt: dict[str, object],
) -> pd.DataFrame:
    accepted_ids = tuple(str(value) for value in source_receipt["accepted_game_ids"])
    accepted_set = set(accepted_ids)
    game_ids = _frame_game_ids(maps, "maps").astype(str)
    selected = maps.loc[game_ids.isin(accepted_set)].copy()
    selected_ids = _frame_game_ids(selected, "maps").astype(str)
    if selected_ids.duplicated().any() or set(selected_ids) != accepted_set:
        raise RuntimeError("maps do not match the accepted census for series binding")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--fold-spec", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--crosswalk", required=True, type=Path)
    parser.add_argument("--crosswalk-receipt", required=True, type=Path)
    parser.add_argument("--crosswalk-receipt-file-sha256", required=True)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    source_receipt_path = args.source_receipt.resolve()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise RuntimeError("output directory is missing or unsafe")
    if any(output_dir.iterdir()):
        raise RuntimeError("output directory is not empty")
    source_receipt = _load_json(source_receipt_path)
    fold = _load_json(args.fold_spec.resolve())
    maps = pd.read_parquet(source_root / "maps.parquet")
    players = pd.read_parquet(source_root / "oe_player_games.parquet")
    teams = pd.read_parquet(source_root / "oe_team_games.parquet")
    train_ids = tuple(str(value) for value in fold["train_game_ids"])
    validation_ids = tuple(str(value) for value in fold["validation_game_ids"])
    output_ids = tuple(sorted((*train_ids, *validation_ids)))
    accepted_maps = _accepted_map_frame(maps, source_receipt=source_receipt)
    bound_maps = bind_verified_leaguepedia_series_crosswalk(
        accepted_maps,
        crosswalk_path=args.crosswalk.resolve(),
        receipt_path=args.crosswalk_receipt.resolve(),
        source_receipt=source_receipt,
        expected_receipt_file_sha256=str(args.crosswalk_receipt_file_sha256),
    )
    model_frame = _map_model_frame(
        bound_maps,
        verified_source_receipt_sha256=str(source_receipt["receipt_sha256"]),
        verified_source_receipt=source_receipt,
        verified_crosswalk_receipt_file_sha256=str(
            args.crosswalk_receipt_file_sha256
        ),
    )
    eligible_ids = tuple(
        sorted(str(value) for value in source_receipt["model_eligible_game_ids"])
    )
    eligible_frame = model_frame[
        model_frame["game_id"].astype(str).isin(eligible_ids)
    ].copy()
    if tuple(sorted(eligible_frame["game_id"].astype(str))) != eligible_ids:
        raise RuntimeError("series assignment does not cover the eligible census")
    eligible_series = eligible_frame["series_id"].astype("string").str.strip()
    eligible_mapped = eligible_frame.get("_series_crosswalk_mapped")
    if (
        eligible_mapped is None
        or not pd.api.types.is_bool_dtype(eligible_mapped.dtype)
        or eligible_mapped.isna().any()
        or not bool(eligible_mapped.all())
        or eligible_series.isna().any()
        or not bool(eligible_series.str.startswith("leaguepedia:").all())
    ):
        raise RuntimeError(
            "model-eligible census lacks exact Leaguepedia series coverage"
        )
    fold_frame = eligible_frame[
        eligible_frame["game_id"].astype(str).isin(output_ids)
    ].copy()
    if tuple(sorted(fold_frame["game_id"].astype(str))) != output_ids:
        raise RuntimeError("series assignment does not cover the fold census")
    train_series = tuple(
        sorted(
            set(
                fold_frame.loc[
                    fold_frame["game_id"].astype(str).isin(train_ids), "series_id"
                ].astype(str)
            )
        )
    )
    validation_series = tuple(
        sorted(
            set(
                fold_frame.loc[
                    fold_frame["game_id"].astype(str).isin(validation_ids),
                    "series_id",
                ].astype(str)
            )
        )
    )
    fold_series_contract = {
        "train_series_ids": list(train_series),
        "train_series_count": len(train_series),
        "train_series_identity_sha256": identity_sha256(train_series),
        "validation_series_ids": list(validation_series),
        "validation_series_count": len(validation_series),
        "validation_series_identity_sha256": identity_sha256(validation_series),
    }
    if any(fold.get(key) != value for key, value in fold_series_contract.items()):
        raise RuntimeError("scaling series partition differs from fold spec")
    if set(train_series) & set(validation_series):
        raise RuntimeError("scaling fold series overlap")
    crosswalk_binding = bound_maps.attrs.get(
        "verified_leaguepedia_series_crosswalk"
    )
    if not isinstance(crosswalk_binding, dict):
        raise RuntimeError("verified series binding is missing")
    started = time.perf_counter()
    native, native_receipt = build_scaling_feature_ledger(
        maps,
        players,
        teams,
        source_receipt=source_receipt,
        source_receipt_sha256=str(source_receipt["receipt_sha256"]),
        train_game_ids=train_ids,
        validation_game_ids=validation_ids,
        fit_window_end=fold["fit_window_end"],
        model_eligible_only=True,
        output_game_ids=output_ids,
    )
    elapsed = time.perf_counter() - started
    native_path = output_dir / "scaling-native.parquet"
    native.to_parquet(native_path, index=False)
    native_receipt_path = output_dir / "scaling-native-receipt.json"
    _write_json(native_receipt_path, native_receipt)
    artifact_path = output_dir / "scaling-features.parquet"
    native[["game_id", *SCALING_CURVE_SIGNED_MAP_FEATURES]].to_parquet(
        artifact_path, index=False
    )
    producer_receipt_path = output_dir / "scaling-producer-receipt.json"
    adapter = write_rating_feature_producer_receipt(
        "strict_prior_atomized_scaling",
        artifact_path,
        producer_receipt_path,
        native_artifact_path=native_path,
        native_receipt_path=native_receipt_path,
        source_receipt=source_receipt,
        source_receipt_path=source_receipt_path,
        train_game_ids=train_ids,
        validation_game_ids=validation_ids,
        fit_window_end=fold["fit_window_end"],
    )
    manifest = build_rating_feature_producer_manifest([adapter])
    _write_json(output_dir / "scaling-adapter.json", adapter)
    _write_json(output_dir / "scaling-producer-manifest.json", manifest)
    series_receipt = {
        "schema_version": "scryglass:future-value-scaling-series-binding:v1",
        "status": "research_only",
        "fold": int(fold["fold"]),
        "fit_window_end": fold["fit_window_end"],
        "train_game_ids": list(train_ids),
        "validation_game_ids": list(validation_ids),
        "source_receipt_sha256": source_receipt["receipt_sha256"],
        "source_identity_sha256": source_receipt["source_identity_sha256"],
        "model_eligible_game_count": source_receipt["model_eligible_game_count"],
        "model_eligible_identity_sha256": source_receipt[
            "model_eligible_identity_sha256"
        ],
        "crosswalk_receipt_file_sha256": str(
            args.crosswalk_receipt_file_sha256
        ),
        "crosswalk_sha256": crosswalk_binding["crosswalk_sha256"],
        "crosswalk_assignment_sha256": crosswalk_binding["assignment_sha256"],
        "eligible_series_assignment_sha256": _series_assignment_sha256(
            eligible_frame
        ),
        "fold_series_assignment_sha256": _series_assignment_sha256(fold_frame),
        **fold_series_contract,
        "authority": {
            "research_only": True,
            "public_player_rating": False,
            "public_team_rating": False,
            "public_probability": False,
            "promotion": False,
            "deployment": False,
        },
    }
    series_receipt["receipt_sha256"] = hashlib.sha256(
        _canonical_bytes(series_receipt)
    ).hexdigest()
    _write_json(output_dir / "scaling-series-binding.json", series_receipt)
    run = {
        "fold": int(fold["fold"]),
        "rows": len(native),
        "columns": len(native.columns),
        "elapsed_seconds": elapsed,
        "native_artifact_sha256": _sha256(native_path),
        "feature_artifact_sha256": _sha256(artifact_path),
        "native_receipt_sha256": native_receipt["receipt_sha256"],
        "producer_receipt_sha256": json.loads(
            producer_receipt_path.read_text(encoding="utf-8")
        )["receipt_sha256"],
        "series_binding_receipt_sha256": series_receipt["receipt_sha256"],
        "authority": {
            "research_only": True,
            "public_player_rating": False,
            "public_team_rating": False,
            "public_probability": False,
            "promotion": False,
            "deployment": False,
        },
    }
    _write_json(output_dir / "scaling-run.json", run)
    print(json.dumps(run, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
