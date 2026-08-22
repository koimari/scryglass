"""Build one producer-owned fold-local current-rating artifact and receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import pandas as pd

from lol_kills.research.future_value_rating import (
    _frame_game_ids,
    _map_model_frame,
    bind_verified_leaguepedia_series_crosswalk,
    build_rating_feature_producer_manifest,
    write_rating_feature_producer_receipt,
)
from lol_kills.research.future_value_rating_ledger import (
    build_fold_current_rating_feature_ledger,
    validate_fold_current_rating_feature_ledger,
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


def _accepted_map_frame(
    maps: pd.DataFrame,
    *,
    source_receipt: dict[str, object],
) -> pd.DataFrame:
    """Select the exact accepted map census before series assignment."""

    accepted_ids = tuple(str(value) for value in source_receipt["accepted_game_ids"])
    accepted_set = set(accepted_ids)
    game_ids = _frame_game_ids(maps, "maps").astype(str)
    selected = maps.loc[game_ids.isin(accepted_set)].copy()
    selected_ids = _frame_game_ids(selected, "maps").astype(str)
    if selected_ids.duplicated().any() or set(selected_ids) != accepted_set:
        raise RuntimeError("maps do not match the accepted census for series binding")
    return selected


def _validate_fold_series_contract(
    fold: dict[str, object],
    model_frame: pd.DataFrame,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    train_ids = {str(value) for value in fold["train_game_ids"]}
    validation_ids = {str(value) for value in fold["validation_game_ids"]}
    train_series = tuple(
        sorted(
            set(
                model_frame.loc[
                    model_frame["game_id"].astype(str).isin(train_ids),
                    "series_id",
                ].astype(str)
            )
        )
    )
    validation_series = tuple(
        sorted(
            set(
                model_frame.loc[
                    model_frame["game_id"].astype(str).isin(validation_ids),
                    "series_id",
                ].astype(str)
            )
        )
    )
    expected = {
        "train_series_ids": list(train_series),
        "train_series_count": len(train_series),
        "train_series_identity_sha256": identity_sha256(train_series),
        "validation_series_ids": list(validation_series),
        "validation_series_count": len(validation_series),
        "validation_series_identity_sha256": identity_sha256(validation_series),
    }
    if any(fold.get(key) != value for key, value in expected.items()):
        raise RuntimeError("current-rating series partition differs from fold spec")
    if set(train_series) & set(validation_series):
        raise RuntimeError("current-rating fold series overlap")
    return train_series, validation_series


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--fold-spec", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--crosswalk", type=Path)
    parser.add_argument("--crosswalk-receipt", type=Path)
    parser.add_argument("--crosswalk-receipt-file-sha256")
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    source_receipt_path = args.source_receipt.resolve()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir() or output_dir.is_symlink() or any(output_dir.iterdir()):
        raise RuntimeError("output directory must be safe and empty")
    source_receipt = _load_json(source_receipt_path)
    fold = _load_json(args.fold_spec.resolve())
    maps = pd.read_parquet(source_root / "maps.parquet")
    players = pd.read_parquet(source_root / "oe_player_games.parquet")
    teams = pd.read_parquet(source_root / "oe_team_games.parquet")
    crosswalk_values = (
        args.crosswalk,
        args.crosswalk_receipt,
        args.crosswalk_receipt_file_sha256,
    )
    if any(value is not None for value in crosswalk_values) and not all(
        value is not None for value in crosswalk_values
    ):
        raise RuntimeError("crosswalk inputs must be supplied together")
    series_by_game = None
    series_source = "conservative_series_superset"
    series_receipt_file_sha256 = None
    if args.crosswalk is not None and args.crosswalk_receipt is not None:
        accepted_maps = _accepted_map_frame(maps, source_receipt=source_receipt)
        bound_maps = bind_verified_leaguepedia_series_crosswalk(
            accepted_maps,
            crosswalk_path=args.crosswalk.resolve(),
            receipt_path=args.crosswalk_receipt.resolve(),
            source_receipt=source_receipt,
            expected_receipt_file_sha256=str(
                args.crosswalk_receipt_file_sha256
            ),
        )
        model_frame = _map_model_frame(
            bound_maps,
            verified_source_receipt=source_receipt,
            verified_source_receipt_sha256=str(source_receipt["receipt_sha256"]),
            verified_crosswalk_receipt_file_sha256=str(
                args.crosswalk_receipt_file_sha256
            ),
        )
        eligible_ids = {
            str(value) for value in source_receipt["model_eligible_game_ids"]
        }
        model_frame = model_frame[
            model_frame["game_id"].astype(str).isin(eligible_ids)
        ].copy()
        if set(model_frame["game_id"].astype(str)) != eligible_ids:
            raise RuntimeError("series frame does not match the eligible census")
        mapped = model_frame.get("_series_crosswalk_mapped")
        assignments = model_frame.get("_series_crosswalk_assignment")
        series = model_frame["series_id"].astype("string").str.strip()
        if (
            mapped is None
            or assignments is None
            or not pd.api.types.is_bool_dtype(mapped.dtype)
            or mapped.isna().any()
            or not bool(mapped.all())
            or assignments.astype("string").str.strip().isna().any()
            or assignments.astype("string").str.strip().eq("").any()
            or series.isna().any()
            or not bool(series.str.startswith("leaguepedia:").all())
        ):
            raise RuntimeError(
                "model-eligible census lacks exact Leaguepedia series coverage"
            )
        series_by_game = dict(
            zip(
                model_frame["game_id"].astype(str),
                model_frame["series_id"].astype(str),
            )
        )
        _validate_fold_series_contract(fold, model_frame)
        series_source = "verified_leaguepedia_series_crosswalk"
        series_receipt_file_sha256 = str(
            args.crosswalk_receipt_file_sha256
        )
    train_ids = tuple(str(value) for value in fold["train_game_ids"])
    validation_ids = tuple(str(value) for value in fold["validation_game_ids"])
    started = time.perf_counter()
    ledger, native_receipt = build_fold_current_rating_feature_ledger(
        maps,
        players,
        teams,
        source_receipt=source_receipt,
        train_game_ids=train_ids,
        validation_game_ids=validation_ids,
        fit_window_end=fold["fit_window_end"],
        destination=output_dir,
        series_by_game=series_by_game,
        series_partition_source=series_source,
        series_partition_receipt_file_sha256=series_receipt_file_sha256,
    )
    elapsed = time.perf_counter() - started
    native_path = output_dir / "current-rating-feature-ledger.parquet"
    native_receipt_path = output_dir / "current-rating-feature-ledger.receipt.json"
    validate_fold_current_rating_feature_ledger(
        ledger,
        native_receipt,
        source_receipt=source_receipt,
        train_game_ids=train_ids,
        validation_game_ids=validation_ids,
        fit_window_end=fold["fit_window_end"],
    )
    if series_source == "verified_leaguepedia_series_crosswalk" and any(
        native_receipt.get(key) != fold.get(key)
        for key in (
            "train_series_ids",
            "train_series_count",
            "train_series_identity_sha256",
            "validation_series_ids",
            "validation_series_count",
            "validation_series_identity_sha256",
        )
    ):
        raise RuntimeError("current-rating receipt series differs from fold spec")
    producer_receipt_path = output_dir / "current-rating-producer-receipt.json"
    adapter = write_rating_feature_producer_receipt(
        "current_sequential_rating",
        native_path,
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
    _write_json(output_dir / "current-rating-adapter.json", adapter)
    _write_json(output_dir / "current-rating-producer-manifest.json", manifest)
    run = {
        "fold": int(fold["fold"]),
        "rows": len(ledger),
        "elapsed_seconds": elapsed,
        "artifact_sha256": _sha256(native_path),
        "native_receipt_sha256": native_receipt["receipt_sha256"],
        "producer_receipt_sha256": json.loads(
            producer_receipt_path.read_text(encoding="utf-8")
        )["receipt_sha256"],
        "authority": native_receipt["authority"],
    }
    _write_json(output_dir / "current-rating-run.json", run)
    print(json.dumps(run, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
