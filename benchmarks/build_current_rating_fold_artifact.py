"""Build one producer-owned fold-local current-rating artifact and receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import pandas as pd

from lol_kills.research.future_value_rating import (
    _map_model_frame,
    bind_verified_leaguepedia_series_crosswalk,
    build_rating_feature_producer_manifest,
    write_rating_feature_producer_receipt,
)
from lol_kills.research.future_value_rating_ledger import (
    build_fold_current_rating_feature_ledger,
    validate_fold_current_rating_feature_ledger,
)


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
        bound_maps = bind_verified_leaguepedia_series_crosswalk(
            maps,
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
        series_by_game = dict(
            zip(
                model_frame["game_id"].astype(str),
                model_frame["series_id"].astype(str),
            )
        )
        series_source = str(model_frame.attrs["series_cluster_source"])
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
