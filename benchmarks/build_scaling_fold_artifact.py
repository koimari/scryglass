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
    build_rating_feature_producer_manifest,
    write_rating_feature_producer_receipt,
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
