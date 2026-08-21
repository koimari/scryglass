"""Write series-safe future-value fold specifications from verified inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from lol_kills.research.future_value_rating import (
    _map_model_frame,
    bind_verified_leaguepedia_series_crosswalk,
    chronological_whole_series_folds,
    validate_future_value_source_receipt_payload,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


SCHEMA_VERSION = "scryglass:future-value-fold-spec-bundle:v1"


class FoldSpecError(RuntimeError):
    """The fold specifications cannot be bound safely."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FoldSpecError(f"{label} is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FoldSpecError(f"{label} cannot be read") from error
    if not isinstance(value, dict):
        raise FoldSpecError(f"{label} must be a JSON object")
    return value


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_fold_specs(
    *,
    maps: pd.DataFrame,
    source_receipt: Mapping[str, Any],
    crosswalk_path: Path,
    crosswalk_receipt_path: Path,
    expected_crosswalk_receipt_file_sha256: str,
    n_folds: int = 3,
) -> dict[str, Any]:
    """Build chronological folds from one verified mixed series partition."""

    validate_future_value_source_receipt_payload(source_receipt)
    bound_maps = bind_verified_leaguepedia_series_crosswalk(
        maps,
        crosswalk_path=crosswalk_path,
        receipt_path=crosswalk_receipt_path,
        source_receipt=source_receipt,
        expected_receipt_file_sha256=expected_crosswalk_receipt_file_sha256,
    )
    model_frame = _map_model_frame(
        bound_maps,
        verified_source_receipt_sha256=str(source_receipt["receipt_sha256"]),
        verified_source_receipt=source_receipt,
        verified_crosswalk_receipt_file_sha256=expected_crosswalk_receipt_file_sha256,
    )
    eligible_ids = tuple(
        sorted(str(value) for value in source_receipt["model_eligible_game_ids"])
    )
    model_frame = model_frame[
        model_frame["game_id"].astype(str).isin(eligible_ids)
    ].copy()
    if tuple(sorted(model_frame["game_id"].astype(str))) != eligible_ids:
        raise FoldSpecError("model frame does not match the eligible source census")
    folds = chronological_whole_series_folds(
        model_frame,
        n_folds=n_folds,
        verified_model_frame=model_frame,
    )
    if len(folds) != n_folds:
        raise FoldSpecError("fold count changed")
    records: list[dict[str, Any]] = []
    validation_seen: set[str] = set()
    for expected_fold, raw in enumerate(folds, start=1):
        train_ids = tuple(sorted(str(value) for value in raw["train_game_ids"]))
        validation_ids = tuple(
            sorted(str(value) for value in raw["validation_game_ids"])
        )
        if int(raw["fold"]) != expected_fold:
            raise FoldSpecError("fold order changed")
        if set(train_ids) & set(validation_ids):
            raise FoldSpecError("fold train and validation IDs overlap")
        if validation_seen & set(validation_ids):
            raise FoldSpecError("validation IDs repeat across folds")
        validation_seen.update(validation_ids)
        train_series = set(
            model_frame.loc[
                model_frame["game_id"].astype(str).isin(train_ids), "series_id"
            ].astype(str)
        )
        validation_series = set(
            model_frame.loc[
                model_frame["game_id"].astype(str).isin(validation_ids), "series_id"
            ].astype(str)
        )
        if train_series & validation_series:
            raise FoldSpecError("fold series overlap")
        records.append(
            {
                "fold": expected_fold,
                "fit_window_end": str(raw["validation_start"]),
                "train_game_ids": list(train_ids),
                "validation_game_ids": list(validation_ids),
                "train_game_count": len(train_ids),
                "train_identity_sha256": identity_sha256(train_ids),
                "validation_game_count": len(validation_ids),
                "validation_identity_sha256": identity_sha256(validation_ids),
                "validation_start": str(raw["validation_start"]),
                "validation_end": str(raw["validation_end"]),
                "boundary_excluded_game_count": int(
                    raw["overlap_audit"]["excluded_boundary_map_count"]
                ),
                "overlap_audit": dict(raw["overlap_audit"]),
            }
        )
    audit = dict(model_frame.attrs.get("series_cluster_audit") or {})
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "research_only",
        "source": {
            "source_receipt_sha256": source_receipt["receipt_sha256"],
            "source_identity_sha256": source_receipt["source_identity_sha256"],
            "model_eligible_game_count": len(eligible_ids),
            "model_eligible_identity_sha256": identity_sha256(eligible_ids),
        },
        "series_partition": audit,
        "folds": records,
        "authority": {
            "research_only": True,
            "public_player_rating": False,
            "public_team_rating": False,
            "public_probability": False,
            "promotion": False,
            "deployment": False,
        },
    }
    payload["bundle_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--crosswalk", type=Path, required=True)
    parser.add_argument("--crosswalk-receipt", type=Path, required=True)
    parser.add_argument("--crosswalk-receipt-file-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args(argv)
    output_root = args.output_root.resolve()
    if not output_root.is_dir() or output_root.is_symlink() or any(output_root.iterdir()):
        raise FoldSpecError("output root must be a safe empty directory")
    source_receipt_path = args.source_receipt.resolve()
    source_receipt = _load_json(source_receipt_path, "source receipt")
    payload = build_fold_specs(
        maps=pd.read_parquet(args.maps.resolve()),
        source_receipt=source_receipt,
        crosswalk_path=args.crosswalk.resolve(),
        crosswalk_receipt_path=args.crosswalk_receipt.resolve(),
        expected_crosswalk_receipt_file_sha256=args.crosswalk_receipt_file_sha256,
        n_folds=args.folds,
    )
    for record in payload["folds"]:
        path = output_root / f"fold-{record['fold']}-spec.json"
        path.write_bytes(
            _canonical_bytes(
                {
                    "fold": record["fold"],
                    "fit_window_end": record["fit_window_end"],
                    "train_game_ids": record["train_game_ids"],
                    "validation_game_ids": record["validation_game_ids"],
                }
            )
        )
    bundle_path = output_root / "fold-spec-bundle.json"
    bundle_path.write_bytes(_canonical_bytes(payload))
    print(
        json.dumps(
            {
                "output": str(bundle_path),
                "sha256": _sha256_path(bundle_path),
                "bundle_sha256": payload["bundle_sha256"],
                "folds": len(payload["folds"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
