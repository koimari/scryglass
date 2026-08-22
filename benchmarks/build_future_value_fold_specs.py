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


def _accepted_map_frame(
    maps: pd.DataFrame,
    source_receipt: Mapping[str, Any],
) -> pd.DataFrame:
    """Return one map row for each accepted game and exclude declared extras."""

    game_column = "game_uid" if "game_uid" in maps.columns else "game_id"
    if game_column not in maps.columns:
        raise FoldSpecError("maps have no game identity column")
    game_ids = maps[game_column].astype("string").str.strip()
    if game_ids.isna().any() or game_ids.eq("").any():
        raise FoldSpecError("maps contain missing game identities")
    accepted_ids = tuple(str(value) for value in source_receipt["accepted_game_ids"])
    accepted_set = set(accepted_ids)
    extra_binding = source_receipt.get("source_extra_game_ids")
    declared_extras = {
        str(value)
        for value in (
            extra_binding.get("maps", ())
            if isinstance(extra_binding, Mapping)
            else ()
        )
    }
    physical_ids = set(game_ids.astype(str))
    unknown = physical_ids - accepted_set - declared_extras
    if unknown:
        raise FoldSpecError("maps contain undeclared game identities")
    missing = accepted_set - physical_ids
    if missing:
        raise FoldSpecError("maps are missing accepted game identities")
    accepted_mask = game_ids.astype(str).isin(accepted_set)
    accepted = maps.loc[accepted_mask].copy()
    accepted_game_ids = accepted[game_column].astype("string").str.strip()
    if accepted_game_ids.duplicated().any():
        raise FoldSpecError("maps contain duplicate accepted game identities")
    if set(accepted_game_ids.astype(str)) != accepted_set:
        raise FoldSpecError("accepted map frame differs from source receipt")
    return accepted.reset_index(drop=True)


def _eligible_series_audit(model_frame: pd.DataFrame) -> dict[str, Any]:
    """Scope crosswalk authority to the exact eligible model rows."""

    audit = dict(model_frame.attrs.get("series_cluster_audit") or {})
    required_hashes = (
        "crosswalk_assignment_sha256",
        "crosswalk_sha256",
        "crosswalk_artifact_sha256",
        "crosswalk_receipt_sha256",
    )
    if any(not str(audit.get(field) or "") for field in required_hashes):
        raise FoldSpecError("series partition provenance is incomplete")
    if "_series_crosswalk_mapped" not in model_frame.columns:
        raise FoldSpecError("series partition has no exact crosswalk flags")
    mapped = model_frame["_series_crosswalk_mapped"]
    if not pd.api.types.is_bool_dtype(mapped.dtype) or mapped.isna().any():
        raise FoldSpecError("series partition crosswalk flags are invalid")
    series = model_frame["series_id"].astype("string").str.strip()
    if series.isna().any() or series.eq("").any():
        raise FoldSpecError("series partition assignments are incomplete")
    exact = mapped.astype(bool) & series.str.startswith("leaguepedia:")
    counts = series.value_counts(sort=False)
    proxy_series = series.loc[~exact]
    audit.update(
        {
            "scope": "model_eligible_census",
            "map_count": int(len(model_frame)),
            "mapped_game_count": int(exact.sum()),
            "unmatched_game_count": int((~exact).sum()),
            "mapped_series_count": int(series.loc[exact].nunique()),
            "promoted_game_count": int(exact.sum()),
            "promoted_series_count": int(series.loc[exact].nunique()),
            "retained_proxy_game_count": int((~exact).sum()),
            "retained_proxy_cluster_count": int(proxy_series.nunique()),
            "partial_series_blocker": bool((~exact).any()),
            "authoritative": bool(exact.all()),
            "authoritative_series_blocker": (
                None
                if bool(exact.all())
                else "authoritative_series_id_missing_proxy_cluster_used"
            ),
            "cluster_count": int(series.nunique()),
            "colliding_cluster_count": int(counts.gt(1).sum()),
            "collision_extra_map_count": int(
                counts.loc[counts.gt(1)].sub(1).sum()
            ),
            "max_cluster_size": int(counts.max()),
        }
    )
    return audit


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
    accepted_maps = _accepted_map_frame(maps, source_receipt)
    bound_maps = bind_verified_leaguepedia_series_crosswalk(
        accepted_maps,
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
    audit = _eligible_series_audit(model_frame)
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
