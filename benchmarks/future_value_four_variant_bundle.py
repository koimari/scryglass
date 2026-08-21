"""Build the durable fold ledgers for the four future-value variants.

This command joins the independently produced current-rating and scaling
artifacts.  It binds each fold to the same verified source receipt and the
same conservative series partition used by ``evaluate_future_value``.
The output stays research-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from lol_kills.research.future_value_rating import (
    CURRENT_RATING_SIGNED_MAP_FEATURES,
    RATING_VARIANT_ORDER,
    SCALING_CURVE_SIGNED_MAP_FEATURES,
    _map_model_frame,
    bind_rating_feature_ledger,
    build_rating_feature_producer_manifest,
    chronological_whole_series_folds,
    rating_variant_config_receipt,
    validate_future_value_source_receipt_payload,
    write_rating_feature_producer_receipt,
)
from lol_kills.research.atomized_rf_composite import build_scaling_feature_ledger
from lol_kills.research.future_value_rating_ledger import (
    build_fold_current_rating_feature_ledger,
    validate_fold_current_rating_feature_ledger,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


SCHEMA_VERSION = "scryglass:future-value-four-variant-ledger-bundle:v1"


class FourVariantBundleError(RuntimeError):
    """The four-variant ledger bundle cannot be built safely."""


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
        raise FourVariantBundleError(f"{label} is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FourVariantBundleError(f"{label} cannot be read") from error
    if not isinstance(value, dict):
        raise FourVariantBundleError(f"{label} must be a JSON object")
    return value


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest(manifest: Mapping[str, Any], label: str) -> dict[str, Any]:
    payload = dict(manifest)
    claimed = payload.pop("manifest_sha256", None)
    if not isinstance(claimed, str) or hashlib.sha256(_canonical_bytes(payload)).hexdigest() != claimed:
        raise FourVariantBundleError(f"{label} self hash changed")
    if payload.get("status") != "research_only":
        raise FourVariantBundleError(f"{label} status is invalid")
    authority = payload.get("authority")
    if not isinstance(authority, Mapping) or authority.get("research_only") is not True:
        raise FourVariantBundleError(f"{label} authority is invalid")
    if any(bool(value) for key, value in authority.items() if key != "research_only"):
        raise FourVariantBundleError(f"{label} grants authority")
    return dict(manifest)


def _verify_scaling_native_receipt(
    receipt: Mapping[str, Any],
    *,
    source_receipt: Mapping[str, Any],
    train_ids: tuple[str, ...],
    validation_ids: tuple[str, ...],
    fit_window_end: str,
    artifact: pd.DataFrame,
) -> None:
    payload = dict(receipt)
    claimed = payload.pop("receipt_sha256", None)
    if not isinstance(claimed, str) or hashlib.sha256(_canonical_bytes(payload)).hexdigest() != claimed:
        raise FourVariantBundleError("scaling native receipt self hash changed")
    if receipt.get("status") != "research_only":
        raise FourVariantBundleError("scaling native status is invalid")
    if receipt.get("authority") is not False or receipt.get("public_authority") is not False:
        raise FourVariantBundleError("scaling native receipt grants authority")
    if receipt.get("source_receipt_sha256") != source_receipt.get("receipt_sha256"):
        raise FourVariantBundleError("scaling native source receipt changed")
    if receipt.get("source_identity_sha256") != source_receipt.get("source_identity_sha256"):
        raise FourVariantBundleError("scaling native source identity changed")
    if receipt.get("evaluation_mode") != "fold_local" or receipt.get("fold_evaluation_usable") is not True:
        raise FourVariantBundleError("scaling native artifact is not fold-local")
    if receipt.get("fold_blocker") is not None:
        raise FourVariantBundleError("scaling native artifact carries a fold blocker")
    if tuple(receipt.get("train_game_ids") or ()) != train_ids:
        raise FourVariantBundleError("scaling native training IDs changed")
    if tuple(receipt.get("validation_game_ids") or ()) != validation_ids:
        raise FourVariantBundleError("scaling native validation IDs changed")
    if receipt.get("fit_window_end") != fit_window_end:
        raise FourVariantBundleError("scaling native cutoff changed")
    expected_ids = tuple(sorted((*train_ids, *validation_ids)))
    artifact_ids = tuple(sorted(artifact["game_id"].astype(str)))
    if artifact_ids != expected_ids:
        raise FourVariantBundleError("scaling artifact coverage changed")
    if receipt.get("output_game_count") != len(expected_ids):
        raise FourVariantBundleError("scaling native output count changed")
    if receipt.get("output_identity_sha256") != identity_sha256(expected_ids):
        raise FourVariantBundleError("scaling native output identity changed")
    if tuple(sorted(str(value) for value in receipt.get("output_game_ids") or ())) != expected_ids:
        raise FourVariantBundleError("scaling native output IDs changed")


def _descriptor_adapter(path: Path, expected_name: str) -> dict[str, Any]:
    value = _load_json(path, f"{expected_name} adapter")
    record_fields = {
        "name",
        "artifact",
        "native_artifact",
        "receipt",
        "native_receipt",
    }
    if set(value) != record_fields or value.get("name") != expected_name:
        raise FourVariantBundleError(f"{expected_name} adapter schema changed")
    for field in ("artifact", "native_artifact", "receipt", "native_receipt"):
        record = value.get(field)
        if not isinstance(record, Mapping) or set(record) != {"path", "bytes", "sha256"}:
            raise FourVariantBundleError(f"{expected_name} {field} record changed")
        file_path = Path(str(record["path"]))
        if not file_path.is_file() or file_path.is_symlink():
            raise FourVariantBundleError(f"{expected_name} {field} is missing or unsafe")
        if int(record["bytes"]) != file_path.stat().st_size or str(record["sha256"]) != _sha256_path(file_path):
            raise FourVariantBundleError(f"{expected_name} {field} bytes changed")
    return value


def _json_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"], utc=True).map(
        lambda value: value.isoformat().replace("+00:00", "Z")
    )
    rows = output.to_dict("records")
    for row in rows:
        for key, value in tuple(row.items()):
            if isinstance(value, np.generic):
                row[key] = value.item()
    return rows


def _write_json_no_clobber(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FourVariantBundleError(f"output already exists: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise FourVariantBundleError(f"output parent is unsafe: {path.parent}")
    path.write_bytes(_canonical_bytes(value))


def _derive_inner_fold_spec(
    model_frame: pd.DataFrame,
    *,
    outer_fold: int,
    outer_train_ids: tuple[str, ...],
    outer_validation_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Derive one nested chronological fold from an outer training census."""

    if not outer_train_ids or not outer_validation_ids:
        raise FourVariantBundleError("nested fold has an empty outer partition")
    if set(outer_train_ids) & set(outer_validation_ids):
        raise FourVariantBundleError("nested fold outer partitions overlap")
    outer_train = model_frame[
        model_frame["game_id"].astype(str).isin(outer_train_ids)
    ].copy()
    if tuple(sorted(outer_train["game_id"].astype(str))) != outer_train_ids:
        raise FourVariantBundleError("nested fold outer training census is incomplete")
    try:
        candidates = chronological_whole_series_folds(
            outer_train,
            n_folds=1,
            verified_model_frame=outer_train,
        )
    except Exception as error:
        raise FourVariantBundleError("nested chronological fold construction failed") from error
    if len(candidates) != 1:
        raise FourVariantBundleError("nested chronological fold count changed")
    candidate = dict(candidates[0])
    inner_train_ids = tuple(sorted(str(value) for value in candidate["train_game_ids"]))
    inner_validation_ids = tuple(
        sorted(str(value) for value in candidate["validation_game_ids"])
    )
    if set(inner_train_ids) & set(inner_validation_ids):
        raise FourVariantBundleError("nested fold train and validation IDs overlap")
    if set(inner_train_ids) | set(inner_validation_ids) != set(outer_train_ids):
        raise FourVariantBundleError("nested fold does not cover the outer training census")
    if set(inner_train_ids) & set(outer_validation_ids) or set(inner_validation_ids) & set(
        outer_validation_ids
    ):
        raise FourVariantBundleError("nested fold contains outer validation IDs")
    candidate.update(
        {
            "fold": int(outer_fold),
            "outer_fold": int(outer_fold),
            "inner_fold": int(candidate["fold"]),
            "fit_window_end": str(candidate["validation_start"]),
            "outer_train_game_ids": list(outer_train_ids),
            "outer_train_identity_sha256": identity_sha256(outer_train_ids),
            "outer_validation_game_ids": list(outer_validation_ids),
            "outer_validation_identity_sha256": identity_sha256(outer_validation_ids),
        }
    )
    return candidate


def _prepare_inner_output_root(path: Path) -> Path:
    root = path.resolve()
    if root.exists() and root.is_symlink():
        raise FourVariantBundleError("nested inner output root is a symlink")
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir() or root.is_symlink() or any(root.iterdir()):
        raise FourVariantBundleError("nested inner output root must be empty")
    return root


def _build_inner_fold_artifacts(
    *,
    source_receipt_path: Path,
    source_receipt: Mapping[str, Any],
    maps: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    identity: pd.DataFrame,
    fold_number: int,
    inner_fold: Mapping[str, Any],
    output_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Build and bind durable producer artifacts for one inner fold."""

    fold_root = output_root / f"fold-{fold_number}"
    current_root = fold_root / "current"
    scaling_root = fold_root / "scaling"
    for directory in (fold_root, current_root, scaling_root):
        if directory.exists() and directory.is_symlink():
            raise FourVariantBundleError("nested inner artifact directory is a symlink")
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise FourVariantBundleError("nested inner artifact directory is not empty")

    train_ids = tuple(sorted(str(value) for value in inner_fold["train_game_ids"]))
    validation_ids = tuple(
        sorted(str(value) for value in inner_fold["validation_game_ids"])
    )
    fit_window_end = str(inner_fold["validation_start"])
    output_ids = tuple(sorted((*train_ids, *validation_ids)))

    current, current_native = build_fold_current_rating_feature_ledger(
        maps,
        players,
        teams,
        source_receipt=source_receipt,
        train_game_ids=train_ids,
        validation_game_ids=validation_ids,
        fit_window_end=fit_window_end,
        destination=current_root,
    )
    validate_fold_current_rating_feature_ledger(
        current,
        current_native,
        source_receipt=source_receipt,
        train_game_ids=train_ids,
        validation_game_ids=validation_ids,
        fit_window_end=fit_window_end,
        source_frames={"maps": maps, "players": players, "teams": teams},
    )
    current_path = current_root / "current-rating-feature-ledger.parquet"
    current_native_path = current_root / "current-rating-feature-ledger.receipt.json"
    current_receipt_path = current_root / "current-rating-producer-receipt.json"
    current_adapter = write_rating_feature_producer_receipt(
        "current_sequential_rating",
        current_path,
        current_receipt_path,
        native_artifact_path=current_path,
        native_receipt_path=current_native_path,
        source_receipt=source_receipt,
        source_receipt_path=source_receipt_path,
        train_game_ids=train_ids,
        validation_game_ids=validation_ids,
        fit_window_end=fit_window_end,
    )
    _write_json_no_clobber(current_root / "current-rating-adapter.json", current_adapter)
    current_manifest = build_rating_feature_producer_manifest([current_adapter])
    _write_json_no_clobber(
        current_root / "current-rating-producer-manifest.json", current_manifest
    )

    scaling_native, scaling_native_receipt = build_scaling_feature_ledger(
        maps,
        players,
        teams,
        source_receipt=source_receipt,
        source_receipt_sha256=str(source_receipt["receipt_sha256"]),
        train_game_ids=train_ids,
        validation_game_ids=validation_ids,
        fit_window_end=fit_window_end,
        model_eligible_only=True,
        output_game_ids=output_ids,
    )
    scaling_native_path = scaling_root / "scaling-native.parquet"
    scaling_native.to_parquet(scaling_native_path, index=False)
    _write_json_no_clobber(
        scaling_root / "scaling-native-receipt.json", scaling_native_receipt
    )
    scaling_path = scaling_root / "scaling-features.parquet"
    scaling_native[["game_id", *SCALING_CURVE_SIGNED_MAP_FEATURES]].to_parquet(
        scaling_path, index=False
    )
    scaling_receipt_path = scaling_root / "scaling-producer-receipt.json"
    scaling_adapter = write_rating_feature_producer_receipt(
        "strict_prior_atomized_scaling",
        scaling_path,
        scaling_receipt_path,
        native_artifact_path=scaling_native_path,
        native_receipt_path=scaling_root / "scaling-native-receipt.json",
        source_receipt=source_receipt,
        source_receipt_path=source_receipt_path,
        train_game_ids=train_ids,
        validation_game_ids=validation_ids,
        fit_window_end=fit_window_end,
    )
    _write_json_no_clobber(scaling_root / "scaling-adapter.json", scaling_adapter)
    scaling_manifest = build_rating_feature_producer_manifest([scaling_adapter])
    _write_json_no_clobber(
        scaling_root / "scaling-producer-manifest.json", scaling_manifest
    )

    _verify_scaling_native_receipt(
        scaling_native_receipt,
        source_receipt=source_receipt,
        train_ids=train_ids,
        validation_ids=validation_ids,
        fit_window_end=fit_window_end,
        artifact=scaling_native,
    )
    _verify_manifest(current_manifest, "nested current rating producer manifest")
    _verify_manifest(scaling_manifest, "nested scaling producer manifest")
    _descriptor_adapter(
        current_root / "current-rating-adapter.json", "current_sequential_rating"
    )
    _descriptor_adapter(
        scaling_root / "scaling-adapter.json", "strict_prior_atomized_scaling"
    )

    inner_identity = identity[identity["game_id"].astype(str).isin(output_ids)].copy()
    if tuple(sorted(inner_identity["game_id"].astype(str))) != output_ids:
        raise FourVariantBundleError("nested identity rows are incomplete")
    raw = inner_identity.merge(
        current[["game_id", *CURRENT_RATING_SIGNED_MAP_FEATURES]],
        on="game_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        scaling_native[["game_id", *SCALING_CURVE_SIGNED_MAP_FEATURES]],
        on="game_id",
        how="inner",
        validate="one_to_one",
    )
    if tuple(sorted(raw["game_id"].astype(str))) != output_ids:
        raise FourVariantBundleError("nested joined feature rows are incomplete")
    combined_manifest = build_rating_feature_producer_manifest(
        [current_adapter, scaling_adapter]
    )
    variant_inputs = {
        "current_only": (CURRENT_RATING_SIGNED_MAP_FEATURES, current_manifest),
        "future_player_form": (CURRENT_RATING_SIGNED_MAP_FEATURES, current_manifest),
        "scaling_curve": (
            (*CURRENT_RATING_SIGNED_MAP_FEATURES, *SCALING_CURVE_SIGNED_MAP_FEATURES),
            combined_manifest,
        ),
        "both": (
            (*CURRENT_RATING_SIGNED_MAP_FEATURES, *SCALING_CURVE_SIGNED_MAP_FEATURES),
            combined_manifest,
        ),
    }
    records: dict[str, dict[str, Any]] = {}
    feature_digests: dict[str, str] = {}
    producer_receipts: dict[str, str] = {}
    for variant, (feature_names, manifest) in variant_inputs.items():
        bound = bind_rating_feature_ledger(
            raw[["game_id", "date", "series_id", *feature_names]],
            source_receipt=source_receipt,
            train_game_ids=train_ids,
            validation_game_ids=validation_ids,
            fit_window_end=fit_window_end,
            feature_names=feature_names,
            producer=manifest,
        )
        if set(bound["game_id"].astype(str)) & set(
            identity.loc[~identity["game_id"].astype(str).isin(output_ids), "game_id"].astype(str)
        ):
            raise FourVariantBundleError("nested bound ledger contains outside IDs")
        records[variant] = {"rows": _json_rows(bound), "attrs": dict(bound.attrs)}
        feature_digests[variant] = str(bound.attrs["feature_value_digest"])
        producer_receipts[variant] = str(bound.attrs["producer_receipt_sha256"])

    receipt = {
        "fold": int(fold_number),
        "inner_fold": int(inner_fold["inner_fold"]),
        "fit_window_end": fit_window_end,
        "outer_train_game_count": len(inner_fold["outer_train_game_ids"]),
        "outer_train_identity_sha256": inner_fold["outer_train_identity_sha256"],
        "outer_validation_game_count": len(inner_fold["outer_validation_game_ids"]),
        "outer_validation_identity_sha256": inner_fold[
            "outer_validation_identity_sha256"
        ],
        "inner_train_game_count": len(train_ids),
        "inner_train_identity_sha256": identity_sha256(train_ids),
        "inner_validation_game_count": len(validation_ids),
        "inner_validation_identity_sha256": identity_sha256(validation_ids),
        "output_game_count": len(output_ids),
        "output_identity_sha256": identity_sha256(output_ids),
        "current_native_receipt_sha256": current_native["receipt_sha256"],
        "scaling_native_receipt_sha256": scaling_native_receipt["receipt_sha256"],
        "current_manifest_sha256": current_manifest["manifest_sha256"],
        "scaling_manifest_sha256": scaling_manifest["manifest_sha256"],
        "combined_manifest_sha256": combined_manifest["manifest_sha256"],
        "variant_feature_value_digests": feature_digests,
        "variant_producer_receipt_sha256": producer_receipts,
        "strict_prior": True,
        "outer_validation_excluded": True,
        "overlap_audit": dict(inner_fold["overlap_audit"]),
    }
    receipt["receipt_sha256"] = hashlib.sha256(_canonical_bytes(receipt)).hexdigest()
    _write_json_no_clobber(fold_root / "inner-fold-receipt.json", receipt)
    return records, receipt


def build_bundle(
    *,
    source_root: Path,
    source_receipt_path: Path,
    folds_root: Path,
    inner_output_root: Path | None = None,
) -> dict[str, Any]:
    source_receipt = _load_json(source_receipt_path, "source receipt")
    validate_future_value_source_receipt_payload(source_receipt)
    maps = pd.read_parquet(source_root / "maps.parquet")
    players = pd.read_parquet(source_root / "oe_player_games.parquet")
    teams = pd.read_parquet(source_root / "oe_team_games.parquet")
    model_frame = _map_model_frame(maps)
    eligible_ids = tuple(sorted(str(value) for value in source_receipt["model_eligible_game_ids"]))
    model_frame = model_frame[model_frame["game_id"].astype(str).isin(eligible_ids)].copy()
    if tuple(sorted(model_frame["game_id"].astype(str))) != eligible_ids:
        raise FourVariantBundleError("model frame does not match the eligible census")
    identity = model_frame[["game_id", "date", "series_id"]].copy()
    nested_root = _prepare_inner_output_root(
        folds_root / "nested-inner-v1"
        if inner_output_root is None
        else inner_output_root
    )
    variants: dict[str, dict[str, Any]] = {
        variant.value: {"folds": {}, "inner_folds": {}}
        for variant in RATING_VARIANT_ORDER
    }
    fold_receipts: list[dict[str, Any]] = []
    inner_fold_receipts: list[dict[str, Any]] = []
    for fold_number in (1, 2, 3):
        fold_spec = _load_json(folds_root / f"fold-{fold_number}-spec.json", "fold spec")
        if int(fold_spec.get("fold") or -1) != fold_number:
            raise FourVariantBundleError("fold number changed")
        train_ids = tuple(sorted(str(value) for value in fold_spec["train_game_ids"]))
        validation_ids = tuple(sorted(str(value) for value in fold_spec["validation_game_ids"]))
        fit_window_end = str(fold_spec["fit_window_end"])
        fold_ids = tuple(sorted((*train_ids, *validation_ids)))
        if set(train_ids) & set(validation_ids):
            raise FourVariantBundleError("fold train and validation IDs overlap")
        fold_identity = identity[identity["game_id"].astype(str).isin(fold_ids)].copy()
        if tuple(sorted(fold_identity["game_id"].astype(str))) != fold_ids:
            raise FourVariantBundleError("fold identity rows are incomplete")
        train_series = set(fold_identity.loc[fold_identity["game_id"].isin(train_ids), "series_id"])
        validation_series = set(
            fold_identity.loc[fold_identity["game_id"].isin(validation_ids), "series_id"]
        )
        if train_series & validation_series:
            raise FourVariantBundleError("fold series overlap under the evaluator partition")

        current_root = folds_root / f"fold-{fold_number}" / "current-v2"
        current_path = current_root / "current-rating-feature-ledger.parquet"
        current = pd.read_parquet(current_path)
        current_native = _load_json(
            current_root / "current-rating-feature-ledger.receipt.json",
            "current rating native receipt",
        )
        validate_fold_current_rating_feature_ledger(
            current,
            current_native,
            source_receipt=source_receipt,
            train_game_ids=train_ids,
            validation_game_ids=validation_ids,
            fit_window_end=fit_window_end,
            source_frames={"maps": maps, "players": players, "teams": teams},
        )
        current_manifest = _verify_manifest(
            _load_json(
                current_root / "current-rating-producer-manifest.json",
                "current rating producer manifest",
            ),
            "current rating producer manifest",
        )

        scaling_root = folds_root / f"fold-{fold_number}" / "scaling-v2"
        scaling_path = scaling_root / "scaling-features.parquet"
        scaling = pd.read_parquet(scaling_path)
        scaling_native_frame = pd.read_parquet(
            scaling_root / "scaling-native.parquet"
        )
        scaling_native = _load_json(
            scaling_root / "scaling-native-receipt.json", "scaling native receipt"
        )
        _verify_scaling_native_receipt(
            scaling_native,
            source_receipt=source_receipt,
            train_ids=train_ids,
            validation_ids=validation_ids,
            fit_window_end=fit_window_end,
            artifact=scaling_native_frame,
        )
        scaling_adapter = _descriptor_adapter(
            scaling_root / "scaling-adapter.json", "strict_prior_atomized_scaling"
        )
        current_adapter = dict(current_manifest["adapters"][0])
        combined_manifest = build_rating_feature_producer_manifest(
            [current_adapter, scaling_adapter]
        )

        raw = fold_identity.merge(
            current[["game_id", *CURRENT_RATING_SIGNED_MAP_FEATURES]],
            on="game_id",
            how="inner",
            validate="one_to_one",
        ).merge(
            scaling[["game_id", *SCALING_CURVE_SIGNED_MAP_FEATURES]],
            on="game_id",
            how="inner",
            validate="one_to_one",
        )
        if tuple(sorted(raw["game_id"].astype(str))) != fold_ids:
            raise FourVariantBundleError("joined fold feature rows are incomplete")
        variant_inputs = {
            "current_only": (
                CURRENT_RATING_SIGNED_MAP_FEATURES,
                current_manifest,
            ),
            "future_player_form": (
                CURRENT_RATING_SIGNED_MAP_FEATURES,
                current_manifest,
            ),
            "scaling_curve": (
                (*CURRENT_RATING_SIGNED_MAP_FEATURES, *SCALING_CURVE_SIGNED_MAP_FEATURES),
                combined_manifest,
            ),
            "both": (
                (*CURRENT_RATING_SIGNED_MAP_FEATURES, *SCALING_CURVE_SIGNED_MAP_FEATURES),
                combined_manifest,
            ),
        }
        for variant, (feature_names, manifest) in variant_inputs.items():
            bound = bind_rating_feature_ledger(
                raw[["game_id", "date", "series_id", *feature_names]],
                source_receipt=source_receipt,
                train_game_ids=train_ids,
                validation_game_ids=validation_ids,
                fit_window_end=fit_window_end,
                feature_names=feature_names,
                producer=manifest,
            )
            variants[variant]["folds"][str(fold_number)] = {
                "rows": _json_rows(bound),
                "attrs": dict(bound.attrs),
            }
        inner_fold = _derive_inner_fold_spec(
            model_frame,
            outer_fold=fold_number,
            outer_train_ids=train_ids,
            outer_validation_ids=validation_ids,
        )
        inner_records, inner_receipt = _build_inner_fold_artifacts(
            source_receipt_path=source_receipt_path,
            source_receipt=source_receipt,
            maps=maps,
            players=players,
            teams=teams,
            identity=identity,
            fold_number=fold_number,
            inner_fold=inner_fold,
            output_root=nested_root,
        )
        for variant in variants:
            variants[variant]["inner_folds"][str(fold_number)] = inner_records[variant]
        inner_fold_receipts.append(inner_receipt)
        fold_receipts.append(
            {
                "fold": fold_number,
                "fit_window_end": fit_window_end,
                "train_game_count": len(train_ids),
                "train_identity_sha256": identity_sha256(train_ids),
                "validation_game_count": len(validation_ids),
                "validation_identity_sha256": identity_sha256(validation_ids),
                "output_game_count": len(fold_ids),
                "output_identity_sha256": identity_sha256(fold_ids),
                "current_native_receipt_sha256": current_native["receipt_sha256"],
                "scaling_native_receipt_sha256": scaling_native["receipt_sha256"],
            }
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "research_only",
        "source": {
            "source_as_of": source_receipt["source_as_of"],
            "source_game_count": source_receipt["source_game_count"],
            "source_identity_sha256": source_receipt["source_identity_sha256"],
            "model_eligible_game_count": source_receipt["model_eligible_game_count"],
            "model_eligible_identity_sha256": source_receipt[
                "model_eligible_identity_sha256"
            ],
            "source_receipt_sha256": source_receipt["receipt_sha256"],
            "source_receipt_file_sha256": _sha256_path(source_receipt_path),
        },
        "fold_receipts": fold_receipts,
        "nested_inner": {
            "status": "verified",
            "output_root": str(nested_root),
            "fold_count": len(inner_fold_receipts),
            "fold_receipts": inner_fold_receipts,
        },
        "variant_configs": {
            variant.value: rating_variant_config_receipt(variant)
            for variant in RATING_VARIANT_ORDER
        },
        "variants": variants,
        "authority": {
            "research_only": True,
            "public_player_rating": False,
            "public_team_rating": False,
            "public_probability": False,
            "promotion": False,
            "merge": False,
            "deployment": False,
            "odds": False,
            "expected_value": False,
            "recommendation": False,
            "betting": False,
        },
    }
    payload["bundle_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--folds-root", required=True, type=Path)
    parser.add_argument(
        "--inner-output-root",
        type=Path,
        default=None,
        help="Durable empty directory for nested inner producer artifacts",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FourVariantBundleError("bundle output already exists")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise FourVariantBundleError("bundle output parent is unsafe")
    payload = build_bundle(
        source_root=args.source_root.resolve(),
        source_receipt_path=args.source_receipt.resolve(),
        folds_root=args.folds_root.resolve(),
        inner_output_root=(
            None if args.inner_output_root is None else args.inner_output_root.resolve()
        ),
    )
    output.write_bytes(_canonical_bytes(payload))
    print(
        json.dumps(
            {
                "output": str(output),
                "bytes": output.stat().st_size,
                "sha256": _sha256_path(output),
                "bundle_sha256": payload["bundle_sha256"],
                "variants": list(payload["variants"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
