"""Fit and receipt a source-bound V2 future-player-form development model.

The fit uses the frozen accepted source and the existing current-rating
feature ledger.  It emits a model artifact with authority set to false.  The
evaluation blockers are copied into the final receipt.  This keeps an
unevaluated fit from becoming a production rating by accident.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

from lol_kills.ratings.global_player_bt import PrefixBaselineCache
from lol_kills.research.future_value_rating import (
    PREDECLARED_VARIANT_REGULARIZATION_C,
    CURRENT_RATING_SIGNED_MAP_FEATURES,
    FutureValueFoldModel,
    FutureValueSourceError,
    RatingVariant,
    _antisymmetric_design_matrix,
    _canonical_json_bytes,
    _frame_game_ids,
    _fit_zero_intercept_logistic,
    _map_model_frame,
    _role,
    _sha256_path,
    _variant_imputation_values,
    build_future_value_design,
    build_strict_prior_player_form,
    fit_rank3_player_champion_role_atoms,
    rating_feature_values_sha256,
    rating_variant_config,
    validate_future_value_source_receipt_payload,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256
from lol_kills.research.future_value_snapshots import _latest_player_form


SCHEMA_VERSION = "scryglass:future-value-final-fit:v1"
DEFAULT_EVALUATION = Path(
    "/private/tmp/scryglass-four-variant-runs/evaluation-v2/future_player_form/model.json"
)
DEFAULT_CACHE = Path(
    "/private/tmp/scryglass-four-variant-runs/current-ratings/player/player_prefix_baseline_cache.json"
)


class FinalFitError(FutureValueSourceError):
    """The final research fit cannot be bound safely."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FinalFitError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalFitError(f"{label} cannot be read: {path}") from error
    if not isinstance(value, dict):
        raise FinalFitError(f"{label} must be an object")
    return value


def _canonical_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _verify_source_receipt(
    source_receipt: Mapping[str, Any],
    source_receipt_path: Path,
    *,
    expected_source_receipt_sha256: str | None,
) -> None:
    validate_future_value_source_receipt_payload(source_receipt)
    if not expected_source_receipt_sha256:
        raise FinalFitError("independent source receipt file hash is required")
    expected_file_sha = _sha256_path(source_receipt_path)
    expected_file_sha_from_argument = str(expected_source_receipt_sha256).lower()
    if expected_file_sha != expected_file_sha_from_argument:
        raise FinalFitError("source receipt file hash changed")
    root = source_receipt_path.parent.resolve()
    source_files = source_receipt.get("source_files")
    if not isinstance(source_files, Mapping):
        raise FinalFitError("source receipt file bindings are missing")
    for label, record in source_files.items():
        if not isinstance(record, Mapping):
            raise FinalFitError(f"source receipt file binding is invalid: {label}")
        locator = record.get("locator") or record.get("path")
        if not isinstance(locator, str) or not locator.strip():
            raise FinalFitError(f"source receipt file locator is missing: {label}")
        path = Path(locator)
        if path.is_absolute() or ".." in path.parts:
            raise FinalFitError(f"source receipt file locator is unsafe: {label}")
        path = (root / path).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise FinalFitError(f"source receipt file escapes freeze root: {label}") from error
        if path.is_symlink() or not path.is_file():
            raise FinalFitError(f"source receipt file is missing: {label}")
        if int(record.get("bytes") or -1) != path.stat().st_size:
            raise FinalFitError(f"source receipt file bytes changed: {label}")
        if str(record.get("sha256") or "").lower() != _sha256_path(path):
            raise FinalFitError(f"source receipt file hash changed: {label}")


def _verify_source_frames(
    source_root: Path,
    source_receipt: Mapping[str, Any],
) -> None:
    """Verify the exact parquet frames used by the final fit."""

    source_files = source_receipt.get("source_files")
    if not isinstance(source_files, Mapping):
        raise FinalFitError("source receipt frame bindings are missing")
    expected_names = {
        "maps": "maps.parquet",
        "players": "oe_player_games.parquet",
        "teams": "oe_team_games.parquet",
    }
    for label, name in expected_names.items():
        record = source_files.get(label)
        path = source_root / name
        if not isinstance(record, Mapping) or path.is_symlink() or not path.is_file():
            raise FinalFitError(f"frozen source frame is missing: {label}")
        if int(record.get("bytes") or -1) != path.stat().st_size:
            raise FinalFitError(f"frozen source frame bytes changed: {label}")
        if str(record.get("sha256") or "").lower() != _sha256_path(path):
            raise FinalFitError(f"frozen source frame hash changed: {label}")


def _evaluation_blockers(path: Path, source_receipt: Mapping[str, Any]) -> tuple[str, ...]:
    """Load the prior V2 evaluation and bind its exact source identity."""

    payload = _load_json(path, "V2 evaluation")
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise FinalFitError("V2 evaluation source binding is missing")
    for field, expected in (
        ("source_as_of", source_receipt.get("source_as_of")),
        ("source_game_count", source_receipt.get("source_game_count")),
        ("source_identity_sha256", source_receipt.get("source_identity_sha256")),
        ("source_receipt_sha256", source_receipt.get("receipt_sha256")),
    ):
        if source.get(field) != expected:
            raise FinalFitError(f"V2 evaluation source binding changed: {field}")
    variant = payload.get("variants", {}).get("future_player_form")
    if not isinstance(variant, Mapping):
        raise FinalFitError("V2 future-player-form evaluation is missing")
    blockers = variant.get("blockers")
    if not isinstance(blockers, list):
        raise FinalFitError("V2 evaluation blocker list is missing")
    return tuple(sorted({str(value) for value in blockers}))


def _source_frame_sha256(path: Path) -> str:
    return _sha256_path(path)


def _bind_current_rating_features(
    current_frame: pd.DataFrame,
    current_receipt: Mapping[str, Any],
    *,
    source_receipt: Mapping[str, Any],
    source_receipt_path: Path,
    current_artifact_path: Path,
    current_receipt_path: Path,
    implementation_path: Path,
    fit_game_ids: tuple[str, ...],
    fit_window_end: str,
    expected_current_receipt_sha256: str | None,
    expected_current_artifact_sha256: str | None,
) -> dict[str, Any]:
    expected_receipt_file_hash = str(expected_current_receipt_sha256 or "").lower()
    expected_artifact_hash = str(expected_current_artifact_sha256 or "").lower()
    if re.fullmatch(r"[0-9a-f]{64}", expected_receipt_file_hash) is None:
        raise FinalFitError("independent current rating receipt hash is required")
    if re.fullmatch(r"[0-9a-f]{64}", expected_artifact_hash) is None:
        raise FinalFitError("independent current rating artifact hash is required")
    if (
        current_receipt_path.is_symlink()
        or not current_receipt_path.is_file()
        or _sha256_path(current_receipt_path) != expected_receipt_file_hash
    ):
        raise FinalFitError("current rating receipt file changed")
    if (
        current_artifact_path.is_symlink()
        or not current_artifact_path.is_file()
        or _sha256_path(current_artifact_path) != expected_artifact_hash
    ):
        raise FinalFitError("current rating artifact file changed")
    claimed_receipt_hash = str(current_receipt.get("receipt_sha256") or "").lower()
    receipt_payload = dict(current_receipt)
    receipt_payload.pop("receipt_sha256", None)
    if len(claimed_receipt_hash) != 64 or _canonical_sha(receipt_payload) != claimed_receipt_hash:
        raise FinalFitError("current rating receipt self-hash changed")
    if current_receipt.get("source_receipt_sha256") != source_receipt.get("receipt_sha256"):
        raise FinalFitError("current rating ledger source receipt changed")
    if current_receipt.get("source_identity_sha256") != source_receipt.get(
        "source_identity_sha256"
    ):
        raise FinalFitError("current rating ledger source identity changed")
    expected_artifact = current_receipt.get("artifact_sha256")
    if expected_artifact != _sha256_path(current_artifact_path):
        raise FinalFitError("current rating ledger artifact changed")
    if int(current_receipt.get("rows") or -1) != len(current_frame):
        raise FinalFitError("current rating ledger row count changed")
    required = {"game_id", "date", "series_id", *CURRENT_RATING_SIGNED_MAP_FEATURES}
    if not required.issubset(current_frame.columns):
        raise FinalFitError("current rating ledger feature schema is incomplete")
    ids = tuple(sorted(current_frame["game_id"].astype(str)))
    if ids != fit_game_ids:
        raise FinalFitError("current rating ledger fit census changed")
    values = current_frame[list(CURRENT_RATING_SIGNED_MAP_FEATURES)].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise FinalFitError("current rating ledger contains non-finite values")
    value_digest = rating_feature_values_sha256(
        current_frame, CURRENT_RATING_SIGNED_MAP_FEATURES
    )
    declared_value_digest = str(current_receipt.get("feature_value_digest") or "").lower()
    if re.fullmatch(r"[0-9a-f]{64}", declared_value_digest) is None:
        raise FinalFitError("current rating receipt feature-value digest is required")
    if declared_value_digest != value_digest:
        raise FinalFitError("current rating receipt feature-value digest changed")
    return {
        "schema_version": "scryglass:future-value-final-current-rating-binding:v1",
        "producer_name": "current_sequential_rating",
        "producer_receipt_sha256": str(current_receipt.get("receipt_sha256") or ""),
        "producer_receipt_schema_version": str(current_receipt.get("schema_version") or ""),
        "producer_receipt_self_hash_verified": True,
        "producer_receipt_file": {
            "path": str(current_receipt_path),
            "bytes": current_receipt_path.stat().st_size,
            "sha256": expected_receipt_file_hash,
        },
        "source_receipt_sha256": str(source_receipt["receipt_sha256"]),
        "source_identity_sha256": str(source_receipt["source_identity_sha256"]),
        "source_receipt_file": {
            "path": str(source_receipt_path),
            "bytes": source_receipt_path.stat().st_size,
            "sha256": _sha256_path(source_receipt_path),
        },
        "artifact": {
            "path": str(current_artifact_path),
            "bytes": current_artifact_path.stat().st_size,
            "sha256": expected_artifact_hash,
        },
        "feature_names": list(CURRENT_RATING_SIGNED_MAP_FEATURES),
        "feature_value_digest": value_digest,
        "fit_game_ids": list(fit_game_ids),
        "fit_game_identity_sha256": identity_sha256(fit_game_ids),
        "fit_window_end": fit_window_end,
        "evaluation_mode": "final_fit",
        "strict_prior_timing": "source_bound_current_rating_before_snapshot_as_of",
        "same_timestamp_policy": "score_full_utc_timestamp_batch_before_training_updates",
        "code": {
            "implementation_locator": "lol_kills/research/future_value_rating_ledger.py",
            "implementation_sha256": _sha256_path(implementation_path),
        },
        "rows": len(current_frame),
        "game_identity_sha256": identity_sha256(ids),
    }


def fit_final_v2(
    *,
    source_root: Path,
    source_receipt_path: Path,
    current_root: Path,
    evaluation_path: Path,
    output_dir: Path,
    baseline_cache_path: Path | None = None,
    expected_source_receipt_sha256: str | None = None,
    expected_current_receipt_sha256: str | None = None,
    expected_current_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    source_receipt = _load_json(source_receipt_path, "source receipt")
    _verify_source_receipt(
        source_receipt,
        source_receipt_path,
        expected_source_receipt_sha256=expected_source_receipt_sha256,
    )
    evaluation_blockers = _evaluation_blockers(evaluation_path, source_receipt)
    eligible_ids = tuple(sorted(str(value) for value in source_receipt["model_eligible_game_ids"]))
    source_root = source_root.resolve()
    _verify_source_frames(source_root, source_receipt)
    source_maps_path = source_root / "maps.parquet"
    source_players_path = source_root / "oe_player_games.parquet"
    source_teams_path = source_root / "oe_team_games.parquet"
    maps = pd.read_parquet(source_maps_path)
    players = pd.read_parquet(source_players_path)
    teams = pd.read_parquet(source_teams_path)
    model_frame = _map_model_frame(maps)
    model_frame = model_frame[model_frame["game_id"].astype(str).isin(eligible_ids)].copy()
    if tuple(sorted(model_frame["game_id"].astype(str))) != eligible_ids:
        raise FinalFitError("model maps do not match the frozen eligible census")

    current_artifact_path = current_root / "current-rating-ledger.parquet"
    current_receipt_path = current_root / "current-rating-ledger-receipt.json"
    current_frame = pd.read_parquet(current_artifact_path)
    current_receipt = _load_json(current_receipt_path, "current rating ledger receipt")
    fit_window_end = str(source_receipt["source_as_of"])
    binding = _bind_current_rating_features(
        current_frame,
        current_receipt,
        source_receipt=source_receipt,
        source_receipt_path=source_receipt_path,
        current_artifact_path=current_artifact_path,
        current_receipt_path=current_receipt_path,
        implementation_path=Path(__file__).resolve().parents[1]
        / "lol_kills/research/future_value_rating_ledger.py",
        fit_game_ids=eligible_ids,
        fit_window_end=fit_window_end,
        expected_current_receipt_sha256=expected_current_receipt_sha256,
        expected_current_artifact_sha256=expected_current_artifact_sha256,
    )

    started = time.perf_counter()
    form_cache = None
    if baseline_cache_path is not None and baseline_cache_path.is_file():
        manifest = _load_json(baseline_cache_path, "baseline cache manifest")
        form_cache = PrefixBaselineCache(
            storage_path=baseline_cache_path,
            source_identity=str(manifest.get("source_identity") or ""),
            schema_fingerprint=str(manifest.get("schema_fingerprint") or ""),
        )
    # This is the role-normalized strict-prior form contract.  The cache is
    # read-only for the final fit and is never flushed by this command.
    normalized_players = players.copy()
    normalized_players["game_id"] = _frame_game_ids(normalized_players, "players").astype(str)
    normalized_players["player_id"] = normalized_players["playerid"].astype(str)
    normalized_players["team_id"] = normalized_players["teamid"].astype(str)
    normalized_players["side"] = normalized_players["side"].astype(str).str.casefold()
    normalized_players["role"] = normalized_players["position"].map(_role)
    normalized_players["date"] = pd.to_datetime(
        normalized_players.get("date"), utc=True, errors="coerce"
    )
    form = _latest_player_form(
        maps,
        normalized_players,
        baseline_cache=form_cache,
    )
    # The evaluator design expects an effective-support field.  The robust
    # role-normalized baseline exposes an integer prior count, which is the
    # effective support for this final snapshot contract.
    for metric in (
        "cs_per_min",
        "gold_per_min",
        "gold_share_pct",
        "damage_per_min",
        "damage_share_pct",
        "kda_role_weighted",
        "wards_per_min",
        "wards_cleared_per_min",
    ):
        support = f"prior_form_{metric}_support"
        effective = f"prior_form_{metric}_effective_support"
        if support not in form.columns:
            raise FinalFitError(f"strict-prior support field is missing: {support}")
        form[effective] = pd.to_numeric(form[support], errors="coerce")
    form = form[form["game_id"].astype(str).isin(eligible_ids)].copy()
    if set(form["game_id"].astype(str)) != set(eligible_ids):
        raise FinalFitError("strict-prior form does not cover the eligible census")
    atom_model = fit_rank3_player_champion_role_atoms(
        form,
        train_game_ids=eligible_ids,
        # The source snapshot is inclusive.  Strict-prior player features keep
        # the boundary map out of its own history, while the final fit may use
        # its observed target as the last historical training outcome.
        fit_window_end=None,
    )
    design = build_future_value_design(
        model_frame,
        form,
        atom_model,
        verified_model_frame=model_frame,
    )
    current_join = current_frame[
        ["game_id", "date", "series_id", *CURRENT_RATING_SIGNED_MAP_FEATURES]
    ].copy()
    current_join["game_id"] = current_join["game_id"].astype(str)
    current_join["date"] = pd.to_datetime(current_join["date"], utc=True, errors="coerce")
    current_join = current_join.sort_values("game_id", kind="stable")
    design = design.merge(
        current_join[["game_id", *CURRENT_RATING_SIGNED_MAP_FEATURES]],
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    if design[list(CURRENT_RATING_SIGNED_MAP_FEATURES)].isna().any().any():
        raise FinalFitError("current rating feature join is incomplete")
    config = rating_variant_config(RatingVariant.FUTURE_PLAYER_FORM)
    imputation = _variant_imputation_values(design, config)
    matrix = _antisymmetric_design_matrix(
        design,
        imputation,
        feature_names=config.feature_names,
    )
    scales = matrix.std(axis=0, ddof=0)
    scales = np.where(np.isfinite(scales) & (scales > 1e-12), scales, 1.0)
    target = pd.to_numeric(design["target"], errors="coerce")
    if not target.isin({0, 1}).all() or target.nunique() != 2:
        raise FinalFitError("final fit target is incomplete")
    classifier, optimizer = _fit_zero_intercept_logistic(
        matrix / scales,
        target.to_numpy(dtype=int),
        regularization_c=PREDECLARED_VARIANT_REGULARIZATION_C,
    )
    source_frame_hashes = {
        "maps": _source_frame_sha256(source_maps_path),
        "players": _source_frame_sha256(source_players_path),
        "teams": _source_frame_sha256(source_teams_path),
    }
    ledger_binding = {
        **binding,
        "source_frame_sha256": source_frame_hashes,
        "producer_names": ["current_sequential_rating"],
        "validation_game_ids": [],
        "validation_game_identity_sha256": identity_sha256(()),
        "fit_date_min": str(current_frame["date"].min()),
        "fit_date_max": str(current_frame["date"].max()),
    }
    model = FutureValueFoldModel(
        feature_names=tuple(config.feature_names),
        means=np.zeros(matrix.shape[1], dtype=float),
        scales=scales.astype(float),
        imputation_values=np.asarray(imputation, dtype=float),
        coefficients=classifier.coef_[0].astype(float),
        intercept=0.0,
        regularization_selection={
            "method": "predeclared_regularization_c",
            "candidate_grid": [float(PREDECLARED_VARIANT_REGULARIZATION_C)],
            "selected_c": float(PREDECLARED_VARIANT_REGULARIZATION_C),
            "inner_ledger_status": "missing",
            "blockers": ["nested_inner_feature_ledger_missing_fixed_c_used"],
            "selection_scope": "predeclared_before_final_fit",
        },
        optimizer_evidence=optimizer,
        atom_model=atom_model,
        fit_game_ids=eligible_ids,
        fit_window_end=fit_window_end,
        train_rows=len(design),
        withheld_rows=0,
        source_receipt=dict(source_receipt),
        variant=RatingVariant.FUTURE_PLAYER_FORM,
        feature_ledger_binding=ledger_binding,
    )
    parameters = model.parameter_receipt()
    blockers = set(evaluation_blockers)
    blockers.update(model.regularization_selection.get("blockers", ()))
    blockers.update(
        {
            "final_calibration_receipt_missing",
            "authoritative_series_id_missing_proxy_cluster_used",
            "current_rating_player_team_identity_missing_for_rank_diffs",
        }
    )
    receipt = model.receipt()
    receipt.update(
        {
            "schema_version": "scryglass:future-value-model-fit:v1",
            "status": "research_only_blocked",
            "fit_contract_schema_version": SCHEMA_VERSION,
            "fit_game_ids": list(eligible_ids),
            "fit_game_identity_sha256": identity_sha256(eligible_ids),
            "fit_window_end": fit_window_end,
            "form_contract": "strict_prior_role_and_competition_tier_normalized_v1",
            "current_rating_feature_binding": binding,
            "source_frame_sha256": source_frame_hashes,
            "code_binding": {
                "future_value_rating_sha256": _sha256_path(
                    Path(__file__).resolve().parents[1]
                    / "lol_kills/research/future_value_rating.py"
                ),
                "future_value_rating_ledger_sha256": binding["code"]["implementation_sha256"],
                "snapshot_producer_sha256": _sha256_path(
                    Path(__file__).resolve().parents[1]
                    / "lol_kills/research/future_value_snapshots.py"
                ),
            },
            "transformation_binding": {
                "variant": RatingVariant.FUTURE_PLAYER_FORM.value,
                "variant_receipt": config.receipt(),
                "feature_names": list(config.feature_names),
                "feature_schema_sha256": _canonical_sha(config.receipt()),
                "imputation_values": [float(value) for value in imputation],
                "scales": [float(value) for value in scales],
                "rank_3_parameter_sha256": parameters["rank_3"]["parameter_sha256"],
                "parameter_sha256": parameters["parameter_sha256"],
            },
            "optimizer_convergence": optimizer,
            "evaluation_receipt": {
                "path": str(evaluation_path),
                "bytes": evaluation_path.stat().st_size,
                "sha256": _sha256_path(evaluation_path),
                "variant": RatingVariant.FUTURE_PLAYER_FORM.value,
                "blockers": list(evaluation_blockers),
            },
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
            "blockers": sorted(blockers),
            "fit_elapsed_seconds": time.perf_counter() - started,
        }
    )
    receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = _canonical_sha(receipt)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FinalFitError(f"final-fit output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    model_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": receipt["status"],
        "authority": receipt["authority"],
        "source": receipt["source_binding"],
        "receipt_sha256": receipt["receipt_sha256"],
        "receipt": receipt,
        "parameters": parameters,
    }
    (output_dir / "final-v2-model.json").write_text(
        json.dumps(model_payload, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "final-v2-model-receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    run = {
        "schema_version": SCHEMA_VERSION,
        "status": receipt["status"],
        "source_receipt_sha256": source_receipt["receipt_sha256"],
        "source_identity_sha256": source_receipt["source_identity_sha256"],
        "fit_game_count": len(eligible_ids),
        "fit_game_identity_sha256": identity_sha256(eligible_ids),
        "fit_window_end": fit_window_end,
        "model_receipt_sha256": receipt["receipt_sha256"],
        "model_artifact_sha256": _sha256_path(output_dir / "final-v2-model.json"),
        "blockers": sorted(blockers),
        "authority": receipt["authority"],
    }
    (output_dir / "final-fit-run.json").write_text(
        json.dumps(run, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--current-root", required=True, type=Path)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--baseline-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--source-receipt-sha256", required=True)
    parser.add_argument("--current-receipt-sha256", required=True)
    parser.add_argument("--current-artifact-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run = fit_final_v2(
        source_root=args.source_root,
        source_receipt_path=args.source_receipt,
        current_root=args.current_root,
        evaluation_path=args.evaluation,
        output_dir=args.output_dir,
        baseline_cache_path=args.baseline_cache,
        expected_source_receipt_sha256=args.source_receipt_sha256,
        expected_current_receipt_sha256=args.current_receipt_sha256,
        expected_current_artifact_sha256=args.current_artifact_sha256,
    )
    print(json.dumps(run, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
