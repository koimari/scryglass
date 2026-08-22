"""Build a source-bound full-census V2 Tier List shadow offset ledger.

The ledger uses the final research-only V2 player-form model.  Every input
feature is the state available before its map.  The fitted coefficients use
the complete model-eligible census, so this artifact is retrospective.  It is
not a chronological evaluation result and cannot grant public authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from lol_kills.research.future_value_rating import (
    CURRENT_RATING_SIGNED_MAP_FEATURES,
    FORM_METRICS,
    FutureValueSourceError,
    Rank3AtomModel,
    RatingVariant,
    _antisymmetric_design_matrix,
    _frame_game_ids,
    _map_model_frame,
    _role,
    build_future_value_design,
    rating_feature_values_sha256,
    rating_variant_config,
    validate_future_value_source_receipt_payload,
)
from lol_kills.research.future_value_snapshots import _latest_player_form
from lol_kills.research.future_value_tierlist import (
    make_offset_provenance,
    offset_values_sha256,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


LEDGER_SCHEMA_VERSION = "scryglass:future-value-v2-tier-offset-ledger:v1"
RECEIPT_SCHEMA_VERSION = "scryglass:future-value-v2-tier-offset-receipt:v1"
FINAL_MODEL_SCHEMA_VERSION = "scryglass:future-value-final-fit:v1"
AUTHORITY = {
    "research_only": True,
    "public_tierlist": False,
    "public_player_rating": False,
    "public_team_rating": False,
    "public_probability": False,
    "promotion": False,
    "merge": False,
    "deployment": False,
    "betting": False,
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


class V2TierShadowError(FutureValueSourceError):
    """The V2 Tier shadow cannot prove its source or timing contract."""


@dataclass(frozen=True)
class V2TierShadowResult:
    ledger_path: Path
    receipt_path: Path
    receipt: Mapping[str, Any]
    offsets: Mapping[str, float]
    provenance: Mapping[str, Any]
    game_ids: tuple[str, ...]


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_hash(value: object, label: str) -> str:
    text = str(value or "").lower()
    if _SHA256.fullmatch(text) is None:
        raise V2TierShadowError(f"{label} must be a SHA-256 hash")
    return text


def _file(path: Path | str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise V2TierShadowError(f"{label} is missing or unsafe: {resolved}")
    return resolved


def _load_json(path: Path | str, label: str) -> tuple[dict[str, Any], Path]:
    file_path = _file(path, label)
    try:
        value = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V2TierShadowError(f"{label} cannot be read") from error
    if not isinstance(value, dict):
        raise V2TierShadowError(f"{label} must be a JSON object")
    return value, file_path


def _verify_file(path: Path, expected_sha256: object, label: str) -> str:
    expected = _require_hash(expected_sha256, f"expected {label} hash")
    if sha256_path(path) != expected:
        raise V2TierShadowError(f"{label} bytes changed")
    return expected


def _verify_authority(value: object, label: str) -> None:
    if not isinstance(value, Mapping) or value.get("research_only") is not True:
        raise V2TierShadowError(f"{label} research authority is missing")
    if any(bool(flag) for name, flag in value.items() if name != "research_only"):
        raise V2TierShadowError(f"{label} grants authority")


def _verify_self_hash(payload: Mapping[str, Any], field: str, label: str) -> str:
    claimed = _require_hash(payload.get(field), f"{label} {field}")
    unsigned = dict(payload)
    unsigned.pop(field, None)
    if _canonical_sha256(unsigned) != claimed:
        raise V2TierShadowError(f"{label} self-hash changed")
    return claimed


def _source_record_path(
    source_root: Path,
    receipt: Mapping[str, Any],
    label: str,
) -> Path:
    records = receipt.get("source_files")
    record = records.get(label) if isinstance(records, Mapping) else None
    if not isinstance(record, Mapping):
        raise V2TierShadowError(f"source file record is missing: {label}")
    locator = str(record.get("locator") or record.get("path") or "").strip()
    relative = Path(locator)
    if not locator or relative.is_absolute() or ".." in relative.parts:
        raise V2TierShadowError(f"source file locator is unsafe: {label}")
    candidate = (source_root / relative).resolve()
    try:
        candidate.relative_to(source_root)
    except ValueError as error:
        raise V2TierShadowError(f"source file escapes freeze root: {label}") from error
    candidate = _file(candidate, f"source file {label}")
    if int(record.get("bytes") or -1) != candidate.stat().st_size:
        raise V2TierShadowError(f"source file byte count changed: {label}")
    _verify_file(candidate, record.get("sha256"), f"source file {label}")
    return candidate


def verify_source_freeze(
    source_root: Path | str,
    source_receipt_path: Path | str,
    *,
    expected_source_receipt_file_sha256: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Verify the canonical receipt and frozen maps, player, and team files."""

    root = Path(source_root).expanduser().resolve()
    receipt, receipt_path = _load_json(source_receipt_path, "source receipt")
    _verify_file(
        receipt_path,
        expected_source_receipt_file_sha256,
        "source receipt",
    )
    validate_future_value_source_receipt_payload(receipt)
    paths = {
        label: _source_record_path(root, receipt, label)
        for label in ("maps", "players", "teams")
    }
    return receipt, paths


def verify_final_v2_model(
    model_path: Path | str,
    model_receipt_path: Path | str,
    run_receipt_path: Path | str,
    *,
    expected_model_sha256: str,
    expected_model_receipt_file_sha256: str,
    expected_run_receipt_sha256: str,
    source_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify the final model, its parameter receipt, and its run receipt."""

    model, model_file = _load_json(model_path, "final V2 model")
    receipt, receipt_file = _load_json(model_receipt_path, "final V2 model receipt")
    run, run_file = _load_json(run_receipt_path, "final V2 run receipt")
    model_sha = _verify_file(model_file, expected_model_sha256, "final V2 model")
    _verify_file(
        receipt_file,
        expected_model_receipt_file_sha256,
        "final V2 model receipt",
    )
    _verify_file(run_file, expected_run_receipt_sha256, "final V2 run receipt")
    if model.get("schema_version") != FINAL_MODEL_SCHEMA_VERSION:
        raise V2TierShadowError("final V2 model schema changed")
    if run.get("schema_version") != FINAL_MODEL_SCHEMA_VERSION:
        raise V2TierShadowError("final V2 run schema changed")
    if model.get("status") != "research_only_blocked" or run.get("status") != "research_only_blocked":
        raise V2TierShadowError("final V2 model status changed")
    _verify_authority(model.get("authority"), "final V2 model")
    _verify_authority(receipt.get("authority"), "final V2 receipt")
    _verify_authority(run.get("authority"), "final V2 run")
    receipt_hash = _verify_self_hash(receipt, "receipt_sha256", "final V2 receipt")
    if model.get("receipt_sha256") != receipt_hash or model.get("receipt") != receipt:
        raise V2TierShadowError("final V2 embedded receipt changed")
    if run.get("model_receipt_sha256") != receipt_hash:
        raise V2TierShadowError("final V2 run receipt binding changed")
    if run.get("model_artifact_sha256") != model_sha:
        raise V2TierShadowError("final V2 run model binding changed")
    for field in ("source_receipt_sha256", "source_identity_sha256"):
        if run.get(field) != source_receipt.get(
            "receipt_sha256" if field == "source_receipt_sha256" else field
        ):
            raise V2TierShadowError(f"final V2 source binding changed: {field}")
    eligible = tuple(sorted(str(value) for value in source_receipt["model_eligible_game_ids"]))
    if (
        tuple(str(value) for value in receipt.get("fit_game_ids", ())) != eligible
        or run.get("fit_game_count") != len(eligible)
        or run.get("fit_game_identity_sha256") != identity_sha256(eligible)
    ):
        raise V2TierShadowError("final V2 fit census changed")
    if run.get("fit_window_end") != source_receipt.get("source_as_of"):
        raise V2TierShadowError("final V2 fit cutoff changed")
    parameters = model.get("parameters")
    if not isinstance(parameters, Mapping):
        raise V2TierShadowError("final V2 parameters are missing")
    parameter_hash = _verify_self_hash(
        parameters, "parameter_sha256", "final V2 parameters"
    )
    if receipt.get("parameter_sha256") != parameter_hash:
        raise V2TierShadowError("final V2 parameter receipt changed")
    if parameters.get("variant") != RatingVariant.FUTURE_PLAYER_FORM.value:
        raise V2TierShadowError("final V2 variant changed")
    config = rating_variant_config(RatingVariant.FUTURE_PLAYER_FORM)
    if tuple(parameters.get("feature_names", ())) != tuple(config.feature_names):
        raise V2TierShadowError("final V2 feature schema changed")
    rank_3 = parameters.get("rank_3")
    if not isinstance(rank_3, Mapping):
        raise V2TierShadowError("final V2 rank-3 parameters are missing")
    _verify_self_hash(rank_3, "parameter_sha256", "final V2 rank-3 parameters")
    return model, receipt, run


def verify_current_rating_ledger(
    ledger_path: Path | str,
    receipt_path: Path | str,
    *,
    expected_ledger_sha256: str,
    expected_receipt_file_sha256: str,
    source_receipt: Mapping[str, Any],
    final_model_receipt: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Verify the strict-prior current-rating features used by the final fit."""

    ledger_file = _file(ledger_path, "current-rating ledger")
    receipt, receipt_file = _load_json(receipt_path, "current-rating receipt")
    ledger_sha = _verify_file(
        ledger_file, expected_ledger_sha256, "current-rating ledger"
    )
    _verify_file(
        receipt_file,
        expected_receipt_file_sha256,
        "current-rating receipt",
    )
    _verify_self_hash(receipt, "receipt_sha256", "current-rating receipt")
    binding = final_model_receipt.get("current_rating_feature_binding")
    if not isinstance(binding, Mapping):
        raise V2TierShadowError("final V2 current-rating binding is missing")
    artifact = binding.get("artifact")
    if not isinstance(artifact, Mapping) or artifact.get("sha256") != ledger_sha:
        raise V2TierShadowError("final V2 current-rating artifact changed")
    if receipt.get("artifact_sha256") != ledger_sha:
        raise V2TierShadowError("current-rating receipt artifact changed")
    for field in ("source_receipt_sha256", "source_identity_sha256"):
        expected = source_receipt.get(
            "receipt_sha256" if field == "source_receipt_sha256" else field
        )
        if receipt.get(field) != expected or binding.get(field) != expected:
            raise V2TierShadowError(f"current-rating source changed: {field}")
    if binding.get("strict_prior_timing") != "source_bound_current_rating_before_snapshot_as_of":
        raise V2TierShadowError("current-rating strict-prior timing changed")
    if binding.get("same_timestamp_policy") != "score_full_utc_timestamp_batch_before_training_updates":
        raise V2TierShadowError("current-rating timestamp policy changed")
    try:
        frame = pd.read_parquet(ledger_file)
    except (OSError, ValueError) as error:
        raise V2TierShadowError("current-rating ledger cannot be read") from error
    required = {"game_id", "date", *CURRENT_RATING_SIGNED_MAP_FEATURES}
    if not required.issubset(frame.columns) or frame["game_id"].astype(str).duplicated().any():
        raise V2TierShadowError("current-rating ledger schema changed")
    eligible = tuple(sorted(str(value) for value in source_receipt["model_eligible_game_ids"]))
    if tuple(sorted(frame["game_id"].astype(str))) != eligible:
        raise V2TierShadowError("current-rating ledger coverage changed")
    if tuple(str(value) for value in binding.get("fit_game_ids", ())) != eligible:
        raise V2TierShadowError("current-rating fit census changed")
    values = frame[list(CURRENT_RATING_SIGNED_MAP_FEATURES)].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise V2TierShadowError("current-rating ledger contains non-finite features")
    digest = rating_feature_values_sha256(frame, CURRENT_RATING_SIGNED_MAP_FEATURES)
    if binding.get("feature_value_digest") != digest or receipt.get("feature_value_digest") != digest:
        raise V2TierShadowError("current-rating feature values changed")
    return frame, receipt


def _rank_3_model(parameters: Mapping[str, Any]) -> Rank3AtomModel:
    rank = parameters["rank_3"]
    try:
        model = Rank3AtomModel(
            metric_names=tuple(str(value) for value in rank["metric_names"]),
            rank=int(rank["rank"]),
            center=np.asarray(rank["center"], dtype=float),
            scale=np.asarray(rank["scale"], dtype=float),
            components=np.asarray(rank["components"], dtype=float),
            champion_role_coordinates={
                str(key): tuple(float(value) for value in values)
                for key, values in rank["champion_role_coordinates"].items()
            },
            champion_role_support={
                str(key): int(value)
                for key, value in rank["champion_role_support"].items()
            },
            fit_game_ids=tuple(str(value) for value in rank["fit_game_ids"]),
            fit_window_end=str(rank["fit_window_end"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise V2TierShadowError("final V2 rank-3 parameters are invalid") from error
    if model.parameter_receipt().get("parameter_sha256") != rank.get("parameter_sha256"):
        raise V2TierShadowError("final V2 rank-3 reconstruction changed")
    return model


def build_v2_design(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    current_rating: pd.DataFrame,
    *,
    eligible_game_ids: Sequence[str],
    parameters: Mapping[str, Any],
) -> pd.DataFrame:
    """Rebuild the final-fit strict-prior feature design for every map."""

    eligible = tuple(sorted(str(value) for value in eligible_game_ids))
    normalized = players.copy()
    normalized["game_id"] = _frame_game_ids(normalized, "players").astype(str)
    normalized["player_id"] = normalized["playerid"].astype(str)
    normalized["team_id"] = normalized["teamid"].astype(str)
    normalized["side"] = normalized["side"].astype(str).str.casefold()
    normalized["role"] = normalized["position"].map(_role)
    normalized["date"] = pd.to_datetime(normalized.get("date"), utc=True, errors="coerce")
    form = _latest_player_form(maps, normalized)
    for metric in FORM_METRICS:
        support = f"prior_form_{metric}_support"
        effective = f"prior_form_{metric}_effective_support"
        if support not in form.columns:
            raise V2TierShadowError(f"strict-prior form support is missing: {support}")
        form[effective] = pd.to_numeric(form[support], errors="coerce")
    form = form[form["game_id"].astype(str).isin(eligible)].copy()
    if set(form["game_id"].astype(str)) != set(eligible):
        raise V2TierShadowError("strict-prior form coverage changed")
    model_frame = _map_model_frame(maps)
    model_frame = model_frame[model_frame["game_id"].astype(str).isin(eligible)].copy()
    if tuple(sorted(model_frame["game_id"].astype(str))) != eligible:
        raise V2TierShadowError("frozen maps do not cover the V2 census")
    design = build_future_value_design(
        model_frame,
        form,
        _rank_3_model(parameters),
        verified_model_frame=model_frame,
    )
    signed = current_rating[["game_id", *CURRENT_RATING_SIGNED_MAP_FEATURES]].copy()
    signed["game_id"] = signed["game_id"].astype(str)
    design = design.merge(signed, on="game_id", how="left", validate="one_to_one")
    if design[list(CURRENT_RATING_SIGNED_MAP_FEATURES)].isna().any().any():
        raise V2TierShadowError("current-rating feature join is incomplete")
    if tuple(sorted(design["game_id"].astype(str))) != eligible:
        raise V2TierShadowError("V2 design coverage changed")
    return design


def score_v2_design(
    design: pd.DataFrame,
    parameters: Mapping[str, Any],
    *,
    expected_game_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, float], str]:
    """Score a verified design and return canonical Tier offset rows."""

    feature_names = tuple(str(value) for value in parameters.get("feature_names", ()))
    if feature_names != tuple(
        rating_variant_config(RatingVariant.FUTURE_PLAYER_FORM).feature_names
    ):
        raise V2TierShadowError("V2 score feature schema changed")
    try:
        imputation_map = parameters["fold_local_side_imputation"]
        scale_map = parameters["feature_scales"]
        coefficient_map = parameters["coefficients"]
        imputation = np.asarray([imputation_map[name] for name in feature_names], dtype=float)
        scales = np.asarray([scale_map[name] for name in feature_names], dtype=float)
        coefficients = np.asarray([coefficient_map[name] for name in feature_names], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise V2TierShadowError("V2 score parameters are incomplete") from error
    if (
        not np.isfinite(imputation).all()
        or not np.isfinite(scales).all()
        or not np.isfinite(coefficients).all()
        or (scales <= 0.0).any()
        or float(parameters.get("intercept", math.nan)) != 0.0
    ):
        raise V2TierShadowError("V2 score parameters are invalid")
    matrix = _antisymmetric_design_matrix(
        design,
        imputation,
        feature_names=feature_names,
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        logits = (matrix / scales) @ coefficients
    if not np.isfinite(logits).all():
        raise V2TierShadowError("V2 score produced non-finite logits")
    work = design[["game_id", "date", "target"]].copy()
    work["game_id"] = work["game_id"].astype(str)
    if work["game_id"].duplicated().any():
        raise V2TierShadowError("V2 design contains duplicate game IDs")
    expected = tuple(sorted(str(value) for value in expected_game_ids))
    if tuple(sorted(work["game_id"])) != expected:
        raise V2TierShadowError("V2 score census changed")
    dates = pd.to_datetime(work["date"], utc=True, errors="coerce")
    targets = pd.to_numeric(work["target"], errors="coerce")
    if dates.isna().any() or not targets.isin({0, 1}).all():
        raise V2TierShadowError("V2 score date or target is invalid")
    work["date"] = dates
    work["target"] = targets.astype(int)
    work["v2_offset_logit"] = logits
    work = work.sort_values(["date", "game_id"], kind="mergesort")
    rows = [
        {
            "game_id": str(row.game_id),
            "date": pd.Timestamp(row.date).isoformat().replace("+00:00", "Z"),
            "target": int(row.target),
            "v2_offset_logit": float(row.v2_offset_logit),
        }
        for row in work.itertuples(index=False)
    ]
    offsets = {row["game_id"]: row["v2_offset_logit"] for row in rows}
    matrix_rows = [
        {
            "game_id": game_id,
            "values": [float(value) for value in matrix[index]],
        }
        for index, game_id in enumerate(design["game_id"].astype(str))
    ]
    return rows, offsets, _canonical_sha256(matrix_rows)


def _target_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return _canonical_sha256(
        [
            {"game_id": str(row["game_id"]), "target": int(row["target"])}
            for row in sorted(rows, key=lambda item: str(item["game_id"]))
        ]
    )


def verify_target_parity(
    rows: Sequence[Mapping[str, Any]],
    maps: pd.DataFrame,
    *,
    expected_game_ids: Sequence[str],
) -> str:
    """Verify every offset target against the frozen map outcome."""

    model_frame = _map_model_frame(maps)
    target_by_id = model_frame.set_index(model_frame["game_id"].astype(str))["target"]
    expected = tuple(sorted(str(value) for value in expected_game_ids))
    if target_by_id.index.duplicated().any() or not set(expected).issubset(target_by_id.index):
        raise V2TierShadowError("frozen target identities are incomplete")
    for row in rows:
        game_id = str(row["game_id"])
        if float(row["target"]) != float(target_by_id.loc[game_id]):
            raise V2TierShadowError(f"V2 target changed: {game_id}")
    return _target_rows_sha256(rows)


def write_v2_tier_offset_ledger(
    destination: Path | str,
    *,
    rows: Sequence[Mapping[str, Any]],
    offsets: Mapping[str, float],
    design_matrix_sha256: str,
    source_receipt: Mapping[str, Any],
    source_receipt_file: Path,
    source_files: Mapping[str, Path],
    model_file: Path,
    model_receipt_file: Path,
    run_receipt_file: Path,
    current_ledger_file: Path,
    current_receipt_file: Path,
    target_rows_sha256: str,
) -> V2TierShadowResult:
    """Write the immutable research ledger and its compact sidecar receipt."""

    path = Path(destination).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise V2TierShadowError(f"V2 Tier ledger destination exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    game_ids = tuple(sorted(str(row["game_id"]) for row in rows))
    eligible = tuple(sorted(str(value) for value in source_receipt["model_eligible_game_ids"]))
    if game_ids != eligible or set(offsets) != set(eligible):
        raise V2TierShadowError("V2 Tier ledger does not cover the full eligible census")
    rows_hash = _canonical_sha256(list(rows))
    offsets_hash = offset_values_sha256(offsets)
    bindings = {
        "source_receipt": {
            "path": str(source_receipt_file),
            "bytes": source_receipt_file.stat().st_size,
            "sha256": sha256_path(source_receipt_file),
            "receipt_sha256": source_receipt["receipt_sha256"],
        },
        "source_files": {
            label: {
                "path": str(file_path),
                "bytes": file_path.stat().st_size,
                "sha256": sha256_path(file_path),
            }
            for label, file_path in sorted(source_files.items())
        },
        "final_model": {
            "path": str(model_file),
            "bytes": model_file.stat().st_size,
            "sha256": sha256_path(model_file),
        },
        "final_model_receipt": {
            "path": str(model_receipt_file),
            "bytes": model_receipt_file.stat().st_size,
            "sha256": sha256_path(model_receipt_file),
        },
        "final_run_receipt": {
            "path": str(run_receipt_file),
            "bytes": run_receipt_file.stat().st_size,
            "sha256": sha256_path(run_receipt_file),
        },
        "current_rating_ledger": {
            "path": str(current_ledger_file),
            "bytes": current_ledger_file.stat().st_size,
            "sha256": sha256_path(current_ledger_file),
        },
        "current_rating_receipt": {
            "path": str(current_receipt_file),
            "bytes": current_receipt_file.stat().st_size,
            "sha256": sha256_path(current_receipt_file),
        },
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_path(Path(__file__).resolve()),
        },
    }
    payload: dict[str, Any] = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "status": "research_only",
        "authority": dict(AUTHORITY),
        "variant": RatingVariant.FUTURE_PLAYER_FORM.value,
        "source": {
            "source_as_of": source_receipt["source_as_of"],
            "accepted_game_count": source_receipt["source_game_count"],
            "accepted_identity_sha256": source_receipt["source_identity_sha256"],
            "model_eligible_game_count": len(eligible),
            "model_eligible_identity_sha256": identity_sha256(eligible),
            "excluded_game_count": int(source_receipt["source_game_count"]) - len(eligible),
        },
        "timing": {
            "feature_state": "strict_prior_before_each_map",
            "same_timestamp_policy": "batch_exclude_same_timestamp",
            "model_fit_scope": "retrospective_full_model_eligible_census",
            "chronological_evaluation_suitable": False,
        },
        "bindings": bindings,
        "feature_names": list(
            rating_variant_config(RatingVariant.FUTURE_PLAYER_FORM).feature_names
        ),
        "design_matrix_sha256": _require_hash(
            design_matrix_sha256, "V2 design matrix hash"
        ),
        "game_ids": list(game_ids),
        "game_count": len(game_ids),
        "game_identity_sha256": identity_sha256(game_ids),
        "target_rows_sha256": _require_hash(target_rows_sha256, "V2 target hash"),
        "offsets_sha256": offsets_hash,
        "rows_sha256": rows_hash,
        "rows": list(rows),
        "blockers": [
            "retrospective_full_census_model_fit_not_chronological_evaluation",
            "model_identity_exclusions_prevent_full_accepted_census_coverage",
            "public_tierlist_authority_missing",
        ],
    }
    raw = canonical_json_bytes(payload) + b"\n"
    path.write_bytes(raw)
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "research_only",
        "authority": dict(AUTHORITY),
        "source_receipt_sha256": source_receipt["receipt_sha256"],
        "source_identity_sha256": source_receipt["source_identity_sha256"],
        "game_count": len(game_ids),
        "game_identity_sha256": identity_sha256(game_ids),
        "target_rows_sha256": payload["target_rows_sha256"],
        "offsets_sha256": offsets_hash,
        "rows_sha256": rows_hash,
        "artifact_locator": path.name,
        "artifact_bytes": len(raw),
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "bindings_sha256": _canonical_sha256(bindings),
        "timing": payload["timing"],
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    receipt_path = path.with_name(path.stem + "-receipt.json")
    receipt_path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    provenance = make_offset_provenance(
        variant=RatingVariant.FUTURE_PLAYER_FORM.value,
        offsets=offsets,
        source_receipt_sha256=str(source_receipt["receipt_sha256"]),
    )
    return V2TierShadowResult(
        ledger_path=path,
        receipt_path=receipt_path,
        receipt=receipt,
        offsets=dict(offsets),
        provenance=provenance,
        game_ids=game_ids,
    )


def load_v2_tier_offset_ledger(
    ledger_path: Path | str,
    receipt_path: Path | str,
    *,
    source_receipt: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Verify a saved ledger and return pooled-candidate inputs."""

    payload, ledger_file = _load_json(ledger_path, "V2 Tier offset ledger")
    receipt, receipt_file = _load_json(receipt_path, "V2 Tier offset receipt")
    if payload.get("schema_version") != LEDGER_SCHEMA_VERSION or receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise V2TierShadowError("V2 Tier ledger schema changed")
    _verify_authority(payload.get("authority"), "V2 Tier ledger")
    _verify_authority(receipt.get("authority"), "V2 Tier receipt")
    _verify_self_hash(receipt, "receipt_sha256", "V2 Tier receipt")
    if receipt.get("artifact_locator") not in {ledger_file.name, str(ledger_file)}:
        raise V2TierShadowError("V2 Tier ledger locator changed")
    if int(receipt.get("artifact_bytes") or -1) != ledger_file.stat().st_size:
        raise V2TierShadowError("V2 Tier ledger byte count changed")
    _verify_file(ledger_file, receipt.get("artifact_sha256"), "V2 Tier ledger")
    if receipt.get("source_receipt_sha256") != source_receipt.get("receipt_sha256"):
        raise V2TierShadowError("V2 Tier source receipt changed")
    if receipt.get("source_identity_sha256") != source_receipt.get("source_identity_sha256"):
        raise V2TierShadowError("V2 Tier source identity changed")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise V2TierShadowError("V2 Tier rows are invalid")
    if _canonical_sha256(rows) != payload.get("rows_sha256") or payload.get("rows_sha256") != receipt.get("rows_sha256"):
        raise V2TierShadowError("V2 Tier row values changed")
    game_ids = tuple(sorted(str(row.get("game_id") or "") for row in rows))
    eligible = tuple(sorted(str(value) for value in source_receipt["model_eligible_game_ids"]))
    if game_ids != eligible or payload.get("game_count") != len(eligible):
        raise V2TierShadowError("V2 Tier ledger coverage changed")
    if payload.get("game_identity_sha256") != identity_sha256(eligible) or receipt.get("game_identity_sha256") != identity_sha256(eligible):
        raise V2TierShadowError("V2 Tier ledger identity changed")
    if _target_rows_sha256(rows) != payload.get("target_rows_sha256") or payload.get("target_rows_sha256") != receipt.get("target_rows_sha256"):
        raise V2TierShadowError("V2 Tier target values changed")
    offsets: dict[str, float] = {}
    for row in rows:
        game_id = str(row.get("game_id") or "")
        try:
            value = float(row.get("v2_offset_logit"))
        except (TypeError, ValueError) as error:
            raise V2TierShadowError("V2 Tier offset is invalid") from error
        if not game_id or game_id in offsets or not math.isfinite(value):
            raise V2TierShadowError("V2 Tier offset identity or value is invalid")
        offsets[game_id] = value
    if offset_values_sha256(offsets) != payload.get("offsets_sha256") or payload.get("offsets_sha256") != receipt.get("offsets_sha256"):
        raise V2TierShadowError("V2 Tier offset values changed")
    if payload.get("timing") != receipt.get("timing") or payload["timing"].get("chronological_evaluation_suitable") is not False:
        raise V2TierShadowError("V2 Tier timing contract changed")
    provenance = make_offset_provenance(
        variant=RatingVariant.FUTURE_PLAYER_FORM.value,
        offsets=offsets,
        source_receipt_sha256=str(source_receipt["receipt_sha256"]),
    )
    return offsets, provenance


def build_frozen_v2_tier_shadow(
    *,
    source_root: Path | str,
    source_receipt_path: Path | str,
    expected_source_receipt_file_sha256: str,
    model_path: Path | str,
    model_receipt_path: Path | str,
    run_receipt_path: Path | str,
    expected_model_sha256: str,
    expected_model_receipt_file_sha256: str,
    expected_run_receipt_sha256: str,
    current_ledger_path: Path | str,
    current_receipt_path: Path | str,
    expected_current_ledger_sha256: str,
    expected_current_receipt_file_sha256: str,
    destination: Path | str,
) -> V2TierShadowResult:
    """Build one exact full model-eligible V2 Tier offset ledger."""

    source_receipt, source_files = verify_source_freeze(
        source_root,
        source_receipt_path,
        expected_source_receipt_file_sha256=expected_source_receipt_file_sha256,
    )
    model, model_receipt, _run = verify_final_v2_model(
        model_path,
        model_receipt_path,
        run_receipt_path,
        expected_model_sha256=expected_model_sha256,
        expected_model_receipt_file_sha256=expected_model_receipt_file_sha256,
        expected_run_receipt_sha256=expected_run_receipt_sha256,
        source_receipt=source_receipt,
    )
    current, _current_receipt = verify_current_rating_ledger(
        current_ledger_path,
        current_receipt_path,
        expected_ledger_sha256=expected_current_ledger_sha256,
        expected_receipt_file_sha256=expected_current_receipt_file_sha256,
        source_receipt=source_receipt,
        final_model_receipt=model_receipt,
    )
    try:
        maps = pd.read_parquet(source_files["maps"])
        players = pd.read_parquet(source_files["players"])
    except (OSError, ValueError) as error:
        raise V2TierShadowError("frozen source frames cannot be read") from error
    eligible = tuple(sorted(str(value) for value in source_receipt["model_eligible_game_ids"]))
    design = build_v2_design(
        maps,
        players,
        current,
        eligible_game_ids=eligible,
        parameters=model["parameters"],
    )
    rows, offsets, design_hash = score_v2_design(
        design,
        model["parameters"],
        expected_game_ids=eligible,
    )
    target_hash = verify_target_parity(rows, maps, expected_game_ids=eligible)
    return write_v2_tier_offset_ledger(
        destination,
        rows=rows,
        offsets=offsets,
        design_matrix_sha256=design_hash,
        source_receipt=source_receipt,
        source_receipt_file=_file(source_receipt_path, "source receipt"),
        source_files=source_files,
        model_file=_file(model_path, "final V2 model"),
        model_receipt_file=_file(model_receipt_path, "final V2 model receipt"),
        run_receipt_file=_file(run_receipt_path, "final V2 run receipt"),
        current_ledger_file=_file(current_ledger_path, "current-rating ledger"),
        current_receipt_file=_file(current_receipt_path, "current-rating receipt"),
        target_rows_sha256=target_hash,
    )


__all__ = [
    "AUTHORITY",
    "LEDGER_SCHEMA_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "V2TierShadowError",
    "V2TierShadowResult",
    "build_frozen_v2_tier_shadow",
    "build_v2_design",
    "canonical_json_bytes",
    "load_v2_tier_offset_ledger",
    "score_v2_design",
    "sha256_path",
    "verify_current_rating_ledger",
    "verify_final_v2_model",
    "verify_source_freeze",
    "verify_target_parity",
    "write_v2_tier_offset_ledger",
]

# The original import path remains stable for downstream V2 jobs.  New work
# should use the variant-neutral names below.  Keep the legacy V2 functions
# above unchanged because their sealed artifact schema is still readable.
from lol_kills.research.future_value_tier_shadow import (  # noqa: E402
    FORM_VARIANTS as VARIANT_FORM_VARIANTS,
    SCALING_VARIANTS as VARIANT_SCALING_VARIANTS,
    VARIANTS as TIER_SHADOW_VARIANTS,
    TierShadowError,
    TierShadowResult,
    build_frozen_four_variant_tier_shadows,
    build_frozen_fourway_tier_shadow,
    build_frozen_variant_tier_shadow,
    build_variant_design,
    load_tier_offset_ledger,
    score_variant_design,
    verify_final_model,
    verify_full_scaling_ledger,
    verify_scaling_rating_ledger,
    verify_variant_model_receipt,
    write_tier_offset_ledger,
)
VARIANTS = TIER_SHADOW_VARIANTS

__all__.extend(
    [
        "TierShadowError",
        "TierShadowResult",
        "TIER_SHADOW_VARIANTS",
        "VARIANTS",
        "VARIANT_FORM_VARIANTS",
        "VARIANT_SCALING_VARIANTS",
        "build_frozen_four_variant_tier_shadows",
        "build_frozen_fourway_tier_shadow",
        "build_frozen_variant_tier_shadow",
        "build_variant_design",
        "load_tier_offset_ledger",
        "score_variant_design",
        "verify_final_model",
        "verify_full_scaling_ledger",
        "verify_scaling_rating_ledger",
        "verify_variant_model_receipt",
        "write_tier_offset_ledger",
    ]
)
