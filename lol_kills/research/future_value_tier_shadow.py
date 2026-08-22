"""Build retrospective, source-bound Tier shadows for all four rating variants.

The four variants share one exact model-eligible map universe.  Current rating
features are used by every variant.  Player form is used by V2 and V4.  The
full model-eligible scaling ledger is used by V3 and V4.  The model fit is
retrospective, so these artifacts are research evidence only.

This module has no public Tier List authority.  It does not use validation
fold offsets.  It scores only the full fit census bound by each final model
receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from lol_kills.research.future_value_rating import (
    CURRENT_RATING_SIGNED_MAP_FEATURES,
    FORM_METRICS,
    RATING_VARIANT_ORDER,
    SCALING_CURVE_SIGNED_MAP_FEATURES,
    FutureValueSourceError,
    Rank3AtomModel,
    RatingVariant,
    _antisymmetric_design_matrix,
    _frame_game_ids,
    _map_model_frame,
    _role,
    _scaling_native_rows_sha256,
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


LEDGER_SCHEMA_VERSION = "scryglass:future-value-tier-offset-ledger:v2"
RECEIPT_SCHEMA_VERSION = "scryglass:future-value-tier-offset-receipt:v2"
FINAL_MODEL_SCHEMA_VERSION = "scryglass:future-value-final-fit:v1"
MODEL_RECEIPT_SCHEMA_VERSIONS = frozenset(
    {FINAL_MODEL_SCHEMA_VERSION, "scryglass:future-value-model-fit:v1"}
)
SCALING_LEDGER_SCHEMA_VERSION = "scryglass:atomized-scaling-feature-ledger:v1"
VARIANTS = tuple(variant.value for variant in RATING_VARIANT_ORDER)
FORM_VARIANTS = frozenset(
    {RatingVariant.FUTURE_PLAYER_FORM.value, RatingVariant.BOTH.value}
)
SCALING_VARIANTS = frozenset(
    {RatingVariant.SCALING_CURVE.value, RatingVariant.BOTH.value}
)
AUTHORITY = {
    "research_only": True,
    "public_tierlist": False,
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
}
FINAL_MODEL_AUTHORITY = {
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
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


class TierShadowError(FutureValueSourceError):
    """A variant-neutral Tier shadow cannot prove its inputs or timing."""


@dataclass(frozen=True)
class TierShadowResult:
    """Paths and sealed values for one variant offset ledger."""

    variant: str
    ledger_path: Path
    receipt_path: Path
    receipt: Mapping[str, Any]
    offsets: Mapping[str, float]
    provenance: Mapping[str, Any]
    game_ids: tuple[str, ...]


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TierShadowError("value cannot be represented as canonical JSON") from error


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
        raise TierShadowError(f"{label} must be a SHA-256 hash")
    return text


def _variant(value: RatingVariant | str) -> RatingVariant:
    try:
        return value if isinstance(value, RatingVariant) else RatingVariant(str(value))
    except ValueError as error:
        raise TierShadowError(f"unknown rating variant: {value}") from error


def _variant_name(value: RatingVariant | str) -> str:
    return _variant(value).value


def _file(path: Path | str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise TierShadowError(f"{label} is missing or unsafe: {resolved}")
    return resolved


def _load_json(path: Path | str, label: str) -> tuple[dict[str, Any], Path]:
    file_path = _file(path, label)
    try:
        value = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TierShadowError(f"{label} cannot be read") from error
    if not isinstance(value, dict):
        raise TierShadowError(f"{label} must be a JSON object")
    return value, file_path


def _verify_file(path: Path, expected_sha256: object, label: str) -> str:
    expected = _require_hash(expected_sha256, f"expected {label} hash")
    actual = sha256_path(path)
    if actual != expected:
        raise TierShadowError(f"{label} bytes changed")
    return actual


def _verify_authority(
    value: object,
    label: str,
    *,
    expected: Mapping[str, bool] = AUTHORITY,
) -> None:
    if not isinstance(value, Mapping) or dict(value) != dict(expected):
        raise TierShadowError(f"{label} authority is incomplete or changed")


def _verify_self_hash(payload: Mapping[str, Any], field: str, label: str) -> str:
    claimed = _require_hash(payload.get(field), f"{label} {field}")
    unsigned = dict(payload)
    unsigned.pop(field, None)
    if _canonical_sha256(unsigned) != claimed:
        raise TierShadowError(f"{label} self-hash changed")
    return claimed


def _source_record_path(source_root: Path, receipt: Mapping[str, Any], label: str) -> Path:
    records = receipt.get("source_files")
    record = records.get(label) if isinstance(records, Mapping) else None
    if not isinstance(record, Mapping):
        raise TierShadowError(f"source file record is missing: {label}")
    locator = str(record.get("locator") or record.get("path") or "").strip()
    relative = Path(locator)
    if not locator or relative.is_absolute() or ".." in relative.parts:
        raise TierShadowError(f"source file locator is unsafe: {label}")
    candidate = (source_root / relative).resolve()
    try:
        candidate.relative_to(source_root)
    except ValueError as error:
        raise TierShadowError(f"source file escapes freeze root: {label}") from error
    candidate = _file(candidate, f"source file {label}")
    if int(record.get("bytes") or -1) != candidate.stat().st_size:
        raise TierShadowError(f"source file byte count changed: {label}")
    _verify_file(candidate, record.get("sha256"), f"source file {label}")
    return candidate


def verify_source_freeze(
    source_root: Path | str,
    source_receipt_path: Path | str,
    *,
    expected_source_receipt_file_sha256: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Verify the canonical receipt and its frozen source frames."""

    root = Path(source_root).expanduser().resolve()
    receipt, receipt_path = _load_json(source_receipt_path, "source receipt")
    _verify_file(receipt_path, expected_source_receipt_file_sha256, "source receipt")
    try:
        validate_future_value_source_receipt_payload(receipt)
    except ValueError as error:
        raise TierShadowError("source receipt is not canonical") from error
    paths = {
        label: _source_record_path(root, receipt, label)
        for label in ("maps", "players", "teams")
    }
    return receipt, paths


def _source_ids(source_receipt: Mapping[str, Any]) -> tuple[str, ...]:
    values = source_receipt.get("model_eligible_game_ids")
    if not isinstance(values, (list, tuple)):
        raise TierShadowError("model-eligible game IDs are missing")
    ids = tuple(sorted(str(value) for value in values))
    if not ids or len(set(ids)) != len(ids):
        raise TierShadowError("model-eligible game IDs are not canonical")
    declared = source_receipt.get("model_eligible_identity_sha256")
    if declared != identity_sha256(ids):
        raise TierShadowError("model-eligible game identity changed")
    return ids


def _source_binding_check(
    binding: Mapping[str, Any],
    source_receipt: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if not isinstance(binding, Mapping):
        raise TierShadowError(f"{label} source binding is missing")
    eligible = _source_ids(source_receipt)
    expected = {
        "source_as_of": source_receipt.get("source_as_of"),
        "source_receipt_sha256": source_receipt.get("receipt_sha256"),
        "source_identity_sha256": source_receipt.get("source_identity_sha256"),
        "source_game_count": source_receipt.get("source_game_count"),
        "model_eligible_game_count": len(eligible),
        "model_eligible_game_ids": list(eligible),
        "model_eligible_identity_sha256": identity_sha256(eligible),
    }
    if "accepted_game_ids" in source_receipt:
        expected["accepted_game_ids"] = [
            str(item) for item in source_receipt["accepted_game_ids"]
        ]
    if "source_files" in source_receipt:
        expected["source_files"] = source_receipt["source_files"]
    if any(value is None for value in expected.values()):
        raise TierShadowError(f"{label} source receipt is incomplete")
    if set(binding) != set(expected):
        raise TierShadowError(f"{label} source binding is incomplete")
    for field, value in expected.items():
        actual = binding.get(field)
        if field == "model_eligible_game_ids":
            try:
                actual = [str(item) for item in actual]
            except (TypeError, ValueError):
                actual = None
        if actual != value:
            raise TierShadowError(f"{label} source binding changed: {field}")


def _verify_fit_binding(
    payload: Mapping[str, Any],
    label: str,
    *,
    ids_field: str,
    identity_field: str,
    eligible: tuple[str, ...],
    fit_window_end: object,
) -> None:
    ids = payload.get(ids_field)
    try:
        actual_ids = tuple(str(item) for item in ids)
    except (TypeError, ValueError):
        actual_ids = ()
    if actual_ids != eligible:
        raise TierShadowError(f"{label} fit census is missing or changed")
    if payload.get("fit_game_count") != len(eligible):
        raise TierShadowError(f"{label} fit count is missing or changed")
    if payload.get("fit_game_identity_sha256") != identity_sha256(eligible):
        raise TierShadowError(f"{label} fit identity is missing or changed")
    if payload.get(identity_field) != identity_sha256(eligible):
        raise TierShadowError(f"{label} fit identity is missing or changed")
    if payload.get("fit_window_end") != fit_window_end:
        raise TierShadowError(f"{label} fit cutoff is missing or changed")


def _verify_recorded_file_binding(binding: object, label: str) -> None:
    if not isinstance(binding, Mapping):
        raise TierShadowError(f"{label} file binding is missing")
    path_value = binding.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        raise TierShadowError(f"{label} file path is missing")
    path = _file(path_value, label)
    if binding.get("bytes") != path.stat().st_size:
        raise TierShadowError(f"{label} bytes changed")
    _verify_file(path, binding.get("sha256"), label)


def _verify_recorded_bindings(
    bindings: object,
    receipt: Mapping[str, Any],
    *,
    variant: str,
) -> None:
    if not isinstance(bindings, Mapping):
        raise TierShadowError("Tier ledger input bindings are missing")
    if receipt.get("bindings_sha256") != _canonical_sha256(bindings):
        raise TierShadowError("Tier ledger input bindings changed")
    required = {
        "source_receipt",
        "source_files",
        "final_model",
        "final_model_receipt",
        "final_run_receipt",
        "current_rating_ledger",
        "current_rating_receipt",
    }
    if variant in SCALING_VARIANTS:
        required.update({"full_scaling_ledger", "full_scaling_receipt"})
    if not required.issubset(bindings):
        raise TierShadowError("Tier ledger input bindings are incomplete")
    for field in required - {"source_files"}:
        _verify_recorded_file_binding(bindings.get(field), f"Tier binding {field}")
    source_files = bindings.get("source_files")
    if not isinstance(source_files, Mapping) or not {"maps", "players", "teams"}.issubset(source_files):
        raise TierShadowError("Tier source file bindings are incomplete")
    for field, binding in source_files.items():
        _verify_recorded_file_binding(binding, f"Tier source file {field}")


def _extract_parameters(model: Mapping[str, Any], receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    parameters = model.get("parameters")
    if not isinstance(parameters, Mapping):
        parameters = receipt.get("parameters")
    if not isinstance(parameters, Mapping):
        raise TierShadowError("final model parameters are missing")
    return parameters


def _verify_parameter_shape(
    parameters: Mapping[str, Any],
    variant: RatingVariant,
) -> Mapping[str, Any]:
    config = rating_variant_config(variant)
    names = tuple(str(value) for value in parameters.get("feature_names", ()))
    if names != tuple(config.feature_names):
        raise TierShadowError(f"{variant.value} model feature schema changed")
    declared_variant = parameters.get("variant")
    if declared_variant is not None and str(declared_variant) != variant.value:
        raise TierShadowError(f"{variant.value} model variant changed")
    variant_receipt = parameters.get("variant_receipt")
    if variant_receipt is not None:
        if not isinstance(variant_receipt, Mapping):
            raise TierShadowError(f"{variant.value} variant receipt is invalid")
        expected_receipt = rating_variant_config(variant).receipt()
        if dict(variant_receipt) != expected_receipt:
            raise TierShadowError(f"{variant.value} variant receipt changed")
    try:
        imputation = [float(parameters["fold_local_side_imputation"][name]) for name in names]
        scales = [float(parameters["feature_scales"][name]) for name in names]
        coefficients = [float(parameters["coefficients"][name]) for name in names]
    except (KeyError, TypeError, ValueError) as error:
        raise TierShadowError(f"{variant.value} model parameters are incomplete") from error
    if (
        not np.isfinite(imputation).all()
        or not np.isfinite(scales).all()
        or not np.isfinite(coefficients).all()
        or any(value <= 0.0 for value in scales)
        or float(parameters.get("intercept", math.nan)) != 0.0
    ):
        raise TierShadowError(f"{variant.value} model parameters are invalid")
    declared_hash = parameters.get("parameter_sha256")
    if declared_hash is not None:
        unsigned = dict(parameters)
        unsigned.pop("parameter_sha256", None)
        if _canonical_sha256(unsigned) != str(declared_hash).lower():
            raise TierShadowError(f"{variant.value} model parameter hash changed")
    return parameters


def verify_final_model(
    model_path: Path | str,
    model_receipt_path: Path | str,
    run_receipt_path: Path | str,
    *,
    variant: RatingVariant | str,
    expected_model_sha256: str,
    expected_model_receipt_file_sha256: str,
    expected_run_receipt_sha256: str,
    source_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify one final model receipt without using chronological predictions."""

    resolved = _variant(variant)
    model, model_file = _load_json(model_path, f"final {resolved.value} model")
    receipt, receipt_file = _load_json(
        model_receipt_path, f"final {resolved.value} model receipt"
    )
    run, run_file = _load_json(run_receipt_path, f"final {resolved.value} run receipt")
    model_sha = _verify_file(model_file, expected_model_sha256, "final model")
    _verify_file(receipt_file, expected_model_receipt_file_sha256, "final model receipt")
    _verify_file(run_file, expected_run_receipt_sha256, "final run receipt")
    for payload, label in (
        (model, "final model"),
        (receipt, "final model receipt"),
        (run, "final run receipt"),
    ):
        allowed_schemas = (
            MODEL_RECEIPT_SCHEMA_VERSIONS
            if label == "final model receipt"
            else {FINAL_MODEL_SCHEMA_VERSION}
        )
        if payload.get("schema_version") not in allowed_schemas:
            raise TierShadowError(f"{label} schema changed")
        if payload.get("status") not in {"research_only", "research_only_blocked"}:
            raise TierShadowError(f"{label} status grants authority")
        _verify_authority(
            payload.get("authority"),
            label,
            expected=FINAL_MODEL_AUTHORITY,
        )
    receipt_hash = _verify_self_hash(receipt, "receipt_sha256", "final model receipt")
    if model.get("receipt_sha256") is not None and model.get("receipt_sha256") != receipt_hash:
        raise TierShadowError("final model receipt binding changed")
    if model.get("receipt") is not None and model.get("receipt") != receipt:
        raise TierShadowError("final model embedded receipt changed")
    if run.get("model_receipt_sha256") is not None and run.get("model_receipt_sha256") != receipt_hash:
        raise TierShadowError("final run receipt model binding changed")
    if run.get("model_artifact_sha256") is not None and run.get("model_artifact_sha256") != model_sha:
        raise TierShadowError("final run receipt artifact binding changed")
    source_binding = receipt.get("source_binding")
    _source_binding_check(source_binding, source_receipt, label="final model receipt")
    source_summary = model.get("source")
    _source_binding_check(source_summary, source_receipt, label="final model")
    if run.get("source_receipt_sha256") != source_receipt.get("receipt_sha256"):
        raise TierShadowError("final run source receipt binding is missing or changed")
    if run.get("source_identity_sha256") != source_receipt.get("source_identity_sha256"):
        raise TierShadowError("final run source identity binding is missing or changed")
    declared_variant = receipt.get("variant") or model.get("variant") or run.get("variant")
    if declared_variant is not None and str(declared_variant) != resolved.value:
        raise TierShadowError(f"{resolved.value} model variant changed")
    config = rating_variant_config(resolved)
    for container, label in ((receipt, "final model receipt"), (model, "final model")):
        claimed_config = container.get("variant_config")
        if claimed_config is not None and claimed_config != config.receipt():
            raise TierShadowError(f"{label} variant configuration changed")
        transformation = container.get("transformation_binding")
        if isinstance(transformation, Mapping):
            if transformation.get("variant") not in (None, resolved.value):
                raise TierShadowError(f"{label} transformation variant changed")
            if transformation.get("variant_receipt") not in (None, config.receipt()):
                raise TierShadowError(f"{label} transformation receipt changed")
            if transformation.get("feature_names") is not None and tuple(
                transformation["feature_names"]
            ) != tuple(config.feature_names):
                raise TierShadowError(f"{label} transformation feature order changed")
    dependencies = receipt.get("variant_dependencies")
    if isinstance(dependencies, Mapping):
        expected_dependencies = {
            "current_rating": True,
            "future_player_form": resolved in {
                RatingVariant.FUTURE_PLAYER_FORM,
                RatingVariant.BOTH,
            },
            "scaling_curve": resolved in {
                RatingVariant.SCALING_CURVE,
                RatingVariant.BOTH,
            },
        }
        if {str(key): bool(value) for key, value in dependencies.items()} != expected_dependencies:
            raise TierShadowError(f"{resolved.value} model dependency contract changed")
    eligible = _source_ids(source_receipt)
    fit_window_end = source_receipt.get("source_as_of")
    if fit_window_end is None:
        raise TierShadowError("final model source cutoff is missing")
    _verify_fit_binding(
        receipt,
        "final model receipt",
        ids_field="fit_game_ids",
        identity_field="fit_game_identity_sha256",
        eligible=eligible,
        fit_window_end=fit_window_end,
    )
    _verify_fit_binding(
        run,
        "final run receipt",
        ids_field="eligible_game_ids",
        identity_field="eligible_game_identity_sha256",
        eligible=eligible,
        fit_window_end=fit_window_end,
    )
    timing = receipt.get("timing") or run.get("timing")
    if timing is not None:
        if not isinstance(timing, Mapping):
            raise TierShadowError("final model timing receipt is invalid")
        if timing.get("chronological_evaluation_suitable") is True:
            raise TierShadowError("final model is marked chronological")
        if "model_fit_scope" in timing and timing.get("model_fit_scope") != "retrospective_full_model_eligible_census":
            raise TierShadowError("final model fit scope changed")
    parameters = _extract_parameters(model, receipt)
    _verify_parameter_shape(parameters, resolved)
    if resolved in FORM_VARIANTS:
        rank = parameters.get("rank_3")
        if not isinstance(rank, Mapping):
            raise TierShadowError(f"{resolved.value} rank-3 parameters are missing")
        _verify_self_hash(rank, "parameter_sha256", f"{resolved.value} rank-3 parameters")
    return model, receipt, run


def verify_variant_model_receipt(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Alias with an explicit variant-neutral name."""

    return verify_final_model(*args, **kwargs)


verify_variant_model = verify_final_model


def verify_current_rating_ledger(
    ledger_path: Path | str,
    receipt_path: Path | str,
    *,
    expected_ledger_sha256: str,
    expected_receipt_file_sha256: str,
    source_receipt: Mapping[str, Any],
    final_model_receipt: Mapping[str, Any] | None = None,
    expected_game_ids: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Verify the full eligible current-rating feature ledger."""

    ledger_file = _file(ledger_path, "current-rating ledger")
    receipt, receipt_file = _load_json(receipt_path, "current-rating receipt")
    _verify_file(ledger_file, expected_ledger_sha256, "current-rating ledger")
    _verify_file(receipt_file, expected_receipt_file_sha256, "current-rating receipt")
    _verify_self_hash(receipt, "receipt_sha256", "current-rating receipt")
    if receipt.get("source_receipt_sha256") != source_receipt.get("receipt_sha256"):
        raise TierShadowError("current-rating source receipt changed")
    if receipt.get("source_identity_sha256") != source_receipt.get("source_identity_sha256"):
        raise TierShadowError("current-rating source identity changed")
    if final_model_receipt is not None:
        binding = final_model_receipt.get("current_rating_feature_binding")
        if isinstance(binding, Mapping):
            if binding.get("artifact", {}).get("sha256") not in {None, expected_ledger_sha256}:
                raise TierShadowError("current-rating model artifact binding changed")
            _source_binding_check(binding, source_receipt, label="current-rating model")
    try:
        frame = pd.read_parquet(ledger_file)
    except (OSError, ValueError) as error:
        raise TierShadowError("current-rating ledger cannot be read") from error
    required = {"game_id", "date", *CURRENT_RATING_SIGNED_MAP_FEATURES}
    if not required.issubset(frame.columns):
        raise TierShadowError("current-rating ledger schema changed")
    ids = tuple(sorted(str(value) for value in frame["game_id"]))
    expected = tuple(sorted(str(value) for value in (expected_game_ids or _source_ids(source_receipt))))
    if len(ids) != len(set(ids)) or ids != expected:
        raise TierShadowError("current-rating ledger universe changed")
    values = frame[list(CURRENT_RATING_SIGNED_MAP_FEATURES)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise TierShadowError("current-rating ledger contains non-finite features")
    digest = rating_feature_values_sha256(frame, CURRENT_RATING_SIGNED_MAP_FEATURES)
    declared = receipt.get("feature_value_digest")
    if declared is not None and declared != digest:
        raise TierShadowError("current-rating feature values changed")
    return frame, receipt


def verify_full_scaling_ledger(
    ledger_path: Path | str,
    receipt_path: Path | str,
    *,
    expected_ledger_sha256: str,
    expected_receipt_file_sha256: str,
    source_receipt: Mapping[str, Any],
    expected_game_ids: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Verify the full model-eligible scaling ledger required by V3 and V4."""

    ledger_file = _file(ledger_path, "full scaling ledger")
    receipt, receipt_file = _load_json(receipt_path, "scaling ledger receipt")
    _verify_file(ledger_file, expected_ledger_sha256, "full scaling ledger")
    _verify_file(receipt_file, expected_receipt_file_sha256, "scaling ledger receipt")
    _verify_self_hash(receipt, "receipt_sha256", "scaling ledger receipt")
    if receipt.get("schema_version") != SCALING_LEDGER_SCHEMA_VERSION:
        raise TierShadowError("scaling ledger receipt schema changed")
    if receipt.get("status") != "research_only" or receipt.get("authority") is not False:
        raise TierShadowError("scaling ledger authority changed")
    if receipt.get("public_authority") is not False:
        raise TierShadowError("scaling ledger public authority changed")
    if receipt.get("source_receipt_sha256") != source_receipt.get("receipt_sha256"):
        raise TierShadowError("scaling ledger source receipt changed")
    if receipt.get("source_identity_sha256") != source_receipt.get("source_identity_sha256"):
        raise TierShadowError("scaling ledger source identity changed")
    if receipt.get("model_eligible_only") is not True:
        raise TierShadowError("scaling ledger is not model-eligible-only")
    if receipt.get("evaluation_mode") != "online_full_census":
        raise TierShadowError("scaling ledger is not the full online census")
    try:
        frame = pd.read_parquet(ledger_file)
    except (OSError, ValueError) as error:
        raise TierShadowError("full scaling ledger cannot be read") from error
    required = {"game_id", "date", *SCALING_CURVE_SIGNED_MAP_FEATURES}
    if not required.issubset(frame.columns):
        raise TierShadowError("scaling ledger feature schema changed")
    ids = tuple(sorted(str(value) for value in frame["game_id"]))
    expected = tuple(sorted(str(value) for value in (expected_game_ids or _source_ids(source_receipt))))
    if len(ids) != len(set(ids)) or ids != expected:
        raise TierShadowError("scaling ledger universe changed")
    values = frame[list(SCALING_CURVE_SIGNED_MAP_FEATURES)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise TierShadowError("scaling ledger contains non-finite signed features")
    output_ids = tuple(sorted(str(value) for value in receipt.get("output_game_ids", ())))
    if output_ids and output_ids != expected:
        raise TierShadowError("scaling ledger receipt output universe changed")
    if receipt.get("output_game_count") is not None and receipt.get("output_game_count") != len(expected):
        raise TierShadowError("scaling ledger output count changed")
    if receipt.get("output_identity_sha256") is not None and receipt.get("output_identity_sha256") != identity_sha256(expected):
        raise TierShadowError("scaling ledger output identity changed")
    row_digest = receipt.get("row_value_digest_sha256")
    if row_digest is not None:
        try:
            actual_row_digest = _scaling_native_rows_sha256(frame)
        except (KeyError, TypeError, ValueError) as error:
            raise TierShadowError("scaling ledger row digest cannot be rebuilt") from error
        if actual_row_digest != row_digest:
            raise TierShadowError("scaling ledger row values changed")
    return frame, receipt


def verify_scaling_rating_ledger(*args: Any, **kwargs: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compatibility alias for the full scaling-ledger verifier."""

    return verify_full_scaling_ledger(*args, **kwargs)


def _rank_3_model(parameters: Mapping[str, Any]) -> Rank3AtomModel:
    rank = parameters.get("rank_3")
    if not isinstance(rank, Mapping):
        raise TierShadowError("rank-3 parameters are missing")
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
                str(key): int(value) for key, value in rank["champion_role_support"].items()
            },
            fit_game_ids=tuple(str(value) for value in rank["fit_game_ids"]),
            fit_window_end=str(rank["fit_window_end"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise TierShadowError("rank-3 parameters are invalid") from error
    if model.parameter_receipt().get("parameter_sha256") != rank.get("parameter_sha256"):
        raise TierShadowError("rank-3 parameter hash changed")
    return model


def _ensure_ids(frame: pd.DataFrame, expected: tuple[str, ...], label: str) -> pd.DataFrame:
    if "game_id" not in frame.columns:
        raise TierShadowError(f"{label} has no game_id column")
    work = frame.copy()
    work["game_id"] = work["game_id"].astype(str)
    ids = tuple(sorted(work["game_id"]))
    if len(ids) != len(set(ids)) or ids != expected:
        missing = len(set(expected) - set(ids))
        extra = len(set(ids) - set(expected))
        raise TierShadowError(f"{label} universe changed (missing={missing}, extra={extra})")
    return work


def _join_features(
    design: pd.DataFrame,
    ledger: pd.DataFrame,
    features: Sequence[str],
    label: str,
    expected: tuple[str, ...],
) -> pd.DataFrame:
    work = _ensure_ids(ledger, expected, label)
    if work["game_id"].duplicated().any():
        raise TierShadowError(f"{label} contains duplicate game IDs")
    missing = sorted(set(features) - set(work.columns))
    if missing:
        raise TierShadowError(f"{label} is missing features: {', '.join(missing)}")
    values = work[list(features)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise TierShadowError(f"{label} contains non-finite features")
    return design.merge(work[["game_id", *features]], on="game_id", how="left", validate="one_to_one")


def _normalise_form(maps: pd.DataFrame, players: pd.DataFrame, expected: tuple[str, ...]) -> pd.DataFrame:
    required_direct = {
        "game_id",
        "player_id",
        "side",
        "role",
        "date",
        *[f"prior_form_{metric}" for metric in FORM_METRICS],
    }
    direct_form = required_direct.issubset(players.columns)
    if direct_form:
        form = players.copy()
        form["game_id"] = form["game_id"].astype(str)
        form["player_id"] = form["player_id"].astype(str)
        form["side"] = form["side"].astype(str).str.casefold()
        form["role"] = form["role"].map(_role)
    else:
        form = players.copy()
        form["game_id"] = _frame_game_ids(form, "players").astype(str)
        form["player_id"] = form["playerid"].astype(str)
        form["team_id"] = form["teamid"].astype(str)
        form["side"] = form["side"].astype(str).str.casefold()
        form["role"] = form["position"].map(_role)
        form["date"] = pd.to_datetime(form.get("date"), utc=True, errors="coerce")
        form = _latest_player_form(maps, form)
    if form["role"].isna().any():
        raise TierShadowError("future form has an unknown role")
    all_form_ids = tuple(sorted(str(value) for value in form["game_id"].unique()))
    if direct_form and all_form_ids != expected:
        raise TierShadowError("future form universe changed")
    form = form[form["game_id"].astype(str).isin(expected)].copy()
    ids = tuple(sorted(str(value) for value in form["game_id"].unique()))
    if ids != expected:
        raise TierShadowError("future form universe changed")
    for metric in FORM_METRICS:
        support = f"prior_form_{metric}_support"
        effective = f"prior_form_{metric}_effective_support"
        if support not in form.columns:
            raise TierShadowError(f"future form support is missing: {support}")
        if effective not in form.columns:
            form[effective] = pd.to_numeric(form[support], errors="coerce")
    return form


def build_variant_design(
    maps: pd.DataFrame,
    players: pd.DataFrame | None = None,
    current_rating: pd.DataFrame | None = None,
    scaling_ledger: pd.DataFrame | None = None,
    *,
    variant: RatingVariant | str,
    eligible_game_ids: Sequence[str],
    parameters: Mapping[str, Any],
    current_rating_ledger: pd.DataFrame | None = None,
    full_scaling_ledger: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one variant design from the same exact eligible map universe."""

    if current_rating is None:
        current_rating = current_rating_ledger
    elif current_rating_ledger is not None:
        raise TierShadowError("current-rating ledger was supplied twice")
    if scaling_ledger is None:
        scaling_ledger = full_scaling_ledger
    elif full_scaling_ledger is not None:
        raise TierShadowError("scaling ledger was supplied twice")
    resolved = _variant(variant)
    config = rating_variant_config(resolved)
    expected = tuple(sorted(str(value) for value in eligible_game_ids))
    if not expected or len(set(expected)) != len(expected):
        raise TierShadowError("eligible game IDs are not canonical")
    model_frame = _map_model_frame(maps)
    model_frame["game_id"] = model_frame["game_id"].astype(str)
    model_frame = model_frame[model_frame["game_id"].isin(expected)].copy()
    if tuple(sorted(model_frame["game_id"])) != expected:
        raise TierShadowError("maps do not cover the exact eligible universe")
    params = _verify_parameter_shape(parameters, resolved)
    if resolved.value in FORM_VARIANTS:
        if players is None:
            raise TierShadowError(f"{resolved.value} requires the player-form source")
        form = _normalise_form(maps, players, expected)
        design = build_future_value_design(
            model_frame,
            form,
            _rank_3_model(params),
            verified_model_frame=model_frame,
        )
    else:
        design = model_frame[["game_id", "date", "target"]].copy()
    if current_rating is None:
        raise TierShadowError(f"{resolved.value} requires the current-rating ledger")
    design = _join_features(
        design,
        current_rating,
        CURRENT_RATING_SIGNED_MAP_FEATURES,
        "current-rating ledger",
        expected,
    )
    if resolved.value in SCALING_VARIANTS:
        if scaling_ledger is None:
            raise TierShadowError(f"{resolved.value} requires the full scaling ledger")
        design = _join_features(
            design,
            scaling_ledger,
            SCALING_CURVE_SIGNED_MAP_FEATURES,
            "full scaling ledger",
            expected,
        )
    for name in config.feature_names:
        if name not in design.columns and f"__blue_{name}" not in design.columns:
            raise TierShadowError(f"{resolved.value} design is missing feature: {name}")
    design.attrs["variant"] = resolved.value
    design.attrs["variant_receipt"] = config.receipt()
    design.attrs["eligible_game_ids"] = list(expected)
    return design


def build_v2_design(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    current_rating: pd.DataFrame,
    *,
    eligible_game_ids: Sequence[str],
    parameters: Mapping[str, Any],
) -> pd.DataFrame:
    """Compatibility wrapper for the former V2 design builder."""

    return build_variant_design(
        maps,
        players,
        current_rating,
        None,
        variant=RatingVariant.FUTURE_PLAYER_FORM,
        eligible_game_ids=eligible_game_ids,
        parameters=parameters,
    )


def score_variant_design(
    design: pd.DataFrame,
    parameters: Mapping[str, Any],
    *,
    variant: RatingVariant | str | None = None,
    expected_game_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, float], str]:
    """Score one verified design and emit neutral ``offset_logit`` rows."""

    resolved = _variant(variant or str(parameters.get("variant") or design.attrs.get("variant") or ""))
    config = rating_variant_config(resolved)
    params = _verify_parameter_shape(parameters, resolved)
    names = tuple(config.feature_names)
    imputation = np.asarray(
        [params["fold_local_side_imputation"][name] for name in names], dtype=float
    )
    scales = np.asarray([params["feature_scales"][name] for name in names], dtype=float)
    coefficients = np.asarray([params["coefficients"][name] for name in names], dtype=float)
    matrix = _antisymmetric_design_matrix(design, imputation, feature_names=names)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        logits = (matrix / scales) @ coefficients
    if not np.isfinite(logits).all():
        raise TierShadowError(f"{resolved.value} score produced non-finite logits")
    work = design[["game_id", "date", "target"]].copy()
    work["game_id"] = work["game_id"].astype(str)
    expected = tuple(sorted(str(value) for value in expected_game_ids))
    if tuple(sorted(work["game_id"])) != expected or work["game_id"].duplicated().any():
        raise TierShadowError(f"{resolved.value} score census changed")
    dates = pd.to_datetime(work["date"], utc=True, errors="coerce")
    targets = pd.to_numeric(work["target"], errors="coerce")
    if dates.isna().any() or not targets.isin({0, 1}).all():
        raise TierShadowError(f"{resolved.value} score date or target is invalid")
    work["date"] = dates
    work["target"] = targets.astype(int)
    work["offset_logit"] = logits
    work["_matrix_index"] = np.arange(len(work), dtype=int)
    work = work.sort_values(["date", "game_id"], kind="mergesort")
    rows = [
        {
            "game_id": str(row.game_id),
            "date": pd.Timestamp(row.date).isoformat().replace("+00:00", "Z"),
            "target": int(row.target),
            "offset_logit": float(row.offset_logit),
        }
        for row in work.itertuples(index=False)
    ]
    offsets = {row["game_id"]: float(row["offset_logit"]) for row in rows}
    matrix_rows = [
        {
            "game_id": str(game_id),
            "values": [float(value) for value in matrix[index]],
        }
        for index, game_id in sorted(
            enumerate(design["game_id"].astype(str)), key=lambda item: item[1]
        )
    ]
    return rows, offsets, _canonical_sha256(matrix_rows)


def score_v2_design(
    design: pd.DataFrame,
    parameters: Mapping[str, Any],
    *,
    expected_game_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, float], str]:
    """Compatibility wrapper that preserves the legacy V2 row field."""

    rows, offsets, digest = score_variant_design(
        design,
        parameters,
        variant=RatingVariant.FUTURE_PLAYER_FORM,
        expected_game_ids=expected_game_ids,
    )
    return [
        {**row, "v2_offset_logit": row.pop("offset_logit")}
        for row in rows
    ], offsets, digest


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
    """Verify every neutral offset target against the frozen map outcome."""

    model_frame = _map_model_frame(maps)
    model_frame["game_id"] = model_frame["game_id"].astype(str)
    model_frame = model_frame[model_frame["game_id"].isin(expected_game_ids)].copy()
    target_by_id = model_frame.set_index("game_id")["target"]
    expected = tuple(sorted(str(value) for value in expected_game_ids))
    if tuple(sorted(target_by_id.index)) != expected:
        raise TierShadowError("frozen target identities are incomplete")
    row_ids = tuple(sorted(str(row.get("game_id") or "") for row in rows))
    if row_ids != expected:
        raise TierShadowError("target rows do not cover the eligible universe")
    for row in rows:
        game_id = str(row["game_id"])
        if float(row["target"]) != float(target_by_id.loc[game_id]):
            raise TierShadowError(f"{game_id} target changed")
    return _target_rows_sha256(rows)


def _file_binding(path: Path, label: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_path(path),
        "label": label,
    }


def write_tier_offset_ledger(
    destination: Path | str,
    *,
    variant: RatingVariant | str,
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
    scaling_ledger_file: Path | None = None,
    scaling_receipt_file: Path | None = None,
) -> TierShadowResult:
    """Write one immutable variant-neutral retrospective offset ledger."""

    resolved = _variant_name(variant)
    path = Path(destination).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise TierShadowError(f"Tier ledger destination exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    game_ids = tuple(sorted(str(row["game_id"]) for row in rows))
    eligible = _source_ids(source_receipt)
    if game_ids != eligible or set(offsets) != set(eligible):
        raise TierShadowError("Tier ledger does not cover the full eligible census")
    normalized_rows = []
    for row in rows:
        if "offset_logit" not in row:
            raise TierShadowError("Tier ledger rows must use neutral offset_logit")
        normalized_rows.append(
            {
                "game_id": str(row["game_id"]),
                "date": str(row["date"]),
                "target": int(row["target"]),
                "offset_logit": float(row["offset_logit"]),
            }
        )
    bindings: dict[str, Any] = {
        "source_receipt": _file_binding(source_receipt_file, "source receipt"),
        "source_files": {
            label: _file_binding(file_path, label)
            for label, file_path in sorted(source_files.items())
        },
        "final_model": _file_binding(model_file, "final model"),
        "final_model_receipt": _file_binding(model_receipt_file, "final model receipt"),
        "final_run_receipt": _file_binding(run_receipt_file, "final run receipt"),
        "current_rating_ledger": _file_binding(current_ledger_file, "current rating"),
        "current_rating_receipt": _file_binding(current_receipt_file, "current receipt"),
    }
    if scaling_ledger_file is not None:
        bindings["full_scaling_ledger"] = _file_binding(scaling_ledger_file, "full scaling")
    if scaling_receipt_file is not None:
        bindings["full_scaling_receipt"] = _file_binding(scaling_receipt_file, "scaling receipt")
    offsets_hash = offset_values_sha256(offsets)
    payload: dict[str, Any] = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "status": "research_only",
        "authority": dict(AUTHORITY),
        "variant": resolved,
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
            "validation_offsets_used": False,
        },
        "bindings": bindings,
        "feature_names": list(rating_variant_config(resolved).feature_names),
        "design_matrix_sha256": _require_hash(design_matrix_sha256, "design matrix hash"),
        "game_ids": list(game_ids),
        "game_count": len(game_ids),
        "game_identity_sha256": identity_sha256(game_ids),
        "target_rows_sha256": _require_hash(target_rows_sha256, "target hash"),
        "offsets_sha256": offsets_hash,
        "rows_sha256": _canonical_sha256(normalized_rows),
        "rows": normalized_rows,
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
        "variant": resolved,
        "source_receipt_sha256": source_receipt["receipt_sha256"],
        "source_identity_sha256": source_receipt["source_identity_sha256"],
        "game_count": len(game_ids),
        "game_identity_sha256": identity_sha256(game_ids),
        "target_rows_sha256": payload["target_rows_sha256"],
        "offsets_sha256": offsets_hash,
        "rows_sha256": payload["rows_sha256"],
        "artifact_locator": path.name,
        "artifact_bytes": len(raw),
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "bindings_sha256": _canonical_sha256(bindings),
        "timing": payload["timing"],
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    receipt_path = path.with_name(path.stem + "-receipt.json")
    receipt_path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    offsets_copy = {str(key): float(value) for key, value in offsets.items()}
    provenance = make_offset_provenance(
        variant=resolved,
        offsets=offsets_copy,
        source_receipt_sha256=str(source_receipt["receipt_sha256"]),
    )
    return TierShadowResult(
        variant=resolved,
        ledger_path=path,
        receipt_path=receipt_path,
        receipt=receipt,
        offsets=offsets_copy,
        provenance=provenance,
        game_ids=game_ids,
    )


def load_tier_offset_ledger(
    ledger_path: Path | str,
    receipt_path: Path | str,
    *,
    source_receipt: Mapping[str, Any],
    variant: RatingVariant | str | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Verify one neutral offset ledger and return its pooled-candidate input."""

    payload, ledger_file = _load_json(ledger_path, "Tier offset ledger")
    receipt, _receipt_file = _load_json(receipt_path, "Tier offset receipt")
    if payload.get("schema_version") != LEDGER_SCHEMA_VERSION or receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise TierShadowError("Tier offset ledger schema changed")
    _verify_authority(payload.get("authority"), "Tier offset ledger")
    _verify_authority(receipt.get("authority"), "Tier offset receipt")
    _verify_self_hash(receipt, "receipt_sha256", "Tier offset receipt")
    resolved_name = str(payload.get("variant") or "")
    if resolved_name not in VARIANTS:
        raise TierShadowError("Tier offset variant is invalid")
    if variant is not None and resolved_name != _variant_name(variant):
        raise TierShadowError("Tier offset variant changed")
    if receipt.get("variant") != resolved_name:
        raise TierShadowError("Tier receipt variant changed")
    _verify_recorded_bindings(
        payload.get("bindings"),
        receipt,
        variant=resolved_name,
    )
    if receipt.get("artifact_locator") not in {ledger_file.name, str(ledger_file)}:
        raise TierShadowError("Tier ledger locator changed")
    if int(receipt.get("artifact_bytes") or -1) != ledger_file.stat().st_size:
        raise TierShadowError("Tier ledger byte count changed")
    _verify_file(ledger_file, receipt.get("artifact_sha256"), "Tier offset ledger")
    if receipt.get("source_receipt_sha256") != source_receipt.get("receipt_sha256"):
        raise TierShadowError("Tier source receipt changed")
    if receipt.get("source_identity_sha256") != source_receipt.get("source_identity_sha256"):
        raise TierShadowError("Tier source identity changed")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise TierShadowError("Tier offset rows are invalid")
    if _canonical_sha256(rows) != payload.get("rows_sha256") or payload.get("rows_sha256") != receipt.get("rows_sha256"):
        raise TierShadowError("Tier offset row values changed")
    eligible = _source_ids(source_receipt)
    game_ids = tuple(sorted(str(row.get("game_id") or "") for row in rows))
    if game_ids != eligible or payload.get("game_count") != len(eligible):
        raise TierShadowError("Tier offset ledger coverage changed")
    if payload.get("game_identity_sha256") != identity_sha256(eligible) or receipt.get("game_identity_sha256") != identity_sha256(eligible):
        raise TierShadowError("Tier offset ledger identity changed")
    if _target_rows_sha256(rows) != payload.get("target_rows_sha256") or payload.get("target_rows_sha256") != receipt.get("target_rows_sha256"):
        raise TierShadowError("Tier offset target values changed")
    offsets: dict[str, float] = {}
    for row in rows:
        game_id = str(row.get("game_id") or "")
        if "offset_logit" not in row:
            raise TierShadowError("Tier offset field is not neutral")
        try:
            value = float(row["offset_logit"])
        except (TypeError, ValueError) as error:
            raise TierShadowError("Tier offset is invalid") from error
        if not game_id or game_id in offsets or not math.isfinite(value):
            raise TierShadowError("Tier offset identity or value is invalid")
        offsets[game_id] = value
    if offset_values_sha256(offsets) != payload.get("offsets_sha256") or payload.get("offsets_sha256") != receipt.get("offsets_sha256"):
        raise TierShadowError("Tier offset values changed")
    timing = payload.get("timing")
    expected_timing = {
        "feature_state": "strict_prior_before_each_map",
        "same_timestamp_policy": "batch_exclude_same_timestamp",
        "model_fit_scope": "retrospective_full_model_eligible_census",
        "chronological_evaluation_suitable": False,
        "validation_offsets_used": False,
    }
    if timing != expected_timing or receipt.get("timing") != expected_timing:
        raise TierShadowError("Tier offset timing changed")
    provenance = make_offset_provenance(
        variant=resolved_name,
        offsets=offsets,
        source_receipt_sha256=str(source_receipt["receipt_sha256"]),
    )
    return offsets, provenance


def build_frozen_variant_tier_shadow(
    *,
    source_root: Path | str,
    source_receipt_path: Path | str,
    expected_source_receipt_file_sha256: str,
    variant: RatingVariant | str,
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
    scaling_ledger_path: Path | str | None = None,
    scaling_receipt_path: Path | str | None = None,
    expected_scaling_ledger_sha256: str | None = None,
    expected_scaling_receipt_file_sha256: str | None = None,
    destination: Path | str,
) -> TierShadowResult:
    """Build one exact retrospective full model-eligible variant ledger."""

    resolved = _variant(variant)
    source_receipt, source_files = verify_source_freeze(
        source_root,
        source_receipt_path,
        expected_source_receipt_file_sha256=expected_source_receipt_file_sha256,
    )
    eligible = _source_ids(source_receipt)
    model, model_receipt, _run = verify_final_model(
        model_path,
        model_receipt_path,
        run_receipt_path,
        variant=resolved,
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
        expected_game_ids=eligible,
    )
    scaling = None
    scaling_receipt_file = None
    if resolved.value in SCALING_VARIANTS:
        if any(value is None for value in (scaling_ledger_path, scaling_receipt_path, expected_scaling_ledger_sha256, expected_scaling_receipt_file_sha256)):
            raise TierShadowError(f"{resolved.value} requires the full scaling ledger inputs")
        scaling, _scaling_receipt = verify_full_scaling_ledger(
            scaling_ledger_path,
            scaling_receipt_path,
            expected_ledger_sha256=str(expected_scaling_ledger_sha256),
            expected_receipt_file_sha256=str(expected_scaling_receipt_file_sha256),
            source_receipt=source_receipt,
            expected_game_ids=eligible,
        )
        scaling_receipt_file = _file(scaling_receipt_path, "scaling receipt")
    try:
        maps = pd.read_parquet(source_files["maps"])
        players = pd.read_parquet(source_files["players"])
    except (OSError, ValueError) as error:
        raise TierShadowError("frozen source frames cannot be read") from error
    parameters = _extract_parameters(model, model_receipt)
    design = build_variant_design(
        maps,
        players if resolved.value in FORM_VARIANTS else None,
        current,
        scaling,
        variant=resolved,
        eligible_game_ids=eligible,
        parameters=parameters,
    )
    rows, offsets, design_hash = score_variant_design(
        design,
        parameters,
        variant=resolved,
        expected_game_ids=eligible,
    )
    target_hash = verify_target_parity(rows, maps, expected_game_ids=eligible)
    return write_tier_offset_ledger(
        destination,
        variant=resolved,
        rows=rows,
        offsets=offsets,
        design_matrix_sha256=design_hash,
        source_receipt=source_receipt,
        source_receipt_file=_file(source_receipt_path, "source receipt"),
        source_files=source_files,
        model_file=_file(model_path, "final model"),
        model_receipt_file=_file(model_receipt_path, "final model receipt"),
        run_receipt_file=_file(run_receipt_path, "final run receipt"),
        current_ledger_file=_file(current_ledger_path, "current rating ledger"),
        current_receipt_file=_file(current_receipt_path, "current rating receipt"),
        scaling_ledger_file=(
            _file(scaling_ledger_path, "full scaling ledger")
            if scaling_ledger_path is not None
            else None
        ),
        scaling_receipt_file=scaling_receipt_file,
        target_rows_sha256=target_hash,
    )


def build_frozen_fourway_tier_shadow(
    *,
    source_root: Path | str,
    source_receipt_path: Path | str,
    expected_source_receipt_file_sha256: str,
    model_inputs: Mapping[str, Mapping[str, Any]],
    current_ledger_path: Path | str,
    current_receipt_path: Path | str,
    expected_current_ledger_sha256: str,
    expected_current_receipt_file_sha256: str,
    scaling_ledger_path: Path | str | None = None,
    scaling_receipt_path: Path | str | None = None,
    expected_scaling_ledger_sha256: str | None = None,
    expected_scaling_receipt_file_sha256: str | None = None,
    destinations: Mapping[str, Path | str],
) -> dict[str, TierShadowResult]:
    """Build all four shadows and require identical IDs and target rows.

    Each ``model_inputs`` value contains the model, model receipt, run receipt,
    and their expected hashes.  The current ledger is shared.  V3 and V4
    receive the same full online scaling ledger.  Validation-fold artifacts
    are never accepted as substitutes.
    """

    if set(model_inputs) != set(VARIANTS) or set(destinations) != set(VARIANTS):
        raise TierShadowError("four-way model inputs and destinations are incomplete")
    results: dict[str, TierShadowResult] = {}
    for variant in VARIANTS:
        inputs = model_inputs[variant]
        required = {
            "model_path",
            "model_receipt_path",
            "run_receipt_path",
            "expected_model_sha256",
            "expected_model_receipt_file_sha256",
            "expected_run_receipt_sha256",
        }
        if not required.issubset(inputs):
            raise TierShadowError(f"{variant} model inputs are incomplete")
        results[variant] = build_frozen_variant_tier_shadow(
            source_root=source_root,
            source_receipt_path=source_receipt_path,
            expected_source_receipt_file_sha256=expected_source_receipt_file_sha256,
            variant=variant,
            model_path=inputs["model_path"],
            model_receipt_path=inputs["model_receipt_path"],
            run_receipt_path=inputs["run_receipt_path"],
            expected_model_sha256=str(inputs["expected_model_sha256"]),
            expected_model_receipt_file_sha256=str(inputs["expected_model_receipt_file_sha256"]),
            expected_run_receipt_sha256=str(inputs["expected_run_receipt_sha256"]),
            current_ledger_path=current_ledger_path,
            current_receipt_path=current_receipt_path,
            expected_current_ledger_sha256=expected_current_ledger_sha256,
            expected_current_receipt_file_sha256=expected_current_receipt_file_sha256,
            scaling_ledger_path=scaling_ledger_path,
            scaling_receipt_path=scaling_receipt_path,
            expected_scaling_ledger_sha256=expected_scaling_ledger_sha256,
            expected_scaling_receipt_file_sha256=expected_scaling_receipt_file_sha256,
            destination=destinations[variant],
        )
    identities = {variant: result.game_ids for variant, result in results.items()}
    if len(set(identities.values())) != 1:
        raise TierShadowError("four-way offset universes differ")
    target_hashes: dict[str, str] = {}
    for variant, result in results.items():
        payload, _ = _load_json(result.ledger_path, f"{variant} Tier offset ledger")
        target_hashes[variant] = str(payload.get("target_rows_sha256") or "")
    if len(set(target_hashes.values())) != 1:
        raise TierShadowError("four-way offset targets differ")
    return results


build_frozen_four_variant_tier_shadows = build_frozen_fourway_tier_shadow


__all__ = [
    "AUTHORITY",
    "FINAL_MODEL_SCHEMA_VERSION",
    "MODEL_RECEIPT_SCHEMA_VERSIONS",
    "FORM_VARIANTS",
    "LEDGER_SCHEMA_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "SCALING_LEDGER_SCHEMA_VERSION",
    "SCALING_VARIANTS",
    "TierShadowError",
    "TierShadowResult",
    "VARIANTS",
    "build_frozen_four_variant_tier_shadows",
    "build_frozen_fourway_tier_shadow",
    "build_frozen_variant_tier_shadow",
    "build_variant_design",
    "build_v2_design",
    "canonical_json_bytes",
    "load_tier_offset_ledger",
    "score_variant_design",
    "score_v2_design",
    "sha256_path",
    "verify_current_rating_ledger",
    "verify_final_model",
    "verify_full_scaling_ledger",
    "verify_scaling_rating_ledger",
    "verify_source_freeze",
    "verify_target_parity",
    "verify_variant_model",
    "verify_variant_model_receipt",
    "write_tier_offset_ledger",
]
