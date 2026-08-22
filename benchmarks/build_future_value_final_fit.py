"""Fit and receipt source-bound final models for four future-value variants.

Each fit uses the frozen accepted source, one verified current-rating ledger,
and the declared dependent feature families.  It emits a model artifact with
authority set to false.  Evaluation blockers are copied into the final
receipt.  An unevaluated fit cannot become a production rating by accident.
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
    REGULARIZATION_GRID,
    CURRENT_RATING_SIGNED_MAP_FEATURES,
    FORM_METRICS,
    RANK_3,
    RATING_VARIANT_ORDER,
    SCALING_CURVE_SIGNED_MAP_FEATURES,
    FutureValueFoldModel,
    FutureValueSourceError,
    RatingVariant,
    Rank3AtomModel,
    _antisymmetric_design_matrix,
    _canonical_json_bytes,
    _frame_game_ids,
    _fit_zero_intercept_logistic,
    _map_model_frame,
    _role,
    _scaling_native_rows_sha256,
    _sha256_path,
    _stable_identity,
    _variant_imputation_values,
    build_future_value_design,
    fit_rank3_player_champion_role_atoms,
    rating_feature_values_sha256,
    rating_variant_config,
    validate_future_value_source_receipt_payload,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256
from lol_kills.research.future_value_snapshots import _latest_player_form


SCHEMA_VERSION = "scryglass:future-value-final-fit:v1"
VARIANTS = tuple(variant.value for variant in RATING_VARIANT_ORDER)
VARIANT_ORDER = VARIANTS
VARIANT_CONFIGS = {name: rating_variant_config(name).receipt() for name in VARIANTS}
AUTHORITY = {
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


def _resolve_source_file(
    source_root: Path,
    label: str,
    record: Mapping[str, Any],
) -> Path:
    locator = record.get("locator") or record.get("path")
    if not isinstance(locator, str) or not locator.strip():
        raise FinalFitError(f"source receipt file locator is missing: {label}")
    relative = Path(locator)
    if relative.is_absolute() or ".." in relative.parts:
        raise FinalFitError(f"source receipt file locator is unsafe: {label}")
    root = source_root.resolve()
    candidate = root / relative
    if candidate.is_symlink() or not candidate.is_file():
        raise FinalFitError(f"source receipt file is missing: {label}")
    path = candidate.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise FinalFitError(f"source receipt file escapes freeze root: {label}") from error
    declared_bytes = record.get("bytes")
    if isinstance(declared_bytes, bool) or not isinstance(declared_bytes, int):
        raise FinalFitError(f"source receipt file bytes are invalid: {label}")
    if declared_bytes != path.stat().st_size:
        raise FinalFitError(f"source receipt file bytes changed: {label}")
    declared_sha = str(record.get("sha256") or "").lower()
    if declared_sha != _sha256_path(path):
        raise FinalFitError(f"source receipt file hash changed: {label}")
    return path


def _verify_source_receipt(
    source_receipt: Mapping[str, Any],
    source_receipt_path: Path,
    *,
    source_root: Path,
    expected_source_receipt_sha256: str | None,
) -> None:
    source_root = Path(source_root).expanduser()
    validate_future_value_source_receipt_payload(source_receipt)
    if not expected_source_receipt_sha256:
        raise FinalFitError("independent source receipt file hash is required")
    expected_file_sha = _sha256_path(source_receipt_path)
    expected_file_sha_from_argument = str(expected_source_receipt_sha256).lower()
    if expected_file_sha != expected_file_sha_from_argument:
        raise FinalFitError("source receipt file hash changed")
    if source_root.is_symlink() or not source_root.is_dir():
        raise FinalFitError(f"source root is missing or unsafe: {source_root}")
    root = source_root.resolve()
    source_files = source_receipt.get("source_files")
    if not isinstance(source_files, Mapping):
        raise FinalFitError("source receipt file bindings are missing")
    for label, record in source_files.items():
        if not isinstance(record, Mapping):
            raise FinalFitError(f"source receipt file binding is invalid: {label}")
        _resolve_source_file(root, str(label), record)


def _verify_source_frames(
    source_root: Path,
    source_receipt: Mapping[str, Any],
) -> dict[str, Path]:
    """Verify the exact parquet frames used by the final fit."""

    source_files = source_receipt.get("source_files")
    if not isinstance(source_files, Mapping):
        raise FinalFitError("source receipt frame bindings are missing")
    verified: dict[str, Path] = {}
    root = Path(source_root).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise FinalFitError(f"source root is missing or unsafe: {source_root}")
    for label in ("maps", "players", "teams"):
        record = source_files.get(label)
        if not isinstance(record, Mapping):
            raise FinalFitError(f"frozen source frame is missing: {label}")
        try:
            verified[label] = _resolve_source_file(root, label, record)
        except FinalFitError as error:
            raise FinalFitError(f"frozen source frame {error}") from error
    return verified


def _resolve_variant(value: RatingVariant | str) -> RatingVariant:
    try:
        return value if isinstance(value, RatingVariant) else RatingVariant(str(value))
    except (TypeError, ValueError) as error:
        raise FinalFitError(f"unknown rating variant: {value!r}") from error


def _variant_feature_order(variant: RatingVariant | str) -> tuple[str, ...]:
    """Return the registered feature order for one final-fit variant."""

    return tuple(rating_variant_config(_resolve_variant(variant)).feature_names)


variant_feature_order = _variant_feature_order


def _variant_dependencies(variant: RatingVariant | str) -> dict[str, bool]:
    """Return the input families required by the selected variant."""

    resolved = _resolve_variant(variant)
    return {
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


VARIANT_DEPENDENCIES = {
    **{
        variant.value: _variant_dependencies(variant)
        for variant in RATING_VARIANT_ORDER
    },
    **{
        variant: _variant_dependencies(variant)
        for variant in RATING_VARIANT_ORDER
    },
}


def _canonical_rows_digest(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    label: str,
) -> str:
    """Hash finite values after a stable game-ID sort.

    This digest does not depend on parquet row order.  It binds the exact
    values consumed by the final model.
    """

    if "game_id" not in frame.columns:
        raise FinalFitError(f"{label} digest requires game_id")
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise FinalFitError(f"{label} digest is missing: {', '.join(missing)}")
    work = frame[["game_id", *columns]].copy()
    work["game_id"] = work["game_id"].astype(str)
    if work["game_id"].eq("").any() or work["game_id"].duplicated().any():
        raise FinalFitError(f"{label} digest game IDs are invalid")
    rows: list[dict[str, Any]] = []
    for row in work.sort_values("game_id", kind="stable").to_dict("records"):
        output: dict[str, Any] = {"game_id": str(row.pop("game_id"))}
        for column in columns:
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if not math.isfinite(float(value)):
                raise FinalFitError(f"{label} contains a non-finite value: {column}")
            output[column] = float(value)
        rows.append(output)
    return hashlib.sha256(_canonical_json_bytes(rows)).hexdigest()


def _target_digest(frame: pd.DataFrame) -> str:
    """Return the order-independent target digest for a design frame."""

    return _canonical_rows_digest(frame, ("target",), label="target")


def _form_digest(frame: pd.DataFrame) -> str:
    """Hash strict-prior player rows while retaining the ten-row game shape."""

    key_columns = ("game_id", "side", "role", "player_id", "champion")
    metric_columns = tuple(
        column
        for metric in FORM_METRICS
        for column in (
            f"prior_form_{metric}",
            f"prior_form_{metric}_support",
            f"prior_form_{metric}_effective_support",
        )
    )
    missing = sorted(set((*key_columns, *metric_columns)) - set(frame.columns))
    if missing:
        raise FinalFitError("strict-prior form digest is missing: " + ", ".join(missing))
    work = frame[[*key_columns, *metric_columns]].copy()
    for column in key_columns:
        work[column] = work[column].astype(str)
    rows: list[dict[str, Any]] = []
    for row in work.sort_values(list(key_columns), kind="stable").to_dict("records"):
        output = {column: str(row[column]) for column in key_columns}
        for column in metric_columns:
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if not math.isfinite(float(value)):
                raise FinalFitError(f"strict-prior form digest contains a non-finite value: {column}")
            output[column] = float(value)
        rows.append(output)
    return hashlib.sha256(_canonical_json_bytes(rows)).hexdigest()


def _design_digest(
    frame: pd.DataFrame,
    feature_names: tuple[str, ...] | list[str],
) -> str:
    """Return a digest of selected signed and side-level design inputs."""

    selected: dict[str, pd.Series] = {}
    for feature in feature_names:
        if feature in frame.columns:
            selected[feature] = frame[feature]
            continue
        for side in ("blue", "red"):
            column = f"__{side}_{feature}"
            if column not in frame.columns:
                raise FinalFitError(f"design digest is missing: {feature}")
            selected[column] = frame[column]
    work = frame[["game_id"]].copy()
    for column, values in selected.items():
        work[column] = values
    return _canonical_rows_digest(
        work,
        tuple(column for column in work.columns if column != "game_id"),
        label="design",
    )


design_value_digest = _design_digest
target_value_digest = _target_digest


def _neutral_form_and_atom(
    players: pd.DataFrame,
    eligible_ids: tuple[str, ...],
) -> tuple[pd.DataFrame, Rank3AtomModel]:
    """Build a deterministic zero form for variants without form inputs."""

    required = {"game_id", "player_id", "side", "role", "champion", "date", "team_id"}
    missing = sorted(required - set(players.columns))
    if missing:
        raise FinalFitError("neutral form structure is missing: " + ", ".join(missing))
    form = players[
        ["game_id", "player_id", "side", "role", "champion", "date", "team_id"]
    ].copy()
    form["game_id"] = form["game_id"].astype(str)
    form = form[form["game_id"].isin(eligible_ids)].copy()
    for metric in FORM_METRICS:
        form[f"prior_form_{metric}"] = 0.0
        form[f"prior_form_{metric}_support"] = 0.0
        form[f"prior_form_{metric}_effective_support"] = 0.0
    form["side"] = form["side"].astype(str).str.casefold()
    form["role"] = form["role"].map(_role)
    if form["role"].isna().any():
        raise FinalFitError("neutral form structure contains an unknown role")
    atom = Rank3AtomModel(
        metric_names=tuple(f"prior_form_{metric}" for metric in FORM_METRICS),
        rank=RANK_3,
        center=np.zeros(len(FORM_METRICS), dtype=float),
        scale=np.ones(len(FORM_METRICS), dtype=float),
        components=np.zeros((RANK_3, len(FORM_METRICS)), dtype=float),
        champion_role_coordinates={},
        champion_role_support={},
        fit_game_ids=eligible_ids,
        fit_window_end="neutral_no_form_input",
    )
    return form, atom


def _validate_source_stable_ids(
    players: pd.DataFrame,
    teams: pd.DataFrame,
) -> None:
    """Require non-empty stable OE player and team IDs in every final fit."""

    player_column = next(
        (column for column in ("playerid", "player_id") if column in players.columns),
        None,
    )
    player_team_column = next(
        (column for column in ("teamid", "team_id") if column in players.columns),
        None,
    )
    team_column = next(
        (column for column in ("teamid", "team_id") if column in teams.columns),
        None,
    )
    if (
        players.empty
        or teams.empty
        or player_column is None
        or player_team_column is None
        or team_column is None
    ):
        raise FinalFitError("source stable player and team ID columns are missing")
    if not players[player_column].map(
        lambda value: _stable_identity(value, "oe:player:")
    ).all():
        raise FinalFitError("source players contain an invalid stable player ID")
    if not players[player_team_column].map(
        lambda value: _stable_identity(value, "oe:team:")
    ).all():
        raise FinalFitError("source players contain an invalid stable team ID")
    if not teams[team_column].map(
        lambda value: _stable_identity(value, "oe:team:")
    ).all():
        raise FinalFitError("source teams contain an invalid stable team ID")


def _eligible_source_identity_frames(
    players: pd.DataFrame,
    teams: pd.DataFrame,
    eligible_ids: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate stable identities inside the exact model-eligible census."""

    eligible = set(eligible_ids)
    eligible_players = players.loc[
        _frame_game_ids(players, "players").astype(str).isin(eligible)
    ].copy()
    eligible_teams = teams.loc[
        _frame_game_ids(teams, "teams").astype(str).isin(eligible)
    ].copy()
    _validate_source_stable_ids(eligible_players, eligible_teams)
    return eligible_players, eligible_teams


def _evaluation_blockers(
    path: Path,
    source_receipt: Mapping[str, Any],
    variant: RatingVariant | str = RatingVariant.FUTURE_PLAYER_FORM,
) -> tuple[str, ...]:
    """Load variant evaluation evidence and bind its exact source identity."""

    resolved = _resolve_variant(variant)
    label = f"{resolved.value} evaluation"
    payload = _load_json(path, label)
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise FinalFitError(f"{label} source binding is missing")
    for field, expected in (
        ("source_as_of", source_receipt.get("source_as_of")),
        ("source_game_count", source_receipt.get("source_game_count")),
        ("source_identity_sha256", source_receipt.get("source_identity_sha256")),
        ("source_receipt_sha256", source_receipt.get("receipt_sha256")),
    ):
        if source.get(field) != expected:
            raise FinalFitError(f"{label} source binding changed: {field}")
    variants = payload.get("variants")
    variant_payload = variants.get(resolved.value) if isinstance(variants, Mapping) else None
    if not isinstance(variant_payload, Mapping):
        raise FinalFitError(f"{label} variant evidence is missing")
    if variant_payload.get("variant") not in (None, resolved.value):
        raise FinalFitError(f"{label} variant changed")
    evaluation_authority = variant_payload.get("authority")
    if isinstance(evaluation_authority, Mapping) and (
        evaluation_authority.get("research_only") is not True
        or evaluation_authority.get("promotion") is True
        or evaluation_authority.get("deployment") is True
        or evaluation_authority.get("public_probability") is True
    ):
        raise FinalFitError(f"{label} authority is invalid")
    config = rating_variant_config(resolved)
    claimed_config = variant_payload.get("variant_receipt")
    if claimed_config is not None and claimed_config != config.receipt():
        raise FinalFitError(f"{label} variant configuration changed")
    claimed_features = variant_payload.get("feature_names")
    if claimed_features is not None and tuple(claimed_features) != config.feature_names:
        raise FinalFitError(f"{label} feature order changed")
    blockers = variant_payload.get("blockers")
    if not isinstance(blockers, list):
        raise FinalFitError(f"{label} blocker list is missing")
    output = {str(value) for value in blockers}
    evaluation = variant_payload.get("evaluation")
    if not isinstance(evaluation, Mapping):
        output.add("final_calibration_receipt_missing")
        output.add("support_uncertainty_proxy_not_calibrated")
        return tuple(sorted(output))
    point = evaluation.get("strict_prior_calibration")
    if (
        not isinstance(point, Mapping)
        or point.get("status") != "available"
        or point.get("blockers")
    ):
        output.add("final_calibration_receipt_missing")
    support = evaluation.get("support_uncertainty_calibration")
    support_coverage = support.get("coverage") if isinstance(support, Mapping) else None
    if (
        not isinstance(support, Mapping)
        or support.get("status") != "research_only"
        or support.get("blockers")
        or not isinstance(support_coverage, Mapping)
        or support_coverage.get("complete_enough") is not True
        or float(support_coverage.get("calibrated_row_fraction", 0.0)) != 1.0
        or support_coverage.get("first_fold_without_history") is not False
    ):
        output.add("support_uncertainty_proxy_not_calibrated")
    return tuple(sorted(output))


def _verified_nested_selection(
    path: Path,
    source_receipt: Mapping[str, Any],
    *,
    expected_file_sha256: str | None = None,
    variant: RatingVariant = RatingVariant.FUTURE_PLAYER_FORM,
) -> dict[str, Any]:
    """Verify nested candidate-C evidence before a final fit consumes it.

    The evidence is fold-local.  Each outer fold must carry a separately bound
    inner feature ledger.  The final fit accepts a C only when every supplied
    outer fold selected the same value from the declared grid and every
    candidate fit converged.
    """

    payload = _load_json(path, "nested regularization evidence")
    if expected_file_sha256 is None:
        raise FinalFitError("independent nested selection file hash is required")
    expected_file_sha256 = str(expected_file_sha256).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_file_sha256) or _sha256_path(path) != expected_file_sha256:
        raise FinalFitError("nested selection evidence file changed")
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise FinalFitError("nested selection source binding is missing")
    for field in (
        "source_as_of",
        "source_game_count",
        "source_identity_sha256",
        "source_receipt_sha256",
    ):
        if source.get(field) != (
            source_receipt.get("receipt_sha256")
            if field == "source_receipt_sha256"
            else source_receipt.get(field)
        ):
            raise FinalFitError(f"nested selection source binding changed: {field}")
    variants = payload.get("variants")
    if isinstance(variants, Mapping):
        variant_payload = variants.get(variant.value)
        if not isinstance(variant_payload, Mapping):
            raise FinalFitError("nested selection variant is missing")
        claimed_variant_receipt = variant_payload.get("variant_receipt")
        if claimed_variant_receipt is not None and claimed_variant_receipt != rating_variant_config(variant).receipt():
            raise FinalFitError("nested selection variant configuration changed")
        claimed_feature_names = variant_payload.get("feature_names")
        if claimed_feature_names is not None and tuple(claimed_feature_names) != rating_variant_config(variant).feature_names:
            raise FinalFitError("nested selection variant feature order changed")
        folds = variant_payload.get("folds")
    elif payload.get("variant") == variant.value:
        folds = payload.get("folds")
    else:
        raise FinalFitError("nested selection variants are missing")
    if not isinstance(folds, list) or not folds:
        raise FinalFitError("nested selection fold evidence is missing")
    selections: list[dict[str, Any]] = []
    for fold in folds:
        if not isinstance(fold, Mapping):
            raise FinalFitError("nested selection fold evidence is invalid")
        selection = fold.get("regularization_selection")
        if not isinstance(selection, Mapping):
            raise FinalFitError("nested selection receipt is missing from a fold")
        if selection.get("method") != "nested_chronological_whole_series_log_loss":
            raise FinalFitError("nested selection method is not chronological")
        if selection.get("inner_ledger_status") != "verified":
            raise FinalFitError("nested selection inner ledger is not verified")
        if selection.get("blockers"):
            raise FinalFitError("nested selection carries blockers")
        if selection.get("variant") != variant.value:
            raise FinalFitError("nested selection variant changed")
        selection_receipt = selection.get("variant_receipt")
        if selection_receipt is not None and selection_receipt != rating_variant_config(variant).receipt():
            raise FinalFitError("nested selection variant configuration changed")
        grid = tuple(float(value) for value in selection.get("candidate_grid", ()))
        if grid != tuple(float(value) for value in REGULARIZATION_GRID):
            raise FinalFitError("nested selection candidate grid changed")
        selected = float(selection.get("selected_c"))
        if not math.isfinite(selected) or selected not in grid:
            raise FinalFitError("nested selection selected C is invalid")
        scores = selection.get("candidate_scores")
        if not isinstance(scores, list) or len(scores) != len(grid):
            raise FinalFitError("nested selection candidate scores are incomplete")
        score_cs = tuple(float(row.get("c")) for row in scores if isinstance(row, Mapping))
        if score_cs != grid:
            raise FinalFitError("nested selection candidate score grid changed")
        for row in scores:
            if not isinstance(row, Mapping):
                raise FinalFitError("nested selection candidate score is invalid")
            optimizer = row.get("optimizer")
            if not isinstance(optimizer, Mapping) or optimizer.get("success") is not True:
                raise FinalFitError("nested selection candidate optimizer did not converge")
            if optimizer.get("finite_coefficients") is not True:
                raise FinalFitError("nested selection candidate coefficients are non-finite")
            if not isinstance(row.get("prediction_sha256"), str) or not re.fullmatch(
                r"[0-9a-f]{64}", row["prediction_sha256"], re.I
            ):
                raise FinalFitError("nested selection prediction digest is invalid")
        binding = selection.get("inner_feature_ledger_binding")
        if not isinstance(binding, Mapping):
            raise FinalFitError("nested selection inner feature ledger binding is missing")
        config = rating_variant_config(variant)
        binding_variant = binding.get("variant")
        if binding_variant not in (None, variant.value):
            raise FinalFitError("nested selection feature ledger variant changed")
        binding_features = binding.get("feature_names")
        if binding_features is not None and tuple(binding_features) != config.signed_map_features:
            raise FinalFitError("nested selection feature ledger feature order changed")
        binding_signed_features = binding.get("signed_map_features")
        if binding_signed_features is not None and tuple(binding_signed_features) != config.signed_map_features:
            raise FinalFitError("nested selection feature ledger signed family changed")
        for field, expected in (
            ("source_receipt_sha256", source_receipt.get("receipt_sha256")),
            ("source_identity_sha256", source_receipt.get("source_identity_sha256")),
        ):
            if binding.get(field) != expected:
                raise FinalFitError(f"nested selection feature ledger source changed: {field}")
        for field in (
            "producer_receipt_sha256",
            "ledger_rows_sha256",
            "feature_value_digest",
            "game_identity_sha256",
            "fit_game_identity_sha256",
            "validation_game_identity_sha256",
            "binding_sha256",
        ):
            if not isinstance(binding.get(field), str) or not re.fullmatch(
                r"[0-9a-f]{64}", binding[field], re.I
            ):
                raise FinalFitError(f"nested selection feature ledger binding is incomplete: {field}")
        binding_payload = dict(binding)
        binding_payload.pop("binding_sha256", None)
        if _canonical_sha(binding_payload) != str(binding.get("binding_sha256") or "").lower():
            raise FinalFitError("nested selection feature ledger binding hash changed")
        artifacts = binding.get("producer_artifacts")
        if not isinstance(artifacts, Mapping) or not artifacts:
            raise FinalFitError("nested selection producer artifact bindings are missing")

        def verify_artifact_records(value: Any, label: str) -> int:
            if isinstance(value, Mapping):
                if set(value) == {"path", "bytes", "sha256"}:
                    artifact_path = Path(str(value["path"]))
                    if (
                        not artifact_path.is_absolute()
                        or artifact_path.is_symlink()
                        or not artifact_path.is_file()
                        or int(value["bytes"]) != artifact_path.stat().st_size
                        or str(value["sha256"]).lower() != _sha256_path(artifact_path)
                    ):
                        raise FinalFitError(f"nested selection artifact binding changed: {label}")
                    return 1
                return sum(
                    verify_artifact_records(child, f"{label}.{key}")
                    for key, child in value.items()
                )
            if isinstance(value, (list, tuple)):
                return sum(
                    verify_artifact_records(child, f"{label}[{index}]")
                    for index, child in enumerate(value)
                )
            return 0

        if verify_artifact_records(artifacts, "producer_artifacts") < 1:
            raise FinalFitError("nested selection producer artifact records are missing")
        train_ids = tuple(str(value) for value in binding.get("fit_game_ids", ()))
        validation_ids = tuple(str(value) for value in binding.get("validation_game_ids", ()))
        if not train_ids or not validation_ids or set(train_ids) & set(validation_ids):
            raise FinalFitError("nested selection feature ledger IDs are invalid")
        inner_start = selection.get("inner_validation_start")
        inner_end = selection.get("inner_validation_end")
        fit_cutoff = binding.get("fit_window_end")
        fit_max = binding.get("fit_date_max")
        validation_min = binding.get("validation_date_min")
        validation_max = binding.get("validation_date_max")
        if not all(
            isinstance(value, str)
            and value
            for value in (inner_start, inner_end, fit_cutoff, fit_max, validation_min, validation_max)
        ):
            raise FinalFitError("nested selection chronology evidence is incomplete")
        inner_start_stamp = pd.Timestamp(inner_start)
        inner_end_stamp = pd.Timestamp(inner_end)
        cutoff_stamp = pd.Timestamp(fit_cutoff)
        fit_max_stamp = pd.Timestamp(fit_max)
        validation_min_stamp = pd.Timestamp(validation_min)
        validation_max_stamp = pd.Timestamp(validation_max)
        if any(
            stamp.tzinfo is None
            for stamp in (
                inner_start_stamp,
                inner_end_stamp,
                cutoff_stamp,
                fit_max_stamp,
                validation_min_stamp,
                validation_max_stamp,
            )
        ):
            raise FinalFitError("nested selection chronology timestamps must include a timezone")
        if not (
            cutoff_stamp == inner_start_stamp
            and fit_max_stamp < inner_start_stamp
            and validation_min_stamp >= inner_start_stamp
            and validation_max_stamp <= inner_end_stamp
            and validation_min_stamp <= validation_max_stamp
        ):
            raise FinalFitError("nested selection chronology violates strict prior timing")
        if binding.get("fit_game_identity_sha256") != identity_sha256(train_ids):
            raise FinalFitError("nested selection inner training identity changed")
        if binding.get("validation_game_identity_sha256") != identity_sha256(validation_ids):
            raise FinalFitError("nested selection inner validation identity changed")
        selections.append(
            {
                "fold": int(fold.get("fold") or 0),
                "selected_c": selected,
                "candidate_grid": list(grid),
                "inner_feature_ledger_binding": dict(binding),
                "inner_train_identity_sha256": binding["fit_game_identity_sha256"],
                "inner_validation_identity_sha256": binding[
                    "validation_game_identity_sha256"
                ],
                "inner_validation_start": selection.get("inner_validation_start"),
                "inner_validation_end": selection.get("inner_validation_end"),
            }
        )
    selected_values = {float(row["selected_c"]) for row in selections}
    if len(selected_values) != 1:
        raise FinalFitError("nested selection folds disagree on selected C")
    return {
        "schema_version": "scryglass:future-value-nested-selection-binding:v1",
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": expected_file_sha256,
        "source_receipt_sha256": str(source_receipt["receipt_sha256"]),
        "source_identity_sha256": str(source_receipt["source_identity_sha256"]),
        "variant": variant.value,
        "folds": selections,
        "selected_c": float(next(iter(selected_values))),
        "candidate_grid": list(float(value) for value in REGULARIZATION_GRID),
    }


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
    file_receipt = _load_json(current_receipt_path, "current rating receipt file")
    if dict(file_receipt) != dict(current_receipt):
        raise FinalFitError("current rating receipt payload changed")
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
    current_authority = current_receipt.get("authority")
    if isinstance(current_authority, Mapping) and (
        current_authority.get("research_only") is not True
        or current_authority.get("promotion") is True
        or current_authority.get("deployment") is True
    ):
        raise FinalFitError("current rating ledger authority is invalid")
    claimed_features = current_receipt.get("feature_names")
    if claimed_features is not None and tuple(claimed_features) != tuple(
        CURRENT_RATING_SIGNED_MAP_FEATURES
    ):
        raise FinalFitError("current rating ledger feature order changed")
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
    receipt_ids = current_receipt.get("fit_game_ids")
    if receipt_ids is not None and tuple(sorted(str(value) for value in receipt_ids)) != fit_game_ids:
        raise FinalFitError("current rating ledger receipt game census changed")
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
        "ledger_rows_sha256": str(current_receipt.get("ledger_rows_sha256") or "") or None,
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
        "authority": dict(current_authority)
        if isinstance(current_authority, Mapping)
        else dict(AUTHORITY),
    }


def _verify_file_record(path: Path, record: Mapping[str, Any], label: str) -> dict[str, Any]:
    """Verify one absolute file record and return its canonical copy."""

    if path.is_symlink() or not path.is_file():
        raise FinalFitError(f"{label} is missing or unsafe")
    if int(record.get("bytes") or -1) != path.stat().st_size:
        raise FinalFitError(f"{label} bytes changed")
    digest = str(record.get("sha256") or "").lower()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None or digest != _sha256_path(path):
        raise FinalFitError(f"{label} hash changed")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def _verify_manifest_file_records(value: Any, label: str) -> int:
    """Verify recursive absolute records in an online producer manifest."""

    if isinstance(value, Mapping):
        if set(value) == {"path", "bytes", "sha256"}:
            path = Path(str(value["path"]))
            _verify_file_record(path, value, label)
            return 1
        return sum(
            _verify_manifest_file_records(child, f"{label}.{key}")
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return sum(
            _verify_manifest_file_records(child, f"{label}[{index}]")
            for index, child in enumerate(value)
        )
    return 0


def _bind_scaling_features(
    scaling_frame: pd.DataFrame,
    scaling_receipt: Mapping[str, Any],
    *,
    source_receipt: Mapping[str, Any],
    source_receipt_path: Path,
    scaling_artifact_path: Path,
    scaling_receipt_path: Path,
    fit_game_ids: tuple[str, ...],
    fit_window_end: str,
    expected_scaling_receipt_sha256: str | None,
    expected_scaling_artifact_sha256: str | None,
    scaling_manifest_path: Path | None = None,
    expected_scaling_manifest_sha256: str | None = None,
    expected_scaling_feature_receipt_sha256: str | None = None,
    expected_scaling_feature_artifact_sha256: str | None = None,
    expected_scaling_feature_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind the one online full-census scaling ledger used by final fits."""

    expected_receipt_hash = str(
        expected_scaling_receipt_sha256 or expected_scaling_feature_receipt_sha256 or ""
    ).lower()
    expected_artifact_hash = str(
        expected_scaling_artifact_sha256 or expected_scaling_feature_artifact_sha256 or ""
    ).lower()
    if re.fullmatch(r"[0-9a-f]{64}", expected_receipt_hash) is None:
        raise FinalFitError("independent scaling feature receipt hash is required")
    if re.fullmatch(r"[0-9a-f]{64}", expected_artifact_hash) is None:
        raise FinalFitError("independent scaling feature artifact hash is required")
    receipt_record = _verify_file_record(
        scaling_receipt_path,
        {"bytes": scaling_receipt_path.stat().st_size, "sha256": expected_receipt_hash},
        "scaling feature receipt file",
    )
    if receipt_record["sha256"] != expected_receipt_hash:
        raise FinalFitError("scaling feature receipt file hash changed")
    file_receipt = _load_json(scaling_receipt_path, "scaling feature receipt file")
    if dict(file_receipt) != dict(scaling_receipt):
        raise FinalFitError("scaling feature receipt payload changed")
    artifact_record = _verify_file_record(
        scaling_artifact_path,
        {"bytes": scaling_artifact_path.stat().st_size, "sha256": expected_artifact_hash},
        "scaling feature artifact file",
    )
    claimed_receipt_hash = str(scaling_receipt.get("receipt_sha256") or "").lower()
    receipt_payload = dict(scaling_receipt)
    receipt_payload.pop("receipt_sha256", None)
    if len(claimed_receipt_hash) != 64 or _canonical_sha(receipt_payload) != claimed_receipt_hash:
        raise FinalFitError("scaling feature receipt self-hash changed")
    if scaling_receipt.get("schema_version") != "scryglass:atomized-scaling-feature-ledger:v1":
        raise FinalFitError("scaling feature receipt schema changed")
    if scaling_receipt.get("status") != "research_only" or scaling_receipt.get("authority") is not False:
        raise FinalFitError("scaling feature receipt authority is invalid")
    if scaling_receipt.get("public_authority") is not False:
        raise FinalFitError("scaling feature receipt public authority is invalid")
    if scaling_receipt.get("evaluation_mode") != "online_full_census":
        raise FinalFitError("scaling feature receipt must use online_full_census")
    if scaling_receipt.get("source_receipt_sha256") != source_receipt.get("receipt_sha256"):
        raise FinalFitError("scaling feature source receipt changed")
    if scaling_receipt.get("source_identity_sha256") != source_receipt.get("source_identity_sha256"):
        raise FinalFitError("scaling feature source identity changed")
    try:
        source_cutoff = pd.Timestamp(source_receipt["source_as_of"])
        scaling_cutoff = pd.Timestamp(scaling_receipt["source_as_of"])
        if source_cutoff.tzinfo is None or scaling_cutoff.tzinfo is None:
            raise ValueError("timezone is missing")
        source_cutoff_text = source_cutoff.tz_convert("UTC").isoformat().replace("+00:00", "Z")
        scaling_cutoff_text = scaling_cutoff.tz_convert("UTC").isoformat().replace("+00:00", "Z")
    except (KeyError, TypeError, ValueError) as error:
        raise FinalFitError("scaling feature source cutoff is invalid") from error
    if scaling_cutoff_text != source_cutoff_text:
        raise FinalFitError("scaling feature source cutoff changed")
    expected_ids = tuple(sorted(str(value) for value in fit_game_ids))
    output_ids = tuple(sorted(str(value) for value in scaling_receipt.get("output_game_ids", ())))
    if output_ids != expected_ids:
        raise FinalFitError("scaling feature output census changed")
    if scaling_receipt.get("output_game_count") != len(expected_ids):
        raise FinalFitError("scaling feature output count changed")
    if scaling_receipt.get("output_identity_sha256") != identity_sha256(expected_ids):
        raise FinalFitError("scaling feature output identity changed")
    if len(scaling_frame) != len(expected_ids):
        raise FinalFitError("scaling feature row count changed")
    if scaling_receipt.get("rows") is not None and scaling_receipt.get("rows") != len(scaling_frame):
        raise FinalFitError("scaling feature receipt row count changed")
    if tuple(sorted(scaling_frame["game_id"].astype(str))) != expected_ids:
        raise FinalFitError("scaling feature artifact census changed")
    artifact_claim = scaling_receipt.get("artifact_sha256")
    if artifact_claim is not None and str(artifact_claim).lower() != artifact_record["sha256"]:
        raise FinalFitError("scaling feature artifact binding changed")
    if scaling_receipt.get("artifact_bytes") is not None and scaling_receipt.get("artifact_bytes") != artifact_record["bytes"]:
        raise FinalFitError("scaling feature artifact bytes changed")
    feature_names = tuple(str(value) for value in SCALING_CURVE_SIGNED_MAP_FEATURES)
    claimed_columns = scaling_receipt.get("feature_names") or scaling_receipt.get("columns")
    if claimed_columns is not None:
        claimed_set = set(str(value) for value in claimed_columns)
        if not set(feature_names).issubset(claimed_set):
            raise FinalFitError("scaling feature order is incomplete")
        if scaling_receipt.get("columns") is not None and tuple(str(value) for value in scaling_receipt["columns"]) != tuple(str(value) for value in scaling_frame.columns):
            raise FinalFitError("scaling feature artifact columns changed")
    if not set(feature_names).issubset(scaling_frame.columns):
        raise FinalFitError("scaling feature columns are incomplete")
    values = scaling_frame[list(feature_names)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise FinalFitError("scaling feature artifact contains non-finite values")
    value_digest = rating_feature_values_sha256(scaling_frame, feature_names)
    declared_feature_digest = str(
        scaling_receipt.get("feature_value_digest")
        or scaling_receipt.get("row_feature_digest_sha256")
        or ""
    ).lower()
    row_digest = str(scaling_receipt.get("row_value_digest_sha256") or "").lower()
    if re.fullmatch(r"[0-9a-f]{64}", declared_feature_digest) is not None and declared_feature_digest != value_digest:
        raise FinalFitError("scaling feature value digest changed")
    if re.fullmatch(r"[0-9a-f]{64}", row_digest) is None:
        raise FinalFitError("scaling feature row digest is required")
    if {"date", "game_id"}.issubset(scaling_frame.columns):
        recomputed_row_digest = _scaling_native_rows_sha256(scaling_frame)
    else:
        recomputed_row_digest = value_digest
    if row_digest != recomputed_row_digest:
        raise FinalFitError("scaling feature row digest changed")
    if declared_feature_digest and declared_feature_digest != value_digest:
        raise FinalFitError("scaling feature value digest changed")
    source_file_record = None
    if source_receipt_path.is_file() and not source_receipt_path.is_symlink():
        source_file_record = {
            "path": str(source_receipt_path),
            "bytes": source_receipt_path.stat().st_size,
            "sha256": _sha256_path(source_receipt_path),
        }
    manifest_record = None
    manifest = None
    if scaling_manifest_path is not None:
        expected_manifest_hash = str(
            expected_scaling_manifest_sha256
            or expected_scaling_feature_manifest_sha256
            or ""
        ).lower()
        if re.fullmatch(r"[0-9a-f]{64}", expected_manifest_hash) is None:
            raise FinalFitError("independent scaling feature manifest hash is required")
        manifest = _load_json(scaling_manifest_path, "scaling feature manifest")
        manifest_record = _verify_file_record(
            scaling_manifest_path,
            {"bytes": scaling_manifest_path.stat().st_size, "sha256": expected_manifest_hash},
            "scaling feature manifest file",
        )
        manifest_hash_field = (
            "manifest_sha256" if manifest.get("manifest_sha256") is not None else "receipt_sha256"
        )
        claimed_manifest_hash = str(manifest.get(manifest_hash_field) or "").lower()
        manifest_payload = dict(manifest)
        manifest_payload.pop(manifest_hash_field, None)
        if len(claimed_manifest_hash) != 64 or _canonical_sha(manifest_payload) != claimed_manifest_hash:
            raise FinalFitError("scaling feature manifest self-hash changed")
        manifest_authority = manifest.get("authority")
        if manifest_authority is not False and not isinstance(manifest_authority, Mapping):
            raise FinalFitError("scaling feature manifest authority is invalid")
        if isinstance(manifest_authority, Mapping) and (
            manifest_authority.get("research_only") is not True
            or manifest_authority.get("promotion") is True
            or manifest_authority.get("deployment") is True
        ):
            raise FinalFitError("scaling feature manifest authority is invalid")
        if manifest.get("schema_version") == "scryglass:scaling-ledger-artifact:v1":
            if str(manifest.get("artifact_sha256") or "").lower() != artifact_record["sha256"]:
                raise FinalFitError("scaling feature manifest artifact changed")
            manifest_artifact_path = Path(str(manifest.get("artifact_path") or ""))
            if manifest_artifact_path.resolve() != scaling_artifact_path.resolve():
                raise FinalFitError("scaling feature manifest artifact path changed")
            _verify_file_record(
                manifest_artifact_path,
                {
                    "bytes": manifest.get("artifact_bytes"),
                    "sha256": manifest.get("artifact_sha256"),
                },
                "scaling feature manifest artifact",
            )
            producer_path = Path(str(manifest.get("producer_receipt_path") or ""))
            if producer_path.resolve() != scaling_receipt_path.resolve():
                raise FinalFitError("scaling feature manifest producer receipt path changed")
            producer_hash = str(manifest.get("producer_receipt_file_sha256") or "").lower()
            if re.fullmatch(r"[0-9a-f]{64}", producer_hash) is None:
                raise FinalFitError("scaling feature manifest producer receipt hash is missing")
            producer_record = _verify_file_record(
                producer_path,
                {"bytes": producer_path.stat().st_size, "sha256": producer_hash},
                "scaling feature manifest producer receipt",
            )
            producer_payload = _load_json(producer_path, "scaling producer receipt")
            if str(producer_payload.get("receipt_sha256") or "").lower() != str(
                manifest.get("producer_receipt_sha256") or ""
            ).lower():
                raise FinalFitError("scaling feature manifest producer receipt changed")
        elif _verify_manifest_file_records(manifest, "scaling feature manifest") < 1:
            raise FinalFitError("scaling feature manifest file bindings are missing")
    return {
        "schema_version": "scryglass:future-value-final-scaling-feature-binding:v1",
        "producer_name": "strict_prior_atomized_scaling",
        "producer_receipt_sha256": claimed_receipt_hash,
        "producer_receipt_schema_version": str(scaling_receipt.get("schema_version") or ""),
        "producer_receipt_file": receipt_record,
        "artifact": artifact_record,
        "manifest_file": manifest_record,
        "manifest_sha256": (
            str(manifest.get("manifest_sha256") or manifest.get("receipt_sha256"))
            if manifest
            else None
        ),
        "source_receipt_sha256": str(source_receipt["receipt_sha256"]),
        "source_identity_sha256": str(source_receipt["source_identity_sha256"]),
        "source_receipt_file": source_file_record,
        "feature_names": list(feature_names),
        "feature_value_digest": value_digest,
        "row_value_digest_sha256": row_digest,
        "fit_game_ids": list(expected_ids),
        "fit_game_identity_sha256": identity_sha256(expected_ids),
        "fit_window_end": fit_window_end,
        "evaluation_mode": "online_full_census",
        "authority": dict(AUTHORITY),
    }


_bind_scaling_curve_features = _bind_scaling_features


def _candidate_path(root: Path | None, names: tuple[str, ...], label: str) -> Path:
    if root is None:
        raise FinalFitError(f"{label} root is required")
    for name in names:
        path = root / name
        if path.is_file() and not path.is_symlink():
            return path
    raise FinalFitError(f"{label} artifact is missing")


def fit_final_variant(
    *,
    variant: RatingVariant | str,
    source_root: Path,
    source_receipt_path: Path,
    current_root: Path,
    evaluation_path: Path,
    output_dir: Path,
    baseline_cache_path: Path | None = None,
    scaling_root: Path | None = None,
    scaling_artifact_path: Path | None = None,
    scaling_receipt_path: Path | None = None,
    scaling_manifest_path: Path | None = None,
    expected_source_receipt_sha256: str | None = None,
    expected_evaluation_sha256: str | None = None,
    expected_evaluation_file_sha256: str | None = None,
    expected_current_receipt_sha256: str | None = None,
    expected_current_artifact_sha256: str | None = None,
    expected_scaling_receipt_sha256: str | None = None,
    expected_scaling_artifact_sha256: str | None = None,
    expected_scaling_manifest_sha256: str | None = None,
    expected_scaling_feature_receipt_sha256: str | None = None,
    expected_scaling_feature_artifact_sha256: str | None = None,
    expected_scaling_feature_manifest_sha256: str | None = None,
    nested_selection_path: Path | None = None,
    expected_nested_selection_sha256: str | None = None,
) -> dict[str, Any]:
    """Fit one source-bound member of the four-variant final-fit family."""

    resolved = _resolve_variant(variant)
    dependencies = _variant_dependencies(resolved)
    config = rating_variant_config(resolved)
    source_root = Path(source_root).expanduser()
    source_receipt = _load_json(source_receipt_path, "source receipt")
    _verify_source_receipt(
        source_receipt,
        source_receipt_path,
        source_root=source_root,
        expected_source_receipt_sha256=expected_source_receipt_sha256,
    )
    source_root = source_root.resolve()
    evaluation_hashes = tuple(
        str(value).lower()
        for value in (expected_evaluation_sha256, expected_evaluation_file_sha256)
        if value is not None
    )
    if not evaluation_hashes:
        raise FinalFitError("independent evaluation file hash is required")
    if len(set(evaluation_hashes)) != 1:
        raise FinalFitError("evaluation file hash arguments disagree")
    expected_evaluation_hash = evaluation_hashes[0]
    if re.fullmatch(r"[0-9a-f]{64}", expected_evaluation_hash) is None:
        raise FinalFitError("independent evaluation file hash is invalid")
    if _sha256_path(evaluation_path) != expected_evaluation_hash:
        raise FinalFitError("evaluation evidence file changed")
    evaluation_blockers = _evaluation_blockers(
        evaluation_path, source_receipt, resolved
    )
    if nested_selection_path is None:
        raise FinalFitError("verified nested selection evidence is required")
    if "nested_inner_feature_ledger_missing_fixed_c_used" in evaluation_blockers:
        raise FinalFitError(
            "evaluation still carries the missing nested feature ledger blocker"
        )
    nested_selection = _verified_nested_selection(
        nested_selection_path,
        source_receipt,
        expected_file_sha256=expected_nested_selection_sha256,
        variant=resolved,
    )
    eligible_ids = tuple(
        sorted(str(value) for value in source_receipt["model_eligible_game_ids"])
    )
    source_frames = _verify_source_frames(source_root, source_receipt)
    source_maps_path = source_frames["maps"]
    source_players_path = source_frames["players"]
    source_teams_path = source_frames["teams"]
    maps = pd.read_parquet(source_maps_path)
    players = pd.read_parquet(source_players_path)
    teams = pd.read_parquet(source_teams_path)
    players, teams = _eligible_source_identity_frames(players, teams, eligible_ids)
    model_frame = _map_model_frame(maps)
    model_frame = model_frame[
        model_frame["game_id"].astype(str).isin(eligible_ids)
    ].copy()
    if tuple(sorted(model_frame["game_id"].astype(str))) != eligible_ids:
        raise FinalFitError("model maps do not match the frozen eligible census")

    current_artifact_path = _candidate_path(
        current_root,
        ("current-rating-ledger.parquet", "current-rating-feature-ledger.parquet"),
        "current rating",
    )
    current_receipt_path = _candidate_path(
        current_root,
        (
            "current-rating-ledger-receipt.json",
            "current-rating-feature-ledger.receipt.json",
        ),
        "current rating receipt",
    )
    current_frame = pd.read_parquet(current_artifact_path)
    current_receipt = _load_json(current_receipt_path, "current rating ledger receipt")
    fit_window_end = str(source_receipt["source_as_of"])
    current_binding = _bind_current_rating_features(
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

    scaling_binding = None
    scaling_frame = None
    if dependencies["scaling_curve"]:
        if scaling_artifact_path is None and scaling_root is None:
            raise FinalFitError(
                f"scaling feature receipt is required for variant {resolved.value}"
            )
        if scaling_receipt_path is None and scaling_root is None:
            raise FinalFitError(
                f"scaling feature receipt is required for variant {resolved.value}"
            )
        if scaling_artifact_path is None:
            scaling_artifact_path = _candidate_path(
                scaling_root,
                (
                    "scaling-feature-ledger-online.parquet",
                    "scaling-features.parquet",
                    "scaling-feature-ledger.parquet",
                ),
                "scaling feature",
            )
        if scaling_receipt_path is None:
            scaling_receipt_path = _candidate_path(
                scaling_root,
                (
                    "scaling-feature-ledger-online-receipt.json",
                    "scaling-native-receipt.json",
                    "scaling-producer-receipt.json",
                    "scaling-feature-ledger-receipt.json",
                ),
                "scaling feature receipt",
            )
        if scaling_manifest_path is None and scaling_root is not None:
            for candidate in (
                "scaling-feature-ledger-online-manifest.json",
                "scaling-producer-manifest.json",
                "scaling-feature-ledger-manifest.json",
            ):
                path = scaling_root / candidate
                if path.is_file() and not path.is_symlink():
                    scaling_manifest_path = path
                    break
        scaling_frame = pd.read_parquet(scaling_artifact_path)
        scaling_receipt = _load_json(
            scaling_receipt_path, "scaling feature ledger receipt"
        )
        scaling_binding = _bind_scaling_features(
            scaling_frame,
            scaling_receipt,
            source_receipt=source_receipt,
            source_receipt_path=source_receipt_path,
            scaling_artifact_path=scaling_artifact_path,
            scaling_receipt_path=scaling_receipt_path,
            scaling_manifest_path=scaling_manifest_path,
            fit_game_ids=eligible_ids,
            fit_window_end=fit_window_end,
            expected_scaling_receipt_sha256=expected_scaling_receipt_sha256,
            expected_scaling_artifact_sha256=expected_scaling_artifact_sha256,
            expected_scaling_manifest_sha256=expected_scaling_manifest_sha256,
            expected_scaling_feature_receipt_sha256=expected_scaling_feature_receipt_sha256,
            expected_scaling_feature_artifact_sha256=expected_scaling_feature_artifact_sha256,
            expected_scaling_feature_manifest_sha256=expected_scaling_feature_manifest_sha256,
        )

    started = time.perf_counter()
    normalized_players = players.copy()
    normalized_players["game_id"] = _frame_game_ids(
        normalized_players, "players"
    ).astype(str)
    normalized_players["player_id"] = normalized_players["playerid"].astype(str)
    normalized_players["team_id"] = normalized_players["teamid"].astype(str)
    normalized_players["side"] = normalized_players["side"].astype(str).str.casefold()
    normalized_players["role"] = normalized_players["position"].map(_role)
    normalized_players["date"] = pd.to_datetime(
        normalized_players.get("date"), utc=True, errors="coerce"
    )
    if dependencies["future_player_form"]:
        form_cache = None
        if baseline_cache_path is not None and baseline_cache_path.is_file():
            cache_manifest = _load_json(baseline_cache_path, "baseline cache manifest")
            form_cache = PrefixBaselineCache(
                storage_path=baseline_cache_path,
                source_identity=str(cache_manifest.get("source_identity") or ""),
                schema_fingerprint=str(cache_manifest.get("schema_fingerprint") or ""),
            )
        form = _latest_player_form(
            maps,
            normalized_players,
            baseline_cache=form_cache,
        )
        for metric in FORM_METRICS:
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
            fit_window_end=None,
        )
        form_binding = {
            "status": "verified",
            "fit_game_ids": list(eligible_ids),
            "fit_game_identity_sha256": identity_sha256(eligible_ids),
            "eligible_game_ids": list(eligible_ids),
            "eligible_game_identity_sha256": identity_sha256(eligible_ids),
            "form_value_digest": _form_digest(form),
            "rank_3_parameter_sha256": atom_model.parameter_receipt()["parameter_sha256"],
        }
    else:
        form, atom_model = _neutral_form_and_atom(normalized_players, eligible_ids)
        form_binding = {
            "status": "not_used",
            "reason": "variant_does_not_select_future_player_form",
        }

    design = build_future_value_design(
        model_frame,
        form,
        atom_model,
        verified_model_frame=model_frame,
    )
    design.attrs["variant"] = resolved.value
    design.attrs["variant_receipt"] = config.receipt()
    current_join = current_frame[["game_id", *CURRENT_RATING_SIGNED_MAP_FEATURES]].copy()
    current_join["game_id"] = current_join["game_id"].astype(str)
    design = design.merge(
        current_join,
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    if design[list(CURRENT_RATING_SIGNED_MAP_FEATURES)].isna().any().any():
        raise FinalFitError("current rating feature join is incomplete")
    if scaling_frame is not None:
        scaling_join = scaling_frame[["game_id", *SCALING_CURVE_SIGNED_MAP_FEATURES]].copy()
        scaling_join["game_id"] = scaling_join["game_id"].astype(str)
        design = design.merge(
            scaling_join,
            on="game_id",
            how="left",
            validate="one_to_one",
        )
        if design[list(SCALING_CURVE_SIGNED_MAP_FEATURES)].isna().any().any():
            raise FinalFitError("scaling feature join is incomplete")
    feature_names = tuple(config.feature_names)
    for feature in feature_names:
        if feature in config.signed_map_features:
            if feature not in design.columns:
                raise FinalFitError(f"final design is missing signed feature: {feature}")
        elif f"__blue_{feature}" not in design.columns or f"__red_{feature}" not in design.columns:
            raise FinalFitError(f"final design is missing side feature: {feature}")
    imputation = _variant_imputation_values(design, config)
    matrix = _antisymmetric_design_matrix(
        design,
        imputation,
        feature_names=feature_names,
    )
    scales = matrix.std(axis=0, ddof=0)
    scales = np.where(np.isfinite(scales) & (scales > 1e-12), scales, 1.0)
    target = pd.to_numeric(design["target"], errors="coerce")
    if not target.isin({0, 1}).all() or target.nunique() != 2:
        raise FinalFitError("final fit target is incomplete")
    classifier, optimizer = _fit_zero_intercept_logistic(
        matrix / scales,
        target.to_numpy(dtype=int),
        regularization_c=float(nested_selection["selected_c"]),
    )
    source_frame_hashes = {
        "maps": _source_frame_sha256(source_maps_path),
        "players": _source_frame_sha256(source_players_path),
        "teams": _source_frame_sha256(source_teams_path),
    }
    producer_names = ["current_sequential_rating"]
    if scaling_binding is not None:
        producer_names.append("strict_prior_atomized_scaling")
    ledger_binding: dict[str, Any] = {
        "schema_version": "scryglass:future-value-final-feature-binding:v1",
        "variant": resolved.value,
        "feature_names": list(feature_names),
        "signed_map_features": list(config.signed_map_features),
        "producer_names": producer_names,
        "current_rating": current_binding,
        "scaling_curve": scaling_binding,
        "source_frame_sha256": source_frame_hashes,
        "fit_game_ids": list(eligible_ids),
        "fit_game_identity_sha256": identity_sha256(eligible_ids),
        "validation_game_ids": [],
        "validation_game_identity_sha256": identity_sha256(()),
        "fit_window_end": fit_window_end,
        "fit_date_min": str(current_frame["date"].min()),
        "fit_date_max": str(current_frame["date"].max()),
    }
    ledger_binding["binding_sha256"] = _canonical_sha(ledger_binding)
    model = FutureValueFoldModel(
        feature_names=feature_names,
        means=np.zeros(matrix.shape[1], dtype=float),
        scales=scales.astype(float),
        imputation_values=np.asarray(imputation, dtype=float),
        coefficients=classifier.coef_[0].astype(float),
        intercept=0.0,
        regularization_selection={
            "method": "verified_nested_chronological_whole_series_log_loss",
            "candidate_grid": list(nested_selection["candidate_grid"]),
            "selected_c": float(nested_selection["selected_c"]),
            "inner_ledger_status": "verified",
            "blockers": [],
            "selection_scope": "verified_outer_fold_inner_evidence",
            "nested_selection_binding": nested_selection,
        },
        optimizer_evidence=optimizer,
        atom_model=atom_model,
        fit_game_ids=eligible_ids,
        fit_window_end=fit_window_end,
        train_rows=len(design),
        withheld_rows=0,
        source_receipt=dict(source_receipt),
        variant=resolved,
        feature_ledger_binding=ledger_binding,
    )
    parameters = model.parameter_receipt()
    blockers = set(evaluation_blockers)
    blockers.update(model.regularization_selection.get("blockers", ()))
    target_digest = _target_digest(design)
    design_digest = _design_digest(design, feature_names)
    receipt = model.receipt()
    receipt.update(
        {
            "schema_version": "scryglass:future-value-model-fit:v1",
            "status": "research_only_blocked",
            "fit_contract_schema_version": SCHEMA_VERSION,
            "fit_game_ids": list(eligible_ids),
            "fit_game_identity_sha256": identity_sha256(eligible_ids),
            "fit_window_end": fit_window_end,
            "variant": resolved.value,
            "variant_config": config.receipt(),
            "variant_dependencies": dependencies,
            "form_contract": (
                "strict_prior_role_and_competition_tier_normalized_v1"
                if dependencies["future_player_form"]
                else "not_used"
            ),
            "form_binding": form_binding,
            "current_rating_feature_binding": current_binding,
            "scaling_feature_binding": scaling_binding,
            "scaling_curve_feature_binding": scaling_binding,
            "source_frame_sha256": source_frame_hashes,
            "target_digest_sha256": target_digest,
            "design_digest_sha256": design_digest,
            "feature_value_digests": {
                "current_rating": current_binding["feature_value_digest"],
                "future_player_form": form_binding.get("form_value_digest"),
                "scaling_curve": scaling_binding["feature_value_digest"]
                if scaling_binding is not None
                else None,
            },
            "code_binding": {
                "future_value_rating_sha256": _sha256_path(
                    Path(__file__).resolve().parents[1]
                    / "lol_kills/research/future_value_rating.py"
                ),
                "future_value_rating_ledger_sha256": current_binding["code"][
                    "implementation_sha256"
                ],
                "snapshot_producer_sha256": _sha256_path(
                    Path(__file__).resolve().parents[1]
                    / "lol_kills/research/future_value_snapshots.py"
                ),
            },
            "transformation_binding": {
                "variant": resolved.value,
                "variant_receipt": config.receipt(),
                "variant_config_sha256": config.config_sha256,
                "feature_names": list(feature_names),
                "signed_map_features": list(config.signed_map_features),
                "side_level_features": list(config.side_level_features),
                "feature_schema_sha256": _canonical_sha(config.receipt()),
                "imputation_values": [float(value) for value in imputation],
                "scales": [float(value) for value in scales],
                "rank_3_parameter_sha256": parameters["rank_3"]["parameter_sha256"],
                "parameter_sha256": parameters["parameter_sha256"],
                "target_digest_sha256": target_digest,
                "design_digest_sha256": design_digest,
            },
            "optimizer_convergence": optimizer,
            "evaluation_receipt": {
                "path": str(evaluation_path),
                "bytes": evaluation_path.stat().st_size,
                "sha256": _sha256_path(evaluation_path),
                "variant": resolved.value,
                "blockers": list(evaluation_blockers),
            },
            "nested_selection_receipt": nested_selection,
            "authority": dict(AUTHORITY),
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
    model_name = f"final-v{config.ordinal}-model.json"
    receipt_name = f"final-v{config.ordinal}-model-receipt.json"
    model_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": receipt["status"],
        "authority": receipt["authority"],
        "source": receipt["source_binding"],
        "variant": resolved.value,
        "receipt_sha256": receipt["receipt_sha256"],
        "receipt": receipt,
        "parameters": parameters,
    }
    (output_dir / model_name).write_text(
        json.dumps(model_payload, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / receipt_name).write_text(
        json.dumps(receipt, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    run = {
        "schema_version": SCHEMA_VERSION,
        "status": receipt["status"],
        "variant": resolved.value,
        "variant_config_sha256": config.config_sha256,
        "source_receipt_sha256": source_receipt["receipt_sha256"],
        "source_identity_sha256": source_receipt["source_identity_sha256"],
        "fit_game_count": len(eligible_ids),
        "fit_game_identity_sha256": identity_sha256(eligible_ids),
        "eligible_game_ids": list(eligible_ids),
        "eligible_game_identity_sha256": identity_sha256(eligible_ids),
        "fit_window_end": fit_window_end,
        "feature_names": list(feature_names),
        "target_digest_sha256": target_digest,
        "design_digest_sha256": design_digest,
        "nested_selection": nested_selection,
        "model_receipt_sha256": receipt["receipt_sha256"],
        "model_artifact_sha256": _sha256_path(output_dir / model_name),
        "blockers": sorted(blockers),
        "authority": dict(AUTHORITY),
    }
    (output_dir / "final-fit-run.json").write_text(
        json.dumps(run, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return run


def fit_final(
    *,
    variant: RatingVariant | str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Public generic entry point for one of the four final-fit variants."""

    return fit_final_variant(variant=variant, **kwargs)


def fit_final_v2(**kwargs: Any) -> dict[str, Any]:
    """Compatibility entry point for the former V2-only builder."""

    return fit_final_variant(variant=RatingVariant.FUTURE_PLAYER_FORM, **kwargs)


def fit_final_v1(**kwargs: Any) -> dict[str, Any]:
    return fit_final_variant(variant=RatingVariant.CURRENT_ONLY, **kwargs)


def fit_final_v3(**kwargs: Any) -> dict[str, Any]:
    return fit_final_variant(variant=RatingVariant.SCALING_CURVE, **kwargs)


def fit_final_v4(**kwargs: Any) -> dict[str, Any]:
    return fit_final_variant(variant=RatingVariant.BOTH, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default=RatingVariant.FUTURE_PLAYER_FORM.value)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--current-root", required=True, type=Path)
    parser.add_argument("--scaling-root", "--scaling-feature-root", dest="scaling_root", type=Path)
    parser.add_argument(
        "--scaling-artifact",
        "--scaling-artifact-path",
        "--scaling-feature-artifact",
        dest="scaling_artifact_path",
        type=Path,
    )
    parser.add_argument(
        "--scaling-receipt",
        "--scaling-receipt-path",
        "--scaling-feature-receipt",
        dest="scaling_receipt_path",
        type=Path,
    )
    parser.add_argument(
        "--scaling-manifest",
        "--scaling-manifest-path",
        "--scaling-feature-manifest",
        dest="scaling_manifest_path",
        type=Path,
    )
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument(
        "--evaluation-sha256",
        "--evaluation-file-sha256",
        dest="evaluation_sha256",
        required=True,
    )
    parser.add_argument("--baseline-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--source-receipt-sha256", required=True)
    parser.add_argument("--current-receipt-sha256", required=True)
    parser.add_argument("--current-artifact-sha256", required=True)
    parser.add_argument("--scaling-receipt-sha256", "--scaling-feature-receipt-sha256", dest="scaling_receipt_sha256")
    parser.add_argument("--scaling-artifact-sha256", "--scaling-feature-artifact-sha256", dest="scaling_artifact_sha256")
    parser.add_argument("--scaling-manifest-sha256", "--scaling-feature-manifest-sha256", dest="scaling_manifest_sha256")
    parser.add_argument("--nested-selection", type=Path, required=True)
    parser.add_argument("--nested-selection-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run = fit_final_variant(
        variant=args.variant,
        source_root=args.source_root,
        source_receipt_path=args.source_receipt,
        current_root=args.current_root,
        scaling_root=args.scaling_root,
        scaling_artifact_path=args.scaling_artifact_path,
        scaling_receipt_path=args.scaling_receipt_path,
        scaling_manifest_path=args.scaling_manifest_path,
        evaluation_path=args.evaluation,
        output_dir=args.output_dir,
        baseline_cache_path=args.baseline_cache,
        expected_source_receipt_sha256=args.source_receipt_sha256,
        expected_evaluation_sha256=args.evaluation_sha256,
        expected_current_receipt_sha256=args.current_receipt_sha256,
        expected_current_artifact_sha256=args.current_artifact_sha256,
        expected_scaling_receipt_sha256=args.scaling_receipt_sha256,
        expected_scaling_artifact_sha256=args.scaling_artifact_sha256,
        expected_scaling_manifest_sha256=args.scaling_manifest_sha256,
        nested_selection_path=args.nested_selection,
        expected_nested_selection_sha256=args.nested_selection_sha256,
    )
    print(json.dumps(run, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
