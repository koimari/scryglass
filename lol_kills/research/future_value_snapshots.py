"""Research-only as-of player and team value snapshots.

This module consumes the future-value model contract.  It keeps the public
rating and Tier List artifacts unchanged.

The snapshot is valid only when a final model receipt binds the complete
model-eligible census and the source-bound current-rating feature ledger.
Fold models remain useful for research, but they cannot silently become a
final snapshot model.  In that case this module writes a blocked receipt with
the exact blockers and no invented values.
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
    MODEL_FIT_SCHEMA_VERSION,
    RANK_3,
    SCALING_CURVE_SIGNED_MAP_FEATURES,
    TEAM_CONTEXT_FEATURES,
    RatingVariant,
    FutureValueFoldModel,
    FutureValueSourceError,
    Rank3AtomModel,
    _canonical_json_bytes,
    _frame_game_ids,
    _role,
    rating_feature_values_sha256,
    _stable_identity,
    _team_history_features,
    _utc_text,
    _utc_timestamp,
    build_strict_prior_player_form,
    validate_future_value_source_receipt_payload,
)


SCHEMA_VERSION = "scryglass:future-value-snapshot:v1"
SNAPSHOT_RECEIPT_SCHEMA_VERSION = "scryglass:future-value-snapshot-receipt:v1"
SNAPSHOT_AUTHORITY = {
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

PLAYER_VALUE_FEATURE_PREFIXES = (
    "player_form_",
    "rank_3_player_atom_",
    "rank_3_champion_role_atom_",
)
QUALITY_FEATURES = frozenset(
    {
        "player_form_missing_rate",
        "rank_3_atom_missing_rate",
        "rank_3_champion_role_atom_missing_rate",
        "player_form_support_mean",
        "player_form_effective_support_mean",
    }
)
TEAM_FEATURES = frozenset({"team_prior_win_diff", "roster_continuity_diff"})
IGNORED_SNAPSHOT_FEATURES = frozenset(
    {
        *CURRENT_RATING_SIGNED_MAP_FEATURES,
        *SCALING_CURVE_SIGNED_MAP_FEATURES,
        *TEAM_CONTEXT_FEATURES,
        *TEAM_FEATURES,
    }
)

# These blockers describe promotion evidence.  They do not prevent a valid,
# source-bound research calculation.  Unknown blockers remain computation
# blockers and stop the snapshot.
PROMOTION_ONLY_BLOCKERS = frozenset(
    {
        "authoritative_series_id_missing_proxy_cluster_used",
        "current_player_team_rating_comparison_missing",
        "current_rating_player_team_identity_missing_for_rank_diffs",
        "final_calibration_receipt_missing",
        "final_fit_status_not_authorized",
        "nested_inner_feature_ledger_missing_fixed_c_used",
        "patch_transfer_sparse_validation_support",
        "patch_transfer_unseen_training_group",
        "phase_model_series_partition_non_comparable",
        "regional_transfer_sparse_validation_support",
        "regional_transfer_unseen_training_group",
        "roster_change_labels_missing",
        "support_uncertainty_proxy_not_calibrated",
        "team_context_not_in_final_model",
        "tournament_boundary_field_missing",
        "tournament_boundary_slice_missing",
    }
)
RESEARCH_FIT_STATUSES = frozenset(
    {"research_only", "research_only_blocked", "research_only_partial"}
)
RANK_UNIVERSE = "common_verified_finite_ids"
RANK_ELIGIBILITY_FILTER = "verified_nonempty_id_and_finite_value"
RANK_DIRECTION = "descending_value_rank_1_highest"
FULL_SNAPSHOT_RANK_STATUS = "incomparable"


class FutureValueSnapshotError(FutureValueSourceError):
    """The research snapshot cannot be built safely."""


@dataclass(frozen=True)
class FinalFitAuthorization:
    """The result of the final-fit gate."""

    status: str
    blockers: tuple[str, ...]
    model_receipt_sha256: str | None
    source_receipt_sha256: str

    @property
    def authorized(self) -> bool:
        return self.status == "authorized" and not self.blockers

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "authorized": self.authorized,
            "blockers": list(self.blockers),
            "model_receipt_sha256": self.model_receipt_sha256,
            "source_receipt_sha256": self.source_receipt_sha256,
        }


@dataclass(frozen=True)
class FutureValueSnapshotResult:
    """Research snapshot rows and their source-bound receipt."""

    status: str
    blockers: tuple[str, ...]
    player_rows: tuple[Mapping[str, Any], ...]
    team_rows: tuple[Mapping[str, Any], ...]
    player_rank_diffs: tuple[Mapping[str, Any], ...]
    team_rank_diffs: tuple[Mapping[str, Any], ...]
    receipt: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "blockers": list(self.blockers),
            "player_rows": [dict(row) for row in self.player_rows],
            "team_rows": [dict(row) for row in self.team_rows],
            "player_rank_diffs": [dict(row) for row in self.player_rank_diffs],
            "team_rank_diffs": [dict(row) for row in self.team_rank_diffs],
            "receipt": dict(self.receipt),
        }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _canonical_ids(values: Iterable[Any]) -> tuple[str, ...]:
    output = tuple(sorted({str(value) for value in values if str(value).strip()}))
    if not output:
        raise FutureValueSnapshotError("snapshot identity set is empty")
    return output


def _receipt_hash(receipt: Mapping[str, Any]) -> str:
    payload = dict(receipt)
    claimed = payload.pop("receipt_sha256", None)
    if not isinstance(claimed, str) or re.fullmatch(r"[0-9a-f]{64}", claimed, re.I) is None:
        raise FutureValueSnapshotError("snapshot receipt hash is invalid")
    if _sha256_bytes(_canonical_json_bytes(payload)) != claimed.lower():
        raise FutureValueSnapshotError("snapshot receipt hash does not match payload")
    return claimed.lower()


def _source_binding(source_receipt: Mapping[str, Any]) -> dict[str, Any]:
    try:
        validate_future_value_source_receipt_payload(source_receipt)
    except Exception as error:
        raise FutureValueSnapshotError(f"source receipt failed validation: {error}") from error
    return {
        "source_as_of": str(source_receipt["source_as_of"]),
        "source_game_count": int(source_receipt["source_game_count"]),
        "source_identity_sha256": str(source_receipt["source_identity_sha256"]),
        "model_eligible_game_count": int(source_receipt["model_eligible_game_count"]),
        "model_eligible_identity_sha256": str(
            source_receipt["model_eligible_identity_sha256"]
        ),
        "model_eligible_game_ids": list(source_receipt["model_eligible_game_ids"]),
        "source_receipt_sha256": str(source_receipt["receipt_sha256"]),
    }


def _model_receipt_from(model: Any, model_receipt: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if model_receipt is not None:
        return model_receipt
    if model is None or not hasattr(model, "receipt"):
        return None
    value = model.receipt()
    return value if isinstance(value, Mapping) else None


def _validate_model_object_binding(
    model: Any,
    model_receipt: Mapping[str, Any] | None,
) -> None:
    """Bind an explicitly supplied receipt to the supplied model object."""

    if model is None or model_receipt is None:
        return
    claimed_hash = _receipt_hash(model_receipt)
    loaded_receipt_hash = str(
        getattr(model, "_bound_final_fit_receipt_sha256", "") or ""
    ).lower()
    if loaded_receipt_hash and loaded_receipt_hash != claimed_hash:
        raise FutureValueSnapshotError(
            "explicit model receipt does not bind loaded model artifact"
        )
    parameter_receipt = getattr(model, "parameter_receipt", None)
    if callable(parameter_receipt):
        parameters = parameter_receipt()
        if not isinstance(parameters, Mapping):
            raise FutureValueSnapshotError("explicit model parameters are invalid")
        if str(parameters.get("parameter_sha256") or "").lower() != str(
            model_receipt.get("parameter_sha256") or ""
        ).lower():
            raise FutureValueSnapshotError("explicit model receipt does not bind model parameters")
    object_receipt = _model_receipt_from(model, None)
    if object_receipt is None:
        raise FutureValueSnapshotError("explicit model receipt does not bind model object")

    def _contains_model_fields(
        explicit: Mapping[str, Any],
        object_value: Mapping[str, Any],
        *,
        root: bool = False,
    ) -> bool:
        for key, value in object_value.items():
            if root and key == "receipt_sha256":
                if value and str(value).lower() != claimed_hash:
                    return False
                continue
            if key not in explicit:
                return False
            explicit_value = explicit[key]
            if isinstance(value, Mapping):
                if not isinstance(explicit_value, Mapping) or not _contains_model_fields(
                    explicit_value, value, root=False
                ):
                    return False
            elif explicit_value != value:
                return False
        return True

    if not _contains_model_fields(model_receipt, object_receipt, root=True):
        raise FutureValueSnapshotError(
            "explicit model receipt does not bind model metadata"
        )


def load_final_fit_model(
    model_artifact_path: Path,
    model_receipt_path: Path,
    *,
    source_receipt: Mapping[str, Any],
) -> tuple[FutureValueFoldModel, Mapping[str, Any]]:
    """Load a JSON final-fit model after checking its receipt bindings.

    The loader accepts only the model artifact emitted by the final-fit
    benchmark.  It uses no pickle or executable model state.  A sibling
    ``final-fit-run.json`` receives an additional byte and receipt check when
    present.
    """

    artifact_path = Path(model_artifact_path)
    receipt_path = Path(model_receipt_path)
    for path, label in ((artifact_path, "model artifact"), (receipt_path, "model receipt")):
        if not path.is_file() or path.is_symlink():
            raise FutureValueSnapshotError(f"{label} is missing or unsafe: {path}")
    try:
        receipt_value = json.loads(receipt_path.read_text(encoding="utf-8"))
        artifact_value = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FutureValueSnapshotError("final-fit model JSON cannot be read") from error
    if not isinstance(receipt_value, Mapping) or not isinstance(artifact_value, Mapping):
        raise FutureValueSnapshotError("final-fit model JSON must contain objects")
    receipt = dict(receipt_value)
    try:
        receipt_hash = _receipt_hash(receipt)
    except FutureValueSnapshotError as error:
        raise FutureValueSnapshotError("final-fit model receipt is invalid") from error
    if artifact_value.get("receipt_sha256") != receipt_hash:
        raise FutureValueSnapshotError("final-fit model artifact receipt binding changed")
    if artifact_value.get("status") != receipt.get("status"):
        raise FutureValueSnapshotError("final-fit model status binding changed")
    code_binding = receipt.get("code_binding")
    if not isinstance(code_binding, Mapping):
        raise FutureValueSnapshotError("final-fit code binding is missing")
    if str(code_binding.get("snapshot_producer_sha256") or "").lower() != _sha256_file(
        Path(__file__)
    ):
        raise FutureValueSnapshotError("final-fit snapshot producer binding changed")
    run_path = receipt_path.parent / "final-fit-run.json"
    if run_path.is_file() and not run_path.is_symlink():
        try:
            run_value = json.loads(run_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FutureValueSnapshotError("final-fit run receipt cannot be read") from error
        if not isinstance(run_value, Mapping):
            raise FutureValueSnapshotError("final-fit run receipt is invalid")
        if run_value.get("model_receipt_sha256") != receipt_hash:
            raise FutureValueSnapshotError("final-fit run receipt hash changed")
        if run_value.get("model_artifact_sha256") != _sha256_file(artifact_path):
            raise FutureValueSnapshotError("final-fit model artifact hash changed")
    parameters = artifact_value.get("parameters")
    if not isinstance(parameters, Mapping):
        raise FutureValueSnapshotError("final-fit model parameters are missing")
    parameter_payload = dict(parameters)
    claimed_parameter_hash = parameter_payload.pop("parameter_sha256", None)
    if not isinstance(claimed_parameter_hash, str) or _sha256_bytes(
        _canonical_json_bytes(parameter_payload)
    ) != claimed_parameter_hash:
        raise FutureValueSnapshotError("final-fit model parameter hash changed")
    if claimed_parameter_hash != receipt.get("parameter_sha256"):
        raise FutureValueSnapshotError("final-fit parameter receipt binding changed")
    feature_binding = receipt.get("feature_ledger_binding")
    if not isinstance(feature_binding, Mapping):
        raise FutureValueSnapshotError("final-fit current-rating feature binding is missing")
    artifact_record = feature_binding.get("artifact")
    producer_receipt_record = feature_binding.get("producer_receipt_file")
    feature_names_binding = tuple(str(value) for value in feature_binding.get("feature_names") or ())
    if (
        not isinstance(artifact_record, Mapping)
        or not isinstance(producer_receipt_record, Mapping)
        or not feature_names_binding
    ):
        raise FutureValueSnapshotError("final-fit current-rating artifact binding is invalid")
    producer_receipt_path = Path(str(producer_receipt_record.get("path") or ""))
    if (
        not producer_receipt_path.is_absolute()
        or ".." in producer_receipt_path.parts
        or producer_receipt_path.is_symlink()
        or not producer_receipt_path.is_file()
        or int(producer_receipt_record.get("bytes") or -1)
        != producer_receipt_path.stat().st_size
        or str(producer_receipt_record.get("sha256") or "").lower()
        != _sha256_file(producer_receipt_path)
    ):
        raise FutureValueSnapshotError(
            "final-fit current-rating receipt file binding changed"
        )
    try:
        producer_receipt_value = json.loads(
            producer_receipt_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FutureValueSnapshotError(
            "final-fit current-rating receipt cannot be loaded"
        ) from error
    if not isinstance(producer_receipt_value, Mapping):
        raise FutureValueSnapshotError("final-fit current-rating receipt is invalid")
    producer_payload = dict(producer_receipt_value)
    producer_claimed_hash = str(producer_payload.pop("receipt_sha256", "")).lower()
    if (
        producer_claimed_hash
        != str(feature_binding.get("producer_receipt_sha256") or "").lower()
        or _sha256_bytes(_canonical_json_bytes(producer_payload))
        != producer_claimed_hash
    ):
        raise FutureValueSnapshotError(
            "final-fit current-rating receipt payload changed"
        )
    if str(producer_receipt_value.get("feature_value_digest") or "").lower() != str(
        feature_binding.get("feature_value_digest") or ""
    ).lower():
        raise FutureValueSnapshotError(
            "final-fit current-rating receipt feature digest changed"
        )
    feature_artifact_path = Path(str(artifact_record.get("path") or ""))
    if (
        not feature_artifact_path.is_absolute()
        or ".." in feature_artifact_path.parts
        or feature_artifact_path.is_symlink()
        or not feature_artifact_path.is_file()
        or int(artifact_record.get("bytes") or -1) != feature_artifact_path.stat().st_size
        or str(artifact_record.get("sha256") or "").lower() != _sha256_file(feature_artifact_path)
    ):
        raise FutureValueSnapshotError("final-fit current-rating artifact binding changed")
    try:
        feature_frame = pd.read_parquet(feature_artifact_path)
    except Exception as error:
        raise FutureValueSnapshotError("final-fit current-rating artifact cannot be loaded") from error
    required_feature_columns = {"game_id", "date", "series_id", *feature_names_binding}
    if not required_feature_columns.issubset(feature_frame.columns):
        raise FutureValueSnapshotError("final-fit current-rating artifact schema changed")
    if rating_feature_values_sha256(feature_frame, feature_names_binding) != str(
        feature_binding.get("feature_value_digest") or ""
    ).lower():
        raise FutureValueSnapshotError("final-fit current-rating feature values changed")
    feature_names = tuple(str(value) for value in parameters.get("feature_names") or ())
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise FutureValueSnapshotError("final-fit model feature names are invalid")

    def _parameter_vector(name: str) -> np.ndarray:
        values = parameters.get(name)
        if not isinstance(values, Mapping):
            raise FutureValueSnapshotError(f"final-fit parameter vector is missing: {name}")
        try:
            vector = np.asarray([float(values[feature]) for feature in feature_names], dtype=float)
        except (KeyError, TypeError, ValueError) as error:
            raise FutureValueSnapshotError(f"final-fit parameter vector is invalid: {name}") from error
        if not np.isfinite(vector).all():
            raise FutureValueSnapshotError(f"final-fit parameter vector is non-finite: {name}")
        return vector

    rank_payload = parameters.get("rank_3")
    if not isinstance(rank_payload, Mapping):
        raise FutureValueSnapshotError("final-fit rank-3 parameters are missing")
    try:
        rank = int(rank_payload["rank"])
        metric_names = tuple(str(value) for value in rank_payload["metric_names"])
        center = np.asarray(rank_payload["center"], dtype=float)
        scale = np.asarray(rank_payload["scale"], dtype=float)
        components = np.asarray(rank_payload["components"], dtype=float)
        coordinates = {
            str(key): tuple(float(value) for value in values)
            for key, values in dict(rank_payload["champion_role_coordinates"]).items()
        }
        support = {
            str(key): int(value)
            for key, value in dict(rank_payload["champion_role_support"]).items()
        }
        atom_fit_ids = tuple(str(value) for value in rank_payload["fit_game_ids"])
        atom_fit_window_end = str(rank_payload["fit_window_end"])
    except (KeyError, TypeError, ValueError) as error:
        raise FutureValueSnapshotError("final-fit rank-3 parameters are invalid") from error
    if rank != RANK_3 or len(metric_names) != len(center) or len(center) != len(scale):
        raise FutureValueSnapshotError("final-fit rank-3 dimensions are invalid")
    if components.shape != (rank, len(metric_names)):
        raise FutureValueSnapshotError("final-fit rank-3 component shape is invalid")
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or not np.isfinite(components).all():
        raise FutureValueSnapshotError("final-fit rank-3 parameters are non-finite")
    atom_parameter_payload = dict(rank_payload)
    claimed_atom_hash = atom_parameter_payload.pop("parameter_sha256", None)
    if not isinstance(claimed_atom_hash, str) or _sha256_bytes(
        _canonical_json_bytes(atom_parameter_payload)
    ) != claimed_atom_hash:
        raise FutureValueSnapshotError("final-fit rank-3 parameter hash changed")
    atom_model = Rank3AtomModel(
        metric_names=metric_names,
        rank=rank,
        center=center,
        scale=scale,
        components=components,
        champion_role_coordinates=coordinates,
        champion_role_support=support,
        fit_game_ids=atom_fit_ids,
        fit_window_end=atom_fit_window_end,
    )
    try:
        variant = RatingVariant(str(parameters["variant"]))
        coefficients = np.asarray(
            [float(dict(parameters["coefficients"])[feature]) for feature in feature_names],
            dtype=float,
        )
        intercept = float(parameters.get("intercept", 0.0))
    except (KeyError, TypeError, ValueError) as error:
        raise FutureValueSnapshotError("final-fit model coefficients are invalid") from error
    if not np.isfinite(coefficients).all() or not math.isfinite(intercept):
        raise FutureValueSnapshotError("final-fit model coefficients are non-finite")
    model = FutureValueFoldModel(
        feature_names=feature_names,
        means=_parameter_vector("feature_means"),
        scales=_parameter_vector("feature_scales"),
        imputation_values=_parameter_vector("fold_local_side_imputation"),
        coefficients=coefficients,
        intercept=intercept,
        regularization_selection=dict(parameters.get("regularization_selection") or {}),
        optimizer_evidence=dict(parameters.get("optimizer_evidence") or {}),
        atom_model=atom_model,
        fit_game_ids=tuple(str(value) for value in receipt.get("fit_game_ids") or ()),
        fit_window_end=str(receipt.get("fit_window_end") or ""),
        train_rows=int(receipt.get("train_rows") or 0),
        withheld_rows=int(receipt.get("withheld_rows") or 0),
        source_receipt=dict(source_receipt),
        variant=variant,
        feature_ledger_binding=dict(receipt.get("feature_ledger_binding") or {}),
    )
    object.__setattr__(model, "_bound_final_fit_receipt_sha256", receipt_hash)
    object.__setattr__(
        model,
        "_bound_final_fit_artifact_sha256",
        _sha256_file(artifact_path),
    )
    return model, receipt


def authorize_final_fit(
    model_receipt: Mapping[str, Any] | None,
    source_receipt: Mapping[str, Any],
    *,
    require_complete_census: bool = True,
) -> FinalFitAuthorization:
    """Check whether a model receipt can produce an as-of snapshot.

    The gate is intentionally stricter than the fold-evaluation gate.  A
    fold receipt cannot be promoted to a final source snapshot by inference.
    """

    source = _source_binding(source_receipt)
    blockers: set[str] = set()
    if model_receipt is None:
        blockers.add("final_fit_receipt_missing")
        return FinalFitAuthorization(
            "blocked", tuple(sorted(blockers)), None, source["source_receipt_sha256"]
        )

    try:
        model_hash = _receipt_hash(model_receipt)
    except FutureValueSnapshotError:
        model_hash = None
        blockers.add("final_fit_receipt_hash_invalid")

    if model_receipt.get("schema_version") != MODEL_FIT_SCHEMA_VERSION:
        blockers.add("final_fit_receipt_schema_invalid")
    model_status = str(model_receipt.get("status") or "")
    if model_status != "final_fit_authorized":
        blockers.add("final_fit_status_not_authorized")
        if model_status not in RESEARCH_FIT_STATUSES:
            blockers.add("final_fit_status_invalid")
    declared_blockers = model_receipt.get("blockers")
    if declared_blockers is not None:
        if not isinstance(declared_blockers, (list, tuple)):
            blockers.add("final_fit_blocker_list_invalid")
        else:
            blockers.update(str(value) for value in declared_blockers)
    variant = str(model_receipt.get("variant") or "")
    if variant not in {RatingVariant.FUTURE_PLAYER_FORM.value, RatingVariant.BOTH.value}:
        blockers.add("final_fit_variant_not_future_player_form")

    source_binding = model_receipt.get("source_binding")
    if not isinstance(source_binding, Mapping):
        blockers.add("final_fit_source_binding_missing")
    else:
        if source_binding.get("source_receipt_sha256") != source["source_receipt_sha256"]:
            blockers.add("final_fit_source_receipt_mismatch")
        if source_binding.get("source_identity_sha256") != source["source_identity_sha256"]:
            blockers.add("final_fit_source_identity_mismatch")
        if source_binding.get("source_as_of") != source["source_as_of"]:
            blockers.add("final_fit_source_as_of_mismatch")
        if source_binding.get("model_eligible_identity_sha256") != source[
            "model_eligible_identity_sha256"
        ]:
            blockers.add("final_fit_eligible_identity_mismatch")

    eligible_ids = set(str(value) for value in source["model_eligible_game_ids"])
    fit_ids = tuple(str(value) for value in model_receipt.get("fit_game_ids") or ())
    if not fit_ids:
        blockers.add("final_fit_game_ids_missing")
    elif not set(fit_ids).issubset(eligible_ids):
        blockers.add("final_fit_contains_game_outside_eligible_census")
    elif require_complete_census and set(fit_ids) != eligible_ids:
        blockers.add("final_fit_not_bound_to_complete_model_eligible_census")

    fit_end_value = model_receipt.get("fit_window_end")
    try:
        fit_end = _utc_timestamp(fit_end_value, "fit_window_end")
        source_end = _utc_timestamp(source["source_as_of"], "source_as_of")
        if fit_end > source_end:
            blockers.add("final_fit_window_after_source_as_of")
    except FutureValueSourceError:
        blockers.add("final_fit_window_end_invalid")

    feature_binding = model_receipt.get("feature_ledger_binding")
    if not isinstance(feature_binding, Mapping):
        blockers.add("current_rating_feature_ledger_binding_missing")
    else:
        if feature_binding.get("source_receipt_sha256") != source[
            "source_receipt_sha256"
        ]:
            blockers.add("current_rating_feature_source_receipt_mismatch")
        if feature_binding.get("source_identity_sha256") != source[
            "source_identity_sha256"
        ]:
            blockers.add("current_rating_feature_source_identity_mismatch")
        if not feature_binding.get("producer_receipt_sha256"):
            blockers.add("current_rating_feature_producer_receipt_missing")
        producer_names = feature_binding.get("producer_names")
        if producer_names is not None and "current_sequential_rating" not in set(
            str(value) for value in producer_names
        ):
            blockers.add("current_rating_feature_producer_missing")

    regularization = model_receipt.get("regularization_selection")
    if isinstance(regularization, Mapping) and regularization.get("blockers"):
        blockers.update(str(value) for value in regularization["blockers"])
    elif not isinstance(regularization, Mapping):
        blockers.add("final_fit_regularization_evidence_missing")

    optimizer = model_receipt.get("optimizer_evidence")
    if not isinstance(optimizer, Mapping) or optimizer.get("success") is not True:
        blockers.add("final_fit_optimizer_not_verified")
    if not isinstance(optimizer, Mapping) or optimizer.get("finite_coefficients") is not True:
        blockers.add("final_fit_coefficients_not_verified")

    rank_three = model_receipt.get("rank_3")
    if not isinstance(rank_three, Mapping) or not rank_three.get("parameter_sha256"):
        blockers.add("final_fit_rank_3_parameters_missing")

    return FinalFitAuthorization(
        "authorized" if not blockers else "blocked",
        tuple(sorted(blockers)),
        model_hash,
        source["source_receipt_sha256"],
    )


def _computation_blockers(
    authorization: FinalFitAuthorization,
) -> tuple[str, ...]:
    """Return blockers that prevent a source-bound research calculation."""

    return tuple(
        sorted(
            blocker
            for blocker in authorization.blockers
            if blocker not in PROMOTION_ONLY_BLOCKERS
        )
    )


def _normalise_source_frames(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    source_receipt: Mapping[str, Any],
    as_of: Any | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    source = _source_binding(source_receipt)
    cutoff = _utc_timestamp(as_of or source["source_as_of"], "snapshot_as_of")
    eligible = set(str(value) for value in source["model_eligible_game_ids"])

    map_frame = maps.copy()
    if "date" not in map_frame.columns or "y_blue_win" not in map_frame.columns:
        raise FutureValueSnapshotError("snapshot maps require date and y_blue_win")
    map_frame["game_id"] = _frame_game_ids(map_frame, "maps").astype(str)
    map_frame["date"] = pd.to_datetime(map_frame.get("date"), utc=True, errors="coerce")
    map_frame["target"] = pd.to_numeric(map_frame.get("y_blue_win"), errors="coerce")
    if map_frame["game_id"].duplicated().any() or map_frame["date"].isna().any():
        raise FutureValueSnapshotError("snapshot maps have invalid identity or date")
    if map_frame["date"].gt(cutoff).any():
        raise FutureValueSnapshotError("snapshot maps contain rows after as_of")
    map_frame = map_frame[map_frame["game_id"].isin(eligible)].copy()
    if set(map_frame["game_id"]) != eligible:
        raise FutureValueSnapshotError("snapshot maps do not match the eligible census")
    if not map_frame["target"].isin({0, 1}).all():
        raise FutureValueSnapshotError("snapshot maps contain an invalid result")

    player_frame = players.copy()
    player_frame["game_id"] = _frame_game_ids(player_frame, "players").astype(str)
    team_frame = teams.copy()
    team_frame["game_id"] = _frame_game_ids(team_frame, "teams").astype(str)
    for frame, label in ((player_frame, "players"), (team_frame, "teams")):
        frame.drop(frame.index[~frame["game_id"].isin(eligible)], inplace=True)
        if "date" in frame:
            frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
            if frame["date"].isna().any() or frame["date"].gt(cutoff).any():
                raise FutureValueSnapshotError(f"snapshot {label} has invalid or future dates")
            map_dates = map_frame.set_index("game_id")["date"]
            expected_dates = frame["game_id"].map(map_dates)
            if expected_dates.isna().any() or ~frame["date"].eq(expected_dates).all():
                raise FutureValueSnapshotError(
                    f"snapshot {label} duplicate or dates do not match source map dates"
                )

    player_required = {"playerid", "teamid", "playername", "side", "position", "champion"}
    if not player_required.issubset(player_frame.columns):
        raise FutureValueSnapshotError(
            "snapshot players are missing: " + ", ".join(sorted(player_required - set(player_frame.columns)))
        )
    if not player_frame["playerid"].map(lambda value: _stable_identity(value, "oe:player:")).all():
        raise FutureValueSnapshotError("snapshot players have unstable player identity")
    if not player_frame["teamid"].map(lambda value: _stable_identity(value, "oe:team:")).all():
        raise FutureValueSnapshotError("snapshot players have unstable team identity")
    player_frame["player_id"] = player_frame["playerid"].astype(str)
    player_frame["team_id"] = player_frame["teamid"].astype(str)
    player_frame["side"] = player_frame["side"].map(lambda value: str(value).strip().casefold())
    player_frame["role"] = player_frame["position"].map(_role)
    if player_frame[["side", "role"]].isna().any().any() or not player_frame["side"].isin({"blue", "red"}).all():
        raise FutureValueSnapshotError("snapshot players have an unknown side or role")
    counts = player_frame.groupby("game_id", sort=False).size()
    if not counts.eq(10).all() or set(counts.index) != eligible:
        raise FutureValueSnapshotError("snapshot players require ten rows per eligible map")
    slots = player_frame.groupby(["game_id", "side"], sort=False)["role"].agg(
        lambda values: tuple(sorted(values))
    )
    expected_roles = tuple(sorted(("top", "jungle", "mid", "bot", "support")))
    if not slots.map(lambda value: value == expected_roles).all():
        raise FutureValueSnapshotError("snapshot players require exact five unique roles per side")
    if player_frame.duplicated(["game_id", "player_id"]).any():
        raise FutureValueSnapshotError("snapshot players contain duplicate player identities")

    if "side" not in team_frame.columns or "teamid" not in team_frame.columns:
        raise FutureValueSnapshotError("snapshot teams require side and teamid")
    team_frame["side"] = team_frame["side"].map(lambda value: str(value).strip().casefold())
    if not team_frame["side"].isin({"blue", "red"}).all():
        raise FutureValueSnapshotError("snapshot teams have an unknown side")
    team_counts = team_frame.groupby("game_id", sort=False).size()
    if not team_counts.eq(2).all() or set(team_counts.index) != eligible:
        raise FutureValueSnapshotError("snapshot teams require two rows per eligible map")
    if not team_frame["teamid"].map(
        lambda value: _stable_identity(value, "oe:team:")
    ).all():
        raise FutureValueSnapshotError("snapshot team rows have unstable team identity")
    if team_frame.duplicated(["game_id", "side"]).any():
        raise FutureValueSnapshotError("snapshot teams contain duplicate sides")
    player_team_by_side = (
        player_frame.groupby(["game_id", "side"], sort=False)["team_id"]
        .agg(lambda values: tuple(sorted(set(values))))
    )
    team_team_by_side = team_frame.set_index(["game_id", "side"])["teamid"].astype(str)
    for key, team_ids in player_team_by_side.items():
        if len(team_ids) != 1 or team_team_by_side.get(key) != team_ids[0]:
            raise FutureValueSnapshotError("snapshot player and team identities do not match")

    return map_frame, player_frame, team_frame, cutoff


def _latest_player_form(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    *,
    baseline_cache: Any | None = None,
) -> pd.DataFrame:
    """Build strict-prior form and select one unambiguous row per player."""

    strict = build_strict_prior_player_form(
        maps,
        players,
        baseline_cache=baseline_cache,
    )
    identity = players[
        ["game_id", "player_id", "team_id", "playername", "champion"]
    ].copy()
    identity["role"] = players["role"]
    identity["side"] = players["side"]
    identity["date"] = players["date"]
    identity = identity.drop_duplicates(
        ["game_id", "player_id", "role", "side"], keep=False
    )
    if identity.empty:
        raise FutureValueSnapshotError("snapshot player identity rows are ambiguous")
    strict["game_id"] = strict["game_id"].astype(str)
    strict["player_id"] = strict["player_id"].astype(str)
    strict["side"] = strict["side"].astype(str).str.casefold()
    strict["role"] = strict["role"].map(_role)
    strict["date"] = pd.to_datetime(strict["date"], utc=True, errors="coerce")
    joined = strict.merge(
        identity,
        on=["game_id", "player_id", "role", "side"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_source"),
    )
    if joined[["team_id", "champion", "playername"]].isna().any().any():
        raise FutureValueSnapshotError("strict-prior form lost player identity")
    if joined["date_source"].isna().any():
        raise FutureValueSnapshotError("strict-prior form date is missing")
    joined["date"] = pd.to_datetime(joined["date_source"], utc=True, errors="coerce")
    return joined.drop(columns=["date_source"])


def _latest_rows(form: pd.DataFrame, *, key: str, label: str) -> pd.DataFrame:
    ordered = form.sort_values([key, "date", "game_id"], kind="stable")
    last_dates = ordered.groupby(key, sort=False)["date"].transform("max")
    latest = ordered[ordered["date"].eq(last_dates)].copy()
    if latest.duplicated(key).any():
        duplicates = sorted(set(latest.loc[latest.duplicated(key, keep=False), key].astype(str)))
        raise FutureValueSnapshotError(
            f"{label} has ambiguous latest timestamp rows: {', '.join(duplicates[:5])}"
        )
    return latest


def _latest_team_roster(form: pd.DataFrame) -> pd.DataFrame:
    """Select one exact five-player roster for each current team."""

    ordered = form.sort_values(["team_id", "date", "game_id", "role", "player_id"], kind="stable")
    last_dates = ordered.groupby("team_id", sort=False)["date"].transform("max")
    latest = ordered[ordered["date"].eq(last_dates)].copy()
    if latest.empty:
        raise FutureValueSnapshotError("current team roster is empty")
    for team_id, group in latest.groupby("team_id", sort=False):
        if group[["game_id", "side"]].drop_duplicates().shape[0] != 1:
            raise FutureValueSnapshotError(
                f"team has ambiguous latest roster context: {team_id}"
            )
        if len(group) != 5 or group["player_id"].nunique() != 5:
            raise FutureValueSnapshotError(
                f"team requires exactly five current-roster players: {team_id}"
            )
        if group["role"].nunique() != 5:
            raise FutureValueSnapshotError(
                f"team requires five current-roster roles: {team_id}"
            )
    return latest


def _feature_column(feature: str) -> str | None:
    if feature in QUALITY_FEATURES:
        return feature
    if feature.startswith("player_form_"):
        return "prior_form_" + feature.removeprefix("player_form_")
    if feature.startswith("rank_3_"):
        return feature
    return None


def _player_contributions(
    model: Any,
    form: pd.DataFrame,
) -> pd.DataFrame:
    if not hasattr(model, "atom_model"):
        raise FutureValueSnapshotError("final model has no rank-3 atom model")
    atoms = model.atom_model.transform(form)
    work = pd.concat([form.reset_index(drop=True), atoms.reset_index(drop=True)], axis=1)
    atom_player_available = atoms["rank_3_player_atom_available"].astype(bool)
    atom_champion_available = atoms["rank_3_champion_role_atom_available"].astype(bool)
    # ``form`` can retain the source frame's row labels after the latest-row
    # selection.  ``work`` has a fresh range index, so assign atom flags by
    # position.  Label-aligned assignment would turn almost every valid atom
    # into NaN and mark the player as missing.
    work["rank_3_atom_missing_rate"] = (
        ~atom_player_available.to_numpy()
    ).astype(float)
    work["rank_3_champion_role_atom_missing_rate"] = (
        ~atom_champion_available.to_numpy()
    ).astype(float)
    support_columns = [f"prior_form_{metric}_support" for metric in FORM_METRICS]
    effective_columns = [f"prior_form_{metric}_effective_support" for metric in FORM_METRICS]
    if not set(support_columns).issubset(work.columns):
        raise FutureValueSnapshotError("final model form support columns are missing")
    support = work[support_columns].apply(pd.to_numeric, errors="coerce")
    effective_source = effective_columns if set(effective_columns).issubset(work.columns) else support_columns
    effective = work[effective_source].apply(pd.to_numeric, errors="coerce")
    work["player_form_missing_rate"] = work[
        [f"prior_form_{metric}" for metric in FORM_METRICS]
    ].apply(pd.to_numeric, errors="coerce").isna().mean(axis=1)
    work["player_form_support_mean"] = support.mean(axis=1, skipna=True)
    work["player_form_effective_support_mean"] = effective.mean(axis=1, skipna=True)
    feature_names = tuple(str(value) for value in model.feature_names)
    scales = np.asarray(model.scales, dtype=float)
    coefficients = np.asarray(model.coefficients, dtype=float)
    imputation = np.asarray(model.imputation_values, dtype=float)
    if len(feature_names) != len(scales) or len(scales) != len(coefficients) or len(scales) != len(imputation):
        raise FutureValueSnapshotError("final model parameter dimensions are invalid")
    if not np.isfinite(scales).all() or not np.isfinite(coefficients).all() or not np.isfinite(imputation).all():
        raise FutureValueSnapshotError("final model parameters are non-finite")
    output = work[["game_id", "player_id", "team_id", "playername", "champion", "role", "side", "date"]].copy()
    output["role_normalized_form_logit"] = 0.0
    output["rank_3_player_atom_logit"] = 0.0
    output["champion_role_atom_logit"] = 0.0
    output["data_quality_logit"] = 0.0
    for atom_index in range(1, RANK_3 + 1):
        output[f"rank_3_player_atom_{atom_index}_logit"] = 0.0
        output[f"rank_3_champion_role_atom_{atom_index}_logit"] = 0.0
    output["model_feature_missing"] = False
    for index, feature in enumerate(feature_names):
        column = _feature_column(feature)
        if column is None:
            if feature not in IGNORED_SNAPSHOT_FEATURES:
                raise FutureValueSnapshotError(f"final model feature is unsupported: {feature}")
            continue
        if column not in work.columns:
            raise FutureValueSnapshotError(f"final model feature is missing from form: {column}")
        values = pd.to_numeric(work[column], errors="coerce").to_numpy(dtype=float)
        missing = ~np.isfinite(values)
        values = np.where(missing, imputation[index], values)
        contribution = values / scales[index] * coefficients[index]
        if not np.isfinite(contribution).all():
            raise FutureValueSnapshotError(f"final model contribution is non-finite: {feature}")
        output["model_feature_missing"] |= missing
        if feature.startswith("player_form_"):
            output["role_normalized_form_logit"] += contribution
        elif feature.startswith("rank_3_player_atom_"):
            output["rank_3_player_atom_logit"] += contribution
            atom_index = feature.removeprefix("rank_3_player_atom_")
            if atom_index in {"1", "2", "3"}:
                output[f"rank_3_player_atom_{atom_index}_logit"] += contribution
        elif feature.startswith("rank_3_champion_role_atom_"):
            output["champion_role_atom_logit"] += contribution
            atom_index = feature.removeprefix("rank_3_champion_role_atom_")
            if atom_index in {"1", "2", "3"}:
                output[f"rank_3_champion_role_atom_{atom_index}_logit"] += contribution
        elif feature in QUALITY_FEATURES:
            output["data_quality_logit"] += contribution
    output["role_normalized_player_value_logit"] = (
        output["role_normalized_form_logit"]
        + output["rank_3_player_atom_logit"]
        + output["data_quality_logit"]
    )
    output["future_player_value_with_champion_logit"] = (
        output["role_normalized_player_value_logit"]
        + output["champion_role_atom_logit"]
    )
    output["minimum_metric_support"] = support.min(axis=1, skipna=True).fillna(0.0)
    output["minimum_effective_support"] = effective.min(axis=1, skipna=True).fillna(0.0)
    output["form_missing_rate"] = work["player_form_missing_rate"]
    output["rank_3_champion_role_atom_support"] = pd.to_numeric(
        work.get("rank_3_champion_role_support", 0), errors="coerce"
    ).fillna(0).astype(int)
    output["uncertainty_proxy"] = 1.0 / np.sqrt(1.0 + output["minimum_effective_support"])
    output["support_status"] = np.select(
        [
            output["model_feature_missing"],
            output["minimum_effective_support"].lt(5.0)
            | output["rank_3_champion_role_atom_support"].lt(1),
        ],
        ["missing_features", "sparse"],
        default="adequate",
    )
    output["champion_dependent_status"] = np.where(
        work["champion"].notna() & work["role"].notna(), "available", "missing"
    )
    return output


def _json_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (np.generic,)):
            value = value.item()
        if isinstance(value, pd.Timestamp):
            value = _utc_text(value)
        if isinstance(value, float) and not math.isfinite(value):
            value = None
        output[str(key)] = value
    return output


def _rank_diffs(
    future_rows: Sequence[Mapping[str, Any]],
    current: pd.DataFrame | None,
    *,
    identity: str,
    future_value: str,
    current_value_candidates: Sequence[str],
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any], dict[str, dict[str, Any]]]:
    blockers: list[str] = []
    futures = pd.DataFrame(list(future_rows))
    total_future = int(len(futures))
    assignments: dict[str, dict[str, Any]] = {}

    def _identity_digest(values: Iterable[Any]) -> str:
        ids = sorted({str(value) for value in values if str(value).strip()})
        return _sha256_bytes(
            _canonical_json_bytes({"identity": identity, "ids": ids})
        )

    def _full_rank_contract(
        *,
        current_size: int,
        future_size: int,
        current_field: str | None,
        future_field: str | None,
    ) -> dict[str, Any]:
        return {
            "status": FULL_SNAPSHOT_RANK_STATUS,
            "reason": "full snapshot ranks use separate universes",
            "current_universe_size": int(current_size),
            "future_universe_size": int(future_size),
            "current_value_field": current_field,
            "future_value_field": future_field,
            "rank_direction": RANK_DIRECTION,
        }

    coverage: dict[str, Any] = {
        "future_rows": total_future,
        "current_rows": 0,
        "matched_rows": 0,
        "unmatched_rows": total_future,
        "join_rate": 0.0 if total_future else None,
        "status": "unavailable",
        "rank_universe": RANK_UNIVERSE,
        "eligibility_filter": RANK_ELIGIBILITY_FILTER,
        "common_universe_size": 0,
        "common_identity_sha256": _identity_digest(()),
        "identity_sha256": _identity_digest(()),
        "current_value_field": None,
        "future_value_field": future_value,
        "rank_direction": RANK_DIRECTION,
        "paired_row_digest_sha256": _sha256_bytes(_canonical_json_bytes([])),
        "paired_row_digest": _sha256_bytes(_canonical_json_bytes([])),
        "finite_current_rows": 0,
        "finite_future_rows": 0,
        "full_snapshot_ranks": _full_rank_contract(
            current_size=0,
            future_size=0,
            current_field=None,
            future_field=future_value,
        ),
    }

    if futures.empty or identity not in futures.columns or future_value not in futures.columns:
        return [], [f"future_{identity}_snapshot_missing"], coverage, assignments

    futures[identity] = futures[identity].astype("string")
    futures[future_value] = pd.to_numeric(futures[future_value], errors="coerce")
    future_with_identity = futures[
        futures[identity].notna() & futures[identity].str.strip().ne("")
    ].copy()
    if future_with_identity[identity].duplicated().any():
        return [], [f"future_{identity}_snapshot_identity_or_value_ambiguous"], coverage, assignments
    future_finite = future_with_identity[
        future_with_identity[future_value].map(lambda value: _finite(value) is not None)
    ].copy()
    coverage["finite_future_rows"] = int(len(future_finite))

    if current is None:
        coverage["full_snapshot_ranks"] = _full_rank_contract(
            current_size=0,
            future_size=len(future_finite),
            current_field=None,
            future_field=future_value,
        )
        return [], [f"current_{identity}_rating_snapshot_missing"], coverage, assignments
    if identity not in current.columns:
        coverage["full_snapshot_ranks"] = _full_rank_contract(
            current_size=0,
            future_size=len(future_finite),
            current_field=None,
            future_field=future_value,
        )
        return [], [f"current_{identity}_rating_identity_missing"], coverage, assignments
    value_column = next((name for name in current_value_candidates if name in current.columns), None)
    if value_column is None:
        coverage["full_snapshot_ranks"] = _full_rank_contract(
            current_size=0,
            future_size=len(future_finite),
            current_field=None,
            future_field=future_value,
        )
        return [], [f"current_{identity}_rating_value_missing"], coverage, assignments

    frame = current[[identity, value_column]].copy()
    frame[identity] = frame[identity].astype("string")
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    current_with_identity = frame[
        frame[identity].notna() & frame[identity].str.strip().ne("")
    ].copy()
    if current_with_identity[identity].duplicated().any():
        return [], [f"current_{identity}_rating_identity_or_value_ambiguous"], coverage, assignments
    frame = current_with_identity[
        current_with_identity[value_column].map(lambda value: _finite(value) is not None)
    ].copy()
    coverage["current_rows"] = int(len(frame))
    coverage["finite_current_rows"] = int(len(frame))
    coverage["current_value_field"] = value_column
    coverage["full_snapshot_ranks"] = _full_rank_contract(
        current_size=len(frame),
        future_size=len(future_finite),
        current_field=value_column,
        future_field=future_value,
    )
    if future_finite.empty:
        coverage["status"] = "no_finite_future_values"
        return [], [], coverage, assignments

    current_ids = set(frame[identity].astype(str))
    future_ids = set(future_finite[identity].astype(str))
    common_ids = tuple(sorted(current_ids & future_ids))
    common_id_set = set(common_ids)
    coverage["common_universe_size"] = len(common_ids)
    coverage["common_identity_sha256"] = _identity_digest(common_ids)
    coverage["identity_sha256"] = coverage["common_identity_sha256"]
    if not common_ids:
        coverage["status"] = "partial"
        coverage["unmatched_rows"] = total_future
        coverage["join_rate"] = 0.0 if total_future else None
        return [], [], coverage, assignments

    common_current = frame[frame[identity].astype(str).isin(common_id_set)].copy()
    common_future = future_finite[future_finite[identity].astype(str).isin(common_id_set)].copy()
    common_current["__rank"] = common_current[value_column].rank(
        method="min", ascending=False
    )
    common_future["__rank"] = common_future[future_value].rank(
        method="min", ascending=False
    )
    current_rank = dict(zip(common_current[identity].astype(str), common_current["__rank"]))
    current_value = dict(zip(common_current[identity].astype(str), common_current[value_column]))
    future_rank = dict(zip(common_future[identity].astype(str), common_future["__rank"]))
    future_value_by_id = dict(zip(common_future[identity].astype(str), common_future[future_value]))

    paired_rows: list[dict[str, Any]] = []
    for key in common_ids:
        current_rank_value = int(current_rank[key])
        future_rank_value = int(future_rank[key])
        rank_delta = int(current_rank_value - future_rank_value)
        paired_rows.append(
            {
                identity: key,
                "current_rank": current_rank_value,
                "future_rank": future_rank_value,
                "rank_delta": rank_delta,
                "current_value": float(current_value[key]),
                "future_value": float(future_value_by_id[key]),
            }
        )
    paired_digest = _sha256_bytes(_canonical_json_bytes(paired_rows))
    coverage["paired_row_digest_sha256"] = paired_digest
    coverage["paired_row_digest"] = paired_digest

    output: list[dict[str, Any]] = []
    for row in common_future.to_dict("records"):
        key = str(row[identity])
        current_rank_value = int(current_rank[key])
        future_rank_value = int(future_rank[key])
        rank_delta = int(current_rank_value - future_rank_value)
        assignment = {
            "future_rank": future_rank_value,
            "current_rank": current_rank_value,
            "rank_delta": rank_delta,
            "rank_universe": RANK_UNIVERSE,
            "rank_comparability": "comparable",
            "full_snapshot_rank_status": FULL_SNAPSHOT_RANK_STATUS,
        }
        assignments[key] = assignment
        output.append(
            {
                identity: key,
                "current_rank": current_rank_value,
                "future_rank": future_rank_value,
                "rank_delta": rank_delta,
                "current_value": float(current_value[key]),
                "future_value": float(future_value_by_id[key]),
                "rank_universe": RANK_UNIVERSE,
                "rank_comparability": "comparable",
                "full_snapshot_rank_status": FULL_SNAPSHOT_RANK_STATUS,
            }
        )
    matched = len(output)
    coverage.update(
        {
            "matched_rows": matched,
            "unmatched_rows": max(0, total_future - matched),
            "join_rate": float(matched / total_future) if total_future else None,
            "status": "complete" if matched == total_future else "partial",
        }
    )
    return output, blockers, coverage, assignments


def _rank_extremes(rows: Sequence[Mapping[str, Any]], *, limit: int = 10) -> dict[str, list[dict[str, Any]]]:
    """Return only matched rank changes for research inspection."""

    frame = pd.DataFrame(list(rows))
    if frame.empty or "rank_delta" not in frame.columns:
        return {"largest_positive": [], "largest_negative": []}
    frame["rank_delta"] = pd.to_numeric(frame["rank_delta"], errors="coerce")
    frame = frame[frame["rank_delta"].notna()].copy()
    if frame.empty:
        return {"largest_positive": [], "largest_negative": []}
    columns = [
        column
        for column in ("player_id", "team_id", "current_rank", "future_rank", "rank_delta")
        if column in frame.columns
    ]
    positive = frame.sort_values(["rank_delta"], ascending=False, kind="stable").head(limit)
    negative = frame.sort_values(["rank_delta"], ascending=True, kind="stable").head(limit)
    return {
        "largest_positive": positive[columns].to_dict("records"),
        "largest_negative": negative[columns].to_dict("records"),
    }


def _blocked_result(
    source: Mapping[str, Any],
    authorization: FinalFitAuthorization,
    *,
    extra_blockers: Iterable[str] = (),
) -> FutureValueSnapshotResult:
    blockers = tuple(sorted(set(authorization.blockers) | set(extra_blockers)))
    payload: dict[str, Any] = {
        "schema_version": SNAPSHOT_RECEIPT_SCHEMA_VERSION,
        "status": "blocked",
        "authority": dict(SNAPSHOT_AUTHORITY),
        "source": _source_binding(source),
        "fit": authorization.as_dict(),
        "as_of": source["source_as_of"],
        "player_row_count": 0,
        "team_row_count": 0,
        "player_rank_diff_count": 0,
        "team_rank_diff_count": 0,
        "blockers": list(blockers),
        "tierlists": {"recalculated": False, "status": "unchanged"},
    }
    payload["receipt_sha256"] = _sha256_bytes(_canonical_json_bytes(payload))
    return FutureValueSnapshotResult(
        "blocked", blockers, (), (), (), (), payload
    )


def build_future_value_snapshots(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    *,
    source_receipt: Mapping[str, Any],
    model: FutureValueFoldModel | Any | None = None,
    model_receipt: Mapping[str, Any] | None = None,
    current_player_ratings: pd.DataFrame | None = None,
    current_team_ratings: pd.DataFrame | None = None,
    as_of: Any | None = None,
    baseline_cache: Any | None = None,
) -> FutureValueSnapshotResult:
    """Build one source-bound, research-only player/team snapshot.

    A missing or unapproved final fit produces a blocked result.  The source
    and lineup checks still run before scoring, so a malformed source never
    produces a plausible empty snapshot.
    """

    source = _source_binding(source_receipt)
    bound_model_receipt = _model_receipt_from(model, model_receipt)
    _validate_model_object_binding(model, model_receipt)
    auth = authorize_final_fit(bound_model_receipt, source_receipt)
    # Validate the source before the fit gate result is returned.  A blocked
    # model must not hide a malformed or future-dated accepted source.
    map_frame, player_frame, team_frame, cutoff = _normalise_source_frames(
        maps, players, teams, source_receipt, as_of
    )
    computation_blockers = _computation_blockers(auth)
    if computation_blockers:
        return _blocked_result(
            source_receipt,
            auth,
            extra_blockers=computation_blockers,
        )
    if model is None:
        return _blocked_result(source_receipt, auth, extra_blockers=("final_fit_model_object_missing",))

    form = _latest_player_form(map_frame, player_frame, baseline_cache=baseline_cache)
    form = form[form["date"].le(cutoff)].copy()
    latest = _latest_rows(form, key="player_id", label="player")
    if len(latest) != form["player_id"].nunique():
        raise FutureValueSnapshotError("latest player snapshot is incomplete")
    latest_roster = _latest_team_roster(form)
    contributions = _player_contributions(model, latest)
    roster_contributions = _player_contributions(model, latest_roster)
    player_rows: list[dict[str, Any]] = []
    for row in contributions.to_dict("records"):
        row_status = str(row["support_status"])
        row_has_missing = bool(row["model_feature_missing"])
        row_blockers: list[str] = []
        if row_status == "missing_features":
            row_blockers.append("missing_model_feature_value")
        elif row_status == "sparse":
            row_blockers.append("sparse_player_support")
        if "support_uncertainty_proxy_not_calibrated" in auth.blockers:
            row_blockers.append("support_uncertainty_proxy_not_calibrated")
        player_rows.append(
            _json_row(
                {
                    "player_id": row["player_id"],
                    "player": row["playername"],
                    "team_id": row["team_id"],
                    "role": row["role"],
                    "champion": row["champion"],
                    "last_game_id": row.get("game_id"),
                    "last_game_date": row["date"],
                    "role_normalized_form_logit": None
                    if row_has_missing
                    else row["role_normalized_form_logit"],
                    "rank_3_player_atom_logit": None
                    if row_has_missing
                    else row["rank_3_player_atom_logit"],
                    "champion_role_atom_logit": None
                    if row_has_missing
                    else row["champion_role_atom_logit"],
                    **{
                        f"rank_3_player_atom_{index}_logit": None
                        if row_has_missing
                        else row[f"rank_3_player_atom_{index}_logit"]
                        for index in range(1, RANK_3 + 1)
                    },
                    **{
                        f"rank_3_champion_role_atom_{index}_logit": None
                        if row_has_missing
                        else row[f"rank_3_champion_role_atom_{index}_logit"]
                        for index in range(1, RANK_3 + 1)
                    },
                    "future_player_value_logit": None
                    if row_has_missing
                    else row["role_normalized_player_value_logit"],
                    "future_player_value_with_champion_logit": None
                    if row_has_missing
                    else row["future_player_value_with_champion_logit"],
                    "minimum_metric_support": row["minimum_metric_support"],
                    "minimum_effective_support": row["minimum_effective_support"],
                    "rank_3_champion_role_atom_support": row[
                        "rank_3_champion_role_atom_support"
                    ],
                    "uncertainty_proxy": row["uncertainty_proxy"],
                    "model_feature_missing": bool(row["model_feature_missing"]),
                    "champion_dependent_status": row["champion_dependent_status"],
                    "support_status": row_status,
                    "status": "research_only" if row_status == "adequate" else f"research_only_{row_status}",
                    "blockers": sorted(set(row_blockers)),
                    "current_rank": None,
                    "future_rank": None,
                    "rank_delta": None,
                    "rank_join_status": "pending",
                }
            )
        )

    contribution_frame = roster_contributions.copy()
    contribution_frame["game_id"] = latest_roster["game_id"].to_numpy()
    contribution_frame["team_id"] = latest_roster["team_id"].to_numpy()
    contribution_frame["side"] = latest_roster["side"].to_numpy()
    contribution_frame["role"] = latest_roster["role"].to_numpy()
    team_counts = contribution_frame.groupby("team_id", sort=False)["player_id"].nunique()
    if not team_counts.eq(5).all():
        raise FutureValueSnapshotError("future team value requires exact five current-roster players")
    role_counts = contribution_frame.groupby("team_id", sort=False)["role"].nunique()
    if not role_counts.eq(5).all():
        raise FutureValueSnapshotError("future team value requires five unique current-roster roles")
    team_context = _team_history_features(map_frame, form)
    team_context = team_context.sort_values(["side", "game_id"], kind="stable")
    # Use the latest context row by team.  The historical helper is strict
    # about roster shape and keeps the side-specific continuity state separate.
    context_rows: list[dict[str, Any]] = []
    for team_id, group in latest_roster.groupby("team_id", sort=False):
        dates = pd.to_datetime(group["date"], utc=True)
        latest_date = dates.max()
        latest_group = group[dates.eq(latest_date)]
        if latest_group[["game_id", "side"]].drop_duplicates().shape[0] != 1:
            raise FutureValueSnapshotError("team has ambiguous latest roster context")
        game_id = str(latest_group["game_id"].iloc[0])
        side = str(latest_group["side"].iloc[0])
        context_match = team_context[team_context["game_id"].astype(str).eq(game_id) & team_context["side"].astype(str).eq(side)]
        if len(context_match) != 1:
            raise FutureValueSnapshotError("team context is incomplete or ambiguous")
        context_rows.append(
            {
                "team_id": str(team_id),
                "side": side,
                "last_game_id": game_id,
                "last_game_date": latest_date,
                "prior_team_win": context_match["prior_team_win"].iloc[0],
                "prior_team_support": context_match["prior_team_support"].iloc[0],
                "roster_continuity": context_match["roster_continuity"].iloc[0],
            }
        )
    context_frame = pd.DataFrame(context_rows)
    team_rows: list[dict[str, Any]] = []
    team_feature_names = set(str(value) for value in model.feature_names) & TEAM_FEATURES
    if not team_feature_names:
        team_blocker = ("team_context_not_in_final_model",)
    else:
        team_blocker = ()
    for team_id, group in contribution_frame.groupby("team_id", sort=True):
        context = context_frame[context_frame["team_id"].astype(str).eq(str(team_id))]
        if len(context) != 1:
            raise FutureValueSnapshotError("team context rows are ambiguous")
        player_value = float(group["role_normalized_player_value_logit"].sum())
        champion_value = float(group["champion_role_atom_logit"].sum())
        team_context_value: float | None = None
        if team_feature_names:
            team_context_value = 0.0
            for feature_index, feature in enumerate(str(value) for value in model.feature_names):
                if feature not in TEAM_FEATURES:
                    continue
                source_name = {
                    "team_prior_win_diff": "prior_team_win",
                    "roster_continuity_diff": "roster_continuity",
                }[feature]
                raw_value = _finite(context[source_name].iloc[0])
                imputation = _finite(np.asarray(model.imputation_values, dtype=float)[feature_index])
                if raw_value is None:
                    raw_value = imputation
                scale = _finite(np.asarray(model.scales, dtype=float)[feature_index])
                coefficient = _finite(np.asarray(model.coefficients, dtype=float)[feature_index])
                if raw_value is None or imputation is None or scale is None or coefficient is None or scale == 0:
                    raise FutureValueSnapshotError("team context model parameter is invalid")
                team_context_value += raw_value / scale * coefficient
        team_has_missing = bool(group["model_feature_missing"].any())
        team_status = "missing_features" if team_has_missing else "adequate"
        team_blockers: list[str] = []
        if team_has_missing:
            team_blockers.append("missing_model_feature_value")
        if "support_uncertainty_proxy_not_calibrated" in auth.blockers:
            team_blockers.append("support_uncertainty_proxy_not_calibrated")
        team_rows.append(
            _json_row(
                {
                    "team_id": str(team_id),
                    "side": str(group["side"].iloc[0]),
                    "last_game_id": str(context["last_game_id"].iloc[0]),
                    "last_game_date": context["last_game_date"].iloc[0],
                    "roster_player_count": int(len(group)),
                    "roster_player_ids": sorted(str(value) for value in group["player_id"]),
                    "role_normalized_player_value_logit": None
                    if team_has_missing
                    else player_value,
                    "champion_role_atom_logit": None
                    if team_has_missing
                    else champion_value,
                    "team_context_logit": None
                    if team_has_missing
                    else team_context_value,
                    "future_team_value_logit": None
                    if team_has_missing
                    else (
                        player_value + team_context_value
                        if team_context_value is not None
                        else player_value
                    ),
                    "team_context_status": (
                        "available" if team_context_value is not None else "missing_model_feature"
                    ),
                    "prior_team_win": context["prior_team_win"].iloc[0],
                    "prior_team_support": context["prior_team_support"].iloc[0],
                    "roster_continuity": context["roster_continuity"].iloc[0],
                    "support_status": team_status,
                    "status": "research_only" if team_status == "adequate" else "research_only_missing_features",
                    "blockers": sorted(set(team_blockers)),
                    "current_rank": None,
                    "future_rank": None,
                    "rank_delta": None,
                    "rank_join_status": "pending",
                }
            )
        )

    player_rank_diffs, player_rank_blockers, player_rank_coverage, player_rank_assignments = _rank_diffs(
        player_rows,
        current_player_ratings,
        identity="player_id",
        future_value="future_player_value_logit",
        current_value_candidates=("mu_effective", "mu_total", "rating"),
    )
    team_rank_diffs, team_rank_blockers, team_rank_coverage, team_rank_assignments = _rank_diffs(
        team_rows,
        current_team_ratings,
        identity="team_id",
        future_value="future_team_value_logit",
        current_value_candidates=("mu_effective", "mu_total", "rating"),
    )
    for row in player_rows:
        assignment = player_rank_assignments.get(str(row["player_id"]))
        if assignment is not None:
            row.update(assignment)
            row["rank_join_status"] = (
                "current_rating_unavailable"
                if player_rank_coverage["status"] == "unavailable"
                else "matched"
                if assignment["current_rank"] is not None
                else "current_rating_unmatched"
            )
        else:
            row["rank_join_status"] = (
                "future_value_unavailable"
                if _finite(row.get("future_player_value_logit")) is None
                else "current_rating_unavailable"
                if player_rank_coverage["status"] == "unavailable"
                else "current_rating_unmatched"
            )
    for row in team_rows:
        assignment = team_rank_assignments.get(str(row["team_id"]))
        if assignment is not None:
            row.update(assignment)
            row["rank_join_status"] = (
                "current_rating_unavailable"
                if team_rank_coverage["status"] == "unavailable"
                else "matched"
                if assignment["current_rank"] is not None
                else "current_rating_unmatched"
            )
        else:
            row["rank_join_status"] = (
                "future_value_unavailable"
                if _finite(row.get("future_team_value_logit")) is None
                else "current_rating_unavailable"
                if team_rank_coverage["status"] == "unavailable"
                else "current_rating_unmatched"
            )
    blockers = tuple(
        sorted(
            set(
                (
                    *auth.blockers,
                    *team_blocker,
                    *player_rank_blockers,
                    *team_rank_blockers,
                )
            )
        )
    )
    rank_extremes = {
        "player": _rank_extremes(player_rank_diffs),
        "team": _rank_extremes(team_rank_diffs),
    }
    payload: dict[str, Any] = {
        "schema_version": SNAPSHOT_RECEIPT_SCHEMA_VERSION,
        "status": "research_only" if not blockers else "research_only_partial",
        "authority": dict(SNAPSHOT_AUTHORITY),
        "source": source,
        "as_of": _utc_text(cutoff),
        "model": {
            "schema_version": MODEL_FIT_SCHEMA_VERSION,
            "variant": str(bound_model_receipt.get("variant"))
            if bound_model_receipt is not None
            else None,
            "receipt_sha256": auth.model_receipt_sha256,
        },
        "fit": auth.as_dict(),
        "player_row_count": len(player_rows),
        "team_row_count": len(team_rows),
        "player_rank_diff_count": len(player_rank_diffs),
        "team_rank_diff_count": len(team_rank_diffs),
        "rank_coverage": {
            "player": player_rank_coverage,
            "team": team_rank_coverage,
        },
        "rank_diff_extremes": rank_extremes,
        "blockers": list(blockers),
        "tierlists": {"recalculated": False, "status": "unchanged"},
    }
    payload["receipt_sha256"] = _sha256_bytes(_canonical_json_bytes(payload))
    return FutureValueSnapshotResult(
        "research_only" if not blockers else "research_only_partial",
        blockers,
        tuple(player_rows),
        tuple(team_rows),
        tuple(player_rank_diffs),
        tuple(team_rank_diffs),
        payload,
    )


def write_snapshot_bundle(destination: Path, result: FutureValueSnapshotResult) -> dict[str, Any]:
    """Write a research-only snapshot bundle to a new directory."""

    if destination.exists():
        raise FutureValueSnapshotError(f"snapshot output already exists: {destination}")
    destination.mkdir(parents=True)
    paths = {
        "player_snapshot": destination / "future-player-value-snapshot.json",
        "team_snapshot": destination / "future-team-value-snapshot.json",
        "player_rank_diffs": destination / "future-player-rank-diffs.json",
        "team_rank_diffs": destination / "future-team-rank-diffs.json",
        "receipt": destination / "future-value-snapshot-receipt.json",
    }
    rows = {
        "player_snapshot": list(result.player_rows),
        "team_snapshot": list(result.team_rows),
        "player_rank_diffs": list(result.player_rank_diffs),
        "team_rank_diffs": list(result.team_rank_diffs),
    }
    for key, path in paths.items():
        if key in rows:
            payload = {
                "schema_version": SCHEMA_VERSION,
                "status": result.status,
                "authority": dict(SNAPSHOT_AUTHORITY),
                "source_receipt_sha256": result.receipt["source"]["source_receipt_sha256"],
                "rows": rows[key],
                "blockers": list(result.blockers),
            }
            if key in {"player_rank_diffs", "team_rank_diffs"}:
                scope = "player" if key.startswith("player") else "team"
                payload["rank_coverage"] = dict(
                    result.receipt.get("rank_coverage", {}).get(scope, {})
                )
        else:
            payload = dict(result.receipt)
        path.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True, allow_nan=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": result.status,
        "authority": dict(SNAPSHOT_AUTHORITY),
        "source_receipt_sha256": result.receipt["source"]["source_receipt_sha256"],
        "files": {
            key: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for key, path in paths.items()
        },
        "blockers": list(result.blockers),
    }
    manifest["manifest_sha256"] = _sha256_bytes(_canonical_json_bytes(manifest))
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, sort_keys=True, allow_nan=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "FinalFitAuthorization",
    "FutureValueSnapshotError",
    "FutureValueSnapshotResult",
    "SNAPSHOT_AUTHORITY",
    "SNAPSHOT_RECEIPT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "authorize_final_fit",
    "build_future_value_snapshots",
    "load_final_fit_model",
    "write_snapshot_bundle",
]
