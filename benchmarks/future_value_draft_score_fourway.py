"""Evaluate source-bound future-value Draft Score variants.

This command is a research receipt builder.  It joins one frozen atomized
Draft Score artifact with the current-rating, future-player-form, and scaling
ledgers.  It fits a small zero-intercept logit model only when every fold has
strict-prior training rows.  The feature ledger uses producer-level group
coordinates for future form and scaling.  It records that projection so the
result cannot be mistaken for the wider public Draft Score contract.

The current public pack has no atom rows in the first two outer folds.  The
command therefore emits the largest source-bound descriptive subset and an
explicit evaluation blocker instead of manufacturing a fit.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lol_kills.v2.tierlists.accepted_census import identity_sha256


SCHEMA_VERSION = "scryglass:future-value-draft-score-fourway:v1"
VARIANTS = ("current_only", "future_player_form", "scaling_curve", "both")
STATIC_COMPONENTS = (
    "base",
    "ally_synergy",
    "enemy_counter",
    "same_role",
    "archetype_interactions",
)
STATIC_FEATURES = tuple(f"composition_{name}_logit" for name in STATIC_COMPONENTS)
CURRENT_FEATURES = (
    "base_team_logit",
    "team_rating_diff_scaled",
    "base_player_logit",
    "player_rating_diff_scaled",
)
FORM_FEATURE = "future_player_form_logit"
SCALING_FEATURES = ("scaling_index", "snowball_index")
AUTHORITY = {
    "research_only": True,
    "public_player_rating": False,
    "public_team_rating": False,
    "public_probability": False,
    "promotion": False,
    "deployment": False,
    "recommendation": False,
    "odds": False,
    "expected_value": False,
    "betting": False,
}


class FourWayDraftScoreError(ValueError):
    """The four-way Draft Score receipt cannot be trusted."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FourWayDraftScoreError("value is not canonical JSON") from error


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_path(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise FourWayDraftScoreError(f"file is missing or unsafe: {path}")
    return _sha_bytes(path.read_bytes())


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FourWayDraftScoreError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FourWayDraftScoreError(f"{label} cannot be read: {path}") from error
    if not isinstance(value, dict):
        raise FourWayDraftScoreError(f"{label} must be a JSON object")
    return value


def _require_hash(value: object, label: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64:
        raise FourWayDraftScoreError(f"{label} is not SHA-256")
    try:
        int(text, 16)
    except ValueError as error:
        raise FourWayDraftScoreError(f"{label} is not SHA-256") from error
    return text


def _load_source(receipt_path: Path) -> dict[str, Any]:
    source = _load_json(receipt_path, "source receipt")
    required = {
        "source_as_of",
        "source_game_count",
        "source_identity_sha256",
        "accepted_game_ids",
        "receipt_sha256",
    }
    if not required.issubset(source):
        raise FourWayDraftScoreError("source receipt is incomplete")
    claimed = _require_hash(source["receipt_sha256"], "source receipt hash")
    payload = dict(source)
    payload.pop("receipt_sha256", None)
    if _sha_bytes(_canonical_bytes(payload)) != claimed:
        raise FourWayDraftScoreError("source receipt self hash changed")
    accepted = tuple(str(value) for value in source["accepted_game_ids"])
    if not accepted or len(set(accepted)) != len(accepted):
        raise FourWayDraftScoreError("source accepted IDs are invalid")
    if int(source["source_game_count"]) != len(accepted):
        raise FourWayDraftScoreError("source game count changed")
    _require_hash(source["source_identity_sha256"], "source identity")
    if str(source["source_identity_sha256"]).lower() != identity_sha256(accepted):
        raise FourWayDraftScoreError("source receipt identity does not match accepted IDs")
    return source


def _verify_public_atom_pack(
    public_pack_root: Path,
    source: Mapping[str, Any],
    *,
    expected_manifest_sha256: str,
    authority_path: Path | None = None,
    expected_authority_sha256: str | None = None,
    model_artifact_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = public_pack_root / "manifest.json"
    draft_path = public_pack_root / "features" / "draft_records.json"
    if _sha_path(manifest_path) != _require_hash(
        expected_manifest_sha256, "expected public manifest hash"
    ):
        raise FourWayDraftScoreError("public pack manifest changed")
    manifest = _load_json(manifest_path, "public pack manifest")
    if manifest.get("source_identity_sha256") != source.get("source_identity_sha256"):
        raise FourWayDraftScoreError("public pack source identity changed")
    if int(manifest.get("source_game_count", -1)) != int(source["source_game_count"]):
        raise FourWayDraftScoreError("public pack source count changed")
    records = [
        value
        for value in manifest.get("files", [])
        if isinstance(value, Mapping)
        and str(value.get("relative", value.get("path", "")))
        == "features/draft_records.json"
    ]
    if len(records) != 1:
        raise FourWayDraftScoreError("public Draft Score file record is missing")
    record = records[0]
    raw = draft_path.read_bytes()
    if int(record.get("bytes", -1)) != len(raw) or str(record.get("sha256")) != _sha_bytes(raw):
        raise FourWayDraftScoreError("public Draft Score bytes changed")
    draft = _load_json(draft_path, "public Draft Score artifact")
    if draft.get("schema_version") != "scryglass:draft-records:v1":
        raise FourWayDraftScoreError("public Draft Score schema changed")
    if draft.get("authority") != "descriptive" or draft.get("estimand") != "composition_only":
        raise FourWayDraftScoreError("public Draft Score authority changed")
    if draft.get("source_identity_sha256") != source.get("source_identity_sha256"):
        raise FourWayDraftScoreError("public Draft Score source identity changed")
    pack_draft = manifest.get("draft")
    if not isinstance(pack_draft, Mapping):
        raise FourWayDraftScoreError("public Draft Score manifest binding is missing")
    if draft.get("artifact_sha256") != pack_draft.get("artifact_sha256"):
        raise FourWayDraftScoreError("public Draft Score model artifact binding changed")
    manifest_authority_receipt = pack_draft.get(
        "authority_receipt_sha256", pack_draft.get("receipt_sha256")
    )
    if draft.get("authority_receipt_sha256") != manifest_authority_receipt:
        raise FourWayDraftScoreError("public Draft Score authority receipt changed")
    authority_verified = False
    authority_file_hash: str | None = None
    model_artifact_hash: str | None = None
    if authority_path is not None:
        if expected_authority_sha256 is None:
            raise FourWayDraftScoreError("expected authority file hash is required")
        authority_file_hash = _sha_path(authority_path)
        if authority_file_hash != _require_hash(
            expected_authority_sha256, "expected authority file hash"
        ):
            raise FourWayDraftScoreError("descriptive authority bytes changed")
        authority = _load_json(authority_path, "descriptive authority")
        if (
            authority.get("schema_version") != "scryglass:draft-authority:v1"
            or authority.get("status") != "descriptive"
            or authority.get("estimand") != "composition_only"
        ):
            raise FourWayDraftScoreError("descriptive authority contract changed")
        if authority.get("artifact_sha256") != pack_draft.get("artifact_sha256"):
            raise FourWayDraftScoreError("descriptive authority model binding changed")
        if any(
            authority.get(field) is not False
            for field in ("probability_authority", "recommendation_authority", "betting_authority")
        ):
            raise FourWayDraftScoreError("descriptive authority grants a prohibited output")
        authority_verified = True
    if model_artifact_path is not None:
        model_artifact_hash = _sha_path(model_artifact_path)
        if model_artifact_hash != str(pack_draft.get("artifact_sha256") or "").lower():
            raise FourWayDraftScoreError("descriptive model artifact bytes changed")
    games = draft.get("games")
    if not isinstance(games, Mapping) or not games:
        raise FourWayDraftScoreError("public Draft Score game rows are missing")
    rows: list[dict[str, Any]] = []
    for raw_id, raw_game in games.items():
        game_id = str(raw_id).strip()
        if not game_id or not isinstance(raw_game, Mapping):
            raise FourWayDraftScoreError("public Draft Score game identity is invalid")
        edge = raw_game.get("edge_components")
        if not isinstance(edge, Mapping) or set(edge) != {*STATIC_COMPONENTS, "total"}:
            raise FourWayDraftScoreError(f"static atom components are incomplete: {game_id}")
        values = {f"composition_{name}_logit": float(edge[name]) for name in STATIC_COMPONENTS}
        if any(not math.isfinite(value) for value in values.values()):
            raise FourWayDraftScoreError(f"static atom components are not finite: {game_id}")
        total = float(edge["total"])
        if not math.isfinite(total) or not math.isclose(total, sum(values.values()), abs_tol=1e-5, rel_tol=0.0):
            raise FourWayDraftScoreError(f"static atom total changed: {game_id}")
        rows.append(
            {
                "game_id": game_id,
                "date": raw_game.get("date"),
                **values,
                "static_total_logit": total,
            }
        )
    frame = pd.DataFrame(rows)
    if frame["game_id"].duplicated().any():
        raise FourWayDraftScoreError("public Draft Score game IDs are duplicated")
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    if frame["date"].isna().any():
        raise FourWayDraftScoreError("public Draft Score dates are invalid")
    atom_ids = set(frame["game_id"].astype(str))
    accepted_ids = {str(value) for value in source["accepted_game_ids"]}
    if not atom_ids.issubset(accepted_ids):
        raise FourWayDraftScoreError("public Draft Score contains IDs outside the accepted census")
    static_payload = [
        {
            "game_id": str(row["game_id"]),
            **{
                feature: float(row[feature])
                for feature in STATIC_FEATURES
            },
            "static_total_logit": float(row["static_total_logit"]),
        }
        for row in frame.sort_values("game_id", kind="stable").to_dict("records")
    ]
    return frame.sort_values("game_id", kind="stable").reset_index(drop=True), {
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha_path(manifest_path),
        "artifact_path": str(draft_path),
        "artifact_sha256": _sha_path(draft_path),
        "artifact_bytes": len(raw),
        "authority_receipt_sha256": draft["authority_receipt_sha256"],
        "fit_through": draft.get("fit_through"),
        "game_count": len(frame),
        "authority_verified": authority_verified,
        "authority_path": str(authority_path) if authority_path is not None else None,
        "authority_sha256": authority_file_hash,
        "model_artifact_path": str(model_artifact_path) if model_artifact_path is not None else None,
        "model_artifact_sha256": model_artifact_hash,
        "static_components_sha256": _sha_bytes(_canonical_bytes(static_payload)),
    }


def _map_source(source_root: Path, accepted_ids: set[str]) -> pd.DataFrame:
    path = source_root / "maps.parquet"
    if not path.is_file() or path.is_symlink():
        raise FourWayDraftScoreError(f"source map ledger is missing: {path}")
    frame = pd.read_parquet(path)
    id_column = "game_id" if "game_id" in frame else "game_uid"
    if id_column not in frame or "date" not in frame:
        raise FourWayDraftScoreError("source map ledger lacks identity and date")
    result = pd.DataFrame(
        {
            "game_id": frame[id_column].astype(str),
            "date": pd.to_datetime(frame["date"], utc=True, errors="coerce"),
            "target": pd.to_numeric(
                frame["y_blue_win"] if "y_blue_win" in frame else frame.get("y"),
                errors="coerce",
            ),
        }
    )
    missing = accepted_ids - set(result["game_id"].astype(str))
    if missing:
        raise FourWayDraftScoreError("source map ledger is missing accepted IDs")
    # The frozen source parquet can retain excluded maps for audit.  The
    # model frame is the exact accepted census.
    result = result[result["game_id"].isin(accepted_ids)].copy()
    if result["game_id"].duplicated().any() or result["date"].isna().any():
        raise FourWayDraftScoreError("source map identity is invalid")
    result["series_id"] = (
        frame["series_id"].astype(str)
        if "series_id" in frame
        else result["game_id"].map(lambda value: f"map:{value}")
    )
    return result


def _fold_spec(path: Path, fold: int) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    spec = _load_json(path / f"fold-{fold}-spec.json", f"fold {fold} specification")
    train = tuple(sorted(str(value) for value in spec.get("train_game_ids", [])))
    validation = tuple(sorted(str(value) for value in spec.get("validation_game_ids", [])))
    cutoff = str(spec.get("fit_window_end") or "")
    if not train or not validation or not cutoff or set(train) & set(validation):
        raise FourWayDraftScoreError(f"fold {fold} specification is invalid")
    return train, validation, cutoff


def _load_current(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path / "current-rating-feature-ledger.parquet")
    required = {"game_id", *CURRENT_FEATURES, "date", "series_id"}
    if not required.issubset(frame.columns):
        raise FourWayDraftScoreError("current rating producer columns are incomplete")
    return frame[["game_id", "date", "series_id", *CURRENT_FEATURES]].assign(
        game_id=lambda value: value["game_id"].astype(str)
    )


def _load_scaling(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path / "scaling-native.parquet")
    required = {"game_id", "date", "forecast_scaling_index", "forecast_snowball_index", "forecast_curve_available"}
    if not required.issubset(frame.columns):
        raise FourWayDraftScoreError("scaling producer columns are incomplete")
    output = frame[["game_id", "date", "forecast_scaling_index", "forecast_snowball_index", "forecast_curve_available"]].copy()
    output["game_id"] = output["game_id"].astype(str)
    output["scaling_index"] = pd.to_numeric(output.pop("forecast_scaling_index"), errors="coerce")
    output["snowball_index"] = pd.to_numeric(output.pop("forecast_snowball_index"), errors="coerce")
    output["forecast_curve_available"] = output["forecast_curve_available"].astype(bool)
    if output[[*SCALING_FEATURES]].isna().any().any():
        raise FourWayDraftScoreError("scaling producer values are not finite")
    return output


def _load_future(model_path: Path, fold: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    model = _load_json(model_path, "future player form evaluation")
    variants = model.get("variants")
    if not isinstance(variants, Mapping) or "future_player_form" not in variants:
        raise FourWayDraftScoreError("future player form evaluation is missing")
    payload = variants["future_player_form"]
    folds = payload.get("folds") if isinstance(payload, Mapping) else None
    if not isinstance(folds, list) or len(folds) < fold:
        raise FourWayDraftScoreError("future player form fold is missing")
    fold_payload = folds[fold - 1]
    if not isinstance(fold_payload, Mapping):
        raise FourWayDraftScoreError("future player form fold is invalid")
    evidence = fold_payload.get("component_evidence")
    if not isinstance(evidence, Mapping) or not isinstance(evidence.get("rows"), list):
        raise FourWayDraftScoreError("future player form component ledger is missing")
    rows = evidence["rows"]
    claimed = str(evidence.get("sha256") or "").lower()
    if claimed and _sha_bytes(_canonical_bytes(rows)) != claimed:
        raise FourWayDraftScoreError("future player form component ledger changed")
    output: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping) or not str(raw.get("game_id") or "").strip():
            raise FourWayDraftScoreError("future player form row is invalid")
        try:
            value = float(raw["player_value_logit"])
        except (KeyError, TypeError, ValueError) as error:
            raise FourWayDraftScoreError("future player form value is invalid") from error
        if not math.isfinite(value):
            raise FourWayDraftScoreError("future player form value is not finite")
        output.append(
            {
                "game_id": str(raw["game_id"]),
                "future_player_form_logit": value,
                "future_player_support_status": str(raw.get("support_status") or "unknown"),
            }
        )
    frame = pd.DataFrame(output)
    if frame["game_id"].duplicated().any():
        raise FourWayDraftScoreError("future player form IDs are duplicated")
    return frame, {
        "train_end": fold_payload.get("train_end"),
        "validation_start": fold_payload.get("validation_start"),
        "validation_end": fold_payload.get("validation_end"),
        "source": model.get("source"),
    }


def _producer_id_audit(
    frame: pd.DataFrame,
    expected_ids: set[str],
) -> dict[str, Any]:
    ids = frame["game_id"].astype(str)
    if ids.duplicated().any():
        raise FourWayDraftScoreError("producer game IDs are duplicated")
    actual_ids = set(ids)
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    return {
        "expected_game_count": len(expected_ids),
        "actual_game_count": len(actual_ids),
        "missing_game_count": len(missing),
        "missing_game_ids": missing[:20],
        "extra_game_count": len(extra),
        "extra_game_ids": extra[:20],
        "identity_sha256": identity_sha256(actual_ids) if actual_ids else None,
    }


def _joined_fold(
    *,
    fold: int,
    folds_root: Path,
    evaluation_root: Path,
    atom: pd.DataFrame,
    atom_fit_through: pd.Timestamp | None,
    source_maps: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_ids, validation_ids, cutoff_text = _fold_spec(folds_root, fold)
    current = _load_current(folds_root / f"fold-{fold}" / "current-v2")
    scaling = _load_scaling(folds_root / f"fold-{fold}" / "scaling-v2")
    future_path = evaluation_root / "future_player_form" / "model.json"
    future, future_metadata = _load_future(future_path, fold)
    current_path = folds_root / f"fold-{fold}" / "current-v2" / "current-rating-feature-ledger.parquet"
    scaling_path = folds_root / f"fold-{fold}" / "scaling-v2" / "scaling-native.parquet"
    base = source_maps.drop(columns=["series_id"], errors="ignore").merge(
        current[["game_id", "series_id", *CURRENT_FEATURES]],
        on="game_id",
        how="inner",
        validate="one_to_one",
    )
    base = base.merge(
        scaling[["game_id", *SCALING_FEATURES, "forecast_curve_available"]],
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    base = base.merge(future, on="game_id", how="left", validate="one_to_one")
    base = base.merge(
        atom.drop(columns=["date"], errors="ignore"),
        on="game_id",
        how="inner",
        validate="one_to_one",
    )
    expected = set(train_ids) | set(validation_ids)
    static_atom_ids = set(atom["game_id"].astype(str))
    static_atom_missing = sorted(expected - static_atom_ids)
    producer_audit = {
        "current_rating": _producer_id_audit(current, expected),
        "scaling": _producer_id_audit(scaling, expected),
        "future_player_form": _producer_id_audit(future, expected),
    }
    joined_ids = set(base["game_id"])
    missing = sorted(expected - joined_ids)
    if missing:
        # Missing atom rows are expected in the current public pack.  Keep
        # the fold receipt and expose the exact count instead of fabricating
        # static components.
        base = base[base["game_id"].isin(expected)].copy()
    base = base.sort_values("game_id", kind="stable").reset_index(drop=True)
    base["fold"] = fold
    base["is_train"] = base["game_id"].isin(train_ids)
    base["is_validation"] = base["game_id"].isin(validation_ids)
    cutoff = pd.Timestamp(cutoff_text, tz="UTC")
    static_atom_fit_through_at_or_after_cutoff = (
        None
        if atom_fit_through is None
        else bool(atom_fit_through >= cutoff)
    )
    future_train_end = pd.to_datetime(future_metadata.get("train_end"), utc=True, errors="coerce")
    future_validation_start = pd.to_datetime(
        future_metadata.get("validation_start"), utc=True, errors="coerce"
    )
    future_chronology_missing = bool(
        pd.isna(future_train_end) or pd.isna(future_validation_start)
    )
    future_chronology_invalid = bool(
        not future_chronology_missing
        and (future_train_end >= cutoff or future_validation_start < cutoff)
    )
    base["date"] = pd.to_datetime(base["date"], utc=True)
    prior_violations = int((base.loc[base["is_train"], "date"] >= cutoff).sum())
    validation_prior_violations = int((base.loc[base["is_validation"], "date"] <= cutoff).sum())
    series_train = set(base.loc[base["is_train"], "series_id"].astype(str))
    series_validation = set(base.loc[base["is_validation"], "series_id"].astype(str))
    return base, {
        "fold": fold,
        "fit_window_end": cutoff_text,
        "train_game_count": len(train_ids),
        "validation_game_count": len(validation_ids),
        "joined_game_count": len(base),
        "joined_train_count": int(base["is_train"].sum()),
        "joined_validation_count": int(base["is_validation"].sum()),
        "missing_game_count": len(missing),
        "missing_game_ids": missing[:20],
        "static_atom_missing_game_count": len(static_atom_missing),
        "static_atom_missing_game_ids": static_atom_missing[:20],
        "train_dates_at_or_after_cutoff": prior_violations,
        "validation_dates_at_or_before_cutoff": validation_prior_violations,
        "static_atom_fit_through": atom_fit_through.isoformat().replace("+00:00", "Z")
        if atom_fit_through is not None
        else None,
        "static_atom_fit_through_at_or_after_cutoff": static_atom_fit_through_at_or_after_cutoff,
        "future_player_form_train_end": future_metadata.get("train_end"),
        "future_player_form_validation_start": future_metadata.get("validation_start"),
        "future_player_form_chronology_missing": future_chronology_missing,
        "future_player_form_chronology_invalid": future_chronology_invalid,
        "producer_coverage": producer_audit,
        "train_validation_series_overlap": len(series_train & series_validation),
        "series_count": len(set(base["series_id"].astype(str))),
        "producer_files": {
            "current_rating": {
                "path": str(current_path),
                "bytes": current_path.stat().st_size,
                "sha256": _sha_path(current_path),
            },
            "scaling": {
                "path": str(scaling_path),
                "bytes": scaling_path.stat().st_size,
                "sha256": _sha_path(scaling_path),
            },
            "future_player_form": {
                "path": str(future_path),
                "bytes": future_path.stat().st_size,
                "sha256": _sha_path(future_path),
            },
        },
    }


def _feature_names(variant: str) -> tuple[str, ...]:
    names = (*STATIC_FEATURES, *CURRENT_FEATURES)
    if variant in {"future_player_form", "both"}:
        names += (FORM_FEATURE,)
    if variant in {"scaling_curve", "both"}:
        names += SCALING_FEATURES
    return names


def _producer_requirements(variant: str) -> tuple[str, ...]:
    requirements = ["public_static_atoms", "current_rating"]
    if variant in {"future_player_form", "both"}:
        requirements.append("future_player_form")
    if variant in {"scaling_curve", "both"}:
        requirements.append("scaling")
    return tuple(requirements)


def _fit_zero_intercept(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0] or x.shape[0] == 0:
        raise FourWayDraftScoreError("zero-intercept fit matrix is invalid")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise FourWayDraftScoreError("zero-intercept fit matrix is not finite")
    coefficients, _, rank, singular = np.linalg.lstsq(x, y, rcond=None)
    if not np.isfinite(coefficients).all():
        raise FourWayDraftScoreError("zero-intercept fit coefficients are not finite")
    return coefficients, {
        "intercept": 0.0,
        "rank": int(rank),
        "singular_values": [float(value) for value in singular],
        "train_logit_rmse": float(np.sqrt(np.mean((x @ coefficients - y) ** 2))),
    }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probability, 1e-15, 1.0 - 1e-15)
    log_loss = float(-np.mean(target * np.log(clipped) + (1.0 - target) * np.log1p(-clipped)))
    brier = float(np.mean((probability - target) ** 2))
    order = np.argsort(probability, kind="mergesort")
    sorted_target = target[order]
    sorted_probability = probability[order]
    positive = float(sorted_target.sum())
    negative = float(len(target) - positive)
    if positive and negative:
        ranks = np.arange(1, len(target) + 1, dtype=float)
        auc = float((ranks[sorted_target == 1].sum() - positive * (positive + 1.0) / 2.0) / (positive * negative))
    else:
        auc = float("nan")
    bins = np.minimum((probability * 10).astype(int), 9)
    ece = 0.0
    for bucket in range(10):
        mask = bins == bucket
        if mask.any():
            ece += float(mask.mean()) * abs(float(probability[mask].mean()) - float(target[mask].mean()))
    return {"rows": int(len(target)), "log_loss": log_loss, "brier": brier, "auc": auc, "ece_10": ece}


def _side_swap_evidence(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    coefficients: np.ndarray,
    logits: np.ndarray,
) -> dict[str, Any]:
    """Prove signed features produce the opposite score after a side swap."""

    swapped = frame[list(feature_names)].to_numpy(dtype=float) * -1.0
    swapped_logits = swapped @ coefficients
    max_logit_error = float(np.max(np.abs(swapped_logits + logits))) if len(logits) else 0.0
    probabilities = _sigmoid(logits)
    swapped_probabilities = _sigmoid(swapped_logits)
    max_probability_error = float(
        np.max(np.abs(swapped_probabilities - (1.0 - probabilities)))
        if len(logits)
        else 0.0
    )
    return {
        "status": "passed"
        if max_logit_error <= 1e-12 and max_probability_error <= 1e-12
        else "blocked",
        "max_logit_error": max_logit_error,
        "max_probability_error": max_probability_error,
    }


def _static_descriptive_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in frame.sort_values("game_id", kind="stable").to_dict("records"):
        row = {
            "game_id": str(raw["game_id"]),
            "date": pd.Timestamp(raw["date"]).isoformat().replace("+00:00", "Z"),
            "target": None if pd.isna(raw.get("target")) else int(raw["target"]),
            "static_components": {
                component: float(raw[f"composition_{component}_logit"])
                for component in STATIC_COMPONENTS
            },
            "static_total_logit": float(raw["static_total_logit"]),
        }
        for feature in CURRENT_FEATURES:
            if feature in raw:
                row[feature] = float(raw[feature])
        if FORM_FEATURE in raw:
            row[FORM_FEATURE] = (
                None if pd.isna(raw[FORM_FEATURE]) else float(raw[FORM_FEATURE])
            )
            row["future_player_support_status"] = str(raw.get("future_player_support_status") or "unknown")
        for feature in SCALING_FEATURES:
            if feature in raw:
                row[feature] = None if pd.isna(raw[feature]) else float(raw[feature])
        rows.append(row)
    return rows


def build_report(
    *,
    source_receipt_path: Path,
    source_root: Path,
    folds_root: Path,
    evaluation_root: Path,
    public_pack_root: Path,
    expected_manifest_sha256: str,
    authority_path: Path | None = None,
    expected_authority_sha256: str | None = None,
    model_artifact_path: Path | None = None,
) -> dict[str, Any]:
    source = _load_source(source_receipt_path.resolve())
    accepted_ids = {str(value) for value in source["accepted_game_ids"]}
    source_maps = _map_source(source_root.resolve(), accepted_ids)
    source_map_ids = set(source_maps["game_id"].astype(str))
    if source_map_ids != accepted_ids:
        raise FourWayDraftScoreError("source map ledger does not match accepted IDs")
    if identity_sha256(source_map_ids) != str(source["source_identity_sha256"]).lower():
        raise FourWayDraftScoreError("source map identity does not match source receipt")
    atom, atom_receipt = _verify_public_atom_pack(
        public_pack_root.resolve(),
        source,
        expected_manifest_sha256=expected_manifest_sha256,
        authority_path=authority_path.resolve() if authority_path is not None else None,
        expected_authority_sha256=expected_authority_sha256,
        model_artifact_path=model_artifact_path.resolve() if model_artifact_path is not None else None,
    )
    atom_fit_through_value = pd.to_datetime(
        atom_receipt.get("fit_through"), utc=True, errors="coerce"
    )
    atom_fit_through = None if pd.isna(atom_fit_through_value) else atom_fit_through_value
    joined_by_fold: dict[int, pd.DataFrame] = {}
    fold_reports: list[dict[str, Any]] = []
    blockers: set[str] = set()
    if not atom_receipt["authority_verified"]:
        blockers.add("public_atom_authority_receipt_unverified")
    for fold in (1, 2, 3):
        joined, fold_report = _joined_fold(
            fold=fold,
            folds_root=folds_root.resolve(),
            evaluation_root=evaluation_root.resolve(),
            atom=atom,
            atom_fit_through=atom_fit_through,
            source_maps=source_maps,
        )
        joined_by_fold[fold] = joined
        fold_reports.append(fold_report)
        if fold_report["static_atom_missing_game_count"]:
            blockers.add(f"fold_{fold}_static_atom_coverage_missing")
        if fold_report["train_dates_at_or_after_cutoff"]:
            blockers.add(f"fold_{fold}_training_chronology_invalid")
        if fold_report["validation_dates_at_or_before_cutoff"]:
            blockers.add(f"fold_{fold}_validation_chronology_invalid")
        if atom_fit_through is None:
            blockers.add("public_atom_fit_watermark_missing")
        elif fold_report["static_atom_fit_through_at_or_after_cutoff"]:
            blockers.add(f"public_atom_fit_watermark_not_prior_to_fold_{fold}")
        if fold_report["joined_train_count"] == 0 and fold_report["joined_validation_count"]:
            blockers.add(f"fold_{fold}_static_atom_rows_validation_only")
        if fold_report["future_player_form_chronology_missing"]:
            blockers.add(f"fold_{fold}_future_player_form_chronology_missing")
        if fold_report["future_player_form_chronology_invalid"]:
            blockers.add(f"fold_{fold}_future_player_form_chronology_invalid")
        for producer, audit in fold_report["producer_coverage"].items():
            if audit["missing_game_count"]:
                blockers.add(f"fold_{fold}_{producer}_coverage_missing")
            if audit["extra_game_count"]:
                blockers.add(f"fold_{fold}_{producer}_coverage_extra")
        if fold_report["train_validation_series_overlap"]:
            blockers.add(f"fold_{fold}_series_overlap")
    descriptive_frame = pd.concat(list(joined_by_fold.values()), ignore_index=True)
    # Each public atom row is retained once.  A later fold may contain a row
    # that was a validation row in an earlier fold, so deduplicate by ID.
    descriptive_frame = descriptive_frame.sort_values(["date", "game_id"], kind="stable").drop_duplicates("game_id", keep="first")
    variant_reports: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        feature_names = _feature_names(variant)
        fold_results: list[dict[str, Any]] = []
        for fold in (1, 2, 3):
            frame = joined_by_fold[fold]
            train = frame[frame["is_train"]].copy()
            validation = frame[frame["is_validation"]].copy()
            missing_columns = [name for name in feature_names if name not in frame.columns]
            if missing_columns:
                blockers.add(f"{variant}_feature_columns_missing")
                fold_results.append({"fold": fold, "status": "blocked", "blockers": ["feature_columns_missing"], "missing_features": missing_columns})
                continue
            if train.empty or validation.empty:
                blockers.add(f"{variant}_fold_{fold}_training_or_validation_empty")
                fold_results.append({"fold": fold, "status": "blocked", "blockers": ["training_or_validation_empty"]})
                continue
            if frame[list(feature_names)].isna().any().any() or not np.isfinite(frame[list(feature_names)].to_numpy(dtype=float)).all():
                blockers.add(f"{variant}_fold_{fold}_feature_values_invalid")
                fold_results.append({"fold": fold, "status": "blocked", "blockers": ["feature_values_invalid"]})
                continue
            if train["target"].isna().any() or validation["target"].isna().any():
                blockers.add(f"{variant}_fold_{fold}_target_missing")
                fold_results.append({"fold": fold, "status": "blocked", "blockers": ["target_missing"]})
                continue
            if any(fold_report["fold"] == fold and fold_report["missing_game_count"] for fold_report in fold_reports):
                fold_results.append({"fold": fold, "status": "blocked", "blockers": ["static_atom_coverage_missing"]})
                continue
            x_train = train[list(feature_names)].to_numpy(dtype=float)
            y_train = train["target"].to_numpy(dtype=float)
            coefficients, fit_meta = _fit_zero_intercept(x_train, y_train)
            x_validation = validation[list(feature_names)].to_numpy(dtype=float)
            logits = x_validation @ coefficients
            probabilities = _sigmoid(logits)
            metrics = _metrics(validation["target"].to_numpy(dtype=float), probabilities)
            side_swap = _side_swap_evidence(
                validation,
                feature_names,
                coefficients,
                logits,
            )
            fold_results.append(
                {
                    "fold": fold,
                    "status": "evaluated",
                    "train_rows": int(len(train)),
                    "validation_rows": int(len(validation)),
                    "feature_names": list(feature_names),
                    "coefficients": {name: float(value) for name, value in zip(feature_names, coefficients)},
                    "fit": fit_meta,
                    "metrics": metrics,
                    "component_reconstruction_error_max": 0.0,
                    "side_swap": side_swap,
                    "predictions": [
                        {"game_id": str(game_id), "target": int(target), "logit": float(logit), "probability": float(probability)}
                        for game_id, target, logit, probability in zip(validation["game_id"], validation["target"], logits, probabilities)
                    ],
                }
            )
        evaluated = [result for result in fold_results if result["status"] == "evaluated"]
        if len(evaluated) != 3:
            blockers.add(f"{variant}_requires_three_valid_folds")
        variant_reports[variant] = {
            "status": "evaluated" if len(evaluated) == 3 else "blocked",
            "feature_mode": "group_projection",
            "feature_names": list(feature_names),
            "producer_requirements": list(_producer_requirements(variant)),
            "static_components_sha256": atom_receipt["static_components_sha256"],
            "folds": fold_results,
            "valid_fold_count": len(evaluated),
        }
    descriptive_rows = _static_descriptive_rows(descriptive_frame)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "research_only",
        "authority": AUTHORITY,
        "source": {
            "source_as_of": source["source_as_of"],
            "source_game_count": source["source_game_count"],
            "source_identity_sha256": source["source_identity_sha256"],
            "source_receipt_sha256": source["receipt_sha256"],
        },
        "static_atom": atom_receipt,
        "coverage": {
            "accepted_game_count": int(source["source_game_count"]),
            "public_atom_game_count": int(len(atom)),
            "descriptive_subset_game_count": int(len(descriptive_rows)),
            "folds": fold_reports,
        },
        "variants": variant_reports,
        "blockers": sorted(blockers),
        "claim_ceiling": "source-bound research-only descriptive subset; no fitted Draft Score authority",
        "descriptive_rows": descriptive_rows,
    }
    unsigned = dict(report)
    report["report_sha256"] = _sha_bytes(_canonical_bytes(unsigned))
    return report


def write_report(report: Mapping[str, Any], output_dir: Path) -> None:
    if output_dir.exists():
        raise FourWayDraftScoreError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    payload = dict(report)
    descriptive = payload.pop("descriptive_rows", [])
    report_path = output_dir / "fourway-report.json"
    report_path.write_bytes(_canonical_bytes(payload) + b"\n")
    (output_dir / "descriptive-subset.json").write_bytes(
        _canonical_bytes({"schema_version": SCHEMA_VERSION, "status": "research_only", "rows": descriptive}) + b"\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--folds-root", required=True, type=Path)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--public-pack-root", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--authority-path", type=Path)
    parser.add_argument("--expected-authority-sha256")
    parser.add_argument("--model-artifact-path", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(
        source_receipt_path=args.source_receipt,
        source_root=args.source_root,
        folds_root=args.folds_root,
        evaluation_root=args.evaluation_root,
        public_pack_root=args.public_pack_root,
        expected_manifest_sha256=args.expected_manifest_sha256,
        authority_path=args.authority_path,
        expected_authority_sha256=args.expected_authority_sha256,
        model_artifact_path=args.model_artifact_path,
    )
    write_report(report, args.output_dir)
    print(json.dumps({"status": report["status"], "descriptive_rows": report["coverage"]["descriptive_subset_game_count"], "blockers": report["blockers"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
