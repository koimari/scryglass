"""Research-only four-way Tier List shadows for future-value ratings.

The public Tier List model uses a chronological team Elo offset.  This module
keeps the champion, patch, role, atom, outcome, and fit contracts fixed while
replacing that offset with one verified strict-prior rating prediction.  The
result is a controlled nuisance-offset study.  It never grants public authority.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from lol_kills.research.future_value_rating import (
    validate_future_value_source_receipt_payload,
)
from lol_kills.v2.tierlists.accepted_census import (
    canonical_game_ids,
    identity_sha256,
)
from lol_kills.v2.tierlists.pooled_candidate import (
    PRE_MAP_OFFSET_PROVENANCE_SCHEMA,
)


SCHEMA_VERSION = "scryglass:future-value-tierlist-fourway:v1"
TRUST_SCHEMA_VERSION = "scryglass:future-value-tierlist-freeze:v1"
PINNED_TRUST_MANIFEST_RAW_SHA256 = (
    "95d2cd56cb33fa2105c65edbbb2a562f3e4b170bedf44e255373ed25bbb5919c"
)
VARIANTS = ("current_only", "future_player_form", "scaling_curve", "both")
REFERENCE_VARIANT = "current_only"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
AUTHORITY = {
    "research_only": True,
    "public_tierlist": False,
    "promotion": False,
    "deployment": False,
    "public_probability": False,
    "odds": False,
    "expected_value": False,
    "recommendation": False,
    "betting": False,
}
ROW_VALUE_FIELDS = (
    "rank",
    "tier_bucket",
    "tier_value_pp",
    "strength_score",
    "strength_sd_logit",
    "rating",
    "played_maps",
    "counterability_status",
    "counterability",
    "matchup_maps",
    "matchup_opponents",
)
NUMERIC_DELTA_FIELDS = (
    "tier_value_pp",
    "strength_score",
    "strength_sd_logit",
    "rating",
    "counterability",
    "matchup_maps",
)


class FutureValueTierListError(ValueError):
    """The four-way Tier List shadow contract failed closed."""


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise FutureValueTierListError("value is not canonical JSON") from error


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise FutureValueTierListError(f"{label} is not a SHA-256 digest")
    return value


def load_trust_manifest(path: Path, *, expected_raw_sha256: str) -> dict[str, Any]:
    """Load one externally pinned, closed freeze manifest."""

    expected = _require_hash(expected_raw_sha256, "expected trust manifest hash")
    if not path.is_file() or path.is_symlink():
        raise FutureValueTierListError("trust manifest is missing or unsafe")
    if sha256_path(path) != expected:
        raise FutureValueTierListError("trust manifest file hash changed")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FutureValueTierListError("trust manifest cannot be read") from error
    if not isinstance(payload, Mapping):
        raise FutureValueTierListError("trust manifest is not an object")
    required = {
        "schema_version",
        "status",
        "authority",
        "source",
        "evaluations",
        "tier_assets",
        "baseline_candidate",
        "trust_root_sha256",
    }
    if set(payload) != required:
        raise FutureValueTierListError("trust manifest schema is closed")
    if payload.get("schema_version") != TRUST_SCHEMA_VERSION:
        raise FutureValueTierListError("trust manifest schema changed")
    if payload.get("status") != "research_only" or payload.get("authority") != AUTHORITY:
        raise FutureValueTierListError("trust manifest authority changed")
    unsigned = dict(payload)
    claimed = _require_hash(unsigned.pop("trust_root_sha256"), "trust root hash")
    if sha256_bytes(canonical_json_bytes(unsigned)) != claimed:
        raise FutureValueTierListError("trust manifest self hash changed")
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, Mapping) or set(evaluations) != set(VARIANTS):
        raise FutureValueTierListError("trust manifest variant set changed")
    for variant, record in evaluations.items():
        if not isinstance(record, Mapping) or set(record) != {"locator", "raw_sha256"}:
            raise FutureValueTierListError(f"{variant} trust record changed")
        _require_hash(record.get("raw_sha256"), f"{variant} model hash")
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise FutureValueTierListError("trust manifest source binding is missing")
    for field in (
        "source_as_of",
        "source_game_count",
        "source_identity_sha256",
        "source_receipt_sha256",
        "source_receipt_file_sha256",
        "player_source_sha256",
        "maps_source_sha256",
        "meta_source_sha256",
        "model_eligible_game_count",
        "model_eligible_identity_sha256",
    ):
        if field not in source:
            raise FutureValueTierListError(f"trust manifest source field is missing: {field}")
    for field in (
        "source_identity_sha256",
        "source_receipt_sha256",
        "source_receipt_file_sha256",
        "player_source_sha256",
        "maps_source_sha256",
        "meta_source_sha256",
        "model_eligible_identity_sha256",
    ):
        _require_hash(source.get(field), f"trust source {field}")
    eligible_count = source.get("model_eligible_game_count")
    if (
        isinstance(eligible_count, bool)
        or not isinstance(eligible_count, int)
        or eligible_count <= 0
    ):
        raise FutureValueTierListError("trust source model-eligible count is invalid")
    assets = payload.get("tier_assets")
    if not isinstance(assets, Mapping) or not assets:
        raise FutureValueTierListError("trust manifest Tier assets are missing")
    for locator, digest in assets.items():
        if not isinstance(locator, str) or locator.startswith("/") or ".." in Path(locator).parts:
            raise FutureValueTierListError("Tier asset locator is unsafe")
        _require_hash(digest, f"Tier asset hash {locator}")
    baseline = payload.get("baseline_candidate")
    if not isinstance(baseline, Mapping) or set(baseline) != {"locator", "raw_sha256"}:
        raise FutureValueTierListError("baseline Tier candidate binding changed")
    _require_hash(baseline.get("raw_sha256"), "baseline Tier candidate hash")
    return dict(payload)


def _verify_authority(value: object, label: str) -> None:
    if not isinstance(value, Mapping) or value.get("research_only") is not True:
        raise FutureValueTierListError(f"{label} research authority is missing")
    if any(bool(flag) for name, flag in value.items() if name != "research_only"):
        raise FutureValueTierListError(f"{label} grants public authority")


def _finite_probability(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise FutureValueTierListError(f"{label} is not numeric") from error
    if not math.isfinite(number) or not 0.0 < number < 1.0:
        raise FutureValueTierListError(f"{label} is not a finite open probability")
    return number


def offset_values_sha256(offsets: Mapping[str, float]) -> str:
    """Hash one canonical, ordered map of game logits."""

    rows = [
        {"game_id": game_id, "logit": float(offsets[game_id])}
        for game_id in sorted(offsets)
    ]
    return sha256_bytes(canonical_json_bytes(rows))


def _required_utc_timestamp(value: object, label: str) -> pd.Timestamp:
    """Parse a required timezone-aware UTC timestamp."""

    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise FutureValueTierListError(f"{label} is not a timestamp") from error
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise FutureValueTierListError(f"{label} must be timezone-aware")
    return stamp.tz_convert("UTC")


def _validate_fold_chronology(
    result: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    maps_path: Path | None = None,
    expected_maps_sha256: str | None = None,
) -> dict[str, Any]:
    """Require fold receipts that prove strict-prior scoring.

    The model artifact contains fold boundaries and validation game IDs.  The
    optional frozen maps input completes the check by binding every scored
    game date to its fold interval.  Production callers must provide it.
    """

    folds = result.get("folds")
    if not isinstance(folds, list) or len(folds) != 3:
        raise FutureValueTierListError(f"{variant} fold chronology is missing")
    by_fold: dict[int, list[str]] = {1: [], 2: [], 3: []}
    for row in rows:
        try:
            fold = int(row.get("fold"))
        except (TypeError, ValueError) as error:
            raise FutureValueTierListError(f"{variant} prediction fold is invalid") from error
        if fold not in by_fold:
            raise FutureValueTierListError(f"{variant} prediction fold is outside the fold receipt")
        game_id = str(row.get("game_id") or "").strip()
        if not game_id:
            raise FutureValueTierListError(f"{variant} prediction game identity is missing")
        by_fold[fold].append(game_id)

    seen_folds: set[int] = set()
    seen_games: set[str] = set()
    fold_audit: list[dict[str, Any]] = []
    chronology_blockers: set[str] = set()
    for fold_record in folds:
        if not isinstance(fold_record, Mapping):
            raise FutureValueTierListError(f"{variant} fold receipt is invalid")
        try:
            fold = int(fold_record.get("fold"))
        except (TypeError, ValueError) as error:
            raise FutureValueTierListError(f"{variant} fold receipt number is invalid") from error
        if fold not in by_fold or fold in seen_folds:
            raise FutureValueTierListError(f"{variant} fold receipt set is invalid")
        seen_folds.add(fold)
        scored_ids = by_fold[fold]
        if not scored_ids or len(scored_ids) != len(set(scored_ids)):
            raise FutureValueTierListError(f"{variant} fold scored IDs are invalid")
        paired_ids = fold_record.get("paired_game_ids")
        if not isinstance(paired_ids, list) or [str(value) for value in paired_ids] != scored_ids:
            raise FutureValueTierListError(f"{variant} fold scored IDs do not match the ledger")
        expected_fold_identity = identity_sha256(scored_ids)
        if fold_record.get("paired_game_id_count") != len(scored_ids):
            raise FutureValueTierListError(f"{variant} fold scored count changed")
        if fold_record.get("validation_game_id_count") != len(scored_ids):
            raise FutureValueTierListError(f"{variant} fold validation count changed")
        if fold_record.get("validation_game_identity_sha256") != expected_fold_identity:
            raise FutureValueTierListError(f"{variant} fold validation identity changed")
        if fold_record.get("paired_game_identity_sha256") not in (None, expected_fold_identity):
            raise FutureValueTierListError(f"{variant} fold paired identity changed")
        if seen_games.intersection(scored_ids):
            raise FutureValueTierListError(f"{variant} fold validation IDs overlap")
        seen_games.update(scored_ids)

        train_end = _required_utc_timestamp(fold_record.get("train_end"), f"{variant} fold {fold} train_end")
        validation_start = _required_utc_timestamp(
            fold_record.get("validation_start"), f"{variant} fold {fold} validation_start"
        )
        validation_end = _required_utc_timestamp(
            fold_record.get("validation_end"), f"{variant} fold {fold} validation_end"
        )
        interval_start = _required_utc_timestamp(
            fold_record.get("validation_interval_start"),
            f"{variant} fold {fold} validation_interval_start",
        )
        interval_end = _required_utc_timestamp(
            fold_record.get("validation_interval_end"),
            f"{variant} fold {fold} validation_interval_end",
        )
        if not (train_end < validation_start and interval_start <= validation_start <= validation_end <= interval_end):
            raise FutureValueTierListError(f"{variant} fold {fold} chronology is not ordered")
        ledger_binding = fold_record.get("feature_ledger_binding")
        if not isinstance(ledger_binding, Mapping):
            raise FutureValueTierListError(f"{variant} fold {fold} feature timing binding is missing")
        fit_max = _required_utc_timestamp(
            ledger_binding.get("fit_date_max"), f"{variant} fold {fold} fit_date_max"
        )
        fit_window_end = _required_utc_timestamp(
            ledger_binding.get("fit_window_end"), f"{variant} fold {fold} fit_window_end"
        )
        if not fit_max < validation_start or fit_window_end > validation_start:
            raise FutureValueTierListError(f"{variant} fold {fold} fit is not strictly prior")
        if ledger_binding.get("strict_prior_timing") != "fit_rows_strictly_before_cutoff":
            raise FutureValueTierListError(f"{variant} fold {fold} strict-prior contract is missing")
        if ledger_binding.get("same_timestamp_policy") != "batch_exclude_same_timestamp":
            raise FutureValueTierListError(f"{variant} fold {fold} timestamp policy is missing")
        series_safety = ledger_binding.get("series_safety")
        if not isinstance(series_safety, Mapping) or series_safety.get("policy") != "whole_series_disjoint":
            raise FutureValueTierListError(f"{variant} fold {fold} series safety is missing")
        train_series_identity = _require_hash(
            series_safety.get("train_series_identity_sha256"),
            f"{variant} fold {fold} train series identity",
        )
        validation_series_identity = _require_hash(
            series_safety.get("validation_series_identity_sha256"),
            f"{variant} fold {fold} validation series identity",
        )

        # The rating ledger currently stores only digests for its series
        # partition.  A digest supplied by the producer does not let this
        # consumer prove that the train and validation clusters are disjoint.
        # A future artifact may carry the canonical IDs.  When it does, bind
        # the IDs to the existing digests and check the partition here.
        raw_train_series_ids = series_safety.get("train_series_ids")
        raw_validation_series_ids = series_safety.get("validation_series_ids")
        if not isinstance(raw_train_series_ids, list) or not isinstance(raw_validation_series_ids, list):
            series_disjointness = {
                "status": "blocked",
                "policy": "whole_series_disjoint",
                "train_series_identity_sha256": train_series_identity,
                "validation_series_identity_sha256": validation_series_identity,
                "blocker": "series_disjointness_not_independently_verified",
                "limitation": "exact train and validation cluster IDs are not present in the feature ledger",
            }
            chronology_blocker = "series_disjointness_not_independently_verified"
        else:
            train_series_ids = [str(value).strip() for value in raw_train_series_ids]
            validation_series_ids = [str(value).strip() for value in raw_validation_series_ids]
            if (
                not train_series_ids
                or not validation_series_ids
                or any(not value for value in train_series_ids + validation_series_ids)
                or len(train_series_ids) != len(set(train_series_ids))
                or len(validation_series_ids) != len(set(validation_series_ids))
            ):
                raise FutureValueTierListError(f"{variant} fold {fold} series IDs are invalid")
            if set(train_series_ids) & set(validation_series_ids):
                raise FutureValueTierListError(f"{variant} fold {fold} series IDs overlap")
            if identity_sha256(train_series_ids) != train_series_identity:
                raise FutureValueTierListError(f"{variant} fold {fold} train series identity changed")
            if identity_sha256(validation_series_ids) != validation_series_identity:
                raise FutureValueTierListError(
                    f"{variant} fold {fold} validation series identity changed"
                )
            series_disjointness = {
                "status": "verified",
                "policy": "whole_series_disjoint",
                "train_series_identity_sha256": train_series_identity,
                "validation_series_identity_sha256": validation_series_identity,
                "train_series_ids": sorted(train_series_ids),
                "validation_series_ids": sorted(validation_series_ids),
                "train_series_count": len(train_series_ids),
                "validation_series_count": len(validation_series_ids),
            }
            chronology_blocker = None

        fold_audit.append(
            {
                "fold": fold,
                "game_count": len(scored_ids),
                "game_identity_sha256": expected_fold_identity,
                "fit_date_max": fit_max.isoformat().replace("+00:00", "Z"),
                "validation_interval_start": interval_start.isoformat().replace("+00:00", "Z"),
                "validation_interval_end": interval_end.isoformat().replace("+00:00", "Z"),
                "series_disjointness": series_disjointness,
            }
        )
        if chronology_blocker is not None:
            chronology_blockers.add(chronology_blocker)

    if seen_folds != {1, 2, 3} or seen_games != {str(row.get("game_id")) for row in rows}:
        raise FutureValueTierListError(f"{variant} fold coverage does not match the ledger")

    date_check = "deferred_to_frozen_maps"
    if maps_path is not None:
        if expected_maps_sha256 is None:
            raise FutureValueTierListError("frozen maps hash is required for chronology")
        expected_maps = _require_hash(expected_maps_sha256, "chronology maps hash")
        if not maps_path.is_file() or maps_path.is_symlink() or sha256_path(maps_path) != expected_maps:
            raise FutureValueTierListError("chronology maps bytes changed")
        try:
            frame = pd.read_parquet(maps_path, columns=["game_uid", "date"])
        except (OSError, ValueError, KeyError) as error:
            raise FutureValueTierListError("chronology maps cannot be read") from error
        frame["game_uid"] = frame["game_uid"].astype(str)
        if frame["game_uid"].duplicated().any():
            raise FutureValueTierListError("chronology maps contain duplicate IDs")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
        if frame["date"].isna().any():
            raise FutureValueTierListError("chronology maps contain invalid dates")
        dates = frame.set_index("game_uid")["date"]
        for fold_record in folds:
            fold = int(fold_record["fold"])
            validation_start = _required_utc_timestamp(
                fold_record["validation_start"],
                f"{variant} fold {fold} validation_start",
            )
            validation_end = _required_utc_timestamp(
                fold_record["validation_end"],
                f"{variant} fold {fold} validation_end",
            )
            missing = sorted(set(by_fold[fold]) - set(dates.index))
            if missing:
                raise FutureValueTierListError(f"{variant} fold {fold} scored maps are missing")
            fold_dates = dates.loc[by_fold[fold]]
            if bool((fold_dates < validation_start).any()) or bool((fold_dates > validation_end).any()):
                raise FutureValueTierListError(f"{variant} fold {fold} scored dates leave the validation window")
        date_check = "verified_against_frozen_maps"
    blockers = sorted(chronology_blockers)
    return {
        "status": "blocked" if blockers else "verified",
        "date_check": date_check,
        "blockers": blockers,
        "folds": sorted(fold_audit, key=lambda item: item["fold"]),
    }


def load_prediction_offsets(
    path: Path,
    *,
    variant: str,
    expected_raw_sha256: str,
    source: Mapping[str, Any],
    maps_path: Path,
    expected_maps_sha256: str,
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    """Verify one fitted evaluation artifact and return strict-prior logits."""

    if variant not in VARIANTS:
        raise FutureValueTierListError("unknown rating variant")
    expected = _require_hash(expected_raw_sha256, f"{variant} expected model hash")
    if not path.is_file() or path.is_symlink() or sha256_path(path) != expected:
        raise FutureValueTierListError(f"{variant} model artifact bytes changed")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FutureValueTierListError(f"{variant} model artifact cannot be read") from error
    if document.get("schema_version") != "scryglass:future-value-four-variant-evaluation:v1":
        raise FutureValueTierListError(f"{variant} evaluation schema changed")
    model_source = document.get("source")
    if not isinstance(model_source, Mapping):
        raise FutureValueTierListError(f"{variant} model source is missing")
    expected_source_fields = {
        "source_as_of": source["source_as_of"],
        "source_game_count": source["source_game_count"],
        "source_identity_sha256": source["source_identity_sha256"],
        "source_receipt_sha256": source["source_receipt_sha256"],
    }
    for field, expected_value in expected_source_fields.items():
        if model_source.get(field) != expected_value:
            raise FutureValueTierListError(f"{variant} model source changed: {field}")
    variants = document.get("variants")
    if not isinstance(variants, Mapping) or set(variants) != {variant}:
        raise FutureValueTierListError(f"{variant} model variant set changed")
    result = variants[variant]
    if not isinstance(result, Mapping) or result.get("status") != "development_evaluated":
        raise FutureValueTierListError(f"{variant} model is not evaluated")
    if result.get("variant") != variant:
        raise FutureValueTierListError(f"{variant} result identity changed")
    _verify_authority(result.get("authority"), f"{variant} result")
    result_source = result.get("source")
    if not isinstance(result_source, Mapping):
        raise FutureValueTierListError(f"{variant} result source is missing")
    for field, expected_value in {
        **expected_source_fields,
        "source_receipt_file_sha256": source["source_receipt_file_sha256"],
        **(
            {
                "model_eligible_game_count": source["model_eligible_game_count"],
                "model_eligible_identity_sha256": source["model_eligible_identity_sha256"],
            }
            if "model_eligible_game_count" in source
            and "model_eligible_identity_sha256" in source
            else {}
        ),
    }.items():
        if result_source.get(field) != expected_value:
            raise FutureValueTierListError(f"{variant} result source changed: {field}")
    expected_accepted_ids = source.get("accepted_game_ids")
    expected_eligible_ids = source.get("model_eligible_game_ids")
    if expected_accepted_ids is not None and result_source.get("accepted_game_ids") != expected_accepted_ids:
        raise FutureValueTierListError(f"{variant} result accepted census changed")
    if expected_eligible_ids is not None and result_source.get("model_eligible_game_ids") != expected_eligible_ids:
        raise FutureValueTierListError(f"{variant} result eligible census changed")
    ledger = result.get("prediction_ledger")
    if not isinstance(ledger, Mapping) or ledger.get("schema_version") not in {
        "scryglass:future-value-prediction-ledger:v1",
        "scryglass:future-value-prediction-ledger:v2",
    }:
        raise FutureValueTierListError(f"{variant} prediction ledger is missing")
    rows = ledger.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise FutureValueTierListError(f"{variant} prediction rows are invalid")
    if ledger.get("row_count") != len(rows):
        raise FutureValueTierListError(f"{variant} prediction row count changed")
    if sha256_bytes(canonical_json_bytes(rows)) != _require_hash(
        ledger.get("sha256"), f"{variant} prediction ledger hash"
    ):
        raise FutureValueTierListError(f"{variant} prediction ledger values changed")
    chronology = _validate_fold_chronology(
        result,
        rows,
        variant=variant,
        maps_path=maps_path,
        expected_maps_sha256=expected_maps_sha256,
    )
    chronology_blockers = chronology.get("blockers", [])
    if not isinstance(chronology_blockers, list) or not all(
        isinstance(value, str) for value in chronology_blockers
    ):
        raise FutureValueTierListError(f"{variant} chronology blocker binding is invalid")
    offsets: dict[str, float] = {}
    targets: dict[str, float] = {}
    ordered_ids: list[str] = []
    for row in rows:
        game_id = str(row.get("game_id") or "").strip()
        if not game_id or game_id in offsets:
            raise FutureValueTierListError(f"{variant} prediction identity is invalid")
        try:
            fold = int(row.get("fold"))
            target = float(row.get("target"))
        except (TypeError, ValueError) as error:
            raise FutureValueTierListError(f"{variant} prediction row is invalid") from error
        if fold not in (1, 2, 3) or target not in (0.0, 1.0):
            raise FutureValueTierListError(f"{variant} fold or target is invalid")
        probability = _finite_probability(row.get("candidate"), f"{variant} {game_id}")
        offsets[game_id] = math.log(probability / (1.0 - probability))
        targets[game_id] = target
        ordered_ids.append(game_id)
    if identity_sha256(ordered_ids) != _require_hash(
        ledger.get("game_identity_sha256"), f"{variant} game identity"
    ):
        raise FutureValueTierListError(f"{variant} prediction game identity changed")
    offsets_hash = offset_values_sha256(offsets)
    return offsets, targets, {
        "variant": variant,
        "producer": f"future_value_rating:{variant}",
        "artifact_locator": str(path),
        "artifact_raw_sha256": expected,
        "prediction_ledger_sha256": str(ledger["sha256"]),
        "offsets_sha256": offsets_hash,
        "variant_receipt_sha256": str(result.get("variant_receipt", {}).get("receipt_sha256") or ""),
        "blockers": sorted(
            {
                *(str(value) for value in result.get("blockers", [])),
                *chronology_blockers,
            }
        ),
        "chronology": chronology,
    }


def validate_common_prediction_universe(
    offsets: Mapping[str, Mapping[str, float]],
    targets: Mapping[str, Mapping[str, float]],
    *,
    accepted_game_ids: Sequence[str],
    maps_path: Path,
    expected_maps_sha256: str,
) -> tuple[list[str], dict[str, Any]]:
    """Require the same verified map and target universe for all variants."""

    if set(offsets) != set(VARIANTS) or set(targets) != set(VARIANTS):
        raise FutureValueTierListError("four-way prediction set is incomplete")
    game_sets = {variant: set(offsets[variant]) for variant in VARIANTS}
    reference = game_sets[REFERENCE_VARIANT]
    if not reference or any(game_sets[variant] != reference for variant in VARIANTS):
        raise FutureValueTierListError("four-way prediction game sets differ")
    if any(set(targets[variant]) != reference for variant in VARIANTS):
        raise FutureValueTierListError("four-way target game sets differ")
    accepted = set(canonical_game_ids(accepted_game_ids))
    if not reference.issubset(accepted):
        raise FutureValueTierListError("prediction universe is outside the accepted census")
    expected_maps = _require_hash(expected_maps_sha256, "maps source hash")
    if not maps_path.is_file() or maps_path.is_symlink() or sha256_path(maps_path) != expected_maps:
        raise FutureValueTierListError("maps source bytes changed")
    frame = pd.read_parquet(maps_path, columns=["game_uid", "y_blue_win"])
    frame["game_uid"] = frame["game_uid"].astype(str)
    scoped = frame[frame["game_uid"].isin(reference)].copy()
    if scoped["game_uid"].duplicated().any() or set(scoped["game_uid"]) != reference:
        raise FutureValueTierListError("maps source identities are incomplete or duplicate")
    actual = scoped.set_index("game_uid")["y_blue_win"].astype(float).to_dict()
    for game_id in reference:
        values = {float(targets[variant][game_id]) for variant in VARIANTS}
        if len(values) != 1 or values != {float(actual[game_id])}:
            raise FutureValueTierListError(f"four-way target changed for {game_id}")
    ordered = sorted(reference)
    return ordered, {
        "game_count": len(ordered),
        "game_identity_sha256": identity_sha256(ordered),
        "maps_source_sha256": expected_maps,
        "target_rows_sha256": sha256_bytes(
            canonical_json_bytes(
                [{"game_id": game_id, "target": int(actual[game_id])} for game_id in ordered]
            )
        ),
    }


def make_offset_provenance(
    *,
    variant: str,
    offsets: Mapping[str, float],
    source_receipt_sha256: str,
) -> dict[str, Any]:
    """Seal the exact selected-map offset vector for the pooled builder."""

    ordered_ids = sorted(offsets)
    payload: dict[str, Any] = {
        "schema_version": PRE_MAP_OFFSET_PROVENANCE_SCHEMA,
        "status": "research_only",
        "authority": False,
        "producer": f"future_value_rating:{variant}",
        "timing": "strict_prior_pre_map",
        "source_receipt_sha256": _require_hash(source_receipt_sha256, "source receipt hash"),
        "source_identity_sha256": identity_sha256(ordered_ids),
        "source_game_count": len(ordered_ids),
        "offsets_sha256": offset_values_sha256(offsets),
    }
    payload["receipt_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def _candidate_rows(candidate: Mapping[str, Any]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    output: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for cell in candidate.get("cells", []):
        if not isinstance(cell, Mapping):
            continue
        scope_id = str(cell.get("scope_id") or "")
        role = str(cell.get("role") or "")
        patches = cell.get("patches")
        if not scope_id or not role or not isinstance(patches, list) or len(patches) != 1:
            raise FutureValueTierListError("candidate cell identity is invalid")
        patch = str(patches[0])
        for row in cell.get("rows", []):
            if not isinstance(row, Mapping):
                raise FutureValueTierListError("candidate Tier row is invalid")
            champion_id = str(row.get("champion_id") or "")
            if not champion_id:
                raise FutureValueTierListError("candidate champion identity is missing")
            key = (scope_id, patch, role, champion_id)
            if key in output:
                raise FutureValueTierListError("candidate Tier identity is duplicate")
            output[key] = {
                "champion": str(row.get("champion") or champion_id),
                **{field: row.get(field) for field in ROW_VALUE_FIELDS},
            }
    if not output:
        raise FutureValueTierListError("candidate has no Tier rows")
    return output


def validate_candidate(
    candidate: Mapping[str, Any],
    *,
    variant: str,
    universe: Mapping[str, Any],
    expected_source_receipt_sha256: str,
    expected_offsets_sha256: str,
    expected_producer: str,
) -> dict[str, Any]:
    """Validate one pooled candidate independently of the worker result.

    The worker output is treated as untrusted data.  The self hash, source
    universe, offset coverage, and offset provenance must agree with the
    verified frozen inputs before a candidate enters the four-way diff.
    """

    if variant not in VARIANTS:
        raise FutureValueTierListError("unknown candidate variant")
    if not isinstance(candidate, Mapping):
        raise FutureValueTierListError(f"{variant} candidate is not an object")
    if candidate.get("schema_version") != "scryglass:champion-role-elo-candidate:v2":
        raise FutureValueTierListError(f"{variant} candidate schema changed")
    if candidate.get("status") != "development_only" or candidate.get("development_only") is not True:
        raise FutureValueTierListError(f"{variant} candidate is not development-only")
    if candidate.get("publication_eligible") is not False or candidate.get("production_eligible") is not False:
        raise FutureValueTierListError(f"{variant} candidate authority changed")
    claimed_artifact = _require_hash(candidate.get("artifact_sha256"), f"{variant} candidate artifact hash")
    unsigned = {key: value for key, value in candidate.items() if key != "artifact_sha256"}
    if sha256_bytes(canonical_json_bytes(unsigned)) != claimed_artifact:
        raise FutureValueTierListError(f"{variant} candidate artifact bytes changed")
    expected_game_count = universe.get("game_count")
    expected_game_identity = universe.get("game_identity_sha256")
    if (
        isinstance(expected_game_count, bool)
        or not isinstance(expected_game_count, int)
        or expected_game_count <= 0
        or _require_hash(expected_game_identity, f"{variant} universe identity") is None
    ):
        raise FutureValueTierListError("candidate universe binding is invalid")
    candidate_source = candidate.get("source")
    if not isinstance(candidate_source, Mapping):
        raise FutureValueTierListError(f"{variant} candidate source is missing")
    for field in ("maps_replayed", "maps_used_in_joint_likelihood"):
        if candidate_source.get(field) != expected_game_count:
            raise FutureValueTierListError(f"{variant} candidate {field} changed")
    if candidate_source.get("source_identity_sha256") != expected_game_identity:
        raise FutureValueTierListError(f"{variant} candidate source identity changed")
    expected_offsets = _require_hash(expected_offsets_sha256, f"{variant} expected offsets hash")
    if expected_producer != f"future_value_rating:{variant}":
        raise FutureValueTierListError(f"{variant} expected producer is not trusted")
    override = candidate.get("pre_map_offset_override")
    if not isinstance(override, Mapping) or override.get("applied") is not True:
        raise FutureValueTierListError(f"{variant} candidate offset override is missing")
    if override.get("game_count") != expected_game_count or override.get("game_identity_sha256") != expected_game_identity:
        raise FutureValueTierListError(f"{variant} candidate offset coverage changed")
    offsets_hash = _require_hash(override.get("offsets_sha256"), f"{variant} candidate offsets hash")
    provenance = override.get("provenance")
    if not isinstance(provenance, Mapping):
        raise FutureValueTierListError(f"{variant} candidate offset provenance is missing")
    if set(provenance) != {
        "schema_version",
        "status",
        "authority",
        "producer",
        "timing",
        "source_receipt_sha256",
        "source_identity_sha256",
        "source_game_count",
        "offsets_sha256",
        "receipt_sha256",
    }:
        raise FutureValueTierListError(f"{variant} candidate offset provenance schema changed")
    if provenance.get("schema_version") != PRE_MAP_OFFSET_PROVENANCE_SCHEMA or provenance.get("status") != "research_only":
        raise FutureValueTierListError(f"{variant} candidate offset provenance status changed")
    if provenance.get("authority") is not False or provenance.get("timing") != "strict_prior_pre_map":
        raise FutureValueTierListError(f"{variant} candidate offset timing changed")
    if provenance.get("producer") != expected_producer:
        raise FutureValueTierListError(f"{variant} candidate offset producer changed")
    if provenance.get("source_receipt_sha256") != _require_hash(
        expected_source_receipt_sha256, f"{variant} expected source receipt"
    ):
        raise FutureValueTierListError(f"{variant} candidate offset source receipt changed")
    if provenance.get("source_identity_sha256") != expected_game_identity or provenance.get("source_game_count") != expected_game_count:
        raise FutureValueTierListError(f"{variant} candidate offset source census changed")
    if expected_offsets != offsets_hash:
        raise FutureValueTierListError(f"{variant} candidate offset values differ from the verified model")
    if provenance.get("offsets_sha256") != offsets_hash or override.get("offsets_sha256") != offsets_hash:
        raise FutureValueTierListError(f"{variant} candidate offset values changed")
    claimed_receipt = _require_hash(provenance.get("receipt_sha256"), f"{variant} offset receipt hash")
    if sha256_bytes(canonical_json_bytes({key: value for key, value in provenance.items() if key != "receipt_sha256"})) != claimed_receipt:
        raise FutureValueTierListError(f"{variant} candidate offset receipt bytes changed")
    return {
        "variant": variant,
        "artifact_sha256": claimed_artifact,
        "source_identity_sha256": str(expected_game_identity),
        "game_count": expected_game_count,
        "offsets_sha256": offsets_hash,
        "producer": expected_producer,
    }


def _numeric_delta(candidate: object, reference: object) -> float | None:
    if candidate is None or reference is None:
        return None
    try:
        value = float(candidate) - float(reference)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _rank_slice(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "changed_rank_count": 0,
            "changed_tier_count": 0,
            "mean_absolute_rank_movement": None,
            "maximum_absolute_rank_movement": None,
        }
    movements = [int(row["delta"]["rank_delta"]) for row in rows]
    return {
        "row_count": len(rows),
        "changed_rank_count": sum(value != 0 for value in movements),
        "changed_tier_count": sum(bool(row["delta"]["tier_changed"]) for row in rows),
        "mean_absolute_rank_movement": float(np.mean(np.abs(movements))),
        "maximum_absolute_rank_movement": int(max(map(abs, movements))),
    }


def _cell_rank_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = row["key"]
        grouped.setdefault((key["scope_id"], key["patch"], key["role"]), []).append(row)
    weighted_correlation = 0.0
    weighted_rows = 0
    inversions = 0
    comparable_pairs = 0
    overlap_totals = {5: 0.0, 10: 0.0, 25: 0.0}
    overlap_cells = {5: 0, 10: 0, 25: 0}
    for cell_rows in grouped.values():
        reference_ranks = np.asarray([float(row["reference"]["rank"]) for row in cell_rows])
        selected_ranks = np.asarray([float(row["selected"]["rank"]) for row in cell_rows])
        if len(cell_rows) > 1:
            correlation = float(np.corrcoef(reference_ranks, selected_ranks)[0, 1])
            weighted_correlation += correlation * len(cell_rows)
            weighted_rows += len(cell_rows)
            for left in range(len(cell_rows)):
                for right in range(left + 1, len(cell_rows)):
                    comparable_pairs += 1
                    if (
                        (reference_ranks[left] - reference_ranks[right])
                        * (selected_ranks[left] - selected_ranks[right])
                        < 0.0
                    ):
                        inversions += 1
        for requested in overlap_totals:
            size = min(requested, len(cell_rows))
            if size == 0:
                continue
            reference_top = {
                row["key"]["champion_id"]
                for row in sorted(cell_rows, key=lambda item: int(item["reference"]["rank"]))[:size]
            }
            selected_top = {
                row["key"]["champion_id"]
                for row in sorted(cell_rows, key=lambda item: int(item["selected"]["rank"]))[:size]
            }
            overlap_totals[requested] += len(reference_top & selected_top) / size
            overlap_cells[requested] += 1
    return {
        "cell_count": len(grouped),
        "weighted_cell_rank_correlation": (
            weighted_correlation / weighted_rows if weighted_rows else None
        ),
        "pairwise_inversions": inversions,
        "comparable_rank_pairs": comparable_pairs,
        "pairwise_inversion_rate": inversions / comparable_pairs if comparable_pairs else None,
        "mean_cell_top_overlap": {
            str(size): overlap_totals[size] / overlap_cells[size]
            if overlap_cells[size]
            else None
            for size in overlap_totals
        },
    }


def build_fourway_diff(
    candidates: Mapping[str, Mapping[str, Any]],
    *,
    source: Mapping[str, Any],
    universe: Mapping[str, Any],
    model_bindings: Mapping[str, Mapping[str, Any]],
    trust_manifest_raw_sha256: str,
    baseline_candidate_raw_sha256: str,
) -> dict[str, Any]:
    """Build exact common-universe Tier rank and value differences."""

    if set(candidates) != set(VARIANTS):
        raise FutureValueTierListError("four Tier candidates are required")
    if "source_identity_sha256" not in source or "source_receipt_sha256" not in source:
        raise FutureValueTierListError("four-way source binding is incomplete")
    for variant in VARIANTS:
        binding = model_bindings.get(variant)
        if not isinstance(binding, Mapping):
            raise FutureValueTierListError(f"{variant} model binding is missing")
        if "offsets_sha256" not in binding or "producer" not in binding:
            raise FutureValueTierListError(f"{variant} model offset binding is incomplete")
        chronology = binding.get("chronology")
        if chronology is not None:
            if not isinstance(chronology, Mapping):
                raise FutureValueTierListError(f"{variant} chronology binding is invalid")
            chronology_status = chronology.get("status")
            chronology_blockers = chronology.get("blockers")
            if chronology_status == "blocked":
                if not isinstance(chronology_blockers, list) or not chronology_blockers:
                    raise FutureValueTierListError(
                        f"{variant} blocked chronology has no explicit blocker"
                    )
                if not all(isinstance(value, str) and value for value in chronology_blockers):
                    raise FutureValueTierListError(f"{variant} chronology blockers are invalid")
            elif chronology_status != "verified":
                raise FutureValueTierListError(f"{variant} chronology is not verified")
        validate_candidate(
            candidates[variant],
            variant=variant,
            universe=universe,
            expected_source_receipt_sha256=str(source["source_receipt_sha256"]),
            expected_offsets_sha256=str(binding["offsets_sha256"]),
            expected_producer=str(binding["producer"]),
        )
    rows_by_variant = {variant: _candidate_rows(candidates[variant]) for variant in VARIANTS}
    reference_keys = set(rows_by_variant[REFERENCE_VARIANT])
    if any(set(rows_by_variant[variant]) != reference_keys for variant in VARIANTS):
        raise FutureValueTierListError("four-way Tier row universes differ")
    ordered_keys = sorted(reference_keys)
    common_identity = sha256_bytes(
        canonical_json_bytes(
            [
                {"scope_id": key[0], "patch": key[1], "role": key[2], "champion_id": key[3]}
                for key in ordered_keys
            ]
        )
    )
    comparisons: dict[str, Any] = {}
    all_rows: dict[str, list[dict[str, Any]]] = {}
    inherited_blockers = {
        blocker
        for binding in model_bindings.values()
        for blocker in binding.get("blockers", [])
    }
    for binding in model_bindings.values():
        chronology = binding.get("chronology")
        if isinstance(chronology, Mapping):
            inherited_blockers.update(chronology.get("blockers", []))
    for variant in VARIANTS:
        paired_rows: list[dict[str, Any]] = []
        rank_moves: list[int] = []
        transitions: Counter[str] = Counter()
        changed_cells: set[tuple[str, str, str]] = set()
        for key in ordered_keys:
            reference = rows_by_variant[REFERENCE_VARIANT][key]
            selected = rows_by_variant[variant][key]
            try:
                rank_delta = int(reference["rank"]) - int(selected["rank"])
            except (TypeError, ValueError) as error:
                raise FutureValueTierListError("Tier rank is missing or invalid") from error
            tier_changed = reference["tier_bucket"] != selected["tier_bucket"]
            if rank_delta or tier_changed:
                changed_cells.add(key[:3])
            rank_moves.append(rank_delta)
            transitions[f"{reference['tier_bucket']} -> {selected['tier_bucket']}"] += 1
            delta = {
                field: _numeric_delta(selected[field], reference[field])
                for field in NUMERIC_DELTA_FIELDS
            }
            delta.update({"rank_delta": rank_delta, "tier_changed": tier_changed})
            paired_rows.append(
                {
                    "key": {
                        "scope_id": key[0],
                        "patch": key[1],
                        "role": key[2],
                        "champion_id": key[3],
                        "champion": selected["champion"],
                    },
                    "reference": reference,
                    "selected": selected,
                    "delta": delta,
                }
            )
        rank_reference = np.asarray(
            [float(rows_by_variant[REFERENCE_VARIANT][key]["rank"]) for key in ordered_keys]
        )
        rank_selected = np.asarray([float(rows_by_variant[variant][key]["rank"]) for key in ordered_keys])
        rank_correlation = float(np.corrcoef(rank_reference, rank_selected)[0, 1])
        top_movers = sorted(
            [
                row
                for row in paired_rows
                if int(row["delta"]["rank_delta"]) != 0
                or bool(row["delta"]["tier_changed"])
            ],
            key=lambda row: (
                -abs(int(row["delta"]["rank_delta"])),
                -int(row["delta"]["rank_delta"]),
                row["key"]["scope_id"],
                row["key"]["role"],
                row["key"]["champion_id"],
            ),
        )[:25]
        row_digest = sha256_bytes(canonical_json_bytes(paired_rows))
        latest_patch = max(key[1] for key in ordered_keys)
        support_slices = {
            str(minimum): _rank_slice(
                [
                    row
                    for row in paired_rows
                    if int(row["reference"]["played_maps"] or 0) >= minimum
                ]
            )
            for minimum in (1, 3, 5, 10, 20)
        }
        latest_rows = [row for row in paired_rows if row["key"]["patch"] == latest_patch]
        numeric_summaries: dict[str, Any] = {}
        for field in NUMERIC_DELTA_FIELDS:
            values = [
                float(row["delta"][field])
                for row in paired_rows
                if row["delta"][field] is not None
            ]
            numeric_summaries[field] = {
                "finite_count": len(values),
                "changed_count": sum(value != 0.0 for value in values),
                "mean_delta": float(np.mean(values)) if values else None,
                "mean_absolute_delta": float(np.mean(np.abs(values))) if values else None,
                "maximum_absolute_delta": float(max(map(abs, values))) if values else None,
            }
        comparisons[variant] = {
            "reference_variant": REFERENCE_VARIANT,
            "row_count": len(paired_rows),
            "changed_rank_count": sum(move != 0 for move in rank_moves),
            "changed_tier_count": sum(row["delta"]["tier_changed"] for row in paired_rows),
            "changed_cell_count": len(changed_cells),
            "mean_absolute_rank_movement": float(np.mean(np.abs(rank_moves))),
            "maximum_absolute_rank_movement": int(max(map(abs, rank_moves), default=0)),
            "rank_correlation": rank_correlation,
            "cell_rank_metrics": _cell_rank_metrics(paired_rows),
            "latest_patch": latest_patch,
            "latest_patch_metrics": _rank_slice(latest_rows),
            "support_slices_by_minimum_played_maps": support_slices,
            "numeric_delta_summaries": numeric_summaries,
            "tier_transition_matrix": dict(sorted(transitions.items())),
            "paired_row_digest_sha256": row_digest,
            "top_movers": top_movers,
        }
        all_rows[variant] = paired_rows
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "research_only",
        "authority": dict(AUTHORITY),
        "source": dict(source),
        "trust_manifest_raw_sha256": _require_hash(
            trust_manifest_raw_sha256, "trust manifest raw hash"
        ),
        "baseline_public_candidate": {
            "status": "unchanged_non_comparable_full_census_reference",
            "raw_sha256": _require_hash(
                baseline_candidate_raw_sha256, "baseline candidate hash"
            ),
        },
        "evaluation_universe": {
            **dict(universe),
            "rank_universe": "common_verified_finite_ids_per_patch_role_cell",
            "row_identity_fields": ["scope_id", "patch", "role", "champion_id"],
            "common_row_count": len(ordered_keys),
            "common_identity_sha256": common_identity,
        },
        "model_bindings": {variant: dict(model_bindings[variant]) for variant in VARIANTS},
        "candidate_artifacts": {
            variant: {
                "artifact_sha256": candidates[variant].get("artifact_sha256"),
                "source_identity_sha256": candidates[variant].get("source", {}).get(
                    "source_identity_sha256"
                ),
                "pre_map_offset_override": candidates[variant].get("pre_map_offset_override"),
            }
            for variant in VARIANTS
        },
        "comparisons": comparisons,
        "rows": all_rows,
        "blockers": sorted(
            inherited_blockers
            | {
                "authoritative_series_id_missing_proxy_cluster_used",
                "tierlist_shadow_common_validation_subset_not_public_candidate",
                "tierlist_offset_model_includes_roster_and_composition_context_new_estimand",
                "public_tierlist_authority_false",
            }
        ),
    }
    report["report_sha256"] = sha256_bytes(canonical_json_bytes(report))
    return report


__all__ = [
    "AUTHORITY",
    "FutureValueTierListError",
    "REFERENCE_VARIANT",
    "PINNED_TRUST_MANIFEST_RAW_SHA256",
    "SCHEMA_VERSION",
    "TRUST_SCHEMA_VERSION",
    "VARIANTS",
    "build_fourway_diff",
    "canonical_json_bytes",
    "load_prediction_offsets",
    "load_trust_manifest",
    "make_offset_provenance",
    "sha256_bytes",
    "sha256_path",
    "validate_candidate",
    "validate_common_prediction_universe",
]
