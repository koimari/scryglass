"""Research contract and accepted-source bridge for future value ratings.

This module keeps the public descriptive ratings unchanged.  It prepares one
accepted Oracle's Elixir census for a development model that estimates future
player and team value from information available before each map.

The full accepted census remains in the source receipt.  A narrower model
population can exclude maps whose stable player or team identity is missing.
Those exclusions are explicit and content bound.  They are never filled with
names, zeroes, or inferred identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import hashlib
import json
import math
import re
from types import MappingProxyType
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from scipy.optimize import minimize

from lol_kills.etl.source_keys import canonical_source_game_key
from lol_kills.etl.aliases import normalize_team
from lol_kills.ratings.global_player_bt import (
    ANCHOR_METRIC_Z_CLIP,
    PrefixBaselineCache,
    _contribution_metrics,
    _role_normalized_composite,
)
from lol_kills.research.future_phase_curve import (
    MODEL_VERSION as PHASE_CURVE_PRODUCER_VERSION,
    PHASE_FEATURE_DECLARATION,
    PHASE_SHAPE_FEATURES,
    PHASE_SHAPE_INVARIANT_FEATURES,
    PHASE_SHAPE_SIGNED_FEATURES,
    SCHEMA_VERSION as PHASE_CURVE_SCHEMA_VERSION,
)
from lol_kills.v2.tierlists.accepted_census import (
    canonical_game_ids,
    identity_sha256,
    load_census,
)


SCHEMA_VERSION = "scryglass:future-value-rating-source:v1"
MODEL_CONTRACT_VERSION = "scryglass:future-value-rating-contract:v1"
MODEL_FIT_SCHEMA_VERSION = "scryglass:future-value-model-fit:v1"
TIME_DECAY_HALF_LIFE_DAYS = 120.0
RANK_3 = 3
SIDE_SWAP_MEAN_TOLERANCE = 1e-12
SIDE_SWAP_MAX_TOLERANCE = 1e-12
REGULARIZATION_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
# Variant ledgers are external producer outputs.  A nested selector cannot
# reuse an outer ledger.  Until a separately bound inner ledger is supplied,
# every explicit variant uses this predeclared value and records the blocker.
PREDECLARED_VARIANT_REGULARIZATION_C = 0.1
FORM_METRICS = (
    "cs_per_min",
    "gold_per_min",
    "gold_share_pct",
    "damage_per_min",
    "damage_share_pct",
    "kda_role_weighted",
    "wards_per_min",
    "wards_cleared_per_min",
)
CHECKPOINTS = (10, 15, 20, 25)
ROLES = ("top", "jungle", "mid", "bot", "support")
SIDES = ("blue", "red")

SOURCE_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "source_as_of",
        "source_game_count",
        "source_identity_sha256",
        "accepted_game_ids",
        "model_eligible_game_count",
        "model_eligible_identity_sha256",
        "model_eligible_game_ids",
        "source_rows",
        "source_extra_game_ids",
        "identity_coverage",
        "checkpoint_coverage",
        "model_exclusions",
        "source_files",
        "model_contract",
        "authority",
        "receipt_sha256",
    }
)
SOURCE_RECEIPT_AUTHORITY = MappingProxyType(
    {
        "research_only": True,
        "public_player_rating": False,
        "public_team_rating": False,
        "public_probability": False,
        "promotion": False,
        "merge": False,
        "deployment": False,
    }
)
SOURCE_RECEIPT_REQUIRED_FILES = frozenset(
    {"maps", "players", "teams", "accepted_census"}
)
SOURCE_FILE_RECORD_FIELDS = frozenset(
    {"path", "locator", "bytes", "sha256", "year"}
)
LEAGUEPEDIA_CROSSWALK_RECEIPT_SCHEMA_VERSION = (
    "scryglass:verified-oe-leaguepedia-series-crosswalk-receipt:v1"
)
LEAGUEPEDIA_CROSSWALK_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "authority",
        "artifact",
        "crosswalk_sha256",
        "source_receipt_sha256",
        "source_identity_sha256",
        "accepted_game_count",
        "accepted_game_identity_sha256",
        "assignment_count",
        "assignment_sha256",
        "mapped_game_count",
        "mapped_game_identity_sha256",
        "mapped_game_ids",
        "receipt_sha256",
    }
)
LEAGUEPEDIA_CROSSWALK_RECEIPT_AUTHORITY = MappingProxyType(
    {
        "research_only": True,
        "public": False,
        "authoritative_series": False,
        "promotion": False,
        "deployment": False,
    }
)
LEAGUEPEDIA_CROSSWALK_SOURCE = "mixed:leaguepedia_crosswalk+conservative_series_superset"


class RatingVariant(str, Enum):
    """The four frozen feature contracts for future-value development."""

    CURRENT_ONLY = "current_only"
    FUTURE_PLAYER_FORM = "future_player_form"
    SCALING_CURVE = "scaling_curve"
    BOTH = "both"


RATING_VARIANT_ORDER = (
    RatingVariant.CURRENT_ONLY,
    RatingVariant.FUTURE_PLAYER_FORM,
    RatingVariant.SCALING_CURVE,
    RatingVariant.BOTH,
)
RATING_VARIANT_ORDINALS = MappingProxyType(
    {variant: ordinal for ordinal, variant in enumerate(RATING_VARIANT_ORDER, 1)}
)


FORBIDDEN_PREGAME_PATTERNS = (
    re.compile(r"^(?:target|observed|current)_", re.IGNORECASE),
    re.compile(r"(?:^|_)(?:gold|xp|cs|kills|assists|deaths)at(?:10|15|20|25)$", re.IGNORECASE),
    re.compile(r"(?:^|_)(?:result|winner|blue_win|y_blue_win)$", re.IGNORECASE),
)


class FutureValueSourceError(ValueError):
    """The accepted source cannot support a fail-closed future-value build."""


@dataclass(frozen=True)
class AcceptedFutureValueSource:
    """Accepted source frames and their exact research receipt."""

    maps: pd.DataFrame
    players: pd.DataFrame
    teams: pd.DataFrame
    eligible_game_ids: tuple[str, ...]
    receipt: Mapping[str, Any]


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FutureValueSourceError("receipt contains a non-canonical value") from error


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_timestamp(value: Any, field: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise FutureValueSourceError(f"{field} is not a timestamp") from error
    if pd.isna(stamp):
        raise FutureValueSourceError(f"{field} is missing")
    if stamp.tzinfo is None:
        raise FutureValueSourceError(f"{field} must include a timezone")
    return stamp.tz_convert("UTC")


def _utc_text(value: Any) -> str:
    return _utc_timestamp(value, "timestamp").isoformat().replace("+00:00", "Z")


def _frame_game_ids(frame: pd.DataFrame, label: str) -> pd.Series:
    if "game_uid" in frame.columns:
        fallback = frame["gameid"] if "gameid" in frame.columns else None
        values = [
            canonical_source_game_key(
                value,
                fallback.loc[index] if fallback is not None else None,
            )
            for index, value in frame["game_uid"].items()
        ]
    elif "gameid" in frame.columns:
        values = [canonical_source_game_key(value) for value in frame["gameid"]]
    elif "game_id" in frame.columns:
        values = [canonical_source_game_key(value) for value in frame["game_id"]]
    else:
        raise FutureValueSourceError(f"{label} has no game identity column")
    result = pd.Series(values, index=frame.index, dtype="string")
    if result.isna().any() or result.str.strip().eq("").any():
        raise FutureValueSourceError(f"{label} contains an empty canonical game identity")
    return result


def _validate_verified_source_receipt(
    receipt: Mapping[str, Any] | None,
    map_frame: pd.DataFrame,
    *,
    require_full_eligible_set: bool,
    train_game_ids: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Validate the source receipt before using a fit or evaluation frame."""

    accepted_ids, eligible_ids = validate_future_value_source_receipt_payload(receipt)
    map_ids = tuple(sorted(map_frame["game_id"].astype(str)))
    eligible_set = set(eligible_ids)
    if not set(map_ids).issubset(eligible_set):
        raise FutureValueSourceError("model frame contains game IDs outside the eligible census")
    if require_full_eligible_set and set(map_ids) != eligible_set:
        raise FutureValueSourceError("model frame does not match the eligible census exactly")
    source_cutoff = _utc_timestamp(receipt["source_as_of"], "source_as_of")
    if map_frame["date"].gt(source_cutoff).any():
        raise FutureValueSourceError("model frame contains rows after source_as_of")
    if train_game_ids is not None:
        train_ids = tuple(sorted({str(value) for value in train_game_ids}))
        if not set(train_ids).issubset(set(map_ids)):
            raise FutureValueSourceError("fit IDs are outside the verified model frame")
        return train_ids
    return eligible_ids


def validate_future_value_source_receipt_payload(
    receipt: Mapping[str, Any] | None,
    *,
    expected_receipt_sha256: str | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate the complete canonical research source receipt.

    A receipt hash can be supplied by a frozen manifest.  Source records that
    contain durable paths are checked against the bytes at those paths.  A
    locator-only record remains a provenance claim until its source root is
    available to the caller.
    """

    if not isinstance(receipt, Mapping):
        raise FutureValueSourceError("verified source receipt is required")
    if set(receipt) != set(SOURCE_RECEIPT_FIELDS):
        raise FutureValueSourceError("verified source receipt schema is not canonical")
    if receipt.get("schema_version") != SCHEMA_VERSION or receipt.get(
        "status"
    ) != "accepted_source_bound_development_only":
        raise FutureValueSourceError("verified source receipt status is invalid")
    if dict(receipt.get("authority") or {}) != dict(SOURCE_RECEIPT_AUTHORITY):
        raise FutureValueSourceError("verified source receipt authority is invalid")
    receipt_hash = receipt.get("receipt_sha256")
    if not isinstance(receipt_hash, str) or re.fullmatch(
        r"[0-9a-f]{64}", receipt_hash, re.I
    ) is None:
        raise FutureValueSourceError("verified source receipt hash is invalid")
    payload = dict(receipt)
    payload.pop("receipt_sha256", None)
    if hashlib.sha256(_canonical_json_bytes(payload)).hexdigest() != receipt_hash:
        raise FutureValueSourceError("verified source receipt hash does not match payload")
    if expected_receipt_sha256 is not None:
        expected_hash = str(expected_receipt_sha256).lower()
        if re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
            raise FutureValueSourceError("expected verified source receipt hash is invalid")
        if receipt_hash.lower() != expected_hash:
            raise FutureValueSourceError("verified source receipt hash differs from expected")
    accepted_ids = tuple(str(value) for value in receipt["accepted_game_ids"])
    eligible_ids = tuple(str(value) for value in receipt["model_eligible_game_ids"])
    if (
        not accepted_ids
        or tuple(canonical_game_ids(accepted_ids)) != accepted_ids
        or int(receipt["source_game_count"]) != len(accepted_ids)
        or str(receipt["source_identity_sha256"]) != identity_sha256(accepted_ids)
        or int(receipt["model_eligible_game_count"]) != len(eligible_ids)
        or tuple(canonical_game_ids(eligible_ids)) != eligible_ids
        or str(receipt["model_eligible_identity_sha256"]) != identity_sha256(eligible_ids)
        or not set(eligible_ids).issubset(set(accepted_ids))
    ):
        raise FutureValueSourceError("verified source receipt census identity is invalid")
    _utc_timestamp(receipt["source_as_of"], "source_as_of")
    source_files = receipt["source_files"]
    if not isinstance(source_files, Mapping) or not SOURCE_RECEIPT_REQUIRED_FILES.issubset(
        source_files
    ):
        raise FutureValueSourceError("verified source receipt file bindings are incomplete")
    for label, record in source_files.items():
        if (
            not isinstance(label, str)
            or not label.strip()
            or not isinstance(record, Mapping)
            or not set(record).issubset(SOURCE_FILE_RECORD_FIELDS)
            or not isinstance(record.get("bytes"), int)
            or isinstance(record.get("bytes"), bool)
            or int(record["bytes"]) <= 0
        ):
            raise FutureValueSourceError(f"verified source file record is invalid: {label}")
        locators = (record.get("path"), record.get("locator"))
        if sum(isinstance(value, str) and bool(value.strip()) for value in locators) != 1:
            raise FutureValueSourceError(f"verified source file locator is invalid: {label}")
        expected_file_hash = str(record.get("sha256") or "").lower()
        if re.fullmatch(r"[0-9a-f]{64}", expected_file_hash, re.I) is None:
            raise FutureValueSourceError(f"verified source file hash is invalid: {label}")
        if isinstance(record.get("path"), str) and record["path"].strip():
            path = _safe_file_path(record["path"], f"verified source file {label}")
            actual_bytes = path.stat().st_size
            if int(record["bytes"]) != actual_bytes:
                raise FutureValueSourceError(f"verified source file bytes changed: {label}")
            if _sha256_path(path) != expected_file_hash:
                raise FutureValueSourceError(f"verified source file hash changed: {label}")
        if "year" in record and (
            isinstance(record["year"], bool) or not isinstance(record["year"], int)
        ):
            raise FutureValueSourceError(f"verified source file year is invalid: {label}")
    structured = (
        "source_rows",
        "source_extra_game_ids",
        "identity_coverage",
        "checkpoint_coverage",
        "model_exclusions",
        "model_contract",
    )
    if any(not isinstance(receipt.get(field), Mapping) for field in structured):
        raise FutureValueSourceError("verified source receipt evidence is incomplete")
    return accepted_ids, eligible_ids


def _side(value: Any) -> str | None:
    text = str(value or "").strip().casefold()
    if text in {"blue", "b"}:
        return "blue"
    if text in {"red", "r"}:
        return "red"
    return None


def _role(value: Any) -> str | None:
    if value is None or bool(pd.isna(value)):
        return None
    text = str(value).strip().casefold()
    return {
        "top": "top",
        "jng": "jungle",
        "jg": "jungle",
        "jungle": "jungle",
        "mid": "mid",
        "middle": "mid",
        "bot": "bot",
        "adc": "bot",
        "carry": "bot",
        "sup": "support",
        "support": "support",
        "utility": "support",
    }.get(text)


def _stable_identity(value: Any, prefix: str) -> bool:
    return isinstance(value, str) and value.strip().startswith(prefix)


def _checkpoint_coverage(frame: pd.DataFrame, *, rows_per_map: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    game_count = int(frame["_game_id"].nunique())
    if "gamelength" in frame.columns:
        duration = pd.to_numeric(frame["gamelength"], errors="coerce").groupby(
            frame["_game_id"], sort=False
        ).max()
    else:
        duration = pd.Series(dtype=float)
    if "date" in frame.columns:
        year = (
            pd.to_datetime(frame["date"], utc=True, errors="coerce")
            .groupby(frame["_game_id"], sort=False)
            .first()
            .dt.year
        )
    else:
        year = pd.Series(dtype="Int64")
    for checkpoint in CHECKPOINTS:
        columns = [
            f"{stem}at{checkpoint}"
            for stem in ("gold", "xp", "cs", "kills", "assists", "deaths")
        ]
        missing_columns = [column for column in columns if column not in frame.columns]
        if missing_columns:
            result[str(checkpoint)] = {
                "complete_maps": 0,
                "coverage": 0.0,
                "missing_columns": missing_columns,
            }
            continue
        finite = frame[columns].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
        by_game = finite.groupby(frame["_game_id"], sort=False).sum()
        complete_mask = by_game.eq(rows_per_map)
        complete = int(complete_mask.sum())
        eligible_mask = duration.ge(checkpoint * 60).reindex(complete_mask.index, fill_value=False)
        eligible_count = int(eligible_mask.sum())
        eligible_complete = int((complete_mask & eligible_mask).sum())
        by_year: dict[str, Any] = {}
        for raw_year in sorted(int(value) for value in year.dropna().unique()):
            year_ids = set(year.index[year.eq(raw_year)])
            selected = complete_mask.index.to_series().isin(year_ids).to_numpy()
            selected_complete = complete_mask.iloc[selected]
            selected_eligible = eligible_mask.iloc[selected]
            by_year[str(raw_year)] = {
                "maps": int(selected.sum()),
                "complete_maps": int(selected_complete.sum()),
                "coverage": float(selected_complete.mean()) if selected.any() else 0.0,
                "duration_eligible_maps": int(selected_eligible.sum()),
                "duration_eligible_complete_maps": int(
                    (selected_complete & selected_eligible).sum()
                ),
                "duration_eligible_coverage": (
                    float(selected_complete[selected_eligible].mean())
                    if selected_eligible.any()
                    else None
                ),
            }
        result[str(checkpoint)] = {
            "complete_maps": complete,
            "coverage": complete / max(game_count, 1),
            "duration_eligible_maps": eligible_count,
            "duration_eligible_complete_maps": eligible_complete,
            "duration_eligible_coverage": (
                eligible_complete / eligible_count if eligible_count else None
            ),
            "by_year": by_year,
            "missing_columns": [],
        }
    return result


def _model_eligibility(players: pd.DataFrame, teams: pd.DataFrame) -> tuple[tuple[str, ...], dict[str, list[str]]]:
    reasons: dict[str, list[str]] = {}
    player_groups = {str(key): value for key, value in players.groupby("_game_id", sort=False)}
    team_groups = {str(key): value for key, value in teams.groupby("_game_id", sort=False)}
    for game_id in sorted(set(player_groups) | set(team_groups)):
        current: list[str] = []
        player_rows = player_groups.get(game_id)
        team_rows = team_groups.get(game_id)
        if player_rows is None or len(player_rows) != 10:
            current.append("player_row_count_not_10")
        if team_rows is None or len(team_rows) != 2:
            current.append("team_row_count_not_2")
        if player_rows is not None and len(player_rows) == 10:
            slots = {
                (_side(side), _role(role))
                for side, role in zip(player_rows["side"], player_rows["position"])
            }
            expected = {(side, role) for side in SIDES for role in ROLES}
            if slots != expected:
                current.append("player_role_or_side_closure_invalid")
            player_identity_column = "playerid" if "playerid" in player_rows.columns else None
            team_identity_column = "teamid" if "teamid" in player_rows.columns else None
            if player_identity_column is None or not player_rows[player_identity_column].map(
                lambda value: _stable_identity(value, "oe:player:")
            ).all():
                current.append("stable_player_identity_missing")
            elif player_rows[player_identity_column].astype(str).nunique() != 10:
                current.append("player_identity_not_unique")
            if team_identity_column is None or not player_rows[team_identity_column].map(
                lambda value: _stable_identity(value, "oe:team:")
            ).all():
                current.append("stable_team_identity_missing")
            elif (
                player_rows.assign(_side_key=player_rows["side"].map(_side))
                .groupby("_side_key", sort=False)[team_identity_column]
                .nunique()
                .reindex(SIDES, fill_value=0)
                .ne(1)
                .any()
                or player_rows[team_identity_column].astype(str).nunique() != 2
            ):
                current.append("player_team_identity_closure_invalid")
            if "champion" not in player_rows.columns or player_rows["champion"].isna().any():
                current.append("champion_identity_missing")
            elif player_rows["champion"].astype(str).str.strip().replace("", pd.NA).nunique() != 10:
                current.append("champion_identity_not_unique")
        if team_rows is not None and len(team_rows) == 2:
            sides = {_side(value) for value in team_rows["side"]}
            if sides != set(SIDES):
                current.append("team_side_closure_invalid")
            if "teamid" not in team_rows.columns or not team_rows["teamid"].map(
                lambda value: _stable_identity(value, "oe:team:")
            ).all():
                current.append("stable_team_row_identity_missing")
            elif team_rows["teamid"].astype(str).nunique() != 2:
                current.append("team_row_identity_not_unique")
            elif player_rows is not None and len(player_rows) == 10:
                player_team_by_side = {
                    _side(side): str(team_id)
                    for side, team_id in zip(player_rows["side"], player_rows["teamid"])
                }
                team_row_by_side = {
                    _side(side): str(team_id)
                    for side, team_id in zip(team_rows["side"], team_rows["teamid"])
                }
                if player_team_by_side != team_row_by_side:
                    current.append("player_team_row_identity_mismatch")
        if current:
            reasons[game_id] = sorted(set(current))
    eligible = tuple(sorted(set(player_groups) & set(team_groups) - set(reasons)))
    return eligible, reasons


def assert_pregame_feature_names(feature_names: Iterable[str]) -> None:
    """Reject current-map state and outcome fields from a pregame design."""

    forbidden = sorted(
        {
            str(name)
            for name in feature_names
            if any(pattern.search(str(name)) for pattern in FORBIDDEN_PREGAME_PATTERNS)
        }
    )
    if forbidden:
        raise FutureValueSourceError(
            "pregame feature set contains current-map information: " + ", ".join(forbidden)
        )


def _strict_prior_block_mean(
    values: pd.DataFrame,
    *,
    entity_column: str,
    date_column: str,
    metric_columns: Sequence[str],
) -> pd.DataFrame:
    """Return expanding means from earlier timestamp blocks only.

    Rows for one entity at the same timestamp share the same prior state.  A
    later append therefore cannot change an earlier output row.
    """

    required = {entity_column, date_column, *metric_columns}
    missing = sorted(required - set(values.columns))
    if missing:
        raise FutureValueSourceError(
            "strict-prior form input is missing columns: " + ", ".join(missing)
        )
    work = values[[entity_column, date_column, *metric_columns]].copy()
    work[date_column] = pd.to_datetime(work[date_column], utc=True, errors="coerce")
    if work[entity_column].isna().any() or work[date_column].isna().any():
        raise FutureValueSourceError("strict-prior form identity or date is missing")
    keys = [entity_column, date_column]
    output = work[keys].copy()
    for metric in metric_columns:
        numeric = pd.to_numeric(work[metric], errors="coerce").astype(float)
        numeric = numeric.where(np.isfinite(numeric))
        metric_rows = work[keys].copy()
        metric_rows["_value"] = numeric
        blocks = (
            metric_rows.groupby(keys, sort=False, observed=True)["_value"]
            .agg(["sum", "count"])
            .reset_index()
            .sort_values(keys, kind="stable")
        )
        blocks["_prior_sum"] = (
            blocks.groupby(entity_column, sort=False, observed=True)["sum"].cumsum()
            - blocks["sum"]
        )
        blocks["_prior_count"] = (
            blocks.groupby(entity_column, sort=False, observed=True)["count"].cumsum()
            - blocks["count"]
        )
        blocks[f"prior_form_{metric}"] = blocks["_prior_sum"] / blocks[
            "_prior_count"
        ].where(blocks["_prior_count"] > 0)
        blocks[f"prior_form_{metric}_support"] = blocks["_prior_count"].astype(int)
        output = output.merge(
            blocks[
                [
                    *keys,
                    f"prior_form_{metric}",
                    f"prior_form_{metric}_support",
                ]
            ],
            on=keys,
            how="left",
            validate="many_to_one",
        )
    if len(output) != len(values):
        raise FutureValueSourceError("strict-prior form changed row grain")
    return output


def build_strict_prior_player_form(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    *,
    baseline_cache: PrefixBaselineCache | None = None,
) -> pd.DataFrame:
    """Build fitted-weight inputs from prior player performance only.

    The robust role and competition-tier normalization is shared with the
    descriptive global player anchor.  This function keeps each normalized
    metric separate.  The future-value model must fit their weights inside
    chronological development folds.
    """

    map_frame = maps.copy()
    map_frame["_game_id"] = _frame_game_ids(map_frame, "maps")
    map_frame["_date"] = pd.to_datetime(
        map_frame.get("date"), utc=True, errors="coerce"
    )
    if map_frame["_game_id"].duplicated().any() or map_frame["_date"].isna().any():
        raise FutureValueSourceError("player form map identity or date is invalid")
    map_dates = pd.Series(
        map_frame["_date"].dt.tz_localize(None).to_numpy(),
        index=pd.Index(map_frame["_game_id"].astype(str), name="game_id"),
    )
    metrics = _contribution_metrics(players, map_dates)
    if metrics.empty:
        raise FutureValueSourceError("player contribution metrics are unavailable")
    _composite, normalization, diagnostics, normalized_z, prior_counts = (
        _role_normalized_composite(
            metrics,
            baseline_cache=baseline_cache,
            _return_components=True,
            _group_mode="role+competition_tier",
        )
    )
    if normalization != "role+competition_tier":
        raise FutureValueSourceError("player form normalization scope changed")
    player_identity = players.loc[metrics.index, "playerid"] if "playerid" in players else None
    if player_identity is None:
        raise FutureValueSourceError("stable player identity column is missing")
    stable = player_identity.map(lambda value: _stable_identity(value, "oe:player:"))
    base = metrics.loc[stable, ["_game_id", "_date", "_side", "_role"]].copy()
    base["player_id"] = player_identity.loc[stable].astype(str)
    metric_columns = list(normalized_z.columns)
    if not metric_columns:
        raise FutureValueSourceError("no normalized player form metrics are available")
    clipped = normalized_z.loc[stable, metric_columns].clip(
        lower=-ANCHOR_METRIC_Z_CLIP,
        upper=ANCHOR_METRIC_Z_CLIP,
    )
    form_input = pd.concat(
        [
            base[["player_id", "_date"]].reset_index(drop=True),
            clipped.reset_index(drop=True),
        ],
        axis=1,
    )
    prior = _strict_prior_block_mean(
        form_input,
        entity_column="player_id",
        date_column="_date",
        metric_columns=metric_columns,
    )
    output = pd.concat(
        [
            base.reset_index(drop=True),
            prior.drop(columns=["player_id", "_date"]).reset_index(drop=True),
        ],
        axis=1,
    ).rename(
        columns={
            "_game_id": "game_id",
            "_date": "date",
            "_side": "side",
            "_role": "role",
        }
    )
    feature_names = [
        column
        for column in output.columns
        if column.startswith("prior_form_") and not column.endswith("_support")
    ]
    assert_pregame_feature_names(feature_names)
    output.attrs["normalization"] = normalization
    output.attrs["baseline_diagnostics"] = diagnostics
    output.attrs["prior_count_diagnostics"] = {
        metric: {
            "maximum": int(pd.to_numeric(prior_counts.loc[stable, metric], errors="coerce").max()),
            "observed_rows": int(prior_counts.loc[stable, metric].notna().sum()),
        }
        for metric in metric_columns
    }
    return output


def _strict_prior_time_decay(
    values: pd.DataFrame,
    *,
    entity_column: str,
    date_column: str,
    metric_columns: Sequence[str],
    half_life_days: float,
) -> pd.DataFrame:
    """Return exponentially decayed values from earlier timestamp blocks.

    The state is updated after a complete timestamp block.  This gives every
    row at one timestamp the same prior state.  The running state uses a
    recurrence, so the work is linear in the number of rows.  Missing metric
    values do not enter the numerator or denominator.
    """

    if not math.isfinite(float(half_life_days)) or float(half_life_days) <= 0.0:
        raise FutureValueSourceError("time-decay half life must be positive")
    required = {entity_column, date_column, *metric_columns}
    missing = sorted(required - set(values.columns))
    if missing:
        raise FutureValueSourceError(
            "time-decayed form input is missing columns: " + ", ".join(missing)
        )
    work = values[[entity_column, date_column, *metric_columns]].copy()
    work[date_column] = pd.to_datetime(work[date_column], utc=True, errors="coerce")
    if work[entity_column].isna().any() or work[date_column].isna().any():
        raise FutureValueSourceError("time-decayed form identity or date is missing")
    work[entity_column] = work[entity_column].astype(str)
    original_index = work.index.copy()
    work = work.reset_index(drop=True)
    entity_values = work[entity_column].to_numpy(dtype=object)
    date_values = work[date_column].astype("int64").to_numpy(dtype=np.int64)
    metric_values = {
        metric: pd.to_numeric(work[metric], errors="coerce").to_numpy(dtype=float)
        for metric in metric_columns
    }
    result_values = {
        metric: np.full(len(work), np.nan, dtype=float) for metric in metric_columns
    }
    result_support = {
        metric: np.zeros(len(work), dtype=np.int64) for metric in metric_columns
    }
    result_effective = {
        metric: np.zeros(len(work), dtype=float) for metric in metric_columns
    }
    half_life = float(half_life_days)
    decay_rate = math.log(2.0) / half_life
    # Work on integer positions.  This preserves duplicate source indexes and
    # prevents a merge from changing the player-map grain.
    entity_codes, _ = pd.factorize(entity_values, sort=False)
    for entity_code in np.unique(entity_codes):
        positions = np.flatnonzero(entity_codes == entity_code).astype(np.int64)
        order = positions[np.argsort(date_values[positions], kind="stable")]
        dates = date_values[order]
        starts = np.empty(len(order), dtype=bool)
        starts[0] = True
        starts[1:] = dates[1:] != dates[:-1]
        block_starts = np.flatnonzero(starts)
        block_ends = np.concatenate((block_starts[1:], np.asarray([len(order)])))
        sums = np.zeros(len(metric_columns), dtype=float)
        effective = np.zeros(len(metric_columns), dtype=float)
        support = np.zeros(len(metric_columns), dtype=np.int64)
        previous_date: int | None = None
        for block_start, block_end in zip(block_starts, block_ends):
            current_date = int(dates[block_start])
            if previous_date is not None:
                elapsed_days = (current_date - previous_date) / 86_400_000_000_000.0
                factor = math.exp(-decay_rate * max(elapsed_days, 0.0))
                sums *= factor
                effective *= factor
            current_positions = order[block_start:block_end]
            for metric_index, metric in enumerate(metric_columns):
                numeric = metric_values[metric][current_positions]
                finite = np.isfinite(numeric)
                prior_value = (
                    sums[metric_index] / effective[metric_index]
                    if effective[metric_index] > 0.0
                    else np.nan
                )
                result_values[metric][current_positions] = prior_value
                result_support[metric][current_positions] = int(support[metric_index])
                result_effective[metric][current_positions] = float(effective[metric_index])
                if finite.any():
                    finite_values = numeric[finite]
                    sums[metric_index] += float(finite_values.sum())
                    effective[metric_index] += float(len(finite_values))
                    support[metric_index] += int(len(finite_values))
            previous_date = current_date
    output = pd.DataFrame(index=pd.RangeIndex(len(work)))
    for metric in metric_columns:
        output[f"prior_form_{metric}"] = result_values[metric]
        output[f"prior_form_{metric}_support"] = result_support[metric]
        output[f"prior_form_{metric}_effective_support"] = result_effective[metric]
    output.index = original_index
    return output


def build_time_decayed_prior_player_form(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    *,
    half_life_days: float = TIME_DECAY_HALF_LIFE_DAYS,
) -> pd.DataFrame:
    """Build strictly-prior, exponentially time-decayed player form.

    Final whole-map metrics enter only rows at later timestamps.  The source
    identity, role, champion, and team remain explicit in the returned frame.
    A missing metric stays missing with a zero support count.
    """

    map_frame = maps.copy()
    map_frame["_game_id"] = _frame_game_ids(map_frame, "maps")
    map_frame["_date"] = pd.to_datetime(map_frame.get("date"), utc=True, errors="coerce")
    if map_frame["_game_id"].duplicated().any() or map_frame["_date"].isna().any():
        raise FutureValueSourceError("player form map identity or date is invalid")
    map_dates = pd.Series(
        map_frame["_date"].dt.tz_localize(None).to_numpy(),
        index=pd.Index(map_frame["_game_id"].astype(str), name="game_id"),
    )
    metrics = _contribution_metrics(players, map_dates)
    if metrics.empty:
        raise FutureValueSourceError("player contribution metrics are unavailable")
    required_identity = {"playerid", "teamid", "champion"}
    if not required_identity.issubset(players.columns):
        raise FutureValueSourceError("player form identity columns are missing")
    player_ids = players.loc[metrics.index, "playerid"].astype("string")
    team_ids = players.loc[metrics.index, "teamid"].astype("string")
    champions = players.loc[metrics.index, "champion"].astype("string").str.strip()
    roles = metrics["_role"].map(_role)
    stable_player = player_ids.map(lambda value: _stable_identity(value, "oe:player:"))
    stable_team = team_ids.map(lambda value: _stable_identity(value, "oe:team:"))
    if not bool(stable_player.all()) or not bool(stable_team.all()):
        raise FutureValueSourceError("player form contains unstable player or team identity")
    if champions.isna().any() or champions.eq("").any():
        raise FutureValueSourceError("player form contains missing champion identity")
    if roles.isna().any():
        raise FutureValueSourceError("player form contains an unknown role")
    base = pd.DataFrame(
        {
            "game_id": metrics["_game_id"].astype(str).to_numpy(),
            "date": pd.to_datetime(metrics["_date"], utc=True).to_numpy(),
            "side": metrics["_side"].astype(str).to_numpy(),
            "role": roles.astype(str).to_numpy(),
            "player_id": player_ids.astype(str).to_numpy(),
            "team_id": team_ids.astype(str).to_numpy(),
            "champion": champions.astype(str).to_numpy(),
            "competition_tier": metrics["_tier"].astype("string").to_numpy(),
        }
    )
    prior = _strict_prior_time_decay(
        pd.concat(
            [
                base[["player_id", "date"]].reset_index(drop=True),
                metrics[list(FORM_METRICS)].reset_index(drop=True),
            ],
            axis=1,
        ),
        entity_column="player_id",
        date_column="date",
        metric_columns=FORM_METRICS,
        half_life_days=half_life_days,
    )
    output = pd.concat([base.reset_index(drop=True), prior.reset_index(drop=True)], axis=1)
    output.attrs["half_life_days"] = float(half_life_days)
    output.attrs["form_contract"] = "strict_prior_timestamp_block_exponential_v1"
    return output


@dataclass(frozen=True)
class Rank3AtomModel:
    """Fold-local rank-three champion-role representation."""

    metric_names: tuple[str, ...]
    rank: int
    center: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    champion_role_coordinates: Mapping[str, tuple[float, ...]]
    champion_role_support: Mapping[str, int]
    fit_game_ids: tuple[str, ...]
    fit_window_end: str

    def parameter_receipt(self) -> dict[str, Any]:
        """Return every fitted atom parameter and one canonical digest."""

        parameters: dict[str, Any] = {
            "metric_names": list(self.metric_names),
            "rank": int(self.rank),
            "center": [float(value) for value in self.center],
            "scale": [float(value) for value in self.scale],
            "components": [
                [float(value) for value in row] for row in self.components
            ],
            "champion_role_coordinates": {
                str(key): [float(value) for value in self.champion_role_coordinates[key]]
                for key in sorted(self.champion_role_coordinates)
            },
            "champion_role_support": {
                str(key): int(self.champion_role_support[key])
                for key in sorted(self.champion_role_support)
            },
            "fit_game_ids": list(self.fit_game_ids),
            "fit_game_identity_sha256": identity_sha256(self.fit_game_ids),
            "fit_window_end": self.fit_window_end,
        }
        parameters["parameter_sha256"] = hashlib.sha256(
            _canonical_json_bytes(parameters)
        ).hexdigest()
        return parameters

    def transform(self, form: pd.DataFrame) -> pd.DataFrame:
        required = {"champion", "role", "player_id", *self.metric_names}
        missing = sorted(required - set(form.columns))
        if missing:
            raise FutureValueSourceError(
                "rank-3 atom input is missing columns: " + ", ".join(missing)
            )
        player_names = [f"rank_3_player_atom_{index + 1}" for index in range(self.rank)]
        champion_names = [
            f"rank_3_champion_role_atom_{index + 1}" for index in range(self.rank)
        ]
        values = form[list(self.metric_names)].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        player_available = np.isfinite(values).all(axis=1)
        player_coordinates = np.full((len(form), self.rank), np.nan, dtype=float)
        if player_available.any():
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                candidate_coordinates = (
                    (values[player_available] - self.center) / self.scale
                ) @ self.components.T
            finite_coordinates = np.isfinite(candidate_coordinates).all(axis=1)
            available_positions = np.flatnonzero(player_available)
            player_coordinates[available_positions[finite_coordinates]] = (
                candidate_coordinates[finite_coordinates]
            )
            player_available[available_positions[~finite_coordinates]] = False

        keys = [
            _champion_role_key(champion, role)
            for champion, role in zip(form["champion"], form["role"])
        ]
        champion_coordinates = np.full((len(form), self.rank), np.nan, dtype=float)
        champion_support = np.zeros(len(form), dtype=np.int64)
        champion_available = np.zeros(len(form), dtype=bool)
        known_keys = tuple(self.champion_role_coordinates)
        known_codes = {key: position for position, key in enumerate(known_keys)}
        codes = np.asarray([known_codes.get(key, -1) for key in keys], dtype=np.int64)
        known = codes >= 0
        if known.any():
            coordinate_matrix = np.asarray(
                [self.champion_role_coordinates[key] for key in known_keys],
                dtype=float,
            )
            support_values = np.asarray(
                [self.champion_role_support.get(key, 0) for key in known_keys],
                dtype=np.int64,
            )
            champion_coordinates[known] = coordinate_matrix[codes[known]]
            champion_support[known] = support_values[codes[known]]
            champion_available[known] = True

        result = pd.DataFrame(index=form.index)
        for position, name in enumerate(player_names):
            result[name] = player_coordinates[:, position]
        for position, name in enumerate(champion_names):
            result[name] = champion_coordinates[:, position]
        result["rank_3_player_atom_available"] = player_available
        result["rank_3_champion_role_atom_available"] = champion_available
        result["rank_3_champion_role_support"] = champion_support
        return result


def _champion_role_key(champion: Any, role: Any) -> str:
    if champion is None or bool(pd.isna(champion)):
        raise FutureValueSourceError("rank-3 atom has missing champion or role")
    champion_text = str(champion).strip().casefold()
    role_text = _role(role)
    if champion_text in {"", "nan", "none", "<na>"} or role_text is None:
        raise FutureValueSourceError("rank-3 atom has missing champion or role")
    return f"{champion_text}|{role_text}"


def fit_rank3_player_champion_role_atoms(
    form: pd.DataFrame,
    *,
    train_game_ids: Iterable[str],
    rank: int = RANK_3,
    min_cell_support: int = 1,
    fit_window_end: Any | None = None,
) -> Rank3AtomModel:
    """Fit a rank-three representation from one chronological train fold."""

    if int(rank) != RANK_3:
        raise FutureValueSourceError("future-value atom rank is fixed at three")
    if int(min_cell_support) < 1:
        raise FutureValueSourceError("rank-3 atom support floor is invalid")
    train_ids = tuple(sorted({str(value) for value in train_game_ids}))
    if not train_ids:
        raise FutureValueSourceError("rank-3 atom fit has no training games")
    required = {"game_id", "date", "champion", "role", *[f"prior_form_{m}" for m in FORM_METRICS]}
    missing = sorted(required - set(form.columns))
    if missing:
        raise FutureValueSourceError("rank-3 atom form is missing columns: " + ", ".join(missing))
    train = form[form["game_id"].astype(str).isin(train_ids)].copy()
    if train.empty:
        raise FutureValueSourceError("rank-3 atom training games are absent")
    observed_train_ids = set(train["game_id"].astype(str))
    missing_train_ids = sorted(set(train_ids) - observed_train_ids)
    if missing_train_ids:
        raise FutureValueSourceError(
            "rank-3 atom training games are missing: " + ", ".join(missing_train_ids[:5])
        )
    train["date"] = pd.to_datetime(train["date"], utc=True, errors="coerce")
    if train["date"].isna().any():
        raise FutureValueSourceError("rank-3 atom training date is invalid")
    if fit_window_end is not None:
        boundary = _utc_timestamp(fit_window_end, "fit_window_end")
        if train["date"].ge(boundary).any():
            raise FutureValueSourceError("rank-3 atom fit includes a boundary or future row")
    metric_columns = [f"prior_form_{metric}" for metric in FORM_METRICS]
    train["_key"] = [
        _champion_role_key(champion, role)
        for champion, role in zip(train["champion"], train["role"])
    ]
    grouped = train.groupby("_key", sort=True, observed=True)
    aggregate = grouped[metric_columns].mean()
    support = grouped[metric_columns].count().min(axis=1)
    eligible = aggregate.notna().all(axis=1) & support.ge(int(min_cell_support))
    aggregate = aggregate.loc[eligible]
    support = support.loc[eligible]
    if len(aggregate) < RANK_3 or len(metric_columns) < RANK_3:
        raise FutureValueSourceError("rank-3 atom fit has insufficient complete champion-role cells")
    matrix = aggregate.to_numpy(dtype=float)
    center = matrix.mean(axis=0)
    scale = matrix.std(axis=0, ddof=0)
    if not np.isfinite(matrix).all():
        raise FutureValueSourceError("rank-3 atom aggregate contains non-finite values")
    if not np.isfinite(center).all():
        raise FutureValueSourceError("rank-3 atom center contains non-finite values")
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    scaled = (matrix - center) / scale
    if not np.isfinite(scaled).all():
        raise FutureValueSourceError("rank-3 atom standardization contains non-finite values")
    _u, _s, components = np.linalg.svd(scaled, full_matrices=False)
    components = components[:RANK_3].copy()
    for component_index in range(RANK_3):
        pivot = int(np.argmax(np.abs(components[component_index])))
        if components[component_index, pivot] < 0.0:
            components[component_index] *= -1.0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        coordinates = scaled @ components.T
    if not np.isfinite(coordinates).all():
        raise FutureValueSourceError("rank-3 atom coordinates contain non-finite values")
    fit_end = (
        _utc_text(fit_window_end)
        if fit_window_end is not None
        else _utc_text(train["date"].max())
    )
    return Rank3AtomModel(
        metric_names=tuple(metric_columns),
        rank=RANK_3,
        center=center,
        scale=scale,
        components=components,
        champion_role_coordinates={
            str(key): tuple(float(value) for value in row)
            for key, row in zip(aggregate.index, coordinates)
        },
        champion_role_support={str(key): int(value) for key, value in support.items()},
        fit_game_ids=train_ids,
        fit_window_end=fit_end,
    )


def _verified_authoritative_series_column(
    maps: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    verified_source_receipt_sha256: str,
) -> str | None:
    receipt = maps.attrs.get("verified_series_receipt")
    if not isinstance(receipt, Mapping):
        return None
    payload = dict(receipt)
    claimed_hash = payload.pop("receipt_sha256", None)
    if not isinstance(claimed_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", claimed_hash
    ):
        raise FutureValueSourceError("authoritative series receipt hash is invalid")
    if hashlib.sha256(_canonical_json_bytes(payload)).hexdigest() != claimed_hash:
        raise FutureValueSourceError("authoritative series receipt changed")
    column = payload.get("series_column")
    game_count = payload.get("game_count")
    if (
        payload.get("source_type") != "verified_grid_series"
        or column != "grid_series_id"
        or not isinstance(game_count, int)
        or isinstance(game_count, bool)
        or game_count != len(frame)
        or payload.get("game_identity_sha256")
        != identity_sha256(frame["game_id"].astype(str))
        or payload.get("source_receipt_sha256")
        != verified_source_receipt_sha256
        or column not in frame.columns
    ):
        raise FutureValueSourceError("authoritative series receipt does not bind maps")
    series = frame[column].astype("string").str.strip()
    if series.isna().any() or series.eq("").any():
        raise FutureValueSourceError("authoritative series identity is incomplete")
    assignment_rows = sorted(
        (
            {"game_id": str(game_id), "series_id": str(series_id)}
            for game_id, series_id in zip(frame["game_id"], series)
        ),
        key=lambda row: row["game_id"],
    )
    if payload.get("series_assignment_sha256") != hashlib.sha256(
        _canonical_json_bytes(assignment_rows)
    ).hexdigest():
        raise FutureValueSourceError("authoritative series assignments changed")
    team_columns = next(
        (
            (blue_name, red_name)
            for blue_name, red_name in (
                ("blue_teamid", "red_teamid"),
                ("blue_team_key", "red_team_key"),
                ("blue_team", "red_team"),
            )
            if blue_name in frame.columns and red_name in frame.columns
        ),
        None,
    )
    if team_columns is None:
        raise FutureValueSourceError("authoritative series team identity is missing")
    blue_column, red_column = team_columns
    pair_rows = []
    series_pairs: dict[str, set[str]] = {}
    for game_id, series_id, blue_team, red_team in zip(
        frame["game_id"], series, frame[blue_column], frame[red_column]
    ):
        teams = sorted(
            (str(blue_team).strip().casefold(), str(red_team).strip().casefold())
        )
        if not all(teams):
            raise FutureValueSourceError("authoritative series team identity is empty")
        pair = "|".join(teams)
        series_pairs.setdefault(str(series_id), set()).add(pair)
        pair_rows.append(
            {"game_id": str(game_id), "series_id": str(series_id), "team_pair": pair}
        )
    if any(len(pairs) != 1 for pairs in series_pairs.values()):
        raise FutureValueSourceError("authoritative series spans multiple team pairs")
    pair_rows.sort(key=lambda row: row["game_id"])
    if payload.get("series_pair_assignment_sha256") != hashlib.sha256(
        _canonical_json_bytes(pair_rows)
    ).hexdigest():
        raise FutureValueSourceError("authoritative series team assignments changed")
    return str(column)


def _conservative_series_partition(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign the source-neutral proxy partition used without a crosswalk."""

    team_columns = next(
        (
            (blue_name, red_name)
            for blue_name, red_name in (
                ("blue_teamid", "red_teamid"),
                ("blue_team_key", "red_team_key"),
                ("blue_team", "red_team"),
            )
            if blue_name in frame.columns and red_name in frame.columns
        ),
        None,
    )
    if team_columns is not None:
        blue_team, red_team = team_columns
        blue = frame[blue_team].astype("string").str.strip().str.casefold()
        red = frame[red_team].astype("string").str.strip().str.casefold()
        league = frame.get("league", pd.Series("", index=frame.index)).astype("string")
        tournament = frame.get("tournament", pd.Series("", index=frame.index)).astype("string")
        valid = blue.notna() & red.notna() & blue.ne("") & red.ne("")
        if bool(valid.all()):
            pair = ["|".join(sorted((left, right))) for left, right in zip(blue, red)]
            league_key = league.fillna("<missing>").astype(str).str.strip().str.casefold()
            tournament_key = (
                tournament.fillna("<missing>").astype(str).str.strip().str.casefold()
            )
            frame["series_id"] = [
                "proxy:"
                + "|".join((str(league_value), str(tournament_value), team_pair))
                for league_value, tournament_value, team_pair in zip(
                    league_key, tournament_key, pair
                )
            ]
            cluster_sizes = frame["series_id"].value_counts(sort=False)
            colliding = cluster_sizes.gt(1)
            frame.attrs["series_cluster_source"] = "conservative_series_superset"
            frame.attrs["series_cluster_audit"] = {
                "source": "conservative_series_superset",
                "authoritative": False,
                "cluster_count": int(len(cluster_sizes)),
                "map_count": int(len(frame)),
                "colliding_cluster_count": int(colliding.sum()),
                "collision_extra_map_count": int(
                    cluster_sizes.loc[colliding].sub(1).sum()
                ),
                "max_cluster_size": int(cluster_sizes.max()),
                "key_fields": ["league", "tournament", "unordered_team_pair"],
                "team_identity_columns": [blue_team, red_team],
                "stable_team_ids": team_columns == ("blue_teamid", "red_teamid"),
            }
            return frame
    frame["series_id"] = frame["game_id"]
    frame.attrs["series_cluster_source"] = "game_id_fallback"
    frame.attrs["series_cluster_audit"] = {
        "source": "game_id_fallback",
        "authoritative": False,
        "cluster_count": int(len(frame)),
        "map_count": int(len(frame)),
        "colliding_cluster_count": 0,
        "collision_extra_map_count": 0,
        "max_cluster_size": 1,
    }
    return frame


def _crosswalk_team_columns(frame: pd.DataFrame) -> tuple[str, str] | None:
    for pair in (
        ("blue_team_key", "red_team_key"),
        ("blue_team", "red_team"),
        ("blue_teamid", "red_teamid"),
    ):
        if pair[0] in frame.columns and pair[1] in frame.columns:
            return pair
    return None


def _apply_verified_leaguepedia_partition(
    maps: pd.DataFrame,
    frame: pd.DataFrame,
    binding: Mapping[str, Any],
    *,
    verified_source_receipt_sha256: str | None,
) -> pd.DataFrame:
    """Mix verified Leaguepedia assignments with conservative proxy rows."""

    source_hash = str(binding.get("source_receipt_sha256") or "")
    if verified_source_receipt_sha256 is not None and source_hash != str(
        verified_source_receipt_sha256
    ):
        raise FutureValueSourceError("Leaguepedia crosswalk source receipt does not match maps")
    partition = _conservative_series_partition(frame)
    proxy_audit = dict(partition.attrs.get("series_cluster_audit") or {})
    # The verified binding contains every assignment. Pandas otherwise deep
    # copies that large attribute on each column selection below.
    partition.attrs.clear()
    partition = partition.copy(deep=False)
    partition.attrs.clear()
    assignments = binding.get("assignments")
    if not isinstance(assignments, Sequence) or isinstance(
        assignments, (str, bytes, bytearray)
    ):
        raise FutureValueSourceError("Leaguepedia crosswalk assignments are missing")
    assignment_by_id = {str(row["oe_game_id"]): row for row in assignments}
    team_columns = _crosswalk_team_columns(maps)
    map_ids = set(partition["game_id"].astype(str))
    mapped_ids = sorted(map_ids & set(assignment_by_id))
    if mapped_ids and team_columns is None:
        raise FutureValueSourceError("Leaguepedia crosswalk team columns are missing")
    if team_columns is not None:
        blue_column, red_column = team_columns
        raw_ids = _frame_game_ids(maps, "maps").astype(str)
        actual_pair_by_id = {
            str(game_id): tuple(
                sorted(
                    (
                        _leaguepedia_team_key(blue_team),
                        _leaguepedia_team_key(red_team),
                    )
                )
            )
            for game_id, blue_team, red_team in zip(
                raw_ids,
                maps[blue_column].to_numpy(copy=False),
                maps[red_column].to_numpy(copy=False),
            )
        }
        for game_id in mapped_ids:
            assignment = assignment_by_id[game_id]
            expected_pair = tuple(
                sorted(_leaguepedia_team_key(value) for value in assignment["normalized_team_set"])
            )
            actual_pair = actual_pair_by_id[game_id]
            if not expected_pair or actual_pair != expected_pair:
                raise FutureValueSourceError(
                    "Leaguepedia crosswalk team pair does not match OE maps"
                )
    # A conservative proxy can contain more than one real series.  Promote a
    # mapped series only when the current frame contains every row in that
    # proxy and the crosswalk series record lists exactly those rows.  This
    # keeps an incomplete crosswalk from splitting one proxy cluster.
    series_membership = binding.get("series_membership")
    if not isinstance(series_membership, Mapping):
        raise FutureValueSourceError("Leaguepedia crosswalk series membership is missing")
    proxy_ids = partition["series_id"].astype(str)
    game_ids = partition["game_id"].astype(str)
    promoted_ids: set[str] = set()
    promoted_series_by_game: dict[str, str] = {}
    retained_proxy_ids: set[str] = set()
    proxy_game_ids: dict[str, set[str]] = {}
    for proxy_id, game_id in zip(proxy_ids.to_numpy(), game_ids.to_numpy()):
        proxy_game_ids.setdefault(str(proxy_id), set()).add(str(game_id))
    for proxy_id, group_game_ids in proxy_game_ids.items():
        group_assignments = [
            assignment_by_id[game_id]
            for game_id in sorted(group_game_ids)
            if game_id in assignment_by_id
        ]
        if len(group_assignments) != len(group_game_ids):
            retained_proxy_ids.add(str(proxy_id))
            continue
        assignments_by_series: dict[str, set[str]] = {}
        for assignment in group_assignments:
            assignments_by_series.setdefault(str(assignment["series_id"]), set()).add(
                str(assignment["oe_game_id"])
            )
        valid_group = True
        for series_id, assigned_ids in assignments_by_series.items():
            membership = series_membership.get(series_id)
            if not isinstance(membership, Mapping):
                valid_group = False
                break
            evidence_ids = {
                str(value) for value in membership.get("oe_game_ids", ())
            }
            # The complete source frame is the census bound to this binding.
            # Require exact membership here.  A series row missing from the
            # frame keeps the whole proxy group conservative.
            if evidence_ids != assigned_ids:
                valid_group = False
                break
        if not valid_group:
            retained_proxy_ids.add(str(proxy_id))
            continue
        for series_id, assigned_ids in assignments_by_series.items():
            promoted_ids.update(assigned_ids)
            for game_id in assigned_ids:
                promoted_series_by_game[game_id] = "leaguepedia:" + series_id
    if promoted_series_by_game:
        promoted_mask = game_ids.isin(promoted_series_by_game)
        partition.loc[promoted_mask, "series_id"] = game_ids.loc[promoted_mask].map(
            promoted_series_by_game
        )
    cluster_sizes = partition["series_id"].astype(str).value_counts(sort=False)
    colliding = cluster_sizes.gt(1)
    partition.attrs["series_cluster_source"] = LEAGUEPEDIA_CROSSWALK_SOURCE
    partition.attrs["series_cluster_audit"] = {
        **proxy_audit,
        "source": LEAGUEPEDIA_CROSSWALK_SOURCE,
        "authoritative": False,
        "mapped_series_authoritative": True,
        "mapped_game_count": len(mapped_ids),
        "unmatched_game_count": int(len(partition) - len(mapped_ids)),
        "mapped_series_count": len({str(assignment_by_id[game_id]["series_id"]) for game_id in mapped_ids}),
        "promoted_game_count": int(len(promoted_ids)),
        "promoted_series_count": int(
            partition.loc[partition["game_id"].astype(str).isin(promoted_ids), "series_id"]
            .astype(str)
            .str.removeprefix("leaguepedia:")
            .nunique()
        ),
        "retained_proxy_game_count": int(len(partition) - len(promoted_ids)),
        "retained_proxy_cluster_count": int(len(retained_proxy_ids)),
        "partial_series_blocker": bool(retained_proxy_ids),
        "cluster_count": int(len(cluster_sizes)),
        "map_count": int(len(partition)),
        "colliding_cluster_count": int(colliding.sum()),
        "collision_extra_map_count": int(
            cluster_sizes.loc[colliding].sub(1).sum()
        ),
        "max_cluster_size": int(cluster_sizes.max()),
        "crosswalk_artifact_sha256": binding.get("artifact_sha256"),
        "crosswalk_sha256": binding.get("crosswalk_sha256"),
        "crosswalk_receipt_sha256": binding.get("receipt_sha256"),
        "crosswalk_assignment_sha256": binding.get("assignment_sha256"),
        "source_receipt_sha256": source_hash,
        "authoritative_series_blocker": "authoritative_series_id_missing_proxy_cluster_used",
    }
    return partition


def _map_model_frame(
    maps: pd.DataFrame,
    *,
    verified_source_receipt_sha256: str | None = None,
    verified_source_receipt: Mapping[str, Any] | None = None,
    verified_crosswalk_receipt_file_sha256: str | None = None,
) -> pd.DataFrame:
    required = {"date", "y_blue_win"}
    missing = sorted(required - set(maps.columns))
    if missing:
        raise FutureValueSourceError("model maps are missing columns: " + ", ".join(missing))
    frame = maps.copy()
    frame["game_id"] = _frame_game_ids(frame, "maps").astype(str)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame["target"] = pd.to_numeric(frame["y_blue_win"], errors="coerce")
    if frame["game_id"].duplicated().any() or frame["date"].isna().any():
        raise FutureValueSourceError("model maps do not have one dated row per game")
    if not frame["target"].isin({0, 1}).all():
        raise FutureValueSourceError("model maps contain an invalid result target")
    crosswalk_attrs = maps.attrs.get("verified_leaguepedia_series_crosswalk")
    if crosswalk_attrs is not None and not isinstance(crosswalk_attrs, Mapping):
        raise FutureValueSourceError("Leaguepedia crosswalk binding is invalid")
    if crosswalk_attrs is not None and isinstance(
        maps.attrs.get("verified_series_receipt"), Mapping
    ):
        raise FutureValueSourceError("map frame has multiple series authority bindings")
    if crosswalk_attrs is not None:
        if not isinstance(verified_source_receipt, Mapping):
            raise FutureValueSourceError(
                "verified source receipt is required for Leaguepedia crosswalk"
            )
        if verified_source_receipt_sha256 is None:
            verified_source_receipt_sha256 = str(
                verified_source_receipt.get("receipt_sha256") or ""
            )
        artifact_path = crosswalk_attrs.get("artifact_path")
        receipt_path = crosswalk_attrs.get("receipt_path")
        if not artifact_path or not receipt_path:
            raise FutureValueSourceError("Leaguepedia crosswalk file binding is missing")
        if verified_crosswalk_receipt_file_sha256 is None:
            raise FutureValueSourceError(
                "independent Leaguepedia crosswalk receipt file hash is required"
            )
        binding = load_verified_leaguepedia_series_crosswalk(
            artifact_path,
            receipt_path,
            source_receipt=verified_source_receipt,
            expected_source_receipt_sha256=verified_source_receipt_sha256,
            expected_receipt_file_sha256=verified_crosswalk_receipt_file_sha256,
        )
        for key in (
            "artifact_sha256",
            "receipt_sha256",
            "crosswalk_sha256",
            "assignment_sha256",
            "receipt_file_sha256",
        ):
            if crosswalk_attrs.get(key) != binding.get(key):
                raise FutureValueSourceError(
                    f"Leaguepedia crosswalk binding changed: {key}"
                )
        return _apply_verified_leaguepedia_partition(
            maps,
            frame,
            binding,
            verified_source_receipt_sha256=verified_source_receipt_sha256,
        )
    series_column = (
        _verified_authoritative_series_column(
            maps,
            frame,
            verified_source_receipt_sha256=verified_source_receipt_sha256,
        )
        if verified_source_receipt_sha256 is not None
        and isinstance(maps.attrs.get("verified_series_receipt"), Mapping)
        else None
    )
    valid_authoritative_series = False
    if series_column is not None:
        series = frame[series_column].astype("string").str.strip()
        valid_authoritative_series = bool(series.notna().all() and series.ne("").all())
        if valid_authoritative_series:
            frame["series_id"] = series
            cluster_sizes = series.value_counts(sort=False)
            colliding = cluster_sizes.gt(1)
            frame.attrs["series_cluster_source"] = f"authoritative:{series_column}"
            frame.attrs["series_cluster_audit"] = {
                "source": f"authoritative:{series_column}",
                "authoritative": True,
                "cluster_count": int(len(cluster_sizes)),
                "map_count": int(len(frame)),
                "colliding_cluster_count": int(colliding.sum()),
                "collision_extra_map_count": int(
                    cluster_sizes.loc[colliding].sub(1).sum()
                ),
                "max_cluster_size": int(cluster_sizes.max()),
                "receipt_sha256": maps.attrs["verified_series_receipt"][
                    "receipt_sha256"
                ],
            }
    if not valid_authoritative_series:
        frame = _conservative_series_partition(frame)
    return frame


def _team_history_features(
    map_frame: pd.DataFrame,
    form: pd.DataFrame,
) -> pd.DataFrame:
    """Create strictly-prior team win and roster continuity state."""

    team_rows = (
        form.groupby(["game_id", "side", "team_id"], sort=False, observed=True)
        .size()
        .reset_index(name="roster_rows")
    )
    if not team_rows["roster_rows"].eq(5).all():
        raise FutureValueSourceError("team history requires exact five-player rosters")
    game_info = map_frame[["game_id", "date", "target"]].copy()
    team_rows = team_rows.merge(game_info, on="game_id", how="left", validate="many_to_one")
    team_rows["team_result"] = np.where(
        team_rows["side"].astype(str).str.casefold().eq("blue"),
        team_rows["target"],
        1.0 - team_rows["target"],
    )
    prior = _strict_prior_time_decay(
        team_rows[["team_id", "date", "team_result"]],
        entity_column="team_id",
        date_column="date",
        metric_columns=["team_result"],
        half_life_days=TIME_DECAY_HALF_LIFE_DAYS,
    )
    team_rows["prior_team_win"] = prior["prior_form_team_result"].to_numpy()
    team_rows["prior_team_support"] = prior["prior_form_team_result_support"].to_numpy()

    rosters = {
        (str(game_id), str(side)): frozenset(group["player_id"].astype(str))
        for (game_id, side), group in form.groupby(["game_id", "side"], sort=False, observed=True)
    }
    team_rows = team_rows.sort_values(["team_id", "date", "game_id"], kind="stable")
    continuity = np.full(len(team_rows), np.nan, dtype=float)
    previous_roster: dict[str, frozenset[str] | None] = {}
    previous_date: dict[str, pd.Timestamp | None] = {}
    for position, row in enumerate(team_rows.itertuples(index=False)):
        team_id = str(row.team_id)
        current_date = pd.Timestamp(row.date)
        if previous_date.get(team_id) is not None and previous_date[team_id] < current_date:
            current = rosters.get((str(row.game_id), str(row.side)))
            prior_roster = previous_roster.get(team_id)
            if current is not None and prior_roster:
                continuity[position] = len(current & prior_roster) / 5.0
        previous_roster[team_id] = rosters.get((str(row.game_id), str(row.side)))
        previous_date[team_id] = current_date
    team_rows["roster_continuity"] = continuity
    return team_rows[[
        "game_id",
        "side",
        "prior_team_win",
        "prior_team_support",
        "roster_continuity",
    ]]


SIDE_LEVEL_TO_MODEL_FEATURE = {
    **{f"player_form_{metric}": f"player_form_{metric}" for metric in FORM_METRICS},
    **{
        f"rank_3_player_atom_{index}": f"rank_3_player_atom_{index}"
        for index in range(1, RANK_3 + 1)
    },
    **{
        f"rank_3_champion_role_atom_{index}": f"rank_3_champion_role_atom_{index}"
        for index in range(1, RANK_3 + 1)
    },
    "team_prior_win": "team_prior_win_diff",
    "roster_continuity": "roster_continuity_diff",
    "player_form_missing_rate": "player_form_missing_rate_diff",
    "rank_3_atom_missing_rate": "rank_3_atom_missing_rate_diff",
    "rank_3_champion_role_atom_missing_rate": (
        "rank_3_champion_role_atom_missing_rate_diff"
    ),
    "player_form_support_mean": "player_form_support_mean_diff",
    "player_form_effective_support_mean": "player_form_effective_support_mean_diff",
}
MODEL_FEATURES = tuple(SIDE_LEVEL_TO_MODEL_FEATURE.values())
CENTERED_ATOM_LEVEL_FEATURES = frozenset(
    feature
    for feature in SIDE_LEVEL_TO_MODEL_FEATURE
    if feature.startswith("rank_3_player_atom_")
    or feature.startswith("rank_3_champion_role_atom_")
)


# These declarations are the only feature families accepted by the variant
# contract.  Keep the names in source order.  The order is part of each
# canonical receipt and therefore part of the model identity.
RATING_VARIANT_SCHEMA_VERSION = "scryglass:future-value-rating-variants:v2"
RATING_FEATURE_LEDGER_SCHEMA_VERSION = "scryglass:future-value-rating-feature-ledger:v1"
RATING_FEATURE_PRODUCER_SCHEMA_VERSION = (
    "scryglass:future-value-rating-feature-producer:v1"
)
RATING_FEATURE_PRODUCER_RECEIPT_SCHEMA_VERSION = (
    "scryglass:future-value-rating-feature-producer-receipt:v2"
)
RATING_FEATURE_PRODUCER_MANIFEST_SCHEMA_VERSION = (
    "scryglass:future-value-rating-feature-producer-manifest:v2"
)
RATING_FEATURE_PRODUCER_AUTHORITY = MappingProxyType(
    {
        "research_only": True,
        "public_player_rating": False,
        "public_team_rating": False,
        "public_probability": False,
        "promotion": False,
        "merge": False,
        "deployment": False,
    }
)
CURRENT_RATING_SIGNED_MAP_FEATURES = (
    "base_team_logit",
    "team_rating_diff_scaled",
    "base_player_logit",
    "player_rating_diff_scaled",
)
CURRENT_RATING_STRENGTH_FEATURES = (
    "base_team_logit",
    "base_player_logit",
)
CURRENT_RATING_RAW_DIFFERENCE_FEATURES = (
    "team_rating_diff_scaled",
    "player_rating_diff_scaled",
)
CURRENT_RATING_FEATURE_SEMANTICS = MappingProxyType(
    {
        "base_team_logit": "shrunk_team_strength_logit; source shrinkage carries uncertainty",
        "base_player_logit": "shrunk_player_strength_logit; source shrinkage carries uncertainty",
        "team_rating_diff_scaled": "raw_team_strength_difference_scaled; uncertainty stays external",
        "player_rating_diff_scaled": "raw_player_strength_difference_scaled; uncertainty stays external",
    }
)

FUTURE_PLAYER_FORM_SIDE_FEATURES = tuple(
    feature
    for feature in SIDE_LEVEL_TO_MODEL_FEATURE
    if feature not in {"team_prior_win", "roster_continuity"}
)

SCALING_CURVE_SIGNED_MAP_FEATURES = tuple(
    f"forecast_{metric}_diff_{checkpoint}"
    for checkpoint in CHECKPOINTS
    for metric in ("gold", "xp")
)

# The atomized producer owns the checkpoint names and the phase producer owns
# the complete shape diagnostics.  The diagnostics remain report fields.  The
# eight forecast differences are the only scaling inputs in the four-way fit.
SCALING_CURVE_PRODUCER_SCHEMA_VERSION = PHASE_CURVE_SCHEMA_VERSION
SCALING_CURVE_PRODUCER_VERSION = PHASE_CURVE_PRODUCER_VERSION
SCALING_CURVE_FEATURE_DECLARATION = tuple(PHASE_FEATURE_DECLARATION)
SCALING_CURVE_SHAPE_FEATURES = tuple(PHASE_SHAPE_FEATURES)
SCALING_CURVE_SIGNED_SHAPE_FEATURES = tuple(PHASE_SHAPE_SIGNED_FEATURES)
SCALING_CURVE_INVARIANT_SHAPE_FEATURES = tuple(PHASE_SHAPE_INVARIANT_FEATURES)
SCALING_CURVE_DERIVED_FEATURES = tuple(
    dict.fromkeys(
        feature
        for feature in (
            *SCALING_CURVE_FEATURE_DECLARATION,
            *SCALING_CURVE_SHAPE_FEATURES,
            "scaling_index",
            "snowball_index",
            "comeback_resilience",
        )
        if feature not in SCALING_CURVE_SIGNED_MAP_FEATURES
    )
)

_RATING_VARIANT_FEATURES = {
    RatingVariant.CURRENT_ONLY: CURRENT_RATING_SIGNED_MAP_FEATURES,
    # The current rating is the comparison baseline.  The named candidate
    # families are added to it for the three other variants.
    RatingVariant.FUTURE_PLAYER_FORM: (
        *CURRENT_RATING_SIGNED_MAP_FEATURES,
        *FUTURE_PLAYER_FORM_SIDE_FEATURES,
    ),
    RatingVariant.SCALING_CURVE: (
        *CURRENT_RATING_SIGNED_MAP_FEATURES,
        *SCALING_CURVE_SIGNED_MAP_FEATURES,
    ),
    RatingVariant.BOTH: (
        *CURRENT_RATING_SIGNED_MAP_FEATURES,
        *FUTURE_PLAYER_FORM_SIDE_FEATURES,
        *SCALING_CURVE_SIGNED_MAP_FEATURES,
    ),
}
_RATING_VARIANT_SIGNED_FEATURES = {
    RatingVariant.CURRENT_ONLY: CURRENT_RATING_SIGNED_MAP_FEATURES,
    RatingVariant.FUTURE_PLAYER_FORM: CURRENT_RATING_SIGNED_MAP_FEATURES,
    RatingVariant.SCALING_CURVE: (
        *CURRENT_RATING_SIGNED_MAP_FEATURES,
        *SCALING_CURVE_SIGNED_MAP_FEATURES,
    ),
    RatingVariant.BOTH: (
        *CURRENT_RATING_SIGNED_MAP_FEATURES,
        *SCALING_CURVE_SIGNED_MAP_FEATURES,
    ),
}
_RATING_VARIANT_SIDE_FEATURES = {
    RatingVariant.CURRENT_ONLY: (),
    RatingVariant.FUTURE_PLAYER_FORM: FUTURE_PLAYER_FORM_SIDE_FEATURES,
    RatingVariant.SCALING_CURVE: (),
    RatingVariant.BOTH: FUTURE_PLAYER_FORM_SIDE_FEATURES,
}
_RATING_MODEL_FEATURE_UNIVERSE = frozenset(
    {
        *CURRENT_RATING_SIGNED_MAP_FEATURES,
        *FUTURE_PLAYER_FORM_SIDE_FEATURES,
        *SCALING_CURVE_SIGNED_MAP_FEATURES,
    }
)
TEAM_CONTEXT_FEATURES = (
    "team_prior_win",
    "roster_continuity",
    "team_prior_win_diff",
    "roster_continuity_diff",
)
_RATING_TEAM_CONTEXT_FEATURES = frozenset(TEAM_CONTEXT_FEATURES)
_RATING_DERIVED_FEATURES = frozenset(
    {
        *SCALING_CURVE_DERIVED_FEATURES,
    }
)


def _trusted_producer_spec(
    *,
    name: str,
    feature_family: str,
    feature_names: Sequence[str],
    implementation_locator: str,
    implementation_version: str,
) -> dict[str, Any]:
    """Build one code-owned producer adapter declaration.

    The implementation digest is derived from this declaration.  A caller
    must use one of these exact declarations.  A caller cannot mint a name,
    feature family, or implementation digest in a ledger receipt.
    """

    payload: dict[str, Any] = {
        "schema_version": RATING_FEATURE_PRODUCER_SCHEMA_VERSION,
        "name": name,
        "feature_family": feature_family,
        "feature_names": list(feature_names),
        "implementation_locator": implementation_locator,
        "implementation_version": implementation_version,
        "strict_prior_timing": "fit_rows_strictly_before_cutoff",
        "same_timestamp_policy": "batch_exclude_same_timestamp",
        "series_safety": "whole_series_disjoint",
        "target_free": True,
    }
    implementation_path = Path(__file__).resolve().parents[2] / implementation_locator
    if not implementation_path.is_file() or implementation_path.is_symlink():
        raise FutureValueSourceError(
            f"trusted rating feature producer source is missing: {implementation_locator}"
        )
    payload["implementation_sha256"] = _sha256_path(implementation_path)
    return payload


_REGISTERED_FEATURE_PRODUCER_SPECS = MappingProxyType(
    {
        "current_sequential_rating": MappingProxyType(
            _trusted_producer_spec(
                name="current_sequential_rating",
                feature_family="current_rating",
                feature_names=CURRENT_RATING_SIGNED_MAP_FEATURES,
                implementation_locator="lol_kills/ratings/player_elo.py",
                implementation_version="sequential-rating-v1",
            )
        ),
        "strict_prior_atomized_scaling": MappingProxyType(
            _trusted_producer_spec(
                name="strict_prior_atomized_scaling",
                feature_family="scaling_curve",
                feature_names=SCALING_CURVE_SIGNED_MAP_FEATURES,
                implementation_locator="lol_kills/research/atomized_rf_composite.py",
                implementation_version="strict-prior-scaling-v1",
            )
        ),
    }
)


def trusted_feature_producer_receipt(
    name: str,
    *,
    row_values_sha256: str | None = None,
) -> dict[str, Any]:
    """Return an immutable-by-value receipt for a registered producer.

    The returned value is a declaration, not a trust grant.  ``bind`` checks
    every field against the code-owned registry before it records the receipt.
    """

    key = str(name).strip()
    spec = _REGISTERED_FEATURE_PRODUCER_SPECS.get(key)
    if spec is None:
        raise FutureValueSourceError(f"unknown rating feature producer: {name}")
    payload = dict(spec)
    if row_values_sha256 is not None:
        if re.fullmatch(r"[0-9a-f]{64}", str(row_values_sha256), re.I) is None:
            raise FutureValueSourceError("rating feature producer row digest is invalid")
        payload["row_values_sha256"] = str(row_values_sha256).lower()
    payload["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    return payload


def _verified_source_receipt_for_ledger(
    receipt: Mapping[str, Any] | None,
) -> tuple[str, str]:
    """Verify the canonical source receipt without trusting ledger metadata."""

    validate_future_value_source_receipt_payload(receipt)
    return str(receipt["source_identity_sha256"]), str(receipt["receipt_sha256"])


def _verified_producer_adapters(
    producer: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], ...]:
    """Validate producer names and implementation hashes against the registry."""

    if not isinstance(producer, Mapping):
        raise FutureValueSourceError("trusted rating feature producer is required")
    raw_adapters = producer.get("adapters")
    if raw_adapters is None:
        raw_adapters = [producer]
    if not isinstance(raw_adapters, (list, tuple)) or not raw_adapters:
        raise FutureValueSourceError("rating feature producer adapters are missing")
    adapters: list[dict[str, Any]] = []
    for raw in raw_adapters:
        if not isinstance(raw, Mapping):
            raise FutureValueSourceError("rating feature producer adapter is invalid")
        name = str(raw.get("name") or "").strip()
        expected = _REGISTERED_FEATURE_PRODUCER_SPECS.get(name)
        if expected is None:
            raise FutureValueSourceError(f"unknown rating feature producer: {name}")
        allowed = set(expected) | {"receipt_sha256", "row_values_sha256"}
        if set(raw) - allowed:
            raise FutureValueSourceError("rating feature producer declaration has unknown fields")
        row_digest = str(raw.get("row_values_sha256") or "")
        if re.fullmatch(r"[0-9a-f]{64}", row_digest, re.I) is None:
            raise FutureValueSourceError(
                "rating feature producer row-value digest is required"
            )
        raw_payload = dict(raw)
        claimed_receipt = raw_payload.pop("receipt_sha256", None)
        if claimed_receipt is not None:
            if not isinstance(claimed_receipt, str) or hashlib.sha256(
                _canonical_json_bytes(raw_payload)
            ).hexdigest() != claimed_receipt:
                raise FutureValueSourceError(
                    "rating feature producer declaration receipt changed"
                )
        raw_payload.pop("row_values_sha256", None)
        if dict(raw_payload) != dict(expected):
            raise FutureValueSourceError(
                f"rating feature producer declaration changed: {name}"
            )
        if name in {str(item.get("name")) for item in adapters}:
            raise FutureValueSourceError("rating feature producer adapters are duplicated")
        verified = dict(expected)
        verified["row_values_sha256"] = row_digest.lower()
        adapters.append(verified)
    return tuple(adapters)


def _safe_file_path(value: Any, field: str, *, allow_missing: bool = False) -> Path:
    """Return one absolute, non-symlink file path.

    Producer receipts are research inputs.  A receipt cannot redirect the
    evaluator through a relative path, a symlink, or a directory.  The check
    covers each existing path component.  This matters when a parent
    directory is replaced after a receipt was written.
    """

    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise FutureValueSourceError(f"{field} path is invalid")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise FutureValueSourceError(f"{field} path is not safe")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise FutureValueSourceError(f"{field} path contains a symlink")
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise FutureValueSourceError(f"{field} path is not a regular file")
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise FutureValueSourceError(f"{field} path cannot be resolved") from error
        if resolved != path:
            raise FutureValueSourceError(f"{field} path is not canonical")
        return path
    if not allow_missing:
        raise FutureValueSourceError(f"{field} file is missing")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise FutureValueSourceError(f"{field} parent directory is unsafe")
    return path


def _leaguepedia_team_key(value: Any) -> str:
    """Normalize a team value for an exact crosswalk pair comparison."""

    if value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA:
        value = ""
    text = str(normalize_team(str(value))).strip().casefold()
    text = re.sub(r"[-_]+", " ", text)
    return " ".join(text.split())


def _leaguepedia_assignment_rows(
    assignments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in assignments]
    try:
        rows.sort(key=lambda row: str(row["oe_game_id"]))
    except (KeyError, TypeError) as error:
        raise FutureValueSourceError("Leaguepedia crosswalk assignments are invalid") from error
    return rows


def _leaguepedia_assignment_sha256(
    assignments: Sequence[Mapping[str, Any]],
) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(_leaguepedia_assignment_rows(assignments))
    ).hexdigest()


def _load_json_file(path: Path, field: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FutureValueSourceError(f"{field} cannot be read") from error
    if not isinstance(value, Mapping):
        raise FutureValueSourceError(f"{field} must be a JSON object")
    return value


def load_verified_leaguepedia_series_crosswalk(
    crosswalk_path: Path | str,
    receipt_path: Path | str,
    *,
    source_receipt: Mapping[str, Any],
    expected_source_receipt_sha256: str | None = None,
    expected_receipt_file_sha256: str,
) -> dict[str, Any]:
    """Load a byte-bound partial Leaguepedia series crosswalk.

    The crosswalk self-hash is checked against a separate receipt.  The
    receipt binds the artifact bytes, the accepted OE census, and every
    mapped assignment.  A caller cannot make a DataFrame attribute into
    series evidence without these files.
    """

    artifact_path = _safe_file_path(crosswalk_path, "Leaguepedia crosswalk artifact")
    receipt_file = _safe_file_path(receipt_path, "Leaguepedia crosswalk receipt")
    expected_receipt_file_sha256 = str(expected_receipt_file_sha256 or "").lower()
    if re.fullmatch(r"[0-9a-f]{64}", expected_receipt_file_sha256) is None:
        raise FutureValueSourceError("Leaguepedia crosswalk receipt file hash is invalid")
    actual_receipt_file_sha256 = _sha256_path(receipt_file)
    if actual_receipt_file_sha256 != expected_receipt_file_sha256:
        raise FutureValueSourceError("Leaguepedia crosswalk receipt file changed")
    source_accepted, source_eligible = validate_future_value_source_receipt_payload(
        source_receipt,
        expected_receipt_sha256=expected_source_receipt_sha256,
    )
    artifact_bytes = artifact_path.stat().st_size
    artifact_sha256 = _sha256_path(artifact_path)
    receipt_payload = _load_json_file(receipt_file, "Leaguepedia crosswalk receipt")
    if set(receipt_payload) != set(LEAGUEPEDIA_CROSSWALK_RECEIPT_FIELDS):
        raise FutureValueSourceError("Leaguepedia crosswalk receipt schema is invalid")
    if receipt_payload.get("schema_version") != LEAGUEPEDIA_CROSSWALK_RECEIPT_SCHEMA_VERSION:
        raise FutureValueSourceError("Leaguepedia crosswalk receipt version is invalid")
    if receipt_payload.get("status") != "verified_research_only":
        raise FutureValueSourceError("Leaguepedia crosswalk receipt status is invalid")
    if dict(receipt_payload.get("authority") or {}) != dict(
        LEAGUEPEDIA_CROSSWALK_RECEIPT_AUTHORITY
    ):
        raise FutureValueSourceError("Leaguepedia crosswalk receipt authority is invalid")
    receipt_hash = receipt_payload.get("receipt_sha256")
    receipt_body = dict(receipt_payload)
    receipt_body.pop("receipt_sha256", None)
    if not isinstance(receipt_hash, str) or re.fullmatch(r"[0-9a-f]{64}", receipt_hash, re.I) is None:
        raise FutureValueSourceError("Leaguepedia crosswalk receipt hash is invalid")
    if hashlib.sha256(_canonical_json_bytes(receipt_body)).hexdigest() != receipt_hash:
        raise FutureValueSourceError("Leaguepedia crosswalk receipt hash changed")
    artifact_record = receipt_payload.get("artifact")
    if not isinstance(artifact_record, Mapping) or set(artifact_record) != {
        "path", "bytes", "sha256"
    }:
        raise FutureValueSourceError("Leaguepedia crosswalk artifact binding is invalid")
    if Path(str(artifact_record.get("path"))).resolve() != artifact_path.resolve():
        raise FutureValueSourceError("Leaguepedia crosswalk artifact path changed")
    if artifact_record.get("bytes") != artifact_bytes or str(
        artifact_record.get("sha256") or ""
    ).lower() != artifact_sha256:
        raise FutureValueSourceError("Leaguepedia crosswalk artifact bytes changed")

    artifact = _load_json_file(artifact_path, "Leaguepedia crosswalk artifact")
    try:
        from lol_kills.research.oe_leaguepedia_series_crosswalk import verify_crosswalk

        verify_crosswalk(artifact)
    except (ImportError, ValueError) as error:
        raise FutureValueSourceError("Leaguepedia crosswalk artifact verification failed") from error
    crosswalk_hash = str(artifact.get("crosswalk_sha256") or "").lower()
    if receipt_payload.get("crosswalk_sha256") != crosswalk_hash:
        raise FutureValueSourceError("Leaguepedia crosswalk self-hash binding changed")
    source_hash = str(source_receipt["receipt_sha256"]).lower()
    if receipt_payload.get("source_receipt_sha256") != source_hash:
        raise FutureValueSourceError("Leaguepedia crosswalk source receipt changed")
    if receipt_payload.get("source_identity_sha256") != source_receipt[
        "source_identity_sha256"
    ]:
        raise FutureValueSourceError("Leaguepedia crosswalk source identity changed")
    if receipt_payload.get("accepted_game_count") != len(source_accepted):
        raise FutureValueSourceError("Leaguepedia crosswalk accepted count changed")
    if receipt_payload.get("accepted_game_identity_sha256") != identity_sha256(
        source_accepted
    ):
        raise FutureValueSourceError("Leaguepedia crosswalk accepted identity changed")

    binding = artifact.get("source_binding")
    if not isinstance(binding, Mapping):
        raise FutureValueSourceError("Leaguepedia crosswalk source binding is missing")
    if binding.get("receipt_sha256") != source_hash:
        raise FutureValueSourceError("Leaguepedia crosswalk source binding changed")
    if tuple(binding.get("accepted_game_ids") or ()) != source_accepted:
        raise FutureValueSourceError("Leaguepedia crosswalk accepted game IDs changed")
    if binding.get("accepted_game_count") != len(source_accepted) or binding.get(
        "accepted_game_identity_sha256"
    ) != identity_sha256(source_accepted):
        raise FutureValueSourceError("Leaguepedia crosswalk accepted census binding changed")
    if tuple(binding.get("selected_game_ids") or ()) != source_accepted or binding.get(
        "selected_is_full_accepted_census"
    ) is not True:
        raise FutureValueSourceError("Leaguepedia crosswalk selected census is incomplete")
    if tuple(binding.get("model_eligible_game_ids") or ()) != source_eligible:
        raise FutureValueSourceError("Leaguepedia crosswalk model census changed")
    if binding.get("model_eligible_game_count") != len(source_eligible) or binding.get(
        "model_eligible_game_identity_sha256"
    ) != identity_sha256(source_eligible):
        raise FutureValueSourceError("Leaguepedia crosswalk model census identity changed")

    if artifact.get("status") != "partial_authoritative_coverage":
        raise FutureValueSourceError("Leaguepedia crosswalk is not a partial artifact")
    assignments_value = artifact.get("assignments")
    if not isinstance(assignments_value, list):
        raise FutureValueSourceError("Leaguepedia crosswalk assignments are missing")
    assignments: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in assignments_value:
        if not isinstance(raw, Mapping):
            raise FutureValueSourceError("Leaguepedia crosswalk assignment is invalid")
        game_id = str(raw.get("oe_game_id") or "")
        series_id = str(raw.get("series_id") or "").strip()
        team_set = raw.get("normalized_team_set")
        if not game_id or not series_id or game_id in seen_ids:
            raise FutureValueSourceError("Leaguepedia crosswalk assignment identity is invalid")
        if tuple(canonical_game_ids((game_id,))) != (game_id,):
            raise FutureValueSourceError("Leaguepedia crosswalk assignment game ID is invalid")
        if not isinstance(team_set, list) or len(team_set) != 2 or any(
            not isinstance(value, str) or not value.strip() for value in team_set
        ) or team_set != sorted(team_set):
            raise FutureValueSourceError("Leaguepedia crosswalk assignment team set is invalid")
        if raw.get("outcome_used") is not False:
            raise FutureValueSourceError("Leaguepedia crosswalk assignment uses an outcome")
        if game_id not in set(source_accepted):
            raise FutureValueSourceError("Leaguepedia crosswalk assignment is outside source census")
        seen_ids.add(game_id)
        assignments.append(dict(raw))
    assignments = _leaguepedia_assignment_rows(assignments)
    assignment_ids = tuple(row["oe_game_id"] for row in assignments)
    assignment_hash = _leaguepedia_assignment_sha256(assignments)
    if receipt_payload.get("assignment_count") != len(assignments) or receipt_payload.get(
        "assignment_sha256"
    ) != assignment_hash:
        raise FutureValueSourceError("Leaguepedia crosswalk assignment hash changed")
    if tuple(receipt_payload.get("mapped_game_ids") or ()) != assignment_ids:
        raise FutureValueSourceError("Leaguepedia crosswalk mapped game IDs changed")
    if receipt_payload.get("mapped_game_count") != len(assignment_ids) or receipt_payload.get(
        "mapped_game_identity_sha256"
    ) != identity_sha256(assignment_ids):
        raise FutureValueSourceError("Leaguepedia crosswalk mapped identity changed")
    coverage = artifact.get("coverage")
    if not isinstance(coverage, Mapping) or coverage.get("mapped_game_count") != len(assignments):
        raise FutureValueSourceError("Leaguepedia crosswalk coverage changed")
    raw_series = artifact.get("series")
    if not isinstance(raw_series, list):
        raise FutureValueSourceError("Leaguepedia crosswalk series membership is missing")
    series_membership: dict[str, dict[str, Any]] = {}
    for raw_series_row in raw_series:
        if not isinstance(raw_series_row, Mapping):
            raise FutureValueSourceError("Leaguepedia crosswalk series membership is invalid")
        series_id = str(raw_series_row.get("series_id") or "").strip()
        raw_ids = raw_series_row.get("oe_game_ids")
        raw_team_set = raw_series_row.get("normalized_team_set")
        if not series_id or not isinstance(raw_ids, list) or not raw_ids:
            raise FutureValueSourceError("Leaguepedia crosswalk series membership is incomplete")
        try:
            series_ids = tuple(canonical_game_ids(str(value) for value in raw_ids))
        except (TypeError, ValueError) as error:
            raise FutureValueSourceError("Leaguepedia crosswalk series game IDs are invalid") from error
        if len(set(series_ids)) != len(series_ids) or not isinstance(raw_team_set, list):
            raise FutureValueSourceError("Leaguepedia crosswalk series membership is invalid")
        team_set = tuple(sorted(_leaguepedia_team_key(value) for value in raw_team_set))
        if len(team_set) != 2 or any(not value for value in team_set):
            raise FutureValueSourceError("Leaguepedia crosswalk series team set is invalid")
        if series_id in series_membership:
            raise FutureValueSourceError("Leaguepedia crosswalk series IDs are duplicated")
        series_membership[series_id] = {
            "oe_game_ids": list(series_ids),
            "normalized_team_set": list(team_set),
        }
    assignments_by_series: dict[str, set[str]] = {}
    assignment_team_sets_by_series: dict[str, set[tuple[str, ...]]] = {}
    for assignment in assignments:
        series_id = str(assignment["series_id"])
        assignments_by_series.setdefault(series_id, set()).add(str(assignment["oe_game_id"]))
        assignment_team_sets_by_series.setdefault(series_id, set()).add(
            tuple(
                sorted(
                    _leaguepedia_team_key(value)
                    for value in assignment["normalized_team_set"]
                )
            )
        )
        membership = series_membership.get(series_id)
        if membership is None:
            raise FutureValueSourceError("Leaguepedia crosswalk series membership is missing")
    for series_id, assigned_ids in assignments_by_series.items():
        membership = series_membership[series_id]
        if set(membership["oe_game_ids"]) != assigned_ids:
            raise FutureValueSourceError("Leaguepedia crosswalk series membership changed")
        assignment_team_sets = assignment_team_sets_by_series[series_id]
        if assignment_team_sets != {
            tuple(membership["normalized_team_set"])
        }:
            raise FutureValueSourceError("Leaguepedia crosswalk series team binding changed")
    return {
        "artifact_path": str(artifact_path),
        "artifact_bytes": int(artifact_bytes),
        "artifact_sha256": artifact_sha256,
        "receipt_path": str(receipt_file),
        "receipt_bytes": int(receipt_file.stat().st_size),
        "receipt_file_sha256": _sha256_path(receipt_file),
        "expected_receipt_file_sha256": expected_receipt_file_sha256,
        "receipt_sha256": receipt_hash,
        "crosswalk_sha256": crosswalk_hash,
        "source_receipt_sha256": source_hash,
        "source_identity_sha256": str(source_receipt["source_identity_sha256"]),
        "accepted_game_ids": list(source_accepted),
        "model_eligible_game_ids": list(source_eligible),
        "assignment_sha256": assignment_hash,
        "mapped_game_ids": list(assignment_ids),
        "assignments": assignments,
        "series_membership": series_membership,
    }


def bind_verified_leaguepedia_series_crosswalk(
    maps: pd.DataFrame,
    *,
    crosswalk_path: Path | str,
    receipt_path: Path | str,
    source_receipt: Mapping[str, Any],
    expected_receipt_file_sha256: str,
) -> pd.DataFrame:
    """Attach a verified mixed series partition to an OE map frame."""

    binding = load_verified_leaguepedia_series_crosswalk(
        crosswalk_path,
        receipt_path,
        source_receipt=source_receipt,
        expected_source_receipt_sha256=str(source_receipt.get("receipt_sha256") or ""),
        expected_receipt_file_sha256=expected_receipt_file_sha256,
    )
    result = maps.copy()
    result.attrs = dict(maps.attrs)
    result.attrs["verified_leaguepedia_series_crosswalk"] = {
        key: value
        for key, value in binding.items()
        if key != "assignments"
    }
    model_frame = _map_model_frame(
        result,
        verified_source_receipt_sha256=str(source_receipt["receipt_sha256"]),
        verified_source_receipt=source_receipt,
        verified_crosswalk_receipt_file_sha256=expected_receipt_file_sha256,
    )
    frame_ids = set(model_frame["game_id"].astype(str))
    accepted_ids = set(map(str, source_receipt["accepted_game_ids"]))
    eligible_ids = set(map(str, source_receipt["model_eligible_game_ids"]))
    extra_sources = source_receipt.get("source_extra_game_ids")
    map_extra_ids = set(
        map(
            str,
            extra_sources.get("maps", ())
            if isinstance(extra_sources, Mapping)
            else (),
        )
    )
    if not frame_ids.issubset(accepted_ids | map_extra_ids):
        raise FutureValueSourceError(
            "Leaguepedia crosswalk maps are outside the accepted census"
        )
    if accepted_ids.issubset(frame_ids):
        validation_frame = model_frame[
            model_frame["game_id"].astype(str).isin(eligible_ids)
        ].copy()
    elif frame_ids.issubset(eligible_ids):
        validation_frame = model_frame
    else:
        raise FutureValueSourceError(
            "Leaguepedia crosswalk maps mix eligible and excluded rows"
        )
    _validate_verified_source_receipt(
        source_receipt,
        validation_frame,
        require_full_eligible_set=set(validation_frame["game_id"].astype(str))
        == eligible_ids,
    )
    return result


def _file_record(
    value: Any,
    field: str,
    *,
    allow_missing: bool = False,
) -> dict[str, Any]:
    """Validate and return a byte-bound file record."""

    if not isinstance(value, Mapping):
        raise FutureValueSourceError(f"{field} file binding is invalid")
    if set(value) != {"path", "bytes", "sha256"}:
        raise FutureValueSourceError(f"{field} file binding schema is invalid")
    path = _safe_file_path(value.get("path"), field, allow_missing=allow_missing)
    if allow_missing and not path.exists():
        return {
            "path": str(path),
            "bytes": None,
            "sha256": None,
        }
    size = path.stat().st_size
    digest = _sha256_path(path)
    if isinstance(value.get("bytes"), bool):
        raise FutureValueSourceError(f"{field} byte count is invalid")
    try:
        expected_bytes = int(value["bytes"])
    except (TypeError, ValueError) as error:
        raise FutureValueSourceError(f"{field} byte count is invalid") from error
    expected_sha = str(value.get("sha256") or "").lower()
    if expected_bytes != size:
        raise FutureValueSourceError(f"{field} byte count changed")
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None or expected_sha != digest:
        raise FutureValueSourceError(f"{field} file hash changed")
    return {"path": str(path), "bytes": int(size), "sha256": digest}


def _load_feature_artifact(path: Path, feature_names: Sequence[str]) -> pd.DataFrame:
    """Load a producer artifact into the canonical game/value frame."""

    suffix = path.suffix.casefold()
    try:
        if suffix in {".parquet", ".pq"}:
            frame = pd.read_parquet(path)
        elif suffix == ".csv":
            frame = pd.read_csv(path)
        elif suffix in {".json", ".jsonl"}:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping) and isinstance(payload.get("rows"), list):
                payload = payload["rows"]
            if not isinstance(payload, list):
                raise FutureValueSourceError("rating feature artifact JSON rows are invalid")
            frame = pd.DataFrame(payload)
        else:
            raise FutureValueSourceError(
                "rating feature artifact format is unsupported; use parquet, CSV, or JSON"
            )
    except FutureValueSourceError:
        raise
    except Exception as error:
        raise FutureValueSourceError("rating feature artifact cannot be loaded") from error
    required = {"game_id", *feature_names}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise FutureValueSourceError(
            "rating feature artifact is missing: " + ", ".join(missing)
        )
    result = frame[list(dict.fromkeys(("game_id", *feature_names)))].copy()
    result["game_id"] = result["game_id"].astype(str)
    if result["game_id"].eq("").any() or result["game_id"].duplicated().any():
        raise FutureValueSourceError("rating feature artifact game IDs are not unique")
    for name in feature_names:
        values = pd.to_numeric(result[name], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise FutureValueSourceError(
                f"rating feature artifact contains missing values: {name}"
            )
        result[name] = values
    return result


def _compare_artifact_values(
    frame: pd.DataFrame,
    artifact: pd.DataFrame,
    feature_names: Sequence[str],
) -> None:
    """Require caller rows and independently loaded artifact rows to match."""

    left_ids = tuple(sorted(frame["game_id"].astype(str)))
    right_ids = tuple(sorted(artifact["game_id"].astype(str)))
    if left_ids != right_ids:
        raise FutureValueSourceError("rating feature artifact game identity changed")
    left = frame.copy()
    right = artifact.copy()
    left["game_id"] = left["game_id"].astype(str)
    right["game_id"] = right["game_id"].astype(str)
    left = left.sort_values("game_id", kind="stable").reset_index(drop=True)
    right = right.sort_values("game_id", kind="stable").reset_index(drop=True)
    for name in feature_names:
        left_values = pd.to_numeric(left[name], errors="coerce").to_numpy(dtype=float)
        right_values = pd.to_numeric(right[name], errors="coerce").to_numpy(dtype=float)
        if not np.array_equal(left_values, right_values, equal_nan=False):
            raise FutureValueSourceError(
                f"rating feature artifact values differ from caller frame: {name}"
            )


def _load_native_producer_artifact(path: Path, name: str) -> pd.DataFrame:
    """Load the producer-owned ledger used by a native receipt.

    Native receipts describe parquet ledgers emitted by the registered
    producers.  A narrow adapter artifact may contain only selected columns,
    while native validation needs the date and series columns as well.
    """

    if path.suffix.casefold() not in {".parquet", ".pq"}:
        raise FutureValueSourceError(f"{name} native artifact must be parquet")
    try:
        native = pd.read_parquet(path)
    except Exception as error:
        raise FutureValueSourceError(f"{name} native artifact cannot be loaded") from error
    if not isinstance(native, pd.DataFrame) or native.empty:
        raise FutureValueSourceError(f"{name} native artifact is empty")
    required = {"game_id", "date"}
    if name == "current_sequential_rating":
        required.add("series_id")
    missing = sorted(required - set(native.columns))
    if missing:
        raise FutureValueSourceError(
            f"{name} native artifact is missing: " + ", ".join(missing)
        )
    native = native.copy()
    native["game_id"] = native["game_id"].astype(str)
    native["date"] = pd.to_datetime(native["date"], utc=True, errors="coerce")
    if "series_id" in native.columns:
        native["series_id"] = native["series_id"].astype(str).str.strip()
    if (
        native["game_id"].eq("").any()
        or native["game_id"].duplicated().any()
        or native["date"].isna().any()
        or (
            "series_id" in native.columns
            and (
                native["series_id"].eq("").any()
                or native["series_id"].eq("nan").any()
            )
        )
    ):
        raise FutureValueSourceError(f"{name} native artifact identity is invalid")
    return native


def _scaling_native_rows_sha256(frame: pd.DataFrame) -> str:
    """Recompute the atomized scaling producer's full-row digest."""

    try:
        from lol_kills.research.atomized_rf_composite import (
            _scaling_json_value,
            _strict_canonical_sha256,
        )
    except Exception as error:  # pragma: no cover - import failure is an environment error
        raise FutureValueSourceError("scaling producer implementation is unavailable") from error
    ordered = frame.sort_values(["date", "game_id"], kind="stable")
    rows = [
        {
            str(column): _scaling_json_value(value)
            for column, value in row.items()
        }
        for row in ordered.to_dict("records")
    ]
    return str(_strict_canonical_sha256(rows)).lower()


def _validate_native_producer_receipt(
    name: str,
    receipt: Mapping[str, Any],
    *,
    native_artifact: pd.DataFrame,
    native_artifact_record: Mapping[str, Any],
    source_receipt: Mapping[str, Any],
    train_ids: Sequence[str],
    validation_ids: Sequence[str],
    cutoff: str,
    expected_features: Sequence[str],
) -> None:
    """Verify the producer-owned native receipt before adapter values bind."""

    if name == "current_sequential_rating":
        try:
            from lol_kills.research.future_value_rating_ledger import (
                validate_fold_current_rating_feature_ledger,
            )
            validate_fold_current_rating_feature_ledger(
                native_artifact,
                receipt,
                source_receipt=source_receipt,
                train_game_ids=train_ids,
                validation_game_ids=validation_ids,
                fit_window_end=cutoff,
            )
        except FutureValueSourceError:
            raise
        except Exception as error:
            raise FutureValueSourceError(
                "current rating native receipt failed validation"
            ) from error
        if receipt.get("artifact") != dict(native_artifact_record):
            raise FutureValueSourceError("current rating native artifact binding changed")
        frame_features = tuple(str(value) for value in receipt.get("feature_names", ()))
        if frame_features != tuple(expected_features):
            raise FutureValueSourceError("current rating native feature list changed")
        source_frames = receipt.get("source_frame_sha256")
        if not isinstance(source_frames, Mapping) or set(source_frames) != {
            "maps", "players", "teams"
        } or any(
            re.fullmatch(r"[0-9a-f]{64}", str(source_frames.get(label) or ""), re.I)
            is None
            for label in ("maps", "players", "teams")
        ):
            raise FutureValueSourceError("current rating native source frame binding is invalid")
        return

    if name != "strict_prior_atomized_scaling":
        raise FutureValueSourceError(f"unknown native producer: {name}")

    allowed = {
        "accepted_game_count",
        "accepted_game_ids",
        "authority",
        "checkpoint_targets",
        "columns",
        "evaluation_mode",
        "excluded_extra_game_count",
        "excluded_extra_identity_sha256",
        "fit_window_end",
        "fold_blocker",
        "fold_evaluation_usable",
        "implementation_sha256",
        "model_eligible_only",
        "model_excluded_game_count",
        "model_excluded_identity_sha256",
        "output_game_count",
        "output_game_ids",
        "output_identity_sha256",
        "public_authority",
        "receipt_sha256",
        "row_value_digest_sha256",
        "rows",
        "same_timestamp_batch_receipts",
        "same_timestamp_batching",
        "same_timestamp_policy",
        "source_as_of",
        "source_extra_game_ids",
        "source_frame_sha256",
        "source_identity_sha256",
        "source_receipt_sha256",
        "source_row_value_sha256",
        "schema_version",
        "status",
        "team_identity",
        "train_game_ids",
        "train_identity_sha256",
        "validation_game_ids",
        "validation_identity_sha256",
    }
    if set(receipt) != allowed:
        raise FutureValueSourceError("scaling native receipt schema is invalid")
    claimed_hash = receipt.get("receipt_sha256")
    payload = dict(receipt)
    payload.pop("receipt_sha256", None)
    if not isinstance(claimed_hash, str) or re.fullmatch(r"[0-9a-f]{64}", claimed_hash, re.I) is None:
        raise FutureValueSourceError("scaling native receipt hash is invalid")
    if hashlib.sha256(_canonical_json_bytes(payload)).hexdigest() != claimed_hash.lower():
        raise FutureValueSourceError("scaling native receipt hash changed")
    if receipt.get("schema_version") != "scryglass:atomized-scaling-feature-ledger:v1":
        raise FutureValueSourceError("scaling native receipt schema is invalid")
    if receipt.get("status") != "research_only" or receipt.get("authority") is not False:
        raise FutureValueSourceError("scaling native receipt authority is invalid")
    if receipt.get("public_authority") is not False:
        raise FutureValueSourceError("scaling native receipt public authority is invalid")
    if receipt.get("source_receipt_sha256") != source_receipt.get("receipt_sha256"):
        raise FutureValueSourceError("scaling native source receipt binding changed")
    if receipt.get("source_identity_sha256") != source_receipt.get("source_identity_sha256"):
        raise FutureValueSourceError("scaling native source identity changed")
    if receipt.get("source_as_of") != _utc_text(source_receipt["source_as_of"]):
        raise FutureValueSourceError("scaling native source cutoff changed")
    accepted_ids = tuple(str(value) for value in source_receipt["accepted_game_ids"])
    if tuple(str(value) for value in receipt.get("accepted_game_ids", ())) != accepted_ids:
        raise FutureValueSourceError("scaling native accepted census changed")
    if receipt.get("accepted_game_count") != len(accepted_ids):
        raise FutureValueSourceError("scaling native accepted count changed")
    model_ids = tuple(sorted(str(value) for value in native_artifact["game_id"]))
    if tuple(sorted(str(value) for value in receipt.get("output_game_ids", ()))) != model_ids:
        raise FutureValueSourceError("scaling native output census changed")
    if receipt.get("output_game_count") != len(model_ids) or receipt.get(
        "output_identity_sha256"
    ) != identity_sha256(model_ids):
        raise FutureValueSourceError("scaling native output identity changed")
    if receipt.get("rows") != len(native_artifact):
        raise FutureValueSourceError("scaling native row count changed")
    columns = tuple(str(value) for value in receipt.get("columns", ()))
    if columns != tuple(str(value) for value in native_artifact.columns):
        raise FutureValueSourceError("scaling native artifact columns changed")
    if any(feature not in native_artifact.columns for feature in expected_features):
        raise FutureValueSourceError("scaling native feature columns are incomplete")
    implementation_path = Path(__file__).resolve().parents[2] / (
        "lol_kills/research/atomized_rf_composite.py"
    )
    if receipt.get("implementation_sha256") != _sha256_path(implementation_path):
        raise FutureValueSourceError("scaling native implementation binding changed")
    source_frames = receipt.get("source_frame_sha256")
    if not isinstance(source_frames, Mapping) or set(source_frames) != {
        "maps", "players", "teams"
    } or any(
        re.fullmatch(r"[0-9a-f]{64}", str(source_frames.get(label) or ""), re.I)
        is None
        for label in ("maps", "players", "teams")
    ):
        raise FutureValueSourceError("scaling native source frame binding is invalid")
    if receipt.get("evaluation_mode") != "fold_local" or receipt.get("fold_evaluation_usable") is not True:
        raise FutureValueSourceError("scaling native evaluation mode is invalid")
    if tuple(str(value) for value in receipt.get("train_game_ids", ())) != tuple(train_ids):
        raise FutureValueSourceError("scaling native training IDs changed")
    if tuple(str(value) for value in receipt.get("validation_game_ids", ())) != tuple(validation_ids):
        raise FutureValueSourceError("scaling native validation IDs changed")
    if receipt.get("train_identity_sha256") != identity_sha256(train_ids) or receipt.get(
        "validation_identity_sha256"
    ) != identity_sha256(validation_ids):
        raise FutureValueSourceError("scaling native fold identity changed")
    if receipt.get("fit_window_end") != cutoff:
        raise FutureValueSourceError("scaling native fit cutoff changed")
    row_digest = str(receipt.get("row_value_digest_sha256") or "").lower()
    if re.fullmatch(r"[0-9a-f]{64}", row_digest, re.I) is None:
        raise FutureValueSourceError("scaling native row digest is invalid")
    if _scaling_native_rows_sha256(native_artifact) != row_digest:
        raise FutureValueSourceError("scaling native artifact values changed")
    if receipt.get("artifact") is not None:
        raise FutureValueSourceError("scaling native receipt has an unexpected artifact binding")


def _compare_native_frame_bindings(
    native_frame: pd.DataFrame,
    frame: pd.DataFrame,
    name: str,
) -> None:
    """Bind native output dates and optional series IDs to the fold frame."""

    left = native_frame[["game_id", "date"]].copy()
    right = frame[["game_id", "date"]].copy()
    left["game_id"] = left["game_id"].astype(str)
    right["game_id"] = right["game_id"].astype(str)
    left["date"] = pd.to_datetime(left["date"], utc=True, errors="coerce")
    right["date"] = pd.to_datetime(right["date"], utc=True, errors="coerce")
    left = left.sort_values("game_id", kind="stable").reset_index(drop=True)
    right = right.sort_values("game_id", kind="stable").reset_index(drop=True)
    if tuple(left["game_id"]) != tuple(right["game_id"]) or not left["date"].equals(
        right["date"]
    ):
        raise FutureValueSourceError(f"{name} native date binding changed")
    if "series_id" in native_frame.columns:
        native_series = native_frame[["game_id", "series_id"]].copy()
        frame_series = frame[["game_id", "series_id"]].copy()
        native_series["game_id"] = native_series["game_id"].astype(str)
        frame_series["game_id"] = frame_series["game_id"].astype(str)
        native_series["series_id"] = native_series["series_id"].astype(str).str.strip()
        frame_series["series_id"] = frame_series["series_id"].astype(str).str.strip()
        native_series = native_series.sort_values("game_id", kind="stable").reset_index(drop=True)
        frame_series = frame_series.sort_values("game_id", kind="stable").reset_index(drop=True)
        if not native_series.equals(frame_series):
            raise FutureValueSourceError(f"{name} native series binding changed")


def _producer_manifest_descriptors(
    producer: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], ...]:
    """Validate the durable producer manifest envelope."""

    if not isinstance(producer, Mapping):
        raise FutureValueSourceError(
            "file-backed rating feature producer manifest is required"
        )
    allowed = {
        "schema_version",
        "status",
        "authority",
        "adapters",
        "manifest_sha256",
    }
    if set(producer) != allowed:
        raise FutureValueSourceError(
            "synthetic or incomplete rating feature producer manifest is not allowed"
        )
    if producer.get("schema_version") != RATING_FEATURE_PRODUCER_MANIFEST_SCHEMA_VERSION:
        raise FutureValueSourceError("rating feature producer manifest schema is invalid")
    if producer.get("status") != "research_only":
        raise FutureValueSourceError("rating feature producer manifest status is invalid")
    if dict(producer.get("authority") or {}) != dict(RATING_FEATURE_PRODUCER_AUTHORITY):
        raise FutureValueSourceError("rating feature producer manifest authority is invalid")
    claimed = producer.get("manifest_sha256")
    payload = dict(producer)
    payload.pop("manifest_sha256", None)
    if not isinstance(claimed, str) or re.fullmatch(r"[0-9a-f]{64}", claimed) is None:
        raise FutureValueSourceError("rating feature producer manifest hash is invalid")
    if hashlib.sha256(_canonical_json_bytes(payload)).hexdigest() != claimed:
        raise FutureValueSourceError("rating feature producer manifest changed")
    raw_adapters = producer.get("adapters")
    if not isinstance(raw_adapters, (list, tuple)) or not raw_adapters:
        raise FutureValueSourceError("rating feature producer manifest adapters are missing")
    output: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw in raw_adapters:
        if not isinstance(raw, Mapping) or set(raw) != {
            "name", "artifact", "native_artifact", "receipt", "native_receipt"
        }:
            raise FutureValueSourceError("rating feature producer manifest adapter is invalid")
        name = str(raw.get("name") or "").strip()
        if name in names:
            raise FutureValueSourceError("rating feature producer adapters are duplicated")
        names.add(name)
        expected = _REGISTERED_FEATURE_PRODUCER_SPECS.get(name)
        if expected is None:
            raise FutureValueSourceError(f"unknown rating feature producer: {name}")
        artifact = _file_record(raw.get("artifact"), f"{name} artifact")
        native_artifact = _file_record(raw.get("native_artifact"), f"{name} native artifact")
        receipt = _file_record(raw.get("receipt"), f"{name} receipt")
        native_receipt = _file_record(raw.get("native_receipt"), f"{name} native receipt")
        output.append(
            {
                "name": name,
                "artifact": artifact,
                "native_artifact": native_artifact,
                "receipt": receipt,
                "native_receipt": native_receipt,
                "expected": expected,
            }
        )
    return tuple(output)


def _verify_durable_producer_adapters(
    producer: Mapping[str, Any] | None,
    *,
    frame: pd.DataFrame,
    source_receipt: Mapping[str, Any],
    source_identity: str,
    source_hash: str,
    train_ids: Sequence[str],
    validation_ids: Sequence[str],
    cutoff: str,
    config_features: Sequence[str],
    evaluation_mode: str,
) -> tuple[dict[str, Any], ...]:
    """Load and verify every durable producer artifact independently."""

    if evaluation_mode != "fold_local":
        raise FutureValueSourceError("rating feature producer evaluation mode is invalid")
    descriptors = _producer_manifest_descriptors(producer)
    verified: list[dict[str, Any]] = []
    for descriptor in descriptors:
        name = str(descriptor["name"])
        expected = dict(descriptor["expected"])
        receipt_path = _safe_file_path(descriptor["receipt"]["path"], f"{name} receipt")
        try:
            receipt_value = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise FutureValueSourceError(f"{name} producer receipt cannot be loaded") from error
        if not isinstance(receipt_value, Mapping):
            raise FutureValueSourceError(f"{name} producer receipt is invalid")
        receipt = dict(receipt_value)
        allowed_receipt = {
            "schema_version",
            "status",
            "authority",
            "producer",
            "artifact",
            "native_artifact",
            "native_receipt",
            "source_identity_sha256",
            "source_receipt_sha256",
            "source_receipt_file",
            "fit_game_ids",
            "fit_game_identity_sha256",
            "validation_game_ids",
            "validation_game_identity_sha256",
            "fit_window_end",
            "evaluation_mode",
            "feature_names",
            "row_values_sha256",
            "receipt_sha256",
        }
        if set(receipt) != allowed_receipt:
            raise FutureValueSourceError(f"{name} producer receipt schema is invalid")
        claimed_receipt_hash = receipt.pop("receipt_sha256", None)
        if not isinstance(claimed_receipt_hash, str) or re.fullmatch(
            r"[0-9a-f]{64}", claimed_receipt_hash
        ) is None:
            raise FutureValueSourceError(f"{name} producer receipt self hash is invalid")
        if hashlib.sha256(_canonical_json_bytes(receipt)).hexdigest() != claimed_receipt_hash:
            raise FutureValueSourceError(f"{name} producer receipt self hash changed")
        if receipt.get("schema_version") != RATING_FEATURE_PRODUCER_RECEIPT_SCHEMA_VERSION:
            raise FutureValueSourceError(f"{name} producer receipt schema is invalid")
        if receipt.get("status") != "research_only" or dict(
            receipt.get("authority") or {}
        ) != dict(RATING_FEATURE_PRODUCER_AUTHORITY):
            raise FutureValueSourceError(f"{name} producer receipt authority is invalid")
        if dict(receipt.get("producer") or {}) != expected:
            raise FutureValueSourceError(f"{name} producer implementation binding changed")
        if receipt.get("source_identity_sha256") != source_identity:
            raise FutureValueSourceError(f"{name} producer source identity changed")
        if receipt.get("source_receipt_sha256") != source_hash:
            raise FutureValueSourceError(f"{name} producer source receipt changed")
        if receipt.get("fit_window_end") != cutoff:
            raise FutureValueSourceError(f"{name} producer cutoff changed")
        if receipt.get("evaluation_mode") != evaluation_mode:
            raise FutureValueSourceError(f"{name} producer evaluation mode changed")
        if tuple(str(value) for value in receipt.get("fit_game_ids", ())) != tuple(train_ids):
            raise FutureValueSourceError(f"{name} producer training IDs changed")
        if tuple(str(value) for value in receipt.get("validation_game_ids", ())) != tuple(
            validation_ids
        ):
            raise FutureValueSourceError(f"{name} producer validation IDs changed")
        if receipt.get("fit_game_identity_sha256") != identity_sha256(train_ids):
            raise FutureValueSourceError(f"{name} producer training identity changed")
        if receipt.get("validation_game_identity_sha256") != identity_sha256(validation_ids):
            raise FutureValueSourceError(f"{name} producer validation identity changed")
        expected_features = tuple(str(value) for value in expected["feature_names"])
        if tuple(str(value) for value in receipt.get("feature_names", ())) != expected_features:
            raise FutureValueSourceError(f"{name} producer feature list changed")
        artifact_record = _file_record(receipt.get("artifact"), f"{name} artifact")
        if artifact_record != descriptor["artifact"]:
            raise FutureValueSourceError(f"{name} producer artifact binding changed")
        native_artifact_record = _file_record(
            receipt.get("native_artifact"), f"{name} native artifact"
        )
        if native_artifact_record != descriptor["native_artifact"]:
            raise FutureValueSourceError(f"{name} producer native artifact binding changed")
        native_receipt_record = _file_record(
            receipt.get("native_receipt"), f"{name} native receipt"
        )
        if native_receipt_record != descriptor["native_receipt"]:
            raise FutureValueSourceError(f"{name} producer native receipt binding changed")
        if receipt.get("source_receipt_file") is not None:
            source_file = _file_record(receipt["source_receipt_file"], f"{name} source receipt")
            try:
                source_payload = json.loads(
                    Path(source_file["path"]).read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as error:
                raise FutureValueSourceError(f"{name} source receipt file cannot be loaded") from error
            _verified_identity, verified_hash = _verified_source_receipt_for_ledger(source_payload)
            if verified_hash != source_hash or source_payload != dict(source_receipt):
                raise FutureValueSourceError(f"{name} source receipt file binding changed")
        artifact_path = _safe_file_path(artifact_record["path"], f"{name} artifact")
        artifact_frame = _load_feature_artifact(artifact_path, expected_features)
        native_artifact_record = descriptor["native_artifact"]
        native_artifact_path = _safe_file_path(
            native_artifact_record["path"], f"{name} native artifact"
        )
        native_artifact_frame = _load_native_producer_artifact(native_artifact_path, name)
        native_receipt_path = _safe_file_path(
            descriptor["native_receipt"]["path"], f"{name} native receipt"
        )
        try:
            native_receipt_value = json.loads(
                native_receipt_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FutureValueSourceError(
                f"{name} native receipt cannot be loaded"
            ) from error
        if not isinstance(native_receipt_value, Mapping):
            raise FutureValueSourceError(f"{name} native receipt is invalid")
        _validate_native_producer_receipt(
            name,
            native_receipt_value,
            native_artifact=native_artifact_frame,
            native_artifact_record=native_artifact_record,
            source_receipt=source_receipt,
            train_ids=train_ids,
            validation_ids=validation_ids,
            cutoff=cutoff,
            expected_features=expected_features,
        )
        _compare_native_frame_bindings(native_artifact_frame, frame, name)
        _compare_artifact_values(native_artifact_frame, artifact_frame, expected_features)
        if set(artifact_frame["game_id"].astype(str)) != set(frame["game_id"].astype(str)):
            raise FutureValueSourceError(f"{name} producer artifact game identity changed")
        row_digest = _ledger_rows_sha256(artifact_frame, expected_features)
        claimed_row_digest = str(receipt.get("row_values_sha256") or "").lower()
        if claimed_row_digest != row_digest:
            raise FutureValueSourceError(f"{name} producer artifact values changed")
        _compare_artifact_values(frame, artifact_frame, expected_features)
        code_receipt = trusted_feature_producer_receipt(
            name,
            row_values_sha256=row_digest,
        )
        verified.append(
            {
                "name": name,
                "feature_names": list(expected_features),
                "row_values_sha256": row_digest,
                "artifact": artifact_record,
                "native_artifact": native_artifact_record,
                "receipt": descriptor["receipt"],
                "native_receipt": descriptor["native_receipt"],
                "receipt_sha256": claimed_receipt_hash,
                "code_receipt": code_receipt,
            }
        )
    feature_names = tuple(
        feature for adapter in verified for feature in adapter["feature_names"]
    )
    if len(set(feature_names)) != len(feature_names) or set(feature_names) != set(config_features):
        raise FutureValueSourceError("rating feature producer adapters do not match features")
    return tuple(verified)


def _safe_output_file_path(value: Any, field: str) -> Path:
    """Validate a new receipt destination before a producer writes it."""

    path = _safe_file_path(value, field, allow_missing=True)
    if path.exists():
        raise FutureValueSourceError(f"{field} output already exists")
    return path


def write_rating_feature_producer_receipt(
    name: str,
    artifact_path: Path | str,
    receipt_path: Path | str,
    *,
    native_artifact_path: Path | str,
    native_receipt_path: Path | str,
    source_receipt: Mapping[str, Any],
    train_game_ids: Iterable[str],
    validation_game_ids: Iterable[str],
    fit_window_end: Any,
    evaluation_mode: str = "fold_local",
    source_receipt_path: Path | str | None = None,
) -> dict[str, Any]:
    """Write one canonical receipt for a file-backed producer artifact.

    The selected adapter artifact is paired with the producer's full native
    ledger and its native receipt.  The native pair is mandatory.  This keeps
    a caller from creating a finite table and then self-sealing it as a
    current-rating or scaling output.
    """

    key = str(name).strip()
    expected = _REGISTERED_FEATURE_PRODUCER_SPECS.get(key)
    if expected is None:
        raise FutureValueSourceError(f"unknown rating feature producer: {name}")
    if evaluation_mode != "fold_local":
        raise FutureValueSourceError("rating feature producer evaluation mode is invalid")
    source_identity, source_hash = _verified_source_receipt_for_ledger(source_receipt)
    artifact_file = _safe_file_path(artifact_path, f"{key} artifact")
    artifact = _file_record(
        {
            "path": str(artifact_file),
            "bytes": artifact_file.stat().st_size,
            "sha256": _sha256_path(artifact_file),
        },
        f"{key} artifact",
    )
    native_artifact_file = _safe_file_path(
        native_artifact_path, f"{key} native artifact"
    )
    native_artifact = _file_record(
        {
            "path": str(native_artifact_file),
            "bytes": native_artifact_file.stat().st_size,
            "sha256": _sha256_path(native_artifact_file),
        },
        f"{key} native artifact",
    )
    feature_names = tuple(str(value) for value in expected["feature_names"])
    native_frame = _load_native_producer_artifact(native_artifact_file, key)
    artifact_frame = _load_feature_artifact(Path(artifact["path"]), feature_names)
    model_ids = tuple(sorted(artifact_frame["game_id"].astype(str)))
    train_ids = tuple(sorted({str(value) for value in train_game_ids}))
    validation_ids = tuple(sorted({str(value) for value in validation_game_ids}))
    if not train_ids or not validation_ids or set(train_ids) & set(validation_ids):
        raise FutureValueSourceError("file-backed producer fold IDs are invalid")
    if tuple(sorted((*train_ids, *validation_ids))) != model_ids:
        raise FutureValueSourceError("file-backed producer fold IDs do not match artifact")
    native_ids = tuple(sorted(native_frame["game_id"].astype(str)))
    if native_ids != model_ids:
        raise FutureValueSourceError("file-backed native producer IDs do not match artifact")
    cutoff = _utc_text(fit_window_end)
    source_file = None
    if source_receipt_path is not None:
        source_receipt_file = _safe_file_path(source_receipt_path, f"{key} source receipt")
        source_file = _file_record(
            {
                "path": str(source_receipt_file),
                "bytes": source_receipt_file.stat().st_size,
                "sha256": _sha256_path(source_receipt_file),
            },
            f"{key} source receipt",
        )
        payload = json.loads(Path(source_file["path"]).read_text(encoding="utf-8"))
        _verified_identity, verified_hash = _verified_source_receipt_for_ledger(payload)
        if verified_hash != source_hash or dict(payload) != dict(source_receipt):
            raise FutureValueSourceError(f"{key} source receipt file does not match source")
    native_receipt_file = _safe_file_path(native_receipt_path, f"{key} native receipt")
    native_receipt_record = _file_record(
        {
            "path": str(native_receipt_file),
            "bytes": native_receipt_file.stat().st_size,
            "sha256": _sha256_path(native_receipt_file),
        },
        f"{key} native receipt",
    )
    try:
        native_payload = json.loads(native_receipt_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FutureValueSourceError(f"{key} native receipt cannot be loaded") from error
    if not isinstance(native_payload, Mapping):
        raise FutureValueSourceError(f"{key} native receipt is invalid")
    cutoff = _utc_text(fit_window_end)
    _validate_native_producer_receipt(
        key,
        native_payload,
        native_artifact=native_frame,
        native_artifact_record=native_artifact,
        source_receipt=source_receipt,
        train_ids=train_ids,
        validation_ids=validation_ids,
        cutoff=cutoff,
        expected_features=feature_names,
    )
    _compare_artifact_values(native_frame, artifact_frame, feature_names)
    payload: dict[str, Any] = {
        "schema_version": RATING_FEATURE_PRODUCER_RECEIPT_SCHEMA_VERSION,
        "status": "research_only",
        "authority": dict(RATING_FEATURE_PRODUCER_AUTHORITY),
        "producer": dict(expected),
        "artifact": artifact,
        "native_artifact": native_artifact,
        "native_receipt": native_receipt_record,
        "source_identity_sha256": source_identity,
        "source_receipt_sha256": source_hash,
        "source_receipt_file": source_file,
        "fit_game_ids": list(train_ids),
        "fit_game_identity_sha256": identity_sha256(train_ids),
        "validation_game_ids": list(validation_ids),
        "validation_game_identity_sha256": identity_sha256(validation_ids),
        "fit_window_end": cutoff,
        "evaluation_mode": evaluation_mode,
        "feature_names": list(feature_names),
        "row_values_sha256": _ledger_rows_sha256(artifact_frame, feature_names),
    }
    payload["receipt_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    destination = _safe_output_file_path(receipt_path, f"{key} receipt")
    destination.write_bytes(_canonical_json_bytes(payload))
    receipt_record = _file_record(
        {
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": _sha256_path(destination),
        },
        f"{key} receipt",
    )
    return {
        "name": key,
        "artifact": artifact,
        "native_artifact": native_artifact,
        "receipt": receipt_record,
        "native_receipt": native_receipt_record,
    }


def build_rating_feature_producer_manifest(
    adapters: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the immutable manifest consumed by rating ledger binding."""

    normalized = []
    for adapter in adapters:
        if not isinstance(adapter, Mapping):
            raise FutureValueSourceError("rating feature producer adapter is invalid")
        if set(adapter) != {
            "name", "artifact", "native_artifact", "receipt", "native_receipt"
        }:
            raise FutureValueSourceError("rating feature producer adapter schema is invalid")
        normalized.append(
            {
                "name": str(adapter["name"]),
                "artifact": _file_record(adapter["artifact"], f"{adapter['name']} artifact"),
                "native_artifact": _file_record(
                    adapter["native_artifact"], f"{adapter['name']} native artifact"
                ),
                "receipt": _file_record(adapter["receipt"], f"{adapter['name']} receipt"),
                "native_receipt": _file_record(
                    adapter["native_receipt"], f"{adapter['name']} native receipt"
                ),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": RATING_FEATURE_PRODUCER_MANIFEST_SCHEMA_VERSION,
        "status": "research_only",
        "authority": dict(RATING_FEATURE_PRODUCER_AUTHORITY),
        "adapters": normalized,
    }
    if not normalized:
        raise FutureValueSourceError("rating feature producer manifest is empty")
    payload["manifest_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return payload


def _resolve_rating_variant(value: RatingVariant | str) -> RatingVariant:
    if isinstance(value, RatingVariant):
        return value
    if isinstance(value, str):
        try:
            return RatingVariant(value.strip())
        except ValueError as error:
            raise FutureValueSourceError(f"unknown rating variant: {value}") from error
    raise FutureValueSourceError(f"unknown rating variant: {value!r}")


def _rating_feature_name_reason(name: Any) -> str | None:
    """Return a fail-closed reason for a proposed model feature name."""

    text = str(name).strip()
    if not text:
        return "empty feature name"
    normalized = re.sub(r"[^a-z0-9]", "", text.casefold())
    if any(
        token in normalized
        for token in (
            "target",
            "outcome",
            "result",
            "winner",
            "observed",
            "actual",
            "final",
            "censored",
        )
    ):
        return "target-like or final-state feature name"
    checkpoint = re.search(
        r"(?:gold|xp|cs|kills|assists|deaths)(?:diff|difference)?(?:at|diff|difference)?"
        r"(?:10|15|20|25)$",
        normalized,
    )
    if checkpoint and not normalized.startswith("forecast"):
        return "current-checkpoint feature name"
    final_metric = {
        "cspm",
        "cspermin",
        "dpm",
        "damageshare",
        "damagepermin",
        "kills",
        "deaths",
        "assists",
        "totalgold",
        "earnedgold",
        "gamelength",
        "duration",
    }
    if normalized in final_metric:
        return "final-map feature name"
    return None


def classify_rating_feature(name: str) -> str:
    """Classify one canonical feature as signed-map, side-level, or derived."""

    text = str(name)
    if text in CURRENT_RATING_SIGNED_MAP_FEATURES or text in SCALING_CURVE_SIGNED_MAP_FEATURES:
        return "signed_map"
    if text in FUTURE_PLAYER_FORM_SIDE_FEATURES:
        return "side_level"
    if text in _RATING_TEAM_CONTEXT_FEATURES:
        return "team_context"
    if text in _RATING_DERIVED_FEATURES:
        return "derived"
    return "unknown"


def is_signed_map_feature(name: str) -> bool:
    return classify_rating_feature(name) == "signed_map"


def is_side_level_feature(name: str) -> bool:
    return classify_rating_feature(name) == "side_level"


def assert_rating_feature_names(feature_names: Iterable[str]) -> tuple[str, ...]:
    """Validate one exact model feature family list.

    Feature lists are closed over the declarations above.  This helper does
    not permit callers to add an unregistered feature, a current checkpoint,
    or a target/final-state column.
    """

    try:
        names = tuple(str(name) for name in feature_names)
    except TypeError as error:
        raise FutureValueSourceError("rating feature names must be iterable") from error
    if len(set(names)) != len(names):
        raise FutureValueSourceError("rating feature names contain duplicates")
    for name in names:
        reason = _rating_feature_name_reason(name)
        if reason is not None:
            raise FutureValueSourceError(f"rating feature is forbidden: {name} ({reason})")
        classification = classify_rating_feature(name)
        if classification == "derived":
            raise FutureValueSourceError(
                f"rating derived diagnostic is not a model feature: {name}"
            )
        if classification == "team_context":
            raise FutureValueSourceError(
                f"rating team context is excluded from the player-form family: {name}"
            )
        if name not in _RATING_MODEL_FEATURE_UNIVERSE:
            raise FutureValueSourceError(f"rating feature is not registered: {name}")
    return names


@dataclass(frozen=True)
class RatingVariantConfig:
    """Immutable, canonical feature selection for one rating variant."""

    variant: RatingVariant
    feature_names: tuple[str, ...]
    signed_map_features: tuple[str, ...]
    side_level_features: tuple[str, ...]
    excluded_features: tuple[str, ...] = SCALING_CURVE_DERIVED_FEATURES

    def __post_init__(self) -> None:
        variant = _resolve_rating_variant(self.variant)
        object.__setattr__(self, "variant", variant)
        names = tuple(self.feature_names)
        signed = tuple(self.signed_map_features)
        side = tuple(self.side_level_features)
        excluded = tuple(self.excluded_features)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "signed_map_features", signed)
        object.__setattr__(self, "side_level_features", side)
        object.__setattr__(self, "excluded_features", excluded)
        assert_rating_feature_names(names)
        assert_rating_feature_names(signed)
        assert_rating_feature_names(side)
        if set(names) != set((*signed, *side)):
            raise FutureValueSourceError(
                f"rating variant {variant.value} feature families do not match feature_names"
            )
        if names != tuple(_RATING_VARIANT_FEATURES[variant]):
            raise FutureValueSourceError(
                f"rating variant {variant.value} does not use its canonical feature list"
            )
        if signed != tuple(_RATING_VARIANT_SIGNED_FEATURES[variant]):
            raise FutureValueSourceError(
                f"rating variant {variant.value} signed-map family changed"
            )
        if side != tuple(_RATING_VARIANT_SIDE_FEATURES[variant]):
            raise FutureValueSourceError(
                f"rating variant {variant.value} side-level family changed"
            )
        if excluded != SCALING_CURVE_DERIVED_FEATURES:
            raise FutureValueSourceError("rating variant excluded diagnostics changed")

    @property
    def feature_families(self) -> Mapping[str, tuple[str, ...]]:
        """Return the immutable family selection used by this variant."""

        selected: dict[str, tuple[str, ...]] = {}
        if self.signed_map_features:
            selected["current_rating"] = CURRENT_RATING_SIGNED_MAP_FEATURES
        if self.side_level_features:
            selected["future_player_form"] = FUTURE_PLAYER_FORM_SIDE_FEATURES
        if any(
            feature in SCALING_CURVE_SIGNED_MAP_FEATURES
            for feature in self.signed_map_features
        ):
            selected["scaling_curve"] = SCALING_CURVE_SIGNED_MAP_FEATURES
        return MappingProxyType(selected)

    @property
    def name(self) -> str:
        return self.variant.value

    @property
    def ordinal(self) -> int:
        return int(RATING_VARIANT_ORDINALS[self.variant])

    @property
    def label(self) -> str:
        return f"V{self.ordinal}"

    @property
    def config_sha256(self) -> str:
        return rating_variant_config_sha256(self.variant)

    def receipt(self) -> dict[str, Any]:
        return rating_variant_config_receipt(self.variant)


def rating_variant_config(variant: RatingVariant | str) -> RatingVariantConfig:
    """Return one immutable registered variant configuration."""

    resolved = _resolve_rating_variant(variant)
    return RATING_VARIANT_CONFIGS[resolved]


def get_rating_variant_config(
    variant: RatingVariant | str,
    feature_names: Iterable[str] | None = None,
) -> RatingVariantConfig:
    """Return a config and reject any caller-supplied feature-list mutation."""

    config = rating_variant_config(variant)
    if feature_names is not None:
        candidate = assert_rating_feature_names(feature_names)
        if candidate != config.feature_names:
            raise FutureValueSourceError(
                f"arbitrary feature list is not allowed for rating variant {config.variant.value}"
            )
    return config


def assert_rating_variant_features(
    variant: RatingVariant | str,
    feature_names: Iterable[str],
) -> tuple[str, ...]:
    """Require a feature list to equal the registered variant exactly."""

    return get_rating_variant_config(variant, feature_names).feature_names


def rating_variant_config_receipt(variant: RatingVariant | str) -> dict[str, Any]:
    """Return a canonical receipt for one registered variant."""

    config = rating_variant_config(variant)
    payload: dict[str, Any] = {
        "schema_version": RATING_VARIANT_SCHEMA_VERSION,
        "ordinal": config.ordinal,
        "label": config.label,
        "variant": config.variant.value,
        "feature_names": list(config.feature_names),
        "signed_map_features": list(config.signed_map_features),
        "side_level_features": list(config.side_level_features),
        "excluded_features": list(config.excluded_features),
        "current_rating_feature_semantics": dict(CURRENT_RATING_FEATURE_SEMANTICS),
        "scaling_curve_producer": {
            "schema_version": SCALING_CURVE_PRODUCER_SCHEMA_VERSION,
            "model_version": SCALING_CURVE_PRODUCER_VERSION,
            "feature_declaration": list(SCALING_CURVE_FEATURE_DECLARATION),
            "shape_features": list(SCALING_CURVE_SHAPE_FEATURES),
            "signed_shape_features": list(SCALING_CURVE_SIGNED_SHAPE_FEATURES),
            "invariant_shape_features": list(SCALING_CURVE_INVARIANT_SHAPE_FEATURES),
        },
    }
    payload["config_sha256"] = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    payload["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    return payload


def rating_variant_config_sha256(variant: RatingVariant | str) -> str:
    return str(rating_variant_config_receipt(variant)["config_sha256"])


def rating_variant_registry_receipt() -> dict[str, Any]:
    """Return one canonical receipt binding all four variant configs."""

    payload: dict[str, Any] = {
        "schema_version": RATING_VARIANT_SCHEMA_VERSION,
        "variants": [rating_variant_config_receipt(variant) for variant in RATING_VARIANT_ORDER],
    }
    payload["registry_sha256"] = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    payload["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    return payload


def rating_variant_registry_sha256() -> str:
    return str(rating_variant_registry_receipt()["registry_sha256"])


def rating_variant_configs() -> Mapping[RatingVariant, RatingVariantConfig]:
    """Return the immutable four-variant registry."""

    return RATING_VARIANT_CONFIGS


RATING_VARIANT_CONFIGS = MappingProxyType(
    {
        variant: RatingVariantConfig(
            variant=variant,
            feature_names=tuple(_RATING_VARIANT_FEATURES[variant]),
            signed_map_features=tuple(_RATING_VARIANT_SIGNED_FEATURES[variant]),
            side_level_features=tuple(_RATING_VARIANT_SIDE_FEATURES[variant]),
        )
        for variant in RATING_VARIANT_ORDER
    }
)


def _ledger_rows_sha256(frame: pd.DataFrame, feature_names: Sequence[str]) -> str:
    rows: list[dict[str, Any]] = []
    for row in frame.sort_values("game_id", kind="stable").itertuples(index=False):
        values = {str(name): getattr(row, str(name)) for name in ("game_id", *feature_names)}
        canonical: dict[str, Any] = {"game_id": str(values.pop("game_id"))}
        for name in feature_names:
            value = values[name]
            number = float(value)
            if not math.isfinite(number):
                raise FutureValueSourceError(f"rating feature ledger contains a non-finite value: {name}")
            canonical[name] = number
        rows.append(canonical)
    return hashlib.sha256(_canonical_json_bytes(rows)).hexdigest()


def rating_feature_values_sha256(
    frame: pd.DataFrame,
    feature_names: Iterable[str],
) -> str:
    """Hash canonical feature values for an independently bound producer."""

    names = assert_rating_feature_names(feature_names)
    return _ledger_rows_sha256(frame, names)


def bind_rating_feature_ledger(
    frame: pd.DataFrame,
    *,
    source_receipt: Mapping[str, Any],
    train_game_ids: Iterable[str],
    fit_window_end: Any,
    feature_names: Iterable[str],
    producer: Mapping[str, Any] | None = None,
    validation_game_ids: Iterable[str] | None = None,
    evaluation_mode: str = "fold_local",
) -> pd.DataFrame:
    """Bind one fold-bound, out-of-sample signed-feature ledger.

    The returned frame carries its receipt in ``DataFrame.attrs``.  The frame
    must contain one row for every map in the fold's model frame.  The producer
    receipt records the training census and strict cutoff.  A caller cannot
    replace a feature list or source identity after binding.
    """

    if not isinstance(frame, pd.DataFrame):
        raise FutureValueSourceError("rating feature ledger must be a DataFrame")
    config_features = assert_rating_feature_names(feature_names)
    if not config_features or any(
        classify_rating_feature(name) != "signed_map" for name in config_features
    ):
        raise FutureValueSourceError("rating feature ledger accepts signed map features only")
    source_identity, source_hash = _verified_source_receipt_for_ledger(source_receipt)
    required = {"game_id", "date", "series_id", *config_features}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise FutureValueSourceError("rating feature ledger is missing: " + ", ".join(missing))
    work = frame[
        list(dict.fromkeys(("game_id", "date", "series_id", *config_features)))
    ].copy()
    work["game_id"] = work["game_id"].astype(str)
    if work["game_id"].eq("").any() or work["game_id"].duplicated().any():
        raise FutureValueSourceError("rating feature ledger game IDs are not unique")
    dates = pd.to_datetime(work["date"], utc=True, errors="coerce")
    if dates.isna().any():
        raise FutureValueSourceError("rating feature ledger dates are invalid")
    work["date"] = dates
    work["series_id"] = work["series_id"].astype(str).str.strip()
    if work["series_id"].eq("").any() or work["series_id"].eq("nan").any():
        raise FutureValueSourceError("rating feature ledger series IDs are invalid")
    for name in config_features:
        values = pd.to_numeric(work[name], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise FutureValueSourceError(f"rating feature ledger contains missing values: {name}")
        work[name] = values
    train_ids = tuple(sorted({str(value) for value in train_game_ids}))
    if not train_ids or not set(train_ids).issubset(set(work["game_id"])):
        raise FutureValueSourceError("rating ledger training game IDs are incomplete")
    model_ids = tuple(sorted(work["game_id"].astype(str)))
    if validation_game_ids is None:
        validation_ids = tuple(sorted(set(model_ids) - set(train_ids)))
    else:
        validation_ids = tuple(sorted({str(value) for value in validation_game_ids}))
    if not validation_ids or set(validation_ids) != set(model_ids) - set(train_ids):
        raise FutureValueSourceError(
            "rating ledger validation game IDs do not match the model frame"
        )
    cutoff = _utc_text(fit_window_end)
    cutoff_stamp = _utc_timestamp(cutoff, "fit_window_end")
    train_dates = dates[work["game_id"].isin(train_ids)]
    if train_dates.empty or not bool(train_dates.lt(cutoff_stamp).all()):
        raise FutureValueSourceError("rating ledger training rows violate strict-prior cutoff")
    train_series = set(work.loc[work["game_id"].isin(train_ids), "series_id"])
    validation_series = set(
        work.loc[work["game_id"].isin(validation_ids), "series_id"]
    )
    if train_series & validation_series:
        raise FutureValueSourceError("rating ledger train and validation series overlap")
    producer_adapters = _verify_durable_producer_adapters(
        producer,
        frame=work,
        source_receipt=source_receipt,
        source_identity=source_identity,
        source_hash=source_hash,
        train_ids=train_ids,
        validation_ids=validation_ids,
        cutoff=cutoff,
        config_features=config_features,
        evaluation_mode=evaluation_mode,
    )
    producer_feature_names = tuple(
        feature
        for adapter in producer_adapters
        for feature in adapter["feature_names"]
    )
    if len(set(producer_feature_names)) != len(producer_feature_names):
        raise FutureValueSourceError("rating feature producer feature families overlap")
    if set(producer_feature_names) != set(config_features):
        raise FutureValueSourceError(
            "rating feature producer feature families do not match ledger features"
        )
    rows_hash = _ledger_rows_sha256(work, config_features)
    producer_payload: dict[str, Any] = {
        "schema_version": RATING_FEATURE_LEDGER_SCHEMA_VERSION,
        "source_identity_sha256": source_identity,
        "source_receipt_sha256": source_hash,
        "game_identity_sha256": identity_sha256(model_ids),
        "fit_game_identity_sha256": identity_sha256(train_ids),
        "fit_game_ids": list(train_ids),
        "fit_date_min": _utc_text(train_dates.min()),
        "fit_date_max": _utc_text(train_dates.max()),
        "fit_window_end": cutoff,
        "validation_game_identity_sha256": identity_sha256(validation_ids),
        "validation_game_ids": list(validation_ids),
        "feature_names": list(config_features),
        "ledger_rows_sha256": rows_hash,
        "feature_value_digest": rows_hash,
        "evaluation_mode": evaluation_mode,
        "strict_prior_timing": "fit_rows_strictly_before_cutoff",
        "same_timestamp_policy": "batch_exclude_same_timestamp",
        "series_safety": {
            "policy": "whole_series_disjoint",
            "train_series_identity_sha256": identity_sha256(tuple(sorted(train_series))),
            "validation_series_identity_sha256": identity_sha256(
                tuple(sorted(validation_series))
            ),
        },
        "producer_adapters": [dict(adapter["code_receipt"]) for adapter in producer_adapters],
        "producer_artifacts": dict(producer or {}),
    }
    producer_payload["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(producer_payload)
    ).hexdigest()
    work.attrs["schema_version"] = RATING_FEATURE_LEDGER_SCHEMA_VERSION
    work.attrs["source_identity_sha256"] = source_identity
    work.attrs["source_receipt_sha256"] = source_hash
    work.attrs["fit_game_ids"] = list(train_ids)
    work.attrs["fit_game_identity_sha256"] = identity_sha256(train_ids)
    work.attrs["game_identity_sha256"] = identity_sha256(model_ids)
    work.attrs["fit_window_end"] = cutoff
    work.attrs["fit_date_min"] = producer_payload["fit_date_min"]
    work.attrs["fit_date_max"] = producer_payload["fit_date_max"]
    work.attrs["validation_game_ids"] = list(validation_ids)
    work.attrs["validation_game_identity_sha256"] = producer_payload[
        "validation_game_identity_sha256"
    ]
    work.attrs["feature_names"] = list(config_features)
    work.attrs["ledger_rows_sha256"] = rows_hash
    work.attrs["ledger_sha256"] = rows_hash
    work.attrs["feature_value_digest"] = rows_hash
    work.attrs["strict_prior_timing"] = producer_payload["strict_prior_timing"]
    work.attrs["same_timestamp_policy"] = producer_payload["same_timestamp_policy"]
    work.attrs["evaluation_mode"] = evaluation_mode
    work.attrs["series_safety"] = producer_payload["series_safety"]
    work.attrs["producer_receipt"] = producer_payload
    work.attrs["producer_receipt_sha256"] = producer_payload["receipt_sha256"]
    return work


def validate_rating_feature_ledger(
    ledger: pd.DataFrame | None,
    *,
    feature_names: Iterable[str],
    model_game_ids: Iterable[str],
    train_game_ids: Iterable[str],
    fit_window_end: Any,
    source_receipt: Mapping[str, Any] | None,
    evaluation_mode: str = "fold_local",
) -> pd.DataFrame:
    """Verify a fold-bound external ledger before it enters a design matrix."""

    config_features = assert_rating_feature_names(feature_names)
    if not config_features or any(
        classify_rating_feature(name) != "signed_map" for name in config_features
    ):
        raise FutureValueSourceError("rating feature ledger accepts signed map features only")
    if ledger is None:
        raise FutureValueSourceError("rating feature ledger is required for signed map features")
    if not isinstance(ledger, pd.DataFrame):
        raise FutureValueSourceError("rating feature ledger must be a DataFrame")
    required = {"game_id", "date", "series_id", *config_features}
    missing = sorted(required - set(ledger.columns))
    if missing:
        raise FutureValueSourceError("rating feature ledger is missing: " + ", ".join(missing))
    work = ledger.copy()
    work["game_id"] = work["game_id"].astype(str)
    model_ids = tuple(sorted({str(value) for value in model_game_ids}))
    train_ids = tuple(sorted({str(value) for value in train_game_ids}))
    ledger_ids = tuple(sorted(work["game_id"]))
    if ledger["game_id"].duplicated().any() or ledger_ids != model_ids:
        raise FutureValueSourceError("rating feature ledger game IDs do not match the model frame")
    attrs = work.attrs
    if attrs.get("schema_version") != RATING_FEATURE_LEDGER_SCHEMA_VERSION:
        raise FutureValueSourceError("rating feature ledger schema is invalid")
    source_identity = str(attrs.get("source_identity_sha256") or "")
    expected_source_identity = str((source_receipt or {}).get("source_identity_sha256") or "")
    if not source_identity or source_identity != expected_source_identity:
        raise FutureValueSourceError("rating feature ledger source identity does not match source receipt")
    _verified_identity, expected_receipt_hash = _verified_source_receipt_for_ledger(
        source_receipt
    )
    if attrs.get("source_receipt_sha256") != expected_receipt_hash:
        raise FutureValueSourceError("rating feature ledger source receipt binding changed")
    if tuple(str(value) for value in attrs.get("feature_names", ())) != config_features:
        raise FutureValueSourceError("rating feature ledger feature list is not canonical")
    attr_train_ids = tuple(sorted({str(value) for value in attrs.get("fit_game_ids", ())}))
    if attr_train_ids != train_ids:
        raise FutureValueSourceError("rating feature ledger training IDs do not match fold")
    if attrs.get("fit_game_identity_sha256") != identity_sha256(train_ids):
        raise FutureValueSourceError("rating feature ledger fit identity does not match fold")
    if attrs.get("game_identity_sha256") != identity_sha256(model_ids):
        raise FutureValueSourceError("rating feature ledger game identity does not match frame")
    validation_ids = tuple(sorted({str(value) for value in attrs.get("validation_game_ids", ())}))
    expected_validation_ids = tuple(sorted(set(model_ids) - set(train_ids)))
    if validation_ids != expected_validation_ids:
        raise FutureValueSourceError("rating feature ledger validation IDs do not match frame")
    if attrs.get("validation_game_identity_sha256") != identity_sha256(validation_ids):
        raise FutureValueSourceError("rating feature ledger validation identity does not match frame")
    expected_cutoff = _utc_text(fit_window_end)
    if str(attrs.get("fit_window_end") or "") != expected_cutoff:
        raise FutureValueSourceError("rating feature ledger cutoff does not match fold")
    if attrs.get("evaluation_mode") != evaluation_mode:
        raise FutureValueSourceError("rating feature ledger evaluation mode changed")
    producer = attrs.get("producer_receipt")
    if not isinstance(producer, Mapping):
        raise FutureValueSourceError("rating feature ledger producer receipt is missing")
    producer_payload = dict(producer)
    producer_hash = producer_payload.pop("receipt_sha256", None)
    if not isinstance(producer_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", producer_hash):
        raise FutureValueSourceError("rating feature ledger producer hash is invalid")
    if hashlib.sha256(_canonical_json_bytes(producer_payload)).hexdigest() != producer_hash:
        raise FutureValueSourceError("rating feature ledger producer receipt changed")
    if str(attrs.get("producer_receipt_sha256") or "") != producer_hash:
        raise FutureValueSourceError("rating feature ledger producer hash binding changed")
    if producer.get("schema_version") != RATING_FEATURE_LEDGER_SCHEMA_VERSION:
        raise FutureValueSourceError("rating feature ledger producer schema is invalid")
    if producer.get("source_identity_sha256") != source_identity:
        raise FutureValueSourceError("rating feature ledger producer source identity changed")
    if producer.get("source_receipt_sha256") != expected_receipt_hash:
        raise FutureValueSourceError("rating feature ledger producer source receipt changed")
    if tuple(str(value) for value in producer.get("fit_game_ids", ())) != train_ids:
        raise FutureValueSourceError("rating feature ledger producer training IDs changed")
    if producer.get("fit_window_end") != expected_cutoff:
        raise FutureValueSourceError("rating feature ledger producer cutoff changed")
    if producer.get("evaluation_mode") != evaluation_mode:
        raise FutureValueSourceError("rating feature ledger producer evaluation mode changed")
    if tuple(str(value) for value in producer.get("feature_names", ())) != config_features:
        raise FutureValueSourceError("rating feature ledger producer features changed")
    if tuple(str(value) for value in producer.get("validation_game_ids", ())) != validation_ids:
        raise FutureValueSourceError("rating feature ledger producer validation IDs changed")
    if producer.get("validation_game_identity_sha256") != identity_sha256(validation_ids):
        raise FutureValueSourceError("rating feature ledger producer validation identity changed")
    if producer.get("strict_prior_timing") != "fit_rows_strictly_before_cutoff":
        raise FutureValueSourceError("rating feature ledger strict-prior timing policy changed")
    if producer.get("same_timestamp_policy") != "batch_exclude_same_timestamp":
        raise FutureValueSourceError("rating feature ledger same-timestamp policy changed")
    producer_series_safety = producer.get("series_safety")
    if not isinstance(producer_series_safety, Mapping) or producer_series_safety.get(
        "policy"
    ) != "whole_series_disjoint":
        raise FutureValueSourceError("rating feature ledger series safety policy is invalid")
    dates = pd.to_datetime(work["date"], utc=True, errors="coerce")
    if dates.isna().any():
        raise FutureValueSourceError("rating feature ledger dates are invalid")
    cutoff_stamp = _utc_timestamp(expected_cutoff, "fit_window_end")
    fit_dates = dates[work["game_id"].isin(train_ids)]
    if fit_dates.empty or not bool(fit_dates.lt(cutoff_stamp).all()):
        raise FutureValueSourceError("rating feature ledger training rows violate strict-prior cutoff")
    train_series = set(work.loc[work["game_id"].isin(train_ids), "series_id"].astype(str))
    validation_series = set(
        work.loc[work["game_id"].isin(validation_ids), "series_id"].astype(str)
    )
    if train_series & validation_series:
        raise FutureValueSourceError("rating feature ledger train and validation series overlap")
    if producer_series_safety.get("train_series_identity_sha256") != identity_sha256(
        tuple(sorted(train_series))
    ) or producer_series_safety.get("validation_series_identity_sha256") != identity_sha256(
        tuple(sorted(validation_series))
    ):
        raise FutureValueSourceError("rating feature ledger series safety binding changed")
    durable_adapters = _verify_durable_producer_adapters(
        producer.get("producer_artifacts"),
        frame=work,
        source_receipt=source_receipt,
        source_identity=source_identity,
        source_hash=expected_receipt_hash,
        train_ids=train_ids,
        validation_ids=validation_ids,
        cutoff=expected_cutoff,
        config_features=config_features,
        evaluation_mode=evaluation_mode,
    )
    durable_names = tuple(str(adapter["name"]) for adapter in durable_adapters)
    expected_adapters = _verified_producer_adapters(
        {"adapters": producer.get("producer_adapters", ())}
    )
    expected_adapter_names = tuple(str(adapter["name"]) for adapter in expected_adapters)
    if not expected_adapter_names:
        raise FutureValueSourceError("rating feature ledger producer adapters are missing")
    if durable_names != expected_adapter_names:
        raise FutureValueSourceError("rating feature ledger producer adapter binding changed")
    adapter_features = tuple(
        feature for adapter in expected_adapters for feature in adapter["feature_names"]
    )
    if set(adapter_features) != set(config_features) or len(adapter_features) != len(config_features):
        raise FutureValueSourceError("rating feature ledger producer adapters do not match features")
    for adapter in expected_adapters:
        adapter_rows_hash = _ledger_rows_sha256(
            work, tuple(str(value) for value in adapter["feature_names"])
        )
        if adapter.get("row_values_sha256") != adapter_rows_hash:
            raise FutureValueSourceError(
                f"rating feature ledger producer feature values changed: {adapter['name']}"
            )
    rows_hash = _ledger_rows_sha256(work, config_features)
    claimed_rows_hash = str(attrs.get("ledger_rows_sha256") or attrs.get("ledger_sha256") or "")
    if claimed_rows_hash != rows_hash or producer.get("ledger_rows_sha256") != rows_hash:
        raise FutureValueSourceError("rating feature ledger row hash changed")
    if producer.get("feature_value_digest") != rows_hash or attrs.get(
        "feature_value_digest"
    ) != rows_hash:
        raise FutureValueSourceError("rating feature ledger feature values changed")
    if producer.get("game_identity_sha256") != identity_sha256(model_ids):
        raise FutureValueSourceError("rating feature ledger game identity changed")
    if producer.get("fit_game_identity_sha256") != identity_sha256(train_ids):
        raise FutureValueSourceError("rating feature ledger fit identity changed")
    row_bindings = {
        "source_identity_sha256": source_identity,
        "producer_receipt_sha256": producer_hash,
        "fit_window_end": expected_cutoff,
        "fit_game_identity_sha256": identity_sha256(train_ids),
        "game_identity_sha256": identity_sha256(model_ids),
    }
    for column, expected in row_bindings.items():
        if column in work.columns:
            values = work[column].astype(str)
            if not values.eq(str(expected)).all():
                raise FutureValueSourceError(f"rating feature ledger row binding changed: {column}")
    for name in config_features:
        values = pd.to_numeric(work[name], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise FutureValueSourceError(f"rating feature ledger contains missing values: {name}")
        work[name] = values
    result = work[["game_id", "date", "series_id", *config_features]].copy()
    result.attrs = dict(attrs)
    return result


def _side_level_column(side: str, feature: str) -> str:
    return f"__{side}_{feature}"


def build_future_value_design(
    maps: pd.DataFrame,
    form: pd.DataFrame,
    atom_model: Rank3AtomModel,
    *,
    verified_model_frame: pd.DataFrame | None = None,
    variant: RatingVariant | str | None = None,
    feature_ledger: pd.DataFrame | None = None,
    source_receipt: Mapping[str, Any] | None = None,
    train_game_ids: Iterable[str] | None = None,
    fit_window_end: Any | None = None,
) -> pd.DataFrame:
    """Build side-neutral map differences from pregame state only."""

    map_frame = (
        verified_model_frame.copy()
        if verified_model_frame is not None
        else _map_model_frame(maps)
    )
    variant_config = (
        None if variant is None else rating_variant_config(variant)
    )
    if variant_config is not None:
        if train_game_ids is None or fit_window_end is None:
            raise FutureValueSourceError(
                "explicit rating variants require fold training IDs and cutoff"
            )
        signed_ledger = validate_rating_feature_ledger(
            feature_ledger,
            feature_names=variant_config.signed_map_features,
            model_game_ids=map_frame["game_id"].astype(str),
            train_game_ids=train_game_ids,
            fit_window_end=fit_window_end,
            source_receipt=source_receipt,
        )
    else:
        signed_ledger = None
    required = {"game_id", "date", "side", "role", "player_id", *[f"prior_form_{m}" for m in FORM_METRICS]}
    missing = sorted(required - set(form.columns))
    if missing:
        raise FutureValueSourceError("future-value form is missing columns: " + ", ".join(missing))
    work = form.copy()
    work["game_id"] = work["game_id"].astype(str)
    work["side"] = work["side"].astype(str).str.casefold()
    work["role"] = work["role"].map(_role)
    if work["role"].isna().any():
        raise FutureValueSourceError("future-value form has an unknown role")
    map_ids = set(map_frame["game_id"].astype(str))
    form_ids = set(work["game_id"])
    if form_ids != map_ids:
        missing_games = sorted(map_ids - form_ids)
        extra_games = sorted(form_ids - map_ids)
        raise FutureValueSourceError(
            "future-value form game IDs do not match maps"
            f" (missing={len(missing_games)}, extra={len(extra_games)})"
        )
    if work[["game_id", "side", "role"]].duplicated().any():
        raise FutureValueSourceError("future-value form has duplicate game-side-role rows")
    atoms = atom_model.transform(work)
    work = pd.concat([work.reset_index(drop=True), atoms.reset_index(drop=True)], axis=1)
    side_feature_names = [
        *[f"prior_form_{metric}" for metric in FORM_METRICS],
        *[f"rank_3_player_atom_{index}" for index in range(1, RANK_3 + 1)],
        *[f"rank_3_champion_role_atom_{index}" for index in range(1, RANK_3 + 1)],
    ]
    form_metric_names = [f"prior_form_{metric}" for metric in FORM_METRICS]
    group_key = work[["game_id", "side"]]
    counts = group_key.groupby(["game_id", "side"], sort=False, observed=True).size()
    if not counts.eq(5).all():
        raise FutureValueSourceError("future-value design requires exact five-player sides")
    side_sets = {
        (str(game_id), str(side)): set(group["role"])
        for (game_id, side), group in work.groupby(["game_id", "side"], sort=False, observed=True)
    }
    expected_roles = set(ROLES)
    if any(roles != expected_roles for roles in side_sets.values()):
        raise FutureValueSourceError("future-value design requires one complete role set per side")
    side_values = work[["game_id", "side", *side_feature_names]].copy()
    side_values[side_feature_names] = side_values[side_feature_names].apply(
        pd.to_numeric, errors="coerce"
    )
    side_values[side_feature_names] = side_values[side_feature_names].mask(
        ~np.isfinite(side_values[side_feature_names])
    )
    grouped_values = side_values.groupby(
        ["game_id", "side"], sort=False, observed=True
    )
    champion_atom_columns = [
        f"rank_3_champion_role_atom_{index}" for index in range(1, RANK_3 + 1)
    ]
    champion_atom_missing = (
        work[champion_atom_columns].isna()
        .groupby([work["game_id"], work["side"]], sort=False, observed=True)
        .mean()
        .mean(axis=1)
    )
    side_means = grouped_values[side_feature_names].sum(min_count=5)
    side_finite_counts = grouped_values[side_feature_names].count()
    side_means = side_means.mask(side_finite_counts.lt(5))
    side_wide = side_means.unstack("side")
    side_missing = (
        side_values[form_metric_names].isna()
        .groupby([side_values["game_id"], side_values["side"]], sort=False, observed=True)
        .mean()
        .mean(axis=1)
    )
    support_columns = [f"prior_form_{metric}_support" for metric in FORM_METRICS]
    effective_support_columns = [
        f"prior_form_{metric}_effective_support" for metric in FORM_METRICS
    ]
    support_values = work[["game_id", "side", *support_columns, *effective_support_columns]].copy()
    support_values[support_columns + effective_support_columns] = support_values[
        support_columns + effective_support_columns
    ].apply(pd.to_numeric, errors="coerce")
    support_values[support_columns + effective_support_columns] = support_values[
        support_columns + effective_support_columns
    ].mask(~np.isfinite(support_values[support_columns + effective_support_columns]))
    support_summary = support_values.groupby(
        ["game_id", "side"], sort=False, observed=True
    )[[*support_columns, *effective_support_columns]].mean()
    player_atom_columns = [
        f"rank_3_player_atom_{index}" for index in range(1, RANK_3 + 1)
    ]
    rank_missing = (
        work[player_atom_columns].isna()
        .groupby([work["game_id"], work["side"]], sort=False, observed=True)
        .mean()
        .mean(axis=1)
    )
    rank_support = work[["game_id", "side", "rank_3_champion_role_support"]].copy()
    rank_support["rank_3_champion_role_support"] = pd.to_numeric(
        rank_support["rank_3_champion_role_support"], errors="coerce"
    )
    rank_support = rank_support.groupby(
        ["game_id", "side"], sort=False, observed=True
    )["rank_3_champion_role_support"].mean().unstack("side")
    team_state = _team_history_features(map_frame, work).drop_duplicates(
        ["game_id", "side"], keep="first"
    )
    team_wide = team_state.set_index(["game_id", "side"])[
        ["prior_team_win", "roster_continuity"]
    ].unstack("side")
    design = map_frame[["game_id", "date", "series_id", "target"]].copy()
    design = design.set_index("game_id", drop=False)

    def put_side_levels(source_name: str, levels: pd.DataFrame) -> None:
        model_name = SIDE_LEVEL_TO_MODEL_FEATURE[source_name]
        blue = levels.get("blue", pd.Series(dtype=float)).reindex(design.index)
        red = levels.get("red", pd.Series(dtype=float)).reindex(design.index)
        design[_side_level_column("blue", source_name)] = blue
        design[_side_level_column("red", source_name)] = red
        design[model_name] = blue.sub(red)

    for source_name in side_feature_names:
        output_name = source_name.replace("prior_form_", "player_form_")
        put_side_levels(output_name, side_wide[source_name])
    put_side_levels("team_prior_win", team_wide["prior_team_win"])
    put_side_levels("roster_continuity", team_wide["roster_continuity"])
    design["blue_roster_continuity"] = (
        team_wide["roster_continuity"].get("blue", pd.Series(dtype=float))
        .reindex(design.index)
    )
    design["red_roster_continuity"] = (
        team_wide["roster_continuity"].get("red", pd.Series(dtype=float))
        .reindex(design.index)
    )
    put_side_levels("player_form_missing_rate", side_missing.unstack("side"))
    put_side_levels("rank_3_atom_missing_rate", rank_missing.unstack("side"))
    put_side_levels(
        "rank_3_champion_role_atom_missing_rate",
        champion_atom_missing.unstack("side"),
    )
    support_mean = support_summary[support_columns].mean(axis=1).unstack("side")
    effective_support_mean = support_summary[effective_support_columns].mean(axis=1).unstack(
        "side"
    )
    put_side_levels("player_form_support_mean", support_mean)
    put_side_levels("player_form_effective_support_mean", effective_support_mean)
    design["player_form_support_mean"] = support_mean.mean(axis=1).reindex(
        design.index
    )
    design["player_form_effective_support_mean"] = effective_support_mean.mean(
        axis=1
    ).reindex(design.index)
    design["player_form_support_uncertainty_proxy"] = (
        1.0 / np.sqrt(1.0 + design["player_form_effective_support_mean"])
    )
    design["rank_3_atom_support_uncertainty_proxy"] = (
        1.0 / np.sqrt(1.0 + rank_support.mean(axis=1).reindex(design.index))
    )
    minimum_metric_support = support_values.groupby(
        "game_id", sort=False, observed=True
    )[support_columns].min().min(axis=1)
    minimum_effective_support = support_values.groupby(
        "game_id", sort=False, observed=True
    )[effective_support_columns].min().min(axis=1)
    minimum_atom_support = rank_support.min(axis=1)
    design["player_form_minimum_metric_support"] = minimum_metric_support.reindex(
        design.index
    )
    design["player_form_minimum_effective_support"] = (
        minimum_effective_support.reindex(design.index)
    )
    design["rank_3_champion_role_minimum_support"] = minimum_atom_support.reindex(
        design.index
    )
    raw_side_columns = [
        _side_level_column(side, source_name)
        for source_name in SIDE_LEVEL_TO_MODEL_FEATURE
        for side in SIDES
    ]
    design["model_missing_feature_names"] = [
        sorted(
            source_name
            for source_name in SIDE_LEVEL_TO_MODEL_FEATURE
            if not (
                np.isfinite(row[_side_level_column("blue", source_name)])
                and np.isfinite(row[_side_level_column("red", source_name)])
            )
        )
        for _, row in design.iterrows()
    ]
    design["model_missing_feature_count"] = design[
        "model_missing_feature_names"
    ].map(len)
    if variant_config is None:
        complete_columns = raw_side_columns
    else:
        complete_columns = []
        for feature in variant_config.feature_names:
            if feature in variant_config.signed_map_features:
                complete_columns.append(feature)
            else:
                complete_columns.extend(
                    [_side_level_column("blue", feature), _side_level_column("red", feature)]
                )
    if signed_ledger is not None:
        signed_lookup = signed_ledger.set_index("game_id")
        for feature in variant_config.signed_map_features:
            design[feature] = signed_lookup[feature].reindex(design.index)
    design["model_features_complete"] = np.isfinite(
        design[complete_columns].to_numpy(dtype=float)
    ).all(axis=1)
    for metadata_name in (
        "league",
        "competition_scope",
        "patch",
        "oe_patch_token",
        "tournament",
    ):
        if metadata_name in map_frame.columns:
            design[metadata_name] = (
                map_frame.set_index("game_id")[metadata_name].reindex(design.index)
            )
    if "tournament" in map_frame.columns:
        tournament = design["tournament"].astype("string").str.strip()
        tournament_key = tournament.fillna("")
        ordered = design.assign(_tournament_key=tournament_key).sort_values(
            "date", kind="stable"
        )
        boundary = ordered["_tournament_key"].ne(ordered["_tournament_key"].shift(1))
        boundary &= ordered["_tournament_key"].ne("")
        design["tournament_boundary"] = boundary.reindex(design.index).fillna(False).astype(bool)
    selected_feature_names = (
        MODEL_FEATURES if variant_config is None else variant_config.feature_names
    )
    design = design.reset_index(drop=True)
    assert_pregame_feature_names(selected_feature_names)
    design.attrs["feature_names"] = selected_feature_names
    design.attrs["variant"] = variant_config.variant.value if variant_config is not None else None
    design.attrs["variant_receipt"] = (
        variant_config.receipt() if variant_config is not None else None
    )
    if signed_ledger is not None:
        design.attrs["feature_ledger"] = {
            "schema_version": RATING_FEATURE_LEDGER_SCHEMA_VERSION,
            "source_identity_sha256": signed_ledger.attrs.get("source_identity_sha256"),
            "source_receipt_sha256": signed_ledger.attrs.get("source_receipt_sha256"),
            "producer_receipt_sha256": signed_ledger.attrs.get("producer_receipt_sha256"),
            "ledger_rows_sha256": signed_ledger.attrs.get("ledger_rows_sha256"),
            "feature_value_digest": signed_ledger.attrs.get("feature_value_digest"),
            "fit_window_end": signed_ledger.attrs.get("fit_window_end"),
            "fit_date_min": signed_ledger.attrs.get("fit_date_min"),
            "fit_date_max": signed_ledger.attrs.get("fit_date_max"),
            "fit_game_ids": list(signed_ledger.attrs.get("fit_game_ids", ())),
            "validation_game_ids": list(
                signed_ledger.attrs.get("validation_game_ids", ())
            ),
            "validation_game_identity_sha256": signed_ledger.attrs.get(
                "validation_game_identity_sha256"
            ),
            "strict_prior_timing": signed_ledger.attrs.get("strict_prior_timing"),
            "same_timestamp_policy": signed_ledger.attrs.get(
                "same_timestamp_policy"
            ),
            "series_safety": signed_ledger.attrs.get("series_safety"),
            "feature_names": list(variant_config.signed_map_features),
        }
    design.attrs["series_cluster_source"] = map_frame.attrs.get("series_cluster_source")
    design.attrs["series_cluster_audit"] = map_frame.attrs.get("series_cluster_audit")
    return design


def _fold_level_imputation_values(
    train: pd.DataFrame,
    feature_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Fit one side-neutral imputation value per side-level feature."""

    selected = tuple(
        SIDE_LEVEL_TO_MODEL_FEATURE if feature_names is None else feature_names
    )
    if any(name not in SIDE_LEVEL_TO_MODEL_FEATURE for name in selected):
        raise FutureValueSourceError("fold imputation received a non-side-level feature")
    values: list[float] = []
    for source_name in selected:
        columns = [
            _side_level_column("blue", source_name),
            _side_level_column("red", source_name),
        ]
        missing = sorted(set(columns) - set(train.columns))
        if missing:
            raise FutureValueSourceError(
                "imputation design is missing: " + ", ".join(missing)
            )
        pooled = train[columns].apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=float
        )
        finite = pooled[np.isfinite(pooled)]
        if finite.size:
            values.append(float(np.median(finite)))
        elif source_name in CENTERED_ATOM_LEVEL_FEATURES:
            values.append(0.0)
        else:
            raise FutureValueSourceError(
                f"non-centered imputation feature is all missing: {source_name}"
            )
    return np.asarray(values, dtype=float)


def _antisymmetric_design_matrix(
    design: pd.DataFrame,
    imputation_values: np.ndarray,
    feature_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Build blue-minus-red values after equal fold-local side imputation."""

    imputation = np.asarray(imputation_values, dtype=float)
    selected = tuple(feature_names or SIDE_LEVEL_TO_MODEL_FEATURE)
    if imputation.shape != (len(selected),) or not np.isfinite(
        imputation
    ).all():
        raise FutureValueSourceError("fold-local imputation values are invalid")
    columns: list[np.ndarray] = []
    inverse_side_names = {
        value: key for key, value in SIDE_LEVEL_TO_MODEL_FEATURE.items()
    }
    for feature_index, source_name in enumerate(selected):
        if classify_rating_feature(source_name) == "signed_map":
            if source_name not in design.columns:
                raise FutureValueSourceError("prediction design is missing: " + source_name)
            numeric = pd.to_numeric(design[source_name], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(numeric).all():
                raise FutureValueSourceError("signed map feature contains missing values: " + source_name)
            columns.append(numeric)
            continue
        source_key = source_name if source_name in SIDE_LEVEL_TO_MODEL_FEATURE else inverse_side_names.get(source_name)
        if source_key is None:
            raise FutureValueSourceError("prediction design contains an unknown feature: " + source_name)
        side_values: list[np.ndarray] = []
        for side in SIDES:
            column = _side_level_column(side, source_key)
            if column not in design.columns:
                raise FutureValueSourceError("prediction design is missing: " + column)
            numeric = pd.to_numeric(design[column], errors="coerce").to_numpy(
                dtype=float
            )
            side_values.append(
                np.where(np.isfinite(numeric), numeric, imputation[feature_index])
            )
        columns.append(side_values[0] - side_values[1])
    matrix = np.column_stack(columns)
    if not np.isfinite(matrix).all():
        raise FutureValueSourceError("antisymmetric design matrix is non-finite")
    return matrix


def _variant_imputation_values(
    train: pd.DataFrame,
    config: RatingVariantConfig,
) -> np.ndarray:
    side_values = _fold_level_imputation_values(train, config.side_level_features)
    side_lookup = dict(zip(config.side_level_features, side_values))
    return np.asarray(
        [0.0 if classify_rating_feature(name) == "signed_map" else side_lookup[name] for name in config.feature_names],
        dtype=float,
    )


def build_rating_variant_matrix(
    design: pd.DataFrame,
    variant: RatingVariant | str,
    *,
    imputation_values: np.ndarray | None = None,
) -> np.ndarray:
    """Build the selected fold matrix from a verified design frame."""

    config = rating_variant_config(variant)
    values = (
        _variant_imputation_values(design, config)
        if imputation_values is None
        else np.asarray(imputation_values, dtype=float)
    )
    return _antisymmetric_design_matrix(
        design,
        values,
        feature_names=config.feature_names,
    )


def _fit_zero_intercept_logistic(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    regularization_c: float,
) -> tuple[LogisticRegression, dict[str, Any]]:
    max_iterations = 1000
    classifier = LogisticRegression(
        C=float(regularization_c),
        penalty="l2",
        solver="lbfgs",
        fit_intercept=False,
        max_iter=max_iterations,
        random_state=0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            classifier.fit(matrix, target)
    convergence_messages = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    iterations = [int(value) for value in np.asarray(classifier.n_iter_).ravel()]
    finite = bool(np.isfinite(classifier.coef_).all())
    converged = bool(
        finite
        and not convergence_messages
        and iterations
        and max(iterations) < max_iterations
    )
    evidence = {
        "solver": "lbfgs",
        "success": converged,
        "finite_coefficients": finite,
        "iterations": iterations,
        "max_iterations": max_iterations,
        "convergence_warnings": convergence_messages,
        "regularization_c": float(regularization_c),
        "coefficient_sha256": hashlib.sha256(
            _canonical_json_bytes(classifier.coef_.astype(float).tolist())
        ).hexdigest(),
    }
    if not converged:
        raise FutureValueSourceError("future-value classifier did not converge")
    return classifier, evidence


def _select_fold_regularization(
    map_frame: pd.DataFrame,
    form: pd.DataFrame,
    *,
    train_game_ids: tuple[str, ...],
    rank: int,
    min_cell_support: int,
    variant: RatingVariant | None = None,
    feature_ledger: pd.DataFrame | None = None,
    inner_feature_ledger: pd.DataFrame | None = None,
    source_receipt: Mapping[str, Any] | None = None,
    fit_window_end: Any | None = None,
) -> dict[str, Any]:
    """Select L2 strength on one strictly earlier nested chronological fold."""

    if variant is not None:
        config = rating_variant_config(variant)
        if inner_feature_ledger is None:
            # The outer producer ledger is valid for the outer fit only.
            # Reusing it in an inner selector would make the selector see
            # future producer state.  Keep the fixed value as an explicit
            # research blocker until a separately bound inner ledger exists.
            return {
                "method": "predeclared_regularization_c",
                "candidate_grid": [float(PREDECLARED_VARIANT_REGULARIZATION_C)],
                "selected_c": float(PREDECLARED_VARIANT_REGULARIZATION_C),
                "inner_ledger_status": "missing",
                "blockers": ["nested_inner_feature_ledger_missing_fixed_c_used"],
                "outer_ledger_reuse": False,
                "selection_scope": "predeclared_before_outer_fold",
                "variant": config.variant.value,
            }

        outer_train_maps = map_frame[map_frame["game_id"].isin(train_game_ids)].copy()
        outer_train_ids = tuple(sorted(str(value) for value in train_game_ids))
        outer_model_ids = tuple(sorted(outer_train_maps["game_id"].astype(str)))
        if outer_model_ids != outer_train_ids:
            raise FutureValueSourceError("nested selector outer training census is incomplete")
        inner_fold = chronological_whole_series_folds(
            outer_train_maps,
            n_folds=1,
            verified_model_frame=outer_train_maps,
        )[0]
        inner_train_ids = tuple(str(value) for value in inner_fold["train_game_ids"])
        inner_validation_ids = tuple(
            str(value) for value in inner_fold["validation_game_ids"]
        )
        if set(inner_train_ids) & set(inner_validation_ids):
            raise FutureValueSourceError("nested selector train and validation IDs overlap")
        if set(inner_train_ids) | set(inner_validation_ids) != set(outer_train_ids):
            raise FutureValueSourceError("nested selector does not cover the outer training census")

        # An inner producer ledger is a separate source-bound artifact.  Its
        # model census is exactly the outer training census.  This proves that
        # no outer validation row, feature value, or producer state entered
        # candidate-C selection.
        inner_ledger = validate_rating_feature_ledger(
            inner_feature_ledger,
            feature_names=config.signed_map_features,
            model_game_ids=outer_train_ids,
            train_game_ids=inner_train_ids,
            fit_window_end=inner_fold["validation_start"],
            source_receipt=source_receipt,
        )
        if set(inner_ledger["game_id"].astype(str)) & set(
            map_frame.loc[~map_frame["game_id"].isin(outer_train_ids), "game_id"].astype(str)
        ):
            raise FutureValueSourceError("nested inner feature ledger contains outer validation IDs")
        inner_dates = pd.to_datetime(inner_ledger["date"], utc=True, errors="coerce")
        if inner_dates.isna().any():
            raise FutureValueSourceError("nested inner feature ledger has invalid dates")
        inner_cutoff = _utc_timestamp(inner_fold["validation_start"], "inner validation start")
        if not bool(inner_dates.loc[inner_ledger["game_id"].isin(inner_train_ids)].lt(inner_cutoff).all()):
            raise FutureValueSourceError("nested inner feature ledger training rows violate cutoff")
        if not bool(inner_dates.loc[inner_ledger["game_id"].isin(inner_validation_ids)].ge(inner_cutoff).all()):
            raise FutureValueSourceError("nested inner feature ledger validation rows violate cutoff")

        inner_form = form[form["game_id"].astype(str).isin(outer_train_ids)].copy()
        if set(inner_form["game_id"].astype(str)) != set(outer_train_ids):
            raise FutureValueSourceError("nested inner form does not cover the outer training census")
        atom_model = fit_rank3_player_champion_role_atoms(
            inner_form,
            train_game_ids=inner_train_ids,
            rank=rank,
            min_cell_support=min_cell_support,
            fit_window_end=inner_fold["validation_start"],
        )
        inner_design = build_future_value_design(
            outer_train_maps,
            inner_form,
            atom_model,
            verified_model_frame=outer_train_maps,
            variant=config.variant,
            feature_ledger=inner_ledger,
            source_receipt=source_receipt,
            train_game_ids=inner_train_ids,
            fit_window_end=inner_fold["validation_start"],
        )
        inner_train = inner_design[inner_design["game_id"].isin(inner_train_ids)].copy()
        inner_validation = inner_design[
            inner_design["game_id"].isin(inner_validation_ids)
        ].copy()
        train_target = pd.to_numeric(inner_train["target"], errors="coerce")
        validation_target = pd.to_numeric(inner_validation["target"], errors="coerce")
        if (
            len(inner_train) < 10
            or len(inner_validation) < 10
            or train_target.nunique() != 2
            or validation_target.nunique() != 2
        ):
            raise FutureValueSourceError(
                "nested regularization fold has insufficient two-class rows"
            )
        selected_features = tuple(config.feature_names)
        imputation = _variant_imputation_values(inner_train, config)
        matrix = _antisymmetric_design_matrix(
            inner_train,
            imputation,
            feature_names=selected_features,
        )
        validation_matrix = _antisymmetric_design_matrix(
            inner_validation,
            imputation,
            feature_names=selected_features,
        )
        scales = matrix.std(axis=0, ddof=0)
        scales = np.where(np.isfinite(scales) & (scales > 1e-12), scales, 1.0)
        candidate_scores: list[dict[str, Any]] = []
        inner_atom_receipt = atom_model.parameter_receipt()
        inner_feature_binding = {
            "schema_version": RATING_FEATURE_LEDGER_SCHEMA_VERSION,
            "source_identity_sha256": inner_ledger.attrs.get("source_identity_sha256"),
            "source_receipt_sha256": inner_ledger.attrs.get("source_receipt_sha256"),
            "producer_receipt_sha256": inner_ledger.attrs.get("producer_receipt_sha256"),
            "ledger_rows_sha256": inner_ledger.attrs.get("ledger_rows_sha256"),
            "feature_value_digest": inner_ledger.attrs.get("feature_value_digest"),
            "feature_names": list(config.signed_map_features),
            "game_identity_sha256": inner_ledger.attrs.get("game_identity_sha256"),
            "fit_game_identity_sha256": inner_ledger.attrs.get("fit_game_identity_sha256"),
            "validation_game_identity_sha256": inner_ledger.attrs.get(
                "validation_game_identity_sha256"
            ),
            "fit_window_end": inner_ledger.attrs.get("fit_window_end"),
            "fit_date_min": inner_ledger.attrs.get("fit_date_min"),
            "fit_date_max": inner_ledger.attrs.get("fit_date_max"),
            "validation_date_min": _utc_text(
                inner_dates.loc[inner_ledger["game_id"].isin(inner_validation_ids)].min()
            ),
            "validation_date_max": _utc_text(
                inner_dates.loc[inner_ledger["game_id"].isin(inner_validation_ids)].max()
            ),
            "fit_game_ids": list(inner_ledger.attrs.get("fit_game_ids", ())),
            "validation_game_ids": list(inner_ledger.attrs.get("validation_game_ids", ())),
            "producer_artifacts": dict(
                inner_ledger.attrs.get("producer_receipt", {}).get(
                    "producer_artifacts", {}
                )
                or {}
            ),
        }
        inner_feature_binding["binding_sha256"] = hashlib.sha256(
            _canonical_json_bytes(inner_feature_binding)
        ).hexdigest()
        transform_payload = {
            "atom_parameter_sha256": inner_atom_receipt["parameter_sha256"],
            "imputation_values": [float(value) for value in imputation],
            "scales": [float(value) for value in scales],
            "feature_names": list(selected_features),
            "feature_value_digest": inner_feature_binding["feature_value_digest"],
            "feature_ledger_binding_sha256": inner_feature_binding["binding_sha256"],
        }
        transform_sha256 = hashlib.sha256(
            _canonical_json_bytes(transform_payload)
        ).hexdigest()
        for regularization_c in REGULARIZATION_GRID:
            classifier, optimizer = _fit_zero_intercept_logistic(
                matrix / scales,
                train_target.to_numpy(dtype=int),
                regularization_c=float(regularization_c),
            )
            if (
                optimizer.get("success") is not True
                or optimizer.get("finite_coefficients") is not True
            ):
                raise FutureValueSourceError(
                    "nested regularization candidate optimizer evidence is not converged"
                )
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                probability = classifier.predict_proba(validation_matrix / scales)[:, 1]
            if not np.isfinite(probability).all():
                raise FutureValueSourceError("nested regularization prediction is non-finite")
            prediction_rows = [
                {
                    "game_id": str(game_id),
                    "target": int(target_value),
                    "probability": float(probability_value),
                }
                for game_id, target_value, probability_value in zip(
                    inner_validation["game_id"], validation_target, probability
                )
            ]
            candidate_scores.append(
                {
                    "c": float(regularization_c),
                    "log_loss": float(
                        log_loss(validation_target.to_numpy(dtype=int), probability)
                    ),
                    "optimizer": optimizer,
                    "prediction_sha256": hashlib.sha256(
                        _canonical_json_bytes(prediction_rows)
                    ).hexdigest(),
                }
            )
        selected = min(candidate_scores, key=lambda row: (row["log_loss"], row["c"]))
        return {
            "method": "nested_chronological_whole_series_log_loss",
            "candidate_grid": [float(value) for value in REGULARIZATION_GRID],
            "candidate_scores": candidate_scores,
            "selected_c": float(selected["c"]),
            "inner_ledger_status": "verified",
            "blockers": [],
            "outer_ledger_reuse": False,
            "selection_scope": "inner_train_and_validation_only",
            "variant": config.variant.value,
            "source_receipt_sha256": str(source_receipt["receipt_sha256"]),
            "source_identity_sha256": str(source_receipt["source_identity_sha256"]),
            "outer_train_game_count": len(outer_train_ids),
            "outer_train_identity_sha256": identity_sha256(outer_train_ids),
            "outer_validation_game_count": len(
                set(map_frame["game_id"].astype(str)) - set(outer_train_ids)
            ),
            "outer_validation_identity_sha256": identity_sha256(
                tuple(sorted(set(map_frame["game_id"].astype(str)) - set(outer_train_ids)))
            ),
            "inner_feature_ledger_binding": inner_feature_binding,
            "inner_feature_value_digest": inner_feature_binding["feature_value_digest"],
            "inner_producer_receipt_sha256": inner_feature_binding[
                "producer_receipt_sha256"
            ],
            "inner_transform_sha256": transform_sha256,
            "inner_atom_parameter_sha256": inner_atom_receipt["parameter_sha256"],
            "inner_train_game_count": len(inner_train_ids),
            "inner_train_identity_sha256": identity_sha256(inner_train_ids),
            "inner_validation_game_count": len(inner_validation_ids),
            "inner_validation_identity_sha256": identity_sha256(inner_validation_ids),
            "inner_validation_start": str(inner_fold["validation_start"]),
            "inner_validation_end": str(inner_fold["validation_end"]),
            "inner_overlap_audit": dict(inner_fold["overlap_audit"]),
            "optimizer_evidence": {
                "all_candidates_converged": all(
                    bool(row["optimizer"].get("success"))
                    and bool(row["optimizer"].get("finite_coefficients"))
                    for row in candidate_scores
                ),
                "selected_candidate": dict(selected["optimizer"]),
            },
        }

    outer_train_maps = map_frame[map_frame["game_id"].isin(train_game_ids)].copy()
    inner_fold = chronological_whole_series_folds(
        outer_train_maps,
        n_folds=1,
        verified_model_frame=outer_train_maps,
    )[0]
    inner_train_ids = tuple(str(value) for value in inner_fold["train_game_ids"])
    inner_validation_ids = tuple(
        str(value) for value in inner_fold["validation_game_ids"]
    )
    atom_model = fit_rank3_player_champion_role_atoms(
        form,
        train_game_ids=inner_train_ids,
        rank=rank,
        min_cell_support=min_cell_support,
        fit_window_end=inner_fold["validation_start"],
    )
    design = build_future_value_design(
        map_frame,
        form,
        atom_model,
        verified_model_frame=map_frame,
        variant=variant,
        feature_ledger=feature_ledger,
        source_receipt=source_receipt,
        train_game_ids=train_game_ids if variant is not None else None,
        fit_window_end=fit_window_end if variant is not None else None,
    )
    inner_train = design[design["game_id"].isin(inner_train_ids)].copy()
    inner_validation = design[
        design["game_id"].isin(inner_validation_ids)
    ].copy()
    train_target = pd.to_numeric(inner_train["target"], errors="coerce")
    validation_target = pd.to_numeric(inner_validation["target"], errors="coerce")
    if (
        len(inner_train) < 10
        or len(inner_validation) < 10
        or train_target.nunique() != 2
        or validation_target.nunique() != 2
    ):
        raise FutureValueSourceError(
            "nested regularization fold has insufficient two-class rows"
        )
    selected_features = tuple(
        MODEL_FEATURES if variant is None else rating_variant_config(variant).feature_names
    )
    if variant is None:
        imputation = _fold_level_imputation_values(inner_train)
    else:
        imputation = _variant_imputation_values(
            inner_train,
            rating_variant_config(variant),
        )
    if variant is None:
        matrix = _antisymmetric_design_matrix(inner_train, imputation)
        validation_matrix = _antisymmetric_design_matrix(inner_validation, imputation)
    else:
        matrix = _antisymmetric_design_matrix(
            inner_train,
            imputation,
            feature_names=selected_features,
        )
        validation_matrix = _antisymmetric_design_matrix(
            inner_validation,
            imputation,
            feature_names=selected_features,
        )
    scales = matrix.std(axis=0, ddof=0)
    scales = np.where(np.isfinite(scales) & (scales > 1e-12), scales, 1.0)
    candidate_scores: list[dict[str, Any]] = []
    inner_atom_receipt = atom_model.parameter_receipt()
    transform_payload = {
        "atom_parameter_sha256": inner_atom_receipt["parameter_sha256"],
        "imputation_values": [float(value) for value in imputation],
        "scales": [float(value) for value in scales],
        "feature_names": list(selected_features),
    }
    transform_sha256 = hashlib.sha256(
        _canonical_json_bytes(transform_payload)
    ).hexdigest()
    for regularization_c in REGULARIZATION_GRID:
        classifier, optimizer = _fit_zero_intercept_logistic(
            matrix / scales,
            train_target.to_numpy(dtype=int),
            regularization_c=float(regularization_c),
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            probability = classifier.predict_proba(validation_matrix / scales)[:, 1]
        if not np.isfinite(probability).all():
            raise FutureValueSourceError(
                "nested regularization prediction is non-finite"
            )
        prediction_rows = [
            {
                "game_id": str(game_id),
                "target": int(target_value),
                "probability": float(probability_value),
            }
            for game_id, target_value, probability_value in zip(
                inner_validation["game_id"], validation_target, probability
            )
        ]
        candidate_scores.append(
            {
                "c": float(regularization_c),
                "log_loss": float(
                    log_loss(validation_target.to_numpy(dtype=int), probability)
                ),
                "optimizer": optimizer,
                "prediction_sha256": hashlib.sha256(
                    _canonical_json_bytes(prediction_rows)
                ).hexdigest(),
            }
        )
    selected = min(candidate_scores, key=lambda row: (row["log_loss"], row["c"]))
    return {
        "method": "nested_chronological_whole_series_log_loss",
        "candidate_grid": [float(value) for value in REGULARIZATION_GRID],
        "candidate_scores": candidate_scores,
        "selected_c": float(selected["c"]),
        "inner_atom_parameter_sha256": inner_atom_receipt["parameter_sha256"],
        "inner_transform_sha256": transform_sha256,
        "inner_train_game_count": len(inner_train_ids),
        "inner_train_identity_sha256": identity_sha256(inner_train_ids),
        "inner_validation_game_count": len(inner_validation_ids),
        "inner_validation_identity_sha256": identity_sha256(inner_validation_ids),
        "inner_validation_start": str(inner_fold["validation_start"]),
        "inner_validation_end": str(inner_fold["validation_end"]),
        "inner_overlap_audit": dict(inner_fold["overlap_audit"]),
    }


@dataclass(frozen=True)
class FutureValueFoldModel:
    """A fitted development model for one chronological fold."""

    feature_names: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    imputation_values: np.ndarray
    coefficients: np.ndarray
    intercept: float
    regularization_selection: Mapping[str, Any]
    optimizer_evidence: Mapping[str, Any]
    atom_model: Rank3AtomModel
    fit_game_ids: tuple[str, ...]
    fit_window_end: str
    train_rows: int
    withheld_rows: int
    source_receipt: Mapping[str, Any]
    variant: RatingVariant | None = None
    feature_ledger_binding: Mapping[str, Any] | None = None

    @property
    def metric_weights(self) -> dict[str, float]:
        metric_features = {f"player_form_{metric}" for metric in FORM_METRICS}
        return {
            feature: float(value)
            for feature, value in zip(self.feature_names, self.coefficients)
            if feature in metric_features
        }

    @property
    def coefficient_map(self) -> dict[str, float]:
        return {
            feature: float(value)
            for feature, value in zip(self.feature_names, self.coefficients)
        }

    def parameter_receipt(self) -> dict[str, Any]:
        """Return the fitted fold parameters needed to reproduce predictions."""

        parameters: dict[str, Any] = {
            "feature_names": list(self.feature_names),
            "variant": self.variant.value if self.variant is not None else None,
            "variant_receipt": (
                rating_variant_config_receipt(self.variant)
                if self.variant is not None
                else None
            ),
            "feature_means": {
                feature: float(value)
                for feature, value in zip(self.feature_names, self.means)
            },
            "feature_scales": {
                feature: float(value)
                for feature, value in zip(self.feature_names, self.scales)
            },
            "fold_local_side_imputation": {
                feature: float(value)
                for feature, value in zip(
                    SIDE_LEVEL_TO_MODEL_FEATURE if self.variant is None else self.feature_names,
                    self.imputation_values,
                )
            },
            "imputation_policy": {
                "finite_features": "fold_local_pooled_side_median",
                "all_missing_centered_atom_coordinates": "neutral_zero",
                "all_missing_non_centered_features": "fail_closed",
                "centered_atom_features": sorted(CENTERED_ATOM_LEVEL_FEATURES),
            },
            "current_rating_feature_semantics": dict(CURRENT_RATING_FEATURE_SEMANTICS),
            "coefficients": self.coefficient_map,
            "intercept": float(self.intercept),
            "antisymmetric_fit": {
                "side_operation": "blue_minus_red_after_equal_side_imputation",
                "centering": "zero",
                "fit_intercept": False,
            },
            "regularization_selection": dict(self.regularization_selection),
            "optimizer_evidence": dict(self.optimizer_evidence),
            "rank_3": self.atom_model.parameter_receipt(),
            "feature_ledger_binding": dict(self.feature_ledger_binding or {}),
        }
        parameters["parameter_sha256"] = hashlib.sha256(
            _canonical_json_bytes(parameters)
        ).hexdigest()
        return parameters

    def predict_logit(self, design: pd.DataFrame) -> pd.Series:
        values = _antisymmetric_design_matrix(
            design,
            self.imputation_values,
            feature_names=self.feature_names,
        )
        scaled = values / self.scales
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            output = scaled @ self.coefficients
        output[~np.isfinite(output)] = np.nan
        return pd.Series(output, index=design.index, name="future_value_logit")

    def predict_probability(self, design: pd.DataFrame) -> pd.Series:
        logits = self.predict_logit(design)
        values = logits.to_numpy(dtype=float)
        finite = np.isfinite(values)
        values[finite] = 1.0 / (1.0 + np.exp(-np.clip(values[finite], -40.0, 40.0)))
        return pd.Series(values, index=design.index, name="future_value_probability")

    def player_value_logit(
        self,
        form: pd.DataFrame,
        design: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return exact map-logit components and explicit player support records."""

        required = {"game_id", "player_id", "side", "role", "champion"}
        missing = sorted(required - set(form.columns))
        if missing:
            raise FutureValueSourceError(
                "player value form is missing: " + ", ".join(missing)
            )
        atoms = self.atom_model.transform(form)
        combined = pd.concat(
            [form.reset_index(drop=True), atoms.reset_index(drop=True)], axis=1
        )
        design_ids = set(design["game_id"].astype(str))
        selected = combined[combined["game_id"].astype(str).isin(design_ids)].copy()
        counts = selected.groupby("game_id", sort=False, observed=True).size()
        if set(counts.index.astype(str)) != design_ids or not counts.eq(10).all():
            raise FutureValueSourceError(
                "player value components require ten player rows per design game"
            )

        matrix = _antisymmetric_design_matrix(
            design,
            self.imputation_values,
            feature_names=self.feature_names,
        )
        contributions = (matrix / self.scales) * self.coefficients
        player_sources = {
            *[f"player_form_{metric}" for metric in FORM_METRICS],
            *[
                f"rank_3_player_atom_{index}"
                for index in range(1, RANK_3 + 1)
            ],
            *[
                f"rank_3_champion_role_atom_{index}"
                for index in range(1, RANK_3 + 1)
            ],
        }
        team_sources = {"team_prior_win", "roster_continuity"}
        rating_sources = set(CURRENT_RATING_SIGNED_MAP_FEATURES)
        scaling_sources = set(SCALING_CURVE_SIGNED_MAP_FEATURES)
        player_indexes = [
            index
            for index, source_name in enumerate(self.feature_names)
            if source_name in player_sources
        ]
        team_indexes = [
            index
            for index, source_name in enumerate(self.feature_names)
            if source_name in team_sources
        ]
        rating_indexes = [
            index
            for index, source_name in enumerate(self.feature_names)
            if source_name in rating_sources
        ]
        scaling_indexes = [
            index
            for index, source_name in enumerate(self.feature_names)
            if source_name in scaling_sources
        ]
        quality_indexes = [
            index
            for index in range(len(self.feature_names))
            if index not in {*player_indexes, *team_indexes, *rating_indexes, *scaling_indexes}
        ]
        output = design[["game_id"]].copy().reset_index(drop=True)
        output["player_value_logit"] = contributions[:, player_indexes].sum(axis=1)
        output["team_context_logit"] = contributions[:, team_indexes].sum(axis=1)
        output["current_rating_logit"] = contributions[:, rating_indexes].sum(axis=1)
        output["scaling_curve_logit"] = contributions[:, scaling_indexes].sum(axis=1)
        output["data_quality_logit"] = contributions[:, quality_indexes].sum(axis=1)
        output["full_model_logit"] = self.predict_logit(design).to_numpy(dtype=float)
        output["component_reconstruction_error"] = output["full_model_logit"] - (
            output["player_value_logit"]
            + output["team_context_logit"]
            + output["current_rating_logit"]
            + output["scaling_curve_logit"]
            + output["data_quality_logit"]
        )
        if output["component_reconstruction_error"].abs().max() > 1e-12:
            raise FutureValueSourceError("player value components do not reconstruct logit")

        metric_support_columns = [
            f"prior_form_{metric}_support" for metric in FORM_METRICS
        ]
        effective_support_columns = [
            f"prior_form_{metric}_effective_support" for metric in FORM_METRICS
        ]
        explicit_value_columns = [
            *[f"prior_form_{metric}" for metric in FORM_METRICS],
            *[
                f"rank_3_player_atom_{index}"
                for index in range(1, RANK_3 + 1)
            ],
            *[
                f"rank_3_champion_role_atom_{index}"
                for index in range(1, RANK_3 + 1)
            ],
        ]
        records_by_game: dict[str, list[dict[str, Any]]] = {}
        for game_id, group in selected.groupby("game_id", sort=False, observed=True):
            records: list[dict[str, Any]] = []
            for row in group.to_dict(orient="records"):
                support = {
                    metric: _ledger_value(row.get(f"prior_form_{metric}_support"))
                    for metric in FORM_METRICS
                }
                effective_support = {
                    metric: _ledger_value(
                        row.get(f"prior_form_{metric}_effective_support")
                    )
                    for metric in FORM_METRICS
                }
                finite_support = [
                    float(value) for value in support.values() if value is not None
                ]
                finite_effective = [
                    float(value)
                    for value in effective_support.values()
                    if value is not None
                ]
                missing_names = sorted(
                    name
                    for name in explicit_value_columns
                    if _ledger_value(row.get(name)) is None
                )
                atom_support = int(row.get("rank_3_champion_role_support") or 0)
                minimum_support = min(finite_support) if finite_support else 0.0
                minimum_effective = min(finite_effective) if finite_effective else 0.0
                status = (
                    "missing"
                    if missing_names
                    else "sparse"
                    if minimum_effective < 5.0 or atom_support < 1
                    else "adequate"
                )
                records.append(
                    {
                        "player_id": str(row["player_id"]),
                        "side": str(row["side"]),
                        "role": str(row["role"]),
                        "metric_support": support,
                        "metric_effective_support": effective_support,
                        "minimum_metric_support": minimum_support,
                        "minimum_effective_support": minimum_effective,
                        "champion_role_atom_support": atom_support,
                        "missing_feature_names": missing_names,
                        "support_status": status,
                    }
                )
            records_by_game[str(game_id)] = records
        output["player_support_records"] = output["game_id"].astype(str).map(
            records_by_game
        )
        output["support_status"] = output["player_support_records"].map(
            lambda records: (
                "missing"
                if any(row["support_status"] == "missing" for row in records)
                else "sparse"
                if any(row["support_status"] == "sparse" for row in records)
                else "adequate"
            )
        )
        return output

    def receipt(self) -> dict[str, Any]:
        parameters = self.parameter_receipt()
        return {
            "schema_version": MODEL_FIT_SCHEMA_VERSION,
            "fit_game_count": len(self.fit_game_ids),
            "fit_game_ids": list(self.fit_game_ids),
            "fit_window_end": self.fit_window_end,
            "feature_names": list(self.feature_names),
            "variant": self.variant.value if self.variant is not None else None,
            "variant_receipt": (
                rating_variant_config_receipt(self.variant)
                if self.variant is not None
                else None
            ),
            "metric_weights": self.metric_weights,
            "coefficients": self.coefficient_map,
            "intercept": float(self.intercept),
            "feature_means": parameters["feature_means"],
            "feature_scales": parameters["feature_scales"],
            "fold_local_side_imputation": parameters[
                "fold_local_side_imputation"
            ],
            "imputation_policy": parameters["imputation_policy"],
            "antisymmetric_fit": parameters["antisymmetric_fit"],
            "regularization_selection": parameters["regularization_selection"],
            "optimizer_evidence": parameters["optimizer_evidence"],
            "parameter_sha256": parameters["parameter_sha256"],
            "rank_3": parameters["rank_3"],
            "feature_ledger_binding": dict(self.feature_ledger_binding or {}),
            "train_rows": int(self.train_rows),
            "withheld_rows": int(self.withheld_rows),
            "source_binding": {
                "source_as_of": self.source_receipt["source_as_of"],
                "source_game_count": int(self.source_receipt["source_game_count"]),
                "source_identity_sha256": self.source_receipt[
                    "source_identity_sha256"
                ],
                "accepted_game_ids": list(self.source_receipt["accepted_game_ids"]),
                "model_eligible_game_count": int(
                    self.source_receipt["model_eligible_game_count"]
                ),
                "model_eligible_identity_sha256": self.source_receipt[
                    "model_eligible_identity_sha256"
                ],
                "model_eligible_game_ids": list(
                    self.source_receipt["model_eligible_game_ids"]
                ),
                "source_files": self.source_receipt["source_files"],
                "source_receipt_sha256": self.source_receipt["receipt_sha256"],
            },
            "authority": {
                "research_only": True,
                "public_player_rating": False,
                "public_team_rating": False,
                "public_probability": False,
                "deployment": False,
                "promotion": False,
            },
        }


def fit_future_value_model(
    maps: pd.DataFrame,
    form: pd.DataFrame,
    *,
    train_game_ids: Iterable[str],
    fit_window_end: Any | None = None,
    rank: int = RANK_3,
    min_cell_support: int = 1,
    source_receipt: Mapping[str, Any] | None = None,
    crosswalk_receipt_file_sha256: str | None = None,
    verified_model_frame: pd.DataFrame | None = None,
    variant: RatingVariant | str | None = None,
    feature_ledger: pd.DataFrame | None = None,
    inner_feature_ledger: pd.DataFrame | None = None,
) -> tuple[FutureValueFoldModel, pd.DataFrame]:
    """Fit one fold with all representation work bound to its train games."""

    map_frame = (
        verified_model_frame.copy()
        if verified_model_frame is not None
        else _map_model_frame(
            maps,
            verified_source_receipt_sha256=(
                str(source_receipt["receipt_sha256"])
                if isinstance(source_receipt, Mapping)
                and "receipt_sha256" in source_receipt
                else None
            ),
            verified_source_receipt=source_receipt,
            verified_crosswalk_receipt_file_sha256=crosswalk_receipt_file_sha256,
        )
    )
    train_ids = tuple(sorted({str(value) for value in train_game_ids}))
    if not train_ids:
        raise FutureValueSourceError("future-value fit has no training games")
    _validate_verified_source_receipt(
        source_receipt,
        map_frame,
        require_full_eligible_set=False,
        train_game_ids=train_ids,
    )
    train_dates = map_frame.loc[map_frame["game_id"].isin(train_ids), "date"]
    if train_dates.empty:
        raise FutureValueSourceError("future-value training games are absent")
    boundary = fit_window_end if fit_window_end is not None else train_dates.max()
    variant_config = None if variant is None else rating_variant_config(variant)
    if variant_config is not None and fit_window_end is None:
        raise FutureValueSourceError("explicit rating variants require a strict fold cutoff")
    regularization_selection = _select_fold_regularization(
        map_frame,
        form,
        train_game_ids=train_ids,
        rank=rank,
        min_cell_support=min_cell_support,
        variant=None if variant_config is None else variant_config.variant,
        feature_ledger=feature_ledger,
        inner_feature_ledger=inner_feature_ledger,
        source_receipt=source_receipt,
        fit_window_end=boundary if variant_config is not None else None,
    )
    atom_model = fit_rank3_player_champion_role_atoms(
        form,
        train_game_ids=train_ids,
        rank=rank,
        min_cell_support=min_cell_support,
        fit_window_end=None if fit_window_end is None else boundary,
    )
    design = build_future_value_design(
        map_frame,
        form,
        atom_model,
        verified_model_frame=map_frame,
        variant=None if variant_config is None else variant_config.variant,
        feature_ledger=feature_ledger,
        source_receipt=source_receipt,
        train_game_ids=train_ids if variant_config is not None else None,
        fit_window_end=boundary if variant_config is not None else None,
    )
    train = design[design["game_id"].isin(train_ids)].copy()
    feature_names = tuple(
        MODEL_FEATURES if variant_config is None else variant_config.feature_names
    )
    target = pd.to_numeric(train["target"], errors="coerce")
    usable = target.isin({0, 1})
    usable_train = train.loc[usable]
    if len(usable_train) < 20 or usable_train["target"].nunique() != 2:
        raise FutureValueSourceError("future-value fold has insufficient two-class training rows")
    if variant_config is None:
        imputation_values = _fold_level_imputation_values(usable_train)
    else:
        imputation_values = _variant_imputation_values(usable_train, variant_config)
    matrix = (
        _antisymmetric_design_matrix(usable_train, imputation_values)
        if variant_config is None
        else _antisymmetric_design_matrix(
            usable_train,
            imputation_values,
            feature_names=feature_names,
        )
    )
    means = np.zeros(matrix.shape[1], dtype=float)
    scales = matrix.std(axis=0, ddof=0)
    scales = np.where(np.isfinite(scales) & (scales > 1e-12), scales, 1.0)
    classifier, optimizer_evidence = _fit_zero_intercept_logistic(
        matrix / scales,
        usable_train["target"].to_numpy(dtype=int),
        regularization_c=float(regularization_selection["selected_c"]),
    )
    model = FutureValueFoldModel(
        feature_names=feature_names,
        means=means,
        scales=scales,
        imputation_values=imputation_values,
        coefficients=classifier.coef_[0].astype(float),
        intercept=0.0,
        regularization_selection=regularization_selection,
        optimizer_evidence=optimizer_evidence,
        atom_model=atom_model,
        fit_game_ids=train_ids,
        fit_window_end=_utc_text(boundary),
        train_rows=int(len(usable_train)),
        withheld_rows=int((~usable).sum()),
        source_receipt=dict(source_receipt or {}),
        variant=None if variant_config is None else variant_config.variant,
        feature_ledger_binding=(
            dict(design.attrs.get("feature_ledger") or {})
            if variant_config is not None
            else None
        ),
    )
    return model, design


def chronological_whole_series_folds(
    maps: pd.DataFrame,
    *,
    n_folds: int = 3,
    verified_model_frame: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Return expanding chronological folds with whole series clusters."""

    if int(n_folds) < 1:
        raise FutureValueSourceError("chronological fold count must be positive")
    frame = (
        verified_model_frame.copy()
        if verified_model_frame is not None
        else _map_model_frame(maps)
    )
    series_summary = (
        frame.groupby("series_id", sort=True, observed=True)
        .agg(first_date=("date", "min"), last_date=("date", "max"))
        .sort_values(["first_date", "series_id"], kind="stable")
    )
    date_blocks = [
        pd.Timestamp(value) for value in sorted(frame["date"].drop_duplicates())
    ]
    if len(date_blocks) < int(n_folds) + 1:
        raise FutureValueSourceError("chronological folds need more timestamp blocks")
    chunks = np.array_split(np.asarray(date_blocks, dtype=object), int(n_folds) + 1)
    folds: list[dict[str, Any]] = []
    for fold_index in range(1, len(chunks)):
        validation_dates = chunks[fold_index]
        if not len(validation_dates):
            continue
        validation_min = pd.Timestamp(validation_dates[0])
        validation_max = pd.Timestamp(validation_dates[-1])
        contained = series_summary["first_date"].ge(validation_min) & series_summary[
            "last_date"
        ].le(validation_max)
        intersects = series_summary["last_date"].ge(validation_min) & series_summary[
            "first_date"
        ].le(validation_max)
        validation_series = set(series_summary.index[contained].astype(str))
        excluded_boundary_series = set(
            series_summary.index[intersects & ~contained].astype(str)
        )
        train_series = {
            str(series_id)
            for series_id, row in series_summary.iterrows()
            if row["last_date"] < validation_min
        }
        train_ids = tuple(sorted(frame.loc[frame["series_id"].astype(str).isin(train_series), "game_id"]))
        validation_ids = tuple(sorted(frame.loc[frame["series_id"].astype(str).isin(validation_series), "game_id"]))
        if not train_ids or not validation_ids:
            continue
        train_max = frame.loc[frame["game_id"].isin(train_ids), "date"].max()
        valid_min = frame.loc[frame["game_id"].isin(validation_ids), "date"].min()
        valid_max = frame.loc[frame["game_id"].isin(validation_ids), "date"].max()
        if not pd.Timestamp(train_max) < pd.Timestamp(valid_min):
            raise FutureValueSourceError("chronological fold has a non-strict date boundary")
        if pd.Timestamp(valid_min) < validation_min or pd.Timestamp(valid_max) > validation_max:
            raise FutureValueSourceError("validation cluster crosses its chronological interval")
        excluded_boundary_map_count = int(
            frame["series_id"].astype(str).isin(excluded_boundary_series).sum()
        )
        folds.append(
            {
                "fold": int(fold_index),
                "train_game_ids": train_ids,
                "validation_game_ids": validation_ids,
                "train_series_ids": tuple(sorted(train_series)),
                "validation_series_ids": tuple(sorted(validation_series)),
                "train_end": _utc_text(train_max),
                "validation_start": _utc_text(valid_min),
                "validation_end": _utc_text(valid_max),
                "validation_interval_start": _utc_text(validation_min),
                "validation_interval_end": _utc_text(validation_max),
                "overlap_audit": {
                    "boundary_cluster_policy": "exclude_cluster_from_validation_interval",
                    "excluded_boundary_cluster_count": len(excluded_boundary_series),
                    "excluded_boundary_map_count": excluded_boundary_map_count,
                    "validation_game_identity_sha256": identity_sha256(validation_ids),
                },
            }
        )
    if len(folds) != int(n_folds):
        raise FutureValueSourceError(
            "chronological fold construction did not produce every requested fold"
        )
    seen_validation_ids: set[str] = set()
    previous_end: pd.Timestamp | None = None
    for fold in folds:
        current_ids = set(map(str, fold["validation_game_ids"]))
        overlap = seen_validation_ids & current_ids
        if overlap:
            raise FutureValueSourceError("chronological validation folds overlap by game ID")
        current_start = _utc_timestamp(fold["validation_start"], "validation_start")
        current_end = _utc_timestamp(fold["validation_end"], "validation_end")
        if previous_end is not None and not previous_end < current_start:
            raise FutureValueSourceError("chronological validation intervals overlap")
        fold["overlap_audit"]["prior_validation_game_overlap_count"] = 0
        fold["overlap_audit"]["prior_validation_interval_overlap"] = False
        seen_validation_ids.update(current_ids)
        previous_end = current_end
    return folds


def _classification_metrics(target: pd.Series, probability: pd.Series) -> dict[str, Any]:
    valid = target.notna() & probability.notna()
    if not valid.any():
        return {"rows": 0, "log_loss": None, "brier": None, "auc": None}
    y = target.loc[valid].astype(int).to_numpy()
    p = np.clip(probability.loc[valid].to_numpy(dtype=float), 1e-8, 1.0 - 1e-8)
    return {
        "rows": int(len(y)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])) if len(np.unique(y)) == 2 else None,
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
    }


CALIBRATION_SCHEMA_VERSION = "scryglass:future-value-strict-prior-calibration:v1"
CALIBRATION_MAX_SLOPE = 100.0
CALIBRATION_MIN_SLOPE = 1e-9


def _stable_sigmoid(logit: np.ndarray) -> np.ndarray:
    """Return finite probabilities without changing the supplied logit evidence."""

    values = np.asarray(logit, dtype=float)
    output = np.empty_like(values, dtype=float)
    positive = values >= 0.0
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        output[positive] = 1.0 / (1.0 + np.exp(-np.clip(values[positive], -40.0, 40.0)))
        exp_values = np.exp(np.clip(values[~positive], -40.0, 40.0))
        output[~positive] = exp_values / (1.0 + exp_values)
    return output


def _calibration_input_sha256(
    fold_numbers: Sequence[int],
    game_ids: Sequence[str],
    logits: Sequence[float],
    targets: Sequence[int],
) -> str:
    rows = [
        {
            "fold": int(fold),
            "game_id": str(game_id),
            "raw_logit": float(logit),
            "target": int(target),
        }
        for fold, game_id, logit, target in zip(
            fold_numbers, game_ids, logits, targets
        )
    ]
    return hashlib.sha256(_canonical_json_bytes(rows)).hexdigest()


def _fit_strict_prior_calibration(
    raw_logits: Sequence[float],
    targets: Sequence[int],
    *,
    source_receipt_sha256: str,
    current_fold: int,
    current_validation_game_ids: Sequence[str],
    current_validation_start: str,
    prior_fold_numbers: Sequence[int] = (),
    prior_game_ids: Sequence[str] = (),
    prior_validation_ends: Sequence[str] = (),
) -> dict[str, Any]:
    """Fit one positive, zero-intercept slope on prior outer-fold rows only.

    The first outer fold uses the identity transform.  It carries a named
    blocker because no earlier validation outcome exists.  Later folds fit
    the slope from the supplied rows, which must all come from earlier folds.
    """

    if not isinstance(source_receipt_sha256, str) or len(source_receipt_sha256) != 64:
        raise FutureValueSourceError("calibration source receipt hash is invalid")
    if int(current_fold) < 1:
        raise FutureValueSourceError("calibration fold number is invalid")
    logits = np.asarray(raw_logits, dtype=float)
    y = np.asarray(targets, dtype=int)
    folds = np.asarray(prior_fold_numbers, dtype=int)
    game_ids = tuple(str(value) for value in prior_game_ids)
    if len(logits) != len(y) or len(logits) != len(folds) or len(logits) != len(game_ids):
        raise FutureValueSourceError("calibration input lengths do not match")
    if any(int(value) >= int(current_fold) for value in folds):
        raise FutureValueSourceError("calibration input includes the current or a future fold")
    if len(set(game_ids)) != len(game_ids):
        raise FutureValueSourceError("calibration input game IDs are not unique")
    if len(prior_validation_ends) not in (0, len(set(folds))):
        raise FutureValueSourceError("calibration prior fold dates are invalid")
    if not isinstance(current_validation_start, str) or not current_validation_start.strip():
        raise FutureValueSourceError("calibration current validation start is missing")
    current_start = _utc_timestamp(current_validation_start, "calibration current validation start")
    if any(
        not _utc_timestamp(value, "calibration prior validation end") < current_start
        for value in prior_validation_ends
    ):
        raise FutureValueSourceError("calibration prior validation fold is not strictly earlier")
    if len(logits) and (
        not np.isfinite(logits).all()
        or not np.isin(y, (0, 1)).all()
    ):
        raise FutureValueSourceError("calibration input contains non-finite values")
    if len(logits) and len(np.unique(y)) != 2:
        raise FutureValueSourceError("calibration input needs both outcome classes")

    current_ids = tuple(str(value) for value in current_validation_game_ids)
    if len(set(current_ids)) != len(current_ids):
        raise FutureValueSourceError("calibration current game IDs are not unique")
    prior_fold_set = tuple(sorted({int(value) for value in folds}))
    input_hash = _calibration_input_sha256(folds, game_ids, logits, y)
    base: dict[str, Any] = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "method": "strict_prior_outer_validation_zero_intercept_positive_slope",
        "source_receipt_sha256": source_receipt_sha256,
        "current_fold": int(current_fold),
        "current_validation_game_count": len(current_ids),
        "current_validation_game_identity_sha256": identity_sha256(current_ids),
        "current_validation_start": str(current_validation_start),
        "prior_fold_numbers": list(prior_fold_set),
        "prior_fold_identity_sha256": hashlib.sha256(
            _canonical_json_bytes(list(prior_fold_set))
        ).hexdigest(),
        "fit_game_count": len(game_ids),
        "fit_game_ids": sorted(game_ids),
        "fit_game_identity_sha256": identity_sha256(game_ids),
        "fit_input_sha256": input_hash,
        "strict_prior": True,
        "uses_current_validation": False,
        "positive_slope": True,
        "zero_intercept": True,
        "fit_rows": int(len(logits)),
        "blockers": [],
    }
    if not len(logits):
        base.update(
            {
                "status": "available",
                "mode": "identity",
                "slope": 1.0,
                "fit_target_identity_sha256": identity_sha256(()),
                "optimizer_evidence": {
                    "method": "identity_no_prior_outer_validation_fold",
                    "success": True,
                    "status": "identity",
                    "slope": 1.0,
                    "finite_slope": True,
                    "positive_slope": True,
                    "zero_intercept": True,
                    "iterations": 0,
                    "objective_value": None,
                    "gradient_inf_norm": None,
                },
                "blockers": ["calibration_prior_validation_folds_missing"],
            }
        )
        return base

    def objective(parameter: np.ndarray) -> tuple[float, np.ndarray]:
        slope = float(parameter[0])
        scaled = slope * logits
        # logaddexp is stable for the objective.  The sigmoid is bounded only
        # for the gradient, which keeps the raw logits in the receipt.
        value = float(np.mean(np.logaddexp(0.0, scaled) - y * scaled))
        gradient = float(np.mean((_stable_sigmoid(scaled) - y) * logits))
        return value, np.asarray([gradient], dtype=float)

    result = minimize(
        lambda parameter: objective(parameter)[0],
        np.asarray([1.0], dtype=float),
        jac=lambda parameter: objective(parameter)[1],
        method="L-BFGS-B",
        bounds=((CALIBRATION_MIN_SLOPE, CALIBRATION_MAX_SLOPE),),
        options={"maxiter": 1000, "ftol": 1e-15, "gtol": 1e-10, "maxls": 50},
    )
    slope = float(np.asarray(result.x, dtype=float).ravel()[0])
    objective_value, gradient = objective(np.asarray([slope], dtype=float))
    finite = bool(
        np.isfinite(slope)
        and np.isfinite(objective_value)
        and np.isfinite(gradient).all()
    )
    success = bool(result.success and finite and slope > 0.0)
    optimizer_evidence = {
        "method": "scipy.optimize.minimize:L-BFGS-B",
        "success": success,
        "status": int(result.status),
        "message": str(result.message),
        "slope": slope,
        "finite_slope": finite,
        "positive_slope": bool(slope > 0.0),
        "zero_intercept": True,
        "iterations": int(getattr(result, "nit", 0)),
        "function_evaluations": int(getattr(result, "nfev", 0)),
        "max_iterations": 1000,
        "objective_value": objective_value,
        "gradient_inf_norm": float(np.max(np.abs(gradient))),
        "bounds": [CALIBRATION_MIN_SLOPE, CALIBRATION_MAX_SLOPE],
    }
    if not success:
        raise FutureValueSourceError("strict-prior calibration optimizer did not converge")
    base.update(
        {
            "status": "available",
            "mode": "fitted",
            "slope": slope,
            "fit_target_identity_sha256": identity_sha256(
                [str(value) for value in y.tolist()]
            ),
            "prior_validation_end": sorted(str(value) for value in prior_validation_ends),
            "optimizer_evidence": optimizer_evidence,
        }
    )
    return base


def _apply_strict_prior_calibration(
    raw_logit: Sequence[float],
    raw_probability: Sequence[float],
    calibration: Mapping[str, Any],
) -> tuple[pd.Series, pd.Series]:
    """Apply a verified calibration slope and retain raw probability/logit."""

    logits = np.asarray(raw_logit, dtype=float)
    raw = np.asarray(raw_probability, dtype=float)
    if len(logits) != len(raw):
        raise FutureValueSourceError("calibration prediction lengths do not match")
    slope = float(calibration.get("slope", float("nan")))
    if not np.isfinite(slope) or slope <= 0.0:
        raise FutureValueSourceError("calibration slope is invalid")
    if np.isinf(logits).any() or np.isinf(raw).any():
        raise FutureValueSourceError("calibration predictions are non-finite")
    finite = np.isfinite(logits) & np.isfinite(raw)
    calibrated_logit = np.full(len(logits), np.nan, dtype=float)
    calibrated_probability = np.full(len(logits), np.nan, dtype=float)
    calibrated_logit[finite] = slope * logits[finite]
    calibrated_probability[finite] = _stable_sigmoid(calibrated_logit[finite])
    if str(calibration.get("mode")) == "identity":
        # Preserve the classifier's exact raw probability for the first fold.
        calibrated_probability = raw.copy()
        calibrated_logit = logits.copy()
    return (
        pd.Series(calibrated_logit, dtype=float),
        pd.Series(calibrated_probability, dtype=float),
    )


def _calibration_metrics(
    target: pd.Series,
    probability: pd.Series,
    *,
    bins: int = 10,
) -> dict[str, Any]:
    """Report validation-only reliability bins and expected calibration error."""

    valid = target.notna() & probability.notna()
    if not valid.any():
        return {"status": "unavailable", "rows": 0, "blockers": ["calibration_rows_missing"]}
    y = target.loc[valid].astype(int).to_numpy()
    p = np.clip(probability.loc[valid].to_numpy(dtype=float), 1e-8, 1.0 - 1e-8)
    bin_index = np.minimum(np.floor(p * bins).astype(int), bins - 1)
    rows: list[dict[str, Any]] = []
    weighted_error = 0.0
    max_error = 0.0
    for bin_number in range(bins):
        selected = bin_index == bin_number
        count = int(selected.sum())
        if count == 0:
            continue
        mean_probability = float(p[selected].mean())
        observed_rate = float(y[selected].mean())
        absolute_error = abs(mean_probability - observed_rate)
        weighted_error += count / len(y) * absolute_error
        max_error = max(max_error, absolute_error)
        rows.append(
            {
                "bin": bin_number,
                "rows": count,
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
                "absolute_error": absolute_error,
            }
        )
    return {
        "status": "available",
        "rows": int(len(y)),
        "bins": rows,
        "expected_calibration_error": float(weighted_error),
        "max_absolute_error": float(max_error),
        "blockers": [],
    }


def _slice_labels(frame: pd.DataFrame, field: str) -> pd.Series | None:
    if field not in frame.columns:
        return None
    values = frame[field].astype("string").str.strip()
    return values.where(values.notna() & values.ne(""), "<missing>")


def _group_slice_metrics(
    target: pd.Series,
    probability: pd.Series,
    validation_labels: pd.Series | None,
    training_labels: pd.Series | None,
    *,
    slice_name: str,
    minimum_rows: int = 20,
) -> dict[str, Any]:
    """Score validation groups and bind each group to its training support."""

    if validation_labels is None:
        return {
            "status": "unavailable",
            "groups": {},
            "blockers": [f"{slice_name}_field_missing"],
        }
    labels = validation_labels.reindex(target.index)
    if labels.isna().all():
        return {
            "status": "unavailable",
            "groups": {},
            "blockers": [f"{slice_name}_labels_missing"],
        }
    train = training_labels if training_labels is not None else pd.Series(dtype="string")
    train = train.astype("string")
    blockers: list[str] = []
    groups: dict[str, Any] = {}
    for raw_group in sorted(str(value) for value in labels.dropna().unique()):
        selected = labels.astype(str).eq(raw_group)
        group_target = target.loc[selected]
        group_probability = probability.loc[selected]
        metrics = _classification_metrics(group_target, group_probability)
        train_rows = int(train.eq(raw_group).sum())
        if metrics["rows"] < minimum_rows:
            blockers.append(f"{slice_name}_sparse_validation_support")
        if train_rows == 0:
            blockers.append(f"{slice_name}_unseen_training_group")
        groups[raw_group] = {
            "training_rows": train_rows,
            "validation_rows": int(len(group_target)),
            "metrics": metrics,
        }
    if "<missing>" in groups:
        blockers.append(f"{slice_name}_labels_missing")
    return {
        "status": "available",
        "groups": groups,
        "blockers": sorted(set(blockers)),
    }


def _missingness_metrics(
    validation: pd.DataFrame,
    target: pd.Series,
    probability: pd.Series,
    raw_probability: pd.Series | None = None,
) -> dict[str, Any]:
    """Report fold-local imputation coverage for complete and incomplete rows."""

    if "model_features_complete" not in validation.columns:
        return {"status": "unavailable", "blockers": ["missingness_indicator_missing"]}
    complete = validation["model_features_complete"].astype(bool)
    paired = target.notna() & probability.notna()
    incomplete_target = target.notna() & ~complete
    incomplete_predicted = paired & ~complete
    blockers = []
    if int(incomplete_predicted.sum()) != int(incomplete_target.sum()):
        blockers.append("incomplete_feature_prediction_missing")
    if not bool((paired & complete).any()):
        blockers.append("complete_case_validation_rows_missing")
    status = "blocked" if "incomplete_feature_prediction_missing" in blockers else (
        "imputed_only" if not bool((paired & complete).any()) else "available"
    )
    missing_feature_counts: dict[str, int] = {}
    if "model_missing_feature_names" in validation.columns:
        for names in validation["model_missing_feature_names"]:
            for name in names:
                missing_feature_counts[str(name)] = (
                    missing_feature_counts.get(str(name), 0) + 1
                )
    report = {
        "status": status,
        "total_rows": int(len(validation)),
        "complete_case_rows": int(complete.sum()),
        "incomplete_case_rows": int((~complete).sum()),
        "predicted_rows": int(paired.sum()),
        "withheld_rows": int((~paired).sum()),
        "complete_case_metrics": _classification_metrics(
            target.loc[paired & complete], probability.loc[paired & complete]
        ),
        "incomplete_case_metrics": _classification_metrics(
            target.loc[incomplete_predicted], probability.loc[incomplete_predicted]
        ),
        "imputed_prediction_rows": int(incomplete_predicted.sum()),
        "imputed_prediction_coverage": (
            float(incomplete_predicted.sum() / incomplete_target.sum())
            if incomplete_target.any()
            else 1.0
        ),
        "imputation_contract": "fold_local_equal_side_median",
        "missing_feature_counts": dict(sorted(missing_feature_counts.items())),
        "blockers": blockers,
    }
    if raw_probability is not None:
        raw_paired = target.notna() & raw_probability.notna()
        report["raw_complete_case_metrics"] = _classification_metrics(
            target.loc[raw_paired & complete], raw_probability.loc[raw_paired & complete]
        )
        report["raw_incomplete_case_metrics"] = _classification_metrics(
            target.loc[raw_paired & ~complete], raw_probability.loc[raw_paired & ~complete]
        )
        report["raw_predicted_rows"] = int(raw_paired.sum())
    return report


def _side_swap_metrics(
    model: FutureValueFoldModel,
    validation: pd.DataFrame,
    target: pd.Series,
    probability: pd.Series,
    *,
    raw_probability: pd.Series | None = None,
    calibration_slope: float = 1.0,
) -> dict[str, Any]:
    """Check the antisymmetry of the same validation rows after a side swap."""

    paired = target.notna() & probability.notna()
    if not paired.any():
        return {"status": "unavailable", "rows": 0, "blockers": ["side_swap_rows_missing"]}
    swapped = validation.copy()
    for source_name in SIDE_LEVEL_TO_MODEL_FEATURE:
        blue_column = _side_level_column("blue", source_name)
        red_column = _side_level_column("red", source_name)
        blue = swapped[blue_column].copy()
        swapped[blue_column] = swapped[red_column].to_numpy()
        swapped[red_column] = blue.to_numpy()
    for feature_name in model.feature_names:
        if is_signed_map_feature(feature_name):
            if feature_name not in swapped.columns:
                return {
                    "status": "blocked",
                    "rows": 0,
                    "blockers": ["side_swap_signed_feature_missing"],
                }
            swapped[feature_name] = -pd.to_numeric(
                swapped[feature_name], errors="coerce"
            )
    swapped_logit = model.predict_logit(swapped).loc[paired]
    swapped_probability = pd.Series(
        _stable_sigmoid(
            float(calibration_slope) * swapped_logit.to_numpy(dtype=float)
        ),
        index=swapped_logit.index,
    )
    swapped_target = 1.0 - target.loc[paired]
    original_probability = probability.loc[paired]
    symmetry_error = np.abs(
        swapped_probability.to_numpy(dtype=float)
        - (1.0 - original_probability.to_numpy(dtype=float))
    )
    mean_error = float(np.nanmean(symmetry_error))
    max_error = float(np.nanmax(symmetry_error))
    within_tolerance = bool(
        np.isfinite(mean_error)
        and np.isfinite(max_error)
        and mean_error <= SIDE_SWAP_MEAN_TOLERANCE
        and max_error <= SIDE_SWAP_MAX_TOLERANCE
    )
    report = {
        "status": "available" if within_tolerance else "blocked",
        "rows": int(len(swapped_probability)),
        "metrics": _classification_metrics(swapped_target, swapped_probability),
        "mean_probability_complement_error": mean_error,
        "max_probability_complement_error": max_error,
        "mean_probability_complement_tolerance": SIDE_SWAP_MEAN_TOLERANCE,
        "max_probability_complement_tolerance": SIDE_SWAP_MAX_TOLERANCE,
        "within_tolerance": within_tolerance,
        "blockers": (
            []
            if within_tolerance
            else ["side_swap_probability_complement_tolerance_exceeded"]
        ),
    }
    if raw_probability is not None:
        raw_swapped_probability = model.predict_probability(swapped).loc[paired]
        raw_error = np.abs(
            raw_swapped_probability.to_numpy(dtype=float)
            - (1.0 - raw_probability.loc[paired].to_numpy(dtype=float))
        )
        report["raw_metrics"] = _classification_metrics(
            swapped_target,
            raw_swapped_probability,
        )
        report["raw_max_probability_complement_error"] = float(np.nanmax(raw_error))
        report["calibration_slope"] = float(calibration_slope)
    return report


def _roster_change_labels(frame: pd.DataFrame) -> pd.Series | None:
    required = {"blue_roster_continuity", "red_roster_continuity"}
    if not required.issubset(frame.columns):
        return None
    blue = pd.to_numeric(frame["blue_roster_continuity"], errors="coerce")
    red = pd.to_numeric(frame["red_roster_continuity"], errors="coerce")
    if not bool((blue.notna() | red.notna()).any()):
        return None
    labels = pd.Series("prior_roster_unavailable", index=frame.index, dtype="string")
    both_available = blue.notna() & red.notna()
    stable = both_available & blue.ge(1.0) & red.ge(1.0)
    changed = both_available & (blue.lt(1.0) | red.lt(1.0))
    labels.loc[stable] = "stable_roster"
    labels.loc[changed] = "roster_change"
    return labels


def _support_labels(frame: pd.DataFrame, threshold: float = 5.0) -> pd.Series | None:
    field = "player_form_minimum_effective_support"
    if field not in frame.columns:
        return None
    support = pd.to_numeric(frame[field], errors="coerce")
    labels = pd.Series("<missing>", index=frame.index, dtype="string")
    labels.loc[support.ge(float(threshold))] = "adequate_support"
    labels.loc[support.lt(float(threshold)) & support.notna()] = "sparse_support"
    return labels


def _tournament_boundary_labels(frame: pd.DataFrame) -> pd.Series | None:
    if "tournament" not in frame.columns or "tournament_boundary" not in frame.columns:
        return None
    tournament = _slice_labels(frame, "tournament")
    if tournament is None or tournament.eq("<missing>").all():
        return None
    labels = pd.Series("tournament_interior", index=frame.index, dtype="string")
    labels.loc[frame["tournament_boundary"].astype(bool)] = "tournament_boundary"
    labels.loc[tournament.eq("<missing>")] = "<missing>"
    return labels


def _baseline_output_alignment(
    validation: pd.DataFrame,
    output: pd.DataFrame | None,
    *,
    game_id_column: str,
    probability_column: str,
    method: str,
    excluded_game_ids: Iterable[str] = (),
) -> tuple[pd.Series, dict[str, Any]]:
    """Align one research baseline to validation rows and record exclusions."""

    requested_ids = tuple(validation["game_id"].astype(str))
    requested_set = set(requested_ids)
    empty = pd.Series(np.nan, index=validation.index, dtype=float)
    report: dict[str, Any] = {
        "method": method,
        "status": "unavailable",
        "requested_rows": int(len(requested_ids)),
        "requested_game_ids": list(requested_ids),
        "scored_rows": 0,
        "scored_game_ids": [],
        "missing_game_ids": list(requested_ids),
        "extra_game_ids": [],
        "excluded_game_ids": sorted({str(value) for value in excluded_game_ids}),
        "blockers": [f"{method}_prediction_missing"],
    }
    if output is None:
        return empty, report
    required = {game_id_column, probability_column}
    if not required.issubset(output.columns):
        report["blockers"] = [f"{method}_prediction_columns_missing"]
        return empty, report
    output_ids = output[game_id_column].astype(str)
    if output_ids.duplicated().any():
        report["blockers"] = [f"{method}_duplicate_prediction_ids"]
        return empty, report
    output_values = pd.to_numeric(output[probability_column], errors="coerce")
    finite = np.isfinite(output_values.to_numpy(dtype=float))
    valid_probability = finite & output_values.ge(0.0).to_numpy() & output_values.le(1.0).to_numpy()
    invalid_ids = output_ids.loc[~valid_probability].astype(str).tolist()
    output_set = set(output_ids)
    extra_ids = sorted(output_set - requested_set)
    excluded_set = {str(value) for value in excluded_game_ids}
    usable = valid_probability & ~output_ids.isin(excluded_set).to_numpy()
    lookup = pd.Series(
        output_values.loc[usable].to_numpy(dtype=float),
        index=output_ids.loc[usable].astype(str).to_numpy(),
        dtype=float,
    )
    aligned = validation["game_id"].astype(str).map(lookup).astype(float)
    scored_ids = sorted(
        set(validation.loc[aligned.notna(), "game_id"].astype(str))
    )
    missing_ids = sorted(requested_set - set(scored_ids))
    blockers: list[str] = []
    if missing_ids:
        blockers.append(f"{method}_coverage_incomplete")
    if invalid_ids:
        blockers.append(f"{method}_invalid_probability")
    if extra_ids:
        blockers.append(f"{method}_extra_prediction_ids")
    report.update(
        {
            "status": "available" if not blockers else "partial",
            "scored_rows": int(aligned.notna().sum()),
            "scored_game_ids": scored_ids,
            "missing_game_ids": missing_ids,
            "extra_game_ids": extra_ids,
            "invalid_probability_game_ids": sorted(set(invalid_ids)),
            "blockers": sorted(set(blockers)),
        }
    )
    return aligned, report


def _baseline_source_binding(
    method: str,
    baseline_receipt: Mapping[str, Any] | None,
    source_receipt: Mapping[str, Any],
    *,
    train_game_ids: Sequence[str],
    validation_game_ids: Sequence[str],
    strict_cutoff: str,
    expected_implementation_sha256: str,
    expected_config_sha256: str,
    expected_series_source: str | None = None,
    expected_series_authoritative: bool | None = None,
) -> dict[str, Any]:
    """Check source, fold, cutoff, and implementation bindings in a baseline receipt."""

    expected_train_identity = identity_sha256(train_game_ids)
    expected_validation_identity = identity_sha256(validation_game_ids)
    report: dict[str, Any] = {
        "status": "unavailable",
        "source_receipt_sha256": str(source_receipt["receipt_sha256"]),
        "source_as_of": str(source_receipt["source_as_of"]),
        "source_game_count": int(source_receipt["source_game_count"]),
        "source_identity_sha256": str(source_receipt["source_identity_sha256"]),
        "model_eligible_game_count": int(source_receipt["model_eligible_game_count"]),
        "model_eligible_identity_sha256": str(
            source_receipt["model_eligible_identity_sha256"]
        ),
        "accepted_game_id_count": int(len(source_receipt["accepted_game_ids"])),
        "model_eligible_game_id_count": int(
            len(source_receipt["model_eligible_game_ids"])
        ),
        "train_game_identity_sha256": expected_train_identity,
        "validation_game_identity_sha256": expected_validation_identity,
        "strict_cutoff": strict_cutoff,
        "blockers": [],
    }
    if not isinstance(baseline_receipt, Mapping):
        report["blockers"] = [f"{method}_receipt_missing"]
        return report
    blockers: list[str] = []
    if method == "sequential_player_elo":
        source_hash = baseline_receipt.get("source_receipt_sha256")
        eligible_hash = baseline_receipt.get("model_eligible_identity_sha256")
        train_identity = baseline_receipt.get("train_game_identity_sha256")
        validation_identity = baseline_receipt.get("validation_game_identity_sha256")
        actual_cutoff = baseline_receipt.get("strict_cutoff")
        implementation_hash = baseline_receipt.get("implementation_digest")
        config = baseline_receipt.get("rating_config")
        if isinstance(config, Mapping):
            config_hash = hashlib.sha256(_canonical_json_bytes(dict(config))).hexdigest()
        else:
            config_hash = None
    else:
        source = baseline_receipt.get("source")
        source = source if isinstance(source, Mapping) else {}
        source_hash = source.get("receipt_sha256")
        eligible_hash = source.get("model_eligible_identity_sha256")
        train_receipt = baseline_receipt.get("train_receipt")
        train_receipt = train_receipt if isinstance(train_receipt, Mapping) else {}
        validation_receipt = baseline_receipt.get("validation_receipt")
        validation_receipt = (
            validation_receipt if isinstance(validation_receipt, Mapping) else {}
        )
        train_identity = train_receipt.get("identity_sha256")
        validation_identity = validation_receipt.get("identity_sha256")
        actual_cutoff = baseline_receipt.get("cutoff")
        implementation_hash = baseline_receipt.get("implementation_sha256")
        config_hash = baseline_receipt.get("config_sha256")
        series_identity = baseline_receipt.get("series_identity")
        if not isinstance(series_identity, Mapping):
            series_identity = source.get("series_identity")
        if isinstance(series_identity, Mapping):
            report["series_identity"] = dict(series_identity)
        else:
            series_identity = None
    if str(source_hash or "").lower() != str(source_receipt["receipt_sha256"]).lower():
        blockers.append(f"{method}_source_receipt_mismatch")
    if str(eligible_hash or "").lower() != str(
        source_receipt["model_eligible_identity_sha256"]
    ).lower():
        blockers.append(f"{method}_eligible_identity_mismatch")
    if str(train_identity or "").lower() != expected_train_identity:
        blockers.append(f"{method}_train_identity_mismatch")
    if str(validation_identity or "").lower() != expected_validation_identity:
        blockers.append(f"{method}_validation_identity_mismatch")
    if method == "sequential_player_elo":
        if str(actual_cutoff or "") != strict_cutoff:
            blockers.append(f"{method}_strict_cutoff_mismatch")
    if not isinstance(implementation_hash, str) or len(implementation_hash) != 64:
        blockers.append(f"{method}_implementation_hash_missing")
    elif implementation_hash.lower() != expected_implementation_sha256.lower():
        blockers.append(f"{method}_implementation_hash_mismatch")
    if not isinstance(config_hash, str) or len(config_hash) != 64:
        blockers.append(f"{method}_config_hash_missing")
    elif config_hash.lower() != expected_config_sha256.lower():
        blockers.append(f"{method}_config_hash_mismatch")
    if method == "hierarchical_bt":
        if not isinstance(series_identity, Mapping):
            blockers.append(f"{method}_series_identity_missing")
        else:
            source_types = series_identity.get("source_types")
            source_types = (
                [str(value) for value in source_types]
                if isinstance(source_types, Sequence)
                and not isinstance(source_types, (str, bytes, bytearray))
                else []
            )
            if expected_series_source and expected_series_source not in source_types:
                blockers.append(f"{method}_series_source_mismatch")
            if (
                expected_series_authoritative is not None
                and bool(series_identity.get("authoritative"))
                != bool(expected_series_authoritative)
            ):
                blockers.append(f"{method}_series_authority_mismatch")
        fit = baseline_receipt.get("fit")
        fit = fit if isinstance(fit, Mapping) else {}
        if fit.get("optimizer_success") is not True:
            blockers.append(f"{method}_optimizer_not_converged")
        if fit.get("converged") is not True or fit.get("optimizer_status") != 0:
            blockers.append(f"{method}_optimizer_status_invalid")
        if fit.get("finite_fit_evidence") is not True:
            blockers.append(f"{method}_finite_fit_evidence_missing")
        for evidence_name in ("objective_value", "gradient_inf_norm"):
            evidence = fit.get(evidence_name)
            if (
                not isinstance(evidence, (int, float, np.integer, np.floating))
                or not np.isfinite(float(evidence))
            ):
                blockers.append(f"{method}_{evidence_name}_nonfinite")
        if not isinstance(fit.get("optimizer_message"), str) or not str(
            fit.get("optimizer_message")
        ).strip():
            blockers.append(f"{method}_optimizer_evidence_missing")
        terms = baseline_receipt.get("terms")
        term_values: list[Any] = []
        if isinstance(terms, Mapping):
            for value in terms.values():
                if isinstance(value, Mapping):
                    term_values.extend(value.values())
                else:
                    term_values.append(value)
        if not term_values or any(
            not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(float(value))
            for value in term_values
        ):
            blockers.append(f"{method}_fit_terms_nonfinite")
    report.update(
        {
            "status": "available" if not blockers else "blocked",
            "implementation_sha256": str(implementation_hash or ""),
            "config_sha256": str(config_hash or ""),
            "baseline_receipt": dict(baseline_receipt),
            "blockers": sorted(set(blockers)),
        }
    )
    return report


def _bind_baseline_fold_series(
    maps: pd.DataFrame,
    *,
    requested_game_ids: set[str],
    full_map_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any], str]:
    """Subset prevalidated full-source series assignments for one baseline fold."""

    raw_map_ids = _frame_game_ids(maps, "maps").astype(str)
    fold_maps = maps.loc[raw_map_ids.isin(requested_game_ids)].copy().reset_index(
        drop=True
    )
    fold_map_ids = _frame_game_ids(fold_maps, "maps").astype(str)
    full_series = full_map_frame.set_index("game_id", drop=False)
    fold_series = full_series.reindex(fold_map_ids.to_numpy())
    if fold_series["series_id"].isna().any():
        raise FutureValueSourceError("baseline fold series assignments are missing")
    series_source = str(full_map_frame.attrs.get("series_cluster_source", ""))
    series_authoritative = series_source.startswith("authoritative:")
    series_id_column = "grid_series_id" if series_authoritative else "series_id"
    fold_maps[series_id_column] = fold_series["series_id"].astype(str).to_numpy()
    fold_maps["series_id_source"] = series_source
    subset_sizes = fold_series["series_id"].astype(str).value_counts(sort=False)
    series_cluster = {
        "source": series_source,
        "audit": {
            **dict(full_map_frame.attrs.get("series_cluster_audit") or {}),
            "fold_subset_map_count": int(len(fold_series)),
            "fold_subset_cluster_count": int(len(subset_sizes)),
            "fold_subset_max_cluster_size": int(subset_sizes.max()),
        },
        "authoritative": series_authoritative,
        "series_id_column": series_id_column,
    }
    return fold_maps, fold_map_ids, series_cluster, series_id_column


def _run_current_rating_baselines(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    train_game_ids: Sequence[str],
    validation_game_ids: Sequence[str],
    strict_cutoff: str,
    source_receipt: Mapping[str, Any],
    full_map_frame: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Run fold-local current rating baselines without production artifacts."""

    from lol_kills.ratings.hierarchical_bt import (
        HIERARCHICAL_IMPLEMENTATION_SHA256,
        HierarchicalBTConfig,
        fit_hierarchical_bt_research_prediction,
    )
    from lol_kills.ratings.player_elo import (
        PlayerEloConfig,
        _sequential_baseline_implementation_digest,
        build_sequential_player_elo_baseline,
    )

    requested = set(str(value) for value in (*train_game_ids, *validation_game_ids))
    fold_maps, fold_map_ids, series_cluster, series_id_column = (
        _bind_baseline_fold_series(
            maps,
            requested_game_ids=requested,
            full_map_frame=full_map_frame,
        )
    )
    fold_players = players.copy().reset_index(drop=True)
    validation_ids = validation["game_id"].astype(str)
    reports: dict[str, Any] = {}
    errors: dict[str, str] = {}
    sequential_implementation_sha256 = _sequential_baseline_implementation_digest()
    sequential_config_sha256 = hashlib.sha256(
        _canonical_json_bytes(dict(PlayerEloConfig().__dict__))
    ).hexdigest()
    hierarchical_config_sha256 = hashlib.sha256(
        _canonical_json_bytes(dict(HierarchicalBTConfig().__dict__))
    ).hexdigest()
    sequential_output: pd.DataFrame | None = None
    sequential_receipt: Mapping[str, Any] | None = None
    try:
        sequential_output, sequential_receipt = build_sequential_player_elo_baseline(
            fold_maps,
            fold_players,
            train_game_ids=train_game_ids,
            validation_game_ids=validation_game_ids,
            strict_cutoff=strict_cutoff,
            source_receipt=source_receipt,
        )
    except Exception as error:
        errors["sequential_player_elo"] = f"{type(error).__name__}: {error}"
    seq_binding = _baseline_source_binding(
        "sequential_player_elo",
        sequential_receipt,
        source_receipt,
        train_game_ids=train_game_ids,
        validation_game_ids=validation_game_ids,
        strict_cutoff=strict_cutoff,
        expected_implementation_sha256=sequential_implementation_sha256,
        expected_config_sha256=sequential_config_sha256,
    )
    if sequential_receipt is None:
        seq_binding["implementation_sha256"] = sequential_implementation_sha256
        seq_binding["config_sha256"] = sequential_config_sha256
    seq_probability, seq_alignment = _baseline_output_alignment(
        validation,
        sequential_output,
        game_id_column="game_uid",
        probability_column="p_player_elo",
        method="sequential_player_elo",
    )
    if seq_binding["status"] != "available":
        seq_probability.loc[:] = np.nan
        seq_alignment["blockers"] = sorted(
            set(seq_alignment["blockers"]) | set(seq_binding["blockers"])
        )
        seq_alignment["status"] = "blocked"
    if "sequential_player_elo" in errors:
        seq_alignment["error"] = errors["sequential_player_elo"]
    reports["sequential_player_elo"] = {
        **seq_alignment,
        "source_binding": seq_binding,
    }

    hierarchical_output: pd.DataFrame | None = None
    hierarchical_receipt: Mapping[str, Any] | None = None
    hierarchical_cutoff = pd.Timestamp(strict_cutoff) - pd.Timedelta(nanoseconds=1)
    try:
        import inspect

        hierarchical_kwargs: dict[str, Any] = {}
        if "series_id_column" in inspect.signature(
            fit_hierarchical_bt_research_prediction
        ).parameters:
            hierarchical_kwargs["series_id_column"] = series_id_column
        hierarchical_receipt = fit_hierarchical_bt_research_prediction(
            fold_maps[fold_map_ids.isin(set(train_game_ids))].copy(),
            fold_maps[fold_map_ids.isin(set(validation_game_ids))].copy(),
            cutoff=hierarchical_cutoff,
            source_receipt=source_receipt,
            source_identity_sha256=str(source_receipt["source_identity_sha256"]),
            train_source_identity_sha256=identity_sha256(train_game_ids),
            validation_source_identity_sha256=identity_sha256(validation_game_ids),
            **hierarchical_kwargs,
        )
        hierarchical_output = pd.DataFrame(hierarchical_receipt.get("predictions", []))
    except Exception as error:
        errors["hierarchical_bt"] = f"{type(error).__name__}: {error}"
    hierarchical_binding = _baseline_source_binding(
        "hierarchical_bt",
        hierarchical_receipt,
        source_receipt,
        train_game_ids=train_game_ids,
        validation_game_ids=validation_game_ids,
        strict_cutoff=hierarchical_cutoff.isoformat(),
        expected_implementation_sha256=HIERARCHICAL_IMPLEMENTATION_SHA256,
        expected_config_sha256=hierarchical_config_sha256,
        expected_series_source=str(series_cluster.get("source") or ""),
        expected_series_authoritative=bool(series_cluster.get("authoritative")),
    )
    if hierarchical_receipt is None:
        hierarchical_binding["implementation_sha256"] = HIERARCHICAL_IMPLEMENTATION_SHA256
        hierarchical_binding["config_sha256"] = hierarchical_config_sha256
    excluded_hierarchical: Iterable[str] = ()
    if isinstance(hierarchical_receipt, Mapping):
        missing = hierarchical_receipt.get("missing")
        if isinstance(missing, Mapping):
            excluded_hierarchical = tuple(
                str(value) for value in missing.get("unseen_model_game_ids", [])
            )
        excluded_hierarchical = tuple(
            {*excluded_hierarchical, *map(str, hierarchical_receipt.get("missing_ids", []))}
        )
    hierarchical_probability, hierarchical_alignment = _baseline_output_alignment(
        validation,
        hierarchical_output,
        game_id_column="game_id",
        probability_column="predicted_blue_win",
        method="hierarchical_bt",
        excluded_game_ids=excluded_hierarchical,
    )
    # Keep the exclusion reason from the hierarchical research receipt.  The
    # finite prediction count is a valid partial comparison when validation
    # teams were unseen in the training fold.  The reason must stay visible in
    # the method-specific evidence so downstream reports do not treat the
    # missing rows as imputed values.
    if isinstance(hierarchical_receipt, Mapping):
        missing = hierarchical_receipt.get("missing")
        if isinstance(missing, Mapping):
            missing_blockers = missing.get("blockers")
            if isinstance(missing_blockers, Sequence) and not isinstance(
                missing_blockers, (str, bytes, bytearray)
            ):
                hierarchical_alignment["blockers"] = sorted(
                    {str(value) for value in hierarchical_alignment["blockers"]}
                    | {str(value) for value in missing_blockers}
                )
            if missing.get("unseen_model_game_ids"):
                hierarchical_alignment["exclusion_reason"] = (
                    "validation rows with unseen teams are excluded"
                )
                hierarchical_alignment["unseen_team_keys"] = list(
                    missing.get("unseen_team_keys", [])
                )
    if hierarchical_binding["status"] != "available":
        hierarchical_probability.loc[:] = np.nan
        hierarchical_alignment["blockers"] = sorted(
            set(hierarchical_alignment["blockers"])
            | set(hierarchical_binding["blockers"])
        )
        hierarchical_alignment["status"] = "blocked"
    if "hierarchical_bt" in errors:
        hierarchical_alignment["error"] = errors["hierarchical_bt"]
        if "grid_series_id" in errors["hierarchical_bt"]:
            hierarchical_alignment["blockers"] = sorted(
                set(hierarchical_alignment["blockers"])
                | {"hierarchical_bt_authoritative_series_id_missing"}
            )
    reports["hierarchical_bt"] = {
        **hierarchical_alignment,
        "source_binding": hierarchical_binding,
        "series_cluster": series_cluster,
    }
    reports["errors"] = errors
    return seq_probability, hierarchical_probability, reports


def _ledger_value(value: Any) -> float | None:
    """Return one finite JSON-safe ledger value."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _current_rating_method_comparison(
    validation: pd.DataFrame,
    target: pd.Series,
    paired_mask: pd.Series,
    sequential_probability: pd.Series,
    hierarchical_probability: pd.Series,
    current_reports: Mapping[str, Any],
) -> dict[str, Any]:
    """Report each current-rating method on its own finite paired cohort.

    The sequential player Elo baseline and hierarchical BT baseline have
    different coverage contracts.  The method rows stay separate.  The
    shared cohort is retained for an all-method comparison and never receives
    an imputed hierarchical probability.
    """

    paired_ids_set = set(validation.loc[paired_mask, "game_id"].astype(str))
    current_mask = (
        paired_mask
        & sequential_probability.notna()
        & hierarchical_probability.notna()
    )
    current_ids = tuple(
        sorted(validation.loc[current_mask, "game_id"].astype(str))
    )
    common_ids_set = set(current_ids)
    current_blockers: list[str] = []
    for method_name in ("sequential_player_elo", "hierarchical_bt"):
        method_report = current_reports[method_name]
        if method_report.get("status") != "available":
            current_blockers.extend(str(value) for value in method_report["blockers"])
        binding = method_report.get("source_binding", {})
        if binding.get("status") != "available":
            current_blockers.extend(str(value) for value in binding.get("blockers", []))
    if common_ids_set != paired_ids_set:
        current_blockers.append("current_rating_row_id_parity_incomplete")

    method_specific: dict[str, dict[str, Any]] = {}
    for method_name, values in (
        ("sequential_player_elo", sequential_probability),
        ("hierarchical_bt", hierarchical_probability),
    ):
        method_report = current_reports[method_name]
        method_binding = method_report.get("source_binding", {})
        method_mask = paired_mask & values.notna()
        method_target = target.loc[method_mask]
        method_values = values.loc[method_mask]
        method_ids = sorted(
            set(validation.loc[method_mask, "game_id"].astype(str))
        )
        method_blockers = {
            str(value) for value in method_report.get("blockers", [])
        }
        method_blockers.update(
            str(value) for value in method_binding.get("blockers", [])
        )
        method_requested_rows = int(paired_mask.sum())
        method_scored_rows = int(len(method_values))
        if (
            method_report.get("status") == "available"
            and method_binding.get("status") == "available"
            and method_scored_rows == method_requested_rows
            and not method_blockers
        ):
            method_status = "available"
        elif method_scored_rows:
            method_status = "partial"
        else:
            method_status = "blocked"
        method_specific[method_name] = {
            "status": method_status,
            "requested_rows": method_requested_rows,
            "scored_rows": method_scored_rows,
            "scored_game_ids": method_ids,
            "missing_game_ids": sorted(set(paired_ids_set) - set(method_ids)),
            "metrics": _classification_metrics(method_target, method_values),
            "calibration": _calibration_metrics(method_target, method_values),
            "source_binding_status": str(
                method_binding.get("status") or "unavailable"
            ),
            "blockers": sorted(method_blockers),
            "exclusion_reason": method_report.get("exclusion_reason"),
        }
    return {
        "current_mask": current_mask,
        "common_ids": current_ids,
        "blockers": sorted(set(current_blockers)),
        "method_specific": method_specific,
        "common_all_method": {
            "status": "available" if not current_blockers else "blocked",
            "rows": int(current_mask.sum()),
            "game_ids": list(current_ids),
            "blockers": sorted(set(current_blockers)),
        },
    }


def evaluate_future_value(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    *,
    n_folds: int = 3,
    half_life_days: float = TIME_DECAY_HALF_LIFE_DAYS,
    min_cell_support: int = 1,
    source_receipt: Mapping[str, Any] | None = None,
    source_receipt_path: str | None = None,
    source_receipt_file_sha256: str | None = None,
    crosswalk_receipt_file_sha256: str | None = None,
    runtime_receipt_path: str | None = None,
    variant: RatingVariant | str | None = None,
    feature_ledger: pd.DataFrame | Mapping[Any, pd.DataFrame] | None = None,
    inner_feature_ledger: pd.DataFrame | Mapping[Any, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Run a development-only chronological whole-series evaluation."""

    variant_config = None if variant is None else rating_variant_config(variant)
    if variant_config is not None and isinstance(feature_ledger, Mapping) and "outer" in feature_ledger:
        nested_outer = feature_ledger.get("outer")
        nested_inner = feature_ledger.get("inner")
        if not isinstance(nested_outer, Mapping) or not isinstance(nested_inner, Mapping):
            raise FutureValueSourceError("nested feature ledger binding is invalid")
        if inner_feature_ledger is not None:
            raise FutureValueSourceError(
                "nested feature ledger inner binding was supplied twice"
            )
        feature_ledger = nested_outer
        inner_feature_ledger = nested_inner
    if maps.attrs.get("verified_leaguepedia_series_crosswalk") is not None:
        map_frame = _map_model_frame(
            maps,
            verified_source_receipt=(
                source_receipt if isinstance(source_receipt, Mapping) else None
            ),
            verified_source_receipt_sha256=(
                str(source_receipt["receipt_sha256"])
                if isinstance(source_receipt, Mapping)
                and "receipt_sha256" in source_receipt
                else None
            ),
            verified_crosswalk_receipt_file_sha256=crosswalk_receipt_file_sha256,
        )
    else:
        map_frame = _map_model_frame(maps)
    verified_eligible_ids = _validate_verified_source_receipt(
        source_receipt,
        map_frame,
        require_full_eligible_set=True,
    )
    if isinstance(maps.attrs.get("verified_series_receipt"), Mapping) or isinstance(
        maps.attrs.get("verified_leaguepedia_series_crosswalk"), Mapping
    ):
        map_frame = _map_model_frame(
            maps,
            verified_source_receipt_sha256=str(source_receipt["receipt_sha256"]),
            verified_source_receipt=source_receipt,
            verified_crosswalk_receipt_file_sha256=crosswalk_receipt_file_sha256,
        )
        _validate_verified_source_receipt(
            source_receipt,
            map_frame,
            require_full_eligible_set=True,
        )
    if (source_receipt_path is None) != (source_receipt_file_sha256 is None):
        raise FutureValueSourceError("source receipt path and file hash must be paired")
    if source_receipt_path is not None:
        durable_receipt = Path(source_receipt_path)
        if not durable_receipt.is_file() or durable_receipt.is_symlink():
            raise FutureValueSourceError("durable source receipt is missing or unsafe")
        actual_receipt_file_sha256 = _sha256_path(durable_receipt)
        if actual_receipt_file_sha256 != str(source_receipt_file_sha256).lower():
            raise FutureValueSourceError("durable source receipt file hash changed")
        try:
            durable_payload = json.loads(durable_receipt.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FutureValueSourceError("durable source receipt cannot be read") from error
        if not isinstance(durable_payload, Mapping):
            raise FutureValueSourceError("durable source receipt payload changed")
        validate_future_value_source_receipt_payload(
            durable_payload,
            expected_receipt_sha256=str(source_receipt.get("receipt_sha256") or ""),
        )
        if durable_payload != dict(source_receipt):
            raise FutureValueSourceError("durable source receipt payload changed")
    form = build_time_decayed_prior_player_form(map_frame, players, half_life_days=half_life_days)
    folds = chronological_whole_series_folds(
        map_frame,
        n_folds=n_folds,
        verified_model_frame=map_frame,
    )
    fold_reports: list[dict[str, Any]] = []
    pooled_targets: list[pd.Series] = []
    pooled_predictions: list[pd.Series] = []
    pooled_raw_predictions: list[pd.Series] = []
    pooled_baselines: list[pd.Series] = []
    calibration_history_logits: list[float] = []
    calibration_history_targets: list[int] = []
    calibration_history_folds: list[int] = []
    calibration_history_game_ids: list[str] = []
    calibration_history_ends: list[str] = []
    pooled_slice_blockers: list[str] = []
    pooled_current_targets: list[pd.Series] = []
    pooled_current_predictions: dict[str, list[pd.Series]] = {
        "candidate": [],
        "raw_candidate": [],
        "intercept_baseline": [],
        "sequential_player_elo": [],
        "hierarchical_bt": [],
    }
    pooled_candidate_paired_targets: dict[str, list[pd.Series]] = {
        "candidate": [],
        "raw_candidate": [],
        "intercept_baseline": [],
        "sequential_player_elo": [],
        "hierarchical_bt": [],
    }
    pooled_candidate_paired_predictions: dict[str, list[pd.Series]] = {
        "candidate": [],
        "raw_candidate": [],
        "intercept_baseline": [],
        "sequential_player_elo": [],
        "hierarchical_bt": [],
    }
    current_fold_reports: list[dict[str, Any]] = []
    prediction_ledger_rows: list[dict[str, Any]] = []
    for fold in folds:
        fold_model_ids = set(
            str(value)
            for value in (*fold["train_game_ids"], *fold["validation_game_ids"])
        )
        fold_map_frame = map_frame[
            map_frame["game_id"].astype(str).isin(fold_model_ids)
        ].copy()
        fold_form = form[form["game_id"].astype(str).isin(fold_model_ids)].copy()
        if (
            set(fold_map_frame["game_id"].astype(str)) != fold_model_ids
            or set(fold_form["game_id"].astype(str)) != fold_model_ids
        ):
            raise FutureValueSourceError("future-value fold source rows are incomplete")
        fold_ledger: pd.DataFrame | None
        if isinstance(feature_ledger, Mapping):
            fold_ledger = feature_ledger.get(fold["fold"])
            if fold_ledger is None:
                fold_ledger = feature_ledger.get(str(fold["fold"]))
        else:
            fold_ledger = feature_ledger
        fold_inner_ledger: pd.DataFrame | None
        if isinstance(inner_feature_ledger, Mapping):
            fold_inner_ledger = inner_feature_ledger.get(fold["fold"])
            if fold_inner_ledger is None:
                fold_inner_ledger = inner_feature_ledger.get(str(fold["fold"]))
        else:
            fold_inner_ledger = inner_feature_ledger
        model, design = fit_future_value_model(
            fold_map_frame,
            fold_form,
            train_game_ids=fold["train_game_ids"],
            fit_window_end=fold["validation_start"],
            min_cell_support=min_cell_support,
            source_receipt=source_receipt,
            crosswalk_receipt_file_sha256=crosswalk_receipt_file_sha256,
            verified_model_frame=fold_map_frame,
            variant=None if variant_config is None else variant_config.variant,
            feature_ledger=fold_ledger,
            inner_feature_ledger=fold_inner_ledger,
        )
        validation = design[design["game_id"].isin(fold["validation_game_ids"])].copy()
        raw_logit = model.predict_logit(validation)
        raw_prediction = model.predict_probability(validation)
        calibration_fit = _fit_strict_prior_calibration(
            np.asarray(calibration_history_logits, dtype=float),
            np.asarray(calibration_history_targets, dtype=int),
            source_receipt_sha256=str(source_receipt["receipt_sha256"]),
            current_fold=int(fold["fold"]),
            current_validation_game_ids=tuple(
                str(value) for value in fold["validation_game_ids"]
            ),
            current_validation_start=str(fold["validation_start"]),
            prior_fold_numbers=tuple(calibration_history_folds),
            prior_game_ids=tuple(calibration_history_game_ids),
            prior_validation_ends=tuple(calibration_history_ends),
        )
        calibrated_logit, prediction = _apply_strict_prior_calibration(
            raw_logit.to_numpy(dtype=float),
            raw_prediction.to_numpy(dtype=float),
            calibration_fit,
        )
        calibrated_logit.index = validation.index
        prediction.index = validation.index
        target = validation["target"].astype(float)
        paired_mask = target.notna() & prediction.notna() & raw_logit.notna()
        paired_target = target.loc[paired_mask]
        paired_prediction = prediction.loc[paired_mask]
        paired_raw_prediction = raw_prediction.loc[paired_mask]
        paired_raw_logit = raw_logit.loc[paired_mask]
        baseline_probability = pd.Series(
            float(design.loc[design["game_id"].isin(fold["train_game_ids"]), "target"].mean()),
            index=validation.index,
        )
        paired_baseline = baseline_probability.loc[paired_mask]
        if not (
            paired_target.index.equals(paired_prediction.index)
            and paired_target.index.equals(paired_baseline.index)
        ):
            raise FutureValueSourceError("candidate and baseline rows are not paired")
        sequential_probability, hierarchical_probability, current_reports = (
            _run_current_rating_baselines(
                maps,
                players,
                validation,
                train_game_ids=tuple(str(value) for value in fold["train_game_ids"]),
                validation_game_ids=tuple(
                    str(value) for value in fold["validation_game_ids"]
                ),
                strict_cutoff=str(fold["validation_start"]),
                source_receipt=source_receipt,
                full_map_frame=map_frame,
            )
        )
        current_evidence = _current_rating_method_comparison(
            validation,
            target,
            paired_mask,
            sequential_probability,
            hierarchical_probability,
            current_reports,
        )
        current_mask = current_evidence["current_mask"]
        current_ids = tuple(current_evidence["common_ids"])
        common_ids_set = set(current_ids)
        current_blockers = list(current_evidence["blockers"])
        method_specific_current_methods = current_evidence["method_specific"]
        paired_ids_set = set(validation.loc[paired_mask, "game_id"].astype(str))
        common_target = target.loc[current_mask]
        common_predictions = {
            "candidate": prediction.loc[current_mask],
            "raw_candidate": raw_prediction.loc[current_mask],
            "intercept_baseline": baseline_probability.loc[current_mask],
            "sequential_player_elo": sequential_probability.loc[current_mask],
            "hierarchical_bt": hierarchical_probability.loc[current_mask],
        }
        current_methods = {
            method_name: {
                "metrics": _classification_metrics(common_target, values),
                "calibration": _calibration_metrics(common_target, values),
                "rows": int(len(values)),
                "game_ids": list(current_ids),
            }
            for method_name, values in common_predictions.items()
        }
        candidate_paired_methods = {
            "candidate": prediction,
            "raw_candidate": raw_prediction,
            "intercept_baseline": baseline_probability,
            "sequential_player_elo": sequential_probability,
            "hierarchical_bt": hierarchical_probability,
        }
        candidate_paired_method_reports: dict[str, Any] = {}
        for method_name, values in candidate_paired_methods.items():
            method_mask = paired_mask & values.notna()
            method_target = target.loc[method_mask]
            method_values = values.loc[method_mask]
            candidate_paired_method_reports[method_name] = {
                "rows": int(len(method_values)),
                "game_ids": sorted(
                    set(validation.loc[method_mask, "game_id"].astype(str))
                ),
                "metrics": _classification_metrics(method_target, method_values),
                "calibration": _calibration_metrics(method_target, method_values),
            }
            if len(method_values):
                pooled_candidate_paired_targets[method_name].append(method_target)
                pooled_candidate_paired_predictions[method_name].append(method_values)
        finite_ids = {
            "candidate": sorted(
                set(validation.loc[paired_mask, "game_id"].astype(str))
            ),
            "intercept_baseline": sorted(
                set(validation.loc[paired_mask, "game_id"].astype(str))
            ),
            "sequential_player_elo": sorted(
                set(validation.loc[sequential_probability.notna(), "game_id"].astype(str))
            ),
            "hierarchical_bt": sorted(
                set(validation.loc[hierarchical_probability.notna(), "game_id"].astype(str))
            ),
        }
        current_complete = (
            not current_blockers
            and len(current_ids) == len(paired_target)
            and all(
                current_reports[method_name].get("status") == "available"
                and current_reports[method_name]
                .get("source_binding", {})
                .get("status")
                == "available"
                for method_name in ("sequential_player_elo", "hierarchical_bt")
            )
        )
        current_comparison = {
            "status": "available" if current_complete else "blocked",
            "strict_cutoff": str(fold["validation_start"]),
            "requested_paired_rows": int(len(paired_target)),
            "common_finite_rows": int(len(current_ids)),
            "common_finite_game_ids": list(current_ids),
            "excluded_paired_game_ids": sorted(paired_ids_set - common_ids_set),
            "method_finite_game_ids": finite_ids,
            "row_id_parity": {
                "common_ids_equal_for_all_methods": bool(
                    all(values == list(current_ids) for values in finite_ids.values())
                ),
                "scored_game_ids": list(current_ids),
            },
            "methods": current_methods,
            "method_specific": method_specific_current_methods,
            "common_all_method": {
                **dict(current_evidence["common_all_method"]),
                "methods": current_methods,
            },
            "candidate_paired_methods": candidate_paired_method_reports,
            "baselines": {
                method_name: current_reports[method_name]
                for method_name in ("sequential_player_elo", "hierarchical_bt")
            },
            "errors": dict(current_reports.get("errors", {})),
            "series_cluster": current_reports["hierarchical_bt"].get(
                "series_cluster"
            ),
            "blockers": sorted(set(current_blockers)),
        }
        for row_index, game_id in zip(validation.index, validation["game_id"]):
            prediction_ledger_rows.append(
                {
                    "fold": int(fold["fold"]),
                    "game_id": str(game_id),
                    "target": _ledger_value(target.loc[row_index]),
                    "candidate": _ledger_value(prediction.loc[row_index]),
                    "candidate_raw_probability": _ledger_value(
                        raw_prediction.loc[row_index]
                    ),
                    "candidate_raw_logit": _ledger_value(raw_logit.loc[row_index]),
                    "candidate_calibrated_logit": _ledger_value(
                        calibrated_logit.loc[row_index]
                    ),
                    "calibration_slope": float(calibration_fit["slope"]),
                    "intercept": _ledger_value(baseline_probability.loc[row_index]),
                    "sequential_player_elo": _ledger_value(
                        sequential_probability.loc[row_index]
                    ),
                    "hierarchical_bt": _ledger_value(
                        hierarchical_probability.loc[row_index]
                    ),
                    "minimum_metric_support": _ledger_value(
                        validation.loc[
                            row_index, "player_form_minimum_metric_support"
                        ]
                    ),
                    "minimum_effective_support": _ledger_value(
                        validation.loc[
                            row_index, "player_form_minimum_effective_support"
                        ]
                    ),
                    "minimum_atom_support": _ledger_value(
                        validation.loc[
                            row_index, "rank_3_champion_role_minimum_support"
                        ]
                    ),
                    "missing_feature_names": list(
                        validation.loc[row_index, "model_missing_feature_names"]
                    ),
                    "support_status": str(
                        _support_labels(validation.loc[[row_index]]).iloc[0]
                    ),
                }
            )
        current_fold_reports.append(current_comparison)
        if len(current_ids):
            pooled_current_targets.append(common_target)
            for method_name, values in common_predictions.items():
                pooled_current_predictions[method_name].append(values)
        train_design = design[design["game_id"].isin(fold["train_game_ids"])].copy()
        calibration = _calibration_metrics(paired_target, paired_prediction)
        raw_calibration = _calibration_metrics(paired_target, paired_raw_prediction)
        baseline_calibration = _calibration_metrics(paired_target, paired_baseline)
        region_slice = _group_slice_metrics(
            paired_target,
            paired_prediction,
            _slice_labels(validation, "league"),
            _slice_labels(train_design, "league"),
            slice_name="regional_transfer",
        )
        patch_field = (
            "patch"
            if "patch" in validation.columns
            else "oe_patch_token"
            if "oe_patch_token" in validation.columns
            else "patch"
        )
        patch_slice = _group_slice_metrics(
            paired_target,
            paired_prediction,
            _slice_labels(validation, patch_field),
            _slice_labels(train_design, patch_field),
            slice_name="patch_transfer",
        )
        roster_slice = _group_slice_metrics(
            paired_target,
            paired_prediction,
            _roster_change_labels(validation),
            _roster_change_labels(train_design),
            slice_name="roster_change",
        )
        tournament_slice = _group_slice_metrics(
            paired_target,
            paired_prediction,
            _tournament_boundary_labels(validation),
            _tournament_boundary_labels(train_design),
            slice_name="tournament_boundary",
        )
        support_slice = _group_slice_metrics(
            paired_target,
            paired_prediction,
            _support_labels(validation),
            _support_labels(train_design),
            slice_name="sparse_support",
        )
        missingness = _missingness_metrics(
            validation,
            target,
            prediction,
            raw_probability=raw_prediction,
        )
        side_swap = _side_swap_metrics(
            model,
            validation,
            target,
            prediction,
            raw_probability=raw_prediction,
            calibration_slope=float(calibration_fit["slope"]),
        )
        slice_reports = {
            "regional_transfer": region_slice,
            "patch_transfer": patch_slice,
            "roster_change": roster_slice,
            "tournament_boundary": tournament_slice,
            "sparse_support": support_slice,
        }
        for report in (*slice_reports.values(), missingness, side_swap):
            pooled_slice_blockers.extend(str(value) for value in report.get("blockers", []))
        pooled_slice_blockers.extend(
            str(value)
            for value in model.regularization_selection.get("blockers", [])
        )
        paired_game_ids = tuple(
            sorted(validation.loc[paired_mask, "game_id"].astype(str))
        )
        pooled_targets.append(paired_target)
        pooled_predictions.append(paired_prediction)
        pooled_raw_predictions.append(paired_raw_prediction)
        pooled_baselines.append(paired_baseline)
        model_parameters = model.parameter_receipt()
        component_frame = model.player_value_logit(fold_form, validation)
        component_rows = [
            {
                "game_id": str(row.game_id),
                "raw_player_value_logit": float(row.player_value_logit),
                "raw_team_context_logit": float(row.team_context_logit),
                "raw_current_rating_logit": float(row.current_rating_logit),
                "raw_scaling_curve_logit": float(row.scaling_curve_logit),
                "raw_data_quality_logit": float(row.data_quality_logit),
                "raw_full_model_logit": float(row.full_model_logit),
                "player_value_logit": float(row.player_value_logit),
                "team_context_logit": float(row.team_context_logit),
                "current_rating_logit": float(row.current_rating_logit),
                "scaling_curve_logit": float(row.scaling_curve_logit),
                "data_quality_logit": float(row.data_quality_logit),
                "full_model_logit": float(row.full_model_logit),
                "calibration_slope": float(calibration_fit["slope"]),
                "calibrated_player_value_logit": float(
                    row.player_value_logit * calibration_fit["slope"]
                ),
                "calibrated_team_context_logit": float(
                    row.team_context_logit * calibration_fit["slope"]
                ),
                "calibrated_current_rating_logit": float(
                    row.current_rating_logit * calibration_fit["slope"]
                ),
                "calibrated_scaling_curve_logit": float(
                    row.scaling_curve_logit * calibration_fit["slope"]
                ),
                "calibrated_data_quality_logit": float(
                    row.data_quality_logit * calibration_fit["slope"]
                ),
                "calibrated_full_model_logit": float(
                    row.full_model_logit * calibration_fit["slope"]
                ),
                "component_reconstruction_error": float(
                    row.component_reconstruction_error
                ),
                "calibrated_component_reconstruction_error": float(
                    row.component_reconstruction_error * calibration_fit["slope"]
                ),
                "support_status": str(row.support_status),
                "player_support_records": row.player_support_records,
            }
            for row in component_frame.itertuples(index=False)
        ]
        component_sha256 = hashlib.sha256(
            _canonical_json_bytes(component_rows)
        ).hexdigest()
        fold_reports.append(
            {
                "fold": fold["fold"],
                "train_end": fold["train_end"],
                "validation_start": fold["validation_start"],
                "validation_end": fold["validation_end"],
                "validation_interval_start": fold["validation_interval_start"],
                "validation_interval_end": fold["validation_interval_end"],
                "validation_overlap_audit": fold["overlap_audit"],
                "train_series_count": len(fold["train_series_ids"]),
                "validation_series_count": len(fold["validation_series_ids"]),
                "candidate": _classification_metrics(paired_target, paired_prediction),
                "raw_candidate": _classification_metrics(
                    paired_target, paired_raw_prediction
                ),
                "intercept_baseline": _classification_metrics(paired_target, paired_baseline),
                "paired_rows": int(len(paired_target)),
                "paired_game_id_count": len(paired_game_ids),
                "paired_game_ids": list(paired_game_ids),
                "train_game_id_count": len(fold["train_game_ids"]),
                "train_game_identity_sha256": identity_sha256(fold["train_game_ids"]),
                "validation_game_id_count": len(fold["validation_game_ids"]),
                "validation_game_identity_sha256": identity_sha256(
                    fold["validation_game_ids"]
                ),
                "model_intercept": float(model.intercept),
                "variant": model.variant.value if model.variant is not None else None,
                "variant_receipt": (
                    rating_variant_config_receipt(model.variant)
                    if model.variant is not None
                    else None
                ),
                "feature_ledger_binding": dict(model.feature_ledger_binding or {}),
                "feature_means": model_parameters["feature_means"],
                "feature_scales": model_parameters["feature_scales"],
                "fold_local_side_imputation": model_parameters[
                    "fold_local_side_imputation"
                ],
                "imputation_policy": model_parameters["imputation_policy"],
                "antisymmetric_fit": model_parameters["antisymmetric_fit"],
                "regularization_selection": model_parameters[
                    "regularization_selection"
                ],
                "optimizer_evidence": model_parameters["optimizer_evidence"],
                "coefficients": model_parameters["coefficients"],
                "model_parameter_sha256": model_parameters["parameter_sha256"],
                "rank_3": model_parameters["rank_3"],
                "prediction_coverage": float(prediction.notna().mean()),
                "withheld_rows": int(prediction.isna().sum()),
                "metric_weights": model.metric_weights,
                "calibration": calibration,
                "raw_calibration": raw_calibration,
                "calibration_fit": calibration_fit,
                "baseline_calibration": baseline_calibration,
                "regional_transfer": region_slice,
                "patch_transfer": patch_slice,
                "roster_change": roster_slice,
                "tournament_boundary": tournament_slice,
                "sparse_support": support_slice,
                "missingness": missingness,
                "side_swap": side_swap,
                "current_rating_comparison": current_comparison,
                "component_evidence": {
                    "schema_version": "scryglass:future-value-logit-components:v1",
                    "row_count": len(component_rows),
                    "sha256": component_sha256,
                    "maximum_absolute_reconstruction_error": float(
                        component_frame["component_reconstruction_error"]
                        .abs()
                        .max()
                    ),
                    "maximum_absolute_calibrated_reconstruction_error": float(
                        component_frame["component_reconstruction_error"]
                        .abs()
                        .max()
                        * float(calibration_fit["slope"])
                    ),
                    "calibration_fit": calibration_fit,
                    "rows": component_rows,
                },
            }
        )
        calibration_history_logits.extend(float(value) for value in paired_raw_logit)
        calibration_history_targets.extend(int(value) for value in paired_target)
        calibration_history_folds.extend([int(fold["fold"])] * len(paired_raw_logit))
        calibration_history_game_ids.extend(
            str(value) for value in validation.loc[paired_mask, "game_id"]
        )
        calibration_history_ends.append(str(fold["validation_end"]))
    cluster_source = map_frame.attrs.get("series_cluster_source")
    cluster_audit = map_frame.attrs.get("series_cluster_audit")
    blockers = [
        "support_uncertainty_proxy_not_calibrated",
    ]
    if any(report["status"] != "available" for report in current_fold_reports):
        blockers.append("current_player_team_rating_comparison_missing")
    if int(n_folds) < 3 or len(fold_reports) < int(n_folds):
        blockers.append("complete_chronological_evaluation_missing")
    if int(n_folds) < 3:
        blockers.append("bounded_fold_count_below_protocol")
    pooled_target = pd.concat(pooled_targets, ignore_index=True)
    pooled_prediction = pd.concat(pooled_predictions, ignore_index=True)
    pooled_raw_prediction = pd.concat(pooled_raw_predictions, ignore_index=True)
    pooled_baseline = pd.concat(pooled_baselines, ignore_index=True)
    pooled_calibration_report = _calibration_metrics(pooled_target, pooled_prediction)
    pooled_raw_calibration_report = _calibration_metrics(
        pooled_target, pooled_raw_prediction
    )
    pooled_baseline_calibration_report = _calibration_metrics(pooled_target, pooled_baseline)
    blockers.extend(pooled_slice_blockers)
    blockers.extend(
        str(value)
        for report in fold_reports
        for value in report["calibration_fit"].get("blockers", [])
    )
    for slice_name in (
        "regional_transfer",
        "patch_transfer",
        "roster_change",
        "tournament_boundary",
        "sparse_support",
    ):
        if any(
            report[slice_name].get("status") != "available"
            for report in fold_reports
        ):
            blockers.append(f"{slice_name}_slice_missing")
    if pooled_calibration_report["status"] != "available":
        blockers.append("calibration_evidence_missing")
    blockers = sorted(set(blockers))
    if not str(cluster_source).startswith("authoritative:"):
        blockers.append("authoritative_series_id_missing_proxy_cluster_used")
        blockers.append("phase_model_series_partition_non_comparable")
    blockers = sorted(set(blockers))
    pooled_candidate_paired_methods: dict[str, Any] = {}
    for method_name, targets in pooled_candidate_paired_targets.items():
        predictions = pooled_candidate_paired_predictions[method_name]
        if not targets:
            continue
        method_target = pd.concat(targets, ignore_index=True)
        method_prediction = pd.concat(predictions, ignore_index=True)
        pooled_candidate_paired_methods[method_name] = {
            "rows": int(len(method_target)),
            "metrics": _classification_metrics(method_target, method_prediction),
            "calibration": _calibration_metrics(method_target, method_prediction),
        }
    pooled_method_specific_current: dict[str, dict[str, Any]] = {}
    for method_name in ("sequential_player_elo", "hierarchical_bt"):
        fold_methods = [
            report.get("method_specific", {}).get(method_name, {})
            for report in current_fold_reports
        ]
        method_rows = int(
            pooled_candidate_paired_methods.get(method_name, {}).get("rows", 0)
        )
        method_requested_rows = int(
            sum(int(report.get("requested_rows", 0)) for report in fold_methods)
        )
        method_scored_ids = sorted(
            {
                str(game_id)
                for report in fold_methods
                for game_id in report.get("scored_game_ids", [])
            }
        )
        method_blockers = sorted(
            {
                str(blocker)
                for report in fold_methods
                for blocker in report.get("blockers", [])
            }
        )
        method_statuses = [str(report.get("status") or "blocked") for report in fold_methods]
        if (
            method_statuses
            and all(status == "available" for status in method_statuses)
            and method_rows == method_requested_rows
            and not method_blockers
        ):
            method_status = "available"
        elif method_rows:
            method_status = "partial"
        else:
            method_status = "blocked"
        method_payload = dict(pooled_candidate_paired_methods.get(method_name, {}))
        method_payload.update(
            {
                "status": method_status,
                "requested_rows": method_requested_rows,
                "scored_rows": method_rows,
                "scored_game_ids": method_scored_ids,
                "missing_game_count": max(0, method_requested_rows - method_rows),
                "fold_statuses": method_statuses,
                "blockers": method_blockers,
                "exclusion_reasons": sorted(
                    {
                        str(report["exclusion_reason"])
                        for report in fold_methods
                        if report.get("exclusion_reason")
                    }
                ),
            }
        )
        pooled_method_specific_current[method_name] = method_payload
    pooled_current_comparison: dict[str, Any]
    if pooled_current_targets:
        pooled_current_target = pd.concat(pooled_current_targets, ignore_index=True)
        pooled_current_methods = {
            method_name: {
                "metrics": _classification_metrics(
                    pooled_current_target,
                    pd.concat(values, ignore_index=True),
                ),
                "calibration": _calibration_metrics(
                    pooled_current_target,
                    pd.concat(values, ignore_index=True),
                ),
                "rows": int(len(pooled_current_target)),
            }
            for method_name, values in pooled_current_predictions.items()
            if values
        }
        pooled_current_comparison = {
            "status": (
                "available"
                if current_fold_reports
                and all(report["status"] == "available" for report in current_fold_reports)
                else "blocked"
            ),
            "requested_folds": int(n_folds),
            "valid_comparison_folds": int(
                sum(report["status"] == "available" for report in current_fold_reports)
            ),
            "rows": int(len(pooled_current_target)),
            "methods": pooled_current_methods,
            "method_specific": pooled_method_specific_current,
            "common_all_method": {
                "status": (
                    "available"
                    if current_fold_reports
                    and all(report["status"] == "available" for report in current_fold_reports)
                    else "blocked"
                ),
                "rows": int(len(pooled_current_target)),
                "game_ids": sorted(
                    {
                        str(game_id)
                        for report in current_fold_reports
                        for game_id in report.get("common_finite_game_ids", [])
                    }
                ),
                "methods": pooled_current_methods,
                "blockers": sorted(
                    {
                        blocker
                        for report in current_fold_reports
                        for blocker in report["blockers"]
                    }
                ),
            },
            "candidate_paired_methods": pooled_candidate_paired_methods,
            "blockers": sorted(
                {
                    blocker
                    for report in current_fold_reports
                    for blocker in report["blockers"]
                }
            ),
        }
    else:
        pooled_current_comparison = {
            "status": "blocked",
            "requested_folds": int(n_folds),
            "valid_comparison_folds": 0,
            "rows": 0,
            "methods": {},
            "method_specific": pooled_method_specific_current,
            "common_all_method": {
                "status": "blocked",
                "rows": 0,
                "game_ids": [],
                "methods": {},
                "blockers": ["current_rating_no_common_finite_rows"],
            },
            "candidate_paired_methods": pooled_candidate_paired_methods,
            "blockers": ["current_rating_no_common_finite_rows"],
        }
    prediction_ledger_rows = sorted(
        prediction_ledger_rows,
        key=lambda row: (int(row["fold"]), str(row["game_id"])),
    )
    prediction_ledger = {
        "schema_version": "scryglass:future-value-prediction-ledger:v2",
        "columns": [
            "fold",
            "game_id",
            "target",
            "candidate",
            "candidate_raw_probability",
            "candidate_raw_logit",
            "candidate_calibrated_logit",
            "calibration_slope",
            "intercept",
            "sequential_player_elo",
            "hierarchical_bt",
            "minimum_metric_support",
            "minimum_effective_support",
            "minimum_atom_support",
            "missing_feature_names",
            "support_status",
        ],
        "row_count": len(prediction_ledger_rows),
        "game_identity_sha256": identity_sha256(
            row["game_id"] for row in prediction_ledger_rows
        ),
        "rows": prediction_ledger_rows,
    }
    prediction_ledger["sha256"] = hashlib.sha256(
        _canonical_json_bytes(prediction_ledger_rows)
    ).hexdigest()
    source_payload = {
        "game_count": int(len(map_frame)),
        "source_game_count": int(source_receipt["source_game_count"]),
        "source_identity_sha256": str(source_receipt["source_identity_sha256"]),
        "accepted_game_ids": list(source_receipt["accepted_game_ids"]),
        "model_eligible_game_count": int(len(verified_eligible_ids)),
        "model_eligible_identity_sha256": identity_sha256(verified_eligible_ids),
        "model_eligible_game_ids": list(verified_eligible_ids),
        "series_cluster_source": cluster_source,
        "series_cluster_audit": cluster_audit,
        "cross_model_series_partition": "non_comparable",
        "half_life_days": float(half_life_days),
        "source_as_of": _utc_text(source_receipt["source_as_of"]),
        "source_files": source_receipt["source_files"],
        "source_receipt_sha256": str(source_receipt["receipt_sha256"]),
        "source_latest": _utc_text(map_frame["date"].max()),
    }
    if source_receipt_path is not None:
        source_payload["source_receipt_path"] = source_receipt_path
        source_payload["source_receipt_file_sha256"] = source_receipt_file_sha256
    result = {
        "schema_version": MODEL_FIT_SCHEMA_VERSION,
        "status": "development_evaluated",
        "variant": variant_config.variant.value if variant_config is not None else None,
        "variant_receipt": (
            variant_config.receipt() if variant_config is not None else None
        ),
        "source": source_payload,
        "evaluation": {
            "requested_folds": int(n_folds),
            "valid_folds": len(fold_reports),
            "minimum_folds": 3,
            "pooled_rows": int(len(pooled_target)),
            "pooled_candidate": _classification_metrics(pooled_target, pooled_prediction),
            "pooled_raw_candidate": _classification_metrics(
                pooled_target, pooled_raw_prediction
            ),
            "pooled_intercept_baseline": _classification_metrics(
                pooled_target, pooled_baseline
            ),
            "pooled_calibration": pooled_calibration_report,
            "pooled_raw_calibration": pooled_raw_calibration_report,
            "pooled_baseline_calibration": pooled_baseline_calibration_report,
            "pooled_current_rating_comparison": pooled_current_comparison,
            "validation_overlap_audit": {
                "status": "passed",
                "validation_game_count": len(
                    {
                        game_id
                        for report in fold_reports
                        for game_id in report["paired_game_ids"]
                    }
                ),
                "fold_validation_game_count_sum": sum(
                    report["validation_game_id_count"] for report in fold_reports
                ),
                "game_id_overlap_count": 0,
                "interval_overlap_count": 0,
                "fold_windows": [
                    {
                        "fold": report["fold"],
                        "validation_start": report["validation_start"],
                        "validation_end": report["validation_end"],
                        "validation_game_identity_sha256": report[
                            "validation_game_identity_sha256"
                        ],
                    }
                    for report in fold_reports
                ],
            },
            "component_reconstruction_audit": {
                "status": "passed",
                "row_count": sum(
                    report["component_evidence"]["row_count"]
                    for report in fold_reports
                ),
                "maximum_absolute_error": max(
                    report["component_evidence"][
                        "maximum_absolute_reconstruction_error"
                    ]
                    for report in fold_reports
                ),
                "fold_component_sha256": [
                    report["component_evidence"]["sha256"]
                    for report in fold_reports
                ],
                "maximum_absolute_calibrated_error": max(
                    report["component_evidence"][
                        "maximum_absolute_calibrated_reconstruction_error"
                    ]
                    for report in fold_reports
                ),
            },
            "strict_prior_calibration": {
                "schema_version": CALIBRATION_SCHEMA_VERSION,
                "status": "available",
                "source_receipt_sha256": str(source_receipt["receipt_sha256"]),
                "method": "strict_prior_outer_validation_zero_intercept_positive_slope",
                "uses_current_validation": False,
                "folds": [report["calibration_fit"] for report in fold_reports],
                "blockers": sorted(
                    {
                        blocker
                        for report in fold_reports
                        for blocker in report["calibration_fit"].get("blockers", [])
                    }
                ),
            },
        },
        "folds": fold_reports,
        "prediction_ledger": prediction_ledger,
        "blockers": blockers,
        "authority": {
            "research_only": True,
            "public_player_rating": False,
            "public_team_rating": False,
            "public_probability": False,
            "odds": False,
            "expected_value": False,
            "recommendation": False,
            "betting": False,
            "promotion": False,
            "deployment": False,
        },
    }
    if runtime_receipt_path is not None:
        result["runtime_receipt_path"] = runtime_receipt_path
    return result


def future_value_model_contract() -> dict[str, Any]:
    """Return the development contract for future player and team value."""

    return {
        "schema_version": MODEL_CONTRACT_VERSION,
        "status": "development_only",
        "rating_variants": [rating_variant_config_receipt(variant) for variant in RATING_VARIANT_ORDER],
        "scaling_curve_producer": {
            "schema_version": SCALING_CURVE_PRODUCER_SCHEMA_VERSION,
            "model_version": SCALING_CURVE_PRODUCER_VERSION,
            "feature_declaration": list(SCALING_CURVE_FEATURE_DECLARATION),
            "shape_features": list(SCALING_CURVE_SHAPE_FEATURES),
            "shape_signed_features": list(SCALING_CURVE_SIGNED_SHAPE_FEATURES),
            "shape_invariant_features": list(SCALING_CURVE_INVARIANT_SHAPE_FEATURES),
        },
        "estimands": {
            "future_player_value": (
                "Development-only marginal map-win logit contribution for one player in "
                "the next eligible map, using strictly prior form, role, champion-role "
                "atoms, and explicit support diagnostics."
            ),
            "future_team_value": (
                "Development-only exact-five roster map-win logit difference before the next "
                "eligible map. It uses roster player-form aggregates, prior team win state, "
                "and roster continuity."
            ),
        },
        "player_components": [
            "strictly_prior_time_decayed_performance_form",
            "fitted_metric_weights",
            "fold_local_rank_3_player_atom",
            "fold_local_rank_3_champion_role_atom",
            "support_and_missingness_diagnostics",
        ],
        "team_components": [
            "exact_roster_player_sum",
            "strictly_prior_team_win_state",
            "roster_continuity",
            "support_and_missingness_diagnostics",
        ],
        "phase_outputs": [
            "expected_gold_curve_10_15_20_25",
            "expected_xp_curve_10_15_20_25",
            "scaling_index",
            "snowball_index",
            "comeback_resilience",
        ],
        "information_boundary": {
            "same_timestamp_maps": "independent batch; updates become available after the full timestamp block",
            "current_checkpoint_targets": "target only; forbidden from the same map pregame vector",
            "current_final_metrics": "update future history only after the map",
            "future_rows": "forbidden from baselines, embeddings, loadings, and hyperparameter selection",
        },
        "fit_contract": {
            "representation": "atom-derived rank-3 champion embedding refit inside each chronological fold",
            "player_state": "strictly prior time-decayed form and fold-local rank-3 atoms summed across each exact five-player side",
            "team_state": "blue-minus-red difference of exact five-player side sums plus strictly prior team win state and roster continuity",
            "metric_weights": "fit inside development folds; no hand-assigned performance weights",
            "side_symmetry": "zero-intercept linear logit over blue-minus-red features after equal fold-local side imputation",
            "missing_values": "fit one fold-local median per side-level feature and apply it equally to blue and red; allow neutral zero only for all-missing centered atom coordinates and fail closed otherwise",
            "regularization": "select L2 strength inside each outer training fold with a nested chronological whole-series log-loss comparison",
            "role_identity": "normalize top, jungle, mid, bot, and support aliases before every atom fit and transform lookup",
            "optimizer": "require finite converged zero-intercept L-BFGS evidence for every nested candidate and final fold fit",
        },
        "evaluation": [
            "non-overlapping chronological intervals with whole clusters and boundary-cluster exclusion",
            "intercept baseline and proper scores",
            "conservative series-superset clusters with collision audit",
            "validation calibration bins and pooled proper scores",
            "regional and patch transfer slices with training-support checks",
            "roster-change and tournament-boundary slices",
            "separate complete and imputed-row proper scores with imputed-only blockers",
            "fold-local equal-side imputation with signed missingness and support indicators",
            "minimum and per-metric support diagnostics with named missing features",
            "structural side-swap probability-complement identity",
            "serialized player, team-context, and data-quality logit reconstruction with per-player support evidence",
        ],
        "future_scope_blockers": [
            "current_player_team_rating_comparison",
            "composition_specific_phase_curve",
            "calibrated_uncertainty",
            "authoritative_series_identity",
            "authoritative_series_cluster_evaluation",
            "phase_model_series_partition_comparability",
        ],
        "authority": {
            "public_player_rating": False,
            "public_team_rating": False,
            "public_probability": False,
            "odds": False,
            "expected_value": False,
            "recommendation": False,
            "betting": False,
            "promotion": False,
            "merge": False,
            "deployment": False,
        },
    }


def bind_accepted_future_value_source(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    *,
    census: Mapping[str, Any],
    source_as_of: Any,
    source_files: Mapping[str, Mapping[str, Any]] | None = None,
) -> AcceptedFutureValueSource:
    """Filter three OE frames to one exact accepted census and bind coverage."""

    accepted_ids = tuple(str(value) for value in census.get("game_ids", ()))
    expected_count = census.get("game_count")
    expected_identity = census.get("source_identity_sha256")
    if (
        not accepted_ids
        or list(canonical_game_ids(accepted_ids)) != list(accepted_ids)
        or expected_count != len(accepted_ids)
        or expected_identity != identity_sha256(accepted_ids)
    ):
        raise FutureValueSourceError("accepted census binding is invalid")
    accepted_set = set(accepted_ids)
    cutoff = _utc_timestamp(source_as_of, "source_as_of")

    scoped: list[pd.DataFrame] = []
    extra_ids: dict[str, list[str]] = {}
    for label, raw in (("maps", maps), ("players", players), ("teams", teams)):
        frame = raw.copy()
        frame["_game_id"] = _frame_game_ids(frame, label)
        available_ids = set(frame["_game_id"].astype(str))
        missing = sorted(accepted_set - available_ids)
        if missing:
            raise FutureValueSourceError(
                f"{label} is missing {len(missing)} accepted game IDs"
            )
        extra_ids[label] = sorted(available_ids - accepted_set)
        frame = frame[frame["_game_id"].isin(accepted_set)].copy()
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
            if frame["date"].isna().any():
                raise FutureValueSourceError(f"{label} contains an invalid accepted date")
            if frame["date"].gt(cutoff).any():
                raise FutureValueSourceError(f"{label} contains rows after source_as_of")
        scoped.append(frame)
    scoped_maps, scoped_players, scoped_teams = scoped
    if len(scoped_maps) != len(accepted_ids) or scoped_maps["_game_id"].duplicated().any():
        raise FutureValueSourceError("accepted map grain is not exactly one row per game")
    outcome = pd.to_numeric(scoped_maps.get("y_blue_win"), errors="coerce")
    if not outcome.isin({0, 1}).all():
        raise FutureValueSourceError("accepted maps contain a missing or invalid result target")

    player_counts = scoped_players.groupby("_game_id", sort=False).size()
    team_counts = scoped_teams.groupby("_game_id", sort=False).size()
    if not player_counts.reindex(accepted_ids, fill_value=0).eq(10).all():
        raise FutureValueSourceError("accepted source does not contain ten player rows per map")
    if not team_counts.reindex(accepted_ids, fill_value=0).eq(2).all():
        raise FutureValueSourceError("accepted source does not contain two team rows per map")

    eligible_ids, exclusion_reasons = _model_eligibility(scoped_players, scoped_teams)
    reason_counts: dict[str, int] = {}
    for reasons in exclusion_reasons.values():
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    stable_player_rows = int(
        scoped_players.get("playerid", pd.Series(index=scoped_players.index, dtype=object)).map(
            lambda value: _stable_identity(value, "oe:player:")
        ).sum()
    )
    stable_team_rows = int(
        scoped_players.get("teamid", pd.Series(index=scoped_players.index, dtype=object)).map(
            lambda value: _stable_identity(value, "oe:team:")
        ).sum()
    )
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "accepted_source_bound_development_only",
        "source_as_of": _utc_text(cutoff),
        "source_game_count": len(accepted_ids),
        "source_identity_sha256": expected_identity,
        "accepted_game_ids": list(accepted_ids),
        "model_eligible_game_count": len(eligible_ids),
        "model_eligible_identity_sha256": identity_sha256(eligible_ids),
        "model_eligible_game_ids": list(eligible_ids),
        "source_rows": {
            "maps": len(scoped_maps),
            "players": len(scoped_players),
            "teams": len(scoped_teams),
        },
        "source_extra_game_ids": extra_ids,
        "identity_coverage": {
            "stable_player_rows": stable_player_rows,
            "stable_player_row_fraction": stable_player_rows / max(len(scoped_players), 1),
            "stable_team_rows": stable_team_rows,
            "stable_team_row_fraction": stable_team_rows / max(len(scoped_players), 1),
        },
        "checkpoint_coverage": {
            "player": _checkpoint_coverage(scoped_players, rows_per_map=10),
            "team": _checkpoint_coverage(scoped_teams, rows_per_map=2),
        },
        "model_exclusions": {
            "game_count": len(exclusion_reasons),
            "reason_counts": dict(sorted(reason_counts.items())),
            "by_game": dict(sorted(exclusion_reasons.items())),
        },
        "source_files": dict(source_files or {}),
        "model_contract": future_value_model_contract(),
        "authority": {
            "research_only": True,
            "public_player_rating": False,
            "public_team_rating": False,
            "public_probability": False,
            "promotion": False,
            "merge": False,
            "deployment": False,
        },
    }
    receipt["receipt_sha256"] = hashlib.sha256(_canonical_json_bytes(receipt)).hexdigest()
    return AcceptedFutureValueSource(
        maps=scoped_maps.drop(columns=["_game_id"]),
        players=scoped_players.drop(columns=["_game_id"]),
        teams=scoped_teams.drop(columns=["_game_id"]),
        eligible_game_ids=eligible_ids,
        receipt=receipt,
    )


def load_accepted_future_value_source(
    *,
    oe_root: Path,
    census_path: Path,
    source_as_of: Any,
) -> AcceptedFutureValueSource:
    """Load an accepted runtime source without modifying worker files."""

    paths = {
        "maps": oe_root / "maps.parquet",
        "players": oe_root / "oe_player_games.parquet",
        "teams": oe_root / "oe_team_games.parquet",
    }
    for path in paths.values():
        if not path.is_file() or path.is_symlink():
            raise FutureValueSourceError(f"source file is missing or unsafe: {path}")
    census = load_census(census_path)
    source_files = {
        label: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256_path(path),
        }
        for label, path in paths.items()
    }
    source_files["accepted_census"] = {
        "path": str(census_path),
        "bytes": census_path.stat().st_size,
        "sha256": _sha256_path(census_path),
    }
    return bind_accepted_future_value_source(
        pd.read_parquet(paths["maps"]),
        pd.read_parquet(paths["players"]),
        pd.read_parquet(paths["teams"]),
        census=census,
        source_as_of=source_as_of,
        source_files=source_files,
    )


def write_source_receipt(path: Path, source: AcceptedFutureValueSource) -> None:
    """Write one canonical, development-only accepted-source receipt."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(source.receipt)
    raw = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(raw, encoding="utf-8")


def team_value_difference(
    blue_player_values: Sequence[float],
    red_player_values: Sequence[float],
    *,
    blue_team_residual: float = 0.0,
    red_team_residual: float = 0.0,
    blue_context_value: float = 0.0,
    red_context_value: float = 0.0,
) -> float:
    """Aggregate exact five-player values with an antisymmetric team contract."""

    if len(blue_player_values) != 5 or len(red_player_values) != 5:
        raise FutureValueSourceError("future team value requires two exact five-player rosters")
    values = [
        *blue_player_values,
        *red_player_values,
        blue_team_residual,
        red_team_residual,
        blue_context_value,
        red_context_value,
    ]
    if any(isinstance(value, bool) or not math.isfinite(float(value)) for value in values):
        raise FutureValueSourceError("future team value contains a non-finite component")
    return float(
        sum(float(value) for value in blue_player_values)
        - sum(float(value) for value in red_player_values)
        + float(blue_team_residual)
        - float(red_team_residual)
        + float(blue_context_value)
        - float(red_context_value)
    )


__all__ = [
    "AcceptedFutureValueSource",
    "FutureValueFoldModel",
    "FutureValueSourceError",
    "LEAGUEPEDIA_CROSSWALK_RECEIPT_SCHEMA_VERSION",
    "LEAGUEPEDIA_CROSSWALK_SOURCE",
    "CURRENT_RATING_RAW_DIFFERENCE_FEATURES",
    "CURRENT_RATING_SIGNED_MAP_FEATURES",
    "CURRENT_RATING_STRENGTH_FEATURES",
    "CURRENT_RATING_FEATURE_SEMANTICS",
    "RATING_FEATURE_LEDGER_SCHEMA_VERSION",
    "RATING_FEATURE_PRODUCER_SCHEMA_VERSION",
    "RATING_FEATURE_PRODUCER_RECEIPT_SCHEMA_VERSION",
    "RATING_FEATURE_PRODUCER_MANIFEST_SCHEMA_VERSION",
    "FUTURE_PLAYER_FORM_SIDE_FEATURES",
    "MODEL_FEATURES",
    "MODEL_FIT_SCHEMA_VERSION",
    "Rank3AtomModel",
    "RATING_VARIANT_CONFIGS",
    "RATING_VARIANT_ORDER",
    "RATING_VARIANT_ORDINALS",
    "RATING_VARIANT_SCHEMA_VERSION",
    "RatingVariant",
    "RatingVariantConfig",
    "SCALING_CURVE_DERIVED_FEATURES",
    "SCALING_CURVE_FEATURE_DECLARATION",
    "SCALING_CURVE_PRODUCER_SCHEMA_VERSION",
    "SCALING_CURVE_PRODUCER_VERSION",
    "SCALING_CURVE_SHAPE_FEATURES",
    "SCALING_CURVE_SIGNED_SHAPE_FEATURES",
    "SCALING_CURVE_INVARIANT_SHAPE_FEATURES",
    "SCALING_CURVE_SIGNED_MAP_FEATURES",
    "TEAM_CONTEXT_FEATURES",
    "assert_pregame_feature_names",
    "assert_rating_feature_names",
    "assert_rating_variant_features",
    "bind_accepted_future_value_source",
    "bind_verified_leaguepedia_series_crosswalk",
    "bind_rating_feature_ledger",
    "build_strict_prior_player_form",
    "build_future_value_design",
    "build_time_decayed_prior_player_form",
    "classify_rating_feature",
    "chronological_whole_series_folds",
    "evaluate_future_value",
    "fit_future_value_model",
    "fit_rank3_player_champion_role_atoms",
    "future_value_model_contract",
    "get_rating_variant_config",
    "is_side_level_feature",
    "is_signed_map_feature",
    "load_accepted_future_value_source",
    "load_verified_leaguepedia_series_crosswalk",
    "rating_variant_config",
    "rating_variant_config_receipt",
    "rating_variant_config_sha256",
    "rating_variant_registry_receipt",
    "rating_variant_registry_sha256",
    "rating_variant_configs",
    "rating_feature_values_sha256",
    "build_rating_feature_producer_manifest",
    "write_rating_feature_producer_receipt",
    "team_value_difference",
    "trusted_feature_producer_receipt",
    "validate_rating_feature_ledger",
    "build_rating_variant_matrix",
    "write_source_receipt",
]
