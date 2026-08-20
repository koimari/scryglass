"""Development-only OE phase curves for future player and team value.

This module models blue-minus-red checkpoint state from information available
before the map.  It has one strict boundary: a checkpoint value describes the
target for the current map.  A final whole-map metric may enter a later map
only through a strict prior history.

The module has no GRID input and it does not emit win probabilities, odds,
expected value, recommendations, or betting data.  A fitted artifact remains
``development_only`` until an independent promotion process approves it.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from lol_kills.etl.source_keys import canonical_source_game_key
from lol_kills.v2.tierlists.accepted_census import canonical_game_ids, identity_sha256


PHASES = (10, 15, 20, 25)
PHASE_KEYS = tuple(str(value) for value in PHASES)
SCHEMA_VERSION = "scryglass:future-phase-curve:v1"
MODEL_VERSION = "future-phase-curve-v1"
SOURCE = "oracle_elixir_only"

# These are final map metrics.  The function accepts aliases that occur in
# OE exports, but checkpoint columns never enter this list.
FINAL_METRIC_ALIASES = (
    "cspm",
    "cspermin",
    "earnedgpm",
    "earned_gpm",
    "earnedgoldshare",
    "dpm",
    "damageshare",
    "damage_share",
    "kills",
    "deaths",
    "assists",
    "visionscore",
    "vision_score",
    "wardsplaced",
    "wardskilled",
    "damagetochampions",
    "damagetowers",
    "damagetotowers",
    "damagetakenperminute",
    "damagemitigatedperminute",
    "monsterkillsownjungle",
    "monsterkillsenemyjungle",
    "earnedgold",
    "totalgold",
)

_ID_COLUMNS = {
    "game_uid",
    "gameid",
    "game_id",
    "date",
    "played_at",
    "game_date",
    "start_time",
    "series_id",
    "seriesid",
    "match_id",
    "matchid",
    "league",
    "region",
    "patch",
    "oe_patch_token",
    "client_patch",
    "public_patch",
    "side",
    "teamcolor",
    "team_color",
    "teamid",
    "team_id",
    "playerid",
    "player_id",
    "position",
    "role",
    "champion",
    "tournament",
}


class FuturePhaseCurveError(ValueError):
    """Raised when phase inputs violate the source or time contract."""


@dataclass(frozen=True)
class BoundPhaseSource:
    """A phase frame bound to one accepted source census."""

    frame: pd.DataFrame
    receipt: Mapping[str, Any]


def _canonical_json_bytes(value: object) -> bytes:
    try:
        import json

        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FuturePhaseCurveError("source receipt is not canonical JSON") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _as_timestamp(value: Any, field: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise FuturePhaseCurveError(f"{field} must be a timezone-aware timestamp")
    return stamp.tz_convert("UTC")


def _game_series(frame: pd.DataFrame, label: str = "phase frame") -> pd.Series:
    column = next(
        (name for name in ("game_uid", "gameid", "game_id") if name in frame.columns),
        None,
    )
    if column is None:
        raise FuturePhaseCurveError(f"{label} has no game identity column")
    fallback = frame["gameid"] if column == "game_uid" and "gameid" in frame.columns else None
    values = [
        canonical_source_game_key(value, fallback.loc[index] if fallback is not None else None)
        for index, value in frame[column].items()
    ]
    result = pd.Series(values, index=frame.index, dtype="string")
    if result.eq("").any() or result.isna().any():
        raise FuturePhaseCurveError(f"{label} contains an empty game identity")
    return result


def _date_series(frame: pd.DataFrame, label: str = "phase frame") -> pd.Series:
    column = next(
        (name for name in ("date", "played_at", "game_date", "start_time") if name in frame.columns),
        None,
    )
    if column is None:
        raise FuturePhaseCurveError(f"{label} has no date column")
    result = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if result.isna().any():
        raise FuturePhaseCurveError(f"{label} contains an invalid date")
    return result


def _normalised_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().casefold())


def _is_checkpoint_name(value: Any) -> bool:
    name = _normalised_name(value)
    return bool(
        re.search(
            r"(?:gold|xp|cs|kills|assists|deaths)(?:at|diffat|differenceat)(?:10|15|20|25)$",
            name,
        )
        or re.search(r"(?:gold|xp|cs|kills|assists|deaths)(?:10|15|20|25)$", name)
    )


def _is_forbidden_pregame_name(value: Any) -> bool:
    name = _normalised_name(value)
    if _is_checkpoint_name(value):
        return True
    forbidden = (
        "result",
        "winner",
        "bluewin",
        "finalresult",
        "gamelength",
        "duration",
        "objectives",
        "firstblood",
        "firstdragon",
        "firsttower",
        "baron",
        "inhibitor",
        "gold30",
        "xp30",
    )
    return any(token in name for token in forbidden)


def assert_pregame_feature_names(feature_names: Iterable[str]) -> None:
    """Reject current checkpoint, final-outcome, and censoring fields."""

    forbidden = sorted({str(name) for name in feature_names if _is_forbidden_pregame_name(name)})
    if forbidden:
        raise FuturePhaseCurveError(
            "pregame phase features contain current-map or final-state fields: "
            + ", ".join(forbidden)
        )


def _side(value: Any) -> str | None:
    name = str(value or "").strip().casefold()
    return {"blue": "blue", "b": "blue", "red": "red", "r": "red"}.get(name)


def _target_value(row: Mapping[str, Any], kind: str, phase: int) -> float | None:
    names = (
        f"{kind}diffat{phase}",
        f"{kind}_diff_{phase}",
        f"{kind}_diffat{phase}",
    )
    for name in names:
        value = row.get(name)
        if value is not None and pd.notna(value):
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                return number
    return None


def _state_value(row: Mapping[str, Any], kind: str, phase: int) -> float | None:
    names = (f"{kind}at{phase}", f"{kind}_at_{phase}", f"{kind}At{phase}")
    for name in names:
        value = row.get(name)
        if value is not None and pd.notna(value):
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                return number
    return None


def _duration_seconds(row: Mapping[str, Any]) -> float | None:
    for name in ("gamelength", "game_length", "duration_seconds", "duration"):
        value = row.get(name)
        if value is None or pd.isna(value):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            # OE gamelength is in seconds. A small duration value is treated
            # as seconds because phase rows use minute thresholds.
            return number
    return None


def prepare_phase_frame(
    maps: pd.DataFrame,
    teams: pd.DataFrame,
    *,
    pregame_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create one blue-oriented phase target row per accepted map.

    Team checkpoint fields become targets only.  Current checkpoint values in
    either input frame never become feature columns.  A short game produces a
    censored missing target when duration proves that the checkpoint was not
    reached.
    """

    maps_value = maps.copy()
    teams_value = teams.copy()
    maps_value["_game_id"] = _game_series(maps_value, "maps")
    teams_value["_game_id"] = _game_series(teams_value, "teams")
    maps_value["_date"] = _date_series(maps_value, "maps")
    teams_value["_date"] = _date_series(teams_value, "teams")
    if maps_value["_game_id"].duplicated().any():
        raise FuturePhaseCurveError("maps must contain one row per game")
    side_column = next(
        (name for name in ("side", "teamcolor", "team_color") if name in teams_value.columns),
        None,
    )
    if side_column is None:
        raise FuturePhaseCurveError("teams has no side column")
    teams_value["_side"] = teams_value[side_column].map(_side)
    if teams_value["_side"].isna().any():
        raise FuturePhaseCurveError("teams contains an unknown side")
    grouped = {key: group for key, group in teams_value.groupby("_game_id", sort=False)}
    feature_map: dict[str, Mapping[str, Any]] = {}
    if pregame_features is not None:
        feature_value = pregame_features.copy()
        feature_value["_game_id"] = _game_series(feature_value, "pregame features")
        if feature_value["_game_id"].duplicated().any():
            raise FuturePhaseCurveError("pregame features must contain one row per game")
        feature_names = [
            str(name)
            for name in feature_value.columns
            if name not in {"_game_id", "game_uid", "gameid", "game_id"}
        ]
        assert_pregame_feature_names(feature_names)
        feature_map = {
            str(row["_game_id"]): row.to_dict()
            for _, row in feature_value.iterrows()
        }

    rows: list[dict[str, Any]] = []
    map_index = maps_value.set_index("_game_id", drop=False)
    for game_id, map_row in map_index.iterrows():
        current_teams = grouped.get(game_id)
        if current_teams is None or len(current_teams) != 2:
            raise FuturePhaseCurveError(f"game {game_id} does not have two team rows")
        sides = set(current_teams["_side"])
        if sides != {"blue", "red"}:
            raise FuturePhaseCurveError(f"game {game_id} has invalid team sides")
        blue = current_teams.loc[current_teams["_side"].eq("blue")].iloc[0].to_dict()
        red = current_teams.loc[current_teams["_side"].eq("red")].iloc[0].to_dict()
        map_dict = map_row.to_dict()
        row: dict[str, Any] = {
            "game_uid": str(game_id),
            "date": map_row["_date"],
            "league": map_dict.get("league", blue.get("league")),
            "region": map_dict.get("region", blue.get("region")),
            "patch": map_dict.get("patch", map_dict.get("oe_patch_token", blue.get("patch"))),
            "series_id": map_dict.get("series_id", map_dict.get("seriesid")),
        }
        for name, value in feature_map.get(str(game_id), {}).items():
            if name not in {"_game_id", "game_uid", "gameid", "game_id"}:
                row[name] = value
        duration = _duration_seconds(map_dict)
        if duration is None:
            duration = _duration_seconds(blue) or _duration_seconds(red)
        for phase in PHASES:
            for kind in ("gold", "xp"):
                target = _target_value(blue, kind, phase)
                if target is None:
                    left = _state_value(blue, kind, phase)
                    right = _state_value(red, kind, phase)
                    if left is not None and right is not None:
                        target = left - right
                censored = duration is not None and duration < float(phase * 60)
                if censored:
                    target = None
                target_name = f"{kind}_diff_{phase}"
                row[target_name] = target
                row[f"{target_name}_missing"] = target is None
                row[f"{target_name}_censored"] = bool(censored)
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        raise FuturePhaseCurveError("phase source has no maps")
    return result


def _infer_final_metrics(frame: pd.DataFrame) -> tuple[str, ...]:
    available = {_normalised_name(name): str(name) for name in frame.columns}
    result: list[str] = []
    for alias in FINAL_METRIC_ALIASES:
        name = available.get(_normalised_name(alias))
        if name and name not in result and not _is_checkpoint_name(name):
            result.append(name)
    return tuple(result)


def strict_prior_final_history(
    frame: pd.DataFrame,
    *,
    entity_column: str,
    date_column: str,
    metric_columns: Sequence[str] | None = None,
    output_prefix: str = "prior_form_",
) -> pd.DataFrame:
    """Return final-map metrics from earlier timestamp blocks only.

    Every row at one entity and timestamp receives the same history.  The
    current row and all same-timestamp rows stay out of the history.  This
    preserves batch independence and blocks current-map checkpoint leakage.
    """

    required = {entity_column, date_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise FuturePhaseCurveError("prior history input is missing: " + ", ".join(missing))
    metrics = tuple(metric_columns or _infer_final_metrics(frame))
    if not metrics:
        raise FuturePhaseCurveError("prior history has no approved final metrics")
    assert_pregame_feature_names(metrics)
    work = frame[[entity_column, date_column, *metrics]].copy()
    work[date_column] = pd.to_datetime(work[date_column], utc=True, errors="coerce")
    if work[entity_column].isna().any() or work[date_column].isna().any():
        raise FuturePhaseCurveError("prior history identity or date is missing")
    key_frame = work[[entity_column, date_column]].copy()
    blocks = key_frame.drop_duplicates().sort_values([entity_column, date_column], kind="stable")
    output = key_frame.copy()
    for metric in metrics:
        values = pd.to_numeric(work[metric], errors="coerce")
        values = values.where(np.isfinite(values))
        metric_frame = key_frame.copy()
        metric_frame["_value"] = values.to_numpy(dtype=float)
        aggregate = (
            metric_frame.groupby([entity_column, date_column], sort=False, observed=True)["_value"]
            .agg(["sum", "count"])
            .reset_index()
            .sort_values([entity_column, date_column], kind="stable")
        )
        aggregate["_prior_sum"] = (
            aggregate.groupby(entity_column, sort=False, observed=True)["sum"].cumsum()
            - aggregate["sum"]
        )
        aggregate["_prior_count"] = (
            aggregate.groupby(entity_column, sort=False, observed=True)["count"].cumsum()
            - aggregate["count"]
        )
        aggregate[f"{output_prefix}{metric}"] = aggregate["_prior_sum"] / aggregate[
            "_prior_count"
        ].where(aggregate["_prior_count"] > 0)
        aggregate[f"{output_prefix}{metric}_support"] = aggregate["_prior_count"].astype(int)
        output = output.merge(
            aggregate[
                [
                    entity_column,
                    date_column,
                    f"{output_prefix}{metric}",
                    f"{output_prefix}{metric}_support",
                ]
            ],
            on=[entity_column, date_column],
            how="left",
            validate="many_to_one",
        )
    if len(output) != len(frame):
        raise FuturePhaseCurveError("prior history changed row grain")
    return output


def build_strict_prior_team_features(
    teams: pd.DataFrame,
    *,
    metric_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build side-neutral team final-form differences for each map."""

    work = teams.copy()
    work["_game_id"] = _game_series(work, "teams")
    work["_date"] = _date_series(work, "teams")
    identity = next((name for name in ("teamid", "team_id") if name in work.columns), None)
    if identity is None:
        raise FuturePhaseCurveError("team final history has no stable team identity")
    side_column = next((name for name in ("side", "teamcolor", "team_color") if name in work.columns), None)
    if side_column is None:
        raise FuturePhaseCurveError("team final history has no side column")
    work["_side"] = work[side_column].map(_side)
    if work["_side"].isna().any():
        raise FuturePhaseCurveError("team final history has an unknown side")
    history = strict_prior_final_history(
        work,
        entity_column=identity,
        date_column="_date",
        metric_columns=metric_columns,
    )
    history_columns = [
        name
        for name in history.columns
        if name.startswith("prior_form_")
    ]
    history = pd.concat(
        [work[["_game_id", "_date", "_side"]].reset_index(drop=True), history[history_columns].reset_index(drop=True)],
        axis=1,
    )
    rows: list[dict[str, Any]] = []
    for game_id, group in history.groupby("_game_id", sort=False):
        if set(group["_side"]) != {"blue", "red"}:
            raise FuturePhaseCurveError(f"team history game {game_id} has invalid sides")
        blue = group.loc[group["_side"].eq("blue")].iloc[0]
        red = group.loc[group["_side"].eq("red")].iloc[0]
        row: dict[str, Any] = {"game_uid": str(game_id), "date": blue["_date"]}
        for name in history_columns:
            if name.endswith("_support"):
                row[f"{name}_min"] = int(min(int(blue[name]), int(red[name])))
                continue
            left = blue[name]
            right = red[name]
            row[f"{name}_diff"] = (
                float(left) - float(right)
                if pd.notna(left) and pd.notna(right)
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def bind_phase_source(
    frame: pd.DataFrame,
    source_receipt: Mapping[str, Any],
    *,
    allow_subset: bool = False,
) -> BoundPhaseSource:
    """Bind one phase frame to the accepted census and source cutoff."""

    required = ("source_as_of", "source_game_count", "source_identity_sha256", "accepted_game_ids")
    missing = [name for name in required if name not in source_receipt]
    if missing:
        raise FuturePhaseCurveError("source receipt is missing: " + ", ".join(missing))
    accepted = tuple(str(value) for value in source_receipt["accepted_game_ids"])
    canonical = canonical_game_ids(accepted)
    if tuple(accepted) != canonical:
        raise FuturePhaseCurveError("source receipt accepted game IDs are not canonical")
    if int(source_receipt["source_game_count"]) != len(accepted):
        raise FuturePhaseCurveError("source receipt game count does not match accepted IDs")
    expected_hash = identity_sha256(accepted)
    if str(source_receipt["source_identity_sha256"]).lower() != expected_hash:
        raise FuturePhaseCurveError("source receipt identity hash does not match accepted IDs")
    cutoff = _as_timestamp(source_receipt["source_as_of"], "source_as_of")
    value = frame.copy()
    value["_game_id"] = _game_series(value, "phase frame")
    value["_date"] = _date_series(value, "phase frame")
    accepted_set = set(accepted)
    available = set(value["_game_id"])
    missing_ids = sorted(accepted_set - available)
    if missing_ids and not allow_subset:
        raise FuturePhaseCurveError(f"phase frame is missing {len(missing_ids)} accepted games")
    if value["_game_id"].duplicated().any():
        raise FuturePhaseCurveError("phase frame must contain one row per accepted game")
    selected = value.loc[value["_game_id"].isin(accepted_set)].copy()
    if not allow_subset and len(selected) != len(accepted):
        raise FuturePhaseCurveError("phase frame grain does not match accepted census")
    if selected["_date"].gt(cutoff).any():
        raise FuturePhaseCurveError("phase frame contains rows after source_as_of")
    selected = selected.drop(columns=["_game_id", "_date"])
    receipt = dict(source_receipt)
    receipt["source_identity_sha256"] = expected_hash
    receipt["source_game_count"] = len(accepted)
    return BoundPhaseSource(frame=selected, receipt=receipt)


def _default_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    excluded = {
        "game_uid",
        "gameid",
        "game_id",
        "date",
        "played_at",
        "game_date",
        "start_time",
        "league",
        "region",
        "patch",
        "oe_patch_token",
        "public_patch",
        "series_id",
    }
    candidates: list[str] = []
    for name in frame.columns:
        text = str(name)
        if text in excluded or text.endswith("_missing") or text.endswith("_censored"):
            continue
        if text.startswith(("prior_", "form_", "rating_", "atom_", "continuity_", "roster_")):
            candidates.append(text)
    assert_pregame_feature_names(candidates)
    return tuple(candidates)


def _design(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    names: list[str] = []
    columns: list[np.ndarray] = []
    for name in feature_columns:
        if name not in frame.columns:
            raise FuturePhaseCurveError(f"phase feature is missing: {name}")
        assert_pregame_feature_names([name])
        values = pd.to_numeric(frame[name], errors="coerce")
        missing = (~np.isfinite(values.to_numpy(dtype=float))).astype(float)
        numeric = values.fillna(0.0).to_numpy(dtype=float)
        names.append(str(name))
        columns.append(numeric)
        names.append(f"{name}__missing")
        columns.append(missing)
    if not columns:
        raise FuturePhaseCurveError("phase model has no pregame features")
    return np.column_stack(columns), tuple(names)


def _target(frame: pd.DataFrame, kind: str, phase: int) -> np.ndarray:
    name = f"{kind}_diff_{phase}"
    if name not in frame.columns:
        return np.full(len(frame), np.nan, dtype=float)
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
    return np.where(np.isfinite(values), values, np.nan)


def _fit_one(matrix: np.ndarray, target: np.ndarray, alpha: float) -> dict[str, Any] | None:
    valid = np.isfinite(target)
    if int(valid.sum()) < 2:
        return None
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(matrix[valid], target[valid])
    residuals = target[valid] - model.predict(matrix[valid])
    sigma = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
    return {
        "intercept": float(model.intercept_),
        "coefficients": [float(value) for value in model.coef_],
        "train_rows": int(valid.sum()),
        "residual_sd": sigma if math.isfinite(sigma) else None,
        "rmse": float(np.sqrt(np.mean(residuals * residuals))),
        "mae": float(np.mean(np.abs(residuals))),
    }


def _observed_comeback(frame: pd.DataFrame) -> dict[str, Any]:
    early = _target(frame, "gold", 10)
    late = _target(frame, "gold", 25)
    valid = np.isfinite(early) & np.isfinite(late) & (early < 0.0)
    if not valid.any():
        return {"value": None, "support": 0, "definition": "share of early-behind maps with a smaller late deficit"}
    recovered = (late[valid] > early[valid]).astype(float)
    return {
        "value": float(np.mean(recovered)),
        "support": int(valid.sum()),
        "definition": "share of early-behind maps with a smaller late deficit",
    }


def fit_phase_curve(
    frame: pd.DataFrame,
    *,
    source_receipt: Mapping[str, Any],
    feature_columns: Sequence[str] | None = None,
    alpha: float = 10.0,
    model_version: str = MODEL_VERSION,
) -> dict[str, Any]:
    """Fit OE-only gold and XP curves from strictly pregame features.

    The returned artifact stays development-only.  It has no probability or
    public-authority field.  Evaluation folds must call this function on the
    training rows only.
    """

    bound = bind_phase_source(frame, source_receipt, allow_subset=True)
    value = bound.frame.copy()
    selected_features = tuple(feature_columns or _default_feature_columns(value))
    assert_pregame_feature_names(selected_features)
    matrix, design_names = _design(value, selected_features)
    models: dict[str, dict[str, dict[str, Any] | None]] = {"gold": {}, "xp": {}}
    coverage: dict[str, dict[str, Any]] = {"gold": {}, "xp": {}}
    for kind in ("gold", "xp"):
        for phase in PHASES:
            target = _target(value, kind, phase)
            valid = np.isfinite(target)
            censored_name = f"{kind}_diff_{phase}_censored"
            censored_count = int(value[censored_name].fillna(False).astype(bool).sum()) if censored_name in value else 0
            coverage[kind][str(phase)] = {
                "rows": int(valid.sum()),
                "coverage": float(valid.mean()) if len(valid) else 0.0,
                "missing_rows": int((~valid).sum()),
                "censored_rows": censored_count,
            }
            models[kind][str(phase)] = _fit_one(matrix, target, alpha)
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": model_version,
        "authority": "development_only",
        "source": SOURCE,
        "source_as_of": bound.receipt["source_as_of"],
        "source_game_count": int(bound.receipt["source_game_count"]),
        "source_identity_sha256": str(bound.receipt["source_identity_sha256"]),
        "accepted_game_ids": list(bound.receipt["accepted_game_ids"]),
        "source_receipt_sha256": _sha256_bytes(_canonical_json_bytes(bound.receipt)),
        "feature_columns": list(selected_features),
        "design_columns": list(design_names),
        "models": models,
        "coverage": coverage,
        "support": {
            kind: {
                phase: (models[kind][phase] or {}).get("train_rows", 0)
                for phase in PHASE_KEYS
            }
            for kind in ("gold", "xp")
        },
        "uncertainty": {
            kind: {
                phase: (models[kind][phase] or {}).get("residual_sd")
                for phase in PHASE_KEYS
            }
            for kind in ("gold", "xp")
        },
        "curve_definitions": {
            "scaling_index": "XP slope acceleration: (xp25-xp15) - (xp15-xp10)",
            "snowball_index": "gold slope acceleration: (gold15-gold10) - (gold25-gold20)",
            "comeback_resilience": "descriptive conditional recovery share, not a win probability",
        },
        "comeback_resilience": _observed_comeback(value),
        "leakage_contract": {
            "source": "OE only",
            "current_checkpoint_targets": "target_only",
            "final_metrics": "strict_prior_timestamp_blocks_only",
            "same_timestamp_batch": "excluded from prior history",
            "censoring": "checkpoint not reached is missing target",
            "grid_dependency": False,
        },
        "authority_gates": {
            "public_player_rating": False,
            "public_team_rating": False,
            "public_probability": False,
            "odds": False,
            "expected_value": False,
            "recommendation": False,
            "betting": False,
            "promotion": False,
        },
    }


def _predict_one(model: Mapping[str, Any] | None, vector: np.ndarray) -> tuple[float | None, float | None]:
    if not isinstance(model, Mapping):
        return None, None
    coefficients = np.asarray(model.get("coefficients") or [], dtype=float)
    if len(coefficients) != len(vector):
        return None, None
    value = float(model.get("intercept") or 0.0) + float(coefficients @ vector)
    residual_sd = model.get("residual_sd")
    try:
        uncertainty = float(residual_sd) if residual_sd is not None else None
    except (TypeError, ValueError):
        uncertainty = None
    return value if math.isfinite(value) else None, uncertainty


def score_phase_curve(
    artifact: Mapping[str, Any],
    features: Mapping[str, Any],
) -> dict[str, Any]:
    """Score an artifact for research inspection.

    The result remains marked ``development_only``.  It contains no win
    probability and cannot authorize public model output.
    """

    if artifact.get("authority") != "development_only":
        raise FuturePhaseCurveError("phase artifact authority is not development_only")
    feature_columns = tuple(str(name) for name in artifact.get("feature_columns") or ())
    assert_pregame_feature_names(feature_columns)
    vector_values: list[float] = []
    missing_features: list[str] = []
    for name in feature_columns:
        raw = features.get(name)
        try:
            number = float(raw)
        except (TypeError, ValueError):
            number = math.nan
        if not math.isfinite(number):
            missing_features.append(name)
            number = 0.0
        vector_values.extend((number, 1.0 if name in missing_features else 0.0))
    vector = np.asarray(vector_values, dtype=float)
    expected_gold: dict[str, float | None] = {}
    expected_xp: dict[str, float | None] = {}
    uncertainty_gold: dict[str, float | None] = {}
    uncertainty_xp: dict[str, float | None] = {}
    for phase in PHASES:
        value, sigma = _predict_one((artifact.get("models") or {}).get("gold", {}).get(str(phase)), vector)
        expected_gold[str(phase)] = round(value, 4) if value is not None else None
        uncertainty_gold[str(phase)] = round(sigma, 4) if sigma is not None else None
        value, sigma = _predict_one((artifact.get("models") or {}).get("xp", {}).get(str(phase)), vector)
        expected_xp[str(phase)] = round(value, 4) if value is not None else None
        uncertainty_xp[str(phase)] = round(sigma, 4) if sigma is not None else None
    gold_values = [expected_gold[str(phase)] for phase in PHASES]
    xp_values = [expected_xp[str(phase)] for phase in PHASES]
    scaling = (
        (xp_values[3] - xp_values[2]) - (xp_values[1] - xp_values[0])
        if all(value is not None for value in xp_values)
        else None
    )
    snowball = (
        (gold_values[1] - gold_values[0]) - (gold_values[3] - gold_values[2])
        if all(value is not None for value in gold_values)
        else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": artifact.get("model_version"),
        "authority": "development_only",
        "source": artifact.get("source", SOURCE),
        "expected_gold_curve": expected_gold,
        "expected_xp_curve": expected_xp,
        "uncertainty_gold": uncertainty_gold,
        "uncertainty_xp": uncertainty_xp,
        "support": artifact.get("support", {}),
        "coverage": artifact.get("coverage", {}),
        "scaling_index": round(float(scaling), 4) if scaling is not None else None,
        "snowball_index": round(float(snowball), 4) if snowball is not None else None,
        "comeback_resilience": artifact.get("comeback_resilience"),
        "missing_features": missing_features,
        "authority_gates": artifact.get("authority_gates", {}),
    }


def chronological_folds(
    frame: pd.DataFrame,
    *,
    n_splits: int = 3,
    min_train_rows: int = 1,
    cluster_column: str | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return chronological, timestamp-blocked train/test indices.

    With ``cluster_column`` set, a cluster appears in one split only.  This
    supports series-cluster-safe evaluation when the source has series IDs.
    """

    if n_splits < 1:
        raise FuturePhaseCurveError("n_splits must be positive")
    dates = _date_series(frame)
    groups = dates.dt.floor("s")
    unique_dates = pd.Series(groups.unique()).sort_values().tolist()
    if len(unique_dates) < 2:
        return ()
    boundaries = np.array_split(np.asarray(unique_dates, dtype=object), n_splits)
    output: list[tuple[np.ndarray, np.ndarray]] = []
    cluster_values = frame[cluster_column].astype("string") if cluster_column and cluster_column in frame else None
    for block in boundaries:
        if len(block) == 0:
            continue
        test_mask = groups.isin(block)
        train_mask = groups < min(block)
        if cluster_values is not None:
            test_clusters = set(cluster_values.loc[test_mask].dropna().tolist())
            train_mask &= ~cluster_values.isin(test_clusters)
        train = np.flatnonzero(train_mask.to_numpy())
        test = np.flatnonzero(test_mask.to_numpy())
        if len(train) < min_train_rows or len(test) == 0:
            continue
        output.append((train, test))
    return tuple(output)


def evaluate_phase_curve(
    frame: pd.DataFrame,
    *,
    source_receipt: Mapping[str, Any],
    feature_columns: Sequence[str],
    n_splits: int = 3,
    cluster_column: str | None = None,
    alpha: float = 10.0,
) -> dict[str, Any]:
    """Evaluate each phase on future rows with fold-internal fitting."""

    folds = chronological_folds(
        frame,
        n_splits=n_splits,
        min_train_rows=max(1, len(feature_columns) + 1),
        cluster_column=cluster_column,
    )
    errors: dict[str, dict[str, list[float]]] = {
        kind: {str(phase): [] for phase in PHASES} for kind in ("gold", "xp")
    }
    fold_rows: list[dict[str, Any]] = []
    for fold_number, (train_indices, test_indices) in enumerate(folds):
        train = frame.iloc[train_indices].copy()
        test = frame.iloc[test_indices].copy()
        artifact = fit_phase_curve(
            train,
            source_receipt=source_receipt,
            feature_columns=feature_columns,
            alpha=alpha,
        )
        matrix, _ = _design(test, feature_columns)
        row: dict[str, Any] = {"fold": fold_number, "train_rows": len(train), "test_rows": len(test)}
        for kind in ("gold", "xp"):
            for phase in PHASES:
                model = (artifact.get("models") or {}).get(kind, {}).get(str(phase))
                target = _target(test, kind, phase)
                if isinstance(model, Mapping):
                    coefficients = np.asarray(model.get("coefficients") or [], dtype=float)
                    prediction = float(model.get("intercept") or 0.0) + matrix @ coefficients
                else:
                    prediction = np.full(len(test), np.nan)
                valid = np.isfinite(target) & np.isfinite(prediction)
                if valid.any():
                    residual = target[valid] - prediction[valid]
                    errors[kind][str(phase)].extend(float(value) for value in residual)
                row[f"{kind}_{phase}_rows"] = int(valid.sum())
        fold_rows.append(row)
    metrics: dict[str, dict[str, Any]] = {kind: {} for kind in ("gold", "xp")}
    for kind in ("gold", "xp"):
        for phase in PHASES:
            residual = np.asarray(errors[kind][str(phase)], dtype=float)
            metrics[kind][str(phase)] = {
                "rows": int(len(residual)),
                "rmse": float(np.sqrt(np.mean(residual * residual))) if len(residual) else None,
                "mae": float(np.mean(np.abs(residual))) if len(residual) else None,
            }
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "chronological_fold_internal_ridge",
        "folds": fold_rows,
        "metrics": metrics,
        "fold_count": len(fold_rows),
        "cluster_safe": bool(cluster_column),
        "authority": "development_only",
    }


def side_swap_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the same phase rows from the opposing side's perspective."""

    result = frame.copy()
    for name in result.columns:
        text = str(name).casefold()
        if ("_diff_" in text or text.endswith("_diff")) and not (
            text.endswith("_missing") or text.endswith("_censored")
        ):
            values = pd.to_numeric(result[name], errors="coerce")
            result[name] = -values
    if "y_blue_win" in result.columns:
        labels = pd.to_numeric(result["y_blue_win"], errors="coerce")
        result["y_blue_win"] = labels.where(labels.isna(), 1.0 - labels)
    return result


__all__ = [
    "BoundPhaseSource",
    "FINAL_METRIC_ALIASES",
    "FuturePhaseCurveError",
    "MODEL_VERSION",
    "PHASES",
    "SCHEMA_VERSION",
    "assert_pregame_feature_names",
    "bind_phase_source",
    "build_strict_prior_team_features",
    "chronological_folds",
    "evaluate_phase_curve",
    "fit_phase_curve",
    "prepare_phase_frame",
    "score_phase_curve",
    "side_swap_frame",
    "strict_prior_final_history",
]
