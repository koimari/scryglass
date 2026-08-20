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
from lol_kills.research.atomized_rf_composite import (
    CHECKPOINTS as ATOM_CHECKPOINTS,
    GROUP_COLUMNS as ATOM_GROUP_COLUMNS,
)
from lol_kills.v2.tierlists.accepted_census import canonical_game_ids, identity_sha256


PHASES = tuple(int(value) for value in ATOM_CHECKPOINTS)
PHASE_KEYS = tuple(str(value) for value in PHASES)
SCHEMA_VERSION = "scryglass:future-phase-curve:v1"
MODEL_VERSION = "future-phase-curve-v1"
SOURCE = "oracle_elixir_only"
PHASE_FEATURE_FAMILY = "checkpoint_forecasts"
PHASE_FEATURE_DECLARATION = tuple(ATOM_GROUP_COLUMNS[PHASE_FEATURE_FAMILY])

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


def _series_cluster_labels(metadata: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Build outcome-free series clusters and record their provenance.

    Numeric annual IDs provide a stable series proxy.  Other rows use the
    UTC date, competition label, tournament, and unordered stable team keys.
    A row with incomplete identity keeps a game-level fallback and remains a
    promotion blocker.
    """

    labels: list[str] = []
    sources: list[str] = []
    numeric_pattern = re.compile(r"^(\d+-\d+_game)(?:_\d+)?$")

    def token(value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        text = str(value).strip()
        return text or None

    for _, row in metadata.iterrows():
        game_uid = token(row.get("game_uid"))
        if game_uid is None:
            raise FuturePhaseCurveError("series cluster input has an empty game identity")
        existing = token(row.get("series_id"))
        numeric_match = numeric_pattern.fullmatch(game_uid)
        if existing is not None:
            labels.append(existing)
            sources.append("exact_id_proxy")
            continue
        if numeric_match is not None:
            labels.append(numeric_match.group(1))
            sources.append("exact_id_proxy")
            continue
        date_value = pd.to_datetime(row.get("date"), utc=True, errors="coerce")
        date_token = date_value.strftime("%Y-%m-%d") if pd.notna(date_value) else None
        league = token(row.get("league_source")) or token(row.get("league"))
        tournament = token(row.get("tournament")) or "<missing>"
        blue_team = token(row.get("blue_team_key"))
        red_team = token(row.get("red_team_key"))
        team_keys = sorted(value for value in (blue_team, red_team) if value is not None)
        if date_token is not None and league is not None and len(team_keys) == 2:
            labels.append(
                "team-date:"
                + "|".join((date_token, league, tournament, team_keys[0], team_keys[1]))
            )
            sources.append("team_date_proxy")
            continue
        labels.append("game-fallback:" + game_uid)
        sources.append("game_fallback")
    return (
        pd.Series(labels, index=metadata.index, dtype="string"),
        pd.Series(sources, index=metadata.index, dtype="string"),
    )


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
    # A historical value is safe only when its name declares the lagged
    # boundary.  Bare final-map fields can otherwise enter a pregame vector.
    history_prefix = (
        "prior_",
        "history_",
        "rolling_",
        "lag_",
        "form_",
        "rating_",
        "atom_",
        "continuity_",
        "roster_",
    )
    if name in {_normalised_name(alias) for alias in FINAL_METRIC_ALIASES} and not name.startswith(
        history_prefix
    ):
        return True
    forbidden = (
        "target",
        "observed",
        "current",
        "result",
        "winner",
        "bluewin",
        "finalresult",
        "gamelength",
        "duration",
        "gameclock",
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
    counts = teams_value.groupby("_game_id", sort=False, observed=True)["_side"].agg(
        rows="size", sides="nunique"
    )
    invalid = counts[(counts["rows"] != 2) | (counts["sides"] != 2)]
    if not invalid.empty:
        raise FuturePhaseCurveError(
            f"game {str(invalid.index[0])} does not have two team rows"
        )
    blue = teams_value.loc[teams_value["_side"].eq("blue")].set_index("_game_id", drop=False)
    red = teams_value.loc[teams_value["_side"].eq("red")].set_index("_game_id", drop=False)
    map_index = maps_value.set_index("_game_id", drop=False)
    if not set(map_index.index).issubset(blue.index) or not set(map_index.index).issubset(red.index):
        missing = sorted(set(map_index.index) - set(blue.index) - set(red.index))
        raise FuturePhaseCurveError(f"phase source is missing team rows: {missing[:3]}")

    def numeric_coalesce(source: pd.DataFrame, names: Sequence[str]) -> pd.Series:
        result = pd.Series(np.nan, index=source.index, dtype=float)
        for name in names:
            if name in source.columns:
                result = result.fillna(pd.to_numeric(source[name], errors="coerce"))
        return result

    result = pd.DataFrame(index=map_index.index)
    result["game_uid"] = map_index["_game_id"].astype(str)
    result["date"] = map_index["_date"]
    for output, names in (
        ("league", ("league",)),
        ("region", ("region", "league_source")),
        ("patch", ("patch", "oe_patch_token")),
        ("series_id", ("series_id", "seriesid")),
    ):
        source = next((name for name in names if name in map_index.columns), None)
        if source is not None:
            result[output] = map_index[source]
        else:
            fallback = next((name for name in names if name in blue.columns), None)
            result[output] = blue.reindex(map_index.index)[fallback] if fallback else pd.NA
    cluster_metadata = pd.DataFrame(index=map_index.index)
    cluster_metadata["game_uid"] = result["game_uid"]
    cluster_metadata["date"] = result["date"]
    cluster_metadata["series_id"] = result["series_id"]
    for name in ("league", "league_source", "tournament", "blue_team_key", "red_team_key"):
        if name in map_index.columns:
            cluster_metadata[name] = map_index[name]
        else:
            cluster_metadata[name] = pd.NA
    series_labels, series_sources = _series_cluster_labels(cluster_metadata)
    result["series_id"] = series_labels
    result["series_id_source"] = series_sources

    duration = numeric_coalesce(map_index, ("gamelength", "game_length", "duration_seconds", "duration"))
    duration = duration.combine_first(
        numeric_coalesce(blue.reindex(map_index.index), ("gamelength", "game_length", "duration_seconds", "duration"))
    )
    duration = duration.combine_first(
        numeric_coalesce(red.reindex(map_index.index), ("gamelength", "game_length", "duration_seconds", "duration"))
    )
    result["duration_seconds"] = duration

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
        feature_value = feature_value.set_index("_game_id").drop(
            columns=[name for name in ("game_uid", "gameid", "game_id") if name in feature_value],
            errors="ignore",
        )
        result = result.join(feature_value, how="left")

    for phase in PHASES:
        for kind in ("gold", "xp"):
            direct = numeric_coalesce(
                blue.reindex(map_index.index),
                (f"{kind}diffat{phase}", f"{kind}_diff_{phase}", f"{kind}_diffat{phase}"),
            )
            left = numeric_coalesce(
                blue.reindex(map_index.index),
                (f"{kind}at{phase}", f"{kind}_at_{phase}", f"{kind}At{phase}"),
            )
            right = numeric_coalesce(
                red.reindex(map_index.index),
                (f"{kind}at{phase}", f"{kind}_at_{phase}", f"{kind}At{phase}"),
            )
            target = direct.combine_first(left - right)
            censored = duration.notna() & duration.lt(float(phase * 60))
            target = target.mask(censored)
            target_name = f"{kind}_diff_{phase}"
            result[target_name] = target
            result[f"{target_name}_missing"] = target.isna()
            result[f"{target_name}_censored"] = censored.astype(bool)
    if result.empty:
        raise FuturePhaseCurveError("phase source has no maps")
    return result.reset_index(drop=True)


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
    forbidden_history = sorted(
        name
        for name in metrics
        if _is_checkpoint_name(name)
        or (
            _is_forbidden_pregame_name(name)
            and _normalised_name(name) in {"result", "winner", "ybluewin"}
        )
    )
    if forbidden_history:
        raise FuturePhaseCurveError(
            "prior history contains current-map fields: " + ", ".join(forbidden_history)
        )
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
        # OE metrics have different native units.  Fixed source units keep
        # Ridge conditioning stable and make train/test scoring reproducible.
        normalized_name = _normalised_name(name)
        if "gold" in normalized_name or "xp" in normalized_name:
            numeric = numeric / 10000.0
        elif "dpm" in normalized_name or "damage" in normalized_name:
            numeric = numeric / 1000.0
        elif "vision" in normalized_name or "ward" in normalized_name:
            numeric = numeric / 100.0
        elif (
            "cs" in normalized_name
            or "kill" in normalized_name
            or "assist" in normalized_name
        ):
            numeric = numeric / 100.0
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


def _target_coverage(frame: pd.DataFrame, kind: str, phase: int) -> dict[str, Any]:
    """Describe target support while keeping short maps out of the denominator.

    A target is at risk when its duration reaches the checkpoint.  Unknown
    duration stays in the source denominator because an observed checkpoint is
    still usable.  A missing value at an at-risk row is ordinary source
    missingness, not censoring.
    """

    target = _target(frame, kind, phase)
    target_name = f"{kind}_diff_{phase}"
    observed = np.isfinite(target)
    censored = (
        frame[f"{target_name}_censored"].fillna(False).astype(bool).to_numpy()
        if f"{target_name}_censored" in frame
        else np.zeros(len(frame), dtype=bool)
    )
    missing = ~observed
    duration_known = None
    if "duration_seconds" in frame:
        duration_known = pd.to_numeric(frame["duration_seconds"], errors="coerce").to_numpy(
            dtype=float
        )
    elif "gamelength" in frame:
        duration_known = pd.to_numeric(frame["gamelength"], errors="coerce").to_numpy(dtype=float)
    if duration_known is None:
        at_risk = ~censored
    else:
        at_risk = (~np.isfinite(duration_known)) | (duration_known >= float(phase * 60))
    return {
        "rows": int(len(frame)),
        "at_risk_rows": int(at_risk.sum()),
        "observed_rows": int(observed.sum()),
        "coverage": float(observed.mean()) if len(frame) else 0.0,
        "at_risk_coverage": float(observed[at_risk].mean()) if at_risk.any() else None,
        "missing_rows": int(missing.sum()),
        "censored_rows": int(censored.sum()),
        "uncensored_missing_rows": int((missing & ~censored).sum()),
    }


def phase_curve_measures(
    gold_values: Sequence[float | None],
    xp_values: Sequence[float | None],
) -> dict[str, float | None]:
    """Derive signed curve measures from four checkpoint predictions."""

    if len(gold_values) != len(PHASES) or len(xp_values) != len(PHASES):
        raise FuturePhaseCurveError("phase measures need all four checkpoints")
    gold = [float(value) if value is not None and math.isfinite(float(value)) else None for value in gold_values]
    xp = [float(value) if value is not None and math.isfinite(float(value)) else None for value in xp_values]
    scaling = (
        (xp[3] - xp[2]) - (xp[1] - xp[0])
        if all(value is not None for value in xp)
        else None
    )
    snowball = (
        (gold[1] - gold[0]) - (gold[3] - gold[2])
        if all(value is not None for value in gold)
        else None
    )
    return {
        "scaling_index": float(scaling) if scaling is not None else None,
        "snowball_index": float(snowball) if snowball is not None else None,
    }


def _fit_one(matrix: np.ndarray, target: np.ndarray, alpha: float) -> dict[str, Any] | None:
    valid = np.isfinite(target)
    if int(valid.sum()) < 2:
        return None
    # Phase targets are blue-minus-red quantities.  A zero intercept keeps the
    # fitted curve antisymmetric under a blue/red relabeling.
    model = Ridge(alpha=float(alpha), fit_intercept=False, solver="lsqr")
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
    by_window: dict[str, dict[str, Any]] = {}
    for start, end in ((10, 15), (10, 20), (10, 25), (15, 20), (15, 25)):
        early = _target(frame, "gold", start)
        late = _target(frame, "gold", end)
        valid = np.isfinite(early) & np.isfinite(late) & (early < 0.0)
        key = f"{start}_to_{end}"
        if not valid.any():
            by_window[key] = {"value": None, "support": 0}
            continue
        recovered = (late[valid] > early[valid]).astype(float)
        by_window[key] = {
            "value": float(np.mean(recovered)),
            "support": int(valid.sum()),
        }
    primary = by_window["10_to_25"]
    return {
        "value": primary["value"],
        "support": primary["support"],
        "by_window": by_window,
        "definition": "share of early-behind maps with a smaller late gold deficit",
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
            coverage[kind][str(phase)] = _target_coverage(value, kind, phase)
            models[kind][str(phase)] = _fit_one(matrix, target, alpha)
    observed_gold = [
        float(value[f"gold_diff_{phase}"].mean())
        if f"gold_diff_{phase}" in value and value[f"gold_diff_{phase}"].notna().any()
        else None
        for phase in PHASES
    ]
    observed_xp = [
        float(value[f"xp_diff_{phase}"].mean())
        if f"xp_diff_{phase}" in value and value[f"xp_diff_{phase}"].notna().any()
        else None
        for phase in PHASES
    ]
    observed_measures = phase_curve_measures(observed_gold, observed_xp)
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
        "feature_family": PHASE_FEATURE_FAMILY,
        "feature_declaration": list(PHASE_FEATURE_DECLARATION),
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
        "observed_curve_measures": observed_measures,
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
    measures = phase_curve_measures(gold_values, xp_values)
    gold_sigma = [uncertainty_gold[str(phase)] for phase in PHASES]
    xp_sigma = [uncertainty_xp[str(phase)] for phase in PHASES]
    scaling_sigma = (
        float(math.sqrt(sum(float(xp_sigma[index]) ** 2 for index in (0, 1, 2, 3))))
        if all(value is not None for value in xp_sigma)
        else None
    )
    snowball_sigma = (
        float(math.sqrt(sum(float(gold_sigma[index]) ** 2 for index in (0, 1, 2, 3))))
        if all(value is not None for value in gold_sigma)
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
        "scaling_index": (
            round(float(measures["scaling_index"]), 4)
            if measures["scaling_index"] is not None
            else None
        ),
        "snowball_index": (
            round(float(measures["snowball_index"]), 4)
            if measures["snowball_index"] is not None
            else None
        ),
        "uncertainty_scaling_index": round(scaling_sigma, 4) if scaling_sigma is not None else None,
        "uncertainty_snowball_index": round(snowball_sigma, 4) if snowball_sigma is not None else None,
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
    # A series is one evaluation unit.  A series can span several timestamps,
    # so splitting by timestamp would put one series in several test folds.
    if cluster_column and cluster_column in frame.columns:
        cluster_values = frame[cluster_column].astype("string")
    else:
        cluster_values = _game_series(frame, "phase frame")
    fallback_values = _game_series(frame, "phase frame")
    cluster_values = cluster_values.fillna(fallback_values)
    cluster_frame = pd.DataFrame({"cluster": cluster_values, "date": dates})
    cluster_dates = (
        cluster_frame.groupby("cluster", sort=False, observed=True)["date"]
        .agg(first="min", last="max")
        .sort_values(["last", "first"], kind="stable")
    )
    if len(cluster_dates) < 2:
        return ()
    boundaries = np.array_split(cluster_dates.index.to_numpy(dtype=object), n_splits)
    output: list[tuple[np.ndarray, np.ndarray]] = []
    for block in boundaries:
        if len(block) == 0:
            continue
        test_clusters = set(block.tolist())
        test_mask = cluster_values.isin(test_clusters)
        test_start = dates.loc[test_mask].min()
        cluster_last = cluster_values.map(cluster_dates["last"])
        train_mask = (
            dates.lt(test_start)
            & ~cluster_values.isin(test_clusters)
            & cluster_last.lt(test_start)
        )
        train = np.flatnonzero(train_mask.to_numpy())
        test = np.flatnonzero(test_mask.to_numpy())
        if len(train) < min_train_rows or len(test) == 0:
            continue
        output.append((train, test))
    return tuple(output)


def _cluster_boundary_diagnostics(
    frame: pd.DataFrame,
    test_indices: np.ndarray,
    cluster_column: str | None,
) -> dict[str, Any]:
    """Report rows kept out because their cluster continues into the future."""

    dates = _date_series(frame)
    if cluster_column and cluster_column in frame.columns:
        cluster_values = frame[cluster_column].astype("string")
    else:
        cluster_values = _game_series(frame, "phase frame")
    cluster_values = cluster_values.fillna(_game_series(frame, "phase frame"))
    test_mask = pd.Series(False, index=frame.index)
    test_mask.iloc[test_indices] = True
    test_start = dates.loc[test_mask].min()
    cluster_last = pd.DataFrame(
        {"cluster": cluster_values, "date": dates}, index=frame.index
    ).groupby("cluster", sort=False, observed=True)["date"].transform("max")
    boundary_mask = (
        dates.lt(test_start)
        & ~test_mask
        & ~cluster_values.isin(set(cluster_values.loc[test_mask].tolist()))
        & cluster_last.ge(test_start)
    )
    boundary_clusters = sorted(str(value) for value in cluster_values.loc[boundary_mask].unique())
    test_cluster_prior_rows = int(
        (
            dates.lt(test_start)
            & ~test_mask
            & cluster_values.isin(set(cluster_values.loc[test_mask].tolist()))
        ).sum()
    )
    return {
        "test_start": test_start.isoformat() if pd.notna(test_start) else None,
        "test_clusters": int(cluster_values.loc[test_mask].nunique()),
        "boundary_excluded_rows": int(boundary_mask.sum()),
        "boundary_excluded_clusters": len(boundary_clusters),
        "boundary_cluster_ids": boundary_clusters,
        "test_cluster_prior_rows": test_cluster_prior_rows,
        "definition": "rows before test start whose cluster has a row at or after test start",
    }


def _prediction_errors(
    artifact: Mapping[str, Any],
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    matrix, _ = _design(frame, feature_columns)
    missing_count = np.zeros(len(frame), dtype=int)
    for name in feature_columns:
        values = pd.to_numeric(frame[name], errors="coerce")
        missing_count += (~np.isfinite(values.to_numpy(dtype=float))).astype(int)
    errors: dict[str, dict[str, np.ndarray]] = {
        kind: {} for kind in ("gold", "xp")
    }
    for kind in ("gold", "xp"):
        for phase in PHASES:
            model = (artifact.get("models") or {}).get(kind, {}).get(str(phase))
            target = _target(frame, kind, phase)
            if isinstance(model, Mapping):
                coefficients = np.asarray(model.get("coefficients") or [], dtype=float)
                prediction = float(model.get("intercept") or 0.0) + matrix @ coefficients
            else:
                prediction = np.full(len(frame), np.nan)
            errors[kind][str(phase)] = target - prediction
    return errors, missing_count


def side_swap_invariance_report(
    artifact: Mapping[str, Any],
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    """Check that model outputs change sign after a blue/red swap."""

    swapped = side_swap_frame(frame)
    original_matrix, _ = _design(frame, feature_columns)
    swapped_matrix, _ = _design(swapped, feature_columns)
    complete = np.ones(len(frame), dtype=bool)
    for name in feature_columns:
        complete &= np.isfinite(
            pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        )
    report: dict[str, Any] = {}
    for kind in ("gold", "xp"):
        for phase in PHASES:
            model = (artifact.get("models") or {}).get(kind, {}).get(str(phase))
            key = f"{kind}_{phase}"
            if not isinstance(model, Mapping):
                report[key] = {"rows": 0, "max_abs_sum": None, "passed": False}
                continue
            coefficients = np.asarray(model.get("coefficients") or [], dtype=float)
            original = float(model.get("intercept") or 0.0) + original_matrix @ coefficients
            swapped_values = float(model.get("intercept") or 0.0) + swapped_matrix @ coefficients
            finite = np.isfinite(original) & np.isfinite(swapped_values) & complete
            max_abs_sum = (
                float(np.max(np.abs(original[finite] + swapped_values[finite])))
                if finite.any()
                else None
            )
            report[key] = {
                "rows": int(finite.sum()),
                "excluded_missing_rows": int((~complete).sum()),
                "max_abs_sum": max_abs_sum,
                "passed": bool(max_abs_sum is not None and max_abs_sum <= 1e-8),
            }
    return {
        "passed": bool(report) and all(bool(item["passed"]) for item in report.values()),
        "metrics": report,
        "definition": "predicted blue-minus-red curve plus swapped red-minus-blue curve",
    }


def _error_summary(values: Sequence[float]) -> dict[str, Any]:
    residual = np.asarray(values, dtype=float)
    valid = residual[np.isfinite(residual)]
    return {
        "rows": int(len(valid)),
        "rmse": float(np.sqrt(np.mean(valid * valid))) if len(valid) else None,
        "mae": float(np.mean(np.abs(valid))) if len(valid) else None,
        "bias": float(np.mean(valid)) if len(valid) else None,
    }


def _evaluate_transfer_slices(
    frame: pd.DataFrame,
    *,
    source_receipt: Mapping[str, Any],
    feature_columns: Sequence[str],
    columns: Sequence[str],
    alpha: float,
    max_groups_per_column: int | None = None,
) -> dict[str, Any]:
    """Evaluate earlier rows from other groups against each transfer group."""

    dates = _date_series(frame)
    output: dict[str, Any] = {}
    for column in columns:
        if column not in frame.columns:
            output[column] = {"available": False, "reason": "column_missing", "groups": {}}
            continue
        groups = frame[column].astype("string")
        groups = groups.where(groups.notna() & groups.str.strip().ne(""), "__missing__")
        reports: dict[str, Any] = {}
        unique_groups = sorted(str(value) for value in groups.unique())
        if max_groups_per_column is not None:
            unique_groups = unique_groups[: max(0, int(max_groups_per_column))]
        for group in unique_groups:
            test_mask = groups.eq(group)
            if not test_mask.any():
                continue
            test_start = dates.loc[test_mask].min()
            train_mask = dates.lt(test_start) & ~test_mask
            train = frame.loc[train_mask].copy()
            test = frame.loc[test_mask & dates.ge(test_start)].copy()
            if len(train) < max(1, len(feature_columns) + 1) or test.empty:
                reports[group] = {
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                    "available": False,
                    "reason": "insufficient_chronological_support",
                }
                continue
            artifact = fit_phase_curve(
                train,
                source_receipt=source_receipt,
                feature_columns=feature_columns,
                alpha=alpha,
            )
            residuals, _missing = _prediction_errors(artifact, test, feature_columns)
            metric_report: dict[str, Any] = {}
            for kind in ("gold", "xp"):
                metric_report[kind] = {}
                for phase in PHASES:
                    residual = residuals[kind][str(phase)]
                    target = _target(test, kind, phase)
                    valid = np.isfinite(residual) & np.isfinite(target)
                    model_summary = _error_summary(residual[valid])
                    baseline_summary = _error_summary(target[valid])
                    metric_report[kind][str(phase)] = {
                        **model_summary,
                        "baseline_zero": baseline_summary,
                        "baseline_rows_match": bool(
                            model_summary["rows"] == baseline_summary["rows"]
                        ),
                    }
            reports[group] = {
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "available": True,
                "metrics": metric_report,
            }
        output[column] = {
            "available": bool(reports),
            "groups": reports,
            "definition": "train on earlier rows from other groups; test on later held-out group",
        }
    return output


def evaluate_phase_curve(
    frame: pd.DataFrame,
    *,
    source_receipt: Mapping[str, Any],
    feature_columns: Sequence[str],
    n_splits: int = 3,
    cluster_column: str | None = None,
    alpha: float = 10.0,
    transfer_columns: Sequence[str] = ("region", "patch"),
    max_transfer_groups: int | None = None,
) -> dict[str, Any]:
    """Evaluate each phase on future rows with fold-internal fitting."""

    bound = bind_phase_source(frame, source_receipt, allow_subset=True)
    value = bound.frame.copy()

    folds = chronological_folds(
        value,
        n_splits=n_splits,
        min_train_rows=max(1, len(feature_columns) + 1),
        cluster_column=cluster_column,
    )
    errors: dict[str, dict[str, list[float]]] = {
        kind: {str(phase): [] for phase in PHASES} for kind in ("gold", "xp")
    }
    baseline_errors: dict[str, dict[str, list[float]]] = {
        kind: {str(phase): [] for phase in PHASES} for kind in ("gold", "xp")
    }
    missingness_errors: dict[str, dict[str, dict[str, list[float]]]] = {
        kind: {
            str(phase): {"complete": [], "any_missing": []}
            for phase in PHASES
        }
        for kind in ("gold", "xp")
    }
    fold_rows: list[dict[str, Any]] = []
    side_swap_checks: list[dict[str, Any]] = []
    for fold_number, (train_indices, test_indices) in enumerate(folds):
        train = value.iloc[train_indices].copy()
        test = value.iloc[test_indices].copy()
        artifact = fit_phase_curve(
            train,
            source_receipt=source_receipt,
            feature_columns=feature_columns,
            alpha=alpha,
        )
        prediction_errors, missing_count = _prediction_errors(artifact, test, feature_columns)
        side_swap_checks.append(side_swap_invariance_report(artifact, test, feature_columns))
        boundary = _cluster_boundary_diagnostics(value, test_indices, cluster_column)
        row: dict[str, Any] = {
            "fold": fold_number,
            "train_rows": len(train),
            "test_rows": len(test),
            "cluster_boundary_exclusions": boundary,
        }
        for kind in ("gold", "xp"):
            for phase in PHASES:
                residual = prediction_errors[kind][str(phase)]
                target = _target(test, kind, phase)
                valid = np.isfinite(residual)
                if valid.any():
                    errors[kind][str(phase)].extend(float(value) for value in residual)
                    baseline_errors[kind][str(phase)].extend(
                        float(value) for value in target[valid]
                    )
                    missingness_errors[kind][str(phase)]["complete"].extend(
                        float(value)
                        for value in residual[valid & (missing_count == 0)]
                    )
                    missingness_errors[kind][str(phase)]["any_missing"].extend(
                        float(value)
                        for value in residual[valid & (missing_count > 0)]
                    )
                row[f"{kind}_{phase}_rows"] = int(valid.sum())
        fold_rows.append(row)
    metrics: dict[str, dict[str, Any]] = {kind: {} for kind in ("gold", "xp")}
    for kind in ("gold", "xp"):
        for phase in PHASES:
            metrics[kind][str(phase)] = _error_summary(errors[kind][str(phase)])
            baseline = _error_summary(baseline_errors[kind][str(phase)])
            metrics[kind][str(phase)]["baseline_zero"] = baseline
            model_rows = metrics[kind][str(phase)]["rows"]
            metrics[kind][str(phase)]["baseline_rows_match"] = bool(
                baseline["rows"] == model_rows
            )
            if (
                baseline["rmse"] is not None
                and metrics[kind][str(phase)]["rmse"] is not None
            ):
                metrics[kind][str(phase)]["rmse_gain_vs_zero"] = float(
                    baseline["rmse"] - metrics[kind][str(phase)]["rmse"]
                )
            if (
                baseline["mae"] is not None
                and metrics[kind][str(phase)]["mae"] is not None
            ):
                metrics[kind][str(phase)]["mae_gain_vs_zero"] = float(
                    baseline["mae"] - metrics[kind][str(phase)]["mae"]
                )
            metrics[kind][str(phase)]["missingness"] = {
                key: _error_summary(value)
                for key, value in missingness_errors[kind][str(phase)].items()
            }
    transfer = _evaluate_transfer_slices(
        value,
        source_receipt=bound.receipt,
        feature_columns=feature_columns,
        columns=transfer_columns,
        alpha=alpha,
        max_groups_per_column=max_transfer_groups,
    )
    side_swap = {
        "passed": bool(side_swap_checks) and all(item["passed"] for item in side_swap_checks),
        "folds": side_swap_checks,
        "definition": "predicted blue-minus-red curve plus swapped red-minus-blue curve",
    }
    fallback_rows = int(
        value["series_id_source"].astype("string").eq("game_fallback").sum()
        if "series_id_source" in value.columns
        else 0
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "chronological_fold_internal_ridge",
        "source": SOURCE,
        "source_as_of": bound.receipt["source_as_of"],
        "source_game_count": int(bound.receipt["source_game_count"]),
        "source_identity_sha256": str(bound.receipt["source_identity_sha256"]),
        "accepted_game_ids": list(bound.receipt["accepted_game_ids"]),
        "source_receipt_sha256": _sha256_bytes(_canonical_json_bytes(bound.receipt)),
        "folds": fold_rows,
        "metrics": metrics,
        "fold_count": len(fold_rows),
        "cluster_safe": fallback_rows == 0,
        "cluster_column": cluster_column or "game_uid",
        "cluster_fallback_rows": fallback_rows,
        "cluster_boundary_exclusions": {
            "rows": int(
                sum(
                    int(row["cluster_boundary_exclusions"]["boundary_excluded_rows"])
                    for row in fold_rows
                )
            ),
            "clusters": int(
                sum(
                    int(row["cluster_boundary_exclusions"]["boundary_excluded_clusters"])
                    for row in fold_rows
                )
            ),
            "folds": [row["cluster_boundary_exclusions"] for row in fold_rows],
            "definition": "cluster last date must be before test start for train eligibility",
        },
        "missingness": {
            kind: {
                phase: metrics[kind][phase]["missingness"]
                for phase in PHASE_KEYS
            }
            for kind in ("gold", "xp")
        },
        "transfer": transfer,
        "regional_transfer": transfer.get("region", {}),
        "patch_transfer": transfer.get("patch", {}),
        "side_swap_invariance": side_swap,
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
    "PHASE_FEATURE_DECLARATION",
    "PHASE_FEATURE_FAMILY",
    "PHASES",
    "SCHEMA_VERSION",
    "assert_pregame_feature_names",
    "bind_phase_source",
    "build_strict_prior_team_features",
    "chronological_folds",
    "evaluate_phase_curve",
    "fit_phase_curve",
    "phase_curve_measures",
    "prepare_phase_frame",
    "score_phase_curve",
    "side_swap_invariance_report",
    "side_swap_frame",
    "strict_prior_final_history",
]
