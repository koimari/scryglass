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
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import hashlib
import json
import math
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from lol_kills.etl.source_keys import canonical_source_game_key
from lol_kills.ratings.global_player_bt import (
    ANCHOR_METRIC_Z_CLIP,
    PrefixBaselineCache,
    _contribution_metrics,
    _role_normalized_composite,
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
PLAYER_FINAL_METRICS = (
    "cspm",
    "earned gpm",
    "earnedgoldshare",
    "dpm",
    "damageshare",
    "kills",
    "deaths",
    "assists",
    "wpm",
    "wcpm",
)
PLAYER_CHECKPOINT_METRICS = tuple(
    f"{stem}at{checkpoint}"
    for checkpoint in CHECKPOINTS
    for stem in ("gold", "xp", "cs", "kills", "assists", "deaths")
)
TEAM_CHECKPOINT_METRICS = tuple(
    f"{stem}at{checkpoint}"
    for checkpoint in CHECKPOINTS
    for stem in ("gold", "xp", "cs", "kills", "assists", "deaths")
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

    if not isinstance(receipt, Mapping):
        raise FutureValueSourceError("verified source receipt is required")
    required = (
        "source_as_of",
        "source_game_count",
        "source_identity_sha256",
        "accepted_game_ids",
        "model_eligible_game_count",
        "model_eligible_identity_sha256",
        "model_eligible_game_ids",
        "source_files",
        "receipt_sha256",
    )
    if any(field not in receipt for field in required):
        raise FutureValueSourceError("verified source receipt is incomplete")
    receipt_hash = receipt.get("receipt_sha256")
    if not isinstance(receipt_hash, str) or re.fullmatch(r"[0-9a-f]{64}", receipt_hash, re.I) is None:
        raise FutureValueSourceError("verified source receipt hash is invalid")
    payload = dict(receipt)
    payload.pop("receipt_sha256", None)
    if hashlib.sha256(_canonical_json_bytes(payload)).hexdigest() != receipt_hash:
        raise FutureValueSourceError("verified source receipt hash does not match payload")
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
    ):
        raise FutureValueSourceError("verified source receipt census identity is invalid")
    _utc_timestamp(receipt["source_as_of"], "source_as_of")
    source_files = receipt["source_files"]
    if not isinstance(source_files, Mapping) or not source_files:
        raise FutureValueSourceError("verified source receipt has no source file hashes")
    for label, record in source_files.items():
        if not isinstance(record, Mapping) or not isinstance(record.get("bytes"), int):
            raise FutureValueSourceError(f"verified source file record is invalid: {label}")
        if re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256") or ""), re.I) is None:
            raise FutureValueSourceError(f"verified source file hash is invalid: {label}")
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


def _side(value: Any) -> str | None:
    text = str(value or "").strip().casefold()
    if text in {"blue", "b"}:
        return "blue"
    if text in {"red", "r"}:
        return "red"
    return None


def _role(value: Any) -> str | None:
    text = str(value or "").strip().casefold()
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
    unique_keys = work[keys].drop_duplicates().sort_values(keys, kind="stable")
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
    stable_player = player_ids.map(lambda value: _stable_identity(value, "oe:player:"))
    stable_team = team_ids.map(lambda value: _stable_identity(value, "oe:team:"))
    if not bool(stable_player.all()) or not bool(stable_team.all()):
        raise FutureValueSourceError("player form contains unstable player or team identity")
    if champions.isna().any() or champions.eq("").any():
        raise FutureValueSourceError("player form contains missing champion identity")
    base = pd.DataFrame(
        {
            "game_id": metrics["_game_id"].astype(str).to_numpy(),
            "date": pd.to_datetime(metrics["_date"], utc=True).to_numpy(),
            "side": metrics["_side"].astype(str).to_numpy(),
            "role": metrics["_role"].astype(str).to_numpy(),
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
    champion_text = str(champion).strip().casefold()
    role_text = str(role).strip().casefold()
    if not champion_text or champion_text == "nan" or not role_text or role_text == "nan":
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


def _map_model_frame(maps: pd.DataFrame) -> pd.DataFrame:
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
    series_column = next(
        (name for name in ("series_id", "seriesid", "match_id", "matchid") if name in frame.columns),
        None,
    )
    valid_authoritative_series = False
    if series_column is not None:
        series = frame[series_column].astype("string").str.strip()
        valid_authoritative_series = bool(series.notna().all() and series.ne("").all())
        if valid_authoritative_series:
            frame["series_id"] = series
            frame.attrs["series_cluster_source"] = f"authoritative:{series_column}"
    if not valid_authoritative_series:
        team_columns = next(
            (
                (blue_name, red_name)
                for blue_name, red_name in (
                    ("blue_team_key", "red_team_key"),
                    ("blue_teamid", "red_teamid"),
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
                frame["series_id"] = [
                    "proxy:"
                    + "|".join(
                        (
                            str(league_value).strip().casefold(),
                            str(tournament_value).strip().casefold(),
                            str(day.date()),
                            team_pair,
                        )
                    )
                    for league_value, tournament_value, day, team_pair in zip(
                        league, tournament, frame["date"], pair
                    )
                ]
                frame.attrs["series_cluster_source"] = "unordered_team_pair_day_proxy"
            else:
                frame["series_id"] = frame["game_id"]
                frame.attrs["series_cluster_source"] = "game_id_fallback"
        else:
            frame["series_id"] = frame["game_id"]
            frame.attrs["series_cluster_source"] = "game_id_fallback"
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


MODEL_FEATURES = tuple(
    [f"player_form_{metric}" for metric in FORM_METRICS]
    + [f"rank_3_player_atom_{index}" for index in range(1, RANK_3 + 1)]
    + [f"rank_3_champion_role_atom_{index}" for index in range(1, RANK_3 + 1)]
    + [
        "team_prior_win_diff",
        "roster_continuity_diff",
        "player_form_missing_rate",
        "rank_3_atom_missing",
        "rank_3_champion_role_atom_missing",
    ]
)


def build_future_value_design(
    maps: pd.DataFrame,
    form: pd.DataFrame,
    atom_model: Rank3AtomModel,
) -> pd.DataFrame:
    """Build side-neutral map differences from pregame state only."""

    map_frame = _map_model_frame(maps)
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
        .groupby(level=0, sort=False)
        .mean()
    )
    side_values[champion_atom_columns] = side_values[champion_atom_columns].fillna(0.0)
    grouped_values = side_values.groupby(
        ["game_id", "side"], sort=False, observed=True
    )
    side_means = grouped_values[side_feature_names].mean()
    side_finite_counts = grouped_values[side_feature_names].count()
    side_means = side_means.mask(side_finite_counts.lt(5))
    side_wide = side_means.unstack("side")
    side_missing = (
        side_values[form_metric_names].isna()
        .groupby([side_values["game_id"], side_values["side"]], sort=False, observed=True)
        .mean()
        .mean(axis=1)
        .groupby(level=0, sort=False)
        .mean()
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
        .groupby(level=0, sort=False)
        .mean()
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
    for source_name in side_feature_names:
        output_name = source_name.replace("prior_form_", "player_form_")
        blue = side_wide[source_name].get("blue", pd.Series(dtype=float))
        red = side_wide[source_name].get("red", pd.Series(dtype=float))
        design[output_name] = blue.sub(red).reindex(design.index)
    design["team_prior_win_diff"] = (
        team_wide["prior_team_win"].get("blue", pd.Series(dtype=float))
        .sub(team_wide["prior_team_win"].get("red", pd.Series(dtype=float)))
        .reindex(design.index)
    )
    design["roster_continuity_diff"] = (
        team_wide["roster_continuity"].get("blue", pd.Series(dtype=float))
        .sub(team_wide["roster_continuity"].get("red", pd.Series(dtype=float)))
        .reindex(design.index)
    )
    design["blue_roster_continuity"] = (
        team_wide["roster_continuity"].get("blue", pd.Series(dtype=float))
        .reindex(design.index)
    )
    design["red_roster_continuity"] = (
        team_wide["roster_continuity"].get("red", pd.Series(dtype=float))
        .reindex(design.index)
    )
    design["player_form_missing_rate"] = side_missing.reindex(design.index)
    support_mean = support_summary.groupby(level=0, sort=False)[support_columns].mean().mean(axis=1)
    effective_support_mean = support_summary.groupby(level=0, sort=False)[
        effective_support_columns
    ].mean().mean(axis=1)
    design["player_form_support_mean"] = support_mean.reindex(design.index)
    design["player_form_effective_support_mean"] = effective_support_mean.reindex(design.index)
    design["player_form_support_uncertainty_proxy"] = 1.0 / np.sqrt(
        1.0 + design["player_form_effective_support_mean"]
    )
    design["rank_3_atom_missing"] = rank_missing.reindex(design.index)
    design["rank_3_champion_role_atom_missing"] = champion_atom_missing.reindex(
        design.index
    )
    design["rank_3_atom_support_uncertainty_proxy"] = 1.0 / np.sqrt(
        1.0 + rank_support.mean(axis=1).reindex(design.index)
    )
    design["model_features_complete"] = np.isfinite(
        design[list(MODEL_FEATURES)].to_numpy(dtype=float)
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
    design = design.reset_index(drop=True)
    assert_pregame_feature_names(MODEL_FEATURES)
    design.attrs["feature_names"] = MODEL_FEATURES
    design.attrs["series_cluster_source"] = map_frame.attrs.get("series_cluster_source")
    return design


@dataclass(frozen=True)
class FutureValueFoldModel:
    """A fitted development model for one chronological fold."""

    feature_names: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    coefficients: np.ndarray
    intercept: float
    atom_model: Rank3AtomModel
    fit_game_ids: tuple[str, ...]
    fit_window_end: str
    train_rows: int
    withheld_rows: int
    source_receipt: Mapping[str, Any]

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

    def predict_logit(self, design: pd.DataFrame) -> pd.Series:
        missing = sorted(set(self.feature_names) - set(design.columns))
        if missing:
            raise FutureValueSourceError("prediction design is missing: " + ", ".join(missing))
        values = design[list(self.feature_names)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        available = np.isfinite(values).all(axis=1)
        output = np.full(len(design), np.nan, dtype=float)
        if available.any():
            scaled = (values[available] - self.means) / self.scales
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                candidate = self.intercept + scaled @ self.coefficients
            available_positions = np.flatnonzero(available)
            finite_candidate = np.isfinite(candidate)
            output[available_positions[finite_candidate]] = candidate[finite_candidate]
        return pd.Series(output, index=design.index, name="future_value_logit")

    def predict_probability(self, design: pd.DataFrame) -> pd.Series:
        logits = self.predict_logit(design)
        values = logits.to_numpy(dtype=float)
        finite = np.isfinite(values)
        values[finite] = 1.0 / (1.0 + np.exp(-np.clip(values[finite], -40.0, 40.0)))
        return pd.Series(values, index=design.index, name="future_value_probability")

    def player_value_logit(self, form: pd.DataFrame) -> pd.DataFrame:
        """Return per-player logit contributions with support and uncertainty."""

        atoms = self.atom_model.transform(form)
        combined = pd.concat([form.reset_index(drop=True), atoms.reset_index(drop=True)], axis=1)
        output = pd.DataFrame(index=combined.index)
        values = np.zeros(len(combined), dtype=float)
        available = np.ones(len(combined), dtype=bool)
        metric_features = {f"player_form_{metric}" for metric in FORM_METRICS}
        player_features = {
            feature_index: feature
            for feature_index, feature in enumerate(self.feature_names)
            if feature in metric_features or feature.startswith("rank_3_")
        }
        for feature_index, feature in player_features.items():
            source = (
                feature.replace("player_form_", "prior_form_")
                if feature.startswith("player_form_")
                else feature
            )
            if source not in combined.columns:
                available[:] = False
                continue
            numeric = pd.to_numeric(combined[source], errors="coerce").to_numpy(dtype=float)
            available &= np.isfinite(numeric)
            values += np.where(
                np.isfinite(numeric),
                ((numeric - self.means[feature_index]) / self.scales[feature_index])
                * self.coefficients[feature_index]
                / 5.0,
                0.0,
            )
        output["player_value_logit"] = np.where(available, values, np.nan)
        support = combined[
            [f"prior_form_{metric}_effective_support" for metric in FORM_METRICS]
        ].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        output["support"] = support
        output["support_uncertainty_proxy"] = 1.0 / np.sqrt(1.0 + support)
        output["available"] = available
        output.index = form.index
        return output

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": MODEL_FIT_SCHEMA_VERSION,
            "fit_game_count": len(self.fit_game_ids),
            "fit_game_ids": list(self.fit_game_ids),
            "fit_window_end": self.fit_window_end,
            "feature_names": list(self.feature_names),
            "metric_weights": self.metric_weights,
            "coefficients": self.coefficient_map,
            "intercept": float(self.intercept),
            "rank_3": {
                "rank": int(self.atom_model.rank),
                "fit_window_end": self.atom_model.fit_window_end,
                "champion_role_cells": len(self.atom_model.champion_role_coordinates),
                "fit_game_count": len(self.atom_model.fit_game_ids),
                "fit_game_ids": list(self.atom_model.fit_game_ids),
                "fit_game_identity_sha256": identity_sha256(self.atom_model.fit_game_ids),
            },
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
) -> tuple[FutureValueFoldModel, pd.DataFrame]:
    """Fit one fold with all representation work bound to its train games."""

    map_frame = _map_model_frame(maps)
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
    atom_model = fit_rank3_player_champion_role_atoms(
        form,
        train_game_ids=train_ids,
        rank=rank,
        min_cell_support=min_cell_support,
        fit_window_end=None if fit_window_end is None else boundary,
    )
    design = build_future_value_design(map_frame, form, atom_model)
    train = design[design["game_id"].isin(train_ids)].copy()
    feature_names = tuple(MODEL_FEATURES)
    numeric_train = train[list(feature_names)].apply(pd.to_numeric, errors="coerce")
    complete = np.isfinite(numeric_train.to_numpy(dtype=float)).all(axis=1)
    target = pd.to_numeric(train["target"], errors="coerce")
    usable = complete & target.isin({0, 1})
    usable_train = train.loc[usable]
    if len(usable_train) < 20 or usable_train["target"].nunique() != 2:
        raise FutureValueSourceError("future-value fold has insufficient complete two-class training rows")
    matrix = usable_train[list(feature_names)].to_numpy(dtype=float)
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0, ddof=0)
    scales = np.where(np.isfinite(scales) & (scales > 1e-12), scales, 1.0)
    classifier = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=0,
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        classifier.fit(
            (matrix - means) / scales,
            usable_train["target"].to_numpy(dtype=int),
        )
    if not np.isfinite(classifier.coef_).all() or not np.isfinite(classifier.intercept_).all():
        raise FutureValueSourceError("future-value classifier fit is non-finite")
    model = FutureValueFoldModel(
        feature_names=feature_names,
        means=means,
        scales=scales,
        coefficients=classifier.coef_[0].astype(float),
        intercept=float(classifier.intercept_[0]),
        atom_model=atom_model,
        fit_game_ids=train_ids,
        fit_window_end=_utc_text(boundary),
        train_rows=int(len(usable_train)),
        withheld_rows=int((~usable).sum()),
        source_receipt=dict(source_receipt or {}),
    )
    return model, design


def chronological_whole_series_folds(
    maps: pd.DataFrame,
    *,
    n_folds: int = 3,
) -> list[dict[str, Any]]:
    """Return expanding chronological folds with whole series clusters."""

    if int(n_folds) < 1:
        raise FutureValueSourceError("chronological fold count must be positive")
    frame = _map_model_frame(maps)
    series_summary = (
        frame.groupby("series_id", sort=True, observed=True)
        .agg(first_date=("date", "min"), last_date=("date", "max"))
        .sort_values(["first_date", "series_id"], kind="stable")
    )
    date_blocks = list(series_summary.groupby("first_date", sort=True, observed=True))
    if len(date_blocks) < int(n_folds) + 1:
        raise FutureValueSourceError("chronological folds need more timestamp blocks")
    chunks = np.array_split(np.arange(len(date_blocks)), int(n_folds) + 1)
    folds: list[dict[str, Any]] = []
    for fold_index in range(1, len(chunks)):
        validation_blocks = [date_blocks[int(index)] for index in chunks[fold_index]]
        if not validation_blocks:
            continue
        validation_series = {str(series_id) for _date, block in validation_blocks for series_id in block.index}
        validation_min = min(block["first_date"].min() for _date, block in validation_blocks)
        train_series = {
            str(series_id)
            for series_id, row in series_summary.iterrows()
            if str(series_id) not in validation_series and row["last_date"] < validation_min
        }
        train_ids = tuple(sorted(frame.loc[frame["series_id"].astype(str).isin(train_series), "game_id"]))
        validation_ids = tuple(sorted(frame.loc[frame["series_id"].astype(str).isin(validation_series), "game_id"]))
        if not train_ids or not validation_ids:
            continue
        train_max = frame.loc[frame["game_id"].isin(train_ids), "date"].max()
        valid_min = frame.loc[frame["game_id"].isin(validation_ids), "date"].min()
        if not pd.Timestamp(train_max) < pd.Timestamp(valid_min):
            raise FutureValueSourceError("chronological fold has a non-strict date boundary")
        folds.append(
            {
                "fold": int(fold_index),
                "train_game_ids": train_ids,
                "validation_game_ids": validation_ids,
                "train_series_ids": tuple(sorted(train_series)),
                "validation_series_ids": tuple(sorted(validation_series)),
                "train_end": _utc_text(train_max),
                "validation_start": _utc_text(valid_min),
            }
        )
    if not folds:
        raise FutureValueSourceError("chronological fold construction produced no usable folds")
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
) -> dict[str, Any]:
    """Report complete-case coverage without hiding withheld rows."""

    if "model_features_complete" not in validation.columns:
        return {"status": "unavailable", "blockers": ["missingness_indicator_missing"]}
    complete = validation["model_features_complete"].astype(bool)
    paired = target.notna() & probability.notna()
    return {
        "status": "available",
        "total_rows": int(len(validation)),
        "complete_case_rows": int(complete.sum()),
        "incomplete_case_rows": int((~complete).sum()),
        "predicted_rows": int(paired.sum()),
        "withheld_rows": int((~paired).sum()),
        "complete_case_metrics": _classification_metrics(
            target.loc[paired & complete], probability.loc[paired & complete]
        ),
        "blockers": ["complete_case_missingness_only"] if (~complete).any() else [],
    }


def _side_swap_metrics(
    model: FutureValueFoldModel,
    validation: pd.DataFrame,
    target: pd.Series,
    probability: pd.Series,
) -> dict[str, Any]:
    """Check the antisymmetry of the same validation rows after a side swap."""

    paired = target.notna() & probability.notna()
    if not paired.any():
        return {"status": "unavailable", "rows": 0, "blockers": ["side_swap_rows_missing"]}
    swapped = validation.copy()
    scalar_features = {
        "player_form_missing_rate",
        "rank_3_atom_missing",
        "rank_3_champion_role_atom_missing",
    }
    for feature in MODEL_FEATURES:
        if feature not in scalar_features:
            swapped[feature] = -pd.to_numeric(swapped[feature], errors="coerce")
    swapped_probability = model.predict_probability(swapped).loc[paired]
    swapped_target = 1.0 - target.loc[paired]
    original_probability = probability.loc[paired]
    symmetry_error = np.abs(
        swapped_probability.to_numpy(dtype=float)
        - (1.0 - original_probability.to_numpy(dtype=float))
    )
    return {
        "status": "available",
        "rows": int(len(swapped_probability)),
        "metrics": _classification_metrics(swapped_target, swapped_probability),
        "mean_probability_complement_error": float(np.nanmean(symmetry_error)),
        "max_probability_complement_error": float(np.nanmax(symmetry_error)),
        "blockers": [],
    }


def _roster_change_labels(frame: pd.DataFrame) -> pd.Series | None:
    required = {"blue_roster_continuity", "red_roster_continuity"}
    if not required.issubset(frame.columns):
        return None
    blue = pd.to_numeric(frame["blue_roster_continuity"], errors="coerce")
    red = pd.to_numeric(frame["red_roster_continuity"], errors="coerce")
    labels = pd.Series("<missing>", index=frame.index, dtype="string")
    stable = blue.ge(1.0) & red.ge(1.0)
    changed = blue.lt(1.0) | red.lt(1.0)
    labels.loc[stable] = "stable_roster"
    labels.loc[changed] = "roster_change"
    return labels


def _support_labels(frame: pd.DataFrame, threshold: float = 5.0) -> pd.Series | None:
    field = "player_form_effective_support_mean"
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


def evaluate_future_value(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    *,
    n_folds: int = 3,
    half_life_days: float = TIME_DECAY_HALF_LIFE_DAYS,
    min_cell_support: int = 1,
    source_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a development-only chronological whole-series evaluation."""

    map_frame = _map_model_frame(maps)
    verified_eligible_ids = _validate_verified_source_receipt(
        source_receipt,
        map_frame,
        require_full_eligible_set=True,
    )
    form = build_time_decayed_prior_player_form(map_frame, players, half_life_days=half_life_days)
    folds = chronological_whole_series_folds(map_frame, n_folds=n_folds)
    fold_reports: list[dict[str, Any]] = []
    pooled_targets: list[pd.Series] = []
    pooled_predictions: list[pd.Series] = []
    pooled_baselines: list[pd.Series] = []
    pooled_slice_blockers: list[str] = []
    for fold in folds:
        model, design = fit_future_value_model(
            map_frame,
            form,
            train_game_ids=fold["train_game_ids"],
            fit_window_end=fold["validation_start"],
            min_cell_support=min_cell_support,
            source_receipt=source_receipt,
        )
        validation = design[design["game_id"].isin(fold["validation_game_ids"])].copy()
        prediction = model.predict_probability(validation)
        target = validation["target"].astype(float)
        paired_mask = target.notna() & prediction.notna()
        paired_target = target.loc[paired_mask]
        paired_prediction = prediction.loc[paired_mask]
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
        train_design = design[design["game_id"].isin(fold["train_game_ids"])].copy()
        calibration = _calibration_metrics(paired_target, paired_prediction)
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
        missingness = _missingness_metrics(validation, target, prediction)
        side_swap = _side_swap_metrics(model, validation, target, prediction)
        slice_reports = {
            "regional_transfer": region_slice,
            "patch_transfer": patch_slice,
            "roster_change": roster_slice,
            "tournament_boundary": tournament_slice,
            "sparse_support": support_slice,
        }
        for report in (*slice_reports.values(), missingness, side_swap):
            pooled_slice_blockers.extend(str(value) for value in report.get("blockers", []))
        paired_game_ids = tuple(
            sorted(validation.loc[paired_mask, "game_id"].astype(str))
        )
        pooled_targets.append(paired_target)
        pooled_predictions.append(paired_prediction)
        pooled_baselines.append(paired_baseline)
        fold_reports.append(
            {
                "fold": fold["fold"],
                "train_end": fold["train_end"],
                "validation_start": fold["validation_start"],
                "train_series_count": len(fold["train_series_ids"]),
                "validation_series_count": len(fold["validation_series_ids"]),
                "candidate": _classification_metrics(paired_target, paired_prediction),
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
                "coefficients": model.coefficient_map,
                "rank_3": model.receipt()["rank_3"],
                "prediction_coverage": float(prediction.notna().mean()),
                "withheld_rows": int(prediction.isna().sum()),
                "metric_weights": model.metric_weights,
                "calibration": calibration,
                "baseline_calibration": baseline_calibration,
                "regional_transfer": region_slice,
                "patch_transfer": patch_slice,
                "roster_change": roster_slice,
                "tournament_boundary": tournament_slice,
                "sparse_support": support_slice,
                "missingness": missingness,
                "side_swap": side_swap,
            }
        )
    cluster_source = map_frame.attrs.get("series_cluster_source")
    blockers = [
        "current_player_team_rating_comparison_missing",
        "support_uncertainty_proxy_not_calibrated",
    ]
    if int(n_folds) < 3 or len(fold_reports) < int(n_folds):
        blockers.append("complete_chronological_evaluation_missing")
    if int(n_folds) < 3:
        blockers.append("bounded_fold_count_below_protocol")
    pooled_target = pd.concat(pooled_targets, ignore_index=True)
    pooled_prediction = pd.concat(pooled_predictions, ignore_index=True)
    pooled_baseline = pd.concat(pooled_baselines, ignore_index=True)
    pooled_calibration_report = _calibration_metrics(pooled_target, pooled_prediction)
    pooled_baseline_calibration_report = _calibration_metrics(pooled_target, pooled_baseline)
    blockers.extend(pooled_slice_blockers)
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
    return {
        "schema_version": MODEL_FIT_SCHEMA_VERSION,
        "status": "development_evaluated",
        "source": {
            "game_count": int(len(map_frame)),
            "source_game_count": int(source_receipt["source_game_count"]),
            "source_identity_sha256": str(source_receipt["source_identity_sha256"]),
            "accepted_game_ids": list(source_receipt["accepted_game_ids"]),
            "model_eligible_game_count": int(len(verified_eligible_ids)),
            "model_eligible_identity_sha256": identity_sha256(verified_eligible_ids),
            "model_eligible_game_ids": list(verified_eligible_ids),
            "series_cluster_source": cluster_source,
            "half_life_days": float(half_life_days),
            "source_as_of": _utc_text(source_receipt["source_as_of"]),
            "source_files": source_receipt["source_files"],
            "source_receipt_sha256": str(source_receipt["receipt_sha256"]),
            "source_latest": _utc_text(map_frame["date"].max()),
        },
        "evaluation": {
            "requested_folds": int(n_folds),
            "valid_folds": len(fold_reports),
            "minimum_folds": 3,
            "pooled_rows": int(len(pooled_target)),
            "pooled_candidate": _classification_metrics(pooled_target, pooled_prediction),
            "pooled_intercept_baseline": _classification_metrics(
                pooled_target, pooled_baseline
            ),
            "pooled_calibration": pooled_calibration_report,
            "pooled_baseline_calibration": pooled_baseline_calibration_report,
        },
        "folds": fold_reports,
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


def future_value_model_contract() -> dict[str, Any]:
    """Return the development contract for future player and team value."""

    return {
        "schema_version": MODEL_CONTRACT_VERSION,
        "status": "development_only",
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
            "player_state": "strictly prior time-decayed form with fold-local rank-3 atoms and support diagnostics",
            "team_state": "exact-five roster aggregation plus strictly prior team win state and roster continuity",
            "metric_weights": "fit inside development folds; no hand-assigned performance weights",
        },
        "evaluation": [
            "chronological whole-series folds",
            "intercept baseline and proper scores",
            "proxy series clusters with an authoritative-series blocker",
            "validation calibration bins and pooled proper scores",
            "regional and patch transfer slices with training-support checks",
            "roster-change and tournament-boundary slices",
            "complete-case and withheld-row missingness counts",
            "sparse-support slice and support-uncertainty diagnostics",
            "side-swap invariance diagnostic",
        ],
        "future_scope_blockers": [
            "current_player_team_rating_comparison",
            "composition_specific_phase_curve",
            "calibrated_uncertainty",
            "missingness_robust_fit",
            "authoritative_series_identity",
            "authoritative_series_cluster_evaluation",
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
    "MODEL_FEATURES",
    "MODEL_FIT_SCHEMA_VERSION",
    "Rank3AtomModel",
    "assert_pregame_feature_names",
    "bind_accepted_future_value_source",
    "build_strict_prior_player_form",
    "build_future_value_design",
    "build_time_decayed_prior_player_form",
    "chronological_whole_series_folds",
    "evaluate_future_value",
    "fit_future_value_model",
    "fit_rank3_player_champion_role_atoms",
    "future_value_model_contract",
    "load_accepted_future_value_source",
    "team_value_difference",
    "write_source_receipt",
]
