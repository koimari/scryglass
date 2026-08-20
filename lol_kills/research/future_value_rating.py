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
    else:
        raise FutureValueSourceError(f"{label} has no game identity column")
    result = pd.Series(values, index=frame.index, dtype="string")
    if result.isna().any() or result.str.strip().eq("").any():
        raise FutureValueSourceError(f"{label} contains an empty canonical game identity")
    return result


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


def future_value_model_contract() -> dict[str, Any]:
    """Return the development contract for future player and team value."""

    return {
        "schema_version": MODEL_CONTRACT_VERSION,
        "status": "development_only",
        "estimands": {
            "future_player_value": (
                "Expected marginal map-win logit contribution for one player at the next "
                "map boundary, conditional on role, champion atom profile, opponent, patch, "
                "competition, and strictly prior form."
            ),
            "future_team_value": (
                "Expected exact-roster map-win logit before the next map. It is the roster "
                "sum plus team continuity, regional bridge, composition, and phase-curve terms."
            ),
        },
        "player_components": [
            "global_dynamic_skill",
            "role_effect",
            "rank_3_player_atom_loading",
            "strictly_prior_performance_form",
            "uncertainty_and_support",
        ],
        "team_components": [
            "exact_roster_player_sum",
            "roster_continuity_and_synergy_residual",
            "competition_and_regional_bridge",
            "neutral_draft_value",
            "composition_specific_phase_curve",
            "uncertainty_and_source_coverage",
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
            "player_state": "dynamic hierarchical latent value with shrinkage and explicit variance",
            "team_state": "exact roster aggregation plus separately regularized team residual",
            "metric_weights": "fit inside development folds; no hand-assigned performance weights",
        },
        "evaluation": [
            "chronological whole-series folds",
            "paired identical-row comparison with current ratings",
            "calibration and proper scores",
            "regional and patch transfer",
            "roster-change and tournament-boundary slices",
            "missingness, censoring, sparse support, and side-swap checks",
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
    "FutureValueSourceError",
    "assert_pregame_feature_names",
    "bind_accepted_future_value_source",
    "build_strict_prior_player_form",
    "future_value_model_contract",
    "load_accepted_future_value_source",
    "team_value_difference",
    "write_source_receipt",
]
