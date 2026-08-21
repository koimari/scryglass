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
    FORM_METRICS,
    MODEL_FIT_SCHEMA_VERSION,
    RANK_3,
    RatingVariant,
    FutureValueFoldModel,
    FutureValueSourceError,
    _canonical_json_bytes,
    _frame_game_ids,
    _role,
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
        if "date" in frame:
            frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
            if frame["date"].isna().any() or frame["date"].gt(cutoff).any():
                raise FutureValueSnapshotError(f"snapshot {label} has invalid or future dates")
        frame.drop(frame.index[~frame["game_id"].isin(eligible)], inplace=True)

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


def _feature_column(feature: str) -> str | None:
    if feature.startswith("player_form_"):
        return "prior_form_" + feature.removeprefix("player_form_")
    if feature.startswith("rank_3_"):
        return feature
    if feature in QUALITY_FEATURES:
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
    feature_names = tuple(str(value) for value in model.feature_names)
    scales = np.asarray(model.scales, dtype=float)
    coefficients = np.asarray(model.coefficients, dtype=float)
    imputation = np.asarray(model.imputation_values, dtype=float)
    if len(feature_names) != len(scales) or len(scales) != len(coefficients) or len(scales) != len(imputation):
        raise FutureValueSnapshotError("final model parameter dimensions are invalid")
    if not np.isfinite(scales).all() or not np.isfinite(coefficients).all() or not np.isfinite(imputation).all():
        raise FutureValueSnapshotError("final model parameters are non-finite")
    output = work[["player_id", "team_id", "playername", "champion", "role", "side", "date"]].copy()
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
    support_columns = [f"prior_form_{metric}_support" for metric in FORM_METRICS]
    effective_columns = [f"prior_form_{metric}_effective_support" for metric in FORM_METRICS]
    support = work[support_columns].apply(pd.to_numeric, errors="coerce")
    effective_source = effective_columns if set(effective_columns).issubset(work.columns) else support_columns
    effective = work[effective_source].apply(pd.to_numeric, errors="coerce")
    output["minimum_metric_support"] = support.min(axis=1, skipna=True).fillna(0.0)
    output["minimum_effective_support"] = effective.min(axis=1, skipna=True).fillna(0.0)
    output["form_missing_rate"] = work[
        [f"prior_form_{metric}" for metric in FORM_METRICS]
    ].apply(pd.to_numeric, errors="coerce").isna().mean(axis=1)
    output["rank_3_champion_role_atom_support"] = pd.to_numeric(
        work.get("rank_3_champion_role_support", 0), errors="coerce"
    ).fillna(0).astype(int)
    output["uncertainty_proxy"] = 1.0 / np.sqrt(1.0 + output["minimum_effective_support"])
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
) -> tuple[list[dict[str, Any]], list[str]]:
    blockers: list[str] = []
    if current is None:
        return [], [f"current_{identity}_rating_snapshot_missing"]
    if identity not in current.columns:
        return [], [f"current_{identity}_rating_identity_missing"]
    value_column = next((name for name in current_value_candidates if name in current.columns), None)
    if value_column is None:
        return [], [f"current_{identity}_rating_value_missing"]
    frame = current[[identity, value_column]].copy()
    frame[identity] = frame[identity].astype(str)
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    if frame[identity].duplicated().any() or frame[value_column].isna().any():
        return [], [f"current_{identity}_rating_identity_or_value_ambiguous"]
    futures = pd.DataFrame(list(future_rows))
    if futures.empty or identity not in futures.columns:
        return [], [f"future_{identity}_snapshot_missing"]
    futures[identity] = futures[identity].astype(str)
    futures[future_value] = pd.to_numeric(futures[future_value], errors="coerce")
    if futures[identity].duplicated().any() or futures[future_value].isna().any():
        return [], [f"future_{identity}_snapshot_identity_or_value_ambiguous"]
    current["__rank"] = pd.to_numeric(current[value_column], errors="coerce").rank(
        method="min", ascending=False
    )
    current_rank = dict(zip(current[identity].astype(str), current["__rank"]))
    futures["__rank"] = futures[future_value].rank(method="min", ascending=False)
    output: list[dict[str, Any]] = []
    for row in futures.to_dict("records"):
        key = str(row[identity])
        if key not in current_rank:
            continue
        output.append(
            {
                identity: key,
                "current_rank": int(current_rank[key]),
                "future_rank": int(row["__rank"]),
                "rank_delta": int(current_rank[key] - row["__rank"]),
                "current_value": float(
                    current.loc[current[identity].astype(str).eq(key), value_column].iloc[0]
                ),
                "future_value": float(row[future_value]),
            }
        )
    return output, blockers


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
    auth = authorize_final_fit(
        _model_receipt_from(model, model_receipt), source_receipt
    )
    # Validate the source before the fit gate result is returned.  A blocked
    # model must not hide a malformed or future-dated accepted source.
    map_frame, player_frame, team_frame, cutoff = _normalise_source_frames(
        maps, players, teams, source_receipt, as_of
    )
    if not auth.authorized:
        return _blocked_result(source_receipt, auth)
    if model is None:
        return _blocked_result(source_receipt, auth, extra_blockers=("final_fit_model_object_missing",))

    form = _latest_player_form(map_frame, player_frame, baseline_cache=baseline_cache)
    form = form[form["date"].le(cutoff)].copy()
    latest = _latest_rows(form, key="player_id", label="player")
    if len(latest) != form["player_id"].nunique():
        raise FutureValueSnapshotError("latest player snapshot is incomplete")
    contributions = _player_contributions(model, latest)
    player_rows: list[dict[str, Any]] = []
    for row in contributions.to_dict("records"):
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
                    "role_normalized_form_logit": row["role_normalized_form_logit"],
                    "rank_3_player_atom_logit": row["rank_3_player_atom_logit"],
                    "champion_role_atom_logit": row["champion_role_atom_logit"],
                    **{
                        f"rank_3_player_atom_{index}_logit": row[
                            f"rank_3_player_atom_{index}_logit"
                        ]
                        for index in range(1, RANK_3 + 1)
                    },
                    **{
                        f"rank_3_champion_role_atom_{index}_logit": row[
                            f"rank_3_champion_role_atom_{index}_logit"
                        ]
                        for index in range(1, RANK_3 + 1)
                    },
                    "future_player_value_logit": row["role_normalized_player_value_logit"],
                    "future_player_value_with_champion_logit": row[
                        "future_player_value_with_champion_logit"
                    ],
                    "minimum_metric_support": row["minimum_metric_support"],
                    "minimum_effective_support": row["minimum_effective_support"],
                    "rank_3_champion_role_atom_support": row[
                        "rank_3_champion_role_atom_support"
                    ],
                    "uncertainty_proxy": row["uncertainty_proxy"],
                    "model_feature_missing": bool(row["model_feature_missing"]),
                    "champion_dependent_status": row["champion_dependent_status"],
                }
            )
        )

    contribution_frame = contributions.copy()
    contribution_frame["game_id"] = latest["game_id"].to_numpy()
    contribution_frame["team_id"] = latest["team_id"].to_numpy()
    contribution_frame["side"] = latest["side"].to_numpy()
    contribution_frame["role"] = latest["role"].to_numpy()
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
    for team_id, group in form.groupby("team_id", sort=False):
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
        team_rows.append(
            _json_row(
                {
                    "team_id": str(team_id),
                    "side": str(group["side"].iloc[0]),
                    "last_game_id": str(context["last_game_id"].iloc[0]),
                    "last_game_date": context["last_game_date"].iloc[0],
                    "roster_player_count": int(len(group)),
                    "roster_player_ids": sorted(str(value) for value in group["player_id"]),
                    "role_normalized_player_value_logit": player_value,
                    "champion_role_atom_logit": champion_value,
                    "team_context_logit": team_context_value,
                    "future_team_value_logit": (
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
                }
            )
        )

    player_rank_diffs, player_rank_blockers = _rank_diffs(
        player_rows,
        current_player_ratings,
        identity="player_id",
        future_value="future_player_value_logit",
        current_value_candidates=("mu_effective", "mu_total", "rating"),
    )
    team_rank_diffs, team_rank_blockers = _rank_diffs(
        team_rows,
        current_team_ratings,
        identity="team_id",
        future_value="future_team_value_logit",
        current_value_candidates=("mu_effective", "mu_total", "rating"),
    )
    blockers = tuple(sorted(set((*team_blocker, *player_rank_blockers, *team_rank_blockers))))
    payload: dict[str, Any] = {
        "schema_version": SNAPSHOT_RECEIPT_SCHEMA_VERSION,
        "status": "research_only" if not blockers else "research_only_partial",
        "authority": dict(SNAPSHOT_AUTHORITY),
        "source": source,
        "as_of": _utc_text(cutoff),
        "model": {
            "schema_version": MODEL_FIT_SCHEMA_VERSION,
            "variant": str(model_receipt.get("variant")),
            "receipt_sha256": auth.model_receipt_sha256,
        },
        "fit": auth.as_dict(),
        "player_row_count": len(player_rows),
        "team_row_count": len(team_rows),
        "player_rank_diff_count": len(player_rank_diffs),
        "team_rank_diff_count": len(team_rank_diffs),
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
    "write_snapshot_bundle",
]
