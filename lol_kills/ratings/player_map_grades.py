"""Descriptive per-map player grades with pre-game comparison baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from lol_kills.etl.source_keys import canonical_source_game_key


GRADE_CONTRACT = "scryglass:player-map-grade:v2"
ROLE_ALIASES = {
    "top": "top",
    "jng": "jungle",
    "jungle": "jungle",
    "mid": "mid",
    "bot": "bot",
    "adc": "bot",
    "sup": "support",
    "support": "support",
}
CORE_INPUTS = (
    "kills",
    "deaths",
    "assists",
    "teamkills",
    "gamelength",
    "dpm",
    "damageshare",
    "totalgold",
    "cspm",
    "wpm",
    "wcpm",
)
METRICS = (
    "kill_participation",
    "survival",
    "damage_per_minute",
    "gold_per_minute",
    "damage_share",
    "creep_score_per_minute",
    "wards_per_minute",
    "ward_clears_per_minute",
)


def _historical_z(
    frame: pd.DataFrame,
    metric: str,
    groups: list[str],
    *,
    minimum: int,
) -> tuple[pd.Series, pd.Series]:
    """Return a z-score against prior calendar days and its sample count."""

    keys = [*groups, "_day"]
    source = frame[keys + [metric]].copy()
    source["_square"] = source[metric] ** 2
    daily = source.groupby(keys, dropna=False, sort=False).agg(
        _sum=(metric, "sum"),
        _square_sum=("_square", "sum"),
        _count=(metric, "count"),
    ).reset_index()
    daily = daily.sort_values([*groups, "_day"], kind="stable")
    grouped = daily.groupby(groups, dropna=False, sort=False)
    daily["_prior_sum"] = grouped["_sum"].cumsum() - daily["_sum"]
    daily["_prior_square_sum"] = grouped["_square_sum"].cumsum() - daily["_square_sum"]
    daily["_prior_count"] = grouped["_count"].cumsum() - daily["_count"]
    count = daily["_prior_count"].astype(float)
    mean = daily["_prior_sum"] / count.where(count.gt(0))
    variance = daily["_prior_square_sum"] / count.where(count.gt(0)) - mean**2
    daily["_mean"] = mean
    daily["_std"] = np.sqrt(variance.clip(lower=0))
    lookup = daily.set_index(keys)[["_mean", "_std", "_prior_count"]]
    row_keys = pd.MultiIndex.from_frame(frame[keys])
    stats = lookup.reindex(row_keys)
    valid = stats["_prior_count"].ge(minimum) & stats["_std"].gt(1e-9)
    safe_std = stats["_std"].where(stats["_std"].gt(1e-9))
    z = (frame[metric].to_numpy() - stats["_mean"].to_numpy()) / safe_std.to_numpy()
    return pd.Series(z, index=frame.index).where(valid.to_numpy()), pd.Series(
        stats["_prior_count"].to_numpy(), index=frame.index
    ).where(valid.to_numpy())


def _grade(score: float) -> str:
    if score >= 90:
        return "A+"
    if score >= 75:
        return "A"
    if score >= 55:
        return "B"
    if score >= 35:
        return "C"
    if score >= 15:
        return "D"
    return "F"


def compute_player_map_grades(players: pd.DataFrame) -> pd.DataFrame:
    """Compute one fail-closed descriptive grade row per player and map.

    The game result is deliberately unused. Historical comparisons exclude the
    current calendar day, which prevents a map or its opponent from entering a
    pre-game baseline.
    """

    identity = ["game_uid", "gameid", "date", "league", "competition_tier", "side", "position", "playername"]
    required = {*identity[2:], *CORE_INPUTS}
    if players is None or players.empty or not required.issubset(players.columns):
        return pd.DataFrame(columns=["game_id", "player", "grade_status", "grade_reason"])

    frame = players[[column for column in (*identity, *CORE_INPUTS) if column in players.columns]].copy()
    fallback = frame["gameid"] if "gameid" in frame.columns else None
    source = frame["game_uid"] if "game_uid" in frame.columns else fallback
    if source is None:
        return pd.DataFrame(columns=["game_id", "player", "grade_status", "grade_reason"])
    frame["game_id"] = [
        canonical_source_game_key(value, fallback.loc[index] if fallback is not None else None)
        for index, value in source.items()
    ]
    frame["player"] = frame["playername"].astype("string").str.strip()
    frame["role"] = frame["position"].astype(str).str.casefold().map(ROLE_ALIASES)
    frame["side"] = frame["side"].astype(str).str.title()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame["_day"] = frame["date"].dt.floor("D")
    frame["league"] = frame["league"].astype("string").fillna("UNKNOWN")
    frame["competition_tier"] = frame["competition_tier"].astype("string").fillna("other")
    for column in CORE_INPUTS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[
        frame["game_id"].astype(str).str.strip().ne("")
        & frame["player"].notna()
        & frame["player"].ne("")
        & frame["role"].notna()
        & frame["side"].isin({"Blue", "Red"})
        & frame["date"].notna()
    ].sort_values(["date", "game_id", "side", "role"], kind="stable").reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame(columns=["game_id", "player", "grade_status", "grade_reason"])

    duration = frame["gamelength"].where(frame["gamelength"].gt(0))
    team_kills = frame["teamkills"].where(frame["teamkills"].gt(0), 1)
    frame["kill_participation"] = ((frame["kills"] + frame["assists"]) / team_kills).clip(0, 1.5)
    frame["survival"] = -(frame["deaths"] * 1800 / duration)
    frame["damage_per_minute"] = frame["dpm"]
    frame["gold_per_minute"] = frame["totalgold"] * 60 / duration
    frame["damage_share"] = frame["damageshare"]
    frame["creep_score_per_minute"] = frame["cspm"]
    frame["wards_per_minute"] = frame["wpm"]
    frame["ward_clears_per_minute"] = frame["wcpm"]

    metric_z: list[str] = []
    baseline_counts: list[pd.Series] = []
    for metric in METRICS:
        league_z, league_n = _historical_z(frame, metric, ["league", "role"], minimum=40)
        tier_z, tier_n = _historical_z(frame, metric, ["competition_tier", "role"], minimum=80)
        role_z, role_n = _historical_z(frame, metric, ["role"], minimum=120)
        z = league_z.combine_first(tier_z).combine_first(role_z).clip(-3, 3)
        count = league_n.combine_first(tier_n).combine_first(role_n)
        column = f"_z_{metric}"
        frame[column] = z
        metric_z.append(column)
        baseline_counts.append(count)
    frame["_base"] = frame[metric_z].mean(axis=1, skipna=False)
    frame["_baseline_games"] = pd.concat(baseline_counts, axis=1).min(axis=1, skipna=False)

    self_z, self_n = _historical_z(frame, "_base", ["player"], minimum=10)
    frame["_self"] = self_z.clip(-3, 3)
    frame["_self_games"] = self_n
    team_group = frame.groupby(["game_id", "side"], sort=False)["_base"]
    team_mean = team_group.transform("mean")
    team_std = team_group.transform(lambda values: values.std(ddof=0))
    team_count = team_group.transform("count")
    frame["_team"] = ((frame["_base"] - team_mean) / team_std.where(team_std.gt(1e-9))).clip(-3, 3)
    role_group = frame.groupby(["game_id", "role"], sort=False)["_base"]
    role_sum = role_group.transform("sum")
    role_count = role_group.transform("count")
    frame["_opponent"] = ((frame["_base"] - (role_sum - frame["_base"])) / np.sqrt(2)).clip(-3, 3)
    frame["_league_role"] = frame["_base"].clip(-3, 3)
    components = ["_self", "_team", "_opponent", "_league_role"]
    complete = (
        frame[components].notna().all(axis=1)
        & team_count.eq(5)
        & role_count.eq(2)
        & frame[metric_z].notna().all(axis=1)
    )
    frame["_composite"] = frame[components].mean(axis=1).where(complete)
    frame["grade_score"] = (50 + 22 * frame["_composite"]).clip(1, 99).round(1)
    frame["grade"] = frame["grade_score"].map(lambda value: _grade(float(value)) if pd.notna(value) else None)
    frame["grade_status"] = np.where(complete, "available", "unavailable")
    frame["grade_reason"] = np.where(
        frame[metric_z].isna().any(axis=1),
        "Core match statistics or pre-game league-role history are incomplete.",
        np.where(
            frame["_self"].isna(),
            "The player has fewer than 10 prior comparable maps.",
            np.where(
                ~team_count.eq(5) | ~role_count.eq(2),
                "The map does not contain one complete five-role lineup per side.",
                "A comparison component is unavailable.",
            ),
        ),
    )
    frame.loc[complete, "grade_reason"] = None
    output = frame[["game_id", "player", "role", "side", "grade_status", "grade_reason", "grade", "grade_score", "_baseline_games", "_self_games", *components]].copy()
    output.rename(
        columns={
            "_baseline_games": "baseline_games",
            "_self_games": "self_baseline_games",
            "_self": "grade_self",
            "_team": "grade_team",
            "_opponent": "grade_opponent",
            "_league_role": "grade_league_role",
        },
        inplace=True,
    )
    return output


def grade_payload(row: pd.Series | dict[str, Any] | None) -> dict[str, Any]:
    """Serialize one grade row into the public profile contract."""

    if row is None:
        return {"status": "unavailable", "reason": "Grade inputs are unavailable."}
    get = row.get
    if get("grade_status") != "available":
        return {"status": "unavailable", "reason": str(get("grade_reason") or "Grade inputs are unavailable.")}
    return {
        "status": "available",
        "grade": str(get("grade")),
        "score": float(get("grade_score")),
        "baseline_games": int(get("baseline_games")),
        "self_baseline_games": int(get("self_baseline_games")),
        "components": {
            "self": round(float(get("grade_self")), 3),
            "team": round(float(get("grade_team")), 3),
            "opponent": round(float(get("grade_opponent")), 3),
            "league_role": round(float(get("grade_league_role")), 3),
        },
    }
