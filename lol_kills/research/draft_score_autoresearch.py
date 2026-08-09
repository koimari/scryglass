"""Autoresearch harness for the deterministic Scryglass Draft Score.

The July 2026 Leaguepedia ledger is a retrospective, result-blind evaluation
set.  This harness never opens outcome data until it has already been captured
in the sealed ledger, and it keeps the last chronological holdout untouched
while candidates are selected.

It is intentionally an evaluation/search harness rather than an automatic
production promotion mechanism.  A candidate can improve this study while
still failing future-patch, league, or strict pre-event validation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lol_kills.etl.aliases import normalize_team


SCHEMA_VERSION = "scryglass:draft-score-autoresearch:v1"


def sigmoid(value: float) -> float:
    if value >= 35:
        return 1.0
    if value <= -35:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def logit(probability: float) -> float:
    p = min(1 - 1e-9, max(1e-9, probability))
    return math.log(p / (1 - p))


def pct_logit(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)) or value <= 0 or value >= 100:
        return None
    return logit(float(value) / 100.0)


def _rfc(value: str) -> str:
    return value


def _team_key(value: str) -> str:
    normalized = normalize_team(str(value or "").strip())
    normalized = re.sub(r"\s+\([^)]*\)$", "", normalized)
    return normalized.casefold()


def _series_key(fixture_id: str) -> str:
    match = re.match(r"^(.*)_\d+$", fixture_id)
    return match.group(1) if match else fixture_id


def _outcome_y(row: Mapping[str, Any]) -> int:
    return 1 if row["outcome"]["winner_side"] == "blue" else 0


def _components(row: Mapping[str, Any]) -> dict[str, float]:
    drivers = row["score"]["output"]["draft_score"]["actual_blue"]["drivers"]
    return {
        "draft_logit": logit(row["score"]["output"]["draft_score"]["actual_blue"]["blue_pct"] / 100.0),
        "base": float(drivers.get("base", 0.0)),
        "synergy": float(drivers.get("synergy", 0.0)),
        "counter": float(drivers.get("counter", 0.0)),
        "lane": float(drivers.get("same_role", 0.0)),
        "player": float(drivers.get("player_comfort", 0.0)),
        "total": float(drivers.get("total", 0.0)),
        "blue_side": 1.0,
        "player_context": 1.0
        if row["score"]["output"].get("player_context_policy", {}).get("status") == "applied"
        else 0.0,
    }


def _existing_context(row: Mapping[str, Any]) -> dict[str, float]:
    output = row["score"]["output"]
    result: dict[str, float] = {}
    winning = output.get("winning_expectation")
    composite = output.get("composite")
    winning_logit = pct_logit(winning.get("blue_pct")) if isinstance(winning, Mapping) else None
    composite_logit = pct_logit(composite.get("blue_pct")) if isinstance(composite, Mapping) else None
    result["strength_logit"] = winning_logit or 0.0
    result["composite_logit"] = composite_logit or 0.0
    result["strength_available"] = 1.0 if winning_logit is not None else 0.0
    result["composite_available"] = 1.0 if composite_logit is not None else 0.0
    return result


def _static_team_context(row: Mapping[str, Any], context: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict[str, float]:
    ratings = {
        _team_key(str(team.get("team", ""))): float(team.get("rating"))
        for team in context.get("teams", [])
        if isinstance(team, Mapping) and isinstance(team.get("rating"), (int, float))
    }
    blue = ratings.get(_team_key(row["pregame"]["blue"]["team"]))
    red = ratings.get(_team_key(row["pregame"]["red"]["team"]))
    calibration = runtime.get("elo_calibration", {}).get("team", {})
    if blue is None or red is None:
        return {"team_context_logit": 0.0, "team_context_available": 0.0}
    intercept = float(calibration.get("intercept", 0.0))
    coefficient = float(calibration.get("coef", 0.0))
    return {
        "team_context_logit": intercept + coefficient * ((blue - red) / 400.0),
        "team_context_available": 1.0,
    }


def _categorical_features(
    row: Mapping[str, Any],
    *,
    pairs: bool = False,
    cross: bool = False,
    identity_kind: str = "champions",
) -> dict[str, float]:
    pregame = row["pregame"]
    features: dict[str, float] = {"blue_side": 1.0}
    roles = ["top", "jungle", "mid", "bot", "support"]
    blue = pregame["blue"]["picks"]
    red = pregame["red"]["picks"]
    if identity_kind in {"champions", "champions_teams_players"}:
        for index, role in enumerate(roles):
            if index < len(blue):
                features[f"champ:{role}:{blue[index]}"] = features.get(f"champ:{role}:{blue[index]}", 0.0) + 1.0
            if index < len(red):
                features[f"champ:{role}:{red[index]}"] = features.get(f"champ:{role}:{red[index]}", 0.0) - 1.0
    if pairs and identity_kind in {"champions", "champions_teams_players"}:
        for side, picks, sign in (("blue", blue, 1.0), ("red", red, -1.0)):
            for first_index in range(len(picks)):
                for second_index in range(first_index + 1, len(picks)):
                    key = f"ally:{min(picks[first_index], picks[second_index])}|{max(picks[first_index], picks[second_index])}"
                    features[key] = features.get(key, 0.0) + sign
    if cross and identity_kind in {"champions", "champions_teams_players"}:
        for blue_champion in blue:
            for red_champion in red:
                key = f"cross:{blue_champion}>{red_champion}"
                features[key] = features.get(key, 0.0) + 1.0
    if identity_kind in {"teams", "teams_players", "champions_teams_players", "teams_players_matchup"}:
        blue_team = _team_key(str(pregame["blue"]["team"]))
        red_team = _team_key(str(pregame["red"]["team"]))
        features[f"team:blue:{blue_team}"] = 1.0
        features[f"team:red:{red_team}"] = -1.0
    if identity_kind in {"players", "teams_players", "champions_teams_players", "teams_players_matchup"}:
        for side, sign in (("blue", 1.0), ("red", -1.0)):
            for item in pregame[side]["players"]:
                player = str(item.get("player", "")).casefold()
                role = str(item.get("role", "")).casefold()
                if player:
                    features[f"player:{role}:{player}"] = features.get(f"player:{role}:{player}", 0.0) + sign
    if identity_kind == "teams_players_matchup":
        blue_team = _team_key(str(pregame["blue"]["team"]))
        red_team = _team_key(str(pregame["red"]["team"]))
        features[f"matchup:{blue_team}>{red_team}"] = 1.0
    return features


def _champion_features(row: Mapping[str, Any], *, pairs: bool = False, cross: bool = False) -> dict[str, float]:
    return _categorical_features(row, pairs=pairs, cross=cross, identity_kind="champions")


def _online_features(
    rows: list[Mapping[str, Any]],
    prior_rows: list[Mapping[str, Any]] | None = None,
    update_k: float = 0.20,
) -> list[dict[str, float]]:
    """Build strictly pre-map global/team/lineup form features.

    Rows with identical timestamps are scored before any row in that timestamp
    updates the ratings, preventing concurrent maps from seeing one another's
    outcomes.
    """

    team_rating: dict[str, float] = defaultdict(float)
    league_team_rating: dict[tuple[str, str], float] = defaultdict(float)
    player_rating: dict[str, float] = defaultdict(float)
    team_wins: dict[str, int] = defaultdict(int)
    team_games: dict[str, int] = defaultdict(int)
    series_state: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"wins": defaultdict(int), "last": None}
    )
    output: list[dict[str, float]] = []

    def apply_update(
        blue_team: str,
        red_team: str,
        league: str,
        y: float,
        blue_players: list[str],
        red_players: list[str],
        series_id: str,
    ) -> None:
        expected = sigmoid(team_rating[blue_team] - team_rating[red_team])
        delta = update_k * (y - expected)
        team_rating[blue_team] += delta
        team_rating[red_team] -= delta
        expected_league = sigmoid(
            league_team_rating[(league, blue_team)] - league_team_rating[(league, red_team)]
        )
        delta_league = update_k * (y - expected_league)
        league_team_rating[(league, blue_team)] += delta_league
        league_team_rating[(league, red_team)] -= delta_league
        player_delta = update_k * (y - 0.5)
        for player in blue_players:
            player_rating[player] += player_delta
        for player in red_players:
            player_rating[player] -= player_delta
        team_wins[blue_team] += int(y)
        team_wins[red_team] += int(1 - y)
        team_games[blue_team] += 1
        team_games[red_team] += 1
        series = series_state[series_id]
        winner_team = blue_team if y else red_team
        series["wins"][winner_team] += 1
        series["last"] = winner_team

    for prior in sorted(prior_rows or [], key=lambda item: (str(item.get("date", "")), str(item.get("game_id", "")))):
        blue_team = _team_key(str(prior.get("team1", "")))
        red_team = _team_key(str(prior.get("team2", "")))
        winner = _team_key(str(prior.get("winner", "")))
        if not blue_team or not red_team or winner not in {blue_team, red_team}:
            continue
        prior_league = str(prior.get("tournament", "UNKNOWN")).split(" ", 1)[0] or "UNKNOWN"
        apply_update(
            blue_team,
            red_team,
            prior_league,
            1.0 if winner == blue_team else 0.0,
            [],
            [],
            _series_key(str(prior.get("game_id", ""))),
        )
    index = 0
    while index < len(rows):
        timestamp = rows[index]["pregame"]["event_start"]
        end = index + 1
        while end < len(rows) and rows[end]["pregame"]["event_start"] == timestamp:
            end += 1
        for row in rows[index:end]:
            pregame = row["pregame"]
            league = str(pregame["competition"].get("league", "UNKNOWN"))
            blue_team = _team_key(pregame["blue"]["team"])
            red_team = _team_key(pregame["red"]["team"])
            blue_players = [str(item["player"]).casefold() for item in pregame["blue"]["players"]]
            red_players = [str(item["player"]).casefold() for item in pregame["red"]["players"]]
            blue_form = (team_wins[blue_team] + 1.0) / (team_games[blue_team] + 2.0)
            red_form = (team_wins[red_team] + 1.0) / (team_games[red_team] + 2.0)
            series = series_state[_series_key(pregame["fixture_id"])]
            series_win_diff = float(series["wins"][blue_team] - series["wins"][red_team])
            series_last = 0.0
            if series["last"] == blue_team:
                series_last = 1.0
            elif series["last"] == red_team:
                series_last = -1.0
            output.append(
                {
                    "team_elo": team_rating[blue_team] - team_rating[red_team],
                    "league_team_elo": league_team_rating[(league, blue_team)] - league_team_rating[(league, red_team)],
                    "player_elo": (
                        sum(player_rating[player] for player in blue_players) / max(len(blue_players), 1)
                        - sum(player_rating[player] for player in red_players) / max(len(red_players), 1)
                    ),
                    "team_form": logit(blue_form) - logit(red_form),
                    "prior_games": float(min(team_games[blue_team], team_games[red_team])),
                    "series_win_diff": series_win_diff,
                    "series_last": series_last,
                    "series_maps_played": float(series["wins"][blue_team] + series["wins"][red_team]),
                }
            )
        # The update constants are fixed features, not search parameters.  The
        # candidate loop can decide whether these strictly pre-map signals help.
        for row in rows[index:end]:
            pregame = row["pregame"]
            league = str(pregame["competition"].get("league", "UNKNOWN"))
            blue_team = _team_key(pregame["blue"]["team"])
            red_team = _team_key(pregame["red"]["team"])
            y = float(_outcome_y(row))
            blue_players = [str(item["player"]).casefold() for item in pregame["blue"]["players"]]
            red_players = [str(item["player"]).casefold() for item in pregame["red"]["players"]]
            apply_update(
                blue_team,
                red_team,
                league,
                y,
                blue_players,
                red_players,
                _series_key(pregame["fixture_id"]),
            )
        index = end
    return output


@dataclass
class Record:
    row: Mapping[str, Any]
    y: int
    component: dict[str, float]
    context: dict[str, float]
    online: dict[str, float]
    categorical: dict[str, float]


def load_records(path: Path, prior_path: Path | None = None, online_k: float = 0.20) -> list[Record]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda row: (row["pregame"]["event_start"], row["pregame"]["fixture_id"]))
    prior_rows = []
    if prior_path is not None:
        if not prior_path.exists():
            raise FileNotFoundError(f"prior rows file does not exist: {prior_path}")
        prior_rows = [json.loads(line) for line in prior_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    online = _online_features(rows, prior_rows, update_k=online_k)
    context_path = Path("apps/scryglass/data/draft/context.json")
    runtime_path = Path("apps/scryglass/data/draft/runtime.json")
    static_context = json.loads(context_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    records: list[Record] = []
    for row, online_features in zip(rows, online):
        context_features = _existing_context(row)
        context_features.update(_static_team_context(row, static_context, runtime))
        records.append(
            Record(
                row=row,
                y=_outcome_y(row),
                component=_components(row),
                context=context_features,
                online=online_features,
                categorical=_champion_features(row),
            )
        )
    return records


def metrics(y: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    prediction = probability >= threshold
    accuracy = float(np.mean(prediction == y)) if len(y) else float("nan")
    clipped = np.clip(probability, 1e-9, 1 - 1e-9)
    logloss = float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))) if len(y) else float("nan")
    brier = float(np.mean((clipped - y) ** 2)) if len(y) else float("nan")
    return {
        "n": int(len(y)),
        "correct": int(np.sum(prediction == y)),
        "accuracy_pct": round(100 * accuracy, 4),
        "logloss": round(logloss, 6),
        "brier": round(brier, 6),
        "threshold": threshold,
    }


def fit_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.linspace(0.30, 0.70, 81)
    scores = [(float(np.mean((probability >= threshold) == y)), -abs(threshold - 0.5), threshold) for threshold in candidates]
    return max(scores)[2]


def _predict_probability(model: Any, features: Any) -> np.ndarray:
    """Predict without allowing sparse/high-C numerical noise into the run."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            probability = np.asarray(model.predict_proba(features)[:, 1], dtype=float)
    if not np.all(np.isfinite(probability)):
        raise ValueError("candidate produced non-finite probabilities")
    return np.clip(probability, 1e-9, 1 - 1e-9)


def feature_vector(record: Record, name: str) -> dict[str, float]:
    component = record.component
    context = record.context
    online = record.online
    if name == "pure_components":
        return {key: component[key] for key in ("base", "synergy", "counter", "lane", "blue_side")}
    if name == "draft_logit_plus_online":
        return {"draft_logit": component["draft_logit"], "blue_side": component["blue_side"], **online}
    if name == "draft_logit_plus_online_league":
        return {
            "draft_logit": component["draft_logit"],
            "blue_side": component["blue_side"],
            **online,
            f"league:{record.row['pregame']['competition'].get('league', 'UNKNOWN')}": 1.0,
        }
    if name == "online_only":
        return {key: online[key] for key in ("team_elo", "league_team_elo", "player_elo", "team_form", "prior_games", "series_win_diff", "series_last", "series_maps_played")}
    if name == "pure_components_plus_player":
        return {key: component[key] for key in ("base", "synergy", "counter", "lane", "player", "blue_side", "player_context")}
    if name == "pure_components_plus_league":
        return {
            **{key: component[key] for key in ("base", "synergy", "counter", "lane", "player", "blue_side", "player_context")},
            f"league:{record.row['pregame']['competition'].get('league', 'UNKNOWN')}": 1.0,
        }
    if name == "pure_components_plus_existing_strength":
        return {
            **{key: component[key] for key in ("base", "synergy", "counter", "lane", "player", "blue_side", "player_context")},
            **context,
        }
    if name == "pure_components_plus_team_context":
        return {
            **{key: component[key] for key in ("base", "synergy", "counter", "lane", "player", "blue_side", "player_context")},
            **{key: context[key] for key in ("team_context_logit", "team_context_available")},
        }
    if name == "pure_components_plus_online":
        return {
            **{key: component[key] for key in ("base", "synergy", "counter", "lane", "player", "blue_side", "player_context")},
            **online,
        }
    if name == "pure_components_plus_online_league":
        return {
            **{key: component[key] for key in ("base", "synergy", "counter", "lane", "player", "blue_side", "player_context")},
            **online,
            f"league:{record.row['pregame']['competition'].get('league', 'UNKNOWN')}": 1.0,
        }
    if name == "pure_components_plus_series":
        return {
            **{key: component[key] for key in ("base", "synergy", "counter", "lane", "player", "blue_side", "player_context")},
            **{key: online[key] for key in ("series_win_diff", "series_last", "series_maps_played")},
        }
    if name == "all_numeric":
        return {**component, **context, **online}
    raise KeyError(name)


def fit_numeric(records: list[Record], train_indices: list[int], feature_name: str, c_value: float) -> tuple[Pipeline, float, list[str]]:
    names = sorted({key for index in train_indices for key in feature_vector(records[index], feature_name)})
    x_train = np.array([[feature_vector(records[index], feature_name).get(name, 0.0) for name in names] for index in train_indices])
    y_train = np.array([records[index].y for index in train_indices])
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(C=c_value, max_iter=4000, solver="lbfgs")),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(x_train, y_train)
    train_probability = _predict_probability(model, x_train)
    threshold = fit_threshold(y_train, train_probability)
    return model, threshold, names


def predict_numeric(model: Pipeline, records: list[Record], indices: list[int], feature_name: str, names: list[str]) -> np.ndarray:
    x = np.array([[feature_vector(records[index], feature_name).get(name, 0.0) for name in names] for index in indices])
    return _predict_probability(model, x)


def fit_categorical(
    records: list[Record],
    train_indices: list[int],
    *,
    pairs: bool,
    cross: bool,
    identity_kind: str = "champions",
    c_value: float,
) -> tuple[Pipeline, DictVectorizer, float]:
    dictionaries = [_categorical_features(records[index].row, pairs=pairs, cross=cross, identity_kind=identity_kind) for index in train_indices]
    y_train = np.array([records[index].y for index in train_indices])
    vectorizer = DictVectorizer()
    x_train = vectorizer.fit_transform(dictionaries)
    model = Pipeline(
        [
            ("scale", StandardScaler(with_mean=False)),
            ("logistic", LogisticRegression(C=c_value, max_iter=4000, solver="liblinear")),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(x_train, y_train)
    train_probability = _predict_probability(model, x_train)
    threshold = fit_threshold(y_train, train_probability)
    return model, vectorizer, threshold


def predict_categorical(
    model: Pipeline,
    vectorizer: DictVectorizer,
    records: list[Record],
    indices: list[int],
    *,
    pairs: bool,
    cross: bool,
    identity_kind: str = "champions",
) -> np.ndarray:
    dictionaries = [_categorical_features(records[index].row, pairs=pairs, cross=cross, identity_kind=identity_kind) for index in indices]
    return _predict_probability(model, vectorizer.transform(dictionaries))


def baseline_probability(record: Record, name: str) -> float:
    component = record.component
    context = record.context
    if name == "draft_score":
        return sigmoid(component["draft_logit"])
    if name == "draft_total_logit":
        return sigmoid(component["total"])
    if name == "existing_composite_fallback":
        if context["composite_available"]:
            return sigmoid(context["composite_logit"])
        return sigmoid(component["draft_logit"])
    if name == "existing_strength_fallback":
        if context["strength_available"]:
            return sigmoid(context["strength_logit"])
        return sigmoid(component["draft_logit"])
    if name == "existing_team_context_fallback":
        if context["team_context_available"]:
            return sigmoid(context["team_context_logit"])
        return sigmoid(component["draft_logit"])
    raise KeyError(name)


def evaluate_baseline(records: list[Record], indices: list[int], name: str) -> tuple[dict[str, float], np.ndarray]:
    y = np.array([records[index].y for index in indices])
    probability = np.array([baseline_probability(records[index], name) for index in indices])
    return metrics(y, probability), probability


def run_search(records: list[Record], output_dir: Path) -> dict[str, Any]:
    n = len(records)
    train_end = int(n * 0.60)
    validation_end = int(n * 0.80)
    train = list(range(0, train_end))
    validation = list(range(train_end, validation_end))
    final = list(range(validation_end, n))
    candidates: list[dict[str, Any]] = []

    def add_baseline(name: str, estimand: str) -> None:
        train_metrics, _ = evaluate_baseline(records, train, name)
        validation_metrics, _ = evaluate_baseline(records, validation, name)
        candidates.append({"name": name, "family": "baseline", "estimand": estimand, "train": train_metrics, "validation": validation_metrics})

    add_baseline("draft_score", "pure draft composition plus fixed blue-side convention")
    add_baseline("draft_total_logit", "unshrunk signed component edge")
    add_baseline("existing_composite_fallback", "roster/team composite where available, otherwise draft score")
    add_baseline("existing_strength_fallback", "existing roster/team strength where available, otherwise draft score")
    add_baseline("existing_team_context_fallback", "fixed team context where available, otherwise draft score")

    for feature_name in ("pure_components", "draft_logit_plus_online", "draft_logit_plus_online_league", "online_only", "pure_components_plus_player", "pure_components_plus_league", "pure_components_plus_existing_strength", "pure_components_plus_team_context", "pure_components_plus_series", "pure_components_plus_online", "pure_components_plus_online_league", "all_numeric"):
        for c_value in (0.01, 0.1, 1.0, 10.0):
            model, threshold, names = fit_numeric(records, train, feature_name, c_value)
            y_train = np.array([records[index].y for index in train])
            y_validation = np.array([records[index].y for index in validation])
            p_train = predict_numeric(model, records, train, feature_name, names)
            p_validation = predict_numeric(model, records, validation, feature_name, names)
            candidates.append(
                {
                    "name": f"numeric:{feature_name}:C={c_value}",
                    "family": "numeric_logistic",
                    "estimand": "pure draft composition" if "existing" not in feature_name and "online" not in feature_name and feature_name != "all_numeric" else "contextual or online augmentation",
                    "hyperparameters": {"features": feature_name, "C": c_value, "threshold": threshold, "feature_names": names},
                    "train": metrics(y_train, p_train, threshold),
                    "validation": metrics(y_validation, p_validation, threshold),
                }
            )

    for identity_kind in ("champions", "teams", "players", "teams_players", "champions_teams_players", "teams_players_matchup"):
        pair_modes = ((False, False), (True, False), (False, True), (True, True)) if identity_kind in {"champions", "champions_teams_players"} else ((False, False),)
        for pairs, cross in pair_modes:
            for c_value in (0.001, 0.01, 0.1, 1.0):
                model, vectorizer, threshold = fit_categorical(
                    records,
                    train,
                    pairs=pairs,
                    cross=cross,
                    identity_kind=identity_kind,
                    c_value=c_value,
                )
                y_train = np.array([records[index].y for index in train])
                y_validation = np.array([records[index].y for index in validation])
                p_train = predict_categorical(
                    model,
                    vectorizer,
                    records,
                    train,
                    pairs=pairs,
                    cross=cross,
                    identity_kind=identity_kind,
                )
                p_validation = predict_categorical(
                    model,
                    vectorizer,
                    records,
                    validation,
                    pairs=pairs,
                    cross=cross,
                    identity_kind=identity_kind,
                )
                candidates.append(
                    {
                        "name": f"categorical:{identity_kind}:pairs={pairs}:cross={cross}:C={c_value}",
                        "family": "categorical_logistic",
                        "estimand": (
                            "pure draft composition fit from held-in champions and interactions"
                            if identity_kind in {"champions", "champions_teams_players"}
                            else "contextual winner prediction fit from pre-map team/player identities"
                        ),
                        "hyperparameters": {
                            "identity_kind": identity_kind,
                            "pairs": pairs,
                            "cross": cross,
                            "C": c_value,
                            "threshold": threshold,
                            "features": len(vectorizer.feature_names_),
                        },
                        "train": metrics(y_train, p_train, threshold),
                        "validation": metrics(y_validation, p_validation, threshold),
                    }
                )

    # Choose by validation accuracy, then validation log loss, then simpler
    # candidate name for deterministic tie-breaking.  The final holdout is
    # intentionally not consulted here.
    candidates.sort(key=lambda candidate: (-candidate["validation"]["accuracy_pct"], candidate["validation"]["logloss"], candidate["name"]))
    selected = candidates[0]

    # Refit the selected model on train + validation only, then evaluate the
    # untouched final block.  This supports final reporting but never feeds
    # the final labels back into candidate selection.
    final_result: dict[str, Any] = {"name": selected["name"], "validation": selected["validation"]}
    selected_name = selected["name"]
    if selected_name in {"draft_score", "draft_total_logit", "existing_composite_fallback", "existing_strength_fallback", "existing_team_context_fallback"}:
        final_metrics, _ = evaluate_baseline(records, final, selected_name)
        refit_metrics, _ = evaluate_baseline(records, train + validation, selected_name)
    elif selected["family"] == "numeric_logistic":
        feature_name = selected["hyperparameters"]["features"]
        c_value = selected["hyperparameters"]["C"]
        model, threshold, names = fit_numeric(records, train + validation, feature_name, c_value)
        refit_y = np.array([records[index].y for index in train + validation])
        final_y = np.array([records[index].y for index in final])
        refit_metrics = metrics(refit_y, predict_numeric(model, records, train + validation, feature_name, names), threshold)
        final_metrics = metrics(final_y, predict_numeric(model, records, final, feature_name, names), threshold)
    else:
        hp = selected["hyperparameters"]
        model, vectorizer, threshold = fit_categorical(
            records,
            train + validation,
            pairs=hp["pairs"],
            cross=hp["cross"],
            identity_kind=hp.get("identity_kind", "champions"),
            c_value=hp["C"],
        )
        refit_y = np.array([records[index].y for index in train + validation])
        final_y = np.array([records[index].y for index in final])
        refit_metrics = metrics(
            refit_y,
            predict_categorical(
                model,
                vectorizer,
                records,
                train + validation,
                pairs=hp["pairs"],
                cross=hp["cross"],
                identity_kind=hp.get("identity_kind", "champions"),
            ),
            threshold,
        )
        final_metrics = metrics(
            final_y,
            predict_categorical(
                model,
                vectorizer,
                records,
                final,
                pairs=hp["pairs"],
                cross=hp["cross"],
                identity_kind=hp.get("identity_kind", "champions"),
            ),
            threshold,
        )
    final_result["refit_train_validation"] = refit_metrics
    final_result["final_holdout"] = final_metrics

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "objective": "maximize chronological held-out map-winner accuracy while preserving pure-draft versus context distinction",
        "dataset": {
            "maps": n,
            "first_event_start": records[0].row["pregame"]["event_start"],
            "last_event_start": records[-1].row["pregame"]["event_start"],
            "train_maps": len(train),
            "validation_maps": len(validation),
            "final_holdout_maps": len(final),
        },
        "search": {
            "candidate_count": len(candidates),
            "selection_rule": "validation accuracy, then validation log loss, then name; final holdout untouched",
            "target_accuracy_pct": 80.0,
        },
        "baseline": next(candidate for candidate in candidates if candidate["name"] == "draft_score"),
        "selected": selected,
        "final": final_result,
        "top_20": candidates[:20],
    }
    (output_dir / "autoresearch-results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "autoresearch-journal.jsonl").open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate, ensure_ascii=False, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prior-rows", type=Path)
    parser.add_argument("--online-k", type=float, default=0.20)
    args = parser.parse_args()
    result = run_search(load_records(args.ledger, args.prior_rows, online_k=args.online_k), args.output_dir)
    print(json.dumps({"candidate_count": result["search"]["candidate_count"], "selected": result["selected"], "final": result["final"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
