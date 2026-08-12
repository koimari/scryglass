"""Leakage-safe composition evidence for completed Oracle's Elixir games.

This module has two jobs:

* evaluate a role-conditioned champion model on chronological holdouts;
* score accepted games with a model fit strictly before each game's date.

The model is a descriptive composition signal. It is separate from the
private Draft Score, team ratings, player ratings, and player game grades.
Private checkpoints contain coefficients. Public records contain only signed
contributions and their prior role-game support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from lol_kills.etl.aliases import normalize_champ, normalize_team
from lol_kills.etl.source_keys import canonical_source_game_key


SCHEMA_VERSION = "scryglass:composition-signal:v1"
MODEL_VERSION = "composition-signal-v1"
REGULARIZATION_C = 0.03
MIN_SUPPORT_GAMES = 40
MIN_TRAINING_GAMES = 100
CALIBRATION_SLOPE_TOLERANCE = 0.35
CALIBRATION_INTERCEPT_TOLERANCE = 0.15
ROLES = ("top", "jng", "mid", "bot", "sup")
MODEL_TERMS = (
    "pre_game_team_strength_gap",
    "rating_uncertainty",
    "league",
    "patch",
    "role_conditioned_champion_effects",
)
EXCLUDED_TERMS = (
    "role_pair_interactions",
    "team_draft_history",
)
ROLE_ALIASES = {
    "top": "top",
    "jng": "jng",
    "jungle": "jng",
    "jungler": "jng",
    "mid": "mid",
    "middle": "mid",
    "bot": "bot",
    "adc": "bot",
    "bottom": "bot",
    "sup": "sup",
    "support": "sup",
    "utility": "sup",
}
PUBLIC_STATUS = ("available", "limited", "unavailable")
PUBLIC_EVIDENCE = ("available", "limited", "unavailable")
PUBLIC_PRIVATE_FIELDS = frozenset(
    {
        "coefficients",
        "feature_names",
        "intercept",
        "support",
        "train_games",
        "training_rows",
        "probability",
        "win_probability",
        "odds",
    }
)
NOTE = (
    "A descriptive composition signal from champion and role information "
    "available before the game. Values are model contribution units. A "
    "positive value helps that side's composition. It does not grade player "
    "execution, change team ratings, or provide a betting probability."
)


class CompositionSignalError(RuntimeError):
    """Raised when a composition signal cannot be built safely."""


def _role(value: Any) -> str:
    raw = str(value or "").strip().casefold()
    return ROLE_ALIASES.get(raw, raw[:3])


def _champion(value: Any) -> str:
    return normalize_champ(str(value or "").strip())


def _timestamp(value: Any) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        return parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC")


def _rfc(value: Any) -> str:
    return _timestamp(value).isoformat().replace("+00:00", "Z")


def _day(value: Any) -> pd.Timestamp:
    return _timestamp(value).normalize()


def _number(value: Any, default: float | None = None) -> float | None:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return default
    return float(parsed)


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if bool(pd.isna(value)):
            return default
    except (TypeError, ValueError):
        pass
    return str(value).strip() or default


def _json_number(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), 6)


def _digest(values: Iterable[str]) -> str:
    canonical = sorted({str(value) for value in values if str(value).strip()})
    return hashlib.sha256(("\n".join(canonical) + "\n").encode("utf-8")).hexdigest()


def _patch(value: Any) -> str:
    if value is None or pd.isna(value):
        return "UNKNOWN"
    text = str(value).strip()
    return text or "UNKNOWN"


def _strength_lookup(features: pd.DataFrame | Mapping[str, Mapping[str, Any]] | None) -> dict[str, dict[str, float | None]]:
    if features is None:
        return {}
    if isinstance(features, Mapping):
        output: dict[str, dict[str, float | None]] = {}
        for key, value in features.items():
            if not isinstance(value, Mapping):
                continue
            output[canonical_source_game_key(key)] = {
                "mu_diff": _number(value.get("mu_diff")),
                "sigma_pair": _number(value.get("sigma_pair")),
            }
        return output
    if features.empty:
        return {}
    id_column = next(
        (column for column in ("game_uid", "gameid", "oe_gameid") if column in features.columns),
        None,
    )
    if id_column is None:
        return {}
    output = {}
    for _, row in features.iterrows():
        game_id = canonical_source_game_key(row.get(id_column))
        if not game_id:
            continue
        output[game_id] = {
            "mu_diff": _number(row.get("mu_diff")),
            "sigma_pair": _number(row.get("sigma_pair")),
        }
    return output


def _complete_game_from_group(game_id: str, group: pd.DataFrame, strength: Mapping[str, Any]) -> dict[str, Any] | None:
    if len(group) != 10 or group["_player_key"].nunique() != 10:
        return None
    sides: dict[str, dict[str, dict[str, str]]] = {}
    teams: dict[str, str] = {}
    champions: list[str] = []
    for side in ("Blue", "Red"):
        side_rows = group[group["_side"] == side]
        if len(side_rows) != 5:
            return None
        if side_rows["_role"].nunique() != 5:
            return None
        if side_rows["_team"].nunique() != 1:
            return None
        picks: dict[str, dict[str, str]] = {}
        for role in ROLES:
            hit = side_rows[side_rows["_role"] == role]
            if len(hit) != 1:
                return None
            row = hit.iloc[0]
            champion = str(row.get("_champion") or "")
            player = str(row.get("_player") or "").strip()
            team = str(row.get("_team") or "").strip()
            if not champion or not player or not team:
                return None
            picks[role] = {"champion": champion, "player": player}
            champions.append(champion)
            teams[side] = team
        sides[side] = picks
    if len(set(champions)) != 10 or not teams.get("Blue") or not teams.get("Red"):
        return None
    if teams["Blue"] == teams["Red"]:
        return None
    blue_results = group[group["_side"] == "Blue"]["_result"].dropna().unique()
    red_results = group[group["_side"] == "Red"]["_result"].dropna().unique()
    if (
        len(blue_results) != 1
        or len(red_results) != 1
        or float(blue_results[0]) not in (0.0, 1.0)
        or float(red_results[0]) != 1.0 - float(blue_results[0])
    ):
        return None
    date = group["_date"].max()
    if pd.isna(date):
        return None
    strength_row = dict(strength or {})
    mu_diff = _number(strength_row.get("mu_diff"))
    sigma_pair = _number(strength_row.get("sigma_pair"))
    return {
        "game_uid": game_id,
        "date": pd.Timestamp(date),
        "league": _text(
            group.get("league", pd.Series(["UNKNOWN"])).iloc[0], "UNKNOWN"
        ).upper(),
        "patch": _patch(
            group.get("patch", pd.Series(["UNKNOWN"])).iloc[0]
            if "patch" in group
            else None
        ),
        "blue_team": teams["Blue"],
        "red_team": teams["Red"],
        "y": int(blue_results[0]),
        "blue": sides["Blue"],
        "red": sides["Red"],
        "mu_diff": mu_diff,
        "sigma_pair": sigma_pair,
        "controls_available": mu_diff is not None and sigma_pair is not None,
        "series_id": _text(
            group.get("grid_series_id", pd.Series([""])).iloc[0]
        ),
        "tournament": _text(
            group.get("tournament", pd.Series([""])).iloc[0]
        ),
    }


def build_composition_games(
    players: pd.DataFrame,
    *,
    strength_features: pd.DataFrame | Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build only exact, role-complete, ten-champion games."""

    required = {"playername", "teamname", "side", "position", "result", "date", "champion"}
    if players is None or players.empty or not required.issubset(players.columns):
        return []
    selected_columns = [
        column
        for column in (
            "game_uid",
            "gameid",
            "oe_gameid",
            "date",
            "side",
            "position",
            "playername",
            "teamname",
            "result",
            "champion",
            "league",
            "patch",
            "grid_series_id",
            "tournament",
        )
        if column in players.columns
    ]
    frame = players[selected_columns].copy()
    id_column = next(
        (column for column in ("game_uid", "gameid", "oe_gameid") if column in frame.columns),
        None,
    )
    source_id = frame[id_column] if id_column is not None else None
    if source_id is None:
        return []
    fallback_column = next(
        (column for column in ("gameid", "oe_gameid") if column in frame.columns and column != id_column),
        None,
    )
    fallback = frame[fallback_column] if fallback_column is not None else None
    frame["_game_id"] = [
        canonical_source_game_key(value, fallback.loc[index] if fallback is not None else None)
        for index, value in source_id.items()
    ]
    frame["_date"] = pd.to_datetime(frame["date"], format="mixed", utc=True, errors="coerce")
    frame["_side"] = frame["side"].astype(str).str.title()
    frame["_role"] = frame["position"].map(_role)
    frame["_player"] = frame["playername"].map(_text)
    frame["_player_key"] = frame["_player"].str.casefold()
    frame["_team"] = frame["teamname"].map(lambda value: normalize_team(_text(value)))
    frame["_champion"] = frame["champion"].map(lambda value: _champion(_text(value)))
    frame["_result"] = pd.to_numeric(frame["result"], errors="coerce")
    frame = frame[
        frame["_game_id"].astype(str).str.strip().ne("")
        & frame["_date"].notna()
        & frame["_side"].isin({"Blue", "Red"})
        & frame["_role"].isin(ROLES)
    ].copy()
    if frame.empty:
        return []
    strength = _strength_lookup(strength_features)
    games = []
    for game_id, group in frame.groupby("_game_id", sort=False):
        game = _complete_game_from_group(str(game_id), group, strength.get(str(game_id), {}))
        if game is not None:
            games.append(game)
    return sorted(games, key=lambda game: (game["date"], game["game_uid"]))


def _validate_game(game: Mapping[str, Any]) -> tuple[bool, str]:
    if not isinstance(game, Mapping):
        return False, "game identity is missing"
    blue = game.get("blue")
    red = game.get("red")
    if not isinstance(blue, Mapping) or not isinstance(red, Mapping):
        return False, "both sides are required"
    if set(blue) != set(ROLES) or set(red) != set(ROLES):
        return False, "each side needs all five roles"
    champions: list[str] = []
    for side in (blue, red):
        for role in ROLES:
            pick = side.get(role)
            if not isinstance(pick, Mapping) or not _champion(pick.get("champion")):
                return False, "a champion or role is missing"
            champions.append(_champion(pick.get("champion")))
    if len(set(champions)) != 10:
        return False, "the draft does not contain ten unique champions"
    if not str(game.get("blue_team") or "").strip() or not str(game.get("red_team") or "").strip():
        return False, "team identities are missing"
    if normalize_team(str(game["blue_team"])) == normalize_team(str(game["red_team"])):
        return False, "team identities collide"
    if game.get("controls_available") is not True:
        return False, "pre-game strength controls are missing"
    return True, ""


def _feature_names(games: Sequence[Mapping[str, Any]]) -> list[str]:
    names = {"control|mu_diff", "control|sigma_pair", "control|blue_side"}
    for game in games:
        names.add(f"league|{game.get('league') or 'UNKNOWN'}")
        names.add(f"patch|{game.get('patch') or 'UNKNOWN'}")
        for side, sign in (("blue", 1), ("red", -1)):
            del sign
            for role in ROLES:
                names.add(f"draft|{role}|{_champion(game[side][role].get('champion'))}")
    controls = sorted(name for name in names if name.startswith("control|"))
    context = sorted(name for name in names if name.startswith(("league|", "patch|")))
    draft = sorted(name for name in names if name.startswith("draft|"))
    return controls + context + draft


def _matrix(games: Sequence[Mapping[str, Any]], names: Sequence[str], *, include_draft: bool) -> sparse.csr_matrix:
    columns = {name: index for index, name in enumerate(names)}
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []

    def add(row: int, name: str, value: float) -> None:
        column = columns.get(name)
        if column is None:
            return
        rows.append(row)
        cols.append(column)
        values.append(float(value))

    for row_index, game in enumerate(games):
        add(row_index, "control|mu_diff", float(game.get("mu_diff") or 0.0) / 400.0)
        add(row_index, "control|sigma_pair", float(game.get("sigma_pair") or 0.0) / 120.0)
        add(row_index, "control|blue_side", 1.0)
        add(row_index, f"league|{game.get('league') or 'UNKNOWN'}", 1.0)
        add(row_index, f"patch|{game.get('patch') or 'UNKNOWN'}", 1.0)
        if include_draft:
            for side, sign in (("blue", 1.0), ("red", -1.0)):
                for role in ROLES:
                    champion = _champion(game[side][role].get("champion"))
                    add(row_index, f"draft|{role}|{champion}", sign)
    return sparse.csr_matrix((values, (rows, cols)), shape=(len(games), len(names)), dtype=np.float64)


@dataclass(frozen=True)
class FittedCompositionModel:
    model_version: str
    fit_through: str
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    support: dict[str, int]
    train_games: int
    worker_commit: str | None = None

    def coefficient(self, role: str, champion: str) -> float:
        key = f"draft|{role}|{_champion(champion)}"
        try:
            index = self.feature_names.index(key)
        except ValueError:
            return 0.0
        return float(self.coefficients[index])

    def logit(self, game: Mapping[str, Any], *, include_draft: bool = True) -> float:
        matrix = _matrix([game], self.feature_names, include_draft=include_draft)
        value = self.intercept + matrix @ np.asarray(self.coefficients, dtype=float)
        return float(np.asarray(value).reshape(-1)[0])

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_version": self.model_version,
            "fit_through": self.fit_through,
            "feature_names": list(self.feature_names),
            "coefficients": [_json_number(value) for value in self.coefficients],
            "intercept": _json_number(self.intercept),
            "support": self.support,
            "train_games": self.train_games,
            "worker_commit": self.worker_commit,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "FittedCompositionModel":
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise CompositionSignalError("composition checkpoint schema is not supported")
        return cls(
            model_version=str(payload["model_version"]),
            fit_through=str(payload["fit_through"]),
            feature_names=tuple(str(value) for value in payload["feature_names"]),
            coefficients=tuple(float(value) for value in payload["coefficients"]),
            intercept=float(payload["intercept"]),
            support={str(key): int(value) for key, value in dict(payload["support"]).items()},
            train_games=int(payload["train_games"]),
            worker_commit=str(payload.get("worker_commit") or "") or None,
        )


def _fit_model(
    games: Sequence[Mapping[str, Any]],
    *,
    names: Sequence[str],
    include_draft: bool = True,
    min_training_games: int = MIN_TRAINING_GAMES,
    worker_commit: str | None = None,
) -> FittedCompositionModel | None:
    usable = [game for game in games if game.get("controls_available", False)]
    if len(usable) < min_training_games or len({int(game["y"]) for game in usable}) < 2:
        return None
    model = LogisticRegression(
        C=REGULARIZATION_C,
        solver="liblinear",
        max_iter=2000,
        random_state=461,
    )
    matrix = _matrix(usable, names, include_draft=include_draft)
    outcomes = np.asarray([int(game["y"]) for game in usable], dtype=np.int8)
    model.fit(matrix, outcomes)
    support: dict[str, int] = {}
    if include_draft:
        for game in usable:
            for side in ("blue", "red"):
                for role in ROLES:
                    champion = _champion(game[side][role].get("champion"))
                    key = f"{role}|{champion}"
                    support[key] = support.get(key, 0) + 1
    fit_through = _rfc(max(game["date"] for game in usable))
    return FittedCompositionModel(
        model_version=MODEL_VERSION if include_draft else f"{MODEL_VERSION}:baseline",
        fit_through=fit_through,
        feature_names=tuple(names),
        coefficients=tuple(float(value) for value in model.coef_[0]),
        intercept=float(model.intercept_[0]),
        support=support,
        train_games=len(usable),
        worker_commit=worker_commit,
    )


def _cache_key(
    source_digest: str,
    cutoff: pd.Timestamp,
    *,
    names: Sequence[str],
    worker_commit: str | None,
) -> str:
    material = "|".join(
        (MODEL_VERSION, source_digest, _rfc(cutoff), _digest(names), worker_commit or "")
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _game_fingerprint(game: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "game_uid": str(game.get("game_uid") or ""),
        "date": _rfc(game["date"]),
        "league": str(game.get("league") or ""),
        "patch": str(game.get("patch") or ""),
        "blue_team": str(game.get("blue_team") or ""),
        "red_team": str(game.get("red_team") or ""),
        "y": int(game.get("y", 0)),
        "mu_diff": game.get("mu_diff"),
        "sigma_pair": game.get("sigma_pair"),
        "blue": {
            role: _champion(game["blue"][role].get("champion"))
            for role in ROLES
        },
        "red": {
            role: _champion(game["red"][role].get("champion"))
            for role in ROLES
        },
    }


def _games_digest(games: Sequence[Mapping[str, Any]]) -> str:
    payload = [_game_fingerprint(game) for game in sorted(games, key=lambda item: str(item["game_uid"]))]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _load_or_fit(
    games: Sequence[Mapping[str, Any]],
    cutoff: pd.Timestamp,
    *,
    cache_dir: Path | None,
    min_training_games: int,
    worker_commit: str | None,
) -> tuple[FittedCompositionModel | None, bool]:
    training_games = [game for game in games if _day(game["date"]) < _day(cutoff)]
    training_names = _feature_names(training_games)
    training_digest = _games_digest(training_games)
    path = None
    if cache_dir is not None:
        path = cache_dir / "checkpoints" / f"{_cache_key(training_digest, cutoff, names=training_names, worker_commit=worker_commit)}.json"
        if path.exists():
            try:
                cached = FittedCompositionModel.from_json(
                    json.loads(path.read_text(encoding="utf-8"))
                )
                if cached.worker_commit == worker_commit:
                    return cached, True
            except (OSError, ValueError, KeyError, TypeError, CompositionSignalError):
                pass
    model = _fit_model(
        training_games,
        names=training_names,
        include_draft=True,
        min_training_games=min_training_games,
        worker_commit=worker_commit,
    )
    if model is not None and path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(model.to_json(), separators=(",", ":")), encoding="utf-8")
    return model, False


def _unavailable(game: Mapping[str, Any], reason: str) -> dict[str, Any]:
    picks = []
    for side in ("Blue", "Red"):
        side_data = game.get(side.lower()) if isinstance(game.get(side.lower()), Mapping) else {}
        for role in ROLES:
            pick = side_data.get(role) if isinstance(side_data, Mapping) else {}
            picks.append(
                {
                    "side": side,
                    "role": role,
                    "champion": _champion(pick.get("champion")) if isinstance(pick, Mapping) else "",
                    "contribution": None,
                    "prior_role_games": 0,
                    "evidence_status": "unavailable",
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "unavailable",
        "model_version": MODEL_VERSION,
        "fit_through": None,
        "blue": {"signal": None, "prior_role_games": 0},
        "red": {"signal": None, "prior_role_games": 0},
        "picks": picks,
        "note": NOTE,
        "reason": reason,
    }


def public_signal_for_game(
    game: Mapping[str, Any],
    model: FittedCompositionModel | None,
    *,
    min_support_games: int = MIN_SUPPORT_GAMES,
) -> dict[str, Any]:
    """Build the public-safe evidence object for one complete game."""

    valid, reason = _validate_game(game)
    if not valid:
        return _unavailable(game, reason)
    if model is None:
        return _unavailable(game, "No earlier accepted games support this signal yet.")
    picks: list[dict[str, Any]] = []
    side_signals: dict[str, float] = {"Blue": 0.0, "Red": 0.0}
    side_support: dict[str, int] = {"Blue": 0, "Red": 0}
    limited = False
    for side in ("Blue", "Red"):
        for role in ROLES:
            champion = _champion(game[side.lower()][role].get("champion"))
            support = int(model.support.get(f"{role}|{champion}", 0))
            coefficient = model.coefficient(role, champion)
            supported = support >= min_support_games
            if supported:
                side_signals[side] += coefficient
                side_support[side] += support
            else:
                limited = True
            picks.append(
                {
                    "side": side,
                    "role": role,
                    "champion": champion,
                    "contribution": _json_number(coefficient) if supported else None,
                    "prior_role_games": support,
                    "evidence_status": "available" if supported else "limited",
                }
            )
    status = "limited" if limited else "available"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "model_version": model.model_version,
        "fit_through": model.fit_through,
        "blue": {
            "signal": _json_number(side_signals["Blue"]) if status == "available" else None,
            "prior_role_games": side_support["Blue"],
        },
        "red": {
            "signal": _json_number(side_signals["Red"]) if status == "available" else None,
            "prior_role_games": side_support["Red"],
        },
        "picks": picks,
        "note": NOTE,
    }


def validate_public_signal(
    signal: Mapping[str, Any],
    game: Mapping[str, Any],
    *,
    min_support_games: int = MIN_SUPPORT_GAMES,
) -> None:
    """Validate one public signal against its published ten-player game."""

    if not isinstance(signal, Mapping):
        raise CompositionSignalError("composition signal is not an object")
    leaked = sorted(
        key
        for key in signal
        if str(key) in PUBLIC_PRIVATE_FIELDS
    )
    if leaked:
        raise CompositionSignalError(
            "private composition fields are present: " + ", ".join(leaked)
        )
    required = {
        "schema_version",
        "status",
        "model_version",
        "fit_through",
        "blue",
        "red",
        "picks",
        "note",
    }
    missing = sorted(required.difference(signal))
    if missing:
        raise CompositionSignalError(
            "composition signal is missing: " + ", ".join(missing)
        )
    if signal.get("schema_version") != SCHEMA_VERSION:
        raise CompositionSignalError("composition signal schema is not supported")
    status = str(signal.get("status") or "")
    if status not in PUBLIC_STATUS:
        raise CompositionSignalError("composition signal status is invalid")
    if not str(signal.get("model_version") or "").strip():
        raise CompositionSignalError("composition signal model version is missing")
    if not str(signal.get("note") or "").strip():
        raise CompositionSignalError("composition signal note is missing")

    players = game.get("players") if isinstance(game, Mapping) else None
    if not isinstance(players, list) or len(players) != 10:
        raise CompositionSignalError("published composition game needs ten players")
    expected: dict[tuple[str, str], str] = {}
    champions: list[str] = []
    for player in players:
        if not isinstance(player, Mapping):
            raise CompositionSignalError("published composition player is malformed")
        side = str(player.get("side") or "").strip().title()
        role = _role(player.get("role"))
        champion = _champion(player.get("champion"))
        key = (side, role)
        if side not in {"Blue", "Red"} or role not in ROLES or not champion:
            raise CompositionSignalError("published composition identity is incomplete")
        if key in expected:
            raise CompositionSignalError("published composition has duplicate roles")
        expected[key] = champion
        champions.append(champion)
    if set(expected) != {(side, role) for side in ("Blue", "Red") for role in ROLES}:
        raise CompositionSignalError("published composition does not have two complete sides")
    if len(set(champions)) != 10:
        raise CompositionSignalError("published composition does not have ten unique champions")

    fit_through = signal.get("fit_through")
    try:
        game_date = _timestamp(game.get("date"))
    except (TypeError, ValueError, OverflowError) as error:
        raise CompositionSignalError("published composition game date is invalid") from error
    if pd.isna(game_date):
        raise CompositionSignalError("published composition game date is missing")
    if status == "unavailable":
        if fit_through is not None:
            raise CompositionSignalError("unavailable composition signal has a fit watermark")
    else:
        if fit_through is None:
            raise CompositionSignalError("supported composition signal has no fit watermark")
        try:
            fit_date = _timestamp(fit_through)
            if pd.isna(fit_date) or fit_date >= game_date:
                raise CompositionSignalError(
                    "composition signal fit watermark is not before the game"
                )
        except CompositionSignalError:
            raise
        except (TypeError, ValueError, OverflowError) as error:
            raise CompositionSignalError("composition signal dates are invalid") from error

    side_payloads: dict[str, Mapping[str, Any]] = {}
    for side in ("blue", "red"):
        value = signal.get(side)
        if not isinstance(value, Mapping):
            raise CompositionSignalError(f"{side} composition summary is malformed")
        side_payloads[side] = value
        support = value.get("prior_role_games")
        if isinstance(support, bool) or not isinstance(support, int) or support < 0:
            raise CompositionSignalError(f"{side} composition support is invalid")
        summary = value.get("signal")
        if summary is not None and (
            not isinstance(summary, (int, float))
            or isinstance(summary, bool)
            or not np.isfinite(float(summary))
        ):
            raise CompositionSignalError(f"{side} composition summary is invalid")
        if status != "available" and summary is not None:
            raise CompositionSignalError(
                f"{status} composition signal exposes a team summary"
            )
        if status == "available" and summary is None:
            raise CompositionSignalError("available composition signal has no team summary")

    picks = signal.get("picks")
    if not isinstance(picks, list) or len(picks) != 10:
        raise CompositionSignalError("composition signal needs ten picks")
    seen: set[tuple[str, str]] = set()
    contribution_totals = {"Blue": 0.0, "Red": 0.0}
    evidence_statuses: list[str] = []
    for pick in picks:
        if not isinstance(pick, Mapping):
            raise CompositionSignalError("composition pick is malformed")
        side = str(pick.get("side") or "").strip().title()
        role = _role(pick.get("role"))
        champion = _champion(pick.get("champion"))
        key = (side, role)
        if key in seen or key not in expected or champion != expected[key]:
            raise CompositionSignalError("composition pick identity does not match the game")
        seen.add(key)
        support = pick.get("prior_role_games")
        if isinstance(support, bool) or not isinstance(support, int) or support < 0:
            raise CompositionSignalError("composition pick support is invalid")
        evidence = str(pick.get("evidence_status") or "")
        if evidence not in PUBLIC_EVIDENCE:
            raise CompositionSignalError("composition pick evidence status is invalid")
        contribution = pick.get("contribution")
        if contribution is not None and (
            not isinstance(contribution, (int, float))
            or isinstance(contribution, bool)
            or not np.isfinite(float(contribution))
        ):
            raise CompositionSignalError("composition pick contribution is invalid")
        if evidence == "available":
            if support < min_support_games or contribution is None:
                raise CompositionSignalError("available composition pick lacks support")
            contribution_totals[side] += float(contribution)
        elif evidence == "limited":
            if support >= min_support_games or contribution is not None:
                raise CompositionSignalError("limited composition pick has full support")
        elif contribution is not None:
            raise CompositionSignalError("unavailable composition pick has a value")
        evidence_statuses.append(evidence)

    if seen != set(expected):
        raise CompositionSignalError("composition signal has incomplete pick identities")
    if status == "available":
        if any(evidence != "available" for evidence in evidence_statuses):
            raise CompositionSignalError("available composition signal has limited picks")
        for side in ("Blue", "Red"):
            summary = float(side_payloads[side.lower()]["signal"])
            if not np.isclose(summary, contribution_totals[side], atol=1e-5):
                raise CompositionSignalError(
                    f"{side} composition summary does not match its picks"
                )
    elif status == "limited":
        if "limited" not in evidence_statuses:
            raise CompositionSignalError("limited composition signal has no limited pick")
    else:
        if any(evidence != "unavailable" for evidence in evidence_statuses):
            raise CompositionSignalError("unavailable composition signal has supported picks")


@dataclass(frozen=True)
class CompositionScoreResult:
    signals: dict[str, dict[str, Any]]
    audit: dict[str, Any]


def score_games_temporally(
    games: Sequence[Mapping[str, Any]],
    *,
    target_game_ids: Iterable[str] | None = None,
    cache_dir: Path | None = None,
    source_digest: str | None = None,
    worker_commit: str | None = None,
    min_support_games: int = MIN_SUPPORT_GAMES,
    min_training_games: int = MIN_TRAINING_GAMES,
) -> CompositionScoreResult:
    """Score target games from checkpoints fit before each target date."""

    ordered = sorted(games, key=lambda game: (_timestamp(game["date"]), str(game["game_uid"])))
    target_ids = (
        {canonical_source_game_key(value) for value in target_game_ids if canonical_source_game_key(value)}
        if target_game_ids is not None
        else {str(game["game_uid"]) for game in ordered}
    )
    digest = source_digest or _digest(str(game["game_uid"]) for game in ordered)
    by_date: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    for game in ordered:
        if str(game["game_uid"]) in target_ids:
            by_date.setdefault(_day(game["date"]), []).append(dict(game))
    signals: dict[str, dict[str, Any]] = {}
    cache_hits = 0
    for cutoff, target_games in sorted(by_date.items()):
        model, hit = _load_or_fit(
            ordered,
            cutoff,
            cache_dir=cache_dir,
            min_training_games=min_training_games,
            worker_commit=worker_commit,
        )
        cache_hits += int(hit)
        for game in target_games:
            signals[str(game["game_uid"])] = public_signal_for_game(
                game,
                model,
                min_support_games=min_support_games,
            )
    statuses = {status: 0 for status in PUBLIC_STATUS}
    fit_dates: list[str] = []
    for signal in signals.values():
        statuses[str(signal.get("status"))] = statuses.get(str(signal.get("status")), 0) + 1
        if signal.get("fit_through"):
            fit_dates.append(str(signal["fit_through"]))
    return CompositionScoreResult(
        signals=signals,
        audit={
            "schema_version": SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "included_terms": list(MODEL_TERMS),
            "excluded_terms": list(EXCLUDED_TERMS),
            "training_order": "earlier accepted calendar-date clusters only",
            "status": "available" if statuses["available"] else "limited" if statuses["limited"] else "unavailable",
            "target_games": len(signals),
            "available_games": statuses["available"],
            "limited_games": statuses["limited"],
            "unavailable_games": statuses["unavailable"],
            "fit_through": max(fit_dates) if fit_dates else None,
            "source_identity_sha256": digest,
            "cache_hits": cache_hits,
            "worker_commit": worker_commit or os.environ.get("GIT_COMMIT") or os.environ.get("SCRYGLASS_WORKER_COMMIT"),
            "min_support_games": min_support_games,
            "regularization_c": REGULARIZATION_C,
        },
    )


def _probability(logit: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0))))


def _metrics(outcomes: Sequence[int], probabilities: Sequence[float]) -> dict[str, float | int | None]:
    y = np.asarray(outcomes, dtype=np.int8)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-5, 1 - 1e-5)
    auc = None
    if len(np.unique(y)) == 2:
        auc = round(float(roc_auc_score(y, p)), 6)
    return {
        "n": int(len(y)),
        "brier": round(float(brier_score_loss(y, p)), 6),
        "log_loss": round(float(log_loss(y, p, labels=[0, 1])), 6),
        "auc": auc,
        **_calibration(y, p),
    }


def _calibration(outcomes: np.ndarray, probabilities: np.ndarray) -> dict[str, float | None]:
    if len(outcomes) < 2 or len(np.unique(outcomes)) < 2:
        return {"calibration_slope": None, "calibration_intercept": None}
    logits = np.log(np.clip(probabilities, 1e-5, 1 - 1e-5) / np.clip(1 - probabilities, 1e-5, 1 - 1e-5))
    model = LogisticRegression(C=1e6, solver="liblinear", max_iter=1000)
    model.fit(logits.reshape(-1, 1), outcomes)
    return {
        "calibration_slope": round(float(model.coef_[0][0]), 6),
        "calibration_intercept": round(float(model.intercept_[0]), 6),
    }


def _calibration_within_tolerance(metrics: Mapping[str, Any]) -> bool:
    slope = metrics.get("calibration_slope")
    intercept = metrics.get("calibration_intercept")
    return (
        (slope is None or abs(float(slope) - 1.0) <= CALIBRATION_SLOPE_TOLERANCE)
        and (intercept is None or abs(float(intercept)) <= CALIBRATION_INTERCEPT_TOLERANCE)
    )


def _cluster_key(game: Mapping[str, Any]) -> str:
    return str(game.get("series_id") or game.get("grid_series_id") or game.get("tournament") or game["game_uid"])


def _support_bucket(value: int) -> str:
    if value < 10:
        return "0-9"
    if value < MIN_SUPPORT_GAMES:
        return "10-39"
    if value < 80:
        return "40-79"
    if value < 160:
        return "80-159"
    return "160+"


def _history_features(games: Sequence[Mapping[str, Any]], model: FittedCompositionModel) -> np.ndarray:
    histories: dict[str, list[float]] = {}
    values: list[float] = []
    for game in games:
        blue_signal = sum(model.coefficient(role, game["blue"][role]["champion"]) for role in ROLES)
        red_signal = sum(model.coefficient(role, game["red"][role]["champion"]) for role in ROLES)
        blue_history = histories.get(normalize_team(str(game["blue_team"])), [])
        red_history = histories.get(normalize_team(str(game["red_team"])), [])
        values.append(
            (float(np.mean(blue_history)) if blue_history else 0.0)
            - (float(np.mean(red_history)) if red_history else 0.0)
        )
        histories.setdefault(normalize_team(str(game["blue_team"])), []).append(blue_signal)
        histories.setdefault(normalize_team(str(game["red_team"])), []).append(red_signal)
    return np.asarray(values, dtype=float)


def _bootstrap_deltas(
    outcomes: Sequence[int],
    baseline: Sequence[float],
    draft: Sequence[float],
    clusters: Sequence[str],
    *,
    reps: int,
    seed: int,
) -> dict[str, dict[str, float | None]]:
    groups: dict[str, list[int]] = {}
    for index, cluster in enumerate(clusters):
        groups.setdefault(cluster, []).append(index)
    names = list(groups)
    if not names or reps < 1:
        return {"brier_delta": {"lower": None, "upper": None}, "log_loss_delta": {"lower": None, "upper": None}}
    rng = np.random.default_rng(seed)
    deltas: list[tuple[float, float]] = []
    y = np.asarray(outcomes)
    base = np.asarray(baseline)
    comp = np.asarray(draft)
    for _ in range(reps):
        selected = rng.integers(0, len(names), size=len(names))
        indexes = [index for selected_index in selected for index in groups[names[selected_index]]]
        deltas.append(
            (
                float(brier_score_loss(y[indexes], comp[indexes]) - brier_score_loss(y[indexes], base[indexes])),
                float(
                    log_loss(
                        y[indexes],
                        np.clip(comp[indexes], 1e-5, 1 - 1e-5),
                        labels=[0, 1],
                    )
                    - log_loss(
                        y[indexes],
                        np.clip(base[indexes], 1e-5, 1 - 1e-5),
                        labels=[0, 1],
                    )
                ),
            )
        )
    array = np.asarray(deltas)
    return {
        "brier_delta": {"lower": round(float(np.quantile(array[:, 0], 0.025)), 6), "upper": round(float(np.quantile(array[:, 0], 0.975)), 6)},
        "log_loss_delta": {"lower": round(float(np.quantile(array[:, 1], 0.025)), 6), "upper": round(float(np.quantile(array[:, 1], 0.975)), 6)},
    }


def evaluate_composition_signal(
    games: Sequence[Mapping[str, Any]],
    *,
    source_hash: str | None = None,
    canonical_game_identity_sha256: str | None = None,
    worker_commit: str | None = None,
    bootstrap_reps: int = 200,
    seed: int = 461,
    min_training_games: int = MIN_TRAINING_GAMES,
) -> dict[str, Any]:
    """Run four chronological holdouts for the composition candidate."""

    ordered = [dict(game) for game in sorted(games, key=lambda item: (_timestamp(item["date"]), str(item["game_uid"]))) if game.get("controls_available", False)]
    if len(ordered) < max(20, min_training_games + 4):
        raise CompositionSignalError("not enough complete games for the four-window evaluation")
    windows: list[dict[str, Any]] = []
    pooled_y: list[int] = []
    pooled_base: list[float] = []
    pooled_draft: list[float] = []
    pooled_history: list[float] = []
    pooled_clusters: list[str] = []
    per_role: dict[str, dict[str, int]] = {role: {"available_picks": 0, "total_picks": 0} for role in ROLES}
    per_support: dict[str, dict[str, int | float | None]] = {
        bucket: {"picks": 0, "available_picks": 0, "prior_games_total": 0}
        for bucket in ("0-9", "10-39", "40-79", "80-159", "160+")
    }
    date_clusters: list[list[dict[str, Any]]] = []
    for game in ordered:
        if not date_clusters or _day(date_clusters[-1][0]["date"]) != _day(game["date"]):
            date_clusters.append([])
        date_clusters[-1].append(game)
    boundaries = np.linspace(0, len(date_clusters), 6).astype(int)
    for window_index in range(4):
        train = [game for cluster in date_clusters[: boundaries[window_index + 1]] for game in cluster]
        validation = [game for cluster in date_clusters[boundaries[window_index + 1] : boundaries[window_index + 2]] for game in cluster]
        training_names = _feature_names(train)
        baseline = _fit_model(train, names=training_names, include_draft=False, min_training_games=min_training_games)
        draft = _fit_model(train, names=training_names, include_draft=True, min_training_games=min_training_games)
        if baseline is None or draft is None or not validation:
            continue
        baseline_probabilities = [_probability(baseline.logit(game, include_draft=False)) for game in validation]
        draft_probabilities = [_probability(draft.logit(game, include_draft=True)) for game in validation]
        y = [int(game["y"]) for game in validation]
        windows.append(
            {
                "window": window_index + 1,
                "fit_through": draft.fit_through,
                "holdout_from": _rfc(validation[0]["date"]),
                "holdout_through": _rfc(validation[-1]["date"]),
                "baseline": _metrics(y, baseline_probabilities),
                "draft_augmented": _metrics(y, draft_probabilities),
            }
        )
        pooled_y.extend(y)
        pooled_base.extend(baseline_probabilities)
        pooled_draft.extend(draft_probabilities)
        pooled_clusters.extend(_cluster_key(game) for game in validation)
        for game in validation:
            for role in ROLES:
                per_role[role]["total_picks"] += 2
                for side in ("blue", "red"):
                    champion = _champion(game[side][role]["champion"])
                    support = draft.support.get(f"{role}|{champion}", 0)
                    bucket = per_support[_support_bucket(support)]
                    bucket["picks"] += 1
                    bucket["prior_games_total"] += support
                    if support >= MIN_SUPPORT_GAMES:
                        per_role[role]["available_picks"] += 1
                        bucket["available_picks"] += 1
        history = _history_features(train + validation, draft)
        train_history = history[: len(train)]
        validation_history = history[len(train) :]
        history_model = LogisticRegression(C=REGULARIZATION_C, solver="liblinear", max_iter=2000, random_state=461)
        history_train_x = np.column_stack([_matrix(train, draft.feature_names, include_draft=True) @ np.asarray(draft.coefficients), train_history])
        history_model.fit(history_train_x, [int(game["y"]) for game in train])
        history_validation_x = np.column_stack([_matrix(validation, draft.feature_names, include_draft=True) @ np.asarray(draft.coefficients), validation_history])
        history_probabilities = history_model.predict_proba(history_validation_x)[:, 1]
        windows[-1]["draft_plus_team_history"] = _metrics(y, history_probabilities)
        pooled_history.extend(float(value) for value in history_probabilities)
    if not windows:
        raise CompositionSignalError("chronological evaluation did not produce a valid holdout")
    bootstrap = _bootstrap_deltas(pooled_y, pooled_base, pooled_draft, pooled_clusters, reps=bootstrap_reps, seed=seed)
    history_bootstrap = _bootstrap_deltas(
        pooled_y,
        pooled_draft,
        pooled_history,
        pooled_clusters,
        reps=bootstrap_reps,
        seed=seed + 1,
    )
    improved_brier = sum(row["draft_augmented"]["brier"] < row["baseline"]["brier"] for row in windows)
    improved_log_loss = sum(row["draft_augmented"]["log_loss"] < row["baseline"]["log_loss"] for row in windows)
    calibration_ok = all(
        _calibration_within_tolerance(row["draft_augmented"])
        for row in windows
    )
    gate = {
        "brier_improved_windows": improved_brier,
        "log_loss_improved_windows": improved_log_loss,
        "brier_improved_in_three_windows": improved_brier >= 3,
        "log_loss_improved_in_three_windows": improved_log_loss >= 3,
        "pooled_brier_interval_supports_improvement": (bootstrap["brier_delta"]["upper"] or 1.0) < 0,
        "pooled_log_loss_interval_supports_improvement": (bootstrap["log_loss_delta"]["upper"] or 1.0) < 0,
        "calibration_within_tolerance": calibration_ok,
    }
    gate["composition_candidate_passes"] = all(
        gate[key]
        for key in (
            "brier_improved_in_three_windows",
            "log_loss_improved_in_three_windows",
            "pooled_brier_interval_supports_improvement",
            "pooled_log_loss_interval_supports_improvement",
            "calibration_within_tolerance",
        )
    )
    history_brier_improved = sum(
        row["draft_plus_team_history"]["brier"] < row["draft_augmented"]["brier"]
        for row in windows
        if row.get("draft_plus_team_history")
    )
    history_log_loss_improved = sum(
        row["draft_plus_team_history"]["log_loss"] < row["draft_augmented"]["log_loss"]
        for row in windows
        if row.get("draft_plus_team_history")
    )
    history_calibration_ok = all(
        _calibration_within_tolerance(row.get("draft_plus_team_history", {}))
        for row in windows
    )
    team_history_gate = {
        "brier_improved_windows": history_brier_improved,
        "log_loss_improved_windows": history_log_loss_improved,
        "brier_improved_in_three_windows": history_brier_improved >= 3,
        "log_loss_improved_in_three_windows": history_log_loss_improved >= 3,
        "pooled_brier_interval_supports_improvement": (history_bootstrap["brier_delta"]["upper"] or 1.0) < 0,
        "pooled_log_loss_interval_supports_improvement": (history_bootstrap["log_loss_delta"]["upper"] or 1.0) < 0,
        "calibration_within_tolerance": history_calibration_ok,
    }
    team_history_gate["rating_integration_eligible"] = all(
        team_history_gate[key]
        for key in (
            "brier_improved_in_three_windows",
            "log_loss_improved_in_three_windows",
            "pooled_brier_interval_supports_improvement",
            "pooled_log_loss_interval_supports_improvement",
            "calibration_within_tolerance",
        )
    )
    for bucket in per_support.values():
        bucket["mean_prior_games"] = round(
            bucket["prior_games_total"] / bucket["picks"], 2
        ) if bucket["picks"] else None
    digest = _digest(str(game["game_uid"]) for game in ordered)
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "included_terms": list(MODEL_TERMS),
        "excluded_terms": list(EXCLUDED_TERMS),
        "training_order": "earlier accepted calendar-date clusters only",
        "regularization_c": REGULARIZATION_C,
        "calibration_tolerance": {
            "slope": CALIBRATION_SLOPE_TOLERANCE,
            "intercept": CALIBRATION_INTERCEPT_TOLERANCE,
        },
        "source_hash": source_hash or digest,
        "canonical_game_identity_sha256": canonical_game_identity_sha256 or digest,
        "worker_commit": worker_commit or os.environ.get("GIT_COMMIT") or os.environ.get("SCRYGLASS_WORKER_COMMIT"),
        "fit_through": _rfc(date_clusters[boundaries[1] - 1][-1]["date"]),
        "games": len(ordered),
        "holdout_windows": windows,
        "pooled_bootstrap": bootstrap,
        "team_history_bootstrap": history_bootstrap,
        "per_role_support": per_role,
        "per_support_count": per_support,
        "team_history_diagnostic": {
            "included_in_team_rating": False,
            "interpretation": "Diagnostic only. Team drafting history stays outside team ratings until a separate held-out gate passes.",
            "promotion_gate": team_history_gate,
        },
        "role_pair_model": {
            "included_in_v1": False,
            "reason": "The role-pair candidate did not meet the registered holdout gate.",
        },
        "promotion_gate": gate,
    }


def write_evaluation_report(report: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", type=Path, required=True)
    parser.add_argument("--strength", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-hash")
    parser.add_argument("--canonical-game-digest")
    parser.add_argument("--worker-commit")
    args = parser.parse_args()
    players = pd.read_parquet(args.players)
    strength = pd.read_parquet(args.strength)
    report = evaluate_composition_signal(
        build_composition_games(players, strength_features=strength),
        source_hash=args.source_hash,
        canonical_game_identity_sha256=args.canonical_game_digest,
        worker_commit=args.worker_commit,
    )
    write_evaluation_report(report, args.out)


if __name__ == "__main__":
    _main()
