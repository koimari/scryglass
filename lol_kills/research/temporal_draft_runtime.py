"""Build and backtest expanding pre-event Draft Score runtimes.

This is the temporal upgrade for the retrospective Leaguepedia ledger.  Each
snapshot is fit only from maps whose event time is strictly before the
snapshot cutoff.  The July target draft is scored into a separate file before
the target outcomes are read for evaluation.

The artifact intentionally exposes two estimates:

* ``p_blue_draft``: composition terms only;
* ``p_blue_context``: composition plus historical team/player/league identity
  terms, when the pre-match lineup evidence is available.

The second estimate is contextual winner prediction, not pure draft strength.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import SGDClassifier

from lol_kills.draft_recommendation import (
    _feature_rows,
    _fit,
    _recency_weights,
    _vocabulary,
    build_games,
)
from lol_kills.etl.aliases import normalize_champ, normalize_team
from lol_kills.etl.manual_leaguepedia import resolve_time_sliced_lineup
from lol_kills.etl.manual_leaguepedia_batch import _competition, _json_rows, _safe
from lol_kills.etl.roster_receipts import (
    RosterReceiptError,
    load_receipt_manifest,
)
from lol_kills.research.player_champion_features import (
    PLAYER_CHAMPION_FEATURES,
    player_champion_feature_rows,
)
from lol_kills.v2.data.common import sha256_canonical_object


SCHEMA_VERSION = "scryglass:temporal-draft-runtime:v1"
ROLES = ("top", "jng", "mid", "bot", "sup")
ROLE_ALIASES = {
    "top": "top",
    "jungle": "jng",
    "jng": "jng",
    "mid": "mid",
    "middle": "mid",
    "bot": "bot",
    "bottom": "bot",
    "adc": "bot",
    "support": "sup",
    "sup": "sup",
    "utility": "sup",
}
DRAFT_PREFIXES = ("M|", "MR|", "S|", "AS|", "C|", "AC|", "R|")
CONTEXT_PREFIXES = ("L|", "T|", "P|")
BLUE_SIDE_BONUS = 0.03
DEFAULT_ALPHA = 0.001
DEFAULT_HALF_LIFE_DAYS = 365


class TemporalRuntimeError(RuntimeError):
    """Raised when a temporal snapshot cannot be built safely."""


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _role(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return ROLE_ALIASES.get(raw, raw[:3])


def _side(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"1", "blue"}:
        return "Blue"
    if raw in {"2", "red"}:
        return "Red"
    return ""


def _utc_naive(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _rfc(timestamp: pd.Timestamp) -> str:
    return timestamp.tz_localize("UTC").isoformat().replace("+00:00", "Z")


def _sigmoid(value: float) -> float:
    if value >= 35:
        return 1.0
    if value <= -35:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


ONLINE_FEATURES = (
    "team_elo",
    "league_team_elo",
    "player_elo",
    "team_form",
    "prior_games",
    "series_win_diff",
    "series_last",
    "series_maps_played",
)


def _series_key(game_id: str) -> str:
    match = re.match(r"^(.*)_\d+$", game_id)
    return match.group(1) if match else game_id


def _online_feature_rows(games: list[dict[str, Any]], *, update_k: float = 0.20) -> dict[str, dict[str, float]]:
    """Return features formed before each map's outcome is applied."""

    ordered = sorted(games, key=lambda game: (game["date"], game["game_uid"]))
    team_rating: dict[str, float] = {}
    league_team_rating: dict[tuple[str, str], float] = {}
    player_rating: dict[str, float] = {}
    team_wins: dict[str, int] = {}
    team_games: dict[str, int] = {}
    series_state: dict[str, dict[str, Any]] = {}
    result: dict[str, dict[str, float]] = {}

    def number(mapping: dict[Any, float], key: Any) -> float:
        return float(mapping.get(key, 0.0))

    def apply_update(game: Mapping[str, Any]) -> None:
        blue_team = str(game["blue_team"])
        red_team = str(game["red_team"])
        league = str(game["league"])
        y = float(game["y"])
        expected = _sigmoid(number(team_rating, blue_team) - number(team_rating, red_team))
        delta = update_k * (y - expected)
        team_rating[blue_team] = number(team_rating, blue_team) + delta
        team_rating[red_team] = number(team_rating, red_team) - delta
        league_blue = (league, blue_team)
        league_red = (league, red_team)
        expected_league = _sigmoid(number(league_team_rating, league_blue) - number(league_team_rating, league_red))
        league_delta = update_k * (y - expected_league)
        league_team_rating[league_blue] = number(league_team_rating, league_blue) + league_delta
        league_team_rating[league_red] = number(league_team_rating, league_red) - league_delta
        player_delta = update_k * (y - 0.5)
        for side, sign in (("blue", 1.0), ("red", -1.0)):
            for role in ROLES:
                player = str(game[side][role]["player"]).casefold()
                if player:
                    player_rating[player] = number(player_rating, player) + sign * player_delta
        team_wins[blue_team] = team_wins.get(blue_team, 0) + int(y)
        team_wins[red_team] = team_wins.get(red_team, 0) + int(1 - y)
        team_games[blue_team] = team_games.get(blue_team, 0) + 1
        team_games[red_team] = team_games.get(red_team, 0) + 1
        state = series_state.setdefault(_series_key(str(game["game_uid"])), {"wins": {}, "last": None})
        winner = blue_team if y else red_team
        state["wins"][winner] = state["wins"].get(winner, 0) + 1
        state["last"] = winner

    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end]["date"] == ordered[index]["date"]:
            end += 1
        for game in ordered[index:end]:
            blue_team = str(game["blue_team"])
            red_team = str(game["red_team"])
            league = str(game["league"])
            blue_players = [str(game["blue"][role]["player"]).casefold() for role in ROLES]
            red_players = [str(game["red"][role]["player"]).casefold() for role in ROLES]
            blue_form = (team_wins.get(blue_team, 0) + 1.0) / (team_games.get(blue_team, 0) + 2.0)
            red_form = (team_wins.get(red_team, 0) + 1.0) / (team_games.get(red_team, 0) + 2.0)
            state = series_state.setdefault(_series_key(str(game["game_uid"])), {"wins": {}, "last": None})
            blue_wins = int(state["wins"].get(blue_team, 0))
            red_wins = int(state["wins"].get(red_team, 0))
            last = state["last"]
            result[str(game["game_uid"])] = {
                "team_elo": number(team_rating, blue_team) - number(team_rating, red_team),
                "league_team_elo": number(league_team_rating, (league, blue_team)) - number(league_team_rating, (league, red_team)),
                "player_elo": (
                    sum(number(player_rating, player) for player in blue_players) / max(len(blue_players), 1)
                    - sum(number(player_rating, player) for player in red_players) / max(len(red_players), 1)
                ),
                "team_form": math.log(blue_form / (1 - blue_form)) - math.log(red_form / (1 - red_form)),
                "prior_games": float(min(team_games.get(blue_team, 0), team_games.get(red_team, 0))),
                "series_win_diff": float(blue_wins - red_wins),
                "series_last": 1.0 if last == blue_team else -1.0 if last == red_team else 0.0,
                "series_maps_played": float(blue_wins + red_wins),
            }
        for game in ordered[index:end]:
            apply_update(game)
        index = end
    return result


def _catalog(run_dir: Path, *, prior: bool) -> dict[str, dict[str, Any]]:
    if prior:
        directory = run_dir / "autoresearch" / "raw" / "prior-games"
        pattern = "prior-page-*.json"
    else:
        directory = run_dir / "raw" / "catalog"
        pattern = "games-page-*.json"
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob(pattern)):
        for row in _json_rows(path):
            game_id = _safe(row.get("GameId"))
            if game_id:
                result.setdefault(game_id, row)
    return result


def _outcomes(run_dir: Path, *, prior: bool) -> dict[str, dict[str, Any]]:
    if prior:
        rows = []
        for path in sorted((run_dir / "autoresearch" / "raw" / "prior-games").glob("prior-page-*.json")):
            rows.extend(_json_rows(path))
        return {_safe(row.get("GameId")): row for row in rows if _safe(row.get("GameId"))}
    return {_safe(row.get("GameId")): row for row in _load_jsonl(run_dir / "normalized-outcome-rows.jsonl")}


def _source_frame(run_dir: Path) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Convert prior + July raw rows into the trainer's complete-game shape."""

    prior_catalog = _catalog(run_dir, prior=True)
    prior_outcomes = _outcomes(run_dir, prior=True)
    prior_drafts = _load_jsonl(run_dir / "autoresearch" / "raw" / "prior-drafts" / "normalized-prior-draft-rows.jsonl")
    target_catalog = _catalog(run_dir, prior=False)
    target_outcomes = _outcomes(run_dir, prior=False)
    target_drafts = _load_jsonl(run_dir / "normalized-draft-rows.jsonl")

    rows: list[dict[str, Any]] = []
    by_game: dict[str, list[dict[str, Any]]] = {}
    for row in prior_drafts:
        by_game.setdefault(str(row["game_id"]), []).append(row)
    for row in target_drafts:
        by_game.setdefault(str(row["GameId"]), []).append(
            {
                "game_id": row["GameId"],
                "team": row.get("Team"),
                "player": row.get("Name"),
                "champion": row.get("Champion"),
                "role": row.get("IngameRole"),
                "side": row.get("Side"),
                "date": row.get("DateTime UTC"),
            }
        )

    metadata: dict[str, dict[str, Any]] = {}
    for game_id, catalog_row in prior_catalog.items():
        outcome = prior_outcomes.get(game_id)
        if outcome is None:
            continue
        metadata[game_id] = {
            "team1": normalize_team(_safe(catalog_row.get("Team1"))),
            "team2": normalize_team(_safe(catalog_row.get("Team2"))),
            "winner": normalize_team(_safe(outcome.get("WinTeam"))),
            "tournament": _safe(catalog_row.get("Tournament")),
            "date": _safe(catalog_row.get("DateTime UTC")),
            "prior": True,
        }
    for game_id, catalog_row in target_catalog.items():
        outcome = target_outcomes.get(game_id)
        if outcome is None:
            continue
        metadata[game_id] = {
            "team1": normalize_team(_safe(outcome.get("Team1") or catalog_row.get("Team1"))),
            "team2": normalize_team(_safe(outcome.get("Team2") or catalog_row.get("Team2"))),
            "winner": normalize_team(_safe(outcome.get("WinTeam"))),
            "tournament": _safe(catalog_row.get("Tournament")),
            "date": _safe(catalog_row.get("DateTime UTC") or outcome.get("DateTime UTC")),
            "prior": False,
        }

    for game_id, players in by_game.items():
        meta = metadata.get(game_id)
        if meta is None or meta["winner"] not in {meta["team1"], meta["team2"]}:
            continue
        y = 1.0 if meta["winner"] == meta["team1"] else 0.0
        league = _competition(meta["tournament"])["league"]
        for player in players:
            side = _side(player.get("side"))
            role = _role(player.get("role"))
            champion = normalize_champ(_safe(player.get("champion")))
            player_name = _safe(player.get("player") or player.get("Name"))
            team = normalize_team(_safe(player.get("team") or player.get("Team")))
            if side not in {"Blue", "Red"} or role not in ROLES or not champion or not player_name or not team:
                continue
            rows.append(
                {
                    "game_uid": game_id,
                    "date": _utc_naive(meta["date"]),
                    "league": league,
                    "side": side,
                    "position": role,
                    "champion": champion,
                    "playername": player_name,
                    "teamname": team,
                    "result": y,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise TemporalRuntimeError("no complete source rows available for temporal training")
    return frame, target_catalog, target_outcomes


def _target_game(run: Mapping[str, Any]) -> dict[str, Any]:
    pregame = run["pregame"]
    role_map = {"top": "top", "jungle": "jng", "mid": "mid", "bot": "bot", "support": "sup"}

    def side(name: str) -> dict[str, Any]:
        source = pregame[name]
        players = {role_map[str(row["role"])] : str(row["player"]) for row in source["players"]}
        picks = {role_map[role]: normalize_champ(str(champion)) for role, champion in zip(("top", "jungle", "mid", "bot", "support"), source["picks"])}
        return {
            role: {"champion": picks[role], "player": players.get(role, "")}
            for role in ROLES
        }

    return {
        "game_uid": str(pregame["fixture_id"]),
        "date": _utc_naive(pregame["event_start"]),
        "league": str(pregame["competition"].get("league") or "UNKNOWN").upper(),
        "blue_team": normalize_team(str(pregame["blue"]["team"])),
        "red_team": normalize_team(str(pregame["red"]["team"])),
        "blue": side("blue"),
        "red": side("red"),
    }


def _feature_group(key: str) -> str:
    if key.startswith(DRAFT_PREFIXES):
        return "draft"
    if key.startswith(CONTEXT_PREFIXES):
        return "context"
    return "other"


@dataclass
class TemporalSnapshot:
    cutoff: pd.Timestamp
    training_games: int
    intercept: float
    vocabulary: dict[str, int]
    coefficients: dict[str, float]
    alpha: float
    half_life_days: int

    def score(self, game: Mapping[str, Any], *, allow_context: bool) -> dict[str, Any]:
        matrix = _feature_rows([dict(game)], self.vocabulary)
        row = matrix.getrow(0)
        groups = {"draft": 0.0, "context": 0.0, "other": 0.0}
        components = {
            "champion": 0.0,
            "role_main": 0.0,
            "synergy": 0.0,
            "direct_counter": 0.0,
            "composition_counter": 0.0,
            "lane": 0.0,
        }
        inverse = {index: key for key, index in self.vocabulary.items()}
        for column, value in zip(row.indices, row.data):
            key = inverse[int(column)]
            contribution = float(value) * float(self.coefficients.get(key, 0.0))
            groups[_feature_group(key)] += contribution
            if key.startswith("M|"):
                components["champion"] += contribution
            elif key.startswith("MR|"):
                components["role_main"] += contribution
            elif key.startswith(("S|", "AS|")):
                components["synergy"] += contribution
            elif key.startswith("C|"):
                components["direct_counter"] += contribution
            elif key.startswith("AC|"):
                components["composition_counter"] += contribution
            elif key.startswith("R|"):
                components["lane"] += contribution
        base = self.intercept + BLUE_SIDE_BONUS
        draft_edge = base + groups["draft"]
        context_edge = draft_edge + groups["context"]
        context_probability = _sigmoid(context_edge) if allow_context else None
        return {
            "model_version": SCHEMA_VERSION,
            "snapshot_as_of": _rfc(self.cutoff),
            "training_games": self.training_games,
            "p_blue_draft": round(_sigmoid(draft_edge), 6),
            "p_blue_context": round(context_probability, 6) if context_probability is not None else None,
            "draft_edge": round(draft_edge, 6),
            "context_edge": round(context_edge, 6) if allow_context else None,
            "components": {key: round(value, 6) for key, value in components.items()},
            "context_available": bool(allow_context),
        }

    def artifact(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_version": "temporal-draft-v1.0.0",
            "snapshot_as_of": _rfc(self.cutoff),
            "training_games": self.training_games,
            "fit": {
                "alpha": self.alpha,
                "half_life_days": self.half_life_days,
                "blue_side_bonus": BLUE_SIDE_BONUS,
                "feature_semantics": {
                    "draft": list(DRAFT_PREFIXES),
                    "context": list(CONTEXT_PREFIXES),
                },
            },
            "intercept": round(self.intercept, 12),
            "feature_coefficients": {
                key: round(value, 12)
                for key, value in sorted(self.coefficients.items())
                if math.isfinite(value) and abs(value) > 1e-12
            },
        }


@dataclass
class TemporalHybrid:
    cutoff: pd.Timestamp
    training_games: int
    intercept: float
    vocabulary: dict[str, int]
    draft_coefficients: dict[str, float]
    online_coefficients: dict[str, float]
    online_means: dict[str, float]
    online_scales: dict[str, float]
    player_champion_coefficients: dict[str, float]
    player_champion_means: dict[str, float]
    player_champion_scales: dict[str, float]
    update_k: float

    def score(
        self,
        game: Mapping[str, Any],
        online: Mapping[str, float],
        *,
        player_champion: Mapping[str, Any] | None = None,
        allow_context: bool,
    ) -> dict[str, Any]:
        matrix = _feature_rows([dict(game)], self.vocabulary)
        row = matrix.getrow(0)
        inverse = {index: key for key, index in self.vocabulary.items()}
        draft_edge = self.intercept + BLUE_SIDE_BONUS
        sparse_context = 0.0
        for column, value in zip(row.indices, row.data):
            key = inverse[int(column)]
            contribution = float(value) * self.draft_coefficients.get(key, 0.0)
            if _feature_group(key) == "draft":
                draft_edge += contribution
            else:
                sparse_context += contribution
        online_edge = 0.0
        standardized: dict[str, float] = {}
        for name in ONLINE_FEATURES:
            value = (float(online.get(name, 0.0)) - self.online_means[name]) / self.online_scales[name]
            standardized[name] = value
            online_edge += value * self.online_coefficients[name]
        player_champion_edge = 0.0
        player_champion_standardized: dict[str, float] = {}
        player_champion_breakdown: list[dict[str, Any]] = []
        for name in PLAYER_CHAMPION_FEATURES:
            raw_value = float((player_champion or {}).get(name, 0.0))
            value = (raw_value - self.player_champion_means[name]) / self.player_champion_scales[name]
            player_champion_standardized[name] = value
            player_champion_edge += value * self.player_champion_coefficients[name]
        if player_champion:
            player_champion_breakdown = list(player_champion.get("breakdown") or [])
        context_without_draft_edge = (
            self.intercept
            + BLUE_SIDE_BONUS
            + sparse_context
            + online_edge
            + player_champion_edge
        )
        context_edge = draft_edge + sparse_context + online_edge + player_champion_edge
        return {
            "model_version": "temporal-hybrid-v1.3.0",
            "snapshot_as_of": _rfc(self.cutoff),
            "training_games": self.training_games,
            "update_k": self.update_k,
            "p_blue_draft": round(_sigmoid(draft_edge), 6),
            "p_blue_context_without_draft": (
                round(_sigmoid(context_without_draft_edge), 6)
                if allow_context
                else None
            ),
            "p_blue_context": round(_sigmoid(context_edge), 6) if allow_context else None,
            "draft_edge": round(draft_edge, 6),
            "context_without_draft_edge": (
                round(context_without_draft_edge, 6) if allow_context else None
            ),
            "context_edge": round(context_edge, 6) if allow_context else None,
            "sparse_context_edge": round(sparse_context, 6),
            "online_context_edge": round(online_edge, 6),
            "player_champion_context_edge": round(player_champion_edge, 6),
            "online_features": {key: round(value, 6) for key, value in standardized.items()},
            "player_champion_features": {
                key: round(value, 6)
                for key, value in player_champion_standardized.items()
            },
            "player_champion_breakdown": player_champion_breakdown,
            "context_available": bool(allow_context),
        }

    def artifact(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_version": "temporal-hybrid-v1.3.0",
            "snapshot_as_of": _rfc(self.cutoff),
            "training_games": self.training_games,
            "fit": {
                "alpha": DEFAULT_ALPHA,
                "half_life_days": DEFAULT_HALF_LIFE_DAYS,
                "blue_side_bonus": BLUE_SIDE_BONUS,
                "online_update_k": self.update_k,
                "player_champion_half_life_days": DEFAULT_HALF_LIFE_DAYS,
                "player_champion_priors": {
                    "player_champion": 8.0,
                    "player_role": 12.0,
                    "champion_role": 18.0,
                    "interaction": 16.0,
                },
            },
            "intercept": round(self.intercept, 12),
            "draft_feature_coefficients": {
                key: round(value, 12)
                for key, value in sorted(self.draft_coefficients.items())
                if math.isfinite(value) and abs(value) > 1e-12
            },
            "online_feature_coefficients": {
                key: round(value, 12) for key, value in sorted(self.online_coefficients.items())
            },
            "online_feature_means": self.online_means,
            "online_feature_scales": self.online_scales,
            "player_champion_feature_coefficients": {
                key: round(value, 12)
                for key, value in self.player_champion_coefficients.items()
            },
            "player_champion_feature_means": self.player_champion_means,
            "player_champion_feature_scales": self.player_champion_scales,
        }


def fit_snapshot(games: list[dict[str, Any]], cutoff: pd.Timestamp) -> TemporalSnapshot:
    training = [game for game in games if game["date"] < cutoff]
    if not training:
        raise TemporalRuntimeError(f"no maps precede snapshot cutoff {_rfc(cutoff)}")
    if any(game["date"] >= cutoff for game in training):
        raise TemporalRuntimeError("temporal training filter admitted a map at/after cutoff")
    vocabulary, _ = _vocabulary(training)
    matrix = _feature_rows(training, vocabulary)
    outcomes = np.array([int(game["y"]) for game in training], dtype=int)
    reference = max(game["date"] for game in training)
    weights = _recency_weights(training, reference, DEFAULT_HALF_LIFE_DAYS)
    model = _fit(matrix, outcomes, weights, DEFAULT_ALPHA)
    coefficient = model.coef_[0]
    coefficients = {key: float(coefficient[index]) for key, index in vocabulary.items()}
    return TemporalSnapshot(
        cutoff=cutoff,
        training_games=len(training),
        intercept=float(model.intercept_[0]),
        vocabulary=vocabulary,
        coefficients=coefficients,
        alpha=DEFAULT_ALPHA,
        half_life_days=DEFAULT_HALF_LIFE_DAYS,
    )


def fit_hybrid(games: list[dict[str, Any]], cutoff: pd.Timestamp, *, update_k: float = 0.20) -> TemporalHybrid:
    training = [game for game in games if game["date"] < cutoff]
    if not training:
        raise TemporalRuntimeError(f"no maps precede hybrid snapshot cutoff {_rfc(cutoff)}")
    if any(game["date"] >= cutoff for game in training):
        raise TemporalRuntimeError("hybrid temporal training filter admitted a map at/after cutoff")
    vocabulary, _ = _vocabulary(training)
    draft_matrix = _feature_rows(training, vocabulary)
    online_rows = _online_feature_rows(games, update_k=update_k)
    player_champion_rows = player_champion_feature_rows(
        games,
        half_life_days=DEFAULT_HALF_LIFE_DAYS,
    )
    online_values = np.array(
        [[online_rows[str(game["game_uid"])][name] for name in ONLINE_FEATURES] for game in training],
        dtype=float,
    )
    means = online_values.mean(axis=0)
    scales = online_values.std(axis=0)
    scales[scales < 1e-9] = 1.0
    online_scaled = (online_values - means) / scales
    player_champion_values = np.array(
        [
            [
                float(player_champion_rows[str(game["game_uid"])].get(name, 0.0))
                for name in PLAYER_CHAMPION_FEATURES
            ]
            for game in training
        ],
        dtype=float,
    )
    player_champion_means = player_champion_values.mean(axis=0)
    player_champion_scales = player_champion_values.std(axis=0)
    player_champion_scales[player_champion_scales < 1e-9] = 1.0
    player_champion_scaled = (
        player_champion_values - player_champion_means
    ) / player_champion_scales
    matrix = sparse.hstack(
        [
            draft_matrix,
            sparse.csr_matrix(online_scaled),
            sparse.csr_matrix(player_champion_scaled),
        ],
        format="csr",
    )
    outcomes = np.array([int(game["y"]) for game in training], dtype=int)
    reference = max(game["date"] for game in training)
    weights = _recency_weights(training, reference, DEFAULT_HALF_LIFE_DAYS)
    model = _fit(matrix, outcomes, weights, DEFAULT_ALPHA)
    coefficient = model.coef_[0]
    draft_coefficients = {
        key: float(coefficient[index])
        for key, index in vocabulary.items()
    }
    online_coefficients = {
        name: float(coefficient[len(vocabulary) + index])
        for index, name in enumerate(ONLINE_FEATURES)
    }
    player_champion_start = len(vocabulary) + len(ONLINE_FEATURES)
    player_champion_coefficients = {
        name: float(coefficient[player_champion_start + index])
        for index, name in enumerate(PLAYER_CHAMPION_FEATURES)
    }
    return TemporalHybrid(
        cutoff=cutoff,
        training_games=len(training),
        intercept=float(model.intercept_[0]),
        vocabulary=vocabulary,
        draft_coefficients=draft_coefficients,
        online_coefficients=online_coefficients,
        online_means={name: round(float(means[index]), 12) for index, name in enumerate(ONLINE_FEATURES)},
        online_scales={name: round(float(scales[index]), 12) for index, name in enumerate(ONLINE_FEATURES)},
        player_champion_coefficients=player_champion_coefficients,
        player_champion_means={
            name: round(float(player_champion_means[index]), 12)
            for index, name in enumerate(PLAYER_CHAMPION_FEATURES)
        },
        player_champion_scales={
            name: round(float(player_champion_scales[index]), 12)
            for index, name in enumerate(PLAYER_CHAMPION_FEATURES)
        },
        update_k=update_k,
    )


def _metrics(rows: list[dict[str, Any]], probability_key: str) -> dict[str, Any]:
    usable = [row for row in rows if row["score"].get(probability_key) is not None]
    y = np.array([row["actual_blue_win"] for row in usable], dtype=float)
    p = np.clip(np.array([row["score"][probability_key] for row in usable], dtype=float), 1e-9, 1 - 1e-9)
    if not len(usable):
        return {"n": 0, "correct": 0, "accuracy_pct": None, "brier": None, "logloss": None}
    predicted = p >= 0.5
    return {
        "n": len(usable),
        "correct": int(np.sum(predicted == y)),
        "accuracy_pct": round(100 * float(np.mean(predicted == y)), 4),
        "brier": round(float(np.mean((p - y) ** 2)), 6),
        "logloss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 6),
    }


def _metric_delta(
    candidate: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any]:
    if candidate.get("n") != reference.get("n") or not candidate.get("n"):
        return {
            "n": 0,
            "brier_delta": None,
            "logloss_delta": None,
            "interpretation": "negative_delta_means_draft_terms_help",
        }
    return {
        "n": int(candidate["n"]),
        "brier_delta": round(
            float(candidate["brier"]) - float(reference["brier"]), 6
        ),
        "logloss_delta": round(
            float(candidate["logloss"]) - float(reference["logloss"]), 6
        ),
        "interpretation": "negative_delta_means_draft_terms_help",
    }


def _load_roster_events(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"roster evidence file does not exist: {path}")
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    value = json.loads(raw)
    if not isinstance(value, list):
        raise TemporalRuntimeError("roster evidence JSON must be an array")
    return [dict(row) for row in value]


def _lineup_status(
    run: Mapping[str, Any],
    events: list[dict[str, Any]],
    *,
    strict_roster: bool,
    receipts: Mapping[str, Mapping[str, Any]] | None = None,
    receipt_manifest_sha256: str | None = None,
) -> tuple[str, dict[str, Any]]:
    pregame = run["pregame"]
    fixture_id = str(pregame.get("fixture_id") or "")
    as_of = str(pregame["as_of"])
    event_start = str(pregame["event_start"])
    if receipts is not None:
        receipt = receipts.get(fixture_id)
        evidence: dict[str, Any] = {
            "source": "hash_bound_pre_event_roster_receipt",
            "manifest_sha256": receipt_manifest_sha256,
            "fixture_id": fixture_id,
        }
        if not isinstance(receipt, Mapping):
            evidence["reason"] = "fixture_receipt_missing"
            return "unavailable", evidence
        evidence["fixture_evidence_hash"] = receipt.get("evidence_hash")
        evidence["authority_status"] = receipt.get("authority_status")
        if receipt.get("authority_status") != "confirmed":
            evidence["reason"] = "fixture_receipt_unavailable"
            evidence["blockers"] = list(receipt.get("blockers") or [])
            return "unavailable", evidence
        if receipt.get("event_start") != event_start or receipt.get("as_of") != as_of:
            evidence["reason"] = "fixture_receipt_time_binding_mismatch"
            return "mismatch", evidence
        teams = receipt.get("teams")
        if not isinstance(teams, Mapping):
            evidence["reason"] = "fixture_receipt_teams_missing"
            return "unavailable", evidence
        for side_name in ("blue", "red"):
            team_receipt = teams.get(side_name)
            if not isinstance(team_receipt, Mapping):
                evidence[side_name] = {"status": "missing"}
                return "unavailable", evidence
            expected_team = normalize_team(str(pregame[side_name]["team"]))
            receipt_team = normalize_team(str(team_receipt.get("team") or ""))
            expected = sorted(
                (
                    _role(row.get("role")),
                    " ".join(str(row.get("player") or "").split()).casefold(),
                )
                for row in pregame[side_name]["players"]
                if isinstance(row, Mapping)
            )
            actual = sorted(
                (
                    _role(row.get("role")),
                    " ".join(str(row.get("player") or "").split()).casefold(),
                )
                for row in team_receipt.get("players", [])
                if isinstance(row, Mapping)
            )
            team_matches = expected_team == receipt_team
            lineup_matches = expected == actual
            evidence[side_name] = {
                "status": "verified" if team_matches and lineup_matches else "mismatch",
                "team_matches": team_matches,
                "lineup_matches": lineup_matches,
                "receipt_evidence_hash": team_receipt.get("evidence_hash"),
                "expected_lineup_sha256": sha256_canonical_object(expected),
                "receipt_lineup_sha256": sha256_canonical_object(actual),
            }
            if not team_matches or not lineup_matches:
                evidence["reason"] = "fixture_receipt_identity_mismatch"
                return "mismatch", evidence
        return "verified_preevent", evidence

    if not events:
        return ("unavailable" if strict_roster else "retrospective_lineup_only", {})
    evidence: dict[str, Any] = {}
    for side_name in ("blue", "red"):
        team = str(pregame[side_name]["team"])
        resolved = resolve_time_sliced_lineup(events, team, event_start=event_start, as_of=as_of)
        evidence[side_name] = resolved
        if resolved.get("status") != "ok":
            return "unavailable", evidence
        expected = {(str(row["role"]), str(row["player"])) for row in pregame[side_name]["players"]}
        actual = {(str(row["role"]), str(row["player"])) for row in resolved["players"]}
        if expected != actual:
            return "mismatch", evidence
    return "verified_preevent", evidence


def run_temporal_backtest(
    run_dir: Path,
    output_dir: Path,
    *,
    cadence_days: int = 1,
    roster_events_path: Path | None = None,
    roster_receipt_manifest_path: Path | None = None,
    strict_roster: bool = False,
) -> dict[str, Any]:
    if cadence_days < 1:
        raise TemporalRuntimeError("cadence_days must be positive")
    frozen = _load_jsonl(run_dir / "frozen-ledger.jsonl")
    frozen.sort(key=lambda row: (row["pregame"]["event_start"], row["pregame"]["fixture_id"]))
    if not frozen:
        raise TemporalRuntimeError("frozen ledger is empty")
    frame, _, target_outcomes = _source_frame(run_dir)
    games = build_games(frame)
    target_days = sorted({_utc_naive(row["pregame"]["event_start"]).normalize() for row in frozen})
    first_day = target_days[0]
    if roster_events_path is not None and roster_receipt_manifest_path is not None:
        raise TemporalRuntimeError(
            "roster events and a roster receipt manifest are mutually exclusive"
        )
    events = _load_roster_events(roster_events_path)
    receipt_readiness: dict[str, Any] | None = None
    receipt_index: dict[str, dict[str, Any]] | None = None
    if roster_receipt_manifest_path is not None:
        try:
            receipt_readiness, receipt_index = load_receipt_manifest(
                roster_receipt_manifest_path
            )
        except RosterReceiptError as exc:
            raise TemporalRuntimeError(f"roster receipt package is invalid: {exc}") from exc
    roster_source = (
        {
            "kind": "hash_bound_pre_event_roster_receipts",
            **dict(receipt_readiness or {}),
        }
        if receipt_index is not None
        else {
            "kind": "legacy_roster_events" if roster_events_path is not None else "none",
            "path": str(roster_events_path) if roster_events_path is not None else None,
            "sha256": _sha_file(roster_events_path) if roster_events_path is not None else None,
            "authority_status": "legacy_time_sliced" if roster_events_path is not None else "unavailable",
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = output_dir / "snapshots"
    scored: list[dict[str, Any]] = []
    snapshot_meta: list[dict[str, Any]] = []
    snapshots: dict[pd.Timestamp, TemporalSnapshot] = {}

    for day in target_days:
        bucket_index = (day - first_day).days // cadence_days
        bucket_start = first_day + pd.Timedelta(days=bucket_index * cadence_days)
        cutoff = bucket_start
        if bucket_start not in snapshots:
            snapshot = fit_snapshot(games, cutoff)
            snapshots[bucket_start] = snapshot
            artifact = snapshot.artifact()
            artifact_path = snapshots_dir / f"snapshot-{bucket_start.date().isoformat()}.json"
            _write_json(artifact_path, artifact)
            snapshot_meta.append(
                {
                    "snapshot_as_of": _rfc(cutoff),
                    "training_games": snapshot.training_games,
                    "artifact": artifact_path.name,
                    "artifact_sha256": _sha_file(artifact_path),
                }
            )
        snapshot = snapshots[bucket_start]
        for run in [row for row in frozen if _utc_naive(row["pregame"]["event_start"]).normalize() == day]:
            lineup_status, lineup_evidence = _lineup_status(
                run,
                events,
                strict_roster=strict_roster,
                receipts=receipt_index,
                receipt_manifest_sha256=(receipt_readiness or {}).get("manifest_sha256"),
            )
            game = _target_game(run)
            score = snapshot.score(game, allow_context=lineup_status == "verified_preevent" or not strict_roster)
            scored.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "fixture_id": run["pregame"]["fixture_id"],
                    "event_start": run["pregame"]["event_start"],
                    "snapshot": {
                        "as_of": score["snapshot_as_of"],
                        "training_games": score["training_games"],
                    },
                    "lineup": {
                        "status": lineup_status,
                        "evidence": lineup_evidence,
                    },
                    "score": score,
                }
            )

    scored.sort(key=lambda row: (row["event_start"], row["fixture_id"]))
    # This file is intentionally outcome-free.  Evaluation is performed after
    # it is written, keeping the phase boundary visible in the artifact set.
    _write_jsonl(output_dir / "temporal-scored-ledger.jsonl", scored)

    outcome_by_game = target_outcomes
    evaluated: list[dict[str, Any]] = []
    for row in scored:
        outcome = outcome_by_game.get(row["fixture_id"])
        frozen_row = next(item for item in frozen if item["pregame"]["fixture_id"] == row["fixture_id"])
        if outcome is None:
            continue
        winner = normalize_team(_safe(outcome.get("WinTeam")))
        blue = normalize_team(str(frozen_row["pregame"]["blue"]["team"]))
        row_copy = dict(row)
        row_copy["actual_blue_win"] = 1 if winner == blue else 0
        evaluated.append(row_copy)
    evaluation = {
        "schema_version": SCHEMA_VERSION,
        "maps": len(evaluated),
        "cadence_days": cadence_days,
        "strict_roster": strict_roster,
        "pure_draft": _metrics(evaluated, "p_blue_draft"),
        "contextual": _metrics(evaluated, "p_blue_context"),
        "contextual_lineup_coverage": sum(
            1 for row in evaluated if row["lineup"]["status"] == "verified_preevent"
        ),
        "roster_source": roster_source,
        "snapshot_count": len(snapshot_meta),
        "snapshots": snapshot_meta,
    }
    _write_json(output_dir / "temporal-evaluation.json", evaluation)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "availability_status": "adaptive_development_replay_not_independent",
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "source_hashes": {
            "frozen_ledger": _sha_file(run_dir / "frozen-ledger.jsonl"),
            "target_outcomes": _sha_file(run_dir / "normalized-outcome-rows.jsonl"),
            "prior_outcomes": _sha_file(run_dir / "autoresearch/raw/prior-games/normalized-prior-rows.jsonl"),
            "prior_drafts": _sha_file(run_dir / "autoresearch/raw/prior-drafts/normalized-prior-draft-rows.jsonl"),
            "runner": _sha_file(Path(__file__)),
            "roster_receipt_manifest": (receipt_readiness or {}).get("manifest_sha256"),
            "roster_receipt_file": (receipt_readiness or {}).get("receipt_file_sha256"),
            "legacy_roster_events": (
                _sha_file(roster_events_path) if roster_events_path is not None else None
            ),
        },
        "fit_policy": {
            "strict_before_cutoff": True,
            "alpha": DEFAULT_ALPHA,
            "half_life_days": DEFAULT_HALF_LIFE_DAYS,
            "blue_side_bonus": BLUE_SIDE_BONUS,
            "draft_prefixes": list(DRAFT_PREFIXES),
            "context_prefixes": list(CONTEXT_PREFIXES),
        },
        "evaluation": evaluation,
        "temporal_score_written_before_outcome_evaluation": True,
        "claim_ceiling": {
            "adaptive_development_diagnostic": True,
            "independent_validation": False,
            "production_probability": False,
            "betting": False,
        },
    }
    manifest["manifest_sha256"] = sha256_canonical_object(manifest)
    _write_json(output_dir / "temporal-run-manifest.json", manifest)
    return evaluation


def run_hybrid_backtest(
    run_dir: Path,
    output_dir: Path,
    *,
    update_k: float = 0.20,
    roster_events_path: Path | None = None,
    roster_receipt_manifest_path: Path | None = None,
    strict_roster: bool = False,
) -> dict[str, Any]:
    """Fit once before July, then use pre-map online ratings through July."""

    frozen = _load_jsonl(run_dir / "frozen-ledger.jsonl")
    frozen.sort(key=lambda row: (row["pregame"]["event_start"], row["pregame"]["fixture_id"]))
    if not frozen:
        raise TemporalRuntimeError("frozen ledger is empty")
    frame, _, target_outcomes = _source_frame(run_dir)
    games = build_games(frame)
    first_day = min(_utc_naive(row["pregame"]["event_start"]).normalize() for row in frozen)
    hybrid = fit_hybrid(games, first_day, update_k=update_k)
    online_rows = _online_feature_rows(games, update_k=update_k)
    player_champion_rows = player_champion_feature_rows(
        games,
        half_life_days=DEFAULT_HALF_LIFE_DAYS,
    )
    if roster_events_path is not None and roster_receipt_manifest_path is not None:
        raise TemporalRuntimeError(
            "roster events and a roster receipt manifest are mutually exclusive"
        )
    events = _load_roster_events(roster_events_path)
    receipt_readiness: dict[str, Any] | None = None
    receipt_index: dict[str, dict[str, Any]] | None = None
    if roster_receipt_manifest_path is not None:
        try:
            receipt_readiness, receipt_index = load_receipt_manifest(
                roster_receipt_manifest_path
            )
        except RosterReceiptError as exc:
            raise TemporalRuntimeError(f"roster receipt package is invalid: {exc}") from exc
    roster_source = (
        {
            "kind": "hash_bound_pre_event_roster_receipts",
            **dict(receipt_readiness or {}),
        }
        if receipt_index is not None
        else {
            "kind": "legacy_roster_events" if roster_events_path is not None else "none",
            "path": str(roster_events_path) if roster_events_path is not None else None,
            "sha256": _sha_file(roster_events_path) if roster_events_path is not None else None,
            "authority_status": "legacy_time_sliced" if roster_events_path is not None else "unavailable",
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "temporal-hybrid-runtime.json"
    _write_json(artifact_path, hybrid.artifact())
    games_by_id = {str(game["game_uid"]): game for game in games}
    scored: list[dict[str, Any]] = []
    for run in frozen:
        fixture_id = str(run["pregame"]["fixture_id"])
        game = games_by_id.get(fixture_id) or _target_game(run)
        lineup_status, lineup_evidence = _lineup_status(
            run,
            events,
            strict_roster=strict_roster,
            receipts=receipt_index,
            receipt_manifest_sha256=(receipt_readiness or {}).get("manifest_sha256"),
        )
        score = hybrid.score(
            _target_game(run),
            online_rows[fixture_id],
            player_champion=player_champion_rows[fixture_id],
            allow_context=lineup_status == "verified_preevent" or not strict_roster,
        )
        scored.append(
            {
                "schema_version": SCHEMA_VERSION,
                "fixture_id": fixture_id,
                "event_start": run["pregame"]["event_start"],
                "snapshot": {
                    "as_of": score["snapshot_as_of"],
                    "training_games": score["training_games"],
                },
                "lineup": {"status": lineup_status, "evidence": lineup_evidence},
                "score": score,
            }
        )
    scored.sort(key=lambda row: (row["event_start"], row["fixture_id"]))
    _write_jsonl(output_dir / "temporal-hybrid-scored-ledger.jsonl", scored)

    frozen_by_id = {str(row["pregame"]["fixture_id"]): row for row in frozen}
    evaluated: list[dict[str, Any]] = []
    for row in scored:
        outcome = target_outcomes.get(row["fixture_id"])
        source = frozen_by_id[row["fixture_id"]]
        if outcome is None:
            continue
        winner = normalize_team(_safe(outcome.get("WinTeam")))
        blue = normalize_team(str(source["pregame"]["blue"]["team"]))
        copy_row = dict(row)
        copy_row["actual_blue_win"] = int(winner == blue)
        evaluated.append(copy_row)
    pure_draft_metrics = _metrics(evaluated, "p_blue_draft")
    context_without_draft_metrics = _metrics(
        evaluated, "p_blue_context_without_draft"
    )
    contextual_metrics = _metrics(evaluated, "p_blue_context")
    evaluation = {
        "schema_version": SCHEMA_VERSION,
        "model_version": "temporal-hybrid-v1.3.0",
        "maps": len(evaluated),
        "snapshot_as_of": _rfc(first_day),
        "training_games": hybrid.training_games,
        "update_k": update_k,
        "strict_roster": strict_roster,
        "pure_draft": pure_draft_metrics,
        "context_without_draft": context_without_draft_metrics,
        "contextual": contextual_metrics,
        "incremental_draft_against_same_context": _metric_delta(
            contextual_metrics, context_without_draft_metrics
        ),
        "contextual_lineup_coverage": sum(
            1 for row in evaluated if row["lineup"]["status"] == "verified_preevent"
        ),
        "roster_source": roster_source,
        "artifact": artifact_path.name,
        "artifact_sha256": _sha_file(artifact_path),
    }
    _write_json(output_dir / "temporal-hybrid-evaluation.json", evaluation)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "availability_status": "adaptive_development_replay_not_independent",
        "source_hashes": {
            "frozen_ledger": _sha_file(run_dir / "frozen-ledger.jsonl"),
            "target_outcomes": _sha_file(run_dir / "normalized-outcome-rows.jsonl"),
            "prior_outcomes": _sha_file(run_dir / "autoresearch/raw/prior-games/normalized-prior-rows.jsonl"),
            "prior_drafts": _sha_file(run_dir / "autoresearch/raw/prior-drafts/normalized-prior-draft-rows.jsonl"),
            "runner": _sha_file(Path(__file__)),
            "roster_receipt_manifest": (receipt_readiness or {}).get("manifest_sha256"),
            "roster_receipt_file": (receipt_readiness or {}).get("receipt_file_sha256"),
            "legacy_roster_events": (
                _sha_file(roster_events_path) if roster_events_path is not None else None
            ),
        },
        "fit_policy": {
            "strict_before_cutoff": True,
            "alpha": DEFAULT_ALPHA,
            "half_life_days": DEFAULT_HALF_LIFE_DAYS,
            "blue_side_bonus": BLUE_SIDE_BONUS,
            "online_update_k": update_k,
            "player_champion_half_life_days": DEFAULT_HALF_LIFE_DAYS,
            "player_champion_features": list(PLAYER_CHAMPION_FEATURES),
            "incremental_draft_comparator": (
                "same fitted context with composition coefficients removed; "
                "intercept and blue-side constant retained"
            ),
        },
        "evaluation": evaluation,
        "score_written_before_outcome_evaluation": True,
        "claim_ceiling": {
            "adaptive_development_diagnostic": True,
            "independent_validation": False,
            "production_probability": False,
            "betting": False,
        },
    }
    manifest["manifest_sha256"] = sha256_canonical_object(manifest)
    _write_json(output_dir / "temporal-hybrid-manifest.json", manifest)
    return evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cadence-days", type=int, default=1)
    parser.add_argument("--roster-events", type=Path)
    parser.add_argument("--roster-receipt-manifest", type=Path)
    parser.add_argument("--strict-roster", action="store_true")
    parser.add_argument("--hybrid-output-dir", type=Path)
    parser.add_argument("--hybrid-update-k", type=float, default=0.20)
    parser.add_argument("--hybrid-only", action="store_true")
    args = parser.parse_args()
    result: dict[str, Any] = {}
    if not args.hybrid_only:
        result["temporal"] = run_temporal_backtest(
            args.run_dir,
            args.output_dir,
            cadence_days=args.cadence_days,
            roster_events_path=args.roster_events,
            roster_receipt_manifest_path=args.roster_receipt_manifest,
            strict_roster=args.strict_roster,
        )
    if args.hybrid_output_dir is not None:
        result["hybrid"] = run_hybrid_backtest(
            args.run_dir,
            args.hybrid_output_dir,
            update_k=args.hybrid_update_k,
            roster_events_path=args.roster_events,
            roster_receipt_manifest_path=args.roster_receipt_manifest,
            strict_roster=args.strict_roster,
        )
    if not result:
        raise SystemExit("provide --hybrid-output-dir or omit --hybrid-only")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
