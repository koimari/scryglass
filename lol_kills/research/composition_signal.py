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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from lol_kills.etl.aliases import normalize_champ, normalize_team
from lol_kills.etl.source_keys import canonical_source_game_key
from lol_kills.v2.champions.atoms.depth2_aggregate import (
    DEFAULT_ARTIFACT_PATH as DEPTH2_ARTIFACT_PATH,
    Depth2AggregateError,
    load_depth2_artifact,
)


SCHEMA_VERSION = "scryglass:composition-signal:v1"
MODEL_VERSION = "composition-signal-v3"
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
    "atomized_champion_effects",
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
PUBLIC_EVIDENCE = ("available", "atom_estimate", "limited", "unavailable")
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

# Round-5 frontier: mechanic corpus + strictly-prior features. The champion
# mechanic corpus is embedded (aggregate families/mechanics/relations per
# champion) so production never depends on the LCC repository at runtime.
ATOM_CORPUS_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "lol"
    / "v2"
    / "champions"
    / "atom-corpus-aggregate-v1.json"
)
ATOM_FAMILIES = (
    "crowd-control-mobility",
    "damage",
    "heal-shield",
    "interaction",
    "stack-transform-summon-resource",
    "vision-economy",
)
ATOM_MECHANIC_KEYS = (
    "execute", "revive", "transform", "summon", "stealth", "ward",
    "dash", "hook", "global", "shield", "heal", "cc",
)
_ATOM_SLUG_ALIASES = {
    "wukong": "monkeyking",
    "nunu & willump": "nunu",
    "renata glasc": "renata",
}


def _atom_slug(champion: Any) -> str:
    """Compact corpus slug for a champion display name."""
    name = str(champion or "").strip().casefold()
    if name in _ATOM_SLUG_ALIASES:
        return _ATOM_SLUG_ALIASES[name]
    return "".join(character for character in name if character.isalnum())


_ATOM_CORPUS: dict[str, dict[str, Any]] | None = None


def _atom_corpus() -> dict[str, dict[str, Any]]:
    """Per-champion aggregate corpus rows (families, mechanics, relations)."""
    global _ATOM_CORPUS
    if _ATOM_CORPUS is not None:
        return _ATOM_CORPUS
    payload: dict[str, Any] = {}
    try:
        raw = json.loads(ATOM_CORPUS_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and isinstance(raw.get("champions"), dict):
            payload = raw
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        payload = {}
    _ATOM_CORPUS = payload
    return payload


def _atom_zero_families() -> np.ndarray:
    return np.zeros(len(ATOM_FAMILIES))


def _atom_zero_mechanics() -> np.ndarray:
    return np.zeros(len(ATOM_MECHANIC_KEYS))


def _corpus_family_counts(champion: Any) -> np.ndarray:
    row = _atom_corpus().get("champions", {}).get(_atom_slug(champion))
    if not row:
        return _atom_zero_families()
    counts = row.get("families")
    if not isinstance(counts, list) or len(counts) != len(ATOM_FAMILIES):
        return _atom_zero_families()
    return np.asarray([float(value) for value in counts], dtype=float) / 5.0


def _corpus_mechanic_flags(champion: Any) -> np.ndarray:
    row = _atom_corpus().get("champions", {}).get(_atom_slug(champion))
    if not row:
        return _atom_zero_mechanics()
    flags = row.get("mechanics")
    if not isinstance(flags, list) or len(flags) != len(ATOM_MECHANIC_KEYS):
        return _atom_zero_mechanics()
    return np.asarray([float(value) for value in flags], dtype=float)


def _corpus_relation_targets(champion: Any) -> set[str]:
    row = _atom_corpus().get("champions", {}).get(_atom_slug(champion))
    if not row:
        return set()
    targets = row.get("relations")
    if not isinstance(targets, list):
        return set()
    return {str(value) for value in targets}


def _corpus_game_features(game: Mapping[str, Any]) -> np.ndarray:
    """Per-game corpus features: family diff (6), mechanic diff (12),
    per-role family diff (30), relations counter-coverage (1)."""
    blue_fam = _atom_zero_families()
    red_fam = _atom_zero_families()
    blue_mech = _atom_zero_mechanics()
    red_mech = _atom_zero_mechanics()
    blue_rel: set[str] = set()
    red_rel: set[str] = set()
    role_fam: list[np.ndarray] = []
    for role in ROLES:
        bf = _atom_zero_families()
        rf = _atom_zero_families()
        blue_champion = _champion(game.get("blue", {}).get(role, {}).get("champion"))
        red_champion = _champion(game.get("red", {}).get(role, {}).get("champion"))
        bf += _corpus_family_counts(blue_champion)
        rf += _corpus_family_counts(red_champion)
        blue_mech += _corpus_mechanic_flags(blue_champion)
        red_mech += _corpus_mechanic_flags(red_champion)
        blue_rel |= _corpus_relation_targets(blue_champion)
        red_rel |= _corpus_relation_targets(red_champion)
        role_fam.append(bf - rf)
        blue_fam += bf
        red_fam += rf
    blue_fam_keys = {ATOM_FAMILIES[i] for i, value in enumerate(blue_fam) if value > 0}
    red_fam_keys = {ATOM_FAMILIES[i] for i, value in enumerate(red_fam) if value > 0}
    blue_counters = float(len(blue_rel & red_fam_keys) / max(len(blue_rel), 1))
    red_counters = float(len(red_rel & blue_fam_keys) / max(len(red_rel), 1))
    return np.concatenate(
        [
            blue_fam - red_fam,
            blue_mech - red_mech,
            np.concatenate(role_fam),
            [blue_counters - red_counters],
        ]
    )


_ATOM_DEPTH2_PATH = DEPTH2_ARTIFACT_PATH
_ATOM_DEPTH2_CACHE: dict[str, dict[str, float]] | None = None
_ATOM_VECTOR_CACHE: dict[str, np.ndarray | None] = {}


def _atom_depth2_index() -> dict[str, dict[str, float]]:
    """The depth-2 numeric atom index (per-champion d2_* descriptors)."""
    global _ATOM_DEPTH2_CACHE
    if _ATOM_DEPTH2_CACHE is None:
        try:
            _ATOM_DEPTH2_CACHE = load_depth2_artifact(_ATOM_DEPTH2_PATH)
        except Depth2AggregateError:
            _ATOM_DEPTH2_CACHE = {}
    return _ATOM_DEPTH2_CACHE


_DEPTH2_KEYS_CACHE: list[str] | None = None

def _depth2_keys() -> list[str]:
    global _DEPTH2_KEYS_CACHE
    if _DEPTH2_KEYS_CACHE is None:
        index = _atom_depth2_index()
        keys: list[str] = []
        for entry in index.values():
            for key in entry:
                if key not in keys:
                    keys.append(key)
        _DEPTH2_KEYS_CACHE = sorted(keys)
    return list(_DEPTH2_KEYS_CACHE)


_ATOM_DEPTH3_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "lol"
    / "v2"
    / "champions"
    / "atom-corpus-aggregate-v3.json"
)
_ATOM_DEPTH3_CACHE: dict[str, dict[str, float]] | None = None
_ATOM_DEPTH4_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "lol"
    / "v2"
    / "champions"
    / "atom-corpus-aggregate-v4.json"
)
_DEPTH4_KEYS = (
    "d4_dmg_cd", "d4_burst_cd", "d4_cd_uptime", "d4_cd_x_uptime",
    "d4_recast_share", "d4_channel_share", "d4_cast_share", "d4_impact_share",
    "d4_travel_share", "d4_aftermath_share", "d4_persistent_share", "d4_chain_len",
)
_ATOM_DEPTH4_CACHE: dict[str, dict[str, float]] | None = None


def _atom_depth4_index() -> dict[str, dict[str, float]]:
    """The depth-4 atom index (per-champion d4_* ability/state descriptors).

    Built from atom-corpus-aggregate-v4.json; falls back to an empty index
    when the corpus is absent so older checkouts keep working.
    """
    global _ATOM_DEPTH4_CACHE
    if _ATOM_DEPTH4_CACHE is None:
        try:
            payload = json.loads(_ATOM_DEPTH4_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            payload = {}
        _ATOM_DEPTH4_CACHE = {
            str(key): {str(k): float(v) for k, v in dict(entry).items()}
            for key, entry in (payload.get("champions") or {}).items()
            if isinstance(entry, dict)
        }
    return _ATOM_DEPTH4_CACHE


def _depth4_keys() -> list[str]:
    index = _atom_depth4_index()
    keys: list[str] = []
    for entry in index.values():
        for key in entry:
            if key not in keys:
                keys.append(key)
    return sorted(keys)


def _depth4_game_row(game: Mapping[str, Any]) -> np.ndarray:
    index = _atom_depth4_index()

    def mean_of(side: str, key: str) -> float:
        values = [
            index.get(_atom_slug(str(game[side][role].get("champion") or "")), {}).get(key, 0.0)
            for role in ROLES
        ]
        return float(np.mean(values)) if values else 0.0

    return np.asarray([mean_of("blue", key) - mean_of("red", key) for key in _depth4_keys()], dtype=float)


# State-space team dynamics (Glicko-2-lite), strictly prior: the feature is
# read from the state BEFORE this game's outcome updates it.
_SS_KEYS = ("ss_mu_diff", "ss_sig_diff", "ss_p_diff", "ss_vol_diff")


def _build_ss_rows(ordered_games: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    rows: dict[str, np.ndarray] = {}
    state: dict[str, dict[str, float]] = {}

    def get_team(team: str) -> dict[str, float]:
        entry = state.get(team)
        if entry is None:
            entry = {"mu": 1500.0, "sig": 350.0, "vol": 0.5}
            state[team] = entry
        return entry

    for game in ordered_games:
        blue_team = str(game.get("blue_team") or "").strip()
        red_team = str(game.get("red_team") or "").strip()
        b = get_team(blue_team)
        r = get_team(red_team)
        scale = 400.0
        expected = 1.0 / (1.0 + 10 ** ((r["mu"] - b["mu"]) / scale))
        rows[str(game.get("game_uid"))] = np.asarray([
            b["mu"] - r["mu"],
            b["sig"] - r["sig"],
            expected - 0.5,
            b["vol"] - r["vol"],
        ], dtype=float)
        outcome = int(game.get("y") or 0)
        k = 0.35
        b["mu"] += k * scale * (outcome - expected)
        r["mu"] -= k * scale * (outcome - expected)
        residual = abs(outcome - expected)
        for entry in (b, r):
            entry["vol"] = 0.9 * entry["vol"] + 0.1 * residual
            entry["sig"] = max(40.0, entry["sig"] * 0.995)
    return rows


def _ss_game_row(game: Mapping[str, Any], rows: Mapping[str, np.ndarray]) -> np.ndarray:
    return rows.get(str(game.get("game_uid")), np.zeros(len(_SS_KEYS)))


def _atom_depth3_index() -> dict[str, dict[str, float]]:
    """The depth-3 atom index (per-champion d3_* state/cycle descriptors)."""
    global _ATOM_DEPTH3_CACHE
    if _ATOM_DEPTH3_CACHE is None:
        try:
            payload = json.loads(_ATOM_DEPTH3_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            payload = {}
        _ATOM_DEPTH3_CACHE = {
            str(key): {str(k): float(v) for k, v in dict(entry).items()}
            for key, entry in (payload.get("champions") or {}).items()
            if isinstance(entry, dict)
        }
    return _ATOM_DEPTH3_CACHE


def _depth3_keys() -> list[str]:
    index = _atom_depth3_index()
    keys: list[str] = []
    for entry in index.values():
        for key in entry:
            if key not in keys:
                keys.append(key)
    return sorted(keys)


def _depth3_game_row(game: Mapping[str, Any]) -> np.ndarray:
    index = _atom_depth3_index()

    def mean_of(side: str, key: str) -> float:
        values = [
            index.get(_atom_slug(str(game[side][role].get("champion") or "")), {}).get(key, 0.0)
            for role in ROLES
        ]
        return float(np.mean(values)) if values else 0.0

    return np.asarray([mean_of("blue", key) - mean_of("red", key) for key in _depth3_keys()], dtype=float)


def _champion_depth2(champion: str) -> dict[str, float]:
    return _atom_depth2_index().get(_atom_slug(champion), {})


def _cached_atom_vector(champion: str) -> np.ndarray | None:
    slug = _corpus_slug(champion)
    if slug not in _ATOM_VECTOR_CACHE:
        _ATOM_VECTOR_CACHE[slug] = _atom_feature_vector(champion)
    return _ATOM_VECTOR_CACHE[slug]


def _depth2_game_row(game: Mapping[str, Any]) -> np.ndarray:
    """Blue-minus-red per-descriptor means over the 5 picks (depth-2 numeric spine)."""
    blue = [_champion_depth2(game.get("blue", {}).get(role, {}).get("champion")) for role in ROLES]
    red = [_champion_depth2(game.get("red", {}).get(role, {}).get("champion")) for role in ROLES]

    def mean_of(entries: Sequence[Mapping[str, float]], key: str) -> float:
        values = [entry.get(key, 0.0) for entry in entries]
        return float(np.mean(values)) if values else 0.0

    return np.asarray([mean_of(blue, key) - mean_of(red, key) for key in _depth2_keys()], dtype=float)


class CompositionSignalError(RuntimeError):
    """Raised when a composition signal cannot be built safely."""


def _role(value: Any) -> str:
    raw = str(value or "").strip().casefold()
    return ROLE_ALIASES.get(raw, raw[:3])


from functools import lru_cache as _lru_cache

@_lru_cache(maxsize=65536)
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
    bans: dict[str, list[str]] = {}
    for side in ("Blue", "Red"):
        side_rows = group[group["_side"] == side]
        if len(side_rows) != 5:
            return None
        if side_rows["_role"].nunique() != 5:
            return None
        if side_rows["_team"].nunique() != 1:
            return None
        ban_slots = [f"ban{index}" for index in range(1, 6)]
        if all(column in side_rows.columns for column in ban_slots):
            first = side_rows.iloc[0]
            bans[side.lower()] = [
                _text(first.get(column))
                for column in ban_slots
                if _text(first.get(column))
            ]
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
            stats: dict[str, float] = {}
            for column in ("kills", "deaths", "damageshare", "cspm", "visionscore"):
                if column in group.columns:
                    stats[column] = _number(row.get(column), 0.0) or 0.0
            picks[role] = {"champion": champion, "player": player, "stats": stats}
            champions.append(champion)
            teams[side] = team
        sides[side] = picks
    player_stats: dict[str, dict[str, float]] = {}
    for side in ("Blue", "Red"):
        for role in ROLES:
            stats = sides[side][role].get("stats") or {}
            player_key = str(sides[side][role].get("player") or "").casefold()
            if player_key:
                player_stats[player_key] = {
                    "kills": float(stats.get("kills", 0.0)),
                    "deaths": float(stats.get("deaths", 0.0)),
                    "damageshare": float(stats.get("damageshare", 0.0)),
                    "cspm": float(stats.get("cspm", 0.0)),
                    "visionscore": float(stats.get("visionscore", 0.0)),
                }
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
        "bans": bans,
        "player_stats": player_stats,
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
    player_ratings: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Build only exact, role-complete, ten-champion games."""

    player_elo_lookup: dict[str, dict[str, float]] = {}
    if player_ratings is not None and not player_ratings.empty:
        columns = [c for c in ("game_uid", "player_mu_diff", "p_player_elo", "player_sigma_pair") if c in player_ratings.columns]
        if "game_uid" in columns:
            for _, row in player_ratings[columns].drop_duplicates("game_uid").iterrows():
                gid = canonical_source_game_key(row.get("game_uid"))
                if gid:
                    player_elo_lookup[str(gid)] = {
                        "diff": _number(row.get("player_mu_diff"), 0.0) or 0.0,
                        "p": _number(row.get("p_player_elo"), 0.5) or 0.5,
                        "sigma": _number(row.get("player_sigma_pair"), 0.0) or 0.0,
                    }

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
            "ban1", "ban2", "ban3", "ban4", "ban5",
            "kills", "deaths", "damageshare", "cspm", "visionscore",
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
    _ROLE_SLOT = {"top": 0, "jng": 1, "mid": 2, "bot": 3, "sup": 4}
    frame["_slot"] = frame["_side"].eq("Red").astype(np.int8) * 5 + frame["_role"].map(_ROLE_SLOT).astype(np.int8)
    frame = frame.sort_values(["_game_id", "_slot"]).reset_index(drop=True)
    gid = frame["_game_id"].to_numpy(dtype=object)
    slot = frame["_slot"].to_numpy(dtype=np.int8)
    pkey = frame["_player_key"].to_numpy(dtype=object)
    player = frame["_player"].to_numpy(dtype=object)
    team = frame["_team"].to_numpy(dtype=object)
    champ = frame["_champion"].to_numpy(dtype=object)
    result = frame["_result"].to_numpy(dtype=float)
    dates = frame["_date"].to_numpy(dtype="datetime64[ns]")
    ban_cols = [c for c in ("ban1", "ban2", "ban3", "ban4", "ban5") if c in frame.columns]
    bans_arr = {c: frame[c].to_numpy(dtype=object) for c in ban_cols}
    stat_cols = [c for c in ("kills", "deaths", "damageshare", "cspm", "visionscore") if c in frame.columns]
    stats_arr = {c: frame[c].to_numpy(dtype=object) for c in stat_cols}
    league_arr = frame["league"].to_numpy(dtype=object) if "league" in frame.columns else None
    patch_arr = frame["patch"].to_numpy(dtype=object) if "patch" in frame.columns else None
    series_arr = frame["grid_series_id"].to_numpy(dtype=object) if "grid_series_id" in frame.columns else None
    tourn_arr = frame["tournament"].to_numpy(dtype=object) if "tournament" in frame.columns else None
    # contiguous blocks per game (frame sorted by game_id); process
    # chronologically like the original ((max date, game id) order).
    starts = np.flatnonzero(np.concatenate(([True], gid[1:] != gid[:-1])))
    ends = np.concatenate((starts[1:], [len(gid)]))
    block_dates = np.maximum.reduceat(dates, starts)
    order = np.lexsort((gid[starts], block_dates))
    block_starts = starts[order]
    block_ends = ends[order]
    game_ids = gid[block_starts]

    def _update_exp(experience: dict[tuple[str, str], int], pkeys: np.ndarray, champs: np.ndarray) -> None:
        for pk, ch in zip(pkeys, champs):
            pk = str(pk or "")
            ch = _champion(ch)
            if pk and ch:
                experience[(pk, ch)] = experience.get((pk, ch), 0) + 1

    games = []
    experience: dict[tuple[str, str], int] = {}
    for gi, game_id in enumerate(game_ids):
        s = int(block_starts[gi])
        e = int(block_ends[gi])
        n = e - s
        if n != 10 or not np.array_equal(slot[s:e], np.arange(10, dtype=np.int8)):
            _update_exp(experience, pkey[s:e], champ[s:e])
            continue
        sides: dict[str, dict[str, dict[str, str]]] = {}
        teams: dict[str, str] = {}
        champions: list[str] = []
        valid = True
        for side, lo, hi in (("Blue", 0, 5), ("Red", 5, 10)):
            idx = np.arange(s + lo, s + hi)
            side_team = set(str(team[i]).strip() for i in idx)
            if len(side_team) != 1 or not any(side_team):
                valid = False
                break
            team_name = next(iter(side_team))
            pick_map: dict[str, dict[str, str]] = {}
            for role, i in zip(ROLES, idx):
                c = str(champ[i] or "")
                p = str(player[i] or "").strip()
                if not c or not p:
                    valid = False
                    break
                stats = {col: _number(stats_arr[col][i], 0.0) or 0.0 for col in stat_cols}
                pick_map[role] = {"champion": c, "player": p, "stats": stats}
                champions.append(c)
            if not valid:
                break
            teams[side] = team_name
            sides[side] = pick_map
        if not valid:
            _update_exp(experience, pkey[s:e], champ[s:e])
            continue
        if len(set(pkey[s:e])) != 10 or len(set(champions)) != 10 or not teams.get("Blue") or not teams.get("Red"):
            _update_exp(experience, pkey[s:e], champ[s:e])
            continue
        if teams["Blue"] == teams["Red"]:
            _update_exp(experience, pkey[s:e], champ[s:e])
            continue
        blue_results = result[s : s + 5]
        red_results = result[s + 5 : s + 10]
        if (
            np.isnan(blue_results).any()
            or np.isnan(red_results).any()
            or len(set(blue_results.tolist())) != 1
            or len(set(red_results.tolist())) != 1
            or float(blue_results[0]) not in (0.0, 1.0)
            or float(red_results[0]) != 1.0 - float(blue_results[0])
        ):
            _update_exp(experience, pkey[s:e], champ[s:e])
            continue
        date = pd.Timestamp(dates[s : s + 10].max(), tz="UTC")
        if pd.isna(date):
            _update_exp(experience, pkey[s:e], champ[s:e])
            continue
        strength_row = dict(strength.get(str(game_id)) or {})
        mu_diff = _number(strength_row.get("mu_diff"))
        sigma_pair = _number(strength_row.get("sigma_pair"))
        bans: dict[str, list[str]] = {}
        if ban_cols:
            bans["blue"] = [_text(bans_arr[c][s]) for c in ban_cols if _text(bans_arr[c][s])]
            bans["red"] = [_text(bans_arr[c][s + 5]) for c in ban_cols if _text(bans_arr[c][s + 5])]
        player_stats: dict[str, dict[str, float]] = {}
        for side in ("Blue", "Red"):
            for role in ROLES:
                stats = sides[side][role].get("stats") or {}
                player_key = str(sides[side][role].get("player") or "").casefold()
                if player_key:
                    player_stats[player_key] = {
                        "kills": float(stats.get("kills", 0.0)),
                        "deaths": float(stats.get("deaths", 0.0)),
                        "damageshare": float(stats.get("damageshare", 0.0)),
                        "cspm": float(stats.get("cspm", 0.0)),
                        "visionscore": float(stats.get("visionscore", 0.0)),
                    }
        first = s
        game = {
            "game_uid": str(game_id),
            "date": date,
            "league": _text(league_arr[first], "UNKNOWN").upper() if league_arr is not None else "UNKNOWN",
            "patch": _patch(patch_arr[first] if patch_arr is not None else None),
            "blue_team": teams["Blue"],
            "red_team": teams["Red"],
            "y": int(float(blue_results[0])),
            "blue": sides["Blue"],
            "red": sides["Red"],
            "bans": bans,
            "player_stats": player_stats,
            "mu_diff": mu_diff,
            "sigma_pair": sigma_pair,
            "controls_available": mu_diff is not None and sigma_pair is not None,
            "series_id": _text(series_arr[first]) if series_arr is not None else "",
            "tournament": _text(tourn_arr[first]) if tourn_arr is not None else "",
        }
        elo = player_elo_lookup.get(str(game_id))
        if elo:
            game["player_elo"] = elo
        blue_exp = sum(
            experience.get((str(pick.get("player") or "").casefold(), _champion(pick.get("champion"))), 0)
            for pick in game["blue"].values()
        )
        red_exp = sum(
            experience.get((str(pick.get("player") or "").casefold(), _champion(pick.get("champion"))), 0)
            for pick in game["red"].values()
        )
        game["blue_exp"] = blue_exp
        game["red_exp"] = red_exp
        for side in ("blue", "red"):
            for role in ROLES:
                pick = game[side][role]
                pick["experience"] = experience.get(
                    (str(pick.get("player") or "").casefold(), _champion(pick.get("champion"))), 0
                )
        games.append(game)
        _update_exp(experience, pkey[s:e], champ[s:e])
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
    names = {"control|mu_diff", "control|sigma_pair", "control|blue_side", "control|exp_diff"}
    for game in games:
        names.add(f"league|{game.get('league') or 'UNKNOWN'}")
        names.add(f"patch|{game.get('patch') or 'UNKNOWN'}")
        for side, sign in (("blue", 1), ("red", -1)):
            del sign
            for role in ROLES:
                names.add(f"draft|{role}|{_champion(game[side][role].get('champion'))}")
                names.add(f"draft|exp|{role}")
                for key in _atom_term_keys():
                    names.add(f"atom|{role}|{key}")
    controls = sorted(name for name in names if name.startswith("control|"))
    context = sorted(name for name in names if name.startswith(("league|", "patch|")))
    draft = sorted(name for name in names if name.startswith("draft|"))
    atom = sorted(name for name in names if name.startswith("atom|"))
    return controls + context + draft + atom


_MATRIX_ROW_CACHE: dict[tuple[Any, tuple[str, ...], bool], tuple[list[int], list[float]]] = {}

def _matrix(game_or_games: Any, names: Sequence[str], *, include_draft: bool) -> sparse.csr_matrix:
    games: Sequence[Mapping[str, Any]] = game_or_games
    columns = {name: index for index, name in enumerate(names)}
    names_tuple = tuple(names)
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for row_index, game in enumerate(games):
        key = (id(game), names_tuple, include_draft)
        cached = _MATRIX_ROW_CACHE.get(key)
        if cached is None:
            if len(_MATRIX_ROW_CACHE) > 60000:
                _MATRIX_ROW_CACHE.clear()
            game_rows: list[int] = []
            game_cols: list[int] = []
            game_vals: list[float] = []
            _matrix_row(game, names, columns, game_rows, game_cols, game_vals, include_draft=include_draft)
            cached = (game_cols, game_vals)
            _MATRIX_ROW_CACHE[key] = cached
        for column_index, value in zip(cached[0], cached[1]):
            rows.append(row_index)
            cols.append(column_index)
            values.append(value)
    return sparse.csr_matrix(
        (np.asarray(values, dtype=float), (np.asarray(rows, dtype=int), np.asarray(cols, dtype=int))),
        shape=(len(games), len(names)),
        dtype=float,
    )


def _matrix_row(
    game: Mapping[str, Any],
    names: Sequence[str],
    columns: Mapping[str, int],
    rows: list[int],
    cols: list[int],
    values: list[float],
    *,
    include_draft: bool,
) -> None:
    def add(name: str, value: float) -> None:
        column = columns.get(name)
        if column is None:
            return
        rows.append(0)
        cols.append(column)
        values.append(float(value))

    add("control|mu_diff", float(game.get("mu_diff") or 0.0) / 400.0)
    add("control|sigma_pair", float(game.get("sigma_pair") or 0.0) / 120.0)
    add("control|blue_side", 1.0)
    add(f"league|{game.get('league') or 'UNKNOWN'}", 1.0)
    add(f"patch|{game.get('patch') or 'UNKNOWN'}", 1.0)
    if include_draft:
        add(
            "control|exp_diff",
            (float(game.get("blue_exp") or 0.0) - float(game.get("red_exp") or 0.0)) / 100.0,
        )
        for role in ROLES:
            add(
                f"draft|exp|{role}",
                (
                    float(game["blue"][role].get("experience") or 0.0)
                    - float(game["red"][role].get("experience") or 0.0)
                )
                / 50.0,
            )
        for side, sign in (("blue", 1.0), ("red", -1.0)):
            for role in ROLES:
                champion = _champion(game[side][role].get("champion"))
                add(f"draft|{role}|{champion}", sign)
                for key in _atom_term_keys():
                    descriptor = _atom_desc_value(champion, key)
                    if descriptor:
                        add(f"atom|{role}|{key}", sign * descriptor)


@dataclass(frozen=True)
class FittedCompositionModel:
    model_version: str
    fit_through: str
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    support: dict[str, int]
    train_games: int
    regularization_c: float = REGULARIZATION_C
    worker_commit: str | None = None
    atom_prior: dict[str, Any] | None = None
    code_digest: str | None = None

    def coefficient(self, role: str, champion: str) -> float:
        key = f"draft|{role}|{_champion(champion)}"
        try:
            index = self.feature_names.index(key)
        except ValueError:
            return _atom_prior_coefficient(self, role, champion)
        return float(self.coefficients[index])

    def pick_contribution(self, role: str, champion: str) -> float:
        """Per-pick production contribution: the champion term plus the
        coefficient-weighted atomized descriptors (d2/d3/d4)."""
        value = self.coefficient(role, champion)
        for key in _atom_term_keys():
            name = f"atom|{role}|{key}"
            try:
                index = self.feature_names.index(name)
            except ValueError:
                continue
            descriptor = _atom_desc_value(champion, key)
            if descriptor:
                value += float(self.coefficients[index]) * descriptor
        return value

    def has_atom_prior(self) -> bool:
        return isinstance(self.atom_prior, dict) and bool(self.atom_prior)

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
            "code_digest": self.code_digest,
            "atom_prior": self.atom_prior,
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
            regularization_c=float(payload.get("regularization_c") or REGULARIZATION_C),
            worker_commit=str(payload.get("worker_commit") or "") or None,
            code_digest=str(payload.get("code_digest") or "") or None,
            atom_prior=(
                dict(payload["atom_prior"])
                if isinstance(payload.get("atom_prior"), dict)
                else None
            ),
        )


def _fit_model(
    games: Sequence[Mapping[str, Any]],
    *,
    names: Sequence[str],
    include_draft: bool = True,
    min_training_games: int = MIN_TRAINING_GAMES,
    regularization_c: float = REGULARIZATION_C,
    worker_commit: str | None = None,
) -> FittedCompositionModel | None:
    usable = [game for game in games if game.get("controls_available", False)]
    if len(usable) < min_training_games or len({int(game["y"]) for game in usable}) < 2:
        return None
    model = LogisticRegression(
        C=regularization_c,
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
    atom_prior = _fit_atom_prior_from_coefficients(
        tuple(names),
        tuple(float(value) for value in model.coef_[0]),
        support,
        min_support_games=ATOM_PRIOR_MIN_SUPPORT,
    ) if include_draft else None
    return FittedCompositionModel(
        model_version=MODEL_VERSION if include_draft else f"{MODEL_VERSION}:baseline",
        fit_through=fit_through,
        feature_names=tuple(names),
        coefficients=tuple(float(value) for value in model.coef_[0]),
        intercept=float(model.intercept_[0]),
        support=support,
        train_games=len(usable),
        regularization_c=float(regularization_c),
        worker_commit=worker_commit,
        code_digest=_composition_code_digest(),
        atom_prior=atom_prior,
    )


_ATOM_AGGREGATE_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "lol"
    / "v2"
    / "champions"
    / "atom-corpus-aggregate-v1.json"
)


def _corpus_slug(value: str) -> str:
    """Lowercase slug matching the LCC corpus keys (Lee Sin -> leesin, K'Sante -> ksante)."""
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _atom_feature_vector(champion: str) -> np.ndarray | None:
    """The 18-dim atom aggregate vector (6 family counts + 12 mechanic flags)."""
    try:
        payload = json.loads(_ATOM_AGGREGATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    entry = (payload.get("champions") or {}).get(_corpus_slug(champion))
    if not isinstance(entry, dict):
        return None
    families = entry.get("families") or []
    mechanics = entry.get("mechanics") or []
    vector = [float(value) for value in families] + [float(value) for value in mechanics]
    if len(vector) != 18:
        return None
    return np.asarray(vector, dtype=float)


ATOM_PRIOR_MIN_SUPPORT = 10


def _fit_atom_prior_from_coefficients(
    feature_names: Sequence[str],
    coefficients: Sequence[float],
    support: Mapping[str, int],
    *,
    min_support_games: int = MIN_SUPPORT_GAMES,
) -> dict[str, Any] | None:
    """Per-role ridge mapping atom features to draft coefficients for unseen picks."""

    names = tuple(feature_names)
    role_models: dict[str, Any] = {}
    for role in ROLES:
        xs: list[np.ndarray] = []
        ys: list[float] = []
        for key in names:
            prefix = f"draft|{role}|"
            if not key.startswith(prefix):
                continue
            champion = key[len(prefix):]
            if support.get(f"{role}|{champion}", 0) < min_support_games:
                continue
            vector = _atom_feature_vector(champion)
            if vector is None:
                continue
            xs.append(vector)
            ys.append(float(coefficients[names.index(key)]))
        if len(xs) < 8:
            continue
        matrix = np.asarray(xs, dtype=float)
        targets = np.asarray(ys, dtype=float)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std[std == 0] = 1.0
        normalized = (matrix - mean) / std
        ridge = Ridge(alpha=1.0, random_state=461)
        ridge.fit(normalized, targets)
        role_models[role] = {
            "mean": [float(value) for value in mean],
            "std": [float(value) for value in std],
            "coef": [float(value) for value in ridge.coef_],
            "intercept": float(ridge.intercept_),
            "train_pairs": len(xs),
        }
    return role_models or None


def _corpus_champions() -> list[str]:
    try:
        payload = json.loads(_ATOM_AGGREGATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return []
    champions = payload.get("champions") or {}
    return sorted(str(value) for value in champions)


def _atom_prior_coefficient(model: FittedCompositionModel, role: str, champion: str) -> float:
    """Atom-estimated draft coefficient for a pick with no historical support."""

    if not model.has_atom_prior():
        return 0.0
    entry = model.atom_prior.get(role)
    if not isinstance(entry, dict):
        return 0.0
    vector = _atom_feature_vector(champion)
    if vector is None:
        return 0.0
    mean = np.asarray(entry.get("mean") or [], dtype=float)
    std = np.asarray(entry.get("std") or [], dtype=float)
    coef = np.asarray(entry.get("coef") or [], dtype=float)
    if len(vector) != len(mean) or len(vector) != len(coef):
        return 0.0
    normalized = (vector - mean) / np.where(std == 0, 1.0, std)
    return float(np.dot(normalized, coef) + float(entry.get("intercept") or 0.0))


def _select_regularization(
    games: Sequence[Mapping[str, Any]],
    *,
    names: Sequence[str],
    candidates: Sequence[float],
    internal_fraction: float,
    min_training_games: int,
    worker_commit: str | None,
) -> float:
    """Pick the draft-model regularization strength on an internal date split.

    The most recent `internal_fraction` of the training fold (by calendar
    date) serves as an internal validation set; every candidate C is fitted
    on the earlier part only. The winning C is returned and then used to
    refit the full training fold, so no validation-window game influences
    the choice.
    """

    cutoff = max(int(len(games) * (1.0 - internal_fraction)), min_training_games)
    fit_games = list(games)[:cutoff]
    check_games = list(games)[cutoff:]
    if len(check_games) < 8 or len({int(game["y"]) for game in check_games}) < 2:
        return REGULARIZATION_C
    best_c = REGULARIZATION_C
    best_brier = float("inf")
    check_y = [int(game["y"]) for game in check_games]
    for candidate in candidates:
        model = _fit_model(
            fit_games,
            names=names,
            include_draft=True,
            min_training_games=min_training_games,
            regularization_c=candidate,
            worker_commit=worker_commit,
        )
        if model is None:
            continue
        probabilities = [_probability(model.logit(game, include_draft=True)) for game in check_games]
        check_brier = brier_score_loss(
            check_y, np.clip(np.asarray(probabilities, dtype=float), 1e-5, 1 - 1e-5)
        )
        if check_brier < best_brier:
            best_brier = check_brier
            best_c = candidate
    return best_c


def _select_history_regularization(
    train: Sequence[Mapping[str, Any]],
    history_train_x: np.ndarray,
    *,
    candidates: Sequence[float],
    internal_fraction: float,
    min_training_games: int,
) -> float:
    """Pick the team-history model regularization on an internal date split.

    The most recent `internal_fraction` of the training fold (by position in
    the strictly-lagged history sequence) is held out; every candidate C is
    fit on the earlier part and scored on the tail with Brier. The winning C
    is used for the full-training-fold refit. No validation game is used.
    """

    cutoff = max(int(len(train) * (1.0 - internal_fraction)), min_training_games)
    if cutoff >= len(train) or len(train) - cutoff < 8:
        return 1.0
    fit_y = [int(game["y"]) for game in train[:cutoff]]
    check_y = [int(game["y"]) for game in train[cutoff:]]
    if len(set(check_y)) < 2:
        return 1.0
    best_c = 1.0
    best_brier = float("inf")
    for candidate in candidates:
        model = LogisticRegression(C=candidate, solver="liblinear", max_iter=2000, random_state=461)
        model.fit(history_train_x[:cutoff], fit_y)
        probabilities = model.predict_proba(history_train_x[cutoff:])[:, 1]
        check_brier = brier_score_loss(
            check_y, np.clip(np.asarray(probabilities, dtype=float), 1e-5, 1 - 1e-5)
        )
        if check_brier < best_brier:
            best_brier = check_brier
            best_c = candidate
    return best_c


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


def _composition_code_digest() -> str:
    """Digest the scorer and every atom corpus used by the fit.

    The eval/checkpoint caches must be invalidated when this code changes,
    not merely when the worker deploys. The atom corpora are fit-time model
    inputs, so a corpus edit must invalidate the checkpoint as well. Missing
    files stay part of the digest as an explicit marker.
    """
    inputs = (
        ("composition_signal.py", Path(__file__)),
        ("atom-corpus-v1", ATOM_CORPUS_PATH),
        ("atom-corpus-v2", _ATOM_DEPTH2_PATH),
        ("atom-corpus-v3", _ATOM_DEPTH3_PATH),
        ("atom-corpus-v4", _ATOM_DEPTH4_PATH),
    )
    material: list[tuple[str, str]] = []
    for label, path in inputs:
        try:
            content = path.read_bytes()
        except OSError:
            content = b"<missing>"
        material.append((label, hashlib.sha256(content).hexdigest()))
    encoded = json.dumps(material, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


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
        cache_key = _cache_key(training_digest, cutoff, names=training_names, worker_commit=worker_commit)
        stable_key = _cache_key(training_digest, cutoff, names=training_names, worker_commit=None)
        for candidate_key in (cache_key, stable_key):
            candidate_path = cache_dir / "checkpoints" / f"{candidate_key}.json"
            if not candidate_path.exists():
                continue
            try:
                cached = FittedCompositionModel.from_json(
                    json.loads(candidate_path.read_text(encoding="utf-8"))
                )
                if cached.worker_commit == worker_commit or (
                    cached.code_digest and cached.code_digest == _composition_code_digest()
                ):
                    return cached, True
            except (OSError, ValueError, KeyError, TypeError, CompositionSignalError):
                pass
        path = cache_dir / "checkpoints" / f"{stable_key}.json"
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
            coefficient = model.pick_contribution(role, champion)
            supported = support >= min_support_games
            if supported:
                evidence_status = "available"
                side_signals[side] += coefficient
                side_support[side] += support
            elif model.has_atom_prior():
                evidence_status = "atom_estimate"
                side_signals[side] += coefficient
            else:
                evidence_status = "limited"
                limited = True
            picks.append(
                {
                    "side": side,
                    "role": role,
                    "champion": champion,
                    "contribution": _json_number(coefficient) if evidence_status != "limited" else None,
                    "prior_role_games": support,
                    "evidence_status": evidence_status,
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
        elif evidence == "atom_estimate":
            if support >= min_support_games or contribution is None:
                raise CompositionSignalError("atom_estimate composition pick is malformed")
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
        if any(evidence not in {"available", "atom_estimate"} for evidence in evidence_statuses):
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
    """Strictly-lagged team history features for every game in order.

    Returns an (n, 3) matrix with columns: (1) rolling mean of the team's
    prior draft signal, (2) recency-weighted win-rate momentum of the two
    teams, (3) log prior games played by both teams. Every feature at row i
    is computed only from matches strictly before position i.
    """

    draft_mean: dict[str, tuple[float, int]] = {}
    momentum: dict[str, float] = {}
    games_count: dict[str, int] = {}
    rows: list[tuple[float, float, float]] = []
    for game in games:
        blue_team = normalize_team(str(game["blue_team"]))
        red_team = normalize_team(str(game["red_team"]))
        blue_signal = sum(
            model.coefficient(role, game["blue"][role]["champion"]) for role in ROLES
        )
        red_signal = sum(
            model.coefficient(role, game["red"][role]["champion"]) for role in ROLES
        )
        blue_prior_mean, blue_prior_count = draft_mean.get(blue_team, (0.0, 0))
        red_prior_mean, red_prior_count = draft_mean.get(red_team, (0.0, 0))
        shrink = 5.0
        blue_draft = (
            blue_prior_mean * blue_prior_count / (blue_prior_count + shrink)
            if blue_prior_count
            else 0.0
        )
        red_draft = (
            red_prior_mean * red_prior_count / (red_prior_count + shrink)
            if red_prior_count
            else 0.0
        )
        rows.append(
            (
                float(blue_draft - red_draft),
                float(momentum.get(blue_team, 0.0) - momentum.get(red_team, 0.0)),
                float(np.log1p(games_count.get(blue_team, 0) + games_count.get(red_team, 0))),
            )
        )
        blue_count, red_count = games_count.get(blue_team, 0), games_count.get(red_team, 0)
        blue_total, red_total = blue_prior_count, red_prior_count
        blue_sum, red_sum = blue_prior_mean * blue_prior_count, red_prior_mean * red_prior_count
        draft_mean[blue_team] = ((blue_sum + blue_signal) / (blue_total + 1), blue_total + 1)
        draft_mean[red_team] = ((red_sum + red_signal) / (red_total + 1), red_total + 1)
        alpha = 0.2
        outcome = float(int(game["y"]))
        momentum[blue_team] = alpha * outcome + (1.0 - alpha) * momentum.get(blue_team, 0.5)
        momentum[red_team] = alpha * (1.0 - outcome) + (1.0 - alpha) * momentum.get(red_team, 0.5)
        games_count[blue_team] = blue_count + 1
        games_count[red_team] = red_count + 1
    return np.asarray(rows, dtype=float)


def _recalibrate_history_probabilities(
    train: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    draft: FittedCompositionModel,
    train_x: np.ndarray,
    validation_x: np.ndarray,
    history_model: LogisticRegression,
    *,
    folds: int = 4,
    shrink: float = 0.5,
    fold_c: float = 1.0,
) -> np.ndarray:
    """Affine-recalibrate the team-history model using training-fold data only.

    A fresh draft model is fit on the earlier part of the training fold and
    the history model is refit on that same earlier part; the held-out tail
    of the training fold provides out-of-fold-style logits for the affine
    recalibrator, which is then applied to the validation logits.
    """

    _ = folds
    ordered = [dict(game) for game in train if game.get("controls_available", False)]
    cutoff = max(int(len(ordered) * 0.8), 8)
    fit_games = ordered[:cutoff]
    cal_games = ordered[cutoff:]
    if len(cal_games) < 8 or len({int(game["y"]) for game in cal_games}) < 2:
        return history_model.predict_proba(validation_x)[:, 1]
    names = _feature_names(fit_games)
    fold_draft = _fit_model(
        fit_games,
        names=names,
        include_draft=True,
        regularization_c=draft.regularization_c,
    )
    if fold_draft is None:
        return history_model.predict_proba(validation_x)[:, 1]
    sequence = fit_games + cal_games
    history = _history_features(sequence, fold_draft)
    fit_history = history[: len(fit_games)]
    cal_history = history[len(fit_games) :]
    fold_model = LogisticRegression(C=fold_c, solver="liblinear", max_iter=2000, random_state=461)
    fold_train_x = np.column_stack(
        [
            _matrix(fit_games, fold_draft.feature_names, include_draft=True) @ np.asarray(fold_draft.coefficients),
            fit_history,
        ]
    )
    fold_model.fit(fold_train_x, [int(game["y"]) for game in fit_games])
    cal_x = np.column_stack(
        [
            _matrix(cal_games, fold_draft.feature_names, include_draft=True) @ np.asarray(fold_draft.coefficients),
            cal_history,
        ]
    )
    cal_probabilities = fold_model.predict_proba(cal_x)[:, 1]
    cal_logits = np.log(np.clip(cal_probabilities, 1e-5, 1 - 1e-5) / np.clip(1 - cal_probabilities, 1e-5, 1 - 1e-5))
    calibrator = LogisticRegression(C=1e6, solver="liblinear", max_iter=1000)
    calibrator.fit(cal_logits.reshape(-1, 1), [int(game["y"]) for game in cal_games])
    a = float(calibrator.intercept_[0])
    b = float(calibrator.coef_[0][0])
    # Shrink the recalibrator toward identity so a small calibration tail
    # cannot over-correct a single window.
    a_eff = shrink * a
    b_eff = 1.0 + shrink * (b - 1.0)
    raw = history_model.predict_proba(validation_x)[:, 1]
    raw_logits = np.log(np.clip(raw, 1e-5, 1 - 1e-5) / np.clip(1 - raw, 1e-5, 1 - 1e-5))
    return 1.0 / (1.0 + np.exp(-np.clip(a_eff + b_eff * raw_logits, -30.0, 30.0)))


def _apply_oof_recalibration(
    train: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    baseline: FittedCompositionModel,
    draft: FittedCompositionModel,
    *,
    folds: int = 4,
) -> tuple[list[float], list[float]]:
    """Recalibrate using out-of-fold predictions from the training fold only.

    The training fold is split into `folds` chronological blocks. For each
    block the draft and baseline models are refit on the other blocks and
    used to predict the held-out block, giving out-of-fold logits that
    mimic the model's behavior on unseen data. An affine logit recalibrator
    is fit on those out-of-fold predictions and applied unchanged to the
    validation window. No validation game is used.
    """

    ordered = sorted(
        (dict(game) for game in train if game.get("controls_available", False)),
        key=lambda item: (_timestamp(item["date"]), str(item["game_uid"])),
    )
    if len(ordered) < 2 * folds or len({int(game["y"]) for game in ordered}) < 2:
        return (
            [_probability(baseline.logit(game, include_draft=False)) for game in validation],
            [_probability(draft.logit(game, include_draft=True)) for game in validation],
        )
    names = tuple(baseline.feature_names)
    blocks = [
        ordered[index * len(ordered) // folds : (index + 1) * len(ordered) // folds]
        for index in range(folds)
    ]
    oof_draft_logits: list[float] = []
    oof_baseline_logits: list[float] = []
    oof_y: list[int] = []
    for block_index, block in enumerate(blocks):
        fit_games = [game for other_index, other in enumerate(blocks) if other_index != block_index for game in other]
        fit_names = _feature_names(fit_games)
        fold_draft = _fit_model(
            fit_games,
            names=fit_names,
            include_draft=True,
            regularization_c=draft.regularization_c,
        )
        fold_baseline = _fit_model(
            fit_games,
            names=fit_names,
            include_draft=False,
            regularization_c=draft.regularization_c,
        )
        if fold_draft is None or fold_baseline is None:
            continue
        for game in block:
            oof_draft_logits.append(fold_draft.logit(game, include_draft=True))
            oof_baseline_logits.append(fold_baseline.logit(game, include_draft=False))
            oof_y.append(int(game["y"]))
    if len(set(oof_y)) < 2 or len(oof_y) < 12:
        return (
            [_probability(baseline.logit(game, include_draft=False)) for game in validation],
            [_probability(draft.logit(game, include_draft=True)) for game in validation],
        )

    def fit_transform(logits: Sequence[float], outcomes: Sequence[int]) -> tuple[float, float]:
        calibrator = LogisticRegression(C=1e6, solver="liblinear", max_iter=1000)
        calibrator.fit(np.asarray(logits, dtype=float).reshape(-1, 1), np.asarray(outcomes, dtype=np.int8))
        return float(calibrator.intercept_[0]), float(calibrator.coef_[0][0])

    baseline_a, baseline_b = fit_transform(oof_baseline_logits, oof_y)
    draft_a, draft_b = fit_transform(oof_draft_logits, oof_y)
    return (
        [
            _probability(baseline_a + baseline_b * baseline.logit(game, include_draft=False))
            for game in validation
        ],
        [_probability(draft_a + draft_b * draft.logit(game, include_draft=True)) for game in validation],
    )


def _match_delta_intervals(
    windows: Sequence[Mapping[str, Any]],
    *,
    reps: int,
    seed: int,
    label: str,
) -> dict[str, dict[str, float | None]]:
    """Window-stratified bootstrap of per-match paired score deltas.

    For every validation match the paired delta (candidate - reference) is
    computed for brier and log loss. Each window is resampled with
    replacement at its own size and the pooled mean delta across windows is
    the bootstrap statistic, giving a 95% percentile interval.
    """

    if label == "history_vs_draft":
        reference_key, candidate_key = "draft_augmented", "draft_plus_team_history"
    else:
        reference_key, candidate_key = "baseline", "draft_augmented"
    window_pairs: list[tuple[list[float], list[float], list[float]]] = []
    for window in windows:
        reference = (window.get(reference_key) or {}).get("probabilities")
        candidate = (window.get(candidate_key) or {}).get("probabilities")
        outcomes = (window.get(candidate_key) or {}).get("outcomes")
        if reference is None or candidate is None or outcomes is None:
            return {"brier_delta": {"lower": None, "upper": None}, "log_loss_delta": {"lower": None, "upper": None}}
        window_pairs.append((list(outcomes), list(reference), list(candidate)))
    if not window_pairs:
        return {"brier_delta": {"lower": None, "upper": None}, "log_loss_delta": {"lower": None, "upper": None}}
    rng = np.random.default_rng(seed)
    brier_means: list[float] = []
    ll_means: list[float] = []
    for _ in range(reps):
        brier_total = 0.0
        ll_total = 0.0
        count = 0
        for outcomes, reference, candidate in window_pairs:
            indexes = rng.integers(0, len(outcomes), size=len(outcomes))
            y = np.asarray(outcomes)[indexes]
            base = np.clip(np.asarray(reference)[indexes], 1e-5, 1 - 1e-5)
            comp = np.clip(np.asarray(candidate)[indexes], 1e-5, 1 - 1e-5)
            brier_total += float(np.sum((comp - y) ** 2) - np.sum((base - y) ** 2))
            ll_total += float(np.sum(-y * np.log(comp)) - np.sum(-y * np.log(base)))
            count += len(indexes)
        brier_means.append(brier_total / count)
        ll_means.append(ll_total / count)
    array = np.asarray(list(zip(brier_means, ll_means)), dtype=float)
    return {
        "brier_delta": {
            "lower": round(float(np.quantile(array[:, 0], 0.025)), 6),
            "upper": round(float(np.quantile(array[:, 0], 0.975)), 6),
        },
        "log_loss_delta": {
            "lower": round(float(np.quantile(array[:, 1], 0.025)), 6),
            "upper": round(float(np.quantile(array[:, 1], 0.975)), 6),
        },
    }


# ---------------------------------------------------------------------------
# Round-5 frontier: bans, mastery rows, strictly-prior features, corpus, CatBoost
# ---------------------------------------------------------------------------


def _ban_table(games: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, list[str]]]:
    """game_uid -> {'blue': [...], 'red': [...]} raw ban names from the games."""
    table: dict[str, dict[str, list[str]]] = {}
    for game in games:
        bans = game.get("bans")
        if isinstance(bans, dict) and bans.get("blue") and bans.get("red"):
            table[str(game["game_uid"])] = {
                "blue": [str(value) for value in bans.get("blue", [])],
                "red": [str(value) for value in bans.get("red", [])],
            }
    return table


def _exp_rows_log1p(games: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Strictly-prior per-role player-champion mastery, log1p scaled."""
    prior: dict[tuple[str, str], int] = {}
    rows: list[list[float]] = []
    for game in games:
        blue = [
            prior.get(
                (str(game["blue"][role]["player"]).casefold(),
                 _champion(game["blue"][role]["champion"])), 0
            )
            for role in ROLES
        ]
        red = [
            prior.get(
                (str(game["red"][role]["player"]).casefold(),
                 _champion(game["red"][role]["champion"])), 0
            )
            for role in ROLES
        ]
        rows.append([
            (np.log1p(blue[index]) - np.log1p(red[index])) / np.log1p(40.0)
            for index in range(len(ROLES))
        ])
        for role in ROLES:
            for side in ("blue", "red"):
                key = (
                    str(game[side][role]["player"]).casefold(),
                    _champion(game[side][role]["champion"]),
                )
                prior[key] = prior.get(key, 0) + 1
    return np.asarray(rows, dtype=float)


_BAN_TABLE_CACHE: dict[str, dict[str, list[str]]] | None = None


def _bans_v2_feature_builder(
    train: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    draft: FittedCompositionModel,
) -> Any:
    """Ban tax + per-role ban coverage + ban-rate edge + mastery hits.

    Returns a callable rows(games) -> (n, 8) strictly-prior ban features.
    """
    global _BAN_TABLE_CACHE
    if _BAN_TABLE_CACHE is None:
        _BAN_TABLE_CACHE = _ban_table([*train, *validation])
    bans = _BAN_TABLE_CACHE
    role_order = ROLES
    ban_counts: dict[str, int] = {}
    pick_counts: dict[str, int] = {}
    mastery: dict[tuple[str, str], int] = {}
    champion_roles: dict[str, list[str]] = {}
    for game in train:
        game_bans = bans.get(str(game["game_uid"]))
        if game_bans:
            for side in ("blue", "red"):
                for champion in game_bans[side]:
                    ban_counts[_champion(champion)] = ban_counts.get(_champion(champion), 0) + 1
        for side in ("blue", "red"):
            for role in role_order:
                champion = _champion(game[side][role]["champion"])
                player = str(game[side][role]["player"]).casefold()
                pick_counts[champion] = pick_counts.get(champion, 0) + 1
                mastery[(player, champion)] = mastery.get((player, champion), 0) + 1
                champion_roles.setdefault(champion, []).append(role)
    top_champion: dict[str, str] = {}
    top_count: dict[str, int] = {}
    for (player, champion), count in mastery.items():
        if count > top_count.get(player, 0):
            top_champion[player] = champion
            top_count[player] = count
    role_top3: dict[str, list[str]] = {}
    for champion, roles in champion_roles.items():
        for role in set(roles):
            role_top3.setdefault(role, []).append((pick_counts.get(champion, 0), champion))
    for role in role_top3:
        role_top3[role] = [champion for _, champion in sorted(role_top3[role], reverse=True)[:3]]
    total_bans = max(sum(ban_counts.values()), 1)
    total_picks = max(sum(pick_counts.values()), 1)
    meta = sorted(pick_counts, key=pick_counts.get, reverse=True)[:20]
    meta_value = {champion: pick_counts.get(champion, 0) for champion in meta}

    def rows(games: Sequence[Mapping[str, Any]]) -> np.ndarray:
        out: list[list[float]] = []
        for game in games:
            game_bans = bans.get(str(game["game_uid"]))
            blue_bans = [_champion(value) for value in (game_bans["blue"] if game_bans else [])]
            red_bans = [_champion(value) for value in (game_bans["red"] if game_bans else [])]
            ban_rate_diff = (
                sum(ban_counts.get(c, 0) for c in blue_bans)
                - sum(ban_counts.get(c, 0) for c in red_bans)
            ) / total_bans
            ban_tax_diff = (
                sum(meta_value.get(c, 0) for c in blue_bans)
                - sum(meta_value.get(c, 0) for c in red_bans)
            ) / total_picks
            coverage: list[float] = []
            for role in role_order:
                top = role_top3.get(role, [])
                blue_cov = sum(1 for c in blue_bans if c in top)
                red_cov = sum(1 for c in red_bans if c in top)
                coverage.append(float(blue_cov - red_cov))
            blue_hits = sum(
                mastery.get(
                    (str(game["blue"][role]["player"]).casefold(),
                     _champion(game["blue"][role]["champion"])), 0
                )
                for role in role_order
                if _champion(game["blue"][role]["champion"]) in red_bans
            )
            red_hits = sum(
                mastery.get(
                    (str(game["red"][role]["player"]).casefold(),
                     _champion(game["red"][role]["champion"])), 0
                )
                for role in role_order
                if _champion(game["red"][role]["champion"]) in blue_bans
            )
            mastery_hit_diff = (blue_hits - red_hits) / max(sum(mastery.values()), 1)
            out.append([
                float(ban_rate_diff), float(ban_tax_diff), *coverage, float(mastery_hit_diff),
            ])
        return np.asarray(out, dtype=float)

    return rows


_FRONTIER_NAMES: list[str] = []
_FRONTIER: dict[str, np.ndarray] = {}


def _frontier_names() -> list[str]:
    return list(_FRONTIER_NAMES)


def _frontier_rows(games_list: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if not _FRONTIER_NAMES:
        return np.zeros((len(games_list), 0))
    rows = []
    for game in games_list:
        row = _FRONTIER.get(str(game["game_uid"]))
        rows.append(
            row if row is not None else np.zeros(len(_FRONTIER_NAMES))
        )
    return np.asarray(rows, dtype=float)




# Atom descriptor terms for the production linear model.  These are the
# per-pick champion descriptors (depth-2/3/4), looked up alias-aware so
# Wukong/Renata Glasc/Nunu & Willump resolve to their corpus keys.
_ATOM_TERM_KEYS: tuple[str, ...] | None = None


def _atom_term_keys() -> tuple[str, ...]:
    global _ATOM_TERM_KEYS
    if _ATOM_TERM_KEYS is None:
        keys: list[str] = []
        for key in _depth2_keys():
            keys.append(key)
        for key in _depth3_keys():
            keys.append(key)
        for key in _depth4_keys():
            keys.append(key)
        _ATOM_TERM_KEYS = tuple(keys)
    return _ATOM_TERM_KEYS


_ATOM_DESC_SCALE: dict[str, float] | None = None


def _atom_desc_scale() -> dict[str, float]:
    """Per-key corpus-max normalization for the production linear model.

    The atom descriptors are STATIC champion knowledge (not outcome data), so
    normalizing by the corpus maximum is strictly-prior safe.  Scaling keeps
    liblinear convergence fast and makes every descriptor comparable; the
    CatBoost frontier rows keep the raw values (trees handle scale natively).
    """
    global _ATOM_DESC_SCALE
    if _ATOM_DESC_SCALE is None:
        corpora = (
            (_atom_depth2_index(), "d2_"),
            (_atom_depth3_index(), "d3_"),
            (_atom_depth4_index(), "d4_"),
        )
        scale: dict[str, float] = {}
        for index, prefix in corpora:
            for key in _atom_term_keys():
                if not key.startswith(prefix):
                    continue
                maximum = max((abs(entry.get(key, 0.0)) for entry in index.values()), default=0.0)
                scale[key] = maximum if maximum > 0.0 else 1.0
        _ATOM_DESC_SCALE = scale
    return _ATOM_DESC_SCALE


def _atom_desc_value(champion: str, key: str) -> float:
    """Alias-aware, scale-normalized per-champion descriptor across d2/d3/d4."""
    slug = _atom_slug(str(champion or ""))
    if key.startswith("d4_"):
        raw = float(_atom_depth4_index().get(slug, {}).get(key, 0.0))
    elif key.startswith("d3_"):
        raw = float(_atom_depth3_index().get(slug, {}).get(key, 0.0))
    else:
        raw = float(_atom_depth2_index().get(slug, {}).get(key, 0.0))
    return raw / _atom_desc_scale().get(key, 1.0)

def _build_frontier(ordered_games: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """One date-ordered strictly-prior pass; every feature uses prior games only."""
    global _FRONTIER_NAMES, _FRONTIER
    bans = _ban_table(ordered_games)
    ss_rows = _build_ss_rows(ordered_games)

    patch_first: dict[str, pd.Timestamp] = {}
    for game in ordered_games:
        patch = str(game.get("patch") or "UNKNOWN")
        date = _timestamp(game["date"])
        if patch not in patch_first or date < patch_first[patch]:
            patch_first[patch] = date
    pick_counts: dict[str, int] = {}
    pick_total = 0
    form_alpha = 0.1
    player_form: dict[str, np.ndarray] = {}
    h2h: dict[str, dict[str, float]] = {}
    blue_wr: dict[str, float] = {}
    red_wr: dict[str, float] = {}
    pools: dict[str, Any] = {}
    lineups: dict[str, Any] = {}
    matchup: dict[tuple, list[int]] = {}
    # L7: strictly-prior per-player atom-family proficiency (read before update)
    l7_profile: dict[str, np.ndarray] = {}
    l7_alpha = 0.05
    # L7 per-role: strictly-prior per-(player, role) atom-family proficiency
    l7r_profile: dict[tuple[str, str], np.ndarray] = {}

    names = [
        "f1_h2h", "f2_kda", "f2_ds", "f2_cspm", "f2_vis",
        "f3_roster_last", "f3_roster_last3",
        "f4_pool_overlap", "f5_ban1_meta", "f6_patch_recency", "f7_side_pref",
        "f_l5_elo_p", "f_l5_elo_sigma",
        *[f"f8_matchup_{role}" for role in ROLES],
        *[f"corp_fam{index}" for index in range(len(ATOM_FAMILIES))],
        *[f"corp_mech{index}" for index in range(len(ATOM_MECHANIC_KEYS))],
        *[f"corp_role{role}_{index}" for role in ROLES for index in range(len(ATOM_FAMILIES))],
        "corp_counters",
        *[f"d2_{key}" for key in _depth2_keys()],
        *[f"d3_{key}" for key in _depth3_keys()],
        *[f"d4_{key}" for key in _depth4_keys()],
        *[f"ss_{name}" for name in _SS_KEYS],
        *[f"l7_{name}" for name in ("ccm", "dmg", "heal", "int", "stack", "vision")],
        *[f"l7r_{role}_{name}" for role in ROLES for name in ("ccm", "dmg", "heal", "int", "stack", "vision")],
    ]
    _FRONTIER_NAMES = names
    _FRONTIER = {}

    def ewma(state: float | None, value: float, alpha: float) -> float:
        return value if state is None else alpha * value + (1 - alpha) * state

    stats_by_game: list[tuple[pd.Timestamp, str, float, float, float, float, float]] = []
    for game in ordered_games:
        date = _timestamp(game["date"])
        for player_key, stats in (game.get("player_stats") or {}).items():
            stats_by_game.append((
                date,
                str(player_key),
                float(stats.get("damageshare", 0.0)),
                float(stats.get("cspm", 0.0)),
                float(stats.get("visionscore", 0.0)),
                float(stats.get("kills", 0.0)),
                float(stats.get("deaths", 0.0)),
            ))
    stats_by_game.sort(key=lambda item: item[0])
    stats_ptr = 0
    rows: dict[str, np.ndarray] = {}
    for game in ordered_games:
        game_date = _timestamp(game["date"])
        game_uid = str(game["game_uid"])
        while stats_ptr < len(stats_by_game) and stats_by_game[stats_ptr][0] < game_date:
            _, player, dmg, cspm, vis, kills, deaths = stats_by_game[stats_ptr]
            kda = (kills + deaths) / 2.0 if (kills + deaths) > 0 else 0.0
            prior = player_form.get(player)
            if prior is None:
                player_form[player] = np.asarray([kda, dmg, cspm, vis], dtype=float)
            else:
                player_form[player] = (1 - form_alpha) * prior + form_alpha * np.asarray(
                    [kda, dmg, cspm, vis], dtype=float
                )
            stats_ptr += 1

        blue_team = str(game["blue_team"]).strip()
        red_team = str(game["red_team"]).strip()
        blue_players = [str(game["blue"][role]["player"]).strip().casefold() for role in ROLES]
        red_players = [str(game["red"][role]["player"]).strip().casefold() for role in ROLES]
        blue_champs = [str(game["blue"][role]["champion"]) for role in ROLES]
        red_champs = [str(game["red"][role]["champion"]) for role in ROLES]

        row: list[float] = []
        blue_prior = (h2h.get(blue_team) or {}).get(red_team)
        row.append(0.0 if blue_prior is None else blue_prior - 0.5)

        blue_forms = [player_form.get(p) for p in blue_players]
        red_forms = [player_form.get(p) for p in red_players]

        def avg(forms: Sequence[np.ndarray | None], index: int) -> float:
            values = [f[index] for f in forms if f is not None]
            return float(np.mean(values)) if values else 0.0

        for index in range(4):
            row.append(avg(blue_forms, index) - avg(red_forms, index))

        def roster_overlap(players: Sequence[str], team: str) -> float:
            last = lineups.get(team)
            if not last:
                return 0.0
            any3: set[str] = set()
            for lineup in list(last)[-3:]:
                any3 |= lineup
            last_set = last[-1]
            return float(len(set(players) & last_set)) / 5.0

        row.append(roster_overlap(blue_players, blue_team))
        row.append(roster_overlap(red_players, red_team))

        def pool(team: str) -> set[str]:
            return set(pools.get(team, []))

        bp, rp = pool(blue_team), pool(red_team)
        row.append(float(len(bp & rp) / max(len(bp | rp), 1)) if bp or rp else 0.0)

        game_bans = bans.get(game_uid)
        ban1_meta = [0.0, 0.0]
        if game_bans:
            for side_index, side in enumerate(("blue", "red")):
                slot = game_bans.get(side) or []
                if slot:
                    ban1_meta[side_index] = pick_counts.get(_atom_slug(slot[0]), 0) / max(pick_total, 1)
        row.append(ban1_meta[0] - ban1_meta[1])

        patch = str(game.get("patch") or "UNKNOWN")
        first = patch_first.get(patch)
        row.append(float((game_date - first).total_seconds() / 86400.0) if first is not None else 0.0)

        row.append((blue_wr.get(blue_team) or 0.5) - (red_wr.get(red_team) or 0.5))

        elo_state = game.get("player_elo")
        row.append((float(elo_state.get("p", 0.5)) - 0.5) if elo_state else 0.0)
        row.append(float(elo_state.get("sigma", 0.0)) if elo_state else 0.0)

        for lane_index, role in enumerate(ROLES):
            key = (role, blue_champs[lane_index], red_champs[lane_index])
            wins, meets = matchup.get(key, [0, 0])
            rate = (wins + 2.5 * 0.5) / (meets + 2.5) if meets >= 0 else 0.5
            row.append(rate - 0.5)

        row.extend(_corpus_game_features(game))
        row.extend(_depth2_game_row(game))
        row.extend(_depth3_game_row(game))
        row.extend(_depth4_game_row(game))
        row.extend(_ss_game_row(game, ss_rows))

        # L7 row from PRIOR player profiles (strictly prior: read before update)
        def l7_team(players: Sequence[str], champs: Sequence[str], team_outcome: float) -> list[np.ndarray]:
            vals: list[np.ndarray] = []
            for player, champ in zip(players, champs):
                vector = _cached_atom_vector(champ)
                fam = vector[:6] if vector is not None else np.zeros(6)
                prior = l7_profile.get(player)
                if prior is None:
                    prior = np.zeros(6)
                vals.append(prior.copy())
                l7_profile[player] = (1 - l7_alpha) * prior + l7_alpha * (team_outcome * fam)
            return vals

        # L7 per-role row from PRIOR per-(player, role) profiles (strictly prior)
        def l7r_team(players: Sequence[str], champs: Sequence[str], team_outcome: float) -> list[np.ndarray]:
            vals: list[np.ndarray] = []
            for role, player, champ in zip(ROLES, players, champs):
                vector = _cached_atom_vector(champ)
                fam = vector[:6] if vector is not None else np.zeros(6)
                key = (player, role)
                prior = l7r_profile.get(key)
                if prior is None:
                    prior = np.zeros(6)
                vals.append(prior.copy())
                l7r_profile[key] = (1 - l7_alpha) * prior + l7_alpha * (team_outcome * fam)
            return vals

        outcome = int(game["y"])
        blue_prof = l7_team(blue_players, blue_champs, float(outcome))
        red_prof = l7_team(red_players, red_champs, 1.0 - float(outcome))
        row.extend(np.mean(blue_prof, axis=0) - np.mean(red_prof, axis=0))
        blue_rprof = l7r_team(blue_players, blue_champs, float(outcome))
        red_rprof = l7r_team(red_players, red_champs, 1.0 - float(outcome))
        row.extend(np.concatenate(blue_rprof) - np.concatenate(red_rprof))
        rows[game_uid] = np.asarray(row, dtype=float)
        h2h.setdefault(blue_team, {})[red_team] = ewma(
            h2h.get(blue_team, {}).get(red_team), float(outcome), alpha=0.2)
        h2h.setdefault(red_team, {})[blue_team] = ewma(
            h2h.get(red_team, {}).get(blue_team), 1.0 - float(outcome), alpha=0.2)
        blue_wr[blue_team] = ewma(blue_wr.get(blue_team), float(outcome), alpha=0.1)
        red_wr[red_team] = ewma(red_wr.get(red_team), 1.0 - float(outcome), alpha=0.1)
        pools.setdefault(blue_team, []).extend(blue_champs)
        if len(pools[blue_team]) > 10:
            pools[blue_team] = pools[blue_team][-10:]
        pools.setdefault(red_team, []).extend(red_champs)
        if len(pools[red_team]) > 10:
            pools[red_team] = pools[red_team][-10:]
        lineups.setdefault(blue_team, []).append(frozenset(blue_players))
        if len(lineups[blue_team]) > 5:
            lineups[blue_team] = lineups[blue_team][-5:]
        lineups.setdefault(red_team, []).append(frozenset(red_players))
        if len(lineups[red_team]) > 5:
            lineups[red_team] = lineups[red_team][-5:]
        for lane_index, role in enumerate(ROLES):
            key = (role, blue_champs[lane_index], red_champs[lane_index])
            state = matchup.setdefault(key, [0, 0])
            state[1] += 1
            if outcome == 1:
                state[0] += 1
        for champion in [*blue_champs, *red_champs]:
            pick_counts[_atom_slug(champion)] = pick_counts.get(_atom_slug(champion), 0) + 1
            pick_total += 1
    _FRONTIER = rows
    return rows


def _full_feature_columns(
    train: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    draft: FittedCompositionModel,
    tl: np.ndarray,
    vl: np.ndarray,
    th: np.ndarray,
    vh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Winner feature matrix: [linpred, history, log1p mastery, bans_v2, frontier]."""
    ban_rows = _bans_v2_feature_builder(train, validation, draft)
    train_x = np.column_stack([
        tl, th, _exp_rows_log1p(train), ban_rows(train), _frontier_rows(train),
    ])
    validation_x = np.column_stack([
        vl, vh, _exp_rows_log1p(validation), ban_rows(validation), _frontier_rows(validation),
    ])
    return train_x, validation_x


def _catboost_factory(params: Mapping[str, Any]) -> Any:
    def factory():
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=int(params.get("n_estimators", 600)),
            learning_rate=float(params.get("lr", 0.03)),
            depth=int(params.get("depth", 4)),
            l2_leaf_reg=float(params.get("reg", 8.0)),
            random_seed=461,
            verbose=0,
            allow_writing_files=False,
        )
    return factory


def _production_style_recalibrate(
    train: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    draft: FittedCompositionModel,
    model_factory: Any,
    raw_val_probs: np.ndarray,
    *,
    shrink: float = 0.5,
) -> np.ndarray:
    """Identity-shrunk affine recalibration fit strictly inside the training
    fold for any model class (mirrors the round-5 winner)."""
    ordered = sorted(
        (dict(game) for game in train if game.get("controls_available", False)),
        key=lambda item: (_timestamp(item["date"]), str(item["game_uid"])),
    )
    cutoff = max(int(len(ordered) * 0.8), 8)
    fit_games = ordered[:cutoff]
    cal_games = ordered[cutoff:]
    clipped_raw = np.clip(np.asarray(raw_val_probs, dtype=float), 1e-5, 1 - 1e-5)
    if len(cal_games) < 8 or len({int(game["y"]) for game in cal_games}) < 2:
        return clipped_raw
    names = _feature_names(fit_games)
    fold_draft = _fit_model(
        fit_games,
        names=names,
        include_draft=True,
        regularization_c=draft.regularization_c,
    )
    if fold_draft is None:
        return clipped_raw
    sequence = fit_games + cal_games
    history = _history_features(sequence, fold_draft)
    fit_history = history[: len(fit_games)]
    cal_history = history[len(fit_games):]
    fit_linpred = np.asarray(
        _matrix(fit_games, fold_draft.feature_names, include_draft=True)
        @ np.asarray(fold_draft.coefficients),
        dtype=float,
    )
    cal_linpred = np.asarray(
        _matrix(cal_games, fold_draft.feature_names, include_draft=True)
        @ np.asarray(fold_draft.coefficients),
        dtype=float,
    )
    fold_model = model_factory()
    fold_model.fit(
        np.column_stack([fit_linpred, fit_history]),
        [int(game["y"]) for game in fit_games],
    )
    cal_probs = fold_model.predict_proba(np.column_stack([cal_linpred, cal_history]))[:, 1]
    calibrator = LogisticRegression(C=1e6, solver="liblinear", max_iter=1000)
    calibrator.fit(
        np.log(np.clip(cal_probs, 1e-5, 1 - 1e-5) / (1 - np.clip(cal_probs, 1e-5, 1 - 1e-5))).reshape(-1, 1),
        [int(game["y"]) for game in cal_games],
    )
    a = float(calibrator.intercept_[0])
    b = float(calibrator.coef_[0][0])
    a_eff = shrink * a
    b_eff = 1.0 + shrink * (b - 1.0)
    raw_logits = np.log(clipped_raw / (1 - clipped_raw))
    return 1.0 / (1.0 + np.exp(-np.clip(a_eff + b_eff * raw_logits, -30.0, 30.0)))


def evaluate_composition_signal(
    games: Sequence[Mapping[str, Any]],
    *,
    source_hash: str | None = None,
    canonical_game_identity_sha256: str | None = None,
    worker_commit: str | None = None,
    bootstrap_reps: int = 200,
    seed: int = 461,
    min_training_games: int = MIN_TRAINING_GAMES,
    history_calibrate_shrink: float = 0.25,
) -> dict[str, Any]:
    """Run four chronological holdouts for the composition candidate.

    The candidate uses per-window regularization selected on an internal
    date split, out-of-fold affine recalibration fit strictly inside each
    training fold, per-match window-stratified bootstrap intervals, and
    strictly-lagged team-history features with an identity-shrunk
    recalibrator. No validation-window game influences a fit or transform.
    """

    ordered = [dict(game) for game in sorted(games, key=lambda item: (_timestamp(item["date"]), str(item["game_uid"]))) if game.get("controls_available", False)]
    if len(ordered) < max(20, min_training_games + 4):
        raise CompositionSignalError("not enough complete games for the four-window evaluation")
    windows: list[dict[str, Any]] = []
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
    # Round-5 winner: one global date-ordered strictly-prior frontier pass
    # (F1-F8 + champion-mechanic corpus features) shared by all windows.
    _build_frontier(ordered)
    for window_index in range(4):
        train = [game for cluster in date_clusters[: boundaries[window_index + 1]] for game in cluster]
        validation = [game for cluster in date_clusters[boundaries[window_index + 1] : boundaries[window_index + 2]] for game in cluster]
        training_names = _feature_names(train)
        fit_c = _select_regularization(
            train,
            names=training_names,
            candidates=(0.003, 0.01, 0.03, 0.1, 0.3, 1.0),
            internal_fraction=0.15,
            min_training_games=min_training_games,
            worker_commit=worker_commit,
        )
        baseline = _fit_model(train, names=training_names, include_draft=False, min_training_games=min_training_games, regularization_c=fit_c, worker_commit=worker_commit)
        draft = _fit_model(train, names=training_names, include_draft=True, min_training_games=min_training_games, regularization_c=fit_c, worker_commit=worker_commit)
        if baseline is None or draft is None or not validation:
            continue
        baseline_probabilities, draft_probabilities = _apply_oof_recalibration(
            train,
            validation,
            baseline,
            draft,
        )
        y = [int(game["y"]) for game in validation]
        window_payload = {
            "window": window_index + 1,
            "fit_through": draft.fit_through,
            "holdout_from": _rfc(validation[0]["date"]),
            "holdout_through": _rfc(validation[-1]["date"]),
            "baseline": _metrics(y, baseline_probabilities),
            "draft_augmented": _metrics(y, draft_probabilities),
        }
        window_payload["baseline"]["probabilities"] = [
            float(value) for value in baseline_probabilities
        ]
        window_payload["baseline"]["outcomes"] = list(y)
        window_payload["draft_augmented"]["probabilities"] = [
            float(value) for value in draft_probabilities
        ]
        window_payload["draft_augmented"]["outcomes"] = list(y)
        windows.append(window_payload)
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
        train_linpred = np.asarray(
            _matrix(train, draft.feature_names, include_draft=True) @ np.asarray(draft.coefficients),
            dtype=float,
        )
        validation_linpred = np.asarray(
            _matrix(validation, draft.feature_names, include_draft=True) @ np.asarray(draft.coefficients),
            dtype=float,
        )
        try:
            # Round-5 winner: CatBoost over the full frontier feature set with
            # identity-shrunk in-fold recalibration (no validation contact).
            train_x, validation_x = _full_feature_columns(
                train,
                validation,
                draft,
                train_linpred,
                validation_linpred,
                train_history,
                validation_history,
            )
            catboost_model = _catboost_factory({})().fit(
                train_x,
                [int(game["y"]) for game in train],
            )
            raw_probabilities = catboost_model.predict_proba(validation_x)[:, 1]
            history_probabilities = _production_style_recalibrate(
                train,
                validation,
                draft,
                lambda: LogisticRegression(C=1.0, solver="liblinear", max_iter=2000, random_state=461),
                raw_probabilities,
                shrink=0.5,
            )
        except ImportError:
            # Fallback without catboost: round-2 history logistic.
            history_train_x = np.column_stack([train_linpred, train_history])
            history_c = _select_history_regularization(
                train,
                history_train_x,
                candidates=(0.03, 0.1, 0.3, 1.0, 3.0),
                internal_fraction=0.15,
                min_training_games=min_training_games,
            )
            history_model = LogisticRegression(C=history_c, solver="liblinear", max_iter=2000, random_state=461)
            history_model.fit(history_train_x, [int(game["y"]) for game in train])
            history_validation_x = np.column_stack([validation_linpred, validation_history])
            history_probabilities = _recalibrate_history_probabilities(
                train,
                validation,
                draft,
                history_train_x,
                history_validation_x,
                history_model,
                folds=4,
                shrink=history_calibrate_shrink,
                fold_c=history_c,
            )
        windows[-1]["draft_plus_team_history"] = _metrics(y, history_probabilities)
        windows[-1]["draft_plus_team_history"]["probabilities"] = [
            float(value) for value in history_probabilities
        ]
        windows[-1]["draft_plus_team_history"]["outcomes"] = list(y)
    if not windows:
        raise CompositionSignalError("chronological evaluation did not produce a valid holdout")
    bootstrap = _match_delta_intervals(
        windows,
        reps=bootstrap_reps,
        seed=seed,
        label="draft_vs_baseline",
    )
    history_bootstrap = _match_delta_intervals(
        windows,
        reps=bootstrap_reps,
        seed=seed + 1,
        label="history_vs_draft",
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
