"""Research-only atomized Random Forest composite.

The module keeps three products separate:

* the public composition-only descriptive Draft Score, which this module never edits;
* a pre-match outcome composite built from strictly prior information;
* a live-state composite contract, where observed state can be used after map start.

Layer A uses exact historical statistical fields.  Layer B uses patch-bound
League Combat Calculator mechanic atoms.  Layer B fails closed when a map has
no exact patch-time atom snapshot.  No current mechanics snapshot is applied
to an older patch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_squared_error, roc_auc_score

from lol_kills.etl.aliases import normalize_team


SCHEMA_VERSION = "scryglass:atomized-rf-composite-research:v3"
FEATURE_SCHEMA_VERSION = "scryglass:atomized-rf-layer-a:v3"
MECHANICS_SCHEMA_VERSION = "scryglass:exact-lcc-mechanics-atoms:v1"
TRAIN_END = pd.Timestamp("2026-05-01", tz="UTC")
VALIDATION_END = pd.Timestamp("2026-07-01", tz="UTC")
CONSUMED_TEST_END = pd.Timestamp("2026-08-09", tz="UTC")
MOMENTUM_WINDOW_GAMES = 7
MOMENTUM_SCALE = 80.0
CHECKPOINTS = (10, 15, 20, 25)
TARGET_LEAGUES = ("LCS", "LEC", "LCK", "LPL", "MSI", "EWC", "Worlds")
RANDOM_SEED = 461
LOCKED_DATASET_SHA256 = "e2561f4b15942d4b72cdcbdaa14dd2dfeba39a6edf2b1fd2d01c3ce3bff531fe"
LOCKED_REPORT_SHA256 = "598540dd2128cbb1a15ee62dbe4cb28d690dd792c826dee4c3f16908c8d655a7"
LCC_26_16_BRIDGE_SHA256 = "bff75179d93a6c5d9bf4bc3927e1ca7b12f31335cef9249449f2595093ed1942"
LCC_26_16_COMMIT = "5b0ad6eabc3396ea714a270e3d87ad1dc72c6fac"
LCC_26_15_SEED_COMMIT = "f0718a98c29dcf5559ffa98c46a487cd52d9c9e3"
RAW_2026_IDENTITY_SHA256 = "9467bb8d3f15571a2445a4728d2d38290fa78079c4fe6b2d1037b54d0b988a65"
RAW_2026_DRIVE_REVISION = "0B_SP330uQdQ_cHhDN1h6RVI3NUkraUNyZWprQnVDSThEeE5jPQ"
SMOOTHING_PRIOR_GAMES = 5.0

FEATURE_COVERAGE_THRESHOLDS: dict[str, float] = {
    "team_rating": 0.95,
    "player_rating": 0.95,
    "player_exact_performance": 0.50,
    "exact_ally_enemy_pairs": 0.35,
    "checkpoint_forecasts": 0.50,
    "parity_conditioned_performance": 0.20,
    "team_momentum": 0.50,
    "patch_exact_performance": 0.35,
}

HISTORICAL_METRICS: dict[str, str] = {
    **{
        f"{name}_{checkpoint}": f"{source}at{checkpoint}"
        for checkpoint in CHECKPOINTS
        for name, source in (
            ("gold_diff", "golddiff"),
            ("xp_diff", "xpdiff"),
            ("cs_diff", "csdiff"),
            ("kills", "kills"),
            ("assists", "assists"),
            ("deaths", "deaths"),
        )
    },
    "damage_to_champions": "damagetochampions",
    "damage_taken_per_minute": "damagetakenperminute",
    "damage_mitigated_per_minute": "damagemitigatedperminute",
    "damage_to_towers": "damagetotowers",
    "vision_score": "visionscore",
    "wards_placed": "wardsplaced",
    "wards_killed": "wardskilled",
    "own_jungle_monsters": "monsterkillsownjungle",
    "enemy_jungle_monsters": "monsterkillsenemyjungle",
    "earned_gold": "earnedgold",
    "earned_gold_share": "earnedgoldshare",
    "damage_share": "damageshare",
    "result_residual": "__result_residual__",
}

# These are broad labels.  They are never emitted as Layer B model inputs.
FORBIDDEN_GENERAL_LABELS = frozenset(
    {"engage", "poke", "scaling", "snowball", "frontline", "teamfight", "archetype"}
)

def _metric_columns(prefixes: Sequence[str], *, checkpoints: Sequence[int] | None = None) -> tuple[str, ...]:
    output: list[str] = []
    for prefix in prefixes:
        for checkpoint in checkpoints or (0,):
            stem = f"{prefix}_{checkpoint}" if checkpoint else prefix
            for metric in HISTORICAL_METRICS:
                output.extend((f"{stem}_{metric}", f"{stem}_{metric}_missing"))
    return tuple(output)


GROUP_COLUMNS: dict[str, tuple[str, ...]] = {
    "team_rating": (
        "base_team_logit",
        "team_rating_diff_scaled",
        "team_rating_available",
        "team_rating_missing",
    ),
    "player_rating": (
        "base_player_logit",
        "player_rating_diff_scaled",
        "player_lineup_complete",
        "player_rating_available",
        "player_rating_missing",
    ),
    "player_exact_performance": (
        "history_unique_player_maps_min",
    ) + _metric_columns(("history_player_champion",)),
    "exact_ally_enemy_pairs": _metric_columns(
        ("history_ally_champion_pair", "history_enemy_champion_pair")
    ),
    "checkpoint_forecasts": tuple(
        value
        for checkpoint in CHECKPOINTS
        for value in (
            f"forecast_gold_diff_{checkpoint}",
            f"forecast_gold_support_{checkpoint}",
            f"forecast_gold_player_coverage_{checkpoint}",
            f"forecast_gold_available_{checkpoint}",
            f"forecast_gold_missing_{checkpoint}",
            f"forecast_xp_diff_{checkpoint}",
            f"forecast_xp_support_{checkpoint}",
            f"forecast_xp_player_coverage_{checkpoint}",
            f"forecast_xp_available_{checkpoint}",
            f"forecast_xp_missing_{checkpoint}",
        )
    )
    + tuple(
        value
        for first, second in zip(CHECKPOINTS, CHECKPOINTS[1:])
        for value in (f"forecast_gold_slope_{first}_{second}", f"forecast_xp_slope_{first}_{second}")
    )
    + (
        "forecast_peak_checkpoint_signed",
        "forecast_peak_magnitude",
        "forecast_curve_available",
        "forecast_curve_missing",
    ),
    "parity_conditioned_performance": _metric_columns(
        ("parity_player_champion",), checkpoints=CHECKPOINTS
    ),
    "team_momentum": (
        "team_momentum_points_diff",
        "player_momentum_points_diff",
        "team_momentum_count_difference",
        "team_momentum_coverage",
        "team_momentum_available",
        "team_momentum_missing",
        "player_momentum_count_difference",
        "player_momentum_coverage",
        "player_momentum_available",
        "player_momentum_missing",
    ),
    "patch_exact_performance": _metric_columns(
        ("patch_player_champion", "patch_champion")
    ),
}
MODEL_COLUMNS = ("blue_side",) + tuple(
    column for columns in GROUP_COLUMNS.values() for column in columns
)
FEATURE_AVAILABILITY_COLUMNS = {
    group: f"availability_{group}" for group in GROUP_COLUMNS
}
TARGET_PREFIX = "target_"
LOCKED_BASELINE_COLUMNS = (
    "base_team_logit",
    "base_player_logit",
    "team_rating_diff_scaled",
    "player_rating_diff_scaled",
    "blue_side",
    "player_lineup_complete",
    "team_wr_diff_g40",
    "team_residual_diff_g40",
    "team_last_diff_g40",
    "team_streak_diff_g40",
    "team_count_diff_g40",
    "player_wr_diff_g40",
    "player_residual_diff_g40",
    "player_last_diff_g40",
    "player_streak_diff_g40",
    "player_count_diff_g40",
    "player_coverage_g40",
)


class AtomizedResearchError(ValueError):
    """Raised when a research authority or leakage contract fails."""


@dataclass
class RunningStat:
    total: float = 0.0
    count: int = 0

    def add(self, value: float) -> None:
        if math.isfinite(value):
            self.total += float(value)
            self.count += 1

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0


@dataclass(frozen=True)
class RFConfig:
    n_estimators: int
    max_depth: int | None
    min_samples_leaf: int
    max_features: str | float
    class_weight: str | None
    bootstrap: bool
    max_samples: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "class_weight": self.class_weight,
            "bootstrap": self.bootstrap,
            "max_samples": self.max_samples,
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
        }


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _side(value: Any) -> str:
    normalized = str(value or "").strip().casefold()
    if normalized == "blue":
        return "Blue"
    if normalized == "red":
        return "Red"
    raise AtomizedResearchError(f"invalid side {value!r}")


def _player_id(row: Mapping[str, Any]) -> str:
    value = str(row.get("playerid") or row.get("player_id") or "").strip()
    if not value.startswith("oe:player:"):
        raise AtomizedResearchError("stable OE player ID is required")
    return value


def _team_id(row: Mapping[str, Any]) -> str:
    value = str(row.get("teamid") or "").strip()
    if not value.startswith("oe:team:"):
        raise AtomizedResearchError("stable OE team ID is required")
    return value


def _game_id(row: Mapping[str, Any]) -> str:
    return str(row.get("game_uid") or row.get("gameid") or "").strip()


def normalize_source_patch(value: Any, date: Any) -> str:
    """Preserve OE patch identity when CSV float parsing drops a zero."""

    token = str(value or "").strip()
    timestamp = pd.Timestamp(date)
    if token == "16.1" and timestamp >= pd.Timestamp("2026-04-01", tz="UTC"):
        return "16.10"
    return token


def _stat_mean(state: Mapping[Any, RunningStat], keys: Iterable[Any]) -> tuple[float, int]:
    values = [state[key] for key in keys if key in state and state[key].count]
    if not values:
        return 0.0, 0
    support = sum(value.count for value in values)
    return float(np.average([value.mean for value in values], weights=[value.count for value in values])), support


def _side_difference(
    state: Mapping[Any, RunningStat],
    blue_keys: Iterable[Any],
    red_keys: Iterable[Any],
) -> tuple[float, int]:
    blue, blue_n = _stat_mean(state, blue_keys)
    red, red_n = _stat_mean(state, red_keys)
    return blue - red, min(blue_n, red_n)


def _equal_weight_team_forecast(
    state: Mapping[Any, RunningStat], keys: Iterable[Any]
) -> tuple[float, int, float, int]:
    """Return a five-player team total with one equal term per current player."""

    keys = list(keys)
    values = [state.get(key) for key in keys]
    observed = [value for value in values if value is not None and value.count]
    coverage = len(observed) / len(keys) if keys else 0.0
    support = min((value.count for value in observed), default=0)
    if not keys or len(observed) != len(keys):
        return 0.0, support, coverage, 1
    return float(sum(value.mean for value in observed)), support, coverage, 0


def _phase_curve_features(
    gold_differences: Sequence[float],
    xp_differences: Sequence[float],
    *,
    available: bool,
) -> dict[str, float]:
    """Derive side-antisymmetric curve fields from team-total forecasts."""

    if len(gold_differences) != len(CHECKPOINTS) or len(xp_differences) != len(
        CHECKPOINTS
    ):
        raise AtomizedResearchError("phase curve needs every checkpoint")
    output: dict[str, float] = {}
    if not available:
        for first, second in zip(CHECKPOINTS, CHECKPOINTS[1:]):
            output[f"forecast_gold_slope_{first}_{second}"] = 0.0
            output[f"forecast_xp_slope_{first}_{second}"] = 0.0
        output.update(
            {
                "forecast_peak_checkpoint_signed": 0.0,
                "forecast_peak_magnitude": 0.0,
                "forecast_curve_available": 0.0,
                "forecast_curve_missing": 1.0,
            }
        )
        return output
    for index, (first, second) in enumerate(zip(CHECKPOINTS, CHECKPOINTS[1:])):
        span = float(second - first)
        output[f"forecast_gold_slope_{first}_{second}"] = (
            float(gold_differences[index + 1]) - float(gold_differences[index])
        ) / span
        output[f"forecast_xp_slope_{first}_{second}"] = (
            float(xp_differences[index + 1]) - float(xp_differences[index])
        ) / span
    curve = np.asarray(gold_differences, dtype=float) / 1000.0 + np.asarray(
        xp_differences, dtype=float
    ) / 1000.0
    peak_index = int(np.argmax(np.abs(curve)))
    peak = float(curve[peak_index])
    output.update(
        {
            "forecast_peak_checkpoint_signed": float(
                math.copysign(CHECKPOINTS[peak_index], peak)
            )
            if peak
            else 0.0,
            "forecast_peak_magnitude": peak,
            "forecast_curve_available": 1.0,
            "forecast_curve_missing": 0.0,
        }
    )
    return output


def _unique_player_map_support(
    state: Mapping[tuple[str, str], set[str]], keys: Iterable[tuple[str, str]]
) -> int:
    """Count distinct prior player-map observations once across metric families."""

    observations = {
        (player, game_uid)
        for player, champion in keys
        for game_uid in state.get((player, champion), set())
    }
    return len(observations)


def _resolved_roster_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Bind the exact ten-player, team, side, role, and champion assignment."""

    roster = sorted(
        (
            _side(row.get("side")),
            str(row.get("position") or "").strip().casefold(),
            _team_id(row),
            _player_id(row),
            str(row.get("champion") or "").strip().casefold(),
        )
        for row in rows
    )
    if len(roster) != 10 or len({(side, role) for side, role, *_ in roster}) != 10:
        raise AtomizedResearchError("rating roster binding requires ten unique side-role rows")
    return canonical_sha256(roster)


def _locked_rating_authority(
    base_row: Mapping[str, Any], *, resolved_roster_sha256: str
) -> dict[str, float]:
    """Require a hash-bound source receipt for every locked rating value."""

    team_values = [
        _finite(base_row.get("base_team_logit")),
        _finite(base_row.get("team_rating_diff_scaled")),
    ]
    player_values = [
        _finite(base_row.get("base_player_logit")),
        _finite(base_row.get("player_rating_diff_scaled")),
    ]
    source_sha256 = str(base_row.get("rating_source_sha256") or "")
    bound_roster_sha256 = str(base_row.get("rating_roster_sha256") or "")
    receipt_payload = {
        "schema_version": base_row.get("rating_receipt_schema"),
        "source_available": _finite(base_row.get("rating_source_available")),
        "source_sha256": source_sha256,
        "roster_sha256": bound_roster_sha256,
    }
    receipt_matches = bool(
        str(base_row.get("rating_receipt_sha256") or "")
        == canonical_sha256(receipt_payload)
    )
    source_receipt_available = bool(
        base_row.get("rating_receipt_schema")
        == "scryglass:resolved-rating-source:v1"
        and _finite(base_row.get("rating_source_available")) == 1.0
        and re.fullmatch(r"[0-9a-f]{64}", source_sha256)
        and receipt_matches
    )
    roster_matches = bool(
        source_receipt_available
        and bound_roster_sha256 == resolved_roster_sha256
    )
    lineup_complete = _finite(base_row.get("player_lineup_complete")) == 1.0
    team_available = roster_matches and all(value is not None for value in team_values)
    player_available = (
        roster_matches
        and lineup_complete
        and all(value is not None for value in player_values)
    )
    return {
        "base_team_logit": float(team_values[0]) if team_available else 0.0,
        "team_rating_diff_scaled": float(team_values[1]) if team_available else 0.0,
        "team_rating_available": float(team_available),
        "team_rating_missing": float(not team_available),
        "base_player_logit": float(player_values[0]) if player_available else 0.0,
        "player_rating_diff_scaled": float(player_values[1]) if player_available else 0.0,
        "player_lineup_complete": float(player_available),
        "player_rating_available": float(player_available),
        "player_rating_missing": float(not player_available),
        "rating_source_receipt_available": float(source_receipt_available),
        "rating_source_receipt_hash_match": float(receipt_matches),
        "rating_roster_receipt_match": float(roster_matches),
    }


def _momentum_features(
    team_history: Mapping[str, deque[float]],
    player_history: Mapping[str, deque[float]],
    blue_team: str,
    red_team: str,
    blue_players: Sequence[str],
    red_players: Sequence[str],
) -> dict[str, float]:
    """Build the seven-map scale-80 candidate from resolved Layer A IDs."""

    blue_team_mean, blue_team_n = _recent_mean(team_history, blue_team)
    red_team_mean, red_team_n = _recent_mean(team_history, red_team)
    team_available = blue_team_n > 0 and red_team_n > 0
    player_rows = [
        _recent_mean(player_history, player)
        for player in [*blue_players, *red_players]
    ]
    player_coverage = sum(count > 0 for _, count in player_rows) / 10.0
    player_available = len(player_rows) == 10 and player_coverage == 1.0
    blue_player_rows = player_rows[:5]
    red_player_rows = player_rows[5:]
    player_difference = (
        float(np.mean([value for value, _ in blue_player_rows]))
        - float(np.mean([value for value, _ in red_player_rows]))
        if player_available
        else 0.0
    )
    return {
        "team_momentum_points_diff": MOMENTUM_SCALE
        * (blue_team_mean - red_team_mean if team_available else 0.0),
        "team_momentum_count_difference": float(blue_team_n - red_team_n),
        "team_momentum_coverage": float(min(blue_team_n, red_team_n) / MOMENTUM_WINDOW_GAMES),
        "team_momentum_available": float(team_available),
        "team_momentum_missing": float(not team_available),
        "player_momentum_points_diff": MOMENTUM_SCALE * player_difference,
        "player_momentum_count_difference": float(
            sum(count for _, count in blue_player_rows)
            - sum(count for _, count in red_player_rows)
        ),
        "player_momentum_coverage": float(player_coverage),
        "player_momentum_available": float(player_available),
        "player_momentum_missing": float(not player_available),
    }


def _shrunk_metric_mean(
    state: Mapping[Any, RunningStat],
    global_state: Mapping[str, RunningStat],
    keys: Iterable[tuple[Any, ...]],
    metric: str,
) -> tuple[float, int, int]:
    prior = global_state.get(metric, RunningStat()).mean
    values = [state[(*key, metric)] for key in keys if (*key, metric) in state]
    if not values:
        return prior, 0, 1
    support = sum(value.count for value in values)
    total = sum(value.total for value in values)
    estimate = (total + SMOOTHING_PRIOR_GAMES * prior) / (
        support + SMOOTHING_PRIOR_GAMES
    )
    return float(estimate), support, 0


def _emit_metric_family(
    output: MutableMapping[str, Any],
    *,
    prefix: str,
    state: Mapping[Any, RunningStat],
    global_state: Mapping[str, RunningStat],
    blue_keys: Iterable[tuple[Any, ...]],
    red_keys: Iterable[tuple[Any, ...]],
) -> None:
    blue_keys = list(blue_keys)
    red_keys = list(red_keys)
    for metric in HISTORICAL_METRICS:
        blue, blue_n, blue_missing = _shrunk_metric_mean(
            state, global_state, blue_keys, metric
        )
        red, red_n, red_missing = _shrunk_metric_mean(
            state, global_state, red_keys, metric
        )
        output[f"{prefix}_{metric}"] = blue - red
        output[f"{prefix}_{metric}_support"] = min(blue_n, red_n)
        output[f"{prefix}_{metric}_missing"] = max(blue_missing, red_missing)


def _annotate_feature_availability(matrix: pd.DataFrame) -> pd.DataFrame:
    """Add one explicit authority flag for each model feature group."""

    matrix = matrix.copy()

    def any_support(prefixes: Sequence[str]) -> pd.Series:
        columns = [
            column
            for column in matrix.columns
            if column.endswith("_support")
            and any(column.startswith(prefix) for prefix in prefixes)
        ]
        if not columns:
            return pd.Series(False, index=matrix.index)
        return matrix[columns].max(axis=1) > 0

    matrix[FEATURE_AVAILABILITY_COLUMNS["team_rating"]] = matrix[
        "team_rating_available"
    ].astype(float)
    matrix[FEATURE_AVAILABILITY_COLUMNS["player_rating"]] = matrix[
        "player_rating_available"
    ].astype(float)
    matrix[FEATURE_AVAILABILITY_COLUMNS["player_exact_performance"]] = (
        (matrix["history_unique_player_maps_min"] > 0)
        & any_support(("history_player_champion_result_residual",))
    ).astype(float)
    matrix[FEATURE_AVAILABILITY_COLUMNS["exact_ally_enemy_pairs"]] = (
        any_support(("history_ally_champion_pair",))
        & any_support(("history_enemy_champion_pair",))
    ).astype(float)
    matrix[FEATURE_AVAILABILITY_COLUMNS["checkpoint_forecasts"]] = matrix[
        "forecast_curve_available"
    ].astype(float)
    matrix[FEATURE_AVAILABILITY_COLUMNS["parity_conditioned_performance"]] = (
        any_support(("parity_player_champion",))
    ).astype(float)
    matrix[FEATURE_AVAILABILITY_COLUMNS["team_momentum"]] = (
        (matrix["team_momentum_available"] == 1)
        & (matrix["player_momentum_available"] == 1)
    ).astype(float)
    matrix[FEATURE_AVAILABILITY_COLUMNS["patch_exact_performance"]] = (
        any_support(("patch_player_champion", "patch_champion"))
    ).astype(float)
    return matrix


def _coverage_scopes(
    matrix: pd.DataFrame,
    *,
    prospective_start: Any | None = None,
    prospective_end: Any | None = None,
) -> tuple[pd.DataFrame, list[tuple[str, str, pd.DataFrame, str]], dict[str, Any]]:
    work = matrix.copy()
    work["date"] = pd.to_datetime(work["date"], utc=True, errors="raise")
    start = pd.Timestamp(prospective_start) if prospective_start is not None else None
    end = pd.Timestamp(prospective_end) if prospective_end is not None else None
    if start is not None and start.tzinfo is None:
        start = start.tz_localize("UTC")
    if end is not None and end.tzinfo is None:
        end = end.tz_localize("UTC")
    if (start is None) != (end is None):
        raise AtomizedResearchError("prospective start and end must be supplied together")
    if start is not None and (start < CONSUMED_TEST_END or end <= start):
        raise AtomizedResearchError("prospective range must start after consumed evidence")

    work["split"] = np.select(
        [work["date"] < TRAIN_END, work["date"] < VALIDATION_END],
        ["train", "validation"],
        default="consumed_audit",
    )
    development = work[work["date"] < VALIDATION_END]
    audit = work[work["date"] >= VALIDATION_END]
    if start is not None:
        prospective_mask = (work["date"] >= start) & (work["date"] < end)
        prospective = work[prospective_mask]
        audit = audit[~audit.index.isin(prospective.index)]
    else:
        prospective = work.iloc[0:0]

    scopes: list[tuple[str, str, pd.DataFrame, str]] = [
        ("overall", "development", development, "development_gate")
    ]
    scopes.extend(
        ("split", str(value), group, "development_gate")
        for value, group in development.groupby("split", sort=True)
    )
    scopes.extend(
        ("league", str(value), group, "development_gate")
        for value, group in development.groupby("league", sort=True)
    )
    scopes.extend(
        (
            "split_league",
            f"{split}|{league}",
            group,
            "development_gate",
        )
        for (split, league), group in development.groupby(
            ["split", "league"], sort=True
        )
    )
    if len(audit):
        scopes.append(("audit", "consumed", audit, "report_only"))
        scopes.extend(
            ("audit_league", str(value), group, "report_only")
            for value, group in audit.groupby("league", sort=True)
        )
    if start is not None:
        scopes.append(("prospective", "all", prospective, "report_only"))
        scopes.extend(
            ("prospective_league", str(value), group, "report_only")
            for value, group in prospective.groupby("league", sort=True)
        )
    range_receipt = {
        "development_end_exclusive": VALIDATION_END.isoformat(),
        "prospective_start_inclusive": start.isoformat() if start is not None else None,
        "prospective_end_exclusive": end.isoformat() if end is not None else None,
        "configuration_performs_source_io": False,
    }
    return work, scopes, range_receipt


def feature_group_coverage_report(
    matrix: pd.DataFrame,
    *,
    thresholds: Mapping[str, float] = FEATURE_COVERAGE_THRESHOLDS,
    minimum_rows: int = 20,
    prospective_start: Any | None = None,
    prospective_end: Any | None = None,
) -> dict[str, Any]:
    """Gate development covariates and report later coverage without feedback."""

    work, scopes, range_receipt = _coverage_scopes(
        matrix,
        prospective_start=prospective_start,
        prospective_end=prospective_end,
    )
    missing = [
        FEATURE_AVAILABILITY_COLUMNS[group]
        for group in thresholds
        if FEATURE_AVAILABILITY_COLUMNS[group] not in work.columns
    ]
    if missing:
        raise AtomizedResearchError(
            f"feature group availability columns are missing: {missing}"
        )

    rows: list[dict[str, Any]] = []
    for dimension, value, group, gate_role in scopes:
        eligible = gate_role == "development_gate" and len(group) >= minimum_rows
        for feature_group, threshold in thresholds.items():
            coverage = float(group[FEATURE_AVAILABILITY_COLUMNS[feature_group]].mean()) if len(group) else 0.0
            rows.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "feature_group": feature_group,
                    "rows": int(len(group)),
                    "coverage": coverage,
                    "missing_rate": 1.0 - coverage,
                    "threshold": float(threshold),
                    "gate_role": gate_role,
                    "eligible_for_gate": eligible,
                    "passed": bool(coverage >= threshold) if eligible else None,
                }
            )
    failures = [row for row in rows if row["passed"] is False]
    development = work[work["date"] < VALIDATION_END].copy()
    covariate_columns = [
        FEATURE_AVAILABILITY_COLUMNS[group] for group in sorted(thresholds)
    ]
    frozen_covariates = [
        {
            "row": str(row.get("game_uid") or index),
            "date": pd.Timestamp(row["date"]).isoformat(),
            "league": str(row["league"]),
            **{column: float(row[column]) for column in covariate_columns},
        }
        for index, row in development.sort_values(
            ["date", "league"], kind="stable"
        ).iterrows()
    ]
    eligibility_receipt = {
        "schema_version": "scryglass:atomized-rf-development-coverage-freeze:v1",
        "uses_outcomes_or_targets": False,
        "development_rows": int(len(development)),
        "covariate_sha256": canonical_sha256(frozen_covariates),
        "policy_sha256": canonical_sha256(
            {
                "thresholds": dict(thresholds),
                "minimum_rows": minimum_rows,
                "development_end_exclusive": VALIDATION_END.isoformat(),
            }
        ),
        "frozen_before_prospective_audit": True,
    }
    return {
        "minimum_rows": minimum_rows,
        "thresholds": dict(thresholds),
        "range": range_receipt,
        "eligibility_receipt": eligibility_receipt,
        "rows": rows,
        "failures": failures,
        "passed": not failures,
    }


def phase_coverage_report(
    matrix: pd.DataFrame,
    *,
    prospective_start: Any | None = None,
    prospective_end: Any | None = None,
) -> dict[str, Any]:
    """Report phase eligibility, targets, and forecasts by split and league."""

    _work, scopes, range_receipt = _coverage_scopes(
        matrix,
        prospective_start=prospective_start,
        prospective_end=prospective_end,
    )
    rows: list[dict[str, Any]] = []
    for dimension, value, group, gate_role in scopes:
        for checkpoint in CHECKPOINTS:
            for metric in ("gold", "xp"):
                target = group[f"target_{metric}_diff_{checkpoint}"].notna()
                forecast = (
                    group[f"forecast_{metric}_available_{checkpoint}"] == 1
                )
                rows.append(
                    {
                        "dimension": dimension,
                        "value": value,
                        "gate_role": gate_role,
                        "checkpoint": checkpoint,
                        "metric": metric,
                        "eligible_rows": int(len(group)),
                        "target_available": int(target.sum()),
                        "target_coverage": float(target.mean()) if len(group) else 0.0,
                        "forecast_available": int(forecast.sum()),
                        "forecast_coverage": float(forecast.mean()) if len(group) else 0.0,
                        "joint_available": int((target & forecast).sum()),
                        "joint_coverage": float((target & forecast).mean())
                        if len(group)
                        else 0.0,
                    }
                )
    return {"range": range_receipt, "rows": rows}


def _recent_mean(history: Mapping[str, deque[float]], key: str) -> tuple[float, int]:
    values = history.get(key)
    if not values:
        return 0.0, 0
    return float(np.mean(values)), len(values)


def _series_ids(maps: pd.DataFrame) -> dict[str, str]:
    """Infer stable series clusters from teams, league, and time gaps.

    OE does not provide one complete series identifier.  Consecutive maps for
    the same unordered team pair and league remain in one cluster when the gap
    is at most eight hours.
    """

    result: dict[str, str] = {}
    last: dict[tuple[str, str, str], tuple[pd.Timestamp, int]] = {}
    for row in maps.sort_values(["date", "game_uid"], kind="stable").to_dict("records"):
        key = (
            str(row.get("league") or "UNKNOWN"),
            min(str(row.get("blue_team_key") or row.get("blue_team")), str(row.get("red_team_key") or row.get("red_team"))),
            max(str(row.get("blue_team_key") or row.get("blue_team")), str(row.get("red_team_key") or row.get("red_team"))),
        )
        timestamp = pd.Timestamp(row["date"])
        prior = last.get(key)
        sequence = 1 if prior is None or timestamp - prior[0] > pd.Timedelta(hours=8) else prior[1]
        if prior is None or timestamp - prior[0] > pd.Timedelta(hours=8):
            sequence = 1 if prior is None else prior[1] + 1
        last[key] = (timestamp, sequence)
        result[str(row["game_uid"])] = f"{key[0]}|{key[1]}|{key[2]}|{sequence}"
    return result


def _validate_no_current_state_features(columns: Sequence[str]) -> None:
    forbidden = [column for column in columns if column.startswith(TARGET_PREFIX)]
    if forbidden:
        raise AtomizedResearchError(f"pre-match feature list contains current-game targets: {forbidden}")
    for column in columns:
        if re.search(r"(?:observed|current)_(?:gold|xp|cs|kill|objective)", column):
            raise AtomizedResearchError(f"pre-match feature contains live state: {column}")


def _load_frames(maps_path: Path, players_path: Path, team_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    maps = pd.read_parquet(maps_path)
    players = pd.read_parquet(players_path)
    teams = pd.read_parquet(team_path)
    for frame in (maps, players, teams):
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
    maps["game_uid"] = maps["game_uid"].astype(str)
    if maps["game_uid"].duplicated().any():
        raise AtomizedResearchError("map grain is duplicated")
    players["game_uid"] = players["game_uid"].where(players["game_uid"].notna(), players["gameid"])
    players["game_uid"] = players["game_uid"].astype(str)
    teams["game_uid"] = teams["game_uid"].where(teams["game_uid"].notna(), teams["gameid"])
    teams["game_uid"] = teams["game_uid"].astype(str)
    return maps, players, teams


def build_layer_a_matrix(
    *,
    base_dataset: Path,
    maps_path: Path,
    players_path: Path,
    team_path: Path,
    identity_overlay_players_path: Path | None = None,
    raw_identity_overlay_csv: Path | None = None,
    cache_dir: Path,
    force: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the strictly lagged statistical-atom matrix.

    Gold and XP checkpoint values update histories after the map.  The same
    fields remain targets for the current map.  They never enter the current
    pre-match feature vector.
    """

    sources = {
        "base_dataset": sha256_path(base_dataset),
        "maps": sha256_path(maps_path),
        "players": sha256_path(players_path),
        "teams": sha256_path(team_path),
        "identity_overlay_players": (
            sha256_path(identity_overlay_players_path)
            if identity_overlay_players_path is not None
            else None
        ),
        "raw_identity_overlay_csv": (
            sha256_path(raw_identity_overlay_csv)
            if raw_identity_overlay_csv is not None
            else None
        ),
        "code": sha256_path(Path(__file__)),
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "momentum_window": MOMENTUM_WINDOW_GAMES,
        "momentum_scale": MOMENTUM_SCALE,
    }
    digest = canonical_sha256(sources)
    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = cache_dir / f"layer-a-{digest}.parquet"
    manifest_path = cache_dir / f"layer-a-{digest}.manifest.json"
    if matrix_path.exists() and manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("matrix_sha256") != sha256_path(matrix_path):
            raise AtomizedResearchError("cached feature matrix digest mismatch")
        return pd.read_parquet(matrix_path), manifest

    started = time.perf_counter()
    base = pd.read_parquet(base_dataset)
    base["date"] = pd.to_datetime(base["date"], utc=True, errors="raise")
    if sha256_path(base_dataset) != LOCKED_DATASET_SHA256:
        raise AtomizedResearchError("locked baseline dataset SHA-256 changed")
    maps, players, teams = _load_frames(maps_path, players_path, team_path)
    identity_overlay: dict[tuple[str, str, str, str], tuple[str, str]] = {}
    raw_identity_overlay: dict[tuple[str, str, str, str, str, str], tuple[str, str]] = {}
    if identity_overlay_players_path is not None:
        overlay = pd.read_parquet(identity_overlay_players_path)
        overlay["game_uid"] = overlay["game_uid"].where(
            overlay["game_uid"].notna(), overlay["gameid"]
        ).astype(str)
        for row in overlay.to_dict("records"):
            player = str(row.get("playerid") or "")
            team = str(row.get("teamid") or "")
            if player.startswith("oe:player:") and team.startswith("oe:team:"):
                identity_overlay[
                    (
                        str(row.get("game_uid")),
                        str(row.get("side") or "").strip().casefold(),
                        str(row.get("position") or "").strip().casefold(),
                        str(row.get("champion") or "").strip().casefold(),
                    )
                ] = (player, team)
    if raw_identity_overlay_csv is not None:
        if sha256_path(raw_identity_overlay_csv) != RAW_2026_IDENTITY_SHA256:
            raise AtomizedResearchError("accepted raw 2026 identity source changed")
        raw_overlay = pd.read_csv(raw_identity_overlay_csv, low_memory=False)
        raw_overlay = raw_overlay[raw_overlay["position"].astype(str).str.casefold() != "team"]
        raw_overlay["date"] = pd.to_datetime(raw_overlay["date"], utc=True, errors="raise")
        for row in raw_overlay.to_dict("records"):
            player = str(row.get("playerid") or "")
            team = str(row.get("teamid") or "")
            if player.startswith("oe:player:") and team.startswith("oe:team:"):
                raw_identity_overlay[
                    (
                        pd.Timestamp(row["date"]).isoformat(),
                        str(row.get("league") or "").strip(),
                        normalize_team(str(row.get("teamname") or "")).strip().casefold(),
                        str(row.get("side") or "").strip().casefold(),
                        str(row.get("position") or "").strip().casefold(),
                        str(row.get("champion") or "").strip().casefold(),
                    )
                ] = (player, team)
    base_ids = set(base["game_uid"].astype(str))
    maps_by_id = maps.set_index("game_uid", drop=False)
    missing_maps = sorted(base_ids - set(maps_by_id.index))
    if missing_maps:
        raise AtomizedResearchError(f"refreshed map source misses {len(missing_maps)} baseline maps")

    player_groups = {key: value.copy() for key, value in players.groupby("game_uid", sort=False)}
    team_groups = {key: value.copy() for key, value in teams.groupby("game_uid", sort=False)}
    base_by_id = base.set_index("game_uid", drop=False)
    base_team_probability_by_id = {
        str(row["game_uid"]): float(1.0 / (1.0 + math.exp(-float(row["base_team_logit"]))))
        for row in base.to_dict("records")
    }
    base_player_probability_by_id = {
        str(row["game_uid"]): float(
            1.0 / (1.0 + math.exp(-float(row["base_player_logit"])))
        )
        for row in base.to_dict("records")
        if _finite(row.get("base_player_logit")) is not None
        and _finite(row.get("player_lineup_complete")) == 1.0
    }
    series = _series_ids(maps[maps["game_uid"].isin(base_ids)].copy())

    player_champion: MutableMapping[Any, RunningStat] = defaultdict(RunningStat)
    ally_pair: MutableMapping[Any, RunningStat] = defaultdict(RunningStat)
    enemy_pair: MutableMapping[Any, RunningStat] = defaultdict(RunningStat)
    forecast: MutableMapping[Any, RunningStat] = defaultdict(RunningStat)
    parity: MutableMapping[Any, RunningStat] = defaultdict(RunningStat)
    patch_player_champion: MutableMapping[Any, RunningStat] = defaultdict(RunningStat)
    patch_champion: MutableMapping[Any, RunningStat] = defaultdict(RunningStat)
    global_metric: MutableMapping[str, RunningStat] = defaultdict(RunningStat)
    player_champion_maps: MutableMapping[tuple[str, str], set[str]] = defaultdict(set)
    team_momentum_history: MutableMapping[str, deque[float]] = defaultdict(
        lambda: deque(maxlen=MOMENTUM_WINDOW_GAMES)
    )
    player_momentum_history: MutableMapping[str, deque[float]] = defaultdict(
        lambda: deque(maxlen=MOMENTUM_WINDOW_GAMES)
    )
    output: list[dict[str, Any]] = []
    rejected_lineups = 0
    exclusion_reasons: dict[str, str] = {}
    overlay_recovered_games: set[str] = set()
    raw_overlay_recovered_games: set[str] = set()
    rating_authority_by_game: dict[str, dict[str, float]] = {}

    # The locked experiment contains these seven leagues. Histories use the
    # same competition universe, which avoids processing unrelated minor-league
    # rows and keeps the feature authority aligned with the evaluation set.
    ordered = maps[
        (maps["date"] < CONSUMED_TEST_END) & maps["league"].isin(TARGET_LEAGUES)
    ].sort_values(["date", "game_uid"], kind="stable")
    for _, same_time in ordered.groupby("date", sort=False):
        pending: list[tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]] = []
        for map_row in same_time.to_dict("records"):
            game_uid = str(map_row["game_uid"])
            lineup = player_groups.get(game_uid)
            team_rows = team_groups.get(game_uid)
            if lineup is None or team_rows is None or len(lineup) != 10 or len(team_rows) != 2:
                rejected_lineups += 1
                if game_uid in base_ids:
                    exclusion_reasons[game_uid] = "incomplete_source_rows"
                continue
            try:
                rows = lineup.to_dict("records")
                for row in rows:
                    overlay_identity = identity_overlay.get(
                        (
                            game_uid,
                            str(row.get("side") or "").strip().casefold(),
                            str(row.get("position") or "").strip().casefold(),
                            str(row.get("champion") or "").strip().casefold(),
                        )
                    )
                    needs_identity = not (
                        str(row.get("playerid") or "").startswith("oe:player:")
                        and str(row.get("teamid") or "").startswith("oe:team:")
                    )
                    if overlay_identity is not None and needs_identity:
                        row["playerid"], row["teamid"] = overlay_identity
                        overlay_recovered_games.add(game_uid)
                    raw_overlay_identity = raw_identity_overlay.get(
                        (
                            pd.Timestamp(map_row["date"]).isoformat(),
                            str(map_row.get("league") or "").strip(),
                            normalize_team(str(row.get("teamname") or "")).strip().casefold(),
                            str(row.get("side") or "").strip().casefold(),
                            str(row.get("position") or "").strip().casefold(),
                            str(row.get("champion") or "").strip().casefold(),
                        )
                    )
                    needs_identity = not (
                        str(row.get("playerid") or "").startswith("oe:player:")
                        and str(row.get("teamid") or "").startswith("oe:team:")
                    )
                    if raw_overlay_identity is not None and needs_identity:
                        row["playerid"], row["teamid"] = raw_overlay_identity
                        raw_overlay_recovered_games.add(game_uid)
                for row in rows:
                    _player_id(row)
                    _team_id(row)
                    _side(row.get("side"))
            except AtomizedResearchError:
                rejected_lineups += 1
                if game_uid in base_ids:
                    exclusion_reasons[game_uid] = "stable_identity_unresolved_or_ambiguous"
                continue
            lineup = pd.DataFrame(rows)
            by_side = {
                side: [row for row in rows if _side(row.get("side")) == side]
                for side in ("Blue", "Red")
            }
            if any(len(by_side[side]) != 5 for side in by_side):
                rejected_lineups += 1
                if game_uid in base_ids:
                    exclusion_reasons[game_uid] = "invalid_side_lineup"
                continue

            if game_uid in base_ids:
                blue = by_side["Blue"]
                red = by_side["Red"]
                blue_pc = [(_player_id(row), str(row["champion"])) for row in blue]
                red_pc = [(_player_id(row), str(row["champion"])) for row in red]

                def pair_keys(side_rows: list[dict[str, Any]], other_rows: list[dict[str, Any]], allied: bool) -> list[Any]:
                    keys: list[Any] = []
                    for row in side_rows:
                        own = str(row["champion"])
                        partner_rows = side_rows if allied else other_rows
                        for partner in partner_rows:
                            champion = str(partner["champion"])
                            if allied and champion == own:
                                continue
                            keys.append((_player_id(row), own, champion))
                    return keys

                blue_ally = pair_keys(blue, red, True)
                red_ally = pair_keys(red, blue, True)
                blue_enemy = pair_keys(blue, red, False)
                red_enemy = pair_keys(red, blue, False)
                patch = normalize_source_patch(
                    map_row.get("patch") or lineup.iloc[0].get("patch"), map_row["date"]
                )
                base_row = base_by_id.loc[game_uid].to_dict()
                rating_authority = _locked_rating_authority(
                    base_row,
                    resolved_roster_sha256=_resolved_roster_sha256(rows),
                )
                rating_authority_by_game[game_uid] = rating_authority
                blue_team = _team_id(blue[0])
                red_team = _team_id(red[0])
                blue_players = [_player_id(row) for row in blue]
                red_players = [_player_id(row) for row in red]
                feature_row: dict[str, Any] = {
                    **base_row,
                    "series_id": series[game_uid],
                    "source_patch": patch,
                    **rating_authority,
                    **_momentum_features(
                        team_momentum_history,
                        player_momentum_history,
                        blue_team,
                        red_team,
                        blue_players,
                        red_players,
                    ),
                }
                blue_unique_player_maps = _unique_player_map_support(
                    player_champion_maps, blue_pc
                )
                red_unique_player_maps = _unique_player_map_support(
                    player_champion_maps, red_pc
                )
                feature_row.update(
                    {
                        "history_unique_player_maps_blue": float(
                            blue_unique_player_maps
                        ),
                        "history_unique_player_maps_red": float(red_unique_player_maps),
                        "history_unique_player_maps_min": float(
                            min(blue_unique_player_maps, red_unique_player_maps)
                        ),
                    }
                )
                _emit_metric_family(
                    feature_row,
                    prefix="history_player_champion",
                    state=player_champion,
                    global_state=global_metric,
                    blue_keys=blue_pc,
                    red_keys=red_pc,
                )
                _emit_metric_family(
                    feature_row,
                    prefix="history_ally_champion_pair",
                    state=ally_pair,
                    global_state=global_metric,
                    blue_keys=blue_ally,
                    red_keys=red_ally,
                )
                _emit_metric_family(
                    feature_row,
                    prefix="history_enemy_champion_pair",
                    state=enemy_pair,
                    global_state=global_metric,
                    blue_keys=blue_enemy,
                    red_keys=red_enemy,
                )
                _emit_metric_family(
                    feature_row,
                    prefix="patch_player_champion",
                    state=patch_player_champion,
                    global_state=global_metric,
                    blue_keys=[(patch, *key) for key in blue_pc],
                    red_keys=[(patch, *key) for key in red_pc],
                )
                _emit_metric_family(
                    feature_row,
                    prefix="patch_champion",
                    state=patch_champion,
                    global_state=global_metric,
                    blue_keys=[(patch, str(row["champion"])) for row in blue],
                    red_keys=[(patch, str(row["champion"])) for row in red],
                )
                phase_available = True
                for checkpoint in CHECKPOINTS:
                    for metric in ("gold", "xp"):
                        state_keys_blue = [
                            (_player_id(row), str(row["champion"]), checkpoint, metric) for row in blue
                        ]
                        state_keys_red = [
                            (_player_id(row), str(row["champion"]), checkpoint, metric) for row in red
                        ]
                        blue_total, blue_support, blue_coverage, blue_missing = (
                            _equal_weight_team_forecast(forecast, state_keys_blue)
                        )
                        red_total, red_support, red_coverage, red_missing = (
                            _equal_weight_team_forecast(forecast, state_keys_red)
                        )
                        available = not (blue_missing or red_missing)
                        phase_available = phase_available and available
                        feature_row[f"forecast_{metric}_diff_{checkpoint}"] = (
                            blue_total - red_total if available else 0.0
                        )
                        feature_row[f"forecast_{metric}_support_{checkpoint}"] = min(
                            blue_support, red_support
                        )
                        feature_row[
                            f"forecast_{metric}_player_coverage_{checkpoint}"
                        ] = min(blue_coverage, red_coverage)
                        feature_row[f"forecast_{metric}_available_{checkpoint}"] = float(
                            available
                        )
                        feature_row[f"forecast_{metric}_missing_{checkpoint}"] = float(
                            not available
                        )
                    _emit_metric_family(
                        feature_row,
                        prefix=f"parity_player_champion_{checkpoint}",
                        state=parity,
                        global_state=global_metric,
                        blue_keys=[
                            (_player_id(row), str(row["champion"]), checkpoint, "gold_xp_both_250")
                            for row in blue
                        ],
                        red_keys=[
                            (_player_id(row), str(row["champion"]), checkpoint, "gold_xp_both_250")
                            for row in red
                        ],
                    )
                    blue_team_row = next(row for row in team_rows.to_dict("records") if _side(row["side"]) == "Blue")
                    feature_row[f"target_gold_diff_{checkpoint}"] = _finite(
                        blue_team_row.get(f"golddiffat{checkpoint}")
                    )
                    feature_row[f"target_xp_diff_{checkpoint}"] = _finite(
                        blue_team_row.get(f"xpdiffat{checkpoint}")
                    )
                feature_row.update(
                    _phase_curve_features(
                        [
                            feature_row[f"forecast_gold_diff_{checkpoint}"]
                            for checkpoint in CHECKPOINTS
                        ],
                        [
                            feature_row[f"forecast_xp_diff_{checkpoint}"]
                            for checkpoint in CHECKPOINTS
                        ],
                        available=phase_available,
                    )
                )
                output.append(feature_row)
            pending.append((map_row, lineup, team_rows))

        # Equal timestamps update only after every feature row at this time exists.
        for map_row, lineup, _team_rows in pending:
            rows = lineup.to_dict("records")
            by_side = {
                side: [row for row in rows if _side(row.get("side")) == side]
                for side in ("Blue", "Red")
            }
            patch = normalize_source_patch(
                map_row.get("patch") or lineup.iloc[0].get("patch"), map_row["date"]
            )
            y_blue = int(map_row["y_blue_win"])
            for side in ("Blue", "Red"):
                side_result = y_blue if side == "Blue" else 1 - y_blue
                game_uid = str(map_row["game_uid"])
                rating_authority = rating_authority_by_game.get(game_uid, {})
                base_team_probability_blue = base_team_probability_by_id.get(game_uid)
                team_residual = None
                if (
                    base_team_probability_blue is not None
                    and rating_authority.get("team_rating_available") == 1.0
                ):
                    base_probability = (
                        base_team_probability_blue
                        if side == "Blue"
                        else 1.0 - base_team_probability_blue
                    )
                    team_residual = float(side_result) - base_probability
                base_player_probability_blue = base_player_probability_by_id.get(game_uid)
                player_residual = None
                if (
                    base_player_probability_blue is not None
                    and rating_authority.get("player_rating_available") == 1.0
                ):
                    base_probability = (
                        base_player_probability_blue
                        if side == "Blue"
                        else 1.0 - base_player_probability_blue
                    )
                    player_residual = float(side_result) - base_probability
                own_rows = by_side[side]
                enemy_rows = by_side["Red" if side == "Blue" else "Blue"]
                if team_residual is not None:
                    team_momentum_history[_team_id(own_rows[0])].append(team_residual)
                for row in own_rows:
                    player = _player_id(row)
                    champion = str(row["champion"])
                    player_champion_maps[(player, champion)].add(
                        str(map_row["game_uid"])
                    )
                    if player_residual is not None:
                        player_momentum_history[player].append(player_residual)
                    values: dict[str, float] = {}
                    if team_residual is not None:
                        values["result_residual"] = team_residual
                    for metric, source in HISTORICAL_METRICS.items():
                        if source == "__result_residual__":
                            continue
                        value = _finite(row.get(source))
                        if value is not None:
                            values[metric] = value
                    for metric, value in values.items():
                        global_metric[metric].add(value)
                        player_champion[(player, champion, metric)].add(value)
                        patch_player_champion[(patch, player, champion, metric)].add(value)
                        patch_champion[(patch, champion, metric)].add(value)
                        for ally in own_rows:
                            ally_champion = str(ally["champion"])
                            if ally_champion != champion:
                                ally_pair[(player, champion, ally_champion, metric)].add(value)
                        for enemy in enemy_rows:
                            enemy_pair[(player, champion, str(enemy["champion"]), metric)].add(value)
                    for checkpoint in CHECKPOINTS:
                        gold = _finite(row.get(f"golddiffat{checkpoint}"))
                        xp = _finite(row.get(f"xpdiffat{checkpoint}"))
                        if gold is not None:
                            forecast[(player, champion, checkpoint, "gold")].add(gold)
                        if xp is not None:
                            forecast[(player, champion, checkpoint, "xp")].add(xp)
                        if gold is not None and xp is not None and abs(gold) <= 250 and abs(xp) <= 250:
                            for metric, value in values.items():
                                parity[
                                    (
                                        player,
                                        champion,
                                        checkpoint,
                                        "gold_xp_both_250",
                                        metric,
                                    )
                                ].add(value)

    matrix = pd.DataFrame(output).sort_values(
        ["date", "game_uid"], kind="stable"
    ).reset_index(drop=True)
    matrix = _annotate_feature_availability(matrix)
    coverage = len(matrix) / len(base)
    if coverage < 0.85:
        raise AtomizedResearchError(
            f"strict lineup authority covers only {len(matrix)} of {len(base)} locked maps"
        )
    excluded_locked_maps = sorted(base_ids - set(matrix["game_uid"].astype(str)))
    coverage_frame = base[["game_uid", "date", "league", "y"]].copy()
    coverage_frame["game_uid"] = coverage_frame["game_uid"].astype(str)
    coverage_frame["included"] = coverage_frame["game_uid"].isin(set(matrix["game_uid"].astype(str)))
    coverage_frame["split"] = np.select(
        [coverage_frame["date"] < TRAIN_END, coverage_frame["date"] < VALIDATION_END],
        ["train", "validation"],
        default="test",
    )
    coverage_frame["patch"] = coverage_frame["game_uid"].map(
        lambda game_uid: (
            str(player_groups[game_uid].iloc[0].get("patch") or "unknown")
            if game_uid in player_groups and len(player_groups[game_uid])
            else "unknown"
        )
    )

    def coverage_table(column: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for value, group in coverage_frame.groupby(column, dropna=False, sort=True):
            rows.append(
                {
                    column: str(value),
                    "total": int(len(group)),
                    "included": int(group["included"].sum()),
                    "coverage": float(group["included"].mean()),
                }
            )
        return rows

    coverage_bias = {
        dimension: coverage_table(dimension)
        for dimension in ("split", "league", "patch", "y")
    }
    material_undercoverage = [
        {"dimension": dimension, **row}
        for dimension, rows in coverage_bias.items()
        for row in rows
        if row["total"] >= 20 and row["coverage"] < max(0.75, coverage - 0.15)
    ]
    feature_coverage = feature_group_coverage_report(matrix)
    phase_coverage = phase_coverage_report(matrix)
    _validate_no_current_state_features(MODEL_COLUMNS)
    if matrix[list(MODEL_COLUMNS)].isna().any().any():
        raise AtomizedResearchError("model feature matrix contains missing values")
    matrix.to_parquet(matrix_path, index=False)
    manifest = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "cache_digest": digest,
        "matrix_path": str(matrix_path),
        "matrix_sha256": sha256_path(matrix_path),
        "rows": int(len(matrix)),
        "locked_rows": int(len(base)),
        "stable_lineup_coverage": coverage,
        "excluded_locked_maps": excluded_locked_maps,
        "exclusion_reason_counts": {
            str(reason): int(count)
            for reason, count in pd.Series(
                [exclusion_reasons.get(game_uid, "unknown") for game_uid in excluded_locked_maps]
            )
            .value_counts()
            .items()
        },
        "identity_overlay": {
            "join_key": ["canonical_game_id", "side", "role", "champion"],
            "source_rows": len(identity_overlay),
            "mapping_sha256": canonical_sha256(
                [
                    [*key, *value]
                    for key, value in sorted(identity_overlay.items())
                ]
            ),
            "recovered_locked_games": len(overlay_recovered_games & base_ids),
            "unresolved_locked_games": len(excluded_locked_maps),
            "later_roster_identity_imported": False,
        },
        "raw_drive_identity_overlay": {
            "join_key": ["timestamp", "league", "team", "side", "role", "champion"],
            "source_rows": len(raw_identity_overlay),
            "mapping_sha256": canonical_sha256(
                [[*key, *value] for key, value in sorted(raw_identity_overlay.items())]
            ),
            "recovered_locked_games": len(raw_overlay_recovered_games & base_ids),
            "unresolved_locked_games": len(excluded_locked_maps),
            "exact_same_map_rows_only": True,
            "drive_revision": RAW_2026_DRIVE_REVISION,
        },
        "coverage_bias": coverage_bias,
        "material_undercoverage": material_undercoverage,
        "feature_group_coverage": feature_coverage,
        "phase_coverage": phase_coverage,
        "evaluation_gate": (
            "test_blocked_material_coverage_bias"
            if material_undercoverage
            else (
                "test_blocked_feature_group_coverage"
                if not feature_coverage["passed"]
                else "eligible"
            )
        ),
        "columns": list(MODEL_COLUMNS),
        "feature_groups": {key: list(value) for key, value in GROUP_COLUMNS.items()},
        "feature_authority": {
            "team_rating": {
                "fields": list(GROUP_COLUMNS["team_rating"]),
                "source": "hash-bound locked momentum dataset",
                "required_upstream_receipt": "scryglass:resolved-rating-source:v1",
                "required_bindings": [
                    "rating_source_sha256",
                    "rating_roster_sha256",
                    "rating_source_available",
                    "rating_receipt_sha256",
                ],
                "missingness": "explicit available and missing flags",
                "temporal_cutoff": "strictly before map",
            },
            "player_rating": {
                "fields": list(GROUP_COLUMNS["player_rating"]),
                "source": "hash-bound locked momentum dataset",
                "identity_rule": "source receipt roster hash must match exact resolved team, player, side, role, and champion IDs",
                "missingness": "neutral numeric value with an explicit missing flag",
                "temporal_cutoff": "strictly before map",
            },
            "historical_statistical_atoms": {
                "metrics": HISTORICAL_METRICS,
                "families": [
                    "player_by_champion",
                    "directed_allied_champion_pair",
                    "directed_enemy_champion_pair",
                ],
                "smoothing": {
                    "method": "training-history global mean",
                    "prior_rows": SMOOTHING_PRIOR_GAMES,
                },
                "model_outputs_per_metric": ["value", "missing"],
                "audit_only_support": True,
                "source_sha256": sources["players"],
                "temporal_cutoff": "strictly before map with equal timestamp batching",
            },
            "phase_forecast": {
                "raw_outputs": [
                    f"expected_{metric}_diff_{checkpoint}"
                    for checkpoint in CHECKPOINTS
                    for metric in ("gold", "xp")
                ],
                "aggregation": "one equal-weight term per current player; blue team total minus red team total",
                "derived_outputs": [
                    "checkpoint slopes",
                    "signed peak checkpoint",
                    "signed peak magnitude",
                ],
                "availability": "explicit per checkpoint and summarized by league",
                "current_checkpoint_use": "target only",
                "source_sha256": sources["players"],
                "temporal_cutoff": "strictly before map",
            },
            "parity_conditioned": {
                "condition": "absolute player gold diff <= 250 and absolute player XP diff <= 250",
                "checkpoints": list(CHECKPOINTS),
                "metrics": list(HISTORICAL_METRICS),
                "model_outputs_per_metric": ["value", "missing"],
                "audit_only_support": True,
                "source_sha256": sources["players"],
                "temporal_cutoff": "strictly before map",
            },
            "momentum": {
                "fields": list(GROUP_COLUMNS["team_momentum"]),
                "definition": "seven-map mean of outcome minus strictly prior base probability, scaled by 80",
                "identity_rule": "residual history updates only after an exact roster-bound rating source receipt passes",
                "current_locked_source_status": "blocked_missing_rating_roster_receipt",
                "missingness": "neutral numeric value with explicit coverage and missing flags",
                "source_sha256": sources["base_dataset"],
                "temporal_cutoff": "strictly before map",
            },
            "patch_statistical_atoms": {
                "families": ["patch_player_champion", "patch_champion"],
                "metrics": list(HISTORICAL_METRICS),
                "model_outputs_per_metric": ["value", "missing"],
                "audit_only_support": True,
                "source_sha256": sources["players"],
                "temporal_cutoff": "same source patch and strictly before map",
            },
        },
        "sources": sources,
        "rejected_lineups": rejected_lineups,
        "leakage_controls": {
            "strictly_prior_updates": True,
            "equal_timestamp_batching": True,
            "stable_player_ids": True,
            "stable_team_ids": True,
            "checkpoint_values_are_current_map_targets_only": True,
            "final_outcome_is_not_a_feature": True,
            "sealed_after_2026_08_08": True,
            "history_competition_scope": list(TARGET_LEAGUES),
        },
        "wall_seconds": time.perf_counter() - started,
    }
    _write_json(manifest_path, manifest)
    return matrix, manifest


def _split(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "train":
        return frame[frame["date"] < TRAIN_END]
    if name == "validation":
        return frame[(frame["date"] >= TRAIN_END) & (frame["date"] < VALIDATION_END)]
    if name == "test":
        return frame[(frame["date"] >= VALIDATION_END) & (frame["date"] < CONSUMED_TEST_END)]
    raise KeyError(name)


def finalize_cached_layer_a_manifest(
    *,
    matrix_path: Path,
    base_dataset: Path,
    historical_players_path: Path,
    overlay_players_path: Path,
) -> dict[str, Any]:
    """Finalize a receipt when matrix creation succeeded before JSON output."""

    matrix = pd.read_parquet(matrix_path)
    base = pd.read_parquet(base_dataset)
    base["date"] = pd.to_datetime(base["date"], utc=True, errors="raise")
    if len(matrix) < 1600 or not set(MODEL_COLUMNS).issubset(matrix.columns):
        raise AtomizedResearchError("cached matrix shape or feature contract changed")
    if matrix["game_uid"].duplicated().any():
        raise AtomizedResearchError("cached matrix has duplicate maps")
    included = set(matrix["game_uid"].astype(str))
    excluded = base[~base["game_uid"].astype(str).isin(included)].copy()
    excluded["reason"] = "stable_identity_unresolved_in_source_and_same_game_overlay"
    excluded_path = matrix_path.with_suffix(".unresolved-identities.json")
    unresolved_payload = {
        "schema_version": "scryglass:atomized-rf-unresolved-identities:v1",
        "rows": [
            {
                "game_uid": str(row["game_uid"]),
                "date": pd.Timestamp(row["date"]).isoformat(),
                "league": str(row["league"]),
                "outcome_blue": int(row["y"]),
                "reason": str(row["reason"]),
            }
            for row in excluded.to_dict("records")
        ],
    }
    _write_json(excluded_path, unresolved_payload)

    coverage = base[["game_uid", "date", "league", "y"]].copy()
    coverage["included"] = coverage["game_uid"].astype(str).isin(included)
    coverage["split"] = np.select(
        [coverage["date"] < TRAIN_END, coverage["date"] < VALIDATION_END],
        ["train", "validation"],
        default="test",
    )
    history = pd.read_parquet(historical_players_path, columns=["game_uid", "gameid", "patch"])
    history["game_uid"] = history["game_uid"].where(
        history["game_uid"].notna(), history["gameid"]
    ).astype(str)
    patch_by_game = history.groupby("game_uid", sort=False)["patch"].first().astype(str)
    coverage["patch"] = coverage["game_uid"].astype(str).map(patch_by_game).fillna("unknown")

    def table(column: str) -> list[dict[str, Any]]:
        return [
            {
                column: str(value),
                "total": int(len(group)),
                "included": int(group["included"].sum()),
                "coverage": float(group["included"].mean()),
            }
            for value, group in coverage.groupby(column, dropna=False, sort=True)
        ]

    coverage_bias = {column: table(column) for column in ("split", "league", "patch", "y")}
    overall = float(coverage["included"].mean())
    material = [
        {"dimension": dimension, **row}
        for dimension, rows in coverage_bias.items()
        for row in rows
        if row["total"] >= 20 and row["coverage"] < max(0.75, overall - 0.15)
    ]
    overlay_sha = sha256_path(overlay_players_path)
    manifest = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "finalization": "recovered_after_receipt_serialization_failure",
        "matrix_path": str(matrix_path),
        "matrix_sha256": sha256_path(matrix_path),
        "rows": int(len(matrix)),
        "locked_rows": int(len(base)),
        "columns": list(MODEL_COLUMNS),
        "feature_groups": {key: list(value) for key, value in GROUP_COLUMNS.items()},
        "sources": {
            "base_dataset": sha256_path(base_dataset),
            "historical_players": sha256_path(historical_players_path),
            "identity_overlay_players": overlay_sha,
        },
        "identity_overlay": {
            "join_key": ["canonical_game_id", "side", "role", "champion"],
            "recovered_locked_games": sum(
                game_uid.startswith("oe:game:") for game_uid in included
            ),
            "unresolved_locked_games": int(len(excluded)),
            "later_roster_identity_imported": False,
        },
        "stable_lineup_coverage": overall,
        "coverage_bias": coverage_bias,
        "material_undercoverage": material,
        "evaluation_gate": (
            "test_blocked_material_coverage_bias" if material else "eligible"
        ),
        "unresolved_identity_artifact": {
            "path": str(excluded_path),
            "sha256": sha256_path(excluded_path),
        },
        "leakage_controls": {
            "strictly_prior_updates": True,
            "equal_timestamp_batching": True,
            "checkpoint_values_are_current_map_targets_only": True,
            "outcome_is_not_a_feature": True,
        },
    }
    manifest_path = matrix_path.with_suffix(".manifest.json")
    _write_json(manifest_path, manifest)
    return manifest


def _ece(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    order = np.argsort(probability)
    groups = np.array_split(order, min(bins, len(order)))
    return float(
        sum(len(indices) * abs(float(y[indices].mean()) - float(probability[indices].mean())) for indices in groups if len(indices))
        / len(y)
    )


def metric_report(y: Sequence[int], probability: Sequence[float]) -> dict[str, Any]:
    target = np.asarray(y, dtype=int)
    values = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "n": int(len(target)),
        "auc": float(roc_auc_score(target, values)) if len(np.unique(target)) == 2 else None,
        "brier": float(brier_score_loss(target, values)),
        "log_loss": float(log_loss(target, values, labels=[0, 1])),
        "ece_equal_frequency_10": _ece(target, values),
    }


def _fit_rf(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    config: RFConfig,
) -> RandomForestClassifier:
    _validate_no_current_state_features(columns)
    model = RandomForestClassifier(**config.as_dict())
    model.fit(frame[list(columns)].astype(float), frame["y"].astype(int))
    return model


def _expanding_series_folds(
    frame: pd.DataFrame, *, requested_folds: int = 3
) -> list[tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    """Create forward-only folds with whole inferred series.

    A series that crosses a fold boundary is omitted from that fold. This
    keeps every training timestamp earlier than every validation timestamp.
    """

    work = frame.reset_index(drop=True).copy()
    work["date"] = pd.to_datetime(work["date"], utc=True, errors="raise")
    bounds = (
        work.groupby("series_id", sort=False)["date"]
        .agg(["min", "max"])
        .sort_values(["min", "max"], kind="stable")
    )
    if len(bounds) < 12:
        raise AtomizedResearchError("series groups are insufficient for chronological folds")
    blocks = [list(block) for block in np.array_split(bounds.index.to_numpy(), requested_folds + 1)]
    folds: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]] = []
    for index in range(1, len(blocks)):
        validation_groups = blocks[index]
        if not validation_groups:
            continue
        validation_start = bounds.loc[validation_groups, "min"].min()
        training_groups = [
            group
            for block in blocks[:index]
            for group in block
            if bounds.loc[group, "max"] < validation_start
        ]
        train_index = np.flatnonzero(work["series_id"].isin(training_groups).to_numpy())
        validation_index = np.flatnonzero(work["series_id"].isin(validation_groups).to_numpy())
        if len(train_index) < 150 or len(validation_index) < 40:
            continue
        train_max = work.iloc[train_index]["date"].max()
        validation_min = work.iloc[validation_index]["date"].min()
        if not train_max < validation_min:
            raise AtomizedResearchError("chronological calibration fold leaks later series")
        folds.append(
            (
                train_index,
                validation_index,
                {
                    "train_rows": int(len(train_index)),
                    "validation_rows": int(len(validation_index)),
                    "train_max": train_max.isoformat(),
                    "validation_min": validation_min.isoformat(),
                    "whole_series": True,
                },
            )
        )
    if len(folds) < 2:
        raise AtomizedResearchError("chronological series folds are insufficient")
    return folds


def _calibration_outer_audit(
    fold_probabilities: Sequence[np.ndarray],
    fold_targets: Sequence[np.ndarray],
) -> dict[str, Any]:
    """Accept calibration only after it improves both proper scores per outer fold."""

    if len(fold_probabilities) != len(fold_targets) or len(fold_probabilities) < 2:
        raise AtomizedResearchError("calibration audit needs two matched outer folds")
    prior_probability: list[np.ndarray] = []
    prior_targets: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for index, (probability, target) in enumerate(
        zip(fold_probabilities, fold_targets)
    ):
        probability = np.clip(np.asarray(probability, dtype=float), 1e-5, 1 - 1e-5)
        target = np.asarray(target, dtype=int)
        if index:
            fit_probability = np.concatenate(prior_probability)
            fit_target = np.concatenate(prior_targets)
            if len(np.unique(fit_target)) < 2:
                rows.append(
                    {
                        "outer_fold": index + 1,
                        "accepted": False,
                        "reason": "prior calibration rows have one outcome class",
                    }
                )
            else:
                fit_logits = np.log(fit_probability / (1 - fit_probability))
                calibrator = LogisticRegression(C=1.0, random_state=RANDOM_SEED)
                calibrator.fit(fit_logits.reshape(-1, 1), fit_target)
                logits = np.log(probability / (1 - probability))
                calibrated = calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]
                raw_metrics = metric_report(target, probability)
                calibrated_metrics = metric_report(target, calibrated)
                accepted = bool(
                    calibrated_metrics["brier"] < raw_metrics["brier"]
                    and calibrated_metrics["log_loss"] < raw_metrics["log_loss"]
                )
                rows.append(
                    {
                        "outer_fold": index + 1,
                        "accepted": accepted,
                        "raw": raw_metrics,
                        "calibrated": calibrated_metrics,
                    }
                )
        prior_probability.append(probability)
        prior_targets.append(target)
    accepted = bool(rows) and all(row["accepted"] for row in rows)
    return {
        "accepted": accepted,
        "acceptance_rule": "Brier score and log loss improve in every development outer fold",
        "outer_folds": rows,
        "learner_family": "same frozen RandomForest configuration",
    }


def _group_calibrator(
    train: pd.DataFrame,
    columns: Sequence[str],
    *,
    config: RFConfig,
) -> tuple[LogisticRegression | None, dict[str, Any]]:
    fold_probabilities: list[np.ndarray] = []
    fold_targets: list[np.ndarray] = []
    fold_receipts: list[dict[str, Any]] = []
    for train_index, validation_index, fold_audit in _expanding_series_folds(train):
        model = _fit_rf(train.iloc[train_index], columns, config=config)
        fold_probabilities.append(
            model.predict_proba(
                train.iloc[validation_index][list(columns)].astype(float)
            )[:, 1]
        )
        fold_targets.append(
            train.iloc[validation_index]["y"].astype(int).to_numpy()
        )
        fold_receipts.append(fold_audit)
    audit = _calibration_outer_audit(fold_probabilities, fold_targets)
    audit["folds"] = fold_receipts
    audit["config"] = config.as_dict()
    if not audit["accepted"]:
        return None, audit
    raw = np.clip(np.concatenate(fold_probabilities), 1e-5, 1 - 1e-5)
    targets = np.concatenate(fold_targets)
    logits = np.log(raw / (1 - raw))
    calibrator = LogisticRegression(C=1.0, random_state=RANDOM_SEED)
    calibrator.fit(logits.reshape(-1, 1), targets)
    return calibrator, audit


def _calibrated_probability(
    model: RandomForestClassifier,
    calibrator: LogisticRegression | None,
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> np.ndarray:
    raw = np.clip(model.predict_proba(frame[list(columns)].astype(float))[:, 1], 1e-5, 1 - 1e-5)
    if calibrator is None:
        return raw
    logits = np.log(raw / (1 - raw))
    return calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]


def reproduce_locked_baseline(dataset: Path) -> dict[str, Any]:
    started = time.perf_counter()
    if sha256_path(dataset) != LOCKED_DATASET_SHA256:
        raise AtomizedResearchError("locked baseline dataset changed")
    frame = pd.read_parquet(dataset)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
    columns = list(LOCKED_BASELINE_COLUMNS)
    train = _split(frame, "train")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(train[columns].astype(float), train["y"].astype(int))
    metrics = {}
    for name in ("validation", "test"):
        part = _split(frame, name)
        metrics[name] = metric_report(part["y"], model.predict_proba(part[columns].astype(float))[:, 1])
    return {
        "schema_version": "scryglass:locked-momentum-rf-replay:v1",
        "dataset_sha256": sha256_path(dataset),
        "features": columns,
        "hyperparameters": {
            "n_estimators": 300,
            "max_depth": 8,
            "min_samples_leaf": 10,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
        },
        "metrics": metrics,
        "wall_seconds": time.perf_counter() - started,
        "authority": {"public": False, "production": False, "probability": False},
    }


def _cluster_bootstrap_auc(
    frame: pd.DataFrame, probability: np.ndarray, *, repetitions: int = 1000
) -> dict[str, Any]:
    target = frame["y"].to_numpy(dtype=int)
    group_values = frame["series_id"].astype(str).to_numpy()
    clusters = [np.flatnonzero(group_values == group) for group in sorted(set(group_values))]
    rng = np.random.default_rng(RANDOM_SEED)
    values: list[float] = []
    for _ in range(repetitions):
        indices = np.concatenate(
            [clusters[index] for index in rng.integers(0, len(clusters), len(clusters))]
        )
        if len(np.unique(target[indices])) == 2:
            values.append(float(roc_auc_score(target[indices], probability[indices])))
    return {
        "cluster": "inferred_series",
        "repetitions": repetitions,
        "median": float(np.median(values)),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _rf_search_candidates() -> list[RFConfig]:
    """Return a bounded search that covers every agreed RF control."""

    return [
        RFConfig(300, 6, 4, "sqrt", "balanced_subsample", True, 0.7),
        RFConfig(600, 6, 10, 0.25, None, True, None),
        RFConfig(300, 8, 4, 0.5, "balanced_subsample", True, 0.7),
        RFConfig(600, 8, 10, "sqrt", "balanced_subsample", True, None),
        RFConfig(300, 10, 8, 0.25, None, True, 0.7),
        RFConfig(600, 10, 16, "sqrt", "balanced_subsample", True, None),
        RFConfig(300, None, 10, "sqrt", "balanced_subsample", True, 0.7),
        RFConfig(600, None, 20, 0.25, None, True, None),
        RFConfig(300, 8, 8, "sqrt", "balanced_subsample", False, None),
        RFConfig(600, 10, 16, 0.25, None, False, None),
        RFConfig(300, None, 20, "sqrt", None, False, None),
        RFConfig(600, 6, 4, 0.5, "balanced_subsample", False, None),
    ]


def _resource_config(config: RFConfig, trees: int) -> RFConfig:
    return RFConfig(
        n_estimators=trees,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        class_weight=config.class_weight,
        bootstrap=config.bootstrap,
        max_samples=config.max_samples,
    )


def _matched_comparison_config(config: RFConfig) -> RFConfig:
    """Keep the frozen learner and tree count for every full-model comparison."""

    return config


def _raw_fold_score(
    frame: pd.DataFrame,
    columns: Sequence[str],
    config: RFConfig,
) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for train_index, validation_index, fold_audit in _expanding_series_folds(frame):
        model = _fit_rf(frame.iloc[train_index], columns, config=config)
        probability = model.predict_proba(
            frame.iloc[validation_index][list(columns)].astype(float)
        )[:, 1]
        metrics.append(metric_report(frame.iloc[validation_index]["y"], probability))
        audit.append(fold_audit)
    return {
        "mean_auc": float(np.mean([value["auc"] for value in metrics])),
        "mean_log_loss": float(np.mean([value["log_loss"] for value in metrics])),
        "mean_brier": float(np.mean([value["brier"] for value in metrics])),
        "folds": metrics,
        "fold_audit": audit,
    }


def select_rf_config(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    columns: Sequence[str],
    *,
    cache_dir: Path,
    matrix_sha256: str,
    baseline_validation_probability: np.ndarray,
) -> tuple[RFConfig, dict[str, Any]]:
    """Run a cached two-stage chronological search before test access."""

    baseline_metric = metric_report(validation["y"], baseline_validation_probability)
    auc_floor = float(baseline_metric["auc"])
    search_identity = {
        "schema": "scryglass:atomized-rf-search:v1",
        "matrix_sha256": matrix_sha256,
        "columns_sha256": canonical_sha256(list(columns)),
        "candidates": [candidate.as_dict() for candidate in _rf_search_candidates()],
        "selection": "validation log loss with locked validation AUC floor",
        "matched_baseline_validation": baseline_metric,
        "matched_baseline_probability_sha256": hashlib.sha256(
            np.asarray(baseline_validation_probability, dtype="<f8").tobytes()
        ).hexdigest(),
    }
    digest = canonical_sha256(search_identity)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"rf-search-{digest}.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        selected = payload["selected_config"]
        return RFConfig(
            n_estimators=int(selected["n_estimators"]),
            max_depth=selected["max_depth"],
            min_samples_leaf=int(selected["min_samples_leaf"]),
            max_features=selected["max_features"],
            class_weight=selected["class_weight"],
            bootstrap=bool(selected["bootstrap"]),
            max_samples=selected["max_samples"],
        ), payload

    started = time.perf_counter()
    stage_one: list[dict[str, Any]] = []
    for candidate in _rf_search_candidates():
        resource = _resource_config(candidate, 120)
        score = _raw_fold_score(train, columns, resource)
        stage_one.append({"config": candidate.as_dict(), "resource_trees": 120, **score})
    survivors = sorted(stage_one, key=lambda row: row["mean_log_loss"])[:4]
    stage_two: list[dict[str, Any]] = []
    for row in survivors:
        raw = row["config"]
        candidate = RFConfig(
            n_estimators=int(raw["n_estimators"]),
            max_depth=raw["max_depth"],
            min_samples_leaf=int(raw["min_samples_leaf"]),
            max_features=raw["max_features"],
            class_weight=raw["class_weight"],
            bootstrap=bool(raw["bootstrap"]),
            max_samples=raw["max_samples"],
        )
        model = _fit_rf(train, columns, config=candidate)
        probability = model.predict_proba(validation[list(columns)].astype(float))[:, 1]
        stage_two.append({"config": candidate.as_dict(), **metric_report(validation["y"], probability)})
    passing = [row for row in stage_two if row["auc"] is not None and row["auc"] >= auc_floor]
    eligible = passing or stage_two
    selected_row = min(eligible, key=lambda row: row["log_loss"])
    payload = {
        **search_identity,
        "cache_digest": digest,
        "stage_one": stage_one,
        "stage_two": stage_two,
        "auc_floor": auc_floor,
        "auc_floor_passed": bool(passing),
        "selected_config": selected_row["config"],
        "selected_validation": {
            key: selected_row[key] for key in ("n", "auc", "brier", "log_loss", "ece_equal_frequency_10")
        },
        "test_accessed_during_selection": False,
        "wall_seconds": time.perf_counter() - started,
    }
    _write_json(path, payload)
    return RFConfig(
        n_estimators=int(selected_row["config"]["n_estimators"]),
        max_depth=selected_row["config"]["max_depth"],
        min_samples_leaf=int(selected_row["config"]["min_samples_leaf"]),
        max_features=selected_row["config"]["max_features"],
        class_weight=selected_row["config"]["class_weight"],
        bootstrap=bool(selected_row["config"]["bootstrap"]),
        max_samples=selected_row["config"]["max_samples"],
    ), payload


def _cluster_bootstrap_differences(
    frame: pd.DataFrame,
    candidate_probability: np.ndarray,
    baseline_probability: np.ndarray,
    *,
    repetitions: int = 1000,
) -> dict[str, Any]:
    target = frame["y"].to_numpy(dtype=int)
    group_values = frame["series_id"].astype(str).to_numpy()
    clusters = [np.flatnonzero(group_values == group) for group in sorted(set(group_values))]
    rng = np.random.default_rng(RANDOM_SEED)
    values: dict[str, list[float]] = {"auc": [], "brier": [], "log_loss": []}
    for _ in range(repetitions):
        indices = np.concatenate(
            [clusters[index] for index in rng.integers(0, len(clusters), len(clusters))]
        )
        if len(np.unique(target[indices])) != 2:
            continue
        candidate = metric_report(target[indices], candidate_probability[indices])
        baseline = metric_report(target[indices], baseline_probability[indices])
        values["auc"].append(candidate["auc"] - baseline["auc"])
        values["brier"].append(candidate["brier"] - baseline["brier"])
        values["log_loss"].append(candidate["log_loss"] - baseline["log_loss"])
    return {
        metric: {
            "median": float(np.median(samples)),
            "lower_95": float(np.quantile(samples, 0.025)),
            "upper_95": float(np.quantile(samples, 0.975)),
        }
        for metric, samples in values.items()
    } | {"cluster": "inferred_series", "repetitions": repetitions}


def run_layer_a_experiment(
    matrix: pd.DataFrame,
    *,
    cache_dir: Path,
    matrix_sha256: str,
    test_eligible: bool,
    frozen_config: RFConfig | None = None,
    frozen_search_receipt: Mapping[str, Any] | None = None,
    prospective_start: Any | None = None,
    prospective_end: Any | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    matrix = matrix.copy()
    matrix["date"] = pd.to_datetime(matrix["date"], utc=True, errors="raise")
    feature_coverage = feature_group_coverage_report(
        matrix,
        prospective_start=prospective_start,
        prospective_end=prospective_end,
    )
    phase_coverage = phase_coverage_report(
        matrix,
        prospective_start=prospective_start,
        prospective_end=prospective_end,
    )
    if not feature_coverage["passed"]:
        raise AtomizedResearchError(
            "feature-group coverage gate blocks model fitting"
        )
    train = _split(matrix, "train")
    validation = _split(matrix, "validation")
    test = _split(matrix, "test")
    if tuple(train.shape)[0] < 500 or len(validation) < 200 or len(test) < 100:
        raise AtomizedResearchError("chronological experiment splits are too small")
    selection_baseline_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    selection_baseline_model.fit(
        train[list(LOCKED_BASELINE_COLUMNS)].astype(float), train["y"].astype(int)
    )
    selection_baseline_validation_probability = selection_baseline_model.predict_proba(
        validation[list(LOCKED_BASELINE_COLUMNS)].astype(float)
    )[:, 1]
    if frozen_config is None:
        selected, search = select_rf_config(
            train,
            validation,
            MODEL_COLUMNS,
            cache_dir=cache_dir,
            matrix_sha256=matrix_sha256,
            baseline_validation_probability=selection_baseline_validation_probability,
        )
    else:
        selected = frozen_config
        search = dict(frozen_search_receipt or {})
        search["reused_frozen_config"] = True
        search["test_accessed_during_selection"] = False
    evaluation_parts = [("validation", validation)]
    if test_eligible:
        evaluation_parts.append(("test", test))
    matched_baseline_key = canonical_sha256(
        {
            "matrix_sha256": matrix_sha256,
            "config": selected.as_dict(),
            "columns": list(LOCKED_BASELINE_COLUMNS),
            "splits": [name for name, _ in evaluation_parts],
            "calibration": "expanding-series-sigmoid-v2-proper-score-gated",
        }
    )
    matched_baseline_path = cache_dir / f"matched-baseline-{matched_baseline_key}.npz"
    matched_baseline_calibration_path = cache_dir / (
        f"matched-baseline-{matched_baseline_key}.calibration.json"
    )
    if matched_baseline_path.exists() and matched_baseline_calibration_path.exists():
        cached_baseline = np.load(matched_baseline_path)
        baseline_probabilities = {
            name: cached_baseline[name] for name, _ in evaluation_parts
        }
        baseline_calibration_audit = json.loads(
            matched_baseline_calibration_path.read_text(encoding="utf-8")
        )
        matched_baseline_cache_hit = True
    else:
        matched_baseline_model = _fit_rf(
            train, LOCKED_BASELINE_COLUMNS, config=selected
        )
        matched_baseline_calibrator, baseline_calibration_audit = _group_calibrator(
            train, LOCKED_BASELINE_COLUMNS, config=selected
        )
        baseline_probabilities = {
            name: _calibrated_probability(
                matched_baseline_model,
                matched_baseline_calibrator,
                part,
                LOCKED_BASELINE_COLUMNS,
            )
            for name, part in evaluation_parts
        }
        np.savez_compressed(matched_baseline_path, **baseline_probabilities)
        _write_json(
            matched_baseline_calibration_path, baseline_calibration_audit
        )
        matched_baseline_cache_hit = False
    prediction_key = canonical_sha256(
        {
            "matrix_sha256": matrix_sha256,
            "config": selected.as_dict(),
            "columns": list(MODEL_COLUMNS),
            "splits": [name for name, _ in evaluation_parts],
            "calibration": "expanding-series-sigmoid-v2-proper-score-gated",
        }
    )
    prediction_path = cache_dir / f"predictions-{prediction_key}.npz"
    calibration_path = cache_dir / f"predictions-{prediction_key}.calibration.json"
    if prediction_path.exists() and calibration_path.exists():
        cached = np.load(prediction_path)
        probabilities = {name: cached[name] for name, _ in evaluation_parts}
        calibration_audit = json.loads(calibration_path.read_text(encoding="utf-8"))
        prediction_cache_hit = True
    else:
        model = _fit_rf(train, MODEL_COLUMNS, config=selected)
        calibrator, calibration_audit = _group_calibrator(
            train, MODEL_COLUMNS, config=selected
        )
        probabilities = {
            name: _calibrated_probability(model, calibrator, part, MODEL_COLUMNS)
            for name, part in evaluation_parts
        }
        np.savez_compressed(prediction_path, **probabilities)
        _write_json(calibration_path, calibration_audit)
        prediction_cache_hit = False
    report: dict[str, Any] = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "estimand": "pre_match_map_outcome_composite",
        "features": list(MODEL_COLUMNS),
        "feature_groups": {key: list(value) for key, value in GROUP_COLUMNS.items()},
        "feature_group_coverage": feature_coverage,
        "phase_coverage": phase_coverage,
        "hyperparameters": selected.as_dict(),
        "search": search,
        "calibration": {
            "method": "expanding chronological whole-series sigmoid",
            **calibration_audit,
        },
        "comparison_learner": {
            "rule": "candidate, incremental baseline, ablations, regional transfers, and shuffle control use the same frozen RF configuration and calibration gate",
            "config": selected.as_dict(),
        },
        "prediction_cache": {
            "path": str(prediction_path),
            "sha256": sha256_path(prediction_path),
            "hit": prediction_cache_hit,
        },
        "splits": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "consumed_test": int(len(test)),
            "consumed_test_evaluated": test_eligible,
            "prospective": feature_coverage["range"],
        },
        "metrics": {
            name: metric_report(part["y"], probabilities[name])
            for name, part in evaluation_parts
        },
        "matched_baseline": {
            "learner_config": selected.as_dict(),
            "calibration": baseline_calibration_audit,
            "cache": {
                "path": str(matched_baseline_path),
                "sha256": sha256_path(matched_baseline_path),
                "hit": matched_baseline_cache_hit,
            },
            "validation": metric_report(
                validation["y"], baseline_probabilities["validation"]
            ),
            "test": (
                metric_report(test["y"], baseline_probabilities["test"])
                if "test" in baseline_probabilities
                else {"status": "blocked_material_coverage_bias"}
            ),
            "probability_vectors": {
                "validation_sha256": hashlib.sha256(
                    np.asarray(
                        baseline_probabilities["validation"], dtype="<f8"
                    ).tobytes()
                ).hexdigest(),
                "test_sha256": (
                    hashlib.sha256(
                        np.asarray(
                            baseline_probabilities["test"], dtype="<f8"
                        ).tobytes()
                    ).hexdigest()
                    if "test" in baseline_probabilities
                    else None
                ),
            },
        },
        "selection_reference": {
            "learner": "locked 300-tree baseline harness",
            "calibration": "raw",
            "validation": metric_report(
                validation["y"], selection_baseline_validation_probability
            ),
            "incremental_comparison": False,
        },
        "group_ablation": {},
        "regional_test": {},
        "patch_transfer": {},
        "sparse_coverage": {},
        "phase_error": {},
        "authority": {
            "research_only": True,
            "public": False,
            "production": False,
            "probability": False,
            "promotion": False,
        },
    }
    for group, removed in GROUP_COLUMNS.items():
        columns = [column for column in MODEL_COLUMNS if column not in removed]
        ablation_config = _matched_comparison_config(selected)
        candidate = _fit_rf(train, columns, config=ablation_config)
        candidate_calibrator, _ = _group_calibrator(
            train, columns, config=ablation_config
        )
        report["group_ablation"][group] = {
            name: metric_report(
                part["y"], _calibrated_probability(candidate, candidate_calibrator, part, columns)
            )
            for name, part in evaluation_parts
        }
    for league in sorted(test["league"].unique()) if test_eligible else []:
        held = test[test["league"] == league]
        if len(held) < 20 or held["y"].nunique() < 2:
            continue
        regional_train = pd.concat([train, validation], ignore_index=True)
        regional_train = regional_train[regional_train["league"] != league]
        if len(regional_train) < 300:
            continue
        transfer_config = _matched_comparison_config(selected)
        regional_model = _fit_rf(regional_train, MODEL_COLUMNS, config=transfer_config)
        # Calibration stays group-aware and excludes the held region.
        regional_calibrator, _ = _group_calibrator(
            regional_train, MODEL_COLUMNS, config=transfer_config
        )
        report["regional_test"][league] = metric_report(
            held["y"], _calibrated_probability(regional_model, regional_calibrator, held, MODEL_COLUMNS)
        )
    transfer_frame = test if test_eligible else validation
    transfer_probability = probabilities["test" if test_eligible else "validation"]
    for patch in sorted(transfer_frame["source_patch"].astype(str).unique()):
        held = transfer_frame[transfer_frame["source_patch"].astype(str) == patch]
        if len(held) >= 20 and held["y"].nunique() == 2:
            held_probability = transfer_probability[
                transfer_frame["source_patch"].astype(str).to_numpy() == patch
            ]
            report["patch_transfer"][patch] = metric_report(held["y"], held_probability)
    for checkpoint in CHECKPOINTS:
        report["phase_error"][str(checkpoint)] = {}
        for metric in ("gold", "xp"):
            target = f"target_{metric}_diff_{checkpoint}"
            feature = f"forecast_{metric}_diff_{checkpoint}"
            phase_frame = test if test_eligible else validation
            available = phase_frame[target].notna() & (
                phase_frame[f"forecast_{metric}_available_{checkpoint}"] == 1
            )
            report["phase_error"][str(checkpoint)][metric] = {
                "split": "test" if test_eligible else "validation",
                "n": int(available.sum()),
                "rmse": float(
                    mean_squared_error(
                        phase_frame.loc[available, target], phase_frame.loc[available, feature]
                    )
                    ** 0.5
                )
                if available.any()
                else None,
            }
    if test_eligible and "test" in baseline_probabilities:
        report["test_auc_series_cluster_bootstrap"] = _cluster_bootstrap_auc(
            test, probabilities["test"]
        )
        report["baseline_same_matrix"] = metric_report(
            test["y"], baseline_probabilities["test"]
        )
        report["test_difference_bootstrap"] = _cluster_bootstrap_differences(
            test, probabilities["test"], baseline_probabilities["test"]
        )
    else:
        report["test_gate"] = {
            "status": "blocked",
            "reason": "material stable-identity coverage bias",
        }
    support_total = transfer_frame["history_unique_player_maps_min"].astype(float)
    low_cut = float(support_total.quantile(0.25))
    high_cut = float(support_total.quantile(0.75))
    for label, mask in (
        ("lowest_quartile", support_total <= low_cut),
        ("highest_quartile", support_total >= high_cut),
    ):
        held = transfer_frame[mask]
        if len(held) and held["y"].nunique() == 2:
            report["sparse_coverage"][label] = {
                "split": "test" if test_eligible else "validation",
                "support_unit": "unique prior player-map observations on the less-supported side",
                "support_boundary": low_cut if label == "lowest_quartile" else high_cut,
                **metric_report(held["y"], transfer_probability[mask.to_numpy()]),
            }
    # Controls do not consume new hyperparameter choices.
    shuffled = train.copy()
    shuffled["y"] = np.random.default_rng(RANDOM_SEED).permutation(shuffled["y"].to_numpy())
    shuffled_model = _fit_rf(
        shuffled,
        MODEL_COLUMNS,
        config=_matched_comparison_config(selected),
    )
    report["shuffle_controls"] = {
        "training_labels": metric_report(
            transfer_frame["y"],
            shuffled_model.predict_proba(
                transfer_frame[list(MODEL_COLUMNS)].astype(float)
            )[:, 1],
        )
    }
    report["wall_seconds"] = time.perf_counter() - started
    return report


def _git_show(repo: Path, commit: str, locator: str) -> bytes:
    process = subprocess.run(
        ["git", "-C", str(repo), "show", f"{commit}:{locator}"],
        capture_output=True,
        check=False,
    )
    if process.returncode:
        raise AtomizedResearchError(f"LCC commit {commit} lacks {locator}")
    return process.stdout


def exact_mechanic_keys(atoms: Sequence[Mapping[str, Any]]) -> set[str]:
    """Return exact LCC mechanic inputs without family or ontology labels."""

    result: set[str] = set()
    for atom in atoms:
        atom_id = str(atom.get("atom_id") or "").strip()
        trigger = str(atom.get("trigger") or "").strip()
        target = str(atom.get("target_policy") or "").strip()
        if not atom_id or not trigger or not target:
            raise AtomizedResearchError("exact atom lacks ID, trigger, or target policy")
        prefix = f"id={atom_id}|trigger={trigger}|target={target}"
        result.add(prefix)
        parameters = atom.get("parameters")
        if not isinstance(parameters, Mapping):
            raise AtomizedResearchError("exact atom parameters are missing")
        for name, raw in sorted(parameters.items()):
            if name == "damage_type" and isinstance(raw, str):
                result.add(f"{prefix}|parameter={name}|value={raw}")
            elif isinstance(raw, (int, float)) and math.isfinite(float(raw)):
                result.add(f"{prefix}|parameter={name}")
            elif isinstance(raw, list) and any(_finite(value) is not None for value in raw):
                result.add(f"{prefix}|parameter={name}|rank_vector")
            elif isinstance(raw, Mapping):
                for child, value in sorted(raw.items()):
                    if _finite(value) is not None:
                        result.add(f"{prefix}|parameter={name}.{child}")
        for relation in atom.get("relations") or []:
            result.add(f"{prefix}|relation={relation}")
        for state in atom.get("states") or []:
            if isinstance(state, Mapping) and state.get("state"):
                result.add(f"{prefix}|state={state['state']}")
    lowered = "\n".join(result).casefold()
    found = sorted(label for label in FORBIDDEN_GENERAL_LABELS if re.search(rf"(?:^|[=|._-]){label}(?:$|[=|._-])", lowered))
    if found:
        raise AtomizedResearchError(f"broad mechanic labels entered exact feature space: {found}")
    return result


def mechanics_inventory(
    *,
    lcc_repo: Path,
    bridge_26_15: Path,
    bridge_26_16: Path,
    receipt_26_16: Path,
    maps_path: Path,
    base_dataset: Path,
) -> dict[str, Any]:
    """Audit exact patch-time mechanics coverage without fitting a model."""

    maps = pd.read_parquet(maps_path, columns=["game_uid", "patch", "date"])
    maps["date"] = pd.to_datetime(maps["date"], utc=True, errors="raise")
    inventory: dict[str, Any] = {
        "schema_version": MECHANICS_SCHEMA_VERSION,
        "rules": {
            "family_counts_excluded": True,
            "ontology_probabilities_excluded": True,
            "broad_labels_excluded": sorted(FORBIDDEN_GENERAL_LABELS),
            "historical_backfill_forbidden": True,
        },
        "snapshots": {},
    }
    process = subprocess.run(
        [
            "git",
            "-C",
            str(lcc_repo),
            "ls-tree",
            "-r",
            "--name-only",
            LCC_26_15_SEED_COMMIT,
            "data/atoms",
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    seed_paths = sorted(
        path
        for path in process.stdout.splitlines()
        if re.fullmatch(r"data/atoms/[a-z0-9]+\.atoms\.json", path)
    )
    seed_hashes: dict[str, str] = {}
    seed_atoms = 0
    seed_numeric_parameters: set[str] = set()
    seed_categorical_fields = {
        "atom_id",
        "behavior",
        "trigger",
        "target_policy",
        "damage_type",
        "relation",
        "state",
        "cycle",
        "affects_type_flag_bit",
    }
    for locator in seed_paths:
        raw = _git_show(lcc_repo, LCC_26_15_SEED_COMMIT, locator)
        atoms = json.loads(raw)
        if not isinstance(atoms, list):
            raise AtomizedResearchError(f"invalid 26.15 seed atom file: {locator}")
        seed_hashes[locator] = hashlib.sha256(raw).hexdigest()
        seed_atoms += len(atoms)
        for atom in atoms:
            for name, value in (atom.get("parameters") or {}).items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    if name == "range" and abs(float(value)) >= 10000:
                        continue
                    if name != "affects_type_flags":
                        seed_numeric_parameters.add(str(name))
    inventory["ledger"] = {
        "architecture": "versioned_game_atom_ledger",
        "seed": {
            "public_patch": "26.15",
            "commit": LCC_26_15_SEED_COMMIT,
            "champion_files": len(seed_paths),
            "atoms": seed_atoms,
            "file_manifest_sha256": canonical_sha256(seed_hashes),
            "status": (
                "available" if len(seed_paths) == 173 and seed_atoms == 6017 else "unavailable"
            ),
        },
        "stable_atom_id": (
            "entity_type/entity_key/spell_or_passive/source_atom_id/behavior/ordinal"
        ),
        "entity_types": [
            "champion",
            "spell",
            "passive",
            "item",
            "rune",
            "objective",
            "buff",
            "debuff",
            "effect",
            "trigger",
            "target",
            "formula_term",
            "cooldown",
            "cost",
            "range",
            "duration",
            "stack",
            "reset",
            "state_transition",
        ],
        "patch_event_contract": {
            "required": [
                "from_patch",
                "to_patch",
                "source_receipts",
                "prior_ledger_sha256",
                "delta_sha256",
                "result_ledger_sha256",
                "atom_changes",
            ],
            "allowed_changes": [
                "value",
                "formula",
                "relation",
                "activation",
            ],
            "unchanged_atoms": "carried forward through the hash chain",
            "silent_snapshot_reuse": "rejected",
        },
        "model_surface": {
            "prematch": "champion-native atoms plus strictly prior build and rune distributions",
            "live": "prematch atoms plus observed items, runes, buffs, debuffs, and state",
        },
        "raw_field_contract": {
            "categorical_fields": sorted(seed_categorical_fields),
            "numeric_parameters": sorted(seed_numeric_parameters),
            "missing_mask_per_numeric_parameter": True,
            "excluded": [
                "family",
                "tempo_class",
                "d2_burst",
                "d2_dps",
                "mean_range_aggregate",
                "sentinel_range",
                "duplicate_derived_value",
            ],
            "units": {
                "cooldown": "seconds",
                "range": "game_units",
                "cost": "resource_units",
                "duration": "seconds",
                "ratio": "source_formula_ratio",
            },
        },
    }
    for source_patch, public_patch, bridge_path in (
        ("16.15", "26.15", bridge_26_15),
        ("16.16", "26.16", bridge_26_16),
    ):
        bridge_bytes = bridge_path.read_bytes()
        bridge = json.loads(bridge_bytes)
        commit = str(bridge.get("provenance", {}).get("lcc_commit") or "")
        expected = {
            locator: digest
            for locator, digest in bridge.get("provenance", {}).get("file_sha256", {}).items()
            if locator.startswith("data/atoms/") and locator.endswith(".atoms.json")
        }
        valid: dict[str, list[dict[str, Any]]] = {}
        missing: list[str] = []
        changed: list[str] = []
        for locator, digest in sorted(expected.items()):
            try:
                raw = _git_show(lcc_repo, commit, locator)
            except AtomizedResearchError:
                missing.append(locator)
                continue
            if hashlib.sha256(raw).hexdigest() != digest:
                changed.append(locator)
                continue
            atoms = json.loads(raw)
            if not isinstance(atoms, list):
                changed.append(locator)
                continue
            valid[locator] = atoms
        exact_keys: set[str] = set()
        if len(valid) >= 170 and not missing and not changed:
            for atoms in valid.values():
                exact_keys.update(exact_mechanic_keys(atoms))
        status = "available" if len(valid) >= 170 and not missing and not changed else "unavailable"
        patch_maps = maps[maps["patch"].astype(str) == source_patch]
        inventory["snapshots"][public_patch] = {
            "status": status,
            "source_patch": source_patch,
            "lcc_commit": commit,
            "bridge_path": str(bridge_path),
            "bridge_sha256": hashlib.sha256(bridge_bytes).hexdigest(),
            "expected_detail_files": len(expected),
            "valid_detail_files": len(valid),
            "missing_files": missing,
            "changed_files": changed,
            "exact_feature_key_count": len(exact_keys),
            "accepted_maps": int(patch_maps["game_uid"].nunique()),
            "date_min": patch_maps["date"].min().isoformat() if len(patch_maps) else None,
            "date_max": patch_maps["date"].max().isoformat() if len(patch_maps) else None,
        }
    current = inventory["snapshots"]["26.16"]
    receipt = json.loads(receipt_26_16.read_text(encoding="utf-8"))
    if current["bridge_sha256"] != LCC_26_16_BRIDGE_SHA256:
        raise AtomizedResearchError("26.16 bridge raw SHA-256 changed")
    if current["lcc_commit"] != LCC_26_16_COMMIT:
        raise AtomizedResearchError("26.16 LCC commit changed")
    if receipt.get("public_patch") != "26.16" or receipt.get("client_patch") != "16.16":
        raise AtomizedResearchError("26.16 receipt patch identity changed")
    aggregate_path = Path("data/lol/v2/champions/atom-corpus-aggregate-v2.json")
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    current.update(
        {
            "status": "rejected_hybrid_authority",
            "exact_feature_key_count": 0,
            "aggregate_path": str(aggregate_path),
            "aggregate_sha256": sha256_path(aggregate_path),
            "aggregate_lcc_commit": aggregate.get("provenance", {}).get("lcc_commit"),
            "authority_audit": {
                "wiki_patch": "16.16",
                "binary_atom_patch": "16.15",
                "binary_atom_commit": "e5917c4d7d94e6ac32fab5c7f120db4d00c3cfeb",
                "matched_binary_files": 167,
                "decision": "reject",
                "reason": "The aggregate combines 16.16 Wiki input with the 16.15 LCC binary atom corpus.",
                "required_repair": "Apply verified 26.16 delta events to the 26.15 seed ledger.",
            },
        }
    )
    inventory["fit_decision"] = {
        "status": "blocked",
        "reason": "A patch-matched 26.16 mechanics snapshot is absent. The accepted patch also has seven maps.",
        "minimum_required": "patch-time exact snapshots across train, validation, and test",
        "prospective_only": True,
    }
    base = pd.read_parquet(base_dataset, columns=["game_uid", "date", "league"])
    base["game_uid"] = base["game_uid"].astype(str)
    patch_15 = maps[maps["patch"].astype(str) == "16.15"]
    rated_patch_15 = patch_15[patch_15["game_uid"].astype(str).isin(set(base["game_uid"]))]
    inventory["mechanics_source_matrix"] = {
        "public_patch": "26.15",
        "source_patch": "16.15",
        "exact_mechanics_maps": int(patch_15["game_uid"].nunique()),
        "full_composite_intersection_maps": int(rated_patch_15["game_uid"].nunique()),
        "exact_mechanics_date_min": patch_15["date"].min().isoformat(),
        "exact_mechanics_date_max": patch_15["date"].max().isoformat(),
        "full_composite_chronological_split": "unavailable",
        "fit_status": "blocked",
        "reason": (
            "Only the late consumed-test period intersects the locked full-composite matrix; "
            "a separate train, validation, and holdout split is absent."
        ),
        "mechanics_only_fit": "withheld because it would omit required rating and history groups",
    }
    return inventory


def live_state_contract() -> dict[str, Any]:
    return {
        "schema_version": "scryglass:atomized-rf-live-state-contract:v1",
        "status": "research_contract_only",
        "inputs_after_map_start": {
            str(checkpoint): [
                f"observed_gold_diff_{checkpoint}",
                f"observed_xp_diff_{checkpoint}",
                f"observed_cs_diff_{checkpoint}",
                f"observed_kills_{checkpoint}",
                f"observed_assists_{checkpoint}",
                f"observed_deaths_{checkpoint}",
            ]
            for checkpoint in CHECKPOINTS
        },
        "prematch_exclusion": "Observed checkpoint state is never a pre-match feature.",
        "gold_30": {"status": "unavailable", "reason": "OE has no gold@30 field."},
        "authority": {"public": False, "production": False, "probability": False},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "inventory", "layer-a", "all"), default="all")
    parser.add_argument(
        "--base-dataset",
        type=Path,
        default=Path("/private/tmp/scryglass-momentum-autoresearch/momentum-dataset.parquet"),
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=Path("/private/tmp/scryglass-momentum-autoresearch/rating-comparison.json"),
    )
    parser.add_argument(
        "--oe-root", type=Path, default=Path("/private/tmp/scryglass-r9e-promotion/oe_live_16_16_v3")
    )
    parser.add_argument(
        "--historical-oe-root",
        type=Path,
        default=Path("/Users/river/Projects/scryglass/data/lol/warehouse/parquet/oe_live"),
    )
    parser.add_argument(
        "--lcc-repo", type=Path, default=Path("/Users/river/Projects/league-combat-calculator")
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("/private/tmp/scryglass-atomized-rf-cache"))
    parser.add_argument("--prospective-start")
    parser.add_argument("--prospective-end")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "estimands": {
            "composition_only_descriptive_draft_score": "unchanged_external_product",
            "pre_match_composite": "research_only",
            "live_state_composite": "separate_research_contract",
        },
        "authority": {
            "research_only": True,
            "public": False,
            "production": False,
            "probability": False,
            "promotion": False,
        },
    }
    if args.baseline_report.exists() and sha256_path(args.baseline_report) != LOCKED_REPORT_SHA256:
        raise AtomizedResearchError("locked momentum report SHA-256 changed")
    if args.mode in {"baseline", "all"}:
        report["locked_baseline"] = reproduce_locked_baseline(args.base_dataset)
    if args.mode in {"inventory", "all"}:
        report["mechanics_layer"] = mechanics_inventory(
            lcc_repo=args.lcc_repo,
            bridge_26_15=Path("data/lol/v2/champions/lcc-atom-bridge-v1.json"),
            bridge_26_16=Path("data/lol/v2/champions/lcc-atom-bridge-26.16.json"),
            receipt_26_16=Path("data/lol/v2/champions/lcc-atom-refresh-26.16-receipt.json"),
            maps_path=args.oe_root / "maps.parquet",
            base_dataset=args.base_dataset,
        )
    if args.mode in {"layer-a", "all"}:
        matrix, manifest = build_layer_a_matrix(
            base_dataset=args.base_dataset,
            maps_path=args.historical_oe_root / "maps.parquet",
            players_path=args.historical_oe_root / "oe_player_games.parquet",
            team_path=args.historical_oe_root / "oe_team_games.parquet",
            identity_overlay_players_path=args.oe_root / "oe_player_games.parquet",
            raw_identity_overlay_csv=Path(
                "/Users/river/Library/Application Support/Scryglass Worker/runtime/data/lol/warehouse/raw/2026_LoL_esports_match_data_from_OraclesElixir.csv"
            ),
            cache_dir=args.cache_dir,
            force=args.force,
        )
        report["layer_a_matrix"] = manifest
        report["layer_a_experiment"] = run_layer_a_experiment(
            matrix,
            cache_dir=args.cache_dir,
            matrix_sha256=manifest["matrix_sha256"],
            test_eligible=manifest["evaluation_gate"] == "eligible",
            prospective_start=args.prospective_start,
            prospective_end=args.prospective_end,
        )
    report["live_state"] = live_state_contract()
    report["report_sha256"] = canonical_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
