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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_squared_error, roc_auc_score

from lol_kills.etl.aliases import normalize_team


SCHEMA_VERSION = "scryglass:atomized-rf-composite-research:v1"
FEATURE_SCHEMA_VERSION = "scryglass:atomized-rf-layer-a:v1"
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
                output.extend((f"{stem}_{metric}", f"{stem}_{metric}_support", f"{stem}_{metric}_missing"))
    return tuple(output)


GROUP_COLUMNS: dict[str, tuple[str, ...]] = {
    "team_rating": ("base_team_logit", "team_rating_diff_scaled"),
    "player_rating": (
        "base_player_logit",
        "player_rating_diff_scaled",
        "player_lineup_complete",
    ),
    "player_exact_performance": _metric_columns(("history_player_champion",)),
    "exact_ally_enemy_pairs": _metric_columns(
        ("history_ally_champion_pair", "history_enemy_champion_pair")
    ),
    "checkpoint_forecasts": tuple(
        value
        for checkpoint in CHECKPOINTS
        for value in (
            f"forecast_gold_diff_{checkpoint}",
            f"forecast_gold_support_{checkpoint}",
            f"forecast_xp_diff_{checkpoint}",
            f"forecast_xp_support_{checkpoint}",
        )
    )
    + tuple(
        value
        for first, second in zip(CHECKPOINTS, CHECKPOINTS[1:])
        for value in (f"forecast_gold_slope_{first}_{second}", f"forecast_xp_slope_{first}_{second}")
    )
    + ("forecast_peak_checkpoint", "forecast_peak_magnitude"),
    "parity_conditioned_performance": _metric_columns(
        ("parity_player_champion",), checkpoints=CHECKPOINTS
    ),
    "team_momentum": (
        "team_momentum_points_diff",
        "player_momentum_points_diff",
        "team_momentum_count_difference",
        "player_momentum_count_difference",
        "player_momentum_coverage",
    ),
    "patch_exact_performance": _metric_columns(
        ("patch_player_champion", "patch_champion")
    ),
}
MODEL_COLUMNS = ("blue_side",) + tuple(
    column for columns in GROUP_COLUMNS.values() for column in columns
)
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
    base_probability_by_id = {
        str(row["game_uid"]): float(1.0 / (1.0 + math.exp(-float(row["base_team_logit"]))))
        for row in base.to_dict("records")
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
    output: list[dict[str, Any]] = []
    rejected_lineups = 0
    exclusion_reasons: dict[str, str] = {}
    overlay_recovered_games: set[str] = set()
    raw_overlay_recovered_games: set[str] = set()

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
                feature_row: dict[str, Any] = {
                    **base_row,
                    "series_id": series[game_uid],
                    # The hash-bound momentum dataset stores outcome minus the
                    # strictly prior base probability.  Scale 80 is applied
                    # only after the seven-map residual window is formed.
                    "team_momentum_points_diff": MOMENTUM_SCALE
                    * float(base_row["team_residual_diff_g7"]),
                    "player_momentum_points_diff": MOMENTUM_SCALE
                    * float(base_row["player_residual_diff_g7"]),
                    "team_momentum_count_difference": float(base_row["team_count_diff_g7"]),
                    "player_momentum_count_difference": float(base_row["player_count_diff_g7"]),
                    "player_momentum_coverage": float(base_row["player_coverage_g7"]),
                    "source_patch": patch,
                }
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
                for checkpoint in CHECKPOINTS:
                    for metric in ("gold", "xp"):
                        state_keys_blue = [
                            (_player_id(row), str(row["champion"]), checkpoint, metric) for row in blue
                        ]
                        state_keys_red = [
                            (_player_id(row), str(row["champion"]), checkpoint, metric) for row in red
                        ]
                        value, support = _side_difference(forecast, state_keys_blue, state_keys_red)
                        feature_row[f"forecast_{metric}_diff_{checkpoint}"] = value
                        feature_row[f"forecast_{metric}_support_{checkpoint}"] = support
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
                for first, second in zip(CHECKPOINTS, CHECKPOINTS[1:]):
                    span = float(second - first)
                    feature_row[f"forecast_gold_slope_{first}_{second}"] = (
                        feature_row[f"forecast_gold_diff_{second}"]
                        - feature_row[f"forecast_gold_diff_{first}"]
                    ) / span
                    feature_row[f"forecast_xp_slope_{first}_{second}"] = (
                        feature_row[f"forecast_xp_diff_{second}"]
                        - feature_row[f"forecast_xp_diff_{first}"]
                    ) / span
                curve = np.asarray(
                    [
                        feature_row[f"forecast_gold_diff_{checkpoint}"] / 1000.0
                        + feature_row[f"forecast_xp_diff_{checkpoint}"] / 1000.0
                        for checkpoint in CHECKPOINTS
                    ],
                    dtype=float,
                )
                peak_index = int(np.argmax(curve))
                feature_row["forecast_peak_checkpoint"] = float(CHECKPOINTS[peak_index])
                feature_row["forecast_peak_magnitude"] = float(curve[peak_index])
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
                base_probability_blue = base_probability_by_id.get(str(map_row["game_uid"]))
                residual = None
                if base_probability_blue is not None:
                    base_probability = (
                        base_probability_blue if side == "Blue" else 1.0 - base_probability_blue
                    )
                    residual = float(side_result) - base_probability
                own_rows = by_side[side]
                enemy_rows = by_side["Red" if side == "Blue" else "Blue"]
                for row in own_rows:
                    player = _player_id(row)
                    champion = str(row["champion"])
                    values: dict[str, float] = {}
                    if residual is not None:
                        values["result_residual"] = residual
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

    matrix = pd.DataFrame(output).sort_values(["date", "game_uid"], kind="stable").reset_index(drop=True)
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
        "evaluation_gate": (
            "test_blocked_material_coverage_bias"
            if material_undercoverage
            else "eligible"
        ),
        "columns": list(MODEL_COLUMNS),
        "feature_groups": {key: list(value) for key, value in GROUP_COLUMNS.items()},
        "feature_authority": {
            "team_rating": {
                "fields": ["base_team_logit", "team_rating_diff_scaled"],
                "source": "hash-bound locked momentum dataset",
                "temporal_cutoff": "strictly before map",
            },
            "player_rating": {
                "fields": ["base_player_logit", "player_rating_diff_scaled", "player_lineup_complete"],
                "source": "hash-bound locked momentum dataset",
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
                "outputs_per_metric": ["value", "support", "missing"],
                "source_sha256": sources["players"],
                "temporal_cutoff": "strictly before map with equal timestamp batching",
            },
            "phase_forecast": {
                "raw_outputs": [
                    f"expected_{metric}_diff_{checkpoint}"
                    for checkpoint in CHECKPOINTS
                    for metric in ("gold", "xp")
                ],
                "derived_outputs": ["checkpoint slopes", "peak checkpoint", "peak magnitude"],
                "current_checkpoint_use": "target only",
                "source_sha256": sources["players"],
                "temporal_cutoff": "strictly before map",
            },
            "parity_conditioned": {
                "condition": "absolute player gold diff <= 250 and absolute player XP diff <= 250",
                "checkpoints": list(CHECKPOINTS),
                "metrics": list(HISTORICAL_METRICS),
                "outputs_per_metric": ["value", "support", "missing"],
                "source_sha256": sources["players"],
                "temporal_cutoff": "strictly before map",
            },
            "momentum": {
                "fields": list(GROUP_COLUMNS["team_momentum"]),
                "definition": "seven-map mean of outcome minus strictly prior base probability, scaled by 80",
                "default_when_receipt_disabled": 0,
                "receipt_enabled_in_research_matrix": True,
                "source_sha256": sources["base_dataset"],
                "temporal_cutoff": "strictly before map",
            },
            "patch_statistical_atoms": {
                "families": ["patch_player_champion", "patch_champion"],
                "metrics": list(HISTORICAL_METRICS),
                "outputs_per_metric": ["value", "support", "missing"],
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


def _group_calibrator(
    train: pd.DataFrame,
    columns: Sequence[str],
    *,
    config: RFConfig,
) -> tuple[LogisticRegression, list[dict[str, Any]]]:
    oof: list[float] = []
    targets: list[int] = []
    audit: list[dict[str, Any]] = []
    calibration_config = RFConfig(
        n_estimators=min(config.n_estimators, 250),
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        class_weight=config.class_weight,
        bootstrap=config.bootstrap,
        max_samples=config.max_samples,
    )
    for train_index, validation_index, fold_audit in _expanding_series_folds(train):
        model = _fit_rf(train.iloc[train_index], columns, config=calibration_config)
        oof.extend(
            model.predict_proba(train.iloc[validation_index][list(columns)].astype(float))[:, 1]
        )
        targets.extend(train.iloc[validation_index]["y"].astype(int).tolist())
        audit.append(fold_audit)
    raw = np.clip(np.asarray(oof, dtype=float), 1e-5, 1 - 1e-5)
    logits = np.log(raw / (1 - raw))
    calibrator = LogisticRegression(C=1.0, random_state=RANDOM_SEED)
    calibrator.fit(logits.reshape(-1, 1), np.asarray(targets, dtype=int))
    return calibrator, audit


def _calibrated_probability(
    model: RandomForestClassifier,
    calibrator: LogisticRegression,
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> np.ndarray:
    raw = np.clip(model.predict_proba(frame[list(columns)].astype(float))[:, 1], 1e-5, 1 - 1e-5)
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
) -> dict[str, Any]:
    started = time.perf_counter()
    matrix = matrix.copy()
    matrix["date"] = pd.to_datetime(matrix["date"], utc=True, errors="raise")
    train = _split(matrix, "train")
    validation = _split(matrix, "validation")
    test = _split(matrix, "test")
    if tuple(train.shape)[0] < 500 or len(validation) < 200 or len(test) < 100:
        raise AtomizedResearchError("chronological experiment splits are too small")
    baseline_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    baseline_model.fit(train[list(LOCKED_BASELINE_COLUMNS)].astype(float), train["y"].astype(int))
    baseline_validation_probability = baseline_model.predict_proba(
        validation[list(LOCKED_BASELINE_COLUMNS)].astype(float)
    )[:, 1]
    baseline_test_probability = (
        baseline_model.predict_proba(test[list(LOCKED_BASELINE_COLUMNS)].astype(float))[:, 1]
        if test_eligible
        else None
    )
    if frozen_config is None:
        selected, search = select_rf_config(
            train,
            validation,
            MODEL_COLUMNS,
            cache_dir=cache_dir,
            matrix_sha256=matrix_sha256,
            baseline_validation_probability=baseline_validation_probability,
        )
    else:
        selected = frozen_config
        search = dict(frozen_search_receipt or {})
        search["reused_frozen_config"] = True
        search["test_accessed_during_selection"] = False
    evaluation_parts = [("validation", validation)]
    if test_eligible:
        evaluation_parts.append(("test", test))
    prediction_key = canonical_sha256(
        {
            "matrix_sha256": matrix_sha256,
            "config": selected.as_dict(),
            "columns": list(MODEL_COLUMNS),
            "splits": [name for name, _ in evaluation_parts],
            "calibration": "expanding-series-sigmoid-v1",
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
        _write_json(calibration_path, {"folds": calibration_audit})
        prediction_cache_hit = False
    report: dict[str, Any] = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "estimand": "pre_match_map_outcome_composite",
        "features": list(MODEL_COLUMNS),
        "feature_groups": {key: list(value) for key, value in GROUP_COLUMNS.items()},
        "hyperparameters": selected.as_dict(),
        "search": search,
        "calibration": {
            "method": "expanding chronological whole-series sigmoid",
            "folds": (
                calibration_audit.get("folds", [])
                if isinstance(calibration_audit, Mapping)
                else calibration_audit
            ),
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
            "prospective_after": "2026-08-08T21:50:46Z",
        },
        "metrics": {
            name: metric_report(part["y"], probabilities[name])
            for name, part in evaluation_parts
        },
        "matched_baseline": {
            "validation": metric_report(validation["y"], baseline_validation_probability),
            "test": (
                metric_report(test["y"], baseline_test_probability)
                if baseline_test_probability is not None
                else {"status": "blocked_material_coverage_bias"}
            ),
            "probability_vectors": {
                "validation_sha256": hashlib.sha256(
                    np.asarray(baseline_validation_probability, dtype="<f8").tobytes()
                ).hexdigest(),
                "test_sha256": (
                    hashlib.sha256(
                        np.asarray(baseline_test_probability, dtype="<f8").tobytes()
                    ).hexdigest()
                    if baseline_test_probability is not None
                    else None
                ),
            },
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
        ablation_config = _resource_config(selected, min(selected.n_estimators, 250))
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
        transfer_config = _resource_config(selected, min(selected.n_estimators, 250))
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
            available = phase_frame[target].notna()
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
    if test_eligible and baseline_test_probability is not None:
        report["test_auc_series_cluster_bootstrap"] = _cluster_bootstrap_auc(
            test, probabilities["test"]
        )
        report["baseline_same_matrix"] = metric_report(test["y"], baseline_test_probability)
        report["test_difference_bootstrap"] = _cluster_bootstrap_differences(
            test, probabilities["test"], baseline_test_probability
        )
    else:
        report["test_gate"] = {
            "status": "blocked",
            "reason": "material stable-identity coverage bias",
        }
    support_columns = [
        column
        for column in MODEL_COLUMNS
        if column.endswith("_support") and column in transfer_frame.columns
    ]
    support_total = transfer_frame[support_columns].sum(axis=1)
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
                "support_boundary": low_cut if label == "lowest_quartile" else high_cut,
                **metric_report(held["y"], transfer_probability[mask.to_numpy()]),
            }
    # Controls do not consume new hyperparameter choices.
    shuffled = train.copy()
    shuffled["y"] = np.random.default_rng(RANDOM_SEED).permutation(shuffled["y"].to_numpy())
    shuffled_model = _fit_rf(
        shuffled,
        MODEL_COLUMNS,
        config=_resource_config(selected, min(selected.n_estimators, 250)),
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
        )
    report["live_state"] = live_state_contract()
    report["report_sha256"] = canonical_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
