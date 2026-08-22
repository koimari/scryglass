"""Build strict-prior composition atoms and player-form features.

The producer uses the existing ``composition_signal`` implementation.  It
fits one composition model per accepted calendar date.  A target date sees
only earlier calendar dates.  It emits a source-bound JSON artifact for the
four-way Draft Score benchmark.

The player-form ledger uses final Oracle's Elixir fields from earlier dates.
It does not use the target game's final fields or result.  It is a raw
historical feature, not an outcome-fitted player rating.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lol_kills.research import composition_signal
from lol_kills.v2.tierlists.accepted_census import identity_sha256


SCHEMA_VERSION = "scryglass:strict-prior-composition-atoms:v1"
FORM_SCHEMA_VERSION = "scryglass:strict-prior-player-form:v1"
FORM_METRICS = (
    "dpm",
    "damageshare",
    "earnedgoldshare",
    "cspm",
    "kills",
    "deaths",
    "assists",
)
FORM_WEIGHTS = {
    "dpm": 1.0,
    "damageshare": 1.0,
    "earnedgoldshare": 1.0,
    "cspm": 0.5,
    "kills": 0.5,
    "deaths": -0.5,
    "assists": 0.5,
}
FORM_SCALES = {
    "dpm": 1000.0,
    "damageshare": 0.20,
    "earnedgoldshare": 0.20,
    "cspm": 10.0,
    "kills": 5.0,
    "deaths": 5.0,
    "assists": 8.0,
}
ROLES = ("top", "jng", "mid", "bot", "sup")
STATIC_COMPONENTS = (
    "base",
    "ally_synergy",
    "enemy_counter",
    "same_role",
    "archetype_interactions",
)

# The composition producer needs a minimum number of prior maps before it can
# fit a model.  The first model-eligible maps can therefore have complete
# ten-player input while no prior fitted composition model exists.  A neutral
# zero edge is a valid strict-prior baseline for that state.  It carries an
# explicit evidence status so it cannot be confused with fitted composition
# evidence or with a missing source row.
COLD_START_STATUS = "cold_start_neutral"
COLD_START_REASON = "No earlier accepted games support this signal yet."
COLD_START_EDGE = {
    "base": 0.0,
    "ally_synergy": 0.0,
    "enemy_counter": 0.0,
    "same_role": 0.0,
    "archetype_interactions": 0.0,
    "total": 0.0,
}
COLD_START_CONTRACT = {
    "status": COLD_START_STATUS,
    "fit_through": None,
    "edge_components": "all_zero",
    "reason": COLD_START_REASON,
}


_PARALLEL_SCORE_CONTEXT: tuple[Any, ...] | None = None


def _score_composition_batch_worker(
    batch_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if _PARALLEL_SCORE_CONTEXT is None:
        raise StrictPriorAtomError("parallel composition worker is not initialized")
    (
        games,
        source,
        cache_dir,
        worker_commit,
        min_support_games,
        min_training_games,
    ) = _PARALLEL_SCORE_CONTEXT
    result = composition_signal.score_games_temporally(
        games,
        target_game_ids=batch_ids,
        cache_dir=cache_dir,
        source_digest=str(source["source_identity_sha256"]),
        worker_commit=worker_commit,
        min_support_games=min_support_games,
        min_training_games=min_training_games,
        composition_only=True,
    )
    return result.signals, result.audit


class StrictPriorAtomError(ValueError):
    """The strict-prior producer cannot create a trusted artifact."""


class _RunningStats:
    """Small online aggregate used by the expanding form pass.

    The old implementation kept every prior value and called ``np.mean`` and
    ``np.std`` for each target player.  This stores the same population
    statistics once per player-role and role bucket.  A bucket is updated only
    after its whole target date has been scored.
    """

    __slots__ = ("count", "total", "squared_total")

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.squared_total = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.squared_total += value * value

    def mean(self) -> float | None:
        return self.total / self.count if self.count else None

    def std(self) -> float | None:
        if not self.count:
            return None
        mean = self.total / self.count
        variance = self.squared_total / self.count - mean * mean
        return math.sqrt(max(variance, 0.0))


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise StrictPriorAtomError("value is not canonical JSON") from error


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_path(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise StrictPriorAtomError(f"file is missing or unsafe: {path}")
    return _sha_bytes(path.read_bytes())


def _hash_record(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": _sha_bytes(raw)}


def _require_hash(value: object, label: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64:
        raise StrictPriorAtomError(f"{label} is not SHA-256")
    try:
        int(text, 16)
    except ValueError as error:
        raise StrictPriorAtomError(f"{label} is not SHA-256") from error
    return text


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise StrictPriorAtomError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StrictPriorAtomError(f"{label} cannot be read") from error
    if not isinstance(value, dict):
        raise StrictPriorAtomError(f"{label} must be an object")
    return value


def load_source_receipt(path: Path) -> dict[str, Any]:
    source = _load_json(path, "source receipt")
    required = {
        "source_as_of",
        "source_game_count",
        "source_identity_sha256",
        "accepted_game_ids",
        "receipt_sha256",
        "source_files",
    }
    if not required.issubset(source):
        raise StrictPriorAtomError("source receipt is incomplete")
    claimed = _require_hash(source["receipt_sha256"], "source receipt hash")
    payload = dict(source)
    payload.pop("receipt_sha256", None)
    if _sha_bytes(_canonical(payload)) != claimed:
        raise StrictPriorAtomError("source receipt self hash changed")
    accepted = tuple(sorted(str(value) for value in source["accepted_game_ids"]))
    if not accepted or len(set(accepted)) != len(accepted):
        raise StrictPriorAtomError("source accepted IDs are invalid")
    if int(source["source_game_count"]) != len(accepted):
        raise StrictPriorAtomError("source game count changed")
    expected_identity = identity_sha256(accepted)
    if str(source["source_identity_sha256"]).lower() != expected_identity:
        raise StrictPriorAtomError("source identity does not match accepted IDs")
    return source


def _source_file_binding(source: Mapping[str, Any], key: str, path: Path) -> dict[str, Any]:
    files = source.get("source_files")
    record = files.get(key) if isinstance(files, Mapping) else None
    if not isinstance(record, Mapping):
        raise StrictPriorAtomError(f"source receipt has no {key} file record")
    actual = _hash_record(path)
    if (
        int(record.get("bytes", -1)) != actual["bytes"]
        or str(record.get("sha256", "")).lower() != actual["sha256"]
    ):
        raise StrictPriorAtomError(f"{key} source bytes changed")
    return {
        "locator": str(record.get("locator") or ""),
        "path": str(path),
        "bytes": actual["bytes"],
        "sha256": actual["sha256"],
    }


def _game_id_column(frame: pd.DataFrame) -> str:
    for column in ("game_uid", "gameid", "oe_gameid"):
        if column in frame.columns:
            return column
    raise StrictPriorAtomError("source frame has no game identity")


def load_maps(path: Path, source: Mapping[str, Any]) -> pd.DataFrame:
    _source_file_binding(source, "maps", path)
    frame = pd.read_parquet(path)
    column = _game_id_column(frame)
    if "date" not in frame.columns:
        raise StrictPriorAtomError("map ledger has no date")
    result = pd.DataFrame(
        {
            "game_id": frame[column].map(str),
            "date": pd.to_datetime(frame["date"], utc=True, errors="coerce"),
            "target": pd.to_numeric(
                frame["y_blue_win"] if "y_blue_win" in frame else None,
                errors="coerce",
            ),
        }
    )
    accepted = {str(value) for value in source["accepted_game_ids"]}
    result = result[result["game_id"].isin(accepted)].copy()
    if set(result["game_id"]) != accepted or result["game_id"].duplicated().any():
        raise StrictPriorAtomError("map ledger does not match accepted census")
    if result["date"].isna().any():
        raise StrictPriorAtomError("map dates are invalid")
    return result.sort_values(["date", "game_id"], kind="stable").reset_index(drop=True)


def load_players(path: Path, source: Mapping[str, Any]) -> pd.DataFrame:
    _source_file_binding(source, "players", path)
    frame = pd.read_parquet(path)
    required = {"game_uid", "date", "side", "position", "playername", "teamname", "champion", "result"}
    if not required.issubset(frame.columns):
        raise StrictPriorAtomError("player ledger lacks composition columns")
    accepted = {str(value) for value in source["accepted_game_ids"]}
    frame = frame[frame["game_uid"].map(str).isin(accepted)].copy()
    counts = frame["game_uid"].map(str).value_counts()
    if set(counts.index) != accepted or not counts.eq(10).all():
        raise StrictPriorAtomError("player ledger does not contain ten rows per accepted map")
    frame["game_uid"] = frame["game_uid"].map(str)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    if frame["date"].isna().any():
        raise StrictPriorAtomError("player dates are invalid")
    return frame


def _neutral_strength(ids: set[str]) -> dict[str, dict[str, float]]:
    # composition_signal's public validator expects control fields even on its
    # composition-only path.  The fitted model excludes these fields.  The
    # neutral scaffold is bound in the receipt and never enters the output.
    return {game_id: {"mu_diff": 0.0, "sigma_pair": 0.0} for game_id in ids}


def build_games(players: pd.DataFrame, accepted_ids: set[str]) -> list[dict[str, Any]]:
    games = composition_signal.build_composition_games(
        players,
        strength_features=_neutral_strength(accepted_ids),
    )
    seen = {str(game["game_uid"]) for game in games}
    if not seen.issubset(accepted_ids):
        raise StrictPriorAtomError("composition builder emitted an unknown map")
    return games


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _edge_from_signal(signal: Mapping[str, Any]) -> dict[str, float] | None:
    if str(signal.get("status")) != "available":
        return None
    blue = signal.get("blue")
    red = signal.get("red")
    if not isinstance(blue, Mapping) or not isinstance(red, Mapping):
        return None
    blue_components = blue.get("components")
    red_components = red.get("components")
    if not isinstance(blue_components, Mapping) or not isinstance(red_components, Mapping):
        return None
    values: dict[str, float] = {}
    mapping = {
        "base": "base",
        "ally_synergy": "ally_synergy",
        "enemy_counter": "enemy_counter",
        "same_role": "same_role",
        # The existing composition producer calls its atom term "atomized".
        # Keep the public five-component contract while binding the mapping.
        "archetype_interactions": "atomized",
    }
    for output_name, source_name in mapping.items():
        left = _finite(blue_components.get(source_name))
        right = _finite(red_components.get(source_name))
        if left is None or right is None:
            return None
        values[output_name] = left - right
    values["total"] = sum(values.values())
    return values


def _cold_start_edge(signal: Mapping[str, Any]) -> dict[str, float] | None:
    """Return an explicit neutral edge for the no-history state only."""

    if (
        str(signal.get("status")) != "unavailable"
        or str(signal.get("reason") or "") != COLD_START_REASON
        or signal.get("fit_through") is not None
    ):
        return None
    return dict(COLD_START_EDGE)


def _atom_row_from_signal(
    *,
    game_id: str,
    date: object,
    signal: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert one signal while preserving the no-history evidence label."""

    fit_through = signal.get("fit_through")
    edge = _edge_from_signal(signal)
    evidence_status = "available"
    if edge is None:
        edge = _cold_start_edge(signal)
        if edge is not None:
            evidence_status = COLD_START_STATUS
    status = evidence_status if edge is not None else str(signal.get("status") or "unavailable")
    return {
        "game_id": game_id,
        "date": _iso(date),
        "fit_through": _iso(pd.to_datetime(fit_through, utc=True))
        if fit_through is not None
        else None,
        "status": status,
        "reason": signal.get("reason"),
        "edge_components": edge,
        "blue_prior_role_games": signal.get("blue", {}).get("prior_role_games")
        if isinstance(signal.get("blue"), Mapping)
        else None,
        "red_prior_role_games": signal.get("red", {}).get("prior_role_games")
        if isinstance(signal.get("red"), Mapping)
        else None,
    }


def _iso(value: object) -> str:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def score_static_atoms(
    games: list[dict[str, Any]],
    source: Mapping[str, Any],
    maps: pd.DataFrame,
    *,
    cache_dir: Path,
    min_training_games: int = composition_signal.MIN_TRAINING_GAMES,
    min_support_games: int = 0,
    worker_commit: str | None = None,
    workers: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_ids = {str(game["game_uid"]) for game in games}

    def score_batch(batch_ids: set[str]) -> composition_signal.CompositionScoreResult:
        return composition_signal.score_games_temporally(
            games,
            target_game_ids=batch_ids,
            cache_dir=cache_dir,
            source_digest=str(source["source_identity_sha256"]),
            worker_commit=worker_commit,
            min_support_games=min_support_games,
            min_training_games=min_training_games,
            composition_only=True,
        )

    # A batch owns complete calendar-date clusters.  This keeps every cache
    # checkpoint single-writer while allowing independent date fits to use
    # separate local cores.  The merged result is sorted by game ID below.
    date_groups: dict[pd.Timestamp, set[str]] = defaultdict(set)
    date_by_id = dict(zip(maps["game_id"].astype(str), maps["date"]))
    for game_id in target_ids:
        date_groups[pd.Timestamp(date_by_id[game_id]).normalize()].add(game_id)
    batch_count = max(1, min(int(workers), len(date_groups)))
    batches: list[set[str]] = [set() for _ in range(batch_count)]
    for index, date in enumerate(sorted(date_groups)):
        batches[index % batch_count].update(date_groups[date])
    if batch_count == 1:
        batch_results = [score_batch(batches[0])]
    else:
        global _PARALLEL_SCORE_CONTEXT
        _PARALLEL_SCORE_CONTEXT = (
            games,
            source,
            cache_dir,
            worker_commit,
            min_support_games,
            min_training_games,
        )
        try:
            context = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=batch_count, mp_context=context) as executor:
                raw_results = list(executor.map(_score_composition_batch_worker, batches))
        finally:
            _PARALLEL_SCORE_CONTEXT = None
        batch_results = [
            composition_signal.CompositionScoreResult(signals=signals, audit=audit)
            for signals, audit in raw_results
        ]
    signals: dict[str, dict[str, Any]] = {}
    audits: list[Mapping[str, Any]] = []
    for batch_result in batch_results:
        signals.update(batch_result.signals)
        audits.append(batch_result.audit)
    fit_dates = [
        str(signal["fit_through"])
        for signal in signals.values()
        if signal.get("fit_through")
    ]
    status_counts = {status: 0 for status in composition_signal.PUBLIC_STATUS}
    for signal in signals.values():
        status = str(signal.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
    merged_audit: dict[str, Any] = dict(audits[0] if audits else {})
    merged_audit.update(
        {
            "target_games": len(signals),
            "available_games": status_counts.get("available", 0),
            "limited_games": status_counts.get("limited", 0),
            "unavailable_games": status_counts.get("unavailable", 0),
            "fit_through": max(fit_dates) if fit_dates else None,
            "cache_hits": sum(int(audit.get("cache_hits", 0)) for audit in audits),
            "parallel_workers": batch_count,
            "parallel_date_batches": len(date_groups),
        }
    )
    result = composition_signal.CompositionScoreResult(signals=signals, audit=merged_audit)
    dates = dict(zip(maps["game_id"].astype(str), maps["date"]))
    rows_by_id: dict[str, dict[str, Any]] = {}
    for game in games:
        game_id = str(game["game_uid"])
        target_date = pd.Timestamp(dates[game_id])
        signal = result.signals.get(game_id)
        if signal is None:
            raise StrictPriorAtomError(f"composition scorer omitted {game_id}")
        fit_through = signal.get("fit_through")
        fit_stamp = pd.to_datetime(fit_through, utc=True, errors="coerce")
        if fit_through is not None and (
            pd.isna(fit_stamp) or fit_stamp.normalize() >= target_date.normalize()
        ):
            raise StrictPriorAtomError(f"composition scorer leaked the target date: {game_id}")
        rows_by_id[game_id] = _atom_row_from_signal(
            game_id=game_id,
            date=target_date,
            signal=signal,
        )
    for game_id in sorted(set(str(value) for value in source["accepted_game_ids"]) - set(rows_by_id)):
        map_date = maps.loc[maps["game_id"].astype(str).eq(game_id), "date"]
        rows_by_id[game_id] = {
            "game_id": game_id,
            "date": _iso(map_date.iloc[0]),
            "fit_through": None,
            "status": "unavailable",
            "reason": "composition_input_incomplete",
            "edge_components": None,
            "blue_prior_role_games": None,
            "red_prior_role_games": None,
        }
    rows = [rows_by_id[game_id] for game_id in sorted(rows_by_id)]
    coverage = {
        "accepted_game_count": int(source["source_game_count"]),
        "composition_input_game_count": len(games),
        "available_game_count": sum(
            row["status"] in {"available", COLD_START_STATUS} for row in rows
        ),
        "cold_start_neutral_game_count": sum(
            row["status"] == COLD_START_STATUS for row in rows
        ),
        "unavailable_game_count": sum(
            row["status"] not in {"available", COLD_START_STATUS} for row in rows
        ),
        "fit_through_min": min(
            (row["fit_through"] for row in rows if row["fit_through"]),
            default=None,
        ),
        "fit_through_max": max(
            (row["fit_through"] for row in rows if row["fit_through"]),
            default=None,
        ),
        "min_training_games": min_training_games,
        "min_support_games": min_support_games,
    }
    return rows, {"score_audit": result.audit, "coverage": coverage}


def _metric_value(row: Mapping[str, Any], metric: str) -> float | None:
    value = row.get(metric)
    return _finite(value)


def build_player_form(players: pd.DataFrame, maps: pd.DataFrame) -> list[dict[str, Any]]:
    """Build an expanding player-form ledger with date-cluster isolation."""

    frame = players.copy()
    frame["game_id"] = frame["game_uid"].map(str)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame["role"] = frame["position"].astype(str).str.casefold()
    frame["player_key"] = frame.apply(
        lambda row: str(row.get("playerid") or "").strip()
        or "name:" + str(row.get("playername") or "").strip().casefold()
        + "|team:" + str(row.get("teamid") or row.get("teamname") or "").strip().casefold(),
        axis=1,
    )
    frame = frame.sort_values(["date", "game_id", "side", "position"], kind="stable")
    player_role_history: dict[tuple[str, str], dict[str, _RunningStats]] = defaultdict(
        lambda: defaultdict(_RunningStats)
    )
    player_history: dict[str, dict[str, _RunningStats]] = defaultdict(
        lambda: defaultdict(_RunningStats)
    )
    role_history: dict[str, dict[str, _RunningStats]] = defaultdict(
        lambda: defaultdict(_RunningStats)
    )
    rows: list[dict[str, Any]] = []

    def history_value(
        history: Mapping[str, _RunningStats], metric: str
    ) -> float | None:
        stats = history.get(metric)
        return stats.mean() if stats is not None else None

    def form_for_group(group: pd.DataFrame) -> dict[str, Any]:
        side_scores: dict[str, list[float]] = {"Blue": [], "Red": []}
        side_metric_values: dict[str, dict[str, list[float]]] = {
            side: {metric: [] for metric in FORM_METRICS} for side in ("Blue", "Red")
        }
        support = {"Blue": 0, "Red": 0}
        for _, raw in group.iterrows():
            key = str(raw["player_key"])
            role = str(raw["role"])
            role_hist = player_role_history.get((key, role), {})
            all_hist = player_history.get(key, {})
            values: dict[str, float] = {}
            z_values: list[float] = []
            for metric in FORM_METRICS:
                value = history_value(role_hist, metric)
                if value is None:
                    value = history_value(all_hist, metric)
                global_stats = role_history.get(role, {}).get(metric)
                if value is None or global_stats is None or not global_stats.count:
                    continue
                mean = global_stats.mean()
                std = global_stats.std()
                if mean is None or std is None:
                    continue
                scale = max(std, FORM_SCALES[metric] * 0.05)
                z = (value - mean) / scale
                z_values.append(FORM_WEIGHTS[metric] * z)
                values[metric] = value
                side_metric_values[str(raw["side"])][metric].append(value)
            if z_values:
                side = str(raw["side"])
                side_scores[side].append(float(np.mean(z_values)))
                support[side] += 1
        blue = float(np.mean(side_scores["Blue"])) if side_scores["Blue"] else None
        red = float(np.mean(side_scores["Red"])) if side_scores["Red"] else None
        metric_diffs = {
            f"player_form_{metric}_diff": (
                float(np.mean(side_metric_values["Blue"][metric]))
                - float(np.mean(side_metric_values["Red"][metric]))
                if side_metric_values["Blue"][metric] and side_metric_values["Red"][metric]
                else None
            )
            for metric in FORM_METRICS
        }
        return {
            "blue": blue,
            "red": red,
            "support_blue": support["Blue"],
            "support_red": support["Red"],
            "metric_diffs": metric_diffs,
        }

    for day, day_frame in frame.groupby(frame["date"].dt.normalize(), sort=True):
        for game_id, group in day_frame.groupby("game_id", sort=True):
            value = form_for_group(group)
            blue = value["blue"]
            red = value["red"]
            feature = blue - red if blue is not None and red is not None else None
            rows.append(
                {
                    "game_id": str(game_id),
                    "date": _iso(group["date"].max()),
                    "fit_through": _iso(frame.loc[frame["date"] < day, "date"].max())
                    if frame["date"].lt(day).any()
                    else None,
                    "status": "available" if feature is not None else "unavailable",
                    "future_player_form_logit": feature,
                    "support_blue": value["support_blue"],
                    "support_red": value["support_red"],
                    **value["metric_diffs"],
                }
            )
        # All target games in a date cluster are scored before final fields
        # from that date enter any later feature vector.
        for _, raw in day_frame.iterrows():
            key = str(raw["player_key"])
            role = str(raw["role"])
            for metric in FORM_METRICS:
                value = _metric_value(raw, metric)
                if value is None:
                    continue
                player_role_history[(key, role)][metric].add(value)
                player_history[key][metric].add(value)
                role_history[role][metric].add(value)
    return sorted(rows, key=lambda row: str(row["game_id"]))


def build_artifacts(
    *,
    source_receipt_path: Path,
    players_path: Path,
    maps_path: Path,
    cache_dir: Path,
    atom_output_path: Path,
    form_output_path: Path,
    min_training_games: int = composition_signal.MIN_TRAINING_GAMES,
    min_support_games: int = 0,
    worker_commit: str | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    source = load_source_receipt(source_receipt_path.resolve())
    players = load_players(players_path.resolve(), source)
    maps = load_maps(maps_path.resolve(), source)
    accepted_ids = {str(value) for value in source["accepted_game_ids"]}
    games = build_games(players, accepted_ids)
    game_dates = dict(zip(maps["game_id"].astype(str), maps["date"]))
    if any(_iso(game["date"]) != _iso(game_dates[str(game["game_uid"])]) for game in games):
        raise StrictPriorAtomError("composition game date differs from map ledger")
    rows, audit = score_static_atoms(
        games,
        source,
        maps,
        cache_dir=cache_dir.resolve(),
        min_training_games=min_training_games,
        min_support_games=min_support_games,
        worker_commit=worker_commit,
        workers=workers,
    )
    form_rows = build_player_form(players, maps)
    producer_code_hash = _sha_path(Path(__file__).resolve())
    composition_code_hash = _sha_path(Path(composition_signal.__file__).resolve())
    common = {
        "source_as_of": source["source_as_of"],
        "source_game_count": source["source_game_count"],
        "source_identity_sha256": source["source_identity_sha256"],
        "source_receipt_sha256": source["receipt_sha256"],
        "input_files": {
            "players": _source_file_binding(source, "players", players_path.resolve()),
            "maps": _source_file_binding(source, "maps", maps_path.resolve()),
        },
    }
    atom_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "research_only",
        "authority": {
            "research_only": True,
            "public": False,
            "probability": False,
            "promotion": False,
            "deployment": False,
        },
        "source": common,
        "producer": {
            "producer_name": "strict_prior_composition_signal",
            "producer_family": "composition_signal",
            "model_version": composition_signal.DESCRIPTIVE_MODEL_VERSION,
            "composition_signal_code_sha256": composition_code_hash,
            "producer_code_sha256": producer_code_hash,
            "training_order": "earlier accepted calendar-date clusters only",
            "control_scaffold": "neutral_mu_diff_and_sigma_pair_excluded_by_composition_only_fit",
            "component_mapping": {
                "base": "composition_signal.blue/red.components.base",
                "ally_synergy": "composition_signal.blue/red.components.ally_synergy",
                "enemy_counter": "composition_signal.blue/red.components.enemy_counter",
                "same_role": "composition_signal.blue/red.components.same_role",
                "archetype_interactions": "composition_signal.blue/red.components.atomized",
            },
            "cold_start_contract": dict(COLD_START_CONTRACT),
        },
        "coverage": audit["coverage"],
        "score_audit": audit["score_audit"],
        "rows": rows,
        "rows_sha256": _sha_bytes(_canonical(rows)),
    }
    atom_payload["artifact_sha256"] = _sha_bytes(_canonical(atom_payload))
    form_payload: dict[str, Any] = {
        "schema_version": FORM_SCHEMA_VERSION,
        "status": "research_only",
        "authority": {
            "research_only": True,
            "public": False,
            "probability": False,
            "promotion": False,
            "deployment": False,
        },
        "source": common,
        "producer": {
            "producer_name": "strict_prior_player_form",
            "producer_family": "oe_historical_form",
            "training_order": "earlier accepted calendar-date clusters only",
            "metrics": list(FORM_METRICS),
            "weights": FORM_WEIGHTS,
            "scales": FORM_SCALES,
            "feature_contract": "raw_prior_player_metric_composite; no target or current final metric",
            "producer_code_sha256": producer_code_hash,
        },
        "coverage": {
            "accepted_game_count": int(source["source_game_count"]),
            "row_count": len(form_rows),
            "available_row_count": sum(row["status"] == "available" for row in form_rows),
            "unavailable_row_count": sum(row["status"] != "available" for row in form_rows),
        },
        "rows": form_rows,
        "rows_sha256": _sha_bytes(_canonical(form_rows)),
    }
    form_payload["artifact_sha256"] = _sha_bytes(_canonical(form_payload))
    atom_output_path.parent.mkdir(parents=True, exist_ok=True)
    form_output_path.parent.mkdir(parents=True, exist_ok=True)
    atom_output_path.write_bytes(_canonical(atom_payload) + b"\n")
    form_output_path.write_bytes(_canonical(form_payload) + b"\n")
    return {
        "atom": atom_payload,
        "form": form_payload,
        "outputs": {
            "atoms": _hash_record(atom_output_path),
            "form": _hash_record(form_output_path),
        },
    }


def _fold_spec(path: Path, fold: int) -> tuple[tuple[str, ...], tuple[str, ...], pd.Timestamp]:
    payload = _load_json(path / f"fold-{fold}-spec.json", f"fold {fold} specification")
    train_ids = tuple(sorted(str(value) for value in payload.get("train_game_ids", [])))
    validation_ids = tuple(sorted(str(value) for value in payload.get("validation_game_ids", [])))
    cutoff = pd.to_datetime(payload.get("fit_window_end"), utc=True, errors="coerce")
    if not train_ids or not validation_ids or set(train_ids) & set(validation_ids) or pd.isna(cutoff):
        raise StrictPriorAtomError(f"fold {fold} specification is invalid")
    return train_ids, validation_ids, pd.Timestamp(cutoff)


def _unavailable_atom_row(game_id: str, date: object, reason: str) -> dict[str, Any]:
    return {
        "game_id": game_id,
        "date": _iso(date),
        "fit_through": None,
        "status": "unavailable",
        "reason": reason,
        "edge_components": None,
        "blue_prior_role_games": None,
        "red_prior_role_games": None,
    }


def _unavailable_form_row(game_id: str, date: object, reason: str) -> dict[str, Any]:
    return {
        "game_id": game_id,
        "date": _iso(date),
        "fit_through": None,
        "status": "unavailable",
        "reason": reason,
        "future_player_form_logit": None,
        "support_blue": 0,
        "support_red": 0,
        **{f"player_form_{metric}_diff": None for metric in FORM_METRICS},
    }


def build_fold_artifacts(
    *,
    source_receipt_path: Path,
    players_path: Path,
    maps_path: Path,
    folds_root: Path,
    output_root: Path,
    cache_dir: Path,
    min_training_games: int = composition_signal.MIN_TRAINING_GAMES,
    min_support_games: int = 0,
    worker_commit: str | None = None,
    workers: int = 1,
    selected_folds: Sequence[int] = (1, 2, 3),
) -> dict[str, Any]:
    """Build one frozen strict-prior atom and form ledger for each outer fold."""

    source = load_source_receipt(source_receipt_path.resolve())
    players = load_players(players_path.resolve(), source)
    maps = load_maps(maps_path.resolve(), source)
    accepted_ids = {str(value) for value in source["accepted_game_ids"]}
    games = build_games(players, accepted_ids)
    games_by_id = {str(game["game_uid"]): game for game in games}
    map_dates = dict(zip(maps["game_id"].astype(str), maps["date"]))
    specs = {fold: _fold_spec(folds_root.resolve(), fold) for fold in (1, 2, 3)}
    validation_by_fold = {fold: set(specs[fold][1]) for fold in (1, 2, 3)}
    producer_code_hash = _sha_path(Path(__file__).resolve())
    composition_code_hash = _sha_path(Path(composition_signal.__file__).resolve())
    common_source = {
        "source_as_of": source["source_as_of"],
        "source_game_count": source["source_game_count"],
        "source_identity_sha256": source["source_identity_sha256"],
        "source_receipt_sha256": source["receipt_sha256"],
        "input_files": {
            "players": _source_file_binding(source, "players", players_path.resolve()),
            "maps": _source_file_binding(source, "maps", maps_path.resolve()),
        },
    }
    outputs: dict[str, Any] = {}
    requested = {int(value) for value in selected_folds}
    if not requested or not requested.issubset({1, 2, 3}):
        raise StrictPriorAtomError("selected folds must be in 1, 2, 3")
    for fold in sorted(requested):
        prior_validation_ids = set().union(
            *(validation_by_fold[prior] for prior in range(1, fold))
        ) if fold > 1 else set()
        train_ids, validation_ids_tuple, cutoff = specs[fold]
        validation_ids = set(validation_ids_tuple)
        train_set = set(train_ids)
        excluded_prior_validation = train_set & prior_validation_ids
        effective_train_ids = train_set - prior_validation_ids
        expected_ids = train_set | validation_ids
        train_games = [games_by_id[game_id] for game_id in sorted(effective_train_ids & set(games_by_id))]
        fold_source = dict(source)
        fold_source["accepted_game_ids"] = sorted(effective_train_ids)
        fold_source["source_game_count"] = len(effective_train_ids)
        train_rows, train_audit = score_static_atoms(
            train_games,
            fold_source,
            maps[maps["game_id"].isin(effective_train_ids)],
            cache_dir=(cache_dir / f"fold-{fold}").resolve(),
            min_training_games=min_training_games,
            min_support_games=min_support_games,
            worker_commit=worker_commit,
            workers=workers,
        )
        atom_rows_by_id = {str(row["game_id"]): row for row in train_rows}

        validation_history_games = [
            game
            for game in train_games
            if pd.Timestamp(game["date"]).normalize() < cutoff.normalize()
        ]
        validation_names = composition_signal._descriptive_feature_names(validation_history_games)
        validation_model = composition_signal._fit_model(
            validation_history_games,
            names=validation_names,
            include_draft=True,
            min_training_games=min_training_games,
            worker_commit=worker_commit,
            composition_only=True,
        )
        for game_id in sorted(validation_ids):
            game = games_by_id.get(game_id)
            if game is None:
                atom_rows_by_id[game_id] = _unavailable_atom_row(
                    game_id, map_dates[game_id], "composition_input_incomplete"
                )
                continue
            signal = composition_signal.public_signal_for_game(
                game,
                validation_model,
                min_support_games=min_support_games,
            )
            atom_rows_by_id[game_id] = _atom_row_from_signal(
                game_id=game_id,
                date=map_dates[game_id],
                signal=signal,
            )
        for game_id in sorted(excluded_prior_validation):
            atom_rows_by_id[game_id] = _unavailable_atom_row(
                game_id, map_dates[game_id], "excluded_previous_outer_validation"
            )
        for game_id in sorted(expected_ids - set(atom_rows_by_id)):
            atom_rows_by_id[game_id] = _unavailable_atom_row(
                game_id, map_dates[game_id], "composition_input_incomplete"
            )
        atom_rows = [atom_rows_by_id[game_id] for game_id in sorted(expected_ids)]

        form_players = players[players["game_uid"].isin(effective_train_ids | validation_ids)].copy()
        form_players.loc[form_players["game_uid"].isin(validation_ids), "date"] = cutoff
        form_rows = build_player_form(
            form_players,
            maps[maps["game_id"].isin(effective_train_ids | validation_ids)],
        )
        form_rows_by_id = {str(row["game_id"]): row for row in form_rows}
        for game_id in validation_ids:
            if game_id in form_rows_by_id:
                form_rows_by_id[game_id]["date"] = _iso(map_dates[game_id])
        for game_id in sorted(excluded_prior_validation):
            form_rows_by_id[game_id] = _unavailable_form_row(
                game_id, map_dates[game_id], "excluded_previous_outer_validation"
            )
        for game_id in sorted(expected_ids - set(form_rows_by_id)):
            form_rows_by_id[game_id] = _unavailable_form_row(
                game_id, map_dates[game_id], "player_input_incomplete"
            )
        form_rows = [form_rows_by_id[game_id] for game_id in sorted(expected_ids)]

        fold_contract = {
            "fold": fold,
            "fit_window_end": _iso(cutoff),
            "train_game_count": len(train_set),
            "train_game_identity_sha256": identity_sha256(sorted(train_set)),
            "effective_train_game_count": len(effective_train_ids),
            "effective_train_game_identity_sha256": identity_sha256(sorted(effective_train_ids)),
            "validation_game_count": len(validation_ids),
            "validation_game_identity_sha256": identity_sha256(sorted(validation_ids)),
            "excluded_previous_validation_game_count": len(excluded_prior_validation),
            "excluded_previous_validation_identity_sha256": identity_sha256(sorted(excluded_prior_validation)),
            "validation_feature_state": "frozen_effective_training_before_cutoff_calendar_day",
        }
        producer = {
            "producer_code_sha256": producer_code_hash,
            "training_order": "effective outer-fold training rows only; validation state frozen at cutoff",
        }
        atom_payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "research_only",
            "authority": {"research_only": True, "public": False, "probability": False, "promotion": False, "deployment": False},
            "source": common_source,
            "fold_contract": fold_contract,
            "producer": {
                **producer,
                "producer_name": "strict_prior_composition_signal_outer_fold",
                "composition_signal_code_sha256": composition_code_hash,
                "component_mapping": {
                    "base": "composition_signal.blue/red.components.base",
                    "ally_synergy": "composition_signal.blue/red.components.ally_synergy",
                    "enemy_counter": "composition_signal.blue/red.components.enemy_counter",
                    "same_role": "composition_signal.blue/red.components.same_role",
                    "archetype_interactions": "composition_signal.blue/red.components.atomized",
                },
                "cold_start_contract": dict(COLD_START_CONTRACT),
            },
            "coverage": {
                "row_count": len(atom_rows),
                "available_game_count": sum(
                    row["status"] in {"available", COLD_START_STATUS}
                    for row in atom_rows
                ),
                "cold_start_neutral_game_count": sum(
                    row["status"] == COLD_START_STATUS for row in atom_rows
                ),
                "unavailable_game_count": sum(
                    row["status"] not in {"available", COLD_START_STATUS}
                    for row in atom_rows
                ),
                "fit_through_max": max((row["fit_through"] for row in atom_rows if row.get("fit_through")), default=None),
            },
            "score_audit": train_audit,
            "rows": atom_rows,
            "rows_sha256": _sha_bytes(_canonical(atom_rows)),
        }
        atom_payload["artifact_sha256"] = _sha_bytes(_canonical(atom_payload))
        form_payload: dict[str, Any] = {
            "schema_version": FORM_SCHEMA_VERSION,
            "status": "research_only",
            "authority": {"research_only": True, "public": False, "probability": False, "promotion": False, "deployment": False},
            "source": common_source,
            "fold_contract": fold_contract,
            "producer": {
                **producer,
                "producer_name": "strict_prior_player_form_outer_fold",
                "metrics": list(FORM_METRICS),
                "feature_contract": "raw prior player metrics from effective fold training only",
            },
            "coverage": {
                "row_count": len(form_rows),
                "available_row_count": sum(row["status"] == "available" for row in form_rows),
                "unavailable_row_count": sum(row["status"] != "available" for row in form_rows),
                "fit_through_max": max((row["fit_through"] for row in form_rows if row.get("fit_through")), default=None),
            },
            "rows": form_rows,
            "rows_sha256": _sha_bytes(_canonical(form_rows)),
        }
        form_payload["artifact_sha256"] = _sha_bytes(_canonical(form_payload))
        fold_root = output_root.resolve() / f"fold-{fold}"
        fold_root.mkdir(parents=True, exist_ok=True)
        atom_path = fold_root / "strict-prior-composition-atoms.json"
        form_path = fold_root / "strict-prior-player-form.json"
        atom_path.write_bytes(_canonical(atom_payload) + b"\n")
        form_path.write_bytes(_canonical(form_payload) + b"\n")
        outputs[str(fold)] = {
            "atoms": _hash_record(atom_path),
            "form": _hash_record(form_path),
            "fold_contract": fold_contract,
        }
    return {"outputs": outputs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--players", required=True, type=Path)
    parser.add_argument("--maps", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--atoms-out", type=Path)
    parser.add_argument("--form-out", type=Path)
    parser.add_argument("--folds-root", type=Path)
    parser.add_argument("--fold-output-root", type=Path)
    parser.add_argument("--only-fold", type=int, choices=(1, 2, 3), action="append")
    parser.add_argument("--min-training-games", type=int, default=composition_signal.MIN_TRAINING_GAMES)
    parser.add_argument("--min-support-games", type=int, default=0)
    parser.add_argument("--worker-commit")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.folds_root is not None or args.fold_output_root is not None:
        if args.folds_root is None or args.fold_output_root is None:
            parser.error("--folds-root and --fold-output-root are required together")
        result = build_fold_artifacts(
            source_receipt_path=args.source_receipt,
            players_path=args.players,
            maps_path=args.maps,
            folds_root=args.folds_root,
            output_root=args.fold_output_root,
            cache_dir=args.cache_dir,
            min_training_games=args.min_training_games,
            min_support_games=args.min_support_games,
            worker_commit=args.worker_commit,
            workers=args.workers,
            selected_folds=tuple(args.only_fold or (1, 2, 3)),
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.atoms_out is None or args.form_out is None:
        parser.error("--atoms-out and --form-out are required for a source-wide build")
    result = build_artifacts(
        source_receipt_path=args.source_receipt,
        players_path=args.players,
        maps_path=args.maps,
        cache_dir=args.cache_dir,
        atom_output_path=args.atoms_out,
        form_output_path=args.form_out,
        min_training_games=args.min_training_games,
        min_support_games=args.min_support_games,
        worker_commit=args.worker_commit,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "atom_output": result["outputs"]["atoms"],
                "form_output": result["outputs"]["form"],
                "atom_coverage": result["atom"]["coverage"],
                "form_coverage": result["form"]["coverage"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
