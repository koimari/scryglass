"""Regularized hierarchical Bradley--Terry ratings for current ladders.

This is the conservative public-rating reference model.  It is deliberately
separate from the sequential Dual Elo feature generator: the latter remains a
useful pre-match benchmark, while this module fits a global organization
effect plus a partially pooled home-league effect for the current ladder.

Important design choices:

* maps are collapsed to one observation per series so Bo3/Bo5 maps do not
  receive five times the weight of a Bo1;
* organization identity is independent of the event label, so LCS/MSI/EWC
  appearances share one team effect;
* league effects are strongly regularized and only become precise through
  cross-league bridges; a disconnected domestic ladder therefore gets a wide
  interval instead of an unjustified global rank;
* recency weights are explicit and the fit can be cut off at any date for
  rolling-origin validation.

The posterior uncertainty is a local Laplace approximation to the penalized
MAP fit.  It is suitable for conservative display/ranking and diagnostics;
it is not presented as a fully sampled Bayesian posterior.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from lol_kills.etl.competition import (
    INTERNATIONAL_LEAGUES,
    REGIONAL_LEAGUES,
    TAXONOMY_VERSION,
    canonicalize_competition_frame,
    team_identity_key,
)
from lol_kills.etl.paths import FEATURES_DIR
from lol_kills.etl.source_keys import canonical_source_game_key
from lol_kills.ratings.validation import audit_rating_inputs
from lol_kills.v2.tierlists.accepted_census import canonical_game_ids, identity_sha256


LOGIT_TO_ELO = 400.0 / math.log(10.0)
HIERARCHICAL_CACHE_SCHEMA = "scryglass:hierarchical-bt-cache:v1"
HIERARCHICAL_SNAPSHOT_SCHEMA = "scryglass:hierarchical-bt-snapshot:v1"
HIERARCHICAL_CACHE_MANIFEST = "ratings_hierarchical_cache.json"
HIERARCHICAL_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_CACHE_SLOTS = {
    "current": "ratings_hierarchical_snapshot.parquet",
    "previous": "ratings_hierarchical_previous_snapshot.parquet",
}
_HIERARCHICAL_SNAPSHOT_COLUMNS = (
    "team",
    "team_key",
    "mu_total",
    "mu_regional",
    "mu_meta",
    "sigma",
    "rating_p10",
    "n_series",
    "n_maps",
    "international_series",
    "home_league",
    "last_game_date",
    "model",
)
_CACHE_CONTENT_COLUMNS = (
    "date",
    "blue_team",
    "red_team",
    "blue_teamname",
    "red_teamname",
    "teamname",
    "team",
    "y_blue_win",
    "league",
    "league_source",
    "tournament",
    "is_international",
    "grid_series_id",
    "game",
    "game_uid",
    "gameid",
    "oe_gameid",
)
_OBSERVATION_INPUT_COLUMNS = frozenset(
    {
        "date",
        "y_blue_win",
        "league",
        "league_source",
        "tournament",
        "is_international",
        "game_uid",
        "gameid",
        "oe_gameid",
        "blue_team",
        "red_team",
        "blue_teamname",
        "red_teamname",
        "teamname",
        "team",
        "grid_series_id",
        "game",
    }
)


@dataclass(frozen=True)
class HierarchicalBTConfig:
    base_rating: float = 1500.0
    half_life_days: float = 365.0
    team_l2: float = 40.0
    league_l2: float = 100.0
    side_l2: float = 100.0
    min_sigma: float = 20.0
    unbridged_league_sigma: float = 45.0
    bridge_target_series: int = 8
    conservative_z: float = 1.6448536269514722  # one-sided 90th percentile
    max_iter: int = 500


def _cache_as_of(as_of: pd.Timestamp | None) -> str | None:
    if as_of is None:
        return None
    stamp = pd.Timestamp(as_of)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.isoformat()


def _cache_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _cache_date(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.isoformat()


def _frame_source_identity(
    maps: pd.DataFrame,
) -> tuple[str | None, int | None, str | None]:
    """Return stable identities for the exact map rows supplied to the fit."""

    if maps is None or maps.empty:
        return None, None, None
    source_column = next(
        (column for column in ("game_uid", "gameid", "oe_gameid") if column in maps.columns),
        None,
    )
    if source_column is None:
        return None, None, None
    game_ids = [canonical_source_game_key(value) for value in maps[source_column].tolist()]
    if not game_ids or any(not value for value in game_ids) or len(set(game_ids)) != len(game_ids):
        return None, None, None
    identity = hashlib.sha256(("\n".join(sorted(game_ids)) + "\n").encode("utf-8")).hexdigest()
    content_columns: list[list[str]] = []
    for column in _CACHE_CONTENT_COLUMNS:
        if column not in maps.columns:
            content_columns.append([""] * len(game_ids))
            continue
        values = maps[column].tolist()
        if column == "date":
            content_columns.append([_cache_date(value) for value in values])
        else:
            content_columns.append([_cache_text(value) for value in values])
    rows: list[list[str]] = []
    for index, game_id in enumerate(game_ids):
        rows.append([game_id, *(values[index] for values in content_columns)])
    rows.sort(key=lambda values: values[0])
    content = "\n".join("\x1f".join(values) for values in rows) + "\n"
    content_identity = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return identity, len(game_ids), content_identity


def _cache_key(
    maps: pd.DataFrame,
    *,
    as_of: pd.Timestamp | None,
    cfg: HierarchicalBTConfig,
    source_identity_sha256: str | None,
    cache_slot: str,
) -> dict[str, Any] | None:
    frame_identity, frame_game_count, frame_content_identity = _frame_source_identity(maps)
    if frame_identity is None or frame_game_count is None or frame_content_identity is None:
        return None
    source_identity = str(source_identity_sha256 or frame_identity).strip()
    if not source_identity:
        return None
    return {
        "schema": HIERARCHICAL_CACHE_SCHEMA,
        "implementation_sha256": HIERARCHICAL_IMPLEMENTATION_SHA256,
        "slot": cache_slot,
        "source_identity_sha256": source_identity,
        "frame_identity_sha256": frame_identity,
        "frame_content_sha256": frame_content_identity,
        "source_game_count": frame_game_count,
        "as_of": _cache_as_of(as_of),
        "config": dict(cfg.__dict__),
    }


def _cache_paths(cache_dir: Path, cache_slot: str) -> tuple[Path, Path]:
    try:
        snapshot_name = _CACHE_SLOTS[cache_slot]
    except KeyError as error:
        raise ValueError(f"unknown hierarchical cache slot: {cache_slot}") from error
    return cache_dir / snapshot_name, cache_dir / HIERARCHICAL_CACHE_MANIFEST


def _read_cache_manifest(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_cached_fit(
    cache_dir: Path,
    *,
    cache_slot: str,
    key: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    snapshot_path, manifest_path = _cache_paths(cache_dir, cache_slot)
    manifest = _read_cache_manifest(manifest_path)
    entry = manifest.get(cache_slot) if manifest is not None else None
    if not isinstance(entry, dict) or entry.get("key") != key:
        return None
    metadata = entry.get("metadata")
    snapshot_info = entry.get("snapshot")
    if not isinstance(metadata, dict) or not isinstance(snapshot_info, dict):
        return None
    if not snapshot_path.is_file():
        return None
    if snapshot_info.get("schema") != HIERARCHICAL_SNAPSHOT_SCHEMA:
        return None
    if snapshot_info.get("columns") != list(_HIERARCHICAL_SNAPSHOT_COLUMNS):
        return None
    expected_bytes = snapshot_info.get("byte_count")
    expected_sha256 = snapshot_info.get("sha256")
    if not isinstance(expected_bytes, int) or not isinstance(expected_sha256, str):
        return None
    try:
        snapshot_bytes = snapshot_path.read_bytes()
        if len(snapshot_bytes) != expected_bytes:
            return None
        if hashlib.sha256(snapshot_bytes).hexdigest() != expected_sha256:
            return None
        snapshot = pd.read_parquet(snapshot_path)
    except (OSError, ValueError, ImportError):
        return None
    if list(snapshot.columns) != list(_HIERARCHICAL_SNAPSHOT_COLUMNS):
        return None
    return snapshot, metadata


def _write_cached_fit(
    cache_dir: Path,
    *,
    cache_slot: str,
    key: dict[str, Any],
    snapshot: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    snapshot_path, manifest_path = _cache_paths(cache_dir, cache_slot)
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot_tmp = snapshot_path.with_name(f".{snapshot_path.name}.{os.getpid()}.tmp")
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    try:
        snapshot.to_parquet(snapshot_tmp, index=False)
        snapshot_bytes = snapshot_tmp.read_bytes()
        if list(snapshot.columns) != list(_HIERARCHICAL_SNAPSHOT_COLUMNS):
            raise ValueError("hierarchical snapshot columns do not match cache schema")
        os.replace(snapshot_tmp, snapshot_path)
        manifest = _read_cache_manifest(manifest_path) or {}
        manifest[cache_slot] = {
            "key": key,
            "metadata": metadata,
            "snapshot": {
                "schema": HIERARCHICAL_SNAPSHOT_SCHEMA,
                "columns": list(_HIERARCHICAL_SNAPSHOT_COLUMNS),
                "byte_count": len(snapshot_bytes),
                "sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
            },
        }
        manifest_tmp.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(manifest_tmp, manifest_path)
    finally:
        snapshot_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() in {"", "nan", "None"}


def _authoritative_series_id(row: pd.Series) -> str | None:
    """Return the source-provided series id when present.

    Only the GRID adapter writes ``grid_series_id`` today; it is the
    authoritative series identity and carries source evidence with it (the
    ``source``/``source_grid`` flags on the row and the adapter revision in
    ``lol_kills/etl/grid_ingest.py``).  A missing or empty id means the source
    has no safe series identity for this map.
    """

    explicit = row.get("grid_series_id")
    if not _is_missing(explicit):
        return str(explicit).strip()
    return None


def _game_key(row: pd.Series) -> str:
    """Return a stable game-level identity that never merges unrelated maps.

    ``game_uid`` is the canonical per-map identity in every warehouse frame
    (Oracle's Elixir ``gameid`` or Leaguepedia ``GameId``) and is unique per
    match.  When a frame lacks it, fall back to a date/teams/game-number key.

    The four-hour date bucket and sorted-team pairing are intentionally NOT
    used as a grouping key here: they merge unrelated matches and change
    outcome, side, recency, series count, uncertainty, and every downstream
    rating.
    """

    uid = row.get("game_uid")
    if not _is_missing(uid):
        return f"game:{str(uid).strip()}"
    date = (
        pd.Timestamp(row["date"]).strftime("%Y-%m-%dT%H:%M:%SZ")
        if pd.notna(row.get("date"))
        else "unknown-date"
    )
    a, b = sorted((team_identity_key(row.get("blue_team")), team_identity_key(row.get("red_team"))))
    game = row.get("game")
    game_bit = f"|game-{game}" if not _is_missing(game) else ""
    return f"derived-map:{date}|{a}|{b}{game_bit}"


def _series_identity(frame_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign every map an explicit series identity; audit unsafe/tied maps.

    Rules (issue #44):

    * An authoritative source series id (``grid_series_id``) groups maps into
      one series observation only when the group is internally consistent:
      the same unordered team pair appears in every map of the group.  A
      reused id that points at different team pairs is unsafe and its maps
      fall back to stable game-level keys.
    * Without a safe series id, each map keeps its own stable game-level key.
      There is no derived time/team bucket.
    * Series maps whose results do not produce a strict majority (a tied or
      incomplete feed) are unresolved: they are preserved in the returned
      audit trail and excluded from primary series inference because the
      series outcome is not identified.

    Returns ``(frame_rows, audit)`` where ``frame_rows`` gains ``series_key``
    and ``series_source`` columns (``grid`` or ``none``).
    """

    out = frame_rows.copy()
    out["series_key"] = out.apply(_game_key, axis=1)
    out["series_source"] = "none"
    out["series_id_present"] = out["grid_series_id"].fillna("").astype(str).str.strip().ne("")

    unsafe: set[str] = set()
    authoritative = out[out["series_id_present"]].copy()
    if not authoritative.empty:
        authoritative["_pair"] = authoritative.apply(
            lambda row: "|".join(sorted((str(row["blue"]), str(row["red"])))), axis=1
        )
        pair_counts = authoritative.groupby("grid_series_id")["_pair"].nunique()
        unsafe = set(str(value) for value in pair_counts[pair_counts > 1].index)
        safe = authoritative[~authoritative["grid_series_id"].astype(str).isin(unsafe)]
        out.loc[safe.index, "series_key"] = "grid:" + safe["grid_series_id"].astype(str)
        out.loc[safe.index, "series_source"] = "grid"

    audit: dict[str, Any] = {
        "unsafe_series_ids": sorted(unsafe),
        "n_unsafe_maps": int(out["series_id_present"].sum() - (out["series_source"] == "grid").sum()),
    }
    return out, audit


def _observations(
    maps: pd.DataFrame,
    as_of: pd.Timestamp | None,
    half_life_days: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build series-collapsed observations with explicit series identity.

    Returns ``(observations, audit)``.  Each row of ``observations`` is either
    one authoritative source series (``series_source == "grid"``) or one map
    with a stable game-level key (``series_source == "none"``).  The audit
    dict records maps excluded from primary inference (unsafe series ids and
    tied/incomplete feeds) so they stay inspectable.
    """

    if maps is None:
        frame = canonicalize_competition_frame(maps)
    else:
        input_columns = [column for column in maps.columns if column in _OBSERVATION_INPUT_COLUMNS]
        frame = canonicalize_competition_frame(maps.loc[:, input_columns])
    if frame is None or frame.empty:
        return pd.DataFrame(), {
            "n_unresolved_maps": 0,
            "n_unresolved_series": 0,
            "unresolved_series_ids": [],
            "unresolved_map_uids": [],
            "unsafe_series_ids": [],
            "n_unsafe_maps": 0,
        }
    frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce", utc=True).dt.tz_localize(None)
    frame["y_blue_win"] = pd.to_numeric(frame.get("y_blue_win"), errors="coerce")
    frame = frame.dropna(subset=["date", "y_blue_win"]).copy()
    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_convert("UTC").tz_localize(None)
        frame = frame[frame["date"] <= cutoff].copy()
    frame = frame.sort_values("date")
    if frame.empty:
        return pd.DataFrame(), {
            "n_unresolved_maps": 0,
            "n_unresolved_series": 0,
            "unresolved_series_ids": [],
            "unresolved_map_uids": [],
            "unsafe_series_ids": [],
            "n_unsafe_maps": 0,
        }

    # Home league is the latest observed regional affiliation before the
    # match.  A first domestic row establishes the affiliation only after its
    # pre-match state is recorded; international rows never overwrite it.
    home_league: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        blue_name = str(row.get("blue_team") or "")
        red_name = str(row.get("red_team") or "")
        blue = team_identity_key(blue_name)
        red = team_identity_key(red_name)
        source_league = str(row.get("league") or "UNKNOWN")
        blue_home = home_league.get(blue, source_league if source_league in REGIONAL_LEAGUES else "UNKNOWN")
        red_home = home_league.get(red, source_league if source_league in REGIONAL_LEAGUES else "UNKNOWN")
        game_uid = row.get("game_uid")
        grid_series_id = _authoritative_series_id(row) or ""
        game = row.get("game")
        records.append(
            {
                "date": row["date"],
                "blue": blue,
                "red": red,
                "blue_name": blue_name,
                "red_name": red_name,
                "blue_home": blue_home,
                "red_home": red_home,
                "y_blue": float(row["y_blue_win"]),
                "league": source_league,
                "is_international": bool(row.get("is_international", source_league in INTERNATIONAL_LEAGUES)),
                "game_uid": "" if _is_missing(game_uid) else str(game_uid).strip(),
                "grid_series_id": grid_series_id,
                "game": "" if _is_missing(game) else str(game).strip(),
            }
        )
        if source_league in REGIONAL_LEAGUES:
            home_league[blue] = source_league
            home_league[red] = source_league

    frame_rows = pd.DataFrame(records)
    frame_rows, identity_audit = _series_identity(frame_rows)

    if frame_rows["series_key"].is_unique:
        blue = frame_rows["blue"].astype(str)
        red = frame_rows["red"].astype(str)
        a_is_blue = blue.le(red)
        y_blue = frame_rows["y_blue"].astype(float)
        collapsed_frame = pd.DataFrame(
            {
                "series_key": frame_rows["series_key"].astype(str),
                "series_source": frame_rows["series_source"].astype(str),
                "game_uid": frame_rows["game_uid"].astype(str),
                "date": frame_rows["date"],
                "team_a": blue.where(a_is_blue, red),
                "team_b": red.where(a_is_blue, blue),
                "team_a_name": frame_rows["blue_name"].where(
                    a_is_blue, frame_rows["red_name"]
                ),
                "team_b_name": frame_rows["red_name"].where(
                    a_is_blue, frame_rows["blue_name"]
                ),
                "home_a": frame_rows["blue_home"].where(
                    a_is_blue, frame_rows["red_home"]
                ),
                "home_b": frame_rows["red_home"].where(
                    a_is_blue, frame_rows["blue_home"]
                ),
                "y_a": y_blue.where(a_is_blue, 1.0 - y_blue),
                "n_maps": np.ones(len(frame_rows), dtype=int),
                "a_blue_share": a_is_blue.astype(float),
                "international": frame_rows["is_international"].astype(bool),
            }
        )
        collapsed = collapsed_frame.to_dict("records")
        unresolved: list[dict[str, Any]] = []
    else:
        collapsed = []
        unresolved = []
        for key, group in frame_rows.groupby("series_key", sort=False):
            pairs = set(
                group.apply(
                    lambda row: "|".join(sorted((str(row["blue"]), str(row["red"])))), axis=1
                )
            )
            if len(pairs) != 1:
                # Exact-duplicate fallback keys with different team pairs cannot
                # be merged into one observation; keep them for audit only.
                unresolved.extend(group.to_dict("records"))
                continue
            a, b = sorted((str(group["blue"].iloc[0]), str(group["red"].iloc[0])))
            a_rows = group[group["blue"].eq(a)]
            a_wins = float(a_rows["y_blue"].sum()) + float((group[group["red"].eq(a)]["y_blue"] == 0).sum())
            n_maps = len(group)
            if a_wins * 2 == n_maps:
                # Tied/incomplete feed: the series outcome is not identified.
                # Preserve the maps for audit; exclude from primary inference.
                unresolved.extend(group.to_dict("records"))
                continue
            # A strict majority over ALL maps defines the series winner; the
            # first map is never selected as an outcome shortcut.
            y_a = 1.0 if a_wins > n_maps / 2 else 0.0
            a_blue_share = float(a_rows["blue"].eq(a).sum()) / n_maps
            first = group.iloc[0]
            source_a = a_rows.iloc[0] if not a_rows.empty else first
            b_rows = group[group["blue"].eq(b)]
            source_b = b_rows.iloc[0] if not b_rows.empty else first
            collapsed.append(
                {
                    "series_key": key,
                    "series_source": str(group["series_source"].iloc[0]),
                    "game_uid": ",".join(str(value) for value in group["game_uid"] if str(value)),
                    "date": first["date"],
                    "team_a": a,
                    "team_b": b,
                    "team_a_name": first["blue_name"] if first["blue"] == a else first["red_name"],
                    "team_b_name": first["red_name"] if first["blue"] == a else first["blue_name"],
                    "home_a": source_a["blue_home"] if source_a["blue"] == a else source_a["red_home"],
                    "home_b": source_b["blue_home"] if source_b["blue"] == b else source_b["red_home"],
                    "y_a": y_a,
                    "n_maps": n_maps,
                    "a_blue_share": a_blue_share,
                    "international": bool(group["is_international"].any()),
                }
            )

    out = pd.DataFrame(collapsed).sort_values("date").reset_index(drop=True)
    if not out.empty:
        cutoff = out["date"].max()
        out["weight"] = np.exp(
            -((cutoff - out["date"]).dt.total_seconds() / 86400.0) / max(half_life_days, 1.0)
        )
    unresolved_frame = pd.DataFrame(unresolved)
    unresolved_ids: list[str] = []
    unresolved_uids: list[str] = []
    if not unresolved_frame.empty:
        unresolved_ids = sorted(
            set(
                str(value)
                for value in unresolved_frame["grid_series_id"].fillna("")
                if str(value)
            )
        )
        unresolved_uids = sorted(
            set(str(value) for value in unresolved_frame["game_uid"] if str(value))
        )
    audit: dict[str, Any] = {
        "n_unresolved_maps": len(unresolved),
        "n_unresolved_series": int(unresolved_frame["series_key"].nunique()) if not unresolved_frame.empty else 0,
        "unresolved_series_ids": unresolved_ids,
        "unresolved_map_uids": unresolved_uids,
        "unsafe_series_ids": identity_audit["unsafe_series_ids"],
        "n_unsafe_maps": identity_audit["n_unsafe_maps"],
    }
    return out, audit


def _design(observations: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    teams = sorted(set(observations["team_a"]) | set(observations["team_b"]))
    leagues = sorted(
        (set(observations["home_a"]) | set(observations["home_b"])) - {"UNKNOWN"}
    )
    team_idx = {value: i for i, value in enumerate(teams)}
    league_idx = {value: i for i, value in enumerate(leagues)}
    X = np.zeros((len(observations), len(teams) + len(leagues) + 1), dtype=float)
    for i, row in observations.iterrows():
        X[i, team_idx[row["team_a"]]] += 1.0
        X[i, team_idx[row["team_b"]]] -= 1.0
        if row["home_a"] in league_idx:
            X[i, len(teams) + league_idx[row["home_a"]]] += 1.0
        if row["home_b"] in league_idx:
            X[i, len(teams) + league_idx[row["home_b"]]] -= 1.0
        # Side exposure: the observation's team A blue share minus team B's.
        # For a Bo1 this is +/-1 exactly; for a multi-map series it keeps
        # every map's side information instead of the first map only.
        X[i, -1] = 2.0 * float(row["a_blue_share"]) - 1.0
    return X, teams, leagues


RESEARCH_PREDICTION_SCHEMA = "scryglass:hierarchical-bt-map-prediction:v1"


def _research_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("research prediction value is not canonical JSON") from error


def _research_sha256(value: object) -> str:
    return hashlib.sha256(_research_json_bytes(value)).hexdigest()


def _research_id_identity(game_ids: tuple[str, ...] | list[str]) -> str:
    return identity_sha256(game_ids)


def _research_verified_source_receipt(
    source_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Verify the exact source receipt used by a research prediction."""

    if not isinstance(source_receipt, Mapping):
        raise ValueError("verified source receipt is required")
    receipt = dict(source_receipt)
    receipt_hash = str(receipt.get("receipt_sha256") or "").lower()
    if len(receipt_hash) != 64 or any(
        character not in "0123456789abcdef" for character in receipt_hash
    ):
        raise ValueError("source receipt hash is invalid")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    if _research_sha256(unsigned) != receipt_hash:
        raise ValueError("source receipt hash does not match its payload")

    accepted_raw = receipt.get("accepted_game_ids")
    eligible_raw = receipt.get("model_eligible_game_ids")
    if not isinstance(accepted_raw, list) or not isinstance(eligible_raw, list):
        raise ValueError("source receipt game ID lists are required")
    if not all(isinstance(value, str) for value in (*accepted_raw, *eligible_raw)):
        raise ValueError("source receipt game IDs must be strings")
    accepted_ids = tuple(str(value) for value in accepted_raw)
    eligible_ids = tuple(str(value) for value in eligible_raw)
    if not accepted_ids:
        raise ValueError("source receipt accepted game IDs are empty")
    if tuple(canonical_game_ids(accepted_ids)) != accepted_ids:
        raise ValueError("source receipt accepted game IDs are not canonical and unique")
    if tuple(canonical_game_ids(eligible_ids)) != eligible_ids:
        raise ValueError("source receipt eligible game IDs are not canonical and unique")
    if not set(eligible_ids).issubset(set(accepted_ids)):
        raise ValueError("source receipt eligible IDs are outside the accepted census")
    source_identity = str(receipt.get("source_identity_sha256") or "").lower()
    eligible_identity = str(
        receipt.get("model_eligible_identity_sha256") or ""
    ).lower()
    if source_identity != _research_id_identity(accepted_ids):
        raise ValueError("source receipt accepted identity is invalid")
    if eligible_identity != _research_id_identity(eligible_ids):
        raise ValueError("source receipt eligible identity is invalid")
    if receipt.get("source_game_count") != len(accepted_ids):
        raise ValueError("source receipt accepted count is invalid")
    if receipt.get("model_eligible_game_count") != len(eligible_ids):
        raise ValueError("source receipt eligible count is invalid")
    if "source_as_of" not in receipt:
        raise ValueError("source receipt source_as_of is required")
    source_as_of = _research_cutoff(receipt["source_as_of"]).isoformat()
    return {
        "receipt_sha256": receipt_hash,
        "source_as_of": source_as_of,
        "source_identity_sha256": source_identity,
        "source_game_count": len(accepted_ids),
        "accepted_game_ids": accepted_ids,
        "model_eligible_identity_sha256": eligible_identity,
        "model_eligible_game_count": len(eligible_ids),
        "model_eligible_game_ids": eligible_ids,
    }


def _research_series_ids(frame: pd.DataFrame, label: str) -> tuple[str, ...]:
    """Require authoritative series IDs for leakage-safe fold boundaries."""

    if "grid_series_id" not in frame.columns:
        raise ValueError(f"{label} frame has no safe grid_series_id")
    values = tuple(_cache_text(value) for value in frame["grid_series_id"].tolist())
    if any(not value for value in values):
        raise ValueError(f"{label} frame has missing safe grid_series_id")
    return values


def _research_series_pairs(frame: pd.DataFrame, label: str) -> dict[str, str]:
    """Bind each source series ID to one unordered team pair."""

    _research_series_ids(frame, label)
    blue_column = next(
        (column for column in ("blue_team", "blue_teamname") if column in frame.columns),
        None,
    )
    red_column = next(
        (column for column in ("red_team", "red_teamname") if column in frame.columns),
        None,
    )
    if blue_column is None or red_column is None:
        raise ValueError(f"{label} frame has no safe team pair columns")
    pairs: dict[str, str] = {}
    for series_id, blue_value, red_value in zip(
        frame["grid_series_id"], frame[blue_column], frame[red_column]
    ):
        blue = team_identity_key(blue_value)
        red = team_identity_key(red_value)
        if blue == "unknown-team" or red == "unknown-team":
            raise ValueError(f"{label} frame has an unsafe team pair")
        pair = "|".join(sorted((blue, red)))
        key = _cache_text(series_id)
        previous = pairs.get(key)
        if previous is not None and previous != pair:
            raise ValueError(f"{label} grid_series_id maps to multiple team pairs: {key}")
        pairs[key] = pair
    return pairs


def _research_order_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Sort source rows by stable metadata before fitting the fold."""

    ordered = frame.copy()
    game_ids, _ = _research_game_ids(ordered, label)
    dates = _research_dates(ordered, label)
    series_ids = _research_series_ids(ordered, label)
    ordered["_research_order_date"] = dates.to_numpy()
    ordered["_research_order_game"] = game_ids
    ordered["_research_order_series"] = series_ids
    ordered = ordered.sort_values(
        ["_research_order_date", "_research_order_series", "_research_order_game"],
        kind="mergesort",
    )
    return ordered.drop(
        columns=[
            "_research_order_date",
            "_research_order_game",
            "_research_order_series",
        ]
    ).reset_index(drop=True)


def _research_game_ids(frame: pd.DataFrame, label: str) -> tuple[list[str], tuple[str, ...]]:
    if frame is None or frame.empty:
        raise ValueError(f"{label} frame is empty")
    source_column = next(
        (column for column in ("game_uid", "gameid", "oe_gameid") if column in frame.columns),
        None,
    )
    if source_column is None:
        raise ValueError(f"{label} frame has no game identity column")
    ordered = [canonical_source_game_key(value) for value in frame[source_column].tolist()]
    if any(not value for value in ordered):
        raise ValueError(f"{label} frame has an empty game identity")
    if len(set(ordered)) != len(ordered):
        raise ValueError(f"{label} frame has duplicate game identities")
    return ordered, tuple(sorted(ordered))


def _research_identity(
    value: str | None,
    fallback: str,
    label: str,
) -> str:
    identity = fallback if value is None else str(value).strip().lower()
    if len(identity) != 64 or any(character not in "0123456789abcdef" for character in identity):
        raise ValueError(f"{label} source identity must be a SHA-256 hex digest")
    return identity


def _research_dates(frame: pd.DataFrame, label: str) -> pd.Series:
    date_column = next(
        (column for column in ("date", "played_at", "game_date", "start_time") if column in frame.columns),
        None,
    )
    if date_column is None:
        raise ValueError(f"{label} frame has no date column")
    dates = pd.to_datetime(frame[date_column], errors="coerce", utc=True)
    if dates.isna().any():
        raise ValueError(f"{label} frame has missing or invalid dates")
    return dates


def _research_cutoff(value: pd.Timestamp) -> pd.Timestamp:
    cutoff = pd.Timestamp(value)
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise ValueError("research prediction cutoff must be timezone-aware")
    return cutoff.tz_convert("UTC")


def _research_train_outcomes(train_maps: pd.DataFrame) -> None:
    if "y_blue_win" not in train_maps.columns:
        raise ValueError("training frame must contain y_blue_win")
    outcomes = pd.to_numeric(train_maps["y_blue_win"], errors="coerce")
    if outcomes.isna().any() or not outcomes.isin([0.0, 1.0]).all():
        raise ValueError("training frame contains an invalid y_blue_win outcome")


def _research_fit_state(
    observations: pd.DataFrame,
    cfg: HierarchicalBTConfig,
) -> dict[str, Any]:
    """Fit parameters for the in-memory validation scorer."""

    X, teams, leagues = _design(observations)
    y = observations["y_a"].to_numpy(float)
    weight = observations["weight"].to_numpy(float)
    n_team = len(teams)
    n_league = len(leagues)
    penalty = np.array(
        [cfg.team_l2] * n_team + [cfg.league_l2] * n_league + [cfg.side_l2],
        dtype=float,
    )

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        safe_beta = np.clip(
            np.nan_to_num(beta, nan=0.0, posinf=8.0, neginf=-8.0),
            -8.0,
            8.0,
        )
        eta = np.einsum("ij,j->i", X, safe_beta, optimize=True)
        probability = expit(eta)
        value = float(
            np.sum(weight * (np.logaddexp(0.0, eta) - y * eta))
            + 0.5 * np.sum(penalty * safe_beta * safe_beta)
        )
        gradient = (
            np.einsum("i,ij->j", weight * (probability - y), X, optimize=True)
            + penalty * safe_beta
        )
        return value, gradient

    result = minimize(
        objective,
        np.zeros(X.shape[1], dtype=float),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-8.0, 8.0)] * X.shape[1],
        options={"maxiter": cfg.max_iter, "ftol": 1e-10, "gtol": 1e-8},
    )
    beta = np.clip(
        np.nan_to_num(result.x, nan=0.0, posinf=8.0, neginf=-8.0),
        -8.0,
        8.0,
    )
    return {
        "beta": beta,
        "teams": teams,
        "leagues": leagues,
        "team_index": {team: index for index, team in enumerate(teams)},
        "league_index": {league: index for index, league in enumerate(leagues)},
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "n_iterations": int(getattr(result, "nit", 0) or 0),
        "n_observations": int(len(observations)),
        "n_maps": int(observations["n_maps"].sum()),
    }


def _research_home_state(observations: pd.DataFrame) -> dict[str, str]:
    home: dict[str, str] = {}
    sort_columns = [
        column
        for column in ("date", "series_key", "game_uid")
        if column in observations.columns
    ]
    ordered = observations.sort_values(sort_columns, kind="mergesort")
    for row in ordered.itertuples(index=False):
        for team, league in ((row.team_a, row.home_a), (row.team_b, row.home_b)):
            if str(league) != "UNKNOWN":
                home[str(team)] = str(league)
    return home


def _research_validation_rows(
    validation_maps: pd.DataFrame,
    *,
    fit_state: dict[str, Any],
    training_observations: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[str], list[str], list[str]]:
    """Score validation metadata without reading any validation outcome field."""

    input_columns = [column for column in validation_maps.columns if column in _OBSERVATION_INPUT_COLUMNS]
    value = canonicalize_competition_frame(validation_maps.loc[:, input_columns])
    ordered_ids, _ = _research_game_ids(validation_maps, "validation")
    dates = _research_dates(validation_maps, "validation")
    home = _research_home_state(training_observations)
    team_index = fit_state["team_index"]
    league_index = fit_state["league_index"]
    beta = fit_state["beta"]
    n_team = len(fit_state["teams"])
    rows: list[dict[str, Any]] = []
    missing_ids: list[str] = []
    unseen_teams: set[str] = set()
    unseen_game_ids: set[str] = set()
    indexed = value.reset_index(drop=True)
    for position, row in indexed.iterrows():
        game_id = ordered_ids[position]
        blue_name = row.get("blue_team")
        red_name = row.get("red_team")
        blue_text = "" if blue_name is None or pd.isna(blue_name) else str(blue_name).strip()
        red_text = "" if red_name is None or pd.isna(red_name) else str(red_name).strip()
        blue = team_identity_key(blue_text)
        red = team_identity_key(red_text)
        if not blue_text or not red_text or blue == "unknown-team" or red == "unknown-team":
            missing_ids.append(game_id)
            continue
        league = _cache_text(row.get("league")) or "UNKNOWN"
        blue_home = home.get(blue, league if league in REGIONAL_LEAGUES else "UNKNOWN")
        red_home = home.get(red, league if league in REGIONAL_LEAGUES else "UNKNOWN")
        a_is_blue = blue <= red
        team_a = blue if a_is_blue else red
        team_b = red if a_is_blue else blue
        if team_a not in team_index:
            unseen_teams.add(team_a)
        if team_b not in team_index:
            unseen_teams.add(team_b)
        if team_a not in team_index or team_b not in team_index:
            unseen_game_ids.add(game_id)
            missing_ids.append(game_id)
            continue
        vector = np.zeros(len(beta), dtype=float)
        if team_a in team_index:
            vector[team_index[team_a]] += 1.0
        if team_b in team_index:
            vector[team_index[team_b]] -= 1.0
        if blue_home in league_index:
            vector[n_team + league_index[blue_home]] += 1.0 if a_is_blue else -1.0
        if red_home in league_index:
            vector[n_team + league_index[red_home]] += -1.0 if a_is_blue else 1.0
        side = 1.0 if a_is_blue else -1.0
        vector[-1] = side
        orientation = 1.0 if a_is_blue else -1.0
        team_logit = float(np.dot(vector[:n_team], beta[:n_team]))
        league_logit = float(np.dot(vector[n_team:-1], beta[n_team:-1]))
        side_logit = float(vector[-1] * beta[-1])
        logit = orientation * (team_logit + league_logit + side_logit)
        rows.append(
            {
                "game_id": game_id,
                "date": pd.Timestamp(dates.iloc[position]).isoformat(),
                "blue_team": blue_text,
                "red_team": red_text,
                "blue_team_key": blue,
                "red_team_key": red,
                "league": league,
                "blue_home_league": blue_home,
                "red_home_league": red_home,
                "side_term": side,
                "team_a_known": bool(team_a in team_index),
                "team_b_known": bool(team_b in team_index),
                "league_a_known": bool((blue_home if a_is_blue else red_home) in league_index),
                "league_b_known": bool((red_home if a_is_blue else blue_home) in league_index),
                "team_logit": orientation * team_logit,
                "league_logit": orientation * league_logit,
                "side_logit": orientation * side_logit,
                "predicted_logit": logit,
                "predicted_blue_win": float(expit(logit)),
            }
        )
    return rows, sorted(set(missing_ids)), sorted(unseen_teams), sorted(unseen_game_ids)


def fit_hierarchical_bt_research_prediction(
    train_maps: pd.DataFrame,
    validation_maps: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    cfg: HierarchicalBTConfig | None = None,
    source_receipt: Mapping[str, Any] | None = None,
    source_identity_sha256: str | None = None,
    train_source_identity_sha256: str | None = None,
    validation_source_identity_sha256: str | None = None,
) -> dict[str, Any]:
    """Fit on train maps and score later maps without reading validation outcomes.

    This helper is for paired research evaluation.  The cutoff is strict:
    training rows must be at or before it, and validation rows must be after
    it.  The returned rows contain metadata and predictions only.  No files,
    caches, or public rating artifacts are written.
    """

    cfg = cfg or HierarchicalBTConfig()
    cutoff_value = _research_cutoff(cutoff)
    _train_ordered_ids, train_ids = _research_game_ids(train_maps, "training")
    _validation_ordered_ids, validation_ids = _research_game_ids(
        validation_maps, "validation"
    )
    train_series_pairs = _research_series_pairs(train_maps, "training")
    validation_series_pairs = _research_series_pairs(validation_maps, "validation")
    shared_series_ids = sorted(set(train_series_pairs) & set(validation_series_pairs))
    if shared_series_ids:
        raise ValueError(
            "training and validation share grid_series_id: "
            + ", ".join(shared_series_ids)
        )
    if set(train_ids) & set(validation_ids):
        raise ValueError("training and validation game identities overlap")
    train_dates = _research_dates(train_maps, "training")
    validation_dates = _research_dates(validation_maps, "validation")
    if (train_dates > cutoff_value).any():
        raise ValueError("training frame contains rows after the strict cutoff")
    if (validation_dates <= cutoff_value).any():
        raise ValueError("validation frame contains rows at or before the strict cutoff")
    _research_train_outcomes(train_maps)
    verified_source = _research_verified_source_receipt(source_receipt)
    source_as_of = pd.Timestamp(verified_source["source_as_of"])
    if (train_dates > source_as_of).any() or (validation_dates > source_as_of).any():
        raise ValueError("training or validation rows exceed the verified source_as_of")
    eligible_ids = set(verified_source["model_eligible_game_ids"])
    requested_ids = set(train_ids) | set(validation_ids)
    if not requested_ids.issubset(eligible_ids):
        raise ValueError("train or validation IDs are outside the verified source census")
    computed_train_identity = _research_id_identity(train_ids)
    computed_validation_identity = _research_id_identity(validation_ids)
    if (
        train_source_identity_sha256 is not None
        and str(train_source_identity_sha256).lower() != computed_train_identity
    ):
        raise ValueError("training source identity does not match the exact training IDs")
    if (
        validation_source_identity_sha256 is not None
        and str(validation_source_identity_sha256).lower() != computed_validation_identity
    ):
        raise ValueError(
            "validation source identity does not match the exact validation IDs"
        )
    source_identity = verified_source["source_identity_sha256"]
    if (
        source_identity_sha256 is not None
        and str(source_identity_sha256).lower() != source_identity
    ):
        raise ValueError("source identity does not match the verified source receipt")
    train_identity = _research_identity(
        train_source_identity_sha256,
        computed_train_identity,
        "training",
    )
    validation_identity = _research_identity(
        validation_source_identity_sha256,
        computed_validation_identity,
        "validation",
    )
    ordered_train_maps = _research_order_frame(train_maps, "training")

    observations, series_audit = _observations(
        ordered_train_maps,
        cutoff_value,
        cfg.half_life_days,
    )
    if observations.empty:
        raise ValueError("training frame has no resolvable observations")
    if series_audit.get("n_unsafe_maps") or series_audit.get("n_unresolved_maps"):
        raise ValueError("training series identity is not safe for paired evaluation")
    fit_state = _research_fit_state(observations, cfg)
    prediction_rows, missing_ids, unseen_teams, unseen_game_ids = _research_validation_rows(
        validation_maps,
        fit_state=fit_state,
        training_observations=observations,
    )
    prediction_rows = sorted(prediction_rows, key=lambda row: row["game_id"])
    output_sha256 = _research_sha256(prediction_rows)
    config_payload = dict(cfg.__dict__)
    config_sha256 = _research_sha256(config_payload)
    terms = {
        "team_logit": {
            team: float(fit_state["beta"][index])
            for team, index in fit_state["team_index"].items()
        },
        "league_logit": {
            league: float(fit_state["beta"][len(fit_state["teams"]) + index])
            for league, index in fit_state["league_index"].items()
        },
        "side_logit": float(fit_state["beta"][-1]),
    }
    train_receipt = {
        "game_ids": list(train_ids),
        "game_count": len(train_ids),
        "identity_sha256": _research_id_identity(train_ids),
        "source_identity_sha256": train_identity,
        "ordered_input_ids": list(train_ids),
        "input_game_ids": list(train_ids),
    }
    validation_receipt = {
        "game_ids": list(validation_ids),
        "game_count": len(validation_ids),
        "identity_sha256": _research_id_identity(validation_ids),
        "source_identity_sha256": validation_identity,
        "ordered_input_ids": list(validation_ids),
        "input_game_ids": list(validation_ids),
        "scored_game_ids": [row["game_id"] for row in prediction_rows],
        "missing_game_ids": missing_ids,
    }
    return {
        "schema_version": RESEARCH_PREDICTION_SCHEMA,
        "authority": "research_only",
        "writes_artifacts": False,
        "strict_cutoff": True,
        "cutoff": cutoff_value.isoformat(),
        "source_identity_sha256": source_identity,
        "source": {
            "source_identity_sha256": source_identity,
            "receipt_sha256": verified_source["receipt_sha256"],
            "source_as_of": verified_source["source_as_of"],
            "source_game_count": verified_source["source_game_count"],
            "train_source_identity_sha256": train_identity,
            "validation_source_identity_sha256": validation_identity,
            "model_eligible_game_count": verified_source["model_eligible_game_count"],
            "model_eligible_identity_sha256": verified_source[
                "model_eligible_identity_sha256"
            ],
        },
        "scope_game_identity_sha256": _research_id_identity(
            sorted(requested_ids)
        ),
        "train_game_ids": list(train_ids),
        "validation_game_ids": list(validation_ids),
        "train_receipt": train_receipt,
        "validation_receipt": validation_receipt,
        "missing_ids": missing_ids,
        "missing": {
            "validation_game_ids": missing_ids,
            "unseen_team_keys": unseen_teams,
            "unseen_model_game_ids": unseen_game_ids,
            "blockers": [
                blocker
                for blocker in (
                    "validation rows with unseen teams are excluded"
                    if unseen_game_ids
                    else None,
                    "validation rows without a finite prediction are missing"
                    if missing_ids and not unseen_game_ids
                    else None,
                )
                if blocker is not None
            ],
        },
        "config": config_payload,
        "config_sha256": config_sha256,
        "implementation_sha256": HIERARCHICAL_IMPLEMENTATION_SHA256,
        "fit": {
            "n_observations": fit_state["n_observations"],
            "n_maps": fit_state["n_maps"],
            "optimizer_success": fit_state["optimizer_success"],
            "optimizer_message": fit_state["optimizer_message"],
            "n_iterations": fit_state["n_iterations"],
            "series_identity": series_audit,
        },
        "terms": terms,
        "predictions": prediction_rows,
        "output_row_count": len(prediction_rows),
        "output_sha256": output_sha256,
    }


def fit_hierarchical_bt(
    maps: pd.DataFrame,
    cfg: HierarchicalBTConfig | None = None,
    as_of: pd.Timestamp | None = None,
    write: bool = True,
    output_dir: Path | None = None,
    cache_dir: Path | None = None,
    source_identity_sha256: str | None = None,
    cache_slot: str = "current",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit the current conservative ladder and optionally persist its snapshot."""

    cfg = cfg or HierarchicalBTConfig()
    cache_key = _cache_key(
        maps,
        as_of=as_of,
        cfg=cfg,
        source_identity_sha256=source_identity_sha256,
        cache_slot=cache_slot,
    )
    if cache_dir is not None and cache_key is not None:
        cached = _load_cached_fit(
            Path(cache_dir),
            cache_slot=cache_slot,
            key=cache_key,
        )
        if cached is not None:
            snapshot, metadata = cached
            if write and cache_slot == "current":
                destination = Path(output_dir or FEATURES_DIR)
                destination.mkdir(parents=True, exist_ok=True)
                snapshot.to_parquet(destination / "ratings_hierarchical_snapshot.parquet", index=False)
                snapshot.to_parquet(destination / "ratings_snapshot.parquet", index=False)
                (destination / "ratings_meta.json").write_text(
                    json.dumps(metadata, indent=2),
                    encoding="utf-8",
                )
                (destination / "ratings_hierarchical_meta.json").write_text(
                    json.dumps(metadata, indent=2),
                    encoding="utf-8",
                )
            return snapshot, metadata
    input_audit = audit_rating_inputs(maps)
    obs, series_audit = _observations(maps, as_of, cfg.half_life_days)
    if obs.empty:
        empty = pd.DataFrame(columns=["team", "team_key", "mu_total", "sigma"])
        return empty, {
            "model": "hierarchical_bt",
            "n_series": 0,
            "taxonomy_version": TAXONOMY_VERSION,
            "input_audit": input_audit,
            "series_identity": {
                "revision": "2026-08-09.1",
                "n_authoritative_series": 0,
                "n_game_level_maps": 0,
                **series_audit,
            },
        }

    X, teams, leagues = _design(obs)
    y = obs["y_a"].to_numpy(float)
    weight = obs["weight"].to_numpy(float)
    n_team = len(teams)
    n_league = len(leagues)
    penalty = np.array(
        [cfg.team_l2] * n_team + [cfg.league_l2] * n_league + [cfg.side_l2],
        dtype=float,
    )

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        # L-BFGS-B may probe a non-finite point during a failed line search;
        # keep the likelihood numerically bounded and let the explicit
        # parameter bounds handle separation.
        safe_beta = np.clip(np.nan_to_num(beta, nan=0.0, posinf=8.0, neginf=-8.0), -8.0, 8.0)
        eta = np.einsum("ij,j->i", X, safe_beta, optimize=True)
        p = expit(eta)
        value = float(
            np.sum(weight * (np.logaddexp(0.0, eta) - y * eta))
            + 0.5 * np.sum(penalty * safe_beta * safe_beta)
        )
        gradient = np.einsum("i,ij->j", weight * (p - y), X, optimize=True) + penalty * safe_beta
        return value, gradient

    result = minimize(
        objective,
        np.zeros(X.shape[1], dtype=float),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-8.0, 8.0)] * X.shape[1],
        options={"maxiter": cfg.max_iter, "ftol": 1e-10, "gtol": 1e-8},
    )
    beta = np.clip(np.nan_to_num(result.x, nan=0.0, posinf=8.0, neginf=-8.0), -8.0, 8.0)
    p = expit(np.einsum("ij,j->i", X, beta, optimize=True))
    hessian = np.einsum(
        "i,ij,ik->jk", weight * p * (1.0 - p), X, X, optimize=True
    ) + np.diag(penalty)
    try:
        covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(hessian, rcond=1e-10)

    team_idx = {value: i for i, value in enumerate(teams)}
    league_idx = {value: i for i, value in enumerate(leagues)}
    latest = obs.sort_values("date").groupby("team_a").tail(1)
    latest_b = obs.sort_values("date").groupby("team_b").tail(1)
    home_by_team: dict[str, str] = {}
    home_at: dict[str, pd.Timestamp] = {}
    for _, row in pd.concat([latest, latest_b]).iterrows():
        for team, home in ((row["team_a"], row["home_a"]), (row["team_b"], row["home_b"])):
            if team not in home_at or row["date"] >= home_at[team]:
                home_at[team] = row["date"]
                home_by_team[team] = home
    display_by_team: dict[str, str] = {}
    for _, row in obs.iterrows():
        display_by_team.setdefault(row["team_a"], row["team_a_name"])
        display_by_team.setdefault(row["team_b"], row["team_b_name"])

    rows: list[dict[str, Any]] = []
    for team in teams:
        home = home_by_team.get(team, "UNKNOWN")
        vector = np.zeros(X.shape[1], dtype=float)
        vector[team_idx[team]] = 1.0
        if home in league_idx:
            vector[n_team + league_idx[home]] = 1.0
        mean_logit = float(np.dot(vector, beta))
        variance = max(float(np.einsum("i,ij,j->", vector, covariance, vector)), 0.0)
        sigma = max(cfg.min_sigma, LOGIT_TO_ELO * math.sqrt(variance))
        rating = cfg.base_rating + LOGIT_TO_ELO * mean_logit
        team_obs = obs[(obs["team_a"] == team) | (obs["team_b"] == team)]
        intl = int(team_obs["international"].sum())
        last_game_date = None
        if not team_obs.empty and pd.notna(team_obs["date"].max()):
            last_game_date = pd.Timestamp(team_obs["date"].max()).isoformat()
        bridge_gap = max(0.0, 1.0 - min(intl, cfg.bridge_target_series) / max(cfg.bridge_target_series, 1))
        sigma = math.hypot(sigma, cfg.unbridged_league_sigma * bridge_gap)
        rows.append(
            {
                "team": display_by_team.get(team, team),
                "team_key": team,
                "mu_total": rating,
                "mu_regional": cfg.base_rating + LOGIT_TO_ELO * beta[team_idx[team]],
                "mu_meta": LOGIT_TO_ELO * (beta[n_team + league_idx[home]] if home in league_idx else 0.0),
                "sigma": sigma,
                "rating_p10": rating - cfg.conservative_z * sigma,
                "n_series": int(len(team_obs)),
                "n_maps": int(team_obs["n_maps"].sum()),
                "international_series": intl,
                "home_league": home,
                "last_game_date": last_game_date,
                "model": "hierarchical_bt",
            }
        )
    snapshot = pd.DataFrame(rows).sort_values("rating_p10", ascending=False).reset_index(drop=True)
    meta: dict[str, Any] = {
        "model": "hierarchical_bt",
        "taxonomy_version": TAXONOMY_VERSION,
        "n_series": int(len(obs)),
        "n_maps": int(obs["n_maps"].sum()),
        "n_teams": int(len(teams)),
        "n_leagues": int(len(leagues)),
        "as_of": str(obs["date"].max()),
        "config": cfg.__dict__,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "input_audit": input_audit,
        "series_identity": {
            "revision": "2026-08-09.1",
            "n_authoritative_series": int((obs["series_source"] == "grid").sum()),
            "n_game_level_maps": int((obs["series_source"] == "none").sum()),
            **series_audit,
        },
        "note": "Series-collapsed penalized MAP Bradley-Terry with explicit series identity (authoritative GRID series id when safe, stable game-level keys otherwise) and local Laplace uncertainty plus explicit uncertainty inflation for teams without international bridges; use rating_p10 for conservative rank.",
    }
    if cache_dir is not None and cache_key is not None:
        _write_cached_fit(
            Path(cache_dir),
            cache_slot=cache_slot,
            key=cache_key,
            snapshot=snapshot,
            metadata=meta,
        )
    if write:
        destination = Path(output_dir or FEATURES_DIR)
        destination.mkdir(parents=True, exist_ok=True)
        snapshot.to_parquet(destination / "ratings_hierarchical_snapshot.parquet", index=False)
        # The hierarchical fit is the public ladder snapshot.  The sequential
        # benchmark remains available as ratings_dual_snapshot.parquet.
        snapshot.to_parquet(destination / "ratings_snapshot.parquet", index=False)
        (destination / "ratings_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (destination / "ratings_hierarchical_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return snapshot, meta


def _sunday_utc(as_of: pd.Timestamp | None) -> pd.Timestamp:
    stamp = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.normalize() - pd.Timedelta(days=(stamp.weekday() + 1) % 7)


def _recent_team_baseline_anchor(
    previous_as_of: pd.Timestamp | None,
    sunday_baseline: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> pd.Timestamp:
    """Previous-refresh movement anchor with safe fallbacks.

    The CLI passes ISO strings with a trailing ``Z`` (tz-aware); the cutoff
    from the refresh is naive.  Normalize every input so the comparison is
    robust regardless of caller convention.
    """
    if previous_as_of is None:
        return sunday_baseline
    anchor = pd.Timestamp(previous_as_of)
    if anchor.tzinfo is not None:
        anchor = anchor.tz_convert("UTC").tz_localize(None)
    base = pd.Timestamp(sunday_baseline)
    if base.tzinfo is not None:
        base = base.tz_convert("UTC").tz_localize(None)
    cap = pd.Timestamp(cutoff)
    if cap.tzinfo is not None:
        cap = cap.tz_convert("UTC").tz_localize(None)
    if anchor >= cap or anchor < base - pd.Timedelta(days=400):
        return base
    return anchor


def _frame_through_cutoff(maps: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Return the exact source rows visible to a cutoff fit."""

    if maps is None or maps.empty or "date" not in maps.columns:
        return maps.copy() if maps is not None else pd.DataFrame()
    dates = pd.to_datetime(maps["date"], errors="coerce", utc=True)
    cap = pd.Timestamp(cutoff)
    if pd.isna(cap):
        return maps.iloc[0:0].copy()
    if cap.tzinfo is None:
        cap = cap.tz_localize("UTC")
    else:
        cap = cap.tz_convert("UTC")
    return maps.loc[dates.le(cap)].copy()


def build_team_weekly_ranks(
    maps: pd.DataFrame,
    *,
    as_of: pd.Timestamp | None = None,
    min_series: int = 5,
    previous_as_of: pd.Timestamp | None = None,
    current: pd.DataFrame | None = None,
    cache_dir: Path | None = None,
    source_identity_sha256: str | None = None,
) -> dict[str, Any]:
    """Return team rank movement against the previous refresh's ladder.

    Both snapshots use the same hierarchical fit and the same conservative
    ``rating_p10`` ordering as the public team ladder. The recent baseline is
    the previous refresh's cutoff when ``previous_as_of`` is provided (so
    movement reflects every published cycle), falling back to the prior
    Sunday snapshot otherwise. New games therefore change the ladder and its
    movement in one refresh.
    """

    if min_series < 1:
        raise ValueError("min_series must be positive")
    week_start = _sunday_utc(as_of)
    previous_start = week_start - pd.Timedelta(days=7)
    cutoff = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    recent_anchor = _recent_team_baseline_anchor(previous_as_of, previous_start, cutoff)
    if current is None:
        current, _ = fit_hierarchical_bt(maps, as_of=cutoff, write=False)
    previous_cutoff = recent_anchor - pd.Timedelta(microseconds=1)
    previous_maps = _frame_through_cutoff(maps, previous_cutoff)
    previous_source_identity, _, _ = _frame_source_identity(previous_maps)
    previous, _ = fit_hierarchical_bt(
        previous_maps,
        as_of=previous_cutoff,
        write=False,
        cache_dir=cache_dir,
        source_identity_sha256=previous_source_identity,
        cache_slot="previous",
    )

    def order(snapshot: pd.DataFrame) -> tuple[dict[str, int], dict[str, float]]:
        if snapshot.empty:
            return {}, {}
        eligible = snapshot[snapshot["n_series"].fillna(0).ge(min_series)].copy()
        eligible["rank_value"] = pd.to_numeric(eligible["rating_p10"], errors="coerce")
        eligible["mu_value"] = pd.to_numeric(eligible["mu_total"], errors="coerce")
        eligible = eligible.dropna(subset=["rank_value"])
        eligible["team_sort"] = eligible["team"].astype(str).str.casefold()
        eligible = eligible.sort_values(["rank_value", "team_sort"], ascending=[False, True])
        ranks = {str(team): rank for rank, team in enumerate(eligible["team"].astype(str), start=1)}
        mus = {
            str(team): float(mu) for team, mu in zip(eligible["team"].astype(str), eligible["mu_value"])
        }
        return ranks, mus

    current_rank, current_mu = order(current)
    previous_rank, previous_mu = order(previous)
    current_through = pd.Timestamp(cutoff)
    if current_through.tzinfo is not None:
        current_through = current_through.tz_convert("UTC").tz_localize(None)
    by_team: dict[str, dict[str, int | None]] = {}
    for team, rank in current_rank.items():
        prior = previous_rank.get(team)
        prior_mu = previous_mu.get(team)
        mu = current_mu.get(team)
        by_team[team] = {
            "rank": rank,
            "delta": (prior - rank) if prior is not None else None,
            "mu_delta": (mu - prior_mu) if (mu is not None and prior_mu is not None) else None,
        }

    return {
        "as_of": f"{week_start.isoformat()}Z",
        "previous_as_of": f"{recent_anchor.isoformat()}Z",
        "current_through": f"{current_through.isoformat()}Z",
        "min_series": int(min_series),
        "by_team": by_team,
        "note": "Rank movement compares conservative team rating against the previous refresh (or the prior Sunday when no earlier refresh exists); positive delta means a climb.",
    }
