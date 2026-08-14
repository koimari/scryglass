"""Append validated Oracle's Elixir games to Supabase and the local cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
import pandas as pd

from lol_kills.etl.oe_ingest import (
    _normalize_patch_column,
    _validate_oe_csv,
    parse_oe_csv,
    validate_accepted_source_receipt,
)
from lol_kills.etl.riot_patch_receipts import RiotPatchReceiptError, load_patch_receipts
from lol_kills.etl.source_keys import canonical_source_game_key
from lol_kills.ratings.player_map_grades import CORE_INPUTS


GAME_SCHEMA = "scryglass:oe-game:v1"
IMPORT_SCHEMA = "scryglass:oe-database-import:v1"

# Oracle's Elixir revisions can remove or re-identify stored games (forfeits,
# replays, ID renumbering). The import fails closed when the source loses a
# stored game unless the game is reviewed and acknowledged here. Each entry
# records the canonical game ID and why it was removed from the OE source.
REVIEWED_REMOVED_GAME_IDS: dict[str, str] = {
    # LJL 2026-08-11 series L Guide Gaming vs Uwinks (BO3 at 08:09/09:13/10:18Z)
    # was re-identified by OE in the 2026-08-12T16:08Z revision; the same
    # matchup appears under new IDs (LOLTMNT01_442537, LOLTMNT01_441618).
    "LOLTMNT01_441503": "oe-revision-2026-08-12: ljl series re-identified",
    "LOLTMNT01_442486": "oe-revision-2026-08-12: ljl series re-identified",
    "LOLTMNT01_442514": "oe-revision-2026-08-12: ljl series re-identified",
}
STATE_SCHEMA = "scryglass:oe-local-cache-state:v1"
TRANSFORM_VERSION = "oe-normalization:v3"
REQUEST_TIMEOUT_SECONDS = 180.0
# Keep version writes below the Supabase statement budget. This matters during
# a transform migration, when one unchanged source can rewrite the full cache.
WRITE_BATCH_SIZE = 20
WRITE_CONCURRENCY = 4
READ_PAGE_SIZE = 1_000
ROLE_ORDER = {"top": 0, "jng": 1, "mid": 2, "bot": 3, "sup": 4}
SIDE_ORDER = {"Blue": 0, "Red": 1}


class OeDatabaseError(RuntimeError):
    """An incremental OE database update failed its safety checks."""


@dataclass(frozen=True)
class PreparedGame:
    canonical_game_id: str
    payload_sha256: str
    source_year: int
    game_date: str
    league: str
    patch: str | None
    statistics_complete: bool
    source_file_sha256: str

    def version_row(self, payload: dict[str, Any]) -> dict[str, Any]:
        if _canonical_json_sha256(payload) != self.payload_sha256:
            raise OeDatabaseError(
                f"prepared OE payload changed for {self.canonical_game_id}"
            )
        return {
            "canonical_game_id": self.canonical_game_id,
            "payload_sha256": self.payload_sha256,
            "source_year": self.source_year,
            "game_date": self.game_date,
            "league": self.league,
            "patch": self.patch,
            "statistics_complete": self.statistics_complete,
            "source_file_sha256": self.source_file_sha256,
            "payload": payload,
        }

    def current_row(self) -> dict[str, Any]:
        return {
            "canonical_game_id": self.canonical_game_id,
            "payload_sha256": self.payload_sha256,
            "source_year": self.source_year,
            "game_date": self.game_date,
            "league": self.league,
            "patch": self.patch,
            "statistics_complete": self.statistics_complete,
            "source_file_sha256": self.source_file_sha256,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }


@dataclass
class PreparedImport:
    year: int
    csv_path: Path
    source: dict[str, Any]
    team_rows: pd.DataFrame
    player_rows: pd.DataFrame
    team_group_indices: dict[str, tuple[int, ...]]
    player_group_indices: dict[str, tuple[int, ...]]
    games: dict[str, PreparedGame]
    source_game_ids: tuple[str, ...]
    quarantined_game_ids: tuple[str, ...]
    quarantined_games: dict[str, str]
    _team_row_payloads: list[dict[str, Any]] | None = None
    _player_row_payloads: list[dict[str, Any]] | None = None

    def payload_rows_for(self, game_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Precomputed canonical row payloads for one game (fast path)."""
        if self._team_row_payloads is None or self._player_row_payloads is None:
            return _game_frames_and_payload_rows(self, game_id)
        team_indices = self.team_group_indices[game_id]
        player_indices = self.player_group_indices[game_id]
        return (
            [self._team_row_payloads[pos] for pos in team_indices],
            [self._player_row_payloads[pos] for pos in player_indices],
        )


def _project_url(value: str) -> str:
    raw = value.strip().rstrip("/")
    parsed = urllib.parse.urlsplit(raw)
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or parsed.port
        or parsed.query
        or parsed.fragment
        or not (parsed.hostname or "").endswith(".supabase.co")
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("Supabase URL must be an HTTPS project URL")
    return raw


def _secret_key(value: str) -> str:
    key = value.strip()
    if not key.startswith("sb_secret_") or len(key) < 24 or any(char.isspace() for char in key):
        raise ValueError("Supabase secret key is malformed")
    return key


def _canonical_json_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _canonical_string_sha256(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _parse_game_date(date_text: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(date_text)
    except (ValueError, TypeError):
        return pd.NaT
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    return parsed


def _game_identity_sha256(game_ids: Iterable[str]) -> str:
    canonical = sorted({str(game_id) for game_id in game_ids if str(game_id)})
    raw = ("\n".join(canonical) + "\n").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _payload_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    raw = frame.to_json(
        orient="records",
        date_format="iso",
        date_unit="us",
        double_precision=15,
        force_ascii=False,
    )
    rows = json.loads(raw)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise OeDatabaseError("OE rows could not be serialized")
    return rows


def _payload_rows_fast(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Faster _payload_rows: same row dicts, ujson decode of the frame JSON."""
    raw = frame.to_json(
        orient="records",
        date_format="iso",
        date_unit="us",
        double_precision=15,
        force_ascii=False,
    )
    try:
        from pandas._libs import json as _ujson
        rows = _ujson.loads(raw)
    except (ImportError, AttributeError):
        rows = json.loads(raw)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise OeDatabaseError("OE rows could not be serialized")
    return rows


def _serialize_rows_chunk(chunk: list[dict[str, Any]]) -> list[str]:
    return [
        json.dumps(row, ensure_ascii=False, sort_keys=False, separators=(",", ":"))
        for row in chunk
    ]


def _serialize_rows_parallel(rows: list[dict[str, Any]], workers: int = 8) -> list[str]:
    if len(rows) < 20_000 or workers <= 1:
        return _serialize_rows_chunk(rows)
    chunk_size = max(len(rows) // workers, 1)
    chunks = [rows[index : index + chunk_size] for index in range(0, len(rows), chunk_size)]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as executor:
        serialized = list(executor.map(_serialize_rows_chunk, chunks))
    return [value for chunk in serialized for value in chunk]


def _clean_role(value: Any) -> str:
    token = str(value or "").strip().casefold()
    return {"jungle": "jng", "support": "sup", "adc": "bot"}.get(token, token)


def _clean_side(value: Any) -> str:
    return str(value or "").strip().title()


def _single_text(frame: pd.DataFrame, column: str) -> str | None:
    if column not in frame.columns:
        return None
    values = {
        str(value).strip()
        for value in frame[column]
        if pd.notna(value) and str(value).strip() and str(value).strip().casefold() != "nan"
    }
    if len(values) != 1:
        return None
    return next(iter(values))


def _statistics_complete(player_rows: pd.DataFrame) -> bool:
    if any(column not in player_rows.columns for column in CORE_INPUTS):
        return False
    statistics = player_rows[list(CORE_INPUTS)].apply(pd.to_numeric, errors="coerce")
    nonnegative = (
        "kills",
        "deaths",
        "assists",
        "teamkills",
        "dpm",
        "damageshare",
        "totalgold",
        "cspm",
        "wpm",
        "wcpm",
    )
    complete = (
        statistics.notna().all().all()
        and statistics[list(nonnegative)].ge(0).all().all()
        and statistics["gamelength"].gt(0).all()
        and statistics["kills"].le(statistics["teamkills"]).all()
        and statistics["damageshare"].le(1).all()
    )
    if not bool(complete):
        return False
    if "datacompleteness" in player_rows.columns and not player_rows[
        "datacompleteness"
    ].astype(str).str.casefold().eq("complete").all():
        return False
    for side in ("Blue", "Red"):
        side_mask = player_rows["side"].map(_clean_side).eq(side)
        damage_share = pd.to_numeric(
            player_rows.loc[side_mask, "damageshare"], errors="coerce"
        ).sum()
        if abs(float(damage_share) - 1.0) > 1e-5:
            return False
    return True


def _identity_error(team_rows: pd.DataFrame, player_rows: pd.DataFrame) -> str | None:
    if len(team_rows) != 2 or len(player_rows) != 10:
        return "row_count"
    required_team = {"side", "result", "teamname", "date", "league"}
    required_player = {
        "side",
        "position",
        "playername",
        "champion",
        "teamname",
        "date",
        "league",
    }
    if not required_team.issubset(team_rows.columns) or not required_player.issubset(
        player_rows.columns
    ):
        return "schema"
    team_sides = team_rows["side"].map(_clean_side)
    team_results = pd.to_numeric(team_rows["result"], errors="coerce")
    team_names = team_rows["teamname"].astype("string").fillna("").str.strip()
    if (
        set(team_sides) != {"Blue", "Red"}
        or set(team_results.dropna().astype(int)) != {0, 1}
        or team_names.eq("").any()
        or team_names.str.casefold().nunique() != 2
    ):
        return "teams"
    names = player_rows["playername"].astype("string").fillna("").str.strip()
    champions = player_rows["champion"].astype("string").fillna("").str.strip()
    if names.eq("").any() or names.str.casefold().nunique() != 10:
        return "players"
    if champions.eq("").any():
        return "champions"
    for side in ("Blue", "Red"):
        team_row = team_rows.loc[team_sides.eq(side)]
        side_players = player_rows.loc[player_rows["side"].map(_clean_side).eq(side)]
        if len(team_row) != 1 or len(side_players) != 5:
            return "sides"
        roles = side_players["position"].map(_clean_role)
        if set(roles) != set(ROLE_ORDER) or roles.nunique() != 5:
            return "roles"
        expected_team = str(team_row.iloc[0]["teamname"]).strip().casefold()
        player_teams = {
            str(value).strip().casefold()
            for value in side_players["teamname"]
            if pd.notna(value)
        }
        if player_teams != {expected_team}:
            return "player_teams"
    combined = pd.concat(
        [team_rows[["date", "league"]], player_rows[["date", "league"]]],
        ignore_index=True,
    )
    if _single_text(combined, "date") is None or _single_text(combined, "league") is None:
        return "game_metadata"
    return None


def _sorted_game_rows(
    team_rows: pd.DataFrame, player_rows: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    teams = team_rows.copy()
    teams["_side_order"] = teams["side"].map(lambda value: SIDE_ORDER.get(_clean_side(value), 99))
    teams = teams.sort_values(["_side_order"]).drop(columns=["_side_order"])
    players = player_rows.copy()
    players["_side_order"] = players["side"].map(
        lambda value: SIDE_ORDER.get(_clean_side(value), 99)
    )
    players["_role_order"] = players["position"].map(
        lambda value: ROLE_ORDER.get(_clean_role(value), 99)
    )
    players = players.sort_values(["_side_order", "_role_order"]).drop(
        columns=["_side_order", "_role_order"]
    )
    return teams, players


def _group_indices(frame: pd.DataFrame) -> dict[str, tuple[int, ...]]:
    return {
        str(game_id): tuple(int(index) for index in indices)
        for game_id, indices in frame.groupby("gameid", sort=False).indices.items()
    }


def _game_frames(
    prepared: PreparedImport, game_id: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        team_indices = prepared.team_group_indices[game_id]
        player_indices = prepared.player_group_indices[game_id]
    except KeyError as error:
        raise OeDatabaseError(f"prepared OE game is missing rows: {game_id}") from error
    return _sorted_game_rows(
        prepared.team_rows.iloc[list(team_indices)],
        prepared.player_rows.iloc[list(player_indices)],
    )


def _game_frames_and_payload_rows(
    prepared: PreparedImport, game_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    teams, players = _game_frames(prepared, game_id)
    return _payload_rows(teams), _payload_rows(players)


def payload_for_game(prepared: PreparedImport, game_id: str) -> dict[str, Any]:
    team_rows, player_rows = prepared.payload_rows_for(game_id)
    return {
        "schema_version": GAME_SCHEMA,
        "canonical_game_id": game_id,
        "team_rows": team_rows,
        "player_rows": player_rows,
    }




def _canonical_order(frame: pd.DataFrame, *, by_role: bool) -> pd.DataFrame:
    """One global stable sort so every game's rows are contiguous and canonical."""
    game_ids = frame["gameid"].astype(str).to_numpy()
    side_order = np.fromiter(
        (SIDE_ORDER.get(_clean_side(value), 99) for value in frame["side"].astype(str).to_numpy()),
        dtype=np.int64,
        count=len(frame),
    )
    keys: list[np.ndarray] = []
    if by_role:
        role_order = np.fromiter(
            (ROLE_ORDER.get(_clean_role(value), 99) for value in frame["position"].astype(str).to_numpy()),
            dtype=np.int64,
            count=len(frame),
        )
        keys.append(role_order)
    keys.append(side_order)
    keys.append(game_ids)
    order = np.lexsort(keys)
    return frame.iloc[order].reset_index(drop=True)


def _clean_text_values(series: pd.Series) -> np.ndarray:
    """Object array of stripped strings; None for empty or 'nan' values."""
    values = series.to_numpy(dtype=object)
    out = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        if value is None or pd.isna(value):
            out[index] = None
            continue
        cleaned = str(value).strip()
        out[index] = (
            None
            if (not cleaned or cleaned.casefold() in {"nan", "nat", "none", "<na>"})
            else cleaned
        )
    return out


def _single_text_per_game(game_ids: np.ndarray, values: np.ndarray) -> dict[str, str | None]:
    """For each contiguous game: the single distinct non-empty value, else None."""
    result: dict[str, str | None] = {}
    total = len(game_ids)
    index = 0
    while index < total:
        game_id = str(game_ids[index])
        end = index + 1
        while end < total and str(game_ids[end]) == game_id:
            end += 1
        distinct: set[str] = set()
        value: str | None = None
        for pos in range(index, end):
            candidate = values[pos]
            if candidate is None:
                continue
            if value is None:
                value = candidate
            distinct.add(candidate)
        result[game_id] = value if len(distinct) == 1 else None
        index = end
    return result


def _merge_single_text(*maps: dict[str, str | None]) -> dict[str, str | None]:
    """Per game: the common single value across maps, or None when they differ."""
    game_ids = set()
    for mapping in maps:
        game_ids.update(mapping)
    result: dict[str, str | None] = {}
    for game_id in game_ids:
        values = {
            value
            for mapping in maps
            if (value := mapping.get(game_id)) is not None and str(value).strip()
        }
        result[game_id] = next(iter(values)) if len(values) <= 1 else None
    return result


def _game_identity_error(
    team: dict[str, np.ndarray], t_index: tuple[int, ...],
    players: dict[str, np.ndarray], p_index: tuple[int, ...],
    date_single: str | None, league_single: str | None,
) -> str | None:
    if len(t_index) != 2 or len(p_index) != 10:
        return "row_count"
    team_sides = set(team["side"][list(t_index)])
    if team_sides != {"Blue", "Red"}:
        return "teams"
    team_result_values = pd.to_numeric(
        pd.Series(team["result"][list(t_index)]), errors="coerce"
    )
    if team_result_values.isna().any():
        return "teams"
    team_results = {int(value) for value in team_result_values}
    if team_results != {0, 1}:
        return "teams"
    team_names = team["name"][list(t_index)]
    if any(not value for value in team_names) or len(set(team_names)) != 2:
        return "teams"
    player_names = players["name"][list(p_index)]
    if any(not value for value in player_names) or len(set(player_names)) != 10:
        return "players"
    player_champions = players["champion"][list(p_index)]
    if any(not value for value in player_champions):
        return "champions"
    for side in ("Blue", "Red"):
        team_row = [pos for pos in t_index if team["side"][pos] == side]
        side_positions = [pos for pos in p_index if players["side"][pos] == side]
        if len(team_row) != 1 or len(side_positions) != 5:
            return "sides"
        roles = {players["role"][pos] for pos in side_positions}
        if roles != set(ROLE_ORDER) or len(roles) != 5:
            return "roles"
        expected_team = str(team["name"][team_row[0]]).strip().casefold()
        player_teams = {
            str(players["team"][pos]).strip().casefold()
            for pos in side_positions
        }
        if player_teams != {expected_team}:
            return "player_teams"
    if date_single is None or league_single is None:
        return "game_metadata"
    return None


def _game_statistics_complete(players: dict[str, np.ndarray], p_index: tuple[int, ...]) -> bool:
    positions = list(p_index)
    stats = players["stat"]
    if any(column not in stats for column in CORE_INPUTS):
        return False
    for column in CORE_INPUTS:
        if np.isnan(stats[column][positions]).any():
            return False
    for column in ("kills", "deaths", "assists", "teamkills", "dpm", "damageshare", "totalgold", "cspm", "wpm", "wcpm"):
        if (stats[column][positions] < 0).any():
            return False
    if (stats["gamelength"][positions] <= 0).any():
        return False
    if (stats["kills"][positions] > stats["teamkills"][positions]).any():
        return False
    if (stats["damageshare"][positions] > 1).any():
        return False
    completeness = players["datacompleteness"]
    if len(completeness) and any(
        str(value).casefold() != "complete" for value in completeness[positions]
    ):
        return False
    for side in ("Blue", "Red"):
        side_positions = [pos for pos in positions if players["side"][pos] == side]
        damage_share = float(np.sum(stats["damageshare"][side_positions]))
        if abs(damage_share - 1.0) > 1e-5:
            return False
    return True



def _prepare_import_fast(
    csv_path: Path,
    year: int,
    *,
    source: dict[str, Any] | None = None,
    patch_receipts: Mapping[str, Mapping[str, Any]] | None = None,
) -> PreparedImport:
    """Vectorized prepare_import: one global sort + precomputed arrays.

    Semantics are byte-identical to prepare_import (same payloads, hashes,
    quarantines, and receipts); only the pandas per-game overhead is removed.
    """
    path = csv_path.expanduser().resolve()
    source = source or _validate_oe_csv(path, str(year))
    team_rows, player_rows = parse_oe_csv(path, patch_receipts=patch_receipts)
    for frame in (team_rows, player_rows):
        frame["gameid"] = frame["gameid"].map(canonical_source_game_key)
        frame.reset_index(drop=True, inplace=True)
    source_ids = sorted(
        set(team_rows["gameid"].astype(str)).union(player_rows["gameid"].astype(str))
        - {""}
    )
    team_rows = _canonical_order(team_rows, by_role=False)
    player_rows = _canonical_order(player_rows, by_role=True)
    team_ids = team_rows["gameid"].astype(str).to_numpy()
    player_ids = player_rows["gameid"].astype(str).to_numpy()
    # Column-sort once so every row's keys are already in sorted order; then
    # json.dumps with sort_keys=False produces the exact canonical bytes the
    # original sort_keys=True serialization would, without re-sorting per row.
    team_rows_sorted = team_rows[list(sorted(team_rows.columns))]
    player_rows_sorted = player_rows[list(sorted(player_rows.columns))]
    all_team_rows = _payload_rows_fast(team_rows_sorted)
    all_player_rows = _payload_rows_fast(player_rows_sorted)
    team_row_json = _serialize_rows_chunk(all_team_rows)
    player_row_json = _serialize_rows_chunk(all_player_rows)
    game_id_json = {
        game_id: json.dumps(game_id, ensure_ascii=False)[1:-1]
        for game_id in source_ids
    }

    team_date = _single_text_per_game(team_ids, _clean_text_values(team_rows["date"]))
    player_date = _single_text_per_game(player_ids, _clean_text_values(player_rows["date"]))
    date_single = _merge_single_text(team_date, player_date)
    team_league = _single_text_per_game(team_ids, _clean_text_values(team_rows["league"]))
    player_league = _single_text_per_game(player_ids, _clean_text_values(player_rows["league"]))
    league_single = _merge_single_text(team_league, player_league)
    team_patch = (
        _single_text_per_game(team_ids, _clean_text_values(team_rows["patch"]))
        if "patch" in team_rows.columns
        else {}
    )
    player_patch = (
        _single_text_per_game(player_ids, _clean_text_values(player_rows["patch"]))
        if "patch" in player_rows.columns
        else {}
    )

    team_arrays: dict[str, np.ndarray] = {
        "gameid": team_ids,
        "side": np.fromiter(
            (_clean_side(value) for value in team_rows["side"].astype(str).to_numpy()),
            dtype=object,
            count=len(team_rows),
        ),
        "name": _clean_text_values(team_rows["teamname"]),
        "result": pd.to_numeric(team_rows["result"], errors="coerce").to_numpy(dtype=float),
    }
    player_arrays: dict[str, np.ndarray] = {
        "gameid": player_ids,
        "side": np.fromiter(
            (_clean_side(value) for value in player_rows["side"].astype(str).to_numpy()),
            dtype=object,
            count=len(player_rows),
        ),
        "name": _clean_text_values(player_rows["playername"]),
        "champion": _clean_text_values(player_rows["champion"]),
        "team": _clean_text_values(player_rows["teamname"]),
        "role": np.fromiter(
            (_clean_role(value) for value in player_rows["position"].astype(str).to_numpy()),
            dtype=object,
            count=len(player_rows),
        ),
        "datacompleteness": (
            _clean_text_values(player_rows["datacompleteness"])
            if "datacompleteness" in player_rows.columns
            else np.empty(0, dtype=object)
        ),
        "stat": {
            column: pd.to_numeric(player_rows[column], errors="coerce").to_numpy(dtype=float)
            for column in CORE_INPUTS
            if column in player_rows.columns
        },
    }

    team_group = _group_indices(team_rows)
    player_group = _group_indices(player_rows)
    games: dict[str, PreparedGame] = {}
    quarantined: dict[str, str] = {}
    for game_id in source_ids:
        t_index = team_group.get(game_id, ())
        p_index = player_group.get(game_id, ())
        date_text = date_single.get(game_id)
        league = league_single.get(game_id)
        identity_error = _game_identity_error(
            team_arrays, t_index, player_arrays, p_index, date_text, league
        )
        if identity_error is not None:
            quarantined[game_id] = identity_error[:500]
            continue
        parsed_date = _parse_game_date(date_text)
        if pd.isna(parsed_date) or not league:
            quarantined[game_id] = (
                "game_date_invalid" if pd.isna(parsed_date) else "league_missing"
            )
            continue
        patch = team_patch.get(game_id) or player_patch.get(game_id)
        team_join = ",".join(team_row_json[pos] for pos in t_index)
        player_join = ",".join(player_row_json[pos] for pos in p_index)
        payload_text = (
            '{"canonical_game_id":"'
            + game_id_json[game_id]
            + '","player_rows":['
            + player_join
            + '],"schema_version":"'
            + GAME_SCHEMA
            + '","team_rows":['
            + team_join
            + "]}"
        )
        games[game_id] = PreparedGame(
            canonical_game_id=game_id,
            payload_sha256=_canonical_string_sha256(payload_text),
            source_year=year,
            game_date=parsed_date.isoformat(),
            league=league,
            patch=patch,
            statistics_complete=_game_statistics_complete(player_arrays, p_index),
            source_file_sha256=str(source["raw_sha256"]),
        )
    return PreparedImport(
        year=year,
        csv_path=path,
        source=source,
        team_rows=team_rows,
        player_rows=player_rows,
        team_group_indices=team_group,
        player_group_indices=player_group,
        games=games,
        source_game_ids=tuple(source_ids),
        quarantined_game_ids=tuple(sorted(quarantined)),
        quarantined_games=quarantined,
        _team_row_payloads=all_team_rows,
        _player_row_payloads=all_player_rows,
    )


def prepare_import(
    csv_path: Path,
    year: int,
    *,
    source: dict[str, Any] | None = None,
    patch_receipts: Mapping[str, Mapping[str, Any]] | None = None,
) -> PreparedImport:
    """Vectorized OE import preparation (globally sorted rows + numpy checks).

    Output is byte-identical to the previous per-game pandas implementation:
    same payloads, payload sha256s, quarantines, and receipts.
    """
    return _prepare_import_fast(
        csv_path,
        year,
        source=source,
        patch_receipts=patch_receipts,
    )


def _batches(values: list[Any], size: int = WRITE_BATCH_SIZE) -> Iterator[list[Any]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


class SupabaseOeDatabase:
    """Restricted PostgREST client for the private OE ingestion tables."""

    def __init__(self, project_url: str, secret_key: str, *, opener: Any | None = None) -> None:
        self.project_url = _project_url(project_url)
        self._secret_key = _secret_key(secret_key)
        self._opener = opener or urllib.request.build_opener()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(project_url={self.project_url!r}, secret_key=<redacted>)"

    def _request(
        self,
        method: str,
        path: str,
        payload: object | None = None,
        *,
        prefer: str | None = None,
    ) -> Any:
        raw_payload = None
        headers = {"apikey": self._secret_key, "Accept": "application/json"}
        if payload is not None:
            raw_payload = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if prefer:
            headers["Prefer"] = prefer
        request = urllib.request.Request(
            f"{self.project_url}/rest/v1/{path.lstrip('/')}",
            data=raw_payload,
            method=method,
            headers=headers,
        )
        try:
            with self._opener.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            detail = ""
            try:
                body = json.loads(error.read().decode("utf-8"))
                detail = str(body.get("message") or body.get("hint") or "")[:300]
            except Exception:  # noqa: BLE001
                pass
            suffix = f": {detail}" if detail else ""
            raise OeDatabaseError(
                f"Supabase OE request failed with HTTP {error.code} for "
                f"{path.split('?', 1)[0]}{suffix}"
            ) from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise OeDatabaseError(
                f"Supabase OE request failed for {path.split('?', 1)[0]}"
            ) from error
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise OeDatabaseError("Supabase OE response is invalid JSON") from error

    def current_hashes(self, year: int) -> dict[str, str]:
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = self._request(
                "GET",
                "scryglass_oe_games"
                f"?source_year=eq.{year}&select=canonical_game_id,payload_sha256"
                f"&order=canonical_game_id&limit={READ_PAGE_SIZE}&offset={offset}",
            )
            if not isinstance(page, list) or any(not isinstance(row, dict) for row in page):
                raise OeDatabaseError("Supabase OE game index is malformed")
            rows.extend(page)
            if len(page) < READ_PAGE_SIZE:
                break
            offset += READ_PAGE_SIZE
        return {
            str(row["canonical_game_id"]): str(row["payload_sha256"])
            for row in rows
        }

    def import_receipt(
        self, year: int, source_file_sha256: str, transform_version: str
    ) -> dict[str, Any] | None:
        rows = self._request(
            "GET",
            "scryglass_oe_imports"
            f"?source_year=eq.{year}&source_file_sha256=eq.{source_file_sha256}"
            f"&transform_version=eq.{urllib.parse.quote(transform_version, safe='')}"
            "&select=*&limit=1",
        )
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise OeDatabaseError("Supabase OE import receipt is malformed")
        return rows[0] if rows else None

    def append_versions(self, rows: list[dict[str, Any]]) -> None:
        for batch in _batches(rows):
            self._request(
                "POST",
                "scryglass_oe_game_versions?on_conflict=canonical_game_id,payload_sha256",
                batch,
                prefer="resolution=ignore-duplicates,return=minimal",
            )

    def upsert_current(self, rows: list[dict[str, Any]]) -> None:
        for batch in _batches(rows, READ_PAGE_SIZE):
            self._request(
                "POST",
                "scryglass_oe_games?on_conflict=canonical_game_id",
                batch,
                prefer="resolution=merge-duplicates,return=minimal",
            )

    def record_import(self, row: dict[str, Any]) -> None:
        self._request(
            "POST",
            "scryglass_oe_imports?on_conflict=source_year,source_file_sha256",
            [row],
            prefer="resolution=merge-duplicates,return=minimal",
        )


def _frame_game_ids(frame: pd.DataFrame) -> pd.Series:
    if frame.empty or "gameid" not in frame.columns:
        return pd.Series(dtype="string")
    return frame["gameid"].map(canonical_source_game_key).astype("string")


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(destination: Path, payload: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _cache_is_current(parquet_dir: Path, year: int, source_sha256: str) -> bool:
    if not (parquet_dir / "oe_team_games.parquet").is_file() or not (
        parquet_dir / "oe_player_games.parquet"
    ).is_file():
        return False
    state = _load_json(parquet_dir / "oe_incremental_state.json")
    years = state.get("years")
    if not isinstance(years, dict):
        return False
    year_state = years.get(str(year))
    return bool(
        isinstance(year_state, dict)
        and year_state.get("source_file_sha256") == source_sha256
    )


def _current_result_from_receipt(
    receipt: dict[str, Any], source: dict[str, Any], parquet_dir: Path
) -> dict[str, Any]:
    meta = _load_json(parquet_dir / "oe_meta.json")
    quarantined = receipt.get("quarantined_game_ids")
    quarantined_count = len(quarantined) if isinstance(quarantined, list) else 0
    accepted = int(receipt["accepted_games"])
    return {
        "schema_version": IMPORT_SCHEMA,
        "source_year": int(receipt["source_year"]),
        "source_file_sha256": source["raw_sha256"],
        "source_observed_through": receipt["source_observed_through"],
        "source_games": int(receipt["source_games"]),
        "accepted_games": accepted,
        "new_games": 0,
        "corrected_games": 0,
        "unchanged_games": accepted,
        "quarantined_games": quarantined_count,
        "statistics_complete_games": int(receipt["statistics_complete_games"]),
        "cache": {
            "replaced_games": 0,
            "removed_malformed_games": 0,
            "team_rows": int(meta.get("n_team_rows") or 0),
            "player_rows": int(meta.get("n_player_rows") or 0),
        },
        "status": "current",
    }


def update_local_cache(prepared: PreparedImport, parquet_dir: Path) -> dict[str, Any]:
    team_path = parquet_dir / "oe_team_games.parquet"
    player_path = parquet_dir / "oe_player_games.parquet"
    state_path = parquet_dir / "oe_incremental_state.json"
    meta_path = parquet_dir / "oe_meta.json"
    cached_team = (
        _normalize_patch_column(pd.read_parquet(team_path))
        if team_path.is_file()
        else pd.DataFrame()
    )
    cached_players = (
        _normalize_patch_column(pd.read_parquet(player_path))
        if player_path.is_file()
        else pd.DataFrame()
    )
    state = _load_json(state_path)
    years = state.get("years") if isinstance(state.get("years"), dict) else {}
    prior_year = years.get(str(prepared.year)) if isinstance(years, dict) else None
    prior_hashes = (
        prior_year.get("game_hashes")
        if isinstance(prior_year, dict) and isinstance(prior_year.get("game_hashes"), dict)
        else {}
    )
    current_hashes = {
        game_id: game.payload_sha256 for game_id, game in prepared.games.items()
    }
    local_team_ids = set(_frame_game_ids(cached_team).dropna().astype(str))
    local_player_ids = set(_frame_game_ids(cached_players).dropna().astype(str))
    source_ids = set(current_hashes)
    changed_ids = {
        game_id
        for game_id, digest in current_hashes.items()
        if prior_hashes.get(game_id) != digest
    }
    changed_ids.update(source_ids.difference(local_team_ids.intersection(local_player_ids)))
    stale_ids = (
        local_team_ids.union(local_player_ids)
        .difference(source_ids)
        .intersection(set(prepared.source_game_ids).union(REVIEWED_REMOVED_GAME_IDS))
    )
    remove_ids = changed_ids.union(stale_ids)
    if remove_ids:
        if not cached_team.empty:
            cached_team = cached_team.loc[~_frame_game_ids(cached_team).isin(remove_ids)].copy()
        if not cached_players.empty:
            cached_players = cached_players.loc[
                ~_frame_game_ids(cached_players).isin(remove_ids)
            ].copy()
        replacement_team = prepared.team_rows.loc[
            _frame_game_ids(prepared.team_rows).isin(changed_ids)
        ].copy()
        replacement_players = prepared.player_rows.loc[
            _frame_game_ids(prepared.player_rows).isin(changed_ids)
        ].copy()
        team = pd.concat([cached_team, replacement_team], ignore_index=True, sort=False)
        players = pd.concat(
            [cached_players, replacement_players], ignore_index=True, sort=False
        )
        team = team.sort_values("date").drop_duplicates(["gameid", "side"], keep="last")
        players = players.sort_values("date").drop_duplicates(
            ["gameid", "side", "position"], keep="last"
        )
        _atomic_parquet(team, team_path)
        _atomic_parquet(players, player_path)
    else:
        team = cached_team
        players = cached_players

    def year_ids(frame: pd.DataFrame) -> set[str]:
        if "oe_year" in frame.columns:
            frame = frame.loc[
                pd.to_numeric(frame["oe_year"], errors="coerce").eq(prepared.year)
            ]
        return set(_frame_game_ids(frame).dropna().astype(str))

    final_team_ids = year_ids(team)
    final_player_ids = year_ids(players)
    if final_team_ids != source_ids or final_player_ids != source_ids:
        raise OeDatabaseError(
            "local Parquet cache identity does not match accepted Supabase games"
        )
    identity_digest = _game_identity_sha256(source_ids)

    years = dict(years)
    years[str(prepared.year)] = {
        "source_file_sha256": prepared.source["raw_sha256"],
        "source_observed_through": prepared.source["date_max_utc"],
        "game_hashes": current_hashes,
        "canonical_game_identity_digest": identity_digest,
    }
    _atomic_json(
        state_path,
        {
            "schema_version": STATE_SCHEMA,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "years": years,
        },
    )
    previous_meta = _load_json(meta_path)
    source_files = [
        item
        for item in previous_meta.get("source_files", [])
        if isinstance(item, dict) and item.get("year") != prepared.year
    ]
    source_files.append(
        {
            "locator": str(prepared.csv_path),
            "year": prepared.year,
            **prepared.source,
        }
    )
    _atomic_json(
        meta_path,
        {
            "schema_version": "scryglass:oe-normalized-cache:v3",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "n_team_rows": int(len(team)),
            "n_player_rows": int(len(players)),
            "n_team_cols": int(len(team.columns)),
            "n_player_cols": int(len(players.columns)),
            "n_games": int(_frame_game_ids(team).nunique()),
            "source_files": sorted(source_files, key=lambda item: int(item.get("year") or 0)),
            "schema": "full_oe_incremental",
            "team_columns": list(team.columns),
            "player_columns": list(players.columns),
        },
    )
    return {
        "replaced_games": len(changed_ids),
        "removed_malformed_games": len(stale_ids),
        "team_rows": int(len(team)),
        "player_rows": int(len(players)),
        "canonical_game_identity_digest": identity_digest,
    }


def sync_csv(
    csv_path: Path,
    year: int,
    *,
    project_url: str,
    secret_key: str,
    parquet_dir: Path,
    client: SupabaseOeDatabase | Any | None = None,
    source_receipt: Path | None = None,
    patch_receipt_catalog: Path | None = None,
) -> dict[str, Any]:
    database = client or SupabaseOeDatabase(project_url, secret_key)
    path = csv_path.expanduser().resolve()
    source = _validate_oe_csv(path, str(year))
    accepted_source = (
        validate_accepted_source_receipt(source_receipt, path, year)
        if source_receipt is not None
        else None
    )
    try:
        patch_receipts = (
            load_patch_receipts(patch_receipt_catalog)
            if patch_receipt_catalog is not None
            else None
        )
    except RiotPatchReceiptError as exc:
        raise OeDatabaseError("Riot patch receipt catalog is invalid") from exc
    receipt = database.import_receipt(
        year, str(source["raw_sha256"]), TRANSFORM_VERSION
    )
    if (
        receipt is not None
        and patch_receipt_catalog is None
        and _cache_is_current(parquet_dir, year, str(source["raw_sha256"]))
    ):
        return {
            **_current_result_from_receipt(receipt, source, parquet_dir),
            "accepted_source_receipt": accepted_source,
            "riot_patch_receipts": 0,
            "worker_commit": os.environ.get("SCRYGLASS_WORKER_COMMIT") or None,
        }
    prepared = prepare_import(path, year, source=source, patch_receipts=patch_receipts)
    existing = database.current_hashes(year)
    accepted_hashes = {
        game_id: game.payload_sha256 for game_id, game in prepared.games.items()
    }
    missing_existing = sorted(
        set(existing).difference(accepted_hashes).difference(REVIEWED_REMOVED_GAME_IDS)
    )
    if missing_existing:
        preview = ", ".join(missing_existing[:5])
        raise OeDatabaseError(
            f"OE source lost {len(missing_existing)} stored games; first: {preview}"
        )
    new_ids = sorted(set(accepted_hashes).difference(existing))
    corrected_ids = sorted(
        game_id
        for game_id in set(accepted_hashes).intersection(existing)
        if accepted_hashes[game_id] != existing[game_id]
    )
    unchanged_ids = sorted(
        game_id
        for game_id in set(accepted_hashes).intersection(existing)
        if accepted_hashes[game_id] == existing[game_id]
    )
    changed_ids = [*new_ids, *corrected_ids]
    if changed_ids:
        # Build rows once via the precomputed payloads (no per-game to_json and
        # no re-serialization per upload); the payload_sha256 was computed
        # during prepare from these exact row dicts.
        version_rows_by_game: dict[str, dict[str, Any]] = {}
        for game_id in changed_ids:
            game = prepared.games[game_id]
            team_rows, player_rows = prepared.payload_rows_for(game_id)
            version_rows_by_game[game_id] = {
                "canonical_game_id": game_id,
                "payload_sha256": game.payload_sha256,
                "source_year": game.source_year,
                "game_date": game.game_date,
                "league": game.league,
                "patch": game.patch,
                "statistics_complete": game.statistics_complete,
                "source_file_sha256": game.source_file_sha256,
                "payload": {
                    "schema_version": GAME_SCHEMA,
                    "canonical_game_id": game_id,
                    "team_rows": team_rows,
                    "player_rows": player_rows,
                },
            }
        current_rows_by_game: dict[str, dict[str, Any]] = {
            game_id: prepared.games[game_id].current_row() for game_id in changed_ids
        }

        batches = list(_batches(changed_ids))
        if len(batches) <= 1:
            for game_id_batch in batches:
                database.append_versions(
                    [version_rows_by_game[game_id] for game_id in game_id_batch]
                )
            for game_id_batch in batches:
                database.upsert_current(
                    [current_rows_by_game[game_id] for game_id in game_id_batch]
                )
        else:
            # Phase 1: every immutable version insert must succeed before any
            # current pointer advances (the previous two-phase contract).
            with ThreadPoolExecutor(max_workers=WRITE_CONCURRENCY) as executor:
                list(executor.map(
                    lambda batch: database.append_versions(
                        [version_rows_by_game[game_id] for game_id in batch]
                    ),
                    batches,
                ))
            with ThreadPoolExecutor(max_workers=WRITE_CONCURRENCY) as executor:
                list(executor.map(
                    lambda batch: database.upsert_current(
                        [current_rows_by_game[game_id] for game_id in batch]
                    ),
                    batches,
                ))
    readback = database.current_hashes(year)
    mismatched = sorted(
        game_id
        for game_id, digest in accepted_hashes.items()
        if readback.get(game_id) != digest
    )
    if mismatched:
        raise OeDatabaseError(
            f"Supabase OE readback failed for {len(mismatched)} games; first: {mismatched[0]}"
        )
    cache = update_local_cache(prepared, parquet_dir)
    patch_receipt_count = len(
        {
            str(receipt.get("receipt_canonical_sha256"))
            for receipt in (patch_receipts or {}).values()
            if isinstance(receipt, Mapping)
        }
    )
    import_row = {
        "source_year": year,
        "source_file_sha256": prepared.source["raw_sha256"],
        "transform_version": TRANSFORM_VERSION,
        "source_bytes": prepared.source["bytes"],
        "source_rows": prepared.source["row_count"],
        "source_games": len(prepared.source_game_ids),
        "accepted_games": len(prepared.games),
        "new_games": len(new_ids),
        "corrected_games": len(corrected_ids),
        "unchanged_games": len(unchanged_ids),
        "quarantined_game_ids": list(prepared.quarantined_game_ids),
        "quarantined_games": prepared.quarantined_games,
        "statistics_complete_games": sum(
            game.statistics_complete for game in prepared.games.values()
        ),
        "source_observed_through": prepared.source["date_max_utc"],
        "riot_patch_receipts": patch_receipt_count,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    database.record_import(import_row)
    return {
        "schema_version": IMPORT_SCHEMA,
        "source_year": year,
        "source_file_sha256": prepared.source["raw_sha256"],
        "source_observed_through": prepared.source["date_max_utc"],
        "source_games": len(prepared.source_game_ids),
        "accepted_games": len(prepared.games),
        "new_games": len(new_ids),
        "corrected_games": len(corrected_ids),
        "unchanged_games": len(unchanged_ids),
        "quarantined_games": len(prepared.quarantined_game_ids),
        "statistics_complete_games": sum(
            game.statistics_complete for game in prepared.games.values()
        ),
        "cache": cache,
        "status": "updated" if changed_ids else "current",
        "accepted_source_receipt": accepted_source,
        "riot_patch_receipts": patch_receipt_count,
        "worker_commit": os.environ.get("SCRYGLASS_WORKER_COMMIT") or None,
    }


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise OeDatabaseError(f"required environment variable is missing: {name}")
    return value


def validate_import_receipt(
    import_receipt: Path,
    source_receipt: Path,
    csv_path: Path,
    year: int,
) -> dict[str, Any]:
    accepted_source = validate_accepted_source_receipt(source_receipt, csv_path, year)
    try:
        payload = json.loads(import_receipt.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OeDatabaseError("accepted import receipt is invalid") from error
    if not isinstance(payload, dict):
        raise OeDatabaseError("accepted import receipt is not an object")
    nested = payload.get("accepted_source_receipt")
    if (
        payload.get("source_file_sha256") != accepted_source["raw_sha256"]
        or not isinstance(nested, dict)
        or nested.get("receipt_canonical_sha256")
        != accepted_source["receipt_canonical_sha256"]
    ):
        raise OeDatabaseError("accepted import receipt does not match the source receipt")
    expected_commit = os.environ.get("SCRYGLASS_WORKER_COMMIT", "").strip()
    if expected_commit and payload.get("worker_commit") != expected_commit:
        raise OeDatabaseError("accepted import receipt belongs to a different worker commit")
    return payload


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument(
        "--patch-receipts",
        type=Path,
        help="Optional Riot official-feed receipt catalog for OE game IDs.",
    )
    parser.add_argument("--result-output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=Path("data/lol/warehouse/parquet"),
    )
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    if arguments.validate_only:
        if arguments.result_output is None:
            parser.error("--result-output is required with --validate-only")
        result = validate_import_receipt(
            arguments.result_output,
            arguments.source_receipt,
            arguments.csv,
            arguments.year,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    result = sync_csv(
        arguments.csv,
        arguments.year,
        project_url=_required_environment("SCRYGLASS_SUPABASE_URL"),
        secret_key=_required_environment("SCRYGLASS_SUPABASE_SECRET_KEY"),
        parquet_dir=arguments.parquet_dir.resolve(),
        source_receipt=arguments.source_receipt,
        patch_receipt_catalog=arguments.patch_receipts,
    )
    if arguments.result_output is not None:
        _atomic_json(arguments.result_output.expanduser().resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
