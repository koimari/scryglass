"""Pinned, outcome-isolated multi-league Player Rating development input.

This module is intentionally a development adapter, not rating authority.  It
opens two independently pinned Oracle's Elixir warehouse files and exposes:

* exact observed ten-player lineups for LCS, LEC, LCK, LPL, MSI, and EWC;
* series-atomic TRAIN / DEVELOPMENT / VALIDATION observations before the
  sealed-final boundary; and
* outcome-free metadata for the sealed period.

The warehouse does not contain an authoritative series identifier for every
league.  LPL ``bmid`` values are used where their redundant URL/game identity
agrees.  All other series IDs are conservative dependence clusters and are
labelled as such.  Invalid maps quarantine their complete cluster; identities,
lineups, counters, or folds are never repaired or filled.

Raw SHA-256 values are mandatory arguments.  Code-held or self-reported
digests would turn file identity into self-authorization, so this module has
no accepted-snapshot constants and no default pins.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MAPS_LOCATOR = "data/lol/warehouse/parquet/maps.parquet"
DEFAULT_PLAYERS_LOCATOR = "data/lol/warehouse/parquet/players.parquet"
SCHEMA_VERSION = "scryglass:multileague-player-development-input:v1"

LEAGUES = ("LCS", "LEC", "LCK", "LPL", "MSI", "EWC")
DOMESTIC_LEAGUES = ("LCS", "LEC", "LCK", "LPL")
INTERNATIONAL_LEAGUES = ("MSI", "EWC")
ROLE_ORDER = ("top", "jungle", "mid", "bot", "support")
SOURCE_ROLE = {
    "top": "top",
    "jng": "jungle",
    "mid": "mid",
    "bot": "bot",
    "sup": "support",
}

DATA_START = pd.Timestamp("2025-01-01T00:00:00")
DEVELOPMENT_START = pd.Timestamp("2025-07-01T00:00:00")
VALIDATION_START = pd.Timestamp("2026-01-01T00:00:00")
SEALED_FINAL_START = pd.Timestamp("2026-04-01T00:00:00")
DERIVED_SERIES_MAX_GAP_HOURS = 8.0
MISSING_SOURCE_SPLIT = "__MISSING_SOURCE_SPLIT__"

MAP_METADATA_COLUMNS = (
    "game_uid",
    "oe_gameid",
    "url",
    "league",
    "year",
    "split",
    "playoffs",
    "date",
    "game",
    "patch",
    "competition_scope",
    "event_kind",
    "is_international",
    "blue_team_key",
    "red_team_key",
    "blue_team",
    "red_team",
    "blue_teamid",
    "red_teamid",
    "source_lp",
    "lp_matched",
    "lp_game_id",
)
MAP_DEVELOPMENT_COLUMNS = MAP_METADATA_COLUMNS + ("y_blue_win",)
PLAYER_METADATA_COLUMNS = (
    "game_uid",
    "league",
    "date",
    "game",
    "side",
    "position",
    "playername",
    "playerid",
    "teamname",
    "teamid",
    "team_key",
)

FORBIDDEN_SEALED_COLUMNS = frozenset(
    {
        "y_blue_win",
        "blue_result",
        "red_result",
        "result",
        "kills",
        "deaths",
        "assists",
        "total_kills",
    }
)

CLAIM_CEILING: Mapping[str, bool] = {
    "private_retrospective_model_fit": True,
    "private_rank_selection": True,
    "historical_observed_lineup": True,
    "authoritative_series_identity_all_leagues": False,
    "pre_event_roster_authority": False,
    "sealed_final_targets_accessed": False,
    "prediction": False,
    "production": False,
    "publication": False,
    "promotion": False,
    "probability": False,
    "odds": False,
    "expected_value": False,
    "bet_recommendation": False,
}


class MultiLeagueDevelopmentError(ValueError):
    """The pinned development input violated an identity or isolation rule."""


@dataclass(frozen=True)
class WarehousePins:
    maps_sha256: str
    players_sha256: str


@dataclass(frozen=True)
class PlayerSlot:
    role: str
    player_id: str
    player_name: str
    team_id: str


@dataclass(frozen=True)
class ObservedLineup:
    side: str
    team_id: str
    team_key: str
    team_name: str
    players: tuple[PlayerSlot, ...]


@dataclass(frozen=True)
class DevelopmentMap:
    game_id: str
    series_id: str
    series_identity_kind: str
    fold_id: str
    league: str
    source_local_start: str
    game_number: int
    patch_token: str
    blue_lineup: ObservedLineup
    red_lineup: ObservedLineup
    blue_win: int


@dataclass(frozen=True)
class SealedMapMetadata:
    """A sealed-period map with no outcome field by construction."""

    game_id: str
    series_id: str
    series_identity_kind: str
    league: str
    source_local_start: str
    game_number: int
    patch_token: str
    blue_lineup: ObservedLineup
    red_lineup: ObservedLineup


@dataclass(frozen=True)
class DevelopmentSeries:
    series_id: str
    series_identity_kind: str
    fold_id: str
    league: str
    source_local_start: str
    source_local_end: str
    maps: tuple[DevelopmentMap, ...]


@dataclass(frozen=True)
class SealedSeriesMetadata:
    series_id: str
    series_identity_kind: str
    league: str
    source_local_start: str
    source_local_end: str
    maps: tuple[SealedMapMetadata, ...]


@dataclass(frozen=True)
class QuarantinedCluster:
    cluster_id: str
    game_ids: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PrivateMultiLeagueRatingInput:
    schema_version: str
    maps_locator: str
    players_locator: str
    maps_sha256: str
    players_sha256: str
    development_selected_rows_sha256: str
    sealed_selected_metadata_sha256: str
    player_selected_metadata_sha256: str
    cluster_partition_sha256: str
    development_series: tuple[DevelopmentSeries, ...]
    sealed_series_metadata: tuple[SealedSeriesMetadata, ...]
    quarantined_clusters: tuple[QuarantinedCluster, ...]
    coverage: Mapping[str, Any]
    claim_ceiling: Mapping[str, bool]


@dataclass(frozen=True)
class _MapMetadata:
    game_id: str
    url: str | None
    league: str
    year: int
    split: str
    playoffs: bool
    at: pd.Timestamp
    game_number: int
    patch_token: str
    competition_scope: str
    event_kind: str
    is_international: bool
    blue_team_key: str
    red_team_key: str
    blue_team_name: str
    red_team_name: str
    blue_team_id: str
    red_team_id: str
    source_lp: bool
    lp_matched: bool
    lp_game_id: str | None
    blue_win: int | None


@dataclass(frozen=True)
class _Cluster:
    cluster_id: str
    identity_kind: str
    members: tuple[_MapMetadata, ...]


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise MultiLeagueDevelopmentError("canonical input contains a non-finite value") from error


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MultiLeagueDevelopmentError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_repo_file(root: Path, locator: str) -> Path:
    if not isinstance(locator, str) or "\\" in locator:
        raise MultiLeagueDevelopmentError("warehouse locator must be a relative POSIX path")
    relative = PurePosixPath(locator)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or locator != "/".join(relative.parts)
    ):
        raise MultiLeagueDevelopmentError("warehouse locator must be a contained canonical relative POSIX path")
    try:
        root_real = root.resolve(strict=True)
    except FileNotFoundError as error:
        raise MultiLeagueDevelopmentError("warehouse root is missing") from error
    current = root_real
    metadata: os.stat_result | None = None
    for index, part in enumerate(relative.parts):
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as error:
            raise MultiLeagueDevelopmentError(f"warehouse artifact is missing: {locator}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise MultiLeagueDevelopmentError(f"warehouse artifact path contains a symlink: {locator}")
        if index < len(relative.parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise MultiLeagueDevelopmentError(f"warehouse artifact parent is not a directory: {locator}")
    assert metadata is not None
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise MultiLeagueDevelopmentError(
            f"warehouse artifact must be an unaliased regular file: {locator}"
        )
    if current.resolve(strict=True).parent != (root_real / relative.parent).resolve(strict=True):
        raise MultiLeagueDevelopmentError(f"warehouse artifact path alias rejected: {locator}")
    return current


def _present(value: Any) -> bool:
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def _text(value: Any, label: str) -> str:
    if not _present(value):
        raise MultiLeagueDevelopmentError(f"{label} is missing")
    return str(value).strip()


def _optional_text(value: Any) -> str | None:
    return str(value).strip() if _present(value) else None


def _integer(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not _present(value):
        raise MultiLeagueDevelopmentError(f"{label} is missing")
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise MultiLeagueDevelopmentError(f"{label} must be integral")
    return int(number)


def _boolean(value: Any, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    raise MultiLeagueDevelopmentError(f"{label} must be boolean")


def _timestamp(value: Any) -> pd.Timestamp:
    if not _present(value):
        raise MultiLeagueDevelopmentError("map timestamp is missing")
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise MultiLeagueDevelopmentError("map timestamp is invalid")
    if result.tzinfo is not None:
        raise MultiLeagueDevelopmentError(
            "warehouse timestamp unexpectedly asserts a timezone; source-naive policy changed"
        )
    return result


def _patch_token(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)) or not _present(value):
        raise MultiLeagueDevelopmentError("patch is missing")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise MultiLeagueDevelopmentError("patch is invalid")
    centesimal = round(number * 100.0)
    if abs(number * 100.0 - centesimal) > 1e-8:
        raise MultiLeagueDevelopmentError("patch is not an exact centesimal source token")
    return f"{centesimal / 100.0:.2f}"


def _fold(at: pd.Timestamp) -> str:
    if at < DATA_START:
        raise MultiLeagueDevelopmentError("map predates the accepted population")
    if at < DEVELOPMENT_START:
        return "TRAIN"
    if at < VALIDATION_START:
        return "DEVELOPMENT"
    if at < SEALED_FINAL_START:
        return "VALIDATION"
    return "SEALED_FINAL"


def _frame_payload(frame: pd.DataFrame) -> dict[str, Any]:
    rows: list[list[Any]] = []
    for raw in frame.itertuples(index=False, name=None):
        row: list[Any] = []
        for value in raw:
            if value is None or pd.isna(value):
                row.append(None)
            elif isinstance(value, pd.Timestamp):
                row.append(value.isoformat())
            elif isinstance(value, np.generic):
                row.append(value.item())
            else:
                row.append(value)
        rows.append(row)
    rows.sort(key=_canonical_bytes)
    return {"columns": list(frame.columns), "rows": rows}


def _read_exact(
    reader: Callable[..., pd.DataFrame],
    path: Path,
    columns: Sequence[str],
    filters: Sequence[tuple[str, str, Any]],
) -> pd.DataFrame:
    frame = reader(path, columns=list(columns), filters=list(filters), engine="pyarrow")
    if not isinstance(frame, pd.DataFrame) or tuple(frame.columns) != tuple(columns):
        raise MultiLeagueDevelopmentError("Parquet reader did not preserve the exact projected column allowlist")
    return frame


def _map_metadata(row: Mapping[str, Any], *, outcome_accessed: bool) -> _MapMetadata:
    game_id = _text(row.get("game_uid"), "game_uid")
    if _text(row.get("oe_gameid"), "oe_gameid") != game_id:
        raise MultiLeagueDevelopmentError("ambiguous map identity")
    league = _text(row.get("league"), "league").upper()
    if league not in LEAGUES:
        raise MultiLeagueDevelopmentError("map is outside the selected league population")
    at = _timestamp(row.get("date"))
    fold = _fold(at)
    if outcome_accessed != (fold != "SEALED_FINAL"):
        raise MultiLeagueDevelopmentError("target projection crossed the sealed-final boundary")
    game_number = _integer(row.get("game"), "source game counter")
    if game_number < 1 or game_number > 5:
        raise MultiLeagueDevelopmentError("source game counter is outside 1..5")
    blue_win: int | None = None
    if outcome_accessed:
        blue_win = _integer(row.get("y_blue_win"), "blue-win target")
        if blue_win not in (0, 1):
            raise MultiLeagueDevelopmentError("blue-win target must be binary")
    return _MapMetadata(
        game_id=game_id,
        url=_optional_text(row.get("url")),
        league=league,
        year=_integer(row.get("year"), "year"),
        # OE omits the split label for some tournament phases.  Missingness is
        # retained as an explicit source token; it is neither inferred nor
        # treated as a reason to discard an otherwise identifiable series.
        split=_optional_text(row.get("split")) or MISSING_SOURCE_SPLIT,
        playoffs=_boolean(row.get("playoffs"), "playoffs"),
        at=at,
        game_number=game_number,
        patch_token=_patch_token(row.get("patch")),
        competition_scope=_text(row.get("competition_scope"), "competition_scope"),
        event_kind=_text(row.get("event_kind"), "event_kind"),
        is_international=_boolean(row.get("is_international"), "is_international"),
        blue_team_key=_text(row.get("blue_team_key"), "blue_team_key"),
        red_team_key=_text(row.get("red_team_key"), "red_team_key"),
        blue_team_name=_text(row.get("blue_team"), "blue team name"),
        red_team_name=_text(row.get("red_team"), "red team name"),
        blue_team_id=_text(row.get("blue_teamid"), "blue team id"),
        red_team_id=_text(row.get("red_teamid"), "red team id"),
        source_lp=_boolean(row.get("source_lp"), "source_lp"),
        lp_matched=_boolean(row.get("lp_matched"), "lp_matched"),
        lp_game_id=_optional_text(row.get("lp_game_id")),
        blue_win=blue_win,
    )


def _map_context(item: _MapMetadata) -> tuple[Any, ...]:
    teams = tuple(sorted((item.blue_team_id, item.red_team_id)))
    return (
        item.league,
        item.year,
        item.split,
        item.playoffs,
        item.competition_scope,
        item.event_kind,
        item.is_international,
        teams[0],
        teams[1],
    )


def _lpl_series_id(item: _MapMetadata) -> str:
    match = re.search(r"(?:[?&])bmid=(\d+)(?:&|$)", item.url or "")
    identity = re.fullmatch(r"(\d+)-\1_game_([1-5])", item.game_id)
    if (
        match is None
        or identity is None
        or match.group(1) != identity.group(1)
        or int(identity.group(2)) != item.game_number
    ):
        raise MultiLeagueDevelopmentError("LPL bmid and map identity disagree")
    return f"lpl-bmid:{match.group(1)}"


def _derived_cluster_id(context: tuple[Any, ...], first: _MapMetadata) -> str:
    digest = _canonical_sha256(
        {
            "context": list(context),
            "first_game_id": first.game_id,
            "first_source_local_start": first.at.isoformat(),
        }
    )
    return f"derived-dependence-cluster:{digest[:24]}"


def _cluster_maps(
    maps: Sequence[_MapMetadata],
) -> tuple[list[_Cluster], dict[str, set[str]], dict[str, set[str]]]:
    """Return candidate clusters plus map- and cluster-level invalidity."""

    by_id: dict[str, list[_MapMetadata]] = defaultdict(list)
    for item in maps:
        by_id[item.game_id].append(item)
    invalid_maps: dict[str, set[str]] = defaultdict(set)
    for game_id, members in by_id.items():
        if len(members) != 1:
            invalid_maps[game_id].add("duplicate_map_identity")

    collision: dict[tuple[Any, ...], list[_MapMetadata]] = defaultdict(list)
    for item in maps:
        collision[_map_context(item) + (item.at.isoformat(), item.game_number)].append(item)
    for members in collision.values():
        if len(members) > 1:
            for item in members:
                invalid_maps[item.game_id].add("exact_context_time_counter_collision")

    lpl: dict[str, list[_MapMetadata]] = defaultdict(list)
    derived_contexts: dict[tuple[Any, ...], list[_MapMetadata]] = defaultdict(list)
    for item in maps:
        if item.game_id in invalid_maps:
            continue
        if item.league == "LPL":
            try:
                lpl[_lpl_series_id(item)].append(item)
            except MultiLeagueDevelopmentError:
                invalid_maps[item.game_id].add("invalid_lpl_bmid_identity")
        else:
            derived_contexts[_map_context(item)].append(item)

    clusters: list[_Cluster] = []
    for cluster_id, members in lpl.items():
        clusters.append(
            _Cluster(
                cluster_id=cluster_id,
                identity_kind="SOURCE_LPL_BMID",
                members=tuple(sorted(members, key=lambda item: (item.at, item.game_id))),
            )
        )
    for context in sorted(derived_contexts, key=_canonical_bytes):
        ordered = sorted(derived_contexts[context], key=lambda item: (item.at, item.game_id))
        current: list[_MapMetadata] = []
        cluster_id = ""
        previous: _MapMetadata | None = None
        for item in ordered:
            gap = (
                math.inf
                if previous is None
                else (item.at - previous.at).total_seconds() / 3600.0
            )
            continues = (
                previous is not None
                and item.game_number == previous.game_number + 1
                and item.patch_token == previous.patch_token
                and 0.0 < gap <= DERIVED_SERIES_MAX_GAP_HOURS
            )
            if not continues:
                if current:
                    clusters.append(
                        _Cluster(cluster_id, "DERIVED_DEPENDENCE_CLUSTER", tuple(current))
                    )
                current = [item]
                cluster_id = _derived_cluster_id(context, item)
            else:
                current.append(item)
            previous = item
        if current:
            clusters.append(_Cluster(cluster_id, "DERIVED_DEPENDENCE_CLUSTER", tuple(current)))

    invalid_clusters: dict[str, set[str]] = defaultdict(set)
    for cluster in clusters:
        members = cluster.members
        counters = [item.game_number for item in members]
        if counters != list(range(1, len(members) + 1)):
            invalid_clusters[cluster.cluster_id].add("noncontiguous_or_missing_game_counter")
        if len(members) > 5:
            invalid_clusters[cluster.cluster_id].add("more_than_five_maps")
        if len({item.game_id for item in members}) != len(members):
            invalid_clusters[cluster.cluster_id].add("duplicate_map_within_cluster")
        if len({_fold(item.at) for item in members}) != 1:
            invalid_clusters[cluster.cluster_id].add("series_crosses_temporal_fold")
        if len({item.patch_token for item in members}) != 1:
            invalid_clusters[cluster.cluster_id].add("series_crosses_patch")
        span = (members[-1].at - members[0].at).total_seconds() / 3600.0
        if span > DERIVED_SERIES_MAX_GAP_HOURS:
            invalid_clusters[cluster.cluster_id].add("series_span_exceeds_eight_hours")

    return clusters, invalid_maps, invalid_clusters


def _lineups(
    players: pd.DataFrame,
    maps: Mapping[str, _MapMetadata],
    selected_game_ids: set[str],
) -> tuple[dict[str, tuple[ObservedLineup, ObservedLineup]], dict[str, set[str]]]:
    lineups: dict[str, tuple[ObservedLineup, ObservedLineup]] = {}
    invalid: dict[str, set[str]] = defaultdict(set)
    player_game_counts = Counter(str(value).strip() for value in players["game_uid"] if _present(value))
    for game_id, group in players.groupby("game_uid", sort=False, dropna=False):
        identity = str(game_id).strip() if _present(game_id) else "<missing>"
        if identity not in maps:
            # Rows for a selected map that already failed map identity checks
            # are not orphans; their cluster is already quarantined.  Only a
            # player-game identity absent from the selected map population is
            # an actual cross-file orphan.
            if identity not in selected_game_ids:
                invalid[identity].add("orphan_player_rows")
            continue
        item = maps[identity]
        if player_game_counts[identity] != 10 or len(group) != 10:
            invalid[identity].add("lineup_not_exactly_ten_rows")
            continue
        built: list[ObservedLineup] = []
        all_player_ids: list[str] = []
        for side, team_id, team_key, team_name in (
            ("blue", item.blue_team_id, item.blue_team_key, item.blue_team_name),
            ("red", item.red_team_id, item.red_team_key, item.red_team_name),
        ):
            side_rows = group[group["side"].astype(str).str.lower().eq(side)]
            canonical: dict[str, PlayerSlot] = {}
            if len(side_rows) != 5:
                invalid[identity].add(f"{side}_lineup_not_five_rows")
                continue
            for row in side_rows.to_dict("records"):
                source_role = str(row.get("position") or "").strip().lower()
                role = SOURCE_ROLE.get(source_role)
                if role is None or role in canonical:
                    invalid[identity].add(f"{side}_role_closure_invalid")
                    continue
                try:
                    player_id = _text(row.get("playerid"), "player id")
                    player_name = _text(row.get("playername"), "player name")
                    row_team_id = _text(row.get("teamid"), "player-row team id")
                except MultiLeagueDevelopmentError:
                    invalid[identity].add(f"{side}_stable_player_or_team_identity_missing")
                    continue
                if row_team_id != team_id:
                    invalid[identity].add(f"{side}_player_team_identity_mismatch")
                    continue
                canonical[role] = PlayerSlot(role, player_id, player_name, team_id)
                all_player_ids.append(player_id)
            if set(canonical) != set(ROLE_ORDER):
                invalid[identity].add(f"{side}_role_closure_invalid")
                continue
            built.append(
                ObservedLineup(
                    side=side,
                    team_id=team_id,
                    team_key=team_key,
                    team_name=team_name,
                    players=tuple(canonical[role] for role in ROLE_ORDER),
                )
            )
        if len(built) != 2 or len(set(all_player_ids)) != 10:
            invalid[identity].add("lineup_player_identity_not_globally_distinct")
            continue
        if identity not in invalid:
            lineups[identity] = (built[0], built[1])

    for game_id in maps:
        if game_id not in lineups and game_id not in invalid:
            invalid[game_id].add("player_rows_missing")
    return lineups, invalid


def _development_map(
    item: _MapMetadata,
    cluster: _Cluster,
    lineup: tuple[ObservedLineup, ObservedLineup],
) -> DevelopmentMap:
    if item.blue_win not in (0, 1):
        raise MultiLeagueDevelopmentError("development map lost its binary target")
    return DevelopmentMap(
        game_id=item.game_id,
        series_id=cluster.cluster_id,
        series_identity_kind=cluster.identity_kind,
        fold_id=_fold(item.at),
        league=item.league,
        source_local_start=item.at.isoformat(),
        game_number=item.game_number,
        patch_token=item.patch_token,
        blue_lineup=lineup[0],
        red_lineup=lineup[1],
        blue_win=item.blue_win,
    )


def _sealed_map(
    item: _MapMetadata,
    cluster: _Cluster,
    lineup: tuple[ObservedLineup, ObservedLineup],
) -> SealedMapMetadata:
    if item.blue_win is not None:
        raise MultiLeagueDevelopmentError("sealed metadata unexpectedly contains a target")
    return SealedMapMetadata(
        game_id=item.game_id,
        series_id=cluster.cluster_id,
        series_identity_kind=cluster.identity_kind,
        league=item.league,
        source_local_start=item.at.isoformat(),
        game_number=item.game_number,
        patch_token=item.patch_token,
        blue_lineup=lineup[0],
        red_lineup=lineup[1],
    )


def _coverage(
    development: Sequence[DevelopmentSeries],
    sealed: Sequence[SealedSeriesMetadata],
    quarantine: Sequence[QuarantinedCluster],
    selected_maps: int,
) -> dict[str, Any]:
    folds = ("TRAIN", "DEVELOPMENT", "VALIDATION")
    fold_rows = []
    for fold in folds:
        values = [series for series in development if series.fold_id == fold]
        fold_rows.append(
            {
                "fold_id": fold,
                "series": len(values),
                "maps": sum(len(series.maps) for series in values),
                "by_league": [
                    {
                        "league": league,
                        "series": sum(series.league == league for series in values),
                        "maps": sum(
                            len(series.maps) for series in values if series.league == league
                        ),
                    }
                    for league in LEAGUES
                ],
            }
        )
    accepted_maps = sum(len(series.maps) for series in development) + sum(
        len(series.maps) for series in sealed
    )
    quarantined_ids = {game_id for value in quarantine for game_id in value.game_ids}
    if accepted_maps + len(quarantined_ids) != selected_maps:
        raise MultiLeagueDevelopmentError("selected map coverage does not reconcile")
    return {
        "population": {
            "leagues": list(LEAGUES),
            "start_inclusive": DATA_START.isoformat(),
            "sealed_final_start_inclusive": SEALED_FINAL_START.isoformat(),
            "source_time_semantics": "timezone-naive warehouse timestamp",
        },
        "selected_maps": selected_maps,
        "accepted_maps": accepted_maps,
        "development_maps": sum(len(series.maps) for series in development),
        "sealed_metadata_maps": sum(len(series.maps) for series in sealed),
        "quarantined_maps": len(quarantined_ids),
        "development_series": len(development),
        "sealed_metadata_series": len(sealed),
        "quarantined_clusters": len(quarantine),
        "folds": fold_rows,
    }


def load_multileague_development_input(
    *,
    expected_maps_sha256: str,
    expected_players_sha256: str,
    root: Path = ROOT,
    maps_locator: str = DEFAULT_MAPS_LOCATOR,
    players_locator: str = DEFAULT_PLAYERS_LOCATOR,
    parquet_reader: Callable[..., pd.DataFrame] = pd.read_parquet,
) -> PrivateMultiLeagueRatingInput:
    """Load an externally pinned development input without reading final targets."""

    pins = WarehousePins(
        maps_sha256=_require_sha256(expected_maps_sha256, "expected_maps_sha256"),
        players_sha256=_require_sha256(expected_players_sha256, "expected_players_sha256"),
    )
    maps_path = _safe_repo_file(root, maps_locator)
    players_path = _safe_repo_file(root, players_locator)
    if _raw_sha256(maps_path) != pins.maps_sha256:
        raise MultiLeagueDevelopmentError("maps Parquet does not match the independent expected digest")
    if _raw_sha256(players_path) != pins.players_sha256:
        raise MultiLeagueDevelopmentError("players Parquet does not match the independent expected digest")

    common_filters = [("league", "in", list(LEAGUES)), ("date", ">=", DATA_START)]
    development_frame = _read_exact(
        parquet_reader,
        maps_path,
        MAP_DEVELOPMENT_COLUMNS,
        [*common_filters, ("date", "<", SEALED_FINAL_START)],
    )
    if FORBIDDEN_SEALED_COLUMNS.intersection(MAP_METADATA_COLUMNS):
        raise MultiLeagueDevelopmentError("sealed metadata projection includes a target column")
    sealed_frame = _read_exact(
        parquet_reader,
        maps_path,
        MAP_METADATA_COLUMNS,
        [*common_filters, ("date", ">=", SEALED_FINAL_START)],
    )
    player_frame = _read_exact(
        parquet_reader,
        players_path,
        PLAYER_METADATA_COLUMNS,
        common_filters,
    )
    all_selected_ids = {
        str(value).strip()
        for frame in (development_frame, sealed_frame)
        for value in frame["game_uid"]
        if _present(value)
    }

    records: list[_MapMetadata] = []
    row_errors: dict[str, set[str]] = defaultdict(set)
    for outcome_accessed, frame in ((True, development_frame), (False, sealed_frame)):
        for row in frame.to_dict("records"):
            candidate = str(row.get("game_uid") or "<missing>").strip()
            try:
                records.append(_map_metadata(row, outcome_accessed=outcome_accessed))
            except (MultiLeagueDevelopmentError, TypeError, ValueError):
                row_errors[candidate].add("invalid_map_metadata_or_target")
    if not records:
        raise MultiLeagueDevelopmentError("selected multi-league map population is empty")

    clusters, invalid_maps, invalid_clusters = _cluster_maps(records)
    for game_id, reasons in row_errors.items():
        invalid_maps[game_id].update(reasons)
    maps_by_id = {item.game_id: item for item in records if item.game_id not in invalid_maps}
    lineups, invalid_lineups = _lineups(player_frame, maps_by_id, all_selected_ids)
    for game_id, reasons in invalid_lineups.items():
        invalid_maps[game_id].update(reasons)

    quarantine: list[QuarantinedCluster] = []
    development: list[DevelopmentSeries] = []
    sealed: list[SealedSeriesMetadata] = []
    clustered_ids: set[str] = set()
    for cluster in sorted(clusters, key=lambda item: (item.members[0].at, item.cluster_id)):
        member_ids = tuple(item.game_id for item in cluster.members)
        clustered_ids.update(member_ids)
        reasons = set(invalid_clusters.get(cluster.cluster_id, set()))
        for game_id in member_ids:
            reasons.update(invalid_maps.get(game_id, set()))
        if reasons:
            quarantine.append(
                QuarantinedCluster(cluster.cluster_id, tuple(sorted(member_ids)), tuple(sorted(reasons)))
            )
            continue
        fold = _fold(cluster.members[0].at)
        if any(_fold(item.at) != fold for item in cluster.members):
            raise MultiLeagueDevelopmentError("series fold atomicity drifted after validation")
        league = cluster.members[0].league
        if any(item.league != league for item in cluster.members):
            raise MultiLeagueDevelopmentError("series crosses leagues")
        start = cluster.members[0].at.isoformat()
        end = cluster.members[-1].at.isoformat()
        if fold == "SEALED_FINAL":
            maps = tuple(_sealed_map(item, cluster, lineups[item.game_id]) for item in cluster.members)
            sealed.append(
                SealedSeriesMetadata(
                    cluster.cluster_id,
                    cluster.identity_kind,
                    league,
                    start,
                    end,
                    maps,
                )
            )
        else:
            maps = tuple(
                _development_map(item, cluster, lineups[item.game_id]) for item in cluster.members
            )
            development.append(
                DevelopmentSeries(
                    cluster.cluster_id,
                    cluster.identity_kind,
                    fold,
                    league,
                    start,
                    end,
                    maps,
                )
            )

    # Maps rejected before a cluster identity existed remain explicit
    # singleton quarantines.  No target value is copied into the record.
    for game_id in sorted(all_selected_ids - clustered_ids):
        quarantine.append(
            QuarantinedCluster(
                f"unclustered-map:{game_id}",
                (game_id,),
                tuple(sorted(invalid_maps.get(game_id, {"series_identity_unavailable"}))),
            )
        )
    quarantine.sort(key=lambda item: (item.game_ids, item.cluster_id))

    development.sort(key=lambda item: (item.source_local_start, item.series_id))
    sealed.sort(key=lambda item: (item.source_local_start, item.series_id))
    coverage = _coverage(development, sealed, quarantine, len(all_selected_ids))
    partition = [
        {
            "series_id": item.series_id,
            "identity_kind": item.series_identity_kind,
            "fold_id": item.fold_id,
            "game_ids": [value.game_id for value in item.maps],
        }
        for item in development
    ] + [
        {
            "series_id": item.series_id,
            "identity_kind": item.series_identity_kind,
            "fold_id": "SEALED_FINAL",
            "game_ids": [value.game_id for value in item.maps],
        }
        for item in sealed
    ]

    return PrivateMultiLeagueRatingInput(
        schema_version=SCHEMA_VERSION,
        maps_locator=maps_locator,
        players_locator=players_locator,
        maps_sha256=pins.maps_sha256,
        players_sha256=pins.players_sha256,
        development_selected_rows_sha256=_canonical_sha256(_frame_payload(development_frame)),
        sealed_selected_metadata_sha256=_canonical_sha256(_frame_payload(sealed_frame)),
        player_selected_metadata_sha256=_canonical_sha256(_frame_payload(player_frame)),
        cluster_partition_sha256=_canonical_sha256(partition),
        development_series=tuple(development),
        sealed_series_metadata=tuple(sealed),
        quarantined_clusters=tuple(quarantine),
        coverage=coverage,
        claim_ceiling=dict(CLAIM_CEILING),
    )


def source_local_datetime(value: str) -> datetime:
    """Parse an adapter timestamp without asserting an unavailable timezone."""

    if not isinstance(value, str):
        raise MultiLeagueDevelopmentError("source-local timestamp must be a string")
    try:
        result = datetime.fromisoformat(value)
    except ValueError as error:
        raise MultiLeagueDevelopmentError("source-local timestamp is invalid") from error
    if result.tzinfo is not None:
        raise MultiLeagueDevelopmentError("source-local timestamp must remain timezone-naive")
    return result
