"""Private, fail-closed GRID market-data feasibility foundation.

This module is not a model, recommender, price calculator, or serving path.
It uses exact provider/Riot identities, already-local GRID Series Events, and
the smallest explicitly requested Riot LiveStats downloads to inventory and
validate timestamped checkpoint evidence and final labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import pandas as pd

from lol_kills.etl.grid_ingest import _api_key, _download, _file_list
from lol_kills.grid_live_foundation import (
    PROVISIONAL_MAX_STATE_AGE_MS,
    build_foundation_artifact,
    context_from_grid_games_parquet,
    write_immutable_receipt,
)


SCHEMA_VERSION = "scryglass.grid-market-foundation.v1"
RETRIEVAL_RECEIPT_SCHEMA = "scryglass.grid-market-retrieval-receipt.v1"
REPORT_SCHEMA = "scryglass.grid-market-feasibility-report.v1"
DEFAULT_CHECKPOINTS = (10, 15, 20, 25)
DEFAULT_ROOT = Path("data/lol/warehouse/private_grid/market_foundation/v1")
DEFAULT_GRID_RAW = Path("data/lol/warehouse/raw_grid")
DEFAULT_GAMES = Path("data/lol/warehouse/grid_drakes/games.parquet")
CATALOG_PATH = (
    Path.home()
    / ".codex"
    / "skills"
    / "query-grid-research"
    / "assets"
    / "grid-capability-catalog.v1.json"
)

TARGET_SPECS: tuple[dict[str, Any], ...] = (
    {
        "target": "first_tower",
        "kind": "binary_team",
        "riot_label": "first building_destroyed where buildingType=turret; taker is opposite teamID",
        "grid_verification": "destroyTower objective completedFirst",
        "checkpoint": "cumulative turret events and first taker at or before checkpoint",
        "market_semantics": "clear_event_definition",
    },
    {
        "target": "first_inhibitor",
        "kind": "binary_team_or_none",
        "riot_label": "first building_destroyed where buildingType=inhibitor; taker is opposite teamID",
        "grid_verification": "destroyFortifier objective completedFirst",
        "checkpoint": "cumulative inhibitor events and first taker at or before checkpoint",
        "market_semantics": "clear_event_definition",
    },
    {
        "target": "total_tower_destructions",
        "kind": "count",
        "riot_label": "count building_destroyed where buildingType=turret",
        "grid_verification": "sum destroyTower objective completionCount",
        "checkpoint": "cumulative turret events at or before checkpoint",
        "market_semantics": "external_market_rule_needed_for_respawned_nexus_turrets",
    },
    {
        "target": "total_inhibitor_destructions",
        "kind": "count",
        "riot_label": "count building_destroyed where buildingType=inhibitor",
        "grid_verification": "sum destroyFortifier objective completionCount",
        "checkpoint": "cumulative inhibitor events at or before checkpoint",
        "market_semantics": "destruction_events_include_repeat_destructions",
    },
    {
        "target": "first_dragon",
        "kind": "binary_team_or_none",
        "riot_label": "first epic_monster_kill where monsterType=dragon",
        "grid_verification": "slayDragon objective completedFirst",
        "checkpoint": "cumulative dragons and first taker at or before checkpoint",
        "market_semantics": "clear_event_definition",
    },
    {
        "target": "total_dragons",
        "kind": "count",
        "riot_label": "count epic_monster_kill where monsterType=dragon",
        "grid_verification": "sum slayDragon objective completionCount",
        "checkpoint": "cumulative dragons at or before checkpoint",
        "market_semantics": "clear_event_definition",
    },
    {
        "target": "first_baron",
        "kind": "binary_team_or_none",
        "riot_label": "first epic_monster_kill where monsterType=baron",
        "grid_verification": "slayBaron objective completedFirst",
        "checkpoint": "cumulative barons and first taker at or before checkpoint",
        "market_semantics": "clear_event_definition",
    },
    {
        "target": "total_barons",
        "kind": "count",
        "riot_label": "count epic_monster_kill where monsterType=baron",
        "grid_verification": "sum slayBaron objective completionCount",
        "checkpoint": "cumulative barons at or before checkpoint",
        "market_semantics": "clear_event_definition",
    },
    {
        "target": "first_blood",
        "kind": "binary_team",
        "riot_label": "first champion_kill killerTeamID",
        "grid_verification": "team firstKill",
        "checkpoint": "cumulative kills and first taker at or before checkpoint",
        "market_semantics": "clear_event_definition",
    },
    {
        "target": "unique_towers_destroyed",
        "kind": "count",
        "riot_label": "unique destroyed-team/lane/tier/nexus-name identities",
        "grid_verification": "no independent unique-structure count identified",
        "checkpoint": "unique turret identities at or before checkpoint",
        "market_semantics": "derived_diagnostic_not_a_verified_market_label",
    },
)


class GridMarketFoundationError(RuntimeError):
    """Credential-free failure in private market foundation processing."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_file_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in sorted(row.items())
        if not any(
            marker in str(key).lower()
            for marker in ("url", "token", "key", "secret", "signature")
        )
    }


def _known_series_context(
    games_path: Path, series_id: str
) -> tuple[str, dict[str, Any]]:
    frame = pd.read_parquet(games_path)
    required = {"series_id", "game_id", "complete", "competition_level"}
    if not required.issubset(frame.columns):
        raise GridMarketFoundationError("GRID games context schema is incomplete")
    rows = frame.loc[frame["series_id"].astype(str) == str(series_id)]
    if len(rows) != 1:
        raise GridMarketFoundationError(
            f"series {series_id} does not resolve to exactly one local game"
        )
    row = rows.iloc[0]
    if bool(row["complete"]) is not True:
        raise GridMarketFoundationError(f"series {series_id} is not completed")
    if str(row["competition_level"]) not in {"major", "other-pro"}:
        raise GridMarketFoundationError(f"series {series_id} is not verified professional")
    game_id = str(row["game_id"])
    context = context_from_grid_games_parquet(
        games_path, series_id=str(series_id), provider_game_id=game_id
    )
    return game_id, context


def retrieve_riot_events(
    *,
    series_id: str,
    games_path: Path,
    output_root: Path,
    key: str,
) -> dict[str, Any]:
    """Download one series-scoped Riot event file to a content-addressed path."""
    if not str(series_id).isdigit():
        raise GridMarketFoundationError("series ID must be numeric")
    game_id, context = _known_series_context(games_path, str(series_id))
    files = _file_list(key, str(series_id))
    matches = [
        row
        for row in files
        if str(row.get("id") or "") == "events-riot-game-1"
    ]
    if len(matches) != 1:
        raise GridMarketFoundationError(
            f"series {series_id} has no unique events-riot-game-1 file"
        )
    file_row = matches[0]
    if str(file_row.get("status") or "") != "ready":
        raise GridMarketFoundationError(f"series {series_id} Riot events are not ready")
    signed_url = str(file_row.get("fullURL") or "")
    if not signed_url:
        raise GridMarketFoundationError(
            f"series {series_id} Riot events have no download capability"
        )

    raw_dir = output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".events_{series_id}_1_riot.", suffix=".jsonl", dir=raw_dir
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        if not _download(signed_url, key, temporary):
            raise GridMarketFoundationError(
                f"series {series_id} Riot event download was rate-limited"
            )
        source_sha = _sha256_file(temporary)
        destination = raw_dir / f"events_{series_id}_1_riot_{source_sha}.jsonl"
        if destination.exists():
            if _sha256_file(destination) != source_sha:
                raise GridMarketFoundationError(
                    f"content-addressed path conflict for series {series_id}"
                )
            temporary.unlink()
        else:
            os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    safe_metadata = _safe_file_metadata(file_row)
    receipt = {
        "schema_version": RETRIEVAL_RECEIPT_SCHEMA,
        "retrieved_at": _utc_now(),
        "scope": "private_personal_research_only",
        "provider_series_id": str(series_id),
        "provider_game_id": game_id,
        "grid_context_record_sha256": context["record_sha256"],
        "file_id": "events-riot-game-1",
        "file_metadata": safe_metadata,
        "file_metadata_sha256": _hash(safe_metadata),
        "raw_path": str(destination),
        "raw_sha256": source_sha,
        "raw_bytes": destination.stat().st_size,
        "credentials_serialized": False,
        "signed_url_retained": False,
        "mutations_used": False,
    }
    receipt["receipt_sha256"] = _hash(receipt)
    receipt_path = (
        output_root
        / "receipts"
        / f"retrieval-{series_id}-{receipt['receipt_sha256']}.json"
    )
    write_immutable_receipt(receipt_path, receipt)
    return {**receipt, "receipt_path": str(receipt_path)}


def _iter_riot(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            value = json.loads(raw)
            if isinstance(value, dict):
                yield value


def _iter_grid_transactions(path: Path) -> Iterator[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".jsonl"))
        if len(members) != 1:
            raise GridMarketFoundationError(
                f"{path} must contain exactly one Series Events JSONL"
            )
        with archive.open(members[0]) as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                value = json.loads(raw)
                if isinstance(value, dict):
                    yield value


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _opposite_team(team_id: int | None) -> int | None:
    if team_id == 100:
        return 200
    if team_id == 200:
        return 100
    return None


def _event_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("rootGameID"),
        row.get("generationID"),
        row.get("sequenceIndex"),
        row.get("rfc461Schema"),
    )


def _canonical_riot_events(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen: dict[tuple[Any, ...], str] = {}
    conflicts: set[tuple[Any, ...]] = set()
    duplicates = 0
    ids: set[tuple[str, str, str]] = set()
    game_infos: list[dict[str, Any]] = []
    game_ends: list[dict[str, Any]] = []
    for row in _iter_riot(path):
        key = _event_key(row)
        row_hash = _hash(row)
        if key in seen:
            if seen[key] == row_hash:
                duplicates += 1
                continue
            conflicts.add(key)
            continue
        seen[key] = row_hash
        platform = str(row.get("platformID") or "")
        game_id = str(row.get("gameID") or "")
        root_id = str(row.get("rootGameID") or "")
        if platform and game_id and root_id:
            ids.add((platform, game_id, root_id))
        if row.get("rfc461Schema") == "game_info":
            game_infos.append(row)
        if row.get("rfc461Schema") == "game_end":
            game_ends.append(row)
        rows.append(row)
    blockers: list[str] = []
    if conflicts:
        blockers.append("riot.conflicting_event_revision")
    if len(ids) != 1:
        blockers.append("identity.riot_game_tuple_not_unique")
    if len(game_infos) != 1:
        blockers.append("identity.riot_game_info_not_unique")
    if len(game_ends) != 1:
        blockers.append("outcome.riot_game_end_not_unique")
    return {
        "rows": rows,
        "identity": next(iter(ids)) if len(ids) == 1 else None,
        "game_info": game_infos[0] if len(game_infos) == 1 else None,
        "game_end": game_ends[0] if len(game_ends) == 1 else None,
        "duplicates": duplicates,
        "conflicts": len(conflicts),
        "blockers": sorted(set(blockers)),
    }


def _team_objective(team: Mapping[str, Any], objective_type: str) -> tuple[int, bool]:
    matches = [
        row
        for row in team.get("objectives") or []
        if isinstance(row, Mapping) and str(row.get("type") or "") == objective_type
    ]
    if len(matches) > 1:
        raise GridMarketFoundationError(
            f"duplicate GRID objective state for {objective_type}"
        )
    if not matches:
        return 0, False
    return int(matches[0].get("completionCount") or 0), bool(
        matches[0].get("completedFirst")
    )


def _grid_final_state(path: Path, provider_game_id: str) -> dict[str, Any]:
    latest: tuple[int, Mapping[str, Any]] | None = None
    sequences: list[int] = []
    for transaction in _iter_grid_transactions(path):
        sequence = _as_int(transaction.get("sequenceNumber"))
        if sequence is not None:
            sequences.append(sequence)
        for event in transaction.get("events") or []:
            if not isinstance(event, Mapping):
                continue
            state = event.get("seriesState")
            if not isinstance(state, Mapping):
                continue
            for game in state.get("games") or []:
                if (
                    isinstance(game, Mapping)
                    and str(game.get("id") or "") == provider_game_id
                    and (latest is None or (sequence or -1) >= latest[0])
                ):
                    latest = (sequence or -1, game)
    if latest is None:
        return {"status": "unavailable", "blockers": ["grid.final_state_missing"]}
    game = latest[1]
    teams = [team for team in game.get("teams") or [] if isinstance(team, Mapping)]
    if game.get("finished") is not True or len(teams) != 2:
        return {
            "status": "unavailable",
            "blockers": ["grid.final_game_or_team_state_incomplete"],
        }
    side_to_riot = {"blue": 100, "red": 200}
    derived: dict[int, dict[str, Any]] = {}
    for team in teams:
        riot_team = side_to_riot.get(str(team.get("side") or "").lower())
        if riot_team is None or riot_team in derived:
            return {
                "status": "unavailable",
                "blockers": ["identity.grid_team_side_ambiguous"],
            }
        tower_count, tower_first = _team_objective(team, "destroyTower")
        inhibitor_count, inhibitor_first = _team_objective(team, "destroyFortifier")
        dragon_count, dragon_first = _team_objective(team, "slayDragon")
        baron_count, baron_first = _team_objective(team, "slayBaron")
        derived[riot_team] = {
            "provider_team_id": str(team.get("id") or ""),
            "side": str(team.get("side") or "").lower(),
            "won": bool(team.get("won")),
            "kills": _as_int(team.get("kills")),
            "first_blood": bool(team.get("firstKill")),
            "tower_count": tower_count,
            "first_tower": tower_first,
            "inhibitor_count": inhibitor_count,
            "first_inhibitor": inhibitor_first,
            "dragon_count": dragon_count,
            "first_dragon": dragon_first,
            "baron_count": baron_count,
            "first_baron": baron_first,
        }
    ordered = sorted(set(sequences))
    gaps = [
        [left + 1, right - 1]
        for left, right in zip(ordered, ordered[1:])
        if right > left + 1
    ]
    return {
        "status": "verified",
        "blockers": [],
        "source_sequence": latest[0],
        "sequence_gaps": gaps,
        "teams": {str(key): value for key, value in sorted(derived.items())},
    }


def _first_team(rows: Sequence[Mapping[str, Any]], team_field: str) -> int | None:
    if not rows:
        return None
    ordered = sorted(
        rows,
        key=lambda row: (
            _as_int(row.get("gameTime")) if _as_int(row.get("gameTime")) is not None else 10**18,
            _as_int(row.get("sequenceIndex"))
            if _as_int(row.get("sequenceIndex")) is not None
            else 10**18,
        ),
    )
    return _as_int(ordered[0].get(team_field))


def _riot_market_rows(rows: Sequence[Mapping[str, Any]], cutoff_ms: int | None) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if cutoff_ms is None
        or (
            _as_int(row.get("gameTime")) is not None
            and int(row["gameTime"]) <= cutoff_ms
        )
    ]
    buildings = [
        row for row in eligible if row.get("rfc461Schema") == "building_destroyed"
    ]
    turrets = [
        row for row in buildings if str(row.get("buildingType") or "").lower() == "turret"
    ]
    inhibitors = [
        row
        for row in buildings
        if str(row.get("buildingType") or "").lower() == "inhibitor"
    ]
    monsters = [
        row for row in eligible if row.get("rfc461Schema") == "epic_monster_kill"
    ]
    dragons = [
        row for row in monsters if str(row.get("monsterType") or "").lower() == "dragon"
    ]
    barons = [
        row for row in monsters if str(row.get("monsterType") or "").lower() == "baron"
    ]
    kills = [row for row in eligible if row.get("rfc461Schema") == "champion_kill"]
    unique_turrets = {
        (
            _as_int(row.get("teamID")),
            str(row.get("lane") or ""),
            str(row.get("turretTier") or ""),
            str(row.get("nexusTurretName") or ""),
        )
        for row in turrets
    }
    return {
        "total_tower_destructions": len(turrets),
        "unique_towers_destroyed": len(unique_turrets),
        "first_tower": _opposite_team(_first_team(turrets, "teamID")),
        "total_inhibitor_destructions": len(inhibitors),
        "first_inhibitor": _opposite_team(_first_team(inhibitors, "teamID")),
        "total_dragons": len(dragons),
        "first_dragon": _first_team(dragons, "killerTeamID"),
        "total_barons": len(barons),
        "first_baron": _first_team(barons, "killerTeamID"),
        "total_kills": len(kills),
        "first_blood": _first_team(kills, "killerTeamID"),
    }


def _grid_labels(final_state: Mapping[str, Any]) -> dict[str, Any]:
    teams = {
        int(team_id): value
        for team_id, value in (final_state.get("teams") or {}).items()
    }

    def total(field: str) -> int:
        return sum(int(row.get(field) or 0) for row in teams.values())

    def first(field: str) -> int | None:
        winners = [team_id for team_id, row in teams.items() if row.get(field)]
        return winners[0] if len(winners) == 1 else None

    return {
        "total_tower_destructions": total("tower_count"),
        "first_tower": first("first_tower"),
        "total_inhibitor_destructions": total("inhibitor_count"),
        "first_inhibitor": first("first_inhibitor"),
        "total_dragons": total("dragon_count"),
        "first_dragon": first("first_dragon"),
        "total_barons": total("baron_count"),
        "first_baron": first("first_baron"),
        "total_kills": total("kills"),
        "first_blood": first("first_blood"),
    }


def _grid_checkpoint_state(
    path: Path,
    provider_game_id: str,
    minute: int,
    maximum_age_ms: int,
) -> dict[str, Any]:
    cutoff_seconds = minute * 60
    best: tuple[float, int, Mapping[str, Any]] | None = None
    for transaction in _iter_grid_transactions(path):
        sequence = _as_int(transaction.get("sequenceNumber"))
        for event in transaction.get("events") or []:
            if not isinstance(event, Mapping) or event.get("includesFullState") is not True:
                continue
            state = event.get("seriesState")
            if not isinstance(state, Mapping):
                continue
            for game in state.get("games") or []:
                if not isinstance(game, Mapping) or str(game.get("id") or "") != provider_game_id:
                    continue
                clock = game.get("clock") or {}
                seconds = clock.get("currentSeconds") if isinstance(clock, Mapping) else None
                if not isinstance(seconds, (int, float)) or seconds > cutoff_seconds:
                    continue
                candidate = (float(seconds), sequence or -1, game)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
    if best is None:
        return {
            "status": "unavailable",
            "blockers": ["checkpoint.grid_full_state_at_or_before_missing"],
        }
    state_time_ms = int(best[0] * 1000)
    age_ms = minute * 60_000 - state_time_ms
    blockers: list[str] = []
    if age_ms < 0:
        blockers.append("checkpoint.post_checkpoint_state_forbidden")
    if age_ms > maximum_age_ms:
        blockers.append("checkpoint.grid_full_state_stale")
    teams = [row for row in best[2].get("teams") or [] if isinstance(row, Mapping)]
    if len(teams) != 2:
        blockers.append("checkpoint.grid_team_state_incomplete")
    values: dict[int, dict[str, Any]] = {}
    for team in teams:
        riot_team = {"blue": 100, "red": 200}.get(
            str(team.get("side") or "").lower()
        )
        if riot_team is None or riot_team in values:
            blockers.append("checkpoint.grid_team_side_ambiguous")
            continue
        tower_count, tower_first = _team_objective(team, "destroyTower")
        inhibitor_count, inhibitor_first = _team_objective(team, "destroyFortifier")
        dragon_count, dragon_first = _team_objective(team, "slayDragon")
        baron_count, baron_first = _team_objective(team, "slayBaron")
        values[riot_team] = {
            "provider_team_id": str(team.get("id") or ""),
            "side": str(team.get("side") or "").lower(),
            "kills": _as_int(team.get("kills")),
            "money": _as_int(team.get("money")),
            "net_worth": _as_int(team.get("netWorth")),
            "experience_points": _as_int(team.get("experiencePoints")),
            "vision_score": _as_int(team.get("visionScore")),
            "tower_count": tower_count,
            "first_tower": tower_first,
            "inhibitor_count": inhibitor_count,
            "first_inhibitor": inhibitor_first,
            "dragon_count": dragon_count,
            "first_dragon": dragon_first,
            "baron_count": baron_count,
            "first_baron": baron_first,
        }
    return {
        "status": "eligible" if not blockers else "unavailable",
        "blockers": sorted(set(blockers)),
        "state_game_time_ms": state_time_ms,
        "state_age_ms": age_ms,
        "source_sequence": best[1],
        "teams": {str(key): value for key, value in sorted(values.items())},
    }


def _checkpoint_grid_market_values(state: Mapping[str, Any]) -> dict[str, Any]:
    teams = {
        int(team_id): value for team_id, value in (state.get("teams") or {}).items()
    }

    def total(field: str) -> int:
        return sum(int(row.get(field) or 0) for row in teams.values())

    def first(field: str) -> int | None:
        values = [team_id for team_id, row in teams.items() if row.get(field)]
        return values[0] if len(values) == 1 else None

    return {
        "total_tower_destructions": total("tower_count"),
        "first_tower": first("first_tower"),
        "total_inhibitor_destructions": total("inhibitor_count"),
        "first_inhibitor": first("first_inhibitor"),
        "total_dragons": total("dragon_count"),
        "first_dragon": first("first_dragon"),
        "total_barons": total("baron_count"),
        "first_baron": first("first_baron"),
        "total_kills": total("kills"),
        "first_blood": None,
    }


def _locate_riot_file(
    series_id: str, grid_raw_dir: Path, private_root: Path
) -> Path | None:
    retrieved = sorted((private_root / "raw").glob(f"events_{series_id}_1_riot_*.jsonl"))
    if len(retrieved) == 1:
        return retrieved[0]
    if len(retrieved) > 1:
        hashes = {_sha256_file(path) for path in retrieved}
        if len(hashes) != 1:
            return None
        return retrieved[-1]
    local = grid_raw_dir / f"events_{series_id}_1_riot.jsonl"
    return local if local.is_file() else None


def _game_record(
    *,
    series_id: str,
    games_path: Path,
    grid_raw_dir: Path,
    private_root: Path,
    checkpoints: Sequence[int],
    maximum_age_ms: int,
) -> dict[str, Any]:
    game_id, context = _known_series_context(games_path, series_id)
    provider_path = grid_raw_dir / f"events_{series_id}_grid.jsonl.zip"
    series_meta_path = grid_raw_dir / f"series_{series_id}.json"
    riot_path = _locate_riot_file(series_id, grid_raw_dir, private_root)
    blockers: list[str] = []
    if not provider_path.is_file():
        blockers.append("source.grid_series_events_missing")
    if not series_meta_path.is_file():
        blockers.append("source.grid_series_metadata_missing")
    if riot_path is None:
        blockers.append("source.riot_events_missing_or_ambiguous")
    if blockers:
        return {
            "provider_series_id": series_id,
            "provider_game_id": game_id,
            "status": "unavailable",
            "blockers": sorted(blockers),
        }

    assert riot_path is not None
    foundation = build_foundation_artifact(
        provider_events_path=provider_path,
        riot_events_path=riot_path,
        series_metadata_path=series_meta_path,
        context=context,
        checkpoints=checkpoints,
        maximum_state_age_ms=maximum_age_ms,
        retention_class="private-personal-research",
    )
    riot = _canonical_riot_events(riot_path)
    grid_final = _grid_final_state(provider_path, game_id)
    blockers.extend(riot["blockers"])
    if foundation["verified_game"]["status"] != "verified":
        blockers.extend(foundation["verified_game"]["blockers"])
    if grid_final["status"] != "verified":
        blockers.extend(grid_final["blockers"])

    riot_labels = _riot_market_rows(riot["rows"], cutoff_ms=None)
    grid_labels = (
        _grid_labels(grid_final) if grid_final["status"] == "verified" else {}
    )
    label_checks: dict[str, dict[str, Any]] = {}
    for spec in TARGET_SPECS:
        target = spec["target"]
        if target == "unique_towers_destroyed":
            label_checks[target] = {
                "status": "unavailable",
                "value": riot_labels.get(target),
                "blockers": ["label.independent_unique_tower_verification_missing"],
            }
            continue
        riot_value = riot_labels.get(target)
        grid_value = grid_labels.get(target)
        target_blockers: list[str] = []
        if blockers:
            target_blockers.append("label.game_evidence_unavailable")
        if riot_value != grid_value:
            target_blockers.append("label.grid_riot_conflict")
        label_checks[target] = {
            "status": "verified" if not target_blockers else "unavailable",
            "value": riot_value if not target_blockers else None,
            "riot_value": riot_value,
            "grid_value": grid_value,
            "blockers": sorted(target_blockers),
        }

    checkpoint_rows: list[dict[str, Any]] = []
    baseline_by_minute = {
        int(row["minute"]): row for row in foundation["checkpoint_states"]
    }
    for minute in checkpoints:
        riot_values = _riot_market_rows(riot["rows"], cutoff_ms=int(minute) * 60_000)
        grid_state = _grid_checkpoint_state(
            provider_path, game_id, int(minute), maximum_age_ms
        )
        grid_values = (
            _checkpoint_grid_market_values(grid_state)
            if grid_state["status"] == "eligible"
            else {}
        )
        baseline = baseline_by_minute.get(int(minute)) or {}
        checkpoint_blockers: list[str] = []
        grid_corroboration_blockers = list(grid_state.get("blockers") or [])
        if blockers:
            checkpoint_blockers.append("checkpoint.game_evidence_unavailable")
        if baseline.get("historical_model_evidence_status") != "eligible":
            checkpoint_blockers.extend(
                baseline.get("historical_model_evidence_blockers") or []
            )
        for target in (
            "total_tower_destructions",
            "first_tower",
            "total_inhibitor_destructions",
            "first_inhibitor",
            "total_dragons",
            "first_dragon",
            "total_barons",
            "first_baron",
            "total_kills",
        ):
            if grid_state["status"] == "eligible" and riot_values[target] != grid_values[target]:
                checkpoint_blockers.append(
                    f"checkpoint.grid_riot_conflict.{target}"
                )
        checkpoint_rows.append(
            {
                "minute": int(minute),
                "status": "eligible" if not checkpoint_blockers else "unavailable",
                "blockers": sorted(set(checkpoint_blockers)),
                "rule": "riot_events_and_latest_riot_stats_at_or_before_checkpoint",
                "grid_state_game_time_ms": grid_state.get("state_game_time_ms"),
                "grid_state_age_ms": grid_state.get("state_age_ms"),
                "grid_source_sequence": grid_state.get("source_sequence"),
                "grid_corroboration_status": grid_state.get("status"),
                "grid_corroboration_blockers": sorted(
                    set(grid_corroboration_blockers)
                ),
                "riot_values": riot_values,
                "grid_values": grid_values,
                "baseline": {
                    "current_kills": baseline.get("current_kills"),
                    "gold_difference": baseline.get("gold_difference"),
                    "source_sequence": baseline.get("source_sequence"),
                    "state_game_time_ms": baseline.get("state_game_time_ms"),
                    "state_age_ms": baseline.get("state_age_ms"),
                },
                "candidate_grid_team_state": grid_state.get("teams") or {},
                "event_cadence_status": "unavailable",
                "event_cadence_blockers": [
                    "checkpoint.provider_sequence_gaps_preclude_complete_cadence"
                ]
                if grid_final.get("sequence_gaps")
                else [],
            }
        )

    verified = foundation["verified_game"]
    return {
        "provider_series_id": series_id,
        "provider_game_id": game_id,
        "riot_platform_id": verified.get("riot_platform_id"),
        "riot_game_id": verified.get("riot_game_id"),
        "riot_root_game_id": verified.get("riot_root_game_id"),
        "league": verified.get("league"),
        "patch": verified.get("patch"),
        "status": "verified" if not blockers else "unavailable",
        "blockers": sorted(set(blockers)),
        "source_receipts": {
            "grid_events": {
                "path": str(provider_path),
                "sha256": _sha256_file(provider_path),
            },
            "riot_events": {
                "path": str(riot_path),
                "sha256": _sha256_file(riot_path),
            },
            "series_metadata": {
                "path": str(series_meta_path),
                "sha256": _sha256_file(series_meta_path),
            },
            "context": {
                "path": str(games_path),
                "sha256": context["source_sha256"],
                "record_sha256": context["record_sha256"],
            },
        },
        "identity": {
            "provider_to_riot_status": verified.get("status"),
            "provider_to_riot_team_ids": verified.get("provider_to_riot_team_ids"),
            "teams": verified.get("teams"),
        },
        "labels": label_checks,
        "checkpoints": checkpoint_rows,
        "replay_receipt": foundation["replay_receipt"],
    }


def _market_summary(games: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for spec in TARGET_SPECS:
        target = spec["target"]
        verified_games = sum(
            1
            for game in games
            if (game.get("labels") or {}).get(target, {}).get("status") == "verified"
        )
        eligible_checkpoints = sum(
            1
            for game in games
            for checkpoint in game.get("checkpoints") or []
            if checkpoint.get("status") == "eligible"
            and (game.get("labels") or {}).get(target, {}).get("status")
            == "verified"
        )
        total_checkpoints = sum(len(game.get("checkpoints") or []) for game in games)
        blockers: list[str] = []
        if verified_games < 30:
            blockers.append("sample.minimum_30_verified_maps_not_met")
        if eligible_checkpoints < 24:
            blockers.append("sample.minimum_24_eligible_checkpoints_not_met")
        if spec["market_semantics"] in {
            "external_market_rule_needed_for_respawned_nexus_turrets",
            "derived_diagnostic_not_a_verified_market_label",
        }:
            blockers.append("market.definition_not_verified")
        if target == "unique_towers_destroyed":
            blockers.append("label.independent_verification_missing")
        structural = (
            "verified_on_bounded_sample"
            if verified_games > 0 and target != "unique_towers_destroyed"
            else "unavailable"
        )
        summaries.append(
            {
                **spec,
                "structural_feasibility": structural,
                "verified_label_games": verified_games,
                "games_total": len(games),
                "eligible_checkpoints": eligible_checkpoints,
                "checkpoints_total": total_checkpoints,
                "checkpoint_coverage": (
                    eligible_checkpoints / total_checkpoints
                    if total_checkpoints
                    else 0.0
                ),
                "research_authority_status": "unavailable",
                "research_authority_blockers": sorted(set(blockers)),
                "probability_authorized": False,
                "fair_odds_authorized": False,
                "edge_authorized": False,
                "expected_value_authorized": False,
            }
        )
    summaries.sort(
        key=lambda row: (
            row["structural_feasibility"] == "verified_on_bounded_sample",
            row["verified_label_games"],
            row["checkpoint_coverage"],
            row["market_semantics"] == "clear_event_definition",
        ),
        reverse=True,
    )
    for index, row in enumerate(summaries, start=1):
        row["feasibility_rank"] = index
    return summaries


def build_report(
    *,
    series_ids: Sequence[str],
    games_path: Path,
    grid_raw_dir: Path,
    private_root: Path,
    checkpoints: Sequence[int] = DEFAULT_CHECKPOINTS,
    maximum_age_ms: int = PROVISIONAL_MAX_STATE_AGE_MS,
) -> dict[str, Any]:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    games = [
        _game_record(
            series_id=str(series_id),
            games_path=games_path,
            grid_raw_dir=grid_raw_dir,
            private_root=private_root,
            checkpoints=checkpoints,
            maximum_age_ms=maximum_age_ms,
        )
        for series_id in series_ids
    ]
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "foundation_schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "scope": {
            "privacy": "private_personal_research_only",
            "classifier_built": False,
            "probability_authorized": False,
            "fair_odds_authorized": False,
            "edge_authorized": False,
            "expected_value_authorized": False,
            "publication_authorized": False,
        },
        "configuration": {
            "series_ids": [str(value) for value in series_ids],
            "checkpoints": [int(value) for value in checkpoints],
            "maximum_grid_state_age_ms": int(maximum_age_ms),
            "checkpoint_rule": "riot_events_and_latest_riot_stats_at_or_before_checkpoint",
            "grid_state_role": "optional_corroboration_subject_to_five_second_age_bound",
            "minimum_verified_maps_for_authority": 30,
            "minimum_eligible_checkpoints_for_authority": 24,
        },
        "catalog_provenance": {
            "path": str(CATALOG_PATH),
            "catalog_sha256": catalog["catalog_sha256"],
            "generated_at": catalog["generated_at"],
            "endpoint_schema_hashes": {
                row["endpoint_id"]: row["schema_sha256"]
                for row in catalog["endpoints"]
            },
        },
        "target_inventory": list(TARGET_SPECS),
        "games": games,
        "market_feasibility": _market_summary(games),
        "coverage": {
            "games_total": len(games),
            "games_verified": sum(
                1 for game in games if game.get("status") == "verified"
            ),
            "games_unavailable": sum(
                1 for game in games if game.get("status") != "verified"
            ),
            "checkpoint_rows": sum(len(game.get("checkpoints") or []) for game in games),
            "eligible_checkpoint_rows": sum(
                1
                for game in games
                for row in game.get("checkpoints") or []
                if row.get("status") == "eligible"
            ),
            "blocker_counts": dict(
                sorted(
                    Counter(
                        blocker
                        for game in games
                        for blocker in game.get("blockers") or []
                    ).items()
                )
            ),
        },
        "known_limitations": [
            "This is a bounded feasibility sample from one league and exact patch, not a modeling cohort.",
            "GRID Series Events sequence gaps make complete event-cadence features unavailable.",
            "Riot LiveStats events and stats delivered through GRID provide the primary timestamped checkpoint path; GRID Series State is only corroboration when at or before the checkpoint and within the five-second provisional age bound.",
            "Total-towers market semantics require external rules for repeated destruction of respawned Nexus turrets.",
            "No market prices or bookmaker benchmark were collected.",
            "No probability, fair odds, edge, expected value, or betting classification is authorized.",
        ],
    }
    report["report_sha256"] = _hash(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    retrieve = subparsers.add_parser("retrieve")
    retrieve.add_argument("--series", required=True)
    retrieve.add_argument("--games", type=Path, default=DEFAULT_GAMES)
    retrieve.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    retrieve.add_argument("--grid-env-file", type=Path)

    build = subparsers.add_parser("build")
    build.add_argument("--series", required=True)
    build.add_argument("--games", type=Path, default=DEFAULT_GAMES)
    build.add_argument("--grid-raw", type=Path, default=DEFAULT_GRID_RAW)
    build.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    build.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    series_ids = [value.strip() for value in args.series.split(",") if value.strip()]
    if not series_ids:
        raise GridMarketFoundationError("at least one series ID is required")
    if args.command == "retrieve":
        key = _api_key(args.grid_env_file)
        receipts = [
            retrieve_riot_events(
                series_id=series_id,
                games_path=args.games,
                output_root=args.root,
                key=key,
            )
            for series_id in series_ids
        ]
        print(
            json.dumps(
                {
                    "status": "ok",
                    "series": series_ids,
                    "files_downloaded": len(receipts),
                    "receipts": [
                        {
                            "series_id": row["provider_series_id"],
                            "raw_sha256": row["raw_sha256"],
                            "raw_bytes": row["raw_bytes"],
                            "receipt_path": row["receipt_path"],
                        }
                        for row in receipts
                    ],
                    "credentials_serialized": False,
                    "signed_urls_retained": False,
                },
                sort_keys=True,
            )
        )
        return 0
    report = build_report(
        series_ids=series_ids,
        games_path=args.games,
        grid_raw_dir=args.grid_raw,
        private_root=args.root,
    )
    output = args.output or args.root / "reports" / (
        f"market-feasibility-{report['report_sha256']}.json"
    )
    write_immutable_receipt(output, report)
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(output),
                "report_sha256": report["report_sha256"],
                "games_total": report["coverage"]["games_total"],
                "games_verified": report["coverage"]["games_verified"],
                "eligible_checkpoint_rows": report["coverage"][
                    "eligible_checkpoint_rows"
                ],
                "authority": "unavailable",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
