"""Offline, fail-closed GRID/Riot live-data foundation.

This module deliberately stops before feature selection, modeling, pricing, or
serving.  It turns already-local GRID Series Events and Riot live-stats files
into content-addressed evidence for one completed professional game.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


RAW_CAPTURE_SCHEMA = "scryglass.grid-live.raw-capture.v1"
VERIFIED_GAME_SCHEMA = "scryglass.grid-live.verified-game.v1"
CHECKPOINT_SCHEMA = "scryglass.grid-live.checkpoint-state.v1"
REPLAY_RECEIPT_SCHEMA = "scryglass.grid-live.replay-receipt.v1"
AUTHORITY_RECEIPT_SCHEMA = "scryglass.grid-live.authority-receipt.v1"
COHORT_MANIFEST_SCHEMA = "scryglass.grid-live.cohort-manifest.v1"
FOUNDATION_ARTIFACT_SCHEMA = "scryglass.grid-live.foundation-artifact.v1"

DEFAULT_CHECKPOINTS = (10, 15, 20, 25)
PROVISIONAL_MAX_STATE_AGE_MS = 5_000
REQUIRED_RIOT_TEAMS = (100, 200)


class GridLiveFoundationError(RuntimeError):
    """Base error for invalid offline foundation inputs."""


class ImmutableReceiptConflict(GridLiveFoundationError):
    """An immutable receipt path already contains different bytes."""


@dataclass(frozen=True)
class RawCapture:
    schema_version: str
    capture_id: str
    source: str
    path: str
    byte_count: int
    sha256: str
    provider_series_id: str | None
    provider_game_id: str | None
    riot_platform_id: str | None
    riot_game_id: str | None
    riot_root_game_id: str | None
    first_provider_timestamp: str | None
    last_provider_timestamp: str | None
    first_sequence: int | None
    last_sequence: int | None
    schema_fingerprint_sha256: str
    received_at: str | None
    retention_class: str
    live_latency_authority: bool


@dataclass(frozen=True)
class VerifiedGame:
    schema_version: str
    status: str
    blockers: tuple[str, ...]
    provider_series_id: str | None
    provider_game_id: str | None
    riot_platform_id: str | None
    riot_game_id: str | None
    riot_root_game_id: str | None
    tournament: str | None
    league: str | None
    patch: str | None
    provider_to_riot_team_ids: tuple[tuple[str, int], ...]
    teams: tuple[dict[str, Any], ...]
    players: tuple[dict[str, Any], ...]
    started_at: str | None
    ended_at: str | None
    game_end_time_ms: int | None
    winner_provider_team_id: str | None
    winner_riot_team_id: int | None
    team_kills: tuple[tuple[int, int], ...]
    total_kills: int | None
    evidence_capture_ids: tuple[str, ...]


@dataclass(frozen=True)
class CheckpointState:
    schema_version: str
    minute: int
    state_status: str
    historical_model_evidence_status: str
    historical_model_evidence_blockers: tuple[str, ...]
    prospective_live_latency_status: str
    prospective_live_latency_blockers: tuple[str, ...]
    state_game_time_ms: int | None
    state_age_ms: int | None
    source_sequence: int | None
    source_event_sha256: str | None
    current_kills: int | None
    blue_kills: int | None
    red_kills: int | None
    blue_gold: int | None
    red_gold: int | None
    gold_difference: int | None
    stream_watermark_sequence: int | None
    sequence_gaps_before_state: tuple[tuple[int, int], ...]
    received_at: str | None
    live_latency_authority: bool


@dataclass(frozen=True)
class ReplayReceipt:
    schema_version: str
    parser_version: str
    source_capture_ids: tuple[str, ...]
    source_sha256: tuple[str, ...]
    input_event_count: int
    canonical_event_count: int
    duplicate_event_count: int
    late_event_count: int
    conflicting_revision_sequences: tuple[int, ...]
    nonnegative_sequence_gaps: tuple[tuple[int, int], ...]
    schema_fingerprint_sha256: str
    derived_evidence_sha256: str
    deterministic_replay: bool


@dataclass(frozen=True)
class AuthorityReceipt:
    schema_version: str
    status: str
    blockers: tuple[str, ...]
    historical_replay_evidence_status: str
    historical_replay_evidence_blockers: tuple[str, ...]
    model_evaluation_status: str
    model_evaluation_blockers: tuple[str, ...]
    prospective_live_latency_status: str
    prospective_live_latency_blockers: tuple[str, ...]
    market_comparison_status: str
    market_comparison_blockers: tuple[str, ...]
    approved_league: str | None
    approved_patch: str | None
    approved_checkpoints: tuple[int, ...]
    probability_authorized: bool
    fair_odds_authorized: bool
    edge_authorized: bool
    expected_value_authorized: bool
    evidence_sha256: str


@dataclass(frozen=True)
class CohortManifest:
    schema_version: str
    games_total: int
    games_verified: int
    games_unavailable: int
    checkpoint_coverage: tuple[dict[str, Any], ...]
    blocker_counts: tuple[tuple[str, int], ...]
    raw_capture_ids: tuple[str, ...]
    replay_receipt_sha256: str
    authority_receipt: AuthorityReceipt


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", str(value or "")))


def _normalize_patch(value: Any) -> str:
    parts = str(value or "").strip().split(".")
    if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        return ""
    return f"{int(parts[0])}.{int(parts[1])}"


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted


def _iter_jsonl(path: Path) -> Iterator[tuple[bytes, dict[str, Any]]]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = sorted(name for name in archive.namelist() if name.endswith(".jsonl"))
            if len(members) != 1:
                raise GridLiveFoundationError(
                    f"{path} must contain exactly one JSONL member"
                )
            with archive.open(members[0]) as handle:
                for raw in handle:
                    stripped = raw.strip()
                    if stripped:
                        value = json.loads(stripped)
                        if not isinstance(value, dict):
                            raise GridLiveFoundationError(
                                f"{path} contains a non-object JSONL row"
                            )
                        yield stripped, value
        return
    with path.open("rb") as handle:
        for raw in handle:
            stripped = raw.strip()
            if stripped:
                value = json.loads(stripped)
                if not isinstance(value, dict):
                    raise GridLiveFoundationError(
                        f"{path} contains a non-object JSONL row"
                    )
                yield stripped, value


def _event_external_id(player: Mapping[str, Any], provider: str) -> str | None:
    for link in player.get("externalLinks") or []:
        if not isinstance(link, Mapping):
            continue
        data_provider = link.get("dataProvider") or {}
        external = link.get("externalEntity") or {}
        if (
            isinstance(data_provider, Mapping)
            and str(data_provider.get("name") or "") == provider
            and isinstance(external, Mapping)
        ):
            value = str(external.get("id") or "").strip()
            if value:
                return value
    return None


def _game_from_state(state: Mapping[str, Any], game_id: str) -> Mapping[str, Any]:
    for game in state.get("games") or []:
        if isinstance(game, Mapping) and str(game.get("id") or "") == game_id:
            return game
    return {}


def _team_rows_from_state(
    state: Mapping[str, Any], game_id: str
) -> list[Mapping[str, Any]]:
    game = _game_from_state(state, game_id)
    rows = [team for team in game.get("teams") or [] if isinstance(team, Mapping)]
    return rows


def _provider_archive(path: Path) -> dict[str, Any]:
    transactions: list[dict[str, Any]] = []
    schemas: set[str] = set()
    series_ids: set[str] = set()
    timestamps: list[str] = []
    sequences: list[int] = []
    duplicate_count = 0
    late_count = 0
    conflicting: set[int] = set()
    seen: dict[int, str] = {}
    maximum_seen: int | None = None
    input_count = 0

    for raw, transaction in _iter_jsonl(path):
        input_count += 1
        series_id = str(transaction.get("seriesId") or "").strip()
        if series_id:
            series_ids.add(series_id)
        occurred_at = str(transaction.get("occurredAt") or "").strip()
        if occurred_at:
            timestamps.append(occurred_at)
        sequence = _as_int(transaction.get("sequenceNumber"))
        row_hash = hashlib.sha256(raw).hexdigest()
        if sequence is None:
            schemas.add("transaction:sequence-missing")
        else:
            if maximum_seen is not None and sequence < maximum_seen:
                late_count += 1
            maximum_seen = sequence if maximum_seen is None else max(maximum_seen, sequence)
            if sequence in seen:
                if seen[sequence] == row_hash:
                    duplicate_count += 1
                    continue
                conflicting.add(sequence)
                continue
            seen[sequence] = row_hash
            sequences.append(sequence)
        for event in transaction.get("events") or []:
            if isinstance(event, Mapping):
                schemas.add(f"event:{event.get('type') or 'unknown'}")
        transactions.append(transaction)

    ordered = sorted(
        transactions,
        key=lambda row: (
            _as_int(row.get("sequenceNumber"))
            if _as_int(row.get("sequenceNumber")) is not None
            else math.inf,
            str(row.get("occurredAt") or ""),
            str(row.get("id") or ""),
        ),
    )
    starts: list[tuple[dict[str, Any], Mapping[str, Any]]] = []
    provider_winners: set[tuple[str, str]] = set()
    ended_games: set[str] = set()
    latest_kills: dict[tuple[str, str], int] = {}

    for transaction in ordered:
        for event in transaction.get("events") or []:
            if not isinstance(event, Mapping):
                continue
            event_type = str(event.get("type") or "")
            state = event.get("seriesState") or {}
            if not isinstance(state, Mapping):
                state = {}
            target = event.get("target") or {}
            target_id = (
                str(target.get("id") or "")
                if isinstance(target, Mapping)
                else ""
            )
            if event_type == "series-started-game" and target_id:
                starts.append((transaction, event))
            if event_type == "team-won-game" and target_id:
                actor = event.get("actor") or {}
                actor_id = (
                    str(actor.get("id") or "")
                    if isinstance(actor, Mapping)
                    else ""
                )
                if actor_id:
                    provider_winners.add((target_id, actor_id))
            if event_type == "series-ended-game" and target_id:
                ended_games.add(target_id)
            for team in state.get("teams") or []:
                if not isinstance(team, Mapping):
                    continue
                team_id = str(team.get("id") or "")
                kills = _as_int(team.get("kills"))
                if team_id and kills is not None:
                    for game in state.get("games") or []:
                        if isinstance(game, Mapping):
                            game_id = str(game.get("id") or "")
                            if game_id:
                                latest_kills[(game_id, team_id)] = kills
            for game in state.get("games") or []:
                if not isinstance(game, Mapping):
                    continue
                game_id = str(game.get("id") or "")
                for team in game.get("teams") or []:
                    if not isinstance(team, Mapping):
                        continue
                    team_id = str(team.get("id") or "")
                    kills = _as_int(team.get("kills"))
                    if game_id and team_id and kills is not None:
                        latest_kills[(game_id, team_id)] = kills

    return {
        "transactions": ordered,
        "series_ids": sorted(series_ids),
        "timestamps": sorted(timestamps),
        "sequences": sorted(set(sequences)),
        "schemas": sorted(schemas),
        "starts": starts,
        "provider_winners": sorted(provider_winners),
        "ended_games": sorted(ended_games),
        "latest_kills": latest_kills,
        "input_count": input_count,
        "canonical_count": len(ordered),
        "duplicate_count": duplicate_count,
        "late_count": late_count,
        "conflicting_sequences": sorted(conflicting),
    }


def _riot_archive(path: Path) -> dict[str, Any]:
    schemas: set[str] = set()
    timestamps: list[str] = []
    sequences: list[int] = []
    sequence_hashes: dict[int, str] = {}
    duplicates = 0
    late = 0
    conflicting: set[int] = set()
    maximum_seen: int | None = None
    game_infos: list[dict[str, Any]] = []
    game_ends: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    kill_counts: Counter[int] = Counter()
    input_count = 0

    for raw, row in _iter_jsonl(path):
        input_count += 1
        schema = str(row.get("rfc461Schema") or "")
        schemas.add(schema or "unknown")
        timestamp = str(row.get("rfc460Timestamp") or "").strip()
        if timestamp:
            timestamps.append(timestamp)
        sequence = _as_int(row.get("sequenceIndex"))
        row_hash = hashlib.sha256(raw).hexdigest()
        if sequence is None:
            conflicting.add(-2_147_483_648)
            continue
        if maximum_seen is not None and sequence < maximum_seen:
            late += 1
        maximum_seen = sequence if maximum_seen is None else max(maximum_seen, sequence)
        if sequence in sequence_hashes:
            if sequence_hashes[sequence] == row_hash:
                duplicates += 1
                continue
            conflicting.add(sequence)
            continue
        sequence_hashes[sequence] = row_hash
        sequences.append(sequence)

        if schema == "game_info":
            game_infos.append(row)
        elif schema == "game_end":
            game_ends.append(row)
        elif schema == "champion_kill":
            team_id = _as_int(row.get("killerTeamID"))
            if team_id in REQUIRED_RIOT_TEAMS:
                kill_counts[team_id] += 1
        elif schema == "stats_update":
            game_time = _as_int(row.get("gameTime"))
            teams = []
            for team in row.get("teams") or []:
                if not isinstance(team, Mapping):
                    continue
                teams.append(
                    {
                        "team_id": _as_int(team.get("teamID")),
                        "kills": _as_int(team.get("championsKills")),
                        "gold": _as_int(team.get("totalGold")),
                    }
                )
            stats.append(
                {
                    "sequence": sequence,
                    "game_time_ms": game_time,
                    "timestamp": timestamp or None,
                    "teams": teams,
                    "event_sha256": row_hash,
                }
            )

    nonnegative = sorted(sequence for sequence in set(sequences) if sequence >= 0)
    gaps = [
        (left + 1, right - 1)
        for left, right in zip(nonnegative, nonnegative[1:])
        if right > left + 1
    ]
    stats.sort(
        key=lambda row: (
            row["game_time_ms"] if row["game_time_ms"] is not None else math.inf,
            row["sequence"],
            row["event_sha256"],
        )
    )
    return {
        "schemas": sorted(schemas),
        "timestamps": sorted(timestamps),
        "sequences": sorted(set(sequences)),
        "stats": stats,
        "game_infos": game_infos,
        "game_ends": game_ends,
        "kill_counts": kill_counts,
        "input_count": input_count,
        "canonical_count": len(sequence_hashes),
        "duplicate_count": duplicates,
        "late_count": late,
        "conflicting_sequences": sorted(conflicting),
        "nonnegative_sequence_gaps": gaps,
    }


def _sequence_gaps(sequences: Sequence[int]) -> list[tuple[int, int]]:
    ordered = sorted(set(sequences))
    return [
        (left + 1, right - 1)
        for left, right in zip(ordered, ordered[1:])
        if right > left + 1
    ]


def _capture(
    *,
    path: Path,
    source: str,
    provider_series_id: str | None,
    provider_game_id: str | None,
    riot_platform_id: str | None,
    riot_game_id: str | None,
    riot_root_game_id: str | None,
    timestamps: Sequence[str],
    sequences: Sequence[int],
    schemas: Sequence[str],
    retention_class: str,
) -> RawCapture:
    digest = _sha256_file(path)
    core = {
        "source": source,
        "sha256": digest,
        "provider_series_id": provider_series_id,
        "provider_game_id": provider_game_id,
        "riot_platform_id": riot_platform_id,
        "riot_game_id": riot_game_id,
    }
    return RawCapture(
        schema_version=RAW_CAPTURE_SCHEMA,
        capture_id=f"sha256:{_hash_value(core)}",
        source=source,
        path=str(path),
        byte_count=path.stat().st_size,
        sha256=digest,
        provider_series_id=provider_series_id,
        provider_game_id=provider_game_id,
        riot_platform_id=riot_platform_id,
        riot_game_id=riot_game_id,
        riot_root_game_id=riot_root_game_id,
        first_provider_timestamp=min(timestamps) if timestamps else None,
        last_provider_timestamp=max(timestamps) if timestamps else None,
        first_sequence=min(sequences) if sequences else None,
        last_sequence=max(sequences) if sequences else None,
        schema_fingerprint_sha256=_hash_value(sorted(set(schemas))),
        received_at=None,
        retention_class=retention_class,
        live_latency_authority=False,
    )


def context_from_grid_games_parquet(
    path: Path,
    *,
    series_id: str,
    provider_game_id: str,
) -> dict[str, Any]:
    """Build an exact, content-addressed context receipt from local compact GRID."""
    import pandas as pd

    frame = pd.read_parquet(path)
    selected = frame[
        (frame["series_id"].astype(str) == str(series_id))
        & (frame["game_id"].astype(str) == str(provider_game_id))
    ]
    if len(selected) != 1:
        raise GridLiveFoundationError(
            "context source must contain exactly one series/game row"
        )
    row = selected.iloc[0]
    record = {
        "provider_series_id": str(row["series_id"]),
        "provider_game_id": str(row["game_id"]),
        "tournament": str(row["tournament"]),
        "league": str(row["league"]),
        "patch": _normalize_patch(row["patch"]),
        "complete": bool(row["complete"]),
        "winner_provider_team_id": str(row["winner_team_id"]),
        "teams": sorted(
            [
                {
                    "provider_team_id": str(row["team_1_id"]),
                    "name": str(row["team_1_name"]),
                    "side": str(row["team_1_side"]).lower(),
                },
                {
                    "provider_team_id": str(row["team_2_id"]),
                    "name": str(row["team_2_name"]),
                    "side": str(row["team_2_side"]).lower(),
                },
            ],
            key=lambda team: team["provider_team_id"],
        ),
    }
    return {
        "schema_version": "scryglass.grid-live.context-evidence.v1",
        "source_path": str(path),
        "source_sha256": _sha256_file(path),
        "record": record,
        "record_sha256": _hash_value(record),
    }


def _validate_context(context: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    record = context.get("record")
    if not isinstance(record, Mapping):
        return {}, ["identity.context_record_missing"]
    record_dict = dict(record)
    if str(context.get("record_sha256") or "") != _hash_value(record_dict):
        blockers.append("identity.context_record_hash_mismatch")
    source_path = Path(str(context.get("source_path") or ""))
    source_sha = str(context.get("source_sha256") or "")
    if not _is_sha256(source_sha):
        blockers.append("identity.context_source_hash_missing")
    elif not source_path.is_file():
        blockers.append("identity.context_source_missing")
    elif _sha256_file(source_path) != source_sha:
        blockers.append("identity.context_source_hash_mismatch")
    required = (
        "provider_series_id",
        "provider_game_id",
        "tournament",
        "league",
        "patch",
        "winner_provider_team_id",
        "teams",
    )
    for field in required:
        if record_dict.get(field) in (None, "", []):
            blockers.append(f"identity.context_{field}_missing")
    if record_dict.get("complete") is not True:
        blockers.append("outcome.context_game_not_complete")
    return record_dict, blockers


def _unique_payload(
    rows: Sequence[dict[str, Any]],
    *,
    missing_blocker: str,
    conflict_blocker: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not rows:
        return None, [missing_blocker]
    by_hash = {_hash_value(row): row for row in rows}
    if len(by_hash) != 1:
        return None, [conflict_blocker]
    return next(iter(by_hash.values())), []


def _provider_identity(
    provider: Mapping[str, Any],
    context_record: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    expected_series = str(context_record.get("provider_series_id") or "")
    expected_game = str(context_record.get("provider_game_id") or "")
    if provider.get("series_ids") != [expected_series]:
        blockers.append("identity.provider_series_mismatch_or_ambiguous")
    candidates = []
    for transaction, event in provider.get("starts") or []:
        target = event.get("target") or {}
        if isinstance(target, Mapping) and str(target.get("id") or "") == expected_game:
            candidates.append((transaction, event))
    if len(candidates) != 1:
        blockers.append("identity.provider_game_start_missing_or_ambiguous")
        return {}, blockers
    transaction, event = candidates[0]
    state = event.get("seriesState") or {}
    game = _game_from_state(state, expected_game)
    if not game:
        blockers.append("identity.provider_game_state_missing")
        return {}, blockers
    patch = _normalize_patch((game.get("titleVersion") or {}).get("name"))
    teams = []
    all_puuids: list[str] = []
    for team in game.get("teams") or []:
        if not isinstance(team, Mapping):
            continue
        players = []
        for player in team.get("players") or []:
            if not isinstance(player, Mapping):
                continue
            puuid = _event_external_id(player, "RIOT_PUUID")
            players.append(
                {
                    "provider_player_id": str(player.get("id") or ""),
                    "provider_name": str(player.get("name") or ""),
                    "puuid": puuid,
                }
            )
            if puuid:
                all_puuids.append(puuid)
        teams.append(
            {
                "provider_team_id": str(team.get("id") or ""),
                "name": str(team.get("name") or ""),
                "side": str(team.get("side") or "").lower(),
                "players": players,
                "kills_at_start": _as_int(team.get("kills")),
            }
        )
    if len(teams) != 2 or {team["side"] for team in teams} != {"blue", "red"}:
        blockers.append("identity.provider_team_side_invalid")
    if any(len(team["players"]) != 5 for team in teams):
        blockers.append("identity.provider_roster_not_5v5")
    if len(all_puuids) != 10 or len(set(all_puuids)) != 10:
        blockers.append("identity.provider_puuid_invalid_or_ambiguous")
    return {
        "series_id": expected_series,
        "game_id": expected_game,
        "started_at": str(game.get("startedAt") or transaction.get("occurredAt") or "")
        or None,
        "patch": patch,
        "teams": teams,
    }, blockers


def _verify_game(
    *,
    provider: Mapping[str, Any],
    riot: Mapping[str, Any],
    provider_identity: Mapping[str, Any],
    series_meta: Mapping[str, Any],
    context_record: Mapping[str, Any],
    captures: Sequence[RawCapture],
    initial_blockers: Iterable[str],
) -> VerifiedGame:
    blockers = list(initial_blockers)
    game_info, info_blockers = _unique_payload(
        riot.get("game_infos") or [],
        missing_blocker="identity.riot_game_info_missing",
        conflict_blocker="identity.riot_game_info_conflicting",
    )
    game_end, end_blockers = _unique_payload(
        riot.get("game_ends") or [],
        missing_blocker="outcome.riot_game_end_missing",
        conflict_blocker="outcome.riot_game_end_conflicting",
    )
    blockers.extend(info_blockers)
    blockers.extend(end_blockers)
    if provider.get("conflicting_sequences"):
        blockers.append("replay.provider_conflicting_revision")
    if riot.get("conflicting_sequences"):
        blockers.append("replay.riot_conflicting_revision")

    expected_series = str(context_record.get("provider_series_id") or "")
    expected_game = str(context_record.get("provider_game_id") or "")
    if str(series_meta.get("id") or "") != expected_series:
        blockers.append("identity.series_metadata_id_mismatch")
    if str(series_meta.get("tournament") or "") != str(
        context_record.get("tournament") or ""
    ):
        blockers.append("identity.tournament_mismatch")
    if _normalize_patch(provider_identity.get("patch")) != _normalize_patch(
        context_record.get("patch")
    ):
        blockers.append("identity.provider_patch_mismatch")

    riot_platform = str((game_info or {}).get("platformID") or "") or None
    riot_game_id_value = _as_int((game_info or {}).get("gameID"))
    riot_root_game_id_value = _as_int((game_info or {}).get("rootGameID"))
    riot_patch = _normalize_patch((game_info or {}).get("gameVersion"))
    if not riot_platform or riot_game_id_value is None or riot_root_game_id_value is None:
        blockers.append("identity.riot_game_identifier_missing")
    if riot_patch != _normalize_patch(context_record.get("patch")):
        blockers.append("identity.riot_patch_mismatch")

    riot_participants = [
        participant
        for participant in (game_info or {}).get("participants") or []
        if isinstance(participant, Mapping)
    ]
    riot_puuids = [str(participant.get("puuid") or "") for participant in riot_participants]
    if (
        len(riot_participants) != 10
        or any(not puuid for puuid in riot_puuids)
        or len(set(riot_puuids)) != 10
    ):
        blockers.append("identity.riot_puuid_invalid_or_ambiguous")
    provider_teams = list(provider_identity.get("teams") or [])
    provider_puuids = {
        str(player.get("puuid") or "")
        for team in provider_teams
        for player in team.get("players") or []
    }
    if provider_puuids != set(riot_puuids):
        blockers.append("identity.provider_riot_puuid_set_mismatch")

    riot_by_team: dict[int, set[str]] = defaultdict(set)
    participant_by_puuid: dict[str, Mapping[str, Any]] = {}
    for participant in riot_participants:
        team_id = _as_int(participant.get("teamID"))
        puuid = str(participant.get("puuid") or "")
        if team_id is not None and puuid:
            riot_by_team[team_id].add(puuid)
            participant_by_puuid[puuid] = participant
    team_map: dict[str, int] = {}
    for provider_team in provider_teams:
        puuids = {
            str(player.get("puuid") or "")
            for player in provider_team.get("players") or []
        }
        matches = [
            team_id for team_id, candidate in riot_by_team.items() if candidate == puuids
        ]
        if len(matches) != 1:
            blockers.append("identity.provider_riot_team_mapping_ambiguous")
        else:
            team_map[str(provider_team.get("provider_team_id") or "")] = matches[0]
    if set(team_map.values()) != set(REQUIRED_RIOT_TEAMS) or len(team_map) != 2:
        blockers.append("identity.provider_riot_team_mapping_incomplete")

    context_teams = {
        str(team.get("provider_team_id") or ""): team
        for team in context_record.get("teams") or []
        if isinstance(team, Mapping)
    }
    for provider_team in provider_teams:
        team_id = str(provider_team.get("provider_team_id") or "")
        context_team = context_teams.get(team_id)
        if not context_team:
            blockers.append("identity.context_team_missing")
            continue
        if (
            str(context_team.get("name") or "") != provider_team.get("name")
            or str(context_team.get("side") or "").lower() != provider_team.get("side")
        ):
            blockers.append("identity.context_team_name_or_side_mismatch")

    provider_winners = {
        team_id
        for game_id, team_id in provider.get("provider_winners") or []
        if game_id == expected_game
    }
    provider_winner = next(iter(provider_winners)) if len(provider_winners) == 1 else None
    if len(provider_winners) != 1:
        blockers.append("outcome.provider_winner_missing_or_conflicting")
    if expected_game not in set(provider.get("ended_games") or []):
        blockers.append("outcome.provider_game_end_missing")
    if provider_winner and provider_winner != str(
        context_record.get("winner_provider_team_id") or ""
    ):
        blockers.append("outcome.context_winner_conflict")
    riot_winner = _as_int((game_end or {}).get("winningTeam"))
    if riot_winner not in REQUIRED_RIOT_TEAMS:
        blockers.append("outcome.riot_winner_invalid")
    elif provider_winner and team_map.get(provider_winner) != riot_winner:
        blockers.append("outcome.provider_riot_winner_conflict")

    game_end_time = _as_int((game_end or {}).get("gameTime"))
    final_stats = [
        row
        for row in riot.get("stats") or []
        if row.get("game_time_ms") is not None
        and game_end_time is not None
        and row["game_time_ms"] <= game_end_time
    ]
    final_state = final_stats[-1] if final_stats else None
    if not final_state or final_state.get("game_time_ms") != game_end_time:
        blockers.append("outcome.final_stats_at_game_end_missing")
    final_team_kills: dict[int, int] = {}
    for team in (final_state or {}).get("teams") or []:
        team_id = team.get("team_id")
        kills = team.get("kills")
        if team_id in REQUIRED_RIOT_TEAMS and kills is not None:
            final_team_kills[int(team_id)] = int(kills)
    if set(final_team_kills) != set(REQUIRED_RIOT_TEAMS):
        blockers.append("outcome.final_team_kills_missing")
    event_kills = {
        team_id: int((riot.get("kill_counts") or {}).get(team_id, 0))
        for team_id in REQUIRED_RIOT_TEAMS
    }
    if final_team_kills and final_team_kills != event_kills:
        blockers.append("outcome.riot_kill_event_total_conflict")
    for provider_team_id, riot_team_id in team_map.items():
        provider_kills = (provider.get("latest_kills") or {}).get(
            (expected_game, provider_team_id)
        )
        if (
            provider_kills is None
            or riot_team_id not in final_team_kills
            or provider_kills != final_team_kills[riot_team_id]
        ):
            blockers.append("outcome.provider_riot_total_kills_conflict")
            break

    teams = tuple(
        sorted(
            (
                {
                    "provider_team_id": str(team.get("provider_team_id") or ""),
                    "riot_team_id": team_map.get(
                        str(team.get("provider_team_id") or "")
                    ),
                    "name": str(team.get("name") or ""),
                    "side": str(team.get("side") or ""),
                }
                for team in provider_teams
            ),
            key=lambda team: team["provider_team_id"],
        )
    )
    players = []
    for team in provider_teams:
        provider_team_id = str(team.get("provider_team_id") or "")
        for player in team.get("players") or []:
            puuid = str(player.get("puuid") or "")
            riot_player = participant_by_puuid.get(puuid) or {}
            players.append(
                {
                    "provider_player_id": str(player.get("provider_player_id") or ""),
                    "provider_team_id": provider_team_id,
                    "riot_participant_id": _as_int(riot_player.get("participantID")),
                    "riot_team_id": _as_int(riot_player.get("teamID")),
                    "puuid": puuid or None,
                    "provider_name": str(player.get("provider_name") or ""),
                    "role": str(riot_player.get("role") or ""),
                    "champion": str(riot_player.get("championName") or ""),
                }
            )

    unique_blockers = tuple(sorted(set(blockers)))
    return VerifiedGame(
        schema_version=VERIFIED_GAME_SCHEMA,
        status="verified" if not unique_blockers else "unavailable",
        blockers=unique_blockers,
        provider_series_id=expected_series or None,
        provider_game_id=expected_game or None,
        riot_platform_id=riot_platform,
        riot_game_id=str(riot_game_id_value) if riot_game_id_value is not None else None,
        riot_root_game_id=(
            str(riot_root_game_id_value)
            if riot_root_game_id_value is not None
            else None
        ),
        tournament=str(context_record.get("tournament") or "") or None,
        league=str(context_record.get("league") or "") or None,
        patch=_normalize_patch(context_record.get("patch")) or None,
        provider_to_riot_team_ids=tuple(sorted(team_map.items())),
        teams=teams,
        players=tuple(
            sorted(
                players,
                key=lambda player: (
                    player["riot_team_id"] or 0,
                    player["riot_participant_id"] or 0,
                ),
            )
        ),
        started_at=str(provider_identity.get("started_at") or "") or None,
        ended_at=str((game_end or {}).get("rfc460Timestamp") or "") or None,
        game_end_time_ms=game_end_time,
        winner_provider_team_id=provider_winner,
        winner_riot_team_id=riot_winner,
        team_kills=tuple(sorted(final_team_kills.items())),
        total_kills=sum(final_team_kills.values()) if final_team_kills else None,
        evidence_capture_ids=tuple(capture.capture_id for capture in captures),
    )


def _checkpoint_states(
    riot: Mapping[str, Any],
    *,
    verified_game: VerifiedGame,
    checkpoints: Sequence[int],
    maximum_state_age_ms: int,
) -> list[CheckpointState]:
    output = []
    conflicting = bool(riot.get("conflicting_sequences"))
    all_gaps = list(riot.get("nonnegative_sequence_gaps") or [])
    watermark = max(
        (sequence for sequence in riot.get("sequences") or [] if sequence >= 0),
        default=None,
    )
    for minute in checkpoints:
        blockers: list[str] = []
        cutoff = int(minute) * 60_000
        candidates = [
            row
            for row in riot.get("stats") or []
            if row.get("game_time_ms") is not None and row["game_time_ms"] <= cutoff
        ]
        state = candidates[-1] if candidates else None
        if state is None:
            blockers.append("checkpoint.state_at_or_before_missing")
        if conflicting:
            blockers.append("replay.riot_conflicting_revision")
        state_sequence = state.get("sequence") if state else None
        gaps_before = [
            gap
            for gap in all_gaps
            if state_sequence is not None and gap[0] <= state_sequence
        ]
        if gaps_before:
            blockers.append("checkpoint.sequence_gap_before_state")
        age = cutoff - state["game_time_ms"] if state else None
        if age is not None and age < 0:
            blockers.append("checkpoint.post_checkpoint_state_forbidden")
        if age is not None and age > maximum_state_age_ms:
            blockers.append("checkpoint.state_stale")
        teams: dict[int, dict[str, Any]] = {}
        for team in (state or {}).get("teams") or []:
            team_id = team.get("team_id")
            if team_id in REQUIRED_RIOT_TEAMS:
                teams[int(team_id)] = team
        if set(teams) != set(REQUIRED_RIOT_TEAMS):
            blockers.append("checkpoint.team_state_incomplete")
        for team_id in REQUIRED_RIOT_TEAMS:
            row = teams.get(team_id) or {}
            if row.get("kills") is None or row.get("gold") is None:
                blockers.append("checkpoint.baseline_field_missing")
                break
        if verified_game.status != "verified":
            blockers.append("checkpoint.game_identity_or_outcome_unavailable")

        blue = teams.get(100) or {}
        red = teams.get(200) or {}
        blue_kills = blue.get("kills")
        red_kills = red.get("kills")
        blue_gold = blue.get("gold")
        red_gold = red.get("gold")
        unique_blockers = tuple(sorted(set(blockers)))
        state_valid = not unique_blockers
        output.append(
            CheckpointState(
                schema_version=CHECKPOINT_SCHEMA,
                minute=int(minute),
                state_status="valid" if state_valid else "unavailable",
                historical_model_evidence_status=(
                    "eligible" if state_valid else "unavailable"
                ),
                historical_model_evidence_blockers=unique_blockers,
                prospective_live_latency_status="unavailable",
                prospective_live_latency_blockers=(
                    "latency.prospective_receive_time_missing",
                ),
                state_game_time_ms=state.get("game_time_ms") if state else None,
                state_age_ms=age,
                source_sequence=state_sequence,
                source_event_sha256=state.get("event_sha256") if state else None,
                current_kills=(
                    int(blue_kills) + int(red_kills)
                    if blue_kills is not None and red_kills is not None
                    else None
                ),
                blue_kills=int(blue_kills) if blue_kills is not None else None,
                red_kills=int(red_kills) if red_kills is not None else None,
                blue_gold=int(blue_gold) if blue_gold is not None else None,
                red_gold=int(red_gold) if red_gold is not None else None,
                gold_difference=(
                    int(blue_gold) - int(red_gold)
                    if blue_gold is not None and red_gold is not None
                    else None
                ),
                stream_watermark_sequence=watermark,
                sequence_gaps_before_state=tuple(gaps_before),
                received_at=None,
                live_latency_authority=False,
            )
        )
    return output


def build_foundation_artifact(
    *,
    provider_events_path: Path,
    riot_events_path: Path,
    series_metadata_path: Path,
    context: Mapping[str, Any],
    checkpoints: Sequence[int] = DEFAULT_CHECKPOINTS,
    maximum_state_age_ms: int = PROVISIONAL_MAX_STATE_AGE_MS,
    retention_class: str = "local-research-unverified-rights",
) -> dict[str, Any]:
    """Build one offline evidence artifact without granting live authority."""
    if not checkpoints or any(int(value) <= 0 for value in checkpoints):
        raise GridLiveFoundationError("checkpoints must be positive exact minutes")
    if len(set(int(value) for value in checkpoints)) != len(checkpoints):
        raise GridLiveFoundationError("checkpoints must be unique")
    if maximum_state_age_ms < 0:
        raise GridLiveFoundationError("maximum_state_age_ms cannot be negative")
    for path in (provider_events_path, riot_events_path, series_metadata_path):
        if not path.is_file():
            raise GridLiveFoundationError(f"required local source is missing: {path}")

    context_record, context_blockers = _validate_context(context)
    provider = _provider_archive(provider_events_path)
    riot = _riot_archive(riot_events_path)
    series_meta = json.loads(series_metadata_path.read_text(encoding="utf-8"))
    if not isinstance(series_meta, dict):
        raise GridLiveFoundationError("series metadata must be a JSON object")
    provider_identity, identity_blockers = _provider_identity(provider, context_record)

    game_info, _ = _unique_payload(
        riot.get("game_infos") or [],
        missing_blocker="identity.riot_game_info_missing",
        conflict_blocker="identity.riot_game_info_conflicting",
    )
    captures = [
        _capture(
            path=provider_events_path,
            source="grid-series-events",
            provider_series_id=str(context_record.get("provider_series_id") or "") or None,
            provider_game_id=str(context_record.get("provider_game_id") or "") or None,
            riot_platform_id=None,
            riot_game_id=None,
            riot_root_game_id=None,
            timestamps=provider.get("timestamps") or [],
            sequences=provider.get("sequences") or [],
            schemas=provider.get("schemas") or [],
            retention_class=retention_class,
        ),
        _capture(
            path=riot_events_path,
            source="riot-live-stats-file",
            provider_series_id=str(context_record.get("provider_series_id") or "") or None,
            provider_game_id=str(context_record.get("provider_game_id") or "") or None,
            riot_platform_id=str((game_info or {}).get("platformID") or "") or None,
            riot_game_id=(
                str((game_info or {}).get("gameID"))
                if (game_info or {}).get("gameID") is not None
                else None
            ),
            riot_root_game_id=(
                str((game_info or {}).get("rootGameID"))
                if (game_info or {}).get("rootGameID") is not None
                else None
            ),
            timestamps=riot.get("timestamps") or [],
            sequences=riot.get("sequences") or [],
            schemas=riot.get("schemas") or [],
            retention_class=retention_class,
        ),
        _capture(
            path=series_metadata_path,
            source="grid-series-metadata",
            provider_series_id=str(series_meta.get("id") or "") or None,
            provider_game_id=None,
            riot_platform_id=None,
            riot_game_id=None,
            riot_root_game_id=None,
            timestamps=[str(series_meta.get("date") or "")]
            if series_meta.get("date")
            else [],
            sequences=[],
            schemas=["grid-series-metadata"],
            retention_class=retention_class,
        ),
    ]
    verified = _verify_game(
        provider=provider,
        riot=riot,
        provider_identity=provider_identity,
        series_meta=series_meta,
        context_record=context_record,
        captures=captures,
        initial_blockers=(*context_blockers, *identity_blockers),
    )
    checkpoint_states = _checkpoint_states(
        riot,
        verified_game=verified,
        checkpoints=tuple(int(value) for value in checkpoints),
        maximum_state_age_ms=maximum_state_age_ms,
    )

    evidence = {
        "verified_game": asdict(verified),
        "checkpoint_states": [asdict(state) for state in checkpoint_states],
    }
    derived_hash = _hash_value(evidence)
    replay = ReplayReceipt(
        schema_version=REPLAY_RECEIPT_SCHEMA,
        parser_version=FOUNDATION_ARTIFACT_SCHEMA,
        source_capture_ids=tuple(capture.capture_id for capture in captures),
        source_sha256=tuple(capture.sha256 for capture in captures),
        input_event_count=int(provider["input_count"]) + int(riot["input_count"]),
        canonical_event_count=int(provider["canonical_count"])
        + int(riot["canonical_count"]),
        duplicate_event_count=int(provider["duplicate_count"])
        + int(riot["duplicate_count"]),
        late_event_count=int(provider["late_count"]) + int(riot["late_count"]),
        conflicting_revision_sequences=tuple(
            sorted(
                set(provider.get("conflicting_sequences") or [])
                | set(riot.get("conflicting_sequences") or [])
            )
        ),
        nonnegative_sequence_gaps=tuple(
            riot.get("nonnegative_sequence_gaps") or []
        ),
        schema_fingerprint_sha256=_hash_value(
            {
                "provider": provider.get("schemas") or [],
                "riot": riot.get("schemas") or [],
            }
        ),
        derived_evidence_sha256=derived_hash,
        deterministic_replay=not bool(
            provider.get("conflicting_sequences") or riot.get("conflicting_sequences")
        ),
    )

    historical_replay_blockers: set[str] = set()
    if verified.status != "verified":
        historical_replay_blockers.add("historical.game_verification_failed")
    if any(state.state_status != "valid" for state in checkpoint_states):
        historical_replay_blockers.add("historical.checkpoint_state_failed")
    if not replay.deterministic_replay:
        historical_replay_blockers.add("historical.replay_conflict")

    model_evaluation_blockers = {
        "model.coverage_threshold_not_met",
        "model.feature_selection_not_run",
        "model.calibration_not_run",
        "model.heldout_evaluation_not_run",
        "model.exact_patch_sample_not_established",
        "model.league_sample_not_established",
    }
    prospective_latency_blockers = {
        "latency.prospective_receive_time_missing",
        "latency.prospective_watermark_not_established",
    }
    if provider.get("sequences") and _sequence_gaps(provider["sequences"]):
        prospective_latency_blockers.add("latency.provider_stream_sequence_gaps")
    market_comparison_blockers = {
        "market.price_or_external_benchmark_missing",
        "market.comparative_heldout_evaluation_not_run",
    }
    serving_blockers = {
        "serving.offline_first_slice_only",
        "serving.freshness_not_established",
        *historical_replay_blockers,
        *model_evaluation_blockers,
        *prospective_latency_blockers,
        *market_comparison_blockers,
    }
    authority_core = {
        "verified_game_sha256": _hash_value(asdict(verified)),
        "checkpoint_states_sha256": _hash_value(
            [asdict(state) for state in checkpoint_states]
        ),
        "replay_receipt_sha256": _hash_value(asdict(replay)),
        "historical_replay_blockers": sorted(historical_replay_blockers),
        "model_evaluation_blockers": sorted(model_evaluation_blockers),
        "prospective_latency_blockers": sorted(prospective_latency_blockers),
        "market_comparison_blockers": sorted(market_comparison_blockers),
        "serving_blockers": sorted(serving_blockers),
    }
    authority = AuthorityReceipt(
        schema_version=AUTHORITY_RECEIPT_SCHEMA,
        status="unavailable",
        blockers=tuple(sorted(serving_blockers)),
        historical_replay_evidence_status=(
            "eligible" if not historical_replay_blockers else "unavailable"
        ),
        historical_replay_evidence_blockers=tuple(
            sorted(historical_replay_blockers)
        ),
        model_evaluation_status="unavailable",
        model_evaluation_blockers=tuple(sorted(model_evaluation_blockers)),
        prospective_live_latency_status="unavailable",
        prospective_live_latency_blockers=tuple(
            sorted(prospective_latency_blockers)
        ),
        market_comparison_status="unavailable",
        market_comparison_blockers=tuple(sorted(market_comparison_blockers)),
        approved_league=None,
        approved_patch=None,
        approved_checkpoints=(),
        probability_authorized=False,
        fair_odds_authorized=False,
        edge_authorized=False,
        expected_value_authorized=False,
        evidence_sha256=_hash_value(authority_core),
    )

    blocker_counter: Counter[str] = Counter(verified.blockers)
    for state in checkpoint_states:
        blocker_counter.update(state.historical_model_evidence_blockers)
        blocker_counter.update(state.prospective_live_latency_blockers)
    blocker_counter.update(authority.blockers)
    coverage = []
    for state in checkpoint_states:
        coverage.append(
            {
                "minute": state.minute,
                "games_total": 1,
                "state_valid": int(state.state_status == "valid"),
                "historical_model_evidence_eligible": int(
                    state.historical_model_evidence_status == "eligible"
                ),
                "prospective_live_latency_authorized": 0,
                "state_coverage": float(state.state_status == "valid"),
                "historical_model_evidence_coverage": float(
                    state.historical_model_evidence_status == "eligible"
                ),
                "prospective_live_latency_coverage": 0.0,
                "historical_blockers": list(
                    state.historical_model_evidence_blockers
                ),
                "prospective_live_latency_blockers": list(
                    state.prospective_live_latency_blockers
                ),
            }
        )
    manifest = CohortManifest(
        schema_version=COHORT_MANIFEST_SCHEMA,
        games_total=1,
        games_verified=int(verified.status == "verified"),
        games_unavailable=int(verified.status != "verified"),
        checkpoint_coverage=tuple(coverage),
        blocker_counts=tuple(sorted(blocker_counter.items())),
        raw_capture_ids=tuple(capture.capture_id for capture in captures),
        replay_receipt_sha256=_hash_value(asdict(replay)),
        authority_receipt=authority,
    )
    artifact = {
        "schema_version": FOUNDATION_ARTIFACT_SCHEMA,
        "contracts": {
            "raw_capture": RAW_CAPTURE_SCHEMA,
            "verified_game": VERIFIED_GAME_SCHEMA,
            "checkpoint_state": CHECKPOINT_SCHEMA,
            "cohort_manifest": COHORT_MANIFEST_SCHEMA,
            "authority_receipt": AUTHORITY_RECEIPT_SCHEMA,
        },
        "configuration": {
            "checkpoints": [int(value) for value in checkpoints],
            "maximum_state_age_ms": maximum_state_age_ms,
            "checkpoint_rule": "last_state_with_game_time_at_or_before_checkpoint",
            "interpolation_authorized": False,
            "historical_files_may_support_model_evidence": True,
            "historical_files_confer_live_latency_authority": False,
            "grid_access_alone_confers_market_edge_authority": False,
            "market_price_or_external_benchmark_required": True,
        },
        "context_evidence": dict(context),
        "raw_captures": [asdict(capture) for capture in captures],
        "verified_game": asdict(verified),
        "checkpoint_states": [asdict(state) for state in checkpoint_states],
        "replay_receipt": asdict(replay),
        "cohort_manifest": asdict(manifest),
    }
    artifact["artifact_sha256"] = _hash_value(artifact)
    return artifact


def write_immutable_receipt(path: Path, payload: Mapping[str, Any]) -> str:
    """Write canonical receipt bytes once; identical replay is idempotent."""
    data = _canonical_bytes(dict(payload)) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    absolute_parent = path.parent.absolute()
    current = absolute_parent
    while True:
        if current.is_symlink():
            raise ImmutableReceiptConflict(
                f"receipt parent cannot be a symlink: {current}"
            )
        if current == current.parent:
            break
        current = current.parent
    if path.exists() or path.is_symlink():
        current_stat = path.lstat()
        if (
            not stat.S_ISREG(current_stat.st_mode)
            or path.is_symlink()
            or current_stat.st_nlink != 1
        ):
            raise ImmutableReceiptConflict(
                f"receipt target is not a unique regular file: {path}"
            )
        if path.read_bytes() != data:
            raise ImmutableReceiptConflict(
                f"immutable receipt already contains different bytes: {path}"
            )
        return hashlib.sha256(data).hexdigest()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-events", type=Path, required=True)
    parser.add_argument("--riot-events", type=Path, required=True)
    parser.add_argument("--series-metadata", type=Path, required=True)
    parser.add_argument("--context-parquet", type=Path, required=True)
    parser.add_argument("--series-id", required=True)
    parser.add_argument("--provider-game-id", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    context = context_from_grid_games_parquet(
        args.context_parquet,
        series_id=args.series_id,
        provider_game_id=args.provider_game_id,
    )
    artifact = build_foundation_artifact(
        provider_events_path=args.provider_events,
        riot_events_path=args.riot_events,
        series_metadata_path=args.series_metadata,
        context=context,
    )
    if args.output:
        receipt_sha = write_immutable_receipt(args.output, artifact)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "receipt_sha256": receipt_sha,
                    "artifact_sha256": artifact["artifact_sha256"],
                    "verified_game_status": artifact["verified_game"]["status"],
                    "authority_status": artifact["cohort_manifest"][
                        "authority_receipt"
                    ]["status"],
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
