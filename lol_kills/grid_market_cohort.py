"""Private historical GRID objective-market cohort builder.

This module is data-foundation only.  It does not train or serve a model and
does not compute probabilities, prices, edges, or expected values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from lol_kills.etl.grid_ingest import (
    GRAPHQL_ENDPOINT,
    GridIngestError,
    _api_key,
    _download,
    _file_list,
    _graphql,
)
from lol_kills.grid_live_foundation import write_immutable_receipt


SCHEMA_VERSION = "scryglass.grid-objective-cohort.v1"
RECEIPT_SCHEMA_VERSION = "scryglass.grid-objective-retrieval-receipt.v1"
MANIFEST_SCHEMA_VERSION = "scryglass.grid-objective-cohort-manifest.v1"
REGISTRY_VERSION = "scryglass.grid-major-league-season-registry.v3"
DEFAULT_ROOT = Path("data/lol/warehouse/private_grid/market_cohort/v1")
CHECKPOINT_MINUTES = (10, 15, 20, 25)
TARGETS = (
    "first_tower",
    "first_inhibitor",
    "first_dragon",
    "total_dragons",
    "first_baron",
    "total_barons",
    "first_blood",
    "total_inhibitor_destructions",
)

# Exact GRID tournament IDs, discovered from authenticated Central Data.
# Challenger/academy tournaments are intentionally absent.
LEAGUE_TOURNAMENTS: dict[str, tuple[str, ...]] = {
    # Exact 2026 top-level major-league tournaments discovered from Central
    # Data. Child phases are included by the bounded allSeries query;
    # challengers, academy, and showmatches are excluded.
    "LCS": ("827908", "828910", "829743"),
    "LCK": ("827842", "829037", "830184"),
    "LEC": ("827699", "828971", "829802"),
    "LPL": ("827802", "829116", "830263"),
    "CBLOL": ("827732", "829014", "830215"),
}

EVENT_FAMILIES = {
    "player-killed-player": "kill",
    "player-completed-destroyTower": "tower",
    "team-completed-destroyTower": "tower",
    "player-completed-destroyFortifier": "inhibitor",
    "team-completed-destroyFortifier": "inhibitor",
    "player-completed-slayDragon": "dragon",
    "team-completed-slayDragon": "dragon",
    "player-completed-slayBaron": "baron",
    "team-completed-slayBaron": "baron",
}
ELEMENTAL_DRAKE_SUFFIXES = (
    "CloudDrake",
    "ChemtechDrake",
    "HextechDrake",
    "InfernalDrake",
    "MountainDrake",
    "OceanDrake",
    "ElderDrake",
)


class GridMarketCohortError(RuntimeError):
    """Credential-free fail-closed cohort error."""


class GridMarketTransientError(GridMarketCohortError):
    """Retryable provider condition that must not become a cohort exclusion."""


class GridMarketQuotaStop(GridMarketCohortError):
    """Chronological whole-series boundary reached before the map maximum."""


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


def _safe_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in sorted(row.items())
        if not any(
            marker in str(key).lower()
            for marker in ("url", "token", "key", "secret", "signature")
        )
    }


def _catalog_provenance() -> dict[str, Any]:
    path = (
        Path.home()
        / ".codex"
        / "skills"
        / "query-grid-research"
        / "assets"
        / "grid-capability-catalog.v1.json"
    )
    if not path.is_file():
        raise GridMarketCohortError("GRID capability catalog is unavailable")
    value = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "catalog_version": value.get("catalog_version"),
        "catalog_sha256": _sha256_file(path),
        "endpoint_schema_hashes": {
            row["endpoint_id"]: row["schema_sha256"]
            for row in value.get("endpoints") or []
            if isinstance(row, Mapping)
            and row.get("endpoint_id")
            and row.get("schema_sha256")
        },
    }


class _CentralDataClient:
    """Small read-only client with conservative request pacing."""

    def __init__(self, key: str, minimum_interval_seconds: float = 1.7):
        self.key = key
        self.minimum_interval_seconds = minimum_interval_seconds
        self._last_call = 0.0
        self.player_cache: dict[str, dict[str, Any]] = {}

    def query(
        self, query: str, variables: Mapping[str, Any], *, retries: int = 4
    ) -> dict[str, Any]:
        for attempt in range(retries + 1):
            delay = self.minimum_interval_seconds - (time.monotonic() - self._last_call)
            if delay > 0:
                time.sleep(delay)
            self._last_call = time.monotonic()
            try:
                return _graphql(self.key, GRAPHQL_ENDPOINT, query, variables)
            except GridIngestError as exc:
                message = str(exc)
                if (
                    attempt >= retries
                    or "rate limit" not in message.lower()
                    and "ENHANCE_YOUR_CALM" not in message
                ):
                    raise
                time.sleep(12.0)
        raise AssertionError("unreachable")

    def players(self, provider_player_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        requested = sorted(set(str(value) for value in provider_player_ids))
        if any(not value.isdigit() for value in requested):
            raise GridMarketCohortError("provider player IDs must be numeric")
        missing = [value for value in requested if value not in self.player_cache]
        if missing:
            selections = [
                f"p{index}: player(id: $id{index}) {{ id externalLinks "
                "{dataProvider {name} externalEntity {id}}}"
                for index, _player_id in enumerate(missing)
            ]
            declarations = ", ".join(
                f"$id{index}: ID!" for index, _player_id in enumerate(missing)
            )
            variables = {
                f"id{index}": player_id
                for index, player_id in enumerate(missing)
            }
            data = self.query(
                f"query CohortPlayers({declarations}) {{"
                + " ".join(selections)
                + "}",
                variables,
            )
            for index, provider_id in enumerate(missing):
                player = data.get(f"p{index}") or {}
                self.player_cache[provider_id] = dict(player)
        return {
            provider_id: self.player_cache.get(provider_id, {})
            for provider_id in requested
        }


def discover_series(
    client: _CentralDataClient,
    *,
    league: str,
    tournament_ids: Sequence[str],
    maximum_rows: int,
    start_time: str | None = None,
    end_time: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Discover a bounded, chronologically ordered exact-tournament cohort."""
    if league not in LEAGUE_TOURNAMENTS:
        raise GridMarketCohortError(f"unregistered league: {league}")
    declared = set(LEAGUE_TOURNAMENTS[league])
    requested = tuple(str(value) for value in tournament_ids)
    if not requested or not set(requested).issubset(declared):
        raise GridMarketCohortError("tournament IDs are not in the exact league registry")
    query = """
    query CohortSeries(
      $after: String
      $ids: [ID!]!
      $start_time: String
      $end_time: String
    ) {
      allSeries(
        first: 50
        after: $after
        filter: {
          titleId: 3
          type: ESPORTS
          private: {equals: false}
          startTimeScheduled: {gte: $start_time, lte: $end_time}
          tournament: {id: {in: $ids}, includeChildren: {equals: true}}
        }
        orderBy: StartTimeScheduled
        orderDirection: ASC
      ) {
        pageInfo {hasNextPage endCursor}
        edges {
          node {
            id
            type
            startTimeScheduled
            updatedAt
            private
            tournament {id name parent {id name}}
            externalLinks {dataProvider {name} externalEntity {id}}
            teams {
              baseInfo {
                id
                name
                externalLinks {dataProvider {name} externalEntity {id}}
              }
            }
          }
        }
      }
    }
    """
    cursor: str | None = None
    rows: list[dict[str, Any]] = []
    cursor_receipts: list[dict[str, Any]] = []
    while len(rows) < maximum_rows:
        variables = {
            "after": cursor,
            "ids": list(requested),
            "start_time": start_time,
            "end_time": end_time,
        }
        data = client.query(query, variables)
        block = data.get("allSeries") or {}
        edges = block.get("edges") or []
        response_hash = _hash(block)
        page_info = block.get("pageInfo") or {}
        cursor_receipts.append(
            {
                "request_cursor": cursor,
                "response_sha256": response_hash,
                "row_count": len(edges),
                "has_next_page": bool(page_info.get("hasNextPage")),
                "end_cursor": page_info.get("endCursor"),
            }
        )
        for edge in edges:
            node = (edge or {}).get("node") or {}
            if (
                str(node.get("type") or "") != "ESPORTS"
                or node.get("private") is True
                or not str(node.get("id") or "").isdigit()
            ):
                continue
            teams = [
                (row or {}).get("baseInfo") or {}
                for row in node.get("teams") or []
                if isinstance(row, Mapping)
            ]
            if len(teams) != 2 or any(not str(team.get("id") or "") for team in teams):
                continue
            rows.append(dict(node))
            if len(rows) >= maximum_rows:
                break
        if len(rows) >= maximum_rows or not page_info.get("hasNextPage"):
            stop_reason = (
                "maximum_rows_reached"
                if len(rows) >= maximum_rows
                else "provider_end_of_pages"
            )
            break
        cursor = page_info.get("endCursor")
        if not cursor:
            stop_reason = "missing_next_cursor"
            break
    else:
        stop_reason = "maximum_rows_reached"
    receipt = {
        "endpoint": "central_data",
        "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "league": league,
        "tournament_ids": list(requested),
        "page_size": 50,
        "maximum_rows": maximum_rows,
        "start_time": start_time,
        "end_time": end_time,
        "pages": cursor_receipts,
        "stop_reason": stop_reason,
        "retrieved_at": _utc_now(),
        "mutations_used": False,
    }
    return rows[:maximum_rows], receipt


def _download_file(
    *,
    key: str,
    series_id: str,
    file_row: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    file_id = str(file_row.get("id") or "")
    if str(file_row.get("status") or "") != "ready" or not file_id:
        raise GridMarketCohortError("requested GRID file is not uniquely ready")
    url = str(file_row.get("fullURL") or "")
    if not url:
        raise GridMarketCohortError("requested GRID file has no download capability")
    suffix = ".jsonl.zip" if file_id == "events-grid" else ".json"
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{series_id}-{file_id}.", suffix=suffix, dir=raw_dir
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        time.sleep(0.5)
        downloaded = False
        for attempt in range(4):
            if _download(url, key, temporary):
                downloaded = True
                break
            if attempt < 3:
                time.sleep(15.0 * (attempt + 1))
        if not downloaded:
            raise GridMarketTransientError(
                "GRID file download remained rate-limited after bounded backoff"
            )
        source_sha = _sha256_file(temporary)
        destination = raw_dir / f"{series_id}-{file_id}-{source_sha}{suffix}"
        if destination.exists():
            if _sha256_file(destination) != source_sha:
                raise GridMarketCohortError("content-addressed raw path conflict")
            temporary.unlink()
        else:
            os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "retrieved_at": _utc_now(),
        "scope": "private_personal_research_only",
        "provider_series_id": series_id,
        "file_id": file_id,
        "file_metadata": _safe_metadata(file_row),
        "raw_path": str(destination),
        "raw_sha256": source_sha,
        "raw_bytes": destination.stat().st_size,
        "credentials_serialized": False,
        "signed_url_retained": False,
        "mutations_used": False,
    }
    receipt["receipt_sha256"] = _hash(receipt)
    receipt_path = (
        root
        / "receipts"
        / f"retrieval-{series_id}-{file_id}-{receipt['receipt_sha256']}.json"
    )
    write_immutable_receipt(receipt_path, receipt)
    return {**receipt, "receipt_path": str(receipt_path)}


def _existing_download(
    *, series_id: str, file_id: str, root: Path
) -> dict[str, Any] | None:
    matches: list[dict[str, Any]] = []
    for receipt_path in sorted(
        (root / "receipts").glob(f"retrieval-{series_id}-{file_id}-*.json")
    ):
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            raw_path = Path(str(receipt.get("raw_path") or ""))
            if (
                receipt.get("schema_version") == RECEIPT_SCHEMA_VERSION
                and receipt.get("provider_series_id") == series_id
                and receipt.get("file_id") == file_id
                and raw_path.is_file()
                and _sha256_file(raw_path) == receipt.get("raw_sha256")
            ):
                matches.append({**receipt, "receipt_path": str(receipt_path)})
        except (OSError, ValueError, TypeError):
            continue
    unique = {row["raw_sha256"]: row for row in matches}
    return next(iter(unique.values())) if len(unique) == 1 else None


def retrieve_series_files(
    *,
    key: str,
    series_id: str,
    root: Path,
    maximum_summaries: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Retrieve only compact event/final files, never replay or Riot telemetry."""
    rows = _file_list(key, series_id)
    summary_ids = sorted(
        str(row.get("id") or "")
        for row in rows
        if str(row.get("id") or "").startswith("state-summary-riot-game-")
        and str(row.get("status") or "") == "ready"
    )
    required = {
        file_id: [
            row
            for row in rows
            if str(row.get("id") or "") == file_id
            and str(row.get("status") or "") == "ready"
        ]
        for file_id in ("events-grid", "state-grid")
    }
    # Fail before retaining any bytes when the compact evidence bundle is not
    # structurally available.
    if any(len(matches) != 1 for matches in required.values()) or not summary_ids:
        raise GridMarketCohortError("compact GRID evidence bundle is incomplete")
    if maximum_summaries is not None and len(summary_ids) > maximum_summaries:
        raise GridMarketQuotaStop("series would cross the league map maximum")
    wanted = {"events-grid", "state-grid", *summary_ids}
    receipts: dict[str, dict[str, Any]] = {}
    for file_id in sorted(wanted):
        existing = _existing_download(
            series_id=series_id, file_id=file_id, root=root
        )
        if existing is not None:
            receipts[file_id] = existing
            continue
        matches = [
            row
            for row in rows
            if str(row.get("id") or "") == file_id
            and str(row.get("status") or "") == "ready"
        ]
        if len(matches) != 1:
            continue
        receipts[file_id] = _download_file(
            key=key, series_id=series_id, file_row=matches[0], root=root
        )
    return receipts


def _external_ids(
    links: Iterable[Mapping[str, Any]], provider: str
) -> list[str]:
    values = []
    for link in links:
        data_provider = (link.get("dataProvider") or {}).get("name")
        external_id = (link.get("externalEntity") or {}).get("id")
        if str(data_provider or "") == provider and external_id is not None:
            values.append(str(external_id))
    return sorted(set(values))


def _objective(team: Mapping[str, Any], name: str) -> tuple[int, bool] | None:
    row = (team.get("objectives") or {}).get(name)
    if not isinstance(row, Mapping):
        return None
    kills = row.get("kills")
    first = row.get("first")
    if isinstance(kills, bool) or not isinstance(kills, int) or not isinstance(first, bool):
        return None
    return kills, first


def _summary_labels(summary: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    if (
        summary.get("endOfGameResult") is not None
        and summary.get("endOfGameResult") != "GameComplete"
    ):
        blockers.append("outcome.riot_game_not_complete")
    if (
        not isinstance(summary.get("gameEndTimestamp"), int)
        or not isinstance(summary.get("gameDuration"), int)
        or int(summary.get("gameDuration") or 0) <= 0
    ):
        blockers.append("outcome.riot_game_end_missing")
    teams = summary.get("teams") or []
    if (
        len(teams) != 2
        or {team.get("teamId") for team in teams if isinstance(team, Mapping)}
        != {100, 200}
    ):
        blockers.append("identity.riot_teams_not_exactly_100_200")
        return {}, blockers
    by_id = {int(team["teamId"]): team for team in teams}
    winners = [
        team_id
        for team_id, team in by_id.items()
        if team.get("win") is True or str(team.get("win") or "").lower() == "win"
    ]
    if len(winners) != 1:
        blockers.append("outcome.riot_winner_not_unique")

    def total(name: str) -> int | None:
        values = [_objective(by_id[team_id], name) for team_id in (100, 200)]
        return None if any(value is None for value in values) else sum(value[0] for value in values if value)

    def first(name: str) -> int | None:
        values = [
            team_id
            for team_id in (100, 200)
            if (_objective(by_id[team_id], name) or (0, False))[1]
        ]
        return values[0] if len(values) == 1 else None

    labels = {
        "first_tower": first("tower"),
        "first_inhibitor": first("inhibitor"),
        "first_dragon": first("dragon"),
        "total_dragons": total("dragon"),
        "first_baron": first("baron"),
        "total_barons": total("baron"),
        "first_blood": first("champion"),
        "total_inhibitor_destructions": total("inhibitor"),
    }
    for target, value in labels.items():
        if target.startswith("total_") and value is None:
            blockers.append(f"label.{target}_missing")
    return labels, blockers


def _grid_labels(game: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    teams = game.get("teams") or []
    if (
        game.get("finished") is not True
        or len(teams) != 2
        or {str(team.get("side") or "") for team in teams} != {"blue", "red"}
    ):
        return {}, ["outcome.grid_final_state_incomplete"]
    by_riot = {
        100 if team["side"] == "blue" else 200: team
        for team in teams
    }

    def objective(team_id: int, name: str) -> tuple[int, bool | None] | None:
        rows = [
            row
            for row in by_riot[team_id].get("objectives") or []
            if str(row.get("type") or "") == name
        ]
        if len(rows) != 1:
            return None
        count = rows[0].get("completionCount")
        first = rows[0].get("completedFirst")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or first is not None
            and not isinstance(first, bool)
        ):
            return None
        return count, first

    def total(name: str) -> int | None:
        values = [objective(team_id, name) for team_id in (100, 200)]
        return None if any(value is None for value in values) else sum(value[0] for value in values if value)

    def first(name: str) -> int | None:
        values = [
            team_id
            for team_id in (100, 200)
            if (objective(team_id, name) or (0, None))[1] is True
        ]
        return values[0] if len(values) == 1 else None

    labels = {
        "first_tower": first("destroyTower"),
        "first_inhibitor": first("destroyFortifier"),
        "first_dragon": first("slayDragon"),
        "total_dragons": total("slayDragon"),
        "first_baron": first("slayBaron"),
        "total_barons": total("slayBaron"),
        "first_blood": (
            next(
                (
                    team_id
                    for team_id in (100, 200)
                    if by_riot[team_id].get("firstKill") is True
                ),
                None,
            )
            if any("firstKill" in by_riot[team_id] for team_id in (100, 200))
            else None
        ),
        "total_inhibitor_destructions": total("destroyFortifier"),
    }
    return labels, []


def _iter_grid_events(path: Path) -> Iterator[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".jsonl")]
        if len(members) != 1:
            raise GridMarketCohortError("events-grid archive must have one JSONL member")
        with archive.open(members[0]) as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                transaction = json.loads(raw)
                if not isinstance(transaction, Mapping):
                    continue
                for event in transaction.get("events") or []:
                    if isinstance(event, dict):
                        event["_provider_sequence"] = transaction.get("sequenceNumber")
                        yield event


def index_grid_events(path: Path) -> dict[str, Any]:
    """Parse one series archive once and index immutable events by game ID."""
    events = list(_iter_grid_events(path))
    by_game: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        for game in (event.get("seriesState") or {}).get("games") or []:
            if not isinstance(game, Mapping):
                continue
            game_id = str(game.get("id") or "")
            if game_id:
                by_game.setdefault(game_id, []).append(event)
    sequences = sorted(
        {
            int(event["_provider_sequence"])
            for event in events
            if isinstance(event.get("_provider_sequence"), int)
            and not isinstance(event.get("_provider_sequence"), bool)
        }
    )
    return {
        "events_by_game": by_game,
        "provider_sequences": sequences,
        "archive_event_count": len(events),
    }


def _event_clock(event: Mapping[str, Any], provider_game_id: str) -> int | None:
    games = [
        game
        for game in (event.get("seriesState") or {}).get("games") or []
        if isinstance(game, Mapping) and str(game.get("id") or "") == provider_game_id
    ]
    if len(games) != 1:
        return None
    seconds = (games[0].get("clock") or {}).get("currentSeconds")
    return seconds if isinstance(seconds, int) and not isinstance(seconds, bool) else None


def _event_team(
    event: Mapping[str, Any],
    provider_team_to_riot: Mapping[str, int],
) -> int | None:
    state = (event.get("actor") or {}).get("state") or {}
    side = str(state.get("side") or "").lower()
    if side == "blue":
        return 100
    if side == "red":
        return 200
    provider_team_id = str(
        state.get("teamId") or (event.get("actor") or {}).get("id") or ""
    )
    return provider_team_to_riot.get(provider_team_id)


def extract_checkpoints(
    events_path: Path,
    *,
    provider_game_id: str,
    provider_team_to_riot: Mapping[str, int] | None = None,
    checkpoints: Sequence[int] = CHECKPOINT_MINUTES,
    event_index: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract cumulative event evidence with conservative whole-second cutoff."""
    canonical: dict[str, tuple[str, dict[str, Any]]] = {}
    duplicates = 0
    conflicts: list[str] = []
    missing_clock = 0
    missing_team = 0
    provider_team_to_riot = dict(provider_team_to_riot or {})
    indexed = dict(event_index or index_grid_events(events_path))
    candidate_events = list(
        (indexed.get("events_by_game") or {}).get(provider_game_id) or []
    )
    sequences = list(indexed.get("provider_sequences") or [])
    sequence_gaps = [
        [left + 1, right - 1]
        for left, right in zip(sequences, sequences[1:])
        if right > left + 1
    ]
    has_aggregate_dragon = any(
        event.get("type") in {
            "player-completed-slayDragon",
            "team-completed-slayDragon",
        }
        and provider_game_id
        in {
            str(game.get("id") or "")
            for game in (event.get("seriesState") or {}).get("games") or []
            if isinstance(game, Mapping)
        }
        for event in candidate_events
    )
    for event in candidate_events:
        event_type = str(event.get("type") or "")
        family = EVENT_FAMILIES.get(event_type)
        if (
            family is None
            and not has_aggregate_dragon
            and event_type.startswith(("player-completed-slay", "team-completed-slay"))
            and event_type.endswith(ELEMENTAL_DRAKE_SUFFIXES)
        ):
            family = "dragon"
        if family is None:
            continue
        event_game_ids = {
            str(game.get("id") or "")
            for game in (event.get("seriesState") or {}).get("games") or []
            if isinstance(game, Mapping)
        }
        if provider_game_id not in event_game_ids:
            continue
        event_id = str(event.get("id") or "")
        if not event_id:
            conflicts.append("missing-event-id")
            continue
        event_hash = _hash({key: value for key, value in event.items() if key != "_provider_sequence"})
        if event_id in canonical:
            if canonical[event_id][0] == event_hash:
                duplicates += 1
            else:
                conflicts.append(event_id)
            continue
        clock = _event_clock(event, provider_game_id)
        team = _event_team(event, provider_team_to_riot)
        if clock is None:
            missing_clock += 1
        if team is None:
            missing_team += 1
        canonical[event_id] = (
            event_hash,
            {"id": event_id, "family": family, "second": clock, "team_id": team},
        )
    evidence = [value[1] for value in canonical.values()]
    blockers = []
    if conflicts:
        blockers.append("events.conflicting_or_missing_event_identity")
    if missing_clock:
        blockers.append("checkpoint.event_clock_missing")
    if missing_team:
        blockers.append("identity.event_actor_team_missing")

    def values(cutoff_minute: int | None) -> dict[str, Any]:
        # Whole-second timestamps cannot prove an event at exactly XX:00.000 was
        # known by the boundary. Strictly exclude the equal second.
        rows = [
            row
            for row in evidence
            if row["second"] is not None
            and (
                cutoff_minute is None
                or int(row["second"]) < int(cutoff_minute) * 60
            )
        ]

        def family_rows(name: str) -> list[dict[str, Any]]:
            return sorted(
                (row for row in rows if row["family"] == name),
                key=lambda row: (row["second"], row["id"]),
            )

        def first(name: str) -> int | None:
            selected = family_rows(name)
            return selected[0]["team_id"] if selected else None

        return {
            "current_kills": len(family_rows("kill")),
            "first_blood": first("kill"),
            "first_tower": first("tower"),
            "first_inhibitor": first("inhibitor"),
            "total_inhibitor_destructions": len(family_rows("inhibitor")),
            "first_dragon": first("dragon"),
            "total_dragons": len(family_rows("dragon")),
            "first_baron": first("baron"),
            "total_barons": len(family_rows("baron")),
        }

    return {
        "status": "eligible" if not blockers else "unavailable",
        "blockers": blockers,
        "timestamp_precision": "whole_second",
        "cutoff_rule": "event_second_strictly_less_than_checkpoint_second",
        "checkpoints": [
            {"minute": minute, "values": values(minute)} for minute in checkpoints
        ],
        "final_event_values": values(None),
        "event_receipt": {
            "canonical_target_event_count": len(evidence),
            "duplicate_event_count": duplicates,
            "conflicting_event_ids": sorted(set(conflicts)),
            "missing_clock_count": missing_clock,
            "missing_team_count": missing_team,
            "provider_sequence_min": sequences[0] if sequences else None,
            "provider_sequence_max": sequences[-1] if sequences else None,
            "provider_sequence_gap_count": sum(
                right - left + 1 for left, right in sequence_gaps
            ),
            "provider_sequence_gap_intervals": sequence_gaps,
            "sequence_gap_interpretation": (
                "event_cadence_unavailable; target cumulative counts remain "
                "eligible only when separately verified final totals agree"
            ),
            "derived_evidence_sha256": _hash(sorted(evidence, key=lambda row: row["id"])),
        },
    }


def _player_crosswalk(
    client: _CentralDataClient,
    *,
    provider_player_ids: Sequence[str],
    provider_player_to_riot_team: Mapping[str, int],
    summary: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    unique_ids = sorted(set(str(value) for value in provider_player_ids if value))
    if len(unique_ids) != 10:
        return [], ["identity.provider_players_not_exactly_ten"]
    data = client.players(unique_ids)
    participants = summary.get("participants") or []
    by_summoner: dict[str, list[Mapping[str, Any]]] = {}
    for row in participants:
        if isinstance(row, Mapping) and row.get("summonerId") is not None:
            by_summoner.setdefault(str(row["summonerId"]), []).append(row)
    crosswalk: list[dict[str, Any]] = []
    blockers: list[str] = []
    used_puuids: set[str] = set()
    for provider_id in unique_ids:
        player = data.get(provider_id) or {}
        if str(player.get("id") or "") != provider_id:
            blockers.append("identity.central_player_missing")
            continue
        live_ids = _external_ids(player.get("externalLinks") or [], "LOL_LIVE")
        matches = [
            participant
            for live_id in live_ids
            for participant in by_summoner.get(live_id, [])
        ]
        deduped = {str(row.get("puuid") or ""): row for row in matches}
        if len(deduped) != 1 or "" in deduped:
            blockers.append("identity.player_live_id_not_one_to_one")
            continue
        puuid, participant = next(iter(deduped.items()))
        if puuid in used_puuids:
            blockers.append("identity.riot_puuid_reused")
            continue
        expected_team_id = provider_player_to_riot_team.get(provider_id)
        if expected_team_id not in {100, 200} or participant.get("teamId") != expected_team_id:
            blockers.append("identity.player_team_side_conflict")
            continue
        used_puuids.add(puuid)
        crosswalk.append(
            {
                "provider_player_id": provider_id,
                "riot_summoner_id": str(participant["summonerId"]),
                "riot_puuid": puuid,
                "riot_team_id": participant.get("teamId"),
                "participant_id": participant.get("participantId"),
            }
        )
    if len(participants) != 10:
        blockers.append("identity.riot_participants_not_exactly_ten")
    if len(crosswalk) != 10:
        blockers.append("identity.player_crosswalk_incomplete")
    return sorted(crosswalk, key=lambda row: row["provider_player_id"]), sorted(
        set(blockers)
    )


def build_game_record(
    *,
    client: _CentralDataClient,
    league: str,
    series: Mapping[str, Any],
    grid_state_path: Path,
    summary_path: Path,
    events_path: Path,
    source_receipts: Mapping[str, Any],
    event_index: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state = json.loads(grid_state_path.read_text(encoding="utf-8"))
    state = state.get("seriesState") or {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    blockers: list[str] = []
    if str(state.get("id") or "") != str(series.get("id") or ""):
        blockers.append("identity.series_id_conflict")
    platform = str(summary.get("platformId") or "")
    riot_game_id = str(summary.get("gameId") or "")
    expected_live_id = f"{platform}_{riot_game_id}"
    games = [
        game
        for game in state.get("games") or []
        if expected_live_id
        in _external_ids(game.get("externalLinks") or [], "LOL_LIVE")
    ]
    if len(games) != 1:
        return {
            "schema_version": SCHEMA_VERSION,
            "league": league,
            "provider_series_id": str(series.get("id") or ""),
            "status": "quarantined",
            "blockers": ["identity.provider_riot_game_not_one_to_one"],
        }
    game = games[0]
    provider_game_id = str(game.get("id") or "")
    provider_teams = {
        str(team.get("id") or ""): team
        for team in game.get("teams") or []
        if isinstance(team, Mapping)
    }
    series_team_ids = {
        str(((row or {}).get("baseInfo") or {}).get("id") or "")
        for row in series.get("teams") or []
    }
    if len(provider_teams) != 2 or set(provider_teams) != series_team_ids:
        blockers.append("identity.provider_team_set_conflict")
    side_crosswalk = []
    provider_team_to_riot: dict[str, int] = {}
    for provider_id, team in sorted(provider_teams.items()):
        side = str(team.get("side") or "")
        riot_team_id = 100 if side == "blue" else 200 if side == "red" else None
        if riot_team_id is None:
            blockers.append("identity.provider_team_side_invalid")
        else:
            provider_team_to_riot[provider_id] = riot_team_id
        side_crosswalk.append(
            {
                "provider_team_id": provider_id,
                "riot_team_id": riot_team_id,
                "side": side,
            }
        )
    player_ids = [
        str(player.get("id") or "")
        for team in provider_teams.values()
        for player in team.get("players") or []
        if isinstance(player, Mapping)
    ]
    provider_player_to_riot_team = {
        str(player.get("id") or ""): (
            100 if str(team.get("side") or "") == "blue" else 200
        )
        for team in provider_teams.values()
        for player in team.get("players") or []
        if isinstance(player, Mapping)
        and str(team.get("side") or "") in {"blue", "red"}
    }
    player_crosswalk, player_blockers = _player_crosswalk(
        client,
        provider_player_ids=player_ids,
        provider_player_to_riot_team=provider_player_to_riot_team,
        summary=summary,
    )
    blockers.extend(player_blockers)
    summary_labels, summary_blockers = _summary_labels(summary)
    grid_labels, grid_blockers = _grid_labels(game)
    blockers.extend(summary_blockers)
    blockers.extend(grid_blockers)
    for target in TARGETS:
        if (
            grid_labels.get(target) is not None
            and summary_labels.get(target) != grid_labels.get(target)
        ):
            blockers.append(f"label.grid_riot_{target}_conflict")
    checkpoint_evidence = extract_checkpoints(
        events_path,
        provider_game_id=provider_game_id,
        provider_team_to_riot=provider_team_to_riot,
        event_index=event_index,
    )
    blockers.extend(checkpoint_evidence["blockers"])
    event_final = checkpoint_evidence["final_event_values"]
    event_to_label = {
        target: event_final.get(target)
        for target in TARGETS
    }
    for target in TARGETS:
        if event_to_label.get(target) != summary_labels.get(target):
            blockers.append(f"label.events_riot_{target}_conflict")
    if event_final.get("current_kills") != sum(
        int(row.get("kills") or 0) for row in summary.get("participants") or []
    ):
        blockers.append("label.events_riot_total_kills_conflict")
    status = "verified" if not blockers else "quarantined"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "blockers": sorted(set(blockers)),
        "league": league,
        "patch": str(summary.get("gameVersion") or ""),
        "chronology": {
            "series_start_time_scheduled": series.get("startTimeScheduled"),
            "game_start_timestamp_ms": summary.get("gameStartTimestamp"),
            "game_end_timestamp_ms": summary.get("gameEndTimestamp"),
            "game_duration_seconds": summary.get("gameDuration"),
        },
        "identity": {
            "provider_series_id": str(series.get("id") or ""),
            "provider_series_metadata_sha256": _hash(series),
            "provider_series_external_ids": {
                "LOL": _external_ids(series.get("externalLinks") or [], "LOL")
            },
            "provider_tournament": {
                "id": str((series.get("tournament") or {}).get("id") or ""),
                "name": str((series.get("tournament") or {}).get("name") or ""),
                "parent_id": str(
                    ((series.get("tournament") or {}).get("parent") or {}).get("id")
                    or ""
                ),
            },
            "provider_game_id": provider_game_id,
            "riot_platform_id": platform,
            "riot_game_id": riot_game_id,
            "provider_game_external_id": expected_live_id,
            "teams": side_crosswalk,
            "players": player_crosswalk,
        },
        "labels": summary_labels if status == "verified" else {},
        "outcomes": (
            {
                "total_kills": event_final.get("current_kills"),
                "game_duration_seconds": summary.get("gameDuration"),
            }
            if status == "verified"
            else {}
        ),
        "checkpoints": (
            checkpoint_evidence["checkpoints"] if status == "verified" else []
        ),
        "event_receipt": checkpoint_evidence["event_receipt"],
        "source_receipts": dict(source_receipts),
    }


def _paths_from_receipts(
    receipts: Mapping[str, Mapping[str, Any]]
) -> tuple[Path, Path, list[Path]]:
    events = receipts.get("events-grid")
    state = receipts.get("state-grid")
    summaries = [
        Path(row["raw_path"])
        for file_id, row in receipts.items()
        if file_id.startswith("state-summary-riot-game-")
    ]
    if not events or not state:
        raise GridMarketCohortError("compact GRID event/state files unavailable")
    return Path(events["raw_path"]), Path(state["raw_path"]), sorted(summaries)


def build_cohort(
    *,
    root: Path = DEFAULT_ROOT,
    quota_per_league: int = 250,
    maximum_series_per_league: int = 180,
    leagues: Sequence[str] = tuple(LEAGUE_TOURNAMENTS),
    start_time: str | None = None,
    end_time: str | None = None,
    key: str | None = None,
) -> dict[str, Any]:
    if not 1 <= quota_per_league <= 500:
        raise GridMarketCohortError("quota_per_league must be within 1..500")
    if maximum_series_per_league < 1:
        raise GridMarketCohortError("maximum_series_per_league must be positive")
    key = key or _api_key()
    client = _CentralDataClient(key)
    catalog = _catalog_provenance()
    verified: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    discovery_receipts: list[dict[str, Any]] = []
    for league in leagues:
        candidates, discovery = discover_series(
            client,
            league=league,
            tournament_ids=LEAGUE_TOURNAMENTS[league],
            maximum_rows=maximum_series_per_league,
            start_time=start_time,
            end_time=end_time,
        )
        discovery_receipts.append(discovery)
        league_count = 0
        for series in candidates:
            if league_count >= quota_per_league:
                break
            series_id = str(series.get("id") or "")
            try:
                receipts = retrieve_series_files(
                    key=key,
                    series_id=series_id,
                    root=root,
                    maximum_summaries=quota_per_league - league_count,
                )
                events_path, state_path, summary_paths = _paths_from_receipts(receipts)
                if not summary_paths:
                    quarantined.append(
                        {
                            "league": league,
                            "provider_series_id": series_id,
                            "status": "quarantined",
                            "blockers": ["source.riot_summary_missing"],
                        }
                    )
                    continue
                receipt_refs = {
                    file_id: {
                        "receipt_path": row["receipt_path"],
                        "receipt_sha256": row["receipt_sha256"],
                        "raw_sha256": row["raw_sha256"],
                    }
                    for file_id, row in receipts.items()
                }
                event_index = index_grid_events(events_path)
                for summary_path in summary_paths:
                    if league_count >= quota_per_league:
                        break
                    record = build_game_record(
                        client=client,
                        league=league,
                        series=series,
                        grid_state_path=state_path,
                        summary_path=summary_path,
                        events_path=events_path,
                        source_receipts=receipt_refs,
                        event_index=event_index,
                    )
                    if record["status"] == "verified":
                        verified.append(record)
                        league_count += 1
                    else:
                        quarantined.append(record)
            except GridMarketTransientError:
                raise
            except GridMarketQuotaStop:
                break
            except (GridMarketCohortError, GridIngestError, OSError, ValueError) as exc:
                quarantined.append(
                    {
                        "league": league,
                        "provider_series_id": series_id,
                        "status": "quarantined",
                        "blockers": [f"processing.{type(exc).__name__}"],
                    }
                )
    counts_by_league = Counter(row["league"] for row in verified)
    labels_by_league = {
        league: {
            target: sum(target in row.get("labels", {}) for row in verified if row["league"] == league)
            for target in TARGETS
        }
        for league in leagues
    }
    blocker_counts = Counter(
        blocker for row in quarantined for blocker in row.get("blockers") or []
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "scope": {
            "privacy": "private_personal_research_only",
            "data_foundation_only": True,
            "models_trained": False,
            "probabilities_authorized": False,
            "fair_odds_authorized": False,
            "edge_authorized": False,
            "expected_value_authorized": False,
            "publication_authorized": False,
            "mutations_used": False,
        },
        "configuration": {
            "registry_version": REGISTRY_VERSION,
            "quota_per_league": quota_per_league,
            "maximum_series_per_league": maximum_series_per_league,
            "leagues": list(leagues),
            "start_time": start_time,
            "end_time": end_time,
            "exact_tournament_registry": {
                league: list(LEAGUE_TOURNAMENTS[league]) for league in leagues
            },
            "checkpoints": list(CHECKPOINT_MINUTES),
            "whole_second_cutoff_rule": "event_second_strictly_less_than_checkpoint_second",
            "targets": list(TARGETS),
            "total_towers_excluded": True,
        },
        "catalog_provenance": catalog,
        "discovery_receipts": discovery_receipts,
        "coverage": {
            "verified_maps_total": len(verified),
            "verified_maps_by_league": dict(sorted(counts_by_league.items())),
            "verified_labels_by_league": labels_by_league,
            "quarantined_records_total": len(quarantined),
            "blocker_counts": dict(sorted(blocker_counts.items())),
        },
        "verified_games": verified,
        "quarantine": quarantined,
        "authority": {
            "cohort_eligible_for_future_evaluation": len(verified) > 0,
            "model_or_betting_authority": "unavailable",
            "blockers": [
                "evaluation.not_run",
                "calibration.not_run",
                "market_benchmark.not_collected",
            ],
        },
    }
    manifest["manifest_sha256"] = _hash(manifest)
    path = root / "manifests" / f"objective-cohort-{manifest['manifest_sha256']}.json"
    write_immutable_receipt(path, manifest)
    return {**manifest, "manifest_path": str(path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--quota-per-league", type=int, default=250)
    parser.add_argument("--maximum-series-per-league", type=int, default=180)
    parser.add_argument("--start-time")
    parser.add_argument("--end-time")
    parser.add_argument(
        "--league",
        action="append",
        choices=sorted(LEAGUE_TOURNAMENTS),
        dest="leagues",
    )
    args = parser.parse_args(argv)
    manifest = build_cohort(
        root=args.root,
        quota_per_league=args.quota_per_league,
        maximum_series_per_league=args.maximum_series_per_league,
        leagues=args.leagues or tuple(LEAGUE_TOURNAMENTS),
        start_time=args.start_time,
        end_time=args.end_time,
    )
    print(
        json.dumps(
            {
                "manifest_path": manifest["manifest_path"],
                "manifest_sha256": manifest["manifest_sha256"],
                "coverage": manifest["coverage"],
                "authority": manifest["authority"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
