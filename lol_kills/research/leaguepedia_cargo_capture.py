"""Capture Leaguepedia Cargo source arrays for the OE series crosswalk.

The capture lane is local and research-only.  It uses only the public Cargo
HTTPS endpoint.  It sends no credentials.  Each request is bounded by a
date window and a strict Cargo row limit.  A response whose row count reaches
the limit is rejected because truncation cannot be disproved.

The module keeps exact response bytes in a stable cache.  A later run can
resume from verified cache files.  It assembles deterministic
``ScoreboardGames.json``, ``MatchSchedule.json``, and ``Tournaments.json``
files and writes a self-hashed manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from lol_kills.net import require_https_url


SCHEMA_VERSION = "scryglass:leaguepedia-cargo-capture:v1"
CARGO_ROOT = "https://lol.fandom.com/wiki/Special:CargoExport"
CARGO_HOSTS = frozenset({"lol.fandom.com"})
USER_AGENT = "Scryglass-research/leaguepedia-series-crosswalk-v1"
MAX_CARGO_LIMIT = 500
MAX_RESPONSE_BYTES = 64 * 1024 * 1024
DEFAULT_WINDOW_DAYS = 1
DEFAULT_TIMEOUT_SECONDS = 90.0

SCOREBOARD_FIELDS = (
    "GameId",
    "DateTime_UTC",
    "Team1",
    "Team2",
    "Patch",
    "Tournament",
    "OverviewPage",
)
SCHEDULE_FIELDS = (
    "MatchId",
    "DateTime_UTC",
    "Team1",
    "Team2",
    "Patch",
    "OverviewPage",
    "HasTime",
    "BestOf",
    "Tab",
    "Winner",
)
TOURNAMENT_FIELDS = (
    "Name",
    "OverviewPage",
    "DateStart",
    "Date",
    "Region",
    "League",
    "TournamentLevel",
    "IsOfficial",
)
TABLES = ("ScoreboardGames", "MatchSchedule", "Tournaments")
FetchBytes = Callable[[str, Mapping[str, str]], bytes]


class CargoCaptureError(RuntimeError):
    """Raised when a Cargo capture cannot prove complete source bytes."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CargoCaptureError("capture payload is not canonical JSON") from error


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _rfc3339(value: datetime) -> str:
    return _utc(value).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _date(value: str | date, *, field: str) -> date:
    if isinstance(value, datetime):
        return _utc(value).date()
    if isinstance(value, date):
        return value
    raw = str(value or "").strip()
    try:
        return date.fromisoformat(raw)
    except ValueError as error:
        raise CargoCaptureError(f"{field} must be an ISO date") from error


def _date_text(value: date) -> str:
    return value.isoformat()


def _date_window(value: str | date, *, field: str) -> tuple[str, str]:
    day = _date(value, field=field)
    return f"{day.isoformat()} 00:00:00", f"{(day + timedelta(days=1)).isoformat()} 00:00:00"


def _cargo_url(
    table: str,
    fields: Sequence[str],
    where: str,
    *,
    order_by: str,
    limit: int,
) -> str:
    if table not in TABLES:
        raise CargoCaptureError(f"unsupported Cargo table: {table}")
    params = {
        "tables": table,
        "fields": ",".join(f"{table}.{field}" for field in fields),
        "where": where,
        "order_by": order_by,
        "limit": str(limit),
        "format": "json",
    }
    url = CARGO_ROOT + "?" + urllib.parse.urlencode(params)
    try:
        return require_https_url(url, hosts=CARGO_HOSTS)
    except ValueError as error:
        raise CargoCaptureError("Cargo URL failed HTTPS host validation") from error


def _fetch_https(url: str, headers: Mapping[str, str], *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> bytes:
    try:
        safe_url = require_https_url(url, hosts=CARGO_HOSTS)
    except ValueError as error:
        raise CargoCaptureError("remote URL is outside the Leaguepedia HTTPS allowlist") from error
    request_headers = {"User-Agent": USER_AGENT, "Accept": "application/json", **dict(headers)}
    # Credentials and browser state are intentionally absent from this request.
    if any(key.casefold() in {"authorization", "cookie", "proxy-authorization"} for key in request_headers):
        raise CargoCaptureError("capture headers cannot contain credentials")
    request = urllib.request.Request(safe_url, headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except (OSError, TimeoutError) as error:
        raise CargoCaptureError("Leaguepedia Cargo request failed") from error
    if len(raw) > MAX_RESPONSE_BYTES:
        raise CargoCaptureError("Leaguepedia Cargo response exceeds the byte limit")
    return raw


def _write_atomic(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
        os.close(descriptor)
        temporary = Path(name)
        temporary.write_bytes(raw)
        os.replace(temporary, path)
    except OSError as error:
        raise CargoCaptureError(f"capture file cannot be written: {path}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_json_bytes(raw: bytes, *, label: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CargoCaptureError(f"Cargo response is not valid JSON: {label}") from error
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise CargoCaptureError(f"Cargo response must be an array of objects: {label}")
    return [dict(row) for row in value]


def _row_id(table: str, row: Mapping[str, Any]) -> str:
    if table == "ScoreboardGames":
        names = ("GameId", "game_id", "gameid")
    elif table == "MatchSchedule":
        names = ("MatchId", "match_id", "series_id")
    else:
        names = ("OverviewPage", "overview_page", "Name", "name")
    for name in names:
        value = str(row.get(name) or "").strip()
        if value:
            return value
    raise CargoCaptureError(f"{table} row has no stable identity")


def _validate_rows(table: str, rows: Sequence[Mapping[str, Any]], *, limit: int, label: str) -> list[dict[str, Any]]:
    if len(rows) >= limit:
        raise CargoCaptureError(
            f"Cargo response may be truncated: {label} returned {len(rows)} rows at limit {limit}"
        )
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for row in rows:
        row_id = _row_id(table, row)
        if row_id in seen:
            raise CargoCaptureError(f"duplicate {table} identity in response: {row_id}")
        seen.add(row_id)
        normalized.append(dict(row))
    return normalized


def _sort_rows(table: str, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    def text(row: Mapping[str, Any], *keys: str) -> str:
        for key in keys:
            value = str(row.get(key) or "").strip()
            if value:
                return value
        return ""

    if table == "ScoreboardGames":
        key = lambda row: (text(row, "DateTime UTC", "DateTime_UTC"), _row_id(table, row))
    elif table == "MatchSchedule":
        key = lambda row: (text(row, "DateTime UTC", "DateTime_UTC"), _row_id(table, row))
    else:
        key = lambda row: (
            text(row, "DateStart", "DateStart_UTC"),
            text(row, "OverviewPage", "Name"),
            _row_id(table, row),
        )
    return sorted((dict(row) for row in rows), key=key)


def _request_record_path(root: Path, table: str, request_url: str) -> tuple[Path, Path]:
    request_hash = _sha256_bytes(request_url.encode("utf-8"))
    directory = root / "raw" / table
    return directory / f"{request_hash}.json", directory / f"{request_hash}.meta.json"


def _load_cached_response(
    raw_path: Path,
    meta_path: Path,
    *,
    request_url: str,
    table: str,
    limit: int,
) -> tuple[bytes, dict[str, Any]]:
    if raw_path.is_symlink() or meta_path.is_symlink() or not raw_path.is_file() or not meta_path.is_file():
        raise CargoCaptureError(f"cached Cargo response is incomplete: {raw_path}")
    try:
        raw = raw_path.read_bytes()
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CargoCaptureError(f"cached Cargo response cannot be read: {raw_path}") from error
    if not isinstance(metadata, Mapping):
        raise CargoCaptureError(f"cached Cargo response metadata is invalid: {meta_path}")
    if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("status", "complete") == "rejected":
        raise CargoCaptureError(f"cached Cargo response was not accepted: {meta_path}")
    if metadata.get("request_url") != request_url or metadata.get("table") != table:
        raise CargoCaptureError(f"cached Cargo request identity changed: {raw_path}")
    if metadata.get("sha256") != _sha256_bytes(raw) or metadata.get("bytes") != len(raw):
        raise CargoCaptureError(f"cached Cargo response hash changed: {raw_path}")
    rows = _read_json_bytes(raw, label=str(raw_path))
    _validate_rows(table, rows, limit=limit, label=str(raw_path))
    if metadata.get("row_count") != len(rows):
        raise CargoCaptureError(f"cached Cargo response row count changed: {raw_path}")
    return raw, dict(metadata)


def _capture_one(
    *,
    root: Path,
    table: str,
    request_url: str,
    window_start: str,
    window_end_exclusive: str,
    limit: int,
    fetcher: FetchBytes,
    captured_at: str,
    resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_path, meta_path = _request_record_path(root, table, request_url)
    cache_hit = False
    if resume and (raw_path.exists() or meta_path.exists()):
        raw, metadata = _load_cached_response(
            raw_path,
            meta_path,
            request_url=request_url,
            table=table,
            limit=limit,
        )
        cache_hit = True
    else:
        raw = fetcher(request_url, {"User-Agent": USER_AGENT, "Accept": "application/json"})
        if not isinstance(raw, (bytes, bytearray)):
            raise CargoCaptureError("capture fetcher must return raw response bytes")
        raw = bytes(raw)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise CargoCaptureError("Leaguepedia Cargo response exceeds the byte limit")
        # Keep the exact response before semantic validation.  A rejected
        # response is still evidence and must remain auditable on disk.
        _write_atomic(raw_path, raw)
        try:
            rows = _read_json_bytes(raw, label=request_url)
            rows = _validate_rows(table, rows, limit=limit, label=request_url)
        except CargoCaptureError as error:
            rejected_metadata = {
                "schema_version": SCHEMA_VERSION,
                "table": table,
                "request_url": request_url,
                "window_start": window_start,
                "window_end_exclusive": window_end_exclusive,
                "retrieved_at": captured_at,
                "bytes": len(raw),
                "sha256": _sha256_bytes(raw),
                "status": "rejected",
                "error": str(error),
            }
            _write_atomic(meta_path, _canonical_json_bytes(rejected_metadata) + b"\n")
            raise
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "table": table,
            "request_url": request_url,
            "window_start": window_start,
            "window_end_exclusive": window_end_exclusive,
            "retrieved_at": captured_at,
            "bytes": len(raw),
            "sha256": _sha256_bytes(raw),
            "row_count": len(rows),
        }
        _write_atomic(meta_path, _canonical_json_bytes(metadata) + b"\n")
    rows = _validate_rows(table, _read_json_bytes(raw, label=str(raw_path)), limit=limit, label=str(raw_path))
    record = {
        **metadata,
        "path": str(raw_path.relative_to(root)),
        "cache_hit": cache_hit,
    }
    return rows, record


def _query_window(
    table: str,
    fields: Sequence[str],
    date_field: str,
    start: date,
    end_exclusive: date,
    *,
    limit: int,
) -> str:
    start_text = f"{start.isoformat()} 00:00:00"
    end_text = f"{end_exclusive.isoformat()} 00:00:00"
    where = f'{table}.{date_field} >= "{start_text}" AND {table}.{date_field} < "{end_text}"'
    return _cargo_url(
        table,
        fields,
        where,
        order_by=f"{table}.{date_field} ASC",
        limit=limit,
    )


def _query_tournaments_window(start: date, end_exclusive: date, *, limit: int) -> str:
    where = (
        f'Tournaments.Date >= "{start.isoformat()}" AND '
        f'Tournaments.Date < "{end_exclusive.isoformat()}" AND '
        f'Tournaments.DateStart <= "{(end_exclusive - timedelta(days=1)).isoformat()}"'
    )
    return _cargo_url(
        "Tournaments",
        TOURNAMENT_FIELDS,
        where,
        order_by="Tournaments.DateStart ASC",
        limit=limit,
    )


def _assembled_record(root: Path, table: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    filename = {
        "ScoreboardGames": "ScoreboardGames.json",
        "MatchSchedule": "MatchSchedule.json",
        "Tournaments": "Tournaments.json",
    }[table]
    path = root / "assembled" / filename
    raw = _canonical_json_bytes(_sort_rows(table, rows)) + b"\n"
    _write_atomic(path, raw)
    return {
        "table": table,
        "path": str(path.relative_to(root)),
        "bytes": len(raw),
        "sha256": _sha256_bytes(raw),
        "row_count": len(rows),
    }


def capture_leaguepedia_sources(
    *,
    start_date: str | date,
    end_date: str | date,
    root: Path,
    window_days: int = DEFAULT_WINDOW_DAYS,
    limit: int = MAX_CARGO_LIMIT,
    fetcher: FetchBytes | None = None,
    captured_at: datetime | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Capture bounded Leaguepedia Cargo arrays and return the manifest.

    ``end_date`` is inclusive.  ScoreboardGames, MatchSchedule, and
    Tournaments use bounded half-open windows.  Tournaments rows are selected
    by their end date and retain the start-date overlap constraint.
    The default fetcher is the HTTPS-only client.  Tests can inject a byte
    fetcher that receives the URL and the explicit request headers.
    """

    start = _date(start_date, field="start_date")
    end = _date(end_date, field="end_date")
    if end < start:
        raise CargoCaptureError("end_date is before start_date")
    if window_days < 1 or window_days > 31:
        raise CargoCaptureError("window_days must be between 1 and 31")
    if limit < 1 or limit > MAX_CARGO_LIMIT:
        raise CargoCaptureError(f"limit must be between 1 and {MAX_CARGO_LIMIT}")
    if (end - start).days > 3660:
        raise CargoCaptureError("capture date range is too large")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    fetch = fetcher or (lambda url, headers: _fetch_https(url, headers))
    observed_at = _utc(captured_at or datetime.now(timezone.utc))
    captured_text = _rfc3339(observed_at)
    all_rows: dict[str, list[dict[str, Any]]] = {table: [] for table in TABLES}
    response_records: list[dict[str, Any]] = []
    seen_ids: dict[str, set[str]] = {table: set() for table in TABLES}

    cursor = start
    while cursor <= end:
        window_end = min(end + timedelta(days=1), cursor + timedelta(days=window_days))
        queries = (
            (
                "ScoreboardGames",
                SCOREBOARD_FIELDS,
                "DateTime_UTC",
            ),
            (
                "MatchSchedule",
                SCHEDULE_FIELDS,
                "DateTime_UTC",
            ),
        )
        for table, fields, date_field in queries:
            request_url = _query_window(
                table,
                fields,
                date_field,
                cursor,
                window_end,
                limit=limit,
            )
            rows, record = _capture_one(
                root=root,
                table=table,
                request_url=request_url,
                window_start=f"{cursor.isoformat()}T00:00:00Z",
                window_end_exclusive=f"{window_end.isoformat()}T00:00:00Z",
                limit=limit,
                fetcher=fetch,
                captured_at=captured_text,
                resume=resume,
            )
            for row in rows:
                row_id = _row_id(table, row)
                if row_id in seen_ids[table]:
                    raise CargoCaptureError(f"duplicate {table} identity across windows: {row_id}")
                seen_ids[table].add(row_id)
                all_rows[table].append(row)
            response_records.append(record)
        tournament_url = _query_tournaments_window(cursor, window_end, limit=limit)
        tournament_rows, tournament_record = _capture_one(
            root=root,
            table="Tournaments",
            request_url=tournament_url,
            window_start=f"{cursor.isoformat()}T00:00:00Z",
            window_end_exclusive=f"{window_end.isoformat()}T00:00:00Z",
            limit=limit,
            fetcher=fetch,
            captured_at=captured_text,
            resume=resume,
        )
        for row in tournament_rows:
            row_id = _row_id("Tournaments", row)
            if row_id in seen_ids["Tournaments"]:
                raise CargoCaptureError(f"duplicate Tournaments identity across windows: {row_id}")
            seen_ids["Tournaments"].add(row_id)
            all_rows["Tournaments"].append(row)
        response_records.append(tournament_record)
        cursor = window_end

    assembled = {
        table: _assembled_record(root, table, all_rows[table])
        for table in TABLES
    }
    response_records.sort(key=lambda record: (record["table"], record["request_url"]))
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "captured_at": captured_text,
        "status": "complete_raw_capture",
        "authority": {
            "research_only": True,
            "public": False,
            "probability": False,
            "draft": False,
            "promotion": False,
            "deployment": False,
        },
        "source": {
            "name": "Leaguepedia Cargo",
            "host": "lol.fandom.com",
            "user_agent": USER_AGENT,
            "credentials_used": False,
        },
        "requested_window": {
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "window_days": window_days,
            "inclusive_end": True,
        },
        "query_contract": {
            "max_rows_per_request": limit,
            "truncation_policy": "reject_when_row_count_reaches_limit",
            "duplicate_policy": "reject_by_stable_table_identity",
            "cache_policy": "resume_only_after_raw_bytes_and_metadata_hash_verification",
            "response_format": "JSON array of objects",
        },
        "response_records": response_records,
        "assembled": assembled,
        "coverage": {
            "scoreboardgames_rows": len(all_rows["ScoreboardGames"]),
            "matchschedule_rows": len(all_rows["MatchSchedule"]),
            "tournaments_rows": len(all_rows["Tournaments"]),
            "requests": len(response_records),
            "cache_hits": sum(bool(record.get("cache_hit")) for record in response_records),
        },
    }
    payload["manifest_sha256"] = _sha256_bytes(_canonical_json_bytes(payload))
    manifest_path = root / "capture-manifest.json"
    _write_atomic(manifest_path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    payload["manifest_path"] = str(manifest_path)
    return payload


def verify_capture_manifest(payload: Mapping[str, Any]) -> None:
    """Verify a capture manifest self-hash and its research-only authority."""

    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
        raise CargoCaptureError("capture manifest schema is invalid")
    claimed = str(payload.get("manifest_sha256") or "").lower()
    if len(claimed) != 64 or any(char not in "0123456789abcdef" for char in claimed):
        raise CargoCaptureError("capture manifest self-hash is invalid")
    body = dict(payload)
    body.pop("manifest_sha256", None)
    body.pop("manifest_path", None)
    if _sha256_bytes(_canonical_json_bytes(body)) != claimed:
        raise CargoCaptureError("capture manifest self-hash does not match payload")
    authority = payload.get("authority")
    if not isinstance(authority, Mapping) or authority.get("public") is not False:
        raise CargoCaptureError("capture manifest grants public authority")
    if payload.get("status") != "complete_raw_capture":
        raise CargoCaptureError("capture manifest status is not complete")


__all__ = [
    "CargoCaptureError",
    "CARGO_ROOT",
    "SCHEMA_VERSION",
    "USER_AGENT",
    "capture_leaguepedia_sources",
    "verify_capture_manifest",
]
