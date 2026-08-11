"""Build the optional public match schedule from Leaguepedia Cargo.

This lane is display-only. A network or source failure must never block the
Oracle's Elixir ratings pack.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping

from lol_kills.etl.aliases import normalize_team


SCHEMA_VERSION = "scryglass:public-schedule:v1"
CARGO_ROOT = "https://lol.fandom.com/wiki/Special:CargoExport"
SOURCE_PAGE = "https://lol.fandom.com/wiki/Leaguepedia:Community"
USER_AGENT = "Scryglass-public-schedule/1.0"
FetchJson = Callable[[str], Any]


class PublicScheduleError(RuntimeError):
    """The optional display schedule is malformed or unavailable."""


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _rfc(value: datetime) -> str:
    return _utc(value).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _cargo_date(value: datetime) -> str:
    return _utc(value).strftime("%Y-%m-%d %H:%M:%S")


def _parse_cargo_time(value: Any) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        raise PublicScheduleError("Leaguepedia schedule time is empty")
    try:
        return datetime.strptime(raw[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError as error:
        raise PublicScheduleError("Leaguepedia schedule time is malformed") from error


def _row_value(row: Mapping[str, Any], field: str) -> Any:
    candidates = (
        field,
        field.replace("_", " "),
        field.replace("DateTime_UTC", "DateTime UTC"),
    )
    for candidate in candidates:
        if candidate in row:
            return row[candidate]
    return None


def _fetch_json(url: str) -> Any:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30.0) as response:
            raw = response.read()
    except (OSError, TimeoutError) as error:
        raise PublicScheduleError("Leaguepedia schedule request failed") from error
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise PublicScheduleError("Leaguepedia schedule response is invalid") from error


def _cargo_url(
    table: str,
    fields: tuple[str, ...],
    where: str,
    order_by: str,
    *,
    limit: int = 500,
) -> str:
    params = {
        "tables": table,
        "fields": ",".join(f"{table}.{field}" for field in fields),
        "where": where,
        "order_by": order_by,
        "limit": str(limit),
        "format": "json",
    }
    return CARGO_ROOT + "?" + urllib.parse.urlencode(params)


def _rows(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise PublicScheduleError(f"Leaguepedia {label} response is malformed")
    return [dict(row) for row in value]


def public_region(value: Any) -> str:
    raw = str(value or "").strip().casefold()
    if raw in {"north america", "south america", "brazil", "latin america", "americas"}:
        return "Americas"
    if raw in {"emea", "europe", "turkey", "cis", "middle east", "africa"}:
        return "EMEA"
    if raw in {
        "asia", "asia pacific", "china", "korea", "japan", "vietnam", "taiwan",
        "hong kong", "macau", "southeast asia", "oceania",
    }:
        return "Asia"
    if raw in {"international", "world"}:
        return "International"
    return "Other"


def _wiki_url(page: str) -> str:
    quoted = urllib.parse.quote(page.replace(" ", "_"), safe="/()")
    return f"https://lol.fandom.com/wiki/{quoted}"


def _fallback_tournament_name(page: str) -> str:
    return " ".join(page.replace("/2026 Season/", " 2026 ").replace("/", " ").split())


def build_public_schedule(
    *,
    now: datetime | None = None,
    fetch_json: FetchJson = _fetch_json,
) -> dict[str, Any]:
    """Fetch a compact future-fixture and tournament display artifact."""

    observed_at = _utc(now or datetime.now(timezone.utc))
    schedule_start = observed_at - timedelta(hours=6)
    schedule_end = observed_at + timedelta(days=14)
    tournament_start = observed_at - timedelta(days=14)
    tournament_end = observed_at + timedelta(days=90)

    tournament_fields = (
        "Name", "OverviewPage", "DateStart", "Date", "Region", "League",
        "TournamentLevel", "IsOfficial",
    )
    tournament_where = (
        f'Tournaments.Date >= "{tournament_start.date().isoformat()}" AND '
        f'Tournaments.DateStart <= "{tournament_end.date().isoformat()}"'
    )
    tournament_url = _cargo_url(
        "Tournaments",
        tournament_fields,
        tournament_where,
        "Tournaments.DateStart ASC",
    )
    tournament_rows = _rows(fetch_json(tournament_url), "tournament")

    tournaments: list[dict[str, Any]] = []
    tournament_by_page: dict[str, dict[str, Any]] = {}
    today = observed_at.date()
    for row in tournament_rows:
        page = str(_row_value(row, "OverviewPage") or "").strip()
        name = str(_row_value(row, "Name") or page).strip()
        start_text = str(_row_value(row, "DateStart") or "").strip()[:10]
        end_text = str(_row_value(row, "Date") or "").strip()[:10]
        if not page or not name or len(start_text) != 10 or len(end_text) != 10:
            continue
        try:
            start_date = datetime.strptime(start_text, "%Y-%m-%d").date()
            end_date = datetime.strptime(end_text, "%Y-%m-%d").date()
        except ValueError:
            continue
        status = "upcoming" if start_date > today else "past" if end_date < today else "current"
        item = {
            "name": name,
            "overview_page": page,
            "url": _wiki_url(page),
            "start_date": start_text,
            "end_date": end_text,
            "region": public_region(_row_value(row, "Region")),
            "league": str(_row_value(row, "League") or "").strip() or None,
            "level": str(_row_value(row, "TournamentLevel") or "").strip() or None,
            "official": _row_value(row, "IsOfficial") in {1, "1", True},
            "status": status,
        }
        tournaments.append(item)
        tournament_by_page[page] = item

    schedule_fields = (
        "MatchId", "OverviewPage", "Team1", "Team2", "DateTime_UTC", "HasTime",
        "BestOf", "Tab", "Winner", "N_MatchInTab",
    )
    schedule_where = (
        f'MatchSchedule.DateTime_UTC >= "{_cargo_date(schedule_start)}" AND '
        f'MatchSchedule.DateTime_UTC <= "{_cargo_date(schedule_end)}"'
    )
    schedule_url = _cargo_url(
        "MatchSchedule",
        schedule_fields,
        schedule_where,
        "MatchSchedule.DateTime_UTC ASC",
    )
    schedule_rows = _rows(fetch_json(schedule_url), "match")

    upcoming: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in schedule_rows:
        winner = _row_value(row, "Winner")
        if winner not in {None, "", 0, "0"}:
            continue
        match_id = str(_row_value(row, "MatchId") or "").strip()
        team1 = normalize_team(str(_row_value(row, "Team1") or "").strip())
        team2 = normalize_team(str(_row_value(row, "Team2") or "").strip())
        if not match_id or match_id in seen or not team1 or not team2:
            continue
        start = _parse_cargo_time(_row_value(row, "DateTime_UTC"))
        if start < schedule_start or start > schedule_end:
            continue
        page = str(_row_value(row, "OverviewPage") or "").strip()
        tournament = tournament_by_page.get(page)
        upcoming.append(
            {
                "series_id": match_id,
                "start_utc": _rfc(start),
                "has_time": _row_value(row, "HasTime") in {1, "1", True},
                "status": "live" if start <= observed_at else "scheduled",
                "team1": team1,
                "team2": team2,
                "best_of": int(_row_value(row, "BestOf") or 0) or None,
                "tournament": tournament["name"] if tournament else _fallback_tournament_name(page),
                "overview_page": page,
                "tournament_url": _wiki_url(page) if page else None,
                "stage": str(_row_value(row, "Tab") or "").strip() or None,
                "region": tournament["region"] if tournament else "Other",
                "level": tournament["level"] if tournament else None,
            }
        )
        seen.add(match_id)

    tournaments.sort(key=lambda item: (item["start_date"], item["name"]))
    upcoming.sort(key=lambda item: (item["start_utc"], item["series_id"]))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source": "Leaguepedia Cargo",
        "source_url": SOURCE_PAGE,
        "as_of": _rfc(observed_at),
        "refresh_status": "fresh",
        "upcoming": upcoming,
        "tournaments": tournaments,
    }
    validate_public_schedule(payload)
    return payload


def validate_public_schedule(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise PublicScheduleError("public schedule schema is invalid")
    if not isinstance(payload.get("upcoming"), list) or not isinstance(payload.get("tournaments"), list):
        raise PublicScheduleError("public schedule lists are missing")
    series_ids: set[str] = set()
    for row in payload["upcoming"]:
        if not isinstance(row, Mapping):
            raise PublicScheduleError("public schedule row is malformed")
        series_id = str(row.get("series_id") or "")
        if not series_id or series_id in series_ids:
            raise PublicScheduleError("public schedule identity is invalid")
        _parse_cargo_time(str(row.get("start_utc") or "").replace("T", " ").replace("Z", ""))
        if not row.get("team1") or not row.get("team2"):
            raise PublicScheduleError("public schedule team is missing")
        series_ids.add(series_id)


__all__ = [
    "PublicScheduleError",
    "SCHEMA_VERSION",
    "build_public_schedule",
    "public_region",
    "validate_public_schedule",
]
