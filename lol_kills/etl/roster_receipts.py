"""Capture patch-safe, pre-event roster receipts from Leaguepedia revisions.

Current team pages are not treated as historical truth.  For each team, this
module captures the page revision history spanning the evaluation window,
selects the newest revision strictly before each fixture cutoff, renders that
specific revision, and parses only the active five-role roster table.  A
fixture receipt is confirmed only when every role resolves to one player and
the selected source revision predates the event.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.error import HTTPError, URLError

from lol_kills.etl.aliases import normalize_team
from lol_kills.v2.data.common import parse_rfc3339, to_rfc3339


SCHEMA_VERSION = "scryglass:roster-receipts:v1"
API_ROOT = "https://lol.fandom.com/api.php"
WIKI_ROOT = "https://lol.fandom.com/wiki/"
USER_AGENT = "Scryglass-pre-event-roster-receipts/1.0"
ROLES = ("top", "jungle", "mid", "bot", "support")
OUTCOME_FIELDS = frozenset(
    {
        "actual_blue_win",
        "blue_win",
        "result",
        "winner",
        "winner_team_id",
        "won",
        "WinTeam",
        "LossTeam",
        "Team1Kills",
        "Team2Kills",
        "Gamelength_Number",
    }
)
ROLE_BY_TEXT = {
    "top laner": "top",
    "top": "top",
    "jungler": "jungle",
    "jungle": "jungle",
    "mid laner": "mid",
    "mid": "mid",
    "bot laner": "bot",
    "bot": "bot",
    "support": "support",
}


class RosterReceiptError(ValueError):
    """Raised when a roster receipt input is malformed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _parse_time(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise RosterReceiptError(f"{field} must be RFC-3339")
    try:
        return parse_rfc3339(value)
    except Exception as exc:
        raise RosterReceiptError(f"{field} must be RFC-3339") from exc


def _rfc(value: datetime) -> str:
    return to_rfc3339(value)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha_object(value: Any) -> str:
    return _sha_bytes(_canonical(value))


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json(path: Path, value: Any) -> None:
    _write_atomic(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _slug(value: str) -> str:
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-")
    return candidate or "team"


def _api_url(params: Mapping[str, Any]) -> str:
    return API_ROOT + "?" + urllib.parse.urlencode(params)


def _fetch_json(url: str, *, timeout: float) -> tuple[bytes, dict[str, Any]]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except (OSError, HTTPError, URLError) as exc:
        raise RosterReceiptError(f"Leaguepedia request failed: {url}: {exc}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RosterReceiptError(f"Leaguepedia returned invalid JSON: {url}") from exc
    if not isinstance(parsed, dict):
        raise RosterReceiptError(f"Leaguepedia JSON response is not an object: {url}")
    return raw, parsed


def history_url(team_page: str, *, newest: str, oldest: str) -> str:
    return _api_url(_history_params(team_page, newest=newest, oldest=oldest))


def _history_params(team_page: str, *, newest: str, oldest: str) -> dict[str, str]:
    """Return the first revision-history request parameters.

    MediaWiki supplies continuation tokens for long histories.  Keeping the
    base parameters in one helper makes it harder for the initial request and
    subsequent pages to drift apart.
    """

    return {
        "action": "query",
        "prop": "revisions",
        "titles": team_page,
        "rvprop": "ids|timestamp",
        "rvlimit": "max",
        "rvstart": _rfc(_parse_time(newest, "newest")),
        "rvend": _rfc(_parse_time(oldest, "oldest")),
        "rvdir": "older",
        "format": "json",
        "formatversion": "2",
    }


def revision_content_url(team_page: str, revision_id: int) -> str:
    return _api_url(
        {
            "action": "query",
            "prop": "revisions",
            "rvprop": "ids|timestamp|content",
            "rvslots": "main",
            "revids": str(revision_id),
            "format": "json",
            "formatversion": "2",
        }
    )


def rendered_revision_url(revision_id: int) -> str:
    return _api_url(
        {
            "action": "parse",
            "oldid": str(revision_id),
            "prop": "text",
            "format": "json",
            "formatversion": "2",
        }
    )


def _page_revisions(payload: Mapping[str, Any]) -> tuple[str | None, list[dict[str, Any]]]:
    pages = payload.get("query", {}).get("pages", [])
    if not isinstance(pages, list) or not pages:
        return None, []
    page = pages[0]
    if not isinstance(page, Mapping):
        return None, []
    title = str(page.get("title")) if page.get("title") else None
    raw_revisions = page.get("revisions", [])
    if not isinstance(raw_revisions, list):
        return title, []
    revisions: list[dict[str, Any]] = []
    for raw in raw_revisions:
        if not isinstance(raw, Mapping) or raw.get("revid") is None or raw.get("timestamp") is None:
            continue
        revisions.append({"revision_id": int(raw["revid"]), "revision_timestamp": str(raw["timestamp"])})
    return title, revisions


def capture_team_history(
    team_page: str,
    *,
    newest: str,
    oldest: str,
    output_dir: Path,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Capture all revision metadata for one page over the evaluation window."""

    newest_dt = _parse_time(newest, "newest")
    oldest_dt = _parse_time(oldest, "oldest")
    if oldest_dt > newest_dt:
        raise RosterReceiptError("oldest must not be after newest")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "history-manifest.json"
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                isinstance(existing, dict)
                and existing.get("team_page") == team_page
                and existing.get("newest") == _rfc(newest_dt)
                and existing.get("oldest") == _rfc(oldest_dt)
            ):
                return existing
        except (OSError, json.JSONDecodeError):
            pass

    params = _history_params(team_page, newest=_rfc(newest_dt), oldest=_rfc(oldest_dt))
    pages: list[dict[str, Any]] = []
    all_revisions: dict[int, dict[str, Any]] = {}
    title: str | None = None
    page_index = 0
    while True:
        url = _api_url(params)
        raw, payload = _fetch_json(url, timeout=timeout)
        raw_path = output_dir / f"history-{page_index:03d}.json"
        _write_atomic(raw_path, raw)
        page_title, revisions = _page_revisions(payload)
        title = title or page_title
        for revision in revisions:
            all_revisions[int(revision["revision_id"])] = revision
        pages.append(
            {
                "raw_file": str(raw_path.name),
                "source_url": url,
                "payload_sha256": _sha_bytes(raw),
                "revision_count": len(revisions),
            }
        )
        continuation = payload.get("continue")
        if not isinstance(continuation, Mapping) or not continuation.get("rvcontinue"):
            break
        params = dict(params)
        params["rvcontinue"] = str(continuation["rvcontinue"])
        params["continue"] = str(continuation.get("continue", "||"))
        page_index += 1

    revisions = sorted(all_revisions.values(), key=lambda row: (row["revision_timestamp"], row["revision_id"]))
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "team_page": team_page,
        "resolved_title": title,
        "newest": _rfc(newest_dt),
        "oldest": _rfc(oldest_dt),
        "captured_at": _utc_now(),
        "pages": pages,
        "revisions": revisions,
        "status": "ok" if revisions else "no_revision_in_window",
    }
    manifest = {**unsigned, "manifest_sha256": _sha_object(unsigned)}
    _write_json(manifest_path, manifest)
    return manifest


def _content_from_payload(payload: Mapping[str, Any]) -> tuple[int, str, str] | None:
    pages = payload.get("query", {}).get("pages", [])
    if not isinstance(pages, list) or not pages:
        return None
    page = pages[0]
    if not isinstance(page, Mapping):
        return None
    revisions = page.get("revisions", [])
    if not isinstance(revisions, list) or not revisions:
        return None
    revision = revisions[0]
    if not isinstance(revision, Mapping):
        return None
    slots = revision.get("slots", {})
    main = slots.get("main", {}) if isinstance(slots, Mapping) else {}
    content = main.get("content") if isinstance(main, Mapping) else None
    if not isinstance(content, str):
        content = revision.get("content")
    if not isinstance(content, str) or revision.get("revid") is None or revision.get("timestamp") is None:
        return None
    return int(revision["revid"]), str(revision["timestamp"]), content


def capture_revision_payload(
    team_page: str,
    revision_id: int,
    *,
    output_dir: Path,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Capture wikitext and rendered HTML for one selected revision."""

    output_dir.mkdir(parents=True, exist_ok=True)
    revision_path = output_dir / f"revision-{revision_id}.json"
    rendered_path = output_dir / f"rendered-{revision_id}.json"
    if revision_path.exists() and rendered_path.exists():
        try:
            return json.loads((output_dir / f"payload-{revision_id}.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass

    content_url = revision_content_url(team_page, revision_id)
    raw_revision, revision_payload = _fetch_json(content_url, timeout=timeout)
    content = _content_from_payload(revision_payload)
    if content is None:
        raise RosterReceiptError(f"selected revision has no wikitext: {team_page} rev={revision_id}")
    rendered_url = rendered_revision_url(revision_id)
    raw_rendered, rendered_payload = _fetch_json(rendered_url, timeout=timeout)
    html = rendered_payload.get("parse", {}).get("text")
    if not isinstance(html, str):
        raise RosterReceiptError(f"selected revision has no rendered HTML: {team_page} rev={revision_id}")
    _write_atomic(revision_path, raw_revision)
    _write_atomic(rendered_path, raw_rendered)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "team_page": team_page,
        "revision_id": content[0],
        "revision_timestamp": content[1],
        "revision_source_url": content_url,
        "rendered_source_url": rendered_url,
        "revision_payload_sha256": _sha_bytes(raw_revision),
        "rendered_payload_sha256": _sha_bytes(raw_rendered),
        "content_sha256": _sha_bytes(content[2].encode("utf-8")),
        "rendered_html_sha256": _sha_bytes(html.encode("utf-8")),
        "revision_file": revision_path.name,
        "rendered_file": rendered_path.name,
    }
    _write_json(output_dir / f"payload-{revision_id}.json", {**payload, "content": content[2], "html": html})
    return payload


class _ActiveRosterParser(HTMLParser):
    """Small parser for the rendered ``team-members-current`` table."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._table_depth = 0
        self._target_table = False
        self._row: dict[str, Any] | None = None
        self._cell_class: str | None = None
        self._cell_text: list[str] = []
        self._player_anchor_depth = 0
        self._player_anchor_text: list[str] = []
        self.rows: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = {key: value or "" for key, value in attrs}
        if tag == "table":
            classes = set(attrs_map.get("class", "").split())
            if self._target_table:
                self._table_depth += 1
            elif "team-members-current" in classes:
                self._target_table = True
                self._table_depth = 1
            return
        if not self._target_table or self._table_depth != 1:
            return
        if tag == "tr":
            self._row = {}
            self._cell_class = None
            return
        if self._row is None:
            return
        if tag in {"td", "th"}:
            self._cell_class = attrs_map.get("class", "").split()[0] if attrs_map.get("class") else "_unknown"
            self._cell_text = []
            if self._cell_class == "team-members-player":
                self._player_anchor_depth = 0
                self._player_anchor_text = []
        elif tag == "a" and self._cell_class == "team-members-player":
            self._player_anchor_depth = 1
            self._player_anchor_text = []
        elif self._player_anchor_depth and tag not in {"br", "span"}:
            self._player_anchor_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "table" and self._target_table:
            self._table_depth -= 1
            if self._table_depth == 0:
                self._target_table = False
            return
        if not self._target_table or self._table_depth != 1:
            return
        if tag == "a" and self._player_anchor_depth:
            self._row["player"] = "".join(self._player_anchor_text).strip() if self._row is not None else ""
            self._player_anchor_depth = 0
            return
        if tag in {"td", "th"}:
            if self._row is not None and self._cell_class:
                self._row[self._cell_class] = " ".join("".join(self._cell_text).split())
            self._cell_class = None
            self._cell_text = []
            return
        if tag == "tr" and self._row is not None:
            if self._row.get("player") and self._row.get("team-members-role"):
                self.rows.append(dict(self._row))
            self._row = None

    def handle_data(self, data: str) -> None:
        if not self._target_table or self._table_depth != 1 or self._row is None:
            return
        if self._cell_class:
            self._cell_text.append(data)
        if self._player_anchor_depth:
            self._player_anchor_text.append(data)


def parse_active_roster(html: str) -> tuple[dict[str, str], ...]:
    parser = _ActiveRosterParser()
    parser.feed(html)
    selected: list[dict[str, str]] = []
    for row in parser.rows:
        role_text = " ".join(str(row.get("team-members-role", "")).lower().split())
        role = ROLE_BY_TEXT.get(role_text)
        player = " ".join(str(row.get("player", "")).split())
        if role and player:
            selected.append({"role": role, "player": player})
    return tuple(selected)


def _select_revision(history: Mapping[str, Any], as_of: str) -> dict[str, Any] | None:
    cutoff = _parse_time(as_of, "as_of")
    revisions = history.get("revisions", [])
    candidates: list[dict[str, Any]] = []
    if not isinstance(revisions, list):
        return None
    for raw in revisions:
        if not isinstance(raw, Mapping):
            continue
        try:
            timestamp = _parse_time(raw.get("revision_timestamp"), "revision_timestamp")
            revision_id = int(raw.get("revision_id"))
        except (RosterReceiptError, TypeError, ValueError):
            continue
        if timestamp < cutoff:
            candidates.append({"revision_id": revision_id, "revision_timestamp": _rfc(timestamp)})
    return max(candidates, key=lambda row: (row["revision_timestamp"], row["revision_id"])) if candidates else None


def lineup_receipt(
    *,
    fixture_id: str,
    team: str,
    event_start: str,
    as_of: str,
    history: Mapping[str, Any],
    payload: Mapping[str, Any] | None,
    capture_at: str,
) -> dict[str, Any]:
    """Build one strict receipt; unavailable is the default on ambiguity."""

    blockers: list[str] = []
    revision = _select_revision(history, as_of)
    if revision is None:
        blockers.append("no_team_page_revision_strictly_before_event")
    players: tuple[dict[str, str], ...] = ()
    evidence: dict[str, Any] = {
        "team_page": history.get("resolved_title") or history.get("team_page") or team,
        "history_manifest_sha256": history.get("manifest_sha256"),
        "capture_at": capture_at,
    }
    if revision is not None:
        if payload is None:
            blockers.append("selected_revision_payload_missing")
        else:
            players = parse_active_roster(str(payload.get("html", "")))
            evidence.update(
                {
                    "revision_id": revision["revision_id"],
                    "revision_timestamp": revision["revision_timestamp"],
                    "revision_payload_sha256": payload.get("revision_payload_sha256"),
                    "rendered_payload_sha256": payload.get("rendered_payload_sha256"),
                    "content_sha256": payload.get("content_sha256"),
                    "rendered_html_sha256": payload.get("rendered_html_sha256"),
                }
            )
            by_role: dict[str, list[str]] = {role: [] for role in ROLES}
            for row in players:
                by_role[row["role"]].append(row["player"])
            for role in ROLES:
                if len(by_role[role]) != 1:
                    blockers.append(f"active_roster_role_{role}_arity_{len(by_role[role])}")
            if len({row["player"] for row in players}) != len(players):
                blockers.append("active_roster_players_not_unique")
            if len(players) != 5:
                blockers.append("active_roster_not_exactly_five")
    evidence_hash = _sha_object({"fixture_id": fixture_id, "team": team, "evidence": evidence, "players": list(players)})
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture_id,
        "team": team,
        "event_start": event_start,
        "as_of": as_of,
        "players": list(players),
        "authority_status": "confirmed" if not blockers else "unavailable",
        "authority_basis": "historical_team_page_active_roster_exact_five",
        "blockers": sorted(set(blockers)),
        "evidence": evidence,
        "evidence_hash": evidence_hash,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _strict_json_object(raw: str, field: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise RosterReceiptError(f"{field} contains duplicate key {key!r}")
            value[key] = item
        return value

    try:
        payload = json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except RosterReceiptError:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        raise RosterReceiptError(f"{field} is not strict JSON") from exc
    if not isinstance(payload, dict):
        raise RosterReceiptError(f"{field} must contain a JSON object")
    return payload


def _resolve_receipt_file(manifest_path: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RosterReceiptError("receipt_file is missing")
    candidate = Path(value)
    if candidate.is_absolute() or candidate.is_file():
        return candidate
    relative = manifest_path.parent / candidate
    if relative.is_file():
        return relative
    raise RosterReceiptError(f"receipt_file does not exist: {value}")


def _validate_confirmed_team_receipt(
    receipt: Mapping[str, Any],
    *,
    fixture_id: str,
    event_start: str,
    as_of: str,
    side: str,
) -> None:
    prefix = f"receipt.{fixture_id}.teams.{side}"
    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise RosterReceiptError(f"{prefix}.schema_version is not supported")
    if receipt.get("fixture_id") != fixture_id:
        raise RosterReceiptError(f"{prefix}.fixture_id does not match")
    if receipt.get("event_start") != event_start or receipt.get("as_of") != as_of:
        raise RosterReceiptError(f"{prefix} time binding does not match")
    team = str(receipt.get("team") or "").strip()
    if not team:
        raise RosterReceiptError(f"{prefix}.team is missing")
    blockers = receipt.get("blockers")
    if not isinstance(blockers, list) or blockers:
        raise RosterReceiptError(f"{prefix}.blockers must be an empty list")
    if receipt.get("authority_status") != "confirmed":
        raise RosterReceiptError(f"{prefix}.authority_status is not confirmed")
    if receipt.get("authority_basis") != "historical_team_page_active_roster_exact_five":
        raise RosterReceiptError(f"{prefix}.authority_basis is not supported")
    players = receipt.get("players")
    if not isinstance(players, list) or len(players) != len(ROLES):
        raise RosterReceiptError(f"{prefix}.players is not an exact five-role roster")
    observed_roles: list[str] = []
    observed_players: list[str] = []
    for index, player in enumerate(players):
        if not isinstance(player, Mapping):
            raise RosterReceiptError(f"{prefix}.players[{index}] is not an object")
        role = str(player.get("role") or "")
        identity = str(player.get("player") or "").strip()
        if role not in ROLES or not identity:
            raise RosterReceiptError(f"{prefix}.players[{index}] is invalid")
        observed_roles.append(role)
        observed_players.append(identity)
    if tuple(observed_roles) != ROLES:
        raise RosterReceiptError(f"{prefix}.players roles are not canonical")
    if len(set(observed_players)) != len(observed_players):
        raise RosterReceiptError(f"{prefix}.players identities are not unique")
    evidence = receipt.get("evidence")
    if not isinstance(evidence, Mapping):
        raise RosterReceiptError(f"{prefix}.evidence is missing")
    if OUTCOME_FIELDS.intersection(evidence):
        raise RosterReceiptError(f"{prefix}.evidence contains an outcome field")
    revision_timestamp = _parse_time(
        evidence.get("revision_timestamp"), f"{prefix}.evidence.revision_timestamp"
    )
    if revision_timestamp >= _parse_time(as_of, f"{prefix}.as_of"):
        raise RosterReceiptError(f"{prefix}.evidence revision is not strictly pre-event")
    expected_hash = _sha_object(
        {
            "fixture_id": fixture_id,
            "team": team,
            "evidence": dict(evidence),
            "players": [dict(player) for player in players],
        }
    )
    if receipt.get("evidence_hash") != expected_hash:
        raise RosterReceiptError(f"{prefix}.evidence_hash does not match")


def load_receipt_manifest(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate and load a hash-bound result-free lineup receipt package.

    This validates the exact manifest and JSONL bytes plus every confirmed
    fixture's temporal, team, role, player, and evidence-hash bindings.  It
    does not grant model, probability, publication, or betting authority.
    """

    if not manifest_path.is_file():
        raise RosterReceiptError(f"receipt manifest does not exist: {manifest_path}")
    manifest = _strict_json_object(
        manifest_path.read_text(encoding="utf-8"), "receipt manifest"
    )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RosterReceiptError("receipt manifest schema_version is not supported")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != _sha_object(unsigned):
        raise RosterReceiptError("receipt manifest hash does not match")
    receipt_path = _resolve_receipt_file(manifest_path, manifest.get("receipt_file"))
    receipt_bytes = receipt_path.read_bytes()
    receipt_sha256 = _sha_bytes(receipt_bytes)
    if manifest.get("receipt_file_sha256") != receipt_sha256:
        raise RosterReceiptError("receipt file hash does not match")

    index: dict[str, dict[str, Any]] = {}
    confirmed = 0
    for line_number, raw_line in enumerate(receipt_bytes.decode("utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        row = _strict_json_object(raw_line, f"receipt line {line_number}")
        if row.get("schema_version") != SCHEMA_VERSION:
            raise RosterReceiptError(f"receipt line {line_number} schema_version is not supported")
        if OUTCOME_FIELDS.intersection(row):
            raise RosterReceiptError(f"receipt line {line_number} contains an outcome field")
        fixture_id = str(row.get("fixture_id") or "").strip()
        if not fixture_id or fixture_id in index:
            raise RosterReceiptError(f"receipt line {line_number} fixture_id is missing or duplicated")
        event_start = str(row.get("event_start") or "")
        as_of = str(row.get("as_of") or "")
        if _parse_time(as_of, f"receipt.{fixture_id}.as_of") >= _parse_time(
            event_start, f"receipt.{fixture_id}.event_start"
        ):
            raise RosterReceiptError(f"receipt.{fixture_id} cutoff is not before event start")
        blockers = row.get("blockers")
        if not isinstance(blockers, list) or any(not isinstance(item, str) for item in blockers):
            raise RosterReceiptError(f"receipt.{fixture_id}.blockers is invalid")
        teams = row.get("teams")
        if not isinstance(teams, Mapping) or set(teams) != {"blue", "red"}:
            raise RosterReceiptError(f"receipt.{fixture_id}.teams is invalid")
        expected_fixture_hash = _sha_object(dict(teams))
        if row.get("evidence_hash") != expected_fixture_hash:
            raise RosterReceiptError(f"receipt.{fixture_id}.evidence_hash does not match")
        status = row.get("authority_status")
        if status not in {"confirmed", "unavailable"}:
            raise RosterReceiptError(f"receipt.{fixture_id}.authority_status is invalid")
        if status == "confirmed":
            if blockers:
                raise RosterReceiptError(f"receipt.{fixture_id} is confirmed with blockers")
            for side in ("blue", "red"):
                team_receipt = teams.get(side)
                if not isinstance(team_receipt, Mapping):
                    raise RosterReceiptError(f"receipt.{fixture_id}.teams.{side} is invalid")
                _validate_confirmed_team_receipt(
                    team_receipt,
                    fixture_id=fixture_id,
                    event_start=event_start,
                    as_of=as_of,
                    side=side,
                )
            if normalize_team(str(teams["blue"].get("team") or "")) == normalize_team(
                str(teams["red"].get("team") or "")
            ):
                raise RosterReceiptError(f"receipt.{fixture_id} teams are not distinct")
            confirmed += 1
        index[fixture_id] = row

    fixture_count = int(manifest.get("fixture_count", -1))
    confirmed_count = int(manifest.get("confirmed_fixture_count", -1))
    unavailable_count = int(manifest.get("unavailable_fixture_count", -1))
    if fixture_count != len(index):
        raise RosterReceiptError("receipt manifest fixture_count does not match")
    if confirmed_count != confirmed:
        raise RosterReceiptError("receipt manifest confirmed_fixture_count does not match")
    if unavailable_count != fixture_count - confirmed_count:
        raise RosterReceiptError("receipt manifest unavailable_fixture_count does not match")
    claim_ceiling = manifest.get("claim_ceiling")
    if not isinstance(claim_ceiling, Mapping):
        raise RosterReceiptError("receipt manifest claim_ceiling is missing")
    if (
        claim_ceiling.get("pre_event_lineup_authority") is not bool(confirmed)
        or claim_ceiling.get("winner_prediction") is not False
        or claim_ceiling.get("publication") is not False
    ):
        raise RosterReceiptError("receipt manifest claim_ceiling is invalid")
    readiness = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if confirmed == fixture_count else "partial",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "receipt_file": str(receipt_path),
        "receipt_file_sha256": receipt_sha256,
        "fixture_count": fixture_count,
        "confirmed_fixture_count": confirmed_count,
        "unavailable_fixture_count": unavailable_count,
        "claim_ceiling": dict(claim_ceiling),
        "model_authority": False,
        "probability_authority": False,
        "betting_authority": False,
    }
    return readiness, index


def build_receipts(
    run_dir: Path,
    output_dir: Path,
    *,
    workers: int = 4,
    timeout: float = 60.0,
    no_capture: bool = False,
) -> dict[str, Any]:
    """Capture team histories and materialize one receipt for every game."""

    rows = _load_jsonl(run_dir / "frozen-ledger.jsonl")
    if not rows:
        raise RosterReceiptError("frozen ledger is empty")
    team_cutoffs: dict[str, list[datetime]] = {}
    for row in rows:
        pregame = row.get("pregame", {})
        event_start = _parse_time(pregame.get("event_start"), "pregame.event_start")
        for side in ("blue", "red"):
            team = normalize_team(str(pregame.get(side, {}).get("team", "")).strip())
            if team:
                team_cutoffs.setdefault(team, []).append(event_start)
    teams_root = output_dir / "teams"
    histories: dict[str, dict[str, Any]] = {}
    if no_capture:
        for team in team_cutoffs:
            manifest_path = teams_root / _slug(team) / "history-manifest.json"
            if manifest_path.exists():
                histories[team] = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        def capture_one(team: str) -> tuple[str, dict[str, Any]]:
            times = team_cutoffs[team]
            return team, capture_team_history(
                team,
                newest=_rfc(max(times)),
                # A stable 2026 roster may have been established before the
                # first evaluation date. Keep enough history to select that
                # pre-event revision instead of falsely reporting no source.
                oldest=_rfc(min(times) - timedelta(days=370)),
                output_dir=teams_root / _slug(team),
                timeout=timeout,
            )

        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {executor.submit(capture_one, team): team for team in sorted(team_cutoffs)}
            for future in as_completed(futures):
                team = futures[future]
                try:
                    name, history = future.result()
                    histories[name] = history
                except Exception as exc:
                    histories[team] = {
                        "schema_version": SCHEMA_VERSION,
                        "team_page": team,
                        "status": "capture_failed",
                        "captured_at": _utc_now(),
                        "error": str(exc),
                        "revisions": [],
                    }
                    _write_json(teams_root / _slug(team) / "history-manifest.json", histories[team])

    # Fetch each selected revision once. Multiple games commonly share the
    # same team-page revision; prefetching by (team, revision) avoids making
    # the receipt pass depend on serial network latency.
    payload_requests: dict[tuple[str, int], Path] = {}
    if not no_capture:
        for row in rows:
            pregame = row["pregame"]
            as_of = str(pregame["as_of"])
            for side in ("blue", "red"):
                team = normalize_team(str(pregame[side]["team"]))
                history = histories.get(team, {})
                selected = _select_revision(history, as_of)
                if selected is None or history.get("status") != "ok":
                    continue
                payload_requests[(team, int(selected["revision_id"]))] = teams_root / _slug(team)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(
                    capture_revision_payload,
                    team,
                    revision_id,
                    output_dir=team_dir,
                    timeout=timeout,
                ): (team, revision_id)
                for (team, revision_id), team_dir in sorted(payload_requests.items())
                if not (team_dir / f"payload-{revision_id}.json").exists()
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    # The fixture receipt records the specific missing
                    # payload below; one failed team must not discard the
                    # other captured histories.
                    pass

    receipts: list[dict[str, Any]] = []
    for row in rows:
        pregame = row["pregame"]
        fixture_id = str(pregame["fixture_id"])
        event_start = str(pregame["event_start"])
        as_of = str(pregame["as_of"])
        team_receipts: dict[str, Any] = {}
        blockers: list[str] = []
        for side in ("blue", "red"):
            team = normalize_team(str(pregame[side]["team"]))
            history = histories.get(team, {"team_page": team, "revisions": [], "status": "history_missing"})
            selected = _select_revision(history, as_of)
            payload = None
            if selected is not None and history.get("status") == "ok":
                payload_path = teams_root / _slug(team) / f"payload-{selected['revision_id']}.json"
                try:
                    if not payload_path.exists() and not no_capture:
                        capture_revision_payload(
                            team,
                            int(selected["revision_id"]),
                            output_dir=teams_root / _slug(team),
                            timeout=timeout,
                        )
                    if payload_path.exists():
                        stored = json.loads(payload_path.read_text(encoding="utf-8"))
                        payload = stored if isinstance(stored, Mapping) else None
                except Exception as exc:
                    blockers.append(f"{side}_revision_payload_capture_failed:{type(exc).__name__}")
            receipt = lineup_receipt(
                fixture_id=fixture_id,
                team=team,
                event_start=event_start,
                as_of=as_of,
                history=history,
                payload=payload,
                capture_at=str(history.get("captured_at", "")),
            )
            team_receipts[side] = receipt
            blockers.extend(f"{side}:{value}" for value in receipt["blockers"])
        receipts.append(
            {
                "schema_version": SCHEMA_VERSION,
                "fixture_id": fixture_id,
                "event_start": event_start,
                "as_of": as_of,
                "authority_status": "confirmed" if not blockers else "unavailable",
                "blockers": sorted(set(blockers)),
                "teams": team_receipts,
                "evidence_hash": _sha_object(team_receipts),
            }
        )

    receipts.sort(key=lambda row: (row["event_start"], row["fixture_id"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "lineup-receipts.jsonl"
    _write_atomic(receipt_path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in receipts).encode("utf-8"))
    confirmed = sum(row["authority_status"] == "confirmed" for row in receipts)
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "captured_at": _utc_now(),
        "team_count": len(team_cutoffs),
        "fixture_count": len(receipts),
        "confirmed_fixture_count": confirmed,
        "unavailable_fixture_count": len(receipts) - confirmed,
        "receipt_file": str(receipt_path),
        "receipt_file_sha256": _sha_bytes(receipt_path.read_bytes()),
        "teams": sorted(team_cutoffs),
        "claim_ceiling": {
            "pre_event_lineup_authority": bool(confirmed),
            "winner_prediction": False,
            "publication": False,
        },
    }
    manifest = {**unsigned, "manifest_sha256": _sha_object(unsigned)}
    _write_json(output_dir / "receipt-manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--no-capture", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.timeout <= 0:
        parser.error("--workers must be positive and --timeout must be greater than zero")
    try:
        result = build_receipts(
            args.run_dir,
            args.output_dir,
            workers=args.workers,
            timeout=args.timeout,
            no_capture=args.no_capture,
        )
    except (OSError, RosterReceiptError, ValueError) as exc:
        print(str(exc))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
