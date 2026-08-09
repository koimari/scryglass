"""Private exact-roster capture with independently pinned registration.

The capture step only creates a candidate.  Runtime use requires a separate
registry whose digest is supplied out of band, so a roster file cannot grant
itself authority merely by being internally self-consistent.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
RECEIPT_SCHEMA_VERSION = "scryglass.private-pregame-roster.v1"
REGISTRY_SCHEMA_VERSION = "scryglass.private-pregame-roster-registry.v1"
REGISTRY_SCOPE = "private_personal_decision_support"
RECEIPT_PREFIX = PurePosixPath("data/lol/private_pregame_rosters/receipts")
ROLES = ("top", "jungle", "mid", "bot", "support")
SIDES = ("blue", "red")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_SOURCE_PAYLOAD_BYTES = 5_000_000


class PregameRosterError(ValueError):
    """A pregame roster receipt or registry violates its frozen contract."""


class RegisteredPregameRosterUnavailable(PregameRosterError):
    """No independently registered exact roster exists for the event."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PregameRosterError("value is not canonical finite JSON") from exc


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PregameRosterError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PregameRosterError(f"non-finite JSON number in {label}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PregameRosterError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise PregameRosterError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise PregameRosterError(f"{label} keys do not match the frozen contract")


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PregameRosterError(f"{label} must be a non-empty string")
    return value.strip()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise PregameRosterError(f"{label} must be a lowercase SHA-256")
    return value


def _timestamp(value: Any, label: str) -> datetime:
    text = _nonempty(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PregameRosterError(f"{label} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise PregameRosterError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _decode_source_payload(value: Any) -> bytes:
    text = _nonempty(value, "source_payload_base64")
    try:
        raw = base64.b64decode(text, validate=True)
    except (TypeError, ValueError) as exc:
        raise PregameRosterError("source_payload_base64 is not strict base64") from exc
    if not raw or len(raw) > MAX_SOURCE_PAYLOAD_BYTES:
        raise PregameRosterError("source payload is empty or exceeds the size limit")
    if base64.b64encode(raw).decode("ascii") != text:
        raise PregameRosterError("source_payload_base64 is not canonical")
    return raw


def _normalize_teams(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PregameRosterError("teams must contain exact blue and red rosters")
    if len(value) != len(SIDES):
        raise PregameRosterError("teams must contain exact blue and red rosters")
    normalized: list[dict[str, Any]] = []
    all_players: set[str] = set()
    for index, side in enumerate(SIDES):
        team = value[index]
        if not isinstance(team, Mapping):
            raise PregameRosterError(f"teams.{side} must be a mapping")
        _exact_keys(
            team,
            {
                "side",
                "organization_id",
                "organization_name",
                "roster_id",
                "players",
            },
            f"teams.{side}",
        )
        if team.get("side") != side:
            raise PregameRosterError("teams must be ordered blue then red")
        players = team.get("players")
        if not isinstance(players, list) or len(players) != len(ROLES):
            raise PregameRosterError(f"teams.{side} requires exactly five players")
        normalized_players: list[dict[str, str]] = []
        side_players: set[str] = set()
        for role_index, role in enumerate(ROLES):
            player = players[role_index]
            if not isinstance(player, Mapping):
                raise PregameRosterError(f"teams.{side}.players.{role} is invalid")
            _exact_keys(
                player,
                {"role", "player_id", "display_name"},
                f"teams.{side}.players.{role}",
            )
            if player.get("role") != role:
                raise PregameRosterError(
                    f"teams.{side}.players must be ordered {ROLES}"
                )
            player_id = _nonempty(
                player.get("player_id"), f"teams.{side}.players.{role}.player_id"
            )
            if player_id in side_players or player_id in all_players:
                raise PregameRosterError("pregame rosters cannot repeat a player")
            side_players.add(player_id)
            all_players.add(player_id)
            normalized_players.append(
                {
                    "role": role,
                    "player_id": player_id,
                    "display_name": _nonempty(
                        player.get("display_name"),
                        f"teams.{side}.players.{role}.display_name",
                    ),
                }
            )
        normalized.append(
            {
                "side": side,
                "organization_id": _nonempty(
                    team.get("organization_id"), f"teams.{side}.organization_id"
                ),
                "organization_name": _nonempty(
                    team.get("organization_name"),
                    f"teams.{side}.organization_name",
                ),
                "roster_id": _nonempty(
                    team.get("roster_id"), f"teams.{side}.roster_id"
                ),
                "players": normalized_players,
            }
        )
    if len({team["organization_id"] for team in normalized}) != 2:
        raise PregameRosterError("pregame roster organizations must be distinct")
    if len({team["roster_id"] for team in normalized}) != 2:
        raise PregameRosterError("pregame roster ids must be distinct")
    return normalized


def build_pregame_roster_receipt(
    *,
    raw_source_payload: bytes,
    source: str,
    source_url: str,
    source_record_id: str,
    source_updated_at: str,
    available_at: str,
    captured_at: str,
    event_id: str,
    event_start: str,
    league: str,
    teams: Sequence[Mapping[str, Any]],
    capture_protocol_sha256: str,
    rights_status: str = "reviewed",
) -> dict[str, Any]:
    """Build a non-authorizing exact-roster candidate from captured bytes."""
    if not isinstance(raw_source_payload, bytes) or not raw_source_payload:
        raise PregameRosterError("raw_source_payload must be non-empty bytes")
    if len(raw_source_payload) > MAX_SOURCE_PAYLOAD_BYTES:
        raise PregameRosterError("source payload exceeds the size limit")
    parsed_url = urlparse(_nonempty(source_url, "source_url"))
    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise PregameRosterError("source_url must be an absolute HTTPS URL")
    league_id = _nonempty(league, "league")
    if league_id != league_id.upper():
        raise PregameRosterError("league must use its canonical uppercase id")
    if rights_status != "reviewed":
        raise PregameRosterError("pregame roster source rights must be reviewed")
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "source": _nonempty(source, "source"),
        "source_url": source_url,
        "source_record_id": _nonempty(source_record_id, "source_record_id"),
        "source_payload_sha256": hashlib.sha256(raw_source_payload).hexdigest(),
        "source_payload_base64": base64.b64encode(raw_source_payload).decode("ascii"),
        "source_updated_at": source_updated_at,
        "available_at": available_at,
        "captured_at": captured_at,
        "event_id": _nonempty(event_id, "event_id"),
        "event_start": event_start,
        "league": league_id,
        "rights_status": rights_status,
        "capture_protocol_sha256": _sha(
            capture_protocol_sha256, "capture_protocol_sha256"
        ),
        "teams": _normalize_teams(teams),
    }
    validate_pregame_roster_receipt(receipt)
    return receipt


def validate_pregame_roster_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_receipt_sha256: str | None = None,
    expected_capture_protocol_sha256: str | None = None,
) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise PregameRosterError("pregame roster receipt must be a mapping")
    _exact_keys(
        receipt,
        {
            "schema_version",
            "source",
            "source_url",
            "source_record_id",
            "source_payload_sha256",
            "source_payload_base64",
            "source_updated_at",
            "available_at",
            "captured_at",
            "event_id",
            "event_start",
            "league",
            "rights_status",
            "capture_protocol_sha256",
            "teams",
        },
        "pregame roster receipt",
    )
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise PregameRosterError("pregame roster receipt schema is not recognized")
    actual_sha256 = sha256_json(receipt)
    if expected_receipt_sha256 is not None:
        if actual_sha256 != _sha(expected_receipt_sha256, "expected_receipt_sha256"):
            raise PregameRosterError("pregame roster receipt digest mismatch")
    _nonempty(receipt.get("source"), "source")
    parsed_url = urlparse(_nonempty(receipt.get("source_url"), "source_url"))
    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise PregameRosterError("source_url must be an absolute HTTPS URL")
    _nonempty(receipt.get("source_record_id"), "source_record_id")
    raw = _decode_source_payload(receipt.get("source_payload_base64"))
    if hashlib.sha256(raw).hexdigest() != _sha(
        receipt.get("source_payload_sha256"), "source_payload_sha256"
    ):
        raise PregameRosterError("source payload digest mismatch")
    source_updated = _timestamp(receipt.get("source_updated_at"), "source_updated_at")
    available = _timestamp(receipt.get("available_at"), "available_at")
    captured = _timestamp(receipt.get("captured_at"), "captured_at")
    event_start = _timestamp(receipt.get("event_start"), "event_start")
    if source_updated > captured:
        raise PregameRosterError("source_updated_at cannot be after captured_at")
    if available > captured:
        raise PregameRosterError("available_at cannot be after captured_at")
    if available >= event_start or captured >= event_start:
        raise PregameRosterError("pregame roster was not available and captured before event_start")
    _nonempty(receipt.get("event_id"), "event_id")
    league = _nonempty(receipt.get("league"), "league")
    if league != league.upper():
        raise PregameRosterError("league must use its canonical uppercase id")
    if receipt.get("rights_status") != "reviewed":
        raise PregameRosterError("pregame roster source rights must be reviewed")
    protocol_sha = _sha(
        receipt.get("capture_protocol_sha256"), "capture_protocol_sha256"
    )
    if (
        expected_capture_protocol_sha256 is not None
        and protocol_sha
        != _sha(
            expected_capture_protocol_sha256,
            "expected_capture_protocol_sha256",
        )
    ):
        raise PregameRosterError("capture protocol binding mismatch")
    return {
        **dict(receipt),
        "teams": _normalize_teams(receipt.get("teams")),
        "receipt_sha256": actual_sha256,
    }


def _validate_receipt_locator(locator: Any) -> PurePosixPath:
    path = PurePosixPath(_nonempty(locator, "receipt_locator"))
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or tuple(path.parts[: len(RECEIPT_PREFIX.parts)]) != RECEIPT_PREFIX.parts
        or path.suffix != ".json"
    ):
        raise PregameRosterError("receipt_locator is outside the private roster root")
    return path


def build_pregame_roster_registry(
    *,
    receipts: Sequence[tuple[str, Mapping[str, Any]]],
    registry_id: str,
    independent_reviewer_id: str,
    issued_at: str,
    capture_protocol_sha256: str,
) -> dict[str, Any]:
    """Build a registry candidate whose digest still needs external pinning."""
    _timestamp(issued_at, "issued_at")
    protocol_sha = _sha(capture_protocol_sha256, "capture_protocol_sha256")
    entries: list[dict[str, Any]] = []
    for locator, candidate in receipts:
        receipt = validate_pregame_roster_receipt(
            candidate,
            expected_capture_protocol_sha256=protocol_sha,
        )
        _validate_receipt_locator(locator)
        entries.append(
            {
                "event_id": receipt["event_id"],
                "event_start": receipt["event_start"],
                "league": receipt["league"],
                "blue_organization_id": receipt["teams"][0]["organization_id"],
                "blue_organization_name": receipt["teams"][0]["organization_name"],
                "red_organization_id": receipt["teams"][1]["organization_id"],
                "red_organization_name": receipt["teams"][1]["organization_name"],
                "source_record_id": receipt["source_record_id"],
                "receipt_locator": locator,
                "receipt_sha256": receipt["receipt_sha256"],
            }
        )
    registry = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "status": "approved",
        "scope": REGISTRY_SCOPE,
        "public_or_transactional_use": False,
        "registry_id": _nonempty(registry_id, "registry_id"),
        "independent_reviewer_id": _nonempty(
            independent_reviewer_id, "independent_reviewer_id"
        ),
        "issued_at": issued_at,
        "capture_protocol_sha256": protocol_sha,
        "entries": sorted(entries, key=lambda entry: entry["event_id"]),
    }
    validate_pregame_roster_registry(
        registry, expected_registry_sha256=sha256_json(registry)
    )
    return registry


def validate_pregame_roster_registry(
    registry: Mapping[str, Any], *, expected_registry_sha256: str | None
) -> dict[str, Any]:
    if expected_registry_sha256 is None:
        raise RegisteredPregameRosterUnavailable("roster_registry_not_registered")
    expected_sha = _sha(expected_registry_sha256, "expected_registry_sha256")
    if not isinstance(registry, Mapping):
        raise PregameRosterError("pregame roster registry must be a mapping")
    if sha256_json(registry) != expected_sha:
        raise RegisteredPregameRosterUnavailable("roster_registry_digest_mismatch")
    _exact_keys(
        registry,
        {
            "schema_version",
            "status",
            "scope",
            "public_or_transactional_use",
            "registry_id",
            "independent_reviewer_id",
            "issued_at",
            "capture_protocol_sha256",
            "entries",
        },
        "pregame roster registry",
    )
    if (
        registry.get("schema_version") != REGISTRY_SCHEMA_VERSION
        or registry.get("status") != "approved"
        or registry.get("scope") != REGISTRY_SCOPE
        or registry.get("public_or_transactional_use") is not False
    ):
        raise PregameRosterError("pregame roster registry is not approved for private support")
    _nonempty(registry.get("registry_id"), "registry_id")
    _nonempty(registry.get("independent_reviewer_id"), "independent_reviewer_id")
    _timestamp(registry.get("issued_at"), "issued_at")
    _sha(registry.get("capture_protocol_sha256"), "capture_protocol_sha256")
    entries = registry.get("entries")
    if not isinstance(entries, list) or not entries:
        raise PregameRosterError("pregame roster registry entries must be non-empty")
    expected_keys = {
        "event_id",
        "event_start",
        "league",
        "blue_organization_id",
        "blue_organization_name",
        "red_organization_id",
        "red_organization_name",
        "source_record_id",
        "receipt_locator",
        "receipt_sha256",
    }
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise PregameRosterError("pregame roster registry entry must be a mapping")
        _exact_keys(entry, expected_keys, "pregame roster registry entry")
        event_id = _nonempty(entry.get("event_id"), "entry.event_id")
        if event_id in seen:
            raise PregameRosterError("pregame roster registry contains an ambiguous event")
        seen.add(event_id)
        _timestamp(entry.get("event_start"), "entry.event_start")
        for field in (
            "league",
            "blue_organization_id",
            "blue_organization_name",
            "red_organization_id",
            "red_organization_name",
            "source_record_id",
        ):
            _nonempty(entry.get(field), f"entry.{field}")
        _validate_receipt_locator(entry.get("receipt_locator"))
        _sha(entry.get("receipt_sha256"), "entry.receipt_sha256")
        normalized.append(dict(entry))
    if normalized != sorted(normalized, key=lambda entry: entry["event_id"]):
        raise PregameRosterError("pregame roster registry entries are not ordered")
    return {**dict(registry), "entries": normalized}


def _safe_repo_file(root: Path, locator: str) -> Path:
    relative = PurePosixPath(locator)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise RegisteredPregameRosterUnavailable("roster_artifact_path_invalid")
    root_real = root.resolve(strict=True)
    current = root_real
    for part in relative.parts:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as exc:
            raise RegisteredPregameRosterUnavailable("roster_artifact_missing") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RegisteredPregameRosterUnavailable("roster_artifact_symlink_rejected")
    metadata = os.lstat(current)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise RegisteredPregameRosterUnavailable(
            "roster_artifact_not_unaliased_file"
        )
    try:
        current.resolve(strict=True).relative_to(root_real)
    except ValueError as exc:
        raise RegisteredPregameRosterUnavailable("roster_artifact_path_escape") from exc
    return current


def load_registered_pregame_roster(
    *,
    registry_locator: str,
    expected_registry_sha256: str | None,
    event_id: str,
    event_start: str,
    league: str,
    blue_organization_name: str,
    red_organization_name: str,
    as_of: datetime,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Load one exact roster through an independently pinned event registry."""
    if not expected_registry_sha256:
        raise RegisteredPregameRosterUnavailable("roster_registry_not_registered")
    registry_path = _safe_repo_file(root, registry_locator)
    registry = validate_pregame_roster_registry(
        _read_json_bytes(registry_path.read_bytes(), "pregame roster registry"),
        expected_registry_sha256=expected_registry_sha256,
    )
    if _timestamp(registry["issued_at"], "issued_at") > as_of.astimezone(timezone.utc):
        raise RegisteredPregameRosterUnavailable("roster_registry_from_future")
    matches = [entry for entry in registry["entries"] if entry["event_id"] == event_id]
    if len(matches) != 1:
        raise RegisteredPregameRosterUnavailable("registered_pregame_roster_unavailable")
    entry = matches[0]
    if _timestamp(entry.get("event_start"), "entry.event_start") != _timestamp(
        event_start, "event_start"
    ):
        raise RegisteredPregameRosterUnavailable(
            "roster_event_start_binding_mismatch"
        )
    expected_bindings = {
        "league": league,
        "blue_organization_name": blue_organization_name,
        "red_organization_name": red_organization_name,
    }
    for field, expected in expected_bindings.items():
        if entry.get(field) != expected:
            raise RegisteredPregameRosterUnavailable(f"roster_{field}_binding_mismatch")
    receipt_path = _safe_repo_file(root, entry["receipt_locator"])
    receipt = validate_pregame_roster_receipt(
        _read_json_bytes(receipt_path.read_bytes(), "pregame roster receipt"),
        expected_receipt_sha256=entry["receipt_sha256"],
        expected_capture_protocol_sha256=registry["capture_protocol_sha256"],
    )
    for field in (
        "event_id",
        "event_start",
        "league",
        "source_record_id",
    ):
        if receipt.get(field) != entry.get(field):
            raise RegisteredPregameRosterUnavailable(f"roster_{field}_binding_mismatch")
    for side_index, side in enumerate(SIDES):
        team = receipt["teams"][side_index]
        for suffix in ("organization_id", "organization_name"):
            if team[suffix] != entry[f"{side}_{suffix}"]:
                raise RegisteredPregameRosterUnavailable(
                    f"roster_{side}_{suffix}_binding_mismatch"
                )
    return {
        "status": "registered",
        "roster": {
            key: value for key, value in receipt.items() if key != "receipt_sha256"
        },
        "receipt_sha256": receipt["receipt_sha256"],
        "registry_id": registry["registry_id"],
        "registry_sha256": expected_registry_sha256,
        "capture_protocol_sha256": registry["capture_protocol_sha256"],
    }


__all__ = [
    "PregameRosterError",
    "RegisteredPregameRosterUnavailable",
    "build_pregame_roster_receipt",
    "build_pregame_roster_registry",
    "canonical_bytes",
    "load_registered_pregame_roster",
    "sha256_json",
    "validate_pregame_roster_receipt",
    "validate_pregame_roster_registry",
]
