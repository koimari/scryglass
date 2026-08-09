"""Normalize prospective GRID draft receipts for terminal Draft Score evaluation.

GRID Series Events identifies the ordered team pick/ban actions before a map
starts, but the locally observed pre-start records do not identify the final
player-role assignment.  This adapter therefore keeps two evidence lanes
separate: exact system-clocked GRID message receipts for action order and an
explicit reviewed role-assignment input.  It never guesses a flex assignment.

The generated payloads are private, outcome-free evaluation inputs.  They do
not create probability, odds, recommendation, or betting authority.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from lol_kills.etl.grid_series_events import (
    GridSeriesEventsError,
    RECEIPT_SOURCE_ID,
    SERIES_EVENTS_BASE,
    transaction_sequence,
    validate_received_transaction_envelope,
)
from lol_kills.v2.data.common import ROLES

from . import future_prediction_ledger as ledger


ROOT = Path(__file__).resolve().parents[4]
SOURCE_LOCATOR = "lol_kills/v2/draft/terminal/grid_future_source_v1.py"
TRANSPORT_LOCATOR = "lol_kills/etl/grid_series_events.py"
READINESS_REGISTRY_LOCATOR = (
    "lol_kills/v2/draft/terminal/grid_source_readiness_registry_v1.py"
)
CONTEXT_SCHEMA_VERSION = "scryglass:grid-terminal-draft-capture-context:v1"
DRAFT_SOURCE_SCHEMA_VERSION = "scryglass:grid-terminal-draft-source:v1"
MAP_START_SOURCE_SCHEMA_VERSION = "scryglass:grid-map-start-source:v1"
SOURCE_ID = "grid-series-events-plus-reviewed-role-assignment-v1"
PATCH_RE = re.compile(r"^26\.(?:0[1-9]|1[0-9]|2[0-9])$")
SECRET_QUERY_RE = re.compile(
    r"(?:[?&]|^)(?:key|token|auth|authorization|password|secret)=",
    re.IGNORECASE,
)
ALLOWED_ASSIGNMENT_METHODS = frozenset(
    {"reviewed_live_broadcast", "reviewed_provider_assignment_source"}
)


class GridFutureSourceError(ValueError):
    """A GRID terminal-draft or map-start input failed closed."""


@dataclass(frozen=True)
class PreparedDraftInputs:
    metadata_raw: bytes
    source_payload_raw: bytes
    validated_at_utc: datetime
    last_draft_received_at_utc: datetime


@dataclass(frozen=True)
class PreparedMapStartInputs:
    metadata_raw: bytes
    source_payload_raw: bytes
    actual_map_start_utc: datetime
    source_received_at_utc: datetime


@dataclass(frozen=True)
class _EnvelopeRecord:
    sequence_number: int
    received_at_utc: datetime
    envelope_raw_sha256: str
    message_raw_sha256: str
    transaction: dict[str, Any]


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise GridFutureSourceError("GRID adapter value is not canonical JSON") from exc


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in items:
        if key in value:
            raise GridFutureSourceError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _read_object(raw: bytes, field: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                GridFutureSourceError(
                    f"non-finite JSON number in {field}: {token}"
                )
            ),
        )
    except GridFutureSourceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GridFutureSourceError(f"{field} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise GridFutureSourceError(f"{field} must be a JSON object")
    return value


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GridFutureSourceError(f"{field} must be a non-empty string")
    return value.strip()


def _timestamp(value: Any, field: str) -> datetime:
    text = _nonempty(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GridFutureSourceError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise GridFutureSourceError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _clock_sample(clock: Callable[[], datetime], field: str) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise GridFutureSourceError(
            f"{field} clock must return a timezone-aware datetime"
        )
    return value.astimezone(timezone.utc)


def _safe_url(value: Any, field: str) -> str:
    url = _nonempty(value, field)
    if SECRET_QUERY_RE.search(url):
        raise GridFutureSourceError(f"{field} contains credential-like query data")
    return url


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GridFutureSourceError(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise GridFutureSourceError(f"{field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise GridFutureSourceError(f"{field} must be an integer") from exc
    if parsed < 1:
        raise GridFutureSourceError(f"{field} must be positive")
    return parsed


def _source_record(root: Path, locator: str) -> dict[str, Any]:
    path = root / locator
    if not path.is_file():
        raise GridFutureSourceError(f"GRID adapter source is missing: {locator}")
    return {
        "locator": locator,
        "bytes": path.stat().st_size,
        "raw_sha256": _sha256_path(path),
    }


def _registered_readiness(root: Path) -> dict[str, Any]:
    try:
        from .grid_source_readiness_registry_v1 import (
            REGISTERED_GRID_SOURCE_ARTIFACT_SHA256,
            REGISTERED_GRID_SOURCE_LOCATOR,
            REGISTERED_GRID_SOURCE_RAW_SHA256,
            validate_registered_grid_source_readiness_v1,
        )
    except ImportError as exc:
        raise GridFutureSourceError(
            "registered GRID source readiness is not installed"
        ) from exc
    try:
        readiness = validate_registered_grid_source_readiness_v1(root=root)
    except (OSError, ValueError, RuntimeError) as exc:
        raise GridFutureSourceError(str(exc)) from exc
    capability = readiness.get("capability_conclusion") or {}
    if (
        capability.get("terminal_pick_ban_prestart_observed_in_all_archives")
        is not True
        or capability.get("prestart_role_assignment_available_from_grid")
        is not False
        or capability.get("prospective_system_receipts_required") is not True
        or capability.get("retrospective_archives_qualify_future_evidence")
        is not False
    ):
        raise GridFutureSourceError("registered GRID capability boundary changed")
    return {
        "locator": REGISTERED_GRID_SOURCE_LOCATOR.as_posix(),
        "raw_sha256": REGISTERED_GRID_SOURCE_RAW_SHA256,
        "artifact_sha256": REGISTERED_GRID_SOURCE_ARTIFACT_SHA256,
        "registry_source": _source_record(root, READINESS_REGISTRY_LOCATOR),
    }


def _validate_attestation(
    value: Any, *, field: str, kind: str
) -> dict[str, Any]:
    common = {"source_id", "source_url", "source_record_id", "rights_status"}
    if kind == "fixture":
        expected = common | {
            "series_identity_verified",
            "game_identity_verified",
            "team_crosswalk_verified",
        }
    else:
        expected = common | {
            "observation_method",
            "observed_before_map_start",
        }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise GridFutureSourceError(f"{field} structure changed")
    result = dict(value)
    for name in ("source_id", "source_record_id"):
        result[name] = _nonempty(result.get(name), f"{field}.{name}")
    result["source_url"] = _safe_url(result.get("source_url"), f"{field}.source_url")
    if result.get("rights_status") != "reviewed":
        raise GridFutureSourceError(f"{field} rights are not reviewed")
    if kind == "fixture":
        for name in (
            "series_identity_verified",
            "game_identity_verified",
            "team_crosswalk_verified",
        ):
            if result.get(name) is not True:
                raise GridFutureSourceError(f"{field}.{name} did not pass")
    else:
        if result.get("observation_method") not in ALLOWED_ASSIGNMENT_METHODS:
            raise GridFutureSourceError(
                f"{field}.observation_method is not an allowed reviewed source"
            )
        if result.get("observed_before_map_start") is not True:
            raise GridFutureSourceError(
                f"{field} was not observed before map start"
            )
    return result


def validate_capture_context(raw: bytes) -> dict[str, Any]:
    value = _read_object(raw, "GRID capture context")
    expected = {
        "schema_version",
        "event_id",
        "series_id",
        "game_number",
        "league",
        "patch",
        "grid_game_id",
        "teams",
        "provider_fixture_attestation",
        "role_assignment_attestation",
        "role_assignments",
    }
    if set(value) != expected or value.get("schema_version") != CONTEXT_SCHEMA_VERSION:
        raise GridFutureSourceError("GRID capture context structure changed")
    for field in ("event_id", "league", "grid_game_id"):
        value[field] = _nonempty(value.get(field), f"context.{field}")
    series_id = _nonempty(value.get("series_id"), "context.series_id")
    if not series_id.isdigit():
        raise GridFutureSourceError("context.series_id must be numeric")
    value["series_id"] = series_id
    value["game_number"] = _sequence(value.get("game_number"), "context.game_number")
    patch = _nonempty(value.get("patch"), "context.patch")
    if not PATCH_RE.fullmatch(patch):
        raise GridFutureSourceError("context.patch is outside the frozen 26.xx format")
    value["patch"] = patch

    teams = value.get("teams")
    team_expected = {
        "side",
        "grid_team_id",
        "grid_team_name",
        "organization_id",
        "organization_name",
    }
    if not isinstance(teams, list) or len(teams) != 2:
        raise GridFutureSourceError("context.teams must contain blue and red")
    checked_teams: list[dict[str, str]] = []
    for team in teams:
        if not isinstance(team, Mapping) or set(team) != team_expected:
            raise GridFutureSourceError("context team structure changed")
        side = team.get("side")
        if side not in {"blue", "red"}:
            raise GridFutureSourceError("context team side is invalid")
        checked = {"side": str(side)}
        for field in team_expected - {"side"}:
            checked[field] = _nonempty(team.get(field), f"context.team.{field}")
        checked_teams.append(checked)
    if [team["side"] for team in checked_teams] != ["blue", "red"]:
        raise GridFutureSourceError("context teams must be ordered blue then red")
    for field in ("grid_team_id", "grid_team_name", "organization_id"):
        if len({team[field] for team in checked_teams}) != 2:
            raise GridFutureSourceError(f"context team {field} values are duplicated")
    value["teams"] = checked_teams
    value["provider_fixture_attestation"] = _validate_attestation(
        value.get("provider_fixture_attestation"),
        field="context.provider_fixture_attestation",
        kind="fixture",
    )
    value["role_assignment_attestation"] = _validate_attestation(
        value.get("role_assignment_attestation"),
        field="context.role_assignment_attestation",
        kind="assignment",
    )

    assignments = value.get("role_assignments")
    assignment_expected = {"side", "role", "champion_name"}
    if not isinstance(assignments, list) or len(assignments) != 10:
        raise GridFutureSourceError("context.role_assignments must contain ten rows")
    checked_assignments: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    champions: set[str] = set()
    for assignment in assignments:
        if not isinstance(assignment, Mapping) or set(assignment) != assignment_expected:
            raise GridFutureSourceError("context role assignment structure changed")
        side = assignment.get("side")
        role = assignment.get("role")
        champion = _nonempty(
            assignment.get("champion_name"), "context.assignment.champion_name"
        )
        if side not in {"blue", "red"} or role not in ROLES:
            raise GridFutureSourceError("context role assignment side or role is invalid")
        key = (str(side), str(role))
        if key in seen or champion in champions:
            raise GridFutureSourceError("context role assignments are duplicated")
        seen.add(key)
        champions.add(champion)
        checked_assignments.append(
            {"side": str(side), "role": str(role), "champion_name": champion}
        )
    expected_slots = {(side, role) for side in ("blue", "red") for role in ROLES}
    if seen != expected_slots:
        raise GridFutureSourceError("context role assignments do not fill both teams")
    value["role_assignments"] = checked_assignments
    return value


def _load_envelopes(raw: bytes, *, series_id: str) -> list[_EnvelopeRecord]:
    parts = raw.split(b"\n")
    if parts and parts[-1] == b"":
        parts.pop()
    if not parts or any(not part for part in parts):
        raise GridFutureSourceError("GRID receipt log contains no records or blank lines")
    records: list[_EnvelopeRecord] = []
    for index, line in enumerate(parts, 1):
        envelope = _read_object(line, f"GRID receipt envelope line {index}")
        try:
            checked, _, transaction = validate_received_transaction_envelope(envelope)
        except GridSeriesEventsError as exc:
            raise GridFutureSourceError(str(exc)) from exc
        if checked["series_id"] != series_id:
            raise GridFutureSourceError("GRID receipt log series identity changed")
        sequence = transaction_sequence(transaction)
        if sequence is None:
            raise GridFutureSourceError("GRID receipt transaction sequence is invalid")
        records.append(
            _EnvelopeRecord(
                sequence_number=sequence,
                received_at_utc=_timestamp(
                    checked["received_at_utc"], "GRID envelope received_at_utc"
                ),
                envelope_raw_sha256=_sha256_bytes(line),
                message_raw_sha256=str(checked["message"]["raw_sha256"]),
                transaction=transaction,
            )
        )
    if any(
        current.sequence_number <= previous.sequence_number
        for previous, current in zip(records, records[1:])
    ):
        raise GridFutureSourceError(
            "GRID receipt log transaction sequences are not strictly increasing"
        )
    if any(
        current.received_at_utc < previous.received_at_utc
        for previous, current in zip(records, records[1:])
    ):
        raise GridFutureSourceError(
            "GRID receipt log system receive times moved backwards"
        )
    return records


def _team_by_side(context: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(team["side"]): team for team in context["teams"]}


def _draft_action_from_event(
    *,
    event: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    event_type = event.get("type")
    if event_type not in {"team-picked-character", "team-banned-character"}:
        raise GridFutureSourceError("GRID draft event type is invalid")
    expected_kind = "pick" if event_type == "team-picked-character" else "ban"
    actor = _mapping(event.get("actor"), "GRID draft actor")
    target = _mapping(event.get("target"), "GRID draft target")
    actor_state = _mapping(actor.get("state"), "GRID draft actor state")
    if actor.get("type") != "team" or target.get("type") != "character":
        raise GridFutureSourceError("GRID draft actor or target type changed")
    side = actor_state.get("side")
    if side not in {"blue", "red"}:
        raise GridFutureSourceError("GRID draft actor has no blue/red side")
    team = _team_by_side(context)[str(side)]
    if (
        actor.get("id") != team["grid_team_id"]
        or actor_state.get("name") != team["grid_team_name"]
        or _mapping(actor_state.get("game"), "GRID draft actor game").get("id")
        != context["grid_game_id"]
    ):
        raise GridFutureSourceError("GRID draft team/game crosswalk changed")
    delta = _mapping(event.get("seriesStateDelta"), "GRID draft series delta")
    games = delta.get("games")
    if not isinstance(games, list) or len(games) != 1:
        raise GridFutureSourceError("GRID draft event must change exactly one game")
    game = _mapping(games[0], "GRID draft game delta")
    if game.get("id") != context["grid_game_id"]:
        raise GridFutureSourceError("GRID draft event game identity changed")
    draft_actions = game.get("draftActions")
    if not isinstance(draft_actions, list) or len(draft_actions) != 1:
        raise GridFutureSourceError("GRID draft event must contain exactly one action")
    action = _mapping(draft_actions[0], "GRID draft action")
    drafter = _mapping(action.get("drafter"), "GRID draft action drafter")
    draftable = _mapping(action.get("draftable"), "GRID draft action draftable")
    target_state = _mapping(target.get("state"), "GRID draft target state")
    action_id = _nonempty(action.get("id"), "GRID draft action id")
    character_id = _nonempty(draftable.get("id"), "GRID draft character id")
    champion_name = _nonempty(
        draftable.get("name"), "GRID draft champion name"
    )
    if (
        action.get("type") != expected_kind
        or drafter.get("type") != "team"
        or drafter.get("id") != team["grid_team_id"]
        or draftable.get("type") != "character"
        or target.get("id") != character_id
        or target_state.get("id") != character_id
        or target_state.get("name") != champion_name
    ):
        raise GridFutureSourceError("GRID draft action/event fields do not reconcile")
    return {
        "slot": _sequence(action.get("sequenceNumber"), "GRID draft action slot"),
        "action_id": action_id,
        "side": str(side),
        "kind": expected_kind,
        "champion_name": champion_name,
        "grid_character_id": character_id,
        "event_id": _nonempty(event.get("id"), "GRID draft event id"),
    }


def _draft_event_game_id(event: Mapping[str, Any]) -> str:
    delta = _mapping(event.get("seriesStateDelta"), "GRID draft series delta")
    games = delta.get("games")
    if not isinstance(games, list) or len(games) != 1:
        raise GridFutureSourceError("GRID draft event must change exactly one game")
    game = _mapping(games[0], "GRID draft game delta")
    return _nonempty(game.get("id"), "GRID draft game id")


def _target_start_from_event(
    event: Mapping[str, Any], *, grid_game_id: str
) -> datetime | None:
    if event.get("type") != "series-started-game":
        return None
    target = _mapping(event.get("target"), "GRID map-start target")
    if target.get("type") != "game" or target.get("id") != grid_game_id:
        return None
    target_delta = _mapping(target.get("stateDelta"), "GRID map-start target delta")
    series_delta = _mapping(
        event.get("seriesStateDelta"), "GRID map-start series delta"
    )
    games = series_delta.get("games")
    if not isinstance(games, list) or len(games) != 1:
        raise GridFutureSourceError("GRID map-start event must change exactly one game")
    game_delta = _mapping(games[0], "GRID map-start game delta")
    if (
        target_delta.get("id") != grid_game_id
        or target_delta.get("started") is not True
        or game_delta.get("id") != grid_game_id
        or game_delta.get("started") is not True
    ):
        raise GridFutureSourceError("GRID map-start game identity/state changed")
    target_start = _timestamp(
        target_delta.get("startedAt"), "GRID target map-start timestamp"
    )
    delta_start = _timestamp(
        game_delta.get("startedAt"), "GRID series map-start timestamp"
    )
    if target_start != delta_start:
        raise GridFutureSourceError("GRID map-start timestamps disagree")
    return target_start


def _transaction_identity(
    record: _EnvelopeRecord, *, event: Mapping[str, Any], event_index: int
) -> dict[str, Any]:
    transaction = record.transaction
    return {
        "transaction_id": _nonempty(transaction.get("id"), "GRID transaction id"),
        "transaction_sequence_number": record.sequence_number,
        "transaction_occurred_at_utc": _timestamp(
            transaction.get("occurredAt"), "GRID transaction occurredAt"
        ).isoformat(),
        "event_index": event_index,
        "event_id": _nonempty(event.get("id"), "GRID event id"),
        "message_raw_sha256": record.message_raw_sha256,
        "envelope_raw_sha256": record.envelope_raw_sha256,
        "received_at_utc": record.received_at_utc.isoformat(),
    }


def prepare_terminal_draft_inputs(
    *,
    receipt_log_raw: bytes,
    context_raw: bytes,
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> PreparedDraftInputs:
    context = validate_capture_context(context_raw)
    readiness = _registered_readiness(root)
    records = _load_envelopes(receipt_log_raw, series_id=context["series_id"])
    assignment_by_pick = {
        (item["side"], item["champion_name"]): item["role"]
        for item in context["role_assignments"]
    }
    extracted: list[dict[str, Any]] = []
    source_actions: list[dict[str, Any]] = []
    start_seen = False
    for record in records:
        events = record.transaction.get("events")
        if not isinstance(events, list):
            raise GridFutureSourceError("GRID transaction events must be a list")
        for event_index, event_value in enumerate(events):
            event = _mapping(event_value, "GRID transaction event")
            start = _target_start_from_event(
                event, grid_game_id=context["grid_game_id"]
            )
            if start is not None:
                start_seen = True
            if event.get("type") not in {
                "team-picked-character",
                "team-banned-character",
            }:
                continue
            if _draft_event_game_id(event) != context["grid_game_id"]:
                continue
            action = _draft_action_from_event(event=event, context=context)
            identity = _transaction_identity(
                record, event=event, event_index=event_index
            )
            action["transaction_sequence_number"] = record.sequence_number
            action["received_at_utc"] = record.received_at_utc
            action["transaction_occurred_at_utc"] = _timestamp(
                record.transaction.get("occurredAt"),
                "GRID draft transaction occurredAt",
            )
            extracted.append(action)
            source_actions.append(
                {
                    **identity,
                    "slot": action["slot"],
                    "side": action["side"],
                    "kind": action["kind"],
                    "grid_action_id": action["action_id"],
                    "grid_character_id": action["grid_character_id"],
                    "champion_name": action["champion_name"],
                }
            )
    if start_seen:
        raise GridFutureSourceError(
            "target map-start was already received; refusing a draft prediction"
        )
    if len(extracted) != 20 or [item["slot"] for item in extracted] != list(
        range(1, 21)
    ):
        raise GridFutureSourceError(
            "GRID terminal draft must contain contiguous action slots 1 through 20"
        )
    if any(
        current["transaction_sequence_number"]
        <= previous["transaction_sequence_number"]
        for previous, current in zip(extracted, extracted[1:])
    ):
        raise GridFutureSourceError(
            "GRID terminal draft transaction order is not strictly increasing"
        )
    if any(
        current["transaction_occurred_at_utc"]
        < previous["transaction_occurred_at_utc"]
        for previous, current in zip(extracted, extracted[1:])
    ):
        raise GridFutureSourceError("GRID draft provider times moved backwards")
    if len({item["action_id"] for item in extracted}) != 20:
        raise GridFutureSourceError("GRID terminal draft action ids are duplicated")
    if (
        len({item["grid_character_id"] for item in extracted}) != 20
        or len({item["champion_name"] for item in extracted}) != 20
    ):
        raise GridFutureSourceError("GRID terminal draft champions are duplicated")
    counts = {
        (side, kind): sum(
            item["side"] == side and item["kind"] == kind for item in extracted
        )
        for side in ("blue", "red")
        for kind in ("pick", "ban")
    }
    if any(value != 5 for value in counts.values()):
        raise GridFutureSourceError(
            "GRID terminal draft must contain five picks and five bans per side"
        )
    picked = {
        (item["side"], item["champion_name"])
        for item in extracted
        if item["kind"] == "pick"
    }
    if picked != set(assignment_by_pick):
        raise GridFutureSourceError(
            "reviewed role assignments do not exactly match the ten GRID picks"
        )

    validated_at = _clock_sample(clock, "GRID terminal draft validation")
    last_received = max(item["received_at_utc"] for item in extracted)
    if validated_at <= last_received:
        raise GridFutureSourceError(
            "GRID terminal draft validation did not follow its received evidence"
        )
    source_binding = _source_record(root, SOURCE_LOCATOR)
    transport_binding = _source_record(root, TRANSPORT_LOCATOR)
    source_payload = {
        "schema_version": DRAFT_SOURCE_SCHEMA_VERSION,
        "source_id": SOURCE_ID,
        "use_boundary": "private_outcome_free_future_evaluation_only",
        "series_id": context["series_id"],
        "grid_game_id": context["grid_game_id"],
        "receipt_log_raw_sha256": _sha256_bytes(receipt_log_raw),
        "capture_context_raw_sha256": _sha256_bytes(context_raw),
        "capture_context": context,
        "receipt_window": {
            "first_draft_transaction_sequence": extracted[0][
                "transaction_sequence_number"
            ],
            "last_draft_transaction_sequence": extracted[-1][
                "transaction_sequence_number"
            ],
            "first_draft_received_at_utc": min(
                item["received_at_utc"] for item in extracted
            ).isoformat(),
            "last_draft_received_at_utc": last_received.isoformat(),
            "system_clocked_receipts": True,
        },
        "terminal_actions": source_actions,
        "validation": {
            "validated_at_utc": validated_at.isoformat(),
            "action_slots_contiguous_1_through_20": True,
            "five_picks_and_five_bans_per_side": True,
            "blue_red_grid_team_crosswalk_exact": True,
            "reviewed_role_assignments_exactly_match_picks": True,
            "target_map_start_not_received": True,
            "raw_grid_messages_embedded_in_source_payload": False,
            "retrospective_archive_used_as_future_evidence": False,
        },
        "implementation": {
            "adapter": source_binding,
            "transport": transport_binding,
            "registered_grid_source_readiness": readiness,
        },
        "claim_ceiling": (
            "Exact pre-start pick-ban and reviewed role-assignment input for "
            "future evaluation only; no probability, odds, recommendation, or "
            "betting authority."
        ),
    }
    source_payload_raw = _canonical_bytes(source_payload)
    team_by_side = _team_by_side(context)
    side_picks: dict[str, dict[str, str]] = {"blue": {}, "red": {}}
    actions: list[dict[str, Any]] = []
    final_assignments: list[dict[str, Any]] = []
    for item in extracted:
        role = assignment_by_pick.get((item["side"], item["champion_name"]))
        actions.append(
            {
                "slot": item["slot"],
                "action_id": item["action_id"],
                "side": item["side"],
                "kind": item["kind"],
                "champion_id": item["champion_name"],
                "role_set": [role] if role is not None else [],
            }
        )
        if role is not None:
            side_picks[item["side"]][role] = item["champion_name"]
            final_assignments.append(
                {
                    "action_id": item["action_id"],
                    "side": item["side"],
                    "champion_id": item["champion_name"],
                    "role": role,
                }
            )
    if any(set(picks) != set(ROLES) for picks in side_picks.values()):
        raise GridFutureSourceError("GRID terminal role assignments are incomplete")
    metadata = {
        "schema_version": "scryglass:terminal-draft-capture-input:v1",
        "event_id": context["event_id"],
        "series_id": context["series_id"],
        "game_number": context["game_number"],
        "league": context["league"],
        "patch": context["patch"],
        "blue_organization_id": team_by_side["blue"]["organization_id"],
        "blue_organization_name": team_by_side["blue"]["organization_name"],
        "red_organization_id": team_by_side["red"]["organization_id"],
        "red_organization_name": team_by_side["red"]["organization_name"],
        "source": {
            "source_id": SOURCE_ID,
            "source_url": f"{SERIES_EVENTS_BASE}/{context['series_id']}",
            "source_record_id": (
                f"{context['series_id']}:{context['grid_game_id']}:"
                f"draft:{extracted[-1]['action_id']}"
            ),
            "available_at_utc": last_received.isoformat(),
            "rights_status": "reviewed",
            "payload_raw_sha256": _sha256_bytes(source_payload_raw),
        },
        "protocol_validation": {
            "protocol_id": "grid-series-events-terminal-draft-v1",
            "validator_id": SOURCE_LOCATOR,
            "validator_sha256": source_binding["raw_sha256"],
            "validated_at_utc": validated_at.isoformat(),
            "action_order_verified": True,
            "pick_ban_counts_verified": True,
            "blue_red_side_mapping_verified": True,
        },
        "blue": side_picks["blue"],
        "red": side_picks["red"],
        "actions": actions,
        "final_assignments": final_assignments,
    }
    return PreparedDraftInputs(
        metadata_raw=_canonical_bytes(metadata),
        source_payload_raw=source_payload_raw,
        validated_at_utc=validated_at,
        last_draft_received_at_utc=last_received,
    )


def prepare_map_start_inputs(
    *,
    receipt_log_raw: bytes,
    context_raw: bytes,
    root: Path = ROOT,
) -> PreparedMapStartInputs:
    context = validate_capture_context(context_raw)
    readiness = _registered_readiness(root)
    records = _load_envelopes(receipt_log_raw, series_id=context["series_id"])
    starts: list[tuple[_EnvelopeRecord, Mapping[str, Any], int, datetime]] = []
    draft_sequences: list[int] = []
    for record in records:
        events = record.transaction.get("events")
        if not isinstance(events, list):
            raise GridFutureSourceError("GRID transaction events must be a list")
        for event_index, event_value in enumerate(events):
            event = _mapping(event_value, "GRID transaction event")
            start = _target_start_from_event(
                event, grid_game_id=context["grid_game_id"]
            )
            if start is not None:
                starts.append((record, event, event_index, start))
            if event.get("type") in {
                "team-picked-character",
                "team-banned-character",
            }:
                if _draft_event_game_id(event) != context["grid_game_id"]:
                    continue
                action = _draft_action_from_event(event=event, context=context)
                draft_sequences.append(action["slot"])
    if len(starts) != 1:
        raise GridFutureSourceError(
            "GRID map-start capture requires exactly one target start event"
        )
    record, event, event_index, actual_start = starts[0]
    if record is not records[-1]:
        raise GridFutureSourceError(
            "GRID map-start receipt log must end at the target start event"
        )
    provider_occurred = _timestamp(
        record.transaction.get("occurredAt"), "GRID map-start occurredAt"
    )
    if provider_occurred < actual_start:
        raise GridFutureSourceError("GRID map-start event predates its startedAt value")
    if record.received_at_utc < actual_start:
        raise GridFutureSourceError(
            "system-clocked GRID map-start receipt predates provider startedAt"
        )
    if draft_sequences and draft_sequences != list(range(1, 21)):
        raise GridFutureSourceError(
            "GRID map-start log contains an incomplete target draft"
        )
    identity = _transaction_identity(
        record, event=event, event_index=event_index
    )
    source_payload = {
        "schema_version": MAP_START_SOURCE_SCHEMA_VERSION,
        "source_id": RECEIPT_SOURCE_ID,
        "use_boundary": "private_actual_map_start_evaluation_evidence_only",
        "series_id": context["series_id"],
        "grid_game_id": context["grid_game_id"],
        "receipt_log_raw_sha256": _sha256_bytes(receipt_log_raw),
        "capture_context_raw_sha256": _sha256_bytes(context_raw),
        "map_start_signal": {
            **identity,
            "event_type": "series-started-game",
            "actual_map_start_utc": actual_start.isoformat(),
        },
        "validation": {
            "target_game_identity_exact": True,
            "provider_start_fields_reconcile": True,
            "source_received_at_or_after_actual_start": True,
            "raw_grid_message_embedded_in_source_payload": False,
            "retrospective_archive_used_as_future_evidence": False,
        },
        "implementation": {
            "adapter": _source_record(root, SOURCE_LOCATOR),
            "transport": _source_record(root, TRANSPORT_LOCATOR),
            "registered_grid_source_readiness": readiness,
        },
        "claim_ceiling": (
            "Outcome-free actual map-start evidence only; no model, probability, "
            "odds, recommendation, or betting authority."
        ),
    }
    source_payload_raw = _canonical_bytes(source_payload)
    metadata = {
        "schema_version": "scryglass:actual-map-start-capture-input:v1",
        "event_id": context["event_id"],
        "series_id": context["series_id"],
        "game_number": context["game_number"],
        "league": context["league"],
        "patch": context["patch"],
        "actual_map_start_utc": actual_start.isoformat(),
        "source": {
            "source_id": RECEIPT_SOURCE_ID,
            "source_url": f"{SERIES_EVENTS_BASE}/{context['series_id']}",
            "source_record_id": (
                f"{context['series_id']}:{context['grid_game_id']}:"
                f"start:{identity['event_id']}"
            ),
            "available_at_utc": record.received_at_utc.isoformat(),
            "rights_status": "reviewed",
            "payload_raw_sha256": _sha256_bytes(source_payload_raw),
        },
    }
    return PreparedMapStartInputs(
        metadata_raw=_canonical_bytes(metadata),
        source_payload_raw=source_payload_raw,
        actual_map_start_utc=actual_start,
        source_received_at_utc=record.received_at_utc,
    )


def build_grid_draft_prediction_receipt(
    *,
    receipt_log_raw: bytes,
    context_raw: bytes,
    ratings_receipt_raw: bytes,
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    prepared = prepare_terminal_draft_inputs(
        receipt_log_raw=receipt_log_raw,
        context_raw=context_raw,
        root=root,
        clock=clock,
    )
    return ledger.build_draft_prediction_receipt(
        ratings_receipt_raw=ratings_receipt_raw,
        draft_metadata_raw=prepared.metadata_raw,
        draft_source_payload_raw=prepared.source_payload_raw,
        root=root,
        clock=clock,
    )


def build_grid_map_start_receipt(
    *,
    receipt_log_raw: bytes,
    context_raw: bytes,
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    prepared = prepare_map_start_inputs(
        receipt_log_raw=receipt_log_raw,
        context_raw=context_raw,
        root=root,
    )
    return ledger.build_map_start_receipt(
        map_start_metadata_raw=prepared.metadata_raw,
        map_start_source_payload_raw=prepared.source_payload_raw,
        root=root,
        clock=clock,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--receipt-log", type=Path, required=True)
    capture.add_argument("--context", type=Path, required=True)
    capture.add_argument("--ratings-receipt", type=Path, required=True)
    capture.add_argument("--out", type=Path, required=True)
    map_start = subparsers.add_parser("map-start")
    map_start.add_argument("--receipt-log", type=Path, required=True)
    map_start.add_argument("--context", type=Path, required=True)
    map_start.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    try:
        if args.command == "capture":
            payload = build_grid_draft_prediction_receipt(
                receipt_log_raw=args.receipt_log.read_bytes(),
                context_raw=args.context.read_bytes(),
                ratings_receipt_raw=args.ratings_receipt.read_bytes(),
                root=args.root,
            )
        else:
            payload = build_grid_map_start_receipt(
                receipt_log_raw=args.receipt_log.read_bytes(),
                context_raw=args.context.read_bytes(),
                root=args.root,
            )
        raw_sha256 = ledger.write_no_clobber(args.out, payload)
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc))
        return 2
    print(
        json.dumps(
            {
                "command": args.command,
                "output": str(args.out),
                "raw_sha256": raw_sha256,
                "artifact_sha256": payload["artifact_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONTEXT_SCHEMA_VERSION",
    "DRAFT_SOURCE_SCHEMA_VERSION",
    "GridFutureSourceError",
    "MAP_START_SOURCE_SCHEMA_VERSION",
    "PreparedDraftInputs",
    "PreparedMapStartInputs",
    "build_grid_draft_prediction_receipt",
    "build_grid_map_start_receipt",
    "prepare_map_start_inputs",
    "prepare_terminal_draft_inputs",
    "validate_capture_context",
]
