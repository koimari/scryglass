"""Strict G1 pre-event roster evidence for contextual Draft Score requests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from lol_kills.v2.data.common import ROLES, parse_rfc3339


G1_SCHEMA_VERSION = "scryglass:g1-roster-payload:v1"


class G1RosterError(ValueError):
    """Raised when a contextual roster payload is not independently usable."""


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise G1RosterError(f"{field} must be a non-empty string")
    return value.strip()


def _strict_json(raw: bytes) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise G1RosterError(f"roster payload contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except G1RosterError:
        raise
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise G1RosterError("roster payload must be strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise G1RosterError("roster payload must be a JSON object")
    return payload


def _starter_side(payload: Mapping[str, Any], side: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    if set(payload) != {"roster_id", "starters"}:
        raise G1RosterError(f"rosters.{side} keys do not match the frozen contract")
    roster_id = _nonempty(payload["roster_id"], f"rosters.{side}.roster_id")
    starters = payload["starters"]
    if not isinstance(starters, Sequence) or isinstance(starters, (str, bytes)) or len(starters) != len(ROLES):
        raise G1RosterError(f"rosters.{side}.starters must contain exactly five players")
    normalized: list[tuple[str, str]] = []
    for index, starter in enumerate(starters):
        if not isinstance(starter, Mapping) or set(starter) != {"role", "player_id"}:
            raise G1RosterError(f"rosters.{side}.starters[{index}] is malformed")
        role = _nonempty(starter["role"], f"rosters.{side}.starters[{index}].role")
        player_id = _nonempty(starter["player_id"], f"rosters.{side}.starters[{index}].player_id")
        if role not in ROLES:
            raise G1RosterError(f"rosters.{side}.starters[{index}].role is not canonical")
        normalized.append((role, player_id))
    if [role for role, _ in normalized] != list(ROLES):
        raise G1RosterError(f"rosters.{side}.starters must be ordered as {ROLES}")
    if len({player_id for _, player_id in normalized}) != len(normalized):
        raise G1RosterError(f"rosters.{side}.starters cannot repeat a player")
    return roster_id, tuple(normalized)


@dataclass(frozen=True)
class G1RosterEvidence:
    """A hash-bound, pre-event exact-starter source record."""

    event_start: str
    source_record_id: str
    source_available_at: str
    source_retrieved_at: str
    source_rights_status: str
    source_payload_sha256: str
    roster_a_id: str
    roster_b_id: str
    starters_a: tuple[tuple[str, str], ...]
    starters_b: tuple[tuple[str, str], ...]

    @classmethod
    def from_payload_bytes(cls, raw: bytes) -> "G1RosterEvidence":
        if not isinstance(raw, bytes) or not raw:
            raise G1RosterError("roster payload must be non-empty bytes")
        payload = _strict_json(raw)
        expected_keys = {
            "schema_version",
            "source_record_id",
            "event_start",
            "available_at",
            "retrieved_at",
            "rights_status",
            "rosters",
        }
        if set(payload) != expected_keys:
            raise G1RosterError("roster payload keys do not match the frozen G1 contract")
        if payload["schema_version"] != G1_SCHEMA_VERSION:
            raise G1RosterError("roster payload schema_version is not supported")
        event_start = _nonempty(payload["event_start"], "event_start")
        available_at = _nonempty(payload["available_at"], "available_at")
        retrieved_at = _nonempty(payload["retrieved_at"], "retrieved_at")
        start_time = parse_rfc3339(event_start)
        available_time = parse_rfc3339(available_at)
        retrieved_time = parse_rfc3339(retrieved_at)
        if available_time >= start_time:
            raise G1RosterError("roster source was not available before event_start")
        if retrieved_time < available_time:
            raise G1RosterError("roster retrieval time predates source availability")
        if payload["rights_status"] != "reviewed":
            raise G1RosterError("contextual roster rights must be reviewed")
        source_record_id = _nonempty(payload["source_record_id"], "source_record_id")
        rosters = payload["rosters"]
        if not isinstance(rosters, Mapping) or set(rosters) != {"A", "B"}:
            raise G1RosterError("rosters must contain exactly canonical sides A and B")
        roster_a_id, starters_a = _starter_side(rosters["A"], "A")
        roster_b_id, starters_b = _starter_side(rosters["B"], "B")
        player_ids = [player for _, player in (*starters_a, *starters_b)]
        if len(set(player_ids)) != len(player_ids):
            raise G1RosterError("contextual rosters cannot share a starter")
        if roster_a_id == roster_b_id:
            raise G1RosterError("contextual rosters require distinct roster ids")
        return cls(
            event_start=event_start,
            source_record_id=source_record_id,
            source_available_at=available_at,
            source_retrieved_at=retrieved_at,
            source_rights_status="reviewed",
            source_payload_sha256=hashlib.sha256(raw).hexdigest(),
            roster_a_id=roster_a_id,
            roster_b_id=roster_b_id,
            starters_a=starters_a,
            starters_b=starters_b,
        )

    def is_available_for(self, event_start: str) -> bool:
        try:
            return (
                self.source_rights_status == "reviewed"
                and parse_rfc3339(self.event_start) == parse_rfc3339(event_start)
                and parse_rfc3339(self.source_available_at) < parse_rfc3339(event_start)
            )
        except (TypeError, ValueError):
            return False

    def as_mapping(self) -> dict[str, Any]:
        return {
            "event_start": self.event_start,
            "source_record_id": self.source_record_id,
            "source_available_at": self.source_available_at,
            "source_retrieved_at": self.source_retrieved_at,
            "source_rights_status": self.source_rights_status,
            "source_payload_sha256": self.source_payload_sha256,
            "roster_a_id": self.roster_a_id,
            "roster_b_id": self.roster_b_id,
            "starters_a": [{"role": role, "player_id": player} for role, player in self.starters_a],
            "starters_b": [{"role": role, "player_id": player} for role, player in self.starters_b],
        }


__all__ = ["G1_SCHEMA_VERSION", "G1RosterError", "G1RosterEvidence"]
