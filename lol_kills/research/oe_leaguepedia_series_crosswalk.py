"""Build a fail-closed Oracle's Elixir to Leaguepedia series crosswalk.

This module joins frozen OE game rows to already captured Leaguepedia
``ScoreboardGames`` and ``MatchSchedule`` rows.  It does not fetch data and
it does not use match outcomes.  A row is assigned only when the team set,
competition mapping, patch evidence, and bounded timestamp identify one
ScoreboardGames row.  The ScoreboardGames game prefix must then identify one
MatchSchedule row.

The result is research-only.  A partial result can describe authoritative
coverage for the mapped rows, but it never claims complete census coverage.
The caller must pass raw source bytes or regular files.  This lets the
builder verify the source hashes instead of trusting metadata supplied by a
caller.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from lol_kills.etl.aliases import normalize_team
from lol_kills.v2.tierlists.accepted_census import canonical_game_ids, identity_sha256


SCHEMA_VERSION = "scryglass:oe-leaguepedia-series-crosswalk:v1"
SOURCE_RECORD_LABELS = ("oe", "scoreboardgames", "matchschedule")
DEFAULT_MAX_GAME_TIME_DELTA_SECONDS = 300
DEFAULT_MAX_FIRST_GAME_SCHEDULE_DELTA_SECONDS = 6 * 60 * 60
DEFAULT_MAX_LATER_GAME_SCHEDULE_AGE_SECONDS = 6 * 60 * 60
_HEX64 = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


class CrosswalkError(ValueError):
    """Raised when source evidence cannot support a safe crosswalk."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CrosswalkError("source contains non-canonical JSON") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _payload_hash(value: Sequence[Mapping[str, Any]]) -> tuple[str, int]:
    raw = _canonical_json_bytes([dict(row) for row in value])
    return _sha256_bytes(raw), len(raw)


def _assignment_rows(
    assignments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in assignments]
    try:
        rows.sort(key=lambda row: str(row["oe_game_id"]))
    except (KeyError, TypeError) as error:
        raise CrosswalkError("crosswalk assignments are invalid") from error
    return rows


def _assignment_sha256(assignments: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(_canonical_json_bytes(_assignment_rows(assignments)))


def _parse_timestamp(value: Any, *, field: str) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        raise CrosswalkError(f"{field} timestamp is missing")
    candidate = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        parsed = None
    if parsed is None:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                parsed = datetime.strptime(raw[:19], fmt).replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue
    if parsed is None:
        raise CrosswalkError(f"{field} timestamp is malformed")
    if parsed.tzinfo is None:
        # Cargo timestamps are UTC even though Cargo omits the offset.
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _timestamp(row: Mapping[str, Any], *, label: str) -> datetime:
    fields = (
        "DateTime UTC",
        "DateTime_UTC",
        "datetime_utc",
        "start_utc",
        "date",
        "start_time",
        "timestamp",
    )
    for field in fields:
        if field in row and str(row.get(field) or "").strip():
            return _parse_timestamp(row.get(field), field=f"{label}.{field}")
    raise CrosswalkError(f"{label} has no supported timestamp field")


def _text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _norm(value: Any) -> str:
    text = unicodedata.normalize("NFKC", _text(value)).casefold()
    return " ".join(text.split())


def _team_key(value: Any) -> str:
    canonical = normalize_team(_text(value))
    return _norm(canonical)


def _alias_name_key(value: Any) -> str:
    text = unicodedata.normalize("NFKC", _text(value)).casefold()
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text)
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _team_set(
    values: Iterable[Any], *, aliases: Mapping[str, str] | None = None
) -> frozenset[str]:
    resolved = (
        aliases.get(_alias_name_key(value), _text(value)) if aliases else value
        for value in values
    )
    result = frozenset(_team_key(value) for value in resolved if _team_key(value))
    if len(result) != 2:
        raise CrosswalkError("team set must contain exactly two distinct teams")
    return result


def _row_team_set(
    row: Mapping[str, Any],
    *,
    label: str,
    aliases: Mapping[str, str] | None = None,
) -> frozenset[str]:
    list_fields = ("teams", "team_set")
    for field in list_fields:
        values = row.get(field)
        if isinstance(values, (list, tuple)):
            try:
                return _team_set(values, aliases=aliases)
            except CrosswalkError as error:
                raise CrosswalkError(f"{label}.{field}: {error}") from error
    pairs = (
        ("Team1", "Team2"),
        ("team1", "team2"),
        ("blue_team", "red_team"),
        ("blue", "red"),
        ("team_a", "team_b"),
    )
    for left, right in pairs:
        if left in row or right in row:
            try:
                return _team_set((row.get(left), row.get(right)), aliases=aliases)
            except CrosswalkError as error:
                raise CrosswalkError(f"{label}.{left}/{right}: {error}") from error
    raise CrosswalkError(f"{label} has no supported team pair")


def _first(row: Mapping[str, Any], names: Sequence[str]) -> str:
    for name in names:
        value = _text(row.get(name))
        if value:
            return value
    return ""


def _oe_game_id(row: Mapping[str, Any]) -> str:
    raw = _first(row, ("gameid", "game_uid", "game_id", "GameId"))
    if not raw:
        raise CrosswalkError("OE row has no game identity")
    values = canonical_game_ids((raw,))
    if len(values) != 1:
        raise CrosswalkError("OE row has no canonical game identity")
    return values[0]


def _scoreboard_game_id(row: Mapping[str, Any]) -> str:
    value = _first(row, ("GameId", "game_id", "gameid"))
    if not value:
        raise CrosswalkError("ScoreboardGames row has no GameId")
    return value


def _schedule_match_id(row: Mapping[str, Any]) -> str:
    value = _first(row, ("MatchId", "match_id", "series_id"))
    if not value:
        raise CrosswalkError("MatchSchedule row has no MatchId")
    return value


def _game_prefix_and_order(game_id: str) -> tuple[str, int]:
    prefix, separator, ordinal = game_id.rpartition("_")
    if not separator or not prefix or not ordinal.isdigit() or int(ordinal) < 1:
        raise CrosswalkError(f"ScoreboardGames.GameId has no positive ordinal: {game_id}")
    return prefix, int(ordinal)


def _mapping_section(mapping: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    value = mapping.get(section)
    if isinstance(value, Mapping):
        return value
    return {}


def _resolve_tournament_mapping(
    mapping: Mapping[str, Any], source_tournament: str
) -> Mapping[str, Any] | None:
    """Select an exact source-tournament submapping when one is declared."""

    table = mapping.get("tournaments", mapping.get("tournament_map"))
    if not isinstance(table, Mapping):
        return mapping
    if not source_tournament:
        return None
    source_key = _norm(source_tournament)
    for key, value in table.items():
        if _norm(key) == source_key:
            if not isinstance(value, Mapping):
                return None
            merged = dict(mapping)
            merged.update(dict(value))
            return merged
    return None


def _mapping_values(section: Mapping[str, Any], *names: str) -> tuple[str, ...]:
    values: list[str] = []
    for name in names:
        value = section.get(name)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
            values.extend(str(item) for item in value)
    return tuple(_norm(value) for value in values if _norm(value))


def _competition_matches(
    row: Mapping[str, Any],
    section: Mapping[str, Any],
    *,
    require_constraint: bool,
) -> tuple[bool, dict[str, Any]]:
    """Apply explicit Leaguepedia competition mapping evidence."""

    constraints = {
        "league": _mapping_values(section, "league", "leagues"),
        "overview_page": _mapping_values(
            section, "overview_page", "overview_pages", "pages"
        ),
        "tournament": _mapping_values(section, "tournament", "tournaments"),
    }
    if require_constraint and not any(constraints.values()):
        return False, {"reason": "competition_mapping_has_no_constraints"}
    evidence: dict[str, Any] = {}
    row_values = {
        "league": _norm(_first(row, ("League", "league"))),
        "overview_page": _norm(_first(row, ("OverviewPage", "overview_page"))),
        "tournament": _norm(_first(row, ("Tournament", "tournament"))),
    }
    for field, allowed in constraints.items():
        if not allowed:
            continue
        if row_values[field] not in allowed:
            return False, {"reason": f"{field}_not_in_explicit_mapping", field: row_values[field]}
        evidence[field] = row_values[field]
    return True, evidence


def _patch_matches(
    source_patch: str,
    row: Mapping[str, Any],
    section: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    if not source_patch:
        return True, {"source_patch_available": False, "row_patch_available": False}
    row_patch = _norm(_first(row, ("Patch", "patch", "patch_version")))
    if not row_patch:
        return True, {"source_patch_available": True, "row_patch_available": False}
    patch_map = section.get("patches", section.get("patch_map", {}))
    if not patch_map and isinstance(section.get("scoreboard"), Mapping):
        scoreboard_section = section["scoreboard"]
        patch_map = scoreboard_section.get("patches", scoreboard_section.get("patch_map", {}))
    allowed: tuple[str, ...]
    if isinstance(patch_map, Mapping):
        value = patch_map.get(source_patch) or patch_map.get(_norm(source_patch))
        if isinstance(value, str):
            allowed = (_norm(value),)
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
            allowed = tuple(_norm(item) for item in value if _norm(item))
        else:
            allowed = (_norm(source_patch),)
    else:
        allowed = (_norm(source_patch),)
    matched = row_patch in allowed
    return matched, {
        "source_patch": _norm(source_patch),
        "row_patch": row_patch,
        "allowed_row_patches": list(allowed),
    }


def _validate_source_receipt(
    source_receipt: Mapping[str, Any], selected_ids: Sequence[str]
) -> dict[str, Any]:
    if not isinstance(source_receipt, Mapping):
        raise CrosswalkError("a verified source receipt is required")
    raw_ids = source_receipt.get("accepted_game_ids")
    if not isinstance(raw_ids, list) or not raw_ids or not all(isinstance(value, str) for value in raw_ids):
        raise CrosswalkError("source receipt accepted_game_ids are invalid")
    accepted_ids = tuple(canonical_game_ids(raw_ids))
    if list(accepted_ids) != raw_ids:
        raise CrosswalkError("source receipt accepted_game_ids are not canonical and unique")
    receipt_hash = str(source_receipt.get("receipt_sha256") or "").lower()
    if not _HEX64.fullmatch(receipt_hash):
        raise CrosswalkError("source receipt hash is required")
    payload = dict(source_receipt)
    payload.pop("receipt_sha256", None)
    if _sha256_bytes(_canonical_json_bytes(payload)) != receipt_hash:
        raise CrosswalkError("source receipt hash does not match payload")
    source_authority = source_receipt.get("authority")
    if isinstance(source_authority, Mapping):
        if any(
            source_authority.get(field) is True
            for field in (
                "public",
                "public_player_rating",
                "public_team_rating",
                "public_probability",
                "promotion",
                "deployment",
            )
        ):
            raise CrosswalkError("source receipt grants public authority")
    try:
        count = int(source_receipt.get("source_game_count"))
    except (TypeError, ValueError) as error:
        raise CrosswalkError("source receipt source_game_count is invalid") from error
    claimed_identity = str(source_receipt.get("source_identity_sha256") or "").lower()
    if count != len(accepted_ids) or claimed_identity != identity_sha256(accepted_ids):
        raise CrosswalkError("source receipt census identity is invalid")
    model_ids_value = source_receipt.get("model_eligible_game_ids")
    model_binding: dict[str, Any] = {}
    if model_ids_value is not None:
        if not isinstance(model_ids_value, list) or not model_ids_value or not all(
            isinstance(value, str) for value in model_ids_value
        ):
            raise CrosswalkError("source receipt model_eligible_game_ids are invalid")
        model_ids = tuple(canonical_game_ids(model_ids_value))
        if list(model_ids) != model_ids_value or not set(model_ids).issubset(set(accepted_ids)):
            raise CrosswalkError("source receipt model-eligible census identity is invalid")
        try:
            model_count = int(source_receipt.get("model_eligible_game_count"))
        except (TypeError, ValueError) as error:
            raise CrosswalkError("source receipt model_eligible_game_count is invalid") from error
        model_identity = str(source_receipt.get("model_eligible_identity_sha256") or "").lower()
        if model_count != len(model_ids) or model_identity != identity_sha256(model_ids):
            raise CrosswalkError("source receipt model-eligible census identity is invalid")
        model_binding = {
            "model_eligible_game_count": len(model_ids),
            "model_eligible_game_identity_sha256": identity_sha256(model_ids),
            "model_eligible_game_ids": list(model_ids),
        }
    selected = tuple(canonical_game_ids(selected_ids))
    if not set(selected).issubset(set(accepted_ids)):
        raise CrosswalkError("OE rows contain game IDs outside the accepted census")
    return {
        "receipt_sha256": receipt_hash,
        "accepted_game_count": len(accepted_ids),
        "accepted_game_identity_sha256": identity_sha256(accepted_ids),
        "accepted_game_ids": list(accepted_ids),
        "selected_game_count": len(selected),
        "selected_game_identity_sha256": identity_sha256(selected),
        "selected_game_ids": list(selected),
        "selected_is_full_accepted_census": selected == accepted_ids,
        **model_binding,
    }


def _read_and_verify_source_record(
    label: str,
    record: Mapping[str, Any],
    payload: Sequence[Mapping[str, Any]],
    raw_source_bytes: Mapping[str, bytes] | None,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise CrosswalkError(f"source record is invalid: {label}")
    locator = _first(record, ("url", "locator", "path"))
    retrieved_at = _first(record, ("retrieved_at", "retrieval_time", "captured_at"))
    if not locator or not retrieved_at:
        raise CrosswalkError(f"source record needs locator and retrieval time: {label}")
    _parse_timestamp(retrieved_at, field=f"source_records.{label}.retrieved_at")
    expected_hash = str(record.get("sha256") or record.get("raw_sha256") or "").lower()
    expected_bytes = record.get("bytes", record.get("raw_bytes"))
    source_bytes: bytes | None = None
    if raw_source_bytes is not None and label in raw_source_bytes:
        source_bytes = bytes(raw_source_bytes[label])
    else:
        path_text = _first(record, ("path",))
        if path_text:
            path = Path(path_text)
            if not path.is_file() or path.is_symlink():
                raise CrosswalkError(f"source file is missing or unsafe: {label}")
            try:
                source_bytes = path.read_bytes()
            except OSError as error:
                raise CrosswalkError(f"source file cannot be read: {label}") from error
    if source_bytes is None:
        raise CrosswalkError(f"raw source bytes or a source path are required: {label}")
    actual_hash = _sha256_bytes(source_bytes)
    if not _HEX64.fullmatch(expected_hash) or expected_hash != actual_hash:
        raise CrosswalkError(f"source file hash does not match: {label}")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes != len(source_bytes):
        raise CrosswalkError(f"source file byte count does not match: {label}")
    payload_hash, payload_bytes = _payload_hash(payload)
    claimed_payload_hash = str(record.get("payload_sha256") or "").lower()
    claimed_payload_bytes = record.get("payload_bytes")
    if claimed_payload_hash != payload_hash or claimed_payload_bytes != payload_bytes:
        raise CrosswalkError(f"source payload hash does not match: {label}")
    return {
        "locator": locator,
        "retrieved_at": _parse_timestamp(retrieved_at, field="retrieved_at").isoformat().replace("+00:00", "Z"),
        "sha256": actual_hash,
        "bytes": len(source_bytes),
        "payload_sha256": payload_hash,
        "payload_bytes": payload_bytes,
        "integrity_verified": True,
    }


def _validate_source_records(
    source_records: Mapping[str, Mapping[str, Any]],
    payloads: Mapping[str, Sequence[Mapping[str, Any]]],
    raw_source_bytes: Mapping[str, bytes] | None,
) -> dict[str, Any]:
    if set(source_records) != set(SOURCE_RECORD_LABELS):
        raise CrosswalkError("source records must cover OE, ScoreboardGames, and MatchSchedule")
    return {
        label: _read_and_verify_source_record(
            label, source_records[label], payloads[label], raw_source_bytes
        )
        for label in SOURCE_RECORD_LABELS
    }


def _prepared_scoreboard_rows(rows: Sequence[Mapping[str, Any]], issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        row = dict(raw)
        try:
            game_id = _scoreboard_game_id(row)
            prefix, order = _game_prefix_and_order(game_id)
            if game_id in seen:
                raise CrosswalkError(f"duplicate GameId: {game_id}")
            teams = _row_team_set(row, label=f"scoreboard[{index}]")
            stamp = _timestamp(row, label=f"scoreboard[{index}]")
            tournament = _first(row, ("Tournament", "tournament"))
            seen.add(game_id)
            prepared.append({**row, "_game_id": game_id, "_prefix": prefix, "_order": order, "_teams": teams, "_stamp": stamp, "_tournament": tournament})
        except CrosswalkError as error:
            issues.append({"kind": "invalid_scoreboard_row", "index": index, "error": str(error)})
    return prepared


def _prepared_schedule_rows(rows: Sequence[Mapping[str, Any]], issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        row = dict(raw)
        try:
            match_id = _schedule_match_id(row)
            if match_id in seen:
                raise CrosswalkError(f"duplicate MatchId: {match_id}")
            teams = _row_team_set(row, label=f"schedule[{index}]")
            stamp = _timestamp(row, label=f"schedule[{index}]")
            seen.add(match_id)
            prepared.append({**row, "_match_id": match_id, "_teams": teams, "_stamp": stamp})
        except CrosswalkError as error:
            issues.append({"kind": "invalid_schedule_row", "index": index, "error": str(error)})
    return prepared


def build_oe_leaguepedia_series_crosswalk(
    oe_rows: Sequence[Mapping[str, Any]],
    scoreboard_rows: Sequence[Mapping[str, Any]],
    schedule_rows: Sequence[Mapping[str, Any]],
    *,
    source_receipt: Mapping[str, Any],
    source_records: Mapping[str, Mapping[str, Any]],
    competition_mapping: Mapping[str, Mapping[str, Any]],
    captured_at: str,
    raw_source_bytes: Mapping[str, bytes] | None = None,
    oe_team_aliases: Mapping[str, str] | None = None,
    alias_binding: Mapping[str, Any] | None = None,
    allow_partial: bool = False,
    max_game_time_delta_seconds: int = DEFAULT_MAX_GAME_TIME_DELTA_SECONDS,
    max_first_game_schedule_delta_seconds: int = DEFAULT_MAX_FIRST_GAME_SCHEDULE_DELTA_SECONDS,
    max_later_game_schedule_age_seconds: int = DEFAULT_MAX_LATER_GAME_SCHEDULE_AGE_SECONDS,
) -> dict[str, Any]:
    """Build one source-bound crosswalk from captured arrays.

    ``competition_mapping`` is explicit by source league.  Each entry can
    contain ``scoreboard`` and ``schedule`` sections with exact values for
    ``league``, ``overview_page``, and ``tournament``.  Its ``patches`` map
    can sit at the entry root or in the scoreboard section.  It maps an OE
    patch to one or more Leaguepedia patch tokens.
    """

    if not isinstance(captured_at, str) or not captured_at.strip():
        raise CrosswalkError("captured_at is required")
    captured_stamp = _parse_timestamp(captured_at, field="captured_at")
    for name, value in (
        ("oe_rows", oe_rows),
        ("scoreboard_rows", scoreboard_rows),
        ("schedule_rows", schedule_rows),
    ):
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise CrosswalkError(f"{name} must be an array")
        if any(not isinstance(row, Mapping) for row in value):
            raise CrosswalkError(f"{name} contains a non-object row")
    if max_game_time_delta_seconds < 0 or max_first_game_schedule_delta_seconds < 0 or max_later_game_schedule_age_seconds < 0:
        raise CrosswalkError("timestamp bounds must be non-negative")

    issues: list[dict[str, Any]] = []
    oe_prepared: list[dict[str, Any]] = []
    seen_oe_ids: set[str] = set()
    for index, raw in enumerate(oe_rows):
        row = dict(raw)
        try:
            game_id = _oe_game_id(row)
            if game_id in seen_oe_ids:
                raise CrosswalkError(f"duplicate OE game ID: {game_id}")
            stable_team_keys = row.get("team_keys")
            source_teams = (
                _team_set(stable_team_keys)
                if isinstance(stable_team_keys, (list, tuple))
                else _row_team_set(row, label=f"oe[{index}]")
            )
            teams = _row_team_set(
                row,
                label=f"oe[{index}]",
                aliases=oe_team_aliases,
            )
            stamp = _timestamp(row, label=f"oe[{index}]")
            league = _first(row, ("league", "League"))
            if not league:
                raise CrosswalkError("OE league is missing")
            patch = _first(row, ("patch", "Patch", "patch_version"))
            tournament = _first(row, ("tournament", "Tournament", "event"))
            if _norm(league) not in {_norm(key) for key in competition_mapping}:
                raise CrosswalkError(f"no explicit competition mapping for source league: {league}")
            seen_oe_ids.add(game_id)
            oe_prepared.append({**row, "_game_id": game_id, "_teams": teams, "_source_teams": source_teams, "_stamp": stamp, "_league": league, "_patch": patch, "_tournament": tournament})
        except CrosswalkError as error:
            issues.append({"kind": "invalid_oe_row", "index": index, "error": str(error)})

    source_binding = _validate_source_receipt(source_receipt, [row["_game_id"] for row in oe_prepared])
    normalized_source_records = _validate_source_records(
        source_records,
        {"oe": oe_rows, "scoreboardgames": scoreboard_rows, "matchschedule": schedule_rows},
        raw_source_bytes,
    )
    scoreboard = _prepared_scoreboard_rows(scoreboard_rows, issues)
    schedule = _prepared_schedule_rows(schedule_rows, issues)
    schedules_by_match: dict[str, list[dict[str, Any]]] = {}
    for row in schedule:
        schedules_by_match.setdefault(row["_match_id"], []).append(row)

    assignments: list[dict[str, Any]] = []
    used_scoreboard_ids: set[str] = set()
    for oe in oe_prepared:
        source_league_key = _norm(oe["_league"])
        mapping = next(
            (value for key, value in competition_mapping.items() if _norm(key) == source_league_key),
            None,
        )
        if not isinstance(mapping, Mapping):
            issues.append({"kind": "competition_mapping_missing", "oe_game_id": oe["_game_id"]})
            continue
        mapping = _resolve_tournament_mapping(mapping, oe["_tournament"])
        if not isinstance(mapping, Mapping):
            issues.append({
                "kind": "tournament_mapping_missing",
                "oe_game_id": oe["_game_id"],
                "source_tournament": oe["_tournament"],
            })
            continue
        scoreboard_section = _mapping_section(mapping, "scoreboard")
        schedule_section = _mapping_section(mapping, "schedule")
        candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for candidate in scoreboard:
            if candidate["_teams"] != oe["_teams"]:
                continue
            delta = abs((candidate["_stamp"] - oe["_stamp"]).total_seconds())
            if delta > max_game_time_delta_seconds:
                continue
            competition_ok, competition_evidence = _competition_matches(
                candidate, scoreboard_section, require_constraint=True
            )
            if not competition_ok:
                continue
            patch_ok, patch_evidence = _patch_matches(oe["_patch"], candidate, mapping)
            if not patch_ok:
                continue
            candidates.append((candidate, {"timestamp_delta_seconds": delta, "competition": competition_evidence, "patch": patch_evidence}))
        if len(candidates) != 1:
            issues.append({
                "kind": "scoreboard_identity_ambiguous" if len(candidates) > 1 else "scoreboard_identity_missing",
                "oe_game_id": oe["_game_id"],
                "candidate_count": len(candidates),
                "candidate_game_ids": [candidate["_game_id"] for candidate, _ in candidates],
            })
            continue
        selected, evidence = candidates[0]
        scoreboard_id = selected["_game_id"]
        if scoreboard_id in used_scoreboard_ids:
            issues.append({"kind": "duplicate_source_assignment", "oe_game_id": oe["_game_id"], "scoreboard_game_id": scoreboard_id})
            continue
        match_id = selected["_prefix"]
        match_candidates = schedules_by_match.get(match_id, [])
        schedule_candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for schedule_row in match_candidates:
            if schedule_row["_teams"] != oe["_teams"]:
                continue
            competition_ok, schedule_competition_evidence = _competition_matches(
                schedule_row, schedule_section, require_constraint=True
            )
            if not competition_ok:
                continue
            patch_ok, schedule_patch_evidence = _patch_matches(oe["_patch"], schedule_row, mapping)
            if not patch_ok:
                continue
            schedule_delta = (oe["_stamp"] - schedule_row["_stamp"]).total_seconds()
            order = int(selected["_order"])
            schedule_bound = (
                max_first_game_schedule_delta_seconds
                if order == 1
                else max_later_game_schedule_age_seconds
            )
            if abs(schedule_delta) > schedule_bound:
                continue
            schedule_candidates.append((schedule_row, {
                "series_timestamp_delta_seconds": schedule_delta,
                "competition": schedule_competition_evidence,
                "patch": schedule_patch_evidence,
            }))
        if len(schedule_candidates) != 1:
            issues.append({
                "kind": "schedule_identity_ambiguous" if len(schedule_candidates) > 1 else "schedule_identity_missing",
                "oe_game_id": oe["_game_id"],
                "scoreboard_game_id": scoreboard_id,
                "match_id": match_id,
                "candidate_count": len(schedule_candidates),
            })
            continue
        schedule_row, schedule_evidence = schedule_candidates[0]
        scoreboard_tournament = str(selected.get("_tournament") or "").strip()
        if not scoreboard_tournament:
            issues.append(
                {
                    "kind": "scoreboard_tournament_missing",
                    "oe_game_id": oe["_game_id"],
                    "scoreboard_game_id": scoreboard_id,
                    "series_id": match_id,
                }
            )
            continue
        used_scoreboard_ids.add(scoreboard_id)
        assignments.append({
            "oe_game_id": oe["_game_id"],
            "scoreboard_game_id": scoreboard_id,
            "scoreboard_game_order": int(selected["_order"]),
            "series_id": match_id,
            "normalized_team_set": sorted(oe["_source_teams"]),
            "oe_timestamp": oe["_stamp"].isoformat().replace("+00:00", "Z"),
            "scoreboard_timestamp": selected["_stamp"].isoformat().replace("+00:00", "Z"),
            "series_timestamp": schedule_row["_stamp"].isoformat().replace("+00:00", "Z"),
            "source_league": oe["_league"],
            "source_tournament": oe["_tournament"] or None,
            "source_patch": oe["_patch"] or None,
            "scoreboard_tournament": scoreboard_tournament,
            "scoreboard_game_id_prefix": selected["_prefix"],
            "evidence": {**evidence, "schedule": schedule_evidence},
            "outcome_used": False,
            "assignment_method": "exact_team_set_competition_patch_bounded_timestamp_then_exact_game_id_prefix",
        })

    assignments.sort(key=lambda row: (row["oe_timestamp"], row["oe_game_id"]))
    assignments_by_series: dict[str, list[dict[str, Any]]] = {}
    for assignment in assignments:
        assignments_by_series.setdefault(str(assignment["series_id"]), []).append(
            assignment
        )
    conflicting_series: set[str] = set()
    for series_id, series_assignments in assignments_by_series.items():
        values_by_key: dict[str, str] = {}
        for assignment in series_assignments:
            value = str(assignment.get("scoreboard_tournament") or "").strip()
            key = _norm(value)
            if not key:
                issues.append(
                    {
                        "kind": "series_tournament_missing",
                        "series_id": series_id,
                        "oe_game_ids": sorted(
                            str(row["oe_game_id"]) for row in series_assignments
                        ),
                    }
                )
                conflicting_series.add(series_id)
                break
            values_by_key.setdefault(key, value)
        if len(values_by_key) > 1:
            conflicting_series.add(series_id)
            issues.append(
                {
                    "kind": "series_tournament_conflict",
                    "series_id": series_id,
                    "scoreboard_tournaments": sorted(values_by_key.values()),
                    "oe_game_ids": sorted(
                        str(row["oe_game_id"]) for row in series_assignments
                    ),
                }
            )
    if conflicting_series:
        assignments = [
            assignment
            for assignment in assignments
            if str(assignment["series_id"]) not in conflicting_series
        ]
    selected_ids = [row["_game_id"] for row in oe_prepared]
    mapped_ids = [row["oe_game_id"] for row in assignments]
    if len(assignments) != len(oe_prepared):
        issues.append({
            "kind": "incomplete_crosswalk",
            "mapped_game_count": len(assignments),
            "selected_game_count": len(oe_prepared),
            "unmatched_game_ids": sorted(set(selected_ids) - set(mapped_ids)),
        })
    complete_selected = len(assignments) == len(oe_prepared) and not issues
    if not complete_selected and not allow_partial:
        status = "rejected_incomplete"
    elif complete_selected:
        status = "complete_authoritative_coverage"
    else:
        status = "partial_authoritative_coverage"

    series: dict[str, dict[str, Any]] = {}
    for assignment in assignments:
        item = series.setdefault(assignment["series_id"], {
            "series_id": assignment["series_id"],
            "normalized_team_set": assignment["normalized_team_set"],
            "scoreboard_tournament": assignment["scoreboard_tournament"],
            "game_orders": [],
            "oe_game_ids": [],
        })
        item["game_orders"].append(assignment["scoreboard_game_order"])
        item["oe_game_ids"].append(assignment["oe_game_id"])
    for item in series.values():
        item["game_orders"].sort()
        item["oe_game_ids"].sort()

    coverage = {
        "selected_game_count": len(oe_prepared),
        "mapped_game_count": len(assignments),
        "unmatched_game_count": len(oe_prepared) - len(assignments),
        "accepted_game_count": source_binding["accepted_game_count"],
        "mapped_selected_coverage": (len(assignments) / len(oe_prepared)) if oe_prepared else 0.0,
        "mapped_accepted_coverage": len(assignments) / source_binding["accepted_game_count"],
        "selected_is_full_accepted_census": source_binding["selected_is_full_accepted_census"],
        "mapped_is_full_accepted_census": set(mapped_ids) == set(source_binding["accepted_game_ids"]),
        "scope": "mapped_rows_only" if not complete_selected else "selected_census",
        "authority": "research_only_authoritative_for_mapped_rows",
        "complete": complete_selected,
    }
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "captured_at": captured_stamp.isoformat().replace("+00:00", "Z"),
        "status": status,
        "authority": {
            "research_only": True,
            "public": False,
            "probability": False,
            "draft": False,
            "promotion": False,
            "deployment": False,
        },
        "source_binding": source_binding,
        "source_records": normalized_source_records,
        "alias_binding": dict(alias_binding or {}),
        "competition_mapping": {
            str(key): dict(value) if isinstance(value, Mapping) else value
            for key, value in competition_mapping.items()
        },
        "raw_sources": {
            "oe": [dict(row) for row in oe_rows],
            "scoreboardgames": [dict(row) for row in scoreboard_rows],
            "matchschedule": [dict(row) for row in schedule_rows],
        },
        "join_contract": {
            "team_identity": "verified_alias_normalized_unordered_two_team_set_for_join; original_OE_team_set_for_assignment_binding",
            "competition_mapping": "explicit_source_league_to_scoreboard_and_schedule_values",
            "patch_identity": "match_when_both_source_and_target_patch_are_available",
            "timestamp_bounds": {
                "game_seconds": max_game_time_delta_seconds,
                "first_game_schedule_absolute_seconds": max_first_game_schedule_delta_seconds,
                "later_game_schedule_absolute_seconds": max_later_game_schedule_age_seconds,
            },
            "timestamp_timezone": "UTC; naive Leaguepedia Cargo values are interpreted as UTC",
            "game_id_prefix_to_match_id": "exact",
            "one_to_one": True,
            "ambiguity_policy": "reject",
            "duplicate_policy": "reject",
            "unmatched_policy": "reject_assignment_and_record_issue",
            "outcome_used": False,
            "outcome_policy": "ignored_and_never_used_for_matching",
            "tournament_binding": {
                "source": "ScoreboardGames.Tournament",
                "assignment_field": "scoreboard_tournament",
                "series_policy": "one_non_empty_value_per_series",
                "conflict_policy": "reject_series",
            },
        },
        "coverage": coverage,
        "assignments": assignments,
        "series": sorted(series.values(), key=lambda item: item["series_id"]),
        "issues": issues,
    }
    result["assignment_sha256"] = _assignment_sha256(assignments)
    result["crosswalk_sha256"] = _sha256_bytes(_canonical_json_bytes(result))
    return result


def verify_crosswalk(payload: Mapping[str, Any]) -> None:
    """Verify the self-hash and core safety claims of a crosswalk artifact."""

    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
        raise CrosswalkError("crosswalk schema is invalid")
    claimed = str(payload.get("crosswalk_sha256") or "").lower()
    if not _HEX64.fullmatch(claimed):
        raise CrosswalkError("crosswalk self-hash is invalid")
    body = dict(payload)
    body.pop("crosswalk_sha256", None)
    if _sha256_bytes(_canonical_json_bytes(body)) != claimed:
        raise CrosswalkError("crosswalk self-hash does not match payload")
    if (payload.get("authority") or {}).get("public") is not False:
        raise CrosswalkError("crosswalk grants public authority")
    assignments = payload.get("assignments")
    if not isinstance(assignments, list) or any(row.get("outcome_used") is not False for row in assignments if isinstance(row, Mapping)):
        raise CrosswalkError("crosswalk assignments are not outcome-free")
    claimed_assignment_hash = payload.get("assignment_sha256")
    if claimed_assignment_hash is not None:
        claimed_assignment_hash = str(claimed_assignment_hash).lower()
        if not _HEX64.fullmatch(claimed_assignment_hash):
            raise CrosswalkError("crosswalk assignment hash is invalid")
        if claimed_assignment_hash != _assignment_sha256(assignments):
            raise CrosswalkError("crosswalk assignment hash does not match payload")
    join_contract = payload.get("join_contract")
    tournament_binding = (
        join_contract.get("tournament_binding")
        if isinstance(join_contract, Mapping)
        else None
    )
    if tournament_binding is not None:
        if not isinstance(tournament_binding, Mapping) or dict(tournament_binding) != {
            "source": "ScoreboardGames.Tournament",
            "assignment_field": "scoreboard_tournament",
            "series_policy": "one_non_empty_value_per_series",
            "conflict_policy": "reject_series",
        }:
            raise CrosswalkError("crosswalk tournament binding is invalid")
        assignments_by_series: dict[str, set[str]] = {}
        for row in assignments:
            if not isinstance(row, Mapping):
                raise CrosswalkError("crosswalk assignment is invalid")
            series_id = str(row.get("series_id") or "").strip()
            tournament = str(row.get("scoreboard_tournament") or "").strip()
            if not series_id or not tournament:
                raise CrosswalkError("crosswalk assignment tournament is missing")
            assignments_by_series.setdefault(series_id, set()).add(_norm(tournament))
        if any(len(values) != 1 for values in assignments_by_series.values()):
            raise CrosswalkError("crosswalk series tournament is conflicting")
        series_rows = payload.get("series")
        if not isinstance(series_rows, list):
            raise CrosswalkError("crosswalk series summaries are missing")
        summary_by_series: dict[str, str] = {}
        for row in series_rows:
            if not isinstance(row, Mapping):
                raise CrosswalkError("crosswalk series summary is invalid")
            series_id = str(row.get("series_id") or "").strip()
            tournament = str(row.get("scoreboard_tournament") or "").strip()
            if not series_id or not tournament:
                raise CrosswalkError("crosswalk series tournament is missing")
            if series_id in summary_by_series:
                raise CrosswalkError("crosswalk series summary is duplicated")
            summary_by_series[series_id] = _norm(tournament)
        if summary_by_series != {
            series_id: next(iter(values))
            for series_id, values in assignments_by_series.items()
        }:
            raise CrosswalkError("crosswalk series tournament summary does not match assignments")
    coverage = payload.get("coverage")
    if not isinstance(coverage, Mapping) or coverage.get("complete") is not True:
        if payload.get("status") != "partial_authoritative_coverage":
            raise CrosswalkError("incomplete crosswalk does not declare partial coverage")


__all__ = [
    "CrosswalkError",
    "SCHEMA_VERSION",
    "build_oe_leaguepedia_series_crosswalk",
    "verify_crosswalk",
]
