"""Build a fail-closed Oracle's Elixir to Leaguepedia series crosswalk.

This module joins frozen OE game rows to already captured Leaguepedia
``ScoreboardGames``, ``MatchSchedule``, and ``Tournaments`` rows.  It does not
fetch data and it does not use match outcomes.  A row is assigned only when
the team set, competition mapping, patch evidence, bounded timestamp, and
independent tournament record identify one ScoreboardGames row.  The
ScoreboardGames game prefix must then identify one MatchSchedule row.

The result is research-only.  A partial result can describe authoritative
coverage for the mapped rows, but it never claims complete census coverage.
The caller must pass raw source bytes or regular files.  This lets the
builder verify the source hashes instead of trusting metadata supplied by a
caller.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from lol_kills.etl.aliases import normalize_team
from lol_kills.v2.tierlists.accepted_census import canonical_game_ids, identity_sha256


SCHEMA_VERSION = "scryglass:oe-leaguepedia-series-crosswalk:v2"
SOURCE_RECORD_LABELS = ("oe", "scoreboardgames", "matchschedule", "tournaments")
DEFAULT_MAX_GAME_TIME_DELTA_SECONDS = 300
DEFAULT_MAX_FIRST_GAME_SCHEDULE_DELTA_SECONDS = 6 * 60 * 60
DEFAULT_MAX_LATER_GAME_SCHEDULE_AGE_SECONDS = 6 * 60 * 60
_HEX64 = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_OUTCOME_FIELD_TOKENS = {
    "outcome",
    "result",
    "victor",
    "victory",
    "win",
    "winner",
    "winning",
    "won",
}
_OUTCOME_FREE_PROJECTION = {
    "scope": "all_embedded_raw_source_rows",
    "policy": "remove_top_level_outcome_result_winner_and_win_fields",
    "original_file_bytes_bound_separately": True,
}
_SCOREBOARD_IDENTITY = {
    "primary": "exact_OE.gameid_equals_ScoreboardGames.RiotPlatformGameId",
    "fallback": "verified_alias_team_set_competition_patch_and_bounded_timestamp",
    "direct_identity_uses_team_or_timestamp": False,
}
_TOURNAMENT_BINDING = {
    "source": "ScoreboardGames.Tournament",
    "assignment_field": "scoreboard_tournament",
    "source_record": "tournaments",
    "source_table": "Tournaments",
    "identity_fields": ["Name", "OverviewPage", "League"],
    "competition_policy": "overview_page_and_league_match_scoreboard_and_explicit_mapping",
    "series_policy": "one_non_empty_value_per_series",
    "conflict_policy": "reject_series",
}
_LEGACY_TOURNAMENT_BINDING = {
    "source": "ScoreboardGames.Tournament",
    "assignment_field": "scoreboard_tournament",
    "series_policy": "one_non_empty_value_per_series",
    "conflict_policy": "reject_series",
}


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


def _field_tokens(value: object) -> tuple[str, ...]:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(value))
    return tuple(
        token
        for token in re.sub(r"[^0-9A-Za-z]+", "_", text).casefold().split("_")
        if token
    )


def _is_outcome_field(value: object) -> bool:
    return bool(set(_field_tokens(value)) & _OUTCOME_FIELD_TOKENS)


def _outcome_free_rows(
    value: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {str(key): item for key, item in row.items() if not _is_outcome_field(key)}
        for row in value
    ]


def _read_safe_source_rows(
    path_text: object,
    *,
    label: str,
) -> tuple[bytes, list[dict[str, Any]]]:
    path = Path(str(path_text or ""))
    if not path.is_absolute():
        raise CrosswalkError(f"crosswalk source path is invalid: {label}")
    lexical = Path(os.path.abspath(path))
    if lexical != path:
        raise CrosswalkError(f"crosswalk source path is invalid: {label}")
    current = Path(lexical.anchor)
    try:
        for part in lexical.parts[1:]:
            current = current / part
            mode = os.lstat(current).st_mode
            if stat.S_ISLNK(mode):
                raise CrosswalkError(
                    f"crosswalk source path contains a symlink: {label}"
                )
        if not stat.S_ISREG(os.lstat(lexical).st_mode):
            raise CrosswalkError(f"crosswalk source path is not a file: {label}")
        raw = lexical.read_bytes()
    except CrosswalkError:
        raise
    except OSError as error:
        raise CrosswalkError(
            f"crosswalk source file cannot be read: {label}"
        ) from error
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CrosswalkError(
            f"crosswalk source file is not valid JSON: {label}"
        ) from error
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise CrosswalkError(
            f"crosswalk source file must contain a row array: {label}"
        )
    return raw, [dict(row) for row in value]


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


def _scoreboard_riot_platform_game_id(row: Mapping[str, Any]) -> str:
    value = _first(
        row,
        (
            "RiotPlatformGameId",
            "riot_platform_game_id",
            "riotplatformgameid",
        ),
    )
    if not value:
        return ""
    values = canonical_game_ids((value,))
    if len(values) != 1:
        raise CrosswalkError("ScoreboardGames RiotPlatformGameId is invalid")
    return values[0]


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
        return True, {
            "source_patch_available": False,
            "row_patch_available": False,
            "matched": True,
        }
    row_patch = _norm(_first(row, ("Patch", "patch", "patch_version")))
    if not row_patch:
        return True, {
            "source_patch_available": True,
            "row_patch_available": False,
            "matched": True,
        }
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
        "matched": matched,
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
    source_payload_hash, source_payload_bytes = _payload_hash(payload)
    claimed_payload_hash = str(record.get("payload_sha256") or "").lower()
    claimed_payload_bytes = record.get("payload_bytes")
    if (
        claimed_payload_hash != source_payload_hash
        or claimed_payload_bytes != source_payload_bytes
    ):
        raise CrosswalkError(f"source payload hash does not match: {label}")
    projected_payload_hash, projected_payload_bytes = _payload_hash(
        _outcome_free_rows(payload)
    )
    normalized = {
        "locator": locator,
        "retrieved_at": _parse_timestamp(retrieved_at, field="retrieved_at").isoformat().replace("+00:00", "Z"),
        "sha256": actual_hash,
        "bytes": len(source_bytes),
        "source_payload_sha256": source_payload_hash,
        "source_payload_bytes": source_payload_bytes,
        "payload_sha256": projected_payload_hash,
        "payload_bytes": projected_payload_bytes,
        "payload_projection": dict(_OUTCOME_FREE_PROJECTION),
        "integrity_verified": True,
    }
    path_text = _first(record, ("path",))
    if path_text:
        normalized["path"] = str(Path(path_text).resolve())
    return normalized


def _validate_source_records(
    source_records: Mapping[str, Mapping[str, Any]],
    payloads: Mapping[str, Sequence[Mapping[str, Any]]],
    raw_source_bytes: Mapping[str, bytes] | None,
) -> dict[str, Any]:
    if set(source_records) != set(SOURCE_RECORD_LABELS):
        raise CrosswalkError(
            "source records must cover OE, ScoreboardGames, MatchSchedule, and Tournaments"
        )
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
            riot_platform_game_id = _scoreboard_riot_platform_game_id(row)
            seen.add(game_id)
            prepared.append({**row, "_game_id": game_id, "_riot_platform_game_id": riot_platform_game_id, "_prefix": prefix, "_order": order, "_teams": teams, "_stamp": stamp, "_tournament": tournament})
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


def _prepared_tournament_rows(
    rows: Sequence[Mapping[str, Any]], issues: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    prepared: dict[str, list[dict[str, Any]]] = {}
    for index, raw in enumerate(rows):
        row = dict(raw)
        name = _first(row, ("Name", "name"))
        overview_page = _first(row, ("OverviewPage", "overview_page"))
        league = _first(row, ("League", "league"))
        if not name or not overview_page or not league:
            issues.append(
                {
                    "kind": "invalid_tournament_row",
                    "index": index,
                    "reason": "Name, OverviewPage, and League are required",
                }
            )
            continue
        prepared.setdefault(_norm(name), []).append(
            {
                **row,
                "_name": name,
                "_overview_page": overview_page,
                "_league": league,
            }
        )
    return prepared


def _tournament_matches_competition(
    tournament: Mapping[str, Any],
    scoreboard: Mapping[str, Any],
    scoreboard_section: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    tournament_name = str(tournament.get("_name") or "").strip()
    tournament_overview = _norm(tournament.get("_overview_page"))
    scoreboard_overview = _norm(
        _first(scoreboard, ("OverviewPage", "overview_page"))
    )
    tournament_league = _norm(tournament.get("_league"))
    scoreboard_league = _norm(_first(scoreboard, ("League", "league")))
    if not tournament_name or not tournament_overview or not tournament_league:
        return False, {"reason": "tournament_identity_fields_missing"}
    if not scoreboard_overview or tournament_overview != scoreboard_overview:
        return False, {
            "reason": "tournament_overview_page_mismatch",
            "tournament_overview_page": tournament_overview,
            "scoreboard_overview_page": scoreboard_overview,
        }
    if scoreboard_league and tournament_league != scoreboard_league:
        return False, {
            "reason": "tournament_league_mismatch",
            "tournament_league": tournament_league,
            "scoreboard_league": scoreboard_league,
        }
    allowed_leagues = _mapping_values(scoreboard_section, "league", "leagues")
    allowed_overviews = _mapping_values(
        scoreboard_section, "overview_page", "overview_pages", "pages"
    )
    allowed_tournaments = _mapping_values(
        scoreboard_section, "tournament", "tournaments"
    )
    if allowed_leagues and tournament_league not in allowed_leagues:
        return False, {
            "reason": "tournament_league_not_in_explicit_mapping",
            "tournament_league": tournament_league,
        }
    if allowed_overviews and tournament_overview not in allowed_overviews:
        return False, {
            "reason": "tournament_overview_page_not_in_explicit_mapping",
            "tournament_overview_page": tournament_overview,
        }
    if allowed_tournaments and _norm(tournament_name) not in allowed_tournaments:
        return False, {
            "reason": "tournament_name_not_in_explicit_mapping",
            "tournament_name": _norm(tournament_name),
        }
    return True, {
        "name": tournament_name,
        "overview_page": tournament_overview,
        "league": tournament_league,
    }


def build_oe_leaguepedia_series_crosswalk(
    oe_rows: Sequence[Mapping[str, Any]],
    scoreboard_rows: Sequence[Mapping[str, Any]],
    schedule_rows: Sequence[Mapping[str, Any]],
    tournaments_rows: Sequence[Mapping[str, Any]] | None = None,
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
        ("tournaments_rows", tournaments_rows),
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
        {
            "oe": oe_rows,
            "scoreboardgames": scoreboard_rows,
            "matchschedule": schedule_rows,
            "tournaments": tournaments_rows or (),
        },
        raw_source_bytes,
    )
    scoreboard = _prepared_scoreboard_rows(scoreboard_rows, issues)
    scoreboard_by_riot_platform_game_id: dict[str, list[dict[str, Any]]] = {}
    for row in scoreboard:
        riot_platform_game_id = str(row.get("_riot_platform_game_id") or "")
        if riot_platform_game_id:
            scoreboard_by_riot_platform_game_id.setdefault(
                riot_platform_game_id, []
            ).append(row)
    schedule = _prepared_schedule_rows(schedule_rows, issues)
    tournaments = _prepared_tournament_rows(tournaments_rows or (), issues)
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
        direct_candidates = scoreboard_by_riot_platform_game_id.get(
            oe["_game_id"], []
        )
        direct_disambiguation: dict[str, Any] | None = None
        if len(direct_candidates) > 1:
            schedule_unique_candidates = [
                candidate
                for candidate in direct_candidates
                if len(schedules_by_match.get(candidate["_prefix"], [])) == 1
            ]
            if len(schedule_unique_candidates) == 1:
                direct_candidates = schedule_unique_candidates
                direct_disambiguation = {
                    "policy": "select_only_one_exact_RiotPlatformGameId_candidate_with_one_MatchSchedule_row_for_its_GameId_prefix",
                    "candidate_count": len(
                        scoreboard_by_riot_platform_game_id[oe["_game_id"]]
                    ),
                    "eligible_candidate_count": 1,
                    "eligible_scoreboard_game_id": direct_candidates[0]["_game_id"],
                    "eligible_game_id_prefix": direct_candidates[0]["_prefix"],
                }
            else:
                issues.append(
                    {
                        "kind": "riot_platform_identity_ambiguous",
                        "oe_game_id": oe["_game_id"],
                        "candidate_count": len(direct_candidates),
                        "eligible_candidate_count": len(schedule_unique_candidates),
                        "candidate_game_ids": sorted(
                            candidate["_game_id"] for candidate in direct_candidates
                        ),
                    }
                )
                continue
        assignment_method: str
        if len(direct_candidates) == 1:
            selected = direct_candidates[0]
            competition_ok, competition_evidence = _competition_matches(
                selected, scoreboard_section, require_constraint=True
            )
            if not competition_ok:
                issues.append(
                    {
                        "kind": "riot_platform_identity_competition_mismatch",
                        "oe_game_id": oe["_game_id"],
                        "scoreboard_game_id": selected["_game_id"],
                    }
                )
                continue
            _patch_ok, patch_evidence = _patch_matches(
                oe["_patch"], selected, mapping
            )
            evidence = {
                "identity": {
                    "source_field": "OE.gameid",
                    "target_field": "ScoreboardGames.RiotPlatformGameId",
                    "value": oe["_game_id"],
                    "exact": True,
                },
                "competition": competition_evidence,
                "patch": {
                    **patch_evidence,
                    "identity_gate": "exact_riot_platform_game_id",
                    "identity_gate_enforced": False,
                },
            }
            if direct_disambiguation is not None:
                evidence["identity_disambiguation"] = direct_disambiguation
            assignment_method = "exact_riot_platform_game_id_then_exact_game_id_prefix"
            if direct_disambiguation is not None:
                assignment_method = (
                    "exact_riot_platform_game_id_disambiguated_by_unique_"
                    "matchschedule_prefix_then_exact_game_id_prefix"
                )
        else:
            candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for candidate in scoreboard:
                if candidate["_teams"] != oe["_teams"]:
                    continue
                delta = abs(
                    (candidate["_stamp"] - oe["_stamp"]).total_seconds()
                )
                if delta > max_game_time_delta_seconds:
                    continue
                competition_ok, competition_evidence = _competition_matches(
                    candidate, scoreboard_section, require_constraint=True
                )
                if not competition_ok:
                    continue
                patch_ok, patch_evidence = _patch_matches(
                    oe["_patch"], candidate, mapping
                )
                if not patch_ok:
                    continue
                candidates.append(
                    (
                        candidate,
                        {
                            "timestamp_delta_seconds": delta,
                            "competition": competition_evidence,
                            "patch": patch_evidence,
                        },
                    )
                )
            if len(candidates) != 1:
                issues.append(
                    {
                        "kind": (
                            "scoreboard_identity_ambiguous"
                            if len(candidates) > 1
                            else "scoreboard_identity_missing"
                        ),
                        "oe_game_id": oe["_game_id"],
                        "candidate_count": len(candidates),
                        "candidate_game_ids": [
                            candidate["_game_id"] for candidate, _ in candidates
                        ],
                    }
                )
                continue
            selected, evidence = candidates[0]
            assignment_method = (
                "exact_team_set_competition_patch_bounded_timestamp_then_"
                "exact_game_id_prefix"
            )
        scoreboard_id = selected["_game_id"]
        if scoreboard_id in used_scoreboard_ids:
            issues.append({"kind": "duplicate_source_assignment", "oe_game_id": oe["_game_id"], "scoreboard_game_id": scoreboard_id})
            continue
        scoreboard_tournament = str(selected.get("_tournament") or "").strip()
        if not scoreboard_tournament:
            issues.append(
                {
                    "kind": "scoreboard_tournament_missing",
                    "oe_game_id": oe["_game_id"],
                    "scoreboard_game_id": scoreboard_id,
                    "series_id": selected["_prefix"],
                }
            )
            continue
        tournament_candidates = tournaments.get(_norm(scoreboard_tournament), [])
        if not tournament_candidates:
            issues.append(
                {
                    "kind": "tournament_identity_missing",
                    "oe_game_id": oe["_game_id"],
                    "scoreboard_game_id": scoreboard_id,
                    "scoreboard_tournament": scoreboard_tournament,
                }
            )
            continue
        if len(tournament_candidates) != 1:
            issues.append(
                {
                    "kind": "tournament_identity_ambiguous",
                    "oe_game_id": oe["_game_id"],
                    "scoreboard_game_id": scoreboard_id,
                    "scoreboard_tournament": scoreboard_tournament,
                    "candidate_count": len(tournament_candidates),
                }
            )
            continue
        tournament_row = tournament_candidates[0]
        tournament_ok, tournament_evidence = _tournament_matches_competition(
            tournament_row, selected, scoreboard_section
        )
        if not tournament_ok:
            issues.append(
                {
                    "kind": "tournament_competition_mismatch",
                    "oe_game_id": oe["_game_id"],
                    "scoreboard_game_id": scoreboard_id,
                    "scoreboard_tournament": scoreboard_tournament,
                    "evidence": tournament_evidence,
                }
            )
            continue
        match_id = selected["_prefix"]
        match_candidates = schedules_by_match.get(match_id, [])
        schedule_candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
        direct_scoreboard_identity = assignment_method.startswith(
            "exact_riot_platform_game_id"
        )
        for schedule_row in match_candidates:
            if (
                not direct_scoreboard_identity
                and schedule_row["_teams"] != oe["_teams"]
            ):
                continue
            competition_ok, schedule_competition_evidence = _competition_matches(
                schedule_row, schedule_section, require_constraint=True
            )
            if not competition_ok:
                continue
            patch_ok, schedule_patch_evidence = _patch_matches(oe["_patch"], schedule_row, mapping)
            if not patch_ok and not direct_scoreboard_identity:
                continue
            schedule_delta = (oe["_stamp"] - schedule_row["_stamp"]).total_seconds()
            order = int(selected["_order"])
            schedule_bound = (
                max_first_game_schedule_delta_seconds
                if order == 1
                else max_later_game_schedule_age_seconds
            )
            if not direct_scoreboard_identity and abs(schedule_delta) > schedule_bound:
                continue
            schedule_candidates.append(
                (
                    schedule_row,
                    {
                        "identity": {
                            "source_field": "ScoreboardGames.GameId prefix",
                            "target_field": "MatchSchedule.MatchId",
                            "value": match_id,
                            "exact": True,
                        },
                        "series_timestamp_delta_seconds": schedule_delta,
                        "team_set_consistent": schedule_row["_teams"]
                        == oe["_teams"],
                        "timestamp_bound_used_for_identity": not direct_scoreboard_identity,
                        "competition": schedule_competition_evidence,
                        "patch": {
                            **schedule_patch_evidence,
                            "identity_gate": (
                                "exact_riot_platform_game_id"
                                if direct_scoreboard_identity
                                else "team_set_competition_patch_timestamp"
                            ),
                            "identity_gate_enforced": not direct_scoreboard_identity,
                        },
                    },
                )
            )
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
        used_scoreboard_ids.add(scoreboard_id)
        assignments.append({
            "oe_game_id": oe["_game_id"],
            "scoreboard_game_id": scoreboard_id,
            "scoreboard_riot_platform_game_id": (
                selected.get("_riot_platform_game_id") or None
            ),
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
            "evidence": {
                **evidence,
                "schedule": schedule_evidence,
                "tournament": tournament_evidence,
            },
            "outcome_used": False,
            "assignment_method": assignment_method,
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
            "oe": _outcome_free_rows(oe_rows),
            "scoreboardgames": _outcome_free_rows(scoreboard_rows),
            "matchschedule": _outcome_free_rows(schedule_rows),
            "tournaments": _outcome_free_rows(tournaments_rows or ()),
        },
        "join_contract": {
            "scoreboard_identity": dict(_SCOREBOARD_IDENTITY),
            "team_identity": "verified_alias_normalized_unordered_two_team_set_for_join; original_OE_team_set_for_assignment_binding",
            "competition_mapping": "explicit_source_league_to_scoreboard_and_schedule_values",
            "patch_identity": {
                "fallback": "required_when_both_source_and_target_patch_are_available",
                "exact_riot_platform_game_id": "diagnostic_only_when_both_source_and_target_patch_are_available",
            },
            "timestamp_bounds": {
                "game_seconds": max_game_time_delta_seconds,
                "first_game_schedule_absolute_seconds": max_first_game_schedule_delta_seconds,
                "later_game_schedule_absolute_seconds": max_later_game_schedule_age_seconds,
            },
            "timestamp_timezone": "UTC; naive Leaguepedia Cargo values are interpreted as UTC",
            "game_id_prefix_to_match_id": "exact",
            "one_to_one": True,
            "ambiguity_policy": "reject",
            "duplicate_policy": {
                "default": "reject",
                "exact_riot_platform_game_id": "accept_only_one_candidate_with_exactly_one_MatchSchedule_row_for_its_GameId_prefix",
            },
            "unmatched_policy": "reject_assignment_and_record_issue",
            "outcome_used": False,
            "outcome_policy": "ignored_and_never_used_for_matching",
            "raw_source_projection": dict(_OUTCOME_FREE_PROJECTION),
            "tournament_binding": dict(_TOURNAMENT_BINDING),
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
    source_binding = payload.get("source_binding")
    if not isinstance(source_binding, Mapping):
        raise CrosswalkError("crosswalk source binding is missing")
    accepted_values = source_binding.get("accepted_game_ids")
    selected_values = source_binding.get("selected_game_ids")
    if not isinstance(accepted_values, list) or not isinstance(selected_values, list):
        raise CrosswalkError("crosswalk source census IDs are missing")
    try:
        accepted_ids = tuple(canonical_game_ids(accepted_values))
        selected_ids = tuple(canonical_game_ids(selected_values))
    except (TypeError, ValueError) as error:
        raise CrosswalkError("crosswalk source census IDs are invalid") from error
    if list(accepted_ids) != accepted_values or list(selected_ids) != selected_values:
        raise CrosswalkError("crosswalk source census IDs are not canonical")
    if (
        source_binding.get("accepted_game_count") != len(accepted_ids)
        or source_binding.get("accepted_game_identity_sha256")
        != identity_sha256(accepted_ids)
        or source_binding.get("selected_game_count") != len(selected_ids)
        or source_binding.get("selected_game_identity_sha256")
        != identity_sha256(selected_ids)
        or not set(selected_ids).issubset(set(accepted_ids))
    ):
        raise CrosswalkError("crosswalk source census binding is invalid")
    assignment_ids = [
        str(row.get("oe_game_id") or "")
        for row in assignments
        if isinstance(row, Mapping)
    ]
    if len(assignment_ids) != len(assignments) or len(set(assignment_ids)) != len(
        assignment_ids
    ):
        raise CrosswalkError("crosswalk assignment IDs are invalid")
    if not set(assignment_ids).issubset(set(selected_ids)):
        raise CrosswalkError("crosswalk assignments escape selected source census")
    join_contract = payload.get("join_contract")
    tournament_binding = (
        join_contract.get("tournament_binding")
        if isinstance(join_contract, Mapping)
        else None
    )
    if tournament_binding is not None:
        is_current_binding = (
            isinstance(tournament_binding, Mapping)
            and dict(tournament_binding) == _TOURNAMENT_BINDING
        )
        is_legacy_partial_binding = (
            payload.get("status") == "partial_authoritative_coverage"
            and isinstance(tournament_binding, Mapping)
            and dict(tournament_binding) == _LEGACY_TOURNAMENT_BINDING
        )
        if not is_current_binding and not is_legacy_partial_binding:
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
        if is_current_binding:
            source_records = payload.get("source_records")
            raw_sources = payload.get("raw_sources")
            if not isinstance(source_records, Mapping) or not isinstance(
                raw_sources, Mapping
            ):
                raise CrosswalkError("crosswalk source evidence is missing")
            if set(source_records) != set(SOURCE_RECORD_LABELS):
                raise CrosswalkError("crosswalk source record labels are invalid")
            for label in SOURCE_RECORD_LABELS:
                record = source_records.get(label)
                rows = raw_sources.get(label)
                if not isinstance(record, Mapping) or not isinstance(rows, list) or any(
                    not isinstance(row, Mapping) for row in rows
                ):
                    raise CrosswalkError(
                        f"crosswalk raw source evidence is invalid: {label}"
                    )
                if _outcome_free_rows(rows) != rows:
                    raise CrosswalkError(
                        f"crosswalk raw source projection contains outcome fields: {label}"
                    )
                payload_hash, payload_bytes = _payload_hash(rows)
                if (
                    record.get("integrity_verified") is not True
                    or record.get("payload_sha256") != payload_hash
                    or record.get("payload_bytes") != payload_bytes
                    or record.get("payload_projection")
                    != _OUTCOME_FREE_PROJECTION
                ):
                    raise CrosswalkError(
                        f"crosswalk raw source payload binding changed: {label}"
                    )
                if (
                    not _HEX64.fullmatch(str(record.get("sha256") or ""))
                    or isinstance(record.get("bytes"), bool)
                    or not isinstance(record.get("bytes"), int)
                    or record.get("bytes", 0) < 0
                    or not _HEX64.fullmatch(
                        str(record.get("source_payload_sha256") or "")
                    )
                    or isinstance(record.get("source_payload_bytes"), bool)
                    or not isinstance(record.get("source_payload_bytes"), int)
                    or record.get("source_payload_bytes", 0) < 0
                ):
                    raise CrosswalkError(
                        f"crosswalk original source binding is invalid: {label}"
                    )
                source_raw, source_rows = _read_safe_source_rows(
                    record.get("path"), label=label
                )
                source_payload_hash, source_payload_bytes = _payload_hash(source_rows)
                if (
                    len(source_raw) != record.get("bytes")
                    or _sha256_bytes(source_raw) != record.get("sha256")
                    or source_payload_hash != record.get("source_payload_sha256")
                    or source_payload_bytes != record.get("source_payload_bytes")
                ):
                    raise CrosswalkError(
                        f"crosswalk original source bytes changed: {label}"
                    )
                if _outcome_free_rows(source_rows) != rows:
                    raise CrosswalkError(
                        f"crosswalk raw source projection differs from source file: {label}"
                    )

            if (
                not isinstance(join_contract, Mapping)
                or join_contract.get("raw_source_projection")
                != _OUTCOME_FREE_PROJECTION
                or join_contract.get("scoreboard_identity")
                != _SCOREBOARD_IDENTITY
            ):
                raise CrosswalkError("crosswalk raw source projection is invalid")

            capture_binding = payload.get("capture_manifest_binding")
            if capture_binding is not None:
                assembled = (
                    capture_binding.get("assembled")
                    if isinstance(capture_binding, Mapping)
                    else None
                )
                if not isinstance(assembled, Mapping) or set(assembled) != {
                    "ScoreboardGames",
                    "MatchSchedule",
                    "Tournaments",
                }:
                    raise CrosswalkError(
                        "crosswalk capture manifest assembled binding is invalid"
                    )
                capture_labels = {
                    "scoreboardgames": "ScoreboardGames",
                    "matchschedule": "MatchSchedule",
                    "tournaments": "Tournaments",
                }
                for source_label, capture_label in capture_labels.items():
                    capture_record = assembled.get(capture_label)
                    source_record = source_records.get(source_label)
                    if not isinstance(capture_record, Mapping) or not isinstance(
                        source_record, Mapping
                    ):
                        raise CrosswalkError(
                            "crosswalk capture manifest source binding is missing"
                        )
                    capture_path = str(capture_record.get("path") or "")
                    source_path = str(source_record.get("path") or "")
                    if (
                        not Path(capture_path).is_absolute()
                        or capture_path != str(Path(capture_path).resolve())
                        or source_path != capture_path
                        or capture_record.get("bytes")
                        != source_record.get("bytes")
                        or capture_record.get("sha256")
                        != source_record.get("sha256")
                    ):
                        raise CrosswalkError(
                            f"crosswalk capture manifest differs from source record: {capture_label}"
                        )

            has_direct_assignments = any(
                str(assignment.get("assignment_method") or "").startswith(
                    "exact_riot_platform_game_id"
                )
                for assignment in assignments
                if isinstance(assignment, Mapping)
            )
            scoreboard_by_id: dict[str, list[dict[str, Any]]] = {}
            scoreboard_by_riot_platform_game_id: dict[
                str, list[dict[str, Any]]
            ] = {}
            for raw in raw_sources["scoreboardgames"]:
                row = dict(raw)
                try:
                    scoreboard_by_id.setdefault(_scoreboard_game_id(row), []).append(row)
                    if has_direct_assignments:
                        riot_platform_game_id = _scoreboard_riot_platform_game_id(
                            row
                        )
                        if riot_platform_game_id:
                            scoreboard_by_riot_platform_game_id.setdefault(
                                riot_platform_game_id, []
                            ).append(row)
                except CrosswalkError as error:
                    raise CrosswalkError(
                        "crosswalk raw ScoreboardGames identity is invalid"
                    ) from error
            oe_by_id: dict[str, list[dict[str, Any]]] = {}
            for raw in raw_sources["oe"]:
                row = dict(raw)
                try:
                    oe_by_id.setdefault(_oe_game_id(row), []).append(row)
                except CrosswalkError as error:
                    raise CrosswalkError(
                        "crosswalk raw OE identity is invalid"
                    ) from error
            schedule_by_id: dict[str, list[dict[str, Any]]] = {}
            for raw in raw_sources["matchschedule"]:
                row = dict(raw)
                try:
                    schedule_by_id.setdefault(_schedule_match_id(row), []).append(
                        row
                    )
                except CrosswalkError:
                    continue
            tournament_issues: list[dict[str, Any]] = []
            tournaments_by_name = _prepared_tournament_rows(
                raw_sources["tournaments"], tournament_issues
            )
            recorded_issues = payload.get("issues")
            if not isinstance(recorded_issues, list) or any(
                not isinstance(issue, Mapping) for issue in recorded_issues
            ):
                raise CrosswalkError("crosswalk issue records are invalid")
            recorded_tournament_issues = [
                dict(issue)
                for issue in recorded_issues
                if issue.get("kind") == "invalid_tournament_row"
            ]
            if recorded_tournament_issues != tournament_issues:
                raise CrosswalkError(
                    "crosswalk invalid tournament issues differ from raw source"
                )
            competition_mapping = payload.get("competition_mapping")
            if not isinstance(competition_mapping, Mapping):
                raise CrosswalkError("crosswalk competition mapping is missing")
            for assignment in assignments:
                assert isinstance(assignment, Mapping)
                scoreboard_game_id = str(
                    assignment.get("scoreboard_game_id") or ""
                ).strip()
                scoreboard_candidates = scoreboard_by_id.get(scoreboard_game_id, [])
                if len(scoreboard_candidates) != 1:
                    raise CrosswalkError(
                        "crosswalk assignment ScoreboardGames evidence is invalid"
                    )
                scoreboard_row = scoreboard_candidates[0]
                oe_game_id = str(assignment.get("oe_game_id") or "").strip()
                if len(oe_by_id.get(oe_game_id, [])) != 1:
                    raise CrosswalkError(
                        "crosswalk assignment OE evidence is invalid"
                    )
                try:
                    riot_platform_game_id = _scoreboard_riot_platform_game_id(
                        scoreboard_row
                    )
                    game_prefix, game_order = _game_prefix_and_order(
                        scoreboard_game_id
                    )
                except CrosswalkError as error:
                    raise CrosswalkError(
                        "crosswalk assignment direct identity is invalid"
                    ) from error
                if (
                    str(
                        assignment.get("scoreboard_riot_platform_game_id") or ""
                    ).strip()
                    != riot_platform_game_id
                    or assignment.get("scoreboard_game_id_prefix") != game_prefix
                    or assignment.get("series_id") != game_prefix
                    or assignment.get("scoreboard_game_order") != game_order
                ):
                    raise CrosswalkError(
                        "crosswalk assignment game identity changed"
                    )
                method = str(assignment.get("assignment_method") or "")
                allowed_methods = {
                    "exact_riot_platform_game_id_then_exact_game_id_prefix",
                    (
                        "exact_riot_platform_game_id_disambiguated_by_unique_"
                        "matchschedule_prefix_then_exact_game_id_prefix"
                    ),
                    (
                        "exact_team_set_competition_patch_bounded_timestamp_then_"
                        "exact_game_id_prefix"
                    ),
                }
                if method not in allowed_methods:
                    raise CrosswalkError(
                        "crosswalk assignment method is invalid"
                    )
                evidence = assignment.get("evidence")
                if not isinstance(evidence, Mapping):
                    raise CrosswalkError(
                        "crosswalk assignment evidence is invalid"
                    )
                if method.startswith("exact_riot_platform_game_id"):
                    expected_identity = {
                        "source_field": "OE.gameid",
                        "target_field": "ScoreboardGames.RiotPlatformGameId",
                        "value": oe_game_id,
                        "exact": True,
                    }
                    if (
                        riot_platform_game_id != oe_game_id
                        or evidence.get("identity") != expected_identity
                    ):
                        raise CrosswalkError(
                            "crosswalk direct Riot game identity changed"
                        )
                    direct_candidates = scoreboard_by_riot_platform_game_id.get(
                        oe_game_id, []
                    )
                    if len(direct_candidates) == 0:
                        raise CrosswalkError(
                            "crosswalk direct Riot game identity is missing"
                        )
                    if len(direct_candidates) > 1:
                        eligible_candidates = []
                        for candidate in direct_candidates:
                            try:
                                candidate_prefix, _candidate_order = (
                                    _game_prefix_and_order(_scoreboard_game_id(candidate))
                                )
                            except CrosswalkError as error:
                                raise CrosswalkError(
                                    "crosswalk direct Riot game candidate identity is invalid"
                                ) from error
                            if len(schedule_by_id.get(candidate_prefix, [])) == 1:
                                eligible_candidates.append(
                                    (candidate, candidate_prefix)
                                )
                        if (
                            len(eligible_candidates) != 1
                            or _scoreboard_game_id(eligible_candidates[0][0])
                            != scoreboard_game_id
                            or method
                            != (
                                "exact_riot_platform_game_id_disambiguated_by_unique_"
                                "matchschedule_prefix_then_exact_game_id_prefix"
                            )
                        ):
                            raise CrosswalkError(
                                "crosswalk direct Riot game identity ambiguity changed"
                            )
                        expected_disambiguation = {
                            "policy": "select_only_one_exact_RiotPlatformGameId_candidate_with_one_MatchSchedule_row_for_its_GameId_prefix",
                            "candidate_count": len(direct_candidates),
                            "eligible_candidate_count": 1,
                            "eligible_scoreboard_game_id": scoreboard_game_id,
                            "eligible_game_id_prefix": eligible_candidates[0][1],
                        }
                        if evidence.get("identity_disambiguation") != expected_disambiguation:
                            raise CrosswalkError(
                                "crosswalk direct Riot game disambiguation evidence changed"
                            )
                    elif "identity_disambiguation" in evidence:
                        raise CrosswalkError(
                            "crosswalk direct Riot game disambiguation is unexpected"
                        )
                schedule_candidates = schedule_by_id.get(game_prefix, [])
                if len(schedule_candidates) != 1:
                    raise CrosswalkError(
                        "crosswalk assignment MatchSchedule evidence is invalid"
                    )
                schedule_evidence = evidence.get("schedule")
                expected_schedule_identity = {
                    "source_field": "ScoreboardGames.GameId prefix",
                    "target_field": "MatchSchedule.MatchId",
                    "value": game_prefix,
                    "exact": True,
                }
                if (
                    not isinstance(schedule_evidence, Mapping)
                    or schedule_evidence.get("identity")
                    != expected_schedule_identity
                ):
                    raise CrosswalkError(
                        "crosswalk schedule identity evidence changed"
                    )
                scoreboard_tournament = _first(
                    scoreboard_row, ("Tournament", "tournament")
                )
                if _norm(scoreboard_tournament) != _norm(
                    assignment.get("scoreboard_tournament")
                ):
                    raise CrosswalkError(
                        "crosswalk assignment tournament differs from ScoreboardGames"
                    )
                tournament_candidates = tournaments_by_name.get(
                    _norm(scoreboard_tournament), []
                )
                if len(tournament_candidates) != 1:
                    raise CrosswalkError(
                        "crosswalk assignment tournament source evidence is invalid"
                    )
                source_league = _norm(assignment.get("source_league"))
                mapping = next(
                    (
                        value
                        for key, value in competition_mapping.items()
                        if _norm(key) == source_league
                    ),
                    None,
                )
                if not isinstance(mapping, Mapping):
                    raise CrosswalkError(
                        "crosswalk assignment competition mapping is invalid"
                    )
                mapping = _resolve_tournament_mapping(
                    mapping, assignment.get("source_tournament")
                )
                if not isinstance(mapping, Mapping):
                    raise CrosswalkError(
                        "crosswalk assignment tournament mapping is invalid"
                    )
                oe_row = oe_by_id[oe_game_id][0]
                _score_patch_ok, expected_score_patch = _patch_matches(
                    _first(oe_row, ("patch", "Patch", "patch_version")),
                    scoreboard_row,
                    mapping,
                )
                observed_score_patch = evidence.get("patch")
                if method.startswith("exact_riot_platform_game_id"):
                    expected_score_patch = {
                        **expected_score_patch,
                        "identity_gate": "exact_riot_platform_game_id",
                        "identity_gate_enforced": False,
                    }
                if observed_score_patch != expected_score_patch:
                    raise CrosswalkError(
                        "crosswalk scoreboard patch evidence changed"
                    )
                schedule_row = schedule_candidates[0]
                _schedule_patch_ok, expected_schedule_patch = _patch_matches(
                    _first(oe_row, ("patch", "Patch", "patch_version")),
                    schedule_row,
                    mapping,
                )
                observed_schedule_patch = (
                    schedule_evidence.get("patch")
                    if isinstance(schedule_evidence, Mapping)
                    else None
                )
                if method.startswith("exact_riot_platform_game_id"):
                    expected_schedule_patch = {
                        **expected_schedule_patch,
                        "identity_gate": "exact_riot_platform_game_id",
                        "identity_gate_enforced": False,
                    }
                else:
                    expected_schedule_patch = {
                        **expected_schedule_patch,
                        "identity_gate": "team_set_competition_patch_timestamp",
                        "identity_gate_enforced": True,
                    }
                if observed_schedule_patch != expected_schedule_patch:
                    raise CrosswalkError(
                        "crosswalk schedule patch evidence changed"
                    )
                tournament_ok, expected_evidence = _tournament_matches_competition(
                    tournament_candidates[0],
                    scoreboard_row,
                    _mapping_section(mapping, "scoreboard"),
                )
                observed_evidence = (
                    evidence.get("tournament")
                    if isinstance(evidence, Mapping)
                    else None
                )
                if not tournament_ok or observed_evidence != expected_evidence:
                    raise CrosswalkError(
                        "crosswalk assignment tournament evidence changed"
                    )
    coverage = payload.get("coverage")
    if not isinstance(coverage, Mapping):
        raise CrosswalkError("crosswalk coverage is missing")
    if (
        coverage.get("selected_game_count") != len(selected_ids)
        or coverage.get("accepted_game_count") != len(accepted_ids)
        or coverage.get("mapped_game_count") != len(assignment_ids)
        or coverage.get("unmatched_game_count")
        != len(selected_ids) - len(assignment_ids)
        or coverage.get("selected_is_full_accepted_census")
        is not (selected_ids == accepted_ids)
        or coverage.get("mapped_is_full_accepted_census")
        is not (set(assignment_ids) == set(accepted_ids))
    ):
        raise CrosswalkError("crosswalk coverage does not match source evidence")
    if payload.get("status") == "complete_authoritative_coverage":
        if not isinstance(tournament_binding, Mapping):
            raise CrosswalkError("complete crosswalk tournament binding is missing")
        if dict(tournament_binding) != _TOURNAMENT_BINDING:
            raise CrosswalkError("complete crosswalk tournament binding is invalid")
        source_records = payload.get("source_records")
        tournament_record = (
            source_records.get("tournaments")
            if isinstance(source_records, Mapping)
            else None
        )
        if not isinstance(tournament_record, Mapping) or tournament_record.get(
            "integrity_verified"
        ) is not True:
            raise CrosswalkError("complete crosswalk tournament source record is missing")
        raw_sources = payload.get("raw_sources")
        if not isinstance(raw_sources, Mapping) or not isinstance(
            raw_sources.get("tournaments"), list
        ):
            raise CrosswalkError("complete crosswalk tournament source rows are missing")
    if coverage.get("complete") is not True:
        if payload.get("status") != "partial_authoritative_coverage":
            raise CrosswalkError("incomplete crosswalk does not declare partial coverage")


__all__ = [
    "CrosswalkError",
    "SCHEMA_VERSION",
    "build_oe_leaguepedia_series_crosswalk",
    "verify_crosswalk",
]
