"""Derive conservative OE to Leaguepedia team aliases from captured rows.

The derivation uses only unique timestamp matches.  It does not use winners,
team side, row order, tournament outcomes, or a pre-existing alias table.
When one team name is shared by both rows, the other pair supplies one alias
candidate.  A pair with no shared name is ambiguous and stays in review.

Every accepted mapping needs repeated evidence from distinct game IDs.  A
singleton is never promoted to the accepted mapping.  Input records contain
raw-byte and canonical-payload hashes, and the output binds both exact input
hashes in a self-hashed research artifact.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "scryglass:oe-leaguepedia-team-alias-derivation:v1"
DEFAULT_MAX_TIMESTAMP_DELTA_SECONDS = 300
DEFAULT_MIN_REPEATED_EVIDENCE = 2
_HEX64 = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


class AliasDerivationError(ValueError):
    """Raised when alias evidence or source identity is unsafe."""


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
        raise AliasDerivationError("alias payload is not canonical JSON") from error


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _parse_time(value: Any, *, field: str) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        raise AliasDerivationError(f"{field} timestamp is missing")
    parsed: datetime | None = None
    candidate = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                parsed = datetime.strptime(raw[:19], fmt).replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue
    if parsed is None:
        raise AliasDerivationError(f"{field} timestamp is malformed")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _name_key(value: Any) -> str:
    """Return a conservative name key for anchoring an observed pair."""

    text = unicodedata.normalize("NFKC", _text(value)).casefold()
    text = text.replace("’", "'").replace("`", "'")
    # Leaguepedia often appends an explicit disambiguator to a team page.
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text)
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _game_id(row: Mapping[str, Any], *, label: str) -> str:
    for field in ("gameid", "game_uid", "game_id", "GameId"):
        value = _text(row.get(field))
        if value:
            return value
    raise AliasDerivationError(f"{label} has no game identity")


def _match_id(row: Mapping[str, Any], *, label: str) -> str:
    for field in ("MatchId", "match_id", "series_id"):
        value = _text(row.get(field))
        if value:
            return value
    raise AliasDerivationError(f"{label} has no MatchId")


def _game_prefix(game_id: str, *, label: str) -> str:
    prefix, separator, ordinal = game_id.rpartition("_")
    if not separator or not prefix or not ordinal.isdigit() or int(ordinal) < 1:
        raise AliasDerivationError(f"{label} GameId has no positive ordinal")
    return prefix


def _timestamp(row: Mapping[str, Any], *, label: str) -> datetime:
    for field in (
        "DateTime UTC",
        "DateTime_UTC",
        "datetime_utc",
        "start_utc",
        "date",
        "timestamp",
    ):
        if _text(row.get(field)):
            return _parse_time(row[field], field=f"{label}.{field}")
    raise AliasDerivationError(f"{label} has no supported timestamp")


def _team_pair(row: Mapping[str, Any], *, label: str, scoreboard: bool) -> tuple[str, str]:
    if scoreboard:
        pairs = (("Team1", "Team2"), ("team1", "team2"), ("blue", "red"))
    else:
        pairs = (
            ("team1", "team2"),
            ("Team1", "Team2"),
            ("blue_team", "red_team"),
            ("blue", "red"),
        )
    values: tuple[str, str] | None = None
    for left, right in pairs:
        if left in row or right in row:
            values = (_text(row.get(left)), _text(row.get(right)))
            break
    if values is None:
        raw_values = row.get("teams") or row.get("team_set")
        if isinstance(raw_values, (list, tuple)) and len(raw_values) == 2:
            values = (_text(raw_values[0]), _text(raw_values[1]))
    if values is None or not values[0] or not values[1]:
        raise AliasDerivationError(f"{label} has no complete team pair")
    if _name_key(values[0]) == _name_key(values[1]):
        raise AliasDerivationError(f"{label} has duplicate team identities")
    return values


def _stable_team_keys(
    row: Mapping[str, Any],
    *,
    label: str,
    teams: tuple[str, str],
) -> tuple[str, str] | None:
    """Read exact stable OE IDs paired with the two display names."""

    if "team_keys" not in row or row.get("team_keys") is None:
        return None
    raw_values = row.get("team_keys")
    if (
        isinstance(raw_values, (str, bytes, bytearray))
        or not isinstance(raw_values, Sequence)
        or len(raw_values) != len(teams)
    ):
        raise AliasDerivationError(f"{label}.team_keys must pair exactly with teams")
    values = tuple(_text(value) for value in raw_values)
    if any(not value for value in values) or len(set(values)) != len(values):
        raise AliasDerivationError(f"{label}.team_keys must contain two distinct non-empty IDs")
    return values[0], values[1]


def _stable_team_key_binding(
    oe_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Build a digest over exact per-game OE team IDs when supplied."""

    has_team_keys = ["team_keys" in row and row.get("team_keys") is not None for row in oe_rows]
    if not any(has_team_keys):
        return None
    if not all(has_team_keys):
        raise AliasDerivationError("OE team_keys must be present for every row when supplied")
    rows: list[dict[str, Any]] = []
    seen_game_ids: set[str] = set()
    for index, raw_row in enumerate(oe_rows):
        row = dict(raw_row)
        game_id = _game_id(row, label=f"oe[{index}]")
        if game_id in seen_game_ids:
            raise AliasDerivationError(f"duplicate OE game ID in team-key binding: {game_id}")
        teams = _team_pair(row, label=f"oe[{index}]", scoreboard=False)
        team_keys = _stable_team_keys(row, label=f"oe[{index}]", teams=teams)
        if team_keys is None:
            raise AliasDerivationError(f"oe[{index}].team_keys is missing")
        seen_game_ids.add(game_id)
        rows.append({"game_id": game_id, "teams": list(teams), "team_keys": list(team_keys)})
    rows.sort(key=lambda item: item["game_id"])
    canonical = _canonical_json_bytes(rows)
    return {
        "field": "team_keys",
        "rows": rows,
        "row_count": len(rows),
        "rows_sha256": _sha256(canonical),
        "stable_oe_identity_sha256": _sha256(canonical),
    }


def _verify_stable_team_key_binding(binding: Mapping[str, Any]) -> str:
    if _text(binding.get("field")) != "team_keys":
        raise AliasDerivationError("stable OE team-key field is invalid")
    rows = binding.get("rows")
    if not isinstance(rows, list) or not rows:
        raise AliasDerivationError("stable OE team-key rows are missing")
    if binding.get("row_count") != len(rows):
        raise AliasDerivationError("stable OE team-key row count changed")
    previous_game_id = ""
    seen_game_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise AliasDerivationError("stable OE team-key row is invalid")
        game_id = _text(row.get("game_id"))
        teams = row.get("teams")
        team_keys = row.get("team_keys")
        if (
            not game_id
            or game_id in seen_game_ids
            or game_id < previous_game_id
            or not isinstance(teams, list)
            or len(teams) != 2
            or not all(isinstance(value, str) and value.strip() for value in teams)
            or not isinstance(team_keys, list)
            or len(team_keys) != 2
            or not all(isinstance(value, str) and value.strip() for value in team_keys)
            or len(set(map(str, team_keys))) != 2
        ):
            raise AliasDerivationError("stable OE team-key row is not an exact pair")
        seen_game_ids.add(game_id)
        previous_game_id = game_id
    digest = _sha256(_canonical_json_bytes(rows))
    if _text(binding.get("rows_sha256")).lower() != digest:
        raise AliasDerivationError("stable OE team-key digest changed")
    if _text(binding.get("stable_oe_identity_sha256")).lower() != digest:
        raise AliasDerivationError("stable OE identity digest changed")
    return digest


def _infer_pairs(
    source_names: tuple[str, str],
    target_names: tuple[str, str],
    *,
    require_exactly_one_anchor: bool,
) -> tuple[list[tuple[str, str]] | None, str | None]:
    source_by_key = {_name_key(name): name for name in source_names}
    target_by_key = {_name_key(name): name for name in target_names}
    shared_keys = set(source_by_key).intersection(target_by_key)
    if require_exactly_one_anchor and len(shared_keys) != 1:
        return None, "expected_exactly_one_shared_normalized_team_anchor"
    if len(shared_keys) == 1:
        anchor = next(iter(shared_keys))
        source_other = next(key for key in source_by_key if key != anchor)
        target_other = next(key for key in target_by_key if key != anchor)
        return [
            (source_by_key[anchor], target_by_key[anchor]),
            (source_by_key[source_other], target_by_key[target_other]),
        ], None
    if len(shared_keys) == 2 and not require_exactly_one_anchor:
        return [(source_by_key[key], target_by_key[key]) for key in sorted(shared_keys)], None
    return None, "no_shared_normalized_team_anchor; row_order_not_used"


def _source_payload_hash(rows: Sequence[Mapping[str, Any]]) -> tuple[str, int]:
    raw = _canonical_json_bytes([dict(row) for row in rows])
    return _sha256(raw), len(raw)


def _verify_input_record(
    label: str,
    rows: Sequence[Mapping[str, Any]],
    record: Mapping[str, Any],
    raw_source_bytes: Mapping[str, bytes] | None,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise AliasDerivationError(f"{label} input record is invalid")
    locator = _text(record.get("url") or record.get("locator") or record.get("path"))
    retrieved_at = _text(record.get("retrieved_at") or record.get("retrieval_time") or record.get("captured_at"))
    if not locator or not retrieved_at:
        raise AliasDerivationError(f"{label} input record needs locator and retrieval time")
    _parse_time(retrieved_at, field=f"{label}.retrieved_at")
    raw: bytes | None = None
    if raw_source_bytes is not None and label in raw_source_bytes:
        raw = bytes(raw_source_bytes[label])
    elif _text(record.get("path")):
        path = Path(_text(record["path"]))
        if path.is_symlink() or not path.is_file():
            raise AliasDerivationError(f"{label} input path is missing or unsafe")
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise AliasDerivationError(f"{label} input path cannot be read") from error
    if raw is None:
        raise AliasDerivationError(f"{label} raw bytes or a regular source path are required")
    claimed_sha = _text(record.get("sha256") or record.get("raw_sha256")).lower()
    claimed_bytes = record.get("bytes", record.get("raw_bytes"))
    if not _HEX64.fullmatch(claimed_sha) or claimed_sha != _sha256(raw):
        raise AliasDerivationError(f"{label} raw source hash does not match")
    if isinstance(claimed_bytes, bool) or not isinstance(claimed_bytes, int) or claimed_bytes != len(raw):
        raise AliasDerivationError(f"{label} raw source byte count does not match")
    payload_sha, payload_bytes = _source_payload_hash(rows)
    if _text(record.get("payload_sha256")).lower() != payload_sha or record.get("payload_bytes") != payload_bytes:
        raise AliasDerivationError(f"{label} source payload hash does not match")
    return {
        "locator": locator,
        "retrieved_at": _parse_time(retrieved_at, field=f"{label}.retrieved_at").isoformat().replace("+00:00", "Z"),
        "sha256": claimed_sha,
        "bytes": len(raw),
        "payload_sha256": payload_sha,
        "payload_bytes": payload_bytes,
        "integrity_verified": True,
    }


def _evidence_summary(
    source_key: str,
    target_key: str,
    source_name: str,
    target_name: str,
    evidence: Sequence[Mapping[str, Any]],
    *,
    minimum: int,
    status: str,
    reason: str | None = None,
) -> dict[str, Any]:
    game_ids = sorted({str(item["oe_game_id"]) for item in evidence})
    scoreboard_ids = sorted({str(item["scoreboard_game_id"]) for item in evidence})
    match_ids = sorted(
        {
            str(item["matchschedule_match_id"])
            for item in evidence
            if item.get("matchschedule_match_id") is not None
        }
    )
    deltas = [
        float(item["timestamp_delta_seconds"])
        for item in evidence
        if item.get("timestamp_delta_seconds") is not None
    ]
    source_names = [str(item["oe_team"]) for item in evidence]
    target_names = [str(item["leaguepedia_team"]) for item in evidence]
    stable_team_keys = sorted(
        {
            str(item["oe_stable_team_key"])
            for item in evidence
            if item.get("oe_stable_team_key") is not None
        }
    )

    def canonical_display(values: Sequence[str]) -> str:
        counts = Counter(values)
        return sorted(counts, key=lambda value: (-counts[value], _name_key(value), value))[0]

    allowed_source_names = sorted(set(source_names), key=lambda value: (_name_key(value), value))
    allowed_target_names = sorted(set(target_names), key=lambda value: (_name_key(value), value))
    result: dict[str, Any] = {
        "oe_team": source_name,
        "oe_team_key": source_key,
        "leaguepedia_team": target_name,
        "leaguepedia_team_key": target_key,
        "canonical_source_name": canonical_display(source_names),
        "allowed_source_names": allowed_source_names,
        "canonical_leaguepedia_name": canonical_display(target_names),
        "allowed_leaguepedia_names": allowed_target_names,
        "equivalence_group": f"team-alias:{source_key}:{target_key}",
        "evidence_count": len(evidence),
        "distinct_oe_game_count": len(game_ids),
        "distinct_scoreboard_game_count": len(scoreboard_ids),
        "oe_game_ids": game_ids,
        "scoreboard_game_ids": scoreboard_ids,
        "max_timestamp_delta_seconds": max(deltas) if deltas else None,
        "min_timestamp_delta_seconds": min(deltas) if deltas else None,
        "minimum_repeated_evidence": minimum,
        "status": status,
    }
    if match_ids:
        result["matchschedule_match_ids"] = match_ids
    if stable_team_keys:
        result["stable_oe_team_keys"] = stable_team_keys
    if reason:
        result["reason"] = reason
    return result


def _finalize_evidence(
    evidence_by_pair: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    *,
    minimum: int,
    conflict_prefix: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source_to_targets: dict[str, set[str]] = {}
    target_to_sources: dict[str, set[str]] = {}
    pair_names: dict[tuple[str, str], tuple[str, str]] = {}
    for (source_key, target_key), evidence in evidence_by_pair.items():
        source_to_targets.setdefault(source_key, set()).add(target_key)
        target_to_sources.setdefault(target_key, set()).add(source_key)
        pair_names[(source_key, target_key)] = (
            str(evidence[0]["oe_team"]),
            str(evidence[0]["leaguepedia_team"]),
        )
    conflicts: list[dict[str, Any]] = []
    prefix = f"{conflict_prefix}_" if conflict_prefix else ""
    for source_key, targets in sorted(source_to_targets.items()):
        if len(targets) > 1:
            conflicts.append({
                "kind": f"{prefix}source_alias_conflict",
                "source_team_key": source_key,
                "target_team_keys": sorted(targets),
            })
    for target_key, sources in sorted(target_to_sources.items()):
        if len(sources) > 1:
            conflicts.append({
                "kind": f"{prefix}target_alias_conflict",
                "target_team_key": target_key,
                "source_team_keys": sorted(sources),
            })
    accepted: list[dict[str, Any]] = []
    review_only: list[dict[str, Any]] = []
    for pair, evidence in sorted(evidence_by_pair.items()):
        source_key, target_key = pair
        source_name, target_name = pair_names[pair]
        is_conflict = len(source_to_targets[source_key]) > 1 or len(target_to_sources[target_key]) > 1
        evidence_count = len({str(item["oe_game_id"]) for item in evidence})
        if is_conflict:
            review_only.append(_evidence_summary(source_key, target_key, source_name, target_name, evidence, minimum=minimum, status="blocked", reason="one_to_one_conflict"))
        elif evidence_count < minimum:
            review_only.append(_evidence_summary(source_key, target_key, source_name, target_name, evidence, minimum=minimum, status="review_only", reason="singleton_or_insufficient_repeated_evidence"))
        else:
            accepted.append(_evidence_summary(source_key, target_key, source_name, target_name, evidence, minimum=minimum, status="accepted"))
    return accepted, review_only, conflicts


def _canonical_alias_records(
    entries: Sequence[Mapping[str, Any]],
    *,
    target_system: str,
) -> list[dict[str, Any]]:
    return [
        {
            "target_system": target_system,
            "canonical_source_name": item["canonical_source_name"],
            "source_name_key": item["oe_team_key"],
            "allowed_source_names": item["allowed_source_names"],
            "canonical_target_name": item["canonical_leaguepedia_name"],
            "target_name_key": item["leaguepedia_team_key"],
            "allowed_target_names": item["allowed_leaguepedia_names"],
            "equivalence_group": f"{target_system}:{item['equivalence_group']}",
        }
        for item in entries
    ]


def derive_team_alias_mapping(
    oe_rows: Sequence[Mapping[str, Any]],
    scoreboard_rows: Sequence[Mapping[str, Any]],
    *,
    oe_source_record: Mapping[str, Any],
    scoreboard_source_record: Mapping[str, Any],
    schedule_rows: Sequence[Mapping[str, Any]] | None = None,
    schedule_source_record: Mapping[str, Any] | None = None,
    raw_source_bytes: Mapping[str, bytes] | None = None,
    max_timestamp_delta_seconds: int = DEFAULT_MAX_TIMESTAMP_DELTA_SECONDS,
    minimum_repeated_evidence: int = DEFAULT_MIN_REPEATED_EVIDENCE,
    captured_at: str | None = None,
) -> dict[str, Any]:
    """Derive repeated, one-to-one team aliases from unique timestamp joins."""

    if max_timestamp_delta_seconds < 0:
        raise AliasDerivationError("timestamp bound must be non-negative")
    if minimum_repeated_evidence < 2:
        raise AliasDerivationError("minimum repeated evidence must be at least two games")
    if (schedule_rows is None) != (schedule_source_record is None):
        raise AliasDerivationError("schedule rows and schedule source record must be supplied together")
    source_row_sets: list[tuple[str, Sequence[Mapping[str, Any]]]] = [
        ("oe", oe_rows),
        ("scoreboardgames", scoreboard_rows),
    ]
    if schedule_rows is not None:
        source_row_sets.append(("matchschedule", schedule_rows))
    for label, rows in source_row_sets:
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
            raise AliasDerivationError(f"{label} rows must be an array")
        if any(not isinstance(row, Mapping) for row in rows):
            raise AliasDerivationError(f"{label} rows contain a non-object")

    stable_team_key_binding = _stable_team_key_binding(oe_rows)
    input_records = {
        "oe": _verify_input_record("oe", oe_rows, oe_source_record, raw_source_bytes),
        "scoreboardgames": _verify_input_record(
            "scoreboardgames", scoreboard_rows, scoreboard_source_record, raw_source_bytes
        ),
    }
    if schedule_rows is not None and schedule_source_record is not None:
        input_records["matchschedule"] = _verify_input_record(
            "matchschedule", schedule_rows, schedule_source_record, raw_source_bytes
        )
    if stable_team_key_binding is not None:
        input_records["oe"]["stable_team_key_binding"] = stable_team_key_binding
    observed_at = _parse_time(captured_at, field="captured_at").isoformat().replace("+00:00", "Z") if captured_at else datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    issues: list[dict[str, Any]] = []
    oe_prepared: list[dict[str, Any]] = []
    seen_oe_ids: set[str] = set()
    stable_keys_by_game_id = {
        str(item["game_id"]): tuple(str(value) for value in item["team_keys"])
        for item in (stable_team_key_binding["rows"] if stable_team_key_binding else ())
    }
    for index, raw_row in enumerate(oe_rows):
        row = dict(raw_row)
        try:
            game_id = _game_id(row, label=f"oe[{index}]")
            if game_id in seen_oe_ids:
                raise AliasDerivationError(f"duplicate OE game ID: {game_id}")
            stamp = _timestamp(row, label=f"oe[{index}]")
            teams = _team_pair(row, label=f"oe[{index}]", scoreboard=False)
            stable_keys = stable_keys_by_game_id.get(game_id)
            seen_oe_ids.add(game_id)
            oe_prepared.append({
                "row": row,
                "index": index,
                "game_id": game_id,
                "stamp": stamp,
                "teams": teams,
                "stable_team_keys": stable_keys,
            })
        except AliasDerivationError as error:
            issues.append({"kind": "invalid_oe_row", "index": index, "error": str(error)})

    stable_display_to_keys: dict[str, set[str]] = {}
    if stable_team_key_binding is not None:
        for oe in oe_prepared:
            stable_keys = oe.get("stable_team_keys")
            if not isinstance(stable_keys, tuple) or len(stable_keys) != 2:
                issues.append({
                    "kind": "invalid_oe_team_key_row",
                    "oe_game_id": oe["game_id"],
                    "error": "stable team-key binding is incomplete",
                })
                continue
            for display_name, stable_key in zip(oe["teams"], stable_keys):
                stable_display_to_keys.setdefault(_name_key(display_name), set()).add(stable_key)
    stable_display_conflicts = {
        display_key: sorted(stable_keys)
        for display_key, stable_keys in stable_display_to_keys.items()
        if len(stable_keys) > 1
    }
    for display_key, stable_keys in sorted(stable_display_conflicts.items()):
        issues.append({
            "kind": "oe_display_name_stable_key_conflict",
            "oe_team_name_key": display_key,
            "stable_team_keys": stable_keys,
        })

    scoreboard_prepared: list[dict[str, Any]] = []
    seen_scoreboard_ids: set[str] = set()
    for index, raw_row in enumerate(scoreboard_rows):
        row = dict(raw_row)
        try:
            game_id = _game_id(row, label=f"scoreboardgames[{index}]")
            if game_id in seen_scoreboard_ids:
                raise AliasDerivationError(f"duplicate ScoreboardGames GameId: {game_id}")
            stamp = _timestamp(row, label=f"scoreboardgames[{index}]")
            teams = _team_pair(row, label=f"scoreboardgames[{index}]", scoreboard=True)
            seen_scoreboard_ids.add(game_id)
            scoreboard_prepared.append({"row": row, "index": index, "game_id": game_id, "stamp": stamp, "teams": teams})
        except AliasDerivationError as error:
            issues.append({"kind": "invalid_scoreboard_row", "index": index, "error": str(error)})

    schedule_prepared: list[dict[str, Any]] = []
    if schedule_rows is not None:
        seen_match_ids: set[str] = set()
        for index, raw_row in enumerate(schedule_rows):
            row = dict(raw_row)
            try:
                match_id = _match_id(row, label=f"matchschedule[{index}]")
                if match_id in seen_match_ids:
                    raise AliasDerivationError(f"duplicate MatchSchedule MatchId: {match_id}")
                teams = _team_pair(row, label=f"matchschedule[{index}]", scoreboard=True)
                seen_match_ids.add(match_id)
                schedule_prepared.append({"row": row, "index": index, "match_id": match_id, "teams": teams})
            except AliasDerivationError as error:
                issues.append({"kind": "invalid_matchschedule_row", "index": index, "error": str(error)})

    candidate_evidence: dict[tuple[str, str], list[dict[str, Any]]] = {}
    used_scoreboard_ids: set[str] = set()
    matched_rows = 0
    ambiguous_rows = 0
    unmatched_rows = 0
    pair_ambiguous_rows = 0

    for oe in oe_prepared:
        if stable_team_key_binding is not None:
            conflicting_names = [
                _name_key(name)
                for name in oe["teams"]
                if _name_key(name) in stable_display_conflicts
            ]
            if conflicting_names:
                issues.append({
                    "kind": "oe_display_name_stable_key_conflict_row",
                    "oe_game_id": oe["game_id"],
                    "oe_team_name_keys": sorted(set(conflicting_names)),
                    "stable_team_keys": sorted({
                        stable_key
                        for display_key in set(conflicting_names)
                        for stable_key in stable_display_conflicts[display_key]
                    }),
                })
                continue
        candidates = [
            scoreboard
            for scoreboard in scoreboard_prepared
            if abs((scoreboard["stamp"] - oe["stamp"]).total_seconds()) <= max_timestamp_delta_seconds
        ]
        if len(candidates) != 1:
            kind = "timestamp_ambiguous" if len(candidates) > 1 else "timestamp_unmatched"
            issues.append({
                "kind": kind,
                "oe_game_id": oe["game_id"],
                "candidate_count": len(candidates),
                "candidate_scoreboard_game_ids": [item["game_id"] for item in candidates],
            })
            ambiguous_rows += kind == "timestamp_ambiguous"
            unmatched_rows += kind == "timestamp_unmatched"
            continue
        scoreboard = candidates[0]
        if scoreboard["game_id"] in used_scoreboard_ids:
            issues.append({
                "kind": "scoreboard_assignment_reused",
                "oe_game_id": oe["game_id"],
                "scoreboard_game_id": scoreboard["game_id"],
            })
            continue
        source_names = oe["teams"]
        target_names = scoreboard["teams"]
        pairs, pair_reason = _infer_pairs(
            source_names,
            target_names,
            require_exactly_one_anchor=False,
        )
        if pairs is None:
            issues.append({
                "kind": "team_pair_ambiguous",
                "oe_game_id": oe["game_id"],
                "scoreboard_game_id": scoreboard["game_id"],
                "oe_teams": list(source_names),
                "scoreboard_teams": list(target_names),
                "reason": pair_reason or "team-pair inference was ambiguous",
            })
            pair_ambiguous_rows += 1
            continue
        used_scoreboard_ids.add(scoreboard["game_id"])
        matched_rows += 1
        delta = abs((scoreboard["stamp"] - oe["stamp"]).total_seconds())
        stable_by_name = {
            _name_key(name): stable_key
            for name, stable_key in zip(oe["teams"], oe.get("stable_team_keys") or ())
        }
        for source_name, target_name in pairs:
            stable_key = stable_by_name.get(_name_key(source_name))
            source_key = stable_key if stable_team_key_binding is not None else _name_key(source_name)
            target_key = _name_key(target_name)
            evidence = {
                "oe_team": source_name,
                "leaguepedia_team": target_name,
                "oe_game_id": oe["game_id"],
                "scoreboard_game_id": scoreboard["game_id"],
                "timestamp_delta_seconds": delta,
                "outcome_used": False,
            }
            if stable_key is not None:
                evidence["oe_stable_team_key"] = stable_key
            candidate_evidence.setdefault((source_key, target_key), []).append(evidence)

    scoreboard_schedule_evidence: dict[tuple[str, str], list[dict[str, Any]]] = {}
    schedule_pairs_by_scoreboard_game: dict[str, dict[str, tuple[str, str, str]]] = {}
    schedule_ambiguous_rows = 0
    schedule_unmatched_rows = 0
    schedule_pair_ambiguous_rows = 0
    if schedule_rows is not None:
        schedule_by_match_id: dict[str, list[dict[str, Any]]] = {}
        for schedule in schedule_prepared:
            schedule_by_match_id.setdefault(schedule["match_id"], []).append(schedule)
        for scoreboard in scoreboard_prepared:
            try:
                prefix = _game_prefix(scoreboard["game_id"], label=f"scoreboardgames[{scoreboard['index']}]")
            except AliasDerivationError as error:
                issues.append({
                    "kind": "invalid_scoreboard_game_prefix",
                    "scoreboard_game_id": scoreboard["game_id"],
                    "error": str(error),
                })
                continue
            candidates = schedule_by_match_id.get(prefix, [])
            if len(candidates) != 1:
                kind = "matchschedule_prefix_ambiguous" if len(candidates) > 1 else "matchschedule_prefix_missing"
                issues.append({
                    "kind": kind,
                    "scoreboard_game_id": scoreboard["game_id"],
                    "match_id": prefix,
                    "candidate_count": len(candidates),
                })
                schedule_ambiguous_rows += kind.endswith("ambiguous")
                schedule_unmatched_rows += kind.endswith("missing")
                continue
            schedule = candidates[0]
            pairs, pair_reason = _infer_pairs(
                scoreboard["teams"],
                schedule["teams"],
                require_exactly_one_anchor=True,
            )
            if pairs is None:
                issues.append({
                    "kind": "matchschedule_team_pair_ambiguous",
                    "scoreboard_game_id": scoreboard["game_id"],
                    "match_id": prefix,
                    "scoreboard_teams": list(scoreboard["teams"]),
                    "matchschedule_teams": list(schedule["teams"]),
                    "reason": pair_reason or "schedule team-pair inference was ambiguous",
                })
                schedule_pair_ambiguous_rows += 1
                continue
            per_game = schedule_pairs_by_scoreboard_game.setdefault(scoreboard["game_id"], {})
            for source_name, target_name in pairs:
                source_key = _name_key(source_name)
                target_key = _name_key(target_name)
                evidence = {
                    "oe_team": source_name,
                    "leaguepedia_team": target_name,
                    "scoreboard_game_id": scoreboard["game_id"],
                    "matchschedule_match_id": schedule["match_id"],
                    "oe_game_id": scoreboard["game_id"],
                    "timestamp_delta_seconds": None,
                    "outcome_used": False,
                }
                scoreboard_schedule_evidence.setdefault((source_key, target_key), []).append(evidence)
                per_game[source_key] = (target_key, target_name, schedule["match_id"])

    # A transitive OE -> MatchSchedule alias may use only a base pair that
    # already satisfies the repeated, one-to-one OE -> ScoreboardGames rule.
    base_source_to_targets: dict[str, set[str]] = {}
    base_target_to_sources: dict[str, set[str]] = {}
    for source_key, target_key in candidate_evidence:
        base_source_to_targets.setdefault(source_key, set()).add(target_key)
        base_target_to_sources.setdefault(target_key, set()).add(source_key)
    accepted_base_pairs = {
        pair
        for pair, evidence_rows in candidate_evidence.items()
        if len({str(item["oe_game_id"]) for item in evidence_rows}) >= minimum_repeated_evidence
        and len(base_source_to_targets.get(pair[0], set())) == 1
        and len(base_target_to_sources.get(pair[1], set())) == 1
    }

    oe_schedule_evidence: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if schedule_rows is not None:
        for (source_key, scoreboard_key), evidence_rows in candidate_evidence.items():
            if (source_key, scoreboard_key) not in accepted_base_pairs:
                continue
            for source_evidence in evidence_rows:
                schedule_pair = schedule_pairs_by_scoreboard_game.get(
                    str(source_evidence["scoreboard_game_id"]), {}
                ).get(scoreboard_key)
                if schedule_pair is None:
                    continue
                schedule_key, schedule_name, match_id = schedule_pair
                source_name = str(source_evidence["oe_team"])
                transitive_evidence = {
                    "oe_team": source_name,
                    "leaguepedia_team": schedule_name,
                    "oe_game_id": source_evidence["oe_game_id"],
                    "scoreboard_game_id": source_evidence["scoreboard_game_id"],
                    "matchschedule_match_id": match_id,
                    "timestamp_delta_seconds": source_evidence["timestamp_delta_seconds"],
                    "outcome_used": False,
                }
                if source_evidence.get("oe_stable_team_key") is not None:
                    transitive_evidence["oe_stable_team_key"] = source_evidence["oe_stable_team_key"]
                oe_schedule_evidence.setdefault((source_key, schedule_key), []).append(transitive_evidence)

    accepted, review_only, conflicts = _finalize_evidence(
        candidate_evidence,
        minimum=minimum_repeated_evidence,
    )
    schedule_accepted, schedule_review_only, schedule_conflicts = _finalize_evidence(
        scoreboard_schedule_evidence,
        minimum=minimum_repeated_evidence,
        conflict_prefix="scoreboard_schedule",
    )
    transitive_accepted, transitive_review_only, transitive_conflicts = _finalize_evidence(
        oe_schedule_evidence,
        minimum=minimum_repeated_evidence,
        conflict_prefix="oe_schedule",
    )
    issues.extend(conflicts)
    issues.extend(schedule_conflicts)
    issues.extend(transitive_conflicts)
    all_review_only = review_only + schedule_review_only + transitive_review_only
    all_conflicts = conflicts + schedule_conflicts + transitive_conflicts

    status = "complete_research_only"
    if all_conflicts or all_review_only or issues:
        status = "review_required"
    if not accepted:
        status = "blocked_no_safe_mapping"
    source_names = sorted({_name_key(name) for row in oe_prepared for name in row["teams"]})
    stable_source_keys = sorted(
        {
            stable_key
            for row in oe_prepared
            for stable_key in (row.get("stable_team_keys") or ())
        }
    )
    accepted_display_names = {
        _name_key(name)
        for item in accepted
        for name in item.get("allowed_source_names", ())
    }
    accepted_stable_keys = {
        item["oe_team_key"]
        for item in accepted
        if item["oe_team_key"] in set(stable_source_keys)
    }
    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "captured_at": observed_at,
        "status": status,
        "authority": {
            "research_only": True,
            "public": False,
            "public_player_rating": False,
            "public_team_rating": False,
            "promotion": False,
            "deployment": False,
        },
        "input_bindings": input_records,
        "derivation_contract": {
            "match_rule": "unique_scoreboard_row_within_bounded_timestamp",
            "max_timestamp_delta_seconds": max_timestamp_delta_seconds,
            "minimum_repeated_evidence": minimum_repeated_evidence,
            "evidence_unit": "distinct_OE_game_id_and_distinct_ScoreboardGames_GameId",
            "pair_rule": "one_shared_normalized_name_anchor_then_remaining_pair",
            "no_shared_anchor_policy": "block_without_row_order_or_outcome_inference",
            "one_to_one_policy": "reject_source_or_target_conflicts",
            "singleton_policy": "review_only_and_never_accepted",
            "oe_team_key_field": "team_keys" if stable_team_key_binding is not None else None,
            "oe_team_key_rule": (
                "exact_nonempty_per_game_pair; display rebrands share one stable key"
                if stable_team_key_binding is not None
                else "not_supplied; normalized_display_name_is_identity"
            ),
            "schedule_bridge": {
                "enabled": schedule_rows is not None,
                "game_id_prefix_to_match_id": "exact_unique_prefix",
                "pair_rule": "exactly_one_shared_normalized_team_anchor",
                "direct_mapping": "ScoreboardGames team name to MatchSchedule team name",
                "transitive_mapping": "OE team name to MatchSchedule team name through accepted game evidence",
            },
            "outcome_used": False,
            "outcome_fields_ignored": True,
        },
        "mapping": accepted,
        "scoreboard_schedule_mapping": schedule_accepted,
        "oe_to_matchschedule_mapping": transitive_accepted,
        "canonical_aliases": (
            _canonical_alias_records(accepted, target_system="ScoreboardGames")
            + _canonical_alias_records(schedule_accepted, target_system="MatchSchedule")
            + _canonical_alias_records(transitive_accepted, target_system="OE_to_MatchSchedule")
        ),
        "review_only": all_review_only,
        "conflicts": all_conflicts,
        "issues": issues,
        "stable_oe_team_key_binding": stable_team_key_binding,
        "coverage": {
            "oe_rows": len(oe_rows),
            "valid_oe_rows": len(oe_prepared),
            "scoreboard_rows": len(scoreboard_rows),
            "valid_scoreboard_rows": len(scoreboard_prepared),
            "matchschedule_rows": len(schedule_rows) if schedule_rows is not None else 0,
            "valid_matchschedule_rows": len(schedule_prepared),
            "timestamp_matched_rows": matched_rows,
            "timestamp_ambiguous_rows": ambiguous_rows,
            "timestamp_unmatched_rows": unmatched_rows,
            "team_pair_ambiguous_rows": pair_ambiguous_rows,
            "matchschedule_prefix_matched_rows": len(schedule_pairs_by_scoreboard_game),
            "matchschedule_prefix_ambiguous_rows": schedule_ambiguous_rows,
            "matchschedule_prefix_unmatched_rows": schedule_unmatched_rows,
            "matchschedule_team_pair_ambiguous_rows": schedule_pair_ambiguous_rows,
            "oe_team_name_count": len(source_names),
            "stable_oe_team_key_enabled": stable_team_key_binding is not None,
            "stable_oe_team_key_count": len(stable_source_keys),
            "stable_display_name_conflict_count": len(stable_display_conflicts),
            "accepted_alias_count": len(accepted),
            "scoreboard_schedule_alias_count": len(schedule_accepted),
            "oe_matchschedule_alias_count": len(transitive_accepted),
            "review_alias_count": len(all_review_only),
            "conflict_count": len(all_conflicts),
            "accepted_name_coverage": len(accepted_display_names) / len(source_names) if source_names else 0.0,
            "accepted_stable_team_key_coverage": (
                len(accepted_stable_keys) / len(stable_source_keys)
                if stable_source_keys
                else None
            ),
        },
        "audit": {
            "unmatched_oe_game_ids": sorted(issue["oe_game_id"] for issue in issues if issue.get("kind") == "timestamp_unmatched"),
            "ambiguous_oe_game_ids": sorted(issue["oe_game_id"] for issue in issues if issue.get("kind") == "timestamp_ambiguous"),
            "pair_ambiguous_oe_game_ids": sorted(issue["oe_game_id"] for issue in issues if issue.get("kind") == "team_pair_ambiguous"),
            "missing_matchschedule_scoreboard_game_ids": sorted(
                issue["scoreboard_game_id"]
                for issue in issues
                if issue.get("kind") == "matchschedule_prefix_missing"
            ),
            "ambiguous_matchschedule_scoreboard_game_ids": sorted(
                issue["scoreboard_game_id"]
                for issue in issues
                if issue.get("kind") == "matchschedule_prefix_ambiguous"
            ),
            "matchschedule_pair_ambiguous_scoreboard_game_ids": sorted(
                issue["scoreboard_game_id"]
                for issue in issues
                if issue.get("kind") == "matchschedule_team_pair_ambiguous"
            ),
            "stable_display_name_conflict_keys": sorted(stable_display_conflicts),
            "outcome_used": False,
        },
    }
    output["mapping_sha256"] = _sha256(_canonical_json_bytes(output))
    return output


def verify_alias_mapping(payload: Mapping[str, Any]) -> None:
    """Verify a derived mapping's self-hash and non-authoritative contract."""

    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
        raise AliasDerivationError("alias mapping schema is invalid")
    claimed = _text(payload.get("mapping_sha256")).lower()
    if not _HEX64.fullmatch(claimed):
        raise AliasDerivationError("alias mapping self-hash is invalid")
    body = dict(payload)
    body.pop("mapping_sha256", None)
    if _sha256(_canonical_json_bytes(body)) != claimed:
        raise AliasDerivationError("alias mapping self-hash does not match payload")
    authority = payload.get("authority")
    if not isinstance(authority, Mapping) or authority.get("public") is not False:
        raise AliasDerivationError("alias mapping grants public authority")
    contract = payload.get("derivation_contract")
    if not isinstance(contract, Mapping) or contract.get("outcome_used") is not False:
        raise AliasDerivationError("alias mapping outcome contract is invalid")

    try:
        minimum = int(contract["minimum_repeated_evidence"])
    except (KeyError, TypeError, ValueError) as error:
        raise AliasDerivationError("alias mapping evidence threshold is invalid") from error
    if minimum < 2:
        raise AliasDerivationError("alias mapping evidence threshold is unsafe")
    bindings = payload.get("input_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) < {"oe", "scoreboardgames"}:
        raise AliasDerivationError("alias mapping input bindings are incomplete")
    for label, record in bindings.items():
        if not isinstance(record, Mapping) or record.get("integrity_verified") is not True:
            raise AliasDerivationError(f"{label} input binding is not verified")
        for field in ("sha256", "payload_sha256"):
            if not _HEX64.fullmatch(_text(record.get(field)).lower()):
                raise AliasDerivationError(f"{label} input binding {field} is invalid")
        for field in ("bytes", "payload_bytes"):
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise AliasDerivationError(f"{label} input binding {field} is invalid")
    oe_stable_binding = bindings["oe"].get("stable_team_key_binding")
    top_stable_binding = payload.get("stable_oe_team_key_binding")
    if oe_stable_binding is None:
        if top_stable_binding is not None:
            raise AliasDerivationError("stable OE team-key binding is duplicated inconsistently")
    else:
        if not isinstance(oe_stable_binding, Mapping):
            raise AliasDerivationError("stable OE team-key binding is invalid")
        _verify_stable_team_key_binding(oe_stable_binding)
        if not isinstance(top_stable_binding, Mapping) or dict(top_stable_binding) != dict(oe_stable_binding):
            raise AliasDerivationError("stable OE team-key binding is not consistently recorded")
    stable_ids = {
        str(value)
        for row in (oe_stable_binding.get("rows", ()) if isinstance(oe_stable_binding, Mapping) else ())
        for value in (row.get("team_keys", ()) if isinstance(row, Mapping) else ())
    }

    section_systems = {
        "mapping": "ScoreboardGames",
        "scoreboard_schedule_mapping": "MatchSchedule",
        "oe_to_matchschedule_mapping": "OE_to_MatchSchedule",
    }
    expected_aliases: set[tuple[str, str, str]] = set()
    for section, target_system in section_systems.items():
        rows = payload.get(section)
        if not isinstance(rows, list):
            raise AliasDerivationError(f"{section} is not an array")
        section_pairs: set[tuple[str, str]] = set()
        section_source_targets: dict[str, set[str]] = {}
        section_target_sources: dict[str, set[str]] = {}
        for item in rows:
            if not isinstance(item, Mapping) or item.get("status") != "accepted":
                raise AliasDerivationError(f"{section} contains a non-accepted row")
            try:
                evidence_count = int(item.get("evidence_count", 0))
                distinct_game_count = int(item.get("distinct_oe_game_count", 0))
                distinct_scoreboard_count = int(item.get("distinct_scoreboard_game_count", 0))
            except (TypeError, ValueError) as error:
                raise AliasDerivationError(f"{section} contains invalid evidence counts") from error
            if evidence_count < minimum or distinct_game_count < minimum or distinct_scoreboard_count < minimum:
                raise AliasDerivationError(f"{section} contains insufficient evidence")
            source_key = _text(item.get("oe_team_key"))
            target_key = _text(item.get("leaguepedia_team_key"))
            if not source_key or not target_key:
                raise AliasDerivationError(f"{section} contains an empty team key")
            pair = (source_key, target_key)
            if pair in section_pairs:
                raise AliasDerivationError(f"{section} contains duplicate accepted pairs")
            section_pairs.add(pair)
            section_source_targets.setdefault(source_key, set()).add(target_key)
            section_target_sources.setdefault(target_key, set()).add(source_key)
            game_ids = item.get("oe_game_ids")
            scoreboard_ids = item.get("scoreboard_game_ids")
            if (
                not isinstance(game_ids, list)
                or len(game_ids) != len(set(map(str, game_ids)))
                or len(game_ids) < minimum
                or not isinstance(scoreboard_ids, list)
                or len(scoreboard_ids) != len(set(map(str, scoreboard_ids)))
                or len(scoreboard_ids) < minimum
            ):
                raise AliasDerivationError(f"{section} contains non-distinct game evidence")
            match_ids = item.get("matchschedule_match_ids")
            if match_ids is not None and (
                not isinstance(match_ids, list)
                or len(match_ids) != len(set(map(str, match_ids)))
                or not match_ids
            ):
                raise AliasDerivationError(f"{section} contains invalid MatchSchedule evidence")
            if len(section_source_targets[source_key]) > 1 or len(section_target_sources[target_key]) > 1:
                raise AliasDerivationError(f"{section} violates one-to-one alias consistency")
            if section in {"mapping", "oe_to_matchschedule_mapping"} and oe_stable_binding is not None:
                stable_item_keys = item.get("stable_oe_team_keys")
                if (
                    not isinstance(stable_item_keys, list)
                    or stable_item_keys != [source_key]
                    or source_key not in stable_ids
                ):
                    raise AliasDerivationError(f"{section} stable OE team-key binding is incomplete")
            expected_aliases.add((target_system, source_key, target_key))

    aliases = payload.get("canonical_aliases")
    if not isinstance(aliases, list):
        raise AliasDerivationError("canonical alias records are missing")
    actual_aliases: set[tuple[str, str, str]] = set()
    allowed_systems = set(section_systems.values())
    for alias in aliases:
        if not isinstance(alias, Mapping):
            raise AliasDerivationError("canonical alias record is invalid")
        target_system = _text(alias.get("target_system"))
        source_key = _text(alias.get("source_name_key"))
        target_key = _text(alias.get("target_name_key"))
        if target_system not in allowed_systems or not source_key or not target_key:
            raise AliasDerivationError("canonical alias record identity is invalid")
        source_name = _text(alias.get("canonical_source_name"))
        target_name = _text(alias.get("canonical_target_name"))
        source_names = alias.get("allowed_source_names")
        target_names = alias.get("allowed_target_names")
        if (
            not source_name
            or not target_name
            or not isinstance(source_names, list)
            or not source_names
            or not all(_text(value) for value in source_names)
            or not isinstance(target_names, list)
            or not target_names
            or not all(_text(value) for value in target_names)
            or not _text(alias.get("equivalence_group"))
        ):
            raise AliasDerivationError("canonical alias record is incomplete")
        if source_name not in source_names:
            raise AliasDerivationError("canonical source name is not allowed")
        if target_name not in target_names:
            raise AliasDerivationError("canonical target name is not allowed")
        identity = (target_system, source_key, target_key)
        if identity in actual_aliases:
            raise AliasDerivationError("canonical alias records contain duplicates")
        actual_aliases.add(identity)
    if actual_aliases != expected_aliases:
        raise AliasDerivationError("canonical alias records do not match accepted mappings")

    schedule_bridge = contract.get("schedule_bridge")
    if payload.get("scoreboard_schedule_mapping") or payload.get("oe_to_matchschedule_mapping"):
        if not isinstance(schedule_bridge, Mapping) or schedule_bridge.get("enabled") is not True:
            raise AliasDerivationError("schedule bridge contract is missing")
        if "matchschedule" not in bindings:
            raise AliasDerivationError("schedule aliases have no MatchSchedule binding")
    if any(item.get("outcome_used") is True for item in payload.get("issues", ()) if isinstance(item, Mapping)):
        raise AliasDerivationError("alias mapping contains outcome-derived issue evidence")


def load_verified_alias_mapping(
    path: Path | str,
    *,
    expected_oe_payload_sha256: str | None = None,
    expected_scoreboard_payload_sha256: str | None = None,
    expected_matchschedule_payload_sha256: str | None = None,
    expected_stable_team_key_rows_sha256: str | None = None,
    allow_review_only: bool = False,
) -> dict[str, Any]:
    """Load an alias artifact against caller-supplied source payload digests.

    The expected OE and ScoreboardGames payload hashes are required.  A
    MatchSchedule and stable OE team-key hashes are required when the artifact
    contains those bindings.
    The returned ``source_to_allowed_targets`` index uses normalized source
    keys and contains only accepted, repeated-evidence mappings.
    """

    artifact_path = Path(path)
    if artifact_path.is_symlink() or not artifact_path.is_file():
        raise AliasDerivationError("alias artifact path is missing or unsafe")
    try:
        raw = artifact_path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AliasDerivationError("alias artifact cannot be read") from error
    if not isinstance(value, Mapping):
        raise AliasDerivationError("alias artifact must be an object")
    payload = dict(value)
    verify_alias_mapping(payload)
    status = _text(payload.get("status"))
    if status == "blocked_no_safe_mapping":
        raise AliasDerivationError("alias artifact has no safe mapping")
    if status != "complete_research_only" and not allow_review_only:
        raise AliasDerivationError("alias artifact is review-only")

    expected = {
        "oe": expected_oe_payload_sha256,
        "scoreboardgames": expected_scoreboard_payload_sha256,
    }
    bindings = payload["input_bindings"]
    if "matchschedule" in bindings:
        expected["matchschedule"] = expected_matchschedule_payload_sha256
    for label, expected_hash in expected.items():
        if expected_hash is None or not _HEX64.fullmatch(str(expected_hash).lower()):
            raise AliasDerivationError(f"expected {label} payload hash is required")
        actual_hash = _text(bindings[label].get("payload_sha256")).lower()
        if actual_hash != str(expected_hash).lower():
            raise AliasDerivationError(f"{label} source payload identity changed")
    stable_binding = bindings["oe"].get("stable_team_key_binding")
    if stable_binding is not None:
        if expected_stable_team_key_rows_sha256 is None or not _HEX64.fullmatch(
            str(expected_stable_team_key_rows_sha256).lower()
        ):
            raise AliasDerivationError("expected stable OE team-key digest is required")
        actual_stable_digest = _verify_stable_team_key_binding(stable_binding)
        if actual_stable_digest != str(expected_stable_team_key_rows_sha256).lower():
            raise AliasDerivationError("stable OE team-key identity changed")

    entries = [dict(item) for item in payload["canonical_aliases"]]
    index: dict[str, dict[str, list[str]]] = {}
    for item in entries:
        system = str(item["target_system"])
        source_key = str(item["source_name_key"])
        target_names = [str(value) for value in item["allowed_target_names"]]
        system_index = index.setdefault(system, {})
        if source_key in system_index and system_index[source_key] != target_names:
            raise AliasDerivationError("alias artifact has conflicting source targets")
        system_index[source_key] = target_names
    return {
        "schema_version": payload["schema_version"],
        "artifact_sha256": _sha256(raw),
        "mapping_sha256": payload["mapping_sha256"],
        "status": status,
        "input_bindings": {str(key): dict(value) for key, value in bindings.items()},
        "entries": entries,
        "source_to_allowed_targets": index,
    }


__all__ = [
    "AliasDerivationError",
    "DEFAULT_MAX_TIMESTAMP_DELTA_SECONDS",
    "DEFAULT_MIN_REPEATED_EVIDENCE",
    "SCHEMA_VERSION",
    "derive_team_alias_mapping",
    "load_verified_alias_mapping",
    "verify_alias_mapping",
]
