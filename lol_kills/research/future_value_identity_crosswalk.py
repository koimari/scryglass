"""Outcome-free, strict-prior identity candidates for future-value research.

Oracle's Elixir rows can contain a name while the stable OE player or team
identity is missing.  This module records candidates from identities that
were already observed before the target map.  It never changes the accepted
census and it never turns a candidate into a model-eligible row.

The output is a research receipt.  It contains only identity evidence,
timestamps, and source bindings.  Match outcomes and final-game metrics are
not copied to the receipt.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json
import math
import re
import unicodedata

import pandas as pd

from lol_kills.etl.aliases import normalize_team
from lol_kills.etl.source_keys import canonical_source_game_key
from lol_kills.v2.tierlists.accepted_census import canonical_game_ids, identity_sha256


SCHEMA_VERSION = "scryglass:future-value-identity-crosswalk:v2"
STATUS = "research_only_identity_candidates"
SOURCE_RECEIPT_SCHEMA = "scryglass:future-value-rating-source:v1"
SOURCE_RECEIPT_STATUS = "accepted_source_bound_development_only"
AUTHORITY = {
    "research_only": True,
    "public": False,
    "promotion": False,
    "deployment": False,
    "model_eligible_census": False,
}
ROLES = ("top", "jungle", "mid", "bot", "support")
SIDES = ("blue", "red")
_HEX64 = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_OUTCOME_KEY_TOKENS = {"final", "outcome", "result", "winner", "win", "won"}
_SOURCE_RECEIPT_FIELDS = {
    "accepted_game_ids",
    "authority",
    "checkpoint_coverage",
    "identity_coverage",
    "model_contract",
    "model_eligible_game_count",
    "model_eligible_game_ids",
    "model_eligible_identity_sha256",
    "model_exclusions",
    "receipt_sha256",
    "schema_version",
    "source_as_of",
    "source_extra_game_ids",
    "source_files",
    "source_game_count",
    "source_identity_sha256",
    "source_rows",
    "status",
}
_SOURCE_AUTHORITY = {
    "deployment": False,
    "merge": False,
    "promotion": False,
    "public_player_rating": False,
    "public_probability": False,
    "public_team_rating": False,
    "research_only": True,
}
_CROSSWALK_FIELDS = {
    "accepted_game_ids", "assignments", "authority", "candidate_key_types",
    "counts", "method", "receipt_sha256", "rejected", "schema_version",
    "source_as_of", "source_file_records", "source_game_count",
    "source_identity_sha256", "source_receipt_file", "source_receipt_sha256",
    "status",
}


class IdentityCrosswalkError(ValueError):
    """Raised when source identity evidence cannot be verified."""


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
        raise IdentityCrosswalkError("crosswalk contains non-canonical JSON") from error


def canonical_sha256(value: object) -> str:
    """Return the canonical JSON hash used by crosswalk receipts."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _string(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _hash(value: Any) -> str | None:
    text = _string(value)
    if text is None or _HEX64.fullmatch(text) is None:
        return None
    return text.lower()


def _timestamp(value: Any, *, label: str) -> datetime:
    text = _string(value)
    if text is None:
        raise IdentityCrosswalkError(f"{label} timestamp is missing")
    candidate = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as error:
        raise IdentityCrosswalkError(f"{label} timestamp is invalid") from error
    if parsed.tzinfo is None:
        raise IdentityCrosswalkError(f"{label} timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _timestamp_or_none(value: Any) -> datetime | None:
    try:
        return _timestamp(value, label="row")
    except IdentityCrosswalkError:
        return None


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _norm(value: Any) -> str | None:
    text = _string(value)
    if text is None:
        return None
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split()) or None


def _team_norm(value: Any) -> str | None:
    text = _string(value)
    if text is None:
        return None
    try:
        text = str(normalize_team(text))
    except (TypeError, ValueError):
        pass
    return _norm(text)


def _side(value: Any) -> str | None:
    text = _norm(value)
    return {"blue": "blue", "b": "blue", "red": "red", "r": "red"}.get(text or "")


def _role(value: Any) -> str | None:
    text = _norm(value)
    return {
        "top": "top",
        "jng": "jungle",
        "jg": "jungle",
        "jungle": "jungle",
        "mid": "mid",
        "middle": "mid",
        "bot": "bot",
        "adc": "bot",
        "carry": "bot",
        "sup": "support",
        "support": "support",
        "utility": "support",
    }.get(text or "")


def _identity(value: Any, prefix: str) -> str | None:
    text = _string(value)
    if text is None or not text.startswith(prefix):
        return None
    return text


def _frame(value: Any, label: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        result = value.copy()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = pd.DataFrame(list(value)).copy()
    else:
        raise IdentityCrosswalkError(f"{label} rows are invalid")
    if result.empty:
        raise IdentityCrosswalkError(f"{label} rows are empty")
    return result


def _game_column(frame: pd.DataFrame, label: str) -> str:
    for column in ("game_uid", "gameid", "game_id"):
        if column in frame.columns:
            return column
    raise IdentityCrosswalkError(f"{label} has no game identity column")


def _date_column(frame: pd.DataFrame, label: str) -> str:
    for column in ("date", "game_date", "timestamp"):
        if column in frame.columns:
            return column
    raise IdentityCrosswalkError(f"{label} has no timestamp column")


def _canonical_frame_games(frame: pd.DataFrame, label: str) -> pd.Series:
    column = _game_column(frame, label)
    result = frame[column].map(lambda value: _string(value) or "")
    if result.eq("").any():
        raise IdentityCrosswalkError(f"{label} has an empty game identity")
    return result


def _safe_regular_file(value: Any, label: str) -> Path:
    text = _string(value)
    if text is None:
        raise IdentityCrosswalkError(f"{label} source file path is required")
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        raise IdentityCrosswalkError(f"{label} source file path is unsafe")
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise IdentityCrosswalkError(f"{label} source file path is unsafe")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise IdentityCrosswalkError(f"{label} source file path is unsafe") from error
    if not resolved.is_file():
        raise IdentityCrosswalkError(f"{label} source file path is unsafe")
    return resolved


def _valid_file_record(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise IdentityCrosswalkError(f"{label} source file record is invalid")
    if set(map(str, value)) - {"bytes", "locator", "path", "sha256", "year"}:
        raise IdentityCrosswalkError(f"{label} source file record has unknown fields")
    digest = _hash(value.get("sha256"))
    bytes_value = value.get("bytes")
    locator = _string(value.get("locator"))
    path = _safe_regular_file(value.get("path"), label)
    if (
        digest is None
        or isinstance(bytes_value, bool)
        or not isinstance(bytes_value, int)
        or bytes_value <= 0
    ):
        raise IdentityCrosswalkError(f"{label} source file record is invalid")
    if locator is None:
        locator = path.name
    result = {
        "bytes": int(bytes_value),
        "locator": locator,
        "path": str(path),
        "sha256": digest,
    }
    if "year" in value:
        year = value.get("year")
        if isinstance(year, bool) or not isinstance(year, int):
            raise IdentityCrosswalkError(f"{label} source file year is invalid")
        result["year"] = year
    if path.stat().st_size != int(bytes_value):
        raise IdentityCrosswalkError(f"{label} source file byte count changed")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise IdentityCrosswalkError(f"{label} source file hash changed")
    return result


def _bound_source_receipt(
    source_receipt: Mapping[str, Any],
    source_receipt_file_record: Mapping[str, Any] | None,
    trusted_source_receipt_file_sha256: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _valid_file_record(source_receipt_file_record, "source receipt")
    trusted_digest = _hash(trusted_source_receipt_file_sha256)
    if trusted_digest is None or trusted_digest != record["sha256"]:
        raise IdentityCrosswalkError("source receipt file trust digest is invalid")
    try:
        loaded = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IdentityCrosswalkError("source receipt file is invalid") from error
    if not isinstance(loaded, dict) or dict(source_receipt) != loaded:
        raise IdentityCrosswalkError("source receipt does not match verified bytes")
    return loaded, record


def _source_receipt(
    source_receipt: Mapping[str, Any],
) -> tuple[tuple[str, ...], datetime, str, str]:
    if not isinstance(source_receipt, Mapping):
        raise IdentityCrosswalkError("source receipt is required")
    if set(source_receipt) - _SOURCE_RECEIPT_FIELDS:
        raise IdentityCrosswalkError("source receipt contains unknown fields")
    if (
        source_receipt.get("schema_version") != SOURCE_RECEIPT_SCHEMA
        or source_receipt.get("status") != SOURCE_RECEIPT_STATUS
    ):
        raise IdentityCrosswalkError("source receipt schema or status is invalid")
    accepted_raw = source_receipt.get("accepted_game_ids")
    if not isinstance(accepted_raw, Sequence) or isinstance(
        accepted_raw, (str, bytes, bytearray)
    ):
        raise IdentityCrosswalkError("source receipt accepted IDs are invalid")
    accepted = canonical_game_ids(accepted_raw)
    if tuple(str(value) for value in accepted_raw) != accepted or not accepted:
        raise IdentityCrosswalkError("source receipt accepted IDs are not canonical")
    if source_receipt.get("source_game_count") != len(accepted):
        raise IdentityCrosswalkError("source receipt game count is invalid")
    source_identity = _hash(source_receipt.get("source_identity_sha256"))
    if source_identity != identity_sha256(accepted):
        raise IdentityCrosswalkError("source receipt identity is invalid")
    source_as_of = _timestamp(source_receipt.get("source_as_of"), label="source_as_of")
    claimed = _hash(source_receipt.get("receipt_sha256"))
    if claimed is None:
        raise IdentityCrosswalkError("source receipt hash is invalid")
    payload = dict(source_receipt)
    payload.pop("receipt_sha256", None)
    if canonical_sha256(payload) != claimed:
        raise IdentityCrosswalkError("source receipt hash does not match payload")
    authority = source_receipt.get("authority")
    if not isinstance(authority, Mapping) or dict(authority) != _SOURCE_AUTHORITY:
        raise IdentityCrosswalkError("source receipt authority is invalid")
    return accepted, source_as_of, source_identity, claimed


def _source_records(
    source_receipt: Mapping[str, Any],
    supplied: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    value: Mapping[str, Any] | None = supplied
    if value is None:
        receipt_records = source_receipt.get("source_files")
        value = receipt_records if isinstance(receipt_records, Mapping) else None
    if not isinstance(value, Mapping) or not value:
        raise IdentityCrosswalkError("source file hashes are required")
    result = {
        str(label): _valid_file_record(record, str(label))
        for label, record in sorted(value.items(), key=lambda item: str(item[0]))
    }
    receipt_records = source_receipt.get("source_files")
    if not isinstance(receipt_records, Mapping) or set(result) != set(receipt_records):
        raise IdentityCrosswalkError("source file binding set is invalid")
    if not {"maps", "players", "teams"}.issubset(result):
        raise IdentityCrosswalkError("maps, players, and teams source files are required")
    if isinstance(receipt_records, Mapping):
        for label, expected in receipt_records.items():
            if label not in result:
                raise IdentityCrosswalkError(f"source file binding is missing: {label}")
            expected_hash = _hash(expected.get("sha256")) if isinstance(expected, Mapping) else None
            expected_bytes = expected.get("bytes") if isinstance(expected, Mapping) else None
            if (
                expected_hash is not None
                and result[label]["sha256"] != expected_hash
            ) or (
                isinstance(expected_bytes, int)
                and not isinstance(expected_bytes, bool)
                and result[label]["bytes"] != expected_bytes
            ):
                raise IdentityCrosswalkError(f"source file binding changed: {label}")
    return result


def _load_source_frame(record: Mapping[str, Any], label: str) -> pd.DataFrame:
    path = Path(str(record["path"]))
    suffix = path.suffix.casefold()
    try:
        if suffix in {".parquet", ".pq"}:
            frame = pd.read_parquet(path)
        elif suffix in {".csv", ".gz"}:
            frame = pd.read_csv(path, low_memory=False)
        elif suffix in {".jsonl", ".ndjson"}:
            frame = pd.read_json(path, lines=True)
        elif suffix == ".json":
            loaded = json.loads(path.read_text(encoding="utf-8"))
            frame = pd.DataFrame(loaded)
        else:
            raise IdentityCrosswalkError(f"{label} source file type is unsupported")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise IdentityCrosswalkError(f"{label} source file cannot be loaded") from error
    return _frame(frame, label)


def _frame_from_verified_file(
    supplied: Any,
    record: Mapping[str, Any],
    label: str,
) -> pd.DataFrame:
    supplied_frame = _frame(supplied, label)
    verified_frame = _load_source_frame(record, label)
    try:
        pd.testing.assert_frame_equal(
            supplied_frame.reset_index(drop=True),
            verified_frame.reset_index(drop=True),
            check_dtype=True,
            check_like=False,
        )
    except AssertionError as error:
        raise IdentityCrosswalkError(
            f"{label} rows do not match verified source bytes"
        ) from error
    return verified_frame


def _history_rows(
    frame: pd.DataFrame,
    *,
    game_column: str,
    date_column: str,
    id_column: str,
    prefix: str,
    name_column: str,
    team_column: str,
    league_column: str,
    side_column: str | None,
    role_column: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        identity = _identity(row.get(id_column), prefix)
        name = _norm(row.get(name_column))
        row_date = _timestamp_or_none(row.get(date_column))
        game_id = _string(row.get(game_column))
        if identity is None or name is None or row_date is None or game_id is None:
            continue
        rows.append(
            {
                "id": identity,
                "name": name,
                "team": _team_norm(row.get(team_column)),
                "league": _norm(row.get(league_column)),
                "side": _side(row.get(side_column)) if side_column else None,
                "role": _role(row.get(role_column)) if role_column else None,
                "game_id": game_id,
                "timestamp": row_date,
            }
        )
    return rows


def _history_index(rows: Sequence[Mapping[str, Any]], *, kind: str) -> dict[str, dict[Any, list[Mapping[str, Any]]]]:
    index: dict[str, dict[Any, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        index["name"][row["name"]].append(row)
        if kind == "player":
            if row["team"] is not None and row["league"] is not None:
                index["name_team_league"][(row["name"], row["team"], row["league"])].append(row)
            if row["team"] is not None:
                index["name_team"][(row["name"], row["team"])].append(row)
            if row["league"] is not None:
                index["name_league"][(row["name"], row["league"])].append(row)
            if row["role"] is not None:
                index["name_role"][(row["name"], row["role"])].append(row)
        elif row["league"] is not None:
            index["name_league"][(row["name"], row["league"])].append(row)
    return index


def _candidate(
    *,
    name: str | None,
    team: str | None,
    league: str | None,
    role: str | None,
    target_date: datetime,
    history: Mapping[str, Mapping[Any, Sequence[Mapping[str, Any]]]],
    kind: str,
) -> dict[str, Any]:
    if name is None:
        return {
            "status": "rejected",
            "reason": f"missing_{kind}_name",
            "candidate_key_type": None,
            "candidate_ids": [],
            "evidence": [],
        }
    by_name = history.get("name", {})
    prior = [row for row in by_name.get(name, ()) if row["timestamp"] < target_date]
    key_options: list[tuple[str, set[str], Any]] = []
    if kind == "player":
        if team is not None and league is not None:
            rows = [
                row
                for row in history.get("name_team_league", {}).get((name, team, league), ())
                if row["timestamp"] < target_date
            ]
            key_options.append(
                (
                    "player_name+team_name+league",
                    {str(row["id"]) for row in rows},
                    lambda row: row in rows,
                )
            )
        if team is not None:
            rows = [
                row
                for row in history.get("name_team", {}).get((name, team), ())
                if row["timestamp"] < target_date
            ]
            key_options.append(
                (
                    "player_name+team_name",
                    {str(row["id"]) for row in rows},
                    lambda row: row in rows,
                )
            )
        if league is not None:
            rows = [
                row
                for row in history.get("name_league", {}).get((name, league), ())
                if row["timestamp"] < target_date
            ]
            key_options.append(
                (
                    "player_name+league",
                    {str(row["id"]) for row in rows},
                    lambda row: row in rows,
                )
            )
        if role is not None:
            rows = [
                row
                for row in history.get("name_role", {}).get((name, role), ())
                if row["timestamp"] < target_date
            ]
            key_options.append(
                (
                    "player_name+role",
                    {str(row["id"]) for row in rows},
                    lambda row: row in rows,
                )
            )
    else:
        if team is not None and league is not None:
            rows = [
                row
                for row in history.get("name_league", {}).get((name, league), ())
                if row["timestamp"] < target_date
            ]
            key_options.append(
                (
                    "team_name+league",
                    {str(row["id"]) for row in rows if row["name"] == name and row["team"] == team},
                    lambda row: row in rows and row["name"] == name and row["team"] == team,
                )
            )
        if team is not None:
            rows = [
                row
                for row in history.get("name", {}).get(name, ())
                if row["timestamp"] < target_date and row["team"] == team
            ]
            key_options.append(
                (
                    "team_name",
                    {str(row["id"]) for row in rows},
                    lambda row: row in rows,
                )
            )
    selected: tuple[str, set[str], Any] | None = None
    for option in key_options:
        if option[1]:
            selected = option
            break
    if selected is None:
        return {
            "status": "rejected",
            "reason": f"{kind}_identity_not_seen_strict_prior",
            "candidate_key_type": key_options[0][0] if key_options else None,
            "candidate_ids": [],
            "evidence": [],
        }
    key_type, candidate_ids, predicate = selected
    evidence_by_id: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in prior:
        if predicate(row):
            evidence_by_id[str(row["id"])].append(row)
    if len(candidate_ids) != 1:
        return {
            "status": "rejected",
            "reason": f"ambiguous_{kind}_identity",
            "candidate_key_type": key_type,
            "candidate_ids": sorted(candidate_ids),
            "evidence": [],
        }
    candidate_id = next(iter(candidate_ids))
    evidence = sorted(
        {
            (str(row["game_id"]), _iso(row["timestamp"]))
            for row in evidence_by_id[candidate_id]
        }
    )
    evidence_rows = [
        {"game_id": game_id, "timestamp": timestamp, "outcome_used": False}
        for game_id, timestamp in evidence
    ]
    first_seen = min((_timestamp(row["timestamp"], label="evidence") for row in evidence_rows), default=None)
    if first_seen is None:
        return {
            "status": "rejected",
            "reason": f"{kind}_identity_evidence_missing",
            "candidate_key_type": key_type,
            "candidate_ids": [candidate_id],
            "evidence": [],
        }
    return {
        "status": "candidate",
        "candidate_id": candidate_id,
        "candidate_key_type": key_type,
        "candidate_ids": [candidate_id],
        "evidence": evidence_rows,
        "first_seen_timestamp": _iso(first_seen),
    }


def _map_groups(
    frame: pd.DataFrame,
    *,
    game_column: str,
    accepted: set[str],
    selected: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    games = _canonical_frame_games(frame, "source")
    work = frame.copy()
    work["__crosswalk_game_id"] = games
    wanted = accepted if selected is None else accepted & selected
    work = work[work["__crosswalk_game_id"].isin(wanted)]
    return {
        str(game_id): group.drop(columns=["__crosswalk_game_id"])
        for game_id, group in work.groupby("__crosswalk_game_id", sort=False)
    }


def _authoritative_map_dates(
    frame: pd.DataFrame,
    game_ids: pd.Series,
    *,
    date_column: str,
    accepted: set[str],
) -> dict[str, datetime]:
    """Return one trusted UTC date for every accepted map.

    The map frame is the only source of chronology.  A duplicate accepted map
    or a map without a valid date makes the source unsafe for strict-prior
    identity work.
    """

    dates: dict[str, datetime] = {}
    for game_id, raw_date in zip(game_ids.tolist(), frame[date_column].tolist()):
        game_id = str(game_id)
        if game_id not in accepted:
            continue
        if game_id in dates:
            raise IdentityCrosswalkError(
                f"accepted maps contain duplicate map rows: {game_id}"
            )
        dates[game_id] = _timestamp(raw_date, label=f"maps {game_id}")
    missing = sorted(accepted - set(dates))
    if missing:
        raise IdentityCrosswalkError(
            f"accepted maps are missing authoritative dates: {missing[:5]}"
        )
    return dates


def _validate_source_row_dates(
    frame: pd.DataFrame,
    game_ids: pd.Series,
    *,
    date_column: str,
    map_dates: Mapping[str, datetime],
    accepted: set[str],
    label: str,
    rows_per_map: int,
) -> None:
    """Require complete accepted rows and exact equality to map timestamps."""

    counts: Counter[str] = Counter(
        str(game_id)
        for game_id in game_ids.tolist()
        if str(game_id) in accepted
    )
    missing = sorted(accepted - set(counts))
    if missing:
        raise IdentityCrosswalkError(
            f"{label} are missing accepted maps: {missing[:5]}"
        )
    invalid_counts = sorted(
        game_id for game_id in accepted if counts[game_id] != rows_per_map
    )
    if invalid_counts:
        raise IdentityCrosswalkError(
            f"{label} have duplicate or incomplete map rows: {invalid_counts[:5]}"
        )

    mismatches: list[str] = []
    for game_id, raw_date in zip(game_ids.tolist(), frame[date_column].tolist()):
        game_id = str(game_id)
        if game_id not in accepted:
            continue
        try:
            row_date = _timestamp(raw_date, label=f"{label} {game_id}")
        except IdentityCrosswalkError as error:
            raise IdentityCrosswalkError(
                f"{label} contain an invalid accepted map date: {game_id}"
            ) from error
        if row_date != map_dates[game_id]:
            mismatches.append(game_id)
    if mismatches:
        raise IdentityCrosswalkError(
            f"{label} game dates do not match authoritative map dates: "
            + ", ".join(sorted(set(mismatches))[:5])
        )


def _require_columns(frame: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise IdentityCrosswalkError(f"{label} is missing columns: {', '.join(missing)}")


def _outcome_free(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() == "outcome_used":
                if item is not False:
                    return False
                continue
            if _is_outcome_key(key) or not _outcome_free(item):
                return False
        return True
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return all(_outcome_free(item) for item in value)
    return True


def _is_outcome_key(value: Any) -> bool:
    text = str(value).casefold()
    tokens = [token for token in re.split(r"[^a-z0-9]+", text) if token]
    compact = "".join(tokens)
    return any(token in _OUTCOME_KEY_TOKENS for token in tokens) or any(
        compact.startswith(prefix) or compact.endswith(prefix)
        for prefix in ("final", "outcome", "result", "winner")
    )


def _allowed_keys(value: Any, allowed: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(map(str, value)) - allowed:
        raise IdentityCrosswalkError(f"{label} schema is invalid")


def _closed_crosswalk_schema(crosswalk: Mapping[str, Any]) -> None:
    if set(crosswalk) != _CROSSWALK_FIELDS:
        raise IdentityCrosswalkError("identity crosswalk schema is not closed")
    _allowed_keys(crosswalk.get("method"), {
        "accepted_identity_prefixes", "anonymous_ids_allowed",
        "model_eligibility_changed", "outcome_used", "target_timestamp_field",
        "timestamp_rule",
    }, "identity method")
    _allowed_keys(crosswalk.get("counts"), {
        "accepted_game_count", "candidate_game_count",
        "candidate_player_assignment_count", "candidate_team_assignment_count",
        "rejected_game_count", "unresolved_reason_counts",
    }, "identity counts")
    for assignment in crosswalk.get("assignments") or ():
        _allowed_keys(assignment, {
            "game_id", "outcome_used", "player_assignments", "target_timestamp",
            "team_assignments",
        }, "identity assignment")
        for row in assignment.get("player_assignments") or ():
            _allowed_keys(row, {
                "candidate_key_type", "evidence", "first_seen_timestamp",
                "outcome_used", "player_id", "player_name", "role", "side",
                "team_id",
            }, "identity player assignment")
            for evidence in row.get("evidence") or ():
                _allowed_keys(evidence, {"game_id", "outcome_used", "timestamp"}, "identity evidence")
        for row in assignment.get("team_assignments") or ():
            _allowed_keys(row, {
                "candidate_key_type", "evidence", "first_seen_timestamp",
                "outcome_used", "side", "team_id", "team_name",
            }, "identity team assignment")
            for evidence in row.get("evidence") or ():
                _allowed_keys(evidence, {"game_id", "outcome_used", "timestamp"}, "identity evidence")
    for rejected in crosswalk.get("rejected") or ():
        _allowed_keys(rejected, {
            "game_id", "outcome_used", "player_rejections", "reasons",
            "target_timestamp", "team_rejections",
        }, "identity rejection")
        for row in rejected.get("player_rejections") or ():
            _allowed_keys(row, {
                "candidate_ids", "candidate_key_type", "outcome_used",
                "player_name", "reason", "role", "side",
            }, "identity player rejection")
        for row in rejected.get("team_rejections") or ():
            _allowed_keys(row, {
                "candidate_ids", "candidate_key_type", "outcome_used", "reason",
                "side", "team_name",
            }, "identity team rejection")


def _build_identity_crosswalk(
    *,
    maps: pd.DataFrame | Sequence[Mapping[str, Any]],
    players: pd.DataFrame | Sequence[Mapping[str, Any]],
    teams: pd.DataFrame | Sequence[Mapping[str, Any]],
    source_receipt: Mapping[str, Any],
    source_receipt_file_record: Mapping[str, Any] | None = None,
    trusted_source_receipt_file_sha256: str | None = None,
    source_file_records: Mapping[str, Mapping[str, Any]] | None = None,
    source_records: Mapping[str, Mapping[str, Any]] | None = None,
    _verify_replay: bool = True,
) -> dict[str, Any]:
    """Build strict-prior identity candidates for every accepted map.

    Existing stable IDs are evidence from the target source row.  Missing IDs
    are proposed only from rows dated strictly before that map.  A candidate
    map is emitted only when all ten player slots, both team rows, and both
    player/team identity views agree.
    """

    if source_file_records is not None and source_records is not None:
        raise IdentityCrosswalkError("source file records were supplied twice")
    source_receipt, receipt_file = _bound_source_receipt(
        source_receipt,
        source_receipt_file_record,
        trusted_source_receipt_file_sha256,
    )
    accepted_ids, source_as_of, source_identity, source_receipt_hash = _source_receipt(source_receipt)
    records = _source_records(source_receipt, source_file_records or source_records)
    maps_frame = _frame_from_verified_file(maps, records["maps"], "maps")
    players_frame = _frame_from_verified_file(players, records["players"], "players")
    teams_frame = _frame_from_verified_file(teams, records["teams"], "teams")
    map_game_column = _game_column(maps_frame, "maps")
    player_game_column = _game_column(players_frame, "players")
    team_game_column = _game_column(teams_frame, "teams")
    map_date_column = _date_column(maps_frame, "maps")
    player_date_column = _date_column(players_frame, "players")
    team_date_column = _date_column(teams_frame, "teams")
    _require_columns(maps_frame, [map_game_column, map_date_column], "maps")
    _require_columns(
        players_frame,
        [player_game_column, player_date_column, "side", "position", "playername", "playerid", "teamname", "teamid", "champion"],
        "players",
    )
    _require_columns(
        teams_frame,
        [team_game_column, team_date_column, "side", "teamname", "teamid"],
        "teams",
    )
    accepted_set = set(accepted_ids)
    map_ids = _canonical_frame_games(maps_frame, "maps")
    accepted_map_ids = set(map_ids) & accepted_set
    if accepted_map_ids != accepted_set:
        missing = sorted(accepted_set - accepted_map_ids)
        raise IdentityCrosswalkError(f"accepted maps are missing from source frames: {missing[:5]}")
    if map_ids[map_ids.isin(accepted_set)].duplicated().any():
        raise IdentityCrosswalkError("accepted maps contain duplicate map rows")
    player_ids = _canonical_frame_games(players_frame, "players")
    team_ids = _canonical_frame_games(teams_frame, "teams")
    map_dates = _authoritative_map_dates(
        maps_frame,
        map_ids,
        date_column=map_date_column,
        accepted=accepted_set,
    )
    _validate_source_row_dates(
        players_frame,
        player_ids,
        date_column=player_date_column,
        map_dates=map_dates,
        accepted=accepted_set,
        label="players",
        rows_per_map=10,
    )
    _validate_source_row_dates(
        teams_frame,
        team_ids,
        date_column=team_date_column,
        map_dates=map_dates,
        accepted=accepted_set,
        label="teams",
        rows_per_map=2,
    )
    # Frozen parquet files can retain physical rows that the accepted census
    # excludes.  They are ignored after identity validation of the row key.
    player_missing_mask = players_frame["playerid"].map(
        lambda value: _identity(value, "oe:player:") is None
    ) | players_frame["teamid"].map(lambda value: _identity(value, "oe:team:") is None)
    team_missing_mask = teams_frame["teamid"].map(
        lambda value: _identity(value, "oe:team:") is None
    )
    candidate_game_ids = set(player_ids[player_missing_mask]) | set(team_ids[team_missing_mask])
    candidate_game_ids &= accepted_set
    map_groups = _map_groups(
        maps_frame, game_column=map_game_column, accepted=accepted_set, selected=candidate_game_ids
    )
    player_groups = _map_groups(
        players_frame, game_column=player_game_column, accepted=accepted_set, selected=candidate_game_ids
    )
    team_groups = _map_groups(
        teams_frame, game_column=team_game_column, accepted=accepted_set, selected=candidate_game_ids
    )
    player_history_columns = [
        player_game_column,
        player_date_column,
        "playerid",
        "playername",
        "teamname",
        "league" if "league" in players_frame.columns else player_game_column,
        "side",
        "position",
    ]
    player_history_frame = players_frame.loc[
        player_ids.isin(accepted_set), player_history_columns
    ].copy()
    player_history_frame["__canonical_map_date"] = player_history_frame[
        player_game_column
    ].map(canonical_source_game_key).map(map_dates)
    if "league" not in player_history_frame.columns:
        player_history_frame["league"] = ""
    player_history_rows = _history_rows(
        player_history_frame,
        game_column=player_game_column,
        date_column="__canonical_map_date",
        id_column="playerid",
        prefix="oe:player:",
        name_column="playername",
        team_column="teamname",
        league_column="league",
        side_column="side",
        role_column="position",
    )
    team_history_frames: list[pd.DataFrame] = []
    for frame, frame_ids, game_column, date_column in (
        (players_frame, player_ids, player_game_column, player_date_column),
        (teams_frame, team_ids, team_game_column, team_date_column),
    ):
        columns = [game_column, date_column, "teamid", "teamname", "side"]
        if "league" in frame.columns:
            columns.append("league")
        selected = frame.loc[frame_ids.isin(accepted_set), columns].copy()
        selected = selected.rename(columns={game_column: "__game", date_column: "__date"})
        selected["__canonical_map_date"] = selected["__game"].map(
            canonical_source_game_key
        ).map(map_dates)
        if "league" not in selected.columns:
            selected["league"] = ""
        team_history_frames.append(selected)
    team_history = _history_rows(
        pd.concat(team_history_frames, ignore_index=True),
        game_column="__game",
        date_column="__canonical_map_date",
        id_column="teamid",
        prefix="oe:team:",
        name_column="teamname",
        team_column="teamname",
        league_column="league",
        side_column="side",
        role_column=None,
    )
    # Team history rows use teamname for both the display and lookup fields.
    # Rebuild the normalized display field explicitly for clarity.
    for row in team_history:
        row["name"] = _team_norm(row["name"])
        row["team"] = _team_norm(row["team"])
    player_history = _history_index(player_history_rows, kind="player")
    team_history_index = _history_index(team_history, kind="team")
    assignments: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    missing_identity_maps = 0
    for game_id in accepted_ids:
        if game_id not in candidate_game_ids:
            continue
        map_group = map_groups[game_id]
        player_group = player_groups.get(game_id, pd.DataFrame())
        team_group = team_groups.get(game_id, pd.DataFrame())
        target_date = map_dates.get(game_id)
        reasons: list[str] = []
        if target_date is None:
            reasons.append("target_timestamp_missing")
        elif target_date > source_as_of:
            reasons.append("target_after_source_as_of")
        if len(player_group) != 10:
            reasons.append("player_row_count_not_10")
        if len(team_group) != 2:
            reasons.append("team_row_count_not_2")
        player_assignments: list[dict[str, Any]] = []
        team_assignments: list[dict[str, Any]] = []
        player_rejections: list[dict[str, Any]] = []
        team_rejections: list[dict[str, Any]] = []
        if len(player_group) == 10:
            slots = {(_side(row.get("side")), _role(row.get("position"))) for _, row in player_group.iterrows()}
            expected = {(side, role) for side in SIDES for role in ROLES}
            if slots != expected:
                reasons.append("player_role_or_side_closure_invalid")
            champions_by_side: dict[str | None, list[str | None]] = defaultdict(list)
            for _, row in player_group.iterrows():
                champions_by_side[_side(row.get("side"))].append(_norm(row.get("champion")))
            if any(
                len(champions_by_side.get(side, [])) != 5
                or any(value is None for value in champions_by_side.get(side, []))
                or len(set(champions_by_side.get(side, []))) != 5
                for side in SIDES
            ):
                reasons.append("champion_identity_not_unique")
        if len(team_group) == 2 and {
            _side(value) for value in team_group["side"]
        } != set(SIDES):
            reasons.append("team_side_closure_invalid")
        target_has_missing_identity = (
            len(player_group) == 10
            and any(_identity(value, "oe:player:") is None for value in player_group["playerid"])
        ) or (
            len(player_group) == 10
            and any(_identity(value, "oe:team:") is None for value in player_group["teamid"])
        ) or (
            len(team_group) == 2
            and any(_identity(value, "oe:team:") is None for value in team_group["teamid"])
        )
        if not target_has_missing_identity:
            # Existing eligible maps are not candidates.  Their historical
            # rows remain available to the strict-prior index above.
            continue
        if target_date is not None and len(player_group) == 10 and len(team_group) == 2:
            missing_identity = False
            for _, row in player_group.iterrows():
                side = _side(row.get("side"))
                role = _role(row.get("position"))
                current_id = _identity(row.get("playerid"), "oe:player:")
                if current_id is not None:
                    player_assignments.append(
                        {
                            "side": side,
                            "role": role,
                            "player_name": _string(row.get("playername")),
                            "player_id": current_id,
                            "candidate_key_type": "source_row_stable_id",
                            "first_seen_timestamp": _iso(target_date),
                            "evidence": [{"game_id": game_id, "timestamp": _iso(target_date), "outcome_used": False}],
                            "outcome_used": False,
                        }
                    )
                else:
                    missing_identity = True
                    candidate = _candidate(
                        name=_norm(row.get("playername")),
                        team=_team_norm(row.get("teamname")),
                        league=_norm(row.get("league")),
                        role=role,
                        target_date=target_date,
                        history=player_history,
                        kind="player",
                    )
                    if candidate["status"] == "candidate":
                        player_assignments.append(
                            {
                                "side": side,
                                "role": role,
                                "player_name": _string(row.get("playername")),
                                "player_id": candidate["candidate_id"],
                                "candidate_key_type": candidate["candidate_key_type"],
                                "first_seen_timestamp": candidate["first_seen_timestamp"],
                                "evidence": candidate["evidence"],
                                "outcome_used": False,
                            }
                        )
                    else:
                        player_rejections.append(
                            {
                                "side": side,
                                "role": role,
                                "player_name": _string(row.get("playername")),
                                "candidate_key_type": candidate["candidate_key_type"],
                                "candidate_ids": candidate["candidate_ids"],
                                "reason": candidate["reason"],
                                "outcome_used": False,
                            }
                        )
            if missing_identity:
                missing_identity_maps += 1
            for side in SIDES:
                side_player_rows = player_group[player_group["side"].map(_side) == side]
                side_team_ids = {
                    _identity(value, "oe:team:")
                    for value in side_player_rows["teamid"]
                    if _identity(value, "oe:team:") is not None
                }
                team_row = team_group[team_group["side"].map(_side) == side]
                team_row_ids = {
                    _identity(value, "oe:team:")
                    for value in team_row["teamid"]
                    if _identity(value, "oe:team:") is not None
                }
                fixed_ids = side_team_ids | team_row_ids
                if len(fixed_ids) > 1:
                    team_rejections.append({"side": side, "reason": "team_identity_conflict", "candidate_ids": sorted(fixed_ids), "outcome_used": False})
                    continue
                team_name = _string(team_row.iloc[0].get("teamname")) if len(team_row) == 1 else None
                if team_name is None and len(side_player_rows):
                    names = [_string(value) for value in side_player_rows["teamname"]]
                    names = [value for value in names if value is not None]
                    if len(set(_team_norm(value) for value in names)) == 1 and names:
                        team_name = names[0]
                team_candidate: dict[str, Any]
                if fixed_ids:
                    team_id = next(iter(fixed_ids))
                    team_candidate = {
                        "status": "candidate",
                        "candidate_id": team_id,
                        "candidate_key_type": "source_row_stable_id",
                        "first_seen_timestamp": _iso(target_date),
                        "evidence": [{"game_id": game_id, "timestamp": _iso(target_date), "outcome_used": False}],
                    }
                else:
                    team_candidate = _candidate(
                        name=_team_norm(team_name),
                        team=_team_norm(team_name),
                        league=_norm(team_row.iloc[0].get("league")) if len(team_row) else _norm(side_player_rows.iloc[0].get("league")) if len(side_player_rows) else None,
                        role=None,
                        target_date=target_date,
                        history=team_history_index,
                        kind="team",
                    )
                if team_candidate["status"] == "candidate":
                    team_assignments.append(
                        {
                            "side": side,
                            "team_name": team_name,
                            "team_id": team_candidate["candidate_id"],
                            "candidate_key_type": team_candidate["candidate_key_type"],
                            "first_seen_timestamp": team_candidate["first_seen_timestamp"],
                            "evidence": team_candidate["evidence"],
                            "outcome_used": False,
                        }
                    )
                else:
                    team_rejections.append(
                        {
                            "side": side,
                            "team_name": team_name,
                            "candidate_key_type": team_candidate.get("candidate_key_type"),
                            "candidate_ids": team_candidate.get("candidate_ids", []),
                            "reason": team_candidate["reason"],
                            "outcome_used": False,
                        }
                    )
        if len(player_assignments) == 10:
            player_ids_for_map = [row["player_id"] for row in player_assignments]
            if len(set(player_ids_for_map)) != 10:
                reasons.append("player_identity_not_unique")
        if len(team_assignments) == 2:
            team_ids_for_map = [row["team_id"] for row in team_assignments]
            if len(set(team_ids_for_map)) != 2:
                reasons.append("team_identity_not_unique")
            player_team_by_side = {
                _side(row.get("side")): _identity(row.get("teamid"), "oe:team:")
                for _, row in player_group.iterrows()
            }
            for assignment in team_assignments:
                side = assignment["side"]
                player_side_ids = {
                    value
                    for _, row in player_group[player_group["side"].map(_side) == side].iterrows()
                    if (value := _identity(row.get("teamid"), "oe:team:")) is not None
                }
                if player_side_ids and player_side_ids != {assignment["team_id"]}:
                    reasons.append("player_team_row_identity_mismatch")
            team_id_by_side = {row["side"]: row["team_id"] for row in team_assignments}
            for assignment in player_assignments:
                assignment["team_id"] = team_id_by_side.get(assignment["side"])
        if player_rejections:
            reasons.append("player_identity_crosswalk_unresolved")
            reasons.extend(str(row["reason"]) for row in player_rejections)
        if team_rejections:
            reasons.append("team_identity_crosswalk_unresolved")
            reasons.extend(str(row["reason"]) for row in team_rejections if row.get("reason"))
        reasons = sorted(set(reasons))
        if reasons or len(player_assignments) != 10 or len(team_assignments) != 2:
            for reason in reasons or ["identity_crosswalk_unresolved"]:
                reason_counts[reason] += 1
            if missing_identity or player_rejections or team_rejections:
                rejected.append(
                    {
                        "game_id": game_id,
                        "target_timestamp": _iso(target_date) if target_date else None,
                        "reasons": reasons or ["identity_crosswalk_unresolved"],
                        "player_rejections": player_rejections,
                        "team_rejections": team_rejections,
                        "outcome_used": False,
                    }
                )
            continue
        # Emit only maps that required a missing identity.  Existing eligible
        # maps are not candidates and remain outside this artifact's scope.
        target_player_rows = player_group["playerid"].map(lambda value: _identity(value, "oe:player:"))
        target_team_rows = team_group["teamid"].map(lambda value: _identity(value, "oe:team:"))
        if target_player_rows.isna().any() or target_team_rows.isna().any():
            assignments.append(
                {
                    "game_id": game_id,
                    "target_timestamp": _iso(target_date),
                    "player_assignments": sorted(player_assignments, key=lambda row: (row["side"], row["role"])),
                    "team_assignments": sorted(team_assignments, key=lambda row: row["side"]),
                    "outcome_used": False,
                }
            )
    counts = {
        "accepted_game_count": len(accepted_ids),
        "candidate_game_count": len(assignments),
        "rejected_game_count": len(rejected),
        "candidate_player_assignment_count": sum(
            sum(1 for row in assignment["player_assignments"] if row["candidate_key_type"] != "source_row_stable_id")
            for assignment in assignments
        ),
        "candidate_team_assignment_count": sum(
            sum(1 for row in assignment["team_assignments"] if row["candidate_key_type"] != "source_row_stable_id")
            for assignment in assignments
        ),
        "unresolved_reason_counts": dict(sorted(reason_counts.items())),
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "authority": dict(AUTHORITY),
        "source_as_of": _iso(source_as_of),
        "source_game_count": len(accepted_ids),
        "source_identity_sha256": source_identity,
        "source_receipt_sha256": source_receipt_hash,
        "source_receipt_file": receipt_file,
        "accepted_game_ids": list(accepted_ids),
        "source_file_records": records,
        "candidate_key_types": [
            "source_row_stable_id",
            "player_name+team_name+league",
            "player_name+team_name",
            "player_name+league",
            "player_name+role",
            "team_name+league",
            "team_name",
        ],
        "method": {
            "timestamp_rule": "strictly_prior",
            "target_timestamp_field": "date",
            "accepted_identity_prefixes": ["oe:player:", "oe:team:"],
            "anonymous_ids_allowed": False,
            "outcome_used": False,
            "model_eligibility_changed": False,
        },
        "counts": counts,
        "assignments": assignments,
        "rejected": rejected,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    if _verify_replay:
        verify_identity_crosswalk(
            payload,
            source_receipt=source_receipt,
            source_receipt_file_record=receipt_file,
            trusted_source_receipt_file_sha256=receipt_file["sha256"],
            source_file_records=records,
        )
    return payload


def build_identity_crosswalk(
    *,
    maps: pd.DataFrame | Sequence[Mapping[str, Any]],
    players: pd.DataFrame | Sequence[Mapping[str, Any]],
    teams: pd.DataFrame | Sequence[Mapping[str, Any]],
    source_receipt: Mapping[str, Any],
    source_receipt_file_record: Mapping[str, Any] | None = None,
    trusted_source_receipt_file_sha256: str | None = None,
    source_file_records: Mapping[str, Mapping[str, Any]] | None = None,
    source_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build and replay-verify a frozen-source identity crosswalk."""

    return _build_identity_crosswalk(
        maps=maps,
        players=players,
        teams=teams,
        source_receipt=source_receipt,
        source_receipt_file_record=source_receipt_file_record,
        trusted_source_receipt_file_sha256=trusted_source_receipt_file_sha256,
        source_file_records=source_file_records,
        source_records=source_records,
        _verify_replay=True,
    )


def _verify_evidence(
    rows: Sequence[Any],
    *,
    target_timestamp: datetime,
    accepted: set[str],
    label: str,
    allow_target: bool = False,
) -> None:
    for row in rows:
        if not isinstance(row, Mapping):
            raise IdentityCrosswalkError(f"{label} evidence is invalid")
        game_id = _string(row.get("game_id"))
        timestamp = _timestamp(row.get("timestamp"), label=f"{label} evidence")
        if game_id not in accepted or timestamp > target_timestamp or (
            timestamp == target_timestamp and not allow_target
        ):
            raise IdentityCrosswalkError(f"{label} evidence is not strict prior")
        if row.get("outcome_used") is not False:
            raise IdentityCrosswalkError(f"{label} evidence uses an outcome")


def verify_identity_crosswalk(
    crosswalk: Mapping[str, Any],
    *,
    source_receipt: Mapping[str, Any],
    source_receipt_file_record: Mapping[str, Any] | None = None,
    trusted_source_receipt_file_sha256: str | None = None,
    source_file_records: Mapping[str, Mapping[str, Any]] | None = None,
    source_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Verify a crosswalk receipt and its source bindings.

    The verifier checks the canonical receipt, census identity, file hashes,
    strict-prior timestamps, identity prefixes, and outcome-free fields.
    """

    if not isinstance(crosswalk, Mapping):
        raise IdentityCrosswalkError("identity crosswalk is required")
    source_receipt, receipt_file = _bound_source_receipt(
        source_receipt,
        source_receipt_file_record,
        trusted_source_receipt_file_sha256,
    )
    accepted, source_as_of, source_identity, source_receipt_hash = _source_receipt(source_receipt)
    records = _source_records(source_receipt, source_file_records or source_records)
    if crosswalk.get("schema_version") != SCHEMA_VERSION or crosswalk.get("status") != STATUS:
        raise IdentityCrosswalkError("identity crosswalk schema or status is invalid")
    if not _outcome_free(crosswalk):
        raise IdentityCrosswalkError("identity crosswalk contains outcome fields")
    _closed_crosswalk_schema(crosswalk)
    if dict(crosswalk.get("authority") or {}) != AUTHORITY:
        raise IdentityCrosswalkError("identity crosswalk authority is invalid")
    claimed = _hash(crosswalk.get("receipt_sha256"))
    if claimed is None:
        raise IdentityCrosswalkError("identity crosswalk hash is invalid")
    body = dict(crosswalk)
    body.pop("receipt_sha256", None)
    if canonical_sha256(body) != claimed:
        raise IdentityCrosswalkError("identity crosswalk hash does not match payload")
    if (
        tuple(crosswalk.get("accepted_game_ids") or ()) != accepted
        or crosswalk.get("source_game_count") != len(accepted)
        or crosswalk.get("source_identity_sha256") != source_identity
        or crosswalk.get("source_receipt_sha256") != source_receipt_hash
        or dict(crosswalk.get("source_receipt_file") or {}) != receipt_file
        or _timestamp(crosswalk.get("source_as_of"), label="crosswalk source_as_of") != source_as_of
    ):
        raise IdentityCrosswalkError("identity crosswalk source binding is invalid")
    if dict(crosswalk.get("source_file_records") or {}) != records:
        raise IdentityCrosswalkError("identity crosswalk source file binding is invalid")
    assignments = crosswalk.get("assignments")
    rejected = crosswalk.get("rejected")
    if not isinstance(assignments, Sequence) or isinstance(assignments, (str, bytes, bytearray)):
        raise IdentityCrosswalkError("identity crosswalk assignments are invalid")
    if not isinstance(rejected, Sequence) or isinstance(rejected, (str, bytes, bytearray)):
        raise IdentityCrosswalkError("identity crosswalk rejected rows are invalid")
    seen_games: set[str] = set()
    for assignment in assignments:
        if not isinstance(assignment, Mapping):
            raise IdentityCrosswalkError("identity assignment is invalid")
        game_id = _string(assignment.get("game_id"))
        target = _timestamp(assignment.get("target_timestamp"), label="assignment target")
        if game_id not in accepted or game_id in seen_games or target > source_as_of:
            raise IdentityCrosswalkError("identity assignment binding is invalid")
        seen_games.add(game_id)
        player_rows = assignment.get("player_assignments")
        team_rows = assignment.get("team_assignments")
        if not isinstance(player_rows, Sequence) or isinstance(player_rows, (str, bytes, bytearray)) or len(player_rows) != 10:
            raise IdentityCrosswalkError("identity assignment player closure is invalid")
        if not isinstance(team_rows, Sequence) or isinstance(team_rows, (str, bytes, bytearray)) or len(team_rows) != 2:
            raise IdentityCrosswalkError("identity assignment team closure is invalid")
        slots = set()
        player_ids: list[str] = []
        player_team_by_side: dict[str, str] = {}
        for row in player_rows:
            if not isinstance(row, Mapping) or row.get("outcome_used") is not False:
                raise IdentityCrosswalkError("identity player assignment is invalid")
            slot = (_side(row.get("side")), _role(row.get("role")))
            if slot in slots or slot[0] not in SIDES or slot[1] not in ROLES:
                raise IdentityCrosswalkError("identity player role or side closure is invalid")
            slots.add(slot)
            player_id = _identity(row.get("player_id"), "oe:player:")
            if player_id is None or player_id in player_ids:
                raise IdentityCrosswalkError("identity player ID is invalid or duplicated")
            player_ids.append(player_id)
            team_id = _identity(row.get("team_id"), "oe:team:")
            if team_id is None:
                raise IdentityCrosswalkError("identity player team binding is invalid")
            if slot[0] in player_team_by_side and player_team_by_side[slot[0]] != team_id:
                raise IdentityCrosswalkError("identity player team binding is inconsistent")
            player_team_by_side[slot[0]] = team_id
            evidence = row.get("evidence")
            if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes, bytearray)):
                raise IdentityCrosswalkError("identity player evidence is invalid")
            _verify_evidence(
                evidence,
                target_timestamp=target,
                accepted=set(accepted),
                label="player",
                allow_target=row.get("candidate_key_type") == "source_row_stable_id",
            )
            first_seen = _timestamp(row.get("first_seen_timestamp"), label="player first_seen")
            if evidence and first_seen != min(_timestamp(item["timestamp"], label="player evidence") for item in evidence):
                raise IdentityCrosswalkError("player first-seen timestamp is invalid")
            if row.get("candidate_key_type") == "anonymous_id":
                raise IdentityCrosswalkError("anonymous player IDs are not allowed")
        if slots != {(side, role) for side in SIDES for role in ROLES}:
            raise IdentityCrosswalkError("identity player slot closure is incomplete")
        team_ids: list[str] = []
        team_sides: set[str] = set()
        for row in team_rows:
            if not isinstance(row, Mapping) or row.get("outcome_used") is not False:
                raise IdentityCrosswalkError("identity team assignment is invalid")
            side = _side(row.get("side"))
            if side is None or side in team_sides:
                raise IdentityCrosswalkError("identity team side closure is invalid")
            team_sides.add(side)
            team_id = _identity(row.get("team_id"), "oe:team:")
            if team_id is None or team_id in team_ids:
                raise IdentityCrosswalkError("identity team ID is invalid or duplicated")
            team_ids.append(team_id)
            evidence = row.get("evidence")
            if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes, bytearray)):
                raise IdentityCrosswalkError("identity team evidence is invalid")
            if row.get("candidate_key_type") == "anonymous_id":
                raise IdentityCrosswalkError("anonymous team IDs are not allowed")
            _verify_evidence(
                evidence,
                target_timestamp=target,
                accepted=set(accepted),
                label="team",
                allow_target=row.get("candidate_key_type") == "source_row_stable_id",
            )
            first_seen = _timestamp(row.get("first_seen_timestamp"), label="team first_seen")
            if evidence and first_seen != min(_timestamp(item["timestamp"], label="team evidence") for item in evidence):
                raise IdentityCrosswalkError("team first-seen timestamp is invalid")
        team_id_by_side = {row["side"]: row["team_id"] for row in team_rows}
        if player_team_by_side != team_id_by_side:
            raise IdentityCrosswalkError("identity player team binding does not match team assignment")
    for row in rejected:
        if not isinstance(row, Mapping) or row.get("outcome_used") is not False:
            raise IdentityCrosswalkError("identity rejection is invalid")
        game_id = _string(row.get("game_id"))
        if game_id not in accepted or game_id in seen_games:
            raise IdentityCrosswalkError("identity rejection binding is invalid")
        seen_games.add(game_id)
        if not isinstance(row.get("reasons"), Sequence) or not row.get("reasons"):
            raise IdentityCrosswalkError("identity rejection reason is missing")
    if set(seen_games) - set(accepted):
        raise IdentityCrosswalkError("identity crosswalk contains unknown maps")
    expected = _build_identity_crosswalk(
        maps=_load_source_frame(records["maps"], "maps"),
        players=_load_source_frame(records["players"], "players"),
        teams=_load_source_frame(records["teams"], "teams"),
        source_receipt=source_receipt,
        source_receipt_file_record=receipt_file,
        trusted_source_receipt_file_sha256=receipt_file["sha256"],
        source_file_records=records,
        _verify_replay=False,
    )
    if dict(crosswalk) != expected:
        raise IdentityCrosswalkError(
            "identity crosswalk does not match verified source-row replay"
        )


# Descriptive aliases keep callers explicit while allowing the module to be
# discovered under either the crosswalk or candidate-builder terminology.
build_identity_candidate_crosswalk = build_identity_crosswalk
verify_identity_candidate_crosswalk = verify_identity_crosswalk


__all__ = [
    "AUTHORITY",
    "IdentityCrosswalkError",
    "SCHEMA_VERSION",
    "STATUS",
    "build_identity_candidate_crosswalk",
    "build_identity_crosswalk",
    "canonical_sha256",
    "verify_identity_candidate_crosswalk",
    "verify_identity_crosswalk",
]
