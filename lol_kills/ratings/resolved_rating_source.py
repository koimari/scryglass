"""Build deterministic, exact-roster rating source receipts.

The RF research harness consumes four fields on each map row:

``rating_source_available``
    Explicit source availability.  A neutral numeric value is not evidence.
``rating_source_sha256``
    Digest of the immutable rating source artifact.
``rating_roster_sha256``
    Digest of the exact map roster and its source identity.
``rating_receipt_sha256``
    Digest that binds the source digest and roster digest together.

This module only joins already-produced pre-game rating values to an exact
roster.  It does not fit a rating model.  Invalid map inputs return an
unavailable receipt through :func:`build_resolved_rating_source`; callers that
need the validation error can pass ``strict=True`` or use
:func:`build_roster_sha256` directly.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


SCHEMA_VERSION = "scryglass:resolved-rating-source:v1"
ROSTER_HASH_SCHEMA_VERSION = "scryglass:resolved-rating-roster:v1"
ROLES = ("top", "jungle", "mid", "bot", "support")
SIDES = ("blue", "red")
STABLE_PLAYER_PREFIX = "oe:player:"
STABLE_TEAM_PREFIX = "oe:team:"

# These are the rating fields that the atomized RF consumer reads.  Other
# numeric fields are deliberately ignored.  This keeps the enrichment from
# becoming an accidental current-state feature channel.
RATING_VALUE_FIELDS = (
    "base_team_logit",
    "team_rating_diff_scaled",
    "base_player_logit",
    "player_rating_diff_scaled",
    "player_lineup_complete",
)

RATING_TIME_FIELDS = (
    "rating_as_of",
    "rating_timestamp",
    "as_of",
    "effective_at",
)

OUTCOME_OR_CURRENT_TOKENS = frozenset(
    {
        "actualbluewin",
        "bluewin",
        "current",
        "cs",
        "damage",
        "deaths",
        "gamelength",
        "gold",
        "kills",
        "live",
        "objective",
        "observed",
        "result",
        "target",
        "winner",
        "win",
        "xp",
    }
)


class ResolvedRatingSourceError(ValueError):
    """A rating source or exact roster violates the frozen contract."""


def _canonical_bytes(value: Any) -> bytes:
    """Serialize finite JSON with one stable representation."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ResolvedRatingSourceError("value is not canonical finite JSON") from exc


def canonical_sha256(value: Any) -> str:
    """Return a SHA-256 digest over canonical JSON."""

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Return a SHA-256 digest over raw source bytes."""

    if not isinstance(value, bytes) or not value:
        raise ResolvedRatingSourceError("source artifact bytes must be non-empty")
    return hashlib.sha256(value).hexdigest()


def _source_artifact_sha256(value: Any) -> str:
    if isinstance(value, bytes):
        return sha256_bytes(value)
    if isinstance(value, Path):
        try:
            raw = value.read_bytes()
        except OSError as exc:
            raise ResolvedRatingSourceError("rating source artifact is unavailable") from exc
        return sha256_bytes(raw)
    if isinstance(value, Mapping) or isinstance(value, list):
        return canonical_sha256(value)
    raise ResolvedRatingSourceError(
        "source_artifact must be non-empty bytes, a path, or canonical JSON"
    )


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResolvedRatingSourceError(f"{field} must be a non-empty string")
    return value.strip()


def _game_id(row: Mapping[str, Any] | None, value: Any = None) -> str:
    if value is None and row is not None:
        value = row.get("game_uid") or row.get("game_id") or row.get("gameid")
    return _text(value, "game_id")


def _timestamp(value: Any, field: str) -> tuple[str, datetime]:
    if isinstance(value, pd.Timestamp):
        parsed = value.to_pydatetime()
    elif isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ResolvedRatingSourceError(f"{field} must be RFC-3339") from exc
    else:
        raise ResolvedRatingSourceError(f"{field} must be RFC-3339")
    if parsed.tzinfo is None:
        raise ResolvedRatingSourceError(f"{field} must include a timezone")
    # Keep fractional seconds.  They can distinguish two source observations
    # in the same displayed second and are part of the roster binding.
    utc = parsed.astimezone(timezone.utc)
    return utc.isoformat().replace("+00:00", "Z"), utc


def _map_timestamp(row: Mapping[str, Any]) -> tuple[str, datetime]:
    for field in ("timestamp", "game_timestamp", "date", "event_start", "game_start"):
        if row.get(field) is not None:
            return _timestamp(row[field], f"map.{field}")
    raise ResolvedRatingSourceError("map timestamp is missing")


def _rating_timestamp(row: Mapping[str, Any]) -> tuple[str, datetime]:
    for field in RATING_TIME_FIELDS:
        if row.get(field) is not None:
            return _timestamp(row[field], f"rating.{field}")
    raise ResolvedRatingSourceError(
        "rating source has no as-of timestamp and cannot prove pre-game values"
    )


def _side(value: Any) -> str:
    token = _text(value, "side").casefold()
    if token not in SIDES:
        raise ResolvedRatingSourceError("side must be blue or red")
    return token


_ROLE_ALIASES = {
    "top": "top",
    "toplane": "top",
    "toplaner": "top",
    "jungle": "jungle",
    "junglelane": "jungle",
    "jungler": "jungle",
    "jng": "jungle",
    "jung": "jungle",
    "mid": "mid",
    "midlaner": "mid",
    "bot": "bot",
    "botlaner": "bot",
    "adc": "bot",
    "bottom": "bot",
    "support": "support",
    "sup": "support",
    "utility": "support",
}


def _role(value: Any) -> str:
    token = "".join(ch for ch in _text(value, "role").casefold() if ch.isalnum())
    try:
        return _ROLE_ALIASES[token]
    except KeyError as exc:
        raise ResolvedRatingSourceError(f"role is not canonical: {value!r}") from exc


def _stable_id(value: Any, prefix: str, field: str) -> str:
    token = _text(value, field)
    if not token.startswith(prefix) or len(token) == len(prefix):
        raise ResolvedRatingSourceError(f"{field} must be a stable {prefix[:-1]} ID")
    return token


def _player_id(row: Mapping[str, Any]) -> str:
    value = row.get("player_id") or row.get("playerid") or row.get("player")
    return _stable_id(value, STABLE_PLAYER_PREFIX, "player_id")


def _team_id(row: Mapping[str, Any]) -> str:
    value = row.get("team_id") or row.get("teamid") or row.get("team")
    return _stable_id(value, STABLE_TEAM_PREFIX, "team_id")


def _champion(row: Mapping[str, Any]) -> str:
    value = row.get("champion") or row.get("champion_name")
    return _text(value, "champion").casefold()


def _source_identity(value: Any) -> Any:
    if value is None:
        raise ResolvedRatingSourceError("rating source identity is missing")
    if isinstance(value, str):
        return _text(value, "source_identity")
    if not isinstance(value, Mapping):
        raise ResolvedRatingSourceError("source_identity must be a string or mapping")
    # Force a finite, deterministic representation now.  The returned value is
    # copied through the hash payload and is never mutated by the caller.
    try:
        return json.loads(_canonical_bytes(dict(value)).decode("utf-8"))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ResolvedRatingSourceError("source_identity is not canonical") from exc


def _roster_payload(
    *,
    game_id: str,
    timestamp: str,
    source_identity: Any,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ResolvedRatingSourceError("roster rows must be a sequence")
    if len(rows) != 10:
        raise ResolvedRatingSourceError("roster requires exactly ten player rows")

    normalized: list[dict[str, str]] = []
    seen_players: set[str] = set()
    seen_slots: set[tuple[str, str]] = set()
    seen_teams: dict[str, str] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ResolvedRatingSourceError("roster row must be a mapping")
        side = _side(raw.get("side"))
        role = _role(raw.get("role") or raw.get("position"))
        team_id = _team_id(raw)
        player_id = _player_id(raw)
        champion = _champion(raw)
        slot = (side, role)
        if slot in seen_slots:
            raise ResolvedRatingSourceError("roster contains a duplicate side-role slot")
        if player_id in seen_players:
            raise ResolvedRatingSourceError("roster contains a duplicate player ID")
        prior_team = seen_teams.setdefault(side, team_id)
        if prior_team != team_id:
            raise ResolvedRatingSourceError("one side maps to multiple team IDs")
        seen_players.add(player_id)
        seen_slots.add(slot)
        normalized.append(
            {
                "side": side,
                "role": role,
                "team_id": team_id,
                "player_id": player_id,
                "champion": champion,
            }
        )

    expected_slots = {(side, role) for side in SIDES for role in ROLES}
    if seen_slots != expected_slots:
        raise ResolvedRatingSourceError("roster does not contain one canonical role per side")
    teams = [seen_teams.get(side) for side in SIDES]
    if any(team is None for team in teams) or len(set(teams)) != 2:
        raise ResolvedRatingSourceError("roster requires two distinct team IDs")

    normalized.sort(key=lambda item: (SIDES.index(item["side"]), ROLES.index(item["role"])))
    return {
        "schema_version": ROSTER_HASH_SCHEMA_VERSION,
        "game_id": game_id,
        "timestamp": timestamp,
        "source_identity": source_identity,
        "teams": [
            {"side": side, "team_id": seen_teams[side]}
            for side in SIDES
        ],
        "players": normalized,
    }


def build_roster_payload(
    *,
    game_id: Any,
    timestamp: Any,
    source_identity: Any,
    roster_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the canonical payload used for the exact roster hash."""

    normalized_game_id = _game_id(None, game_id)
    normalized_timestamp, _ = _timestamp(timestamp, "map.timestamp")
    return _roster_payload(
        game_id=normalized_game_id,
        timestamp=normalized_timestamp,
        source_identity=_source_identity(source_identity),
        rows=roster_rows,
    )


def build_roster_sha256(
    *,
    game_id: Any,
    timestamp: Any,
    source_identity: Any,
    roster_rows: Sequence[Mapping[str, Any]],
) -> str:
    """Hash game, source identity, team IDs, and the exact ten-player roster."""

    return canonical_sha256(
        build_roster_payload(
            game_id=game_id,
            timestamp=timestamp,
            source_identity=source_identity,
            roster_rows=roster_rows,
        )
    )


def build_rating_receipt_sha256(
    *,
    rating_source_available: float | int | bool,
    rating_source_sha256: str,
    rating_roster_sha256: str,
) -> str:
    """Hash the exact receipt payload consumed by the RF harness."""

    available = _availability(rating_source_available)
    source = _text(rating_source_sha256, "rating_source_sha256")
    roster = _text(rating_roster_sha256, "rating_roster_sha256")
    if len(source) != 64 or any(ch not in "0123456789abcdef" for ch in source):
        raise ResolvedRatingSourceError("rating_source_sha256 must be lowercase SHA-256")
    if len(roster) != 64 or any(ch not in "0123456789abcdef" for ch in roster):
        raise ResolvedRatingSourceError("rating_roster_sha256 must be lowercase SHA-256")
    return canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "source_available": available,
            "source_sha256": source,
            "roster_sha256": roster,
        }
    )


def _availability(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ResolvedRatingSourceError("availability must be finite 0 or 1") from exc
    if not math.isfinite(number) or number not in (0.0, 1.0):
        raise ResolvedRatingSourceError("availability must be finite 0 or 1")
    return number


def _field_token(value: Any) -> str:
    return "".join(ch for ch in str(value).casefold() if ch.isalnum())


def _contains_outcome_or_current(value: Any, path: str = "rating") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            token = _field_token(key)
            current_prefix = token.startswith(("current", "observed", "live"))
            outcome_token = "outcome" in token or token.endswith("winner")
            if token in OUTCOME_OR_CURRENT_TOKENS or current_prefix or outcome_token:
                return f"{path}.{key}"
            nested = _contains_outcome_or_current(item, f"{path}.{key}")
            if nested:
                return nested
    elif isinstance(value, list):
        for index, item in enumerate(value):
            nested = _contains_outcome_or_current(item, f"{path}[{index}]")
            if nested:
                return nested
    return None


def _rating_values(
    values: Mapping[str, Any],
    *,
    map_timestamp: datetime,
) -> tuple[dict[str, float], str, datetime]:
    if not isinstance(values, Mapping):
        raise ResolvedRatingSourceError("rating values must be a mapping")
    leaked = _contains_outcome_or_current(values)
    if leaked:
        raise ResolvedRatingSourceError(f"current or outcome field is forbidden: {leaked}")
    rating_timestamp, rating_dt = _rating_timestamp(values)
    if rating_dt >= map_timestamp:
        raise ResolvedRatingSourceError("rating values are not strictly pre-game")
    normalized: dict[str, float] = {}
    for field in RATING_VALUE_FIELDS:
        if field not in values:
            continue
        if values[field] is None:
            # Missing numeric ratings remain a valid source receipt.  The
            # explicit availability fields below keep their neutral output
            # from being mistaken for evidence.
            continue
        try:
            number = float(values[field])
        except (TypeError, ValueError) as exc:
            raise ResolvedRatingSourceError(f"rating value is not numeric: {field}") from exc
        if not math.isfinite(number):
            raise ResolvedRatingSourceError(f"rating value is not finite: {field}")
        normalized[field] = number
    return normalized, rating_timestamp, rating_dt


def _empty_rating_values() -> dict[str, float]:
    return {field: 0.0 for field in RATING_VALUE_FIELDS}


def _availability_for_values(values: Mapping[str, float]) -> tuple[float, float]:
    team = float(all(field in values for field in RATING_VALUE_FIELDS[:2]))
    player = float(
        all(field in values for field in RATING_VALUE_FIELDS[2:])
        and values.get("player_lineup_complete") == 1.0
    )
    return team, player


def _unavailable(
    *,
    game_id: str | None,
    timestamp: str | None,
    source_sha256: str | None,
    reason: str,
) -> dict[str, Any]:
    values = _empty_rating_values()
    return {
        "schema_version": SCHEMA_VERSION,
        "rating_receipt_schema": SCHEMA_VERSION,
        "game_uid": game_id,
        "rating_timestamp": timestamp,
        "rating_source_available": 0.0,
        "rating_source_sha256": source_sha256,
        "rating_roster_sha256": None,
        "rating_receipt_sha256": None,
        "rating_source_reason": reason,
        "rating_values_available": 0.0,
        "rating_values_missing": 1.0,
        "team_rating_available": 0.0,
        "team_rating_missing": 1.0,
        "player_rating_available": 0.0,
        "player_rating_missing": 1.0,
        **values,
    }


def _build_strict(
    *,
    game_id: Any,
    game_timestamp: Any,
    source_identity: Any,
    source_artifact: Any,
    roster_rows: Sequence[Mapping[str, Any]],
    rating_values: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_game_id = _game_id(None, game_id)
    map_timestamp, map_dt = _timestamp(game_timestamp, "map.timestamp")
    identity = _source_identity(source_identity)
    source_sha256 = _source_artifact_sha256(source_artifact)
    normalized_values, rating_timestamp, _ = _rating_values(
        rating_values,
        map_timestamp=map_dt,
    )
    roster_payload = _roster_payload(
        game_id=normalized_game_id,
        timestamp=map_timestamp,
        source_identity=identity,
        rows=roster_rows,
    )
    roster_sha256 = canonical_sha256(roster_payload)
    receipt_sha256 = build_rating_receipt_sha256(
        rating_source_available=1.0,
        rating_source_sha256=source_sha256,
        rating_roster_sha256=roster_sha256,
    )
    team_available, player_available = _availability_for_values(normalized_values)
    values = _empty_rating_values()
    values.update(normalized_values)
    values_available = float(team_available or player_available)
    return {
        "schema_version": SCHEMA_VERSION,
        "rating_receipt_schema": SCHEMA_VERSION,
        "game_uid": normalized_game_id,
        "rating_timestamp": rating_timestamp,
        "rating_source_identity": identity,
        "rating_source_available": 1.0,
        "rating_source_sha256": source_sha256,
        "rating_roster_sha256": roster_sha256,
        "rating_receipt_sha256": receipt_sha256,
        "rating_source_reason": "available",
        "rating_values_available": values_available,
        "rating_values_missing": float(not values_available),
        "team_rating_available": team_available,
        "team_rating_missing": float(not team_available),
        "player_rating_available": player_available,
        "player_rating_missing": float(not player_available),
        **values,
    }


def build_resolved_rating_source(
    *,
    game_id: Any,
    game_timestamp: Any,
    source_identity: Any,
    source_artifact: Any,
    roster_rows: Sequence[Mapping[str, Any]],
    rating_values: Mapping[str, Any],
    strict: bool = False,
) -> dict[str, Any]:
    """Build one map receipt, returning explicit unavailable fields on failure."""

    try:
        return _build_strict(
            game_id=game_id,
            game_timestamp=game_timestamp,
            source_identity=source_identity,
            source_artifact=source_artifact,
            roster_rows=roster_rows,
            rating_values=rating_values,
        )
    except ResolvedRatingSourceError as exc:
        if strict:
            raise
        try:
            normalized_game_id = _game_id(None, game_id)
        except ResolvedRatingSourceError:
            normalized_game_id = None
        try:
            normalized_timestamp, _ = _timestamp(game_timestamp, "map.timestamp")
        except ResolvedRatingSourceError:
            normalized_timestamp = None
        try:
            source_sha256 = _source_artifact_sha256(source_artifact)
        except ResolvedRatingSourceError:
            source_sha256 = None
        return _unavailable(
            game_id=normalized_game_id,
            timestamp=normalized_timestamp,
            source_sha256=source_sha256,
            reason=str(exc),
        )


def require_resolved_rating_source(**kwargs: Any) -> dict[str, Any]:
    """Strict convenience wrapper for callers that need validation errors."""

    kwargs["strict"] = True
    return build_resolved_rating_source(**kwargs)


def _records(value: Any, label: str) -> list[dict[str, Any]]:
    if isinstance(value, pd.DataFrame):
        return [dict(row) for row in value.to_dict(orient="records")]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ResolvedRatingSourceError(f"{label} must be a sequence or DataFrame")
    rows: list[dict[str, Any]] = []
    for row in value:
        if not isinstance(row, Mapping):
            raise ResolvedRatingSourceError(f"{label} row must be a mapping")
        rows.append(dict(row))
    return rows


def _group_by_game(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        try:
            key = _game_id(row)
        except ResolvedRatingSourceError as exc:
            raise ResolvedRatingSourceError(f"{label} row has no game ID") from exc
        grouped[key].append(dict(row))
    return grouped


def enrich_rating_frame(
    maps: Any,
    roster_rows: Any,
    rating_rows: Any,
    *,
    source_identity: Any = None,
    source_artifact: Any,
    strict: bool = False,
) -> pd.DataFrame:
    """Attach exact-roster receipts to one deterministic per-map frame.

    ``maps`` supplies map IDs and start timestamps. ``roster_rows`` supplies
    ten player rows per map. ``rating_rows`` supplies one pre-game rating row
    per map. The output is sorted by UTC map timestamp and game ID. Same-time
    maps are independent and cannot update each other.
    """

    map_records = _records(maps, "maps")
    roster_records = _records(roster_rows, "roster_rows")
    rating_records = _records(rating_rows, "rating_rows")
    roster_by_game = _group_by_game(roster_records, "roster")
    rating_by_game = _group_by_game(rating_records, "rating")
    map_ids: list[str] = []
    for row in map_records:
        try:
            map_ids.append(_game_id(row))
        except ResolvedRatingSourceError:
            map_ids.append("")
    duplicate_map_ids = {game for game, count in Counter(map_ids).items() if count > 1}
    source_sha256: str | None
    try:
        source_sha256 = _source_artifact_sha256(source_artifact)
    except ResolvedRatingSourceError:
        source_sha256 = None
    output: list[dict[str, Any]] = []
    for raw_map in map_records:
        try:
            game_id = _game_id(raw_map)
        except ResolvedRatingSourceError:
            game_id = ""
        try:
            timestamp, _ = _map_timestamp(raw_map)
        except ResolvedRatingSourceError:
            timestamp = None
        row = {
            "game_uid": game_id,
            "date": timestamp,
        }
        try:
            if game_id in duplicate_map_ids:
                raise ResolvedRatingSourceError("map game ID is duplicated")
            roster = roster_by_game.get(game_id, [])
            rating = rating_by_game.get(game_id, [])
            if len(roster) != 10:
                raise ResolvedRatingSourceError("map does not have exactly ten roster rows")
            if len(rating) != 1:
                raise ResolvedRatingSourceError("map rating source row is missing or ambiguous")
            resolved_identity = source_identity
            if resolved_identity is None:
                resolved_identity = rating[0].get("source_identity")
            values = dict(rating[0])
            result = _build_strict(
                game_id=game_id,
                game_timestamp=raw_map.get("timestamp")
                or raw_map.get("game_timestamp")
                or raw_map.get("date")
                or raw_map.get("event_start")
                or raw_map.get("game_start"),
                source_identity=resolved_identity,
                source_artifact=source_artifact,
                roster_rows=roster,
                rating_values=values,
            )
        except ResolvedRatingSourceError as exc:
            if strict:
                raise
            result = _unavailable(
                game_id=game_id,
                timestamp=timestamp,
                source_sha256=source_sha256,
                reason=str(exc),
            )
        row.update(result)
        output.append(row)
    output.sort(key=lambda row: (str(row.get("date") or ""), str(row["game_uid"])))
    return pd.DataFrame(output)


def enrich_rating_rows(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Compatibility alias for :func:`enrich_rating_frame`."""

    return enrich_rating_frame(*args, **kwargs)


__all__ = [
    "ROLES",
    "SCHEMA_VERSION",
    "SIDES",
    "ResolvedRatingSourceError",
    "build_rating_receipt_sha256",
    "build_resolved_rating_source",
    "build_roster_payload",
    "build_roster_sha256",
    "canonical_sha256",
    "enrich_rating_frame",
    "enrich_rating_rows",
    "require_resolved_rating_source",
    "sha256_bytes",
]
