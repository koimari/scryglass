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
import os
import re
import stat
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd


SCHEMA_VERSION = "scryglass:resolved-rating-source:v1"
ROSTER_HASH_SCHEMA_VERSION = "scryglass:resolved-rating-roster:v1"
BATCH_RECEIPT_SCHEMA_VERSION = "scryglass:resolved-rating-batch:v1"
EQUAL_TIMESTAMP_BATCHING_POLICY = "same-utc-timestamp-independent-map-v1"
ROLES = ("top", "jungle", "mid", "bot", "support")
SIDES = ("blue", "red")
STABLE_PLAYER_PREFIX = "oe:player:"
STABLE_TEAM_PREFIX = "oe:team:"

# The producer accepts in-memory fixtures as well as paths to immutable
# source artifacts.  These bounds stop a malformed receipt input from
# turning hashing and recursive validation into an unbounded operation.
MAX_SOURCE_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_CANONICAL_BYTES = 16 * 1024 * 1024
MAX_SCALAR_UTF8_BYTES = 64 * 1024
MAX_INPUT_BATCH_BYTES = 16 * 1024 * 1024
MAX_INPUT_BATCH_ITEMS = 200_000
MAX_NESTING = 32
MAX_ITEMS = 100_000
MAX_INPUT_ROWS = 100_000
MAX_BATCH_MAPS = 100_000
SHA256_CHUNK_BYTES = 1024 * 1024

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

RATING_APPROVED_INPUT_FIELDS = frozenset(
    {
        *RATING_VALUE_FIELDS,
        *RATING_TIME_FIELDS,
        "game_uid",
        "game_id",
        "gameid",
        "source_identity",
        "rating_source_identity",
        "rating_source_available",
        "rating_values_available",
        "rating_values_missing",
        "team_rating_available",
        "team_rating_missing",
        "player_rating_available",
        "player_rating_missing",
        "rating_source_sha256",
        "rating_roster_sha256",
        "rating_receipt_sha256",
        "rating_receipt_schema",
        "rating_source_reason",
        "rating_batch_timestamp",
        "rating_batching_policy",
        "rating_batch_receipt_sha256",
    }
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
        "winnerteamid",
        "win",
        "xp",
        "ybluewin",
        "yredwin",
        "mapresultflag",
    }
)


class ResolvedRatingSourceError(ValueError):
    """A rating source or exact roster violates the frozen contract."""


class _InputBudget:
    """Count input scalars before any canonical JSON serialization."""

    def __init__(self) -> None:
        self.bytes = 0
        self.items = 0

    def add(self, value: Any, *, path: str = "input", depth: int = 0) -> None:
        if depth > MAX_NESTING:
            raise ResolvedRatingSourceError(
                f"{path} exceeds the maximum nesting depth of {MAX_NESTING}"
            )
        self.items += 1
        if self.items > MAX_INPUT_BATCH_ITEMS:
            raise ResolvedRatingSourceError(
                f"input batch exceeds the {MAX_INPUT_BATCH_ITEMS}-item budget"
            )
        if isinstance(value, Mapping):
            for key, item in value.items():
                self.add(key, path=f"{path}.<key>", depth=depth + 1)
                self.add(item, path=f"{path}.{key}", depth=depth + 1)
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                self.add(item, path=f"{path}[{index}]", depth=depth + 1)
            return
        if isinstance(value, str):
            try:
                scalar_bytes = len(value.encode("utf-8"))
            except UnicodeEncodeError as exc:
                raise ResolvedRatingSourceError(
                    f"{path} is not valid UTF-8"
                ) from exc
            if scalar_bytes > MAX_SCALAR_UTF8_BYTES:
                raise ResolvedRatingSourceError(
                    f"{path} exceeds the {MAX_SCALAR_UTF8_BYTES}-byte scalar cap"
                )
            self.bytes += scalar_bytes
        elif isinstance(value, bytes):
            if len(value) > MAX_SCALAR_UTF8_BYTES:
                raise ResolvedRatingSourceError(
                    f"{path} exceeds the {MAX_SCALAR_UTF8_BYTES}-byte scalar cap"
                )
            self.bytes += len(value)
        if self.bytes > MAX_INPUT_BATCH_BYTES:
            raise ResolvedRatingSourceError(
                f"input batch exceeds the {MAX_INPUT_BATCH_BYTES}-byte budget"
            )


def _input_budget(*values: Any) -> _InputBudget:
    budget = _InputBudget()
    for index, value in enumerate(values):
        budget.add(value, path=f"input[{index}]")
    return budget


def _guard_shape(
    value: Any,
    *,
    path: str = "value",
    depth: int = 0,
    state: dict[str, Any] | None = None,
    require_json: bool = False,
) -> None:
    """Apply bounded nesting and item checks before recursive processing."""

    if state is None:
        state = {"items": 0, "active": set()}
    if depth > MAX_NESTING:
        raise ResolvedRatingSourceError(
            f"{path} exceeds the maximum nesting depth of {MAX_NESTING}"
        )
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in state["active"]:
            raise ResolvedRatingSourceError(f"{path} contains a recursive value")
        state["active"].add(identity)
        try:
            for key, item in value.items():
                state["items"] += 1
                if state["items"] > MAX_ITEMS:
                    raise ResolvedRatingSourceError(
                        f"{path} exceeds the maximum item count of {MAX_ITEMS}"
                    )
                if require_json and not isinstance(key, str):
                    raise ResolvedRatingSourceError(
                        f"{path} contains a non-string JSON object key"
                    )
                _guard_shape(
                    item,
                    path=f"{path}.{key}",
                    depth=depth + 1,
                    state=state,
                    require_json=require_json,
                )
        finally:
            state["active"].remove(identity)
        return
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in state["active"]:
            raise ResolvedRatingSourceError(f"{path} contains a recursive value")
        state["active"].add(identity)
        try:
            for index, item in enumerate(value):
                state["items"] += 1
                if state["items"] > MAX_ITEMS:
                    raise ResolvedRatingSourceError(
                        f"{path} exceeds the maximum item count of {MAX_ITEMS}"
                    )
                _guard_shape(
                    item,
                    path=f"{path}[{index}]",
                    depth=depth + 1,
                    state=state,
                    require_json=require_json,
                )
        finally:
            state["active"].remove(identity)
        return
    if require_json and not (
        value is None
        or isinstance(value, (str, int, float, bool))
    ):
        raise ResolvedRatingSourceError(f"{path} is not JSON-compatible")


def _canonical_bytes(value: Any) -> bytes:
    """Serialize finite JSON with one stable representation and a byte cap."""

    _input_budget(value)
    _guard_shape(value, require_json=True)
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ResolvedRatingSourceError("value is not canonical finite JSON") from exc
    if len(raw) > MAX_CANONICAL_BYTES:
        raise ResolvedRatingSourceError(
            f"canonical value exceeds {MAX_CANONICAL_BYTES} bytes"
        )
    return raw


def canonical_sha256(value: Any) -> str:
    """Return a SHA-256 digest over canonical JSON."""

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Return a SHA-256 digest over raw source bytes."""

    if not isinstance(value, bytes) or not value:
        raise ResolvedRatingSourceError("source artifact bytes must be non-empty")
    if len(value) > MAX_SOURCE_ARTIFACT_BYTES:
        raise ResolvedRatingSourceError(
            f"source artifact exceeds {MAX_SOURCE_ARTIFACT_BYTES} bytes"
        )
    return hashlib.sha256(value).hexdigest()


def _safe_artifact_path(value: Path, allowed_root: Path | str | None) -> Path:
    if allowed_root is None:
        raise ResolvedRatingSourceError(
            "allowed_root is required for a path source artifact"
        )
    root = Path(allowed_root).absolute()
    path = Path(value)
    if not path.is_absolute():
        path = Path(os.path.abspath(path))
    else:
        path = Path(os.path.abspath(path))
    try:
        root_meta = os.lstat(root)
    except OSError as exc:
        raise ResolvedRatingSourceError("allowed_root is unavailable") from exc
    if stat.S_ISLNK(root_meta.st_mode):
        raise ResolvedRatingSourceError("allowed_root symlink is rejected")
    if not stat.S_ISDIR(root_meta.st_mode):
        raise ResolvedRatingSourceError("allowed_root must be a directory")
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ResolvedRatingSourceError(
            "source artifact is outside allowed_root"
        ) from exc
    current = root
    parts = relative.parts
    if not parts:
        raise ResolvedRatingSourceError("source artifact must be a regular file")
    for index, part in enumerate(parts):
        current = current / part
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise ResolvedRatingSourceError(
                "rating source artifact is unavailable"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ResolvedRatingSourceError("rating source artifact symlink is rejected")
        if index < len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise ResolvedRatingSourceError("source artifact parent is not a directory")
        if index == len(parts) - 1 and not stat.S_ISREG(metadata.st_mode):
            raise ResolvedRatingSourceError(
                "source artifact must be a regular file"
            )
    return path


def _streaming_file_sha256(path: Path, *, allowed_root: Path | None = None) -> str:
    try:
        if allowed_root is None:
            descriptor = os.open(
                str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
        else:
            descriptor = os.open(
                str(allowed_root),
                os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
            )
            parts = path.relative_to(allowed_root).parts
            for part in parts[:-1]:
                next_descriptor = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
                os.close(descriptor)
                descriptor = next_descriptor
            final_descriptor = os.open(
                parts[-1],
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = final_descriptor
    except OSError as exc:
        raise ResolvedRatingSourceError(
            "rating source artifact is unavailable"
        ) from exc
    digest = hashlib.sha256()
    total = 0
    try:
        with os.fdopen(descriptor, "rb") as stream:
            while True:
                chunk = stream.read(SHA256_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_SOURCE_ARTIFACT_BYTES:
                    raise ResolvedRatingSourceError(
                        f"source artifact exceeds {MAX_SOURCE_ARTIFACT_BYTES} bytes"
                    )
                digest.update(chunk)
    except ResolvedRatingSourceError:
        raise
    except OSError as exc:
        raise ResolvedRatingSourceError(
            "rating source artifact is unavailable"
        ) from exc
    if total == 0:
        raise ResolvedRatingSourceError("source artifact bytes must be non-empty")
    return digest.hexdigest()


def _source_artifact_sha256(
    value: Any,
    *,
    allowed_root: Path | str | None = None,
) -> str:
    if isinstance(value, bytes):
        return sha256_bytes(value)
    if isinstance(value, (Path, os.PathLike, str)):
        path = _safe_artifact_path(Path(value), allowed_root)
        return _streaming_file_sha256(
            path,
            allowed_root=Path(allowed_root).absolute() if allowed_root is not None else None,
        )
    if isinstance(value, Mapping) or isinstance(value, list):
        return canonical_sha256(value)
    raise ResolvedRatingSourceError(
        "source_artifact must be non-empty bytes, a path, or canonical JSON"
    )


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResolvedRatingSourceError(f"{field} must be a non-empty string")
    return value.strip()


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, bool) and missing


def _alias_values(
    row: Mapping[str, Any], aliases: Sequence[str]
) -> list[tuple[str, Any]]:
    return [
        (alias, row[alias])
        for alias in aliases
        if alias in row and not _is_missing(row[alias])
    ]


def _resolve_alias(
    row: Mapping[str, Any],
    aliases: Sequence[str],
    field: str,
    *,
    normalize: Callable[[Any], Any] | None = None,
    required: bool = True,
) -> Any:
    present = _alias_values(row, aliases)
    if not present:
        if required:
            raise ResolvedRatingSourceError(f"{field} is missing")
        return None
    normalizer = normalize or (lambda value: value)
    first_name, first_value = present[0]
    first_normalized = normalizer(first_value)
    for alias, value in present[1:]:
        if normalizer(value) != first_normalized:
            raise ResolvedRatingSourceError(
                f"conflicting aliases for {field}: {first_name} and {alias}"
            )
    return first_value


def _game_id(row: Mapping[str, Any] | None, value: Any = None) -> str:
    if value is None and row is not None:
        value = _resolve_alias(
            row,
            ("game_uid", "game_id", "gameid"),
            "game_id",
            normalize=lambda item: _text(item, "game_id"),
        )
    return _text(value, "game_id")


def _timestamp(value: Any, field: str) -> tuple[str, datetime]:
    if isinstance(value, pd.Timestamp):
        if value.nanosecond:
            raise ResolvedRatingSourceError(
                f"{field} must not contain nonzero nanoseconds"
            )
        parsed = value.to_pydatetime()
    elif isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})",
            text,
        ):
            raise ResolvedRatingSourceError(f"{field} must be RFC-3339")
        fraction = re.search(r"\.(\d+)(?=(?:Z|[+-]\d{2}:?\d{2})$)", text)
        if fraction:
            digits = fraction.group(1)
            if len(digits) > 6 and any(char != "0" for char in digits[6:]):
                raise ResolvedRatingSourceError(
                    f"{field} must not contain nonzero nanoseconds"
                )
            if len(digits) > 6:
                text = (
                    text[: fraction.start(1)]
                    + digits[:6]
                    + text[fraction.end(1) :]
                )
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
    aliases = ("timestamp", "game_timestamp", "date", "event_start", "game_start")
    present = _alias_values(row, aliases)
    if not present:
        raise ResolvedRatingSourceError("map timestamp is missing")
    parsed: list[tuple[str, datetime, str]] = []
    for field, value in present:
        normalized, utc = _timestamp(value, f"map.{field}")
        parsed.append((normalized, utc, field))
    first = parsed[0]
    for _, utc, field in parsed[1:]:
        if utc != first[1]:
            raise ResolvedRatingSourceError(
                f"conflicting aliases for map timestamp: {first[2]} and {field}"
            )
    return first[0], first[1]


def _rating_timestamp(row: Mapping[str, Any]) -> tuple[str, datetime]:
    present = _alias_values(row, RATING_TIME_FIELDS)
    if not present:
        raise ResolvedRatingSourceError(
            "rating source has no as-of timestamp and cannot prove pre-game values"
        )
    parsed: list[tuple[str, datetime, str]] = []
    for field, value in present:
        normalized, utc = _timestamp(value, f"rating.{field}")
        parsed.append((normalized, utc, field))
    first = parsed[0]
    for _, utc, field in parsed[1:]:
        if utc != first[1]:
            raise ResolvedRatingSourceError(
                f"conflicting aliases for rating timestamp: {first[2]} and {field}"
            )
    return first[0], first[1]


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


def _role_from_row(row: Mapping[str, Any]) -> str:
    aliases = ("role", "position", "position_name")
    present = _alias_values(row, aliases)
    if not present:
        raise ResolvedRatingSourceError("role is missing")
    normalized: list[tuple[str, str]] = [
        (alias, _role(value)) for alias, value in present
    ]
    first_alias, first_role = normalized[0]
    for alias, role in normalized[1:]:
        if role != first_role:
            raise ResolvedRatingSourceError(
                f"conflicting aliases for role: {first_alias} and {alias}"
            )
    return first_role


def _stable_id(value: Any, prefix: str, field: str) -> str:
    token = _text(value, field)
    if not token.startswith(prefix) or len(token) == len(prefix):
        raise ResolvedRatingSourceError(f"{field} must be a stable {prefix[:-1]} ID")
    return token


def _player_id(row: Mapping[str, Any]) -> str:
    value = _resolve_alias(
        row,
        ("player_id", "playerid", "player"),
        "player_id",
        normalize=lambda item: _stable_id(item, STABLE_PLAYER_PREFIX, "player_id"),
    )
    return _stable_id(value, STABLE_PLAYER_PREFIX, "player_id")


def _team_id(row: Mapping[str, Any]) -> str:
    value = _resolve_alias(
        row,
        ("team_id", "teamid", "team"),
        "team_id",
        normalize=lambda item: _stable_id(item, STABLE_TEAM_PREFIX, "team_id"),
    )
    return _stable_id(value, STABLE_TEAM_PREFIX, "team_id")


def _champion(row: Mapping[str, Any]) -> str:
    value = _resolve_alias(
        row,
        ("champion", "champion_name"),
        "champion",
        normalize=lambda item: _text(item, "champion").casefold(),
    )
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
        return json.loads(_canonical_bytes(value).decode("utf-8"))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ResolvedRatingSourceError("source_identity is not canonical") from exc


def _optional_game_id(row: Mapping[str, Any]) -> str | None:
    if not _alias_values(row, ("game_uid", "game_id", "gameid")):
        return None
    return _game_id(row)


def _roster_payload(
    *,
    game_id: str,
    timestamp: str,
    source_identity: Any,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ResolvedRatingSourceError("roster rows must be a sequence")
    if len(rows) > 10:
        raise ResolvedRatingSourceError("roster exceeds the ten-row cap")
    if len(rows) != 10:
        raise ResolvedRatingSourceError("roster requires exactly ten player rows")

    normalized: list[dict[str, str]] = []
    seen_players: set[str] = set()
    seen_slots: set[tuple[str, str]] = set()
    seen_champions: set[str] = set()
    seen_teams: dict[str, str] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ResolvedRatingSourceError("roster row must be a mapping")
        _guard_shape(raw, path="roster row")
        side = _side(raw.get("side"))
        role = _role_from_row(raw)
        team_id = _team_id(raw)
        player_id = _player_id(raw)
        champion = _champion(raw)
        slot = (side, role)
        if slot in seen_slots:
            raise ResolvedRatingSourceError("roster contains a duplicate side-role slot")
        if player_id in seen_players:
            raise ResolvedRatingSourceError("roster contains a duplicate player ID")
        if champion in seen_champions:
            raise ResolvedRatingSourceError("roster contains a duplicate champion")
        prior_team = seen_teams.setdefault(side, team_id)
        if prior_team != team_id:
            raise ResolvedRatingSourceError("one side maps to multiple team IDs")
        seen_players.add(player_id)
        seen_slots.add(slot)
        seen_champions.add(champion)
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

    _input_budget(game_id, timestamp, source_identity, roster_rows)
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


def _require_sha256(value: Any, field: str) -> str:
    token = _text(value, field)
    if len(token) != 64 or any(ch not in "0123456789abcdef" for ch in token):
        raise ResolvedRatingSourceError(f"{field} must be lowercase SHA-256")
    return token


def _canonical_rating_values(values: Mapping[str, Any] | None) -> dict[str, float | None]:
    if values is None:
        return {field: None for field in RATING_VALUE_FIELDS}
    if not isinstance(values, Mapping):
        raise ResolvedRatingSourceError("rating_values must be a mapping")
    unknown = set(values) - set(RATING_VALUE_FIELDS)
    if unknown:
        raise ResolvedRatingSourceError(
            f"rating value field is not approved: {sorted(map(str, unknown))}"
        )
    _guard_shape(values, path="rating_values")
    normalized: dict[str, float | None] = {}
    for field in RATING_VALUE_FIELDS:
        raw = values.get(field)
        if _is_missing(raw):
            normalized[field] = None
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise ResolvedRatingSourceError(f"rating value is not numeric: {field}") from exc
        if not math.isfinite(number):
            raise ResolvedRatingSourceError(f"rating value is not finite: {field}")
        normalized[field] = number
    return normalized


def build_rating_batch_receipt_sha256(
    *,
    timestamp: Any,
    game_ids: Sequence[Any],
    policy: str = EQUAL_TIMESTAMP_BATCHING_POLICY,
) -> str:
    """Hash the immutable equal-timestamp batch membership and policy."""

    _input_budget(timestamp, game_ids, policy)
    if not isinstance(game_ids, Sequence) or isinstance(game_ids, (str, bytes)):
        raise ResolvedRatingSourceError("batch game IDs must be a sequence")
    if len(game_ids) == 0:
        raise ResolvedRatingSourceError("batch game IDs must be non-empty")
    if len(game_ids) > MAX_BATCH_MAPS:
        raise ResolvedRatingSourceError(
            f"batch exceeds the {MAX_BATCH_MAPS}-map cap"
        )
    normalized_timestamp, _ = _timestamp(timestamp, "batch.timestamp")
    normalized_policy = _text(policy, "batch.policy")
    if normalized_policy != EQUAL_TIMESTAMP_BATCHING_POLICY:
        raise ResolvedRatingSourceError("unsupported equal-timestamp batching policy")
    normalized_ids = [_text(game_id, "batch.game_id") for game_id in game_ids]
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ResolvedRatingSourceError("batch contains a duplicate map ID")
    return canonical_sha256(
        {
            "schema_version": BATCH_RECEIPT_SCHEMA_VERSION,
            "policy": normalized_policy,
            "timestamp": normalized_timestamp,
            "game_ids": sorted(normalized_ids),
        }
    )


def build_rating_receipt_payload(
    *,
    rating_source_available: float | int | bool,
    rating_source_sha256: str,
    rating_roster_sha256: str,
    rating_values: Mapping[str, Any] | None = None,
    rating_timestamp: Any | None = None,
    team_rating_available: float | int | bool | None = None,
    player_rating_available: float | int | bool | None = None,
    rating_values_available: float | int | bool | None = None,
    batching_policy: str = EQUAL_TIMESTAMP_BATCHING_POLICY,
    batch_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Return the canonical payload bound by ``rating_receipt_sha256``."""

    _input_budget(
        rating_source_available,
        rating_source_sha256,
        rating_roster_sha256,
        rating_values,
        rating_timestamp,
        team_rating_available,
        player_rating_available,
        rating_values_available,
        batching_policy,
        batch_receipt_sha256,
    )
    available = _availability(rating_source_available)
    source = _require_sha256(rating_source_sha256, "rating_source_sha256")
    roster = _require_sha256(rating_roster_sha256, "rating_roster_sha256")
    canonical_values = _canonical_rating_values(rating_values)
    if rating_timestamp is None:
        normalized_rating_timestamp = None
    else:
        normalized_rating_timestamp, _ = _timestamp(
            rating_timestamp, "rating.timestamp"
        )
    derived_team, derived_player = _availability_for_values(canonical_values)
    team_available = (
        derived_team
        if team_rating_available is None
        else _availability(team_rating_available)
    )
    player_available = (
        derived_player
        if player_rating_available is None
        else _availability(player_rating_available)
    )
    values_available = (
        float(team_available or player_available)
        if rating_values_available is None
        else _availability(rating_values_available)
    )
    normalized_policy = _text(batching_policy, "batching_policy")
    if normalized_policy != EQUAL_TIMESTAMP_BATCHING_POLICY:
        raise ResolvedRatingSourceError("unsupported equal-timestamp batching policy")
    if batch_receipt_sha256 is not None:
        batch_receipt_sha256 = _require_sha256(
            batch_receipt_sha256, "batch_receipt_sha256"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "source_available": available,
        "source_sha256": source,
        "roster_sha256": roster,
        "rating_timestamp": normalized_rating_timestamp,
        "rating_values": canonical_values,
        "rating_values_available": values_available,
        "team_rating_available": team_available,
        "player_rating_available": player_available,
        "equal_timestamp_batching": {
            "policy": normalized_policy,
            "receipt_sha256": batch_receipt_sha256,
        },
    }


def build_rating_receipt_sha256(
    *,
    rating_source_available: float | int | bool,
    rating_source_sha256: str,
    rating_roster_sha256: str,
    rating_values: Mapping[str, Any] | None = None,
    rating_timestamp: Any | None = None,
    team_rating_available: float | int | bool | None = None,
    player_rating_available: float | int | bool | None = None,
    rating_values_available: float | int | bool | None = None,
    batching_policy: str = EQUAL_TIMESTAMP_BATCHING_POLICY,
    batch_receipt_sha256: str | None = None,
) -> str:
    """Hash source, exact roster, rating values, time, and batch context."""

    return canonical_sha256(
        build_rating_receipt_payload(
            rating_source_available=rating_source_available,
            rating_source_sha256=rating_source_sha256,
            rating_roster_sha256=rating_roster_sha256,
            rating_values=rating_values,
            rating_timestamp=rating_timestamp,
            team_rating_available=team_rating_available,
            player_rating_available=player_rating_available,
            rating_values_available=rating_values_available,
            batching_policy=batching_policy,
            batch_receipt_sha256=batch_receipt_sha256,
        )
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
            outcome_token = (
                "outcome" in token
                or "result" in token
                or "winner" in token
                or token.endswith("win")
                or token.startswith(("ybluewin", "yredwin", "mapresult"))
            )
            if token in OUTCOME_OR_CURRENT_TOKENS or current_prefix or outcome_token:
                return f"{path}.{key}"
            nested = _contains_outcome_or_current(item, f"{path}.{key}")
            if nested:
                return nested
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            nested = _contains_outcome_or_current(item, f"{path}[{index}]")
            if nested:
                return nested
    return None


def _rating_values(
    values: Mapping[str, Any],
    *,
    map_timestamp: datetime,
    expected_game_id: str | None = None,
) -> tuple[dict[str, float | None], str, datetime]:
    if not isinstance(values, Mapping):
        raise ResolvedRatingSourceError("rating values must be a mapping")
    unknown = set(values) - RATING_APPROVED_INPUT_FIELDS
    if unknown:
        raise ResolvedRatingSourceError(
            f"rating row field is not approved: {sorted(map(str, unknown))}"
        )
    for key, value in values.items():
        if key in {"source_identity", "rating_source_identity"}:
            continue
        if isinstance(value, (Mapping, list, tuple)):
            raise ResolvedRatingSourceError(
                f"rating row field must be scalar: {key}"
            )
    _guard_shape(values, path="rating values")
    embedded_game_id = _optional_game_id(values)
    if (
        expected_game_id is not None
        and embedded_game_id is not None
        and embedded_game_id != expected_game_id
    ):
        raise ResolvedRatingSourceError(
            "rating row game ID does not match the requested map"
        )
    rating_timestamp, rating_dt = _rating_timestamp(values)
    if rating_dt >= map_timestamp:
        raise ResolvedRatingSourceError("rating values are not strictly pre-game")
    normalized: dict[str, float | None] = {}
    for field in RATING_VALUE_FIELDS:
        if field not in values:
            normalized[field] = None
            continue
        if _is_missing(values[field]):
            normalized[field] = None
            continue
        try:
            number = float(values[field])
        except (TypeError, ValueError) as exc:
            raise ResolvedRatingSourceError(f"rating value is not numeric: {field}") from exc
        if not math.isfinite(number):
            raise ResolvedRatingSourceError(f"rating value is not finite: {field}")
        normalized[field] = number
    return normalized, rating_timestamp, rating_dt


def _empty_rating_values() -> dict[str, float | None]:
    return {field: None for field in RATING_VALUE_FIELDS}


def _availability_for_values(values: Mapping[str, Any]) -> tuple[float, float]:
    team = float(all(values.get(field) is not None for field in RATING_VALUE_FIELDS[:2]))
    player = float(
        all(values.get(field) is not None for field in RATING_VALUE_FIELDS[2:])
        and values.get("player_lineup_complete") == 1.0
    )
    return team, player


def _unavailable(
    *,
    game_id: str | None,
    timestamp: str | None,
    source_sha256: str | None,
    reason: str,
    batch_timestamp: str | None = None,
    batch_receipt_sha256: str | None = None,
    batching_policy: str = EQUAL_TIMESTAMP_BATCHING_POLICY,
) -> dict[str, Any]:
    values = _empty_rating_values()
    return {
        "schema_version": SCHEMA_VERSION,
        "rating_receipt_schema": SCHEMA_VERSION,
        "game_uid": game_id,
        "rating_timestamp": None,
        "rating_source_available": 0.0,
        "rating_source_sha256": source_sha256,
        "rating_roster_sha256": None,
        "rating_receipt_sha256": None,
        "rating_values": values.copy(),
        "rating_batch_timestamp": batch_timestamp or timestamp,
        "rating_batching_policy": batching_policy,
        "rating_batch_receipt_sha256": batch_receipt_sha256,
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
    allowed_root: Path | str | None = None,
    batching_policy: str = EQUAL_TIMESTAMP_BATCHING_POLICY,
    batch_timestamp: Any | None = None,
    batch_game_ids: Sequence[Any] | None = None,
    batch_receipt_sha256: str | None = None,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    normalized_game_id = _game_id(None, game_id)
    map_timestamp, map_dt = _timestamp(game_timestamp, "map.timestamp")
    identity = _source_identity(source_identity)
    if source_sha256 is None:
        source_sha256 = _source_artifact_sha256(
            source_artifact,
            allowed_root=allowed_root,
        )
    normalized_values, rating_timestamp, _ = _rating_values(
        rating_values,
        map_timestamp=map_dt,
        expected_game_id=normalized_game_id,
    )
    if "rating_source_available" in rating_values and _availability(
        rating_values["rating_source_available"]
    ) == 0.0:
        raise ResolvedRatingSourceError("rating row declares source unavailable")
    if "rating_values_available" in rating_values and _availability(
        rating_values["rating_values_available"]
    ) == 0.0:
        normalized_values = {field: None for field in RATING_VALUE_FIELDS}
    if "team_rating_available" in rating_values and _availability(
        rating_values["team_rating_available"]
    ) == 0.0:
        normalized_values["base_team_logit"] = None
        normalized_values["team_rating_diff_scaled"] = None
    if "player_rating_available" in rating_values and _availability(
        rating_values["player_rating_available"]
    ) == 0.0:
        normalized_values["base_player_logit"] = None
        normalized_values["player_rating_diff_scaled"] = None
        normalized_values["player_lineup_complete"] = None
    if "team_rating_missing" in rating_values and _availability(
        rating_values["team_rating_missing"]
    ) == 1.0:
        normalized_values["base_team_logit"] = None
        normalized_values["team_rating_diff_scaled"] = None
    if "player_rating_missing" in rating_values and _availability(
        rating_values["player_rating_missing"]
    ) == 1.0:
        normalized_values["base_player_logit"] = None
        normalized_values["player_rating_diff_scaled"] = None
        normalized_values["player_lineup_complete"] = None
    roster_payload = _roster_payload(
        game_id=normalized_game_id,
        timestamp=map_timestamp,
        source_identity=identity,
        rows=roster_rows,
    )
    roster_sha256 = canonical_sha256(roster_payload)
    normalized_batch_timestamp = batch_timestamp or map_timestamp
    _, batch_dt = _timestamp(normalized_batch_timestamp, "batch.timestamp")
    if batch_dt != map_dt:
        raise ResolvedRatingSourceError("batch timestamp must equal map timestamp")
    if batch_game_ids is None:
        batch_game_ids = (normalized_game_id,)
    expected_batch_receipt = build_rating_batch_receipt_sha256(
        timestamp=normalized_batch_timestamp,
        game_ids=batch_game_ids,
        policy=batching_policy,
    )
    if batch_receipt_sha256 is None:
        batch_receipt_sha256 = expected_batch_receipt
    elif _require_sha256(batch_receipt_sha256, "batch_receipt_sha256") != expected_batch_receipt:
        raise ResolvedRatingSourceError("equal-timestamp batch receipt does not match batch")
    receipt_sha256 = build_rating_receipt_sha256(
        rating_source_available=1.0,
        rating_source_sha256=source_sha256,
        rating_roster_sha256=roster_sha256,
        rating_values=normalized_values,
        rating_timestamp=rating_timestamp,
        team_rating_available=None,
        player_rating_available=None,
        rating_values_available=None,
        batching_policy=batching_policy,
        batch_receipt_sha256=batch_receipt_sha256,
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
        "rating_values": values.copy(),
        "rating_batch_timestamp": normalized_batch_timestamp,
        "rating_batching_policy": batching_policy,
        "rating_batch_receipt_sha256": batch_receipt_sha256,
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
    allowed_root: Path | str | None = None,
    batching_policy: str = EQUAL_TIMESTAMP_BATCHING_POLICY,
    batch_timestamp: Any | None = None,
    batch_game_ids: Sequence[Any] | None = None,
    batch_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Build one map receipt, returning explicit unavailable fields on failure."""

    try:
        _input_budget(
            game_id,
            game_timestamp,
            source_identity,
            roster_rows,
            rating_values,
            batching_policy,
            batch_timestamp,
            batch_game_ids,
            batch_receipt_sha256,
        )
        return _build_strict(
            game_id=game_id,
            game_timestamp=game_timestamp,
            source_identity=source_identity,
            source_artifact=source_artifact,
            roster_rows=roster_rows,
            rating_values=rating_values,
            allowed_root=allowed_root,
            batching_policy=batching_policy,
            batch_timestamp=batch_timestamp,
            batch_game_ids=batch_game_ids,
            batch_receipt_sha256=batch_receipt_sha256,
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
            source_sha256 = _source_artifact_sha256(
                source_artifact,
                allowed_root=allowed_root,
            )
        except ResolvedRatingSourceError:
            source_sha256 = None
        return _unavailable(
            game_id=normalized_game_id,
            timestamp=normalized_timestamp,
            source_sha256=source_sha256,
            reason=str(exc),
            batch_timestamp=(
                str(batch_timestamp) if batch_timestamp is not None else normalized_timestamp
            ),
            batch_receipt_sha256=batch_receipt_sha256,
            batching_policy=batching_policy,
        )


def require_resolved_rating_source(**kwargs: Any) -> dict[str, Any]:
    """Strict convenience wrapper for callers that need validation errors."""

    kwargs["strict"] = True
    return build_resolved_rating_source(**kwargs)


def _records(
    value: Any,
    label: str,
    *,
    budget: _InputBudget | None = None,
) -> list[dict[str, Any]]:
    if budget is None:
        budget = _InputBudget()
    if isinstance(value, pd.DataFrame):
        if len(value) > MAX_INPUT_ROWS:
            raise ResolvedRatingSourceError(
                f"{label} exceeds the {MAX_INPUT_ROWS}-row cap"
            )
        records = [dict(row) for row in value.to_dict(orient="records")]
        for row in records:
            budget.add(row, path=f"{label} row")
            _guard_shape(row, path=f"{label} row")
        return records
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ResolvedRatingSourceError(f"{label} must be a sequence or DataFrame")
    if len(value) > MAX_INPUT_ROWS:
        raise ResolvedRatingSourceError(
            f"{label} exceeds the {MAX_INPUT_ROWS}-row cap"
        )
    rows: list[dict[str, Any]] = []
    for row in value:
        if not isinstance(row, Mapping):
            raise ResolvedRatingSourceError(f"{label} row must be a mapping")
        copied = dict(row)
        budget.add(copied, path=f"{label} row")
        _guard_shape(copied, path=f"{label} row")
        rows.append(copied)
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
    allowed_root: Path | str | None = None,
    batching_policy: str = EQUAL_TIMESTAMP_BATCHING_POLICY,
) -> pd.DataFrame:
    """Attach exact-roster receipts to one deterministic per-map frame.

    ``maps`` supplies map IDs and start timestamps. ``roster_rows`` supplies
    ten player rows per map. ``rating_rows`` supplies one pre-game rating row
    per map. The output is sorted by UTC map timestamp and game ID. Same-time
    maps are independent and cannot update each other.
    """

    budget = _InputBudget()
    budget.add(source_identity, path="source_identity")
    budget.add(batching_policy, path="batching_policy")
    map_records = _records(maps, "maps", budget=budget)
    roster_records = _records(roster_rows, "roster_rows", budget=budget)
    rating_records = _records(rating_rows, "rating_rows", budget=budget)
    roster_by_game = _group_by_game(roster_records, "roster")
    rating_by_game = _group_by_game(rating_records, "rating")
    map_context: list[tuple[dict[str, Any], str, str | None, str | None]] = []
    map_ids: list[str] = []
    for row in map_records:
        try:
            game_id = _game_id(row)
        except ResolvedRatingSourceError:
            game_id = ""
        try:
            timestamp, _ = _map_timestamp(row)
            timestamp_error = None
        except ResolvedRatingSourceError as exc:
            timestamp = None
            timestamp_error = str(exc)
        map_ids.append(game_id)
        map_context.append((row, game_id, timestamp, timestamp_error))
    duplicate_map_ids = {
        game for game, count in Counter(game for game in map_ids if game).items() if count > 1
    }
    if duplicate_map_ids:
        raise ResolvedRatingSourceError("map batch contains a duplicate map ID")
    batches: dict[str, list[str]] = defaultdict(list)
    for _, game_id, timestamp, _ in map_context:
        if game_id and timestamp is not None:
            batches[timestamp].append(game_id)
    batch_receipts: dict[str, str] = {}
    for timestamp, game_ids in batches.items():
        batch_receipts[timestamp] = build_rating_batch_receipt_sha256(
            timestamp=timestamp,
            game_ids=game_ids,
            policy=batching_policy,
        )
    source_sha256: str | None
    try:
        source_sha256 = _source_artifact_sha256(
            source_artifact,
            allowed_root=allowed_root,
        )
    except ResolvedRatingSourceError:
        source_sha256 = None
    output: list[dict[str, Any]] = []
    for raw_map, game_id, timestamp, timestamp_error in map_context:
        row = dict(raw_map)
        row.update({"game_uid": game_id, "date": timestamp})
        try:
            if not game_id:
                raise ResolvedRatingSourceError("map row has no game ID")
            if timestamp_error is not None:
                raise ResolvedRatingSourceError(timestamp_error)
            if timestamp is None:
                raise ResolvedRatingSourceError("map timestamp is missing")
            roster = roster_by_game.get(game_id, [])
            rating = rating_by_game.get(game_id, [])
            if len(roster) != 10:
                raise ResolvedRatingSourceError("map does not have exactly ten roster rows")
            if len(rating) != 1:
                raise ResolvedRatingSourceError("map rating source row is missing or ambiguous")
            resolved_identity = source_identity
            if resolved_identity is None:
                resolved_identity = _resolve_alias(
                    rating[0],
                    ("source_identity", "rating_source_identity"),
                    "source_identity",
                )
            else:
                embedded_identity = _resolve_alias(
                    rating[0],
                    ("source_identity", "rating_source_identity"),
                    "source_identity",
                    required=False,
                )
                if embedded_identity is not None:
                    if _source_identity(embedded_identity) != _source_identity(source_identity):
                        raise ResolvedRatingSourceError(
                            "conflicting aliases for source_identity"
                        )
            values = dict(rating[0])
            result = _build_strict(
                game_id=game_id,
                game_timestamp=timestamp,
                source_identity=resolved_identity,
                source_artifact=source_artifact,
                roster_rows=roster,
                rating_values=values,
                allowed_root=allowed_root,
                batching_policy=batching_policy,
                batch_timestamp=timestamp,
                batch_game_ids=batches[timestamp],
                batch_receipt_sha256=batch_receipts[timestamp],
                source_sha256=source_sha256,
            )
        except ResolvedRatingSourceError as exc:
            if strict:
                raise
            result = _unavailable(
                game_id=game_id,
                timestamp=timestamp,
                source_sha256=source_sha256,
                reason=str(exc),
                batch_timestamp=timestamp,
                batch_receipt_sha256=(
                    batch_receipts.get(timestamp) if timestamp is not None else None
                ),
                batching_policy=batching_policy,
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
