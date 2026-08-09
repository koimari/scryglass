"""Common helpers for L1 contract-safe timestamp, hash, and role utilities."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable


ROLES = ("top", "jungle", "mid", "bot", "support")
ROLE_ALIASES = {
    "top": "top",
    "jungle": "jungle",
    "jg": "jungle",
    "jng": "jungle",
    "mid": "mid",
    "middle": "mid",
    "midlane": "mid",
    "bot": "bot",
    "bottom": "bot",
    "adc": "bot",
    "attackdamagecarry": "bot",
    "support": "support",
    "sup": "support",
    "supporter": "support",
}


RFC3339_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
)

_RFC3339_LENIENT_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)


class ContractError(ValueError):
    """Base class for typed contract violations."""


class TimestampError(ContractError):
    """Raised when timestamp parsing or ordering is invalid."""


class ForecastError(ContractError):
    """Raised when forecast-style contracts are violated."""


class LeakageError(ForecastError):
    """Raised when historical/forecast leakage is detected."""


class StalenessError(ContractError):
    """Raised when freshness constraints are violated."""


def canonicalize_json(value: Any) -> str:
    """Return canonical JSON for deterministic byte-level hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_json_bytes(value: Any) -> bytes:
    """Return canonical JSON bytes for deterministic hashing."""

    return canonicalize_json(value).encode("utf-8")


def sha256_canonical_object(value: Any) -> str:
    """Deterministic SHA-256 over a canonical JSON object."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(payload: bytes | bytearray | memoryview, *, label: str = "payload") -> str:
    """SHA-256 over exact bytes.

    Uses raw bytes; this must be used for persisted-file or payload hashes.
    """

    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    if isinstance(payload, (bytes, bytearray)):
        return hashlib.sha256(bytes(payload)).hexdigest()
    raise TypeError(f"{label} must be bytes-like")


def sha256_hex(value: Any) -> str:
    """Return deterministic SHA-256 on canonical JSON object representation."""

    if isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError("sha256_hex accepts structured objects, not raw bytes")

    return sha256_canonical_object(value)


def sha256_hex_bytes(payload: bytes | str) -> str:
    """Return deterministic SHA-256 over exact raw bytes."""

    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("sha256_hex_bytes requires bytes-like input")
    return sha256_bytes(payload)


def sha256_canonical_object_hash(value: Any) -> str:
    """Explicit canonical-object hash alias."""

    return sha256_canonical_object(value)


def sha256_raw_bytes_hash(payload: bytes | bytearray | memoryview, *, label: str = "payload") -> str:
    """Explicit raw-bytes hash alias."""

    return sha256_bytes(payload, label=label)


def canonical_id(prefix: str, token: str) -> str:
    """Build a stable namespaced identifier."""

    if not prefix:
        raise ValueError("stable id prefix is required")
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(token).strip().lower())
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    if not normalized:
        raise ValueError("stable id token cannot be empty after normalization")
    return f"scryglass:{prefix}:{normalized}"


def parse_rfc3339(value: str) -> datetime:
    """Parse a UTC RFC-3339 string."""

    if not isinstance(value, str):
        raise TimestampError("timestamp must be an RFC-3339 string")
    candidate = value.strip()
    if not _RFC3339_LENIENT_RE.match(candidate):
        raise TimestampError(f"invalid RFC-3339 timestamp: {value!r}")

    iso = candidate
    if candidate.endswith("Z"):
        iso = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(iso)
    except ValueError as err:
        raise TimestampError(f"invalid RFC-3339 timestamp: {value}") from err

    if parsed.tzinfo is None:
        raise TimestampError(f"timestamp missing timezone: {value}")
    return parsed.astimezone(timezone.utc)


def to_rfc3339(value: datetime) -> str:
    """Serialize a timezone-aware datetime as RFC-3339 UTC with no micros."""

    if value.tzinfo is None:
        raise TimestampError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_optional_rfc3339(value: str | None) -> datetime | None:
    return parse_rfc3339(value) if value is not None else None


def is_rfc3339(value: str) -> bool:
    if not isinstance(value, str):
        return False
    if not RFC3339_RE.match(value):
        return False
    try:
        parse_rfc3339(value)
    except TimestampError:
        return False
    return True


def _ensure_utc(value: datetime, *, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TimestampError(f"{field_name} must be a datetime")
    if value.tzinfo is None:
        raise TimestampError(f"{field_name} must include timezone")
    try:
        value.timestamp()
    except (OverflowError, OSError, ValueError) as err:
        raise TimestampError(f"{field_name} must be a finite timestamp") from err
    return value.astimezone(timezone.utc)


def _coerce_utc(value: datetime | None, *, field_name: str) -> datetime:
    if value is None:
        raise TimestampError(f"{field_name} is required")
    return _ensure_utc(value, field_name=field_name)


def _require_non_empty(value: str | None, field_name: str) -> str:
    if value is None or not str(value).strip():
        raise ContractError(f"{field_name} is required")
    return str(value).strip()


@dataclass(frozen=True)
class ContractTimePoint:
    """Canonical timestamp triple used by multiple L1 registries."""

    source_updated_at: datetime
    observed_at: datetime
    available_at: datetime

    def __post_init__(self) -> None:
        source_updated = _ensure_utc(self.source_updated_at, field_name="source_updated_at")
        observed = _ensure_utc(self.observed_at, field_name="observed_at")
        available = _ensure_utc(self.available_at, field_name="available_at")

        if source_updated > observed:
            raise ForecastError("source_updated_at must be <= observed_at")
        if available > observed:
            raise ForecastError("available_at must be <= observed_at")

        object.__setattr__(self, "source_updated_at", source_updated)
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "available_at", available)

    @property
    def source_updated(self) -> datetime:
        return self.source_updated_at

    @property
    def as_of_order(self) -> tuple[datetime, datetime, datetime]:
        return (
            self.observed_at,
            self.source_updated_at,
            self.available_at,
        )


def canonicalize_role(raw: str) -> str:
    if not isinstance(raw, str):
        raise ContractError(f"invalid role type: {type(raw)!r}")
    canonical = ROLE_ALIASES.get(raw.strip().lower())
    if canonical is None:
        raise ContractError(f"unknown role: {raw!r}")
    return canonical


def normalize_role_set(raw_roles: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(canonicalize_role(role) for role in raw_roles)


def is_role_tuple_valid(roles: tuple[str, ...]) -> bool:
    if len(roles) != len(ROLES):
        return False
    return set(roles) == set(ROLES)


def enforce_forecast_contract(
    event_start: datetime,
    feature_available_at: Iterable[datetime],
    *,
    mode: str,
) -> None:
    """Enforce raw leakage condition for feature availability."""

    event_start = _ensure_utc(event_start, field_name="event_start")
    if mode not in {"forecast", "forecast_simulation", "hindsight"}:
        raise ForecastError(f"unknown mode: {mode!r}")

    for available_at in feature_available_at:
        available_at = _ensure_utc(available_at, field_name="feature available_at")
        if mode in {"forecast", "forecast_simulation"} and not (available_at < event_start):
            raise LeakageError(f"{mode} features require available_at < event_start")
        if mode == "hindsight" and available_at > event_start:
            raise ForecastError("hindsight feature availability should not be after event_start")


def enforce_forecast_mode_contract(
    *,
    mode: str,
    event_start: datetime,
    available_at: datetime,
    as_of: datetime,
) -> None:
    """Guard forecast/hindsight mode semantics including as-of filtering."""

    event_start = _ensure_utc(event_start, field_name="event_start")
    available_at = _ensure_utc(available_at, field_name="available_at")
    as_of = _ensure_utc(as_of, field_name="as_of")

    if mode == "forecast":
        if not (available_at < event_start):
            raise ForecastError("forecast requires availability strictly before event_start")
        if available_at > as_of:
            raise ForecastError("forecast feature availability cannot exceed as_of")
        if as_of >= event_start:
            raise ForecastError("forecast mode requires as_of before event_start")

    elif mode == "forecast_simulation":
        if available_at >= event_start:
            raise LeakageError("forecast_simulation features require available_at < event_start")
        if as_of < event_start:
            raise ForecastError("forecast_simulation requires as_of at or after event_start")

    elif mode == "hindsight":
        if as_of < event_start:
            raise ForecastError("hindsight mode requires as_of at or after event_start")

    else:  # pragma: no cover (defensive)
        raise ForecastError(f"unknown mode: {mode}")


def enforce_hindsight_mode(
    requirement: str,
    *,
    mode: str,
    forecast_id: str | None = None,
) -> None:
    if requirement == "forecast" and mode != "forecast":
        raise ForecastError("forecast requirement violated")
    if requirement == "hindsight" and mode != "hindsight":
        raise ForecastError("hindsight requirement violated")
    if mode == "hindsight" and not forecast_id:
        raise ForecastError("hindsight mode requires related forecast reference")


def check_mode_availability(mode: str, mode_required: str, *, event_start: datetime, as_of: datetime) -> None:
    """Enforce mode compatibility and as-of ordering against event boundaries."""

    event_start = _ensure_utc(event_start, field_name="event_start")
    as_of = _ensure_utc(as_of, field_name="as_of")

    if mode_required not in {"forecast", "hindsight", "forecast_simulation"}:
        raise ForecastError(f"unknown required mode: {mode_required!r}")
    if mode != mode_required:
        raise ForecastError(f"mode mismatch: expected {mode_required}, got {mode}")
    if mode == "forecast" and not (as_of < event_start):
        raise LeakageError("forecast mode requires as_of before event_start")
    if mode == "forecast_simulation" and as_of < event_start:
        raise LeakageError("forecast_simulation mode requires as_of >= event_start")


def enforce_staleness(source_updated_at: datetime, as_of: datetime, *, limit_seconds: int) -> bool:
    source_updated_at = _ensure_utc(source_updated_at, field_name="source_updated_at")
    as_of = _ensure_utc(as_of, field_name="as_of")
    age_seconds = int((as_of - source_updated_at).total_seconds())
    if age_seconds < 0:
        raise StalenessError("as_of cannot be before source_updated_at")
    return age_seconds <= limit_seconds


def enforce_non_negative_age(reference: datetime, target: datetime, *, field_name: str) -> None:
    """Reject future references passed as past expectations."""

    reference_dt = _ensure_utc(reference, field_name=f"{field_name}_reference")
    target_dt = _ensure_utc(target, field_name=field_name)
    if target_dt > reference_dt:
        raise TimestampError(f"{field_name} cannot be in the future relative to reference")


def validate_snapshot_availability(
    *,
    max_available_at: datetime,
    as_of: datetime,
    limit_seconds: int | None = None,
) -> None:
    """Assert a snapshot is valid for a specific as-of instant."""

    max_available_at = _ensure_utc(max_available_at, field_name="max_available_at")
    as_of = _ensure_utc(as_of, field_name="as_of")
    if max_available_at > as_of:
        raise ForecastError("max_available_at exceeds as_of")
    if limit_seconds is not None:
        age_seconds = int((as_of - max_available_at).total_seconds())
        if age_seconds < 0:
            raise StalenessError("as_of cannot be before max_available_at")
        if age_seconds > limit_seconds:
            raise StalenessError("snapshot is stale for this context")


def available_at_or_observed(
    *,
    available_at: datetime | None,
    source_updated_at: datetime | None,
    observed_at: datetime,
) -> datetime:
    observed_at = _ensure_utc(observed_at, field_name="observed_at")
    if available_at is None:
        if source_updated_at is None:
            return observed_at
        return _ensure_utc(source_updated_at, field_name="source_updated_at")
    return _ensure_utc(available_at, field_name="available_at")


def require_rfc3339_times(*values: datetime) -> None:
    for value in values:
        to_rfc3339(value)


def stable_id(namespace: str, token: str) -> str:
    """Alias preserved for existing callers."""

    return canonical_id(namespace, token)


def enforce_no_naive_datetimes(*values: Any, field_label: str) -> tuple[datetime, ...]:
    """Convert values to UTC while rejecting naive datetimes."""

    converted: list[datetime] = []
    for value in values:
        converted.append(_ensure_utc(_coerce_datetime(value, field_name=field_label), field_name=field_label))
    return tuple(converted)


def _coerce_datetime(value: Any, *, field_name: str) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise TimestampError(f"{field_name} must include timezone")
        return _ensure_utc(value, field_name=field_name)
    if isinstance(value, str):
        return parse_rfc3339(value)
    raise TimestampError(f"{field_name} must be datetime or RFC-3339 string")


def validate_time_axis(*, source_updated_at: datetime, observed_at: datetime, available_at: datetime) -> None:
    """Validate universal source-time ordering used across L1 registries."""

    enforce_as_of_order(
        as_of=observed_at,
        source_updated_at=source_updated_at,
        observed_at=observed_at,
        available_at=available_at,
    )


def validate_freshness_window(*, available_at: datetime, as_of: datetime, limit_seconds: int | None) -> str:
    """Return freshness state and enforce freshness rules if configured."""

    available_at = _ensure_utc(available_at, field_name="available_at")
    as_of = _ensure_utc(as_of, field_name="as_of")

    if available_at > as_of:
        raise StalenessError("availability cannot be after as_of")
    if limit_seconds is None:
        return "fresh"

    age_seconds = int((as_of - available_at).total_seconds())
    if age_seconds < 0:
        raise StalenessError("as_of cannot be before available_at")
    return "fresh" if age_seconds <= limit_seconds else "stale"


def enforce_future_guard(reference: datetime, candidate: datetime, *, field_name: str) -> None:
    """Reject candidate timestamps later than the reference instant."""

    reference_dt = _ensure_utc(reference, field_name="reference")
    candidate_dt = _ensure_utc(candidate, field_name=field_name)
    if candidate_dt > reference_dt:
        raise TimestampError(f"{field_name} cannot exceed reference timestamp")


def enforce_mode_non_future(value: datetime, *, field_name: str) -> datetime:
    return enforce_future_guard(
        reference=_coerce_datetime(
            to_rfc3339(datetime.now(timezone.utc)),
            field_name="reference",
        ),
        candidate=value,
        field_name=field_name,
    )


def enforce_as_of_order(
    *,
    as_of: datetime,
    source_updated_at: datetime,
    observed_at: datetime,
    available_at: datetime,
) -> None:
    """Conservative ordering used by cross-module validators."""

    as_of = _ensure_utc(as_of, field_name="as_of")
    source_updated_at = _ensure_utc(source_updated_at, field_name="source_updated_at")
    observed_at = _ensure_utc(observed_at, field_name="observed_at")
    available_at = _ensure_utc(available_at, field_name="available_at")

    if source_updated_at > observed_at:
        raise TimestampError("source_updated_at cannot be later than observed_at")
    if observed_at > as_of:
        raise TimestampError("observed_at cannot be later than as_of")
    if available_at > observed_at:
        raise TimestampError("available_at cannot be later than observed_at")
    if available_at > as_of:
        raise TimestampError("available_at cannot be later than as_of")


def enforce_contract_time_axis(
    *,
    source_updated_at: datetime,
    observed_at: datetime,
    available_at: datetime,
    as_of: datetime | None = None,
) -> None:
    """Shared axis check used by core registry validators."""

    as_of_value = _ensure_utc(_coerce_datetime(as_of, field_name="as_of") if as_of is not None else observed_at, field_name="as_of")
    enforce_as_of_order(
        as_of=as_of_value,
        source_updated_at=source_updated_at,
        observed_at=observed_at,
        available_at=available_at,
    )


def validate_ordered_time_axis(
    *,
    source_updated_at: datetime,
    observed_at: datetime,
    available_at: datetime,
) -> None:
    """Shared axis validation used across registry modules."""

    source_updated_at = _ensure_utc(source_updated_at, field_name="source_updated_at")
    observed_at = _ensure_utc(observed_at, field_name="observed_at")
    available_at = _ensure_utc(available_at, field_name="available_at")

    if source_updated_at > observed_at:
        raise TimestampError("source_updated_at cannot be after observed_at")
    if available_at > observed_at:
        raise TimestampError("available_at cannot be after observed_at")


def enforce_utc_range(
    min_value: datetime,
    value: datetime,
    max_value: datetime,
    *,
    field_name: str,
) -> None:
    min_value_dt = _ensure_utc(min_value, field_name="min_value")
    value_dt = _ensure_utc(value, field_name=field_name)
    max_value_dt = _ensure_utc(max_value, field_name="max_value")
    if not (min_value_dt <= value_dt <= max_value_dt):
        raise TimestampError(f"{field_name} must be within [min_value, max_value]")


def as_timedelta_seconds(value: timedelta) -> int:
    return int(value.total_seconds())


def enforce_row_ids_present(rows: Iterable[Any], *, id_fields: tuple[str, ...], source: str) -> None:
    for row in rows:
        for field in id_fields:
            if not getattr(row, field, None):
                raise ContractError(f"{source} row {getattr(row, 'row_id', '<unknown>')} missing {field}")


def ensure_ascii_sorted_tuple(values: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values if str(value).strip())
    return tuple(sorted(normalized))


def ensure_sorted_unique(values: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys((str(value).strip() for value in values if str(value).strip())))
    return tuple(sorted(normalized))


__all__ = [
    "ContractError",
    "TimestampError",
    "ForecastError",
    "LeakageError",
    "StalenessError",
    "canonical_id",
    "canonicalize_json",
    "canonical_json_bytes",
    "sha256_hex",
    "sha256_hex_bytes",
    "parse_rfc3339",
    "parse_optional_rfc3339",
    "to_rfc3339",
    "is_rfc3339",
    "ContractTimePoint",
    "require_rfc3339_times",
    "ROLES",
    "ROLE_ALIASES",
    "canonicalize_role",
    "normalize_role_set",
    "is_role_tuple_valid",
    "enforce_forecast_contract",
    "enforce_forecast_mode_contract",
    "available_at_or_observed",
    "enforce_hindsight_mode",
    "check_mode_availability",
    "enforce_staleness",
    "validate_snapshot_availability",
    "stable_id",
    "enforce_no_naive_datetimes",
    "enforce_future_guard",
    "enforce_as_of_order",
    "enforce_contract_time_axis",
    "enforce_mode_non_future",
    "enforce_utc_range",
    "as_timedelta_seconds",
    "enforce_row_ids_present",
    "sha256_canonical_object_hash",
    "sha256_raw_bytes_hash",
    "ensure_ascii_sorted_tuple",
    "ensure_sorted_unique",
]
