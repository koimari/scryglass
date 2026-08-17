"""Canonical schema helpers for the versioned League atom ledger."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Mapping

LEDGER_SCHEMA_ID = "scryglass.league-atom-ledger.v1"
BASE_SCHEMA_ID = "scryglass.league-atom-ledger.base.v1"
DELTA_SCHEMA_ID = "scryglass.league-atom-ledger.delta.v1"
RECEIPT_SCHEMA_ID = "scryglass.league-atom-ledger.replay-receipt.v1"
BASE_AUTHORITY_STATUS = "exact_patch_bound_source_corpus"
PARTIAL_DELTA_AUTHORITY_STATUS = "partial_delta_pilot"

ATOM_CATEGORIES: tuple[str, ...] = (
    "champion",
    "spell",
    "passive",
    "item",
    "rune",
    "objective",
    "buff",
    "debuff",
    "effect",
    "trigger",
    "target",
    "formula",
    "cooldown",
    "cost",
    "range",
    "duration",
    "stack",
    "reset",
    "state-transition",
)

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_PATCH_RE = re.compile(r"^(\d+)\.(\d+)$")


class AtomLedgerError(ValueError):
    """Base class for fail-closed ledger errors."""


class AtomLedgerIntegrityError(AtomLedgerError):
    """A hash, schema, base, or chain binding is invalid."""


class AtomLedgerConflictError(AtomLedgerError):
    """Two delta claims conflict or one event was applied twice."""


class AtomLedgerFutureDataError(AtomLedgerError):
    """A replay asks for evidence that was unavailable at its cutoff."""


class AtomLedgerCoverageError(AtomLedgerError):
    """A consumer asks for model-ready data outside measured coverage."""


def canonical_serialization(payload: object) -> str:
    """Return the one ledger JSON representation used for all hashes."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_sha256(payload: object) -> str:
    """Hash one JSON-compatible value with the canonical representation."""

    return hashlib.sha256(canonical_serialization(payload).encode("utf-8")).hexdigest()


def require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise AtomLedgerIntegrityError(f"{label} must be a lowercase SHA-256")
    return value


def parse_patch(value: object, label: str = "patch") -> tuple[int, int]:
    if not isinstance(value, str):
        raise AtomLedgerIntegrityError(f"{label} must use major.minor form")
    match = _PATCH_RE.fullmatch(value)
    if match is None:
        raise AtomLedgerIntegrityError(f"{label} must use major.minor form")
    return int(match.group(1)), int(match.group(2))


def parse_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise AtomLedgerIntegrityError(f"{label} must be a UTC timestamp ending in Z")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00").astimezone(timezone.utc)
    except ValueError as exc:
        raise AtomLedgerIntegrityError(
            f"{label} must be an ISO-8601 timestamp"
        ) from exc


def stable_atom_id(category: str, identity: Mapping[str, object]) -> str:
    """Create an identity hash that stays stable when numeric fields change."""

    if category not in ATOM_CATEGORIES:
        raise AtomLedgerIntegrityError(f"unsupported atom category: {category}")
    required = (
        "domain",
        "entity",
        "source_atom_id",
        "source_locator",
        "source_slot",
    )
    if any(
        not isinstance(identity.get(key), str) or not identity[key] for key in required
    ):
        raise AtomLedgerIntegrityError(
            "stable atom identity has a missing string field"
        )
    digest = canonical_sha256({key: identity[key] for key in required})[:32]
    return f"lolatom:v1:{category}:{digest}"


def signed_hash(payload: Mapping[str, Any], hash_field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(hash_field, None)
    return canonical_sha256(unsigned)


def validate_signed_hash(
    payload: Mapping[str, Any], hash_field: str, label: str
) -> str:
    submitted = require_hash(payload.get(hash_field), f"{label}.{hash_field}")
    expected = signed_hash(payload, hash_field)
    if submitted != expected:
        raise AtomLedgerIntegrityError(f"{label} canonical hash mismatch")
    return submitted
