"""Schema, constants, and canonical validation for the LCC atom bridge."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

BRIDGE_SCHEMA_ID = "scryglass.lcc-atom-bridge.v1"
BRIDGE_VERSION = "lcc-atom-bridge-v1"
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
CHAMPION_ID_RE = re.compile(r"^riot:champion:[0-9]+$")

# LCC atom families that describe champion mechanics (the bridge scope).
# "interaction" is a cross-cutting family in LCC; the rest are the six
# champion-mechanic families of the LCC classifier v3/v4.
CHAMPION_ATOM_FAMILIES: tuple[str, ...] = (
    "crowd-control-mobility",
    "damage",
    "heal-shield",
    "interaction",
    "stack-transform-summon-resource",
    "vision-economy",
)

# Ontology dimensions and labels (must match lol_kills/v2/champions/schema.py).
DIMENSION_LABELS: Mapping[str, tuple[str, ...]] = {
    "damage_profile": ("poke", "burst", "artillery", "teamfight"),
    "effective_range": ("point_blank", "short", "mid", "long"),
    "engage": ("none", "single", "area", "hard"),
    "peel": ("none", "defensive", "forced", "zone"),
    "mobility": ("low", "moderate", "high"),
    "wave_control": ("trade", "push", "siege", "freeze"),
    "scaling": ("early", "mid", "late", "late_spike"),
    "target_access": ("frontline", "hook", "reposition", "split_push"),
    "durability_frontline": ("squishy", "frontline", "tank", "duelist"),
    "crowd_control": ("none", "slow", "root", "silence", "stun"),
    "sustain": ("none", "self_heal", "shield", "drain"),
    "tempo": ("slow_buildup", "early_spread", "mid_pressure", "late_reset"),
}
DIMENSIONS: tuple[str, ...] = tuple(DIMENSION_LABELS)


class AtomBridgeError(ValueError):
    """Raised when bridge inputs, artifact, or consumption fail closed."""


def canonical_serialization(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_serialization(payload).encode("utf-8")).hexdigest()


def require_object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AtomBridgeError(f"{label} must be an object")
    return value


def require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise AtomBridgeError(f"{label} must be a non-empty string")
    return value


def require_hash(value: object, label: str) -> str:
    text = require_string(value, label)
    if not HASH_RE.fullmatch(text):
        raise AtomBridgeError(f"{label} must be a lowercase sha256")
    return text


def require_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AtomBridgeError(f"{label} must be numeric")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise AtomBridgeError(f"{label} must be finite")
    return number


def require_exact_fields(value: Mapping[str, Any], required: set[str], allowed: set[str], label: str) -> None:
    missing = required - set(value)
    extra = set(value) - allowed
    if missing:
        raise AtomBridgeError(f"{label} missing required field(s): {', '.join(sorted(missing))}")
    if extra:
        raise AtomBridgeError(f"{label} has unknown field(s): {', '.join(sorted(extra))}")
