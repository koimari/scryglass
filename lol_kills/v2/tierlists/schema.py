"""Shared schema vocabulary and validation for L9 tier lists.

Regional scope follows AGENTS.md: Europe = LEC only, Americas = LCS only for
the headline scopes; international scopes (MSI, EWC, WORLDS) stay separate.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from lol_kills.v2.data.common import ROLES, canonicalize_role, parse_rfc3339

TIERLIST_SCHEMA_ID = "scryglass.tierlist-artifact.v1"
ARTIFACT_KIND = "tier_list_development"

PATCH_RE = re.compile(r"^\d{1,2}\.\d{1,2}$")
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

# competition_tier values in the OE warehouse and the v2 taxonomy
COMPETITION_TIERS: tuple[str, ...] = ("tier1", "tier2", "tier3")

# international scopes kept separate per event (never merged into one filter)
INTERNATIONAL_SCOPES: tuple[str, ...] = ("MSI", "EWC", "WORLDS")
INTERNATIONAL_EVENT_KINDS: tuple[str, ...] = ("msi", "ewc", "worlds")

# region -> leagues (user-facing region filter).  Europe = LEC only,
# Americas = LCS only; Asia lists the headline circuits; international stays
# per-event.
REGIONS: dict[str, tuple[str, ...]] = {
    "europe": ("LEC",),
    "americas": ("LCS",),
    "asia": ("LCK", "LPL", "PCS", "VCS", "LJL"),
    "international": INTERNATIONAL_SCOPES,
}

ALL_LEAGUES: tuple[str, ...] = (
    "LEC", "LCS", "LCK", "LPL", "PCS", "VCS", "LJL", "CBLOL", "LLA", "LCO",
    "LFL", "PRM", "EM", "NACL", "LDL", "LCKC", "TCL", "AL", "LCP", "LTA",
)

# OE warehouse position vocabulary (abbreviated) -> canonical v2 role
POSITION_ALIASES: dict[str, str] = {
    "top": "top",
    "jng": "jungle",
    "jungle": "jungle",
    "mid": "mid",
    "bot": "bot",
    "sup": "support",
    "support": "support",
}


class TierListError(ValueError):
    """Raised when tier-list inputs or artifacts fail closed."""


def validate_role(role: str) -> str:
    if role not in ROLES:
        raise TierListError(f"unknown role: {role}")
    return role


def validate_patch(patch: str) -> str:
    if not PATCH_RE.fullmatch(patch):
        raise TierListError(f"invalid patch: {patch}")
    return patch


def validate_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TierListError(f"{label} must be a non-empty RFC-3339 string")
    try:
        return parse_rfc3339(value).isoformat().replace("+00:00", "Z").replace("T00:00:00", "T00:00:00")
    except Exception as exc:
        raise TierListError(f"{label} is not RFC-3339 UTC") from exc


def canonical_serialization(payload: Mapping[str, Any]) -> str:
    import json

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TierListError(f"{label} must be a non-empty string")
    return value
