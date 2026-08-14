"""Keep Riot's public patch and client patch namespaces separate.

Riot's 2026 patch notes use ``26.x``. CommunityDragon and Data Dragon use
the client namespace ``16.x``. A patch receipt carries both values. A bare
string is accepted at ingestion time, then converted to the canonical pair.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone


class PatchIdentityError(ValueError):
    """Raised when a patch value cannot be mapped without guessing."""


# Keep the current live patch in one place.  Historical fixtures must pass
# their own explicit patch.  Ingestion tools use this value only when the
# operator asks for the current matrix without naming individual patches.
CURRENT_PUBLIC_PATCH = "26.16"
CURRENT_CLIENT_PATCH = "16.16"

# Oracle's Elixir can retain the prior client token after Riot releases a new
# live patch. Keep that source token in ``oe_patch_token``. Apply a correction
# only when the row also carries realm evidence. Esports tournaments can stay
# on the prior patch after the public client changes.
OE_PATCH_CORRECTION_26_16 = {
    "source_token": "16.15",
    "corrected_token": "16.16",
    "effective_from": "2026-08-12T00:00:00Z",
    "official_release_at": "2026-08-11T18:00:00Z",
    "authority_url": (
        "https://www.leagueoflegends.com/en-us/news/game-updates/"
        "league-of-legends-patch-26-16-notes/"
    ),
    "reason": "OE retained the prior token after the 26.16 live release.",
    "requires_realm_evidence": True,
}


_PATCH_RE = re.compile(r"^v?(?P<major>\d{2})\.(?P<minor>\d{1,2})(?:\.\d+)?$")


def _text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "nat", "none", "<na>"} else text


@dataclass(frozen=True)
class PatchIdentity:
    """Canonical public and client labels for one season patch."""

    public_patch: str
    client_patch: str


def _parse(value: object) -> tuple[int, int]:
    text = _text(value)
    match = _PATCH_RE.fullmatch(text)
    if match is None:
        raise PatchIdentityError(f"malformed patch label: {value!r}")
    major = int(match.group("major"))
    minor = int(match.group("minor"))
    if minor > 99:
        raise PatchIdentityError(f"malformed patch minor: {value!r}")
    return major, minor


def _client_label(major: int, minor: int) -> str:
    return f"{major}.{minor}" if minor < 10 else f"{major}.{minor:02d}"


def canonical_patch(value: object) -> PatchIdentity:
    """Return the exact public/client pair for a 2025+ patch label.

    ``25.x`` maps to the 2025 client namespace ``15.x`` and ``26.x`` maps to
    ``16.x``. Inputs from either namespace are accepted. Build suffixes such
    as ``16.16.1`` are source versions and are dropped from the pair.
    """

    major, minor = _parse(value)
    if major in {25, 26}:
        public_major = major
        client_major = major - 10
    elif major in {15, 16}:
        public_major = major + 10
        client_major = major
    else:
        raise PatchIdentityError(
            f"unsupported patch season {major}; expected 25/26 or 15/16"
        )
    return PatchIdentity(
        public_patch=f"{public_major}.{minor:02d}",
        client_patch=_client_label(client_major, minor),
    )


def public_patch(value: object) -> str:
    """Return the canonical public label, for example ``26.16``."""

    return canonical_patch(value).public_patch


def client_patch(value: object) -> str:
    """Return the exact CommunityDragon/Data Dragon namespace."""

    return canonical_patch(value).client_patch


def corrected_oe_patch_token(
    value: object,
    event_time: object,
    *,
    realm_patch: object | None = None,
    realm_kind: object | None = None,
) -> tuple[str, dict[str, str] | None]:
    """Correct an OE token only with explicit game-realm evidence.

    A date alone cannot identify the game patch. Tournament realms often lag
    the public client. ``realm_patch`` may carry a server-reported patch, or
    ``realm_kind='live'`` may authorize the dated live-client rule. Unknown
    and tournament realms preserve the OE token.
    """

    text = _text(value)
    try:
        token = canonical_patch(text).client_patch
    except PatchIdentityError:
        return text, None
    if token != OE_PATCH_CORRECTION_26_16["source_token"]:
        return token, None
    kind = _text(realm_kind).casefold()
    if kind in {"tournament", "tournament_realm", "esports"}:
        return token, None
    explicit_patch = _text(realm_patch)
    if explicit_patch:
        try:
            if canonical_patch(explicit_patch).client_patch != OE_PATCH_CORRECTION_26_16["corrected_token"]:
                return token, None
        except PatchIdentityError:
            return token, None
    elif kind not in {"live", "ranked", "public"}:
        return token, None
    raw_time = _text(event_time)
    if raw_time.endswith("Z"):
        raw_time = raw_time[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw_time)
    except ValueError:
        return token, None
    if parsed.tzinfo is None:
        return token, None
    instant = parsed.astimezone(timezone.utc)
    boundary = datetime.fromisoformat(
        OE_PATCH_CORRECTION_26_16["effective_from"].replace("Z", "+00:00")
    )
    if instant < boundary:
        return token, None
    return OE_PATCH_CORRECTION_26_16["corrected_token"], dict(
        OE_PATCH_CORRECTION_26_16
    )


__all__ = [
    "CURRENT_CLIENT_PATCH",
    "CURRENT_PUBLIC_PATCH",
    "OE_PATCH_CORRECTION_26_16",
    "PatchIdentity",
    "PatchIdentityError",
    "canonical_patch",
    "client_patch",
    "corrected_oe_patch_token",
    "public_patch",
]
