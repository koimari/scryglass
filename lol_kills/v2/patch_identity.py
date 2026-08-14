"""Keep Riot's public patch and client patch namespaces separate.

Riot's 2026 patch notes use ``26.x``. CommunityDragon and Data Dragon use
the client namespace ``16.x``. A patch receipt carries both values. A bare
string is accepted at ingestion time, then converted to the canonical pair.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class PatchIdentityError(ValueError):
    """Raised when a patch value cannot be mapped without guessing."""


# Keep the current live patch in one place.  Historical fixtures must pass
# their own explicit patch.  Ingestion tools use this value only when the
# operator asks for the current matrix without naming individual patches.
CURRENT_PUBLIC_PATCH = "26.16"
CURRENT_CLIENT_PATCH = "16.16"


_PATCH_RE = re.compile(r"^v?(?P<major>\d{2})\.(?P<minor>\d{1,2})(?:\.\d+)?$")


@dataclass(frozen=True)
class PatchIdentity:
    """Canonical public and client labels for one season patch."""

    public_patch: str
    client_patch: str


def _parse(value: object) -> tuple[int, int]:
    text = str(value or "").strip()
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


__all__ = [
    "CURRENT_CLIENT_PATCH",
    "CURRENT_PUBLIC_PATCH",
    "PatchIdentity",
    "PatchIdentityError",
    "canonical_patch",
    "client_patch",
    "public_patch",
]
