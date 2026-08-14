"""Small network guards for repository HTTP clients."""

from __future__ import annotations

from collections.abc import Collection
from urllib.parse import urlsplit


class NetworkTargetError(ValueError):
    """Raised when a remote URL is outside the caller's allowlist."""


def require_https_url(
    value: str,
    *,
    hosts: Collection[str],
    allow_subdomains: bool = False,
) -> str:
    """Return one canonical HTTPS URL whose host is explicitly allowed."""

    parsed = urlsplit(str(value))
    host = (parsed.hostname or "").casefold()
    normalized_hosts = {item.casefold() for item in hosts}
    host_allowed = host in normalized_hosts or (
        allow_subdomains
        and any(host.endswith(f".{candidate}") for candidate in normalized_hosts)
    )
    if (
        parsed.scheme != "https"
        or not host_allowed
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
    ):
        raise NetworkTargetError("remote URL is outside the approved HTTPS hosts")
    return parsed.geturl()
