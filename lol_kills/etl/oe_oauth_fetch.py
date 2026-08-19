"""Authenticated "make a copy" download of an annual Oracle's Elixir CSV.

:mod:`lol_kills.etl.oe_fetch` fetches the annual export anonymously, which is
the right default: it needs no account, no secret, and no consent screen.  It
has one failure mode that no amount of retrying fixes.  A public Drive object
carries a per-file *anonymous* download quota, and once enough of the internet
has pulled the file that day, Drive answers every anonymous request with an
HTTP 200 interstitial instead of the bytes.  The launcher's only remaining
answer was to open Brave on the download link, which is exactly the GUI
dependency the headless fetcher was written to remove: no logged-in desktop,
no data.

The quota is charged against the *anonymous* pool, not against the file.  An
authenticated user may copy a file they can read into their own Drive, and the
copy is a new object in their own quota with an untouched download budget.  So
this module logs in once, and thereafter each fetch is:

    files.copy(source) -> files.get(copy, alt=media) -> files.delete(copy)

The copy is temporary in the strongest sense available: it is named
distinctively, and it is deleted in a ``finally`` block so a failed download
does not leave litter accumulating in the operator's Drive.

Three properties are load-bearing.

**Secrets never touch the filesystem.**  The refresh token and the client
credentials live in the macOS Keychain and nowhere else - not in a dotfile,
not in a launchd plist, not in the environment of a scheduled run.  Nothing
this module prints, on either stream, can carry token material: every line of
output goes through :func:`_redact` at the last possible moment.

**Fail-closed downloads.**  The validation is not re-implemented here.  The
streaming, interstitial sniff, Content-Length reconciliation, minimum-size
check and atomic rename are :mod:`lol_kills.etl.oe_fetch`'s, imported and
called, so the authenticated transport cannot drift into accepting a body the
anonymous one would refuse.  An authenticated request is not a trusted one.

**Stdlib only.**  Same reason as ``oe_fetch``: a fetcher that cannot start
because a heavy optional dependency is broken is a fetcher that cannot recover
the pipeline.  ``google-api-python-client`` would pull in a dependency tree
larger than the rest of this ETL, to issue three HTTP requests.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Mapping

from lol_kills.etl.oe_fetch import (
    DEFAULT_DRIVE_FILE_IDS,
    SNIFF_BYTES,
    SOCKET_TIMEOUT_SECONDS,
    OeFetchError,
    OeFetchQuotaError,
    _remaining,
    _require_csv_head,
    _require_file_id,
    commit_staged_download,
    download_url,
    resolve_drive_file_id,
    staging_path,
    stream_validated_body,
)
from lol_kills.net import NetworkTargetError, require_https_url

__all__ = [
    "EXIT_FAILED",
    "EXIT_OK",
    "EXIT_QUOTA_BLOCKED",
    "EXIT_UNCONFIGURED",
    "KEYCHAIN_ACCOUNT",
    "KEYCHAIN_ENV_OVERRIDE",
    "KEYCHAIN_SERVICE",
    "OeOauthError",
    "OeOauthQuotaError",
    "OeOauthUnconfigured",
    "fetch_via_copy",
    "main",
    "read_stored_login",
    "temp_copy_name",
]


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

# The Drive REST API.  Note this is NOT covered by oe_fetch.RESPONSE_HOSTS,
# which deliberately allows only google.com and googleusercontent.com; widening
# that set would loosen the anonymous downloader's redirect guard for no reason,
# so this module carries its own.
DRIVE_API_HOST = "www.googleapis.com"
TOKEN_HOST = "oauth2.googleapis.com"
AUTH_HOST = "accounts.google.com"

# A media download is handed off to a regional content host, so the response
# may legitimately arrive from a googleusercontent subdomain.  Nothing outside
# Google is ever accepted, which is what stops an open redirect from pointing
# this downloader at a third-party mirror.
RESPONSE_HOSTS = frozenset({"googleapis.com", "google.com", "googleusercontent.com"})

# The full Drive scope.  ``drive.file`` only grants access to files this client
# created, which cannot read the public source file, and there is no narrower
# scope that permits "read someone else's public file and copy it".
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"

CALLBACK_PATH = "/callback"

# --------------------------------------------------------------------------
# Keychain
# --------------------------------------------------------------------------

KEYCHAIN_SERVICE = "scryglass-oe-drive-oauth"
KEYCHAIN_ACCOUNT = "scryglass"
SECURITY_BIN = "/usr/bin/security"

# Test and CI override.  Read BEFORE the Keychain so the whole fetch path can
# be exercised on a machine with no Keychain entry and no Google account at
# all.  This is NEVER a production mechanism: a scheduled run's environment is
# visible to every child process it spawns and to anything that dumps `ps -E`
# or a crash report, so real credentials belong in the Keychain and nowhere
# else.  The launcher does not set this variable.
KEYCHAIN_ENV_OVERRIDE = "SCRYGLASS_OE_OAUTH_JSON"

REQUIRED_LOGIN_FIELDS = ("client_id", "client_secret", "refresh_token")

# --------------------------------------------------------------------------
# Budgets
# --------------------------------------------------------------------------

DEFAULT_TIMEOUT_SECONDS = 240.0
# A one-time interactive consent may reasonably take a few minutes.
LOGIN_TIMEOUT_SECONDS = 300.0
# The API answers are small JSON documents; the cap only stops an unbounded
# read from a misbehaving endpoint.
MAX_JSON_BYTES = 1 << 20
MAX_ERROR_BYTES = 64 << 10

EXIT_OK = 0
EXIT_FAILED = 1
# EX_UNAVAILABLE.  The launcher reads this as "this rung is not configured on
# this machine" and moves down the ladder without treating it as a fault.
EXIT_UNCONFIGURED = 69
# EX_TEMPFAIL, matching oe_fetch: blocked, not broken.
EXIT_QUOTA_BLOCKED = 75


class OeOauthError(RuntimeError):
    """An authenticated Oracle's Elixir download could not be completed."""


class OeOauthUnconfigured(OeOauthError):
    """No usable OAuth login is stored, or the stored one is no longer valid.

    Distinguished from a plain failure because the launcher must treat it as
    "this rung does not apply here" rather than "the feed is broken": a worker
    that has never run ``--login`` is a normal, expected state.
    """


class OeOauthQuotaError(OeOauthError):
    """Drive refused to serve the bytes even through an owned copy."""


# --------------------------------------------------------------------------
# Redaction
# --------------------------------------------------------------------------

_REDACTED = "<redacted>"

# Shapes of Google credential material, as a backstop for anything that is
# echoed back to us rather than supplied by us: OAuth access tokens, refresh
# tokens, installed-app client secrets, and authorization codes.
_SECRET_SHAPES = (
    re.compile(r"ya29\.[A-Za-z0-9._~+/\-]{10,}"),
    re.compile(r"1//[A-Za-z0-9._~+/\-]{10,}"),
    re.compile(r"GOCSPX-[A-Za-z0-9._~\-]{5,}"),
    re.compile(r"\b4/[A-Za-z0-9._~\-]{20,}"),
)

# Exact values this process has handled.  Populated as credentials are read or
# minted so that _redact can remove them verbatim, which covers secrets whose
# shape does not match any pattern above (Google has changed token formats
# before, and a client secret from an older project does not carry the GOCSPX
# prefix).
_known_secrets: set[str] = set()


def _remember_secret(value: object) -> None:
    """Record one credential so it can never appear in this process's output."""

    text = str(value or "")
    # Short strings are not credentials, and redacting them would corrupt
    # unrelated output.
    if len(text) >= 8:
        _known_secrets.add(text)


def _redact(text: str) -> str:
    """Remove every known or credential-shaped substring from ``text``."""

    cleaned = str(text)
    # Longest first, so a token that contains another known value as a prefix
    # is replaced whole rather than leaving a tail behind.
    for secret in sorted(_known_secrets, key=len, reverse=True):
        cleaned = cleaned.replace(secret, _REDACTED)
    for pattern in _SECRET_SHAPES:
        cleaned = pattern.sub(_REDACTED, cleaned)
    return cleaned


def _emit(payload: Mapping[str, Any], *, stream: Any) -> None:
    """Print one JSON line, redacted at the last possible moment.

    Every write this module makes goes through here, so redaction is a single
    chokepoint rather than a discipline applied at each call site.
    """

    print(_redact(json.dumps(dict(payload), sort_keys=True)), file=stream)


def _say(message: str, *, stream: Any) -> None:
    print(_redact(str(message)), file=stream)


# --------------------------------------------------------------------------
# Stored login
# --------------------------------------------------------------------------


def _read_keychain_blob() -> str:
    """Return the raw Keychain password, or raise :class:`OeOauthUnconfigured`."""

    try:
        result = subprocess.run(
            [
                SECURITY_BIN,
                "find-generic-password",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                KEYCHAIN_ACCOUNT,
                "-w",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise OeOauthUnconfigured(
            f"could not read the Keychain item {KEYCHAIN_SERVICE}: {exc}"
        ) from exc
    if result.returncode != 0:
        # security(1) says "The specified item could not be found in the
        # keychain."  Its stderr is deliberately not echoed: it is not useful
        # here and this module does not forward text it did not compose.
        raise OeOauthUnconfigured("no OAuth login stored; run --login")
    return result.stdout


def _parse_login(raw: str, *, origin: str) -> dict[str, str]:
    """Validate the stored blob and register its secrets for redaction."""

    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise OeOauthUnconfigured(
            f"the stored OAuth login ({origin}) is not JSON; run --login"
        ) from exc
    if not isinstance(payload, dict):
        raise OeOauthUnconfigured(
            f"the stored OAuth login ({origin}) is not a JSON object; run --login"
        )

    login: dict[str, str] = {}
    missing: list[str] = []
    for field in REQUIRED_LOGIN_FIELDS:
        value = str(payload.get(field) or "").strip()
        if not value:
            missing.append(field)
        else:
            login[field] = value
    if missing:
        # Names only.  The values are exactly what must not be reported.
        raise OeOauthUnconfigured(
            f"the stored OAuth login ({origin}) is missing "
            f"{', '.join(missing)}; run --login"
        )

    _remember_secret(login["client_secret"])
    _remember_secret(login["refresh_token"])
    return login


def read_stored_login() -> dict[str, str]:
    """Return the stored OAuth credentials.

    The environment override is consulted FIRST and exists only so tests and
    CI can drive the fetch path without a Keychain; see the comment on
    :data:`KEYCHAIN_ENV_OVERRIDE`.  Production reads the Keychain.
    """

    override = os.environ.get(KEYCHAIN_ENV_OVERRIDE)
    if override is not None and override.strip():
        return _parse_login(override, origin=f"${KEYCHAIN_ENV_OVERRIDE}")
    return _parse_login(_read_keychain_blob(), origin="Keychain")


def store_login(login: Mapping[str, str]) -> None:
    """Write the credential blob into the Keychain, replacing any previous one."""

    payload = json.dumps(
        {field: str(login[field]) for field in REQUIRED_LOGIN_FIELDS},
        sort_keys=True,
    )
    # The blob is passed on argv because security(1) offers no way to read a
    # password for add-generic-password from stdin.  That makes it briefly
    # visible to a `ps` running as the same user on the same machine at the
    # same moment, which is the price of using the tool as documented; it is
    # never written to a file, and -U replaces rather than duplicates the item.
    try:
        result = subprocess.run(
            [
                SECURITY_BIN,
                "add-generic-password",
                "-U",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                KEYCHAIN_ACCOUNT,
                "-w",
                payload,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise OeOauthError(f"could not write the Keychain item: {exc}") from exc
    if result.returncode != 0:
        raise OeOauthError(
            f"security(1) refused to store the login (status {result.returncode})"
        )


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


def _require_google_response(url: str) -> None:
    """Refuse a body that a redirect fetched from outside Google."""

    try:
        require_https_url(url, hosts=RESPONSE_HOSTS, allow_subdomains=True)
    except NetworkTargetError as exc:
        raise OeOauthError(f"Drive redirected the download off Google to {url}") from exc


def _error_detail(exc: urllib.error.HTTPError) -> str:
    """Return a bounded, redacted description of an HTTP error body."""

    try:
        body = exc.read(MAX_ERROR_BYTES)
    except Exception:  # noqa: BLE001 - a stubborn error body is not fatal
        body = b""
    return _redact(bytes(body).decode("utf-8", "replace").strip())


def _api_url(path: str, query: Mapping[str, str] | None = None) -> str:
    url = f"https://{DRIVE_API_HOST}{path}"
    if query:
        url = f"{url}?{urllib.parse.urlencode(dict(query))}"
    return require_https_url(url, hosts={DRIVE_API_HOST})


def _authorized_request(
    method: str,
    url: str,
    *,
    access_token: str,
    body: Mapping[str, Any] | None = None,
    accept: str = "application/json",
) -> urllib.request.Request:
    data: bytes | None = None
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": accept,
        "Accept-Encoding": "identity",
    }
    if body is not None:
        data = json.dumps(dict(body)).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    return urllib.request.Request(url, data=data, headers=headers, method=method)


def _classify_http_error(exc: urllib.error.HTTPError, *, what: str) -> OeOauthError:
    """Translate a Drive API HTTP error into the right exception class."""

    detail = _error_detail(exc)
    lowered = detail.lower()
    if exc.code == 403 and ("quota" in lowered or "cannotcopyfile" in lowered):
        return OeOauthQuotaError(
            f"Drive refused to {what}: HTTP 403, {detail or 'no detail given'}"
        )
    if exc.code == 429:
        # Not named by the spec, but the same class of answer: Drive is rate
        # limiting, not broken, and the launcher's "blocked" rung is the honest
        # place for it.
        return OeOauthQuotaError(
            f"Drive rate limited the request to {what}: HTTP 429, "
            f"{detail or 'no detail given'}"
        )
    if exc.code in (401, 403) and (
        "invalid_credentials" in lowered
        or "autherror" in lowered
        or "unauthorized" in lowered
    ):
        return OeOauthUnconfigured(
            f"Drive rejected the access token when asked to {what}; re-run --login"
        )
    return OeOauthError(
        f"Drive returned HTTP {exc.code} when asked to {what}: "
        f"{detail or 'no detail given'}"
    )


# Hosts a bearer-carrying request may ever touch. urllib's default redirect
# handler copies the Authorization header onto the redirected request BEFORE
# any caller-side check can run, so a Drive response carrying an off-Google
# Location would disclose the full-scope token to an arbitrary host. This
# handler validates every destination BEFORE following it; within these hosts
# the copied header is intended (googleusercontent serves alt=media bodies).
_ALLOWED_TOKEN_HOSTS = (
    ".googleapis.com",
    ".google.com",
    ".googleusercontent.com",
)


def _host_may_receive_token(url: str) -> bool:
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    return bool(host) and any(
        host == allowed.lstrip(".") or host.endswith(allowed)
        for allowed in _ALLOWED_TOKEN_HOSTS
    )


class _GoogleOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse to follow any redirect whose destination is not a Google host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: N802
        if not _host_may_receive_token(newurl):
            raise OeOauthError(
                "Drive redirected an authorized request to a non-Google host; "
                "refusing to follow it"
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_AUTHORIZED_OPENER = urllib.request.build_opener(_GoogleOnlyRedirectHandler())


def _open_authorized(request, *, timeout: float):
    """Open a bearer-carrying request through the redirect-validating opener."""

    if not _host_may_receive_token(request.full_url):
        raise OeOauthError(
            "refusing to send the Drive token to a non-Google host: "
            + urllib.parse.urlsplit(request.full_url).netloc
        )
    return _AUTHORIZED_OPENER.open(request, timeout=timeout)


def _api_json(
    method: str,
    url: str,
    *,
    access_token: str,
    what: str,
    body: Mapping[str, Any] | None = None,
    timeout: float,
) -> dict[str, Any]:
    request = _authorized_request(method, url, access_token=access_token, body=body)
    try:
        with _open_authorized(request, timeout=timeout) as response:
            raw = response.read(MAX_JSON_BYTES)
    except urllib.error.HTTPError as exc:
        raise _classify_http_error(exc, what=what) from exc
    except urllib.error.URLError as exc:
        raise OeOauthError(f"the request to {what} failed: {exc.reason}") from exc
    except OSError as exc:
        raise OeOauthError(f"the request to {what} failed: {exc}") from exc

    text = bytes(raw).decode("utf-8", "replace").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise OeOauthError(f"Drive returned a non-JSON answer when asked to {what}") from exc
    if not isinstance(payload, dict):
        raise OeOauthError(f"Drive returned an unexpected answer when asked to {what}")
    return payload


# --------------------------------------------------------------------------
# Token refresh
# --------------------------------------------------------------------------

# Token-endpoint errors that mean the stored login is dead rather than that the
# request was malformed.  All of them require a fresh consent, so the launcher
# must see "not configured" and move on rather than escalate a fault.
_DEAD_LOGIN_ERRORS = ("invalid_grant", "invalid_client", "unauthorized_client")


def _token_endpoint(form: Mapping[str, str], *, timeout: float, what: str) -> dict[str, Any]:
    url = require_https_url(f"https://{TOKEN_HOST}/token", hosts={TOKEN_HOST})
    data = urllib.parse.urlencode(dict(form)).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with _open_authorized(request, timeout=timeout) as response:
            raw = response.read(MAX_JSON_BYTES)
    except urllib.error.HTTPError as exc:
        detail = _error_detail(exc)
        lowered = detail.lower()
        if any(marker in lowered for marker in _DEAD_LOGIN_ERRORS):
            raise OeOauthUnconfigured(
                f"the stored OAuth login is no longer accepted by Google "
                f"({detail or f'HTTP {exc.code}'}); re-run --login"
            ) from exc
        raise OeOauthError(
            f"the token endpoint refused to {what}: HTTP {exc.code}, "
            f"{detail or 'no detail given'}"
        ) from exc
    except urllib.error.URLError as exc:
        raise OeOauthError(f"could not reach the token endpoint to {what}: {exc.reason}") from exc
    except OSError as exc:
        raise OeOauthError(f"could not reach the token endpoint to {what}: {exc}") from exc

    try:
        payload = json.loads(bytes(raw).decode("utf-8", "replace"))
    except ValueError as exc:
        raise OeOauthError(f"the token endpoint returned a non-JSON answer to {what}") from exc
    if not isinstance(payload, dict):
        raise OeOauthError(f"the token endpoint returned an unexpected answer to {what}")

    _remember_secret(payload.get("access_token"))
    _remember_secret(payload.get("refresh_token"))
    return payload


def refresh_access_token(
    login: Mapping[str, str], *, timeout: float = SOCKET_TIMEOUT_SECONDS
) -> tuple[str, str]:
    """Exchange the stored refresh token for a short-lived access token."""

    payload = _token_endpoint(
        {
            "client_id": login["client_id"],
            "client_secret": login["client_secret"],
            "refresh_token": login["refresh_token"],
            "grant_type": "refresh_token",
        },
        timeout=timeout,
        what="refresh the access token",
    )
    access_token = str(payload.get("access_token") or "")
    if not access_token:
        raise OeOauthUnconfigured(
            "the token endpoint returned no access token; re-run --login"
        )
    return access_token, str(payload.get("scope") or "")


# --------------------------------------------------------------------------
# Copy / download / delete
# --------------------------------------------------------------------------


def temp_copy_name(year: str | int) -> str:
    """Return a distinctive, collision-proof name for the throwaway copy.

    Distinctive on purpose: if a delete ever fails, an operator scanning their
    Drive must be able to tell at a glance that the leftover belongs to this
    job and is safe to remove.
    """

    return f"scryglass-tmp-oe-{year}-{secrets.token_hex(6)}"


def _copy_drive_file(
    source_id: str, *, access_token: str, name: str, timeout: float
) -> str:
    url = _api_url(f"/drive/v3/files/{source_id}/copy", {"fields": "id"})
    payload = _api_json(
        "POST",
        url,
        access_token=access_token,
        what=f"copy Drive file {source_id}",
        body={"name": name},
        timeout=timeout,
    )
    copy_id = str(payload.get("id") or "")
    try:
        return _require_file_id(copy_id, kind="file")
    except OeFetchError as exc:
        raise OeOauthError("Drive's copy response named no usable file id") from exc


def _delete_drive_file(copy_id: str, *, access_token: str, timeout: float) -> bool:
    """Delete the temporary copy.  Never raises: it runs in a ``finally``.

    A delete failure must not replace the exception that is already on its way
    up, which would turn a diagnosable download failure into a confusing
    cleanup failure.  It is reported on stderr and recorded in the evidence
    instead.
    """

    url = _api_url(f"/drive/v3/files/{copy_id}")
    request = _authorized_request("DELETE", url, access_token=access_token)
    try:
        with _open_authorized(request, timeout=timeout):
            return True
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            # Already gone is the desired end state.
            return True
        _say(
            f"oe_oauth_fetch: could not delete the temporary Drive copy "
            f"{copy_id}: HTTP {exc.code}",
            stream=sys.stderr,
        )
    except Exception as exc:  # noqa: BLE001 - cleanup must never mask the cause
        _say(
            f"oe_oauth_fetch: could not delete the temporary Drive copy "
            f"{copy_id}: {exc}",
            stream=sys.stderr,
        )
    return False


def _download_copy(
    copy_id: str,
    *,
    access_token: str,
    destination: Path,
    deadline: float,
    source_url: str,
) -> tuple[int, str]:
    """Stream the copy to ``destination`` through oe_fetch's fail-closed path."""

    url = _api_url(f"/drive/v3/files/{copy_id}", {"alt": "media"})
    part_path = staging_path(destination)
    committed = False
    try:
        request = _authorized_request(
            "GET",
            url,
            access_token=access_token,
            accept="text/csv,application/octet-stream;q=0.9,*/*;q=0.1",
        )
        timeout = min(SOCKET_TIMEOUT_SECONDS, _remaining(deadline))
        try:
            response = _open_authorized(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            raise _classify_http_error(exc, what=f"download Drive file {copy_id}") from exc
        except urllib.error.URLError as exc:
            raise OeOauthError(f"the copy download failed: {exc.reason}") from exc
        except OSError as exc:
            raise OeOauthError(f"the copy download failed: {exc}") from exc

        with response:
            final_url = url
            getter = getattr(response, "geturl", None)
            if callable(getter):
                try:
                    final_url = str(getter() or url)
                except Exception:  # noqa: BLE001 - a stubborn response is not fatal
                    final_url = url
            _require_google_response(final_url)
            total, sha256 = stream_validated_body(
                response, part_path, url=final_url, deadline=deadline
            )

        commit_staged_download(part_path, destination, url=source_url, total=total)
        committed = True
        # Belt on the braces: re-sniff what is now on the destination path
        # before this module tells the launcher the file is good.  The staged
        # check ran against the pre-rename inode; this runs against the file
        # the launcher will actually hand to the importer.
        with destination.open("rb") as handle:
            _require_csv_head(
                handle.read(SNIFF_BYTES), url=source_url, origin="committed file"
            )
    except BaseException:
        # Leave nothing behind that a later run - or the launcher's own
        # staging cleanup - could mistake for a validated download.
        part_path.unlink(missing_ok=True)
        if committed:
            destination.unlink(missing_ok=True)
        raise
    return total, sha256


def _attempt_copy_download(
    *,
    source_id: str,
    year: str,
    destination: Path,
    access_token: str,
    deadline: float,
    started: float,
    scope: str,
    resolved_from_folder: bool,
) -> dict[str, Any]:
    """Copy one source id, download the copy, and always delete the copy."""

    name = temp_copy_name(year)
    copy_id: str | None = None
    copy_deleted = False
    try:
        copy_id = _copy_drive_file(
            source_id,
            access_token=access_token,
            name=name,
            timeout=min(SOCKET_TIMEOUT_SECONDS, _remaining(deadline)),
        )
        total, sha256 = _download_copy(
            copy_id,
            access_token=access_token,
            destination=destination,
            deadline=deadline,
            source_url=download_url(source_id),
        )
    finally:
        # Unconditional cleanup: a copy that was created must not survive this
        # call, whether the download succeeded, failed validation, or blew the
        # time budget.  _remaining floors at five seconds, so the delete still
        # gets a chance even when the download consumed the whole budget.
        if copy_id:
            copy_deleted = _delete_drive_file(
                copy_id,
                access_token=access_token,
                timeout=min(SOCKET_TIMEOUT_SECONDS, _remaining(deadline)),
            )

    return {
        "year": int(year),
        "destination": str(destination),
        # The SOURCE object, not the throwaway copy: the copy is deleted by the
        # time this returns, and the receipt must name the Drive object the
        # bytes actually originate from.  This is also what keeps the launcher's
        # existing evidence reader working unchanged.
        "drive_file_id": source_id,
        "url": download_url(source_id),
        "resolved_from_folder": bool(resolved_from_folder),
        "copy_used": True,
        "copy_deleted": bool(copy_deleted),
        "bytes": int(total),
        "sha256": sha256,
        "scope": scope,
        "duration_seconds": round(time.monotonic() - started, 3),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "transport": "oauth_drive_copy",
    }


def fetch_via_copy(
    year: int | str,
    destination: Path,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Fetch the annual CSV for ``year`` through an owned Drive copy.

    Raises :class:`OeOauthUnconfigured` when no usable login is stored,
    :class:`OeOauthQuotaError` when Drive refuses even the copy, and
    :class:`OeOauthError` for every other failure.
    """

    started = time.monotonic()
    deadline = started + float(timeout_seconds)
    try:
        normalized_year = str(int(year))
    except (TypeError, ValueError) as exc:
        raise OeOauthError(f"not a usable season year: {year!r}") from exc

    destination = Path(destination).expanduser()
    if destination.is_dir():
        raise OeOauthError(f"destination is a directory: {destination}")

    login = read_stored_login()
    access_token, scope = refresh_access_token(
        login, timeout=min(SOCKET_TIMEOUT_SECONDS, _remaining(deadline))
    )

    attempted: list[str] = []
    last_error: BaseException | None = None

    def _attempt(source_id: str, *, resolved_from_folder: bool) -> dict[str, Any]:
        return _attempt_copy_download(
            source_id=source_id,
            year=normalized_year,
            destination=destination,
            access_token=access_token,
            deadline=deadline,
            started=started,
            scope=scope,
            resolved_from_folder=resolved_from_folder,
        )

    pinned = DEFAULT_DRIVE_FILE_IDS.get(normalized_year)
    if pinned:
        attempted.append(pinned)
        try:
            return _attempt(pinned, resolved_from_folder=False)
        except OeOauthUnconfigured:
            # A dead token is not a per-file problem; a second id would fail
            # the same way.  Let the launcher hear "not configured" at once.
            raise
        except (OeOauthError, OeFetchError, NetworkTargetError, OSError) as exc:
            last_error = exc

    # The pinned id may simply be out of date: OE re-uploads the annual file
    # occasionally, which mints a new id.  Ask the official folder what it holds
    # now.  A folder listing is not a file download, so it is not subject to the
    # anonymous download quota that sent the run down this rung to begin with.
    resolved: str | None = None
    try:
        resolved = resolve_drive_file_id(
            normalized_year, timeout_seconds=_remaining(deadline)
        )
    except OeFetchError as exc:
        if last_error is None:
            last_error = exc

    if resolved and resolved not in attempted:
        attempted.append(resolved)
        try:
            return _attempt(resolved, resolved_from_folder=True)
        except OeOauthUnconfigured:
            raise
        except (OeOauthError, OeFetchError, NetworkTargetError, OSError) as exc:
            last_error = exc

    if not attempted:
        raise OeOauthError(
            f"no Drive file id is known for {normalized_year} and the official "
            f"folder does not list one"
        ) from last_error

    summary = (
        f"OAuth Drive-copy download failed for {normalized_year} after "
        f"{len(attempted)} attempt(s): {last_error}"
    )
    # Classify by the TERMINAL attempt, exactly as oe_fetch does: reporting
    # "quota" when the last thing that went wrong was a transport failure would
    # let the launcher's stale-reuse rung paper over a real breakage.
    if isinstance(last_error, (OeOauthQuotaError, OeFetchQuotaError)):
        raise OeOauthQuotaError(summary) from last_error
    raise OeOauthError(summary) from last_error


# --------------------------------------------------------------------------
# One-time login
# --------------------------------------------------------------------------


class _CallbackHandler(BaseHTTPRequestHandler):
    """Receive exactly one OAuth redirect on the loopback interface."""

    # The default BaseHTTPRequestHandler access log writes the full request
    # line - including "?code=4/..." - to stderr.  Silence it: the whole point
    # of this module is that authorization material never reaches a log.
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path != CALLBACK_PATH:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        params = urllib.parse.parse_qs(parsed.query)
        setattr(
            self.server,
            "oauth_result",
            {key: values[0] for key, values in params.items() if values},
        )
        body = (
            b"<!doctype html><html><body><h1>Scryglass</h1>"
            b"<p>Login received. You can close this tab.</p></body></html>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _pkce_pair() -> tuple[str, str]:
    """Return (code_verifier, S256 code_challenge)."""

    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return verifier, challenge


def _consent_url(*, client_id: str, redirect_uri: str, challenge: str, state: str) -> str:
    query = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": DRIVE_SCOPE,
            # Without both of these Google returns no refresh token on a repeat
            # authorisation, and an unattended job cannot re-consent.
            "access_type": "offline",
            "prompt": "consent",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
        }
    )
    return require_https_url(
        f"https://{AUTH_HOST}/o/oauth2/v2/auth?{query}", hosts={AUTH_HOST}
    )


def _await_callback(server: HTTPServer, *, deadline: float) -> dict[str, str]:
    setattr(server, "oauth_result", None)
    server.timeout = 5
    while getattr(server, "oauth_result", None) is None:
        if time.monotonic() > deadline:
            raise OeOauthError("no OAuth redirect arrived before the login timed out")
        server.handle_request()
    return dict(getattr(server, "oauth_result") or {})


def login(*, stdin_prompt: Any = None) -> str:
    """Run the one-time installed-app consent flow; return the granted scope."""

    prompt = stdin_prompt or input
    client_id = str(prompt("Google OAuth client id: ")).strip()
    if not client_id:
        raise OeOauthError("no client id was entered")
    # getpass keeps the secret off the terminal, out of the scrollback, and out
    # of any transcript of this session.
    client_secret = str(getpass.getpass("Google OAuth client secret: ")).strip()
    if not client_secret:
        raise OeOauthError("no client secret was entered")
    _remember_secret(client_secret)

    verifier, challenge = _pkce_pair()
    state = secrets.token_urlsafe(24)

    server = HTTPServer(("127.0.0.1", 0), _CallbackHandler)
    try:
        port = int(server.server_port)
        redirect_uri = f"http://127.0.0.1:{port}{CALLBACK_PATH}"
        url = _consent_url(
            client_id=client_id,
            redirect_uri=redirect_uri,
            challenge=challenge,
            state=state,
        )
        _say("Open this URL to grant Drive access:", stream=sys.stdout)
        _say(url, stream=sys.stdout)
        sys.stdout.flush()
        try:
            subprocess.run(["/usr/bin/open", url], check=False, timeout=15)
        except (OSError, subprocess.SubprocessError):
            # Printing the URL is the contract; opening a browser is a courtesy.
            pass

        result = _await_callback(server, deadline=time.monotonic() + LOGIN_TIMEOUT_SECONDS)
    finally:
        server.server_close()

    if result.get("error"):
        raise OeOauthError(f"Google refused the consent: {result['error']}")
    # Constant-time compare: the state is the only thing standing between this
    # loopback listener and a code planted by another local process.
    if not secrets.compare_digest(str(result.get("state") or ""), state):
        raise OeOauthError("the OAuth redirect carried the wrong state; login refused")
    code = str(result.get("code") or "")
    if not code:
        raise OeOauthError("the OAuth redirect carried no authorization code")
    _remember_secret(code)

    payload = _token_endpoint(
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "code_verifier": verifier,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        },
        timeout=SOCKET_TIMEOUT_SECONDS,
        what="exchange the authorization code",
    )
    refresh_token = str(payload.get("refresh_token") or "")
    if not refresh_token:
        raise OeOauthError(
            "Google returned no refresh token; the consent must be granted with "
            "access_type=offline and prompt=consent"
        )

    store_login(
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
        }
    )
    return str(payload.get("scope") or DRIVE_SCOPE)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lol_kills.etl.oe_oauth_fetch",
        description=(
            "Download one annual Oracle's Elixir CSV through an authenticated "
            "Drive copy, so an exhausted anonymous quota does not stop the feed."
        ),
    )
    parser.add_argument(
        "--login",
        action="store_true",
        help="run the one-time consent flow and store the login in the Keychain",
    )
    parser.add_argument("--year", type=int, help="season year, e.g. 2026")
    parser.add_argument("--destination", type=Path, help="path to write the CSV to")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"wall-clock budget for the fetch (default {DEFAULT_TIMEOUT_SECONDS:g})",
    )
    arguments = parser.parse_args(argv)

    if arguments.login:
        try:
            scope = login()
        except OeOauthError as exc:
            _emit({"status": "login_failed", "error": str(exc)}, stream=sys.stderr)
            return EXIT_FAILED
        # Deliberately minimal: what was stored, and what it can do.  Nothing
        # about the credential itself.
        _say("login stored in keychain", stream=sys.stdout)
        _say(f"granted scope: {scope}", stream=sys.stdout)
        return EXIT_OK

    if arguments.year is None or arguments.destination is None:
        parser.error("--year and --destination are required unless --login is given")

    try:
        evidence = fetch_via_copy(
            arguments.year,
            arguments.destination,
            timeout_seconds=arguments.timeout_seconds,
        )
    except OeOauthUnconfigured as exc:
        _emit(
            {
                "status": "unconfigured",
                "year": arguments.year,
                "destination": str(arguments.destination),
                "error": str(exc),
            },
            stream=sys.stderr,
        )
        return EXIT_UNCONFIGURED
    except (OeOauthQuotaError, OeFetchQuotaError) as exc:
        _emit(
            {
                "status": "quota_blocked",
                "year": arguments.year,
                "destination": str(arguments.destination),
                "error": str(exc),
            },
            stream=sys.stderr,
        )
        return EXIT_QUOTA_BLOCKED
    except (OeOauthError, OeFetchError, NetworkTargetError, OSError) as exc:
        _emit(
            {
                "status": "failed",
                "year": arguments.year,
                "destination": str(arguments.destination),
                "error": str(exc),
            },
            stream=sys.stderr,
        )
        return EXIT_FAILED

    _emit({"status": "downloaded", **evidence}, stream=sys.stdout)
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - thin shell entry point
    raise SystemExit(main())
