"""Headless download of an annual Oracle's Elixir CSV export.

The public refresh launcher used to obtain the annual CSV by opening Brave on a
Google Drive download link and then polling the browser's download directory.
That made an unattended, scheduled job depend on a logged-in GUI session: no
desktop, no data.  The file is a public Drive object, so a plain anonymous
HTTPS GET fetches the same bytes.  This module is that GET, and the browser
path is demoted to a last resort in the launcher.

Two things make an anonymous Drive download unsafe to trust naively:

* Drive answers a quota-blocked or consent-gated request with **HTTP 200** and
  a small HTML page, not with an error status.  Writing that page to the annual
  CSV path would replace real data with an interstitial.
* A connection cut mid-stream leaves a syntactically plausible but truncated
  CSV.

So this module is fail-closed by construction: it sniffs the first bytes before
it opens the output file, streams to a sibling ``.part`` file, re-checks the
committed bytes on disk, and only then renames.  The destination path is never
touched unless a validated body reached the disk in full.  ``_validate_oe_csv``
in :mod:`lol_kills.etl.oe_ingest` and the source-receipt binding remain the
authoritative checks downstream; the checks here are a belt on those braces,
placed early enough that a bad body never lands on the annual CSV path at all.

Stdlib only, deliberately.  This runs before pandas is needed, and a fetcher
that cannot start because a heavy optional dependency is broken is a fetcher
that cannot recover the pipeline.

Sources are restricted to the official Oracle's Elixir Drive folder and its
files.  No third-party mirror is ever contacted.
"""

from __future__ import annotations

import argparse
import hashlib
import html as html_module
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from lol_kills.etl.paths import OE_DRIVE_IDS, OE_FOLDER
from lol_kills.net import NetworkTargetError, require_https_url

__all__ = [
    "DEFAULT_DRIVE_FILE_IDS",
    "OeFetchError",
    "OeFetchQuotaError",
    "download_url",
    "fetch_oe_csv",
    "folder_listing_url",
    "main",
    "parse_folder_listing",
    "resolve_drive_file_id",
]


# Known Drive file ids per year.  This is deliberately *the same* mapping the
# rest of the ETL uses rather than a second copy: two independent OE resolvers
# had already drifted apart once in this repository (see the comment on
# ``lol_kills.etl.paths._print_path``), and a stale duplicate here would send
# the launcher at a file id nobody maintains.
DEFAULT_DRIVE_FILE_IDS: dict[str, str] = dict(OE_DRIVE_IDS)

# Anonymous download endpoint.  ``drive.google.com/uc?export=download`` answers
# large public files with a "virus scan" confirmation page; the usercontent
# host with ``confirm=t`` is the endpoint that serves the bytes directly.
DOWNLOAD_HOST = "drive.usercontent.google.com"
# The folder listing lives on the main Drive host.
FOLDER_HOST = "drive.google.com"
# A redirect must stay inside Google.  Drive hands large downloads to a
# regional googleusercontent host, so subdomains of both are allowed - and
# nothing else, which is what keeps an open redirect from pointing this
# downloader at a third-party mirror.
RESPONSE_HOSTS = frozenset({"google.com", "googleusercontent.com"})

# A browser-like agent.  Drive varies its interstitial behaviour by agent, and
# the default ``Python-urllib/3.x`` is the one most likely to be gated.  No
# credentials, cookies, or tokens are ever sent: the file is public and the
# request must stay anonymous.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

DEFAULT_TIMEOUT_SECONDS = 240.0
# Per-socket-operation bound.  The wall-clock bound is enforced separately in
# the streaming loop, because a socket timeout only bounds a single read and a
# slow trickle could otherwise outlast any deadline.
SOCKET_TIMEOUT_SECONDS = 60.0
# Enough of the body to see the whole CSV header line (160+ columns) before
# anything is written to disk.
SNIFF_BYTES = 8192
CHUNK_BYTES = 1 << 20
# The folder listing is ~12 KB; the cap only stops an unbounded read.
FOLDER_LISTING_MAX_BYTES = 4 << 20

# Mirrors ``lol_kills.etl.oe_ingest.OE_MIN_DOWNLOAD_BYTES``.  Duplicated rather
# than imported to keep this module free of the pandas import chain; a test
# asserts the two stay equal.
MIN_CSV_BYTES = 10_000
# The real export has 160+ columns.  Any Drive interstitial - even one without
# a doctype - has nothing resembling that many commas on its first line, so
# this catches HTML-ish bodies that slipped past the tag sniff.
MIN_HEADER_COMMAS = 20

EXIT_OK = 0
EXIT_FAILED = 1
# EX_TEMPFAIL.  The launcher reads this as "blocked, not broken" and reuses the
# existing CSV when it is still fresh, matching the other 75s in that script.
EXIT_QUOTA_BLOCKED = 75

_FILE_ID_RE = re.compile(r"\A[A-Za-z0-9_-]{10,128}\Z")
_ENTRY_MARKER = 'id="entry-'
_ENTRY_TITLE_RE = re.compile(r'flip-entry-title"[^>]*>([^<]*)<')


class OeFetchError(RuntimeError):
    """A headless Oracle's Elixir download could not be completed safely."""


class OeFetchQuotaError(OeFetchError):
    """Drive answered with an interstitial instead of the CSV.

    Raised for the whole family of "HTTP 200, but this is not your file"
    responses: the anonymous per-file quota block, a consent or virus-scan
    page, and any other HTML body.  The launcher distinguishes this from a
    broken network because the two need opposite responses - a quota block
    should reuse yesterday's still-fresh CSV rather than escalate to a browser,
    while a broken network should not silently pretend the feed is healthy.
    """


def _require_file_id(value: str | None, *, kind: str = "file") -> str:
    """Return a Drive id that is safe to interpolate into a URL."""

    candidate = str(value or "").strip()
    if not _FILE_ID_RE.match(candidate):
        raise OeFetchError(f"not a usable Google Drive {kind} id: {value!r}")
    return candidate


def _folder_id_from_locator(locator: str) -> str:
    """Extract the folder id from the canonical OE folder URL."""

    path = urllib.parse.urlsplit(str(locator)).path
    return path.rstrip("/").rsplit("/", 1)[-1]


OE_FOLDER_ID = _folder_id_from_locator(OE_FOLDER)


def download_url(file_id: str) -> str:
    """Return the anonymous download URL for one public Drive file."""

    query = urllib.parse.urlencode(
        {"id": _require_file_id(file_id), "export": "download", "confirm": "t"}
    )
    return require_https_url(
        f"https://{DOWNLOAD_HOST}/download?{query}", hosts={DOWNLOAD_HOST}
    )


def folder_listing_url(folder_id: str | None = None) -> str:
    """Return the anonymous listing URL for the official OE Drive folder.

    ``embeddedfolderview`` renders file names and ids without a sign-in.  The
    documented locator carries a ``#list`` fragment; it selects a client-side
    view and is never sent to the server, so it is omitted here.
    """

    query = urllib.parse.urlencode(
        {"id": _require_file_id(folder_id or OE_FOLDER_ID, kind="folder")}
    )
    return require_https_url(
        f"https://{FOLDER_HOST}/embeddedfolderview?{query}", hosts={FOLDER_HOST}
    )


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/csv,text/html;q=0.5,*/*;q=0.1",
            "Accept-Encoding": "identity",
        },
        method="GET",
    )


def _remaining(deadline: float, *, floor: float = 5.0) -> float:
    return max(floor, deadline - time.monotonic())


def _open(url: str, *, timeout: float) -> Any:
    """Open one URL, translating every transport failure into OeFetchError."""

    try:
        return urllib.request.urlopen(_request(url), timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise OeFetchError(f"Drive returned HTTP {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        raise OeFetchError(f"Drive request failed for {url}: {exc.reason}") from exc
    except OSError as exc:
        raise OeFetchError(f"Drive request failed for {url}: {exc}") from exc


def _final_url(response: Any, requested: str) -> str:
    """Return the URL the response actually came from, after any redirect."""

    getter = getattr(response, "geturl", None)
    if not callable(getter):
        return requested
    try:
        return str(getter() or requested)
    except Exception:  # noqa: BLE001 - a stubborn response object is not fatal
        return requested


def _require_google_response(url: str) -> None:
    """Refuse a body that a redirect fetched from outside Google."""

    try:
        require_https_url(url, hosts=RESPONSE_HOSTS, allow_subdomains=True)
    except NetworkTargetError as exc:
        raise OeFetchError(
            f"Drive redirected the download off Google to {url}"
        ) from exc


def _declared_length(response: Any) -> int | None:
    headers = getattr(response, "headers", None)
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    raw = getter("Content-Length")
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _require_csv_head(data: bytes, *, url: str, origin: str) -> None:
    """Refuse anything that is not the start of the annual CSV.

    ``origin`` names where the bytes came from so a failure says whether the
    interstitial arrived over the wire or was found on disk after the write.
    """

    if not data:
        raise OeFetchQuotaError(f"Drive returned an empty body ({origin}) for {url}")

    window = data[:512]
    lowered = window.lower()
    if (
        window.lstrip()[:1] == b"<"
        or b"<!doctype html" in lowered
        or b"<html" in lowered
    ):
        detail = ""
        if b"quota" in data[:4096].lower():
            detail = "; Drive reports the anonymous quota for this file is exhausted"
        raise OeFetchQuotaError(
            f"Drive returned an HTML interstitial instead of CSV ({origin}) "
            f"for {url}{detail}"
        )

    first_line = data.split(b"\n", 1)[0]
    commas = first_line.count(b",")
    if commas < MIN_HEADER_COMMAS:
        raise OeFetchQuotaError(
            f"Drive returned a body whose first line has {commas} commas "
            f"({origin}), far short of the {MIN_HEADER_COMMAS} a real annual "
            f"export header carries, for {url}"
        )


def parse_folder_listing(document: str) -> dict[str, str]:
    """Map file name to Drive id for the entries in a folder listing page.

    ``embeddedfolderview`` emits one block per file that opens with
    ``id="entry-<file id>"`` and carries the display name in a
    ``flip-entry-title`` element, so splitting on the id marker keeps each id
    paired with its own name instead of relying on two independent scans
    lining up.
    """

    entries: dict[str, str] = {}
    for chunk in str(document).split(_ENTRY_MARKER)[1:]:
        file_id, _, remainder = chunk.partition('"')
        if not _FILE_ID_RE.match(file_id.strip()):
            continue
        match = _ENTRY_TITLE_RE.search(remainder)
        if match is None:
            continue
        name = html_module.unescape(match.group(1)).strip()
        if name and name not in entries:
            entries[name] = file_id.strip()
    return entries


def annual_csv_name(year: str | int) -> str:
    return f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"


def _select_annual_file_id(entries: Mapping[str, str], year: str) -> str | None:
    exact = annual_csv_name(year)
    if exact in entries:
        return entries[exact]
    # Tolerate a small upstream rename.  This stays safe because the only
    # listing ever parsed is the official OE folder's.
    for name, file_id in entries.items():
        folded = name.casefold()
        if (
            folded.startswith(f"{year}_")
            and folded.endswith(".csv")
            and "oracleselixir" in folded
        ):
            return file_id
    return None


def resolve_drive_file_id(
    year: str | int, *, timeout_seconds: float = 60.0
) -> str | None:
    """Look the year's Drive file id up in the official folder listing.

    Returns ``None`` when the folder does not name a file for that year, which
    is a normal answer early in a season.  Raises :class:`OeFetchError` when
    the listing itself could not be fetched.
    """

    url = folder_listing_url()
    with _open(url, timeout=min(SOCKET_TIMEOUT_SECONDS, float(timeout_seconds))) as response:
        _require_google_response(_final_url(response, url))
        try:
            body = response.read(FOLDER_LISTING_MAX_BYTES)
        except OSError as exc:
            raise OeFetchError(f"Drive folder listing read failed: {exc}") from exc
    entries = parse_folder_listing(bytes(body).decode("utf-8", "replace"))
    if not entries:
        raise OeFetchError(f"Drive folder listing named no files: {url}")
    return _select_annual_file_id(entries, str(year))


def _stream_to_part(url: str, part_path: Path, *, deadline: float) -> tuple[int, str]:
    """Stream a validated body into ``part_path``; return (bytes, sha256).

    The head of the body is inspected *before* the output file is opened, so an
    interstitial never reaches the filesystem at all.
    """

    digest = hashlib.sha256()
    total = 0
    timeout = min(SOCKET_TIMEOUT_SECONDS, _remaining(deadline))
    with _open(url, timeout=timeout) as response:
        final_url = _final_url(response, url)
        _require_google_response(final_url)
        try:
            head = response.read(SNIFF_BYTES)
        except OSError as exc:
            raise OeFetchError(f"Drive read failed for {final_url}: {exc}") from exc
        _require_csv_head(bytes(head), url=final_url, origin="wire")
        declared = _declared_length(response)

        part_path.parent.mkdir(parents=True, exist_ok=True)
        part_path.unlink(missing_ok=True)
        with part_path.open("wb") as handle:
            digest.update(head)
            total += len(head)
            handle.write(head)
            while True:
                if time.monotonic() > deadline:
                    raise OeFetchError(
                        f"headless download exceeded its time budget after "
                        f"{total} bytes from {final_url}"
                    )
                try:
                    chunk = response.read(CHUNK_BYTES)
                except OSError as exc:
                    raise OeFetchError(
                        f"Drive read failed after {total} bytes from {final_url}: {exc}"
                    ) from exc
                if not chunk:
                    break
                digest.update(chunk)
                total += len(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())

    if declared is not None and declared != total:
        raise OeFetchError(
            f"headless download is truncated: Content-Length declared "
            f"{declared} bytes, {total} arrived from {final_url}"
        )
    if total < MIN_CSV_BYTES:
        raise OeFetchError(
            f"headless download is too small to be an annual export "
            f"({total} bytes, minimum {MIN_CSV_BYTES}) from {final_url}"
        )
    return total, digest.hexdigest()


def _attempt_download(
    *,
    year: str,
    file_id: str,
    destination: Path,
    deadline: float,
    started: float,
    resolved_from_folder: bool,
) -> dict[str, Any]:
    """Download one Drive file id into ``destination``, atomically."""

    url = download_url(file_id)
    part_path = destination.with_suffix(".part")
    if part_path == destination:
        raise OeFetchError(f"destination leaves no room for a staging file: {destination}")

    try:
        total, sha256 = _stream_to_part(url, part_path, deadline=deadline)
        # Re-read what actually landed.  The wire sniff saw a buffer; this sees
        # the bytes that are about to become the annual CSV.
        with part_path.open("rb") as handle:
            _require_csv_head(handle.read(SNIFF_BYTES), url=url, origin="staged file")
        on_disk = part_path.stat().st_size
        if on_disk != total:
            raise OeFetchError(
                f"staged file is {on_disk} bytes but {total} were written: {part_path}"
            )
        os.replace(part_path, destination)
    except BaseException:
        # Never leave a partial or rejected body behind for the launcher's
        # stable-size poll - or for a later run - to mistake for real data.
        part_path.unlink(missing_ok=True)
        raise

    return {
        "year": int(year),
        "destination": str(destination),
        "url": url,
        "drive_file_id": file_id,
        "resolved_from_folder": bool(resolved_from_folder),
        "bytes": int(total),
        "sha256": sha256,
        "duration_seconds": round(time.monotonic() - started, 3),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "transport": "anonymous_https_drive_download",
    }


def fetch_oe_csv(
    year: int,
    destination: Path,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Download the annual Oracle's Elixir CSV for ``year`` to ``destination``.

    The resolution ladder is: the pinned Drive file id for the year, then - if
    that fails - the id the official folder listing currently advertises, in
    case it rotated.  ``destination`` is written by atomic rename and is left
    untouched unless a fully validated body arrived.

    Raises :class:`OeFetchQuotaError` when Drive answered with an interstitial
    (the file is fine, the anonymous quota is not) and :class:`OeFetchError`
    for every other failure.
    """

    started = time.monotonic()
    deadline = started + float(timeout_seconds)
    try:
        normalized_year = str(int(year))
    except (TypeError, ValueError) as exc:
        raise OeFetchError(f"not a usable season year: {year!r}") from exc

    destination = Path(destination).expanduser()
    if destination.is_dir():
        raise OeFetchError(f"destination is a directory: {destination}")

    attempted: list[str] = []
    last_error: BaseException | None = None

    pinned = DEFAULT_DRIVE_FILE_IDS.get(normalized_year)
    if pinned:
        attempted.append(pinned)
        try:
            return _attempt_download(
                year=normalized_year,
                file_id=pinned,
                destination=destination,
                deadline=deadline,
                started=started,
                resolved_from_folder=False,
            )
        except OeFetchQuotaError as exc:
            last_error = exc
        except OeFetchError as exc:
            last_error = exc

    # The pinned id may simply be out of date: OE re-uploads the annual file
    # occasionally, which mints a new id.  Ask the folder what it holds now.
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
            return _attempt_download(
                year=normalized_year,
                file_id=resolved,
                destination=destination,
                deadline=deadline,
                started=started,
                resolved_from_folder=True,
            )
        except OeFetchQuotaError as exc:
            last_error = exc
        except OeFetchError as exc:
            last_error = exc

    if not attempted:
        raise OeFetchError(
            f"no Drive file id is known for {normalized_year} and the official "
            f"folder does not list one"
        ) from last_error

    summary = (
        f"headless Oracle's Elixir download failed for {normalized_year} after "
        f"{len(attempted)} attempt(s): {last_error}"
    )
    # Classify by the TERMINAL attempt. Exit 75 tells the launcher "blocked,
    # not broken - the cached inbox file may be reused". If the pinned id was
    # quota-blocked but the folder resolved a CURRENT id whose download then
    # failed with a network error or truncation, the current file is NOT known
    # to be quota-blocked, and reporting quota would let a stale cache paper
    # over a real transport failure. Quota is only reported when the final
    # attempt itself was a quota interstitial.
    if isinstance(last_error, OeFetchQuotaError):
        raise OeFetchQuotaError(summary) from last_error
    raise OeFetchError(summary) from last_error


def _emit(payload: Mapping[str, Any], *, stream: Any) -> None:
    print(json.dumps(dict(payload), sort_keys=True), file=stream)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lol_kills.etl.oe_fetch",
        description=(
            "Download one annual Oracle's Elixir CSV export headlessly, "
            "atomically, and fail-closed."
        ),
    )
    parser.add_argument("--year", type=int, required=True, help="season year, e.g. 2026")
    parser.add_argument(
        "--destination", type=Path, required=True, help="path to write the CSV to"
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"wall-clock budget for the download (default {DEFAULT_TIMEOUT_SECONDS:g})",
    )
    arguments = parser.parse_args(argv)

    try:
        evidence = fetch_oe_csv(
            arguments.year,
            arguments.destination,
            timeout_seconds=arguments.timeout_seconds,
        )
    except OeFetchQuotaError as exc:
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
    except (OeFetchError, NetworkTargetError, OSError) as exc:
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
