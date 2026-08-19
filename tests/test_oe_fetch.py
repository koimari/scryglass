"""Behaviour tests for the headless Oracle's Elixir downloader.

No test here touches the network: ``urllib.request.urlopen`` is replaced by a
router that answers from in-memory fixtures, which lets the awkward cases -
a quota interstitial served with HTTP 200, a rotated Drive file id, a body
truncated against its own Content-Length - be exercised deterministically.

The property under test throughout is fail-closed behaviour: the destination
path must never receive anything but a fully validated body, and no staging
file may survive a failure.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import pytest

from lol_kills.etl import oe_fetch
from lol_kills.etl.paths import OE_DRIVE_IDS

PINNED_2026 = OE_DRIVE_IDS["2026"]
ROTATED_2026 = "1RotatedIdForTheAnnualExport_0000000"

QUOTA_HTML = (
    b"<!DOCTYPE html><html><head><title>Google Drive - Quota exceeded</title>"
    b"</head><body><p>Sorry, you can't view or download this file at this "
    b"time.</p><p>Too many users have viewed or downloaded this file "
    b"recently.</p></body></html>"
)

FOLDER_HTML_TEMPLATE = """<html><body>
<div class="flip-entry" id="entry-{id_2025}">
  <span class="flip-entry-title">2025_LoL_esports_match_data_from_OraclesElixir.csv</span>
</div>
<div class="flip-entry" id="entry-{id_2026}">
  <span class="flip-entry-title">2026_LoL_esports_match_data_from_OraclesElixir.csv</span>
</div>
</body></html>"""


_COLUMNS = [
    "gameid",
    "league",
    "date",
    "side",
    "position",
    "teamname",
    "kills",
    *[f"stat_{index:03d}" for index in range(160)],
]


def _oe_csv_bytes(*, rows: int = 200) -> bytes:
    """A body shaped like the real annual export: 167 columns, many rows."""

    lines = [",".join(_COLUMNS)]
    for index in range(rows):
        values = [
            f"g{index}",
            "LCS",
            "2026-08-01T00:00:00Z",
            "Blue",
            "team",
            "Blue Team",
            "12",
            *[str(index)] * 160,
        ]
        lines.append(",".join(values))
    return ("\n".join(lines) + "\n").encode("utf-8")


class _FakeResponse:
    def __init__(
        self,
        body: bytes,
        *,
        url: str,
        content_length: int | None = None,
        final_url: str | None = None,
    ) -> None:
        self._buffer = io.BytesIO(body)
        self._url = final_url or url
        declared = len(body) if content_length is None else content_length
        self.headers = {"Content-Length": str(declared)}
        self.status = 200

    def read(self, amount: int = -1) -> bytes:
        return self._buffer.read(amount)

    def geturl(self) -> str:
        return self._url

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_exc: object) -> bool:
        self._buffer.close()
        return False


def _install_router(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[str], object]
) -> list[str]:
    """Route every urlopen through ``handler`` and record the URLs asked for."""

    calls: list[str] = []

    def fake_urlopen(request: object, timeout: float | None = None) -> object:
        url = getattr(request, "full_url", None) or str(request)
        calls.append(url)
        result = handler(url)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return calls


def _download_handler(bodies: dict[str, bytes], folder: bytes | None = None):
    """Serve per-file-id download bodies and, optionally, a folder listing."""

    def handler(url: str) -> object:
        if "embeddedfolderview" in url:
            if folder is None:
                raise urllib.error.URLError("folder listing unavailable")
            return _FakeResponse(folder, url=url)
        for file_id, body in bodies.items():
            if f"id={file_id}" in url:
                return _FakeResponse(body, url=url)
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]

    return handler


def _destination(tmp_path: Path) -> Path:
    return tmp_path / oe_fetch.annual_csv_name(2026)


def _part_of(destination: Path) -> Path:
    return destination.with_suffix(".part")


# --------------------------------------------------------------------------
# happy path
# --------------------------------------------------------------------------


def test_downloads_the_pinned_file_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _oe_csv_bytes()
    assert len(payload) >= oe_fetch.MIN_CSV_BYTES
    destination = _destination(tmp_path)
    # A staging file left by an earlier killed run must not survive.
    _part_of(destination).write_bytes(b"leftover from a killed run")
    calls = _install_router(
        monkeypatch, _download_handler({PINNED_2026: payload})
    )

    evidence = oe_fetch.fetch_oe_csv(2026, destination)

    assert destination.read_bytes() == payload
    assert not _part_of(destination).exists()
    assert evidence["sha256"] == hashlib.sha256(payload).hexdigest()
    assert evidence["bytes"] == len(payload)
    assert evidence["drive_file_id"] == PINNED_2026
    assert evidence["resolved_from_folder"] is False
    assert evidence["url"].startswith("https://drive.usercontent.google.com/download?")
    assert PINNED_2026 in evidence["url"]
    assert evidence["duration_seconds"] >= 0
    # The evidence must survive a JSON round trip: the launcher logs it.
    assert json.loads(json.dumps(evidence))["bytes"] == len(payload)
    # The pinned id worked, so the folder listing is never fetched.
    assert len(calls) == 1
    assert "embeddedfolderview" not in calls[0]


def test_destination_parent_is_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _oe_csv_bytes()
    destination = tmp_path / "inbox" / oe_fetch.annual_csv_name(2026)
    _install_router(monkeypatch, _download_handler({PINNED_2026: payload}))

    oe_fetch.fetch_oe_csv(2026, destination)

    assert destination.read_bytes() == payload


# --------------------------------------------------------------------------
# interstitials
# --------------------------------------------------------------------------


def test_quota_interstitial_raises_and_never_touches_the_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = _destination(tmp_path)
    sentinel = _oe_csv_bytes(rows=30)
    destination.write_bytes(sentinel)
    _install_router(
        monkeypatch, _download_handler({PINNED_2026: QUOTA_HTML}, folder=QUOTA_HTML)
    )

    with pytest.raises(oe_fetch.OeFetchQuotaError) as excinfo:
        oe_fetch.fetch_oe_csv(2026, destination)

    assert "interstitial" in str(excinfo.value)
    assert "quota" in str(excinfo.value).casefold()
    # Fail-closed: yesterday's good bytes are still there, untouched.
    assert destination.read_bytes() == sentinel
    assert not _part_of(destination).exists()


def test_quota_interstitial_does_not_create_a_missing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = _destination(tmp_path)
    _install_router(
        monkeypatch, _download_handler({PINNED_2026: QUOTA_HTML}, folder=QUOTA_HTML)
    )

    with pytest.raises(oe_fetch.OeFetchQuotaError):
        oe_fetch.fetch_oe_csv(2026, destination)

    assert not destination.exists()
    assert not _part_of(destination).exists()


def test_html_without_a_doctype_is_still_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"\n  <html><body>sign in to continue</body></html>" + b"x" * 20_000
    destination = _destination(tmp_path)
    _install_router(monkeypatch, _download_handler({PINNED_2026: body}))

    with pytest.raises(oe_fetch.OeFetchQuotaError):
        oe_fetch.fetch_oe_csv(2026, destination)

    assert not destination.exists()


def test_first_line_comma_check_rejects_a_narrow_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CSV-ish body with no HTML markers is still not the annual export.

    Nothing here starts with ``<`` and there is no doctype, so only the header
    width test can catch it. The body is deliberately larger than the minimum
    size so the size floor cannot be what rejects it.
    """

    rows = ["gameid,league,date"] + ["g1,LCS,2026-08-01"] * 2000
    body = ("\n".join(rows) + "\n").encode("utf-8")
    assert len(body) > oe_fetch.MIN_CSV_BYTES
    destination = _destination(tmp_path)
    _install_router(monkeypatch, _download_handler({PINNED_2026: body}))

    with pytest.raises(oe_fetch.OeFetchQuotaError) as excinfo:
        oe_fetch.fetch_oe_csv(2026, destination)

    assert "commas" in str(excinfo.value)
    assert not destination.exists()
    assert not _part_of(destination).exists()


def test_empty_body_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = _destination(tmp_path)
    _install_router(monkeypatch, _download_handler({PINNED_2026: b""}))

    with pytest.raises(oe_fetch.OeFetchQuotaError):
        oe_fetch.fetch_oe_csv(2026, destination)

    assert not destination.exists()


# --------------------------------------------------------------------------
# folder re-resolution
# --------------------------------------------------------------------------


def test_folder_listing_recovers_a_rotated_file_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _oe_csv_bytes()
    folder = FOLDER_HTML_TEMPLATE.format(
        id_2025=OE_DRIVE_IDS["2025"], id_2026=ROTATED_2026
    ).encode("utf-8")
    destination = _destination(tmp_path)
    calls = _install_router(
        monkeypatch,
        _download_handler(
            {PINNED_2026: QUOTA_HTML, ROTATED_2026: payload}, folder=folder
        ),
    )

    evidence = oe_fetch.fetch_oe_csv(2026, destination)

    assert destination.read_bytes() == payload
    assert evidence["drive_file_id"] == ROTATED_2026
    assert evidence["resolved_from_folder"] is True
    assert not _part_of(destination).exists()
    assert len(calls) == 3
    assert PINNED_2026 in calls[0]
    assert "embeddedfolderview" in calls[1]
    assert ROTATED_2026 in calls[2]


def test_folder_listing_naming_the_same_id_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = FOLDER_HTML_TEMPLATE.format(
        id_2025=OE_DRIVE_IDS["2025"], id_2026=PINNED_2026
    ).encode("utf-8")
    destination = _destination(tmp_path)
    calls = _install_router(
        monkeypatch, _download_handler({PINNED_2026: QUOTA_HTML}, folder=folder)
    )

    with pytest.raises(oe_fetch.OeFetchQuotaError):
        oe_fetch.fetch_oe_csv(2026, destination)

    assert len(calls) == 2
    assert not destination.exists()


def test_unknown_year_is_resolved_from_the_folder_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A year with no pinned id is a normal state at a season rollover."""

    assert "2031" not in oe_fetch.DEFAULT_DRIVE_FILE_IDS
    payload = _oe_csv_bytes()
    folder = (
        '<div id="entry-1FutureSeasonFileId_00000000">'
        '<span class="flip-entry-title">'
        "2031_LoL_esports_match_data_from_OraclesElixir.csv</span></div>"
    ).encode("utf-8")
    destination = tmp_path / oe_fetch.annual_csv_name(2031)
    _install_router(
        monkeypatch,
        _download_handler({"1FutureSeasonFileId_00000000": payload}, folder=folder),
    )

    evidence = oe_fetch.fetch_oe_csv(2031, destination)

    assert evidence["resolved_from_folder"] is True
    assert destination.read_bytes() == payload


def test_unknown_year_absent_from_the_folder_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = FOLDER_HTML_TEMPLATE.format(
        id_2025=OE_DRIVE_IDS["2025"], id_2026=PINNED_2026
    ).encode("utf-8")
    destination = tmp_path / oe_fetch.annual_csv_name(2031)
    _install_router(monkeypatch, _download_handler({}, folder=folder))

    with pytest.raises(oe_fetch.OeFetchError) as excinfo:
        oe_fetch.fetch_oe_csv(2031, destination)

    assert not isinstance(excinfo.value, oe_fetch.OeFetchQuotaError)
    assert "no Drive file id is known" in str(excinfo.value)
    assert not destination.exists()


def test_parse_folder_listing_pairs_each_name_with_its_own_id() -> None:
    document = FOLDER_HTML_TEMPLATE.format(
        id_2025=OE_DRIVE_IDS["2025"], id_2026=PINNED_2026
    )

    entries = oe_fetch.parse_folder_listing(document)

    assert entries == {
        "2025_LoL_esports_match_data_from_OraclesElixir.csv": OE_DRIVE_IDS["2025"],
        "2026_LoL_esports_match_data_from_OraclesElixir.csv": PINNED_2026,
    }


def test_parse_folder_listing_ignores_junk_entries() -> None:
    document = (
        '<div id="entry-short"><span class="flip-entry-title">a.csv</span></div>'
        '<div id="entry-1ValidLookingFileId_0001"><span>no title here</span></div>'
    )

    assert oe_fetch.parse_folder_listing(document) == {}


# --------------------------------------------------------------------------
# transport failures
# --------------------------------------------------------------------------


def test_truncated_body_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _oe_csv_bytes()
    destination = _destination(tmp_path)

    def handler(url: str) -> object:
        if "embeddedfolderview" in url:
            raise urllib.error.URLError("no listing")
        return _FakeResponse(payload, url=url, content_length=len(payload) + 5_000)

    _install_router(monkeypatch, handler)

    with pytest.raises(oe_fetch.OeFetchError) as excinfo:
        oe_fetch.fetch_oe_csv(2026, destination)

    assert not isinstance(excinfo.value, oe_fetch.OeFetchQuotaError)
    assert "truncated" in str(excinfo.value)
    assert not destination.exists()
    assert not _part_of(destination).exists()


def test_body_below_the_size_floor_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _oe_csv_bytes(rows=1)
    assert len(payload) < oe_fetch.MIN_CSV_BYTES
    destination = _destination(tmp_path)
    _install_router(monkeypatch, _download_handler({PINNED_2026: payload}))

    with pytest.raises(oe_fetch.OeFetchError) as excinfo:
        oe_fetch.fetch_oe_csv(2026, destination)

    assert "too small" in str(excinfo.value)
    assert not destination.exists()
    assert not _part_of(destination).exists()


def test_a_redirect_off_google_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An open redirect must not turn this into a third-party mirror client."""

    payload = _oe_csv_bytes()
    folder = FOLDER_HTML_TEMPLATE.format(
        id_2025=OE_DRIVE_IDS["2025"], id_2026=PINNED_2026
    ).encode("utf-8")
    destination = _destination(tmp_path)

    def handler(url: str) -> object:
        if "embeddedfolderview" in url:
            return _FakeResponse(folder, url=url)
        return _FakeResponse(
            payload, url=url, final_url="https://oracleselixir.example.net/2026.csv"
        )

    _install_router(monkeypatch, handler)

    with pytest.raises(oe_fetch.OeFetchError) as excinfo:
        oe_fetch.fetch_oe_csv(2026, destination)

    assert "off Google" in str(excinfo.value)
    assert not destination.exists()
    assert not _part_of(destination).exists()


def test_network_failure_is_not_reported_as_a_quota_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = _destination(tmp_path)

    def handler(url: str) -> object:
        raise urllib.error.URLError("network is unreachable")

    _install_router(monkeypatch, handler)

    with pytest.raises(oe_fetch.OeFetchError) as excinfo:
        oe_fetch.fetch_oe_csv(2026, destination)

    assert not isinstance(excinfo.value, oe_fetch.OeFetchQuotaError)
    assert not destination.exists()


def test_download_url_rejects_a_junk_file_id() -> None:
    with pytest.raises(oe_fetch.OeFetchError):
        oe_fetch.download_url("../../etc/passwd")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_exits_zero_and_prints_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _oe_csv_bytes()
    destination = _destination(tmp_path)
    _install_router(monkeypatch, _download_handler({PINNED_2026: payload}))

    code = oe_fetch.main(["--year", "2026", "--destination", str(destination)])

    assert code == 0
    evidence = json.loads(capsys.readouterr().out.strip())
    assert evidence["status"] == "downloaded"
    assert evidence["sha256"] == hashlib.sha256(payload).hexdigest()
    assert evidence["destination"] == str(destination)


def test_cli_exits_75_on_a_quota_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = _destination(tmp_path)
    _install_router(
        monkeypatch, _download_handler({PINNED_2026: QUOTA_HTML}, folder=QUOTA_HTML)
    )

    code = oe_fetch.main(["--year", "2026", "--destination", str(destination)])

    assert code == oe_fetch.EXIT_QUOTA_BLOCKED == 75
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err.strip())["status"] == "quota_blocked"
    assert not destination.exists()
    assert not _part_of(destination).exists()


def test_cli_exits_one_on_a_transport_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = _destination(tmp_path)

    def handler(url: str) -> object:
        raise urllib.error.URLError("network is unreachable")

    _install_router(monkeypatch, handler)

    code = oe_fetch.main(["--year", "2026", "--destination", str(destination)])

    assert code == oe_fetch.EXIT_FAILED == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err.strip())["status"] == "failed"
    assert not destination.exists()


# --------------------------------------------------------------------------
# drift guards
# --------------------------------------------------------------------------


def test_pinned_ids_are_the_shared_map_not_a_second_copy() -> None:
    """Two independent OE resolvers have drifted apart in this repo before."""

    assert oe_fetch.DEFAULT_DRIVE_FILE_IDS == dict(OE_DRIVE_IDS)
    assert oe_fetch.DEFAULT_DRIVE_FILE_IDS["2026"] == PINNED_2026


def test_size_floor_matches_the_ingest_validator() -> None:
    from lol_kills.etl.oe_ingest import OE_MIN_DOWNLOAD_BYTES

    assert oe_fetch.MIN_CSV_BYTES == OE_MIN_DOWNLOAD_BYTES


def test_the_folder_id_is_the_official_oracles_elixir_folder() -> None:
    from lol_kills.etl.paths import OE_FOLDER

    assert oe_fetch.OE_FOLDER_ID == "1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH"
    assert oe_fetch.OE_FOLDER_ID in OE_FOLDER
    assert oe_fetch.folder_listing_url().startswith(
        "https://drive.google.com/embeddedfolderview?"
    )
