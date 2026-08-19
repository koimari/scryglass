"""Behaviour tests for the OAuth "make a copy" Oracle's Elixir downloader.

No test here touches the network, the Keychain, or a Google account.
``urllib.request.urlopen`` is replaced by a router that answers from in-memory
fixtures, and the Keychain read is replaced at its single seam, which lets the
awkward cases - a dead refresh token, a copy refused for quota, an interstitial
served through the API - be exercised deterministically.

Three properties are under test throughout:

* **The temporary Drive copy is always deleted.**  It is created in the
  operator's own Drive, so a leak is litter that accumulates every six hours.
* **Fail-closed.**  ``destination`` never receives anything but a fully
  validated body, and no staging file survives a failure.
* **No credential material is ever printed.**  Both streams are captured and
  asserted against the exact secrets the fixtures used.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

from lol_kills.etl import oe_fetch, oe_oauth_fetch
from lol_kills.etl.paths import OE_DRIVE_IDS

PINNED_2026 = OE_DRIVE_IDS["2026"]
ROTATED_2026 = "1RotatedIdForTheAnnualExport_0000000"
COPY_ID = "1TemporaryCopyIdInTheOwnersDrive_000"

# Deliberately shaped like the real thing so the redaction test exercises both
# the exact-value path and the pattern path.
CLIENT_ID = "1234567890-abcdefghijklmnop.apps.googleusercontent.com"
CLIENT_SECRET = "GOCSPX-ThisIsTheClientSecretValue"
REFRESH_TOKEN = "1//0gThisIsTheStoredRefreshTokenValue"
ACCESS_TOKEN = "ya29.a0ThisIsTheMintedAccessTokenValue"

SECRETS = (CLIENT_SECRET, REFRESH_TOKEN, ACCESS_TOKEN)

LOGIN_JSON = json.dumps(
    {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "refresh_token": REFRESH_TOKEN,
    }
)

QUOTA_HTML = (
    b"<!DOCTYPE html><html><head><title>Google Drive - Quota exceeded</title>"
    b"</head><body><p>Sorry, you can't view or download this file at this "
    b"time.</p></body></html>"
)

FOLDER_HTML = f"""<html><body>
<div class="flip-entry" id="entry-{ROTATED_2026}">
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
        lines.append(
            ",".join(
                [
                    f"g{index}",
                    "LCS",
                    "2026-08-01T00:00:00Z",
                    "Blue",
                    "team",
                    "Blue Team",
                    "12",
                    *[str(index)] * 160,
                ]
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


CSV_BODY = _oe_csv_bytes()
CSV_SHA256 = hashlib.sha256(CSV_BODY).hexdigest()


# ---------------------------------------------------------------------------
# Transport doubles
# ---------------------------------------------------------------------------


@dataclass
class _Call:
    method: str
    url: str
    body: Any = None


class _FakeResponse:
    def __init__(self, body: bytes, *, url: str, content_length: int | None = None):
        self._buffer = io.BytesIO(body)
        self._url = url
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


def _json_response(payload: dict[str, Any], url: str) -> _FakeResponse:
    return _FakeResponse(json.dumps(payload).encode("utf-8"), url=url)


def _http_error(url: str, code: int, payload: dict[str, Any]) -> urllib.error.HTTPError:
    body = json.dumps(payload).encode("utf-8")
    return urllib.error.HTTPError(url, code, "error", {}, io.BytesIO(body))


def _drive_error(reason: str, message: str) -> dict[str, Any]:
    return {"error": {"errors": [{"reason": reason, "message": message}], "message": message}}


def _install_router(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[_Call], object]
) -> list[_Call]:
    """Route every urlopen through ``handler`` and record what was asked for."""

    calls: list[_Call] = []

    def fake_urlopen(request: Any, timeout: float | None = None) -> object:
        url = getattr(request, "full_url", None) or str(request)
        method = "GET"
        getter = getattr(request, "get_method", None)
        if callable(getter):
            method = getter()
        raw = getattr(request, "data", None)
        body: Any = None
        if raw:
            text = raw.decode("utf-8")
            try:
                body = json.loads(text)
            except ValueError:
                body = text
        call = _Call(method=method, url=url, body=body)
        calls.append(call)
        result = handler(call)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return calls


def _default_handler(
    *,
    csv_body: bytes = CSV_BODY,
    copy_result: Callable[[_Call], object] | None = None,
    media_result: Callable[[_Call], object] | None = None,
    token_result: Callable[[_Call], object] | None = None,
) -> Callable[[_Call], object]:
    """A router that walks the whole happy path unless a stage is overridden.

    Overrides are FACTORIES, not values, and deliberately so: the resolution
    ladder tries a second source id after the first fails, and a single shared
    response or HTTPError would arrive at the second attempt already read to
    EOF - which silently turned a quota refusal into an unclassifiable one and
    would have made these tests pass for the wrong reason.
    """

    def handler(call: _Call) -> object:
        if "oauth2.googleapis.com/token" in call.url:
            if token_result is not None:
                return token_result(call)
            return _json_response(
                {
                    "access_token": ACCESS_TOKEN,
                    "expires_in": 3599,
                    "scope": oe_oauth_fetch.DRIVE_SCOPE,
                    "token_type": "Bearer",
                },
                call.url,
            )
        if call.url.endswith("/copy") or "/copy?" in call.url:
            if copy_result is not None:
                return copy_result(call)
            return _json_response({"id": COPY_ID}, call.url)
        if "alt=media" in call.url:
            if media_result is not None:
                return media_result(call)
            return _FakeResponse(csv_body, url=call.url)
        if call.method == "DELETE":
            return _FakeResponse(b"", url=call.url)
        if "embeddedfolderview" in call.url:
            return _FakeResponse(FOLDER_HTML.encode("utf-8"), url=call.url)
        raise AssertionError(f"unexpected request: {call.method} {call.url}")

    return handler


@pytest.fixture(autouse=True)
def _isolate_secret_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the redaction registry from leaking between tests.

    ``_known_secrets`` is process-global by design - once this process has seen
    a credential, nothing it prints may contain it again - so each test gets a
    fresh copy rather than inheriting the previous test's fixtures.
    """

    monkeypatch.setattr(oe_oauth_fetch, "_known_secrets", set())


@pytest.fixture
def stored_login(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer the Keychain read at its single seam, with no Keychain involved."""

    monkeypatch.delenv(oe_oauth_fetch.KEYCHAIN_ENV_OVERRIDE, raising=False)
    monkeypatch.setattr(oe_oauth_fetch, "_read_keychain_blob", lambda: LOGIN_JSON)


def _run(destination: Path, *, year: int = 2026) -> int:
    return oe_oauth_fetch.main(
        ["--year", str(year), "--destination", str(destination)]
    )


# ---------------------------------------------------------------------------
# Stored login
# ---------------------------------------------------------------------------


def test_env_override_is_read_before_the_keychain(monkeypatch: pytest.MonkeyPatch) -> None:
    def _explode() -> str:
        raise AssertionError("the Keychain must not be consulted when the override is set")

    monkeypatch.setattr(oe_oauth_fetch, "_read_keychain_blob", _explode)
    monkeypatch.setenv(oe_oauth_fetch.KEYCHAIN_ENV_OVERRIDE, LOGIN_JSON)

    login = oe_oauth_fetch.read_stored_login()

    assert login["client_id"] == CLIENT_ID
    assert login["refresh_token"] == REFRESH_TOKEN


def test_blank_env_override_falls_through_to_the_keychain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(oe_oauth_fetch.KEYCHAIN_ENV_OVERRIDE, "   ")
    monkeypatch.setattr(oe_oauth_fetch, "_read_keychain_blob", lambda: LOGIN_JSON)

    assert oe_oauth_fetch.read_stored_login()["client_id"] == CLIENT_ID


def test_no_stored_login_exits_69(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(oe_oauth_fetch.KEYCHAIN_ENV_OVERRIDE, raising=False)

    def _absent() -> str:
        raise oe_oauth_fetch.OeOauthUnconfigured("no OAuth login stored; run --login")

    monkeypatch.setattr(oe_oauth_fetch, "_read_keychain_blob", _absent)
    _install_router(monkeypatch, lambda call: AssertionError("no request expected"))

    destination = tmp_path / "stage.csv"
    assert _run(destination) == oe_oauth_fetch.EXIT_UNCONFIGURED
    assert not destination.exists()

    reported = json.loads(capsys.readouterr().err.strip())
    assert reported["status"] == "unconfigured"
    assert "--login" in reported["error"]


def test_a_login_missing_a_field_exits_69(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(oe_oauth_fetch.KEYCHAIN_ENV_OVERRIDE, raising=False)
    monkeypatch.setattr(
        oe_oauth_fetch,
        "_read_keychain_blob",
        lambda: json.dumps({"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET}),
    )
    _install_router(monkeypatch, lambda call: AssertionError("no request expected"))

    assert _run(tmp_path / "stage.csv") == oe_oauth_fetch.EXIT_UNCONFIGURED
    reported = json.loads(capsys.readouterr().err.strip())
    assert "refresh_token" in reported["error"]
    # The names of the missing fields are reportable; the values of the present
    # ones are not.
    assert CLIENT_SECRET not in reported["error"]


# ---------------------------------------------------------------------------
# Token refresh
# ---------------------------------------------------------------------------


def test_refresh_access_token_returns_the_token_and_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_router(monkeypatch, _default_handler())

    token, scope = oe_oauth_fetch.refresh_access_token(
        {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "refresh_token": REFRESH_TOKEN,
        }
    )

    assert token == ACCESS_TOKEN
    assert scope == oe_oauth_fetch.DRIVE_SCOPE
    assert calls[0].method == "POST"
    assert calls[0].url == "https://oauth2.googleapis.com/token"
    assert "grant_type=refresh_token" in calls[0].body


def test_invalid_grant_exits_69(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stored_login: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def dead(call: _Call) -> object:
        return _http_error(
            call.url,
            400,
            {
                "error": "invalid_grant",
                "error_description": "Token has been expired or revoked.",
            },
        )

    _install_router(monkeypatch, _default_handler(token_result=dead))

    destination = tmp_path / "stage.csv"
    assert _run(destination) == oe_oauth_fetch.EXIT_UNCONFIGURED
    assert not destination.exists()

    reported = json.loads(capsys.readouterr().err.strip())
    assert reported["status"] == "unconfigured"
    assert "--login" in reported["error"]


def test_a_broken_token_endpoint_is_not_reported_as_unconfigured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stored_login: None
) -> None:
    """A 500 is a fault, not a dead login: exit 1 so nothing reads it as 'skip'."""

    def broken(call: _Call) -> object:
        return _http_error(call.url, 500, {"error": "backendError"})

    _install_router(monkeypatch, _default_handler(token_result=broken))

    assert _run(tmp_path / "stage.csv") == oe_oauth_fetch.EXIT_FAILED


# ---------------------------------------------------------------------------
# Copy -> download -> delete
# ---------------------------------------------------------------------------


def test_happy_path_copies_downloads_and_deletes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stored_login: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = _install_router(monkeypatch, _default_handler())
    destination = tmp_path / "stage.csv"

    assert _run(destination) == oe_oauth_fetch.EXIT_OK

    # The bytes landed, atomically, with no staging file left behind.
    assert destination.read_bytes() == CSV_BODY
    assert not (tmp_path / "stage.part").exists()
    assert sorted(item.name for item in tmp_path.iterdir()) == ["stage.csv"]

    methods = [(call.method, call.url) for call in calls]
    assert methods[0][0] == "POST" and "oauth2.googleapis.com/token" in methods[0][1]
    assert methods[1][0] == "POST" and methods[1][1].startswith(
        f"https://www.googleapis.com/drive/v3/files/{PINNED_2026}/copy"
    )
    assert methods[2][0] == "GET" and "alt=media" in methods[2][1]
    # The copy is deleted, by id, and it is the last thing that happens.
    assert methods[3] == (
        "DELETE",
        f"https://www.googleapis.com/drive/v3/files/{COPY_ID}",
    )
    assert len(methods) == 4

    # The copy was given the distinctive throwaway name.
    assert calls[1].body["name"].startswith("scryglass-tmp-oe-2026-")

    evidence = json.loads(capsys.readouterr().out.strip())
    assert evidence["status"] == "downloaded"
    assert evidence["transport"] == "oauth_drive_copy"
    assert evidence["copy_used"] is True
    assert evidence["copy_deleted"] is True
    # The SOURCE id, not the ephemeral copy: that is what the receipt binds to.
    assert evidence["drive_file_id"] == PINNED_2026
    assert COPY_ID not in json.dumps(evidence)
    assert evidence["bytes"] == len(CSV_BODY)
    assert evidence["sha256"] == CSV_SHA256


def test_the_destination_is_written_by_rename_not_in_place(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stored_login: None
) -> None:
    """The staging file is what gets streamed; the destination only ever appears
    whole, by ``os.replace``."""

    seen: list[tuple[str, str]] = []
    real_replace = oe_oauth_fetch.commit_staged_download

    def _watched(part_path: Path, destination: Path, **kwargs: Any) -> None:
        # At this moment the bytes are all in the staging file and the
        # destination does not exist yet.
        seen.append((part_path.name, destination.name))
        assert part_path.exists()
        assert not destination.exists()
        return real_replace(part_path, destination, **kwargs)

    monkeypatch.setattr(oe_oauth_fetch, "commit_staged_download", _watched)
    _install_router(monkeypatch, _default_handler())

    destination = tmp_path / "stage.csv"
    assert _run(destination) == oe_oauth_fetch.EXIT_OK
    assert seen == [("stage.part", "stage.csv")]


def test_copy_refused_for_quota_exits_75(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stored_login: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def refusal(call: _Call) -> object:
        return _http_error(
            call.url,
            403,
            _drive_error(
                "downloadQuotaExceeded",
                "The download quota for this file has been exceeded.",
            ),
        )

    # Both source ids are refused, so the run ends on a quota answer.
    calls = _install_router(monkeypatch, _default_handler(copy_result=refusal))

    destination = tmp_path / "stage.csv"
    assert _run(destination) == oe_oauth_fetch.EXIT_QUOTA_BLOCKED
    assert not destination.exists()
    assert not (tmp_path / "stage.part").exists()

    # No copy was ever created, so there is nothing to delete.
    assert not [call for call in calls if call.method == "DELETE"]

    reported = json.loads(capsys.readouterr().err.strip())
    assert reported["status"] == "quota_blocked"


def test_cannot_copy_file_exits_75(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stored_login: None
) -> None:
    def refusal(call: _Call) -> object:
        return _http_error(
            call.url, 403, _drive_error("cannotCopyFile", "The file cannot be copied.")
        )

    _install_router(monkeypatch, _default_handler(copy_result=refusal))

    assert _run(tmp_path / "stage.csv") == oe_oauth_fetch.EXIT_QUOTA_BLOCKED


def test_a_copy_that_exists_is_deleted_even_when_the_download_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stored_login: None
) -> None:
    """The copy is created, then the media read is refused: cleanup must still run."""

    def refusal(call: _Call) -> object:
        return _http_error(
            call.url, 403, _drive_error("downloadQuotaExceeded", "quota exceeded")
        )

    calls = _install_router(monkeypatch, _default_handler(media_result=refusal))

    destination = tmp_path / "stage.csv"
    assert _run(destination) == oe_oauth_fetch.EXIT_QUOTA_BLOCKED
    assert not destination.exists()

    deletes = [call for call in calls if call.method == "DELETE"]
    # One per attempted source id, and each names the copy it created.
    assert deletes
    assert all(call.url.endswith(f"/files/{COPY_ID}") for call in deletes)


def test_an_interstitial_through_the_api_writes_nothing_and_deletes_the_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stored_login: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """HTTP 200 plus an HTML body is the exact failure this whole module exists
    to survive; it must not become the annual CSV."""

    calls = _install_router(monkeypatch, _default_handler(csv_body=QUOTA_HTML))

    destination = tmp_path / "stage.csv"
    status = _run(destination)

    assert status != oe_oauth_fetch.EXIT_OK
    assert status == oe_oauth_fetch.EXIT_QUOTA_BLOCKED
    assert not destination.exists()
    assert not (tmp_path / "stage.part").exists()
    assert list(tmp_path.iterdir()) == []

    deletes = [call for call in calls if call.method == "DELETE"]
    assert deletes and all(call.url.endswith(f"/files/{COPY_ID}") for call in deletes)

    reported = json.loads(capsys.readouterr().err.strip())
    assert "interstitial" in reported["error"]


def test_a_truncated_body_writes_nothing_and_deletes_the_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stored_login: None
) -> None:
    """A body that disagrees with its own Content-Length is refused."""

    def truncated(call: _Call) -> object:
        return _FakeResponse(
            CSV_BODY[: len(CSV_BODY) // 2],
            url=call.url,
            content_length=len(CSV_BODY),
        )

    calls = _install_router(monkeypatch, _default_handler(media_result=truncated))

    destination = tmp_path / "stage.csv"
    assert _run(destination) == oe_oauth_fetch.EXIT_FAILED
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []
    assert [call for call in calls if call.method == "DELETE"]


def test_a_rotated_source_id_is_resolved_from_the_official_folder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stored_login: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The pinned id 404s; the folder listing names the current one."""

    def handler(call: _Call) -> object:
        if f"/files/{PINNED_2026}/copy" in call.url:
            return _http_error(call.url, 404, _drive_error("notFound", "File not found."))
        return _default_handler()(call)

    calls = _install_router(monkeypatch, handler)

    destination = tmp_path / "stage.csv"
    assert _run(destination) == oe_oauth_fetch.EXIT_OK
    assert destination.read_bytes() == CSV_BODY

    assert any("embeddedfolderview" in call.url for call in calls)
    assert any(f"/files/{ROTATED_2026}/copy" in call.url for call in calls)

    evidence = json.loads(capsys.readouterr().out.strip())
    assert evidence["drive_file_id"] == ROTATED_2026
    assert evidence["resolved_from_folder"] is True


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def _assert_no_secrets(text: str) -> None:
    for secret in SECRETS:
        assert secret not in text, f"credential material reached the output: {secret[:6]}..."
    # Nothing that merely looks like a credential either.
    for marker in ("ya29.", "GOCSPX-", "1//0g"):
        assert marker not in text


def test_no_token_material_reaches_either_stream_on_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stored_login: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_router(monkeypatch, _default_handler())

    assert _run(tmp_path / "stage.csv") == oe_oauth_fetch.EXIT_OK

    captured = capsys.readouterr()
    _assert_no_secrets(captured.out)
    _assert_no_secrets(captured.err)


def test_no_token_material_reaches_either_stream_when_google_echoes_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stored_login: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The nastiest case: an error body that quotes the credential back at us."""

    def echoed(call: _Call) -> object:
        return _http_error(
            call.url,
            400,
            _drive_error(
                "badRequest",
                f"Invalid request with token {ACCESS_TOKEN} for client "
                f"{CLIENT_SECRET} and refresh {REFRESH_TOKEN}",
            ),
        )

    _install_router(monkeypatch, _default_handler(copy_result=echoed))

    assert _run(tmp_path / "stage.csv") == oe_oauth_fetch.EXIT_FAILED

    captured = capsys.readouterr()
    _assert_no_secrets(captured.out)
    _assert_no_secrets(captured.err)
    # The error is still reported - redaction must not swallow the diagnosis.
    assert "Invalid request with token" in captured.err


def test_the_callback_handler_does_not_log_the_authorization_code() -> None:
    """BaseHTTPRequestHandler's default access log writes the full query string,
    which carries ``?code=...``.  It must be silenced."""

    assert (
        oe_oauth_fetch._CallbackHandler.log_message
        is not __import__("http.server", fromlist=["x"]).BaseHTTPRequestHandler.log_message
    )


# ---------------------------------------------------------------------------
# Shared fail-closed machinery
# ---------------------------------------------------------------------------


def test_the_validation_is_oe_fetch_s_and_not_a_second_copy() -> None:
    """The point of the import: one interstitial check, not two that can drift."""

    assert oe_oauth_fetch.stream_validated_body is oe_fetch.stream_validated_body
    assert oe_oauth_fetch.commit_staged_download is oe_fetch.commit_staged_download
    assert oe_oauth_fetch._require_csv_head is oe_fetch._require_csv_head
    assert oe_oauth_fetch.staging_path is oe_fetch.staging_path


# ---------------------------------------------------------------------------
# Launcher ladder
# ---------------------------------------------------------------------------

LAUNCHER = Path(__file__).resolve().parents[1] / "ops/launchd/run-public-refresh.sh"


def test_the_launcher_places_the_oauth_rung_between_headless_and_reuse() -> None:
    """The ladder order is the whole point of the change, so pin it here.

    anonymous headless -> OAuth Drive copy -> cached inbox reuse -> Brave.
    """

    script = LAUNCHER.read_text(encoding="utf-8")

    headless = script.index("lol_kills.etl.oe_fetch")
    oauth = script.index("lol_kills.etl.oe_oauth_fetch")
    reuse = script.index('SCRYGLASS_OE_TRANSPORT="cached_inbox_reuse"')
    brave = script.index('SCRYGLASS_OE_TRANSPORT="brave_origin_browser_download"')

    assert headless < oauth < reuse < brave

    # Exit 69 must not be treated as a failure of the feed.
    assert "oe_oauth_status == 69" in script or "69)" in script
    assert 'SCRYGLASS_OE_TRANSPORT="oauth_drive_copy"' in script
