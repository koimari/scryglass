from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from lol_kills.etl import oe_ingest


def _csv_bytes(*, date: str, gameid: str = "g1") -> bytes:
    rows = [
        "gameid,league,date,side,position,teamname,kills",
        f"{gameid},LCS,{date},Blue,team,Blue Team,12",
        f"{gameid},LCS,{date},Red,team,Red Team,8",
        f"{gameid},LCS,{date},Blue,top,Blue Team,2",
        f"{gameid},LCS,{date},Red,top,Red Team,1",
    ]
    return ("\n".join(rows) + "\n").encode("utf-8")


def _configure_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    receipts = tmp_path / "receipts"
    raw.mkdir()
    monkeypatch.setattr(oe_ingest, "RAW_OE_DIR", raw)
    monkeypatch.setattr(oe_ingest, "OE_RECEIPT_DIR", receipts)
    monkeypatch.setattr(oe_ingest, "OE_DRIVE_IDS", {"2026": "public-file-id"})
    monkeypatch.setattr(oe_ingest, "OE_MIN_DOWNLOAD_BYTES", 1)
    return raw, receipts


def _install_fake_gdown(monkeypatch: pytest.MonkeyPatch, download: object) -> None:
    monkeypatch.setitem(sys.modules, "gdown", SimpleNamespace(download=download))


def test_download_mode_keeps_valid_existing_bytes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    raw, receipts = _configure_paths(monkeypatch, tmp_path)
    destination = raw / "2026_LoL_esports_match_data_from_OraclesElixir.csv"
    original = _csv_bytes(date="2026-06-14T22:24:48Z")
    destination.write_bytes(original)

    def unexpected_download(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("validated cache must not be downloaded without force")

    _install_fake_gdown(monkeypatch, unexpected_download)
    assert oe_ingest.download_oe_years(["2026"]) == [destination]
    assert destination.read_bytes() == original
    assert not receipts.exists()


def test_download_mode_uses_cached_remote_signature(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    raw, _receipts = _configure_paths(monkeypatch, tmp_path)
    destination = raw / "2026_LoL_esports_match_data_from_OraclesElixir.csv"
    original = _csv_bytes(date="2026-06-14T22:24:48Z")
    destination.write_bytes(original)
    signature = {"bytes": len(original), "last_modified": "Mon, 10 Aug 2026 07:04:22 GMT"}
    oe_ingest._write_remote_state({"2026": signature})
    monkeypatch.setattr(oe_ingest, "_remote_file_signature", lambda _url: signature)

    def unexpected_download(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unchanged remote file must use the validated cache")

    _install_fake_gdown(monkeypatch, unexpected_download)
    assert oe_ingest.download_oe_years(["2026"]) == [destination]
    assert destination.read_bytes() == original
    state = json.loads((tmp_path / "receipts/remote-state.json").read_text())
    assert state["2026"]["bytes"] == len(original)
    assert state["2026"]["checked_at_utc"]


def test_download_mode_refreshes_when_remote_signature_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw, _receipts = _configure_paths(monkeypatch, tmp_path)
    destination = raw / "2026_LoL_esports_match_data_from_OraclesElixir.csv"
    original = _csv_bytes(date="2026-06-14T22:24:48Z")
    replacement = _csv_bytes(date="2026-07-26T22:00:00Z", gameid="g2")
    destination.write_bytes(original)
    oe_ingest._write_remote_state(
        {"2026": {"bytes": len(original), "last_modified": "old"}}
    )
    monkeypatch.setattr(
        oe_ingest,
        "_remote_file_signature",
        lambda _url: {"bytes": len(replacement), "last_modified": "new"},
    )

    def fake_download(_url: str, output: str, quiet: bool) -> str:
        assert quiet is False
        Path(output).write_bytes(replacement)
        return output

    _install_fake_gdown(monkeypatch, fake_download)
    assert oe_ingest.download_oe_years(["2026"]) == [destination]
    assert destination.read_bytes() == replacement
    state = json.loads((tmp_path / "receipts/remote-state.json").read_text())
    assert state["2026"]["bytes"] == len(replacement)


def test_refresh_stages_archives_and_receipts_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw, receipts = _configure_paths(monkeypatch, tmp_path)
    destination = raw / "2026_LoL_esports_match_data_from_OraclesElixir.csv"
    original = _csv_bytes(date="2026-06-14T22:24:48Z")
    replacement = _csv_bytes(date="2026-07-26T22:00:00Z", gameid="g2")
    destination.write_bytes(original)

    def fake_download(_url: str, output: str, quiet: bool) -> str:
        assert quiet is False
        Path(output).write_bytes(replacement)
        return output

    _install_fake_gdown(monkeypatch, fake_download)
    assert oe_ingest.download_oe_years([2026], force=True) == [destination]
    assert destination.read_bytes() == replacement

    old_sha = hashlib.sha256(original).hexdigest()
    archive = raw / "archive" / f"{destination.stem}.{old_sha}.csv"
    assert archive.read_bytes() == original

    receipt_paths = list(receipts.glob("*.json"))
    assert len(receipt_paths) == 1
    receipt = json.loads(receipt_paths[0].read_text())
    claimed = receipt.pop("receipt_canonical_sha256")
    assert claimed == oe_ingest._canonical_sha256(receipt)
    assert receipt["schema"] == "scryglass:oe-source-refresh:v1"
    assert receipt["status"] == "refreshed"
    assert receipt["candidate"]["raw_sha256"] == hashlib.sha256(replacement).hexdigest()
    assert receipt["candidate"]["date_max_utc"] == "2026-07-26T22:00:00+00:00"
    assert receipt["previous"]["raw_sha256"] == old_sha
    assert receipt["previous"]["archive_locator"].endswith(archive.name)
    assert receipt["authority"]["betting_authority"] is False
    loaded = oe_ingest.load_refresh_receipt(receipt_paths[0])
    assert loaded["receipt_canonical_sha256"] == claimed

    tampered = json.loads(receipt_paths[0].read_text())
    tampered["authority"]["betting_authority"] = True
    receipt_paths[0].write_text(json.dumps(tampered))
    with pytest.raises(oe_ingest.OeDownloadError):
        oe_ingest.load_refresh_receipt(receipt_paths[0])


@pytest.mark.parametrize(
    "replacement",
    [
        b"<!doctype html><html>quota</html>",
        _csv_bytes(date="2026-05-01T00:00:00Z", gameid="older"),
    ],
)
def test_failed_strict_refresh_preserves_existing_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    replacement: bytes,
) -> None:
    raw, receipts = _configure_paths(monkeypatch, tmp_path)
    destination = raw / "2026_LoL_esports_match_data_from_OraclesElixir.csv"
    original = _csv_bytes(date="2026-06-14T22:24:48Z")
    destination.write_bytes(original)

    def fake_download(_url: str, output: str, quiet: bool) -> str:
        assert quiet is False
        Path(output).write_bytes(replacement)
        return output

    _install_fake_gdown(monkeypatch, fake_download)
    with pytest.raises(oe_ingest.OeDownloadError):
        oe_ingest.download_oe_years(["2026"], force=True)

    assert destination.read_bytes() == original
    assert not (raw / "archive").exists()
    assert not receipts.exists()
    assert not list(raw.glob("*.download"))


def test_browser_download_is_validated_archived_and_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw, receipts = _configure_paths(monkeypatch, tmp_path)
    destination = raw / "2026_LoL_esports_match_data_from_OraclesElixir.csv"
    original = _csv_bytes(date="2026-06-14T22:24:48Z")
    replacement = _csv_bytes(date="2026-08-11T10:50:41Z", gameid="g2")
    destination.write_bytes(original)
    browser_file = tmp_path / "browser.csv"
    browser_file.write_bytes(replacement)

    installed = oe_ingest.install_browser_download(browser_file, 2026)

    assert installed == destination
    assert destination.read_bytes() == replacement
    old_sha = hashlib.sha256(original).hexdigest()
    archive = raw / "archive" / f"{destination.stem}.{old_sha}.csv"
    assert archive.read_bytes() == original
    receipt_path = next(receipts.glob("*.json"))
    receipt = oe_ingest.load_refresh_receipt(receipt_path)
    assert receipt["source"]["transport"] == "brave_origin_browser_download"
    assert receipt["candidate"]["date_max_utc"] == "2026-08-11T10:50:41+00:00"

    accepted = oe_ingest.validate_accepted_source_receipt(receipt_path, destination, 2026)
    assert accepted["raw_sha256"] == hashlib.sha256(replacement).hexdigest()
    destination.write_bytes(_csv_bytes(date="2026-08-11T10:50:41Z", gameid="changed"))
    with pytest.raises(oe_ingest.OeDownloadError, match="does not bind"):
        oe_ingest.validate_accepted_source_receipt(receipt_path, destination, 2026)


def test_browser_download_refuses_temporal_regression(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw, _receipts = _configure_paths(monkeypatch, tmp_path)
    destination = raw / "2026_LoL_esports_match_data_from_OraclesElixir.csv"
    original = _csv_bytes(date="2026-08-11T10:50:41Z")
    destination.write_bytes(original)
    browser_file = tmp_path / "browser.csv"
    browser_file.write_bytes(_csv_bytes(date="2026-08-10T22:03:59Z"))

    with pytest.raises(oe_ingest.OeDownloadError, match="date_max regressed"):
        oe_ingest.install_browser_download(browser_file, 2026)

    assert destination.read_bytes() == original


def test_parse_normalizes_numeric_playoffs_to_nullable_boolean(tmp_path: Path) -> None:
    path = tmp_path / "2026_LoL_esports_match_data_from_OraclesElixir.csv"
    path.write_text(
        "gameid,date,position,teamname,champion,playoffs\n"
        "g1,2026-01-01T00:00:00Z,team,Blue Team,,1\n"
        "g1,2026-01-01T00:00:00Z,top,Blue Team,Gnar,0\n"
    )
    teams, players = oe_ingest.parse_oe_csv(path)
    assert str(teams["playoffs"].dtype) == "boolean"
    assert str(players["playoffs"].dtype) == "boolean"
    assert bool(teams.iloc[0]["playoffs"]) is True
    assert bool(players.iloc[0]["playoffs"]) is False
