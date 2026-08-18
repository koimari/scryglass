"""Staleness detection for the local Oracle's Elixir annual CSVs."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from lol_kills.etl.oe_ingest import (
    OE_SOURCE_MAX_AGE_DAYS,
    OeDownloadError,
    check_oe_source_freshness,
)


def _csv(tmp_path, year: str, *, age_days: float):
    path = tmp_path / f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"
    path.write_text("gameid,date\n", encoding="utf-8")
    stamp = (datetime.now(timezone.utc) - timedelta(days=age_days)).timestamp()
    os.utime(path, (stamp, stamp))
    return path


def test_no_paths_is_quiet(capsys):
    assert check_oe_source_freshness([]) is None
    assert capsys.readouterr().out == ""


def test_fresh_source_is_quiet(tmp_path, capsys):
    paths = [_csv(tmp_path, "2026", age_days=0.5)]
    assert check_oe_source_freshness(paths) is None
    assert capsys.readouterr().out == ""


def test_stale_source_warns_but_does_not_raise(tmp_path, capsys):
    paths = [_csv(tmp_path, "2026", age_days=OE_SOURCE_MAX_AGE_DAYS + 14)]
    assert check_oe_source_freshness(paths) is None
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "stale Oracle's Elixir source" in out
    assert "2026_LoL_esports_match_data_from_OraclesElixir.csv" in out


def test_freshest_file_decides(tmp_path, capsys):
    """An old back-year must not make a current annual file look stale."""

    paths = [
        _csv(tmp_path, "2023", age_days=400),
        _csv(tmp_path, "2026", age_days=0.25),
    ]
    assert check_oe_source_freshness(paths) is None
    assert capsys.readouterr().out == ""


def test_strict_mode_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRYGLASS_OE_STRICT_FRESHNESS", "1")
    paths = [_csv(tmp_path, "2026", age_days=OE_SOURCE_MAX_AGE_DAYS + 1)]
    with pytest.raises(OeDownloadError, match="stale Oracle's Elixir source"):
        check_oe_source_freshness(paths)


def test_strict_mode_stays_quiet_when_fresh(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SCRYGLASS_OE_STRICT_FRESHNESS", "1")
    paths = [_csv(tmp_path, "2026", age_days=0.1)]
    assert check_oe_source_freshness(paths) is None
    assert capsys.readouterr().out == ""


def test_boundary_is_inclusive(tmp_path, capsys):
    """Exactly at the limit is still acceptable; just past it warns."""

    ok = [_csv(tmp_path, "2026", age_days=OE_SOURCE_MAX_AGE_DAYS - 0.01)]
    assert check_oe_source_freshness(ok) is None
    assert capsys.readouterr().out == ""

    late = [_csv(tmp_path, "2025", age_days=OE_SOURCE_MAX_AGE_DAYS + 0.01)]
    assert check_oe_source_freshness(late) is None
    assert "WARNING" in capsys.readouterr().out


def test_custom_max_age(tmp_path, capsys):
    paths = [_csv(tmp_path, "2026", age_days=5)]
    assert check_oe_source_freshness(paths, max_age_days=30) is None
    assert capsys.readouterr().out == ""

    assert check_oe_source_freshness(paths, max_age_days=1) is None
    assert "WARNING" in capsys.readouterr().out


def test_injected_now_controls_the_clock(tmp_path, capsys):
    paths = [_csv(tmp_path, "2026", age_days=0)]
    future = datetime.now(timezone.utc) + timedelta(days=OE_SOURCE_MAX_AGE_DAYS + 5)
    assert check_oe_source_freshness(paths, now=future) is None
    assert "WARNING" in capsys.readouterr().out
