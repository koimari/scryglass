"""Tests for the L9 scope index + filter layer (built on canonical cells)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lol_kills.v2.tierlists.filters import TierListIndex, TierListError

ROOT = Path(__file__).resolve().parents[3]
INDEX = ROOT / "data/lol/v2/tierlists/index-v1.json"


@pytest.fixture(scope="module")
def index() -> TierListIndex:
    return TierListIndex.load(ROOT)


def test_index_is_canonical():
    payload = json.loads(INDEX.read_text())
    submitted = payload["artifact_sha256"]
    unsigned = {k: v for k, v in payload.items() if k != "artifact_sha256"}
    import hashlib
    assert hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == submitted


def test_index_lists_default_cells(index):
    cells = index.cells
    assert len(cells) == 30
    leagues = {c["league"] for c in cells if c["league"]}
    assert leagues == {"LEC", "LCS", "LCK", "LPL"}
    events = {c["event_kind"] for c in cells if c["event_kind"]}
    assert events == {"msi", "ewc"}
    for cell in cells:
        assert cell["row_count"] >= 0
        assert cell["status"] in ("development_only", "unavailable")


def test_every_available_cell_has_rows_and_played_only(index):
    for cell in index.cells:
        if cell["status"] != "development_only":
            continue
        assert cell["row_count"] >= 1


def test_merged_rows_are_played_only_and_ranked(index):
    rows = index.merged_rows()
    assert rows
    for row in rows:
        assert row["played_maps"] >= 1
        assert row["rank"] >= 1
        assert row["tier_bucket"] in ("S", "A", "B", "C", "D")
        assert row["champion_id"].startswith("riot:champion:")


def test_filter_region_europe_is_lec_only(index):
    rows = index.filter_rows(region="europe")
    assert rows
    assert {row["league"] for row in rows} == {"LEC"}


def test_filter_international_msi_and_ewc_are_separate(index):
    msi = index.filter_rows(international="msi")
    ewc = index.filter_rows(international="ewc")
    assert msi and ewc
    assert all(row["event_kind"] == "msi" for row in msi)
    assert all(row["event_kind"] == "ewc" for row in ewc)
    assert {row["champion"] for row in msi} != {row["champion"] for row in ewc}


def test_filter_role_and_patch(index):
    rows = index.filter_rows(role="mid", league="LEC")
    assert rows
    assert all(row["role"] == "mid" and row["league"] == "LEC" for row in rows)
    assert len({row["patch"] for row in rows}) == 1


def test_tampered_cell_is_rejected(tmp_path):
    payload = json.loads(INDEX.read_text())
    cell = next(c for c in payload["cells"] if c["status"] == "development_only")
    locator = ROOT / cell["locator"]
    cell_payload = json.loads(locator.read_text())
    assert cell_payload["rows"]
    cell_payload["rows"][0]["verified_appearance_count"] = 999
    fake = tmp_path / "cells"
    fake.mkdir()
    (fake / locator.name).write_text(json.dumps(cell_payload))
    import shutil
    payload["cells"] = [dict(cell, locator=str(fake / locator.name), raw_sha256="0" * 64)]
    (tmp_path / "index.json").write_text(json.dumps(payload))
    with pytest.raises(TierListError):
        TierListIndex.load(tmp_path, index_path=Path("index.json"))


def test_counterability_never_fabricated(index):
    for row in index.merged_rows():
        assert row["counterability_status"] in ("unavailable", "available")
