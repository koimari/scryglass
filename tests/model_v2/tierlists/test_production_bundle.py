"""Tests for the source-bound descriptive production bundle."""

from __future__ import annotations

import json
from pathlib import Path

from lol_kills.v2.tierlists.production_bundle import (
    _commit_sha,
    verify_production_index,
)


ROOT = Path(__file__).resolve().parents[3]


def test_commit_sha_prefers_explicit_full_deploy_sha(monkeypatch) -> None:
    expected = "a" * 40
    monkeypatch.setenv("SCRYGLASS_DEPLOY_COMMIT_SHA", expected)
    monkeypatch.setenv("VERCEL_GIT_COMMIT_SHA", "short-sha")

    assert _commit_sha(Path("/tmp/scryglass-no-git-root")) == expected


def test_production_bundle_has_all_roles_and_public_mirror() -> None:
    report = verify_production_index(ROOT)

    assert report["cell_count"] == 285
    assert report["scope_count"] == 57
    assert report["production_cell_count"] == 285
    assert report["raw_sha256"]


def test_production_index_and_cells_are_public_eligible() -> None:
    index = json.loads(
        (ROOT / "data/lol/v2/tierlists/production/index-v1.json").read_text()
    )

    assert index["development_only"] is False
    assert index["production_eligible"] is True
    assert index["publication_eligible"] is True
    assert set(index["options"]["roles"]) == {"top", "jungle", "mid", "bot", "support"}
    assert all(cell["status"] == "production" for cell in index["cells"])
    allowed = {"Z Blind", "Z Counter", "S Blind", "S Counter", "A", "B", "C", "D"}
    sample = json.loads((ROOT / index["cells"][0]["locator"]).read_text())
    assert all(row["tier_bucket"] in allowed for row in sample["rows"])
