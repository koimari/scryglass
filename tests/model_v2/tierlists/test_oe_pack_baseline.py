"""Tests for the committed OE baseline restore used by ephemeral workers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lol_kills.etl.restore_oe_pack_baseline import restore_baseline


def _write_fixture_pack(root: Path) -> None:
    pack_root = root / "apps/scryglass/public/packs/v-test"
    for kind in ("player_games", "team_games"):
        for year in ("2025", "2026"):
            path = pack_root / kind / f"year={year}" / "part.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "gameid": f"game-{kind}-{year}",
                "date": f"{year}-01-11T11:11:24Z",
                "league": "LCK",
                "competition_tier": "tier1",
                "event_kind": "",
                "patch": "15.1",
                "side": "Blue",
                "teamname": "Team",
                "result": 1,
            }
            if kind == "player_games":
                row.update({"position": "top", "champion": "Aatrox", "playername": "Player"})
            pd.DataFrame([row]).to_parquet(path, index=False)
    (root / "apps/scryglass/public/packs/latest.json").write_text(
        json.dumps({"pack_id": "v-test", "base_url": "https://example.invalid"}) + "\n",
        encoding="utf-8",
    )
    (pack_root / "manifest.json").write_text(
        json.dumps(
            {
                "pack_id": "v-test",
                "filters": {"years": [2025, 2026]},
                "ingest": {"oe_live_meta": {"source_latest": "2026-01-11T11:11:24Z"}},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_restore_baseline_writes_both_normalized_sources(tmp_path: Path) -> None:
    _write_fixture_pack(tmp_path)

    receipt = restore_baseline(tmp_path)

    assert receipt["pack_id"] == "v-test"
    assert receipt["years"] == ["2025", "2026"]
    assert receipt["outputs"]["player_games"]["rows"] == 2
    assert receipt["outputs"]["team_games"]["rows"] == 2
    assert (tmp_path / "data/lol/warehouse/parquet/oe_player_games.parquet").is_file()
    assert (tmp_path / "data/lol/warehouse/parquet/oe_team_games.parquet").is_file()
    meta = json.loads(
        (tmp_path / "data/lol/warehouse/parquet/oe_baseline_meta.json").read_text(encoding="utf-8")
    )
    assert meta["receipt_canonical_sha256"]
