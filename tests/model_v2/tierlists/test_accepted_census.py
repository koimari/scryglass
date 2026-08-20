from __future__ import annotations

import json
from pathlib import Path

import pytest

from lol_kills.export.public_pack import source_identity_sha256
from lol_kills.v2.tierlists.accepted_census import (
    AcceptedCensusError,
    load_census,
    write_census,
)


def test_census_round_trip_uses_the_public_release_identity(tmp_path: Path) -> None:
    path = tmp_path / "accepted.json"

    written = write_census(path, ["game-b", "game-a", "game-a"])
    loaded = load_census(path)

    assert loaded == written
    assert loaded["game_ids"] == ["game-a", "game-b"]
    assert loaded["source_identity_sha256"] == source_identity_sha256(
        ["game-a", "game-b"]
    )


def test_census_rejects_a_tampered_identity(tmp_path: Path) -> None:
    path = tmp_path / "accepted.json"
    payload = write_census(path, ["game-a"])
    payload["source_identity_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AcceptedCensusError, match="binding"):
        load_census(path)
