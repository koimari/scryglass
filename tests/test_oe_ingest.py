from __future__ import annotations

import pandas as pd

import lol_kills.etl.oe_ingest as oe_ingest
from lol_kills.etl.oe_ingest import _normalize_identity


def test_oe_identity_fields_are_trimmed_without_inventing_missing_ids() -> None:
    frame = pd.DataFrame(
        {
            "playername": [" Player One ", "Player Two"],
            "playerid": [" id-1 ", None],
            "teamid": [" team-1 ", pd.NA],
            "teamname": [" T1 ", "Gen.G"],
            "champion": [" Kaisa ", "Azir"],
            "date": ["2026-01-01", "2026-01-02"],
        }
    )

    normalized = _normalize_identity(frame, players=True)

    assert normalized["playername"].tolist() == ["Player One", "Player Two"]
    assert normalized.loc[0, "playerid"] == "id-1"
    assert pd.isna(normalized.loc[1, "playerid"])
    assert normalized.loc[0, "teamid"] == "team-1"
    assert pd.isna(normalized.loc[1, "teamid"])


def test_cached_oe_loader_never_promotes_reconciled_grid_rows(
    tmp_path,
    monkeypatch,
) -> None:
    parquet = tmp_path / "parquet"
    parquet.mkdir()
    pd.DataFrame(
        [
            {"gameid": "oe", "side": "Blue", "source": "oe"},
            {"gameid": "grid", "side": "Blue", "source": "grid"},
        ]
    ).to_parquet(parquet / "oe_team_games.parquet", index=False)
    pd.DataFrame(
        [
            {
                "gameid": "oe",
                "side": "Blue",
                "position": "top",
                "source": "oe",
            },
            {
                "gameid": "grid",
                "side": "Blue",
                "position": "top",
                "source": "grid",
            },
        ]
    ).to_parquet(parquet / "oe_player_games.parquet", index=False)
    monkeypatch.setattr(oe_ingest, "PARQUET_DIR", parquet)

    teams, players = oe_ingest.load_cached_oe()

    assert teams["gameid"].tolist() == ["oe"]
    assert players["gameid"].tolist() == ["oe"]
