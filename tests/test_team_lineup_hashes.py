from __future__ import annotations

import pandas as pd

from lol_kills.ratings.dual_elo import lineup_hashes_from_players


def test_lineup_hashes_use_sorted_unique_names_and_champion_fallback() -> None:
    players = pd.DataFrame(
        [
            {"game_uid": "g1", "teamname": "KC", "playername": "z", "champion": "Zed"},
            {"game_uid": "g1", "teamname": "KC", "playername": "a", "champion": "Ahri"},
            {"game_uid": "g1", "teamname": "KC", "playername": "a", "champion": "Ahri"},
            {"game_uid": "g1", "teamname": "Other", "playername": None, "champion": "Zed"},
            {"game_uid": "g1", "teamname": "Other", "playername": None, "champion": "Ahri"},
        ]
    )

    assert lineup_hashes_from_players(players) == {
        "g1|Karmine Corp": "a|z",
        "g1|Other": "Ahri|Zed",
    }


def test_lineup_hashes_fall_back_when_player_names_are_absent() -> None:
    players = pd.DataFrame(
        [
            {"gameid": "g1", "team": "A", "champion": "Zed"},
            {"gameid": "g1", "team": "A", "champion": "Ahri"},
        ]
    )

    assert lineup_hashes_from_players(players) == {"g1|A": "Ahri|Zed"}
