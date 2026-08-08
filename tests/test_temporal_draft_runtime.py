from __future__ import annotations

from lol_kills.research.temporal_draft_runtime import (
    _lineup_status,
    _online_feature_rows,
)
from lol_kills.research.player_champion_features import player_champion_feature_rows


def _game(game_id: str, when: str, y: float, blue: str = "Blue", red: str = "Red") -> dict:
    players = {
        (side, role): {"player": f"{side}-{role}"}
        for side in ("blue", "red")
        for role in ("top", "jng", "mid", "bot", "sup")
    }
    return {
        "game_uid": game_id,
        "date": __import__("pandas").Timestamp(when),
        "league": "TEST",
        "blue_team": blue,
        "red_team": red,
        "blue": {role: players["blue", role] for role in ("top", "jng", "mid", "bot", "sup")},
        "red": {role: players["red", role] for role in ("top", "jng", "mid", "bot", "sup")},
        "y": y,
    }


def _draft_game(game_id: str, when: str, y: float) -> dict:
    game = _game(game_id, when, y)
    for side in ("blue", "red"):
        for role in game[side]:
            game[side][role]["champion"] = f"champion-{role}"
    return game


def test_online_features_do_not_share_same_timestamp_outcomes() -> None:
    first = _game("series_1_1", "2026-07-01 10:00:00", 1.0)
    second = _game("series_1_2", "2026-07-01 10:00:00", 0.0)
    features = _online_feature_rows([first, second])
    assert features[first["game_uid"]] == features[second["game_uid"]]


def test_player_champion_features_do_not_share_same_timestamp_outcomes() -> None:
    first = _draft_game("series_1_1", "2026-07-01 10:00:00", 1.0)
    second = _draft_game("series_1_2", "2026-07-01 10:00:00", 0.0)
    features = player_champion_feature_rows([first, second])
    assert features[first["game_uid"]] == features[second["game_uid"]]


def test_player_champion_features_use_only_prior_maps() -> None:
    first_win = _draft_game("series_1_1", "2026-07-01 10:00:00", 1.0)
    first_loss = _draft_game("series_1_1", "2026-07-01 10:00:00", 0.0)
    later = _draft_game("series_1_2", "2026-07-01 10:01:00", 1.0)
    win_features = player_champion_feature_rows([first_win, later])[later["game_uid"]]
    loss_features = player_champion_feature_rows([first_loss, later])[later["game_uid"]]
    assert win_features["player_champion_edge"] != loss_features["player_champion_edge"]


def test_strict_roster_without_packet_is_unavailable() -> None:
    run = {
        "pregame": {
            "as_of": "2026-07-01T09:00:00Z",
            "event_start": "2026-07-01T10:00:00Z",
            "blue": {"team": "Blue", "players": []},
            "red": {"team": "Red", "players": []},
        }
    }
    status, evidence = _lineup_status(run, [], strict_roster=True)
    assert status == "unavailable"
    assert evidence == {}


def test_non_strict_mode_marks_observed_lineup_as_retrospective() -> None:
    run = {
        "pregame": {
            "as_of": "2026-07-01T09:00:00Z",
            "event_start": "2026-07-01T10:00:00Z",
            "blue": {"team": "Blue", "players": []},
            "red": {"team": "Red", "players": []},
        }
    }
    status, evidence = _lineup_status(run, [], strict_roster=False)
    assert status == "retrospective_lineup_only"
    assert evidence == {}


def test_strict_mode_accepts_only_exact_hash_bound_fixture_receipt() -> None:
    fixture_id = "fixture-1"
    players = [
        {"role": role, "player": f"Blue-{role}"}
        for role in ("top", "jng", "mid", "bot", "sup")
    ]
    red_players = [
        {"role": role, "player": f"Red-{role}"}
        for role in ("top", "jng", "mid", "bot", "sup")
    ]
    run = {
        "pregame": {
            "fixture_id": fixture_id,
            "as_of": "2026-07-01T09:59:59Z",
            "event_start": "2026-07-01T10:00:00Z",
            "blue": {"team": "Blue", "players": players},
            "red": {"team": "Red", "players": red_players},
        }
    }
    receipt = {
        "fixture_id": fixture_id,
        "as_of": "2026-07-01T09:59:59Z",
        "event_start": "2026-07-01T10:00:00Z",
        "authority_status": "confirmed",
        "blockers": [],
        "evidence_hash": "a" * 64,
        "teams": {
            "blue": {
                "team": "Blue",
                "players": [
                    {"role": "jungle" if row["role"] == "jng" else "support" if row["role"] == "sup" else row["role"], "player": row["player"]}
                    for row in players
                ],
                "evidence_hash": "b" * 64,
            },
            "red": {
                "team": "Red",
                "players": [
                    {"role": "jungle" if row["role"] == "jng" else "support" if row["role"] == "sup" else row["role"], "player": row["player"]}
                    for row in red_players
                ],
                "evidence_hash": "c" * 64,
            },
        },
    }

    status, evidence = _lineup_status(
        run,
        [],
        strict_roster=True,
        receipts={fixture_id: receipt},
        receipt_manifest_sha256="d" * 64,
    )

    assert status == "verified_preevent"
    assert evidence["manifest_sha256"] == "d" * 64
    assert evidence["blue"]["lineup_matches"] is True
    assert evidence["red"]["lineup_matches"] is True


def test_strict_receipt_identity_mismatch_stays_unavailable_for_context() -> None:
    run = {
        "pregame": {
            "fixture_id": "fixture-1",
            "as_of": "2026-07-01T09:59:59Z",
            "event_start": "2026-07-01T10:00:00Z",
            "blue": {"team": "Blue", "players": [{"role": "top", "player": "Expected"}]},
            "red": {"team": "Red", "players": [{"role": "top", "player": "Expected Red"}]},
        }
    }
    receipt = {
        "fixture_id": "fixture-1",
        "as_of": "2026-07-01T09:59:59Z",
        "event_start": "2026-07-01T10:00:00Z",
        "authority_status": "confirmed",
        "blockers": [],
        "evidence_hash": "a" * 64,
        "teams": {
            "blue": {"team": "Blue", "players": [{"role": "top", "player": "Different"}], "evidence_hash": "b" * 64},
            "red": {"team": "Red", "players": [{"role": "top", "player": "Expected Red"}], "evidence_hash": "c" * 64},
        },
    }

    status, evidence = _lineup_status(
        run,
        [],
        strict_roster=True,
        receipts={"fixture-1": receipt},
    )

    assert status == "mismatch"
    assert evidence["reason"] == "fixture_receipt_identity_mismatch"
