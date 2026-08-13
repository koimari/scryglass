import pyarrow as pa
import pandas as pd
import pytest
from copy import deepcopy

from lol_kills.export import pack_spec
from lol_kills.export.pack_records import (
    _first_pick_value,
    build_player_champion_records,
    build_maps_frame_from_team_games,
    build_profile_records,
    merge_accepted_profile_games,
    public_team_affiliation,
)
from lol_kills.export.public_pack import (
    _attach_public_team_evidence,
    _attach_published_draft_pools,
    _complete_player_game_ids,
    _ensure_year_column,
    _filter_years,
    _public_player_rating_rows,
    _validate_public_composition_records,
    _validate_public_record_tiers,
    build_draft_records_payload,
    source_identity_sha256,
)
from lol_kills.research.composition_signal import CompositionSignalError


def test_build_draft_records_payload() -> None:
    games = [
        {
            "game_uid": "g1", "date": "2026-08-01", "league": "LEC",
            "blue_team": "Team A", "red_team": "Team B",
        }
    ]
    signals = {
        "g1": {
            "status": "available",
            "blue": {"signal": 0.25, "prior_role_games": 90},
            "red": {"signal": -0.05, "prior_role_games": 80},
            "picks": [],
        }
    }
    payload = build_draft_records_payload(
        signals, games, {"model_version": "composition-signal-v3", "fit_through": "2026-08-01"}
    )
    assert payload["games"]["g1"]["draft_edge"] == 0.3
    assert payload["games"]["g1"]["blue_team"] == "Team A"
    signals["g2"] = {"status": "unavailable", "blue": {"signal": None}, "red": {"signal": None}}
    payload2 = build_draft_records_payload(
        signals, games + [{"game_uid": "g2", "date": "2026-08-02"}], None
    )
    assert "g2" not in payload2["games"]


def test_public_team_ratings_receive_evidence_fields() -> None:
    ratings = pd.DataFrame(
        [
            {
                "team": "Gen.G",
                "team_key": "gen-g",
                "sigma": 20.0,
                "n_series": 18,
                "last_game_date": "2026-08-08T12:00:00Z",
                "home_league": "LCK",
                "international_series": 4,
            }
        ]
    )

    output = _attach_public_team_evidence(
        ratings,
        source_as_of=pd.Timestamp("2026-08-09T12:00:00Z"),
        weekly_ranks={"by_team": {"Gen.G": {"mu_delta": -2.5}}},
    )

    row = output.iloc[0]
    assert row["evidence_stability"] == 2.5
    assert row["evidence_active"] == 1
    assert row["evidence_state"] != "unsupported"


def test_incomplete_or_ambiguous_player_maps_wait_for_later_refresh() -> None:
    rows = []
    for game_id, blue_bot, blue_support in (
        ("complete", "Blue Bot", "Blue Support"),
        ("duplicate", "unknown player", "unknown player"),
    ):
        for side, prefix in (("Blue", "Blue"), ("Red", "Red")):
            for role in ("top", "jng", "mid", "bot", "sup"):
                player = f"{prefix} {role}"
                if side == "Blue" and role == "bot":
                    player = blue_bot
                if side == "Blue" and role == "sup":
                    player = blue_support
                rows.append(
                    {
                        "game_uid": game_id,
                        "side": side,
                        "position": role,
                        "playername": player,
                    }
                )

    assert _complete_player_game_ids(pd.DataFrame(rows)) == {"complete"}


def test_public_pack_contains_only_rating_display_files() -> None:
    assert set(pack_spec.PUBLIC_RATING_REQUIRED_FILES) == {
        "features/ratings_snapshot.json",
        "features/player_ratings_snapshot.json",
        "features/team_records.json",
        "features/team_weekly_ranks.json",
        "features/player_records.json",
        "features/player_champion_records.json",
        "features/profile_records.json",
        "features/match_index.json",
        "features/match_records_2025.json",
        "features/match_records_2026.json",
        "features/player_weekly_ranks.json",
        "features/player_metadata.json",
    }


def test_player_champion_records_are_compact_and_sorted() -> None:
    records = build_player_champion_records(
        pd.DataFrame(
            [
                {"playername": "Inspired", "position": "jng", "champion": "Ivern", "result": 1, "kills": 2, "deaths": 1, "assists": 12},
                {"playername": "Inspired", "position": "jng", "champion": "Ivern", "result": 0, "kills": 1, "deaths": 3, "assists": 8},
                {"playername": "Inspired", "position": "jng", "champion": "Xin Zhao", "result": 1, "kills": 5, "deaths": 2, "assists": 7},
            ]
        )
    )

    assert records["Inspired"] == [
        {
            "champion": "Ivern",
            "games": 2,
            "wins": 1,
            "losses": 1,
            "wr": 0.5,
            "kills": 1.5,
            "deaths": 2.0,
            "assists": 10.0,
        },
        {
            "champion": "Xin Zhao",
            "games": 1,
            "wins": 1,
            "losses": 0,
            "wr": 1.0,
            "kills": 5.0,
            "deaths": 2.0,
            "assists": 7.0,
        },
    ]


def test_profile_records_normalize_recent_games_without_raw_tables() -> None:
    rows = []
    for side, team, result in (("Blue", "LYON", 1), ("Red", "Other", 0)):
        for role, player, champion in (
            ("top", f"{team} Top", "Gnar"),
            ("jng", "Inspired" if team == "LYON" else f"{team} Jungle", "Ivern"),
            ("mid", f"{team} Mid", "Ahri"),
            ("bot", f"{team} Bot", "Ezreal"),
            ("sup", f"{team} Support", "Nautilus"),
        ):
            rows.append(
                {
                    "game_uid": "oe-api:game-1",
                    "date": "2026-08-09T12:00:00Z",
                    "league": "LCS",
                    "side": side,
                    "teamname": team,
                    "playername": player,
                    "position": role,
                    "champion": champion,
                    "result": result,
                    "kills": 2,
                    "deaths": 1,
                    "assists": 8,
                    "teamkills": 10,
                    "gamelength": 1800,
                    "dpm": 512.5,
                    "damageshare": 0.22,
                    "totalgold": 11000,
                    "minionkills": 180,
                    "monsterkills": 20,
                    "cspm": 6.67,
                    "visionscore": 31,
                    "wardsplaced": 14,
                    "golddiffat10": 125,
                    "dragons": 3 if side == "Blue" else 1,
                    "heralds": 1,
                    "void_grubs": 4 if side == "Blue" else 2,
                    "barons": 1 if side == "Blue" else 0,
                    "atakhans": 0,
                    "towers": 8 if side == "Blue" else 3,
                    "inhibitors": 2 if side == "Blue" else 0,
                }
            )

    payload = build_profile_records(
        pd.DataFrame(rows),
        champion_image_urls={"Ivern": "https://example.test/ivern.png"},
    )

    assert payload["schema_version"] == "scryglass:profile-records:v3"
    assert payload["grade_contract"] == "scryglass:player-map-grade:v2"
    assert payload["players"]["Inspired"] == ["game-1"]
    assert payload["teams"]["LYON"] == ["game-1"]
    game = payload["games"]["game-1"]
    assert game["blue_team"] == "LYON"
    assert game["red_team"] == "Other"
    assert game["competition_tier"] == "tier1"
    assert game["duration_seconds"] == 1800
    assert game["team_stats"]["Blue"]["dragons"] == 3
    assert game["team_stats"]["Blue"]["gold"] == 55000
    inspired = next(row for row in game["players"] if row["player"] == "Inspired")
    assert inspired["role"] == "jungle"
    assert inspired["cs"] == 200
    assert inspired["damage_per_minute"] == 512.5
    assert inspired["vision_score"] == 31
    assert inspired["grade"]["status"] == "unavailable"
    assert payload["champion_images"]["Ivern"] == "https://example.test/ivern.png"


def test_profile_records_keep_public_composition_evidence_optional() -> None:
    rows = []
    for side, team, result in (("Blue", "A", 1), ("Red", "B", 0)):
        for role in ("top", "jng", "mid", "bot", "sup"):
            rows.append(
                {
                    "game_uid": "game-1",
                    "date": "2026-08-09T12:00:00Z",
                    "league": "LCS",
                    "side": side,
                    "teamname": team,
                    "playername": f"{team}-{role}",
                    "position": role,
                    "champion": f"{side}-{role}",
                    "result": result,
                    "kills": 1,
                    "deaths": 1,
                    "assists": 5,
                }
            )
    evidence = {
        "schema_version": "scryglass:composition-signal:v1",
        "status": "available",
        "model_version": "composition-signal-v1",
        "fit_through": "2026-08-08T00:00:00Z",
        "blue": {"signal": 0.5, "prior_role_games": 200},
        "red": {"signal": -0.1, "prior_role_games": 200},
        "picks": [],
        "note": "Descriptive composition signal.",
    }

    payload = build_profile_records(
        pd.DataFrame(rows),
        composition_signals={"game-1": evidence},
    )

    assert payload["games"]["game-1"]["draft_contribution"] == evidence
    assert "coefficients" not in payload["games"]["game-1"]["draft_contribution"]


def test_public_composition_evidence_matches_the_published_picks() -> None:
    roles = ("top", "jungle", "mid", "bot", "support")
    players = [
        {
            "player": f"{side}-{role}",
            "side": side,
            "role": role,
            "champion": f"{side}-{role}",
        }
        for side in ("Blue", "Red")
        for role in roles
    ]
    picks = [
        {
            "side": side,
            "role": {"jungle": "jng", "support": "sup"}.get(role, role),
            "champion": f"{side}-{role}",
            "contribution": 0.1,
            "prior_role_games": 40,
            "evidence_status": "available",
        }
        for side in ("Blue", "Red")
        for role in roles
    ]
    game = {
        "game_id": "game-1",
        "date": "2026-08-09T12:00:00Z",
        "players": players,
    }
    signal = {
        "schema_version": "scryglass:composition-signal:v1",
        "status": "available",
        "model_version": "composition-signal-v1",
        "fit_through": "2026-08-08T00:00:00Z",
        "blue": {"signal": 0.5, "prior_role_games": 200},
        "red": {"signal": 0.5, "prior_role_games": 200},
        "picks": picks,
        "note": "Values are model contribution units.",
    }

    assert _validate_public_composition_records({"games": {"game-1": {**game, "draft_contribution": signal}}}) == {
        "games": 1,
        "available": 1,
        "limited": 0,
        "unavailable": 0,
    }

    leaked = deepcopy(signal)
    leaked["coefficients"] = [0.4]
    with pytest.raises(CompositionSignalError, match="private composition fields"):
        _validate_public_composition_records({"games": {"game-1": {**game, "draft_contribution": leaked}}})

    same_fit = deepcopy(signal)
    same_fit["fit_through"] = game["date"]
    with pytest.raises(CompositionSignalError, match="fit watermark"):
        _validate_public_composition_records({"games": {"game-1": {**game, "draft_contribution": same_fit}}})


def test_published_draft_pool_uses_bans_and_fails_closed_without_them() -> None:
    roles = ("top", "jungle", "mid", "bot", "support")
    picked = []
    players = []
    picks = []
    order = 1
    for side, team in (("Blue", "A"), ("Red", "B")):
        for role in roles:
            champion = f"{side}-{role}"
            players.append({"player": f"{team}-{role}", "side": side, "role": role, "champion": champion})
            picked.append({"side": side, "role": role, "champion": champion, "order": order})
            picks.append({"side": side, "role": {"jungle": "jng", "support": "sup"}.get(role, role), "champion": champion, "contribution": 0.1, "prior_role_games": 40, "evidence_status": "available"})
            order += 1
    game = {
        "game_id": "game-1",
        "patch": "16.1",
        "players": players,
        "draft_pool": {
            "schema_version": "scryglass:draft-pool:v1",
            "status": "limited",
            "source": "oracle-elixir",
            "patch": "16.1",
            "bans": {"Blue": ["Ban Blue 1", "Ban Blue 2", "Ban Blue 3", "Ban Blue 4", "Ban Blue 5"], "Red": ["Ban Red 1", "Ban Red 2", "Ban Red 3", "Ban Red 4", "Ban Red 5"]},
            "picked": picked,
            "unpicked": [],
        },
        "draft_contribution": {
            "schema_version": "scryglass:composition-signal:v1",
            "status": "available",
            "model_version": "test",
            "fit_through": "2026-08-01T00:00:00Z",
            "blue": {"signal": 0.5, "prior_role_games": 200},
            "red": {"signal": 0.5, "prior_role_games": 200},
            "picks": picks,
            "note": "test",
        },
    }
    tier_rows = []
    for role in roles:
        selected = f"Blue-{role}"
        tier_rows.extend([
            {"patch": "16.10", "role": role, "champion": selected, "rank": 2},
            {"patch": "16.10", "role": role, "champion": "Red-" + role, "rank": 1},
            {"patch": "16.10", "role": role, "champion": "Best-" + role, "rank": 3},
        ])
    payload = {"games": {"game-1": game}}
    audit = _attach_published_draft_pools(payload, {"rows": tier_rows})
    assert audit["quality_games"] == 1
    assert game["draft_pool"]["patch"] == "16.10"
    assert game["draft_pool"]["evaluated_picks"] == 10
    assert game["draft_contribution"]["picks"][0]["best_available"] is False
    assert game["draft_contribution"]["picks"][5]["best_available"] is True
    game["draft_pool"]["bans"]["Red"] = []
    _attach_published_draft_pools(payload, {"rows": tier_rows})
    assert game["draft_contribution"]["picks"][0]["best_available"] is None


def test_first_pick_falls_back_from_null_map_metadata() -> None:
    group = pd.DataFrame([{"blue_firstPick": pd.NA, "firstPick": "Blue"}])
    assert _first_pick_value(group, {"blue_first_pick": float("nan")}) is True


def test_team_map_adapter_preserves_public_draft_metadata() -> None:
    rows = []
    for side, team, first_pick in (("Blue", "A", 1), ("Red", "B", 0)):
        rows.append(
            {
                "game_uid": "draft-map",
                "date": "2026-08-01",
                "league": "LCK",
                "side": side,
                "position": "team",
                "teamname": team,
                "result": 1 if side == "Blue" else 0,
                "patch": "16.1",
                "firstPick": first_pick,
                **{f"ban{slot}": f"{side}-ban-{slot}" for slot in range(1, 6)},
                **{f"pick{slot}": f"{side}-pick-{slot}" for slot in range(1, 6)},
            }
        )

    maps = build_maps_frame_from_team_games(pd.DataFrame(rows))

    assert maps.loc[0, "patch"] == "16.1"
    assert maps.loc[0, "blue_firstPick"] == 1
    assert maps.loc[0, "red_firstPick"] == 0
    assert maps.loc[0, "blue_ban4"] == "Blue-ban-4"
    assert maps.loc[0, "red_pick5"] == "Red-pick-5"


def test_published_draft_pool_applies_second_phase_bans_after_sixth_pick() -> None:
    picked = [
        {"side": "Blue" if index % 2 else "Red", "role": "mid", "champion": f"Pick-{index}", "order": index}
        for index in range(1, 11)
    ]
    game = {
        "game_id": "game-2",
        "patch": "16.1",
        "draft_pool": {
            "patch": "16.1",
            "bans": {
                "Blue": ["Blue ban 1", "Blue ban 2", "Blue ban 3", "Second phase best", "Blue ban 5"],
                "Red": ["Red ban 1", "Red ban 2", "Red ban 3", "Red ban 4", "Red ban 5"],
            },
            "picked": picked,
        },
        "draft_contribution": {"picks": [dict(item, contribution=0.1) for item in picked]},
    }
    tier_rows = [
        {"patch": "16.10", "role": "mid", "champion": "Second phase best", "rank": 1},
        *[
            {"patch": "16.10", "role": "mid", "champion": f"Pick-{index}", "rank": index + 1}
            for index in range(1, 11)
        ],
    ]

    _attach_published_draft_pools({"games": {"game-2": game}}, {"rows": tier_rows})

    assert game["draft_pool"]["evaluated_picks"] == 10
    assert game["draft_pool"]["picked"][0]["best_available"] is False
    assert game["draft_pool"]["picked"][6]["best_available"] is True


def test_profile_records_keep_old_maps_in_the_archive_only() -> None:
    rows = []
    roles = ("top", "jng", "mid", "bot", "sup")
    for game_id, date in (("old-map", "2025-01-10"), ("new-map", "2026-08-09")):
        for side, team, result in (("Blue", "A", 1), ("Red", "B", 0)):
            for role in roles:
                rows.append(
                    {
                        "game_uid": game_id,
                        "date": date,
                        "league": "LCS",
                        "side": side,
                        "teamname": team,
                        "playername": f"{team}-{role}",
                        "position": role,
                        "champion": f"Champion-{role}",
                        "result": result,
                        "kills": 2,
                        "deaths": 1,
                        "assists": 8,
                    }
                )

    payload = build_profile_records(pd.DataFrame(rows), include_archive=True)

    assert set(payload["games"]) == {"new-map"}
    assert set(payload["_archive_games"]) == {"old-map", "new-map"}


def test_match_refresh_preserves_accepted_kda_and_grade() -> None:
    candidate = {
        "game_id": "game-1",
        "date": "2026-08-09T12:00:00Z",
        "blue_team": "A",
        "red_team": "B",
        "blue_win": 1,
        "players": [
            {
                "player": f"player-{index}",
                "side": "Blue" if index < 5 else "Red",
                "role": ("top", "jungle", "mid", "bot", "support")[index % 5],
                "champion": f"champion-{index}",
                "kills": None,
                "deaths": None,
                "assists": None,
                "grade": {"status": "unavailable", "reason": "stats pending"},
            }
            for index in range(10)
        ],
    }
    accepted = deepcopy(candidate)
    for player in accepted["players"]:
        player["kills"] = 2
        player["deaths"] = 1
        player["assists"] = 8
        player["grade"] = {"status": "available", "grade": "A"}

    merged = merge_accepted_profile_games({"game-1": candidate}, {"game-1": accepted})

    assert merged["game-1"]["players"][0]["kills"] == 2
    assert merged["game-1"]["players"][0]["grade"]["grade"] == "A"


def test_public_player_ratings_exclude_disconnected_rows() -> None:
    rows = [
        {"player": "Inspired", "evidence_disconnected": 0, "evidence_state": "observed"},
        {"player": "Baus", "evidence_disconnected": 1, "evidence_state": "disconnected"},
        {"player": "Caedrel", "evidence_disconnected": 0, "evidence_state": "disconnected"},
    ]

    assert _public_player_rating_rows(rows) == [rows[0]]


def test_public_pack_withholds_raw_rows_models_and_studies() -> None:
    forbidden = set(pack_spec.FORBIDDEN_PUBLIC_MODEL_FILES)
    assert "models/" in forbidden
    assert "studies/" in forbidden
    assert "team_games/" in forbidden
    assert "player_games/" in forbidden
    assert "maps/" in forbidden


def test_live_map_overlay_gets_partition_year_from_date() -> None:
    table = pa.table(
        {
            "game_uid": ["g-2025", "g-2026"],
            "date": ["2025-12-31T23:00:00Z", "2026-01-01T01:00:00Z"],
        }
    )

    enriched = _ensure_year_column(table)

    assert enriched["year"].to_pylist() == [2025, 2026]


def test_live_overlay_prefers_normalized_oe_year_when_columns_disagree() -> None:
    table = pa.table(
        {
            "year": [2025, 2025, 2026],
            "oe_year": [2025, 2026, 2027],
        }
    )

    filtered = _filter_years(table, (2025, 2026), ("year", "oe_year"))

    assert filtered.to_pydict() == {
        "year": [2025, 2025],
        "oe_year": [2025, 2026],
    }


def test_source_identity_digest_is_canonical_order_independent() -> None:
    left = source_identity_sha256(["oe-api:game-2", "game-1", "oe-api:game-1"])
    right = source_identity_sha256(["game-1", "game-2"])
    assert left == right


def test_excluded_team_has_no_public_affiliation() -> None:
    assert public_team_affiliation("Los Ratones") is None
    assert public_team_affiliation("Gen.G") == "Gen.G"
    assert public_team_affiliation("LYON (2024 American Team)") == "LYON"


def test_public_record_tier_must_match_the_canonical_league() -> None:
    _validate_public_record_tiers(
        {"Gen.G": {"leagues": ["LCK"], "current_league": "LCK", "current_tier": "tier1"}},
        label="team",
    )
    with pytest.raises(RuntimeError, match="inconsistent league tier"):
        _validate_public_record_tiers(
            {"Gen.G": {"leagues": ["LCK"], "current_league": "LCK", "current_tier": "tier3"}},
            label="team",
        )


def test_public_record_rejects_transport_label_as_a_league() -> None:
    with pytest.raises(RuntimeError, match="transport label"):
        _validate_public_record_tiers(
            {
                "Gen.G": {
                    "leagues": ["ORACLE_ELIXIR_API"],
                    "current_league": "ORACLE_ELIXIR_API",
                    "current_tier": "tier3",
                }
            },
            label="team",
        )
