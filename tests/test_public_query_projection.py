from __future__ import annotations

import hashlib
import json

import pytest

from lol_kills.export.public_query_projection import (
    QUERY_API_SCHEMA,
    QUERY_DATASETS,
    QUERY_PROJECTION_PATH,
    TIER_QUERY_DATASETS,
    PublicQueryProjectionError,
    build_public_query_projection,
    build_tier_query_datasets,
    canonical_query_bytes,
    normalize_public_key,
    write_public_query_projection,
)


def _game() -> dict:
    return {
        "game_id": "game-1",
        "date": "2026-08-12T18:00:00Z",
        "league": "LCK",
        "competition_tier": "tier1",
        "blue_team": "T1",
        "red_team": "Gen.G",
        "blue_win": 1,
        "players": [
            {
                "player": "Faker",
                "side": "Blue",
                "role": "mid",
                "champion": "Galio",
                "kills": 2,
                "deaths": 1,
                "assists": 8,
                "grade": {"status": "available", "grade": "A"},
            }
        ],
    }


def _projection(**overrides: object) -> dict:
    game = _game()
    arguments = {
        "release_id": "v2026.08.13.120000",
        "player_ratings": [
            {
                "player": "Faker",
                "mu_total": 1713.0,
                "sigma": 28.0,
                "n_maps": 314,
                "last_team": "T1",
                "home_league": "LCK",
                "evidence_active": 1,
            }
        ],
        "team_ratings": [
            {
                "team": "T1",
                "mu_total": 1599.0,
                "sigma": 25.0,
                "rating_p10": 1588.0,
                "n_maps": 314,
                "home_league": "LCK",
                "evidence_active": 1,
            },
            {
                "team": "Gen.G",
                "mu_total": 1604.0,
                "sigma": 25.0,
                "rating_p10": 1590.0,
                "n_maps": 300,
                "home_league": "LCK",
                "evidence_active": 1,
            },
        ],
        "player_records": {
            "Faker": {
                "games": 314,
                "wins": 204,
                "wr": 204 / 314,
                "primary_role": "mid",
                "current_team": "T1",
                "current_league": "LCK",
                "current_tier": "tier1",
            }
        },
        "team_records": {
            "T1": {
                "games": 314,
                "wins": 204,
                "wr": 204 / 314,
                "current_league": "LCK",
                "current_tier": "tier1",
            },
            "Gen.G": {
                "games": 300,
                "wins": 210,
                "wr": 0.7,
                "current_league": "LCK",
                "current_tier": "tier1",
            },
        },
        "player_champion_records": {
            "Faker": [
                {
                    "champion": "Galio",
                    "games": 37,
                    "wins": 31,
                    "losses": 6,
                    "wr": 31 / 37,
                }
            ]
        },
        "profile_records": {
            "schema_version": "scryglass:profile-records:v3",
            "champion_images": {"Galio": "https://cdn.communitydragon.org/galio.png"},
            "games": {"game-1": game},
            "players": {"Faker": ["game-1"]},
            "teams": {"T1": ["game-1"], "Gen.G": ["game-1"]},
        },
        "archive_games": {"game-1": game},
        "player_weekly_ranks": {"by_player": {"Faker": {"tier1": {"rank": 2, "delta": 1}}}},
        "team_weekly_ranks": {"by_team": {"T1": {"rank": 2, "delta": 1}}},
        "player_metadata": {"Faker": {"country": "South Korea", "country_code": "KR"}},
        "leaderboards": {
            "players": {
                "Faker": {
                    "grade_a_games": 1,
                    "grade_games": 1,
                    "recent_form": 1.0,
                }
            },
            "teams": [{"team": "T1", "recent": [{"game_id": "game-1"}]}],
        },
    }
    arguments.update(overrides)
    return build_public_query_projection(**arguments)


def test_query_projection_is_bounded_release_bound_and_draft_free() -> None:
    projection = _projection()

    assert projection["schema_version"] == QUERY_API_SCHEMA
    assert projection["release_id"] == "v2026.08.13.120000"
    assert tuple(projection["datasets"]) == QUERY_DATASETS
    assert projection["datasets"]["players"][0]["name"] == "Faker"
    assert projection["datasets"]["players"][0]["grade_a_games"] == 1
    assert projection["datasets"]["teams"][0]["adjusted_rating"] is not None
    assert projection["datasets"]["games"][0]["game_id"] == "game-1"
    assert len(projection["datasets"]["identity_games"]) == 3
    assert projection["datasets"]["player_champions"][0]["score"] is not None
    assert any(
        row["alias_key"] == "skt" and row["kind"] == "team"
        for row in projection["datasets"]["aliases"]
    )
    assert all(
        row["source_bytes"] <= 64 * 1024
        for rows in projection["datasets"].values()
        for row in rows
    )
    assert all(len(row["row_sha256"]) == 64 for row in projection["datasets"]["players"])
    assert all(len(receipt["row_digest_sha256"]) == 64 for receipt in projection["receipts"].values())
    assert all(
        receipt["rows"] == len(projection["datasets"][dataset])
        and len(receipt["sha256"]) == 64
        for dataset, receipt in projection["receipts"].items()
    )
    for rows in projection["datasets"].values():
        for row in rows:
            source = {key: value for key, value in row.items() if key != "row_sha256"}
            assert row["row_sha256"] == hashlib.sha256(canonical_query_bytes(source)).hexdigest()


def test_query_projection_rejects_any_nested_draft_field() -> None:
    game = _game()
    game["draft_pool"] = {"status": "complete"}

    with pytest.raises(PublicQueryProjectionError, match="Draft fields"):
        _projection(archive_games={"game-1": game})


def test_query_projection_disambiguates_casefolded_identity_for_exact_resolution() -> None:
    projection = _projection(
        player_records={
            "Faker": {"games": 1},
            "FAKER": {"games": 1},
        }
    )

    players = {row["name"]: row for row in projection["datasets"]["players"]}
    assert set(players) == {"Faker", "FAKER"}
    assert players["Faker"]["player_id"] != players["FAKER"]["player_id"]
    for name, row in players.items():
        suffix = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
        search_key = f"faker#{suffix}"
        assert row["search_key"] == search_key
        assert row["player_id"] == hashlib.sha256(
            f"player\0{search_key}".encode("utf-8")
        ).hexdigest()

    aliases = {
        row["alias_key"]: row["identity_id"]
        for row in projection["datasets"]["aliases"]
        if row["kind"] == "player"
    }
    assert "faker" not in aliases
    assert aliases == {
        row["search_key"]: row["player_id"] for row in players.values()
    }


def test_query_projection_writes_internal_receipt(tmp_path) -> None:
    projection = _projection()
    metadata = write_public_query_projection(projection, tmp_path)

    destination = tmp_path / QUERY_PROJECTION_PATH
    assert destination.is_file()
    assert metadata["schema_version"] == QUERY_API_SCHEMA
    assert metadata["projection"]["bytes"] == destination.stat().st_size
    assert metadata["datasets"] == projection["receipts"]
    assert json.loads(destination.read_text(encoding="utf-8"))["release_id"] == projection["release_id"]


def test_normalize_public_key_matches_route_identity_rules() -> None:
    assert normalize_public_key("  Kai’Sa  ") == "kai'sa"


def test_tier_projection_splits_scopes_matrices_and_similarity() -> None:
    tier = {
        "status": "available",
        "rows": [
            {
                "scope_id": "16.15-mid",
                "patch": "16.15",
                "role": "mid",
                "champion": "Galio",
                "champion_id": "riot:champion:3",
                "rank": 1,
                "played_maps": 37,
                "tier_value_pp": 4.1,
            }
        ],
        "scopes": [
            {
                "scope_id": "16.15-mid",
                "patch": "16.15",
                "role": "mid",
                "row_count": 1,
                "regional_views": [
                    {
                        "id": "LEC",
                        "label": "LEC",
                        "maps": 12,
                        "basis": "patch_wide_model_with_regional_appearance_filter",
                        "rows": [
                            {
                                "champion": "Galio",
                                "champion_id": "riot:champion:3",
                                "regional_rank": 1,
                                "global_rank": 1,
                                "strength_score_pp": 4.1,
                                "played_maps": 9,
                                "sample_status": "observed",
                            }
                        ],
                    }
                ],
                "response_matrix": {
                    "champions": [{"champion": "Galio", "champion_id": "riot:champion:3"}],
                    "edge_pp": [[0.0]],
                    "interval_low_pp": [[-1.0]],
                    "interval_high_pp": [[1.0]],
                    "evidence": [["supported"]],
                    "effective_maps": [[37]],
                },
            }
        ],
        "structural_similarity": {
            "schema_version": "scryglass:champion-structural-similarity:v1",
            "source_atom_bridge_sha256": "a" * 64,
            "minimum_similarity": 0.5,
            "weights": {"role": 1.0},
            "champions": [
                {
                    "champion_id": "riot:champion:3",
                    "champion": "Galio",
                    "positions": ["mid"],
                    "roles": ["Mage"],
                    "profile_status": "atom_detail",
                    "traits": [],
                }
            ],
            "similarity": [[1.0]],
        },
    }

    datasets = build_tier_query_datasets(tier)

    assert tuple(datasets) == TIER_QUERY_DATASETS
    base = next(row for row in datasets["tier_rows"] if row.get("tier") is None and row.get("region") is None)
    regional = next(row for row in datasets["tier_rows"] if row.get("region") == "LEC")
    tier_one = next(row for row in datasets["tier_rows"] if row.get("tier") == "tier1" and row.get("region") is None)
    assert base["played_maps"] == 37
    assert regional["played_maps"] == 9
    assert regional["league"] == "LEC"
    assert tier_one["played_maps"] == 9
    assert datasets["tier_scopes"][0]["payload"]["regional_views"][0]["rows"] == []
    assert datasets["tier_scopes"][0]["payload"].get("response_matrix") is None
    assert datasets["tier_matrix_rows"][0]["scope_id"] == "26.15-mid"
    assert datasets["tier_rows"][0]["patch"] == "26.15"
    assert datasets["tier_similarity_edges"][0]["score"] == 1.0


def test_tier_projection_rejects_draft_fields_and_unapproved_images() -> None:
    with pytest.raises(PublicQueryProjectionError, match="Draft"):
        build_tier_query_datasets({"rows": [{"draft_score": 1}], "scopes": []})

    with pytest.raises(PublicQueryProjectionError, match="image URL"):
        _projection(
            profile_records={
                "champion_images": {"Galio": "https://example.com/galio.png"},
                "players": {},
                "teams": {},
            },
            archive_games={},
        )
