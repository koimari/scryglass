from __future__ import annotations

import copy
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
    validate_public_query_projection,
    write_public_query_projection,
)


_ARTIFACT_SHA256 = "a" * 64
_RECEIPT_SHA256 = "b" * 64


def _draft_authority(release_id: str = "v2026.08.13.120000") -> dict:
    return {
        "schema_version": "scryglass:draft-authority:v1",
        "status": "descriptive",
        "authority": "descriptive",
        "estimand": "composition_only",
        "release_id": release_id,
        "model_version": "descriptive-draft-score-v1",
        "artifact_sha256": _ARTIFACT_SHA256,
        "receipt_sha256": _RECEIPT_SHA256,
        "probability_authority": False,
        "recommendation_authority": False,
        "betting_authority": False,
    }


def _draft_records(
    games: dict[str, dict],
    release_id: str = "v2026.08.13.120000",
) -> dict:
    return {
        "schema_version": "scryglass:draft-records:v1",
        "authority": "descriptive",
        "estimand": "composition_only",
        "release_id": release_id,
        "model_version": "descriptive-draft-score-v1",
        "artifact_sha256": _ARTIFACT_SHA256,
        "authority_receipt_sha256": _RECEIPT_SHA256,
        "games": {
            game_id: {"draft_pool": copy.deepcopy(game.get("draft_pool"))}
            for game_id, game in games.items()
            if "draft_contribution" in game
        },
    }


def _descriptive_game() -> dict:
    roles = ("top", "jng", "mid", "bot", "sup")
    players = []
    picks = []
    for side in ("Blue", "Red"):
        for role in roles:
            champion = f"{side}-{role}"
            player = "Faker" if side == "Blue" and role == "mid" else f"{side}-{role}-player"
            players.append(
                {
                    "player": player,
                    "side": side,
                    "role": role,
                    "champion": champion,
                }
            )
            picks.append(
                {
                    "side": side,
                    "role": role,
                    "champion": champion,
                    "contribution": 0.06 if side == "Blue" else 0.02,
                    "prior_role_games": 80,
                    "evidence_status": "available",
                }
            )
    blue_components = {
        "base": 0.3,
        "archetype_interactions": 0.1,
        "ally_synergy": 0.2,
        "enemy_counter": 0.1,
        "same_role": 0.05,
    }
    red_components = {
        "base": 0.1,
        "archetype_interactions": 0.05,
        "ally_synergy": 0.1,
        "enemy_counter": -0.05,
        "same_role": 0.0,
    }
    return {
        "game_id": "game-1",
        "date": "2026-08-12T18:00:00Z",
        "league": "LCK",
        "competition_tier": "tier1",
        "blue_team": "T1",
        "red_team": "Gen.G",
        "blue_win": 1,
        "players": players,
        "draft_pool": {
            "status": "complete",
            "evaluated_picks": 10,
            "bans": {
                "Blue": [f"Blue-ban-{index}" for index in range(5)],
                "Red": [f"Red-ban-{index}" for index in range(5)],
            },
            "picked": [
                {
                    "side": pick["side"],
                    "role": pick["role"],
                    "champion": pick["champion"],
                    "best_available": pick["side"] == "Blue",
                    "order": index,
                }
                for index, pick in enumerate(picks, start=1)
            ],
        },
        "draft_contribution": {
            "schema_version": "scryglass:draft-descriptive-signal:v1",
            "status": "available",
            "authority": "descriptive",
            "estimand": "composition_only",
            "model_version": "descriptive-draft-score-v1",
            "artifact_sha256": _ARTIFACT_SHA256,
            "fit_through": None,
            "blue": {
                "signal": 0.75,
                "prior_role_games": 400,
                "components": blue_components,
            },
            "red": {
                "signal": 0.2,
                "prior_role_games": 400,
                "components": red_components,
            },
            "edge_components": {
                "base": 0.2,
                "archetype_interactions": 0.05,
                "ally_synergy": 0.1,
                "enemy_counter": 0.15,
                "same_role": 0.05,
                "total": 0.55,
            },
            "picks": picks,
            "note": "Internal source note.",
        },
    }


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
    if (
        "draft_records" not in overrides
        and isinstance(arguments.get("draft_authority"), dict)
        and arguments["draft_authority"].get("status") == "descriptive"
    ):
        arguments["draft_records"] = _draft_records(arguments["archive_games"])
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
    assert projection["datasets"]["games"][0]["payload"]["players"][0]["champion_image_url"] == "https://cdn.communitydragon.org/galio.png"
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


def test_player_movement_uses_the_current_tier_nested_delta() -> None:
    projection = _projection(
        player_weekly_ranks={
            "by_player": {
                "Faker": {
                    "tier1": {"rank": 2, "delta": -3},
                    "all": {"rank": 5, "delta": 7},
                }
            }
        }
    )

    row = projection["datasets"]["players"][0]
    assert row["movement"] == -3
    assert row["payload"]["weekly"]["tier1"]["delta"] == -3


def test_query_projection_strips_any_nested_draft_field_from_query_rows() -> None:
    game = _game()
    game["draft_pool"] = {"status": "complete"}

    projection = _projection(archive_games={"game-1": game})

    def keys(value: object) -> list[str]:
        if isinstance(value, dict):
            return [str(key) for key in value] + [key for child in value.values() for key in keys(child)]
        if isinstance(value, list):
            return [key for child in value for key in keys(child)]
        return []

    assert "draft_pool" not in {key.casefold() for key in keys(projection)}


def test_query_projection_accepts_only_receipt_bound_descriptive_subset() -> None:
    game = _descriptive_game()
    projection = _projection(
        archive_games={"game-1": game},
        profile_records={
            "schema_version": "scryglass:profile-records:v3",
            "champion_images": {},
            "games": {"game-1": game},
            "players": {"Faker": ["game-1"]},
            "teams": {"T1": ["game-1"], "Gen.G": ["game-1"]},
        },
        draft_authority=_draft_authority(),
    )

    game_payload = projection["datasets"]["games"][0]["payload"]
    signal = game_payload["draft_contribution"]
    assert signal["schema_version"] == "scryglass:draft-descriptive-signal:v1"
    assert signal["authority"] == "descriptive"
    assert signal["estimand"] == "composition_only"
    assert signal["artifact_sha256"] == _ARTIFACT_SHA256
    assert signal["authority_receipt_sha256"] == _RECEIPT_SHA256
    assert signal["edge_components"]["archetype_interactions"] == 0.05
    assert "draft_pool" not in game_payload
    assert "probability" not in json.dumps(signal)

    player = next(row for row in projection["datasets"]["players"] if row["name"] == "Faker")
    assert player["payload"]["draft_metric"]["pick_contribution"] == 0.06
    assert player["payload"]["draft_metric"]["best_available_rate"] == 1.0
    assert player["payload"]["draft_metric"]["ban_coverage"] == 1.0
    assert player["payload"]["draft_metric"]["pool_definition"] == (
        "Best available champion in the published unbanned role pool"
    )
    team = next(row for row in projection["datasets"]["teams"] if row["name"] == "T1")
    assert team["payload"]["draft_metric"]["draft_edge"] == 0.55
    assert team["payload"]["draft_metric"]["positive_edge_rate"] == 1.0
    validate_public_query_projection(
        projection,
        release_id="v2026.08.13.120000",
    )


def test_query_projection_rejects_predictive_and_legacy_atomized_fields() -> None:
    predictive = _descriptive_game()
    predictive["draft_contribution"]["probability"] = 0.75
    with pytest.raises(PublicQueryProjectionError, match="forbidden field"):
        _projection(
            archive_games={"game-1": predictive},
            draft_authority=_draft_authority(),
        )

    legacy = _descriptive_game()
    components = legacy["draft_contribution"]["edge_components"]
    components["atomized_archetypes"] = components.pop("archetype_interactions")
    with pytest.raises(PublicQueryProjectionError, match="component ledger"):
        _projection(
            archive_games={"game-1": legacy},
            draft_authority=_draft_authority(),
        )

    unsupported_atom = _descriptive_game()
    unsupported_atom["draft_contribution"]["picks"][0][
        "evidence_status"
    ] = "atom_estimate"
    with pytest.raises(PublicQueryProjectionError, match="evidence is unavailable"):
        _projection(
            archive_games={"game-1": unsupported_atom},
            draft_authority=_draft_authority(),
        )


def test_query_projection_binds_descriptive_rows_to_the_active_release_authority() -> None:
    with pytest.raises(PublicQueryProjectionError, match="release-bound"):
        _projection(
            archive_games={"game-1": _descriptive_game()},
            draft_authority=_draft_authority("v2026.08.13.999999"),
        )

    projection = _projection(
        archive_games={"game-1": _descriptive_game()},
        draft_authority=_draft_authority(),
    )
    projection["datasets"]["games"][0]["payload"]["draft_contribution"][
        "authority_receipt_sha256"
    ] = "c" * 64
    with pytest.raises(PublicQueryProjectionError, match="not canonical"):
        validate_public_query_projection(
            projection,
            release_id="v2026.08.13.120000",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("release_id", "v2026.08.13.999999"),
        ("model_version", "different-model"),
        ("artifact_sha256", "c" * 64),
        ("authority_receipt_sha256", "d" * 64),
    ],
)
def test_query_projection_rejects_unbound_draft_records(
    field: str,
    value: str,
) -> None:
    game = _descriptive_game()
    records = _draft_records({"game-1": game})
    records[field] = value

    with pytest.raises(PublicQueryProjectionError, match="do not match"):
        _projection(
            archive_games={"game-1": game},
            draft_authority=_draft_authority(),
            draft_records=records,
        )


def test_query_projection_requires_draft_records_for_descriptive_authority() -> None:
    game = _descriptive_game()
    with pytest.raises(PublicQueryProjectionError, match="records are required"):
        _projection(
            archive_games={"game-1": game},
            draft_authority=_draft_authority(),
            draft_records=None,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda metric: metric.pop("best_available_rate"), "keys are not exact"),
        (lambda metric: metric.pop("games"), "not an integer"),
        (lambda metric: metric.pop("pick_contribution"), "keys are not exact"),
        (lambda metric: metric.pop("pool_definition"), "keys are not exact"),
        (lambda metric: metric.pop("ban_coverage"), "keys are not exact"),
        (lambda metric: metric.pop("scope"), "keys are not exact"),
        (lambda metric: metric.__setitem__("best_available_rate", 1.1), "rate is invalid"),
        (lambda metric: metric.__setitem__("ban_coverage", "complete"), "not finite"),
        (lambda metric: metric.__setitem__("pick_contribution", "high"), "not finite"),
        (lambda metric: metric.__setitem__("games", 0), "has no evidence"),
        (lambda metric: metric.__setitem__("probability", 0.7), "forbidden field"),
        (lambda metric: metric.__setitem__("draft_score", 0.2), "keys are not exact"),
    ],
)
def test_query_projection_rejects_incomplete_or_unsafe_player_draft_metric(
    mutation,
    message: str,
) -> None:
    projection = _projection(
        archive_games={"game-1": _descriptive_game()},
        draft_authority=_draft_authority(),
    )
    player = next(
        row for row in projection["datasets"]["players"] if row["name"] == "Faker"
    )
    mutation(player["payload"]["draft_metric"])

    with pytest.raises(PublicQueryProjectionError, match=message):
        validate_public_query_projection(
            projection,
            release_id="v2026.08.13.120000",
        )


def test_query_projection_omits_draft_fields_when_authority_is_unavailable() -> None:
    game = _descriptive_game()
    projection = _projection(archive_games={"game-1": game})

    assert projection["draft_authority"]["status"] == "unavailable"
    assert "draft_contribution" not in projection["datasets"]["games"][0]["payload"]
    assert all(
        "draft_metric" not in row["payload"]
        for dataset in ("players", "teams")
        for row in projection["datasets"][dataset]
    )
    validate_public_query_projection(
        projection,
        release_id="v2026.08.13.120000",
    )


def test_query_game_payload_omits_observed_and_final_gold_fields() -> None:
    game = _descriptive_game()
    game["players"][0]["gold"] = 12345
    game["players"][0]["gold_diff_at_10"] = 321
    game["team_stats"] = {"Blue": {"gold": 1000, "kills": 3}}
    projection = _projection(archive_games={"game-1": game})
    payload = projection["datasets"]["games"][0]["payload"]
    assert "gold" not in payload["players"][0]
    assert "gold_diff_at_10" not in payload["players"][0]
    assert "gold" not in payload["team_stats"]["Blue"]


def test_query_projection_rejects_archive_signal_outside_draft_records() -> None:
    complete = _descriptive_game()
    extra = _descriptive_game()
    extra["game_id"] = "game-2"
    extra["date"] = "2026-08-13T18:00:00Z"

    with pytest.raises(PublicQueryProjectionError, match="outside the validated record set"):
        _projection(
            archive_games={"game-1": complete, "game-2": extra},
            draft_authority=_draft_authority(),
            draft_records=_draft_records({"game-1": complete}),
        )


def test_query_projection_rejects_incomplete_authorized_pool() -> None:
    incomplete = _descriptive_game()
    incomplete["draft_pool"]["evaluated_picks"] = 9

    with pytest.raises(PublicQueryProjectionError, match="matching complete pool"):
        _projection(
            archive_games={"game-1": incomplete},
            draft_authority=_draft_authority(),
            draft_records=_draft_records({"game-1": incomplete}),
        )



def test_query_projection_keeps_composition_score_without_pool_metrics() -> None:
    no_pool = _descriptive_game()
    no_pool.pop("draft_pool")
    projection = _projection(
        archive_games={"game-1": no_pool},
        draft_authority=_draft_authority(),
        draft_records=_draft_records({"game-1": no_pool}),
    )

    game = projection["datasets"]["games"][0]
    assert game["payload"]["draft_contribution"]["status"] == "available"
    player = next(
        row for row in projection["datasets"]["players"] if row["name"] == "Faker"
    )
    assert "draft_metric" not in player["payload"]


def test_query_projection_keeps_shrunk_role_estimate() -> None:
    game = _descriptive_game()
    estimated = game["draft_contribution"]["picks"][0]
    estimated["evidence_status"] = "role_estimate"
    estimated["prior_role_games"] = 14

    projection = _projection(
        archive_games={"game-1": game},
        draft_authority=_draft_authority(),
        draft_records=_draft_records({"game-1": game}),
    )

    picks = projection["datasets"]["games"][0]["payload"]["draft_contribution"]["picks"]
    assert picks[0]["evidence_status"] == "role_estimate"


def test_query_projection_uses_exact_complete_pool_record_set() -> None:
    complete = _descriptive_game()
    unrelated = _game()
    unrelated["game_id"] = "game-2"
    records = _draft_records({"game-1": complete})
    projection = _projection(
        archive_games={"game-1": complete, "game-2": unrelated},
        draft_authority=_draft_authority(),
        draft_records=records,
    )

    player = next(
        row for row in projection["datasets"]["players"] if row["name"] == "Faker"
    )
    metric = player["payload"]["draft_metric"]
    assert metric["games"] == 1
    assert metric["best_available_rate"] == 1.0
    assert metric["pick_contribution"] == 0.06
    assert metric["ban_coverage"] == 1.0
    projected_ids = {
        row["game_id"]
        for row in projection["datasets"]["games"]
        if "draft_contribution" in row["payload"]
    }
    assert projected_ids == set(records["games"])


def test_query_projection_rejects_record_without_archive_signal() -> None:
    complete = _descriptive_game()
    without_signal = copy.deepcopy(complete)
    without_signal.pop("draft_contribution")

    with pytest.raises(PublicQueryProjectionError, match="no archive signal"):
        _projection(
            archive_games={"game-1": without_signal},
            draft_authority=_draft_authority(),
            draft_records=_draft_records({"game-1": complete}),
        )


def test_player_draft_metrics_preserve_case_distinct_source_identities() -> None:
    upper = _descriptive_game()
    lower = _descriptive_game()
    lower["game_id"] = "game-2"
    lower["date"] = "2026-08-13T18:00:00Z"
    next(
        player
        for player in upper["players"]
        if player["side"] == "Blue" and player["role"] == "mid"
    )["player"] = "Random"
    next(
        player
        for player in lower["players"]
        if player["side"] == "Blue" and player["role"] == "mid"
    )["player"] = "random"
    lower_pick = next(
        pick
        for pick in lower["draft_contribution"]["picks"]
        if pick["side"] == "Blue" and pick["role"] == "mid"
    )
    lower_pick["contribution"] = 0.4
    next(
        pick
        for pick in lower["draft_pool"]["picked"]
        if pick["side"] == "Blue" and pick["role"] == "mid"
    )["best_available"] = False

    projection = _projection(
        player_records={
            "Random": {"games": 1, "primary_role": "mid"},
            "random": {"games": 1, "primary_role": "mid"},
        },
        archive_games={"game-1": upper, "game-2": lower},
        draft_authority=_draft_authority(),
    )

    players = {
        row["name"]: row
        for row in projection["datasets"]["players"]
        if row["name"] in {"Random", "random"}
    }
    assert players["Random"]["player_id"] != players["random"]["player_id"]
    assert players["Random"]["payload"]["draft_metric"]["pick_contribution"] == 0.06
    assert players["Random"]["payload"]["draft_metric"]["best_available_rate"] == 1.0
    assert players["random"]["payload"]["draft_metric"]["pick_contribution"] == 0.4
    assert players["random"]["payload"]["draft_metric"]["best_available_rate"] == 0.0
    ambiguous_aliases = [
        row
        for row in projection["datasets"]["aliases"]
        if row["kind"] == "player" and row["alias_key"] == "random"
    ]
    assert ambiguous_aliases == []


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


def test_tier_projection_keeps_same_patch_roles_in_separate_scopes() -> None:
    body = {
        "rows": [
            {
                "scope_id": "patch:26.14",
                "patch": "26.14",
                "role": "mid",
                "champion": "Galio",
                "champion_id": "riot:champion:3",
                "rank": 1,
            },
            {
                "scope_id": "patch:26.14",
                "patch": "26.14",
                "role": "top",
                "champion": "Galio",
                "champion_id": "riot:champion:3",
                "rank": 1,
            },
        ],
        "scopes": [
            {"scope_id": "patch:26.14", "patch": "26.14", "role": "mid", "regional_views": []},
            {"scope_id": "patch:26.14", "patch": "26.14", "role": "top", "regional_views": []},
        ],
    }

    datasets = build_tier_query_datasets(body)

    assert {row["scope_id"] for row in datasets["tier_rows"]} == {"26.14-mid", "26.14-top"}
    assert {row["scope_id"] for row in datasets["tier_scopes"]} == {"26.14-mid", "26.14-top"}
    assert len({row["row_key"] for row in datasets["tier_rows"]}) == len(datasets["tier_rows"])


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
