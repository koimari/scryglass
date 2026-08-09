import pyarrow as pa
import pandas as pd
import pytest

from lol_kills.export import pack_spec
from lol_kills.export.pack_records import build_player_champion_records, public_team_affiliation
from lol_kills.export.public_pack import (
    _ensure_year_column,
    _filter_years,
    _public_player_rating_rows,
    _validate_public_record_tiers,
    source_identity_sha256,
)


def test_public_pack_contains_only_rating_display_files() -> None:
    assert set(pack_spec.PUBLIC_RATING_REQUIRED_FILES) == {
        "features/ratings_snapshot.json",
        "features/player_ratings_snapshot.json",
        "features/team_records.json",
        "features/team_weekly_ranks.json",
        "features/player_records.json",
        "features/player_champion_records.json",
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
