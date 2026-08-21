from __future__ import annotations

import json

import pandas as pd
import pyarrow as pa

from lol_kills.research.future_value_snapshots import (
    FutureValueSnapshotResult,
    _rank_diffs,
    write_snapshot_bundle,
)

from lol_kills.export.pack_spec import (
    PLAYER_RATINGS_SNAPSHOT_COLS,
    RATINGS_SNAPSHOT_COLS,
)
from lol_kills.export.public_pack import serialize_rating_snapshot_rows
from lol_kills.ratings.source_identity import (
    attach_player_ids,
    attach_team_ids,
    attach_weekly_ids,
    build_rating_identity_maps,
)


def test_ambiguous_display_alias_is_unmapped() -> None:
    players = pd.DataFrame(
        {
            "playername": ["Ace", "Ace", "Support"],
            "playerid": [
                "oe:player:one",
                "oe:player:two",
                "oe:player:support",
            ],
            "teamname": ["Alpha", "Beta", "Alpha"],
            "teamid": ["oe:team:alpha", "oe:team:beta", "oe:team:alpha"],
        }
    )
    teams = pd.DataFrame(
        {
            "teamname": ["Zen", "Zen"],
            "teamid": ["oe:team:zen-one", "oe:team:zen-two"],
        }
    )
    identities = build_rating_identity_maps(
        players, teams, source_identity_sha256="a" * 64
    )

    assert identities.player_id_for("Ace") is None
    assert identities.player_id_for("Support") == "oe:player:support"
    assert identities.coverage["player"]["ambiguous_display_keys"]["ace"] == [
        "oe:player:one",
        "oe:player:two",
    ]
    assert identities.team_id_for("Zen") is None
    assert "zen" in identities.coverage["team"]["ambiguous_display_keys"]
    assert identities.coverage["status"] == "partial"


def test_missing_source_ids_remain_null() -> None:
    players = pd.DataFrame(
        {
            "playername": ["Missing"],
            "playerid": [None],
            "teamname": ["Alpha"],
            "teamid": [None],
        }
    )
    identities = build_rating_identity_maps(players)
    player_snapshot = attach_player_ids(
        pd.DataFrame({"player": ["Missing"], "last_team": ["Alpha"]}),
        identities,
    )
    team_snapshot = attach_team_ids(pd.DataFrame({"team": ["Alpha"]}), identities)

    assert pd.isna(player_snapshot.loc[0, "player_id"])
    assert pd.isna(player_snapshot.loc[0, "team_id"])
    assert pd.isna(team_snapshot.loc[0, "team_id"])
    assert identities.coverage["player"]["missing_id_rows"] == 1
    assert identities.coverage["team"]["missing_id_rows"] == 1


def test_verified_one_to_one_join_reaches_snapshots_and_weekly_rows() -> None:
    players = pd.DataFrame(
        {
            "playername": ["Player One", "Player One", "Player Two"],
            "playerid": [
                "oe:player:one",
                "oe:player:one",
                "oe:player:two",
            ],
            "teamname": ["Alpha", "Alpha", "Beta"],
            "teamid": ["oe:team:alpha", "oe:team:alpha", "oe:team:beta"],
        }
    )
    identities = build_rating_identity_maps(
        players,
        source_identity_sha256="b" * 64,
        source_game_count=2,
    )

    player_snapshot = attach_player_ids(
        pd.DataFrame(
            {
                "player": ["Player One", "Player Two"],
                "last_team": ["Alpha", "Beta"],
            }
        ),
        identities,
    )
    team_snapshot = attach_team_ids(
        pd.DataFrame({"team": ["Alpha", "Beta"]}), identities
    )
    weekly = attach_weekly_ids(
        {"by_player": {"Player One": {"all": {"rank": 1}}}},
        identities,
        kind="player",
    )

    assert player_snapshot["player_id"].tolist() == [
        "oe:player:one",
        "oe:player:two",
    ]
    assert player_snapshot["team_id"].tolist() == [
        "oe:team:alpha",
        "oe:team:beta",
    ]
    assert team_snapshot["team_id"].tolist() == [
        "oe:team:alpha",
        "oe:team:beta",
    ]
    assert weekly["by_player"]["Player One"]["player_id"] == "oe:player:one"
    assert weekly["identity"]["source_identity_sha256"] == "b" * 64
    assert identities.coverage["status"] == "complete"


def test_public_snapshot_schema_accepts_stable_ids_without_dropping_old_fields() -> None:
    team_table = pa.Table.from_pandas(
        pd.DataFrame(
            {
                "team": ["Alpha"],
                "team_key": ["alpha"],
                "team_id": ["oe:team:alpha"],
                "mu_total": [1500.0],
            }
        ),
        preserve_index=False,
    )
    player_table = pa.Table.from_pandas(
        pd.DataFrame(
            {
                "player": ["Player One"],
                "player_id": ["oe:player:one"],
                "team_id": ["oe:team:alpha"],
                "mu_total": [1500.0],
            }
        ),
        preserve_index=False,
    )

    team_rows, team_columns = serialize_rating_snapshot_rows(
        team_table, RATINGS_SNAPSHOT_COLS
    )
    player_rows, player_columns = serialize_rating_snapshot_rows(
        player_table, PLAYER_RATINGS_SNAPSHOT_COLS
    )

    assert "team_id" in team_columns
    assert team_rows[0]["team_id"] == "oe:team:alpha"
    assert "player_id" in player_columns
    assert "team_id" in player_columns
    assert player_rows[0]["player_id"] == "oe:player:one"
    assert player_rows[0]["team_id"] == "oe:team:alpha"


def test_rank_join_ignores_unmapped_current_rows() -> None:
    rows, blockers, coverage, _ = _rank_diffs(
        [
            {
                "player_id": "oe:player:one",
                "future_player_value_logit": 0.4,
            }
        ],
        pd.DataFrame(
            {
                "player_id": ["oe:player:one", pd.NA],
                "mu_effective": [1600.0, 1500.0],
            }
        ),
        identity="player_id",
        future_value="future_player_value_logit",
        current_value_candidates=("mu_effective",),
    )

    assert blockers == []
    assert coverage["current_rows"] == 1
    assert coverage["matched_rows"] == 1
    assert coverage["join_rate"] == 1.0
    assert rows[0]["player_id"] == "oe:player:one"


def test_rank_join_ranks_only_on_common_verified_finite_id_universe() -> None:
    rows, blockers, coverage, assignments = _rank_diffs(
        [
            {
                "player_id": "oe:player:a",
                "future_player_value_logit": 30.0,
            },
            {
                "player_id": "oe:player:b",
                "future_player_value_logit": 20.0,
            },
            {
                "player_id": "oe:player:c",
                "future_player_value_logit": 10.0,
            },
        ],
        pd.DataFrame(
            {
                "player_id": [
                    "oe:player:a",
                    "oe:player:b",
                    "oe:player:d",
                ],
                "mu_effective": [100.0, 90.0, 95.0],
            }
        ),
        identity="player_id",
        future_value="future_player_value_logit",
        current_value_candidates=("mu_effective",),
    )

    assert blockers == []
    assert [row["player_id"] for row in rows] == [
        "oe:player:a",
        "oe:player:b",
    ]
    b_row = next(row for row in rows if row["player_id"] == "oe:player:b")
    assert b_row["current_rank"] == 2
    assert b_row["future_rank"] == 2
    assert b_row["rank_delta"] == 0
    assert b_row["rank_universe"] == "common_verified_finite_ids"
    assert b_row["rank_comparability"] == "comparable"
    assert b_row["full_snapshot_rank_status"] == "incomparable"
    assert assignments["oe:player:b"] == {
        "current_rank": 2,
        "future_rank": 2,
        "rank_delta": 0,
        "rank_universe": "common_verified_finite_ids",
        "rank_comparability": "comparable",
        "full_snapshot_rank_status": "incomparable",
    }
    assert coverage["future_rows"] == 3
    assert coverage["current_rows"] == 3
    assert coverage["finite_future_rows"] == 3
    assert coverage["finite_current_rows"] == 3
    assert coverage["common_universe_size"] == 2
    assert coverage["matched_rows"] == 2
    assert coverage["unmatched_rows"] == 1
    assert coverage["rank_universe"] == "common_verified_finite_ids"
    assert coverage["eligibility_filter"] == "verified_nonempty_id_and_finite_value"
    assert coverage["current_value_field"] == "mu_effective"
    assert coverage["future_value_field"] == "future_player_value_logit"
    assert coverage["rank_direction"] == "descending_value_rank_1_highest"
    assert coverage["full_snapshot_ranks"] == {
        "status": "incomparable",
        "reason": "full snapshot ranks use separate universes",
        "current_universe_size": 3,
        "future_universe_size": 3,
        "current_value_field": "mu_effective",
        "future_value_field": "future_player_value_logit",
        "rank_direction": "descending_value_rank_1_highest",
    }
    assert coverage["common_identity_sha256"] == coverage["identity_sha256"]
    assert coverage["paired_row_digest_sha256"] == coverage["paired_row_digest"]
    assert len(coverage["common_identity_sha256"]) == 64
    assert len(coverage["paired_row_digest_sha256"]) == 64

    changed_rows, changed_blockers, changed_coverage, _ = _rank_diffs(
        [
            {"player_id": "oe:player:a", "future_player_value_logit": 30.0},
            {"player_id": "oe:player:b", "future_player_value_logit": 20.0},
            {"player_id": "oe:player:c", "future_player_value_logit": 10.0},
        ],
        pd.DataFrame(
            {
                "player_id": ["oe:player:a", "oe:player:b", "oe:player:d"],
                "mu_effective": [100.0, 89.0, 95.0],
            }
        ),
        identity="player_id",
        future_value="future_player_value_logit",
        current_value_candidates=("mu_effective",),
    )
    assert changed_blockers == []
    assert changed_rows
    assert coverage["paired_row_digest_sha256"] != changed_coverage[
        "paired_row_digest_sha256"
    ]


def test_rank_join_rejects_duplicate_verified_id_before_value_filter() -> None:
    rows, blockers, coverage, assignments = _rank_diffs(
        [{"team_id": "oe:team:a", "future_team_value_logit": 1.0}],
        pd.DataFrame(
            {
                "team_id": ["oe:team:a", "oe:team:a"],
                "mu_effective": [1500.0, float("nan")],
            }
        ),
        identity="team_id",
        future_value="future_team_value_logit",
        current_value_candidates=("mu_effective",),
    )

    assert rows == []
    assert assignments == {}
    assert blockers == ["current_team_id_rating_identity_or_value_ambiguous"]
    assert coverage["common_universe_size"] == 0


def test_rank_join_excludes_nonfinite_values_from_common_universe() -> None:
    rows, blockers, coverage, _ = _rank_diffs(
        [
            {"team_id": "oe:team:a", "future_team_value_logit": float("inf")},
            {"team_id": "oe:team:b", "future_team_value_logit": 2.0},
        ],
        pd.DataFrame(
            {
                "team_id": ["oe:team:a", "oe:team:b"],
                "mu_effective": [float("-inf"), 1500.0],
            }
        ),
        identity="team_id",
        future_value="future_team_value_logit",
        current_value_candidates=("mu_effective",),
    )

    assert blockers == []
    assert [row["team_id"] for row in rows] == ["oe:team:b"]
    assert coverage["finite_current_rows"] == 1
    assert coverage["finite_future_rows"] == 1
    assert coverage["common_universe_size"] == 1


def test_rank_diff_artifact_carries_rank_coverage(tmp_path) -> None:
    coverage = {
        "rank_universe": "common_verified_finite_ids",
        "common_universe_size": 1,
        "common_identity_sha256": "b" * 64,
        "paired_row_digest_sha256": "c" * 64,
        "current_value_field": "mu_effective",
        "future_value_field": "future_player_value_logit",
        "rank_direction": "descending_value_rank_1_highest",
    }
    result = FutureValueSnapshotResult(
        status="research_only",
        blockers=(),
        player_rows=(),
        team_rows=(),
        player_rank_diffs=(
            {
                "player_id": "oe:player:a",
                "current_rank": 1,
                "future_rank": 1,
                "rank_delta": 0,
            },
        ),
        team_rank_diffs=(),
        receipt={
            "source": {"source_receipt_sha256": "a" * 64},
            "rank_coverage": {"player": coverage, "team": {}},
        },
    )

    write_snapshot_bundle(tmp_path / "bundle", result)
    payload = json.loads(
        (tmp_path / "bundle" / "future-player-rank-diffs.json").read_text()
    )
    assert payload["rows"]
    assert payload["source_receipt_sha256"] == "a" * 64
    assert payload["rank_coverage"] == coverage
