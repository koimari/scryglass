from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from lol_kills.research.future_value_rating import (
    FutureValueSourceError,
    _strict_prior_block_mean,
    assert_pregame_feature_names,
    bind_accepted_future_value_source,
    future_value_model_contract,
    team_value_difference,
)
from lol_kills.research.future_phase_curve import prepare_phase_frame
from lol_kills.research.future_value_training import (
    FutureValueTrainingError,
    frozen_census,
    verify_annual_sources,
)
from lol_kills.v2.tierlists.accepted_census import census_payload


ROLES = ("top", "jungle", "mid", "bot", "support")


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    game_ids = ("oe-api:1", "oe-api:2")
    maps = pd.DataFrame(
        [
            {
                "game_uid": game_id,
                "date": "2026-08-20T10:00:00Z",
                "y_blue_win": index % 2,
            }
            for index, game_id in enumerate(game_ids)
        ]
        + [
            {
                "game_uid": "oe-api:extra",
                "date": "2026-08-20T10:30:00Z",
                "y_blue_win": 1,
            }
        ]
    )
    player_rows = []
    for game_id in (*game_ids, "oe-api:extra"):
        for side_index, side in enumerate(("Blue", "Red")):
            for role_index, role in enumerate(ROLES):
                row = {
                    "game_uid": game_id,
                    "gameid": game_id,
                    "date": "2026-08-20T10:00:00Z",
                    "side": side,
                    "position": role,
                    "playerid": f"oe:player:{game_id}:{side}:{role}",
                    "teamid": f"oe:team:{game_id}:{side}",
                    "champion": f"Champion-{side_index}-{role_index}",
                }
                for checkpoint in (10, 15, 20, 25):
                    for stem in ("gold", "xp", "cs", "kills", "assists", "deaths"):
                        row[f"{stem}at{checkpoint}"] = 100 + side_index
                player_rows.append(row)
    team_rows = []
    for game_id in (*game_ids, "oe-api:extra"):
        for side_index, side in enumerate(("Blue", "Red")):
            row = {
                "game_uid": game_id,
                "gameid": game_id,
                "date": "2026-08-20T10:00:00Z",
                "side": side,
                "position": "team",
                "teamid": f"oe:team:{game_id}:{side}",
            }
            for checkpoint in (10, 15, 20, 25):
                for stem in ("gold", "xp", "cs", "kills", "assists", "deaths"):
                    row[f"{stem}at{checkpoint}"] = 500 + side_index
            team_rows.append(row)
    return maps, pd.DataFrame(player_rows), pd.DataFrame(team_rows), census_payload(game_ids)


def test_accepted_source_binds_full_census_and_records_extra_rows() -> None:
    maps, players, teams, census = _frames()
    source = bind_accepted_future_value_source(
        maps,
        players,
        teams,
        census=census,
        source_as_of="2026-08-20T11:00:00Z",
    )
    assert source.receipt["source_game_count"] == 2
    assert source.receipt["accepted_game_ids"] == census["game_ids"]
    assert source.receipt["source_identity_sha256"] == census["source_identity_sha256"]
    assert source.receipt["model_eligible_game_count"] == 2
    assert source.receipt["source_extra_game_ids"]["maps"] == ["extra"]
    assert len(source.maps) == 2
    assert len(source.players) == 20
    assert len(source.teams) == 4
    assert source.receipt["checkpoint_coverage"]["player"]["25"]["coverage"] == 1.0


def test_missing_accepted_game_fails_closed() -> None:
    maps, players, teams, census = _frames()
    players = players[players["game_uid"] != "oe-api:2"]
    with pytest.raises(FutureValueSourceError, match="players is missing 1 accepted"):
        bind_accepted_future_value_source(
            maps,
            players,
            teams,
            census=census,
            source_as_of="2026-08-20T11:00:00Z",
        )


def test_missing_stable_identity_is_explicit_model_exclusion() -> None:
    maps, players, teams, census = _frames()
    players.loc[
        (players["game_uid"] == "oe-api:2") & (players.index == players[players["game_uid"] == "oe-api:2"].index[0]),
        "playerid",
    ] = None
    source = bind_accepted_future_value_source(
        maps,
        players,
        teams,
        census=census,
        source_as_of="2026-08-20T11:00:00Z",
    )
    assert source.receipt["model_eligible_game_count"] == 1
    assert source.receipt["model_exclusions"]["by_game"]["2"] == [
        "stable_player_identity_missing"
    ]


def test_exact_roster_identity_mismatch_is_explicit_model_exclusion() -> None:
    maps, players, teams, census = _frames()
    changed_index = players[
        (players["game_uid"] == "oe-api:2") & (players["side"] == "Blue")
    ].index[0]
    players.loc[changed_index, "playerid"] = players.loc[changed_index + 1, "playerid"]
    teams.loc[
        (teams["game_uid"] == "oe-api:2") & (teams["side"] == "Red"),
        "teamid",
    ] = "oe:team:wrong:red"
    source = bind_accepted_future_value_source(
        maps,
        players,
        teams,
        census=census,
        source_as_of="2026-08-20T11:00:00Z",
    )
    assert source.receipt["model_exclusions"]["by_game"]["2"] == [
        "player_identity_not_unique",
        "player_team_row_identity_mismatch",
    ]


def test_cross_side_champion_mirror_is_model_eligible() -> None:
    maps, players, teams, census = _frames()
    game_mask = players["game_uid"].eq("oe-api:2")
    blue_top = players.index[game_mask & players["side"].eq("Blue")][0]
    red_top = players.index[game_mask & players["side"].eq("Red")][0]
    players.loc[red_top, "champion"] = players.loc[blue_top, "champion"]

    source = bind_accepted_future_value_source(
        maps,
        players,
        teams,
        census=census,
        source_as_of="2026-08-20T11:00:00Z",
    )

    assert source.receipt["model_eligible_game_count"] == 2
    assert "2" not in source.receipt["model_exclusions"]["by_game"]


def test_same_side_duplicate_champion_is_model_exclusion() -> None:
    maps, players, teams, census = _frames()
    game_mask = players["game_uid"].eq("oe-api:2")
    blue_rows = players.index[game_mask & players["side"].eq("Blue")]
    players.loc[blue_rows[1], "champion"] = players.loc[blue_rows[0], "champion"]

    source = bind_accepted_future_value_source(
        maps,
        players,
        teams,
        census=census,
        source_as_of="2026-08-20T11:00:00Z",
    )

    assert source.receipt["model_exclusions"]["by_game"]["2"] == [
        "champion_identity_not_unique"
    ]


def test_census_mutation_fails_closed() -> None:
    maps, players, teams, census = _frames()
    changed = copy.deepcopy(census)
    changed["source_identity_sha256"] = "0" * 64
    with pytest.raises(FutureValueSourceError, match="census binding"):
        bind_accepted_future_value_source(
            maps,
            players,
            teams,
            census=changed,
            source_as_of="2026-08-20T11:00:00Z",
        )


def test_pregame_feature_gate_rejects_current_state_and_outcome() -> None:
    assert_pregame_feature_names(
        ["history_player_cspm", "forecast_gold_diff_15", "rank_3_atom_loading_2"]
    )
    with pytest.raises(FutureValueSourceError, match="current-map information"):
        assert_pregame_feature_names(["history_player_cspm", "goldat15", "y_blue_win"])


def test_strict_prior_form_excludes_current_and_same_timestamp_rows() -> None:
    frame = pd.DataFrame(
        [
            {"player": "p1", "date": "2026-08-01T10:00:00Z", "metric": 1.0},
            {"player": "p1", "date": "2026-08-02T10:00:00Z", "metric": 10.0},
            {"player": "p1", "date": "2026-08-02T10:00:00Z", "metric": 20.0},
            {"player": "p1", "date": "2026-08-03T10:00:00Z", "metric": 4.0},
        ]
    )
    result = _strict_prior_block_mean(
        frame,
        entity_column="player",
        date_column="date",
        metric_columns=["metric"],
    )
    assert pd.isna(result.loc[0, "prior_form_metric"])
    assert result.loc[1, "prior_form_metric"] == pytest.approx(1.0)
    assert result.loc[2, "prior_form_metric"] == pytest.approx(1.0)
    assert result.loc[3, "prior_form_metric"] == pytest.approx((1.0 + 10.0 + 20.0) / 3.0)
    assert result["prior_form_metric_support"].tolist() == [0, 1, 1, 3]


def test_strict_prior_form_is_append_invariant() -> None:
    base = pd.DataFrame(
        [
            {"player": "p1", "date": "2026-08-01T10:00:00Z", "metric": 1.0},
            {"player": "p1", "date": "2026-08-02T10:00:00Z", "metric": 3.0},
        ]
    )
    future = pd.concat(
        [
            base,
            pd.DataFrame(
                [{"player": "p1", "date": "2026-08-03T10:00:00Z", "metric": 1000.0}]
            ),
        ],
        ignore_index=True,
    )
    left = _strict_prior_block_mean(
        base,
        entity_column="player",
        date_column="date",
        metric_columns=["metric"],
    )
    right = _strict_prior_block_mean(
        future,
        entity_column="player",
        date_column="date",
        metric_columns=["metric"],
    ).iloc[: len(base)]
    pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True))


def test_team_value_is_side_swap_antisymmetric() -> None:
    blue = [0.4, 0.2, -0.1, 0.3, 0.1]
    red = [0.1, -0.2, 0.2, 0.0, 0.15]
    value = team_value_difference(
        blue,
        red,
        blue_team_residual=0.2,
        red_team_residual=-0.1,
        blue_context_value=-0.05,
        red_context_value=0.08,
    )
    swapped = team_value_difference(
        red,
        blue,
        blue_team_residual=-0.1,
        red_team_residual=0.2,
        blue_context_value=0.08,
        red_context_value=-0.05,
    )
    assert value == pytest.approx(-swapped, abs=1e-12)


def test_contract_keeps_every_authority_unavailable() -> None:
    contract = future_value_model_contract()
    assert contract["status"] == "development_only"
    assert not any(contract["authority"].values())
    assert contract["fit_contract"]["metric_weights"].startswith("fit inside")


def test_frozen_protocol_matches_executable_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "data/lol/v2/evaluation/future-value-rating-protocol-v1.json").read_text(
            encoding="utf-8"
        )
    )
    executable = future_value_model_contract()
    for field in (
        "schema_version",
        "status",
        "estimands",
        "player_components",
        "team_components",
        "phase_outputs",
        "information_boundary",
        "fit_contract",
        "evaluation",
        "authority",
    ):
        assert payload[field] == executable[field]


def test_frozen_census_excludes_named_source_rows() -> None:
    maps = pd.DataFrame(
        {
            "game_uid": ["oe-api:1", "oe-api:2", "oe-api:3"],
        }
    )
    expected = census_payload(["oe-api:1", "oe-api:2"])
    freeze = {
        "accepted_census": {
            "excluded_game_ids": ["oe-api:3"],
            "source_game_count": expected["game_count"],
            "source_identity_sha256": expected["source_identity_sha256"],
        },
        "unfiltered_source_game_count": 3,
        "unfiltered_source_identity_sha256": census_payload(
            ["oe-api:1", "oe-api:2", "oe-api:3"]
        )["source_identity_sha256"],
    }
    assert frozen_census(maps, freeze) == expected


def test_annual_source_verification_rejects_changed_bytes(tmp_path: Path) -> None:
    raw = b"gameid,side\n1,Blue\n"
    path = tmp_path / "2026.csv"
    path.write_bytes(raw)
    freeze = {
        "oe_annual_sources": [
            {
                "year": 2026,
                "name": path.name,
                "bytes": len(raw),
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
            }
        ]
    }
    assert verify_annual_sources(tmp_path, freeze)["2026"]["bytes"] == len(raw)
    path.write_bytes(raw + b"2,Red\n")
    with pytest.raises(FutureValueTrainingError, match="source changed"):
        verify_annual_sources(tmp_path, freeze)


def test_phase_targets_are_censored_after_a_short_game() -> None:
    maps = pd.DataFrame(
        [
            {
                "game_uid": "oe-api:1",
                "date": "2026-08-20T10:00:00Z",
                "gamelength": 20 * 60,
            }
        ]
    )
    teams = pd.DataFrame(
        [
            {
                "game_uid": "oe-api:1",
                "date": "2026-08-20T10:00:00Z",
                "side": side,
                **{
                    f"{kind}at{phase}": base + phase
                    for kind, base in (("gold", gold), ("xp", xp))
                    for phase in (10, 15, 20, 25)
                },
            }
            for side, gold, xp in (("Blue", 1000, 800), ("Red", 900, 750))
        ]
    )
    result = prepare_phase_frame(maps, teams)
    assert result.loc[0, "gold_diff_20"] == 100
    assert pd.isna(result.loc[0, "gold_diff_25"])
    assert bool(result.loc[0, "gold_diff_25_censored"])
