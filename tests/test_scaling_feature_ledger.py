from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from lol_kills.research.atomized_rf_composite import (
    AtomizedResearchError,
    RATING_BATCH_POLICY,
    build_scaling_feature_ledger,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


IDS = ("g1", "g2", "g3")
ROLES = ("top", "jng", "mid", "bot", "sup")


def _source_receipt(*, row_digest: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "scryglass:future-value-rating-source:v1",
        "source_as_of": "2026-01-04T00:00:00Z",
        "source_game_count": len(IDS),
        "source_identity_sha256": identity_sha256(IDS),
        "accepted_game_ids": list(IDS),
        "model_eligible_game_count": len(IDS),
        "model_eligible_identity_sha256": identity_sha256(IDS),
        "model_eligible_game_ids": list(IDS),
        "authority": {"research_only": True},
        "source_files": {
            label: {
                "bytes": 1,
                "locator": label,
                "sha256": "a" * 64,
            }
            for label in (
                "annual_2025",
                "annual_2026",
                "bridge_oe_api_meta.json",
                "bridge_oe_api_player_games.parquet",
                "bridge_oe_api_team_games.parquet",
            )
        },
    }
    if row_digest is not None:
        payload["source_row_value_sha256"] = row_digest
    payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return payload


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    maps: list[dict[str, object]] = []
    players: list[dict[str, object]] = []
    teams: list[dict[str, object]] = []
    times = {
        "g1": "2026-01-01T00:00:00Z",
        "g2": "2026-01-01T00:00:00Z",
        "g3": "2026-01-02T00:00:00Z",
    }
    for game_id in IDS:
        maps.append({"game_uid": game_id, "date": times[game_id]})
        for side, team in (("Blue", f"b-{game_id}"), ("Red", f"r-{game_id}")):
            teams.append(
                {
                    "gameid": game_id,
                    "side": side,
                    "teamid": f"oe:team:{team}",
                }
            )
            for role_index, role in enumerate(ROLES):
                row: dict[str, object] = {
                    "gameid": game_id,
                    "side": side,
                    "position": role,
                    "playerid": f"oe:player:{side.casefold()}-{role}",
                    "teamid": f"oe:team:{team}",
                    "champion": f"champion-{role}",
                }
                # The signed OE player checkpoint difference is the source
                # update.  Side labels are swapped without changing it.
                side_sign = 1 if side == "Blue" else -1
                for checkpoint in (10, 15, 20, 25):
                    row[f"golddiffat{checkpoint}"] = side_sign * (100 + role_index * 10)
                    row[f"xpdiffat{checkpoint}"] = side_sign * (50 + role_index * 5)
                players.append(row)
    return pd.DataFrame(maps), pd.DataFrame(players), pd.DataFrame(teams)


def _build(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    *,
    source: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    return build_scaling_feature_ledger(
        maps,
        players,
        teams,
        source_receipt=source or _source_receipt(),
    )


def test_ledger_is_strict_prior_and_same_timestamp_maps_are_independent() -> None:
    maps, players, teams = _frames()
    ledger, receipt = _build(maps, players, teams)

    first = ledger.set_index("game_id")
    assert first.loc["g1", "forecast_gold_available_10"] == 0.0
    assert first.loc["g2", "forecast_gold_available_10"] == 0.0
    assert first.loc["g3", "forecast_gold_available_10"] == 1.0
    assert first.loc["g3", "forecast_gold_diff_10"] == 1200.0
    assert receipt["same_timestamp_policy"] == RATING_BATCH_POLICY
    assert receipt["same_timestamp_batching"] == "score_all_maps_then_update_all_maps"
    assert receipt["fold_evaluation_usable"] is False
    assert receipt["fold_blocker"]
    assert all("target_" not in column for column in ledger.columns)


def test_current_checkpoint_change_does_not_change_own_row_but_changes_later_rows() -> None:
    maps, players, teams = _frames()
    baseline, _ = _build(maps, players, teams)
    mutated_players = players.copy()
    mask = (mutated_players["gameid"] == "g1") & (mutated_players["side"] == "Blue")
    mutated_players.loc[mask, "golddiffat10"] = mutated_players.loc[mask, "golddiffat10"] + 1000
    changed, _ = _build(maps, mutated_players, teams)

    left = baseline.set_index("game_id")
    right = changed.set_index("game_id")
    assert left.loc["g1", "forecast_gold_diff_10"] == right.loc["g1", "forecast_gold_diff_10"]
    assert left.loc["g2", "forecast_gold_diff_10"] == right.loc["g2", "forecast_gold_diff_10"]
    assert left.loc["g3", "forecast_gold_diff_10"] != right.loc["g3", "forecast_gold_diff_10"]


def test_side_swap_negates_signed_forecasts_and_keeps_curve_invariants() -> None:
    maps, players, teams = _frames()
    original, _ = _build(maps, players, teams)
    swapped_players = players.copy()
    swapped_players["side"] = swapped_players["side"].map({"Blue": "Red", "Red": "Blue"})
    swapped_teams = teams.copy()
    swapped_teams["side"] = swapped_teams["side"].map({"Blue": "Red", "Red": "Blue"})
    swapped, _ = _build(maps, swapped_players, swapped_teams)

    left = original.set_index("game_id")
    right = swapped.set_index("game_id")
    for game_id in IDS:
        assert right.loc[game_id, "forecast_gold_diff_10"] == -left.loc[game_id, "forecast_gold_diff_10"]
        assert right.loc[game_id, "forecast_xp_diff_25"] == -left.loc[game_id, "forecast_xp_diff_25"]
        left_crossovers = left.loc[game_id, "forecast_gold_crossover_count"]
        right_crossovers = right.loc[game_id, "forecast_gold_crossover_count"]
        assert (
            pd.isna(left_crossovers) and pd.isna(right_crossovers)
        ) or right_crossovers == left_crossovers
        assert right.loc[game_id, "forecast_curve_available"] == left.loc[game_id, "forecast_curve_available"]


def test_missing_checkpoint_stays_explicitly_missing() -> None:
    maps, players, teams = _frames()
    mutated = players.copy()
    mask = (mutated["gameid"] == "g1") & (mutated["side"] == "Blue") & (mutated["position"] == "top")
    mutated.loc[mask, "golddiffat10"] = None
    ledger, _ = _build(maps, mutated, teams)
    row = ledger.set_index("game_id").loc["g1"]
    assert row["forecast_gold_available_10"] == 0.0
    assert row["forecast_gold_missing_10"] == 1.0
    assert pd.isna(row["forecast_gold_early_mean"])


def test_census_deletion_and_bound_source_value_drift_fail_closed() -> None:
    maps, players, teams = _frames()
    baseline, producer_receipt = _build(maps, players, teams)
    bound_source = _source_receipt(row_digest=str(producer_receipt["source_row_value_sha256"]))
    with pytest.raises(AtomizedResearchError, match="row values"):
        mutated = players.copy()
        mutated.loc[0, "golddiffat10"] = 9999
        _build(maps, mutated, teams, source=bound_source)

    with pytest.raises(AtomizedResearchError, match="census mismatch"):
        _build(maps[maps["game_uid"] != "g2"], players, teams)


def test_excluded_source_maps_are_bound_then_filtered() -> None:
    maps, players, teams = _frames()
    extra_maps: list[dict[str, object]] = []
    extra_players: list[dict[str, object]] = []
    extra_teams: list[dict[str, object]] = []
    for index in range(4, 10):
        game_id = f"excluded-{index}"
        extra_maps.append({"game_uid": game_id, "date": "2026-01-03T00:00:00Z"})
        for side, team in (("Blue", f"b-{game_id}"), ("Red", f"r-{game_id}")):
            extra_teams.append(
                {"gameid": game_id, "side": side, "teamid": f"oe:team:{team}"}
            )
            for role in ROLES:
                row = {
                    "gameid": game_id,
                    "side": side,
                    "position": role,
                    "playerid": f"oe:player:{side.casefold()}-{role}-{game_id}",
                    "teamid": f"oe:team:{team}",
                    "champion": f"champion-{role}",
                }
                for checkpoint in (10, 15, 20, 25):
                    row[f"golddiffat{checkpoint}"] = 1.0
                    row[f"xpdiffat{checkpoint}"] = 1.0
                extra_players.append(row)
    maps = pd.concat([maps, pd.DataFrame(extra_maps)], ignore_index=True)
    players = pd.concat([players, pd.DataFrame(extra_players)], ignore_index=True)
    teams = pd.concat([teams, pd.DataFrame(extra_teams)], ignore_index=True)

    ledger, receipt = _build(maps, players, teams)
    assert list(ledger["game_id"]) == list(IDS)
    assert receipt["excluded_extra_game_count"] == 6
    assert receipt["source_extra_game_ids"]["maps"] == [f"excluded-{index}" for index in range(4, 10)]
    assert receipt["excluded_extra_identity_sha256"] == identity_sha256(
        [f"excluded-{index}" for index in range(4, 10)]
    )


def test_missing_stable_team_ids_use_alias_fallback_for_both_sides() -> None:
    maps, players, teams = _frames()
    aliases = {"Blue": "Blue Org", "Red": "Red Org"}
    players = players.copy()
    teams = teams.copy()
    players["teamname"] = players["side"].map(aliases)
    teams["teamname"] = teams["side"].map(aliases)
    players["teamid"] = None
    teams["teamid"] = None

    ledger, receipt = _build(maps, players, teams)
    assert len(ledger) == len(IDS)
    assert receipt["team_identity"]["mode_counts"] == {
        "stable_oe": 0,
        "alias_fallback": len(IDS),
    }
    assert receipt["team_identity"]["stable_oe_identity_sha256"] == identity_sha256([])
    assert receipt["team_identity"]["alias_fallback_identity_sha256"] == identity_sha256(IDS)


def test_alias_display_name_and_slug_key_resolve_to_one_team() -> None:
    maps, players, teams = _frames()
    maps = maps.copy()
    maps["blue_team"] = "Diversion Gaming"
    maps["blue_team_key"] = "diversion-gaming"
    maps["red_team"] = "Mighty Eagles"
    maps["red_team_key"] = "mighty-eagles"
    players = players.copy()
    teams = teams.copy()
    players["teamname"] = players["side"].map(
        {"Blue": "Diversion Gaming", "Red": "Mighty Eagles"}
    )
    teams["teamname"] = teams["side"].map(
        {"Blue": "Diversion Gaming", "Red": "Mighty Eagles"}
    )
    players.loc[players["side"].eq("Red"), "teamid"] = None
    teams.loc[teams["side"].eq("Red"), "teamid"] = None

    ledger, receipt = _build(maps, players, teams)
    assert len(ledger) == len(IDS)
    assert receipt["team_identity"]["mode_counts"]["alias_fallback"] == len(IDS)


def test_team_identity_mismatch_fails_closed() -> None:
    maps, players, teams = _frames()
    mismatched_stable = teams.copy()
    mask = (mismatched_stable["gameid"] == "g1") & (mismatched_stable["side"] == "Blue")
    mismatched_stable.loc[mask, "teamid"] = "oe:team:wrong"
    with pytest.raises(AtomizedResearchError, match="IDs disagree"):
        _build(maps, players, mismatched_stable)

    aliases = {"Blue": "Blue Org", "Red": "Red Org"}
    alias_players = players.copy()
    alias_teams = teams.copy()
    alias_players["teamid"] = None
    alias_teams["teamid"] = None
    alias_players["teamname"] = alias_players["side"].map(aliases)
    alias_teams["teamname"] = alias_teams["side"].map(aliases)
    alias_players.loc[
        (alias_players["gameid"] == "g1") & (alias_players["side"] == "Blue"),
        "teamname",
    ] = "Different Org"
    with pytest.raises(AtomizedResearchError, match="conflicting team aliases"):
        _build(maps, alias_players, alias_teams)


def test_fold_local_mode_updates_train_only_and_binds_fold_identity() -> None:
    maps, players, teams = _frames()
    maps = maps.copy()
    maps.loc[maps["game_uid"] == "g2", "date"] = "2026-01-02T00:00:00Z"
    maps.loc[maps["game_uid"] == "g3", "date"] = "2026-01-03T00:00:00Z"
    kwargs = {
        "train_game_ids": ["g1"],
        "validation_game_ids": ["g2", "g3"],
        "fit_window_end": "2026-01-02T00:00:00Z",
    }
    baseline, receipt = build_scaling_feature_ledger(
        maps, players, teams, source_receipt=_source_receipt(), **kwargs
    )
    mutated_players = players.copy()
    mask = mutated_players["gameid"] == "g2"
    mutated_players.loc[mask, "golddiffat10"] = mutated_players.loc[mask, "golddiffat10"] + 5000
    changed, changed_receipt = build_scaling_feature_ledger(
        maps, mutated_players, teams, source_receipt=_source_receipt(), **kwargs
    )
    left = baseline.set_index("game_id")
    right = changed.set_index("game_id")
    assert receipt["evaluation_mode"] == "fold_local"
    assert receipt["fold_evaluation_usable"] is True
    assert receipt["train_game_ids"] == ["g1"]
    assert receipt["validation_game_ids"] == ["g2", "g3"]
    assert left.loc["g3", "forecast_gold_diff_10"] == right.loc["g3", "forecast_gold_diff_10"]
    assert changed_receipt["train_identity_sha256"] == receipt["train_identity_sha256"]


def test_receipt_binds_source_identity_and_implementation() -> None:
    maps, players, teams = _frames()
    ledger, receipt = _build(maps, players, teams)
    assert len(ledger) == len(IDS)
    assert receipt["accepted_game_ids"] == list(IDS)
    assert receipt["accepted_game_count"] == len(IDS)
    assert receipt["source_receipt_sha256"] == _source_receipt()["receipt_sha256"]
    assert len(str(receipt["implementation_sha256"])) == 64
    assert len(str(receipt["row_value_digest_sha256"])) == 64
    assert len(str(receipt["receipt_sha256"])) == 64


def test_source_receipt_requires_canonical_research_only_contract() -> None:
    maps, players, teams = _frames()
    forged = _source_receipt()
    forged["forged_extra"] = "resealed"
    unsigned = {key: value for key, value in forged.items() if key != "receipt_sha256"}
    forged["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(AtomizedResearchError, match="unknown fields"):
        _build(maps, players, teams, source=forged)

    blocked = _source_receipt()
    blocked["authority"] = {"research_only": False}
    unsigned = {key: value for key, value in blocked.items() if key != "receipt_sha256"}
    blocked["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(AtomizedResearchError, match="research-only"):
        _build(maps, players, teams, source=blocked)
