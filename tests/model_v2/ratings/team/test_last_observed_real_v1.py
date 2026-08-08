from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

import lol_kills.v2.ratings.team.last_observed_real_v1 as observed


TABLE_SHA256 = "6b807527b1a41a17622015b7d89bc2a38cc05b6058f2a47cd7793a45b5ca3fdf"
TABLE_RAW_SHA256 = "787a3a3e8e9e63eadf2963bbc060c0a4401b6306ff2011baf980d90ec574ec76"


def _checked_artifact() -> dict:
    return json.loads(observed.LAST_OBSERVED_ARTIFACT_PATH.read_text(encoding="utf-8"))


def _recompute_receipt_set(value: dict) -> None:
    for receipt in value["receipts"]:
        unsigned = dict(receipt)
        unsigned.pop("receipt_sha256", None)
        receipt["receipt_sha256"] = observed._sha256(unsigned)
    unsigned = dict(value)
    unsigned.pop("receipt_set_sha256", None)
    value["receipt_set_sha256"] = observed._sha256(unsigned)


def _recompute_table(value: dict) -> None:
    unsigned = dict(value)
    unsigned.pop("artifact_sha256", None)
    value["artifact_sha256"] = observed._sha256(unsigned)


def _map_for_team(team_id: str):
    for item in observed._maps_by_id(observed._load_pinned_g1_input()).values():
        if any(player.source_team_id == team_id for player in item.player_observations):
            return item
    raise AssertionError(f"no accepted G1 map for {team_id}")


def test_checked_track_a_artifact_is_pinned_non_predictive_and_has_frozen_raw_identity() -> None:
    raw = observed.LAST_OBSERVED_ARTIFACT_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == TABLE_RAW_SHA256
    artifact = _checked_artifact()
    verified = observed.verify_last_observed_lpl_team_table(artifact, expected_artifact_sha256=TABLE_SHA256)
    assert verified["artifact_sha256"] == TABLE_SHA256
    assert verified["result_state"] == "PRIVATE_LAST_OBSERVED_DESCRIPTIVE_TABLE"
    assert verified["predictive_comparison"]["status"] == "unavailable"
    assert {"current_roster", "official_roster", "active_roster", "forecast", "publication", "final_holdout", "team_rank"} <= set(verified["private_scope"]["blocked"])


def test_track_a_has_nonempty_rated_and_withheld_population_with_exact_partition_counts() -> None:
    artifact = _checked_artifact()
    assert artifact["counts"] == {
        "teams_observed": 16,
        "rated": 14,
        "withheld": 2,
        "withheld_by_reason": {"STALE_LAST_OBSERVED_RECEIPT": 2},
        "rated_receipt_partition_memberships": {"VALIDATION": 14},
        "withheld_receipt_partition_memberships": {"TRAIN": 2},
    }
    assert len(artifact["rated_teams"]) == 14
    assert len(artifact["withheld_teams"]) == 2
    assert all(item["status"] == "private_development_only" for item in artifact["rated_teams"])


def test_receipt_labels_are_team_specific_and_strictly_before_the_separate_table_boundary() -> None:
    artifact = _checked_artifact()
    frozen = datetime.fromisoformat(artifact["frozen_as_of_source_local"])
    boundary = artifact["label"]
    assert boundary == "last observed roster table at frozen boundary 2026-06-01"
    for item in artifact["rated_teams"]:
        receipt = item["last_observed_exact_five_receipt"]
        date = receipt["source_local_event_start"][:10]
        assert datetime.fromisoformat(receipt["source_local_event_start"]) < frozen
        assert receipt["label"] == f"last observed roster as of {date}"
        assert item["last_observed_source_local_date"] == date
        assert item["last_observed_label"] == f"last observed roster as of {date}"
        assert item["table_label"] == boundary == receipt["table_label"]
        assert date not in boundary


def test_each_rated_receipt_replays_to_the_latest_accepted_observed_team_time_independent_of_map_iteration_order(monkeypatch) -> None:
    baseline = observed.build_last_observed_exact_five_receipts()
    maps = observed._maps_by_id(observed._load_pinned_g1_input())
    for receipt in baseline["receipts"]:
        team_id = receipt["team_id"]
        latest = max(
            item.source_local_event_start
            for item in maps.values()
            if any(player.source_team_id == team_id for player in item.player_observations)
        )
        assert receipt["source_local_event_start"] == latest

    monkeypatch.setattr(observed, "_maps_by_id", lambda _input: dict(reversed(list(maps.items()))))
    assert observed.build_last_observed_exact_five_receipts() == baseline


def test_freshness_diagnostic_replays_full_population_nearest_rank_and_declared_ceiling() -> None:
    diagnostic = _checked_artifact()["freshness_policy"]["supporting_diagnostic"]
    assert diagnostic["interappearance_gap_n"] == 855
    assert diagnostic["quantile"] == 0.90
    assert diagnostic["quantile_convention"] == "nearest_rank_ceiling_zero_based_index"
    assert diagnostic["quantile_index"] == math.ceil(0.90 * 855) - 1 == 769
    assert diagnostic["p90_days"] == 14
    assert diagnostic["cushion_days"] == 7
    assert diagnostic["ceiling_days"] == 21
    assert diagnostic["p90_days"] + diagnostic["cushion_days"] == diagnostic["ceiling_days"]
    assert diagnostic["rated_count_not_used"] is True


def test_exact_player_only_aggregation_and_covariance_are_numerically_reproducible_for_one_rated_row() -> None:
    artifact = _checked_artifact()
    row = artifact["rated_teams"][0]
    receipt = row["last_observed_exact_five_receipt"]
    ratings = observed._posterior_by_player(observed._load_pinned_player_artifact())
    expected_mean = sum(ratings[item["player_id"]]["posterior_mean"] for item in receipt["player_ids_by_role"]) / 5.0
    expected_variance = sum(ratings[item["player_id"]]["posterior_uncertainty"] ** 2 for item in receipt["player_ids_by_role"]) / 25.0
    rating = row["rating"]
    assert rating["team_posterior_display_mean"] == pytest.approx(expected_mean)
    assert rating["team_posterior_display_variance"] == pytest.approx(expected_variance)
    assert rating["team_posterior_display_uncertainty"] == pytest.approx(math.sqrt(expected_variance))
    covariance = rating["player_display_covariance"]
    assert len(covariance) == 5 and all(len(line) == 5 for line in covariance)
    assert all(covariance[i][j] == 0.0 for i in range(5) for j in range(5) if i != j)
    assert rating["covariance_assumption"]["kind"] == "DIAGONAL_ASSUMED_DENSITY_REPRESENTATION"
    assert rating["covariance_assumption"]["joint_covariance_status"] == "unavailable"
    for name in ("lineup_synergy", "policy", "league_rating"):
        assert rating["components"][name]["status"] == "unavailable"
        assert rating["components"][name]["value"] is None


def test_duplicate_or_missing_role_closure_is_withheld_not_coerced() -> None:
    artifact = _checked_artifact()
    team_id = artifact["rated_teams"][0]["team_id"]
    source = _map_for_team(team_id)
    focal = [player for player in source.player_observations if player.source_team_id == team_id]
    broken_count = replace(source, player_observations=tuple(player for player in source.player_observations if player is not focal[0]))
    valid, invalid = observed._lineups_for_map(broken_count)
    assert team_id not in valid
    assert invalid[team_id] == "MISSING_OR_DUPLICATE_ROLES"

    duplicate_role = replace(focal[1], role=focal[0].role)
    broken_role = replace(source, player_observations=tuple(duplicate_role if player is focal[1] else player for player in source.player_observations))
    valid, invalid = observed._lineups_for_map(broken_role)
    assert team_id not in valid
    assert invalid[team_id] == "MISSING_OR_DUPLICATE_ROLES"


def test_cross_team_duplicate_player_identity_is_withheld_not_silently_accepted() -> None:
    source = next(iter(observed._maps_by_id(observed._load_pinned_g1_input()).values()))
    first_team = source.player_observations[0].source_team_id
    second_team = next(player.source_team_id for player in source.player_observations if player.source_team_id != first_team)
    borrowed = next(player for player in source.player_observations if player.source_team_id == first_team)
    victim = next(player for player in source.player_observations if player.source_team_id == second_team)
    collision = replace(victim, source_player_id=borrowed.source_player_id)
    colliding_map = replace(
        source,
        player_observations=tuple(collision if player is victim else player for player in source.player_observations),
    )
    valid, invalid = observed._lineups_for_map(colliding_map)
    assert first_team not in valid and second_team not in valid
    assert invalid[first_team] == invalid[second_team] == "CROSS_TEAM_PLAYER_IDENTITY_COLLISION"


def test_freshness_equality_is_rated_while_one_second_over_the_ceiling_is_withheld(monkeypatch) -> None:
    baseline = observed.build_last_observed_exact_five_receipts()
    frozen = datetime.fromisoformat(observed.FROZEN_AS_OF_SOURCE_LOCAL)
    focal_team = baseline["receipts"][0]["team_id"]

    def altered(age_seconds: float) -> dict:
        value = deepcopy(baseline)
        receipt = next(item for item in value["receipts"] if item["team_id"] == focal_team)
        timestamp = (frozen - timedelta(seconds=age_seconds)).isoformat()
        receipt["source_local_event_start"] = timestamp
        receipt["label"] = f"last observed roster as of {timestamp[:10]}"
        receipt["age_seconds_at_frozen_as_of"] = age_seconds
        return value

    monkeypatch.setattr(observed, "load_last_observed_exact_five_receipts", lambda _value: altered(21 * 86400))
    equality_table = observed.build_last_observed_lpl_team_table()
    assert focal_team in {item["team_id"] for item in equality_table["rated_teams"]}

    monkeypatch.setattr(observed, "load_last_observed_exact_five_receipts", lambda _value: altered(21 * 86400 + 1))
    over_table = observed.build_last_observed_lpl_team_table()
    assert (focal_team, "STALE_LAST_OBSERVED_RECEIPT") in {
        (item["team_id"], item["reason"]) for item in over_table["withheld_teams"]
    }


def test_conflicting_same_time_or_future_source_rows_fail_closed_instead_of_selecting_a_lineup(monkeypatch) -> None:
    artifact = _checked_artifact()
    team_id = artifact["rated_teams"][0]["team_id"]
    source = _map_for_team(team_id)
    maps = observed._maps_by_id(observed._load_pinned_g1_input())
    focal = next(player for player in source.player_observations if player.source_team_id == team_id)

    def altered_map(game_id: str, player_id: str, start: str):
        changed = replace(focal, source_player_id=player_id)
        return replace(
            source,
            source_game_id=game_id,
            source_series_id=f"synthetic:{game_id}",
            source_local_event_start=start,
            player_observations=tuple(changed if player is focal else player for player in source.player_observations),
        )

    diagnostic = observed._freshness_diagnostic(maps)
    conflict_a = altered_map("synthetic-conflict-a", "oe:player:synthetic-a", "2026-05-31T00:00:00")
    conflict_b = altered_map("synthetic-conflict-b", "oe:player:synthetic-b", "2026-05-31T00:00:00")
    conflict_maps = dict(maps, **{conflict_a.source_game_id: conflict_a, conflict_b.source_game_id: conflict_b})
    monkeypatch.setattr(observed, "_maps_by_id", lambda _input: conflict_maps)
    monkeypatch.setattr(observed, "_freshness_diagnostic", lambda _maps: diagnostic)
    conflict = observed.build_last_observed_exact_five_receipts()
    assert next(item for item in conflict["withheld"] if item["team_id"] == team_id)["reason"] == "CONFLICTING_SAME_TIME_LAST_OBSERVATION"

    future = altered_map("synthetic-future", "oe:player:synthetic-future", "2026-06-01T00:00:00")
    monkeypatch.setattr(observed, "_maps_by_id", lambda _input: dict(maps, **{future.source_game_id: future}))
    with pytest.raises(observed.LastObservedTeamUnavailable, match="G1_MAP_AFTER_FROZEN_AS_OF"):
        observed.build_last_observed_exact_five_receipts()


@pytest.mark.parametrize("pin_key", ["g1_rows_sha256", "g2_player_artifact_sha256"])
def test_source_or_player_pin_mutation_plus_self_rehash_cannot_bypass_replay_or_external_identity(pin_key: str) -> None:
    artifact = _checked_artifact()
    receipt_set = deepcopy(artifact["receipt_set"])
    receipt_set["source_pins"][pin_key] = "f" * 64
    for receipt in receipt_set["receipts"]:
        receipt["source_pins"][pin_key] = "f" * 64
    _recompute_receipt_set(receipt_set)
    with pytest.raises(observed.LastObservedTeamUnavailable, match="RECEIPT_SET_SOURCE_PIN_MISMATCH"):
        observed.load_last_observed_exact_five_receipts(receipt_set)

    self_rehashed = deepcopy(artifact)
    self_rehashed["receipt_set"] = receipt_set
    _recompute_table(self_rehashed)
    with pytest.raises(observed.LastObservedTeamError, match="TABLE_ARTIFACT_EXTERNAL_PIN_MISMATCH"):
        observed.verify_last_observed_lpl_team_table(self_rehashed, expected_artifact_sha256=TABLE_SHA256)


def test_track_a_artifact_is_byte_identical_across_two_fresh_processes() -> None:
    root = Path(__file__).parents[4]
    code = (
        "from lol_kills.v2.ratings.team.last_observed_real_v1 import build_last_observed_lpl_team_table; "
        "print(build_last_observed_lpl_team_table()['artifact_sha256'])"
    )
    command = [sys.executable, "-c", code]
    first = subprocess.run(command, cwd=root, check=True, capture_output=True, text=True).stdout
    second = subprocess.run(command, cwd=root, check=True, capture_output=True, text=True).stdout
    assert first == second == TABLE_SHA256 + "\n"


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_writer_rejects_output_aliases_without_touching_backing_bytes(tmp_path: Path, alias_kind: str) -> None:
    backing = tmp_path / "backing.json"
    backing.write_text("do-not-replace", encoding="utf-8")
    output = tmp_path / "artifact.json"
    if alias_kind == "symlink":
        output.symlink_to(backing)
    else:
        os.link(backing, output)
    with pytest.raises(observed.LastObservedTeamError, match="TABLE_OUTPUT_UNSAFE"):
        observed.write_last_observed_lpl_team_table(output)
    assert backing.read_text(encoding="utf-8") == "do-not-replace"


def test_writer_rejects_symlinked_ancestor_and_non_directory_parent(tmp_path: Path) -> None:
    backing_parent = tmp_path / "backing-parent"
    backing_parent.mkdir()
    safe_parent = tmp_path / "safe-parent"
    safe_parent.mkdir()
    unsafe_ancestor = safe_parent / "unsafe-ancestor"
    unsafe_ancestor.symlink_to(backing_parent, target_is_directory=True)
    with pytest.raises(observed.LastObservedTeamError, match="TABLE_OUTPUT_PARENT_UNSAFE"):
        observed.write_last_observed_lpl_team_table(unsafe_ancestor / "nested" / "artifact.json")
    assert not (backing_parent / "nested" / "artifact.json").exists()

    non_directory = tmp_path / "not-a-directory"
    non_directory.write_text("not a directory", encoding="utf-8")
    with pytest.raises(observed.LastObservedTeamError, match="TABLE_OUTPUT_PARENT_UNSAFE"):
        observed.write_last_observed_lpl_team_table(non_directory / "artifact.json")
