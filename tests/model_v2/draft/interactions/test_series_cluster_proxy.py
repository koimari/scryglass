from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from lol_kills.v2.draft.interactions.series_cluster_proxy import (
    MAP_COLUMNS,
    PINNED_PREFLIGHT_PAYLOAD_SHA256,
    DEFAULT_ARTIFACT_PATH,
    DependenceClusterProxyError,
    analyze_frame,
    assert_rolling_folds_do_not_split_cluster,
    canonical_bytes,
    canonical_sha256,
    load_and_replay_nonpromotable_fixture_artifact,
    load_and_replay_artifact,
    map_weighted_cluster_bootstrap_replicate,
    partition_sha256,
    validate_artifact,
    write_nonpromotable_fixture_artifact,
)


def _preflight() -> dict:
    return {
        "artifact_sha256": PINNED_PREFLIGHT_PAYLOAD_SHA256,
        "source": {
            "maps": {"raw_sha256": "a" * 64},
            "player_games": {"raw_sha256": "b" * 64},
        },
        "generator": {"version": "fixture"},
    }


def _row(
    number: int,
    *,
    game_id: str | None = None,
    date: str | None = None,
    patch: float = 16.1,
    blue: str = "team-a",
    red: str = "team-b",
    league: str = "LEC",
    split: str | None = "Spring",
    url: str | None = None,
    lp_game_id: str | None = None,
) -> dict:
    identifier = game_id or f"game-{number}"
    return {
        "oe_gameid": identifier,
        "game_uid": identifier,
        "url": url,
        "league": league,
        "oe_year": 2026,
        "split": split,
        "playoffs": 0,
        "date": pd.Timestamp(date or f"2026-01-01T{10 + number:02}:00:00"),
        "game": number,
        "patch": patch,
        "competition_scope": "regional",
        "event_kind": "domestic",
        "is_international": False,
        "blue_team_key": blue,
        "red_team_key": red,
        "source_lp": lp_game_id is not None,
        "lp_matched": lp_game_id is not None,
        "lp_game_id": lp_game_id,
    }


def _analyze(rows: list[dict]) -> dict:
    return analyze_frame(
        pd.DataFrame(rows, columns=MAP_COLUMNS),
        maps_locator="maps.parquet",
        maps_raw_sha256="c" * 64,
        preflight_payload=_preflight(),
        preflight_locator="preflight.json",
        preflight_raw_sha256="d" * 64,
    )


def _cluster_ids(payload: dict) -> dict[str, str]:
    return {
        row["game_id"]: row["dependence_cluster_id"]
        for row in payload["assignments"]
    }


def test_row_order_and_side_orientation_are_deterministic() -> None:
    rows = [_row(1), _row(2), _row(3)]
    first = _analyze(rows)
    reversed_rows = list(reversed(copy.deepcopy(rows)))
    for row in reversed_rows:
        row["blue_team_key"], row["red_team_key"] = (
            row["red_team_key"],
            row["blue_team_key"],
        )
    second = _analyze(reversed_rows)
    assert _cluster_ids(first) == _cluster_ids(second)
    assert first["cluster_arithmetic"] == second["cluster_arithmetic"]


def test_non_increasing_counter_resets_and_never_searches_backward() -> None:
    payload = _analyze(
        [
            _row(1, game_id="old-1", date="2026-01-01T10:00:00"),
            _row(1, game_id="new-1", date="2026-01-01T11:00:00"),
            _row(2, game_id="new-2", date="2026-01-01T12:00:00"),
        ]
    )
    ids = _cluster_ids(payload)
    assert ids["old-1"] != ids["new-1"]
    assert ids["new-1"] == ids["new-2"]


def test_strict_counter_gap_is_kept_but_manifested() -> None:
    payload = _analyze([_row(1), _row(2), _row(3), _row(5)])
    assert payload["cluster_arithmetic"]["dependence_clusters"] == 1
    assert payload["cluster_arithmetic"]["continued_counter_gaps"] == [
        {"counter_step": 2, "continuations": 1}
    ]
    assert payload["sensitivities"]["exact_counter_step"]["dependence_clusters"] == 2


def test_cross_midnight_20h_resume_is_kept_at_36h_and_split_at_6h_12h() -> None:
    payload = _analyze(
        [
            _row(1, date="2026-01-01T23:00:00"),
            _row(2, date="2026-01-02T19:14:00"),
        ]
    )
    assert payload["cluster_arithmetic"]["dependence_clusters"] == 1
    assert payload["cluster_arithmetic"]["cross_midnight_clusters"] == 1
    assert payload["sensitivities"]["gap_6h"]["dependence_clusters"] == 2
    assert payload["sensitivities"]["gap_12h"]["dependence_clusters"] == 2
    assert payload["sensitivities"]["calendar_day"]["dependence_clusters"] == 2


def test_patch_change_and_gap_over_36h_split_without_excluding_maps() -> None:
    payload = _analyze(
        [
            _row(1, date="2026-01-01T10:00:00"),
            _row(2, date="2026-01-01T11:00:00", patch=16.11),
            _row(3, date="2026-01-03T12:00:00", patch=16.11),
        ]
    )
    assert payload["eligibility"]["assigned_maps"] == 3
    assert payload["cluster_arithmetic"]["dependence_clusters"] == 3


def test_exact_context_time_counter_collision_excludes_all_candidates() -> None:
    left = _row(1, game_id="collision-a")
    right = _row(1, game_id="collision-b")
    payload = _analyze([left, right, _row(2, game_id="kept-2")])
    assert payload["eligibility"]["assigned_maps"] == 1
    assert payload["eligibility"]["exclusion_ledger"] == [
        {"game_id": "collision-a", "reason": "exact_context_time_game_collision"},
        {"game_id": "collision-b", "reason": "exact_context_time_game_collision"},
    ]


@pytest.mark.parametrize("mutation", ("missing_id", "ambiguous_id", "self_pair", "counter"))
def test_invalid_or_ambiguous_identity_is_excluded(mutation: str) -> None:
    row = _row(1)
    if mutation == "missing_id":
        row["oe_gameid"] = None
        row["game_uid"] = None
    elif mutation == "ambiguous_id":
        row["game_uid"] = "different"
    elif mutation == "self_pair":
        row["red_team_key"] = row["blue_team_key"]
    else:
        row["game"] = 0
    payload = _analyze([row])
    assert payload["eligibility"]["assigned_maps"] == 0
    assert payload["eligibility"]["excluded_maps"] == 1


def test_lpl_and_leaguepedia_oracles_agree() -> None:
    rows = [
        _row(
            number,
            game_id=f"12345-12345_game_{number}",
            league="LPL",
            url="https://lpl.qq.com/es/stats.shtml?bmid=12345",
            lp_game_id=f"LPL/2026 Season/Split 1_Week 1_1_{number}",
        )
        for number in (1, 2, 3)
    ]
    payload = _analyze(rows)
    for oracle in payload["oracle_audit"].values():
        if isinstance(oracle, dict):
            assert oracle["maps"] == 3
            assert oracle["oracle_groups"] == 1
            assert oracle["split_oracle_groups"] == 0
            assert oracle["merged_oracle_groups"] == 0


def test_maximum_size_and_assignment_arithmetic_are_enforced() -> None:
    payload = _analyze([_row(number) for number in range(1, 6)])
    assert payload["cluster_arithmetic"]["maximum_cluster_size"] == 5
    validate_artifact(payload)
    changed = copy.deepcopy(payload)
    changed["cluster_arithmetic"]["assigned_maps"] += 1
    changed.pop("artifact_sha256")
    changed["artifact_sha256"] = canonical_sha256(changed)
    with pytest.raises(DependenceClusterProxyError, match="assignment arithmetic"):
        validate_artifact(changed)


def test_rolling_folds_must_not_split_a_dependence_cluster() -> None:
    payload = _analyze([_row(1), _row(2), _row(3)])
    assignments = payload["assignments"]
    assert_rolling_folds_do_not_split_cluster(
        assignments, {"game-1": "train", "game-2": "train", "game-3": "train"}
    )
    with pytest.raises(DependenceClusterProxyError, match="split"):
        assert_rolling_folds_do_not_split_cluster(
            assignments, {"game-1": "train", "game-2": "test", "game-3": "test"}
        )


def test_contract_requires_map_weighted_cluster_resampling() -> None:
    payload = _analyze([_row(1), _row(2)])
    law = payload["downstream_contract"]["bootstrap_sampling_law"]
    assert law["draws_per_replicate"] == "K"
    assert law["replacement"] is True
    assert "uniform" in law["draw_distribution"]
    assert "probability-proportional-to-size cluster draws" in law["forbidden"]


def test_partition_hash_is_label_and_row_order_invariant() -> None:
    payload = _analyze([_row(1), _row(2), _row(3), _row(1, game_id="other-1", blue="c", red="d")])
    assignments = payload["assignments"]
    relabeled = [
        {
            "game_id": assignment["game_id"],
            "dependence_cluster_id": (
                "renamed-a"
                if assignment["dependence_cluster_id"]
                == assignments[0]["dependence_cluster_id"]
                else "renamed-b"
            ),
        }
        for assignment in reversed(assignments)
    ]
    assert partition_sha256(assignments) == partition_sha256(relabeled)


def test_every_sensitivity_has_auditable_partition_comparison() -> None:
    payload = _analyze(
        [
            _row(1, date="2026-01-01T23:00:00"),
            _row(2, date="2026-01-02T19:14:00"),
        ]
    )
    main_hash = payload["cluster_arithmetic"]["partition_sha256"]
    for diagnostic in payload["sensitivities"].values():
        comparison = diagnostic["comparison_to_main"]
        assert comparison["main_partition_sha256"] == main_hash
        assert comparison["candidate_partition_sha256"] == diagnostic["partition_sha256"]
        assert isinstance(comparison["changed_maps"], list)
    assert payload["sensitivities"]["gap_36h"]["partition_sha256"] == main_hash
    assert payload["sensitivities"]["gap_36h"]["comparison_to_main"] == {
        "main_partition_sha256": main_hash,
        "candidate_partition_sha256": main_hash,
        "partition_equal": True,
        "main_clusters_split": 0,
        "candidate_clusters_merging_main": 0,
        "copartition_pairs_split": 0,
        "copartition_pairs_joined": 0,
        "changed_maps": [],
    }
    assert payload["sensitivities"]["gap_12h"]["comparison_to_main"][
        "copartition_pairs_split"
    ] == 1


def test_uniform_cluster_bootstrap_preserves_map_weighted_estimand() -> None:
    result = map_weighted_cluster_bootstrap_replicate(
        {
            "one-map": {"cluster_delta_total": 1.0, "cluster_map_count": 1},
            "three-map": {"cluster_delta_total": 0.0, "cluster_map_count": 3},
        },
        seed=5,
    )
    assert result["observed_map_weighted_point_estimate"] == pytest.approx(0.25)
    assert result["observed_map_weighted_point_estimate"] != pytest.approx(0.5)
    assert result["draw_count"] == result["observed_cluster_count"] == 2
    multiplicities = {
        item["dependence_cluster_id"]: item
        for item in result["draw_multiplicities"]
    }
    assert sum(item["draw_multiplicity"] for item in multiplicities.values()) == 2
    expected_total = sum(
        item["draw_multiplicity"] * item["cluster_delta_total"]
        for item in multiplicities.values()
    )
    expected_count = sum(
        item["draw_multiplicity"] * item["cluster_map_count"]
        for item in multiplicities.values()
    )
    assert result["replicate"] == pytest.approx(expected_total / expected_count)


@pytest.mark.parametrize("mutation", ("assignments", "oracle", "sensitivity"))
def test_source_replay_rejects_caller_rehashed_mutation(
    tmp_path: Path, mutation: str
) -> None:
    maps_path = tmp_path / "maps.parquet"
    preflight_path = tmp_path / "preflight.json"
    artifact_path = tmp_path / "artifact.json"
    pd.DataFrame([_row(1), _row(2)], columns=MAP_COLUMNS).to_parquet(
        maps_path, index=False
    )
    preflight_path.write_bytes(canonical_bytes(_preflight()))
    payload = write_nonpromotable_fixture_artifact(
        artifact_path,
        maps_path=maps_path,
        preflight_path=preflight_path,
    )
    assert load_and_replay_nonpromotable_fixture_artifact(artifact_path) == payload
    changed = copy.deepcopy(payload)
    if mutation == "assignments":
        for assignment in changed["assignments"]:
            assignment["dependence_cluster_id"] = "dependence-cluster:forged"
    elif mutation == "oracle":
        changed["oracle_audit"]["interpretation"] += " forged"
    else:
        changed["sensitivities"]["gap_6h"]["gap_hours"] = 7.0
    changed.pop("artifact_sha256")
    changed["artifact_sha256"] = canonical_sha256(changed)
    artifact_path.write_bytes(canonical_bytes(changed))
    with pytest.raises(
        DependenceClusterProxyError,
        match="source-backed replay does not match|rule contract mismatch",
    ):
        load_and_replay_nonpromotable_fixture_artifact(artifact_path)


def test_production_loader_rejects_nonpromotable_fixture_without_downgrade(
    tmp_path: Path,
) -> None:
    maps_path = tmp_path / "maps.parquet"
    preflight_path = tmp_path / "preflight.json"
    artifact_path = tmp_path / "artifact.json"
    pd.DataFrame([_row(1), _row(2)], columns=MAP_COLUMNS).to_parquet(
        maps_path, index=False
    )
    preflight_path.write_bytes(canonical_bytes(_preflight()))
    write_nonpromotable_fixture_artifact(
        artifact_path, maps_path=maps_path, preflight_path=preflight_path
    )
    with pytest.raises(DependenceClusterProxyError, match="nonpinned source mode"):
        load_and_replay_artifact(artifact_path)


def test_fixture_loader_rejects_noncanonical_persisted_bytes(tmp_path: Path) -> None:
    maps_path = tmp_path / "maps.parquet"
    preflight_path = tmp_path / "preflight.json"
    artifact_path = tmp_path / "artifact.json"
    pd.DataFrame([_row(1)], columns=MAP_COLUMNS).to_parquet(maps_path, index=False)
    preflight_path.write_bytes(canonical_bytes(_preflight()))
    payload = write_nonpromotable_fixture_artifact(
        artifact_path, maps_path=maps_path, preflight_path=preflight_path
    )
    artifact_path.write_bytes(b"\n" + canonical_bytes(payload))
    with pytest.raises(DependenceClusterProxyError, match="not canonical"):
        load_and_replay_nonpromotable_fixture_artifact(artifact_path)


def test_validate_requires_registry_and_exclusion_ledger_arithmetic() -> None:
    payload = _analyze([_row(1)])
    changed = copy.deepcopy(payload)
    changed["eligibility"]["registry_maps"] += 1
    changed.pop("artifact_sha256")
    changed["artifact_sha256"] = canonical_sha256(changed)
    with pytest.raises(DependenceClusterProxyError, match="eligibility arithmetic"):
        validate_artifact(changed)


def test_validate_rejects_rehashed_gap36_partition_mutation() -> None:
    payload = _analyze([_row(1), _row(2)])
    changed = copy.deepcopy(payload)
    changed["sensitivities"]["gap_36h"]["partition_sha256"] = "0" * 64
    changed["sensitivities"]["gap_36h"]["comparison_to_main"][
        "candidate_partition_sha256"
    ] = "0" * 64
    changed["sensitivities"]["gap_36h"]["comparison_to_main"][
        "partition_equal"
    ] = False
    changed.pop("artifact_sha256")
    changed["artifact_sha256"] = canonical_sha256(changed)
    with pytest.raises(DependenceClusterProxyError, match="gap_36h partition"):
        validate_artifact(changed)


def test_validate_recomputes_rehashed_sensitivity_comparison_arithmetic() -> None:
    payload = _analyze(
        [
            _row(1, date="2026-01-01T23:00:00"),
            _row(2, date="2026-01-02T19:14:00"),
        ]
    )
    changed = copy.deepcopy(payload)
    comparison = changed["sensitivities"]["gap_6h"]["comparison_to_main"]
    assert comparison["changed_maps"]
    assert comparison["copartition_pairs_split"] == 1
    comparison["changed_maps"] = []
    comparison["copartition_pairs_split"] = 999_999
    changed.pop("artifact_sha256")
    changed["artifact_sha256"] = canonical_sha256(changed)
    with pytest.raises(DependenceClusterProxyError, match="partition comparison"):
        validate_artifact(changed)


def test_validate_rejects_duplicate_or_assigned_exclusion_ledger_ids() -> None:
    collision_a = _row(1, game_id="collision-a")
    collision_b = _row(1, game_id="collision-b")
    payload = _analyze([collision_a, collision_b, _row(2, game_id="kept-2")])
    changed = copy.deepcopy(payload)
    changed["eligibility"]["exclusion_ledger"] = [
        {
            "game_id": "kept-2",
            "reason": "exact_context_time_game_collision",
        },
        {
            "game_id": "kept-2",
            "reason": "exact_context_time_game_collision",
        },
    ]
    changed.pop("artifact_sha256")
    changed["artifact_sha256"] = canonical_sha256(changed)
    with pytest.raises(
        DependenceClusterProxyError, match="unique, disjoint, and canonical"
    ):
        validate_artifact(changed)


def test_validate_rejects_whitespace_variant_of_same_exclusion_id() -> None:
    collision_a = _row(1, game_id="collision-a")
    collision_b = _row(1, game_id="collision-b")
    payload = _analyze([collision_a, collision_b, _row(2, game_id="kept-2")])
    changed = copy.deepcopy(payload)
    changed["eligibility"]["exclusion_ledger"] = [
        {
            "game_id": " collision-a ",
            "reason": "exact_context_time_game_collision",
        },
        {
            "game_id": "collision-a",
            "reason": "exact_context_time_game_collision",
        },
    ]
    changed.pop("artifact_sha256")
    changed["artifact_sha256"] = canonical_sha256(changed)
    with pytest.raises(DependenceClusterProxyError, match="entry is invalid"):
        validate_artifact(changed)


def test_validate_binds_registry_to_source_map_row_count() -> None:
    payload = _analyze([_row(1), _row(2)])
    changed = copy.deepcopy(payload)
    changed["source"]["maps"]["rows"] = 1
    changed.pop("artifact_sha256")
    changed["artifact_sha256"] = canonical_sha256(changed)
    with pytest.raises(
        DependenceClusterProxyError, match="registry does not match source map rows"
    ):
        validate_artifact(changed)


def test_production_loader_rejects_rehashed_embedded_source_pin(
    tmp_path: Path,
) -> None:
    payload = json.loads(DEFAULT_ARTIFACT_PATH.read_text(encoding="utf-8"))
    payload["source"]["maps"]["raw_sha256"] = "0" * 64
    payload.pop("artifact_sha256")
    payload["artifact_sha256"] = canonical_sha256(payload)
    forged = tmp_path / "forged.json"
    forged.write_bytes(canonical_bytes(payload))
    with pytest.raises(DependenceClusterProxyError, match="embedded production pins"):
        load_and_replay_artifact(forged)


def test_preflight_payload_pin_is_not_replaceable_by_caller_hashing() -> None:
    preflight = _preflight()
    preflight["artifact_sha256"] = "0" * 64
    with pytest.raises(DependenceClusterProxyError, match="preflight payload pin"):
        analyze_frame(
            pd.DataFrame([_row(1)], columns=MAP_COLUMNS),
            maps_locator="maps",
            maps_raw_sha256="a" * 64,
            preflight_payload=preflight,
            preflight_locator="preflight",
            preflight_raw_sha256="b" * 64,
        )
