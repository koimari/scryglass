from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest

from lol_kills.v2.draft.interactions.representation_assay_preflight import (
    MAP_COLUMNS,
    PLAYER_COLUMNS,
    RepresentationAssayPreflightError,
    analyze_frames,
    build_from_parquet,
    canonical_bytes,
    canonical_sha256,
    load_and_replay_artifact,
    validate_artifact,
    write_artifact,
)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    map_rows = []
    player_rows = []
    roles = (("top", "Top"), ("jng", "Jungle"), ("mid", "Mid"), ("bot", "Bot"), ("sup", "Support"))
    for game_number, patch in ((1, 16.01), (2, 16.02)):
        game_id = f"game-{game_number}"
        map_rows.append(
            {
                "oe_gameid": game_id,
                "datacompleteness": "complete",
                "league": "LEC",
                "year": 2026,
                "date": pd.Timestamp(f"2026-01-0{game_number}T12:00:00"),
                "patch": patch,
                "competition_scope": "domestic",
                "event_kind": "league",
                "is_international": False,
            }
        )
        participant = 0
        for side, prefix in (("Blue", "A"), ("Red", "B")):
            for position, role_name in roles:
                participant += 1
                player_rows.append(
                    {
                        "gameid": game_id,
                        "datacompleteness": "complete",
                        "league": "LEC",
                        "year": 2026,
                        "date": pd.Timestamp(f"2026-01-0{game_number}T12:00:00"),
                        "patch": patch,
                        "participantid": participant,
                        "side": side,
                        "position": position,
                        "champion": f"{prefix}{role_name}{game_number}",
                    }
                )
    return (
        pd.DataFrame(map_rows, columns=MAP_COLUMNS),
        pd.DataFrame(player_rows, columns=PLAYER_COLUMNS),
    )


def _analyze(maps: pd.DataFrame, players: pd.DataFrame) -> dict:
    return analyze_frames(
        maps.loc[:, list(MAP_COLUMNS)],
        players.loc[:, list(PLAYER_COLUMNS)],
        maps_raw_sha256="a" * 64,
        players_raw_sha256="b" * 64,
        maps_locator="maps.parquet",
        players_locator="players.parquet",
    )


def _without_physical_lineage(payload: dict) -> dict:
    result = copy.deepcopy(payload)
    result.pop("artifact_sha256")
    for source in result["source"].values():
        source.pop("raw_sha256")
    return result


def test_exact_observation_counts_and_no_rank_or_effect_claims() -> None:
    maps, players = _frames()
    payload = _analyze(maps, players)
    assert payload["eligibility"] == {
        "registry_maps": 2,
        "valid_maps": 2,
        "excluded_maps": 0,
        "exclusion_reasons": [],
        "rejection_ledger": [],
    }
    diagnostics = payload["design_diagnostics"]
    assert diagnostics["rows"] == 2
    assert diagnostics["nonzero_entries"] == 2 * (1 + 10 + 20 + 25)
    assert payload["observation_contract"]["canonical_ally_observations_per_valid_map"] == 20
    assert (
        payload["observation_contract"]["canonical_cross_team_observations_per_valid_map"]
        == 25
    )
    assert payload["development_only"] is True
    assert payload["outcome_free"] is True
    assert payload["predictive_authority"] is False
    assert payload["representation_rank_selected"] is False
    assert payload["authorizes_model_selection"] is False
    assert payload["authorizes_publication"] is False
    assert payload["content_addressing_confers_authority"] is False
    assert payload["claim_ceiling"]["no_effect_estimates"] is True
    validate_artifact(payload)


def test_row_order_and_side_swap_preserve_empirical_diagnostics() -> None:
    maps, players = _frames()
    first = _analyze(maps, players)
    shuffled = _analyze(
        maps.sample(frac=1, random_state=7).reset_index(drop=True),
        players.sample(frac=1, random_state=11).reset_index(drop=True),
    )
    assert first == shuffled

    swapped_players = players.copy()
    swapped_players["side"] = swapped_players["side"].map({"Blue": "Red", "Red": "Blue"})
    swapped = _analyze(maps, swapped_players)
    for key in ("support", "temporal_overlap", "graph_connectivity", "design_diagnostics"):
        assert first[key] == swapped[key]


def test_malformed_map_fails_closed_into_rejection_ledger() -> None:
    maps, players = _frames()
    malformed = players[
        ~((players["gameid"] == "game-1") & (players["participantid"] == 10))
    ].copy()
    payload = _analyze(maps, malformed)
    assert payload["eligibility"]["valid_maps"] == 1
    assert payload["eligibility"]["excluded_maps"] == 1
    assert payload["eligibility"]["rejection_ledger"] == [
        {"game_id": "game-1", "reason": "player_row_count_not_ten"}
    ]
    assert payload["design_diagnostics"]["rows"] == 1

    duplicate_registry = pd.concat([maps, maps.iloc[[0]]], ignore_index=True)
    payload_duplicate = _analyze(duplicate_registry, players)
    assert payload_duplicate["eligibility"]["rejection_ledger"] == [
        {"game_id": "game-1", "reason": "map_registry_row_count_not_one"}
    ]

    incoherent_date = players.copy()
    incoherent_date.loc[
        (incoherent_date["gameid"] == "game-1")
        & (incoherent_date["participantid"] == 1),
        "date",
    ] = pd.Timestamp("2026-01-09T12:00:00")
    payload_date = _analyze(maps, incoherent_date)
    assert payload_date["eligibility"]["rejection_ledger"] == [
        {"game_id": "game-1", "reason": "map_player_metadata_mismatch"}
    ]

    blank_participant = players.copy()
    blank_participant["participantid"] = blank_participant["participantid"].astype(object)
    blank_participant.loc[
        (blank_participant["gameid"] == "game-1")
        & (blank_participant["participantid"] == 1),
        "participantid",
    ] = " "
    payload_participant = _analyze(maps, blank_participant)
    assert payload_participant["eligibility"]["rejection_ledger"] == [
        {"game_id": "game-1", "reason": "invalid_participant_ids"}
    ]

    mismatched_completeness = players.copy()
    mismatched_completeness.loc[
        mismatched_completeness["gameid"] == "game-1", "datacompleteness"
    ] = "partial"
    payload_completeness = _analyze(maps, mismatched_completeness)
    assert payload_completeness["eligibility"]["rejection_ledger"] == [
        {"game_id": "game-1", "reason": "map_player_datacompleteness_mismatch"}
    ]


def test_patch_tokens_are_exact_centesimal_source_values() -> None:
    maps, players = _frames()
    maps.loc[maps["oe_gameid"] == "game-1", "patch"] = 16.1
    players.loc[players["gameid"] == "game-1", "patch"] = 16.1
    payload = _analyze(maps, players)
    patches = {
        record["patch"]
        for record in payload["support"]["node_by_year_patch_league_role"]
    }
    assert "16.10" in patches

    maps.loc[maps["oe_gameid"] == "game-1", "patch"] = 16.014
    players.loc[players["gameid"] == "game-1", "patch"] = 16.014
    rejected = _analyze(maps, players)
    assert rejected["eligibility"]["rejection_ledger"] == [
        {"game_id": "game-1", "reason": "invalid_map_metadata"}
    ]


def test_parquet_reader_is_outcome_independent_and_hashes_selected_inputs(
    tmp_path: Path,
) -> None:
    maps, players = _frames()
    maps_path = tmp_path / "maps.parquet"
    players_path = tmp_path / "players.parquet"
    maps.assign(blue_result=[0, 1], y_blue_win=[0, 1]).to_parquet(maps_path, index=False)
    players.assign(result=[0, 1] * 10, kills=list(range(20))).to_parquet(
        players_path, index=False
    )
    first = build_from_parquet(maps_path, players_path)

    maps.assign(blue_result=[1, 0], y_blue_win=[1, 0]).to_parquet(maps_path, index=False)
    players.assign(result=[1, 0] * 10, kills=list(reversed(range(20)))).to_parquet(
        players_path, index=False
    )
    second = build_from_parquet(maps_path, players_path)
    assert first["source"]["maps"]["raw_sha256"] != second["source"]["maps"]["raw_sha256"]
    assert (
        first["source"]["player_games"]["raw_sha256"]
        != second["source"]["player_games"]["raw_sha256"]
    )
    assert (
        first["source"]["maps"]["selected_input_sha256"]
        == second["source"]["maps"]["selected_input_sha256"]
    )
    assert (
        first["source"]["player_games"]["selected_input_sha256"]
        == second["source"]["player_games"]["selected_input_sha256"]
    )
    assert _without_physical_lineage(first) == _without_physical_lineage(second)


def test_artifact_validation_rejects_claims_and_hash_drift() -> None:
    maps, players = _frames()
    payload = _analyze(maps, players)
    changed = copy.deepcopy(payload)
    changed["authorizes_model_selection"] = True
    unsigned = dict(changed)
    unsigned.pop("artifact_sha256")
    changed["artifact_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(RepresentationAssayPreflightError, match="authority"):
        validate_artifact(changed)

    changed = copy.deepcopy(payload)
    changed["eligibility"]["valid_maps"] = 999
    with pytest.raises(RepresentationAssayPreflightError, match="canonical payload"):
        validate_artifact(changed)

    changed = copy.deepcopy(payload)
    changed["design_diagnostics"]["nonzero_entries"] += 1
    unsigned = dict(changed)
    unsigned.pop("artifact_sha256")
    changed["artifact_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(RepresentationAssayPreflightError, match="design arithmetic"):
        validate_artifact(changed)


def test_analysis_requires_exact_allowlisted_columns() -> None:
    maps, players = _frames()
    with pytest.raises(RepresentationAssayPreflightError, match="allowlist"):
        analyze_frames(
            maps.assign(result=1),
            players,
            maps_raw_sha256="a" * 64,
            players_raw_sha256="b" * 64,
            maps_locator="maps",
            players_locator="players",
        )


def _write_replay_fixture(tmp_path: Path) -> tuple[Path, dict]:
    maps, players = _frames()
    maps_path = tmp_path / "maps.parquet"
    players_path = tmp_path / "players.parquet"
    artifact_path = tmp_path / "artifact.json"
    maps.to_parquet(maps_path, index=False)
    players.to_parquet(players_path, index=False)
    payload = write_artifact(
        artifact_path,
        maps_path=maps_path,
        players_path=players_path,
    )
    return artifact_path, payload


def _rehash(payload: dict) -> dict:
    changed = copy.deepcopy(payload)
    changed.pop("artifact_sha256", None)
    changed["artifact_sha256"] = canonical_sha256(changed)
    return changed


@pytest.mark.parametrize(
    "mutation",
    (
        "temporal_overlap",
        "graph_connectivity",
        "source_datacompleteness_cohorts",
        "structural_rank_nullity",
        "numeric_spectrum",
    ),
)
def test_source_replay_rejects_caller_rehashed_diagnostic_mutations(
    tmp_path: Path,
    mutation: str,
) -> None:
    artifact_path, payload = _write_replay_fixture(tmp_path)
    assert load_and_replay_artifact(artifact_path) == payload
    changed = copy.deepcopy(payload)
    if mutation == "temporal_overlap":
        changed["temporal_overlap"]["metric"] += " mutated"
    elif mutation == "graph_connectivity":
        changed["graph_connectivity"]["scope_statement"] += " mutated"
    elif mutation == "source_datacompleteness_cohorts":
        changed["support"]["source_datacompleteness_cohorts"][0]["nodes"]["support"][
            "p75"
        ] += 1
    elif mutation == "structural_rank_nullity":
        changed["design_diagnostics"]["structural_rank_upper_bound"] -= 1
        changed["design_diagnostics"]["structural_column_nullity_lower_bound"] += 1
    elif mutation == "numeric_spectrum":
        changed["design_diagnostics"]["numeric_spectrum"][
            "condition_number_nonzero_subspace"
        ] += 1
    changed = _rehash(changed)
    artifact_path.write_bytes(canonical_bytes(changed))
    with pytest.raises(
        RepresentationAssayPreflightError,
        match="source-backed replay does not match",
    ):
        load_and_replay_artifact(artifact_path)
