from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from lol_kills.draft_archetypes import ARCHETYPE_NAMES, champ_tags
from lol_kills.research.elemental_drake_explorer_model import (
    DIRECT_FAMILY,
    PUBLIC_WORDING,
    STATE_NUMERIC,
    _augment_joint_runtime,
    _capture_path_is_legal,
    _cell_is_direct_eligible,
    _champion_catalog,
    _champion_element_support,
    _champion_residual_vocabulary_summary,
    _direct_cell_failed_exposure_rules,
    _direct_cell_design,
    _direct_family_gate,
    _direct_raw_design,
    _design_allocation,
    _design_state,
    _effective_runtime,
    _feature_schema,
    _file_provenance,
    _fit_offset_ridge,
    _fitted_logit_score,
    _freeze_champion_residual_spec,
    _game_weights,
    _identifier_set_provenance,
    _publication_audit_partitions,
    _publication_expansion_audit_gate,
    _serialize_observed_champion_cells,
    _side_assignment_is_valid,
    _StandardizedRuntimeScorer,
    _reconciled_runtime_probability,
    _standardization_draft_rows,
    _standardization_state,
    _standardized_element_rankings,
    _temporal_partitions,
    allocation_counterfactual_rows,
    build_explorer_artifact,
    prepare_joint_rows,
    runtime_predict,
    runtime_score,
)
from lol_kills.research.elemental_drake_model import ELEMENTS, _fit


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "series_id": "series-1",
                "game_id": "game-1",
                "date": "2026-01-15T12:00:00Z",
                "tournament": "LCK - Example",
                "competition": "LCK",
                "league": "LCK",
                "region": "korea",
                "competition_level": "tier1",
                "patch": "16.1",
                "complete": True,
                "winner_team_id": "blue",
                "team_1_id": "blue",
                "team_1_name": "Blue Team",
                "team_1_side": "blue",
                "team_1_champions": json.dumps(
                    ["Jinx", "Orianna", "Ornn", "Sejuani", "Lulu"]
                ),
                "team_1_players": json.dumps(
                    ["blue-1", "blue-2", "blue-3", "blue-4", "blue-5"]
                ),
                "team_1_player_ids": json.dumps(
                    ["b1", "b2", "b3", "b4", "b5"]
                ),
                "team_2_id": "red",
                "team_2_name": "Red Team",
                "team_2_side": "red",
                "team_2_champions": json.dumps(
                    ["Renekton", "Lee Sin", "LeBlanc", "Draven", "Nautilus"]
                ),
                "team_2_players": json.dumps(
                    ["red-1", "red-2", "red-3", "red-4", "red-5"]
                ),
                "team_2_player_ids": json.dumps(
                    ["r1", "r2", "r3", "r4", "r5"]
                ),
            }
        ]
    )


def _events() -> pd.DataFrame:
    rows = []
    # Blue reaches soul on its fourth capture. Red receives two in between.
    sequence = [
        ("blue", "infernal"),
        ("red", "ocean"),
        ("blue", "hextech"),
        ("red", "hextech"),
        ("blue", "hextech"),
        ("blue", "hextech"),
    ]
    blue_stack = 0
    red_stack = 0
    for index, (owner, element) in enumerate(sequence, start=1):
        if owner == "blue":
            blue_stack += 1
            owner_stack = blue_stack
            sign = 1
        else:
            red_stack += 1
            owner_stack = red_stack
            sign = -1
        rows.append(
            {
                "series_id": "series-1",
                "game_id": "game-1",
                "date": "2026-01-15T12:00:00Z",
                "occurred_at": f"2026-01-15T12:{6 + index:02d}:00Z",
                "global_index": index,
                "owner_stack": owner_stack,
                "element": element,
                "time_seconds": 300 + index * 120,
                "owner_team_id": owner,
                "owner_side": owner,
                "state_timing": "previous-envelope",
                "state_lag_seconds": 1.0,
                "owner_net_worth": 10_000 + sign * index * 100,
                "opponent_net_worth": 10_000 - sign * index * 100,
                "gold_diff": sign * index * 200,
                "owner_loadout_value": 8_000 + sign * index * 100,
                "opponent_loadout_value": 8_000 - sign * index * 100,
                "loadout_diff": sign * index * 200,
                "owner_unspent_money": 2_000,
                "opponent_unspent_money": 1_900,
                "unspent_money_diff": 100,
                "owner_top_player_net_worth": 2_500,
                "opponent_top_player_net_worth": 2_400,
                "top_player_net_worth_diff": 100,
                "owner_kills": index,
                "opponent_kills": max(0, index - 1),
                "owner_towers": index // 2,
                "opponent_towers": max(0, index // 2 - 1),
            }
        )
    return pd.DataFrame(rows)


def _eligible_support(
    champion: str,
    *,
    element: str = "infernal",
    games: int = 80,
) -> dict[str, object]:
    return {
        "champion": champion,
        "element": element,
        "featureName": f"champion_direct_inventory::{element}::{champion}",
        "trainingGames": games,
        "trainingSeries": 40,
        "orgRosters": 4,
        "ownershipGames": games // 2,
        "nonOwnershipGames": games // 2,
        "wins": games // 2,
        "losses": games // 2,
        "ownershipWins": games // 4,
        "ownershipLosses": games // 4,
        "nonOwnershipWins": games // 4,
        "nonOwnershipLosses": games // 4,
        "effectiveGames": float(games),
        "supportWeight": float(games),
        "tags": sorted(champ_tags(champion)),
    }


def _direct_fit_rows(games: int = 80) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    fillers = ["Orianna", "Sejuani", "Jinx", "Lulu"]
    for index in range(games):
        aatrox_owns = index % 2 == 0
        owner = "Aatrox" if aatrox_owns else "Urgot"
        opponent = "Urgot" if aatrox_owns else "Aatrox"
        owner_won = int(aatrox_owns)
        first = {
            "own_champions": json.dumps([owner, *fillers]),
            "opp_champions": json.dumps([opponent, *fillers]),
            "post_own_count_infernal": 1,
            "post_opp_count_infernal": 0,
            "perspective_won": owner_won,
        }
        second = {
            "own_champions": first["opp_champions"],
            "opp_champions": first["own_champions"],
            "post_own_count_infernal": 0,
            "post_opp_count_infernal": 1,
            "perspective_won": 1 - owner_won,
        }
        for row in (first, second):
            for element in ELEMENTS:
                row.setdefault(f"post_own_count_{element}", 0)
                row.setdefault(f"post_opp_count_{element}", 0)
            rows.append(row)
    return pd.DataFrame(rows)


def _support_partition_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    team_a = ["Aatrox", "Orianna", "Sejuani", "Jinx", "Lulu"]
    team_b = ["Urgot", "Ahri", "Lee Sin", "Draven", "Nautilus"]
    for index in range(100):
        owns = index % 2 == 0
        won = (index // 2) % 2 == 0
        date = pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(
            hours=index
        )
        for perspective, own, opp, own_team, opp_team in (
            ("team_1", team_a, team_b, f"org-a-{index % 4}", f"org-b-{index % 4}"),
            ("team_2", team_b, team_a, f"org-b-{index % 4}", f"org-a-{index % 4}"),
        ):
            team_one = perspective == "team_1"
            row: dict[str, object] = {
                "series_id": f"pre-series-{index:03d}",
                "game_id": f"pre-game-{index:03d}",
                "date": date,
                "stage": 1,
                "perspective": perspective,
                "own_team_id": own_team,
                "opp_team_id": opp_team,
                "own_champions": json.dumps(own),
                "opp_champions": json.dumps(opp),
                "perspective_won": int(won if team_one else not won),
            }
            own_has = owns if team_one else not owns
            for element in ELEMENTS:
                row[f"post_own_count_{element}"] = int(
                    element == "infernal" and own_has
                )
                row[f"post_opp_count_{element}"] = int(
                    element == "infernal" and not own_has
                )
            rows.append(row)
    for index in range(10):
        date = pd.Timestamp("2026-03-02T00:00:00Z") + pd.Timedelta(
            hours=index
        )
        for perspective, own, opp in (
            (
                "team_1",
                ["Zoe", "Orianna", "Sejuani", "Jinx", "Lulu"],
                team_b,
            ),
            (
                "team_2",
                team_b,
                ["Zoe", "Orianna", "Sejuani", "Jinx", "Lulu"],
            ),
        ):
            team_one = perspective == "team_1"
            row = {
                "series_id": f"holdout-series-{index:03d}",
                "game_id": f"holdout-game-{index:03d}",
                "date": date,
                "stage": 1,
                "perspective": perspective,
                "own_team_id": f"holdout-{perspective}-{index % 4}",
                "opp_team_id": f"holdout-opp-{perspective}-{index % 4}",
                "own_champions": json.dumps(own),
                "opp_champions": json.dumps(opp),
                "perspective_won": int(team_one),
            }
            for element in ELEMENTS:
                row[f"post_own_count_{element}"] = int(
                    element == "infernal" and team_one
                )
                row[f"post_opp_count_{element}"] = int(
                    element == "infernal" and not team_one
                )
            rows.append(row)
    return pd.DataFrame(rows)


def test_prepare_rows_are_mirrored_and_complementary() -> None:
    rows = prepare_joint_rows(_games(), _events())
    first = rows[rows["stage"] == 1].sort_values("perspective")

    assert len(rows) == 12
    assert len(first) == 2
    assert first["perspective_won"].sum() == 1
    assert first["took_current"].sum() == 1
    assert first["gold_diff_k"].iloc[0] == -first["gold_diff_k"].iloc[1]
    assert (
        first["post_own_count_infernal"].iloc[0]
        == first["post_opp_count_infernal"].iloc[1]
    )
    assert (
        first["post_opp_count_infernal"].iloc[0]
        == first["post_own_count_infernal"].iloc[1]
    )
    for design_fn in (_design_state, _design_allocation):
        design = design_fn(first)
        assert np.allclose(
            design.iloc[0].to_numpy(),
            -design.iloc[1].to_numpy(),
        )


def test_pre_post_inventory_is_legal_and_soul_is_explicit() -> None:
    rows = prepare_joint_rows(_games(), _events())
    soul = rows[rows["stage"] == 6].sort_values("took_current", ascending=False)

    assert len(soul) == 2
    assert soul.iloc[0]["pre_own_total"] == 3
    assert soul.iloc[0]["post_own_total"] == 4
    assert soul.iloc[0]["own_soul_element_after"] == "hextech"
    assert pd.isna(soul.iloc[0]["opp_soul_element_after"])
    assert soul.iloc[1]["opp_soul_element_after"] == "hextech"
    assert pd.isna(soul.iloc[1]["own_soul_element_after"])
    for element in ELEMENTS:
        assert soul[f"pre_own_count_{element}"].between(0, 4).all()
        assert soul[f"post_own_count_{element}"].between(0, 4).all()


def test_capture_path_enforces_opening_and_rift_rules() -> None:
    games = _games().iloc[0].to_dict()
    team_ids = (games["team_1_id"], games["team_2_id"])
    legal = _events()
    repeated_opening = legal.copy()
    repeated_opening.loc[1, "element"] = repeated_opening.loc[0, "element"]
    repeated_second_at_rift = legal.copy()
    repeated_second_at_rift.loc[2, "element"] = repeated_second_at_rift.loc[
        1, "element"
    ]

    assert _capture_path_is_legal(legal, team_ids)
    assert not _capture_path_is_legal(repeated_opening, team_ids)
    assert not _capture_path_is_legal(repeated_second_at_rift, team_ids)


def test_invalid_side_assignment_is_excluded_from_joint_rows() -> None:
    games = _games()
    assert _side_assignment_is_valid(games.iloc[0].to_dict())
    games.loc[0, "team_2_side"] = "blue"

    assert not _side_assignment_is_valid(games.iloc[0].to_dict())
    assert prepare_joint_rows(games, _events()).empty


def test_both_compositions_change_joint_design() -> None:
    rows = prepare_joint_rows(_games(), _events()).iloc[[0]].copy()
    tag = "scaling_late"
    assert tag in ARCHETYPE_NAMES

    base = _design_state(rows)
    own_changed = rows.copy()
    own_changed[f"own_{tag}"] += 1
    enemy_changed = rows.copy()
    enemy_changed[f"opp_{tag}"] += 1

    own_design = _design_state(own_changed)
    enemy_design = _design_state(enemy_changed)
    assert not np.allclose(base.to_numpy(), own_design.to_numpy())
    assert not np.allclose(base.to_numpy(), enemy_design.to_numpy())
    assert any("own_trait_scaling_late" in column for column in base.columns)
    assert any("enemy_trait_scaling_late" in column for column in base.columns)
    assert any("soul_after" in column for column in base.columns)


def test_allocation_counterfactual_only_changes_treatment_and_derived_state() -> None:
    row = prepare_joint_rows(_games(), _events()).query(
        "stage == 1 and took_current == 1"
    )
    changed = allocation_counterfactual_rows(row, 0)
    changed_columns = {
        column
        for column in row.columns
        if not row[column].reset_index(drop=True).equals(
            changed[column].reset_index(drop=True)
        )
    }
    expected = {
        "took_current",
        "allocation_sign",
        "post_own_total",
        "post_opp_total",
        "own_soul_element_after",
        "opp_soul_element_after",
    }
    expected.update(
        f"post_{side}_count_{element}"
        for side in ("own", "opp")
        for element in ELEMENTS
    )
    assert changed_columns <= expected
    assert {"took_current", "allocation_sign"} <= changed_columns
    assert row.iloc[0]["pre_own_total"] == changed.iloc[0]["pre_own_total"]
    assert row.iloc[0]["gold_diff_k"] == changed.iloc[0]["gold_diff_k"]

    before_design = _design_allocation(row)
    after_design = _design_allocation(changed)
    assert not np.allclose(before_design.to_numpy(), after_design.to_numpy())


def test_equal_total_weight_per_game() -> None:
    rows = pd.DataFrame(
        [
            {"series_id": "s1", "game_id": "short"},
            {"series_id": "s1", "game_id": "short"},
            *[
                {"series_id": "s2", "game_id": "long"}
                for _ in range(10)
            ],
        ]
    )
    weights = _game_weights(rows)

    assert np.isclose(weights[:2].sum(), weights[2:].sum())
    assert np.isclose(weights.mean(), 1.0)


def test_effective_runtime_reproduces_sklearn_with_clipping() -> None:
    design = pd.DataFrame(
        {
            "state": np.linspace(-2.0, 2.0, 200),
            "allocation": [1.0, -1.0] * 100,
            "interaction": np.sin(np.linspace(0.0, 4.0, 200)),
        }
    )
    outcome = np.array([index % 2 for index in range(len(design))])
    fit = _fit(design, outcome, alpha=0.1)
    runtime = _effective_runtime(fit, design)
    query = pd.DataFrame(
        {
            "state": [-1_000_000.0, -0.5, 0.5, 1_000_000.0],
            "allocation": [1.0, -1.0, 1.0, -1.0],
            "interaction": [0.0, 0.1, -0.1, 0.0],
        }
    )

    assert np.allclose(
        runtime_predict(runtime, query),
        fit.predict(query),
        atol=1e-12,
    )
    assert np.allclose(
        runtime_predict(runtime, query),
        expit(runtime_score(runtime, query)),
        atol=1e-12,
    )


def test_standardization_uses_each_actual_draft_perspective_once() -> None:
    rows = prepare_joint_rows(_games(), _events())
    drafts, mirrors = _standardization_draft_rows(rows)

    assert len(drafts) == 2
    assert len(mirrors) == 2
    assert drafts[["series_id", "game_id", "perspective"]].duplicated().sum() == 0
    assert (drafts["perspective"].to_numpy() != mirrors["perspective"].to_numpy()).all()
    assert (drafts["game_id"].to_numpy() == mirrors["game_id"].to_numpy()).all()


def test_standardized_rankings_serialize_stage_metrics_and_support() -> None:
    rows = prepare_joint_rows(_games(), _events())
    rankings = _standardized_element_rankings(
        rows,
        {"intercept": 0.0, "features": []},
    )

    assert len(rankings["rankings"]) == len(ELEMENTS)
    assert {entry["element"] for entry in rankings["rankings"]} == set(ELEMENTS)
    assert all(
        entry["firstCapturePp"] == 0.0
        and entry["secondCapturePp"] == 0.0
        and entry["mapPhaseCapturePp"] == 0.0
        for entry in rankings["rankings"]
    )
    support = rankings["support"]
    assert support["modeledGames"] == 1
    assert support["actualDraftPerspectives"] == 2
    assert support["resolvedCaptures"] == 6
    assert support["soulCaptures"] == 1
    assert support["stageReferencePerspectiveRows"] == {
        str(stage): 2 for stage in range(1, 7)
    }
    assert support["observedFirstCapturesByElement"]["infernal"] == 1
    assert support["observedSecondCapturesByElement"]["ocean"] == 1
    assert support["observedMapPhaseCapturesByElement"]["hextech"] == 4
    assert support["observedSoulCapturesByElement"]["hextech"] == 1
    assert support["legalFirstContextsPerElement"] == 1
    assert support["legalSecondContextsPerElement"] == 10
    assert support["legalOpeningPairsPerMapElement"] == 10
    assert support["openingOwnerAssignmentsPerPair"] == 4
    assert support["legalMapPathsPerElement"] == 40
    assert support["mapCaptureIncrementsPerElement"] == 120
    assert support["mapPathCountsByCaptureLength"] == {
        "2": 10,
        "3": 20,
        "4": 10,
    }
    assert support["championResidualApplied"] is False
    assert support["championResidualFeatureCount"] == 0
    assert "point estimates only" in rankings["pointEstimateCaveat"]
    assert "final increment" in rankings["estimands"]["mapPhaseCapturePp"]


def test_standardized_map_metric_averages_single_captures_and_final_soul() -> None:
    rows = prepare_joint_rows(_games(), _events())

    def feature(name: str, weight: float) -> dict[str, object]:
        return {
            "name": name,
            "weight": weight,
            "clipLow": -20.0,
            "clipHigh": 20.0,
        }

    rankings = _standardized_element_rankings(
        rows,
        {
            "intercept": 0.0,
            "features": [
                feature("post_inventory_diff_infernal", 1.0),
                feature("soul_after_infernal", 1.0),
            ],
        },
    )
    infernal = next(
        entry
        for entry in rankings["rankings"]
        if entry["element"] == "infernal"
    )
    expected_capture = float((expit(1.0) - 0.5) * 100)
    expected_map = float(
        (
            0.25 * (expit(3.0) - 0.5) / 2
            + 0.50 * (expit(4.0) - 0.5) / 3
            + 0.25 * (expit(5.0) - 0.5) / 4
        )
        * 100
    )

    assert infernal["firstCapturePp"] == pytest.approx(
        expected_capture,
        abs=1e-6,
    )
    assert infernal["secondCapturePp"] == pytest.approx(
        expected_capture,
        abs=1e-6,
    )
    assert infernal["mapPhaseCapturePp"] == pytest.approx(
        expected_map,
        abs=1e-6,
    )


def test_cached_standardized_scorer_matches_full_runtime_design() -> None:
    rows = prepare_joint_rows(_games(), _events())
    drafts, mirrors = _standardization_draft_rows(rows)

    def feature(
        name: str,
        weight: float,
        **metadata: object,
    ) -> dict[str, object]:
        return {
            "name": name,
            "weight": weight,
            "clipLow": -20.0,
            "clipHigh": 20.0,
            **metadata,
        }

    runtime = {
        "intercept": 0.17,
        "features": [
            feature("trait_diff_scaling_late", 0.11),
            feature("trait_diff_scaling_late_x_minute", -0.007),
            feature("post_inventory_diff_infernal", 0.4),
            feature("post_infernal_own_trait_scaling_late", 0.13),
            feature("post_infernal_enemy_trait_scaling_late", -0.09),
            feature("soul_after_infernal", 0.31),
            feature(
                "champion_direct_inventory::infernal::Jinx",
                0.23,
                family=DIRECT_FAMILY,
                champion="Jinx",
                element="infernal",
            ),
        ],
    }
    own_inventory = {element: 0 for element in ELEMENTS}
    opp_inventory = {element: 0 for element in ELEMENTS}
    own_inventory["infernal"] = 2
    own_inventory["mountain"] = 1
    own_inventory["ocean"] = 1
    minute = 24.5

    cached = _StandardizedRuntimeScorer(
        runtime,
        drafts,
        mirrors,
    ).probability(
        minute=minute,
        own_inventory=own_inventory,
        opp_inventory=opp_inventory,
        own_soul="infernal",
    )
    full = _reconciled_runtime_probability(
        runtime,
        _standardization_state(
            drafts,
            stage=4,
            minute=minute,
            own_inventory=own_inventory,
            opp_inventory=opp_inventory,
            own_soul="infernal",
        ),
        _standardization_state(
            mirrors,
            stage=4,
            minute=minute,
            own_inventory=opp_inventory,
            opp_inventory=own_inventory,
            opp_soul="infernal",
        ),
    )

    assert np.allclose(cached, full, atol=1e-12)


def test_provenance_hashes_exact_files_and_identifier_sets(tmp_path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"exact model input")

    file_provenance = _file_provenance(source)
    set_provenance = _identifier_set_provenance(["series-b", "series-a", "series-a"])

    assert file_provenance["bytes"] == len(b"exact model input")
    assert len(file_provenance["sha256"]) == 64
    assert set(file_provenance["sha256"]) <= set("0123456789abcdef")
    assert set_provenance["count"] == 2
    assert len(set_provenance["sha256"]) == 64
    assert "series-a" not in str(set_provenance)


def test_evaluation_vocabulary_is_frozen_from_inner_train_only() -> None:
    partitions = _temporal_partitions(_support_partition_rows())
    support = _champion_element_support(partitions.inner_train)
    champions = {str(entry["champion"]) for entry in support}
    spec = _freeze_champion_residual_spec(support, min_games=50)

    assert "Aatrox" in champions
    assert "Zoe" not in champions
    assert "Zoe" not in {
        champion
        for element_spec in spec.elements.values()
        for champion in element_spec.champions
    }
    assert all(
        int(entry["trainingSeries"]) >= 25
        for element_spec in spec.elements.values()
        for entry in element_spec.support
    )


def test_champion_cell_eligibility_is_exposure_only() -> None:
    support = _eligible_support("Aatrox")
    support.update(
        {
            "wins": 0,
            "losses": 80,
            "ownershipWins": 0,
            "ownershipLosses": 40,
            "nonOwnershipWins": 0,
            "nonOwnershipLosses": 40,
        }
    )

    assert _cell_is_direct_eligible(support, min_games=75)
    assert not _direct_cell_failed_exposure_rules(
        support,
        min_games=75,
    )

    support["ownershipGames"] = 19
    assert not _cell_is_direct_eligible(support, min_games=75)
    assert _direct_cell_failed_exposure_rules(
        support,
        min_games=75,
    ) == ["minimum-ownership-games"]


def test_evaluation_audit_and_publication_vocabularies_are_distinct() -> None:
    evaluation = _freeze_champion_residual_spec(
        [
            _eligible_support("Aatrox"),
            _eligible_support("Urgot"),
        ],
        min_games=75,
    )
    audit = _freeze_champion_residual_spec(
        [
            _eligible_support("Aatrox"),
            _eligible_support("Urgot"),
            _eligible_support("Zoe"),
        ],
        min_games=75,
    )
    publication = _freeze_champion_residual_spec(
        [
            _eligible_support("Aatrox"),
            _eligible_support("Urgot"),
            _eligible_support("Zoe"),
            _eligible_support("Jinx"),
        ],
        min_games=75,
    )

    assert _champion_residual_vocabulary_summary(evaluation)["cells"] == 2
    assert _champion_residual_vocabulary_summary(audit)["cells"] == 3
    assert _champion_residual_vocabulary_summary(publication)["cells"] == 4


def test_publication_audit_requires_whole_series() -> None:
    rows = pd.DataFrame(
        [
            {
                "series_id": "train-series",
                "game_id": "train-game",
                "date": pd.Timestamp("2026-06-30T00:00:00Z"),
                "stage": 1,
                "perspective": "team_1",
            },
            {
                "series_id": "audit-series",
                "game_id": "audit-game",
                "date": pd.Timestamp("2026-07-02T00:00:00Z"),
                "stage": 1,
                "perspective": "team_1",
            },
        ]
    )
    partitions = _publication_audit_partitions(
        rows,
        minimum_games=1,
        minimum_series=1,
    )

    assert set(partitions.train["series_id"]) == {"train-series"}
    assert set(partitions.holdout["series_id"]) == {"audit-series"}

    crossing = rows.copy()
    crossing.loc[1, "series_id"] = "train-series"
    with pytest.raises(
        ValueError,
        match="crosses a publication-audit boundary",
    ):
        _publication_audit_partitions(
            crossing,
            minimum_games=1,
            minimum_series=1,
        )


def test_direct_features_are_antisymmetric_and_effect_coded() -> None:
    support = [
        _eligible_support("Aatrox"),
        _eligible_support("Urgot"),
    ]
    spec = _freeze_champion_residual_spec(support, min_games=50)
    rows = _direct_fit_rows(games=1)
    design = _direct_raw_design(rows, spec)
    element_spec = spec.elements["infernal"]

    assert np.allclose(
        design.iloc[0].to_numpy(),
        -design.iloc[1].to_numpy(),
    )
    assert np.allclose(
        element_spec.constraint @ element_spec.basis,
        0.0,
        atol=1e-10,
    )


def test_same_tag_supported_champions_can_have_different_residuals() -> None:
    assert champ_tags("Aatrox") == champ_tags("Urgot")
    spec = _freeze_champion_residual_spec(
        [_eligible_support("Aatrox"), _eligible_support("Urgot")],
        min_games=50,
    )
    rows = _direct_fit_rows()
    outcome = rows["perspective_won"].to_numpy(dtype=int)
    fit = _fit_offset_ridge(
        rows,
        spec,
        base_score=np.zeros(len(rows), dtype=float),
        outcome=outcome,
        sample_weight=np.ones(len(rows), dtype=float),
        selected_lambda=0.03,
    )
    aatrox = fit.raw_coefficients[
        "champion_direct_inventory::infernal::Aatrox"
    ]
    urgot = fit.raw_coefficients[
        "champion_direct_inventory::infernal::Urgot"
    ]

    assert aatrox > 0
    assert urgot < 0
    assert not np.isclose(aatrox, urgot)
    assert fit.constraint_max_abs["infernal"] < 1e-8


def test_direct_fit_fails_closed_on_constraint_violation() -> None:
    spec = _freeze_champion_residual_spec(
        [_eligible_support("Aatrox"), _eligible_support("Urgot")],
        min_games=50,
    )
    spec.elements["infernal"].constraint = np.array([[1.0, 0.0]])
    rows = _direct_fit_rows()

    with pytest.raises(ValueError, match="constraint violation"):
        _fit_offset_ridge(
            rows,
            spec,
            base_score=np.zeros(len(rows), dtype=float),
            outcome=rows["perspective_won"].to_numpy(dtype=int),
            sample_weight=np.ones(len(rows), dtype=float),
            selected_lambda=0.03,
        )


def test_direct_family_gate_fails_closed_on_material_regression() -> None:
    status, reason = _direct_family_gate(
        selected_gate=1.0,
        inner_base={"brier": 0.24, "logLoss": 0.68, "ece10": 0.05},
        inner_augmented={"brier": 0.23, "logLoss": 0.66, "ece10": 0.04},
        holdout_base={"brier": 0.20, "logLoss": 0.60, "ece10": 0.04},
        holdout_augmented={"brier": 0.21, "logLoss": 0.62, "ece10": 0.08},
        delta_brier_interval={"lower": 0.005, "upper": 0.015},
    )

    assert status == "withheld"
    assert "materially worse" in reason


def test_publication_audit_must_clear_base_and_evaluation_vocabulary() -> None:
    status, reason = _publication_expansion_audit_gate(
        selected_gate=1.0,
        base={"brier": 0.24, "logLoss": 0.68, "ece10": 0.05},
        evaluation_vocabulary={
            "brier": 0.20,
            "logLoss": 0.60,
            "ece10": 0.03,
        },
        expanded={"brier": 0.205, "logLoss": 0.60, "ece10": 0.03},
        versus_base_interval={"lower": -0.04, "upper": -0.02},
        versus_evaluation_interval={"lower": 0.003, "upper": 0.007},
    )

    assert status == "withheld"
    assert "original evaluation vocabulary" in reason


def test_augmented_runtime_matches_base_offset_plus_direct_residual() -> None:
    spec = _freeze_champion_residual_spec(
        [_eligible_support("Aatrox"), _eligible_support("Urgot")],
        min_games=50,
    )
    rows = _direct_fit_rows()
    outcome = rows["perspective_won"].to_numpy(dtype=int)
    direct_fit = _fit_offset_ridge(
        rows,
        spec,
        base_score=np.zeros(len(rows), dtype=float),
        outcome=outcome,
        sample_weight=np.ones(len(rows), dtype=float),
        selected_lambda=0.03,
    )
    base_design = pd.DataFrame(
        {
            "state": np.linspace(-1.0, 1.0, len(rows)),
            "side": [1.0, -1.0] * (len(rows) // 2),
        },
        index=rows.index,
    )
    base_fit = _fit(base_design, outcome, alpha=0.1)
    gate = 0.5
    cells = []
    for element_spec in spec.elements.values():
        for cell in element_spec.support:
            feature_name = str(cell["featureName"])
            coefficient = direct_fit.raw_coefficients[feature_name]
            cells.append(
                {
                    **cell,
                    "coefficient": coefficient,
                    "gatedCoefficient": gate * coefficient,
                }
            )
    result = {
        "status": "ready",
        "familyGate": gate,
        "eligibleCells": cells,
    }
    base_runtime = _effective_runtime(base_fit, base_design)
    runtime = _augment_joint_runtime(base_runtime, result, rows)
    runtime_design = pd.concat(
        [base_design, _direct_cell_design(rows, cells)],
        axis=1,
    )
    expected = expit(
        _fitted_logit_score(base_fit, base_design)
        + gate * direct_fit.linear_score(rows, spec)
    )

    assert np.allclose(
        runtime_predict(runtime, runtime_design),
        expected,
        atol=1e-12,
    )
    assert all(
        feature.get("family") == DIRECT_FAMILY
        for feature in runtime["features"]
        if feature["name"].startswith("champion_direct_inventory::")
    )
    assert _augment_joint_runtime(
        base_runtime,
        {"status": "withheld", "familyGate": 0.0},
        rows,
    ) == base_runtime


def test_direct_runtime_excludes_soul_stage_pair_enemy_and_draft_terms() -> None:
    spec = _freeze_champion_residual_spec(
        [_eligible_support("Aatrox"), _eligible_support("Urgot")],
        min_games=50,
    )
    columns = [
        column.casefold()
        for column in _direct_raw_design(_direct_fit_rows(games=2), spec).columns
    ]
    disabled = _feature_schema(
        prepare_joint_rows(_games(), _events())
    )["disabledChampionFamilies"]

    assert columns
    assert not any("soul" in column for column in columns)
    assert not any("stage" in column for column in columns)
    assert not any("ally" in column for column in columns)
    assert not any("enemy" in column for column in columns)
    assert not any("draft" in column for column in columns)
    assert set(disabled) == {
        "championSoul",
        "stageSpecificChampion",
        "allyPair",
        "enemyIdentity",
        "genericDraftScore",
    }
    assert all(entry["status"] == "disabled" for entry in disabled.values())


def test_advertised_state_controls_match_available_runtime_inputs() -> None:
    assert "kill_diff" not in STATE_NUMERIC
    assert "tower_diff" in STATE_NUMERIC


def test_public_wording_rejects_exact_champion_and_leave_claims() -> None:
    wording = " ".join(PUBLIC_WORDING.values()).casefold()

    assert "reconciled allocations" in wording
    assert "archetype prior only" in wording
    assert "partially pooled champion estimate" in wording
    assert "common dragon effect remains team-level" in wording
    assert "not champion win rates" in wording
    assert "not a second archetype bonus" in wording
    assert "not applied to the dragon estimate" in wording
    assert "different pre-match composition estimand" in wording
    assert "not a strategic leave policy" in wording
    assert "not an exact-composition lookup" in wording
    assert "reviewed" not in wording
    assert "direct residual" not in wording
    assert "archetype fallback" not in wording


def test_champion_catalog_labels_allocation_source_without_claiming_evidence() -> None:
    catalog = _champion_catalog({"championGameAppearances": {}})
    tagged = next(entry for entry in catalog if entry["tags"])
    untagged = next(entry for entry in catalog if not entry["tags"])

    assert tagged["allocationKind"] == "reconciled-allocation"
    assert tagged["allocationSource"] == "archetype-fallback"
    assert tagged["fallback"] is None
    assert untagged["allocationKind"] == "reconciled-allocation"
    assert untagged["allocationSource"] == "team-common-only"
    assert untagged["fallback"] == "team-common-only"
    assert set(tagged["elementEvidence"]) == set(ELEMENTS)
    assert all(
        entry["source"] == "team-common-only"
        for entry in untagged["elementEvidence"].values()
    )


def test_champion_catalog_exposes_publication_provenance_and_raw_fallback_support() -> None:
    published = {
        **_eligible_support("Aatrox"),
        "championEligible": True,
        "failedExposureRules": [],
        "vocabularyProvenance": "publication-audit-vocabulary",
        "individualCellValidated": False,
        "coefficient": 0.1,
        "gatedCoefficient": 0.1,
    }
    low_support = _eligible_support("Zoe", games=30)
    observed = _serialize_observed_champion_cells(
        [published, low_support],
        min_games=75,
    )
    catalog = _champion_catalog(
        {"championGameAppearances": {"Aatrox": 80, "Zoe": 30}},
        {
            "status": "ready",
            "familyGate": 1.0,
            "eligibleCells": [published],
            "observedCells": observed,
        },
    )
    aatrox = next(entry for entry in catalog if entry["name"] == "Aatrox")
    zoe = next(entry for entry in catalog if entry["name"] == "Zoe")
    ready = aatrox["elementEvidence"]["infernal"]
    fallback = zoe["elementEvidence"]["infernal"]

    assert ready["championEligible"] is True
    assert ready["vocabularyProvenance"] == "publication-audit-vocabulary"
    assert ready["individualCellValidated"] is False
    assert ready["outcomeCountsUsedForEligibility"] is False
    assert fallback["championEligible"] is False
    assert fallback["trainingGames"] == 30
    assert fallback["status"] == "below-threshold"
    assert fallback["failedExposureRules"] == [
        "minimum-games",
        "minimum-ownership-games",
        "minimum-nonownership-games",
    ]


def test_feature_schema_fail_closes_generic_draft_score_bridge() -> None:
    rows = prepare_joint_rows(_games(), _events())
    schema = _feature_schema(rows)
    allocation = schema["championAllocation"]
    draft_context = schema["genericDraftScoreContext"]

    assert allocation["kind"] == "reconciled-allocation"
    assert allocation["taggedSource"] == "archetype-fallback"
    assert allocation["untaggedSource"] == "team-common-only"
    assert allocation["championSpecificEmpiricalEvidence"] is False
    assert allocation["separatelyFittedChampionEffects"] is False
    assert allocation["directResidualFamily"]["status"] == "unavailable"
    assert allocation["directResidualFamily"]["familyGate"] == 0.0
    assert draft_context["appliedToDragonEstimate"] is False
    assert draft_context["championEffectsApplied"] is False
    assert draft_context["allySynergyApplied"] is False
    assert draft_context["enemyCounterApplied"] is False

    for design_fn in (_design_state, _design_allocation):
        columns = [column.casefold() for column in design_fn(rows).columns]
        assert not any("draft_score" in column for column in columns)
        assert not any("champion_identity" in column for column in columns)


def test_explorer_artifact_is_gated_below_threshold(tmp_path) -> None:
    games_path = tmp_path / "games.parquet"
    events_path = tmp_path / "events.parquet"
    _games().to_parquet(games_path)
    _events().to_parquet(events_path)

    artifact = build_explorer_artifact(
        games_path=games_path,
        events_path=events_path,
        min_games=6_000,
    )

    assert artifact["status"] == "gated"
    assert artifact["games"] == 1
    assert artifact["requiredGames"] == 6_000
    assert len(artifact["provenance"]["inputs"]["games"]["sha256"]) == 64
    assert artifact["provenance"]["inputs"]["games"]["bytes"] > 0
    assert len(
        artifact["provenance"]["inputs"]["championTaxonomy"]["sha256"]
    ) == 64
    assert "publicWording" in artifact
