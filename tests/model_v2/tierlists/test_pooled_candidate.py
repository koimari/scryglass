"""Tests for the public coach-facing pooled tier fields."""

from __future__ import annotations

import numpy as np

from lol_kills.v2.tierlists.pooled_candidate import (
    _blind_point_estimate,
    _build_regional_views,
    _counter_count_point_estimate,
    _matchup_metrics_available,
    _regional_contexts,
    _response_basis,
    _scope_atom_patch,
)


def test_regional_contexts_use_exact_league_identity() -> None:
    assert _regional_contexts({"league": "LCK"}) == ("LCK",)
    assert _regional_contexts({"league": "MSI", "event_kind": "msi"}) == ("INTERNATIONAL",)
    assert _regional_contexts({"league": "LCKC"}) == ()


def test_regional_view_keeps_patch_wide_strength_order() -> None:
    rows = [
        {
            "champion": "Lower global pick",
            "champion_id": "riot:champion:2",
            "rank": 2,
            "tier_value_pp": 4.0,
        },
        {
            "champion": "Higher global pick",
            "champion_id": "riot:champion:1",
            "rank": 1,
            "tier_value_pp": 10.0,
        },
    ]
    views = _build_regional_views(
        rows=rows,
        scope_id="patch:16.14",
        role="mid",
        regional_counts={
            ("patch:16.14", "LCK", "mid"): {
                "riot:champion:2": 4,
                "riot:champion:1": 1,
            }
        },
        regional_game_ids={("patch:16.14", "LCK"): {"game-1", "game-2"}},
    )

    assert len(views) == 1
    assert views[0]["id"] == "LCK"
    assert views[0]["maps"] == 2
    assert [row["champion_id"] for row in views[0]["rows"]] == [
        "riot:champion:1",
        "riot:champion:2",
    ]


def test_complete_oe_matchup_support_does_not_require_an_atom_snapshot() -> None:
    assert _matchup_metrics_available(
        opponent_count=5,
        supported_opponent_count=5,
        contrast_sd=0.4,
    )


def test_oe_matchup_support_remains_fail_closed_for_thin_or_uncertain_rows() -> None:
    assert not _matchup_metrics_available(
        opponent_count=5,
        supported_opponent_count=4,
        contrast_sd=0.4,
    )
    assert not _matchup_metrics_available(
        opponent_count=5,
        supported_opponent_count=5,
        contrast_sd=1.2,
    )


def test_blind_point_estimate_uses_expected_weakest_matchup() -> None:
    probabilities = np.asarray(
        [
            [0.56, 0.54, 0.55],
            [0.61, 0.59, 0.60],
            [0.67, 0.65, 0.66],
            [0.58, 0.56, 0.57],
            [0.63, 0.61, 0.62],
        ]
    )
    score = _blind_point_estimate(probabilities, np.full(5, 0.2))
    assert score == 0.55


def test_counter_count_uses_positive_model_contrasts() -> None:
    theta = np.asarray(
        [
            [0.10, 0.08, 0.09],
            [0.06, 0.05, 0.07],
            [0.01, 0.02, 0.03],
            [-0.02, -0.01, 0.00],
            [0.20, 0.18, 0.19],
        ]
    )
    assert _counter_count_point_estimate(theta) == 3


def test_response_basis_names_observed_atom_and_strength_only_estimates() -> None:
    assert _response_basis(effective_maps=4.5, atom_supported=True) == "observed_pair_plus_model"
    assert _response_basis(effective_maps=0.0, atom_supported=True) == "atom_and_strength_inferred"
    assert _response_basis(effective_maps=0.0, atom_supported=False) == "strength_only_inferred"


def test_scope_atom_patch_uses_the_audited_snapshot_instead_of_the_oe_token() -> None:
    games = [
        {"oe_patch_id": "16.15", "atom_snapshot_patch": "26.15"},
        {"oe_patch_id": "16.15", "atom_snapshot_patch": "26.15"},
    ]
    assert _scope_atom_patch(games) == "26.15"
    assert _scope_atom_patch([*games, {"atom_snapshot_patch": "26.16"}]) is None
