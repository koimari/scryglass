from __future__ import annotations

import pandas as pd
import pytest

from lol_kills.research.future_value_draft_score import (
    AUTHORITY,
    CURVE_ATOM_FAMILIES,
    CURVE_ATOM_INTERACTION_FEATURES,
    DraftScoreProducerBinding,
    DraftScoreVariant,
    FutureValueDraftScoreError,
    PHASE_RAW_FEATURES,
    STATIC_COMPOSITION_FEATURES,
    VARIANT_CONFIGS,
    assert_static_composition_parity,
    build_curve_atom_interactions,
    build_draft_score_variant_design,
    score_draft_score_variant,
    swap_variant_feature_frame,
    validate_feature_names,
    validate_side_swap,
    variant_registry_receipt,
)
from lol_kills.research.future_value_rating import (
    CURRENT_RATING_SIGNED_MAP_FEATURES,
    FUTURE_PLAYER_FORM_SIDE_FEATURES,
)


def _frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, game_id in enumerate(("g1", "g2")):
        row: dict[str, object] = {
            "game_id": game_id,
            "date": f"2026-01-0{index + 2}T00:00:00Z",
        }
        for component_index, feature in enumerate(STATIC_COMPOSITION_FEATURES, start=1):
            row[feature] = float(component_index + index)
        for feature_index, feature in enumerate(CURRENT_RATING_SIGNED_MAP_FEATURES, start=1):
            row[feature] = float(feature_index) * (index + 1)
        for feature_index, feature in enumerate(FUTURE_PLAYER_FORM_SIDE_FEATURES, start=1):
            row[feature] = float(feature_index) / 10.0 + index
        for feature_index, feature in enumerate(PHASE_RAW_FEATURES, start=1):
            row[feature] = float(feature_index * 100 + index * 10)
        for family_index, family in enumerate(CURVE_ATOM_FAMILIES, start=1):
            for role_index, role in enumerate(("top", "jungle", "mid", "bot", "support"), start=1):
                row[f"curve_blue_{family}_{role}"] = float(family_index + role_index + index)
                row[f"curve_red_{family}_{role}"] = float(family_index + role_index + 1 + index)
                row[f"atom_blue_{family}_{role}"] = float(2 * family_index + role_index)
                row[f"atom_red_{family}_{role}"] = float(2 * family_index + role_index + 1)
        rows.append(row)
    return pd.DataFrame(rows)


def _binding() -> DraftScoreProducerBinding:
    return DraftScoreProducerBinding.create(
        source_receipt_sha256="a" * 64,
        accepted_game_ids=("g1", "g2", "fit1"),
        fit_game_ids=("fit1",),
        fit_window_end="2026-01-01T00:00:00Z",
        fold_id="fold-0",
        producer="synthetic-draft-score-ledger",
    )


def test_four_variants_share_static_atoms_and_have_exact_family_selection() -> None:
    assert tuple(VARIANT_CONFIGS) == tuple(DraftScoreVariant)
    assert all(
        tuple(config.static_features) == STATIC_COMPOSITION_FEATURES
        for config in VARIANT_CONFIGS.values()
    )
    assert VARIANT_CONFIGS[DraftScoreVariant.CURRENT_ONLY].feature_names == (
        *STATIC_COMPOSITION_FEATURES,
        *CURRENT_RATING_SIGNED_MAP_FEATURES,
    )
    assert VARIANT_CONFIGS[DraftScoreVariant.FUTURE_PLAYER_FORM].future_player_form_features == (
        *FUTURE_PLAYER_FORM_SIDE_FEATURES,
    )
    assert VARIANT_CONFIGS[DraftScoreVariant.SCALING_CURVE].phase_raw_features == PHASE_RAW_FEATURES
    assert VARIANT_CONFIGS[DraftScoreVariant.BOTH].curve_interaction_features == (
        *CURVE_ATOM_INTERACTION_FEATURES,
    )
    assert variant_registry_receipt()["authority"] is False


def test_all_variants_bind_one_census_and_preserve_static_composition() -> None:
    frame = _frame()
    binding = _binding()
    designs = {
        variant: build_draft_score_variant_design(frame, variant, binding)
        for variant in DraftScoreVariant
    }
    static_hash = assert_static_composition_parity(designs)
    assert all(design.static_composition_sha256 == static_hash for design in designs.values())
    assert all(design.authority is False for design in designs.values())


def test_curve_atom_interaction_is_difference_of_products() -> None:
    frame = _frame().iloc[[0]].copy()
    values = build_curve_atom_interactions(frame)
    expected = (
        frame["curve_blue_role_top"].iloc[0] * frame["atom_blue_role_top"].iloc[0]
        - frame["curve_red_role_top"].iloc[0] * frame["atom_red_role_top"].iloc[0]
    )
    assert values["curve_atom_role_top"].iloc[0] == expected
    assert len(values.columns) == len(CURVE_ATOM_INTERACTION_FEATURES)


def test_component_logits_reconstruct_and_static_components_are_visible() -> None:
    frame = _frame()
    binding = _binding()
    design = build_draft_score_variant_design(frame, DraftScoreVariant.BOTH, binding)
    score = score_draft_score_variant(design)
    assert score.component_reconstruction_error_max <= 1e-12
    assert score.composite_logit.tolist() == score.components["composite_logit"].tolist()
    assert "composition_base_logit" in score.components
    assert "current_rating_logit" in score.components
    assert "future_player_form_logit" in score.components
    assert "scaling_raw_logit" in score.components
    assert "scaling_shape_logit" in score.components
    assert "curve_atom_interaction_logit" in score.components
    assert score.receipt()["authority"] is False


@pytest.mark.parametrize("variant", tuple(DraftScoreVariant))
def test_side_swap_negates_signed_features_and_preserves_phase_invariants(
    variant: DraftScoreVariant,
) -> None:
    design = build_draft_score_variant_design(_frame(), variant, _binding())
    swapped = swap_variant_feature_frame(design.feature_frame, variant)
    result = validate_side_swap(design.feature_frame, swapped, variant)
    assert result["passed"] is True


def test_target_final_and_current_checkpoint_features_fail_closed() -> None:
    for name in ("target", "damageshare", "goldat10", "gold_diff_15", "observed_result"):
        with pytest.raises(FutureValueDraftScoreError):
            validate_feature_names([name])


def test_fold_binding_rejects_overlap_and_same_timestamp() -> None:
    frame = _frame()
    binding = _binding()
    overlapping = frame.copy()
    overlapping.loc[0, "game_id"] = "fit1"
    with pytest.raises(FutureValueDraftScoreError, match="overlap"):
        build_draft_score_variant_design(overlapping, DraftScoreVariant.CURRENT_ONLY, binding)

    same_time = frame.copy()
    same_time.loc[0, "date"] = "2026-01-01T00:00:00Z"
    with pytest.raises(FutureValueDraftScoreError, match="strictly prior"):
        build_draft_score_variant_design(same_time, DraftScoreVariant.CURRENT_ONLY, binding)


def test_static_parity_rejects_mutated_atom_value() -> None:
    frame = _frame()
    binding = _binding()
    first = build_draft_score_variant_design(frame, DraftScoreVariant.CURRENT_ONLY, binding)
    changed = frame.copy()
    changed.loc[0, "composition_counter_logit"] += 0.01
    second = build_draft_score_variant_design(changed, DraftScoreVariant.CURRENT_ONLY, binding)
    with pytest.raises(FutureValueDraftScoreError, match="static composition"):
        assert_static_composition_parity([first, second])
