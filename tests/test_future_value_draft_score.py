from __future__ import annotations

import hashlib
import json

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
    PHASE_SHAPE_INVARIANT_FEATURES,
    STATIC_COMPOSITION_FEATURES,
    VARIANT_CONFIGS,
    assert_static_composition_parity,
    build_curve_atom_interactions,
    build_draft_score_variant_design,
    make_coefficient_receipt,
    score_draft_score_variant,
    static_composition_parity_hash,
    swap_variant_feature_frame,
    swap_raw_blue_red_frame,
    validate_feature_names,
    validate_side_swap,
    validate_raw_side_swap,
    variant_registry_receipt,
)
from lol_kills.research.future_value_rating import (
    CURRENT_RATING_SIGNED_MAP_FEATURES,
    FUTURE_PLAYER_FORM_SIDE_FEATURES,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


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
        row["forecast_curve_available"] = 1.0
        row["forecast_curve_missing"] = 0.0
        for family_index, family in enumerate(CURVE_ATOM_FAMILIES, start=1):
            for role_index, role in enumerate(("top", "jungle", "mid", "bot", "support"), start=1):
                row[f"curve_blue_{family}_{role}"] = float(family_index + role_index + index)
                row[f"curve_red_{family}_{role}"] = float(family_index + role_index + 1 + index)
                row[f"atom_blue_{family}_{role}"] = float(2 * family_index + role_index)
                row[f"atom_red_{family}_{role}"] = float(2 * family_index + role_index + 1)
        rows.append(row)
    return pd.DataFrame(rows)


def _binding() -> DraftScoreProducerBinding:
    accepted = ("fit1", "g1", "g2")
    source = {
        "schema_version": "scryglass:future-value-rating-source:v1",
        "source_as_of": "2026-01-03T00:00:00Z",
        "source_game_count": len(accepted),
        "source_identity_sha256": identity_sha256(accepted),
        "accepted_game_ids": list(accepted),
        "authority": {"research_only": True, "deployment": False, "merge": False, "promotion": False},
    }
    source["receipt_sha256"] = _sha(source)
    return DraftScoreProducerBinding.create(
        source_receipt=source,
        accepted_game_ids=accepted,
        fit_game_ids=("fit1",),
        fit_window_end="2026-01-01T00:00:00Z",
        fit_window_start="2025-12-31T00:00:00Z",
        fit_game_dates={"fit1": "2025-12-31T00:00:00Z"},
        fold_id="fold-0",
        producer="synthetic-draft-score-ledger",
        producer_family="draft_score_features",
        series_safe_evidence={
            "series_safe": True,
            "fit_validation_disjoint": True,
            "source_type": "synthetic_verified",
            "series_column": "series_id",
            "cluster_identity_sha256": "b" * 64,
        },
    )


def _atom_receipt(frame: pd.DataFrame) -> dict[str, object]:
    component_hash = static_composition_parity_hash(frame)
    payload: dict[str, object] = {
        "schema_version": "scryglass:atomized-composition-producer:v1",
        "producer_name": "synthetic-atomized-producer",
        "producer_family": "static_composition",
        "artifact_locator": "research/atomized.parquet",
        "artifact_sha256": "c" * 64,
        "feature_names": list(STATIC_COMPOSITION_FEATURES),
        "component_values_sha256": component_hash,
    }
    payload["receipt_sha256"] = _sha(payload)
    return payload


def _coefficients_and_ledger(design: object) -> tuple[dict[str, float], dict[str, object], pd.DataFrame]:
    coefficients = {feature: 1.0 for feature in design.feature_frame.columns}
    coefficient_receipt = make_coefficient_receipt(design, coefficients)
    ledger = pd.DataFrame(
        {
            "game_id": design.game_ids,
            "model_logit": design.feature_frame.sum(axis=1).to_numpy(),
        }
    )
    return coefficients, coefficient_receipt, ledger


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
        variant: build_draft_score_variant_design(frame, variant, binding, static_atom_receipt=_atom_receipt(frame))
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
    design = build_draft_score_variant_design(frame, DraftScoreVariant.BOTH, binding, static_atom_receipt=_atom_receipt(frame))
    coefficients = {feature: 1.0 for feature in design.feature_frame.columns}
    coefficient_receipt = make_coefficient_receipt(design, coefficients)
    ledger = pd.DataFrame({"game_id": design.game_ids, "model_logit": design.feature_frame.sum(axis=1)})
    score = score_draft_score_variant(
        design,
        coefficients,
        coefficient_receipt=coefficient_receipt,
        independent_prediction_ledger=ledger,
    )
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
    design = build_draft_score_variant_design(_frame(), variant, _binding(), static_atom_receipt=_atom_receipt(_frame()))
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
        build_draft_score_variant_design(overlapping, DraftScoreVariant.CURRENT_ONLY, binding, static_atom_receipt=_atom_receipt(overlapping))

    same_time = frame.copy()
    same_time.loc[0, "date"] = "2026-01-01T00:00:00Z"
    with pytest.raises(FutureValueDraftScoreError, match="strictly prior"):
        build_draft_score_variant_design(same_time, DraftScoreVariant.CURRENT_ONLY, binding, static_atom_receipt=_atom_receipt(same_time))


def test_static_parity_rejects_mutated_atom_value() -> None:
    frame = _frame()
    binding = _binding()
    first = build_draft_score_variant_design(frame, DraftScoreVariant.CURRENT_ONLY, binding, static_atom_receipt=_atom_receipt(frame))
    changed = frame.copy()
    changed.loc[0, "composition_counter_logit"] += 0.01
    with pytest.raises(FutureValueDraftScoreError, match="atomized composition"):
        build_draft_score_variant_design(changed, DraftScoreVariant.CURRENT_ONLY, binding, static_atom_receipt=_atom_receipt(frame))


def test_phase_invariants_are_diagnostics_and_never_score_features() -> None:
    config = VARIANT_CONFIGS[DraftScoreVariant.SCALING_CURVE]
    assert not set(PHASE_SHAPE_INVARIANT_FEATURES).intersection(config.feature_names)
    design = build_draft_score_variant_design(
        _frame(),
        DraftScoreVariant.SCALING_CURVE,
        _binding(),
        static_atom_receipt=_atom_receipt(_frame()),
    )
    assert design.phase_diagnostics is not None
    assert set(PHASE_SHAPE_INVARIANT_FEATURES).issubset(design.phase_diagnostics.columns)


def test_source_receipt_mutation_and_forged_hash_fail_closed() -> None:
    source = dict(_binding().source_receipt or {})
    source["source_as_of"] = "2027-01-01T00:00:00Z"
    with pytest.raises(FutureValueDraftScoreError, match="source receipt hash"):
        DraftScoreProducerBinding.create(
            source_receipt=source,
            accepted_game_ids=("fit1", "g1", "g2"),
            fit_game_ids=("fit1",),
            fit_window_end="2026-01-01T00:00:00Z",
            fit_game_dates={"fit1": "2025-12-31T00:00:00Z"},
            fold_id="fold-0",
            producer="synthetic",
            series_safe_evidence={
                "series_safe": True,
                "fit_validation_disjoint": True,
                "source_type": "synthetic",
                "series_column": "series_id",
                "cluster_identity_sha256": "b" * 64,
            },
        )


def test_static_aliases_and_missing_atom_receipt_fail_closed() -> None:
    frame = _frame().rename(columns={"composition_base_logit": "composition_base"})
    with pytest.raises(FutureValueDraftScoreError, match="composition"):
        build_draft_score_variant_design(frame, DraftScoreVariant.CURRENT_ONLY, _binding())


def test_coefficient_and_independent_prediction_receipts_are_required() -> None:
    frame = _frame()
    design = build_draft_score_variant_design(
        frame,
        DraftScoreVariant.CURRENT_ONLY,
        _binding(),
        static_atom_receipt=_atom_receipt(frame),
    )
    coefficients, receipt, ledger = _coefficients_and_ledger(design)
    with pytest.raises(FutureValueDraftScoreError, match="coefficient"):
        score_draft_score_variant(design)
    changed_ledger = ledger.copy()
    changed_ledger.loc[0, "model_logit"] += 0.1
    with pytest.raises(FutureValueDraftScoreError, match="reconstruct"):
        score_draft_score_variant(
            design,
            coefficients,
            coefficient_receipt=receipt,
            independent_prediction_ledger=changed_ledger,
        )


def test_phase_flags_and_observed_timing_fail_closed() -> None:
    frame = _frame()
    frame.loc[0, "forecast_curve_missing"] = 1.0
    with pytest.raises(FutureValueDraftScoreError, match="availability"):
        build_draft_score_variant_design(
            frame,
            DraftScoreVariant.SCALING_CURVE,
            _binding(),
            static_atom_receipt=_atom_receipt(frame),
        )
    observed = _frame()
    observed["forecast_producer_timing"] = "observed"
    with pytest.raises(FutureValueDraftScoreError, match="timing"):
        build_draft_score_variant_design(
            observed,
            DraftScoreVariant.SCALING_CURVE,
            _binding(),
            static_atom_receipt=_atom_receipt(observed),
        )


def test_raw_blue_red_swap_rebuilds_curve_atom_interactions() -> None:
    frame = _frame()
    swapped = swap_raw_blue_red_frame(frame)
    original = build_curve_atom_interactions(frame)
    rebuilt = build_curve_atom_interactions(swapped)
    pd.testing.assert_frame_equal(rebuilt, -original)


def test_raw_blue_red_swap_rebuilds_the_full_variant() -> None:
    frame = _frame()
    result = validate_raw_side_swap(
        frame,
        DraftScoreVariant.BOTH,
        _binding(),
        static_atom_receipt=_atom_receipt(frame),
    )
    assert result["passed"] is True
