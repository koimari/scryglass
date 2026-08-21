from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

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
    write_independent_prediction_ledger,
    validate_side_swap,
    validate_raw_side_swap,
    variant_registry_receipt,
)
from lol_kills.research.future_value_rating import (
    CURRENT_RATING_SIGNED_MAP_FEATURES,
    FUTURE_PLAYER_FORM_SIDE_FEATURES,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


_FIXTURE_ROOT = Path(tempfile.mkdtemp(prefix="draft-score-receipts-"))
_SOURCE_FILE = _FIXTURE_ROOT / "accepted-census.json"
_SOURCE_PATH = _FIXTURE_ROOT / "source-receipt.json"
_PRODUCER_PATH = _FIXTURE_ROOT / "producer-receipt.json"
_PRODUCER_ARTIFACT_PATH = _FIXTURE_ROOT / "draft_records.json"
_PRODUCER_ARTIFACT_RECEIPT_PATH = _FIXTURE_ROOT / "draft_records-artifact-receipt.json"
_ATOM_ROOT = _FIXTURE_ROOT / "atoms"
_ATOM_ROOT.mkdir(parents=True, exist_ok=True)


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
    _SOURCE_FILE.write_text("accepted census fixture\n", encoding="utf-8")
    source = {
        "schema_version": "scryglass:future-value-rating-source:v1",
        "status": "accepted_source_bound_development_only",
        "source_as_of": "2026-01-03T00:00:00Z",
        "source_game_count": len(accepted),
        "source_identity_sha256": identity_sha256(accepted),
        "accepted_game_ids": list(accepted),
        "source_files": {
            "accepted_census": {
                "locator": _SOURCE_FILE.name,
                "bytes": _SOURCE_FILE.stat().st_size,
                "sha256": hashlib.sha256(_SOURCE_FILE.read_bytes()).hexdigest(),
            }
        },
        "authority": {"research_only": True, "deployment": False, "merge": False, "promotion": False},
    }
    source["receipt_sha256"] = _sha(source)
    _SOURCE_PATH.write_text(json.dumps(source, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    producer_artifact = {
        "schema_version": "scryglass:draft-records:v1",
        "authority": "descriptive",
        "estimand": "composition_only",
        "source_identity_sha256": source["source_identity_sha256"],
        "games": {game_id: {} for game_id in accepted},
    }
    producer_artifact_raw = json.dumps(
        producer_artifact,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    _PRODUCER_ARTIFACT_PATH.write_bytes(producer_artifact_raw)
    producer_artifact_receipt = {
        "schema_version": "scryglass:public-draft-artifact-receipt:v1",
        "artifact_locator": str(_PRODUCER_ARTIFACT_PATH),
        "artifact_bytes": len(producer_artifact_raw),
        "artifact_sha256": hashlib.sha256(producer_artifact_raw).hexdigest(),
        "source_receipt_sha256": source["receipt_sha256"],
        "source_identity_sha256": source["source_identity_sha256"],
    }
    producer_artifact_receipt_raw = json.dumps(
        producer_artifact_receipt,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    _PRODUCER_ARTIFACT_RECEIPT_PATH.write_bytes(producer_artifact_receipt_raw)
    fit_dates = {"fit1": "2025-12-31T00:00:00Z"}
    producer = {
        "schema_version": "scryglass:future-value-draft-score:v1",
        "source_receipt_sha256": source["receipt_sha256"],
        "source_identity_sha256": source["source_identity_sha256"],
        "accepted_game_count": len(accepted),
        "accepted_game_ids": list(accepted),
        "producer_name": "public_descriptive_draft_records",
        "producer_family": "draft_score_features",
        "fit_game_count": 1,
        "fit_game_ids": ["fit1"],
        "fit_game_identity_sha256": identity_sha256(("fit1",)),
        "fit_window_start": "2025-12-31T00:00:00Z",
        "fit_window_end": "2026-01-01T00:00:00Z",
        "fit_game_dates": fit_dates,
        "fold_id": "fold-0",
        "series_safe_evidence": {
            "series_safe": True,
            "fit_validation_disjoint": True,
            "source_type": "synthetic_verified",
            "series_column": "series_id",
            "cluster_identity_sha256": "b" * 64,
        },
        "producer_timing": "pregame_strict_prior",
        "artifact_locator": str(_PRODUCER_ARTIFACT_PATH),
        "artifact_bytes": len(producer_artifact_raw),
        "artifact_sha256": hashlib.sha256(producer_artifact_raw).hexdigest(),
        "artifact_receipt_locator": str(_PRODUCER_ARTIFACT_RECEIPT_PATH),
        "artifact_receipt_bytes": len(producer_artifact_receipt_raw),
        "artifact_receipt_sha256": hashlib.sha256(producer_artifact_receipt_raw).hexdigest(),
    }
    producer["receipt_sha256"] = _sha(producer)
    _PRODUCER_PATH.write_text(json.dumps(producer, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return DraftScoreProducerBinding.create(
        source_receipt=source,
        accepted_game_ids=accepted,
        fit_game_ids=("fit1",),
        fit_window_end="2026-01-01T00:00:00Z",
        fit_window_start="2025-12-31T00:00:00Z",
        fit_game_dates={"fit1": "2025-12-31T00:00:00Z"},
        fold_id="fold-0",
        producer="public_descriptive_draft_records",
        producer_family="draft_score_features",
        series_safe_evidence={
            "series_safe": True,
            "fit_validation_disjoint": True,
            "source_type": "synthetic_verified",
            "series_column": "series_id",
            "cluster_identity_sha256": "b" * 64,
        },
        source_receipt_path=_SOURCE_PATH,
        source_root=_FIXTURE_ROOT,
        producer_receipt_path=_PRODUCER_PATH,
    )


def _atom_receipt(frame: pd.DataFrame) -> dict[str, object]:
    component_hash = static_composition_parity_hash(frame)
    artifact_path = _ATOM_ROOT / "atomized.parquet"
    artifact_payload = {
        "schema_version": "scryglass:draft-records:v1",
        "authority": "descriptive",
        "estimand": "composition_only",
        "source_identity_sha256": _binding().source_identity_sha256,
        "games": {str(game_id): {} for game_id in frame["game_id"]},
    }
    artifact_path.write_bytes(
        json.dumps(artifact_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    artifact_receipt_path = _ATOM_ROOT / "atomized-artifact-receipt.json"
    artifact_payload = {
        "schema_version": "scryglass:public-draft-artifact-receipt:v1",
        "artifact_locator": str(artifact_path),
        "artifact_bytes": artifact_path.stat().st_size,
        "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "source_receipt_sha256": _binding().source_receipt_sha256,
        "source_identity_sha256": _binding().source_identity_sha256,
    }
    artifact_receipt_path.write_text(json.dumps(artifact_payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    payload: dict[str, object] = {
        "schema_version": "scryglass:atomized-composition-producer:v1",
        "producer_name": "public_descriptive_draft_records",
        "producer_family": "static_composition",
        "artifact_locator": str(artifact_path),
        "artifact_bytes": artifact_path.stat().st_size,
        "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "artifact_receipt_locator": str(artifact_receipt_path),
        "artifact_receipt_bytes": artifact_receipt_path.stat().st_size,
        "artifact_receipt_sha256": hashlib.sha256(artifact_receipt_path.read_bytes()).hexdigest(),
        "source_receipt_sha256": _binding().source_receipt_sha256,
        "source_identity_sha256": _binding().source_identity_sha256,
        "feature_names": list(STATIC_COMPOSITION_FEATURES),
        "component_values_sha256": component_hash,
    }
    payload["receipt_sha256"] = _sha(payload)
    return payload


def _atom_path(frame: pd.DataFrame) -> Path:
    component_hash = static_composition_parity_hash(frame)
    return _ATOM_ROOT / f"atom-receipt-{component_hash}.json"


def _write_atom(frame: pd.DataFrame) -> tuple[dict[str, object], Path]:
    payload = _atom_receipt(frame)
    path = _atom_path(frame)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return payload, path


def _atom_kwargs(frame: pd.DataFrame) -> dict[str, object]:
    payload, path = _write_atom(frame)
    return {"static_atom_receipt": payload, "static_atom_receipt_path": path}


def _coefficients_and_ledger(design: object) -> tuple[dict[str, float], dict[str, object], Path, Path]:
    coefficients = {feature: 1.0 for feature in design.feature_frame.columns}
    coefficient_receipt = make_coefficient_receipt(design, coefficients)
    model_artifact_path = _FIXTURE_ROOT / "prediction-model-artifact.json"
    model_artifact_raw = json.dumps(
        {
            "model_id": coefficient_receipt["model_id"],
            "variant": design.variant.value,
            "fit_id": coefficient_receipt["fit_id"],
            "coefficient_sha256": coefficient_receipt["coefficient_sha256"],
            "implementation": "independent-fixture-model",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    model_artifact_path.write_bytes(model_artifact_raw)
    implementation_sha256 = "c" * 64
    model_receipt = {
        "schema_version": "scryglass:future-value-draft-score-model-receipt:v1",
        "model_id": coefficient_receipt["model_id"],
        "model_version": "independent-fixture-v1",
        "variant": design.variant.value,
        "source_receipt_sha256": design.source_binding.source_receipt_sha256,
        "source_identity_sha256": design.source_binding.source_identity_sha256,
        "fold_id": design.source_binding.fold_id,
        "fit_game_ids": list(design.source_binding.fit_game_ids),
        "fit_game_identity_sha256": identity_sha256(design.source_binding.fit_game_ids),
        "fit_id": coefficient_receipt["fit_id"],
        "coefficient_sha256": coefficient_receipt["coefficient_sha256"],
        "artifact_locator": str(model_artifact_path),
        "artifact_bytes": len(model_artifact_raw),
        "artifact_sha256": hashlib.sha256(model_artifact_raw).hexdigest(),
        "implementation_sha256": implementation_sha256,
        "authority": {"research_only": True},
    }
    model_receipt["receipt_sha256"] = _sha(model_receipt)
    model_receipt_path = _FIXTURE_ROOT / "prediction-model-receipt.json"
    model_receipt_raw = json.dumps(model_receipt, sort_keys=True, separators=(",", ":")).encode()
    model_receipt_path.write_bytes(model_receipt_raw)
    rows = [
        {"game_id": game_id, "model_logit": float(value)}
        for game_id, value in zip(design.game_ids, design.feature_frame.sum(axis=1).to_numpy())
    ]
    ledger_path = _FIXTURE_ROOT / "prediction-ledger.json"
    ledger_path.write_text(json.dumps({"rows": rows}, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    ledger_receipt = {
        "schema_version": "scryglass:future-value-draft-score-prediction-ledger:v2",
        "source_receipt_sha256": design.source_binding.source_receipt_sha256,
        "source_identity_sha256": design.source_binding.source_identity_sha256,
        "game_ids": list(design.game_ids),
        "fold_id": design.source_binding.fold_id,
        "model_id": coefficient_receipt["model_id"],
        "fit_game_ids": list(design.source_binding.fit_game_ids),
        "fit_id": coefficient_receipt["fit_id"],
        "fit_game_identity_sha256": identity_sha256(design.source_binding.fit_game_ids),
        "coefficient_sha256": coefficient_receipt["coefficient_sha256"],
        "model_receipt_locator": str(model_receipt_path),
        "model_receipt_bytes": len(model_receipt_raw),
        "model_receipt_sha256": hashlib.sha256(model_receipt_raw).hexdigest(),
        "model_artifact_locator": str(model_artifact_path),
        "model_artifact_bytes": len(model_artifact_raw),
        "model_artifact_sha256": hashlib.sha256(model_artifact_raw).hexdigest(),
        "model_implementation_sha256": implementation_sha256,
        "row_digest_sha256": _sha(rows),
        "artifact_locator": str(ledger_path),
        "artifact_bytes": ledger_path.stat().st_size,
        "artifact_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        "authority": {"research_only": True},
    }
    ledger_receipt["receipt_sha256"] = _sha(ledger_receipt)
    receipt_path = _FIXTURE_ROOT / "prediction-ledger-receipt.json"
    receipt_path.write_text(json.dumps(ledger_receipt, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return coefficients, coefficient_receipt, ledger_path, receipt_path


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
        variant: build_draft_score_variant_design(frame, variant, binding, **_atom_kwargs(frame))
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
    design = build_draft_score_variant_design(frame, DraftScoreVariant.BOTH, binding, **_atom_kwargs(frame))
    coefficients = {feature: 1.0 for feature in design.feature_frame.columns}
    coefficient_receipt = make_coefficient_receipt(design, coefficients)
    _coefficients, _coefficient_receipt, ledger_path, ledger_receipt_path = _coefficients_and_ledger(design)
    score = score_draft_score_variant(
        design,
        coefficients,
        coefficient_receipt=coefficient_receipt,
        independent_prediction_ledger=ledger_path,
        independent_prediction_ledger_receipt=ledger_receipt_path,
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


def test_independent_prediction_writer_binds_model_receipt_and_artifact() -> None:
    frame = _frame()
    design = build_draft_score_variant_design(
        frame,
        DraftScoreVariant.CURRENT_ONLY,
        _binding(),
        **_atom_kwargs(frame),
    )
    coefficients, coefficient_receipt, _ledger_path, _ledger_receipt_path = _coefficients_and_ledger(design)
    del coefficients
    output_path, receipt_path = write_independent_prediction_ledger(
        _FIXTURE_ROOT / "writer-predictions.json",
        design.game_ids,
        design.feature_frame.sum(axis=1).to_numpy(),
        source_receipt_sha256=design.source_binding.source_receipt_sha256,
        source_identity_sha256=design.source_binding.source_identity_sha256,
        fold_id=design.source_binding.fold_id,
        fit_game_ids=design.source_binding.fit_game_ids,
        fit_id=coefficient_receipt["fit_id"],
        model_id=coefficient_receipt["model_id"],
        coefficient_sha256=coefficient_receipt["coefficient_sha256"],
        model_receipt_path=_FIXTURE_ROOT / "prediction-model-receipt.json",
        variant=design.variant,
    )
    score = score_draft_score_variant(
        design,
        {feature: 1.0 for feature in design.feature_frame.columns},
        coefficient_receipt=coefficient_receipt,
        independent_prediction_ledger=output_path,
        independent_prediction_ledger_receipt=receipt_path,
    )
    assert score.independent_prediction_error_max <= 1e-12
    model_artifact = _FIXTURE_ROOT / "prediction-model-artifact.json"
    model_artifact.write_bytes(model_artifact.read_bytes() + b"mutated")
    with pytest.raises(FutureValueDraftScoreError, match="model artifact|bytes"):
        score_draft_score_variant(
            design,
            {feature: 1.0 for feature in design.feature_frame.columns},
            coefficient_receipt=coefficient_receipt,
            independent_prediction_ledger=output_path,
            independent_prediction_ledger_receipt=receipt_path,
        )


@pytest.mark.parametrize("variant", tuple(DraftScoreVariant))
def test_side_swap_negates_signed_features_and_preserves_phase_invariants(
    variant: DraftScoreVariant,
) -> None:
    frame = _frame()
    design = build_draft_score_variant_design(frame, variant, _binding(), **_atom_kwargs(frame))
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
        build_draft_score_variant_design(overlapping, DraftScoreVariant.CURRENT_ONLY, binding, **_atom_kwargs(overlapping))

    same_time = frame.copy()
    same_time.loc[0, "date"] = "2026-01-01T00:00:00Z"
    with pytest.raises(FutureValueDraftScoreError, match="strictly prior"):
        build_draft_score_variant_design(same_time, DraftScoreVariant.CURRENT_ONLY, binding, **_atom_kwargs(same_time))


def test_static_parity_rejects_mutated_atom_value() -> None:
    frame = _frame()
    binding = _binding()
    first = build_draft_score_variant_design(frame, DraftScoreVariant.CURRENT_ONLY, binding, **_atom_kwargs(frame))
    changed = frame.copy()
    changed.loc[0, "composition_enemy_counter_logit"] += 0.01
    with pytest.raises(FutureValueDraftScoreError, match="atomized composition"):
        build_draft_score_variant_design(changed, DraftScoreVariant.CURRENT_ONLY, binding, **_atom_kwargs(frame))


def test_static_atom_receipt_rejects_resealed_path_and_producer_mutation() -> None:
    frame = _frame()
    binding = _binding()
    payload, path = _write_atom(frame)
    payload["producer_name"] = "public_crossfit_draft_score"
    payload["receipt_sha256"] = _sha(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    with pytest.raises(FutureValueDraftScoreError, match="producer artifact|trusted"):
        build_draft_score_variant_design(
            frame,
            DraftScoreVariant.CURRENT_ONLY,
            binding,
            static_atom_receipt=payload,
            static_atom_receipt_path=path,
        )


def test_strict_static_atoms_require_independent_authority_receipt() -> None:
    frame = _frame()
    binding = _binding()
    payload, path = _write_atom(frame)
    with pytest.raises(FutureValueDraftScoreError, match="independent atom authority"):
        build_draft_score_variant_design(
            frame,
            DraftScoreVariant.CURRENT_ONLY,
            binding,
            static_atom_receipt=payload,
            static_atom_receipt_path=path,
            require_independent_static_authority=True,
            static_atom_authority_path=_SOURCE_PATH,
            static_atom_authority_root=_FIXTURE_ROOT,
        )


def test_phase_invariants_are_diagnostics_and_never_score_features() -> None:
    config = VARIANT_CONFIGS[DraftScoreVariant.SCALING_CURVE]
    assert not set(PHASE_SHAPE_INVARIANT_FEATURES).intersection(config.feature_names)
    design = build_draft_score_variant_design(
        _frame(),
        DraftScoreVariant.SCALING_CURVE,
        _binding(),
        **_atom_kwargs(_frame()),
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


def test_self_sealed_producer_receipt_extra_field_fails_closed() -> None:
    source = dict(_binding().source_receipt or {})
    producer = json.loads(_PRODUCER_PATH.read_text())
    producer["forged_by"] = "attacker"
    producer["receipt_sha256"] = _sha(
        {key: value for key, value in producer.items() if key != "receipt_sha256"}
    )
    _PRODUCER_PATH.write_text(
        json.dumps(producer, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(FutureValueDraftScoreError, match="unknown fields"):
        DraftScoreProducerBinding.create(
            source_receipt=source,
            source_receipt_path=_SOURCE_PATH,
            source_root=_FIXTURE_ROOT,
            producer_receipt_path=_PRODUCER_PATH,
            accepted_game_ids=("fit1", "g1", "g2"),
            fit_game_ids=("fit1",),
            fit_window_end="2026-01-01T00:00:00Z",
            fit_window_start="2025-12-31T00:00:00Z",
            fit_game_dates={"fit1": "2025-12-31T00:00:00Z"},
            fold_id="fold-0",
            producer="public_descriptive_draft_records",
            producer_family="draft_score_features",
            series_safe_evidence={
                "series_safe": True,
                "fit_validation_disjoint": True,
                "source_type": "synthetic_verified",
                "series_column": "series_id",
                "cluster_identity_sha256": "b" * 64,
            },
        )
    _binding()


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
        **_atom_kwargs(frame),
    )
    coefficients, receipt, ledger_path, ledger_receipt_path = _coefficients_and_ledger(design)
    with pytest.raises(FutureValueDraftScoreError, match="coefficient"):
        score_draft_score_variant(design)
    changed_ledger = pd.DataFrame({"game_id": design.game_ids, "model_logit": design.feature_frame.sum(axis=1)})
    with pytest.raises(FutureValueDraftScoreError, match="durable path"):
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
            **_atom_kwargs(frame),
        )
    observed = _frame()
    observed["forecast_producer_timing"] = "observed"
    with pytest.raises(FutureValueDraftScoreError, match="timing"):
        build_draft_score_variant_design(
            observed,
            DraftScoreVariant.SCALING_CURVE,
            _binding(),
            **_atom_kwargs(observed),
        )


@pytest.mark.parametrize("timing", ("final_metric", "post_game"))
def test_all_variants_reject_non_pregame_producer_timing(timing: str) -> None:
    source = dict(_binding().source_receipt or {})
    producer = json.loads(_PRODUCER_PATH.read_text())
    producer["producer_timing"] = timing
    producer["receipt_sha256"] = _sha({key: value for key, value in producer.items() if key != "receipt_sha256"})
    _PRODUCER_PATH.write_text(json.dumps(producer, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    with pytest.raises(FutureValueDraftScoreError, match="pregame"):
        DraftScoreProducerBinding.create(
            source_receipt=source,
            source_receipt_path=_SOURCE_PATH,
            source_root=_FIXTURE_ROOT,
            producer_receipt_path=_PRODUCER_PATH,
            accepted_game_ids=("fit1", "g1", "g2"),
            fit_game_ids=("fit1",),
            fit_window_end="2026-01-01T00:00:00Z",
            fit_window_start="2025-12-31T00:00:00Z",
            fit_game_dates={"fit1": "2025-12-31T00:00:00Z"},
            fold_id="fold-0",
            producer="public_descriptive_draft_records",
            producer_family="draft_score_features",
            producer_timing=timing,
            series_safe_evidence={
                "series_safe": True,
                "fit_validation_disjoint": True,
                "source_type": "synthetic_verified",
                "series_column": "series_id",
                "cluster_identity_sha256": "b" * 64,
            },
        )
    # Restore the canonical fixture for the following tests.
    _binding()
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
        **_atom_kwargs(frame),
    )
    assert result["passed"] is True
