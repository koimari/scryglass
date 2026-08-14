from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

import lol_kills.research.draft_phase_curve as phase_curve
from lol_kills.research.draft_phase_curve import (
    BASELINE_AUC_FLOOR,
    PHASES,
    assert_no_gold_leakage,
    atomized_draft_features,
    fit_phase_curve,
    live_state_features,
    pre_match_features_for_draft,
    score_phase_curve,
    unavailable_phase_curve,
)


def test_pre_match_features_reject_observed_gold() -> None:
    with pytest.raises(ValueError, match="post-game"):
        assert_no_gold_leakage(["elo_diff", "goldat10"])


def test_missing_25_is_censored_and_gold_30_is_not_a_target() -> None:
    rows = pd.DataFrame(
        [
            {
                "game_uid": "g1",
                "side": "blue",
                "oe_patch_token": "16.15",
                "goldat10": 1000,
                "goldat15": 1500,
                "goldat20": 2000,
            },
            {
                "game_uid": "g1",
                "side": "red",
                "oe_patch_token": "16.15",
                "goldat10": 900,
                "goldat15": 1400,
                "goldat20": 1900,
            },
        ]
    )
    maps = pd.DataFrame([{"game_uid": "g1", "blue_win": 1}])
    from lol_kills.research.draft_phase_curve import prepare_phase_frame

    prepared = prepare_phase_frame(rows, maps)
    assert prepared.loc[0, "gold_diff_10"] == 100
    assert pd.isna(prepared.loc[0, "gold_diff_25"])
    assert "gold_diff_30" not in prepared.columns


def test_patch_16_16_is_required_for_fit_and_16_15_stays_source_token() -> None:
    frame = pd.DataFrame(
        {
            "game_uid": ["g1", "g2"],
            "date": ["2026-08-01", "2026-08-02"],
            "oe_patch_token": ["16.15", "16.15"],
            "draft_win_logit_blue": [0.1, -0.1],
            "gold_diff_10": [100, -100],
        }
    )
    artifact = fit_phase_curve(frame, min_patch_rows=1)
    assert artifact["authority"] == "unavailable"
    assert "accepted_oe_patch_16.16_rows_missing" in artifact["blockers"]
    assert artifact["source_snapshot"]["oe_source_token"] == "16.15"
    assert artifact["source_snapshot"]["derived_client_patches"] == ["16.15"]


def test_missing_date_fails_chronological_gate_without_crashing() -> None:
    frame = pd.DataFrame(
        {
            "game_uid": [f"g{index}" for index in range(8)],
            "oe_patch_token": ["16.16"] * 8,
            "draft_win_logit_blue": [0.1, -0.1] * 4,
            "gold_diff_10": [100, -100] * 4,
            "y_blue_win": [1, 0] * 4,
        }
    )
    artifact = fit_phase_curve(frame, min_patch_rows=1)
    assert artifact["evaluation"]["gates"]["chronological"] is False


def test_prepared_frame_separates_corrected_patch_from_raw_oe_token() -> None:
    from lol_kills.research.draft_phase_curve import prepare_phase_frame

    rows = pd.DataFrame(
        [
            {
                "game_uid": "g1",
                "side": "blue",
                "patch": "16.16",
                "oe_patch_token": "16.15",
                "goldat10": 1100,
            },
            {
                "game_uid": "g1",
                "side": "red",
                "patch": "16.16",
                "oe_patch_token": "16.15",
                "goldat10": 1000,
            },
        ]
    )
    prepared = prepare_phase_frame(rows, pd.DataFrame([{"game_uid": "g1", "blue_win": 1}]))

    assert prepared.loc[0, "oe_patch_token"] == "16.16"
    assert prepared.loc[0, "oe_source_token"] == "16.15"


def test_lcc_atom_features_are_pre_match_and_live_state_is_separate() -> None:
    features = pre_match_features_for_draft(
        ["Aatrox", "Ahri", "Ashe"],
        ["Garen", "Orianna", "Jinx"],
        0.2,
        league="LCK",
        region="Korea",
        patch="16.16",
    )
    assert features["oe_patch_token"] == "16.16"
    assert all("gold" not in key.casefold() for key in features)
    assert isinstance(atomized_draft_features(["Aatrox"], ["Garen"]), dict)
    live = live_state_features({"gold_diff": 300, "clock": 900, "objectives": {"dragon": 1}})
    assert live["source"] == "timeline_live_state"
    assert "gold_diff" in live


def test_pre_match_patch_feature_prefers_realm_aware_token() -> None:
    features = pre_match_features_for_draft(
        ["Aatrox"],
        ["Garen"],
        0.1,
        patch="16.16",
        extra_features={"patch": "16.16", "oe_patch_token": "16.15"},
    )

    assert features["oe_patch_token"] == "16.16"


def test_phase_output_is_unavailable_without_promotion() -> None:
    artifact = unavailable_phase_curve()
    scored = score_phase_curve(artifact, {"draft_win_logit_blue": 0.2}, draft_logit=0.2)
    assert scored["authority"] == "unavailable"
    assert scored["expected_gold_diff"] == {phase: None for phase in PHASES}
    assert BASELINE_AUC_FLOOR > 0.7


def test_promoted_phase_curve_requires_file_bound_independent_receipt(tmp_path, monkeypatch) -> None:
    models_dir = tmp_path / "data" / "lol" / "models"
    promotion_dir = models_dir / "promotions"
    promotion_dir.mkdir(parents=True)
    artifact_path = models_dir / "draft_phase_curve.json"
    monkeypatch.setattr(phase_curve, "ROOT", tmp_path)
    monkeypatch.setattr(phase_curve, "MODELS_DIR", models_dir)
    monkeypatch.setattr(phase_curve, "PROMOTION_ROOT", promotion_dir)
    gates = {name: True for name in phase_curve.REQUIRED_PROMOTION_GATES}
    artifact = {
        "schema_version": phase_curve.SCHEMA_VERSION,
        "authority": "promoted",
        "source": "oe_only",
        "window": list(PHASES),
        "model_version": "phase-test-v1",
        "design_features": ["num:draft_win_logit_blue"],
        "phase_models": {
            phase: {"coefficients": [1.0], "intercept": 0.0} for phase in PHASES
        },
        "phase_draft_coefficients": {phase: 1.0 for phase in PHASES},
        "evaluation": {
            "auc": 0.8,
            "auc_floor": 0.7,
            "auc_noninferior": True,
            "gates": gates,
        },
        "promotion": {},
    }
    model_hash = hashlib.sha256(
        phase_curve._canonical_json_bytes(phase_curve._phase_model_payload(artifact))
    ).hexdigest()
    artifact_hash = hashlib.sha256(
        phase_curve._canonical_json_bytes(phase_curve._artifact_payload_without_promotion(artifact))
    ).hexdigest()
    receipt = {
        "schema_version": phase_curve.PROMOTION_SCHEMA_VERSION,
        "status": "approved",
        "authority": "independent",
        "model_version": artifact["model_version"],
        "model_sha256": model_hash,
        "artifact_sha256": artifact_hash,
        "gates": gates,
    }
    receipt_path = promotion_dir / "phase-test.json"
    receipt_bytes = phase_curve._canonical_json_bytes(receipt)
    receipt_path.write_bytes(receipt_bytes)
    artifact["promotion"] = {
        "receipt_path": "data/lol/models/promotions/phase-test.json",
        "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "model_sha256": model_hash,
    }
    artifact_path.write_bytes(phase_curve._canonical_json_bytes(artifact))
    loaded = json.loads(artifact_path.read_text(encoding="utf-8"))
    scored = score_phase_curve(
        loaded,
        {"draft_win_logit_blue": 0.2},
        draft_logit=0.2,
        artifact_path=artifact_path,
    )
    assert scored["authority"] == "promoted"

    receipt_path.write_text(receipt_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert score_phase_curve(
        loaded,
        {"draft_win_logit_blue": 0.2},
        draft_logit=0.2,
        artifact_path=artifact_path,
    )["authority"] == "unavailable"
