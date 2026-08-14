from __future__ import annotations

import pandas as pd
import pytest

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
