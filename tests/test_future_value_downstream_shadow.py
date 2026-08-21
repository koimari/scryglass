from __future__ import annotations

import json
from pathlib import Path

from benchmarks.build_future_value_downstream_shadow import (
    SCHEMA_VERSION,
    _component_rows,
    _effect_metrics,
    _real_draft_rows,
    _real_tier_rows,
)


def test_component_rows_prefers_strict_prior_calibrated_evidence() -> None:
    payload = {
        "folds": [
            {
                "component_evidence": {
                    "rows": [
                        {
                            "game_id": "g1",
                            "current_rating_logit": 1.0,
                            "player_value_logit": 2.0,
                            "scaling_curve_logit": 3.0,
                            "full_model_logit": 6.0,
                            "calibrated_current_rating_logit": 0.5,
                            "calibrated_player_value_logit": 1.0,
                            "calibrated_scaling_curve_logit": 1.5,
                            "calibrated_full_model_logit": 3.0,
                            "calibration_slope": 0.5,
                            "support_status": "adequate",
                        }
                    ]
                }
            }
        ]
    }

    rows = _component_rows(payload, "both")

    assert rows["g1"]["current_rating_logit"] == 0.5
    assert rows["g1"]["player_value_logit"] == 1.0
    assert rows["g1"]["scaling_curve_logit"] == 1.5
    assert rows["g1"]["full_model_logit"] == 3.0
    assert rows["g1"]["component_scale"] == "strict_prior_calibrated"


def test_real_draft_rows_keep_the_five_public_components() -> None:
    games = {
        "g1": {
            "league": "LCK",
            "competition_tier": "tier1",
            "draft_edge": 0.12,
            "edge_components": {
                "base": 0.1,
                "ally_synergy": 0.02,
                "archetype_interactions": -0.01,
                "enemy_counter": 0.0,
                "same_role": 0.01,
                "total": 0.12,
            },
        }
    }
    evidence = {
        "g1": {
            "current_rating_logit": 0.2,
            "player_value_logit": 0.3,
            "scaling_curve_logit": 0.4,
            "full_model_logit": 0.9,
        }
    }
    rows, coverage = _real_draft_rows(
        games,
        evidence,
        "both",
        {"g1": {"base_team_logit": 0.05, "base_player_logit": 0.15}},
    )
    assert coverage == {"draft_rows": 1, "draft_rows_missing_components": 0}
    assert rows[0]["base"] == 0.1
    assert rows[0]["ally_synergy"] == 0.02
    assert rows[0]["enemy_counter"] == 0.0
    assert rows[0]["future_player_form_logit"] == 0.3
    assert rows[0]["scaling_raw_logit"] == 0.4
    assert rows[0]["composite_logit"] == 0.9
    assert "crossfit_same_role" not in rows[0]


def test_real_tier_rows_are_deduplicated_by_public_scope_identity() -> None:
    games = {
        "g1": {
            "league": "LCK",
            "draft_pool": {
                "patch": "26.08",
                "picked": [
                    {"champion": "Ahri", "role": "mid", "tier_rank": 2},
                ],
            },
        },
        "g2": {
            "league": "LCK",
            "draft_pool": {
                "patch": "26.08",
                "picked": [
                    {"champion": "Ahri", "role": "mid", "tier_rank": 2},
                ],
            },
        },
    }
    rows = _real_tier_rows(games, {"g1", "g2"})
    assert rows == [
        {"champion": "Ahri", "patch": "26.08", "rank": 2, "role": "mid", "scope": "LCK"}
    ]


def test_effect_metrics_exposes_nullable_variant_coverage(tmp_path: Path) -> None:
    source = {"receipt_sha256": "a" * 64}
    for variant in ("current_only", "future_player_form", "scaling_curve", "both"):
        root = tmp_path / variant
        for relative, rows in {
            "features/draft_records.json": [
                {
                    "game_uid": "g1",
                    "future_player_form_logit": 0.2 if variant in {"future_player_form", "both"} else None,
                    "scaling_raw_logit": 0.1 if variant in {"scaling_curve", "both"} else None,
                    "composite_logit": 0.3 if variant == "current_only" else 0.5,
                }
            ],
            "features/profile_records.json": [
                {
                    "game_uid": "g1",
                    "future_value": 0.2 if variant in {"future_player_form", "both"} else None,
                    "scaling_curve": 0.1 if variant in {"scaling_curve", "both"} else None,
                }
            ],
            "features/match_records_shadow.json": [
                {
                    "game_uid": "g1",
                    "future_player_value": 0.2 if variant in {"future_player_form", "both"} else None,
                    "future_team_value": None,
                }
            ],
        }.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "rows": rows}), encoding="utf-8")
    effects = _effect_metrics(
        {name: tmp_path / name for name in ("current_only", "future_player_form", "scaling_curve", "both")},
        source,
    )
    assert effects["draft"]["future_player_form_logit"]["future_player_form"]["available_rows"] == 1
    assert effects["draft"]["scaling_raw_logit"]["scaling_curve"]["available_rows"] == 1
    assert effects["matches"]["future_team_value"]["both"]["available_rows"] == 0
