from __future__ import annotations

import json
import math

import pytest

from lol_kills.composition_model import (
    CompositionArtifactError,
    CompositionGame,
    _opposition_key,
    _penalty,
    _strength_calibration,
    composition_games_sha256,
    export_runtime,
    feature_values,
    model_code_sha256,
    normalize_patch,
    predict_composition,
)


def _disabled_low_rank() -> dict:
    return {
        "status": "disabled",
        "rank": 0,
        "champions": [],
        "left": [],
        "right": [],
        "reason": "test fixture disables unbounded low-rank terms",
    }


def _available_strength_calibration() -> dict:
    return {
        "schema_version": "1.0.0",
        "status": "available",
        "calibration_id": "strength-calibration-v2-test",
        "fit_cutoff": "2026-01-01T00:00:00Z",
        "holdout_start": "2026-02-01T00:00:00Z",
        "source": {
            "artifact": "data/lol/models/elo_wr_calibration.json",
            "artifact_sha256": "a" * 64,
            "artifact_version": 2,
        },
        "team": {
            "model_id": "strength-calibration-v2-test-team",
            "intercept": 0.0,
            "coef": 2.0,
        },
        "player": {
            "model_id": "strength-calibration-v2-test-player",
            "intercept": 0.0,
            "coef": 2.0,
        },
        "blend": {
            "model_id": "strength-calibration-v2-test-blend",
            "intercept": -2.0,
            "coef_team": 2.0,
            "coef_player": 2.0,
        },
    }


def _model(*, intercept: float = 0.0, low_rank: dict | None = None) -> dict:
    specs = {
        "main|top|A": {"coef": 0.2, "se": 0.01},
        "main|jng|B": {"coef": 0.1, "se": 0.01},
        "main|top|F": {"coef": 0.05, "se": 0.01},
        "synergy|A|B": {"coef": 0.4, "se": 0.02},
    }
    for enemy in ["F", "G", "H", "I", "J"]:
        specs[f"opposition|A|{enemy}"] = {"coef": 0.1, "se": 0.01}
    return {
        "version": 2,
        "model_code_sha256": "c" * 64,
        "training_population_sha256": "d" * 64,
        "numerical_environment": {
            "python": "3.13.0",
            "packages": {
                "numpy": "2.0.0",
                "pandas": "2.0.0",
                "scipy": "1.13.0",
                "scikit-learn": "1.8.0",
            },
        },
        "intercept": intercept,
        "intercept_se": 0.02,
        "feature_specs": specs,
        "role_champion_counts": {"top|A": 100, "jng|B": 2, "top|F": 50},
        "components": ["main", "synergy", "opposition"],
        "prior_n": 25,
        "low_rank": low_rank or _disabled_low_rank(),
        "calibration": {
            "intercept": 0.0,
            "slope": 1.0,
            "covariance": [[0.0, 0.0], [0.0, 0.0]],
        },
        "calibration_source": "test chronological calibration",
        "strength_calibration": {
            "schema_version": "1.0.0",
            "status": "unavailable",
            "reason": "test fixture has no contextual calibration",
            "source": {
                "artifact": "test",
                "artifact_sha256": None,
                "artifact_version": None,
            },
        },
    }


def _draft():
    return (
        ["A", "B", "C", "D", "E"],
        ["F", "G", "H", "I", "J"],
        ["top", "jng", "mid", "bot", "sup"],
        ["top", "jng", "mid", "bot", "sup"],
    )


def test_model_source_and_training_population_hashes_are_stable(tmp_path):
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("first\n", encoding="utf-8")
    second.write_text("second\n", encoding="utf-8")
    assert model_code_sha256((second, first)) == model_code_sha256((first, second))

    blue, red, blue_roles, red_roles = _draft()
    games = [
        CompositionGame(
            game_id="g1",
            blue=tuple(zip(blue_roles, blue)),
            red=tuple(zip(red_roles, red)),
            y=1,
            league="LCK",
            patch="16.14",
            date=None,
        )
    ]
    assert composition_games_sha256(games) == composition_games_sha256(
        list(reversed(games))
    )


def test_full_opposition_uses_all_enemy_champions_and_reconciles():
    blue, red, blue_roles, red_roles = _draft()
    result = predict_composition(
        _model(), blue, red, blue_roles=blue_roles, red_roles=red_roles, league="LCK", patch="16.13"
    )
    rows = result["explanation"]["champions"]
    assert result["explanation"]["reconciles"] is True
    assert result["components"]["opposition_logit"] == 0.5
    assert len(rows) == 10
    assert abs(sum(row["edge_contribution"] for row in rows) - result["components"]["composition_edge"]) < 1e-5
    assert abs(sum(row["enemy_interaction"] for row in rows) - 0.5) < 1e-5


def test_complete_probability_uses_full_calibrated_blue_side_predictor():
    blue, red, blue_roles, red_roles = _draft()
    model = _model(intercept=0.12)
    model["calibration"] = {
        "intercept": 0.3,
        "slope": 0.7,
        "covariance": [[0.2, 0.01], [0.01, 0.1]],
    }
    left = predict_composition(model, blue, red, blue_roles=blue_roles, red_roles=red_roles)
    right = predict_composition(model, red, blue, blue_roles=red_roles, red_roles=blue_roles)
    assert abs(left["components"]["composition_edge"] + right["components"]["composition_edge"]) < 1e-5
    for result in (left, right):
        expected = 1.0 / (
            1.0
            + math.exp(
                -(
                    model["calibration"]["intercept"]
                    + model["calibration"]["slope"]
                    * result["components"]["model_edge"]
                )
            )
        )
        assert result["p_blue_draft"] == pytest.approx(expected, abs=1e-4)
        lo, hi = result["uncertainty"]["p_blue_95"]
        assert 0.0 <= lo <= result["p_blue_draft"] <= hi <= 1.0
    neutral = 1.0 / (
        1.0
        + math.exp(
            -(
                model["calibration"]["intercept"]
                + model["calibration"]["slope"] * model["intercept"]
            )
        )
    )
    assert left["calibration"]["neutral_blue_baseline"] == pytest.approx(
        neutral, abs=1e-4
    )
    assert left["wr_bump_pp"] == pytest.approx(
        100.0 * (left["p_blue_draft"] - neutral), abs=0.02
    )


def test_same_champion_opposition_is_exactly_directionless():
    key, orientation = _opposition_key("A", "A")
    assert key == "opposition|A|A"
    assert orientation == 0
    blue, red, blue_roles, red_roles = _draft()
    red[0] = "A"
    model = _model()
    model["feature_specs"]["opposition|A|A"] = {
        "coef": 100.0,
        "se": 100.0,
    }
    result = predict_composition(
        model,
        blue,
        red,
        blue_roles=blue_roles,
        red_roles=red_roles,
    )
    assert result["components"]["opposition_logit"] == 0.4


def test_side_advantage_is_separate_from_composition_ledger():
    blue, red, blue_roles, red_roles = _draft()
    result = predict_composition(
        _model(intercept=0.12), blue, red, blue_roles=blue_roles, red_roles=red_roles
    )
    assert result["explanation"]["edge"] == round(
        result["explanation"]["composition_edge"] + result["explanation"]["side_advantage"], 6
    )
    assert result["components"]["side_advantage_logit"] == 0.12


def test_role_order_is_invariant_when_role_labels_move_with_picks():
    blue, red, blue_roles, red_roles = _draft()
    base = predict_composition(_model(), blue, red, blue_roles=blue_roles, red_roles=red_roles)
    permuted = predict_composition(
        _model(),
        [blue[1], blue[0], *blue[2:]],
        [red[1], red[0], *red[2:]],
        blue_roles=[blue_roles[1], blue_roles[0], *blue_roles[2:]],
        red_roles=[red_roles[1], red_roles[0], *red_roles[2:]],
    )
    assert permuted["components"]["composition_edge"] == base["components"]["composition_edge"]


def test_active_low_rank_is_rejected_before_any_bounded_score():
    low_rank = {
        "rank": 1,
        "champions": ["A", "F"],
        "left": [[1.0], [2.0]],
        "right": [[3.0], [5.0]],
    }
    model = _model(low_rank=low_rank)
    blue, red, blue_roles, red_roles = _draft()
    with pytest.raises(CompositionArtifactError, match="low_rank"):
        predict_composition(
            model,
            blue,
            red,
            blue_roles=blue_roles,
            red_roles=red_roles,
        )


def test_sparse_terms_receive_stronger_neutral_shrinkage():
    assert _penalty("opposition", 1) > _penalty("opposition", 100)
    assert _penalty("league", 1) > _penalty("main", 100)


def test_numeric_patch_suffix_preserves_early_and_late_patch_identity():
    assert normalize_patch("16.01") == "16.01"
    with pytest.raises(CompositionArtifactError, match="ambiguous patch"):
        normalize_patch("16.1")
    assert (
        normalize_patch("16.1", allow_source_numeric_minor=True)
        == "16.10"
    )


def test_feature_vector_is_independent_of_outcome_label():
    kwargs = {
        "game_id": "g",
        "blue": tuple(zip(["top", "jng", "mid", "bot", "sup"], ["A", "B", "C", "D", "E"])),
        "red": tuple(zip(["top", "jng", "mid", "bot", "sup"], ["F", "G", "H", "I", "J"])),
        "league": "LCK",
        "patch": "16.13",
        "date": None,
    }
    assert feature_values(CompositionGame(y=0, **kwargs)) == feature_values(CompositionGame(y=1, **kwargs))


def test_contextualized_score_uses_team_and_player_strength_separately():
    blue, red, blue_roles, red_roles = _draft()
    model = _model()
    model["strength_calibration"] = _available_strength_calibration()
    raw = predict_composition(model, blue, red, blue_roles=blue_roles, red_roles=red_roles)
    contextualized = predict_composition(
        model,
        blue,
        red,
        blue_roles=blue_roles,
        red_roles=red_roles,
        team_elo_diff=200,
        player_elo_diff=100,
        strength_source="test strength",
    )
    assert raw["contextualized"] is None
    assert contextualized["raw"]["p_blue"] == raw["raw"]["p_blue"]
    assert contextualized["contextualized"]["p_blue"] > contextualized["raw"]["p_blue"]
    assert contextualized["strength"]["team_elo_diff"] == 200
    assert contextualized["strength"]["player_elo_diff"] == 100


def test_contextualized_score_rejects_partial_or_unavailable_calibration():
    blue, red, blue_roles, red_roles = _draft()
    model = _model()
    with pytest.raises(CompositionArtifactError, match="unavailable"):
        predict_composition(
            model,
            blue,
            red,
            blue_roles=blue_roles,
            red_roles=red_roles,
            team_elo_diff=100,
        )

    model["strength_calibration"] = _available_strength_calibration()
    del model["strength_calibration"]["player"]["model_id"]
    with pytest.raises(CompositionArtifactError, match="player model_id"):
        predict_composition(
            model,
            blue,
            red,
            blue_roles=blue_roles,
            red_roles=red_roles,
            team_elo_diff=100,
        )


def test_missing_active_term_uncertainty_rejects_bounds():
    blue, red, blue_roles, red_roles = _draft()
    model = _model()
    del model["feature_specs"]["main|top|A"]["se"]
    with pytest.raises(CompositionArtifactError, match=r"main\|top\|A.se"):
        predict_composition(
            model,
            blue,
            red,
            blue_roles=blue_roles,
            red_roles=red_roles,
        )


def test_active_composition_and_calibration_slope_uncertainty_widen_raw_interval():
    blue, red, blue_roles, red_roles = _draft()
    narrow = _model()
    wide = _model()
    wide["feature_specs"]["main|top|A"]["se"] = 0.5
    wide["calibration"]["covariance"] = [[0.0, 0.0], [0.0, 0.2]]
    narrow_result = predict_composition(
        narrow,
        blue,
        red,
        blue_roles=blue_roles,
        red_roles=red_roles,
    )
    wide_result = predict_composition(
        wide,
        blue,
        red,
        blue_roles=blue_roles,
        red_roles=red_roles,
    )
    narrow_interval = narrow_result["uncertainty"]["p_blue_95"]
    wide_interval = wide_result["uncertainty"]["p_blue_95"]
    assert wide_interval[1] - wide_interval[0] > narrow_interval[1] - narrow_interval[0]


def test_legacy_strength_artifact_is_explicitly_unavailable_without_defaults(
    tmp_path,
):
    path = tmp_path / "elo_wr_calibration.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "team": {"intercept": 0.1},
                "player": {},
                "strength_blend": {},
            }
        ),
        encoding="utf-8",
    )
    calibration = _strength_calibration(path)
    assert calibration["status"] == "unavailable"
    assert "version 2" in calibration["reason"]
    assert "team" not in calibration
    assert "player" not in calibration
    assert "blend" not in calibration


def test_runtime_preserves_public_validation_and_artifact_identity(tmp_path):
    model = {
        **_model(),
        "estimand": "test composition estimand",
        "n_games_fit": 100,
        "n_games_total": 120,
        "date_min": "2025-01-01T00:00:00Z",
        "date_max": "2025-12-31T00:00:00Z",
        "min_support": 3,
        "recency_half_life_days": 365.0,
        "validation": {
            "time_holdout": {"log_loss": 0.69},
            "future_patch_holdout": {"ece_10": 0.08},
        },
        "uncertainty": {
            "schema_version": "1.0.0",
            "method": "test complete uncertainty",
            "active_terms": ["main"],
            "low_rank_status": "disabled",
        },
        "limitations": ["observational"],
    }
    runtime = export_runtime(
        model,
        tmp_path / "runtime.json",
        artifact_sha256="b" * 64,
    )

    assert runtime["artifact_sha256"] == "b" * 64
    assert runtime["n_games_total"] == 120
    assert runtime["validation"]["future_patch_holdout"]["ece_10"] == 0.08
    assert runtime["low_rank"]["status"] == "disabled"
    assert runtime["limitations"] == ["observational"]
