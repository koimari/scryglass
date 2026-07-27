from __future__ import annotations

from lol_kills.composition_model import CompositionGame, _penalty, feature_values, normalize_patch, predict_composition


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
        "intercept": intercept,
        "feature_specs": specs,
        "role_champion_counts": {"top|A": 100, "jng|B": 2, "top|F": 50},
        "components": ["main", "synergy", "opposition"],
        "prior_n": 25,
        "low_rank": low_rank or {"rank": 0, "champions": [], "left": [], "right": []},
        "calibration": {"intercept": 0.0, "slope": 1.0},
        "strength_calibration": {"coef_elo": 0.0},
    }


def _draft():
    return (
        ["A", "B", "C", "D", "E"],
        ["F", "G", "H", "I", "J"],
        ["top", "jng", "mid", "bot", "sup"],
        ["top", "jng", "mid", "bot", "sup"],
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


def test_blue_red_swap_is_antisymmetric_without_side_advantage():
    blue, red, blue_roles, red_roles = _draft()
    model = _model()
    left = predict_composition(model, blue, red, blue_roles=blue_roles, red_roles=red_roles)
    right = predict_composition(model, red, blue, blue_roles=red_roles, red_roles=blue_roles)
    assert abs(left["components"]["composition_edge"] + right["components"]["composition_edge"]) < 1e-5
    assert abs(left["p_blue_draft"] + right["p_blue_draft"] - 1.0) < 1e-4


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


def test_low_rank_residual_remains_antisymmetric():
    low_rank = {
        "rank": 1,
        "champions": ["A", "F"],
        "left": [[1.0], [2.0]],
        "right": [[3.0], [5.0]],
    }
    model = _model(low_rank=low_rank)
    blue, red, blue_roles, red_roles = _draft()
    left = predict_composition(model, blue, red, blue_roles=blue_roles, red_roles=red_roles)
    right = predict_composition(model, red, blue, blue_roles=red_roles, red_roles=blue_roles)
    assert abs(left["components"]["low_rank_logit"] + right["components"]["low_rank_logit"]) < 1e-5


def test_sparse_terms_receive_stronger_neutral_shrinkage():
    assert _penalty("opposition", 1) > _penalty("opposition", 100)
    assert _penalty("league", 1) > _penalty("main", 100)


def test_numeric_patch_suffix_preserves_early_and_late_patch_identity():
    assert normalize_patch("16.01") == "16.01"
    assert normalize_patch("16.1") == "16.10"


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
    model["strength_calibration"] = {
        "team_intercept": 0.0,
        "team_coef": 2.0,
        "player_intercept": 0.0,
        "player_coef": 2.0,
        "blend_intercept": -2.0,
        "blend_coef_team": 2.0,
        "blend_coef_player": 2.0,
    }
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
