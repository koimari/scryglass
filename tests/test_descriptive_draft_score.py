from copy import deepcopy

import pytest

from lol_kills.research.composition_signal import CompositionSignalError, validate_public_signal
from lol_kills.research.descriptive_draft_score import (
    MODEL_VERSION,
    SCHEMA_VERSION,
    load_model,
    score_game,
)


ROLES = ("top", "jng", "mid", "bot", "sup")


def _game() -> dict:
    blue = ("Aatrox", "Vi", "Ahri", "Jinx", "Rakan")
    red = ("Gnar", "Sejuani", "Azir", "Varus", "Nautilus")
    game = {
        "date": "2026-08-14T12:00:00Z",
        "blue": {},
        "red": {},
        "players": [],
    }
    for side, champions in (("blue", blue), ("red", red)):
        for role, champion in zip(ROLES, champions):
            player = f"{side}-{role}"
            game[side][role] = {"champion": champion, "player": player}
            game["players"].append(
                {"side": side.title(), "role": role, "champion": champion, "player": player}
            )
    return game


def test_static_descriptive_score_exposes_real_component_ledger() -> None:
    model, artifact_sha256 = load_model()
    signal = score_game(_game(), model=model, artifact_sha256=artifact_sha256)

    assert signal["schema_version"] == SCHEMA_VERSION
    assert signal["model_version"] == MODEL_VERSION
    assert signal["status"] == "available"
    assert signal["artifact_sha256"] == artifact_sha256
    assert signal["archetype_interaction_source"]["lcc_atoms"] == "excluded"
    assert signal["player_comfort"] == {
        "status": "unavailable",
        "contribution": None,
        "source": None,
        "sha256": None,
        "reason": "No release-bound player familiarity source is available.",
    }
    assert set(signal["edge_components"]) == {
        "base",
        "archetype_interactions",
        "ally_synergy",
        "enemy_counter",
        "same_role",
        "total",
    }
    assert signal["edge_components"]["total"] == round(
        sum(
            signal["edge_components"][key]
            for key in (
                "base",
                "archetype_interactions",
                "ally_synergy",
                "enemy_counter",
                "same_role",
            )
        ),
        6,
    )
    assert "p_blue" not in signal
    assert "probability" not in signal
    assert "draft_win_share" not in signal

    validate_public_signal(signal, _game())


def test_static_descriptive_score_ignores_outcome_and_strength_controls() -> None:
    model, artifact_sha256 = load_model()
    first = score_game(_game(), model=model, artifact_sha256=artifact_sha256)
    changed = deepcopy(_game())
    changed["y"] = 0
    changed["mu_diff"] = 9999
    changed["sigma_pair"] = 0
    changed["league"] = "OTHER"
    changed["patch"] = "99.99"
    second = score_game(changed, model=model, artifact_sha256=artifact_sha256)
    assert first == second


def test_static_descriptive_score_is_antisymmetric_under_side_swap() -> None:
    model, artifact_sha256 = load_model()
    original = _game()
    swapped = deepcopy(original)
    swapped["blue"], swapped["red"] = swapped["red"], swapped["blue"]

    first = score_game(original, model=model, artifact_sha256=artifact_sha256)
    second = score_game(swapped, model=model, artifact_sha256=artifact_sha256)

    for key, value in first["edge_components"].items():
        assert second["edge_components"][key] == -value


def test_static_descriptive_score_rejects_r9e_and_strength_output() -> None:
    model, artifact_sha256 = load_model()
    signal = score_game(_game(), model=model, artifact_sha256=artifact_sha256)
    signal["r9e_state_space"] = {"score": 0.2}

    with pytest.raises(CompositionSignalError, match="private composition fields"):
        validate_public_signal(signal, _game())


def test_static_descriptive_score_rejects_missing_role_input() -> None:
    model, artifact_sha256 = load_model()
    game = _game()
    del game["blue"]["mid"]

    with pytest.raises(ValueError, match="Blue mid champion is missing"):
        score_game(game, model=model, artifact_sha256=artifact_sha256)


def test_static_descriptive_score_rejects_duplicate_champions() -> None:
    model, artifact_sha256 = load_model()
    game = _game()
    game["red"]["mid"]["champion"] = game["blue"]["mid"]["champion"]

    with pytest.raises(ValueError, match="duplicate champions"):
        score_game(game, model=model, artifact_sha256=artifact_sha256)


def test_static_descriptive_score_marks_sparse_off_role_as_shrunk_estimate() -> None:
    model, artifact_sha256 = load_model()
    model["champion_role_counts"]["Akshan"] = {"bot": 1}
    game = _game()
    game["blue"]["bot"]["champion"] = "Akshan"

    signal = score_game(game, model=model, artifact_sha256=artifact_sha256)

    assert signal["status"] == "available"
    assert any(
        pick["champion"] == "Akshan"
        and pick["evidence_status"] == "role_estimate"
        and pick["contribution"] is not None
        for pick in signal["picks"]
    )
    for player in game["players"]:
        if player["side"] == "Blue" and player["role"] == "bot":
            player["champion"] = "Akshan"
    validate_public_signal(signal, game)
