from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import lol_kills.research.private_draft_score_wrapper as wrapper


def _args(scorer: Path, binding: Path) -> Namespace:
    return Namespace(
        scorer=str(scorer),
        recipe_binding=str(binding),
        league="LCK",
        blue_name="Blue",
        red_name="Red",
        blue_players="TopB,JungleB,MidB,BotB,SupportB",
        red_players="TopR,JungleR,MidR,BotR,SupportR",
        blue="Aatrox,Lee Sin,Ahri,Jinx,Thresh",
        red="Renekton,Sejuani,Orianna,Aphelios,Leona",
        as_of=None,
        patch=None,
        r9e_player_elo_p=None,
        r9e_player_elo_sigma=None,
    )


def _fake_scorer(tmp_path: Path) -> Path:
    scorer = tmp_path / "score_private_draft.py"
    scorer.write_text(
        "import json\n"
        "print(json.dumps({\n"
        "    'draft_score': {'blue': 0.12, 'red': -0.12},\n"
        "    'selected_model': 'team_only',\n"
        "    'authority': {\n"
        "        'model_validation_authority': False,\n"
        "        'probability_authority': False,\n"
        "        'recommendation_authority': False,\n"
        "        'betting_authority': False,\n"
        "    },\n"
        "}))\n",
        encoding="utf-8",
    )
    return scorer


def test_wrapper_preserves_owner_score_and_adds_private_recipe(tmp_path: Path) -> None:
    result = wrapper.run(_args(_fake_scorer(tmp_path), tmp_path / "missing-binding.json"))

    assert result["draft_score"] == {"blue": 0.12, "red": -0.12}
    assert result["selected_model"] == "team_only"
    assert all(value is False for value in result["authority"].values())
    assert result["draft_score_recipe"] == {
        "schema_version": "scryglass:private-r9e-development-composite:v1",
        "candidate_id": "R9E_d4_ss",
        "mode": "development_composite",
        "estimand": "composite_map_probability",
        "status": "unavailable",
        "authority": "unavailable",
        "reason": "r9e_binding_unavailable",
        "missing_inputs": ["recipe_binding"],
        "claim_ceiling": {
            "descriptive_draft_score": False,
            "composite_diagnostic": False,
            "public_probability": False,
            "recommendation": False,
            "betting": False,
        },
    }


def test_wrapper_reports_the_exact_r9e_query_contract(tmp_path: Path) -> None:
    binding = tmp_path / "binding.json"
    binding.write_text("{}", encoding="utf-8")
    args = _args(_fake_scorer(tmp_path), binding)

    result = wrapper.run(args)

    assert result["draft_score_recipe"]["reason"] == "r9e_query_inputs_missing"
    assert result["draft_score_recipe"]["missing_inputs"] == [
        "as_of",
        "patch",
        "r9e_player_elo_p",
        "r9e_player_elo_sigma",
    ]


def test_wrapper_keeps_r9e_composite_out_of_draft_score(tmp_path: Path, monkeypatch) -> None:
    composite = {
        "mode": "development_composite",
        "status": "development_only",
        "authority": "unavailable",
    }
    monkeypatch.setattr(wrapper, "_build_recipe", lambda _args: composite)

    result = wrapper.run(_args(_fake_scorer(tmp_path), tmp_path / "binding.json"))

    assert result["development_composite"] is composite
    assert result["draft_score_recipe"]["reason"] == "r9e_checkpoint_is_composite_not_draft_score"
    assert result["draft_score_recipe"]["claim_ceiling"]["descriptive_draft_score"] is False
