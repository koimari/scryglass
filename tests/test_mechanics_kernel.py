from __future__ import annotations

import json
from pathlib import Path

import pytest

from lol_kills.knowledge.mechanics_kernel import (
    UnsupportedFormulaError,
    evaluate_spell_calculation,
)


def _aatrox() -> dict:
    path = Path("data/lol/knowledge/patch-packets/cdragon/16.8/mechanics-index.json")
    if not path.exists():
        pytest.skip("Aatrox CommunityDragon fixture has not been captured")
    payload = json.loads(path.read_text())
    champion = next(
        champion for champion in payload["champions"] if champion["name"] == "Aatrox"
    )
    return champion["mechanics"]


def test_aatrox_q_and_w_use_patch_pinned_formula_graph() -> None:
    mechanics = _aatrox()
    q = next(spell for spell in mechanics["spells"] if spell["script_name"] == "AatroxQ")
    w = next(spell for spell in mechanics["spells"] if spell["script_name"] == "AatroxW")
    assert evaluate_spell_calculation(
        q, "QDamage", ability_rank=1, character_level=1, stat_codes={2: 100.0}
    ) == pytest.approx(70.0)
    assert evaluate_spell_calculation(
        w, "WDamage", ability_rank=5, character_level=9, stat_codes={2: 100.0}
    ) == pytest.approx(110.0)


def test_aatrox_q_edge_is_a_modified_calculation() -> None:
    mechanics = _aatrox()
    q = next(spell for spell in mechanics["spells"] if spell["script_name"] == "AatroxQ")
    assert evaluate_spell_calculation(
        q, "QEdgeDamage", ability_rank=1, character_level=1, stat_codes={2: 100.0}
    ) == pytest.approx(119.0)


def test_unknown_stat_code_fails_closed() -> None:
    mechanics = _aatrox()
    q = next(spell for spell in mechanics["spells"] if spell["script_name"] == "AatroxQ")
    with pytest.raises(UnsupportedFormulaError):
        evaluate_spell_calculation(
            q, "QDamage", ability_rank=1, character_level=1, stat_codes={}
        )
