from __future__ import annotations

from pathlib import Path

from lol_kills.knowledge.lol_oracle import LeagueOracleEngine
from lol_kills.knowledge.quick_mechanics_fastpack import compile_fastpack


INDEX = Path("data/lol/knowledge/patch-packets/cdragon/2026/26.15/mechanics-index.json")


def _engine() -> LeagueOracleEngine:
    return LeagueOracleEngine(
        compile_fastpack(INDEX),
        raw_champion_root=INDEX.parent / "raw" / "champions",
    )


def test_stat_delta_and_max_mana_are_exact() -> None:
    engine = _engine()
    delta = engine.answer("How much attack damage does Malphite gain from level 1 to level 6?")
    assert delta["status"] == "available"
    assert delta["value"] == 15.8
    assert delta["intent"] == "stat_delta"
    assert len(delta["sources"]) >= 3

    maximum = engine.answer("What is Malphite's maximum mana at level 6?")
    assert maximum["status"] == "available"
    assert maximum["value"] == 517
    assert maximum["unit"] == "mana"


def test_two_champion_stat_delta_does_not_collapse_to_the_second_champion() -> None:
    result = _engine().answer(
        "What is the delta between the base AD of Gnar and Darius at level 14?"
    )
    assert result["status"] == "available"
    assert result["intent"] == "champion_stat_comparison"
    assert result["value"] == 25.76
    assert result["components"] == {"Gnar": 98.69, "Darius": 124.45}
    assert len(result["sources"]) >= 5


def test_resource_window_resolves_token_boundaries() -> None:
    result = _engine().answer(
        "Ignoring spending, how much mana does Kalista regenerate in 10 seconds at level 3?"
    )
    assert result["status"] == "available"
    assert result["value"] == 14.96
    assert "Kalista level 3" in result["assumptions"]


def test_spell_cast_budget_uses_rank_one_indexed_costs() -> None:
    result = _engine().answer(
        "At level 6, how many rank-3 casts of Malphite's Q can be made from full resource with no regeneration?"
    )
    assert result["status"] == "available"
    assert result["value"] == 6
    assert "80.00 rank-3 cost" in result["calculation"]
    assert any(item["kind"] == "client" for item in result["sources"])


def test_spell_cast_budget_accepts_player_q_level_wording() -> None:
    result = _engine().answer(
        "How many Qs can Malphite use considering mana cost and Q lvl 3 at level 6?"
    )
    assert result["status"] == "available"
    assert result["value"] == 6
    assert "80.00 rank-3 cost" in result["calculation"]


def test_spell_cast_budget_composes_static_item_and_explicit_manaflow_stacks() -> None:
    result = _engine().answer(
        "On patch 26.15, Malphite is level 6 with rank-3 Seismic Shard, one "
        "Sapphire Crystal equipped, and Manaflow Band at 10 stacks. Starting "
        "at full mana with no regeneration, how many Q casts can he make?"
    )
    assert result["status"] == "available"
    assert result["value"] == 13
    assert result["remainder"] == 27
    assert result["resource_before"] == 1067
    assert result["modifier_components"] == {
        "base_max_resource": 517,
        "item_bonus": 300,
        "rune_bonus": 250,
    }
    assert result["modifier_packet"]["version"] == "modifier-packet-v1.0.0"
    assert result["modifier_packet"]["items"][0]["quantity"] == 1
    assert result["modifier_packet"]["runes"][0]["effective_stacks"] == 10
    assert "floor(1067.00 / 80.00) = 13 casts" in result["calculation"]
    assert any(item["kind"] == "client_item" for item in result["sources"])
    assert any(item.get("revision_id") == 3980907 for item in result["sources"])


def test_spell_cast_budget_accepts_explicit_unstacked_manaflow() -> None:
    result = _engine().answer(
        "At level 6, how many rank-3 casts of Malphite's Q can be made with "
        "Sapphire Crystal and Manaflow Band at 0 stacks from full mana?"
    )
    assert result["status"] == "available"
    assert result["value"] == 10
    assert result["remainder"] == 17
    assert "Manaflow Band 0 stated stacks" in result["calculation"]


def test_spell_cast_budget_never_infers_manaflow_stack_state() -> None:
    result = _engine().answer(
        "At level 6, how many rank-3 casts of Malphite's Q can be made with "
        "Sapphire Crystal and Manaflow Band from full mana?"
    )
    assert result["status"] == "unsupported"
    assert result["value"] is None
    assert "stack state is required" in result["reason"]


def test_spell_cast_budget_blocks_passive_item_combinations() -> None:
    result = _engine().answer(
        "At level 6, how many rank-3 casts of Malphite's Q can be made with "
        "Archangel's Staff and Manaflow Band at 10 stacks from full mana?"
    )
    assert result["status"] == "unsupported"
    assert result["value"] is None
    assert "passive or trigger" in result["reason"]


def test_direct_ap_damage_uses_patch_pinned_formula_graph() -> None:
    result = _engine().answer(
        "What exact damage does Malphite's rank-3 Q deal with 100 AP against a target with no modifiers?"
    )
    assert result["status"] == "available"
    assert result["value"] == 230
    assert result["intent"] == "direct_ability_damage"
    assert "QDamageCalc" in result["calculation"]
    assert any(item["kind"] == "wiki_ability" for item in result["sources"])


def test_static_item_stat_addition_is_exact_and_passives_block() -> None:
    engine = _engine()
    result = engine.answer(
        "With Long Sword equipped, what is Malphite's exact level-6 attack damage?"
    )
    assert result["status"] == "available"
    assert result["value"] == 87.8
    assert result["intent"] == "item_static_stat"
    assert any(item["kind"] == "client_item" for item in result["sources"])

    blocked = engine.answer(
        "With Rabadon's Deathcap equipped, what is Annie's exact level-6 ability power?"
    )
    assert blocked["status"] == "unsupported"
    assert blocked["value"] is None
    assert "passive" in blocked["reason"]


def test_magic_resistance_mitigation_is_explicit_and_source_linked() -> None:
    result = _engine().answer(
        "What exact post-mitigation damage does Malphite's rank-3 Q deal with 100 AP against a target with 50 magic resistance?"
    )
    assert result["status"] == "available"
    assert result["value"] == 153.33
    assert "100/(100+50)" in result["calculation"]
    assert any(item["url"].endswith("Magic_resistance") for item in result["sources"])


def test_physical_ad_damage_requires_explicit_level_and_armor() -> None:
    result = _engine().answer(
        "What exact post-mitigation damage does Aatrox's rank-1 Q deal with 100 total AD at level 6 against a target with 50 armor and no penetration?"
    )
    assert result["status"] == "available"
    assert result["value"] == 46.67
    assert "physical-resistance multiplier" in result["calculation"]


def test_unvalidated_damage_stays_blocked_but_linked() -> None:
    result = _engine().answer(
        "What exact damage does Malphite's rank-3 Q deal with 100 AP against a target with 50 armor?"
    )
    assert result["status"] == "unsupported"
    assert result["value"] is None
    assert result["sources"]


def test_revision_receipted_manaflow_and_turret_rules() -> None:
    engine = _engine()
    mana = engine.answer(
        "With Manaflow Band at 10 stacks, what is Malphite's exact maximum mana at level 6?"
    )
    assert mana["status"] == "available"
    assert mana["value"] == 767
    assert mana["intent"] == "rune_static_stack"
    assert any(item.get("revision_id") == 3980907 for item in mana["sources"])

    turret = engine.answer("At 5:30, what attack damage does the outer turret have?")
    assert turret["status"] == "available"
    assert turret["value"] == 254
    assert turret["intent"] == "structure_static_rule"
    assert any(item.get("revision_id") == 4019795 for item in turret["sources"])

    plates = engine.answer("How much local gold do 3 turret plates grant?")
    assert plates["status"] == "available"
    assert plates["value"] == 360


def test_permanent_stack_formulas_are_narrow_and_exact() -> None:
    engine = _engine()
    nasus = engine.answer(
        "After 120 Q stacks, what bonus physical damage does Nasus's rank-3 Siphoning Strike deal?"
    )
    assert nasus["status"] == "available"
    assert nasus["value"] == 200

    thresh = engine.answer("After 40 Thresh soul stacks, how much ability power does Thresh gain?")
    assert thresh["status"] == "available"
    assert thresh["value"] == 40

    kindred = engine.answer("After 10 Kindred marks, how much bonus attack range does Kindred gain?")
    assert kindred["status"] == "available"
    assert kindred["value"] == 125

    touch = engine.answer(
        "After 2 Touch of the Void stacks, what total true damage does a ranged attack burn a structure for over 4 seconds?"
    )
    assert touch["status"] == "available"
    assert touch["value"] == 48


def test_ordered_damage_sequence_returns_trace_receipt() -> None:
    result = _engine().answer(
        "A target starts with 1000 health, 50 armor, and 50 magic resistance. "
        "In exact order, it takes 100 physical damage, then 200 magic damage, "
        "then 50 true damage. What remaining health does it have?"
    )
    assert result["status"] == "available"
    assert result["value"] == 750
    assert result["intent"] == "ordered_damage_sequence"
    assert result["provenance"]["trace_sha256"]
