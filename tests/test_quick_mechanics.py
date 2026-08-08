from __future__ import annotations

import pytest

from lol_kills.knowledge.quick_mechanics import QuickMechanicsEngine


def _levels(**stats):
    return {str(level): dict(stats) for level in range(1, 19)}


@pytest.fixture()
def engine() -> QuickMechanicsEngine:
    # This fixture is deliberately local.  It exercises the public fastpack
    # contract without requiring a CommunityDragon/compiler refresh.
    pack = {
        "patch": "26.15",
        "provenance": {"source": "test-fastpack", "fastpack_sha256": "f" * 64},
        "champions": {
            "malphite": {
                "name": "Malphite",
                "aliases": ["Malphite"],
                "resource_type": "mana",
                "level_tables": {
                    str(level): {
                        "mp5": 7.2,
                        "mr": 32.1,
                        "ad": 62.0 + level,
                    }
                    for level in range(1, 19)
                },
            },
            "zaahen": {
                "name": "Zaahen",
                "aliases": ["Zaahen"],
                "resource_type": "mana",
                "levels": _levels(mp5=8.0, mr=32.0, ad=65.0),
            },
            "renekton": {
                "name": "Renekton",
                "aliases": ["Renekton"],
                "resource_type": "none",
                "level_tables": {str(level): {"ad": 69.0} for level in range(1, 19)},
            },
            "tristana": {
                "name": "Tristana",
                "aliases": ["Tristana"],
                "resource_type": "mana",
                "levels": _levels(ad=70.0),
            },
        },
        "monsters": {
            "gromp": {
                "name": "Gromp",
                "aliases": ["Gromp"],
                "levels": {
                    str(level): {"hp": 2700.0, "armor": 50.0}
                    for level in range(1, 19)
                },
            }
        },
        "objectives": {
            "voidgrub_camp": {
                "name": "Voidgrub camp",
                "aliases": ["Voidgrub camp", "void grubs", "grubs"],
                "reward": {"gold": 90},
            }
        },
    }
    return QuickMechanicsEngine(pack)


def test_champion_stats_accept_level_abbreviations_and_typo(engine):
    mp5 = engine.answer("What is malphjite mp5 at lvl 13?")
    assert mp5["status"] == "available"
    assert mp5["value"] == pytest.approx(7.2)
    assert mp5["unit"] == "mana per 5 seconds"
    assert mp5["patch"] == "26.15"
    assert mp5["provenance"]["pack_sha256"] == "f" * 64

    mr = engine.answer("Malphite base magic resist at level 13")
    assert mr["status"] == "available"
    assert mr["value"] == pytest.approx(32.1)
    assert mr["unit"] == "magic resist"


def test_mana_resource_and_no_resource_answers(engine):
    result = engine.answer("What is Zaahen's mana regen MP5 at level 13?")
    assert result["status"] == "available"
    assert result["value"] == pytest.approx(8.0)

    result = engine.answer("What is Renekton MP5 at level 13?")
    assert result["status"] == "not_applicable"
    assert result["value"] is None
    assert "no mana" in result["assumptions"][0]


def test_full_voidgrub_camp_gold(engine):
    result = engine.answer("How much gold is a full Voidgrub camp?")
    assert result["status"] == "available"
    assert result["value"] == 90
    assert result["unit"] == "gold"
    assert result["display"] == "90g"


def test_itemless_tristana_gromp_is_58_attacks(engine):
    result = engine.answer(
        "How many basic attacks does Tristana lvl5 need to kill lvl5 Gromp?"
    )
    assert result["status"] == "available"
    assert result["value"] == 58
    assert result["unit"] == "auto attacks"
    assert result["display"] == "58 auto attacks"
    assert any("itemless" in item for item in result["assumptions"])


def test_omitted_target_level_is_explicitly_defaulted(engine):
    result = engine.answer("How many autos does Tristana lvl5 need to kill Gromp?")
    assert result["status"] == "available"
    assert result["value"] == 58
    assert any("defaulted to attacker level 5" in item for item in result["assumptions"])


def test_unsupported_mechanic_fails_closed(engine):
    result = engine.answer("How much damage does Tristana's explosive charge do?")
    assert result["status"] == "unsupported"
    assert result["value"] is None


def test_stat_delta_question_does_not_silently_use_first_level(engine):
    result = engine.answer("How much attack damage does Malphite gain from level 1 to level 6?")
    assert result["status"] == "unsupported"
    assert result["value"] is None
    assert "exactly one champion level" in result["reason"]
