from __future__ import annotations

from pathlib import Path

import pytest

from lol_kills.knowledge.quick_mechanics import QuickMechanicsEngine
from lol_kills.knowledge.quick_mechanics_fastpack import compile_fastpack


INDEX = Path(
    "data/lol/knowledge/patch-packets/cdragon/2026/26.15/mechanics-index.json"
)


@pytest.fixture(scope="module")
def engine() -> QuickMechanicsEngine:
    if not INDEX.exists():
        pytest.skip("the exact 26.15 CommunityDragon packet is not available")
    return QuickMechanicsEngine(compile_fastpack(INDEX))


@pytest.mark.parametrize(
    ("question", "status", "value", "unit"),
    [
        ("what is malphite's mp5 @ lvl 13", "available", 13.32, "mana per 5 seconds"),
        ("malphjite base MR at lvl 13", "available", 50.45, "magic resist"),
        ("zaahen's mp5 at lvl 13", "available", 16.36, "mana per 5 seconds"),
        ("renektons mp5 at lvl 7", "not_applicable", None, "mana per 5 seconds"),
        ("total gold of grubs value", "available", 90, "gold"),
        (
            "how many auto attacks a tristana lvl 5 needs to deal to a gromp "
            "to kill it with no items?",
            "available",
            58,
            "auto attacks",
        ),
    ],
)
def test_conversation_queries_are_single_pass_ready(
    engine: QuickMechanicsEngine,
    question: str,
    status: str,
    value: float | None,
    unit: str,
) -> None:
    answer = engine.answer(question)
    assert answer["status"] == status
    assert answer["unit"] == unit
    if value is None:
        assert answer["value"] is None
    else:
        assert answer["value"] == pytest.approx(value, abs=0.005)
    assert answer["patch"] == "26.15"
    assert answer["display"]


def test_implicit_monster_level_is_disclosed(engine: QuickMechanicsEngine) -> None:
    answer = engine.answer(
        "how many auto attacks a tristana lvl 5 needs to deal to a gromp "
        "to kill it with no items?"
    )
    assumptions = " ".join(answer["assumptions"]).lower()
    assert "gromp" in assumptions
    assert "level 5" in assumptions

