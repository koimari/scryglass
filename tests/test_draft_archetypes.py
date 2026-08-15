from __future__ import annotations

import pytest

from lol_kills.draft_archetypes import champ_tags


@pytest.mark.parametrize(
    ("champion", "expected"),
    (
        ("Yuumi", {"peel_enchanter", "scaling_late"}),
        ("Kai'Sa", {"hypercarry_adc", "skirmisher"}),
        ("Ezreal", {"poke_siege", "skirmisher"}),
        ("Nocturne", {"assassin", "engage"}),
    ),
)
def test_duplicate_archetype_entries_have_one_stable_tag_set(
    champion: str, expected: set[str]
) -> None:
    assert champ_tags(champion) == expected
