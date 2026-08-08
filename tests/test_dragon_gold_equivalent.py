from __future__ import annotations

import pytest

from lol_kills.research.dragon_gold_equivalent import anchors, stack_value, state_from_snapshot, team_value


def test_ocean_uses_champion_native_hp5_as_bead_equivalent() -> None:
    snapshots = [
        {"name": "Jax", "level": 8, "max_health": 1664},
        {"name": "Jarvan IV", "level": 7, "max_health": 1608},
        {"name": "Annie", "level": 8, "max_health": 1269},
        {"name": "Yunara", "level": 7, "max_health": 1187},
        {"name": "Lulu", "level": 6, "max_health": 1093},
    ]

    result = team_value("ocean", snapshots, missing_health_fraction=0.5)

    assert result["complete"] is True
    assert result["priced_gold_equivalent"] == pytest.approx(2215.85206, abs=0.00001)
    assert result["champions"][0]["components"][0]["bead_equivalents"] == pytest.approx(1.425115, abs=0.000001)


def test_cloud_keeps_slow_resistance_explicitly_unpriced() -> None:
    state = {
        "name": "Jax",
        "level": 8,
        "base_move_speed": 350,
    }

    result = stack_value("cloud", state_from_snapshot(state))

    assert result["complete"] is False
    assert result["components"][0]["gold_equivalent"] == pytest.approx(210.0)
    assert result["components"][1]["status"] == "unpriced"


def test_patch_item_anchors_are_read_from_cdragon() -> None:
    result = anchors()

    assert result["base_health_regen_percent"].cost == 300
    assert result["base_health_regen_percent"].amount == 100
    assert result["attack_speed_percent"].cost == 250
    assert result["ability_haste"].cost == 250
