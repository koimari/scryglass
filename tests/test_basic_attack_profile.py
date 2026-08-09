from __future__ import annotations

from pathlib import Path

import pytest

from lol_kills.knowledge.basic_attack_profile import compare_jax_basic_attacks
from lol_kills.knowledge.lol_oracle import LeagueOracleEngine
from lol_kills.knowledge.quick_mechanics_fastpack import compile_fastpack


INDEX = Path("data/lol/knowledge/patch-packets/cdragon/2026/26.15/mechanics-index.json")


def _engine() -> LeagueOracleEngine:
    return LeagueOracleEngine(
        compile_fastpack(INDEX),
        raw_champion_root=INDEX.parent / "raw" / "champions",
    )


def test_jax_static_attack_comparison_is_patch_pinned_and_boundary_explicit() -> None:
    result = compare_jax_basic_attacks(
        _engine(),
        level=14,
        item_names=("Trinity Force", "Sundered Sky"),
        adaptive_shards=1,
        seconds=10,
    )
    assert result["passive"]["per_stack_percent"] == pytest.approx(11)
    assert result["passive"]["max_stacks"] == 8

    zero = result["profiles"]["zero_passive_stacks"]
    itemless = zero["without_items"]
    items = zero["with_items"]
    assert itemless["raw_physical_damage_per_basic_attack"] == pytest.approx(
        124.7825
    )
    assert items["raw_physical_damage_per_basic_attack"] == pytest.approx(205.7825)
    assert itemless["attacks_per_second"] == pytest.approx(0.90025628)
    assert items["attacks_per_second"] == pytest.approx(1.09165628)
    assert itemless["raw_physical_dps"] == pytest.approx(112.3362293)
    assert items["raw_physical_dps"] == pytest.approx(224.6437584)
    assert itemless["discrete_attack_events"] == {
        "first_attack_at_time_zero": 10,
        "first_attack_after_one_full_interval": 9,
    }
    assert items["discrete_attack_events"] == {
        "first_attack_at_time_zero": 11,
        "first_attack_after_one_full_interval": 10,
    }
    assert zero["with_items_minus_without_items"][
        "continuous_attack_intervals_in_window"
    ] == pytest.approx(1.9140000343)
    assert zero["with_items_minus_without_items"][
        "discrete_first_attack_at_time_zero"
    ] == 1

    full = result["profiles"]["full_passive_stacks"]
    assert full["without_items"]["attacks_per_second"] == pytest.approx(1.46169629)
    assert full["with_items"]["attacks_per_second"] == pytest.approx(1.65309629)
    assert full["with_items_minus_without_items"][
        "discrete_first_attack_at_time_zero"
    ] == 2
    ramp = result["ideal_uninterrupted_ramp_from_zero"]["conventions"]
    assert ramp["first_attack_at_time_zero"]["without_items"]["attacks"] == 13
    assert ramp["first_attack_at_time_zero"]["with_items"]["attacks"] == 16
    assert ramp["first_attack_at_time_zero"]["additional_attacks_with_items"] == 3
    assert ramp["first_attack_after_one_full_interval"]["without_items"][
        "attacks"
    ] == 12
    assert ramp["first_attack_after_one_full_interval"]["with_items"][
        "attacks"
    ] == 14
    assert ramp["first_attack_after_one_full_interval"][
        "additional_attacks_with_items"
    ] == 2
    assert result["plain_language_guardrail"]


def test_attack_speed_ratio_is_retained_in_fastpack() -> None:
    pack = compile_fastpack(INDEX)
    jax = pack["champions"]["24"]
    assert jax["base_stats"]["attack_speed_ratio"] == pytest.approx(
        0.6380000114440918
    )
