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


def test_natural_language_jinx_inner_turret_query_infers_defaults_and_returns_variants() -> None:
    result = _engine().answer(
        "what are the most optimal build for a Jinx at 3 items to deal the most DPS in an inner turret?"
    )
    assert result["status"] == "available"
    assert result["intent"] == "turret_dps_optimization"
    assert result["patch"] == "26.15"
    assert result["defaults"]["mode"] == "summoners_rift"
    assert result["defaults"]["turret_health"] == 5000
    assert result["defaults"]["turret_armor"] == 60
    assert result["defaults"]["item_count"] == 3
    assert result["search"]["item_pool_count"] >= 80
    assert result["search"]["combos_evaluated"] > 100_000
    assert len(result["variants"]) == 6
    assert result["headline"]["q_form"] == "pow-pow"
    assert result["headline"]["passive_stacks"] == 0
    assert result["headline"]["dps"] > 0
    assert {"Hullbreaker", "Trinity Force"}.issubset(result["headline"]["build"])
    assert any("auto-attacks only" in item["name"] for item in result["variants"])
    assert any(item.get("revision_id") == 4019795 for item in result["sources"])
    assert any(item.get("revision_id") == 4015400 for item in result["sources"])

