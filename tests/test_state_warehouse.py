from __future__ import annotations

from pathlib import Path

import pytest

from lol_kills.knowledge.lol_oracle import LeagueOracleEngine
from lol_kills.knowledge.quick_mechanics_fastpack import compile_fastpack
from lol_kills.knowledge.state_warehouse import StateWarehouse, WarehouseQueryError
from tools.lol_mechanics_mcp.server import LeagueMechanicsServer


INDEX = Path("data/lol/knowledge/patch-packets/cdragon/2026/26.15/mechanics-index.json")


def _warehouse() -> StateWarehouse:
    pack = compile_fastpack(INDEX)
    oracle = LeagueOracleEngine(pack, raw_champion_root=INDEX.parent / "raw" / "champions")
    return StateWarehouse(pack, oracle=oracle, index_path=INDEX)


def test_materializes_patch_star_schema_and_bounded_sql_cache() -> None:
    warehouse = _warehouse()
    try:
        status = warehouse.status()
        assert status["resident"] is True
        assert status["snapshot"]["patch"] == "26.15"
        assert status["table_counts"]["dim_champion"] >= 200
        query = "SELECT c.name, s.attack_damage FROM dim_champion c JOIN fact_champion_stat s USING (champion_key) WHERE c.name = 'Gnar' AND s.level = 14"
        first = warehouse.query_sql(query)
        second = warehouse.query_sql(query)
        assert first["rows"][0]["name"] == "Gnar"
        assert first["cache_hit"] is False
        assert second["cache_hit"] is True
        assert warehouse.status()["cache_hits"] == 1
    finally:
        warehouse.close()


def test_sql_is_read_only_and_row_bounded() -> None:
    warehouse = _warehouse()
    try:
        with pytest.raises(WarehouseQueryError):
            warehouse.query_sql("DELETE FROM dim_champion")
        result = warehouse.query_sql("SELECT name FROM dim_champion ORDER BY champion_id", max_rows=3)
        assert result["row_count"] == 3
        assert result["truncated"] is True
    finally:
        warehouse.close()


def test_structured_state_composes_static_components_and_blocks_effects() -> None:
    warehouse = _warehouse()
    try:
        available = warehouse.state_query(
            {"champion": "Malphite", "level": 6, "items": ["Sapphire Crystal"], "runes": []}
        )
        assert available["status"] == "available"
        assert available["derived_stats"]["max_resource"] == 817.0

        blocked = warehouse.state_query(
            {"champion": "Malphite", "level": 6, "items": ["Nashor's Tooth"], "runes": []}
        )
        assert blocked["status"] == "unsupported"
        assert blocked["unavailable"]
    finally:
        warehouse.close()


def test_structured_state_can_join_an_ability_without_claiming_execution() -> None:
    warehouse = _warehouse()
    try:
        result = warehouse.state_query(
            {
                "champion": "Malphite",
                "level": 6,
                "ability": {"key": "Q", "rank": 3},
                "items": [],
                "runes": [],
            }
        )
        assert result["status"] == "available"
        assert result["ability"]["name"] == "Seismic Shard"
        assert result["ability"]["cost"] == 80.0
        assert result["ability"]["execution_status"] == "not_yet_implemented"
    finally:
        warehouse.close()


def test_mcp_canonical_tool_exposes_sql_and_natural_state_fallback() -> None:
    server = LeagueMechanicsServer(index_path=INDEX)
    try:
        sql = server.answer(
            {
                "operation": "sql",
                "sql": "SELECT name, level FROM dim_champion JOIN fact_champion_stat USING (champion_key) WHERE name = 'Gnar' AND level = 14",
            }
        )
        assert sql["route"] == "warehouse_sql"
        assert sql["rows"] == [{"name": "Gnar", "level": 14}]

        natural = server.answer({"question": "What are Malphite's stats at level 6 with Sapphire Crystal?"})
        assert natural["route"] == "warehouse_state"
        assert natural["status"] == "available"
        assert natural["derived_stats"]["max_resource"] == 817.0
    finally:
        server.close()
