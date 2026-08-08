from __future__ import annotations

from pathlib import Path

from lol_kills.knowledge.lol_oracle import LeagueOracleEngine
from lol_kills.knowledge.quick_mechanics_fastpack import compile_fastpack
from lol_kills.knowledge.semantic_engine import SemanticOracleEngine


INDEX = Path("data/lol/knowledge/patch-packets/cdragon/2026/26.15/mechanics-index.json")


def _engine() -> SemanticOracleEngine:
    base = LeagueOracleEngine(
        compile_fastpack(INDEX),
        raw_champion_root=INDEX.parent / "raw" / "champions",
    )
    return SemanticOracleEngine(base)


def test_missing_build_state_is_a_typed_contract_not_a_guess() -> None:
    result = _engine().answer("How much damage does Aatrox deal with the current build?")

    assert result["status"] == "needs_input"
    assert result["intent"] == "build_damage"
    assert result["value"] is None
    required = {item["path"] for item in result["required_inputs"]}
    assert {"patch", "mode", "attacker.level", "attacker.items", "attacker.runes", "event_state"} <= required
    assert result["provenance"]["request_sha256"]
    assert len(result["sources"]) >= 2


def test_ambiguous_fight_and_counterfactual_state_stay_explicit() -> None:
    fight = _engine().answer("Does Ahri win the fight if the enemy is stronger?")
    assert fight["status"] == "needs_input"
    assert fight["intent"] == "fight_outcome"
    assert {item["path"] for item in fight["required_inputs"]} >= {
        "opponent.champion",
        "initial_state",
        "events",
        "win_condition",
    }

    counterfactual = _engine().answer(
        "What exact damage would Akali have dealt in the next 10 seconds if the opponent had dodged?"
    )
    assert counterfactual["status"] == "needs_input"
    assert counterfactual["intent"] == "counterfactual"
    assert {item["path"] for item in counterfactual["required_inputs"]} >= {
        "initial_state",
        "events",
        "counterfactual",
        "counterfactual.target_id",
    }


def test_nonexistent_patch_and_cross_mode_rule_are_invalid() -> None:
    result = _engine().answer(
        "On Summoner's Rift patch 99.99, what is the exact Arena-only augment multiplier for Akshan?"
    )

    assert result["status"] == "invalid_scenario"
    assert {item["code"] for item in result["validation"]} >= {
        "patch_not_available",
        "mode_mismatch",
    }
    assert result["value"] is None


def test_complete_direct_damage_request_executes_the_exact_packet() -> None:
    result = _engine().answer(
        "calculate this",
        {
            "intent": "direct_ability_damage",
            "patch": "26.15",
            "mode": "summoners_rift",
            "damage_mode": "post_mitigation",
            "damage_type": "magic",
            "penetration": {},
            "attacker": {
                "champion": "Malphite",
                "level": 6,
                "ability": {"key": "Q", "rank": 3},
                "stats": {"ability_power": 100},
            },
            "target": {"health": 1000, "magic_resist": 50},
        },
    )

    assert result["status"] == "available"
    assert result["value"] == 153.33
    assert result["semantic_intent"] == "direct_ability_damage"
    assert result["provenance"]["semantic_request_sha256"]
    assert any(item["kind"] == "client" for item in result["sources"])


def test_nonempty_build_effects_are_not_silently_ignored() -> None:
    result = _engine().answer(
        "calculate this",
        {
            "intent": "build_damage",
            "patch": "26.15",
            "mode": "summoners_rift",
            "damage_mode": "raw",
            "attacker": {
                "champion": "Malphite",
                "level": 6,
                "ability": {"key": "Q", "rank": 3},
                "stats": {"ability_power": 100},
                "items": ["Rabadon's Deathcap"],
                "runes": [],
                "buffs": {},
                "debuffs": {},
            },
            "event_state": {},
        },
    )

    assert result["status"] == "unsupported"
    assert result["value"] is None
    assert "attacker.items" in result["reason"]


def test_unknown_champion_identity_is_invalid_not_a_default_match() -> None:
    result = _engine().answer(
        "calculate this",
        {
            "intent": "direct_ability_damage",
            "patch": "26.15",
            "mode": "summoners_rift",
            "damage_mode": "raw",
            "attacker": {
                "champion": "NotAChampion",
                "level": 6,
                "ability": {"key": "Q", "rank": 3},
                "stats": {"ability_power": 100},
            },
        },
    )

    assert result["status"] == "invalid_scenario"
    assert any(item["code"] == "unknown_champion" for item in result["validation"])


def test_complete_fight_request_returns_a_deterministic_winner_and_trace() -> None:
    result = _engine().answer(
        "fight",
        {
            "intent": "fight_outcome",
            "patch": "26.15",
            "mode": "summoners_rift",
            "attacker": {"champion": "Malphite", "entity_id": "attacker"},
            "opponent": {"champion": "Darius", "entity_id": "opponent"},
            "win_condition": "first_death",
            "initial_state": {
                "entities": {
                    "attacker": {
                        "champion": "Malphite",
                        "health": 1000,
                        "max_health": 1000,
                        "stats": {"armor": 0, "magic_resist": 0},
                    },
                    "opponent": {
                        "champion": "Darius",
                        "health": 1000,
                        "max_health": 1000,
                        "stats": {"armor": 0, "magic_resist": 0},
                    },
                }
            },
            "events": [
                {
                    "event_id": "hit-1",
                    "at_ms": 100,
                    "source_id": "attacker",
                    "target_id": "opponent",
                    "amount": 1000,
                    "damage_type": "true",
                }
            ],
        },
    )

    assert result["status"] == "available"
    assert result["value"] == "attacker"
    assert result["provenance"]["trace_sha256"]
    assert result["provenance"]["final_state_sha256"]


def test_complete_counterfactual_removes_only_the_named_event() -> None:
    result = _engine().answer(
        "counterfactual",
        {
            "intent": "counterfactual",
            "patch": "26.15",
            "mode": "summoners_rift",
            "attacker": {"champion": "Malphite", "entity_id": "attacker"},
            "initial_state": {
                "entities": {
                    "attacker": {"champion": "Malphite", "health": 1000, "max_health": 1000},
                    "target": {
                        "champion": "Darius",
                        "health": 1000,
                        "max_health": 1000,
                        "stats": {"armor": 0, "magic_resist": 0},
                    },
                }
            },
            "events": [
                {
                    "event_id": "hit-1",
                    "at_ms": 100,
                    "source_id": "attacker",
                    "target_id": "target",
                    "amount": 100,
                    "damage_type": "true",
                },
                {
                    "event_id": "hit-2",
                    "at_ms": 200,
                    "source_id": "attacker",
                    "target_id": "target",
                    "amount": 200,
                    "damage_type": "true",
                },
            ],
            "counterfactual": {"remove_event_id": "hit-2", "target_id": "target"},
        },
    )

    assert result["status"] == "available"
    assert result["value"] == {
        "baseline_remaining_health": 700.0,
        "counterfactual_remaining_health": 900.0,
        "damage_avoided": 200.0,
        "target_id": "target",
    }


def test_base_oracle_exposes_the_semantic_layer_without_changing_answer_contract() -> None:
    base = LeagueOracleEngine(
        compile_fastpack(INDEX),
        raw_champion_root=INDEX.parent / "raw" / "champions",
    )
    result = base.semantic_answer("How much damage does Aatrox deal with the current build?")
    assert result["status"] == "needs_input"
    assert result["intent"] == "build_damage"
