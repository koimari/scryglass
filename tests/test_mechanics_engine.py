from __future__ import annotations

import pytest

from lol_kills.knowledge.mechanics_engine import (
    ApplyCC,
    Combatant,
    Damage,
    Event,
    GameState,
    MechanicsEngine,
    Move,
    ParityReceipt,
)


def _state() -> GameState:
    return GameState(
        entities={
            "blue-top": Combatant(
                entity_id="blue-top",
                team_id="blue",
                champion_id="Aatrox",
                health=100.0,
                max_health=100.0,
                stats={"armor": 100.0, "tenacity": 0.25},
            ),
            "red-top": Combatant(
                entity_id="red-top",
                team_id="red",
                champion_id="Gnar",
                health=100.0,
                max_health=100.0,
                stats={"armor": 100.0, "tenacity": 0.25},
            ),
        }
    )


def test_physical_damage_and_trace_are_deterministic() -> None:
    event = Event(
        at_ms=1000,
        priority=0,
        source_id="blue-top",
        ordinal=0,
        effect=Damage("blue-top", "red-top", 100.0, "physical"),
    )
    engine = MechanicsEngine()
    first = engine.run(_state(), [event])
    second = engine.run(_state(), [event])
    assert first.available
    assert first.state.entities["red-top"].health == pytest.approx(50.0)
    assert first.trace.trace_sha256 == second.trace.trace_sha256
    assert first.state.state_sha256 == second.state.state_sha256


def test_crowd_control_applies_tenacity() -> None:
    event = Event(
        at_ms=1000,
        priority=0,
        source_id="blue-top",
        ordinal=0,
        effect=ApplyCC("blue-top", "red-top", "stun", 1000),
    )
    result = MechanicsEngine().run(_state(), [event])
    assert result.available
    assert result.state.entities["red-top"].crowd_control["stun"] == 1750


def test_unsupported_collision_rule_is_unknown_and_does_not_change_state() -> None:
    state = _state()
    event = Event(
        at_ms=1000,
        priority=0,
        source_id="blue-top",
        ordinal=0,
        effect=Move("blue-top", "red-top", 10.0, 0.0, collision_rule="wall_check"),
    )
    result = MechanicsEngine().run(state, [event])
    assert not result.available
    assert result.state.entities["red-top"].position == (0.0, 0.0)
    assert result.unknowns[0].code == "effect_evaluation_failed"


def test_parity_receipt_is_hash_bound_and_fails_on_mismatch() -> None:
    state_hash = _state().state_sha256
    receipt = ParityReceipt.compare(
        series_id="series",
        game_id="game",
        checkpoint_ms=1000,
        engine_state_sha256=state_hash,
        observed_state_sha256=state_hash,
        source_receipt_sha256="c" * 64,
    )
    assert receipt.passed
    assert receipt.to_mapping()["passed"] is True
    mismatch = ParityReceipt.compare(
        series_id="series",
        game_id="game",
        checkpoint_ms=1000,
        engine_state_sha256=state_hash,
        observed_state_sha256="d" * 64,
        source_receipt_sha256="c" * 64,
    )
    assert not mismatch.passed
    assert mismatch.status == "mismatch"
    assert "state_hash_mismatch" in mismatch.blockers
