"""Deterministic, fail-closed state-transition primitives for League research.

This is the first engine boundary, not a claim that every League rule is
implemented.  A supported effect produces a new state and an auditable trace;
an unsupported or unavailable rule produces an explicit unknown and leaves the
state unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence, Union

from .patch_authority import PatchPacket


SCHEMA_VERSION = "scryglass:mechanics-engine:v1"
ENGINE_VERSION = "mechanics-kernel-v1.0.0"


class MechanicsEngineError(ValueError):
    """Base error for invalid or unavailable mechanics execution."""


class UnsupportedMechanicError(MechanicsEngineError):
    """A requested effect cannot be evaluated by the current kernel."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MechanicsEngineError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MechanicsEngineError(f"{field_name} must be finite")
    return result


@dataclass(frozen=True)
class UnknownReason:
    code: str
    message: str
    path: str | None = None

    def to_mapping(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "path": self.path}


@dataclass(frozen=True)
class Combatant:
    """Canonical player/champion state used by the transition kernel."""

    entity_id: str
    team_id: str
    champion_id: str
    level: int = 1
    experience: float = 0.0
    gold: float = 500.0
    health: float = 1.0
    max_health: float = 1.0
    resources: Mapping[str, float] = field(default_factory=dict)
    stats: Mapping[str, float] = field(default_factory=dict)
    cooldowns: Mapping[str, int] = field(default_factory=dict)
    items: tuple[str, ...] = ()
    runes: tuple[str, ...] = ()
    crowd_control: Mapping[str, int] = field(default_factory=dict)
    buffs: Mapping[str, float] = field(default_factory=dict)
    marks: Mapping[str, int] = field(default_factory=dict)
    position: tuple[float, float] = (0.0, 0.0)
    alive: bool = True
    transform: str | None = None

    def __post_init__(self) -> None:
        if not self.entity_id.strip() or not self.team_id.strip() or not self.champion_id.strip():
            raise MechanicsEngineError("combatant identity fields are required")
        if type(self.level) is not int or not 1 <= self.level <= 18:
            raise MechanicsEngineError("combatant level must be an integer in [1, 18]")
        if self.max_health < 0 or self.health < 0 or self.health > self.max_health:
            raise MechanicsEngineError("combatant health must be within [0, max_health]")
        if len(self.position) != 2:
            raise MechanicsEngineError("combatant position must have two coordinates")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "team_id": self.team_id,
            "champion_id": self.champion_id,
            "level": self.level,
            "experience": self.experience,
            "gold": self.gold,
            "health": self.health,
            "max_health": self.max_health,
            "resources": dict(sorted(self.resources.items())),
            "stats": dict(sorted(self.stats.items())),
            "cooldowns": dict(sorted(self.cooldowns.items())),
            "items": list(self.items),
            "runes": list(self.runes),
            "crowd_control": dict(sorted(self.crowd_control.items())),
            "buffs": dict(sorted(self.buffs.items())),
            "marks": dict(sorted(self.marks.items())),
            "position": list(self.position),
            "alive": self.alive,
            "transform": self.transform,
        }


@dataclass(frozen=True)
class GameState:
    """Full-state container; unsupported fields remain explicit in ``unknowns``."""

    clock_ms: int = 0
    entities: Mapping[str, Combatant] = field(default_factory=dict)
    zones: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    summons: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    objectives: Mapping[str, int] = field(default_factory=dict)
    vision: Mapping[str, Any] = field(default_factory=dict)
    rules: Mapping[str, Any] = field(default_factory=dict)
    unknowns: tuple[UnknownReason, ...] = ()

    def __post_init__(self) -> None:
        if type(self.clock_ms) is not int or self.clock_ms < 0:
            raise MechanicsEngineError("clock_ms must be a non-negative integer")
        if len(self.entities) != len(set(self.entities)):
            raise MechanicsEngineError("entity IDs must be unique")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "clock_ms": self.clock_ms,
            "entities": {
                key: self.entities[key].to_mapping() for key in sorted(self.entities)
            },
            "zones": {key: self.zones[key] for key in sorted(self.zones)},
            "summons": {key: self.summons[key] for key in sorted(self.summons)},
            "objectives": dict(sorted(self.objectives.items())),
            "vision": dict(sorted(self.vision.items())),
            "rules": dict(sorted(self.rules.items())),
            "unknowns": [item.to_mapping() for item in self.unknowns],
        }

    @property
    def state_sha256(self) -> str:
        return _sha(self.to_mapping())


@dataclass(frozen=True)
class Damage:
    source_id: str
    target_id: str
    amount: float
    damage_type: str
    penetration: Mapping[str, float] = field(default_factory=dict)
    target_filter: str | None = None


@dataclass(frozen=True)
class ApplyCC:
    source_id: str
    target_id: str
    kind: str
    duration_ms: int
    tenacity_applies: bool = True


@dataclass(frozen=True)
class ModifyStat:
    source_id: str
    target_id: str
    stat: str
    delta: float
    duration_ms: int | None = None
    stack_rule: str = "add"


@dataclass(frozen=True)
class Move:
    source_id: str
    target_id: str
    dx: float
    dy: float
    collision_rule: str = "none"


@dataclass(frozen=True)
class CreateZone:
    source_id: str
    zone_id: str
    center: tuple[float, float]
    radius: float
    duration_ms: int
    effects: tuple[str, ...] = ()


@dataclass(frozen=True)
class Mark:
    source_id: str
    target_id: str
    mark_id: str
    duration_ms: int
    consume_effect: str | None = None


@dataclass(frozen=True)
class Summon:
    source_id: str
    summon_id: str
    team_id: str
    kind: str
    duration_ms: int


@dataclass(frozen=True)
class Transform:
    source_id: str
    target_id: str
    transform_id: str
    reversible: bool = True


Effect = Union[Damage, ApplyCC, ModifyStat, Move, CreateZone, Mark, Summon, Transform]


def _effect_mapping(effect: Effect) -> dict[str, Any]:
    if isinstance(effect, Damage):
        return {"type": "Damage", "source_id": effect.source_id, "target_id": effect.target_id, "amount": effect.amount, "damage_type": effect.damage_type, "penetration": dict(sorted(effect.penetration.items())), "target_filter": effect.target_filter}
    if isinstance(effect, ApplyCC):
        return {"type": "ApplyCC", "source_id": effect.source_id, "target_id": effect.target_id, "kind": effect.kind, "duration_ms": effect.duration_ms, "tenacity_applies": effect.tenacity_applies}
    if isinstance(effect, ModifyStat):
        return {"type": "ModifyStat", "source_id": effect.source_id, "target_id": effect.target_id, "stat": effect.stat, "delta": effect.delta, "duration_ms": effect.duration_ms, "stack_rule": effect.stack_rule}
    if isinstance(effect, Move):
        return {"type": "Move", "source_id": effect.source_id, "target_id": effect.target_id, "dx": effect.dx, "dy": effect.dy, "collision_rule": effect.collision_rule}
    if isinstance(effect, CreateZone):
        return {"type": "CreateZone", "source_id": effect.source_id, "zone_id": effect.zone_id, "center": list(effect.center), "radius": effect.radius, "duration_ms": effect.duration_ms, "effects": list(effect.effects)}
    if isinstance(effect, Mark):
        return {"type": "Mark", "source_id": effect.source_id, "target_id": effect.target_id, "mark_id": effect.mark_id, "duration_ms": effect.duration_ms, "consume_effect": effect.consume_effect}
    if isinstance(effect, Summon):
        return {"type": "Summon", "source_id": effect.source_id, "summon_id": effect.summon_id, "team_id": effect.team_id, "kind": effect.kind, "duration_ms": effect.duration_ms}
    if isinstance(effect, Transform):
        return {"type": "Transform", "source_id": effect.source_id, "target_id": effect.target_id, "transform_id": effect.transform_id, "reversible": effect.reversible}
    raise MechanicsEngineError(f"unsupported effect object: {type(effect).__name__}")


@dataclass(frozen=True)
class Event:
    at_ms: int
    priority: int
    source_id: str
    ordinal: int
    effect: Effect
    event_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.at_ms) is not int or self.at_ms < 0:
            raise MechanicsEngineError("event at_ms must be a non-negative integer")
        if type(self.priority) is not int or type(self.ordinal) is not int:
            raise MechanicsEngineError("event priority and ordinal must be integers")
        if not self.source_id.strip():
            raise MechanicsEngineError("event source_id is required")

    @property
    def stable_id(self) -> str:
        return self.event_id or _sha({"at_ms": self.at_ms, "priority": self.priority, "source_id": self.source_id, "ordinal": self.ordinal, "effect": _effect_mapping(self.effect)})

    @property
    def sort_key(self) -> tuple[int, int, str, int, str]:
        return (self.at_ms, self.priority, self.source_id, self.ordinal, self.stable_id)


@dataclass(frozen=True)
class Transition:
    event: Event
    state: GameState
    status: str
    applied: Mapping[str, Any]
    unknowns: tuple[UnknownReason, ...] = ()


@dataclass(frozen=True)
class ExecutionTrace:
    transitions: tuple[Mapping[str, Any], ...]
    final_state_sha256: str

    @property
    def trace_sha256(self) -> str:
        return _sha({"transitions": list(self.transitions), "final_state_sha256": self.final_state_sha256})

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "engine_version": ENGINE_VERSION,
            "transitions": list(self.transitions),
            "final_state_sha256": self.final_state_sha256,
            "trace_sha256": self.trace_sha256,
        }


@dataclass(frozen=True)
class ExecutionResult:
    state: GameState
    trace: ExecutionTrace
    unknowns: tuple[UnknownReason, ...]

    @property
    def available(self) -> bool:
        return not self.unknowns


@dataclass(frozen=True)
class ParityReceipt:
    """Hash-bound comparison between an engine checkpoint and observed state."""

    series_id: str
    game_id: str
    checkpoint_ms: int
    engine_state_sha256: str
    observed_state_sha256: str
    source_receipt_sha256: str
    status: str
    blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.series_id.strip() or not self.game_id.strip():
            raise MechanicsEngineError("parity identity fields are required")
        if type(self.checkpoint_ms) is not int or self.checkpoint_ms < 0:
            raise MechanicsEngineError("parity checkpoint_ms must be a non-negative integer")
        for value, name in (
            (self.engine_state_sha256, "engine_state_sha256"),
            (self.observed_state_sha256, "observed_state_sha256"),
            (self.source_receipt_sha256, "source_receipt_sha256"),
        ):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise MechanicsEngineError(f"{name} must be a lowercase SHA-256")
        if self.status not in {"passed", "blocked", "mismatch"}:
            raise MechanicsEngineError(f"unsupported parity status: {self.status}")
        if self.status == "passed" and self.blockers:
            raise MechanicsEngineError("passed parity receipt cannot have blockers")

    @property
    def passed(self) -> bool:
        return self.status == "passed" and not self.blockers

    @classmethod
    def compare(
        cls,
        *,
        series_id: str,
        game_id: str,
        checkpoint_ms: int,
        engine_state_sha256: str,
        observed_state_sha256: str,
        source_receipt_sha256: str,
        blockers: Sequence[str] = (),
    ) -> "ParityReceipt":
        reasons = list(blockers)
        if engine_state_sha256 != observed_state_sha256:
            reasons.append("state_hash_mismatch")
        status = "passed" if not reasons else ("mismatch" if "state_hash_mismatch" in reasons else "blocked")
        return cls(
            series_id=series_id,
            game_id=game_id,
            checkpoint_ms=checkpoint_ms,
            engine_state_sha256=engine_state_sha256,
            observed_state_sha256=observed_state_sha256,
            source_receipt_sha256=source_receipt_sha256,
            status=status,
            blockers=tuple(sorted(set(reasons))),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "series_id": self.series_id,
            "game_id": self.game_id,
            "checkpoint_ms": self.checkpoint_ms,
            "engine_state_sha256": self.engine_state_sha256,
            "observed_state_sha256": self.observed_state_sha256,
            "source_receipt_sha256": self.source_receipt_sha256,
            "status": self.status,
            "blockers": list(self.blockers),
            "passed": self.passed,
        }


def _entity(state: GameState, entity_id: str) -> Combatant | None:
    return state.entities.get(entity_id)


def _replace_entity(state: GameState, updated: Combatant) -> GameState:
    entities = dict(state.entities)
    entities[updated.entity_id] = updated
    return replace(state, entities=entities)


def _resistance_multiplier(resistance: float) -> float:
    if resistance >= 0:
        return 100.0 / (100.0 + resistance)
    return 2.0 - 100.0 / (100.0 - resistance)


def _effective_resistance(target: Combatant, damage: Damage) -> float:
    stat = "armor" if damage.damage_type == "physical" else "magic_resist"
    resistance = float(target.stats.get(stat, 0.0))
    penetration = damage.penetration
    percent = float(penetration.get("percent", 0.0))
    flat = float(penetration.get("flat", 0.0))
    if percent < 0 or percent > 1:
        raise UnsupportedMechanicError("penetration.percent must be in [0, 1]")
    return resistance * (1.0 - percent) - flat


class MechanicsEngine:
    """Pure deterministic event executor with explicit unknown propagation."""

    def __init__(self, *, rules: Mapping[str, Any] | None = None):
        self.rules = dict(rules or {})

    @staticmethod
    def require_executable_cells(packet: PatchPacket, keys: Sequence[str]) -> None:
        blockers: list[str] = []
        for key in keys:
            cell = packet.cell(key)
            if cell is None:
                blockers.append(f"missing:{key}")
            elif not cell.executable:
                blockers.append(f"{cell.status}:{key}")
        if blockers:
            raise UnsupportedMechanicError("required patch cells are unavailable: " + ",".join(blockers))

    def apply(self, state: GameState, event: Event) -> Transition:
        if event.at_ms < state.clock_ms:
            unknown = UnknownReason("event_time_regression", "event time precedes current state clock", event.stable_id)
            return Transition(event, state, "unknown", {}, (unknown,))
        state = replace(state, clock_ms=event.at_ms)
        effect = event.effect
        try:
            if isinstance(effect, Damage):
                return self._damage(state, event, effect)
            if isinstance(effect, ApplyCC):
                return self._cc(state, event, effect)
            if isinstance(effect, ModifyStat):
                return self._modify_stat(state, event, effect)
            if isinstance(effect, Move):
                return self._move(state, event, effect)
            if isinstance(effect, CreateZone):
                return self._zone(state, event, effect)
            if isinstance(effect, Mark):
                return self._mark(state, event, effect)
            if isinstance(effect, Summon):
                return self._summon(state, event, effect)
            if isinstance(effect, Transform):
                return self._transform(state, event, effect)
        except (MechanicsEngineError, KeyError, TypeError, ValueError) as exc:
            unknown = UnknownReason("effect_evaluation_failed", str(exc), event.stable_id)
            return Transition(event, state, "unknown", {}, (unknown,))
        unknown = UnknownReason("unsupported_effect", type(effect).__name__, event.stable_id)
        return Transition(event, state, "unknown", {}, (unknown,))

    def _target_or_unknown(self, state: GameState, event: Event, target_id: str) -> Combatant | Transition:
        target = _entity(state, target_id)
        if target is None:
            return Transition(event, state, "unknown", {}, (UnknownReason("target_missing", f"target {target_id} is absent", target_id),))
        return target

    def _damage(self, state: GameState, event: Event, effect: Damage) -> Transition:
        target = self._target_or_unknown(state, event, effect.target_id)
        if isinstance(target, Transition):
            return target
        amount = _finite_number(effect.amount, "damage.amount")
        if amount < 0:
            raise MechanicsEngineError("damage.amount cannot be negative")
        if effect.damage_type == "true":
            multiplier = 1.0
        elif effect.damage_type in {"physical", "magic"}:
            multiplier = _resistance_multiplier(_effective_resistance(target, effect))
        else:
            raise UnsupportedMechanicError(f"unknown damage type: {effect.damage_type}")
        dealt = amount * multiplier
        health = max(0.0, target.health - dealt)
        updated = replace(target, health=health, alive=health > 0.0)
        next_state = _replace_entity(state, updated)
        applied = {"type": "Damage", "target_id": effect.target_id, "raw_amount": amount, "post_mitigation": dealt, "multiplier": multiplier, "remaining_health": health}
        return Transition(event, next_state, "applied", applied)

    def _cc(self, state: GameState, event: Event, effect: ApplyCC) -> Transition:
        target = self._target_or_unknown(state, event, effect.target_id)
        if isinstance(target, Transition):
            return target
        if type(effect.duration_ms) is not int or effect.duration_ms < 0:
            raise MechanicsEngineError("crowd-control duration must be non-negative integer")
        tenacity = float(target.stats.get("tenacity", 0.0)) if effect.tenacity_applies else 0.0
        if not 0.0 <= tenacity <= 1.0:
            raise UnsupportedMechanicError("tenacity must be in [0, 1]")
        duration = int(round(effect.duration_ms * (1.0 - tenacity)))
        ends_at = state.clock_ms + duration
        cc = dict(target.crowd_control)
        cc[effect.kind] = max(int(cc.get(effect.kind, 0)), ends_at)
        next_state = _replace_entity(state, replace(target, crowd_control=cc))
        return Transition(event, next_state, "applied", {"type": "ApplyCC", "kind": effect.kind, "duration_ms": duration, "ends_at_ms": ends_at})

    def _modify_stat(self, state: GameState, event: Event, effect: ModifyStat) -> Transition:
        target = self._target_or_unknown(state, event, effect.target_id)
        if isinstance(target, Transition):
            return target
        if effect.duration_ms is not None and effect.duration_ms < 0:
            raise MechanicsEngineError("stat modifier duration cannot be negative")
        if effect.stack_rule not in {"add", "replace"}:
            raise UnsupportedMechanicError(f"unsupported stat stack rule: {effect.stack_rule}")
        stats = dict(target.stats)
        delta = _finite_number(effect.delta, "stat delta")
        stats[effect.stat] = stats.get(effect.stat, 0.0) + delta if effect.stack_rule == "add" else delta
        next_state = _replace_entity(state, replace(target, stats=stats))
        return Transition(event, next_state, "applied", {"type": "ModifyStat", "stat": effect.stat, "value": stats[effect.stat], "duration_ms": effect.duration_ms})

    def _move(self, state: GameState, event: Event, effect: Move) -> Transition:
        target = self._target_or_unknown(state, event, effect.target_id)
        if isinstance(target, Transition):
            return target
        if effect.collision_rule != "none":
            raise UnsupportedMechanicError(f"unsupported collision rule: {effect.collision_rule}")
        position = (target.position[0] + _finite_number(effect.dx, "move.dx"), target.position[1] + _finite_number(effect.dy, "move.dy"))
        next_state = _replace_entity(state, replace(target, position=position))
        return Transition(event, next_state, "applied", {"type": "Move", "target_id": effect.target_id, "position": list(position)})

    def _zone(self, state: GameState, event: Event, effect: CreateZone) -> Transition:
        if effect.zone_id in state.zones:
            raise UnsupportedMechanicError(f"zone already exists: {effect.zone_id}")
        if effect.radius < 0 or effect.duration_ms < 0:
            raise MechanicsEngineError("zone radius and duration must be non-negative")
        zones = dict(state.zones)
        zones[effect.zone_id] = {"owner_id": effect.source_id, "center": list(effect.center), "radius": effect.radius, "ends_at_ms": state.clock_ms + effect.duration_ms, "effects": list(effect.effects)}
        return Transition(event, replace(state, zones=zones), "applied", {"type": "CreateZone", "zone_id": effect.zone_id})

    def _mark(self, state: GameState, event: Event, effect: Mark) -> Transition:
        target = self._target_or_unknown(state, event, effect.target_id)
        if isinstance(target, Transition):
            return target
        if effect.duration_ms < 0:
            raise MechanicsEngineError("mark duration cannot be negative")
        marks = dict(target.marks)
        marks[effect.mark_id] = state.clock_ms + effect.duration_ms
        next_state = _replace_entity(state, replace(target, marks=marks))
        return Transition(event, next_state, "applied", {"type": "Mark", "target_id": effect.target_id, "mark_id": effect.mark_id, "ends_at_ms": marks[effect.mark_id], "consume_effect": effect.consume_effect})

    def _summon(self, state: GameState, event: Event, effect: Summon) -> Transition:
        if effect.summon_id in state.summons:
            raise UnsupportedMechanicError(f"summon already exists: {effect.summon_id}")
        if effect.duration_ms < 0:
            raise MechanicsEngineError("summon duration cannot be negative")
        summons = dict(state.summons)
        summons[effect.summon_id] = {"owner_id": effect.source_id, "team_id": effect.team_id, "kind": effect.kind, "ends_at_ms": state.clock_ms + effect.duration_ms}
        return Transition(event, replace(state, summons=summons), "applied", {"type": "Summon", "summon_id": effect.summon_id})

    def _transform(self, state: GameState, event: Event, effect: Transform) -> Transition:
        target = self._target_or_unknown(state, event, effect.target_id)
        if isinstance(target, Transition):
            return target
        next_state = _replace_entity(state, replace(target, transform=effect.transform_id))
        return Transition(event, next_state, "applied", {"type": "Transform", "target_id": effect.target_id, "transform_id": effect.transform_id, "reversible": effect.reversible})

    def run(self, initial: GameState, events: Sequence[Event], *, max_events: int = 100_000) -> ExecutionResult:
        if max_events < 1:
            raise MechanicsEngineError("max_events must be positive")
        ordered = sorted(events, key=lambda event: event.sort_key)
        seen: set[str] = set()
        state = initial
        transitions: list[dict[str, Any]] = []
        unknowns: list[UnknownReason] = []
        for index, event in enumerate(ordered):
            if index >= max_events:
                unknowns.append(UnknownReason("event_limit", f"event limit {max_events} reached", None))
                break
            if event.stable_id in seen:
                unknowns.append(UnknownReason("duplicate_event", f"duplicate event {event.stable_id}", event.stable_id))
                continue
            seen.add(event.stable_id)
            transition = self.apply(state, event)
            state = transition.state
            unknowns.extend(transition.unknowns)
            transitions.append({"event_id": event.stable_id, "at_ms": event.at_ms, "priority": event.priority, "source_id": event.source_id, "effect": _effect_mapping(event.effect), "status": transition.status, "applied": dict(transition.applied), "unknowns": [item.to_mapping() for item in transition.unknowns], "state_sha256": state.state_sha256})
        trace = ExecutionTrace(tuple(transitions), state.state_sha256)
        return ExecutionResult(state, trace, tuple(unknowns))


__all__ = [
    "ApplyCC",
    "Combatant",
    "CreateZone",
    "Damage",
    "Effect",
    "Event",
    "ExecutionResult",
    "ExecutionTrace",
    "GameState",
    "Mark",
    "MechanicsEngine",
    "MechanicsEngineError",
    "ModifyStat",
    "Move",
    "ParityReceipt",
    "Summon",
    "Transform",
    "Transition",
    "UnknownReason",
    "UnsupportedMechanicError",
]
