#!/usr/bin/env python3
"""Patch-pinned Dragon Slayer stat-to-gold formulas.

This module values only the part of a drake buff that has a defensible shop
anchor.  It is intentionally separate from outcome studies: a gold-equivalent
is a stat conversion, not an observed gold reward or a win-probability effect.

Champion input is a snapshot, for example::

    {
      "name": "Jax", "level": 8, "max_health": 1664,
      "base_health_regen_per_5": 11.67625, "bonus_armor": 30,
      "bonus_magic_resist": 0, "attack_damage": 120, "ability_power": 0,
      "attack_speed_ratio": 0.638, "base_move_speed": 350
    }

The item anchors are read from the patch packet rather than copied into a
second source of truth.  Effects without a clean shop-stat equivalent (for
example slow resistance) remain explicitly unpriced.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from lol_kills.knowledge.quick_mechanics_fastpack import compile_fastpack, normalize_alias


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATCH = "26.15"
DEFAULT_INDEX = ROOT / "data" / "lol" / "knowledge" / "patch-packets" / "cdragon" / "2026" / DEFAULT_PATCH / "mechanics-index.json"

DRAGON_STACKS = (1, 2, 3, 4)

# Current Summoner's Rift Dragon Slayer values (one stack).  These are kept in
# one table so a patch update is a single, reviewable change rather than a
# collection of unexplained constants in the calculation branches.
DRAGON_EFFECTS = {
    "infernal": {"attack_damage_rate": 0.03, "ability_power_rate": 0.03},
    "mountain": {"bonus_armor_rate": 0.05, "bonus_magic_resist_rate": 0.05},
    "ocean": {"missing_health_rate_per_5": 0.02},
    "cloud": {"ooc_move_speed_rate": 0.05, "slow_resist_rate": 0.05},
    "hextech": {"ability_haste": 5.0, "bonus_attack_speed_rate": 0.05},
    # Chemtech was raised from 5% to 6% per stack in V13.1b and remains 6%.
    "chemtech": {"tenacity_rate": 0.06, "heal_shield_power_rate": 0.06},
}


@dataclass(frozen=True)
class Anchor:
    stat: str
    item_id: int
    item_name: str
    cost: float
    amount: float
    gold_per_unit: float
    source: str


@dataclass(frozen=True)
class State:
    name: str
    level: int
    max_health: float | None = None
    base_health_regen_per_5: float | None = None
    attack_damage: float | None = None
    ability_power: float | None = None
    bonus_armor: float | None = None
    bonus_magic_resist: float | None = None
    attack_speed_ratio: float | None = None
    base_move_speed: float | None = None
    move_speed_before_buff: float | None = None


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _item_catalog(index_path: Path) -> dict[int, dict[str, Any]]:
    items_path = index_path.parent / "raw" / "items.json"
    rows = json.loads(items_path.read_text(encoding="utf-8"))
    return {int(row["id"]): row for row in rows if isinstance(row, Mapping) and row.get("id") is not None}


def _anchor(catalog: Mapping[int, Mapping[str, Any]], item_id: int, stat: str, amount: float, source: str) -> Anchor:
    row = catalog.get(item_id)
    if row is None:
        raise ValueError(f"item {item_id} is absent from the patch packet")
    cost = _number(row.get("priceTotal"))
    if cost is None or amount <= 0:
        raise ValueError(f"item {item_id} has no usable price/stat amount")
    return Anchor(
        stat=stat,
        item_id=item_id,
        item_name=str(row.get("name") or item_id),
        cost=cost,
        amount=amount,
        gold_per_unit=cost / amount,
        source=source,
    )


def anchors(index_path: Path = DEFAULT_INDEX) -> dict[str, Anchor]:
    """Return the patch-packet item anchors used by the formulas.

    Tenacity and heal/shield power are residual anchors from finished items:
    Mercury's Treads minus its MR and movement speed, and Forbidden Idol minus
    its Faerie Charm-equivalent mana regeneration.
    """
    catalog = _item_catalog(index_path)
    out = {
        "attack_damage": _anchor(catalog, 1036, "attack damage", 10, "Long Sword"),
        "ability_power": _anchor(catalog, 1052, "ability power", 20, "Amplifying Tome"),
        "armor": _anchor(catalog, 1029, "armor", 15, "Cloth Armor"),
        "magic_resist": _anchor(catalog, 1033, "magic resist", 20, "Null-Magic Mantle"),
        "attack_speed_percent": _anchor(catalog, 1042, "bonus attack speed percent", 10, "Dagger"),
        "ability_haste": _anchor(catalog, 2022, "ability haste", 5, "Glowing Mote"),
        "base_health_regen_percent": _anchor(catalog, 1006, "base health regen percent", 100, "Rejuvenation Bead"),
        "move_speed": _anchor(catalog, 1001, "flat movement speed", 25, "Boots"),
    }
    treads = catalog[3111]
    idol = catalog[3114]
    faerie = catalog[1004]
    treads_residual = _number(treads["priceTotal"]) - out["magic_resist"].gold_per_unit * 20 - out["move_speed"].gold_per_unit * 45
    idol_residual = _number(idol["priceTotal"]) - _number(faerie["priceTotal"])
    out["tenacity_percent"] = Anchor("tenacity percent", 3111, str(treads["name"]), treads_residual, 30, treads_residual / 30, "Mercury's Treads residual")
    out["heal_shield_power_percent"] = Anchor("heal/shield power percent", 3114, str(idol["name"]), idol_residual, 8, idol_residual / 8, "Forbidden Idol residual")
    return out


def state_from_snapshot(snapshot: Mapping[str, Any], *, index_path: Path = DEFAULT_INDEX) -> State:
    """Normalize a GRID/fastpack-compatible champion snapshot."""
    name = str(snapshot.get("name") or snapshot.get("champion") or "Unknown")
    level = int(snapshot.get("level") or 1)
    return State(
        name=name,
        level=level,
        max_health=_number(snapshot.get("max_health")),
        base_health_regen_per_5=_number(snapshot.get("base_health_regen_per_5")),
        attack_damage=_number(snapshot.get("attack_damage")),
        ability_power=_number(snapshot.get("ability_power")),
        bonus_armor=_number(snapshot.get("bonus_armor")),
        bonus_magic_resist=_number(snapshot.get("bonus_magic_resist")),
        attack_speed_ratio=_number(snapshot.get("attack_speed_ratio")),
        base_move_speed=_number(snapshot.get("base_move_speed")),
        move_speed_before_buff=_number(snapshot.get("move_speed_before_buff", snapshot.get("current_move_speed", snapshot.get("base_move_speed")))),
    )


def enrich_base_stats(snapshot: Mapping[str, Any], *, index_path: Path = DEFAULT_INDEX) -> State:
    """Fill max health, base HP5, and movement/AS anchors from the patch fastpack.

    Current item/rune-derived bonus stats still belong in the caller's
    snapshot; the patch packet supplies only champion base scaling.
    """
    state = state_from_snapshot(snapshot, index_path=index_path)
    pack = compile_fastpack(index_path)
    champion_id = pack["aliases"].get(normalize_alias(state.name))
    if champion_id is None:
        raise ValueError(f"champion {state.name!r} is not in the patch packet")
    champion = pack["champions"][str(champion_id)]
    level_row = champion["levels"][str(state.level)]
    base = champion["base_stats"]
    return State(
        name=state.name,
        level=state.level,
        max_health=state.max_health if state.max_health is not None else _number(level_row.get("max_health")),
        base_health_regen_per_5=state.base_health_regen_per_5 if state.base_health_regen_per_5 is not None else _number(level_row.get("health_regen_per_5")),
        attack_damage=state.attack_damage,
        ability_power=state.ability_power,
        bonus_armor=state.bonus_armor,
        bonus_magic_resist=state.bonus_magic_resist,
        attack_speed_ratio=state.attack_speed_ratio if state.attack_speed_ratio is not None else _number(base.get("attack_speed_ratio")),
        base_move_speed=state.base_move_speed if state.base_move_speed is not None else _number(level_row.get("move_speed") or base.get("base_move_speed")),
        move_speed_before_buff=state.move_speed_before_buff if state.move_speed_before_buff is not None else state.base_move_speed if state.base_move_speed is not None else _number(level_row.get("move_speed") or base.get("base_move_speed")),
    )


def _priced(stat: str, amount: float, anchor: Anchor, *, details: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "stat": stat,
        "amount": round(amount, 6),
        "status": "priced",
        "gold_equivalent": round(amount * anchor.gold_per_unit, 6),
        "anchor": {
            "item": anchor.item_name,
            "item_id": anchor.item_id,
            "cost": anchor.cost,
            "item_amount": anchor.amount,
            "gold_per_unit": round(anchor.gold_per_unit, 6),
            "source": anchor.source,
        },
        **(dict(details or {})),
    }


def _unpriced(stat: str, amount: float, reason: str) -> dict[str, Any]:
    return {"stat": stat, "amount": round(amount, 6), "status": "unpriced", "gold_equivalent": None, "reason": reason}


def stack_value(dragon: str, state: State, stacks: int = 1, *, missing_health_fraction: float = 0.5, duration_seconds: float | None = None, index_path: Path = DEFAULT_INDEX) -> dict[str, Any]:
    """Value one champion's Dragon Slayer stack(s), with component provenance."""
    if stacks not in DRAGON_STACKS:
        raise ValueError("stacks must be in 1..4")
    dragon = dragon.casefold().replace("water", "ocean").replace("fire", "infernal").replace("earth", "mountain").replace("air", "cloud")
    a = anchors(index_path)
    components: list[dict[str, Any]] = []
    if dragon == "infernal":
        if state.attack_damage is None or state.ability_power is None:
            raise ValueError("Infernal requires attack_damage and ability_power")
        components += [_priced("attack damage", state.attack_damage * DRAGON_EFFECTS[dragon]["attack_damage_rate"] * stacks, a["attack_damage"]), _priced("ability power", state.ability_power * DRAGON_EFFECTS[dragon]["ability_power_rate"] * stacks, a["ability_power"])]
    elif dragon == "mountain":
        if state.bonus_armor is None or state.bonus_magic_resist is None:
            raise ValueError("Mountain requires bonus_armor and bonus_magic_resist")
        components += [_priced("bonus armor", state.bonus_armor * DRAGON_EFFECTS[dragon]["bonus_armor_rate"] * stacks, a["armor"]), _priced("bonus magic resist", state.bonus_magic_resist * DRAGON_EFFECTS[dragon]["bonus_magic_resist_rate"] * stacks, a["magic_resist"])]
    elif dragon == "hextech":
        if state.attack_speed_ratio is None:
            raise ValueError("Hextech requires attack_speed_ratio")
        components += [_priced("ability haste", DRAGON_EFFECTS[dragon]["ability_haste"] * stacks, a["ability_haste"]), _priced("bonus attack speed percent", state.attack_speed_ratio * DRAGON_EFFECTS[dragon]["bonus_attack_speed_rate"] * stacks * 100, a["attack_speed_percent"])]
    elif dragon == "cloud":
        if state.move_speed_before_buff is None:
            raise ValueError("Cloud requires move_speed_before_buff (or base_move_speed)")
        components += [_priced("out-of-combat flat movement speed equivalent", state.move_speed_before_buff * DRAGON_EFFECTS[dragon]["ooc_move_speed_rate"] * stacks, a["move_speed"], details={"conversion_note": "5% OOC movement speed converted using the champion's effective movement speed before Cloud"}), _unpriced("slow resistance percent", DRAGON_EFFECTS[dragon]["slow_resist_rate"] * 100 * stacks, "No clean standalone shop-stat anchor")]
    elif dragon == "chemtech":
        components += [_priced("tenacity percent", DRAGON_EFFECTS[dragon]["tenacity_rate"] * 100 * stacks, a["tenacity_percent"]), _priced("heal/shield power percent", DRAGON_EFFECTS[dragon]["heal_shield_power_rate"] * 100 * stacks, a["heal_shield_power_percent"])]
    elif dragon == "ocean":
        if state.max_health is None or state.base_health_regen_per_5 is None:
            raise ValueError("Ocean requires max_health and base_health_regen_per_5")
        if not 0 <= missing_health_fraction <= 1:
            raise ValueError("missing_health_fraction must be between 0 and 1")
        heal_per_5 = state.max_health * missing_health_fraction * DRAGON_EFFECTS[dragon]["missing_health_rate_per_5"] * stacks
        ticks = 1 if duration_seconds is None else 1 + int(max(0, duration_seconds) // 5)
        bead_heal = state.base_health_regen_per_5 * ticks
        # Rejuvenation Bead prices a *rate* (100% base HP regen), so Ocean's
        # HP cannot be multiplied by the bead's percent gold/unit directly.
        # Instead, compare Ocean's healing over the window with this champion's
        # native HP5 over the same window, then multiply by the bead's 300g.
        bead_equivalents = (heal_per_5 * ticks) / bead_heal if bead_heal > 0 else 0.0
        components.append({
            "stat": "Ocean healing",
            "amount": round(heal_per_5 * ticks, 6),
            "status": "priced",
            "gold_equivalent": round(bead_equivalents * a["base_health_regen_percent"].cost, 6),
            "anchor": {
                "item": a["base_health_regen_percent"].item_name,
                "item_id": a["base_health_regen_percent"].item_id,
                "cost": a["base_health_regen_percent"].cost,
                "item_amount": a["base_health_regen_percent"].amount,
                "gold_per_unit": round(a["base_health_regen_percent"].gold_per_unit, 6),
                "source": a["base_health_regen_percent"].source,
            },
            "healing_per_5": round(heal_per_5, 6),
            "base_hp5_over_window": round(bead_heal, 6),
            "bead_equivalents": round(bead_equivalents, 6),
            "ticks_including_initial_activation": ticks,
            "missing_health_fraction": missing_health_fraction,
            "conversion_note": "Ocean healing / native HP5 over the same window, valued at one Rejuvenation Bead per 100% base HP regen",
        })
    else:
        raise ValueError(f"unsupported dragon {dragon!r}")
    total = sum(c["gold_equivalent"] for c in components if c["gold_equivalent"] is not None)
    return {
        "champion": state.name,
        "level": state.level,
        "dragon": dragon,
        "stacks": stacks,
        "effect_per_stack": DRAGON_EFFECTS[dragon],
        "components": components,
        "priced_gold_equivalent": round(total, 6),
        "complete": all(c["status"] == "priced" for c in components),
    }


def team_value(dragon: str, snapshots: Sequence[Mapping[str, Any] | State], stacks: int = 1, **kwargs: Any) -> dict[str, Any]:
    states = [s if isinstance(s, State) else enrich_base_stats(s, index_path=kwargs.get("index_path", DEFAULT_INDEX)) for s in snapshots]
    rows = [stack_value(dragon, state, stacks, **kwargs) for state in states]
    return {"dragon": dragon, "stacks": stacks, "champions": rows, "priced_gold_equivalent": round(sum(row["priced_gold_equivalent"] for row in rows), 6), "complete": all(row["complete"] for row in rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dragon", choices=("infernal", "mountain", "ocean", "cloud", "hextech", "chemtech"))
    parser.add_argument("states_json", type=Path, help="JSON array of champion snapshot objects")
    parser.add_argument("--stacks", type=int, default=1, choices=DRAGON_STACKS)
    parser.add_argument("--missing-health", type=float, default=0.5)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    args = parser.parse_args()
    snapshots = json.loads(args.states_json.read_text(encoding="utf-8"))
    print(json.dumps(team_value(args.dragon, snapshots, args.stacks, missing_health_fraction=args.missing_health, duration_seconds=args.duration, index_path=args.index), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
