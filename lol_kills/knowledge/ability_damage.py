"""Narrow, source-pinned direct ability-damage evaluation.

This module deliberately exposes only calculations whose client graph can be
evaluated with explicit AP (and, optionally, AD) inputs.  It does not infer
which tooltip component is the complete ability, apply mitigation, or model
passives, target health, item effects, or event order.  Ambiguous or richer
graphs raise ``UnsupportedFormulaError`` so the query layer can remain
fail-closed.
"""

from __future__ import annotations

from typing import Any, Mapping

from .mechanics_kernel import UnsupportedFormulaError, evaluate_spell_calculation


_DIRECT_CALC_NAMES = (
    "{key}DamageCalc",
    "{key}Damage",
    "TotalDamage",
    "CalculatedDamage",
    "DamageToDeal",
    "TooltipTotalDamage",
)


def _contains_level_dependency(value: Any) -> bool:
    if isinstance(value, Mapping):
        if str(value.get("__type", "")).startswith("ByCharLevel"):
            return True
        return any(_contains_level_dependency(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_level_dependency(item) for item in value)
    return False


def direct_damage_calculation_name(
    spell: Mapping[str, Any], *, ability_key: str
) -> str:
    """Return a conservative direct-damage calculation key.

    A raw spell often contains several calculations for shields, cooldowns,
    monster-only modifiers, or tooltip fragments.  Only the small allow-list
    above is treated as the primary direct-damage expression.  The key must
    exist exactly; no substring or first-match fallback is allowed.
    """

    calculations = spell.get("spell_calculations")
    if not isinstance(calculations, Mapping):
        raise UnsupportedFormulaError("spell calculations are not an object")
    for template in _DIRECT_CALC_NAMES:
        key = template.format(key=ability_key.upper())
        if key in calculations and isinstance(calculations[key], Mapping):
            return key
    raise UnsupportedFormulaError(
        "spell has no conservative primary direct-damage calculation"
    )


def evaluate_direct_damage(
    spell: Mapping[str, Any],
    *,
    ability_key: str,
    ability_rank: int,
    ability_power: float,
    attack_damage: float | None = None,
    character_level: int | None = None,
) -> tuple[str, float]:
    """Evaluate a direct raw-damage graph with explicit stat inputs.

    Client stat code 6 is ability power and code 2 is total attack damage.
    Leaving AD out is intentional: formulas that require it fail instead of
    silently assuming a level or base-stat value that the question did not
    provide.
    """

    if not isinstance(ability_key, str) or ability_key.upper() not in {"Q", "W", "E", "R"}:
        raise UnsupportedFormulaError("ability key must be Q, W, E, or R")
    if isinstance(ability_power, bool) or not isinstance(ability_power, (int, float)):
        raise UnsupportedFormulaError("ability power must be numeric")
    stat_codes: dict[int, float] = {6: float(ability_power)}
    if attack_damage is not None:
        if isinstance(attack_damage, bool) or not isinstance(attack_damage, (int, float)):
            raise UnsupportedFormulaError("attack damage must be numeric")
        stat_codes[2] = float(attack_damage)
    key = direct_damage_calculation_name(spell, ability_key=ability_key)
    calculation = spell["spell_calculations"][key]
    if character_level is None and _contains_level_dependency(calculation):
        raise UnsupportedFormulaError(
            "damage graph depends on champion level, but the question omits level"
        )
    value = evaluate_spell_calculation(
        spell,
        key,
        ability_rank=ability_rank,
        character_level=character_level or 1,
        stat_codes=stat_codes,
    )
    return key, value


__all__ = [
    "direct_damage_calculation_name",
    "evaluate_direct_damage",
]
