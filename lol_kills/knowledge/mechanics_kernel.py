"""Small, fail-closed evaluator for CommunityDragon calculation graphs.

This is intentionally a narrow kernel, not a game emulator.  It evaluates
the calculation primitives used by the first mechanics fixtures and raises
on an unsupported primitive instead of silently producing a plausible number.
Stat-code meanings are supplied by the caller because the client enum must be
resolved and tested separately from the arithmetic.  CommunityDragon omits
``mStat`` for the common ability-power coefficient, so the caller-visible
default is the client AP code (6); callers can disable that default by passing
``default_stat_code=None``.
"""

from __future__ import annotations

import math
from functools import reduce
from operator import mul
from typing import Any, Mapping


class UnsupportedFormulaError(ValueError):
    """The raw client formula contains a primitive not implemented here."""


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UnsupportedFormulaError(f"{label} is not numeric")
    out = float(value)
    if not math.isfinite(out):
        raise UnsupportedFormulaError(f"{label} is not finite")
    return out


def _subparts(value: Mapping[str, Any], label: str) -> list[Mapping[str, Any]]:
    raw = value.get("mSubparts")
    if isinstance(raw, list):
        items = raw
    else:
        # Some current client calculations encode a two-term product as
        # mPart1/mPart2 rather than the older mSubparts list.
        items = [value.get("mPart1"), value.get("mPart2")]
        if any(item is None for item in items):
            raise UnsupportedFormulaError(f"{label}.mSubparts is missing")
    return [
        item
        if isinstance(item, Mapping)
        else (_ for _ in ()).throw(UnsupportedFormulaError(f"{label} has a non-object subpart"))
        for item in items
    ]


def _interpolate(level: int, start: float, end: float) -> float:
    if type(level) is not int or not 1 <= level <= 18:
        raise ValueError("character level must be an integer in [1, 18]")
    return start + (end - start) * ((level - 1) / 17.0)


def _spell_value(data_values: Mapping[str, list[Any]], name: str, rank: int) -> float:
    if type(rank) is not int or rank < 1:
        raise ValueError("ability rank must be a positive integer")
    values = data_values.get(name)
    if values is None:
        raise UnsupportedFormulaError(f"unknown client data value: {name}")
    # Client vectors conventionally reserve index zero and use rank as the
    # index.  Do not clamp a missing rank: unsupported data must stay visible.
    if rank >= len(values):
        raise UnsupportedFormulaError(f"rank {rank} is absent from data value {name}")
    return _number(values[rank], f"data value {name}[{rank}]")


def evaluate_part(
    part: Mapping[str, Any],
    *,
    data_values: Mapping[str, list[Any]],
    ability_rank: int,
    character_level: int,
    stat_codes: Mapping[int, float],
    default_stat_code: int | None = 6,
) -> float:
    """Evaluate one raw ``mFormulaParts`` object."""

    kind = str(part.get("__type", ""))
    if kind == "NumberCalculationPart":
        return _number(part.get("mNumber"), "mNumber")
    if kind == "NamedDataValueCalculationPart":
        return _spell_value(data_values, str(part.get("mDataValue")), ability_rank)
    if kind == "StatByCoefficientCalculationPart":
        raw_code = part.get("mStat", default_stat_code)
        if raw_code is None:
            raise UnsupportedFormulaError("stat coefficient omits its stat code")
        code = int(raw_code)
        if code not in stat_codes:
            raise UnsupportedFormulaError(f"unresolved stat code: {code}")
        return stat_codes[code] * _number(part.get("mCoefficient"), "mCoefficient")
    if kind == "StatByNamedDataValueCalculationPart":
        raw_code = part.get("mStat", default_stat_code)
        if raw_code is None:
            raise UnsupportedFormulaError("named stat coefficient omits its stat code")
        code = int(raw_code)
        if code not in stat_codes:
            raise UnsupportedFormulaError(f"unresolved stat code: {code}")
        if "mStatFormula" in part:
            raise UnsupportedFormulaError("mStatFormula is not implemented in the narrow kernel")
        return stat_codes[code] * _spell_value(
            data_values, str(part.get("mDataValue")), ability_rank
        )
    if kind == "ByCharLevelInterpolationCalculationPart":
        return _interpolate(
            character_level,
            _number(part.get("mStartValue"), "mStartValue"),
            _number(part.get("mEndValue"), "mEndValue"),
        )
    if kind == "ByCharLevelBreakpointsCalculationPart":
        value = _number(part.get("mLevel1Value"), "mLevel1Value")
        for breakpoint in part.get("mBreakpoints", []):
            if not isinstance(breakpoint, Mapping):
                raise UnsupportedFormulaError("breakpoint is not an object")
            if character_level >= int(breakpoint["mLevel"]):
                value += _number(
                    breakpoint.get("mAdditionalBonusAtThisLevel"),
                    "mAdditionalBonusAtThisLevel",
                )
        return value
    if kind == "SumOfSubPartsCalculationPart":
        return sum(
            evaluate_part(
                item,
                data_values=data_values,
                ability_rank=ability_rank,
                character_level=character_level,
                stat_codes=stat_codes,
                default_stat_code=default_stat_code,
            )
            for item in _subparts(part, kind)
        )
    if kind == "ProductOfSubPartsCalculationPart":
        return reduce(
            mul,
            (
                evaluate_part(
                    item,
                    data_values=data_values,
                    ability_rank=ability_rank,
                    character_level=character_level,
                    stat_codes=stat_codes,
                    default_stat_code=default_stat_code,
                )
                for item in _subparts(part, kind)
            ),
            1.0,
        )
    raise UnsupportedFormulaError(f"unsupported calculation part: {kind or '<missing type>'}")


def evaluate_calculation(
    calculation: Mapping[str, Any],
    *,
    data_values: Mapping[str, list[Any]],
    ability_rank: int,
    character_level: int,
    stat_codes: Mapping[int, float],
    calculations: Mapping[str, Mapping[str, Any]] | None = None,
    default_stat_code: int | None = 6,
) -> float:
    """Evaluate a raw ``GameCalculation`` or a modified calculation."""

    kind = str(calculation.get("__type", "GameCalculation"))
    if kind == "GameCalculation":
        parts = calculation.get("mFormulaParts", [])
        if not isinstance(parts, list):
            raise UnsupportedFormulaError("GameCalculation.mFormulaParts is not a list")
        value = sum(
            evaluate_part(
                part,
                data_values=data_values,
                ability_rank=ability_rank,
                character_level=character_level,
                stat_codes=stat_codes,
                default_stat_code=default_stat_code,
            )
            for part in parts
            if isinstance(part, Mapping)
        )
        if "mMultiplier" in calculation:
            multiplier = calculation["mMultiplier"]
            if isinstance(multiplier, Mapping):
                value *= evaluate_part(
                    multiplier,
                    data_values=data_values,
                    ability_rank=ability_rank,
                    character_level=character_level,
                    stat_codes=stat_codes,
                    default_stat_code=default_stat_code,
                )
            else:
                value *= _number(multiplier, "mMultiplier")
        return value
    if kind == "GameCalculationModified":
        base_key = calculation.get("mModifiedGameCalculation")
        if not isinstance(base_key, str) or calculations is None:
            raise UnsupportedFormulaError("modified calculation has no resolved base calculation")
        base = calculations.get(base_key)
        if not isinstance(base, Mapping):
            raise UnsupportedFormulaError(f"unknown modified calculation base: {base_key}")
        value = evaluate_calculation(
            base,
            data_values=data_values,
            ability_rank=ability_rank,
            character_level=character_level,
            stat_codes=stat_codes,
            calculations=calculations,
            default_stat_code=default_stat_code,
        )
        multiplier = calculation.get("mMultiplier")
        if isinstance(multiplier, Mapping):
            value *= evaluate_part(
                multiplier,
                data_values=data_values,
                ability_rank=ability_rank,
                character_level=character_level,
                stat_codes=stat_codes,
                default_stat_code=default_stat_code,
            )
        elif multiplier is not None:
            value *= _number(multiplier, "mMultiplier")
        return value
    raise UnsupportedFormulaError(f"unsupported calculation type: {kind}")


def evaluate_spell_calculation(
    spell: Mapping[str, Any],
    calculation_name: str,
    *,
    ability_rank: int,
    character_level: int,
    stat_codes: Mapping[int, float],
    default_stat_code: int | None = 6,
) -> float:
    """Evaluate one named calculation from a normalized spell record."""

    data_values = {
        str(value["name"]): list(value["values"])
        for value in spell.get("data_values", [])
        if isinstance(value, Mapping)
        and isinstance(value.get("name"), str)
        and isinstance(value.get("values"), list)
    }
    calculations = spell.get("spell_calculations", {})
    if not isinstance(calculations, Mapping):
        raise UnsupportedFormulaError("spell calculations are not an object")
    calculation = calculations.get(calculation_name)
    if not isinstance(calculation, Mapping):
        raise UnsupportedFormulaError(f"unknown spell calculation: {calculation_name}")
    return evaluate_calculation(
        calculation,
        data_values=data_values,
        ability_rank=ability_rank,
        character_level=character_level,
        stat_codes=stat_codes,
        calculations=calculations,
        default_stat_code=default_stat_code,
    )
