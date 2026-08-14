"""Patch-pinned static basic-attack and attack-speed profiles.

This calculator answers the narrow question "what does one ordinary auto do,
and how many autos fit in a fixed window?"  It deliberately excludes crits,
on-hits, empowered attacks, target mitigation, animation cancels, and item
passives.  Attack count is reported under two explicit timer conventions so a
boundary-sensitive phrase such as "one auto in ten seconds" cannot hide the
assumption that produced it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .quick_mechanics_fastpack import level_growth_multiplier
from ..v2.patch_identity import CURRENT_PUBLIC_PATCH


ADAPTIVE_SHARD_ATTACK_DAMAGE = 5.4
DEFAULT_ATTACK_SPEED_CAP = 2.5
SCHEMA_VERSION = "scryglass:basic-attack-profile:v1"

ATTACK_SPEED_SOURCE = {
    "kind": "wiki_rule",
    "url": "https://wiki.leagueoflegends.com/en-us/Attack_speed",
    "label": "League Wiki attack speed",
    "page_id": 2950,
    "revision_id": 4035691,
    "revision_timestamp": "2026-06-25T03:17:54Z",
}
ADAPTIVE_SHARD_SOURCE = {
    "kind": "wiki_rule",
    "url": "https://wiki.leagueoflegends.com/en-us/Rune",
    "label": "League Wiki rune shard data",
    "page_id": 1327683,
    "revision_id": 4046438,
    "revision_timestamp": "2026-07-27T22:18:26Z",
    "supports": {"adaptive_shard_attack_damage": ADAPTIVE_SHARD_ATTACK_DAMAGE},
}
JAX_PASSIVE_SOURCE = {
    "kind": "wiki_rule",
    "url": "https://wiki.leagueoflegends.com/en-us/Jax/Patch_history",
    "label": "League Wiki Jax Relentless Assault patch history",
    "page_id": 1484918,
    "revision_id": 4026362,
    "revision_timestamp": "2026-06-09T21:47:13Z",
}


class BasicAttackProfileError(RuntimeError):
    """A requested static attack state cannot be resolved exactly."""


def _norm(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BasicAttackProfileError(f"expected a finite number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise BasicAttackProfileError("numeric input is not finite")
    return result


def _champion(engine: Any, champion_name: str) -> Mapping[str, Any]:
    key = _norm(champion_name)
    candidates = [
        row
        for row in getattr(engine, "_champions", ())
        if key
        in {
            _norm(str(row.get("name") or "")),
            _norm(str(row.get("alias") or "")),
        }
    ]
    base = [row for row in candidates if _norm(str(row.get("alias") or "")) == key]
    resolved = base or candidates
    if len(resolved) != 1:
        raise BasicAttackProfileError(
            f"champion {champion_name!r} does not resolve uniquely in the patch packet"
        )
    return resolved[0]


def _items(engine: Any, item_names: Sequence[str]) -> list[Mapping[str, Any]]:
    available = list(getattr(engine, "_items", {}).values())
    result: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for raw_name in item_names:
        key = _norm(str(raw_name))
        candidates = [item for item in available if _norm(str(item.get("name") or "")) == key]
        if len(candidates) != 1:
            raise BasicAttackProfileError(
                f"item {raw_name!r} does not resolve uniquely in the patch packet"
            )
        if key in seen:
            raise BasicAttackProfileError(f"item {raw_name!r} is duplicated")
        seen.add(key)
        result.append(candidates[0])
    return result


def _static_stat(item: Mapping[str, Any], field: str) -> float:
    stat = (item.get("stats") or {}).get(field)
    if not isinstance(stat, Mapping):
        return 0.0
    return _number(stat.get("value"))


def static_basic_attack_profile(
    engine: Any,
    *,
    champion_name: str,
    level: int,
    item_names: Sequence[str] = (),
    adaptive_shards: int = 0,
    extra_bonus_attack_speed_percent: float = 0.0,
    seconds: float = 10.0,
    attack_speed_cap: float = DEFAULT_ATTACK_SPEED_CAP,
) -> dict[str, Any]:
    if type(level) is not int or not 1 <= level <= 18:
        raise BasicAttackProfileError("level must be an integer from 1 through 18")
    if type(adaptive_shards) is not int or not 0 <= adaptive_shards <= 3:
        raise BasicAttackProfileError("adaptive_shards must be an integer from 0 through 3")
    seconds = _number(seconds)
    if seconds <= 0:
        raise BasicAttackProfileError("seconds must be positive")
    extra_bonus_attack_speed_percent = _number(extra_bonus_attack_speed_percent)
    if extra_bonus_attack_speed_percent < 0:
        raise BasicAttackProfileError("extra bonus attack speed cannot be negative")

    champion = _champion(engine, champion_name)
    level_row = (champion.get("levels") or {}).get(str(level))
    base_stats = champion.get("base_stats") or {}
    if not isinstance(level_row, Mapping) or not isinstance(base_stats, Mapping):
        raise BasicAttackProfileError("champion level or base-stat row is unavailable")
    item_rows = _items(engine, item_names)

    level_attack_damage = _number(level_row.get("attack_damage"))
    item_attack_damage = sum(_static_stat(item, "attack_damage") for item in item_rows)
    adaptive_attack_damage = adaptive_shards * ADAPTIVE_SHARD_ATTACK_DAMAGE
    attack_damage = level_attack_damage + item_attack_damage + adaptive_attack_damage

    base_attack_speed = _number(base_stats.get("attack_speed"))
    attack_speed_ratio = _number(
        base_stats.get("attack_speed_ratio", base_attack_speed)
    )
    growth_percent = _number(base_stats.get("attack_speed_per_level"))
    level_bonus_percent = growth_percent * level_growth_multiplier(level)
    item_bonus_percent = sum(_static_stat(item, "attack_speed") for item in item_rows)
    total_bonus_percent = (
        level_bonus_percent + item_bonus_percent + extra_bonus_attack_speed_percent
    )
    uncapped_attack_speed = base_attack_speed + attack_speed_ratio * (
        total_bonus_percent / 100.0
    )
    attacks_per_second = min(_number(attack_speed_cap), uncapped_attack_speed)
    raw_dps = attack_damage * attacks_per_second
    continuous_attack_intervals = attacks_per_second * seconds
    epsilon = 1e-9
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "available",
        "patch": str(getattr(engine, "patch", "")),
        "champion": str(champion.get("name") or champion_name),
        "level": level,
        "items": [str(item.get("name")) for item in item_rows],
        "adaptive_shards": adaptive_shards,
        "static_stats_only": True,
        "raw_physical_damage_per_basic_attack": attack_damage,
        "attacks_per_second": attacks_per_second,
        "raw_physical_dps": raw_dps,
        "window_seconds": seconds,
        "continuous_attack_intervals_in_window": continuous_attack_intervals,
        "discrete_attack_events": {
            "first_attack_at_time_zero": math.floor(
                continuous_attack_intervals + epsilon
            )
            + 1,
            "first_attack_after_one_full_interval": math.floor(
                continuous_attack_intervals + epsilon
            ),
        },
        "components": {
            "level_attack_damage": level_attack_damage,
            "item_attack_damage": item_attack_damage,
            "adaptive_attack_damage": adaptive_attack_damage,
            "base_attack_speed": base_attack_speed,
            "attack_speed_ratio": attack_speed_ratio,
            "level_bonus_attack_speed_percent": level_bonus_percent,
            "item_bonus_attack_speed_percent": item_bonus_percent,
            "extra_bonus_attack_speed_percent": extra_bonus_attack_speed_percent,
            "uncapped_attack_speed": uncapped_attack_speed,
            "attack_speed_cap": attack_speed_cap,
        },
        "ignored": {
            "item_passives": [
                str(item.get("name")) for item in item_rows if item.get("has_passive")
            ],
            "effects": [
                "crits",
                "on-hits",
                "empowered attacks",
                "target armor",
                "attack resets",
                "windup and animation cancellation",
            ],
        },
        "sources": [
            {
                "kind": "client_champion",
                "label": f"patch-pinned {champion.get('name')} champion data",
                **dict(champion.get("source") or {}),
            },
            *[
                {
                    "kind": "client_item",
                    "label": f"patch-pinned {item.get('name')} item data",
                    "item_id": item.get("id"),
                }
                for item in item_rows
            ],
            ATTACK_SPEED_SOURCE,
            ADAPTIVE_SHARD_SOURCE,
        ],
    }


def jax_passive_attack_speed(engine: Any, *, level: int) -> dict[str, Any]:
    """Read Jax's per-stack AS and maximum stacks from his hashed client bin."""

    champion = _champion(engine, "Jax")
    source = champion.get("source") or {}
    raw_relative = source.get("bin_json_path")
    expected_hash = source.get("bin_sha256")
    raw_root = getattr(engine, "raw_champion_root", None)
    if not isinstance(raw_relative, str) or not isinstance(expected_hash, str) or raw_root is None:
        raise BasicAttackProfileError("Jax passive client source is unavailable")
    packet_root = Path(raw_root).parent.parent
    path = (packet_root / raw_relative).resolve()
    try:
        path.relative_to(packet_root.resolve())
    except ValueError as exc:
        raise BasicAttackProfileError("Jax passive source escapes the packet") from exc
    raw_bytes = path.read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest() != expected_hash:
        raise BasicAttackProfileError("Jax passive source hash does not verify")
    payload = json.loads(raw_bytes)
    spell = payload.get("Characters/Jax/Spells/JaxPassiveAbility/JaxPassive")
    resource = spell.get("mSpell") if isinstance(spell, Mapping) else None
    if not isinstance(resource, Mapping):
        raise BasicAttackProfileError("Jax passive spell resource is unavailable")
    calculations = resource.get("mSpellCalculations") or {}
    calculation = calculations.get("AttackSpeedPerStack")
    parts = calculation.get("mFormulaParts") if isinstance(calculation, Mapping) else None
    if not isinstance(parts, list) or len(parts) != 1 or not isinstance(parts[0], Mapping):
        raise BasicAttackProfileError("Jax passive attack-speed formula is ambiguous")
    part = parts[0]
    per_stack = _number(part.get("mLevel1Value"))
    for breakpoint in part.get("mBreakpoints") or []:
        if not isinstance(breakpoint, Mapping):
            continue
        breakpoint_level = int(breakpoint.get("mLevel") or 0)
        if level >= breakpoint_level:
            per_stack += _number(breakpoint.get("mAdditionalBonusAtThisLevel"))
    max_stack_rows = [
        row
        for row in resource.get("DataValues") or []
        if isinstance(row, Mapping) and row.get("name") == "MaxStacks"
    ]
    if len(max_stack_rows) != 1:
        raise BasicAttackProfileError("Jax passive maximum stacks are ambiguous")
    values = max_stack_rows[0].get("values") or []
    if not values:
        raise BasicAttackProfileError("Jax passive maximum stacks are unavailable")
    max_stacks = int(round(_number(values[0])))
    return {
        "per_stack_percent": per_stack * 100.0,
        "max_stacks": max_stacks,
        "full_stack_percent": per_stack * 100.0 * max_stacks,
        "client_bin_sha256": expected_hash,
        "sources": [JAX_PASSIVE_SOURCE],
    }


def compare_jax_basic_attacks(
    engine: Any,
    *,
    level: int,
    item_names: Sequence[str],
    adaptive_shards: int,
    seconds: float = 10.0,
) -> dict[str, Any]:
    passive = jax_passive_attack_speed(engine, level=level)
    states = {
        "zero_passive_stacks": 0.0,
        "full_passive_stacks": _number(passive["full_stack_percent"]),
    }
    profiles: dict[str, Any] = {}
    for state, extra_as in states.items():
        without_items = static_basic_attack_profile(
            engine,
            champion_name="Jax",
            level=level,
            item_names=(),
            adaptive_shards=adaptive_shards,
            extra_bonus_attack_speed_percent=extra_as,
            seconds=seconds,
        )
        with_items = static_basic_attack_profile(
            engine,
            champion_name="Jax",
            level=level,
            item_names=item_names,
            adaptive_shards=adaptive_shards,
            extra_bonus_attack_speed_percent=extra_as,
            seconds=seconds,
        )
        profiles[state] = {
            "without_items": without_items,
            "with_items": with_items,
            "with_items_minus_without_items": {
                "raw_damage_per_attack": (
                    with_items["raw_physical_damage_per_basic_attack"]
                    - without_items["raw_physical_damage_per_basic_attack"]
                ),
                "attacks_per_second": (
                    with_items["attacks_per_second"]
                    - without_items["attacks_per_second"]
                ),
                "raw_dps": (
                    with_items["raw_physical_dps"]
                    - without_items["raw_physical_dps"]
                ),
                "continuous_attack_intervals_in_window": (
                    with_items["continuous_attack_intervals_in_window"]
                    - without_items["continuous_attack_intervals_in_window"]
                ),
                "discrete_first_attack_at_time_zero": (
                    with_items["discrete_attack_events"]["first_attack_at_time_zero"]
                    - without_items["discrete_attack_events"]["first_attack_at_time_zero"]
                ),
                "discrete_first_attack_after_one_full_interval": (
                    with_items["discrete_attack_events"][
                        "first_attack_after_one_full_interval"
                    ]
                    - without_items["discrete_attack_events"][
                        "first_attack_after_one_full_interval"
                    ]
                ),
            },
        }

    def ramp_schedule(
        profile: Mapping[str, Any], *, first_attack_at_zero: bool
    ) -> dict[str, Any]:
        components = profile["components"]
        base_attack_speed = _number(components["base_attack_speed"])
        ratio = _number(components["attack_speed_ratio"])
        starting_bonus = (
            _number(components["level_bonus_attack_speed_percent"])
            + _number(components["item_bonus_attack_speed_percent"])
        )
        cap = _number(components["attack_speed_cap"])
        per_stack = _number(passive["per_stack_percent"])
        max_stacks = int(passive["max_stacks"])

        def attacks_per_second(stacks: int) -> float:
            return min(
                cap,
                base_attack_speed
                + ratio * ((starting_bonus + per_stack * stacks) / 100.0),
            )

        if first_attack_at_zero:
            elapsed = 0.0
            attacks = 1
            stacks = min(1, max_stacks)
            hit_times = [0.0]
        else:
            elapsed = 1.0 / attacks_per_second(0)
            attacks = 0
            stacks = 0
            hit_times = []
        while elapsed <= seconds + 1e-9:
            if not first_attack_at_zero or attacks > 0:
                if not first_attack_at_zero and attacks == 0:
                    attacks = 1
                    stacks = min(1, max_stacks)
                    hit_times.append(elapsed)
                interval = 1.0 / attacks_per_second(stacks)
                next_hit = elapsed + interval
                if next_hit > seconds + 1e-9:
                    break
                elapsed = next_hit
                attacks += 1
                stacks = min(max_stacks, stacks + 1)
                hit_times.append(elapsed)
        return {
            "attacks": attacks,
            "hit_times_seconds": hit_times,
            "ending_stacks": stacks,
            "first_attack_at_time_zero": first_attack_at_zero,
        }

    zero_profiles = profiles["zero_passive_stacks"]
    ramp: dict[str, Any] = {}
    for convention, first_at_zero in (
        ("first_attack_at_time_zero", True),
        ("first_attack_after_one_full_interval", False),
    ):
        without_items = ramp_schedule(
            zero_profiles["without_items"], first_attack_at_zero=first_at_zero
        )
        with_items = ramp_schedule(
            zero_profiles["with_items"], first_attack_at_zero=first_at_zero
        )
        ramp[convention] = {
            "without_items": without_items,
            "with_items": with_items,
            "additional_attacks_with_items": (
                with_items["attacks"] - without_items["attacks"]
            ),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "available",
        "question_scope": "static ordinary basic attacks only",
        "champion": "Jax",
        "level": level,
        "items": list(item_names),
        "adaptive_shards_in_both_builds": adaptive_shards,
        "window_seconds": seconds,
        "passive": passive,
        "profiles": profiles,
        "ideal_uninterrupted_ramp_from_zero": {
            "conventions": ramp,
            "assumptions": [
                "the target grants one Relentless Assault stack per completed basic attack",
                "the next attack interval uses the newly gained stack",
                "no stack expires, no attack reset occurs, and there is no movement or downtime",
            ],
        },
        "plain_language_guardrail": (
            "The static-state answer ranges from one to two extra discrete attacks. "
            "If Jax starts at zero and ramps uninterrupted, this idealized timer model "
            "gives two to three. Use the continuous rate difference when the actual "
            "stack and timer state are not observed."
        ),
    }


def _compact_jax_table(result: Mapping[str, Any]) -> str:
    lines = [
        "| Jax state | Build | Raw hit | Attacks/s | Raw DPS | Hits in 10s (first at 0) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for state, profile in (result.get("profiles") or {}).items():
        state_label = "0 passive stacks" if state == "zero_passive_stacks" else "8 passive stacks"
        for build_key, build_label in (
            ("without_items", "No items"),
            ("with_items", "Named items"),
        ):
            row = profile[build_key]
            lines.append(
                f"| {state_label} | {build_label} | "
                f"{row['raw_physical_damage_per_basic_attack']:.1f} | "
                f"{row['attacks_per_second']:.3f} | "
                f"{row['raw_physical_dps']:.1f} | "
                f"{row['discrete_attack_events']['first_attack_at_time_zero']} |"
            )
    ramp = result["ideal_uninterrupted_ramp_from_zero"]["conventions"]
    lines.extend(
        [
            "",
            "Ideal uninterrupted ramp from 0 stacks:",
            "",
            f"- First hit at 0s: {ramp['first_attack_at_time_zero']['without_items']['attacks']} → {ramp['first_attack_at_time_zero']['with_items']['attacks']} hits (**{ramp['first_attack_at_time_zero']['additional_attacks_with_items']} extra**).",
            f"- First hit after one interval: {ramp['first_attack_after_one_full_interval']['without_items']['attacks']} → {ramp['first_attack_after_one_full_interval']['with_items']['attacks']} hits (**{ramp['first_attack_after_one_full_interval']['additional_attacks_with_items']} extra**).",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=Path,
        default=Path(
            f"data/lol/knowledge/patch-packets/cdragon/2026/{CURRENT_PUBLIC_PATCH}/mechanics-index.json"
        ),
    )
    parser.add_argument("--champion", default="Jax")
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--items", default="")
    parser.add_argument("--adaptive-shards", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    args = parser.parse_args(argv)

    from .lol_oracle import LeagueOracleEngine
    from .quick_mechanics_fastpack import compile_fastpack

    engine = LeagueOracleEngine(
        compile_fastpack(args.index),
        raw_champion_root=args.index.parent / "raw" / "champions",
    )
    item_names = [value.strip() for value in args.items.split(",") if value.strip()]
    if _norm(args.champion) == "jax":
        result = compare_jax_basic_attacks(
            engine,
            level=args.level,
            item_names=item_names,
            adaptive_shards=args.adaptive_shards,
            seconds=args.seconds,
        )
        print(_compact_jax_table(result) if args.format == "markdown" else json.dumps(result, indent=2, sort_keys=True))
    else:
        result = static_basic_attack_profile(
            engine,
            champion_name=args.champion,
            level=args.level,
            item_names=item_names,
            adaptive_shards=args.adaptive_shards,
            seconds=args.seconds,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BasicAttackProfileError",
    "compare_jax_basic_attacks",
    "jax_passive_attack_speed",
    "static_basic_attack_profile",
]
