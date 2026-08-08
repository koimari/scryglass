"""Exact static item-stat extraction from the patch-pinned client payload."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping


_STAT_RE = re.compile(
    r"<attention>\s*([0-9]+(?:\.[0-9]+)?)\s*(%?)\s*</attention>\s*([^<]+)",
    re.I,
)

_LABELS = {
    "attack damage": ("attack_damage", "attack damage"),
    "attack speed": ("attack_speed", "attack speed"),
    "ability power": ("ability_power", "ability power"),
    "health": ("health", "health"),
    "mana": ("mana", "mana"),
    "armor": ("armor", "armor"),
    "magic resist": ("magic_resist", "magic resist"),
    "ability haste": ("ability_haste", "ability haste"),
    "lethality": ("lethality", "lethality"),
    "critical strike chance": ("critical_strike_chance", "critical strike chance"),
    "move speed": ("move_speed", "move speed"),
    "magic penetration": ("magic_penetration", "magic penetration"),
    "armor penetration": ("armor_penetration", "armor penetration"),
    "life steal": ("life_steal", "life steal"),
}


def parse_static_item_stats(item: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract only the visible ``<stats>`` numbers, never passive text."""

    description = item.get("description")
    if not isinstance(description, str):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for raw_value, percent, raw_label in _STAT_RE.findall(description):
        label = " ".join(raw_label.strip().casefold().split())
        mapped = _LABELS.get(label)
        if mapped is None:
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        field, display = mapped
        result[field] = {
            "value": int(value) if value.is_integer() else value,
            "percent": bool(percent),
            "label": display,
        }
    return result


__all__ = ["parse_static_item_stats"]
