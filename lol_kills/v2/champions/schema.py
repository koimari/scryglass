"""Schema definitions and validation helpers for v2 champion ontology."""

from __future__ import annotations

from collections.abc import Mapping

ROLES: tuple[str, ...] = ("top", "jungle", "mid", "bot", "support")

DIMENSION_DEFINITIONS: dict[str, dict[str, tuple[str, ...]]] = {
    "damage_profile": {
        "description": "damage channel focus",
        "labels": ("poke", "burst", "artillery", "teamfight"),
    },
    "effective_range": {
        "description": "preferred range envelope",
        "labels": ("point_blank", "short", "mid", "long"),
    },
    "engage": {
        "description": "ability to force contact or start fights",
        "labels": ("none", "single", "area", "hard"),
    },
    "peel": {
        "description": "ability to protect a target from pressure",
        "labels": ("none", "defensive", "forced", "zone"),
    },
    "mobility": {
        "description": "rotation speed, gap-closing, repositioning",
        "labels": ("low", "moderate", "high"),
    },
    "wave_control": {
        "description": "lane and wave shaping actions",
        "labels": ("trade", "push", "siege", "freeze"),
    },
    "scaling": {
        "description": "power trend by game phase",
        "labels": ("early", "mid", "late", "late_spike"),
    },
    "target_access": {
        "description": "ways the champion accesses priority targets",
        "labels": ("frontline", "hook", "reposition", "split_push"),
    },
    "durability_frontline": {
        "description": "survivability and frontline suitability",
        "labels": ("squishy", "frontline", "tank", "duelist"),
    },
    "crowd_control": {
        "description": "crowd-control leverage",
        "labels": ("none", "slow", "root", "silence", "stun"),
    },
    "sustain": {
        "description": "self or ally sustain from abilities",
        "labels": ("none", "self_heal", "shield", "drain"),
    },
    "tempo": {
        "description": "draft and fight pacing impact",
        "labels": ("slow_buildup", "early_spread", "mid_pressure", "late_reset"),
    },
}

REQUIRED_DIMENSIONS = tuple(DIMENSION_DEFINITIONS.keys())
DIMENSION_LABELS = {name: tuple(data["labels"]) for name, data in DIMENSION_DEFINITIONS.items()}
DIMENSION_LABEL_ORDER = tuple(
    (name, label) for name, labels in DIMENSION_LABELS.items() for label in labels
)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_dimension_map(dimension: str, labels: Mapping[str, object]) -> dict[str, float]:
    """Validate a single dimension label-probability map."""

    if not isinstance(labels, Mapping):
        raise ValueError(f"dimension '{dimension}' labels must be a mapping")
    if dimension not in DIMENSION_DEFINITIONS:
        raise ValueError(f"unknown dimension: {dimension}")

    allowed_labels = set(DIMENSION_LABELS[dimension])
    observed_labels = set(labels.keys())
    if observed_labels != allowed_labels:
        extra = ", ".join(sorted(observed_labels - allowed_labels))
        missing = ", ".join(sorted(allowed_labels - observed_labels))
        missing_text = f" missing: {missing}" if missing else ""
        extra_text = f" unexpected: {extra}" if extra else ""
        raise ValueError(
            f"dimension '{dimension}' must contain exactly "
            f"{', '.join(sorted(allowed_labels))};{missing_text}{extra_text}"
        )

    out: dict[str, float] = {}
    for label in allowed_labels:
        value = float(labels[label])
        if not _is_number(labels[label]) or not 0.0 <= value <= 1.0:
            raise ValueError(f"dimension '{dimension}.{label}' must be numeric in [0, 1]")
        out[label] = value
    return out


def validate_uncertainty_map(dimension: str, uncertainty: object) -> dict[str, float]:
    """Validate the dimension uncertainty map."""

    if not isinstance(uncertainty, Mapping):
        raise ValueError(f"dimension '{dimension}' uncertainty must be a mapping")
    allowed = set(DIMENSION_LABELS[dimension])
    observed = set(uncertainty.keys())
    if observed - allowed:
        extra = ", ".join(sorted(observed - allowed))
        raise ValueError(f"dimension '{dimension}' uncertainty has unexpected label(s): {extra}")

    out: dict[str, float] = {}
    for label in DIMENSION_LABELS[dimension]:
        value = uncertainty.get(label, 0.0)
        if not _is_number(value) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(
                f"dimension '{dimension}.{label}' uncertainty must be numeric in [0, 1]"
            )
        out[label] = float(value)
    return out
