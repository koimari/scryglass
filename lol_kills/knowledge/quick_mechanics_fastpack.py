"""Compile a small, deterministic League mechanics fastpack.

The CommunityDragon mechanics packet is deliberately kept as a raw-source
receipt.  This module turns the receipt into the read-mostly shape used by the
quick-answer path: one JSON load, then dictionary/array lookups only.  It is
not a general game emulator.  Fields that cannot be resolved from the exact
patch source remain ``None`` and are labelled in ``unresolved``; they are never
silently copied from another champion or patch.

The compiler is network-free.  ``index_path`` must point at a previously
captured ``mechanics-index.json`` and every referenced bin is verified against
the SHA-256 receipt in that index.

Level maps use string keys (JSON's canonical object-key representation),
``"1"`` through ``"18"``.  The dense ``stat_tables`` arrays use level order
and are convenient for a vectorized caller (index ``level - 1``).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "scryglass:quick-mechanics-fastpack:v1"
LEVELS = tuple(range(1, 19))

# The client enum is intentionally retained in each record.  A label is useful
# to the query layer, while the numeric id lets a future compiler distinguish a
# newly introduced client resource without guessing its semantics.
RESOURCE_TYPE_LABELS: dict[int, str] = {
    0: "mana",
    1: "energy",
    2: "fury",
    3: "shield",
    4: "fury",
    5: "heat",
    6: "rage",
    7: "style",
    8: "rage",
    9: "ferocity",
    10: "blood_well",
    11: "flow",
    12: "courage",
    13: "grit",
    14: "void_coral",
}

_RECORD_SUFFIX = "/CharacterRecords/Root"
_CORE_FIELDS: dict[str, str] = {
    "base_health": "baseHPModifiable",
    "health_per_level": "hpPerLevelModifiable",
    "base_health_regen_per_second": "baseStaticHPRegenModifiable",
    "health_regen_per_level_per_second": "hpRegenPerLevelModifiable",
    "base_attack_damage": "baseDamageModifiable",
    "attack_damage_per_level": "damagePerLevelModifiable",
    "base_armor": "baseArmorModifiable",
    "armor_per_level": "armorPerLevelModifiable",
    "base_magic_resist": "baseMR",
    "magic_resist_per_level": "{01262a25}",
    "base_move_speed": "baseMoveSpeedModifiable",
    "attack_range": "attackRangeModifiable",
    "attack_speed": "attackSpeedModifiable",
    "attack_speed_ratio": "attackSpeedRatioModifiable",
    "attack_speed_per_level": "attackSpeedPerLevelModifiable",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )


def normalize_alias(value: str) -> str:
    """Return the stable alias key used by the fast path.

    Accents, punctuation, apostrophes, spaces, and underscores are removed so
    ``K'Sante``, ``K Sante`` and ``KSante`` resolve to the same key.  Unicode
    normalization is done before filtering to keep this deterministic without
    a locale-dependent fuzzy-search library.
    """

    if not isinstance(value, str):
        raise TypeError("alias must be a string")
    decomposed = unicodedata.normalize("NFKD", value).casefold()
    return "".join(ch for ch in decomposed if ch.isalnum())


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _base_value(value: Any) -> float | None:
    if not isinstance(value, Mapping):
        return None
    return _finite_number(value.get("baseValue"))


def _level_multiplier(level: int) -> float:
    """The standard champion growth multiplier for levels 1 through 18."""

    if type(level) is not int or level not in LEVELS:
        raise ValueError("champion level must be an integer in [1, 18]")
    n = level - 1
    return float(n * (0.7025 + 0.0175 * n))


def level_growth_multiplier(level: int) -> float:
    """Public exact champion growth multiplier used by downstream calculators."""

    return _level_multiplier(level)


def _find_record(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    records = [
        value
        for key, value in payload.items()
        if isinstance(key, str)
        and key.endswith(_RECORD_SUFFIX)
        and isinstance(value, Mapping)
        and value.get("__type") == "CharacterRecord"
    ]
    if not records:
        return None
    if len(records) > 1:
        # A duplicate root is ambiguous.  Falling through to the first one
        # would make a plausible but untraceable answer.
        raise ValueError("character bin contains multiple CharacterRecords/Root objects")
    return records[0]


def _resource_info(record: Mapping[str, Any]) -> dict[str, Any]:
    raw = record.get("primaryAbilityResource")
    if not isinstance(raw, Mapping):
        return {
            "type": "none",
            "type_id": None,
            "base_max": None,
            "max_per_level": None,
            "base_regen_per_second": None,
            "regen_per_level_per_second": None,
        }
    raw_id = raw.get("arType")
    type_id = int(raw_id) if isinstance(raw_id, int) and not isinstance(raw_id, bool) else None
    if type_id is None:
        resource_type = "unknown"
    else:
        resource_type = RESOURCE_TYPE_LABELS.get(type_id, f"unknown_{type_id}")
    return {
        "type": resource_type,
        "type_id": type_id,
        "base_max": _base_value(raw.get("{726ee5cd}")),
        "max_per_level": _base_value(raw.get("{6216bf7b}")),
        "base_regen_per_second": _base_value(raw.get("{c4ab3550}")),
        "regen_per_level_per_second": _base_value(raw.get("{3a509002}")),
    }


def _number_or_none(value: float | None) -> float | None:
    # Preserve exact source precision while avoiding -0.0 in JSON.
    if value is None:
        return None
    return 0.0 if value == 0 else float(value)


def _table_value(base: float | None, growth: float | None, level: int) -> float | None:
    if base is None or growth is None:
        return None
    return _number_or_none(base + growth * _level_multiplier(level))


def _constant_table(value: float | None) -> list[float | None]:
    return [_number_or_none(value) for _ in LEVELS]


def _build_stat_tables(
    stats: Mapping[str, float | None], resource: Mapping[str, Any]
) -> dict[str, list[float | None]]:
    tables: dict[str, list[float | None]] = {}

    pairs = {
        "max_health": (stats.get("base_health"), stats.get("health_per_level")),
        "attack_damage": (stats.get("base_attack_damage"), stats.get("attack_damage_per_level")),
        "armor": (stats.get("base_armor"), stats.get("armor_per_level")),
        "magic_resist": (stats.get("base_magic_resist"), stats.get("magic_resist_per_level")),
        "health_regen_per_5": (
            stats.get("base_health_regen_per_second"),
            stats.get("health_regen_per_level_per_second"),
        ),
    }
    for name, (base, growth) in pairs.items():
        values = [_table_value(base, growth, level) for level in LEVELS]
        if name == "health_regen_per_5":
            values = [_number_or_none(value * 5 if value is not None else None) for value in values]
        tables[name] = values

    max_base = resource.get("base_max")
    max_growth = resource.get("max_per_level")
    tables["max_resource"] = [
        _table_value(max_base, max_growth, level) for level in LEVELS
    ]
    regen_base = resource.get("base_regen_per_second")
    regen_growth = resource.get("regen_per_level_per_second")
    regen = [_table_value(regen_base, regen_growth, level) for level in LEVELS]
    tables["resource_regen_per_5"] = [
        _number_or_none(value * 5 if value is not None else None) for value in regen
    ]
    # Short aliases mirror the fields on each level object.  Keeping these in
    # the dense table makes callers independent of the JSON object view.
    tables["hp"] = list(tables["max_health"])
    tables["ad"] = list(tables["attack_damage"])
    tables["mr"] = list(tables["magic_resist"])
    tables["hp5"] = list(tables["health_regen_per_5"])
    tables["mp5"] = [
        value if resource.get("type") == "mana" else None
        for value in tables["resource_regen_per_5"]
    ]
    return tables


def _with_stat_aliases(level: Mapping[str, Any], resource_type: str) -> dict[str, Any]:
    result = dict(level)
    result.update(
        {
            "hp": result["max_health"],
            "ad": result["attack_damage"],
            "mr": result["magic_resist"],
            "hp5": result["health_regen_per_5"],
            "resource_regen_per_5": result["resource_regen_per_5"],
            # MP5 is intentionally unavailable for non-mana resources.  The
            # generic resource_regen_per_5 key remains available where the
            # client exposes a real resource regeneration field.
            "mp5": result["resource_regen_per_5"] if resource_type == "mana" else None,
        }
    )
    return result


def _build_champion(index_path: Path, entry: Mapping[str, Any]) -> tuple[str, dict[str, Any], list[str]]:
    raw_id = entry.get("id")
    if isinstance(raw_id, bool) or not isinstance(raw_id, int):
        raise ValueError(f"champion id is not an integer: {raw_id!r}")
    raw_bin_path = entry.get("bin_json_path")
    if not isinstance(raw_bin_path, str) or not raw_bin_path:
        raise ValueError(f"champion {raw_id} has no bin_json_path")
    bin_path = (index_path.parent / raw_bin_path).resolve()
    try:
        bin_path.relative_to(index_path.parent.resolve())
    except ValueError as exc:
        raise ValueError(f"champion bin escapes index directory: {raw_bin_path!r}") from exc
    if not bin_path.is_file():
        raise FileNotFoundError(f"referenced champion bin is missing: {bin_path}")
    bin_bytes = bin_path.read_bytes()
    expected_hash = entry.get("bin_sha256")
    actual_hash = _sha256_bytes(bin_bytes)
    if not isinstance(expected_hash, str) or expected_hash != actual_hash:
        raise ValueError(
            f"champion bin hash mismatch for {entry.get('alias', raw_id)!r}: "
            f"expected {expected_hash!r}, got {actual_hash}"
        )
    try:
        payload = json.loads(bin_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"champion bin is not JSON: {bin_path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"champion bin root is not an object: {bin_path}")
    record = _find_record(payload)
    unresolved: list[str] = []
    if record is None:
        unresolved.append("character_record")
        record = {}

    stats: dict[str, float | None] = {}
    for name, source_key in _CORE_FIELDS.items():
        value = _base_value(record.get(source_key))
        stats[name] = value
        if value is None:
            unresolved.append(name)
    resource = _resource_info(record)
    if resource["type"] == "unknown" or (
        resource["type_id"] is not None
        and resource["type_id"] not in RESOURCE_TYPE_LABELS
    ):
        unresolved.append("resource_type")

    tables = _build_stat_tables(stats, resource)
    level_maps: dict[str, dict[str, Any]] = {}
    for i, level in enumerate(LEVELS):
        level_maps[str(level)] = _with_stat_aliases(
            {
                "level": level,
                "max_health": tables["max_health"][i],
                "attack_damage": tables["attack_damage"][i],
                "armor": tables["armor"][i],
                "magic_resist": tables["magic_resist"][i],
                "health_regen_per_5": tables["health_regen_per_5"][i],
                "max_resource": tables["max_resource"][i],
                "resource_regen_per_5": tables["resource_regen_per_5"][i],
            },
            str(resource["type"]),
        )

    alias = entry.get("alias")
    if not isinstance(alias, str) or not alias.strip():
        raise ValueError(f"champion {raw_id} has no alias")
    display_name = entry.get("name") or alias
    if not isinstance(display_name, str):
        display_name = alias
    aliases: list[str] = []
    for candidate in (alias, display_name, record.get("mCharacterName")):
        if isinstance(candidate, str) and candidate.strip() and candidate not in aliases:
            aliases.append(candidate)
    normalized_aliases = sorted({normalize_alias(value) for value in aliases if normalize_alias(value)})
    if not normalized_aliases:
        unresolved.append("aliases")

    # Keep raw values and derived values separate: callers can render a
    # provenance/debug panel without reverse-engineering the level arrays.
    champion = {
        "id": raw_id,
        "alias": alias,
        "name": display_name,
        "aliases": aliases,
        "normalized_aliases": normalized_aliases,
        "status": "available" if not unresolved else "partial",
        "unresolved": sorted(set(unresolved)),
        "resource_type": resource["type"],
        "resource_type_id": resource["type_id"],
        "resource": resource,
        "base_stats": stats,
        "stat_tables": tables,
        "levels": level_maps,
        "source": {
            "bin_json_path": raw_bin_path,
            "bin_sha256": actual_hash,
        },
    }
    # The id string is the canonical artifact key.  This avoids collisions for
    # skin/test records whose display name is the same as a champion.
    return str(raw_id), champion, normalized_aliases


def _supplements(patch: str) -> dict[str, Any]:
    """Small, explicit non-champion facts used by the first answer set."""

    if patch != "26.15":
        # These values are Wiki-grounded rather than present in the champion
        # bins.  Do not silently carry them into a newer or older patch.
        return {
            "status": "unavailable",
            "reason": "monster/objective supplements are validated only for patch 26.15",
            "monsters": {},
            "objectives": {},
        }

    gromp_health = [
        2050.0,
        2255.0,
        2460.0,
        2665.0,
        2870.0,
        3075.0,
        3280.0,
        3485.0,
        3690.0,
        3895.0,
        4100.0,
        4202.5,
        4305.0,
        4407.5,
        4510.0,
        4612.5,
        4715.0,
        4817.5,
    ]
    gromp_levels = {
        str(level): {
            "level": level,
            "max_health": gromp_health[level - 1],
            "hp": gromp_health[level - 1],
            "armor": 42.0,
            "magic_resist": 42.0,
            "mr": 42.0,
        }
        for level in LEVELS
    }
    provenance = {
        "kind": "supplement",
        "authority": "League of Legends Wiki",
        "patch": patch,
        "network_free": True,
        "notes": "Explicit patch-pinned supplement; not inferred from champion bins.",
    }
    return {
        "status": "available",
        "monsters": {
            "gromp": {
                "id": "gromp",
                "name": "Gromp",
                "aliases": ["Gromp"],
                "normalized_aliases": ["gromp"],
                "status": "available",
                "levels": gromp_levels,
                "stat_tables": {
                    "max_health": gromp_health,
                    "armor": [42.0 for _ in LEVELS],
                    "magic_resist": [42.0 for _ in LEVELS],
                },
                "provenance": {
                    **provenance,
                    "source_page": "Gromp",
                    "source_url": "https://wiki.leagueoflegends.com/en-us/Gromp",
                    "revision_id": 4016297,
                    "revision_timestamp": "2026-05-10T13:54:32Z",
                    "content_sha256": "a95003a811df280d2fe2f4fffb775eb188f6156f98ebed7866070bbebc6e6383",
                    "formula": "2050 + 205 per level starting at level 3, then +102.5 per level starting at level 12",
                },
            }
        },
        "objectives": {
            "void_grubs": {
                "id": "void_grubs",
                "name": "Void Grubs (full three-grub camp)",
                "aliases": ["Void Grubs", "Voidgrubs", "Grubs", "Grub camp"],
                "normalized_aliases": ["voidgrubs", "grubs", "grubcamp"],
                "count": 3,
                "gold": {
                    "local": 90.0,
                    "global": 0.0,
                    "total_local": 90.0,
                    "total_global": 0.0,
                },
                "cash_local": 90.0,
                "cash_global": 0.0,
                "assumptions": [
                    "full camp means all three Void Grubs are secured by one team",
                    "cash only: experience and Touch of the Void are not included",
                    "local means killer-local gold; there is no global gold share",
                ],
                "provenance": {
                    **provenance,
                    "source_page": "Voidgrub camp",
                    "source_url": "https://wiki.leagueoflegends.com/en-us/Voidgrub_camp",
                    "revision_id": 4015021,
                    "revision_timestamp": "2026-05-02T12:51:17Z",
                    "content_sha256": "a7fec61787c1eddb60790ae624aeb2ddb2b1b676b48ac5bf4f1dd842409b4b14",
                },
            }
        },
    }


def compile_fastpack(index_path: Path) -> dict[str, Any]:
    """Compile and return a deterministic in-memory fastpack.

    ``index_path`` is expected to be the exact-patch mechanics index.  Source
    files and hashes are checked before any derived value is accepted.
    """

    index_path = Path(index_path)
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    index_bytes = index_path.read_bytes()
    index_hash = _sha256_bytes(index_bytes)
    try:
        index = json.loads(index_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"mechanics index is not JSON: {index_path}") from exc
    if not isinstance(index, Mapping):
        raise ValueError("mechanics index root must be an object")
    entries = index.get("champions")
    if not isinstance(entries, list):
        raise ValueError("mechanics index has no champions list")
    patch = index.get("patch")
    client_patch = index.get("client_patch")
    if not isinstance(patch, str) or not patch:
        raise ValueError("mechanics index has no patch")
    if not isinstance(client_patch, str) or not client_patch:
        raise ValueError("mechanics index has no exact client_patch")

    champions: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    bin_hashes: dict[str, str] = {}
    alias_conflicts: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("mechanics index contains a non-object champion entry")
        key, champion, normalized_aliases = _build_champion(index_path, entry)
        if key in champions:
            raise ValueError(f"duplicate champion id in mechanics index: {key}")
        champions[key] = champion
        bin_hashes[key] = champion["source"]["bin_sha256"]
        for normalized in normalized_aliases:
            previous = aliases.get(normalized)
            if previous is None:
                aliases[normalized] = key
            elif previous != key:
                # Base champions win a duplicate skin/display alias; the full
                # Jade_* alias remains available via its own normalized alias.
                previous_champion = champions[previous]
                if previous_champion["alias"].startswith("Jade_") and not champion["alias"].startswith("Jade_"):
                    aliases[normalized] = key
                elif not (
                    previous_champion["alias"].startswith("Jade_")
                    or champion["alias"].startswith("Jade_")
                ):
                    alias_conflicts.append(f"{normalized}:{previous},{key}")

    supplement = _supplements(patch)
    for monster_id, monster in supplement["monsters"].items():
        aliases.setdefault(normalize_alias(monster["name"]), f"monster:{monster_id}")
    for objective_id, objective in supplement["objectives"].items():
        for normalized in objective["normalized_aliases"]:
            aliases.setdefault(normalized, f"objective:{objective_id}")

    source_hash = _sha256_json(
        {
            "index_sha256": index_hash,
            "champion_bins": bin_hashes,
            "supplements_sha256": _sha256_json(supplement),
        }
    )
    pack = {
        "schema_version": SCHEMA_VERSION,
        "kind": "quick_mechanics_fastpack",
        "patch": patch,
        "client_patch": client_patch,
        "source": index.get("source", "CommunityDragon"),
        "source_root": index.get("source_root"),
        "source_hash": source_hash,
        "source_sha256": source_hash,
        "source_hashes": {
            "index_sha256": index_hash,
            "champion_bins": bin_hashes,
            "supplements_sha256": _sha256_json(supplement),
        },
        "index_sha256": index_hash,
        "level_key_type": "string",
        "levels": list(LEVELS),
        "champions": champions,
        "aliases": aliases,
        "alias_conflicts": sorted(set(alias_conflicts)),
        "supplements": supplement,
        # Convenience views keep the first query path simple while
        # ``supplements`` remains the provenance-labelled source boundary.
        "monsters": supplement["monsters"],
        "objectives": supplement["objectives"],
        "assumptions": [
            "Champion derived stats use the exact standard level-growth multiplier at levels 1-18.",
            "Resource regeneration from CharacterRecord is per second and normalized here to per 5 seconds.",
            "A null field is unavailable because the exact patch source did not resolve it; no fallback patch is used.",
            "Monster and objective values are explicit provenance-labelled supplements.",
        ],
    }
    _validate_fastpack(pack)
    return pack


def _validate_fastpack(pack: Mapping[str, Any]) -> None:
    if pack.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported quick mechanics fastpack schema")
    if not isinstance(pack.get("patch"), str) or not pack["patch"]:
        raise ValueError("fastpack has no patch")
    source_hashes = pack.get("source_hashes")
    if not isinstance(source_hashes, Mapping):
        raise ValueError("fastpack has no source hashes")
    if not isinstance(source_hashes.get("index_sha256"), str) or not source_hashes["index_sha256"]:
        raise ValueError("fastpack has no index source hash")
    champion_bins = source_hashes.get("champion_bins")
    if not isinstance(champion_bins, Mapping):
        raise ValueError("fastpack has no champion source hashes")
    if pack.get("level_key_type") != "string" or pack.get("levels") != list(LEVELS):
        raise ValueError("fastpack level contract is malformed")
    champions = pack.get("champions")
    aliases = pack.get("aliases")
    if not isinstance(champions, Mapping) or not champions:
        raise ValueError("fastpack has no champions")
    if not isinstance(aliases, Mapping):
        raise ValueError("fastpack has no alias map")
    for key, champion in champions.items():
        if not isinstance(key, str) or not isinstance(champion, Mapping):
            raise ValueError("fastpack champion map is malformed")
        levels = champion.get("levels")
        if not isinstance(levels, Mapping) or set(levels) != {str(level) for level in LEVELS}:
            raise ValueError(f"champion {key} does not have complete levels 1-18")
        for level in LEVELS:
            if not isinstance(levels[str(level)], Mapping):
                raise ValueError(f"champion {key} level {level} is malformed")
    for alias, target in aliases.items():
        if not isinstance(alias, str) or not isinstance(target, str):
            raise ValueError("fastpack alias map is malformed")
        if not (target in champions or target.startswith("monster:") or target.startswith("objective:")):
            raise ValueError(f"alias points at unknown target: {alias!r} -> {target!r}")


def write_fastpack(index_path: Path, output_path: Path) -> dict[str, Any]:
    """Compile and atomically write a JSON fastpack, returning its payload."""

    pack = compile_fastpack(Path(index_path))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(pack, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    fd, temporary = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".partial", dir=output_path.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return pack


def load_fastpack(path: Path) -> dict[str, Any]:
    """Load and validate a previously compiled, network-free fastpack."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"fastpack is not JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("fastpack root must be an object")
    _validate_fastpack(payload)
    return dict(payload)


__all__ = [
    "SCHEMA_VERSION",
    "LEVELS",
    "compile_fastpack",
    "level_growth_multiplier",
    "load_fastpack",
    "normalize_alias",
    "write_fastpack",
]
