"""Natural-language Jinx-versus-turret build optimization.

This is deliberately a small, deterministic search over the patch-pinned
standard Summoner's Rift item packet.  It is not a general combat simulator:
the model only includes structure interactions with a revision-backed rule
receipt (turret mitigation, Jinx's two Q weapons and passive, Hullbreaker,
Trinity Spellblade, Statikk, Stormrazor, Guinsoo's structure-safe attack-speed
stacks, and Demolish).  Other item passives are kept out of the objective
instead of being guessed from their names.

The public entry point accepts an ordinary question and supplies the usual
player defaults.  It returns several state profiles because "best Jinx build
against an inner turret" has different answers for rocket/minigun, passive
state, ability-assisted Spellblade, and a siege rune.
"""

from __future__ import annotations

import itertools
import math
import re
from typing import Any, Mapping, Sequence
from urllib.parse import quote

from .quick_mechanics_fastpack import level_growth_multiplier
from .wiki_rules import STRUCTURES, wiki_rule_source


OPTIMIZER_VERSION = "turret-dps-optimizer-v1.0.0"

_Q_AS_BY_RANK = (0.30, 0.55, 0.80, 1.05, 1.30)
_TURRET_ARMOR = float(STRUCTURES["inner"].get("armor", 60)) if "armor" in STRUCTURES["inner"] else 60.0
_TURRET_MR = 60.0

_LEVEL_RE = re.compile(r"\b(?:level|lvl|lv)\s*(?:=|:|-)?\s*(\d+)\b", re.I)
_Q_RANK_RE = re.compile(
    r"\b(?:q\s*(?:rank|level|lvl|lv)?|rank\s*[-:=]?\s*q?)\s*[-:=]?(\d+)\b",
    re.I,
)
_ITEM_COUNT_RE = re.compile(
    r"\b(\d+|one|two|three|four|five|six)\s*[- ]\s*(?:completed\s+)?items?\b",
    re.I,
)
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


# Revision receipts for the small Wiki rule set.  Static AD/AS/AP/etc. comes
# from CommunityDragon below; these pages explain target filters and trigger
# semantics that the client item packet does not encode.
_RULE_SOURCES: tuple[dict[str, Any], ...] = (
    wiki_rule_source("Turret"),
    {
        "kind": "wiki_rule",
        "url": "https://wiki.leagueoflegends.com/en-us/Template:Data_Jinx/Switcheroo%21",
        "label": "League Wiki Jinx Switcheroo! data",
        "revision_id": 3992483,
        "revision_timestamp": "2026-02-18T17:10:23Z",
        "content_sha256": "a4a11490d120a72ac8364ac1de8c1ea560badaa50f7cc5de3a4a4ab04b10186a",
    },
    {
        "kind": "wiki_rule",
        "url": "https://wiki.leagueoflegends.com/en-us/Template:Data_Jinx/Get_Excited%21",
        "label": "League Wiki Jinx Get Excited! data",
        "revision_id": 3973632,
        "revision_timestamp": "2025-12-11T18:10:58Z",
    },
    {
        "kind": "wiki_rule",
        "url": "https://wiki.leagueoflegends.com/en-us/Demolish",
        "label": "League Wiki Demolish page",
        "revision_id": 4015400,
        "revision_timestamp": "2026-05-04T13:48:51Z",
    },
    {
        "kind": "wiki_rule",
        "url": "https://wiki.leagueoflegends.com/en-us/Template:Rune_data_Lethal_Tempo",
        "label": "League Wiki Lethal Tempo data",
        "revision_id": 3985224,
        "revision_timestamp": "2026-01-18T23:33:28Z",
    },
    {
        "kind": "wiki_rule",
        "url": "https://wiki.leagueoflegends.com/en-us/Template:Energized_info",
        "label": "League Wiki Energized rules",
        "revision_id": 4016840,
        "revision_timestamp": "2026-05-13T01:02:13Z",
    },
    {
        "kind": "wiki_rule",
        "url": "https://wiki.leagueoflegends.com/en-us/Template:Spellblade_info",
        "label": "League Wiki Spellblade rules",
        "revision_id": 4019664,
        "revision_timestamp": "2026-05-18T14:59:49Z",
    },
    {
        "kind": "wiki_rule",
        "url": "https://wiki.leagueoflegends.com/en-us/Critical_strike",
        "label": "League Wiki critical-strike structure rule",
    },
    {
        "kind": "wiki_rule",
        "url": "https://wiki.leagueoflegends.com/en-us/Armor_penetration",
        "label": "League Wiki armor-penetration rule",
    },
    {
        "kind": "wiki_rule",
        "url": "https://wiki.leagueoflegends.com/en-us/Attack_speed",
        "label": "League Wiki attack-speed cap rule",
        "revision_id": 4035691,
        "revision_timestamp": "2026-06-25T03:17:54Z",
    },
)

_ITEM_RECEIPTS: dict[str, dict[str, Any]] = {
    "Hullbreaker": {
        "revision_id": 3943501,
        "revision_timestamp": "2025-08-09T16:46:44Z",
    },
    "Trinity Force": {
        "revision_id": 3982284,
        "revision_timestamp": "2026-01-08T20:47:00Z",
    },
    "Guinsoo's Rageblade": {
        "revision_id": 4024757,
        "revision_timestamp": "2026-06-02T22:31:27Z",
    },
    "Statikk Shiv": {
        "revision_id": 4044336,
        "revision_timestamp": "2026-07-18T20:32:12Z",
    },
    "Stormrazor": {
        "revision_id": 4022962,
        "revision_timestamp": "2026-05-27T20:56:11Z",
    },
    "Lord Dominik's Regards": {
        "revision_id": 3982536,
        "revision_timestamp": "2026-01-09T08:29:38Z",
    },
    "Runaan's Hurricane": {
        "revision_id": 4027997,
        "revision_timestamp": "2026-06-13T11:24:25Z",
    },
    "Rapid Firecannon": {
        "revision_id": 4025120,
        "revision_timestamp": "2026-06-04T21:13:05Z",
    },
}


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _wiki_url(title: str) -> str:
    return "https://wiki.leagueoflegends.com/en-us/" + quote(
        title.replace(" ", "_"), safe="_-\'()!%"
    )


def _copy_source(source: Mapping[str, Any]) -> dict[str, Any]:
    return dict(source)


def _unique_sources(sources: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for source in sources:
        url = str(source.get("url", ""))
        if not url:
            continue
        previous = result.get(url)
        current = dict(source)
        if previous is None or (
            current.get("revision_id") is not None and previous.get("revision_id") is None
        ):
            result[url] = current
    return list(result.values())


def _item_source(item_name: str) -> dict[str, Any]:
    source: dict[str, Any] = {
        "kind": "wiki_item",
        "url": _wiki_url(item_name),
        "label": f"League Wiki {item_name} page",
    }
    source.update(_ITEM_RECEIPTS.get(item_name, {}))
    return source


def _float_stat(item: Mapping[str, Any], field: str) -> float:
    value = (item.get("stats") or {}).get(field)
    if not isinstance(value, Mapping):
        return 0.0
    raw = value.get("value")
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _is_standard_sr_item(item: Mapping[str, Any]) -> bool:
    item_id = int(item.get("id", 0) or 0)
    categories = tuple(item.get("categories", ()))
    name = str(item.get("name", "")).casefold().strip()
    to_ids = tuple(item.get("to_ids", item.get("to", ())) or ())
    return bool(
        item.get("in_store", item.get("inStore", False))
        and item.get("display_in_item_sets", item.get("displayInItemSets", False))
        and (3000 <= item_id <= 3999 or 6000 <= item_id <= 6999)
        and float(item.get("price_total", item.get("priceTotal", 0)) or 0) >= 1000
        # A non-empty ``to`` list identifies an intermediate component in the
        # client packet.  Finished boots are the intentional exception because
        # ranked SR allows their upgrade path as a purchased slot.
        and (not to_ids or "Boots" in categories)
        and name not in {"deprecated item", "obsolete item"}
        and not item.get("required_champion", item.get("requiredChampion"))
        and not item.get("required_ally", item.get("requiredAlly"))
        and not item.get("required_buff_currency_name", item.get("requiredBuffCurrencyName"))
        and not item.get("is_enchantment", item.get("isEnchantment", False))
    )


def _item_key(item: Mapping[str, Any]) -> tuple[str, int]:
    return (_norm(item.get("name", "")), int(item.get("id", 0) or 0))


def _parse_level(question: str, context: Mapping[str, Any] | None) -> tuple[int, bool]:
    if context and isinstance(context.get("level"), (int, float)):
        level = int(context["level"])
        if 1 <= level <= 18:
            return level, True
    match = _LEVEL_RE.search(question)
    if match is not None:
        level = int(match.group(1))
        if 1 <= level <= 18:
            return level, True
    return 18, False


def _parse_q_rank(question: str, level: int, context: Mapping[str, Any] | None) -> tuple[int, bool]:
    if context and isinstance(context.get("q_rank"), (int, float)):
        rank = int(context["q_rank"])
        if 1 <= rank <= 5:
            return rank, True
    # Accept the common player forms: "Q rank 3", "rank-3 Q", and "Q3".
    match = _Q_RANK_RE.search(question)
    if match is not None:
        rank = int(match.group(1))
        if 1 <= rank <= 5:
            return rank, True
    return min(5, (level + 1) // 2), False


def _parse_item_count(question: str, context: Mapping[str, Any] | None) -> int:
    if context and isinstance(context.get("item_count"), (int, float)):
        count = int(context["item_count"])
        if 1 <= count <= 6:
            return count
    match = _ITEM_COUNT_RE.search(question)
    if match is not None:
        raw = match.group(1).casefold()
        count = _NUMBER_WORDS.get(raw, int(raw) if raw.isdigit() else 3)
        if 1 <= count <= 6:
            return count
    return 3


def _parse_form(question: str, context: Mapping[str, Any] | None) -> str:
    if context and str(context.get("q_form", "")).casefold() in {"pow-pow", "powpow", "minigun", "fishbones", "rockets"}:
        raw = str(context["q_form"]).casefold()
    else:
        raw = question.casefold()
    if "fishbones" in raw or "rocket launcher" in raw or "rockets" in raw or "rocket form" in raw:
        return "fishbones"
    if "pow-pow" in raw or "powpow" in raw or "minigun" in raw:
        return "pow-pow"
    return "both"


def _explicit_passive_stacks(question: str, context: Mapping[str, Any] | None) -> int | None:
    if context and isinstance(context.get("passive_stacks"), (int, float)):
        value = int(context["passive_stacks"])
        return max(0, min(5, value))
    if not re.search(r"get\s*excited|passive|stack", question, re.I):
        return None
    if re.search(r"(?:one|1)[ -]?stack", question, re.I):
        return 1
    match = re.search(r"\b([0-5])\s*(?:get[- ]excited\s+)?stacks?\b", question, re.I)
    return int(match.group(1)) if match else None


def _wants_demolish(question: str, context: Mapping[str, Any] | None) -> bool:
    if context and isinstance(context.get("demolish"), bool):
        return bool(context["demolish"])
    if re.search(r"\b(?:without|no)\s+demolish\b", question, re.I):
        return False
    # Demolish is the default siege rune profile for a turret-DPS question;
    # a separate no-Demolish profile is always returned for comparison.
    return True


def _wants_ability_variant(question: str, context: Mapping[str, Any] | None) -> bool:
    if context and isinstance(context.get("ability_assisted"), bool):
        return bool(context["ability_assisted"])
    if re.search(
        r"\b(?:auto(?:-|\s*)attacks?|autos?)\s+only\b|without\s+abilities?",
        question,
        re.I,
    ):
        return False
    # Include the ability-assisted profile by default because it only uses W
    # off-target to prime Spellblade; W itself never damages the turret.
    return True


def looks_like_jinx_turret_build_query(question: str) -> bool:
    """Recognize the natural-language build/DPS contract before generic paths."""

    if not isinstance(question, str):
        return False
    lower = question.casefold()
    return bool(
        re.search(r"\bjinx\b", lower)
        and re.search(r"\b(?:turret|tower)\b", lower)
        and re.search(r"\b(?:dps|damage\s*(?:per\s*second|/\s*s)?|optimal|best|build|items?)\b", lower)
    )


def _candidate_items(engine: Any) -> list[Mapping[str, Any]]:
    items = [item for item in getattr(engine, "_items", {}).values() if _is_standard_sr_item(item)]
    # Deduplicate by name in case a test-injected engine exposes two standard
    # ids for one item.  Stable id/name ordering makes ties reproducible.
    unique: dict[str, Mapping[str, Any]] = {}
    for item in sorted(items, key=_item_key):
        if _norm(item.get("name", "")) in unique:
            continue
        # Parse each visible static stat once.  The search evaluates roughly
        # a million three-item combinations across the returned profiles;
        # keeping this tiny numeric record on the candidate avoids repeating
        # mapping/float work in every combination.
        prepared = dict(item)
        prepared["_optimizer_features"] = {
            "ad": _float_stat(item, "attack_damage"),
            "ap": _float_stat(item, "ability_power"),
            "as": _float_stat(item, "attack_speed") / 100.0,
            "health": _float_stat(item, "health"),
            "lethality": _float_stat(item, "lethality"),
            "armor_pen": _float_stat(item, "armor_penetration") / 100.0,
            "ability_haste": _float_stat(item, "ability_haste"),
        }
        unique[_norm(item.get("name", ""))] = prepared
    return sorted(unique.values(), key=_item_key)


def _build_features(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = {str(item.get("name", "")) for item in items}
    prepared = [item.get("_optimizer_features") for item in items]
    if all(isinstance(value, Mapping) for value in prepared):
        values = [value for value in prepared if isinstance(value, Mapping)]
        ad = sum(float(value.get("ad", 0.0)) for value in values)
        ap = sum(float(value.get("ap", 0.0)) for value in values)
        attack_speed = sum(float(value.get("as", 0.0)) for value in values)
        health = sum(float(value.get("health", 0.0)) for value in values)
        lethality = sum(float(value.get("lethality", 0.0)) for value in values)
        armor_pen = sum(float(value.get("armor_pen", 0.0)) for value in values)
        ability_haste = sum(float(value.get("ability_haste", 0.0)) for value in values)
    else:
        ad = sum(_float_stat(item, "attack_damage") for item in items)
        ap = sum(_float_stat(item, "ability_power") for item in items)
        attack_speed = sum(_float_stat(item, "attack_speed") for item in items) / 100.0
        health = sum(_float_stat(item, "health") for item in items)
        lethality = sum(_float_stat(item, "lethality") for item in items)
        armor_pen = sum(_float_stat(item, "armor_penetration") for item in items) / 100.0
        ability_haste = sum(_float_stat(item, "ability_haste") for item in items)
    return {
        "names": names,
        "ad": ad,
        "ap": ap,
        "as": attack_speed,
        "health": health,
        "lethality": lethality,
        "armor_pen": armor_pen,
        "ability_haste": ability_haste,
        "boots": any("Boots" in tuple(item.get("categories", ())) for item in items),
        "hullbreaker": "Hullbreaker" in names,
        "trinity": "Trinity Force" in names,
        "guinsoo": "Guinsoo's Rageblade" in names,
        "statikk": "Statikk Shiv" in names,
        "stormrazor": "Stormrazor" in names,
        "rfc": "Rapid Firecannon" in names,
    }


def _resistance_multiplier(resistance: float) -> float:
    return 100.0 / (100.0 + max(0.0, resistance))


def _score_build(
    *,
    items: Sequence[Mapping[str, Any]],
    base_ad: float,
    base_health: float,
    base_attack_speed: float,
    level_attack_speed_bonus: float,
    level: int,
    q_form: str,
    q_rank: int,
    passive_stacks: int,
    demolish: bool,
    ability_assisted: bool,
    turret_armor: float = _TURRET_ARMOR,
    turret_mr: float = _TURRET_MR,
) -> dict[str, Any]:
    features = _build_features(items)
    bonus_ad = features["ad"]
    ability_power = features["ap"]
    max_health = base_health + features["health"]
    # Lethality scales from 60% at level 1 to 100% at level 18.  The default
    # level-18 query therefore sees the item's full listed lethality, while an
    # explicit lower-level query does not silently overstate its penetration.
    effective_lethality = features["lethality"] * (0.60 + 0.40 * level / 18.0)

    # Seething Strike is one of the few item stacks audited against
    # structures.  Fully ramping it is the steady-state interpretation.
    stack_attack_speed = 0.32 if features["guinsoo"] else 0.0
    pow_q_bonus = _Q_AS_BY_RANK[q_rank - 1] if q_form == "pow-pow" else 0.0
    total_bonus_as = level_attack_speed_bonus + features["as"] + stack_attack_speed + pow_q_bonus
    if q_form == "fishbones":
        # Fishbones uses 90% of bonus attack speed; it does not retain the
        # minigun's stacked bonus while the rocket launcher is equipped.
        effective_bonus_as = total_bonus_as * 0.90
    else:
        effective_bonus_as = total_bonus_as
    attacks_per_second = base_attack_speed * (1.0 + effective_bonus_as)
    attacks_per_second *= 1.0 + 0.25 * passive_stacks
    if passive_stacks <= 0:
        # The default cap is 3.003 on current SR; Get Excited raises it.
        attacks_per_second = min(3.003, attacks_per_second)

    # The turret basic-attack rule chooses a physical or magic channel based
    # on bonus AD versus AP.  Jinx's rocket modifier is applied to the AD
    # basic attack portion and is relevant to the primary turret target.
    if bonus_ad >= 0.60 * ability_power:
        channel = "physical"
        raw_attack = base_ad + bonus_ad
        if q_form == "fishbones":
            raw_attack *= 1.10
        effective_armor = turret_armor * max(0.0, 1.0 - features["armor_pen"]) - effective_lethality
        attack_multiplier = _resistance_multiplier(effective_armor)
        primary_damage = raw_attack * attack_multiplier
        hull_damage = (2.10 * base_ad + 0.10 * max_health) * attack_multiplier if features["hullbreaker"] else 0.0
        demolish_damage = (85.0 + 0.28 * max_health) * attack_multiplier if demolish else 0.0
        trinity_damage = (2.0 * base_ad) * attack_multiplier if features["trinity"] else 0.0
        magic_multiplier = _resistance_multiplier(turret_mr)
    else:
        channel = "magic"
        raw_attack = 0.60 * ability_power
        attack_multiplier = _resistance_multiplier(turret_mr)
        primary_damage = raw_attack * attack_multiplier
        hull_damage = (2.10 * base_ad + 0.10 * max_health) * _resistance_multiplier(
            turret_armor * max(0.0, 1.0 - features["armor_pen"]) - effective_lethality
        ) if features["hullbreaker"] else 0.0
        demolish_damage = (85.0 + 0.28 * max_health) * _resistance_multiplier(
            turret_armor * max(0.0, 1.0 - features["armor_pen"]) - effective_lethality
        ) if demolish else 0.0
        trinity_damage = (2.0 * base_ad) * _resistance_multiplier(
            turret_armor * max(0.0, 1.0 - features["armor_pen"]) - effective_lethality
        ) if features["trinity"] else 0.0
        magic_multiplier = attack_multiplier

    # Audited energized effects that can hit a structure.  The attack counts
    # are stationary-target steady-state counts: 15 extra Statikk stacks plus
    # 6 normal Energize per attack (7 attacks), and 6 per Stormrazor attack
    # (17 attacks).  RFC is tracked as a source but omitted from the numeric
    # objective until its current structure-target wording is pinned locally.
    energized_per_attack = 0.0
    if features["statikk"]:
        energized_per_attack += 90.0 / 7.0 * magic_multiplier
    if features["stormrazor"]:
        energized_per_attack += 100.0 / 17.0 * magic_multiplier

    hull_per_attack = hull_damage / 5.0
    item_proc_dps = energized_per_attack * attacks_per_second + hull_per_attack * attacks_per_second

    spellblade_dps = 0.0
    spellblade_rate = 0.0
    if ability_assisted and features["trinity"]:
        # Jinx W rank follows the inferred level.  It is cast off-target only;
        # Q's toggle is explicitly not a Spellblade activation.
        w_rank = min(5, (level + 1) // 2)
        w_base_cooldown = (8.0, 7.0, 6.0, 5.0, 4.0)[w_rank - 1]
        w_cooldown = w_base_cooldown / (1.0 + features["ability_haste"] / 100.0)
        spellblade_rate = min(attacks_per_second, 1.0 / 1.5, 1.0 / w_cooldown)
        spellblade_dps = trinity_damage * spellblade_rate

    demolish_dps = demolish_damage / 30.0 if demolish else 0.0
    dps = primary_damage * attacks_per_second + item_proc_dps + spellblade_dps + demolish_dps
    return {
        "dps": dps,
        "attacks_per_second": attacks_per_second,
        "primary_damage": primary_damage,
        "raw_attack": raw_attack,
        "channel": channel,
        "effective_resistance": (
            turret_armor * max(0.0, 1.0 - features["armor_pen"]) - effective_lethality
            if channel == "physical"
            else turret_mr
        ),
        "bonus_ad": bonus_ad,
        "ability_power": ability_power,
        "listed_lethality": features["lethality"],
        "effective_lethality": effective_lethality,
        "max_health": max_health,
        "item_proc_dps": item_proc_dps,
        "hullbreaker_damage_per_proc": hull_damage,
        "hullbreaker_dps": hull_per_attack * attacks_per_second,
        "spellblade_dps": spellblade_dps,
        "spellblade_rate": spellblade_rate,
        "demolish_dps": demolish_dps,
        "demolish_damage": demolish_damage,
        "features": features,
    }


def _sort_key(result: Mapping[str, Any]) -> tuple[float, tuple[str, ...]]:
    return (-float(result["score"]["dps"]), tuple(result["names"]))


def _search(
    *,
    candidates: Sequence[Mapping[str, Any]],
    slots: int,
    score_kwargs: Mapping[str, Any],
    force_boots: bool = False,
) -> tuple[dict[str, Any], int]:
    best: dict[str, Any] | None = None
    evaluated = 0
    for combo in itertools.combinations(candidates, slots):
        if sum("Boots" in tuple(item.get("categories", ())) for item in combo) > 1:
            continue
        if force_boots and not any("Boots" in tuple(item.get("categories", ())) for item in combo):
            continue
        evaluated += 1
        score = _score_build(items=combo, **score_kwargs)
        candidate = {
            "items": combo,
            "names": tuple(sorted(str(item.get("name", "")) for item in combo)),
            "score": score,
        }
        if best is None or _sort_key(candidate) < _sort_key(best):
            best = candidate
    if best is None:
        raise ValueError("no legal standard SR item combination was available")
    return best, evaluated


def _format(value: float, digits: int = 2) -> int | float:
    rounded = round(float(value), digits)
    return int(rounded) if rounded.is_integer() else rounded


def _variant(
    *,
    label: str,
    result: Mapping[str, Any],
    q_form: str,
    q_rank: int,
    passive_stacks: int,
    demolish: bool,
    ability_assisted: bool,
    slot_convention: str,
    item_sources: Sequence[Mapping[str, Any]],
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    score = result["score"]
    features = score["features"]
    names = list(result["names"])
    calculation = (
        f"{score['primary_damage']:.2f} post-mitigation {score['channel']} damage/attack × "
        f"{score['attacks_per_second']:.4f} attacks/s + {score['item_proc_dps']:.2f} audited item-proc DPS"
    )
    additions: list[str] = []
    if score["hullbreaker_damage_per_proc"]:
        additions.append(
            f"Hullbreaker fifth-hit average {score['hullbreaker_damage_per_proc'] / 5:.2f} damage/attack"
        )
    if ability_assisted and score["spellblade_dps"]:
        additions.append(f"Trinity Spellblade {score['spellblade_dps']:.2f} DPS at {score['spellblade_rate']:.4f} procs/s")
    if demolish:
        additions.append(f"Demolish averaged over its 30 s cooldown {score['demolish_dps']:.2f} DPS")
    if additions:
        calculation += "; " + "; ".join(additions)
    return {
        "name": label,
        "build": names,
        "slots": len(names),
        "slot_convention": slot_convention,
        "q_form": q_form,
        "q_rank": q_rank,
        "passive_stacks": passive_stacks,
        "rune_profile": "Demolish (30 s average)" if demolish else "Lethal Tempo shell; no structure damage",
        "ability_assisted": ability_assisted,
        "dps": _format(score["dps"]),
        "attacks_per_second": round(score["attacks_per_second"], 4),
        "damage_per_attack": _format(score["primary_damage"]),
        "item_proc_dps": _format(score["item_proc_dps"]),
        "demolish_dps": _format(score["demolish_dps"]),
        "spellblade_dps": _format(score["spellblade_dps"]),
        "bonus_ad": _format(score["bonus_ad"]),
        "max_health": _format(score["max_health"]),
        "effective_resistance": _format(score["effective_resistance"]),
        "effective_lethality": _format(score["effective_lethality"]),
        "damage_channel": score["channel"],
        "calculation": calculation + ".",
        "notes": list(notes),
        "item_sources": [_copy_source(source) for source in item_sources],
    }


def optimize_jinx_turret(
    engine: Any,
    question: str,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Answer a Jinx inner-turret build question with inferred defaults."""

    level, level_explicit = _parse_level(question, context)
    q_rank, q_rank_explicit = _parse_q_rank(question, level, context)
    item_count = _parse_item_count(question, context)
    form_request = _parse_form(question, context)
    passive_explicit = _explicit_passive_stacks(question, context)
    ability_default = _wants_ability_variant(question, context)
    demolish_default = _wants_demolish(question, context)

    record = next(
        (
            value
            for value in getattr(engine, "_champions", [])
            if str(value.get("name", "")).casefold() == "jinx"
        ),
        None,
    )
    if not isinstance(record, Mapping):
        return {
            "status": "unsupported",
            "intent": "turret_dps_optimization",
            "display": "Jinx is unavailable in the resident patch packet",
            "value": None,
            "reason": "Jinx champion record is absent from the exact patch packet",
        }
    row = (record.get("levels") or {}).get(str(level))
    base_stats = record.get("base_stats") or {}
    if not isinstance(row, Mapping):
        return {
            "status": "unsupported",
            "intent": "turret_dps_optimization",
            "display": "Jinx level data is unavailable in the resident patch packet",
            "value": None,
            "reason": f"level {level} is absent from the exact patch packet",
        }
    base_ad = float(row.get("attack_damage", base_stats.get("base_attack_damage", 0.0)))
    base_health = float(row.get("max_health", base_stats.get("base_health", 0.0)))
    base_attack_speed = float(base_stats.get("attack_speed", 0.625))
    level_attack_speed_bonus = (
        float(base_stats.get("attack_speed_per_level", 1.0))
        / 100.0
        * level_growth_multiplier(level)
    )

    candidates = _candidate_items(engine)
    if len(candidates) < item_count:
        return {
            "status": "unsupported",
            "intent": "turret_dps_optimization",
            "display": "The standard SR item packet has too few legal completed items",
            "value": None,
            "reason": f"need {item_count} legal items, found {len(candidates)}",
        }

    forms = [form_request] if form_request in {"pow-pow", "fishbones"} else ["pow-pow", "fishbones"]
    passive_states = [passive_explicit] if passive_explicit is not None else [0, 1]

    common_kwargs = {
        "base_ad": base_ad,
        "base_health": base_health,
        "base_attack_speed": base_attack_speed,
        "level_attack_speed_bonus": level_attack_speed_bonus,
        "level": level,
    }
    scenarios: list[dict[str, Any]] = []
    combos_evaluated = 0

    # The default headline is a siege profile (W to prime Trinity + Demolish),
    # while the first comparison is the pure repeatable auto profile.  Every
    # scenario is searched independently so a reader can change one state
    # without treating the headline build as universally optimal.
    for form in forms:
        for passive_stacks in passive_states:
            score_kwargs = {
                **common_kwargs,
                "q_form": form,
                "q_rank": q_rank,
                "passive_stacks": passive_stacks,
                "demolish": demolish_default,
                "ability_assisted": ability_default,
            }
            best, evaluated = _search(
                candidates=candidates,
                slots=item_count,
                score_kwargs=score_kwargs,
            )
            combos_evaluated += evaluated
            item_sources = [_item_source(str(item.get("name", "item"))) for item in best["items"]]
            label = f"{form.title()} · Get Excited {passive_stacks} stack · siege profile"
            scenarios.append(
                _variant(
                    label=label,
                    result=best,
                    q_form=form,
                    q_rank=q_rank,
                    passive_stacks=passive_stacks,
                    demolish=demolish_default,
                    ability_assisted=ability_default,
                    slot_convention=f"{item_count} completed items total (boots eligible)",
                    item_sources=item_sources,
                    notes=(
                        "W is cast off-target only to prime Trinity Spellblade; Jinx W/E/R do not damage a turret.",
                        "Demolish is a steady-state 30-second average, not a first-hit timeline.",
                    ),
                )
            )

    # Comparison anchor: no ability priming and no Demolish.  This is often
    # what a player means by raw sustained basic-attack DPS.
    for form in forms:
        best, evaluated = _search(
            candidates=candidates,
            slots=item_count,
            score_kwargs={
                **common_kwargs,
                "q_form": form,
                "q_rank": q_rank,
                "passive_stacks": 0,
                "demolish": False,
                "ability_assisted": False,
            },
        )
        combos_evaluated += evaluated
        scenarios.append(
            _variant(
                label=f"{form.title()} · auto-attacks only · passive inactive",
                result=best,
                q_form=form,
                q_rank=q_rank,
                passive_stacks=0,
                demolish=False,
                ability_assisted=False,
                slot_convention=f"{item_count} completed items total (boots eligible)",
                item_sources=[_item_source(str(item.get("name", "item"))) for item in best["items"]],
                notes=(
                    "No Demolish, no ability-assisted proc, and no critical strikes: basic attacks cannot critically strike structures.",
                ),
            )
        )

    # Explicitly show the common alternative where the player means three
    # non-boots plus a boot slot.  It is a separate contract, not silently
    # mixed into the three-total headline.
    if item_count == 3 and any("Boots" in tuple(item.get("categories", ())) for item in candidates):
        forced_boot_best, forced_boot_evaluated = _search(
            candidates=candidates,
            slots=3,
            force_boots=True,
            score_kwargs={
                **common_kwargs,
                "q_form": "pow-pow",
                "q_rank": q_rank,
                "passive_stacks": 0,
                "demolish": demolish_default,
                "ability_assisted": ability_default,
            },
        )
        combos_evaluated += forced_boot_evaluated
        scenarios.append(
            _variant(
                label="Pow-Pow · siege profile · one boot required",
                result=forced_boot_best,
                q_form="pow-pow",
                q_rank=q_rank,
                passive_stacks=0,
                demolish=demolish_default,
                ability_assisted=ability_default,
                slot_convention="three completed items with exactly one boot slot",
                item_sources=[_item_source(str(item.get("name", "item"))) for item in forced_boot_best["items"]],
                notes=("Boots are forced for this comparison; pure three-slot DPS is allowed to choose three non-boots.",),
            )
        )

    # Keep the six most useful comparisons, with the headline first.  The
    # ordering is deterministic even when two builds tie.
    scenarios.sort(key=lambda value: (-float(value["dps"]), value["name"], tuple(value["build"])))
    primary = scenarios[0]
    # Prefer the requested/default weapon, the initial (zero-stack) inner
    # turret state, and the requested/default siege switches.  The higher
    # Get Excited row remains visible, but it should not silently become the
    # answer to a question that did not say Jinx had already procced it.
    headline_form = forms[0]
    preferred = [
        item
        for item in scenarios
        if item["q_form"] == headline_form
        and item["passive_stacks"] == (passive_explicit if passive_explicit is not None else 0)
        and item["rune_profile"].startswith("Demolish") == demolish_default
        and item["ability_assisted"] == ability_default
    ]
    if preferred:
        primary = preferred[0]
    scenarios = [primary] + [item for item in scenarios if item is not primary]
    scenarios = scenarios[:6]

    source_list: list[Mapping[str, Any]] = [_copy_source(source) for source in _RULE_SOURCES]
    source_list.append(
        {
            "kind": "client",
            "url": f"https://raw.communitydragon.org/{getattr(engine, 'pack', {}).get('client_patch', '')}/raw/game/data/characters/jinx/jinx.bin.json",
            "label": "patch-pinned CommunityDragon Jinx data",
        }
    )
    source_list.append(
        {
            "kind": "client_item",
            "url": f"https://raw.communitydragon.org/{getattr(engine, 'pack', {}).get('client_patch', '')}/plugins/rcp-be-lol-game-data/global/default/v1/items.json",
            "label": "patch-pinned CommunityDragon item data",
        }
    )
    source_list = _unique_sources(source_list + [source for item in scenarios for source in item.get("item_sources", [])])

    assumptions = [
        "Summoner's Rift inner turret; 5,000 health and 60 armor/60 magic resistance.",
        "Outer turret is already destroyed, an allied minion is present, and no backdoor/fortification modifier is active.",
        f"Jinx level {level}{' (inferred)' if not level_explicit else ''}; Q rank {q_rank}{' (inferred)' if not q_rank_explicit else ''}; fully stacked Pow-Pow where that form is evaluated.",
        f"Exactly {item_count} completed standard SR items searched; boots are legal candidates and at most one boot is allowed.",
        "Steady-state DPS: no arbitrary kill-time window; audited stacks are fully ramped and cooldown effects are averaged.",
        "Critical strikes and champion-only on-hit effects do not damage structures; W/E/R are not counted as turret damage.",
        "Default siege profile uses Demolish averaged over 30 seconds and allows off-target W casts to prime Trinity; the auto-only profile is shown separately.",
    ]
    calculation = (
        f"Headline {primary['dps']} DPS = {primary['damage_per_attack']} post-mitigation "
        f"{primary['damage_channel']} damage/attack × {primary['attacks_per_second']} attacks/s, plus audited Hullbreaker/energized/Spellblade/Demolish terms shown in the variant."
    )
    return {
        "status": "available",
        "intent": "turret_dps_optimization",
        "display": f"{primary['dps']} damage per second (initial inner-turret state): {', '.join(primary['build'])} ({primary['q_form']}, {primary['rune_profile']})",
        "value": primary["dps"],
        "unit": "damage per second",
        "patch": getattr(engine, "patch", None),
        "headline": primary,
        "variants": scenarios,
        "defaults": {
            "mode": "summoners_rift",
            "map_target": "inner turret",
            "level": level,
            "q_rank": q_rank,
            "item_count": item_count,
            "boots_in_item_pool": True,
            "turret_health": int(STRUCTURES["inner"]["health"]),
            "turret_armor": 60,
            "turret_magic_resistance": 60,
            "rune_profile": "Demolish siege profile + Lethal Tempo shell (Lethal Tempo contributes zero against turrets)",
        },
        "assumptions": assumptions,
        "calculation": calculation,
        "search": {
            "item_pool_count": len(candidates),
            "combos_evaluated": combos_evaluated,
            "slot_convention": f"{item_count} completed items total including a boot if selected",
            "tie_break": "DPS descending, then normalized item names",
        },
        "unavailable": [
            "First-30-second kill time and Demolish charge timing are not modeled; use the 30-second average for steady-state DPS.",
            "Unreceipted item passives, ally buffs, turret Bulwark, and live minion/fortification state are intentionally excluded.",
        ],
        "provenance": {
            "engine": "lol-oracle-v1",
            "optimizer": OPTIMIZER_VERSION,
            "pack_sha256": getattr(engine, "pack", {}).get("source_hash"),
            "client_patch": getattr(engine, "pack", {}).get("client_patch"),
        },
        "sources": source_list,
    }


__all__ = [
    "OPTIMIZER_VERSION",
    "looks_like_jinx_turret_build_query",
    "optimize_jinx_turret",
]
