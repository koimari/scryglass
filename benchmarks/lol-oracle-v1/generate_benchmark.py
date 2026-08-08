#!/usr/bin/env python3
"""Generate the frozen 500-question League-oracle benchmark.

The benchmark is deliberately split between questions the current local fast
path can answer and questions that should remain explicitly blocked until the
relevant mechanics/source authority exists.  It is a research fixture, not a
claim that every raw client formula is already executable.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lol_kills.knowledge.lol_oracle import LeagueOracleEngine
from lol_kills.knowledge.item_stats import parse_static_item_stats
from lol_kills.knowledge.quick_mechanics_fastpack import compile_fastpack
from lol_kills.knowledge.wiki_rules import (
    STRUCTURES,
    kindred_bonus_range,
    manaflow_bonus,
    nasus_siphoning_strike_bonus,
    senna_mist_stats,
    thresh_soul_stats,
    touch_of_the_void_burn,
    turret_attack_damage,
    wiki_rule_source,
)


PATCH = "26.15"
CLIENT_PATCH = "16.15"
INDEX = ROOT / "data/lol/knowledge/patch-packets/cdragon/2026/26.15/mechanics-index.json"
RAW = INDEX.parent / "raw"
WIKI_ROOT = "https://wiki.leagueoflegends.com/en-us/"
CLIENT_ROOT = f"https://raw.communitydragon.org/{CLIENT_PATCH}/"


def wiki_url(title: str) -> str:
    return WIKI_ROOT + quote(title.replace(" ", "_"), safe="_-'()")


def client_url(path: str) -> str:
    return CLIENT_ROOT + path.lstrip("/")


def unique_champions(pack: dict[str, Any]) -> list[dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for record in pack["champions"].values():
        champion_id = record.get("id")
        alias = str(record.get("alias", ""))
        name = str(record.get("name", ""))
        if not isinstance(champion_id, int) or champion_id <= 0 or champion_id >= 60000:
            continue
        if record.get("status") != "available" or alias.lower().startswith("jade_"):
            continue
        found.setdefault(name, record)
    return sorted(found.values(), key=lambda item: str(item["name"]).lower())


def source_links(record: dict[str, Any], *, extra: Iterable[str] = ()) -> list[dict[str, str]]:
    source = record.get("source") or {}
    links = [
        {
            "kind": "client",
            "url": client_url(str(source.get("bin_json_path", ""))),
            "label": "patch-pinned CommunityDragon client data",
        },
        {
            "kind": "wiki",
            "url": wiki_url(str(record.get("name", "Champion"))),
            "label": "League Wiki champion page",
        },
        {
            "kind": "wiki_formula",
            "url": wiki_url("Champion statistic"),
            "label": "League Wiki stat-growth formula",
        },
    ]
    for url in extra:
        links.append({"kind": "required", "url": url, "label": "required interaction source"})
    return links


def gromp_links() -> list[dict[str, str]]:
    return [
        {
            "kind": "wiki",
            "url": wiki_url("Gromp"),
            "label": "League Wiki Gromp page",
        },
        {
            "kind": "wiki_formula",
            "url": wiki_url("Champion statistic"),
            "label": "League Wiki stat-growth formula",
        },
    ]


def item_links(item_name: str) -> list[dict[str, str]]:
    return [
        {
            "kind": "client",
            "url": CLIENT_ROOT + "plugins/rcp-be-lol-game-data/global/default/v1/items.json",
            "label": "patch-pinned CommunityDragon item data",
        },
        {
            "kind": "wiki",
            "url": wiki_url(item_name),
            "label": "League Wiki item page",
        },
    ]


def add(
    rows: list[dict[str, Any]],
    *,
    difficulty: str,
    domain: str,
    question: str,
    target_status: str,
    baseline_status: str,
    sources: list[dict[str, str]],
    value: Any = None,
    unit: str | None = None,
    calculation: str | None = None,
    blocker: str | None = None,
) -> None:
    row: dict[str, Any] = {
        "id": f"{difficulty[:1].upper()}-{sum(1 for item in rows if item['difficulty'] == difficulty) + 1:03d}",
        "difficulty": difficulty,
        "domain": domain,
        "question": question,
        "target": {
            "status": target_status,
            "value": value,
            "unit": unit,
            "exact_required": target_status == "available",
        },
        "baseline": {
            "engine": "lol-oracle-v1",
            "expected_status": baseline_status,
        },
        "sources": sources,
    }
    if calculation:
        row["calculation"] = calculation
    if blocker:
        row["blocker"] = blocker
    rows.append(row)


def generate() -> list[dict[str, Any]]:
    pack = compile_fastpack(INDEX)
    oracle = LeagueOracleEngine(pack, raw_champion_root=RAW / "champions")
    champions = unique_champions(pack)
    mana_champions = [
        record for record in champions if record.get("resource_type") == "mana"
    ]
    if len(champions) < 50 or len(mana_champions) < 20:
        raise RuntimeError("the exact patch packet does not contain enough champions")
    rows: list[dict[str, Any]] = []

    # Easy: one exact fact lookup, with the standard level-growth calculation
    # already materialized in the patch fastpack.
    easy_specs = [
        ("attack_damage", "base attack damage", "attack damage", "AD"),
        ("magic_resist", "base magic resistance", "magic resist", "MR"),
    ]
    for i in range(100):
        if i % 3 == 2:
            record = mana_champions[i % len(mana_champions)]
            field, phrase, unit, label = "mp5", "mana regeneration", "mana per 5 seconds", "MP5"
        else:
            record = champions[i % len(champions)]
            field, phrase, unit, label = easy_specs[i % len(easy_specs)]
        level = 1 + ((i * 7) % 18)
        raw_value = record["levels"][str(level)].get(field)
        value = round(float(raw_value), 2)
        add(
            rows,
            difficulty="easy",
            domain="champion_stats",
            question=f"What is {record['name']}'s {phrase} at level {level}?",
            target_status="available",
            baseline_status="available",
            sources=source_links(record),
            value=value,
            unit=unit,
            calculation=f"Read {label} from the exact {PATCH} level-{level} stat row.",
        )

    # Medium: two or more deterministic operations, still using source cells
    # whose semantics are already validated by the current packet.
    medium_stats = [("attack_damage", "attack damage"), ("magic_resist", "magic resist"), ("max_health", "maximum health")]
    for i in range(25):
        record = champions[i % len(champions)]
        field, label = medium_stats[i % len(medium_stats)]
        low = 1 + (i % 6)
        high = low + 5 + (i % 7)
        value = round(float(record["levels"][str(high)][field]) - float(record["levels"][str(low)][field]), 2)
        add(
            rows,
            difficulty="medium",
            domain="stat_delta",
            question=f"How much {label} does {record['name']} gain from level {low} to level {high}?",
            target_status="available",
            baseline_status="available",
            sources=source_links(record),
            value=value,
            unit=label,
            calculation=f"{record['name']} level {high} {label} minus level {low} {label}.",
        )

    for i in range(25):
        record = mana_champions[(i * 3) % len(mana_champions)]
        level = 3 + (i % 16)
        seconds = 10 + (i % 8) * 5
        mp5 = float(record["levels"][str(level)]["mp5"])
        value = round(mp5 * seconds / 5.0, 2)
        add(
            rows,
            difficulty="medium",
            domain="resource_regeneration",
            question=f"Ignoring spending, how much mana does {record['name']} regenerate in {seconds} seconds at level {level}?",
            target_status="available",
            baseline_status="available",
            sources=source_links(record),
            value=value,
            unit="mana",
            calculation=f"{mp5:.2f} MP5 × {seconds}/5 seconds.",
        )

    gromp = pack["monsters"]["gromp"]
    for i in range(25):
        record = champions[(i * 5) % len(champions)]
        attacker_level = 1 + (i % 18)
        target_level = 1 + ((i * 2 + 3) % 18)
        ad = float(record["levels"][str(attacker_level)]["ad"])
        hp = float(gromp["levels"][str(target_level)]["hp"])
        armor = float(gromp["levels"][str(target_level)]["armor"])
        per_attack = ad * (100.0 / (100.0 + armor))
        attacks = int(__import__("math").ceil(hp / per_attack))
        add(
            rows,
            difficulty="medium",
            domain="basic_attack_mitigation",
            question=f"How many itemless basic attacks does {record['name']} at level {attacker_level} need to kill a level {target_level} Gromp?",
            target_status="available",
            baseline_status="available",
            sources=source_links(record, extra=[wiki_url("Gromp")]),
            value=attacks,
            unit="auto attacks",
            calculation=f"ceil({hp:.2f} HP / ({ad:.2f} AD × 100/(100+{armor:.2f} armor))).",
        )

    for i in range(25):
        record = mana_champions[(i * 7) % len(mana_champions)]
        low = 1 + (i % 5)
        high = low + 5 + (i % 8)
        value = round(float(record["levels"][str(high)]["max_resource"]) - float(record["levels"][str(low)]["max_resource"]), 2)
        add(
            rows,
            difficulty="medium",
            domain="resource_growth",
            question=f"How much maximum mana does {record['name']} gain between levels {low} and {high}?",
            target_status="available",
            baseline_status="available",
            sources=source_links(record),
            value=value,
            unit="mana",
            calculation=f"Level {high} maximum mana minus level {low} maximum mana.",
        )

    # Hard: exact-looking ability/item/rune questions.  The direct-damage
    # subset is promoted only when the same resident kernel can evaluate the
    # pinned graph; richer item/rune questions remain blocked.
    spell_records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for record in champions:
        raw_path = RAW / "champions" / f"{record['id']}.json"
        if not raw_path.is_file():
            continue
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        for spell in raw.get("spells", []):
            if isinstance(spell, dict) and spell.get("spellKey") in {"q", "w", "e", "r"}:
                spell_records.append((record, spell))
    if not spell_records:
        raise RuntimeError("no raw champion spell records found")

    mana_spell_records = [
        (record, spell)
        for record, spell in spell_records
        if record.get("resource_type") == "mana"
        and isinstance(spell.get("costCoefficients"), list)
        and any(isinstance(cost, (int, float)) and float(cost) > 0 for cost in spell["costCoefficients"])
    ]
    if len(mana_spell_records) < 25:
        raise RuntimeError("not enough mana spell records for the hard budget benchmark")

    for i in range(25):
        record, spell = mana_spell_records[(i * 7) % len(mana_spell_records)]
        rank = 1 + (i % 5)
        level = max(1, min(18, rank * 2 + 1))
        costs = spell["costCoefficients"]
        if rank > len(costs) or not isinstance(costs[rank - 1], (int, float)) or float(costs[rank - 1]) <= 0:
            rank = 1
        cost = float(costs[rank - 1])
        maximum = float(record["levels"][str(level)]["max_resource"])
        casts = int(maximum // cost)
        add(
            rows,
            difficulty="hard",
            domain="ability_resource_budget",
            question=f"At level {level}, how many rank-{rank} casts of {record['name']}'s {spell['name']} can be made from full resource with no regeneration?",
            target_status="available",
            baseline_status="available",
            sources=source_links(record, extra=[wiki_url(f"{record['name']}/{spell['name']}")]),
            value=casts,
            unit="casts",
            calculation=f"floor({maximum:.2f} maximum mana / {cost:.2f} rank-{rank} cost).",
        )

    damage_specs: list[tuple[dict[str, Any], dict[str, Any], int, int, dict[str, Any]]] = []
    physical_specs: list[tuple[dict[str, Any], dict[str, Any], int, int, int, dict[str, Any]]] = []
    for i in range(25):
        record, spell = spell_records[(i * 11 + 3) % len(spell_records)]
        rank = 1 + (i % 5)
        ap = 50 + (i % 10) * 25
        question = f"What exact damage does {record['name']}'s rank-{rank} {spell['spellKey'].upper()} deal with {ap} AP against a target with no modifiers?"
        calculated = oracle.answer(question)
        available = calculated.get("status") == "available"
        if available:
            damage_specs.append((record, spell, rank, ap, calculated))

    for record, spell in spell_records:
        mechanics_spell = oracle._mechanics_spell(record, str(spell.get("spellKey", "")).upper())  # type: ignore[attr-defined]
        if not isinstance(mechanics_spell, dict) or not isinstance(mechanics_spell.get("bot_data"), dict) or mechanics_spell["bot_data"].get("DamageTag") != 0:
            continue
        for rank in range(1, 6):
            level = 6 + (len(physical_specs) % 10)
            total_ad = 80 + (len(physical_specs) % 8) * 25
            question = (
                f"What exact raw damage does {record['name']}'s rank-{rank} "
                f"{spell['spellKey'].upper()} deal with {total_ad} total AD at level {level} "
                "against a target with no modifiers?"
            )
            calculated = oracle.answer(question)
            if calculated.get("status") == "available":
                physical_specs.append((record, spell, rank, total_ad, level, calculated))
                break
            if len(physical_specs) >= 25:
                break
        if len(physical_specs) >= 25:
            break

    for i in range(25):
        # Keep five deliberately blocked formula examples as a regression
        # guard, and use the accepted AP/AD subsets for the other rows.
        original_record, original_spell = spell_records[(i * 11 + 3) % len(spell_records)]
        original_rank = 1 + (i % 5)
        original_ap = 50 + (i % 10) * 25
        original_question = f"What exact damage does {original_record['name']}'s rank-{original_rank} {original_spell['spellKey'].upper()} deal with {original_ap} AP against a target with no modifiers?"
        if i % 5 == 0:
            calculated = oracle.answer(original_question)
            record, spell, question = original_record, original_spell, original_question
        elif i % 2 == 0 and damage_specs:
            record, spell, rank, ap, calculated = damage_specs[(i // 2) % len(damage_specs)]
            question = f"What exact damage does {record['name']}'s rank-{rank} {spell['spellKey'].upper()} deal with {ap} AP against a target with no modifiers?"
            calculated = oracle.answer(question)
        elif physical_specs:
            record, spell, rank, total_ad, level, calculated = physical_specs[(i // 2) % len(physical_specs)]
            question = (
                f"What exact raw damage does {record['name']}'s rank-{rank} {spell['spellKey'].upper()} "
                f"deal with {total_ad} total AD at level {level} against a target with no modifiers?"
            )
            calculated = oracle.answer(question)
        else:
            calculated = oracle.answer(original_question)
            record, spell, question = original_record, original_spell, original_question
        available = calculated.get("status") == "available"
        add(
            rows,
            difficulty="hard",
            domain="single_ability_damage",
            question=question,
            target_status="available" if available else "blocked",
            baseline_status="available" if available else "unsupported",
            sources=source_links(record, extra=[wiki_url(f"{record['name']}/{spell['name']}")]),
            value=calculated.get("value") if available else None,
            unit=calculated.get("unit") if available else None,
            calculation=calculated.get("calculation") if available else None,
            blocker=None if available else str(calculated.get("reason") or "direct damage graph is not executable by the narrow kernel"),
        )

    item_payload = json.loads((RAW / "items.json").read_text(encoding="utf-8"))
    static_item_rows: list[tuple[dict[str, Any], str, str, float]] = []
    item_labels = (
        ("attack_damage", "attack damage"),
        ("ability_power", "ability power"),
        ("health", "maximum health"),
        ("mana", "maximum mana"),
        ("armor", "armor"),
        ("magic_resist", "magic resist"),
    )
    for item in item_payload:
        if not isinstance(item, dict) or not isinstance(item.get("id"), int):
            continue
        if item["id"] >= 10000 or not item.get("inStore") or not item.get("name"):
            continue
        if re.search(r"<(?:passive|unique|active)", str(item.get("description", "")), re.I):
            continue
        parsed = parse_static_item_stats(item)
        for field, label in item_labels:
            stat = parsed.get(field)
            if isinstance(stat, dict) and not stat.get("percent") and isinstance(stat.get("value"), (int, float)):
                static_item_rows.append((item, field, label, float(stat["value"])))
    static_item_rows.sort(key=lambda row: (str(row[0]["name"]).casefold(), int(row[0]["id"]), row[1]))
    if len(static_item_rows) < 25:
        raise RuntimeError("not enough passive-free static item-stat rows")
    for i in range(25):
        item, field, label, item_value = static_item_rows[(i * 7) % len(static_item_rows)]
        record = champions[(i * 9) % len(champions)]
        level = 1 + i % 18
        champion_field = {
            "attack_damage": "attack_damage",
            "health": "max_health",
            "mana": "max_resource",
            "armor": "armor",
            "magic_resist": "magic_resist",
        }.get(field)
        base = 0.0 if champion_field is None else float(record["levels"][str(level)][champion_field])
        value = round(base + item_value, 2)
        add(
            rows,
            difficulty="hard",
            domain="item_champion_interaction",
            question=f"With {item['name']} equipped, what is {record['name']}'s exact level-{level} {label}?",
            target_status="available",
            baseline_status="available",
            sources=item_links(str(item["name"])) + source_links(record)[:2],
            value=value,
            unit=label,
            calculation=f"{base:.2f} champion {label} + {item_value:.2f} {item['name']} stat.",
        )

    items = sorted(
        [
            item
            for item in item_payload
            if isinstance(item, dict) and item.get("name") and item.get("inStore")
        ],
        key=lambda item: (str(item["name"]).casefold(), int(item.get("id", 0))),
    )

    for i in range(25):
        # Manaflow's permanent +25-per-stack component is now an exact,
        # source-receipted rune slice.  Trigger timing and post-cap regen stay
        # outside the contract; every row states the already-collected stack
        # count explicitly.
        record = mana_champions[(i * 4 + 5) % len(mana_champions)]
        level = 1 + i % 18
        stacks = i % 11
        base = float(record["levels"][str(level)]["max_resource"])
        value = round(base + manaflow_bonus(stacks), 2)
        add(
            rows,
            difficulty="hard",
            domain="rune_interaction",
            question=(
                f"With Manaflow Band at {stacks} stacks, what is {record['name']}'s "
                f"exact maximum mana at level {level}?"
            ),
            target_status="available",
            baseline_status="available",
            sources=source_links(record) + [wiki_rule_source("Manaflow Band")],
            value=value,
            unit="maximum mana",
            calculation=f"{base:.2f} base maximum mana + ({min(stacks, 10)} × 25) Manaflow bonus.",
        )

    for i in range(25):
        health = 1000 + i * 25
        armor = 20 + i * 2
        magic_resist = 30 + i * 3
        physical = 90 + i * 5
        magic = 120 + i * 4
        true = 35 + i
        physical_dealt = physical * 100.0 / (100.0 + armor)
        magic_dealt = magic * 100.0 / (100.0 + magic_resist)
        value = round(health - physical_dealt - magic_dealt - true, 2)
        add(
            rows,
            difficulty="complex",
            domain="ordered_combat_sequence",
            question=(
                f"A target starts with {health} health, {armor} armor, and {magic_resist} magic resistance. "
                f"In exact order, it takes {physical} physical damage, then {magic} magic damage, "
                f"then {true} true damage. What remaining health does it have?"
            ),
            target_status="available",
            baseline_status="available",
            sources=[
                {"kind": "wiki", "url": wiki_url("Damage"), "label": "League Wiki damage rules"},
                {"kind": "wiki", "url": wiki_url("Armor"), "label": "League Wiki armor rules"},
                {"kind": "wiki", "url": wiki_url("Magic resistance"), "label": "League Wiki magic-resistance rules"},
            ],
            value=value,
            unit="remaining health",
            calculation=(
                f"{health} − {physical}×100/(100+{armor}) − {magic}×100/(100+{magic_resist}) − {true} "
                f"= {value:.2f}; events execute in the stated order."
            ),
        )

    for i in range(25):
        if not damage_specs:
            raise RuntimeError("no direct magic-damage rows are available for mitigation benchmark")
        a, spell, rank, ap, _ = damage_specs[i % len(damage_specs)]
        magic_resist = 20 + i * 3
        question = (
            f"What exact post-mitigation damage does {a['name']}'s rank-{rank} "
            f"{spell['spellKey'].upper()} deal with {ap} AP against a target with "
            f"{magic_resist} magic resistance and no penetration?"
        )
        calculated = oracle.answer(question)
        available = calculated.get("status") == "available"
        add(
            rows,
            difficulty="complex",
            domain="direct_magic_mitigation",
            question=question,
            target_status="available" if available else "blocked",
            baseline_status="available" if available else "unsupported",
            sources=source_links(a, extra=[wiki_url(f"{a['name']}/{spell['name']}"), wiki_url("Magic resistance")]),
            value=calculated.get("value") if available else None,
            unit=calculated.get("unit") if available else None,
            calculation=calculated.get("calculation") if available else None,
            blocker=None if available else str(calculated.get("reason") or "direct magic mitigation is not executable by the narrow kernel"),
        )

    structure_kinds = ("outer", "inner", "inhibitor", "nexus")
    for i in range(25):
        kind = structure_kinds[i % len(structure_kinds)]
        variant = i % 3
        turret_source = wiki_rule_source("Turret")
        if variant == 0:
            value = int(STRUCTURES[kind]["health"])
            question = f"At 0:00, how much health does the {kind} turret have?"
            unit = "health"
            calculation = f"{kind.title()} turret base health = {value}."
            sources = [turret_source, wiki_rule_source("Turret Plating")]
        elif variant == 1:
            # Choose clocks from the exact ramp/cap ranges in the page's
            # infoboxes rather than inventing an interpolation point.
            first = int(STRUCTURES[kind]["attack_first_second"])
            cap = int(STRUCTURES[kind]["attack_cap"])
            step = int(STRUCTURES[kind]["attack_step"])
            increments = max(0, min((cap - int(STRUCTURES[kind]["base_attack_damage"])) // step, 1 + (i % 15)))
            seconds = 0 if increments == 0 else first + (increments - 1) * 60
            value = turret_attack_damage(kind, seconds)
            question = f"At {seconds // 60}:{seconds % 60:02d}, what attack damage does the {kind} turret have?"
            unit = "attack damage"
            calculation = f"{kind.title()} turret attack damage at {seconds // 60}:{seconds % 60:02d} = {value}."
            sources = [turret_source, wiki_rule_source("Turret Plating")]
        else:
            plates = 1 + i % 5
            value = plates * 120
            question = f"How much local gold do {plates} turret plates grant?"
            unit = "local gold"
            calculation = f"{plates} plates × 120 local gold per plate = {value}."
            sources = [turret_source, wiki_rule_source("Turret Plating")]
        add(
            rows,
            difficulty="complex",
            domain="structure_and_objective_timing",
            question=question,
            target_status="available",
            baseline_status="available",
            sources=sources,
            value=value,
            unit=unit,
            calculation=calculation,
        )

    for i in range(25):
        variant = i % 5
        if variant == 0:
            rank = 1 + i % 5
            stacks = (i * 17) % 201
            value = nasus_siphoning_strike_bonus(rank, stacks)
            question = f"After {stacks} Q stacks, what bonus physical damage does Nasus's rank-{rank} Siphoning Strike deal?"
            sources = [
                wiki_rule_source("Template:Data Nasus/Siphoning Strike"),
                {"kind": "wiki", "url": wiki_url("Nasus"), "label": "League Wiki Nasus page"},
            ]
            unit = "bonus physical damage"
            calculation = f"{40 + 20 * (rank - 1)} rank-{rank} base + {stacks} Q stacks = {value}."
        elif variant == 1:
            stacks = 5 + (i * 7) % 90
            ap, armor = thresh_soul_stats(stacks)
            request_armor = i % 2 == 1
            value = armor if request_armor else ap
            unit = "bonus armor" if request_armor else "ability power"
            question = f"After {stacks} Thresh soul stacks, how much {unit} does Thresh gain?"
            sources = [
                wiki_rule_source("Template:Data Thresh/Damnation"),
                {"kind": "wiki", "url": wiki_url("Thresh"), "label": "League Wiki Thresh page"},
            ]
            calculation = f"{stacks} souls × 1 {unit} per soul = {value}."
        elif variant == 2:
            stacks = 10 + (i * 9) % 91
            ad, _, _ = senna_mist_stats(stacks)
            value = round(ad, 2)
            question = f"After {stacks} Senna Mist stacks, how much bonus attack damage does Senna gain?"
            sources = [
                wiki_rule_source("Template:Data Senna/Absolution"),
                {"kind": "wiki", "url": wiki_url("Senna"), "label": "League Wiki Senna page"},
            ]
            unit = "bonus attack damage"
            calculation = f"{stacks} Mist × 0.75 bonus AD = {value:.2f}."
        elif variant == 3:
            stacks = i % 26
            value = kindred_bonus_range(stacks)
            question = f"After {stacks} Kindred marks, how much bonus attack range does Kindred gain?"
            sources = [
                wiki_rule_source("Template:Data Kindred/Mark of the Kindred"),
                {"kind": "wiki", "url": wiki_url("Kindred"), "label": "League Wiki Kindred page"},
            ]
            unit = "bonus attack range"
            calculation = f"Kindred mark table at {stacks} marks = {value} bonus range."
        else:
            stacks = 1 + i % 3
            ranged = i % 2 == 1
            _, value = touch_of_the_void_burn(stacks, ranged=ranged)
            question = f"After {stacks} Touch of the Void stacks, what total true damage does a {'ranged' if ranged else 'melee'} attack burn a structure for over 4 seconds?"
            sources = [wiki_rule_source("Template:Buff data Touch of the Void"), wiki_rule_source("Touch of the Void")]
            unit = "true damage over 4 seconds"
            per_tick, _ = touch_of_the_void_burn(stacks, ranged=ranged)
            calculation = f"{per_tick} true damage per 0.5-second tick × 8 ticks = {value}."
        add(
            rows,
            difficulty="complex",
            domain="transformation_and_stacking",
            question=question,
            target_status="available",
            baseline_status="available",
            sources=sources,
            value=value,
            unit=unit,
            calculation=calculation,
        )

    # Impossible: the correct oracle behavior is to explain the missing or
    # contradictory authority rather than invent a number.
    for i in range(25):
        record = champions[i % len(champions)]
        add(
            rows,
            difficulty="impossible",
            domain="missing_patch",
            question=f"How much damage does {record['name']} deal with the current build?",
            target_status="blocked",
            baseline_status="unsupported",
            sources=[
                {"kind": "required", "url": wiki_url(record["name"]), "label": "champion source requires a patch"},
                {"kind": "required", "url": wiki_url("Patch history"), "label": "patch history required"},
            ],
            blocker="No patch, level, rank, target, items, runes, or build is specified.",
        )

    for i in range(25):
        record = champions[(i * 2 + 1) % len(champions)]
        add(
            rows,
            difficulty="impossible",
            domain="ambiguous_entities",
            question=f"Does {record['name']} win the fight if the enemy is stronger?",
            target_status="blocked",
            baseline_status="unsupported",
            sources=[
                {"kind": "wiki", "url": wiki_url(record["name"]), "label": "champion page"},
                {"kind": "wiki", "url": wiki_url("Champion statistic"), "label": "stat definitions"},
            ],
            blocker="The entities, strengths, win condition, and scenario are undefined; this is not an exact calculation.",
        )

    for i in range(25):
        record = champions[(i * 3 + 2) % len(champions)]
        add(
            rows,
            difficulty="impossible",
            domain="hidden_or_outcome_state",
            question=f"What exact damage would {record['name']} have dealt in the next 10 seconds if the opponent had dodged?",
            target_status="blocked",
            baseline_status="unsupported",
            sources=[
                {"kind": "wiki", "url": wiki_url(record["name"]), "label": "champion mechanics source"},
                {"kind": "required", "url": wiki_url("Game time"), "label": "missing event timeline"},
            ],
            blocker="The counterfactual depends on an unobserved event sequence and unspecified state.",
        )

    for i in range(25):
        record = champions[(i * 5 + 3) % len(champions)]
        add(
            rows,
            difficulty="impossible",
            domain="nonexistent_or_cross_mode_rule",
            question=f"On Summoner's Rift patch 99.99, what is the exact Arena-only augment multiplier for {record['name']}?",
            target_status="blocked",
            baseline_status="unsupported",
            sources=[
                {"kind": "required", "url": wiki_url(record["name"]), "label": "champion page"},
                {"kind": "required", "url": wiki_url("Arena"), "label": "mode-specific source"},
            ],
            blocker="The requested patch is nonexistent and the map/mode rules contradict each other.",
        )

    expected = {difficulty: 100 for difficulty in ("easy", "medium", "hard", "complex", "impossible")}
    counts = {difficulty: 0 for difficulty in expected}
    for row in rows:
        counts[row["difficulty"]] += 1
    if counts != expected or len(rows) != 500:
        raise AssertionError(f"benchmark stratification failed: {counts}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("questions.jsonl"))
    args = parser.parse_args()
    rows = generate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"questions": len(rows), "output": str(args.output), "patch": PATCH}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
