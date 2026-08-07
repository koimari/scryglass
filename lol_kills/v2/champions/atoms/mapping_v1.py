"""Curated atom -> ontology mapping (the knowledge bridge), version 1.

Every row maps one LCC atom id (or family-level fallback) to a v2 ontology
(dimension, label) contribution.  Weights are tiers used to turn raw atom
counts into a soft, mechanistic prior:

    weight 1.0  -> strong semantic signal (the atom mostly *is* that label)
    weight 0.5  -> clear supporting signal
    weight 0.25 -> weak/contextual signal

The table is curated from LCC atom definitions (wiki-atoms vocabularies,
atom-relations graph, and the champion atom samples under data/atoms in the
LCC repo).  It is a *prior generator*, not a measurement: consumers must not
present atom-derived scores as empirical or review-accepted ontology values.

Family-level fallbacks apply to champions for which only atom-family presence
is available (LCC atom-summary).  Atom-level rows apply when per-champion
atom detail is available.
"""

from __future__ import annotations

from typing import Any

# (atom_id | family:<family>, dimension, label, weight, evidence_note)
_MAPPING_ROWS: list[tuple[str, str, str, float, str]] = [
    # --- damage family -----------------------------------------------------
    ("damage.ability", "damage_profile", "burst", 0.4, "ability damage concentrates on spell burst"),
    ("damage.aoe", "damage_profile", "teamfight", 0.6, "area damage scales with clumped fights"),
    ("damage.aoe", "wave_control", "push", 0.3, "area damage clears waves"),
    ("damage.bonus", "damage_profile", "burst", 0.25, "bonus damage adds burst tail"),
    ("damage.burn", "damage_profile", "poke", 0.5, "burn damage applies at range over time"),
    ("damage.burn", "damage_profile", "teamfight", 0.25, "burn contributes in extended fights"),
    ("damage.critical-strike", "damage_profile", "burst", 0.5, "crits concentrate damage into spikes"),
    ("damage.dot", "damage_profile", "poke", 0.6, "damage-over-time enables ranged attrition"),
    ("damage.dot", "damage_profile", "teamfight", 0.3, "DoT adds value in extended fights"),
    ("damage.execute", "damage_profile", "burst", 0.7, "execute is a finishing burst tool"),
    ("damage.on-hit", "damage_profile", "teamfight", 0.4, "on-hit damage accrues in sustained fights"),
    ("damage.on-hit", "scaling", "mid", 0.25, "on-hit builds scale with attack speed"),
    ("damage.pet", "damage_profile", "teamfight", 0.4, "pets add a persistent damage channel"),
    ("damage.pet", "wave_control", "push", 0.3, "pets push waves"),
    ("damage.reduction", "durability_frontline", "tank", 0.4, "damage reduction is a tank trait"),
    ("damage.reduction", "peel", "defensive", 0.25, "damage reduction protects allies in some kits"),
    # --- crowd-control-mobility family -------------------------------------
    ("crowd-control-mobility.airborne", "crowd_control", "stun", 0.6, "airborne is a hard crowd control"),
    ("crowd-control-mobility.airborne", "engage", "hard", 0.3, "airborne starts fights"),
    ("crowd-control-mobility.cc-immunity", "peel", "defensive", 0.5, "CC immunity protects the carry"),
    ("crowd-control-mobility.cc-immunity", "durability_frontline", "tank", 0.25, "CC immunity helps front-to-back"),
    ("crowd-control-mobility.dash", "mobility", "high", 0.6, "dashes are the core mobility currency"),
    ("crowd-control-mobility.dash", "target_access", "reposition", 0.4, "dashes reposition onto targets"),
    ("crowd-control-mobility.dash", "engage", "single", 0.25, "dashes enable single-target engages"),
    ("crowd-control-mobility.immobilize", "crowd_control", "stun", 0.5, "immobilize is a hard CC class"),
    ("crowd-control-mobility.knockback", "crowd_control", "stun", 0.5, "knockback is a hard CC class"),
    ("crowd-control-mobility.knockback", "peel", "forced", 0.5, "knockback pushes threats off allies"),
    ("crowd-control-mobility.movement-speed", "mobility", "moderate", 0.4, "movement speed is moderate mobility"),
    ("crowd-control-mobility.root", "crowd_control", "root", 0.7, "root is the root label's defining atom"),
    ("crowd-control-mobility.slow", "crowd_control", "slow", 0.7, "slow is the slow label's defining atom"),
    ("crowd-control-mobility.stun", "crowd_control", "stun", 0.7, "stun is the stun label's defining atom"),
    ("crowd-control-mobility.tenacity", "peel", "defensive", 0.4, "tenacity reduces CC windows"),
    ("crowd-control-mobility.tenacity", "durability_frontline", "tank", 0.25, "tenacity supports frontlines"),
    ("crowd-control-mobility.untargetable", "peel", "defensive", 0.4, "untargetable evades damage"),
    ("crowd-control-mobility.untargetable", "durability_frontline", "duelist", 0.3, "untargetable is a duelist survivability tool"),
    # --- heal-shield family -------------------------------------------------
    ("heal-shield.heal", "sustain", "self_heal", 0.7, "healing is self-sustain"),
    ("heal-shield.revive", "sustain", "self_heal", 0.3, "revive restores the champion"),
    ("heal-shield.revive", "durability_frontline", "tank", 0.25, "revive makes frontlines stickier"),
    ("heal-shield.shield", "sustain", "shield", 0.7, "shielding is the shield label's defining atom"),
    # --- interaction family -------------------------------------------------
    ("interaction.attack-reset", "damage_profile", "burst", 0.5, "attack resets compress damage into burst windows"),
    # --- stack/transform/summon family -------------------------------------
    ("stack-transform-summon-resource.additive-stacking", "scaling", "late", 0.5, "infinite stacking scales into the late game"),
    ("stack-transform-summon-resource.clone", "target_access", "reposition", 0.25, "clones create target ambiguity"),
    ("stack-transform-summon-resource.clone", "damage_profile", "burst", 0.25, "clone bursts duplicate damage windows"),
    ("stack-transform-summon-resource.health-as-resource", "durability_frontline", "duelist", 0.4, "health-as-resource suits duelists"),
    ("stack-transform-summon-resource.health-as-resource", "sustain", "drain", 0.3, "health-as-resource pairs with drain"),
    ("stack-transform-summon-resource.mark", "damage_profile", "burst", 0.3, "marks set up burst follow-through"),
    ("stack-transform-summon-resource.soul", "scaling", "late", 0.3, "soul collection is a stacking scale path"),
    ("stack-transform-summon-resource.stack", "scaling", "late", 0.5, "stacks grow into the late game"),
    ("stack-transform-summon-resource.summon", "damage_profile", "teamfight", 0.4, "summons add persistent teamfight damage"),
    ("stack-transform-summon-resource.summon", "wave_control", "push", 0.4, "summons push waves"),
    ("stack-transform-summon-resource.summoned-terrain-structure", "wave_control", "siege", 0.4, "summoned terrain shapes siege"),
    ("stack-transform-summon-resource.toggle-form", "tempo", "mid_pressure", 0.25, "toggle forms create mid-game pressure windows"),
    ("stack-transform-summon-resource.transform", "scaling", "late_spike", 0.4, "transform kits spike when forms unlock"),
    ("stack-transform-summon-resource.upgrade", "scaling", "late", 0.3, "kit upgrades scale across the game"),
    # --- vision-economy family ---------------------------------------------
    ("vision-economy.stealth", "target_access", "reposition", 0.5, "stealth repositions without vision"),
    ("vision-economy.stealth", "engage", "single", 0.25, "stealth enables pick engages"),
]

# Family-level fallbacks (used when only family presence is known).
_FAMILY_FALLBACKS: list[tuple[str, str, str, float, str]] = [
    ("family:damage", "damage_profile", "teamfight", 0.15, "family presence: damage channel exists"),
    ("family:crowd-control-mobility", "crowd_control", "stun", 0.15, "family presence: hard-CC or mobility atoms exist"),
    ("family:crowd-control-mobility", "mobility", "moderate", 0.15, "family presence: mobility atoms exist"),
    ("family:heal-shield", "sustain", "self_heal", 0.15, "family presence: heal/shield atoms exist"),
    ("family:interaction", "damage_profile", "burst", 0.10, "family presence: interaction atoms exist"),
    ("family:stack-transform-summon-resource", "scaling", "late", 0.10, "family presence: stack/transform/summon atoms exist"),
    ("family:vision-economy", "target_access", "reposition", 0.15, "family presence: stealth/vision atoms exist"),
]

MAPPING_V1: list[dict[str, Any]] = [
    {
        "atom_id": atom_id,
        "dimension": dimension,
        "label": label,
        "weight": weight,
        "evidence_note": note,
    }
    for atom_id, dimension, label, weight, note in _MAPPING_ROWS
]

FAMILY_FALLBACK_V1: list[dict[str, Any]] = [
    {
        "family": family.removeprefix("family:"),
        "dimension": dimension,
        "label": label,
        "weight": weight,
        "evidence_note": note,
    }
    for family, dimension, label, weight, note in _FAMILY_FALLBACKS
]

MAPPING_VERSION = "atom-ontology-mapping-v1"
