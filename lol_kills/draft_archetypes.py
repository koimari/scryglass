#!/usr/bin/env python3
"""
Multi-label champion archetypes for draft → outcome edge hunting.

Tags are intentionally overlapping (e.g. Orianna = control_mage + teamfight_aoe).
"""

from __future__ import annotations

from lol_kills.etl.aliases import normalize_champ

# Multi-label. Keep names stable — used as feature columns arch_{name}_diff.
ARCHETYPE_TAGS: dict[str, set[str]] = {
    # --- Control / zone mages (late teamfight) ---
    "Anivia": {"control_mage", "scaling_late", "teamfight_aoe", "poke_siege"},
    "Orianna": {"control_mage", "teamfight_aoe", "peel_enchanter"},
    "Viktor": {"control_mage", "scaling_late", "poke_siege", "teamfight_aoe"},
    "Cassiopeia": {"control_mage", "scaling_late", "teamfight_aoe"},
    "Azir": {"control_mage", "scaling_late", "teamfight_aoe", "poke_siege"},
    "Aurelion Sol": {"control_mage", "scaling_late", "teamfight_aoe"},
    "Ryze": {"control_mage", "scaling_late", "teamfight_aoe"},
    "Twisted Fate": {"control_mage", "pick", "early_snowball"},
    "Lissandra": {"control_mage", "engage", "teamfight_aoe"},
    "Hwei": {"control_mage", "poke_siege", "teamfight_aoe"},
    "Syndra": {"control_mage", "burst_mage", "pick", "teamfight_aoe"},
    "Taliyah": {"control_mage", "poke_siege", "roam"},
    "Swain": {"control_mage", "teamfight_aoe", "tank_frontline"},
    "Mel": {"control_mage", "burst_mage", "scaling_late"},
    "Neeko": {"control_mage", "engage", "pick"},
    "Seraphine": {"control_mage", "peel_enchanter", "teamfight_aoe"},
    "Karthus": {"control_mage", "scaling_late", "teamfight_aoe"},
    # --- Burst / assassin mages ---
    "Ahri": {"burst_mage", "roam", "pick"},
    "LeBlanc": {"burst_mage", "assassin", "pick", "early_snowball"},
    # Patch 26.13 — mid AP assassin (Ashen Exorcist); no OE history yet
    "Locke": {"burst_mage", "assassin", "pick", "early_snowball"},
    "Zoe": {"burst_mage", "poke_siege", "pick"},
    "Veigar": {"burst_mage", "scaling_late", "pick"},
    "Annie": {"burst_mage", "engage", "teamfight_aoe"},
    "Vex": {"burst_mage", "assassin", "pick"},
    "Aurora": {"burst_mage", "assassin", "pick"},
    "Lux": {"burst_mage", "poke_siege", "pick", "peel_enchanter"},
    "Xerath": {"burst_mage", "poke_siege", "scaling_late"},
    "Brand": {"burst_mage", "teamfight_aoe"},
    "Zyra": {"burst_mage", "poke_siege", "teamfight_aoe"},
    "Heimerdinger": {"poke_siege", "scaling_late", "control_mage"},
    "Vladimir": {"scaling_late", "control_mage", "teamfight_aoe"},
    "Kassadin": {"scaling_late", "assassin", "burst_mage"},
    # --- Engage / catch ---
    "Malphite": {"engage", "tank_frontline", "teamfight_aoe"},
    "Ornn": {"engage", "tank_frontline", "teamfight_aoe", "scaling_late"},
    "Sejuani": {"engage", "tank_frontline", "teamfight_aoe"},
    "Nautilus": {"engage", "tank_frontline", "pick"},
    "Leona": {"engage", "tank_frontline"},
    "Alistar": {"engage", "tank_frontline", "peel_enchanter"},
    "Rell": {"engage", "tank_frontline", "teamfight_aoe"},
    "Amumu": {"engage", "tank_frontline", "teamfight_aoe"},
    "Jarvan IV": {"engage", "early_snowball", "teamfight_aoe"},
    "Wukong": {"engage", "teamfight_aoe", "skirmisher"},
    "Maokai": {"engage", "tank_frontline", "peel_enchanter"},
    "Skarner": {"engage", "tank_frontline", "pick"},
    "Galio": {"engage", "tank_frontline", "peel_enchanter", "teamfight_aoe"},
    "Poppy": {"engage", "peel_enchanter", "tank_frontline"},
    "Rakan": {"engage", "peel_enchanter", "roam"},
    "Blitzcrank": {"pick", "engage"},
    "Thresh": {"pick", "peel_enchanter", "engage"},
    "Pyke": {"pick", "assassin", "roam", "early_snowball"},
    "Morgana": {"pick", "peel_enchanter", "control_mage"},
    "Bard": {"pick", "roam", "peel_enchanter"},
    # --- Enchanters / peel ---
    "Lulu": {"peel_enchanter", "scaling_late"},
    "Janna": {"peel_enchanter"},
    "Nami": {"peel_enchanter"},
    "Milio": {"peel_enchanter", "scaling_late"},
    "Yuumi": {"peel_enchanter", "scaling_late"},
    "Soraka": {"peel_enchanter"},
    "Sona": {"peel_enchanter", "teamfight_aoe"},
    "Renata Glasc": {"peel_enchanter", "teamfight_aoe"},
    "Karma": {"peel_enchanter", "poke_siege"},
    "Zilean": {"peel_enchanter", "scaling_late", "control_mage"},
    # --- Hypercarry / scaling ADC ---
    "Kog'Maw": {"hypercarry_adc", "scaling_late"},
    "Twitch": {"hypercarry_adc", "scaling_late", "assassin"},
    "Jinx": {"hypercarry_adc", "scaling_late", "teamfight_aoe"},
    "Vayne": {"hypercarry_adc", "scaling_late", "skirmisher"},
    "Aphelios": {"hypercarry_adc", "scaling_late"},
    "Zeri": {"hypercarry_adc", "scaling_late"},
    "Smolder": {"hypercarry_adc", "scaling_late", "poke_siege"},
    "Sivir": {"hypercarry_adc", "teamfight_aoe"},
    "Xayah": {"hypercarry_adc", "teamfight_aoe"},
    "Kai'Sa": {"hypercarry_adc", "skirmisher"},
    "Ashe": {"poke_siege", "pick", "utility_adc"},
    "Varus": {"poke_siege", "pick", "hypercarry_adc"},
    "Ezreal": {"poke_siege", "skirmisher"},
    "Jhin": {"poke_siege", "pick"},
    "Caitlyn": {"poke_siege", "early_snowball"},
    "Miss Fortune": {"teamfight_aoe", "early_snowball"},
    "Draven": {"early_snowball", "skirmisher"},
    "Lucian": {"early_snowball", "skirmisher"},
    "Tristana": {"early_snowball", "hypercarry_adc"},
    "Kalista": {"early_snowball", "skirmisher"},
    "Nilah": {"teamfight_aoe", "skirmisher", "hypercarry_adc"},
    "Samira": {"teamfight_aoe", "skirmisher", "early_snowball"},
    "Senna": {"poke_siege", "scaling_late", "utility_adc"},
    # --- Assassins / divers ---
    "Zed": {"assassin", "early_snowball"},
    "Qiyana": {"assassin", "roam", "early_snowball"},
    "Kha'Zix": {"assassin", "early_snowball"},
    "Rengar": {"assassin", "early_snowball"},
    "Akali": {"assassin", "skirmisher", "scaling_late"},
    "Naafiri": {"assassin", "early_snowball"},
    "Ekko": {"assassin", "skirmisher"},
    "Diana": {"assassin", "engage", "teamfight_aoe"},
    "Nocturne": {"assassin", "engage"},
    "Kayn": {"assassin", "skirmisher"},  # may be absent from OE list
    "Bel'Veth": {"skirmisher", "scaling_late", "early_snowball"},
    "Viego": {"skirmisher", "assassin"},
    "Sylas": {"skirmisher", "teamfight_aoe", "engage"},
    # --- Duelists / split ---
    "Fiora": {"skirmisher", "splitpush", "early_snowball"},
    "Camille": {"skirmisher", "splitpush", "engage"},
    "Jax": {"skirmisher", "splitpush", "scaling_late"},
    "Irelia": {"skirmisher", "early_snowball"},
    "Gwen": {"skirmisher", "scaling_late", "teamfight_aoe"},
    "Tryndamere": {"skirmisher", "splitpush", "scaling_late"},
    "Yasuo": {"skirmisher", "teamfight_aoe"},
    "Yone": {"skirmisher", "teamfight_aoe", "scaling_late"},
    "Riven": {"skirmisher", "early_snowball"},
    "Aatrox": {"skirmisher", "teamfight_aoe"},
    "Sett": {"skirmisher", "engage", "teamfight_aoe"},
    "Ambessa": {"skirmisher", "engage"},
    "Gnar": {"skirmisher", "engage", "teamfight_aoe"},
    "Jayce": {"poke_siege", "early_snowball", "skirmisher"},
    "Gangplank": {"poke_siege", "scaling_late", "splitpush"},
    "Yorick": {"splitpush", "scaling_late"},
    "Nasus": {"splitpush", "scaling_late", "tank_frontline"},
    "Illaoi": {"splitpush", "teamfight_aoe"},
    "Urgot": {"skirmisher", "teamfight_aoe"},
    "Kled": {"skirmisher", "early_snowball"},
    "Renekton": {"early_snowball", "skirmisher"},
    "Darius": {"early_snowball", "skirmisher"},
    "Olaf": {"early_snowball", "skirmisher", "engage"},
    "Kennen": {"engage", "teamfight_aoe", "poke_siege"},
    "Rumble": {"teamfight_aoe", "poke_siege"},
    "Mordekaiser": {"skirmisher", "teamfight_aoe", "scaling_late"},
    "Garen": {"skirmisher", "tank_frontline"},
    "Dr. Mundo": {"tank_frontline", "splitpush", "scaling_late"},
    "Sion": {"tank_frontline", "engage", "splitpush"},
    "Cho'Gath": {"tank_frontline", "scaling_late"},
    "K'Sante": {"tank_frontline", "skirmisher"},
    "Shen": {"tank_frontline", "peel_enchanter", "splitpush"},
    "Tahm Kench": {"tank_frontline", "peel_enchanter"},
    "Zac": {"tank_frontline", "engage"},
    "Volibear": {"engage", "skirmisher"},
    "Trundle": {"skirmisher", "splitpush"},
    "Udyr": {"skirmisher", "early_snowball"},
    "Shyvana": {"scaling_late", "teamfight_aoe"},
    "Gragas": {"engage", "peel_enchanter"},
    "Ivern": {"peel_enchanter", "scaling_late"},
    "Lillia": {"scaling_late", "teamfight_aoe", "skirmisher"},
    "Nidalee": {"early_snowball", "poke_siege", "assassin"},
    "Elise": {"early_snowball", "assassin", "pick"},
    "Lee Sin": {"early_snowball", "skirmisher", "engage"},
    "Xin Zhao": {"early_snowball", "engage"},
    "Vi": {"engage", "early_snowball"},
    "Pantheon": {"early_snowball", "roam", "assassin"},
    "Rek'Sai": {"early_snowball", "skirmisher"},
    "Hecarim": {"engage", "early_snowball"},
    "Graves": {"early_snowball", "skirmisher"},
    "Kindred": {"scaling_late", "skirmisher", "hypercarry_adc"},
    "Fiddlesticks": {"engage", "pick", "teamfight_aoe"},
    "Nocturne": {"engage", "assassin"},
    # --- Mid divers / oddballs ---
    "Corki": {"poke_siege", "early_snowball"},
    "Ziggs": {"poke_siege", "siege_specialist", "scaling_late"},
    "Ezreal": {"poke_siege", "skirmisher"},
    "Kai'Sa": {"hypercarry_adc", "skirmisher"},
    "Yuumi": {"peel_enchanter", "scaling_late"},
    "Taric": {"peel_enchanter", "engage", "tank_frontline"},
    "Braun": {"peel_enchanter", "tank_frontline"},
    "Braum": {"peel_enchanter", "tank_frontline"},
}

# Fix Braum typo key if present
ARCHETYPE_TAGS.pop("Braun", None)

ARCHETYPE_NAMES = sorted({t for tags in ARCHETYPE_TAGS.values() for t in tags})


def champ_tags(champ: str) -> set[str]:
    return set(ARCHETYPE_TAGS.get(normalize_champ(champ), set()))


def side_archetype_counts(champs: list[str]) -> dict[str, float]:
    counts = {f"arch_{a}": 0.0 for a in ARCHETYPE_NAMES}
    for c in champs:
        for t in champ_tags(c):
            counts[f"arch_{t}"] += 1.0
    return counts


def draft_archetype_features(blue: list[str], red: list[str]) -> dict[str, float]:
    """Blue − red counts per archetype + absolute presence flags."""
    b, r = side_archetype_counts(blue), side_archetype_counts(red)
    out: dict[str, float] = {}
    for a in ARCHETYPE_NAMES:
        kb, kr = f"arch_{a}", f"arch_{a}"
        out[f"{kb}_blue"] = b[kb]
        out[f"{kb}_red"] = r[kr]
        out[f"{kb}_diff"] = b[kb] - r[kr]
        out[f"{kb}_sum"] = b[kb] + r[kr]
    # Composition ratios
    out["arch_control_vs_engage"] = out["arch_control_mage_diff"] - out["arch_engage_diff"]
    out["arch_scale_vs_snowball"] = out["arch_scaling_late_diff"] - out["arch_early_snowball_diff"]
    out["arch_poke_vs_engage"] = out["arch_poke_siege_diff"] - out["arch_engage_diff"]
    return out


def coverage_report(champs: list[str]) -> dict:
    tagged = sum(1 for c in champs if champ_tags(c))
    return {"n": len(champs), "tagged": tagged, "frac": tagged / max(len(champs), 1)}
