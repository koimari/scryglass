"""Small, revision-pinned League Wiki rule supplements.

The patch packet is the authority for client champion/item numbers.  A few
game-system rules (runes, turrets, and permanent stack effects) are not
present in the client payload that feeds the fastpack.  This module keeps the
audited numeric pieces separate, names the exact Wiki revision used, and
exposes only arithmetic whose inputs are explicit in the caller's question.

These are intentionally narrow supplements, not a general Wiki parser.  If a
question needs a trigger, target filter, mode rule, or event state that is not
represented here, the oracle must remain unavailable.
"""

from __future__ import annotations

from typing import Any


# Metadata is copied from the local League Wiki source vault.  Keeping the
# revision/content hash in every answer makes the source receipt auditable even
# though normal callers only need the public URL.
WIKI_RULE_SOURCES: dict[str, dict[str, Any]] = {
    "Manaflow Band": {
        "url": "https://wiki.leagueoflegends.com/en-us/Manaflow_Band",
        "revision_id": 3980907,
        "revision_timestamp": "2026-01-02T12:17:27Z",
        "content_sha256": "b8ec08d1b6dfe68b50d56c46f38a488e85fdf1a92f1eeee1749b601023b54631",
        "label": "League Wiki Manaflow Band page",
    },
    "Turret": {
        "url": "https://wiki.leagueoflegends.com/en-us/Turret",
        "revision_id": 4019795,
        "revision_timestamp": "2026-05-19T14:01:14Z",
        "content_sha256": "07ec9945eefd3a86db0d013a41b2f8c6c83fbc2d0dbd6d1553e4a3ff749ac5b7",
        "label": "League Wiki turret page",
    },
    "Turret Plating": {
        "url": "https://wiki.leagueoflegends.com/en-us/Turret",
        "revision_id": 4019795,
        "revision_timestamp": "2026-05-19T14:01:14Z",
        "content_sha256": "07ec9945eefd3a86db0d013a41b2f8c6c83fbc2d0dbd6d1553e4a3ff749ac5b7",
        "label": "League Wiki turret plating rules",
    },
    "Touch of the Void": {
        "url": "https://wiki.leagueoflegends.com/en-us/Touch_of_the_Void",
        "revision_id": 4036112,
        "revision_timestamp": "2026-06-26T22:37:18Z",
        "content_sha256": "be9418ab745b45367c83a309946b037eb36b95998d7ed4a37f3ca6f598345c86",
        "label": "League Wiki Touch of the Void page",
    },
    "Template:Data Nasus/Siphoning Strike": {
        "url": "https://wiki.leagueoflegends.com/en-us/Template:Data_Nasus/Siphoning_Strike",
        "revision_id": 3971348,
        "revision_timestamp": "2025-12-02T21:35:41Z",
        "content_sha256": "990f23dafc78fccb05114103c169603173fd8fc938350f2b818120da714f8372",
        "label": "League Wiki Nasus Siphoning Strike data",
    },
    "Template:Data Thresh/Damnation": {
        "url": "https://wiki.leagueoflegends.com/en-us/Template:Data_Thresh/Damnation",
        "revision_id": 4025498,
        "revision_timestamp": "2026-06-06T06:37:57Z",
        "content_sha256": "12db53a7964af561d56bd9bd38b7825a9cb22ac75eb6d466df3fb5168f4f5f90",
        "label": "League Wiki Thresh Damnation data",
    },
    "Template:Data Senna/Absolution": {
        "url": "https://wiki.leagueoflegends.com/en-us/Template:Data_Senna/Absolution",
        "revision_id": 4034707,
        "revision_timestamp": "2026-06-23T21:22:11Z",
        "content_sha256": "375dcdffaf52df41183e2036f0a3fc682168f3f6d012fa6017b21760e0dd5a63",
        "label": "League Wiki Senna Absolution data",
    },
    "Template:Data Kindred/Mark of the Kindred": {
        "url": "https://wiki.leagueoflegends.com/en-us/Template:Data_Kindred/Mark_of_the_Kindred",
        "revision_id": 3994253,
        "revision_timestamp": "2026-02-27T11:10:12Z",
        "content_sha256": "9ac60eae427fac9ba279734dba2c01b34852eb0be84a01d95296328794afc14a",
        "label": "League Wiki Kindred Mark data",
    },
    "Template:Buff data Touch of the Void": {
        "url": "https://wiki.leagueoflegends.com/en-us/Template:Buff_data_Touch_of_the_Void",
        "revision_id": 4022998,
        "revision_timestamp": "2026-05-27T21:30:34Z",
        "content_sha256": "996c8d5f3953e2a5fb24d2fb0085f9beb008807020c0259a7517eb5da207fa24",
        "label": "League Wiki Touch of the Void buff data",
    },
}


STRUCTURES: dict[str, dict[str, float | int]] = {
    # Summoner's Rift values.  The attack-damage clocks are the exact
    # piecewise timings printed in the Turret page infoboxes.
    "outer": {
        "health": 9000,
        "base_attack_damage": 182,
        "attack_step": 12,
        "attack_first_second": 30,
        "attack_cap": 350,
    },
    "inner": {
        "health": 5000,
        "base_attack_damage": 187,
        "attack_step": 16,
        "attack_first_second": 180,
        "attack_cap": 427,
    },
    "inhibitor": {
        "health": 4750,
        "base_attack_damage": 187,
        "attack_step": 16,
        "attack_first_second": 180,
        "attack_cap": 427,
    },
    "nexus": {
        "health": 3500,
        "base_attack_damage": 165,
        "attack_step": 16,
        "attack_first_second": 180,
        "attack_cap": 405,
    },
}


def turret_attack_damage(kind: str, seconds: int) -> int:
    """Return the SR turret attack damage at an explicit game clock."""

    if kind not in STRUCTURES:
        raise KeyError(kind)
    if type(seconds) is not int or seconds < 0:
        raise ValueError("seconds must be a non-negative integer")
    rule = STRUCTURES[kind]
    first = int(rule["attack_first_second"])
    base = int(rule["base_attack_damage"])
    step = int(rule["attack_step"])
    cap = int(rule["attack_cap"])
    if seconds < first:
        return base
    increments = (seconds - first) // 60 + 1
    return min(cap, base + step * increments)


def manaflow_bonus(stacks: int) -> int:
    """Manaflow Band's permanent maximum-mana component."""

    if type(stacks) is not int or stacks < 0:
        raise ValueError("stacks must be a non-negative integer")
    return min(stacks, 10) * 25


def nasus_siphoning_strike_bonus(rank: int, stacks: int) -> int:
    """Nasus Q bonus physical damage from rank and stored Q stacks."""

    if type(rank) is not int or not 1 <= rank <= 5:
        raise ValueError("rank must be in [1, 5]")
    if type(stacks) is not int or stacks < 0:
        raise ValueError("stacks must be non-negative")
    return 40 + 20 * (rank - 1) + stacks


def thresh_soul_stats(stacks: int) -> tuple[int, int]:
    """Return (ability power, bonus armor) from Thresh soul stacks."""

    if type(stacks) is not int or stacks < 0:
        raise ValueError("stacks must be non-negative")
    return stacks, stacks


def senna_mist_stats(stacks: int) -> tuple[float, int, int]:
    """Return (bonus AD, bonus range, crit chance percentage) from Mist."""

    if type(stacks) is not int or stacks < 0:
        raise ValueError("stacks must be non-negative")
    milestones = stacks // 20
    return stacks * 0.75, milestones * 20, milestones * 10


def kindred_bonus_range(stacks: int) -> int:
    """Return Lamb's bonus range from the Mark of the Kindred table."""

    if type(stacks) is not int or stacks < 0:
        raise ValueError("stacks must be non-negative")
    if stacks < 4:
        return 0
    return min(250, 75 + 25 * ((stacks - 4) // 3))


def touch_of_the_void_burn(stacks: int, *, ranged: bool = False) -> tuple[int, int]:
    """Return (damage_per_tick, four_second_total) for structure burn.

    The current local snapshot's V26.11 patch-history entry gives melee
    values 4/12/16 and ranged values 2/6/8 for one, two, and three stacks.
    The burn ticks every 0.5 seconds for four seconds (eight ticks).
    """

    if type(stacks) is not int or not 1 <= stacks <= 3:
        raise ValueError("Touch of the Void supports 1-3 explicit stacks")
    melee = (4, 12, 16)[stacks - 1]
    per_tick = melee // 2 if ranged else melee
    return per_tick, per_tick * 8


def wiki_rule_source(title: str) -> dict[str, Any]:
    """Return a copy suitable for a response's source list."""

    try:
        source = WIKI_RULE_SOURCES[title]
    except KeyError as exc:
        raise KeyError(f"no pinned Wiki rule source for {title!r}") from exc
    return {
        "kind": "wiki_rule",
        "url": source["url"],
        "label": source["label"],
        "revision_id": source["revision_id"],
        "revision_timestamp": source["revision_timestamp"],
        "content_sha256": source["content_sha256"],
    }


__all__ = [
    "STRUCTURES",
    "WIKI_RULE_SOURCES",
    "kindred_bonus_range",
    "manaflow_bonus",
    "nasus_siphoning_strike_bonus",
    "senna_mist_stats",
    "thresh_soul_stats",
    "touch_of_the_void_burn",
    "turret_attack_damage",
    "wiki_rule_source",
]
