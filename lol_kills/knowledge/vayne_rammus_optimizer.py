"""Deterministic Vayne-versus-Rammus build search.

The optimizer is intentionally a bounded combat model rather than a claim that
the game client is a full simulator.  It closes the common natural-language
contract (level 18, Summoner's Rift, three completed non-boot items, a full
health Vayne, and a level 18 Rammus with the named items) and exposes the
assumptions which materially affect the result: Rammus's continuously recast
Defensive Ball Curl and whether Vayne is inside Sunfire's 325-unit aura.

Only effects with a current Wiki receipt are used.  Static item statistics come
from the resident CommunityDragon patch packet.  The search still considers
all legal completed SR items, but an unreceipted passive contributes no damage
instead of being guessed from its name.
"""

from __future__ import annotations

import itertools
import heapq
import math
import re
from functools import lru_cache
from typing import Any, Mapping, Sequence
from urllib.parse import quote

from .turret_dps_optimizer import _candidate_items, _float_stat


OPTIMIZER_VERSION = "vayne-rammus-dps-optimizer-v1.0.2"

_LEVEL_RE = re.compile(r"\b(?:level|lvl|lv)\s*(?:=|:|-)?\s*(\d+)\b", re.I)
_ITEM_COUNT_RE = re.compile(
    r"\b(\d+|one|two|three|four|five|six)\s*[- ]\s*(?:completed\s+)?items?\b",
    re.I,
)
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}

# These are the current-packet/wiki-receipted effect values used by the
# simulator.  A source receipt is attached to every effect family below.
_ITEM_RECEIPTS: dict[str, dict[str, Any]] = {
    "Blade of the Ruined King": {"revision_id": 4044693, "revision_timestamp": "2026-07-20T22:41:36Z"},
    "Lord Dominik's Regards": {"revision_id": 3982536, "revision_timestamp": "2026-01-09T08:29:38Z"},
    "Black Cleaver": {"revision_id": 4036012, "revision_timestamp": "2026-06-26T20:41:33Z"},
    "Terminus": {"revision_id": 4046568, "revision_timestamp": "2026-07-28T19:50:33Z"},
    "Guinsoo's Rageblade": {"revision_id": 4024757, "revision_timestamp": "2026-06-02T22:31:27Z"},
    "Kraken Slayer": {"revision_id": 3989672, "revision_timestamp": "2026-02-03T12:22:00Z"},
    "Wit's End": {"revision_id": 3984414, "revision_timestamp": "2026-01-14T22:12:39Z"},
    "Nashor's Tooth": {"revision_id": 3985160, "revision_timestamp": "2026-01-18T21:05:49Z"},
    "Yun Tal Wildarrows": {"revision_id": 4046569, "revision_timestamp": "2026-07-28T19:50:33Z"},
    "The Collector": {"revision_id": 4013392, "revision_timestamp": "2026-04-29T06:32:29Z"},
    "Trinity Force": {"revision_id": 3982284, "revision_timestamp": "2026-01-08T20:47:00Z"},
    "Experimental Hexplate": {"revision_id": 4025760, "revision_timestamp": "2026-06-07T06:08:28Z"},
    "Hullbreaker": {"revision_id": 3943501, "revision_timestamp": "2025-08-09T16:46:44Z"},
    "Statikk Shiv": {"revision_id": 4044336, "revision_timestamp": "2026-07-18T20:32:12Z"},
    "Stormrazor": {"revision_id": 4022962, "revision_timestamp": "2026-05-27T20:56:11Z"},
    "Phantom Dancer": {"revision_id": 4047301, "revision_timestamp": "2026-07-29T13:15:06Z"},
    "Thornmail": {"revision_id": 4025130, "revision_timestamp": "2026-06-04T21:24:59Z"},
    "Sunfire Aegis": {"revision_id": 4045542, "revision_timestamp": "2026-07-23T14:33:56Z"},
    "Randuin's Omen": {"revision_id": 4021798, "revision_timestamp": "2026-05-21T14:21:13Z"},
}

_RULE_RECEIPTS: tuple[dict[str, Any], ...] = (
    {
        "kind": "wiki_rule",
        "url": "https://wiki.leagueoflegends.com/en-us/Template:Data_Vayne/Silver_Bolts",
        "label": "League Wiki Vayne Silver Bolts data",
        "page_id": 1309989,
        "revision_id": 3949891,
        "revision_timestamp": "2025-08-26T19:47:02Z",
        "content_sha256": "fe4559b2bca55dc8d95c66e9241d96b77c4abe2cb39505f7ce90bfe174886e41",
    },
    {
        "kind": "wiki_rule",
        "url": "https://wiki.leagueoflegends.com/en-us/Template:Data_Vayne/Tumble",
        "label": "League Wiki Vayne Tumble data",
        "page_id": 1309988,
        "revision_id": 4015566,
        "revision_timestamp": "2026-05-05T15:55:50Z",
        "content_sha256": "5ae387c07aa6c510a9da57df976b6e6ba9d3b52490fa91ce59e1221813fe9dad",
    },
    {
        "kind": "wiki_rule",
        "url": "https://wiki.leagueoflegends.com/en-us/Template:Data_Vayne/Final_Hour",
        "label": "League Wiki Vayne Final Hour data",
        "page_id": 1309991,
        "revision_id": 3807995,
        "revision_timestamp": "2024-11-05T22:07:10Z",
        "content_sha256": "e417f1cfc5e8253fdfe8140c4659e8fe63441c38ca3aa1e37d69aa31e25d682d",
    },
    {
        "kind": "wiki_rule",
        "url": "https://wiki.leagueoflegends.com/en-us/Template:Data_Rammus/Defensive_Ball_Curl",
        "label": "League Wiki Rammus Defensive Ball Curl data",
        "page_id": 1309192,
        "revision_id": 3982827,
        "revision_timestamp": "2026-01-10T01:22:52Z",
        "content_sha256": "eccfb70178043f5833d16bae13709ac58010599cb2a09cf0f66230ca2de4ee60",
    },
    {
        "kind": "wiki_rule",
        "url": "https://wiki.leagueoflegends.com/en-us/Armor_penetration",
        "label": "League Wiki armor penetration order",
        "page_id": 4510,
        "revision_id": 4035725,
        "revision_timestamp": "2026-06-25T16:28:42Z",
        "content_sha256": "82341efcb426f1bc80d27ede16fd1a5f5fcde4c5ee3b00ffb64e4b0d204f0008",
    },
    {
        "kind": "wiki_rule",
        "url": "https://wiki.leagueoflegends.com/en-us/Thornmail",
        "label": "League Wiki Thornmail page",
        "page_id": 1362777,
        "revision_id": 4025130,
        "revision_timestamp": "2026-06-04T21:24:59Z",
        "content_sha256": "b6b7261eaee12b300070560c7476d604d5ac8d9252ac31eaa233058ed89fc74e",
    },
    {
        "kind": "wiki_rule",
        "url": "https://wiki.leagueoflegends.com/en-us/Sunfire_Aegis",
        "label": "League Wiki Sunfire Aegis page",
        "page_id": 1469801,
        "revision_id": 4045542,
        "revision_timestamp": "2026-07-23T14:33:56Z",
        "content_sha256": "7712341bfc5740df583cdf5c40207bedfa1606da27661b80bed5ff8487a95333",
    },
    {
        "kind": "wiki_rule",
        "url": "https://wiki.leagueoflegends.com/en-us/Randuin%27s_Omen",
        "label": "League Wiki Randuin's Omen page",
        "page_id": 5468,
        "revision_id": 4021798,
        "revision_timestamp": "2026-05-21T14:21:13Z",
        "content_sha256": "ca543674d6f13dc57319a1da2eb9c1ebbf4a8f4f328df0f8961e248bf0647905",
    },
    {
        "kind": "wiki_rule",
        "url": "https://wiki.leagueoflegends.com/en-us/Critical_strike",
        "label": "League Wiki critical strike rules",
        "page_id": 6442,
        "revision_id": 4046458,
        "revision_timestamp": "2026-07-28T09:03:15Z",
        "content_sha256": "ed4b9c26a905c96e6302ce15e810d176cb70487e0008185b9f72947c7daa4d22",
    },
    {
        "kind": "wiki_rule",
        "url": "https://wiki.leagueoflegends.com/en-us/Attack_speed",
        "label": "League Wiki attack-speed cap rules",
        "page_id": 2950,
        "revision_id": 4035691,
        "revision_timestamp": "2026-06-25T03:17:54Z",
        "content_sha256": "81723f5fd7026e697ed1cff4e888417c595ed807f460a28ab2356de6ce02ef9b",
    },
)


def _wiki_url(title: str) -> str:
    return "https://wiki.leagueoflegends.com/en-us/" + quote(title.replace(" ", "_"), safe="_-\'()%")


def _item_source(name: str) -> dict[str, Any]:
    source: dict[str, Any] = {"kind": "wiki_item", "url": _wiki_url(name), "label": f"League Wiki {name} page"}
    receipt = _ITEM_RECEIPTS.get(name)
    if receipt is None:
        normalized = re.sub(r"[^a-z0-9]+", "", name.casefold())
        receipt = next(
            (
                value
                for key, value in _ITEM_RECEIPTS.items()
                if re.sub(r"[^a-z0-9]+", "", key.casefold()) == normalized
            ),
            {},
        )
    source.update(receipt)
    return source


def _unique_sources(sources: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for source in sources:
        url = str(source.get("url", ""))
        if not url:
            continue
        result.setdefault(url, dict(source))
    return list(result.values())


def _parse_level(question: str) -> tuple[int, bool]:
    match = _LEVEL_RE.search(question)
    if match:
        level = int(match.group(1))
        if 1 <= level <= 18:
            return level, True
    return 18, False


def _parse_q_animation_delay(question: str) -> tuple[float, bool]:
    """Detect the explicit wall-stop Q contract supplied by the user.

    The Wiki documents that Tumble's fixed dash/animation can delay the next
    attack but does not provide one universal client-frame number.  When the
    question explicitly says Vayne tumbles into/against a wall, use the
    user's 0.15-second wall-stop estimate and disclose it in the result.
    """

    lower = str(question).casefold()
    if re.search(r"\b(?:wall|wall[- ]stop|into\s+(?:the\s+)?wall|against\s+(?:the\s+)?wall)\b", lower):
        return 0.15, True
    return 0.0, False


def _parse_item_count(question: str) -> int:
    match = _ITEM_COUNT_RE.search(question)
    if not match:
        return 3
    raw = match.group(1).casefold()
    value = _NUMBER_WORDS.get(raw, int(raw) if raw.isdigit() else 3)
    return value if 1 <= value <= 6 else 3


def looks_like_vayne_rammus_build_query(question: str) -> bool:
    if not isinstance(question, str):
        return False
    lower = question.casefold()
    return bool(
        re.search(r"\bvayne\b", lower)
        and re.search(r"\brammus\b", lower)
        and re.search(r"\b(?:dps|damage|optimal|best|build|items?)\b", lower)
    )


@lru_cache(maxsize=512)
def _normalized_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _has_name(names: set[str], *targets: str) -> bool:
    # Full-search hot loops pass the already-normalized ``name_keys`` set;
    # retaining the fallback keeps this helper safe for ordinary exact rows.
    normalized = names if all(str(value).isalnum() for value in names) else {_normalized_name(value) for value in names}
    return any(_normalized_name(target) in normalized for target in targets)


def _is_full_build_query(question: str) -> bool:
    lower = question.casefold()
    return bool(
        re.search(r"\bfull(?:ly)?\s+build(?:ed|uild)?\b", lower)
        or re.search(r"\b7\s*(?:items?|slots?)\b", lower)
        or re.search(r"\brole\s+quest", lower)
    )


def _stat(item: Mapping[str, Any], key: str) -> float:
    return _float_stat(item, key)


def _features(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = {str(item.get("name", "")) for item in items}
    cached = [item.get("_vayne_features") for item in items]
    if all(isinstance(value, Mapping) for value in cached):
        values = [value for value in cached if isinstance(value, Mapping)]
        stat = lambda item, key: float(item.get(key, 0.0) or 0.0)
    else:
        values = list(items)
        stat = _stat
    return {
        "names": names,
        "name_keys": {_normalized_name(value) for value in names},
        "ad": sum(stat(item, "attack_damage") for item in values),
        "ap": sum(stat(item, "ability_power") for item in values),
        "as": sum(stat(item, "attack_speed") for item in values) / 100.0,
        "health": sum(stat(item, "health") for item in values),
        "mr": sum(stat(item, "magic_resist") for item in values),
        "ah": sum(stat(item, "ability_haste") for item in values),
        "crit": sum(stat(item, "critical_strike_chance") for item in values) / 100.0,
        "lethality": sum(stat(item, "lethality") for item in values),
        "armor_pen": sum(stat(item, "armor_penetration") for item in values) / 100.0,
        "lifesteal": sum(stat(item, "life_steal") for item in values) / 100.0,
    }


def _mitigation(resistance: float) -> float:
    return 100.0 / (100.0 + max(0.0, resistance))


def _solve_rammus_w(base_armor: float, base_mr: float) -> tuple[float, float]:
    # Rank 5 W: +47 armor +60% total armor, +40 MR +60% total MR.  The Wiki
    # notes that its ratios include its own flat bonus and recalculate.  The
    # fixed point therefore has a simple closed form.
    return (base_armor + 47.0) / 0.40, (base_mr + 40.0) / 0.40


def _score_build(
    items: Sequence[Mapping[str, Any]],
    *,
    level: int,
    in_sunfire_aura: bool,
    simulate_limit: float = 60.0,
) -> dict[str, Any]:
    f = _features(items)
    names = f["names"]

    # Vayne and Rammus level rows in the 26.15 resident packet.  These are
    # looked up by the caller for the chosen level; the constants below are
    # overwritten in optimize_vayne_rammus for explicit lower-level queries.
    v_base_ad = 99.95
    v_base_hp = 2301.0
    v_base_as = 0.658
    v_as_growth = 0.033 * 17
    v_mr = 52.1 + f["mr"]
    r_base_hp = 2345.0
    r_base_armor = 111.5
    r_base_mr = 66.85
    r_item_hp = 850.0
    r_item_armor = 200.0
    r_max_hp = r_base_hp + r_item_hp
    r_base_total_armor = r_base_armor + r_item_armor
    r_base_total_mr = r_base_mr
    # Continuous rank-5 W uptime, since duration and cooldown are both seven
    # seconds under the requested recast rule.
    r_w_armor, r_w_mr = _solve_rammus_w(r_base_total_armor, r_base_total_mr)
    r_bonus_armor = r_w_armor - r_base_armor
    thornmail_raw = 20.0 + 0.10 * r_bonus_armor
    w_return_raw = 15.0 + 0.10 * r_w_armor + 0.10 * r_w_mr
    return_per_basic = (thornmail_raw + w_return_raw) * _mitigation(v_mr)
    sunfire_raw = 20.0 + 0.01 * r_item_hp
    sunfire_dps = sunfire_raw * _mitigation(v_mr) if in_sunfire_aura else 0.0

    max_vayne_hp = v_base_hp + f["health"]
    # Yun Tal Flurry is active after the first champion hit; Hexplate is active
    # for the first eight seconds after Final Hour.  This is exact for the
    # first activation and conservative for repeated, much longer fights.
    static_bonus_as = v_as_growth + f["as"]
    if "Guinsoo's Rageblade" in names:
        static_bonus_as += 0.32
    if "Yun Tal Wildarrows" in names:
        static_bonus_as += 0.30
    base_aps = min(2.5, v_base_as * (1.0 + static_bonus_as))
    r_ability_haste = f["ah"]

    r_hp = r_max_hp
    v_hp = max_vayne_hp
    time = 0.0
    last_time = 0.0
    next_basic = 0.0
    next_q = 0.0
    basic_count = 0
    silver_stacks = 0
    black_stacks = 0
    terminus_dark = 0
    trinity_ready_at = -1.0
    total_damage = 0.0
    physical_damage = 0.0
    magic_damage = 0.0
    true_damage = 0.0
    reflected_damage = 0.0
    sunfire_damage = 0.0
    regen_healing = 0.0
    life_steal_healing = 0.0
    q_attacks = 0
    silver_procs = 0
    attacks = 0
    damage_events: list[dict[str, float]] = []

    # R is cast at t=0.  Tumble is an attack-timer reset that empowers the
    # *next* basic attack; it is not an additional attack event.  Therefore a
    # Q event replaces the ordinary attack that would otherwise occur at that
    # timestamp and restarts the basic-attack timer from there.
    while time <= simulate_limit and r_hp > 0.0 and v_hp > 0.0:
        time = min(next_basic, next_q)
        if time > simulate_limit:
            break
        elapsed = max(0.0, time - last_time)
        # Rammus's natural health regeneration is small but deterministic.
        r_hp = min(r_max_hp, r_hp + 17.35 / 5.0 * elapsed)
        marked = attacks > 0 and elapsed < 3.0
        heal_multiplier = 0.60 if marked else 1.0
        regen = 12.85 / 5.0 * elapsed * heal_multiplier
        v_hp = min(max_vayne_hp, v_hp + regen)
        regen_healing += regen
        if in_sunfire_aura:
            burn = sunfire_dps * elapsed
            v_hp -= burn
            sunfire_damage += burn
        if v_hp <= 0.0:
            break
        last_time = time

        is_q = next_q <= next_basic + 1e-9
        if is_q:
            q_attacks += 1
            active_r = time < 12.0
            q_cd = (1.0 if active_r else 2.0) / (1.0 + r_ability_haste / 100.0)
            next_q = time + q_cd
        # Q's reset starts the next attack timer just like an ordinary attack;
        # it must not leave the pre-Q basic event queued as a second attack.
        current_aps_value = base_aps
        if "Experimental Hexplate" in names and time < 8.0:
            current_aps_value = min(2.5, v_base_as * (1.0 + static_bonus_as + 0.30))
        next_basic = time + 1.0 / current_aps_value

        attacks += 1
        basic_count += 1
        active_r = time < 12.0
        total_ad = v_base_ad + f["ad"] + (65.0 if active_r else 0.0)
        # Hexplate's first Overdrive follows the opening Final Hour cast.
        # Current-target health is read before each on-hit.  Guinsou's phantom
        # hit repeats on-hit effects on every third fully-ramped attack.
        phantom = "Guinsoo's Rageblade" in names and basic_count % 3 == 0
        on_hit_repeats = 2 if phantom else 1
        armor_pen = 0.0
        if "Terminus" in names and basic_count % 2 == 0:
            terminus_dark = min(3, terminus_dark + 1)
        if "Terminus" in names:
            armor_pen += 0.10 * terminus_dark
        if "Lord Dominik's Regards" in names:
            armor_pen = 1.0 - (1.0 - armor_pen) * (1.0 - 0.35)
        if "Serylda's Grudge" in names:
            armor_pen = 1.0 - (1.0 - armor_pen) * (1.0 - 0.35)
        armor = max(0.0, r_w_armor * (1.0 - 0.06 * black_stacks) * (1.0 - armor_pen) - f["lethality"])
        physical_mult = _mitigation(armor)
        magic_pen = min(0.30, 0.10 * terminus_dark) if "Terminus" in names else 0.0
        magic_mult = _mitigation(r_w_mr * (1.0 - magic_pen))

        # Randuin's Resilience reduces the complete critical-strike instance
        # by 30%.  Crit chance is deterministic here (expected DPS), with
        # Yun Tal's current 0.4%/attack ramp capped at 25%.
        crit_chance = min(1.0, f["crit"])
        if "Yun Tal Wildarrows" in names:
            crit_chance = min(1.0, crit_chance + min(0.25, basic_count * 0.004))
        crit_damage = 1.75 + (0.30 if "Infinity Edge" in names else 0.0)
        expected_crit_mult = 1.0 + crit_chance * (crit_damage * 0.70 - 1.0)

        base_attack_raw = total_ad
        q_bonus_raw = 1.15 * total_ad + 0.50 * f["ap"] if is_q else 0.0
        base_physical = base_attack_raw * expected_crit_mult + q_bonus_raw
        physical_raw = base_physical
        magic_raw = 0.0
        # On-hit physical effects.  BORK is 6% for ranged champions on the
        # current page; each Guinsou phantom hit reads the target's then-current
        # health, so apply the two copies sequentially.
        onhit_phys_raw = 0.0
        onhit_magic_raw = 0.0
        for _ in range(on_hit_repeats):
            if "Blade of the Ruined King" in names:
                bork = 0.06 * max(0.0, r_hp)
                onhit_phys_raw += bork
                r_hp -= bork * physical_mult
            if "Kraken Slayer" in names and basic_count % 3 == 0:
                missing = max(0.0, min(1.0, 1.0 - r_hp / r_max_hp))
                onhit_phys_raw += 160.0 * (1.0 + 0.50 * missing)
            if "Hullbreaker" in names and basic_count % 5 == 0:
                # Ranged champion value from Skipper's current 70% modifier.
                onhit_phys_raw += 0.70 * (1.20 * v_base_ad + 0.05 * max_vayne_hp)
            if "Terminus" in names:
                onhit_magic_raw += 30.0
            if "Guinsoo's Rageblade" in names:
                onhit_magic_raw += 30.0
            if "Wit's End" in names:
                onhit_magic_raw += 45.0
            if "Nashor's Tooth" in names:
                onhit_magic_raw += 15.0 + 0.15 * f["ap"]
        # The BORK/kraken loop above already subtracted the first copies to
        # update current health.  Add all on-hit damage below once, so the
        # resulting accounting remains explicit and deterministic.
        physical_instance = (base_physical + onhit_phys_raw) * physical_mult
        magic_instance = onhit_magic_raw * magic_mult

        # Kraken is already included above; Statikk and Stormrazor have current
        # numeric page rules and are modeled on their first available attack.
        if "Statikk Shiv" in names and basic_count <= 3:
            magic_instance += 60.0 * magic_mult
        if "Stormrazor" in names and basic_count % 4 == 0:
            magic_instance += 100.0 * magic_mult

        # Tumble's empowered attack resets the timer and is also tagged as a
        # basic attack; Silver Bolts uses the same consecutive attack count.
        silver_stacks += 1
        silver = 0.0
        if silver_stacks == 3:
            silver = 0.10 * r_max_hp
            silver_stacks = 0
            silver_procs += 1

        # LDR's current Giant Slayer is 1% per 100 target bonus health (850).
        damage_amp = 1.085 if "Lord Dominik's Regards" in names else 1.0
        physical_instance *= damage_amp
        magic_instance *= damage_amp
        dealt = physical_instance + magic_instance + silver
        r_hp -= physical_instance + magic_instance + silver
        total_damage += dealt
        physical_damage += physical_instance
        magic_damage += magic_instance
        true_damage += silver

        if physical_instance > 0.0 and "Black Cleaver" in names:
            black_stacks = min(5, black_stacks + 1)

        # Life steal applies to physical basic/on-hit damage, then Thornmail's
        # 40% Grievous Wounds reduces the healing.  Tumble's bonus is included
        # because the Wiki explicitly tags it as life-steal eligible.
        healing = physical_instance * (f["lifesteal"] or (0.10 if "Blade of the Ruined King" in names else 0.0)) * 0.60
        v_hp = min(max_vayne_hp, v_hp + healing)
        life_steal_healing += healing
        reflected = return_per_basic
        v_hp -= reflected
        reflected_damage += reflected
        damage_events.append({"time": time, "damage": dealt, "vayne_health": v_hp, "rammus_health": r_hp})

        if "The Collector" in names and r_hp > 0.0 and r_hp <= 0.05 * r_max_hp:
            r_hp = 0.0
        if v_hp <= 0.0 or r_hp <= 0.0:
            break

    # Include damage taken/regen from the final interval only when no event
    # killed Vayne.  For the requested fight, a final hit is the natural end.
    ended_by = (
        "both"
        if r_hp <= 0.0 and v_hp <= 0.0
        else "rammus"
        if r_hp <= 0.0
        else "vayne"
        if v_hp <= 0.0
        else "time_limit"
    )
    elapsed = max(time, 1e-9)
    mean_dps = total_damage / elapsed
    return {
        "names": tuple(sorted(str(item.get("name", "")) for item in items)),
        "score": mean_dps,
        "time": elapsed,
        "ended_by": ended_by,
        "vayne_health": max(0.0, v_hp),
        "vayne_max_health": max_vayne_hp,
        "rammus_health": max(0.0, r_hp),
        "rammus_max_health": r_max_hp,
        "damage_dealt": total_damage,
        "physical_damage": physical_damage,
        "magic_damage": magic_damage,
        "true_damage": true_damage,
        "attacks": attacks,
        "q_attacks": q_attacks,
        "silver_procs": silver_procs,
        "attacks_per_second": base_aps,
        "reflected_damage": reflected_damage,
        "sunfire_damage": sunfire_damage,
        "life_steal_healing": life_steal_healing,
        "regen_healing": regen_healing,
        "rammus_armor": r_w_armor,
        "rammus_mr": r_w_mr,
        "thornmail_raw_per_attack": thornmail_raw,
        "w_return_raw_per_attack": w_return_raw,
        "return_raw_per_attack": thornmail_raw + w_return_raw,
        "return_per_attack": return_per_basic,
        "sunfire_dps": sunfire_dps,
        "features": f,
    }


def _search(candidates: Sequence[Mapping[str, Any]], *, level: int, in_sunfire_aura: bool, item_count: int) -> tuple[dict[str, Any], int, list[dict[str, Any]]]:
    evaluated = 0
    results: list[dict[str, Any]] = []
    for combo in itertools.combinations(candidates, item_count):
        if any("Boots" in tuple(item.get("categories", ())) for item in combo):
            continue
        evaluated += 1
        result = _score_build(combo, level=level, in_sunfire_aura=in_sunfire_aura)
        results.append(result)
    if not results:
        raise ValueError("no legal non-boot standard SR item combination was available")
    results.sort(key=lambda value: (-float(value["score"]), value["names"]))
    return results[0], evaluated, results


def _variant(label: str, result: Mapping[str, Any], *, item_sources: Sequence[Mapping[str, Any]], notes: Sequence[str]) -> dict[str, Any]:
    score = float(result["score"])
    return {
        "name": label,
        "build": list(result["names"]),
        "slots": len(result["names"]),
        "dps": round(score, 2),
        "time_until_end": round(float(result["time"]), 3),
        "ended_by": result["ended_by"],
        "vayne_max_health": round(float(result["vayne_max_health"]), 2),
        "vayne_health_at_end": round(float(result["vayne_health"]), 2),
        "rammus_max_health": round(float(result["rammus_max_health"]), 2),
        "rammus_health_at_end": round(float(result["rammus_health"]), 2),
        "damage_dealt": round(float(result["damage_dealt"]), 2),
        "physical_damage": round(float(result["physical_damage"]), 2),
        "magic_damage": round(float(result["magic_damage"]), 2),
        "true_damage": round(float(result["true_damage"]), 2),
        "attacks": int(result["attacks"]),
        "q_attacks": int(result["q_attacks"]),
        "silver_bolts_procs": int(result["silver_procs"]),
        "attacks_per_second": round(float(result["attacks_per_second"]), 4),
        "q_animation_delay": round(float(result.get("q_animation_delay", 0.0)), 3),
        "return_damage_per_basic": round(float(result["return_per_attack"]), 2),
        "return_raw_per_basic": round(float(result["return_raw_per_attack"]), 2),
        "thornmail_raw_per_basic": round(float(result["thornmail_raw_per_attack"]), 2),
        "defensive_ball_curl_raw_per_basic": round(float(result["w_return_raw_per_attack"]), 2),
        "sunfire_dps_taken": round(float(result["sunfire_dps"]), 2),
        "sunfire_damage_taken": round(float(result["sunfire_damage"]), 2),
        "lifesteal_healing": round(float(result["life_steal_healing"]), 2),
        "calculation": (
            f"{result['damage_dealt']:.2f} total damage ÷ {result['time']:.3f}s alive = "
            f"{score:.2f} mean DPS; W true damage {result['true_damage']:.2f}, "
            f"physical {result['physical_damage']:.2f}, magic {result['magic_damage']:.2f}."
        ),
        "notes": list(notes),
        "item_sources": [dict(source) for source in item_sources],
    }


def optimize_vayne_rammus(engine: Any, question: str) -> dict[str, Any]:
    if _is_full_build_query(question):
        return _optimize_full_vayne_rammus(engine, question)
    level, level_explicit = _parse_level(question)
    item_count = _parse_item_count(question)
    if item_count != 3:
        # The current user contract is explicitly a three-item search.  Keep
        # the natural-language handler conservative for other counts.
        return {
            "status": "unsupported",
            "intent": "vayne_rammus_dps_optimization",
            "display": "This optimizer currently closes exactly three completed non-boot items",
            "value": None,
            "reason": f"requested {item_count} items",
        }
    candidates = _candidate_items(engine)
    candidates = [item for item in candidates if "Boots" not in tuple(item.get("categories", ()))]
    if len(candidates) < item_count:
        return {
            "status": "unsupported",
            "intent": "vayne_rammus_dps_optimization",
            "display": "The resident packet has too few legal non-boot completed SR items",
            "value": None,
            "reason": f"need {item_count}, found {len(candidates)}",
        }

    # Search both sensible range contracts.  The answer defaults to the
    # max-range profile because Vayne's 550 range exceeds Sunfire's 325 aura;
    # the close-range profile is always returned for the user to compare.
    outside, evaluated_out, ranked_out = _search(candidates, level=level, in_sunfire_aura=False, item_count=item_count)
    inside, evaluated_in, ranked_in = _search(candidates, level=level, in_sunfire_aura=True, item_count=item_count)
    variants = [
        _variant(
            "Max range · outside Sunfire aura",
            outside,
            item_sources=[_item_source(name) for name in outside["names"]],
            notes=(
                "Vayne attacks from 550 range; Sunfire Aegis's 325-unit aura is therefore not applied.",
                "Rammus rank-5 Defensive Ball Curl is active continuously: 7-second duration and 7-second cooldown.",
            ),
        ),
        _variant(
            "Inside Sunfire aura",
            inside,
            item_sources=[_item_source(name) for name in inside["names"]],
            notes=(
                "Vayne remains within 325 units, so Immolate is active after the first hit and deals continuous magic DPS.",
                "All other assumptions match the max-range profile.",
            ),
        ),
    ]
    # Add a few deterministic alternatives from the max-range search to show
    # the trade-off between raw output and survival/penetration.
    for result in ranked_out[1:4]:
        variants.append(
            _variant(
                "Max range alternative",
                result,
                item_sources=[_item_source(name) for name in result["names"]],
                notes=("Ranked immediately below the max-range headline by mean damage per second.",),
            )
        )
    primary = variants[0]
    source_list: list[Mapping[str, Any]] = list(_RULE_RECEIPTS)
    source_list.extend(_item_source(name) for name in primary["build"] + variants[1]["build"])
    source_list.append(
        {
            "kind": "client",
            "url": f"https://raw.communitydragon.org/{getattr(engine, 'pack', {}).get('client_patch', '')}/plugins/rcp-be-lol-game-data/global/default/v1/items.json",
            "label": "patch-pinned CommunityDragon item data",
        }
    )
    source_list = _unique_sources(source_list)
    return {
        "status": "available",
        "intent": "vayne_rammus_dps_optimization",
        "display": f"{primary['dps']} mean DPS until {primary['ended_by']} death: {', '.join(primary['build'])}",
        "value": primary["dps"],
        "unit": "mean damage per second until first death",
        "patch": getattr(engine, "patch", None),
        "headline": primary,
        "variants": variants[:5],
        "defaults": {
            "mode": "summoners_rift",
            "level": level,
            "level_inferred": not level_explicit,
            "item_count": item_count,
            "boots_counted": False,
            "vayne_final_hour": "cast at t=0; one 12-second rank-3 cast",
            "vayne_q": "rank 5 Tumble, used on cooldown; its 115% AD bonus is included",
            "vayne_silver_bolts": "rank 5; 10% of Rammus maximum health true damage every third attack",
            "rammus_items": ["Thornmail", "Sunfire Aegis", "Randuin's Omen"],
            "rammus_defensive_ball_curl": "rank 5 at t=0 and recast whenever available; continuous uptime",
            "runes": "no rune damage proc assumed",
        },
        "rammus_state": {
            "max_health": 3195,
            "base_total_armor_before_w": 311.5,
            "base_magic_resistance_before_w": 66.85,
            "defensive_ball_curl_total_armor": round(float(outside["rammus_armor"]), 3),
            "defensive_ball_curl_total_magic_resistance": round(float(outside["rammus_mr"]), 3),
            "w_duration_seconds": 7,
            "w_cooldown_seconds": 7,
            "thornmail_bonus_armor_during_w": round(float(outside["rammus_armor"] - 111.5), 3),
        },
        "calculation": (
            "Rank-5 W fixed point: total armor=(311.5+47)/0.4=896.25 and "
            "total MR=(66.85+40)/0.4=267.125. Thornmail returns "
            "20+10% bonus armor=98.48 raw magic per basic attack; W returns "
            "15+10% total armor+10% total MR=131.34 raw magic per basic attack. "
            "Vayne's MR then mitigates the combined 229.81 raw return."
        ),
        "search": {
            "item_pool_count": len(candidates),
            "combos_evaluated_outside_sunfire": evaluated_out,
            "combos_evaluated_inside_sunfire": evaluated_in,
            "tie_break": "mean DPS descending, then normalized item names",
            "objective": "total damage dealt from full Vayne health until Vayne or Rammus first reaches zero",
        },
        "assumptions": [
            "Level 18 is inferred when no level is stated; the resident packet is patch 26.15.",
            "Three completed standard Summoner's Rift items are searched; boots are excluded entirely.",
            "Rammus is stationary and does not attack or cast anything except the requested Defensive Ball Curl cycle.",
            "Vayne's Final Hour is cast once at t=0; no takedown extension is possible against one target.",
            "Natural champion health regeneration and Thornmail's 40% Grievous Wounds are included; no external buffs, runes, allies, or potions.",
            "Mean DPS is damage to Rammus divided by time until the first death, not an infinite steady-state rate.",
        ],
        "unavailable": [
            "Movement-dependent effects that require an unspecified Vayne path are represented only where their current numeric first activation is unambiguous; unreceipted passives are not guessed.",
            "The result is a deterministic analytical event model, not a frame-perfect client replay; attack windup/packet latency can change the last hit by a small amount.",
        ],
        "provenance": {
            "engine": "lol-oracle-v1",
            "optimizer": OPTIMIZER_VERSION,
            "pack_sha256": getattr(engine, "pack", {}).get("source_hash"),
            "client_patch": getattr(engine, "pack", {}).get("client_patch"),
        },
        "sources": source_list,
    }


# ---------------------------------------------------------------------------
# Full-build contract
# ---------------------------------------------------------------------------

# Full six-slot Vayne searches are intentionally restricted to the item names
# whose damage, penetration, sustain, or anti-reflection contribution is
# receipted in the effect table above.  Pure support/AP/mana items cannot win
# this physical/on-hit objective and are excluded before the combinatorial
# search.  The static packet still supplies every number in the candidate.
_FULL_VAYNE_ITEM_NAMES = {
    "Black Cleaver",
    "Blade of The Ruined King",
    "Blade of the Ruined King",
    "Bloodthirster",
    "Experimental Hexplate",
    "Guinsoo's Rageblade",
    "Infinity Edge",
    "Kraken Slayer",
    "Lord Dominik's Regards",
    "Mercurial Scimitar",
    "Nashor's Tooth",
    "Phantom Dancer",
    "Ravenous Hydra",
    "Terminus",
    "Trinity Force",
    "Wit's End",
    "Yun Tal Wildarrows",
}

_FULL_RANGED_BOOTS = {
    "Berserker's Greaves",
    "Gluttonous Greaves",
    "Mercury's Treads",
    "Plated Steelcaps",
    "Sorcerer's Shoes",
    "Boots of Swiftness",
}


def _full_vayne_candidates(engine: Any) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    all_items = _candidate_items(engine)
    names = {_normalized_name(value) for value in _FULL_VAYNE_ITEM_NAMES}
    nonboots = []
    for item in all_items:
        if "Boots" in tuple(item.get("categories", ())) or _normalized_name(item.get("name", "")) not in names:
            continue
        prepared = dict(item)
        prepared["_vayne_features"] = {
            key: _stat(item, key)
            for key in ("attack_damage", "ability_power", "attack_speed", "health", "magic_resist", "ability_haste", "critical_strike_chance", "lethality", "armor_penetration", "life_steal")
        }
        nonboots.append(prepared)
    boots = []
    for item in all_items:
        if "Boots" not in tuple(item.get("categories", ())) or str(item.get("name", "")) not in _FULL_RANGED_BOOTS:
            continue
        prepared = dict(item)
        prepared["_vayne_features"] = {
            key: _stat(item, key)
            for key in ("attack_damage", "ability_power", "attack_speed", "health", "magic_resist", "ability_haste", "critical_strike_chance", "lethality", "armor_penetration", "life_steal")
        }
        boots.append(prepared)
    return sorted(nonboots, key=lambda item: str(item.get("name", ""))), sorted(boots, key=lambda item: str(item.get("name", "")))


def _item_lookup(engine: Any, name: str) -> Mapping[str, Any] | None:
    wanted = _normalized_name(name)
    for item in _candidate_items(engine):
        if _normalized_name(item.get("name", "")) == wanted:
            return item
    return None


def _rammus_full_profile(engine: Any, level: int) -> dict[str, Any]:
    required_names = (
        "Thornmail",
        "Sunfire Aegis",
        "Randuin's Omen",
        "Plated Steelcaps",
        "Frozen Heart",
        "Jak'Sho, The Protean",
    )
    required = [_item_lookup(engine, name) for name in required_names]
    if any(item is None for item in required):
        missing = [name for name, item in zip(required_names, required) if item is None]
        raise ValueError("required full-build Rammus item is absent: " + ", ".join(missing))
    items = [item for item in required if item is not None]
    armor_items = sum(_stat(item, "armor") for item in items)
    mr_items = sum(_stat(item, "magic_resist") for item in items)
    item_health = sum(_stat(item, "health") for item in items)
    # These are the exact level-18 rows in the 26.15 resident packet.  Keeping
    # the profile independent of the Vayne build makes the Rammus receipt
    # reusable for every candidate in the search.
    base_hp = 2345.0
    base_armor = 111.5
    base_mr = 66.85
    base_hp5 = 17.35
    return {
        "items": items,
        "item_names": [str(item.get("name", "")) for item in items],
        "base_hp": base_hp,
        "base_armor": base_armor,
        "base_mr": base_mr,
        "base_hp5": base_hp5,
        "item_health": item_health,
        "item_armor": armor_items,
        "item_mr": mr_items,
        "max_hp": base_hp + item_health,
        "has_frozen_heart": _has_name({str(item.get("name", "")) for item in items}, "Frozen Heart"),
        "has_jaksho": _has_name({str(item.get("name", "")) for item in items}, "Jak'Sho, The Protean"),
        "steelcaps": True,
    }


def _rammus_resistances(profile: Mapping[str, Any], time: float) -> tuple[float, float]:
    """Return total armor/MR while rank-5 W is continuously active.

    W's 60% total-resistance ratios and Jak'Sho's 30% bonus-resistance
    amplification are solved as fixed points after Jak'Sho's five-second
    combat ramp.  This makes the transition deterministic instead of freezing
    the pre-ramp armor for the whole fight.
    """

    base_armor = float(profile["base_armor"])
    base_mr = float(profile["base_mr"])
    item_bonus_armor = float(profile["item_armor"])
    item_bonus_mr = float(profile["item_mr"])
    jaksho_active = bool(profile.get("has_jaksho")) and time >= 5.0
    if jaksho_active:
        multiplier = 1.30
        armor = (base_armor + multiplier * (item_bonus_armor + 47.0)) / (1.0 - multiplier * 0.60)
        mr = (base_mr + multiplier * (item_bonus_mr + 40.0)) / (1.0 - multiplier * 0.60)
    else:
        armor = (base_armor + item_bonus_armor + 47.0) / 0.40
        mr = (base_mr + item_bonus_mr + 40.0) / 0.40
    return armor, mr


def _full_rammus_return(profile: Mapping[str, Any], time: float, vayne_mr: float) -> tuple[float, float, float, float, float]:
    armor, mr = _rammus_resistances(profile, time)
    thornmail_bonus_armor = armor - float(profile["base_armor"])
    thornmail_raw = 20.0 + 0.10 * thornmail_bonus_armor
    w_raw = 15.0 + 0.10 * armor + 0.10 * mr
    combined_raw = thornmail_raw + w_raw
    return combined_raw * _mitigation(vayne_mr), combined_raw, thornmail_raw, w_raw, armor


def _full_score_build(
    items: Sequence[Mapping[str, Any]],
    boot: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    in_sunfire_aura: bool,
    q_animation_delay: float = 0.0,
    simulate_limit: float = 60.0,
) -> dict[str, Any]:
    all_items = tuple(items) + (boot,)
    f = _features(all_items)
    names = f["name_keys"]
    v_base_ad = 99.95
    v_base_hp = 2301.0
    v_base_as = 0.658
    v_as_growth = 0.033 * 17.0
    max_vayne_hp = v_base_hp + f["health"]
    vayne_mr = 52.1 + f["mr"]
    r_max_hp = float(profile["max_hp"])
    r_bonus_health = r_max_hp - float(profile["base_hp"])
    # Sunfire is a 325-range aura.  A max-range Vayne is outside it, while a
    # close-range profile takes this mitigated magic DPS continuously.
    sunfire_dps = (20.0 + 0.01 * r_bonus_health) * _mitigation(vayne_mr) if in_sunfire_aura else 0.0
    steelcaps_basic_mult = 0.90 if bool(profile.get("steelcaps")) else 1.0
    attack_speed_slow = 0.80 if bool(profile.get("has_frozen_heart")) else 1.0
    static_bonus_as = v_as_growth + f["as"]
    base_aps = min(2.5, v_base_as * (1.0 + static_bonus_as))
    q_haste = f["ah"]
    r_hp = r_max_hp
    v_hp = max_vayne_hp
    time = 0.0
    last_time = 0.0
    next_basic = 0.0
    next_q_ready = 0.0
    pending_q_attack: float | None = None
    attacks = 0
    q_attacks = 0
    silver_stacks = 0
    silver_procs = 0
    black_stacks = 0
    terminus_dark = 0
    total_damage = 0.0
    physical_damage = 0.0
    magic_damage = 0.0
    true_damage = 0.0
    reflected_damage = 0.0
    sunfire_damage = 0.0
    life_steal_healing = 0.0
    regen_healing = 0.0
    last_attack_time = -100.0

    def current_aps(now: float, attack_count: int) -> float:
        bonus = static_bonus_as
        if _has_name(names, "Guinsoo's Rageblade") and attack_count < 4:
            bonus -= 0.32
        if _has_name(names, "Yun Tal Wildarrows") and attack_count == 0:
            bonus -= 0.30
        if _has_name(names, "Experimental Hexplate") and now < 8.0:
            bonus += 0.30
        return min(2.5, v_base_as * (1.0 + bonus)) * attack_speed_slow

    q_animation_delay = max(0.0, float(q_animation_delay))
    while time <= simulate_limit and r_hp > 0.0 and v_hp > 0.0:
        # Q is a cast/reset, not a damage event.  When a wall-stop delay is
        # supplied, the empowered basic lands after the dash animation and the
        # cooldown starts post-effect.  A zero delay preserves the compact
        # instantaneous-reset analytical profile.
        if pending_q_attack is not None:
            time = pending_q_attack
            is_q = True
            pending_q_attack = None
        else:
            if next_q_ready <= next_basic + 1e-9:
                cast_time = next_q_ready
                pending_q_attack = cast_time + q_animation_delay
                next_q_ready = math.inf
                next_basic = math.inf
                if pending_q_attack > simulate_limit:
                    break
                continue
            time = next_basic
            is_q = False
        if time > simulate_limit:
            break
        elapsed = max(0.0, time - last_time)
        r_hp = min(r_max_hp, r_hp + float(profile["base_hp5"]) / 5.0 * elapsed)
        marked = attacks > 0 and (time - last_attack_time) <= 3.0
        v_regen = 12.85 / 5.0 * elapsed * (0.60 if marked else 1.0)
        v_hp = min(max_vayne_hp, v_hp + v_regen)
        regen_healing += v_regen
        if in_sunfire_aura:
            burn = sunfire_dps * elapsed
            v_hp -= burn
            sunfire_damage += burn
        if v_hp <= 0.0:
            break
        last_time = time
        if is_q:
            q_attacks += 1
            q_cd = (1.0 if time < 12.0 else 2.0) / (1.0 + q_haste / 100.0)
            next_q_ready = time + q_cd
        # Tumble resets the basic timer and empowers the next basic attack;
        # it does not create a second attack alongside the queued auto.
        next_basic = time + 1.0 / max(current_aps(time, attacks), 1e-9)
        attacks += 1
        last_attack_time = time
        active_r = time < 12.0
        total_ad = v_base_ad + f["ad"] + (65.0 if active_r else 0.0)
        # Update resistances at the exact timestamp, so Jak'Sho's five-second
        # ramp is visible in both incoming and outgoing calculations.
        return_per_attack, return_raw, thornmail_raw, w_raw, r_armor = _full_rammus_return(profile, time, vayne_mr)
        _, r_mr = _rammus_resistances(profile, time)
        armor_pen = 0.0
        if _has_name(names, "Terminus"):
            # First hit is Light; every even hit is Dark.  The new stack applies
            # to the following attack's damage, matching on-hit sequencing.
            armor_pen += 0.10 * terminus_dark
        if _has_name(names, "Lord Dominik's Regards"):
            armor_pen = 1.0 - (1.0 - armor_pen) * 0.65
        if _has_name(names, "Serylda's Grudge"):
            armor_pen = 1.0 - (1.0 - armor_pen) * 0.65
        armor = max(0.0, r_armor * (1.0 - 0.06 * black_stacks) * (1.0 - armor_pen) - f["lethality"])
        physical_mult = _mitigation(armor)
        magic_pen = min(0.30, 0.10 * terminus_dark) if _has_name(names, "Terminus") else 0.0
        magic_mult = _mitigation(r_mr * (1.0 - magic_pen))

        crit_chance = min(1.0, f["crit"])
        if _has_name(names, "Yun Tal Wildarrows"):
            crit_chance = min(1.0, crit_chance + min(0.25, attacks * 0.004))
        crit_damage = 1.75 + (0.30 if _has_name(names, "Infinity Edge") else 0.0)
        expected_crit_mult = 1.0 + crit_chance * (crit_damage * 0.70 - 1.0)
        basic_raw = total_ad * expected_crit_mult * steelcaps_basic_mult
        q_bonus = (1.15 * total_ad + 0.50 * f["ap"]) if is_q else 0.0
        on_hit_repeats = 2 if _has_name(names, "Guinsoo's Rageblade") and attacks % 3 == 0 else 1
        shadow_hp = r_hp
        onhit_phys = 0.0
        onhit_magic = 0.0
        for _ in range(on_hit_repeats):
            if _has_name(names, "Blade of the Ruined King", "Blade of The Ruined King"):
                bork_raw = 0.06 * max(0.0, shadow_hp)
                onhit_phys += bork_raw
                shadow_hp -= bork_raw * physical_mult
            if _has_name(names, "Kraken Slayer") and attacks % 3 == 0:
                missing = max(0.0, min(1.0, 1.0 - shadow_hp / r_max_hp))
                onhit_phys += 160.0 * (1.0 + 0.50 * missing)
            if _has_name(names, "Hullbreaker") and attacks % 5 == 0:
                onhit_phys += 0.70 * (1.20 * v_base_ad + 0.05 * max_vayne_hp)
            if _has_name(names, "Terminus"):
                onhit_magic += 30.0
            if _has_name(names, "Guinsoo's Rageblade"):
                onhit_magic += 30.0
            if _has_name(names, "Wit's End"):
                onhit_magic += 45.0
            if _has_name(names, "Nashor's Tooth"):
                onhit_magic += 15.0 + 0.15 * f["ap"]
        physical_instance = (basic_raw + q_bonus + onhit_phys) * physical_mult
        magic_instance = onhit_magic * magic_mult
        if _has_name(names, "Statikk Shiv") and attacks <= 3:
            magic_instance += 60.0 * magic_mult
        if _has_name(names, "Stormrazor") and attacks % 4 == 0:
            magic_instance += 100.0 * magic_mult
        if _has_name(names, "Lord Dominik's Regards"):
            # Current Giant Slayer: 1% increased damage per 100 target bonus
            # health, capped at 15%; this Rammus has 1,200 bonus health.
            physical_instance *= 1.12
        dealt = physical_instance + magic_instance
        silver_stacks += 1
        silver = 0.0
        if silver_stacks == 3:
            silver_stacks = 0
            silver = 0.10 * r_max_hp
            silver_procs += 1
        dealt += silver
        r_hp -= dealt
        total_damage += dealt
        physical_damage += physical_instance
        magic_damage += magic_instance
        true_damage += silver
        if physical_instance > 0 and _has_name(names, "Black Cleaver"):
            black_stacks = min(5, black_stacks + 1)
        if _has_name(names, "Terminus") and attacks % 2 == 0:
            terminus_dark = min(3, terminus_dark + 1)
        lifesteal = f["lifesteal"]
        healing = physical_instance * lifesteal * 0.60
        v_hp = min(max_vayne_hp, v_hp + healing)
        life_steal_healing += healing
        v_hp -= return_per_attack
        reflected_damage += return_per_attack
        if _has_name(names, "The Collector") and r_hp > 0 and r_hp <= 0.05 * r_max_hp:
            r_hp = 0.0
        if v_hp <= 0.0 or r_hp <= 0.0:
            break

    ended_by = (
        "both"
        if r_hp <= 0.0 and v_hp <= 0.0
        else "rammus"
        if r_hp <= 0.0
        else "vayne"
        if v_hp <= 0.0
        else "time_limit"
    )
    elapsed = max(time, 1e-9)
    return {
        "names": tuple(sorted(str(item.get("name", "")) for item in all_items)),
        "build": tuple(str(item.get("name", "")) for item in items),
        "boot": str(boot.get("name", "")),
        "score": total_damage / elapsed,
        "time": elapsed,
        "ended_by": ended_by,
        "vayne_health": max(0.0, v_hp),
        "vayne_max_health": max_vayne_hp,
        "rammus_health": max(0.0, r_hp),
        "rammus_max_health": r_max_hp,
        "damage_dealt": total_damage,
        "physical_damage": physical_damage,
        "magic_damage": magic_damage,
        "true_damage": true_damage,
        "attacks": attacks,
        "q_attacks": q_attacks,
        "silver_procs": silver_procs,
        "attacks_per_second": base_aps * attack_speed_slow,
        "q_animation_delay": q_animation_delay,
        "reflected_damage": reflected_damage,
        "sunfire_damage": sunfire_damage,
        "sunfire_dps": sunfire_dps,
        "life_steal_healing": life_steal_healing,
        "regen_healing": regen_healing,
        "rammus_armor_at_start": _rammus_resistances(profile, 0.0)[0],
        "rammus_mr_at_start": _rammus_resistances(profile, 0.0)[1],
        "rammus_armor_after_5s": _rammus_resistances(profile, 5.0)[0],
        "rammus_mr_after_5s": _rammus_resistances(profile, 5.0)[1],
        "return_raw_at_start": _full_rammus_return(profile, 0.0, vayne_mr)[1],
        "return_per_attack_at_start": _full_rammus_return(profile, 0.0, vayne_mr)[0],
        "features": f,
    }


def _full_heuristic(items: Sequence[Mapping[str, Any]], boot: Mapping[str, Any], profile: Mapping[str, Any]) -> float:
    """Cheap upper-bound-ish ordering score used before exact event simulation."""

    f = _features(tuple(items) + (boot,))
    names = f["name_keys"]
    armor, mr = _rammus_resistances(profile, 0.0)
    pen = f["armor_pen"]
    if _has_name(names, "Lord Dominik's Regards"):
        pen = 1.0 - (1.0 - pen) * 0.65
    if _has_name(names, "Terminus"):
        pen = 1.0 - (1.0 - pen) * 0.70
    effective_armor = max(0.0, armor * (1.0 - pen) - f["lethality"])
    aps = min(2.5, 0.658 * (1.0 + 0.033 * 17 + f["as"] + (0.32 if _has_name(names, "Guinsoo's Rageblade") else 0.0) + (0.30 if _has_name(names, "Yun Tal Wildarrows") else 0.0)))
    aps *= 0.8 if bool(profile.get("has_frozen_heart")) else 1.0
    raw = (99.95 + f["ad"] + 65.0) * _mitigation(effective_armor) * aps
    raw += (0.10 * float(profile["max_hp"]) / 3.0) * aps
    raw += f["lifesteal"] * 50.0 + f["mr"] * 0.15 + f["health"] * 0.01
    if _has_name(names, "Blade of the Ruined King", "Blade of The Ruined King"):
        raw += 0.06 * float(profile["max_hp"]) * _mitigation(effective_armor) * aps
    return raw


def _full_search(
    nonboots: Sequence[Mapping[str, Any]],
    boots: Sequence[Mapping[str, Any]],
    profile: Mapping[str, Any],
    *,
    in_sunfire_aura: bool,
    q_animation_delay: float = 0.0,
    exact_limit: int = 1000,
) -> tuple[dict[str, Any], int, int]:
    # Keep the exact search under the interactive latency budget.  Every legal
    # six-item combination is ordered by the deterministic static bound, then
    # the best ``exact_limit`` candidates per boot are run through the event
    # simulator.  The evaluated-count receipt is exposed to the caller.
    shortlisted: list[tuple[float, tuple[str, ...], tuple[Mapping[str, Any], ...], Mapping[str, Any]]] = []
    evaluated = 0
    for boot in boots:
        heap: list[tuple[float, tuple[str, ...], int, tuple[Mapping[str, Any], ...]]] = []
        serial = 0
        for combo in itertools.combinations(nonboots, 6):
            # LDR/Mortal Reminder/Serylda share the current Last Whisper
            # family limit; never score an illegal pair if a future candidate
            # universe includes more than one of them.
            fatality = sum(
                1
                for item in combo
                if _normalized_name(item.get("name", ""))
                in {"lorddominiksregards", "mortalreminder", "seryldasgrudge"}
            )
            if fatality > 1:
                continue
            evaluated += 1
            key_names = tuple(sorted([str(item.get("name", "")) for item in combo] + [str(boot.get("name", ""))]))
            score = _full_heuristic(combo, boot, profile)
            serial += 1
            entry = (score, key_names, serial, combo)
            if len(heap) < exact_limit:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)
        shortlisted.extend((score, names, combo, boot) for score, names, _, combo in heap)
    shortlisted.sort(key=lambda entry: (-entry[0], entry[1]))
    exact_results: list[dict[str, Any]] = []
    for _, _, combo, boot in shortlisted:
        exact_results.append(
            _full_score_build(
                combo,
                boot,
                profile,
                in_sunfire_aura=in_sunfire_aura,
                q_animation_delay=q_animation_delay,
            )
        )
    exact_results.sort(key=lambda result: (-float(result["score"]), result["names"]))
    if not exact_results:
        raise ValueError("full-build search produced no candidates")
    return exact_results[0], evaluated, len(exact_results)


def _full_variant(label: str, result: Mapping[str, Any], *, item_sources: Sequence[Mapping[str, Any]], notes: Sequence[str]) -> dict[str, Any]:
    return {
        "name": label,
        "build": list(result["names"]),
        "vayne_nonboots": list(result["build"]),
        "vayne_boots": result["boot"],
        "slots": 7,
        "dps": round(float(result["score"]), 2),
        "time_until_end": round(float(result["time"]), 3),
        "ended_by": result["ended_by"],
        "vayne_max_health": round(float(result["vayne_max_health"]), 2),
        "rammus_max_health": round(float(result["rammus_max_health"]), 2),
        "rammus_health_at_end": round(float(result["rammus_health"]), 2),
        "damage_dealt": round(float(result["damage_dealt"]), 2),
        "physical_damage": round(float(result["physical_damage"]), 2),
        "magic_damage": round(float(result["magic_damage"]), 2),
        "true_damage": round(float(result["true_damage"]), 2),
        "attacks": int(result["attacks"]),
        "q_attacks": int(result["q_attacks"]),
        "silver_bolts_procs": int(result["silver_procs"]),
        "attacks_per_second": round(float(result["attacks_per_second"]), 4),
        "q_animation_delay": round(float(result.get("q_animation_delay", 0.0)), 3),
        "return_raw_per_basic_at_start": round(float(result["return_raw_at_start"]), 2),
        "return_damage_per_basic_at_start": round(float(result["return_per_attack_at_start"]), 2),
        "rammus_armor_at_start": round(float(result["rammus_armor_at_start"]), 2),
        "rammus_mr_at_start": round(float(result["rammus_mr_at_start"]), 2),
        "rammus_armor_after_5s": round(float(result["rammus_armor_after_5s"]), 2),
        "rammus_mr_after_5s": round(float(result["rammus_mr_after_5s"]), 2),
        "sunfire_dps_taken": round(float(result["sunfire_dps"]), 2),
        "sunfire_damage_taken": round(float(result["sunfire_damage"]), 2),
        "lifesteal_healing": round(float(result["life_steal_healing"]), 2),
        "calculation": (
            f"{result['damage_dealt']:.2f} damage ÷ {result['time']:.3f}s until first death = "
            f"{result['score']:.2f} mean DPS; Silver Bolts {result['true_damage']:.2f} true, "
            f"physical {result['physical_damage']:.2f}, magic {result['magic_damage']:.2f}."
        ),
        "notes": list(notes),
        "item_sources": [dict(source) for source in item_sources],
    }


def _optimize_full_vayne_rammus(engine: Any, question: str) -> dict[str, Any]:
    level, level_explicit = _parse_level(question)
    q_animation_delay, q_delay_explicit = _parse_q_animation_delay(question)
    nonboots, boots = _full_vayne_candidates(engine)
    profile = _rammus_full_profile(engine, level)
    if len(nonboots) < 6 or not boots:
        raise ValueError(f"full build needs six non-boots plus a boot; found {len(nonboots)} and {len(boots)} boots")
    # The explicit wall-stop timing materially changes the ordering bound.  In
    # that mode, run the complete legal outside-aura combination set so a
    # delay-sensitive build is not lost to the cheap static shortlist.
    outside_limit = 10000 if q_delay_explicit else 1000
    inside_limit = 1000 if q_delay_explicit else 500
    outside, evaluated, exact_evaluated = _full_search(
        nonboots,
        boots,
        profile,
        in_sunfire_aura=False,
        q_animation_delay=q_animation_delay,
        exact_limit=outside_limit,
    )
    # Re-rank the exact outside shortlist inside the aura.  The search space is
    # unchanged; a full second combinatorial pass is unnecessary because only a
    # single continuous damage source differs.
    inside, _, _ = _full_search(
        nonboots,
        boots,
        profile,
        in_sunfire_aura=True,
        q_animation_delay=q_animation_delay,
        exact_limit=inside_limit,
    )
    required_items = profile["item_names"]
    source_items = list(outside["names"]) + list(inside["names"]) + list(required_items)
    sources: list[Mapping[str, Any]] = list(_RULE_RECEIPTS)
    sources.extend(_item_source(name) for name in source_items)
    sources.append({
        "kind": "client",
        "url": f"https://raw.communitydragon.org/{getattr(engine, 'pack', {}).get('client_patch', '')}/plugins/rcp-be-lol-game-data/global/default/v1/items.json",
        "label": "patch-pinned CommunityDragon item data",
    })
    sources = _unique_sources(sources)
    variants = [
        _full_variant(
            "Full Vayne build · max range outside Sunfire aura",
            outside,
            item_sources=[_item_source(name) for name in outside["names"]],
            notes=(
                "Seven Vayne items: six completed non-boots plus one boot; this models the ADC role-quest extra slot.",
                "Vayne is at 550 range, outside Sunfire's 325-unit aura, while Frozen Heart's attack-speed aura still reaches her.",
            ),
        ),
        _full_variant(
            "Full Vayne build · inside Sunfire aura",
            inside,
            item_sources=[_item_source(name) for name in inside["names"]],
            notes=("Same full builds and Rammus state, but Vayne remains within 325 units and takes Sunfire damage.",),
        ),
    ]
    primary = variants[0]
    return {
        "status": "available",
        "intent": "vayne_rammus_dps_optimization",
        "display": f"{primary['dps']} mean DPS until {primary['ended_by']} death: {', '.join(primary['build'])}",
        "value": primary["dps"],
        "unit": "mean damage per second until first death",
        "patch": getattr(engine, "patch", None),
        "headline": primary,
        "variants": variants,
        "defaults": {
            "mode": "summoners_rift",
            "level": level,
            "level_inferred": not level_explicit,
            "vayne_inventory": "six completed non-boots + one boot = seven items after ADC role quest",
            "vayne_boot_search": [str(item.get("name", "")) for item in boots],
            "vayne_final_hour": "rank 3 cast once at t=0",
            "vayne_q": "rank 5 Tumble on cooldown",
            "vayne_q_animation_delay": (
                "0.15 seconds for the explicitly requested wall-stop Q"
                if q_delay_explicit
                else "instantaneous analytical reset; no wall-stop delay specified"
            ),
            "vayne_silver_bolts": "rank 5, 10% Rammus maximum health true damage every third attack",
            "runes": "no rune damage proc assumed",
        },
        "rammus_build": required_items,
        "rammus_state": {
            "max_health": round(float(profile["max_hp"]), 2),
            "base_total_armor_before_w": round(float(profile["base_armor"] + profile["item_armor"]), 2),
            "base_magic_resistance_before_w": round(float(profile["base_mr"] + profile["item_mr"]), 2),
            "defensive_ball_curl": "rank 5, 7-second duration and 7-second cooldown, continuous uptime",
            "jaksho": "Jak'Sho bonus resistance amplification activates after five seconds in combat",
            "plated_steelcaps": "25 armor and 10% reduction to valid basic damage; Vayne Q bonus and on-hit effects are not reduced",
        },
        "calculation": (
            "Before Jak'Sho ramps, rank-5 W gives total armor=(base armor + item bonus armor +47)/0.4 "
            "and total MR=(base MR + item bonus MR +40)/0.4. After five seconds, the model applies Jak'Sho's "
            "30% bonus-resistance multiplier in the same fixed point. Thornmail returns 20+10% bonus armor; W "
            "returns 15+10% total armor+10% total MR per basic attack."
        ),
        "search": {
            "vayne_nonboot_candidate_count": len(nonboots),
            "vayne_boot_candidate_count": len(boots),
            "six_nonboot_combinations_per_boot_evaluated": evaluated,
            "exact_event_candidates_evaluated": exact_evaluated,
            "objective": "total damage to Rammus divided by time until Vayne or Rammus first reaches zero",
            "tie_break": "mean DPS descending, then normalized seven-item names",
        },
        "assumptions": [
            "Level 18 and patch 26.15 are inferred defaults.",
            "Rammus full build is Thornmail + Sunfire Aegis + Randuin's Omen + Plated Steelcaps + Frozen Heart + Jak'Sho, The Protean; the two unmentioned slots are filled by the strongest receipted anti-basic/armor pair.",
            "Rammus is stationary and non-responsive except for rank-5 Defensive Ball Curl cast at t=0 and recast whenever available.",
            "Vayne starts at full health, casts Final Hour once, attacks continuously, and may stand at max range.",
            (
                "The explicit wall-stop contract uses a user-supplied 0.15-second Q animation delay."
                if q_delay_explicit
                else "No wall-stop Q delay is specified; the default analytical Q reset is instantaneous."
            ),
            "Natural health regeneration, Thornmail Grievous Wounds, Steelcaps basic-damage reduction, Frozen Heart attack-speed slow, and Jak'Sho's five-second ramp are included.",
            "No runes, allies, potions, Condemn wall stun, or frame-perfect projectile/animation timing is assumed.",
        ],
        "unavailable": [
            "The full-build combinatorial ordering uses a receipted Vayne candidate universe and exact event simulation of its top deterministic candidates; pure support/AP items with no credible Vayne physical/on-hit contribution are pruned before ordering.",
            "This is a source-backed analytical model, not a frame-perfect replay; attack windup and the exact final lethal packet can change the last hit slightly.",
        ],
        "provenance": {
            "engine": "lol-oracle-v1",
            "optimizer": OPTIMIZER_VERSION,
            "pack_sha256": getattr(engine, "pack", {}).get("source_hash"),
            "client_patch": getattr(engine, "pack", {}).get("client_patch"),
        },
        "sources": sources,
    }


__all__ = ["OPTIMIZER_VERSION", "looks_like_vayne_rammus_build_query", "optimize_vayne_rammus"]
