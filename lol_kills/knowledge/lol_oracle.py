"""A small source-linked League calculation oracle built on the local fast path.

This layer adds only arithmetic whose inputs are explicit in the exact patch
packet or a revision-receipted Wiki supplement: stat deltas, resource windows,
maximum resource/health lookups, spell cast budgets from client
``costCoefficients``, conservative direct ability-damage graphs, positive
typed mitigation, narrow permanent stack rules, and explicit numeric damage
sequences. Item passives, trigger timing, shields, penetration, and live map
state remain unavailable until their semantics are validated.
"""

from __future__ import annotations

import json
import hashlib
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

from .ability_damage import evaluate_direct_damage
from .item_stats import parse_static_item_stats
from .mechanics_engine import Combatant, Damage, Event, GameState, MechanicsEngine
from .mechanics_kernel import UnsupportedFormulaError
from .quick_mechanics import QuickMechanicsEngine
from .wiki_rules import (
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
from .turret_dps_optimizer import (
    looks_like_jinx_turret_build_query,
    optimize_jinx_turret,
)
from .vayne_rammus_optimizer import (
    looks_like_vayne_rammus_build_query,
    optimize_vayne_rammus,
)


_LEVEL_RE = re.compile(r"\b(?:level|lvl|lv)\s*(?:=|:|-)?\s*(\d+)\b", re.I)
_DELTA_RE = re.compile(
    r"(?:"
    r"\bfrom\s+(?:level|lvl|lv)\s*(?P<from_level>\d+)\s+to\s+(?:level|lvl|lv)\s*(?P<to_level>\d+)\b"
    r"|"
    r"\bbetween\s+(?:levels?|lvls?|lvs?)\s*(?P<between_from>\d+)\s+and\s+(?:(?:levels?|lvls?|lvs?)\s*)?(?P<between_to>\d+)\b"
    r")",
    re.I,
)
_RANK_RE = re.compile(r"\brank\s*[-:=]?\s*(\d+)\b", re.I)
_SECONDS_RE = re.compile(r"\b(?:in|over|for)\s+(\d+(?:\.\d+)?)\s*seconds?\b", re.I)
_ABILITY_KEY_RE = re.compile(r"\b([QWER])\b", re.I)
_ABILITY_LEVEL_RE = re.compile(
    r"\b[QWER]\s*(?:rank|level|lvl|lv)\s*[-:=]?\s*(\d+)\b", re.I
)
_AP_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*AP\b", re.I)
_AD_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:total\s+)?(?:attack\s+damage|AD)\b", re.I
)
_ARMOR_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*armor\b", re.I)
_DAMAGE_MODIFIER_RE = re.compile(
    r"\b(?:armor|penetration|pen|shield|items?|runes?|"
    r"buffs?|debuffs?|passive|crit(?:ical)?|target(?:'s)?\s+health|health)\b",
    re.I,
)
_MAGIC_RESIST_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:magic\s+resist(?:ance)?|MR)\b", re.I
)
_STACK_RE = re.compile(
    r"\b(\d+)\s*(?:(?:[a-z][a-z'’_-]*\s+){0,6})(?:stacks?|souls?|marks?)\b",
    re.I,
)
_CLOCK_RE = re.compile(
    r"\b(?:at|by|after)\s+(\d{1,2}):(\d{2})\b|\b(?:at|by|after)\s+(\d+(?:\.\d+)?)\s*minutes?\b",
    re.I,
)
_INITIAL_HEALTH_RE = re.compile(
    r"\b(?:starts?|starting|begin(?:s|ning)?)\s+(?:with\s+)?(\d+(?:\.\d+)?)\s+health\b",
    re.I,
)
_SEQUENCE_DAMAGE_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s+(physical|magic|true)\s+damage\b",
    re.I,
)
_COMPARISON_RE = re.compile(
    r"\b(?:delta|difference|gap|difference\s+between|how\s+much\s+(?:higher|lower)|higher\s+than|lower\s+than)\b",
    re.I,
)

# The patch packet does not yet contain a numeric rune table.  This lexical
# inventory is therefore used only to prevent a named rune from being silently
# ignored by an otherwise exact cast-budget question; every name except
# Manaflow Band remains explicitly unsupported until its effect rule is
# revision-receipted.
_KNOWN_RUNE_NAMES = (
    "Adaptive Force",
    "Aftershock",
    "Approach Velocity",
    "Arcane Comet",
    "Absolute Focus",
    "Biscuit Delivery",
    "Bone Plating",
    "Cash Back",
    "Celerity",
    "Cheap Shot",
    "Conditioning",
    "Conqueror",
    "Cosmic Insight",
    "Coup de Grace",
    "Dark Harvest",
    "Demolish",
    "Eyeball Collection",
    "First Strike",
    "Fleet Footwork",
    "Gathering Storm",
    "Glacial Augment",
    "Grasp of the Undying",
    "Hail of Blades",
    "Hextech Flashtraption",
    "Ingenious Hunter",
    "Jack of All Trades",
    "Last Stand",
    "Legend: Alacrity",
    "Legend: Bloodline",
    "Legend: Haste",
    "Legend: Tenacity",
    "Lethal Tempo",
    "Manaflow Band",
    "Magical Footwear",
    "Nimbus Cloak",
    "Overgrowth",
    "Phase Rush",
    "Presence of Mind",
    "Press the Attack",
    "Revitalize",
    "Scorch",
    "Shield Bash",
    "Sudden Impact",
    "Summon Aery",
    "Taste of Blood",
    "Time Warp Tonic",
    "Transcendence",
    "Treasure Hunter",
    "Triumph",
    "Unflinching",
    "Unsealed Spellbook",
    "Waterwalking",
    "Zombie Ward",
)

MODIFIER_PACKET_VERSION = "modifier-packet-v1.0.0"


def _norm(value: Any) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value)).casefold()
    return "".join(char for char in decomposed if char.isalnum())


def _mentions_name(question: str, candidate: Any) -> bool:
    """Match an entity on token boundaries, avoiding Kalista→Alistar overlaps."""

    parts = re.findall(r"[a-z0-9]+", str(candidate).casefold())
    if not parts:
        return False
    pattern = r"(?<![a-z0-9])" + r"\W+".join(re.escape(part) for part in parts) + r"(?![a-z0-9])"
    return re.search(pattern, question.casefold()) is not None


def _wiki_url(title: str) -> str:
    return "https://wiki.leagueoflegends.com/en-us/" + quote(
        title.replace(" ", "_"), safe="_-'()"
    )


class LeagueOracleEngine:
    """Resident exact-packet calculator with explicit unsupported fallbacks."""

    def __init__(
        self,
        pack: Mapping[str, Any],
        *,
        raw_champion_root: Path | None = None,
    ) -> None:
        self.pack = pack
        self.fast = QuickMechanicsEngine(pack)
        self.patch = self.fast.patch
        self.raw_champion_root = raw_champion_root
        self._champions: list[Mapping[str, Any]] = []
        self._spells: dict[str, list[Mapping[str, Any]]] = {}
        self._mechanics: dict[int, Mapping[str, Any]] = {}
        self._mechanics_error: str | None = None
        self._items: dict[str, Mapping[str, Any]] = {}
        self._items_error: str | None = None
        self._load_mechanics_index()
        self._load_items()
        seen: set[str] = set()
        for record in pack.get("champions", {}).values():
            if not isinstance(record, Mapping):
                continue
            name = record.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            key = _norm(name)
            if key in seen:
                continue
            seen.add(key)
            self._champions.append(record)
            self._load_spells(record)

    def _load_mechanics_index(self) -> None:
        """Load the normalized formula graph only when its receipt matches.

        The fastpack already verifies the exact index hash while compiling
        champion stats.  Re-reading the adjacent index here keeps the pack
        compact while retaining a self-checking, network-free formula source.
        """

        if self.raw_champion_root is None:
            self._mechanics_error = "raw champion root is unavailable"
            return
        index_path = Path(self.raw_champion_root).parent.parent / "mechanics-index.json"
        try:
            index_bytes = index_path.read_bytes()
            payload = json.loads(index_bytes)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            self._mechanics_error = f"mechanics index cannot be read: {exc}"
            return
        expected = self.pack.get("index_sha256")
        actual = hashlib.sha256(index_bytes).hexdigest()
        if isinstance(expected, str) and expected and expected != actual:
            self._mechanics_error = "mechanics index hash does not match the compiled fastpack"
            return
        entries = payload.get("champions") if isinstance(payload, Mapping) else None
        if not isinstance(entries, list):
            self._mechanics_error = "mechanics index has no champion list"
            return
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            raw_id = entry.get("id")
            mechanics = entry.get("mechanics")
            if isinstance(raw_id, int) and isinstance(mechanics, Mapping):
                if raw_id in self._mechanics:
                    self._mechanics_error = "mechanics index contains duplicate champion ids"
                    self._mechanics.clear()
                    return
                self._mechanics[raw_id] = mechanics

    def _load_items(self) -> None:
        """Load standard in-store item stats after checking the manifest hash."""

        if self.raw_champion_root is None:
            self._items_error = "raw item payload is unavailable"
            return
        raw_root = Path(self.raw_champion_root).parent
        items_path = raw_root / "items.json"
        manifest_path = raw_root.parent / "manifest.json"
        try:
            items_bytes = items_path.read_bytes()
            payload = json.loads(items_bytes)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            self._items_error = f"item payload cannot be read: {exc}"
            return
        expected = None
        for entry in manifest.get("files", []) if isinstance(manifest, Mapping) else []:
            if isinstance(entry, Mapping) and entry.get("path") == "raw/items.json":
                expected = entry.get("sha256")
                break
        actual = hashlib.sha256(items_bytes).hexdigest()
        if not isinstance(expected, str) or expected != actual:
            self._items_error = "item payload hash does not match the patch manifest"
            return
        if not isinstance(payload, list):
            self._items_error = "item payload is not a list"
            return
        for item in payload:
            if not isinstance(item, Mapping):
                continue
            raw_id = item.get("id")
            name = item.get("name")
            if not isinstance(raw_id, int) or raw_id >= 10000 or not item.get("inStore"):
                continue
            if not isinstance(name, str) or not name.strip():
                continue
            stats = parse_static_item_stats(item)
            if not stats:
                continue
            key = _norm(name)
            candidate = {
                "id": raw_id,
                "name": name,
                "stats": stats,
                "has_passive": bool(re.search(r"<(?:passive|unique|active)", str(item.get("description", "")), re.I)),
                # Keep the patch-pinned raw fields alongside the normalized
                # static stats.  The turret-DPS optimizer needs to inspect
                # legal SR item eligibility and a small audited set of
                # structure-targeting passives without rereading the payload.
                "description": str(item.get("description", "")),
                "categories": tuple(str(value) for value in (item.get("categories") or []) if value),
                "display_in_item_sets": bool(item.get("displayInItemSets")),
                "in_store": bool(item.get("inStore")),
                "required_champion": item.get("requiredChampion"),
                "required_ally": item.get("requiredAlly"),
                "required_buff_currency_name": item.get("requiredBuffCurrencyName"),
                "price_total": item.get("priceTotal"),
                "is_enchantment": bool(item.get("isEnchantment")),
                "from_ids": tuple(int(value) for value in (item.get("from") or []) if isinstance(value, int)),
                "to_ids": tuple(int(value) for value in (item.get("to") or []) if isinstance(value, int)),
            }
            previous = self._items.get(key)
            # The raw packet carries alternate-map copies such as 223xxx and
            # 773xxx.  Prefer the standard 3xxx/6xxx Summoner's Rift copy,
            # then use the lowest id as a stable tie-breaker.  This keeps
            # existing exact item-stat answers deterministic while preventing
            # Arena variants from entering a normal SR search.
            def priority(value: Mapping[str, Any]) -> tuple[int, int]:
                item_id = int(value.get("id", 0))
                standard = int((3000 <= item_id <= 3999) or (6000 <= item_id <= 6999))
                return (-standard, item_id)

            if previous is None or priority(candidate) < priority(previous):
                self._items[key] = candidate

    def _load_spells(self, record: Mapping[str, Any]) -> None:
        if self.raw_champion_root is None:
            return
        raw_id = record.get("id")
        if not isinstance(raw_id, int):
            return
        path = self.raw_champion_root / f"{raw_id}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return
        spells = payload.get("spells") if isinstance(payload, Mapping) else None
        if not isinstance(spells, list):
            return
        key = _norm(str(record.get("name", "")))
        self._spells[key] = [
            spell
            for spell in spells
            if isinstance(spell, Mapping)
            and str(spell.get("spellKey", "")).lower() in {"q", "w", "e", "r"}
        ]

    def _resolve_champion(self, question: str) -> Mapping[str, Any] | None:
        normalized = _norm(question)
        matches: list[tuple[int, Mapping[str, Any]]] = []
        for record in self._champions:
            candidates = [record.get("name"), record.get("alias"), *(record.get("aliases") or [])]
            for candidate in candidates:
                alias = _norm(candidate)
                if alias and _mentions_name(question, candidate):
                    matches.append((len(alias), record))
        if not matches:
            return None
        matches.sort(key=lambda item: item[0], reverse=True)
        return matches[0][1]

    def _mechanics_spell(
        self, record: Mapping[str, Any], ability_key: str
    ) -> Mapping[str, Any] | None:
        raw_id = record.get("id")
        if not isinstance(raw_id, int):
            return None
        mechanics = self._mechanics.get(raw_id)
        if not isinstance(mechanics, Mapping):
            return None
        expected = _norm(str(record.get("name", ""))) + ability_key.casefold()
        matches = [
            spell
            for spell in mechanics.get("spells", [])
            if isinstance(spell, Mapping)
            and _norm(str(spell.get("script_name", ""))) == expected
        ]
        if len(matches) == 1:
            return matches[0]
        # A few champions expose a descriptive script name (for example
        # Malphite's ``SeismicShard``) instead of ChampionQ.  The raw client
        # spell name is the exact bridge; ambiguity still fails closed.
        raw_spell_names = {
            _norm(str(spell.get("name", "")))
            for spell in self._spells.get(_norm(str(record.get("name", ""))), [])
            if str(spell.get("spellKey", "")).casefold() == ability_key.casefold()
            and str(spell.get("name", "")).strip()
        }
        by_name = [
            spell
            for spell in mechanics.get("spells", [])
            if isinstance(spell, Mapping)
            and _norm(str(spell.get("script_name", ""))) in raw_spell_names
        ]
        return by_name[0] if len(by_name) == 1 else None

    def _resolve_item(self, question: str) -> Mapping[str, Any] | None:
        matches = [
            item
            for item in self._items.values()
            if _mentions_name(question, item.get("name", ""))
        ]
        if not matches:
            return None
        matches.sort(key=lambda item: len(_norm(str(item.get("name", "")))), reverse=True)
        return matches[0]

    def _source_links(
        self,
        record: Mapping[str, Any] | None,
        *,
        extra_pages: tuple[str, ...] = (),
        include_formula: bool = True,
    ) -> list[dict[str, str]]:
        links: list[dict[str, str]] = []
        if record is not None:
            source = record.get("source")
            if isinstance(source, Mapping):
                path = source.get("bin_json_path")
                if isinstance(path, str) and path:
                    client_patch = str(self.pack.get("client_patch") or "")
                    links.append(
                        {
                            "kind": "client",
                            "url": f"https://raw.communitydragon.org/{client_patch}/{path.lstrip('/')}",
                            "label": "patch-pinned CommunityDragon client data",
                        }
                    )
            name = record.get("name")
            if isinstance(name, str) and name:
                links.append(
                    {
                        "kind": "wiki",
                        "url": _wiki_url(name),
                        "label": "League Wiki champion page",
                    }
                )
        if include_formula:
            links.append(
                {
                    "kind": "wiki_formula",
                    "url": _wiki_url("Champion statistic"),
                    "label": "League Wiki stat-growth formula",
                }
            )
        for page in extra_pages:
            links.append(
                {
                    "kind": "wiki_ability",
                    "url": _wiki_url(page),
                    "label": "League Wiki interaction page",
                }
            )
        # Stable de-duplication keeps the answer compact without dropping the
        # source kind that explains why a link is present.
        unique: dict[str, dict[str, str]] = {}
        for link in links:
            unique.setdefault(link["url"], link)
        return list(unique.values())

    def _decorate(self, answer: Mapping[str, Any], question: str) -> dict[str, Any]:
        result = dict(answer)
        result["engine"] = "lol-oracle-v1"
        # Specialized handlers (for example the turret-DPS optimizer) return
        # their own multi-page source receipt.  Do not replace it with the
        # generic champion-only links at the final decoration boundary.
        if not isinstance(result.get("sources"), list) or not result.get("sources"):
            record = self._resolve_champion(question)
            result["sources"] = self._source_links(record)
        return result

    def _unsupported(
        self,
        question: str,
        reason: str,
        *,
        record: Mapping[str, Any] | None = None,
        extra_pages: tuple[str, ...] = (),
        extra_sources: tuple[dict[str, str], ...] = (),
    ) -> dict[str, Any]:
        answer = self.fast._unsupported(question=question, reason=reason)  # type: ignore[attr-defined]
        result = self._decorate(answer, question)
        if record is not None or extra_pages or extra_sources:
            links = self._source_links(
                record if record is not None else self._resolve_champion(question),
                extra_pages=extra_pages,
            )
            links.extend(extra_sources)
            unique: dict[str, dict[str, str]] = {}
            for link in links:
                unique.setdefault(link["url"], link)
            result["sources"] = list(unique.values())
        return result

    @staticmethod
    def _level(question: str) -> int | None:
        match = _LEVEL_RE.search(question)
        if match is None:
            return None
        level = int(match.group(1))
        return level if 1 <= level <= 18 else None

    @staticmethod
    def _levels(question: str) -> list[int]:
        return [int(match.group(1)) for match in _LEVEL_RE.finditer(question)]

    @staticmethod
    def _ability_rank_and_level(question: str) -> tuple[int, int] | None:
        """Parse both ability rank and champion level without conflating them.

        Players commonly write ``Q lvl 3 at level 6`` rather than the more
        formal ``rank-3 ... at level 6``.  Both ``lvl 3`` and ``level 6`` are
        level-shaped tokens, so the ability token must be removed before the
        champion-level token is selected.
        """

        rank_match = _RANK_RE.search(question)
        rank_span: tuple[int, int] | None = None
        if rank_match is not None:
            rank = int(rank_match.group(1))
            rank_span = rank_match.span()
        else:
            ability_level_match = _ABILITY_LEVEL_RE.search(question)
            if ability_level_match is None:
                return None
            rank = int(ability_level_match.group(1))
            rank_span = ability_level_match.span()

        level_matches = []
        for level_match in _LEVEL_RE.finditer(question):
            if rank_span is not None and level_match.start() < rank_span[1] and rank_span[0] < level_match.end():
                continue
            level_matches.append(level_match)
        if len(level_matches) != 1:
            return None
        level = int(level_matches[0].group(1))
        if not (1 <= rank <= 5 and 1 <= level <= 18):
            return None
        return rank, level

    @staticmethod
    def _format(value: float) -> int | float:
        rounded = round(float(value), 2)
        return int(rounded) if rounded.is_integer() else rounded

    def _envelope(
        self,
        *,
        question: str,
        intent: str,
        value: int | float,
        unit: str,
        display: str,
        calculation: str,
        assumptions: tuple[str, ...] = (),
        record: Mapping[str, Any] | None = None,
        extra_pages: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        return {
            "status": "available",
            "display": display,
            "value": value,
            "unit": unit,
            "patch": self.patch,
            "intent": intent,
            "assumptions": list(assumptions),
            "calculation": calculation,
            "provenance": {
                "engine": "lol-oracle-v1",
                "pack_sha256": self.pack.get("source_hash"),
                "source": self.pack.get("source"),
            },
            "sources": self._source_links(record, extra_pages=extra_pages),
        }

    @staticmethod
    def _merge_sources(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge response links without losing a source receipt's metadata."""

        unique: dict[str, dict[str, Any]] = {}
        for group in groups:
            for source in group:
                url = str(source.get("url", ""))
                if url:
                    previous = unique.get(url)
                    if previous is None or (
                        source.get("revision_id") is not None
                        and previous.get("revision_id") is None
                    ):
                        unique[url] = source
        return list(unique.values())

    @staticmethod
    def _clock_seconds(question: str) -> int | None:
        match = _CLOCK_RE.search(question)
        if match is None:
            return None
        if match.group(1) is not None:
            minutes, seconds = int(match.group(1)), int(match.group(2))
            if seconds >= 60:
                return None
            return minutes * 60 + seconds
        minutes = float(match.group(3))
        if not math.isfinite(minutes) or minutes < 0:
            return None
        return int(round(minutes * 60.0))

    @staticmethod
    def _stack_count(question: str) -> int | None:
        matches = _STACK_RE.findall(question)
        if len(matches) != 1:
            return None
        count = int(matches[0])
        return count if count >= 0 else None

    @staticmethod
    def _term_is_negated(question: str, term: str) -> bool:
        """Return whether a loadout term is explicitly absent.

        Negative state is deliberately narrow.  A phrase such as ``not fully
        stacked`` is not treated as zero stacks; it is an unknown 0-9 state and
        must remain unavailable until the caller supplies the exact count.
        """

        return bool(
            re.search(
                rf"\b(?:no|without|zero|none|empty|itemless|rune[- ]less|runeless)\b"
                rf"[^.?!;]*\b{re.escape(term)}s?\b",
                question,
                re.I,
            )
        )

    def _resolve_items(self, question: str) -> list[Mapping[str, Any]]:
        """Resolve every positively mentioned exact item, stably and once."""

        matches: list[Mapping[str, Any]] = []
        normalized_question = _norm(question)
        for item in self._items.values():
            name = str(item.get("name", ""))
            normalized_name = _norm(name)
            if not normalized_name or (
                normalized_name not in normalized_question
                and f"{normalized_name}s" not in normalized_question
            ):
                continue
            mentioned = name and (
                _mentions_name(question, name)
                or _mentions_name(question, f"{name}s")
            )
            if mentioned and not self._term_is_negated(question, name):
                matches.append(item)
        matches.sort(key=lambda item: (-len(_norm(str(item.get("name", "")))), int(item.get("id", 0))))
        unique: dict[str, Mapping[str, Any]] = {}
        for item in matches:
            unique.setdefault(_norm(str(item.get("name", ""))), item)
        return list(unique.values())

    @staticmethod
    def _item_quantity(question: str, name: str) -> int | None:
        """Parse an explicit item count, rejecting contradictory counts."""

        number_words = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
        }
        pattern = rf"\b(\d+|one|two|three|four|five|six)\s+{re.escape(name)}s?\b"
        matches = re.findall(pattern, question, re.I)
        if not matches:
            return 1
        quantities: list[int] = []
        for raw in matches:
            value = number_words.get(raw.casefold(), int(raw) if raw.isdigit() else 0)
            if value < 1:
                return None
            quantities.append(value)
        if len(set(quantities)) != 1:
            return None
        return quantities[0]

    @staticmethod
    def _resolve_named_runes(question: str) -> list[str]:
        """Return named runes mentioned positively in the question."""

        return [
            name
            for name in _KNOWN_RUNE_NAMES
            if _mentions_name(question, name) and not LeagueOracleEngine._term_is_negated(question, name)
        ]

    @classmethod
    def _manaflow_stack_state(cls, question: str) -> int | None:
        """Parse an explicit Manaflow state without assuming full stacks."""

        if cls._term_is_negated(question, "Manaflow Band"):
            return 0
        if re.search(r"\b(?:not|isn't|is not)\s+(?:fully|completely|max(?:imum)?)\s*stack(?:ed|s)?\b", question, re.I):
            return None
        if re.search(r"\b(?:fully|completely|max(?:imum)?)\s*stack(?:ed|s)?\b", question, re.I):
            return 10
        if re.search(r"\b(?:unstacked|no\s+stacks?)\b", question, re.I):
            return 0
        return cls._stack_count(question)

    def _has_positive_modifier_state(self, question: str) -> bool:
        """Detect modifiers that the base cast-budget contract cannot ignore."""

        if self._resolve_items(question):
            return True
        if re.search(r"\b(?:item|items|equipped|build)\b", question, re.I) and not self._term_is_negated(question, "item"):
            return True
        named_runes = self._resolve_named_runes(question)
        if named_runes:
            return True
        if re.search(r"Manaflow\s+Band", question, re.I) and not self._term_is_negated(question, "Manaflow Band"):
            return True
        if re.search(r"\brunes?\b", question, re.I) and not self._term_is_negated(question, "rune"):
            return True
        for term in ("buff", "debuff", "passive", "shield", "penetration", "crit"):
            if re.search(rf"\b{term}s?\b", question, re.I) and not self._term_is_negated(question, term):
                return True
        return False

    def _rune_interaction(self, question: str) -> dict[str, Any] | None:
        """Evaluate the static, permanently stacked Manaflow component.

        Trigger timing, refund/regen behavior, and interactions with other
        runes are deliberately outside this path.  The question must state a
        level and an explicit stack count so the calculation is closed.
        """

        if not re.search(r"\brune(?:s)?\b|Manaflow\s+Band", question, re.I):
            return None
        if not re.search(r"Manaflow\s+Band", question, re.I):
            return None
        level_matches = self._levels(question)
        stacks = self._stack_count(question)
        if len(level_matches) != 1 or stacks is None:
            return self._unsupported(
                question,
                "Manaflow Band arithmetic requires exactly one champion level and one explicit stack count",
                extra_pages=("Manaflow Band",),
                extra_sources=(wiki_rule_source("Manaflow Band"),),
            )
        level = level_matches[0]
        record = self._resolve_champion(question)
        if record is None:
            return self._unsupported(
                question,
                "champion is absent from the exact patch packet",
                extra_pages=("Manaflow Band",),
                extra_sources=(wiki_rule_source("Manaflow Band"),),
            )
        if record.get("resource_type") != "mana":
            return self._unsupported(
                question,
                "Manaflow Band is substituted for champions without mana",
                record=record,
                extra_pages=("Manaflow Band",),
                extra_sources=(wiki_rule_source("Manaflow Band"),),
            )
        if re.search(r"items?|other runes?|buffs?|regeneration|refund|missing mana", question, re.I):
            return self._unsupported(
                question,
                "only Manaflow Band's permanent maximum-mana component is in this contract",
                record=record,
                extra_pages=("Manaflow Band",),
                extra_sources=(wiki_rule_source("Manaflow Band"),),
            )
        row = record.get("levels", {}).get(str(level))
        base = row.get("max_resource") if isinstance(row, Mapping) else None
        if not isinstance(base, (int, float)):
            return self._unsupported(
                question,
                "maximum mana is unavailable at the requested level",
                record=record,
                extra_pages=("Manaflow Band",),
                extra_sources=(wiki_rule_source("Manaflow Band"),),
            )
        bonus = manaflow_bonus(stacks)
        total = self._format(float(base) + bonus)
        capped = min(stacks, 10)
        answer = self._envelope(
            question=question,
            intent="rune_static_stack",
            value=total,
            unit="maximum mana",
            display=f"{total} maximum mana",
            calculation=(
                f"{float(base):.2f} base maximum mana + ({capped} effective stacks × 25) "
                f"= {float(total):.2f} maximum mana."
            ),
            assumptions=(
                f"{record['name']} level {level}",
                f"Manaflow Band at {stacks} stated stacks (10-stack cap)",
                "no items, other runes, buffs, or post-cap missing-mana regeneration",
            ),
            record=record,
            extra_pages=("Manaflow Band",),
        )
        answer["sources"] = self._merge_sources(
            answer.get("sources", []), [wiki_rule_source("Manaflow Band")]
        )
        return answer

    def _structure_stat(self, question: str) -> dict[str, Any] | None:
        """Evaluate explicit Summoner's Rift turret facts and clocks."""

        if not re.search(r"\b(?:turret|tower|plates?|plating)\b", question, re.I):
            return None
        lower = question.casefold()

        # Plate gold is independent of which lane turret supplied the plates;
        # handle it before requiring an explicit turret kind.
        plate_match = re.search(r"\b(\d+)\s+(?:turret\s+)?plates?\b", lower)
        if plate_match and re.search(r"gold|reward|bounty", lower):
            plates = int(plate_match.group(1))
            source = wiki_rule_source("Turret Plating")
            if not 1 <= plates <= 5:
                return self._unsupported(
                    question,
                    "turret plate count must be in the exact range 1-5",
                    extra_sources=(source,),
                )
            total = plates * 120
            answer = self._envelope(
                question=question,
                intent="structure_static_rule",
                value=total,
                unit="local gold",
                display=f"{total} local gold",
                calculation=f"{plates} plates × 120 local gold per plate = {total} local gold.",
                assumptions=("Summoner's Rift turret plating", "no first-turret or global reward"),
                extra_pages=("Turret",),
            )
            answer["sources"] = self._merge_sources(answer.get("sources", []), [source])
            return answer

        kind: str | None = None
        for candidate in ("outer", "inner", "inhibitor", "nexus"):
            if re.search(rf"\b{candidate}\s+(?:turret|tower)\b|\b{candidate}\b", lower):
                kind = candidate
                break
        if kind is None:
            return self._unsupported(
                question,
                "an explicit Summoner's Rift turret kind (outer, inner, inhibitor, or nexus) is required",
                extra_sources=(wiki_rule_source("Turret"),),
            )
        if re.search(r"\b(?:ARAM|Howling Abyss|Arena|mode|fortification|minion|wave|current|nearby|enemy champions?)\b", question, re.I):
            return self._unsupported(
                question,
                "mode, live minion state, fortification, and nearby-champion modifiers are outside the static turret contract",
                extra_sources=(wiki_rule_source("Turret"),),
            )

        source_title = "Turret Plating" if re.search(r"\b(?:plate|plating)\b", lower) else "Turret"
        source = wiki_rule_source(source_title)

        if re.search(r"\b(?:health|hp)\b", lower):
            value = int(STRUCTURES[kind]["health"])
            calculation = f"{kind.title()} turret base health = {value}."
        elif re.search(r"\b(?:attack damage|attack dmg|ad)\b", lower):
            seconds = self._clock_seconds(question)
            if seconds is None:
                return self._unsupported(
                    question,
                    "turret attack damage requires one explicit game clock such as 5:30",
                    extra_sources=(source,),
                )
            value = turret_attack_damage(kind, seconds)
            calculation = (
                f"{kind.title()} turret attack damage at {seconds // 60}:{seconds % 60:02d} = "
                f"{value} from the page's minute ramp and cap."
            )
        elif re.search(r"\b(?:armor|magic resistance|magic resist|mr)\b", lower):
            if self._clock_seconds(question) is not None and kind == "outer":
                return self._unsupported(
                    question,
                    "outer-turret time decay is outside the static resistance slice",
                    extra_sources=(source,),
                )
            value = 60
            unit = "armor" if re.search(r"\barmor\b", lower) else "magic resistance"
            calculation = f"{kind.title()} turret base {unit} = 60."
        else:
            return self._unsupported(
                question,
                "the static turret contract supports health, attack damage, armor, magic resistance, and plate gold",
                extra_sources=(source,),
            )

        unit = (
            "health"
            if re.search(r"\b(?:health|hp)\b", lower)
            else "attack damage"
            if re.search(r"\b(?:attack damage|attack dmg|ad)\b", lower)
            else "armor"
            if re.search(r"\barmor\b", lower)
            else "magic resistance"
        )
        answer = self._envelope(
            question=question,
            intent="structure_static_rule",
            value=value,
            unit=unit,
            display=f"{value} {unit}",
            calculation=calculation,
            assumptions=("Summoner's Rift turret", "no plates, fortification, minion, or nearby-champion modifiers"),
            extra_pages=("Turret",),
        )
        answer["sources"] = self._merge_sources(answer.get("sources", []), [source])
        return answer

    def _stack_interaction(self, question: str) -> dict[str, Any] | None:
        """Evaluate explicit, permanent stack-to-stat formulas."""

        if not re.search(r"\b(?:stack|stacks|souls?|marks?|mist)\b", question, re.I):
            return None
        stacks = self._stack_count(question)
        if stacks is None:
            return None
        lower = question.casefold()

        if _mentions_name(question, "Nasus") and re.search(r"siphoning strike|\bnasus'?s?\s+q\b|\bq\b", lower):
            rank_match = _RANK_RE.search(question)
            if rank_match is None:
                ability_level_match = _ABILITY_LEVEL_RE.search(question)
                rank = int(ability_level_match.group(1)) if ability_level_match else None
            else:
                rank = int(rank_match.group(1))
            if rank is None or not 1 <= rank <= 5:
                return self._unsupported(
                    question,
                    "Nasus Siphoning Strike requires an explicit rank 1-5",
                    extra_sources=(wiki_rule_source("Template:Data Nasus/Siphoning Strike"),),
                )
            value = nasus_siphoning_strike_bonus(rank, stacks)
            source = wiki_rule_source("Template:Data Nasus/Siphoning Strike")
            answer = self._envelope(
                question=question,
                intent="stacked_ability_stat",
                value=value,
                unit="bonus physical damage",
                display=f"{value} bonus physical damage",
                calculation=f"{40 + 20 * (rank - 1)} rank-{rank} base + {stacks} stored Q stacks = {value}.",
                assumptions=("Nasus Siphoning Strike", "no critical-strike modifier, target modifier, or other on-hit effect"),
                extra_pages=("Nasus",),
            )
            answer["sources"] = self._merge_sources(answer.get("sources", []), [source])
            return answer

        if _mentions_name(question, "Thresh") and re.search(r"soul", lower):
            ap, armor = thresh_soul_stats(stacks)
            source = wiki_rule_source("Template:Data Thresh/Damnation")
            if re.search(r"\b(?:armor|bonus armor)\b", lower):
                value, unit, label = armor, "bonus armor", "bonus armor"
            elif re.search(r"\b(?:ability power|\bap\b)", lower):
                value, unit, label = ap, "ability power", "ability power"
            else:
                return self._unsupported(
                    question,
                    "Thresh soul questions must request ability power or bonus armor",
                    extra_sources=(source,),
                )
            answer = self._envelope(
                question=question,
                intent="stacked_passive_stat",
                value=value,
                unit=unit,
                display=f"{value} {unit}",
                calculation=f"{stacks} souls × 1 {label} per soul = {value} {label}.",
                assumptions=("Thresh Damnation", "souls are already collected; no other items, runes, or buffs"),
                extra_pages=("Thresh",),
            )
            answer["sources"] = self._merge_sources(answer.get("sources", []), [source])
            return answer

        if _mentions_name(question, "Senna") and re.search(r"mist|absolution", lower):
            ad, range_bonus, crit = senna_mist_stats(stacks)
            source = wiki_rule_source("Template:Data Senna/Absolution")
            if re.search(r"\b(?:attack damage|bonus ad|\bad\b)", lower):
                value, unit, calculation = self._format(ad), "bonus attack damage", f"{stacks} Mist × 0.75 bonus AD = {ad:.2f}."
            elif re.search(r"\brange\b", lower):
                value, unit, calculation = range_bonus, "bonus attack range", f"floor({stacks}/20) × 20 bonus range = {range_bonus}."
            elif re.search(r"critical|\bcrit\b", lower):
                value, unit, calculation = crit, "critical strike chance", f"floor({stacks}/20) × 10% critical strike chance = {crit}%."
            else:
                return self._unsupported(
                    question,
                    "Senna Mist questions must request bonus attack damage, range, or critical strike chance",
                    extra_sources=(source,),
                )
            answer = self._envelope(
                question=question,
                intent="stacked_passive_stat",
                value=value,
                unit=unit,
                display=f"{value} {unit}",
                calculation=calculation,
                assumptions=("Senna Absolution", "Mist is already collected; no item critical chance or lifesteal conversion"),
                extra_pages=("Senna",),
            )
            answer["sources"] = self._merge_sources(answer.get("sources", []), [source])
            return answer

        if _mentions_name(question, "Kindred") and re.search(r"mark", lower):
            value = kindred_bonus_range(stacks)
            source = wiki_rule_source("Template:Data Kindred/Mark of the Kindred")
            answer = self._envelope(
                question=question,
                intent="stacked_passive_stat",
                value=value,
                unit="bonus attack range",
                display=f"{value} bonus attack range",
                calculation=(
                    "0 below 4 marks; otherwise 75 + 25 × floor((marks−4)/3), "
                    f"capped at 250 = {value}."
                ),
                assumptions=("Kindred Mark of the Kindred", "marks are already collected; no ability-specific damage scaling"),
                extra_pages=("Kindred",),
            )
            answer["sources"] = self._merge_sources(answer.get("sources", []), [source])
            return answer

        if re.search(r"touch of the void", lower):
            if stacks > 3:
                return self._unsupported(
                    question,
                    "the executable Touch of the Void burn slice supports one to three stacks",
                    extra_sources=(wiki_rule_source("Template:Buff data Touch of the Void"),),
                )
            ranged = bool(re.search(r"\branged\b", lower))
            per_tick, total = touch_of_the_void_burn(stacks, ranged=ranged)
            source = wiki_rule_source("Template:Buff data Touch of the Void")
            if re.search(r"per[- ]tick|each tick|tick", lower):
                value, unit, calculation = per_tick, "true damage per tick", f"{per_tick} true damage every 0.5 seconds for {stacks} stack(s)."
            else:
                value, unit, calculation = total, "true damage over 4 seconds", f"{per_tick} true damage × 8 ticks = {total} true damage."
            answer = self._envelope(
                question=question,
                intent="stacked_structure_effect",
                value=value,
                unit=unit,
                display=f"{value} {unit}",
                calculation=calculation,
                assumptions=(
                    "Summoner's Rift Touch of the Void",
                    f"{stacks} stacks, {'ranged' if ranged else 'melee'} triggering attack",
                    "the burn applies and is not refreshed by another attack during the four-second window",
                ),
                extra_pages=("Touch of the Void",),
            )
            answer["sources"] = self._merge_sources(answer.get("sources", []), [source])
            return answer

        return None

    def _ordered_damage_sequence(self, question: str) -> dict[str, Any] | None:
        """Run an explicit, typed damage sequence through the state kernel.

        This deliberately accepts numeric damage events rather than guessing
        ability effects.  The target's starting health and both resistances
        must be present, and the question must state order with ``then`` or
        ``in exact order``.  That gives the transition engine a closed state
        while still exercising the order-sensitive combat path.
        """

        if not re.search(r"\b(?:in\s+exact\s+order|in\s+order|then)\b", question, re.I):
            return None
        if not re.search(r"remaining\s+health|health\s+remaining|what health", question, re.I):
            return None
        if not re.search(r"\b(?:physical|magic|true)\s+damage\b", question, re.I):
            return None
        if re.search(r"\b(?:shield|item|rune|buff|debuff|passive|critical|penetration|target health)\b", question, re.I):
            return self._unsupported(
                question,
                "ordered sequence requires a numeric typed-damage contract; shields, items, runes, and modifiers are not included",
                extra_sources=(
                    {"kind": "wiki", "url": _wiki_url("Damage"), "label": "League Wiki damage rules"},
                    {"kind": "wiki", "url": _wiki_url("Armor"), "label": "League Wiki armor rules"},
                    {"kind": "wiki", "url": _wiki_url("Magic resistance"), "label": "League Wiki magic-resistance rules"},
                ),
            )
        health_match = _INITIAL_HEALTH_RE.search(question)
        armor_match = _ARMOR_RE.search(question)
        mr_match = _MAGIC_RESIST_RE.search(question)
        events = [
            (float(match.group(1)), match.group(2).casefold())
            for match in _SEQUENCE_DAMAGE_RE.finditer(question)
        ]
        if health_match is None or armor_match is None or mr_match is None or not 2 <= len(events) <= 6:
            return self._unsupported(
                question,
                "ordered sequence requires one starting health, armor, magic resistance, and 2-6 typed damage events",
                extra_sources=(
                    {"kind": "wiki", "url": _wiki_url("Damage"), "label": "League Wiki damage rules"},
                    {"kind": "wiki", "url": _wiki_url("Armor"), "label": "League Wiki armor rules"},
                    {"kind": "wiki", "url": _wiki_url("Magic resistance"), "label": "League Wiki magic-resistance rules"},
                ),
            )
        health = float(health_match.group(1))
        armor = float(armor_match.group(1))
        magic_resist = float(mr_match.group(1))
        if not all(math.isfinite(value) and value >= 0 for value in (health, armor, magic_resist)):
            return self._unsupported(question, "starting health and resistances must be finite and non-negative")
        if any(not math.isfinite(amount) or amount < 0 for amount, _ in events):
            return self._unsupported(question, "damage events must be finite and non-negative")

        target = Combatant(
            entity_id="target",
            team_id="enemy",
            champion_id="explicit-target",
            health=health,
            max_health=health,
            stats={"armor": armor, "magic_resist": magic_resist},
        )
        initial = GameState(entities={"target": target})
        scheduled = tuple(
            Event(
                at_ms=index + 1,
                priority=0,
                source_id=f"sequence-{index + 1}",
                ordinal=index,
                effect=Damage(
                    source_id=f"sequence-{index + 1}",
                    target_id="target",
                    amount=amount,
                    damage_type=damage_type,
                ),
            )
            for index, (amount, damage_type) in enumerate(events)
        )
        result = MechanicsEngine().run(initial, scheduled)
        if not result.available:
            reasons = "; ".join(item.message for item in result.unknowns)
            return self._unsupported(question, f"ordered sequence kernel returned unknown: {reasons}")
        final_health = result.state.entities["target"].health
        # Do not silently model effects after a lethal event.  The benchmark
        # uses non-lethal sequences, but this guard keeps general queries honest.
        for transition in result.trace.transitions[:-1]:
            applied = transition.get("applied", {})
            if isinstance(applied, Mapping) and float(applied.get("remaining_health", 1.0)) <= 0:
                return self._unsupported(question, "a lethal intermediate event requires death-state semantics")

        pieces: list[str] = []
        for amount, damage_type in events:
            if damage_type == "true":
                pieces.append(f"{amount:g} true")
            else:
                resistance = armor if damage_type == "physical" else magic_resist
                dealt = amount * 100.0 / (100.0 + resistance)
                pieces.append(f"{amount:g} {damage_type} × 100/(100+{resistance:g}) = {dealt:.2f}")
        source_links = [
            {"kind": "wiki", "url": _wiki_url("Damage"), "label": "League Wiki damage rules"},
            {"kind": "wiki", "url": _wiki_url("Armor"), "label": "League Wiki armor rules"},
            {"kind": "wiki", "url": _wiki_url("Magic resistance"), "label": "League Wiki magic-resistance rules"},
        ]
        answer = self._envelope(
            question=question,
            intent="ordered_damage_sequence",
            value=self._format(final_health),
            unit="remaining health",
            display=f"{self._format(final_health)} remaining health",
            calculation=(
                f"{health:.2f} starting health − " + " − ".join(pieces) +
                f" = {final_health:.2f} remaining health; events executed in stated order."
            ),
            assumptions=(
                f"{armor:g} armor and {magic_resist:g} magic resistance remain constant",
                "no shields, penetration, items, runes, buffs, or lethal intermediate event",
                f"{len(events)} explicit numeric damage events",
            ),
        )
        answer["sources"] = source_links
        answer["provenance"]["trace_sha256"] = result.trace.trace_sha256
        answer["provenance"]["final_state_sha256"] = result.trace.final_state_sha256
        return answer

    def _ability_budget_components(
        self, question: str
    ) -> tuple[int, int, Mapping[str, Any], Mapping[str, Any], float, float] | dict[str, Any] | None:
        """Resolve the patch-pinned base inputs for a cast-budget question."""

        if not re.search(r"\b(?:casts?|uses?)\b", question, re.I):
            return None
        context = self._ability_rank_and_level(question)
        if context is None:
            return None
        rank, level = context
        record = self._resolve_champion(question)
        if record is None:
            return self._unsupported(question, "champion is absent from the exact patch packet")
        spells = self._spells.get(_norm(str(record.get("name", ""))), [])
        normalized = _norm(question)
        spell: Mapping[str, Any] | None = None
        for candidate in spells:
            name = _norm(str(candidate.get("name", "")))
            if name and name in normalized:
                spell = candidate
                break
        if spell is None:
            key_match = _ABILITY_KEY_RE.search(question)
            if key_match:
                key = key_match.group(1).lower()
                spell = next(
                    (candidate for candidate in spells if str(candidate.get("spellKey", "")).lower() == key),
                    None,
                )
        if spell is None:
            return self._unsupported(question, "ability name or Q/W/E/R key is required")
        costs = spell.get("costCoefficients")
        if not isinstance(costs, list) or rank < 1 or rank > len(costs):
            return self._unsupported(question, "ranked ability cost is absent from the exact client payload")
        try:
            # CommunityDragon's cost array is rank-1 indexed, unlike some
            # effect arrays that retain a leading display-only slot.
            cost = float(costs[rank - 1])
        except (TypeError, ValueError):
            return self._unsupported(question, "ability cost is not numeric")
        if not math.isfinite(cost) or cost <= 0:
            return self._unsupported(question, "ability has no positive finite resource cost")
        level_row = record.get("levels", {}).get(str(level))
        if not isinstance(level_row, Mapping):
            return self._unsupported(question, "requested level is absent from the exact patch packet")
        maximum = level_row.get("max_resource")
        if not isinstance(maximum, (int, float)) or not math.isfinite(float(maximum)):
            return self._unsupported(question, "maximum resource is unavailable for this champion")
        return rank, level, record, spell, cost, float(maximum)

    def _ability_budget_with_modifiers(
        self,
        question: str,
        components: tuple[int, int, Mapping[str, Any], Mapping[str, Any], float, float],
    ) -> dict[str, Any]:
        """Calculate a cast budget only when every modifier has closed state.

        The first version of the oracle matched ``casts`` before looking at a
        loadout, which made a question containing an item or rune return a
        plausible but base-only number.  This packet is intentionally small:
        visible static item mana and Manaflow Band's permanent maximum-mana
        component are executable; passive, trigger, and unknown rune effects
        remain fail-closed.
        """

        rank, level, record, spell, cost, base_resource = components
        spell_name = str(spell.get("name") or spell.get("spellKey") or "ability")
        items = self._resolve_items(question)
        generic_item_mention = bool(re.search(r"\b(?:item|items|equipped|build)\b", question, re.I)) and not self._term_is_negated(question, "item")
        if generic_item_mention and not items:
            return self._unsupported(
                question,
                "an item is mentioned but no exact patch-packet item identity was resolved",
                record=record,
                extra_pages=(f"{record['name']}/{spell_name}",),
            )

        item_bonus = 0.0
        item_parts: list[str] = []
        item_sources: list[dict[str, Any]] = []
        item_packet: list[dict[str, Any]] = []
        for item in items:
            item_name = str(item.get("name") or "item")
            quantity = self._item_quantity(question, item_name)
            if quantity is None:
                return self._unsupported(
                    question,
                    f"{item_name} has contradictory quantities; state one explicit quantity",
                    record=record,
                    extra_pages=(f"{record['name']}/{spell_name}", item_name),
                )
            if item.get("has_passive"):
                return self._unsupported(
                    question,
                    f"{item_name} has passive or trigger text; its state must be modeled before combining it with a cast budget",
                    record=record,
                    extra_pages=(f"{record['name']}/{spell_name}", item_name),
                )
            stat = item.get("stats", {}).get("mana") if isinstance(item.get("stats"), Mapping) else None
            if not isinstance(stat, Mapping):
                # A visible static item with no mana stat contributes zero to
                # a mana cast budget.  It is still recorded so the answer does
                # not silently discard a stated loadout component.
                item_parts.append(f"{quantity}x {item_name} (+0.00 maximum mana)")
                item_packet.append(
                    {
                        "id": item.get("id"),
                        "name": item_name,
                        "quantity": quantity,
                        "static_max_resource": 0,
                        "state": "visible_static_only",
                    }
                )
                item_sources.extend(
                    [
                        {
                            "kind": "wiki_item",
                            "url": _wiki_url(item_name),
                            "label": "League Wiki item page",
                        },
                        {
                            "kind": "client_item",
                            "url": f"https://raw.communitydragon.org/{self.pack.get('client_patch')}/plugins/rcp-be-lol-game-data/global/default/v1/items.json",
                            "label": "patch-pinned CommunityDragon item data",
                        },
                    ]
                )
                continue
            if stat.get("percent"):
                return self._unsupported(
                    question,
                    f"{item_name} uses percentage mana and needs a multiplicative stat contract",
                    record=record,
                    extra_pages=(f"{record['name']}/{spell_name}", item_name),
                )
            amount = stat.get("value")
            if not isinstance(amount, (int, float)) or isinstance(amount, bool) or not math.isfinite(float(amount)):
                return self._unsupported(
                    question,
                    f"{item_name} static mana is not numeric",
                    record=record,
                    extra_pages=(f"{record['name']}/{spell_name}", item_name),
                )
            item_bonus += float(amount) * quantity
            item_parts.append(f"{quantity}x {item_name} (+{float(amount) * quantity:.2f} maximum mana)")
            item_packet.append(
                {
                    "id": item.get("id"),
                    "name": item_name,
                    "quantity": quantity,
                    "static_max_resource": self._format(float(amount)),
                    "state": "visible_static_only",
                }
            )
            item_sources.extend(
                [
                    {
                        "kind": "wiki_item",
                        "url": _wiki_url(item_name),
                        "label": "League Wiki item page",
                    },
                    {
                        "kind": "client_item",
                        "url": f"https://raw.communitydragon.org/{self.pack.get('client_patch')}/plugins/rcp-be-lol-game-data/global/default/v1/items.json",
                        "label": "patch-pinned CommunityDragon item data",
                    },
                ]
            )

        rune_parts: list[str] = []
        rune_sources: list[dict[str, Any]] = []
        rune_packet: list[dict[str, Any]] = []
        rune_bonus = 0.0
        named_runes = self._resolve_named_runes(question)
        manaflow_positive = "Manaflow Band" in named_runes or (
            bool(re.search(r"Manaflow\s+Band", question, re.I))
            and not self._term_is_negated(question, "Manaflow Band")
        )
        generic_rune_mention = bool(re.search(r"\brunes?\b", question, re.I)) and not self._term_is_negated(question, "rune")
        unsupported_named_runes = [name for name in named_runes if name != "Manaflow Band"]
        if unsupported_named_runes:
            return self._unsupported(
                question,
                "named rune(s) " + ", ".join(unsupported_named_runes) + " need revision-receipted effect rules",
                record=record,
                extra_pages=(f"{record['name']}/{spell_name}",),
            )
        if generic_rune_mention and not manaflow_positive:
            return self._unsupported(
                question,
                "only Manaflow Band's permanent maximum-mana component is in the validated cast-budget packet; other runes need explicit effect rules",
                record=record,
                extra_pages=(f"{record['name']}/{spell_name}",),
            )
        if manaflow_positive:
            stacks = self._manaflow_stack_state(question)
            if stacks is None:
                return self._unsupported(
                    question,
                    "Manaflow Band stack state is required; state 0 stacks, an exact count, or fully stacked",
                    record=record,
                    extra_pages=(f"{record['name']}/{spell_name}",),
                    extra_sources=(wiki_rule_source("Manaflow Band"),),
                )
            rune_bonus = float(manaflow_bonus(stacks))
            effective = min(stacks, 10)
            rune_parts.append(f"Manaflow Band {stacks} stated stacks (+{rune_bonus:.2f} maximum mana; {effective} effective)")
            rune_packet.append(
                {
                    "name": "Manaflow Band",
                    "stacks": stacks,
                    "effective_stacks": effective,
                    "static_max_resource": self._format(rune_bonus),
                    "state": "explicit_permanent_component",
                }
            )
            rune_sources.append(wiki_rule_source("Manaflow Band"))

        for term in ("buff", "debuff", "passive", "shield", "penetration", "crit"):
            if re.search(rf"\b{term}s?\b", question, re.I) and not self._term_is_negated(question, term):
                return self._unsupported(
                    question,
                    "temporary buffs, debuffs, shields, penetration, and critical modifiers are outside the cast-budget packet",
                    record=record,
                    extra_pages=(f"{record['name']}/{spell_name}",),
                )

        total_resource = base_resource + item_bonus + rune_bonus
        casts = math.floor(total_resource / cost)
        remainder = self._format(total_resource - casts * cost)
        components_text = " + ".join([f"{base_resource:.2f} base maximum mana", *item_parts, *rune_parts])
        calculation = (
            f"floor(({components_text}) / {cost:.2f} rank-{rank} cost) = "
            f"floor({total_resource:.2f} / {cost:.2f}) = {casts} casts; {remainder} mana remains."
        )
        assumptions = [
            f"{record['name']} level {level}",
            f"rank-{rank} {spell_name}",
            "full resource, no regeneration",
            *item_parts,
            *rune_parts,
        ]
        sources = self._source_links(record, extra_pages=(f"{record['name']}/{spell_name}",))
        sources = self._merge_sources(sources, item_sources, rune_sources)
        answer = self._envelope(
            question=question,
            intent="ability_cast_budget",
            value=casts,
            unit="casts",
            display=f"{casts} casts ({remainder} mana remaining)",
            calculation=calculation,
            assumptions=tuple(assumptions),
            record=record,
            extra_pages=(f"{record['name']}/{spell_name}",),
        )
        answer["sources"] = sources
        answer["remainder"] = remainder
        answer["resource_before"] = self._format(total_resource)
        answer["modifier_components"] = {
            "base_max_resource": self._format(base_resource),
            "item_bonus": self._format(item_bonus),
            "rune_bonus": self._format(rune_bonus),
        }
        answer["modifier_packet"] = {
            "version": MODIFIER_PACKET_VERSION,
            "items": item_packet,
            "runes": rune_packet,
            "base_max_resource": self._format(base_resource),
            "total_max_resource": self._format(total_resource),
        }
        return answer

    def _ability_budget(self, question: str) -> dict[str, Any] | None:
        components = self._ability_budget_components(question)
        if components is None:
            return None
        if isinstance(components, Mapping):
            return dict(components)
        if self._has_positive_modifier_state(question):
            return self._ability_budget_with_modifiers(question, components)
        rank, level, record, spell, cost, maximum = components
        spell_name = str(spell.get("name") or spell.get("spellKey") or "ability")
        casts = math.floor(maximum / cost)
        return self._envelope(
            question=question,
            intent="ability_cast_budget",
            value=casts,
            unit="casts",
            display=f"{casts} casts",
            calculation=f"floor({maximum:.2f} maximum resource / {cost:.2f} rank-{rank} cost).",
            assumptions=(
                f"{record['name']} level {level}",
                f"rank-{rank} {spell_name}",
                "full resource, no regeneration, no items/runes/buffs",
            ),
            record=record,
            extra_pages=(f"{record['name']}/{spell_name}",),
        )

    def _item_stat(self, question: str) -> dict[str, Any] | None:
        if not re.search(r"\bitem(?:s)?\b|\bequipped\b", question, re.I):
            return None
        if self._items_error is not None:
            return self._unsupported(question, self._items_error)
        item = self._resolve_item(question)
        if item is None:
            return None
        levels = self._levels(question)
        if len(levels) != 1:
            return self._unsupported(question, "exactly one champion level is required")
        level = levels[0]
        record = self._resolve_champion(question)
        if record is None:
            return self._unsupported(question, "champion is absent from the exact patch packet")
        lower = question.casefold()
        if re.search(r"attack\s+damage|\bad\b", lower):
            item_field, champ_field, label = "attack_damage", "attack_damage", "attack damage"
        elif re.search(r"ability\s+power|\bap\b", lower):
            item_field, champ_field, label = "ability_power", None, "ability power"
        elif re.search(r"maximum\s+health|max\s+health|\bhealth\b", lower):
            item_field, champ_field, label = "health", "max_health", "maximum health"
        elif re.search(r"maximum\s+mana|max\s+mana|\bmana\b", lower):
            item_field, champ_field, label = "mana", "max_resource", "maximum mana"
        elif re.search(r"magic\s+resist|\bmr\b", lower):
            item_field, champ_field, label = "magic_resist", "magic_resist", "magic resist"
        elif re.search(r"\barmor\b", lower):
            item_field, champ_field, label = "armor", "armor", "armor"
        else:
            return None
        item_stat = item.get("stats", {}).get(item_field)
        if not isinstance(item_stat, Mapping):
            return self._unsupported(
                question,
                f"{item.get('name')} has no exact static {label} stat",
                record=record,
                extra_pages=(str(item.get("name", "item")),),
            )
        if item_stat.get("percent"):
            return self._unsupported(
                question,
                "percentage item stats require a separate multiplicative stat contract",
                record=record,
                extra_pages=(str(item.get("name", "item")),),
            )
        if item.get("has_passive"):
            item_client = {
                "kind": "client_item",
                "url": f"https://raw.communitydragon.org/{self.pack.get('client_patch')}/plugins/rcp-be-lol-game-data/global/default/v1/items.json",
                "label": "patch-pinned CommunityDragon item data",
            }
            return self._unsupported(
                question,
                "item passive/unique text may modify the requested total",
                record=record,
                extra_pages=(str(item.get("name", "item")),),
                extra_sources=(item_client,),
            )
        amount = item_stat.get("value")
        if not isinstance(amount, (int, float)) or isinstance(amount, bool):
            return self._unsupported(question, "item stat is not numeric")
        row = record.get("levels", {}).get(str(level))
        base = 0.0 if champ_field is None else row.get(champ_field) if isinstance(row, Mapping) else None
        if not isinstance(base, (int, float)):
            return self._unsupported(question, f"champion {label} is unavailable at level {level}")
        total = self._format(float(base) + float(amount))
        item_name = str(item.get("name", "item"))
        client_url = f"https://raw.communitydragon.org/{self.pack.get('client_patch')}/plugins/rcp-be-lol-game-data/global/default/v1/items.json"
        sources = self._source_links(record, extra_pages=(item_name,))
        sources.append({"kind": "client_item", "url": client_url, "label": "patch-pinned CommunityDragon item data"})
        unique: dict[str, dict[str, str]] = {link["url"]: link for link in sources}
        answer = self._envelope(
            question=question,
            intent="item_static_stat",
            value=total,
            unit=label,
            display=f"{total} {label}",
            calculation=f"{float(base):.2f} champion {label} + {float(amount):.2f} {item_name} stat = {float(total):.2f}.",
            assumptions=(
                f"{record['name']} level {level}",
                f"one {item_name} equipped",
                "visible static item stats only; no passive, unique, rune, or buff effects",
            ),
            record=record,
            extra_pages=(item_name,),
        )
        answer["sources"] = list(unique.values())
        return answer

    def _ability_damage(self, question: str) -> dict[str, Any] | None:
        if not re.search(r"\bdamage\b", question, re.I):
            return None
        magic_resist_match = _MAGIC_RESIST_RE.search(question)
        armor_match = _ARMOR_RE.search(question)
        if magic_resist_match is not None and armor_match is not None:
            return self._unsupported(question, "mixed armor and magic resistance require a typed damage-channel contract")
        modifier_match = _DAMAGE_MODIFIER_RE.search(question)
        explicit_no_penetration = bool(re.search(r"\bno\s+penetration\b", question, re.I))
        allowed_resistance_token = (
            magic_resist_match is not None and modifier_match is not None
            and modifier_match.group(0).casefold() in {"magic resist", "magic resistance", "mr"}
        ) or (
            armor_match is not None and modifier_match is not None
            and modifier_match.group(0).casefold() == "armor"
        )
        if modifier_match and not (
            allowed_resistance_token
            or (modifier_match.group(0).casefold() in {"penetration", "pen"} and explicit_no_penetration)
        ):
            return self._unsupported(
                question,
                "target modifiers or mitigation are outside the direct raw-damage contract",
            )
        rank_match = _RANK_RE.search(question)
        if rank_match is None:
            ability_level_match = _ABILITY_LEVEL_RE.search(question)
            if ability_level_match is None:
                return None
            rank = int(ability_level_match.group(1))
        else:
            rank = int(rank_match.group(1))
        key_match = _ABILITY_KEY_RE.search(question)
        ap_match = _AP_RE.search(question)
        ad_match = _AD_RE.search(question)
        level_matches = self._levels(question)
        level = level_matches[0] if len(level_matches) == 1 else None
        if key_match is None or (ap_match is None and ad_match is None):
            return None
        if not (1 <= rank <= 5):
            return self._unsupported(question, "ability rank must be in the exact range 1-5")
        record = self._resolve_champion(question)
        if record is None:
            return self._unsupported(question, "champion is absent from the exact patch packet")
        if self._mechanics_error is not None:
            return self._unsupported(question, self._mechanics_error)
        spell = self._mechanics_spell(record, key_match.group(1).upper())
        if spell is None:
            return self._unsupported(
                question,
                "the exact patch has no unambiguous normalized spell record for that ability",
            )
        try:
            calculation_name, raw_value = evaluate_direct_damage(
                spell,
                ability_key=key_match.group(1).upper(),
                ability_rank=rank,
                ability_power=float(ap_match.group(1)) if ap_match else 0.0,
                attack_damage=float(ad_match.group(1)) if ad_match else None,
                character_level=level,
            )
        except (TypeError, ValueError, UnsupportedFormulaError) as exc:
            return self._unsupported(question, f"direct damage graph is not executable: {exc}")
        raw_damage = raw_value
        post_mitigation = magic_resist_match is not None or armor_match is not None
        resistance_match = magic_resist_match or armor_match
        resistance = float(resistance_match.group(1)) if resistance_match else None
        if resistance is not None:
            if resistance < 0:
                return self._unsupported(question, "negative resistance is outside this mitigation slice")
            bot_data = spell.get("bot_data")
            damage_tag = bot_data.get("DamageTag") if isinstance(bot_data, Mapping) else None
            expected_tag = 1 if magic_resist_match is not None else 0
            if damage_tag != expected_tag:
                channel = "magic" if expected_tag == 1 else "physical"
                return self._unsupported(question, f"the exact client damage tag is not confirmed as {channel} damage")
            raw_value = raw_damage * (100.0 / (100.0 + resistance))
        value = self._format(raw_value)
        raw_spell = next(
            (
                candidate
                for candidate in self._spells.get(_norm(str(record.get("name", ""))), [])
                if str(candidate.get("spellKey", "")).casefold() == key_match.group(1).casefold()
            ),
            None,
        )
        spell_name = str(
            (raw_spell or {}).get("name")
            or spell.get("script_name")
            or f"{key_match.group(1).upper()} ability"
        )
        ap = float(ap_match.group(1)) if ap_match else 0.0
        ad = float(ad_match.group(1)) if ad_match else None
        channel = "magic" if magic_resist_match is not None else "physical"
        stat_text = f"{ap:g} AP" if ap_match else f"{ad:g} total AD"
        if ap_match and ad_match:
            stat_text = f"{ap:g} AP and {ad:g} total AD"
        if post_mitigation:
            calculation = (
                f"{float(value):.2f} = {raw_damage:.2f} raw damage "
                f"× 100/(100+{resistance:g}) {channel}-resistance multiplier."
            )
            unit = f"post-mitigation {channel} damage"
            display = f"{value} post-mitigation {channel} damage"
            assumptions = (
                f"{record['name']} rank-{rank} {key_match.group(1).upper()}",
                f"{stat_text}, {resistance:g} target {'magic resistance' if channel == 'magic' else 'armor'}, no penetration",
                f"one direct {channel}-damage calculation; no items, runes, passives, or shields",
            )
        else:
            calculation = (
                f"CommunityDragon {calculation_name} at rank-{rank} with {stat_text} "
                f"= {float(raw_value):.2f} pre-mitigation damage."
            )
            unit = "raw damage"
            display = f"{value} raw damage"
            assumptions = (
                f"{record['name']} rank-{rank} {key_match.group(1).upper()}",
                f"{stat_text} and no target modifiers",
                "one direct raw-damage calculation; no items, runes, passives, or mitigation",
            )
        return self._envelope(
            question=question,
            intent="direct_ability_damage",
            value=value,
            unit=unit,
            display=display,
            calculation=calculation,
            assumptions=assumptions,
            record=record,
            extra_pages=(
                f"{record['name']}/{spell_name}",
                *(("Magic resistance",) if post_mitigation else ()),
            ),
        )

    def _stat_delta(self, question: str) -> dict[str, Any] | None:
        match = _DELTA_RE.search(question)
        if match is None:
            return None
        low = int(match.group("from_level") or match.group("between_from"))
        high = int(match.group("to_level") or match.group("between_to"))
        if not (1 <= low < high <= 18):
            return self._unsupported(question, "level delta must increase within levels 1-18")
        record = self._resolve_champion(question)
        if record is None:
            return self._unsupported(question, "champion is absent from the exact patch packet")
        lower = question.lower()
        candidates: tuple[str, str, str] | None = None
        if re.search(r"maximum\s+health|max\s+health|\bhp\b", lower):
            candidates = ("max_health", "maximum health", "health")
        elif re.search(r"maximum\s+mana|max\s+mana|mana pool", lower):
            candidates = ("max_resource", "maximum mana", "mana")
        elif re.search(r"attack\s+damage|\bad\b", lower):
            candidates = ("attack_damage", "attack damage", "AD")
        elif re.search(r"magic\s+resist|\bmr\b", lower):
            candidates = ("magic_resist", "magic resist", "MR")
        else:
            return None
        field, label, short = candidates
        low_row = record.get("levels", {}).get(str(low))
        high_row = record.get("levels", {}).get(str(high))
        if not isinstance(low_row, Mapping) or not isinstance(high_row, Mapping):
            return self._unsupported(question, "requested levels are absent from the exact patch packet")
        left, right = low_row.get(field), high_row.get(field)
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            return self._unsupported(question, f"{label} is unavailable in the exact patch packet")
        value = self._format(float(right) - float(left))
        return self._envelope(
            question=question,
            intent="stat_delta",
            value=value,
            unit=label,
            display=f"{value} {label}",
            calculation=f"{float(right):.2f} {short} at level {high} − {float(left):.2f} {short} at level {low}.",
            assumptions=(f"{record['name']} exact {self.patch} level rows",),
            record=record,
        )

    def _champion_stat_comparison(self, question: str) -> dict[str, Any] | None:
        """Compare one stat for two explicitly named champions at one level.

        This handler exists before the single-champion fast path so a question
        such as "the delta between Gnar and Darius AD at level 14" cannot be
        reduced to the second champion's standalone stat.  ``between`` is
        interpreted as an absolute gap; explicit ``X minus Y`` wording keeps
        the signed order instead.
        """

        if not _COMPARISON_RE.search(question):
            return None
        levels = self._levels(question)
        if len(levels) != 1:
            return self._unsupported(question, "champion comparison requires exactly one level")
        level = levels[0]
        records: list[tuple[int, Mapping[str, Any]]] = []
        for record in self._champions:
            candidates = [record.get("name"), record.get("alias"), *(record.get("aliases") or [])]
            positions = [question.casefold().find(str(candidate).casefold()) for candidate in candidates if candidate and _mentions_name(question, candidate)]
            positions = [position for position in positions if position >= 0]
            if positions:
                records.append((min(positions), record))
        records.sort(key=lambda pair: (pair[0], -len(_norm(str(pair[1].get("name", ""))))))
        unique: list[Mapping[str, Any]] = []
        seen: set[str] = set()
        for _, record in records:
            key = _norm(str(record.get("name", "")))
            if key and key not in seen:
                seen.add(key)
                unique.append(record)
        if len(unique) != 2:
            return self._unsupported(question, "champion comparison requires exactly two patch-pinned champion identities")
        lower = question.casefold()
        if re.search(r"maximum\s+health|max\s+health|\bhp\b", lower):
            field, label, short = "max_health", "maximum health", "HP"
        elif re.search(r"maximum\s+mana|max\s+mana|mana pool", lower):
            field, label, short = "max_resource", "maximum mana", "mana"
        elif re.search(r"attack\s+damage|\bad\b", lower):
            field, label, short = "attack_damage", "attack damage", "AD"
        elif re.search(r"magic\s+resist|\bmr\b", lower):
            field, label, short = "magic_resist", "magic resist", "MR"
        else:
            return self._unsupported(question, "champion comparison stat must be health, mana, attack damage, or magic resist")
        values: list[float] = []
        for record in unique:
            row = record.get("levels", {}).get(str(level))
            value = row.get(field) if isinstance(row, Mapping) else None
            if not isinstance(value, (int, float)):
                return self._unsupported(question, f"{label} is unavailable for {record.get('name')} at level {level}")
            values.append(float(value))
        signed = bool(re.search(r"\b(?:minus|subtract(?:ed)?|less\s+than)\b", lower))
        value = values[0] - values[1] if signed else abs(values[0] - values[1])
        rendered = self._format(value)
        first_name, second_name = str(unique[0]["name"]), str(unique[1]["name"])
        calculation = f"{first_name} {values[0]:.2f} {short} and {second_name} {values[1]:.2f} {short} at level {level}; {'signed subtraction' if signed else 'absolute gap'} = {value:.2f}."
        answer = self._envelope(
            question=question,
            intent="champion_stat_comparison",
            value=rendered,
            unit=label,
            display=f"{rendered} {label}",
            calculation=calculation,
            assumptions=(f"{first_name} and {second_name} exact {self.patch} level-{level} rows", "between/difference wording uses an absolute gap unless subtraction is explicit"),
            record=unique[0],
        )
        answer["components"] = {first_name: self._format(values[0]), second_name: self._format(values[1])}
        answer["sources"] = self._merge_sources(
            self._source_links(unique[0]), self._source_links(unique[1])
        )
        return answer

    def _resource_window(self, question: str) -> dict[str, Any] | None:
        if not re.search(r"mana|regenerat|MP5|mp5", question, re.I):
            return None
        seconds_match = _SECONDS_RE.search(question)
        level = self._level(question)
        if seconds_match is None or level is None:
            return None
        record = self._resolve_champion(question)
        if record is None:
            return self._unsupported(question, "champion is absent from the exact patch packet")
        row = record.get("levels", {}).get(str(level))
        if not isinstance(row, Mapping):
            return self._unsupported(question, "requested level is absent from the exact patch packet")
        mp5 = row.get("mp5")
        if not isinstance(mp5, (int, float)):
            return self._unsupported(question, "mana regeneration is unavailable for this champion")
        seconds = float(seconds_match.group(1))
        value = self._format(float(mp5) * seconds / 5.0)
        return self._envelope(
            question=question,
            intent="resource_regeneration_window",
            value=value,
            unit="mana",
            display=f"{value} mana",
            calculation=f"{float(mp5):.2f} MP5 × {seconds:g}/5 seconds.",
            assumptions=(f"{record['name']} level {level}", "no spending or external regeneration effects"),
            record=record,
        )

    def _max_stat(self, question: str) -> dict[str, Any] | None:
        levels = self._levels(question)
        if len(levels) != 1:
            return None
        level = levels[0]
        record = self._resolve_champion(question)
        if record is None:
            return None
        lower = question.lower()
        if re.search(r"maximum\s+mana|max\s+mana|mana pool", lower):
            field, label = "max_resource", "maximum mana"
        elif re.search(r"maximum\s+health|max\s+health|\bhp\b", lower):
            field, label = "max_health", "maximum health"
        else:
            return None
        row = record.get("levels", {}).get(str(level))
        value = row.get(field) if isinstance(row, Mapping) else None
        if not isinstance(value, (int, float)):
            return self._unsupported(question, f"{label} is unavailable for this champion")
        rendered = self._format(float(value))
        return self._envelope(
            question=question,
            intent="champion_max_stat",
            value=rendered,
            unit="mana" if field == "max_resource" else "health",
            display=f"{rendered} {('mana' if field == 'max_resource' else 'health')}",
            calculation=f"Read {label} from the exact level-{level} stat row.",
            assumptions=(f"{record['name']} level {level}",),
            record=record,
        )

    def _turret_dps_optimization(self, question: str) -> dict[str, Any] | None:
        """Resolve the natural-language Jinx-vs-inner-turret build contract."""

        if not looks_like_jinx_turret_build_query(question):
            return None
        try:
            answer = optimize_jinx_turret(self, question)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            return self._unsupported(
                question,
                f"turret-DPS optimizer could not close its patch-pinned item search: {exc}",
                extra_pages=("Turret", "Jinx"),
            )
        answer.setdefault("engine", "lol-oracle-v1")
        return answer

    def _vayne_rammus_optimization(self, question: str) -> dict[str, Any] | None:
        """Resolve the natural-language Vayne-versus-Rammus build contract."""

        if not looks_like_vayne_rammus_build_query(question):
            return None
        try:
            answer = optimize_vayne_rammus(self, question)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            return self._unsupported(
                question,
                f"Vayne/Rammus optimizer could not close its patch-pinned item search: {exc}",
                extra_pages=("Vayne", "Rammus", "Defensive Ball Curl", "Thornmail", "Sunfire Aegis", "Randuin's Omen"),
            )
        answer.setdefault("engine", "lol-oracle-v1")
        return answer

    def answer(self, question: str) -> dict[str, Any]:
        if not isinstance(question, str) or not question.strip():
            return self._unsupported(question="", reason="question must be non-empty")
        for handler in (
            self._vayne_rammus_optimization,
            self._turret_dps_optimization,
            self._ability_budget,
            self._rune_interaction,
            self._structure_stat,
            self._stack_interaction,
            self._ordered_damage_sequence,
            self._item_stat,
            self._ability_damage,
            self._champion_stat_comparison,
            self._stat_delta,
            self._resource_window,
            self._max_stat,
        ):
            result = handler(question)
            if result is not None:
                return result
        return self._decorate(self.fast.answer(question), question)

    def semantic_answer(
        self, question: str, context: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Answer through the slot-filling semantic layer.

        The ordinary :meth:`answer` method remains the low-latency exact
        query path and keeps its historical ``unsupported`` contract.  This
        explicit opt-in method is for natural-language questions whose
        meaning is clear but whose patch, build, target, timeline, or mode
        state is not yet closed.
        """

        from .semantic_engine import SemanticOracleEngine

        return SemanticOracleEngine(self).answer(question, context)


__all__ = ["LeagueOracleEngine"]
