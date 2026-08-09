"""Sub-second, deterministic answers for a small League mechanics fast path.

The quick engine deliberately has a narrow contract.  It consumes a compiled
``fastpack`` (rather than a wiki or a language model) and answers only facts
which are represented by the pack: pre-computed champion levels, monster
levels, and objective rewards.  Unknown mechanics stay ``unsupported``.  This
module is intentionally tolerant of the pack's wire shape (records may be
lists or mappings and level keys may be strings or integers), while keeping
the calculation itself boring and deterministic.

Supported intents are:

* champion MP5/mana regeneration, magic resistance, and base attack damage;
* full Voidgrub camp gold; and
* itemless basic attacks required to kill a monster.

The parser uses exact aliases first and a small edit-distance fallback for
obvious typos.  Embeddings and external retrieval are intentionally absent
from this fast path.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


ENGINE_VERSION = "quick-mechanics-v1"
_LEVEL_RE = re.compile(r"\b(?:level|lvl|lv)\s*(?:=|:)?\s*(\d+)\b", re.I)
_WORD_RE = re.compile(r"[a-z0-9]+", re.I)


def _norm(value: Any) -> str:
    """Normalize names for exact alias matching.

    Spaces, apostrophes, dashes, and accents are not meaningful to entity
    resolution in this bounded catalogue.  Unicode letters are retained by
    ``str.isalnum``; this also keeps the helper independent of locale.
    """

    return "".join(char.lower() for char in str(value) if char.isalnum())


def _tokens(value: str) -> list[str]:
    return _WORD_RE.findall(value.lower())


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except (TypeError, ValueError):
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def _edit_distance(left: str, right: str, *, limit: int | None = None) -> int:
    """Levenshtein distance with an optional early cutoff."""

    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    if limit is not None and abs(len(left) - len(right)) > limit:
        return limit + 1
    # Keep the shorter string on the columns to reduce allocations.
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        row_min = i
        for j, right_char in enumerate(right, 1):
            value = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (left_char != right_char),
            )
            current.append(value)
            row_min = min(row_min, value)
        if limit is not None and row_min > limit:
            return limit + 1
        previous = current
    return previous[-1]


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError):
        # A compiled fastpack should be JSON, but provenance must remain useful
        # if a caller supplies a Mapping subclass with non-JSON metadata.
        return repr(value).encode("utf-8")


@dataclass(frozen=True)
class _Record:
    key: str
    data: Mapping[str, Any]
    aliases: tuple[str, ...]

    @property
    def label(self) -> str:
        for field in ("name", "canonical", "display_name", "id"):
            value = self.data.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return self.key


class QuickMechanicsEngine:
    """Answer the supported mechanics subset from an in-memory fastpack.

    ``pack`` is intentionally accepted as a mapping rather than a concrete
    compiler class.  This makes the query path cheap to warm and lets callers
    pin the packet to an exact patch and provenance receipt.
    """

    def __init__(self, pack: Mapping[str, Any]):
        if not isinstance(pack, Mapping):
            raise TypeError("fastpack must be a mapping")
        self.pack = pack
        self.patch = self._patch_value(pack)
        self._champions = self._index_records(
            self._section(pack, "champions"), kind="champion"
        )
        self._monsters = self._index_records(
            self._section(pack, "monsters"), kind="monster"
        )
        self._objectives = self._index_records(
            self._section(pack, "objectives"), kind="objective"
        )
        self._provenance = self._make_provenance(pack)

    # ------------------------------------------------------------------
    # Fastpack adaptation and provenance
    # ------------------------------------------------------------------
    @staticmethod
    def _patch_value(pack: Mapping[str, Any]) -> str | None:
        for key in ("patch", "patch_id", "version"):
            value = pack.get(key)
            if isinstance(value, (str, int, float)) and str(value).strip():
                return str(value)
        metadata = pack.get("metadata")
        if isinstance(metadata, Mapping):
            for key in ("patch", "patch_id", "version"):
                value = metadata.get(key)
                if isinstance(value, (str, int, float)) and str(value).strip():
                    return str(value)
        return None

    @staticmethod
    def _section(pack: Mapping[str, Any], name: str) -> Any:
        value = pack.get(name)
        if value is not None:
            return value
        data = pack.get("data")
        if isinstance(data, Mapping):
            return data.get(name, {})
        return {}

    @staticmethod
    def _make_provenance(pack: Mapping[str, Any]) -> dict[str, Any]:
        raw = pack.get("provenance")
        provenance = raw if isinstance(raw, Mapping) else {}
        supplied_hash = None
        for key in ("fastpack_sha256", "pack_sha256", "sha256", "hash"):
            value = provenance.get(key)
            if isinstance(value, str) and value:
                supplied_hash = value
                break
        if supplied_hash is None:
            supplied_hash = hashlib.sha256(_canonical_bytes(pack)).hexdigest()
        source = (
            provenance.get("source")
            or provenance.get("authority")
            or pack.get("source")
        )
        output: dict[str, Any] = {
            "engine": ENGINE_VERSION,
            "pack_sha256": supplied_hash,
        }
        if isinstance(source, str) and source:
            output["source"] = source
        source_hash = pack.get("source_hash") or pack.get("source_sha256")
        if isinstance(source_hash, str) and source_hash:
            output["source_sha256"] = source_hash
        source_hashes = pack.get("source_hashes")
        if isinstance(source_hashes, Mapping):
            index_hash = source_hashes.get("index_sha256")
            if isinstance(index_hash, str) and index_hash:
                output["index_sha256"] = index_hash
        return output

    @staticmethod
    def _index_records(section: Any, *, kind: str) -> tuple[_Record, ...]:
        """Turn list/dict sections into immutable records.

        Mapping sections are commonly keyed by normalized alias.  The key is
        retained as an alias when a record does not repeat it in ``aliases``.
        A nested ``records``/``items`` mapping is accepted for compiler output
        wrappers without broadening the calculation contract.
        """

        records: list[tuple[str, Mapping[str, Any]]] = []
        if isinstance(section, Mapping):
            nested = section.get("records") or section.get("items")
            if isinstance(nested, (Mapping, list, tuple)):
                section = nested
            else:
                for key, value in section.items():
                    if isinstance(value, Mapping):
                        records.append((str(key), value))
        if isinstance(section, (list, tuple)):
            for index, value in enumerate(section):
                if isinstance(value, Mapping):
                    records.append((str(index), value))
        out: list[_Record] = []
        for key, data in records:
            # Numeric map keys are internal champion IDs (e.g. ``54`` and
            # ``60054``), not user-facing aliases.  Keeping them would make a
            # level number such as ``13`` resolve to champion ID 13 before the
            # intended fuzzy name match is attempted.
            candidates: list[str] = [] if key.isdigit() else [key]
            for field in (
                "id",
                "name",
                "canonical",
                "canonical_name",
                "display_name",
                "normalized_name",
            ):
                value = data.get(field)
                if isinstance(value, str):
                    candidates.append(value)
            for field in ("aliases", "normalized_aliases", "alias", "names"):
                values = data.get(field)
                if isinstance(values, str):
                    candidates.append(values)
                elif isinstance(values, Iterable) and not isinstance(values, Mapping):
                    candidates.extend(str(value) for value in values if value is not None)
            normalized = tuple(dict.fromkeys(_norm(value) for value in candidates if _norm(value)))
            if normalized:
                out.append(_Record(key=key, data=data, aliases=normalized))
        return tuple(out)

    # ------------------------------------------------------------------
    # Answer envelope and entity resolution
    # ------------------------------------------------------------------
    def _envelope(
        self,
        *,
        status: str,
        display: str,
        value: Any,
        unit: str | None,
        intent: str,
        assumptions: Sequence[str] = (),
        reason: str | None = None,
    ) -> dict[str, Any]:
        allowed = {"available", "not_applicable", "unsupported"}
        if status not in allowed:
            raise ValueError(f"invalid quick mechanics status: {status}")
        output: dict[str, Any] = {
            "status": status,
            "display": display,
            "value": value,
            "unit": unit,
            "patch": self.patch,
            "intent": intent,
            "assumptions": list(assumptions),
            "provenance": dict(self._provenance),
        }
        if reason:
            output["reason"] = reason
        return output

    def _unsupported(
        self,
        *,
        question: str,
        intent: str = "unsupported",
        reason: str | None = None,
    ) -> dict[str, Any]:
        detail = reason or "question is outside the compiled quick-mechanics subset"
        return self._envelope(
            status="unsupported",
            display="Unsupported by the quick mechanics fast path",
            value=None,
            unit=None,
            intent=intent,
            assumptions=(),
            reason=detail,
        )

    @staticmethod
    def _resolve(query: str, records: Sequence[_Record]) -> tuple[_Record | None, str | None, bool]:
        """Resolve an entity as ``(record, matched_alias, ambiguous)``."""

        normalized_query = _norm(query)
        exact: list[tuple[int, _Record, str]] = []
        for record in records:
            for alias in record.aliases:
                if alias and alias in normalized_query:
                    exact.append((len(alias), record, alias))
        if exact:
            max_len = max(item[0] for item in exact)
            candidates = [(record, alias) for length, record, alias in exact if length == max_len]
            unique = {(record.key, alias): (record, alias) for record, alias in candidates}
            records_found = {record.key for record, _ in unique.values()}
            if len(records_found) == 1:
                record, alias = next(iter(unique.values()))
                return record, alias, False
            # CommunityDragon-derived packs can contain two records for the
            # same displayed champion (for example numeric keys 54 and 60054).
            # If their canonical labels agree, prefer the record with the
            # richest level table instead of surfacing a false ambiguity.
            labels = {_norm(record.label) for record, _ in unique.values()}
            if len(labels) == 1:
                candidates = list(unique.values())
                candidates.sort(
                    key=lambda item: (
                        # Cosmetic/skin bins (currently ``Jade_*``) repeat a
                        # champion display alias but are not the base champion
                        # users mean in a stat question.  The compiler's alias
                        # map applies the same base-first rule.
                        not str(item[0].data.get("alias", "")).lower().startswith("jade_"),
                        len(QuickMechanicsEngine._levels(item[0])),
                        sum(
                            1
                            for level in QuickMechanicsEngine._levels(item[0]).values()
                            if isinstance(level, Mapping)
                        ),
                        item[0].key,
                    ),
                    reverse=True,
                )
                record, alias = candidates[0]
                return record, alias, False
            return None, None, True

        # Fuzzy matching is deliberately conservative: one edit for short
        # names and at most two for longer names.  Compare token windows so a
        # typo in ``malphjite`` can still be found without vector retrieval.
        words = _tokens(query)
        windows: list[str] = []
        for start in range(len(words)):
            for width in range(1, min(4, len(words) - start) + 1):
                windows.append(_norm("".join(words[start : start + width])))
        fuzzy: list[tuple[int, int, _Record, str]] = []
        for record in records:
            for alias in record.aliases:
                threshold = 1 if len(alias) <= 10 else 2
                for window in windows:
                    distance = _edit_distance(alias, window, limit=threshold)
                    if distance <= threshold:
                        fuzzy.append((distance, -len(alias), record, alias))
        if not fuzzy:
            return None, None, False
        best_distance = min(item[0] for item in fuzzy)
        best = [item for item in fuzzy if item[0] == best_distance]
        longest = max(item[1] for item in best)
        best = [item for item in best if item[1] == longest]
        record_keys = {item[2].key for item in best}
        if len(record_keys) != 1:
            labels = {_norm(item[2].label) for item in best}
            if len(labels) == 1:
                # Match the exact-alias base-first policy for duplicate
                # cosmetic bins (e.g. Malphite/Jade_Malphite).
                best.sort(
                    key=lambda item: not str(item[2].data.get("alias", ""))
                    .lower()
                    .startswith("jade_"),
                    reverse=True,
                )
                _, _, record, alias = best[0]
                return record, alias, False
            return None, None, True
        _, _, record, alias = best[0]
        return record, alias, False

    @staticmethod
    def _levels(record: _Record) -> Mapping[Any, Any]:
        for key in ("levels", "level_tables", "stats_by_level", "level_stats"):
            value = record.data.get(key)
            if isinstance(value, Mapping):
                return value
        return {}

    @classmethod
    def _level_data(cls, record: _Record, level: int) -> Mapping[str, Any] | None:
        levels = cls._levels(record)
        value = levels.get(level)
        if value is None:
            value = levels.get(str(level))
        if not isinstance(value, Mapping):
            return None
        return value

    @staticmethod
    def _stat(data: Mapping[str, Any], names: Sequence[str]) -> float | None:
        wanted = {_norm(name) for name in names}
        # Fast path: precomputed level rows are flat in the compiled pack.
        for key, value in data.items():
            if _norm(key) in wanted:
                number = _finite(value)
                if number is not None:
                    return number
        # Tolerate a single ``stats``/``base`` wrapper without recursively
        # interpreting arbitrary formulas or source data.
        for key in ("stats", "base", "values", "derived"):
            nested = data.get(key)
            if isinstance(nested, Mapping):
                for child, value in nested.items():
                    if _norm(child) in wanted:
                        number = _finite(value)
                        if number is not None:
                            return number
        return None

    @staticmethod
    def _level(question: str) -> int | None:
        match = _LEVEL_RE.search(question)
        if match is None:
            return None
        try:
            level = int(match.group(1))
        except ValueError:
            return None
        return level if 1 <= level <= 18 else None

    @staticmethod
    def _all_level_matches(question: str) -> list[re.Match[str]]:
        return list(_LEVEL_RE.finditer(question))

    @staticmethod
    def _resource_type(record: _Record) -> str | None:
        for key in ("resource_type", "resource", "primary_resource"):
            value = record.data.get(key)
            if value is not None:
                normalized = _norm(value)
                if normalized:
                    return normalized
        return None

    @staticmethod
    def _is_no_resource(record: _Record) -> bool:
        resource = QuickMechanicsEngine._resource_type(record)
        return resource in {
            "none",
            "noresource",
            "health",
            "healthcost",
            "null",
            "0",
        }

    @staticmethod
    def _format_number(value: float) -> str:
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.6g}"

    # ------------------------------------------------------------------
    # Public parser / calculators
    # ------------------------------------------------------------------
    def answer(self, question: str) -> dict[str, Any]:
        if not isinstance(question, str) or not question.strip():
            return self._unsupported(question="", reason="question must be non-empty")
        lower = question.lower()
        # Objective rewards are routed before champion stats.  A full-grub
        # question has no level and should never be interpreted as a monster
        # attack simulation.
        if self._looks_like_objective(lower):
            result = self._answer_objective(question)
            if result is not None:
                return result
        if self._looks_like_attack(lower):
            result = self._answer_attacks(question)
            if result is not None:
                return result
        result = self._answer_champion_stat(question)
        if result is not None:
            return result
        return self._unsupported(question=question)

    def _looks_like_objective(self, question: str) -> bool:
        return bool(
            re.search(r"\b(?:gold|g)\b", question)
            and re.search(r"\b(?:void\s*grubs?|grubs?|grub\s*camp)\b", question)
        ) or bool(re.search(r"\b(?:full\s+)?void\s*grubs?\b", question))

    def _answer_objective(self, question: str) -> dict[str, Any] | None:
        record, _, ambiguous = self._resolve(question, self._objectives)
        if ambiguous:
            return self._unsupported(intent="objective_gold", reason="objective name is ambiguous", question=question)
        if record is None:
            return self._unsupported(intent="objective_gold", reason="objective is absent from fastpack", question=question)
        gold = self._stat(
            record.data,
            (
                "gold",
                "total_gold",
                "gold_reward",
                "reward_gold",
                "local_gold",
                "killer_gold",
            ),
        )
        # The compiler's objective supplement stores the cash split as
        # ``gold: {local, global, total_local, ...}``; only local cash belongs
        # in this answer.  Treat an explicitly nested reward as data, not as a
        # guessed sum of unrelated fields.
        if gold is None:
            nested_gold = record.data.get("gold")
            if isinstance(nested_gold, Mapping):
                gold = self._stat(
                    nested_gold,
                    ("total_local", "local", "gold", "total_gold", "cash_local", "killer_local"),
                )
        # Some packs retain the per-grub constant and count alongside the camp
        # record.  Only multiply when the fields explicitly say per-unit;
        # never infer an aggregate from an unrelated reward number.
        if gold is None:
            reward = record.data.get("reward") or record.data.get("rewards") or record.data.get("reward_data")
            if isinstance(reward, Mapping):
                gold = self._stat(reward, ("gold", "total_gold", "local_gold", "killer_gold"))
                if gold is None:
                    per = self._stat(reward, ("gold_per_grub", "gold_per_unit", "per_grub", "per_unit"))
                    count = self._stat(record.data, ("count", "grub_count", "units"))
                    if per is not None and count is not None:
                        gold = per * count
        if gold is None:
            return self._unsupported(intent="objective_gold", reason="gold reward is absent from fastpack", question=question)
        assumptions = ("full Voidgrub camp (three grubs)",)
        return self._envelope(
            status="available",
            display=f"{self._format_number(gold)}g",
            value=int(gold) if gold.is_integer() else gold,
            unit="gold",
            intent="objective_gold",
            assumptions=assumptions,
        )

    def _looks_like_attack(self, question: str) -> bool:
        if not self._monsters:
            return False
        return bool(
            re.search(r"\b(?:kill|kills|killing|take\s+down|how\s+many\s+autos?|auto\s+attacks?|basic\s+attacks?)\b", question)
            and re.search(r"\b(?:to\s+kill|kill|autos?|auto\s+attacks?|basic\s+attacks?)\b", question)
        )

    @staticmethod
    def _entity_positions(question: str, record: _Record) -> list[tuple[int, int]]:
        positions: list[tuple[int, int]] = []
        lowered = question.lower()
        # We only use positions for assigning levels.  Exact aliases are
        # enough here; fuzzy positions are unnecessary for the supported typo
        # stat path.
        for alias in sorted(record.aliases, key=len, reverse=True):
            pieces: list[str] = []
            # The alias is normalized; reconstruct a permissive one-token
            # search for names such as ``voidgrub`` and a spaced fallback.
            pieces.append(alias)
            if len(alias) > 4:
                pieces.append(alias.replace("sol", " sol"))
            for piece in pieces:
                for match in re.finditer(re.escape(piece), lowered):
                    positions.append((match.start(), match.end()))
        return sorted(set(positions))

    @classmethod
    def _assign_attack_levels(
        cls,
        question: str,
        champion: _Record,
        monster: _Record,
    ) -> tuple[int | None, int | None, bool]:
        matches = cls._all_level_matches(question)
        if not matches:
            return None, None, False
        champion_positions = cls._entity_positions(question, champion)
        monster_positions = cls._entity_positions(question, monster)

        def nearest(position: int, spans: Sequence[tuple[int, int]]) -> int:
            if not spans:
                return 10**9
            return min(min(abs(position - start), abs(position - end)) for start, end in spans)

        champion_level: int | None = None
        monster_level: int | None = None
        unassigned: list[int] = []
        for match in matches:
            level = int(match.group(1))
            if not 1 <= level <= 18:
                unassigned.append(level)
                continue
            c_distance = nearest(match.start(), champion_positions)
            m_distance = nearest(match.start(), monster_positions)
            # A level written directly beside an entity is unambiguous.  If
            # wording is broad, preserve normal attacker-then-target order.
            if c_distance <= 16 and c_distance < m_distance and champion_level is None:
                champion_level = level
            elif m_distance <= 16 and m_distance < c_distance and monster_level is None:
                monster_level = level
            else:
                unassigned.append(level)
        for level in unassigned:
            if champion_level is None:
                champion_level = level
            elif monster_level is None:
                monster_level = level
        # ``Tristana lvl5 vs lvl5 Gromp`` has the second level near Gromp, but
        # if entity positions were unavailable we still preserve order.
        if champion_level is None and monster_level is not None and len(matches) == 1:
            return None, monster_level, False
        return champion_level, monster_level, bool(matches)

    @staticmethod
    def _mentions_loadout(question: str) -> bool:
        # Explicitly empty loadouts are supported; any concrete item/rune/
        # ability or buff would require a richer effect engine.
        for term in ("item", "rune", "ability", "buff"):
            if not re.search(rf"\b{term}s?\b", question, re.I):
                continue
            if re.search(rf"\b(?:no|without|zero|0|itemless|rune-less|runeless|ability-less|abilityless)\b[^.?!;]*\b{term}s?\b", question, re.I):
                continue
            return True
        return False

    def _answer_attacks(self, question: str) -> dict[str, Any] | None:
        champion, _, champion_ambiguous = self._resolve(question, self._champions)
        monster, _, monster_ambiguous = self._resolve(question, self._monsters)
        if champion_ambiguous or monster_ambiguous:
            return self._unsupported(intent="attacks_to_kill", reason="attacker or target name is ambiguous", question=question)
        if champion is None or monster is None:
            return self._unsupported(intent="attacks_to_kill", reason="attacker champion and monster target are required", question=question)
        if self._mentions_loadout(question):
            return self._unsupported(intent="attacks_to_kill", reason="items, runes, abilities, and buffs are outside this itemless path", question=question)
        attacker_level, target_level, had_level = self._assign_attack_levels(question, champion, monster)
        if attacker_level is None:
            return self._unsupported(intent="attacks_to_kill", reason="attacker level is required", question=question)
        assumptions: list[str] = [
            "itemless, rune-less, ability-less basic attacks",
            "physical damage with no penetration or temporary buffs",
        ]
        if target_level is None:
            target_level = attacker_level
            assumptions.append(f"{monster.label} level defaulted to attacker level {attacker_level}")
        else:
            assumptions.append(f"{monster.label} level {target_level}")
        attacker_data = self._level_data(champion, attacker_level)
        target_data = self._level_data(monster, target_level)
        if attacker_data is None or target_data is None:
            return self._unsupported(intent="attacks_to_kill", reason="requested level is absent from fastpack", question=question)
        attack_damage = self._stat(attacker_data, ("ad", "attack_damage", "base_ad", "base_attack_damage"))
        health = self._stat(target_data, ("hp", "health", "max_hp", "max_health"))
        armor = self._stat(target_data, ("armor", "base_armor"))
        if attack_damage is None or health is None or armor is None:
            return self._unsupported(intent="attacks_to_kill", reason="attacker AD or target HP/armor is absent from fastpack", question=question)
        if attack_damage <= 0 or health <= 0:
            return self._unsupported(intent="attacks_to_kill", reason="non-positive AD or target health cannot be simulated", question=question)
        if armor >= 0:
            multiplier = 100.0 / (100.0 + armor)
        else:
            multiplier = 2.0 - 100.0 / (100.0 - armor)
        per_attack = attack_damage * multiplier
        if per_attack <= 0:
            return self._unsupported(intent="attacks_to_kill", reason="post-mitigation damage is not positive", question=question)
        attacks = math.ceil(health / per_attack)
        assumptions.append(f"{monster.label} armor {self._format_number(armor)}")
        return self._envelope(
            status="available",
            display=f"{attacks} auto attacks",
            value=int(attacks),
            unit="auto attacks",
            intent="attacks_to_kill",
            assumptions=assumptions,
        )

    def _answer_champion_stat(self, question: str) -> dict[str, Any] | None:
        lower = question.lower()
        stat: tuple[str, str, str, tuple[str, ...]] | None = None
        if re.search(r"(?:\bmp5\b|mp\s*/?\s*5|mana\s*(?:regen|regeneration|recovery)|mana\s+per\s+5)", lower):
            stat = (
                "champion_mp5",
                "mana per 5 seconds",
                "MP5",
                ("mp5", "mana_regen_mp5", "mana_regeneration_mp5", "mana_per_5", "resource_regen_per_5"),
            )
        elif re.search(r"(?:magic\s*resist(?:ance)?|\bmr\b)", lower):
            stat = (
                "champion_mr",
                "magic resist",
                "MR",
                ("mr", "magic_resist", "magic_resistance", "base_mr", "base_magic_resist"),
            )
        elif re.search(r"(?:attack\s*damage|\bad\b)", lower):
            stat = (
                "champion_ad",
                "attack damage",
                "AD",
                ("ad", "attack_damage", "base_ad", "base_attack_damage"),
            )
        else:
            return None
        champion, _, ambiguous = self._resolve(question, self._champions)
        if ambiguous:
            return self._unsupported(intent=stat[0], reason="champion name is ambiguous", question=question)
        if champion is None:
            return self._unsupported(intent=stat[0], reason="champion is absent from fastpack", question=question)
        level_matches = self._all_level_matches(question)
        if len(level_matches) != 1:
            return self._unsupported(
                intent=stat[0],
                reason="exactly one champion level is required for a single-stat lookup",
                question=question,
            )
        level = self._level(question)
        if level is None:
            return self._unsupported(intent=stat[0], reason="champion level 1-18 is required", question=question)
        if stat[0] == "champion_mp5":
            resource = self._resource_type(champion)
            if self._is_no_resource(champion) or (resource is not None and resource not in {"mana", "manapool"}):
                return self._envelope(
                    status="not_applicable",
                    display=f"MP5 is not applicable to {champion.label}",
                    value=None,
                    unit="mana per 5 seconds",
                    intent=stat[0],
                    assumptions=(f"{champion.label} has no mana resource",),
                )
        level_data = self._level_data(champion, level)
        if level_data is None:
            return self._unsupported(intent=stat[0], reason="requested level is absent from fastpack", question=question)
        value = self._stat(level_data, stat[3])
        if value is None:
            return self._unsupported(intent=stat[0], reason="requested stat is absent from fastpack", question=question)
        # Keep the public response compact and stable.  The source fastpack
        # retains full precision; human-facing stat answers are rounded to
        # two decimals at the envelope boundary.
        rounded = round(value, 2)
        number = int(rounded) if rounded.is_integer() else rounded
        return self._envelope(
            status="available",
            display=f"{rounded:.2f} {stat[1]}",
            value=number,
            unit=stat[1],
            intent=stat[0],
            assumptions=(f"{champion.label} level {level}",),
        )


__all__ = ["ENGINE_VERSION", "QuickMechanicsEngine"]
