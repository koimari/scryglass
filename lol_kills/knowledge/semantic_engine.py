"""Semantic query planning for the League calculation oracle.

The resident :mod:`lol_oracle` path is intentionally narrow and exact.  A
natural-language question can still be *semantically* understood even when
the current text does not contain enough state to calculate a number.  This
module is the state/intent layer between that text and the deterministic
oracle:

* it classifies underspecified questions;
* turns missing facts into typed, machine-readable slots;
* rejects contradictory patch/mode/entity requests;
* executes a closed direct-damage request through ``LeagueOracleEngine``;
* executes closed numeric event/counterfactual requests through
  ``MechanicsEngine``.

It never fills a missing level, build, target, timeline, or outcome rule with
an implicit default.  ``needs_input`` means the question is understood and
can become answerable once its contract is filled.  ``invalid_scenario``
means the supplied premise is contradictory or refers to an unavailable
patch/mode.  ``unsupported`` is reserved for a closed request whose rule is
not executable in the currently validated kernel.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

from .lol_oracle import LeagueOracleEngine
from .mechanics_engine import Combatant, Damage, Event, GameState, MechanicsEngine
from .quick_mechanics_fastpack import compile_fastpack
from ..v2.patch_identity import CURRENT_PUBLIC_PATCH


SCHEMA_VERSION = "scryglass:semantic-oracle:v1"
ENGINE_VERSION = "semantic-oracle-v1.0.0"

_PATCH_RE = re.compile(r"\b(?:patch|version|v)\s*=?\s*(\d{1,3}\.\d{1,3})\b", re.I)
_MODE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("summoners_rift", r"summoner['’]?s\s+rift|\bsr\b"),
    ("howling_abyss", r"howling\s+abyss|\baram\b"),
    ("arena", r"\barena\b"),
    ("urf", r"\burf\b"),
    ("nexus_blitz", r"nexus\s+blitz"),
)
_KNOWN_MODES = frozenset(item[0] for item in _MODE_PATTERNS)
_ABILITY_KEY_RE = re.compile(r"\b([QWER])\b", re.I)
_RANK_RE = re.compile(r"\brank\s*[-:=]?\s*(\d+)\b", re.I)
_ABILITY_RANK_RE = re.compile(r"\b([QWER])\s*(?:rank|level|lvl|lv)\s*[-:=]?\s*(\d+)\b", re.I)
_LEVEL_RE = re.compile(r"\b(?:level|lvl|lv)\s*(?:=|:|-)?\s*(\d+)\b", re.I)
_AP_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*AP\b", re.I)
_AD_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:total\s+)?(?:attack\s+damage|AD)\b", re.I)


def _canonical(value: Any) -> Any:
    """Return a JSON-safe, recursively sorted representation."""

    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sha(value: Any) -> str:
    payload = json.dumps(
        _canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _norm(value: Any) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value)).casefold()
    return "".join(char for char in decomposed if char.isalnum())


def _mentions(question: str, candidate: Any) -> bool:
    parts = re.findall(r"[a-z0-9]+", str(candidate).casefold())
    if not parts:
        return False
    pattern = r"(?<![a-z0-9])" + r"\W+".join(re.escape(part) for part in parts) + r"(?![a-z0-9])"
    return re.search(pattern, question.casefold()) is not None


def _wiki_url(title: str) -> str:
    return "https://wiki.leagueoflegends.com/en-us/" + quote(
        title.replace(" ", "_"), safe="_-\'()"
    )


def _client_url(client_patch: str) -> str:
    return (
        f"https://raw.communitydragon.org/{client_patch}/"
        "plugins/rcp-be-lol-game-data/global/default/v1/champions.json"
    )


def _path_value(mapping: Mapping[str, Any], path: str) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _has_value(mapping: Mapping[str, Any], path: str) -> bool:
    value = _path_value(mapping, path)
    return value is not None


def _merge_context(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Merge user context without losing nested attacker/target fields."""

    merged: dict[str, Any] = dict(left)
    for key, value in right.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_context(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = value
    return merged


def _provided_paths(value: Any, prefix: str = "") -> list[str]:
    if not isinstance(value, Mapping):
        return [prefix] if prefix else []
    paths: list[str] = []
    for key in sorted(value, key=str):
        path = f"{prefix}.{key}" if prefix else str(key)
        item = value[key]
        if isinstance(item, Mapping) and item:
            paths.extend(_provided_paths(item, path))
        else:
            paths.append(path)
    return paths


@dataclass(frozen=True)
class SemanticSlot:
    path: str
    value_type: str
    reason: str
    examples: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "type": self.value_type,
            "reason": self.reason,
            "examples": list(self.examples),
        }


@dataclass(frozen=True)
class SemanticIssue:
    code: str
    message: str
    path: str | None = None

    def to_mapping(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "path": self.path}


@dataclass(frozen=True)
class SemanticRequest:
    intent: str
    question: str
    fields: Mapping[str, Any]
    entities: tuple[str, ...] = ()

    @property
    def request_sha256(self) -> str:
        return _sha(self.to_mapping())

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "intent": self.intent,
            "question": self.question,
            "fields": _canonical(self.fields),
            "entities": list(self.entities),
        }


class SemanticOracleEngine:
    """Slot-filling semantic layer backed by one or more exact patch packets."""

    def __init__(
        self,
        oracle: LeagueOracleEngine,
        *,
        available_patch_indices: Mapping[str, Path] | None = None,
        supported_modes: Iterable[str] = _KNOWN_MODES,
    ) -> None:
        self.oracle = oracle
        self.supported_modes = frozenset(str(item) for item in supported_modes)
        self._oracles: dict[str, LeagueOracleEngine] = {str(oracle.patch): oracle}
        self._patch_indices = {
            str(patch): Path(path) for patch, path in (available_patch_indices or {}).items()
        }
        self._patch_indices.update(self._discover_patch_indices(oracle))

    @staticmethod
    def _discover_patch_indices(oracle: LeagueOracleEngine) -> dict[str, Path]:
        root = oracle.raw_champion_root
        if root is None:
            return {}
        patch_dir = Path(root).parent.parent
        year_dir = patch_dir.parent
        if not patch_dir.is_dir() or not year_dir.is_dir():
            return {}
        found: dict[str, Path] = {}
        for index_path in sorted(year_dir.glob("*/mechanics-index.json")):
            try:
                payload = json.loads(index_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            patch = payload.get("patch") if isinstance(payload, Mapping) else None
            if isinstance(patch, str) and patch:
                found.setdefault(patch, index_path)
        return found

    @classmethod
    def from_index(cls, index_path: Path) -> "SemanticOracleEngine":
        index_path = Path(index_path)
        pack = compile_fastpack(index_path)
        oracle = LeagueOracleEngine(
            pack, raw_champion_root=index_path.parent / "raw" / "champions"
        )
        return cls(oracle)

    def _oracle_for_patch(self, patch: str) -> LeagueOracleEngine | None:
        cached = self._oracles.get(patch)
        if cached is not None:
            return cached
        index_path = self._patch_indices.get(patch)
        if index_path is None:
            return None
        try:
            pack = compile_fastpack(index_path)
            loaded = LeagueOracleEngine(
                pack, raw_champion_root=index_path.parent / "raw" / "champions"
            )
        except (OSError, ValueError, TypeError) as exc:
            # Keep the patch known but non-executable.  The caller receives a
            # structured ``needs_input``/``unsupported`` answer, never a
            # fallback to the current patch.
            _ = exc
            return None
        self._oracles[patch] = loaded
        return loaded

    @staticmethod
    def _extract_patch(question: str, fields: Mapping[str, Any]) -> str | None:
        explicit = fields.get("patch")
        if explicit is not None:
            return str(explicit).removeprefix("v")
        match = _PATCH_RE.search(question)
        return match.group(1) if match else None

    @staticmethod
    def _extract_mode(question: str, fields: Mapping[str, Any]) -> str | None:
        explicit = fields.get("mode")
        if explicit is not None:
            value = _norm(explicit)
            aliases = {
                "summonersrift": "summoners_rift",
                "sr": "summoners_rift",
                "howlingabyss": "howling_abyss",
                "aram": "howling_abyss",
                "nexusblitz": "nexus_blitz",
            }
            return aliases.get(value, str(explicit).casefold())
        matches = [mode for mode, pattern in _MODE_PATTERNS if re.search(pattern, question, re.I)]
        # Keep the first explicit mode even when the question names two.  The
        # validator then reports the cross-mode contradiction instead of
        # hiding the mode behind an empty slot.
        return matches[0] if matches else None

    def _champion_names(self, question: str) -> list[str]:
        records: list[Mapping[str, Any]] = []
        seen_ids: set[Any] = set()
        for record in getattr(self.oracle, "_champions", []):
            name = record.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            raw_id = record.get("id", name)
            if raw_id in seen_ids:
                continue
            seen_ids.add(raw_id)
            records.append(record)
        matches: list[tuple[int, int, str]] = []
        for order, record in enumerate(records):
            candidates = [record.get("name"), record.get("alias"), *(record.get("aliases") or [])]
            positions = [question.casefold().find(str(item).casefold()) for item in candidates if item and _mentions(question, item)]
            if positions:
                found_positions = [position for position in positions if position >= 0]
                # Boundary matching also accepts punctuation/spacing variants
                # such as ``K Sante`` for ``K'Sante``; those variants do not
                # have a literal ``str.find`` position.
                first_position = min(found_positions) if found_positions else len(question) + order
                matches.append((first_position, -len(_norm(record["name"])), str(record["name"])))
        matches.sort()
        unique: list[str] = []
        for _, _, name in matches:
            if name not in unique:
                unique.append(name)
        return unique

    def parse(self, question: str, context: Mapping[str, Any] | None = None) -> SemanticRequest:
        if not isinstance(question, str):
            question = ""
        supplied = dict(context or {})
        intent = str(supplied.get("intent") or "").strip().casefold()
        lower = question.casefold()
        if not intent:
            if re.search(r"\b(?:if|had|would have|counterfactual|dodg(?:ed|e)|miss(?:ed|es)?)\b", lower) and re.search(r"\b(?:damage|health|hit|deal)\b", lower):
                intent = "counterfactual"
            elif re.search(r"\b(?:win the fight|who wins|kill|survive)\b", lower):
                intent = "fight_outcome"
            elif re.search(r"\b(?:augment|arena[- ]only|arena[- ]specific|mode[- ]specific)\b", lower) or ("patch" in lower and "multiplier" in lower):
                intent = "mode_rule"
            elif re.search(r"\b(?:current build|with the build|full build|build)\b", lower) and re.search(r"\b(?:damage|deal|output)\b", lower):
                intent = "build_damage"
            elif re.search(r"\b(?:damage|remaining health)\b", lower) and supplied.get("attacker"):
                intent = "direct_ability_damage"
            else:
                intent = "unknown"

        fields = dict(supplied)
        patch = self._extract_patch(question, fields)
        if patch is not None:
            fields["patch"] = patch
        mode = self._extract_mode(question, fields)
        if mode is not None:
            fields["mode"] = mode
        entities = self._champion_names(question)
        if entities:
            fields.setdefault("attacker", {})
            if isinstance(fields["attacker"], Mapping):
                attacker = dict(fields["attacker"])
                attacker.setdefault("champion", entities[0])
                fields["attacker"] = attacker
            if intent == "fight_outcome" and len(entities) >= 2:
                fields.setdefault("opponent", {})
                if isinstance(fields["opponent"], Mapping):
                    opponent = dict(fields["opponent"])
                    opponent.setdefault("champion", entities[1])
                    fields["opponent"] = opponent
        # Natural language shorthand is useful for the small direct-damage
        # executor, but it remains an explicit parse rather than a default.
        ability_match = _ABILITY_RANK_RE.search(question)
        if ability_match:
            attacker = dict(fields.get("attacker") or {})
            ability = dict(attacker.get("ability") or {})
            ability.setdefault("key", ability_match.group(1).upper())
            ability.setdefault("rank", int(ability_match.group(2)))
            attacker["ability"] = ability
            fields["attacker"] = attacker
        elif intent == "direct_ability_damage":
            key_match = _ABILITY_KEY_RE.search(question)
            rank_match = _RANK_RE.search(question)
            if key_match or rank_match:
                attacker = dict(fields.get("attacker") or {})
                ability = dict(attacker.get("ability") or {})
                if key_match:
                    ability.setdefault("key", key_match.group(1).upper())
                if rank_match:
                    ability.setdefault("rank", int(rank_match.group(1)))
                attacker["ability"] = ability
                fields["attacker"] = attacker
        level_matches = [int(match.group(1)) for match in _LEVEL_RE.finditer(question)]
        if len(level_matches) == 1:
            attacker = dict(fields.get("attacker") or {})
            attacker.setdefault("level", level_matches[0])
            fields["attacker"] = attacker
        ap_match = _AP_RE.search(question)
        ad_match = _AD_RE.search(question)
        if ap_match or ad_match:
            attacker = dict(fields.get("attacker") or {})
            stats = dict(attacker.get("stats") or {})
            if ap_match:
                stats.setdefault("ability_power", float(ap_match.group(1)))
            if ad_match:
                stats.setdefault("attack_damage", float(ad_match.group(1)))
            attacker["stats"] = stats
            fields["attacker"] = attacker
        if "post-mitigation" in lower or "post mitigation" in lower:
            fields.setdefault("damage_mode", "post_mitigation")
        elif re.search(r"\b(?:raw|pre-mitigation|pre mitigation)\b", lower):
            fields.setdefault("damage_mode", "raw")
        return SemanticRequest(intent=intent, question=question, fields=fields, entities=tuple(entities))

    def _sources(self, request: SemanticRequest, oracle: LeagueOracleEngine | None = None) -> list[dict[str, Any]]:
        fields = request.fields
        names: list[str] = []
        for path in ("attacker.champion", "opponent.champion", "target.champion"):
            value = _path_value(fields, path)
            if isinstance(value, str) and value and value not in names:
                names.append(value)
        links: list[dict[str, Any]] = [
            {"kind": "required", "url": _wiki_url("Damage"), "label": "League Wiki damage rules"},
            {"kind": "required", "url": _wiki_url("Champion statistic"), "label": "League Wiki champion-stat rules"},
        ]
        if request.intent in {"fight_outcome", "counterfactual"}:
            links.extend([
                {"kind": "required", "url": _wiki_url("Armor"), "label": "League Wiki armor rules"},
                {"kind": "required", "url": _wiki_url("Magic resistance"), "label": "League Wiki magic-resistance rules"},
            ])
        if request.intent == "mode_rule":
            links.extend([
                {"kind": "required", "url": _wiki_url("Patch history"), "label": "League Wiki patch history"},
                {"kind": "required", "url": _wiki_url("Arena"), "label": "League Wiki Arena rules"},
            ])
        for name in names:
            links.append({"kind": "wiki", "url": _wiki_url(name), "label": "League Wiki champion page"})
        if oracle is not None:
            client_patch = str(oracle.pack.get("client_patch") or "")
            if client_patch:
                links.append({"kind": "client", "url": _client_url(client_patch), "label": "patch-pinned CommunityDragon client data"})
        unique: dict[str, dict[str, Any]] = {}
        for link in links:
            unique.setdefault(str(link["url"]), link)
        return list(unique.values())

    def _base_response(
        self,
        request: SemanticRequest,
        *,
        status: str,
        reason: str | None = None,
        slots: Sequence[SemanticSlot] = (),
        issues: Sequence[SemanticIssue] = (),
        oracle: LeagueOracleEngine | None = None,
    ) -> dict[str, Any]:
        fields = request.fields
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "semantic_engine": ENGINE_VERSION,
            "status": status,
            "intent": request.intent,
            "display": None,
            "value": None,
            "unit": None,
            "reason": reason,
            "required_inputs": [slot.to_mapping() for slot in slots],
            "provided_inputs": _provided_paths(fields),
            "validation": [issue.to_mapping() for issue in issues],
            "assumptions": [],
            "calculation": None,
            "patch": fields.get("patch"),
            "mode": fields.get("mode"),
            "provenance": {
                "engine": ENGINE_VERSION,
                "schema_version": SCHEMA_VERSION,
                "request_sha256": request.request_sha256,
                "base_oracle_patch": getattr(oracle or self.oracle, "patch", None),
            },
            "sources": self._sources(request, oracle),
        }
        if status == "needs_input":
            labels = ", ".join(slot.path for slot in slots)
            result["display"] = f"Need explicit state for: {labels}." if labels else "Need explicit state before calculating."
        elif status == "invalid_scenario":
            result["display"] = reason or "The supplied scenario is contradictory."
        elif status == "unsupported":
            result["display"] = reason or "This closed rule is outside the validated execution kernel."
        return result

    @staticmethod
    def _slot(path: str, value_type: str, reason: str, *examples: str) -> SemanticSlot:
        return SemanticSlot(path, value_type, reason, tuple(examples))

    def _validate_common(
        self, request: SemanticRequest
    ) -> tuple[list[SemanticSlot], list[SemanticIssue], LeagueOracleEngine | None]:
        fields = request.fields
        slots: list[SemanticSlot] = []
        issues: list[SemanticIssue] = []
        patch = fields.get("patch")
        if patch is None or patch == "":
            slots.append(self._slot("patch", "string", "An exact patch is required; current is not a reproducible value.", CURRENT_PUBLIC_PATCH))
            selected_oracle = self.oracle
        else:
            patch = str(patch).removeprefix("v")
            if not re.fullmatch(r"\d{1,3}\.\d{1,3}", patch):
                issues.append(SemanticIssue("invalid_patch_format", "patch must look like major.minor, for example 26.16", "patch"))
                selected_oracle = None
            elif patch not in self._patch_indices and patch != str(self.oracle.patch):
                issues.append(SemanticIssue("patch_not_available", f"patch {patch} is not present in the local exact packet registry", "patch"))
                selected_oracle = None
            else:
                selected_oracle = self._oracle_for_patch(patch)
                if selected_oracle is None:
                    slots.append(self._slot("patch_packet", "path or packet receipt", f"patch {patch} is known but its exact packet is not loaded/executable", "data/lol/knowledge/patch-packets/..."))

        mode = fields.get("mode")
        if mode is None or mode == "":
            slots.append(self._slot("mode", "enum", "Map/mode changes item, augment, structure, and damage semantics.", "summoners_rift", "arena"))
        else:
            mode = str(mode).casefold()
            if mode not in self.supported_modes:
                issues.append(SemanticIssue("unknown_mode", f"mode {mode!r} is not in the semantic mode registry", "mode"))

        question_lower = request.question.casefold()
        if re.search(r"summoner['’]?s\s+rift", question_lower) and re.search(r"\barena\b", question_lower):
            issues.append(SemanticIssue("mode_mismatch", "the question combines Summoner's Rift and Arena-only rules", "mode"))
        if str(mode).casefold() == "summoners_rift" and re.search(r"arena[- ](?:only|specific)|arena\s+(?:augment|multiplier)", question_lower):
            issues.append(SemanticIssue("mode_mismatch", "an Arena-only rule cannot be evaluated on Summoner's Rift", "mode"))
        return slots, issues, selected_oracle

    def _validate_entities(self, request: SemanticRequest, *, require_opponent: bool = False) -> list[SemanticSlot]:
        fields = request.fields
        slots: list[SemanticSlot] = []
        attacker = fields.get("attacker")
        if not isinstance(attacker, Mapping) or not isinstance(attacker.get("champion"), str) or not str(attacker.get("champion")).strip():
            slots.append(self._slot("attacker.champion", "champion", "Name the attacking champion; do not leave identity implicit.", "Malphite"))
        if require_opponent:
            opponent = fields.get("opponent")
            if not isinstance(opponent, Mapping) or not isinstance(opponent.get("champion"), str) or not str(opponent.get("champion")).strip():
                slots.append(self._slot("opponent.champion", "champion", "The opposing champion/entity is required; 'the enemy' is not a numeric state.", "Darius"))
        return slots

    @staticmethod
    def _known_champion_matches(oracle: LeagueOracleEngine, value: str) -> list[str]:
        matches: list[str] = []
        for record in getattr(oracle, "_champions", []):
            name = record.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            candidates = [name, record.get("alias"), *(record.get("aliases") or [])]
            if any(_norm(candidate) == _norm(value) for candidate in candidates if candidate):
                if name not in matches:
                    matches.append(name)
        return matches

    def _validate_entity_identity(
        self, request: SemanticRequest, oracle: LeagueOracleEngine | None
    ) -> list[SemanticIssue]:
        if oracle is None:
            return []
        issues: list[SemanticIssue] = []
        for path in ("attacker.champion", "opponent.champion", "target.champion"):
            value = _path_value(request.fields, path)
            if not isinstance(value, str) or not value.strip():
                continue
            matches = self._known_champion_matches(oracle, value)
            if not matches:
                issues.append(SemanticIssue("unknown_champion", f"{value!r} is not an exact champion identity in patch {oracle.patch}", path))
            elif len(matches) > 1:
                issues.append(SemanticIssue("ambiguous_champion", f"{value!r} resolves to multiple exact champion identities", path))
        if request.intent in {"build_damage", "direct_ability_damage"} and len(request.entities) > 1 and not _has_value(request.fields, "target.champion"):
            issues.append(SemanticIssue("ambiguous_entities", "more than one champion is named for a single attacker request; identify the target explicitly or remove the extra name", "attacker.champion"))
        if request.intent == "fight_outcome" and len(request.entities) > 2:
            issues.append(SemanticIssue("ambiguous_entities", "a deterministic two-entity fight cannot infer roles from more than two champion names", "entities"))
        return issues

    def _validate_direct(self, request: SemanticRequest, *, build: bool = False) -> tuple[list[SemanticSlot], list[SemanticIssue]]:
        fields = request.fields
        slots = self._validate_entities(request)
        issues: list[SemanticIssue] = []
        attacker = fields.get("attacker") if isinstance(fields.get("attacker"), Mapping) else {}
        ability = attacker.get("ability") if isinstance(attacker, Mapping) else None
        if not isinstance(attacker, Mapping) or not isinstance(attacker.get("level"), int):
            slots.append(self._slot("attacker.level", "integer[1-18]", "Champion level controls level-scaled stats and spell formulas.", "6"))
        elif not 1 <= int(attacker["level"]) <= 18:
            issues.append(SemanticIssue("invalid_level", "champion level must be an integer in [1, 18]", "attacker.level"))
        if not isinstance(ability, Mapping) or str(ability.get("key", "")).upper() not in {"Q", "W", "E", "R"}:
            slots.append(self._slot("attacker.ability.key", "enum[Q,W,E,R]", "Identify the ability being calculated.", "Q"))
        if not isinstance(ability, Mapping) or not isinstance(ability.get("rank"), int):
            slots.append(self._slot("attacker.ability.rank", "integer[1-5]", "Ability rank is not inferable from champion level.", "3"))
        elif not 1 <= int(ability["rank"]) <= 5:
            issues.append(SemanticIssue("invalid_ability_rank", "ability rank must be an integer in [1, 5]", "attacker.ability.rank"))
        stats = attacker.get("stats") if isinstance(attacker, Mapping) else None
        if not isinstance(stats, Mapping) or not any(isinstance(stats.get(key), (int, float)) and not isinstance(stats.get(key), bool) for key in ("ability_power", "attack_damage")):
            slots.append(self._slot("attacker.stats", "object", "Supply the exact AP or total AD used by the formula; build text alone is not a stat value.", "{\"ability_power\": 100}"))
        damage_mode = fields.get("damage_mode")
        if damage_mode is None:
            slots.append(self._slot("damage_mode", "enum[raw,post_mitigation]", "State whether the result is raw or after a target's defenses.", "post_mitigation"))
        elif str(damage_mode) not in {"raw", "post_mitigation"}:
            issues.append(SemanticIssue("invalid_damage_mode", "damage_mode must be raw or post_mitigation", "damage_mode"))
        if build:
            for path, value_type, reason, example in (
                ("attacker.items", "array", "Current build means the equipped item set must be explicit, including an empty set.", "[]"),
                ("attacker.runes", "array", "Runes can change damage and must be explicit, including an empty set.", "[]"),
                ("attacker.buffs", "object", "Temporary buffs and transformations must be explicit.", "{}"),
                ("attacker.debuffs", "object", "Target/debuff state must be explicit.", "{}"),
                ("event_state", "object", "Cooldowns, stacks, distance, and timing are part of a live build query.", "{}"),
            ):
                if not _has_value(fields, path):
                    slots.append(self._slot(path, value_type, reason, example))
        if str(damage_mode) == "post_mitigation":
            target = fields.get("target") if isinstance(fields.get("target"), Mapping) else None
            if not isinstance(target, Mapping):
                slots.append(self._slot("target", "object", "Post-mitigation damage needs an explicit target state.", "{\"health\": 1000, \"magic_resist\": 50}"))
            else:
                if not isinstance(target.get("health"), (int, float)):
                    slots.append(self._slot("target.health", "number", "Target starting health is part of the closed state.", "1000"))
                defenses = [key for key in ("armor", "magic_resist") if isinstance(target.get(key), (int, float))]
                damage_type = fields.get("damage_type")
                if len(defenses) == 0:
                    slots.append(self._slot("target.defenses", "object", "Provide the defense matching the damage channel.", "{\"magic_resist\": 50}"))
                elif len(defenses) > 1 and damage_type not in {"physical", "magic"}:
                    slots.append(self._slot("damage_type", "enum[physical,magic]", "Both defenses were supplied; select the spell's mitigation channel explicitly.", "magic"))
                if not _has_value(fields, "penetration") and not _has_value(fields, "no_penetration"):
                    slots.append(self._slot("penetration", "object or false", "Penetration cannot be assumed absent in a build query; state it explicitly.", "{}"))
                for path, example in (("target.shields", "[]"), ("target.buffs", "{}")):
                    if build and not _has_value(fields, path):
                        slots.append(self._slot(path, "array/object", "Defensive modifiers must be explicit for an exact build result.", example))
        return slots, issues

    def _validate_event_contract(self, request: SemanticRequest, *, counterfactual: bool = False) -> tuple[list[SemanticSlot], list[SemanticIssue]]:
        fields = request.fields
        slots = self._validate_entities(request, require_opponent=request.intent == "fight_outcome")
        issues: list[SemanticIssue] = []
        if not isinstance(fields.get("initial_state"), Mapping):
            slots.append(self._slot("initial_state", "object", "A deterministic fight needs explicit starting health, defenses, resources, and transforms.", "{\"entities\": {...}}"))
        events = fields.get("events")
        if not isinstance(events, list) or not events:
            slots.append(self._slot("events", "array", "Supply the ordered numeric event timeline; an outcome cannot be inferred from champion names.", "[{\"at_ms\": 100, \"damage_type\": \"magic\", ...}]"))
        elif not all(isinstance(event, Mapping) for event in events):
            issues.append(SemanticIssue("invalid_event_shape", "every event must be an object", "events"))
        if request.intent == "fight_outcome" and not isinstance(fields.get("win_condition"), str):
            slots.append(self._slot("win_condition", "enum", "Define what 'win' means (for example first death).", "first_death"))
        if counterfactual:
            counter = fields.get("counterfactual")
            if not isinstance(counter, Mapping):
                slots.append(self._slot("counterfactual", "object", "State the exact event/time to remove or replace.", "{\"remove_event_id\": \"event-2\"}"))
            elif not counter.get("remove_event_id") and counter.get("remove_index") is None:
                slots.append(self._slot("counterfactual.remove_event_id", "string", "A dodge must identify the exact observed event it changes.", "event-2"))
            target_id = counter.get("target_id") if isinstance(counter, Mapping) else None
            if not isinstance(target_id, str) or not target_id:
                slots.append(self._slot("counterfactual.target_id", "entity id", "Name the target whose counterfactual damage is being compared.", "target"))
        return slots, issues

    def _validate(self, request: SemanticRequest) -> tuple[list[SemanticSlot], list[SemanticIssue], LeagueOracleEngine | None]:
        slots, issues, selected_oracle = self._validate_common(request)
        issues.extend(self._validate_entity_identity(request, selected_oracle))
        if request.intent == "unknown":
            issues.append(SemanticIssue("intent_unrecognized", "the semantic layer could not identify a supported calculation intent", "intent"))
            return slots, issues, selected_oracle
        if request.intent in {"build_damage", "direct_ability_damage"}:
            more_slots, more_issues = self._validate_direct(request, build=request.intent == "build_damage")
            slots.extend(more_slots)
            issues.extend(more_issues)
        elif request.intent == "fight_outcome":
            more_slots, more_issues = self._validate_event_contract(request)
            slots.extend(more_slots)
            issues.extend(more_issues)
        elif request.intent == "counterfactual":
            more_slots, more_issues = self._validate_event_contract(request, counterfactual=True)
            slots.extend(more_slots)
            issues.extend(more_issues)
        elif request.intent == "mode_rule":
            if not isinstance(request.fields.get("rule"), str) or not request.fields.get("rule", "").strip():
                slots.append(self._slot("rule", "string", "Name the exact rule/augment/passive to evaluate.", "Arena-only augment multiplier"))
        else:
            issues.append(SemanticIssue("unsupported_intent", f"intent {request.intent!r} is not executable", "intent"))
        return slots, issues, selected_oracle

    @staticmethod
    def _combatant_from_mapping(entity_id: str, team_id: str, payload: Mapping[str, Any]) -> Combatant:
        level = payload.get("level", 1)
        health = payload.get("health", payload.get("max_health", 1.0))
        max_health = payload.get("max_health", health)
        if not isinstance(level, int) or isinstance(level, bool):
            raise ValueError(f"{entity_id}.level must be an integer")
        if not isinstance(health, (int, float)) or not isinstance(max_health, (int, float)):
            raise ValueError(f"{entity_id} health and max_health must be numeric")
        raw_stats = payload.get("stats") if isinstance(payload.get("stats"), Mapping) else {}
        stats = dict(raw_stats)
        for key in ("armor", "magic_resist", "tenacity"):
            if key in payload:
                stats[key] = payload[key]
        return Combatant(
            entity_id=entity_id,
            team_id=str(payload.get("team_id", team_id)),
            champion_id=str(payload.get("champion", payload.get("champion_id", entity_id))),
            level=level,
            health=float(health),
            max_health=float(max_health),
            stats={key: float(value) for key, value in stats.items() if isinstance(value, (int, float)) and not isinstance(value, bool)},
            resources={key: float(value) for key, value in (payload.get("resources") or {}).items() if isinstance(value, (int, float))},
            cooldowns={key: int(value) for key, value in (payload.get("cooldowns") or {}).items() if isinstance(value, int) and not isinstance(value, bool)},
            items=tuple(str(item) for item in payload.get("items", ()) or ()),
            runes=tuple(str(item) for item in payload.get("runes", ()) or ()),
            buffs={key: float(value) for key, value in (payload.get("buffs") or {}).items() if isinstance(value, (int, float))},
            marks={key: int(value) for key, value in (payload.get("marks") or {}).items() if isinstance(value, int) and not isinstance(value, bool)},
            alive=bool(payload.get("alive", float(health) > 0)),
            transform=str(payload["transform"]) if payload.get("transform") is not None else None,
        )

    def _initial_state(self, fields: Mapping[str, Any]) -> GameState:
        raw = fields.get("initial_state")
        if not isinstance(raw, Mapping):
            raise ValueError("initial_state must be an object")
        entities_raw = raw.get("entities")
        if not isinstance(entities_raw, Mapping):
            raise ValueError("initial_state.entities must be an object")
        entities: dict[str, Combatant] = {}
        for entity_id, payload in entities_raw.items():
            if not isinstance(payload, Mapping):
                raise ValueError(f"initial_state.entities.{entity_id} must be an object")
            entities[str(entity_id)] = self._combatant_from_mapping(str(entity_id), "team", payload)
        return GameState(
            clock_ms=int(raw.get("clock_ms", 0)),
            entities=entities,
            zones=raw.get("zones", {}) if isinstance(raw.get("zones", {}), Mapping) else {},
            summons=raw.get("summons", {}) if isinstance(raw.get("summons", {}), Mapping) else {},
            objectives=raw.get("objectives", {}) if isinstance(raw.get("objectives", {}), Mapping) else {},
            vision=raw.get("vision", {}) if isinstance(raw.get("vision", {}), Mapping) else {},
            rules=raw.get("rules", {}) if isinstance(raw.get("rules", {}), Mapping) else {},
        )

    @staticmethod
    def _events(fields: Mapping[str, Any]) -> tuple[Event, ...]:
        raw_events = fields.get("events")
        if not isinstance(raw_events, list):
            raise ValueError("events must be an array")
        events: list[Event] = []
        for index, raw in enumerate(raw_events):
            if not isinstance(raw, Mapping):
                raise ValueError(f"events[{index}] must be an object")
            at_ms = raw.get("at_ms")
            source_id = raw.get("source_id")
            target_id = raw.get("target_id")
            amount = raw.get("amount")
            damage_type = raw.get("damage_type")
            if not isinstance(at_ms, int) or at_ms < 0:
                raise ValueError(f"events[{index}].at_ms must be a non-negative integer")
            if not all(isinstance(value, str) and value for value in (source_id, target_id, damage_type)):
                raise ValueError(f"events[{index}] needs source_id, target_id, and damage_type")
            if damage_type not in {"physical", "magic", "true"}:
                raise ValueError(f"events[{index}].damage_type is not executable")
            if not isinstance(amount, (int, float)) or isinstance(amount, bool) or not math.isfinite(float(amount)) or float(amount) < 0:
                raise ValueError(f"events[{index}].amount must be finite and non-negative")
            penetration = raw.get("penetration", {})
            if not isinstance(penetration, Mapping):
                raise ValueError(f"events[{index}].penetration must be an object")
            events.append(
                Event(
                    at_ms=at_ms,
                    priority=int(raw.get("priority", 0)),
                    source_id=source_id,
                    ordinal=int(raw.get("ordinal", index)),
                    event_id=str(raw["event_id"]) if raw.get("event_id") else None,
                    effect=Damage(
                        source_id=source_id,
                        target_id=target_id,
                        amount=float(amount),
                        damage_type=damage_type,
                        penetration={str(key): float(value) for key, value in penetration.items() if isinstance(value, (int, float))},
                    ),
                )
            )
        return tuple(events)

    def _execute_direct(self, request: SemanticRequest, oracle: LeagueOracleEngine) -> dict[str, Any]:
        fields = request.fields
        attacker = fields.get("attacker")
        if not isinstance(attacker, Mapping):
            raise ValueError("attacker is missing")
        champion = str(attacker["champion"])
        ability = attacker["ability"]
        stats = attacker["stats"]
        if not isinstance(ability, Mapping) or not isinstance(stats, Mapping):
            raise ValueError("ability and stats are required")
        unsupported_state: list[str] = []
        for path in (
            "attacker.items",
            "attacker.runes",
            "attacker.buffs",
            "attacker.debuffs",
            "event_state",
            "target.shields",
            "target.buffs",
        ):
            value = _path_value(fields, path)
            if value not in (None, {}, [], (), False):
                unsupported_state.append(path)
        penetration = fields.get("penetration")
        if penetration not in (None, {}, [], (), False):
            unsupported_state.append("penetration")
        if unsupported_state:
            return self._base_response(
                request,
                status="unsupported",
                reason="the closed direct-damage bridge does not yet execute non-empty " + ", ".join(sorted(set(unsupported_state))),
                slots=(self._slot("validated_effect_rules", "revision-receipted rule set", "Non-empty items, runes, buffs, shields, penetration, or live state need validated effect semantics."),),
                oracle=oracle,
            )
        key = str(ability["key"]).upper()
        rank = int(ability["rank"])
        level = int(attacker["level"])
        stat_parts: list[str] = []
        if stats.get("ability_power") is not None:
            stat_parts.append(f"{float(stats['ability_power']):g} AP")
        if stats.get("attack_damage") is not None:
            stat_parts.append(f"{float(stats['attack_damage']):g} total AD")
        question = f"What exact damage does {champion}'s rank-{rank} {key} deal with {' and '.join(stat_parts)} at level {level}"
        mode = str(fields.get("damage_mode"))
        if mode == "post_mitigation":
            target = fields["target"]
            damage_type = fields.get("damage_type")
            defenses = [key for key in ("armor", "magic_resist") if isinstance(target.get(key), (int, float))]
            selected_defense = "magic_resist" if damage_type == "magic" else "armor" if damage_type == "physical" else defenses[0]
            label = "magic resistance" if selected_defense == "magic_resist" else "armor"
            question += f" against a target with {float(target[selected_defense]):g} {label} and no penetration"
        else:
            question += " against a target with no modifiers"
        result = oracle.answer(question)
        result = dict(result)
        result["semantic_request_sha256"] = request.request_sha256
        result["semantic_intent"] = request.intent
        result["provenance"] = {
            **dict(result.get("provenance") or {}),
            "semantic_engine": ENGINE_VERSION,
            "semantic_request_sha256": request.request_sha256,
        }
        if result.get("status") == "available":
            result["assumptions"] = list(result.get("assumptions", [])) + [
                "semantic state explicitly sets items, runes, buffs, debuffs, and event_state",
            ]
        return result

    def _execute_events(self, request: SemanticRequest, *, counterfactual: bool = False) -> dict[str, Any]:
        fields = request.fields
        initial = self._initial_state(fields)
        events = self._events(fields)
        baseline = MechanicsEngine().run(initial, events)
        if not baseline.available:
            reason = "; ".join(item.message for item in baseline.unknowns)
            return self._base_response(request, status="unsupported", reason=f"event timeline returned unknown: {reason}")
        selected = baseline
        comparison: dict[str, Any] = {}
        if counterfactual:
            counter = fields["counterfactual"]
            remove_id = counter.get("remove_event_id") if isinstance(counter, Mapping) else None
            remove_index = counter.get("remove_index") if isinstance(counter, Mapping) else None
            if remove_id is not None:
                filtered = tuple(event for event in events if event.stable_id != str(remove_id) and event.event_id != str(remove_id))
                if len(filtered) == len(events):
                    return self._base_response(request, status="invalid_scenario", reason=f"counterfactual event {remove_id!r} is absent", issues=(SemanticIssue("event_not_found", "the counterfactual event id is not in the observed timeline", "counterfactual.remove_event_id"),))
            elif isinstance(remove_index, int) and 0 <= remove_index < len(events):
                filtered = tuple(event for index, event in enumerate(events) if index != remove_index)
            else:
                return self._base_response(request, status="invalid_scenario", reason="counterfactual event selector is invalid")
            selected = MechanicsEngine().run(initial, filtered)
            if not selected.available:
                reason = "; ".join(item.message for item in selected.unknowns)
                return self._base_response(request, status="unsupported", reason=f"counterfactual timeline returned unknown: {reason}")
            target_id = str(counter["target_id"])
            before = baseline.state.entities.get(target_id)
            after = selected.state.entities.get(target_id)
            if before is None or after is None:
                return self._base_response(request, status="invalid_scenario", reason=f"counterfactual target {target_id!r} is absent")
            comparison = {
                "baseline_remaining_health": round(before.health, 6),
                "counterfactual_remaining_health": round(after.health, 6),
                "damage_avoided": round(after.health - before.health, 6),
                "target_id": target_id,
            }
        if request.intent == "fight_outcome":
            attacker_id = str(fields.get("attacker", {}).get("entity_id", "attacker"))
            opponent_id = str(fields.get("opponent", {}).get("entity_id", "opponent"))
            attacker_state = selected.state.entities.get(attacker_id)
            opponent_state = selected.state.entities.get(opponent_id)
            if attacker_state is None or opponent_state is None:
                return self._base_response(request, status="invalid_scenario", reason="attacker/opponent entity ids are absent from initial_state")
            if not attacker_state.alive and not opponent_state.alive:
                winner = "draw"
            elif not attacker_state.alive:
                winner = "opponent"
            elif not opponent_state.alive:
                winner = "attacker"
            else:
                winner = "undecided"
            value: Any = winner
            display = f"{winner} ({attacker_state.health:.2f} vs {opponent_state.health:.2f} health)"
            unit = "winner"
        elif counterfactual:
            value = comparison
            display = f"{comparison['damage_avoided']:g} damage avoided for {comparison['target_id']}"
            unit = "counterfactual damage comparison"
        else:
            value = {entity_id: round(entity.health, 6) for entity_id, entity in sorted(selected.state.entities.items())}
            display = "final health: " + ", ".join(f"{entity_id}={health:g}" for entity_id, health in value.items())
            unit = "final health"
        result = self._base_response(request, status="available")
        result.update(
            {
                "display": display,
                "value": value,
                "unit": unit,
                "calculation": "Explicit events executed in stable event order through MechanicsEngine.",
                "assumptions": ["only the supplied numeric Damage events are evaluated", "no unlisted champion, item, rune, or live-state effects"],
                "provenance": {
                    **result["provenance"],
                    "initial_state_sha256": initial.state_sha256,
                    "trace_sha256": selected.trace.trace_sha256,
                    "final_state_sha256": selected.trace.final_state_sha256,
                },
            }
        )
        return result

    def _execute(self, request: SemanticRequest, oracle: LeagueOracleEngine) -> dict[str, Any]:
        if request.intent in {"build_damage", "direct_ability_damage"}:
            return self._execute_direct(request, oracle)
        if request.intent == "fight_outcome":
            return self._execute_events(request)
        if request.intent == "counterfactual":
            return self._execute_events(request, counterfactual=True)
        if request.intent == "mode_rule":
            return self._base_response(
                request,
                status="unsupported",
                reason="the scenario is syntactically valid, but no revision-receipted executable rule is loaded for this mode-specific request",
                slots=(self._slot("executable_rule", "revision-receipted rule", "A numeric answer requires an executable rule receipt, not only descriptive text."),),
                oracle=oracle,
            )
        return self._base_response(request, status="unsupported", reason=f"intent {request.intent!r} is not executable", oracle=oracle)

    def answer_request(self, request: SemanticRequest | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(request, SemanticRequest):
            parsed = request
        elif isinstance(request, Mapping):
            question = str(request.get("question", ""))
            context = dict(request.get("context") or {}) if isinstance(request.get("context"), Mapping) else dict(request)
            context.pop("question", None)
            parsed = self.parse(question, context)
        else:
            return {"status": "invalid_scenario", "reason": "semantic request must be an object"}
        slots, issues, selected_oracle = self._validate(parsed)
        if issues:
            status = "invalid_scenario" if any(issue.code in {"invalid_patch_format", "patch_not_available", "unknown_mode", "mode_mismatch", "invalid_level", "invalid_ability_rank", "invalid_damage_mode", "intent_unrecognized", "unsupported_intent", "invalid_event_shape", "unknown_champion", "ambiguous_champion", "ambiguous_entities"} for issue in issues) else "needs_input"
            reason = "; ".join(issue.message for issue in issues)
            return self._base_response(parsed, status=status, reason=reason, slots=slots, issues=issues, oracle=selected_oracle)
        if slots:
            return self._base_response(parsed, status="needs_input", slots=slots, oracle=selected_oracle)
        if selected_oracle is None:
            return self._base_response(parsed, status="needs_input", slots=(self._slot("patch_packet", "path or packet receipt", "An exact executable patch packet is required."),))
        try:
            return self._execute(parsed, selected_oracle)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            return self._base_response(parsed, status="invalid_scenario", reason=f"closed state failed validation: {exc}", oracle=selected_oracle)

    def answer(self, question: str, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return self.answer_request(self.parse(question, context))


__all__ = [
    "ENGINE_VERSION",
    "SCHEMA_VERSION",
    "SemanticIssue",
    "SemanticOracleEngine",
    "SemanticRequest",
    "SemanticSlot",
]
