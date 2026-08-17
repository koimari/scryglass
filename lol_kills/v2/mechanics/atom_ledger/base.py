"""Build and load a complete hash-bound base snapshot from LCC atoms."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schema import (
    ATOM_CATEGORIES,
    BASE_AUTHORITY_STATUS,
    BASE_SCHEMA_ID,
    LEDGER_SCHEMA_ID,
    AtomLedgerIntegrityError,
    canonical_sha256,
    stable_atom_id,
    validate_signed_hash,
)

DEFAULT_LCC_REPO = Path("/Users/river/Projects/league-combat-calculator")
DEFAULT_BASE_PATH = Path(__file__).with_name("snapshots") / "lcc-26.15-base.json.gz"
LCC_DOMAINS: tuple[str, ...] = (
    "abilities",
    "champions",
    "economics",
    "items",
    "runes",
    "stats",
)
EXPECTED_DOMAIN_COUNTS: Mapping[str, int] = {
    "abilities": 5093,
    "champions": 5372,
    "economics": 817,
    "items": 1664,
    "runes": 127,
    "stats": 6779,
}
MISSING_GAME_DOMAINS: tuple[str, ...] = (
    "complete_cross_entity_interactions",
    "complete_objective_state_machines",
    "global_system_rules",
    "maps",
    "minions",
    "neutral_monsters",
    "structures",
    "summoner_spells",
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _primary_category(domain: str, source: str) -> str:
    if domain in {"items", "economics"}:
        return "item"
    if domain == "runes":
        return "rune"
    if domain == "stats":
        return "champion"
    lowered = source.lower()
    if ".p[" in lowered or "passive" in lowered:
        return "passive"
    return "spell"


def _categories(
    domain: str, entity: str, raw: Mapping[str, Any], primary: str
) -> list[str]:
    text = " ".join(
        str(raw.get(key, "")) for key in ("atom_id", "behavior", "source", "name")
    ).lower()
    categories = {primary}
    if primary in {"spell", "passive"} or domain == "stats":
        categories.add("champion")
    if domain in {"abilities", "champions", "items", "runes"}:
        categories.add("effect")
    tokens = {
        "buff": ("buff",),
        "debuff": ("debuff",),
        "trigger": ("on_hit", "on-hit", "trigger"),
        "target": ("target",),
        "cooldown": ("cooldown",),
        "cost": ("cost", "economy.total", "economy.sell"),
        "range": ("range", "radius"),
        "duration": ("duration",),
        "stack": ("stack",),
        "reset": ("reset",),
        "state-transition": ("transform", "revive", "stasis", "state", "summon"),
    }
    for category, needles in tokens.items():
        if any(needle in text for needle in needles):
            categories.add(category)
    values = raw.get("values", [])
    if isinstance(values, list) and len(values) > 1:
        categories.add("formula")
    if any(
        word in entity.lower()
        for word in ("turret plating", "baron nashor", "dragon slayer")
    ):
        categories.add("objective")
    return [category for category in ATOM_CATEGORIES if category in categories]


def _authority(domain: str, evidence: Iterable[str]) -> tuple[str, float]:
    # The specialist champion domain reads numeric fields from game binaries.
    # Its Wiki receipts classify behavior and do not refresh those numbers.
    if domain == "champions":
        return "binary_patch_bound", 1.0
    receipts = [item.lower() for item in evidence]
    has_wiki = any(item.startswith("wiki:") for item in receipts)
    has_binary = any(item.startswith("binary:") for item in receipts)
    if has_wiki and has_binary:
        return "wiki_and_binary_patch_bound", 1.0
    if has_wiki:
        return "wiki_patch_bound", 1.0
    if has_binary:
        return "binary_patch_bound", 1.0
    return "lcc_source_receipt_patch_bound", 1.0


def _normalize_atom(
    domain: str, entity: str, raw: Mapping[str, Any], source_slot: int
) -> dict[str, Any]:
    identity = {
        "domain": domain,
        "entity": entity,
        "source_atom_id": str(raw["atom_id"]),
        "source_locator": str(raw["source"]),
        "source_slot": f"slot:{source_slot:03d}",
    }
    behavior = str(raw["behavior"])
    primary = _primary_category(domain, identity["source_locator"])
    evidence = [str(item) for item in raw.get("evidence", [])]
    authority, confidence = _authority(domain, evidence)
    values = raw.get("values", [])
    units = raw.get("units", [])
    if not isinstance(values, list) or not isinstance(units, list):
        raise AtomLedgerIntegrityError(
            f"LCC atom {domain}/{entity} has invalid values or units"
        )
    if len(values) != len(units):
        raise AtomLedgerIntegrityError(
            f"LCC atom {domain}/{entity} has unequal values and units"
        )
    fields: dict[str, dict[str, Any]] = {}
    if values:
        for index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise AtomLedgerIntegrityError(
                    f"LCC atom {domain}/{entity} has a non-numeric value"
                )
            fields[f"value:{index:03d}"] = {
                "value": value,
                "source": identity["source_locator"],
                "unit": str(units[index]) or None,
                "confidence": confidence,
                "missing": False,
                "authority": authority,
            }
    else:
        fields["value:000"] = {
            "value": None,
            "source": identity["source_locator"],
            "unit": None,
            "confidence": confidence,
            "missing": True,
            "authority": authority,
        }
    atom = {
        "atom_id": stable_atom_id(primary, identity),
        "identity": identity,
        "primary_category": primary,
        "behavior": behavior,
        "categories": _categories(domain, entity, raw, primary),
        "name": str(raw.get("name", "")),
        "fields": fields,
        "missing_mask": {name: cell["missing"] for name, cell in fields.items()},
        "evidence": sorted(set(evidence)),
        "source_record_hash": str(raw.get("hash", "")),
        "active": True,
    }
    atom["record_hash"] = canonical_sha256(atom)
    return atom


def build_base_snapshot(lcc_repo: Path = DEFAULT_LCC_REPO) -> dict[str, Any]:
    """Normalize every atom in the certified LCC 26.15 six-domain corpus."""

    root = lcc_repo / "data" / "atoms"
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise AtomLedgerIntegrityError(f"missing LCC 26.15 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    atoms: list[dict[str, Any]] = []
    source_files = {"data/atoms/manifest.json": _file_sha256(manifest_path)}
    domain_counts: dict[str, int] = {}
    for domain in LCC_DOMAINS:
        path = root / f"{domain}.json"
        if not path.is_file():
            raise AtomLedgerIntegrityError(f"missing LCC 26.15 domain: {path}")
        source_files[f"data/atoms/{domain}.json"] = _file_sha256(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("domain") != domain or not isinstance(
            payload.get("objects"), dict
        ):
            raise AtomLedgerIntegrityError(f"invalid LCC domain payload: {domain}")
        domain_atoms: list[dict[str, Any]] = []
        for entity in sorted(payload["objects"]):
            raw_atoms = payload["objects"][entity]
            if not isinstance(raw_atoms, list):
                raise AtomLedgerIntegrityError(
                    f"LCC domain object {domain}/{entity} must be a list"
                )
            source_slots: Counter[tuple[str, str]] = Counter()
            for raw in raw_atoms:
                locator_key = (str(raw["atom_id"]), str(raw["source"]))
                domain_atoms.append(
                    _normalize_atom(
                        domain, entity, raw, source_slot=source_slots[locator_key]
                    )
                )
                source_slots[locator_key] += 1
        expected = EXPECTED_DOMAIN_COUNTS[domain]
        if len(domain_atoms) != expected:
            raise AtomLedgerIntegrityError(
                f"LCC 26.15 {domain} count is {len(domain_atoms)}; expected {expected}"
            )
        domain_counts[domain] = len(domain_atoms)
        atoms.extend(domain_atoms)
    atoms.sort(key=lambda atom: atom["atom_id"])
    atom_ids = [atom["atom_id"] for atom in atoms]
    if len(atom_ids) != len(set(atom_ids)):
        raise AtomLedgerIntegrityError("stable LCC atom IDs collide")
    category_counts: Counter[str] = Counter()
    for atom in atoms:
        category_counts.update(atom["categories"])
    snapshot = {
        "schema_id": BASE_SCHEMA_ID,
        "ledger_schema_id": LEDGER_SCHEMA_ID,
        "snapshot_kind": "base",
        "patch": "26.15",
        "authority_status": BASE_AUTHORITY_STATUS,
        "authority_scope": "exact_to_hash_bound_lcc_26.15_six_domain_corpus",
        "coverage": {
            "status": "measured_partial",
            "source_corpus_ingestion": "complete",
            "full_wiki_game_coverage": False,
            "scope": "lcc_six_domain_vertical_slice",
            "missing_game_domains": list(MISSING_GAME_DOMAINS),
        },
        "source_binding": {
            "project": "league-combat-calculator",
            "certification_commit": "8c6a0ba2e04882ad9f607bf6441e50ddb45ebf01",
            "manifest_schema_version": manifest.get("schema_version"),
            "manifest_domains": manifest.get("domains"),
            "file_sha256": dict(sorted(source_files.items())),
        },
        "domain_counts": domain_counts,
        "category_counts": dict(sorted(category_counts.items())),
        "atom_count": len(atoms),
        "atoms": atoms,
    }
    snapshot["snapshot_hash"] = canonical_sha256(snapshot)
    return snapshot


def validate_base_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    if (
        snapshot.get("schema_id") != BASE_SCHEMA_ID
        or snapshot.get("snapshot_kind") != "base"
    ):
        raise AtomLedgerIntegrityError("base snapshot schema is invalid")
    if snapshot.get("patch") != "26.15":
        raise AtomLedgerIntegrityError("the first ledger base must be patch 26.15")
    if snapshot.get("authority_status") != BASE_AUTHORITY_STATUS:
        raise AtomLedgerIntegrityError("base authority must be exact and patch bound")
    coverage = snapshot.get("coverage")
    if (
        not isinstance(coverage, dict)
        or coverage.get("source_corpus_ingestion") != "complete"
    ):
        raise AtomLedgerIntegrityError("base source corpus coverage is invalid")
    if coverage.get("full_wiki_game_coverage") is not False:
        raise AtomLedgerIntegrityError("base must not claim full Wiki or game coverage")
    validate_signed_hash(snapshot, "snapshot_hash", "base snapshot")
    atoms = snapshot.get("atoms")
    if not isinstance(atoms, list) or len(atoms) != sum(
        EXPECTED_DOMAIN_COUNTS.values()
    ):
        raise AtomLedgerIntegrityError("base snapshot is incomplete")
    atom_ids: set[str] = set()
    for atom in atoms:
        if not isinstance(atom, dict):
            raise AtomLedgerIntegrityError("base atom must be an object")
        atom_id = atom.get("atom_id")
        if not isinstance(atom_id, str) or atom_id in atom_ids:
            raise AtomLedgerIntegrityError("base atom IDs must be unique strings")
        atom_ids.add(atom_id)
        validate_signed_hash(atom, "record_hash", f"atom {atom_id}")
        identity = atom.get("identity")
        primary = atom.get("primary_category")
        if (
            not isinstance(identity, dict)
            or not isinstance(primary, str)
            or stable_atom_id(primary, identity) != atom_id
        ):
            raise AtomLedgerIntegrityError(f"atom {atom_id} stable identity is invalid")
        if not isinstance(atom.get("behavior"), str):
            raise AtomLedgerIntegrityError(
                f"atom {atom_id} behavior must be mutable text"
            )
        fields = atom.get("fields")
        missing_mask = atom.get("missing_mask")
        if not isinstance(fields, dict) or not isinstance(missing_mask, dict):
            raise AtomLedgerIntegrityError(f"atom {atom_id} has invalid fields")
        for field_name, cell in fields.items():
            required = {"value", "source", "unit", "confidence", "missing", "authority"}
            if not isinstance(cell, dict) or set(cell) != required:
                raise AtomLedgerIntegrityError(
                    f"atom {atom_id}.{field_name} field schema is invalid"
                )
            if missing_mask.get(field_name) is not cell["missing"]:
                raise AtomLedgerIntegrityError(
                    f"atom {atom_id}.{field_name} missing mask differs"
                )
    return dict(snapshot)


def load_base_snapshot(path: Path = DEFAULT_BASE_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise AtomLedgerIntegrityError(f"missing base snapshot: {path}")
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise AtomLedgerIntegrityError(f"cannot load base snapshot: {path}") from exc
    if not isinstance(payload, dict):
        raise AtomLedgerIntegrityError("base snapshot must be an object")
    return validate_base_snapshot(payload)


def write_base_snapshot(
    snapshot: Mapping[str, Any], path: Path = DEFAULT_BASE_PATH
) -> None:
    """Write a reproducible gzip artifact. The fixed mtime keeps bytes stable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_handle, mtime=0
        ) as zipped:
            zipped.write(raw)


if __name__ == "__main__":
    built = build_base_snapshot()
    write_base_snapshot(built)
    print(
        f"wrote {DEFAULT_BASE_PATH} ({built['atom_count']} atoms, {built['snapshot_hash']})"
    )
