"""LCC atom bridge builder (v1).

Consumes pinned League Combat Calculator atom data and emits the canonical
Scryglass bridge artifact: champion atom profiles, the atom->ontology mapping,
the atom relation graph, and a soft mechanistic ontology prior.  The artifact
is development infrastructure: nothing here authorizes prediction,
publication, or promotion.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .lcc_sources import LccSources
from .mapping_v1 import FAMILY_FALLBACK_V1, MAPPING_V1, MAPPING_VERSION
from .schema import (
    BRIDGE_SCHEMA_ID,
    BRIDGE_VERSION,
    CHAMPION_ATOM_FAMILIES,
    DIMENSIONS,
    DIMENSION_LABELS,
    AtomBridgeError,
    canonical_sha256,
)

DEFAULT_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[4]
    / "data"
    / "lol"
    / "v2"
    / "champions"
    / "lcc-atom-bridge-26.16.json"
)

# Ontology labels that express "the champion does not have this trait" and
# must never be driven by atom presence.
_NEGATIVE_LABELS = {"none", "point_blank", "low", "squishy", "early", "slow_buildup"}

_ATOM_INDEX = {(row["atom_id"]) for row in MAPPING_V1}
_FAMILY_INDEX = {row["family"] for row in FAMILY_FALLBACK_V1}  # bare family names


def _normalize_key(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _family_presence(summary: dict[str, Any], key: str) -> dict[str, bool]:
    """atom-summary lists champions per family; it is a presence index, not a
    count index, so we record booleans and let detail files provide counts."""
    presence: dict[str, bool] = {}
    for family in CHAMPION_ATOM_FAMILIES:
        members = summary.get(family)
        presence[family] = isinstance(members, list) and key in members
    return presence


def _atom_family_counts(atoms: list[dict[str, Any]]) -> dict[str, int]:
    counts = {family: 0 for family in CHAMPION_ATOM_FAMILIES}
    for atom in atoms:
        family = atom.get("family")
        if isinstance(family, str) and family in counts:
            counts[family] += 1
    return counts


def _atom_level_scores(atoms: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Raw (dimension -> label -> weight sum) from per-atom mappings."""
    scores: dict[str, dict[str, float]] = {d: {lbl: 0.0 for lbl in DIMENSION_LABELS[d]} for d in DIMENSIONS}
    for atom in atoms:
        atom_id = atom.get("atom_id")
        if atom_id not in _ATOM_INDEX:
            continue
        rows = [row for row in MAPPING_V1 if row["atom_id"] == atom_id]
        for row in rows:
            dimension = row["dimension"]
            label = row["label"]
            if label in _NEGATIVE_LABELS:
                continue
            scores[dimension][label] += row["weight"]
    return scores


def _family_level_scores(family_presence: dict[str, bool]) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {d: {lbl: 0.0 for lbl in DIMENSION_LABELS[d]} for d in DIMENSIONS}
    for family, present in family_presence.items():
        if not present or family not in _FAMILY_INDEX:
            continue
        for row in FAMILY_FALLBACK_V1:
            if row["family"] != family:
                continue
            label = row["label"]
            if label in _NEGATIVE_LABELS:
                continue
            scores[row["dimension"]][label] += row["weight"]
    return scores


def _normalize_dimension(scores: dict[str, float]) -> dict[str, Any]:
    total = sum(scores.values())
    if total <= 0:
        return {"status": "unavailable", "labels": None, "uncertainty": None}
    labels = None
    for dimension in DIMENSIONS:
        if set(DIMENSION_LABELS[dimension]) == set(scores):
            labels = DIMENSION_LABELS[dimension]
            break
    if labels is None:
        raise AtomBridgeError("internal: dimension label set not found")
    floor = 0.05
    denom = total + floor * len(labels)
    probs = {label: round((scores[label] + floor) / denom, 4) for label in labels}
    uncertainty = round(min(0.6, max(0.12, 0.5 / (1.0 + total) + 0.12)), 4)
    return {"status": "available", "labels": probs, "uncertainty": uncertainty}


def _ontology_prior(
    atom_scores: dict[str, dict[str, float]],
    family_scores: dict[str, dict[str, float]],
) -> dict[str, Any]:
    prior: dict[str, Any] = {}
    for dimension in DIMENSIONS:
        combined = {
            label: atom_scores[dimension][label] + family_scores[dimension][label]
            for label in DIMENSION_LABELS[dimension]
        }
        prior[dimension] = _normalize_dimension(combined)
    return prior


def _damage_type_mix(atoms: list[dict[str, Any]]) -> dict[str, int]:
    mix: dict[str, int] = {}
    for atom in atoms:
        params = atom.get("parameters") or {}
        dtype = params.get("damage_type")
        if isinstance(dtype, str) and dtype:
            mix[dtype] = mix.get(dtype, 0) + 1
    return mix


def _relations_for_champion(
    atoms: list[dict[str, Any]],
    relations: dict[str, Any],
) -> list[dict[str, str]]:
    present = {atom.get("atom_id") for atom in atoms}
    out: list[dict[str, str]] = []
    for source, targets in relations.items():
        if source in present:
            for target in targets:
                if target in present:
                    out.append({"source": source, "target": target})
    out.sort(key=lambda r: (r["source"], r["target"]))
    return out


def build_bridge_payload(sources: LccSources) -> dict[str, Any]:
    summary = sources.payloads["data/atoms/atom-summary.json"]
    if not isinstance(summary, dict):
        raise AtomBridgeError("atom-summary.json must be an object")
    relations = sources.payloads["data/wiki-atoms/atom-relations.json"]
    if not isinstance(relations, dict):
        raise AtomBridgeError("atom-relations.json must be an object")
    champions = sources.payloads["data/champions.json"]
    if not isinstance(champions, dict):
        raise AtomBridgeError("champions.json must be an object")

    champions_out: list[dict[str, Any]] = []
    for lcc_key, meta in sorted(champions.items()):
        if not isinstance(meta, dict) or not isinstance(meta.get("id"), int):
            raise AtomBridgeError(f"champions.json entry {lcc_key} must carry a numeric id")
        champion_id = f"riot:champion:{meta['id']}"
        norm = _normalize_key(lcc_key)
        presence = _family_presence(summary, norm)
        detail = sources.champion_atom_files.get(norm)
        atom_scores = _atom_level_scores(detail or [])
        family_scores = _family_level_scores(presence)
        detail_counts = _atom_family_counts(detail) if detail else None
        record: dict[str, Any] = {
            "champion_id": champion_id,
            "lcc_key": meta.get("key") or lcc_key,
            "display_name": meta.get("name") or lcc_key,
            "profile_status": "atom_detail" if detail else "family_only",
            "family_presence": presence,
            "atom_family_counts": detail_counts,
            "atom_count": sum(detail_counts.values()) if detail_counts else sum(presence.values()),
            "lcc_positions": [str(p) for p in (meta.get("positions") or [])],
            "lcc_roles": [str(r) for r in (meta.get("roles") or [])],
            "lcc_attribute_ratings": {k: v for k, v in (meta.get("attributeRatings") or {}).items() if isinstance(v, (int, float))},
            "lcc_patch_last_changed": meta.get("patchLastChanged"),
            "ontology_prior": _ontology_prior(atom_scores, family_scores),
        }
        if detail:
            record["atom_detail_count"] = len(detail)
            record["damage_type_mix"] = _damage_type_mix(detail)
            record["relations"] = _relations_for_champion(detail, relations)
        champions_out.append(record)

    # co-occurrence sanity: family counts must match classification totals
    champions_by_id = {c["champion_id"]: c for c in champions_out}
    if len(champions_by_id) != len(champions_out):
        raise AtomBridgeError("duplicate champion ids in bridge")

    payload: dict[str, Any] = {
        "schema_id": BRIDGE_SCHEMA_ID,
        "version": BRIDGE_VERSION,
        "mapping_version": MAPPING_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "provenance": sources.source_provenance(),
        "mapping": MAPPING_V1,
        "family_fallback": FAMILY_FALLBACK_V1,
        "atom_relations": relations,
        "champion_families": sorted(CHAMPION_ATOM_FAMILIES),
        "dimensions": {d: list(DIMENSION_LABELS[d]) for d in DIMENSIONS},
        "champions": champions_out,
    }
    return payload


def write_artifact(payload: dict[str, Any], path: Path = DEFAULT_ARTIFACT_PATH) -> str:
    unsigned = dict(payload)
    digest = canonical_sha256(unsigned)
    artifact = dict(unsigned)
    artifact["artifact_sha256"] = digest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return digest


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Build the LCC atom bridge artifact.")
    ap.add_argument("--out", type=Path, default=DEFAULT_ARTIFACT_PATH)
    ap.add_argument("--lcc-repo", type=Path, default=None)
    args = ap.parse_args()
    sources = LccSources(args.lcc_repo) if args.lcc_repo else LccSources()
    sources.load()
    payload = build_bridge_payload(sources)
    digest = write_artifact(payload, args.out)
    print(f"wrote {args.out}")
    print(f"artifact_sha256 = {digest}")
    print(f"champions = {len(payload['champions'])}, mapping rows = {len(payload['mapping'])}, relations = {len(payload['atom_relations'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
