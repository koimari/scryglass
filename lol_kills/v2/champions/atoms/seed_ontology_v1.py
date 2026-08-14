"""Seed the v2 champion ontology from the LCC atom bridge (v1).

For every champion in the pinned LCC atom bridge artifact, emit a patch-pinned
ontology profile (12 dimensions, per-label probabilities and uncertainties)
so the ontology has full champion coverage. Existing profiles remain intact;
the bridge-derived profile for the current public patch is additive.

Fail-closed rules:
  * champions missing from the bridge are an error (never a silent zero)
  * an ``available`` dimension must carry the exact label set
  * an ``unavailable`` dimension becomes an explicit uniform prior with
    maximum uncertainty (1.0) -- honest ignorance, not fabricated zeros
  * every emitted profile references ``source:lcc-atom-bridge-v1``
  * output is deterministic: stable ordering, canonical JSON
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .consume import AtomBridge, DEFAULT_ARTIFACT_PATH
from ..schema import DIMENSION_LABELS, REQUIRED_DIMENSIONS, ROLES
from ...patch_identity import client_patch as canonical_client_patch, public_patch as canonical_public_patch

SOURCE_ID = "source:lcc-atom-bridge-v1"
PRIOR_PATCH = "26.15"  # compatibility default for older hand-authored callers
SEED_SCHEMA_VERSION = "v2-champion-ontology-1"
SOURCES_SCHEMA_VERSION = "v2-champion-sources-1"
CURATED_CHAMPION_IDS = ("riot:champion:115", "riot:champion:101", "riot:champion:161", "riot:champion:518")

DEFAULT_SEED_PATH = (
    Path(__file__).resolve().parents[4]
    / "data" / "lol" / "v2" / "champions" / "champion-ontology-seed-26.16.json"
)
DEFAULT_SOURCES_PATH = (
    Path(__file__).resolve().parents[4]
    / "data" / "lol" / "v2" / "champions" / "champion-ontology-sources-26.16.json"
)

POSITION_MAP = {
    "TOP": "top",
    "JUNGLE": "jungle",
    "MIDDLE": "mid",
    "BOTTOM": "bot",
    "SUPPORT": "support",
}

CANONICAL_AS_OF = "2026-08-13T23:52:09Z"
SNAPSHOT_ID = "scryglass:v2:champion-ontology:26.16"


class SeedError(ValueError):
    """Raised when ontology seeding cannot proceed fail-closed."""


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _round4(value: float) -> float:
    return round(float(value), 4)


def _bridge_prior_to_dimensions(prior: dict[str, Any]) -> dict[str, Any]:
    """Convert one bridge ontology_prior into the seed dimension map."""
    dimensions: dict[str, Any] = {}
    for dimension in REQUIRED_DIMENSIONS:
        entry = prior.get(dimension) or {}
        status = entry.get("status")
        labels = entry.get("labels")
        uncertainty = entry.get("uncertainty")
        allowed = DIMENSION_LABELS[dimension]
        if status == "available" and isinstance(labels, dict):
            missing = set(allowed) - set(labels)
            if missing:
                raise SeedError(
                    f"bridge dimension {dimension} missing labels: {sorted(missing)}"
                )
            label_values = {label: _round4(float(labels[label])) for label in allowed}
            if not all(0.0 <= v <= 1.0 for v in label_values.values()):
                raise SeedError(f"bridge dimension {dimension} has out-of-range labels")
            if isinstance(uncertainty, (int, float)) and 0.0 <= float(uncertainty) <= 1.0:
                per_label_uncertainty = {label: _round4(float(uncertainty)) for label in allowed}
            else:
                per_label_uncertainty = {label: 1.0 for label in allowed}
        elif status == "unavailable":
            # Honest ignorance: uniform prior, maximum uncertainty.
            uniform = 1.0 / len(allowed)
            label_values = {label: round(uniform, 4) for label in allowed}
            per_label_uncertainty = {label: 1.0 for label in allowed}
        else:
            raise SeedError(
                f"bridge dimension {dimension} has unusable status {status!r}"
            )
        dimensions[dimension] = {
            "labels": label_values,
            "uncertainty": per_label_uncertainty,
        }
    return dimensions


def _role_profile() -> dict[str, Any]:
    return {
        "dimensions": {},
        "residual": {"status": "prior_only", "mean": 0.0, "sigma": 1.0, "observation_count": 0},
        "verified_appearances": {},
        "source_ids": [SOURCE_ID],
    }


def _aliases_for(display_name: str, lcc_key: str, existing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = {item["value"].strip().lower() for item in existing}
    out = [{"value": value} for value in existing]
    for value in (display_name, lcc_key):
        alias = value.strip().lower()
        if alias and alias not in seen:
            seen.add(alias)
            out.append({"value": alias})
    return out


def _roles_from_positions(positions: list[str]) -> list[str]:
    roles: list[str] = []
    for position in positions:
        role = POSITION_MAP.get(position)
        if role is None:
            raise SeedError(f"unknown LCC position: {position}")
        if role not in roles:
            roles.append(role)
    return roles


def build_seed(
    bridge: AtomBridge,
    existing_seed: dict[str, Any],
    *,
    patch: str | None = None,
) -> dict[str, Any]:
    """Build the full-coverage ontology seed (deterministic)."""
    patch = patch or str(bridge.provenance.get("data_patch") or "")
    if not patch or patch == "unknown":
        raise SeedError("bridge provenance has no canonical public data_patch")
    existing_by_id = {row["champion_id"]: row for row in existing_seed["champions"]}
    champions: list[dict[str, Any]] = []

    for champion_id in sorted(bridge.champion_ids()):
        profile = bridge.profile(champion_id)
        if profile is None:
            raise SeedError(f"bridge missing profile for {champion_id}")
        prior = bridge.ontology_prior(champion_id)
        if prior is None:
            raise SeedError(f"bridge missing ontology_prior for {champion_id}")
        display_name = profile.get("display_name") or ""
        lcc_key = profile.get("lcc_key") or ""
        if not display_name:
            raise SeedError(f"bridge missing display_name for {champion_id}")

        positions = profile.get("lcc_positions") or []
        bridge_roles = _roles_from_positions(positions)
        if not bridge_roles:
            raise SeedError(f"bridge has no legal roles for {champion_id}")

        existing = existing_by_id.get(champion_id)
        if existing is not None:
            # Keep the hand-authored entry verbatim; add the current profile
            # only for roles that are legal in the authored entry AND present
            # in the bridge positions.
            curated = json.loads(_canonical(existing))
            authored_legal = set(existing.get("role_legalities") or [])
            seed_roles = [r for r in bridge_roles if r in authored_legal]
            if seed_roles:
                curated["patch_profiles"][patch] = {
                    "role_profiles": {
                        role: _bridge_role_profile(prior, role) for role in seed_roles
                    }
                }
            champions.append(curated)
            continue

        role_profiles = {
            role: _bridge_role_profile(prior, role) for role in bridge_roles
        }
        champions.append(
            {
                "champion_id": champion_id,
                "display_name": display_name,
                "aliases": _aliases_for(display_name, lcc_key, []),
                "role_legalities": bridge_roles,
                "patch_profiles": {
                    patch: {"role_profiles": role_profiles},
                },
            }
        )

    champions.sort(key=lambda row: row["champion_id"])
    return {
        "schema_version": SEED_SCHEMA_VERSION,
        "snapshot_id": SNAPSHOT_ID,
        "as_of": CANONICAL_AS_OF,
        "champions": champions,
    }


def _bridge_role_profile(prior: dict[str, Any], role: str) -> dict[str, Any]:
    if role not in ROLES:
        raise SeedError(f"invalid seed role: {role}")
    profile = _role_profile()
    profile["dimensions"] = _bridge_prior_to_dimensions(prior)
    return profile


def build_sources(
    existing_sources: dict[str, Any], *, patch: str | None = None
) -> dict[str, Any]:
    """Return sources with explicit public and client patch provenance.

    Historical source rows may have been written with the public label in a
    client-data URL. Rebuild those URLs from the canonical patch pair before
    adding the current 26.16 bridge and exact client packet rows.
    """
    rows = list(existing_sources.get("sources") or [])
    if patch is None:
        current = next((row for row in rows if row.get("source_id") == SOURCE_ID), None)
        patch = str((current or {}).get("patch") or PRIOR_PATCH)
    # Idempotent: a re-run replaces the canonical row instead of failing.
    normalized_rows: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        source_patch = row.get("patch") or row.get("public_patch") or row.get("patch_id")
        if row.get("kind") == "riot_datadragon" and source_patch:
            try:
                public = canonical_public_patch(source_patch)
                client = canonical_client_patch(source_patch)
            except (TypeError, ValueError):
                public = client = None
            if public and client:
                row["patch"] = public
                row["public_patch"] = public
                row["client_patch"] = client
                row["patch_id"] = public
                if isinstance(row.get("url"), str) and "/data/" in row["url"]:
                    _, suffix = row["url"].split("/data/", 1)
                    row["url"] = f"https://ddragon.leagueoflegends.com/cdn/{client}.1/data/{suffix}"
        normalized_rows.append(row)
    rows = [row for row in normalized_rows if row.get("source_id") not in {SOURCE_ID, "source:cdragon-26.16", "source:riot-dd-26.16"}]
    current_public = canonical_public_patch(patch)
    current_client = canonical_client_patch(patch)
    rows.append(
        {
            "source_id": SOURCE_ID,
            "patch": current_public,
            "public_patch": current_public,
            "client_patch": current_client,
            "kind": "atom_bridge",
            "locator_kind": "repository_path",
            "locator": "data/lol/v2/champions/lcc-atom-bridge-v1.json",
            "publication_decision": "private_pending_review",
            "reviewed_by": "l3_atom_bridge",
            "reviewed_at": CANONICAL_AS_OF,
            "notes": (
                "Mechanistic champion atoms from the League Combat Calculator "
                f"atom cache (public patch {current_public}, client patch {current_client}), mapped through "
                "scryglass.lcc-atom-bridge.v1. Automated prior; pending review."
            ),
        }
    )
    rows.extend(
        [
            {
                "source_id": "source:cdragon-26.16",
                "patch": current_public,
                "public_patch": current_public,
                "client_patch": current_client,
                "source_version": "16.16.1",
                "kind": "communitydragon",
                "locator_kind": "receipt",
                "locator": "data/lol/v2/champions/lcc-atom-refresh-26.16-receipt.json",
                "url": "https://raw.communitydragon.org/16.16/",
                "publication_decision": "private_pending_review",
                "reviewed_by": "l3_atom_bridge",
                "reviewed_at": CANONICAL_AS_OF,
                "notes": "Exact CommunityDragon 16.16 packet for public patch 26.16; receipt-bound and development-only.",
            },
            {
                "source_id": "source:riot-dd-26.16",
                "patch": current_public,
                "public_patch": current_public,
                "client_patch": current_client,
                "source_version": "16.16.1",
                "kind": "riot_datadragon",
                "url": "https://ddragon.leagueoflegends.com/cdn/16.16.1/data/en_US/champion.json",
                "publication_decision": "private_pending_review",
                "reviewed_by": "l3_authoring",
                "reviewed_at": CANONICAL_AS_OF,
                "patch_id": current_public,
                "notes": "Official champion metadata source version 16.16.1 for public patch 26.16.",
            },
        ]
    )
    rows.sort(key=lambda row: row["source_id"])
    return {
        "schema_version": SOURCES_SCHEMA_VERSION,
        "as_of": CANONICAL_AS_OF,
        "sources": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed the v2 champion ontology from the LCC atom bridge.")
    ap.add_argument("--bridge", type=Path, default=DEFAULT_ARTIFACT_PATH)
    ap.add_argument("--out-seed", type=Path, default=DEFAULT_SEED_PATH)
    ap.add_argument("--out-sources", type=Path, default=DEFAULT_SOURCES_PATH)
    ap.add_argument(
        "--patch",
        help="public patch label; defaults to the bridge provenance data_patch",
    )
    args = ap.parse_args()

    bridge = AtomBridge.load(args.bridge)
    existing_seed = json.loads(args.out_seed.read_text(encoding="utf-8"))
    existing_sources = json.loads(args.out_sources.read_text(encoding="utf-8"))

    patch = args.patch or str(bridge.provenance.get("data_patch") or "")
    if not patch or patch == "unknown":
        raise SeedError("bridge provenance has no canonical public data_patch")
    seed = build_seed(bridge, existing_seed, patch=patch)
    sources = build_sources(existing_sources, patch=patch)

    args.out_seed.write_text(json.dumps(seed, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    args.out_sources.write_text(json.dumps(sources, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out_seed} ({len(seed['champions'])} champions, patch {patch})")
    print(f"wrote {args.out_sources} ({len(sources['sources'])} sources)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
