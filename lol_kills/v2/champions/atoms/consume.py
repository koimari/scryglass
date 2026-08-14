"""Fail-closed reader for the LCC atom bridge artifact.

Consumers (draft interactions, tier lists, team ratings) load the bridge
through this module.  It validates the canonical hash and exposes typed
accessors that return ``None``/``unavailable`` instead of fabricating zeros
when data is missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .schema import (
    BRIDGE_SCHEMA_ID,
    BRIDGE_VERSION,
    AtomBridgeError,
    canonical_sha256,
    require_hash,
    require_object,
    require_string,
)

DEFAULT_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[4]
    / "data"
    / "lol"
    / "v2"
    / "champions"
    / "lcc-atom-bridge-26.16.json"
)


class AtomBridge:
    """Validated, immutable access to the bridge artifact."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        require_object(payload, "bridge artifact")
        if payload.get("schema_id") != BRIDGE_SCHEMA_ID:
            raise AtomBridgeError(f"schema_id must be {BRIDGE_SCHEMA_ID}")
        if payload.get("version") != BRIDGE_VERSION:
            raise AtomBridgeError(f"version must be {BRIDGE_VERSION}")
        submitted = require_hash(payload.get("artifact_sha256"), "artifact_sha256")
        unsigned = dict(payload)
        unsigned.pop("artifact_sha256", None)
        if canonical_sha256(unsigned) != submitted:
            raise AtomBridgeError("artifact_sha256 does not match canonical payload")
        self._payload = dict(payload)
        self._champions = {c["champion_id"]: c for c in payload["champions"]}
        self._mapping = {row["atom_id"]: row for row in payload["mapping"]}
        self._relations = payload.get("atom_relations") or {}

    @classmethod
    def load(cls, path: Path = DEFAULT_ARTIFACT_PATH) -> "AtomBridge":
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise AtomBridgeError(f"cannot read bridge artifact at {path}: {exc}") from exc
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise AtomBridgeError(f"invalid bridge artifact JSON at {path}: {exc}") from exc
        return cls(payload)

    @property
    def artifact_sha256(self) -> str:
        return require_hash(self._payload.get("artifact_sha256"), "artifact_sha256")

    @property
    def generated_at(self) -> str:
        return require_string(self._payload.get("generated_at"), "generated_at")

    @property
    def provenance(self) -> dict[str, Any]:
        return require_object(self._payload.get("provenance"), "provenance")

    @property
    def relations(self) -> Mapping[str, Any]:
        return self._relations

    def champion_ids(self) -> list[str]:
        return sorted(self._champions)

    def profile(self, champion_id: str) -> dict[str, Any] | None:
        return self._champions.get(champion_id)

    def family_presence(self, champion_id: str) -> dict[str, bool] | None:
        profile = self.profile(champion_id)
        if profile is None:
            return None
        presence = profile.get("family_presence")
        return presence if isinstance(presence, Mapping) else None

    def atom_family_counts(self, champion_id: str) -> dict[str, int] | None:
        profile = self.profile(champion_id)
        if profile is None:
            return None
        counts = profile.get("atom_family_counts")
        return counts if isinstance(counts, Mapping) else None

    def ontology_prior(self, champion_id: str) -> dict[str, Any] | None:
        profile = self.profile(champion_id)
        if profile is None:
            return None
        prior = profile.get("ontology_prior")
        return prior if isinstance(prior, Mapping) else None

    def mapping_for_atom(self, atom_id: str) -> list[dict[str, Any]]:
        return [row for row in self._payload["mapping"] if row["atom_id"] == atom_id]
