"""Research-only player and team exposure profiles over the LCC atoms.

This module reads the versioned Scryglass bridge. It does not write to the
League Combat Calculator repository. It keeps static entity exposure separate
from Draft Score and from any public rating or prediction authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .consume import AtomBridge
from .schema import DIMENSION_LABELS

SCHEMA_VERSION = "scryglass:lcc-atom-entity-profiles:v1"
PUBLIC_PATCH = "26.16"
CLIENT_PATCH = "16.16"
DEFAULT_BRIDGE_PATH = (
    Path(__file__).resolve().parents[4]
    / "data"
    / "lol"
    / "v2"
    / "champions"
    / "lcc-atom-bridge-26.16.json"
)
DEFAULT_RECEIPT_PATH = (
    Path(__file__).resolve().parents[4]
    / "data"
    / "lol"
    / "v2"
    / "champions"
    / "lcc-atom-refresh-26.16-receipt.json"
)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class EntityAtomProfileError(ValueError):
    """Raised when atom exposure inputs cannot be verified."""


def _norm(value: Any) -> str:
    return "".join(ch for ch in str(value or "").casefold() if ch.isalnum())


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EntityAtomProfileError(f"{label} must be numeric") from exc
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise EntityAtomProfileError(f"{label} must be finite and at least {minimum}")
    return number


def _source_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    try:
        raw = json.dumps(list(rows), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except (TypeError, ValueError) as exc:
        raise EntityAtomProfileError("source rows must be JSON-compatible") from exc
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _receipt(path: Path, bridge: AtomBridge) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EntityAtomProfileError(f"cannot read atom receipt at {path}") from exc
    if not isinstance(payload, Mapping):
        raise EntityAtomProfileError("atom receipt must be an object")
    bridge_hash = str(payload.get("atom_bridge_artifact_sha256") or "")
    if not _HASH_RE.fullmatch(bridge_hash) or bridge_hash != bridge.artifact_sha256:
        raise EntityAtomProfileError("atom receipt is not bound to the bridge artifact")
    if payload.get("public_patch") != PUBLIC_PATCH or payload.get("client_patch") != CLIENT_PATCH:
        raise EntityAtomProfileError("atom receipt patch identity is not 26.16/16.16")
    receipt_hash = hashlib.sha256(raw).hexdigest()
    return dict(payload), receipt_hash


def _champion_index(bridge: AtomBridge) -> dict[str, str]:
    index: dict[str, str] = {}
    for champion_id in bridge.champion_ids():
        profile = bridge.profile(champion_id) or {}
        values = (champion_id, profile.get("lcc_key"), profile.get("display_name"))
        for value in values:
            key = _norm(value)
            if not key:
                continue
            previous = index.get(key)
            if previous is not None and previous != champion_id:
                raise EntityAtomProfileError(f"ambiguous champion alias in bridge: {value!r}")
            index[key] = champion_id
    return index


def _champion_ref(row: Mapping[str, Any]) -> Any:
    return (
        row.get("champion_id")
        or row.get("champion")
        or row.get("champion_name")
    )


def _entity_name(row: Mapping[str, Any]) -> str:
    for key in ("entity_name", "player_name", "team_name", "player", "team", "name"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    raise EntityAtomProfileError("each source row needs an entity name")


def _entity_type(row: Mapping[str, Any]) -> str:
    value = str(row.get("entity_type") or "").strip().casefold()
    if value not in {"player", "team"}:
        raise EntityAtomProfileError("entity_type must be player or team")
    return value


def _entity_id(entity_type: str, name: str) -> str:
    digest = hashlib.sha256(f"{entity_type}\0{name.casefold()}".encode("utf-8")).hexdigest()
    return f"{entity_type}:{digest[:20]}"


def _family_counts(profile: Mapping[str, Any]) -> Mapping[str, Any]:
    counts = profile.get("atom_family_counts")
    if isinstance(counts, Mapping):
        return counts
    presence = profile.get("family_presence")
    if isinstance(presence, Mapping):
        return {str(key): 1 if bool(value) else 0 for key, value in presence.items()}
    return {}


def _matrix_columns(families: Sequence[str]) -> list[str]:
    columns = [f"family:{family}" for family in families]
    for dimension, labels in DIMENSION_LABELS.items():
        columns.extend(f"ontology:{dimension}:{label}" for label in labels)
    return columns


def _matrix_values(entity: Mapping[str, Any], families: Sequence[str]) -> list[float | None]:
    values: list[float | None] = [
        entity.get("atom_exposure", {}).get(family) for family in families
    ]
    ontology = entity.get("ontology_exposure") or {}
    for dimension, labels in DIMENSION_LABELS.items():
        data = ontology.get(dimension) or {}
        label_values = data.get("labels") if isinstance(data, Mapping) else None
        for label in labels:
            values.append(
                float(label_values[label])
                if isinstance(label_values, Mapping) and label in label_values
                else None
            )
    return values


def build_entity_atom_profiles(
    rows: Sequence[Mapping[str, Any]],
    *,
    bridge: AtomBridge | None = None,
    bridge_path: Path = DEFAULT_BRIDGE_PATH,
    receipt_path: Path = DEFAULT_RECEIPT_PATH,
) -> dict[str, Any]:
    """Aggregate verified champion atom profiles for players and teams.

    Unknown champion names are counted as skipped rows. Missing champion values,
    malformed game counts, and missing bridge receipts fail closed.
    """

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise EntityAtomProfileError("source rows must be a non-empty sequence")
    active_bridge = bridge or AtomBridge.load(bridge_path)
    receipt, receipt_sha256 = _receipt(receipt_path, active_bridge)
    source_rows = [row for row in rows if isinstance(row, Mapping)]
    if len(source_rows) != len(rows):
        raise EntityAtomProfileError("source rows must contain objects")
    source_hash = _source_rows_sha256(source_rows)
    index = _champion_index(active_bridge)
    # AtomBridge deliberately exposes immutable accessors only. Derive the
    # family order from the profiles, then sort it for stable matrices.
    family_names: set[str] = set()
    for champion_id in active_bridge.champion_ids():
        family_names.update(str(key) for key in _family_counts(active_bridge.profile(champion_id) or {}))
    families = tuple(sorted(family_names))
    if not families:
        raise EntityAtomProfileError("bridge has no atom families")

    aggregate: dict[str, dict[str, Any]] = {}
    skipped_unknown = 0
    for row_number, row in enumerate(source_rows, start=1):
        entity_type = _entity_type(row)
        name = _entity_name(row)
        champion_ref = _champion_ref(row)
        if champion_ref in (None, ""):
            raise EntityAtomProfileError(f"source row {row_number} has no champion")
        games = _finite(row.get("games"), f"source row {row_number}.games", minimum=0.0)
        if games <= 0:
            raise EntityAtomProfileError(f"source row {row_number}.games must be positive")
        key = _entity_id(entity_type, name)
        item = aggregate.setdefault(
            key,
            {
                "entity_id": key,
                "entity_type": entity_type,
                "display_name": name,
                "source_rows": 0,
                "contributing_rows": 0,
                "games": 0.0,
                "wins": 0.0,
                "wins_known": True,
                "families": defaultdict(float),
                "ontology": {
                    dimension: {
                        "weights": defaultdict(float),
                        "uncertainty": 0.0,
                        "games": 0.0,
                    }
                    for dimension in DIMENSION_LABELS
                },
                "champions": {},
                "scopes": defaultdict(int),
            },
        )
        item["source_rows"] += 1
        champion_id = index.get(_norm(champion_ref))
        if champion_id is None:
            skipped_unknown += 1
            continue
        item["contributing_rows"] += 1
        item["games"] += games
        wins = row.get("wins")
        if wins in (None, ""):
            item["wins_known"] = False
        else:
            win_count = _finite(wins, f"source row {row_number}.wins", minimum=0.0)
            if win_count > games:
                raise EntityAtomProfileError(f"source row {row_number}.wins exceeds games")
            item["wins"] += win_count
        profile = active_bridge.profile(champion_id)
        if profile is None:
            raise EntityAtomProfileError("bridge index returned a missing champion")
        for family, count in _family_counts(profile).items():
            item["families"][str(family)] += float(count or 0) * games
        for dimension, details in (profile.get("ontology_prior") or {}).items():
            if not isinstance(details, Mapping) or details.get("status") != "available":
                continue
            labels = details.get("labels")
            if not isinstance(labels, Mapping):
                continue
            target = item["ontology"].get(dimension)
            if target is None:
                continue
            target["games"] += games
            target["uncertainty"] += float(details.get("uncertainty") or 0.0) * games
            for label, value in labels.items():
                target["weights"][str(label)] += float(value) * games
        champion_entry = item["champions"].setdefault(
            champion_id,
            {
                "champion_id": champion_id,
                "champion": str(profile.get("display_name") or champion_id),
                "games": 0.0,
            },
        )
        champion_entry["games"] += games
        scope = "/".join(
            str(row.get(key)).strip()
            for key in ("patch", "region", "league", "role")
            if row.get(key) not in (None, "")
        )
        if scope:
            item["scopes"][scope] += 1

    if not aggregate or not any(item["contributing_rows"] for item in aggregate.values()):
        raise EntityAtomProfileError("no source row matched the verified atom bridge")

    entities: list[dict[str, Any]] = []
    for item in aggregate.values():
        if item["contributing_rows"] <= 0:
            continue
        games = float(item["games"])
        atom_exposure = {
            family: round(float(item["families"].get(family, 0.0)) / games, 6)
            for family in families
        }
        ontology_exposure: dict[str, Any] = {}
        for dimension in DIMENSION_LABELS:
            target = item["ontology"][dimension]
            coverage_games = float(target["games"])
            if coverage_games <= 0:
                ontology_exposure[dimension] = {
                    "status": "unavailable",
                    "labels": None,
                    "uncertainty": None,
                    "coverage_games": 0,
                }
                continue
            ontology_exposure[dimension] = {
                "status": "available",
                "labels": {
                    label: round(float(target["weights"].get(label, 0.0)) / coverage_games, 6)
                    for label in DIMENSION_LABELS[dimension]
                },
                "uncertainty": round(float(target["uncertainty"]) / coverage_games, 6),
                "coverage_games": round(coverage_games, 6),
            }
        champion_rows = sorted(
            item["champions"].values(),
            key=lambda value: (-float(value["games"]), value["champion"].casefold()),
        )
        entity = {
            "entity_id": item["entity_id"],
            "entity_type": item["entity_type"],
            "display_name": item["display_name"],
            "source_rows": item["source_rows"],
            "contributing_rows": item["contributing_rows"],
            "coverage": round(item["contributing_rows"] / item["source_rows"], 6),
            "games": round(games, 6),
            "wins": round(float(item["wins"]), 6) if item["wins_known"] else None,
            "atom_exposure": atom_exposure,
            "ontology_exposure": ontology_exposure,
            "champions": [
                {**row, "games": round(float(row["games"]), 6)} for row in champion_rows
            ],
            "scopes": dict(sorted(item["scopes"].items())),
        }
        entities.append(entity)
    entities.sort(key=lambda value: (value["entity_type"], value["display_name"].casefold(), value["entity_id"]))
    columns = _matrix_columns(families)
    matrix = [
        {
            "entity_id": entity["entity_id"],
            "entity_type": entity["entity_type"],
            "display_name": entity["display_name"],
            "values": _matrix_values(entity, families),
        }
        for entity in entities
    ]
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "authority": "development_only",
        "public_patch": PUBLIC_PATCH,
        "client_patch": CLIENT_PATCH,
        "bridge_sha256": active_bridge.artifact_sha256,
        "receipt_sha256": receipt_sha256,
        "source_rows_sha256": source_hash,
        "source_rows": len(source_rows),
        "contributing_rows": sum(int(entity["contributing_rows"]) for entity in entities),
        "skipped_unknown_champions": skipped_unknown,
        "feature_columns": columns,
        "entities": entities,
        "entity_matrix": matrix,
        "claim_ceiling": {
            "prediction": False,
            "causal_effect": False,
            "public_rating": False,
            "public_draft_authority": False,
        },
    }
    result["artifact_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return result


def public_projection() -> dict[str, Any]:
    """Return the public-safe placeholder for this research-only artifact."""

    return {
        "schema_version": SCHEMA_VERSION,
        "authority": "unavailable",
        "reason": "research_entity_atom_profiles",
        "public_patch": PUBLIC_PATCH,
        "client_patch": CLIENT_PATCH,
        "entities": None,
        "entity_matrix": None,
    }


__all__ = [
    "CLIENT_PATCH",
    "DEFAULT_BRIDGE_PATH",
    "DEFAULT_RECEIPT_PATH",
    "EntityAtomProfileError",
    "PUBLIC_PATCH",
    "SCHEMA_VERSION",
    "build_entity_atom_profiles",
    "public_projection",
]
