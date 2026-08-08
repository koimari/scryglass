"""Freeze and validate the outcome-labelled terminal-draft development cohort.

The mutable Oracle's Elixir warehouse is useful for ingestion, but it is not a
stable evaluation input.  This module copies the complete rows that have an
outcome-free dependence-cluster assignment into canonical JSONL bytes and
binds those bytes to a manifest.  The resulting cohort remains development
only: its outcomes are already known, source-time availability is unverified,
and the cluster proxy is not authoritative series identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lol_kills.v2.data.common import ROLES, parse_rfc3339, sha256_canonical_object

from .development_evaluation import DraftRow, load_snapshot as load_warehouse_snapshot


SCHEMA_VERSION = "scryglass:draft-terminal-development-snapshot:v2"
DEFAULT_PAYLOAD = Path(
    "data/lol/v2/models/draft-terminal/development-cohort-clustered-v2.jsonl"
)
DEFAULT_MANIFEST = Path(
    "data/lol/v2/models/draft-terminal/development-cohort-clustered-v2.manifest.json"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ROW_KEYS = {
    "game_id",
    "dependence_cluster_id",
    "event_start",
    "patch",
    "league",
    "team_a",
    "team_b",
    "side_a",
    "side_b",
    "label_a",
}


class DevelopmentSnapshotError(ValueError):
    """Raised when frozen development bytes do not satisfy their manifest."""


def _raw_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_line(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def _strict_object(raw: bytes, field: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise DevelopmentSnapshotError(f"{field} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except DevelopmentSnapshotError:
        raise
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise DevelopmentSnapshotError(f"{field} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise DevelopmentSnapshotError(f"{field} must be an object")
    return value


def _atomic_write_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise DevelopmentSnapshotError(f"refusing to overwrite frozen artifact: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise DevelopmentSnapshotError(f"refusing to overwrite frozen artifact: {path}")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _side_mapping(side: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"role": role, "champion": champion} for role, champion in side]


def _row_mapping(row: DraftRow) -> dict[str, Any]:
    return {
        "game_id": row.game_id,
        "dependence_cluster_id": row.dependence_cluster_id,
        "event_start": row.date.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "patch": row.patch,
        "league": row.league,
        "team_a": row.team_a,
        "team_b": row.team_b,
        "side_a": _side_mapping(row.side_a),
        "side_b": _side_mapping(row.side_b),
        "label_a": row.label_a,
    }


def build_development_snapshot(root: Path) -> tuple[bytes, dict[str, Any]]:
    """Build canonical clustered cohort bytes from the current mutable warehouse."""

    rows, source_hashes = load_warehouse_snapshot(root)
    clustered = [
        row
        for row in rows
        if not row.dependence_cluster_id.startswith("unclustered-game:")
    ]
    excluded = len(rows) - len(clustered)
    if len(clustered) < 1000:
        raise DevelopmentSnapshotError(
            "fewer than 1000 complete dependence-clustered development maps are available"
        )
    clustered.sort(key=lambda row: (row.date, row.dependence_cluster_id, row.game_id))
    payload = b"".join(_canonical_line(_row_mapping(row)) for row in clustered)
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": "draft-terminal-development-cohort-clustered-v2",
        "captured_at": now,
        "status": "frozen_development_only",
        "payload_locator": str(DEFAULT_PAYLOAD),
        "payload_raw_sha256": _raw_sha256(payload),
        "source_snapshot": dict(source_hashes),
        "population": {
            "complete_clustered_rows": len(clustered),
            "dependence_clusters": len(
                {row.dependence_cluster_id for row in clustered}
            ),
            "excluded_without_cluster_assignment": excluded,
            "start": clustered[0].date.astimezone(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "end": clustered[-1].date.astimezone(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "leagues": sorted({row.league for row in clustered}),
            "patches": sorted({row.patch for row in clustered}),
        },
        "cohort_policy": {
            "complete_ten_player_maps_only": True,
            "outcome_free_dependence_cluster_assignment_required": True,
            "unclustered_maps_treated_as_independent": False,
            "authoritative_series_identity": False,
            "participant_identity_available": False,
            "outcomes_in_payload": True,
            "source_time_availability_verified": False,
        },
        "claim_ceiling": {
            "development_diagnostic": True,
            "independent_validation": False,
            "production_probability": False,
            "recommendation": False,
            "betting": False,
        },
    }
    manifest["artifact_sha256"] = sha256_canonical_object(manifest)
    return payload, manifest


def write_development_snapshot(
    root: Path,
    *,
    payload_path: Path = DEFAULT_PAYLOAD,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> tuple[Path, Path]:
    """Atomically create a new frozen cohort and its binding manifest."""

    payload, manifest = build_development_snapshot(root)
    resolved_payload = payload_path if payload_path.is_absolute() else root / payload_path
    resolved_manifest = (
        manifest_path if manifest_path.is_absolute() else root / manifest_path
    )
    if str(payload_path) != manifest["payload_locator"]:
        raise DevelopmentSnapshotError(
            "custom payload path requires a separately versioned manifest schema"
        )
    _atomic_write_new(resolved_payload, payload)
    try:
        _atomic_write_new(
            resolved_manifest,
            (
                json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True)
                + "\n"
            ).encode("ascii"),
        )
    except Exception:
        # Leave payload bytes in place: deleting immutable evidence after a
        # partial write would be less safe than reporting the incomplete pair.
        raise
    return resolved_payload, resolved_manifest


def _parse_side(value: Any, field: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list) or len(value) != len(ROLES):
        raise DevelopmentSnapshotError(f"{field} must contain five role assignments")
    parsed: list[tuple[str, str]] = []
    for expected_role, item in zip(ROLES, value):
        if not isinstance(item, Mapping) or set(item) != {"role", "champion"}:
            raise DevelopmentSnapshotError(f"{field} assignment shape changed")
        role = item.get("role")
        champion = item.get("champion")
        if role != expected_role or not isinstance(champion, str) or not champion:
            raise DevelopmentSnapshotError(f"{field} role order or champion is invalid")
        parsed.append((role, champion))
    if len({champion for _, champion in parsed}) != len(parsed):
        raise DevelopmentSnapshotError(f"{field} contains duplicate champions")
    return tuple(parsed)


def _parse_row(value: Mapping[str, Any], line_number: int) -> DraftRow:
    field = f"development payload line {line_number}"
    if set(value) != _ROW_KEYS:
        raise DevelopmentSnapshotError(f"{field} keys changed")
    for name in (
        "game_id",
        "dependence_cluster_id",
        "patch",
        "league",
        "team_a",
        "team_b",
    ):
        if not isinstance(value.get(name), str) or not value[name]:
            raise DevelopmentSnapshotError(f"{field}.{name} is invalid")
    if value["dependence_cluster_id"].startswith("unclustered-game:"):
        raise DevelopmentSnapshotError(f"{field} admits an unclustered map")
    label = value.get("label_a")
    if isinstance(label, bool) or label not in {0, 1}:
        raise DevelopmentSnapshotError(f"{field}.label_a is invalid")
    try:
        date = parse_rfc3339(value.get("event_start"))
    except (TypeError, ValueError) as exc:
        raise DevelopmentSnapshotError(f"{field}.event_start is invalid") from exc
    side_a = _parse_side(value.get("side_a"), f"{field}.side_a")
    side_b = _parse_side(value.get("side_b"), f"{field}.side_b")
    if len({champion for _, champion in (*side_a, *side_b)}) != 10:
        raise DevelopmentSnapshotError(f"{field} does not contain ten unique champions")
    if value["team_a"] == value["team_b"]:
        raise DevelopmentSnapshotError(f"{field} has identical teams")
    return DraftRow(
        game_id=value["game_id"],
        dependence_cluster_id=value["dependence_cluster_id"],
        date=date,
        patch=value["patch"],
        league=value["league"],
        team_a=value["team_a"],
        team_b=value["team_b"],
        side_a=side_a,
        side_b=side_b,
        label_a=int(label),
    )


def load_development_snapshot(
    root: Path,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> tuple[list[DraftRow], dict[str, Any]]:
    """Validate and load the exact frozen cohort named by its manifest."""

    resolved_manifest = (
        manifest_path if manifest_path.is_absolute() else root / manifest_path
    )
    manifest_raw = resolved_manifest.read_bytes()
    manifest = _strict_object(manifest_raw, "development snapshot manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise DevelopmentSnapshotError("development snapshot schema_version changed")
    if manifest.get("status") != "frozen_development_only":
        raise DevelopmentSnapshotError("development snapshot status changed")
    unsigned = {key: value for key, value in manifest.items() if key != "artifact_sha256"}
    if manifest.get("artifact_sha256") != sha256_canonical_object(unsigned):
        raise DevelopmentSnapshotError("development snapshot manifest hash does not match")
    locator = manifest.get("payload_locator")
    if not isinstance(locator, str) or not locator:
        raise DevelopmentSnapshotError("development snapshot payload locator is missing")
    payload_path = Path(locator)
    if payload_path.is_absolute() or ".." in payload_path.parts:
        raise DevelopmentSnapshotError("development snapshot payload locator is unsafe")
    payload_raw = (root / payload_path).read_bytes()
    if manifest.get("payload_raw_sha256") != _raw_sha256(payload_raw):
        raise DevelopmentSnapshotError("development snapshot payload hash does not match")
    rows: list[DraftRow] = []
    game_ids: set[str] = set()
    previous_key: tuple[datetime, str, str] | None = None
    for line_number, line in enumerate(payload_raw.splitlines(), 1):
        if not line:
            raise DevelopmentSnapshotError(
                f"development payload line {line_number} is blank"
            )
        row = _parse_row(_strict_object(line, f"development payload line {line_number}"), line_number)
        if row.game_id in game_ids:
            raise DevelopmentSnapshotError("development snapshot game_id is duplicated")
        game_ids.add(row.game_id)
        key = (row.date, row.dependence_cluster_id, row.game_id)
        if previous_key is not None and key <= previous_key:
            raise DevelopmentSnapshotError("development snapshot row order is not canonical")
        previous_key = key
        rows.append(row)
    population = manifest.get("population")
    if not isinstance(population, Mapping):
        raise DevelopmentSnapshotError("development snapshot population is missing")
    if len(rows) != population.get("complete_clustered_rows"):
        raise DevelopmentSnapshotError("development snapshot row count does not match")
    if len({row.dependence_cluster_id for row in rows}) != population.get(
        "dependence_clusters"
    ):
        raise DevelopmentSnapshotError("development snapshot cluster count does not match")
    if not rows:
        raise DevelopmentSnapshotError("development snapshot is empty")
    return rows, {
        "development_snapshot_manifest_raw_sha256": _raw_sha256(manifest_raw),
        "development_snapshot_manifest_artifact_sha256": manifest["artifact_sha256"],
        "development_snapshot_payload_raw_sha256": manifest["payload_raw_sha256"],
        "development_snapshot_source": manifest["source_snapshot"],
        "development_snapshot_population": dict(population),
        "availability_status": "frozen_retrospective_development_only",
        "rights_status": "not_revalidated_for_public_serving",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    try:
        payload, manifest = write_development_snapshot(args.root)
    except (OSError, DevelopmentSnapshotError, ValueError) as exc:
        print(str(exc))
        return 2
    print(
        json.dumps(
            {
                "payload": str(payload),
                "manifest": str(manifest),
                "payload_raw_sha256": _raw_sha256(payload.read_bytes()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

