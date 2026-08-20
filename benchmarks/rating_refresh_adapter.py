#!/usr/bin/env python3
"""Run the production rating refresh against one frozen harness fixture.

The adapter stages the frozen files below a private runtime root. It calls the
same ``refresh_ratings`` entrypoint used by the worker. The generated runtime
artifacts stay below the adapter run directory. The adapter then translates
the production manifest into the harness output contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lol_kills.v2.tierlists.accepted_census import load_census
from lol_kills.v2.tierlists.rating_refresh import OUTPUT, refresh_ratings


OUTPUT_SCHEMA = "scryglass:rating-autoresearch-output:v1"
FREEZE_SCHEMA = "scryglass:rating-autoresearch-freeze:v1"


class AdapterError(ValueError):
    """Raised when the harness environment or production output is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()


def _env_path(name: str) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise AdapterError(f"missing environment variable: {name}")
    return Path(value)


def _env_text(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise AdapterError(f"missing environment variable: {name}")
    return value


def _load_fixture() -> tuple[Path, dict[str, Any], dict[str, Any], str, str]:
    fixture_root = _env_path("SCRYGLASS_RATING_AUTORESEARCH_INPUT_ROOT")
    manifest_path = _env_path("SCRYGLASS_RATING_AUTORESEARCH_FIXTURE_MANIFEST")
    expected_digest = _env_text("SCRYGLASS_RATING_AUTORESEARCH_FIXTURE_MANIFEST_SHA256")
    phase = _env_text("SCRYGLASS_RATING_AUTORESEARCH_PHASE")
    variant = _env_text("SCRYGLASS_RATING_AUTORESEARCH_VARIANT")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdapterError(f"fixture manifest cannot be read: {manifest_path}") from error
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != FREEZE_SCHEMA:
        raise AdapterError("fixture manifest schema is invalid")
    if manifest.get("manifest_sha256") != expected_digest or _manifest_digest(manifest) != expected_digest:
        raise AdapterError("fixture manifest digest does not match the harness binding")
    if manifest.get("phase") != phase:
        raise AdapterError("fixture phase does not match the harness environment")
    census_meta = manifest.get("census")
    if not isinstance(census_meta, Mapping):
        raise AdapterError("fixture census metadata is missing")
    census_path = fixture_root / str(census_meta.get("path") or "")
    accepted = load_census(census_path)
    census_digest = _sha256(census_path)
    if census_digest != census_meta.get("sha256"):
        raise AdapterError("fixture census digest does not match its manifest")
    accepted["sha256"] = census_digest
    return fixture_root, dict(manifest), accepted, phase, variant


def _safe_runtime_relative(value: object, label: str) -> Path:
    relative = Path(str(value or ""))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise AdapterError(f"{label} path is invalid: {relative}")
    return relative


def _stage_runtime(
    fixture_root: Path,
    manifest: Mapping[str, Any],
    accepted: Mapping[str, Any],
    *,
    variant: str,
    runtime_owner: str,
) -> Path:
    runtime_value = _env_path("SCRYGLASS_RATING_AUTORESEARCH_RUNTIME_ROOT")
    runtime_root = runtime_value
    owner_path = runtime_root / ".rating-autoresearch-runtime-owner.json"
    if runtime_root.exists():
        if runtime_root.is_symlink() or not runtime_root.is_dir():
            raise AdapterError(f"runtime destination must be a regular directory: {runtime_root}")
        if owner_path.is_symlink():
            raise AdapterError(f"runtime owner file is a symlink: {owner_path}")
        if owner_path.is_file():
            try:
                owner = json.loads(owner_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise AdapterError(f"runtime owner file cannot be read: {owner_path}") from error
            if (
                not isinstance(owner, Mapping)
                or owner.get("variant") != variant
                or owner.get("runtime_owner") != runtime_owner
            ):
                raise AdapterError(f"runtime owner does not match variant: {runtime_root}")
        elif any(runtime_root.iterdir()):
            raise AdapterError(f"runtime destination has no owner and is not empty: {runtime_root}")
    runtime_root.mkdir(parents=True, exist_ok=True)
    if not owner_path.exists():
        _write_json(
            owner_path,
            {
                "schema_version": "scryglass:rating-autoresearch-runtime-owner:v1",
                "variant": variant,
                "runtime_owner": runtime_owner,
            },
        )

    current_paths: set[str] = set()
    for raw_record in manifest.get("files", []):
        if not isinstance(raw_record, Mapping):
            raise AdapterError("fixture input record is invalid")
        fixture_relative = _safe_runtime_relative(raw_record.get("path"), "fixture input")
        if fixture_relative.parts[0] != "inputs":
            raise AdapterError(f"fixture input path is invalid: {fixture_relative}")
        relative = Path(*fixture_relative.parts[1:])
        current_paths.add(relative.as_posix())
        source = fixture_root / fixture_relative
        destination = runtime_root / relative
        if source.is_symlink() or not source.is_file():
            raise AdapterError(f"fixture input is missing: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if destination.stat().st_size != int(raw_record["bytes"]) or _sha256(destination) != raw_record["sha256"]:
            raise AdapterError(f"staged input changed: {fixture_relative}")
    stage_path = runtime_root / ".rating-autoresearch-stage.json"
    if stage_path.is_symlink():
        raise AdapterError(f"runtime stage file is a symlink: {stage_path}")
    if stage_path.is_file():
        try:
            previous_stage = json.loads(stage_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AdapterError(f"runtime stage file cannot be read: {stage_path}") from error
        previous_paths = previous_stage.get("paths", []) if isinstance(previous_stage, Mapping) else []
        if not isinstance(previous_paths, list):
            raise AdapterError(f"runtime stage paths are invalid: {stage_path}")
        for raw_previous in previous_paths:
            previous = _safe_runtime_relative(raw_previous, "previous staged input")
            if previous.as_posix() in current_paths:
                continue
            stale = runtime_root / previous
            if stale.is_symlink():
                raise AdapterError(f"stale staged input is a symlink: {stale}")
            if stale.is_file():
                stale.unlink()
    _write_json(
        stage_path,
        {
            "schema_version": "scryglass:rating-autoresearch-stage:v1",
            "phase": str(manifest["phase"]),
            "fixture_manifest_sha256": str(manifest["manifest_sha256"]),
            "paths": sorted(current_paths),
        },
    )
    census_destination = runtime_root / "accepted-census.json"
    census_source = fixture_root / _safe_runtime_relative(manifest["census"]["path"], "census")
    shutil.copy2(census_source, census_destination)
    if _sha256(census_destination) != accepted["sha256"]:
        raise AdapterError("staged census changed")
    return runtime_root


def _copy_artifact(
    runtime_root: Path,
    artifact_root: Path,
    locator: str,
    expected_sha256: str | None = None,
) -> tuple[Path, str]:
    relative = _safe_runtime_relative(locator, "production artifact")
    source = runtime_root / relative
    if source.is_symlink() or not source.is_file():
        raise AdapterError(f"production artifact is missing: {source}")
    digest = _sha256(source)
    if expected_sha256 is not None and digest != expected_sha256:
        raise AdapterError(f"production artifact digest changed: {locator}")
    destination = artifact_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if _sha256(destination) != digest:
        raise AdapterError(f"copied production artifact changed: {locator}")
    return destination, digest


def _artifact_descriptor(
    runtime_root: Path,
    artifact_root: Path,
    locator: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    path, digest = _copy_artifact(runtime_root, artifact_root, locator, str(metadata.get("sha256")))
    if digest != metadata.get("sha256"):
        raise AdapterError(f"production artifact digest changed: {locator}")
    if "bytes" in metadata and int(metadata["bytes"]) != path.stat().st_size:
        raise AdapterError(f"production artifact byte count changed: {locator}")
    descriptor: dict[str, Any] = {
        "path": str(path),
        "sha256": digest,
        "bytes": int(path.stat().st_size),
    }
    if "rows" in metadata:
        descriptor["rows"] = int(metadata["rows"])
    return descriptor


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(_canonical_json_bytes(payload))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_from_environment(
    *,
    as_of: str | None = None,
    min_games: int = 20,
    min_series: int = 5,
    previous_as_of: str | None = None,
    momentum_window_games: int = 0,
    momentum_scale: float = 0.0,
) -> dict[str, Any]:
    """Stage one fixture, call ``refresh_ratings``, and emit harness output."""

    fixture_root, manifest, accepted, phase, variant = _load_fixture()
    output_path = _env_path("SCRYGLASS_RATING_AUTORESEARCH_OUTPUT_MANIFEST")
    calls_path = _env_path("SCRYGLASS_RATING_AUTORESEARCH_CALL_COUNTS_PATH")
    runtime_owner = _env_text("SCRYGLASS_RATING_AUTORESEARCH_RUNTIME_OWNER")
    runtime_root = _stage_runtime(
        fixture_root,
        manifest,
        accepted,
        variant=variant,
        runtime_owner=runtime_owner,
    )
    refresh_kwargs: dict[str, Any] = {
        "root": runtime_root,
        "min_games": min_games,
        "min_series": min_series,
        "momentum_window_games": momentum_window_games,
        "momentum_scale": momentum_scale,
        "allowed_game_ids": accepted["game_ids"],
    }
    if as_of:
        refresh_kwargs["as_of"] = pd.Timestamp(as_of)
    if previous_as_of:
        refresh_kwargs["previous_as_of"] = pd.Timestamp(previous_as_of)
    # The accepted census stays in the runtime for inspection. The production
    # function receives its canonical IDs through its public allow-list arg.
    payload = refresh_ratings(**refresh_kwargs)
    if not isinstance(payload, Mapping):
        raise AdapterError("rating refresh returned a non-object payload")
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise AdapterError("rating refresh source metadata is missing")
    if int(source.get("source_game_count", -1)) != int(accepted["game_count"]):
        raise AdapterError("rating refresh count differs from accepted census")
    if source.get("source_identity_sha256") != accepted["source_identity_sha256"]:
        raise AdapterError("rating refresh identity differs from accepted census")
    production_manifest_path = runtime_root / OUTPUT
    if not production_manifest_path.is_file():
        raise AdapterError(f"rating refresh manifest is missing: {production_manifest_path}")
    artifact_root = output_path.parent / f"{output_path.stem}.artifacts"
    if artifact_root.exists():
        if artifact_root.is_symlink() or not artifact_root.is_dir() or any(artifact_root.iterdir()):
            raise AdapterError(f"artifact destination must be a new empty directory: {artifact_root}")
    artifact_root.mkdir(parents=True, exist_ok=True)
    copied_manifest, manifest_digest = _copy_artifact(
        runtime_root,
        artifact_root,
        str(OUTPUT),
    )
    artifacts: dict[str, Any] = {
        "rating_manifest": {
            "path": str(copied_manifest),
            "sha256": manifest_digest,
            "bytes": int(copied_manifest.stat().st_size),
        }
    }
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, Mapping) or not raw_artifacts:
        raise AdapterError("rating refresh artifact metadata is missing")
    for name, metadata in raw_artifacts.items():
        if not isinstance(metadata, Mapping) or not isinstance(metadata.get("locator"), str):
            raise AdapterError(f"rating refresh artifact metadata is invalid: {name}")
        artifacts[str(name)] = _artifact_descriptor(
            runtime_root,
            artifact_root,
            metadata["locator"],
            metadata,
        )
    team = payload.get("team", {})
    player = payload.get("player", {})
    if not isinstance(team, Mapping) or not isinstance(player, Mapping):
        raise AdapterError("rating refresh team or player metadata is invalid")
    output_manifest = {
        "schema_version": OUTPUT_SCHEMA,
        "source": {
            "phase": phase,
            "source_game_count": int(source["source_game_count"]),
            "source_identity_sha256": str(source["source_identity_sha256"]),
            "census_sha256": str(accepted["sha256"]),
            "input_manifest_sha256": str(manifest["manifest_sha256"]),
        },
        "run": {
            "phase": phase,
            "variant": variant,
            "entrypoint": "lol_kills.v2.tierlists.rating_refresh.refresh_ratings",
            "runtime_isolated": True,
            "runtime_persistent": True,
            "runtime_owner": runtime_owner,
            "phase_input_manifest_sha256": str(manifest["manifest_sha256"]),
            "accepted_census_bound": True,
        },
        "outputs": artifacts,
        "semantic": {
            "production_schema_version": payload.get("schema_version"),
            "source_as_of": source.get("as_of"),
            "team_snapshot_rows": team.get("snapshot_rows"),
            "player_snapshot_rows": player.get("snapshot_rows"),
            "team_weekly_rows": team.get("weekly_rows"),
            "player_weekly_rows": player.get("weekly_rows"),
        },
    }
    _write_json(output_path, output_manifest)
    _write_json(
        calls_path,
        {
            "phase": phase,
            "variant": variant,
            "counts": {"load_census": 1, "refresh_ratings": 1},
        },
    )
    return output_manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of")
    parser.add_argument("--min-games", type=int, default=20)
    parser.add_argument("--min-series", type=int, default=5)
    parser.add_argument("--previous-as-of")
    parser.add_argument("--momentum-window-games", type=int, default=0)
    parser.add_argument("--momentum-scale", type=float, default=0.0)
    args = parser.parse_args(argv)
    result = run_from_environment(
        as_of=args.as_of,
        min_games=args.min_games,
        min_series=args.min_series,
        previous_as_of=args.previous_as_of,
        momentum_window_games=args.momentum_window_games,
        momentum_scale=args.momentum_scale,
    )
    print(json.dumps({"phase": result["source"]["phase"], "artifacts": sorted(result["outputs"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
