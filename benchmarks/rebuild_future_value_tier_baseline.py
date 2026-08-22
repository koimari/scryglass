"""Rebuild the research Tier baseline on one accepted source census.

The producer stages a closed source freeze in a new output directory. It reads
the supplied source receipt and source parquet files. It calls the pooled Tier
candidate builder with the receipt's accepted game IDs. It writes candidate,
manifest, prospective-evaluation, runtime, and trust-input records under the
same output root.

The output is for research use. The producer does not write public or worker
files. Existing outputs, symlinked inputs, census drift, and changed bytes fail
before the output can be consumed.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Mapping

import pandas as pd

from benchmarks.build_future_value_tier_trust import DEFAULT_TIER_ASSET_LOCATORS
from lol_kills.research.future_value_rating import (
    validate_future_value_source_receipt_payload,
)
from lol_kills.research.future_value_tierlist import AUTHORITY, canonical_json_bytes
from lol_kills.v2.tierlists.accepted_census import canonical_game_ids, identity_sha256
from lol_kills.v2.tierlists.pooled_candidate import build_pooled_candidate


SCHEMA_VERSION = "scryglass:future-value-tier-baseline-rebuild:v1"
BUNDLE_FILE = "tier-trust-inputs.json"
SOURCE_RECEIPT_FILE = "future-value-source-receipt.json"
CANDIDATE_FILE = "baseline/tierlists/champion-elo-candidate-v1.json"
PRODUCTION_MANIFEST_FILE = "baseline/tierlists/production-manifest-v1.json"
PROSPECTIVE_EVALUATION_FILE = "baseline/tierlists/prospective-evaluation-v1.json"
RUNTIME_PLAYER_FILE = "runtime/data/lol/warehouse/parquet/oe_live/oe_player_games.parquet"
RUNTIME_META_FILE = "runtime/data/lol/warehouse/parquet/oe_live/meta.json"
SOURCE_META_FILE = "source/meta.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)

EXPECTED_ACCEPTED_GAME_COUNT = 17758
EXPECTED_ACCEPTED_IDENTITY_SHA256 = (
    "9f4cfc167290a1d1a0111edab39e1e70557777c078835843a2ae11ba67a71ff0"
)
EXPECTED_MODEL_ELIGIBLE_GAME_COUNT = 16562
EXPECTED_MODEL_ELIGIBLE_IDENTITY_SHA256 = (
    "5091d97db411af3b4b9b4cfa0204612b0439252e44752dfca027a171447d98b5"
)

_SOURCE_LABELS = ("maps", "players", "teams", "accepted_census")


class TierBaselineRebuildError(ValueError):
    """The research Tier baseline cannot be rebuilt safely."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise TierBaselineRebuildError(f"cannot read file: {path}") from error
    return digest.hexdigest()


def _require_hash(value: object, label: str) -> str:
    text = str(value or "").lower()
    if SHA256_RE.fullmatch(text) is None:
        raise TierBaselineRebuildError(f"{label} must be a SHA-256 digest")
    return text


def _absolute_path(value: Path | str, label: str, *, directory: bool = False) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute() or ".." in raw.parts:
        raise TierBaselineRebuildError(
            f"{label} must be an absolute path without '..'"
        )
    path = Path(os.path.abspath(os.fspath(raw)))
    try:
        for parent in (path, *path.parents):
            if parent.is_symlink():
                raise TierBaselineRebuildError(
                    f"{label} is a symlink or has a symlink parent"
                )
    except OSError as error:
        raise TierBaselineRebuildError(f"{label} cannot be inspected") from error
    if directory:
        if not path.is_dir():
            raise TierBaselineRebuildError(f"{label} is not a directory: {path}")
    elif not path.is_file():
        raise TierBaselineRebuildError(f"{label} is not a regular file: {path}")
    return path


def _output_root(value: Path | str) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute() or ".." in raw.parts:
        raise TierBaselineRebuildError(
            "output root must be an absolute path without '..'"
        )
    path = Path(os.path.abspath(os.fspath(raw)))
    try:
        for parent in path.parents:
            if parent.is_symlink():
                raise TierBaselineRebuildError(
                    "output root has a symlink parent"
                )
    except OSError as error:
        raise TierBaselineRebuildError("output root cannot be inspected") from error
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_dir():
            raise TierBaselineRebuildError("output root is not a regular directory")
        if any(path.iterdir()):
            raise TierBaselineRebuildError("output root must be empty")
    else:
        path.mkdir(parents=True, exist_ok=False)
    return path


def _relative_path(path: Path, root: Path, label: str) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise TierBaselineRebuildError(f"{label} is outside its root") from error
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise TierBaselineRebuildError(f"{label} has an unsafe locator")
    return relative.as_posix()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TierBaselineRebuildError(f"{label} cannot be read") from error
    if not isinstance(value, dict):
        raise TierBaselineRebuildError(f"{label} must be a JSON object")
    return dict(value)


def _write_json_once(path: Path, value: Mapping[str, Any], label: str) -> str:
    if path.exists() or path.is_symlink():
        raise TierBaselineRebuildError(f"{label} output already exists")
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise TierBaselineRebuildError(f"{label} output has a symlink parent")
    path.parent.mkdir(parents=True, exist_ok=True)
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise TierBaselineRebuildError(f"{label} output has a symlink parent")
    raw = canonical_json_bytes(dict(value)) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    output = dict(value)
    output[field] = hashlib.sha256(
        canonical_json_bytes({key: item for key, item in output.items() if key != field})
    ).hexdigest()
    return output


def _record(path: Path, root: Path, label: str) -> dict[str, Any]:
    safe = _absolute_path(path, label)
    return {
        "locator": _relative_path(safe, root, label),
        "bytes": int(safe.stat().st_size),
        "sha256": sha256_path(safe),
    }


def _copy_once(source: Path, destination: Path, label: str) -> dict[str, Any]:
    source = _absolute_path(source, label)
    if destination.exists() or destination.is_symlink():
        raise TierBaselineRebuildError(f"{label} output already exists")
    if any(parent.is_symlink() for parent in (destination, *destination.parents)):
        raise TierBaselineRebuildError(f"{label} output has a symlink parent")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256_path(source) != sha256_path(destination):
        raise TierBaselineRebuildError(f"{label} copied bytes changed")
    return {
        "bytes": int(destination.stat().st_size),
        "sha256": sha256_path(destination),
    }


def _load_receipt(
    receipt_path: Path,
    *,
    expected_file_sha256: str,
    expected_receipt_sha256: str,
) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    expected_file = _require_hash(
        expected_file_sha256, "expected source receipt file hash"
    )
    if sha256_path(receipt_path) != expected_file:
        raise TierBaselineRebuildError("source receipt file bytes changed")
    receipt = _load_json(receipt_path, "source receipt")
    try:
        accepted_ids, eligible_ids = validate_future_value_source_receipt_payload(
            receipt,
            expected_receipt_sha256=_require_hash(
                expected_receipt_sha256, "expected source receipt hash"
            ),
        )
    except Exception as error:
        raise TierBaselineRebuildError(f"source receipt is not verified: {error}") from error
    return receipt, accepted_ids, eligible_ids


def _receipt_source_path(
    source_root: Path,
    receipt_root: Path,
    receipt: Mapping[str, Any],
    label: str,
) -> Path:
    records = receipt.get("source_files")
    record = records.get(label) if isinstance(records, Mapping) else None
    if not isinstance(record, Mapping):
        raise TierBaselineRebuildError(f"source receipt file binding is missing: {label}")
    locator = record.get("locator")
    absolute = record.get("path")
    if isinstance(locator, str) and locator.strip():
        relative = Path(locator)
        if relative.is_absolute() or ".." in relative.parts:
            raise TierBaselineRebuildError(f"source receipt locator is unsafe: {label}")
        candidate = source_root / relative
        if not candidate.is_file() and label == "accepted_census":
            candidate = receipt_root / relative
    elif isinstance(absolute, str) and absolute.strip():
        candidate = Path(absolute).expanduser()
        try:
            candidate.relative_to(source_root)
        except ValueError as error:
            raise TierBaselineRebuildError(
                f"source receipt file is outside source root: {label}"
            ) from error
    else:
        raise TierBaselineRebuildError(f"source receipt locator is missing: {label}")
    path = _absolute_path(candidate, f"source {label}")
    expected_bytes = record.get("bytes")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
        raise TierBaselineRebuildError(f"source receipt byte count is invalid: {label}")
    if path.stat().st_size != expected_bytes:
        raise TierBaselineRebuildError(f"source {label} bytes changed")
    expected_hash = _require_hash(record.get("sha256"), f"source {label} hash")
    if sha256_path(path) != expected_hash:
        raise TierBaselineRebuildError(f"source {label} bytes changed")
    return path


def _source_identity_ids(path: Path, label: str) -> tuple[str, ...]:
    try:
        columns = pd.read_parquet(path, engine="pyarrow", columns=None).columns
        identity_column = "game_uid" if "game_uid" in columns else "gameid"
        if identity_column not in columns:
            raise TierBaselineRebuildError(f"source {label} has no game identity column")
        frame = pd.read_parquet(path, engine="pyarrow", columns=[identity_column])
    except TierBaselineRebuildError:
        raise
    except Exception as error:
        raise TierBaselineRebuildError(f"source {label} cannot be read") from error
    try:
        return canonical_game_ids(frame[identity_column].astype(str).tolist())
    except Exception as error:
        raise TierBaselineRebuildError(f"source {label} game identity is invalid") from error


def _verify_exact_census(
    source_paths: Mapping[str, Path],
    accepted_ids: tuple[str, ...],
    *,
    expected_count: int | None,
    expected_identity: str | None,
) -> dict[str, Any]:
    expected_set = set(accepted_ids)
    if expected_count is not None and len(accepted_ids) != int(expected_count):
        raise TierBaselineRebuildError(
            f"accepted census count changed: {len(accepted_ids)} != {expected_count}"
        )
    if expected_identity is not None:
        expected_hash = _require_hash(expected_identity, "expected accepted census identity")
        actual_hash = identity_sha256(accepted_ids)
        if actual_hash != expected_hash:
            raise TierBaselineRebuildError("accepted census identity changed")
    source_id_counts: dict[str, int] = {}
    for label in ("maps", "players", "teams"):
        ids = _source_identity_ids(source_paths[label], label)
        missing = sorted(expected_set.difference(ids))
        if missing:
            raise TierBaselineRebuildError(
                f"accepted census is missing from source {label}: "
                f"count={len(missing)} sample={missing[:5]}"
            )
        source_id_counts[label] = len(ids)
    return {
        "accepted_game_count": len(accepted_ids),
        "accepted_identity_sha256": identity_sha256(accepted_ids),
        "accepted_game_ids": list(accepted_ids),
        "source_id_counts": source_id_counts,
    }


def _verify_accepted_census_file(
    path: Path,
    accepted_ids: tuple[str, ...],
) -> None:
    payload = _load_json(path, "accepted census")
    if payload.get("schema_version") != "scryglass:accepted-game-census:v1":
        raise TierBaselineRebuildError("accepted census schema changed")
    raw_ids = payload.get("game_ids")
    if not isinstance(raw_ids, list) or not all(isinstance(value, str) for value in raw_ids):
        raise TierBaselineRebuildError("accepted census game IDs are invalid")
    canonical_ids = canonical_game_ids(raw_ids)
    if list(canonical_ids) != raw_ids:
        raise TierBaselineRebuildError("accepted census game IDs are not canonical and unique")
    if canonical_ids != accepted_ids:
        raise TierBaselineRebuildError("accepted census game IDs changed")
    if payload.get("game_count") != len(accepted_ids):
        raise TierBaselineRebuildError("accepted census count changed")
    if payload.get("source_identity_sha256") != identity_sha256(accepted_ids):
        raise TierBaselineRebuildError("accepted census identity changed")


def _stage_runtime(
    output_root: Path,
    source_players: Path,
    source_meta: Path,
    repository_root: Path,
    *,
    require_assets: bool,
) -> dict[str, Any]:
    runtime_player = output_root / RUNTIME_PLAYER_FILE
    runtime_meta = output_root / RUNTIME_META_FILE
    player_record = _copy_once(source_players, runtime_player, "runtime player source")
    meta_record = _copy_once(source_meta, runtime_meta, "runtime source metadata")
    assets: dict[str, dict[str, Any]] = {}
    for locator in DEFAULT_TIER_ASSET_LOCATORS:
        source = repository_root / locator
        if not source.exists():
            if require_assets:
                raise TierBaselineRebuildError(f"Tier asset is missing: {locator}")
            continue
        source = _absolute_path(source, f"Tier asset {locator}")
        destination = output_root / "runtime" / locator
        record = _copy_once(source, destination, f"Tier asset {locator}")
        assets[locator] = record
    return {
        "root": "runtime",
        "player_source": {"locator": RUNTIME_PLAYER_FILE, **player_record},
        "meta_source": {"locator": RUNTIME_META_FILE, **meta_record},
        "tier_assets": dict(sorted(assets.items())),
    }


def _invoke_candidate_builder(
    builder: Callable[..., Mapping[str, Any]],
    runtime_root: Path,
    accepted_ids: tuple[str, ...],
    source_as_of: str,
) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "source_mode": "oe_only",
        "allowed_game_ids": list(accepted_ids),
        "as_of": pd.Timestamp(source_as_of),
        "expected_live_as_of": pd.Timestamp(source_as_of),
    }
    try:
        signature = inspect.signature(builder)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if not accepts_kwargs:
            arguments = {
                key: value
                for key, value in arguments.items()
                if key in signature.parameters
            }
    except (TypeError, ValueError):
        pass
    try:
        candidate = builder(runtime_root, **arguments)
    except Exception as error:
        raise TierBaselineRebuildError(f"pooled candidate build failed: {error}") from error
    if not isinstance(candidate, Mapping):
        raise TierBaselineRebuildError("pooled candidate builder returned a non-object")
    return dict(candidate)


def _candidate_map_ids(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    model = candidate.get("joint_model")
    map_ids = model.get("map_ids") if isinstance(model, Mapping) else None
    if not isinstance(map_ids, list) or not all(isinstance(value, str) for value in map_ids):
        raise TierBaselineRebuildError("candidate joint model map IDs are missing")
    if len(canonical_game_ids(map_ids)) != len(map_ids):
        raise TierBaselineRebuildError("candidate joint model map IDs are not unique")
    return canonical_game_ids(map_ids)


def _verify_candidate(
    candidate: Mapping[str, Any],
    accepted_ids: tuple[str, ...],
    source_as_of: str,
) -> dict[str, Any]:
    if candidate.get("schema_version") != "scryglass:champion-role-elo-candidate:v2":
        raise TierBaselineRebuildError("candidate schema changed")
    if candidate.get("status") != "development_only":
        raise TierBaselineRebuildError("candidate is not development-only")
    if candidate.get("development_only") is not True:
        raise TierBaselineRebuildError("candidate development authority changed")
    if candidate.get("production_eligible") is not False:
        raise TierBaselineRebuildError("candidate production authority changed")
    if candidate.get("publication_eligible") is not False:
        raise TierBaselineRebuildError("candidate publication authority changed")
    claimed_artifact = _require_hash(candidate.get("artifact_sha256"), "candidate artifact hash")
    unsigned = {key: value for key, value in candidate.items() if key != "artifact_sha256"}
    if hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != claimed_artifact:
        raise TierBaselineRebuildError("candidate self hash changed")
    if candidate.get("as_of") != source_as_of:
        raise TierBaselineRebuildError("candidate cutoff changed")
    source = candidate.get("source")
    if not isinstance(source, Mapping):
        raise TierBaselineRebuildError("candidate source binding is missing")
    expected_count = len(accepted_ids)
    expected_identity = identity_sha256(accepted_ids)
    if source.get("maps_replayed") != expected_count:
        raise TierBaselineRebuildError("candidate map count changed")
    if source.get("maps_used_in_joint_likelihood") != expected_count:
        raise TierBaselineRebuildError("candidate likelihood count changed")
    if source.get("source_identity_sha256") != expected_identity:
        raise TierBaselineRebuildError("candidate source identity changed")
    if source.get("source_latest_replayed") != source_as_of:
        raise TierBaselineRebuildError("candidate source cutoff changed")
    map_ids = _candidate_map_ids(candidate)
    if map_ids != accepted_ids:
        missing = sorted(set(accepted_ids).difference(map_ids))
        extra = sorted(set(map_ids).difference(accepted_ids))
        raise TierBaselineRebuildError(
            f"candidate map identity differs from accepted census "
            f"(missing={len(missing)}, extra={len(extra)})"
        )
    return {
        "artifact_sha256": claimed_artifact,
        "map_count": len(map_ids),
        "game_identity_sha256": identity_sha256(map_ids),
    }


def _authority() -> dict[str, Any]:
    authority = dict(AUTHORITY)
    authority.update(
        {
            "public": False,
            "public_player_rating": False,
            "public_team_rating": False,
        }
    )
    return authority


def _candidate_reference(candidate_locator: str, candidate: Mapping[str, Any], raw_sha256: str) -> dict[str, Any]:
    return {
        "locator": candidate_locator,
        "artifact_sha256": str(candidate["artifact_sha256"]),
        "raw_sha256": raw_sha256,
        "as_of": candidate.get("as_of"),
    }


def _verify_reference(
    payload: Mapping[str, Any],
    candidate: Mapping[str, Any],
    candidate_raw_sha256: str,
    *,
    label: str,
    source: Mapping[str, Any],
) -> None:
    if payload.get("artifact_sha256"):
        claimed = _require_hash(payload.get("artifact_sha256"), f"{label} artifact hash")
        unsigned = {key: value for key, value in payload.items() if key != "artifact_sha256"}
        if hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != claimed:
            raise TierBaselineRebuildError(f"{label} self hash changed")
    reference = payload.get("candidate")
    if not isinstance(reference, Mapping):
        raise TierBaselineRebuildError(f"{label} candidate binding is missing")
    if reference.get("raw_sha256") != candidate_raw_sha256:
        raise TierBaselineRebuildError(f"{label} candidate bytes changed")
    if reference.get("artifact_sha256") != candidate.get("artifact_sha256"):
        raise TierBaselineRebuildError(f"{label} candidate artifact changed")
    source_binding = payload.get("source")
    if isinstance(source_binding, Mapping):
        for field in ("source_game_count", "source_identity_sha256"):
            if field in source_binding and source_binding[field] != source[field]:
                raise TierBaselineRebuildError(f"{label} source binding changed: {field}")


def rebuild_tier_baseline(
    *,
    source_root: Path | str,
    source_receipt_path: Path | str,
    expected_source_receipt_file_sha256: str,
    expected_source_receipt_sha256: str,
    output_root: Path | str,
    repository_root: Path | str | None = None,
    expected_accepted_game_count: int | None = None,
    expected_accepted_identity_sha256: str | None = None,
    expected_model_eligible_game_count: int | None = None,
    expected_model_eligible_identity_sha256: str | None = None,
    candidate_builder: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build one source-bound research Tier baseline and its trust inputs."""

    source_dir = _absolute_path(source_root, "source root", directory=True)
    receipt_path = _absolute_path(source_receipt_path, "source receipt")
    output = _output_root(output_root)
    repository = _absolute_path(
        repository_root or Path(__file__).resolve().parents[1],
        "repository root",
        directory=True,
    )
    receipt, accepted_ids, eligible_ids = _load_receipt(
        receipt_path,
        expected_file_sha256=expected_source_receipt_file_sha256,
        expected_receipt_sha256=expected_source_receipt_sha256,
    )
    census = _verify_exact_census(
        {
            label: _receipt_source_path(
                source_dir,
                receipt_path.parent,
                receipt,
                label,
            )
            for label in _SOURCE_LABELS
            if label != "accepted_census"
        },
        accepted_ids,
        expected_count=expected_accepted_game_count,
        expected_identity=expected_accepted_identity_sha256,
    )
    if expected_model_eligible_game_count is not None and len(eligible_ids) != int(
        expected_model_eligible_game_count
    ):
        raise TierBaselineRebuildError("model-eligible census count changed")
    if expected_model_eligible_identity_sha256 is not None and identity_sha256(
        eligible_ids
    ) != _require_hash(
        expected_model_eligible_identity_sha256,
        "expected model-eligible census identity",
    ):
        raise TierBaselineRebuildError("model-eligible census identity changed")

    source_files = {
        label: _receipt_source_path(source_dir, receipt_path.parent, receipt, label)
        for label in _SOURCE_LABELS
    }
    _verify_accepted_census_file(source_files["accepted_census"], accepted_ids)
    source_meta = _absolute_path(source_dir / "meta.json", "source metadata")
    copied_source: dict[str, dict[str, Any]] = {}
    for label, destination_name in (
        ("maps", "maps.parquet"),
        ("players", "oe_player_games.parquet"),
        ("teams", "oe_team_games.parquet"),
        ("accepted_census", "accepted-census.json"),
    ):
        copied_source[label] = _copy_once(
            source_files[label], output / destination_name, f"source {label}"
        )
    receipt_copy = _copy_once(
        receipt_path,
        output / SOURCE_RECEIPT_FILE,
        "source receipt",
    )
    _copy_once(source_meta, output / SOURCE_META_FILE, "source metadata")

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    builder = candidate_builder or build_pooled_candidate
    runtime = _stage_runtime(
        output,
        source_files["players"],
        source_meta,
        repository,
        require_assets=candidate_builder is None,
    )
    candidate = _invoke_candidate_builder(
        builder,
        output / "runtime",
        accepted_ids,
        str(receipt["source_as_of"]),
    )
    candidate_validation = _verify_candidate(
        candidate,
        accepted_ids,
        str(receipt["source_as_of"]),
    )
    candidate_path = output / CANDIDATE_FILE
    candidate_raw_sha256 = _write_json_once(
        candidate_path,
        candidate,
        "candidate",
    )
    source_binding = {
        "source_as_of": receipt["source_as_of"],
        "source_game_count": len(accepted_ids),
        "source_identity_sha256": identity_sha256(accepted_ids),
        "source_receipt_sha256": receipt["receipt_sha256"],
        "source_receipt_file_sha256": receipt_copy["sha256"],
        "model_eligible_game_count": len(eligible_ids),
        "model_eligible_identity_sha256": identity_sha256(eligible_ids),
        "accepted_game_ids": list(accepted_ids),
        "model_eligible_game_ids": list(eligible_ids),
    }
    candidate_reference = _candidate_reference(
        CANDIDATE_FILE,
        candidate,
        candidate_raw_sha256,
    )
    authority = _authority()
    manifest = _seal(
        {
            "schema_version": "scryglass:tierlist-production-manifest:v1",
            "status": "research_only",
            "authority": authority,
            "production_eligible": False,
            "decision": "hold_research_only",
            "candidate": candidate_reference,
            "source": {
                "source_as_of": receipt["source_as_of"],
                "source_game_count": len(accepted_ids),
                "source_identity_sha256": identity_sha256(accepted_ids),
                "source_receipt_sha256": receipt["receipt_sha256"],
            },
            "claim_ceiling": {
                "production": False,
                "publication": False,
                "recommendation": False,
                "betting": False,
            },
        },
        "artifact_sha256",
    )
    manifest_path = output / PRODUCTION_MANIFEST_FILE
    manifest_raw_sha256 = _write_json_once(
        manifest_path,
        manifest,
        "production manifest",
    )
    prospective = _seal(
        {
            "schema_version": "scryglass:tierlist-forward-evaluation:v1",
            "status": "research_only",
            "authority": authority,
            "production_eligible": False,
            "predictive_authority": False,
            "decision": "research_only",
            "blockers": ["research_only", "prospective_evaluation_not_promoted"],
            "candidate": candidate_reference,
            "source": {
                **source_binding,
                "source_latest": receipt["source_as_of"],
                "source_latest_replayed": receipt["source_as_of"],
            },
            "claim_ceiling": {
                "production": False,
                "publication": False,
                "outcome_calibrated_probability": False,
                "recommendation": False,
                "betting": False,
            },
        },
        "artifact_sha256",
    )
    prospective_path = output / PROSPECTIVE_EVALUATION_FILE
    prospective_raw_sha256 = _write_json_once(
        prospective_path,
        prospective,
        "prospective evaluation",
    )
    _verify_reference(
        manifest,
        candidate,
        candidate_raw_sha256,
        label="production manifest",
        source=source_binding,
    )
    _verify_reference(
        prospective,
        candidate,
        candidate_raw_sha256,
        label="prospective evaluation",
        source=source_binding,
    )
    bundle: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "research_only",
        "authority": authority,
        "source": {
            **source_binding,
            "source_receipt": {
                "locator": SOURCE_RECEIPT_FILE,
                **receipt_copy,
                "receipt_sha256": receipt["receipt_sha256"],
            },
            "source_files": {
                label: {
                    "locator": {
                        "maps": "maps.parquet",
                        "players": "oe_player_games.parquet",
                        "teams": "oe_team_games.parquet",
                        "accepted_census": "accepted-census.json",
                    }[label],
                    **record,
                }
                for label, record in copied_source.items()
            },
            "meta": {
                "locator": SOURCE_META_FILE,
                "bytes": int((output / SOURCE_META_FILE).stat().st_size),
                "sha256": sha256_path(output / SOURCE_META_FILE),
            },
            "census_validation": census,
        },
        "candidate": {
            **candidate_reference,
            "validation": candidate_validation,
        },
        "current_production_manifest": {
            "locator": PRODUCTION_MANIFEST_FILE,
            "bytes": int(manifest_path.stat().st_size),
            "sha256": manifest_raw_sha256,
        },
        "current_prospective_evaluation": {
            "locator": PROSPECTIVE_EVALUATION_FILE,
            "bytes": int(prospective_path.stat().st_size),
            "sha256": prospective_raw_sha256,
        },
        "runtime": runtime,
        "implementation": {
            "candidate_builder": "lol_kills.v2.tierlists.pooled_candidate.build_pooled_candidate",
            "source_mode": "oe_only",
            "allowed_game_ids_exact": True,
            "accepted_game_identity_sha256": identity_sha256(accepted_ids),
            "accepted_game_count": len(accepted_ids),
            "environment": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            },
        },
        "paths": {
            "source_root": ".",
            "source_receipt": SOURCE_RECEIPT_FILE,
            "candidate": CANDIDATE_FILE,
            "production_manifest": PRODUCTION_MANIFEST_FILE,
            "prospective_evaluation": PROSPECTIVE_EVALUATION_FILE,
            "evaluation_root": "evaluations",
        },
    }
    bundle["bundle_sha256"] = hashlib.sha256(
        canonical_json_bytes(bundle)
    ).hexdigest()
    _write_json_once(output / BUNDLE_FILE, bundle, "trust-input bundle")
    return bundle


def load_tier_baseline_bundle(
    path: Path | str,
    *,
    expected_raw_sha256: str | None = None,
) -> dict[str, Any]:
    """Load and verify a previously written closed baseline bundle."""

    bundle_path = _absolute_path(path, "trust-input bundle")
    actual_raw_sha256 = sha256_path(bundle_path)
    if expected_raw_sha256 is not None and actual_raw_sha256 != _require_hash(
        expected_raw_sha256, "expected trust-input bundle hash"
    ):
        raise TierBaselineRebuildError("trust-input bundle bytes changed")
    bundle = _load_json(bundle_path, "trust-input bundle")
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise TierBaselineRebuildError("trust-input bundle schema changed")
    claimed_bundle = _require_hash(bundle.get("bundle_sha256"), "trust-input bundle hash")
    unsigned = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    if hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != claimed_bundle:
        raise TierBaselineRebuildError("trust-input bundle self hash changed")
    if bundle.get("status") != "research_only":
        raise TierBaselineRebuildError("trust-input bundle authority changed")
    if dict(bundle.get("authority") or {}) != _authority():
        raise TierBaselineRebuildError("trust-input bundle authority changed")
    root = bundle_path.parent

    def checked_locator(record: Mapping[str, Any], label: str) -> Path:
        locator = record.get("locator")
        if not isinstance(locator, str) or not locator or Path(locator).is_absolute() or ".." in Path(locator).parts:
            raise TierBaselineRebuildError(f"{label} locator is unsafe")
        target = _absolute_path(root / locator, label)
        expected = _require_hash(record.get("sha256"), f"{label} hash")
        if sha256_path(target) != expected:
            raise TierBaselineRebuildError(f"{label} bytes changed")
        if "bytes" in record and int(record["bytes"]) != target.stat().st_size:
            raise TierBaselineRebuildError(f"{label} byte count changed")
        return target

    source = bundle.get("source")
    if not isinstance(source, Mapping):
        raise TierBaselineRebuildError("trust-input bundle source binding is missing")
    receipt_record = source.get("source_receipt")
    if not isinstance(receipt_record, Mapping):
        raise TierBaselineRebuildError("trust-input bundle receipt binding is missing")
    receipt_path = checked_locator(receipt_record, "source receipt")
    receipt = _load_json(receipt_path, "source receipt")
    try:
        accepted_ids, eligible_ids = validate_future_value_source_receipt_payload(
            receipt,
            expected_receipt_sha256=str(source["source_receipt_sha256"]),
        )
    except Exception as error:
        raise TierBaselineRebuildError(f"trust-input bundle receipt is invalid: {error}") from error
    if tuple(source.get("accepted_game_ids") or ()) != accepted_ids:
        raise TierBaselineRebuildError("trust-input bundle accepted census changed")
    if tuple(source.get("model_eligible_game_ids") or ()) != eligible_ids:
        raise TierBaselineRebuildError("trust-input bundle model-eligible census changed")
    for label in _SOURCE_LABELS:
        records = source.get("source_files")
        record = records.get(label) if isinstance(records, Mapping) else None
        if not isinstance(record, Mapping):
            raise TierBaselineRebuildError(f"trust-input bundle source binding is missing: {label}")
        source_path = checked_locator(record, f"source {label}")
        if label == "accepted_census":
            _verify_accepted_census_file(source_path, accepted_ids)
    candidate_record = bundle.get("candidate")
    if not isinstance(candidate_record, Mapping):
        raise TierBaselineRebuildError("trust-input bundle candidate binding is missing")
    candidate_path = checked_locator(
        {**candidate_record, "sha256": candidate_record.get("raw_sha256")},
        "candidate",
    )
    candidate = _load_json(candidate_path, "candidate")
    candidate_validation = _verify_candidate(
        candidate,
        accepted_ids,
        str(receipt["source_as_of"]),
    )
    if candidate_record.get("artifact_sha256") != candidate.get("artifact_sha256"):
        raise TierBaselineRebuildError("trust-input bundle candidate artifact changed")
    if candidate_validation != candidate_record.get("validation"):
        raise TierBaselineRebuildError("trust-input bundle candidate validation changed")
    source_binding = {
        "source_game_count": len(accepted_ids),
        "source_identity_sha256": identity_sha256(accepted_ids),
    }
    for key in ("current_production_manifest", "current_prospective_evaluation"):
        record = bundle.get(key)
        if not isinstance(record, Mapping):
            raise TierBaselineRebuildError(f"trust-input bundle {key} binding is missing")
        ref_path = checked_locator(record, key)
        payload = _load_json(ref_path, key)
        _verify_reference(
            payload,
            candidate,
            str(candidate_record["raw_sha256"]),
            label=key,
            source=source_binding,
        )
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--source-receipt-file-sha256", required=True)
    parser.add_argument("--source-receipt-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument(
        "--accepted-game-count",
        type=int,
        default=EXPECTED_ACCEPTED_GAME_COUNT,
    )
    parser.add_argument(
        "--accepted-identity-sha256",
        default=EXPECTED_ACCEPTED_IDENTITY_SHA256,
    )
    parser.add_argument(
        "--model-eligible-game-count",
        type=int,
        default=EXPECTED_MODEL_ELIGIBLE_GAME_COUNT,
    )
    parser.add_argument(
        "--model-eligible-identity-sha256",
        default=EXPECTED_MODEL_ELIGIBLE_IDENTITY_SHA256,
    )
    args = parser.parse_args(argv)
    bundle = rebuild_tier_baseline(
        source_root=args.source_root,
        source_receipt_path=args.source_receipt,
        expected_source_receipt_file_sha256=args.source_receipt_file_sha256,
        expected_source_receipt_sha256=args.source_receipt_sha256,
        output_root=args.output_root,
        repository_root=args.repository_root,
        expected_accepted_game_count=args.accepted_game_count,
        expected_accepted_identity_sha256=args.accepted_identity_sha256,
        expected_model_eligible_game_count=args.model_eligible_game_count,
        expected_model_eligible_identity_sha256=args.model_eligible_identity_sha256,
    )
    print(
        json.dumps(
            {
                "status": bundle["status"],
                "output_root": str(Path(args.output_root).expanduser().absolute()),
                "bundle": str(
                    Path(args.output_root).expanduser().absolute() / BUNDLE_FILE
                ),
                "bundle_sha256": bundle["bundle_sha256"],
                "candidate_artifact_sha256": bundle["candidate"]["artifact_sha256"],
                "accepted_game_count": bundle["source"]["source_game_count"],
                "accepted_identity_sha256": bundle["source"]["source_identity_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "BUNDLE_FILE",
    "CANDIDATE_FILE",
    "EXPECTED_ACCEPTED_GAME_COUNT",
    "EXPECTED_ACCEPTED_IDENTITY_SHA256",
    "EXPECTED_MODEL_ELIGIBLE_GAME_COUNT",
    "EXPECTED_MODEL_ELIGIBLE_IDENTITY_SHA256",
    "PRODUCTION_MANIFEST_FILE",
    "PROSPECTIVE_EVALUATION_FILE",
    "SCHEMA_VERSION",
    "TierBaselineRebuildError",
    "load_tier_baseline_bundle",
    "main",
    "rebuild_tier_baseline",
    "sha256_path",
]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TierBaselineRebuildError as error:
        raise SystemExit(f"rebuild_future_value_tier_baseline: {error}") from error
