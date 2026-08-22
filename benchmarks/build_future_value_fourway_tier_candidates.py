"""Build four source-bound research Tier candidates from offset ledgers.

The adapter consumes the closed four-way Tier shadow and the deduplicated
Tier-baseline trust bundle.  It reuses the pooled candidate builder with the
same model-eligible game IDs for every variant.  It never changes the input
baseline and never grants public authority.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from benchmarks.rebuild_future_value_tier_baseline import (
    EXPECTED_ACCEPTED_GAME_COUNT,
    EXPECTED_ACCEPTED_IDENTITY_SHA256,
    EXPECTED_MODEL_ELIGIBLE_GAME_COUNT,
    EXPECTED_MODEL_ELIGIBLE_IDENTITY_SHA256,
    RUNTIME_PLAYER_FILE,
    TierBaselineRebuildError,
    _stage_runtime,
    load_tier_baseline_bundle,
)
from lol_kills.research.future_value_tier_shadow import (
    TierShadowError,
    load_tier_offset_ledger,
    sha256_path,
    verify_target_parity,
)
from lol_kills.research.future_value_tierlist import (
    VARIANTS,
    canonical_json_bytes,
    validate_candidate,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256
from lol_kills.v2.tierlists.champion_elo import SOURCE_LOCATOR
from lol_kills.v2.tierlists.pooled_candidate import build_pooled_candidate


SCHEMA_VERSION = "scryglass:future-value-fourway-tier-candidates:v1"
MANIFEST_FILE = "fourway-tier-candidates-manifest.json"
VARIANT_ORDER = tuple(str(value) for value in VARIANTS)
SHA256_LENGTH = 64

AUTHORITY: dict[str, bool] = {
    "research_only": True,
    "public_player_rating": False,
    "public_team_rating": False,
    "public_tierlist": False,
    "public_draft_score": False,
    "public_probability": False,
    "promotion": False,
    "deployment": False,
    "merge": False,
    "odds": False,
    "expected_value": False,
    "recommendation": False,
    "betting": False,
}


class FourwayTierCandidateError(ValueError):
    """The four-way Tier candidate adapter failed closed."""


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_hash(value: object, label: str) -> str:
    text = str(value or "").lower()
    if len(text) != SHA256_LENGTH or any(character not in "0123456789abcdef" for character in text):
        raise FourwayTierCandidateError(f"{label} must be a SHA-256 digest")
    return text


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FourwayTierCandidateError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FourwayTierCandidateError(f"{label} cannot be read: {path}") from error
    if not isinstance(value, dict):
        raise FourwayTierCandidateError(f"{label} must be an object")
    return value


def _self_hash(payload: Mapping[str, Any], field: str, label: str) -> str:
    claimed = _require_hash(payload.get(field), f"{label} {field}")
    unsigned = dict(payload)
    unsigned.pop(field, None)
    if _hash_bytes(canonical_json_bytes(unsigned)) != claimed:
        raise FourwayTierCandidateError(f"{label} self hash changed")
    return claimed


def _authority_closed(value: object, label: str) -> None:
    if not isinstance(value, Mapping) or value.get("research_only") is not True:
        raise FourwayTierCandidateError(f"{label} does not declare research-only authority")
    enabled = sorted(
        str(key) for key, flag in value.items() if key != "research_only" and flag is True
    )
    if enabled:
        raise FourwayTierCandidateError(
            f"{label} grants authority: {', '.join(enabled)}"
        )


def _safe_output_root(path: Path | str) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute() or ".." in value.parts:
        raise FourwayTierCandidateError("output root must be absolute and contain no '..'")
    if any(parent.is_symlink() for parent in (value, *value.parents)):
        raise FourwayTierCandidateError("output root has a symlink component")
    if value.exists():
        if value.is_symlink() or not value.is_dir() or any(value.iterdir()):
            raise FourwayTierCandidateError("output root must be an empty directory")
    else:
        value.mkdir(parents=True, exist_ok=False)
    return value.resolve()


def _safe_path(path: Path, label: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise FourwayTierCandidateError(f"{label} path is unsafe")
    if any(parent.is_symlink() for parent in (candidate, *candidate.parents)):
        raise FourwayTierCandidateError(f"{label} path has a symlink component")
    if not candidate.is_file() or candidate.is_symlink():
        raise FourwayTierCandidateError(f"{label} is missing or unsafe: {candidate}")
    return candidate


def _file_record(path: Path, label: str) -> dict[str, Any]:
    path = _safe_path(path, label)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_path(path)}


def _verify_file_record(
    record: object,
    *,
    base: Path,
    label: str,
    require_bytes: bool = True,
) -> Path:
    if not isinstance(record, Mapping):
        raise FourwayTierCandidateError(f"{label} file binding is missing")
    raw_path = record.get("path")
    if isinstance(raw_path, str) and raw_path.strip():
        candidate = Path(raw_path).expanduser()
    else:
        locator = record.get("locator")
        if not isinstance(locator, str) or not locator.strip():
            raise FourwayTierCandidateError(f"{label} file locator is missing")
        relative = Path(locator)
        if relative.is_absolute() or ".." in relative.parts:
            raise FourwayTierCandidateError(f"{label} file locator is unsafe")
        candidate = base / relative
    path = _safe_path(candidate, label)
    declared_bytes = record.get("bytes")
    if declared_bytes is None and not require_bytes:
        declared_bytes = path.stat().st_size
    if isinstance(declared_bytes, bool) or not isinstance(declared_bytes, int):
        raise FourwayTierCandidateError(f"{label} byte count is invalid")
    if path.stat().st_size != declared_bytes:
        raise FourwayTierCandidateError(f"{label} bytes changed")
    if sha256_path(path) != _require_hash(record.get("sha256"), f"{label} hash"):
        raise FourwayTierCandidateError(f"{label} bytes changed")
    return path


def _bundle_file(
    bundle_root: Path,
    record: object,
    label: str,
    *,
    require_bytes: bool = True,
) -> Path:
    if not isinstance(record, Mapping):
        raise FourwayTierCandidateError(f"Tier baseline {label} binding is missing")
    locator = record.get("locator")
    if not isinstance(locator, str) or not locator.strip():
        raise FourwayTierCandidateError(f"Tier baseline {label} locator is missing")
    relative = Path(locator)
    if relative.is_absolute() or ".." in relative.parts:
        raise FourwayTierCandidateError(f"Tier baseline {label} locator is unsafe")
    return _verify_file_record(
        record,
        base=bundle_root,
        label=f"Tier baseline {label}",
        require_bytes=require_bytes,
    )


def _source_from_bundle(
    bundle_path: Path,
    bundle: Mapping[str, Any],
    *,
    expected_accepted_game_count: int,
    expected_accepted_identity_sha256: str,
    expected_model_eligible_game_count: int,
    expected_model_eligible_identity_sha256: str,
) -> tuple[dict[str, Any], Path, Path, Path, Path, Path, dict[str, Any]]:
    source = bundle.get("source")
    if not isinstance(source, Mapping):
        raise FourwayTierCandidateError("Tier baseline source binding is missing")
    accepted_ids = tuple(str(value) for value in source.get("accepted_game_ids", ()))
    eligible_ids = tuple(str(value) for value in source.get("model_eligible_game_ids", ()))
    if (
        len(accepted_ids) != expected_accepted_game_count
        or identity_sha256(accepted_ids) != expected_accepted_identity_sha256
        or len(eligible_ids) != expected_model_eligible_game_count
        or identity_sha256(eligible_ids) != expected_model_eligible_identity_sha256
        or tuple(sorted(accepted_ids)) != accepted_ids
        or tuple(sorted(eligible_ids)) != eligible_ids
        or len(set(accepted_ids)) != len(accepted_ids)
        or len(set(eligible_ids)) != len(eligible_ids)
        or not set(eligible_ids).issubset(set(accepted_ids))
    ):
        raise FourwayTierCandidateError("Tier baseline census identity changed")
    source_root = bundle_path.parent.resolve()
    source_files = source.get("source_files")
    if not isinstance(source_files, Mapping):
        raise FourwayTierCandidateError("Tier baseline source files are missing")
    maps_path = _bundle_file(source_root, source_files.get("maps"), "maps")
    players_path = _bundle_file(source_root, source_files.get("players"), "players")
    teams_path = _bundle_file(source_root, source_files.get("teams"), "teams")
    census_path = _bundle_file(source_root, source_files.get("accepted_census"), "accepted census")
    meta_path = _safe_path(source_root / "source/meta.json", "Tier baseline source metadata")
    source_binding = {
        "source_as_of": source.get("source_as_of"),
        "source_game_count": len(accepted_ids),
        "source_identity_sha256": identity_sha256(accepted_ids),
        "source_receipt_sha256": source.get("source_receipt_sha256"),
        "source_receipt_file_sha256": (
            source.get("source_receipt", {}).get("sha256")
            if isinstance(source.get("source_receipt"), Mapping)
            else None
        ),
        "model_eligible_game_count": len(eligible_ids),
        "model_eligible_identity_sha256": identity_sha256(eligible_ids),
        "accepted_game_ids": list(accepted_ids),
        "model_eligible_game_ids": list(eligible_ids),
    }
    expected_source_values = {
        "source_game_count": len(accepted_ids),
        "source_identity_sha256": identity_sha256(accepted_ids),
        "model_eligible_game_count": len(eligible_ids),
        "model_eligible_identity_sha256": identity_sha256(eligible_ids),
    }
    if any(source.get(field) != value for field, value in expected_source_values.items()):
        raise FourwayTierCandidateError("Tier baseline source census binding changed")
    for field in (
        "source_as_of",
        "source_receipt_sha256",
        "source_receipt_file_sha256",
    ):
        if not isinstance(source_binding[field], str) or not source_binding[field]:
            raise FourwayTierCandidateError(f"Tier baseline source binding is incomplete: {field}")
    _require_hash(source_binding["source_receipt_sha256"], "Tier baseline source receipt hash")
    _require_hash(
        source_binding["source_receipt_file_sha256"],
        "Tier baseline source receipt file hash",
    )
    receipt_record = source.get("source_receipt")
    receipt_path = _bundle_file(source_root, receipt_record, "source receipt")
    receipt = _load_json(receipt_path, "Tier baseline source receipt")
    receipt_values = {
        "source_as_of": source_binding["source_as_of"],
        "source_game_count": len(accepted_ids),
        "source_identity_sha256": identity_sha256(accepted_ids),
        "model_eligible_game_count": len(eligible_ids),
        "model_eligible_identity_sha256": identity_sha256(eligible_ids),
    }
    if any(receipt.get(field) != value for field, value in receipt_values.items()):
        raise FourwayTierCandidateError("Tier baseline source receipt census binding changed")
    if receipt.get("receipt_sha256") != source_binding["source_receipt_sha256"]:
        raise FourwayTierCandidateError("Tier baseline source receipt hash changed")
    receipt_record_value = receipt_record.get("receipt_sha256") if isinstance(receipt_record, Mapping) else None
    if receipt_record_value != receipt.get("receipt_sha256"):
        raise FourwayTierCandidateError("Tier baseline source receipt record changed")
    if sha256_path(receipt_path) != source_binding["source_receipt_file_sha256"]:
        raise FourwayTierCandidateError("Tier baseline source receipt file changed")
    receipt_ids = tuple(str(value) for value in receipt.get("model_eligible_game_ids", ()))
    if receipt_ids != eligible_ids:
        raise FourwayTierCandidateError("Tier baseline source receipt eligible census changed")
    if tuple(str(value) for value in receipt.get("accepted_game_ids", ())) != accepted_ids:
        raise FourwayTierCandidateError("Tier baseline source receipt accepted census changed")
    return (
        source_binding,
        maps_path,
        players_path,
        teams_path,
        meta_path,
        receipt_path,
        {
            "accepted_game_ids": list(accepted_ids),
            "model_eligible_game_ids": list(eligible_ids),
            "maps": _file_record(maps_path, "Tier baseline maps"),
            "players": _file_record(players_path, "Tier baseline players"),
            "teams": _file_record(teams_path, "Tier baseline teams"),
            "accepted_census": _file_record(census_path, "Tier baseline accepted census"),
            "meta": _file_record(meta_path, "Tier baseline source metadata"),
            "receipt": _file_record(receipt_path, "Tier baseline source receipt"),
        },
    )


def _verify_shadow_manifest(
    path: Path,
    *,
    expected_sha256: str,
    source: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = _safe_path(path, "fourway Tier shadow manifest")
    actual = sha256_path(path)
    if actual != _require_hash(expected_sha256, "expected fourway Tier shadow manifest hash"):
        raise FourwayTierCandidateError("fourway Tier shadow manifest bytes changed")
    payload = _load_json(path, "fourway Tier shadow manifest")
    if payload.get("schema_version") != "scryglass:future-value-tier-shadow-fourway:v1":
        raise FourwayTierCandidateError("fourway Tier shadow manifest schema changed")
    if payload.get("status") != "research_only":
        raise FourwayTierCandidateError("fourway Tier shadow manifest status changed")
    if dict(payload.get("authority") or {}) != AUTHORITY:
        raise FourwayTierCandidateError("fourway Tier shadow manifest authority changed")
    _self_hash(payload, "manifest_sha256", "fourway Tier shadow manifest")
    manifest_source = payload.get("source")
    if not isinstance(manifest_source, Mapping):
        raise FourwayTierCandidateError("fourway Tier shadow source binding is missing")
    for field in (
        "source_as_of",
        "source_game_count",
        "source_identity_sha256",
        "source_receipt_sha256",
        "source_receipt_file_sha256",
        "model_eligible_game_count",
        "model_eligible_identity_sha256",
    ):
        if manifest_source.get(field) != source.get(field):
            raise FourwayTierCandidateError(
                f"fourway Tier shadow source binding changed: {field}"
            )
    variants = payload.get("variants")
    if not isinstance(variants, Mapping) or set(variants) != set(VARIANT_ORDER):
        raise FourwayTierCandidateError("fourway Tier shadow variant set changed")
    records: dict[str, dict[str, Any]] = {}
    for variant in VARIANT_ORDER:
        raw = variants.get(variant)
        if not isinstance(raw, Mapping) or raw.get("variant") != variant:
            raise FourwayTierCandidateError(f"{variant} Tier shadow variant binding changed")
        ledger_path = _verify_file_record(
            raw.get("ledger"), base=path.parent, label=f"{variant} Tier shadow ledger"
        )
        receipt_path = _verify_file_record(
            raw.get("receipt"), base=path.parent, label=f"{variant} Tier shadow receipt"
        )
        if raw.get("game_count") != source.get("model_eligible_game_count"):
            raise FourwayTierCandidateError(f"{variant} Tier shadow game count changed")
        if raw.get("game_identity_sha256") != source.get("model_eligible_identity_sha256"):
            raise FourwayTierCandidateError(f"{variant} Tier shadow game identity changed")
        records[variant] = {
            "ledger": ledger_path,
            "receipt": receipt_path,
            "manifest": dict(raw),
        }
    return payload, records


def _verify_target_rows(
    ledger_path: Path,
    maps_path: Path,
    eligible_ids: tuple[str, ...],
) -> None:
    payload = _load_json(ledger_path, "Tier offset ledger")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise FourwayTierCandidateError("Tier offset ledger rows are missing")
    try:
        verify_target_parity(rows, pd.read_parquet(maps_path), expected_game_ids=eligible_ids)
    except (OSError, ValueError, TierShadowError) as error:
        raise FourwayTierCandidateError(f"Tier offset target parity failed: {error}") from error


def _invoke_builder(
    builder: Callable[..., Mapping[str, Any]],
    runtime_root: Path,
    *,
    eligible_ids: tuple[str, ...],
    source_as_of: str,
    offsets: Mapping[str, float],
    provenance: Mapping[str, Any],
    source_receipt_sha256: str,
) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "source_mode": "oe_only",
        "allowed_game_ids": list(eligible_ids),
        "as_of": pd.Timestamp(source_as_of),
        "expected_live_as_of": pd.Timestamp(source_as_of),
        "pre_map_offset_override": dict(offsets),
        "pre_map_offset_provenance": dict(provenance),
        "expected_pre_map_offset_source_receipt_sha256": source_receipt_sha256,
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
        raise FourwayTierCandidateError("pooled candidate build failed") from error
    if not isinstance(candidate, Mapping):
        raise FourwayTierCandidateError("pooled candidate builder returned a non-object")
    return dict(candidate)


def _verify_candidate_source(
    candidate_source: object,
    runtime: Mapping[str, Any],
    label: str,
) -> None:
    if not isinstance(candidate_source, Mapping):
        raise FourwayTierCandidateError(f"{label} source binding is missing")
    player_source = runtime.get("player_source")
    if not isinstance(player_source, Mapping):
        raise FourwayTierCandidateError(f"{label} staged player source binding is missing")
    if player_source.get("locator") != RUNTIME_PLAYER_FILE:
        raise FourwayTierCandidateError(f"{label} staged player source locator changed")
    if candidate_source.get("locator") != SOURCE_LOCATOR:
        raise FourwayTierCandidateError(f"{label} candidate source locator changed")
    if candidate_source.get("source_files") != [SOURCE_LOCATOR]:
        raise FourwayTierCandidateError(f"{label} candidate source files changed")
    player_sha256 = _require_hash(
        player_source.get("sha256"), f"{label} staged player source hash"
    )
    expected_binding_sha256 = _hash_bytes(
        canonical_json_bytes(
            [{"locator": SOURCE_LOCATOR, "raw_sha256": player_sha256}]
        )
    )
    if candidate_source.get("raw_sha256") != expected_binding_sha256:
        raise FourwayTierCandidateError(f"{label} candidate source bytes changed")


def _write_json_once(path: Path, value: Mapping[str, Any], label: str) -> str:
    if path.exists() or path.is_symlink():
        raise FourwayTierCandidateError(f"{label} output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(dict(value)) + b"\n"
    path.write_bytes(raw)
    return _hash_bytes(raw)


def build_fourway_tier_candidates(
    *,
    shadow_manifest_path: Path | str,
    baseline_bundle_path: Path | str,
    expected_shadow_manifest_sha256: str,
    expected_baseline_bundle_sha256: str,
    output_root: Path | str,
    repository_root: Path | str | None = None,
    expected_accepted_game_count: int = EXPECTED_ACCEPTED_GAME_COUNT,
    expected_accepted_identity_sha256: str = EXPECTED_ACCEPTED_IDENTITY_SHA256,
    expected_model_eligible_game_count: int = EXPECTED_MODEL_ELIGIBLE_GAME_COUNT,
    expected_model_eligible_identity_sha256: str = EXPECTED_MODEL_ELIGIBLE_IDENTITY_SHA256,
    candidate_builder: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build all four candidates over one verified model-eligible universe."""

    shadow_path = _safe_path(Path(shadow_manifest_path).expanduser(), "fourway Tier shadow manifest")
    baseline_path = _safe_path(Path(baseline_bundle_path).expanduser(), "Tier baseline trust bundle")
    try:
        bundle = load_tier_baseline_bundle(
            baseline_path,
            expected_raw_sha256=_require_hash(
                expected_baseline_bundle_sha256,
                "expected Tier baseline trust bundle hash",
            ),
        )
    except (TierBaselineRebuildError, OSError, ValueError) as error:
        raise FourwayTierCandidateError(f"Tier baseline trust bundle failed validation: {error}") from error
    baseline = bundle.get("candidate")
    if not isinstance(baseline, Mapping):
        raise FourwayTierCandidateError("Tier baseline candidate binding is missing")
    baseline_candidate_path = _bundle_file(
        baseline_path.parent,
        {**baseline, "sha256": baseline.get("raw_sha256")},
        "baseline candidate",
        require_bytes=False,
    )
    baseline_raw_sha256 = sha256_path(baseline_candidate_path)
    baseline_artifact_sha256 = str(baseline.get("artifact_sha256") or "")
    (
        source,
        maps_path,
        players_path,
        _teams_path,
        meta_path,
        source_receipt_path,
        source_files,
    ) = _source_from_bundle(
        baseline_path,
        bundle,
        expected_accepted_game_count=int(expected_accepted_game_count),
        expected_accepted_identity_sha256=_require_hash(
            expected_accepted_identity_sha256,
            "expected accepted census identity",
        ),
        expected_model_eligible_game_count=int(expected_model_eligible_game_count),
        expected_model_eligible_identity_sha256=_require_hash(
            expected_model_eligible_identity_sha256,
            "expected model-eligible census identity",
        ),
    )
    _manifest, records = _verify_shadow_manifest(
        shadow_path,
        expected_sha256=expected_shadow_manifest_sha256,
        source=source,
    )
    receipt = _load_json(source_receipt_path, "Tier baseline source receipt")
    eligible_ids = tuple(str(value) for value in source["model_eligible_game_ids"])
    offsets: dict[str, dict[str, float]] = {}
    provenances: dict[str, dict[str, Any]] = {}
    offset_records: dict[str, dict[str, Any]] = {}
    for variant in VARIANT_ORDER:
        record = records[variant]
        try:
            loaded_offsets, provenance = load_tier_offset_ledger(
                record["ledger"],
                record["receipt"],
                source_receipt=receipt,
                variant=variant,
            )
        except (TierShadowError, OSError, ValueError) as error:
            raise FourwayTierCandidateError(
                f"{variant} Tier offset ledger failed validation: {error}"
            ) from error
        if tuple(sorted(loaded_offsets)) != eligible_ids:
            raise FourwayTierCandidateError(f"{variant} Tier offset census changed")
        manifest_provenance = record["manifest"].get("provenance")
        if manifest_provenance is not None and dict(manifest_provenance) != dict(provenance):
            raise FourwayTierCandidateError(f"{variant} Tier offset provenance changed")
        _verify_target_rows(record["ledger"], maps_path, eligible_ids)
        offsets[variant] = loaded_offsets
        provenances[variant] = provenance
        offset_records[variant] = {
            "ledger": _file_record(record["ledger"], f"{variant} Tier offset ledger"),
            "receipt": _file_record(record["receipt"], f"{variant} Tier offset receipt"),
            "offsets_sha256": provenance["offsets_sha256"],
            "game_count": len(eligible_ids),
            "game_identity_sha256": identity_sha256(eligible_ids),
            "receipt_sha256": _load_json(record["receipt"], f"{variant} Tier offset receipt").get(
                "receipt_sha256"
            ),
        }
    if len({tuple(sorted(values)) for values in offsets.values()}) != 1:
        raise FourwayTierCandidateError("four-way Tier offset universes differ")

    output = _safe_output_root(output_root)
    repository = Path(repository_root).expanduser().resolve() if repository_root is not None else Path(__file__).resolve().parents[1]
    builder = candidate_builder or build_pooled_candidate
    try:
        runtime = _stage_runtime(
            output,
            players_path,
            meta_path,
            repository,
            require_assets=candidate_builder is None,
        )
    except (TierBaselineRebuildError, OSError, ValueError) as error:
        raise FourwayTierCandidateError(
            f"Tier candidate runtime staging failed: {error}"
        ) from error
    runtime_root = output / "runtime"
    candidates: dict[str, dict[str, Any]] = {}
    candidate_records: dict[str, dict[str, Any]] = {}
    universe = {
        "game_count": len(eligible_ids),
        "game_identity_sha256": identity_sha256(eligible_ids),
    }
    for variant in VARIANT_ORDER:
        candidate = _invoke_builder(
            builder,
            runtime_root,
            eligible_ids=eligible_ids,
            source_as_of=str(source["source_as_of"]),
            offsets=offsets[variant],
            provenance=provenances[variant],
            source_receipt_sha256=str(source["source_receipt_sha256"]),
        )
        try:
            validation = validate_candidate(
                candidate,
                variant=variant,
                universe=universe,
                expected_source_receipt_sha256=str(source["source_receipt_sha256"]),
                expected_offsets_sha256=str(provenances[variant]["offsets_sha256"]),
                expected_producer=f"future_value_rating:{variant}",
            )
        except Exception as error:
            raise FourwayTierCandidateError(
                f"{variant} pooled candidate failed validation: {error}"
            ) from error
        candidate_source = candidate.get("source")
        if not isinstance(candidate_source, Mapping):
            raise FourwayTierCandidateError(f"{variant} candidate source binding is missing")
        if candidate.get("as_of") != source["source_as_of"]:
            raise FourwayTierCandidateError(f"{variant} candidate cutoff changed")
        if candidate.get("expected_live_as_of") != source["source_as_of"]:
            raise FourwayTierCandidateError(f"{variant} candidate expected cutoff changed")
        if candidate_source.get("source_latest_replayed") != source["source_as_of"]:
            raise FourwayTierCandidateError(f"{variant} candidate source cutoff changed")
        _verify_candidate_source(candidate_source, runtime, variant)
        candidates[variant] = candidate
        candidate_path = output / "candidates" / f"{variant}.json"
        raw_sha256 = _write_json_once(candidate_path, candidate, f"{variant} candidate")
        candidate_records[variant] = {
            "variant": variant,
            "candidate": {
                "locator": str(candidate_path.relative_to(output)),
                "bytes": candidate_path.stat().st_size,
                "raw_sha256": raw_sha256,
                "artifact_sha256": candidate.get("artifact_sha256"),
            },
            "validation": validation,
            "offsets_sha256": provenances[variant]["offsets_sha256"],
            "offset_receipt_sha256": offset_records[variant]["receipt_sha256"],
            "game_count": len(eligible_ids),
            "game_identity_sha256": identity_sha256(eligible_ids),
        }
    if sha256_path(baseline_candidate_path) != baseline_raw_sha256:
        raise FourwayTierCandidateError("current baseline candidate bytes changed")
    baseline_records = {
        "bundle": _file_record(baseline_path, "Tier baseline trust bundle"),
        "candidate": {
            "path": str(baseline_candidate_path),
            "bytes": baseline_candidate_path.stat().st_size,
            "raw_sha256": baseline_raw_sha256,
            "artifact_sha256": baseline_artifact_sha256,
        },
    }
    for label in ("current_production_manifest", "current_prospective_evaluation"):
        baseline_records[label] = {
            **_file_record(
                _bundle_file(baseline_path.parent, bundle[label], f"Tier baseline {label}"),
                f"Tier baseline {label}",
            ),
            "locator": bundle[label].get("locator"),
        }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "research_only",
        "authority": dict(AUTHORITY),
        "source": {
            **source,
            "source_files": source_files,
        },
        "baseline": baseline_records,
        "offsets": offset_records,
        "variants": candidate_records,
        "runtime": runtime,
        "claim_ceiling": {
            "production": False,
            "publication": False,
            "recommendation": False,
            "betting": False,
        },
        "blockers": [
            "retrospective_full_census_model_fit_not_chronological_evaluation",
            "public_tierlist_authority_missing",
        ],
    }
    manifest["manifest_sha256"] = _hash_bytes(canonical_json_bytes(manifest))
    manifest_path = output / MANIFEST_FILE
    _write_json_once(manifest_path, manifest, "fourway Tier candidate manifest")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-manifest", required=True, type=Path)
    parser.add_argument("--shadow-manifest-sha256", required=True)
    parser.add_argument("--baseline-bundle", required=True, type=Path)
    parser.add_argument("--baseline-bundle-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path)
    args = parser.parse_args(argv)
    result = build_fourway_tier_candidates(
        shadow_manifest_path=args.shadow_manifest,
        expected_shadow_manifest_sha256=args.shadow_manifest_sha256,
        baseline_bundle_path=args.baseline_bundle,
        expected_baseline_bundle_sha256=args.baseline_bundle_sha256,
        output_root=args.output_root,
        repository_root=args.repository_root,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "manifest_sha256": result["manifest_sha256"],
                "variants": list(VARIANT_ORDER),
                "game_count": result["source"]["model_eligible_game_count"],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "AUTHORITY",
    "FourwayTierCandidateError",
    "MANIFEST_FILE",
    "SCHEMA_VERSION",
    "VARIANT_ORDER",
    "build_fourway_tier_candidates",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
