"""Build the byte-bound research trust manifest for the Tier shadow.

The command seals one accepted source receipt, one current Tier candidate, the
four future-value evaluation artifacts, and the code and Tier assets used by
the candidate builder.  It does not fit a model and it does not publish a
Tier List.  The output has the closed shape consumed by
``future_value_tierlist.load_trust_manifest``.

Every input is reopened after its path is checked.  Symlink components,
directory escapes, changed bytes, source census drift, and resealed current
artifacts fail before the output is written.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from benchmarks.future_value_tierlist_full_census_diff import (
    _verify_candidate,
)
from lol_kills.research.future_value_rating import (
    validate_future_value_source_receipt_payload,
)
from lol_kills.research.future_value_tierlist import (
    AUTHORITY,
    VARIANTS,
    FutureValueTierListError,
    canonical_json_bytes,
    load_prediction_offsets,
    validate_common_prediction_universe,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


TRUST_SCHEMA_VERSION = "scryglass:future-value-tierlist-freeze:v1"
SOURCE_RECEIPT_SCHEMA_VERSION = "scryglass:future-value-rating-source:v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)

DEFAULT_TIER_ASSET_LOCATORS = (
    "data/lol/v2/champions/champion-id-crosswalk-v1.json",
    "data/lol/v2/champions/lcc-atom-bridge-26.16.json",
    "data/lol/v2/champions/lcc-atom-bridge-v1.json",
    "data/lol/v2/champions/oe-atom-patch-map-v1.json",
    "data/lol/v2/champions/sources/riot-champion-metadata-16.14.1.json",
)

# These files are part of the model implementation contract.  They are
# stored in the existing ``tier_assets`` field so older shadow consumers can
# verify them without a schema change.
DEFAULT_IMPLEMENTATION_LOCATORS = (
    "benchmarks/future_value_tierlist_fourway.py",
    "lol_kills/research/future_value_tierlist.py",
    "lol_kills/research/future_value_rating.py",
    "lol_kills/v2/tierlists/atom_matchup_features.py",
    "lol_kills/v2/tierlists/champion_elo.py",
    "lol_kills/v2/tierlists/joint_pooled_model.py",
    "lol_kills/v2/tierlists/patch_mapping.py",
    "lol_kills/v2/tierlists/pooled_candidate.py",
)


class TierTrustBuilderError(ValueError):
    """The Tier trust manifest cannot be built safely."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(value: object, label: str) -> str:
    text = str(value or "").lower()
    if SHA256_RE.fullmatch(text) is None:
        raise TierTrustBuilderError(f"{label} must be a SHA-256 digest")
    return text


def _self_hash(payload: Mapping[str, Any], field: str, label: str) -> str:
    claimed = _require_hash(payload.get(field), f"{label} {field}")
    unsigned = dict(payload)
    unsigned.pop(field, None)
    if hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != claimed:
        raise TierTrustBuilderError(f"{label} self hash changed")
    return claimed


def _absolute_without_resolving(path: Path | str) -> Path:
    raw = Path(path).expanduser()
    if ".." in raw.parts:
        raise TierTrustBuilderError(f"unsafe path contains '..': {path}")
    return Path(os.path.abspath(os.fspath(raw)))


def _safe_file(path: Path | str, label: str) -> Path:
    """Open only an existing regular file with no symlink component."""

    candidate = _absolute_without_resolving(path)
    if any(parent.is_symlink() for parent in (candidate, *candidate.parents)):
        raise TierTrustBuilderError(f"{label} is a symlink or has a symlink parent")
    if not candidate.is_file():
        raise TierTrustBuilderError(f"{label} is missing or not a file: {candidate}")
    return candidate


def _safe_directory(path: Path | str, label: str) -> Path:
    candidate = _absolute_without_resolving(path)
    if any(parent.is_symlink() for parent in (candidate, *candidate.parents)):
        raise TierTrustBuilderError(f"{label} is a symlink or has a symlink parent")
    if not candidate.is_dir():
        raise TierTrustBuilderError(f"{label} is missing or not a directory: {candidate}")
    return candidate


def _relative_locator(path: Path, root: Path, label: str) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise TierTrustBuilderError(f"{label} is outside its declared root") from error
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise TierTrustBuilderError(f"{label} has an unsafe locator")
    return relative.as_posix()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TierTrustBuilderError(f"{label} cannot be read") from error
    if not isinstance(value, dict):
        raise TierTrustBuilderError(f"{label} must be a JSON object")
    return value


def _json_safe(value: object) -> object:
    """Convert nested audit values to canonical JSON without losing keys."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _verify_bytes(path: Path, expected: object, label: str) -> dict[str, Any]:
    expected_hash = _require_hash(expected, f"expected {label} hash")
    actual = sha256_path(path)
    if actual != expected_hash:
        raise TierTrustBuilderError(f"{label} bytes changed")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}


def _source_file_from_receipt(
    source_root: Path,
    receipt: Mapping[str, Any],
    label: str,
) -> Path:
    records = receipt.get("source_files")
    record = records.get(label) if isinstance(records, Mapping) else None
    if not isinstance(record, Mapping):
        raise TierTrustBuilderError(f"source file record is missing: {label}")
    locator = record.get("locator")
    path_value = record.get("path")
    if isinstance(locator, str) and locator.strip():
        relative = Path(locator)
        if relative.is_absolute() or ".." in relative.parts:
            raise TierTrustBuilderError(f"source file locator is unsafe: {label}")
        path = _safe_file(source_root / relative, f"source file {label}")
    elif isinstance(path_value, str) and path_value.strip():
        path = _safe_file(path_value, f"source file {label}")
        try:
            path.relative_to(source_root)
        except ValueError as error:
            raise TierTrustBuilderError(
                f"source file {label} is outside the supplied source root"
            ) from error
    else:
        raise TierTrustBuilderError(f"source file locator is missing: {label}")
    declared_bytes = record.get("bytes")
    if isinstance(declared_bytes, bool) or not isinstance(declared_bytes, int):
        raise TierTrustBuilderError(f"source file byte count is invalid: {label}")
    if path.stat().st_size != declared_bytes:
        raise TierTrustBuilderError(f"source file bytes changed: {label}")
    _verify_bytes(path, record.get("sha256"), f"source file {label}")
    return path


def _verify_source(
    source_root: Path,
    receipt_path: Path,
    *,
    expected_receipt_file_sha256: str,
    expected_receipt_sha256: str,
) -> tuple[dict[str, Any], dict[str, Path], dict[str, Any]]:
    _verify_bytes(receipt_path, expected_receipt_file_sha256, "source receipt")
    receipt = _load_json(receipt_path, "source receipt")
    try:
        accepted_ids, eligible_ids = validate_future_value_source_receipt_payload(
            receipt,
            expected_receipt_sha256=expected_receipt_sha256,
        )
    except Exception as error:
        raise TierTrustBuilderError(f"source receipt is not verified: {error}") from error
    if receipt.get("schema_version") != SOURCE_RECEIPT_SCHEMA_VERSION:
        raise TierTrustBuilderError("source receipt schema changed")
    source_files = {
        label: _source_file_from_receipt(source_root, receipt, label)
        for label in ("maps", "players", "teams")
    }
    meta_path = _safe_file(source_root / "source" / "meta.json", "source metadata")
    source_files["meta"] = meta_path
    source = {
        "source_as_of": receipt["source_as_of"],
        "source_game_count": int(receipt["source_game_count"]),
        "source_identity_sha256": str(receipt["source_identity_sha256"]),
        "source_receipt_sha256": str(receipt["receipt_sha256"]),
        "source_receipt_file_sha256": _require_hash(
            expected_receipt_file_sha256, "source receipt file hash"
        ),
        "model_eligible_game_count": int(receipt["model_eligible_game_count"]),
        "model_eligible_identity_sha256": str(
            receipt["model_eligible_identity_sha256"]
        ),
        "accepted_game_ids": list(accepted_ids),
        "model_eligible_game_ids": list(eligible_ids),
        "player_source_sha256": sha256_path(source_files["players"]),
        "maps_source_sha256": sha256_path(source_files["maps"]),
        "meta_source_sha256": sha256_path(meta_path),
        "teams_source_sha256": sha256_path(source_files["teams"]),
    }
    return receipt, source_files, source


def _verify_current_candidate(
    path: Path,
    *,
    source: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = _load_json(path, "current Tier candidate")
    try:
        validation = _verify_candidate(
            candidate,
            label="current Tier candidate",
            expected_game_count=int(source["source_game_count"]),
            expected_identity=str(source["source_identity_sha256"]),
            expected_source_as_of=str(source["source_as_of"]),
        )
    except Exception as error:
        raise TierTrustBuilderError(
            f"current Tier candidate failed the existing validator: {error}"
        ) from error
    return candidate, validation


def _verify_candidate_reference(
    path: Path,
    payload: Mapping[str, Any],
    *,
    label: str,
    candidate_path: Path,
    candidate: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify a production manifest or forward evaluation reference."""

    if "artifact_sha256" in payload:
        _self_hash(payload, "artifact_sha256", label)
    reference = payload.get("candidate")
    if not isinstance(reference, Mapping):
        raise TierTrustBuilderError(f"{label} candidate binding is missing")
    raw_sha = _require_hash(reference.get("raw_sha256"), f"{label} candidate hash")
    candidate_sha = sha256_path(candidate_path)
    if raw_sha != candidate_sha:
        raise TierTrustBuilderError(f"{label} points to changed candidate bytes")
    if reference.get("artifact_sha256") != candidate.get("artifact_sha256"):
        raise TierTrustBuilderError(f"{label} candidate artifact identity changed")
    if "as_of" in reference and reference.get("as_of") != source["source_as_of"]:
        raise TierTrustBuilderError(f"{label} candidate cutoff changed")
    binding = payload.get("source")
    if isinstance(binding, Mapping):
        for field, expected in (
            ("source_game_count", source["source_game_count"]),
            ("source_identity_sha256", source["source_identity_sha256"]),
        ):
            if field in binding and binding.get(field) != expected:
                raise TierTrustBuilderError(f"{label} source binding changed: {field}")
        for field in ("source_latest", "source_latest_replayed", "meta_source_latest"):
            if field in binding and binding.get(field) not in {
                source["source_as_of"],
                str(source["source_as_of"]).replace("Z", "+00:00"),
            }:
                raise TierTrustBuilderError(f"{label} source cutoff changed: {field}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_path(path),
        "schema_version": payload.get("schema_version"),
        "candidate_raw_sha256": candidate_sha,
    }


def _parse_bindings(
    values: list[str] | None,
    *,
    option: str,
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values or []:
        if "=" not in raw:
            raise TierTrustBuilderError(f"{option} must use locator=path")
        locator, path = raw.split("=", 1)
        locator = locator.strip()
        if not locator or not path.strip():
            raise TierTrustBuilderError(f"{option} has an empty locator or path")
        if locator in result:
            raise TierTrustBuilderError(f"{option} repeats locator: {locator}")
        result[locator] = Path(path).expanduser()
    return result


def _asset_bindings(
    repository_root: Path,
    *,
    tier_assets: Mapping[str, Path] | None,
    implementation_files: Mapping[str, Path] | None,
) -> tuple[dict[str, str], dict[str, Any]]:
    provided: dict[str, Path] = {}
    if tier_assets:
        provided.update(tier_assets)
    if implementation_files:
        for locator, path in implementation_files.items():
            if locator in provided and provided[locator] != path:
                raise TierTrustBuilderError(f"asset locator is bound to two paths: {locator}")
            provided[locator] = path
    if not provided:
        for locator in (*DEFAULT_TIER_ASSET_LOCATORS, *DEFAULT_IMPLEMENTATION_LOCATORS):
            candidate = repository_root / locator
            if candidate.is_file() and not candidate.is_symlink():
                provided[locator] = candidate
    if not provided:
        raise TierTrustBuilderError("at least one Tier or implementation asset is required")

    hashes: dict[str, str] = {}
    code_fingerprints: dict[str, str] = {}
    for locator, raw_path in sorted(provided.items()):
        rel = Path(locator)
        if rel.is_absolute() or not rel.parts or ".." in rel.parts:
            raise TierTrustBuilderError(f"asset locator is unsafe: {locator}")
        path = _safe_file(raw_path, f"Tier asset {locator}")
        actual_locator = _relative_locator(path, repository_root, f"Tier asset {locator}")
        if actual_locator != rel.as_posix():
            raise TierTrustBuilderError(
                f"Tier asset locator does not match its repository path: {locator}"
            )
        digest = sha256_path(path)
        hashes[locator] = digest
        if path.suffix == ".py":
            code_fingerprints[locator] = digest
    if not code_fingerprints:
        raise TierTrustBuilderError("implementation code fingerprints are missing")
    return hashes, {
        "files": dict(sorted(code_fingerprints.items())),
        "sha256": hashlib.sha256(canonical_json_bytes(code_fingerprints)).hexdigest(),
    }


def _resolve_evaluations(
    evaluation_root: Path,
    evaluation_paths: Mapping[str, Path],
) -> dict[str, tuple[Path, str]]:
    if set(evaluation_paths) != set(VARIANTS):
        raise TierTrustBuilderError(
            f"evaluation files must cover exactly: {', '.join(VARIANTS)}"
        )
    result: dict[str, tuple[Path, str]] = {}
    for variant in VARIANTS:
        path = _safe_file(evaluation_paths[variant], f"{variant} evaluation")
        locator = _relative_locator(path, evaluation_root, f"{variant} evaluation")
        result[variant] = path, locator
    return result


def build_tier_trust_manifest(
    *,
    source_root: Path | str,
    source_receipt_path: Path | str,
    expected_source_receipt_file_sha256: str,
    expected_source_receipt_sha256: str,
    baseline_candidate_path: Path | str,
    production_manifest_path: Path | str,
    prospective_evaluation_path: Path | str,
    evaluation_root: Path | str,
    evaluation_paths: Mapping[str, Path | str],
    repository_root: Path | str,
    output_path: Path | str,
    tier_assets: Mapping[str, Path | str] | None = None,
    implementation_files: Mapping[str, Path | str] | None = None,
    input_binding_output: Path | str | None = None,
) -> dict[str, Any]:
    """Validate inputs and write one closed, research-only trust manifest."""

    freeze_root = _safe_directory(source_root, "source root")
    repository = _safe_directory(repository_root, "repository root")
    eval_root = _safe_directory(evaluation_root, "evaluation root")
    receipt_path = _safe_file(source_receipt_path, "source receipt")
    if receipt_path != freeze_root / "future-value-source-receipt.json":
        raise TierTrustBuilderError(
            "source receipt must be the freeze root future-value-source-receipt.json"
        )
    receipt, source_files, source = _verify_source(
        freeze_root,
        receipt_path,
        expected_receipt_file_sha256=expected_source_receipt_file_sha256,
        expected_receipt_sha256=expected_source_receipt_sha256,
    )
    baseline_path = _safe_file(baseline_candidate_path, "current Tier candidate")
    baseline_locator = _relative_locator(baseline_path, freeze_root, "current Tier candidate")
    candidate, candidate_validation = _verify_current_candidate(
        baseline_path, source=source
    )

    manifest_path = _safe_file(production_manifest_path, "current Tier production manifest")
    evaluation_summary_path = _safe_file(
        prospective_evaluation_path, "current Tier prospective evaluation"
    )
    manifest_binding = _verify_candidate_reference(
        manifest_path,
        _load_json(manifest_path, "current Tier production manifest"),
        label="current Tier production manifest",
        candidate_path=baseline_path,
        candidate=candidate,
        source=source,
    )
    evaluation_binding = _verify_candidate_reference(
        evaluation_summary_path,
        _load_json(evaluation_summary_path, "current Tier prospective evaluation"),
        label="current Tier prospective evaluation",
        candidate_path=baseline_path,
        candidate=candidate,
        source=source,
    )

    source_binding = dict(source)
    maps_path = source_files["maps"]
    evaluations = _resolve_evaluations(
        eval_root,
        {variant: Path(path) for variant, path in evaluation_paths.items()},
    )
    offsets: dict[str, dict[str, float]] = {}
    targets: dict[str, dict[str, float]] = {}
    model_bindings: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        path, locator = evaluations[variant]
        offsets[variant], targets[variant], binding = load_prediction_offsets(
            path,
            variant=variant,
            expected_raw_sha256=sha256_path(path),
            source=source_binding,
            maps_path=maps_path,
            expected_maps_sha256=source["maps_source_sha256"],
        )
        model_bindings[variant] = {
            **binding,
            "locator": locator,
            "raw_sha256": sha256_path(path),
        }
    try:
        _game_ids, universe = validate_common_prediction_universe(
            offsets,
            targets,
            accepted_game_ids=source["accepted_game_ids"],
            maps_path=maps_path,
            expected_maps_sha256=source["maps_source_sha256"],
        )
    except Exception as error:
        raise TierTrustBuilderError(
            f"evaluation artifacts do not share a verified universe: {error}"
        ) from error
    if universe.get("game_count") != source["model_eligible_game_count"]:
        raise TierTrustBuilderError("evaluation universe is not the model-eligible census")

    asset_hashes, code_fingerprint = _asset_bindings(
        repository,
        tier_assets={
            locator: Path(path)
            for locator, path in (tier_assets or {}).items()
        },
        implementation_files={
            locator: Path(path)
            for locator, path in (implementation_files or {}).items()
        },
    )
    output = _absolute_without_resolving(output_path)
    if output.exists() or output.is_symlink():
        raise TierTrustBuilderError(f"trust manifest output already exists: {output}")
    if any(parent.is_symlink() for parent in (output, *output.parents)):
        raise TierTrustBuilderError("trust manifest output path contains a symlink")
    output.parent.mkdir(parents=True, exist_ok=True)

    trust: dict[str, Any] = {
        "schema_version": TRUST_SCHEMA_VERSION,
        "status": "research_only",
        "authority": dict(AUTHORITY),
        "source": {
            "source_as_of": source["source_as_of"],
            "source_game_count": source["source_game_count"],
            "source_identity_sha256": source["source_identity_sha256"],
            "source_receipt_sha256": source["source_receipt_sha256"],
            "source_receipt_file_sha256": source["source_receipt_file_sha256"],
            "player_source_sha256": source["player_source_sha256"],
            "maps_source_sha256": source["maps_source_sha256"],
            "meta_source_sha256": source["meta_source_sha256"],
            "model_eligible_game_count": source["model_eligible_game_count"],
            "model_eligible_identity_sha256": source[
                "model_eligible_identity_sha256"
            ],
            "accepted_game_ids": source["accepted_game_ids"],
            "model_eligible_game_ids": source["model_eligible_game_ids"],
        },
        "evaluations": {
            variant: {
                "locator": model_bindings[variant]["locator"],
                "raw_sha256": model_bindings[variant]["raw_sha256"],
            }
            for variant in VARIANTS
        },
        "tier_assets": asset_hashes,
        "baseline_candidate": {
            "locator": baseline_locator,
            "raw_sha256": sha256_path(baseline_path),
        },
    }
    unsigned = dict(trust)
    trust["trust_root_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    output.write_bytes(canonical_json_bytes(trust) + b"\n")

    # These bindings keep the exact current production references auditable.
    # The closed trust manifest cannot add fields without breaking its older
    # consumers, so the optional sidecar carries the additional input records.
    input_binding = {
        "schema_version": "scryglass:future-value-tierlist-trust-inputs:v1",
        "status": "research_only",
        "authority": dict(AUTHORITY),
        "source_receipt": {
            "path": str(receipt_path),
            "bytes": receipt_path.stat().st_size,
            "sha256": sha256_path(receipt_path),
            "receipt_sha256": receipt["receipt_sha256"],
        },
        "current_candidate": {
            "path": str(baseline_path),
            "bytes": baseline_path.stat().st_size,
            "sha256": sha256_path(baseline_path),
            "artifact_sha256": candidate["artifact_sha256"],
            "validator": "benchmarks.future_value_tierlist_full_census_diff._verify_candidate",
            "validator_result": candidate_validation,
        },
        "current_production_manifest": manifest_binding,
        "current_prospective_evaluation": evaluation_binding,
        "evaluation_artifacts": model_bindings,
        "tier_assets": {
            locator: {
                "path": str(repository / locator),
                "bytes": (repository / locator).stat().st_size,
                "sha256": digest,
            }
            for locator, digest in sorted(asset_hashes.items())
        },
        "code_fingerprint": code_fingerprint,
        "source_census": {
            "accepted_game_count": source["source_game_count"],
            "accepted_identity_sha256": source["source_identity_sha256"],
            "model_eligible_game_count": source["model_eligible_game_count"],
            "model_eligible_identity_sha256": source[
                "model_eligible_identity_sha256"
            ],
            "accepted_game_ids_sha256": identity_sha256(source["accepted_game_ids"]),
            "model_eligible_game_ids_sha256": identity_sha256(
                source["model_eligible_game_ids"]
            ),
        },
    }
    input_binding = _json_safe(input_binding)
    assert isinstance(input_binding, dict)
    input_binding["input_binding_sha256"] = hashlib.sha256(
        canonical_json_bytes(input_binding)
    ).hexdigest()
    if input_binding_output is not None:
        sidecar = _absolute_without_resolving(input_binding_output)
        if sidecar.exists() or sidecar.is_symlink():
            raise TierTrustBuilderError("input binding output already exists")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_bytes(canonical_json_bytes(input_binding) + b"\n")
    return trust


def _evaluation_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> dict[str, Path]:
    explicit = {
        variant: getattr(args, f"{variant}_evaluation")
        for variant in VARIANTS
        if getattr(args, f"{variant}_evaluation") is not None
    }
    for raw in args.evaluation or []:
        if "=" not in raw:
            raise TierTrustBuilderError("--evaluation must use variant=path")
        variant, path = raw.split("=", 1)
        if variant in explicit:
            raise TierTrustBuilderError(f"evaluation is repeated: {variant}")
        explicit[variant] = Path(path)
    if not explicit and args.evaluation_root is not None:
        root = Path(args.evaluation_root)
        explicit = {variant: root / variant / "model.json" for variant in VARIANTS}
    return explicit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--source-receipt-file-sha256", required=True)
    parser.add_argument("--source-receipt-sha256", required=True)
    parser.add_argument("--baseline-candidate", type=Path, required=True)
    parser.add_argument(
        "--production-manifest",
        "--baseline-manifest",
        dest="production_manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--prospective-evaluation",
        "--baseline-evaluation",
        dest="prospective_evaluation",
        type=Path,
        required=True,
    )
    parser.add_argument("--evaluation-root", type=Path)
    for variant in VARIANTS:
        parser.add_argument(f"--{variant.replace('_', '-')}-evaluation", type=Path)
    parser.add_argument("--evaluation", action="append")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--tier-asset", action="append", help="locator=path")
    parser.add_argument("--implementation-file", action="append", help="locator=path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-binding-output", type=Path)
    args = parser.parse_args(argv)
    evaluations = _evaluation_arguments(parser, args)
    if args.evaluation_root is None:
        raise TierTrustBuilderError("--evaluation-root is required")
    trust = build_tier_trust_manifest(
        source_root=args.source_root,
        source_receipt_path=args.source_receipt,
        expected_source_receipt_file_sha256=args.source_receipt_file_sha256,
        expected_source_receipt_sha256=args.source_receipt_sha256,
        baseline_candidate_path=args.baseline_candidate,
        production_manifest_path=args.production_manifest,
        prospective_evaluation_path=args.prospective_evaluation,
        evaluation_root=args.evaluation_root,
        evaluation_paths=evaluations,
        repository_root=args.repository_root,
        tier_assets=_parse_bindings(args.tier_asset, option="--tier-asset"),
        implementation_files=_parse_bindings(
            args.implementation_file, option="--implementation-file"
        ),
        output_path=args.output,
        input_binding_output=args.input_binding_output,
    )
    print(
        json.dumps(
            {
                "status": trust["status"],
                "output": str(Path(args.output).expanduser().resolve()),
                "trust_root_sha256": trust["trust_root_sha256"],
                "accepted_game_count": trust["source"]["source_game_count"],
                "model_eligible_game_count": trust["source"][
                    "model_eligible_game_count"
                ],
                "evaluation_count": len(trust["evaluations"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TierTrustBuilderError as error:
        raise SystemExit(f"build_future_value_tier_trust: {error}") from error


__all__ = [
    "DEFAULT_IMPLEMENTATION_LOCATORS",
    "DEFAULT_TIER_ASSET_LOCATORS",
    "TRUST_SCHEMA_VERSION",
    "TierTrustBuilderError",
    "build_tier_trust_manifest",
    "canonical_json_bytes",
    "main",
    "sha256_path",
]
