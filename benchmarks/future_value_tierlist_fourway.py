"""Build four source-bound research Tier List shadows in one frozen universe."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any, Mapping

from lol_kills.research.future_value_rating import (
    validate_future_value_source_receipt_payload,
)
from lol_kills.research.future_value_tierlist import (
    AUTHORITY,
    PINNED_TRUST_MANIFEST_RAW_SHA256,
    VARIANTS,
    FutureValueTierListError,
    build_fourway_diff,
    canonical_json_bytes,
    load_prediction_offsets,
    load_trust_manifest,
    make_offset_provenance,
    sha256_path,
    validate_candidate,
    validate_common_prediction_universe,
)
from lol_kills.v2.tierlists.pooled_candidate import build_pooled_candidate


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FutureValueTierListError(f"{label} is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FutureValueTierListError(f"{label} cannot be read") from error
    if not isinstance(value, dict):
        raise FutureValueTierListError(f"{label} is not an object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(dict(value)) + b"\n")


def _implementation_binding(repo_root: Path) -> dict[str, Any]:
    """Bind the clean source tree and runtime that produced a run receipt."""

    try:
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise FutureValueTierListError("implementation git status is unavailable") from error
    if status.stdout.strip():
        raise FutureValueTierListError("implementation working tree must be clean")

    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--verify", "HEAD^{commit}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise FutureValueTierListError("implementation git commit is unavailable") from error
    if not commit:
        raise FutureValueTierListError("implementation git commit is empty")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit.lower()):
        raise FutureValueTierListError("implementation git commit is not a full hash")
    try:
        tracked_raw = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "ls-files",
                "-z",
                "--",
                "lol_kills/v2/tierlists",
                "benchmarks/future_value_tierlist_fourway.py",
                "lol_kills/research/future_value_tierlist.py",
                "lol_kills/research/future_value_rating.py",
            ],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise FutureValueTierListError("implementation source listing is unavailable") from error
    if isinstance(tracked_raw, bytes):
        tracked_text = tracked_raw.decode("utf-8")
    else:
        tracked_text = str(tracked_raw)
    locators = tuple(
        sorted(
            {
                locator
                for locator in tracked_text.split("\0")
                if locator.endswith(".py")
                and (
                    locator.startswith("lol_kills/v2/tierlists/")
                    or locator
                    in {
                        "benchmarks/future_value_tierlist_fourway.py",
                        "lol_kills/research/future_value_tierlist.py",
                        "lol_kills/research/future_value_rating.py",
                    }
                )
            }
        )
    )
    required = {
        "benchmarks/future_value_tierlist_fourway.py",
        "lol_kills/research/future_value_tierlist.py",
        "lol_kills/v2/tierlists/atom_matchup_features.py",
        "lol_kills/v2/tierlists/champion_elo.py",
        "lol_kills/v2/tierlists/joint_pooled_model.py",
        "lol_kills/v2/tierlists/patch_mapping.py",
        "lol_kills/v2/tierlists/pooled_candidate.py",
    }
    if not required.issubset(locators):
        missing = sorted(required.difference(locators))
        raise FutureValueTierListError(f"implementation source files are missing: {missing}")
    files: dict[str, str] = {}
    for locator in locators:
        path = repo_root / locator
        if not path.is_file() or path.is_symlink():
            raise FutureValueTierListError(f"implementation file is missing: {locator}")
        files[locator] = sha256_path(path)
    package_versions: dict[str, str] = {}
    for package in ("numpy", "pandas", "scipy", "pyarrow"):
        try:
            package_versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError as error:
            raise FutureValueTierListError(
                f"implementation package version is unavailable: {package}"
            ) from error
    return {
        "git_commit": commit,
        "files": files,
        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_build": list(platform.python_build()),
            "platform": platform.platform(),
            "packages": package_versions,
        },
    }


def _verify_source(
    source_root: Path,
    trust: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, Path, Path]:
    binding = trust["source"]
    receipt_path = source_root / "future-value-source-receipt.json"
    if sha256_path(receipt_path) != binding["source_receipt_file_sha256"]:
        raise FutureValueTierListError("source receipt file bytes changed")
    receipt = _load_json(receipt_path, "future-value source receipt")
    validate_future_value_source_receipt_payload(
        receipt,
        expected_receipt_sha256=binding["source_receipt_sha256"],
    )
    for field in (
        "source_as_of",
        "source_game_count",
        "source_identity_sha256",
        "model_eligible_game_count",
        "model_eligible_identity_sha256",
    ):
        if receipt.get(field) != binding.get(field):
            raise FutureValueTierListError(f"source receipt field changed: {field}")
    player_path = source_root / "source" / "oe_player_games.parquet"
    maps_path = source_root / "source" / "maps.parquet"
    meta_path = source_root / "source" / "meta.json"
    if (
        not player_path.is_file()
        or player_path.is_symlink()
        or sha256_path(player_path) != binding["player_source_sha256"]
    ):
        raise FutureValueTierListError("frozen player source bytes changed")
    if (
        not maps_path.is_file()
        or maps_path.is_symlink()
        or sha256_path(maps_path) != binding["maps_source_sha256"]
    ):
        raise FutureValueTierListError("frozen maps source bytes changed")
    if (
        not meta_path.is_file()
        or meta_path.is_symlink()
        or sha256_path(meta_path) != binding["meta_source_sha256"]
    ):
        raise FutureValueTierListError("frozen OE metadata bytes changed")
    return receipt, player_path, maps_path, meta_path


def _stage_runtime(
    destination: Path,
    *,
    repo_root: Path,
    player_source: Path,
    meta_source: Path,
    trust: Mapping[str, Any],
) -> Path:
    runtime = destination / "runtime"
    if runtime.exists():
        raise FutureValueTierListError("Tier shadow runtime already exists")
    source_target = runtime / "data/lol/warehouse/parquet/oe_live/oe_player_games.parquet"
    source_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(player_source, source_target)
    meta_target = runtime / "data/lol/warehouse/parquet/oe_live/meta.json"
    meta_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(meta_source, meta_target)
    for locator, expected_hash in trust["tier_assets"].items():
        source = repo_root / locator
        if not source.is_file() or source.is_symlink() or sha256_path(source) != expected_hash:
            raise FutureValueTierListError(f"Tier asset bytes changed: {locator}")
        target = runtime / locator
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return runtime


def _build_variant(
    variant: str,
    runtime_root: str,
    game_ids: list[str],
    offsets: dict[str, float],
    provenance: dict[str, Any],
    expected_source_receipt_sha256: str,
) -> tuple[str, dict[str, Any]]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    candidate = build_pooled_candidate(
        Path(runtime_root),
        source_mode="oe_only",
        allowed_game_ids=game_ids,
        pre_map_offset_override=offsets,
        pre_map_offset_provenance=provenance,
        expected_pre_map_offset_source_receipt_sha256=expected_source_receipt_sha256,
    )
    return variant, candidate


def run_fourway(
    *,
    repo_root: Path,
    source_root: Path,
    evaluation_root: Path,
    trust_manifest_path: Path,
    expected_trust_manifest_sha256: str,
    output_root: Path,
    workers: int,
) -> dict[str, Any]:
    """Verify inputs, build four candidates, and write the exact diff report."""

    implementation = _implementation_binding(repo_root)
    if expected_trust_manifest_sha256 != PINNED_TRUST_MANIFEST_RAW_SHA256:
        raise FutureValueTierListError("Tier shadow trust manifest is not the code-pinned freeze")
    trust = load_trust_manifest(
        trust_manifest_path,
        expected_raw_sha256=expected_trust_manifest_sha256,
    )
    receipt, player_path, maps_path, meta_path = _verify_source(source_root, trust)
    baseline_record = trust["baseline_candidate"]
    baseline_path = source_root / baseline_record["locator"]
    if (
        not baseline_path.is_file()
        or baseline_path.is_symlink()
        or sha256_path(baseline_path) != baseline_record["raw_sha256"]
    ):
        raise FutureValueTierListError("baseline public Tier candidate bytes changed")
    source_binding = {
        "source_as_of": receipt["source_as_of"],
        "source_game_count": receipt["source_game_count"],
        "source_identity_sha256": receipt["source_identity_sha256"],
        "source_receipt_sha256": receipt["receipt_sha256"],
        "source_receipt_file_sha256": trust["source"]["source_receipt_file_sha256"],
        "model_eligible_game_count": receipt["model_eligible_game_count"],
        "model_eligible_identity_sha256": receipt["model_eligible_identity_sha256"],
        "accepted_game_ids": list(receipt["accepted_game_ids"]),
        "model_eligible_game_ids": list(receipt["model_eligible_game_ids"]),
    }
    offsets: dict[str, dict[str, float]] = {}
    targets: dict[str, dict[str, float]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        record = trust["evaluations"][variant]
        model_path = evaluation_root / record["locator"]
        offsets[variant], targets[variant], bindings[variant] = load_prediction_offsets(
            model_path,
            variant=variant,
            expected_raw_sha256=record["raw_sha256"],
            source=source_binding,
            maps_path=maps_path,
            expected_maps_sha256=trust["source"]["maps_source_sha256"],
        )
    game_ids, universe = validate_common_prediction_universe(
        offsets,
        targets,
        accepted_game_ids=receipt["accepted_game_ids"],
        maps_path=maps_path,
        expected_maps_sha256=trust["source"]["maps_source_sha256"],
    )
    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()):
        raise FutureValueTierListError("Tier shadow output root must be empty")
    runtime = _stage_runtime(
        output_root,
        repo_root=repo_root,
        player_source=player_path,
        meta_source=meta_path,
        trust=trust,
    )
    provenances = {
        variant: make_offset_provenance(
            variant=variant,
            offsets=offsets[variant],
            source_receipt_sha256=receipt["receipt_sha256"],
        )
        for variant in VARIANTS
    }
    candidates: dict[str, dict[str, Any]] = {}
    worker_count = min(max(1, int(workers)), len(VARIANTS))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _build_variant,
                variant,
                str(runtime),
                game_ids,
                offsets[variant],
                provenances[variant],
                str(receipt["receipt_sha256"]),
            ): variant
            for variant in VARIANTS
        }
        for future in as_completed(futures):
            variant, candidate = future.result()
            validate_candidate(
                candidate,
                variant=variant,
                universe=universe,
                expected_source_receipt_sha256=str(receipt["receipt_sha256"]),
                expected_offsets_sha256=str(bindings[variant]["offsets_sha256"]),
                expected_producer=str(bindings[variant]["producer"]),
            )
            candidates[variant] = candidate
            _write_json(output_root / "candidates" / f"{variant}.json", candidate)
    report = build_fourway_diff(
        candidates,
        source=source_binding,
        universe=universe,
        model_bindings=bindings,
        trust_manifest_raw_sha256=expected_trust_manifest_sha256,
        baseline_candidate_raw_sha256=baseline_record["raw_sha256"],
    )
    _write_json(output_root / "fourway-tierlist-report.json", report)
    receipt_payload = {
        "schema_version": "scryglass:future-value-tierlist-fourway-run:v1",
        "status": "research_only",
        "authority": dict(AUTHORITY),
        "report_locator": "fourway-tierlist-report.json",
        "report_raw_sha256": sha256_path(output_root / "fourway-tierlist-report.json"),
        "report_sha256": report["report_sha256"],
        "trust_manifest_raw_sha256": expected_trust_manifest_sha256,
        "variant_candidate_raw_sha256": {
            variant: sha256_path(output_root / "candidates" / f"{variant}.json")
            for variant in VARIANTS
        },
        "implementation": implementation,
    }
    receipt_payload["receipt_sha256"] = __import__("hashlib").sha256(
        canonical_json_bytes(receipt_payload)
    ).hexdigest()
    _write_json(output_root / "run-receipt.json", receipt_payload)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--trust-manifest", type=Path, required=True)
    parser.add_argument("--expected-trust-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    report = run_fourway(
        repo_root=args.repo_root.resolve(),
        source_root=args.source_root.resolve(),
        evaluation_root=args.evaluation_root.resolve(),
        trust_manifest_path=args.trust_manifest.resolve(),
        expected_trust_manifest_sha256=args.expected_trust_manifest_sha256,
        output_root=args.output_root.resolve(),
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "report_sha256": report["report_sha256"],
                "evaluation_maps": report["evaluation_universe"]["game_count"],
                "tier_rows": report["evaluation_universe"]["common_row_count"],
                "blockers": report["blockers"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
