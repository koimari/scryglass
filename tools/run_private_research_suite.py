"""Run the complete private research gate across current and frozen lanes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Sequence


HISTORICAL_CONTRACT_COMMIT = "94e5cec16adcf77a45f429e666fb3b5f54f40bd9"
HISTORICAL_MARKET_COMMIT = "87bcff462f6005269b94fd9e604c9e8b2b5ec9c0"

FROZEN_RUNTIME_TESTS = (
    "tests/model_v2/evaluation/test_b3_coverage.py",
    "tests/model_v2/draft/interactions/test_oe_nuisance_baseline.py",
    "tests/model_v2/draft/interactions/test_representation_rank_assay.py",
    "tests/model_v2/draft/terminal/test_development_evaluation.py",
    "tests/model_v2/draft/terminal/test_development_v3.py",
    "tests/model_v2/ratings/player/test_multileague_benchmark.py",
    "tests/model_v2/ratings/player/test_multileague_v3_corrected_adaptive_diagnostic_v1.py",
    "tests/model_v2/ratings/player/test_private_development_runner.py",
    "tests/model_v2/ratings/team/test_last_observed_real_v1.py",
)

HISTORICAL_CONTRACT_TESTS = (
    "tests/model_v2/evaluation/test_b1_sealed_and_contracts.py",
    "tests/model_v2/evaluation/test_contract_validation_remand.py",
)

HISTORICAL_MARKET_TESTS = (
    "tests/model_v2/market/test_phase_one_collection_readiness_v1.py",
)

FROZEN_VERSIONS = {
    "python": "3.9.6",
    "numpy": "2.0.2",
    "pandas": "2.3.3",
    "pyarrow": "21.0.0",
    "scipy": "1.13.1",
}


class PrivateSuiteError(RuntimeError):
    """A private-suite precondition or lane failed."""


@dataclass(frozen=True)
class LaneReceipt:
    name: str
    pass_number: int
    cwd: str
    python: str
    tests: tuple[str, ...]
    command: tuple[str, ...]
    elapsed_seconds: float
    log_path: str
    log_bytes: int
    log_sha256: str
    resumed_from_checkpoint: bool = False


@dataclass(frozen=True)
class LaneSpec:
    name: str
    source_root: Path
    python: Path
    tests: tuple[str, ...]
    extra_args: tuple[str, ...]


@dataclass(frozen=True)
class LaneWorkspace:
    cwd: Path
    private_root: Path
    parent: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _runtime_versions(python: Path) -> dict[str, str]:
    program = (
        "import importlib.metadata,json,platform;"
        "print(json.dumps({'python':platform.python_version(),"
        "**{name:importlib.metadata.version(name) for name in "
        "('numpy','pandas','pyarrow','scipy')}},sort_keys=True))"
    )
    completed = subprocess.run(
        [str(python), "-c", program],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _test_files(root: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in (root / "tests").rglob("test_*.py")
            if path.is_file()
        )
    )


def _absolute_executable(path: Path) -> Path:
    """Keep virtual-environment symlinks while making the path absolute."""

    return Path(os.path.abspath(os.fspath(path)))


def _copy_on_write_tree(source: Path, destination: Path) -> None:
    """Clone a lane input tree without sharing regular-file writes."""

    source = source.resolve()
    destination = destination.resolve()
    if not source.is_dir():
        raise PrivateSuiteError(f"lane source is not a directory: {source}")
    if destination.exists():
        raise PrivateSuiteError(f"lane destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        subprocess.run(
            ["cp", "-cR", str(source), str(destination)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    else:
        shutil.copytree(source, destination, symlinks=True, copy_function=shutil.copy2)


def _lane_specs(
    *,
    current_root: Path,
    contract_root: Path,
    market_root: Path,
    current_python: Path,
    frozen_python: Path,
    current_tests: Sequence[str],
) -> tuple[LaneSpec, ...]:
    return (
        LaneSpec(
            name="current",
            source_root=current_root,
            python=current_python,
            tests=tuple(current_tests),
            extra_args=(),
        ),
        LaneSpec(
            name="frozen-runtime",
            source_root=current_root,
            python=frozen_python,
            tests=tuple(FROZEN_RUNTIME_TESTS),
            extra_args=(),
        ),
        LaneSpec(
            name="historical-contract",
            source_root=contract_root,
            python=frozen_python,
            tests=tuple(HISTORICAL_CONTRACT_TESTS),
            extra_args=(),
        ),
        LaneSpec(
            name="historical-market",
            source_root=market_root,
            python=frozen_python,
            tests=tuple(HISTORICAL_MARKET_TESTS),
            extra_args=(),
        ),
    )


def _partition_current_tests(
    *,
    root: Path,
    tests: Sequence[str],
    shard_count: int,
) -> tuple[tuple[str, ...], ...]:
    """Partition the current test inventory by stable source-size balancing."""

    if shard_count < 1:
        raise PrivateSuiteError("current shard count must be positive")
    ordered = tuple(sorted(set(tests)))
    if len(ordered) != len(tests):
        raise PrivateSuiteError("current test inventory contains duplicates")
    if not ordered:
        raise PrivateSuiteError("current test inventory is empty")
    shard_count = min(shard_count, len(ordered))
    buckets: list[list[str]] = [[] for _ in range(shard_count)]
    weights = [0] * shard_count
    try:
        weighted_paths = sorted(
            (((root / path).stat().st_size, path) for path in ordered),
            key=lambda item: (-item[0], item[1]),
        )
    except OSError as exc:
        raise PrivateSuiteError(
            "current test inventory contains an unavailable file"
        ) from exc
    for weight, path in weighted_paths:
        index = min(range(shard_count), key=lambda value: (weights[value], value))
        buckets[index].append(path)
        weights[index] += weight
    return tuple(tuple(sorted(bucket)) for bucket in buckets)


def _expand_current_spec(
    *,
    spec: LaneSpec,
    root: Path,
    shard_count: int,
) -> tuple[LaneSpec, ...]:
    if spec.name != "current":
        return (spec,)
    partitions = _partition_current_tests(
        root=root,
        tests=spec.tests,
        shard_count=shard_count,
    )
    return tuple(
        LaneSpec(
            name=f"current-shard-{index:02d}",
            source_root=spec.source_root,
            python=spec.python,
            tests=tests,
            extra_args=spec.extra_args,
        )
        for index, tests in enumerate(partitions, start=1)
    )


def _checkpoint_path(receipt: Path, pass_number: int, lane_name: str) -> Path:
    return receipt.with_name(
        f"{receipt.stem}-pass-{pass_number}-{lane_name}.checkpoint.json"
    )


def _lane_log_path(receipt: Path, pass_number: int, lane_name: str) -> Path:
    return receipt.with_name(f"{receipt.stem}-pass-{pass_number}-{lane_name}.log")


def _canonical_json(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _inventory_sha256(tests: Sequence[str]) -> str:
    return hashlib.sha256(_canonical_json({"tests": sorted(tests)})).hexdigest()


def _is_current_lane(name: str) -> bool:
    return name == "current" or name.startswith("current-shard-")


LEGACY_CHECKPOINT_LANES = frozenset(
    {"frozen-runtime", "historical-contract", "historical-market"}
)


def _signed_payload(payload: dict, field: str) -> dict:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    signed = dict(unsigned)
    signed[field] = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    return signed


def _make_lane_workspace(
    *,
    spec: LaneSpec,
    current_root: Path,
    workspace_parent: Path,
) -> LaneWorkspace:
    """Give each lane a private copy-on-write checkout and data root."""

    parent = Path(
        tempfile.mkdtemp(prefix=f"pass-lane-{spec.name}-", dir=workspace_parent)
    )
    cwd = parent / "repo"
    _copy_on_write_tree(spec.source_root, cwd)
    if spec.source_root.resolve() == current_root.resolve():
        private_root = cwd
    else:
        private_root = parent / "private"
        _copy_on_write_tree(current_root, private_root)
    return LaneWorkspace(cwd=cwd, private_root=private_root, parent=parent)


def _lane_environment(
    base: dict[str, str],
    *,
    private_root: Path,
    lcc_root: Path,
) -> dict[str, str]:
    environment = dict(base)
    environment.update(
        {
            "SCRYGLASS_LCC_REPO": str(lcc_root),
            "SCRYGLASS_PRIVATE_TEST_ROOT": str(private_root),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def _remove_workspace(workspace: LaneWorkspace) -> None:
    with suppress(OSError):
        shutil.rmtree(workspace.parent)


def _partition_tests(all_tests: Sequence[str]) -> tuple[str, ...]:
    frozen = set(FROZEN_RUNTIME_TESTS) | set(HISTORICAL_CONTRACT_TESTS)
    missing = frozen - set(all_tests)
    if missing:
        raise PrivateSuiteError(f"frozen test inventory is missing: {sorted(missing)!r}")
    current_tests = tuple(path for path in all_tests if path not in frozen)
    if set(current_tests) | frozen != set(all_tests):
        raise PrivateSuiteError("private test partition is incomplete")
    return current_tests


def _verify_inputs(
    *,
    current_root: Path,
    contract_root: Path,
    market_root: Path,
    current_python: Path,
    frozen_python: Path,
    lcc_root: Path,
) -> tuple[tuple[str, ...], dict[str, str]]:
    for label, path in (
        ("current root", current_root),
        ("historical contract root", contract_root),
        ("historical market root", market_root),
        ("LCC root", lcc_root),
        ("current Python", current_python),
        ("frozen Python", frozen_python),
    ):
        if not path.exists():
            raise PrivateSuiteError(f"{label} is unavailable: {path}")
    if _git_head(contract_root) != HISTORICAL_CONTRACT_COMMIT:
        raise PrivateSuiteError("historical contract worktree commit changed")
    if _git_head(market_root) != HISTORICAL_MARKET_COMMIT:
        raise PrivateSuiteError("historical market worktree commit changed")
    frozen_versions = _runtime_versions(frozen_python)
    if frozen_versions != FROZEN_VERSIONS:
        raise PrivateSuiteError(
            f"frozen runtime changed: {frozen_versions!r}"
        )
    subprocess.run(
        [str(current_python), "-m", "pip", "check"],
        cwd=current_root,
        check=True,
    )
    all_tests = _test_files(current_root)
    current_tests = _partition_tests(all_tests)
    return current_tests, frozen_versions


def _run_lane(
    *,
    name: str,
    pass_number: int,
    cwd: Path,
    python: Path,
    tests: Sequence[str],
    extra_args: Sequence[str],
    environment: dict[str, str],
    log_path: Path,
) -> LaneReceipt:
    command = (str(python), "-m", "pytest", "-q", *extra_args, *tests)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        return_code = process.wait()
    elapsed = time.monotonic() - started
    if return_code != 0:
        raise PrivateSuiteError(f"{name} pass {pass_number} failed")
    return LaneReceipt(
        name=name,
        pass_number=pass_number,
        cwd=str(cwd),
        python=str(python),
        tests=tuple(tests),
        command=command,
        elapsed_seconds=round(elapsed, 3),
        log_path=str(log_path),
        log_bytes=log_path.stat().st_size,
        log_sha256=_sha256(log_path),
    )


def _checkpoint_payload(
    *,
    receipt: Path,
    pass_number: int,
    spec: LaneSpec,
    lane_receipt: LaneReceipt,
    current_root: Path,
    historical_contract_root: Path,
    historical_market_root: Path,
    frozen_versions: dict[str, str],
) -> dict:
    return _signed_payload(
        {
            "schema_version": "scryglass:private-research-suite-lane-checkpoint:v1",
            "status": "passed",
            "suite_receipt": str(receipt),
            "pass_number": pass_number,
            "lane": asdict(lane_receipt),
            "test_inventory_sha256": _inventory_sha256(lane_receipt.tests),
            "source_root": str(spec.source_root),
            "source_commit": _git_head(spec.source_root),
            "current_commit": _git_head(current_root),
            "historical_contract_commit": _git_head(historical_contract_root),
            "historical_market_commit": _git_head(historical_market_root),
            "requirements_ci_lock_sha256": _sha256(
                current_root / "requirements-ci.lock"
            ),
            "runtime": _runtime_versions(spec.python),
            "frozen_runtime": frozen_versions,
        },
        "checkpoint_sha256",
    )


def _write_checkpoint(path: Path, payload: dict) -> None:
    """Write a lane checkpoint atomically and refuse a different checkpoint."""

    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise PrivateSuiteError(f"invalid existing lane checkpoint: {path}") from exc
        if existing != payload:
            raise PrivateSuiteError(f"lane checkpoint already exists: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded)
    os.replace(temporary, path)


def _load_checkpoint(
    *,
    path: Path,
    receipt: Path,
    expected_log_path: Path,
    pass_number: int,
    spec: LaneSpec,
    current_root: Path,
    historical_contract_root: Path,
    historical_market_root: Path,
    frozen_versions: dict[str, str],
) -> LaneReceipt:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise PrivateSuiteError(f"invalid lane checkpoint: {path}") from exc
    if payload.get("schema_version") != (
        "scryglass:private-research-suite-lane-checkpoint:v1"
    ):
        raise PrivateSuiteError(f"unsupported lane checkpoint: {path}")
    if payload.get("status") != "passed":
        raise PrivateSuiteError(f"lane checkpoint is not passed: {path}")
    checkpoint_sha256 = payload.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha256, str):
        raise PrivateSuiteError(f"lane checkpoint digest is missing: {path}")
    if hashlib.sha256(
        _canonical_json(
            {key: value for key, value in payload.items() if key != "checkpoint_sha256"}
        )
    ).hexdigest() != checkpoint_sha256:
        raise PrivateSuiteError(f"lane checkpoint digest mismatch: {path}")
    if payload.get("suite_receipt") != str(receipt):
        raise PrivateSuiteError(f"lane checkpoint receipt binding mismatch: {path}")
    if payload.get("pass_number") != pass_number:
        raise PrivateSuiteError(f"lane checkpoint pass binding mismatch: {path}")
    lane = payload.get("lane")
    if not isinstance(lane, dict) or lane.get("name") != spec.name:
        raise PrivateSuiteError(f"lane checkpoint lane binding mismatch: {path}")
    if lane.get("pass_number") != pass_number:
        raise PrivateSuiteError(f"lane checkpoint pass binding mismatch: {path}")
    legacy_inventory_checkpoint = "test_inventory_sha256" not in payload
    if lane.get("log_path") != str(expected_log_path):
        raise PrivateSuiteError(f"lane checkpoint log binding mismatch: {path}")
    if payload.get("source_root") != str(spec.source_root):
        raise PrivateSuiteError(f"lane checkpoint source binding mismatch: {path}")
    if payload.get("source_commit") != _git_head(spec.source_root):
        raise PrivateSuiteError(f"lane checkpoint source commit changed: {path}")
    if payload.get("current_commit") != _git_head(current_root):
        raise PrivateSuiteError(f"lane checkpoint current commit changed: {path}")
    if payload.get("historical_contract_commit") != _git_head(historical_contract_root):
        raise PrivateSuiteError(f"lane checkpoint contract commit changed: {path}")
    if payload.get("historical_market_commit") != _git_head(historical_market_root):
        raise PrivateSuiteError(f"lane checkpoint market commit changed: {path}")
    if payload.get("requirements_ci_lock_sha256") != _sha256(
        current_root / "requirements-ci.lock"
    ):
        raise PrivateSuiteError(f"lane checkpoint dependency lock changed: {path}")
    if payload.get("runtime") != _runtime_versions(spec.python):
        raise PrivateSuiteError(f"lane checkpoint runtime changed: {path}")
    if payload.get("frozen_runtime") != frozen_versions:
        raise PrivateSuiteError(f"lane checkpoint frozen runtime changed: {path}")
    log_path = Path(lane["log_path"])
    if log_path != expected_log_path or not log_path.is_file() or log_path.is_symlink():
        raise PrivateSuiteError(f"lane checkpoint log is unavailable: {path}")
    log_bytes = log_path.stat().st_size
    log_sha256 = _sha256(log_path)
    if lane.get("log_bytes") != log_bytes or lane.get("log_sha256") != log_sha256:
        raise PrivateSuiteError(f"lane checkpoint log changed: {path}")
    expected_command = (
        str(spec.python),
        "-m",
        "pytest",
        "-q",
        *spec.extra_args,
        *spec.tests,
    )
    if tuple(lane.get("command", ())) != expected_command:
        raise PrivateSuiteError(f"lane checkpoint command changed: {path}")
    if tuple(lane.get("tests", ())) != spec.tests:
        raise PrivateSuiteError(f"lane checkpoint test selection changed: {path}")
    if legacy_inventory_checkpoint:
        if _is_current_lane(spec.name):
            raise PrivateSuiteError(
                f"monolithic current checkpoint is incompatible with sharded specs: {path}"
            )
        if spec.name not in LEGACY_CHECKPOINT_LANES:
            raise PrivateSuiteError(f"legacy checkpoint is incompatible with this lane: {path}")
    elif payload.get("test_inventory_sha256") != _inventory_sha256(
        tuple(lane.get("tests", ()))
    ):
        raise PrivateSuiteError(f"lane checkpoint test inventory digest mismatch: {path}")
    return LaneReceipt(
        name=lane["name"],
        pass_number=lane["pass_number"],
        cwd=lane["cwd"],
        python=lane["python"],
        tests=tuple(lane["tests"]),
        command=tuple(lane["command"]),
        elapsed_seconds=float(lane["elapsed_seconds"]),
        log_path=lane["log_path"],
        log_bytes=log_bytes,
        log_sha256=log_sha256,
        resumed_from_checkpoint=True,
    )


def _run_pass(
    *,
    pass_number: int,
    specs: Sequence[LaneSpec],
    receipt: Path,
    current_root: Path,
    historical_contract_root: Path,
    historical_market_root: Path,
    frozen_versions: dict[str, str],
    lcc_root: Path,
    base_environment: dict[str, str],
    workers: int,
    resume: bool,
) -> list[LaneReceipt]:
    workspace_parent = receipt.parent / f".{receipt.stem}-workspaces-pass-{pass_number}"
    workspace_parent.mkdir(parents=True, exist_ok=True)
    receipts_by_name: dict[str, LaneReceipt] = {}
    active: dict[object, tuple[LaneSpec, LaneWorkspace, Path, Path]] = {}
    failures: list[str] = []
    try:
        for spec in specs:
            checkpoint = _checkpoint_path(receipt, pass_number, spec.name)
            lane_log = _lane_log_path(receipt, pass_number, spec.name)
            if spec.name.startswith("current-shard-"):
                monolithic = _checkpoint_path(receipt, pass_number, "current")
                if monolithic.exists():
                    raise PrivateSuiteError(
                        "monolithic current checkpoint is incompatible with sharded specs: "
                        f"{monolithic}"
                    )
            if checkpoint.exists():
                if not resume:
                    raise PrivateSuiteError(
                        f"lane checkpoint exists, use --resume: {checkpoint}"
                    )
                receipts_by_name[spec.name] = _load_checkpoint(
                    path=checkpoint,
                    receipt=receipt,
                    expected_log_path=lane_log,
                    pass_number=pass_number,
                    spec=spec,
                    current_root=current_root,
                    historical_contract_root=historical_contract_root,
                    historical_market_root=historical_market_root,
                    frozen_versions=frozen_versions,
                )
                print(f"resumed {spec.name} pass {pass_number} from checkpoint")
                continue
            workspace = _make_lane_workspace(
                spec=spec,
                current_root=current_root,
                workspace_parent=workspace_parent,
            )
            active[spec.name] = (spec, workspace, checkpoint, lane_log)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_lane,
                    name=spec.name,
                    pass_number=pass_number,
                    cwd=workspace.cwd,
                    python=spec.python,
                    tests=spec.tests,
                    extra_args=spec.extra_args,
                    environment=_lane_environment(
                        base_environment,
                        private_root=workspace.private_root,
                        lcc_root=lcc_root,
                    ),
                    log_path=lane_log,
                ): (spec, workspace, checkpoint, lane_log)
                for spec, workspace, checkpoint, lane_log in active.values()
            }
            for future in as_completed(futures):
                spec, workspace, checkpoint, lane_log = futures[future]
                try:
                    lane_receipt = future.result()
                    _write_checkpoint(
                        checkpoint,
                        _checkpoint_payload(
                            receipt=receipt,
                            pass_number=pass_number,
                            spec=spec,
                            lane_receipt=lane_receipt,
                            current_root=current_root,
                            historical_contract_root=historical_contract_root,
                            historical_market_root=historical_market_root,
                            frozen_versions=frozen_versions,
                        ),
                    )
                    receipts_by_name[spec.name] = lane_receipt
                    print(
                        f"completed {spec.name} pass {pass_number} "
                        f"in {lane_receipt.elapsed_seconds:.1f}s"
                    )
                except BaseException as exc:
                    failures.append(f"{spec.name}: {exc}")
    finally:
        for _, workspace, _, _ in active.values():
            _remove_workspace(workspace)
        with suppress(OSError):
            workspace_parent.rmdir()
    if failures:
        raise PrivateSuiteError(
            f"private suite pass {pass_number} failed: {', '.join(failures)}"
        )
    if set(receipts_by_name) != {spec.name for spec in specs}:
        missing = {spec.name for spec in specs} - set(receipts_by_name)
        raise PrivateSuiteError(f"private suite pass {pass_number} is incomplete: {missing}")
    return [receipts_by_name[spec.name] for spec in specs]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-root", type=Path, required=True)
    parser.add_argument("--historical-contract-root", type=Path, required=True)
    parser.add_argument("--historical-market-root", type=Path, required=True)
    parser.add_argument("--current-python", type=Path, required=True)
    parser.add_argument("--frozen-python", type=Path, required=True)
    parser.add_argument("--lcc-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--passes", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--current-shards",
        type=int,
        default=None,
        help="current-test shards; defaults to one shard per CPU",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="parallel lane workers; defaults to one worker per lane",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse verified lane checkpoints from an interrupted run",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    current_root = args.current_root.resolve()
    contract_root = args.historical_contract_root.resolve()
    market_root = args.historical_market_root.resolve()
    current_python = _absolute_executable(args.current_python)
    frozen_python = _absolute_executable(args.frozen_python)
    lcc_root = args.lcc_root.resolve()
    receipt = args.receipt.resolve()
    if args.workers is not None and args.workers < 1:
        raise PrivateSuiteError("--workers must be positive")
    if args.current_shards is not None and args.current_shards < 1:
        raise PrivateSuiteError("--current-shards must be positive")
    if receipt.exists() and not args.resume:
        raise PrivateSuiteError(
            f"receipt already exists, choose a new path or use --resume: {receipt}"
        )
    current_tests, frozen_versions = _verify_inputs(
        current_root=current_root,
        contract_root=contract_root,
        market_root=market_root,
        current_python=current_python,
        frozen_python=frozen_python,
        lcc_root=lcc_root,
    )
    environment = dict(os.environ)
    specs = _lane_specs(
        current_root=current_root,
        contract_root=contract_root,
        market_root=market_root,
        current_python=current_python,
        frozen_python=frozen_python,
        current_tests=current_tests,
    )
    current_shards = min(
        args.current_shards or max(1, os.cpu_count() or 1),
        len(current_tests),
    )
    expanded_specs: list[LaneSpec] = []
    for spec in specs:
        expanded_specs.extend(
            _expand_current_spec(
                spec=spec,
                root=current_root,
                shard_count=current_shards,
            )
        )
    specs = tuple(expanded_specs)
    workers = args.workers or min(len(specs), max(1, os.cpu_count() or 1))
    workers = min(workers, len(specs))
    receipts: list[LaneReceipt] = []
    for pass_number in range(1, args.passes + 1):
        receipts.extend(
            _run_pass(
                pass_number=pass_number,
                specs=specs,
                receipt=receipt,
                current_root=current_root,
                historical_contract_root=contract_root,
                historical_market_root=market_root,
                frozen_versions=frozen_versions,
                lcc_root=lcc_root,
                base_environment=environment,
                workers=workers,
                resume=args.resume,
            )
        )
    payload = {
        "schema_version": "scryglass:private-research-suite-receipt:v1",
        "status": "passed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "passes": args.passes,
        "current_shards": current_shards,
        "parallel_workers": workers,
        "resumed": args.resume,
        "current_commit": _git_head(current_root),
        "historical_contract_commit": HISTORICAL_CONTRACT_COMMIT,
        "historical_market_commit": HISTORICAL_MARKET_COMMIT,
        "requirements_ci_lock_sha256": _sha256(current_root / "requirements-ci.lock"),
        "current_runtime": _runtime_versions(current_python),
        "frozen_runtime": frozen_versions,
        "test_files_total": len(_test_files(current_root)),
        "current_test_files": len(current_tests),
        "current_test_inventory_sha256": _inventory_sha256(current_tests),
        "frozen_test_files": len(FROZEN_RUNTIME_TESTS) + len(HISTORICAL_CONTRACT_TESTS),
        "lanes": [asdict(item) for item in receipts],
    }
    unsigned = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["receipt_sha256"] = hashlib.sha256(unsigned).hexdigest()
    receipt.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if receipt.exists():
        try:
            existing = json.loads(receipt.read_text())
        except (OSError, ValueError) as exc:
            raise PrivateSuiteError(f"invalid existing suite receipt: {receipt}") from exc
        if existing != payload:
            raise PrivateSuiteError(f"suite receipt already exists: {receipt}")
    else:
        temporary = receipt.with_name(f".{receipt.name}.{os.getpid()}.tmp")
        temporary.write_text(encoded)
        os.replace(temporary, receipt)
    print(f"private research suite receipt: {receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
