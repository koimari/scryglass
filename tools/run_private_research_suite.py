"""Run the complete private research gate across current and frozen lanes."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence


HISTORICAL_CONTRACT_COMMIT = "94e5cec16adcf77a45f429e666fb3b5f54f40bd9"
HISTORICAL_MARKET_COMMIT = "87bcff462f6005269b94fd9e604c9e8b2b5ec9c0"

FROZEN_RUNTIME_TESTS = (
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
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        for block in iter(process.stdout.readline, b""):
            log.write(block)
            log.flush()
            sys.stdout.buffer.write(block)
            sys.stdout.buffer.flush()
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
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    current_root = args.current_root.resolve()
    contract_root = args.historical_contract_root.resolve()
    market_root = args.historical_market_root.resolve()
    current_python = args.current_python.resolve()
    frozen_python = args.frozen_python.resolve()
    lcc_root = args.lcc_root.resolve()
    receipt = args.receipt.resolve()
    current_tests, frozen_versions = _verify_inputs(
        current_root=current_root,
        contract_root=contract_root,
        market_root=market_root,
        current_python=current_python,
        frozen_python=frozen_python,
        lcc_root=lcc_root,
    )
    environment = dict(os.environ)
    environment.update(
        {
            "SCRYGLASS_LCC_REPO": str(lcc_root),
            "SCRYGLASS_PRIVATE_TEST_ROOT": str(current_root),
        }
    )
    receipts: list[LaneReceipt] = []
    for pass_number in range(1, args.passes + 1):
        lane_specs = (
            (
                "current",
                current_root,
                current_python,
                (),
                tuple(
                    f"--ignore={path}"
                    for path in (*FROZEN_RUNTIME_TESTS, *HISTORICAL_CONTRACT_TESTS)
                ),
            ),
            ("frozen-runtime", current_root, frozen_python, FROZEN_RUNTIME_TESTS, ()),
            ("historical-contract", contract_root, frozen_python, HISTORICAL_CONTRACT_TESTS, ()),
            ("historical-market", market_root, frozen_python, HISTORICAL_MARKET_TESTS, ()),
        )
        for name, cwd, python, tests, extra_args in lane_specs:
            lane_log = receipt.with_name(
                f"{receipt.stem}-pass-{pass_number}-{name}.log"
            )
            receipts.append(
                _run_lane(
                    name=name,
                    pass_number=pass_number,
                    cwd=cwd,
                    python=python,
                    tests=tests,
                    extra_args=extra_args,
                    environment=environment,
                    log_path=lane_log,
                )
            )
    payload = {
        "schema_version": "scryglass:private-research-suite-receipt:v1",
        "status": "passed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "passes": args.passes,
        "current_commit": _git_head(current_root),
        "historical_contract_commit": HISTORICAL_CONTRACT_COMMIT,
        "historical_market_commit": HISTORICAL_MARKET_COMMIT,
        "requirements_ci_lock_sha256": _sha256(current_root / "requirements-ci.lock"),
        "current_runtime": _runtime_versions(current_python),
        "frozen_runtime": frozen_versions,
        "test_files_total": len(_test_files(current_root)),
        "current_test_files": len(current_tests),
        "frozen_test_files": len(FROZEN_RUNTIME_TESTS) + len(HISTORICAL_CONTRACT_TESTS),
        "lanes": [asdict(item) for item in receipts],
    }
    unsigned = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["receipt_sha256"] = hashlib.sha256(unsigned).hexdigest()
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"private research suite receipt: {receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
