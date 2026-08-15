from __future__ import annotations

from pathlib import Path

import pytest

from tools import run_private_research_suite as suite


def test_private_suite_partitions_every_test_file() -> None:
    root = Path(__file__).resolve().parents[1]
    all_tests = set(suite._test_files(root))
    frozen = set(suite.FROZEN_RUNTIME_TESTS) | set(
        suite.HISTORICAL_CONTRACT_TESTS
    )
    assert frozen <= all_tests
    assert (all_tests - frozen) | frozen == all_tests
    assert not (all_tests - frozen) & frozen


def test_private_suite_frozen_commits_and_runtime_are_exact() -> None:
    assert suite.HISTORICAL_CONTRACT_COMMIT == (
        "94e5cec16adcf77a45f429e666fb3b5f54f40bd9"
    )
    assert suite.HISTORICAL_MARKET_COMMIT == (
        "87bcff462f6005269b94fd9e604c9e8b2b5ec9c0"
    )
    assert suite.FROZEN_VERSIONS == {
        "python": "3.9.6",
        "numpy": "2.0.2",
        "pandas": "2.3.3",
        "pyarrow": "21.0.0",
        "scipy": "1.13.1",
    }


def test_private_suite_rejects_missing_frozen_test() -> None:
    with pytest.raises(suite.PrivateSuiteError, match="frozen test inventory"):
        suite._partition_tests(())


def test_private_suite_preserves_virtual_environment_symlink(
    tmp_path: Path,
) -> None:
    target = tmp_path / "python-target"
    target.write_text("")
    virtual_python = tmp_path / "venv-python"
    virtual_python.symlink_to(target)
    assert suite._absolute_executable(virtual_python) == virtual_python


def test_private_suite_lane_specs_keep_exact_test_partition() -> None:
    roots = [Path("/current"), Path("/contract"), Path("/market")]
    specs = suite._lane_specs(
        current_root=roots[0],
        contract_root=roots[1],
        market_root=roots[2],
        current_python=Path("/python-current"),
        frozen_python=Path("/python-frozen"),
    )
    assert tuple(spec.name for spec in specs) == (
        "current",
        "frozen-runtime",
        "historical-contract",
        "historical-market",
    )
    assert specs[0].tests == ()
    assert all(arg.startswith("--ignore=tests/") for arg in specs[0].extra_args)
    assert specs[1].tests == suite.FROZEN_RUNTIME_TESTS
    assert specs[2].tests == suite.HISTORICAL_CONTRACT_TESTS
    assert specs[3].tests == suite.HISTORICAL_MARKET_TESTS


def test_private_suite_lane_copy_does_not_share_regular_file_writes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    original = source / "input.txt"
    original.write_text("source")
    destination = tmp_path / "destination"
    suite._copy_on_write_tree(source, destination)

    (destination / "input.txt").write_text("lane")
    assert original.read_text() == "source"


def test_private_suite_lane_environment_isolated_root(tmp_path: Path) -> None:
    environment = suite._lane_environment(
        {"PATH": "/usr/bin", "SCRYGLASS_PRIVATE_TEST_ROOT": "/old"},
        private_root=tmp_path / "lane-private",
        lcc_root=tmp_path / "lcc",
    )
    assert environment["SCRYGLASS_PRIVATE_TEST_ROOT"] == str(
        tmp_path / "lane-private"
    )
    assert environment["SCRYGLASS_LCC_REPO"] == str(tmp_path / "lcc")
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"


def test_private_suite_checkpoint_rejects_log_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = [tmp_path / name for name in ("current", "contract", "market")]
    for root in roots:
        root.mkdir()
    lock = roots[0] / "requirements-ci.lock"
    lock.write_text("lock")
    monkeypatch.setattr(suite, "_git_head", lambda _root: "head")
    monkeypatch.setattr(
        suite,
        "_runtime_versions",
        lambda _python: dict(suite.FROZEN_VERSIONS),
    )
    monkeypatch.setattr(suite, "_sha256", lambda _path: "a" * 64)
    receipt = tmp_path / "suite.json"
    log_path = tmp_path / "suite-pass-1-current.log"
    log_path.write_text("passed")
    spec = suite.LaneSpec(
        name="current",
        source_root=roots[0],
        python=Path("/python"),
        tests=(),
        extra_args=("--ignore=tests/frozen.py",),
    )
    lane = suite.LaneReceipt(
        name="current",
        pass_number=1,
        cwd=str(tmp_path / "lane-copy"),
        python=str(spec.python),
        tests=(),
        command=(
            str(spec.python),
            "-m",
            "pytest",
            "-q",
            *spec.extra_args,
        ),
        elapsed_seconds=1.0,
        log_path=str(log_path),
        log_bytes=log_path.stat().st_size,
        log_sha256="a" * 64,
    )
    checkpoint = suite._checkpoint_path(receipt, 1, "current")
    suite._write_checkpoint(
        checkpoint,
        suite._checkpoint_payload(
            receipt=receipt,
            pass_number=1,
            spec=spec,
            lane_receipt=lane,
            current_root=roots[0],
            historical_contract_root=roots[1],
            historical_market_root=roots[2],
            frozen_versions=dict(suite.FROZEN_VERSIONS),
        ),
    )
    log_path.write_text("tampered")
    with pytest.raises(suite.PrivateSuiteError, match="log changed"):
        suite._load_checkpoint(
            path=checkpoint,
            receipt=receipt,
            expected_log_path=log_path,
            pass_number=1,
            spec=spec,
            current_root=roots[0],
            historical_contract_root=roots[1],
            historical_market_root=roots[2],
            frozen_versions=dict(suite.FROZEN_VERSIONS),
        )
