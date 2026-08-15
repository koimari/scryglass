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
