from __future__ import annotations

import hashlib
import json
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
        current_tests=("tests/test_b.py", "tests/test_a.py"),
    )
    assert tuple(spec.name for spec in specs) == (
        "current",
        "frozen-runtime",
        "historical-contract",
        "historical-market",
    )
    assert specs[0].tests == ("tests/test_b.py", "tests/test_a.py")
    assert specs[0].extra_args == ()
    assert specs[1].tests == suite.FROZEN_RUNTIME_TESTS
    assert specs[2].tests == suite.HISTORICAL_CONTRACT_TESTS
    assert specs[3].tests == suite.HISTORICAL_MARKET_TESTS


def test_private_suite_current_shards_are_deterministic_and_complete(
    tmp_path: Path,
) -> None:
    files = ("tests/test_a.py", "tests/test_b.py", "tests/test_c.py")
    for index, path in enumerate(files):
        file_path = tmp_path / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"x" * (index + 1))
    first = suite._partition_current_tests(
        root=tmp_path,
        tests=files,
        shard_count=2,
    )
    second = suite._partition_current_tests(
        root=tmp_path,
        tests=files,
        shard_count=2,
    )
    assert first == second
    assert sorted(path for shard in first for path in shard) == sorted(files)
    assert len({path for shard in first for path in shard}) == len(files)


def test_private_suite_current_shard_names_and_commands(
    tmp_path: Path,
) -> None:
    spec = suite.LaneSpec(
        name="current",
        source_root=tmp_path,
        python=Path("/python"),
        tests=("tests/test_a.py", "tests/test_b.py"),
        extra_args=(),
    )
    for path in spec.tests:
        file_path = tmp_path / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("def test_ok(): pass\n")
    expanded = suite._expand_current_spec(spec=spec, root=tmp_path, shard_count=2)
    assert tuple(item.name for item in expanded) == (
        "current-shard-01",
        "current-shard-02",
    )
    assert sorted(path for item in expanded for path in item.tests) == sorted(
        spec.tests
    )


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


def test_private_suite_resumes_green_shards_after_later_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "current"
    source.mkdir()
    (source / "requirements-ci.lock").write_text("lock")
    contract = tmp_path / "contract"
    market = tmp_path / "market"
    contract.mkdir()
    market.mkdir()
    for root in (contract, market):
        (root / "requirements-ci.lock").write_text("lock")
    specs = (
        suite.LaneSpec(
            name="current-shard-01",
            source_root=source,
            python=Path("/python"),
            tests=("tests/test_a.py",),
            extra_args=(),
        ),
        suite.LaneSpec(
            name="current-shard-02",
            source_root=source,
            python=Path("/python"),
            tests=("tests/test_b.py",),
            extra_args=(),
        ),
    )
    monkeypatch.setattr(suite, "_git_head", lambda _root: "head")
    monkeypatch.setattr(
        suite,
        "_runtime_versions",
        lambda _python: dict(suite.FROZEN_VERSIONS),
    )
    monkeypatch.setattr(suite, "_sha256", lambda _path: "a" * 64)
    calls: list[str] = []
    failed = False

    def fake_run_lane(**kwargs: object) -> suite.LaneReceipt:
        nonlocal failed
        name = str(kwargs["name"])
        calls.append(name)
        log_path = kwargs["log_path"]
        assert isinstance(log_path, Path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"{name}\n")
        if name == "current-shard-02" and not failed:
            failed = True
            raise suite.PrivateSuiteError("synthetic shard failure")
        python = kwargs["python"]
        assert isinstance(python, Path)
        tests = tuple(kwargs["tests"])
        extra_args = tuple(kwargs["extra_args"])
        return suite.LaneReceipt(
            name=name,
            pass_number=1,
            cwd=str(kwargs["cwd"]),
            python=str(python),
            tests=tests,
            command=(str(python), "-m", "pytest", "-q", *extra_args, *tests),
            elapsed_seconds=0.1,
            log_path=str(log_path),
            log_bytes=log_path.stat().st_size,
            log_sha256="a" * 64,
        )

    monkeypatch.setattr(suite, "_run_lane", fake_run_lane)
    receipt = tmp_path / "suite.json"
    run_kwargs = dict(
        pass_number=1,
        specs=specs,
        receipt=receipt,
        current_root=source,
        historical_contract_root=contract,
        historical_market_root=market,
        frozen_versions=dict(suite.FROZEN_VERSIONS),
        lcc_root=tmp_path,
        base_environment={},
        workers=2,
    )
    with pytest.raises(suite.PrivateSuiteError, match="private suite pass 1 failed"):
        suite._run_pass(resume=False, **run_kwargs)
    assert calls.count("current-shard-01") == 1
    assert calls.count("current-shard-02") == 1
    assert suite._checkpoint_path(receipt, 1, "current-shard-01").is_file()

    resumed = suite._run_pass(resume=True, **run_kwargs)
    assert [item.name for item in resumed] == [
        "current-shard-01",
        "current-shard-02",
    ]
    assert resumed[0].resumed_from_checkpoint is True
    assert calls.count("current-shard-01") == 1
    assert calls.count("current-shard-02") == 2


def test_private_suite_accepts_legacy_non_current_checkpoint_after_full_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = [tmp_path / name for name in ("current", "contract", "market")]
    for root in roots:
        root.mkdir()
        (root / "requirements-ci.lock").write_text("lock")
    monkeypatch.setattr(suite, "_git_head", lambda _root: "head")
    monkeypatch.setattr(
        suite,
        "_runtime_versions",
        lambda _python: dict(suite.FROZEN_VERSIONS),
    )
    monkeypatch.setattr(suite, "_sha256", lambda _path: "a" * 64)
    receipt = tmp_path / "suite.json"
    log_path = tmp_path / "suite-pass-1-frozen-runtime.log"
    log_path.write_text("passed")
    spec = suite.LaneSpec(
        name="frozen-runtime",
        source_root=roots[0],
        python=Path("/python"),
        tests=("tests/frozen.py",),
        extra_args=(),
    )
    lane = suite.LaneReceipt(
        name=spec.name,
        pass_number=1,
        cwd=str(tmp_path / "lane-copy"),
        python=str(spec.python),
        tests=spec.tests,
        command=(str(spec.python), "-m", "pytest", "-q", *spec.tests),
        elapsed_seconds=1.0,
        log_path=str(log_path),
        log_bytes=log_path.stat().st_size,
        log_sha256="a" * 64,
    )
    payload = suite._checkpoint_payload(
        receipt=receipt,
        pass_number=1,
        spec=spec,
        lane_receipt=lane,
        current_root=roots[0],
        historical_contract_root=roots[1],
        historical_market_root=roots[2],
        frozen_versions=dict(suite.FROZEN_VERSIONS),
    )
    payload.pop("test_inventory_sha256")
    payload["checkpoint_sha256"] = hashlib.sha256(
        suite._canonical_json(
            {key: value for key, value in payload.items() if key != "checkpoint_sha256"}
        )
    ).hexdigest()
    checkpoint = suite._checkpoint_path(receipt, 1, spec.name)
    checkpoint.write_text(json.dumps(payload))
    loaded = suite._load_checkpoint(
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
    assert loaded.resumed_from_checkpoint is True
    assert loaded.tests == spec.tests


def test_private_suite_rejects_legacy_current_checkpoint_for_sharded_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = [tmp_path / name for name in ("current", "contract", "market")]
    for root in roots:
        root.mkdir()
        (root / "requirements-ci.lock").write_text("lock")
    receipt = tmp_path / "suite.json"
    monolithic = suite._checkpoint_path(receipt, 1, "current")
    monolithic.write_text("sealed")
    spec = suite.LaneSpec(
        name="current-shard-01",
        source_root=roots[0],
        python=Path("/python"),
        tests=("tests/test_a.py",),
        extra_args=(),
    )
    with pytest.raises(suite.PrivateSuiteError, match="monolithic current checkpoint"):
        suite._run_pass(
            pass_number=1,
            specs=(spec,),
            receipt=receipt,
            current_root=roots[0],
            historical_contract_root=roots[1],
            historical_market_root=roots[2],
            frozen_versions=dict(suite.FROZEN_VERSIONS),
            lcc_root=tmp_path,
            base_environment={},
            workers=1,
            resume=True,
        )
