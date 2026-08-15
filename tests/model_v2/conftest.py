from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil

import pytest

from lol_kills.v2.draft.terminal import future_prediction_ledger as ledger


@pytest.fixture(scope="session")
def private_test_root() -> Path:
    return Path(
        os.environ.get("SCRYGLASS_PRIVATE_TEST_ROOT", Path(".").resolve())
    ).resolve()


@pytest.fixture(scope="session")
def historical_capture_root(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Expose the sealed pre-drift capture sources to historical replay tests."""

    repo_root = Path(".").resolve()
    private_data_root = Path(
        os.environ.get("SCRYGLASS_PRIVATE_TEST_ROOT", repo_root)
    ).resolve()
    root = tmp_path_factory.mktemp("historical-capture-root")
    historical_patch_source = (
        repo_root
        / "tests/model_v2/fixtures/leaguepedia_patch_revisions_v1.py"
    )
    assert hashlib.sha256(historical_patch_source.read_bytes()).hexdigest() == (
        "b9079a64c8dcba4d17c7762edda4c914a7a42b8b7680044910298b69014ebb58"
    )

    (root / "lol_kills/etl").mkdir(parents=True)
    for child in (repo_root / "lol_kills").iterdir():
        if child.name != "etl":
            (root / "lol_kills" / child.name).symlink_to(
                child,
                target_is_directory=child.is_dir(),
            )
    for child in (repo_root / "lol_kills/etl").iterdir():
        if child.name != "leaguepedia_patch_revisions.py":
            (root / "lol_kills/etl" / child.name).symlink_to(
                child,
                target_is_directory=child.is_dir(),
            )
    shutil.copy2(
        historical_patch_source,
        root / "lol_kills/etl/leaguepedia_patch_revisions.py",
    )
    phase_one_ts = repo_root / "tests/model_v2/fixtures/phase-one-ts"
    for source in phase_one_ts.rglob("*"):
        if source.is_file():
            destination = root / source.relative_to(phase_one_ts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    (root / "docs").symlink_to(repo_root / "docs", target_is_directory=True)
    (root / "data/lol/v2").mkdir(parents=True)
    (root / "data/lol/v2/models").symlink_to(
        repo_root / "data/lol/v2/models",
        target_is_directory=True,
    )
    shutil.copytree(
        repo_root / "data/lol/v2/snapshots",
        root / "data/lol/v2/snapshots",
    )
    private_snapshots = private_data_root / "data/lol/v2/snapshots"
    if private_snapshots.is_dir():
        for source in private_snapshots.rglob("*.parquet"):
            destination = root / source.relative_to(private_data_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(source, destination)
    market_root = Path("data/lol/v2/evaluation/match-winner-market-v1")
    (root / market_root.parent).mkdir(parents=True, exist_ok=True)
    shutil.copytree(repo_root / market_root, root / market_root)

    (root / "data/lol/warehouse/parquet").mkdir(parents=True)
    (root / "data/lol/warehouse/raw_grid").symlink_to(
        private_data_root / "data/lol/warehouse/raw_grid",
        target_is_directory=True,
    )
    for filename in ("maps.parquet", "players.parquet"):
        shutil.copy2(
            private_data_root / f"data/lol/warehouse/parquet/{filename}",
            root / f"data/lol/warehouse/parquet/{filename}",
        )
    (root / ledger.PREDICTION_PREFIX).mkdir(parents=True)
    (root / ledger.MAP_START_PREFIX).mkdir(parents=True)
    return root
