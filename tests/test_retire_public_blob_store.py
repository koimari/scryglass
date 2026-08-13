from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from lol_kills.export.blob_retention import BlobIdentity
from tools.retire_public_blob_store import (
    CONFIRMATION,
    RETIRED_STORE_ID,
    RetiredStoreError,
    inventory,
    inventory_sha256,
    retire,
)


def test_module_entrypoint_resolves_repository_packages() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "tools.retire_public_blob_store", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": ""},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Delete the retired Scryglass public Blob store" in result.stdout


class FakeTransport:
    def __init__(
        self, rows: list[BlobIdentity], *, store_id: str = RETIRED_STORE_ID
    ) -> None:
        self.store_id = store_id
        self.rows = list(rows)
        self.deleted: list[str] = []

    def list_page(self, store_id, *, cursor, limit, deadline_epoch):
        assert store_id == RETIRED_STORE_ID
        assert cursor is None
        assert limit == 1000
        assert deadline_epoch > 0
        return {
            "storeId": self.store_id,
            "blobs": [
                {"pathname": row.pathname, "size": row.size, "etag": row.etag}
                for row in self.rows
            ],
            "hasMore": False,
            "cursor": None,
        }

    def delete_if_match(self, store_id, pathname, *, etag, deadline_epoch):
        assert store_id == RETIRED_STORE_ID
        assert deadline_epoch > 0
        match = next(
            (row for row in self.rows if row.pathname == pathname and row.etag == etag),
            None,
        )
        if match is None:
            return None
        self.rows.remove(match)
        self.deleted.append(pathname)
        return match


def test_retired_store_plan_is_read_only() -> None:
    transport = FakeTransport(
        [
            BlobIdentity("packs/release/manifest.json", 10, "a"),
            BlobIdentity("state/snapshot.tar.gz", 20, "b"),
        ]
    )
    result = retire(transport, execute=False, confirmation="")
    assert result["objects"] == 2
    assert result["bytes"] == 30
    assert result["inventory_sha256"] == inventory_sha256(transport.rows)
    assert "pathname" not in str(result)
    assert transport.deleted == []


def test_retired_store_cleanup_requires_exact_target_and_confirmation() -> None:
    row = BlobIdentity("tierlists/index-v1.json", 10, "a")
    with pytest.raises(RetiredStoreError, match="different Blob store"):
        retire(FakeTransport([row], store_id="other"), execute=False, confirmation="")
    with pytest.raises(RetiredStoreError, match="exact retired-store confirmation"):
        retire(FakeTransport([row]), execute=True, confirmation="wrong")


def test_retired_store_cleanup_rejects_unknown_paths_before_deletion() -> None:
    transport = FakeTransport([BlobIdentity("unrelated/customer.json", 10, "a")])
    with pytest.raises(RetiredStoreError, match="unexpected path"):
        inventory(transport, deadline_epoch=100)
    assert transport.deleted == []


def test_retired_store_cleanup_deletes_each_bound_identity_and_proves_empty() -> None:
    transport = FakeTransport(
        [
            BlobIdentity("rankings/tierlists.json", 10, "a"),
            BlobIdentity("packs/release/features/profile_records.json", 20, "b"),
        ]
    )
    expected = inventory_sha256(transport.rows)
    result = retire(
        transport,
        execute=True,
        confirmation=CONFIRMATION,
        expected_inventory_sha256=expected,
    )
    assert result["deleted"] == 2
    assert result["verified_empty"] is True
    assert result["inventory_sha256"] == expected
    assert result["post_inventory_sha256"] == inventory_sha256([])
    assert transport.rows == []


def test_retired_store_cleanup_requires_the_dry_inventory_digest() -> None:
    row = BlobIdentity("tierlists/index-v1.json", 10, "a")
    transport = FakeTransport([row])
    with pytest.raises(RetiredStoreError, match="dry inventory digest"):
        retire(transport, execute=True, confirmation=CONFIRMATION)
    assert transport.deleted == []


def test_retired_store_cleanup_rejects_inventory_drift_before_deletion() -> None:
    original = BlobIdentity("tierlists/index-v1.json", 10, "a")
    transport = FakeTransport(
        [
            original,
            BlobIdentity("state/new-snapshot.tar.gz", 20, "b"),
        ]
    )
    with pytest.raises(RetiredStoreError, match="changed after the dry run"):
        retire(
            transport,
            execute=True,
            confirmation=CONFIRMATION,
            expected_inventory_sha256=inventory_sha256([original]),
        )
    assert transport.deleted == []


def test_retired_store_inventory_requires_boolean_pagination_state() -> None:
    class InvalidPaginationTransport(FakeTransport):
        def list_page(self, store_id, *, cursor, limit, deadline_epoch):
            page = super().list_page(
                store_id,
                cursor=cursor,
                limit=limit,
                deadline_epoch=deadline_epoch,
            )
            page["hasMore"] = "false"
            return page

    with pytest.raises(RetiredStoreError, match="pagination state"):
        inventory(InvalidPaginationTransport([]), deadline_epoch=100)
