"""Delete the retired Scryglass public Blob store after an exact inventory check."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import asdict
from typing import Protocol

from lol_kills.export.blob_retention import BlobIdentity
from lol_kills.export.vercel_blob_transport import VercelBlobTransport

RETIRED_STORE_ID = "97gks2fobqkgppwx"
ALLOWED_RETIRED_PREFIXES = ("packs/", "rankings/", "state/", "tierlists/")
CONFIRMATION = f"RETIRE-{RETIRED_STORE_ID}"
RECEIPT_SCHEMA_VERSION = "scryglass:retired-public-blob:v1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class RetiredStoreError(RuntimeError):
    """The retired store cleanup could not prove its target or result."""


class Transport(Protocol):
    store_id: str

    def list_page(
        self,
        store_id: str,
        *,
        cursor: str | None,
        limit: int,
        deadline_epoch: int,
    ) -> dict[str, object]: ...

    def delete_if_match(
        self,
        store_id: str,
        pathname: str,
        *,
        etag: str,
        deadline_epoch: int,
    ) -> BlobIdentity | None: ...


def inventory(transport: Transport, *, deadline_epoch: int) -> list[BlobIdentity]:
    """Read the full authenticated store inventory without accepting duplicates."""

    cursor: str | None = None
    seen_cursors: set[str] = set()
    by_path: dict[str, BlobIdentity] = {}
    while True:
        page = transport.list_page(
            RETIRED_STORE_ID,
            cursor=cursor,
            limit=1000,
            deadline_epoch=deadline_epoch,
        )
        if page.get("storeId") != RETIRED_STORE_ID:
            raise RetiredStoreError("the inventory belongs to a different Blob store")
        raw_blobs = page.get("blobs")
        if type(raw_blobs) is not list or len(raw_blobs) > 1000:
            raise RetiredStoreError("the Blob inventory is malformed")
        for raw in raw_blobs:
            if type(raw) is not dict:
                raise RetiredStoreError("the Blob inventory contains a malformed row")
            try:
                item = BlobIdentity(raw["pathname"], raw["size"], raw["etag"])
            except (KeyError, TypeError, ValueError) as error:
                raise RetiredStoreError(
                    "the Blob inventory contains an invalid identity"
                ) from error
            if item.pathname in by_path:
                raise RetiredStoreError("the Blob inventory contains a duplicate path")
            if not item.pathname.startswith(ALLOWED_RETIRED_PREFIXES):
                raise RetiredStoreError(
                    "an unexpected path blocks retired-store cleanup"
                )
            by_path[item.pathname] = item
        has_more = page.get("hasMore")
        next_cursor = page.get("cursor")
        if type(has_more) is not bool:
            raise RetiredStoreError("the Blob inventory pagination state is malformed")
        if not has_more:
            break
        if (
            not isinstance(next_cursor, str)
            or not next_cursor
            or next_cursor in seen_cursors
        ):
            raise RetiredStoreError("the Blob inventory cursor is invalid")
        seen_cursors.add(next_cursor)
        cursor = next_cursor
    return sorted(by_path.values(), key=lambda item: item.pathname)


def inventory_sha256(items: list[BlobIdentity]) -> str:
    raw = json.dumps(
        [asdict(item) for item in sorted(items, key=lambda item: item.pathname)],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def retire(
    transport: Transport,
    *,
    execute: bool,
    confirmation: str,
    expected_inventory_sha256: str = "",
) -> dict[str, object]:
    if transport.store_id != RETIRED_STORE_ID:
        raise RetiredStoreError("the credential belongs to a different Blob store")
    deadline_epoch = int(time.time()) + 3_600
    before = inventory(transport, deadline_epoch=deadline_epoch)
    before_sha256 = inventory_sha256(before)
    result: dict[str, object] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "store_id": RETIRED_STORE_ID,
        "mode": "execute" if execute else "plan",
        "objects": len(before),
        "bytes": sum(item.size for item in before),
        "prefixes": list(ALLOWED_RETIRED_PREFIXES),
        "inventory_sha256": before_sha256,
    }
    if not execute:
        return result
    if confirmation != CONFIRMATION:
        raise RetiredStoreError("the exact retired-store confirmation is required")
    if not SHA256.fullmatch(expected_inventory_sha256):
        raise RetiredStoreError("the dry inventory digest is required")
    if expected_inventory_sha256 != before_sha256:
        raise RetiredStoreError("the Blob inventory changed after the dry run")
    deleted = 0
    for item in before:
        removed = transport.delete_if_match(
            RETIRED_STORE_ID,
            item.pathname,
            etag=item.etag,
            deadline_epoch=deadline_epoch,
        )
        if removed != item:
            raise RetiredStoreError(
                f"conditional deletion was not confirmed: {item.pathname}"
            )
        deleted += 1
    after = inventory(transport, deadline_epoch=deadline_epoch)
    if after:
        raise RetiredStoreError("the retired Blob store is not empty after cleanup")
    return {
        **result,
        "deleted": deleted,
        "verified_empty": True,
        "post_inventory_sha256": inventory_sha256(after),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--expected-inventory-sha256", default="")
    args = parser.parse_args(argv)
    token = os.environ.get("BLOB_READ_WRITE_TOKEN", "")
    if not token:
        raise RetiredStoreError("BLOB_READ_WRITE_TOKEN is required")
    result = retire(
        VercelBlobTransport(token, RETIRED_STORE_ID),
        execute=args.execute,
        confirmation=args.confirmation,
        expected_inventory_sha256=args.expected_inventory_sha256,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
