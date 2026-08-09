"""Repository pins for side-neutral review and ledger-admission code."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
SCHEMA_VERSION = "scryglass:side-neutral-collection-implementation-registry:v1"
REGISTERED_RECORDS = (
    {
        "locator": "lol_kills/v2/ratings/player/side_neutral_protocol_review_v1.py",
        "bytes": 9637,
        "raw_sha256": "5dc70f5d1994d2cf590d3526af5c11d60490739a9202efc22cc2a861d10c6230",
        "role": "external_review_validation",
    },
    {
        "locator": "lol_kills/v2/market/side_neutral_ledger_v1.py",
        "bytes": 22747,
        "raw_sha256": "034a5bcdac01def8ee0f01ab8e50c108c8e4cc7ff393bc4cf4418450be9ec02a",
        "role": "post_review_bundle_admission",
    },
)


class SideNeutralCollectionImplementationRegistryError(RuntimeError):
    """A repository-pinned review or admission implementation drifted."""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_registered_side_neutral_collection_implementation(
    *, root: Path = ROOT
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for expected in REGISTERED_RECORDS:
        path = root / expected["locator"]
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != expected["bytes"]
            or _sha256_path(path) != expected["raw_sha256"]
        ):
            raise SideNeutralCollectionImplementationRegistryError(
                f"registered side-neutral implementation drifted: {expected['locator']}"
            )
        records.append(dict(expected))
    return {
        "schema_version": SCHEMA_VERSION,
        "records": records,
        "repository_code_pin_valid": True,
        "independent_review_present": False,
        "prospective_collection_authorized": False,
        "self_authorizing": False,
    }


__all__ = [
    "REGISTERED_RECORDS",
    "SCHEMA_VERSION",
    "SideNeutralCollectionImplementationRegistryError",
    "validate_registered_side_neutral_collection_implementation",
]
