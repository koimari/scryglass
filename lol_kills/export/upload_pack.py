#!/usr/bin/env python3
"""Retired public Blob command with fail-closed compatibility helpers.

Public application packs publish only through the private, active-release-bound
Supabase Storage contract in :mod:`lol_kills.export.supabase_publication`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACK_ROOT = ROOT / "output" / "public_pack"
ATLAS_PUBLIC = ROOT / "apps" / "scryglass" / "public" / "packs"
BLOB_TIMEOUT_SECONDS = 45.0
PACK_ID_RE = re.compile(r"^v\d{4}\.\d{2}\.\d{2}\.\d{6}$")
PUBLIC_PACK_BLOB_DISABLED = (
    "public pack Blob publication is disabled; use private Supabase Storage publication"
)


def _pack_id(value: object) -> str:
    pack_id = str(value or "")
    if not PACK_ID_RE.fullmatch(pack_id):
        raise RuntimeError("pack ID is invalid")
    return pack_id


def _pack_directory(pack_root: Path, pack_id: str) -> Path:
    root = pack_root.expanduser().resolve(strict=True)
    unresolved = root / _pack_id(pack_id)
    if unresolved.is_symlink():
        raise RuntimeError("pack directory is invalid")
    candidate = unresolved.resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise RuntimeError("pack directory leaves the configured pack root") from error
    if not candidate.is_dir():
        raise RuntimeError("pack directory is invalid")
    return candidate


def _load_manifest(pack_dir: Path) -> dict[str, Any]:
    return json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))


def _public_pack_blob_disabled() -> None:
    raise RuntimeError(PUBLIC_PACK_BLOB_DISABLED)


def _blob_put(
    token: str,
    pathname: str,
    data: bytes,
    content_type: str,
    *,
    cache_control: str | None = None,
    allow_overwrite: bool = False,
) -> str:
    del token, pathname, data, content_type, cache_control, allow_overwrite
    _public_pack_blob_disabled()


def upload_to_blob(pack_dir: Path, pack_id: str, token: str) -> dict[str, str]:
    """Reject the retired public Vercel Blob pack lane."""

    del pack_dir, pack_id, token
    _public_pack_blob_disabled()


def write_atlas_manifest(
    pack_id: str,
    manifest: dict[str, Any],
    *,
    base_url: str,
) -> Path:
    del pack_id, manifest, base_url
    _public_pack_blob_disabled()


def write_vercel_pack_ignore(
    pack_id: str,
    *,
    public_root: Path = ATLAS_PUBLIC,
    ignore_path: Path | None = None,
) -> Path:
    del pack_id, public_root, ignore_path
    _public_pack_blob_disabled()


def publish_blob_pointers(
    token: str,
    pack_id: str,
    manifest: dict[str, Any],
    *,
    base_url: str,
) -> dict[str, str]:
    del token, pack_id, manifest, base_url
    _public_pack_blob_disabled()


def restore_blob_pointers(
    token: str,
    previous: dict[str, str | None],
) -> None:
    del token, previous
    _public_pack_blob_disabled()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Retired public pack publication command")
    ap.add_argument("--pack-root", type=Path, default=DEFAULT_PACK_ROOT)
    ap.add_argument("--pack-id", default=None, help="Default: read output/public_pack/latest.json")
    ap.add_argument(
        "--local-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    ap.parse_args(argv)
    raise SystemExit(PUBLIC_PACK_BLOB_DISABLED)


if __name__ == "__main__":
    raise SystemExit(main())
