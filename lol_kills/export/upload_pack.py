#!/usr/bin/env python3
"""Legacy Blob helpers with public-pack publication disabled.

Public application packs publish only through the private, active-release-bound
Supabase Storage contract in :mod:`lol_kills.export.supabase_publication`.
The generic Blob helper remains for separate snapshot transports.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from lol_kills.net import require_https_url

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


def _is_public_pack_blob_path(pathname: str) -> bool:
    decoded = pathname
    for _ in range(3):
        unquoted = urllib.parse.unquote(decoded)
        if unquoted == decoded:
            break
        decoded = unquoted
    parts: list[str] = []
    for part in decoded.replace("\\", "/").split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return bool(parts) and parts[0].casefold() == "packs"


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
    """Upload a non-pack snapshot through the legacy Blob transport."""

    if _is_public_pack_blob_path(pathname):
        _public_pack_blob_disabled()
    store_id = os.environ.get("BLOB_STORE_ID") or os.environ.get("VERCEL_BLOB_STORE_ID")
    req = urllib.request.Request(
        f"https://blob.vercel-storage.com/{pathname}",
        data=data,
        method="PUT",
        headers={
            "Authorization": f"Bearer {token}",
            "X-API-Version": "7",
            "x-vercel-blob-access": "public",
            **({"x-vercel-blob-store-id": store_id} if store_id else {}),
            "x-add-random-suffix": "0",
            "Content-Type": content_type,
            "x-content-type": content_type,
            **({"Cache-Control": cache_control} if cache_control else {}),
            **({"x-allow-overwrite": "true"} if allow_overwrite else {}),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=BLOB_TIMEOUT_SECONDS) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        err = e.read().decode() if e.fp else str(e)
        raise RuntimeError(f"Blob upload failed for {pathname}: {e.code} {err}") from e
    url = body.get("url") or body.get("downloadUrl")
    if not url:
        raise RuntimeError(f"No URL in Blob response: {body}")
    return url


def _blob_get(url: str, *, timeout: float = 45.0) -> bytes | None:
    """Read a public Blob object; return None when the object does not exist."""

    url = require_https_url(
        url,
        hosts={"blob.vercel-storage.com", "public.blob.vercel-storage.com"},
        allow_subdomains=True,
    )

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise RuntimeError(f"Blob read failed with HTTP {error.code}: {url}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"Blob read failed: {url}") from error


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
