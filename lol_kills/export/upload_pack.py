#!/usr/bin/env python3
"""Upload a public pack to Vercel Blob (or copy into the atlas public/ folder).

Without BLOB_READ_WRITE_TOKEN:
  copies pack → apps/scryglass/public/packs/<pack_id>/
  and writes apps/scryglass/public/packs/manifest.json with relative base_url.

With BLOB_READ_WRITE_TOKEN:
  uses Vercel Blob REST API (put) for each file under packs/<pack_id>/...
  then writes local manifest with absolute CDN base_url.

Usage:
  python3 -m lol_kills.export.upload_pack
  python3 -m lol_kills.export.upload_pack --pack-id v2026.07.25
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACK_ROOT = ROOT / "output" / "public_pack"
ATLAS_PUBLIC = ROOT / "apps" / "scryglass" / "public" / "packs"
BLOB_TIMEOUT_SECONDS = 45.0


def _load_manifest(pack_dir: Path) -> dict[str, Any]:
    return json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))


def copy_to_atlas(pack_dir: Path, pack_id: str) -> Path:
    dest = ATLAS_PUBLIC / pack_id
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(pack_dir, dest)
    return dest


def _blob_put(
    token: str,
    pathname: str,
    data: bytes,
    content_type: str,
    *,
    cache_control: str | None = None,
    allow_overwrite: bool = False,
) -> str:
    """Upload via Vercel Blob REST API; returns blob URL."""
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

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise RuntimeError(f"Blob read failed with HTTP {error.code}: {url}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"Blob read failed: {url}") from error


def _blob_root(base_url: str) -> str:
    marker = "/packs/"
    if marker not in base_url:
        raise RuntimeError("Blob pack URL does not contain the packs path")
    return base_url.split(marker, 1)[0].rstrip("/")


def upload_to_blob(pack_dir: Path, pack_id: str, token: str) -> dict[str, str]:
    urls: dict[str, str] = {}
    for path in sorted(pack_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(pack_dir).as_posix()
        pathname = f"packs/{pack_id}/{rel}"
        suffix = path.suffix.lower()
        ctype = {
            ".json": "application/json",
            ".md": "text/markdown",
            ".parquet": "application/octet-stream",
            ".csv": "text/csv",
        }.get(suffix, "application/octet-stream")
        urls[rel] = _blob_put(token, pathname, path.read_bytes(), ctype)
        print(f"  uploaded {pathname}")
    return urls


def write_atlas_manifest(
    pack_id: str,
    manifest: dict[str, Any],
    *,
    base_url: str,
) -> Path:
    ATLAS_PUBLIC.mkdir(parents=True, exist_ok=True)
    man = dict(manifest)
    man["base_url"] = base_url.rstrip("/")
    # Ensure file paths stay relative to base_url
    out = ATLAS_PUBLIC / "manifest.json"
    out.write_text(json.dumps(man, indent=2), encoding="utf-8")
    # latest pointer
    (ATLAS_PUBLIC / "latest.json").write_text(
        json.dumps({"pack_id": pack_id, "base_url": man["base_url"]}, indent=2),
        encoding="utf-8",
    )
    return out


def write_vercel_pack_ignore(
    pack_id: str,
    *,
    public_root: Path = ATLAS_PUBLIC,
    ignore_path: Path | None = None,
) -> Path:
    """Keep old immutable packs out of the next app deployment."""

    target = ignore_path or public_root.parents[1] / ".vercelignore"
    existing = target.read_text(encoding="utf-8").splitlines() if target.is_file() else []
    preserved = [
        line
        for line in existing
        if not line.startswith("public/packs/")
        and not line.startswith("!public/packs/")
    ]
    old_packs = sorted(
        path.name
        for path in public_root.iterdir()
        if path.is_dir() and path.name != pack_id and path.name.startswith("v")
    )
    lines = [*preserved, *(f"public/packs/{value}/" for value in old_packs)]
    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return target


def publish_blob_pointers(
    token: str,
    pack_id: str,
    manifest: dict[str, Any],
    *,
    base_url: str,
) -> dict[str, str]:
    """Publish mutable discovery files after the immutable pack is complete."""
    published_manifest = dict(manifest)
    published_manifest["base_url"] = base_url.rstrip("/")
    latest = {
        "pack_id": pack_id,
        "base_url": published_manifest["base_url"],
        "created_utc": published_manifest.get("created_utc"),
    }
    payloads = {
        "packs/manifest.json": published_manifest,
        "packs/latest.json": latest,
    }
    urls: dict[str, str] = {}
    root = _blob_root(base_url)
    previous: dict[str, bytes | None] = {
        pathname: _blob_get(f"{root}/{pathname}") for pathname in payloads
    }
    try:
        for pathname, payload in payloads.items():
            raw = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
            urls[pathname] = _blob_put(
                token,
                pathname,
                raw,
                "application/json",
                cache_control="public, max-age=60, must-revalidate",
                allow_overwrite=True,
            )
            readback = _blob_get(f"{root}/{pathname}")
            if readback != raw:
                raise RuntimeError(f"Blob pointer readback failed for {pathname}")
            print(f"  published {pathname}")
    except Exception:
        for pathname, raw in previous.items():
            if raw is not None:
                _blob_put(
                    token,
                    pathname,
                    raw,
                    "application/json",
                    cache_control="public, max-age=60, must-revalidate",
                    allow_overwrite=True,
                )
        raise
    return urls


def restore_blob_pointers(
    token: str,
    previous: dict[str, str | None],
) -> None:
    """Restore previously captured JSON pointers after a failed smoke check."""

    for pathname, raw in previous.items():
        if raw is None:
            continue
        _blob_put(
            token,
            pathname,
            raw.encode("utf-8"),
            "application/json",
            cache_control="public, max-age=60, must-revalidate",
            allow_overwrite=True,
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Publish public pack to atlas / Vercel Blob")
    ap.add_argument("--pack-root", type=Path, default=DEFAULT_PACK_ROOT)
    ap.add_argument("--pack-id", default=None, help="Default: read output/public_pack/latest.json")
    ap.add_argument(
        "--local-only",
        action="store_true",
        help="Skip Blob even if token is set; only copy into apps/scryglass/public/packs",
    )
    args = ap.parse_args(argv)

    pack_id = args.pack_id
    if not pack_id:
        latest = args.pack_root / "latest.json"
        if not latest.exists():
            raise SystemExit("No --pack-id and no latest.json — run public_pack export first")
        pack_id = json.loads(latest.read_text())["pack_id"]

    pack_dir = args.pack_root / pack_id
    if not pack_dir.is_dir():
        raise SystemExit(f"Missing pack directory: {pack_dir}")

    manifest = _load_manifest(pack_dir)
    token = os.environ.get("BLOB_READ_WRITE_TOKEN") or os.environ.get("VERCEL_BLOB_READ_WRITE_TOKEN")

    print(f"Publishing {pack_id} ({manifest['total_bytes']/1024/1024:.1f} MB)")

    if token and not args.local_only:
        print("Uploading to Vercel Blob…")
        urls = upload_to_blob(pack_dir, pack_id, token)
        # Infer base as common prefix
        sample = next(iter(urls.values()))
        # https://….public.blob.vercel-storage.com/packs/v…/manifest.json → base without trailing file
        base = sample.rsplit(f"/packs/{pack_id}/", 1)[0] + f"/packs/{pack_id}"
        # Still copy locally for offline/dev
        copy_to_atlas(pack_dir, pack_id)
        write_atlas_manifest(pack_id, manifest, base_url=base)
        publish_blob_pointers(token, pack_id, manifest, base_url=base)
        print(f"Blob base_url: {base}")
    else:
        if not token:
            print("No BLOB_READ_WRITE_TOKEN — local copy only")
        dest = copy_to_atlas(pack_dir, pack_id)
        # Relative to site origin
        write_atlas_manifest(pack_id, manifest, base_url=f"/packs/{pack_id}")
        print(f"Copied → {dest}")
        print("Atlas manifest base_url=/packs/" + pack_id)

    write_vercel_pack_ignore(pack_id)

    print("Done. Atlas reads public/packs/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
