#!/usr/bin/env python3
"""Validate and publish a public pack to Vercel Blob or the atlas bundle.

Publication is fail closed:

* pack identifiers and every manifest path are validated before path use;
* the manifest is the exact data-file allowlist;
* declared byte sizes and SHA-256 digests are checked before mutation;
* a staged, immutable copy receives the full public-pack release audit;
* immutable files complete before mutable discovery pointers advance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from lol_kills.audit_public_pack import audit_pack, require_release_gate


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACK_ROOT = ROOT / "output" / "public_pack"
ATLAS_PUBLIC = ROOT / "apps" / "lol-atlas" / "public" / "packs"

_PACK_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_NAME = "manifest.json"
_LOCAL_AUXILIARY_FILES = frozenset({"README.md"})


@dataclass(frozen=True)
class ValidatedPack:
    root: Path
    pack_id: str
    manifest: dict[str, Any]
    declared_files: tuple[tuple[str, Path], ...]

    @property
    def publish_files(self) -> tuple[tuple[str, Path], ...]:
        """Files published at the immutable pack URL.

        ``manifest.json`` is the control document for the declared data files;
        it cannot hash-declare itself, so it is handled as the sole published
        control file.
        """

        return (*self.declared_files, (_MANIFEST_NAME, self.root / _MANIFEST_NAME))


def validate_pack_id(value: Any) -> str:
    """Return a pack id that is safe to use as one filesystem basename."""

    if not isinstance(value, str) or not _PACK_ID_PATTERN.fullmatch(value):
        raise ValueError(f"Unsafe pack_id: {value!r}")
    if value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"Unsafe pack_id: {value!r}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(pack_dir: Path) -> dict[str, Any]:
    path = pack_dir / _MANIFEST_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid pack manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Pack manifest must be an object: {path}")
    return payload


def _manifest_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Manifest file path must be a non-empty string: {value!r}")
    if "\\" in value:
        raise RuntimeError(f"Manifest file path must use POSIX separators: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RuntimeError(f"Unsafe manifest file path: {value!r}")
    if value == _MANIFEST_NAME or value in _LOCAL_AUXILIARY_FILES:
        raise RuntimeError(f"Manifest file path collides with a control file: {value!r}")
    return value


def validate_pack(pack_dir: Path, pack_id: str) -> ValidatedPack:
    """Verify the complete pack and return its frozen publication allowlist."""

    pack_id = validate_pack_id(pack_id)
    pack_dir = Path(pack_dir)
    if pack_dir.is_symlink() or not pack_dir.is_dir():
        raise RuntimeError(f"Missing or unsafe pack directory: {pack_dir}")

    manifest = _load_manifest(pack_dir)
    if manifest.get("pack_id") != pack_id:
        raise RuntimeError(
            "Manifest pack_id mismatch: "
            f"expected {pack_id!r}, found {manifest.get('pack_id')!r}"
        )

    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise RuntimeError("Manifest files must be a non-empty list")

    declared: list[tuple[str, Path]] = []
    declared_paths: set[str] = set()
    declared_bytes = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise RuntimeError(f"Manifest file record {index} must be an object")
        relative = _manifest_relative_path(record.get("path"))
        if relative in declared_paths:
            raise RuntimeError(f"Duplicate manifest file path: {relative}")
        if "relative" in record and record["relative"] != relative:
            raise RuntimeError(f"Manifest relative/path mismatch for {relative}")

        size = record.get("bytes")
        digest = record.get("sha256")
        if type(size) is not int or size < 0:
            raise RuntimeError(f"Invalid declared byte size for {relative}: {size!r}")
        if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
            raise RuntimeError(f"Invalid declared SHA-256 for {relative}: {digest!r}")

        path = pack_dir.joinpath(*PurePosixPath(relative).parts)
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Missing or unsafe declared file: {relative}")
        if path.stat().st_size != size:
            raise RuntimeError(f"Declared byte size mismatch for {relative}")
        if _sha256(path) != digest:
            raise RuntimeError(f"Declared SHA-256 mismatch for {relative}")

        declared_paths.add(relative)
        declared_bytes += size
        declared.append((relative, path))

    if type(manifest.get("total_files")) is not int:
        raise RuntimeError("Manifest total_files must be an integer")
    if manifest["total_files"] != len(declared):
        raise RuntimeError("Manifest total_files does not match its file allowlist")
    if type(manifest.get("total_bytes")) is not int:
        raise RuntimeError("Manifest total_bytes must be an integer")
    if manifest["total_bytes"] != declared_bytes:
        raise RuntimeError("Manifest total_bytes does not match declared byte sizes")

    actual_files: set[str] = set()
    for path in pack_dir.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(
                "Pack contains a symlink outside the manifest contract: "
                f"{path.relative_to(pack_dir).as_posix()}"
            )
        if path.is_file():
            actual_files.add(path.relative_to(pack_dir).as_posix())

    expected_files = declared_paths | {_MANIFEST_NAME}
    allowed_local_files = expected_files | _LOCAL_AUXILIARY_FILES
    missing = sorted(expected_files - actual_files)
    extra = sorted(actual_files - allowed_local_files)
    if missing:
        raise RuntimeError(f"Pack is missing allowlisted files: {missing}")
    if extra:
        raise RuntimeError(f"Pack contains undeclared files: {extra}")

    return ValidatedPack(
        root=pack_dir,
        pack_id=pack_id,
        manifest=manifest,
        declared_files=tuple(declared),
    )


@contextmanager
def staged_validated_pack(
    pack_dir: Path,
    pack_id: str,
) -> Iterator[ValidatedPack]:
    """Freeze the validated publication files and verify the frozen copy."""

    source = validate_pack(pack_dir, pack_id)
    with tempfile.TemporaryDirectory(prefix="scryglass-pack-publish-") as temp:
        staged_root = Path(temp) / source.pack_id
        staged_root.mkdir()
        for relative, path in source.publish_files:
            destination = staged_root.joinpath(*PurePosixPath(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        yield validate_pack(staged_root, source.pack_id)


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
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        err = exc.read().decode() if exc.fp else str(exc)
        raise RuntimeError(
            f"Blob upload failed for {pathname}: {exc.code} {err}"
        ) from exc
    url = body.get("url") or body.get("downloadUrl")
    if not url:
        raise RuntimeError(f"No URL in Blob response: {body}")
    return str(url)


def _content_type(path: Path) -> str:
    return {
        ".json": "application/json",
        ".md": "text/markdown",
        ".parquet": "application/octet-stream",
        ".csv": "text/csv",
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".txt": "text/plain",
    }.get(path.suffix.lower(), "application/octet-stream")


def _upload_validated_to_blob(
    pack: ValidatedPack,
    token: str,
) -> dict[str, str]:
    urls: dict[str, str] = {}
    # Data files complete first.  The immutable manifest is written last so it
    # never advertises a partially uploaded immutable pack.
    for relative, path in pack.publish_files:
        pathname = f"packs/{pack.pack_id}/{relative}"
        urls[relative] = _blob_put(
            token,
            pathname,
            path.read_bytes(),
            _content_type(path),
        )
        print(f"  uploaded {pathname}")
    return urls


def upload_to_blob(pack_dir: Path, pack_id: str, token: str) -> dict[str, str]:
    """Validate, audit, stage, and upload one immutable pack."""

    pack_id = validate_pack_id(pack_id)
    with staged_validated_pack(pack_dir, pack_id) as staged:
        report = audit_pack(staged.root)
        require_release_gate(report)
        return _upload_validated_to_blob(staged, token)


def _copy_validated_to_atlas(pack: ValidatedPack) -> Path:
    """Atomically replace the bundled immutable pack with a validated stage."""

    pack_id = validate_pack_id(pack.pack_id)
    ATLAS_PUBLIC.mkdir(parents=True, exist_ok=True)
    destination = ATLAS_PUBLIC / pack_id
    with tempfile.TemporaryDirectory(
        prefix=".scryglass-pack-copy-",
        dir=ATLAS_PUBLIC,
    ) as temp:
        transaction_root = Path(temp)
        candidate = transaction_root / "candidate"
        candidate.mkdir()
        for relative, path in pack.publish_files:
            target = candidate.joinpath(*PurePosixPath(relative).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        validate_pack(candidate, pack_id)

        backup = transaction_root / "previous"
        moved_previous = False
        try:
            if destination.exists():
                if destination.is_symlink() or not destination.is_dir():
                    raise RuntimeError(f"Unsafe atlas pack destination: {destination}")
                os.replace(destination, backup)
                moved_previous = True
            os.replace(candidate, destination)
        except Exception:
            if moved_previous and not destination.exists() and backup.exists():
                os.replace(backup, destination)
            raise
    return destination


def copy_to_atlas(pack_dir: Path, pack_id: str) -> Path:
    """Validate, audit, and transactionally copy one immutable atlas pack."""

    pack_id = validate_pack_id(pack_id)
    with staged_validated_pack(pack_dir, pack_id) as staged:
        report = audit_pack(staged.root)
        require_release_gate(report)
        return _copy_validated_to_atlas(staged)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _pointer_payloads(
    pack_id: str,
    manifest: dict[str, Any],
    base_url: str,
) -> tuple[bytes, bytes]:
    pack_id = validate_pack_id(pack_id)
    if manifest.get("pack_id") != pack_id:
        raise RuntimeError("Pointer manifest pack_id does not match the publication")
    if not isinstance(base_url, str) or not base_url.strip():
        raise RuntimeError("Pointer base_url must be a non-empty string")
    normalized_base = base_url.rstrip("/")
    if not normalized_base.endswith(f"/packs/{pack_id}"):
        raise RuntimeError("Pointer base_url does not target the validated pack_id")
    published_manifest = dict(manifest)
    published_manifest["base_url"] = normalized_base
    latest = {
        "pack_id": pack_id,
        "base_url": normalized_base,
        "created_utc": published_manifest.get("created_utc"),
    }
    # Serialize the complete pointer transaction before any local or remote
    # pointer can advance.
    latest_bytes = (json.dumps(latest, indent=2) + "\n").encode("utf-8")
    manifest_bytes = (
        json.dumps(published_manifest, indent=2) + "\n"
    ).encode("utf-8")
    return latest_bytes, manifest_bytes


def write_atlas_manifest(
    pack_id: str,
    manifest: dict[str, Any],
    *,
    base_url: str,
    release_report: dict[str, Any],
) -> Path:
    require_release_gate(release_report)
    latest_bytes, manifest_bytes = _pointer_payloads(
        pack_id,
        manifest,
        base_url,
    )
    # The full manifest is the controlling reader pointer, so advance it last.
    _write_bytes_atomic(ATLAS_PUBLIC / "latest.json", latest_bytes)
    output = ATLAS_PUBLIC / _MANIFEST_NAME
    _write_bytes_atomic(output, manifest_bytes)
    return output


def publish_blob_pointers(
    token: str,
    pack_id: str,
    manifest: dict[str, Any],
    *,
    base_url: str,
    release_report: dict[str, Any],
) -> dict[str, str]:
    """Publish mutable discovery files after immutable upload and release audit."""

    require_release_gate(release_report)
    pack_id = validate_pack_id(pack_id)
    latest_bytes, manifest_bytes = _pointer_payloads(
        pack_id,
        manifest,
        base_url,
    )
    payloads = (
        ("packs/latest.json", latest_bytes),
        ("packs/manifest.json", manifest_bytes),
    )
    urls: dict[str, str] = {}
    for pathname, payload in payloads:
        urls[pathname] = _blob_put(
            token,
            pathname,
            payload,
            "application/json",
            cache_control="public, max-age=60, must-revalidate",
            allow_overwrite=True,
        )
        print(f"  published {pathname}")
    return urls


def _blob_base_url(urls: dict[str, str], pack_id: str) -> str:
    marker = f"/packs/{pack_id}/"
    bases = {
        url.rsplit(marker, 1)[0] + f"/packs/{pack_id}"
        for url in urls.values()
        if marker in url
    }
    if len(bases) != 1:
        raise RuntimeError("Blob responses do not share the expected immutable pack URL")
    return bases.pop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and publish a public pack to atlas / Vercel Blob"
    )
    parser.add_argument("--pack-root", type=Path, default=DEFAULT_PACK_ROOT)
    parser.add_argument(
        "--pack-id",
        default=None,
        help="Default: read output/public_pack/latest.json",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Skip Blob even if token is set; only copy into apps/lol-atlas/public/packs",
    )
    args = parser.parse_args(argv)

    raw_pack_id = args.pack_id
    if raw_pack_id is None:
        latest = args.pack_root / "latest.json"
        if not latest.exists():
            raise SystemExit(
                "No --pack-id and no latest.json — run public_pack export first"
            )
        try:
            latest_payload = json.loads(latest.read_text(encoding="utf-8"))
            raw_pack_id = latest_payload["pack_id"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError(f"Invalid latest pack pointer: {latest}") from exc

    # Validate before constructing a child path or reaching any deletion/copy.
    pack_id = validate_pack_id(raw_pack_id)
    pack_dir = args.pack_root / pack_id
    token = os.environ.get("BLOB_READ_WRITE_TOKEN") or os.environ.get(
        "VERCEL_BLOB_READ_WRITE_TOKEN"
    )

    with staged_validated_pack(pack_dir, pack_id) as staged:
        report = audit_pack(staged.root)
        require_release_gate(report)
        manifest = staged.manifest
        print(
            f"Publishing {pack_id} "
            f"({manifest['total_bytes'] / 1024 / 1024:.1f} MB)"
        )

        if token and not args.local_only:
            print("Uploading to Vercel Blob…")
            urls = _upload_validated_to_blob(staged, token)
            base_url = _blob_base_url(urls, pack_id)
            _copy_validated_to_atlas(staged)
            write_atlas_manifest(
                pack_id,
                manifest,
                base_url=base_url,
                release_report=report,
            )
            publish_blob_pointers(
                token,
                pack_id,
                manifest,
                base_url=base_url,
                release_report=report,
            )
            print(f"Blob base_url: {base_url}")
        else:
            if not token:
                print("No BLOB_READ_WRITE_TOKEN — local copy only")
            destination = _copy_validated_to_atlas(staged)
            base_url = f"/packs/{pack_id}"
            write_atlas_manifest(
                pack_id,
                manifest,
                base_url=base_url,
                release_report=report,
            )
            print(f"Copied → {destination}")
            print(f"Atlas manifest base_url={base_url}")

    print("Done. Atlas reads public/packs/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
