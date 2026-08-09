"""Generate the exact C1 foundation-freeze artifact chain."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

from lol_kills.v2.evaluation.checkpoint_c1 import (
    AUTHORITY_LOCATOR,
    CONFIG_LOCATOR,
    INPUT_ROLE_LOCATORS,
    REPORT_LOCATOR,
    ROOT,
    _lstat_component,
    _validate_locator,
    build_checkpoint_c1_bundle,
    load_checkpoint_c1,
)
from lol_kills.v2.evaluation.checks import ValidationFailure


OUTPUT_LOCATORS = (CONFIG_LOCATOR, REPORT_LOCATOR, AUTHORITY_LOCATOR)


def _capture_inputs(root: Path) -> dict[str, bytes]:
    return {locator: (root / locator).read_bytes() for locator in INPUT_ROLE_LOCATORS}


def _preflight_outputs(root: Path) -> dict[str, tuple[bool, tuple[int, int] | None, bytes | None]]:
    root = Path(root).absolute()
    _lstat_component(root, leaf=False)
    seen_inodes: dict[tuple[int, int], str] = {}
    result: dict[str, tuple[bool, tuple[int, int] | None, bytes | None]] = {}
    for locator in OUTPUT_LOCATORS:
        pure = _validate_locator(locator)
        current = root
        for part in pure.parts[:-1]:
            current = current / part
            _lstat_component(current, leaf=False)
        leaf = current / pure.parts[-1]
        try:
            info = leaf.lstat()
        except FileNotFoundError:
            result[locator] = (False, None, None)
            continue
        except OSError as exc:
            raise ValidationFailure(f"C1 output leaf unavailable: {locator}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise ValidationFailure(f"C1 output symlink rejected: {locator}")
        if not stat.S_ISREG(info.st_mode):
            raise ValidationFailure(f"C1 output nonregular leaf rejected: {locator}")
        if info.st_nlink != 1:
            raise ValidationFailure(f"C1 output hardlinked leaf rejected: {locator}")
        inode = (info.st_dev, info.st_ino)
        prior = seen_inodes.get(inode)
        if prior is not None:
            raise ValidationFailure(f"C1 output inode role substitution: {prior} and {locator}")
        seen_inodes[inode] = locator
        try:
            old_bytes = leaf.read_bytes()
        except OSError as exc:
            raise ValidationFailure(f"C1 output leaf unreadable: {locator}") from exc
        result[locator] = (True, inode, old_bytes)
    return result


def _exclusive_regular_write(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise ValidationFailure(f"C1 staged write made no progress: {path.name}")
            offset += written
        os.fsync(descriptor)
    except ValidationFailure:
        raise
    except OSError as exc:
        raise ValidationFailure(f"C1 staged output write failed: {path.name}") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as exc:
                raise ValidationFailure(f"C1 staged output close failed: {path.name}") from exc


def _cleanup_exact(paths: list[Path]) -> None:
    for path in paths:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ValidationFailure(f"C1 exact temporary cleanup failed: {path.name}") from exc


def _transactional_replace(
    root: Path,
    bundle: dict[str, bytes],
    preflight: dict[str, tuple[bool, tuple[int, int] | None, bytes | None]],
) -> None:
    token = hashlib.sha256(os.urandom(32)).hexdigest()
    staged: dict[str, Path] = {}
    backups: dict[str, Path] = {}
    temporary_paths: list[Path] = []
    replaced: list[str] = []
    try:
        for index, locator in enumerate(OUTPUT_LOCATORS):
            leaf = root / locator
            staged_path = leaf.parent / f".{leaf.name}.c1-stage-{token}-{index}"
            staged[locator] = staged_path
            temporary_paths.append(staged_path)
            _exclusive_regular_write(staged_path, bundle[locator])
            existed, _, old_bytes = preflight[locator]
            if existed:
                backup_path = leaf.parent / f".{leaf.name}.c1-backup-{token}-{index}"
                backups[locator] = backup_path
                temporary_paths.append(backup_path)
                _exclusive_regular_write(backup_path, old_bytes if old_bytes is not None else b"")
        if _preflight_outputs(root) != preflight:
            raise ValidationFailure("C1 output paths changed after preflight")
        for locator in OUTPUT_LOCATORS:
            try:
                os.replace(staged[locator], root / locator)
            except OSError as exc:
                raise ValidationFailure(f"C1 atomic output replacement failed: {locator}") from exc
            replaced.append(locator)
    except Exception:
        for locator in reversed(replaced):
            existed, _, _ = preflight[locator]
            try:
                if existed:
                    os.replace(backups[locator], root / locator)
                    temporary_paths.remove(backups[locator])
                else:
                    os.unlink(root / locator)
            except OSError as exc:
                raise ValidationFailure(f"C1 output rollback failed: {locator}") from exc
        raise
    finally:
        _cleanup_exact(temporary_paths)


def generate(root: Path = ROOT) -> dict[str, str]:
    root = Path(root).absolute()
    before = _capture_inputs(root)
    first = build_checkpoint_c1_bundle(root)
    second = build_checkpoint_c1_bundle(root)
    if first != second:
        raise ValidationFailure("C1 fresh generation replays were not byte-identical")
    if tuple(first) != OUTPUT_LOCATORS:
        raise ValidationFailure("C1 generator output set is not exact")
    preflight = _preflight_outputs(root)
    _transactional_replace(root, first, preflight)
    if _capture_inputs(root) != before:
        raise ValidationFailure("C1 generation changed frozen input bytes")
    load_checkpoint_c1(root).authenticate()
    replay = build_checkpoint_c1_bundle(root)
    actual = {locator: (root / locator).read_bytes() for locator in OUTPUT_LOCATORS}
    if replay != actual:
        raise ValidationFailure("C1 written artifacts failed exact fresh replay")
    return {locator: hashlib.sha256(actual[locator]).hexdigest() for locator in OUTPUT_LOCATORS}


def main() -> None:
    print(json.dumps(generate(), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
