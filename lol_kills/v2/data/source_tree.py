"""Source-tree hashing utilities for v2 reproducibility proofs."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
from typing import Sequence

from .common import ContractError

PREFIX = b"scryglass-source-tree-v1\x00"


def normalize_source_tree_path(raw_path: str) -> str:
    """Normalize repository-relative allowlist paths to deterministic POSIX form."""

    if not isinstance(raw_path, str):
        raise ContractError(f"allowlist path must be a string: {raw_path!r}")
    if raw_path == "":
        raise ContractError("allowlist path must be non-empty")
    if raw_path == "." or raw_path.startswith("./"):
        raise ContractError(f"relative path traversal is not allowed: {raw_path!r}")
    if "\\" in raw_path:
        raise ContractError(f"path must be POSIX-style: {raw_path!r}")
    if raw_path.startswith("/") or raw_path.startswith("~"):
        raise ContractError(f"absolute path not allowed in source-tree allowlist: {raw_path}")
    if "\x00" in raw_path:
        raise ContractError(f"invalid NUL path in allowlist: {raw_path}")

    path = PurePosixPath(raw_path)
    if path.parts and path.parts[0] in {".", ".."}:
        raise ContractError(f"invalid path traversal: {raw_path}")
    for part in path.parts:
        if not part or part == "." or part == "..":
            raise ContractError(f"invalid path component in source-tree allowlist: {raw_path}")

    normalized = path.as_posix()
    if normalized == "":
        raise ContractError("empty normalized path not allowed")
    if normalized.startswith("/"):
        raise ContractError(f"non-posix absolute path in source-tree allowlist: {raw_path}")
    return normalized


def _assert_path_allowed(root: Path, rel: str) -> Path:
    normalized = normalize_source_tree_path(rel)
    if normalized in {".", ".."}:
        raise ContractError(f"invalid allowlist path: {rel}")
    if ".git" in normalized.split("/"):
        raise ContractError(f".git entries are not allowed in source-tree allowlist: {rel}")

    path = root.joinpath(normalized)
    current = root
    for part in Path(normalized).parts:
        current = current / part
        if current.is_symlink():
            raise ContractError(f"symlink path is not allowed: {rel}")
    root_real = root.resolve()
    path_real = path.resolve()
    if path_real == root_real:
        raise ContractError(f"source-tree allowlist path must be a file: {rel}")
    try:
        path_real.relative_to(root_real)
    except ValueError as err:
        raise ContractError(f"non-descendant path in allowlist: {rel}") from err
    if not path_real.exists():
        raise ContractError(f"source-tree allowlist path does not exist: {rel}")
    if not path_real.is_file():
        raise ContractError(f"source-tree allowlist path must be a file: {rel}")
    if path_real.is_symlink():
        raise ContractError(f"symlink path is not allowed: {rel}")
    if path_real.as_posix().endswith("/"):
        raise ContractError(f"allowlist path must not end with slash: {rel}")
    return path_real


def resolve_repository_file(root: Path, locator: str) -> Path:
    """Resolve one normalized repository-relative regular file without symlinks."""

    normalized = normalize_source_tree_path(locator)
    return _assert_path_allowed(root, normalized)


def canonical_source_tree_sha256(
    root: Path,
    allowlist_paths: Sequence[str],
) -> str:
    """
    Compute the contract-defined source-tree digest.

    Deterministic steps:
    1. normalize every allowlist path to POSIX relative form;
    2. deduplicate normalized paths;
    3. sort paths by byte value;
    4. stream prefix + (path_len, path_bytes, content_len, content_bytes).
    """

    if not allowlist_paths:
        raise ContractError("allowlist must contain at least one path")

    normalized_paths: list[str] = []
    seen: set[str] = set()
    for raw_path in allowlist_paths:
        normalized = normalize_source_tree_path(raw_path)
        if normalized in seen:
            raise ContractError(f"duplicate path in allowlist: {raw_path}")
        seen.add(normalized)
        normalized_paths.append(normalized)

    resolved_paths: list[tuple[str, Path]] = []
    for rel in sorted(normalized_paths):
        resolved_paths.append((rel, _assert_path_allowed(root, rel)))

    digest = hashlib.sha256()
    digest.update(PREFIX)
    for rel, path in resolved_paths:
        path_bytes = rel.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(path_bytes).to_bytes(8, "big", signed=False))
        digest.update(path_bytes)
        digest.update(len(content).to_bytes(8, "big", signed=False))
        digest.update(content)
    return digest.hexdigest()


__all__ = [
    "PREFIX",
    "normalize_source_tree_path",
    "resolve_repository_file",
    "canonical_source_tree_sha256",
]
