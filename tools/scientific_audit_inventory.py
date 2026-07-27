#!/usr/bin/env python3
"""Deterministic source inventory for the Scryglass scientific audit.

The inventory is intentionally derived from file bytes.  It does not claim that
a file is reviewed; it creates the hash/line/symbol surface against which the
source, calculation, and claim ledgers can measure review coverage.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


SOURCE_ROOTS = (
    Path("apps/lol-atlas/src"),
    Path("lol_kills"),
    Path("tests"),
    Path("tools"),
)
SOURCE_SUFFIXES = {".py", ".ts", ".tsx"}
PUBLIC_ENTRY_RE = re.compile(
    r"^apps/lol-atlas/src/app/.+/(?:page|route|layout|not-found)\.tsx?$"
    r"|^apps/lol-atlas/src/app/(?:page|layout|not-found)\.tsx?$"
)
TS_SYMBOL_RE = re.compile(
    r"^(?:export\s+(?:default\s+)?(?:async\s+)?)?"
    r"(?:function|class|type|interface|const)\s+([A-Za-z_$][A-Za-z0-9_$]*)",
    re.MULTILINE,
)
TS_IMPORT_RE = re.compile(
    r"""(?:from\s+|import\s*\()\s*["']([^"']+)["']""",
    re.MULTILINE,
)


def _iter_sources(root: Path, include_tests: bool) -> Iterable[Path]:
    for relative_root in SOURCE_ROOTS:
        if not include_tests and relative_root == Path("tests"):
            continue
        base = root / relative_root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix in SOURCE_SUFFIXES:
                yield path


def _python_metadata(text: str) -> tuple[list[str], list[str], str | None]:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return [], [], f"{exc.msg} at line {exc.lineno}"

    symbols: list[str] = []
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(node.name)
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.append("." * node.level + module)
    return sorted(set(symbols)), sorted(set(imports)), None


def _typescript_metadata(text: str) -> tuple[list[str], list[str], str | None]:
    symbols = sorted(set(TS_SYMBOL_RE.findall(text)))
    imports = sorted(set(TS_IMPORT_RE.findall(text)))
    return symbols, imports, None


def _reachability_hint(relative: str) -> str:
    if PUBLIC_ENTRY_RE.match(relative):
        return "public_entry"
    if relative.startswith("apps/lol-atlas/src/components/"):
        return "frontend_component"
    if relative.startswith("apps/lol-atlas/src/lib/"):
        return "frontend_library"
    if relative.startswith("lol_kills/etl/"):
        return "release_data_pipeline"
    if relative.startswith("lol_kills/export/"):
        return "release_pack_pipeline"
    if relative == "lol_kills/model_tournament.py" or relative.startswith(
        "lol_kills/ml/"
    ):
        return "release_model_validation"
    if relative.startswith("lol_kills/ratings/"):
        return "release_model_pipeline"
    if relative.startswith("lol_kills/research/"):
        return "research_generator_unverified"
    if relative.startswith("lol_kills/"):
        return "backend_unclassified"
    if relative.startswith("tests/"):
        return "test_evidence"
    if relative.startswith("tools/"):
        return "audit_or_release_tooling"
    return "unclassified"


def inventory(root: Path, *, include_tests: bool = True) -> dict[str, Any]:
    root = root.resolve()
    rows: list[dict[str, Any]] = []
    for path in _iter_sources(root, include_tests):
        data = path.read_bytes()
        text = data.decode("utf-8")
        relative = path.relative_to(root).as_posix()
        if path.suffix == ".py":
            symbols, imports, parse_error = _python_metadata(text)
        else:
            symbols, imports, parse_error = _typescript_metadata(text)
        rows.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "lines": len(text.splitlines()),
                "language": "python" if path.suffix == ".py" else "typescript",
                "reachability_hint": _reachability_hint(relative),
                "symbols": symbols,
                "imports": imports,
                "parse_error": parse_error,
                "review_status": "unreviewed",
            }
        )

    groups: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = groups.setdefault(
            row["reachability_hint"],
            {"files": 0, "lines": 0, "bytes": 0, "symbols": 0},
        )
        bucket["files"] += 1
        bucket["lines"] += row["lines"]
        bucket["bytes"] += row["bytes"]
        bucket["symbols"] += len(row["symbols"])

    digest_payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": 1,
        "root": str(root),
        "inventory_sha256": hashlib.sha256(digest_payload).hexdigest(),
        "files": len(rows),
        "lines": sum(row["lines"] for row in rows),
        "symbols": sum(len(row["symbols"]) for row in rows),
        "parse_errors": sum(row["parse_error"] is not None for row in rows),
        "groups": dict(sorted(groups.items())),
        "sources": rows,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Scryglass source audit inventory",
        "",
        f"- Inventory hash: `{payload['inventory_sha256']}`",
        f"- Files: {payload['files']}",
        f"- Lines: {payload['lines']}",
        f"- Symbols: {payload['symbols']}",
        f"- Parse errors: {payload['parse_errors']}",
        "",
        "| Reachability hint | Files | Lines | Symbols |",
        "|---|---:|---:|---:|",
    ]
    for name, group in payload["groups"].items():
        lines.append(
            f"| `{name}` | {group['files']} | {group['lines']} | {group['symbols']} |"
        )
    lines.extend(
        [
            "",
            "| Path | Lines | SHA-256 | Review |",
            "|---|---:|---|---|",
        ]
    )
    for row in payload["sources"]:
        lines.append(
            f"| `{row['path']}` | {row['lines']} | `{row['sha256']}` | "
            f"{row['review_status']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--without-tests", action="store_true")
    args = parser.parse_args()

    payload = inventory(args.root, include_tests=not args.without_tests)
    if args.format == "markdown":
        print(_markdown(payload), end="")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if payload["parse_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
