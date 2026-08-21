"""Freeze exact input bytes for the research-only Draft Score comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "scryglass:future-value-draft-score-freeze:v1"


class DraftScoreFreezeError(ValueError):
    """A required comparison input cannot be frozen."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _file(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise DraftScoreFreezeError(f"input is missing or unsafe: {path}")
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def build_freeze(
    *,
    source_receipt_path: Path,
    folds_root: Path,
    strict_fold_root: Path,
) -> dict[str, Any]:
    folds: dict[str, Any] = {}
    for fold in (1, 2, 3):
        fold_root = folds_root.resolve() / f"fold-{fold}"
        strict_root = strict_fold_root.resolve() / f"fold-{fold}"
        folds[str(fold)] = {
            "spec": _file(folds_root.resolve() / f"fold-{fold}-spec.json"),
            "current": {
                "receipt": _file(fold_root / "current-v2" / "current-rating-feature-ledger.receipt.json"),
                "artifact": _file(fold_root / "current-v2" / "current-rating-feature-ledger.parquet"),
            },
            "scaling": {
                "receipt": _file(fold_root / "scaling-v2" / "scaling-native-receipt.json"),
                "artifact": _file(fold_root / "scaling-v2" / "scaling-native.parquet"),
            },
            "strict_atom": _file(strict_root / "strict-prior-composition-atoms.json"),
            "strict_form": _file(strict_root / "strict-prior-player-form.json"),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_research_only",
        "authority": {
            "research_only": True,
            "public_probability": False,
            "promotion": False,
            "deployment": False,
        },
        "source_receipt": _file(source_receipt_path),
        "folds": folds,
    }


def write_freeze(payload: dict[str, Any], output_path: Path) -> str:
    if output_path.exists():
        raise DraftScoreFreezeError(f"output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical(payload) + b"\n"
    output_path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--folds-root", required=True, type=Path)
    parser.add_argument("--strict-fold-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = build_freeze(
        source_receipt_path=args.source_receipt,
        folds_root=args.folds_root,
        strict_fold_root=args.strict_fold_root,
    )
    digest = write_freeze(payload, args.output)
    print(json.dumps({"path": str(args.output.resolve()), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
