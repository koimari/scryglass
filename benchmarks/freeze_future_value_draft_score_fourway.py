"""Freeze exact input bytes for the research-only Draft Score comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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


def _safe_file(path: Path, *, label: str) -> Path:
    """Return a regular file with no symlink in its path."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = Path(os.path.normpath(str(candidate)))
    if any(part.is_symlink() for part in (candidate, *candidate.parents)):
        raise DraftScoreFreezeError(f"{label} is missing or unsafe: {candidate}")
    if not candidate.is_file():
        raise DraftScoreFreezeError(f"{label} is missing or unsafe: {candidate}")
    return candidate


def _file(path: Path, *, label: str = "input") -> dict[str, Any]:
    path = _safe_file(path, label=label)
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def build_freeze(
    *,
    source_receipt_path: Path,
    folds_root: Path,
    strict_fold_root: Path,
    evaluation_root: Path,
) -> dict[str, Any]:
    evaluation_root = Path(evaluation_root)
    future_model_path = evaluation_root / "future_player_form" / "model.json"
    future_binding = {
        "artifact": _file(
            future_model_path,
            label="future player form model",
        )
    }
    folds: dict[str, Any] = {}
    for fold in (1, 2, 3):
        fold_root = Path(folds_root) / f"fold-{fold}"
        strict_root = Path(strict_fold_root) / f"fold-{fold}"
        folds[str(fold)] = {
            "spec": _file(Path(folds_root) / f"fold-{fold}-spec.json"),
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
            # The v4 evaluator reads one shared future-player-form model for
            # every fold. Keep the binding under each fold so the trust-root
            # contract matches the evaluator's fold lookup.
            "future": {
                "artifact": dict(future_binding["artifact"]),
            },
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
        "source_receipt": _file(source_receipt_path, label="source receipt"),
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
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = build_freeze(
        source_receipt_path=args.source_receipt,
        folds_root=args.folds_root,
        strict_fold_root=args.strict_fold_root,
        evaluation_root=args.evaluation_root,
    )
    digest = write_freeze(payload, args.output)
    print(json.dumps({"path": str(args.output.resolve()), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
