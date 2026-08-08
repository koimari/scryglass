"""Restore the checked-in public OE pack as the worker's historical baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

PARQUET_LOCATOR = Path("data/lol/warehouse/parquet")

LATEST_LOCATOR = Path("apps/scryglass/public/packs/latest.json")
PACKS_LOCATOR = Path("apps/scryglass/public/packs")
SCHEMA_VERSION = "scryglass:oe-pack-baseline-restore:v1"
YEARS = ("2025", "2026")
REQUIRED_COLUMNS = {
    "player_games": {
        "gameid",
        "date",
        "league",
        "competition_tier",
        "event_kind",
        "patch",
        "position",
        "champion",
        "side",
        "teamname",
        "result",
    },
    "team_games": {
        "gameid",
        "date",
        "league",
        "competition_tier",
        "event_kind",
        "patch",
        "side",
        "teamname",
        "result",
    },
}


class OePackBaselineError(RuntimeError):
    """Raised when the committed OE baseline cannot be restored safely."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OePackBaselineError(f"cannot read JSON source: {path}") from exc
    if not isinstance(value, dict):
        raise OePackBaselineError(f"JSON source must be an object: {path}")
    return value


def _pack_root(root: Path) -> tuple[Path, dict[str, Any], str, str]:
    latest_path = root / LATEST_LOCATOR
    latest = _read_json(latest_path)
    pack_id = latest.get("pack_id")
    if not isinstance(pack_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", pack_id):
        raise OePackBaselineError("latest pack id is malformed")
    pack_root = root / PACKS_LOCATOR / pack_id
    if not pack_root.is_dir() or pack_root.is_symlink():
        raise OePackBaselineError(f"latest committed OE pack is missing: {pack_root}")
    manifest_path = pack_root / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("pack_id") != pack_id:
        raise OePackBaselineError("latest pack id does not match its manifest")
    filters = manifest.get("filters")
    years = filters.get("years") if isinstance(filters, dict) else None
    if not isinstance(years, list) or not set(YEARS).issubset({str(year) for year in years}):
        raise OePackBaselineError("latest pack does not contain the 2025-2026 baseline")
    return pack_root, manifest, _sha256_path(latest_path), _sha256_path(manifest_path)


def _load_pack_frame(
    pack_root: Path,
    kind: str,
    *,
    repo_root: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    paths = [
        pack_root / kind / f"year={year}" / "part.parquet"
        for year in YEARS
    ]
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise OePackBaselineError(f"{kind} baseline parquet is incomplete")
    frames = []
    bindings: list[dict[str, Any]] = []
    for path in paths:
        frame = pd.read_parquet(path)
        missing = sorted(REQUIRED_COLUMNS[kind].difference(frame.columns))
        if missing:
            raise OePackBaselineError(f"{kind} baseline is missing columns: {missing}")
        if frame.empty:
            raise OePackBaselineError(f"{kind} baseline is empty: {path}")
        frames.append(frame)
        bindings.append(
            {
                "locator": str(path.relative_to(repo_root)),
                "raw_sha256": _sha256_path(path),
                "rows": len(frame),
            }
        )
    combined = pd.concat(frames, ignore_index=True, sort=False)
    parsed_dates = pd.to_datetime(combined["date"], errors="coerce", utc=True)
    if parsed_dates.isna().any():
        raise OePackBaselineError(f"{kind} baseline contains invalid dates")
    if combined["gameid"].astype(str).str.strip().eq("").any():
        raise OePackBaselineError(f"{kind} baseline contains empty game ids")
    return combined, bindings


def restore_baseline(root: Path | str = Path(".")) -> dict[str, Any]:
    repo_root = Path(root).resolve()
    pack_root, manifest, latest_sha256, manifest_sha256 = _pack_root(repo_root)
    frames: dict[str, pd.DataFrame] = {}
    bindings: dict[str, list[dict[str, Any]]] = {}
    for kind in ("player_games", "team_games"):
        frames[kind], bindings[kind] = _load_pack_frame(
            pack_root,
            kind,
            repo_root=repo_root,
        )

    output_root = repo_root / PARQUET_LOCATOR
    output_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, Any]] = {}
    for kind, frame in frames.items():
        destination = output_root / f"oe_{kind}.parquet"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".restore", dir=output_root
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            frame.to_parquet(temporary, index=False)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        outputs[kind] = {
            "locator": str(destination.relative_to(repo_root)),
            "raw_sha256": _sha256_path(destination),
            "rows": len(frame),
        }

    ingest = manifest.get("ingest")
    oe_live_meta = ingest.get("oe_live_meta") if isinstance(ingest, dict) else None
    source_latest = oe_live_meta.get("source_latest") if isinstance(oe_live_meta, dict) else None
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "restored_at": pd.Timestamp.utcnow().isoformat().replace("+00:00", "Z"),
        "pack_id": manifest["pack_id"],
        "pack_locator": str(pack_root.relative_to(repo_root)),
        "latest_locator": str(LATEST_LOCATOR),
        "latest_sha256": latest_sha256,
        "manifest_sha256": manifest_sha256,
        "source_latest": source_latest,
        "years": list(YEARS),
        "source_files": bindings,
        "outputs": outputs,
        "authority": {
            "descriptive_source_freshness_evidence": True,
            "model_validation_authority": False,
            "probability_authority": False,
            "recommendation_authority": False,
            "betting_authority": False,
        },
        "claim_ceiling": "This receipt binds the prior committed public OE pack as the historical baseline. It does not authorize a model, probability, recommendation, or wager.",
    }
    receipt["receipt_canonical_sha256"] = _sha256_bytes(
        json.dumps(receipt, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    )
    receipt_path = output_root / "oe_baseline_meta.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(restore_baseline(args.root), ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
