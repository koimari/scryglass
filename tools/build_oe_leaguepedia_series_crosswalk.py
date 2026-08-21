"""CLI for a local, source-bound OE to Leaguepedia series crosswalk.

The command reads JSON already downloaded by the caller.  It never performs
network access.  The capture manifest supplies source URLs and retrieval
times.  The command hashes the exact downloaded bytes before calling the
research crosswalk builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from lol_kills.research.oe_leaguepedia_series_crosswalk import (
    CrosswalkError,
    build_oe_leaguepedia_series_crosswalk,
)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CrosswalkError(f"JSON input cannot be read: {path}") from error


def _rows(value: Any, *, label: str) -> list[dict[str, Any]]:
    if isinstance(value, list) and all(isinstance(row, Mapping) for row in value):
        return [dict(row) for row in value]
    if isinstance(value, Mapping):
        for key in ("rows", "games", "data", "result"):
            nested = value.get(key)
            if isinstance(nested, list) and all(isinstance(row, Mapping) for row in nested):
                return [dict(row) for row in nested]
    raise CrosswalkError(f"{label} JSON must be an array of objects")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _source_metadata(manifest: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    sources = manifest.get("sources", manifest.get("source_records"))
    if not isinstance(sources, Mapping) or not isinstance(sources.get(label), Mapping):
        raise CrosswalkError(f"capture manifest has no source record: {label}")
    return sources[label]


def _source_record(
    manifest: Mapping[str, Any],
    label: str,
    path: Path,
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    metadata = dict(_source_metadata(manifest, label))
    locator = str(metadata.get("url") or metadata.get("locator") or path)
    retrieved_at = str(
        metadata.get("retrieved_at")
        or metadata.get("retrieval_time")
        or metadata.get("captured_at")
        or manifest.get("captured_at")
        or ""
    )
    if not retrieved_at:
        raise CrosswalkError(f"capture manifest has no retrieval time: {label}")
    payload = _canonical_bytes(rows)
    actual_hash = _sha256(raw)
    claimed_hash = str(metadata.get("sha256") or metadata.get("raw_sha256") or "").lower()
    claimed_bytes = metadata.get("bytes", metadata.get("raw_bytes"))
    if claimed_hash and claimed_hash != actual_hash:
        raise CrosswalkError(f"capture manifest source hash differs from downloaded bytes: {label}")
    if claimed_bytes is not None and claimed_bytes != len(raw):
        raise CrosswalkError(f"capture manifest source byte count differs from downloaded bytes: {label}")
    return {
        "locator": locator,
        "path": str(path),
        "retrieved_at": retrieved_at,
        "sha256": actual_hash,
        "bytes": len(raw),
        "payload_sha256": _sha256(payload),
        "payload_bytes": len(payload),
    }, raw


def _source_receipt(path: Path) -> dict[str, Any]:
    value = _load_json(path)
    if not isinstance(value, Mapping):
        raise CrosswalkError("source receipt must be a JSON object")
    return dict(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oe", type=Path, required=True, help="downloaded OE game-row JSON")
    parser.add_argument("--scoreboardgames", type=Path, required=True, help="downloaded ScoreboardGames JSON")
    parser.add_argument("--matchschedule", type=Path, required=True, help="downloaded MatchSchedule JSON")
    parser.add_argument("--capture-manifest", type=Path, required=True, help="capture metadata JSON")
    parser.add_argument("--source-receipt", type=Path, required=True, help="canonical OE source receipt JSON")
    parser.add_argument("--competition-map", type=Path, required=True, help="explicit source-to-Leaguepedia mapping JSON")
    parser.add_argument("--output", type=Path, required=True, help="crosswalk output JSON")
    parser.add_argument("--allow-partial", action="store_true", help="emit explicit mapped-row coverage when some OE rows are unmatched")
    parser.add_argument("--captured-at", default=None, help="override capture manifest timestamp")
    args = parser.parse_args(argv)
    try:
        manifest = _load_json(args.capture_manifest)
        source_receipt = _source_receipt(args.source_receipt)
        competition_mapping = _load_json(args.competition_map)
        if not isinstance(manifest, Mapping) or not isinstance(competition_mapping, Mapping):
            raise CrosswalkError("capture manifest and competition map must be objects")
        paths = {
            "oe": args.oe,
            "scoreboardgames": args.scoreboardgames,
            "matchschedule": args.matchschedule,
        }
        loaded = {label: _rows(_load_json(path), label=label) for label, path in paths.items()}
        records: dict[str, dict[str, Any]] = {}
        raw_bytes: dict[str, bytes] = {}
        for label, path in paths.items():
            records[label], raw_bytes[label] = _source_record(manifest, label, path, loaded[label])
        captured_at = str(args.captured_at or manifest.get("captured_at") or "")
        if not captured_at:
            captured_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        result = build_oe_leaguepedia_series_crosswalk(
            loaded["oe"],
            loaded["scoreboardgames"],
            loaded["matchschedule"],
            source_receipt=source_receipt,
            source_records=records,
            competition_mapping=competition_mapping,
            captured_at=captured_at,
            raw_source_bytes=raw_bytes,
            allow_partial=args.allow_partial,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output), "status": result["status"], "coverage": result["coverage"], "issues": len(result["issues"]), "sha256": result["crosswalk_sha256"]}, indent=2, sort_keys=True))
        return 0 if result["status"] != "rejected_incomplete" else 2
    except (CrosswalkError, OSError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
