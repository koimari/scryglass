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
import os
import re
import stat
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from lol_kills.research.oe_leaguepedia_series_crosswalk import (
    CrosswalkError,
    build_oe_leaguepedia_series_crosswalk,
)
from lol_kills.research.oe_leaguepedia_alias_derivation import (
    AliasDerivationError,
    load_verified_alias_mapping,
)
from lol_kills.research.future_value_rating import (
    LEAGUEPEDIA_CROSSWALK_RECEIPT_AUTHORITY,
    LEAGUEPEDIA_CROSSWALK_RECEIPT_SCHEMA_VERSION,
    _leaguepedia_assignment_sha256,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


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


def _safe_capture_path(root: Path, relative: Any, *, label: str) -> Path:
    raw_relative = Path(str(relative or ""))
    if raw_relative.is_absolute() or ".." in raw_relative.parts:
        raise CrosswalkError(f"capture path escapes its root: {label}")
    lexical_root = Path(os.path.abspath(root))
    candidate = lexical_root / raw_relative
    try:
        relative_parts = candidate.relative_to(lexical_root).parts
    except ValueError as error:
        raise CrosswalkError(f"capture path escapes its root: {label}") from error
    try:
        root_mode = os.lstat(lexical_root).st_mode
    except OSError as error:
        raise CrosswalkError(f"capture file is missing or unsafe: {label}") from error
    if stat.S_ISLNK(root_mode):
        raise CrosswalkError(f"capture file is missing or unsafe: {label}")
    current = lexical_root
    for part in relative_parts:
        current = current / part
        try:
            mode = os.lstat(current).st_mode
        except OSError as error:
            raise CrosswalkError(
                f"capture file is missing or unsafe: {label}"
            ) from error
        if stat.S_ISLNK(mode):
            raise CrosswalkError(f"capture file is missing or unsafe: {label}")
    path = candidate.resolve(strict=True)
    try:
        path.relative_to(lexical_root.resolve(strict=True))
    except ValueError as error:
        raise CrosswalkError(f"capture path escapes its root: {label}") from error
    if not path.is_file():
        raise CrosswalkError(f"capture file is missing or unsafe: {label}")
    return path


def _capture_binding(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    raw = path.read_bytes()
    claimed = str(manifest.get("manifest_sha256") or "").lower()
    body = dict(manifest)
    body.pop("manifest_sha256", None)
    if claimed != _sha256(_canonical_bytes(body)):
        raise CrosswalkError("capture manifest self-hash changed")
    root = path.resolve().parent
    assembled = manifest.get("assembled")
    if not isinstance(assembled, Mapping):
        raise CrosswalkError("capture manifest assembled records are missing")
    verified_assembled: dict[str, dict[str, Any]] = {}
    for label in ("ScoreboardGames", "MatchSchedule", "Tournaments"):
        record = assembled.get(label)
        if not isinstance(record, Mapping):
            raise CrosswalkError(f"capture manifest assembled record is missing: {label}")
        source = _safe_capture_path(root, record.get("path"), label=label)
        source_raw = source.read_bytes()
        if record.get("bytes") != len(source_raw) or str(record.get("sha256") or "").lower() != _sha256(source_raw):
            raise CrosswalkError(f"capture manifest assembled bytes changed: {label}")
        verified_assembled[label] = {
            "path": str(source),
            "bytes": len(source_raw),
            "sha256": _sha256(source_raw),
        }
    response_records = manifest.get("response_records")
    if not isinstance(response_records, list) or not response_records:
        raise CrosswalkError("capture manifest response records are missing")
    for index, record in enumerate(response_records):
        if not isinstance(record, Mapping):
            raise CrosswalkError("capture manifest response record is invalid")
        response_path = _safe_capture_path(
            root, record.get("path"), label=f"response[{index}]"
        )
        response_raw = response_path.read_bytes()
        if record.get("bytes") != len(response_raw) or str(record.get("sha256") or "").lower() != _sha256(response_raw):
            raise CrosswalkError(f"capture response bytes changed: {index}")
    return {
        "path": str(path.resolve()),
        "bytes": len(raw),
        "sha256": _sha256(raw),
        "manifest_sha256": claimed,
        "assembled": verified_assembled,
        "response_record_count": len(response_records),
        "response_records_sha256": _sha256(_canonical_bytes(response_records)),
    }


def _alias_lookup(binding: Mapping[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for entry in binding.get("entries", ()):
        if not isinstance(entry, Mapping) or entry.get("target_system") != "ScoreboardGames":
            continue
        target = str(entry.get("canonical_target_name") or "").strip()
        source_names = entry.get("allowed_source_names")
        if not target or not isinstance(source_names, list):
            raise CrosswalkError("verified alias entry is incomplete")
        for source_name in source_names:
            key = " ".join(re.sub(r"[^\w]+", " ", unicodedata.normalize("NFKC", str(source_name)).casefold(), flags=re.UNICODE).split())
            previous = lookup.get(key)
            if previous is not None and previous != target:
                raise CrosswalkError("verified alias names have conflicting targets")
            lookup[key] = target
    return lookup


def _write_receipt(output: Path, artifact_path: Path, result: Mapping[str, Any], source_receipt: Mapping[str, Any]) -> str:
    assignments = [dict(row) for row in result["assignments"]]
    mapped_ids = sorted(str(row["oe_game_id"]) for row in assignments)
    assignment_hash = str(result.get("assignment_sha256") or "").lower()
    if assignment_hash != _leaguepedia_assignment_sha256(assignments):
        raise CrosswalkError("crosswalk assignment hash does not match assignments")
    artifact_raw = artifact_path.read_bytes()
    accepted_ids = list(source_receipt["accepted_game_ids"])
    receipt: dict[str, Any] = {
        "schema_version": LEAGUEPEDIA_CROSSWALK_RECEIPT_SCHEMA_VERSION,
        "status": "verified_research_only",
        "authority": dict(LEAGUEPEDIA_CROSSWALK_RECEIPT_AUTHORITY),
        "artifact": {
            "path": str(artifact_path.resolve()),
            "bytes": len(artifact_raw),
            "sha256": _sha256(artifact_raw),
        },
        "crosswalk_sha256": result["crosswalk_sha256"],
        "source_receipt_sha256": source_receipt["receipt_sha256"],
        "source_identity_sha256": source_receipt["source_identity_sha256"],
        "accepted_game_count": len(accepted_ids),
        "accepted_game_identity_sha256": identity_sha256(accepted_ids),
        "assignment_count": len(assignments),
        "assignment_sha256": assignment_hash,
        "mapped_game_count": len(mapped_ids),
        "mapped_game_identity_sha256": identity_sha256(mapped_ids),
        "mapped_game_ids": mapped_ids,
    }
    receipt["receipt_sha256"] = _sha256(_canonical_bytes(receipt))
    output.write_bytes(_canonical_bytes(receipt))
    return _sha256(output.read_bytes())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oe", type=Path, required=True, help="downloaded OE game-row JSON")
    parser.add_argument("--scoreboardgames", type=Path, required=True, help="downloaded ScoreboardGames JSON")
    parser.add_argument("--matchschedule", type=Path, required=True, help="downloaded MatchSchedule JSON")
    parser.add_argument("--tournaments", type=Path, required=True, help="downloaded Tournaments JSON")
    parser.add_argument("--capture-manifest", type=Path, required=True, help="capture metadata JSON")
    parser.add_argument("--source-records", type=Path, help="exact OE and assembled source records JSON")
    parser.add_argument("--source-receipt", type=Path, required=True, help="canonical OE source receipt JSON")
    parser.add_argument("--competition-map", type=Path, required=True, help="explicit source-to-Leaguepedia mapping JSON")
    parser.add_argument("--alias-artifact", type=Path, help="verified OE to Leaguepedia team alias artifact")
    parser.add_argument("--alias-stable-team-key-rows-sha256", help="independent stable OE team-key digest")
    parser.add_argument("--output", type=Path, required=True, help="crosswalk output JSON")
    parser.add_argument("--receipt-output", type=Path, help="separate verified crosswalk receipt JSON")
    parser.add_argument("--allow-partial", action="store_true", help="emit explicit mapped-row coverage when some OE rows are unmatched")
    parser.add_argument("--captured-at", default=None, help="override capture manifest timestamp")
    args = parser.parse_args(argv)
    try:
        manifest = _load_json(args.capture_manifest)
        source_records_manifest = (
            manifest
            if args.source_records is None
            else _load_json(args.source_records)
        )
        source_receipt = _source_receipt(args.source_receipt)
        competition_mapping = _load_json(args.competition_map)
        if not isinstance(manifest, Mapping) or not isinstance(source_records_manifest, Mapping) or not isinstance(competition_mapping, Mapping):
            raise CrosswalkError("capture manifest, source records, and competition map must be objects")
        paths = {
            "oe": args.oe,
            "scoreboardgames": args.scoreboardgames,
            "matchschedule": args.matchschedule,
            "tournaments": args.tournaments,
        }
        loaded = {label: _rows(_load_json(path), label=label) for label, path in paths.items()}
        records: dict[str, dict[str, Any]] = {}
        raw_bytes: dict[str, bytes] = {}
        for label, path in paths.items():
            records[label], raw_bytes[label] = _source_record(
                source_records_manifest, label, path, loaded[label]
            )
        if (args.alias_artifact is None) != (
            args.alias_stable_team_key_rows_sha256 is None
        ):
            raise CrosswalkError("alias artifact and stable team-key digest must be supplied together")
        capture_binding = (
            _capture_binding(args.capture_manifest, manifest)
            if manifest.get("manifest_sha256") is not None
            else None
        )
        aliases: dict[str, str] = {}
        alias_binding: dict[str, Any] = {}
        if args.alias_artifact is not None:
            try:
                alias = load_verified_alias_mapping(
                    args.alias_artifact,
                    expected_oe_payload_sha256=records["oe"]["payload_sha256"],
                    expected_scoreboard_payload_sha256=records["scoreboardgames"]["payload_sha256"],
                    expected_matchschedule_payload_sha256=records["matchschedule"]["payload_sha256"],
                    expected_stable_team_key_rows_sha256=args.alias_stable_team_key_rows_sha256,
                    allow_review_only=True,
                )
            except AliasDerivationError as error:
                raise CrosswalkError(f"team alias artifact verification failed: {error}") from error
            aliases = _alias_lookup(alias)
            alias_binding = {
                "path": str(args.alias_artifact.resolve()),
                "artifact_sha256": alias["artifact_sha256"],
                "mapping_sha256": alias["mapping_sha256"],
                "status": alias["status"],
                "stable_team_key_rows_sha256": args.alias_stable_team_key_rows_sha256,
                "accepted_lookup_count": len(aliases),
            }
        if args.receipt_output is not None and (
            capture_binding is None or not alias_binding
        ):
            raise CrosswalkError(
                "verified receipt output requires the capture manifest and alias binding"
            )
        captured_at = str(args.captured_at or manifest.get("captured_at") or "")
        if not captured_at:
            captured_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        result = build_oe_leaguepedia_series_crosswalk(
            loaded["oe"],
            loaded["scoreboardgames"],
            loaded["matchschedule"],
            loaded["tournaments"],
            source_receipt=source_receipt,
            source_records=records,
            competition_mapping=competition_mapping,
            captured_at=captured_at,
            raw_source_bytes=raw_bytes,
            oe_team_aliases=aliases,
            alias_binding=alias_binding,
            allow_partial=args.allow_partial,
        )
        if capture_binding is not None:
            result["capture_manifest_binding"] = capture_binding
            result.pop("crosswalk_sha256", None)
            result["crosswalk_sha256"] = _sha256(_canonical_bytes(result))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        receipt_file_sha256 = None
        if args.receipt_output is not None:
            if args.receipt_output.exists() or args.receipt_output.is_symlink():
                raise CrosswalkError("receipt output already exists")
            args.receipt_output.parent.mkdir(parents=True, exist_ok=True)
            receipt_file_sha256 = _write_receipt(
                args.receipt_output,
                args.output,
                result,
                source_receipt,
            )
        print(json.dumps({"output": str(args.output), "receipt_output": str(args.receipt_output), "receipt_file_sha256": receipt_file_sha256, "status": result["status"], "coverage": result["coverage"], "issues": len(result["issues"]), "sha256": result["crosswalk_sha256"]}, indent=2, sort_keys=True))
        return 0 if result["status"] != "rejected_incomplete" else 2
    except (CrosswalkError, OSError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
