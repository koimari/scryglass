"""Deterministic helper for rebuilding the L4 development decision.

This helper prints canonical report content for review.  It does not write,
promote, authorize, or publish artifacts.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys

from .model import ROOT, compare_candidates


def _plain(value):
    if isinstance(value, dict) or hasattr(value, "items"):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def build_decision(root: Path = ROOT) -> dict:
    data = root / "data/lol/v2/models/player"
    config = json.loads((data / "player-rating-config.json").read_text(encoding="utf-8"))
    fixtures = json.loads((data / "player-rating-fixtures.json").read_text(encoding="utf-8"))
    cutoff = json.loads(
        (data / "player-rating-development-report.json").read_text(encoding="utf-8")
    )["evaluation_cutoff"]
    return _plain(compare_candidates(config, fixtures, cutoff))


def _canonical_bytes(value) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def rebuild_owned_artifacts(root: Path = ROOT) -> dict:
    data = root / "data/lol/v2/models/player"
    report_path = data / "player-rating-development-report.json"
    manifest_path = data / "player-rating-manifest.json"
    identity_path = data / "player-rating-candidate-identity.json"
    decision = build_decision(root)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for key in (
        "status",
        "selected_candidate_id",
        "selection_decision_sha256",
        "eligible_origin_count",
        "common_origin_ids",
        "common_origin_sha256",
        "verified_selectable_origin_sha256",
        "diagnostics",
    ):
        report[key] = decision[key]
    _write_json(report_path, report)
    config_path = data / "player-rating-config.json"
    fixtures_path = data / "player-rating-fixtures.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selected_candidate_id"] = report["selected_candidate_id"]
    manifest["selection_decision_sha256"] = report["selection_decision_sha256"]
    manifest["artifact_hashes"] = {
        "config": _sha256(config_path.read_bytes()),
        "fixtures": _sha256(fixtures_path.read_bytes()),
        "report": _sha256(report_path.read_bytes()),
    }
    _write_json(manifest_path, manifest)
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    for ref in identity["artifacts"]:
        path = root / ref["locator"]
        raw = path.read_bytes()
        ref["raw_sha256"] = _sha256(raw)
        if ref["media_type"] == "application/json":
            ref["canonical_sha256"] = _sha256(
                _canonical_bytes(json.loads(raw.decode("utf-8")))
            )
        else:
            ref["canonical_sha256"] = ref["raw_sha256"]
    _write_json(identity_path, identity)
    return {
        "candidate_identity_sha256": _sha256(identity_path.read_bytes()),
        "decision": decision,
        "manifest_sha256": _sha256(manifest_path.read_bytes()),
        "report_sha256": _sha256(report_path.read_bytes()),
        "stdout_hash_definition": "sha256 of complete canonical compact JSON UTF-8 stdout bytes including exactly one trailing LF",
    }


def serialize_stdout(value) -> bytes:
    return _canonical_bytes(value) + b"\n"


def main() -> None:
    output = (
        rebuild_owned_artifacts()
        if sys.argv[1:] == ["--write"]
        else build_decision()
    )
    sys.stdout.buffer.write(serialize_stdout(output))


if __name__ == "__main__":
    main()
