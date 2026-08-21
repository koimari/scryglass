"""Build a source-bound current versus V2 Tier List comparison.

This benchmark compares the frozen current Tier candidate with the frozen V2
candidate on their exact common row universe.  It does not fit a model and it
does not change a public Tier List.  Every input is checked against an
external byte hash before any row is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from lol_kills.research.future_value_rating import (
    validate_future_value_source_receipt_payload,
)
from lol_kills.research.future_value_tierlist import (
    FutureValueTierListError,
    _candidate_rows,
    canonical_json_bytes,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


SCHEMA_VERSION = "scryglass:future-value-tierlist-full-census-diff:v1"
RECEIPT_SCHEMA_VERSION = "scryglass:future-value-tierlist-full-census-diff-receipt:v1"
AUTHORITY = {
    "research_only": True,
    "public_tierlist": False,
    "public_player_rating": False,
    "public_team_rating": False,
    "public_probability": False,
    "promotion": False,
    "merge": False,
    "deployment": False,
    "betting": False,
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_ROW_FIELDS = (
    "rank",
    "tier_bucket",
    "tier_value_pp",
    "strength_score",
    "strength_sd_logit",
    "rating",
    "played_maps",
)


class FullCensusTierDiffError(ValueError):
    """The paired Tier comparison cannot prove its source or values."""


def _hash_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(value: object, label: str) -> str:
    text = str(value or "").lower()
    if _SHA256.fullmatch(text) is None:
        raise FullCensusTierDiffError(f"{label} must be a SHA-256 hash")
    return text


def _safe_file(path: Path | str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise FullCensusTierDiffError(f"{label} is missing or unsafe: {resolved}")
    return resolved


def _verify_file(path: Path, expected_sha256: object, label: str) -> dict[str, Any]:
    expected = _require_hash(expected_sha256, f"expected {label} hash")
    actual = sha256_path(path)
    if actual != expected:
        raise FullCensusTierDiffError(f"{label} bytes changed")
    return {
        "locator": str(path),
        "bytes": path.stat().st_size,
        "sha256": actual,
    }


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FullCensusTierDiffError(f"{label} cannot be read") from error
    if not isinstance(value, dict):
        raise FullCensusTierDiffError(f"{label} must be a JSON object")
    return value


def _verify_candidate(
    candidate: Mapping[str, Any],
    *,
    label: str,
    expected_game_count: int,
    expected_identity: str,
    expected_source_as_of: str,
) -> tuple[dict[tuple[str, str, str, str], dict[str, Any]], str]:
    """Verify one candidate and return its normalized Tier rows."""

    if candidate.get("schema_version") != "scryglass:champion-role-elo-candidate:v2":
        raise FullCensusTierDiffError(f"{label} schema changed")
    if candidate.get("status") != "development_only":
        raise FullCensusTierDiffError(f"{label} is not development-only")
    if candidate.get("development_only") is not True:
        raise FullCensusTierDiffError(f"{label} development flag changed")
    if candidate.get("production_eligible") is not False:
        raise FullCensusTierDiffError(f"{label} production authority changed")
    if candidate.get("publication_eligible") is not False:
        raise FullCensusTierDiffError(f"{label} publication authority changed")
    claimed_artifact = _require_hash(
        candidate.get("artifact_sha256"), f"{label} artifact hash"
    )
    unsigned = dict(candidate)
    unsigned.pop("artifact_sha256", None)
    if _hash_bytes(canonical_json_bytes(unsigned)) != claimed_artifact:
        raise FullCensusTierDiffError(f"{label} self hash changed")
    if candidate.get("as_of") != expected_source_as_of:
        raise FullCensusTierDiffError(f"{label} as_of changed")
    source = candidate.get("source")
    if not isinstance(source, Mapping):
        raise FullCensusTierDiffError(f"{label} source binding is missing")
    if source.get("maps_replayed") != expected_game_count:
        raise FullCensusTierDiffError(f"{label} map count changed")
    if source.get("maps_used_in_joint_likelihood") != expected_game_count:
        raise FullCensusTierDiffError(f"{label} likelihood count changed")
    if source.get("source_identity_sha256") != expected_identity:
        raise FullCensusTierDiffError(f"{label} source identity changed")
    if source.get("source_latest_replayed") != expected_source_as_of:
        raise FullCensusTierDiffError(f"{label} source cutoff changed")
    try:
        rows = _candidate_rows(candidate)
    except FutureValueTierListError as error:
        raise FullCensusTierDiffError(f"{label} Tier row identity is invalid") from error
    for key, row in rows.items():
        if len(key) != 4 or any(not str(part).strip() for part in key):
            raise FullCensusTierDiffError(f"{label} contains an empty row identity")
        try:
            rank = int(row.get("rank"))
        except (TypeError, ValueError) as error:
            raise FullCensusTierDiffError(f"{label} contains an invalid rank") from error
        if rank <= 0:
            raise FullCensusTierDiffError(f"{label} contains a non-positive rank")
        tier = str(row.get("tier_bucket") or "").strip()
        if not tier:
            raise FullCensusTierDiffError(f"{label} contains an empty tier")
        for field in _ROW_FIELDS[2:]:
            value = row.get(field)
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise FullCensusTierDiffError(
                    f"{label} contains a non-numeric {field}"
                ) from error
            if not math.isfinite(number):
                raise FullCensusTierDiffError(f"{label} contains a non-finite {field}")
    return rows, claimed_artifact


def _row_identity(key: tuple[str, str, str, str]) -> dict[str, str]:
    return {
        "scope_id": key[0],
        "patch": key[1],
        "role": key[2],
        "champion_id": key[3],
    }


def _row_values(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in _ROW_FIELDS}


def _build_rows(
    baseline_rows: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    v2_rows: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    common_keys = sorted(set(baseline_rows).intersection(v2_rows))
    if not common_keys:
        raise FullCensusTierDiffError("baseline and V2 have no common Tier rows")
    rows: list[dict[str, Any]] = []
    rank_moves: list[int] = []
    changed_tiers = 0
    for key in common_keys:
        baseline = baseline_rows[key]
        v2 = v2_rows[key]
        rank_delta = int(baseline["rank"]) - int(v2["rank"])
        tier_changed = baseline["tier_bucket"] != v2["tier_bucket"]
        rank_moves.append(rank_delta)
        changed_tiers += int(tier_changed)
        rows.append(
            {
                "key": _row_identity(key),
                "baseline": _row_values(baseline),
                "v2": _row_values(v2),
                "delta": {
                    "rank_delta": rank_delta,
                    "tier_changed": tier_changed,
                },
            }
        )
    row_ids = [_row_identity(key) for key in common_keys]
    movement = {
        "common_row_count": len(common_keys),
        "baseline_only_row_count": len(set(baseline_rows) - set(v2_rows)),
        "v2_only_row_count": len(set(v2_rows) - set(baseline_rows)),
        "changed_rank_count": sum(move != 0 for move in rank_moves),
        "changed_tier_count": changed_tiers,
        "mean_absolute_rank_movement": sum(abs(move) for move in rank_moves)
        / len(rank_moves),
        "maximum_absolute_rank_movement": max(abs(move) for move in rank_moves),
    }
    return rows, {
        "common_identity_sha256": _hash_bytes(canonical_json_bytes(row_ids)),
        **movement,
    }


def _load_source(
    source_receipt_path: Path | str,
    *,
    expected_source_receipt_file_sha256: str,
    expected_source_receipt_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _safe_file(source_receipt_path, "source receipt")
    file_binding = _verify_file(
        path,
        expected_source_receipt_file_sha256,
        "source receipt",
    )
    receipt = _load_json(path, "source receipt")
    try:
        accepted, eligible = validate_future_value_source_receipt_payload(
            receipt,
            expected_receipt_sha256=expected_source_receipt_sha256,
        )
    except ValueError as error:
        raise FullCensusTierDiffError("source receipt is not canonical") from error
    source = {
        "source_as_of": str(receipt["source_as_of"]),
        "source_receipt_sha256": str(receipt["receipt_sha256"]),
        "accepted_game_count": len(accepted),
        "accepted_identity_sha256": identity_sha256(accepted),
        "model_eligible_game_count": len(eligible),
        "model_eligible_identity_sha256": identity_sha256(eligible),
        "accepted_game_ids_sha256": identity_sha256(accepted),
        "model_eligible_game_ids_sha256": identity_sha256(eligible),
    }
    return source, {"source_receipt": file_binding}


def build_full_census_tier_diff(
    *,
    source_receipt_path: Path | str,
    expected_source_receipt_file_sha256: str,
    expected_source_receipt_sha256: str,
    baseline_candidate_path: Path | str,
    expected_baseline_candidate_sha256: str,
    v2_candidate_path: Path | str,
    expected_v2_candidate_sha256: str,
) -> dict[str, Any]:
    """Build a verified current versus V2 common-row Tier comparison."""

    source, source_file = _load_source(
        source_receipt_path,
        expected_source_receipt_file_sha256=expected_source_receipt_file_sha256,
        expected_source_receipt_sha256=expected_source_receipt_sha256,
    )
    baseline_path = _safe_file(baseline_candidate_path, "baseline candidate")
    v2_path = _safe_file(v2_candidate_path, "V2 candidate")
    baseline_file = _verify_file(
        baseline_path,
        expected_baseline_candidate_sha256,
        "baseline candidate",
    )
    v2_file = _verify_file(v2_path, expected_v2_candidate_sha256, "V2 candidate")
    baseline = _load_json(baseline_path, "baseline candidate")
    v2 = _load_json(v2_path, "V2 candidate")
    baseline_rows, baseline_artifact_sha256 = _verify_candidate(
        baseline,
        label="baseline candidate",
        expected_game_count=source["accepted_game_count"],
        expected_identity=source["accepted_identity_sha256"],
        expected_source_as_of=source["source_as_of"],
    )
    v2_rows, v2_artifact_sha256 = _verify_candidate(
        v2,
        label="V2 candidate",
        expected_game_count=source["model_eligible_game_count"],
        expected_identity=source["model_eligible_identity_sha256"],
        expected_source_as_of=source["source_as_of"],
    )
    override = v2.get("pre_map_offset_override")
    if not isinstance(override, Mapping) or override.get("applied") is not True:
        raise FullCensusTierDiffError("V2 pre-map offset binding is missing")
    if override.get("game_count") != source["model_eligible_game_count"]:
        raise FullCensusTierDiffError("V2 offset game count changed")
    if override.get("game_identity_sha256") != source["model_eligible_identity_sha256"]:
        raise FullCensusTierDiffError("V2 offset identity changed")
    provenance = override.get("provenance")
    if not isinstance(provenance, Mapping):
        raise FullCensusTierDiffError("V2 offset provenance is missing")
    if provenance.get("timing") != "strict_prior_pre_map":
        raise FullCensusTierDiffError("V2 offset timing changed")
    if provenance.get("source_receipt_sha256") != source["source_receipt_sha256"]:
        raise FullCensusTierDiffError("V2 offset source receipt changed")
    paired_rows, movement = _build_rows(baseline_rows, v2_rows)
    paired_rows_sha256 = _hash_bytes(canonical_json_bytes(paired_rows))
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "research_only",
        "authority": dict(AUTHORITY),
        "comparison": {
            "reference": "current_only",
            "candidate": "future_player_form",
            "row_identity_fields": ["scope_id", "patch", "role", "champion_id"],
            **movement,
            "paired_rows_sha256": paired_rows_sha256,
        },
        "source": source,
        "inputs": {
            "source_receipt": source_file["source_receipt"],
            "baseline_candidate": {
                **baseline_file,
                "artifact_sha256": baseline_artifact_sha256,
                "source_game_count": source["accepted_game_count"],
                "source_identity_sha256": source["accepted_identity_sha256"],
            },
            "v2_candidate": {
                **v2_file,
                "artifact_sha256": v2_artifact_sha256,
                "source_game_count": source["model_eligible_game_count"],
                "source_identity_sha256": source["model_eligible_identity_sha256"],
            },
        },
        "rows": paired_rows,
        "blockers": [
            "retrospective_full_census_model_fit_not_chronological_evaluation",
            "model_eligible_census_differs_from_accepted_census",
            "public_tierlist_authority_missing",
        ],
    }
    report["report_sha256"] = _hash_bytes(
        canonical_json_bytes({key: value for key, value in report.items()})
    )
    return report


def write_full_census_tier_diff(
    report: Mapping[str, Any],
    *,
    output_root: Path | str,
) -> tuple[Path, Path]:
    """Write a report and receipt into a new empty directory."""

    claimed_report_hash = _require_hash(report.get("report_sha256"), "report hash")
    unsigned_report = dict(report)
    unsigned_report.pop("report_sha256", None)
    if _hash_bytes(canonical_json_bytes(unsigned_report)) != claimed_report_hash:
        raise FullCensusTierDiffError("report self hash changed")
    root = Path(output_root).expanduser().resolve()
    if root.exists() and (root.is_symlink() or any(root.iterdir())):
        raise FullCensusTierDiffError("output root must be empty and safe")
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "current-v2-full-census-tier-diff.json"
    receipt_path = root / "current-v2-full-census-tier-diff-receipt.json"
    report_raw = canonical_json_bytes(dict(report)) + b"\n"
    report_path.write_bytes(report_raw)
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "research_only",
        "authority": dict(AUTHORITY),
        "report": {
            "locator": str(report_path),
            "bytes": len(report_raw),
            "sha256": _hash_bytes(report_raw),
            "report_sha256": report.get("report_sha256"),
        },
        "source": dict(report["source"]),
        "inputs": dict(report["inputs"]),
        "comparison": dict(report["comparison"]),
        "blockers": list(report.get("blockers", [])),
    }
    receipt["receipt_sha256"] = _hash_bytes(canonical_json_bytes(receipt))
    receipt_path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    return report_path, receipt_path


def _load_trust_manifest_inputs(
    *,
    trust_manifest_path: Path,
    expected_trust_manifest_sha256: str,
    source_root: Path,
    v2_candidate_path: Path,
    expected_v2_candidate_sha256: str,
) -> dict[str, Any]:
    """Resolve the pinned freeze into the core comparison arguments."""

    from lol_kills.research.future_value_tierlist import (
        PINNED_TRUST_MANIFEST_RAW_SHA256,
        load_trust_manifest,
    )

    if expected_trust_manifest_sha256 != PINNED_TRUST_MANIFEST_RAW_SHA256:
        raise FullCensusTierDiffError("Tier trust manifest is not code-pinned")
    trust = load_trust_manifest(
        trust_manifest_path,
        expected_raw_sha256=expected_trust_manifest_sha256,
    )
    source_root = source_root.expanduser().resolve()
    source_receipt = _safe_file(
        source_root / "future-value-source-receipt.json",
        "frozen source receipt",
    )
    baseline = _safe_file(
        source_root / str(trust["baseline_candidate"]["locator"]),
        "frozen baseline candidate",
    )
    return {
        "source_receipt_path": source_receipt,
        "expected_source_receipt_file_sha256": trust["source"][
            "source_receipt_file_sha256"
        ],
        "expected_source_receipt_sha256": trust["source"]["source_receipt_sha256"],
        "baseline_candidate_path": baseline,
        "expected_baseline_candidate_sha256": trust["baseline_candidate"]["raw_sha256"],
        "v2_candidate_path": v2_candidate_path,
        "expected_v2_candidate_sha256": expected_v2_candidate_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trust-manifest", type=Path, required=True)
    parser.add_argument("--expected-trust-manifest-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--v2-candidate", type=Path, required=True)
    parser.add_argument("--expected-v2-candidate-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    inputs = _load_trust_manifest_inputs(
        trust_manifest_path=args.trust_manifest.expanduser().resolve(),
        expected_trust_manifest_sha256=args.expected_trust_manifest_sha256,
        source_root=args.source_root,
        v2_candidate_path=args.v2_candidate.expanduser().resolve(),
        expected_v2_candidate_sha256=args.expected_v2_candidate_sha256,
    )
    report = build_full_census_tier_diff(**inputs)
    report_path, receipt_path = write_full_census_tier_diff(
        report,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(report_path),
                "receipt": str(receipt_path),
                "common_row_count": report["comparison"]["common_row_count"],
                "changed_rank_count": report["comparison"]["changed_rank_count"],
                "changed_tier_count": report["comparison"]["changed_tier_count"],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "AUTHORITY",
    "FullCensusTierDiffError",
    "RECEIPT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "build_full_census_tier_diff",
    "canonical_json_bytes",
    "sha256_path",
    "write_full_census_tier_diff",
]


if __name__ == "__main__":
    raise SystemExit(main())
