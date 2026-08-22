"""Build source-bound retrospective Tier List comparisons.

The legacy entry point compares the frozen current candidate with V2.  The
four-way entry point compares all registered rating variants on one exact
model-eligible row universe.  The benchmark does not fit a model or change a
public Tier List.  Every input is checked against an external byte hash before
any row is read.
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
    VARIANTS as TIER_VARIANTS,
    _candidate_rows,
    canonical_json_bytes,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


SCHEMA_VERSION = "scryglass:future-value-tierlist-full-census-diff:v1"
RECEIPT_SCHEMA_VERSION = "scryglass:future-value-tierlist-full-census-diff-receipt:v1"
FINAL_V2_SCOREABILITY_SCHEMA_VERSION = (
    "scryglass:future-value-final-v2-full-census-scoreability:v1"
)
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


def _verify_fourway_candidate(
    candidate: Mapping[str, Any],
    *,
    variant: str,
    expected_game_count: int,
    expected_identity: str,
    expected_source_as_of: str,
    expected_source_receipt_sha256: str,
) -> tuple[dict[tuple[str, str, str, str], dict[str, Any]], str]:
    """Verify one retrospective candidate on the shared eligible census."""

    rows, artifact_sha256 = _verify_candidate(
        candidate,
        label=f"{variant} candidate",
        expected_game_count=expected_game_count,
        expected_identity=expected_identity,
        expected_source_as_of=expected_source_as_of,
    )
    override = candidate.get("pre_map_offset_override")
    if not isinstance(override, Mapping) or override.get("applied") is not True:
        raise FullCensusTierDiffError(f"{variant} candidate pre-map offset binding is missing")
    if (
        override.get("game_count") != expected_game_count
        or override.get("game_identity_sha256") != expected_identity
    ):
        raise FullCensusTierDiffError(f"{variant} candidate offset universe changed")
    provenance = override.get("provenance")
    if not isinstance(provenance, Mapping):
        # Older pooled artifacts used a sibling field.  Accept it for a
        # compatibility read, while requiring the same strict-prior values.
        provenance = candidate.get("pre_map_offset_provenance")
    if not isinstance(provenance, Mapping):
        raise FullCensusTierDiffError(f"{variant} candidate offset provenance is missing")
    if provenance.get("timing") != "strict_prior_pre_map":
        raise FullCensusTierDiffError(f"{variant} candidate offset timing changed")
    if provenance.get("source_receipt_sha256") != expected_source_receipt_sha256:
        raise FullCensusTierDiffError(f"{variant} candidate offset source receipt changed")
    if provenance.get("source_identity_sha256") != expected_identity:
        raise FullCensusTierDiffError(f"{variant} candidate offset source identity changed")
    if provenance.get("source_game_count") != expected_game_count:
        raise FullCensusTierDiffError(f"{variant} candidate offset count changed")
    expected_producer = f"future_value_rating:{variant}"
    if provenance.get("producer") != expected_producer:
        raise FullCensusTierDiffError(f"{variant} candidate offset producer changed")
    if provenance.get("authority") is not False:
        raise FullCensusTierDiffError(f"{variant} candidate offset authority changed")
    claimed_receipt = _require_hash(
        provenance.get("receipt_sha256"), f"{variant} candidate offset receipt hash"
    )
    unsigned = dict(provenance)
    unsigned.pop("receipt_sha256", None)
    if _hash_bytes(canonical_json_bytes(unsigned)) != claimed_receipt:
        raise FullCensusTierDiffError(f"{variant} candidate offset receipt changed")
    return rows, artifact_sha256


def _fourway_movement(
    reference_rows: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    selected_rows: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pair one candidate with current-only rows after exact identity checks."""

    reference_keys = set(reference_rows)
    selected_keys = set(selected_rows)
    if reference_keys != selected_keys:
        missing = sorted(reference_keys - selected_keys)
        extra = sorted(selected_keys - reference_keys)
        raise FullCensusTierDiffError(
            "four-way Tier row universes differ "
            f"(missing={len(missing)}, extra={len(extra)})"
        )
    rows: list[dict[str, Any]] = []
    movements: list[int] = []
    changed_tiers = 0
    for key in sorted(reference_keys):
        reference = reference_rows[key]
        selected = selected_rows[key]
        rank_delta = int(reference["rank"]) - int(selected["rank"])
        tier_changed = reference["tier_bucket"] != selected["tier_bucket"]
        movements.append(rank_delta)
        changed_tiers += int(tier_changed)
        rows.append(
            {
                "key": _row_identity(key),
                "reference": _row_values(reference),
                "selected": _row_values(selected),
                "delta": {
                    "rank_delta": rank_delta,
                    "tier_changed": tier_changed,
                },
            }
        )
    return rows, {
        "row_count": len(rows),
        "changed_rank_count": sum(value != 0 for value in movements),
        "changed_tier_count": changed_tiers,
        "mean_absolute_rank_movement": sum(abs(value) for value in movements) / len(movements),
        "maximum_absolute_rank_movement": max(map(abs, movements), default=0),
        "paired_rows_sha256": _hash_bytes(canonical_json_bytes(rows)),
    }


def build_full_census_fourway_diff(
    *,
    source_receipt_path: Path | str,
    expected_source_receipt_file_sha256: str,
    expected_source_receipt_sha256: str,
    baseline_candidate_path: Path | str,
    expected_baseline_candidate_sha256: str,
    variant_candidate_paths: Mapping[str, Path | str] | None = None,
    expected_variant_candidate_sha256: Mapping[str, str] | None = None,
    candidate_paths: Mapping[str, Path | str] | None = None,
    expected_candidate_sha256: Mapping[str, str] | None = None,
    v2_candidate_path: Path | str | None = None,
    expected_v2_candidate_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a retrospective four-variant diff on one exact eligible universe.

    ``variant_candidate_paths`` is the preferred argument.  The two
    ``candidate_*`` names are accepted as a short compatibility spelling for
    callers that already use a candidate mapping.
    """

    paths = variant_candidate_paths if variant_candidate_paths is not None else candidate_paths
    hashes = (
        expected_variant_candidate_sha256
        if expected_variant_candidate_sha256 is not None
        else expected_candidate_sha256
    )
    if not isinstance(paths, Mapping) or set(paths) != set(TIER_VARIANTS):
        raise FullCensusTierDiffError(
            "four variant candidate paths must contain current_only, future_player_form, scaling_curve, and both"
        )
    if not isinstance(hashes, Mapping) or set(hashes) != set(TIER_VARIANTS):
        raise FullCensusTierDiffError("four variant candidate hashes are incomplete")
    source, source_file = _load_source(
        source_receipt_path,
        expected_source_receipt_file_sha256=expected_source_receipt_file_sha256,
        expected_source_receipt_sha256=expected_source_receipt_sha256,
    )
    baseline_path = _safe_file(baseline_candidate_path, "baseline candidate")
    baseline_file = _verify_file(
        baseline_path,
        expected_baseline_candidate_sha256,
        "baseline candidate",
    )
    baseline = _load_json(baseline_path, "baseline candidate")
    baseline_rows, baseline_artifact_sha256 = _verify_candidate(
        baseline,
        label="baseline candidate",
        expected_game_count=source["accepted_game_count"],
        expected_identity=source["accepted_identity_sha256"],
        expected_source_as_of=source["source_as_of"],
    )
    candidate_rows: dict[str, dict[tuple[str, str, str, str], dict[str, Any]]] = {}
    candidate_artifacts: dict[str, dict[str, Any]] = {}
    for variant in TIER_VARIANTS:
        path = _safe_file(paths[variant], f"{variant} candidate")
        file_binding = _verify_file(path, hashes[variant], f"{variant} candidate")
        candidate = _load_json(path, f"{variant} candidate")
        rows, artifact_sha256 = _verify_fourway_candidate(
            candidate,
            variant=variant,
            expected_game_count=source["model_eligible_game_count"],
            expected_identity=source["model_eligible_identity_sha256"],
            expected_source_as_of=source["source_as_of"],
            expected_source_receipt_sha256=source["source_receipt_sha256"],
        )
        candidate_rows[variant] = rows
        candidate_artifacts[variant] = {
            **file_binding,
            "artifact_sha256": artifact_sha256,
            "source_game_count": source["model_eligible_game_count"],
            "source_identity_sha256": source["model_eligible_identity_sha256"],
        }
    reference = candidate_rows["current_only"]
    comparisons: dict[str, Any] = {}
    paired_rows: dict[str, list[dict[str, Any]]] = {}
    for variant in TIER_VARIANTS:
        rows, movement = _fourway_movement(reference, candidate_rows[variant])
        comparisons[variant] = {
            "reference_variant": "current_only",
            **movement,
        }
        paired_rows[variant] = rows
    reference_keys = sorted(reference)
    common_identity = _hash_bytes(
        canonical_json_bytes([_row_identity(key) for key in reference_keys])
    )
    report: dict[str, Any] = {
        "schema_version": "scryglass:future-value-tierlist-full-census-fourway:v1",
        "status": "research_only",
        "authority": dict(AUTHORITY),
        "timing": {
            "mode": "retrospective_full_model_eligible_census",
            "chronological_evaluation_suitable": False,
            "validation_offsets_used": False,
        },
        "source": source,
        "baseline_public_candidate": {
            "status": "unchanged_non_comparable_full_census_reference",
            "path": baseline_file,
            "artifact_sha256": baseline_artifact_sha256,
            "source_game_count": source["accepted_game_count"],
            "source_identity_sha256": source["accepted_identity_sha256"],
        },
        "candidate_universe": {
            "variant_count": len(TIER_VARIANTS),
            "variants": list(TIER_VARIANTS),
            "game_count": source["model_eligible_game_count"],
            "game_identity_sha256": source["model_eligible_identity_sha256"],
            "target_rows_sha256": source.get("target_rows_sha256"),
            "identical": True,
            "row_identity_fields": ["scope_id", "patch", "role", "champion_id"],
            "common_row_count": len(reference_keys),
            "common_identity_sha256": common_identity,
        },
        "inputs": {
            "source_receipt": source_file["source_receipt"],
            "baseline_candidate": baseline_file,
            "variant_candidates": candidate_artifacts,
        },
        "comparisons": comparisons,
        "rows": paired_rows,
        "blockers": [
            "retrospective_full_census_model_fit_not_chronological_evaluation",
            "model_eligible_census_differs_from_accepted_census",
            "chronological_subset_evidence_not_used_for_full_census_offsets",
            "public_tierlist_authority_missing",
        ],
    }
    report["report_sha256"] = _hash_bytes(canonical_json_bytes(report))
    return report


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


def audit_final_v2_full_census_scoreability(
    *,
    source_receipt_path: Path | str,
    expected_source_receipt_file_sha256: str,
    expected_source_receipt_sha256: str,
    model_receipt_path: Path | str,
    expected_model_receipt_file_sha256: str,
    expected_model_receipt_sha256: str,
) -> dict[str, Any]:
    """Audit whether the final V2 fit can score the accepted census.

    The final fit receipt is the only input used for this audit.  It binds the
    current-rating feature ledger and its game identity.  A receipt for an
    eligible subset cannot be extended to the accepted census, so this helper
    returns a blocked research audit and never creates imputed score rows.
    """

    source, source_file = _load_source(
        source_receipt_path,
        expected_source_receipt_file_sha256=expected_source_receipt_file_sha256,
        expected_source_receipt_sha256=expected_source_receipt_sha256,
    )
    model_path = _safe_file(model_receipt_path, "final V2 model receipt")
    model_file = _verify_file(
        model_path,
        expected_model_receipt_file_sha256,
        "final V2 model receipt",
    )
    model = _load_json(model_path, "final V2 model receipt")
    if model.get("schema_version") != "scryglass:future-value-model-fit:v1":
        raise FullCensusTierDiffError("final V2 model receipt schema changed")
    claimed_model_hash = _require_hash(
        model.get("receipt_sha256"), "final V2 model receipt hash"
    )
    unsigned_model = dict(model)
    unsigned_model.pop("receipt_sha256", None)
    if _hash_bytes(canonical_json_bytes(unsigned_model)) != claimed_model_hash:
        raise FullCensusTierDiffError("final V2 model receipt self hash changed")
    if claimed_model_hash != _require_hash(
        expected_model_receipt_sha256, "expected final V2 model receipt hash"
    ):
        raise FullCensusTierDiffError("final V2 model receipt hash changed")

    binding = model.get("source_binding")
    if not isinstance(binding, Mapping):
        raise FullCensusTierDiffError("final V2 model source binding is missing")
    if binding.get("source_game_count") != source["accepted_game_count"]:
        raise FullCensusTierDiffError("final V2 model accepted count changed")
    if binding.get("source_identity_sha256") != source["accepted_identity_sha256"]:
        raise FullCensusTierDiffError("final V2 model accepted identity changed")
    if binding.get("source_receipt_sha256") != source["source_receipt_sha256"]:
        raise FullCensusTierDiffError("final V2 model source receipt changed")
    if binding.get("model_eligible_game_count") != source["model_eligible_game_count"]:
        raise FullCensusTierDiffError("final V2 model eligible count changed")
    if binding.get("model_eligible_identity_sha256") != source[
        "model_eligible_identity_sha256"
    ]:
        raise FullCensusTierDiffError("final V2 model eligible identity changed")

    ledger = model.get("feature_ledger_binding")
    if not isinstance(ledger, Mapping):
        raise FullCensusTierDiffError("final V2 feature ledger binding is missing")
    artifact = ledger.get("artifact")
    if not isinstance(artifact, Mapping):
        raise FullCensusTierDiffError("final V2 feature ledger artifact binding is missing")
    ledger_rows = ledger.get("rows")
    try:
        ledger_rows = int(ledger_rows)
    except (TypeError, ValueError) as error:
        raise FullCensusTierDiffError("final V2 feature ledger row count is invalid") from error
    if ledger_rows < 0:
        raise FullCensusTierDiffError("final V2 feature ledger row count is negative")
    ledger_identity = _require_hash(
        ledger.get("game_identity_sha256"), "final V2 feature ledger identity"
    )
    if ledger_rows != model.get("fit_game_count"):
        raise FullCensusTierDiffError("final V2 feature ledger fit count changed")
    if ledger_identity != model.get("fit_game_identity_sha256"):
        raise FullCensusTierDiffError("final V2 feature ledger fit identity changed")
    if ledger_rows != source["model_eligible_game_count"]:
        raise FullCensusTierDiffError("final V2 feature ledger eligible count changed")
    if ledger_identity != source["model_eligible_identity_sha256"]:
        raise FullCensusTierDiffError("final V2 feature ledger eligible identity changed")

    missing_count = source["accepted_game_count"] - ledger_rows
    blockers = [
        "retrospective_full_census_model_fit_not_chronological_evaluation",
    ]
    if ledger.get("strict_prior_timing") != (
        "source_bound_current_rating_before_snapshot_as_of"
    ):
        blockers.append("final_v2_feature_ledger_strict_prior_timing_missing")
    if missing_count:
        blockers.append("final_v2_feature_ledger_does_not_cover_accepted_census")
    if model.get("status") != "research_only_blocked":
        blockers.append("final_v2_model_status_not_research_only_blocked")
    for blocker in model.get("blockers", ()):
        if isinstance(blocker, str) and blocker not in blockers:
            blockers.append(blocker)
    can_score = not missing_count and len(blockers) == 0
    return {
        "schema_version": FINAL_V2_SCOREABILITY_SCHEMA_VERSION,
        "status": "research_only" if can_score else "research_only_blocked",
        "authority": dict(AUTHORITY),
        "source": {
            **source,
            "accepted_census_binding": source_file["source_receipt"],
        },
        "model": {
            **model_file,
            "receipt_sha256": claimed_model_hash,
            "status": model.get("status"),
            "fit_game_count": model.get("fit_game_count"),
            "fit_game_identity_sha256": model.get("fit_game_identity_sha256"),
            "feature_ledger": {
                "artifact": dict(artifact),
                "rows": ledger_rows,
                "game_identity_sha256": ledger_identity,
                "strict_prior_timing": ledger.get("strict_prior_timing"),
            },
        },
        "coverage": {
            "accepted_game_count": source["accepted_game_count"],
            "scored_game_count": ledger_rows,
            "missing_game_count": missing_count,
            "scored_identity_sha256": ledger_identity,
            "matches_accepted_census": missing_count == 0,
            "matches_model_eligible_census": True,
        },
        "decision": {
            "can_score_accepted_census": can_score,
            "can_build_source_bound_tier_offset_ledger": can_score,
            "can_promote": False,
        },
        "blockers": blockers,
    }


def build_full_census_tier_diff(
    *,
    source_receipt_path: Path | str,
    expected_source_receipt_file_sha256: str,
    expected_source_receipt_sha256: str,
    baseline_candidate_path: Path | str,
    expected_baseline_candidate_sha256: str,
    v2_candidate_path: Path | str | None = None,
    expected_v2_candidate_sha256: str | None = None,
    variant_candidate_paths: Mapping[str, Path | str] | None = None,
    expected_variant_candidate_sha256: Mapping[str, str] | None = None,
    candidate_paths: Mapping[str, Path | str] | None = None,
    expected_candidate_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a verified current versus V2 common-row Tier comparison."""

    if variant_candidate_paths is not None or candidate_paths is not None:
        return build_full_census_fourway_diff(
            source_receipt_path=source_receipt_path,
            expected_source_receipt_file_sha256=expected_source_receipt_file_sha256,
            expected_source_receipt_sha256=expected_source_receipt_sha256,
            baseline_candidate_path=baseline_candidate_path,
            expected_baseline_candidate_sha256=expected_baseline_candidate_sha256,
            variant_candidate_paths=variant_candidate_paths,
            expected_variant_candidate_sha256=expected_variant_candidate_sha256,
            candidate_paths=candidate_paths,
            expected_candidate_sha256=expected_candidate_sha256,
        )
    if v2_candidate_path is None or expected_v2_candidate_sha256 is None:
        raise FullCensusTierDiffError("V2 candidate inputs are required for the legacy diff")

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
    is_fourway = report.get("schema_version") == "scryglass:future-value-tierlist-full-census-fourway:v1"
    stem = "current-fourway-full-census-tier-diff" if is_fourway else "current-v2-full-census-tier-diff"
    report_path = root / f"{stem}.json"
    receipt_path = root / f"{stem}-receipt.json"
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
    parser.add_argument("--v2-candidate", type=Path)
    parser.add_argument("--expected-v2-candidate-sha256")
    parser.add_argument(
        "--variant-candidate",
        action="append",
        default=[],
        metavar="VARIANT=PATH",
        help="Four-way candidate path. Repeat once per registered variant.",
    )
    parser.add_argument(
        "--expected-variant-candidate-sha256",
        action="append",
        default=[],
        metavar="VARIANT=SHA256",
        help="Four-way candidate hash. Repeat once per registered variant.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.variant_candidate or args.expected_variant_candidate_sha256:
        candidate_paths: dict[str, Path] = {}
        candidate_hashes: dict[str, str] = {}
        for item in args.variant_candidate:
            if "=" not in item:
                raise FullCensusTierDiffError("--variant-candidate must be VARIANT=PATH")
            variant, value = item.split("=", 1)
            candidate_paths[variant] = Path(value).expanduser().resolve()
        for item in args.expected_variant_candidate_sha256:
            if "=" not in item:
                raise FullCensusTierDiffError(
                    "--expected-variant-candidate-sha256 must be VARIANT=SHA256"
                )
            variant, value = item.split("=", 1)
            candidate_hashes[variant] = value
        trust_inputs = _load_trust_manifest_inputs(
            trust_manifest_path=args.trust_manifest.expanduser().resolve(),
            expected_trust_manifest_sha256=args.expected_trust_manifest_sha256,
            source_root=args.source_root,
            v2_candidate_path=args.v2_candidate or Path("/dev/null"),
            expected_v2_candidate_sha256=args.expected_v2_candidate_sha256 or "0" * 64,
        )
        inputs = {
            key: value
            for key, value in trust_inputs.items()
            if key not in {"v2_candidate_path", "expected_v2_candidate_sha256"}
        }
        inputs.update(
            {
                "variant_candidate_paths": candidate_paths,
                "expected_variant_candidate_sha256": candidate_hashes,
            }
        )
        report = build_full_census_fourway_diff(**inputs)
    else:
        if args.v2_candidate is None or args.expected_v2_candidate_sha256 is None:
            raise FullCensusTierDiffError(
                "V2 candidate arguments or four-way candidate arguments are required"
            )
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
    if "comparison" in report:
        output = {
            "status": report["status"],
            "report": str(report_path),
            "receipt": str(receipt_path),
            "common_row_count": report["comparison"]["common_row_count"],
            "changed_rank_count": report["comparison"]["changed_rank_count"],
            "changed_tier_count": report["comparison"]["changed_tier_count"],
        }
    else:
        output = {
            "status": report["status"],
            "report": str(report_path),
            "receipt": str(receipt_path),
            "variant_count": len(report["comparisons"]),
            "common_row_count": report["candidate_universe"]["common_row_count"],
            "identical_candidate_universe": report["candidate_universe"]["identical"],
        }
    print(json.dumps(output, sort_keys=True))
    return 0


__all__ = [
    "AUTHORITY",
    "FINAL_V2_SCOREABILITY_SCHEMA_VERSION",
    "FullCensusTierDiffError",
    "RECEIPT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "build_full_census_fourway_diff",
    "build_full_census_tier_diff",
    "canonical_json_bytes",
    "audit_final_v2_full_census_scoreability",
    "sha256_path",
    "write_full_census_tier_diff",
]


if __name__ == "__main__":
    raise SystemExit(main())
