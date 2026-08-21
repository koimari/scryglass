"""Research-only four-way Tier List shadows for future-value ratings.

The public Tier List model uses a chronological team Elo offset.  This module
keeps the champion, patch, role, atom, outcome, and fit contracts fixed while
replacing that offset with one verified strict-prior rating prediction.  The
result is a controlled nuisance-offset study.  It never grants public authority.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from lol_kills.research.future_value_rating import (
    validate_future_value_source_receipt_payload,
)
from lol_kills.v2.tierlists.accepted_census import (
    canonical_game_ids,
    identity_sha256,
)
from lol_kills.v2.tierlists.pooled_candidate import (
    PRE_MAP_OFFSET_PROVENANCE_SCHEMA,
)


SCHEMA_VERSION = "scryglass:future-value-tierlist-fourway:v1"
TRUST_SCHEMA_VERSION = "scryglass:future-value-tierlist-freeze:v1"
PINNED_TRUST_MANIFEST_RAW_SHA256 = (
    "ca47236f4e750a3862612c17a25d3907e76667c0602ff523e032da417dfedf6a"
)
VARIANTS = ("current_only", "future_player_form", "scaling_curve", "both")
REFERENCE_VARIANT = "current_only"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
AUTHORITY = {
    "research_only": True,
    "public_tierlist": False,
    "promotion": False,
    "deployment": False,
    "public_probability": False,
    "odds": False,
    "expected_value": False,
    "recommendation": False,
    "betting": False,
}
ROW_VALUE_FIELDS = (
    "rank",
    "tier_bucket",
    "tier_value_pp",
    "strength_score",
    "strength_sd_logit",
    "rating",
    "played_maps",
    "counterability_status",
    "counterability",
    "matchup_maps",
    "matchup_opponents",
)
NUMERIC_DELTA_FIELDS = (
    "tier_value_pp",
    "strength_score",
    "strength_sd_logit",
    "rating",
    "counterability",
    "matchup_maps",
)


class FutureValueTierListError(ValueError):
    """The four-way Tier List shadow contract failed closed."""


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise FutureValueTierListError("value is not canonical JSON") from error


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise FutureValueTierListError(f"{label} is not a SHA-256 digest")
    return value


def load_trust_manifest(path: Path, *, expected_raw_sha256: str) -> dict[str, Any]:
    """Load one externally pinned, closed freeze manifest."""

    expected = _require_hash(expected_raw_sha256, "expected trust manifest hash")
    if not path.is_file() or path.is_symlink():
        raise FutureValueTierListError("trust manifest is missing or unsafe")
    if sha256_path(path) != expected:
        raise FutureValueTierListError("trust manifest file hash changed")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FutureValueTierListError("trust manifest cannot be read") from error
    if not isinstance(payload, Mapping):
        raise FutureValueTierListError("trust manifest is not an object")
    required = {
        "schema_version",
        "status",
        "authority",
        "source",
        "evaluations",
        "tier_assets",
        "baseline_candidate",
        "trust_root_sha256",
    }
    if set(payload) != required:
        raise FutureValueTierListError("trust manifest schema is closed")
    if payload.get("schema_version") != TRUST_SCHEMA_VERSION:
        raise FutureValueTierListError("trust manifest schema changed")
    if payload.get("status") != "research_only" or payload.get("authority") != AUTHORITY:
        raise FutureValueTierListError("trust manifest authority changed")
    unsigned = dict(payload)
    claimed = _require_hash(unsigned.pop("trust_root_sha256"), "trust root hash")
    if sha256_bytes(canonical_json_bytes(unsigned)) != claimed:
        raise FutureValueTierListError("trust manifest self hash changed")
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, Mapping) or set(evaluations) != set(VARIANTS):
        raise FutureValueTierListError("trust manifest variant set changed")
    for variant, record in evaluations.items():
        if not isinstance(record, Mapping) or set(record) != {"locator", "raw_sha256"}:
            raise FutureValueTierListError(f"{variant} trust record changed")
        _require_hash(record.get("raw_sha256"), f"{variant} model hash")
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise FutureValueTierListError("trust manifest source binding is missing")
    for field in (
        "source_as_of",
        "source_game_count",
        "source_identity_sha256",
        "source_receipt_sha256",
        "source_receipt_file_sha256",
        "player_source_sha256",
        "maps_source_sha256",
        "meta_source_sha256",
    ):
        if field not in source:
            raise FutureValueTierListError(f"trust manifest source field is missing: {field}")
    for field in (
        "source_identity_sha256",
        "source_receipt_sha256",
        "source_receipt_file_sha256",
        "player_source_sha256",
        "maps_source_sha256",
        "meta_source_sha256",
    ):
        _require_hash(source.get(field), f"trust source {field}")
    assets = payload.get("tier_assets")
    if not isinstance(assets, Mapping) or not assets:
        raise FutureValueTierListError("trust manifest Tier assets are missing")
    for locator, digest in assets.items():
        if not isinstance(locator, str) or locator.startswith("/") or ".." in Path(locator).parts:
            raise FutureValueTierListError("Tier asset locator is unsafe")
        _require_hash(digest, f"Tier asset hash {locator}")
    baseline = payload.get("baseline_candidate")
    if not isinstance(baseline, Mapping) or set(baseline) != {"locator", "raw_sha256"}:
        raise FutureValueTierListError("baseline Tier candidate binding changed")
    _require_hash(baseline.get("raw_sha256"), "baseline Tier candidate hash")
    return dict(payload)


def _verify_authority(value: object, label: str) -> None:
    if not isinstance(value, Mapping) or value.get("research_only") is not True:
        raise FutureValueTierListError(f"{label} research authority is missing")
    if any(bool(flag) for name, flag in value.items() if name != "research_only"):
        raise FutureValueTierListError(f"{label} grants public authority")


def _finite_probability(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise FutureValueTierListError(f"{label} is not numeric") from error
    if not math.isfinite(number) or not 0.0 < number < 1.0:
        raise FutureValueTierListError(f"{label} is not a finite open probability")
    return number


def load_prediction_offsets(
    path: Path,
    *,
    variant: str,
    expected_raw_sha256: str,
    source: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    """Verify one fitted evaluation artifact and return strict-prior logits."""

    if variant not in VARIANTS:
        raise FutureValueTierListError("unknown rating variant")
    expected = _require_hash(expected_raw_sha256, f"{variant} expected model hash")
    if not path.is_file() or path.is_symlink() or sha256_path(path) != expected:
        raise FutureValueTierListError(f"{variant} model artifact bytes changed")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FutureValueTierListError(f"{variant} model artifact cannot be read") from error
    if document.get("schema_version") != "scryglass:future-value-four-variant-evaluation:v1":
        raise FutureValueTierListError(f"{variant} evaluation schema changed")
    model_source = document.get("source")
    if not isinstance(model_source, Mapping):
        raise FutureValueTierListError(f"{variant} model source is missing")
    expected_source_fields = {
        "source_as_of": source["source_as_of"],
        "source_game_count": source["source_game_count"],
        "source_identity_sha256": source["source_identity_sha256"],
        "source_receipt_sha256": source["source_receipt_sha256"],
    }
    for field, expected_value in expected_source_fields.items():
        if model_source.get(field) != expected_value:
            raise FutureValueTierListError(f"{variant} model source changed: {field}")
    variants = document.get("variants")
    if not isinstance(variants, Mapping) or set(variants) != {variant}:
        raise FutureValueTierListError(f"{variant} model variant set changed")
    result = variants[variant]
    if not isinstance(result, Mapping) or result.get("status") != "development_evaluated":
        raise FutureValueTierListError(f"{variant} model is not evaluated")
    if result.get("variant") != variant:
        raise FutureValueTierListError(f"{variant} result identity changed")
    _verify_authority(result.get("authority"), f"{variant} result")
    result_source = result.get("source")
    if not isinstance(result_source, Mapping):
        raise FutureValueTierListError(f"{variant} result source is missing")
    for field, expected_value in {
        **expected_source_fields,
        "source_receipt_file_sha256": source["source_receipt_file_sha256"],
    }.items():
        if result_source.get(field) != expected_value:
            raise FutureValueTierListError(f"{variant} result source changed: {field}")
    ledger = result.get("prediction_ledger")
    if not isinstance(ledger, Mapping) or ledger.get("schema_version") not in {
        "scryglass:future-value-prediction-ledger:v1",
        "scryglass:future-value-prediction-ledger:v2",
    }:
        raise FutureValueTierListError(f"{variant} prediction ledger is missing")
    rows = ledger.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise FutureValueTierListError(f"{variant} prediction rows are invalid")
    if ledger.get("row_count") != len(rows):
        raise FutureValueTierListError(f"{variant} prediction row count changed")
    if sha256_bytes(canonical_json_bytes(rows)) != _require_hash(
        ledger.get("sha256"), f"{variant} prediction ledger hash"
    ):
        raise FutureValueTierListError(f"{variant} prediction ledger values changed")
    offsets: dict[str, float] = {}
    targets: dict[str, float] = {}
    ordered_ids: list[str] = []
    for row in rows:
        game_id = str(row.get("game_id") or "").strip()
        if not game_id or game_id in offsets:
            raise FutureValueTierListError(f"{variant} prediction identity is invalid")
        try:
            fold = int(row.get("fold"))
            target = float(row.get("target"))
        except (TypeError, ValueError) as error:
            raise FutureValueTierListError(f"{variant} prediction row is invalid") from error
        if fold not in (1, 2, 3) or target not in (0.0, 1.0):
            raise FutureValueTierListError(f"{variant} fold or target is invalid")
        probability = _finite_probability(row.get("candidate"), f"{variant} {game_id}")
        offsets[game_id] = math.log(probability / (1.0 - probability))
        targets[game_id] = target
        ordered_ids.append(game_id)
    if identity_sha256(ordered_ids) != _require_hash(
        ledger.get("game_identity_sha256"), f"{variant} game identity"
    ):
        raise FutureValueTierListError(f"{variant} prediction game identity changed")
    return offsets, targets, {
        "variant": variant,
        "artifact_locator": str(path),
        "artifact_raw_sha256": expected,
        "prediction_ledger_sha256": str(ledger["sha256"]),
        "variant_receipt_sha256": str(result.get("variant_receipt", {}).get("receipt_sha256") or ""),
        "blockers": sorted(str(value) for value in result.get("blockers", [])),
    }


def validate_common_prediction_universe(
    offsets: Mapping[str, Mapping[str, float]],
    targets: Mapping[str, Mapping[str, float]],
    *,
    accepted_game_ids: Sequence[str],
    maps_path: Path,
    expected_maps_sha256: str,
) -> tuple[list[str], dict[str, Any]]:
    """Require the same verified map and target universe for all variants."""

    if set(offsets) != set(VARIANTS) or set(targets) != set(VARIANTS):
        raise FutureValueTierListError("four-way prediction set is incomplete")
    game_sets = {variant: set(offsets[variant]) for variant in VARIANTS}
    reference = game_sets[REFERENCE_VARIANT]
    if not reference or any(game_sets[variant] != reference for variant in VARIANTS):
        raise FutureValueTierListError("four-way prediction game sets differ")
    if any(set(targets[variant]) != reference for variant in VARIANTS):
        raise FutureValueTierListError("four-way target game sets differ")
    accepted = set(canonical_game_ids(accepted_game_ids))
    if not reference.issubset(accepted):
        raise FutureValueTierListError("prediction universe is outside the accepted census")
    expected_maps = _require_hash(expected_maps_sha256, "maps source hash")
    if not maps_path.is_file() or maps_path.is_symlink() or sha256_path(maps_path) != expected_maps:
        raise FutureValueTierListError("maps source bytes changed")
    frame = pd.read_parquet(maps_path, columns=["game_uid", "y_blue_win"])
    frame["game_uid"] = frame["game_uid"].astype(str)
    scoped = frame[frame["game_uid"].isin(reference)].copy()
    if scoped["game_uid"].duplicated().any() or set(scoped["game_uid"]) != reference:
        raise FutureValueTierListError("maps source identities are incomplete or duplicate")
    actual = scoped.set_index("game_uid")["y_blue_win"].astype(float).to_dict()
    for game_id in reference:
        values = {float(targets[variant][game_id]) for variant in VARIANTS}
        if len(values) != 1 or values != {float(actual[game_id])}:
            raise FutureValueTierListError(f"four-way target changed for {game_id}")
    ordered = sorted(reference)
    return ordered, {
        "game_count": len(ordered),
        "game_identity_sha256": identity_sha256(ordered),
        "maps_source_sha256": expected_maps,
        "target_rows_sha256": sha256_bytes(
            canonical_json_bytes(
                [{"game_id": game_id, "target": int(actual[game_id])} for game_id in ordered]
            )
        ),
    }


def make_offset_provenance(
    *,
    variant: str,
    offsets: Mapping[str, float],
    source_receipt_sha256: str,
) -> dict[str, Any]:
    """Seal the exact selected-map offset vector for the pooled builder."""

    ordered_ids = sorted(offsets)
    rows = [{"game_id": game_id, "logit": float(offsets[game_id])} for game_id in ordered_ids]
    payload: dict[str, Any] = {
        "schema_version": PRE_MAP_OFFSET_PROVENANCE_SCHEMA,
        "status": "research_only",
        "authority": False,
        "producer": f"future_value_rating:{variant}",
        "timing": "strict_prior_pre_map",
        "source_receipt_sha256": _require_hash(source_receipt_sha256, "source receipt hash"),
        "source_identity_sha256": identity_sha256(ordered_ids),
        "source_game_count": len(ordered_ids),
        "offsets_sha256": sha256_bytes(canonical_json_bytes(rows)),
    }
    payload["receipt_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def _candidate_rows(candidate: Mapping[str, Any]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    output: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for cell in candidate.get("cells", []):
        if not isinstance(cell, Mapping):
            continue
        scope_id = str(cell.get("scope_id") or "")
        role = str(cell.get("role") or "")
        patches = cell.get("patches")
        if not scope_id or not role or not isinstance(patches, list) or len(patches) != 1:
            raise FutureValueTierListError("candidate cell identity is invalid")
        patch = str(patches[0])
        for row in cell.get("rows", []):
            if not isinstance(row, Mapping):
                raise FutureValueTierListError("candidate Tier row is invalid")
            champion_id = str(row.get("champion_id") or "")
            if not champion_id:
                raise FutureValueTierListError("candidate champion identity is missing")
            key = (scope_id, patch, role, champion_id)
            if key in output:
                raise FutureValueTierListError("candidate Tier identity is duplicate")
            output[key] = {
                "champion": str(row.get("champion") or champion_id),
                **{field: row.get(field) for field in ROW_VALUE_FIELDS},
            }
    if not output:
        raise FutureValueTierListError("candidate has no Tier rows")
    return output


def _numeric_delta(candidate: object, reference: object) -> float | None:
    if candidate is None or reference is None:
        return None
    try:
        value = float(candidate) - float(reference)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _rank_slice(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "changed_rank_count": 0,
            "changed_tier_count": 0,
            "mean_absolute_rank_movement": None,
            "maximum_absolute_rank_movement": None,
        }
    movements = [int(row["delta"]["rank_delta"]) for row in rows]
    return {
        "row_count": len(rows),
        "changed_rank_count": sum(value != 0 for value in movements),
        "changed_tier_count": sum(bool(row["delta"]["tier_changed"]) for row in rows),
        "mean_absolute_rank_movement": float(np.mean(np.abs(movements))),
        "maximum_absolute_rank_movement": int(max(map(abs, movements))),
    }


def _cell_rank_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = row["key"]
        grouped.setdefault((key["scope_id"], key["patch"], key["role"]), []).append(row)
    weighted_correlation = 0.0
    weighted_rows = 0
    inversions = 0
    comparable_pairs = 0
    overlap_totals = {5: 0.0, 10: 0.0, 25: 0.0}
    overlap_cells = {5: 0, 10: 0, 25: 0}
    for cell_rows in grouped.values():
        reference_ranks = np.asarray([float(row["reference"]["rank"]) for row in cell_rows])
        selected_ranks = np.asarray([float(row["selected"]["rank"]) for row in cell_rows])
        if len(cell_rows) > 1:
            correlation = float(np.corrcoef(reference_ranks, selected_ranks)[0, 1])
            weighted_correlation += correlation * len(cell_rows)
            weighted_rows += len(cell_rows)
            for left in range(len(cell_rows)):
                for right in range(left + 1, len(cell_rows)):
                    comparable_pairs += 1
                    if (
                        (reference_ranks[left] - reference_ranks[right])
                        * (selected_ranks[left] - selected_ranks[right])
                        < 0.0
                    ):
                        inversions += 1
        for requested in overlap_totals:
            size = min(requested, len(cell_rows))
            if size == 0:
                continue
            reference_top = {
                row["key"]["champion_id"]
                for row in sorted(cell_rows, key=lambda item: int(item["reference"]["rank"]))[:size]
            }
            selected_top = {
                row["key"]["champion_id"]
                for row in sorted(cell_rows, key=lambda item: int(item["selected"]["rank"]))[:size]
            }
            overlap_totals[requested] += len(reference_top & selected_top) / size
            overlap_cells[requested] += 1
    return {
        "cell_count": len(grouped),
        "weighted_cell_rank_correlation": (
            weighted_correlation / weighted_rows if weighted_rows else None
        ),
        "pairwise_inversions": inversions,
        "comparable_rank_pairs": comparable_pairs,
        "pairwise_inversion_rate": inversions / comparable_pairs if comparable_pairs else None,
        "mean_cell_top_overlap": {
            str(size): overlap_totals[size] / overlap_cells[size]
            if overlap_cells[size]
            else None
            for size in overlap_totals
        },
    }


def build_fourway_diff(
    candidates: Mapping[str, Mapping[str, Any]],
    *,
    source: Mapping[str, Any],
    universe: Mapping[str, Any],
    model_bindings: Mapping[str, Mapping[str, Any]],
    trust_manifest_raw_sha256: str,
    baseline_candidate_raw_sha256: str,
) -> dict[str, Any]:
    """Build exact common-universe Tier rank and value differences."""

    if set(candidates) != set(VARIANTS):
        raise FutureValueTierListError("four Tier candidates are required")
    rows_by_variant = {variant: _candidate_rows(candidates[variant]) for variant in VARIANTS}
    reference_keys = set(rows_by_variant[REFERENCE_VARIANT])
    if any(set(rows_by_variant[variant]) != reference_keys for variant in VARIANTS):
        raise FutureValueTierListError("four-way Tier row universes differ")
    ordered_keys = sorted(reference_keys)
    common_identity = sha256_bytes(
        canonical_json_bytes(
            [
                {"scope_id": key[0], "patch": key[1], "role": key[2], "champion_id": key[3]}
                for key in ordered_keys
            ]
        )
    )
    comparisons: dict[str, Any] = {}
    all_rows: dict[str, list[dict[str, Any]]] = {}
    inherited_blockers = {
        blocker
        for binding in model_bindings.values()
        for blocker in binding.get("blockers", [])
    }
    for variant in VARIANTS:
        paired_rows: list[dict[str, Any]] = []
        rank_moves: list[int] = []
        transitions: Counter[str] = Counter()
        changed_cells: set[tuple[str, str, str]] = set()
        for key in ordered_keys:
            reference = rows_by_variant[REFERENCE_VARIANT][key]
            selected = rows_by_variant[variant][key]
            try:
                rank_delta = int(reference["rank"]) - int(selected["rank"])
            except (TypeError, ValueError) as error:
                raise FutureValueTierListError("Tier rank is missing or invalid") from error
            tier_changed = reference["tier_bucket"] != selected["tier_bucket"]
            if rank_delta or tier_changed:
                changed_cells.add(key[:3])
            rank_moves.append(rank_delta)
            transitions[f"{reference['tier_bucket']} -> {selected['tier_bucket']}"] += 1
            delta = {
                field: _numeric_delta(selected[field], reference[field])
                for field in NUMERIC_DELTA_FIELDS
            }
            delta.update({"rank_delta": rank_delta, "tier_changed": tier_changed})
            paired_rows.append(
                {
                    "key": {
                        "scope_id": key[0],
                        "patch": key[1],
                        "role": key[2],
                        "champion_id": key[3],
                        "champion": selected["champion"],
                    },
                    "reference": reference,
                    "selected": selected,
                    "delta": delta,
                }
            )
        rank_reference = np.asarray(
            [float(rows_by_variant[REFERENCE_VARIANT][key]["rank"]) for key in ordered_keys]
        )
        rank_selected = np.asarray([float(rows_by_variant[variant][key]["rank"]) for key in ordered_keys])
        rank_correlation = float(np.corrcoef(rank_reference, rank_selected)[0, 1])
        top_movers = sorted(
            [
                row
                for row in paired_rows
                if int(row["delta"]["rank_delta"]) != 0
                or bool(row["delta"]["tier_changed"])
            ],
            key=lambda row: (
                -abs(int(row["delta"]["rank_delta"])),
                -int(row["delta"]["rank_delta"]),
                row["key"]["scope_id"],
                row["key"]["role"],
                row["key"]["champion_id"],
            ),
        )[:25]
        row_digest = sha256_bytes(canonical_json_bytes(paired_rows))
        latest_patch = max(key[1] for key in ordered_keys)
        support_slices = {
            str(minimum): _rank_slice(
                [
                    row
                    for row in paired_rows
                    if int(row["reference"]["played_maps"] or 0) >= minimum
                ]
            )
            for minimum in (1, 3, 5, 10, 20)
        }
        latest_rows = [row for row in paired_rows if row["key"]["patch"] == latest_patch]
        numeric_summaries: dict[str, Any] = {}
        for field in NUMERIC_DELTA_FIELDS:
            values = [
                float(row["delta"][field])
                for row in paired_rows
                if row["delta"][field] is not None
            ]
            numeric_summaries[field] = {
                "finite_count": len(values),
                "changed_count": sum(value != 0.0 for value in values),
                "mean_delta": float(np.mean(values)) if values else None,
                "mean_absolute_delta": float(np.mean(np.abs(values))) if values else None,
                "maximum_absolute_delta": float(max(map(abs, values))) if values else None,
            }
        comparisons[variant] = {
            "reference_variant": REFERENCE_VARIANT,
            "row_count": len(paired_rows),
            "changed_rank_count": sum(move != 0 for move in rank_moves),
            "changed_tier_count": sum(row["delta"]["tier_changed"] for row in paired_rows),
            "changed_cell_count": len(changed_cells),
            "mean_absolute_rank_movement": float(np.mean(np.abs(rank_moves))),
            "maximum_absolute_rank_movement": int(max(map(abs, rank_moves), default=0)),
            "rank_correlation": rank_correlation,
            "cell_rank_metrics": _cell_rank_metrics(paired_rows),
            "latest_patch": latest_patch,
            "latest_patch_metrics": _rank_slice(latest_rows),
            "support_slices_by_minimum_played_maps": support_slices,
            "numeric_delta_summaries": numeric_summaries,
            "tier_transition_matrix": dict(sorted(transitions.items())),
            "paired_row_digest_sha256": row_digest,
            "top_movers": top_movers,
        }
        all_rows[variant] = paired_rows
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "research_only",
        "authority": dict(AUTHORITY),
        "source": dict(source),
        "trust_manifest_raw_sha256": _require_hash(
            trust_manifest_raw_sha256, "trust manifest raw hash"
        ),
        "baseline_public_candidate": {
            "status": "unchanged_non_comparable_full_census_reference",
            "raw_sha256": _require_hash(
                baseline_candidate_raw_sha256, "baseline candidate hash"
            ),
        },
        "evaluation_universe": {
            **dict(universe),
            "rank_universe": "common_verified_finite_ids_per_patch_role_cell",
            "row_identity_fields": ["scope_id", "patch", "role", "champion_id"],
            "common_row_count": len(ordered_keys),
            "common_identity_sha256": common_identity,
        },
        "model_bindings": {variant: dict(model_bindings[variant]) for variant in VARIANTS},
        "candidate_artifacts": {
            variant: {
                "artifact_sha256": candidates[variant].get("artifact_sha256"),
                "source_identity_sha256": candidates[variant].get("source", {}).get(
                    "source_identity_sha256"
                ),
                "pre_map_offset_override": candidates[variant].get("pre_map_offset_override"),
            }
            for variant in VARIANTS
        },
        "comparisons": comparisons,
        "rows": all_rows,
        "blockers": sorted(
            inherited_blockers
            | {
                "authoritative_series_id_missing_proxy_cluster_used",
                "tierlist_shadow_common_validation_subset_not_public_candidate",
                "tierlist_offset_model_includes_roster_and_composition_context_new_estimand",
                "public_tierlist_authority_false",
            }
        ),
    }
    report["report_sha256"] = sha256_bytes(canonical_json_bytes(report))
    return report


__all__ = [
    "AUTHORITY",
    "FutureValueTierListError",
    "REFERENCE_VARIANT",
    "PINNED_TRUST_MANIFEST_RAW_SHA256",
    "SCHEMA_VERSION",
    "TRUST_SCHEMA_VERSION",
    "VARIANTS",
    "build_fourway_diff",
    "canonical_json_bytes",
    "load_prediction_offsets",
    "load_trust_manifest",
    "make_offset_provenance",
    "sha256_bytes",
    "sha256_path",
    "validate_common_prediction_universe",
]
