"""Fail-closed OE-baseline / GRID promotion gate for Draft Score.

GRID is allowed to replace Oracle's Elixir only for an explicitly defined
competition/date cohort whose every included game has a complete, hash-bound
draft record.  This module validates the gate contract; it does not fetch
GRID data, train a model, or authorize public serving.
"""

from __future__ import annotations

import hashlib
import base64
import json
import math
from datetime import datetime, timezone
from typing import Any, Mapping


SCHEMA_VERSION = "scryglass:draft-terminal-grid-promotion-gate:v1"
ROLES = ("top", "jungle", "mid", "bot", "support")
REQUIRED_HELD_OUT_METRICS = ("log_loss", "brier_score", "ece")
REQUIRED_CHECK_FIELDS = {
    "identity_checks": ("game_id_matches", "teams_are_distinct"),
    "sequence_checks": ("slots_contiguous", "picks_complete"),
    "leakage_checks": ("pre_event_inputs_only", "result_excluded_from_draft_inputs"),
}
SHA256_LENGTH = 64


class GridPromotionGateError(ValueError):
    """The GRID cohort gate input is malformed or does not pass."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_canonical(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise GridPromotionGateError(f"{field} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GridPromotionGateError(f"{field} is not RFC-3339") from exc
    if parsed.tzinfo is None:
        raise GridPromotionGateError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _hash(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != SHA256_LENGTH or any(char not in "0123456789abcdef" for char in value):
        raise GridPromotionGateError(f"{field} is not a lowercase SHA-256")
    return value


def _decode_exact_json(encoded: Any, field: str) -> tuple[bytes, dict[str, Any]]:
    if not isinstance(encoded, str) or not encoded:
        raise GridPromotionGateError(f"{field} is missing")
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise GridPromotionGateError(f"{field} is not valid base64") from exc
    if not raw or base64.b64encode(raw).decode("ascii") != encoded:
        raise GridPromotionGateError(f"{field} is not canonical base64")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise GridPromotionGateError(f"{field} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except GridPromotionGateError:
        raise
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise GridPromotionGateError(f"{field} is not strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise GridPromotionGateError(f"{field} must decode to a JSON object")
    return raw, payload


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise GridPromotionGateError(f"{field} is not finite")
    return float(value)


def _check_flags(
    record: Mapping[str, Any],
    field: str,
    required_fields: tuple[str, ...],
    record_id: str,
    blockers: list[str],
) -> None:
    prefix = f"record.{record_id}."
    value = record.get(field)
    if not isinstance(value, Mapping) or not value:
        blockers.append(f"{prefix}{field}.missing")
        return
    for name in required_fields:
        if name not in value:
            blockers.append(f"{prefix}{field}.{name}.missing")
    for name, passed in sorted(value.items()):
        if passed is not True:
            blockers.append(f"{prefix}{field}.{name}")


def _validate_record(
    record: Mapping[str, Any],
    *,
    expected_game_id: str,
    cohort: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    game_id = str(record.get("game_id") or "")
    if game_id != expected_game_id:
        blockers.append(f"record.{expected_game_id}.identity.game_id_mismatch")
    payload = record.get("payload")
    if not isinstance(payload, Mapping):
        blockers.append(f"record.{expected_game_id}.payload_missing")
        return blockers
    try:
        source_raw, source_payload = _decode_exact_json(
            record.get("source_payload_base64"),
            f"record.{expected_game_id}.source_payload_base64",
        )
        payload_hash = _hash(record.get("source_payload_sha256"), f"record.{expected_game_id}.source_payload_sha256")
        if payload_hash != hashlib.sha256(source_raw).hexdigest():
            blockers.append(f"record.{expected_game_id}.source_payload_hash_mismatch")
        if source_payload != dict(payload):
            blockers.append(f"record.{expected_game_id}.source_payload_content_mismatch")
        record_hash = _hash(record.get("record_sha256"), f"record.{expected_game_id}.record_sha256")
        unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
        if record_hash != sha256_canonical(unsigned):
            blockers.append(f"record.{expected_game_id}.record_hash_mismatch")
    except GridPromotionGateError as exc:
        blockers.append(str(exc))

    payload_game_id = str(payload.get("game_id") or "")
    if payload_game_id != expected_game_id:
        blockers.append(f"record.{expected_game_id}.payload_game_id_mismatch")
    try:
        event_start = _timestamp(payload.get("event_start"), f"record.{expected_game_id}.event_start")
        available_at = _timestamp(payload.get("source_available_at"), f"record.{expected_game_id}.source_available_at")
        retrieved_at = _timestamp(payload.get("source_retrieved_at"), f"record.{expected_game_id}.source_retrieved_at")
        cohort_start = _timestamp(cohort.get("date_start"), "cohort.date_start")
        cohort_end = _timestamp(cohort.get("date_end"), "cohort.date_end")
        if not cohort_start <= event_start < cohort_end:
            blockers.append(f"record.{expected_game_id}.outside_cohort_dates")
        if available_at >= event_start:
            blockers.append(f"record.{expected_game_id}.not_preevent")
        if retrieved_at < available_at:
            blockers.append(f"record.{expected_game_id}.retrieval_before_availability")
    except GridPromotionGateError as exc:
        blockers.append(str(exc))

    if not isinstance(payload.get("patch"), str) or not str(payload["patch"]).strip():
        blockers.append(f"record.{expected_game_id}.patch_missing")
    if payload.get("result") not in {"A", "B"}:
        blockers.append(f"record.{expected_game_id}.result_missing_or_invalid")

    side_mapping = payload.get("side_mapping")
    if not isinstance(side_mapping, Mapping) or set(side_mapping) != {"A", "B"}:
        blockers.append(f"record.{expected_game_id}.side_mapping.missing_or_invalid")
    else:
        game_sides: set[str] = set()
        draft_orders: set[str] = set()
        for canonical_side in ("A", "B"):
            mapping = side_mapping.get(canonical_side)
            if not isinstance(mapping, Mapping) or set(mapping) != {"game_side", "draft_order"}:
                blockers.append(f"record.{expected_game_id}.side_mapping.{canonical_side}.invalid")
                continue
            game_side = mapping.get("game_side")
            draft_order = mapping.get("draft_order")
            if game_side not in {"blue", "red"}:
                blockers.append(f"record.{expected_game_id}.side_mapping.{canonical_side}.game_side_invalid")
            else:
                game_sides.add(game_side)
            if draft_order not in {"first", "second"}:
                blockers.append(f"record.{expected_game_id}.side_mapping.{canonical_side}.draft_order_invalid")
            else:
                draft_orders.add(draft_order)
        if game_sides != {"blue", "red"}:
            blockers.append(f"record.{expected_game_id}.side_mapping.game_sides_not_distinct")
        if draft_orders != {"first", "second"}:
            blockers.append(f"record.{expected_game_id}.side_mapping.draft_orders_not_distinct")

    picks = payload.get("picks")
    if not isinstance(picks, list) or len(picks) != 10:
        blockers.append(f"record.{expected_game_id}.picks.incomplete")
    else:
        slots = [item.get("slot") if isinstance(item, Mapping) else None for item in picks]
        if slots != list(range(1, 11)):
            blockers.append(f"record.{expected_game_id}.picks.sequence_invalid")
        champions: set[str] = set()
        roles_by_side: dict[str, set[str]] = {"A": set(), "B": set()}
        for item in picks:
            if not isinstance(item, Mapping):
                blockers.append(f"record.{expected_game_id}.picks.item_invalid")
                continue
            side = item.get("canonical_side")
            role = item.get("role")
            champion = str(item.get("champion_id") or "")
            if item.get("kind") != "pick":
                blockers.append(f"record.{expected_game_id}.picks.kind_invalid")
            if side not in roles_by_side:
                blockers.append(f"record.{expected_game_id}.picks.side_invalid")
            elif role not in ROLES or role in roles_by_side[side]:
                blockers.append(f"record.{expected_game_id}.picks.role_invalid")
            else:
                roles_by_side[side].add(role)
            if not champion or champion in champions:
                blockers.append(f"record.{expected_game_id}.picks.champion_identity_invalid")
            champions.add(champion)
        if any(roles_by_side[side] != set(ROLES) for side in ("A", "B")):
            blockers.append(f"record.{expected_game_id}.picks.roles_incomplete")
        if len(champions) != 10:
            blockers.append(f"record.{expected_game_id}.picks.champion_count_invalid")

    for field, required_fields in REQUIRED_CHECK_FIELDS.items():
        _check_flags(record, field, required_fields, expected_game_id, blockers)
    return blockers


def _validate_model(manifest: Mapping[str, Any], blockers: list[str]) -> str | None:
    model = manifest.get("model")
    if not isinstance(model, Mapping):
        blockers.append("model.missing")
        return None
    if not isinstance(model.get("model_id"), str) or not model["model_id"].strip():
        blockers.append("model.model_id_missing")
    payload = model.get("payload")
    if not isinstance(payload, Mapping):
        blockers.append("model.payload_missing")
        return None
    try:
        model_raw, model_payload = _decode_exact_json(model.get("payload_base64"), "model.payload_base64")
        model_hash = _hash(model.get("model_sha256"), "model.model_sha256")
        if model_hash != hashlib.sha256(model_raw).hexdigest():
            blockers.append("model.hash_mismatch")
        if model_payload != dict(payload):
            blockers.append("model.payload_content_mismatch")
        return model_hash
    except GridPromotionGateError as exc:
        blockers.append(str(exc))
        return None


def _held_out_check(manifest: Mapping[str, Any], name: str, blockers: list[str]) -> dict[str, Any]:
    initial_blocker_count = len(blockers)
    section = manifest.get("held_out", {}).get(name) if isinstance(manifest.get("held_out"), Mapping) else None
    if not isinstance(section, Mapping):
        blockers.append(f"held_out.{name}.missing")
        return {"status": "missing"}
    oe = section.get("oe")
    grid = section.get("grid")
    tolerances = section.get("max_allowed_delta")
    plan = section.get("predeclared_plan")
    if not isinstance(plan, Mapping):
        blockers.append(f"held_out.{name}.predeclared_plan_missing")
    else:
        try:
            plan_hash = _hash(section.get("predeclared_plan_sha256"), f"held_out.{name}.predeclared_plan_sha256")
            if plan_hash != sha256_canonical(plan):
                blockers.append(f"held_out.{name}.predeclared_plan_hash_mismatch")
        except GridPromotionGateError as exc:
            blockers.append(str(exc))
        if plan.get("cohort_id") != name:
            blockers.append(f"held_out.{name}.predeclared_plan_cohort_mismatch")
        if plan.get("baseline_source") != "OE" or plan.get("candidate_source") != "GRID":
            blockers.append(f"held_out.{name}.predeclared_plan_source_mismatch")
        if plan.get("metrics") != list(REQUIRED_HELD_OUT_METRICS):
            blockers.append(f"held_out.{name}.predeclared_plan_metrics_mismatch")
        if isinstance(tolerances, Mapping) and plan.get("max_allowed_delta") != dict(tolerances):
            blockers.append(f"held_out.{name}.predeclared_plan_tolerance_mismatch")
    try:
        predeclared_at = _timestamp(section.get("predeclared_at"), f"held_out.{name}.predeclared_at")
        cohort_start = _timestamp(
            manifest.get("cohort", {}).get("date_start") if isinstance(manifest.get("cohort"), Mapping) else None,
            "cohort.date_start",
        )
        results_recorded_at = _timestamp(section.get("results_recorded_at"), f"held_out.{name}.results_recorded_at")
        if predeclared_at >= cohort_start:
            blockers.append(f"held_out.{name}.plan_not_predeclared_before_cohort")
        if results_recorded_at <= predeclared_at:
            blockers.append(f"held_out.{name}.results_recorded_not_after_plan")
    except GridPromotionGateError as exc:
        blockers.append(str(exc))
    if not isinstance(oe, Mapping) or not isinstance(grid, Mapping) or not isinstance(tolerances, Mapping):
        blockers.append(f"held_out.{name}.metrics_missing")
        return {"status": "missing"}
    try:
        result_hash = _hash(section.get("results_sha256"), f"held_out.{name}.results_sha256")
        result_payload = {"oe": dict(oe), "grid": dict(grid), "max_allowed_delta": dict(tolerances)}
        if result_hash != sha256_canonical(result_payload):
            blockers.append(f"held_out.{name}.results_hash_mismatch")
    except GridPromotionGateError as exc:
        blockers.append(str(exc))
    comparisons: dict[str, Any] = {}
    for metric in REQUIRED_HELD_OUT_METRICS:
        try:
            oe_value = _number(oe.get(metric), f"held_out.{name}.oe.{metric}")
            grid_value = _number(grid.get(metric), f"held_out.{name}.grid.{metric}")
            tolerance = _number(tolerances.get(metric), f"held_out.{name}.max_allowed_delta.{metric}")
        except GridPromotionGateError as exc:
            blockers.append(str(exc))
            continue
        if tolerance < 0:
            blockers.append(f"held_out.{name}.max_allowed_delta.{metric}.negative")
        delta = grid_value - oe_value
        passed = delta <= tolerance
        comparisons[metric] = {
            "oe": oe_value,
            "grid": grid_value,
            "delta_grid_minus_oe": delta,
            "max_allowed_delta": tolerance,
            "passed": passed,
        }
        if not passed:
            blockers.append(f"held_out.{name}.{metric}.grid_not_noninferior_to_oe")
    return {
        "status": "passed" if comparisons and all(item["passed"] for item in comparisons.values()) and len(blockers) == initial_blocker_count else "failed",
        "comparisons": comparisons,
    }


def evaluate_grid_promotion_gate(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate a complete GRID cohort and retain OE as fallback on failure."""

    blockers: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        blockers.append("manifest.schema_version_invalid")
    if manifest.get("provider") != "GRID":
        blockers.append("manifest.provider_is_not_GRID")
    cohort = manifest.get("cohort")
    if not isinstance(cohort, Mapping):
        blockers.append("cohort.missing")
        cohort = {}
    expected_ids = cohort.get("game_ids")
    if not isinstance(expected_ids, list) or not expected_ids or any(not isinstance(value, str) or not value for value in expected_ids) or len(set(expected_ids)) != len(expected_ids):
        blockers.append("cohort.game_ids_not_exact_unique")
        expected_ids = []
    records = manifest.get("records")
    if not isinstance(records, list):
        blockers.append("records.missing")
        records = []
    by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            blockers.append("records.item_invalid")
            continue
        game_id = str(record.get("game_id") or "")
        if not game_id or game_id in by_id:
            blockers.append(f"records.duplicate_or_missing_game_id:{game_id or '<empty>'}")
            continue
        by_id[game_id] = record
    missing = sorted(set(expected_ids) - set(by_id))
    extra = sorted(set(by_id) - set(expected_ids))
    blockers.extend(f"records.missing:{game_id}" for game_id in missing)
    blockers.extend(f"records.extra:{game_id}" for game_id in extra)
    record_blockers: dict[str, list[str]] = {}
    for game_id in expected_ids:
        record = by_id.get(game_id)
        if record is not None:
            issues = _validate_record(record, expected_game_id=game_id, cohort=cohort)
            if issues:
                record_blockers[game_id] = issues
                blockers.extend(issues)

    model_hash = _validate_model(manifest, blockers)
    verified_data_payload = {
        "cohort": dict(cohort),
        "records": [by_id[game_id] for game_id in expected_ids if game_id in by_id],
    }
    verified_data_hash = sha256_canonical(verified_data_payload)

    held_out = {
        name: _held_out_check(manifest, name, blockers)
        for name in ("validation", "calibration")
    }
    replay = manifest.get("second_replay")
    replay_passed = False
    if not isinstance(replay, Mapping):
        blockers.append("second_replay.missing")
    else:
        for field in ("first_data_sha256", "second_data_sha256", "first_model_sha256", "second_model_sha256"):
            try:
                _hash(replay.get(field), f"second_replay.{field}")
            except GridPromotionGateError as exc:
                blockers.append(str(exc))
        if replay.get("first_data_sha256") != replay.get("second_data_sha256"):
            blockers.append("second_replay.data_hash_changed")
        if replay.get("first_model_sha256") != replay.get("second_model_sha256"):
            blockers.append("second_replay.model_hash_changed")
        if replay.get("first_data_sha256") != verified_data_hash:
            blockers.append("second_replay.data_hash_not_bound_to_manifest")
        if model_hash is None or replay.get("first_model_sha256") != model_hash:
            blockers.append("second_replay.model_hash_not_bound_to_manifest")
        replay_passed = replay.get("status") == "identical" and not any(item.startswith("second_replay.") for item in blockers)
        if not replay_passed and not any(item.startswith("second_replay.") for item in blockers):
            blockers.append("second_replay.not_identical")

    passed = not blockers
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "blocked",
        "primary_source_for_cohort": "GRID" if passed else "OE",
        "public_reproducibility_benchmark": "OE",
        "grid_primary_for_cohort": passed,
        "oe_remains_active": not passed,
        "cohort": dict(cohort),
        "expected_game_count": len(expected_ids),
        "verified_game_count": len(expected_ids) - len(missing) - len([game_id for game_id in extra if game_id in expected_ids]),
        "missing_or_invalid_records": {
            "missing": missing,
            "extra": extra,
            "invalid": record_blockers,
        },
        "held_out": held_out,
        "second_replay": {
            "passed": replay_passed,
            "verified_data_sha256": verified_data_hash,
            "verified_model_sha256": model_hash,
        },
        "blockers": sorted(set(blockers)),
        "manifest_sha256": sha256_canonical(manifest),
    }
