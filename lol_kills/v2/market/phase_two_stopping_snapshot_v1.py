"""Compute the frozen phase-two metadata stopping rule without outcomes."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Callable, Mapping, Sequence

from . import betano_br_quote_adapter_v2 as quote_v2
from . import match_winner_future_protocol_v1 as protocol_source
from . import phase_one_evaluation_v1 as evaluation
from . import phase_two_attempt_completion_v1 as completion
from .match_winner_future_protocol_registry_v1 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256,
    validate_registered_match_winner_future_protocol_v1,
)


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/market/phase_two_stopping_snapshot_v1.py"
SCHEMA_VERSION = "scryglass:phase-two-metadata-stopping-snapshot:v1"
OUTPUT_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/match-winner-market-v1/phase-two/stopping-snapshots-v1"
)
SHADOW_EDGE_MINIMUM = 0.02
AUTHORITY = {
    "phase_two_outcome_opening_authority": False,
    "probability_authority": False,
    "quote_identity_authority": False,
    "expected_value_authority": False,
    "recommendation_authority": False,
    "transaction_authority": False,
    "betting_authority": False,
}
CLAIM_CEILING = (
    "Outcome-free metadata stopping snapshot for the frozen phase-two cohort. "
    "Even a support-met snapshot requires independent pinning and separate "
    "one-time outcome-opening authority; it grants no model, EV, recommendation, "
    "transaction, stake, or betting authority."
)


class PhaseTwoStoppingSnapshotError(RuntimeError):
    """A completion, quote, probability, support rule, or snapshot drifted."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PhaseTwoStoppingSnapshotError("snapshot is not canonical") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise PhaseTwoStoppingSnapshotError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise PhaseTwoStoppingSnapshotError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _clock(clock: Callable[[], datetime]) -> datetime:
    observed = clock()
    if not isinstance(observed, datetime) or observed.tzinfo is None:
        raise PhaseTwoStoppingSnapshotError("snapshot clock must be timezone-aware")
    return observed.astimezone(timezone.utc)


def _source_locks(root: Path) -> list[dict[str, Any]]:
    return [
        evaluation._source_record(root, locator)
        for locator in (
            SOURCE_LOCATOR, completion.SOURCE_LOCATOR, quote_v2.SOURCE_LOCATOR,
            protocol_source.SOURCE_LOCATOR,
        )
    ]


def _completion(
    *, root: Path, locator_value: str, environment: Mapping[str, str]
) -> tuple[str, bytes, dict[str, Any]]:
    locator = evaluation._locator(
        locator_value, completion.event_plan.COMPLETION_PREFIX,
        "completion_locator",
    )
    raw = evaluation._read_regular(root, locator, "phase-two attempt completion")
    try:
        checked = completion.validate_phase_two_attempt_completion_v1(
            evaluation._strict_object(raw, "phase-two attempt completion"),
            root=root,
            environment=environment,
        )
    except Exception as exc:
        raise PhaseTwoStoppingSnapshotError("attempt completion is invalid") from exc
    return locator, raw, checked


def _shadow_signal(
    *, root: Path, item: Mapping[str, Any], environment: Mapping[str, str]
) -> dict[str, Any] | None:
    if item["status"] != "QUALIFIED_QUOTE":
        return None
    quote_binding = item["quote_binding"]
    raw = evaluation._read_regular(root, quote_binding["locator"], "Betano v2 quote")
    quote = quote_v2.validate_betano_map_winner_quote_v2(
        evaluation._strict_object(raw, "Betano v2 quote"),
        root=root,
        environment=environment,
    )
    _, _, probability = quote_v2._probability(
        root=root,
        locator_value=quote["event_probability_v2_binding"]["locator"],
        environment=environment,
    )
    event = probability["event"]
    interval = probability["probability_interval"]
    prices = quote["frozen_v1_transport_quote"]["generic_quote_receipt"]["prices"]
    blue_key = event["selection"]
    red_key = event["opposing_selection"]
    blue_lower = float(interval[0])
    red_lower = 1.0 - float(interval[1])
    blue_edge = blue_lower * float(prices[blue_key]) - 1.0
    red_edge = red_lower * float(prices[red_key]) - 1.0
    blue_qualifies = blue_edge >= SHADOW_EDGE_MINIMUM
    red_qualifies = red_edge >= SHADOW_EDGE_MINIMUM
    exactly_one = blue_qualifies != red_qualifies
    return {
        "blue_selection": blue_key,
        "red_selection": red_key,
        "blue_lower_probability_bound": blue_lower,
        "red_lower_probability_bound": red_lower,
        "blue_decimal_odds": float(prices[blue_key]),
        "red_decimal_odds": float(prices[red_key]),
        "blue_lower_bound_edge": blue_edge,
        "red_lower_bound_edge": red_edge,
        "edge_minimum": SHADOW_EDGE_MINIMUM,
        "exactly_one_side_qualifies": exactly_one,
        "selected_side": blue_key if blue_qualifies and not red_qualifies else red_key if red_qualifies and not blue_qualifies else None,
        "both_sides_qualify_inconsistent": blue_qualifies and red_qualifies,
        "prediction_to_response_seconds": float(
            quote["frozen_v1_transport_quote"]["prediction_binding"][
                "prediction_to_response_seconds"
            ]
        ),
    }


def _quote_chronology(
    *, root: Path, item: Mapping[str, Any], environment: Mapping[str, str]
) -> dict[str, float | None]:
    if item["quote_binding"] is None:
        return {
            "response_to_actual_start_seconds": None,
            "prediction_to_response_seconds": None,
        }
    raw = evaluation._read_regular(
        root, item["quote_binding"]["locator"], "Betano v2 quote"
    )
    quote = quote_v2.validate_betano_map_winner_quote_v2(
        evaluation._strict_object(raw, "Betano v2 quote"),
        root=root,
        environment=environment,
    )
    return {
        "response_to_actual_start_seconds": float(
            item["response_to_actual_start_seconds"]
        ),
        "prediction_to_response_seconds": float(
            quote["frozen_v1_transport_quote"]["prediction_binding"]
            ["prediction_to_response_seconds"]
        ),
    }


def _entries(
    *, completion_locators: Sequence[str], root: Path,
    environment: Mapping[str, str]
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    identities: set[tuple[str, int]] = set()
    for locator_value in completion_locators:
        locator, raw, item = _completion(
            root=root, locator_value=locator_value, environment=environment
        )
        event = item["event"]
        identity = (str(event["event_id"]), int(event["game_number"]))
        if identity in identities:
            raise PhaseTwoStoppingSnapshotError("completion identity repeats")
        identities.add(identity)
        _start_locator, _start_raw, map_start = completion.qualification._map_start(
            root=root,
            locator_value=item["map_start_binding"]["locator"],
        )
        chronology = _quote_chronology(
            root=root, item=item, environment=environment
        )
        entries.append(
            {
                **event,
                "completion_locator": locator,
                "completion_raw_sha256": _sha256(raw),
                "completion_artifact_sha256": item["artifact_sha256"],
                "completion_status": item["status"],
                "failure_code": item["failure_code"],
                "qualified_quote": item["status"] == "QUALIFIED_QUOTE",
                "quote_response_too_late": item["status"]
                == "QUOTE_RESPONSE_TOO_LATE",
                "actual_map_start_utc": map_start["event"][
                    "actual_map_start_utc"
                ],
                **chronology,
                "shadow_signal": _shadow_signal(
                    root=root, item=item, environment=environment
                ),
            }
        )
    entries.sort(key=lambda row: (row["series_id"], row["game_number"], row["event_id"]))
    return entries


def _patch_key(value: str) -> tuple[int, int]:
    major, minor = value.split(".", 1)
    return int(major), int(minor)


def _support(entries: Sequence[Mapping[str, Any]], rule: Mapping[str, Any]) -> dict[str, Any]:
    quoted = [row for row in entries if row["qualified_quote"]]
    patches = sorted({str(row["patch"]) for row in quoted}, key=_patch_key)
    latest_patch = patches[-1] if patches else None
    domestic = {
        league: sum(row["league"] == league for row in quoted)
        for league in rule["domestic_leagues"]
    }
    shadow = sum(
        (row.get("shadow_signal") or {}).get("exactly_one_side_qualifies") is True
        for row in quoted
    )
    inconsistent = sum(
        (row.get("shadow_signal") or {}).get("both_sides_qualify_inconsistent") is True
        for row in quoted
    )
    coverage = len(quoted) / len(entries) if entries else 0.0
    support = {
        "otherwise_eligible_maps": len(entries),
        "eligible_quoted_maps": len(quoted),
        "eligible_series": len({row["series_id"] for row in quoted}),
        "quoted_maps_by_domestic_league": domestic,
        "international_quoted_maps": sum(row["league"] in {"MSI", "EWC"} for row in quoted),
        "distinct_future_patches": len(patches),
        "latest_future_patch": latest_patch,
        "latest_patch_quoted_maps": sum(row["patch"] == latest_patch for row in quoted) if latest_patch else 0,
        "one_or_both_rosters_changed_maps": sum(row["roster_change_stratum"] != "UNCHANGED" for row in quoted),
        "sparse_or_new_player_or_champion_maps": sum(bool(row["sparse_or_new_champion_map"]) for row in quoted),
        "quote_coverage": coverage,
        "shadow_policy_qualifying_maps": shadow,
        "both_sides_shadow_qualify_inconsistent_maps": inconsistent,
        "quote_response_too_late_maps": sum(row["quote_response_too_late"] for row in entries),
        "quote_received_after_map_start_maps": sum(
            row.get("response_to_actual_start_seconds") is not None
            and row["response_to_actual_start_seconds"] < 0.0
            for row in entries
        ),
        "failure_codes": {
            code: sum(row["failure_code"] == code for row in entries)
            for code in sorted({str(row["failure_code"]) for row in entries if row["failure_code"] is not None})
        },
    }
    support_met = (
        support["eligible_quoted_maps"] >= rule["eligible_quoted_maps_minimum"]
        and support["eligible_series"] >= rule["eligible_series_minimum"]
        and all(domestic[league] >= rule["each_domestic_league_quoted_maps_minimum"] for league in rule["domestic_leagues"])
        and support["international_quoted_maps"] >= rule["international_quoted_maps_minimum"]
        and support["distinct_future_patches"] >= rule["distinct_future_patches_minimum"]
        and support["latest_patch_quoted_maps"] >= rule["latest_patch_quoted_maps_minimum"]
        and support["one_or_both_rosters_changed_maps"] >= rule["one_or_both_rosters_changed_maps_minimum"]
        and support["sparse_or_new_player_or_champion_maps"] >= rule["sparse_or_new_player_or_champion_maps_minimum"]
        and coverage >= rule["quote_coverage_of_otherwise_eligible_maps_minimum"]
        and shadow >= rule["shadow_policy_qualifying_maps_minimum"]
    )
    support["support_met"] = support_met
    support["terminal_shadow_support_failure"] = (
        not support_met
        and len(quoted) >= rule["eligible_quoted_maps_maximum_if_shadow_support_not_met"]
        and shadow < rule["shadow_policy_qualifying_maps_minimum"]
    )
    return support


def build_phase_two_stopping_snapshot_v1(
    *, completion_locators: Sequence[str], root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    protocol = validate_registered_match_winner_future_protocol_v1(root=root)
    rule = protocol["phase_two"]["metadata_only_stopping_rule"]
    rows = _entries(
        completion_locators=completion_locators, root=root, environment=environment
    )
    if not rows:
        raise PhaseTwoStoppingSnapshotError("stopping snapshot cannot be empty")
    captured = _clock(clock)
    support = _support(rows, rule)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": "PHASE_TWO_METADATA_SUPPORT_MET_OUTCOMES_UNOPENED" if support["support_met"] else "PHASE_TWO_METADATA_SUPPORT_UNMET_OUTCOMES_UNOPENED",
        "captured_at_utc": captured.isoformat(),
        "protocol_binding": {
            "artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
            "metadata_only_stopping_rule": rule,
        },
        "entries": rows,
        "entries_sha256": _canonical_sha256(rows),
        "support": support,
        "outcome_boundary": {
            "outcomes_present": False,
            "outcomes_accessed": False,
            "manual_post_outcome_exclusion_permitted": False,
            "independent_snapshot_registration_required": True,
            "separate_one_time_outcome_opening_required": True,
        },
        "source_locks": _source_locks(root),
        "authority": dict(AUTHORITY),
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_phase_two_stopping_snapshot_v1(
        payload, root=root, environment=environment
    )


def validate_phase_two_stopping_snapshot_v1(
    payload: Mapping[str, Any], *, root: Path = ROOT,
    environment: Mapping[str, str] = os.environ
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PhaseTwoStoppingSnapshotError("snapshot must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version", "result_state", "captured_at_utc", "protocol_binding",
        "entries", "entries_sha256", "support", "outcome_boundary",
        "source_locks", "authority", "claim_ceiling", "artifact_sha256",
    }:
        raise PhaseTwoStoppingSnapshotError("snapshot structure changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise PhaseTwoStoppingSnapshotError("snapshot hash changed")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise PhaseTwoStoppingSnapshotError("snapshot schema changed")
    _timestamp(value.get("captured_at_utc"), "captured_at_utc")
    protocol = validate_registered_match_winner_future_protocol_v1(root=root)
    rule = protocol["phase_two"]["metadata_only_stopping_rule"]
    if value.get("protocol_binding") != {
        "artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
        "metadata_only_stopping_rule": rule,
    }:
        raise PhaseTwoStoppingSnapshotError("snapshot protocol binding changed")
    stored_entries = value.get("entries")
    if not isinstance(stored_entries, list) or not stored_entries:
        raise PhaseTwoStoppingSnapshotError("snapshot entries are empty")
    locators = [row.get("completion_locator") for row in stored_entries if isinstance(row, Mapping)]
    if len(locators) != len(stored_entries) or any(not isinstance(item, str) for item in locators):
        raise PhaseTwoStoppingSnapshotError("snapshot completion inventory changed")
    expected_entries = _entries(
        completion_locators=locators, root=root, environment=environment
    )
    expected_support = _support(expected_entries, rule)
    if stored_entries != expected_entries or value.get("entries_sha256") != _canonical_sha256(expected_entries):
        raise PhaseTwoStoppingSnapshotError("snapshot entries changed")
    if value.get("support") != expected_support:
        raise PhaseTwoStoppingSnapshotError("snapshot support changed")
    expected_state = "PHASE_TWO_METADATA_SUPPORT_MET_OUTCOMES_UNOPENED" if expected_support["support_met"] else "PHASE_TWO_METADATA_SUPPORT_UNMET_OUTCOMES_UNOPENED"
    if value.get("result_state") != expected_state:
        raise PhaseTwoStoppingSnapshotError("snapshot result state changed")
    if value.get("outcome_boundary") != {
        "outcomes_present": False,
        "outcomes_accessed": False,
        "manual_post_outcome_exclusion_permitted": False,
        "independent_snapshot_registration_required": True,
        "separate_one_time_outcome_opening_required": True,
    }:
        raise PhaseTwoStoppingSnapshotError("snapshot outcome boundary changed")
    if value.get("source_locks") != _source_locks(root):
        raise PhaseTwoStoppingSnapshotError("snapshot source lock changed")
    if value.get("authority") != AUTHORITY or value.get("claim_ceiling") != CLAIM_CEILING:
        raise PhaseTwoStoppingSnapshotError("snapshot exceeds authority")
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PhaseTwoStoppingSnapshotError(f"refusing to replace snapshot: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise PhaseTwoStoppingSnapshotError(f"refusing to replace snapshot: {path}") from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return _sha256(raw)


__all__ = [
    "OUTPUT_PREFIX", "SCHEMA_VERSION", "SOURCE_LOCATOR",
    "PhaseTwoStoppingSnapshotError", "build_phase_two_stopping_snapshot_v1",
    "validate_phase_two_stopping_snapshot_v1", "write_no_clobber",
]
