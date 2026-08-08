"""Lock the terminal Draft Score incremental future evaluation protocol.

The protocol shares the ratings model's August 3 future boundary.  It validates
the Draft Score only as an incremental term added to the exact pre-event rating
forecast; the neutral output remains an equal-strength composition index.  No
future result may be inspected until metadata-only support is met and an
independent party pins both the protocol and outcome-free prediction ledger.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

from lol_kills.v2.data.common import sha256_canonical_object
from lol_kills.v2.ratings.player.multileague_v3_future_protocol import (
    DOMESTIC_LEAGUES,
    FUTURE_SEALED_START,
)
from lol_kills.v2.ratings.player.multileague_v3_registry_v3 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256 as RATINGS_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_LOCATOR as RATINGS_PROTOCOL_LOCATOR,
    REGISTERED_PROTOCOL_RAW_SHA256 as RATINGS_PROTOCOL_RAW_SHA256,
    validate_registered_future_protocol_v3,
)

from .adaptive_temporal_diagnostic import (
    DEFAULT_OUTPUT as ADAPTIVE_DIAGNOSTIC,
    validate_adaptive_temporal_diagnostic,
)
from .candidate_registry_v3 import (
    DEFAULT_OUTPUT as CANDIDATE_REGISTRY,
    validate_candidate_registry_v3,
)


SCHEMA_VERSION = "scryglass:draft-terminal-future-protocol-lock:v1"
RESULT_STATE = "INCREMENTAL_DRAFT_FUTURE_PROTOCOL_LOCKED_EMPTY"
DEFAULT_OUTPUT = Path(
    "data/lol/v2/models/draft-terminal/future-protocol-lock-v1.json"
)
L2_CONTRACT = Path(
    "data/lol/v2/models/draft-terminal/draft-terminal-l2-evaluation-contract.json"
)
SOURCE_LOCKS = (
    "lol_kills/v2/draft/terminal/future_protocol_v1.py",
    "lol_kills/v2/draft/terminal/candidate_registry_v3.py",
    "lol_kills/v2/draft/terminal/development_snapshot.py",
    "lol_kills/v2/draft/terminal/development_evaluation.py",
    "lol_kills/v2/draft/terminal/development_evaluation_v2.py",
    "lol_kills/v2/draft/terminal/development_evaluation_v3.py",
    "lol_kills/v2/draft/terminal/development_artifact_v3.py",
    "lol_kills/v2/draft/terminal/model.py",
    CANDIDATE_REGISTRY.as_posix(),
    ADAPTIVE_DIAGNOSTIC.as_posix(),
    L2_CONTRACT.as_posix(),
    RATINGS_PROTOCOL_LOCATOR.as_posix(),
)


class DraftFutureProtocolError(ValueError):
    """Raised when the locked future Draft Score protocol is inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(root: Path, locator: str) -> dict[str, Any]:
    path = root / locator
    if not path.is_file():
        raise DraftFutureProtocolError(f"future protocol source is missing: {locator}")
    return {
        "locator": locator,
        "bytes": path.stat().st_size,
        "raw_sha256": _sha256(path),
    }


def _clock(clock: Callable[[], datetime], registry_time: datetime) -> datetime:
    observed = clock()
    if not isinstance(observed, datetime) or observed.tzinfo is None:
        raise DraftFutureProtocolError("protocol clock must be timezone aware")
    observed = observed.astimezone(timezone.utc)
    if observed <= registry_time:
        raise DraftFutureProtocolError("protocol lock must follow candidate registry")
    if observed >= FUTURE_SEALED_START.replace(tzinfo=timezone.utc):
        raise DraftFutureProtocolError(
            "protocol lock must precede the future holdout boundary"
        )
    return observed


def _atomic_write_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise DraftFutureProtocolError(f"refusing to overwrite protocol: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise DraftFutureProtocolError(f"refusing to overwrite protocol: {path}")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_future_protocol_v1(
    *,
    root: Path = Path("."),
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    registry_path = root / CANDIDATE_REGISTRY
    registry_raw = registry_path.read_bytes()
    registry = validate_candidate_registry_v3(json.loads(registry_raw), root=root)
    registry_time = datetime.fromisoformat(registry["locked_at_utc"]).astimezone(
        timezone.utc
    )
    observed = _clock(clock, registry_time)
    ratings_protocol = validate_registered_future_protocol_v3(root=root)
    if (
        ratings_protocol.get("artifact_sha256")
        != RATINGS_PROTOCOL_ARTIFACT_SHA256
        or _sha256(root / RATINGS_PROTOCOL_LOCATOR) != RATINGS_PROTOCOL_RAW_SHA256
    ):
        raise DraftFutureProtocolError("registered ratings protocol binding changed")
    adaptive_path = root / ADAPTIVE_DIAGNOSTIC
    adaptive_raw = adaptive_path.read_bytes()
    adaptive = validate_adaptive_temporal_diagnostic(
        json.loads(adaptive_raw), root=root
    )
    if (
        adaptive.get("decision", {}).get("known_app_draft_family_nonharmful")
        is not False
    ):
        raise DraftFutureProtocolError("known adaptive app-model harm was omitted")
    selected = registry["selected_candidate"]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "locked_at_utc": observed.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_inside_builder",
            "observed_wall_clock_utc": observed.isoformat(),
            "user_supplied_timestamp_allowed": False,
            "lock_time_after_candidate_registry": True,
            "lock_time_before_future_boundary": True,
        },
        "candidate_registry": {
            "locator": CANDIDATE_REGISTRY.as_posix(),
            "raw_sha256": hashlib.sha256(registry_raw).hexdigest(),
            "artifact_sha256": registry["artifact_sha256"],
            "locked_at_utc": registry["locked_at_utc"],
        },
        "locked_candidate": {
            "candidate_id": selected["candidate_id"],
            "variant_id": selected["variant_id"],
            "ridge_strength": selected["ridge_strength"],
            "model_version": selected["model_version"],
            "model_as_of": selected["model_as_of"],
            "artifact_locator": selected["artifact_locator"],
            "artifact_raw_sha256": selected["artifact_raw_sha256"],
            "selection_status": "adaptive_development_choice_locked_before_future_holdout",
            "future_reselection_permitted": False,
        },
        "ratings_context": {
            "protocol_locator": RATINGS_PROTOCOL_LOCATOR.as_posix(),
            "protocol_raw_sha256": RATINGS_PROTOCOL_RAW_SHA256,
            "protocol_artifact_sha256": RATINGS_PROTOCOL_ARTIFACT_SHA256,
            "candidate_id": ratings_protocol["locked_candidate"]["candidate_id"],
            "shared_future_boundary": True,
            "ratings_probability_authority_present": False,
        },
        "estimands": {
            "primary": "incremental_predictive_value_of_frozen_rating_context_plus_frozen_draft_terms_over_the_identical_frozen_rating_context_without_draft",
            "combined_candidate_logit": "logit(frozen_rating_candidate_probability)+draft_artifact_calibration_slope_times_terminal_raw_draft_logit",
            "comparator_logit": "logit(frozen_rating_candidate_probability)",
            "neutral_output": "equal_strength_composition_index",
            "neutral_output_directly_outcome_calibrated": False,
            "causal_draft_effect_identified": False,
        },
        "future_holdout": {
            "status": "EMPTY_NOT_YET_ACQUIRED",
            "start_inclusive_source_time": FUTURE_SEALED_START.replace(
                tzinfo=timezone.utc
            ).isoformat(),
            "series_atomic": True,
            "one_time_opening": True,
            "eligibility": {
                "professional_maps_only": True,
                "leagues": [*DOMESTIC_LEAGUES, "MSI", "EWC"],
                "exact_series_id_required": True,
                "exact_fixture_and_map_id_required": True,
                "exact_blue_red_side_mapping_required": True,
                "exact_five_role_terminal_assignments_required": True,
                "protocol_specific_pick_ban_validation_required": True,
                "pre_event_patch_revision_receipt_required": True,
                "pre_event_rating_prediction_receipt_required": True,
                "terminal_draft_payload_bytes_and_hash_required": True,
                "terminal_draft_source_rights_review_required": True,
                "source_available_after_terminal_draft_and_before_actual_map_start": True,
                "actual_map_start_timestamp_authority_required": True,
                "draft_prediction_system_clock_strictly_before_actual_map_start": True,
                "outcome_fields_forbidden_from_prediction_receipt": True,
                "retrospective_backfill_qualifies": False,
            },
            "metadata_only_support_stopping_rule": {
                "eligible_maps_minimum": 1000,
                "eligible_series_minimum": 250,
                "each_domestic_league_maps_minimum": 150,
                "domestic_leagues": list(DOMESTIC_LEAGUES),
                "international_maps_minimum": 100,
                "distinct_future_patches_minimum": 3,
                "latest_future_patch_maps_minimum": 200,
                "sparse_or_new_champion_maps_minimum": 50,
                "stop_at_first_independently_pinned_ledger_meeting_metadata_thresholds": True,
                "outcomes_must_remain_unopened_while_checking_support": True,
            },
        },
        "evaluation": {
            "unit": "map_with_series_clustered_uncertainty",
            "primary_metrics": ["log_loss", "brier_score"],
            "secondary_metrics": [
                "ece_equal_frequency_10_bins",
                "calibration_slope",
                "calibration_intercept",
            ],
            "comparison": "frozen_combined_candidate_minus_frozen_rating_only_comparator",
            "uncertainty": {
                "method": "series_cluster_bootstrap",
                "confidence_level": 0.95,
                "resamples_minimum": 10000,
                "seed_fixed_before_opening": True,
            },
            "primary_pass_rule": {
                "log_loss_delta_upper_95_bound_maximum": 0.0,
                "brier_delta_upper_95_bound_maximum": 0.0,
                "at_least_one_primary_delta_upper_95_bound_strictly_below_zero": True,
                "both_point_deltas_nonpositive": True,
            },
            "reliability_gates": {
                "ece_delta_upper_95_bound_maximum": 0.01,
                "calibration_slope_interval_must_include_one": True,
                "calibration_intercept_interval_must_include_zero": True,
                "each_supported_domestic_league_point_deltas_nonpositive": True,
                "future_patch_point_deltas_nonpositive": True,
                "international_point_deltas_nonpositive": True,
                "python_typescript_replay_parity_required": True,
            },
            "failure_action": "candidate_unavailable_no_probability_no_recommendation_no_bet",
            "no_post_opening_tuning": True,
            "no_candidate_substitution": True,
        },
        "opening_authority": {
            "independent_protocol_review_present": False,
            "independent_ledger_pin_present": False,
            "independent_opening_approval_present": False,
            "approval_must_pin_protocol_registry_model_ratings_protocol_and_ledger_hashes": True,
            "self_authorizing": False,
        },
        "known_adaptive_failure": {
            "locator": ADAPTIVE_DIAGNOSTIC.as_posix(),
            "raw_sha256": hashlib.sha256(adaptive_raw).hexdigest(),
            "artifact_sha256": adaptive["artifact_sha256"],
            "result_state": adaptive["result_state"],
            "applies_to_locked_v3_candidate": False,
            "old_app_family_rehabilitated": False,
        },
        "capture_state": {
            "implementation_present": False,
            "prediction_ledger_present": False,
            "eligible_entries": 0,
            "outcomes_present": False,
            "outcomes_accessed": False,
        },
        "source_locks": [_source_record(root, locator) for locator in SOURCE_LOCKS],
        "decision_outputs": {
            "sealed_evaluation": None,
            "incremental_draft_authority": None,
            "neutral_probability": None,
            "contextual_probability": None,
            "fair_odds": None,
            "expected_value": None,
            "bet_recommendation": None,
        },
        "authority": {
            "model_validation_authority": False,
            "incremental_draft_authority": False,
            "neutral_probability_authority": False,
            "contextual_probability_authority": False,
            "odds_authority": False,
            "expected_value_authority": False,
            "recommendation_authority": False,
            "betting_authority": False,
        },
        "claim_ceiling": (
            "This lock freezes an empty prospective incremental-draft protocol. "
            "It grants no probability, odds, expected-value, recommendation, or betting authority."
        ),
    }
    payload["artifact_sha256"] = sha256_canonical_object(payload)
    return validate_future_protocol_v1(payload, root=root)


def validate_future_protocol_v1(
    payload: Mapping[str, Any], *, root: Path = Path(".")
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise DraftFutureProtocolError("future Draft Score protocol must be an object")
    value = dict(payload)
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("result_state") != RESULT_STATE
    ):
        raise DraftFutureProtocolError("future Draft Score protocol identity changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != sha256_canonical_object(unsigned):
        raise DraftFutureProtocolError("future Draft Score protocol hash changed")
    try:
        locked = datetime.fromisoformat(str(value.get("locked_at_utc"))).astimezone(
            timezone.utc
        )
        registry_locked = datetime.fromisoformat(
            str((value.get("candidate_registry") or {}).get("locked_at_utc"))
        ).astimezone(timezone.utc)
    except ValueError as exc:
        raise DraftFutureProtocolError("future Draft Score protocol time is invalid") from exc
    boundary = FUTURE_SEALED_START.replace(tzinfo=timezone.utc)
    if not registry_locked < locked < boundary:
        raise DraftFutureProtocolError("future protocol clock order changed")
    registry_record = value.get("candidate_registry") or {}
    registry_path = root / str(registry_record.get("locator"))
    registry_raw = registry_path.read_bytes()
    if hashlib.sha256(registry_raw).hexdigest() != registry_record.get("raw_sha256"):
        raise DraftFutureProtocolError("future protocol candidate registry bytes drifted")
    registry = validate_candidate_registry_v3(json.loads(registry_raw), root=root)
    if registry.get("artifact_sha256") != registry_record.get("artifact_sha256"):
        raise DraftFutureProtocolError("future protocol candidate registry hash drifted")
    if (
        (value.get("ratings_context") or {}).get("protocol_raw_sha256")
        != RATINGS_PROTOCOL_RAW_SHA256
        or (value.get("ratings_context") or {}).get("protocol_artifact_sha256")
        != RATINGS_PROTOCOL_ARTIFACT_SHA256
    ):
        raise DraftFutureProtocolError("future protocol ratings binding changed")
    validate_registered_future_protocol_v3(root=root)
    for record in value.get("source_locks") or []:
        if not isinstance(record, Mapping):
            raise DraftFutureProtocolError("future protocol source lock is malformed")
        locator = record.get("locator")
        if (
            not isinstance(locator, str)
            or _sha256(root / locator) != record.get("raw_sha256")
            or (root / locator).stat().st_size != record.get("bytes")
        ):
            raise DraftFutureProtocolError(f"future protocol source lock drifted: {locator}")
    if (value.get("future_holdout") or {}).get("status") != "EMPTY_NOT_YET_ACQUIRED":
        raise DraftFutureProtocolError("future protocol holdout is not empty")
    if (value.get("capture_state") or {}).get("outcomes_accessed") is not False:
        raise DraftFutureProtocolError("future protocol outcome state changed")
    if any((value.get("authority") or {}).values()):
        raise DraftFutureProtocolError("future protocol granted authority")
    if (value.get("estimands") or {}).get(
        "neutral_output_directly_outcome_calibrated"
    ) is not False:
        raise DraftFutureProtocolError("future protocol broadened neutral semantics")
    return value


def write_future_protocol_v1(root: Path) -> Path:
    payload = build_future_protocol_v1(root=root)
    path = root / DEFAULT_OUTPUT
    _atomic_write_new(
        path,
        (json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode(
            "ascii"
        ),
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    try:
        path = write_future_protocol_v1(args.root)
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, DraftFutureProtocolError) as exc:
        print(str(exc))
        return 2
    print(
        json.dumps(
            {
                "protocol": str(path),
                "raw_sha256": _sha256(path),
                "artifact_sha256": payload["artifact_sha256"],
                "locked_at_utc": payload["locked_at_utc"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

