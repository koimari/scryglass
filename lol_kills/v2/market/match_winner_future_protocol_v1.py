"""Lock the two-stage prospective map-winner market evaluation.

Stage one is the already-frozen ratings and terminal-draft holdout beginning
2026-08-03.  Its outcomes must stay sealed until the registered metadata-only
support rules are met.  Only if both model evaluations pass may their frozen
predictions be used to fit the predeclared recalibration and uncertainty
procedure.  Stage two then starts on a disjoint, still-future cohort and
compares the resulting probabilities with contemporaneous Betano Brazil
map-winner quotes.

This protocol is deliberately non-authorizing.  It freezes what would have to
be demonstrated; it does not create probability, expected-value, staking, or
betting authority and it cannot be used to rehabilitate retrospective data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

from lol_kills.bookmaker_quote_capture import (
    EXTRACTION_SCHEMA_VERSION,
    RECEIPT_SCHEMA_VERSION,
)
from lol_kills.v2.data.common import sha256_canonical_object
from lol_kills.v2.draft.terminal.capture_readiness_registry_v1 import (
    REGISTERED_CAPTURE_ARTIFACT_SHA256 as DRAFT_CAPTURE_ARTIFACT_SHA256,
    REGISTERED_CAPTURE_LOCATOR as DRAFT_CAPTURE_LOCATOR,
    REGISTERED_CAPTURE_RAW_SHA256 as DRAFT_CAPTURE_RAW_SHA256,
    validate_registered_capture_readiness_v1 as validate_draft_capture,
)
from lol_kills.v2.draft.terminal.future_protocol_registry_v1 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256 as DRAFT_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_LOCATOR as DRAFT_PROTOCOL_LOCATOR,
    REGISTERED_PROTOCOL_RAW_SHA256 as DRAFT_PROTOCOL_RAW_SHA256,
    validate_registered_future_protocol_v1 as validate_draft_protocol,
)
from lol_kills.v2.draft.terminal.grid_source_readiness_registry_v1 import (
    REGISTERED_GRID_SOURCE_ARTIFACT_SHA256,
    REGISTERED_GRID_SOURCE_LOCATOR,
    REGISTERED_GRID_SOURCE_RAW_SHA256,
    validate_registered_grid_source_readiness_v1,
)
from lol_kills.v2.ratings.player.multileague_v3_capture_registry_v3 import (
    REGISTERED_CAPTURE_ARTIFACT_SHA256 as RATINGS_CAPTURE_ARTIFACT_SHA256,
    REGISTERED_CAPTURE_LOCATOR as RATINGS_CAPTURE_LOCATOR,
    REGISTERED_CAPTURE_RAW_SHA256 as RATINGS_CAPTURE_RAW_SHA256,
    validate_registered_capture_readiness_v3 as validate_ratings_capture,
)
from lol_kills.v2.ratings.player.multileague_v3_future_protocol import (
    DOMESTIC_LEAGUES,
    FUTURE_SEALED_START,
)
from lol_kills.v2.ratings.player.multileague_v3_registry_v3 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256 as RATINGS_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_LOCATOR as RATINGS_PROTOCOL_LOCATOR,
    REGISTERED_PROTOCOL_RAW_SHA256 as RATINGS_PROTOCOL_RAW_SHA256,
    validate_registered_future_protocol_v3 as validate_ratings_protocol,
)


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "scryglass:private-map-winner-future-market-protocol:v1"
RESULT_STATE = "TWO_STAGE_PROSPECTIVE_MARKET_PROTOCOL_LOCKED_EMPTY"
SOURCE_LOCATOR = "lol_kills/v2/market/match_winner_future_protocol_v1.py"
DEFAULT_OUTPUT = Path(
    "data/lol/v2/evaluation/match-winner-market-v1/future-protocol-v1.json"
)
BOOKMAKER_ID = "betano-brazil"
MARKET_TYPE = "match_winner"
SETTLEMENT_RULE_ID = "betano-br-map-winner-shadow-v1"
PHASE_ONE_BOUNDARY = FUTURE_SEALED_START.replace(tzinfo=timezone.utc)
CLAIM_CEILING = (
    "This is an empty two-stage prospective evaluation lock. It grants no "
    "rating, Draft Score, probability, odds, expected-value, recommendation, "
    "staking, transaction, or betting authority."
)
AUTHORITY_KEYS = (
    "ratings_validation_authority",
    "draft_validation_authority",
    "calibration_authority",
    "probability_authority",
    "odds_authority",
    "expected_value_authority",
    "recommendation_authority",
    "staking_authority",
    "betting_authority",
)
SOURCE_LOCKS = (
    SOURCE_LOCATOR,
    "lol_kills/bookmaker_quote_capture.py",
    "lol_kills/v2/ratings/player/multileague_v3_registry_v3.py",
    "lol_kills/v2/ratings/player/multileague_v3_capture_registry_v3.py",
    "lol_kills/v2/draft/terminal/future_protocol_registry_v1.py",
    "lol_kills/v2/draft/terminal/capture_readiness_registry_v1.py",
    "lol_kills/v2/draft/terminal/grid_source_readiness_registry_v1.py",
    RATINGS_PROTOCOL_LOCATOR.as_posix(),
    RATINGS_CAPTURE_LOCATOR.as_posix(),
    DRAFT_PROTOCOL_LOCATOR.as_posix(),
    DRAFT_CAPTURE_LOCATOR.as_posix(),
    REGISTERED_GRID_SOURCE_LOCATOR.as_posix(),
)


class MatchWinnerFutureProtocolError(RuntimeError):
    """The prospective map-winner protocol is malformed or drifted."""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(root: Path, locator: str) -> dict[str, Any]:
    path = root / locator
    if not path.is_file():
        raise MatchWinnerFutureProtocolError(f"bound source unavailable: {locator}")
    return {
        "locator": locator,
        "bytes": path.stat().st_size,
        "raw_sha256": _sha256_path(path),
    }


def _time(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise MatchWinnerFutureProtocolError(f"{label} must be RFC-3339")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MatchWinnerFutureProtocolError(f"{label} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise MatchWinnerFutureProtocolError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _clock_sample(clock: Callable[[], datetime], prerequisite_time: datetime) -> datetime:
    observed = clock()
    if not isinstance(observed, datetime) or observed.tzinfo is None:
        raise MatchWinnerFutureProtocolError(
            "protocol clock must return a timezone-aware datetime"
        )
    observed = observed.astimezone(timezone.utc)
    if observed <= prerequisite_time:
        raise MatchWinnerFutureProtocolError(
            "market protocol lock must follow all prerequisite locks"
        )
    if observed >= PHASE_ONE_BOUNDARY:
        raise MatchWinnerFutureProtocolError(
            "market protocol lock must precede the phase-one future boundary"
        )
    return observed


def _artifact_binding(
    *, locator: Path, raw_sha256: str, artifact_sha256: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    if payload.get("artifact_sha256") != artifact_sha256:
        raise MatchWinnerFutureProtocolError(
            f"registered prerequisite artifact changed: {locator}"
        )
    return {
        "locator": locator.as_posix(),
        "raw_sha256": raw_sha256,
        "artifact_sha256": artifact_sha256,
        "locked_at_utc": payload["locked_at_utc"],
    }


def _prerequisite_bindings(root: Path) -> dict[str, dict[str, Any]]:
    ratings_protocol = validate_ratings_protocol(root=root)
    ratings_capture = validate_ratings_capture(root=root)
    draft_protocol = validate_draft_protocol(root=root)
    draft_capture = validate_draft_capture(root=root)
    grid_source = validate_registered_grid_source_readiness_v1(root=root)
    return {
        "ratings_future_protocol": _artifact_binding(
            locator=RATINGS_PROTOCOL_LOCATOR,
            raw_sha256=RATINGS_PROTOCOL_RAW_SHA256,
            artifact_sha256=RATINGS_PROTOCOL_ARTIFACT_SHA256,
            payload=ratings_protocol,
        ),
        "ratings_capture_readiness": _artifact_binding(
            locator=RATINGS_CAPTURE_LOCATOR,
            raw_sha256=RATINGS_CAPTURE_RAW_SHA256,
            artifact_sha256=RATINGS_CAPTURE_ARTIFACT_SHA256,
            payload=ratings_capture,
        ),
        "draft_future_protocol": _artifact_binding(
            locator=DRAFT_PROTOCOL_LOCATOR,
            raw_sha256=DRAFT_PROTOCOL_RAW_SHA256,
            artifact_sha256=DRAFT_PROTOCOL_ARTIFACT_SHA256,
            payload=draft_protocol,
        ),
        "draft_capture_readiness": _artifact_binding(
            locator=DRAFT_CAPTURE_LOCATOR,
            raw_sha256=DRAFT_CAPTURE_RAW_SHA256,
            artifact_sha256=DRAFT_CAPTURE_ARTIFACT_SHA256,
            payload=draft_capture,
        ),
        "grid_terminal_draft_source_readiness": _artifact_binding(
            locator=REGISTERED_GRID_SOURCE_LOCATOR,
            raw_sha256=REGISTERED_GRID_SOURCE_RAW_SHA256,
            artifact_sha256=REGISTERED_GRID_SOURCE_ARTIFACT_SHA256,
            payload=grid_source,
        ),
    }


def _settlement_contract() -> dict[str, Any]:
    return {
        "settlement_rule_id": SETTLEMENT_RULE_ID,
        "scope": "single_map_league_of_legends_winner_two_way_cash_odds",
        "winning_selection": (
            "winner:<canonical_team_id> whose team is the official final map winner"
        ),
        "official_result_receipt_required": True,
        "bookmaker_specific_terms_snapshot_required_before_phase_two": True,
        "bookmaker_terms_snapshot_status": "NOT_YET_CAPTURED_OR_REVIEWED",
        "exact_market_label_and_map_number_must_match": True,
        "team_aliases_must_resolve_before_quote_capture": True,
        "non_started_map": "void",
        "bookmaker_void_or_refund": "void_zero_profit_zero_loss",
        "same_day_resumption": "follow_exact_registered_bookmaker_terms",
        "postponement_cancellation_remake_forfeit": (
            "unavailable_unless_exact_registered_bookmaker_terms_and_settlement_record_resolve_it"
        ),
        "ambiguous_or_conflicting_result": "unavailable_not_manually_overridden",
        "cash_out_free_bet_boost_builder_or_promotion_qualifies": False,
        "manual_post_outcome_exclusion_permitted": False,
        "bookmaker_terms_alignment_independent_review_required": True,
    }


def _quote_capture_contract(settlement_sha256: str) -> dict[str, Any]:
    return {
        "bookmaker_id": BOOKMAKER_ID,
        "market_type": MARKET_TYPE,
        "settlement_rule_id": SETTLEMENT_RULE_ID,
        "quote_receipt_schema_version": RECEIPT_SCHEMA_VERSION,
        "price_extraction_schema_version": EXTRACTION_SCHEMA_VERSION,
        "settlement_contract_sha256": settlement_sha256,
        "source_specific_adapter_frozen_before_first_phase_two_quote": True,
        "adapter_id_and_source_sha256_required": True,
        "exact_response_body_bytes_and_sha256_required": True,
        "response_body_must_exclude_request_headers_cookies_and_credentials": True,
        "deterministic_extraction_replay_required": True,
        "transport_request_started_system_utc_required": True,
        "transport_response_received_system_utc_required": True,
        "monotonic_transport_duration_required": True,
        "generic_receipt_builder_time_counts_as_transport_time": False,
        "user_supplied_capture_timestamp_allowed": False,
        "market_status_must_be_open": True,
        "exact_two_team_prices_required": True,
        "prediction_receipt_must_precede_quote_request": True,
        "prediction_to_quote_response_seconds_maximum": 30.0,
        "quote_response_must_precede_actual_map_start": True,
        "quote_response_to_actual_map_start_seconds_minimum": 5.0,
        "retrospective_backfill_qualifies": False,
        "quote_presence_does_not_prove_acceptance_limit_or_execution": True,
        "quote_capture_itself_is_betting_authority": False,
    }


def _phase_one_contract() -> dict[str, Any]:
    return {
        "status": "EMPTY_OUTCOMES_SEALED",
        "start_inclusive_source_time": PHASE_ONE_BOUNDARY.isoformat(),
        "purpose": "validate_frozen_ratings_and_incremental_terminal_draft",
        "ratings_protocol_and_capture_must_pass_their_registered_rules": True,
        "draft_protocol_and_capture_must_pass_their_registered_rules": True,
        "same_event_predictions_must_bind_exact_rating_and_draft_receipts": True,
        "both_independent_opening_reviews_required": True,
        "either_model_failure_action": "phase_two_unavailable_no_substitution",
        "model_or_hyperparameter_reselection_after_opening_permitted": False,
        "phase_one_outcomes_may_be_used_for_phase_two_recalibration_only_after_both_pass": True,
        "phase_one_itself_is_market_or_betting_authority": False,
    }


def _recalibration_contract() -> dict[str, Any]:
    return {
        "status": "NOT_YET_FIT",
        "fit_only_after_phase_one_passes": True,
        "fit_dataset": "complete_independently_opened_phase_one_draft_cohort",
        "raw_probability": "frozen_rating_plus_terminal_draft_contextual_probability",
        "method": "bounded_logistic_recalibration",
        "formula": "sigmoid(intercept+slope*logit(clipped_raw_probability))",
        "raw_probability_clip": [1e-6, 0.999999],
        "objective": "unweighted_map_log_loss",
        "intercept_bounds": [-2.0, 2.0],
        "slope_bounds": [0.25, 4.0],
        "initial_parameters": {"intercept": 0.0, "slope": 1.0},
        "optimizer": "scipy.optimize.minimize:L-BFGS-B",
        "optimizer_ftol": 1e-12,
        "optimizer_gtol": 1e-8,
        "maximum_iterations": 10000,
        "rating_only_comparator_recalibrated_by_identical_procedure": True,
        "parameters_and_implementation_sha256_frozen_before_phase_two": True,
        "phase_two_refit_or_online_update_permitted": False,
    }


def _uncertainty_contract() -> dict[str, Any]:
    return {
        "status": "IMPLEMENTATION_NOT_YET_FROZEN",
        "method": "series_cluster_bootstrap_full_prediction_pipeline",
        "confidence_level": 0.95,
        "resamples_minimum": 2000,
        "resampling_units": {
            "development_fit": "series",
            "phase_one_recalibration": "series",
        },
        "candidate_and_hyperparameters_remain_fixed": True,
        "ratings_state_refit_in_each_resample": True,
        "draft_terms_refit_in_each_resample": True,
        "phase_one_recalibration_refit_in_each_resample": True,
        "event_outcome_or_market_price_used_in_interval": False,
        "percentile_interval": [0.025, 0.975],
        "random_seed_and_implementation_sha256_frozen_before_phase_two": True,
        "failure_or_nonconvergence_action": "event_probability_unavailable",
        "interval_is_epistemic_not_a_guarantee_of_binary_outcome_coverage": True,
    }


def _phase_two_contract() -> dict[str, Any]:
    return {
        "status": "NOT_OPEN_NOT_STARTED",
        "purpose": "fresh_market_comparison_and_shadow_policy_evaluation",
        "start_boundary": (
            "strictly_after_independent_phase_one_pass_recalibration_and_uncertainty_registry_lock"
        ),
        "cohort_disjoint_from_phase_one": True,
        "first_quote_source_time_strictly_after_phase_two_lock": True,
        "outcomes_sealed_until_metadata_only_stopping_rule": True,
        "eligibility": {
            "professional_maps_only": True,
            "leagues": [*DOMESTIC_LEAGUES, "MSI", "EWC"],
            "exact_fixture_series_map_patch_sides_and_ten_players_required": True,
            "registered_pre_event_roster_receipt_required": True,
            "registered_pre_event_rating_prediction_required": True,
            "registered_terminal_draft_prediction_and_map_start_required": True,
            "registered_event_probability_and_interval_required": True,
            "registered_betano_quote_and_transport_receipt_required": True,
            "registered_bookmaker_terms_and_settlement_contract_required": True,
            "market_open_and_two_way_cash_prices_required": True,
            "prediction_model_must_not_receive_quote_or_market_features": True,
            "event_outcome_fields_forbidden_from_all_pre_event_receipts": True,
            "retrospective_backfill_qualifies": False,
        },
        "metadata_only_stopping_rule": {
            "eligible_quoted_maps_minimum": 500,
            "eligible_series_minimum": 125,
            "each_domestic_league_quoted_maps_minimum": 75,
            "domestic_leagues": list(DOMESTIC_LEAGUES),
            "international_quoted_maps_minimum": 50,
            "distinct_future_patches_minimum": 3,
            "latest_patch_quoted_maps_minimum": 100,
            "one_or_both_rosters_changed_maps_minimum": 50,
            "sparse_or_new_player_or_champion_maps_minimum": 50,
            "quote_coverage_of_otherwise_eligible_maps_minimum": 0.80,
            "shadow_policy_qualifying_maps_minimum": 100,
            "eligible_quoted_maps_maximum_if_shadow_support_not_met": 1000,
            "stop_at_first_independently_pinned_ledger_meeting_all_minima": True,
            "outcomes_must_remain_unopened_while_checking_support": True,
            "thresholds_are_metadata_floors_not_a_post_hoc_power_claim": True,
        },
    }


def _evaluation_contract() -> dict[str, Any]:
    return {
        "unit": "map_with_series_clustered_uncertainty",
        "market_benchmark": "two_way_normalized_implied_probability_no_vig",
        "primary_comparisons": [
            "recalibrated_combined_model_minus_no_vig_market",
            "recalibrated_combined_model_minus_recalibrated_rating_only",
        ],
        "primary_metrics": ["log_loss", "brier_score"],
        "uncertainty": {
            "method": "paired_series_cluster_bootstrap",
            "confidence_level": 0.95,
            "resamples_minimum": 10000,
            "seed_frozen_before_opening": True,
        },
        "probabilistic_pass_rule": {
            "each_primary_point_delta_maximum": 0.0,
            "each_primary_delta_upper_95_bound_maximum": 0.0,
            "at_least_one_market_delta_upper_95_bound_strictly_below_zero": True,
            "all_comparisons_and_metrics_must_pass": True,
        },
        "calibration_gates": {
            "ece_equal_frequency_10_bins_maximum": 0.03,
            "ece_series_cluster_upper_95_bound_maximum": 0.05,
            "calibration_slope_interval_must_include_one": True,
            "calibration_intercept_interval_must_include_zero": True,
            "league_patch_roster_change_and_sparse_strata_reported": True,
            "no_supported_stratum_log_loss_point_delta_vs_market_above": 0.02,
        },
        "capture_gates": {
            "quote_coverage_minimum": 0.80,
            "prediction_to_quote_response_p95_seconds_maximum": 30.0,
            "quote_after_map_start_count_maximum": 0,
            "extractor_replay_mismatch_count_maximum": 0,
            "team_or_map_binding_mismatch_count_maximum": 0,
        },
        "shadow_policy": {
            "one_unit_flat_stake_for_evaluation_only": True,
            "selection_rule": (
                "choose_the_only_side_if_lower_95_probability_bound_times_offered_decimal_odds_minus_one_is_at_least_0.02"
            ),
            "if_both_sides_qualify": "event_unavailable_as_inconsistent",
            "execution_haircut_fraction_of_positive_profit": 0.01,
            "minimum_qualifying_maps": 100,
            "cluster_bootstrap_roi_lower_95_bound_strictly_above_zero": True,
            "point_roi_after_haircut_strictly_above_zero": True,
            "maximum_drawdown_and_longest_losing_run_reported": True,
            "quoted_shadow_return_is_not_proof_of_executable_return": True,
            "stake_or_bankroll_size_authorized": False,
        },
        "failure_action": (
            "no_probability_no_fair_odds_no_expected_value_no_recommendation_no_betting"
        ),
        "no_post_opening_tuning": True,
        "all_predeclared_gates_required": True,
    }


def build_match_winner_future_protocol_v1(
    *,
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    prerequisites = _prerequisite_bindings(root)
    prerequisite_time = max(
        _time(item["locked_at_utc"], f"{name}.locked_at_utc")
        for name, item in prerequisites.items()
    )
    observed = _clock_sample(clock, prerequisite_time)
    settlement = _settlement_contract()
    settlement_sha256 = sha256_canonical_object(settlement)
    quote_capture = _quote_capture_contract(settlement_sha256)
    quote_capture_sha256 = sha256_canonical_object(quote_capture)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "locked_at_utc": observed.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_inside_builder",
            "observed_wall_clock_utc": observed.isoformat(),
            "user_supplied_timestamp_allowed": False,
            "lock_time_after_all_prerequisites": True,
            "lock_time_before_phase_one_boundary": True,
        },
        "scope": {
            "visibility": "private_personal_research_only",
            "public_scryglass_remains_non_betting": True,
            "bookmaker_id": BOOKMAKER_ID,
            "market_type": MARKET_TYPE,
            "market_unit": "single_map",
            "currency_or_stake_assumption": None,
            "transaction_automation": False,
        },
        "prerequisites": prerequisites,
        "phase_one": _phase_one_contract(),
        "recalibration": _recalibration_contract(),
        "event_uncertainty": _uncertainty_contract(),
        "settlement_contract": settlement,
        "settlement_contract_sha256": settlement_sha256,
        "quote_capture_contract": quote_capture,
        "quote_capture_contract_sha256": quote_capture_sha256,
        "phase_two": _phase_two_contract(),
        "evaluation": _evaluation_contract(),
        "registries": {
            "phase_one_evaluation_authority": None,
            "recalibration_and_uncertainty": None,
            "bookmaker_terms": None,
            "source_specific_quote_adapter": None,
            "phase_two_prediction_ledger": None,
            "phase_two_quote_ledger": None,
            "phase_two_outcome_opening_authority": None,
            "market_authority": None,
        },
        "source_locks": [_source_record(root, locator) for locator in SOURCE_LOCKS],
        "decision_outputs": {
            "model_probability": None,
            "probability_interval": None,
            "fair_odds": None,
            "expected_value": None,
            "edge": None,
            "bet_recommendation": None,
            "stake": None,
        },
        "authority": {name: False for name in AUTHORITY_KEYS},
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = sha256_canonical_object(payload)
    return validate_match_winner_future_protocol_v1(payload, root=root)


def validate_match_winner_future_protocol_v1(
    payload: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise MatchWinnerFutureProtocolError("market protocol must be an object")
    value = dict(payload)
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("result_state") != RESULT_STATE
    ):
        raise MatchWinnerFutureProtocolError("market protocol identity changed")
    unsigned = dict(value)
    declared = unsigned.pop("artifact_sha256", None)
    if declared != sha256_canonical_object(unsigned):
        raise MatchWinnerFutureProtocolError("market protocol canonical hash mismatch")

    prerequisites = _prerequisite_bindings(root)
    if value.get("prerequisites") != prerequisites:
        raise MatchWinnerFutureProtocolError("market protocol prerequisite binding changed")
    prerequisite_time = max(
        _time(item["locked_at_utc"], f"{name}.locked_at_utc")
        for name, item in prerequisites.items()
    )
    locked = _time(value.get("locked_at_utc"), "locked_at_utc")
    if not prerequisite_time < locked < PHASE_ONE_BOUNDARY:
        raise MatchWinnerFutureProtocolError("market protocol clock order changed")
    clock = value.get("clock_attestation")
    if clock != {
        "source": "system_utc_clock_sampled_inside_builder",
        "observed_wall_clock_utc": locked.isoformat(),
        "user_supplied_timestamp_allowed": False,
        "lock_time_after_all_prerequisites": True,
        "lock_time_before_phase_one_boundary": True,
    }:
        raise MatchWinnerFutureProtocolError("market protocol clock attestation changed")

    expected_scope = {
        "visibility": "private_personal_research_only",
        "public_scryglass_remains_non_betting": True,
        "bookmaker_id": BOOKMAKER_ID,
        "market_type": MARKET_TYPE,
        "market_unit": "single_map",
        "currency_or_stake_assumption": None,
        "transaction_automation": False,
    }
    if value.get("scope") != expected_scope:
        raise MatchWinnerFutureProtocolError("market protocol scope changed")
    expected_sections = {
        "phase_one": _phase_one_contract(),
        "recalibration": _recalibration_contract(),
        "event_uncertainty": _uncertainty_contract(),
        "phase_two": _phase_two_contract(),
        "evaluation": _evaluation_contract(),
    }
    for name, expected in expected_sections.items():
        if value.get(name) != expected:
            raise MatchWinnerFutureProtocolError(f"market protocol {name} changed")

    settlement = _settlement_contract()
    settlement_sha256 = sha256_canonical_object(settlement)
    if (
        value.get("settlement_contract") != settlement
        or value.get("settlement_contract_sha256") != settlement_sha256
    ):
        raise MatchWinnerFutureProtocolError("settlement contract changed")
    quote_capture = _quote_capture_contract(settlement_sha256)
    if (
        value.get("quote_capture_contract") != quote_capture
        or value.get("quote_capture_contract_sha256")
        != sha256_canonical_object(quote_capture)
    ):
        raise MatchWinnerFutureProtocolError("quote capture contract changed")

    if value.get("registries") != {
        "phase_one_evaluation_authority": None,
        "recalibration_and_uncertainty": None,
        "bookmaker_terms": None,
        "source_specific_quote_adapter": None,
        "phase_two_prediction_ledger": None,
        "phase_two_quote_ledger": None,
        "phase_two_outcome_opening_authority": None,
        "market_authority": None,
    }:
        raise MatchWinnerFutureProtocolError("empty registry state changed")
    if value.get("decision_outputs") != {
        "model_probability": None,
        "probability_interval": None,
        "fair_odds": None,
        "expected_value": None,
        "edge": None,
        "bet_recommendation": None,
        "stake": None,
    }:
        raise MatchWinnerFutureProtocolError("empty protocol contains decision output")
    authority = value.get("authority") or {}
    if set(authority) != set(AUTHORITY_KEYS) or any(
        authority.get(name) is not False for name in AUTHORITY_KEYS
    ):
        raise MatchWinnerFutureProtocolError("market protocol granted authority")
    if value.get("claim_ceiling") != CLAIM_CEILING:
        raise MatchWinnerFutureProtocolError("market protocol claim ceiling changed")

    records = value.get("source_locks")
    if not isinstance(records, list) or len(records) != len(SOURCE_LOCKS):
        raise MatchWinnerFutureProtocolError("market protocol source inventory changed")
    if [record.get("locator") for record in records if isinstance(record, Mapping)] != list(
        SOURCE_LOCKS
    ):
        raise MatchWinnerFutureProtocolError("market protocol source order changed")
    for record in records:
        if not isinstance(record, Mapping):
            raise MatchWinnerFutureProtocolError("market protocol source lock malformed")
        locator = record.get("locator")
        path = root / str(locator)
        if (
            not isinstance(locator, str)
            or not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or _sha256_path(path) != record.get("raw_sha256")
        ):
            raise MatchWinnerFutureProtocolError(
                f"market protocol source drifted: {locator}"
            )
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    digest = hashlib.sha256(raw).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to replace market protocol: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = args.out if args.out.is_absolute() else args.root / args.out
    try:
        payload = build_match_winner_future_protocol_v1(root=args.root)
        raw_sha256 = write_no_clobber(output, payload)
    except (OSError, ValueError, MatchWinnerFutureProtocolError) as exc:
        print(str(exc))
        return 2
    print(
        json.dumps(
            {
                "protocol": str(output),
                "raw_sha256": raw_sha256,
                "artifact_sha256": payload["artifact_sha256"],
                "locked_at_utc": payload["locked_at_utc"],
                "quote_capture_contract_sha256": payload[
                    "quote_capture_contract_sha256"
                ],
                "settlement_contract_sha256": payload[
                    "settlement_contract_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
